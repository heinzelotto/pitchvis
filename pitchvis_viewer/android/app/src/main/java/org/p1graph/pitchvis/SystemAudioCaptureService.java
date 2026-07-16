package org.p1graph.pitchvis;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioPlaybackCaptureConfiguration;
import android.media.AudioRecord;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.Process;
import android.util.Log;

import androidx.core.app.NotificationCompat;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Captures the system audio mix via MediaProjection and pushes it to Rust.
 *
 * AUDIO ONLY: no VirtualDisplay is ever created and the screen is never captured. MediaProjection
 * is simply the only API Android offers for tapping the playback mix.
 *
 * Requires API 29+ (AudioPlaybackCapture); the caller gates on that.
 */
public class SystemAudioCaptureService extends Service {

    private static final String TAG = "PitchVis";
    private static final String CHANNEL_ID = "pitchvis_system_audio";
    private static final int NOTIF_ID = 1;

    public static final String EXTRA_RESULT_CODE = "resultCode";
    public static final String EXTRA_RESULT_DATA = "resultData";

    private MediaProjection projection;
    private AudioRecord record;
    private Thread reader;
    private volatile boolean running;
    /** Set once we have told Rust that capture started, so onDestroy knows to tell it we stopped. */
    private volatile boolean notifiedStarted;

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        // A null intent means Android restarted us after the process died. We deliberately use
        // START_NOT_STICKY (see the return below) so this should not happen, but if it ever does,
        // the projection token in the original intent is long consumed -- bail out rather than
        // throw or strand a notification.
        if (intent == null) {
            Log.w(TAG, "SystemAudioCapture: restarted with null intent, stopping");
            stopSelf();
            return START_NOT_STICKY;
        }

        // 1. FOREGROUND FIRST. On API 34+, getMediaProjection() throws SecurityException unless a
        //    mediaProjection-typed foreground service is already running. The order here is not a
        //    style choice.
        try {
            startForegroundNotification();
        } catch (Exception e) {
            fail("startForeground failed: " + e);
            return START_NOT_STICKY;
        }

        // 2. Now it is legal to claim the projection.
        int resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, 0);
        Intent resultData = intent.getParcelableExtra(EXTRA_RESULT_DATA);
        if (resultData == null) {
            fail("missing projection result data");
            return START_NOT_STICKY;
        }

        MediaProjectionManager mpm =
                (MediaProjectionManager) getSystemService(Context.MEDIA_PROJECTION_SERVICE);
        try {
            projection = mpm.getMediaProjection(resultCode, resultData);
        } catch (Exception e) {
            fail("getMediaProjection failed: " + e);
            return START_NOT_STICKY;
        }
        if (projection == null) {
            fail("getMediaProjection returned null");
            return START_NOT_STICKY;
        }

        // 3. Mandatory on API 34+. onStop fires when the user taps the system "Stop sharing" chip,
        //    which is why nativeSystemAudioStopped must be the single convergence point for stops.
        projection.registerCallback(new MediaProjection.Callback() {
            @Override
            public void onStop() {
                Log.w(TAG, "SystemAudioCapture: projection stopped by system/user");
                stopSelf();
            }
        }, new Handler(Looper.getMainLooper()));

        // 4. Build the recorder at the device's NATIVE rate in stereo float, and let Rust do the
        //    downmix and the resample to 22050. We deliberately do NOT ask AudioRecord for 22050:
        //    AudioFlinger's capture resampler is built to upsample mic input, and its anti-alias
        //    filtering when decimating 48k by 2.18x is weak and undocumented. Aliased content
        //    folding into the 55-7040 Hz analysis band would paint phantom pitches on the spiral.
        AudioPlaybackCaptureConfiguration config;
        try {
            config = new AudioPlaybackCaptureConfiguration.Builder(projection)
                    .addMatchingUsage(AudioAttributes.USAGE_MEDIA)
                    .addMatchingUsage(AudioAttributes.USAGE_GAME)
                    .addMatchingUsage(AudioAttributes.USAGE_UNKNOWN)
                    // Defensive: PitchVis makes no sound today, but this prevents a feedback loop
                    // if it ever does. Mixing usage-matching with uid-excluding is legal; only
                    // mixing match+exclude of the SAME type is forbidden.
                    .excludeUid(Process.myUid())
                    .build();
        } catch (Exception e) {
            fail("capture config failed: " + e);
            return START_NOT_STICKY;
        }

        int rate = nativeOutputSampleRate();
        int minBuf = AudioRecord.getMinBufferSize(
                rate, AudioFormat.CHANNEL_IN_STEREO, AudioFormat.ENCODING_PCM_FLOAT);
        if (minBuf <= 0) {
            fail("getMinBufferSize returned " + minBuf);
            return START_NOT_STICKY;
        }
        // At least 100 ms of stereo float, so a scheduling hiccup does not drop audio.
        int bufBytes = Math.max(minBuf, rate * 2 * 4 / 10);

        try {
            record = new AudioRecord.Builder()
                    .setAudioFormat(new AudioFormat.Builder()
                            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                            .setSampleRate(rate)
                            .setChannelMask(AudioFormat.CHANNEL_IN_STEREO)
                            .build())
                    .setBufferSizeInBytes(bufBytes)
                    .setAudioPlaybackCaptureConfig(config)
                    .build();
        } catch (Exception e) {
            fail("AudioRecord.build failed: " + e);
            return START_NOT_STICKY;
        }

        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            fail("AudioRecord not initialized");
            return START_NOT_STICKY;
        }

        try {
            record.startRecording();
        } catch (Exception e) {
            fail("startRecording failed: " + e);
            return START_NOT_STICKY;
        }

        final int actualRate = record.getSampleRate();
        final int actualChannels = record.getFormat().getChannelCount();
        Log.w(TAG, "SystemAudioCapture: recording at " + actualRate + " Hz, "
                + actualChannels + " ch (requested " + rate + " Hz stereo float)");

        running = true;
        reader = new Thread(() -> readLoop(actualRate, actualChannels), "pitchvis-syscap");
        reader.start();

        // START_NOT_STICKY is mandatory, not a preference. The app calls std::process::exit(0) on
        // suspend; with START_STICKY, Android would restart this service after that process death
        // and retry getMediaProjection with a single-use, already-consumed token -- yielding a
        // SecurityException or a zombie "casting" notification the user cannot dismiss.
        return START_NOT_STICKY;
    }

    /** The device's native output rate -- what the playback mix actually runs at. */
    private int nativeOutputSampleRate() {
        try {
            AudioManager am = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
            String v = am.getProperty(AudioManager.PROPERTY_OUTPUT_SAMPLE_RATE);
            if (v != null) {
                int rate = Integer.parseInt(v);
                if (rate >= 8000 && rate <= 192000) {
                    return rate;
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "SystemAudioCapture: could not read native output rate", e);
        }
        return 48000;
    }

    private void readLoop(int rate, int channels) {
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO);

        int framesPerRead = Math.max(1, rate / 50); // ~20 ms
        int bytes = framesPerRead * channels * 4;
        ByteBuffer buf = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder());

        // Tell Rust we are live. This is where the mic gets torn down, so it must happen only
        // once capture is genuinely running.
        notifiedStarted = true;
        NativeBridge.nativeSystemAudioStarted(rate, channels);

        while (running) {
            buf.clear();
            int read = record.read(buf, bytes, AudioRecord.READ_BLOCKING);
            if (read <= 0) {
                if (read == AudioRecord.ERROR_INVALID_OPERATION
                        || read == AudioRecord.ERROR_DEAD_OBJECT) {
                    Log.e(TAG, "SystemAudioCapture: read error " + read + ", ending loop");
                    break;
                }
                continue;
            }
            NativeBridge.nativeSystemAudioData(buf, read / (channels * 4));
        }
        Log.w(TAG, "SystemAudioCapture: read loop ended");
    }

    private void startForegroundNotification() {
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                && nm.getNotificationChannel(CHANNEL_ID) == null) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID, "System audio capture", NotificationManager.IMPORTANCE_LOW);
            channel.setDescription("Shown while PitchVis is analyzing system audio");
            channel.setShowBadge(false);
            nm.createNotificationChannel(channel);
        }

        Notification notification = new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("PitchVis")
                .setContentText("Analyzing system audio")
                .setSmallIcon(R.mipmap.ic_launcher)
                .setOngoing(true)
                .setPriority(NotificationCompat.PRIORITY_LOW)
                .build();

        // Call the platform API directly rather than ServiceCompat: the typed 4-arg
        // ServiceCompat.startForeground overload only exists in androidx.core >= 1.12, which this
        // project does not pull in. The service is only ever launched on API 29+ (is_supported
        // gates on it), so the typed overload is always available at runtime.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIF_ID, notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
        } else {
            startForeground(NOTIF_ID, notification);
        }
    }

    /** Report a setup failure to Rust and shut down. */
    private void fail(String reason) {
        Log.e(TAG, "SystemAudioCapture: " + reason);
        NativeBridge.nativeSystemAudioStopped(reason);
        stopSelf();
    }

    @Override
    public void onDestroy() {
        running = false;
        if (reader != null) {
            try {
                reader.join(500);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
            reader = null;
        }
        if (record != null) {
            try {
                record.stop();
            } catch (Exception ignored) {
                // already stopped or never started
            }
            record.release();
            record = null;
        }
        if (projection != null) {
            projection.stop();
            projection = null;
        }
        // Only if we previously said we started -- otherwise fail() already reported the reason
        // and this would overwrite it with a clean stop.
        if (notifiedStarted) {
            notifiedStarted = false;
            NativeBridge.nativeSystemAudioStopped(null);
        }
        super.onDestroy();
    }
}
