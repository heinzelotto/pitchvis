package org.p1graph.pitchvis;

/**
 * Java -> Rust entry points.
 *
 * These live on their own class rather than on MainActivity or the capture service so that the
 * exported JNI symbol names (Java_org_p1graph_pitchvis_NativeBridge_*) stay stable no matter how
 * those classes are later refactored or renamed.
 *
 * Resolution is by symbol name (dlsym), not RegisterNatives, so nothing here needs FindClass.
 * See the module docs in android_system_audio.rs for why that matters.
 */
public final class NativeBridge {
    static {
        // Idempotent: MainActivity also loads this. Declared here too so that the natives are
        // guaranteed resolvable even if a caller reaches this class before the activity.
        System.loadLibrary("pitchvis_viewer_lib");
    }

    private NativeBridge() {}

    /**
     * Called for both the granted and the denied path, as the first statement of
     * onActivityResult. Clears the Rust-side suspend guard.
     */
    public static native void nativeConsentDialogFinished(boolean granted);

    /**
     * Capture is live. Called on the reader thread immediately before the read loop starts --
     * NOT from onStartCommand. Two reasons: it lets Rust do the blocking oboe close() off the
     * main thread, and it guarantees the microphone is only torn down once system audio is
     * genuinely running, so every failure path above leaves the mic untouched.
     *
     * @param sampleRate   the rate AudioRecord actually gave us (not necessarily what we asked)
     * @param channelCount the channel count AudioRecord actually gave us
     */
    public static native void nativeSystemAudioStarted(int sampleRate, int channelCount);

    /**
     * A block of interleaved float PCM. The buffer is a direct ByteBuffer in native byte order,
     * reused across calls, so Rust must copy out of it before returning and must not retain it.
     */
    public static native void nativeSystemAudioData(java.nio.ByteBuffer buffer, int frameCount);

    /**
     * Capture has stopped. This is the single convergence point for all three ways a stop can
     * originate: the in-app button, the system "Stop sharing" chip
     * (MediaProjection.Callback.onStop), and a failure during setup.
     *
     * @param reason null for a clean stop, else a human-readable error to surface in the UI
     */
    public static native void nativeSystemAudioStopped(String reason);
}
