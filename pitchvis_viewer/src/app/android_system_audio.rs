//! System-audio capture on Android via MediaProjection / AudioPlaybackCapture, and the
//! microphone <-> system-audio source switch.
//!
//! This module owns *all* Android audio state. Bevy and the Java layer both talk to the statics
//! here; neither talks to the other directly. The JNI entry points are `extern "system" fn` with
//! no user-data pointer, so they must reach globals — and once one global exists, keeping every
//! state transition in one auditable file is free.
//!
//! # Source switching model
//!
//! The mic (oboe) and system audio (AudioRecord, driven from Java) are mutually exclusive — an
//! *exclusive switch*, never a mix. Only one writes to the shared [`RingBuffer`] at a time.
//!
//! Switching is **close-and-reopen**, not pause/resume: enabling system audio drops the oboe
//! stream entirely (releasing the mic and clearing the OS privacy indicator), and disabling it
//! rebuilds the stream from the same path used at startup.
//!
//! The single most important invariant: **the mic is torn down only once system capture is
//! genuinely live** (in [`Java_org_p1graph_pitchvis_NativeBridge_nativeSystemAudioStarted`], on
//! the Java reader thread). If consent is denied, or the foreground service fails, or
//! `AudioRecord` construction throws, the mic was never touched and the app never glitched —
//! every failure path is a no-op.
//!
//! # JNI conventions
//!
//! Both directions deliberately avoid `FindClass`:
//! * **Java -> Rust** resolves by dlsym against the `Java_org_p1graph_pitchvis_NativeBridge_*`
//!   symbols exported below. No `RegisterNatives`, so no class lookup.
//! * **Rust -> Java** calls `call_method` on the *existing* activity `jobject` from
//!   `AndroidApp::activity_as_ptr()`, the pattern `request_microphone_permission` already uses.
//!
//! `FindClass` for app classes (`org/p1graph/pitchvis/*`) fails from a native thread Java did not
//! start, because that thread's context classloader is the bootstrap loader, which cannot see app
//! classes — which is why `pitchvis_android_serial` caches the classloader and calls `loadClass`.
//! If you ever add a Rust -> Java call to a class *other than the activity instance*, you will hit
//! that wall and need the same workaround. Bootclasspath classes (`android/os/Build$VERSION`) are
//! always resolvable.

use android_activity::AndroidApp;
use jni::{
    objects::{JByteBuffer, JClass, JObject, JString},
    sys::{jboolean, jint, jobject},
    JNIEnv, JavaVM,
};
use oboe::{
    AudioInputCallback, AudioInputStreamSafe, AudioStream, AudioStreamAsync, AudioStreamBuilder,
    DataCallbackResult, Input, InputPreset, Mono, PerformanceMode, SampleRateConversionQuality,
    SharingMode,
};
use pitchvis_audio::RingBuffer;
use rubato::{FftFixedIn, Resampler};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Analysis sample rate. Must equal `pitchvis_analysis::vqt::DEFAULT_SR`; the mic stream and the
/// system-audio resampler both target it.
const TARGET_SR: u32 = 22050;

/// Shared audio buffer, set once from `main()`. Both the mic callback and the system-audio JNI
/// push find it here without touching Bevy.
static RING_BUFFER: OnceLock<Arc<Mutex<RingBuffer>>> = OnceLock::new();

/// The live oboe mic stream, or `None` while system audio is active.
///
/// `AudioStreamAsync<Input, AudioCallback>` is `Send`: its handle is `[u64; 2]` (a plain array,
/// not a raw pointer) and both type params are `Send`, so this static needs no `unsafe impl`.
static MIC_STREAM: Mutex<Option<AudioStreamAsync<Input, AudioCallback>>> = Mutex::new(None);

/// Per-session system-audio state (AGC + resampler + accumulator), present only while active.
static SYS_STATE: Mutex<Option<SysAudioState>> = Mutex::new(None);

/// True while system-audio capture is the active source.
static SYS_AUDIO_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Set while the MediaProjection consent dialog is on screen, so `handle_lifetime_events_system`
/// suppresses its `std::process::exit(0)` on a suspend event triggered by the dialog.
///
/// **Fail-safe, not fail-open**: if this ever stuck at `true`, the app would silently stop
/// exiting on suspend forever, reintroducing the crash class the FIXME in
/// `handle_lifetime_events_system` describes. The driver system in `android_app.rs` therefore
/// also clears it on a timeout, so a lost `onActivityResult` cannot wedge it.
pub static AWAITING_CONSENT: AtomicBool = AtomicBool::new(false);

/// Last error worth surfacing on the button, consumed by [`take_error`].
static LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);

struct SysAudioState {
    agc: dagc::MonoAgc,
    /// `None` when the capture rate already equals [`TARGET_SR`] (a passthrough).
    resampler: Option<FftFixedIn<f32>>,
    channels: usize,
    /// Mono samples awaiting a full resampler input chunk.
    accum: Vec<f32>,
}

// ---------------------------------------------------------------------------------------------
// Shared ring-buffer write path (used by both the mic callback and the system-audio push)
// ---------------------------------------------------------------------------------------------

/// Appends `data` to the ring buffer with AGC, matching the original mic-callback behavior.
///
/// Rejects non-finite input, freezes AGC gain on near-silence (so a fully-silent capture cannot
/// crank gain to infinity), shifts the ring buffer, and AGC-processes only the newly appended
/// tail. `data.len()` must not exceed the buffer length (it never does: chunks are far smaller
/// than `BUFSIZE`).
fn push_samples(
    rb: &mut RingBuffer,
    agc: &mut dagc::MonoAgc,
    data: &[f32],
    latency_ms: Option<f32>,
    sr: u32,
) {
    if data.iter().any(|x| !x.is_finite()) {
        log::warn!("bad audio sample encountered, dropping chunk");
        return;
    }
    let sample_sq_sum: f32 = data.iter().map(|x| x * x).sum();
    agc.freeze_gain(sample_sq_sum < 1e-6);

    rb.buf.drain(..data.len());
    rb.buf.extend_from_slice(data);
    let begin = rb.buf.len() - data.len();
    agc.process(&mut rb.buf[begin..]);
    rb.gain = agc.gain();
    rb.latency_ms = latency_ms;
    rb.chunk_size_ms = data.len() as f32 / sr as f32 * 1000.0;
}

/// Zeroes the ring buffer on a source switch. `BUFSIZE` is ~5.9 s at [`TARGET_SR`] and the VQT
/// reads the newest ~1.5 s, so without this the old source ghosts on the display after a switch.
fn zero_ring_buffer() {
    if let Some(rb) = RING_BUFFER.get() {
        let mut rb = rb.lock().expect("ring buffer poisoned");
        rb.buf.iter_mut().for_each(|x| *x = 0.0);
        rb.gain = 0.0;
        rb.latency_ms = None;
    }
}

fn set_error(msg: String) {
    log::error!("system audio: {msg}");
    *LAST_ERROR.lock().expect("LAST_ERROR poisoned") = Some(msg);
}

/// Consumes and returns the last error, if any. Polled by the Bevy driver system.
pub fn take_error() -> Option<String> {
    LAST_ERROR.lock().expect("LAST_ERROR poisoned").take()
}

/// Whether system-audio capture is currently the active source.
pub fn is_active() -> bool {
    SYS_AUDIO_ACTIVE.load(Ordering::Acquire)
}

fn new_agc() -> Option<dagc::MonoAgc> {
    match dagc::MonoAgc::new(0.07, 0.0001) {
        Ok(a) => Some(a),
        Err(e) => {
            set_error(format!("AGC creation failed: {e:?}"));
            None
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Microphone (oboe) path
// ---------------------------------------------------------------------------------------------

/// Oboe input callback for the microphone. Uses a fresh AGC, never shared with the system-audio
/// path — carrying a converged gain across a source switch would produce a loud transient.
pub struct AudioCallback {
    ring_buffer: Arc<Mutex<RingBuffer>>,
    agc: dagc::MonoAgc,
}

impl AudioInputCallback for AudioCallback {
    type FrameType = (f32, Mono);

    fn on_audio_ready(
        &mut self,
        stream: &mut dyn AudioInputStreamSafe,
        data: &[f32],
    ) -> DataCallbackResult {
        // Safety net: if a switch to system audio is mid-flight, don't also write here. In
        // practice the stream is closed before SYS_AUDIO_ACTIVE flips, so this rarely fires.
        if SYS_AUDIO_ACTIVE.load(Ordering::Acquire) {
            return DataCallbackResult::Continue;
        }
        let latency_ms = stream.calculate_latency_millis().ok().map(|x| x as f32);
        let sr = stream.get_sample_rate() as u32;
        let mut rb = self.ring_buffer.lock().expect("ring buffer poisoned");
        push_samples(&mut rb, &mut self.agc, data, latency_ms, sr);
        DataCallbackResult::Continue
    }
}

/// Builds and starts an oboe mic stream against the shared ring buffer. Returns `Result` (rather
/// than unwrapping like startup did) because the reopen-after-system-audio path can legitimately
/// fail if something else grabbed the mic — and a panic there would kill the app on toggle-*off*.
fn open_mic_stream() -> anyhow::Result<AudioStreamAsync<Input, AudioCallback>> {
    let ring_buffer = RING_BUFFER
        .get()
        .expect("ring buffer must be initialised before opening the mic")
        .clone();
    let agc = dagc::MonoAgc::new(0.07, 0.0001)
        .map_err(|e| anyhow::anyhow!("mono-agc creation failed: {e:?}"))?;

    let callback = AudioCallback { ring_buffer, agc };

    let mut stream = AudioStreamBuilder::default()
        .set_direction::<Input>()
        .set_performance_mode(PerformanceMode::LowLatency)
        .set_sample_rate(TARGET_SR as i32)
        .set_channel_count::<Mono>()
        .set_format::<f32>()
        .set_sharing_mode(SharingMode::Exclusive)
        .set_input_preset(InputPreset::Unprocessed)
        .set_sample_rate_conversion_quality(SampleRateConversionQuality::Best)
        .set_callback(callback)
        .open_stream()
        .map_err(|e| anyhow::anyhow!("open_stream failed: {e:?}"))?;
    stream
        .request_start()
        .map_err(|e| anyhow::anyhow!("request_start failed: {e:?}"))?;
    Ok(stream)
}

/// Opens the mic stream if it is not already running. Called from `main()` at startup and
/// whenever system audio stops. Logs and records an error rather than panicking on failure.
pub fn start_mic() {
    let mut guard = MIC_STREAM.lock().expect("MIC_STREAM poisoned");
    if guard.is_some() {
        return;
    }
    match open_mic_stream() {
        Ok(stream) => *guard = Some(stream),
        Err(e) => set_error(format!("mic (re)open failed: {e}")),
    }
}

/// Closes the mic stream. Drop runs oboe `close()`, which blocks until the in-flight callback
/// returns — so after this call no mic callback can touch the ring buffer.
fn stop_mic() {
    MIC_STREAM.lock().expect("MIC_STREAM poisoned").take();
}

// ---------------------------------------------------------------------------------------------
// Rust -> Java calls
// ---------------------------------------------------------------------------------------------

fn call_activity_void(android_app: &AndroidApp, method: &str) -> anyhow::Result<()> {
    let jvm = unsafe { JavaVM::from_raw(android_app.vm_as_ptr() as _) }?;
    let mut env = jvm.attach_current_thread()?;
    let activity = unsafe { JObject::from_raw(android_app.activity_as_ptr() as jobject) };
    let res = env.call_method(activity, method, "()V", &[]);
    if res.is_err() {
        let _ = env.exception_clear();
    }
    res.map(|_| ()).map_err(Into::into)
}

/// Asks `MainActivity` to show the MediaProjection consent dialog.
///
/// Sets [`AWAITING_CONSENT`] *before* the call so the Bevy thread cannot observe a dialog-induced
/// suspend event before the guard is up. Java hops to the UI thread itself, so this is safe from
/// the Bevy thread and we never build a `Runnable` here.
pub fn request_consent(android_app: &AndroidApp) {
    AWAITING_CONSENT.store(true, Ordering::Release);
    if let Err(e) = call_activity_void(android_app, "requestSystemAudioCapture") {
        log::error!("request_consent: call failed: {e}");
        AWAITING_CONSENT.store(false, Ordering::Release);
    }
}

/// Asks `MainActivity` to stop the capture service. The teardown converges on
/// `nativeSystemAudioStopped`, which restarts the mic.
pub fn request_stop(android_app: &AndroidApp) {
    if let Err(e) = call_activity_void(android_app, "stopSystemAudioCapture") {
        log::error!("request_stop: call failed: {e}");
    }
}

// ---------------------------------------------------------------------------------------------
// Startup / capability
// ---------------------------------------------------------------------------------------------

/// Records the shared ring buffer and starts the mic. Call once from `main()`.
pub fn init(ring_buffer: Arc<Mutex<RingBuffer>>) {
    let _ = RING_BUFFER.set(ring_buffer);
    start_mic();
}

/// Reads `android.os.Build.VERSION.SDK_INT`. `android/os/Build$VERSION` is on the bootclasspath,
/// so this resolves from any thread without the classloader dance in the module docs.
pub fn sdk_int() -> i32 {
    let ctx = ndk_context::android_context();
    let Ok(jvm) = (unsafe { JavaVM::from_raw(ctx.vm().cast()) }) else {
        return 0;
    };
    let Ok(mut env) = jvm.attach_current_thread() else {
        return 0;
    };
    env.get_static_field("android/os/Build$VERSION", "SDK_INT", "I")
        .and_then(|v| v.i())
        .unwrap_or(0)
}

/// True if this device can capture system audio. AudioPlaybackCapture is API 29+, and `minSdk`
/// is 28, so this is a real runtime check.
pub fn is_supported() -> bool {
    sdk_int() >= 29
}

// ---------------------------------------------------------------------------------------------
// Java -> Rust entry points
// ---------------------------------------------------------------------------------------------

/// Called from `onActivityResult` (both granted and denied paths). Clears the suspend guard.
#[no_mangle]
pub extern "system" fn Java_org_p1graph_pitchvis_NativeBridge_nativeConsentDialogFinished(
    _env: JNIEnv,
    _class: JClass,
    result_ok: jboolean,
) {
    let granted = result_ok != 0;
    log::info!("consent dialog finished, granted={granted}");
    AWAITING_CONSENT.store(false, Ordering::Release);
}

/// Capture is live (called on the reader thread, before the read loop). This is where the mic is
/// torn down and the system-audio session is installed — never earlier, so every failure path
/// above leaves the mic running.
#[no_mangle]
pub extern "system" fn Java_org_p1graph_pitchvis_NativeBridge_nativeSystemAudioStarted(
    _env: JNIEnv,
    _class: JClass,
    sample_rate: jint,
    channel_count: jint,
) {
    let rate = sample_rate.max(1) as u32;
    let channels = channel_count.max(1) as usize;
    log::info!("system audio started: {rate} Hz, {channels} ch -> resample to {TARGET_SR} Hz mono");

    // 1. Release the mic first: after this, no mic callback can write the ring buffer.
    stop_mic();
    // 2. Clear stale mic audio so it doesn't ghost on the display.
    zero_ring_buffer();

    // 3. Build the resampler. We capture at the device-native rate and decimate here rather than
    //    asking AudioRecord for 22050, whose downsampling anti-alias filter is weak — aliased HF
    //    content would fold into the analysis band and paint phantom pitches.
    let resampler = if rate == TARGET_SR {
        None
    } else {
        match FftFixedIn::<f32>::new(rate as usize, TARGET_SR as usize, 1024, 2, 1) {
            Ok(r) => Some(r),
            Err(e) => {
                // Degraded: without a resampler the pitch would be wrong, but this is essentially
                // unreachable for real device rates. Surface it and pass through.
                set_error(format!("resampler init failed ({rate}->{TARGET_SR}): {e:?}"));
                None
            }
        }
    };

    let Some(agc) = new_agc() else {
        // Couldn't build the AGC; leave the mic stopped but don't install a half-state.
        return;
    };

    *SYS_STATE.lock().expect("SYS_STATE poisoned") = Some(SysAudioState {
        agc,
        resampler,
        channels,
        accum: Vec::new(),
    });
    // 4. Only now is system audio the active source.
    SYS_AUDIO_ACTIVE.store(true, Ordering::Release);
}

/// A block of interleaved float PCM from the capture reader thread. The `ByteBuffer` is a reused
/// direct buffer in native byte order; we copy out of it and never retain it.
#[no_mangle]
pub extern "system" fn Java_org_p1graph_pitchvis_NativeBridge_nativeSystemAudioData(
    env: JNIEnv,
    _class: JClass,
    buffer: JByteBuffer,
    frame_count: jint,
) {
    if !SYS_AUDIO_ACTIVE.load(Ordering::Acquire) {
        return;
    }
    let mut guard = SYS_STATE.lock().expect("SYS_STATE poisoned");
    let Some(st) = guard.as_mut() else {
        return;
    };

    let frames = frame_count.max(0) as usize;
    let total = frames * st.channels;
    if total == 0 {
        return;
    }

    let ptr = match env.get_direct_buffer_address(&buffer) {
        Ok(p) if !p.is_null() => p as *const f32,
        _ => {
            log::warn!("system audio: null/invalid direct buffer");
            return;
        }
    };
    // SAFETY: Java guarantees `frame_count * channels` little-endian f32 samples in a direct
    // buffer whose capacity we sized on the Java side. The slice is used only within this call.
    let interleaved = unsafe { std::slice::from_raw_parts(ptr, total) };

    // Downmix to mono (append to the accumulator; no ring-buffer lock held here).
    match st.channels {
        1 => st.accum.extend_from_slice(interleaved),
        2 => st
            .accum
            .extend(interleaved.chunks_exact(2).map(|f| 0.5 * (f[0] + f[1]))),
        n => st
            .accum
            .extend(interleaved.chunks_exact(n).map(|f| f.iter().sum::<f32>() / n as f32)),
    }

    // Drain the accumulator through the resampler in whatever chunk size it asks for.
    let mut out: Vec<f32> = Vec::new();
    if let Some(rs) = st.resampler.as_mut() {
        loop {
            let want = rs.input_frames_next();
            if st.accum.len() < want {
                break;
            }
            let chunk: Vec<f32> = st.accum.drain(..want).collect();
            match rs.process(&[chunk], None) {
                Ok(mut o) => out.append(&mut o[0]),
                Err(e) => {
                    log::error!("system audio resample failed: {e}");
                    break;
                }
            }
        }
    } else {
        out.append(&mut st.accum);
    }
    if out.is_empty() {
        return;
    }

    // Resampling is done; only push_samples holds the ring-buffer lock, keeping it off the
    // Bevy frame's critical path.
    if let Some(rb) = RING_BUFFER.get() {
        let mut rb = rb.lock().expect("ring buffer poisoned");
        push_samples(&mut rb, &mut st.agc, &out, None, TARGET_SR);
    }
}

/// Capture stopped. The single convergence point for every stop origin — the in-app button, the
/// system "Stop sharing" chip, and a setup failure. Restarts the mic.
#[no_mangle]
pub extern "system" fn Java_org_p1graph_pitchvis_NativeBridge_nativeSystemAudioStopped(
    mut env: JNIEnv,
    _class: JClass,
    reason: JString,
) {
    SYS_AUDIO_ACTIVE.store(false, Ordering::Release);
    *SYS_STATE.lock().expect("SYS_STATE poisoned") = None;

    if !reason.is_null() {
        if let Ok(s) = env.get_string(&reason) {
            set_error(String::from(s));
        }
    }
    log::info!("system audio stopped, restarting mic");

    zero_ring_buffer();
    start_mic();
}
