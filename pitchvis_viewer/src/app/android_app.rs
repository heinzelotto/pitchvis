use android_activity::AndroidApp;
use bevy::diagnostic::LogDiagnosticsPlugin;
use bevy::input::InputSystems;
use bevy::log::LogPlugin;
use bevy::prelude::*;
use bevy::window::AppLifecycle;
use bevy_persistent::StorageFormat;
use jni::{
    objects::{JObject, JValue},
    sys::{jobject, jstring},
    JavaVM,
};
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use super::android_system_audio;
use super::common::{
    build_common_app, correct_touch_insets, register_common_update_systems,
    register_startup_systems, PlatformConfig, SystemAudioState, SystemBarInset,
};
use crate::analysis_system;
use crate::app::common::ScreenLockState;
use crate::display_system;
use crate::vqt_system;
use pitchvis_analysis::vqt::VqtParameters;

#[cfg_attr(not(feature = "ml"), link(name = "c++_shared"))]
extern "C" {}

// TODO: Make BUFSIZE dynamically adjustable to match n_fft changes from parameter tuning.
// Currently set to 8 * 16384 = 131072 to accommodate the maximum tunable n_fft value.
// This must be >= the maximum n_fft value to prevent integer overflow in vqt_system.rs.
const BUFSIZE: usize = 8 * 16384;
const DEFAULT_FPS: u32 = 60;

fn handle_lifetime_events_system(
    mut lifetime_events: MessageReader<AppLifecycle>,
    // audio_control_tx: ResMut<AudioControlChannelResource>,
    mut exit: MessageWriter<AppExit>,
) {
    use bevy::window::AppLifecycle;

    for event in lifetime_events.read() {
        match event {
            // Upon receiving the `Suspended` event, the application has 1 frame before it is paused
            // As audio happens in an independent thread, it needs to be stopped
            AppLifecycle::WillSuspend | AppLifecycle::Suspended => {
                // The MediaProjection consent dialog is a separate activity. If showing it makes
                // Bevy emit a suspend event, exiting here would kill the app the instant the
                // popup appears. Suppress the exit while the dialog is up.
                //
                // The flag is cleared by `nativeConsentDialogFinished` on both the granted and
                // the denied path, and by a timeout in the driver system, so it cannot stick.
                if android_system_audio::AWAITING_CONSENT.load(Ordering::Acquire) {
                    log::warn!(
                        "PITCHVIS_PROBE: suspend event during MediaProjection consent \
                         — suppressing exit"
                    );
                    continue;
                }

                // FIXME: suspending audio does not seem to be good enough, we still get crashes
                // that block the entire android UI and sometimes seem to require a reboot.
                // audio_control_tx.0.send(AudioControl::Pause).unwrap();

                // log
                log::warn!("Application suspended, exiting.");

                // For now, just quit.
                exit.write(AppExit::Success);

                // This is a workaround to exit the app, but it's not clean. One of them seems to
                // work after the change from native-activity to game-activity and since update to
                // bevy 0.15.
                std::process::exit(0);
            }
            // On `Resumed``, audio can continue playing
            AppLifecycle::WillResume => {
                // audio_control_tx.0.send(AudioControl::Play).unwrap();
            }
            // `Idle`, `Running`
            _ => (),
        }
    }
}

/// Bridges the platform-agnostic [`SystemAudioState`] resource (driven by the debug button) to
/// the Android statics in [`android_system_audio`].
///
/// - Pushes a pending toggle to Java (consent request or stop).
/// - Pulls the ground truth back from `SYS_AUDIO_ACTIVE` and `LAST_ERROR`, so the button follows
///   what is actually running — including a stop initiated from outside the app (the system
///   "Stop sharing" chip).
/// - Fail-safes the suspend guard: if `onActivityResult` is ever lost, clears `AWAITING_CONSENT`
///   after a timeout so it cannot wedge the app into never-suspend.
fn drive_system_audio_system(
    mut sa: ResMut<SystemAudioState>,
    mut requested_at: Local<Option<std::time::Instant>>,
) {
    let Some(app) = bevy::android::ANDROID_APP.get() else {
        return;
    };

    // Push a pending toggle out to Java.
    if sa.pending {
        sa.pending = false;
        if sa.desired {
            android_system_audio::request_consent(app);
            *requested_at = Some(std::time::Instant::now());
        } else {
            android_system_audio::request_stop(app);
        }
    }

    // Fail-safe: never let the suspend guard stick if the consent result is lost.
    if let Some(t) = *requested_at {
        if t.elapsed() > std::time::Duration::from_secs(60) {
            android_system_audio::AWAITING_CONSENT.store(false, Ordering::Release);
            *requested_at = None;
        }
    }

    // Pull ground truth back.
    let active = android_system_audio::is_active();
    if sa.enabled != active {
        sa.enabled = active;
        sa.desired = active;
        *requested_at = None;
    }
    if let Some(err) = android_system_audio::take_error() {
        sa.last_error = Some(err);
        sa.desired = false;
        *requested_at = None;
    }
}

/// System to hide/show Android navigation bar based on screen lock state
///
/// NOTE: This feature is currently disabled due to conflicts with Bevy's BorderlessFullscreen mode.
/// The app already runs in fullscreen, so additional navigation bar hiding may not be necessary.
/// Keeping the system here for future investigation if needed.
fn handle_android_ui_visibility(
    lock_state: Res<ScreenLockState>,
    mut last_state: Local<Option<bool>>,
) {
    // Check if this is the first run or if state actually changed
    let is_locked = lock_state.0;
    if last_state.map_or(false, |prev| prev == is_locked) {
        // State hasn't changed, skip
        return;
    }

    log::info!("handle_android_ui_visibility called: locked={}", is_locked);

    // Update our tracking
    *last_state = Some(is_locked);

    // DISABLED: Window flag manipulation causes freezing
    // The app is already in BorderlessFullscreen mode via Bevy, which should
    // provide adequate fullscreen experience without manual flag manipulation.
    //
    // If we need to re-enable this in the future, we should investigate:
    // 1. Using View.setSystemUiVisibility() via proper UI thread posting
    // 2. Coordinating with Bevy's window management
    // 3. Testing on multiple Android versions

    log::info!(
        "Screen lock {}: Navigation bar control disabled (app already in fullscreen)",
        if is_locked { "enabled" } else { "disabled" }
    );

    // let android_app = match bevy_window::ANDROID_APP.get() {
    //     Some(app) => app,
    //     None => {
    //         log::warn!("Android app not available");
    //         return;
    //     }
    // };
    //
    // if is_locked {
    //     android_app.set_window_flags(
    //         android_activity::WindowManagerFlags::LAYOUT_NO_LIMITS,
    //         android_activity::WindowManagerFlags::LAYOUT_NO_LIMITS,
    //     );
    // } else {
    //     android_app.set_window_flags(
    //         android_activity::WindowManagerFlags::empty(),
    //         android_activity::WindowManagerFlags::LAYOUT_NO_LIMITS,
    //     );
    // }
}

/// Refreshes [`SystemBarInset`] (in logical pixels) from the current Android content rect.
///
/// The content rect is the window area not covered by system UI; its top/left origin is the
/// system-bar inset in *physical* pixels, so we divide by the window scale factor to match
/// the logical coordinates Bevy uses for touch positions and UI layout. In fullscreen the
/// inset is zero, so [`correct_touch_insets`] does nothing; it only bites in split-screen /
/// multi-window, where Android refuses to let us hide the bars.
fn update_system_bar_inset(mut inset: ResMut<SystemBarInset>, windows: Query<&Window>) {
    let Some(app) = bevy::android::ANDROID_APP.get() else {
        return;
    };
    let scale_factor = windows
        .iter()
        .next()
        .map(|w| w.resolution.scale_factor())
        .unwrap_or(1.0);
    if scale_factor <= 0.0 {
        return;
    }
    let rect = app.content_rect();
    inset.top = rect.top.max(0) as f32 / scale_factor;
    inset.left = rect.left.max(0) as f32 / scale_factor;
}

fn microphone_permission_granted(perm: &str) -> bool {
    let ctx = ndk_context::android_context();
    let jvm = unsafe { JavaVM::from_raw(ctx.vm().cast()) }.unwrap();
    let mut cth = jvm.attach_current_thread().unwrap();

    let j_permission = cth.new_string(perm).unwrap();
    let permission_granted = cth
        .call_method(
            &unsafe { JObject::from_raw(ctx.context() as jni::sys::jobject) },
            "checkSelfPermission",
            "(Ljava/lang/String;)I",
            &[JValue::try_from(&j_permission).unwrap()],
        )
        .unwrap();

    permission_granted.i().unwrap() != -1
}

pub fn request_microphone_permission(android_app: &AndroidApp, permission: &str) {
    let jvm = unsafe { JavaVM::from_raw(android_app.vm_as_ptr() as _) }.unwrap();
    let mut cth = jvm.attach_current_thread().unwrap();

    let j_permission = cth.new_string(permission).unwrap();

    let permissions: Vec<jstring> = vec![j_permission.into_raw()];
    let permissions_array = cth
        .new_object_array(
            permissions.len() as i32,
            "java/lang/String",
            JObject::null(),
        )
        .unwrap();
    for (i, permission) in permissions.into_iter().enumerate() {
        cth.set_object_array_element(&permissions_array, i as i32, unsafe {
            JObject::from_raw(permission)
        })
        .unwrap();
    }

    let activity = unsafe { JObject::from_raw(android_app.activity_as_ptr() as jobject) };
    let _res = cth
        .call_method(
            activity,
            "requestPermissions",
            "([Ljava/lang/String;I)V",
            &[JValue::Object(&permissions_array), JValue::Int(3)],
        )
        .unwrap();
}

/// This enum is used to control the audio stream. It is sent to the audio thread via a channel. This allows us to control the audio stream from the bevy thread.
// enum AudioControl {
//     Play,
//     Pause,
// }

// #[derive(Resource)]
// struct AudioControlChannelResource(std::sync::mpsc::Sender<AudioControl>);

// The microphone oboe stream and its `AudioCallback` now live in `android_system_audio`, which
// owns the source switch between the mic and system-audio capture.

// android main function
#[bevy_main]
fn main() -> AppExit {
    std::env::set_var("RUST_BACKTRACE", "1");

    let config_dir = PathBuf::from("/data/data/org.p1graph.pitchvis/files/");

    // Android-specific: Request microphone permission
    if !microphone_permission_granted("android.permission.RECORD_AUDIO") {
        log::info!("requesting microphone permission");
        request_microphone_permission(
            bevy::android::ANDROID_APP
                .get()
                .expect("Bevy must be setup with the #[bevy_main] macro on Android"),
            "android.permission.RECORD_AUDIO",
        );
    }
    // wait until permission is granted
    while !microphone_permission_granted("android.permission.RECORD_AUDIO") {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Android-specific: Keep screen awake
    bevy::android::ANDROID_APP
        .get()
        .expect("Bevy must be setup with the #[bevy_main] macro on Android")
        .set_window_flags(
            android_activity::WindowManagerFlags::KEEP_SCREEN_ON,
            android_activity::WindowManagerFlags::empty(),
        );

    let vqt_parameters: VqtParameters = VqtParameters::default();
    let vqt = pitchvis_analysis::vqt::Vqt::new(&vqt_parameters)
        .expect("Failed to create VQT with default parameters");

    // Android-specific: set up the shared ring buffer and start the microphone. The oboe stream
    // and the mic <-> system-audio switch are owned by `android_system_audio`.
    let mut ring_buffer = pitchvis_audio::RingBuffer {
        buf: Vec::new(),
        gain: 0.0,
        latency_ms: None,
        chunk_size_ms: 0.0,
    };
    ring_buffer.buf.resize(BUFSIZE, 0f32);
    let ring_buffer = std::sync::Arc::new(std::sync::Mutex::new(ring_buffer));

    android_system_audio::init(ring_buffer.clone());

    // Create system closures with BUFSIZE
    let update_vqt_system = vqt_system::update_vqt_to_system(BUFSIZE);
    let update_analysis_state_system = analysis_system::update_analysis_state_to_system();
    let update_display_system = display_system::update_display_to_system(&vqt_parameters.range);

    // Android-specific: Window configuration with fullscreen and logging
    let window_config = Window {
        mode: bevy::window::WindowMode::BorderlessFullscreen(MonitorSelection::Primary),
        ..default()
    };

    // Android-specific: Add LogPlugin to additional plugins
    let log_plugin = LogPlugin {
        filter: "wgpu=error,warn".to_string(),
        level: bevy::log::Level::DEBUG,
        ..default()
    };

    // Build common app with Android-specific config
    let mut app = build_common_app(
        vqt,
        &vqt_parameters,
        PlatformConfig {
            config_dir,
            storage_format: StorageFormat::Toml,
            audio_buffer: ring_buffer,
            window_config,
            additional_plugins: vec![
                Box::new(LogDiagnosticsPlugin::default()),
                Box::new(log_plugin),
            ],
            system_audio_supported: android_system_audio::is_supported(),
        },
        DEFAULT_FPS,
    );

    // Register common startup systems
    register_startup_systems(&mut app, &vqt_parameters.range);

    // Register common update systems
    register_common_update_systems(
        &mut app,
        update_vqt_system,
        update_analysis_state_system,
        update_display_system,
    );

    // Android-specific: Add lifecycle event handler and UI visibility handler
    app.add_systems(
        Update,
        (
            handle_lifetime_events_system,
            handle_android_ui_visibility,
            // Bridges the debug button to the system-audio statics; runs before the button's
            // label-sync system so the button reflects state within the same frame.
            drive_system_audio_system.before(super::common::update_system_audio_button_system),
        ),
    );

    // Android-specific: realign touch coordinates for the system-bar inset in
    // split-screen/multi-window. `update_system_bar_inset` refreshes the inset, then
    // `correct_touch_insets` subtracts it; both run in PreUpdate before InputSystems so
    // bevy_input's Touches aggregation, bevy_ui focus, and user_input_system all see the
    // corrected coordinates.
    app.insert_resource(SystemBarInset::default());
    app.add_systems(
        PreUpdate,
        (update_system_bar_inset, correct_touch_insets)
            .chain()
            .before(InputSystems),
    );

    app.run()
}
