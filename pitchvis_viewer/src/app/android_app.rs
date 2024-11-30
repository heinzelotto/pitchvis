use android_activity::AndroidApp;
use bevy::asset::AssetMetaCheck;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::diagnostic::LogDiagnosticsPlugin;
use bevy::log::LogPlugin;
use bevy::prelude::*;
use bevy::sprite::Material2dPlugin;
use bevy::window::AppLifecycle;
use jni::{
    objects::{JObject, JValue},
    sys::{jobject, jstring},
    JavaVM,
};
use oboe::{
    AudioInputCallback, AudioInputStreamSafe, AudioStream, AudioStreamBuilder, DataCallbackResult,
    Input, InputPreset, Mono, PerformanceMode, SampleRateConversionQuality, SharingMode,
};

use super::common::analysis_text_showhide;
use super::common::close_on_esc;
use super::common::fps_counter_showhide;
use super::common::frame_limiter_system;
use super::common::setup_analysis_text;
use super::common::setup_bloom_ui;
use super::common::setup_fps_counter;
use super::common::update_analysis_text_system;
use super::common::update_bloom_settings;
use super::common::update_fps_text_system;
use super::common::user_input_system;
use super::common::ActiveTouches;
use super::common::SettingsState;
use crate::analysis_system;
use crate::audio_system;
use crate::display_system;
use crate::vqt_system;
use pitchvis_analysis::vqt::VqtParameters;

#[cfg_attr(not(feature = "ml"), link(name = "c++_shared"))]
extern "C" {}

pub const BUFSIZE: usize = 2 * 16384;

fn handle_lifetime_events_system(
    mut lifetime_events: EventReader<AppLifecycle>,
    // audio_control_tx: ResMut<AudioControlChannelResource>,
    mut exit: EventWriter<AppExit>,
) {
    use bevy::window::AppLifecycle;

    for event in lifetime_events.read() {
        match event {
            // Upon receiving the `Suspended` event, the application has 1 frame before it is paused
            // As audio happens in an independent thread, it needs to be stopped
            AppLifecycle::WillSuspend | AppLifecycle::Suspended => {
                // FIXME: suspending audio does not seem to be good enough, we still get crashes
                // that block the entire android UI and sometimes seem to require a reboot.
                // audio_control_tx.0.send(AudioControl::Pause).unwrap();

                // log
                log::warn!("Application suspended, exiting.");

                // For now, just quit.
                // FIXME: also doesn't seem to work, the app is not exiting
                exit.send(AppExit::Success);

                // This is a workaround to exit the app, but it's not clean.
                // And it also doesn't work. This Event is only received when _reopening_ the app.
                std::process::exit(0);
            }
            // On `Resumed``, audio can continue playing
            AppLifecycle::WillResume => {
                // audio_control_tx.0.send(AudioControl::Play).unwrap();
            }
            // `Started` is the only other event for now, more to come in the next Bevy version
            _ => (),
        }
    }
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

struct AudioCallback {
    ring_buffer: std::sync::Arc<std::sync::Mutex<pitchvis_audio::RingBuffer>>,
    agc: dagc::MonoAgc,
}

impl AudioInputCallback for AudioCallback {
    type FrameType = (f32, Mono);

    fn on_audio_ready(
        &mut self,
        stream: &mut dyn AudioInputStreamSafe,
        data: &[f32],
    ) -> DataCallbackResult {
        // check for invalid samples
        if let Some(x) = data.iter().find(|x| !x.is_finite()) {
            log::warn!("bad audio sample encountered: {x}");
            return DataCallbackResult::Continue;
        }

        // agc processing
        let sample_sq_sum = data.iter().map(|x| x.powi(2)).sum::<f32>();
        self.agc.freeze_gain(sample_sq_sum < 1e-6);

        // update ring buffer
        let mut rb = self.ring_buffer.lock().expect("locking failed");
        rb.buf.drain(..data.len());
        rb.buf.extend_from_slice(data);
        let begin = rb.buf.len() - data.len();
        self.agc.process(&mut rb.buf[begin..]);
        rb.gain = self.agc.gain();

        // calculate latency
        // use nix::time::{clock_gettime, ClockId};
        // if let Ok(ts) = stream.get_timestamp(ClockId::CLOCK_MONOTONIC.into()) {
        //     let app_frame_index = stream.get_frames_read();

        //     // ts is a snapshot. We use the sample rate to calculate the corresponding
        //     // time of the latest sample we are currently handling.
        //     let frame_delta = app_frame_index - ts.position;
        //     let frame_time_delta = frame_delta as f64 / SR as f64 * 1e9;
        //     let app_frame_in_hardware_time = ts.timestamp + frame_time_delta as i64;

        //     // We assume that the frame we are currently handling is being processed
        //     // NOW, and calculate the latency by comparing this to the time it originated
        //     // from the hardware.
        //     let app_frame_time =
        //         std::time::Duration::from(clock_gettime(ClockId::CLOCK_MONOTONIC).unwrap())
        //             .as_nanos();
        //     let latency_nanos = app_frame_time - app_frame_in_hardware_time as u128;

        //     rb.latency_ms = Some((latency_nanos as f64 / 1e6) as f32);
        // } else {
        //     rb.latency_ms = None;
        // }

        rb.latency_ms = stream.calculate_latency_millis().ok().map(|x| x as f32);
        rb.chunk_size_ms = data.len() as f32 / stream.get_sample_rate() as f32 * 1000.0;

        DataCallbackResult::Continue
    }
}

// android main function
#[bevy_main]
fn main() -> AppExit {
    std::env::set_var("RUST_BACKTRACE", "1");

    use std::{process::exit, sync::mpsc};

    // env_logger::init();

    if !microphone_permission_granted("android.permission.RECORD_AUDIO") {
        log::info!("requesting microphone permission");
        request_microphone_permission(
            bevy_winit::ANDROID_APP
                .get()
                .expect("Bevy must be setup with the #[bevy_main] macro on Android"),
            "android.permission.RECORD_AUDIO",
        );
    }
    // wait until permission is granted
    while !microphone_permission_granted("android.permission.RECORD_AUDIO") {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // keep screen awake. This is a bitflags! enum, the second argument is an empty bitflags mask
    bevy_winit::ANDROID_APP
        .get()
        .expect("Bevy must be setup with the #[bevy_main] macro on Android")
        .set_window_flags(
            android_activity::WindowManagerFlags::KEEP_SCREEN_ON,
            android_activity::WindowManagerFlags::empty(),
        );

    let vqt_parameters: VqtParameters = VqtParameters::default();

    let vqt = pitchvis_analysis::vqt::Vqt::new(&vqt_parameters);

    // let (audio_control_channel_tx, audio_control_channel_rx) = mpsc::channel::<AudioControl>();
    let mut ring_buffer = pitchvis_audio::RingBuffer {
        buf: Vec::new(),
        gain: 0.0,
        latency_ms: None,
        chunk_size_ms: 0.0,
    };
    ring_buffer.buf.resize(BUFSIZE, 0f32);
    let ring_buffer = std::sync::Mutex::from(ring_buffer);
    let ring_buffer = std::sync::Arc::new(ring_buffer);

    let mut agc = dagc::MonoAgc::new(0.07, 0.0001).expect("mono-agc creation failed");

    let callback = AudioCallback {
        ring_buffer: ring_buffer.clone(),
        agc,
    };

    let mut audio_stream = AudioStreamBuilder::default()
        .set_direction::<Input>()
        .set_performance_mode(PerformanceMode::LowLatency)
        .set_sample_rate(vqt_parameters.sr as i32)
        // TODO: support all microphone channels if `Mono` does not mix them down
        .set_channel_count::<Mono>()
        .set_format::<f32>()
        .set_sharing_mode(SharingMode::Exclusive)
        .set_input_preset(InputPreset::Unprocessed)
        // this will solve bugs on some devices
        // TODO: is `Best` necessary?
        .set_sample_rate_conversion_quality(SampleRateConversionQuality::Best)
        .set_callback(callback)
        .open_stream()
        .unwrap();

    audio_stream.request_start().unwrap();

    let update_vqt_system = vqt_system::update_vqt_to_system(BUFSIZE);
    let update_analysis_state_system = analysis_system::update_analysis_state_to_system();
    let update_display_system = display_system::update_display_to_system(&vqt_parameters.range);

    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    //resizable: false,
                    mode: bevy::window::WindowMode::BorderlessFullscreen,
                    ..default()
                }),
                ..default()
            })
            .set(LogPlugin {
                filter: "wgpu=error,warn".to_string(),
                level: bevy::log::Level::DEBUG,
                ..default()
            })
            .set(AssetPlugin {
                meta_check: AssetMetaCheck::Never,
                ..default()
            }),
        LogDiagnosticsPlugin::default(),
        FrameTimeDiagnosticsPlugin::default(),
        Material2dPlugin::<display_system::material::NoisyColorMaterial>::default(),
    ))
    .insert_resource(vqt_system::VqtResource(vqt))
    .insert_resource(vqt_system::VqtResultResource::new(&vqt_parameters.range))
    .insert_resource(audio_system::AudioBufferResource(ring_buffer))
    .insert_resource(analysis_system::AnalysisStateResource(
        pitchvis_analysis::analysis::AnalysisState::new(
            vqt_parameters.range.clone(),
            pitchvis_analysis::analysis::AnalysisParameters::default(),
        ),
    ))
    .insert_resource(display_system::CylinderEntityListResource(Vec::new()))
    // .insert_resource(AudioControlChannelResource(audio_control_channel_tx))
    .insert_resource(SettingsState {
        display_mode: display_system::DisplayMode::PitchnamesCalmness,
        fps_limit: None,
    })
    .insert_resource(ActiveTouches::default())
    .add_systems(
        Startup,
        (
            display_system::setup_display_to_system(&vqt_parameters.range),
            setup_fps_counter,
            setup_bloom_ui,
            setup_analysis_text,
        ),
    )
    .add_systems(
        Update,
        (
            close_on_esc,
            frame_limiter_system,
            update_vqt_system,
            user_input_system,
            handle_lifetime_events_system,
            update_analysis_state_system.after(update_vqt_system),
            update_display_system.after(update_analysis_state_system),
            update_fps_text_system.after(update_vqt_system),
            fps_counter_showhide,
            update_analysis_text_system.after(update_analysis_state_system),
            analysis_text_showhide,
            update_bloom_settings.after(update_analysis_state_system),
        ),
    );

    // MSAA makes some Android devices panic, this is under investigation
    // https://github.com/bevyengine/bevy/issues/8229
    #[cfg(target_os = "android")]
    app.insert_resource(Msaa::Off);

    app.run()
}
