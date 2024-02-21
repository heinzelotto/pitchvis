#[cfg(target_arch = "wasm32")]
use crate::audio_system::AudioBufferResource;
use anyhow::Result;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
mod analysis_system;
mod audio_system;
mod cqt_system;
mod display_system;
#[cfg(feature = "ml")]
mod ml_system;
mod util;
use android_activity::AndroidApp;
use jni::{
    objects::{JObject, JValue},
    sys::{jobject, jstring}, JavaVM,
};

#[cfg_attr(not(feature = "ml"), link(name = "c++_shared"))]
extern "C" {}

// increasing BUCKETS_PER_SEMITONE or Q will improve frequency resolution at cost of time resolution,
// increasing GAMMA will improve time resolution at lower frequencies.
pub const SR: usize = 22050;
pub const BUFSIZE: usize = 2 * SR;
pub const N_FFT: usize = 2 * 16384;
pub const FREQ_A1: f32 = 55.0;
pub const FREQ_A1_MIDI_KEY_ID: i32 = 33;
pub const UPSCALE_FACTOR: usize = 1;
pub const BUCKETS_PER_SEMITONE: usize = 3 * UPSCALE_FACTOR;
pub const BUCKETS_PER_OCTAVE: usize = 12 * BUCKETS_PER_SEMITONE;
pub const OCTAVES: usize = 7;
pub const SPARSITY_QUANTILE: f32 = 0.999;
pub const Q: f32 = 10.0 / UPSCALE_FACTOR as f32;
pub const GAMMA: f32 = 5.3 * Q;

const FPS: u64 = 30;

#[wasm_bindgen]
#[cfg(target_arch = "wasm32")]
pub async fn main_fun() -> Result<(), JsValue> {
    let audio_stream = pitchvis_audio::audio::AudioStream::async_new(SR, BUFSIZE)
        .await
        .unwrap();

    let cqt = pitchvis_analysis::cqt::Cqt::new(
        SR,
        N_FFT,
        FREQ_A1,
        BUCKETS_PER_OCTAVE,
        OCTAVES,
        SPARSITY_QUANTILE,
        Q,
        GAMMA,
    );

    audio_stream.play().unwrap();

    let update_cqt_system = cqt_system::update_cqt_to_system(BUFSIZE);
    let update_analysis_state_system =
        analysis_system::update_analysis_state_to_system(OCTAVES, BUCKETS_PER_OCTAVE);
    let update_display_system =
        display_system::update_display_to_system(BUCKETS_PER_OCTAVE, OCTAVES);

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        // .add_plugin(MaterialPlugin::<display_system::LineMaterial>::default())
        .insert_resource(cqt_system::CqtResource(cqt))
        .insert_resource(cqt_system::CqtResultResource::new(
            OCTAVES,
            BUCKETS_PER_OCTAVE,
        ))
        .insert_resource(audio_system::AudioBufferResource(
            audio_stream.ring_buffer.clone(),
        ))
        .insert_resource(analysis_system::AnalysisStateResource(
            pitchvis_analysis::analysis::AnalysisState::new(
                OCTAVES * BUCKETS_PER_OCTAVE,
                pitchvis_analysis::analysis::SPECTROGRAM_LENGTH,
            ),
        ))
        .add_startup_system(display_system::setup_display_to_system(
            OCTAVES,
            BUCKETS_PER_OCTAVE,
        ))
        .add_system(bevy::window::close_on_esc)
        .add_system(update_cqt_system)
        .add_system(update_analysis_state_system.after(update_cqt_system))
        .add_system(update_display_system.after(update_analysis_state_system))
        .run();

    Ok(())
}

fn frame_limiter_system() {
    use std::{thread, time};
    thread::sleep(time::Duration::from_millis((1000 / FPS).saturating_sub(5)));
}

#[cfg(not(target_arch = "wasm32"))]
pub fn main_fun() -> Result<()> {
    env_logger::init();

    let audio_stream = pitchvis_audio::audio::AudioStream::new(SR, BUFSIZE).unwrap();

    let cqt = pitchvis_analysis::cqt::Cqt::new(
        SR,
        N_FFT,
        FREQ_A1,
        BUCKETS_PER_OCTAVE,
        OCTAVES,
        SPARSITY_QUANTILE,
        Q,
        GAMMA,
    );

    audio_stream.play().unwrap();

    let update_cqt_system = cqt_system::update_cqt_to_system(BUFSIZE);
    let update_analysis_state_system =
        analysis_system::update_analysis_state_to_system(OCTAVES, BUCKETS_PER_OCTAVE);
    #[cfg(feature = "ml")]
    let update_ml_system = ml_system::update_ml_to_system();
    let update_display_system =
        display_system::update_display_to_system(BUCKETS_PER_OCTAVE, OCTAVES);

    #[cfg(feature = "ml")]
    let ml_model_resource = ml_system::MlModelResource(ml_system::MlModel::new("model.pt"));

    let mut app = App::new();
        app.add_plugins(DefaultPlugins)
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        //.add_plugin(MaterialPlugin::<display_system::LineMaterial>::default())
        .insert_resource(cqt_system::CqtResource(cqt))
        .insert_resource(cqt_system::CqtResultResource::new(
            OCTAVES,
            BUCKETS_PER_OCTAVE,
        ))
        .insert_resource(audio_system::AudioBufferResource(audio_stream.ring_buffer))
        .insert_resource(analysis_system::AnalysisStateResource(
            pitchvis_analysis::analysis::AnalysisState::new(
                OCTAVES * BUCKETS_PER_OCTAVE,
                pitchvis_analysis::analysis::SPECTROGRAM_LENGTH,
            ),
        ));

    #[cfg(feature = "ml")]
        app.insert_resource(ml_model_resource);

    app.insert_resource(display_system::CylinderEntityListResource(Vec::new()))
        .add_startup_system(display_system::setup_display_to_system(
            OCTAVES,
            BUCKETS_PER_OCTAVE,
        ))
        .add_system(bevy::window::close_on_esc)
        .add_system(frame_limiter_system)
        .add_system(update_cqt_system)
        .add_system(update_analysis_state_system.after(update_cqt_system));
#[cfg(feature = "ml")]
        app.add_system(update_ml_system.after(update_analysis_state_system))
        .add_system(update_display_system.after(update_ml_system))
        .run();
#[cfg(not(feature = "ml"))]
        app.add_system(update_display_system.after(update_analysis_state_system))
        .run();

    Ok(())
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

#[cfg(not(feature = "ml"))]
#[bevy_main]
fn main() {
    use bevy::{render::{RenderPlugin, settings::{WgpuSettings, WgpuFeatures}}, log::LogPlugin};

    env_logger::init();

    if !microphone_permission_granted("android.permission.RECORD_AUDIO") {
        log::info!("requesting microphone permission");
        request_microphone_permission(bevy_winit::ANDROID_APP
            .get()
            .expect("Bevy must be setup with the #[bevy_main] macro on Android"), "android.permission.RECORD_AUDIO");
    }
    // wait until permission is granted
    while !microphone_permission_granted("android.permission.RECORD_AUDIO") {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // keep screen awake. This is a bitflags! enum, the second argument is an empty bitflags mask
    bevy_winit::ANDROID_APP
            .get()
            .expect("Bevy must be setup with the #[bevy_main] macro on Android").set_window_flags(
                android_activity::WindowManagerFlags::KEEP_SCREEN_ON,
                android_activity::WindowManagerFlags::empty(),
            );

    //let audio_stream = pitchvis_audio::audio::AudioStream::new(SR, BUFSIZE).unwrap();

    let cqt = pitchvis_analysis::cqt::Cqt::new(
        SR,
        N_FFT,
        FREQ_A1,
        BUCKETS_PER_OCTAVE,
        OCTAVES,
        SPARSITY_QUANTILE,
        Q,
        GAMMA,
    );

    //audio_stream.play().unwrap();

    let mut ring_buffer = pitchvis_audio::audio::RingBuffer {
        buf: Vec::new(),
        gain: 0.0,
    };
    ring_buffer.buf.resize(BUFSIZE, 0f32);
    let ring_buffer = std::sync::Mutex::from(ring_buffer);
    let ring_buffer = std::sync::Arc::new(ring_buffer);
    let mut agc = dagc::MonoAgc::new(0.07, 0.0001).expect("mono-agc creation failed");
    let ring_buffer_input_thread_clone = ring_buffer.clone();
    let mut audio_stream = aaudio::AAudioStreamBuilder::new()
        .unwrap()
        .set_direction(aaudio::Direction::Input)
        .set_performance_mode(aaudio::PerformanceMode::LowLatency)
        .set_sample_rate(SR as i32)
        .set_format(aaudio::Format::F32)
        .set_channel_count(1)
        .set_sharing_mode(aaudio::SharingMode::Exclusive)
        .set_callbacks(
            move |_, data: &mut [u8], frames: i32| {
                let data = unsafe {
                    std::slice::from_raw_parts_mut(data.as_ptr() as *mut f32, frames as usize * 1)
                };
                if let Some(x) = data.iter().find(|x| !x.is_finite()) {
                    log::warn!("bad audio sample encountered: {x}");
                    return aaudio::CallbackResult::Continue;
                }
                let sample_sq_sum = data.iter().map(|x| x.powi(2)).sum::<f32>();
                agc.freeze_gain(sample_sq_sum < 1e-6);

                //log::info!("audio callback");

                let mut rb = ring_buffer_input_thread_clone
                    .lock()
                    .expect("locking failed");
                rb.buf.drain(..data.len());
                rb.buf.extend_from_slice(data);
                let begin = rb.buf.len() - data.len();
                agc.process(&mut rb.buf[begin..]);
                rb.gain = agc.gain();

                aaudio::CallbackResult::Continue
            },
            |_, _| {},
        )
        .open_stream()
        .unwrap();

    audio_stream.request_start().unwrap();

    let update_cqt_system = cqt_system::update_cqt_to_system(BUFSIZE);
    let update_analysis_state_system =
        analysis_system::update_analysis_state_to_system(OCTAVES, BUCKETS_PER_OCTAVE);
    let update_display_system =
        display_system::update_display_to_system(BUCKETS_PER_OCTAVE, OCTAVES);

    let mut app = App::new();
        app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                //resizable: false,
                mode: bevy::window::WindowMode::BorderlessFullscreen,
                ..default()
            }),
            ..default()
        })
        .set(LogPlugin {
            filter: "debug".into(),
            level: bevy::log::Level::DEBUG,
        })
        // .set(RenderPlugin {
        //     wgpu_settings: WgpuSettings {
        //         features: WgpuFeatures::POLYGON_MODE_LINE,
        //         ..default()
        //     },})
    
    )
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        //.add_plugin(MaterialPlugin::<display_system::LineMaterial>::default())
        .insert_resource(cqt_system::CqtResource(cqt))
        .insert_resource(cqt_system::CqtResultResource::new(
            OCTAVES,
            BUCKETS_PER_OCTAVE,
        ))
        .insert_resource(audio_system::AudioBufferResource(ring_buffer))
        .insert_resource(analysis_system::AnalysisStateResource(
            pitchvis_analysis::analysis::AnalysisState::new(
                OCTAVES * BUCKETS_PER_OCTAVE,
                pitchvis_analysis::analysis::SPECTROGRAM_LENGTH,
            ),
        ));

    app.insert_resource(display_system::CylinderEntityListResource(Vec::new()))
        .add_startup_system(display_system::setup_display_to_system(
            OCTAVES,
            BUCKETS_PER_OCTAVE,
        ))
        .add_system(bevy::window::close_on_esc)
        .add_system(frame_limiter_system)
        .add_system(update_cqt_system)
        .add_system(update_analysis_state_system.after(update_cqt_system))
        .add_system(update_display_system.after(update_analysis_state_system));

        // MSAA makes some Android devices panic, this is under investigation
    // https://github.com/bevyengine/bevy/issues/8229
    #[cfg(target_os = "android")]
    app.insert_resource(Msaa::Off);

        app.run();
}
