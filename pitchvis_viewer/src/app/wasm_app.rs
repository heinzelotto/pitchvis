use super::close_on_esc;
use super::fps_counter_showhide;
use super::fps_text_update_system;
use super::setup_fps_counter;
use super::user_input_system;
use crate::analysis_system;
use crate::audio_system;
use crate::display_system;
use crate::vqt_system;
use anyhow::Result;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::sprite::Material2dPlugin;
use wasm_bindgen::prelude::*;

// increasing BUCKETS_PER_SEMITONE or Q will improve frequency resolution at cost of time resolution,
// increasing GAMMA will improve time resolution at lower frequencies.
pub const SR: usize = 22050;
pub const BUFSIZE: usize = 2 * SR;
pub const N_FFT: usize = 2 * 16384;
pub const FREQ_A1: f32 = 55.0;
#[cfg(feature = "ml")]
pub const FREQ_A1_MIDI_KEY_ID: i32 = 33;
pub const UPSCALE_FACTOR: usize = 1;
pub const BUCKETS_PER_SEMITONE: usize = 3 * UPSCALE_FACTOR;
pub const BUCKETS_PER_OCTAVE: usize = 12 * BUCKETS_PER_SEMITONE;
pub const OCTAVES: usize = 7;
pub const SPARSITY_QUANTILE: f32 = 0.999;
pub const Q: f32 = 10.0 / UPSCALE_FACTOR as f32;
pub const GAMMA: f32 = 5.3 * Q;

// TODO: on wasm it's currently difficult to limit FPS without forking bevy.
// Make the animation speed/blurring windows/... independent of the frame rate
const FPS: u64 = 30;

// wasm main function
#[wasm_bindgen]
pub async fn main_fun() -> Result<(), JsValue> {
    let audio_stream = pitchvis_audio::audio::AudioStream::async_new(SR, BUFSIZE)
        .await
        .unwrap();

    let vqt = pitchvis_analysis::vqt::Vqt::new(
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

    let update_vqt_system = vqt_system::update_vqt_to_system(BUFSIZE);
    let update_analysis_state_system =
        analysis_system::update_analysis_state_to_system(OCTAVES, BUCKETS_PER_OCTAVE);
    let update_display_system =
        display_system::update_display_to_system(BUCKETS_PER_OCTAVE, OCTAVES);

    App::new()
        .add_plugins((
            DefaultPlugins,
            LogDiagnosticsPlugin::default(),
            FrameTimeDiagnosticsPlugin::default(),
            Material2dPlugin::<display_system::material::NoisyColorMaterial>::default(),
        ))
        .insert_resource(vqt_system::VqtResource(vqt))
        .insert_resource(vqt_system::VqtResultResource::new(
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
        .insert_resource(display_system::CylinderEntityListResource(Vec::new()))
        .insert_resource(display_system::SettingsState {
            display_mode: display_system::DisplayMode::PitchnamesCalmness,
        })
        // hacky way to limit FPS. Don't know why we have to target 5x the FPS
        .insert_resource(bevy::winit::WinitSettings {
            focused_mode: bevy::winit::UpdateMode::Reactive {
                wait: std::time::Duration::from_millis((1000 / 5) / FPS),
                react_to_device_events: false,
                react_to_user_events: false,
                react_to_window_events: false,
            },
            unfocused_mode: bevy::winit::UpdateMode::Reactive {
                wait: std::time::Duration::from_millis((1000 / 5) / FPS),
                react_to_device_events: false,
                react_to_user_events: false,
                react_to_window_events: false,
            },
        })
        .add_systems(
            Startup,
            (
                display_system::setup_display_to_system(OCTAVES, BUCKETS_PER_OCTAVE),
                setup_fps_counter,
            ),
        )
        .add_systems(
            Update,
            (
                close_on_esc,
                // frame_limiter_system, // FIXME: this is not working on wasm
                update_vqt_system,
                update_analysis_state_system.after(update_vqt_system),
                update_display_system.after(update_analysis_state_system),
                user_input_system,
                fps_text_update_system,
                fps_counter_showhide,
            ),
        )
        .run();

    Ok(())
}

// This enum is used to control the audio stream. It is sent to the audio thread via a channel. This allows us to control the audio stream from the bevy thread.
// enum AudioControl {
//     Play,
//     Pause,
// }

// #[derive(Resource)]
// struct AudioControlChannelResource(std::sync::mpsc::Sender<AudioControl>);

// android main function
