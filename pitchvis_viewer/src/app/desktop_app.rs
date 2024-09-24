use anyhow::Result;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::sprite::Material2dPlugin;

use super::close_on_esc;
use super::fps_counter_showhide;
use super::fps_text_update_system;
use super::frame_limiter_to_system;
use super::setup_bloom_ui;
use super::setup_fps_counter;
use super::update_bloom_settings;
use super::user_input_system;
use crate::analysis_system;
use crate::audio_system;
use crate::display_system;
use crate::vqt_system;

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

const FPS: u64 = 30;
// TODO: make the animation speed/blurring windows/... independent of the frame rate

/// This enum is used to control the audio stream. It is sent to the audio thread via a channel. This allows us to control the audio stream from the bevy thread.
// enum AudioControl {
//     Play,
//     Pause,
// }

// #[derive(Resource)]
// struct AudioControlChannelResource(std::sync::mpsc::Sender<AudioControl>);

// desktop main function
pub fn main_fun() -> Result<()> {
    // env_logger::init();

    let audio_stream = pitchvis_audio::audio::AudioStream::new(SR, BUFSIZE).unwrap();

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
    #[cfg(feature = "ml")]
    let update_ml_system = ml_system::update_ml_to_system();
    let update_display_system =
        display_system::update_display_to_system(BUCKETS_PER_OCTAVE, OCTAVES);

    #[cfg(feature = "ml")]
    let ml_model_resource = ml_system::MlModelResource(ml_system::MlModel::new("model.pt"));

    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins.set(WindowPlugin {
            // primary_window: Some(Window {
            //     present_mode: PresentMode::AutoNoVsync,
            //     ..default()
            // }),
            primary_window: Some(Window {
                title: "Pitchvis".to_string(),
                ..default()
            }),
            ..default()
        }),
        LogDiagnosticsPlugin::default(),
        FrameTimeDiagnosticsPlugin,
        //.add_plugin(MaterialPlugin::<display_system::LineMaterial>::default())
        Material2dPlugin::<display_system::material::NoisyColorMaterial>::default(),
    ))
    .insert_resource(vqt_system::VqtResource(vqt))
    .insert_resource(vqt_system::VqtResultResource::new(
        OCTAVES,
        BUCKETS_PER_OCTAVE,
    ))
    .insert_resource(audio_system::AudioBufferResource(audio_stream.ring_buffer))
    .insert_resource(analysis_system::AnalysisStateResource(
        pitchvis_analysis::analysis::AnalysisState::new(
            OCTAVES * BUCKETS_PER_OCTAVE,
            pitchvis_analysis::analysis::SPECTROGRAM_LENGTH,
        ),
    ))
    .insert_resource(display_system::SettingsState {
        display_mode: display_system::DisplayMode::PitchnamesCalmness,
    });

    #[cfg(feature = "ml")]
    app.insert_resource(ml_model_resource);
    app.insert_resource(display_system::CylinderEntityListResource(Vec::new()))
        .add_systems(
            Startup,
            (
                display_system::setup_display_to_system(OCTAVES, BUCKETS_PER_OCTAVE),
                setup_fps_counter,
                setup_bloom_ui,
            ),
        )
        .add_systems(
            Update,
            (
                close_on_esc,
                frame_limiter_to_system(FPS),
                update_vqt_system,
                user_input_system,
                fps_text_update_system,
                fps_counter_showhide,
                update_analysis_state_system.after(update_vqt_system),
                update_bloom_settings,
            ),
        );
    #[cfg(feature = "ml")]
    app.add_systems(
        Update,
        (
            update_ml_system.after(update_analysis_state_system),
            update_display_system.after(update_ml_system),
        ),
    )
    .run();
    #[cfg(not(feature = "ml"))]
    app.add_systems(
        Update,
        update_display_system.after(update_analysis_state_system),
    )
    .run();

    Ok(())
}

// android main function
