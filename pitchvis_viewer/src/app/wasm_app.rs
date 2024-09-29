use anyhow::Result;
use bevy::asset::AssetMetaCheck;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::sprite::Material2dPlugin;
use bevy::winit::{UpdateMode, WinitSettings};
use pitchvis_analysis::analysis::{AnalysisParameters, AnalysisState};
use wasm_bindgen::prelude::*;

use super::common::fps_counter_showhide;
use super::common::fps_text_update_system;
use super::common::setup_fps_counter;
use super::common::user_input_system;
use super::common::SettingsState;
use super::common::{close_on_esc, setup_bloom_ui, update_bloom_settings};
use crate::analysis_system::{self, AnalysisStateResource};
use crate::audio_system::AudioBufferResource;
use crate::display_system::material::NoisyColorMaterial;
use crate::display_system::{self, CylinderEntityListResource, DisplayMode};
use crate::vqt_system::{self, VqtResource, VqtResultResource};
use pitchvis_analysis::vqt::VqtRange;
use pitchvis_analysis::vqt::{Vqt, VqtParameters};
use pitchvis_audio::AudioStream;

// increasing BUCKETS_PER_SEMITONE or Q will improve frequency resolution at cost of time resolution,
// increasing GAMMA will improve time resolution at lower frequencies.
pub const SR: u32 = 22050;
pub const BUFSIZE: usize = 2 * SR as usize;
pub const N_FFT: usize = 2 * 16384;
pub const FREQ_A1: f32 = 55.0;
#[cfg(feature = "ml")]
pub const FREQ_A1_MIDI_KEY_ID: i32 = 33;
pub const UPSCALE_FACTOR: u16 = 1;
pub const BUCKETS_PER_SEMITONE: u16 = 3 * UPSCALE_FACTOR;
pub const BUCKETS_PER_OCTAVE: u16 = 12 * BUCKETS_PER_SEMITONE;
pub const OCTAVES: u8 = 7;
pub const SPARSITY_QUANTILE: f32 = 0.999;
pub const Q: f32 = 10.0 / UPSCALE_FACTOR as f32;
pub const GAMMA: f32 = 5.3 * Q;

const VQT_PARAMETERS: VqtParameters = VqtParameters {
    sr: SR as f32,
    n_fft: N_FFT,
    range: VqtRange {
        min_freq: FREQ_A1,
        buckets_per_octave: BUCKETS_PER_OCTAVE,
        octaves: OCTAVES,
    },
    sparsity_quantile: SPARSITY_QUANTILE,
    quality: Q,
    gamma: GAMMA,
};

#[derive(Resource)]
struct CurrentFpsLimit(pub Option<u32>);

fn set_frame_limiter_system(
    mut current_limit: ResMut<CurrentFpsLimit>,
    mut winit_settings: ResMut<WinitSettings>,
    settings: Res<SettingsState>,
) {
    if settings.fps_limit != current_limit.0 {
        current_limit.0 = settings.fps_limit;
        *winit_settings = match settings.fps_limit {
            Some(fps) => WinitSettings {
                focused_mode: UpdateMode::Reactive {
                    wait: std::time::Duration::from_millis(1000 / fps as u64),
                    react_to_device_events: true,
                    react_to_user_events: true,
                    react_to_window_events: true,
                },
                unfocused_mode: UpdateMode::Reactive {
                    wait: std::time::Duration::from_millis(1000 / fps as u64),
                    react_to_device_events: true,
                    react_to_user_events: true,
                    react_to_window_events: true,
                },
            },
            None => WinitSettings::game(),
        };
    }
}

// wasm main function
#[wasm_bindgen]
pub async fn main_fun() -> Result<(), JsValue> {
    let audio_stream = pitchvis_audio::async_new_audio_stream(SR, BUFSIZE)
        .await
        .unwrap();

    let vqt = Vqt::new(&VQT_PARAMETERS);

    audio_stream.play().unwrap();

    let update_vqt_system = vqt_system::update_vqt_to_system(BUFSIZE);
    let update_analysis_state_system = analysis_system::update_analysis_state_to_system();
    let update_display_system = display_system::update_display_to_system(&VQT_PARAMETERS.range);

    App::new()
        .add_plugins((
            DefaultPlugins.set(AssetPlugin {
                // needed for the Progressive Web App not to fail looking for a nonexisting
                // .wasm.meta file when offline
                meta_check: AssetMetaCheck::Never,
                ..default()
            }),
            LogDiagnosticsPlugin::default(),
            FrameTimeDiagnosticsPlugin::default(),
            Material2dPlugin::<NoisyColorMaterial>::default(),
        ))
        .insert_resource(VqtResource(vqt))
        .insert_resource(VqtResultResource::new(&VQT_PARAMETERS.range))
        .insert_resource(AudioBufferResource(audio_stream.ring_buffer.clone()))
        .insert_resource(AnalysisStateResource(AnalysisState::new(
            VQT_PARAMETERS.range.clone(),
            AnalysisParameters::default(),
        )))
        .insert_resource(CylinderEntityListResource(Vec::new()))
        .insert_resource(SettingsState {
            display_mode: DisplayMode::PitchnamesCalmness,
            fps_limit: None,
        })
        .insert_resource(CurrentFpsLimit(None))
        // hacky way to limit FPS. Only works when the user is not moving the mouse.
        // And if we set the react_ arguments to false, the FPS limit is all wonky.
        .insert_resource(WinitSettings::game())
        .add_systems(
            Startup,
            (
                display_system::setup_display_to_system(&VQT_PARAMETERS.range),
                setup_fps_counter,
                setup_bloom_ui,
            ),
        )
        .add_systems(
            Update,
            (
                close_on_esc,
                update_vqt_system,
                update_analysis_state_system
                    .clone()
                    .after(update_vqt_system),
                update_display_system.after(update_analysis_state_system),
                user_input_system,
                fps_text_update_system,
                fps_counter_showhide,
                update_bloom_settings,
                set_frame_limiter_system,
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
