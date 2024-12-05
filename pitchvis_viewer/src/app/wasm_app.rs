use anyhow::Result;
use bevy::asset::AssetMetaCheck;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::sprite::Material2dPlugin;
use bevy::window::PresentMode;
use bevy::winit::{UpdateMode, WinitSettings};
use pitchvis_analysis::analysis::{AnalysisParameters, AnalysisState};
use wasm_bindgen::prelude::*;

use super::common::analysis_text_showhide;
use super::common::fps_counter_showhide;
use super::common::setup_analysis_text;
use super::common::setup_fps_counter;
use super::common::update_analysis_text_system;
use super::common::update_fps_text_system;
use super::common::user_input_system;
use super::common::ActiveTouches;
use super::common::SettingsState;
use super::common::{close_on_esc, setup_bloom_ui, update_bloom_settings};
use crate::analysis_system::{self, AnalysisStateResource};
use crate::audio_system::AudioBufferResource;
use crate::display_system::material::NoisyColorMaterial;
use crate::display_system::{self, CylinderEntityListResource, DisplayMode};
use crate::vqt_system::{self, VqtResource, VqtResultResource};
use pitchvis_analysis::vqt::{Vqt, VqtParameters};
use pitchvis_audio::AudioStream;

pub const BUFSIZE: usize = 2 * 16384;

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
    let vqt_parameters = VqtParameters::default();

    let audio_stream = pitchvis_audio::async_new_audio_stream(vqt_parameters.sr as u32, BUFSIZE)
        .await
        .unwrap();

    let vqt = Vqt::new(&vqt_parameters);

    audio_stream.play().unwrap();

    let update_vqt_system = vqt_system::update_vqt_to_system(BUFSIZE);
    let update_analysis_state_system = analysis_system::update_analysis_state_to_system();
    let update_display_system = display_system::update_display_to_system(&vqt_parameters.range);

    App::new()
        .add_plugins((
            DefaultPlugins
                .set(AssetPlugin {
                    // needed for the Progressive Web App not to fail looking for a nonexisting
                    // .wasm.meta file when offline
                    meta_check: AssetMetaCheck::Never,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "PitchVis".into(),
                        name: Some("bevy.app".into()),
                        resolution: (500., 300.).into(),
                        present_mode: PresentMode::AutoVsync,
                        // Tells Wasm to use the canvas with the id "pitchviscanvas" as the main window
                        canvas: Some("#pitchviscanvas".into()),
                        // Tells Wasm to resize the window according to the available canvas
                        fit_canvas_to_parent: true,
                        // Tells Wasm not to override default event handling, like F5, Ctrl+R etc.
                        prevent_default_event_handling: false,
                        ..default()
                    }),
                    ..default()
                }),
            LogDiagnosticsPlugin::default(),
            FrameTimeDiagnosticsPlugin::default(),
            Material2dPlugin::<NoisyColorMaterial>::default(),
        ))
        .insert_resource(VqtResource(vqt))
        .insert_resource(VqtResultResource::new(&vqt_parameters.range))
        .insert_resource(AudioBufferResource(audio_stream.ring_buffer.clone()))
        .insert_resource(AnalysisStateResource(AnalysisState::new(
            vqt_parameters.range.clone(),
            AnalysisParameters::default(),
        )))
        .insert_resource(CylinderEntityListResource(Vec::new()))
        .insert_resource(SettingsState {
            display_mode: DisplayMode::PitchnamesCalmness,
            fps_limit: None,
        })
        .insert_resource(ActiveTouches::default())
        .insert_resource(CurrentFpsLimit(None))
        // hacky way to limit FPS. Only works when the user is not moving the mouse.
        // And if we set the react_ arguments to false, the FPS limit is all wonky.
        .insert_resource(WinitSettings::game())
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
                update_vqt_system,
                update_analysis_state_system.after(update_vqt_system),
                user_input_system,
                update_fps_text_system.after(update_vqt_system),
                fps_counter_showhide,
                update_bloom_settings.after(update_analysis_state_system),
                update_analysis_text_system.after(update_analysis_state_system),
                analysis_text_showhide,
                set_frame_limiter_system,
                update_display_system.after(update_analysis_state_system),
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
