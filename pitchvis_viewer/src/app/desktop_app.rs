use anyhow::Result;
use bevy::asset::AssetMetaCheck;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::prelude::*;
use bevy::sprite::Material2dPlugin;
use bevy::winit::UpdateMode;
use bevy::winit::WinitSettings;
use bevy_persistent::Persistent;
use bevy_persistent::StorageFormat;

use super::common::analysis_text_showhide;
use super::common::button_showhide;
use super::common::close_on_esc;
use super::common::fps_counter_showhide;
use super::common::set_frame_limiter_system;
use super::common::set_vqt_smoothing_system;
use super::common::setup_analysis_text;
use super::common::setup_bloom_ui;
use super::common::setup_buttons;
use super::common::setup_fps_counter;
use super::common::setup_screen_lock_indicator;
use super::common::update_analysis_text_system;
use super::common::update_bloom_settings;
use super::common::update_button_system;
use super::common::update_fps_text_system;
use super::common::update_screen_lock_indicator;
use super::common::user_input_system;
use super::common::ActiveTouches;
use super::common::CurrentFpsLimit;
use super::common::CurrentVQTSmoothingMode;
use super::common::ScreenLockState;
use super::common::SettingsState;
use crate::analysis_system;
use crate::audio_system;
use crate::display_system;
use crate::vqt_system;
use pitchvis_analysis::vqt::VqtParameters;
use pitchvis_audio::AudioStream;

const BUFSIZE: usize = 2 * 16384;
const DEFAULT_FPS: u32 = 60;

/// This enum is used to control the audio stream. It is sent to the audio thread via a channel. This allows us to control the audio stream from the bevy thread.
// enum AudioControl {
//     Play,
//     Pause,
// }

// #[derive(Resource)]
// struct AudioControlChannelResource(std::sync::mpsc::Sender<AudioControl>);

// desktop main function
pub fn main_fun() -> Result<()> {
    env_logger::init();

    // for config file
    let config_dir = dirs::config_dir().unwrap().join("pitchvis");

    let vqt_parameters: VqtParameters = VqtParameters::default();

    let audio_stream = pitchvis_audio::new_audio_stream(vqt_parameters.sr as u32, BUFSIZE).unwrap();

    let vqt = pitchvis_analysis::vqt::Vqt::new(&vqt_parameters);

    audio_stream.play().unwrap();

    let update_vqt_system = vqt_system::update_vqt_to_system(BUFSIZE);
    let update_analysis_state_system = analysis_system::update_analysis_state_to_system();
    #[cfg(feature = "ml")]
    let update_ml_system = ml_system::update_ml_to_system();
    let update_display_system = display_system::update_display_to_system(&vqt_parameters.range);

    #[cfg(feature = "ml")]
    let ml_model_resource = ml_system::MlModelResource(ml_system::MlModel::new("model.pt"));

    let mut persistent_settings_state = Persistent::<SettingsState>::builder()
        .name("settings")
        .format(StorageFormat::Toml)
        .path(config_dir.join("settings.toml"))
        .default(SettingsState {
            display_mode: display_system::DisplayMode::Normal,
            visuals_mode: display_system::VisualsMode::Full,
            fps_limit: Some(DEFAULT_FPS),
            vqt_smoothing_mode: display_system::VQTSmoothingMode::Default,
        })
        .revertible(true)
        .revert_to_default_on_deserialization_errors(true)
        .build()
        .expect("failed to initialize key bindings");
    // Always start in normal mode
    persistent_settings_state.display_mode = display_system::DisplayMode::Normal;

    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    // present_mode: bevy::window::PresentMode::Immediate,
                    title: "PitchVis".to_string(),
                    ..default()
                }),
                ..default()
            })
            .set(AssetPlugin {
                meta_check: AssetMetaCheck::Never,
                ..default()
            }),
        FrameTimeDiagnosticsPlugin::default(),
        Material2dPlugin::<display_system::material::NoisyColorMaterial>::default(),
    ))
    .insert_resource(vqt_system::VqtResource(vqt))
    .insert_resource(vqt_system::VqtResultResource::new(&vqt_parameters.range))
    .insert_resource(audio_system::AudioBufferResource(audio_stream.ring_buffer))
    .insert_resource(analysis_system::AnalysisStateResource(
        pitchvis_analysis::analysis::AnalysisState::new(
            vqt_parameters.range.clone(),
            pitchvis_analysis::analysis::AnalysisParameters::default(),
        ),
    ))
    .insert_resource(persistent_settings_state)
    .insert_resource(CurrentFpsLimit(Some(DEFAULT_FPS)))
    .insert_resource(CurrentVQTSmoothingMode(
        display_system::VQTSmoothingMode::Default,
    ))
    .insert_resource(WinitSettings {
        focused_mode: UpdateMode::reactive(std::time::Duration::from_secs_f32(
            1.0 / DEFAULT_FPS as f32,
        )),
        unfocused_mode: UpdateMode::reactive(std::time::Duration::from_secs_f32(
            1.0 / DEFAULT_FPS as f32,
        )),
    })
    .insert_resource(ScreenLockState(false))
    .insert_resource(ActiveTouches(std::sync::Arc::new(std::sync::Mutex::new(
        std::collections::HashMap::new(),
    ))));

    #[cfg(feature = "ml")]
    app.insert_resource(ml_model_resource);
    app.insert_resource(display_system::CylinderEntityListResource(Vec::new()))
        .add_systems(
            Startup,
            (
                display_system::setup_display_to_system(&vqt_parameters.range),
                setup_fps_counter,
                setup_buttons,
                setup_bloom_ui,
                setup_analysis_text,
                setup_screen_lock_indicator,
            ),
        )
        .add_systems(
            Update,
            (
                close_on_esc,
                update_vqt_system,
                update_analysis_state_system.after(update_vqt_system),
                update_fps_text_system.after(update_vqt_system),
                fps_counter_showhide,
                update_button_system,
                button_showhide,
                user_input_system.after(update_button_system),
                update_bloom_settings.after(update_analysis_state_system),
                update_analysis_text_system.after(update_analysis_state_system),
                analysis_text_showhide,
                update_screen_lock_indicator,
                set_frame_limiter_system,
                set_vqt_smoothing_system,
                update_display_system.after(update_analysis_state_system),
            ),
        );
    #[cfg(feature = "ml")]
    app.add_systems(
        Update,
        (
            update_ml_system.after(update_analysis_state_system),
            update_display_system.after(update_ml_system),
        ),
    );
    app.run();

    Ok(())
}

// android main function
