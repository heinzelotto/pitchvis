use anyhow::Result;
use bevy::asset::AssetMetaCheck;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::prelude::*;
use bevy::sprite::Material2dPlugin;

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
use pitchvis_audio::AudioStream;

const BUFSIZE: usize = 2 * 16384;

// Default FPS when FPS limiting is on. Else FPS is limited by screen refresh rate.
const FPS: u32 = 30;

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

    let VQT_PARAMETERS: VqtParameters = VqtParameters::default();

    let audio_stream = pitchvis_audio::new_audio_stream(VQT_PARAMETERS.sr as u32, BUFSIZE).unwrap();

    let vqt = pitchvis_analysis::vqt::Vqt::new(&VQT_PARAMETERS);

    audio_stream.play().unwrap();

    let update_vqt_system = vqt_system::update_vqt_to_system(BUFSIZE);
    let update_analysis_state_system = analysis_system::update_analysis_state_to_system();
    #[cfg(feature = "ml")]
    let update_ml_system = ml_system::update_ml_to_system();
    let update_display_system = display_system::update_display_to_system(&VQT_PARAMETERS.range);

    #[cfg(feature = "ml")]
    let ml_model_resource = ml_system::MlModelResource(ml_system::MlModel::new("model.pt"));

    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    // present_mode: PresentMode::Immediate,
                    title: "Pitchvis".to_string(),
                    ..default()
                }),
                ..default()
            })
            .set(AssetPlugin {
                meta_check: AssetMetaCheck::Never,
                ..default()
            }),
        FrameTimeDiagnosticsPlugin,
        Material2dPlugin::<display_system::material::NoisyColorMaterial>::default(),
    ))
    .insert_resource(vqt_system::VqtResource(vqt))
    .insert_resource(vqt_system::VqtResultResource::new(&VQT_PARAMETERS.range))
    .insert_resource(audio_system::AudioBufferResource(audio_stream.ring_buffer))
    .insert_resource(analysis_system::AnalysisStateResource(
        pitchvis_analysis::analysis::AnalysisState::new(
            VQT_PARAMETERS.range.clone(),
            pitchvis_analysis::analysis::AnalysisParameters::default(),
        ),
    ))
    .insert_resource(SettingsState {
        display_mode: display_system::DisplayMode::PitchnamesCalmness,
        fps_limit: Some(FPS),
    })
    .insert_resource(ActiveTouches::default());

    #[cfg(feature = "ml")]
    app.insert_resource(ml_model_resource);
    app.insert_resource(display_system::CylinderEntityListResource(Vec::new()))
        .add_systems(
            Startup,
            (
                display_system::setup_display_to_system(&VQT_PARAMETERS.range),
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
                frame_limiter_system,
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
