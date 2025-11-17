use anyhow::Result;
use bevy::prelude::*;
use bevy_persistent::StorageFormat;

use super::common::{
    build_common_app, register_common_update_systems, register_startup_systems, PlatformConfig,
    PlatformSystems,
};
use crate::analysis_system;
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

    // Desktop-specific: CPAL audio stream
    let audio_stream = pitchvis_audio::new_audio_stream(vqt_parameters.sr as u32, BUFSIZE).unwrap();
    let vqt = pitchvis_analysis::vqt::Vqt::new(&vqt_parameters);
    audio_stream.play().unwrap();

    // Create system closures with BUFSIZE
    let update_vqt_system = vqt_system::update_vqt_to_system(BUFSIZE);
    let update_analysis_state_system = analysis_system::update_analysis_state_to_system();
    let update_display_system = display_system::update_display_to_system(&vqt_parameters.range);

    #[cfg(feature = "ml")]
    let update_ml_system = crate::ml_system::update_ml_to_system();

    #[cfg(feature = "ml")]
    let ml_model_resource = crate::ml_system::MlModelResource(crate::ml_system::MlModel::new("model.pt"));

    // Platform-specific window configuration
    let window_config = Window {
        title: "PitchVis".to_string(),
        ..default()
    };

    // Build common app
    let mut app = build_common_app(
        vqt,
        &vqt_parameters,
        PlatformConfig {
            config_dir,
            storage_format: StorageFormat::Toml,
            audio_buffer: audio_stream.ring_buffer,
            window_config,
            additional_plugins: vec![],
        },
        DEFAULT_FPS,
    );

    // Insert ML resource if feature is enabled (desktop-specific)
    #[cfg(feature = "ml")]
    app.insert_resource(ml_model_resource);

    // Register common startup systems
    register_startup_systems(&mut app, &vqt_parameters.range);

    // Register common update systems (bloom disabled on desktop)
    register_common_update_systems(
        &mut app,
        PlatformSystems {
            update_vqt: update_vqt_system,
            update_analysis_state: update_analysis_state_system,
            update_display: update_display_system,
        },
        false, // bloom disabled on desktop
    );

    // Add ML systems if feature is enabled (desktop-specific)
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
