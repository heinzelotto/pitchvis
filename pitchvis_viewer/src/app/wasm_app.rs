use bevy::diagnostic::LogDiagnosticsPlugin;
use bevy::prelude::*;
use bevy::window::PresentMode;
use bevy_persistent::StorageFormat;
use std::path::PathBuf;
use wasm_bindgen::prelude::*;

use super::common::{
    build_common_app, register_common_update_systems, register_startup_systems, PlatformConfig,
};
use crate::analysis_system;
use crate::display_system;
use crate::vqt_system;
use pitchvis_analysis::vqt::VqtParameters;
use pitchvis_audio::AudioStream;

pub const BUFSIZE: usize = 2 * 16384;
const DEFAULT_FPS: u32 = 60;

// wasm main function
#[wasm_bindgen]
pub async fn main_fun() -> Result<(), JsValue> {
    let config_dir = PathBuf::from("local").join("configuration");

    let vqt_parameters = VqtParameters::default();

    // WASM-specific: Async audio stream creation
    let audio_stream = pitchvis_audio::async_new_audio_stream(vqt_parameters.sr as u32, BUFSIZE)
        .await
        .unwrap();

    let vqt = pitchvis_analysis::vqt::Vqt::new(&vqt_parameters);

    audio_stream.play().unwrap();

    // Create system closures with BUFSIZE
    let update_vqt_system = vqt_system::update_vqt_to_system(BUFSIZE);
    let update_analysis_state_system = analysis_system::update_analysis_state_to_system();
    let update_display_system = display_system::update_display_to_system(&vqt_parameters.range);

    // WASM-specific: Window configuration with canvas binding
    let window_config = Window {
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
    };

    // Build common app with WASM-specific config (JSON storage format)
    let mut app = build_common_app(
        vqt,
        &vqt_parameters,
        PlatformConfig {
            config_dir,
            storage_format: StorageFormat::Json,
            audio_buffer: audio_stream.ring_buffer.clone(),
            window_config,
            additional_plugins: vec![Box::new(LogDiagnosticsPlugin::default())],
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

    app.run();

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
