use anyhow::Result;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
mod analysis;
mod analysis_system;
mod audio;
mod audio_system;
mod cqt;
mod cqt_system;
mod display_system;
mod util;

// TODO: make program arguments
const SR: usize = 22050;
const BUFSIZE: usize = 2 * SR;
const N_FFT: usize = 2 * 16384;
const FREQ_A1: f32 = 55.0;
const BUCKETS_PER_OCTAVE: usize = 12 * 5;
const OCTAVES: usize = 6; // TODO: extend to 6
const SPARSITY_QUANTILE: f32 = 0.999;
const Q: f32 = 1.0;
const GAMMA: f32 = 5.0;

const FPS: u64 = 30;

#[wasm_bindgen]
#[cfg(target_arch = "wasm32")]
pub async fn main_fun() -> Result<(), JsValue> {
    console_log::init_with_level(log::Level::Warn).expect("logger init");

    console_error_panic_hook::set_once();

    let audio_stream = audio::AudioStream::async_new(SR, BUFSIZE).await.unwrap();

    let cqt = cqt::Cqt::new(
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

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(MaterialPlugin::<display_system::LineMaterial>::default())
        .insert_resource(cqt_system::CqtResource(cqt))
        .insert_resource(cqt_system::CqtResultResource::default())
        .insert_resource(audio_system::AudioBufferResource(
            audio_stream.ring_buffer.clone(),
        ))
        .insert_resource(analysis_system::AnalysisStateResource(
            analysis::AnalysisState::new(
                OCTAVES * BUCKETS_PER_OCTAVE,
                analysis::SPECTROGRAM_LENGTH,
            ),
        ))
        .add_startup_system(display_system::setup_display_to_system(
            OCTAVES,
            BUCKETS_PER_OCTAVE,
        ))
        .add_system(cqt_system::update_cqt)
        .add_system(analysis_system::update_analysis_state_to_system(
            OCTAVES,
            BUCKETS_PER_OCTAVE,
        ))
        .add_system(display_system::update_display)
        .run();

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn main_fun() -> Result<()> {
    env_logger::init();

    let audio_stream = audio::AudioStream::new(SR, BUFSIZE).unwrap();

    let cqt = cqt::Cqt::new(
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

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(MaterialPlugin::<display_system::LineMaterial>::default())
        .insert_resource(cqt_system::CqtResource(cqt))
        .insert_resource(cqt_system::CqtResultResource::new())
        .insert_resource(audio_system::AudioBufferResource(
            audio_stream.ring_buffer.clone(),
        ))
        .insert_resource(analysis_system::AnalysisStateResource(
            analysis::AnalysisState::new(
                OCTAVES * BUCKETS_PER_OCTAVE,
                analysis::SPECTROGRAM_LENGTH,
            ),
        ))
        .add_system(bevy::window::close_on_esc)
        .add_startup_system(display_system::setup_display_to_system(
            OCTAVES,
            BUCKETS_PER_OCTAVE,
        ))
        .add_system(cqt_system::update_cqt)
        .add_system(analysis_system::update_analysis_state_to_system(
            OCTAVES,
            BUCKETS_PER_OCTAVE,
        ))
        .add_system(display_system::update_display)
        .run();
    Ok(())
}
