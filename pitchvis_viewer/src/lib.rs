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
mod util;

// increasing BUCKETS_PER_SEMITONE or Q will improve frequency resolution at cost of time resolution,
// increasing GAMMA will improve time resolution at lower frequencies.
pub const SR: usize = 22050;
pub const BUFSIZE: usize = 2 * SR;
pub const N_FFT: usize = 2 * 16384;
pub const FREQ_A1: f32 = 55.0;
pub const UPSCALE_FACTOR: usize = 1;
pub const BUCKETS_PER_SEMITONE: usize = 5 * UPSCALE_FACTOR;
pub const BUCKETS_PER_OCTAVE: usize = 12 * BUCKETS_PER_SEMITONE;
pub const OCTAVES: usize = 7;
pub const SPARSITY_QUANTILE: f32 = 0.999;
pub const Q: f32 = 6.0 / UPSCALE_FACTOR as f32;
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
    let update_display_system =
        display_system::update_display_to_system(BUCKETS_PER_OCTAVE, OCTAVES);

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(MaterialPlugin::<display_system::LineMaterial>::default())
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
        ))
        .insert_resource(display_system::CylinderEntityListResource(Vec::new()))
        .add_startup_system(display_system::setup_display_to_system(
            OCTAVES,
            BUCKETS_PER_OCTAVE,
        ))
        .add_system(bevy::window::close_on_esc)
        .add_system(frame_limiter_system)
        .add_system(update_cqt_system)
        .add_system(update_analysis_state_system.after(update_cqt_system))
        .add_system(update_display_system.after(update_analysis_state_system))
        .run();
    Ok(())
}
