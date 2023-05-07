use std::time::Duration;

use anyhow::Result;
use bevy::{app::ScheduleRunnerSettings, prelude::*};
mod analysis_system;
mod audio_system;
mod cqt_system;
mod serial_system;

pub const SR: usize = 22050;
pub const BUFSIZE: usize = 2 * SR;
pub const N_FFT: usize = 2 * 16384;
pub const FREQ_A1: f32 = 55.0;
pub const BUCKETS_PER_OCTAVE: usize = 12 * 3;
pub const OCTAVES: usize = 4;
pub const SPARSITY_QUANTILE: f32 = 0.999;
pub const Q: f32 = 1.0;
pub const GAMMA: f32 = 5.0;

const FPS: u64 = 10;

pub fn main() -> Result<()> {
    println!("Serial output format: 0xFF <num_triples (16 bit)> <r1> <g1> <b1> <r2> <g2> <b2> ...");
    // take command line arguments, e. g. `pitchvis_serial /dev/ttyUSB0 9600`
    let args = std::env::args().collect::<Vec<_>>();
    let path = args
        .get(1)
        .unwrap_or(&String::from("/dev/ttyUSB0"))
        .to_owned();
    let baud_rate = args.get(2).map_or(115_200, |s| s.parse::<u32>().unwrap());

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

    let serial_port_resource = serial_system::SerialPortResource::new(&path, baud_rate);

    audio_stream.play().unwrap();

    let update_cqt_system = cqt_system::update_cqt_to_system(BUFSIZE);
    let update_analysis_state_system =
        analysis_system::update_analysis_state_to_system(OCTAVES, BUCKETS_PER_OCTAVE);
    let update_serial_system = serial_system::update_serial_to_system(BUCKETS_PER_OCTAVE);

    App::new()
        .insert_resource(ScheduleRunnerSettings::run_loop(Duration::from_secs_f32(
            1.0 / FPS as f32,
        )))
        .add_plugins(MinimalPlugins)
        .insert_resource(serial_port_resource)
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
        .add_system(update_cqt_system)
        .add_system(update_analysis_state_system.after(update_cqt_system))
        .add_system(update_serial_system.after(update_analysis_state_system))
        .run();

    Ok(())
}
