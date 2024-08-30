// #![no_std]
// #![no_main]

use std::time::Duration;

use pitchvis_analysis::{analysis::AnalysisState, util::*};
use serialport::SerialPort;

// increasing BUCKETS_PER_SEMITONE or Q will improve frequency resolution at cost of time resolution,
// increasing GAMMA will improve time resolution at lower frequencies.
pub const SR: usize = 22050;
pub const BUFSIZE: usize = 2 * SR;
pub const N_FFT: usize = 2 * 16384;
pub const FREQ_A1: f32 = 55.0;
pub const BUCKETS_PER_SEMITONE: usize = 3;
pub const BUCKETS_PER_OCTAVE: usize = 12 * BUCKETS_PER_SEMITONE;
pub const OCTAVES: usize = 5;
pub const SPARSITY_QUANTILE: f32 = 0.999;
pub const Q: f32 = 10.0;
pub const GAMMA: f32 = 5.3 * Q;

const FPS: u64 = 25;

// color calculation constants
pub const COLORS: [[f32; 3]; 12] = [
    [0.95, 0.10, 0.10], // C
    [0.01, 0.52, 0.71], // C#
    [0.97, 0.79, 0.00], // D
    [0.45, 0.34, 0.63], // Eb
    [0.47, 0.99, 0.02], // E
    [0.88, 0.02, 0.52], // F
    [0.00, 0.80, 0.55], // F#
    [0.99, 0.54, 0.03], // G
    [0.25, 0.30, 0.64], // Ab
    [0.95, 0.99, 0.00], // A
    [0.52, 0.00, 0.60], // Bb
    [0.05, 0.80, 0.15], // H
];
const GRAY_LEVEL: f32 = 5.0;
const EASING_POW: f32 = 2.3;

struct VqtResult {
    pub x_vqt: Vec<f32>,
    pub gain: f32,
}

impl VqtResult {
    pub fn new(octaves: usize, buckets_per_octave: usize) -> Self {
        Self {
            x_vqt: vec![0.0; octaves * buckets_per_octave],
            gain: 1.0,
        }
    }
}

fn update_serial(
    buckets_per_octave: usize,
    analysis_state: &AnalysisState,
    serial_port: &mut dyn SerialPort,
) {
    let k_max = arg_max(&analysis_state.x_vqt_peakfiltered);
    let max_size = analysis_state.x_vqt_peakfiltered[k_max];

    // special value to indicate begin of data
    let mut output: Vec<u8> = vec![0xFF];
    // 16 bit number of RGB triples to follow
    let num_triples: u16 = analysis_state.x_vqt_peakfiltered.len().try_into().unwrap();
    output.push((num_triples / 256) as u8);
    output.push((num_triples % 256) as u8);
    output.extend(
        analysis_state
            .x_vqt_peakfiltered
            .iter()
            .enumerate()
            .flat_map(|(idx, size)| {
                let (mut r, mut g, mut b) = pitchvis_analysis::color_mapping::calculate_color(
                    buckets_per_octave,
                    ((idx + (buckets_per_octave - 3 * (buckets_per_octave / 12))) as f32)
                        % buckets_per_octave as f32,
                    COLORS,
                    GRAY_LEVEL,
                    EASING_POW,
                );

                let color_coefficient = 1.0 - (1.0 - size / max_size).powf(0.18);
                r *= color_coefficient;
                g *= color_coefficient;
                b *= color_coefficient;

                [(r * 254.0) as u8, (g * 254.0) as u8, (b * 254.0) as u8]
            }),
    );
    println!("output: {:02x?}", &output);

    serial_port
        .write_all(output.as_slice())
        .expect("Write failed!");
    serial_port.flush().expect("Flush failed!");
}

pub fn main() {
    println!("Serial output format: 0xFF <num_triples (16 bit)> <r1> <g1> <b1> <r2> <g2> <b2> ...");
    // take command line arguments, e. g. `pitchvis_serial /dev/ttyUSB0 9600`
    let args = std::env::args().collect::<Vec<_>>();
    let path = args
        .get(1)
        .unwrap_or(&String::from("/dev/ttyUSB0"))
        .to_owned();
    let baud_rate = args.get(2).map_or(115_200, |s| s.parse::<u32>().unwrap());
    let mut serial_port = serialport::new(path, baud_rate)
        .timeout(std::time::Duration::from_secs(10)) // TODO: ???
        .open()
        .expect("Failed to open port");
    let audio_stream = pitchvis_audio::audio::AudioStream::new(SR, BUFSIZE).unwrap();
    let mut vqt = pitchvis_analysis::vqt::Vqt::new(
        SR,
        N_FFT,
        FREQ_A1,
        BUCKETS_PER_OCTAVE,
        OCTAVES,
        SPARSITY_QUANTILE,
        Q,
        GAMMA,
    );
    let mut vqt_result = VqtResult::new(OCTAVES, BUCKETS_PER_OCTAVE);
    let mut analysis_state = AnalysisState::new(
        OCTAVES * BUCKETS_PER_OCTAVE,
        pitchvis_analysis::analysis::SPECTROGRAM_LENGTH,
    );

    audio_stream.play().unwrap();

    loop {
        let start_time = std::time::Instant::now();

        let (x, gain) = {
            let mut x = vec![0.0_f32; vqt.n_fft];
            let rb = audio_stream.ring_buffer.lock().unwrap();
            x.copy_from_slice(&rb.buf[(BUFSIZE - vqt.n_fft)..]);
            (x, rb.gain)
        };
        vqt_result.x_vqt = vqt.calculate_vqt_instant_in_db(&x);
        vqt_result.gain = gain;
        analysis_state.preprocess(&vqt_result.x_vqt, OCTAVES, BUCKETS_PER_OCTAVE);
        update_serial(BUCKETS_PER_OCTAVE, &analysis_state, serial_port.as_mut());

        let elapsed = start_time.elapsed();
        let sleep_time = Duration::from_millis(1000 / FPS).saturating_sub(elapsed);
        std::thread::sleep(sleep_time);
    }
}
