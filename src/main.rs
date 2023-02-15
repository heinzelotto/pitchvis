use anyhow::Result;

mod audio;
mod cqt;
mod display;

// TODO: make program arguments
const SR: usize = 22050;
const BUFSIZE: usize = 2 * SR;
const N_FFT: usize = 1 * 16384;
const FREQ_A1: f32 = 55.0;
const BUCKETS_PER_OCTAVE: usize = 12;
const OCTAVES: usize = 6; // TODO: extend to 6
const SPARSITY_QUANTILE: f32 = 0.999;
const Q: f32 = 2.2; 
const GAMMA: f32 = 0.0;

const FPS: f32 = 50.0;

fn main() -> Result<()> {
    let audio_stream = audio::AudioStream::new(SR, BUFSIZE)?;
    let mut cqt = cqt::Cqt::new(
        SR,
        N_FFT,
        FREQ_A1,
        BUCKETS_PER_OCTAVE,
        OCTAVES,
        SPARSITY_QUANTILE,
        Q,
        GAMMA,
    );
    let mut display = display::Display::new(OCTAVES, BUCKETS_PER_OCTAVE);

    audio_stream.play()?;

    loop {
        let t_loop = std::time::Instant::now();

        let t_cqt = std::time::Instant::now();
        let mut x = vec![0.0_f32; N_FFT];
        x.copy_from_slice(&audio_stream.ring_buffer.lock().unwrap()[(BUFSIZE - N_FFT)..]);

        let x_cqt = cqt.calculate_cqt_instant_in_db(&x);
        // println!("CQT calculated in {}ms", t_cqt.elapsed().as_millis());

        let t_render = std::time::Instant::now();
        if !display.render(&x_cqt) {
            break;
        }
        // println!("rendered in {}ms", t_render.elapsed().as_millis());

        let loop_duration = t_loop.elapsed();
        std::thread::sleep(std::time::Duration::from_millis(30).saturating_sub(loop_duration));
    }

    Ok(())
}
