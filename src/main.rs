use anyhow::Result;

mod audio;
mod cqt;
mod display;

const SR: usize = 22050;
const N_FFT: usize = 8192;
const FREQ_A1: f32 = 55.0;
const SPARSITY_QUANTILE: f32 = 0.98;
const OCTAVES: usize = 5;
const BUCKETS_PER_OCTAVE: usize = 48;

const FPS: f32 = 50.0;

fn main() -> Result<()> {
    let audio_stream = audio::AudioStream::new(SR)?;
    let cqt = cqt::Cqt::new(
        SR,
        N_FFT,
        FREQ_A1,
        BUCKETS_PER_OCTAVE,
        OCTAVES,
        SPARSITY_QUANTILE,
    );
    let mut display = display::Display::new(OCTAVES, BUCKETS_PER_OCTAVE);

    audio_stream.play()?;

    loop {
        let t_loop = std::time::Instant::now();

        // let t_cqt = std::time::Instant::now();
        let mut x = vec![0.0_f32; N_FFT];
        x.copy_from_slice(&audio_stream.ring_buffer.lock().unwrap()[(SR - N_FFT)..]);

        let x_cqt = cqt.calculate_cqt_instant_in_db(&x);
        // println!("CQT calculated in {}ms", t_cqt.elapsed().as_millis());

        //let t_render = std::time::Instant::now();
        if !display.render(&x_cqt) {
            break;
        }
        // println!("rendered in {}ms", t_render.elapsed().as_millis());

        let loop_duration = t_loop.elapsed();
        std::thread::sleep(std::time::Duration::from_millis(25).saturating_sub(loop_duration));
    }

    Ok(())
}
