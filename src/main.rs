#![feature(is_sorted)]

use anyhow::Result;
use kiss3d::camera::*;
use kiss3d::planar_camera::PlanarCamera;
use kiss3d::post_processing::*;
use kiss3d::renderer::*;
use kiss3d::window::{State, Window};

mod audio;
mod cqt;
mod display;

// TODO: make program arguments
const SR: usize = 22050;
const BUFSIZE: usize = 2 * SR;
const N_FFT: usize = 2 * 16384;
const FREQ_A1: f32 = 55.0;
const BUCKETS_PER_OCTAVE: usize = 12 * 4;
const OCTAVES: usize = 6; // TODO: extend to 6
const SPARSITY_QUANTILE: f32 = 0.999;
const Q: f32 = 1.4;
const GAMMA: f32 = 5.0;

const FPS: u64 = 30;

struct AppState {
    audio_stream: audio::AudioStream,
    cqt: cqt::Cqt,
    display: display::Display,
}

impl State for AppState {
    fn step(&mut self, window: &mut Window) {
        // let t_loop = std::time::Instant::now();

        //let t_cqt = std::time::Instant::now();
        let (x, gain) = {
            let mut x = vec![0.0_f32; N_FFT];
            let rb = &self.audio_stream.ring_buffer.lock().unwrap();
            x.copy_from_slice(&rb.buf[(BUFSIZE - N_FFT)..]);
            (x, rb.gain)
        };

        let x_cqt = self.cqt.calculate_cqt_instant_in_db(&x);
        // println!("CQT calculated in {}ms", t_cqt.elapsed().as_millis());

        //let t_render = std::time::Instant::now();
        self.display.render(window, &x_cqt, gain);
        // println!("rendered in {}ms", t_render.elapsed().as_millis());

        // let loop_duration = t_loop.elapsed();
        // std::thread::sleep(std::time::Duration::from_millis(30).saturating_sub(loop_duration));
    }

    fn cameras_and_effect_and_renderer(
        &mut self,
    ) -> (
        Option<&mut dyn Camera>,
        Option<&mut dyn PlanarCamera>,
        Option<&mut dyn Renderer>,
        Option<&mut dyn PostProcessingEffect>,
    ) {
        (Some(&mut self.display.cam), None, None, None)
    }
}

pub fn main() -> Result<()> {
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

    let mut window = Window::new("Kiss3d: cube");
    window.set_framerate_limit(Some(FPS));
    let display = display::Display::new(&mut window, OCTAVES, BUCKETS_PER_OCTAVE);

    audio_stream.play().unwrap();

    let app_state = AppState {
        audio_stream,
        cqt,
        display,
    };

    window.render_loop(app_state);

    Ok(())
}
