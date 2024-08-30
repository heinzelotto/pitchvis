use anyhow::Result;
use kiss3d::camera::*;
use kiss3d::planar_camera::PlanarCamera;
use kiss3d::post_processing::*;
use kiss3d::renderer::*;
use kiss3d::window::{State, Window};
use wasm_bindgen::prelude::*;

mod display;

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

struct AppState {
    audio_stream: pitchvis_audio::audio::AudioStream,
    vqt: pitchvis_analysis::vqt::Vqt,
    display: display::Display,
}

impl State for AppState {
    fn step(&mut self, window: &mut Window) {
        let wasm_window = web_sys::window().expect("has window");
        let performance = wasm_window.performance().expect("performance available");

        let t_loop = performance.now();

        //let t_vqt = std::time::Instant::now();
        let (x, gain) = {
            let mut x = vec![0.0_f32; N_FFT];
            let rb = &self.audio_stream.ring_buffer.lock().unwrap();
            x.copy_from_slice(&rb.buf[(BUFSIZE - N_FFT)..]);
            (x, rb.gain)
        };
        let x_vqt = self.vqt.calculate_vqt_instant_in_db(&x);
        // println!("VQT calculated in {}ms", t_vqt.elapsed().as_millis());

        //let t_render = std::time::Instant::now();
        self.display.render(window, &x_vqt, gain);
        // println!("rendered in {}ms", t_render.elapsed().as_millis());

        let time_passed = performance.now() - t_loop;
        assert!(time_passed >= 0.0);
        let loop_duration = std::time::Duration::from_millis(time_passed as u64);
        // TODO:
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

#[wasm_bindgen]
pub async fn main_fun() -> Result<(), JsValue> {
    console_log::init_with_level(log::Level::Debug).expect("logger init");

    console_error_panic_hook::set_once();

    let audio_stream = pitchvis_audio::audio::AudioStream::async_new(SR, BUFSIZE)
        .await
        .unwrap();

    let vqt = pitchvis_analysis::vqt::Vqt::new(
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

    let mut window = Window::new("pitchvis: music visualization");
    let display = display::Display::new(&mut window, OCTAVES, BUCKETS_PER_OCTAVE);

    audio_stream.play().unwrap();

    let app_state = AppState {
        audio_stream,
        vqt,
        display,
    };

    window.render_loop(app_state);

    Ok(())
}
