use anyhow::Result;
use kiss3d::camera::*;
use kiss3d::planar_camera::PlanarCamera;
use kiss3d::post_processing::*;
use kiss3d::renderer::*;
use kiss3d::window::{State, Window};
use wasm_bindgen::prelude::*;

mod audio;
mod cqt;
mod display;

// TODO: make program arguments
const SR: usize = 22050;
const BUFSIZE: usize = 2 * SR;
const N_FFT: usize = 2 * 16384;
const FREQ_A1: f32 = 55.0;
const BUCKETS_PER_OCTAVE: usize = 48;
const OCTAVES: usize = 6; // TODO: extend to 6
const SPARSITY_QUANTILE: f32 = 0.999;
const Q: f32 = 1.4;
const GAMMA: f32 = 5.0;

const _FPS: f32 = 50.0;

struct AppState {
    audio_stream: audio::AudioStream,
    cqt: cqt::Cqt,
    display: display::Display,
}

impl State for AppState {
    fn step(&mut self, window: &mut Window) {
        let wasm_window = web_sys::window().expect("has window");
        let performance = wasm_window.performance().expect("performance available");

        let t_loop = performance.now();

        //let t_cqt = std::time::Instant::now();
        let mut x = vec![0.0_f32; N_FFT];
        x.copy_from_slice(&self.audio_stream.ring_buffer.lock().unwrap()[(BUFSIZE - N_FFT)..]);

        let x_cqt = self.cqt.calculate_cqt_instant_in_db(&x);
        // println!("CQT calculated in {}ms", t_cqt.elapsed().as_millis());

        //let t_render = std::time::Instant::now();
        self.display.render(window, &x_cqt);
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

    let mut window = Window::new("pitchvis: music visualization");
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
