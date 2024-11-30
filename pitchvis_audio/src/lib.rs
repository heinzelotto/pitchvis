#[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
mod audio_desktop;
#[cfg(target_arch = "wasm32")]
mod audio_wasm;

use anyhow::Result;

// TODO: also put the android audio code in this module

#[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
pub use audio_desktop::new_audio_stream;
#[cfg(target_arch = "wasm32")]
pub use audio_wasm::async_new_audio_stream;

pub struct RingBuffer {
    pub buf: Vec<f32>,
    pub gain: f32,
    pub latency_ms: Option<f32>,
    pub chunk_size_ms: f32,
}

pub trait AudioStream {
    fn sr(&self) -> u32;
    fn ring_buffer(&self) -> std::sync::Arc<std::sync::Mutex<RingBuffer>>;
    fn play(&self) -> Result<()>;
}
