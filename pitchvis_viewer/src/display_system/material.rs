//! A shader material for rendering the pitch balls, with a parameter to control the noise level.

use bevy::{
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
    sprite_render::{AlphaMode2d, Material2d},
};

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct NoisyColorMaterial {
    #[uniform(0)]
    pub color: LinearRgba,
    #[uniform(1)]
    pub params: Params, // Padding to ensure 16-Byte alignment
}

#[derive(Debug, Clone, Copy, Default, PartialEq, ShaderType)]
#[repr(C)]
pub struct Params {
    /// Calmness level [0.0, 1.0] - controls ring noise/fullness
    pub calmness: f32,
    /// Global time in seconds - for noise animation
    pub time: f32,
    /// Pitch accuracy [0.0, 1.0], 1.0 = perfectly on pitch
    pub pitch_accuracy: f32,
    /// Pitch deviation in semitones: negative = flat, positive = sharp, 0.0 = perfectly in tune
    pub pitch_deviation: f32,
}

impl Material2d for NoisyColorMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/noisy_color_rings_2d.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
}

/// Material for the scrolling spectrogram display
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct SpectrogramMaterial {
    #[texture(0)]
    #[sampler(1)]
    pub texture: Handle<Image>,
    #[uniform(2)]
    pub scroll_params: SpectrogramScrollParams,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, ShaderType)]
#[repr(C)]
pub struct SpectrogramScrollParams {
    /// Normalized scroll offset [0.0, 1.0] - where in the circular buffer we are
    pub scroll_offset: f32,
    /// Padding for 16-byte alignment
    pub _padding1: f32,
    pub _padding2: f32,
    pub _padding3: f32,
}

impl Material2d for SpectrogramMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/spectrogram_scroll.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
}
