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
    /// Vibrato rate in Hz [0.0, 10.0] - 0.0 if no vibrato
    pub vibrato_rate: f32,
    /// Vibrato extent normalized [0.0, 1.0] - 0.0 if no vibrato
    pub vibrato_extent: f32,
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
