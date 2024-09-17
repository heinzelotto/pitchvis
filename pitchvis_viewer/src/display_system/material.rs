//! A shader material for rendering the pitch balls, with a parameter to control the noise level.

use bevy::{
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderRef, ShaderType},
    sprite::Material2d,
};

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct NoisyColorMaterial {
    #[uniform(0)]
    pub color: LinearRgba,
    #[uniform(1)]
    pub noise_level: Noise, // Padding to ensure 16-Byte alignment
}

#[derive(Debug, Clone, Copy, Default, PartialEq, ShaderType)]
// #[repr(C)]
pub struct Noise {
    /// The red channel. [0.0, 1.0]
    pub val: f32,
    pub _pad1: f32,
    pub _pad2: f32,
    pub _pad3: f32,
}

impl Material2d for NoisyColorMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/noisy_color_2d.wgsl".into()
    }
}
