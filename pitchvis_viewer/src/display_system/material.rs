//! A shader material for rendering the pitch balls, with a parameter to control the noise level.

use bevy::{
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderRef},
    sprite::Material2d,
};

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct NoisyColorMaterial {
    #[uniform(0)]
    pub color: LinearRgba,
    #[uniform(1)]
    pub noise_level: f32,
}

impl Material2d for NoisyColorMaterial {
    // Caused by:
    // In Device::create_render_pipeline
    //   note: label = `transparent_mesh2d_pipeline`
    // In the provided shader, the type given for group 2 binding 1 has a size of 4.
    // As the device does not support `DownlevelFlags::BUFFER_BINDINGS_NOT_16_BYTE_ALIGNED`,
    // the type must have a size that is a multiple of 16 bytes.
    #[cfg(not(target_arch = "wasm32"))]
    fn fragment_shader() -> ShaderRef {
        "shaders/noisy_color_2d.wgsl".into()
    }
}
