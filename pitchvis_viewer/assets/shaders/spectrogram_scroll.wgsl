#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct SpectrogramScrollParams {
    scroll_offset: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(2) @binding(0) var spectrogram_texture: texture_2d<f32>;
@group(2) @binding(1) var spectrogram_sampler: sampler;
@group(2) @binding(2) var<uniform> params: SpectrogramScrollParams;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    // Offset the V coordinate by scroll_offset to create scrolling effect
    // The texture is rotated -90°, so V becomes the horizontal (time) axis
    // We want to shift V so that the current write position appears at the right edge
    //
    // After rotation:
    // - U = frequency axis (vertical, low to high)
    // - V = time axis (horizontal, right to left in texture, but we want newest on right after offset)
    //
    // scroll_offset is write_index / height (normalized position in circular buffer)
    // We want newest data (at write_index) to appear at V=1.0 (right edge after rotation)
    // So we offset V by (1.0 - scroll_offset) and wrap with fract()

    var uv = mesh.uv;
    uv.y = fract(uv.y + (1.0 - params.scroll_offset));

    let color = textureSample(spectrogram_texture, spectrogram_sampler, uv);
    return color;
}
