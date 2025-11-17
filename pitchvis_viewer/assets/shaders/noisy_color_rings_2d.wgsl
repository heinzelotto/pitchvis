#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import bevy_render::globals::Globals

//  MIT License. © Ian McEwan, Stefan Gustavson, Munrocket
//
fn permute4(x: vec4<f32>) -> vec4<f32> { return ((x * 34. + 1.) * x) % vec4<f32>(289.); }
fn taylorInvSqrt4(r: vec4<f32>) -> vec4<f32> { return 1.79284291400159 - 0.85373472095314 * r; }

fn simplexNoise3(v: vec3<f32>) -> f32 {
  let C = vec2<f32>(1. / 6., 1. / 3.);
  let D = vec4<f32>(0., 0.5, 1., 2.);

  // First corner
  var i: vec3<f32>  = floor(v + dot(v, C.yyy));
  let x0 = v - i + dot(i, C.xxx);

  // Other corners
  let g = step(x0.yzx, x0.xyz);
  let l = 1.0 - g;
  let i1 = min(g.xyz, l.zxy);
  let i2 = max(g.xyz, l.zxy);

  // x0 = x0 - 0. + 0. * C
  let x1 = x0 - i1 + 1. * C.xxx;
  let x2 = x0 - i2 + 2. * C.xxx;
  let x3 = x0 - 1. + 3. * C.xxx;

  // Permutations
  i = i % vec3<f32>(289.);
  let p = permute4(permute4(permute4(
      i.z + vec4<f32>(0., i1.z, i2.z, 1. )) +
      i.y + vec4<f32>(0., i1.y, i2.y, 1. )) +
      i.x + vec4<f32>(0., i1.x, i2.x, 1. ));

  // Gradients (NxN points uniformly over a square, mapped onto an octahedron.)
  var n_: f32 = 1. / 7.; // N=7
  let ns = n_ * D.wyz - D.xzx;

  let j = p - 49. * floor(p * ns.z * ns.z); // mod(p, N*N)

  let x_ = floor(j * ns.z);
  let y_ = floor(j - 7.0 * x_); // mod(j, N)

  let x = x_ *ns.x + ns.yyyy;
  let y = y_ *ns.x + ns.yyyy;
  let h = 1.0 - abs(x) - abs(y);

  let b0 = vec4<f32>( x.xy, y.xy );
  let b1 = vec4<f32>( x.zw, y.zw );

  let s0 = floor(b0)*2.0 + 1.0;
  let s1 = floor(b1)*2.0 + 1.0;
  let sh = -step(h, vec4<f32>(0.));

  let a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  let a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  var p0: vec3<f32> = vec3<f32>(a0.xy, h.x);
  var p1: vec3<f32> = vec3<f32>(a0.zw, h.y);
  var p2: vec3<f32> = vec3<f32>(a1.xy, h.z);
  var p3: vec3<f32> = vec3<f32>(a1.zw, h.w);

  // Normalise gradients
  let norm = taylorInvSqrt4(vec4<f32>(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 = p0 * norm.x;
  p1 = p1 * norm.y;
  p2 = p2 * norm.z;
  p3 = p3 * norm.w;

  // Mix final noise value
  var m: vec4<f32> = 0.6 - vec4<f32>(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3));
  m = max(m, vec4<f32>(0.));
  m = m * m;
  return 42. * dot(m * m, vec4<f32>(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

fn hashOld33(p: vec3<f32>) -> vec3<f32> {
    let q = vec3<f32>(
        dot(p, vec3<f32>(127.1, 311.7, 74.7)),
        dot(p, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p, vec3<f32>(113.5, 271.9, 124.6))
    );

    return fract(sin(q) * 43758.5453123);
}

fn hash13(p3: vec3<f32>) -> f32
{
	  var q  = fract(p3 * .1031);
    q += dot(q, q.zyx + 31.32);
    return fract((q.x + q.y) * q.z);
}

fn nois(p3: vec3<f32>) -> f32
{
  let q3 = floor(p3);
  return fract(sin(dot(q3, vec3f(12.9898, 78.233, 69.420))) * 43758.5453);
}

fn smooth_circle_boundary(color: vec4<f32>, uv: vec2<f32>) -> vec4<f32> {
    return mix(color, vec4<f32>(color.rgb, 0.0), smoothstep(0.96, 1.0, length(uv)));
}

struct Params {
    calmness: f32,
    time: f32,
    vibrato_rate: f32,
    vibrato_extent: f32,
    pitch_accuracy: f32,
    pitch_deviation: f32,
}

@group(2) @binding(0) var<uniform> material_color: vec4<f32>;
@group(2) @binding(1) var<uniform> params: Params;

const PI = 3.14159265359;

fn ring(uv: vec2<f32>) -> f32 {
    let r = length(uv);
    let f = sin(r*sqrt(r)*PI*1.0);
    return f*f;
}

// PITCH INDICATOR OPTIONS - Comment/uncomment to test different styles
// Only shows when pitch_accuracy > 0.85 (pretty accurately on pitch)

// OPTION 1: Tiny bright dot in center (ACTIVE)
fn pitch_indicator_center_dot(uv: vec2<f32>, pitch_accuracy: f32, time: f32) -> vec3<f32> {
    let threshold = 0.85;
    if pitch_accuracy < threshold {
        return vec3<f32>(0.0);
    }

    let accuracy_factor = (pitch_accuracy - threshold) / (1.0 - threshold);
    let r = length(uv);
    let dot_size = 0.08;
    let dot_falloff = smoothstep(dot_size, 0.0, r);

    // Subtle pulse
    let pulse = 0.85 + 0.15 * sin(time * 3.0);

    return vec3<f32>(1.0) * dot_falloff * accuracy_factor * pulse;
}

// OPTION 2: Pulsing bright ring (INACTIVE)
// fn pitch_indicator_center_dot(uv: vec2<f32>, pitch_accuracy: f32, time: f32) -> vec3<f32> {
//     let threshold = 0.85;
//     if pitch_accuracy < threshold {
//         return vec3<f32>(0.0);
//     }
//
//     let accuracy_factor = (pitch_accuracy - threshold) / (1.0 - threshold);
//     let r = length(uv);
//
//     // Pulsing ring at edge
//     let pulse = 0.5 + 0.5 * sin(time * 4.0);
//     let ring_center = 0.7;
//     let ring_width = 0.15;
//     let ring_intensity = smoothstep(ring_center + ring_width, ring_center, abs(r - ring_center));
//
//     return vec3<f32>(1.0) * ring_intensity * accuracy_factor * pulse;
// }

// OPTION 3: Four-pointed star/cross pattern (INACTIVE)
// fn pitch_indicator_center_dot(uv: vec2<f32>, pitch_accuracy: f32, time: f32) -> vec3<f32> {
//     let threshold = 0.85;
//     if pitch_accuracy < threshold {
//         return vec3<f32>(0.0);
//     }
//
//     let accuracy_factor = (pitch_accuracy - threshold) / (1.0 - threshold);
//     let r = length(uv);
//     let angle = atan2(uv.y, uv.x);
//
//     // Create 4-pointed star
//     let star_arms = abs(cos(angle * 2.0));
//     let star_size = 0.25;
//     let star_width = 0.08;
//
//     let star_pattern = smoothstep(0.0, star_width, star_arms) * smoothstep(star_size, 0.0, r);
//     let pulse = 0.8 + 0.2 * sin(time * 3.0);
//
//     return vec3<f32>(1.0) * star_pattern * accuracy_factor * pulse;
// }

// OPTION 4: Bright center with radial gradient (INACTIVE)
// fn pitch_indicator_center_dot(uv: vec2<f32>, pitch_accuracy: f32, time: f32) -> vec3<f32> {
//     let threshold = 0.85;
//     if pitch_accuracy < threshold {
//         return vec3<f32>(0.0);
//     }
//
//     let accuracy_factor = (pitch_accuracy - threshold) / (1.0 - threshold);
//     let r = length(uv);
//
//     // Soft radial glow from center
//     let glow_size = 0.3;
//     let glow = smoothstep(glow_size, 0.0, r);
//     let pulse = 0.7 + 0.3 * sin(time * 2.5);
//
//     // Extra bright spot in the very center
//     let center_dot = smoothstep(0.05, 0.0, r) * 0.5;
//
//     return vec3<f32>(1.0) * (glow + center_dot) * accuracy_factor * pulse;
// }

// OPTION 5: Multiple concentric rings flash (INACTIVE)
// fn pitch_indicator_center_dot(uv: vec2<f32>, pitch_accuracy: f32, time: f32) -> vec3<f32> {
//     let threshold = 0.85;
//     if pitch_accuracy < threshold {
//         return vec3<f32>(0.0);
//     }
//
//     let accuracy_factor = (pitch_accuracy - threshold) / (1.0 - threshold);
//     let r = length(uv);
//
//     // Multiple rings that flash in sequence
//     let ring_freq = 8.0;
//     let ring_pattern = sin(r * ring_freq * PI);
//     let ring_highlight = max(0.0, ring_pattern);
//
//     // Animated wave traveling outward
//     let wave = sin(r * 5.0 - time * 4.0);
//     let wave_highlight = smoothstep(0.3, 1.0, wave);
//
//     return vec3<f32>(1.0) * ring_highlight * wave_highlight * accuracy_factor;
// }

// DYNAMIC TUNING INDICATOR OPTIONS
// Shows tuning direction (sharp/flat) and accuracy with animated shapes

// OPTION 1: Spiral star that rotates based on tuning (ACTIVE)
fn tuning_indicator(uv: vec2<f32>, pitch_deviation: f32, time: f32) -> vec3<f32> {
    let r = length(uv);
    if r > 0.25 || r < 0.01 {
        return vec3<f32>(0.0);
    }

    let angle = atan2(uv.y, uv.x);

    // Create a 6-pointed star
    let num_points = 6.0;
    let star_angle = angle * num_points;

    // Spiral effect based on pitch deviation
    // positive deviation (sharp) = clockwise spiral
    // negative deviation (flat) = counterclockwise spiral
    let spiral_amount = pitch_deviation * 4.0; // amplify the effect
    let spiral_angle = star_angle + r * spiral_amount * PI * 4.0;

    // Create star arms
    let star_intensity = max(0.0, cos(spiral_angle)) * (1.0 - smoothstep(0.15, 0.25, r));

    // Pulse based on tuning accuracy
    let accuracy = 1.0 - abs(pitch_deviation) * 2.0; // 1.0 = perfect, 0.0 = way off
    let pulse = 0.7 + 0.3 * sin(time * 3.0);

    // Brighter when more accurate
    let brightness = mix(0.3, 1.0, accuracy) * pulse;

    return vec3<f32>(1.0) * star_intensity * brightness;
}

// OPTION 2: Shape morphing - circle (flat) -> dot (accurate) -> figure-8 (sharp) (INACTIVE)
// fn tuning_indicator(uv: vec2<f32>, pitch_deviation: f32, time: f32) -> vec3<f32> {
//     let r = length(uv);
//     if r > 0.3 {
//         return vec3<f32>(0.0);
//     }
//
//     // When flat (negative deviation): show a circle outline
//     // When accurate (near 0): shrink to a small dot
//     // When sharp (positive deviation): show a figure-8 / lemniscate
//
//     if pitch_deviation < -0.05 {
//         // Flat: circle outline
//         let circle_r = 0.15 + abs(pitch_deviation) * 0.3;
//         let circle = smoothstep(circle_r + 0.02, circle_r, r) - smoothstep(circle_r, circle_r - 0.02, r);
//         let pulse = 0.7 + 0.3 * sin(time * 3.0);
//         return vec3<f32>(1.0, 0.8, 0.6) * circle * pulse; // warm color for flat
//     } else if pitch_deviation > 0.05 {
//         // Sharp: figure-8 (lemniscate)
//         let angle = atan2(uv.y, uv.x);
//         let lemniscate_r = 0.2 * sqrt(abs(cos(2.0 * angle)));
//         let figure8 = smoothstep(0.04, 0.0, abs(r - lemniscate_r));
//         let pulse = 0.7 + 0.3 * sin(time * 3.0);
//         return vec3<f32>(0.6, 0.8, 1.0) * figure8 * pulse; // cool color for sharp
//     } else {
//         // Accurate: small bright dot
//         let dot_size = 0.06;
//         let dot = smoothstep(dot_size, 0.0, r);
//         let pulse = 0.85 + 0.15 * sin(time * 4.0);
//         return vec3<f32>(1.0) * dot * pulse; // bright white for accurate
//     }
// }

// OPTION 3: Triple-loop (triquetra-like) that rotates (INACTIVE)
// fn tuning_indicator(uv: vec2<f32>, pitch_deviation: f32, time: f32) -> vec3<f32> {
//     let r = length(uv);
//     if r > 0.3 || r < 0.01 {
//         return vec3<f32>(0.0);
//     }
//
//     let angle = atan2(uv.y, uv.x);
//
//     // Create 3-lobed pattern
//     let num_lobes = 3.0;
//     let lobe_angle = angle * num_lobes;
//
//     // Rotation based on pitch deviation
//     let rotation = pitch_deviation * PI * 2.0 + time * 0.5;
//     let rotated_angle = lobe_angle + rotation;
//
//     // Create the triple loop pattern
//     let lobe = (cos(rotated_angle) + 1.0) * 0.5;
//     let pattern_r = 0.15 * (1.0 + 0.4 * lobe);
//     let intensity = smoothstep(0.04, 0.0, abs(r - pattern_r));
//
//     // Color based on tuning
//     let color = mix(
//         vec3<f32>(1.0, 0.8, 0.6),  // flat = warm
//         vec3<f32>(0.6, 0.8, 1.0),  // sharp = cool
//         (pitch_deviation + 0.5)
//     );
//
//     let accuracy = 1.0 - abs(pitch_deviation) * 2.0;
//     let brightness = mix(0.4, 1.0, accuracy);
//
//     return color * intensity * brightness;
// }

// OPTION 4: Pulsing rings with directional wave (INACTIVE)
// fn tuning_indicator(uv: vec2<f32>, pitch_deviation: f32, time: f32) -> vec3<f32> {
//     let r = length(uv);
//     if r > 0.28 {
//         return vec3<f32>(0.0);
//     }
//
//     let angle = atan2(uv.y, uv.x);
//
//     // Concentric rings
//     let ring_freq = 15.0;
//     let rings = sin(r * ring_freq * PI);
//
//     // Traveling wave based on tuning
//     // sharp = wave travels outward
//     // flat = wave travels inward
//     let wave_direction = sign(pitch_deviation);
//     let wave = sin(r * 8.0 * PI - time * 4.0 * wave_direction + angle * pitch_deviation * 4.0);
//
//     let pattern = max(0.0, rings) * max(0.0, wave);
//
//     // Color based on tuning
//     let color = mix(
//         vec3<f32>(1.0, 0.7, 0.5),  // flat = orange
//         vec3<f32>(0.5, 0.7, 1.0),  // sharp = blue
//         (pitch_deviation + 0.5)
//     );
//
//     let accuracy = 1.0 - abs(pitch_deviation) * 2.0;
//     let brightness = mix(0.3, 0.9, accuracy);
//
//     return color * pattern * brightness;
// }

// OPTION 5: Breathing cross/plus that bends into curves (INACTIVE)
// fn tuning_indicator(uv: vec2<f32>, pitch_deviation: f32, time: f32) -> vec3<f32> {
//     // Create a + shape that bends into ( or ) based on tuning
//     let bend = pitch_deviation * 3.0;
//
//     // Horizontal arm with bend
//     let h_dist = abs(uv.y) - (0.03 + abs(uv.x * bend * 0.5));
//     let h_arm = smoothstep(0.04, 0.0, h_dist) * smoothstep(0.25, 0.0, abs(uv.x));
//
//     // Vertical arm with bend
//     let v_dist = abs(uv.x) - (0.03 + abs(uv.y * bend * 0.5));
//     let v_arm = smoothstep(0.04, 0.0, v_dist) * smoothstep(0.25, 0.0, abs(uv.y));
//
//     let cross = max(h_arm, v_arm);
//
//     // Breathing effect
//     let pulse = 0.7 + 0.3 * sin(time * 3.0);
//
//     // Color gradient based on tuning
//     let color = mix(
//         vec3<f32>(1.0, 0.6, 0.4),  // flat = warm red
//         vec3<f32>(0.4, 0.6, 1.0),  // sharp = cool blue
//         (pitch_deviation + 0.5)
//     );
//
//     let accuracy = 1.0 - abs(pitch_deviation) * 2.0;
//     let brightness = mix(0.5, 1.0, accuracy) * pulse;
//
//     return color * cross * brightness;
// }

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let calmness = params.calmness;
    let time = params.time;
    let vibrato_rate = params.vibrato_rate;
    let vibrato_extent = params.vibrato_extent;
    let pitch_accuracy = params.pitch_accuracy;
    let pitch_deviation = params.pitch_deviation;

    // goes from 250 to 350
    let time_periodic = 250.0 + time - floor(time/100.0)*100.0;

    var uv = mesh.uv * 2.0 - 1.0;

    // Vibrato visualization: pulsate rings at vibrato rate
    // Only apply if vibrato is detected (rate > 0)
    if (vibrato_rate > 0.1) {
        // Calculate vibrato phase (oscillates at vibrato_rate Hz)
        let vibrato_phase = sin(time * vibrato_rate * 2.0 * PI);

        // Scale UV coordinates to create pulsating effect
        // Amplitude is controlled by vibrato_extent (0.0-1.0)
        // Max pulse is ±5% of size
        let pulse_amplitude = vibrato_extent * 0.05;
        let scale_factor = 1.0 + vibrato_phase * pulse_amplitude;
        uv = uv * scale_factor;

        // Also modulate ring pattern density based on vibrato
        // This creates a "breathing" effect in the rings
    }

    // Base color with rings
    let f_ring = ring(uv);
    let ring_color = vec4<f32>(material_color.rgb, material_color.a*f_ring);

    // Add pitch accuracy indicator (only when very accurate)
    let accuracy_indicator = pitch_indicator_center_dot(uv, pitch_accuracy, time);

    // Add tuning direction indicator (shows sharp/flat with animated shape)
    let tuning_indicator_color = tuning_indicator(uv, pitch_deviation, time);

    let final_color = vec4<f32>(ring_color.rgb + accuracy_indicator + tuning_indicator_color, ring_color.a);

    // high 1-(1-calmness)^3 => more full disk, less ring
    let ring_strength = clamp(1.0-calmness * 1.65, 0.0, 1.0)*clamp(1.0-calmness * 1.65, 0.0, 1.0)*clamp(1.0-calmness * 1.65, 0.0, 1.0);
    return smooth_circle_boundary(mix(material_color, final_color, ring_strength), uv);
}
