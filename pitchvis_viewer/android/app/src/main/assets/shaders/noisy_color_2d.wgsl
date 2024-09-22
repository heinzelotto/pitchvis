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

@group(2) @binding(0) var<uniform> material_color: vec4<f32>;
@group(2) @binding(1) var<uniform> params: vec4<f32>;

// @group(0) @binding(1) var<uniform> globals: Globals;

const PI = 3.14159265359;

fn petal(uv_inp: vec2<f32>) -> f32 {
    var uv = uv_inp;

    // to make sure that the petal is contained in [0, PI/4]
    let rotate_22 = mat2x2<f32>(0.9239, -0.3827, 0.3827, 0.9239);
    uv = rotate_22 * uv;

    uv.x -= 0.5;
    uv.x += 0.5;
    uv.y /= uv.x*uv.x*1.0;
    uv.x -= 0.5;
    let c = 0.5;
    return 1.0-smoothstep(c-0.01, c, length(uv));
}

fn flower(uv: vec2<f32>, num_petals: f32) -> f32 {
    let theta = atan2(uv.y, uv.x);
    let theta_01 = ((theta/(2*PI)+1.0) - trunc((theta/(2*PI) + 1.0)));
    let r = length(uv);

    let angle_repeated_clamped = (theta_01*num_petals - trunc(theta_01 * num_petals))/num_petals * 2.0 * PI;

    let uv_repeated_clamped = vec2<f32>(r*cos(angle_repeated_clamped), r*sin(angle_repeated_clamped));

    return petal(uv_repeated_clamped);
}

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let calmness = params.x;
    let time = params.y;
    // goes from 250 to 350
    let time_periodic = 250.0 + time - floor(time/100.0)*100.0;

    // time stretched by calmness, high calmness => slow time
    let t = time_periodic * clamp(0.33-abs(calmness - 0.33), 0.0, 1.0) / 4.0;
    let rot_t = mat2x2<f32>(cos(t), -sin(t), sin(t), cos(t));
    let uv = rot_t * (mesh.uv * 2.0 - 1.0);

    let f_noise_raw: f32 = simplexNoise3(vec3<f32>(mesh.uv *3.3, time*0.4 + t));
    let f_noise: f32 = clamp(f_noise_raw-0.15, 0.0, 1.0);
    let white = vec3<f32>(1.0, 1.0, 1.0);

    // ~3 to 9 petals, higher calmness => more petals
    let num_petals = trunc(clamp(calmness - 0.30, 0.0, 1.0) * 2.0 * 8.0 + 3.0);
    let f_flower = flower(uv, num_petals);

    // high calmness^3 => some white sparkles
    let flower_color = vec4<f32>(mix(material_color.rgb, white, f_noise*calmness*calmness*calmness*f_flower), material_color.a*f_flower);

    let f_base = smooth_circle_boundary(material_color, uv);

    // high calmness^2 => more flower, less opaque circle
    let flower_strength = clamp(calmness * 1.65, 0.0, 1.0)*clamp(calmness * 1.65, 0.0, 1.0);
    return mix(f_base, flower_color, flower_strength);
}
