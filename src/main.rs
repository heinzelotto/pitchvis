use std::f32::consts::PI;

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use kiss3d::light::Light;
use kiss3d::nalgebra::{UnitQuaternion, Vector3};
use kiss3d::window::Window;
use num_complex::{Complex32, ComplexFloat};
use rustfft::num_traits::Zero;
use rustfft::FftPlanner;

const SR: usize = 22050;
const N_FFT: usize = 4096;
const FREQ_A1: f32 = 55.0;
const SPARSITY_QUANTILE: f32 = 0.98;

const OCTAVES: usize = 5;
const BUCKETS_PER_OCTAVE: usize = 36;

const FPS: f32 = 50.0;
const COLORS: [(f32, f32, f32); 12] = [
    (0.80, 0.40, 0.39), // C
    (0.24, 0.51, 0.66), // C#
    (0.96, 0.77, 0.25), // D
    (0.36, 0.28, 0.49), // Eb
    (0.51, 0.76, 0.30), // E
    (0.74, 0.36, 0.51), // F
    (0.27, 0.62, 0.56), // F#
    (0.91, 0.56, 0.30), // G
    (0.26, 0.31, 0.52), // Ab
    (0.85, 0.87, 0.26), // A
    (0.54, 0.31, 0.53), // Bb
    (0.27, 0.69, 0.39), // H
];

fn calculate_color(bucket: usize) -> (f32, f32, f32) {
    let pitch_continuous = 12.0 * (bucket as f32) / (BUCKETS_PER_OCTAVE as f32);
    let base_color = COLORS[(pitch_continuous.round() as usize) % 12];
    let inaccuracy_cents = (pitch_continuous - pitch_continuous.round()).abs();

    let saturation = 1.0 - (2.0 * inaccuracy_cents) * (2.0 * inaccuracy_cents);
    const GRAY_LEVEL: f32 = 0.5;

    (
        saturation * base_color.0 + (1.0 - saturation) * GRAY_LEVEL,
        saturation * base_color.1 + (1.0 - saturation) * GRAY_LEVEL,
        saturation * base_color.2 + (1.0 - saturation) * GRAY_LEVEL,
    )
}

fn arg_max(sl: &[f32]) -> usize {
    // we have no NaNs
    sl.iter()
        .enumerate()
        .fold(
            (0, f32::MIN),
            |cur, x| if *x.1 > cur.1 { (x.0, *x.1) } else { cur },
        )
        .0
}

fn cqt_kernel(
    sr: usize,
    n: usize,
    min_freq: f32,
    buckets_per_octave: usize,
    octaves: usize,
) -> sprs::CsMat<Complex32> {
    let num_buckets = buckets_per_octave * octaves;
    let window = apodize::hanning_iter(n).collect::<Vec<f64>>();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(N_FFT as usize);
    let mut a = sprs::TriMat::new((num_buckets as usize, n as usize));

    // fill a with the 'atoms', i. e. morlet wavelets
    for k in 0..num_buckets {
        let f_k = min_freq * 2.0_f32.powf(k as f32 / buckets_per_octave as f32);
        let mut v = Vec::new();
        for i in 0..n {
            v.push(
                1.0 / (n as f32)
                    * (window[i] as f32)
                    * (-num_complex::Complex32::i() * 2.0 * PI * (i as f32) * f_k / (sr as f32))
                        .exp(),
            );
        }

        // transform wavelets into frequency space
        fft.process(&mut v);

        // the complex conjugate is what we later need
        v = v.iter_mut().map(|z| z.conj()).collect();

        // filter all values smaller than some value and use sparse arrays
        let mut v_abs = v.iter().map(|z| z.abs()).collect::<Vec<f32>>();
        v_abs.sort_by(|a, b| {
            if a == b {
                std::cmp::Ordering::Equal
            } else if a < b {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });
        let cutoff_value = v_abs[(n as f32 * SPARSITY_QUANTILE) as usize];
        let mut cnt = 0;
        v.iter_mut().for_each(|z| {
            if z.abs() < cutoff_value {
                *z = Complex32::zero();
                cnt += 1;
            }
        });
        println!("for k {k} erased {cnt} points below {cutoff_value}");

        // fill the kernel matrix
        for (i, z) in v.iter().enumerate() {
            if !z.is_zero() {
                a.add_triplet(k, i, *z);
            }
        }
    }

    a.to_csr()
}

fn main() -> Result<()> {
    dbg!(cpal::available_hosts());

    let host = cpal::default_host();

    // let device = host
    //     .devices()?
    //     .find(|d| d.name().unwrap() == "plughw:CARD=USB,DEV=0")
    //     .unwrap();

    let device = host.default_input_device().unwrap();

    println!("{}", device.name()?);
    let stream_config = cpal::StreamConfig {
        channels: 1u16,
        sample_rate: cpal::SampleRate(SR as u32),
        buffer_size: cpal::BufferSize::Default,
    };

    let cqt_kernel = cqt_kernel(SR, N_FFT, FREQ_A1, BUCKETS_PER_OCTAVE, OCTAVES);

    let mut ring_buffer = Vec::new();
    ring_buffer.resize(SR as usize, 0f32);
    let ring_buffer = std::sync::Mutex::from(ring_buffer);
    let ring_buffer = std::sync::Arc::new(ring_buffer);

    let ring_buffer_input_thread_clone = ring_buffer.clone();

    let stream = device.build_input_stream(
        &stream_config,
        move |data: &[f32], info: &cpal::InputCallbackInfo| {
            let mut rb = ring_buffer_input_thread_clone.lock().unwrap();
            rb.drain(..data.len());
            rb.extend_from_slice(&data);
            println!(
                "callback called fs {}, {:?}, {}",
                data.len(),
                info.timestamp().capture,
                rb.len()
            );
        },
        move |err| panic!("{}", err),
    )?;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(N_FFT as usize);

    let mut window = Window::new("Kiss3d: cube");
    let mut cubes = vec![];
    for i in 0..(BUCKETS_PER_OCTAVE * OCTAVES) {
        let mut c = window.add_cube(1.0, 1.0, 1.0);
        let (transl_x, transl_y) = (
            (((i + BUCKETS_PER_OCTAVE - 3 * (BUCKETS_PER_OCTAVE / 12)) as f32)
                / (BUCKETS_PER_OCTAVE as f32)
                * 2.0
                * PI)
                .cos()
                * (i as f32 / BUCKETS_PER_OCTAVE as f32 * 2.0),
            (((i + BUCKETS_PER_OCTAVE - 3 * (BUCKETS_PER_OCTAVE / 12)) as f32)
                / (BUCKETS_PER_OCTAVE as f32)
                * 2.0
                * PI)
                .sin()
                * (i as f32 / BUCKETS_PER_OCTAVE as f32 * 2.0),
        );
        c.prepend_to_local_translation(&kiss3d::nalgebra::Translation::from([
            -transl_x, transl_y, 0.0,
        ]));
        cubes.push(c);
    }

    window.set_light(Light::StickToCamera);

    let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.014);

    stream.play()?;
    loop {
        let t_cqt = std::time::Instant::now();
        let mut x = ring_buffer.lock().unwrap()[(SR - N_FFT)..]
            .iter()
            .map(|f| rustfft::num_complex::Complex32::new(*f, 0.0))
            .collect::<Vec<rustfft::num_complex::Complex32>>();
        fft.process(&mut x);

        // signal fft'd
        let mut x_cqt = vec![Complex32::zero(); BUCKETS_PER_OCTAVE * OCTAVES];
        dbg!(cqt_kernel.shape(), x_cqt.len(), x.len());

        sprs::prod::mul_acc_mat_vec_csr(cqt_kernel.view(), &x, &mut x_cqt);
        let x_cqt: Vec<f32> = x_cqt
            .iter()
            .map(|z| (z.abs().log(10.0) + 1.0).max(0.0) / 2.0)
            .collect();

        println!("CQT calculated in {}ms", t_cqt.elapsed().as_millis());

        let k_max = arg_max(&x_cqt);
        let max = x_cqt[k_max];
        println!("max at idx {k_max}: {max}");

        let t_render = std::time::Instant::now();

        for (i, c) in cubes.iter_mut().enumerate() {
            c.prepend_to_local_rotation(&rot);
            let (r, g, b) = calculate_color(
                (i + BUCKETS_PER_OCTAVE - 3 * (BUCKETS_PER_OCTAVE / 12)) % BUCKETS_PER_OCTAVE,
            );
            c.set_color(r, g, b);
            //c.set_local_scale((x_cqt[i] / 10.0).max(0.1), (x_cqt[i] / 10.0).max(0.1), (x_cqt[i] / 10.0).max(0.1));
            c.set_local_scale(x_cqt[i], x_cqt[i], x_cqt[i]);
            //c.set_local_scale(1.0, 1.0, 1.0);
        }

        if !window.render() {
            break;
        }

        println!("rendered in {}ms", t_render.elapsed().as_millis());
    }

    Ok(())
}
