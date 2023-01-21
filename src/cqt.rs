use num_complex::{Complex32, ComplexFloat};
use rustfft::num_traits::Zero;
use rustfft::FftPlanner;
use std::f32::consts::PI;
use std::ops::DivAssign;

fn max(sl: &[f32]) -> f32 {
    // we have no NaNs
    sl.iter()
        .fold(f32::MIN, |cur, x| if *x > cur { *x } else { cur })
}
fn min(sl: &[f32]) -> f32 {
    // we have no NaNs
    sl.iter()
        .fold(f32::MAX, |cur, x| if *x < cur { *x } else { cur })
}

pub struct Cqt {
    sr: usize,
    n: usize,
    min_freq: f32,
    buckets_per_octave: usize,
    octaves: usize,
    sparsity_quantile: f32,
    quality: f32,
    cqt_kernel: sprs::CsMat<Complex32>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl Cqt {
    pub fn new(
        sr: usize,
        n: usize,
        min_freq: f32,
        buckets_per_octave: usize,
        octaves: usize,
        sparsity_quantile: f32,
        quality: f32,
    ) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n as usize);

        let cqt_kernel = cqt_kernel(
            sr,
            n,
            min_freq,
            buckets_per_octave,
            octaves,
            sparsity_quantile,
            quality,
        );

        Self {
            sr,
            n,
            min_freq,
            buckets_per_octave,
            octaves,
            sparsity_quantile,
            quality,
            cqt_kernel,
            fft,
        }
    }

    pub fn calculate_cqt_instant_in_db(&self, x: &[f32]) -> Vec<f32> {
        let mut x = x
            .iter()
            .map(|f| rustfft::num_complex::Complex32::new(*f, 0.0))
            .collect::<Vec<rustfft::num_complex::Complex32>>();
        self.fft.process(&mut x);

        // TODO: harmonic/percussion source separation possible with our one-sided spectrum?

        // signal fft'd
        let mut x_cqt = vec![Complex32::zero(); self.buckets_per_octave * self.octaves];

        sprs::prod::mul_acc_mat_vec_csr(self.cqt_kernel.view(), x, &mut x_cqt);

        // amp -> db conversion
        let ref_power: f32 = 1.0.powf(2.0);
        let a_min: f32 = 1e-5.powf(2.0);
        let top_db: f32 = 50.0;

        let power: Vec<f32> = x_cqt
            .iter()
            .map(|z| (z.abs() * z.abs()))
            .collect::<Vec<f32>>();

        let mut log_spec = power
            .iter()
            .map(|x| 10.0 * x.max(a_min).log10() - 10.0 * ref_power.max(a_min).log10())
            .collect::<Vec<f32>>();
        let log_spec_max = max(&log_spec);
        log_spec.iter_mut().for_each(|x| {
            if *x < log_spec_max - top_db {
                *x = log_spec_max - top_db
            }
        });

        println!(
            "log_spec min {}, max {}, shifting to 0...",
            min(&log_spec),
            max(&log_spec)
        );

        let mn = min(&log_spec);
        // cut off at 0.0, and don't let it pass top_db
        log_spec.iter_mut().for_each(|x| {
            if mn > 0.0 {
                *x -= mn
            } else if *x < 0.0 {
                *x = 0.0
            }
        });

        // let x_cqt: Vec<f32> = x_cqt
        //     .iter()
        //     .map(|z| (z.abs() * z.abs()).log(10.0).max(-2.0) + 2.0)
        //     .collect();

        // TODO: median filter

        log_spec
    }
}

fn cqt_kernel(
    sr: usize,
    n_fft: usize,
    min_freq: f32,
    buckets_per_octave: usize,
    octaves: usize,
    sparsity_quantile: f32,
    quality: f32,
) -> sprs::CsMat<Complex32> {
    let num_buckets = buckets_per_octave * octaves;
    // TODO: use different window lengths for every pitch
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut a = sprs::TriMat::new((num_buckets as usize, n_fft as usize));

    // fill a with the 'atoms', i. e. morlet wavelets
    for k in 0..num_buckets {
        let f_k = min_freq * 2.0_f32.powf(k as f32 / buckets_per_octave as f32);

        let Q = quality / (2.0.powf(1.0 / buckets_per_octave as f32) - 1.0);

        let window_k = Q * sr as f32 / (f_k); //* (2.0.powf(1.0 / buckets_per_octave as f32) - 1.0));
        dbg!(&window_k);
        let window_k_rounded = window_k.round() as usize;

        assert!(window_k_rounded < n_fft);

        let window = apodize::hamming_iter(window_k_rounded).collect::<Vec<f64>>();

        let mut v = vec![Complex32::zero(); n_fft];
        for i in 0..window_k_rounded {
            v[n_fft - window_k_rounded + i] = 1.0 / window_k
                * (window[i] as f32)
                * (-num_complex::Complex32::i() * 2.0 * PI * (i as f32) * f_k / (sr as f32)).exp();
        }

        // let mut v = vec![Complex32::zero(); n_fft_max];
        // for i in 0..n_fft_k_rounded {
        //     v[n_fft_max / 2 - n_fft_k_rounded / 2 + i] = 1.0 / n_fft_k
        //         * (window[i] as f32)
        //         * (-num_complex::Complex32::i() * 2.0 * PI * (i as f32) * f_k / (sr as f32)).exp();
        // }

        // normalize windowed wavelet in time space
        let norm_1: f32 = v.iter().map(|z| z.abs()).sum();
        dbg!(norm_1);
        v.iter_mut().for_each(|z| z.div_assign(norm_1));

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
        let cutoff_value = v_abs[(n_fft as f32 * sparsity_quantile) as usize];
        let mut cnt = 0;
        v.iter_mut().for_each(|z| {
            if z.abs() < cutoff_value {
                *z = Complex32::zero();
                cnt += 1;
            }
        });
        println!("for k {k} erased {cnt} points below {cutoff_value}");
        let mut cnt_nonzero = 0;
        v.iter().for_each(|z| {
            if z.abs() > 0.00001 {
                cnt_nonzero += 1;
            }
        });
        println!("cnt_nonzero {cnt_nonzero}");

        // fill the kernel matrix
        for (i, z) in v.iter().enumerate() {
            if !z.is_zero() {
                a.add_triplet(k, i, *z);
            }
        }
    }

    a.to_csr()
}
