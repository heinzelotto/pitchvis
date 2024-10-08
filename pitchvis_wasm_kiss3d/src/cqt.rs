use log::{debug, trace};
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

pub struct Vqt {
    sr: usize,
    n_fft: usize,
    min_freq: f32,
    buckets_per_octave: usize,
    octaves: usize,
    _sparsity_quantile: f32,
    _quality: f32,
    _gamma: f32,
    vqt_kernel: Vec<sprs::CsMat<Complex32>>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    _t_diff: f32,
}

impl Vqt {
    #[allow(dead_code)]
    fn test_create_sines(&self, t_diff: f32) -> Vec<f32> {
        let mut wave = vec![0.0; self.n_fft];

        for f in ((12 * (0) + 0)..(12 * (self.octaves - 1) + 12))
            .map(|p| self.min_freq * (2.0).powf(p as f32 / 12.0))
        {
            //let f = 880.0 * 2.0.powf(1.0/12.0);
            for i in 0..wave.len() {
                let amp = (((i as f32 + t_diff * self.sr as f32) * 2.0 * PI / self.sr as f32) * f)
                    .sin()
                    / 12.0;
                wave[i] += amp;
            }
        }

        wave
    }

    pub fn new(
        sr: usize,
        n_fft: usize,
        min_freq: f32,
        buckets_per_octave: usize,
        octaves: usize,
        sparsity_quantile: f32,
        quality: f32,
        gamma: f32,
    ) -> Self {
        let reduced_n_fft = n_fft >> (octaves - 1);

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(reduced_n_fft as usize);

        let vqt_kernel = Self::vqt_kernel(
            sr,
            n_fft,
            min_freq,
            buckets_per_octave,
            octaves,
            sparsity_quantile,
            quality,
            gamma,
        );

        Self {
            sr,
            n_fft,
            min_freq,
            buckets_per_octave,
            octaves,
            _sparsity_quantile: sparsity_quantile,
            _quality: quality,
            _gamma: gamma,
            vqt_kernel,
            fft,
            _t_diff: 0.0,
        }
    }

    fn vqt_kernel(
        sr: usize,
        n_fft: usize,
        min_freq: f32,
        buckets_per_octave: usize,
        octaves: usize,
        sparsity_quantile: f32,
        quality: f32,
        gamma: f32,
    ) -> Vec<sprs::CsMat<Complex32>> {
        //let lowest_freq_of_target_octave = min_freq * 2.0_f32.powf((octaves - 1) as f32);
        //let freqs = (0..buckets_per_octave)
        //    .map(|k| lowest_freq_of_target_octave * 2.0_f32.powf(k as f32 / buckets_per_octave as f32))
        //    .collect::<Vec<f32>>();
        //dbg!(&freqs);

        let freqs = (0..(buckets_per_octave * octaves))
            .map(|k| min_freq * 2.0_f32.powf(k as f32 / buckets_per_octave as f32))
            .collect::<Vec<f32>>();

        let highest_frequency = *freqs.last().unwrap();
        let nyquist_frequency = sr / 2;
        assert!(highest_frequency <= nyquist_frequency as f32);

        // calculate filter window sizes
        let r = 2.0.powf(1.0 / buckets_per_octave as f32);
        // alpha is constant and such that (1+a)*f_{k-1} = (1-a)*f_{k+1}
        let alpha = (r.powf(2.0) - 1.0) / (r.powf(2.0) + 1.0);
        #[allow(non_snake_case)]
        let Q = quality / alpha;
        let window_lengths = freqs
            .iter()
            .map(|f_k| {
                let window_k = Q * sr as f32 / (f_k + gamma / alpha);
                window_k
            })
            .collect::<Vec<f32>>();

        let reduced_n_fft = n_fft >> (octaves - 1);
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(reduced_n_fft);
        //let reduced_n_ffts = (0..octaves).map(|i| n_fft >> i).collect::<Vec<usize>>();
        //let ffts = reduced_n_ffts
        //    .iter()
        //    .map(|i| planner.plan_fft_forward(*i))
        //    .collect::<Vec<std::sync::Arc<dyn rustfft::Fft<f32>>>>();
        let mut kernel_octaves = (0..octaves)
            .map(|_| sprs::TriMat::new((buckets_per_octave as usize, reduced_n_fft)))
            .collect::<Vec<sprs::TriMat<Complex32>>>();
        for (k, (f_k, n_k)) in std::iter::zip(freqs.iter(), window_lengths.iter()).enumerate() {
            let cur_octave = k / buckets_per_octave;
            let scaling = (1 << (octaves - 1 - cur_octave)) as f32;
            let scaled_f_k = f_k * scaling;
            let scaled_n_k = n_k / scaling;

            let scaled_n_k_rounded = scaled_n_k.round() as usize;

            assert!(scaled_n_k_rounded < reduced_n_fft);
            let window = apodize::hanning_iter(scaled_n_k_rounded).collect::<Vec<f64>>();
            let _window_sum = window.iter().sum::<f64>();

            let mut v = vec![Complex32::zero(); reduced_n_fft];
            for i in 0..scaled_n_k_rounded {
                v[reduced_n_fft /*/ 2*/ - scaled_n_k_rounded /*/ 2*/ + i] = 1.0 / scaled_n_k
                    * (window[i] as f32)
                    * (num_complex::Complex32::i()
                        * 2.0
                        * PI
                        * (i as f32/*- window_lengths[k] / 2.0*/)
                        * scaled_f_k
                        / (sr as f32))
                        .exp();
            }

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
            let v_abs_sum = v_abs.iter().sum::<f32>();
            let mut accum = 0.0;
            let mut cutoff_idx = 0;
            while accum < (1.0 - sparsity_quantile) * v_abs_sum {
                accum += v_abs[cutoff_idx];
                cutoff_idx += 1;
            }
            let cutoff_value = v_abs[cutoff_idx - 1];
            //let cutoff_value = v_abs[(reduced_n_fft as f32 * sparsity_quantile) as usize];
            let mut cnt = 0;
            v.iter_mut().for_each(|z| {
                if z.abs() < cutoff_value {
                    *z = Complex32::zero();
                    cnt += 1;
                }
            });
            debug!("for k {k} erased {cnt} points below {cutoff_value} with sum {accum} out of total {v_abs_sum}");

            // fill the kernel matrix
            for (i, z) in v.iter().enumerate() {
                if !z.is_zero() {
                    kernel_octaves[cur_octave].add_triplet(k % buckets_per_octave, i, *z);
                }
            }
        }

        // let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // let mut x_fft = v
        //         .iter()
        //         .map(|f| rustfft::num_complex::Complex32::new(*f, 0.0))
        //         .collect::<Vec<rustfft::num_complex::Complex32>>();

        //     let mut planner = FftPlanner::new();
        //     let fft = planner.plan_fft_forward(v.len());
        //     let inv_fft = planner.plan_fft_inverse(v.len());

        //     fft.process(&mut x_fft);
        //     x_fft.iter_mut().for_each(|z| *z /= (v.len() as f32).sqrt());
        //     dbg!(&x_fft);
        //     for i in (1 + x_fft.len() / 2 / 2)..(x_fft.len() - x_fft.len() / 2 / 2) {
        //         x_fft[i] = num_complex::Complex::zero();
        //     }
        //     let idx = x_fft.len() / 2 / 2;
        //     //x_fft[idx].im = 0.0;
        //     dbg!(&x_fft);
        //     inv_fft.process(&mut x_fft);
        //     x_fft.iter_mut().for_each(|z| *z /= (v.len() as f32).sqrt());
        //     dbg!(&x_fft);
        // panic!();

        kernel_octaves
            .iter()
            .map(|m| m.to_csr())
            .collect::<Vec<sprs::CsMat<num_complex::Complex<f32>>>>()
    }

    fn resample(&self, v: &[f32], factor: usize) -> Vec<f32> {
        let mut x_fft = v
            .iter()
            .map(|f| rustfft::num_complex::Complex32::new(*f, 0.0))
            .collect::<Vec<rustfft::num_complex::Complex32>>();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(v.len());
        let inv_fft = planner.plan_fft_inverse(v.len());

        fft.process(&mut x_fft);
        //0..(1 + x_fft.len()) ()..()
        for i in (1 + x_fft.len() / factor / 2)..(x_fft.len() - x_fft.len() / factor / 2) {
            x_fft[i] = num_complex::Complex::zero();
        }
        inv_fft.process(&mut x_fft);

        x_fft
            .iter()
            .step_by(factor)
            .map(|z| z.re / v.len() as f32)
            .collect::<Vec<f32>>()
    }

    pub fn calculate_vqt_instant_in_db(&mut self, x: &[f32]) -> Vec<f32> {
        // TODO: we are doing a lot of unnecessary ffts here, just because the interface of the resampler
        // neither allows us to reuse the same frame for subsequent downsamplings, nor allows us to do the
        // fft ourselves.

        // let x = self.test_create_sines(self.t_diff);
        // self.t_diff += 0.0002;

        //dbg!(reduced_n_fft);

        let mut x_vqt = vec![Complex32::zero(); self.buckets_per_octave * self.octaves];
        for cur_octave in 0..self.octaves {
            let cur_resampling_factor = 1 << (self.octaves - 1 - cur_octave);
            let cur_fft_length = self.n_fft >> cur_octave;

            let x_selection = &x[(self.n_fft - cur_fft_length)..];
            //dbg!(cur_resampling_factor);
            //dbg!(x_selection.len());

            let vv = self.resample(&x_selection, cur_resampling_factor);

            // calculate the fft over the scaled current octave
            let mut x_fft = vv
                .iter()
                .map(|f| rustfft::num_complex::Complex32::new(*f, 0.0))
                .collect::<Vec<rustfft::num_complex::Complex32>>();
            //dbg!(max(&std::iter::zip(x_selection.iter().copied(), vv.iter().copied()).map(|(a, b)| (a-b).abs()).collect::<Vec<f32>>()));
            //dbg!(std::iter::zip(x_selection.iter().copied(), vv.iter().copied()).take(10).collect::<Vec<(f32,f32)>>());
            self.fft.process(&mut x_fft);
            //dbg!(self.vqt_kernel[cur_octave].shape(), x_fft.len());
            sprs::prod::mul_acc_mat_vec_csr(
                self.vqt_kernel[cur_octave].view(),
                x_fft,
                &mut x_vqt[(cur_octave * self.buckets_per_octave)
                    ..((cur_octave + 1) * self.buckets_per_octave)],
            );
        }

        // TODO: harmonic/percussion source separation possible with our one-sided spectrum?

        // signal fft'd

        // amp -> db conversion
        let ref_power: f32 = 1.0.powf(2.0);
        let a_min: f32 = 1e-6.powf(2.0);
        let top_db: f32 = 60.0;

        let power: Vec<f32> = x_vqt
            .iter()
            .map(|z| (z.abs() * z.abs()))
            .collect::<Vec<f32>>();
        #[allow(unused_variables)]
        let abs: Vec<f32> = x_vqt.iter().map(|z| (z.abs())).collect::<Vec<f32>>();

        let mut log_spec = power
            .iter()
            .map(|x| 10.0 * x.max(a_min).log10() - 10.0 * ref_power.max(a_min).log10())
            .collect::<Vec<f32>>();
        let log_spec_max = max(&log_spec);
        let log_spec_min = min(&log_spec);
        log_spec.iter_mut().for_each(|x| {
            if *x < log_spec_max - top_db {
                *x = log_spec_max - top_db
            }
        });

        trace!(
            "log_spec min {}, max {}, shifting to 0...",
            log_spec_min,
            log_spec_max,
        );

        // cut off at 0.0, and don't let it pass top_db
        log_spec.iter_mut().for_each(|x| {
            if log_spec_min > 0.0 {
                *x -= log_spec_min
            } else if *x < 0.0 {
                *x = 0.0
            }
        });

        // let x_vqt: Vec<f32> = x_vqt
        //     .iter()
        //     .map(|z| (z.abs() * z.abs()).log(10.0).max(-2.0) + 2.0)
        //     .collect();

        // TODO: median filter

        log_spec
        //let spec_max = max(&abs);
        //abs.iter_mut().for_each(|x| *x /= spec_max);
        //abs
    }
}
