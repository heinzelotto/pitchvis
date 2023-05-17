use log::debug;
use num_complex::{Complex32, ComplexFloat};
use rustfft::num_traits::Zero;
use rustfft::{Fft, FftPlanner};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::ops::DivAssign;
use std::sync::Arc;
use std::time::Duration;

/// Returns the maximum value in a slice of floats.
/// This function is necessary because floats do not implement the Ord trait.
/// It assumes that there are no NaN values in the slice.
///
/// # Arguments
///
/// * `sl` - A slice of f32 values.
///
/// # Returns
///
/// * The maximum value in the slice.
fn max(sl: &[f32]) -> f32 {
    sl.iter()
        .fold(f32::MIN, |cur, x| if *x > cur { *x } else { cur })
}

/// Returns the minimum value in a slice of floats.
/// This function is necessary because floats do not implement the Ord trait.
/// It assumes that there are no NaN values in the slice.
///
/// # Arguments
///
/// * `sl` - A slice of f32 values.
///
/// # Returns
///
/// * The minimum value in the slice.
fn min(sl: &[f32]) -> f32 {
    sl.iter()
        .fold(f32::MAX, |cur, x| if *x < cur { *x } else { cur })
}

/// A power of two - sized fft window.
#[derive(Debug, Clone, Copy)]
pub struct UnscaledWindow {
    sufficient_n_fft_size: usize,
    window: (usize, usize),
}

/// The `CqtKernel` struct represents a Constant Q Transform (CQT) kernel,
/// consisting of a filter bank and corresponding windows.
pub struct CqtKernel {
    pub filter_bank: Vec<sprs::CsMat<Complex32>>,
    pub windows: CqtWindows,
}

/// Parameters used to create a single filter in the CQT filter bank.
#[derive(Debug, Clone, Copy)]
struct FilterParams {
    freq: f32,
    window_length: f32,
}

pub struct CqtWindows {
    window_center: f32,
    grouped_by_sr_scaling: Vec<(usize, UnscaledWindow, Vec<FilterParams>)>,
}

/// The `Cqt` struct represents a Constant Q Transform (CQT), a type of spectral transform.
/// The CQT is a time-frequency representation where the frequency bins are geometrically spaced.
pub struct Cqt {
    /// The sample rate (in Hz) of the input audio signal.
    _sr: usize,

    /// The number of samples in the longest Fast Fourier Transform (FFT). This will be decimated
    /// for higher octaves.
    pub n_fft: usize,

    /// The minimum frequency (in Hz) of the lowest note analyzed in the Constant-Q Transform (CQT).
    _min_freq: f32,

    /// The resolution of the CQT, defined in terms of the number of frequency bins per octave.
    /// This value must be a multiple of 12, reflecting the 12 semitones in a musical octave.
    buckets_per_octave: usize,

    /// The total range, in octaves, over which the CQT is computed.
    octaves: usize,

    /// A quantile value used to determine the sparsity of the CQT kernel. A higher value results
    /// in a sparser representation, with only the most impactful frequency bins being used.
    _sparsity_quantile: f32,

    /// A value that determines the quality of the CQT. A higher value results in a more accurate
    /// representation, at the cost of greater time smearing.
    _quality: f32,

    /// A parameter used in the calculation of the CQT, which determines the amount of frequency
    /// smearing in lower octaves. A higher value results in more smearing, which can be useful
    /// for analyzing signals with time-varying frequency content where more time resolution is
    /// desired.
    _gamma: f32,

    /// The CQT kernel, which is a precomputed filter bank used in the computation of the CQT.
    cqt_kernel: CqtKernel,

    /// The delay introduced by the analysis process. This is the amount of time between when a
    /// signal is input to the system and when its CQT is available.
    pub delay: Duration,

    /// A map from FFT sizes to precomputed FFT objects. Each FFT object is used to resample the
    /// input signal to the corresponding size. The map contains a pair of FFTs for each size:
    /// one for forward transformation and one for inverse transformation.
    resample_ffts: HashMap<
        usize,
        (
            std::sync::Arc<dyn rustfft::Fft<f32>>,
            std::sync::Arc<dyn rustfft::Fft<f32>>,
        ),
    >,

    #[allow(dead_code)]
    t_diff: f32,
}

/// Creates a new `Cqt` instance.
///
/// # Arguments
///
/// * `sr` - The sample rate of the input audio signal.
/// * `n_fft` - The number of samples in the longest FFT.
/// * `min_freq` - The minimum frequency of the lowest note analyzed in the CQT.
/// * `buckets_per_octave` - The resolution of the CQT in bins per octave.
/// * `octaves` - The total range in octaves of the CQT.
/// * `sparsity_quantile` - The sparsity quantile.
/// * `quality` - The quality.
/// * `gamma` - The gamma.
///
/// # Returns
///
/// * A new `Cqt` instance.
impl Cqt {
    //#green

    #[allow(dead_code)]
    fn test_create_sines(&self, t_diff: f32) -> Vec<f32> {
        let mut wave = vec![0.0; self.n_fft];

        for f in ((12 * (0) + 0)..(12 * (self.octaves - 1) + 12))
            .step_by(7)
            .map(|p| self._min_freq * (2.0).powf(p as f32 / 12.0))
        {
            //let f = 880.0 * 2.0.powf(1.0/12.0);
            for i in 0..wave.len() {
                let amp = (((i as f32 + t_diff * self._sr as f32) * 2.0 * PI / self._sr as f32)
                    * f)
                    .sin()
                    / 12.0;
                wave[i] += amp;
            }
        }

        wave
    }
    //#

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
        let (cqt_kernel, delay) = Self::cqt_kernel(
            sr,
            n_fft,
            min_freq,
            buckets_per_octave,
            octaves,
            sparsity_quantile,
            quality,
            gamma,
        );

        println!("CQT Analysis delay: {} ms.", delay.as_millis());

        // TODO: get the info on which ffts are needed first, then create them, then also use them for creation of the cqt_kernel. Currently, the cqt_kernel creates its own ffts and only afterwards provides us with the info on which ffts are needed.

        // prepare resample ffts
        let mut resample_ffts = HashMap::new();
        let mut planner = FftPlanner::new();
        for wnd in cqt_kernel.windows.grouped_by_sr_scaling.iter() {
            let cur_unscaled_fft_length = wnd.1.sufficient_n_fft_size;
            if !resample_ffts.contains_key(&cur_unscaled_fft_length) {
                // dbg!(cur_unscaled_fft_length);
                resample_ffts.insert(
                    cur_unscaled_fft_length,
                    (
                        planner.plan_fft_forward(cur_unscaled_fft_length),
                        planner.plan_fft_inverse(cur_unscaled_fft_length), // actually not needed
                    ),
                );
            }

            let cur_scaled_fft_length = wnd.1.sufficient_n_fft_size / wnd.0;
            if !resample_ffts.contains_key(&cur_scaled_fft_length) {
                // dbg!(cur_scaled_fft_length);
                resample_ffts.insert(
                    cur_scaled_fft_length,
                    (
                        planner.plan_fft_forward(cur_scaled_fft_length),
                        planner.plan_fft_inverse(cur_scaled_fft_length),
                    ),
                );
            }
        }

        Self {
            _sr: sr,
            n_fft,
            _min_freq: min_freq,
            buckets_per_octave,
            octaves,
            _sparsity_quantile: sparsity_quantile,
            _quality: quality,
            _gamma: gamma,
            cqt_kernel,
            delay,
            resample_ffts,
            t_diff: 0.0,
        }
    }

    //#orange

    /// Groups the tuples (frequency, window_size) of the filter bank according to the power of two fft that they will be applied to.
    /// It is made sure that the fft size yields a large enough nyquist frequency to cover the highest frequency in the group, even when downsized.
    fn group_window_sizes(
        sr: usize,
        n_fft: usize,
        freqs: &Vec<f32>,
        window_lengths: &Vec<f32>,
    ) -> CqtWindows {
        let annotated_f_w = freqs
            .iter()
            .zip(window_lengths.iter())
            .map(|(f, w)| {
                let (maximum_sr_scaling_factor_for_f, _minimum_scaled_sr_for_f) = {
                    // because of the Gibbs phenomenon appearing in our resampling method, we set
                    // the minimum scaled sr to just a bit higher than theoretically needed one.
                    // This is a bit of a hack, but it works.
                    let grace_factor = 1.15;
                    let minimum_scaled_sr_for_f = (f * 2.0 * grace_factor).ceil() as usize;
                    // find maximum k so that minimum_scaled_sr_for_f is smaller than sr / 2^k
                    let k = (sr as f32 / minimum_scaled_sr_for_f as f32).log2().floor() as usize;
                    let maximum_sr_scaling_factor_for_f = 2.0f32.powf(k as f32);
                    let minimum_scaled_sr_for_f = sr as f32 / maximum_sr_scaling_factor_for_f;

                    (maximum_sr_scaling_factor_for_f, minimum_scaled_sr_for_f)
                };

                let (_maximum_n_fft_scaling_factor_for_w, minimum_needed_window_size) = {
                    let k = (n_fft as f32 / *w as f32).log2().floor() as usize;
                    let maximum_n_fft_scaling_factor_for_w = 2.0f32.powf(k as f32);
                    let minimum_needed_window_size =
                        n_fft / maximum_n_fft_scaling_factor_for_w as usize;

                    (
                        maximum_n_fft_scaling_factor_for_w,
                        minimum_needed_window_size,
                    )
                };

                let _scaled_window_size =
                    minimum_needed_window_size / maximum_sr_scaling_factor_for_f as usize;

                (
                    f,
                    w,
                    maximum_sr_scaling_factor_for_f,
                    minimum_needed_window_size,
                )
            })
            .collect::<Vec<_>>();

        for x in &annotated_f_w {
            debug!(
                "{:5.0}, {:5.0}, {:5.0}, {:5.0}, {:5.0}",
                x.0,
                x.1,
                x.2,
                x.3,
                x.3 as f32 / x.2
            );
        }

        let max_window_length = window_lengths.first().unwrap();
        let window_center = n_fft as f32 - max_window_length / 2.0;

        // partition by sr_scaling
        let mut partitions = Vec::new();
        let mut tmp = annotated_f_w.clone();
        while tmp.len() > 0 {
            let (cur, rem) = tmp.iter().partition(|x| x.2 == tmp[0].2);
            partitions.push(cur);
            tmp = rem;
        }

        let mut grouped: Vec<(usize, UnscaledWindow, Vec<FilterParams>)> = Vec::new();
        for p in partitions.iter() {
            let sufficient_n_fft_size = p.iter().map(|x| x.3 as usize).max().unwrap();

            let sr_scaling = p[0].2 as usize;
            // window size is the minimum window size needed to cover the highest frequency in the group. This chunk will be taken from the signal and then resampled by sr_scaling.
            let unscaled_window_size = p[0].3 as usize;
            assert_eq!(unscaled_window_size, sufficient_n_fft_size);
            let unscaled_window: UnscaledWindow = UnscaledWindow {
                sufficient_n_fft_size,
                window: if (window_center + (unscaled_window_size as f32) / 2.0) < (n_fft as f32) {
                    (
                        (window_center - (unscaled_window_size as f32) / 2.0) as usize,
                        (window_center + (unscaled_window_size as f32) / 2.0) as usize,
                    )
                } else {
                    (n_fft - unscaled_window_size, n_fft)
                },
            };
            debug!(
                "unscaled window {:?}, sr_scaling: {sr_scaling}, ratio: {:5.0}",
                unscaled_window,
                unscaled_window.sufficient_n_fft_size as f32 / sr_scaling as f32,
            );
            let group = p
                .iter()
                .map(|x| {
                    let (f, w, _, _) = x;
                    FilterParams {
                        freq: **f,
                        window_length: **w,
                    }
                })
                .collect::<Vec<_>>();
            grouped.push((sr_scaling, unscaled_window, group));
        }

        CqtWindows {
            window_center,
            grouped_by_sr_scaling: grouped,
        }
    }
    //#

    //#blue
    fn calculate_filter(
        sr: usize,
        sparsity_quantile: f32,
        sr_scaling: usize,
        filter_params: FilterParams,
        fft_window: UnscaledWindow,
        window_center: f32,
        fft: &Arc<dyn Fft<f32>>,
    ) -> Vec<num_complex::Complex32> {
        let scaled_freq = filter_params.freq * sr_scaling as f32;
        let scaled_window_length = filter_params.window_length / sr_scaling as f32;
        let scaled_window_length_rounded = scaled_window_length.round() as usize;
        let scaled_window_center = (window_center - fft_window.window.0 as f32) / sr_scaling as f32;
        let scaled_window_center_rounded = scaled_window_center.floor() as usize;
        let scaled_n_fft = fft_window.sufficient_n_fft_size / sr_scaling;

        assert!(scaled_window_length_rounded <= scaled_n_fft);

        let window = apodize::hanning_iter(scaled_window_length_rounded).collect::<Vec<f64>>();

        let mut v = vec![Complex32::zero(); scaled_n_fft];
        for i in 0..scaled_window_length_rounded {
            v[scaled_window_center_rounded - scaled_window_length_rounded / 2 + i] = 1.0
                / scaled_window_length_rounded as f32
                * (window[i] as f32)
                * (num_complex::Complex32::i()
                    * 2.0
                    * PI
                    * (i as f32/*- window_lengths[k] / 2.0*/)
                    * scaled_freq
                    / (sr as f32))
                    .exp();
        }

        // normalize windowed wavelet in time space
        let norm_1: f32 = v.iter().map(|z| z.abs()).sum();
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
        debug!("for freq {} erased {cnt} points below {cutoff_value} with sum {accum} out of total {v_abs_sum}", filter_params.freq);

        assert_eq!(v.len(), fft_window.sufficient_n_fft_size / sr_scaling);

        v
    }
    //#

    //#red

    /// Calculates the CQT kernel.
    ///
    /// # Arguments
    ///
    /// * `sr` - The sample rate of the input audio signal.
    /// * `n_fft` - The number of samples in the longest FFT.
    /// * `min_freq` - The minimum frequency of the lowest note analyzed in the CQT.
    /// * `buckets_per_octave` - The resolution of the CQT in bins per octave.
    /// * `octaves` - The total range in octaves of the CQT.
    /// * `sparsity_quantile` - The sparsity quantile.
    /// * `quality` - The quality.
    /// * `gamma` - The gamma.
    ///
    /// # Returns
    ///
    /// * A tuple consisting of the `CqtKernel` and the delay as `Duration`.
    fn cqt_kernel(
        sr: usize,
        n_fft: usize,
        min_freq: f32,
        buckets_per_octave: usize,
        octaves: usize,
        sparsity_quantile: f32,
        quality: f32,
        gamma: f32,
    ) -> (CqtKernel, Duration) {
        let freqs = (0..(buckets_per_octave * octaves))
            .map(|k| min_freq * 2.0_f32.powf(k as f32 / buckets_per_octave as f32))
            .collect::<Vec<f32>>();

        let highest_frequency = *freqs.last().unwrap();
        let nyquist_frequency = sr / 2;
        if highest_frequency > nyquist_frequency as f32 {
            panic!(
                "The highest frequency of the CQT kernel is {} Hz, but the Nyquist frequency is {} Hz.",
                highest_frequency, nyquist_frequency
            );
        }

        // calculate filter window sizes
        let r = 2.0.powf(1.0 / buckets_per_octave as f32);
        // alpha is constant and such that (1+a)*f_{k-1} = (1-a)*f_{k+1}
        let alpha = (r.powf(2.0) - 1.0) / (r.powf(2.0) + 1.0);
        #[allow(non_snake_case)]
        let Q = quality / alpha;
        let window_lengths = freqs
            .iter()
            .map(|f_k| Q * sr as f32 / (f_k + gamma / alpha))
            .collect::<Vec<f32>>();

        if window_lengths[0] > n_fft as f32 {
            panic!(
                "The window length of the longest filter is {} samples, but the longest FFT is {} samples.",
                window_lengths[0], n_fft
            );
        }

        let cqt_windows = Self::group_window_sizes(sr, n_fft, &freqs, &window_lengths);
        for v in cqt_windows.grouped_by_sr_scaling.iter() {
            debug!("sr scaling {}, window: {:?}:", v.0, v.1);
            for (i, fp) in v.2.iter().enumerate() {
                debug!("{i} {:?}", fp);
            }
        }

        let mut planner = FftPlanner::new();
        let kernel_octaves = cqt_windows
            .grouped_by_sr_scaling
            .iter()
            .map(|(sr_scaling, wnd, filter_params_vec)| {
                let scaled_n_fft = wnd.sufficient_n_fft_size / sr_scaling;

                let mut mat = sprs::TriMat::new((filter_params_vec.len(), scaled_n_fft));
                let fft = planner.plan_fft_forward(scaled_n_fft);
                debug!("planning fft of size {}", scaled_n_fft);

                for (idx, filter_params) in filter_params_vec.iter().enumerate() {
                    let filter = Self::calculate_filter(
                        sr,
                        sparsity_quantile,
                        *sr_scaling,
                        *filter_params,
                        *wnd,
                        cqt_windows.window_center,
                        &fft,
                    );
                    // fill the kernel matrix
                    for (i, z) in filter.iter().enumerate() {
                        if !z.is_zero() {
                            mat.add_triplet(idx, i, *z / scaled_n_fft as f32 * (sr as f32).sqrt());
                        }
                    }
                }

                mat
            })
            .collect::<Vec<sprs::TriMat<Complex32>>>();

        let kernel = kernel_octaves
            .iter()
            .map(|m| m.to_csr())
            .collect::<Vec<sprs::CsMat<num_complex::Complex<f32>>>>();

        let delay = Duration::from_secs_f32((n_fft as f32 - cqt_windows.window_center) / sr as f32);

        (
            CqtKernel {
                filter_bank: kernel,
                windows: cqt_windows,
            },
            delay,
        )
    }
    //#

    // /// Resamples a given input signal by a specified factor. The resampling is performed by
    // /// transforming the signal into the frequency domain using FFT, zeroing out the high-frequency
    // /// components, and then transforming it back to the time domain. The resulting time-domain signal
    // /// is then downsampled by the given factor.
    // ///
    // /// # Arguments
    // /// * `v`: The input signal to be resampled.
    // /// * `factor`: The resampling factor. The input signal will be downsampled by this factor.
    // ///
    // /// # Returns
    // /// A vector containing the resampled signal.
    // fn _resample(&self, v: &[f32], factor: usize) -> Vec<f32> {
    //     let mut v = vec![v.to_vec()];
    //     dbg!(v[0].len(), v[0].len() / factor);
    //     let res = rubato::FftFixedInOut::<f32>::new(
    //         v[0].len(),
    //         v[0].len() / factor,
    //         v[0].len() / factor,
    //         1,
    //     )
    //     .unwrap()
    //     .process(&v, None)
    //     .unwrap();

    //     res[0].to_vec()
    // }

    //#yellow

    fn resample(&self, v: &[f32], sr_scaling: usize) -> Vec<f32> {
        let fft_size = v.len();

        let mut x_fft = v
            .iter()
            .map(|f| rustfft::num_complex::Complex32::new(*f, 0.0))
            .collect::<Vec<rustfft::num_complex::Complex32>>();

        let (fft, inv_fft) = self.resample_ffts.get(&v.len()).unwrap();

        fft.process(&mut x_fft);
        for i in (1 + fft_size / sr_scaling / 2)..(fft_size - fft_size / sr_scaling / 2) {
            x_fft[i] = num_complex::Complex::zero();
        }
        inv_fft.process(&mut x_fft);

        x_fft
            .iter()
            .step_by(sr_scaling)
            .map(|z| z.re / fft_size as f32)
            .collect::<Vec<f32>>()
    }
    //#

    //#pink

    /// Calculates the Constant-Q Transform (CQT) of the given input signal at a specific time instant.
    /// The result is given in dB scale. The function performs multiple FFTs on resampled versions of
    /// the input signal and applies the precomputed CQT kernel to each FFT output to obtain the CQT.
    /// It then converts the results into the dB scale and combines them into a single output vector.
    ///
    /// # Arguments
    /// * `x`: The input signal for which the CQT is to be calculated.
    ///
    /// # Returns
    /// A vector containing the CQT of the input signal in dB scale.
    pub fn calculate_cqt_instant_in_db(&mut self, x: &[f32]) -> Vec<f32> {
        // TODO: we are doing a lot of unnecessary ffts here, just because the interface of the resampler
        // neither allows us to reuse the same frame for subsequent downsamplings, nor allows us to do the
        // fft ourselves.

        // let x = self.test_create_sines(self.t_diff);
        // self.t_diff += 0.0002;

        //dbg!(reduced_n_fft);

        let mut x_cqt = vec![Complex32::zero(); self.buckets_per_octave * self.octaves];
        let mut offset = 0;
        for (wnd, filters) in self
            .cqt_kernel
            .windows
            .grouped_by_sr_scaling
            .iter()
            .zip(self.cqt_kernel.filter_bank.iter())
        {
            assert_eq!(wnd.2.len(), filters.shape().0);

            let cur_resampling_factor = wnd.0;
            let cur_unscaled_fft_length = wnd.1.sufficient_n_fft_size;
            let cur_scaled_fft_length = cur_unscaled_fft_length / cur_resampling_factor;

            let (window_begin, window_end) = wnd.1.window;
            assert_eq!(window_end - window_begin, cur_unscaled_fft_length);
            let x_selection = &x[window_begin..window_end];
            // dbg!(
            //     x_selection.len(),
            //     cur_unscaled_fft_length,
            //     cur_resampling_factor
            // );

            let vv = self.resample(x_selection, cur_resampling_factor);

            let mut x_fft = vv
                .iter()
                .map(|f| rustfft::num_complex::Complex32::new(*f, 0.0))
                .collect::<Vec<rustfft::num_complex::Complex32>>();
            //dbg!(max(&std::iter::zip(x_selection.iter().copied(), vv.iter().copied()).map(|(a, b)| (a-b).abs()).collect::<Vec<f32>>()));
            //dbg!(std::iter::zip(x_selection.iter().copied(), vv.iter().copied()).take(10).collect::<Vec<(f32,f32)>>());
            assert_eq!(cur_scaled_fft_length, x_fft.len());
            //dbg!(cur_scaled_fft_length);
            self.resample_ffts
                .get(&cur_scaled_fft_length)
                .unwrap()
                .0
                .process(&mut x_fft); // TODO: just get the fft from the resampler, ?if possible
                                      // dbg!(
                                      //     filters.shape(),
                                      //     x_fft.len(),
                                      //     offset,
                                      //     offset + filters.shape().0
                                      // );
            sprs::prod::mul_acc_mat_vec_csr(
                filters.view(),
                x_fft,
                &mut x_cqt[offset..(offset + filters.shape().0)],
            );
            offset += filters.shape().0;
        }

        let power: Vec<f32> = x_cqt
            .iter()
            .map(|z| (z.abs() * z.abs()))
            .collect::<Vec<f32>>();

        // TODO: harmonic/percussion source separation possible with our one-sided spectrum?

        // amp -> db conversion
        Self::power_to_db(&power)
        //Self::power_normalized(&power)
    }
    //#

    fn power_to_db(power: &[f32]) -> Vec<f32> {
        let ref_power: f32 = 0.3.powf(2.0);
        let a_min: f32 = 1e-6.powf(2.0);
        let top_db: f32 = 60.0;

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

        debug!(
            "log_spec min {}, max {}, shifting to 0...",
            log_spec_min, log_spec_max,
        );

        // cut off at 0.0, and don't let it pass top_db
        log_spec.iter_mut().for_each(|x| {
            if log_spec_min > 0.0 {
                *x -= log_spec_min
            } else if *x < 0.0 {
                *x = 0.0
            }
        });

        log_spec
    }

    #[allow(dead_code)]
    fn power_normalized(power: &[f32]) -> Vec<f32> {
        let spec_max = max(&power);
        dbg!(spec_max);
        power
            .iter()
            .map(|x| *x / spec_max * 50.0) //spec_max);
            .collect::<Vec<_>>()
    }
}
