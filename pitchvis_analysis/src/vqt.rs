use log::{debug, error};
use num_complex::{Complex32, ComplexFloat};
use rustfft::num_traits::Zero;
use rustfft::{Fft, FftPlanner};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::ops::DivAssign;
use std::sync::Arc;
use std::time::Duration;

use crate::util::{self, arg_max};

#[derive(Debug, Clone)]
pub struct VqtRange {
    /// The minimum frequency (in Hz) of the lowest note analyzed in the variable-Q Transform
    /// (VQT). Usually A1 (55Hz).
    pub min_freq: f32,

    /// The total range, in octaves, over which the VQT is computed.
    pub octaves: u8,

    /// The resolution of the VQT, defined in terms of the number of frequency bins per octave.
    /// This value must be a multiple of 12, reflecting the 12 semitones in a musical octave.
    pub buckets_per_octave: u16,
}

impl VqtRange {
    /// Returns the number of frequency bins in the VQT kernel.
    pub fn n_buckets(&self) -> usize {
        self.buckets_per_octave as usize * self.octaves as usize
    }
}

// increasing BUCKETS_PER_SEMITONE or Q will improve frequency resolution at cost of time resolution,
// increasing GAMMA will improve time resolution at lower frequencies.
#[derive(Debug, Clone)]
pub struct VqtParameters {
    /// The sample rate (in Hz) of the input audio signal.
    pub sr: u32, // TODO: ?can make f32?

    /// The number of samples in the longest Fast Fourier Transform (FFT). This will be decimated
    /// for higher octaves.
    pub n_fft: usize,

    /// The range of the VQT, min freq, span, and resolution.
    pub range: VqtRange,

    /// A quantile value used to determine the sparsity of the VQT kernel. A higher value results
    /// in a sparser representation, with only the most impactful frequency bins being used.
    pub sparsity_quantile: f32,

    /// A value that determines the quality of the VQT. A higher value results in a more accurate
    /// representation, at the cost of greater time smearing.
    pub quality: f32,

    /// A parameter used in the calculation of the VQT, which determines the amount of frequency
    /// smearing in lower octaves. A higher value results in more smearing, which can be useful
    /// for analyzing signals with time-varying frequency content where more time resolution is
    /// desired.
    pub gamma: f32,
}

/// A power of two - sized fft window.
#[derive(Debug, Clone, Copy)]
pub struct UnscaledWindow {
    sufficient_n_fft_size: usize,
    window: (usize, usize),
}

/// The `VqtKernel` struct represents a Variable Q Transform (VQT) kernel,
/// consisting of a filter bank and corresponding windows.
pub struct VqtKernel {
    pub filter_bank: Vec<sprs::CsMat<Complex32>>,
    pub windows: VqtWindows,
}

/// Parameters used to create a single filter in the VQT filter bank.
#[derive(Debug, Clone, Copy)]
struct FilterParams {
    freq: f32,
    window_length: f32,
}

/// A group of filters that can all be applied on the same (downscaled) fft size, without losing information
#[derive(Debug)]
struct FilterGrouping {
    /// The maximum downsampling factor that can be applied to the signal to still cover the frequency of this filter.
    sr_downscaling_factor: usize,

    /// The window that will be taken from the signal and resampled by sr_downscaling_factor.
    unscaled_window: UnscaledWindow,

    /// The filters that can be applied on this window.
    filters: Vec<FilterParams>,
}

pub struct VqtWindows {
    window_center: f32,
    grouped_by_sr_scaling: Vec<FilterGrouping>,
}

struct Filter {
    v_frequency_domain: Vec<Complex32>,
    bandwidth_3db_in_hz: (f32, f32),
}

/// A pair of precomputed FFTs for a given size:
/// one for forward transformation and one for inverse transformation.
struct PrecomputedFft {
    /// The forward FFT.
    fwd_fft: Arc<dyn Fft<f32>>,

    /// The inverse FFT.
    inv_fft: Arc<dyn Fft<f32>>,
}

/// The `Vqt` struct represents a Variable Q Transform (VQT), a type of spectral transform.
/// It is based on the Constant Q Transform (CQT), a time-frequency representation where the
/// frequency bins are geometrically spaced. The variable Q Transform extends the CQT with an
/// additional parameter γ. A higher value of this parameter increases the amount of frequency
/// smearing in lower octaves but improves time resolution. Since the delay of the filter bank
/// is mostly determined by the lowest octave, this thus gives more control over the total signal
/// processing delay of the filter bank.
pub struct Vqt {
    /// Parameters that define the VQT kernel.
    params: VqtParameters,

    /// The VQT kernel, which is a precomputed filter bank used in the computation of the VQT.
    vqt_kernel: VqtKernel,

    /// The delay introduced by the analysis process. This is the amount of time between when a
    /// signal is input to the system and when its VQT is available.
    pub delay: Duration,

    /// A map from FFT sizes to precomputed FFT objects. Each FFT object is used to resample the
    /// input signal to the corresponding size. The map contains a pair of FFTs for each size:
    /// one for forward transformation and one for inverse transformation.
    resample_ffts: HashMap<usize, PrecomputedFft>,

    #[allow(dead_code)]
    t_diff: f32,
}

impl Vqt {
    pub fn params(&self) -> &VqtParameters {
        &self.params
    }
}

/// Creates a new `Vqt` instance.
///
/// # Arguments
///
/// * `sr` - The sample rate of the input audio signal.
/// * `n_fft` - The number of samples in the longest FFT.
/// * `min_freq` - The minimum frequency of the lowest note analyzed in the VQT.
/// * `buckets_per_octave` - The resolution of the VQT in bins per octave.
/// * `octaves` - The total range in octaves of the VQT.
/// * `sparsity_quantile` - The sparsity quantile.
/// * `quality` - The quality.
/// * `gamma` - The gamma.
///
/// # Returns
///
/// * A new `Vqt` instance.
impl Vqt {
    pub fn new(params: &VqtParameters) -> Self {
        let (vqt_kernel, delay) = Self::vqt_kernel(params);

        println!("VQT Analysis delay: {} ms.", delay.as_millis());

        // TODO: get the info on which ffts are needed first, then create them, then also use them for creation of the vqt_kernel. Currently, the vqt_kernel creates its own ffts and only afterwards provides us with the info on which ffts are needed.

        // prepare resample ffts
        let mut resample_ffts = HashMap::new();
        let mut planner = FftPlanner::new();
        for FilterGrouping {
            sr_downscaling_factor,
            unscaled_window,
            ..
        } in vqt_kernel.windows.grouped_by_sr_scaling.iter()
        {
            let cur_unscaled_fft_length = unscaled_window.sufficient_n_fft_size;
            resample_ffts
                .entry(cur_unscaled_fft_length)
                .or_insert_with(|| {
                    PrecomputedFft {
                        fwd_fft: planner.plan_fft_forward(cur_unscaled_fft_length),
                        inv_fft: planner.plan_fft_inverse(cur_unscaled_fft_length), // actually not needed
                    }
                });

            let cur_scaled_fft_length =
                unscaled_window.sufficient_n_fft_size / sr_downscaling_factor;
            resample_ffts
                .entry(cur_scaled_fft_length)
                .or_insert_with(|| PrecomputedFft {
                    fwd_fft: planner.plan_fft_forward(cur_scaled_fft_length),
                    inv_fft: planner.plan_fft_inverse(cur_scaled_fft_length),
                });
        }

        Self {
            params: params.clone(),
            vqt_kernel,
            delay,
            resample_ffts,
            t_diff: 0.0,
        }
    }

    //#orange
    /// Groups the tuples (frequency, window_size) of the filter bank according to the power of two fft that
    /// they will be applied to. It is made sure that the fft size yields a large enough nyquist frequency
    /// to cover the highest frequency in the group, even when downsized.
    ///
    /// This enables us to make the application of each filter more efficient by computing it on a maximally
    ///  downsampled version of the signal while not losing information.
    fn group_window_sizes(
        sr: u32,
        n_fft: usize,
        freqs: &[f32],
        window_lengths: &[f32],
    ) -> VqtWindows {
        /// Annotated version of the filter parameters, containing computed constraints
        #[derive(Debug, Clone, Copy)]
        struct AnnontatedFilterParams {
            /// The filter parameters, frequency and window size.
            params: FilterParams,

            /// The maximum downsampling factor that can be applied to the signal to still cover the frequency of this filter.
            /// This directly depends on f
            maximum_sr_downscaling_factor_for_f: f32,

            /// The minimum window size needed to cover the frequency of this filter. This depends
            /// on w, which was calculated as a function of f and the VQT quality factor (lower
            /// frequency resolution at lower frequencies to keep window sizes and thus time smear
            /// smaller).
            minimum_needed_window_size: usize,
        }
        let annotated_f_w: Vec<AnnontatedFilterParams> = freqs
            .iter()
            .zip(window_lengths.iter())
            .map(|(f, w)| {
                let maximum_sr_downscaling_factor_for_f = {
                    // because of the Gibbs phenomenon appearing in our resampling method, we set
                    // the minimum scaled sr to just a bit higher than theoretically needed one.
                    // This is a bit of a hack, but it works.
                    let grace_factor = 1.15;
                    let minimum_scaled_sr_for_f = (f * 2.0 * grace_factor).ceil() as usize;
                    // find maximum k so that minimum_scaled_sr_for_f is smaller than sr / 2^k
                    let k = (sr as f32 / minimum_scaled_sr_for_f as f32).log2().floor() as usize;

                    // let minimum_scaled_sr_for_f = sr as f32 / maximum_sr_scaling_factor_for_f;

                    2.0f32.powf(k as f32)
                };

                let minimum_needed_window_size = {
                    let k = (n_fft as f32 / w).log2().floor() as usize;
                    let maximum_n_fft_downscaling_factor_for_w = 2.0f32.powf(k as f32);

                    n_fft / maximum_n_fft_downscaling_factor_for_w as usize
                };

                // let _scaled_window_size =
                //     minimum_needed_window_size / maximum_sr_scaling_factor_for_f as usize;

                AnnontatedFilterParams {
                    params: FilterParams {
                        freq: *f,
                        window_length: *w,
                    },
                    maximum_sr_downscaling_factor_for_f,
                    minimum_needed_window_size,
                }
            })
            .collect();

        for x in &annotated_f_w {
            debug!(
                "f: {:5.1}, w: {:5.1}, max_sr_scaling_factor_for_f: {:5.0}, min_needed_window_size: {:5.0}, x.3/x.2: {:5.0}",
                x.params.freq,
                x.params.window_length,
                x.maximum_sr_downscaling_factor_for_f,
                x.minimum_needed_window_size,
                x.minimum_needed_window_size as f32 / x.maximum_sr_downscaling_factor_for_f
            );
        }

        let max_window_length = window_lengths.first().unwrap();
        let window_center = n_fft as f32 - max_window_length / 2.0;

        // partition by sr_scaling
        let mut partitions = Vec::new();
        let mut tmp = annotated_f_w.clone();
        while !tmp.is_empty() {
            let (cur, rem) = tmp.iter().partition(|x| {
                x.maximum_sr_downscaling_factor_for_f == tmp[0].maximum_sr_downscaling_factor_for_f
            });
            partitions.push(cur);
            tmp = rem;
        }

        let mut grouped: Vec<FilterGrouping> = Vec::new();
        for p in partitions.iter() {
            let sufficient_n_fft_size = p
                .iter()
                .map(|x| x.minimum_needed_window_size)
                .max()
                .unwrap();

            let sr_downscaling_factor = p[0].maximum_sr_downscaling_factor_for_f as usize;
            // window size is the minimum window size needed to cover the highest frequency in the group. This chunk will be taken from the signal and then resampled by sr_scaling.
            let unscaled_window_size = p[0].minimum_needed_window_size;
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
                "unscaled window {:?}, sr_scaling: {sr_downscaling_factor}, ratio: {:5.0}",
                unscaled_window,
                unscaled_window.sufficient_n_fft_size as f32 / sr_downscaling_factor as f32,
            );
            let group = p.iter().map(|x| x.params).collect::<Vec<_>>();
            grouped.push(FilterGrouping {
                sr_downscaling_factor,
                unscaled_window,
                filters: group,
            });
        }

        VqtWindows {
            window_center,
            grouped_by_sr_scaling: grouped,
        }
    }
    //#

    //#blue
    /// Calculates a single filter in the VQT filter bank.
    ///
    /// # Arguments
    ///
    /// * `window_center` - The center of the window in the time domain. We arrange the filters in the time domain
    /// such that all filters are centered around the same time instant.
    fn calculate_filter(
        sr: u32,
        sparsity_quantile: f32,
        sr_scaling: usize,
        filter_params: FilterParams,
        fft_window: UnscaledWindow,
        window_center: f32,
        fft: &Arc<dyn Fft<f32>>,
    ) -> Filter {
        let scaled_freq = filter_params.freq * sr_scaling as f32;
        let scaled_window_length = filter_params.window_length / sr_scaling as f32;
        let scaled_window_length_rounded = scaled_window_length.round() as usize;
        let scaled_window_center = (window_center - fft_window.window.0 as f32) / sr_scaling as f32;
        let scaled_window_center_rounded = scaled_window_center.floor() as usize;
        let scaled_n_fft = fft_window.sufficient_n_fft_size / sr_scaling;

        assert!(scaled_window_length_rounded <= scaled_n_fft);

        // create windowed wavelet. This is calculated as:
        // f(x) = 1 / window_length * window(x) * e^(2 * pi * i * f * x)
        // and it is centered around the center of the window. Everything is scaled by the sr_scaling factor.
        let window = apodize::hanning_iter(scaled_window_length_rounded)
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        let mut v_frequency_domain = vec![Complex32::zero(); scaled_n_fft];
        for i in 0..scaled_window_length_rounded {
            v_frequency_domain
                [scaled_window_center_rounded - scaled_window_length_rounded / 2 + i] = 1.0
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
        let norm_1: f32 = v_frequency_domain.iter().map(|z| z.norm()).sum();
        v_frequency_domain
            .iter_mut()
            .for_each(|z| z.div_assign(norm_1));

        // transform wavelets into frequency space
        fft.process(&mut v_frequency_domain);

        // the complex conjugate is what we later need
        v_frequency_domain = v_frequency_domain.iter_mut().map(|z| z.conj()).collect();

        let mut v_frequency_response = v_frequency_domain
            .iter()
            .map(|z| z.norm())
            .collect::<Vec<f32>>();

        let bandwidth_3db_in_hz =
            calculate_bandwidth(&v_frequency_response, sr as f32 / sr_scaling as f32);
        // dbg!(filter_params.freq, bandwidth);

        // filter all values smaller than some value and use sparse arrays
        v_frequency_response.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let v_abs_sum = v_frequency_response.iter().sum::<f32>();
        let mut accum = 0.0;
        let mut cutoff_idx = 0;
        while accum < (1.0 - sparsity_quantile) * v_abs_sum {
            accum += v_frequency_response[cutoff_idx];
            cutoff_idx += 1;
        }
        let cutoff_value = v_frequency_response[cutoff_idx - 1];
        //let cutoff_value = v_abs[(reduced_n_fft as f32 * sparsity_quantile) as usize];
        let mut cnt = 0;
        v_frequency_domain.iter_mut().for_each(|z| {
            if z.abs() < cutoff_value {
                *z = Complex32::zero();
                cnt += 1;
            }
        });
        debug!("for freq {} erased {cnt} points below {cutoff_value} with sum {accum} out of total {v_abs_sum}", filter_params.freq);

        assert_eq!(
            v_frequency_domain.len(),
            fft_window.sufficient_n_fft_size / sr_scaling
        );

        Filter {
            v_frequency_domain,
            bandwidth_3db_in_hz,
        }
    }
    //#

    //#red

    /// Calculates the VQT kernel.
    ///
    /// # Arguments
    ///
    /// * `sr` - The sample rate of the input audio signal.
    /// * `n_fft` - The number of samples in the longest FFT.
    /// * `min_freq` - The minimum frequency of the lowest note analyzed in the VQT.
    /// * `buckets_per_octave` - The resolution of the VQT in bins per octave.
    /// * `octaves` - The total range in octaves of the VQT.
    /// * `sparsity_quantile` - The sparsity quantile.
    /// * `quality` - The quality.
    /// * `gamma` - The gamma.
    ///
    /// # Returns
    ///
    /// * A tuple consisting of the `VqtKernel` and the delay as `Duration`.
    ///
    /// Note: it is checked that at high frequencies the filters cover the entire band of a
    /// semitone. If this is not satisfied, errors are logged.
    ///
    fn vqt_kernel(params: &VqtParameters) -> (VqtKernel, Duration) {
        let freqs = (0..(params.range.n_buckets()))
            .map(|k| params.range.min_freq * 2.0_f32.powf(k as f32 / params.range.buckets_per_octave as f32))
            .collect::<Vec<f32>>();

        let highest_frequency = *freqs.last().unwrap();
        let nyquist_frequency = params.sr / 2;
        if highest_frequency > nyquist_frequency as f32 {
            panic!(
                "The highest frequency of the VQT kernel is {} Hz, but the Nyquist frequency is {} Hz.",
                highest_frequency, nyquist_frequency
            );
        }

        // calculate filter window sizes
        let r = 2.0.powf(1.0 / params.range.buckets_per_octave as f32);
        // alpha is constant and such that (1+a)*f_{k-1} = (1-a)*f_{k+1}
        let alpha = (r.powf(2.0) - 1.0) / (r.powf(2.0) + 1.0);
        #[allow(non_snake_case)]
        let Q = params.quality / alpha;
        let window_lengths = freqs
            .iter()
            .map(|f_k| Q * params.sr as f32 / (f_k + params.gamma / alpha))
            .collect::<Vec<f32>>();

        if window_lengths[0] > params.n_fft as f32 {
            panic!(
                "The window length of the longest filter is {} samples, but the longest FFT is {} samples.",
                window_lengths[0], params.n_fft
            );
        }

        let vqt_windows =
            Self::group_window_sizes(params.sr, params.n_fft, &freqs, &window_lengths);
        for FilterGrouping {
            sr_downscaling_factor,
            unscaled_window,
            filters,
        } in vqt_windows.grouped_by_sr_scaling.iter()
        {
            debug!(
                "sr scaling {}, window: {:?}:",
                sr_downscaling_factor, unscaled_window
            );
            for (i, fp) in filters.iter().enumerate() {
                let window_length_in_s = fp.window_length / params.sr as f32;
                let wave_num = window_length_in_s * fp.freq;
                let bandwidth_in_hz = 1.0 / window_length_in_s;
                let bandwidth_in_semitones = 12.0 * (1.0 + bandwidth_in_hz / fp.freq).log2();
                debug!("{i}, f: {:.1}, window len: {:.1} [{:.2}ms], wave_num: {:.1}, bandwidth: {:.2}Hz [{:.2} semitones]", 
                fp.freq, fp.window_length, 1000.0 * window_length_in_s, wave_num, bandwidth_in_hz, bandwidth_in_semitones);
            }
        }

        let mut planner = FftPlanner::new();
        let kernel_octaves = vqt_windows
            .grouped_by_sr_scaling
            .iter()
            .map(
                |FilterGrouping {
                     sr_downscaling_factor,
                     unscaled_window,
                     filters,
                 }| {
                    let scaled_n_fft =
                        unscaled_window.sufficient_n_fft_size / sr_downscaling_factor;

                    let mut mat = sprs::TriMat::new((filters.len(), scaled_n_fft));
                    let fft = planner.plan_fft_forward(scaled_n_fft);
                    debug!("planning fft of size {}", scaled_n_fft);

                    let mut last_upper_bandwidth = 0.0;
                    for (idx, filter_params) in filters.iter().enumerate() {
                        let filter = Self::calculate_filter(
                            params.sr,
                            params.sparsity_quantile,
                            *sr_downscaling_factor,
                            *filter_params,
                            *unscaled_window,
                            vqt_windows.window_center,
                            &fft,
                        );
                        if last_upper_bandwidth != 0.0
                            && filter.bandwidth_3db_in_hz.0 > last_upper_bandwidth
                        {
                            error!(
                                "The bandwidth of the filter at index {} is ({:.2}, {:.2}) Hz, but \
                                the last filter's bandwidth ended at {:.2} Hz. This gap equates to \
                                {:.2}% of the current filter's bandwidth.",
                                idx,
                                filter.bandwidth_3db_in_hz.0,
                                filter.bandwidth_3db_in_hz.1,
                                last_upper_bandwidth,
                                100.0 * (filter.bandwidth_3db_in_hz.0 - last_upper_bandwidth)
                                / (filter.bandwidth_3db_in_hz.1 - filter.bandwidth_3db_in_hz.0)
                            );
                            last_upper_bandwidth = filter.bandwidth_3db_in_hz.1;
                        }

                        // fill the kernel matrix
                        for (i, z) in filter.v_frequency_domain.iter().enumerate() {
                            if !z.is_zero() {
                                mat.add_triplet(
                                    idx,
                                    i,
                                    *z / scaled_n_fft as f32 * (params.sr as f32).sqrt(),
                                );
                            }
                        }
                    }

                    mat
                },
            )
            .collect::<Vec<sprs::TriMat<Complex32>>>();

        let kernel = kernel_octaves
            .iter()
            .map(|m| m.to_csr())
            .collect::<Vec<sprs::CsMat<num_complex::Complex<f32>>>>();

        let delay = Duration::from_secs_f32(
            (params.n_fft as f32 - vqt_windows.window_center) / params.sr as f32,
        );

        (
            VqtKernel {
                filter_bank: kernel,
                windows: vqt_windows,
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

        let PrecomputedFft {
            fwd_fft, inv_fft, ..
        } = self.resample_ffts.get(&v.len()).unwrap();

        fwd_fft.process(&mut x_fft);
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

    /// Calculates the variable-Q Transform (VQT) of the given input signal at a specific time instant.
    /// The result is given in dB scale. The function performs multiple FFTs on resampled versions of
    /// the input signal and applies the precomputed VQT kernel to each FFT output to obtain the VQT.
    /// It then converts the results into the dB scale and combines them into a single output vector.
    ///
    /// # Arguments
    /// * `x`: The input signal for which the VQT is to be calculated.
    ///
    /// # Returns
    /// A vector containing the VQT of the input signal in dB scale.
    pub fn calculate_vqt_instant_in_db(&self, x: &[f32]) -> Vec<f32> {
        // TODO: we are doing a lot of unnecessary ffts here, just because the interface of the resampler
        // neither allows us to reuse the same frame for subsequent downsamplings, nor allows us to do the
        // fft ourselves.

        // let x = self.test_create_sines(self.t_diff);
        // self.t_diff += 0.0002;

        //dbg!(reduced_n_fft);

        let mut x_vqt = vec![
            Complex32::zero();
            self.params.range.n_buckets()
        ];
        let mut offset = 0;
        for (
            FilterGrouping {
                sr_downscaling_factor,
                unscaled_window,
                filters,
            },
            filter_matrix,
        ) in self
            .vqt_kernel
            .windows
            .grouped_by_sr_scaling
            .iter()
            .zip(self.vqt_kernel.filter_bank.iter())
        {
            assert_eq!(filters.len(), filter_matrix.shape().0);

            let cur_unscaled_fft_length = unscaled_window.sufficient_n_fft_size;
            let cur_scaled_fft_length = cur_unscaled_fft_length / sr_downscaling_factor;

            let (window_begin, window_end) = unscaled_window.window;
            assert_eq!(window_end - window_begin, cur_unscaled_fft_length);
            let x_selection = &x[window_begin..window_end];
            // dbg!(
            //     x_selection.len(),
            //     cur_unscaled_fft_length,
            //     cur_resampling_factor
            // );

            let vv = self.resample(x_selection, *sr_downscaling_factor);

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
                .fwd_fft
                .process(&mut x_fft); // TODO: just get the fft from the resampler, ?if possible
                                      // dbg!(
                                      //     filters.shape(),
                                      //     x_fft.len(),
                                      //     offset,
                                      //     offset + filters.shape().0
                                      // );
            sprs::prod::mul_acc_mat_vec_csr(
                filter_matrix.view(),
                x_fft,
                &mut x_vqt[offset..(offset + filter_matrix.shape().0)],
            );
            offset += filter_matrix.shape().0;
        }

        let power: Vec<f32> = x_vqt
            .iter()
            .map(|z| (z.abs() * z.abs()))
            .collect::<Vec<f32>>();

        // TODO: harmonic/percussion source separation possible with our one-sided spectrum?

        // amp -> db conversion
        power_to_db(&power)
        //Self::power_normalized(&power)
    }
    //#
}

fn power_to_db(power: &[f32]) -> Vec<f32> {
    let ref_power: f32 = 0.3.powf(2.0);
    let a_min: f32 = 1e-6.powf(2.0);
    let top_db: f32 = 60.0;

    let mut log_spec = power
        .iter()
        .map(|x| 10.0 * x.max(a_min).log10() - 10.0 * ref_power.max(a_min).log10())
        .collect::<Vec<f32>>();
    let log_spec_max = util::max(&log_spec);
    // println!(
    //     "log_spec min {}, max {}, shifting to 0...top_db",
    //     log_spec_min, log_spec_max,
    // );

    log_spec.iter_mut().for_each(|x| {
        if *x < log_spec_max - top_db {
            *x = log_spec_max - top_db
        }
    });

    let log_spec_min = util::min(&log_spec);
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
    let spec_max = util::max(power);
    dbg!(spec_max);
    power
        .iter()
        .map(|x| *x / spec_max * 50.0) //spec_max);
        .collect::<Vec<_>>()
}

/// Finds the -3 dB points of a frequency response.
///
/// We use this to determine the bandwidth of the filters.
fn find_3db_points(frequency_response: &[f32], center_freq_index: usize) -> (usize, usize) {
    let peak_magnitude = frequency_response[center_freq_index];
    let threshold = peak_magnitude / 2.0_f32.sqrt(); // -3 dB point

    let mut lower_bound = center_freq_index;
    while lower_bound > 0 && frequency_response[lower_bound] > threshold {
        lower_bound -= 1;
    }

    let mut upper_bound = center_freq_index;
    while upper_bound < frequency_response.len() - 1 && frequency_response[upper_bound] > threshold
    {
        upper_bound += 1;
    }

    (lower_bound, upper_bound)
}

/// Calculate the bandwidth of a filter in Hz.
///
/// The arguments are assumed to be downsampled such that the frequency response fits to the scaled_sr.
fn calculate_bandwidth(scaled_frequency_response: &[f32], scaled_sr: f32) -> (f32, f32) {
    let center_freq_index = arg_max(&scaled_frequency_response);
    let (lower_bound, upper_bound) = find_3db_points(scaled_frequency_response, center_freq_index);
    let lower_bound_in_hz = lower_bound as f32 * scaled_sr / scaled_frequency_response.len() as f32;
    let upper_bound_in_hz = upper_bound as f32 * scaled_sr / scaled_frequency_response.len() as f32;
    (lower_bound_in_hz, upper_bound_in_hz)
}
