//! # Variable Q Transform (VQT) Implementation
//!
//! This module implements the Variable Q Transform, an extension of the Constant Q Transform
//! (CQT) with an additional parameter γ (gamma) that allows trading off frequency resolution
//! for time resolution at lower frequencies.
//!
//! ## Overview
//!
//! The VQT provides a time-frequency representation with logarithmically-spaced frequency bins,
//! making it ideal for musical analysis where notes are spaced exponentially (each semitone is
//! 2^(1/12) ≈ 1.059 times the previous note's frequency).
//!
//! ### Key Features
//!
//! - **Multi-rate processing**: Different frequency ranges are analyzed at different sample rates,
//!   optimizing computational efficiency without losing information.
//! - **Sparse kernel representation**: Only significant filter coefficients are stored, reducing
//!   memory usage and computation time.
//! - **Variable Q factor**: The gamma parameter allows improved time resolution at lower frequencies
//!   while maintaining good frequency resolution at higher frequencies.
//! - **Bandwidth validation**: Kernel construction warns if adjacent filters leave gaps between
//!   their -3 dB bandwidths.
//!
//! ## Theory
//!
//! ### Constant Q Transform (CQT)
//!
//! In a standard CQT, the frequency bins are logarithmically spaced:
//!
//! ```text
//! f_k = f_min * 2^(k / buckets_per_octave)
//! ```
//!
//! Each filter has a constant Q factor:
//!
//! ```text
//! Q = f_k / Δf_k
//! ```
//!
//! where Δf_k is the bandwidth of the k-th filter.
//!
//! ### Variable Q Transform (VQT)
//!
//! The VQT extends CQT by modifying the window length calculation:
//!
//! ```text
//! w_k = Q * sr / (α * f_k + γ)
//! ```
//!
//! where:
//! - `sr` is the sample rate
//! - `α` is a constant that ensures adjacent filters meet at their -3dB points
//! - `γ` (gamma) is the additional parameter that improves time resolution at low frequencies
//!
//! Higher γ values result in:
//! - Shorter window lengths (better time resolution)
//! - Reduced frequency resolution at lower frequencies
//! - Lower overall latency
//!
//! Note: the `quality` parameter is librosa's `filter_scale`, not the actual quality factor
//! f/Δf. The -3 dB bandwidth of a Hann-windowed filter of duration T is ≈ 1.44/T, so the
//! effective quality factor at high frequencies (where γ is negligible) is
//! Q_eff ≈ quality / (1.44 * α) ≈ 135 for the defaults.
//!
//! ### Where the VQT behaves like an STFT
//!
//! The bandwidth of bin k is Δf_k ≈ 1.44 * (α * f_k + γ) / quality. Below the crossover
//! point α * f = γ (≈ 1.05 kHz for the defaults) the γ term dominates and the transform is
//! effectively constant-bandwidth (STFT-like, bins ≈ 7-9 Hz wide); true constant-Q behavior
//! only emerges above ~2 kHz. In particular, adjacent bass *semitones* (3.3 Hz apart at A1)
//! cannot be resolved as separate peaks below roughly 300 Hz — downstream bass detection
//! compensates by scoring harmonics. For reference, the psychoacoustically motivated
//! (ERB-matched) γ from Schörkhuber et al. would be ≈ 1.9 for these parameters; the default
//! γ ≈ 7.7 deliberately trades bass frequency resolution for latency.
//!
//! ### Latency
//!
//! The analysis delay is half the longest filter window. Because the default γ is tied to
//! the quality factor (γ = 4.8 * Q), the longest window is ≈ sr / 4.8 samples *regardless
//! of Q*, pinning the delay at ≈ 100 ms. Changing `quality` therefore trades frequency
//! resolution at high frequencies at almost no latency cost.
//!
//! All filters are centered on the same time instant, so the transform is a temporally
//! coherent snapshot of the signal `delay` seconds ago. Right-aligning the shorter
//! high-frequency windows at the buffer end would show treble transients sooner, but during
//! glissandi the overtones would then move before their fundamental, visually bending the
//! overtone stack — rejected by design.
//!
//! ## Implementation Details
//!
//! ### Filter Bank Construction
//!
//! Each filter is constructed as a Hann-windowed complex exponential:
//!
//! ```text
//! h_k(n) = hanning(n) * exp(2πi * f_k * n / sr)
//! ```
//!
//! The filters are:
//! 1. Normalized in the time domain (L1 norm), so that a unit-amplitude sinusoid at the
//!    center frequency produces the same response for every bin, independent of window length
//! 2. Transformed to frequency domain via FFT
//! 3. Complex conjugated (for correlation instead of convolution)
//! 4. Sparsified (the smallest coefficients carrying `1 - sparsity_quantile` of the L1 mass
//!    are zeroed) and stored as sparse matrices
//!
//! Phase caveat: each filter's phase origin is its own window start, so the absolute phase
//! of the output coefficients is not comparable across bins. Only magnitudes are meaningful
//! downstream; phase-vocoder-style refinements would require a kernel change.
//!
//! Noise caveat: with L1 normalization the white-noise power gain is proportional to
//! 1/window_length, so the noise floor of the shortest (highest-frequency) filters sits
//! ~9 dB above that of the longest (bass) filters.
//!
//! ### Multi-rate Processing
//!
//! Filters are constructed at the lowest sample rate that still covers their center
//! frequency (with a 15% margin against Gibbs-phenomenon artifacts near the resampling
//! cutoff), keeping their kernels short and sparse. Filters with the same downsampling
//! factor form a group that reads a power-of-two-sized window of the input, all centered
//! on the common window center.
//!
//! At runtime, no explicit resampling is performed. For an ideally low-passed signal,
//! decimation by M in the time domain corresponds to `FFT_decimated[k] = FFT_full[k] / M`
//! in the frequency domain, so every kernel coefficient can index directly into the FFT of
//! its group's *unscaled* window, with the 1/M factor folded into the kernel values at
//! construction time. Groups that share the same window therefore also share a single FFT
//! per frame. Since the input is real, a real-to-complex FFT over the half spectrum
//! suffices; the few filter coefficients that fall on negative frequencies (~1% of the
//! kernel mass, sidelobes of filters close to their group's downsampled Nyquist) are
//! handled exactly via a small conjugate-part matrix, using `X[N-k] = conj(X[k])`.
//!
//! With the default parameters this results in 4 real FFTs (8192, 4096, 2048 and 1024
//! points) and 4 sparse matrix-vector products per frame.
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use pitchvis_analysis::vqt::{Vqt, VqtParameters};
//!
//! // Create VQT with default parameters
//! let params = VqtParameters::default();
//! let mut vqt = Vqt::new(&params).unwrap();
//!
//! // Prepare audio buffer (must be of length params.n_fft)
//! let audio_buffer: Vec<f32> = vec![0.0; params.n_fft];
//!
//! // Compute VQT (returns dB scale)
//! let vqt_result = vqt.calculate_vqt_instant_in_db(&audio_buffer);
//! ```
//!
//! ## Performance Characteristics
//!
//! - **Latency**: Determined primarily by the longest filter (lowest frequency). ≈ 98 ms
//!   with default parameters (see "Latency" above).
//! - **Computation**: One real FFT per distinct analysis window plus one sparse
//!   matrix-vector product per window group. ~0.1 ms per frame on a desktop CPU with
//!   default parameters.
//! - **Memory**: Sparse matrix storage keeps only ~6% of the kernel coefficients
//!   (controlled by `sparsity_quantile`).
//!
//! ## References
//!
//! - [A Matlab Toolbox for Efficient Perfect Reconstruction Time-Frequency Transforms](https://www.researchgate.net/publication/274009051)
//! - [Librosa VQT Implementation](https://librosa.org/doc/main/generated/librosa.vqt.html)
//! - [Constant-Q Transform - Wikipedia](https://en.wikipedia.org/wiki/Constant-q_transform)

use log::{debug, info, warn};
use num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_traits::Zero;
use rustfft::{Fft, FftPlanner};
use std::f32::consts::PI;
use std::sync::Arc;
use std::time::Duration;

use crate::util::arg_max;

/// Default sample rate in Hz (half of CD quality, sufficient for musical analysis up to ~10 kHz).
pub const DEFAULT_SR: u32 = 22050;

/// Default number of samples in the longest FFT (about 1.49 s at 22050 Hz).
pub const DEFAULT_N_FFT: usize = 2 * 16384;

/// Default minimum frequency: A1.
pub const DEFAULT_MIN_FREQ: f32 = 55.0;

/// Increase to analyze with finer frequency resolution at the same Q-factor-per-bin.
pub const DEFAULT_UPSCALE_FACTOR: u16 = 1;

/// Default number of frequency bins per semitone.
pub const DEFAULT_BUCKETS_PER_SEMITONE: u16 = 7 * DEFAULT_UPSCALE_FACTOR;

/// Default number of frequency bins per octave.
pub const DEFAULT_BUCKETS_PER_OCTAVE: u16 = 12 * DEFAULT_BUCKETS_PER_SEMITONE;

/// Default analyzed range in octaves.
pub const DEFAULT_OCTAVES: u8 = 7;

/// Default sparsity quantile (fraction of each filter's L1 mass kept in the sparse kernel).
pub const DEFAULT_SPARSITY_QUANTILE: f32 = 0.999;

/// Default quality factor (librosa's `filter_scale`, see module docs).
///
/// For full coverage, adjacent bins' -3 dB bandwidths must meet:
/// `1.44 * (α * f + γ) / Q >= f * (2^(1/buckets_per_octave) - 1)` for all analyzed f.
/// The binding case is the highest frequency; with γ = 4.8 * Q this solves to Q ≈ 1.63
/// for the default range, so 1.6 covers the full spectrum without gaps. Lower values widen
/// the filters (more overlap, less selectivity); higher values open coverage gaps between
/// adjacent high-frequency bins.
pub const DEFAULT_Q: f32 = 1.6 / DEFAULT_UPSCALE_FACTOR as f32;

/// Default gamma. Tying γ to Q pins the latency at ≈ sr/(2*4.8) samples (see module docs).
pub const DEFAULT_GAMMA: f32 = 4.8 * DEFAULT_Q;

/// Specifies the frequency range and resolution for the VQT analysis.
///
/// This structure defines the "musical range" that the VQT will analyze. The bins are
/// logarithmically spaced across octaves, matching the exponential frequency relationship
/// in music.
///
/// # Examples
///
/// ```
/// use pitchvis_analysis::vqt::VqtRange;
///
/// // Analyze 7 octaves starting from A1 (55Hz) with 84 bins per octave
/// // (7 subdivisions per semitone)
/// let range = VqtRange {
///     min_freq: 55.0,
///     octaves: 7,
///     buckets_per_octave: 84,
/// };
///
/// // Total number of frequency bins
/// assert_eq!(range.n_buckets(), 7 * 84);
/// ```
#[derive(Debug, Clone)]
pub struct VqtRange {
    /// The minimum frequency (in Hz) of the lowest note analyzed in the variable-Q Transform
    /// (VQT). Usually A1 (55Hz).
    pub min_freq: f32,

    /// The total range, in octaves, over which the VQT is computed.
    pub octaves: u8,

    /// The resolution of the VQT, defined in terms of the number of frequency bins per octave.
    /// This value must be a multiple of 12, reflecting the 12 semitones in a musical octave.
    ///
    /// Common values:
    /// - 12: One bin per semitone
    /// - 24: Two bins per semitone
    /// - 84: Seven bins per semitone (default in PitchVis)
    pub buckets_per_octave: u16,
}

impl VqtRange {
    /// Returns the number of frequency bins in the VQT kernel.
    pub fn n_buckets(&self) -> usize {
        self.buckets_per_octave as usize * self.octaves as usize
    }
}

/// Configuration parameters for the Variable Q Transform.
///
/// These parameters control the trade-offs between:
/// - **Frequency resolution** vs **time resolution**
/// - **Accuracy** vs **computational efficiency**
/// - **Latency** vs **frequency resolution at low frequencies**
///
/// # Parameter Relationships
///
/// - Increasing `quality` (Q) → better frequency resolution, worse time resolution, and
///   potential coverage gaps between adjacent high-frequency bins (see [`DEFAULT_Q`])
/// - Increasing `gamma` (γ) → better time resolution at low frequencies, worse frequency resolution
/// - Increasing `sparsity_quantile` → less memory/computation, slightly less accurate
/// - Increasing `buckets_per_octave` → finer frequency resolution, more computation
#[derive(Debug, Clone)]
pub struct VqtParameters {
    /// The sample rate (in Hz) of the input audio signal.
    pub sr: f32,

    /// The number of samples in the longest Fast Fourier Transform (FFT).
    ///
    /// This determines the buffer size and thus the maximum window length. Lower frequencies
    /// require longer windows. Smaller windows are used for higher octaves through multi-rate
    /// processing.
    pub n_fft: usize,

    /// The range of the VQT: minimum frequency, span in octaves, and resolution.
    pub range: VqtRange,

    /// Sparsity quantile (0.0 to 1.0) for filter kernel compression.
    ///
    /// This determines which filter coefficients are kept vs. zeroed out. A value of 0.999
    /// means we keep only the coefficients that account for 99.9% of the filter's L1 mass,
    /// typically reducing storage by ~94%.
    pub sparsity_quantile: f32,

    /// The quality factor determining frequency selectivity.
    ///
    /// This is librosa's `filter_scale`: it scales the filter window lengths via
    /// `w = quality * sr / (α * f + γ)`. It is *not* the effective quality factor
    /// f/Δf, which is ≈ `quality / (1.44 * α)` at high frequencies (≈ 135 for the
    /// defaults).
    ///
    /// Higher values → narrower filters → better frequency resolution → worse time
    /// resolution, and coverage gaps between adjacent bins once
    /// `quality > 1.44 * (α * f_max + γ) / (f_max * (2^(1/buckets_per_octave) - 1))`.
    pub quality: f32,

    /// The gamma (γ) parameter that improves time resolution at lower frequencies.
    ///
    /// This is the key parameter that distinguishes VQT from CQT. It modifies the window
    /// length formula from:
    ///
    /// ```text
    /// w = Q * sr / (α * f)           [CQT]
    /// w = Q * sr / (α * f + γ)       [VQT]
    /// ```
    ///
    /// Effect of higher γ:
    /// - Shorter windows at low frequencies
    /// - Better time resolution (less smearing)
    /// - Lower overall latency
    /// - Reduced frequency resolution at low frequencies
    ///
    /// Below the crossover frequency f = γ/α the transform is effectively
    /// constant-bandwidth rather than constant-Q (see module docs).
    pub gamma: f32,
}

impl Default for VqtParameters {
    fn default() -> Self {
        Self {
            sr: DEFAULT_SR as f32,
            n_fft: DEFAULT_N_FFT,
            range: VqtRange {
                min_freq: DEFAULT_MIN_FREQ,
                octaves: DEFAULT_OCTAVES,
                buckets_per_octave: DEFAULT_BUCKETS_PER_OCTAVE,
            },
            sparsity_quantile: DEFAULT_SPARSITY_QUANTILE,
            quality: DEFAULT_Q,
            gamma: DEFAULT_GAMMA,
        }
    }
}

/// Errors that can occur when constructing a [`Vqt`].
#[derive(Debug, thiserror::Error)]
pub enum VqtError {
    #[error(
        "the highest VQT bin frequency ({highest_frequency} Hz) exceeds the Nyquist \
         frequency ({nyquist_frequency} Hz); reduce octaves or increase the sample rate"
    )]
    AboveNyquist {
        highest_frequency: f32,
        nyquist_frequency: f32,
    },
    #[error(
        "the longest filter window ({window_length} samples) exceeds n_fft ({n_fft} \
         samples); increase n_fft or gamma, or decrease quality"
    )]
    WindowExceedsNFft { window_length: f32, n_fft: usize },
}

/// Parameters of a single filter in the VQT filter bank, annotated with its multi-rate
/// processing constraints.
#[derive(Debug, Clone, Copy)]
struct FilterParams {
    /// Center frequency in Hz.
    freq: f32,

    /// Window length in samples at the original sample rate.
    window_length: f32,

    /// The maximum factor (a power of two) by which the signal can be downsampled while
    /// still covering this filter's frequency, including the anti-Gibbs margin.
    sr_downscaling_factor: usize,

    /// The smallest power-of-two fraction of n_fft that contains this filter's window.
    minimum_needed_window_size: usize,
}

/// A set of filters that are all applied to the FFT of the same slice ("window") of the
/// input buffer.
pub struct WindowGroup {
    /// Start and end of the input slice, relative to an n_fft-sized buffer whose last
    /// sample is "now".
    pub window: (usize, usize),

    /// Sparse filter matrix. Rows are filters in ascending frequency order (contiguous
    /// with the previous group); columns index the half spectrum (length window/2 + 1)
    /// of the window's real FFT.
    pub filter_bank: sprs::CsMat<Complex32>,

    /// Conjugate-part filter matrix for the few filter coefficients that fall on
    /// negative frequencies (sidelobes of filters close to their group's downsampled
    /// Nyquist frequency). For a real input signal, `X[N - k] = conj(X[k])`, so these
    /// contribute `conj(negative_filter_bank · X_half)`. None if no filter in this group
    /// has such coefficients. Holds ~1% of the kernel mass.
    pub negative_filter_bank: Option<sprs::CsMat<Complex32>>,
}

impl WindowGroup {
    pub fn window_size(&self) -> usize {
        self.window.1 - self.window.0
    }
}

/// The precomputed VQT kernel: all filter banks, grouped by the input window they read.
pub struct VqtKernel {
    pub window_groups: Vec<WindowGroup>,
}

/// A single filter's frequency-domain coefficients (at its group's downsampled rate)
/// and its measured bandwidth.
struct Filter {
    v_frequency_domain: Vec<Complex32>,
    bandwidth_3db_in_hz: (f32, f32),
}

/// Preallocated per-frame working memory, sized for the largest window group.
struct ComputeScratch {
    input: Vec<f32>,
    spectrum: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
    x_vqt: Vec<Complex32>,
    x_vqt_negative_part: Vec<Complex32>,
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

    /// One real-to-complex FFT plan per window group (same order as
    /// `vqt_kernel.window_groups`).
    ffts: Vec<Arc<dyn RealToComplex<f32>>>,

    /// Preallocated per-frame working memory.
    scratch: ComputeScratch,
}

impl Vqt {
    /// Creates a new VQT analyzer with the given parameters.
    ///
    /// Returns an error if the parameters are invalid (e.g., quality too high causing the
    /// longest window to exceed n_fft, or the highest frequency exceeding the Nyquist
    /// frequency).
    pub fn new(params: &VqtParameters) -> Result<Self, VqtError> {
        let (vqt_kernel, delay) = Self::vqt_kernel(params)?;

        info!("VQT analysis delay: {} ms.", delay.as_millis());

        let mut planner = RealFftPlanner::<f32>::new();
        let ffts: Vec<Arc<dyn RealToComplex<f32>>> = vqt_kernel
            .window_groups
            .iter()
            .map(|g| planner.plan_fft_forward(g.window_size()))
            .collect();

        let max_window_size = vqt_kernel
            .window_groups
            .iter()
            .map(|g| g.window_size())
            .max()
            .unwrap_or(0);
        let max_fft_scratch = ffts.iter().map(|f| f.get_scratch_len()).max().unwrap_or(0);
        let max_group_filters = vqt_kernel
            .window_groups
            .iter()
            .map(|g| g.filter_bank.rows())
            .max()
            .unwrap_or(0);
        let scratch = ComputeScratch {
            input: vec![0.0; max_window_size],
            spectrum: vec![Complex32::zero(); max_window_size / 2 + 1],
            fft_scratch: vec![Complex32::zero(); max_fft_scratch],
            x_vqt: vec![Complex32::zero(); params.range.n_buckets()],
            x_vqt_negative_part: vec![Complex32::zero(); max_group_filters],
        };

        Ok(Self {
            params: params.clone(),
            vqt_kernel,
            delay,
            ffts,
            scratch,
        })
    }

    pub fn params(&self) -> &VqtParameters {
        &self.params
    }

    pub fn kernel(&self) -> &VqtKernel {
        &self.vqt_kernel
    }

    /// Computes the per-filter parameters of the filter bank: center frequencies, window
    /// lengths and multi-rate constraints.
    fn filter_bank_params(params: &VqtParameters) -> Result<Vec<FilterParams>, VqtError> {
        let highest_frequency = params.range.min_freq
            * 2.0_f32.powf(
                (params.range.n_buckets() - 1) as f32 / params.range.buckets_per_octave as f32,
            );
        let nyquist_frequency = params.sr / 2.0;
        if highest_frequency > nyquist_frequency {
            return Err(VqtError::AboveNyquist {
                highest_frequency,
                nyquist_frequency,
            });
        }

        // alpha is constant and such that (1+a)*f_{k-1} = (1-a)*f_{k+1}, i.e. adjacent
        // filters meet at their -3dB points
        let r = 2.0_f32.powf(1.0 / params.range.buckets_per_octave as f32);
        let alpha = (r * r - 1.0) / (r * r + 1.0);

        let filters = (0..params.range.n_buckets())
            .map(|k| {
                let freq = params.range.min_freq
                    * 2.0_f32.powf(k as f32 / params.range.buckets_per_octave as f32);
                let window_length = params.quality * params.sr / (alpha * freq + params.gamma);

                let sr_downscaling_factor = {
                    // Because of the Gibbs phenomenon appearing near the cutoff of the
                    // implicit brick-wall low-pass, we keep the downsampled Nyquist
                    // frequency 15% above the theoretically needed one.
                    const GRACE_FACTOR: f32 = 1.15;
                    let minimum_scaled_sr = (freq * 2.0 * GRACE_FACTOR).ceil();
                    // maximum k so that minimum_scaled_sr <= sr / 2^k
                    let k = (params.sr / minimum_scaled_sr).log2().floor() as u32;
                    1_usize << k
                };

                let minimum_needed_window_size = {
                    // largest power-of-two reduction of n_fft that still contains the window
                    let k = (params.n_fft as f32 / window_length).log2().floor() as u32;
                    params.n_fft >> k
                };

                FilterParams {
                    freq,
                    window_length,
                    sr_downscaling_factor,
                    minimum_needed_window_size,
                }
            })
            .collect::<Vec<_>>();

        let longest_window = filters[0].window_length;
        if longest_window > params.n_fft as f32 {
            return Err(VqtError::WindowExceedsNFft {
                window_length: longest_window,
                n_fft: params.n_fft,
            });
        }

        for fp in &filters {
            debug!(
                "f: {:6.1} Hz, window: {:6.1} samples [{:5.1} ms], sr/{:<3}, min window size: {:5}",
                fp.freq,
                fp.window_length,
                1000.0 * fp.window_length / params.sr,
                fp.sr_downscaling_factor,
                fp.minimum_needed_window_size,
            );
        }

        Ok(filters)
    }

    /// Calculates the VQT kernel for the given parameters.
    ///
    /// Filters are grouped by downsampling factor, the groups are merged by the input
    /// window they read, and each merged group is stored as one sparse matrix over the
    /// half spectrum of its window (see module docs for the math).
    ///
    /// It is checked that adjacent filters' -3 dB bandwidths leave no gaps; a warning is
    /// logged for every gap found.
    ///
    /// Returns the kernel and the analysis delay.
    fn vqt_kernel(params: &VqtParameters) -> Result<(VqtKernel, Duration), VqtError> {
        let filters = Self::filter_bank_params(params)?;

        // All filters are centered on this instant (relative to an n_fft buffer whose
        // last sample is "now"), so the transform is a temporally coherent snapshot.
        let max_window_length = filters[0].window_length;
        let window_center = params.n_fft as f32 - max_window_length / 2.0;

        /// A contiguous run of filters sharing one downsampling factor.
        struct RateGroup<'a> {
            sr_downscaling_factor: usize,
            window: (usize, usize),
            filters: &'a [FilterParams],
        }

        // The downsampling factor is monotonically non-increasing in frequency, so equal
        // factors are contiguous.
        let rate_groups: Vec<RateGroup> = filters
            .chunk_by(|a, b| a.sr_downscaling_factor == b.sr_downscaling_factor)
            .map(|group| {
                let window_size = group
                    .iter()
                    .map(|fp| fp.minimum_needed_window_size)
                    .max()
                    .unwrap();
                // The window is centered around the common window center if it fits,
                // otherwise it is right-aligned with the buffer end. The filters within
                // are always placed at the common center either way.
                let window = if (window_center + (window_size as f32) / 2.0) < (params.n_fft as f32)
                {
                    (
                        (window_center - (window_size as f32) / 2.0) as usize,
                        (window_center + (window_size as f32) / 2.0) as usize,
                    )
                } else {
                    (params.n_fft - window_size, params.n_fft)
                };
                RateGroup {
                    sr_downscaling_factor: group[0].sr_downscaling_factor,
                    window,
                    filters: group,
                }
            })
            .collect();

        // An arbitrary global gain, chosen so that the dB values produced by power_to_db
        // land in a useful range for the visualization.
        let kernel_gain = params.sr.sqrt();

        let mut planner = FftPlanner::new();
        let mut last_upper_bandwidth = 0.0_f32;

        // Merge rate groups that read the same window; each merged group shares one FFT
        // at runtime.
        let window_groups = rate_groups
            .chunk_by(|a, b| a.window == b.window)
            .map(|window_chunk| {
                let window = window_chunk[0].window;
                let window_size = window.1 - window.0;
                let n_spectrum = window_size / 2 + 1;
                let n_filters: usize = window_chunk.iter().map(|rg| rg.filters.len()).sum();

                debug!(
                    "window {:?} ({} samples): {} filters in {} rate group(s)",
                    window,
                    window_size,
                    n_filters,
                    window_chunk.len()
                );

                let mut mat = sprs::TriMat::new((n_filters, n_spectrum));
                let mut neg_mat = sprs::TriMat::new((n_filters, n_spectrum));
                let mut row = 0;
                for rate_group in window_chunk {
                    let m = rate_group.sr_downscaling_factor;
                    let scaled_n_fft = window_size / m;
                    let fft = planner.plan_fft_forward(scaled_n_fft);

                    for filter_params in rate_group.filters {
                        let filter = Self::calculate_filter(
                            params.sr,
                            params.sparsity_quantile,
                            m,
                            *filter_params,
                            window,
                            window_center,
                            &fft,
                        );

                        debug!(
                            "filter at {:.1} Hz: window {:.1} samples, -3 dB band ({:.2}, {:.2}) Hz",
                            filter_params.freq,
                            filter_params.window_length,
                            filter.bandwidth_3db_in_hz.0,
                            filter.bandwidth_3db_in_hz.1,
                        );
                        if last_upper_bandwidth > 0.0
                            && filter.bandwidth_3db_in_hz.0 > last_upper_bandwidth
                        {
                            warn!(
                                "coverage gap below the filter at {:.1} Hz: its -3 dB band starts \
                                 at {:.2} Hz but the previous filter's band ends at {:.2} Hz \
                                 ({:.1}% of this filter's bandwidth); decrease quality to close \
                                 the gap",
                                filter_params.freq,
                                filter.bandwidth_3db_in_hz.0,
                                last_upper_bandwidth,
                                100.0 * (filter.bandwidth_3db_in_hz.0 - last_upper_bandwidth)
                                    / (filter.bandwidth_3db_in_hz.1 - filter.bandwidth_3db_in_hz.0)
                            );
                        }
                        last_upper_bandwidth = filter.bandwidth_3db_in_hz.1;

                        // Remap the filter's coefficients from its downsampled spectrum
                        // (length scaled_n_fft) onto the half spectrum of the unscaled
                        // window: decimated bin j and full-spectrum bin j have the same
                        // frequency, and FFT_decimated[j] = FFT_full[j] / m, so the 1/m
                        // (together with the 1/scaled_n_fft FFT normalization, i.e.
                        // 1/window_size in total) is folded into the kernel values.
                        //
                        // Coefficients beyond the decimated Nyquist (j > scaled_n_fft/2)
                        // index negative frequencies. For a real input signal these bins
                        // are the conjugates of their mirror bins, so such a coefficient c
                        // contributes c * conj(X_half[scaled_n_fft - j]), which is
                        // accumulated as conj(conj(c) * X_half[scaled_n_fft - j]) via the
                        // conjugate-part matrix.
                        for (j, z) in filter.v_frequency_domain.iter().enumerate() {
                            if z.is_zero() {
                                continue;
                            }
                            let value = *z * kernel_gain / window_size as f32;
                            if j <= scaled_n_fft / 2 {
                                mat.add_triplet(row, j, value);
                            } else {
                                neg_mat.add_triplet(row, scaled_n_fft - j, value.conj());
                            }
                        }

                        row += 1;
                    }
                }

                debug!(
                    "window {:?}: kernel nnz {}, conjugate-part nnz {}",
                    window,
                    mat.nnz(),
                    neg_mat.nnz()
                );

                WindowGroup {
                    window,
                    filter_bank: mat.to_csr(),
                    negative_filter_bank: (neg_mat.nnz() > 0).then(|| neg_mat.to_csr()),
                }
            })
            .collect::<Vec<_>>();

        let delay = Duration::from_secs_f32((params.n_fft as f32 - window_center) / params.sr);

        Ok((VqtKernel { window_groups }, delay))
    }

    /// Calculates a single filter of the VQT filter bank, at the downsampled rate of its
    /// rate group.
    ///
    /// # Arguments
    ///
    /// * `window_center` - The common center of all filters in the time domain, relative
    ///   to the n_fft buffer. We arrange the filters in the time domain such that all
    ///   filters are centered around the same time instant.
    fn calculate_filter(
        sr: f32,
        sparsity_quantile: f32,
        sr_scaling: usize,
        filter_params: FilterParams,
        group_window: (usize, usize),
        window_center: f32,
        fft: &Arc<dyn Fft<f32>>,
    ) -> Filter {
        let scaled_freq = filter_params.freq * sr_scaling as f32;
        let scaled_window_length = filter_params.window_length / sr_scaling as f32;
        let scaled_window_length_rounded = scaled_window_length.round() as usize;
        let scaled_window_center = (window_center - group_window.0 as f32) / sr_scaling as f32;
        let scaled_window_center_rounded = scaled_window_center.floor() as usize;
        let scaled_n_fft = (group_window.1 - group_window.0) / sr_scaling;

        assert!(scaled_window_length_rounded <= scaled_n_fft);
        let filter_begin = scaled_window_center_rounded
            .checked_sub(scaled_window_length_rounded / 2)
            .expect("filter window must fit between the start of its group window and the common window center");
        assert!(
            filter_begin + scaled_window_length_rounded <= scaled_n_fft,
            "filter window must end before the end of its group window"
        );

        // Create the windowed wavelet h(x) = hanning(x) * e^(2 * pi * i * f * x), centered
        // on the common window center. Everything is scaled by the sr_scaling factor.
        let mut v_frequency_domain = vec![Complex32::zero(); scaled_n_fft];
        for (i, w) in apodize::hanning_iter(scaled_window_length_rounded).enumerate() {
            v_frequency_domain[filter_begin + i] =
                w as f32 * (Complex32::i() * 2.0 * PI * (i as f32) * scaled_freq / sr).exp();
        }

        // Normalize the windowed wavelet in the time domain (L1), so that the response to
        // an on-center unit-amplitude sinusoid is independent of the window length.
        let norm_1: f32 = v_frequency_domain.iter().map(|z| z.norm()).sum();
        v_frequency_domain.iter_mut().for_each(|z| *z /= norm_1);

        // Transform the wavelet into frequency space.
        fft.process(&mut v_frequency_domain);

        // The complex conjugate is what we later need (correlation instead of convolution).
        v_frequency_domain.iter_mut().for_each(|z| *z = z.conj());

        let mut v_frequency_response = v_frequency_domain
            .iter()
            .map(|z| z.norm())
            .collect::<Vec<f32>>();

        let bandwidth_3db_in_hz =
            calculate_bandwidth(&v_frequency_response, sr / sr_scaling as f32);

        // Sparsify: zero all coefficients that together carry only (1 - sparsity_quantile)
        // of the filter's L1 mass.
        v_frequency_response.sort_by(f32::total_cmp);
        let v_abs_sum = v_frequency_response.iter().sum::<f32>();
        let mut accum = 0.0;
        let mut cutoff_idx = 0;
        while accum < (1.0 - sparsity_quantile) * v_abs_sum {
            accum += v_frequency_response[cutoff_idx];
            cutoff_idx += 1;
        }
        let cutoff_value = if cutoff_idx == 0 {
            0.0
        } else {
            v_frequency_response[cutoff_idx - 1]
        };
        let mut erased = 0;
        v_frequency_domain.iter_mut().for_each(|z| {
            if z.norm() < cutoff_value {
                *z = Complex32::zero();
                erased += 1;
            }
        });
        debug!(
            "for freq {} erased {erased} points below {cutoff_value} with sum {accum} out of total {v_abs_sum}",
            filter_params.freq
        );

        Filter {
            v_frequency_domain,
            bandwidth_3db_in_hz,
        }
    }

    /// Calculates the variable-Q Transform (VQT) of the given input signal at a specific
    /// time instant. The result is given in dB scale.
    ///
    /// For each window group, the relevant slice of the input is transformed with a real
    /// FFT and the precomputed sparse filter bank is applied to the half spectrum. The
    /// magnitudes are then converted to the dB scale.
    ///
    /// # Arguments
    /// * `x`: The input signal, exactly `n_fft` samples, the last of which is "now".
    ///
    /// # Returns
    /// A vector containing the VQT of the input signal in dB scale.
    pub fn calculate_vqt_instant_in_db(&mut self, x: &[f32]) -> Vec<f32> {
        assert_eq!(
            x.len(),
            self.params.n_fft,
            "input must be exactly n_fft samples"
        );

        self.scratch.x_vqt.fill(Complex32::zero());

        let mut offset = 0;
        for (group, fft) in self.vqt_kernel.window_groups.iter().zip(self.ffts.iter()) {
            let (window_begin, window_end) = group.window;
            let window_size = window_end - window_begin;
            let n_spectrum = window_size / 2 + 1;

            let input = &mut self.scratch.input[..window_size];
            input.copy_from_slice(&x[window_begin..window_end]);

            let spectrum = &mut self.scratch.spectrum[..n_spectrum];
            let fft_scratch = &mut self.scratch.fft_scratch[..fft.get_scratch_len()];
            fft.process_with_scratch(input, spectrum, fft_scratch)
                .expect("buffer sizes match the FFT plan");

            let n_filters = group.filter_bank.rows();
            sprs::prod::mul_acc_mat_vec_csr(
                group.filter_bank.view(),
                &*spectrum,
                &mut self.scratch.x_vqt[offset..(offset + n_filters)],
            );

            if let Some(negative_filter_bank) = &group.negative_filter_bank {
                let negative_part = &mut self.scratch.x_vqt_negative_part[..n_filters];
                negative_part.fill(Complex32::zero());
                sprs::prod::mul_acc_mat_vec_csr(
                    negative_filter_bank.view(),
                    &*spectrum,
                    &mut *negative_part,
                );
                for (acc, neg) in self.scratch.x_vqt[offset..(offset + n_filters)]
                    .iter_mut()
                    .zip(negative_part.iter())
                {
                    *acc += neg.conj();
                }
            }

            offset += n_filters;
        }

        power_to_db(&self.scratch.x_vqt)
    }
}

/// Converts the complex VQT coefficients to a dB scale relative to a fixed reference power,
/// clamped to a 60 dB range below the frame maximum and shifted so that the output is
/// non-negative.
fn power_to_db(x_vqt: &[Complex32]) -> Vec<f32> {
    const REF_POWER: f32 = 0.3 * 0.3;
    const A_MIN: f32 = 1e-6 * 1e-6;
    const TOP_DB: f32 = 60.0;

    let ref_db = 10.0 * REF_POWER.log10();
    let mut log_spec = x_vqt
        .iter()
        .map(|z| 10.0 * z.norm_sqr().max(A_MIN).log10() - ref_db)
        .collect::<Vec<f32>>();

    let mut log_spec_max = f32::MIN;
    let mut log_spec_min = f32::MAX;
    for x in &log_spec {
        log_spec_max = log_spec_max.max(*x);
        log_spec_min = log_spec_min.min(*x);
    }
    let floor = log_spec_max - TOP_DB;
    let log_spec_min = log_spec_min.max(floor);

    // Clamp to the 60 dB range below the maximum. Then shift down to zero if the whole
    // frame is positive, and cut off below 0.0 otherwise.
    log_spec.iter_mut().for_each(|x| {
        let clamped = x.max(floor);
        *x = if log_spec_min > 0.0 {
            clamped - log_spec_min
        } else {
            clamped.max(0.0)
        };
    });

    log_spec
}

/// Finds the -3 dB points of a frequency response.
///
/// We use this to determine the bandwidth of the filters.
///
/// Note: Since we downscale the signal as much as possible, the frequency response only spans a
/// few buckets and the -3 dB points are a very crude approximation of the bandwidth.
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
    let center_freq_index = arg_max(scaled_frequency_response);
    let (lower_bound, upper_bound) = find_3db_points(scaled_frequency_response, center_freq_index);
    let lower_bound_in_hz = lower_bound as f32 * scaled_sr / scaled_frequency_response.len() as f32;
    let upper_bound_in_hz = upper_bound as f32 * scaled_sr / scaled_frequency_response.len() as f32;
    (lower_bound_in_hz, upper_bound_in_hz)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::util::{max, test_create_sines};

    #[test]
    fn test_vqt_bandwidths() {
        let _ = env_logger::try_init();

        let params = VqtParameters::default();
        let mut vqt = Vqt::new(&params).unwrap();

        const SUBDIVISIONS_PER_BUCKET: usize = 20;
        let mut max_single_response = 0.0_f32;
        let mut min_sum_response = f32::INFINITY;
        for i in (SUBDIVISIONS_PER_BUCKET / 2) // skip the first and last half semitone
            ..(params.range.n_buckets() * SUBDIVISIONS_PER_BUCKET - SUBDIVISIONS_PER_BUCKET / 2)
        {
            let freq = params.range.min_freq
                * 2.0_f32.powf(
                    i as f32
                        / (params.range.buckets_per_octave as f32 * SUBDIVISIONS_PER_BUCKET as f32),
                );
            let x = test_create_sines(&params, &[freq], 0.0);
            let vqt_res = vqt.calculate_vqt_instant_in_db(&x);
            let max_single = max(&vqt_res);
            let sum = vqt_res.iter().sum::<f32>();

            max_single_response = max_single_response.max(max_single);
            min_sum_response = min_sum_response.min(sum);
        }
        println!(
            "max_single_response: {:.1}, min_sum_response: {:.1}",
            max_single_response, min_sum_response
        );
        assert!(max_single_response - min_sum_response < 3.0);
    }

    /// The response to a pure tone must not change abruptly when the tone crosses the
    /// boundary between two multi-rate groups (different downsampling factors and/or
    /// analysis windows).
    #[test]
    fn test_vqt_group_boundary_continuity() {
        let _ = env_logger::try_init();

        let params = VqtParameters::default();
        let mut vqt = Vqt::new(&params).unwrap();

        // Collect the lowest frequency of every rate group, skipping the very first.
        let boundaries: Vec<f32> = {
            let filters = Vqt::filter_bank_params(&params).unwrap();
            filters
                .windows(2)
                .filter(|w| w[0].sr_downscaling_factor != w[1].sr_downscaling_factor)
                .map(|w| w[1].freq)
                .collect()
        };
        assert!(!boundaries.is_empty());

        for boundary in boundaries {
            let mut responses = Vec::new();
            const STEPS: i32 = 20;
            for i in -STEPS..=STEPS {
                // sweep ±a quarter semitone around the boundary
                let freq = boundary * 2.0_f32.powf(i as f32 / (STEPS as f32 * 4.0 * 12.0));
                let x = test_create_sines(&params, &[freq], 0.0);
                let vqt_res = vqt.calculate_vqt_instant_in_db(&x);
                responses.push(max(&vqt_res));
            }
            let lo = responses.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = responses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!(
                "boundary {:7.1} Hz: max response in [{:.2}, {:.2}] dB (spread {:.2} dB)",
                boundary,
                lo,
                hi,
                hi - lo
            );
            assert!(
                hi - lo < 3.0,
                "response spread of {:.2} dB across the group boundary at {:.1} Hz",
                hi - lo,
                boundary
            );
        }
    }

    #[test]
    fn test_vqt_delay() {
        let params = VqtParameters::default();
        let vqt = Vqt::new(&params).unwrap();

        println!("VQT delay: {} ms.", vqt.delay.as_millis());
        assert!(vqt.delay.as_millis() < 100);
    }

    #[test]
    fn test_fft_library() {
        // check that the fft library does not multiply the output by 1/n or anything else
        let mut x = vec![0.0; 256];
        x[0] = 1.0;
        let mut x_fft = x
            .iter()
            .map(|f| rustfft::num_complex::Complex32::new(*f, 0.0))
            .collect::<Vec<rustfft::num_complex::Complex32>>();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(256);
        fft.process(&mut x_fft);
        let inv_fft = planner.plan_fft_inverse(256);
        inv_fft.process(&mut x_fft);

        assert_eq!(x_fft[0].re, 256.0);
    }

    #[test]
    fn test_real_fft_library() {
        // check that the real FFT's half spectrum matches the complex FFT's lower half
        let n = 256;
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();

        let mut x_fft = signal
            .iter()
            .map(|f| Complex32::new(*f, 0.0))
            .collect::<Vec<Complex32>>();
        let mut planner = FftPlanner::new();
        planner.plan_fft_forward(n).process(&mut x_fft);

        let mut real_planner = RealFftPlanner::<f32>::new();
        let r2c = real_planner.plan_fft_forward(n);
        let mut input = signal.clone();
        let mut spectrum = r2c.make_output_vec();
        r2c.process(&mut input, &mut spectrum).unwrap();

        assert_eq!(spectrum.len(), n / 2 + 1);
        for (a, b) in spectrum.iter().zip(x_fft.iter()) {
            assert!((a - b).norm() < 1e-3);
        }
    }
}
