/// Peak detection module for spectral analysis.
///
/// This module provides functionality for detecting and enhancing spectral peaks
/// in Variable Q Transform (VQT) data.

use crate::vqt::VqtRange;
use find_peaks::PeakFinder;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct PeakDetectionParameters {
    /// The minimum prominence of a peak to be considered a peak.
    pub min_prominence: f32,
    /// The minimum height of a peak to be considered a peak.
    pub min_height: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct ContinuousPeak {
    /// The estimated precise center of the peak, in buckets starting from the min_freq.
    pub center: f32,
    /// The estimated precise amplitude of the peak, in ???
    pub size: f32,
}

/// Find discrete peaks in VQT spectrum
pub fn find_peaks(
    peak_config: &PeakDetectionParameters,
    vqt: &[f32],
    buckets_per_octave: u16,
) -> HashSet<usize> {
    let mut fp = PeakFinder::new(vqt);
    fp.with_min_prominence(peak_config.min_prominence);
    fp.with_min_height(peak_config.min_height);

    // Add distance constraint to prevent duplicate peaks for vibrating notes
    // Minimum separation: 0.4 semitones (prevents doubles while allowing close harmonies)
    let min_separation_bins = (buckets_per_octave as f32 * 0.4 / 12.0).round() as usize;
    if min_separation_bins > 0 {
        fp.with_min_distance(min_separation_bins);
    }

    let peaks = fp.find_peaks();

    // Filter out lowest A and surroundings (first ~half semitone)
    let min_bin = (buckets_per_octave as usize / 12 + 1) / 2;
    peaks
        .iter()
        .filter(|p| p.middle_position() >= min_bin)
        .map(|p| p.middle_position())
        .collect::<HashSet<usize>>()
}

/// Enhances the detected peaks by estimating the precise center and size of each peak.
///
/// Uses logarithmically-aware quadratic interpolation to account for the non-uniform
/// (logarithmic) spacing of VQT bins. This is critical for high-frequency accuracy,
/// where bins are spaced much wider than at low frequencies.
///
/// The algorithm fits a parabola in log-frequency space to the three points around
/// each peak, then finds the parabola's maximum to estimate sub-bin peak location.
pub fn enhance_peaks_continuous(
    discrete_peaks: &HashSet<usize>,
    vqt: &[f32],
    range: &VqtRange,
) -> Vec<ContinuousPeak> {
    let mut peaks_continuous: Vec<_> = discrete_peaks
        .iter()
        .filter_map(|p| {
            let p = *p;

            if p < 1 || p > range.n_buckets() - 2 {
                // Edge case: use the discrete peak value directly
                return Some(ContinuousPeak {
                    center: p as f32,
                    size: vqt[p],
                });
            }

            // Compute actual frequencies (logarithmically spaced)
            let bins_per_octave = range.buckets_per_octave as f32;
            let f_prev = range.min_freq * 2.0_f32.powf((p - 1) as f32 / bins_per_octave);
            let f_curr = range.min_freq * 2.0_f32.powf(p as f32 / bins_per_octave);
            let f_next = range.min_freq * 2.0_f32.powf((p + 1) as f32 / bins_per_octave);

            // Work in log-frequency space where bins are more evenly spaced
            let log_f = [f_prev.ln(), f_curr.ln(), f_next.ln()];
            let amplitudes = [vqt[p - 1], vqt[p], vqt[p + 1]];

            // Fit parabola: y = a*x^2 + b*x + c using Lagrange interpolation
            // The peak is at x_peak = -b / (2*a)
            let denom = (log_f[0] - log_f[1]) * (log_f[0] - log_f[2]) * (log_f[1] - log_f[2]);

            if denom.abs() < f32::EPSILON {
                // Degenerate case (shouldn't happen with logarithmic spacing)
                return Some(ContinuousPeak {
                    center: p as f32,
                    size: vqt[p],
                });
            }

            // Compute parabola coefficients
            let a = (log_f[2] * (amplitudes[1] - amplitudes[0])
                + log_f[0] * (amplitudes[2] - amplitudes[1])
                + log_f[1] * (amplitudes[0] - amplitudes[2]))
                / denom;

            let b = (log_f[2].powi(2) * (amplitudes[0] - amplitudes[1])
                + log_f[0].powi(2) * (amplitudes[1] - amplitudes[2])
                + log_f[1].powi(2) * (amplitudes[2] - amplitudes[0]))
                / denom;

            // Find peak in log-frequency space
            let log_f_peak = if a.abs() < f32::EPSILON {
                // Nearly linear case, use discrete peak
                log_f[1]
            } else {
                (-b / (2.0 * a)).clamp(log_f[0], log_f[2])
            };

            // Convert log-frequency back to bin index
            // f_peak = min_freq * 2^(bin_index / bins_per_octave)
            // => log2(f_peak / min_freq) = bin_index / bins_per_octave
            // => bin_index = bins_per_octave * log2(f_peak / min_freq)
            let f_peak = log_f_peak.exp();
            let estimated_precise_center = bins_per_octave * (f_peak / range.min_freq).log2();

            // For amplitude, use linear interpolation of VQT values at the estimated peak
            // This is more robust than evaluating the parabola, which can produce
            // incorrect values due to the log-frequency space scaling
            let estimated_precise_center_clamped =
                estimated_precise_center.clamp(0.0, range.n_buckets() as f32 - 1.0);

            let lower_bin = estimated_precise_center_clamped.floor() as usize;
            let upper_bin = (lower_bin + 1).min(range.n_buckets() - 1);
            let fract = estimated_precise_center_clamped.fract();

            let estimated_precise_size = vqt[lower_bin] * (1.0 - fract) + vqt[upper_bin] * fract;

            Some(ContinuousPeak {
                center: estimated_precise_center_clamped,
                size: estimated_precise_size.max(0.0),
            })
        })
        .collect();
    peaks_continuous.sort_by(|a, b| a.center.partial_cmp(&b.center).unwrap());

    peaks_continuous
}

/// Promotes bass peaks that have strong harmonic content.
///
/// Real bass notes have overtones at integer multiples of their fundamental frequency.
/// This function calculates a "harmonic score" for each bass peak based on the presence
/// and strength of harmonics (2f, 3f, 4f, 5f) in the VQT spectrum.
///
/// Peaks with higher harmonic scores get their amplitude boosted, making them more
/// prominent in bass note detection. This helps distinguish real bass notes from:
/// - Low-frequency rumble/noise (no harmonics)
/// - Artifacts (no harmonic structure)
/// - Sub-bass that's just bleeding from higher notes
///
/// # Arguments
/// * `peaks_continuous` - The peaks to process (modified in-place)
/// * `vqt` - The VQT spectrum in dB scale (smoothed)
/// * `range` - The VQT frequency range
/// * `highest_bassnote` - Only process peaks below this bin index
/// * `harmonic_threshold` - Fraction of fundamental **power** required for harmonic presence
///
/// # Note on dB vs Power
/// The VQT is in dB (logarithmic) scale. We convert to power for physically meaningful
/// comparisons: power = 10^(dB/10). The threshold is a fraction of power, not dB.
pub fn promote_bass_peaks_with_harmonics(
    peaks_continuous: &mut [ContinuousPeak],
    vqt: &[f32],
    range: &VqtRange,
    highest_bassnote: usize,
    harmonic_threshold: f32,
) {
    for peak in peaks_continuous.iter_mut() {
        // Only process bass notes
        if peak.center > highest_bassnote as f32 {
            continue;
        }

        // Calculate fundamental frequency from bin index
        let fundamental_freq =
            range.min_freq * 2.0_f32.powf(peak.center / range.buckets_per_octave as f32);

        // Convert fundamental from dB to power
        let fundamental_power = 10.0_f32.powf(peak.size / 10.0);

        // Check for harmonics at 2f, 3f, 4f, 5f
        let mut harmonic_score = 0.0;
        let harmonic_weights = [0.5, 0.3, 0.15, 0.05];

        for (harmonic_num, weight) in (2..=5).zip(harmonic_weights.iter()) {
            let harmonic_freq = fundamental_freq * harmonic_num as f32;

            // Convert to bin index
            let harmonic_bin = if harmonic_freq >= range.min_freq {
                (harmonic_freq.log2() - range.min_freq.log2()) * range.buckets_per_octave as f32
                    / 2.0_f32.log2()
            } else {
                continue;
            };

            // Check if within spectrum
            if harmonic_bin >= 0.0 && harmonic_bin < range.n_buckets() as f32 {
                // Interpolate VQT at harmonic frequency
                let bin_low = harmonic_bin.floor() as usize;
                let bin_high = (harmonic_bin.ceil() as usize).min(range.n_buckets() - 1);
                let frac = harmonic_bin.fract();

                let harmonic_amplitude_db = if bin_low == bin_high {
                    vqt[bin_low]
                } else {
                    vqt[bin_low] * (1.0 - frac) + vqt[bin_high] * frac
                };

                // Convert from dB to power for comparison
                let harmonic_power = 10.0_f32.powf(harmonic_amplitude_db / 10.0);

                // Check if harmonic is present (in power domain)
                let threshold_power = fundamental_power * harmonic_threshold;
                if harmonic_power > threshold_power {
                    // Add to score in power domain
                    harmonic_score += harmonic_power * weight;
                }
            }
        }

        // Boost peak amplitude based on harmonic score
        if harmonic_score > 0.0 {
            // Calculate boost factor in power domain
            let boost_factor = 1.0 + 0.5 * (harmonic_score / fundamental_power.max(1e-6));
            let boost_capped = boost_factor.min(1.5); // Cap at 50%

            // Convert boost back to dB domain
            peak.size += 10.0 * boost_capped.log10();
        }
    }
}
