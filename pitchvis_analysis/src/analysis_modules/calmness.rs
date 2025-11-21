/// Calmness analysis module.
///
/// This module calculates the "calmness" of notes and scenes based on how sustained
/// they are over time. Calmness is high for sustained notes and low for rapidly
/// changing or transient sounds.

use crate::util::EmaMeasurement;
use crate::vqt::VqtRange;
use std::cmp::{max, min};
use std::time::Duration;

use super::peak_detection::find_peaks;
use super::peak_detection::PeakDetectionParameters;

/// Updates calmness measurements for all frequency bins and the overall scene.
///
/// For each bin, take the few bins around it into account as well. If the bin is a
/// peak, it is promoted as calm. Calmness means that the note has been sustained for a while.
///
/// IMPROVEMENTS:
/// 1. Amplitude-weighted: Louder notes contribute more to scene calmness (matches perception)
/// 2. Released note tracking: Recently released notes contribute to prevent abrupt drops
pub fn update_calmness(
    x_vqt: &[f32],
    x_vqt_smoothed_values: &[f32],
    frame_time: Duration,
    range: &VqtRange,
    peak_config: &PeakDetectionParameters,
    calmness: &mut [EmaMeasurement],
    released_note_calmness: &mut [EmaMeasurement],
    smoothed_scene_calmness: &mut EmaMeasurement,
) {
    let mut peaks_around = vec![false; range.n_buckets()];
    // we update the bins around ~ +- 30 ct of a currently detected pitch. This way a small vibrato will
    // not decrease calmness.
    // TODO: test whether /3 or /2 is better
    let radius = range.buckets_per_octave / 12 / 3;

    // We want unsmoothed peaks for this (more responsive)
    let peaks = find_peaks(peak_config, x_vqt, range.buckets_per_octave);
    for p in peaks {
        for i in max(0, p as i32 - radius as i32)
            ..min(range.n_buckets() as i32, p as i32 + radius as i32)
        {
            peaks_around[i as usize] = true;
        }
    }

    // Calculate amplitude-weighted scene calmness
    let mut weighted_calmness_sum = 0.0;
    let mut weight_sum = 0.0;

    for (bin_idx, ((calmness_bin, released_calmness), has_peak)) in calmness
        .iter_mut()
        .zip(released_note_calmness.iter_mut())
        .zip(peaks_around.iter())
        .enumerate()
    {
        if *has_peak {
            // Note is active - update to maximum calmness
            calmness_bin.update_with_timestep(1.0, frame_time);

            // Sync released calmness with active calmness
            *released_calmness = calmness_bin.clone();

            // Weight by amplitude (convert dB to power for proper weighting)
            let amplitude_db = x_vqt_smoothed_values[bin_idx];
            let amplitude_power = 10.0_f32.powf(amplitude_db / 10.0);

            weighted_calmness_sum += calmness_bin.get() * amplitude_power;
            weight_sum += amplitude_power;
        } else {
            // Note is not active - decay both calmness values
            calmness_bin.update_with_timestep(0.0, frame_time);
            released_calmness.update_with_timestep(0.0, frame_time);

            // Recently released notes contribute at 30% weight to prevent abrupt drops
            let released_contribution = released_calmness.get();
            if released_contribution > 0.01 {
                // Still has some calmness from being recently released
                let released_weight = released_contribution * 0.3;
                weighted_calmness_sum += released_contribution * released_weight;
                weight_sum += released_weight;
            }
        }
    }

    if weight_sum > 0.0 {
        smoothed_scene_calmness
            .update_with_timestep(weighted_calmness_sum / weight_sum, frame_time);
    }
}
