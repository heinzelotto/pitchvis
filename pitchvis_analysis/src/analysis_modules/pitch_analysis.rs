/// Pitch accuracy and deviation analysis module.
///
/// This module calculates pitch accuracy (how close to perfect pitch) and pitch deviation
/// (how many semitones sharp or flat) for detected peaks.

use super::peak_detection::ContinuousPeak;

/// Update pitch accuracy and deviation for all bins based on continuous peaks.
///
/// For each continuous peak, calculates:
/// - Pitch accuracy: 1.0 = perfectly on pitch, 0.0 = maximally off pitch (50 cents)
/// - Pitch deviation: negative = flat, positive = sharp, 0.0 = perfectly in tune (in semitones)
pub fn update_pitch_accuracy_and_deviation(
    peaks_continuous: &[ContinuousPeak],
    buckets_per_octave: u16,
    pitch_accuracy: &mut [f32],
    pitch_deviation: &mut [f32],
) {
    // Reset all values to 0.0
    pitch_accuracy.fill(0.0);
    pitch_deviation.fill(0.0);

    // For each continuous peak, calculate pitch accuracy and deviation
    for p in peaks_continuous {
        let center_in_semitones = p.center * 12.0 / buckets_per_octave as f32;

        // Signed deviation in semitones: negative = flat, positive = sharp
        let deviation = center_in_semitones - center_in_semitones.round();

        // Drift is absolute deviation in range [0.0, 0.5]
        let drift = deviation.abs();

        // Convert to accuracy: 1.0 = on pitch, 0.0 = maximally off pitch
        let accuracy = (1.0 - 2.0 * drift).max(0.0);

        // Assign to the corresponding bin (rounded to integer index)
        let bin_idx = p.center.round() as usize;
        if bin_idx < pitch_accuracy.len() {
            pitch_accuracy[bin_idx] = accuracy;
            pitch_deviation[bin_idx] = deviation;
        }
    }
}

/// Update tuning grid inaccuracy measurement.
///
/// Calculates the average inaccuracy of all active notes with respect to a 440Hz tuning grid,
/// measured in cents (1/100 semitone). This is power-weighted to emphasize louder notes.
pub fn update_tuning_inaccuracy(
    peaks_continuous: &[ContinuousPeak],
    buckets_per_octave: u16,
    smoothed_tuning_grid_inaccuracy: &mut crate::util::EmaMeasurement,
    frame_time: std::time::Duration,
) {
    let mut inaccuracy_sum = 0.0;
    let mut power_sum = 0.0;
    for p in peaks_continuous {
        let power = p.size * p.size;
        power_sum += power;

        let center_in_semitones = p.center * 12.0 / buckets_per_octave as f32;
        inaccuracy_sum += (center_in_semitones - center_in_semitones.round()).abs() * power;
    }

    let average_tuning_inaccuracy = if power_sum > 0.0 {
        inaccuracy_sum / power_sum
    } else {
        0.0
    };

    let average_tuning_inaccuracy_in_cents = 100.0 * average_tuning_inaccuracy;

    smoothed_tuning_grid_inaccuracy.update_with_timestep(average_tuning_inaccuracy_in_cents, frame_time);
}
