/// Afterglow effect module.
///
/// This module provides a visual decay effect for the VQT spectrum, creating a
/// "trail" or afterglow that gradually fades away.

/// Update afterglow effect
///
/// The afterglow provides a visual decay effect that enhances the visualization
/// of the spectrum. Lower frequencies decay slower than higher frequencies.
pub fn update_afterglow(
    x_vqt_afterglow: &mut [f32],
    x_vqt_smoothed_values: &[f32],
    n_buckets: usize,
) {
    x_vqt_afterglow
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| {
            *x *= 0.85 - 0.15 * (i as f32 / n_buckets as f32);
            if *x < x_vqt_smoothed_values[i] {
                *x = x_vqt_smoothed_values[i];
            }
        });
}

/// Apply peak filtering to VQT
///
/// Creates a version of the VQT where only peaks are retained, with all other
/// bins set to zero.
pub fn apply_peak_filter(
    x_vqt_smoothed_values: &[f32],
    peaks: &std::collections::HashSet<usize>,
) -> Vec<f32> {
    x_vqt_smoothed_values
        .iter()
        .enumerate()
        .map(|(i, x)| {
            if peaks.contains(&i) {
                *x
            } else {
                0.0
            }
        })
        .collect::<Vec<f32>>()
}
