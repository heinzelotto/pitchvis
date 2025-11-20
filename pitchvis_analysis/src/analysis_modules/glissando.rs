/// Glissando detection module.
///
/// This module tracks peaks over time to detect glissandi (smooth pitch changes).
/// It maintains a list of tracked peaks and creates glissando objects when peaks
/// move significant distances.

use super::peak_detection::ContinuousPeak;
use log::trace;
use std::time::Duration;

/// Represents a tracked peak over time for glissando detection
#[derive(Debug, Clone)]
pub struct TrackedPeak {
    /// Unique identifier for this tracked peak
    pub id: u64,
    /// Current center position in buckets
    pub center: f32,
    /// Current amplitude
    pub size: f32,
    /// Recent position history (center values)
    pub position_history: Vec<f32>,
    /// Time since this peak was last matched to a detected peak (in seconds)
    pub time_since_update: f32,
    /// Total time this peak has been tracked (in seconds)
    pub total_time: f32,
}

/// Represents a completed glissando for rendering
#[derive(Debug, Clone)]
pub struct Glissando {
    /// Path points (center positions in buckets)
    pub path: Vec<f32>,
    /// Average amplitude along the path
    pub average_size: f32,
    /// Time when this glissando was created (for aging/fading)
    pub creation_time: f32,
}

/// Parameters for peak tracking and glissando detection
#[derive(Debug, Clone)]
pub struct GlissandoParameters {
    /// Maximum distance (in buckets) a peak can move between frames to be considered the same peak
    pub peak_tracking_max_distance: f32,
    /// Time (in seconds) after which an unmatched tracked peak is removed
    pub peak_tracking_timeout: f32,
    /// Maximum number of position samples to keep in history per tracked peak
    pub peak_tracking_history_length: usize,
    /// Minimum distance traveled (in buckets) for a tracked peak to become a glissando
    pub glissando_min_distance: f32,
    /// Duration (in seconds) to keep glissandi for rendering before removal
    pub glissando_lifetime: f32,
}

impl Default for GlissandoParameters {
    fn default() -> Self {
        Self {
            peak_tracking_max_distance: 10.5, // ~11 bins (~1.5 semitones for 7 buckets/semitone)
            peak_tracking_timeout: 0.15,      // 150ms without match -> remove
            peak_tracking_history_length: 120, // Keep last 120 samples (~2 seconds at 60fps)
            glissando_min_distance: 14.0,     // Minimum 14 buckets (1 semitone) traveled
            glissando_lifetime: 1.0,          // Keep glissandi visible for 1 second
        }
    }
}

/// Update peak tracking for glissando detection
pub fn update_peak_tracking(
    peaks_continuous: &[ContinuousPeak],
    frame_time: Duration,
    params: &GlissandoParameters,
    tracked_peaks: &mut Vec<TrackedPeak>,
    glissandi: &mut Vec<Glissando>,
    next_peak_id: &mut u64,
    elapsed_time: &mut f32,
) {
    let delta_time = frame_time.as_secs_f32();
    *elapsed_time += delta_time;

    // Age existing tracked peaks
    for tracked in tracked_peaks.iter_mut() {
        tracked.time_since_update += delta_time;
    }

    // Track which peaks have been matched
    let mut matched_tracked_indices = vec![false; tracked_peaks.len()];
    let mut matched_detected_indices = vec![false; peaks_continuous.len()];

    // Match detected peaks to tracked peaks (nearest neighbor within threshold)
    for (detected_idx, detected_peak) in peaks_continuous.iter().enumerate() {
        let mut best_match: Option<(usize, f32)> = None;

        for (tracked_idx, tracked_peak) in tracked_peaks.iter().enumerate() {
            if matched_tracked_indices[tracked_idx] {
                continue; // Already matched
            }

            let distance = (detected_peak.center - tracked_peak.center).abs();
            if distance <= params.peak_tracking_max_distance {
                if let Some((_, best_dist)) = best_match {
                    if distance < best_dist {
                        best_match = Some((tracked_idx, distance));
                    }
                } else {
                    best_match = Some((tracked_idx, distance));
                }
            }
        }

        // If we found a match, update the tracked peak
        if let Some((tracked_idx, _)) = best_match {
            matched_tracked_indices[tracked_idx] = true;
            matched_detected_indices[detected_idx] = true;

            let tracked = &mut tracked_peaks[tracked_idx];
            tracked.center = detected_peak.center;
            tracked.size = detected_peak.size;
            tracked.position_history.push(detected_peak.center);
            tracked.time_since_update = 0.0;
            tracked.total_time += delta_time;

            // Limit history length
            if tracked.position_history.len() > params.peak_tracking_history_length {
                tracked.position_history.drain(0..1);
            }
        }
    }

    // Create new tracked peaks for unmatched detected peaks
    for (detected_idx, detected_peak) in peaks_continuous.iter().enumerate() {
        if !matched_detected_indices[detected_idx] {
            tracked_peaks.push(TrackedPeak {
                id: *next_peak_id,
                center: detected_peak.center,
                size: detected_peak.size,
                position_history: vec![detected_peak.center],
                time_since_update: 0.0,
                total_time: 0.0,
            });
            *next_peak_id += 1;
        }
    }

    // Remove timed-out tracked peaks and create glissandi for significant movements
    let mut i = 0;
    while i < tracked_peaks.len() {
        let tracked = &tracked_peaks[i];

        if tracked.time_since_update > params.peak_tracking_timeout {
            // Check if this peak traveled enough to be considered a glissando
            if tracked.position_history.len() >= 2 {
                let start_pos = tracked.position_history.first().unwrap();
                let end_pos = tracked.position_history.last().unwrap();
                let total_distance = (end_pos - start_pos).abs();

                if total_distance >= params.glissando_min_distance {
                    // Create a glissando
                    let average_size = tracked.size; // Could compute average from history if we stored it

                    glissandi.push(Glissando {
                        path: tracked.position_history.clone(),
                        average_size,
                        creation_time: *elapsed_time,
                    });

                    trace!(
                        "Created glissando: {} -> {}, distance: {:.2} buckets, duration: {:.2}s",
                        start_pos,
                        end_pos,
                        total_distance,
                        tracked.total_time
                    );
                }
            }

            // Remove this tracked peak
            tracked_peaks.swap_remove(i);
        } else {
            i += 1;
        }
    }

    // Age and remove old glissandi
    glissandi.retain(|g| {
        let age = *elapsed_time - g.creation_time;
        age < params.glissando_lifetime
    });
}
