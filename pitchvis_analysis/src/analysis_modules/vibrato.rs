/// Vibrato detection module (currently disabled).
///
/// This module provides functionality for detecting and analyzing vibrato in musical signals.
/// NOTE: This feature is currently disabled as it's in a broken/unreleasable state.

use std::collections::VecDeque;

/// Vibrato health category for feedback
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VibratoCategory {
    StraightTone, // No vibrato detected
    Healthy,      // Good vibrato (4.5-8 Hz, 40-120 cents)
    Wobble,       // Too slow (< 4.5 Hz)
    Tremolo,      // Too fast (> 8 Hz)
    Excessive,    // Too wide (> 120 cents)
    Minimal,      // Too narrow (< 40 cents)
}

/// Per-note vibrato state tracking
#[derive(Debug, Clone)]
pub struct VibratoState {
    /// Frequency history (last 120 samples ≈ 2 seconds at 60 FPS)
    pub frequency_history: VecDeque<f32>,
    /// Time history (for proper rate calculation)
    pub time_history: VecDeque<f32>,
    /// Detected vibrato rate (Hz), 0.0 if no vibrato detected
    pub rate: f32,
    /// Detected vibrato extent (cents peak-to-peak)
    pub extent: f32,
    /// Vibrato regularity score (0.0 = irregular, 1.0 = perfectly regular)
    pub regularity: f32,
    /// Is vibrato currently active?
    pub is_active: bool,
    /// Center frequency of vibrato (Hz)
    pub center_frequency: f32,
    /// Confidence in vibrato detection (0.0-1.0)
    pub confidence: f32,
}

impl VibratoState {
    pub fn new() -> Self {
        Self {
            frequency_history: VecDeque::with_capacity(120),
            time_history: VecDeque::with_capacity(120),
            rate: 0.0,
            extent: 0.0,
            regularity: 0.0,
            is_active: false,
            center_frequency: 0.0,
            confidence: 0.0,
        }
    }

    /// Check if vibrato is healthy (rate and extent in acceptable ranges)
    pub fn is_healthy(&self) -> bool {
        if !self.is_active {
            return true; // No vibrato is fine
        }

        // Healthy vibrato: 4.5-8 Hz rate, 40-120 cents extent
        let rate_ok = self.rate >= 4.5 && self.rate <= 8.0;
        let extent_ok = self.extent >= 40.0 && self.extent <= 120.0;
        let regular_enough = self.regularity >= 0.6;

        rate_ok && extent_ok && regular_enough
    }

    /// Get vibrato category for feedback
    pub fn get_category(&self) -> VibratoCategory {
        if !self.is_active {
            return VibratoCategory::StraightTone;
        }

        if self.rate < 4.5 {
            VibratoCategory::Wobble // Too slow
        } else if self.rate > 8.0 {
            VibratoCategory::Tremolo // Too fast
        } else if self.extent > 120.0 {
            VibratoCategory::Excessive // Too wide
        } else if self.extent < 40.0 {
            VibratoCategory::Minimal // Too narrow
        } else {
            VibratoCategory::Healthy
        }
    }
}

impl Default for VibratoState {
    fn default() -> Self {
        Self::new()
    }
}

/// Public vibrato analysis for a detected peak/note
#[derive(Debug, Clone, Copy)]
pub struct VibratoAnalysis {
    /// Vibrato rate in Hz (0.0 if none detected)
    pub rate: f32,
    /// Vibrato extent in cents peak-to-peak
    pub extent: f32,
    /// Regularity score (0.0-1.0)
    pub regularity: f32,
    /// Is vibrato present?
    pub is_present: bool,
    /// Center frequency (Hz) of the vibrating note
    pub center_frequency: f32,
    /// Vibrato health category
    pub category: VibratoCategory,
}

#[derive(Debug, Clone)]
pub struct VibratoDetectionParameters {
    /// Minimum vibrato rate to detect (Hz)
    pub min_rate: f32,
    /// Maximum vibrato rate to detect (Hz)
    pub max_rate: f32,
    /// Minimum correlation for vibrato detection
    pub min_correlation: f32,
    /// Minimum extent to consider as vibrato (cents)
    pub min_extent: f32,
    /// History window duration (seconds)
    pub history_duration: f32,
    /// Enable vibrato peak consolidation?
    pub consolidate_peaks: bool,
}

impl Default for VibratoDetectionParameters {
    fn default() -> Self {
        Self {
            min_rate: 2.0,
            max_rate: 10.0,
            min_correlation: 0.5,
            min_extent: 20.0,
            history_duration: 2.0,
            consolidate_peaks: true,
        }
    }
}

// NOTE: The implementation methods for vibrato detection are currently disabled
// and would need to be re-enabled and debugged before use.
