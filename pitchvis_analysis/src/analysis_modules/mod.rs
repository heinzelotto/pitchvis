/// Analysis modules - independent analysis features organized by function.
///
/// This module contains individual analysis components that were previously
/// all contained in a single monolithic analysis.rs file.
pub mod afterglow;
pub mod calmness;
pub mod peak_detection;
pub mod pitch_analysis;

// Re-export commonly used types
pub use afterglow::{apply_peak_filter, update_afterglow};
pub use calmness::update_calmness;
pub use peak_detection::{
    enhance_peaks_continuous, find_peaks, promote_bass_peaks_with_harmonics, ContinuousPeak,
    PeakDetectionParameters,
};
pub use pitch_analysis::{update_pitch_accuracy_and_deviation, update_tuning_inaccuracy};
