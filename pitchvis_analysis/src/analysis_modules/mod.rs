/// Analysis modules - independent analysis features organized by function.
///
/// This module contains individual analysis components that were previously
/// all contained in a single monolithic analysis.rs file.

pub mod afterglow;
pub mod attack_detection;
pub mod calmness;
pub mod glissando;
pub mod peak_detection;
pub mod pitch_analysis;
pub mod vibrato;

// Re-export commonly used types
pub use afterglow::{apply_peak_filter, update_afterglow};
pub use attack_detection::{detect_attacks, AttackDetectionParameters, AttackEvent, AttackState};
pub use calmness::update_calmness;
pub use glissando::{update_peak_tracking, Glissando, GlissandoParameters, TrackedPeak};
pub use peak_detection::{
    enhance_peaks_continuous, find_peaks, promote_bass_peaks_with_harmonics, ContinuousPeak,
    PeakDetectionParameters,
};
pub use pitch_analysis::{update_pitch_accuracy_and_deviation, update_tuning_inaccuracy};
pub use vibrato::{VibratoAnalysis, VibratoCategory, VibratoDetectionParameters, VibratoState};
