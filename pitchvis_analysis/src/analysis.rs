/// Useful statistics over a series of VQT frames.
///
/// This module orchestrates various analysis components that extract interesting
/// features from VQT frames, including peak detection, calmness analysis, pitch
/// accuracy, attack detection, and more.
///
/// TODO:
/// There are lots of parameters that can be adjusted and fine-tuned to improve the detection of
/// interesting features. Also, different alternative approaches can be tried out.
///
/// Ideas to try:
///  - keyboard shortcuts in the main app to play around with these parameters on the fly
///  - different smoothing durations for different frequency ranges
///  - different peak detection parameters for bass notes detection and general note detection
///  - second VQT with lower quality for bass notes detection. ? is this better than tuning of
///    the existing VQT?
///  - drop bass notes that don't have overtones in favor of those that still fall in the bass
///    range, but do have them
///  - make note smoothing a function of the calmness of the scene, if this will not cause a
///    feedback loop
///
use crate::{util::*, vqt::VqtRange};
use std::{collections::HashSet, time::Duration};

// Import analysis modules
use crate::analysis_modules::{
    apply_peak_filter, enhance_peaks_continuous, find_peaks, promote_bass_peaks_with_harmonics,
    update_afterglow, update_calmness, update_peak_tracking, update_pitch_accuracy_and_deviation,
    update_tuning_inaccuracy,
};

// Re-export commonly used types from modules for backwards compatibility
pub use crate::analysis_modules::{
    AttackDetectionParameters, AttackEvent, AttackState, ContinuousPeak, Glissando,
    GlissandoParameters, PeakDetectionParameters, TrackedPeak, VibratoAnalysis, VibratoCategory,
    VibratoDetectionParameters, VibratoState,
};

/// Which chord detection implementation to use
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChordDetectorType {
    /// Built-in chord detector (chord.rs)
    Builtin,
    /// External chord_detector library (chord_detector_wrapper.rs)
    External,
}

#[derive(Debug, Clone)]
pub struct AnalysisParameters {
    /// Which chord detection implementation to use
    pub chord_detector_type: ChordDetectorType,
    /// The length of the spectrogram in frames.
    spectrogram_length: usize,
    /// Peak detection parameters for the general peaks.
    peak_config: PeakDetectionParameters,
    /// Peak detection parameters for the bassline peaks.
    pub bassline_peak_config: PeakDetectionParameters,
    /// The highest bass note to be considered.
    pub highest_bassnote: usize,
    /// Base duration for VQT smoothing (will be modulated by frequency and calmness).
    /// Bass notes use longer smoothing, treble notes use shorter smoothing.
    pub vqt_smoothing_duration_base: Duration,
    /// Multiplier for smoothing duration based on scene calmness.
    /// Range: [vqt_smoothing_calmness_min, vqt_smoothing_calmness_max]
    /// At calmness=0 (energetic): uses min multiplier
    /// At calmness=1 (calm): uses max multiplier
    pub vqt_smoothing_calmness_min: f32,
    pub vqt_smoothing_calmness_max: f32,
    /// The duration over which the calmness of a indivitual pitch bin is smoothed.
    pub note_calmness_smoothing_duration: Duration,
    /// The duration over which the calmness of the scene is smoothed.
    pub scene_calmness_smoothing_duration: Duration,
    /// The duration over which the tuning inaccuracy is smoothed.
    pub tuning_inaccuracy_smoothing_duration: Duration,
    /// Threshold for harmonic presence (as fraction of fundamental **power**, not dB).
    /// A harmonic is considered present if its power >= threshold * fundamental_power.
    /// Note: VQT is in dB, so we convert: power = 10^(dB/10)
    harmonic_threshold: f32,
    /// Attack detection parameters (currently disabled)
    #[allow(dead_code)]
    attack_detection_config: AttackDetectionParameters,
    /// Vibrato detection parameters (currently disabled)
    #[allow(dead_code)]
    vibrato_detection_config: VibratoDetectionParameters,
    /// Glissando detection parameters
    glissando_config: GlissandoParameters,
}

impl AnalysisParameters {
    // TODO: add setters for the parameters that shall be adjustable on the fly. Either here or on
    // AnalysisState.
}

impl Default for AnalysisParameters {
    fn default() -> Self {
        Self {
            chord_detector_type: ChordDetectorType::Builtin,
            spectrogram_length: 400,
            peak_config: PeakDetectionParameters {
                min_prominence: 10.0,
                min_height: 4.0,
            },
            bassline_peak_config: PeakDetectionParameters {
                min_prominence: 5.0,
                min_height: 3.5,
            },
            highest_bassnote: 12 * 2 + 4,
            // Base smoothing of 70ms, modulated by frequency (bass longer, treble shorter)
            // and calmness (calm longer, energetic shorter)
            vqt_smoothing_duration_base: Duration::from_millis(70),
            // Calmness multiplier range: energetic 0.6x (42ms), calm 2.0x (140ms)
            vqt_smoothing_calmness_min: 0.6,
            vqt_smoothing_calmness_max: 2.0,
            note_calmness_smoothing_duration: Duration::from_millis(3_500), // TODO: was: 4_500
            scene_calmness_smoothing_duration: Duration::from_millis(800),  // TODO: was: 1_100),
            tuning_inaccuracy_smoothing_duration: Duration::from_millis(4_000),
            // Harmonics need to have at least 30% of fundamental's **power** (not dB!)
            harmonic_threshold: 0.3,
            attack_detection_config: AttackDetectionParameters::default(),
            vibrato_detection_config: VibratoDetectionParameters::default(),
            glissando_config: GlissandoParameters::default(),
        }
    }
}

/// Represents the current state of spectral analysis for musical signals.
///
/// `AnalysisState` stores preprocessed results and history for the purpose of
/// visualizing and analyzing musical spectrums. It contains buffers for storing
/// smoothed variable-Q transform (VQT) results, peak-filtered VQT results,
/// afterglow effects, and other internal computations.
///
/// # Examples
///
/// Assuming the setup of an `AnalysisState` object and some dummy VQT data:
/// ```
/// # use pitchvis_analysis::analysis::{AnalysisState, AnalysisParameters};
/// # use pitchvis_analysis::vqt::VqtRange;
/// # use std::time::Duration;
/// let range = VqtRange { min_freq: 55.0, octaves: 8, buckets_per_octave: 24 };
/// let mut analysis_state = AnalysisState::new(range.clone(), AnalysisParameters::default());
/// let dummy_x_vqt = vec![0.0; range.n_buckets()]; // Replace with actual VQT data
/// analysis_state.preprocess(&dummy_x_vqt, Duration::from_millis(30));
/// ```
pub struct AnalysisState {
    /// The parameters used for the analysis.
    pub params: AnalysisParameters,

    /// The range of the spectrum to be analyzed, specified in octaves and buckets per octave.
    pub range: VqtRange,

    /// The smoothed version of the current VQT frame. Smoothing is performed by averaging
    /// over the `history`.
    pub x_vqt_smoothed: Vec<EmaMeasurement>,

    /// Represents the current VQT frame after filtering out non-peak values.
    /// Dominant frequencies or peaks are retained while others are set to zero.
    pub x_vqt_peakfiltered: Vec<f32>,

    /// Represents the afterglow effects applied to the current VQT frame. This provides
    /// a visual decay effect, enhancing the visualization of the spectrum.
    pub x_vqt_afterglow: Vec<f32>,

    /// A set of indices that have been identified as peaks in the current frame.
    pub peaks: HashSet<usize>,

    /// Contains pairs of the estimated precise center and size of each detected peak.
    pub peaks_continuous: Vec<ContinuousPeak>,

    /// A buffer for the spectrogram visualization. The size is determined by the `spectrum_size`
    /// multiplied by the `spectrogram_length` and further multiplied by 4 for RGBA color data.
    _spectrogram_buffer: Vec<u8>,

    /// Points to the current start position in the circular `spectrogram_buffer`.
    _spectrogram_front_idx: usize,

    /// A precomputed or user-defined list of MIDI pitches for machine learning or other
    /// algorithms, indexed by MIDI number.
    pub ml_midi_base_pitches: Vec<f32>,

    /// A buffer for storing a calmness value for each bin in the spectrum.
    pub calmness: Vec<EmaMeasurement>,

    /// Calmness values for recently released notes (decays when note is not active).
    /// This prevents abrupt calmness drops when sustained notes are released.
    released_note_calmness: Vec<EmaMeasurement>,

    /// Per-bin pitch accuracy: 1.0 = perfectly on pitch, 0.0 = maximally off pitch
    pub pitch_accuracy: Vec<f32>,

    /// Per-bin pitch deviation in semitones: negative = flat, positive = sharp, 0.0 = perfectly in tune
    pub pitch_deviation: Vec<f32>,

    /// the smoothed average calmness of all active bins
    pub smoothed_scene_calmness: EmaMeasurement,

    /// The smoothed average inaccuracy of the notes w. r. t. a 440Hz tuning, in cents (1/100)
    /// semitone.
    ///
    /// This is calculated from the absolute drifts of each note from the nearest
    /// semitone. This means that even music that is on average on pitch can have
    /// an inaccuracy, i. e. because of vibrato. But averaging over the non-absolute
    /// drifts of the notes would be worse, because something totally off-pitch at
    /// the border of +-50 cts would be average to a perfect zero, and a choir with
    /// randomly drifting voices would also average near "zero drift". The stricter
    /// pitch inaccuracy measure based on absolutes is probably more useful.
    ///
    /// Also, overtones that don't fit to the tuning grid will also impact the inaccuracy.
    pub smoothed_tuning_grid_inaccuracy: EmaMeasurement,

    /// Per-bin attack state tracking (onset detection, percussion classification)
    pub attack_state: Vec<AttackState>,

    /// Per-bin percussion score (0.0 = melodic, 1.0 = percussive)
    pub percussion_score: Vec<f32>,

    /// Attack events detected in the current frame (for visualization)
    pub current_attacks: Vec<AttackEvent>,

    /// Per-bin vibrato state tracking
    pub vibrato_states: Vec<VibratoState>,

    /// Accumulated time for vibrato analysis (seconds)
    accumulated_time: f32,

    /// Currently detected chord, if any
    pub detected_chord: Option<crate::chord::DetectedChord>,

    /// Previous chord detection for smoothing
    prev_chord_detection: Option<crate::chord::DetectedChord>,

    /// Time since last chord change (in seconds)
    time_since_chord_change: f32,

    /// Tracked peaks for glissando detection
    pub tracked_peaks: Vec<TrackedPeak>,

    /// Active glissandi for rendering
    pub glissandi: Vec<Glissando>,

    /// Counter for generating unique peak IDs
    next_peak_id: u64,

    /// Elapsed time in seconds since analysis started
    elapsed_time: f32,
}

impl AnalysisState {
    /// Constructs a new instance of the `AnalysisState` with the specified spectrum size and history length.
    ///
    /// This function initializes the state required for the analysis of a musical spectrum. It preallocates buffers for history,
    /// smoothed variable-Q transform (VQT) results, peak-filtered VQT results, afterglow effects, and other internal computations.
    /// The resulting `AnalysisState` is ready to process and analyze the VQT of incoming musical signals.
    ///
    /// # Parameters:
    /// - `range`: The size of the spectrum (octaves and buckets per octave) to be analyzed.
    /// - `history_length`: The number of past spectrums that should be retained for the spectrogram.
    ///
    /// # Returns:
    /// An initialized `AnalysisState` instance.
    pub fn new(range: VqtRange, params: AnalysisParameters) -> Self {
        let n_buckets = range.n_buckets();

        // Initialize frequency-dependent smoothing durations
        // Bass (lower octaves) get longer smoothing, treble (higher octaves) get shorter
        // Formula: base_duration * (1.5 - 0.5 * octave_fraction)
        // Octave 0 (bass): 1.5x, Octave N (treble): 1.0x
        let x_vqt_smoothed = (0..n_buckets)
            .map(|bin_idx| {
                let octave_fraction =
                    bin_idx as f32 / range.buckets_per_octave as f32 / range.octaves as f32;
                let frequency_multiplier = 1.5 - 0.5 * octave_fraction;
                let duration_ms =
                    params.vqt_smoothing_duration_base.as_millis() as f32 * frequency_multiplier;
                EmaMeasurement::new(Some(Duration::from_millis(duration_ms as u64)), 0.0)
            })
            .collect::<Vec<_>>();

        Self {
            params: params.clone(),
            range,
            x_vqt_smoothed,
            x_vqt_peakfiltered: vec![0.0; n_buckets],
            x_vqt_afterglow: vec![0.0; n_buckets],
            peaks: HashSet::new(),
            peaks_continuous: Vec::new(),
            _spectrogram_buffer: vec![0; n_buckets * params.spectrogram_length * 4],
            _spectrogram_front_idx: 0,
            ml_midi_base_pitches: vec![0.0; 128],
            calmness: vec![
                EmaMeasurement::new(Some(params.note_calmness_smoothing_duration), 0.0);
                n_buckets
            ],
            released_note_calmness: vec![
                EmaMeasurement::new(
                    Some(params.note_calmness_smoothing_duration),
                    0.0
                );
                n_buckets
            ],
            pitch_accuracy: vec![0.0; n_buckets],
            pitch_deviation: vec![0.0; n_buckets],
            smoothed_scene_calmness: EmaMeasurement::new(
                Some(params.scene_calmness_smoothing_duration),
                0.0,
            ),
            smoothed_tuning_grid_inaccuracy: EmaMeasurement::new(
                Some(params.tuning_inaccuracy_smoothing_duration),
                0.0,
            ),
            attack_state: vec![AttackState::new(); n_buckets],
            percussion_score: vec![0.0; n_buckets],
            current_attacks: Vec::new(),
            vibrato_states: vec![VibratoState::new(); n_buckets],
            accumulated_time: 0.0,
            detected_chord: None,
            prev_chord_detection: None,
            time_since_chord_change: 0.0,
            tracked_peaks: Vec::new(),
            glissandi: Vec::new(),
            next_peak_id: 0,
            elapsed_time: 0.0,
        }
    }

    /// Updates the VQT smoothing duration parameter.
    ///
    /// This updates the EmaMeasurement objects with the new time horizon while preserving
    /// their current values. When new_duration is None, smoothing is disabled and the VQT
    /// passes through without EMA filtering (direct passthrough as it was before frequency-dependent smoothing).
    ///
    /// # Parameters:
    /// - `new_duration`: The new base smoothing duration to apply, or None to disable smoothing.
    pub fn update_vqt_smoothing_duration(&mut self, new_duration: Option<Duration>) {
        // Store the base duration - use 0ms as a marker for "None" mode
        self.params.vqt_smoothing_duration_base = new_duration.unwrap_or(Duration::from_millis(0));

        // Update each bin's EMA with frequency-dependent duration
        for (bin_idx, ema) in self.x_vqt_smoothed.iter_mut().enumerate() {
            if let Some(base_duration) = new_duration {
                // Apply frequency-dependent multiplier only when smoothing is enabled
                let octave_fraction = bin_idx as f32
                    / self.range.buckets_per_octave as f32
                    / self.range.octaves as f32;
                let frequency_multiplier = 1.5 - 0.5 * octave_fraction;
                let duration_ms = base_duration.as_millis() as f32 * frequency_multiplier;
                ema.set_time_horizon(Some(Duration::from_millis(duration_ms as u64)));
            } else {
                // No smoothing - set to None for direct passthrough
                ema.set_time_horizon(None);
            }
        }
    }

    /// Preprocesses the given variable-Q transform (VQT) data to compute smoothed, peak-filtered results, and related analyses.
    ///
    /// This function takes in a VQT spectrum, represented by the `x_vqt` parameter, and performs several preprocessing steps:
    /// 1. Smoothing over the stored history to reduce noise and enhance relevant features.
    /// 2. Peak detection to identify dominant frequencies.
    /// 3. Continuous peak estimation for more precise spectral representation.
    /// 4. Afterglow effects calculation to create a decay effect on the visual representation of the spectrum.
    ///
    /// It's expected that the `preprocess` function will be called once per frame or per new spectrum.
    ///
    /// # Parameters:
    /// - `x_vqt`: A slice containing the variable-Q transform values of the current musical frame.
    /// - `frame_time`: The duration of this frame.
    ///
    /// # Panics:
    /// This function will panic if the length of `x_vqt` is not equal to the number of buckets in the range.
    pub fn preprocess(&mut self, x_vqt: &[f32], frame_time: Duration) {
        assert!(x_vqt.len() == self.range.n_buckets());

        // Adapt smoothing duration based on scene calmness (only if smoothing is enabled)
        // More calm = longer smoothing (less responsive but cleaner)
        // More energetic = shorter smoothing (more responsive to transients)
        // When base duration is 0ms, smoothing is disabled (None mode)
        let calmness = self.smoothed_scene_calmness.get();
        let calmness_multiplier = self.params.vqt_smoothing_calmness_min
            + (self.params.vqt_smoothing_calmness_max - self.params.vqt_smoothing_calmness_min)
                * calmness;

        // Update smoothing time horizons based on calmness and smooth the VQT
        self.x_vqt_smoothed
            .iter_mut()
            .enumerate()
            .zip(x_vqt.iter())
            .for_each(|((bin_idx, smoothed), x)| {
                // Only update time horizon if base duration is non-zero
                if self.params.vqt_smoothing_duration_base.as_millis() > 0 {
                    // Get base frequency-dependent duration
                    let octave_fraction = bin_idx as f32
                        / self.range.buckets_per_octave as f32
                        / self.range.octaves as f32;
                    let frequency_multiplier = 1.5 - 0.5 * octave_fraction;

                    // Apply calmness multiplier
                    let duration_ms = self.params.vqt_smoothing_duration_base.as_millis() as f32
                        * frequency_multiplier
                        * calmness_multiplier;

                    smoothed.set_time_horizon(Some(Duration::from_millis(duration_ms as u64)));
                }

                smoothed.update_with_timestep(*x, frame_time);
            });

        let x_vqt_smoothed_values = self
            .x_vqt_smoothed
            .iter()
            .map(|x| x.get())
            .collect::<Vec<f32>>();

        // Find peaks (different config for bass notes and higher notes)
        let peaks = find_peaks(
            &self.params.bassline_peak_config,
            &x_vqt_smoothed_values,
            self.range.buckets_per_octave,
        )
        .iter()
        .filter(|p| **p <= self.params.highest_bassnote)
        .chain(
            find_peaks(
                &self.params.peak_config,
                &x_vqt_smoothed_values,
                self.range.buckets_per_octave,
            )
            .iter()
            .filter(|p| **p > self.params.highest_bassnote),
        )
        .cloned()
        .collect();

        let mut peaks_continuous = enhance_peaks_continuous(&peaks, &x_vqt_smoothed_values, &self.range);

        // Boost bass peaks based on harmonic content
        promote_bass_peaks_with_harmonics(
            &mut peaks_continuous,
            &x_vqt_smoothed_values,
            &self.range,
            self.params.highest_bassnote,
            self.params.harmonic_threshold,
        );

        // Apply peak filtering
        let x_vqt_peakfiltered = apply_peak_filter(&x_vqt_smoothed_values, &peaks);

        // Update afterglow
        update_afterglow(
            &mut self.x_vqt_afterglow,
            &x_vqt_smoothed_values,
            self.range.n_buckets(),
        );

        self.peaks = peaks;
        self.x_vqt_peakfiltered = x_vqt_peakfiltered;
        self.peaks_continuous = peaks_continuous;

        // Update calmness analysis
        update_calmness(
            x_vqt,
            &x_vqt_smoothed_values,
            frame_time,
            &self.range,
            &self.params.peak_config,
            &mut self.calmness,
            &mut self.released_note_calmness,
            &mut self.smoothed_scene_calmness,
        );

        // Update tuning inaccuracy
        update_tuning_inaccuracy(
            &self.peaks_continuous,
            self.range.buckets_per_octave,
            &mut self.smoothed_tuning_grid_inaccuracy,
            frame_time,
        );

        // Update pitch accuracy and deviation
        update_pitch_accuracy_and_deviation(
            &self.peaks_continuous,
            self.range.buckets_per_octave,
            &mut self.pitch_accuracy,
            &mut self.pitch_deviation,
        );

        // Update chord detection
        self.update_chord_detection();

        // Update peak tracking for glissando detection
        update_peak_tracking(
            &self.peaks_continuous,
            frame_time,
            &self.params.glissando_config,
            &mut self.tracked_peaks,
            &mut self.glissandi,
            &mut self.next_peak_id,
            &mut self.elapsed_time,
        );

        // Update time tracking
        self.time_since_chord_change += frame_time.as_secs_f32();
        self.accumulated_time += frame_time.as_secs_f32();
    }

    /// Convert bin index to frequency (Hz)
    pub fn bin_to_frequency(&self, bin_idx: usize) -> f32 {
        self.range.min_freq * 2.0_f32.powf(bin_idx as f32 / self.range.buckets_per_octave as f32)
    }

    /// Get vibrato analysis for a specific bin (public API)
    pub fn get_vibrato_analysis(&self, bin_idx: usize) -> Option<VibratoAnalysis> {
        if bin_idx >= self.vibrato_states.len() {
            return None;
        }

        let state = &self.vibrato_states[bin_idx];

        Some(VibratoAnalysis {
            rate: state.rate,
            extent: state.extent,
            regularity: state.regularity,
            is_present: state.is_active,
            center_frequency: state.center_frequency,
            category: state.get_category(),
        })
    }

    fn update_chord_detection(&mut self) {
        use std::collections::HashMap;

        // Build map of active bins with their strengths
        let mut active_bins: HashMap<usize, f32> = HashMap::new();

        for p in &self.peaks_continuous {
            let bin_idx = p.center.round() as usize;
            if bin_idx < self.x_vqt_peakfiltered.len() {
                // Use peak size as strength
                active_bins.insert(bin_idx, p.size);
            }
        }

        // Detect chord using the selected detector
        let new_detection = match self.params.chord_detector_type {
            ChordDetectorType::Builtin => {
                // Use our built-in chord detector from chord.rs
                crate::chord::detect_chord(
                    &active_bins,
                    self.range.buckets_per_octave,
                    self.range.min_freq,
                    2, // minimum 2 notes for a chord
                )
            }
            ChordDetectorType::External => {
                // Use external chord_detector library
                crate::chord_detector_wrapper::detect_chord_with_external_lib(
                    &active_bins,
                    self.range.buckets_per_octave,
                    self.range.min_freq,
                )
                .map(|result| {
                    // Convert ChordDetectorResult to DetectedChord
                    // Parse root from debug format (e.g., "C" = 0, "C#" = 1, etc.)
                    let _root_mapping = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"];
                    let root = result.pitch_classes.first().copied().unwrap_or(0);

                    // NOTE: chord_detector library returns confidence=0.0, which doesn't match our threshold
                    // Use a default confidence that passes our threshold (0.5) since the library detected a chord
                    let confidence = if result.confidence > 0.0 {
                        result.confidence
                    } else {
                        0.7 // Default confidence when external detector finds a match
                    };

                    // Convert chromagram array to HashMap for plausibility calculation
                    let mut pitch_class_map = std::collections::HashMap::new();
                    for (pc, &power) in result.chromagram.iter().enumerate() {
                        if power > 0.0 {
                            pitch_class_map.insert(pc, power);
                        }
                    }

                    // Calculate plausibility using the chromagram data
                    let quality = crate::chord::ChordQuality::Major; // Simplified for now
                    let plausibility = crate::chord::calculate_plausibility(
                        &pitch_class_map,
                        root,
                        &quality,
                    );

                    crate::chord::DetectedChord {
                        root,
                        quality,
                        notes: result.pitch_classes,
                        confidence,
                        plausibility,
                    }
                })
            }
        };

        // Apply temporal smoothing and hysteresis to prevent oscillation
        const MIN_CONFIDENCE_THRESHOLD: f32 = 0.5;
        const CHORD_CHANGE_HYSTERESIS: f32 = 0.15;
        const MIN_STABLE_TIME: f32 = 0.15;

        // Helper function to check if two chords are the same
        let chords_match = |a: &crate::chord::DetectedChord, b: &crate::chord::DetectedChord| {
            a.root == b.root
                && std::mem::discriminant(&a.quality) == std::mem::discriminant(&b.quality)
        };

        match (&self.detected_chord, &new_detection) {
            (None, Some(new_chord)) => {
                if new_chord.confidence >= MIN_CONFIDENCE_THRESHOLD {
                    self.detected_chord = new_detection.clone();
                    self.prev_chord_detection = new_detection;
                    self.time_since_chord_change = 0.0;
                } else {
                    self.detected_chord = None;
                }
            }
            (Some(_current_chord), None) => {
                if self.time_since_chord_change > MIN_STABLE_TIME {
                    self.detected_chord = None;
                    self.prev_chord_detection = None;
                    self.time_since_chord_change = 0.0;
                }
            }
            (Some(current_chord), Some(new_chord)) => {
                if chords_match(current_chord, new_chord) {
                    let mut updated_chord = new_chord.clone();
                    updated_chord.confidence =
                        current_chord.confidence * 0.7 + new_chord.confidence * 0.3;
                    self.detected_chord = Some(updated_chord);
                } else {
                    let confidence_boost = new_chord.confidence - current_chord.confidence;
                    let should_change = (confidence_boost > CHORD_CHANGE_HYSTERESIS)
                        || (current_chord.confidence < 0.4
                            && new_chord.confidence > MIN_CONFIDENCE_THRESHOLD)
                        || (self.time_since_chord_change > MIN_STABLE_TIME
                            && new_chord.confidence > MIN_CONFIDENCE_THRESHOLD);

                    if should_change {
                        self.detected_chord = new_detection.clone();
                        self.prev_chord_detection = new_detection;
                        self.time_since_chord_change = 0.0;
                    }
                }
            }
            (None, None) => {
                self.detected_chord = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_analysis_does_something() {
        let mut analysis = AnalysisState::new(
            VqtRange {
                min_freq: 55.0,
                octaves: 2,
                buckets_per_octave: 24,
            },
            AnalysisParameters::default(),
        );
        analysis.preprocess(&vec![0.0; 48], Duration::from_secs(1));
        assert!(!analysis.x_vqt_smoothed.is_empty());
        assert!(analysis.x_vqt_smoothed.iter().all(|x| x.get() == 0.0));
    }

    // Note: Vibrato tests are currently disabled as the vibrato feature itself is disabled
}
