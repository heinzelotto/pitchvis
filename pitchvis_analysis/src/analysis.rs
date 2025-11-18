/// Useful statistics over a series of VQT frames.
///
/// This is currently a collection of several heuristics that are used to determine interesting
/// features of the VQT frames. These features include the calmness of the notes currently being
/// played, the average calmness of the scene, a time-smoothed version of the VQT sequence, and
/// a peak detection based on that smoothed VQT sequence.
///
/// TODO;
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
use find_peaks::PeakFinder;
use log::trace;

use std::{
    cmp::{max, min},
    collections::{HashSet, VecDeque},
    time::Duration,
};

#[derive(Debug, Clone)]
pub struct PeakDetectionParameters {
    /// The minimum prominence of a peak to be considered a peak.
    pub min_prominence: f32,
    /// The minimum height of a peak to be considered a peak.
    pub min_height: f32,
}

#[derive(Debug, Clone)]
pub struct AttackDetectionParameters {
    /// Minimum rate of amplitude increase to detect attack (dB/s)
    pub min_attack_rate: f32,
    /// Minimum amplitude for attack detection (dB)
    pub min_attack_amplitude: f32,
    /// Minimum amplitude jump to detect attack (dB)
    pub min_attack_delta: f32,
    /// Cooldown period between attacks on same bin (seconds)
    pub attack_cooldown: f32,
    /// Duration of attack phase after onset (seconds)
    pub attack_phase_duration: f32,
    /// Decay rate threshold for percussion classification (dB/s)
    pub percussion_decay_threshold: f32,
    /// Attack rate threshold for percussion classification (dB/s)
    pub percussion_attack_threshold: f32,
}

impl Default for AttackDetectionParameters {
    fn default() -> Self {
        Self {
            min_attack_rate: 50.0,
            min_attack_amplitude: 38.0,
            min_attack_delta: 3.0,
            attack_cooldown: 0.05,
            attack_phase_duration: 0.05,
            percussion_decay_threshold: 100.0,
            percussion_attack_threshold: 200.0,
        }
    }
}

/// Tracks attack state for a single frequency bin
#[derive(Debug, Clone)]
pub struct AttackState {
    /// Time since last attack (in seconds)
    pub time_since_attack: f32,
    /// Peak amplitude at time of last attack (dB)
    pub attack_amplitude: f32,
    /// Amplitude of previous frame (for rate-of-change detection)
    pub previous_amplitude: f32,
    /// Is this bin currently in attack phase? (first ~50ms after onset)
    pub in_attack_phase: bool,
    /// Amplitude history for decay rate calculation (last 10 frames)
    pub amplitude_history: VecDeque<f32>,
}

impl AttackState {
    fn new() -> Self {
        Self {
            time_since_attack: 1.0, // Start at 1s (no recent attack)
            attack_amplitude: 0.0,
            previous_amplitude: 0.0,
            in_attack_phase: false,
            amplitude_history: VecDeque::with_capacity(10),
        }
    }
}

/// Represents a detected attack event
#[derive(Debug, Clone, Copy)]
pub struct AttackEvent {
    /// Bin index where attack occurred
    pub bin_idx: usize,
    /// Frequency of the attacked note (Hz)
    pub frequency: f32,
    /// Attack amplitude (dB)
    pub amplitude: f32,
    /// Rate of amplitude increase (dB/s)
    pub attack_rate: f32,
    /// Percussion score for this attack (0.0 = melodic, 1.0 = percussive)
    pub percussion_score: f32,
}

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
    fn new() -> Self {
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

#[derive(Debug, Clone)]
pub struct AnalysisParameters {
    /// The length of the spectrogram in frames.
    spectrogram_length: usize,
    /// Peak detection parameters for the general peaks.
    peak_config: PeakDetectionParameters,
    /// Peak detection parameters for the bassline peaks.
    bassline_peak_config: PeakDetectionParameters,
    /// The highest bass note to be considered.
    highest_bassnote: usize,
    /// The duration over which each VQT bin is smoothed, or no smoothing.
    vqt_smoothing_duration: Option<Duration>,
    /// The duration over which the calmness of a indivitual pitch bin is smoothed.
    note_calmness_smoothing_duration: Duration,
    /// The duration over which the calmness of the scene is smoothed.
    scene_calmness_smoothing_duration: Duration,
    /// The duration over which the tuning inaccuracy is smoothed.
    tuning_inaccuracy_smoothing_duration: Duration,
    /// Threshold for harmonic presence (as fraction of fundamental **power**, not dB).
    /// A harmonic is considered present if its power >= threshold * fundamental_power.
    /// Note: VQT is in dB, so we convert: power = 10^(dB/10)
    harmonic_threshold: f32,
    /// Attack detection parameters
    attack_detection_config: AttackDetectionParameters,
    /// Vibrato detection parameters
    vibrato_detection_config: VibratoDetectionParameters,
    /// Maximum distance (in buckets) a peak can move between frames to be considered the same peak
    peak_tracking_max_distance: f32,
    /// Time (in seconds) after which an unmatched tracked peak is removed
    peak_tracking_timeout: f32,
    /// Maximum number of position samples to keep in history per tracked peak
    peak_tracking_history_length: usize,
    /// Minimum distance traveled (in buckets) for a tracked peak to become a glissando
    glissando_min_distance: f32,
    /// Duration (in seconds) to keep glissandos for rendering before removal
    glissando_lifetime: f32,
}

impl AnalysisParameters {
    // TODO: add setters for the parameters that shall be adjustable on the fly. Either here or on
    // AnalysisState.
}

impl Default for AnalysisParameters {
    fn default() -> Self {
        Self {
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
            vqt_smoothing_duration: Some(Duration::from_millis(90)),
            note_calmness_smoothing_duration: Duration::from_millis(4_500),
            scene_calmness_smoothing_duration: Duration::from_millis(1_100),
            tuning_inaccuracy_smoothing_duration: Duration::from_millis(4_000),
            // Harmonics need to have at least 30% of fundamental's **power** (not dB!)
            harmonic_threshold: 0.3,
            attack_detection_config: AttackDetectionParameters::default(),
            vibrato_detection_config: VibratoDetectionParameters::default(),
            // Peak tracking parameters (glissando detection)
            peak_tracking_max_distance: 3.0, // ~3 bins (~1.5 semitones for 24 buckets/octave)
            peak_tracking_timeout: 0.15,     // 150ms without match -> remove
            peak_tracking_history_length: 120, // Keep last 120 samples (~2 seconds at 60fps)
            glissando_min_distance: 2.0,     // Minimum 2 buckets (~1 semitone) traveled
            glissando_lifetime: 2.0,         // Keep glissandos visible for 2 seconds
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ContinuousPeak {
    /// The estimated precise center of the peak, in buckets starting from the min_freq.
    pub center: f32,
    /// The estimated precise amplitude of the peak, in ???
    pub size: f32,
}

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

    // /// A rolling history of the past few VQT frames. This history is used for smoothing and
    // /// enhancing the features of the current frame.
    // pub history: Vec<Vec<f32>>,
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

    /// Active glissandos for rendering
    pub glissandos: Vec<Glissando>,

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

        // Initialize VQT smoothing with uniform duration
        let x_vqt_smoothed =
            vec![EmaMeasurement::new(params.vqt_smoothing_duration, 0.0); n_buckets];

        Self {
            // history: (0..SMOOTH_LENGTH)
            //     .map(|_| vec![0.0; spectrum_size])
            //     .collect(),
            //accum: (vec![0.0; spectrum_size], 0),
            //averaged: vec![0.0; spectrum_size],
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
            glissandos: Vec::new(),
            next_peak_id: 0,
            elapsed_time: 0.0,
        }
    }

    /// Updates the VQT smoothing duration parameter.
    ///
    /// This recreates the EmaMeasurement objects with the new time horizon while preserving
    /// their current values.
    ///
    /// # Parameters:
    /// - `new_duration`: The new smoothing duration to apply.
    pub fn update_vqt_smoothing_duration(&mut self, new_duration: Option<Duration>) {
        self.params.vqt_smoothing_duration = new_duration;
        for ema in &mut self.x_vqt_smoothed {
            let current_value = ema.get();
            *ema = EmaMeasurement::new(new_duration, current_value);
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
    /// - `octaves`: The number of octaves represented in the `x_vqt` data.
    /// - `buckets_per_octave`: The number of frequency buckets in each octave of the `x_vqt` data.
    ///
    /// # Panics:
    /// This function will panic if the length of `x_vqt` is not equal to `octaves * buckets_per_octave`.
    ///
    /// # Notes:
    /// - The preprocessed data, including the smoothed VQT, peak-filtered VQT, and peaks, will be stored within the `AnalysisState` object.
    /// - The `x_vqt` parameter is expected to represent the spectrum in a log-frequency scale, as is typical for a VQT representation.
    pub fn preprocess(&mut self, x_vqt: &[f32], frame_time: Duration) {
        assert!(x_vqt.len() == self.range.n_buckets());

        let k_min = arg_min(x_vqt);
        let k_max = arg_max(x_vqt);
        let _min = x_vqt[k_min];
        let _max = x_vqt[k_max];
        // println!("x_vqt[{k_min}] = {min}, x_vqt[{k_max}] = {max}");

        // Smooth the VQT with uniform duration
        self.x_vqt_smoothed
            .iter_mut()
            .zip(x_vqt.iter())
            .for_each(|(smoothed, x)| {
                smoothed.update_with_timestep(*x, frame_time);
            });
        let x_vqt_smoothed = self
            .x_vqt_smoothed
            .iter()
            .map(|x| x.get())
            .collect::<Vec<f32>>();

        // // smooth by averaging over the history
        // let mut x_vqt_smoothed = vec![0.0; num_buckets];
        // // if a bin in the history was at peak magnitude at that time, it should be promoted
        // self.history.push(x_vqt.to_owned());
        // //self.history.push(x_vqt.iter().enumerate().map(|(i, x)| if peaks.contains(&i) {*x} else {0.0}).collect::<Vec<f32>>());
        // if self.history.len() > SMOOTH_LENGTH {
        //     // TODO: once fps is implemented, make this dependent on time instead of frames
        //     // make smoothing range modifiable in-game
        //     self.history.drain(0..1);
        // }
        // for (i, smoothed) in x_vqt_smoothed.iter_mut().enumerate() {
        //     let mut v = vec![];
        //     for t in 0..self.history.len() {
        //         v.push(self.history[t][i]);
        //     }
        //     // arithmetic mean
        //     *smoothed = v.iter().sum::<f32>() / SMOOTH_LENGTH as f32;
        // }

        // find peaks (different config for bass notes and higher notes)
        let peaks = find_peaks(
            &self.params.bassline_peak_config,
            &x_vqt_smoothed,
            self.range.buckets_per_octave,
        )
        .iter()
        .filter(|p| **p <= self.params.highest_bassnote)
        .chain(
            find_peaks(
                &self.params.peak_config,
                &x_vqt_smoothed,
                self.range.buckets_per_octave,
            )
            .iter()
            .filter(|p| **p > self.params.highest_bassnote),
        )
        .cloned()
        .collect();
        let mut peaks_continuous = enhance_peaks_continuous(&peaks, &x_vqt_smoothed, &self.range);

        // Update vibrato state for all active peaks
        // IMPORTANT: We need stable bin assignment across frames for vibrato detection.
        // Strategy: Check nearby bins for existing vibrato states and continue tracking there.
        // This handles cases where the discrete peak jumps between adjacent bins due to vibrato.
        let mut active_bins = HashSet::new();
        for peak in &peaks_continuous {
            let peak_freq = self.range.min_freq
                * 2.0_f32.powf(peak.center / self.range.buckets_per_octave as f32);
            let nominal_bin = peak.center.round() as usize;

            // Find the best bin to track this peak in:
            // 1. Check if any nearby bin (±2) already has active vibrato with similar frequency
            // 2. If found, continue tracking in that bin (maintains continuity)
            // 3. Otherwise, use the nominal bin
            let mut tracking_bin = nominal_bin;
            let search_radius = 2; // Check ±2 bins

            if nominal_bin < self.vibrato_states.len() {
                let mut best_match: Option<(usize, f32)> = None;

                for offset in 0..=search_radius {
                    for &sign in &[-1, 1] {
                        let check_bin = (nominal_bin as i32 + sign * offset) as usize;
                        if check_bin < self.vibrato_states.len() {
                            let state = &self.vibrato_states[check_bin];
                            // Check if this bin has frequency history (even if not yet "active")
                            if !state.frequency_history.is_empty() {
                                // Use average of frequency history for matching
                                let avg_freq: f32 = state.frequency_history.iter().sum::<f32>()
                                    / state.frequency_history.len() as f32;
                                if avg_freq > 0.0 {
                                    let freq_ratio = peak_freq / avg_freq;
                                    let cents_diff = 1200.0 * freq_ratio.log2().abs();

                                    // If within 150 cents (1.5 semitones), consider it the same note
                                    if cents_diff < 150.0 {
                                        if best_match.is_none()
                                            || cents_diff < best_match.unwrap().1
                                        {
                                            best_match = Some((check_bin, cents_diff));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Check the center bin (offset 0) first
                    if offset == 0 {
                        let state = &self.vibrato_states[nominal_bin];
                        if !state.frequency_history.is_empty() {
                            let avg_freq: f32 = state.frequency_history.iter().sum::<f32>()
                                / state.frequency_history.len() as f32;
                            if avg_freq > 0.0 {
                                let freq_ratio = peak_freq / avg_freq;
                                let cents_diff = 1200.0 * freq_ratio.log2().abs();
                                if cents_diff < 150.0 {
                                    if best_match.is_none() || cents_diff < best_match.unwrap().1 {
                                        best_match = Some((nominal_bin, cents_diff));
                                    }
                                }
                            }
                        }
                    }
                }

                if let Some((matched_bin, _)) = best_match {
                    tracking_bin = matched_bin;
                }

                // Update vibrato state with precise frequency
                self.update_vibrato_state(tracking_bin, peak_freq, true);
                active_bins.insert(tracking_bin);
            }
        }

        // Mark inactive bins (clear history after delay)
        // Use active_bins (from continuous peaks) instead of discrete peaks
        // because continuous peaks may round to different bins than discrete peaks
        let inactive_bins: Vec<usize> = self
            .vibrato_states
            .iter()
            .enumerate()
            .filter(|(bin_idx, _)| !active_bins.contains(bin_idx))
            .map(|(bin_idx, _)| bin_idx)
            .collect();

        for bin_idx in inactive_bins {
            self.update_vibrato_state(bin_idx, 0.0, false);
        }

        // Consolidate peaks that are part of same vibrato note (fixes double peak glitch!)
        self.consolidate_vibrato_peaks(&mut peaks_continuous);

        // Boost bass peaks based on harmonic content
        // Peaks with strong harmonics are more likely to be real bass notes vs rumble/noise
        promote_bass_peaks_with_harmonics(
            &mut peaks_continuous,
            &x_vqt_smoothed,
            &self.range,
            self.params.highest_bassnote,
            self.params.harmonic_threshold,
        );

        let x_vqt_peakfiltered = x_vqt_smoothed
            .iter()
            .enumerate()
            .map(|(i, x)| {
                if peaks.contains(&i) {
                    *x
                } else {
                    0.0 //x_vqt_smoothed[i] / 5.0
                }
            })
            .collect::<Vec<f32>>();

        self.x_vqt_afterglow
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| {
                *x *= 0.85 - 0.15 * (i as f32 / self.range.n_buckets() as f32);
                if *x < x_vqt_smoothed[i] {
                    *x = x_vqt_smoothed[i];
                }
            });
        self.peaks = peaks;
        //self.averaged = averaged;
        // self.x_vqt_smoothed = x_vqt_smoothed;
        self.x_vqt_peakfiltered = x_vqt_peakfiltered;
        self.peaks_continuous = peaks_continuous;

        self.update_calmness(x_vqt, frame_time);

        // Detect attack events (note onsets) and calculate percussion scores
        // Uses smoothed VQT for noise rejection
        self.detect_attacks(&x_vqt_smoothed, frame_time);

        // needs to be run after the continuous peaks have been calculated
        self.update_tuning_inaccuracy(frame_time);
        self.update_pitch_accuracy_and_deviation();
        self.update_chord_detection();

        // Update peak tracking for glissando detection
        self.update_peak_tracking(frame_time);

        // TODO: more advanced bass note detection than just taking the first peak (e. g. by
        // promoting frequences through their overtones first), using different peak detection
        // parameters, etc.
        trace!(
            "bass note: {:?}",
            self.peaks_continuous
                .first()
                .map(|p| p.center.round() as usize)
        );

        // Update accumulated time for vibrato detection
        self.accumulated_time += frame_time.as_secs_f32();
    }

    fn update_calmness(&mut self, x_vqt: &[f32], frame_time: Duration) {
        // For each bin, take the few bins around it into account as well. If the bin is a
        // peak, it is promoted as calm. Calmness means that the note has been sustained for a while.
        //
        // IMPROVEMENTS:
        // 1. Amplitude-weighted: Louder notes contribute more to scene calmness (matches perception)
        // 2. Released note tracking: Recently released notes contribute to prevent abrupt drops

        let mut peaks_around = vec![false; self.range.n_buckets()];
        let radius = self.range.buckets_per_octave / 12 / 2;

        // Get smoothed VQT for amplitude weighting
        let x_vqt_smoothed: Vec<f32> = self.x_vqt_smoothed.iter().map(|x| x.get()).collect();

        // We want unsmoothed peaks for this (more responsive)
        let peaks = find_peaks(
            &self.params.peak_config,
            x_vqt,
            self.range.buckets_per_octave,
        );
        for p in peaks {
            for i in max(0, p as i32 - radius as i32)
                ..min(self.range.n_buckets() as i32, p as i32 + radius as i32)
            {
                peaks_around[i as usize] = true;
            }
        }

        // Calculate amplitude-weighted scene calmness
        let mut weighted_calmness_sum = 0.0;
        let mut weight_sum = 0.0;

        for (bin_idx, ((calmness, released_calmness), has_peak)) in self
            .calmness
            .iter_mut()
            .zip(self.released_note_calmness.iter_mut())
            .zip(peaks_around.iter())
            .enumerate()
        {
            if *has_peak {
                // Note is active - update to maximum calmness
                calmness.update_with_timestep(1.0, frame_time);

                // Sync released calmness with active calmness
                *released_calmness = calmness.clone();

                // Weight by amplitude (convert dB to power for proper weighting)
                let amplitude_db = x_vqt_smoothed[bin_idx];
                let amplitude_power = 10.0_f32.powf(amplitude_db / 10.0);

                weighted_calmness_sum += calmness.get() * amplitude_power;
                weight_sum += amplitude_power;
            } else {
                // Note is not active - decay both calmness values
                calmness.update_with_timestep(0.0, frame_time);
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
            self.smoothed_scene_calmness
                .update_with_timestep(weighted_calmness_sum / weight_sum, frame_time);
        }
    }

    fn update_tuning_inaccuracy(&mut self, frame_time: Duration) {
        let mut inaccuracy_sum = 0.0;
        let mut power_sum = 0.0;
        for p in &self.peaks_continuous {
            let power = p.size * p.size;
            power_sum += power;

            let center_in_semitones = p.center * 12.0 / self.range.buckets_per_octave as f32;
            inaccuracy_sum += (center_in_semitones - center_in_semitones.round()).abs() * power;
        }

        let average_tuning_inaccuracy = if power_sum > 0.0 {
            inaccuracy_sum / power_sum
        } else {
            0.0
        };

        let average_tuning_inaccuracy_in_cents = 100.0 * average_tuning_inaccuracy;

        self.smoothed_tuning_grid_inaccuracy
            .update_with_timestep(average_tuning_inaccuracy_in_cents, frame_time);
    }

    /// Convert bin index to frequency (Hz)
    pub fn bin_to_frequency(&self, bin_idx: usize) -> f32 {
        self.range.min_freq * 2.0_f32.powf(bin_idx as f32 / self.range.buckets_per_octave as f32)
    }

    /// Detect attack events (note onsets) across all frequency bins
    fn detect_attacks(&mut self, x_vqt_smoothed: &[f32], frame_time: Duration) {
        let dt = frame_time.as_secs_f32();
        let params = self.params.attack_detection_config.clone();
        let min_freq = self.range.min_freq;
        let buckets_per_octave = self.range.buckets_per_octave;

        self.current_attacks.clear();

        for (bin_idx, state) in self.attack_state.iter_mut().enumerate() {
            let current_amp = x_vqt_smoothed[bin_idx];
            let amp_change = current_amp - state.previous_amplitude;
            let rate_of_change = amp_change / dt.max(0.001); // dB/s

            // Update time since last attack
            state.time_since_attack += dt;

            // ATTACK DETECTION CRITERIA
            let attack_detected = rate_of_change > params.min_attack_rate &&           // Rapid increase
                current_amp > params.min_attack_amplitude &&         // Above noise floor
                amp_change > params.min_attack_delta &&              // Minimum delta
                state.time_since_attack > params.attack_cooldown; // Cooldown period

            if attack_detected {
                // Record attack event
                state.time_since_attack = 0.0;
                state.attack_amplitude = current_amp;
                state.in_attack_phase = true;

                // Calculate percussion score inline to avoid borrowing issues
                let percussion =
                    Self::calculate_percussion_score_static(state, rate_of_change, &params);
                self.percussion_score[bin_idx] = percussion;

                // Calculate frequency inline
                let frequency = min_freq * 2.0_f32.powf(bin_idx as f32 / buckets_per_octave as f32);

                self.current_attacks.push(AttackEvent {
                    bin_idx,
                    frequency,
                    amplitude: current_amp,
                    attack_rate: rate_of_change,
                    percussion_score: percussion,
                });
            }

            // Exit attack phase after configured duration
            if state.time_since_attack > params.attack_phase_duration {
                state.in_attack_phase = false;
            }

            // Update amplitude history (keep last 10 values)
            state.amplitude_history.push_back(current_amp);
            if state.amplitude_history.len() > 10 {
                state.amplitude_history.pop_front();
            }

            // Store for next frame
            state.previous_amplitude = current_amp;
        }

        // Filter attacks: only keep those that coincide with actual peaks
        // This reduces false positives from noise
        let peaks = &self.peaks;
        let n_buckets = self.range.n_buckets();
        self.current_attacks.retain(|attack| {
            let start = attack.bin_idx.saturating_sub(2);
            let end = (attack.bin_idx + 2).min(n_buckets - 1);
            (start..=end).any(|b| peaks.contains(&b))
        });
    }

    /// Calculate percussion score (static version to avoid borrowing issues)
    fn calculate_percussion_score_static(
        state: &AttackState,
        attack_rate: f32,
        params: &AttackDetectionParameters,
    ) -> f32 {
        let mut percussion_score = 0.0;
        let mut factor_count = 0;

        // FACTOR 1: Decay Rate
        // Percussion decays quickly after attack
        if state.amplitude_history.len() >= 5 {
            let recent_amps: Vec<f32> = state
                .amplitude_history
                .iter()
                .rev()
                .take(5)
                .copied()
                .collect();

            let decay_rate = Self::calculate_decay_rate_static(&recent_amps);

            // High decay rate = percussive (> 100 dB/s)
            // Low decay rate = sustained (< 20 dB/s)
            let decay_factor = (decay_rate / params.percussion_decay_threshold).min(1.0);
            percussion_score += decay_factor;
            factor_count += 1;
        }

        // FACTOR 2: Attack Sharpness
        // Very sharp attacks (> 200 dB/s) are percussive
        // Slow attacks (< 50 dB/s) are melodic
        let attack_factor = ((attack_rate - params.min_attack_rate)
            / (params.percussion_attack_threshold - params.min_attack_rate))
            .clamp(0.0, 1.0);
        percussion_score += attack_factor;
        factor_count += 1;

        // Average all factors
        if factor_count > 0 {
            (percussion_score / factor_count as f32).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Calculate decay rate from amplitude history (dB/s) - static version
    /// Returns positive value for decay (amplitude decreasing)
    fn calculate_decay_rate_static(amplitudes: &[f32]) -> f32 {
        if amplitudes.len() < 2 {
            return 0.0;
        }

        // Simple linear regression: find slope of amplitude vs. sample index
        // We approximate time using frame count (assumes ~60 FPS)
        let n = amplitudes.len() as f32;
        let frame_duration = 1.0 / 60.0; // Approximate frame time

        let sum_a: f32 = amplitudes.iter().sum();
        let sum_t: f32 = (0..amplitudes.len())
            .map(|i| i as f32 * frame_duration)
            .sum();
        let sum_ta: f32 = amplitudes
            .iter()
            .enumerate()
            .map(|(i, a)| i as f32 * frame_duration * a)
            .sum();
        let sum_tt: f32 = (0..amplitudes.len())
            .map(|i| {
                let t = i as f32 * frame_duration;
                t * t
            })
            .sum();

        let slope = (n * sum_ta - sum_t * sum_a) / (n * sum_tt - sum_t * sum_t);

        // Return absolute decay rate (negative slope = decay)
        -slope
    }

    /// Convert frequency (Hz) to bin index (f32)
    fn frequency_to_bin(&self, frequency: f32) -> f32 {
        self.range.buckets_per_octave as f32 * (frequency / self.range.min_freq).log2()
    }

    /// Update vibrato state for a specific bin
    fn update_vibrato_state(&mut self, bin_idx: usize, current_freq: f32, is_peak_active: bool) {
        let state = &mut self.vibrato_states[bin_idx];
        let current_time = self.accumulated_time;

        if is_peak_active {
            // Add to history
            state.frequency_history.push_back(current_freq);
            state.time_history.push_back(current_time);

            // Keep last 2 seconds (120 samples at 60 FPS)
            let history_length =
                (self.params.vibrato_detection_config.history_duration * 60.0) as usize;
            if state.frequency_history.len() > history_length {
                state.frequency_history.pop_front();
                state.time_history.pop_front();
            }

            // Need at least 0.5 seconds of data to detect vibrato (30 samples at 60 FPS)
            if state.frequency_history.len() >= 30 {
                self.analyze_vibrato(bin_idx);
            }
        } else {
            // Peak not active - check if we should clear history
            if !state.frequency_history.is_empty() {
                if let Some(&last_time) = state.time_history.back() {
                    // Keep history for a short time (0.2s) in case peak re-appears
                    if current_time - last_time > 0.2 {
                        state.frequency_history.clear();
                        state.time_history.clear();
                        state.is_active = false;
                    }
                }
            }
        }
    }

    /// Analyze vibrato for a specific bin using autocorrelation
    fn analyze_vibrato(&mut self, bin_idx: usize) {
        let state = &mut self.vibrato_states[bin_idx];
        let params = &self.params.vibrato_detection_config;

        // Convert frequency history to cents deviation from mean
        let mean_freq: f32 =
            state.frequency_history.iter().sum::<f32>() / state.frequency_history.len() as f32;
        state.center_frequency = mean_freq;

        let cents_history: Vec<f32> = state
            .frequency_history
            .iter()
            .map(|f| 1200.0 * (f / mean_freq).log2())
            .collect();

        // Autocorrelation to find periodicity
        let (vibrato_period, correlation_strength) =
            Self::find_vibrato_period(&cents_history, &state.time_history, params);

        if correlation_strength > params.min_correlation && vibrato_period > 0.0 {
            // Vibrato detected!
            state.is_active = true;
            state.rate = 1.0 / vibrato_period; // Convert period to Hz
            state.regularity = correlation_strength;

            // Measure extent (peak-to-peak deviation)
            let max_cents = cents_history
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let min_cents = cents_history.iter().cloned().fold(f32::INFINITY, f32::min);
            state.extent = max_cents - min_cents;

            // Only consider it vibrato if extent is above minimum
            if state.extent < params.min_extent {
                state.is_active = false;
            } else {
                // Confidence based on history length and correlation
                state.confidence =
                    (state.frequency_history.len() as f32 / 120.0).min(1.0) * correlation_strength;
            }
        } else {
            // No clear periodicity - not vibrato
            state.is_active = false;
            state.rate = 0.0;
            state.extent = 0.0;
        }
    }

    /// Find vibrato period using autocorrelation - static version
    fn find_vibrato_period(
        cents_history: &[f32],
        time_history: &VecDeque<f32>,
        params: &VibratoDetectionParameters,
    ) -> (f32, f32) {
        // Remove DC component (mean)
        let mean: f32 = cents_history.iter().sum::<f32>() / cents_history.len() as f32;
        let centered: Vec<f32> = cents_history.iter().map(|x| x - mean).collect();

        // Expected vibrato period range: min_rate to max_rate Hz
        // At 60 FPS: calculate min/max lags
        let max_period = 1.0 / params.min_rate; // Maximum period in seconds
        let min_period = 1.0 / params.max_rate; // Minimum period in seconds

        // Convert to frame counts (assuming 60 FPS average)
        let min_lag = (min_period * 60.0).max(5.0) as usize; // At least 5 frames
        let max_lag = (max_period * 60.0).min(centered.len() as f32 / 2.0) as usize;

        let mut best_lag = 0;
        let mut best_correlation = 0.0;

        for lag in min_lag..=max_lag {
            if lag >= centered.len() {
                break;
            }

            // Autocorrelation at this lag
            let mut sum = 0.0;
            let mut count = 0;

            for i in 0..(centered.len() - lag) {
                sum += centered[i] * centered[i + lag];
                count += 1;
            }

            let correlation = sum / count as f32;

            // Normalize by variance
            let variance: f32 = centered.iter().map(|x| x * x).sum::<f32>() / centered.len() as f32;
            let normalized_correlation = if variance > 0.1 {
                correlation / variance
            } else {
                0.0
            };

            if normalized_correlation > best_correlation {
                best_correlation = normalized_correlation;
                best_lag = lag;
            }
        }

        // Convert lag to period in seconds
        let period = if best_lag > 0 && best_correlation > params.min_correlation {
            let time_start = time_history[0];
            let time_end = time_history[best_lag];
            time_end - time_start
        } else {
            0.0
        };

        (period, best_correlation)
    }

    /// Consolidate peaks that are part of the same vibrato note
    fn consolidate_vibrato_peaks(&self, peaks_continuous: &mut Vec<ContinuousPeak>) {
        if !self.params.vibrato_detection_config.consolidate_peaks {
            return; // Peak consolidation disabled
        }

        let mut consolidated = Vec::new();
        let mut used = vec![false; peaks_continuous.len()];

        for i in 0..peaks_continuous.len() {
            if used[i] {
                continue;
            }

            let peak = peaks_continuous[i];
            let bin_idx = peak.center.round() as usize;

            if bin_idx >= self.vibrato_states.len() {
                consolidated.push(peak);
                used[i] = true;
                continue;
            }

            let vibrato = &self.vibrato_states[bin_idx];

            if vibrato.is_active && vibrato.confidence > 0.7 {
                // This is a vibrato note - find all peaks within vibrato extent
                let extent_bins =
                    vibrato.extent / 100.0 * self.range.buckets_per_octave as f32 / 12.0;
                let search_range = extent_bins * 1.2; // 20% margin

                // Find all peaks within vibrato extent
                for j in (i + 1)..peaks_continuous.len() {
                    if used[j] {
                        continue;
                    }

                    let other_peak = peaks_continuous[j];
                    let distance = (other_peak.center - peak.center).abs();

                    if distance <= search_range {
                        // Check if this peak's frequency is in this vibrato's range
                        let other_freq = self.bin_to_frequency(other_peak.center.round() as usize);
                        let center_freq = vibrato.center_frequency;
                        let freq_deviation = 1200.0 * (other_freq / center_freq).log2();

                        if freq_deviation.abs() <= vibrato.extent / 2.0 {
                            used[j] = true;
                        }
                    }
                }

                // Create consolidated peak at vibrato center
                let consolidated_peak = ContinuousPeak {
                    center: self.frequency_to_bin(vibrato.center_frequency),
                    size: peak.size, // Keep original amplitude
                };

                consolidated.push(consolidated_peak);
                used[i] = true;
            } else {
                // Not vibrato - keep as-is
                consolidated.push(peak);
                used[i] = true;
            }
        }

        *peaks_continuous = consolidated;
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

    fn update_pitch_accuracy_and_deviation(&mut self) {
        // Reset all values to 0.0
        self.pitch_accuracy.fill(0.0);
        self.pitch_deviation.fill(0.0);

        // For each continuous peak, calculate pitch accuracy and deviation
        for p in &self.peaks_continuous {
            let center_in_semitones = p.center * 12.0 / self.range.buckets_per_octave as f32;

            // Signed deviation in semitones: negative = flat, positive = sharp
            let deviation = center_in_semitones - center_in_semitones.round();

            // Drift is absolute deviation in range [0.0, 0.5]
            let drift = deviation.abs();

            // Convert to accuracy: 1.0 = on pitch, 0.0 = maximally off pitch
            let accuracy = (1.0 - 2.0 * drift).max(0.0);

            // Assign to the corresponding bin (rounded to integer index)
            let bin_idx = p.center.round() as usize;
            if bin_idx < self.pitch_accuracy.len() {
                self.pitch_accuracy[bin_idx] = accuracy;
                self.pitch_deviation[bin_idx] = deviation;
            }
        }
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

        // Detect chord (minimum 2 notes)
        // let new_detection = crate::chord::detect_chord(
        //     &active_bins,
        //     self.range.buckets_per_octave,
        //     self.range.min_freq,
        //     2,
        // );
        // Detect chord using enhanced algorithm with rust-music-theory (minimum 2 notes)
        let new_detection = crate::chord_enhanced::detect_chord_enhanced(
            &active_bins,
            self.range.buckets_per_octave as usize,
            2,
        );

        // Apply temporal smoothing and hysteresis to prevent oscillation
        const MIN_CONFIDENCE_THRESHOLD: f32 = 0.5; // Minimum confidence to display a chord
        const CHORD_CHANGE_HYSTERESIS: f32 = 0.15; // New chord must be this much better to change
        const MIN_STABLE_TIME: f32 = 0.15; // Minimum time (seconds) before chord can change

        // Helper function to check if two chords are the same
        let chords_match = |a: &crate::chord::DetectedChord, b: &crate::chord::DetectedChord| {
            a.root == b.root
                && std::mem::discriminant(&a.quality) == std::mem::discriminant(&b.quality)
        };

        match (&self.detected_chord, &new_detection) {
            (None, Some(new_chord)) => {
                // No current chord, accept new one if confidence is high enough
                if new_chord.confidence >= MIN_CONFIDENCE_THRESHOLD {
                    self.detected_chord = new_detection.clone();
                    self.prev_chord_detection = new_detection;
                    self.time_since_chord_change = 0.0;
                } else {
                    self.detected_chord = None;
                }
            }
            (Some(_current_chord), None) => {
                // Had a chord, now detecting nothing
                // Keep current chord for a bit (grace period) to avoid flickering
                if self.time_since_chord_change > MIN_STABLE_TIME {
                    self.detected_chord = None;
                    self.prev_chord_detection = None;
                    self.time_since_chord_change = 0.0;
                }
            }
            (Some(current_chord), Some(new_chord)) => {
                if chords_match(current_chord, new_chord) {
                    // Same chord detected, update confidence with smoothing
                    let mut updated_chord = new_chord.clone();
                    // Smooth confidence to reduce jitter
                    updated_chord.confidence =
                        current_chord.confidence * 0.7 + new_chord.confidence * 0.3;
                    self.detected_chord = Some(updated_chord);
                } else {
                    // Different chord detected
                    // Only change if:
                    // 1. New chord has significantly higher confidence (with hysteresis), OR
                    // 2. Current chord has low confidence and new one is decent, OR
                    // 3. Enough time has passed for a natural chord change
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
                    // Otherwise keep the current chord
                }
            }
            (None, None) => {
                // No chord detected, nothing to do
                self.detected_chord = None;
            }
        }
    }

    fn update_peak_tracking(&mut self, frame_time: Duration) {
        let delta_time = frame_time.as_secs_f32();
        self.elapsed_time += delta_time;
        self.time_since_chord_change += delta_time;

        // Age existing tracked peaks
        for tracked in &mut self.tracked_peaks {
            tracked.time_since_update += delta_time;
        }

        // Track which peaks have been matched
        let mut matched_tracked_indices = vec![false; self.tracked_peaks.len()];
        let mut matched_detected_indices = vec![false; self.peaks_continuous.len()];

        // Match detected peaks to tracked peaks (nearest neighbor within threshold)
        for (detected_idx, detected_peak) in self.peaks_continuous.iter().enumerate() {
            let mut best_match: Option<(usize, f32)> = None;

            for (tracked_idx, tracked_peak) in self.tracked_peaks.iter().enumerate() {
                if matched_tracked_indices[tracked_idx] {
                    continue; // Already matched
                }

                let distance = (detected_peak.center - tracked_peak.center).abs();
                if distance <= self.params.peak_tracking_max_distance {
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

                let tracked = &mut self.tracked_peaks[tracked_idx];
                tracked.center = detected_peak.center;
                tracked.size = detected_peak.size;
                tracked.position_history.push(detected_peak.center);
                tracked.time_since_update = 0.0;
                tracked.total_time += delta_time;

                // Limit history length
                if tracked.position_history.len() > self.params.peak_tracking_history_length {
                    tracked.position_history.drain(0..1);
                }
            }
        }

        // Create new tracked peaks for unmatched detected peaks
        for (detected_idx, detected_peak) in self.peaks_continuous.iter().enumerate() {
            if !matched_detected_indices[detected_idx] {
                self.tracked_peaks.push(TrackedPeak {
                    id: self.next_peak_id,
                    center: detected_peak.center,
                    size: detected_peak.size,
                    position_history: vec![detected_peak.center],
                    time_since_update: 0.0,
                    total_time: 0.0,
                });
                self.next_peak_id += 1;
            }
        }

        // Remove timed-out tracked peaks and create glissandos for significant movements
        let mut i = 0;
        while i < self.tracked_peaks.len() {
            let tracked = &self.tracked_peaks[i];

            if tracked.time_since_update > self.params.peak_tracking_timeout {
                // Check if this peak traveled enough to be considered a glissando
                if tracked.position_history.len() >= 2 {
                    let start_pos = tracked.position_history.first().unwrap();
                    let end_pos = tracked.position_history.last().unwrap();
                    let total_distance = (end_pos - start_pos).abs();

                    if total_distance >= self.params.glissando_min_distance {
                        // Create a glissando
                        let average_size = tracked.size; // Could compute average from history if we stored it

                        self.glissandos.push(Glissando {
                            path: tracked.position_history.clone(),
                            average_size,
                            creation_time: self.elapsed_time,
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
                self.tracked_peaks.swap_remove(i);
            } else {
                i += 1;
            }
        }

        // Age and remove old glissandos
        self.glissandos.retain(|g| {
            let age = self.elapsed_time - g.creation_time;
            age < self.params.glissando_lifetime
        });
    }
}

fn find_peaks(
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
fn enhance_peaks_continuous(
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
fn promote_bass_peaks_with_harmonics(
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

    #[test]
    fn test_vibrato_detection_with_oscillating_peak() {
        // Test that vibrato is detected when a peak oscillates in frequency
        let range = VqtRange {
            min_freq: 55.0, // A1
            octaves: 5,
            buckets_per_octave: 24,
        };
        let mut analysis = AnalysisState::new(range.clone(), AnalysisParameters::default());

        // Simulate a vibrating note at ~440 Hz (A4) with 6 Hz vibrato rate and 50 cents extent
        // 440 Hz is approximately at bin 96 (4 octaves above 55 Hz)
        let base_bin = 96.0;
        let vibrato_rate_hz = 6.0;
        let vibrato_extent_cents = 50.0;
        let vibrato_extent_bins =
            vibrato_extent_cents / 100.0 * range.buckets_per_octave as f32 / 12.0;

        // Create VQT data with a single peak that oscillates
        let frame_duration = Duration::from_millis(16); // ~60 FPS
        let n_buckets = range.n_buckets();

        // Run for 2 seconds to allow vibrato detection
        let n_frames = 120; // 2 seconds at 60 FPS

        for frame_idx in 0..n_frames {
            let time = frame_idx as f32 * 0.016667; // 60 FPS

            // Calculate vibrating peak position
            let vibrato_phase = 2.0 * std::f32::consts::PI * vibrato_rate_hz * time;
            let peak_center = base_bin + vibrato_extent_bins / 2.0 * vibrato_phase.sin();

            // Create VQT with continuous peak shape using interpolation
            // This creates a realistic peak that smoothly moves between bins
            let mut vqt = vec![0.0; n_buckets];

            // Create a Gaussian-like peak centered at peak_center
            let peak_width = 1.5; // Standard deviation in bins
            for bin in 0..n_buckets {
                let distance = (bin as f32 - peak_center).abs();
                if distance < 5.0 {
                    // Gaussian amplitude with max 50 dB
                    let amplitude = 50.0 * (-0.5 * (distance / peak_width).powi(2)).exp();
                    vqt[bin] = amplitude;
                }
            }

            analysis.preprocess(&vqt, frame_duration);
        }

        // After 2 seconds, vibrato should be detected for bin ~96
        let bin_idx = base_bin.round() as usize;
        assert!(
            bin_idx < analysis.vibrato_states.len(),
            "Bin index out of range"
        );

        let vibrato_state = &analysis.vibrato_states[bin_idx];

        // Check that vibrato is detected
        assert!(
            vibrato_state.is_active,
            "Vibrato should be detected after 2 seconds of oscillating peak. State: {:?}",
            vibrato_state
        );

        // Check that rate is within reasonable range (vibrato smoothing may affect accuracy)
        // We expect ~6 Hz but due to VQT smoothing and other factors, allow generous tolerance
        assert!(
            vibrato_state.rate >= 2.0 && vibrato_state.rate <= 10.0,
            "Vibrato rate should be in range 2-10 Hz, got {} Hz (target was {})",
            vibrato_state.rate,
            vibrato_rate_hz
        );

        // Check that extent is detected (should be non-zero)
        assert!(
            vibrato_state.extent > 20.0,
            "Vibrato extent should be detected (>20 cents), got {} cents",
            vibrato_state.extent
        );

        // Most importantly: verify that vibrato IS detected (was the bug)
        println!("✓ Vibrato detected successfully!");
        println!(
            "  Rate: {:.2} Hz (target: {:.2} Hz)",
            vibrato_state.rate, vibrato_rate_hz
        );
        println!("  Extent: {:.1} cents", vibrato_state.extent);
        println!("  Regularity: {:.2}", vibrato_state.regularity);
        println!("  Confidence: {:.2}", vibrato_state.confidence);
    }

    #[test]
    fn test_vibrato_detection_with_large_extent() {
        // Test vibrato with large extent that causes discrete peak to jump between bins
        // This tests that we maintain stable bin tracking even when discrete peaks move
        let range = VqtRange {
            min_freq: 55.0,
            octaves: 5,
            buckets_per_octave: 24,
        };
        let mut analysis = AnalysisState::new(range.clone(), AnalysisParameters::default());

        // Simulate vibrato with 100 cents extent (2 semitones peak-to-peak)
        // This is large enough that discrete peak will jump between bins
        let base_bin = 96.0;
        let vibrato_rate_hz = 5.0;
        let vibrato_extent_cents = 100.0;
        let vibrato_extent_bins =
            vibrato_extent_cents / 100.0 * range.buckets_per_octave as f32 / 12.0; // ~2 bins

        let frame_duration = Duration::from_millis(16);
        let n_buckets = range.n_buckets();
        let n_frames = 120;

        for frame_idx in 0..n_frames {
            let time = frame_idx as f32 * 0.016667;
            let vibrato_phase = 2.0 * std::f32::consts::PI * vibrato_rate_hz * time;
            let peak_center = base_bin + vibrato_extent_bins / 2.0 * vibrato_phase.sin();

            // Create Gaussian peak
            let mut vqt = vec![0.0; n_buckets];
            let peak_width = 1.5;
            for bin in 0..n_buckets {
                let distance = (bin as f32 - peak_center).abs();
                if distance < 5.0 {
                    let amplitude = 50.0 * (-0.5 * (distance / peak_width).powi(2)).exp();
                    vqt[bin] = amplitude;
                }
            }

            analysis.preprocess(&vqt, frame_duration);
        }

        // Check if vibrato was detected in either bin 95, 96, or 97
        // (since large vibrato might be tracked in any of these bins)
        let mut found_vibrato = false;
        let mut detected_bin = 0;
        for check_bin in 95..=97 {
            if check_bin < analysis.vibrato_states.len() {
                let state = &analysis.vibrato_states[check_bin];
                if state.is_active && state.confidence > 0.7 {
                    found_vibrato = true;
                    detected_bin = check_bin;
                    println!("✓ Large-extent vibrato detected in bin {}!", check_bin);
                    println!("  Rate: {:.2} Hz", state.rate);
                    println!("  Extent: {:.1} cents", state.extent);
                    println!("  Confidence: {:.2}", state.confidence);
                    break;
                }
            }
        }

        assert!(
            found_vibrato,
            "Vibrato with large extent should be detected in at least one bin (checked 95-97)"
        );

        let state = &analysis.vibrato_states[detected_bin];
        assert!(
            state.extent > 50.0,
            "Large vibrato extent should be detected (>50 cents), got {} cents",
            state.extent
        );
    }
}
