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
    /// Base duration for VQT smoothing (will be modulated by frequency and calmness).
    /// Bass notes use longer smoothing, treble notes use shorter smoothing.
    vqt_smoothing_duration_base: Duration,
    /// Multiplier for smoothing duration based on scene calmness.
    /// Range: [vqt_smoothing_calmness_min, vqt_smoothing_calmness_max]
    /// At calmness=0 (energetic): uses min multiplier
    /// At calmness=1 (calm): uses max multiplier
    vqt_smoothing_calmness_min: f32,
    vqt_smoothing_calmness_max: f32,
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
            // Base smoothing of 70ms, modulated by frequency (bass longer, treble shorter)
            // and calmness (calm longer, energetic shorter)
            vqt_smoothing_duration_base: Duration::from_millis(70),
            // Calmness multiplier range: energetic 0.6x (42ms), calm 2.0x (140ms)
            vqt_smoothing_calmness_min: 0.6,
            vqt_smoothing_calmness_max: 2.0,
            note_calmness_smoothing_duration: Duration::from_millis(4_500),
            scene_calmness_smoothing_duration: Duration::from_millis(1_100),
            tuning_inaccuracy_smoothing_duration: Duration::from_millis(4_000),
            // Harmonics need to have at least 30% of fundamental's **power** (not dB!)
            harmonic_threshold: 0.3,
            attack_detection_config: AttackDetectionParameters::default(),
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
/// # use pitchvis_analysis::analysis::AnalysisState;
/// # use pitchvis_analysis::vqt::VqtRange;
/// # use std::time::Duration;
/// let range = VqtRange { min_freq: 55.0, octaves: 8, buckets_per_octave: 24 };
/// let mut analysis_state = AnalysisState::new(range.clone(), 10);
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
                let octave_fraction = bin_idx as f32 / range.buckets_per_octave as f32 / range.octaves as f32;
                let frequency_multiplier = 1.5 - 0.5 * octave_fraction;
                let duration_ms = params.vqt_smoothing_duration_base.as_millis() as f32 * frequency_multiplier;
                EmaMeasurement::new(Duration::from_millis(duration_ms as u64), 0.0)
            })
            .collect::<Vec<_>>();

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
                EmaMeasurement::new(params.note_calmness_smoothing_duration, 0.0);
                n_buckets
            ],
            released_note_calmness: vec![
                EmaMeasurement::new(params.note_calmness_smoothing_duration, 0.0);
                n_buckets
            ],
            smoothed_scene_calmness: EmaMeasurement::new(
                params.scene_calmness_smoothing_duration,
                0.0,
            ),
            smoothed_tuning_grid_inaccuracy: EmaMeasurement::new(
                params.tuning_inaccuracy_smoothing_duration,
                0.0,
            ),
            attack_state: vec![AttackState::new(); n_buckets],
            percussion_score: vec![0.0; n_buckets],
            current_attacks: Vec::new(),
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

        // Adapt smoothing duration based on scene calmness
        // More calm = longer smoothing (less responsive but cleaner)
        // More energetic = shorter smoothing (more responsive to transients)
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
                // Get base frequency-dependent duration
                let octave_fraction = bin_idx as f32 / self.range.buckets_per_octave as f32 / self.range.octaves as f32;
                let frequency_multiplier = 1.5 - 0.5 * octave_fraction;

                // Apply calmness multiplier
                let duration_ms = self.params.vqt_smoothing_duration_base.as_millis() as f32
                    * frequency_multiplier
                    * calmness_multiplier;

                smoothed.set_time_horizon(Duration::from_millis(duration_ms as u64));
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

        // TODO: more advanced bass note detection than just taking the first peak (e. g. by
        // promoting frequences through their overtones first), using different peak detection
        // parameters, etc.
        trace!(
            "bass note: {:?}",
            self.peaks_continuous
                .first()
                .map(|p| p.center.round() as usize)
        );
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
        let mut amplitude_sum = 0.0;
        for p in &self.peaks_continuous {
            amplitude_sum += p.size;

            let center_in_semitones = p.center * 12.0 / self.range.buckets_per_octave as f32;
            inaccuracy_sum += (center_in_semitones - center_in_semitones.round()).abs() * p.size;
        }

        let average_tuning_inaccuracy = if amplitude_sum > 0.0 {
            inaccuracy_sum / amplitude_sum
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
            let rate_of_change = amp_change / dt.max(0.001);  // dB/s

            // Update time since last attack
            state.time_since_attack += dt;

            // ATTACK DETECTION CRITERIA
            let attack_detected =
                rate_of_change > params.min_attack_rate &&           // Rapid increase
                current_amp > params.min_attack_amplitude &&         // Above noise floor
                amp_change > params.min_attack_delta &&              // Minimum delta
                state.time_since_attack > params.attack_cooldown;    // Cooldown period

            if attack_detected {
                // Record attack event
                state.time_since_attack = 0.0;
                state.attack_amplitude = current_amp;
                state.in_attack_phase = true;

                // Calculate percussion score inline to avoid borrowing issues
                let percussion = Self::calculate_percussion_score_static(
                    state,
                    rate_of_change,
                    &params,
                );
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
            let recent_amps: Vec<f32> = state.amplitude_history.iter()
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
        let attack_factor = ((attack_rate - params.min_attack_rate) /
                           (params.percussion_attack_threshold - params.min_attack_rate))
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
        let frame_duration = 1.0 / 60.0;  // Approximate frame time

        let sum_a: f32 = amplitudes.iter().sum();
        let sum_t: f32 = (0..amplitudes.len()).map(|i| i as f32 * frame_duration).sum();
        let sum_ta: f32 = amplitudes.iter().enumerate()
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
            let estimated_precise_center =
                bins_per_octave * (f_peak / range.min_freq).log2();

            // Evaluate parabola at the peak to get amplitude
            let c = (amplitudes[0] * log_f[1] * log_f[2]
                - amplitudes[1] * log_f[0] * log_f[2]
                + amplitudes[2] * log_f[0] * log_f[1])
                / denom;
            let estimated_precise_size = a * log_f_peak.powi(2) + b * log_f_peak + c;

            // Clamp to valid range and ensure non-negative amplitude
            Some(ContinuousPeak {
                center: estimated_precise_center
                    .clamp(0.0, range.n_buckets() as f32 - 1.0),
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
}
