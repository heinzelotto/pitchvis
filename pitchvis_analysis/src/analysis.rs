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
    collections::HashSet,
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

        Self {
            // history: (0..SMOOTH_LENGTH)
            //     .map(|_| vec![0.0; spectrum_size])
            //     .collect(),
            //accum: (vec![0.0; spectrum_size], 0),
            //averaged: vec![0.0; spectrum_size],
            params: params.clone(),
            range,
            x_vqt_smoothed: vec![
                EmaMeasurement::new(params.vqt_smoothing_duration, 0.0);
                n_buckets
            ],
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
            smoothed_scene_calmness: EmaMeasurement::new(
                Some(params.scene_calmness_smoothing_duration),
                0.0,
            ),
            smoothed_tuning_grid_inaccuracy: EmaMeasurement::new(
                Some(params.tuning_inaccuracy_smoothing_duration),
                0.0,
            ),
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

        // smooth the vqt
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
        let peaks_continuous = enhance_peaks_continuous(&peaks, &x_vqt_smoothed, &self.range);

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

        // needs to be run after the continuous peaks have been calculated
        self.update_tuning_inaccuracy(frame_time);

        // update peak tracking for glissando detection
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
    }

    fn update_calmness(&mut self, x_vqt: &[f32], frame_time: Duration) {
        // for each bin, take the few bins around it into account as well. If the bin is a
        // peak, it is promoted as calm. Calmness currently means that the note has been
        // sustained for a while.

        // FIXME: we only take into account the notes that are currently being played. But
        // the calmness of notes is a function of their history. Should we take into account
        // the calmness of notes that are not currently being played, or that have recently
        // been released?
        // Currently, releasing a note with above average calmness decreases scene calmness.
        // Releasing a note with below average increases scene calmness.
        let mut peaks_around = vec![false; self.range.n_buckets()];
        let radius = self.range.buckets_per_octave / 12 / 2;

        // we want unsmoothed peaks for this
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

        let mut calmness_sum = 0.0;
        let mut calmness_count = 0;
        for (calmness, has_peak) in self.calmness.iter_mut().zip(peaks_around.iter()) {
            if *has_peak {
                calmness.update_with_timestep(1.0, frame_time);
                calmness_sum += calmness.get();
                calmness_count += 1;
            } else {
                calmness.update_with_timestep(0.0, frame_time);
            }
        }
        if calmness_count > 0 {
            self.smoothed_scene_calmness
                .update_with_timestep(calmness_sum / calmness_count as f32, frame_time);
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

    fn update_peak_tracking(&mut self, frame_time: Duration) {
        let delta_time = frame_time.as_secs_f32();
        self.elapsed_time += delta_time;

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
                        let average_size =
                            tracked.size; // Could compute average from history if we stored it

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
    let padding_length = 1;
    let mut x_vqt_padded_left = vec![0.0; padding_length];
    x_vqt_padded_left.extend(vqt.iter());
    let mut fp = PeakFinder::new(&x_vqt_padded_left);
    fp.with_min_prominence(peak_config.min_prominence);
    fp.with_min_height(peak_config.min_height);
    let peaks = fp.find_peaks();
    let peaks = peaks
        .iter()
        .filter(|p| {
            p.middle_position() >= padding_length + (buckets_per_octave as usize / 12 + 1) / 2
        }) // we disregard lowest A and surroundings as peaks
        .map(|p| p.middle_position() - padding_length)
        .collect::<HashSet<usize>>();

    peaks
}

/// Enhances the detected peaks by estimating the precise center and size of each peak.
///
/// This is done by quadratic interpolation. The problem is that currently the bins are
/// not equally spaced, and their frequency resolution of higher bins increases. This
/// means that the shape of the parabola must be adjusted to fit the bin. Currently, I'm
/// not even sure that at high frequencies the filters cover the entire band of a semitone.
///
/// FIXME: determine the function f(k_bin, vqt[peak-1], vqt[peak], vqt[peak+1])
fn enhance_peaks_continuous(
    discrete_peaks: &HashSet<usize>,
    vqt: &[f32],
    range: &VqtRange,
) -> Vec<ContinuousPeak> {
    let mut peaks_continuous: Vec<_> = discrete_peaks
        .iter()
        .filter_map(|p| {
            // TODO: add test for how this behaves when the values of two neighbor bins are very similar
            let p = *p;

            if p < 1 || p > range.n_buckets() - 2 {
                // FIXME: shift the center by one bin instead of discarding the peak
                return None;
            }

            let x = vqt[p] - vqt[p - 1] + f32::EPSILON;
            let y = vqt[p] - vqt[p + 1] + f32::EPSILON;

            let estimated_precise_center = p as f32 + 1.0 / (1.0 + y / x) - 0.5;
            let estimated_precise_size = vqt[estimated_precise_center.trunc() as usize + 1]
                * estimated_precise_center.fract()
                + vqt[estimated_precise_center.trunc() as usize]
                    * (1.0 - estimated_precise_center.fract());

            Some(ContinuousPeak {
                center: estimated_precise_center,
                size: estimated_precise_size,
            })
        })
        .collect();
    peaks_continuous.sort_by(|a, b| a.center.partial_cmp(&b.center).unwrap());

    peaks_continuous
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
