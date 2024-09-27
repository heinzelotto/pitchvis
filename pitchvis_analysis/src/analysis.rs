use crate::{util::*, vqt::VqtRange};
use find_peaks::PeakFinder;

use std::{
    cmp::{max, min},
    collections::HashSet,
    time::Duration,
};

pub const SPECTROGRAM_LENGTH: usize = 400;
const PEAK_MIN_PROMINENCE: f32 = 13.0;
const PEAK_MIN_HEIGHT: f32 = 6.0;
const _BASSLINE_PEAK_MIN_PROMINENCE: f32 = 12.0;
const _BASSLINE_PEAK_MIN_HEIGHT: f32 = 4.0;
const _HIGHEST_BASSNOTE: usize = 12 * 2 + 4;
// const SMOOTH_LENGTH: usize = 3;
/// The duration over which each VQT bin is smoothed.
const VQT_SMOOTHING_DURATION: Duration = Duration::from_millis(90);
/// The duration over which the calmness of a indivitual pitch bin is smoothed.
const NOTE_CALMNESS_SMOOTHING_DURATION: Duration = Duration::from_millis(4_500);
/// The duration over which the calmness of the scene is smoothed.
const SCENE_CALMNESS_SMOOTHING_DURATION: Duration = Duration::from_millis(1_100);

#[derive(Debug, Clone, Copy)]
pub struct ContinuousPeak {
    pub center: f32,
    pub size: f32,
}

/// Represents the current state of spectral analysis for musical signals.
///
/// `AnalysisState` stores preprocessed results and history for the purpose of
/// visualizing and analyzing musical spectrums. It contains buffers for storing
/// smoothed variable-Q transform (VQT) results, peak-filtered VQT results,
/// afterglow effects, and other internal computations.
pub struct AnalysisState {
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
    /// multiplied by the `history_length` and further multiplied by 4 for RGBA color data.
    pub spectrogram_buffer: Vec<u8>,

    /// Points to the current start position in the circular `spectrogram_buffer`.
    pub spectrogram_front_idx: usize,

    /// A precomputed or user-defined list of MIDI pitches for machine learning or other
    /// algorithms, indexed by MIDI number.
    pub ml_midi_base_pitches: Vec<f32>,

    /// A buffer for storing a calmness value for each bin in the spectrum.
    pub calmness: Vec<EmaMeasurement>,

    /// the smoothed average calmness of all active bins
    pub smoothed_scene_calmness: EmaMeasurement,
}

impl AnalysisState {
    /// Constructs a new instance of the `AnalysisState` with the specified spectrum size and history length.
    ///
    /// This function initializes the state required for the analysis of a musical spectrum. It preallocates buffers for history,
    /// smoothed variable-Q transform (VQT) results, peak-filtered VQT results, afterglow effects, and other internal computations.
    /// The resulting `AnalysisState` is ready to process and analyze the VQT of incoming musical signals.
    ///
    /// # Parameters:
    /// - `spectrum_size`: The size of the spectrum (or number of bins) being analyzed.
    /// - `history_length`: The number of past spectrums that should be retained for the spectrogram.
    ///
    /// # Returns:
    /// An initialized `AnalysisState` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// # use analysis::AnalysisState;
    /// let analysis_state = AnalysisState::new(1024, 10);
    /// assert_eq!(analysis_state.x_vqt_smoothed.len(), 1024); // matches the spectrum_size
    /// ```
    pub fn new(spectrum_size: usize, history_length: usize) -> Self {
        let spectrogram_buffer = vec![0; spectrum_size * history_length * 4];

        Self {
            // history: (0..SMOOTH_LENGTH)
            //     .map(|_| vec![0.0; spectrum_size])
            //     .collect(),
            //accum: (vec![0.0; spectrum_size], 0),
            //averaged: vec![0.0; spectrum_size],
            x_vqt_smoothed: vec![EmaMeasurement::new(VQT_SMOOTHING_DURATION, 0.0); spectrum_size],
            x_vqt_peakfiltered: vec![0.0; spectrum_size],
            x_vqt_afterglow: vec![0.0; spectrum_size],
            peaks: HashSet::new(),
            peaks_continuous: Vec::new(),
            spectrogram_buffer,
            spectrogram_front_idx: 0,
            ml_midi_base_pitches: vec![0.0; 128],
            calmness: vec![
                EmaMeasurement::new(NOTE_CALMNESS_SMOOTHING_DURATION, 0.0);
                spectrum_size
            ],
            smoothed_scene_calmness: EmaMeasurement::new(SCENE_CALMNESS_SMOOTHING_DURATION, 0.0),
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
    ///
    /// # Examples
    ///
    /// Assuming the setup of an `AnalysisState` object and some dummy VQT data:
    /// ```
    /// # use analysis::AnalysisState;
    /// let mut analysis_state = AnalysisState::new(1024, 10);
    /// let dummy_x_vqt = vec![0.0; 1024]; // Replace with actual VQT data
    /// analysis_state.preprocess(&dummy_x_vqt, 8, 128); // Assuming 8 octaves and 128 buckets per octave
    /// ```
    pub fn preprocess(&mut self, x_vqt: &[f32], range: &VqtRange, frame_time: Duration) {
        assert!(range.n_buckets() == x_vqt.len());

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

        // find peaks
        let peaks = find_peaks(&x_vqt_smoothed, range.buckets_per_octave);
        let peaks_continuous =
            enhance_peaks_continuous(&peaks, &x_vqt_smoothed, range);

        let x_vqt_peakfiltered = x_vqt_smoothed
            .iter()
            .enumerate()
            .map(|(i, x)| {
                if peaks.contains(&i) {
                    *x
                } else {
                    x_vqt_smoothed[i] / 5.0
                }
            })
            .collect::<Vec<f32>>();

        self.x_vqt_afterglow
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| {
                *x *= 0.85 - 0.15 * (i as f32 / range.n_buckets() as f32);
                if *x < x_vqt_smoothed[i] {
                    *x = x_vqt_smoothed[i];
                }
            });
        self.peaks = peaks;
        //self.averaged = averaged;
        // self.x_vqt_smoothed = x_vqt_smoothed;
        self.x_vqt_peakfiltered = x_vqt_peakfiltered;
        self.peaks_continuous = peaks_continuous;

        self.update_calmness(x_vqt, range, frame_time);
    }

    fn update_calmness(
        &mut self,
        x_vqt: &[f32],
        range: &VqtRange,
        frame_time: Duration,
    ) {
        // for each bin, take the few bins around it into account as well. If the bin is a
        // peak, it is promoted as calm. Calmness currently means that the note has been
        // sustained for a while.

        // FIXME: we only take into account the notes that are currently being played. But
        // the calmness of notes is a function of their history. Should we take into account
        // the calmness of notes that are not currently being played, or that have recently
        // been released?
        // Currently, releasing a note with above average calmness decreases scene calmness.
        // Releasing a note with below average increases scene calmness.
        let mut peaks_around = vec![false; range.n_buckets()];
        let radius = range.buckets_per_octave / 12 / 2;

        // we want unsmoothed peaks for this
        let peaks = find_peaks(x_vqt, range.buckets_per_octave);
        for p in peaks {
            for i in max(0, p as i32 - radius as i32)
                ..min(
                    range.n_buckets() as i32,
                    p as i32 + radius as i32,
                )
            {
                peaks_around[i as usize] = true;
            }
        }

        let mut calmness_sum = 0.0;
        let mut calmness_count = 0;
        for i in 0..range.n_buckets() {
            if peaks_around[i] {
                self.calmness[i].update_with_timestep(1.0, frame_time);
                calmness_sum += self.calmness[i].get();
                calmness_count += 1;
            } else {
                self.calmness[i].update_with_timestep(0.0, frame_time);
            }
        }
        if calmness_count > 0 {
            self.smoothed_scene_calmness
                .update_with_timestep(calmness_sum / calmness_count as f32, frame_time);
        }
    }
}

fn find_peaks(vqt: &[f32], buckets_per_octave: u16) -> HashSet<usize> {
    let padding_length = 1;
    let mut x_vqt_padded_left = vec![0.0; padding_length];
    x_vqt_padded_left.extend(vqt.iter());
    let mut fp = PeakFinder::new(&x_vqt_padded_left);
    fp.with_min_prominence(PEAK_MIN_PROMINENCE);
    fp.with_min_height(PEAK_MIN_HEIGHT);
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
    let mut peaks_continuous = Vec::new();
    for p in discrete_peaks {
        let p = *p;

        if p < 1 || p > range.n_buckets() - 2 {
            continue;
        }

        let x = vqt[p] - vqt[p - 1] + f32::EPSILON;
        let y = vqt[p] - vqt[p + 1] + f32::EPSILON;

        let estimated_precise_center = p as f32 + 1.0 / (1.0 + y / x) - 0.5;
        let estimated_precise_size = vqt[estimated_precise_center.trunc() as usize + 1]
            * estimated_precise_center.fract()
            + vqt[estimated_precise_center.trunc() as usize]
                * (1.0 - estimated_precise_center.fract());
        peaks_continuous.push(ContinuousPeak {
            center: estimated_precise_center,
            size: estimated_precise_size,
        });
    }
    peaks_continuous.sort_by(|a, b| {
        return a.center.partial_cmp(&b.center).unwrap();
    });

    peaks_continuous
}
