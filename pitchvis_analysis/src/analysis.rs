use crate::util::*;
use find_peaks::PeakFinder;

use std::{
    cmp::{max, min},
    collections::HashSet,
};

pub const SPECTROGRAM_LENGTH: usize = 400;
const PEAK_MIN_PROMINENCE: f32 = 13.0;
const PEAK_MIN_HEIGHT: f32 = 6.0;
const _BASSLINE_PEAK_MIN_PROMINENCE: f32 = 12.0;
const _BASSLINE_PEAK_MIN_HEIGHT: f32 = 4.0;
const _HIGHEST_BASSNOTE: usize = 12 * 2 + 4;
const SMOOTH_LENGTH: usize = 3;
const CALMNESS_HISTORY_LENGTH: usize = 75;

/// Represents the current state of spectral analysis for musical signals.
///
/// `AnalysisState` stores preprocessed results and history for the purpose of
/// visualizing and analyzing musical spectrums. It contains buffers for storing
/// smoothed constant-Q transform (CQT) results, peak-filtered CQT results,
/// afterglow effects, and other internal computations.
pub struct AnalysisState {
    /// A rolling history of the past few CQT frames. This history is used for smoothing and
    /// enhancing the features of the current frame.
    pub history: Vec<Vec<f32>>,

    /// The smoothed version of the current CQT frame. Smoothing is performed by averaging
    /// over the `history`.
    pub x_cqt_smoothed: Vec<f32>,

    /// Represents the current CQT frame after filtering out non-peak values.
    /// Dominant frequencies or peaks are retained while others are set to zero.
    pub x_cqt_peakfiltered: Vec<f32>,

    /// Represents the afterglow effects applied to the current CQT frame. This provides
    /// a visual decay effect, enhancing the visualization of the spectrum.
    pub x_cqt_afterglow: Vec<f32>,

    /// A set of indices that have been identified as peaks in the current frame.
    pub peaks: HashSet<usize>,

    /// Contains pairs of the estimated precise center and size of each detected peak.
    pub peaks_continuous: Vec<(f32, f32)>,

    /// A buffer for the spectrogram visualization. The size is determined by the `spectrum_size`
    /// multiplied by the `history_length` and further multiplied by 4 for RGBA color data.
    pub spectrogram_buffer: Vec<u8>,

    /// Points to the current start position in the circular `spectrogram_buffer`.
    pub spectrogram_front_idx: usize,

    /// A precomputed or user-defined list of MIDI pitches for machine learning or other
    /// algorithms, indexed by MIDI number.
    pub ml_midi_base_pitches: Vec<f32>,

    /// A buffer for storing a calmness value for each bin in the spectrum.
    pub calmness: Vec<f32>,
}

impl AnalysisState {
    /// Constructs a new instance of the `AnalysisState` with the specified spectrum size and history length.
    ///
    /// This function initializes the state required for the analysis of a musical spectrum. It preallocates buffers for history,
    /// smoothed constant-Q transform (CQT) results, peak-filtered CQT results, afterglow effects, and other internal computations.
    /// The resulting `AnalysisState` is ready to process and analyze the CQT of incoming musical signals.
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
    /// assert_eq!(analysis_state.x_cqt_smoothed.len(), 1024); // matches the spectrum_size
    /// ```
    pub fn new(spectrum_size: usize, history_length: usize) -> Self {
        let spectrogram_buffer = vec![0; spectrum_size * history_length * 4];

        Self {
            history: (0..SMOOTH_LENGTH)
                .map(|_| vec![0.0; spectrum_size])
                .collect(),
            //accum: (vec![0.0; spectrum_size], 0),
            //averaged: vec![0.0; spectrum_size],
            x_cqt_smoothed: vec![0.0; spectrum_size],
            x_cqt_peakfiltered: vec![0.0; spectrum_size],
            x_cqt_afterglow: vec![0.0; spectrum_size],
            peaks: HashSet::new(),
            peaks_continuous: Vec::new(),
            spectrogram_buffer,
            spectrogram_front_idx: 0,
            ml_midi_base_pitches: vec![0.0; 128],
            calmness: vec![0.0; spectrum_size],
        }
    }

    /// Preprocesses the given constant-Q transform (CQT) data to compute smoothed, peak-filtered results, and related analyses.
    ///
    /// This function takes in a CQT spectrum, represented by the `x_cqt` parameter, and performs several preprocessing steps:
    /// 1. Smoothing over the stored history to reduce noise and enhance relevant features.
    /// 2. Peak detection to identify dominant frequencies.
    /// 3. Continuous peak estimation for more precise spectral representation.
    /// 4. Afterglow effects calculation to create a decay effect on the visual representation of the spectrum.
    ///
    /// It's expected that the `preprocess` function will be called once per frame or per new spectrum.
    ///
    /// # Parameters:
    /// - `x_cqt`: A slice containing the constant-Q transform values of the current musical frame.
    /// - `octaves`: The number of octaves represented in the `x_cqt` data.
    /// - `buckets_per_octave`: The number of frequency buckets in each octave of the `x_cqt` data.
    ///
    /// # Panics:
    /// This function will panic if the length of `x_cqt` is not equal to `octaves * buckets_per_octave`.
    ///
    /// # Notes:
    /// - The preprocessed data, including the smoothed CQT, peak-filtered CQT, and peaks, will be stored within the `AnalysisState` object.
    /// - The `x_cqt` parameter is expected to represent the spectrum in a log-frequency scale, as is typical for a CQT representation.
    ///
    /// # Examples
    ///
    /// Assuming the setup of an `AnalysisState` object and some dummy CQT data:
    /// ```
    /// # use analysis::AnalysisState;
    /// let mut analysis_state = AnalysisState::new(1024, 10);
    /// let dummy_x_cqt = vec![0.0; 1024]; // Replace with actual CQT data
    /// analysis_state.preprocess(&dummy_x_cqt, 8, 128); // Assuming 8 octaves and 128 buckets per octave
    /// ```
    pub fn preprocess(&mut self, x_cqt: &[f32], octaves: usize, buckets_per_octave: usize) {
        let num_buckets = octaves * buckets_per_octave;

        assert!(num_buckets == x_cqt.len());

        let k_min = arg_min(&x_cqt);
        let k_max = arg_max(&x_cqt);
        let _min = x_cqt[k_min];
        let _max = x_cqt[k_max];
        // println!("x_cqt[{k_min}] = {min}, x_cqt[{k_max}] = {max}");

        // smooth by averaging over the history
        let mut x_cqt_smoothed = vec![0.0; num_buckets];
        // if a bin in the history was at peak magnitude at that time, it should be promoted
        self.history.push(x_cqt.to_owned());
        //self.history.push(x_cqt.iter().enumerate().map(|(i, x)| if peaks.contains(&i) {*x} else {0.0}).collect::<Vec<f32>>());
        if self.history.len() > SMOOTH_LENGTH {
            // TODO: once fps is implemented, make this dependent on time instead of frames
            // make smoothing range modifiable in-game
            self.history.drain(0..1);
        }
        for (i, smoothed) in x_cqt_smoothed.iter_mut().enumerate() {
            let mut v = vec![];
            for t in 0..self.history.len() {
                v.push(self.history[t][i]);
            }
            // arithmetic mean
            *smoothed = v.iter().sum::<f32>() / SMOOTH_LENGTH as f32;
        }

        // find peaks
        let peaks = find_peaks(&x_cqt_smoothed, buckets_per_octave);

        let x_cqt_peakfiltered = x_cqt_smoothed
            .iter()
            .enumerate()
            .map(|(i, x)| {
                if peaks.contains(&i) {
                    *x
                } else {
                    0.0 // x_cqt_smoothed[i] / 5.0
                }
            })
            .collect::<Vec<f32>>();

        let mut peaks_continuous = Vec::new();
        for p in &peaks {
            let p = *p;

            if p < 1 || p > octaves * buckets_per_octave - 2 {
                continue;
            }

            let x = x_cqt_smoothed[p] - x_cqt_smoothed[p - 1] + std::f32::EPSILON;
            let y = x_cqt_smoothed[p] - x_cqt_smoothed[p + 1] + std::f32::EPSILON;

            let estimated_precise_center = p as f32 + 1.0 / (1.0 + y / x) - 0.5;
            let estimated_precise_size = x_cqt_smoothed
                [estimated_precise_center.trunc() as usize + 1]
                * estimated_precise_center.fract()
                + x_cqt_smoothed[estimated_precise_center.trunc() as usize]
                    * (1.0 - estimated_precise_center.fract());
            peaks_continuous.push((estimated_precise_center, estimated_precise_size));
        }
        peaks_continuous.sort_by(|a, b| {
            if a.0 == b.0 {
                std::cmp::Ordering::Equal
            } else if a.0 < b.0 {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        self.x_cqt_afterglow
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| {
                *x *= 0.85 - 0.15 * (i as f32 / (octaves * buckets_per_octave) as f32);
                if *x < x_cqt_smoothed[i] {
                    *x = x_cqt_smoothed[i];
                }
            });
        self.peaks = peaks;
        //self.averaged = averaged;
        self.x_cqt_smoothed = x_cqt_smoothed;
        self.x_cqt_peakfiltered = x_cqt_peakfiltered;
        self.peaks_continuous = peaks_continuous;

        self.update_calmness(x_cqt, octaves, buckets_per_octave);
    }

    fn update_calmness(&mut self, x_cqt: &[f32], octaves: usize, buckets_per_octave: usize) {
        // for each bin, take the few bins around it into account as well. If the bin is a
        // peak, it is promoted as calm. Calmness currently means that the note has been
        // sustained for a while.
        let mut peaks_around = vec![false; octaves * buckets_per_octave];
        let radius = buckets_per_octave / 12 / 2;

        // we want unsmoothed peaks for this
        let peaks = find_peaks(x_cqt, buckets_per_octave);
        for p in peaks {
            for i in max(0, p as i32 - radius as i32)
                ..min(
                    (octaves * buckets_per_octave) as i32,
                    p as i32 + radius as i32,
                )
            {
                peaks_around[i as usize] = true;
            }
        }

        for i in 0..octaves * buckets_per_octave {
            if peaks_around[i] {
                self.calmness[i] += 1.0 / CALMNESS_HISTORY_LENGTH as f32;
            }

            self.calmness[i] -= self.calmness[i] / CALMNESS_HISTORY_LENGTH as f32;
        }
    }
}

fn find_peaks(cqt: &[f32], buckets_per_octave: usize) -> HashSet<usize> {
    let padding_length = 1;
    let mut x_cqt_padded_left = vec![0.0; padding_length];
    x_cqt_padded_left.extend(cqt.iter());
    let mut fp = PeakFinder::new(&x_cqt_padded_left);
    fp.with_min_prominence(PEAK_MIN_PROMINENCE);
    fp.with_min_height(PEAK_MIN_HEIGHT);
    let peaks = fp.find_peaks();
    let peaks = peaks
        .iter()
        .filter(|p| p.middle_position() >= padding_length + (buckets_per_octave / 12 + 1) / 2) // we disregard lowest A and surroundings as peaks
        .map(|p| p.middle_position() - padding_length)
        .collect::<HashSet<usize>>();

    peaks
}
