use crate::util::*;
use find_peaks::PeakFinder;

use std::collections::HashSet;

pub const SPECTROGRAM_LENGTH: usize = 400;
const PEAK_MIN_PROMINENCE: f32 = 13.0;
const PEAK_MIN_HEIGHT: f32 = 6.0;
const _BASSLINE_PEAK_MIN_PROMINENCE: f32 = 12.0;
const _BASSLINE_PEAK_MIN_HEIGHT: f32 = 4.0;
const _HIGHEST_BASSNOTE: usize = 12 * 2 + 4;
const SMOOTH_LENGTH: usize = 6;

pub struct AnalysisState {
    pub history: Vec<Vec<f32>>,
    pub x_cqt_smoothed: Vec<f32>,
    pub x_cqt_afterglow: Vec<f32>,
    pub peaks: HashSet<usize>,
    pub peaks_continuous: Vec<(f32, f32)>,

    pub spectrogram_buffer: Vec<u8>,
    pub spectrogram_front_idx: usize,
}

impl AnalysisState {
    pub fn new(spectrum_size: usize, history_length: usize) -> Self {
        let spectrogram_buffer = vec![0; spectrum_size * history_length * 4];

        Self {
            history: Vec::new(),
            x_cqt_smoothed: Vec::new(),
            x_cqt_afterglow: vec![0.0; spectrum_size],
            peaks: HashSet::new(),
            peaks_continuous: Vec::new(),
            spectrogram_buffer,
            spectrogram_front_idx: 0,
        }
    }

    pub fn preprocess(&mut self, x_cqt: &[f32], octaves: usize, buckets_per_octave: usize) {
        // if self.pause_state == PauseState::Paused {
        //     return;
        // }

        let num_buckets = octaves * buckets_per_octave;

        assert!(num_buckets == x_cqt.len());

        let k_min = arg_min(x_cqt);
        let k_max = arg_max(x_cqt);
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

        // let conv_radius = (self.buckets_per_octave / 12) / 2;
        // let x_cqt_smoothed_convoluted = (0..(self.octaves * self.buckets_per_octave))
        //     .map(|idx| {
        //         if idx < conv_radius || idx >= self.octaves * self.buckets_per_octave - conv_radius
        //         {
        //             0.0
        //         } else {
        //             x_cqt_smoothed[(idx - conv_radius)..(idx + conv_radius + 1)]
        //                 .iter()
        //                 .sum::<f32>()
        //         }
        //     })
        //     .collect::<Vec<f32>>();

        // let mut pf2 = PeakFinder::new(&x_cqt_smoothed_convoluted);
        // pf2.with_min_prominence(PEAK_MIN_PROMINENCE + 5.0);
        // pf2.with_min_height(PEAK_MIN_HEIGHT + 5.0);
        // let peaks2 = pf2.find_peaks();
        // let peaks2 = peaks2
        //     .iter()
        //     .map(|p| p.middle_position())
        //     .collect::<HashSet<usize>>();

        // let mut x_cqt_smoothed = x_cqt_smoothed_convoluted
        //     .iter()
        //     .enumerate()
        //     .map(|(i, x)| {
        //         if peaks2.contains(&i) {
        //             *x / conv_radius as f32
        //         } else {
        //             x_cqt_smoothed[i] / 10.0
        //         }
        //     })
        //     .collect::<Vec<f32>>();

        // find peaks
        let padding_length = 1;
        let mut x_cqt_padded_left = vec![0.0; padding_length];
        x_cqt_padded_left.extend(x_cqt_smoothed.iter());
        let mut fp = PeakFinder::new(&x_cqt_padded_left);
        fp.with_min_prominence(PEAK_MIN_PROMINENCE);
        fp.with_min_height(PEAK_MIN_HEIGHT);
        let peaks = fp.find_peaks();
        let peaks = peaks
            .iter()
            .filter(|p| p.middle_position() >= padding_length + (buckets_per_octave / 12 + 1) / 2) // we disregard lowest A and surroundings as peaks
            .map(|p| p.middle_position() - padding_length)
            .collect::<HashSet<usize>>();

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
        self.x_cqt_smoothed = x_cqt_smoothed;
        self.peaks_continuous = peaks_continuous;
    }
}
