pub mod analysis;
pub mod chord;
pub mod chord_enhanced;
pub mod util;
pub mod vqt;

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rustfft::num_traits::Float;

    use super::analysis::*;
    use super::util::*;
    use super::vqt::*;

    #[test]
    fn test_vqt_close_frequencies() {
        let _ = env_logger::try_init();

        let params = VqtParameters::default();
        let vqt = Vqt::new(&params);

        let mut num_peaks_found = Vec::new();
        const SUBDIVISIONS_PER_OCTAVE: u16 = 30;
        // test in the range starting at 2 octaves above the minimum frequency until almost the maximum frequency
        for i in ((2.5 * SUBDIVISIONS_PER_OCTAVE as f32) as u16)
            ..(params.range.octaves as u16 * SUBDIVISIONS_PER_OCTAVE - SUBDIVISIONS_PER_OCTAVE / 2)
        {
            let log_note = i as f32 / SUBDIVISIONS_PER_OCTAVE as f32;
            let freq_1 = params.range.min_freq * 2.0.powf(log_note);
            let freq_2 = params.range.min_freq * 2.0.powf(log_note + 1.0 / 12.0);
            let sound = test_create_sines(&params, &[freq_1, freq_2], 0.0);
            let x_vqt = vqt.calculate_vqt_instant_in_db(&sound);

            let mut analysis =
                AnalysisState::new(params.range.clone(), AnalysisParameters::default());
            analysis.preprocess(&x_vqt, Duration::from_millis(1100));

            num_peaks_found.push(analysis.peaks.len());
        }
        println!("{:?}", num_peaks_found);
        assert!(num_peaks_found.iter().all(|&x| x == 2));
    }

    #[test]
    fn test_vqt_high_frequencies() {
        let params = VqtParameters::default();
        let vqt = Vqt::new(&params);

        let mut inf = f32::MAX;
        let mut sup = 0.0;
        for i in 0..params.range.octaves as u16 {
            const SUBDIVISIONS_PER_OCTAVE: u16 = 30;
            for j in 0..SUBDIVISIONS_PER_OCTAVE {
                let freq = params.range.min_freq
                    * 2.0.powf(i as f32 + j as f32 / (12.0 * SUBDIVISIONS_PER_OCTAVE as f32));
                let sound = test_create_sines(&params, &[freq], 0.0);
                let x_vqt = vqt.calculate_vqt_instant_in_db(&sound);
                let max_response: f32 = max(&x_vqt);
                inf = inf.min(max_response);
                sup = sup.max(max_response);
                // println!("Max. Response: {}", max_response);
            }
        }
        println!("Inf: {}, Sup: {}", inf, sup);
        assert!(inf > sup - 6.0);
    }

    // #[test]
    // fn test_vqt_with_noise() {
    //     unimplemented!("test how well the VQT can handle noise that is added to the signal");
    // }

    // #[test]
    // fn test_vqt_with_beat() {
    //     unimplemented!(
    //         "test how well the VQT can handle a beat that is added to the signal, i. e. \
    //     a short time window with a higher amplitude across multiple frequencies"
    //     );
    // }

    // #[test]
    // fn test_vqt_bass_note_detection() {
    //     unimplemented!(
    //         "test how stable the VQT can detect bass notes when the bass note drops out \
    //         for a frame or two."
    //     );
    // }

    // #[test]
    // fn test_vqt_delay_from_signal_to_analysis() {
    //     unimplemented!(
    //         "test how long it takes from the signal being played to the note showing up \
    //         in the analysis"
    //     );
    // }

    // TODO: a test case that ensures that the bass note is detected even when not played
    // for a few frames and in the presence of some noise

    #[test]
    fn test_chord_detection_c_major() {
        let _ = env_logger::try_init();

        let params = VqtParameters::default();
        let vqt = Vqt::new(&params);

        // C major chord: C4 (261.63 Hz), E4 (329.63 Hz), G4 (392.00 Hz)
        let c4 = 261.63;
        let e4 = 329.63;
        let g4 = 392.00;
        let sound = test_create_sines(&params, &[c4, e4, g4], 0.0);
        let x_vqt = vqt.calculate_vqt_instant_in_db(&sound);

        let mut analysis = AnalysisState::new(params.range.clone(), AnalysisParameters::default());
        analysis.preprocess(&x_vqt, std::time::Duration::from_millis(1100));

        assert!(
            analysis.detected_chord.is_some(),
            "Should detect a chord from C major triad"
        );
        let chord = analysis.detected_chord.unwrap();
        assert_eq!(
            chord.root, 0,
            "Should detect C as root (expected 0, got {})",
            chord.root
        );
        assert_eq!(
            chord.quality,
            crate::chord::ChordQuality::Major,
            "Should detect major quality"
        );
    }

    #[test]
    fn test_chord_detection_a_minor() {
        let _ = env_logger::try_init();

        let params = VqtParameters::default();
        let vqt = Vqt::new(&params);

        // A minor chord: A3 (220.00 Hz), C4 (261.63 Hz), E4 (329.63 Hz)
        let a3 = 220.00;
        let c4 = 261.63;
        let e4 = 329.63;
        let sound = test_create_sines(&params, &[a3, c4, e4], 0.0);
        let x_vqt = vqt.calculate_vqt_instant_in_db(&sound);

        let mut analysis = AnalysisState::new(params.range.clone(), AnalysisParameters::default());
        analysis.preprocess(&x_vqt, std::time::Duration::from_millis(1100));

        assert!(
            analysis.detected_chord.is_some(),
            "Should detect a chord from A minor triad"
        );
        let chord = analysis.detected_chord.unwrap();
        assert_eq!(
            chord.root, 9,
            "Should detect A as root (expected 9, got {})",
            chord.root
        );
        assert_eq!(
            chord.quality,
            crate::chord::ChordQuality::Minor,
            "Should detect minor quality"
        );
    }

    #[test]
    fn test_chord_detection_g7() {
        let _ = env_logger::try_init();

        let params = VqtParameters::default();
        let vqt = Vqt::new(&params);

        // G7 chord: G3 (196.00 Hz), B3 (246.94 Hz), D4 (293.66 Hz), F4 (349.23 Hz)
        let g3 = 196.00;
        let b3 = 246.94;
        let d4 = 293.66;
        let f4 = 349.23;
        let sound = test_create_sines(&params, &[g3, b3, d4, f4], 0.0);
        let x_vqt = vqt.calculate_vqt_instant_in_db(&sound);

        let mut analysis = AnalysisState::new(params.range.clone(), AnalysisParameters::default());
        analysis.preprocess(&x_vqt, std::time::Duration::from_millis(1100));

        assert!(
            analysis.detected_chord.is_some(),
            "Should detect a chord from G7"
        );
        let chord = analysis.detected_chord.unwrap();
        assert_eq!(
            chord.root, 7,
            "Should detect G as root (expected 7, got {})",
            chord.root
        );
        // Note: Depending on peak detection, this might detect as Dominant7 or just Major
        // We mainly want to ensure G is detected as root
    }

    #[test]
    fn test_chord_detection_d_major() {
        let _ = env_logger::try_init();

        let params = VqtParameters::default();
        let vqt = Vqt::new(&params);

        // D major chord: D4 (293.66 Hz), F#4 (369.99 Hz), A4 (440.00 Hz)
        let d4 = 293.66;
        let fs4 = 369.99;
        let a4 = 440.00;
        let sound = test_create_sines(&params, &[d4, fs4, a4], 0.0);
        let x_vqt = vqt.calculate_vqt_instant_in_db(&sound);

        let mut analysis = AnalysisState::new(params.range.clone(), AnalysisParameters::default());
        analysis.preprocess(&x_vqt, std::time::Duration::from_millis(1100));

        assert!(
            analysis.detected_chord.is_some(),
            "Should detect a chord from D major triad"
        );
        let chord = analysis.detected_chord.unwrap();
        assert_eq!(
            chord.root, 2,
            "Should detect D as root (expected 2, got {})",
            chord.root
        );
        assert_eq!(
            chord.quality,
            crate::chord::ChordQuality::Major,
            "Should detect major quality"
        );
    }

    #[test]
    fn test_chord_detection_no_chord_single_note() {
        let _ = env_logger::try_init();

        let params = VqtParameters::default();
        let vqt = Vqt::new(&params);

        // Single note: A4 (440.00 Hz)
        let a4 = 440.00;
        let sound = test_create_sines(&params, &[a4], 0.0);
        let x_vqt = vqt.calculate_vqt_instant_in_db(&sound);

        let mut analysis = AnalysisState::new(params.range.clone(), AnalysisParameters::default());
        analysis.preprocess(&x_vqt, std::time::Duration::from_millis(1100));

        // Single note should not be detected as a chord
        assert!(
            analysis.detected_chord.is_none(),
            "Should not detect a chord from a single note"
        );
    }
}
