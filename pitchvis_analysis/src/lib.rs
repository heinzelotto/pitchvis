pub mod analysis;
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
}
