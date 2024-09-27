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

    pub const SR: u32 = 22050;
    pub const BUFSIZE: usize = 2 * SR as usize;
    pub const N_FFT: usize = 2 * 16384;
    pub const FREQ_A1: f32 = 55.0;
    pub const UPSCALE_FACTOR: u16 = 1;
    pub const BUCKETS_PER_SEMITONE: u16 = 3 * UPSCALE_FACTOR;
    pub const BUCKETS_PER_OCTAVE: u16 = 12 * BUCKETS_PER_SEMITONE;
    pub const OCTAVES: u8 = 7;
    pub const SPARSITY_QUANTILE: f32 = 0.999;
    // FIXME: for logarithmically spaced bins (fixed number of bins per octave) lower Q factor is
    // needed so that higher frequencies are fully covered by bin's bandwidths. Maybe only do Vqt
    // at lower frequencies and then do a normal FFT at higher frequencies?
    // pub const Q: f32 = 10.0 / UPSCALE_FACTOR as f32;
    pub const Q: f32 = 3.0 / UPSCALE_FACTOR as f32;
    pub const GAMMA: f32 = 5.3 * Q;

    const VQT_PARAMETERS: VqtParameters = VqtParameters {
        sr: SR as f32,
        n_fft: N_FFT,
        range: VqtRange {
            min_freq: FREQ_A1,
            buckets_per_octave: BUCKETS_PER_OCTAVE,
            octaves: OCTAVES,
        },
        sparsity_quantile: SPARSITY_QUANTILE,
        quality: Q,
        gamma: GAMMA,
    };

    #[test]
    fn test_vqt_close_frequencies() {
        let vqt = Vqt::new(&VQT_PARAMETERS);

        let mut peaks_found = Vec::new();
        for i in 0..VQT_PARAMETERS.range.octaves as u16 {
            const SUBDIVISIONS_PER_OCTAVE: u16 = 30;
            for j in 0..SUBDIVISIONS_PER_OCTAVE {
                let freq_1 = FREQ_A1
                    * 2.0.powf(i as f32 + j as f32 / (12.0 * SUBDIVISIONS_PER_OCTAVE as f32));
                let freq_2 = FREQ_A1
                    * 2.0.powf(
                        i as f32 + j as f32 / (12.0 * SUBDIVISIONS_PER_OCTAVE as f32) + 1.0 / 12.0,
                    );
                let sound = test_create_sines(&VQT_PARAMETERS, &[freq_1, freq_2], 0.0);
                let x_vqt = vqt.calculate_vqt_instant_in_db(&sound);

                let mut analysis =
                    AnalysisState::new(VQT_PARAMETERS.range, AnalysisParameters::default());
                analysis.preprocess(&x_vqt, Duration::from_millis(1000));

                peaks_found.push(analysis.peaks.len());
            }
        }
        println!("{:?}", peaks_found);
        assert!(peaks_found.iter().all(|&x| x == 2));
    }

    #[test]
    fn test_vqt_high_frequencies() {
        let vqt = Vqt::new(&VQT_PARAMETERS);

        let mut inf = f32::MAX;
        let mut sup = 0.0;
        for i in 0..VQT_PARAMETERS.range.octaves as u16 {
            const SUBDIVISIONS_PER_OCTAVE: u16 = 30;
            for j in 0..SUBDIVISIONS_PER_OCTAVE {
                let freq = FREQ_A1
                    * 2.0.powf(i as f32 + j as f32 / (12.0 * SUBDIVISIONS_PER_OCTAVE as f32));
                let sound = test_create_sines(&VQT_PARAMETERS, &[freq], 0.0);
                let x_vqt = vqt.calculate_vqt_instant_in_db(&sound);
                let max_response: f32 = max(&x_vqt);
                inf = inf.min(max_response);
                sup = sup.max(max_response);
                println!("Max. Response: {}", max_response);
            }
        }
        println!("Inf: {}, Sup: {}", inf, sup);
        assert!(inf > sup / 2.0);
    }

    #[test]
    fn test_vqt_with_noise() {
        unimplemented!("test how well the VQT can handle noise that is added to the signal");
    }

    #[test]
    fn test_vqt_with_beat() {
        unimplemented!(
            "test how well the VQT can handle a beat that is added to the signal, i. e. \
        a short time window with a higher amplitude across multiple frequencies"
        );
    }

    #[test]
    fn test_vqt_bass_note_detection() {
        unimplemented!(
            "test how stable the VQT can detect bass notes when the bass note drops out \
            for a frame or two."
        );
    }
}
