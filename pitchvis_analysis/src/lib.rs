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
    pub const Q: f32 = 10.0 / UPSCALE_FACTOR as f32;
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
        let sound = test_create_sines(
            &VQT_PARAMETERS,
            &[
                FREQ_A1 * 2.0.powf(3.0),
                FREQ_A1 * 2.0.powf(3.0 + 1.0 / 12.0),
            ],
            0.0,
        );
        let x_vqt = vqt.calculate_vqt_instant_in_db(&sound);

        println!("VQT: {:?}", x_vqt);

        let mut analysis = AnalysisState::new(VQT_PARAMETERS.range, 0);
        analysis.preprocess(&x_vqt, Duration::from_millis(30));

        println!(
            "Analysis: {:?}",
            analysis
                .x_vqt_smoothed
                .iter()
                .map(|x| x.get())
                .collect::<Vec<_>>()
        );

        analysis.peaks_continuous.iter().for_each(|p| {
            println!("Peak: {:?}", p);
        });
    }
}
