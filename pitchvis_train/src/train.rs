use anyhow::Result;
use rustysynth::*;
use std::fs::File;
use std::sync::Arc;

pub const SR: usize = 22050;
pub const BUFSIZE: usize = 2 * SR;
pub const N_FFT: usize = 2 * 16384;
pub const FREQ_A1: f32 = 27.5;
pub const FREQ_A1_MIDI_KEY_ID: i32 = 21;
pub const UPSCALE_FACTOR: usize = 4;
pub const BUCKETS_PER_SEMITONE: usize = 5 * UPSCALE_FACTOR;
pub const BUCKETS_PER_OCTAVE: usize = 12 * BUCKETS_PER_SEMITONE;
pub const OCTAVES: usize = 7;
pub const SPARSITY_QUANTILE: f32 = 0.999;
pub const Q: f32 = 0.7 / UPSCALE_FACTOR as f32;
pub const GAMMA: f32 = 11.6 / UPSCALE_FACTOR as f32;

// pub const STEP_MS: usize = 150;

pub fn train() -> Result<()> {
    let mut cqt = pitchvis_analysis::cqt::Cqt::new(
        SR,
        N_FFT,
        FREQ_A1,
        BUCKETS_PER_OCTAVE,
        OCTAVES,
        SPARSITY_QUANTILE,
        Q,
        GAMMA,
    );

    let cqt_delay_in_samples = (cqt.delay.as_millis() as usize * SR) / 1000;
    let cqt_delay_in_samples = (cqt_delay_in_samples / 64) * 64; // round to multiple of 64
    println!(
        "anlyzing MIDI with cqt delay {}ms (= {} samples).",
        cqt.delay.as_millis(),
        cqt_delay_in_samples
    );
    // let step_size_in_chunks = 1;

    let midi_path = "example.mid";
    let soundfont_path = "MuseScore_General.sf2";
    let annotated_cqt = synthesize_midi_to_wav(
        midi_path,
        soundfont_path,
        cqt_delay_in_samples,
        // step_size_in_chunks,
        &mut cqt,
    )?;

    println!("{}", annotated_cqt.len());

    let mut positive_x = Vec::new();
    let mut negative_x = Vec::new();
    for x in annotated_cqt {
        let (p_x, n_x) = center_cqt_and_generate_positive_and_negative_data_points(&x);
        positive_x.extend_from_slice(&p_x);
        negative_x.extend_from_slice(&n_x);
    }

    // average all positive_x
    let mut avg_positive_x = vec![0.0; positive_x[0].len()];
    for v in &positive_x {
        for (i, x) in v.iter().enumerate() {
            avg_positive_x[i] += x;
        }
    }
    for x in &mut avg_positive_x {
        *x /= positive_x.len() as f32;
    }

    println!("avg_positive_x: {:?}", avg_positive_x);

    // let (positive_x, negative_x) = annotated_cqt
    //     .iter()
    //     .flat_map(|x| center_cqt_and_generate_positive_and_negative_data_points(x))
    //     .collect();

    //println!("Active keys: {:?}\ncqt: {:?}", prev_active_keys, x_cqt);
    // let mut v = vec![0f32; 86 * BUCKETS_PER_SEMITONE];
    // let note = 40;
    // for (i, x) in v.iter_mut().enumerate() {
    //     *x = i.abs_diff(note) as f32;
    // }

    // let an = (vec![((note + 33) as i32, 1.0)], v);
    // let bla = center_cqt_and_generate_positive_and_negative_data_points(&an);
    // println!("annotated_cqt {:?}", an);
    // println!("positive samples:");
    // for x in &bla.0 {
    //     println!("{:?}", x);
    // }
    // println!("negative samples:");
    // for x in &bla.1 {
    //     println!("{:?}", x);
    // }

    Ok(())
}

fn synthesize_midi_to_wav(
    midi_path: &str,
    soundfont_path: &str,
    cqt_delay_in_samples: usize,
    // step_size_in_chunks: usize,
    cqt: &mut pitchvis_analysis::cqt::Cqt,
) -> Result<Vec<(Vec<(i32, f32)>, Vec<f32>)>> {
    let mut sf2 = File::open(soundfont_path).unwrap();
    let sound_font = Arc::new(SoundFont::new(&mut sf2).unwrap());

    // Load the MIDI file.
    let mut mid = File::open(midi_path).unwrap();
    let midi_file = Arc::new(MidiFile::new(&mut mid).unwrap());

    // Create the MIDI file sequencer.
    let settings = SynthesizerSettings::new(SR as i32);
    let synthesizer = Synthesizer::new(&sound_font, &settings).unwrap();
    let mut sequencer = MidiFileSequencer::new(synthesizer);

    // Play the MIDI file.
    sequencer.play(&midi_file, false);

    let sample_count = (settings.sample_rate as f64 * midi_file.get_length()) as usize;

    let mut ring_buffer = Vec::new();
    ring_buffer.resize(BUFSIZE, 0f32);

    // chunk size is cqt delay so that we get a reading of active keys every cqt delay
    let mut left: Vec<f32> = vec![0_f32; cqt_delay_in_samples];
    let mut right: Vec<f32> = vec![0_f32; cqt_delay_in_samples];

    let mut annotated_cqt = Vec::new();
    let mut written = 0;
    let mut prev_active_keys;
    let mut active_keys = Vec::new();
    while written < sample_count {
        // Render the waveform.
        sequencer.render(&mut left[..], &mut right[..]);
        written += left.len();

        // downmix to mono into left channel and add to ring buffer
        left.iter_mut()
            .zip(right.iter_mut())
            .for_each(|(l, r)| *l = (*l + *r) / 2.0);
        ring_buffer.drain(..left.len());
        ring_buffer.extend_from_slice(left.as_slice());

        // get active keys
        prev_active_keys = active_keys;
        active_keys = sequencer
            .synthesizer
            .get_active_voices()
            .iter()
            .map(|voice| {
                (
                    voice.key,
                    (voice.current_mix_gain_left + voice.current_mix_gain_right) / 2.0,
                )
            })
            .collect::<Vec<(i32, f32)>>();

        // perform cqt analysis
        let x_cqt =
            cqt.calculate_cqt_instant_in_db(&ring_buffer[(ring_buffer.len() - cqt.n_fft)..]);

        //println!("Active keys: {:?}\ncqt: {:?}", prev_active_keys, x_cqt);
        annotated_cqt.push((prev_active_keys, x_cqt));
    }

    Ok(annotated_cqt)
}

/// This function is used to generate positive and negative data points for a logistic regression model, based on the Constant Q Transform (CQT) of a segment of audio and its corresponding active MIDI keys.
///
/// # Arguments
///
/// * `annotated_cqt`: a tuple containing a vector of active MIDI keys and their intensities, and a vector of CQT transform values.
///
/// # Returns
///
/// * A tuple of two vectors, each containing vectors of CQT values:
///     * The first vector contains the positive samples. For each active key, the CQT is shifted such that the active key is centered, and the CQT values are then truncated/padded to ensure a constant size of 85 semitones (* BUCKETS_PER_SEMITONE), with 40 semitones below and 45 semitones above the active key.
///     * The second vector contains the negative samples. For each active key, several shifted versions of the CQT are generated. The shifts correspond to +/- 3, 4, 5, 6, 7, 8, 9, 12, 19, 24 semitones, but only if the shifted key does not come too close to another active key (i.e., the distance between the shifted key and any other active key is at least a semitones). Like the positive samples, these CQT values are then truncated/padded to ensure a constant size.
///
/// This function is mainly used for preparing data for training a logistic regression model to predict the active MIDI keys from the CQT of a segment of audio.
fn center_cqt_and_generate_positive_and_negative_data_points(
    annotated_cqt: &(Vec<(i32, f32)>, Vec<f32>),
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let (active_keys, cqt_transform) = annotated_cqt;
    let mut positive_samples = Vec::new();
    let mut negative_samples = Vec::new();
    let shift_values = vec![
        -24, -19, -12, -9, -8, -7, -6, -5, -4, -3, 3, 4, 5, 6, 7, 8, 9, 12, 19, 24,
    ];

    for &(key, _) in active_keys.iter() {
        let key_index = (key - FREQ_A1_MIDI_KEY_ID) as usize * BUCKETS_PER_SEMITONE;

        let calc_boundaries = |key_index: usize| {
            let start = key_index
                .checked_sub(40 * BUCKETS_PER_SEMITONE)
                .unwrap_or(0);
            let start_overshoot = (40 * BUCKETS_PER_SEMITONE)
                .checked_sub(key_index)
                .unwrap_or(0);
            let end = (key_index + 46 * BUCKETS_PER_SEMITONE).min(cqt_transform.len());
            let end_overshoot = (key_index + 46 * BUCKETS_PER_SEMITONE)
                .checked_sub(cqt_transform.len())
                .unwrap_or(0);
            (start, start_overshoot, end, end_overshoot)
        };

        // Positive sample
        if key < FREQ_A1_MIDI_KEY_ID || key >= FREQ_A1_MIDI_KEY_ID + OCTAVES as i32 * 12 {
            // println!("midi Key {} is out of range of cqt", key);
            continue;
        }
        let (start, start_overshoot, end, end_overshoot) = calc_boundaries(key_index);
        let mut sample = vec![0f32; 87 * BUCKETS_PER_SEMITONE];
        sample.splice(
            (start_overshoot)..(87 * BUCKETS_PER_SEMITONE - end_overshoot),
            cqt_transform[start..end].iter().cloned(),
        );
        positive_samples.push(sample);

        // Negative samples
        for &shift in shift_values.iter() {
            let shifted_key = key as i32 + shift;
            if shifted_key < FREQ_A1_MIDI_KEY_ID
                || shifted_key >= FREQ_A1_MIDI_KEY_ID + OCTAVES as i32 * 12
            {
                // println!(
                //     "shifted midi Key {} + {} is out of range of cqt",
                //     key, shift
                // );
                continue;
            }

            // Check distance to other active keys
            if active_keys
                .iter()
                .all(|&(other_key, _)| (other_key - shifted_key).abs() >= 1)
            {
                let shifted_key_index =
                    ((key as i32 + shift) - FREQ_A1_MIDI_KEY_ID) as usize * BUCKETS_PER_SEMITONE;

                let (start, start_overshoot, end, end_overshoot) =
                    calc_boundaries(shifted_key_index);
                let mut sample = vec![0f32; 87 * BUCKETS_PER_SEMITONE];
                sample.splice(
                    (start_overshoot)..(87 * BUCKETS_PER_SEMITONE - end_overshoot),
                    cqt_transform[start..end].iter().cloned(),
                );
                negative_samples.push(sample);
            }
        }
    }

    (positive_samples, negative_samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_center_cqt_and_generate_positive_and_negative_data_points() {
        for note in 0..50 {
            let active_keys = vec![((33 + note) as i32, 1.0)]; // MIDI id for FREQ_A1 is 33
            let mut cqt_transform = vec![0f32; 86 * BUCKETS_PER_SEMITONE];
            for (i, x) in cqt_transform.iter_mut().enumerate() {
                *x = (1_000 - i.abs_diff(note)) as f32;
            }
            let annotated_cqt = (active_keys, cqt_transform);

            let (positive_samples, negative_samples) =
                center_cqt_and_generate_positive_and_negative_data_points(&annotated_cqt);

            // Check that the positive samples are correctly centered
            for sample in positive_samples.iter() {
                assert_eq!(sample.len(), 86 * BUCKETS_PER_SEMITONE);
                assert_eq!(sample[40 * BUCKETS_PER_SEMITONE], 1_000f32);
            }

            // Check that the negative samples are correctly centered and shifted
            for (i, &shift) in [3, 4, 5, 6, 7, 8, 9, 12, 19, 24].iter().enumerate() {
                assert_eq!(negative_samples[i].len(), 86 * BUCKETS_PER_SEMITONE);
                assert_eq!(
                    negative_samples[i][(40 + shift) * BUCKETS_PER_SEMITONE],
                    0f32
                );
            }
        }
    }
}
