use anyhow::Result;
use byteorder::{LittleEndian, WriteBytesExt};
use linfa::prelude::*;
use linfa::traits::Fit;
use linfa::traits::Predict;
use ndarray::Axis;
use npyz::TypeStr;
use npyz::WriterBuilder;
use pitchvis_analysis::cqt;
use pitchvis_analysis::util::arg_max;
use rayon::prelude::*;
use rustysynth::*;
use serde_big_array::BigArray;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io;
use std::io::Write;
use std::mem::transmute;
use std::sync::Arc;

// increasing BUCKETS_PER_SEMITONE or Q will improve frequency resolution at cost of time resolution,
// increasing GAMMA will improve time resolution at lower frequencies.
pub const SR: usize = 22050;
pub const BUFSIZE: usize = 2 * SR;
pub const N_FFT: usize = 2 * 16384;
pub const FREQ_A1: f32 = 55.0;
pub const FREQ_A1_MIDI_KEY_ID: i32 = 33;
pub const UPSCALE_FACTOR: usize = 1;
pub const BUCKETS_PER_SEMITONE: usize = 3 * UPSCALE_FACTOR;
pub const BUCKETS_PER_OCTAVE: usize = 12 * BUCKETS_PER_SEMITONE;
pub const OCTAVES: usize = 7;
pub const SPARSITY_QUANTILE: f32 = 0.999;
pub const Q: f32 = 10.0 / UPSCALE_FACTOR as f32;
pub const GAMMA: f32 = 5.3 * Q;

pub const STEP_SIZE_IN_CHUNKS: usize = 3;

fn fit(positive: Vec<(Vec<f32>, f32)>, negative: Vec<(Vec<f32>, f32)>) {
    // Combine positive and negative data into one dataset
    let len_p = positive.len();
    let len_n = negative.len();
    // dbg!(&len_p, &len_n);
    let features: Vec<(Vec<f32>, f32)> = positive.into_iter().chain(negative.into_iter()).collect();
    let targets: Vec<f32> = vec![1.0; len_p]
        .into_iter()
        .chain(vec![0.0; len_n].into_iter())
        .collect();

    // Convert your data to Array2, which is required by linfa
    let features_shape = dbg!((features.len(), features[0].0.len()));
    let features_data = features
        .into_iter()
        .map(|x| x.0)
        .flatten()
        .collect::<Vec<f32>>();
    let records_array = ndarray::Array2::from_shape_vec(features_shape, features_data).unwrap();
    let targets_array = ndarray::Array1::from_shape_vec(targets.len(), targets).unwrap();

    // Create a dataset
    let dataset: Dataset<f32, &str, ndarray::Ix1> = DatasetBase::new(records_array, targets_array)
        .map_targets(|x: &f32| if *x == 1.0 { "positive" } else { "negative" });

    // // Split the dataset into a training and a validation set
    let mut rng = rand::thread_rng();
    let (train, valid) = dataset.shuffle(&mut rng).split_with_ratio(0.9);

    // Create a logistic regression model
    let model = linfa_logistic::LogisticRegression::default().max_iterations(120);

    // Train the model
    let model = model.fit(&train).unwrap();

    dbg!(model.params());
    model
        .params()
        .axis_chunks_iter(Axis(0), BUCKETS_PER_OCTAVE)
        .for_each(|x| {
            println!("{x:?}",);
        });

    // Predict and map targets
    let pred = model.predict(&valid);

    // valid
    //     .as_targets()
    //     .iter()
    //     .zip(pred.iter())
    //     .for_each(|(t, p)| {
    //         println!("target: {}, prediction: {}", t, p);
    //     });

    // Create a confusion matrix
    let cm = pred.confusion_matrix(&valid).unwrap();

    // Print the confusion matrix, this will print a table with four entries. On the diagonal are
    // the number of true-positive and true-negative predictions, off the diagonal are
    // false-positive and false-negative
    println!("{:?}", cm);

    // Calculate the accuracy and Matthew Correlation Coefficient (cross-correlation between
    // predicted and targets)
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());
}

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

    // read all midi files in directory
    let soundfont_path = "MuseScore_General.sf2";
    let mut sf2 = File::open(soundfont_path).unwrap();
    let sound_font = Arc::new(SoundFont::new(&mut sf2).unwrap());

    let midi_paths = fs::read_dir("midi")?
        .map(|res| res.map(|e| e.path()))
        .collect::<Result<Vec<_>, io::Error>>()?;

    let data: Vec<f32> = midi_paths
        .par_iter()
        .flat_map(|p| {
            println!("processing {p:?}");
            let annotated_cqt = synthesize_midi_to_wav(
                p.to_str().unwrap(),
                sound_font.clone(),
                cqt_delay_in_samples,
                STEP_SIZE_IN_CHUNKS,
                &cqt,
            )
            .unwrap_or_else(|_| {
                println!(
                    "failed to synthesize midi file {:?} to data point",
                    p.to_str().unwrap()
                );
                vec![]
            });

            annotated_cqt
                .iter()
                .flat_map(|x| {
                    let (sample, target) = generate_data(&x);
                    assert!(sample.len() == OCTAVES * BUCKETS_PER_OCTAVE);
                    assert!(target.len() == 128);
                    sample
                        .iter()
                        .chain(target.iter())
                        .cloned()
                        .collect::<Vec<f32>>()
                })
                .collect::<Vec<f32>>()
        })
        .collect();

    println!("{}", data.len());

    let mut out_buf = vec![];
    {
        let mut writer = {
            npyz::WriteOptions::new()
                .dtype(npyz::DType::Plain("<f4".parse::<TypeStr>().unwrap()))
                .shape(&[data.len() as u64])
                .writer(&mut out_buf)
                .begin_nd()?
        };

        writer.extend(&data)?;
        writer.finish()?;
    }

    // write to file
    let mut file = File::create("data.npy")?;
    file.write_all(&out_buf)?;

    //fit(positive_x, negative_x);

    // // average all positive_x
    // let mut avg_positive_x = vec![0.0; positive_x[0].len()];
    // for v in &positive_x {
    //     for (i, x) in v.iter().enumerate() {
    //         avg_positive_x[i] += x;
    //     }
    // }
    // for x in &mut avg_positive_x {
    //     *x /= positive_x.len() as f32;
    // }

    // println!("avg_positive_x: {:?}", avg_positive_x);

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
    sound_font: Arc<SoundFont>,
    cqt_delay_in_samples: usize,
    step_size_in_chunks: usize,
    cqt: &pitchvis_analysis::cqt::Cqt,
) -> Result<Vec<(HashMap<i32, f32>, Vec<f32>)>> {
    // Load the MIDI file.
    let mut mid = File::open(midi_path).unwrap();
    let midi_file = Arc::new(MidiFile::new(&mut mid)?);

    // Create the MIDI file sequencer.
    let settings = SynthesizerSettings::new(SR as i32);
    let synthesizer = Synthesizer::new(&sound_font, &settings).unwrap();
    let mut sequencer = MidiFileSequencer::new(synthesizer);

    // Play the MIDI file.
    sequencer.play(&midi_file, false);

    let mut agc = dagc::MonoAgc::new(0.07, 0.001).expect("mono-agc creation failed");

    let sample_count = (settings.sample_rate as f64 * midi_file.get_length()) as usize;
    dbg!(settings.sample_rate, sample_count, midi_file.get_length());

    let mut ring_buffer = Vec::new();
    ring_buffer.resize(BUFSIZE, 0f32);

    // chunk size is cqt delay so that we get a reading of active keys every cqt delay
    let mut left: Vec<f32> = vec![0_f32; cqt_delay_in_samples];
    let mut right: Vec<f32> = vec![0_f32; cqt_delay_in_samples];

    let mut annotated_cqt = Vec::new();
    let mut written = 0;
    let mut prev_active_keys;
    let mut active_keys = HashMap::new();
    let mut chunk_count = 0;
    while written < sample_count {
        chunk_count += 1;

        // Render the waveform.
        sequencer.render(&mut left[..], &mut right[..]);
        written += left.len();

        // downmix to mono into left channel
        left.iter_mut()
            .zip(right.iter_mut())
            .for_each(|(l, r)| *l = (*l + *r) / 2.0);

        // skip agc on silence
        let sample_sq_sum = left.iter().map(|x| x.powi(2)).sum::<f32>();
        agc.freeze_gain(sample_sq_sum < 1e-6);

        //println!("gain: {}", agc.gain());

        // add to ring buffer
        ring_buffer.drain(..left.len());
        ring_buffer.extend_from_slice(left.as_slice());
        let begin = ring_buffer.len() - left.len();
        agc.process(&mut ring_buffer[begin..]);

        if (chunk_count % step_size_in_chunks) != 0 {
            continue;
        }

        // get active keys
        prev_active_keys = active_keys;
        active_keys = HashMap::new();
        sequencer
            .synthesizer
            .get_active_voices()
            .iter()
            // .map(|x| {
            //     println!("{:?}", x.key);
            //     x
            // })
            .for_each(|voice| {
                // add to active keys if gain is greater than existing entry
                let gain =
                    (voice.current_mix_gain_left + voice.current_mix_gain_right) / 2.0 * agc.gain();
                if let Some(existing_gain) = active_keys.get(&voice.key) {
                    if gain > *existing_gain {
                        active_keys.insert(voice.key, gain);
                    }
                } else {
                    active_keys.insert(voice.key, gain);
                }
            });

        // perform cqt analysis
        let x_cqt =
            cqt.calculate_cqt_instant_in_db(&ring_buffer[(ring_buffer.len() - cqt.n_fft)..]);

        // println!("Active keys: {:?}\ncqt:", prev_active_keys);
        // x_cqt.chunks(BUCKETS_PER_OCTAVE).for_each(|x| {
        //     println!("{x:?}",);
        // });
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
    annotated_cqt: &(HashMap<i32, f32>, Vec<f32>),
) -> (Vec<(Vec<f32>, f32)>, Vec<(Vec<f32>, f32)>) {
    let (active_keys, cqt_transform) = annotated_cqt;
    let mut positive_samples = Vec::new();
    let mut negative_samples = Vec::new();
    let shift_values = vec![
        -24, -19, -12, -9, -8, -7, -6, -5, -4, -3, 3, 4, 5, 6, 7, 8, 9, 12, 19, 24,
    ];

    for (&key, &attack) in active_keys.iter() {
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
        let (start, start_overshoot, end, end_overshoot) = dbg!(calc_boundaries(key_index));
        let mut sample = vec![0f32; 87 * BUCKETS_PER_SEMITONE];
        sample.splice(
            (start_overshoot)..(87 * BUCKETS_PER_SEMITONE - end_overshoot),
            cqt_transform[start..end].iter().cloned(),
        );
        // dbg!(sample[40] / sample[arg_max(&sample)]);
        positive_samples.push((sample, attack));

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
                .all(|(&other_key, _)| (other_key - shifted_key).abs() >= 2)
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
                negative_samples.push((sample, attack));
            }
        }
    }

    (positive_samples, negative_samples)
}

fn generate_data(
    annotated_cqt: &(HashMap<i32, f32>, Vec<f32>),
) -> ([f32; OCTAVES * BUCKETS_PER_OCTAVE], [f32; 128]) {
    let (active_keys, cqt_transform) = annotated_cqt;
    let mut samples = [0f32; OCTAVES * BUCKETS_PER_OCTAVE];
    let mut targets = [0f32; 128];

    samples = cqt_transform[..].try_into().unwrap();

    // TODO: generate some pitch wobble, e. g. out of tune instruments. ?Can we do this in the cqt domain ?or should we add random tunings to instruments during midi generation
    // TODO: maybe also use different sound fonts

    for (&key, &attack) in active_keys.iter() {
        targets[key as usize] = (attack > 0.5) as u8 as f32;
    }

    (samples, targets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_center_cqt_and_generate_positive_and_negative_data_points() {
        for note in 30..31 {
            let active_keys = HashMap::from([((33 + note) as i32, 1.0)]); // MIDI id for FREQ_A1 is 33
            let mut cqt_transform = vec![0f32; 86 * BUCKETS_PER_SEMITONE];
            for (i, x) in cqt_transform.iter_mut().enumerate() {
                *x = (1_000 - i.abs_diff(note * BUCKETS_PER_SEMITONE)) as f32;
            }
            // dbgs!(&cqt_transform);
            let annotated_cqt = (active_keys, cqt_transform);

            let (positive_samples, negative_samples) =
                center_cqt_and_generate_positive_and_negative_data_points(&annotated_cqt);

            // Check that the positive samples are correctly centered
            for sample in positive_samples.iter() {
                assert_eq!(sample.0.len(), 86 * BUCKETS_PER_SEMITONE);
                assert_eq!(sample.0[40 * BUCKETS_PER_SEMITONE], 1_000f32);
            }

            // dbg!(&positive_samples, &negative_samples);

            // Check that the negative samples are correctly centered and shifted
            let shifts: [i32; 20] = [
                -24, -19, -12, -9, -8, -7, -6, -5, -4, -3, 3, 4, 5, 6, 7, 8, 9, 12, 19, 24,
            ];
            for (i, &shift) in shifts.iter().enumerate() {
                assert_eq!(
                    negative_samples[i].0[(40 - shift) as usize * BUCKETS_PER_SEMITONE],
                    1_000f32,
                );
            }
        }
    }
}
