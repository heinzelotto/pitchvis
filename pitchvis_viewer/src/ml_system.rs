use tch::{CModule, Device, Tensor};

const OCTAVES: usize = 7;
const BUCKETS_PER_OCTAVE: usize = 36;
const T: usize = 3;

const NUM_MIDI_PITCHES: usize = 128;

use crate::analysis_system::AnalysisStateResource;
use bevy::prelude::*;

pub struct MlModel {
    model: CModule,
}

impl MlModel {
    pub fn new(path: &str) -> Self {
        let model = tch::CModule::load(path).expect("loading model from disk"); // Load the model

        Self { model }
    }
}

pub fn infer(model: &CModule, sl: &[f32]) -> tch::Result<Vec<f32>> {
    // Now you can use this model for inference
    // TODO: use Kind::Half instead of Kind::Float
    let input = Tensor::from(sl)
        .view([1, 1, (T * OCTAVES * BUCKETS_PER_OCTAVE) as i64])
        .to_device(Device::Cuda(0));
    //let input = Tensor::rand(&[1, 1, (T * OCTAVES * BUCKETS_PER_OCTAVE) as i64], (Kind::Float, Device::Cuda(0)));
    let output = model.forward_ts(&[input])?; // Forward pass

    let output: Vec<f32> = Vec::try_from(output.view([NUM_MIDI_PITCHES as i64]))?;
    //println!("{:?}", output);
    //let output: Vec<f32> = output.copy_data(Kind::Float, NUM_MIDI_PITCHES)?;

    Ok(output)
}

#[derive(Resource)]
pub struct MlModelResource(pub MlModel);

pub fn update_ml_to_system(
) -> impl FnMut(Res<MlModelResource>, ResMut<AnalysisStateResource>) + Copy {
    move |ml_model: Res<MlModelResource>, analysis_state: ResMut<AnalysisStateResource>| {
        update_ml(ml_model, analysis_state);
    }
}

pub fn update_ml(
    ml_model: Res<MlModelResource>,
    mut analysis_state: ResMut<AnalysisStateResource>,
) {
    let mut sl = vec![0.0_f32; OCTAVES * BUCKETS_PER_OCTAVE * T];
    for i in 0..T {
        sl[(i * OCTAVES * BUCKETS_PER_OCTAVE)..((i + 1) * OCTAVES * BUCKETS_PER_OCTAVE)]
            .copy_from_slice(&analysis_state.0.history[analysis_state.0.history.len() - T + i]);
    }
    let output = infer(&ml_model.0.model, &sl).expect("run inference");

    let midi_pitches = output
        .iter()
        .enumerate()
        .filter_map(|(midi, p)| if *p > 0.5 { Some(midi) } else { None })
        .collect::<Vec<_>>();
    println!("detected midi pitches: {:?}", midi_pitches);

    analysis_state.0.ml_midi_base_pitches = output.try_into().unwrap();
}
