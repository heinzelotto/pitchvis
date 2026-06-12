use crate::audio_system::AudioBufferResource;
use bevy::prelude::*;
use pitchvis_analysis::{vqt::Vqt, vqt::VqtRange};

#[derive(Resource)]
pub struct VqtResource(pub Vqt);

#[derive(Resource)]
pub struct VqtResultResource {
    pub x_vqt: Vec<f32>,
    pub gain: f32,
}

impl VqtResultResource {
    pub fn new(range: &VqtRange) -> Self {
        Self {
            x_vqt: vec![0.0; range.n_buckets()],
            gain: 1.0,
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn update_vqt_to_system(
    bufsize: usize,
) -> impl FnMut(
    ResMut<VqtResource>,
    Res<AudioBufferResource>,
    ResMut<VqtResultResource>,
    Local<Vec<f32>>,
) + Copy {
    move |vqt: ResMut<VqtResource>,
          rb: Res<AudioBufferResource>,
          vqt_result: ResMut<VqtResultResource>,
          x_scratch: Local<Vec<f32>>| {
        update_vqt(bufsize, vqt, rb, vqt_result, x_scratch);
    }
}

pub fn update_vqt(
    bufsize: usize,
    mut vqt: ResMut<VqtResource>,
    rb: Res<AudioBufferResource>,
    mut vqt_result: ResMut<VqtResultResource>,
    mut x_scratch: Local<Vec<f32>>,
) {
    let n_fft = vqt.0.params().n_fft;

    // Safety check: n_fft must not exceed bufsize
    if n_fft > bufsize {
        error!(
            "VQT n_fft ({}) exceeds buffer size ({}). Skipping VQT calculation.",
            n_fft, bufsize
        );
        // Return early with empty result
        return;
    }

    x_scratch.resize(n_fft, 0.0);
    let gain = {
        let rb = rb.0.lock().unwrap();
        x_scratch.copy_from_slice(&rb.buf[(bufsize - n_fft)..]);
        rb.gain
    };

    vqt_result.x_vqt = vqt.0.calculate_vqt_instant_in_db(&x_scratch);
    vqt_result.gain = gain;
}
