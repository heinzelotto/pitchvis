use crate::audio_system::AudioBufferResource;
use bevy::prelude::*;
use pitchvis_analysis::Vqt;

#[derive(Resource)]
pub struct VqtResource(pub Vqt);

#[derive(Resource)]
pub struct VqtResultResource {
    pub x_vqt: Vec<f32>,
    pub gain: f32,
}

impl VqtResultResource {
    pub fn new(octaves: usize, buckets_per_octave: usize) -> Self {
        Self {
            x_vqt: vec![0.0; octaves * buckets_per_octave],
            gain: 1.0,
        }
    }
}

pub fn update_vqt_to_system(
    bufsize: usize,
) -> impl FnMut(ResMut<VqtResource>, Res<AudioBufferResource>, ResMut<VqtResultResource>) + Copy {
    move |vqt: ResMut<VqtResource>,
          rb: Res<AudioBufferResource>,
          vqt_result: ResMut<VqtResultResource>| {
        update_vqt(bufsize, vqt, rb, vqt_result);
    }
}

pub fn update_vqt(
    bufsize: usize,
    vqt: ResMut<VqtResource>,
    rb: Res<AudioBufferResource>,
    mut vqt_result: ResMut<VqtResultResource>,
) {
    let (x, gain) = {
        let mut x = vec![0.0_f32; vqt.0.n_fft];
        let rb = rb.0.lock().unwrap();
        x.copy_from_slice(&rb.buf[(bufsize - vqt.0.n_fft)..]);
        (x, rb.gain)
    };

    vqt_result.x_vqt = vqt.0.calculate_vqt_instant_in_db(&x);
    vqt_result.gain = gain;
}
