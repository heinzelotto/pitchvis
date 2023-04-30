use crate::audio_system::AudioBufferResource;
use bevy::prelude::*;
use pitchvis_analysis::cqt;

#[derive(Resource)]
pub struct CqtResource(pub cqt::Cqt);

#[derive(Resource)]
pub struct CqtResultResource {
    pub x_cqt: Vec<f32>,
    pub gain: f32,
}

impl CqtResultResource {
    pub fn new() -> Self {
        Self {
            x_cqt: vec![0.0; crate::OCTAVES * crate::BUCKETS_PER_OCTAVE],
            gain: 1.0,
        }
    }
}

pub fn update_cqt(
    mut cqt: ResMut<CqtResource>,
    rb: Res<AudioBufferResource>,
    mut cqt_result: ResMut<CqtResultResource>,
) {
    let (x, gain) = {
        let mut x = vec![0.0_f32; cqt.0.n_fft];
        let rb = rb.0.lock().unwrap();
        x.copy_from_slice(&rb.buf[(crate::BUFSIZE - cqt.0.n_fft)..]);
        (x, rb.gain)
    };

    cqt_result.x_cqt = cqt.0.calculate_cqt_instant_in_db(&x);
    cqt_result.gain = gain;
}
