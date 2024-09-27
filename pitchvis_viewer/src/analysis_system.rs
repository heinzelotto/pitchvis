use std::time::Duration;

use crate::vqt_system::VqtResultResource;
use bevy::prelude::*;
use pitchvis_analysis::AnalysisState;

#[derive(Resource)]
pub struct AnalysisStateResource(pub AnalysisState);

pub fn update_analysis_state_to_system(
    octaves: usize,
    buckets_per_octave: usize,
) -> impl FnMut(ResMut<AnalysisStateResource>, Res<VqtResultResource>, Res<Time>) + Copy {
    move |mut analysis_state: ResMut<AnalysisStateResource>,
          vqt_result: Res<VqtResultResource>,
          time: Res<Time>| {
        analysis_state.0.preprocess(
            &vqt_result.x_vqt,
            octaves,
            buckets_per_octave,
            Duration::from_secs_f64(time.delta_seconds_f64()),
        );
    }
}
