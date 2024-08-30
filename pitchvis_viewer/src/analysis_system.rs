use crate::vqt_system::VqtResultResource;
use bevy::prelude::*;
use pitchvis_analysis::AnalysisState;

#[derive(Resource)]
pub struct AnalysisStateResource(pub AnalysisState);

pub fn update_analysis_state_to_system(
    octaves: usize,
    buckets_per_octave: usize,
) -> impl FnMut(ResMut<AnalysisStateResource>, Res<VqtResultResource>) + Copy {
    move |mut analysis_state: ResMut<AnalysisStateResource>, vqt_result: Res<VqtResultResource>| {
        analysis_state
            .0
            .preprocess(&vqt_result.x_vqt, octaves, buckets_per_octave);
    }
}
