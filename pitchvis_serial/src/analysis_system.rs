use crate::cqt_system::CqtResultResource;
use bevy::prelude::*;
use pitchvis_analysis::analysis::AnalysisState;

#[derive(Resource)]
pub struct AnalysisStateResource(pub AnalysisState);

pub fn update_analysis_state_to_system(
    octaves: usize,
    buckets_per_octave: usize,
) -> impl FnMut(ResMut<AnalysisStateResource>, Res<CqtResultResource>) + Copy {
    move |mut analysis_state: ResMut<AnalysisStateResource>, cqt_result: Res<CqtResultResource>| {
        analysis_state
            .0
            .preprocess(&cqt_result.x_cqt, octaves, buckets_per_octave);
    }
}
