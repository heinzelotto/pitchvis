use std::time::Duration;

use crate::vqt_system::VqtResultResource;
use bevy::prelude::*;
use pitchvis_analysis::{vqt::VqtRange, AnalysisState};

#[derive(Resource)]
pub struct AnalysisStateResource(pub AnalysisState);

pub fn update_analysis_state_to_system(
    range: &VqtRange,
) -> impl FnMut(ResMut<AnalysisStateResource>, Res<VqtResultResource>, Res<Time>) + Clone {
    let range = range.clone();
    move |mut analysis_state: ResMut<AnalysisStateResource>,
          vqt_result: Res<VqtResultResource>,
          time: Res<Time>| {
        analysis_state.0.preprocess(
            &vqt_result.x_vqt,
            &range,
            Duration::from_secs_f64(time.delta_seconds_f64()),
        );
    }
}
