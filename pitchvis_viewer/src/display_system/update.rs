use super::{
    material::NoisyColorMaterial, util::bin_to_spiral, BassCylinder, CylinderEntityListResource,
    DisplayMode, LineList, PitchBall, PitchNameText, Spectrum, SpiderNetSegment, VisualsMode,
    CLEAR_COLOR_GALAXY, CLEAR_COLOR_NEUTRAL,
};
use bevy::{asset::meta::Settings, core_pipeline::bloom::Bloom, prelude::*};
use bevy_persistent::Persistent;
use itertools::Itertools;
use std::collections::HashMap;

use crate::{
    analysis_system::AnalysisStateResource, app::SettingsState, display_system::CLEAR_COLOR_EINK,
    vqt_system::VqtResultResource,
};
use pitchvis_analysis::{
    analysis::{AnalysisState, ContinuousPeak},
    util::*,
    vqt::VqtRange,
};
use pitchvis_colors::{calculate_color, COLORS, EASING_POW, GRAY_LEVEL};

const SPIRAL_SEGMENTS_PER_SEMITONE: u16 = 6;
const PITCH_BALL_SCALE_FACTOR: f32 = 1.0 / 305.0; // TODO: get rid of this, we're engineers, not magicians

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn update_display(
    range: &VqtRange,
    mut set: ParamSet<(
        Query<(
            &PitchBall,
            &mut Visibility,
            &mut Transform,
            &mut MeshMaterial2d<NoisyColorMaterial>,
        )>,
        Query<(
            &BassCylinder,
            &mut Visibility,
            &mut MeshMaterial2d<ColorMaterial>,
        )>,
        Query<(&PitchNameText, &mut Visibility)>,
        Query<(&mut Visibility, &Mesh2d, &mut Transform), With<Spectrum>>,
        Query<&mut Visibility, With<SpiderNetSegment>>,
    )>,
    mut color_materials: ResMut<Assets<ColorMaterial>>,
    mut noisy_color_materials: ResMut<Assets<NoisyColorMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    analysis_state: Res<AnalysisStateResource>,
    vqt_result: Res<VqtResultResource>,
    cylinder_entities: Res<CylinderEntityListResource>,
    settings_state: Res<Persistent<SettingsState>>,
    run_time: Res<Time>,
    mut camera: Query<(
        &mut Camera,
        Option<&mut Bloom>,
        Ref<Projection>, // Ref because we want to check `is_changed` later
    )>,
) -> Result<()> {
    fade_pitch_balls(set.p0(), &mut noisy_color_materials, &run_time, range);

    // Exit early if there are no detected notes. Note that the below sub-systems will not run
    // until a few frames into the app.
    let analysis_state = &analysis_state.0;
    if analysis_state.peaks_continuous.is_empty() {
        return Ok(());
    }

    update_pitch_balls(
        set.p0(),
        &mut noisy_color_materials,
        &run_time,
        &settings_state,
        analysis_state,
        range,
    );

    update_bloom(&mut camera, analysis_state, &settings_state)?;

    update_bass_spiral(
        set.p1(),
        &mut color_materials,
        &cylinder_entities,
        &settings_state,
        &analysis_state.peaks_continuous,
        range.buckets_per_octave,
    );

    update_spectrum(
        set.p3(),
        &mut camera,
        &mut meshes,
        &settings_state,
        &vqt_result,
        range,
    )?;

    show_hide_pitch_names(set.p2(), &settings_state);

    show_hide_spider_net(set.p4(), &settings_state);

    toggle_background(&mut camera, &settings_state)?;

    Ok(())
}

fn fade_pitch_balls(
    mut pitch_balls: Query<(
        &PitchBall,
        &mut Visibility,
        &mut Transform,
        &mut MeshMaterial2d<NoisyColorMaterial>,
    )>,
    noisy_color_materials: &mut ResMut<Assets<NoisyColorMaterial>>,
    run_time: &Res<Time>,
    range: &VqtRange,
) {
    let timestep = run_time.delta();
    for (pitch_ball, mut visibility, mut transform, color) in &mut pitch_balls {
        if *visibility == Visibility::Visible {
            let idx = pitch_ball.0;
            let dropoff_factor_per_30fps_frame =
                0.90 - 0.15 * (idx as f32 / range.n_buckets() as f32);
            let dropoff_factor = dropoff_factor_per_30fps_frame.powf(30.0 * timestep.as_secs_f32());

            let mut size = transform.scale / PITCH_BALL_SCALE_FACTOR;
            size *= dropoff_factor;
            transform.scale = size * PITCH_BALL_SCALE_FACTOR;

            // also make them slightly more transparent when they are smaller
            let color_mat = noisy_color_materials
                .get_mut(&*color)
                .expect("ball color material");
            color_mat.color = color_mat
                .color
                .with_alpha(color_mat.color.alpha() * dropoff_factor);

            // also shift shrinking circles slightly to the background so that they are not cluttering newly appearing larger circles
            transform.translation.z -= 0.001 * 30.0 * timestep.as_secs_f32();

            if size.x * PITCH_BALL_SCALE_FACTOR < 0.003 {
                *visibility = Visibility::Hidden;
            }

            // FIXME: test how it looks when we only show the balls that are currently active in the vqt analysis
            // *visibility = Visibility::Hidden;
        }
    }
}

fn update_pitch_balls(
    mut pitch_balls: Query<(
        &PitchBall,
        &mut Visibility,
        &mut Transform,
        &mut MeshMaterial2d<NoisyColorMaterial>,
    )>,
    noisy_color_materials: &mut ResMut<Assets<NoisyColorMaterial>>,
    run_time: &Res<Time>,
    settings_state: &Res<Persistent<SettingsState>>,
    analysis_state: &AnalysisState,
    range: &VqtRange,
) {
    let k_max = arg_max(
        &analysis_state
            .peaks_continuous
            .iter()
            .map(|p| p.size)
            .collect::<Vec<f32>>(),
    );
    let max_size = analysis_state.peaks_continuous[k_max].size;

    let peaks_rounded = analysis_state
        .peaks_continuous
        .iter()
        .map(|p| (p.center.trunc() as usize, *p))
        .collect::<HashMap<usize, ContinuousPeak>>();

    for (pitch_ball, mut visibility, mut transform, color) in &mut pitch_balls {
        let idx = pitch_ball.0;
        if peaks_rounded.contains_key(&idx) {
            let ContinuousPeak { center, size } = peaks_rounded[&idx];

            let (r, g, b) = calculate_color(
                range.buckets_per_octave,
                (center + (range.buckets_per_octave - 3 * (range.buckets_per_octave / 12)) as f32)
                    % range.buckets_per_octave as f32,
                COLORS,
                GRAY_LEVEL,
                EASING_POW,
            );

            let color_coefficient = 1.0 - (1.0 - size / max_size).powf(2.0);

            let (x, y, _) = bin_to_spiral(range.buckets_per_octave, center);
            // make sure larger circles are drawn on top by adding a small offset proportional to the size
            let z_ordering_offset = (size / max_size - 1.01) * 12.5;
            transform.translation = Vec3::new(x, y, z_ordering_offset);

            let color_mat = noisy_color_materials
                .get_mut(&*color)
                .expect("ball color material");
            color_mat.params.time = run_time.elapsed_secs();
            // color_mat.color = Color::srgb(
            //     r * color_coefficient,
            //     g * color_coefficient,
            //     b * color_coefficient,
            // );
            color_mat.color = Color::srgba(r, g, b, color_coefficient).into();

            #[cfg(feature = "ml")]
            if let Some(midi_pitch) = vqt_bin_to_midi_pitch(buckets_per_octave, idx) {
                let inferred_midi_pitch_strength = analysis_state.ml_midi_base_pitches[midi_pitch];
                if inferred_midi_pitch_strength > 0.35 {
                    color_mat.color = Color::srgba(r, g, b, 1.0);
                } else {
                    color_mat.color = Color::srgba(r, g, b, color_coefficient * 0.1);
                }
            }

            // set calmness visual effect.
            // FIXME: Usually we see values of 0.75 for very calm notes... Fix this to be more intuitive.
            if settings_state.display_mode == DisplayMode::Normal
                || settings_state.display_mode == DisplayMode::Debugging
            {
                color_mat.params.calmness =
                    (analysis_state.calmness[idx].get() - 0.27).clamp(0.0, 1.0);
            } else {
                color_mat.params.calmness = 0.0;
            }

            // scale calm ones even more
            let calmness_scale = 1.0 + 0.2 * color_mat.params.calmness;

            // TODO: scale up new notes to make them more prominent

            // scale down balls in Performance Mode
            let ball_scale_factor = if settings_state.visuals_mode == VisualsMode::Performance {
                0.5
            } else {
                1.0
            };

            // scale and threshold to vanish
            transform.scale =
                Vec3::splat(size * ball_scale_factor * PITCH_BALL_SCALE_FACTOR * calmness_scale);
            if transform.scale.x >= 0.002 {
                *visibility = Visibility::Visible;
            }
        }
    }
    // TODO: ?faster lookup through indexes

    // let mut balls_to_hide = vec![false; pitch_balls.iter().len()];
    // for (_, ContinuousPeak { center, .. }) in peaks_rounded.iter() {
    //     // hide all balls that are close to the currently active peaks
    //     let radius = (range.buckets_per_octave / 12) as f32 * 0.66;
    //     #[allow(clippy::needless_range_loop)]
    //     for i in ((center - radius).round().max(0.0) as usize)
    //         ..=((center + radius)
    //             .round()
    //             .min((pitch_balls.iter().len() - 1) as f32) as usize)
    //     {
    //         balls_to_hide[i] = true;
    //     }
    // }
    // for (idx, _) in peaks_rounded.iter() {
    //     // show peaks itself even if they are close to other peaks
    //     balls_to_hide[*idx] = false;
    // }
    // let mut hidden_cnt = 0;
    // for (pitch_ball, mut visibility, _, _) in &mut pitch_balls {
    //     if *visibility == Visibility::Visible && balls_to_hide[pitch_ball.0] {
    //         *visibility = Visibility::Hidden;
    //         hidden_cnt += 1;
    //     }
    // }
    // log::trace!("Hid {} balls", hidden_cnt);
}

fn update_bloom(
    camera: &mut Query<(&mut Camera, Option<&mut Bloom>, Ref<Projection>)>,
    analysis_state: &AnalysisState,
    settings_state: &SettingsState,
) -> Result<()> {
    if let (_, Some(mut bloom_settings), _) = camera.single_mut()? {
        if settings_state.visuals_mode == VisualsMode::Performance {
            bloom_settings.intensity = 0.0;
            return Ok(());
        }
        bloom_settings.intensity =
            (analysis_state.smoothed_scene_calmness.get() * 1.3).clamp(0.0, 1.0);
    }

    Ok(())
}

fn update_bass_spiral(
    mut bass_cylinders: Query<(
        &BassCylinder,
        &mut Visibility,
        &mut MeshMaterial2d<ColorMaterial>,
    )>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    cylinder_entities: &Res<CylinderEntityListResource>,
    settings_state: &Res<Persistent<SettingsState>>,
    peaks_continuous: &[ContinuousPeak],
    buckets_per_octave: u16,
) {
    //let mut color_map: Vec<i32> = vec![-1; self.buckets_per_octave * self.octaves];
    // for (prev, cur) in peaks.iter().tuple_windows() {
    //     color_map[*prev..*cur].fill(*prev as i32);
    // }
    for (_, mut visibility, _) in &mut bass_cylinders {
        if *visibility == Visibility::Visible {
            *visibility = Visibility::Hidden;
        }
    }
    if settings_state.visuals_mode == VisualsMode::Galaxy {
        return;
    }
    // if gain > 1000.0 {
    //     return;
    // }
    if let Some(ContinuousPeak { center, size }) = peaks_continuous.first() {
        let center = center / buckets_per_octave as f32 * 12.0;
        if center.round() as usize * SPIRAL_SEGMENTS_PER_SEMITONE as usize
            >= cylinder_entities.0.len()
        {
            // lowest peak is outside the maximum bass spiral range
            return;
        }

        // color up to lowest note
        for idx in 0..((center.round() * SPIRAL_SEGMENTS_PER_SEMITONE as f32) as usize) {
            let (_, ref mut visibility, color) = bass_cylinders
                .get_mut(cylinder_entities.0[idx])
                .expect("cylinder entity");
            **visibility = Visibility::Visible;

            let color_map_ref = center.round() * buckets_per_octave as f32 / 12.0;
            let (r, g, b) = calculate_color(
                buckets_per_octave,
                (color_map_ref + (buckets_per_octave - 3 * (buckets_per_octave / 12)) as f32)
                    % buckets_per_octave as f32,
                COLORS,
                GRAY_LEVEL,
                EASING_POW,
            );

            let k_max = arg_max(
                &peaks_continuous
                    .iter()
                    .map(|p| p.size)
                    .collect::<Vec<f32>>(),
            );
            let max_size = peaks_continuous[k_max].size;

            let color_coefficient = 1.0 - (1.0 - size / max_size).powf(2.0);

            materials
                .get_mut(&*color)
                .expect("cylinder color material")
                .color = Color::srgba(r, g, b, color_coefficient);

            // let radius = 0.08;
            // c.set_local_scale(radius, *height, radius);
        }
    }
}

fn update_spectrum(
    mut spectrum: Query<(&mut Visibility, &Mesh2d, &mut Transform), With<Spectrum>>,
    camera: &mut Query<(&mut Camera, Option<&mut Bloom>, Ref<Projection>)>,
    meshes: &mut ResMut<Assets<Mesh>>,
    settings_state: &Res<Persistent<SettingsState>>,
    vqt_result: &Res<VqtResultResource>,
    range: &VqtRange,
) -> Result<()> {
    let (mut visibility, mesh_handle, mut transform) = spectrum.single_mut()?;
    {
        // set visibility
        if settings_state.display_mode == DisplayMode::Debugging {
            *visibility = Visibility::Visible;
        } else {
            *visibility = Visibility::Hidden;
            return Ok(());
        }

        // move to right
        let (_, _, projection) = camera.single_mut()?;
        {
            if let Projection::Orthographic(proj) = &*projection {
                let Rect { max, .. } = proj.area;
                *transform = Transform::from_xyz(
                    max.x - range.n_buckets() as f32 * 0.022 - 0.2,
                    max.y - 4.2,
                    -13.0,
                );
            } else {
                panic!("Not an ortographic projection.");
            }
        }

        let k_max = arg_max(
            &vqt_result
                .x_vqt
                .iter()
                .map(|p| p)
                .cloned()
                .collect::<Vec<f32>>(),
        );
        let max_size = vqt_result.x_vqt[k_max];

        let mut spectrum_mesh: Mesh = LineList {
            lines: vqt_result
                .x_vqt
                .iter()
                .enumerate()
                .map(|(i, amp)| Vec3::new(i as f32 * 0.022, *amp / 10.0, 0.0))
                .tuple_windows()
                .collect::<Vec<(Vec3, Vec3)>>(),
            thickness: 0.02,
        }
        .into();
        let colors = (0..(vqt_result.x_vqt.len() - 1))
            .map(|i| {
                let (r, g, b) = calculate_color(
                    range.buckets_per_octave,
                    ((i as f32)
                        + 0.5
                        + (range.buckets_per_octave - 3 * (range.buckets_per_octave / 12)) as f32)
                        % range.buckets_per_octave as f32,
                    COLORS,
                    GRAY_LEVEL,
                    10.0,
                );
                let color_coefficient =
                    1.0 - (0.5 - vqt_result.x_vqt[i] / max_size / 2.0).powf(0.5);
                [
                    [r, g, b, color_coefficient],
                    [r, g, b, color_coefficient],
                    [r, g, b, color_coefficient],
                    [r, g, b, color_coefficient],
                ]
            })
            .flatten()
            .collect::<Vec<[f32; 4]>>();
        spectrum_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);

        let mesh = meshes
            .get_mut(&mesh_handle.0)
            .expect("spectrum line strip mesh");
        *mesh = spectrum_mesh;
    }

    Ok(())
}

fn show_hide_pitch_names(
    mut pitch_name_text: Query<(&PitchNameText, &mut Visibility)>,
    settings_state: &Res<Persistent<SettingsState>>,
) {
    for (_, mut visibility) in &mut pitch_name_text {
        match settings_state.visuals_mode {
            VisualsMode::Full | VisualsMode::Performance => {
                *visibility = Visibility::Visible;
            }
            _ => {
                *visibility = Visibility::Hidden;
            }
        }
    }
}

fn show_hide_spider_net(
    mut spider_net_segments: Query<&mut Visibility, With<SpiderNetSegment>>,
    settings_state: &Res<Persistent<SettingsState>>, // TODO: ?Changed<SettingsState> possible
) {
    let target_visibility = match settings_state.visuals_mode {
        VisualsMode::Full | VisualsMode::Zen | VisualsMode::Performance => Visibility::Visible,
        _ => Visibility::Hidden,
    };

    let vis = spider_net_segments.iter_mut().next().unwrap();
    if *vis != target_visibility {
        for mut visibility in &mut spider_net_segments {
            *visibility = target_visibility;
        }
    }
}

fn toggle_background(
    camera: &mut Query<(&mut Camera, Option<&mut Bloom>, Ref<Projection>)>,
    settings_state: &Res<Persistent<SettingsState>>,
) -> Result<()> {
    let (mut camera, _, _) = camera.single_mut()?;
    camera.clear_color = match settings_state.visuals_mode {
        VisualsMode::Zen | VisualsMode::Full | VisualsMode::Performance => CLEAR_COLOR_NEUTRAL,
        VisualsMode::Galaxy => CLEAR_COLOR_GALAXY,
    };

    Ok(())
}

// #[derive(PartialEq)]
// pub enum PauseState {
//     Running,
//     Paused,
// }
