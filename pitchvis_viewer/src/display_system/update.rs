use super::{
    material::{NoisyColorMaterial, SpectrogramMaterial},
    util::bin_to_spiral,
    BassCylinder, CalmnessHistogram, ChromaBox, CylinderEntityListResource, DisplayMode, LineList,
    PitchBall, PitchNameText, SceneCalmnessGraph, SceneCalmnessHistory, SpectrogramDisplay,
    SpectrogramResource, Spectrum, SpiderNetSegment, VisualsMode, CLEAR_COLOR_GALAXY,
    CLEAR_COLOR_NEUTRAL,
};
use bevy::{post_process::bloom::Bloom, prelude::*};
use bevy_persistent::Persistent;
use itertools::Itertools;
use std::collections::HashMap;

use crate::{analysis_system::AnalysisStateResource, app::SettingsState};
use pitchvis_analysis::{
    analysis::{AnalysisState, ContinuousPeak},
    util::*,
    vqt::VqtRange,
};
use pitchvis_colors::{calculate_color, COLORS, EASING_POW, GRAY_LEVEL};

const SPIRAL_SEGMENTS_PER_SEMITONE: u16 = 6;
const PITCH_BALL_SCALE_FACTOR: f32 = 1.0 / 305.0; // TODO: get rid of this, we're engineers, not magicians

/// Helper function to convert calmness value to color
/// Returns cyan for calm (>0.7), yellow for medium (>0.3), red for energetic (<=0.3)
fn calmness_to_color(calmness: f32) -> Color {
    if calmness > 0.7 {
        Color::srgb(0.5, 0.8, 1.0) // Cyan for calm
    } else if calmness > 0.3 {
        Color::srgb(1.0, 1.0, 0.5) // Yellow for medium
    } else {
        Color::srgb(1.0, 0.5, 0.5) // Red for energetic
    }
}

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
    cylinder_entities: Res<CylinderEntityListResource>,
    settings_state: Res<Persistent<SettingsState>>,
    run_time: Res<Time>,
    mut camera: Query<(
        &mut Camera,
        Option<&mut Bloom>,
        Ref<Projection>, // Ref because we want to check `is_changed` later
    )>,
    histogram: Query<
        (&mut Visibility, &Mesh2d, &mut Transform),
        (
            With<CalmnessHistogram>,
            Without<PitchBall>,
            Without<BassCylinder>,
            Without<PitchNameText>,
            Without<Spectrum>,
            Without<SpiderNetSegment>,
        ),
    >,
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
        &camera,
        &mut meshes,
        &settings_state,
        analysis_state,
        range,
    )?;

    update_calmness_histogram(
        histogram,
        &camera,
        &mut meshes,
        &analysis_state,
        &settings_state,
        range,
    )?;

    show_hide_pitch_names(set.p2(), &settings_state);

    show_hide_spider_net(set.p4(), &settings_state);

    toggle_background(&mut camera, &settings_state, analysis_state)?;

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
    const VISIBILITY_CUTOFF: f32 = 0.019;

    let timestep = run_time.delta();
    for (pitch_ball, mut visibility, mut transform, color) in &mut pitch_balls {
        let mut size = transform.scale / PITCH_BALL_SCALE_FACTOR;

        if size.x * PITCH_BALL_SCALE_FACTOR >= VISIBILITY_CUTOFF {
            *visibility = Visibility::Visible;

            let idx = pitch_ball.0;
            let dropoff_factor_per_30fps_frame =
                0.85 - 0.15 * (idx as f32 / range.n_buckets() as f32);
            let dropoff_factor = dropoff_factor_per_30fps_frame.powf(30.0 * timestep.as_secs_f32());

            size *= dropoff_factor;
            transform.scale = size * PITCH_BALL_SCALE_FACTOR;

            // also make them slightly more transparent when they are smaller
            if let Some(mut color_mat) = noisy_color_materials.get_mut(&*color) {
                color_mat.color = color_mat
                    .color
                    .with_alpha((color_mat.color.alpha() * dropoff_factor).max(0.7));
            }

            // also shift shrinking circles slightly to the background so that they are not cluttering newly appearing larger circles
            transform.translation.z -= 0.001 * 30.0 * timestep.as_secs_f32();
        }

        if size.x * PITCH_BALL_SCALE_FACTOR < VISIBILITY_CUTOFF {
            *visibility = Visibility::Hidden;
        }
    }

    // for (_, mut visibility, _, _) in &mut pitch_balls {
    //     // FIXME: test how it looks when we only show the balls that are currently active in the vqt analysis
    //     *visibility = Visibility::Hidden;
    // }
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

            // FIXME: colors of pitch balls are not the same as colors of pitch names.
            let color_coefficient = 1.0 - (1.0 - size / max_size).powf(2.0);

            let (x, y, _) = bin_to_spiral(range.buckets_per_octave, center);
            // make sure larger circles are drawn on top by adding a small offset proportional to the size
            let z_ordering_offset = (size / max_size - 1.01) * 12.5;
            transform.translation = Vec3::new(x, y, z_ordering_offset);

            let Some(mut color_mat) = noisy_color_materials.get_mut(&*color) else {
                continue;
            };
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
                color_mat.params.pitch_accuracy = analysis_state.pitch_accuracy[idx];
                color_mat.params.pitch_deviation = analysis_state.pitch_deviation[idx];
            } else {
                color_mat.params.calmness = 0.0;
                color_mat.params.pitch_accuracy = 0.0;
                color_mat.params.pitch_deviation = 0.0;
            }

            // scale calm ones even more
            let calmness_scale = 1.0 + 0.2 * color_mat.params.calmness;

            // Check that there is no clipping with the color values and clip if necessary
            if (color_mat.color.red < 0.0 || color_mat.color.red > 1.0)
                || (color_mat.color.green < 0.0 || color_mat.color.green > 1.0)
                || (color_mat.color.blue < 0.0 || color_mat.color.blue > 1.0)
            {
                warn!("color out of bounds");
                color_mat.color.red = color_mat.color.red.clamp(0.0, 1.0);
                color_mat.color.green = color_mat.color.green.clamp(0.0, 1.0);
                color_mat.color.blue = color_mat.color.blue.clamp(0.0, 1.0);
            }

            // FIXME: colors are glitching out, I don't know where it happens yet...

            // TODO: scale up new notes to make them more prominent

            // scale down balls in Performance Mode
            let ball_scale_factor = if settings_state.visuals_mode == VisualsMode::Performance {
                0.7
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

    let mut balls_to_hide = vec![false; pitch_balls.iter().len()];
    for (_, ContinuousPeak { center, .. }) in peaks_rounded.iter() {
        // hide all balls that are close to the currently active peaks
        let radius = (range.buckets_per_octave / 12) as f32 * 0.23;
        #[allow(clippy::needless_range_loop)]
        for i in ((center - radius).round().max(0.0) as usize)
            ..=((center + radius)
                .round()
                .min((pitch_balls.iter().len() - 1) as f32) as usize)
        {
            balls_to_hide[i] = true;
        }
    }
    for (idx, _) in peaks_rounded.iter() {
        // show peaks itself even if they are close to other peaks
        balls_to_hide[*idx] = false;
    }
    let mut hidden_cnt = 0;
    for (pitch_ball, mut visibility, _, _) in &mut pitch_balls {
        if *visibility == Visibility::Visible && balls_to_hide[pitch_ball.0] {
            *visibility = Visibility::Hidden;
            hidden_cnt += 1;
        }
    }
    if hidden_cnt > 0 {
        log::trace!("Hid {} balls", hidden_cnt);
    }
}

fn update_bloom(
    camera: &mut Query<(&mut Camera, Option<&mut Bloom>, Ref<Projection>)>,
    analysis_state: &AnalysisState,
    settings_state: &SettingsState,
) -> Result<()> {
    if let (_, Some(mut bloom_settings), _) = camera.single_mut()? {
        if !settings_state.enable_bloom || settings_state.visuals_mode == VisualsMode::Performance {
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
            let Ok((_, ref mut visibility, color)) =
                bass_cylinders.get_mut(cylinder_entities.0[idx])
            else {
                continue;
            };
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

            if let Some(mut material) = materials.get_mut(&*color) {
                material.color = Color::srgba(r, g, b, color_coefficient);
            }

            // let radius = 0.08;
            // c.set_local_scale(radius, *height, radius);
        }
    }
}

/// Helper struct for building circle geometry
struct CircleGeometry {
    vertices: Vec<(Vec3, [f32; 3], [f32; 2])>,
    indices: Vec<u32>,
    colors: Vec<[f32; 4]>,
}

impl CircleGeometry {
    fn new(center: Vec3, radius: f32, color: [f32; 4], segments: u32) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut colors = Vec::new();

        // Add center vertex
        vertices.push((center, [0.0, 0.0, 1.0], [0.5, 0.5]));
        colors.push(color);

        // Add circle perimeter vertices and create triangles
        for i in 0..segments {
            let angle = (i as f32 / segments as f32) * std::f32::consts::TAU;
            let x = center.x + radius * angle.cos();
            let y = center.y + radius * angle.sin();

            vertices.push((
                Vec3::new(x, y, center.z),
                [0.0, 0.0, 1.0],
                [0.5 + 0.5 * angle.cos(), 0.5 + 0.5 * angle.sin()],
            ));
            colors.push(color);

            // Create triangle (center, current, next)
            let current = 1 + i;
            let next = 1 + ((i + 1) % segments);
            indices.push(0);
            indices.push(current);
            indices.push(next);
        }

        CircleGeometry {
            vertices,
            indices,
            colors,
        }
    }
}

fn update_spectrum(
    mut spectrum: Query<(&mut Visibility, &Mesh2d, &mut Transform), With<Spectrum>>,
    camera: &Query<(&mut Camera, Option<&mut Bloom>, Ref<Projection>)>,
    meshes: &mut ResMut<Assets<Mesh>>,
    settings_state: &Res<Persistent<SettingsState>>,
    analysis_state: &AnalysisState,
    range: &VqtRange,
) -> Result<()> {
    let (mut visibility, mesh_handle, mut transform) = spectrum.single_mut()?;
    // set visibility
    if settings_state.display_mode == DisplayMode::Debugging {
        *visibility = Visibility::Visible;
    } else {
        *visibility = Visibility::Hidden;
        return Ok(());
    }

    // move to right
    let (_, _, projection) = camera.single()?;
    {
        if let Projection::Orthographic(proj) = &*projection {
            let Rect { max, .. } = proj.area;
            *transform = Transform::from_xyz(
                max.x - range.n_buckets() as f32 * 0.011 - 0.2,
                max.y - 4.2,
                -13.0,
            );
        } else {
            panic!("Not an ortographic projection.");
        }
    }

    let x_vqt_smoothed = &analysis_state
        .x_vqt_smoothed
        .iter()
        .map(|ema| ema.get())
        .collect::<Vec<f32>>();

    let k_max = arg_max(&x_vqt_smoothed);
    let max_size = x_vqt_smoothed[k_max];

    // Build spectrum line geometry
    let line_list = LineList {
        lines: x_vqt_smoothed
            .iter()
            .enumerate()
            .map(|(i, amp)| Vec3::new(i as f32 * 0.011, amp / 10.0, 0.0))
            .tuple_windows()
            .collect::<Vec<(Vec3, Vec3)>>(),
        thickness: 0.02,
    };

    // Collect all vertices from line list
    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();

    // Build line list geometry manually (similar to LineList::into())
    for (p, q) in line_list.lines.iter() {
        let dx = p.x - q.x;
        let dy = p.y - q.y;
        let l = dx.hypot(dy);
        let u = dx * line_list.thickness * 0.5 / l;
        let v = dy * line_list.thickness * 0.5 / l;

        let v0 = Vec3::new(p.x + v, p.y - u, 0.0);
        let v1 = Vec3::new(p.x - v, p.y + u, 0.0);
        let v2 = Vec3::new(q.x - v, q.y + u, 0.0);
        let v3 = Vec3::new(q.x + v, q.y - u, 0.0);

        let prior_len = all_vertices.len();
        all_indices.push(2 + prior_len as u32);
        all_indices.push(1 + prior_len as u32);
        all_indices.push(prior_len as u32);
        all_indices.push(2 + prior_len as u32);
        all_indices.push(prior_len as u32);
        all_indices.push(3 + prior_len as u32);

        all_vertices.push((v0, [0.0, 0.0, 1.0], [0.0, 1.0]));
        all_vertices.push((v1, [0.0, 0.0, 1.0], [0.0, 0.0]));
        all_vertices.push((v2, [0.0, 0.0, 1.0], [1.0, 0.0]));
        all_vertices.push((v3, [0.0, 0.0, 1.0], [1.0, 1.0]));
    }

    // Generate colors for spectrum lines
    let mut all_colors = (0..(x_vqt_smoothed.len() - 1))
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
            let color_coefficient = 1.0 - (0.5 - x_vqt_smoothed[i] / max_size / 2.0).powf(0.5);
            [
                [r, g, b, color_coefficient],
                [r, g, b, color_coefficient],
                [r, g, b, color_coefficient],
                [r, g, b, color_coefficient],
            ]
        })
        .flatten()
        .collect::<Vec<[f32; 4]>>();

    // Add circles for detected peaks
    for peak in &analysis_state.peaks_continuous {
        let peak_x = peak.center * 0.011; // Same scaling as spectrum
        let peak_y = peak.size / 10.0; // Same scaling as spectrum

        // Calculate color for this peak based on its bin position
        let bin = peak.center.round() as usize;
        let (r, g, b) = calculate_color(
            range.buckets_per_octave,
            ((bin as f32)
                + 0.5
                + (range.buckets_per_octave - 3 * (range.buckets_per_octave / 12)) as f32)
                % range.buckets_per_octave as f32,
            COLORS,
            GRAY_LEVEL,
            10.0,
        );

        // Make peaks slightly brighter/more opaque
        let peak_color = [r, g, b, 0.9];

        // Create circle geometry
        let circle = CircleGeometry::new(
            Vec3::new(peak_x, peak_y, 0.0),
            0.08, // Circle radius
            peak_color,
            12, // Number of segments (12 for a smooth circle)
        );

        // Add circle vertices and indices
        let base_index = all_vertices.len() as u32;
        all_vertices.extend(circle.vertices);
        all_indices.extend(circle.indices.iter().map(|&i| i + base_index));
        all_colors.extend(circle.colors);
    }

    // Build final mesh
    use bevy::mesh::{Indices, PrimitiveTopology};
    let positions: Vec<_> = all_vertices.iter().map(|(p, _, _)| *p).collect();
    let normals: Vec<_> = all_vertices.iter().map(|(_, n, _)| *n).collect();
    let uvs: Vec<_> = all_vertices.iter().map(|(_, _, uv)| *uv).collect();

    let mut spectrum_mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    spectrum_mesh.insert_indices(Indices::U32(all_indices));
    spectrum_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    spectrum_mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    spectrum_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    spectrum_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, all_colors);

    if let Some(mut mesh) = meshes.get_mut(&mesh_handle.0) {
        *mesh = spectrum_mesh;
    }

    Ok(())
}

pub fn update_scene_calmness_graph(
    mut history: ResMut<SceneCalmnessHistory>,
    graph_query: Query<&Mesh2d, With<SceneCalmnessGraph>>,
    mut meshes: ResMut<Assets<Mesh>>,
    analysis_state: Res<AnalysisStateResource>,
    settings: Res<Persistent<SettingsState>>,
) {
    // Only update if in debugging mode
    if settings.display_mode != DisplayMode::Debugging {
        return;
    }

    // Get current calmness value
    let current_calmness = analysis_state.0.smoothed_scene_calmness.get();

    // Add to circular buffer - store write_index locally to avoid borrow checker issues
    let write_idx = history.write_index;
    history.values[write_idx] = current_calmness;
    history.write_index = (write_idx + 1) % history.capacity;

    // Build line list from circular buffer
    let mut points = Vec::new();
    for i in 0..history.capacity {
        let buffer_idx = (history.write_index + i) % history.capacity;
        let x = (i as f32 / history.capacity as f32) - 0.5;
        let y = history.values[buffer_idx];
        points.push(Vec3::new(x, y, 0.0));
    }

    // Create line segments with colors based on calmness value
    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();
    let mut all_colors = Vec::new();
    const THICKNESS: f32 = 0.01;

    for i in 0..(points.len() - 1) {
        let p = points[i];
        let q = points[i + 1];

        // Calculate color for this segment
        let buffer_idx = (history.write_index + i) % history.capacity;
        let calmness = history.values[buffer_idx];
        let color = calmness_to_color(calmness);
        let color_array = [
            color.to_srgba().red,
            color.to_srgba().green,
            color.to_srgba().blue,
            color.to_srgba().alpha,
        ];

        // Build quad for line segment
        let dx = p.x - q.x;
        let dy = p.y - q.y;
        let l = dx.hypot(dy);
        let u = dx * THICKNESS * 0.5 / l;
        let v = dy * THICKNESS * 0.5 / l;

        let v0 = Vec3::new(p.x + v, p.y - u, 0.0);
        let v1 = Vec3::new(p.x - v, p.y + u, 0.0);
        let v2 = Vec3::new(q.x - v, q.y + u, 0.0);
        let v3 = Vec3::new(q.x + v, q.y - u, 0.0);

        let base_idx = all_vertices.len() as u32;
        all_indices.extend_from_slice(&[
            base_idx + 2,
            base_idx + 1,
            base_idx,
            base_idx + 2,
            base_idx,
            base_idx + 3,
        ]);

        all_vertices.push((v0, [0.0, 0.0, 1.0], [0.0, 1.0]));
        all_vertices.push((v1, [0.0, 0.0, 1.0], [0.0, 0.0]));
        all_vertices.push((v2, [0.0, 0.0, 1.0], [1.0, 0.0]));
        all_vertices.push((v3, [0.0, 0.0, 1.0], [1.0, 1.0]));

        all_colors.extend_from_slice(&[color_array, color_array, color_array, color_array]);
    }

    // Build final mesh
    use bevy::mesh::{Indices, PrimitiveTopology};
    let positions: Vec<_> = all_vertices.iter().map(|(p, _, _)| *p).collect();
    let normals: Vec<_> = all_vertices.iter().map(|(_, n, _)| *n).collect();
    let uvs: Vec<_> = all_vertices.iter().map(|(_, _, uv)| *uv).collect();

    let mut new_mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    new_mesh.insert_indices(Indices::U32(all_indices));
    new_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    new_mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    new_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    new_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, all_colors);

    // Update the mesh
    if let Ok(mesh_handle) = graph_query.single() {
        if let Some(mut mesh) = meshes.get_mut(&mesh_handle.0) {
            *mesh = new_mesh;
        }
    }
}

pub fn update_calmness_histogram(
    mut histogram_query: Query<
        (&mut Visibility, &Mesh2d, &mut Transform),
        (
            With<CalmnessHistogram>,
            Without<PitchBall>,
            Without<BassCylinder>,
            Without<PitchNameText>,
            Without<Spectrum>,
            Without<SpiderNetSegment>,
        ),
    >,
    camera: &Query<(&mut Camera, Option<&mut Bloom>, Ref<Projection>)>,
    meshes: &mut ResMut<Assets<Mesh>>,
    analysis_state: &AnalysisState,
    settings: &Res<Persistent<SettingsState>>,
    range: &VqtRange,
) -> Result<()> {
    let (mut visibility, mesh_handle, mut transform) = histogram_query.single_mut()?;
    // set visibility
    if settings.display_mode == DisplayMode::Debugging {
        *visibility = Visibility::Visible;
    } else {
        *visibility = Visibility::Hidden;
        return Ok(());
    }

    // move to right
    let (_, _, projection) = camera.single()?;
    {
        if let Projection::Orthographic(proj) = &*projection {
            let Rect { max, .. } = proj.area;
            *transform = Transform::from_xyz(
                max.x - range.n_buckets() as f32 * 0.011 - 0.2,
                max.y - 4.2,
                -13.0,
            )
            .with_scale(Vec3::new(1.0, -1.0, 1.0));
        } else {
            panic!("Not an ortographic projection.");
        }
    }

    let calmness_values = &analysis_state.calmness;

    // Build contour line from calmness values (extending downward, 50% of spectrum height)
    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();
    let mut all_colors = Vec::new();
    const THICKNESS: f32 = 0.01;
    const HEIGHT_SCALE: f32 = 0.5;

    // Build line segments
    for i in 0..(calmness_values.len() - 1) {
        let calmness_0 = calmness_values[i].get();
        let calmness_1 = calmness_values[i + 1].get();

        let p = Vec3::new(i as f32 * 0.011, calmness_0 * HEIGHT_SCALE, 0.0);
        let q = Vec3::new((i + 1) as f32 * 0.011, calmness_1 * HEIGHT_SCALE, 0.0);

        // Calculate color for this segment (using midpoint calmness)
        let avg_calmness = (calmness_0 + calmness_1) / 2.0;
        let color = calmness_to_color(avg_calmness);
        let color_array = [
            color.to_srgba().red,
            color.to_srgba().green,
            color.to_srgba().blue,
            color.to_srgba().alpha,
        ];

        // Build quad for line segment
        let dx = p.x - q.x;
        let dy = p.y - q.y;
        let l = dx.hypot(dy);
        if l < 0.0001 {
            continue; // Skip degenerate segments
        }
        let u = dx * THICKNESS * 0.5 / l;
        let v = dy * THICKNESS * 0.5 / l;

        let v0 = Vec3::new(p.x + v, p.y - u, 0.0);
        let v1 = Vec3::new(p.x - v, p.y + u, 0.0);
        let v2 = Vec3::new(q.x - v, q.y + u, 0.0);
        let v3 = Vec3::new(q.x + v, q.y - u, 0.0);

        let base_idx = all_vertices.len() as u32;
        all_indices.extend_from_slice(&[
            base_idx + 2,
            base_idx + 1,
            base_idx,
            base_idx + 2,
            base_idx,
            base_idx + 3,
        ]);

        all_vertices.push((v0, [0.0, 0.0, 1.0], [0.0, 1.0]));
        all_vertices.push((v1, [0.0, 0.0, 1.0], [0.0, 0.0]));
        all_vertices.push((v2, [0.0, 0.0, 1.0], [1.0, 0.0]));
        all_vertices.push((v3, [0.0, 0.0, 1.0], [1.0, 1.0]));

        all_colors.extend_from_slice(&[color_array, color_array, color_array, color_array]);
    }

    // Build final mesh
    use bevy::mesh::{Indices, PrimitiveTopology};
    let positions: Vec<_> = all_vertices.iter().map(|(p, _, _)| *p).collect();
    let normals: Vec<_> = all_vertices.iter().map(|(_, n, _)| *n).collect();
    let uvs: Vec<_> = all_vertices.iter().map(|(_, _, uv)| *uv).collect();

    let mut new_mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    new_mesh.insert_indices(Indices::U32(all_indices));
    new_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    new_mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    new_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    new_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, all_colors);

    // Update the mesh
    if let Some(mut mesh) = meshes.get_mut(&mesh_handle.0) {
        *mesh = new_mesh;
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

    let Some(vis) = spider_net_segments.iter_mut().next() else {
        return;
    };
    if *vis != target_visibility {
        for mut visibility in &mut spider_net_segments {
            *visibility = target_visibility;
        }
    }
}

fn toggle_background(
    camera: &mut Query<(&mut Camera, Option<&mut Bloom>, Ref<Projection>)>,
    settings_state: &Res<Persistent<SettingsState>>,
    _analysis_state: &AnalysisState,
) -> Result<()> {
    let (mut cam, _, _) = camera.single_mut()?;

    let base_color = match settings_state.visuals_mode {
        VisualsMode::Zen | VisualsMode::Full | VisualsMode::Performance => CLEAR_COLOR_NEUTRAL,
        VisualsMode::Galaxy => CLEAR_COLOR_GALAXY,
    };

    cam.clear_color = base_color;

    Ok(())
}

// #[derive(PartialEq)]
// pub enum PauseState {
//     Running,
//     Paused,
// }

/// Update the spectrogram texture with current VQT data
pub fn update_spectrogram_system(
    spectrogram_res: Option<ResMut<SpectrogramResource>>,
    analysis_state: Res<AnalysisStateResource>,
    mut images: ResMut<Assets<Image>>,
    mut spectrogram_materials: ResMut<Assets<SpectrogramMaterial>>,
    settings: Res<Persistent<SettingsState>>,
) {
    // Only update if in debug mode
    if settings.display_mode != DisplayMode::Debugging {
        return;
    }

    let Some(mut spectrogram_res) = spectrogram_res else {
        return;
    };

    let Some(mut image) = images.get_mut(&spectrogram_res.image_handle) else {
        return;
    };

    let Some(image_data) = image.data.as_mut() else {
        return;
    };

    let analysis_state = &analysis_state.0;
    let width = analysis_state.range.n_buckets();
    let height = spectrogram_res.height;
    let write_idx = spectrogram_res.write_index;

    use super::SpectrogramMode;

    match settings.spectrogram_mode {
        SpectrogramMode::VQT => {
            // Get smoothed VQT data
            let vqt_data = &analysis_state.x_vqt_smoothed;

            // Find max for normalization
            let max_val = vqt_data.iter().map(|v| v.get()).fold(0.0f32, f32::max);

            // Write current VQT frame as a column at write_index
            for (bin_idx, vqt_value) in vqt_data.iter().enumerate() {
                let value_db = vqt_value.get();

                // Normalize and enhance for visualization
                let brightness = if max_val > 0.0 {
                    let normalized = value_db / (max_val + 0.001);
                    ((1.0 - (1.0 - normalized).powf(2.0)) * 1.5).clamp(0.0, 1.0)
                } else {
                    0.0
                };

                // Get pitch color
                let buckets_per_semitone = analysis_state.range.buckets_per_octave / 12;
                let semitone_offset =
                    (analysis_state.range.buckets_per_octave - 3 * buckets_per_semitone) as f32;
                let (r, g, b) = pitchvis_colors::calculate_color(
                    analysis_state.range.buckets_per_octave,
                    (bin_idx as f32 + semitone_offset)
                        % analysis_state.range.buckets_per_octave as f32,
                    COLORS,
                    GRAY_LEVEL,
                    EASING_POW,
                );

                // Calculate pixel position (flipped vertically, newest at top)
                let pixel_idx = ((height - 1 - write_idx) * width + bin_idx) * 4;

                if pixel_idx + 3 < image_data.len() {
                    image_data[pixel_idx] = (r * 255.0 * 1.2).clamp(0.0, 255.0) as u8;
                    image_data[pixel_idx + 1] = (g * 255.0 * 1.2).clamp(0.0, 255.0) as u8;
                    image_data[pixel_idx + 2] = (b * 255.0 * 1.2).clamp(0.0, 255.0) as u8;
                    image_data[pixel_idx + 3] = (brightness * 255.0 * 1.2).clamp(0.0, 255.0) as u8;
                }
            }
        }
        SpectrogramMode::Peaks => {
            // Render only continuous peaks with a small radius
            const PEAK_RADIUS: f32 = 2.0; // bins

            // Find max peak size for normalization
            let max_size = analysis_state
                .peaks_continuous
                .iter()
                .map(|p| p.size)
                .fold(0.0f32, f32::max);

            if max_size > 0.0 {
                for peak in &analysis_state.peaks_continuous {
                    let center = peak.center;
                    let size = peak.size;

                    // Normalize brightness
                    let brightness =
                        ((1.0 - (1.0 - size / max_size).powf(2.0)) * 1.5).clamp(0.0, 1.0);

                    // Get pitch color
                    let buckets_per_semitone = analysis_state.range.buckets_per_octave / 12;
                    let semitone_offset =
                        (analysis_state.range.buckets_per_octave - 3 * buckets_per_semitone) as f32;
                    let (r, g, b) = pitchvis_colors::calculate_color(
                        analysis_state.range.buckets_per_octave,
                        (center + semitone_offset) % analysis_state.range.buckets_per_octave as f32,
                        COLORS,
                        GRAY_LEVEL,
                        EASING_POW,
                    );

                    // Draw the peak with a Gaussian-like falloff
                    let min_bin = (center - PEAK_RADIUS).floor().max(0.0) as usize;
                    let max_bin = (center + PEAK_RADIUS).ceil().min(width as f32) as usize;

                    for bin_idx in min_bin..max_bin {
                        let distance = (bin_idx as f32 - center).abs();
                        if distance <= PEAK_RADIUS {
                            // Gaussian falloff
                            let falloff =
                                (-distance * distance / (PEAK_RADIUS * PEAK_RADIUS * 0.5)).exp();
                            let pixel_brightness = brightness * falloff;

                            // Calculate pixel position
                            let pixel_idx = ((height - 1 - write_idx) * width + bin_idx) * 4;

                            if pixel_idx + 3 < image_data.len() {
                                image_data[pixel_idx] = (r * 255.0 * 1.2).clamp(0.0, 255.0) as u8;
                                image_data[pixel_idx + 1] =
                                    (g * 255.0 * 1.2).clamp(0.0, 255.0) as u8;
                                image_data[pixel_idx + 2] =
                                    (b * 255.0 * 1.2).clamp(0.0, 255.0) as u8;
                                image_data[pixel_idx + 3] =
                                    (pixel_brightness * 255.0 * 1.2).clamp(0.0, 255.0) as u8;
                            }
                        }
                    }
                }
            }
        }
    }

    // Clear the next line (the one we'll write to next time)
    let next_idx = (write_idx + 1) % height;
    for bin_idx in 0..width {
        let pixel_idx = ((height - 1 - next_idx) * width + bin_idx) * 4;
        if pixel_idx + 3 < image_data.len() {
            image_data[pixel_idx] = 0;
            image_data[pixel_idx + 1] = 0;
            image_data[pixel_idx + 2] = 0;
            image_data[pixel_idx + 3] = 0;
        }
    }

    // Update write index
    spectrogram_res.write_index = next_idx;

    // Update material scroll offset for visual scrolling effect
    if let Some(mut material) = spectrogram_materials.get_mut(&spectrogram_res.material_handle) {
        material.scroll_params.scroll_offset = next_idx as f32 / height as f32;
    }
}

/// Update chroma display boxes with current pitch class energies
pub fn update_chroma_system(
    analysis_state: Res<AnalysisStateResource>,
    mut chroma_query: Query<(&ChromaBox, &mut BackgroundColor)>,
    settings: Res<Persistent<SettingsState>>,
) {
    // Only update if in debug mode
    if settings.display_mode != DisplayMode::Debugging {
        return;
    }

    let analysis_state = &analysis_state.0;

    // Calculate chroma features (sum of energy per pitch class)
    let mut chroma = vec![0.0; 12];

    // Calculate the pitch class of bin 0 (min_freq)
    // Reference: C4 = 261.626 Hz is pitch class 0
    // Formula: pitch_class = (12 * log2(freq / C4) + 0.5) % 12
    const C4_FREQ: f32 = 261.626;
    let semitones_from_c4 = 12.0 * (analysis_state.range.min_freq / C4_FREQ).log2();
    let bin_0_pitch_class = ((semitones_from_c4.round() as i32 % 12) + 12) % 12;

    for (bin_idx, measurement) in analysis_state.x_vqt_smoothed.iter().enumerate() {
        // Use proper rounding instead of truncation to get nearest semitone
        // This prevents bins from being incorrectly quantized (e.g., +50 cents being treated as in-tune)
        let semitone = ((bin_idx * 12) as f32 / analysis_state.range.buckets_per_octave as f32)
            .round() as usize;
        // Offset by the pitch class of bin 0
        let pitch_class = ((semitone as i32 + bin_0_pitch_class) % 12) as usize;

        // Convert from dB to power: power = 10^(dB/10)
        let power = 10f32.powf(measurement.get() / 10.0);
        chroma[pitch_class] += power;
    }

    // Normalize to 0-1 range
    let max_chroma = chroma.iter().copied().fold(0.0, f32::max);
    if max_chroma > 0.0 {
        for c in chroma.iter_mut() {
            *c /= max_chroma;
        }
    }

    // Update background colors with alpha based on chroma strength
    for (chroma_box, mut bg_color) in chroma_query.iter_mut() {
        let pitch_class = chroma_box.pitch_class;
        let strength = chroma[pitch_class];

        // Get base color from COLORS
        let [r, g, b] = pitchvis_colors::COLORS[pitch_class];

        // Set alpha based on chroma strength
        bg_color.0 = Color::srgba(r, g, b, strength);
    }
}

/// Show/hide spectrogram based on display mode
pub fn spectrogram_showhide(
    mut query: Query<&mut Visibility, With<SpectrogramDisplay>>,
    settings: Res<Persistent<SettingsState>>,
) {
    for mut visibility in query.iter_mut() {
        *visibility = if settings.display_mode == DisplayMode::Debugging {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

/// Show/hide chroma boxes based on display mode
pub fn chroma_showhide(
    mut query: Query<&mut Visibility, With<ChromaBox>>,
    settings: Res<Persistent<SettingsState>>,
) {
    for mut visibility in query.iter_mut() {
        *visibility = if settings.display_mode == DisplayMode::Debugging {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

/// Show/hide scene calmness graph based on display mode
pub fn scene_calmness_graph_showhide(
    mut query: Query<&mut Visibility, With<SceneCalmnessGraph>>,
    settings: Res<Persistent<SettingsState>>,
) {
    for mut visibility in query.iter_mut() {
        *visibility = if settings.display_mode == DisplayMode::Debugging {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}
