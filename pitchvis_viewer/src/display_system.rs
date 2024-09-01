// TODO: make a config object and pass that around instead of all these parameters

pub(crate) mod material;

use bevy::{
    math::vec3,
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology},
        render_asset::RenderAssetUsages,
    },
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
};
use pitchvis_analysis::{
    color_mapping::{COLORS, EASING_POW, GRAY_LEVEL, PITCH_NAMES},
    util::*,
};

use itertools::Itertools;
use nalgebra::{Rotation3, Vector3};

use std::collections::HashMap;
use std::f32::consts::PI;

use crate::display_system::material::NoisyColorMaterial;

const HIGHEST_BASSNOTE: usize = 12 * 2 + 4;
const CONTINUOUS_PEAKS_MODE: bool = true;

#[derive(Component)]
pub struct PitchBall(usize);

#[derive(Component)]
pub struct BassCylinder;

#[derive(Component)]
pub struct Spectrum;

#[derive(Component)]
pub struct PitchNameText;

#[derive(PartialEq)]
pub enum DisplayMode {
    PitchnamesCalmness,
    Calmness,
    Debugging,
}

#[derive(Resource)]
pub struct SettingsState {
    pub display_mode: DisplayMode,
}

/// keep an index -> entity mapping for the cylinders
#[derive(Resource)]
pub struct CylinderEntityListResource(pub Vec<Entity>);

pub fn setup_display_to_system(
    octaves: usize,
    buckets_per_octave: usize,
) -> impl FnMut(
    Commands,
    ResMut<Assets<Mesh>>,
    ResMut<Assets<ColorMaterial>>,
    ResMut<Assets<NoisyColorMaterial>>,
    ResMut<CylinderEntityListResource>,
) {
    move |commands: Commands,
          meshes: ResMut<Assets<Mesh>>,
          color_materials: ResMut<Assets<ColorMaterial>>,
          noisy_color_materials: ResMut<Assets<NoisyColorMaterial>>,
          cylinder_entities: ResMut<CylinderEntityListResource>| {
        setup_display(
            octaves,
            buckets_per_octave,
            commands,
            meshes,
            color_materials,
            noisy_color_materials,
            cylinder_entities,
        )
    }
}

pub fn setup_display(
    octaves: usize,
    buckets_per_octave: usize,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut color_materials: ResMut<Assets<ColorMaterial>>,
    mut noisy_color_materials: ResMut<Assets<NoisyColorMaterial>>,
    mut cylinder_entities: ResMut<CylinderEntityListResource>,
) {
    assert!(buckets_per_octave % 12 == 0);

    let spiral_points = calculate_spiral_points(octaves, buckets_per_octave);

    for (idx, (x, y, _z)) in spiral_points.iter().enumerate() {
        // spheres
        let noisy_color_material = NoisyColorMaterial {
            color: Color::srgb(1.0, 0.7, 0.6).into(),
            noise_level: 0.0,
        };
        commands.spawn((
            PitchBall(idx),
            MaterialMesh2dBundle {
                mesh: meshes.add(Circle::new(10.0)).into(),
                material: noisy_color_materials.add(noisy_color_material),
                transform: Transform::from_xyz(*x * 1.0, *y * 1.0, -0.01), // needs to be slightly behind the 2d camera
                visibility: if idx % 7 == 0 {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                },
                ..default()
            },
        ));
    }

    for (prev, cur) in spiral_points
        .iter()
        .take(HIGHEST_BASSNOTE * buckets_per_octave / 12)
        .tuple_windows()
    {
        use std::ops::Sub;

        let p = nalgebra::point![prev.0, prev.1, prev.2];
        let q = nalgebra::point![cur.0, cur.1, cur.2];

        let mid = nalgebra::center(&p, &q);
        let h = nalgebra::distance(&p, &q);
        let y_unit: Vector3<f32> = nalgebra::vector![0.0, 1.0, 0.0];
        let v_diff = p.sub(q);

        let mut transform = Transform::from_xyz(mid.x, mid.y, -12.7);
        if let Some(rotation) = Rotation3::rotation_between(&y_unit, &v_diff) {
            let (angx, angy, angz) = rotation.euler_angles();
            transform.rotate(Quat::from_euler(EulerRot::XYZ, angx, angy, angz));
        };

        cylinder_entities.0.push(
            commands
                .spawn((
                    BassCylinder,
                    MaterialMesh2dBundle {
                        mesh: meshes.add(Rectangle::new(0.05, h + 0.01)).into(),
                        material: color_materials.add(Color::srgb(0.8, 0.7, 0.6)),
                        transform,
                        visibility: Visibility::Hidden,
                        ..default()
                    },
                ))
                .id(),
        );
    }

    // draw rays
    let line_list: Vec<(Vec3, Vec3)> = (0..12)
        .map(|i| {
            let radius = octaves as f32 * 2.2; // 19.0 * 0.41421356237
            let (p_y, p_x) = (i as f32 / 12.0 * 2.0 * PI).sin_cos();

            (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(radius * p_x, radius * p_y, 0.0),
            )
        })
        .collect();
    commands.spawn(MaterialMesh2dBundle {
        mesh: meshes
            .add(Mesh::from(LineList {
                lines: line_list,
                flip: false,
                thickness: 0.05,
            }))
            .into(),
        material: color_materials.add(Color::srgb(0.3, 0.3, 0.3)),
        transform: Transform::from_xyz(0.0, 0.0, -13.0),
        ..default()
    });

    // draw spiral
    // TODO: consider a higher resolution, it's a bit choppy with buckets_per_semitone = 3
    let spiral_mesh = LineList {
        lines: spiral_points
            .iter()
            .map(|(x, y, z)| Vec3::new(*x, *y, *z))
            .tuple_windows()
            .collect::<Vec<(Vec3, Vec3)>>(),
        flip: false,
        thickness: 0.05,
    };
    commands.spawn(MaterialMesh2dBundle {
        mesh: meshes.add(spiral_mesh).into(),
        material: color_materials.add(Color::srgb(0.3, 0.3, 0.3)),
        transform: Transform::from_xyz(0.0, 0.0, -13.0),
        ..default()
    });

    // spectrum
    //#[cfg(feature = "ml")]
    // let spectrum_mesh = LineList {
    //     lines: (0..(octaves * buckets_per_octave))
    //         .map(|i| Vec3::new(i as f32 * 0.017, 0.0, 0.0))
    //         .tuple_windows()
    //         .collect::<Vec<(Vec3, Vec3)>>(),
    //     flip: false,
    //     thickness: 0.01,
    // };
    //#[cfg(feature = "ml")]
    // commands.spawn((
    //     Spectrum,
    //     MaterialMesh2dBundle {
    //         mesh: meshes.add(spectrum_mesh).into(),
    //         material: materials.add(Color::srgb(0.25, 0.85, 0.20)),
    //         transform: Transform::from_xyz(-12.0, 3.0, -13.0),
    //         ..default()
    //     },
    // ));

    // light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 10500.0,
            // Shadows makes some Android devices segfault, this is under investigation
            // https://github.com/bevyengine/bevy/issues/8214
            #[cfg(not(target_os = "android"))]
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(0.0, 0.0, 9.0),
        ..default()
    });
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 10500.0,
            //  Shadows makes some Android devices segfault, this is under investigation
            //https://github.com/bevyengine/bevy/issues/8214
            #[cfg(not(target_os = "android"))]
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(0.0, 0.0, -29.0),
        ..default()
    });

    // spawn a camera2dbundle with coordinates that match those of the 3d camera at the z=0 plane
    commands.spawn(Camera2dBundle {
        camera: Camera {
            // renders after / on top of the main camera
            order: 1,
            clear_color: ClearColorConfig::Custom(Color::srgb(0.23, 0.23, 0.25)),
            ..default()
        },
        projection: OrthographicProjection {
            scaling_mode: bevy::render::camera::ScalingMode::FixedVertical(38.0 * 0.414_213_57),
            scale: 1.00,
            ..default()
        },
        ..default()
    });

    // text
    let text_spiral_points = calculate_spiral_points(octaves, 12);
    for (idx, (x, y, _z)) in text_spiral_points[text_spiral_points.len() - 12..]
        .iter()
        .enumerate()
    {
        let pitch_idx = (idx + 12 - 3) % 12;
        let [r, g, b] = COLORS[pitch_idx];
        // squash it a bit in y direction and make it a bit larger in x direction to fit C and F# better
        let (x, y) = (x * (0.85 + 0.025 * x.abs()), y * (0.85 + 0.025 * x.abs()));
        commands.spawn((
            Text2dBundle {
                text: Text::from_section(
                    PITCH_NAMES[pitch_idx],
                    TextStyle {
                        font_size: 40.0,
                        color: Color::srgb(r, g, b),
                        ..default()
                    },
                )
                .with_justify(JustifyText::Center),
                // needs to be slightly behind the 2d camera and the balls
                transform: Transform::from_xyz(x, y, -0.02).with_scale(vec3(0.02, 0.02, 1.0)),
                visibility: Visibility::Visible,
                ..default()
            },
            PitchNameText,
        ));
    }
}

pub fn update_display_to_system(
    buckets_per_octave: usize,
    octaves: usize,
) -> impl FnMut(
    ParamSet<(
        Query<(
            &PitchBall,
            &mut Visibility,
            &mut Transform,
            &mut Handle<NoisyColorMaterial>,
        )>,
        Query<(&BassCylinder, &mut Visibility, &mut Handle<ColorMaterial>)>,
        Query<(&PitchNameText, &mut Visibility)>,
    )>,
    Query<(&Spectrum, &mut Mesh2dHandle)>,
    ResMut<Assets<ColorMaterial>>,
    ResMut<Assets<NoisyColorMaterial>>,
    ResMut<Assets<Mesh>>,
    Res<crate::analysis_system::AnalysisStateResource>,
    Res<crate::vqt_system::VqtResultResource>,
    Res<CylinderEntityListResource>,
    Res<SettingsState>,
) + Copy {
    move |set: ParamSet<(
        Query<(
            &PitchBall,
            &mut Visibility,
            &mut Transform,
            &mut Handle<NoisyColorMaterial>,
        )>,
        Query<(&BassCylinder, &mut Visibility, &mut Handle<ColorMaterial>)>,
        Query<(&PitchNameText, &mut Visibility)>,
    )>,
          spectrum_linestrip: Query<(&Spectrum, &mut Mesh2dHandle)>,
          color_materials: ResMut<Assets<ColorMaterial>>,
          noisy_color_materials: ResMut<Assets<NoisyColorMaterial>>,
          meshes: ResMut<Assets<Mesh>>,
          analysis_state: Res<crate::analysis_system::AnalysisStateResource>,
          vqt_result: Res<crate::vqt_system::VqtResultResource>,
          cylinder_entities: Res<CylinderEntityListResource>,
          settings_state: Res<SettingsState>| {
        update_display(
            buckets_per_octave,
            octaves,
            set,
            spectrum_linestrip,
            color_materials,
            noisy_color_materials,
            meshes,
            analysis_state,
            vqt_result,
            cylinder_entities,
            settings_state,
        );
    }
}

pub fn update_display(
    buckets_per_octave: usize,
    octaves: usize,
    mut set: ParamSet<(
        Query<(
            &PitchBall,
            &mut Visibility,
            &mut Transform,
            &mut Handle<NoisyColorMaterial>,
        )>,
        Query<(&BassCylinder, &mut Visibility, &mut Handle<ColorMaterial>)>,
        Query<(&PitchNameText, &mut Visibility)>,
    )>,
    mut spectrum_linestrip: Query<(&Spectrum, &mut Mesh2dHandle)>,
    color_materials: ResMut<Assets<ColorMaterial>>,
    mut noisy_color_materials: ResMut<Assets<NoisyColorMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    analysis_state: Res<crate::analysis_system::AnalysisStateResource>,
    vqt_result: Res<crate::vqt_system::VqtResultResource>,
    cylinder_entities: Res<CylinderEntityListResource>,
    settings_state: Res<SettingsState>,
) {
    let scale_factor = 1.0 / 305.0;

    for (pitch_ball, mut visibility, mut transform, color) in &mut set.p0() {
        if *visibility == Visibility::Visible {
            let idx = pitch_ball.0;
            let mut size = transform.scale / scale_factor;
            let dropoff_factor = 0.90 - 0.15 * (idx as f32 / (octaves * buckets_per_octave) as f32);
            size *= dropoff_factor;
            transform.scale = size * scale_factor;

            // also make them slightly more transparent when they are smaller
            let color_mat = noisy_color_materials
                .get_mut(&*color)
                .expect("ball color material");
            color_mat.color = color_mat.color.with_alpha(color_mat.color.alpha() * dropoff_factor);

            // also shift shrinking circles slightly to the background so that they are not cluttering newly appearing larger circles
            transform.translation.z -= 0.001;

            if size.x * scale_factor < 0.003 {
                *visibility = Visibility::Hidden;
            }

            // FIXME: test how it looks when we only show the balls that are currently active in the vqt analysis
            // *visibility = Visibility::Hidden;
        }
    }

    let analysis_state = &analysis_state.0;

    if analysis_state.peaks_continuous.is_empty() {
        return;
    }

    let k_max = arg_max(
        &analysis_state
            .peaks_continuous
            .iter()
            .map(|p| p.1)
            .collect::<Vec<f32>>(),
    );
    let max_size = analysis_state.peaks_continuous[k_max].1;

    if CONTINUOUS_PEAKS_MODE {
        let peaks_rounded = analysis_state
            .peaks_continuous
            .iter()
            .map(|p| (p.0.trunc() as usize, *p))
            .collect::<HashMap<usize, (f32, f32)>>();

        for (pitch_ball, mut visibility, mut transform, color) in &mut set.p0() {
            let idx = pitch_ball.0;
            if peaks_rounded.contains_key(&idx) {
                let (center, size) = peaks_rounded[&idx];

                let (r, g, b) = pitchvis_analysis::calculate_color(
                    buckets_per_octave,
                    (center + (buckets_per_octave - 3 * (buckets_per_octave / 12)) as f32)
                        % buckets_per_octave as f32,
                    COLORS,
                    GRAY_LEVEL,
                    EASING_POW,
                );

                let color_coefficient = 1.0 - (1.0 - size / max_size).powf(2.0);

                let (x, y, _) = bin_to_spiral(buckets_per_octave, center);
                // make sure larger circles are drawn on top by adding a small offset proportional to the size
                let z_ordering_offset = (size / max_size - 1.01) * 12.5;
                transform.translation = Vec3::new(x, y, z_ordering_offset);

                let color_mat = noisy_color_materials
                    .get_mut(&*color)
                    .expect("ball color material");
                // color_mat.color = Color::srgb(
                //     r * color_coefficient,
                //     g * color_coefficient,
                //     b * color_coefficient,
                // );
                color_mat.color = Color::srgba(r, g, b, color_coefficient).into();

                #[cfg(feature = "ml")]
                if let Some(midi_pitch) = vqt_bin_to_midi_pitch(buckets_per_octave, idx) {
                    let inferred_midi_pitch_strength =
                        analysis_state.ml_midi_base_pitches[midi_pitch];
                    if inferred_midi_pitch_strength > 0.35 {
                        color_mat.color = Color::srgba(r, g, b, 1.0);
                    } else {
                        color_mat.color = Color::srgba(r, g, b, color_coefficient * 0.1);
                    }
                }

                // set calmness visual effect.
                // FIXME: Usually we see values of 0.75 for very calm notes... Fix this to be more intuitive.
                if settings_state.display_mode == DisplayMode::PitchnamesCalmness
                    || settings_state.display_mode == DisplayMode::Calmness
                    || settings_state.display_mode == DisplayMode::Debugging
                {
                    color_mat.noise_level = (analysis_state.calmness[idx] - 0.27).clamp(0.0, 1.0);
                } else {
                    color_mat.noise_level = 0.0;
                }

                // scale and threshold to vanish
                transform.scale = Vec3::splat(size * scale_factor);
                if transform.scale.x >= 0.002 {
                    *visibility = Visibility::Visible;
                }
            }
        }
        // TODO: ?faster lookup through indexes
    } else {
        // let vqt_base = &analysis_state.x_vqt_peakfiltered;
        let vqt_base = &vqt_result.x_vqt;
        for (pitch_ball, mut visibility, mut transform, color) in &mut set.p0() {
            let idx = pitch_ball.0;
            let size = vqt_base[idx];

            let (r, g, b) = pitchvis_analysis::calculate_color(
                buckets_per_octave,
                (idx as f32 + (buckets_per_octave - 3 * (buckets_per_octave / 12)) as f32)
                    % buckets_per_octave as f32,
                COLORS,
                GRAY_LEVEL,
                EASING_POW,
            );

            let color_coefficient = 1.0 - (1.0 - size / max_size).powf(2.0);

            let (x, y, _) = bin_to_spiral(buckets_per_octave, idx as f32);
            // make sure larger circles are drawn on top by adding a small offset proportional to the size
            let z_ordering_offset = (size / max_size - 1.01) * 12.5;
            transform.translation = Vec3::new(x, y, z_ordering_offset);

            let color_mat = noisy_color_materials
                .get_mut(&*color)
                .expect("ball color material");
            // color_mat.color = Color::srgba(r, g, b, 1.0);
            color_mat.color = Color::srgba(r, g, b, color_coefficient).into();

            transform.scale = Vec3::splat(size * scale_factor);

            // if transform.scale.x >= 0.005 {
            *visibility = Visibility::Visible;
            // }
        }
    }

    update_bass_spiral(
        buckets_per_octave,
        cylinder_entities,
        color_materials,
        set.p1(),
        &analysis_state.peaks_continuous,
    );

    for (_, line_strip) in &mut spectrum_linestrip {
        let spectrum_mesh = LineList {
            lines: vqt_result
                .x_vqt
                .iter()
                .enumerate()
                .map(|(i, amp)| Vec3::new(i as f32 * 0.017, *amp / 10.0, 0.0))
                .tuple_windows()
                .collect::<Vec<(Vec3, Vec3)>>(),
            flip: false,
            thickness: 0.01,
        };
        let mesh = meshes
            .get_mut(&line_strip.0)
            .expect("spectrum line strip mesh");
        *mesh = spectrum_mesh.into();
    }

    for (_, mut visibility) in &mut set.p2() {
        if settings_state.display_mode == DisplayMode::PitchnamesCalmness
            || settings_state.display_mode == DisplayMode::Debugging
        {
            *visibility = Visibility::Visible;
        } else {
            *visibility = Visibility::Hidden;
        }
    }
}

// #[derive(PartialEq)]
// pub enum PauseState {
//     Running,
//     Paused,
// }

/// A list of lines with a start and end position each
#[derive(Debug, Clone)]
pub struct LineList {
    /// Full width and height of the rectangle.
    pub lines: Vec<(Vec3, Vec3)>,
    /// Horizontally-flip the texture coordinates of the resulting mesh.
    pub flip: bool,
    /// Width of the line
    pub thickness: f32,
}

impl From<LineList> for Mesh {
    fn from(strip: LineList) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        // iterate over strip.points, partitioned into tuples of (p, q)
        for (p, q) in strip.lines.iter() {
            let dx = p.x - q.x;
            let dy = p.y - q.y;
            let l = dx.hypot(dy);
            let u = dx * strip.thickness * 0.5 / l;
            let v = dy * strip.thickness * 0.5 / l;

            let v0 = Vec3::new(p.x + v, p.y - u, 0.0);
            let v1 = Vec3::new(p.x - v, p.y + u, 0.0);
            let v2 = Vec3::new(q.x - v, q.y + u, 0.0);
            let v3 = Vec3::new(q.x + v, q.y - u, 0.0);

            let prior_len = vertices.len();
            indices.push(2 + prior_len as u32);
            indices.push(1 + prior_len as u32);
            indices.push(prior_len as u32);
            indices.push(2 + prior_len as u32);
            indices.push(prior_len as u32);
            indices.push(3 + prior_len as u32);

            vertices.push((v0, [0.0, 0.0, 1.0], [0.0, 1.0]));
            vertices.push((v1, [0.0, 0.0, 1.0], [0.0, 0.0]));
            vertices.push((v2, [0.0, 0.0, 1.0], [1.0, 0.0]));
            vertices.push((v3, [0.0, 0.0, 1.0], [1.0, 1.0]));
        }

        let indices = Indices::U32(indices);

        let positions: Vec<_> = vertices.iter().map(|(p, _, _)| *p).collect();
        let normals: Vec<_> = vertices.iter().map(|(_, n, _)| *n).collect();
        let uvs: Vec<_> = vertices.iter().map(|(_, _, uv)| *uv).collect();

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_indices(indices);
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh
    }
}

// pub struct DisplayPlugin {
//     // pub cam: ArcBall,
//     // cubes: Vec<SceneNode>,
//     // /// Tuples of cylinder and fixed height because we reset scaling every frame
//     // cylinders: Vec<(SceneNode, f32)>,
//     // bg_quad: SceneNode,
//     // spectrogram_quad: SceneNode,
//     // spectrogram_tex: Rc<kiss3d::resource::Texture>,
//     octaves: usize,
//     buckets_per_octave: usize,
//     pause_state: PauseState,
//     //x_vqt_smoothed: Vec<f32>,
// }

// impl DisplayPlugin {
//     pub fn new(octaves: usize, buckets_per_octave: usize) -> Self {
//window.set_light(Light::StickToCamera);

// let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.014);

// window.set_line_width(0.7);

// let mut analysis_state =
//     AnalysisState::new(octaves * buckets_per_octave, SPECTROGRAM_LENGTH);
// analysis_state
//     .x_vqt_afterglow
//     .resize_with(octaves * buckets_per_octave, || 0.0);

// let mut bg_quad = window.add_quad(24.0, 12.0, 1, 1);
// let bg_small_jpg = include_bytes!("bg_europe.jpg");
// bg_quad.set_texture_from_memory(bg_small_jpg, "bg_small");
// bg_quad.prepend_to_local_translation(&kiss3d::nalgebra::Translation::from(
//     nalgebra::vector![0.0, 0.0, -0.5],
// ));

// let spectrogram_tex = kiss3d::resource::TextureManager::get_global_manager(move |tm| {
//     let img = image::DynamicImage::new_rgba8(
//         (octaves * buckets_per_octave) as u32,
//         SPECTROGRAM_LENGTH as u32,
//     );
//     tm.add_image(img, "spectrogram")
// });

// let mut spectrogram_quad = window.add_quad(7.0, 10.0, 1, 1);
// spectrogram_quad.set_texture(spectrogram_tex.clone());
// spectrogram_quad.prepend_to_local_translation(&kiss3d::nalgebra::Translation::from(
//     nalgebra::vector![-10.0, -2.1, -0.01],
// ));

//     Self {
//         // cam,
//         // cubes,
//         // cylinders,
//         // bg_quad,
//         // spectrogram_quad,
//         // spectrogram_tex,
//         octaves,
//         buckets_per_octave,
//         //analysis_state,
//         pause_state: PauseState::Running,
//         //x_vqt_smoothed : vec![0.0; octaves * buckets_per_octave],
//     }
// }

// fn toggle_pause(&mut self) {
//     self.pause_state = match self.pause_state {
//         PauseState::Running => PauseState::Paused,
//         _ => PauseState::Running,
//     };
//     debug!("toggling pause");
// }
// fn toggle_bg_display(&mut self) {
//     if self.bg_quad.is_visible() {
//         self.bg_quad.set_visible(false);
//     } else {
//         self.bg_quad.set_visible(true);
//     }
//     debug!("toggling bg image display");
// }

// fn handle_key_events(&mut self, window: &Window) {
//     for event in window.events().iter() {
//         if let kiss3d::event::WindowEvent::Key(c, a, _m) = event.value {
//             match (c, a) {
//                 (kiss3d::event::Key::P, kiss3d::event::Action::Press) => {
//                     self.toggle_pause();
//                 }
//                 (kiss3d::event::Key::I, kiss3d::event::Action::Press) => {
//                     self.toggle_bg_display();
//                 }
//                 _ => {}
//             }
//         }
//     }
// }

// pub fn render(&mut self, window: &mut Window, x_vqt: &[f32], gain: f32) {
//     self.handle_key_events(window);

//     self.preprocess(x_vqt);
//     self.update_spectrogram();
//     self.draw_spectrum(window);
//     self.draw_spider_net(window);
//     self.update_balls();
//     self.update_cylinders(gain);
// }

// fn update_spectrogram(&mut self) {
//     let k_max = arg_max(&self.analysis_state.x_vqt_smoothed);
//     let max_size = self.analysis_state.x_vqt_smoothed[k_max];

//     let width = self.octaves * self.buckets_per_octave;
//     self.analysis_state.spectrogram_buffer[(self.analysis_state.spectrogram_front_idx
//         * width
//         * 4)
//         ..((self.analysis_state.spectrogram_front_idx + 1) * width * 4)]
//         .fill(0);
//     //for (i, x) in self.analysis_state.x_vqt_smoothed.iter().enumerate() {
//     for i in self.analysis_state.peaks.iter() {
//         let x = self.analysis_state.x_vqt_smoothed[*i];
//         let (r, g, b) = calculate_color(
//             self.buckets_per_octave,
//             (*i as f32 + (self.buckets_per_octave - 3 * (self.buckets_per_octave / 12)) as f32)
//                 % self.buckets_per_octave as f32,
//         );
//         let brightness = x / (max_size + EPSILON);
//         let brightness = ((1.0 - (1.0 - brightness).powf(2.0)) * 1.5 * 255.0).clamp(0.0, 255.0);

//         // right to left
//         self.analysis_state.spectrogram_buffer
//             [((self.analysis_state.spectrogram_front_idx + 1) * width - i - 1) * 4 + 0] =
//             (r * brightness * 1.2).clamp(0.0, 255.0) as u8;
//         self.analysis_state.spectrogram_buffer
//             [((self.analysis_state.spectrogram_front_idx + 1) * width - i - 1) * 4 + 1] =
//             (g * brightness * 1.2).clamp(0.0, 255.0) as u8;
//         self.analysis_state.spectrogram_buffer
//             [((self.analysis_state.spectrogram_front_idx + 1) * width - i - 1) * 4 + 2] =
//             (b * brightness * 1.2).clamp(0.0, 255.0) as u8;
//         self.analysis_state.spectrogram_buffer
//             [((self.analysis_state.spectrogram_front_idx + 1) * width - i - 1) * 4 + 3] = 1;

//         if *i < width - 1 {
//             self.analysis_state.spectrogram_buffer
//                 [((self.analysis_state.spectrogram_front_idx + 1) * width - i - 2) * 4 + 0] =
//                 (r * brightness * 1.2).clamp(0.0, 255.0) as u8;
//             self.analysis_state.spectrogram_buffer
//                 [((self.analysis_state.spectrogram_front_idx + 1) * width - i - 2) * 4 + 1] =
//                 (g * brightness * 1.2).clamp(0.0, 255.0) as u8;
//             self.analysis_state.spectrogram_buffer
//                 [((self.analysis_state.spectrogram_front_idx + 1) * width - i - 2) * 4 + 2] =
//                 (b * brightness * 1.2).clamp(0.0, 255.0) as u8;
//             self.analysis_state.spectrogram_buffer
//                 [((self.analysis_state.spectrogram_front_idx + 1) * width - i - 2) * 4 + 3] = 1;
//         }
//         if *i > 0 {
//             self.analysis_state.spectrogram_buffer
//                 [((self.analysis_state.spectrogram_front_idx + 1) * width - i) * 4 + 0] =
//                 (r * brightness * 1.2).clamp(0.0, 255.0) as u8;
//             self.analysis_state.spectrogram_buffer
//                 [((self.analysis_state.spectrogram_front_idx + 1) * width - i) * 4 + 1] =
//                 (g * brightness * 1.2).clamp(0.0, 255.0) as u8;
//             self.analysis_state.spectrogram_buffer
//                 [((self.analysis_state.spectrogram_front_idx + 1) * width - i) * 4 + 2] =
//                 (b * brightness * 1.2).clamp(0.0, 255.0) as u8;
//             self.analysis_state.spectrogram_buffer
//                 [((self.analysis_state.spectrogram_front_idx + 1) * width - i) * 4 + 3] = 1;
//         }
//     }

//     self.analysis_state.spectrogram_front_idx =
//         (self.analysis_state.spectrogram_front_idx + 1) % SPECTROGRAM_LENGTH;
//     self.analysis_state.spectrogram_buffer[(self.analysis_state.spectrogram_front_idx
//         * width
//         * 4)
//         ..((self.analysis_state.spectrogram_front_idx + 1) * width * 4)]
//         .fill(255);

//     // clear also further ahead
//     let further_idx = (self.analysis_state.spectrogram_front_idx + SPECTROGRAM_LENGTH / 10)
//         % SPECTROGRAM_LENGTH;
//     self.analysis_state.spectrogram_buffer
//         [(further_idx * width * 4)..((further_idx + 1) * width * 4)]
//         .fill(0);

//     let context = kiss3d::context::Context::get();
//     context.bind_texture(
//         kiss3d::context::Context::TEXTURE_2D,
//         Some(&self.spectrogram_tex),
//     );
//     context.tex_sub_image2d(
//         kiss3d::context::Context::TEXTURE_2D,
//         0,
//         0,
//         0,
//         (self.octaves * self.buckets_per_octave) as i32,
//         SPECTROGRAM_LENGTH as i32,
//         kiss3d::context::Context::RGBA,
//         Some(&self.analysis_state.spectrogram_buffer),
//     );
// }

// fn draw_spectrum(&mut self, window: &mut Window) {
//     let x_vqt = &self.analysis_state.x_vqt_smoothed;

//     for i in 0..(self.buckets_per_octave * self.octaves - 1) {
//         let x = i as f32 / (self.buckets_per_octave * self.octaves) as f32 * 7.0 - 13.5;
//         let x_next =
//             (i + 1) as f32 / (self.buckets_per_octave * self.octaves) as f32 * 7.0 - 13.5;
//         let y_scale = 7.0;
//         window.draw_line(
//             &Point3::new(x, x_vqt[i] / y_scale + 3.0, 0.0),
//             &Point3::new(x_next, x_vqt[i + 1] / y_scale + 3.0, 0.0),
//             //&Point3::new(x, x_vqt_smoothed[i] /* / y_scale */ + 3.0, 0.0),
//             //&Point3::new(x_next, x_vqt_smoothed[i + 1] /* / y_scale */ + 3.0, 0.0),
//             &Point3::new(0.7, 0.9, 0.0),
//         );
//         if self.analysis_state.peaks.contains(&i) {
//             window.draw_line(
//                 //&Point3::new(x, x_vqt_smoothed[i] /*/ y_scale*/ + 3.0 - 0.1, 0.0),
//                 //&Point3::new(x, x_vqt_smoothed[i] /*/ y_scale*/ + 3.0, 0.0),
//                 &Point3::new(x, x_vqt[i] / y_scale + 3.0 + 0.2, 0.0),
//                 &Point3::new(x, x_vqt[i] / y_scale + 3.0, 0.0),
//                 &Point3::new(1.0, 0.2, 0.0),
//             );
//         }

//         if i % (self.buckets_per_octave / 12) == 0 {
//             window.draw_line(
//                 &Point3::new(x, x_vqt[i] / y_scale + 3.0 - 0.1, 0.0),
//                 &Point3::new(x, x_vqt[i] / y_scale + 3.0, 0.0),
//                 // &Point3::new(x, x_vqt_smoothed[i] /*/ y_scale*/ + 3.0 - 0.1, 0.0),
//                 // &Point3::new(x, x_vqt_smoothed[i] /*/ y_scale*/ + 3.0, 0.0),
//                 &Point3::new(1.0, 0.0, 1.0),
//             );
//         }
//     }
// }

// fn draw_spider_net(&mut self, window: &mut Window) {
//     // draw rays
//     for i in 0..12 {
//         let radius = self.octaves as f32 * 2.2;
//         let (p_y, p_x) = (i as f32 / 12.0 * 2.0 * PI).sin_cos();
//         window.draw_line(
//             &Point3::new(0.0, 0.0, 0.0),
//             &Point3::new(radius * p_x, radius * p_y, 0.0),
//             &Point3::new(0.20, 0.25, 0.20),
//         );
//     }

//     // draw spiral
//     // TODO: make these constant things constant
//     let spiral_points = Self::spiral_points(self.octaves, self.buckets_per_octave);
//     for (prev, cur) in spiral_points.iter().tuple_windows() {
//         window.draw_line(
//             &Point3::new(prev.0, prev.1, prev.2),
//             &Point3::new(cur.0, cur.1, cur.2),
//             &Point3::new(0.25, 0.20, 0.20),
//         );
//     }
// }

// fn update_balls(&mut self) {
//     let scale_factor = 1.0 / 18.0;

//     for (i, c) in self.cubes.iter_mut().enumerate() {
//         if c.is_visible() {
//             let mut size = c.data().local_scale().x / scale_factor;
//             size *= 0.90 - 0.15 * (i as f32 / (self.octaves * self.buckets_per_octave) as f32);
//             c.set_local_scale(
//                 size * scale_factor,
//                 size * scale_factor,
//                 size * scale_factor,
//             );
//             if size * scale_factor < 0.4 {
//                 c.set_visible(false);
//             }
//         }
//     }

//     if self.analysis_state.peaks_continuous.is_empty() {
//         return;
//     }

//     let k_max = arg_max(
//         &self
//             .analysis_state
//             .peaks_continuous
//             .iter()
//             .map(|p| p.1)
//             .collect::<Vec<f32>>(),
//     );
//     let max_size = self.analysis_state.peaks_continuous[k_max].1;

//     for p in &self.analysis_state.peaks_continuous {
//         let (center, size) = *p;

//         let (r, g, b) = calculate_color(
//             self.buckets_per_octave,
//             (center + (self.buckets_per_octave - 3 * (self.buckets_per_octave / 12)) as f32)
//                 % self.buckets_per_octave as f32,
//         );

//         let color_coefficient = 1.0 - (1.0 - size / max_size).powf(2.0);

//         let c = &mut self.cubes[center.trunc() as usize];
//         let (x, y, z) = Self::bin_to_spiral(self.buckets_per_octave, center);
//         c.set_local_translation(kiss3d::nalgebra::Translation::from([x, y, z]));
//         c.set_color(
//             r * color_coefficient,
//             g * color_coefficient,
//             b * color_coefficient,
//         );

//         //c.set_local_scale((x_vqt[i] / 10.0).max(0.1), (x_vqt[i] / 10.0).max(0.1), (x_vqt[i] / 10.0).max(0.1));

//         //let scale_factor = 1.0 / 30.0 * (0.7 + 0.3 * local_maximum as i32 as f32);

//         //let scale_factor = 1.0;

//         c.set_local_scale(
//             size * scale_factor,
//             size * scale_factor,
//             size * scale_factor,
//         );

//         c.set_visible(true);
//     }
// }

fn update_bass_spiral(
    buckets_per_octave: usize,
    cylinder_entities: Res<CylinderEntityListResource>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut bass_cylinders: Query<(&BassCylinder, &mut Visibility, &mut Handle<ColorMaterial>)>,
    peaks_continuous: &[(f32, f32)],
) {
    //let mut color_map: Vec<i32> = vec![-1; self.buckets_per_octave * self.octaves];
    // for (prev, cur) in peaks.iter().tuple_windows() {
    //     color_map[*prev..*cur].fill(*prev as i32);
    // }
    for (_, mut visibility, _) in &mut bass_cylinders {
        *visibility = Visibility::Hidden;
    }
    // if gain > 1000.0 {
    //     return;
    // }
    if let Some((center, size)) = peaks_continuous.first() {
        if center.trunc() as usize >= cylinder_entities.0.len() {
            return;
        }

        // color up to lowest note
        for idx in 0..(center.trunc() as usize) {
            let (_, ref mut visibility, color) = bass_cylinders
                .get_mut(cylinder_entities.0[idx])
                .expect("cylinder entity");
            **visibility = Visibility::Visible;

            let color_map_ref = center.trunc() as usize;
            let (r, g, b) = pitchvis_analysis::color_mapping::calculate_color(
                buckets_per_octave,
                (color_map_ref + buckets_per_octave - 3 * (buckets_per_octave / 12)) as f32
                    % buckets_per_octave as f32,
                COLORS,
                GRAY_LEVEL,
                EASING_POW,
            );

            let k_max = arg_max(&peaks_continuous.iter().map(|p| p.1).collect::<Vec<f32>>());
            let max_size = peaks_continuous[k_max].1;

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

fn calculate_spiral_points(octaves: usize, buckets_per_octave: usize) -> Vec<(f32, f32, f32)> {
    (0..(buckets_per_octave * octaves))
        .map(|i| bin_to_spiral(buckets_per_octave, i as f32))
        .collect()
}

fn bin_to_spiral(buckets_per_octave: usize, x: f32) -> (f32, f32, f32) {
    //let radius = 1.5 * (0.5 + (x / buckets_per_octave as f32).powf(0.75));
    let radius = 2.0 * (0.3 + (x / buckets_per_octave as f32).powf(0.75));
    #[allow(clippy::erasing_op)]
    let (transl_y, transl_x) = ((x + (buckets_per_octave - 0 * (buckets_per_octave / 12)) as f32)
        / (buckets_per_octave as f32)
        * 2.0
        * PI)
        .sin_cos();
    (-1.0 * transl_x * radius, transl_y * radius, 0.0) //17.0 - radius)
}

#[cfg(feature = "ml")]
fn vqt_bin_to_midi_pitch(buckets_per_octave: usize, bin: usize) -> Option<usize> {
    let midi_pitch = (bin as f32 / buckets_per_octave as f32 * 12.0).round() as usize
        + crate::FREQ_A1_MIDI_KEY_ID as usize;
    if midi_pitch > 127 {
        None
    } else {
        Some(midi_pitch)
    }
}
