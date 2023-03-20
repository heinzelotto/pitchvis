use crate::util::*;
use bevy::{
    core_pipeline::clear_color::ClearColorConfig,
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    reflect::TypeUuid,
    render::{
        mesh::{MeshVertexBufferLayout, PrimitiveTopology},
        render_resource::{
            AsBindGroup, PolygonMode, RenderPipelineDescriptor, ShaderRef,
            SpecializedMeshPipelineError,
        },
    },
};

use itertools::Itertools;
use log::debug;
use nalgebra::{Rotation3, Vector3};

use std::collections::HashMap;
use std::f32::consts::PI;

const HIGHEST_BASSNOTE: usize = 12 * 2 + 4;

const _COLORS_WONKY_SATURATION: [(f32, f32, f32); 12] = [
    (0.80, 0.40, 0.39), // C
    (0.24, 0.51, 0.66), // C#
    (0.96, 0.77, 0.25), // D
    (0.36, 0.28, 0.49), // Eb
    (0.51, 0.76, 0.30), // E
    (0.74, 0.36, 0.51), // F
    (0.27, 0.62, 0.56), // F#
    (0.91, 0.56, 0.30), // G
    (0.26, 0.31, 0.52), // Ab
    (0.85, 0.87, 0.26), // A
    (0.54, 0.31, 0.53), // Bb
    (0.27, 0.69, 0.39), // H
];

const COLORS: [(f32, f32, f32); 12] = [
    (0.85, 0.36, 0.36), // C
    (0.01, 0.52, 0.71), // C#
    (0.97, 0.76, 0.05), // D
    //(0.37, 0.28, 0.50), // Eb
    (0.45, 0.34, 0.63), // Eb
    (0.47, 0.77, 0.22), // E
    (0.78, 0.32, 0.52), // Fh
    (0.00, 0.64, 0.56), // F#
    (0.95, 0.54, 0.23), // G
    //(0.26, 0.31, 0.53), // Ab
    (0.30, 0.37, 0.64), // Ab
    (1.00, 0.96, 0.03), // A
    (0.57, 0.30, 0.55), // Bb
    (0.12, 0.71, 0.34), // H
];

fn calculate_color(buckets_per_octave: usize, bucket: f32) -> (f32, f32, f32) {
    const GRAY_LEVEL: f32 = 0.6; // could be the mean lightness of the two neighbors. for now this is good enough.
    const EASING_POW: f32 = 1.3;

    let pitch_continuous = 12.0 * bucket / (buckets_per_octave as f32);
    let base_color = COLORS[(pitch_continuous.round() as usize) % 12];
    let inaccuracy_cents = (pitch_continuous - pitch_continuous.round()).abs();

    let mut base_lcha = Color::rgb(base_color.0, base_color.1, base_color.2).as_lcha();
    if let Color::Lcha {
        ref mut lightness,
        ref mut chroma,
        hue: _,
        alpha: _,
    } = base_lcha
    {
        let saturation = 1.0 - (2.0 * inaccuracy_cents).powf(EASING_POW);
        *chroma *= saturation;
        *lightness = saturation * *lightness + (1.0 - saturation) * GRAY_LEVEL;
    }

    (
        base_lcha.as_rgba().r(),
        base_lcha.as_rgba().g(),
        base_lcha.as_rgba().b(),
    )
}

pub fn update_display(
    mut balls: Query<(
        &PitchBall,
        &mut Visibility,
        &mut Transform,
        &mut Handle<StandardMaterial>,
    )>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    analysis_state: Res<crate::analysis_system::AnalysisStateResource>,
) {
    let scale_factor = 1.0 / 35.0;

    for (pitch_ball, mut visibility, mut transform, _) in &mut balls {
        if *visibility == Visibility::Visible {
            let idx = pitch_ball.0;
            let mut size = transform.scale / scale_factor;
            size *=
                0.90 - 0.15 * (idx as f32 / (crate::OCTAVES * crate::BUCKETS_PER_OCTAVE) as f32);
            transform.scale = size * scale_factor;

            if size.x * scale_factor < 0.2 {
                *visibility = Visibility::Hidden;
            }
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

    let peaks_rounded = analysis_state
        .peaks_continuous
        .iter()
        .map(|p| (p.0.trunc() as usize, *p))
        .collect::<HashMap<usize, (f32, f32)>>();

    for (pitch_ball, mut visibility, mut transform, color) in &mut balls {
        let idx = pitch_ball.0;
        if peaks_rounded.contains_key(&idx) {
            let (center, size) = peaks_rounded[&idx];

            let (r, g, b) = calculate_color(
                crate::BUCKETS_PER_OCTAVE,
                (center
                    + (crate::BUCKETS_PER_OCTAVE - 3 * (crate::BUCKETS_PER_OCTAVE / 12)) as f32)
                    % crate::BUCKETS_PER_OCTAVE as f32,
            );

            let color_coefficient = 1.0 - (1.0 - size / max_size).powf(2.0);

            let (x, y, z) = bin_to_spiral(crate::BUCKETS_PER_OCTAVE, center);
            transform.translation = Vec3::new(x, y, z);

            let mut color_mat = materials.get_mut(&color).expect("ball color material");
            color_mat.base_color = Color::rgb(
                r * color_coefficient,
                g * color_coefficient,
                b * color_coefficient,
            );

            transform.scale = Vec3::splat(size * scale_factor);

            *visibility = Visibility::Visible;
        }
    }
    // TODO: ?faster lookup through indexes
}

#[derive(PartialEq)]
pub enum PauseState {
    Running,
    Paused,
}

#[derive(Component)]
pub struct PitchBall(usize);

#[derive(Component)]
pub struct BassCylinder;

/// A list of points that will have a line drawn between each consecutive points
#[derive(Debug, Clone)]
pub struct LineStrip {
    pub points: Vec<Vec3>,
}

impl From<LineStrip> for Mesh {
    fn from(line: LineStrip) -> Self {
        // This tells wgpu that the positions are a list of points
        // where a line will be drawn between each consecutive point
        let mut mesh = Mesh::new(PrimitiveTopology::LineStrip);

        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, line.points);
        mesh
    }
}

/// A list of lines with a start and end position
#[derive(Debug, Clone)]
pub struct LineList {
    pub lines: Vec<(Vec3, Vec3)>,
}

impl From<LineList> for Mesh {
    fn from(line: LineList) -> Self {
        // This tells wgpu that the positions are list of lines
        // where every pair is a start and end point
        let mut mesh = Mesh::new(PrimitiveTopology::LineList);

        let vertices: Vec<_> = line.lines.into_iter().flat_map(|(a, b)| [a, b]).collect();
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
        mesh
    }
}

#[derive(Default, AsBindGroup, TypeUuid, Debug, Clone)]
#[uuid = "050ce6ac-080a-4d8c-b6b5-b5bab7560d8f"]
pub struct LineMaterial {
    #[uniform(0)]
    color: Color,
}

impl Material for LineMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/line_material.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayout,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // This is the important part to tell bevy to render this material as a line between vertices
        descriptor.primitive.polygon_mode = PolygonMode::Line;
        Ok(())
    }
}

pub fn setup_display_to_system(
    octaves: usize,
    buckets_per_octave: usize,
) -> impl FnMut(
    Commands,
    ResMut<Assets<Mesh>>,
    ResMut<Assets<StandardMaterial>>,
    ResMut<Assets<LineMaterial>>,
) {
    return move |mut commands: Commands,
                 mut meshes: ResMut<Assets<Mesh>>,
                 mut materials: ResMut<Assets<StandardMaterial>>,
                 mut line_materials: ResMut<Assets<LineMaterial>>| {
        let spiral_points = spiral_points(octaves, buckets_per_octave);

        for (idx, (x, y, z)) in spiral_points.iter().enumerate() {
            // spheres
            let mut standard_material: StandardMaterial = Color::rgb(1.0, 0.7, 0.6).into();
            standard_material.perceptual_roughness = 0.3;
            standard_material.metallic = 0.1;
            commands.spawn((
                PitchBall(idx),
                PbrBundle {
                    mesh: meshes.add(
                        Mesh::try_from(shape::Icosphere {
                            radius: 1.0,
                            subdivisions: 4,
                        })
                        .expect("spheres meshes"),
                    ),
                    material: materials.add(standard_material),
                    transform: Transform::from_xyz(*x, *y, *z),
                    visibility: Visibility::Hidden,
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

            let mut transform = Transform::from_xyz(mid.x, mid.y, mid.z);
            if let Some(rotation) = Rotation3::rotation_between(&y_unit, &v_diff) {
                let (angx, angy, angz) = rotation.euler_angles();
                transform.rotate(Quat::from_euler(EulerRot::XYZ, angx, angy, angz));
            };

            //cylinders.push((c, h + 0.01));

            // cylinders
            commands.spawn((
                BassCylinder,
                PbrBundle {
                    mesh: meshes.add(Mesh::from(shape::Cylinder {
                        radius: 0.05,
                        height: h + 0.01,
                        resolution: 5,
                        segments: 1,
                    })),
                    material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
                    transform,
                    visibility: Visibility::Hidden,
                    ..default()
                },
            ));
        }

        // draw rays
        let line_list: Vec<(Vec3, Vec3)> = (0..12)
            .map(|i| {
                let radius = crate::OCTAVES as f32 * 2.2;
                let (p_y, p_x) = (i as f32 / 12.0 * 2.0 * PI).sin_cos();

                (
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(radius * p_x, radius * p_y, 0.0),
                )
            })
            .collect();

        commands.spawn(MaterialMeshBundle {
            mesh: meshes.add(Mesh::from(LineList { lines: line_list })),
            material: line_materials.add(LineMaterial {
                color: Color::rgb(0.25, 0.20, 0.20),
            }),
            ..default()
        });

        commands.spawn(MaterialMeshBundle {
            mesh: meshes.add(Mesh::from(LineStrip {
                points: spiral_points
                    .iter()
                    .map(|(x, y, z)| Vec3::new(*x, *y, *z))
                    .collect::<Vec<Vec3>>(),
            })),
            material: line_materials.add(LineMaterial {
                color: Color::rgb(0.25, 0.20, 0.20),
            }),
            ..default()
        });

        // light
        commands.spawn(PointLightBundle {
            point_light: PointLight {
                intensity: 10500.0,
                shadows_enabled: true,
                ..default()
            },
            transform: Transform::from_xyz(0.0, 0.0, 9.0),
            ..default()
        });
        commands.spawn(PointLightBundle {
            point_light: PointLight {
                intensity: 10500.0,
                shadows_enabled: true,
                ..default()
            },
            transform: Transform::from_xyz(0.0, 0.0, -29.0),
            ..default()
        });

        // camera
        commands.spawn(Camera3dBundle {
            transform: Transform::from_xyz(1.0, 0.0, 19.0).looking_at(Vec3::ZERO, Vec3::Y),
            // clear the whole viewport with the given color
            camera_3d: Camera3d {
                // clear the whole viewport with the given color
                //clear_color: ClearColorConfig::Custom(Color::rgb(0.23, 0.23, 0.25)),
                ..Default::default()
            },
            ..default()
        });
    };
}

pub struct DisplayPlugin {
    // pub cam: ArcBall,
    // cubes: Vec<SceneNode>,
    // /// Tuples of cylinder and fixed height because we reset scaling every frame
    // cylinders: Vec<(SceneNode, f32)>,
    // bg_quad: SceneNode,
    // spectrogram_quad: SceneNode,
    // spectrogram_tex: Rc<kiss3d::resource::Texture>,
    octaves: usize,
    buckets_per_octave: usize,
    pause_state: PauseState,
    //x_cqt_smoothed: Vec<f32>,
}

impl DisplayPlugin {
    pub fn new(octaves: usize, buckets_per_octave: usize) -> Self {
        //window.set_light(Light::StickToCamera);

        // let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.014);

        // window.set_line_width(0.7);

        // let mut analysis_state =
        //     AnalysisState::new(octaves * buckets_per_octave, SPECTROGRAM_LENGTH);
        // analysis_state
        //     .x_cqt_afterglow
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

        Self {
            // cam,
            // cubes,
            // cylinders,
            // bg_quad,
            // spectrogram_quad,
            // spectrogram_tex,
            octaves,
            buckets_per_octave,
            //analysis_state,
            pause_state: PauseState::Running,
            //x_cqt_smoothed : vec![0.0; octaves * buckets_per_octave],
        }
    }

    fn toggle_pause(&mut self) {
        self.pause_state = match self.pause_state {
            PauseState::Running => PauseState::Paused,
            _ => PauseState::Running,
        };
        debug!("toggling pause");
    }
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

    // pub fn render(&mut self, window: &mut Window, x_cqt: &[f32], gain: f32) {
    //     self.handle_key_events(window);

    //     self.preprocess(x_cqt);
    //     self.update_spectrogram();
    //     self.draw_spectrum(window);
    //     self.draw_spider_net(window);
    //     self.update_balls();
    //     self.update_cylinders(gain);
    // }

    // fn update_spectrogram(&mut self) {
    //     let k_max = arg_max(&self.analysis_state.x_cqt_smoothed);
    //     let max_size = self.analysis_state.x_cqt_smoothed[k_max];

    //     let width = self.octaves * self.buckets_per_octave;
    //     self.analysis_state.spectrogram_buffer[(self.analysis_state.spectrogram_front_idx
    //         * width
    //         * 4)
    //         ..((self.analysis_state.spectrogram_front_idx + 1) * width * 4)]
    //         .fill(0);
    //     //for (i, x) in self.analysis_state.x_cqt_smoothed.iter().enumerate() {
    //     for i in self.analysis_state.peaks.iter() {
    //         let x = self.analysis_state.x_cqt_smoothed[*i];
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
    //     let x_cqt = &self.analysis_state.x_cqt_smoothed;

    //     for i in 0..(self.buckets_per_octave * self.octaves - 1) {
    //         let x = i as f32 / (self.buckets_per_octave * self.octaves) as f32 * 7.0 - 13.5;
    //         let x_next =
    //             (i + 1) as f32 / (self.buckets_per_octave * self.octaves) as f32 * 7.0 - 13.5;
    //         let y_scale = 7.0;
    //         window.draw_line(
    //             &Point3::new(x, x_cqt[i] / y_scale + 3.0, 0.0),
    //             &Point3::new(x_next, x_cqt[i + 1] / y_scale + 3.0, 0.0),
    //             //&Point3::new(x, x_cqt_smoothed[i] /* / y_scale */ + 3.0, 0.0),
    //             //&Point3::new(x_next, x_cqt_smoothed[i + 1] /* / y_scale */ + 3.0, 0.0),
    //             &Point3::new(0.7, 0.9, 0.0),
    //         );
    //         if self.analysis_state.peaks.contains(&i) {
    //             window.draw_line(
    //                 //&Point3::new(x, x_cqt_smoothed[i] /*/ y_scale*/ + 3.0 - 0.1, 0.0),
    //                 //&Point3::new(x, x_cqt_smoothed[i] /*/ y_scale*/ + 3.0, 0.0),
    //                 &Point3::new(x, x_cqt[i] / y_scale + 3.0 + 0.2, 0.0),
    //                 &Point3::new(x, x_cqt[i] / y_scale + 3.0, 0.0),
    //                 &Point3::new(1.0, 0.2, 0.0),
    //             );
    //         }

    //         if i % (self.buckets_per_octave / 12) == 0 {
    //             window.draw_line(
    //                 &Point3::new(x, x_cqt[i] / y_scale + 3.0 - 0.1, 0.0),
    //                 &Point3::new(x, x_cqt[i] / y_scale + 3.0, 0.0),
    //                 // &Point3::new(x, x_cqt_smoothed[i] /*/ y_scale*/ + 3.0 - 0.1, 0.0),
    //                 // &Point3::new(x, x_cqt_smoothed[i] /*/ y_scale*/ + 3.0, 0.0),
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

    //         //c.set_local_scale((x_cqt[i] / 10.0).max(0.1), (x_cqt[i] / 10.0).max(0.1), (x_cqt[i] / 10.0).max(0.1));

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

    // fn update_cylinders(&mut self, gain: f32) {
    //     //let mut color_map: Vec<i32> = vec![-1; self.buckets_per_octave * self.octaves];
    //     // for (prev, cur) in peaks.iter().tuple_windows() {
    //     //     color_map[*prev..*cur].fill(*prev as i32);
    //     // }
    //     self.cylinders
    //         .iter_mut()
    //         .for_each(|c| c.0.set_visible(false));
    //     if gain > 1000.0 {
    //         return;
    //     }
    //     if let Some((center, size)) = self.analysis_state.peaks_continuous.first() {
    //         if center.trunc() as usize >= self.cylinders.len() {
    //             return;
    //         }

    //         // color up to lowest note
    //         for idx in 0..(center.trunc() as usize) {
    //             let (ref mut c, ref height) = self.cylinders[idx];
    //             c.set_visible(true);

    //             let color_map_ref = center.trunc() as usize;
    //             let (r, g, b) = calculate_color(
    //                 self.buckets_per_octave,
    //                 (color_map_ref as usize + self.buckets_per_octave
    //                     - 3 * (self.buckets_per_octave / 12)) as f32
    //                     % self.buckets_per_octave as f32,
    //             );

    //             let k_max = arg_max(
    //                 &self
    //                     .analysis_state
    //                     .peaks_continuous
    //                     .iter()
    //                     .map(|p| p.1)
    //                     .collect::<Vec<f32>>(),
    //             );
    //             let max_size = self.analysis_state.peaks_continuous[k_max].1;

    //             let color_coefficient = 1.0 - (1.0 - size / max_size).powf(2.0);

    //             c.set_color(
    //                 r * color_coefficient,
    //                 g * color_coefficient,
    //                 b * color_coefficient,
    //             );

    //             let radius = 0.08;
    //             c.set_local_scale(radius, *height, radius);
    //         }
    //     }
    // }
}

fn spiral_points(octaves: usize, buckets_per_octave: usize) -> Vec<(f32, f32, f32)> {
    (0..(buckets_per_octave * octaves))
        .map(|i| bin_to_spiral(buckets_per_octave, i as f32))
        .collect()
}

fn bin_to_spiral(buckets_per_octave: usize, x: f32) -> (f32, f32, f32) {
    let radius = 1.5 * (0.5 + (x / buckets_per_octave as f32).powf(0.75));
    let (transl_y, transl_x) = ((x + (buckets_per_octave - 3 * (buckets_per_octave / 12)) as f32)
        / (buckets_per_octave as f32)
        * 2.0
        * PI)
        .sin_cos();
    (transl_x * radius, transl_y * radius, 0.0) //17.0 - radius)
}
