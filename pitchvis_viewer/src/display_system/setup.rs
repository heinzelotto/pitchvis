use super::util::calculate_spiral_points;
use super::CylinderEntityListResource;
use super::{material::NoisyColorMaterial, LineList, PitchBall, PitchNameText, Spectrum};
use bevy::{
    core_pipeline::{
        bloom::{BloomCompositeMode, BloomPrefilterSettings, BloomSettings},
        tonemapping::Tonemapping,
    },
    math::vec3,
    prelude::*,
    sprite::MaterialMesh2dBundle,
};
use itertools::Itertools;
use nalgebra::{Rotation3, Vector3};
use std::f32::consts::PI;

use crate::display_system::BassCylinder;
use pitchvis_analysis::{
    color_mapping::{COLORS, PITCH_NAMES},
    vqt::VqtRange,
};

const HIGHEST_BASSNOTE: u16 = 12 * 2 + 4;
const SPIRAL_SEGMENTS_PER_SEMITONE: u16 = 6;

pub fn setup_display(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut color_materials: ResMut<Assets<ColorMaterial>>,
    mut noisy_color_materials: ResMut<Assets<NoisyColorMaterial>>,
    mut cylinder_entities: ResMut<CylinderEntityListResource>,
    range: &VqtRange,
) {
    assert!(range.buckets_per_octave % 12 == 0);

    spawn_pitch_balls(
        &mut commands,
        &mut meshes,
        &mut noisy_color_materials,
        range,
    );

    let visual_spiral_points =
        calculate_spiral_points(range.octaves, 12 * SPIRAL_SEGMENTS_PER_SEMITONE);

    spawn_bass_spiral(
        &mut commands,
        &mut cylinder_entities,
        &mut meshes,
        &mut color_materials,
        range,
        &visual_spiral_points,
    );

    spawn_spider_net(
        &mut commands,
        &mut meshes,
        &mut color_materials,
        range,
        &visual_spiral_points,
    );

    spawn_spectrum(&mut commands, &mut meshes, &mut color_materials, range);

    spawn_light(&mut commands);

    spawn_camera(&mut commands);

    spawn_text(&mut commands, range);
}

fn spawn_pitch_balls(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    noisy_color_materials: &mut ResMut<Assets<NoisyColorMaterial>>,
    range: &VqtRange,
) {
    let spiral_points = calculate_spiral_points(range.octaves, range.buckets_per_octave);

    for (idx, (x, y, _z)) in spiral_points.iter().enumerate() {
        // spheres
        let noisy_color_material = NoisyColorMaterial {
            color: Color::srgb(1.0, 0.7, 0.6).into(),
            params: Default::default(),
        };

        commands.spawn((
            PitchBall(idx),
            MaterialMesh2dBundle {
                mesh: meshes.add(Rectangle::new(20.0, 20.0)).into(),
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
}

fn spawn_bass_spiral(
    commands: &mut Commands,
    cylinder_entities: &mut ResMut<CylinderEntityListResource>,
    meshes: &mut ResMut<Assets<Mesh>>,
    color_materials: &mut ResMut<Assets<ColorMaterial>>,
    range: &VqtRange,
    bass_spiral_points: &[(f32, f32, f32)],
) {
    for (prev, cur) in bass_spiral_points
        .iter()
        .take((HIGHEST_BASSNOTE * range.buckets_per_octave * 2 / 12) as usize)
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
}

fn spawn_spider_net(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    color_materials: &mut ResMut<Assets<ColorMaterial>>,
    range: &VqtRange,
    bass_spiral_points: &[(f32, f32, f32)],
) {
    // draw rays
    let line_list: Vec<(Vec3, Vec3)> = (0..12)
        .map(|i| {
            let radius = range.octaves as f32 * 2.2; // 19.0 * 0.41421356237
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
                thickness: 0.05,
            }))
            .into(),
        material: color_materials.add(Color::srgb(0.3, 0.3, 0.3)),
        transform: Transform::from_xyz(0.0, 0.0, -13.0),
        ..default()
    });

    // draw spiral
    let spiral_mesh = LineList {
        lines: bass_spiral_points
            .iter()
            .map(|(x, y, z)| Vec3::new(*x, *y, *z))
            .tuple_windows()
            .collect::<Vec<(Vec3, Vec3)>>(),
        thickness: 0.05,
    };
    commands.spawn(MaterialMesh2dBundle {
        mesh: meshes.add(spiral_mesh).into(),
        material: color_materials.add(Color::srgb(0.3, 0.3, 0.3)),
        transform: Transform::from_xyz(0.0, 0.0, -13.0),
        ..default()
    });
}

fn spawn_spectrum(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    color_materials: &mut ResMut<Assets<ColorMaterial>>,
    range: &VqtRange,
) {
    // spectrum
    let spectrum_mesh = LineList {
        lines: (0..range.n_buckets())
            .map(|i| Vec3::new(i as f32 * 0.017, 0.0, 0.0))
            .tuple_windows()
            .collect::<Vec<(Vec3, Vec3)>>(),
        thickness: 0.01,
    };
    commands.spawn((
        Spectrum,
        MaterialMesh2dBundle {
            mesh: meshes.add(spectrum_mesh).into(),
            material: color_materials.add(Color::srgb(0.25, 0.85, 0.20)),
            transform: Transform::from_xyz(-12.0, 3.0, -13.0),
            ..default()
        },
    ));
}

fn spawn_light(commands: &mut Commands) {
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
}

fn spawn_camera(commands: &mut Commands) {
    // spawn a camera2dbundle with coordinates that match those of the 3d camera at the z=0 plane
    commands.spawn((
        Camera2dBundle {
            camera: Camera {
                // needed for bloom
                hdr: true,
                // renders after / on top of the main camera
                order: 1,
                clear_color: ClearColorConfig::Custom(Color::srgb(0.23, 0.23, 0.25)),
                ..default()
            },
            tonemapping: Tonemapping::SomewhatBoringDisplayTransform,
            projection: OrthographicProjection {
                scaling_mode: bevy::render::camera::ScalingMode::FixedVertical(38.0 * 0.414_213_57),
                scale: 1.00,
                ..default()
            },
            ..default()
        },
        BloomSettings {
            intensity: 0.0,
            low_frequency_boost: 1.0,
            low_frequency_boost_curvature: 1.0,
            high_pass_frequency: 0.52,
            prefilter_settings: BloomPrefilterSettings {
                threshold: 0.17,
                threshold_softness: 0.82,
            },
            composite_mode: BloomCompositeMode::Additive,
        },
    ));
}

fn spawn_text(commands: &mut Commands, range: &VqtRange) {
    let text_spiral_points = calculate_spiral_points(range.octaves, 12);
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
