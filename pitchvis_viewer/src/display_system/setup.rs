use super::util::calculate_spiral_points;
use super::{material::NoisyColorMaterial, LineList, PitchBall, PitchNameText, Spectrum};
use super::{
    CylinderEntityListResource, GlissandoCurve, GlissandoCurveEntityListResource, RootNoteSlice,
    SpiderNetSegment, CLEAR_COLOR_NEUTRAL,
};
use bevy::camera::ScalingMode;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::post_process::bloom::{Bloom, BloomCompositeMode, BloomPrefilter};
use bevy::prelude::*;
use bevy::render::view::Hdr;
use itertools::Itertools;
use nalgebra::{Rotation3, Vector3};
use std::f32::consts::PI;

use crate::display_system::BassCylinder;
use pitchvis_analysis::vqt::VqtRange;
use pitchvis_colors::{COLORS, PITCH_NAMES};

const HIGHEST_BASSNOTE: u16 = 12 * 2 + 4;
const SPIRAL_SEGMENTS_PER_SEMITONE: u16 = 6;

pub fn setup_display(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut color_materials: ResMut<Assets<ColorMaterial>>,
    mut noisy_color_materials: ResMut<Assets<NoisyColorMaterial>>,
    mut cylinder_entities: ResMut<CylinderEntityListResource>,
    mut glissando_curve_entities: ResMut<GlissandoCurveEntityListResource>,
    range: &VqtRange,
    asset_server: Res<AssetServer>,
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

    spawn_glissando_curves(
        &mut commands,
        &mut glissando_curve_entities,
        &mut meshes,
        &mut color_materials,
    );

    spawn_light(&mut commands);

    spawn_camera(&mut commands);

    spawn_harmonic_lines(&mut commands, &mut meshes, &mut color_materials);

    spawn_chord_display(&mut commands, &asset_server);

    spawn_pitch_names_text(&mut commands, range, asset_server);

    spawn_root_note_slice(&mut commands, &mut meshes, &mut color_materials);
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
            Mesh2d(meshes.add(Rectangle::new(20.0, 20.0)).into()),
            MeshMaterial2d(noisy_color_materials.add(noisy_color_material)),
            Transform::from_xyz(*x * 1.0, *y * 1.0, -0.01), // needs to be slightly behind the 2d camera
            if idx % 17 == 0 {
                // 12 * 7 = 84 and 17 * 5 = 85, so we get a curved 5-star
                Visibility::Visible
            } else {
                Visibility::Hidden
            },
        ));
    }
}

fn spawn_bass_spiral(
    commands: &mut Commands,
    cylinder_entities: &mut ResMut<CylinderEntityListResource>,
    meshes: &mut ResMut<Assets<Mesh>>,
    color_materials: &mut ResMut<Assets<ColorMaterial>>,
    bass_spiral_points: &[(f32, f32, f32)],
) {
    for (prev, cur) in bass_spiral_points
        .iter()
        .take((HIGHEST_BASSNOTE * SPIRAL_SEGMENTS_PER_SEMITONE) as usize)
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
                    Mesh2d(meshes.add(Rectangle::new(0.05, h + 0.01)).into()),
                    MeshMaterial2d(color_materials.add(ColorMaterial {
                        color: Color::srgb(0.8, 0.7, 0.6),
                        alpha_mode: bevy::sprite_render::AlphaMode2d::Blend,
                        texture: None,
                        ..default()
                    })),
                    transform,
                    Visibility::Hidden,
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
    commands.spawn((
        SpiderNetSegment,
        Mesh2d(meshes.add(Mesh::from(LineList {
            lines: line_list,
            thickness: 0.05,
        }))),
        MeshMaterial2d(color_materials.add(ColorMaterial {
            color: Color::srgb(0.3, 0.3, 0.3),
            alpha_mode: bevy::sprite_render::AlphaMode2d::Blend,
            texture: None,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -13.0),
    ));

    // draw spiral
    let spiral_mesh = LineList {
        lines: bass_spiral_points
            .iter()
            .map(|(x, y, z)| Vec3::new(*x, *y, *z))
            .tuple_windows()
            .collect::<Vec<(Vec3, Vec3)>>(),
        thickness: 0.05,
    };
    commands.spawn((
        SpiderNetSegment,
        Mesh2d(meshes.add(spiral_mesh).into()),
        MeshMaterial2d(color_materials.add(Color::srgb(0.3, 0.3, 0.3))),
        Transform::from_xyz(0.0, 0.0, -13.0),
    ));
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
        Mesh2d(meshes.add(spectrum_mesh).into()),
        MeshMaterial2d(color_materials.add(ColorMaterial {
            color: Color::WHITE,
            alpha_mode: bevy::sprite_render::AlphaMode2d::Blend,
            texture: None,
            ..default()
        })),
        Transform::from_xyz(-12.0, 3.0, -13.0),
        Visibility::Hidden,
    ));
}

fn spawn_light(commands: &mut Commands) {
    commands.spawn((
        PointLight {
            intensity: 10500.0,
            // Shadows makes some Android devices segfault, this is under investigation
            // https://github.com/bevyengine/bevy/issues/8214
            #[cfg(not(target_os = "android"))]
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 9.0),
    ));
    commands.spawn((
        PointLight {
            intensity: 10500.0,
            //  Shadows makes some Android devices segfault, this is under investigation
            //https://github.com/bevyengine/bevy/issues/8214
            #[cfg(not(target_os = "android"))]
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, -29.0),
    ));
}

fn spawn_camera(commands: &mut Commands) {
    // spawn a camera2dbundle with coordinates that match those of the 3d camera at the z=0 plane
    commands.spawn((
        Camera2d,
        Hdr, // needed for bloom
        Camera {
            // renders after / on top of the main camera
            order: 1,
            clear_color: CLEAR_COLOR_NEUTRAL,
            ..default()
        },
        Tonemapping::SomewhatBoringDisplayTransform,
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: 38.0 * 0.414_213_57,
            },
            scale: 1.00,
            ..OrthographicProjection::default_2d()
        }),
        // TODO: make bloom removable based on display mode
        Bloom {
            intensity: 0.0,
            low_frequency_boost: 1.0,
            low_frequency_boost_curvature: 1.0,
            high_pass_frequency: 0.52,
            prefilter: BloomPrefilter {
                threshold: 0.17,
                threshold_softness: 0.82,
            },
            composite_mode: BloomCompositeMode::Additive,
            ..Default::default()
        },
        // MSAA makes some Android devices panic, this is under investigation
        // https://github.com/bevyengine/bevy/issues/8229
        #[cfg(target_os = "android")]
        Msaa::Off,
    ));
}

fn spawn_pitch_names_text(
    commands: &mut Commands,
    range: &VqtRange,
    asset_server: Res<AssetServer>,
) {
    let font = asset_server.load("fonts/DejaVuSans.ttf");
    let text_font = TextFont {
        font: font.clone(),
        font_size: 40.0,
        ..Default::default()
    };
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
            PitchNameText,
            Text2d::new(PITCH_NAMES[pitch_idx]),
            TextColor(Color::srgb(r, g, b)),
            text_font.clone(),
            TextLayout::new_with_justify(Justify::Center),
            Transform::from_xyz(x, y, -0.02).with_scale(vec3(0.02, 0.02, 1.0)),
            Visibility::Visible,
        ));
    }
}

fn spawn_harmonic_lines(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    color_materials: &mut ResMut<Assets<ColorMaterial>>,
) {
    // Spawn a single entity that will hold all harmonic lines
    // The mesh will be dynamically updated in the update system
    use super::{HarmonicLine, LineList};

    // Create an empty line mesh initially
    let line_list = LineList {
        lines: Vec::new(),
        thickness: 0.01,
    };
    let mesh: Mesh = line_list.into();

    commands.spawn((
        HarmonicLine,
        Mesh2d(meshes.add(mesh)),
        MeshMaterial2d(color_materials.add(ColorMaterial {
            color: Color::srgba(1.0, 1.0, 1.0, 0.15),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -0.015), // Behind pitch balls but in front of background
        Visibility::Hidden,                    // Hidden by default, shown when chords are detected
    ));
}

fn spawn_chord_display(commands: &mut Commands, asset_server: &Res<AssetServer>) {
    use super::ChordDisplay;

    let font = asset_server.load("fonts/DejaVuSans.ttf");
    let text_font = TextFont {
        font,
        font_size: 60.0,
        ..Default::default()
    };

    commands.spawn((
        ChordDisplay,
        Text2d::new(""),
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.8)),
        text_font,
        TextLayout::new_with_justify(Justify::Left),
        // Position in bottom left: x=-10.5 (left), y=-6.5 (bottom with clearance)
        Transform::from_xyz(-10.5, -6.5, 0.0).with_scale(vec3(0.02, 0.02, 1.0)),
        Visibility::Hidden,
    ));
}

fn spawn_glissando_curves(
    commands: &mut Commands,
    glissando_curve_entities: &mut ResMut<GlissandoCurveEntityListResource>,
    meshes: &mut ResMut<Assets<Mesh>>,
    color_materials: &mut ResMut<Assets<ColorMaterial>>,
) {
    // Create a pool of glissando curve entities (max 20 concurrent glissandos)
    const MAX_GLISSANDOS: usize = 20;

    for index in 0..MAX_GLISSANDOS {
        // Create an empty line mesh
        let mesh = meshes.add(LineList {
            lines: vec![],
            thickness: 0.05,
        });

        let material =
            color_materials.add(ColorMaterial::from_color(Color::srgba(1.0, 1.0, 1.0, 0.0)));

        let entity = commands
            .spawn((
                GlissandoCurve { index },
                Mesh2d(mesh),
                MeshMaterial2d(material),
                Transform::from_xyz(0.0, 0.0, -0.02), // Behind pitch balls (more negative = farther back)
                Visibility::Hidden,
            ))
            .id();

        glissando_curve_entities.0.push(entity);
    }
}

fn spawn_root_note_slice(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    color_materials: &mut ResMut<Assets<ColorMaterial>>,
) {
    // Create a pizza slice mesh pointing from center outward
    // The slice will be dynamically updated in the update system
    // Start with an empty mesh
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::{Indices, PrimitiveTopology};

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );

    // Empty mesh initially - will be filled in update_root_note_slice
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, Vec::<[f32; 2]>::new());
    mesh.insert_indices(Indices::U32(Vec::new()));

    commands.spawn((
        RootNoteSlice,
        Mesh2d(meshes.add(mesh)),
        MeshMaterial2d(color_materials.add(ColorMaterial {
            color: Color::srgba(1.0, 1.0, 1.0, 0.15),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -12.9), // Behind everything except background
        Visibility::Hidden,                   // Hidden by default
    ));
}
