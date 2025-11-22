use super::util::calculate_spiral_points;
use super::{material::NoisyColorMaterial, LineList, PitchBall, PitchNameText, Spectrum};
use super::{
    ChromaBox, CylinderEntityListResource, GlissandoCurve, GlissandoCurveEntityListResource,
    RootNoteSlice, SpectrogramDisplay, SpectrogramResource, SpiderNetSegment, CLEAR_COLOR_NEUTRAL,
};
use bevy::camera::ScalingMode;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::post_process::bloom::{Bloom, BloomCompositeMode, BloomPrefilter};
use bevy::prelude::*;
use bevy::render::view::Hdr;
use itertools::Itertools;
use nalgebra::{Rotation3, Vector3};
use std::f32::consts::PI;

use super::{BassCylinder, GLISSANDO_LINE_THICKNESS};
use pitchvis_analysis::vqt::VqtRange;
use pitchvis_colors::{COLORS, PITCH_NAMES};

const HIGHEST_BASSNOTE: u16 = 12 * 2 + 4;
const SPIRAL_SEGMENTS_PER_SEMITONE: u16 = 6;

pub fn setup_display(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut color_materials: ResMut<Assets<ColorMaterial>>,
    mut noisy_color_materials: ResMut<Assets<NoisyColorMaterial>>,
    mut images: ResMut<Assets<Image>>,
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

    spawn_spectrogram(&mut commands, &mut images, range);

    spawn_chroma_display(&mut commands);
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

        // We don't make all pitch balls visible initially. This will create a nice shape for the intro.
        // 12 * 7 = 84 and 17 * 5 = 85, so we get a curved 5-star
        let is_intro_ball = idx % 17 == 0;

        commands.spawn((
            PitchBall(idx),
            Mesh2d(meshes.add(Sphere::new(1.0).mesh().ico(3).unwrap())),
            MeshMaterial2d(noisy_color_materials.add(noisy_color_material)),
            Transform::from_xyz(*x, *y, 0.0).with_scale(Vec3::new(0.6, 0.6, 0.6)),
            if is_intro_ball {
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
    visual_spiral_points: &[(f32, f32, f32)],
) {
    cylinder_entities.0.clear();

    // create bass cylinders
    for (idx, (prev_point, next_point)) in visual_spiral_points.iter().tuple_windows().enumerate() {
        if idx >= HIGHEST_BASSNOTE as usize * SPIRAL_SEGMENTS_PER_SEMITONE as usize {
            break;
        }

        let radius = 0.12;

        let p = Vector3::new(prev_point.0, prev_point.1, prev_point.2);
        let q = Vector3::new(next_point.0, next_point.1, next_point.2);
        let cylinder_vec = q - p;
        let cylinder_height = cylinder_vec.norm();
        let rotation = Rotation3::face_towards(&cylinder_vec, &Vector3::z());

        let color_index = idx / SPIRAL_SEGMENTS_PER_SEMITONE as usize;
        let (r, g, b, a) = COLORS[color_index % 12];
        let color = Color::srgba(r, g, b, a);

        let m = Mesh3d(meshes.add(Cylinder {
            radius,
            half_height: cylinder_height / 2.0,
        }));

        let (axis, angle) = rotation.axis_angle().unwrap_or_default();

        let q_halfway = p + cylinder_vec / 2.0;

        let entity = commands
            .spawn((
                BassCylinder,
                m,
                MeshMaterial3d(color_materials.add(ColorMaterial {
                    color,
                    ..default()
                })),
                Transform::from_xyz(q_halfway.x, q_halfway.y, q_halfway.z)
                    .with_rotation(Quat::from_axis_angle(
                        Vec3::new(axis.x, axis.y, axis.z),
                        angle,
                    ) * Quat::from_rotation_x(PI / 2.0)),
                Visibility::Hidden,
            ))
            .id();

        cylinder_entities.0.push(entity);
    }
}

fn spawn_spider_net(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    color_materials: &mut ResMut<Assets<ColorMaterial>>,
    range: &VqtRange,
    visual_spiral_points: &[(f32, f32, f32)],
) {
    // draw rays
    for i in 0..12 {
        let radius = range.octaves as f32 * 2.2;
        let (p_y, p_x) = (i as f32 / 12.0 * 2.0 * PI).sin_cos();

        let line_list = LineList {
            lines: vec![
                (Vec3::new(0.0, 0.0, 0.0), Vec3::new(radius * p_x, radius * p_y, 0.0)),
            ],
            thickness: 0.015,
        };

        commands.spawn((
            SpiderNetSegment,
            Mesh2d(meshes.add(Mesh::from(line_list))),
            MeshMaterial2d(color_materials.add(ColorMaterial {
                color: Color::srgb(0.20, 0.25, 0.20),
                ..default()
            })),
            Transform::from_xyz(0.0, 0.0, -12.5),
            Visibility::Hidden,
        ));
    }

    // draw spiral
    for (prev, cur) in visual_spiral_points.iter().tuple_windows() {
        let line_list = LineList {
            lines: vec![
                (Vec3::new(prev.0, prev.1, prev.2), Vec3::new(cur.0, cur.1, cur.2)),
            ],
            thickness: 0.015,
        };

        commands.spawn((
            SpiderNetSegment,
            Mesh2d(meshes.add(Mesh::from(line_list))),
            MeshMaterial2d(color_materials.add(ColorMaterial {
                color: Color::srgb(0.25, 0.20, 0.20),
                ..default()
            })),
            Transform::from_xyz(0.0, 0.0, -12.5),
            Visibility::Hidden,
        ));
    }
}

fn spawn_spectrum(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    color_materials: &mut ResMut<Assets<ColorMaterial>>,
    range: &VqtRange,
) {
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::{Indices, PrimitiveTopology};

    let mut mesh = Mesh::new(
        PrimitiveTopology::LineStrip,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![[0.0, 0.0, 0.0]; range.n_buckets()],
    );
    mesh.insert_indices(Indices::U32((0..range.n_buckets() as u32).collect()));

    commands.spawn((
        Spectrum,
        Visibility::Hidden,
        Mesh2d(meshes.add(mesh)),
        MeshMaterial2d(color_materials.add(ColorMaterial {
            color: Color::srgb(0.7, 0.9, 0.0),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
}

fn spawn_glissando_curves(
    commands: &mut Commands,
    glissando_curve_entities: &mut ResMut<GlissandoCurveEntityListResource>,
    meshes: &mut ResMut<Assets<Mesh>>,
    color_materials: &mut ResMut<Assets<ColorMaterial>>,
) {
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::{Indices, PrimitiveTopology};

    const POOL_SIZE: usize = 10;

    glissando_curve_entities.0.clear();

    for i in 0..POOL_SIZE {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new());
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, Vec::<[f32; 2]>::new());
        mesh.insert_indices(Indices::U32(Vec::new()));

        let entity = commands
            .spawn((
                GlissandoCurve { index: i },
                Mesh2d(meshes.add(mesh)),
                MeshMaterial2d(color_materials.add(ColorMaterial {
                    color: Color::srgba(1.0, 1.0, 1.0, 0.6),
                    ..default()
                })),
                Transform::from_xyz(0.0, 0.0, -0.5),
                Visibility::Hidden,
            ))
            .id();

        glissando_curve_entities.0.push(entity);
    }
}

fn spawn_light(commands: &mut Commands) {
    commands.spawn((
        PointLight {
            color: Color::WHITE,
            intensity: 250_000.0,
            range: 50.0,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 5.0),
    ));
}

fn spawn_camera(commands: &mut Commands) {
    let mut camera = Camera2d::default();
    camera.clear_color = CLEAR_COLOR_NEUTRAL;

    commands.spawn((
        camera,
        Camera {
            hdr: true,
            ..default()
        },
        Tonemapping::None,
        Bloom {
            intensity: 0.01,
            low_frequency_boost: 0.7,
            low_frequency_boost_curvature: 0.95,
            high_pass_frequency: 1.0,
            prefilter: BloomPrefilter {
                threshold: 0.0,
                threshold_softness: 0.0,
            },
            composite_mode: BloomCompositeMode::Additive,
        },
        Hdr,
        Projection::Orthographic(OrthographicProjection {
            near: -1000.0,
            far: 1000.0,
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: 25.0,
            },
            ..OrthographicProjection::default_2d()
        }),
    ));
}

fn spawn_harmonic_lines(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    color_materials: &mut ResMut<Assets<ColorMaterial>>,
) {
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::{Indices, PrimitiveTopology};

    const NUM_HARMONIC_LINES: usize = 20;

    for _ in 0..NUM_HARMONIC_LINES {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new());
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, Vec::<[f32; 2]>::new());
        mesh.insert_indices(Indices::U32(Vec::new()));

        commands.spawn((
            HarmonicLine,
            Mesh2d(meshes.add(mesh)),
            MeshMaterial2d(color_materials.add(ColorMaterial {
                color: Color::srgba(1.0, 1.0, 1.0, 0.15),
                ..default()
            })),
            Transform::from_xyz(0.0, 0.0, -11.0),
            Visibility::Hidden,
        ));
    }
}

fn spawn_chord_display(commands: &mut Commands, asset_server: &Res<AssetServer>) {
    let font = asset_server.load("fonts/DejaVuSans.ttf");

    commands.spawn((
        ChordDisplay,
        Text2d::new(""),
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.9)),
        TextFont {
            font,
            font_size: 48.0,
            ..default()
        },
        Transform::from_xyz(0.0, 10.0, 13.0),
        Visibility::Hidden,
    ));
}

fn spawn_pitch_names_text(
    commands: &mut Commands,
    range: &VqtRange,
    asset_server: Res<AssetServer>,
) {
    let font = asset_server.load("fonts/DejaVuSans.ttf");
    let spiral_points = calculate_spiral_points(range.octaves, range.buckets_per_octave);

    for (idx, (x, y, z)) in spiral_points.iter().enumerate() {
        // We only show the pitch names for notes that are on the chromatic scale
        if idx % (range.buckets_per_octave / 12) != 0 {
            continue;
        }

        let pitch_class = (idx / (range.buckets_per_octave / 12)) % 12;

        let text_distance = 0.7;
        let text_x = x * (1.0 + text_distance);
        let text_y = y * (1.0 + text_distance);

        commands.spawn((
            PitchNameText,
            Text2d::new(PITCH_NAMES[pitch_class]),
            TextColor(Color::srgba(1.0, 1.0, 1.0, 0.7)),
            TextFont {
                font: font.clone(),
                font_size: 24.0,
                ..default()
            },
            TextLayout::new_with_justify(JustifyText::Center),
            Transform::from_xyz(text_x, text_y, *z + 0.1),
            Visibility::Hidden,
        ));
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

fn spawn_spectrogram(
    commands: &mut Commands,
    images: &mut ResMut<Assets<Image>>,
    range: &VqtRange,
) {
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

    // Spectrogram dimensions: width = VQT bins, height = time frames
    const SPECTROGRAM_HEIGHT: usize = 200;
    let width = range.n_buckets();
    let height = SPECTROGRAM_HEIGHT;

    // Create RGBA8 image (4 bytes per pixel)
    let mut image_data = vec![0u8; width * height * 4];

    // Initialize with transparent black
    for pixel in image_data.chunks_exact_mut(4) {
        pixel[0] = 0; // R
        pixel[1] = 0; // G
        pixel[2] = 0; // B
        pixel[3] = 0; // A (transparent)
    }

    let image = Image {
        data: image_data,
        texture_descriptor: bevy::render::render_resource::TextureDescriptor {
            label: Some("spectrogram"),
            size: Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: bevy::render::render_resource::TextureUsages::TEXTURE_BINDING
                | bevy::render::render_resource::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        sampler: bevy::render::render_resource::SamplerDescriptor {
            mag_filter: bevy::render::render_resource::FilterMode::Nearest,
            min_filter: bevy::render::render_resource::FilterMode::Nearest,
            ..default()
        },
        ..default()
    };

    let image_handle = images.add(image);

    // Insert the resource
    commands.insert_resource(SpectrogramResource {
        image_handle: image_handle.clone(),
        write_index: 0,
        height,
    });

    // Spawn the sprite
    // Position at top-middle: x=0, y=10 (adjust as needed)
    // Scale to make it visible
    let display_width = 12.0;
    let display_height = display_width * (height as f32 / width as f32);

    commands.spawn((
        SpectrogramDisplay,
        Sprite {
            image: image_handle,
            custom_size: Some(Vec2::new(display_width, display_height)),
            ..default()
        },
        Transform::from_xyz(0.0, 10.0, 10.0),
        Visibility::Hidden, // Hidden by default, shown in debug mode
    ));
}

fn spawn_chroma_display(commands: &mut Commands) {
    // Create 12 boxes for the 12 pitch classes
    // Position them horizontally at the bottom
    const BOX_SIZE: f32 = 0.6;
    const BOX_SPACING: f32 = 0.7;
    const TOTAL_WIDTH: f32 = BOX_SPACING * 12.0;
    const Y_POSITION: f32 = -10.0;
    const Z_POSITION: f32 = 10.0;

    for pitch_class in 0..12 {
        let x_position = -TOTAL_WIDTH / 2.0 + pitch_class as f32 * BOX_SPACING + BOX_SPACING / 2.0;

        let (r, g, b, _) = COLORS[pitch_class];

        commands.spawn((
            ChromaBox { pitch_class },
            Node {
                position_type: PositionType::Absolute,
                width: Val::Px(40.0),
                height: Val::Px(40.0),
                left: Val::Px(400.0 + pitch_class as f32 * 45.0), // Adjust position as needed
                bottom: Val::Px(10.0),
                ..default()
            },
            BackgroundColor(Color::srgba(r, g, b, 0.0)), // Start transparent
            BorderColor::all(Color::srgba(r, g, b, 0.5)),
            BorderRadius::all(Val::Px(4.0)),
            ZIndex(i32::MAX),
            Visibility::Hidden, // Hidden by default, shown in debug mode
        ));
    }
}
