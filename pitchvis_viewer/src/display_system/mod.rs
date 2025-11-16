pub mod material;
mod setup;
mod update;
mod util;

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    post_process::bloom::Bloom,
    prelude::*,
};
use bevy_persistent::Persistent;
use material::NoisyColorMaterial;
use serde::{Deserialize, Serialize};

use crate::{
    analysis_system::AnalysisStateResource, app::SettingsState, vqt_system::VqtResultResource,
};
use pitchvis_analysis::vqt::VqtRange;

const CLEAR_COLOR_NEUTRAL: ClearColorConfig =
    ClearColorConfig::Custom(Color::srgb(0.23, 0.23, 0.25));
const CLEAR_COLOR_GALAXY: ClearColorConfig = ClearColorConfig::Custom(Color::srgb(0.05, 0.0, 0.05));
const _CLEAR_COLOR_EINK: ClearColorConfig = ClearColorConfig::Custom(Color::srgb(0.95, 0.95, 0.95));

#[derive(Component)]
pub struct PitchBall(usize);

#[derive(Component)]
pub struct BassCylinder;

#[derive(Component)]
pub struct SpiderNetSegment;

#[derive(Component)]
pub struct Spectrum;

#[derive(Component)]
pub struct PitchNameText;

#[derive(PartialEq, Serialize, Deserialize)]
pub enum DisplayMode {
    Normal,
    Debugging,
    // TODO: add a state with low display fidelity for weaker devices
}

// TODO: make use of state transitions
#[derive(States, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VisualsMode {
    Full,
    Zen,         // no pitch names
    Performance, // Faster and more precise (TODO: less smoothing)
    Galaxy,      // pitch balls only, black background
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VQTSmoothingMode {
    None,    // No additional smoothing (5ms)
    Short,   // Minor smoothing (40ms)
    Default, // Default smoothing (90ms)
    Long,    // Longer smoothing (250ms)
}

impl VQTSmoothingMode {
    pub fn to_duration(self) -> Option<std::time::Duration> {
        match self {
            VQTSmoothingMode::None => None,
            VQTSmoothingMode::Short => Some(std::time::Duration::from_millis(40)),
            VQTSmoothingMode::Default => Some(std::time::Duration::from_millis(90)),
            VQTSmoothingMode::Long => Some(std::time::Duration::from_millis(250)),
        }
    }
}

/// keep an index -> entity mapping for the cylinders
#[derive(Resource)]
pub struct CylinderEntityListResource(pub Vec<Entity>);

pub fn setup_display_to_system(
    range: &VqtRange,
) -> impl FnMut(
    Commands,
    ResMut<Assets<Mesh>>,
    ResMut<Assets<ColorMaterial>>,
    ResMut<Assets<NoisyColorMaterial>>,
    ResMut<CylinderEntityListResource>,
    Res<AssetServer>,
) {
    let range = range.clone();
    move |commands: Commands,
          meshes: ResMut<Assets<Mesh>>,
          color_materials: ResMut<Assets<ColorMaterial>>,
          noisy_color_materials: ResMut<Assets<NoisyColorMaterial>>,
          cylinder_entities: ResMut<CylinderEntityListResource>,
          asset_server: Res<AssetServer>| {
        setup::setup_display(
            commands,
            meshes,
            color_materials,
            noisy_color_materials,
            cylinder_entities,
            &range,
            asset_server,
        )
    }
}

// we pass all the ResMuts to only trigger bevy_ecs's change-detection selectively. This may give
// us some extra fps opposed to dereferencing all Mut's and passing the inner values, thus marking
// them changed.
#[allow(clippy::type_complexity)]
// TODO: refactor into plugin that owns the settings state, so we can make use of the bevy States transition functionality
pub fn update_display_to_system(
    range: &VqtRange,
) -> impl FnMut(
    ParamSet<(
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
    ResMut<Assets<ColorMaterial>>,
    ResMut<Assets<NoisyColorMaterial>>,
    ResMut<Assets<Mesh>>,
    Res<AnalysisStateResource>,
    Res<VqtResultResource>,
    Res<CylinderEntityListResource>,
    Res<Persistent<SettingsState>>,
    Res<Time>,
    Query<(&mut Camera, Option<&mut Bloom>, Ref<Projection>)>,
) -> Result<()> {
    let range = range.clone();
    move |set: ParamSet<(
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
          color_materials: ResMut<Assets<ColorMaterial>>,
          noisy_color_materials: ResMut<Assets<NoisyColorMaterial>>,
          meshes: ResMut<Assets<Mesh>>,
          analysis_state: Res<AnalysisStateResource>,
          vqt_result: Res<VqtResultResource>,
          cylinder_entities: Res<CylinderEntityListResource>,
          settings_state: Res<Persistent<SettingsState>>,
          run_time: Res<Time>,
          camera: Query<(&mut Camera, Option<&mut Bloom>, Ref<Projection>)>| {
        update::update_display(
            &range,
            set,
            color_materials,
            noisy_color_materials,
            meshes,
            analysis_state,
            vqt_result,
            cylinder_entities,
            settings_state,
            run_time,
            camera,
        )?;

        Ok(())
    }
}

/// A list of lines with a start and end position each
#[derive(Debug, Clone)]
struct LineList {
    /// Full width and height of the rectangle.
    pub lines: Vec<(Vec3, Vec3)>,
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
//     let PITCH_BALL_SCALE_FACTOR = 1.0 / 18.0;

//     for (i, c) in self.cubes.iter_mut().enumerate() {
//         if c.is_visible() {
//             let mut size = c.data().local_scale().x / PITCH_BALL_SCALE_FACTOR;
//             size *= 0.90 - 0.15 * (i as f32 / (self.octaves * self.buckets_per_octave) as f32);
//             c.set_local_scale(
//                 size * PITCH_BALL_SCALE_FACTOR,
//                 size * PITCH_BALL_SCALE_FACTOR,
//                 size * PITCH_BALL_SCALE_FACTOR,
//             );
//             if size * PITCH_BALL_SCALE_FACTOR < 0.4 {
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

//         //let PITCH_BALL_SCALE_FACTOR = 1.0 / 30.0 * (0.7 + 0.3 * local_maximum as i32 as f32);

//         //let PITCH_BALL_SCALE_FACTOR = 1.0;

//         c.set_local_scale(
//             size * PITCH_BALL_SCALE_FACTOR,
//             size * PITCH_BALL_SCALE_FACTOR,
//             size * PITCH_BALL_SCALE_FACTOR,
//         );

//         c.set_visible(true);
//     }
// }
