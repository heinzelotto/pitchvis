use find_peaks::PeakFinder;
use itertools::Itertools;
use kiss3d::camera::ArcBall;
use kiss3d::light::Light;
use kiss3d::nalgebra;
use kiss3d::scene::SceneNode;
use kiss3d::window::Window;
use log::debug;
use nalgebra::{Point3, Rotation3, Vector3};
use std::collections::HashSet;
use std::f32::consts::PI;
use std::f32::EPSILON;
use std::rc::Rc;

const PEAK_MIN_PROMINENCE: f32 = 13.0;
const PEAK_MIN_HEIGHT: f32 = 6.0;
const _BASSLINE_PEAK_MIN_PROMINENCE: f32 = 12.0;
const _BASSLINE_PEAK_MIN_HEIGHT: f32 = 4.0;
const HIGHEST_BASSNOTE: usize = 12 * 2 + 4;
const SPECTROGRAM_LENGTH: usize = 400;
const SMOOTH_LENGTH: usize = 3;

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
    (0.37, 0.28, 0.50), // Eb
    (0.47, 0.77, 0.22), // E
    (0.78, 0.32, 0.52), // Fh
    (0.00, 0.64, 0.56), // F#
    (0.95, 0.54, 0.23), // G
    (0.26, 0.31, 0.53), // Ab
    (1.00, 0.96, 0.03), // A
    (0.57, 0.30, 0.55), // Bb
    (0.12, 0.71, 0.34), // H
];

fn arg_min(sl: &[f32]) -> usize {
    // we have no NaNs
    sl.iter()
        .enumerate()
        .fold(
            (0, f32::MAX),
            |cur, x| if *x.1 < cur.1 { (x.0, *x.1) } else { cur },
        )
        .0
}

fn arg_max(sl: &[f32]) -> usize {
    // we have no NaNs
    sl.iter()
        .enumerate()
        .fold(
            (0, f32::MIN),
            |cur, x| if *x.1 > cur.1 { (x.0, *x.1) } else { cur },
        )
        .0
}

fn calculate_color(buckets_per_octave: usize, bucket: f32) -> (f32, f32, f32) {
    let pitch_continuous = 12.0 * bucket / (buckets_per_octave as f32);
    let base_color = COLORS[(pitch_continuous.round() as usize) % 12];
    let inaccuracy_cents = (pitch_continuous - pitch_continuous.round()).abs();

    let saturation = 1.0 - (2.0 * inaccuracy_cents).powf(1.5);
    const GRAY_LEVEL: f32 = 0.5;

    (
        saturation * base_color.0 + (1.0 - saturation) * GRAY_LEVEL,
        saturation * base_color.1 + (1.0 - saturation) * GRAY_LEVEL,
        saturation * base_color.2 + (1.0 - saturation) * GRAY_LEVEL,
    )
}

struct AnalysisState {
    history: Vec<Vec<f32>>,
    x_cqt_smoothed: Vec<f32>,
    x_cqt_afterglow: Vec<f32>,
    peaks: HashSet<usize>,
    peaks_continuous: Vec<(f32, f32)>,

    spectrogram_buffer: Vec<u8>,
    spectrogram_front_idx: usize,
}

impl AnalysisState {
    fn new(w: usize, h: usize) -> Self {
        let spectrogram_buffer = vec![0; w * h * 4];

        Self {
            history: Vec::new(),
            x_cqt_smoothed: Vec::new(),
            x_cqt_afterglow: Vec::new(),
            peaks: HashSet::new(),
            peaks_continuous: Vec::new(),
            spectrogram_buffer,
            spectrogram_front_idx: 0,
        }
    }
}

#[derive(PartialEq)]
pub enum PauseState {
    Running,
    PauseRequested,
    Paused(Vec<f32>), // TODO: define a full display state including peaks and scaling, and store that
}

pub struct Display {
    pub cam: ArcBall,
    cubes: Vec<SceneNode>,
    /// Tuples of cylinder and fixed height because we reset scaling every frame
    cylinders: Vec<(SceneNode, f32)>,
    bg_quad: SceneNode,
    spectrogram_quad: SceneNode,
    spectrogram_tex: Rc<kiss3d::resource::Texture>,
    octaves: usize,
    buckets_per_octave: usize,
    analysis_state: AnalysisState,
    pause_state: PauseState,
    //x_cqt_smoothed: Vec<f32>,
}

impl Display {
    pub fn new(window: &mut Window, octaves: usize, buckets_per_octave: usize) -> Self {
        let cam = kiss3d::camera::ArcBall::new(Point3::new(0.0, 0.0, 19.0), Point3::origin());

        let mut cubes = vec![];
        let mut cylinders: Vec<(SceneNode, f32)> = vec![];

        let spiral_points = Self::spiral_points(octaves, buckets_per_octave);

        for (x, y, z) in spiral_points.iter() {
            let mut s = window.add_sphere(1.0);
            s.prepend_to_local_translation(&kiss3d::nalgebra::Translation::from([*x, *y, *z]));
            cubes.push(s);
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

            let mut c = window.add_cylinder(0.05, h + 0.01);
            c.prepend_to_local_translation(&kiss3d::nalgebra::Translation::from(mid));
            if let Some(rotation) = Rotation3::rotation_between(&y_unit, &v_diff) {
                c.prepend_to_local_rotation(&rotation.into());
            }
            cylinders.push((c, h + 0.01));
        }

        window.set_light(Light::StickToCamera);

        // let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.014);

        window.set_line_width(0.7);

        let mut analysis_state =
            AnalysisState::new(octaves * buckets_per_octave, SPECTROGRAM_LENGTH);
        analysis_state
            .x_cqt_afterglow
            .resize_with(octaves * buckets_per_octave, || 0.0);

        let mut bg_quad = window.add_quad(24.0, 12.0, 1, 1);
        let bg_small_jpg = include_bytes!("bg_europe.jpg");
        bg_quad.set_texture_from_memory(bg_small_jpg, "bg_small");
        bg_quad.prepend_to_local_translation(&kiss3d::nalgebra::Translation::from(
            nalgebra::vector![0.0, 0.0, -0.5],
        ));

        let spectrogram_tex = kiss3d::resource::TextureManager::get_global_manager(move |tm| {
            let img = image::DynamicImage::new_rgba8(
                (octaves * buckets_per_octave) as u32,
                SPECTROGRAM_LENGTH as u32,
            );
            tm.add_image(img, "spectrogram")
        });

        let mut spectrogram_quad = window.add_quad(7.0, 10.0, 1, 1);
        spectrogram_quad.set_texture(spectrogram_tex.clone());
        spectrogram_quad.prepend_to_local_translation(&kiss3d::nalgebra::Translation::from(
            nalgebra::vector![-10.0, -2.1, -0.01],
        ));

        Self {
            cam,
            cubes,
            cylinders,
            bg_quad,
            spectrogram_quad,
            spectrogram_tex,
            octaves,
            buckets_per_octave,
            analysis_state,
            pause_state: PauseState::Running,
            //x_cqt_smoothed : vec![0.0; octaves * buckets_per_octave],
        }
    }

    fn toggle_pause(&mut self) {
        self.pause_state = match self.pause_state {
            PauseState::Running => PauseState::PauseRequested,
            _ => PauseState::Running,
        };
        debug!("toggling pause");
    }
    fn toggle_bg_display(&mut self) {
        if self.bg_quad.is_visible() {
            self.bg_quad.set_visible(false);
        } else {
            self.bg_quad.set_visible(true);
        }
        debug!("toggling bg image display");
    }

    fn handle_key_events(&mut self, window: &Window) {
        for event in window.events().iter() {
            if let kiss3d::event::WindowEvent::Key(c, a, _m) = event.value {
                match (c, a) {
                    (kiss3d::event::Key::P, kiss3d::event::Action::Press) => {
                        self.toggle_pause();
                    }
                    (kiss3d::event::Key::I, kiss3d::event::Action::Press) => {
                        self.toggle_bg_display();
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn render(&mut self, window: &mut Window, x_cqt: &[f32], gain: f32) {
        self.handle_key_events(window);

        self.preprocess(x_cqt);
        self.update_spectrogram();
        self.draw_spectrum(window);
        self.draw_spider_net(window);
        self.update_balls();
        self.update_cylinders(gain);
    }

    fn preprocess(&mut self, x_cqt: &[f32]) {
        let num_buckets = self.octaves * self.buckets_per_octave;

        assert!(num_buckets == x_cqt.len());

        let k_min = arg_min(&x_cqt);
        let k_max = arg_max(&x_cqt);
        let _min = x_cqt[k_min];
        let _max = x_cqt[k_max];
        // println!("x_cqt[{k_min}] = {min}, x_cqt[{k_max}] = {max}");

        // smooth by averaging over the history
        let mut x_cqt_smoothed = vec![0.0; num_buckets];
        // if a bin in the history was at peak magnitude at that time, it should be promoted
        self.analysis_state.history.push(x_cqt.to_owned());
        //self.history.push(x_cqt.iter().enumerate().map(|(i, x)| if peaks.contains(&i) {*x} else {0.0}).collect::<Vec<f32>>());
        if self.analysis_state.history.len() > SMOOTH_LENGTH {
            // TODO: once fps is implemented, make this dependent on time instead of frames
            // make smoothing range modifiable in-game
            self.analysis_state.history.drain(0..1);
        }
        for i in 0..num_buckets {
            let mut v = vec![];
            for t in 0..self.analysis_state.history.len() {
                v.push(self.analysis_state.history[t][i]);
            }
            // arithmetic mean
            x_cqt_smoothed[i] = v.iter().sum::<f32>() / SMOOTH_LENGTH as f32;
        }

        // let conv_radius = (self.buckets_per_octave / 12) / 2;
        // let x_cqt_smoothed_convoluted = (0..(self.octaves * self.buckets_per_octave))
        //     .map(|idx| {
        //         if idx < conv_radius || idx >= self.octaves * self.buckets_per_octave - conv_radius
        //         {
        //             0.0
        //         } else {
        //             x_cqt_smoothed[(idx - conv_radius)..(idx + conv_radius + 1)]
        //                 .iter()
        //                 .sum::<f32>()
        //         }
        //     })
        //     .collect::<Vec<f32>>();

        // let mut pf2 = PeakFinder::new(&x_cqt_smoothed_convoluted);
        // pf2.with_min_prominence(PEAK_MIN_PROMINENCE + 5.0);
        // pf2.with_min_height(PEAK_MIN_HEIGHT + 5.0);
        // let peaks2 = pf2.find_peaks();
        // let peaks2 = peaks2
        //     .iter()
        //     .map(|p| p.middle_position())
        //     .collect::<HashSet<usize>>();

        // let mut x_cqt_smoothed = x_cqt_smoothed_convoluted
        //     .iter()
        //     .enumerate()
        //     .map(|(i, x)| {
        //         if peaks2.contains(&i) {
        //             *x / conv_radius as f32
        //         } else {
        //             x_cqt_smoothed[i] / 10.0
        //         }
        //     })
        //     .collect::<Vec<f32>>();

        // find peaks
        let padding_length = 1;
        let mut x_cqt_padded_left = vec![0.0; padding_length];
        x_cqt_padded_left.extend(x_cqt_smoothed.iter());
        let mut fp = PeakFinder::new(&x_cqt_padded_left);
        fp.with_min_prominence(PEAK_MIN_PROMINENCE);
        fp.with_min_height(PEAK_MIN_HEIGHT);
        let peaks = fp.find_peaks();
        let peaks = peaks
            .iter()
            .filter(|p| {
                p.middle_position() >= padding_length + (self.buckets_per_octave / 12 + 1) / 2
            }) // we disregard lowest A and surroundings as peaks
            .map(|p| p.middle_position() - padding_length)
            .collect::<HashSet<usize>>();

        let mut peaks_continuous = Vec::new();
        for p in &peaks {
            let p = *p;

            if p < 1 || p > self.octaves * self.buckets_per_octave - 2 {
                continue;
            }

            let x = x_cqt_smoothed[p] - x_cqt_smoothed[p - 1] + std::f32::EPSILON;
            let y = x_cqt_smoothed[p] - x_cqt_smoothed[p + 1] + std::f32::EPSILON;

            let estimated_precise_center = p as f32 + 1.0 / (1.0 + y / x) - 0.5;
            let estimated_precise_size = x_cqt_smoothed
                [estimated_precise_center.trunc() as usize + 1]
                * estimated_precise_center.fract()
                + x_cqt_smoothed[estimated_precise_center.trunc() as usize]
                    * (1.0 - estimated_precise_center.fract());
            peaks_continuous.push((estimated_precise_center, estimated_precise_size));
        }
        peaks_continuous.sort_by(|a, b| {
            if a.0 == b.0 {
                std::cmp::Ordering::Equal
            } else if a.0 < b.0 {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        if self.pause_state == PauseState::PauseRequested {
            self.pause_state = PauseState::Paused(x_cqt_smoothed.clone())
        }
        if let PauseState::Paused(v) = &self.pause_state {
            x_cqt_smoothed = v.clone();
        }

        // self.analysis_state
        //     .x_cqt_afterglow
        //     .iter_mut()
        //     .enumerate()
        //     .for_each(|(i, x)| {
        //         *x *= 0.85 - 0.15 * (i as f32 / (self.octaves * self.buckets_per_octave) as f32);
        //         if *x < x_cqt_smoothed[i] {
        //             *x = x_cqt_smoothed[i];
        //         }
        //     });

        // TEST unmodified
        //let x_cqt_smoothed = x_cqt.to_vec();

        self.analysis_state.peaks = peaks;
        self.analysis_state.x_cqt_smoothed = x_cqt_smoothed;
        self.analysis_state.peaks_continuous = peaks_continuous;
    }

    fn update_spectrogram(&mut self) {
        let k_max = arg_max(&self.analysis_state.x_cqt_smoothed);
        let max_size = self.analysis_state.x_cqt_smoothed[k_max];

        let width = self.octaves * self.buckets_per_octave;
        self.analysis_state.spectrogram_buffer[(self.analysis_state.spectrogram_front_idx
            * width
            * 4)
            ..((self.analysis_state.spectrogram_front_idx + 1) * width * 4)]
            .fill(0);
        //for (i, x) in self.analysis_state.x_cqt_smoothed.iter().enumerate() {
        for i in self.analysis_state.peaks.iter() {
            let x = self.analysis_state.x_cqt_smoothed[*i];
            let (r, g, b) = calculate_color(
                self.buckets_per_octave,
                (*i as f32 + (self.buckets_per_octave - 3 * (self.buckets_per_octave / 12)) as f32)
                    % self.buckets_per_octave as f32,
            );
            let brightness = x / (max_size + EPSILON);
            let brightness = ((1.0 - (1.0 - brightness).powf(2.0)) * 1.5 * 255.0).clamp(0.0, 255.0);

            // right to left
            self.analysis_state.spectrogram_buffer
                [((self.analysis_state.spectrogram_front_idx + 1) * width - i - 1) * 4 + 0] =
                (r * brightness * 1.2).clamp(0.0, 255.0) as u8;
            self.analysis_state.spectrogram_buffer
                [((self.analysis_state.spectrogram_front_idx + 1) * width - i - 1) * 4 + 1] =
                (g * brightness * 1.2).clamp(0.0, 255.0) as u8;
            self.analysis_state.spectrogram_buffer
                [((self.analysis_state.spectrogram_front_idx + 1) * width - i - 1) * 4 + 2] =
                (b * brightness * 1.2).clamp(0.0, 255.0) as u8;
            self.analysis_state.spectrogram_buffer
                [((self.analysis_state.spectrogram_front_idx + 1) * width - i - 1) * 4 + 3] = 1;

            if *i < width - 1 {
                self.analysis_state.spectrogram_buffer
                    [((self.analysis_state.spectrogram_front_idx + 1) * width - i - 2) * 4 + 0] =
                    (r * brightness * 1.2).clamp(0.0, 255.0) as u8;
                self.analysis_state.spectrogram_buffer
                    [((self.analysis_state.spectrogram_front_idx + 1) * width - i - 2) * 4 + 1] =
                    (g * brightness * 1.2).clamp(0.0, 255.0) as u8;
                self.analysis_state.spectrogram_buffer
                    [((self.analysis_state.spectrogram_front_idx + 1) * width - i - 2) * 4 + 2] =
                    (b * brightness * 1.2).clamp(0.0, 255.0) as u8;
                self.analysis_state.spectrogram_buffer
                    [((self.analysis_state.spectrogram_front_idx + 1) * width - i - 2) * 4 + 3] = 1;
            }
            if *i > 0 {
                self.analysis_state.spectrogram_buffer
                    [((self.analysis_state.spectrogram_front_idx + 1) * width - i) * 4 + 0] =
                    (r * brightness * 1.2).clamp(0.0, 255.0) as u8;
                self.analysis_state.spectrogram_buffer
                    [((self.analysis_state.spectrogram_front_idx + 1) * width - i) * 4 + 1] =
                    (g * brightness * 1.2).clamp(0.0, 255.0) as u8;
                self.analysis_state.spectrogram_buffer
                    [((self.analysis_state.spectrogram_front_idx + 1) * width - i) * 4 + 2] =
                    (b * brightness * 1.2).clamp(0.0, 255.0) as u8;
                self.analysis_state.spectrogram_buffer
                    [((self.analysis_state.spectrogram_front_idx + 1) * width - i) * 4 + 3] = 1;
            }
        }

        self.analysis_state.spectrogram_front_idx =
            (self.analysis_state.spectrogram_front_idx + 1) % SPECTROGRAM_LENGTH;
        self.analysis_state.spectrogram_buffer[(self.analysis_state.spectrogram_front_idx
            * width
            * 4)
            ..((self.analysis_state.spectrogram_front_idx + 1) * width * 4)]
            .fill(255);

        // clear also further ahead
        let further_idx = (self.analysis_state.spectrogram_front_idx + SPECTROGRAM_LENGTH / 10)
            % SPECTROGRAM_LENGTH;
        self.analysis_state.spectrogram_buffer
            [(further_idx * width * 4)..((further_idx + 1) * width * 4)]
            .fill(0);

        let context = kiss3d::context::Context::get();
        context.bind_texture(
            kiss3d::context::Context::TEXTURE_2D,
            Some(&self.spectrogram_tex),
        );
        context.tex_sub_image2d(
            kiss3d::context::Context::TEXTURE_2D,
            0,
            0,
            0,
            (self.octaves * self.buckets_per_octave) as i32,
            SPECTROGRAM_LENGTH as i32,
            kiss3d::context::Context::RGBA,
            Some(&self.analysis_state.spectrogram_buffer),
        );
    }

    fn draw_spectrum(&mut self, window: &mut Window) {
        let x_cqt = &self.analysis_state.x_cqt_smoothed;

        for i in 0..(self.buckets_per_octave * self.octaves - 1) {
            let x = i as f32 / (self.buckets_per_octave * self.octaves) as f32 * 7.0 - 13.5;
            let x_next =
                (i + 1) as f32 / (self.buckets_per_octave * self.octaves) as f32 * 7.0 - 13.5;
            let y_scale = 7.0;
            window.draw_line(
                &Point3::new(x, x_cqt[i] / y_scale + 3.0, 0.0),
                &Point3::new(x_next, x_cqt[i + 1] / y_scale + 3.0, 0.0),
                //&Point3::new(x, x_cqt_smoothed[i] /* / y_scale */ + 3.0, 0.0),
                //&Point3::new(x_next, x_cqt_smoothed[i + 1] /* / y_scale */ + 3.0, 0.0),
                &Point3::new(0.7, 0.9, 0.0),
            );
            if self.analysis_state.peaks.contains(&i) {
                window.draw_line(
                    //&Point3::new(x, x_cqt_smoothed[i] /*/ y_scale*/ + 3.0 - 0.1, 0.0),
                    //&Point3::new(x, x_cqt_smoothed[i] /*/ y_scale*/ + 3.0, 0.0),
                    &Point3::new(x, x_cqt[i] / y_scale + 3.0 + 0.2, 0.0),
                    &Point3::new(x, x_cqt[i] / y_scale + 3.0, 0.0),
                    &Point3::new(1.0, 0.2, 0.0),
                );
            }

            if i % (self.buckets_per_octave / 12) == 0 {
                window.draw_line(
                    &Point3::new(x, x_cqt[i] / y_scale + 3.0 - 0.1, 0.0),
                    &Point3::new(x, x_cqt[i] / y_scale + 3.0, 0.0),
                    // &Point3::new(x, x_cqt_smoothed[i] /*/ y_scale*/ + 3.0 - 0.1, 0.0),
                    // &Point3::new(x, x_cqt_smoothed[i] /*/ y_scale*/ + 3.0, 0.0),
                    &Point3::new(1.0, 0.0, 1.0),
                );
            }
        }
    }

    fn draw_spider_net(&mut self, window: &mut Window) {
        // draw rays
        for i in 0..12 {
            let radius = self.octaves as f32 * 2.2;
            let (p_y, p_x) = (i as f32 / 12.0 * 2.0 * PI).sin_cos();
            window.draw_line(
                &Point3::new(0.0, 0.0, 0.0),
                &Point3::new(radius * p_x, radius * p_y, 0.0),
                &Point3::new(0.20, 0.25, 0.20),
            );
        }

        // draw spiral
        // TODO: make these constant things constant
        let spiral_points = Self::spiral_points(self.octaves, self.buckets_per_octave);
        for (prev, cur) in spiral_points.iter().tuple_windows() {
            window.draw_line(
                &Point3::new(prev.0, prev.1, prev.2),
                &Point3::new(cur.0, cur.1, cur.2),
                &Point3::new(0.25, 0.20, 0.20),
            );
        }
    }

    fn update_balls(&mut self) {
        let scale_factor = 1.0 / 18.0;

        for (i, c) in self.cubes.iter_mut().enumerate() {
            if c.is_visible() {
                let mut size = c.data().local_scale().x / scale_factor;
                size *= 0.90 - 0.15 * (i as f32 / (self.octaves * self.buckets_per_octave) as f32);
                c.set_local_scale(
                    size * scale_factor,
                    size * scale_factor,
                    size * scale_factor,
                );
                if size * scale_factor < 0.4 {
                    c.set_visible(false);
                }
            }
        }

        if self.analysis_state.peaks_continuous.is_empty() {
            return;
        }

        let k_max = arg_max(
            &self
                .analysis_state
                .peaks_continuous
                .iter()
                .map(|p| p.1)
                .collect::<Vec<f32>>(),
        );
        let max_size = self.analysis_state.peaks_continuous[k_max].1;

        for p in &self.analysis_state.peaks_continuous {
            let (center, size) = *p;

            let (r, g, b) = calculate_color(
                self.buckets_per_octave,
                (center + (self.buckets_per_octave - 3 * (self.buckets_per_octave / 12)) as f32)
                    % self.buckets_per_octave as f32,
            );

            let color_coefficient = 1.0 - (1.0 - size / max_size).powf(2.0);

            let c = &mut self.cubes[center.trunc() as usize];
            let (x, y, z) = Self::bin_to_spiral(self.buckets_per_octave, center);
            c.set_local_translation(kiss3d::nalgebra::Translation::from([x, y, z]));
            c.set_color(
                r * color_coefficient,
                g * color_coefficient,
                b * color_coefficient,
            );

            //c.set_local_scale((x_cqt[i] / 10.0).max(0.1), (x_cqt[i] / 10.0).max(0.1), (x_cqt[i] / 10.0).max(0.1));

            //let scale_factor = 1.0 / 30.0 * (0.7 + 0.3 * local_maximum as i32 as f32);

            //let scale_factor = 1.0;

            c.set_local_scale(
                size * scale_factor,
                size * scale_factor,
                size * scale_factor,
            );

            c.set_visible(true);
        }
    }

    fn update_cylinders(&mut self, gain: f32) {
        //let mut color_map: Vec<i32> = vec![-1; self.buckets_per_octave * self.octaves];
        // for (prev, cur) in peaks.iter().tuple_windows() {
        //     color_map[*prev..*cur].fill(*prev as i32);
        // }
        self.cylinders
            .iter_mut()
            .for_each(|c| c.0.set_visible(false));
        if gain > 1000.0 {
            return;
        }
        if let Some((center, size)) = self.analysis_state.peaks_continuous.first() {
            if center.trunc() as usize >= self.cylinders.len() {
                return;
            }

            // color up to lowest note
            for idx in 0..(center.trunc() as usize) {
                let (ref mut c, ref height) = self.cylinders[idx];
                c.set_visible(true);

                let color_map_ref = center.trunc() as usize;
                let (r, g, b) = calculate_color(
                    self.buckets_per_octave,
                    (color_map_ref as usize + self.buckets_per_octave
                        - 3 * (self.buckets_per_octave / 12)) as f32
                        % self.buckets_per_octave as f32,
                );

                let k_max = arg_max(
                    &self
                        .analysis_state
                        .peaks_continuous
                        .iter()
                        .map(|p| p.1)
                        .collect::<Vec<f32>>(),
                );
                let max_size = self.analysis_state.peaks_continuous[k_max].1;

                let color_coefficient = 1.0 - (1.0 - size / max_size).powf(2.0);

                c.set_color(
                    r * color_coefficient,
                    g * color_coefficient,
                    b * color_coefficient,
                );

                let radius = 0.08;
                c.set_local_scale(radius, *height, radius);
            }
        }
    }

    fn spiral_points(octaves: usize, buckets_per_octave: usize) -> Vec<(f32, f32, f32)> {
        (0..(buckets_per_octave * octaves))
            .map(|i| Self::bin_to_spiral(buckets_per_octave, i as f32))
            .collect()
    }

    fn bin_to_spiral(buckets_per_octave: usize, x: f32) -> (f32, f32, f32) {
        //let radius = 1.8 * (0.5 + i as f32 / buckets_per_octave as f32);
        let radius = 1.5 * (0.5 + (x / buckets_per_octave as f32).powf(0.8));
        let (transl_y, transl_x) = ((x
            + (buckets_per_octave - 3 * (buckets_per_octave / 12)) as f32)
            / (buckets_per_octave as f32)
            * 2.0
            * PI)
            .sin_cos();
        (transl_x * radius, transl_y * radius, 0.0) //17.0 - radius)
    }
}
