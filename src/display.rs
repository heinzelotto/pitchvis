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

const PEAK_MIN_PROMINENCE: f32 = 15.0;
const PEAK_MIN_HEIGHT: f32 = 6.0;
const BASSLINE_PEAK_MIN_PROMINENCE: f32 = 8.0;
const BASSLINE_PEAK_MIN_HEIGHT: f32 = 4.0;

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
    (0.78, 0.32, 0.52), // F
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

fn calculate_color(buckets_per_octave: usize, bucket: usize) -> (f32, f32, f32) {
    let pitch_continuous = 12.0 * (bucket as f32) / (buckets_per_octave as f32);
    let base_color = COLORS[(pitch_continuous.round() as usize) % 12];
    let inaccuracy_cents = (pitch_continuous - pitch_continuous.round()).abs();

    let saturation = 1.0 - (2.0 * inaccuracy_cents) * (2.0 * inaccuracy_cents);
    const GRAY_LEVEL: f32 = 0.5;

    (
        saturation * base_color.0 + (1.0 - saturation) * GRAY_LEVEL,
        saturation * base_color.1 + (1.0 - saturation) * GRAY_LEVEL,
        saturation * base_color.2 + (1.0 - saturation) * GRAY_LEVEL,
    )
}

#[derive(Default)]
struct AnalysisState {
    history: Vec<Vec<f32>>,
    x_cqt_smoothed: Vec<f32>,
    x_cqt_afterglow: Vec<f32>,
    peaks: HashSet<usize>,
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

        for (prev, cur) in spiral_points.iter().tuple_windows() {
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

        let mut analysis_state = AnalysisState::default();
        analysis_state
            .x_cqt_afterglow
            .resize_with(octaves * buckets_per_octave, || 0.0);

        Self {
            cam,
            cubes,
            cylinders,
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

    fn handle_key_events(&mut self, window: &Window) {
        for event in window.events().iter() {
            if let kiss3d::event::WindowEvent::Key(c, a, _m) = event.value {
                match (c, a) {
                    (kiss3d::event::Key::P, kiss3d::event::Action::Release) => {
                        self.toggle_pause();
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn render(&mut self, window: &mut Window, x_cqt: &[f32]) {
        self.handle_key_events(window);

        self.preprocess(x_cqt);
        self.draw_spectrum(window);
        self.draw_spider_net(window);
        self.update_balls();
        self.update_cylinders();
    }

    fn preprocess(&mut self, x_cqt: &[f32]) {
        let num_buckets = self.octaves * self.buckets_per_octave;

        assert!(num_buckets == x_cqt.len());

        let k_min = arg_min(&x_cqt);
        let k_max = arg_max(&x_cqt);
        let _min = x_cqt[k_min];
        let _max = x_cqt[k_max];
        // println!("x_cqt[{k_min}] = {min}, x_cqt[{k_max}] = {max}");

        // find peaks
        let padding_length = 1;
        let mut x_cqt_padded_left = vec![0.0; padding_length];
        x_cqt_padded_left.extend(x_cqt.iter());
        let mut fp = PeakFinder::new(&x_cqt_padded_left);
        fp.with_min_prominence(PEAK_MIN_PROMINENCE);
        fp.with_min_height(PEAK_MIN_HEIGHT);
        //fp.with_min_difference(0.2);
        //fp.with_min_prominence(0.1);
        //fp.with_min_height(1.);
        let peaks = fp.find_peaks();
        let peaks = peaks
            .iter()
            .filter(|p| {
                p.middle_position() >= padding_length + (self.buckets_per_octave / 12 + 1) / 2
            }) // we disregard lowest A and surroundings as peaks
            .map(|p| p.middle_position() - padding_length)
            .collect::<HashSet<usize>>();

        // smooth by averaging over the history
        let mut x_cqt_smoothed = vec![0.0; num_buckets];
        // if a bin in the history was at peak magnitude at that time, it should be promoted
        self.analysis_state.history.push(
            x_cqt
                .iter()
                .enumerate()
                .map(|(i, x)| if peaks.contains(&i) { *x } else { *x / 8.0 })
                .collect::<Vec<f32>>(),
        );
        //self.history.push(x_cqt.iter().enumerate().map(|(i, x)| if peaks.contains(&i) {*x} else {0.0}).collect::<Vec<f32>>());
        let smooth_length = 5;
        if self.analysis_state.history.len() > smooth_length {
            // TODO: once fps is implemented, make this dependent on time instead of frames
            // make smoothing range modifiable in-game
            self.analysis_state.history.drain(0..1);
        }
        for i in 0..num_buckets {
            let mut v = vec![];
            for t in 0..self.analysis_state.history.len() {
                v.push(self.analysis_state.history[t][i]);
            }
            // Median
            // v.sort_by(|a, b| {
            //     if a == b {
            //         std::cmp::Ordering::Equal
            //     } else if a < b {
            //         std::cmp::Ordering::Less
            //     } else {
            //         std::cmp::Ordering::Greater
            //     }
            // });
            // x_cqt_smoothed[i] = v[self.history.len() / 2];

            // arithmetic mean
            x_cqt_smoothed[i] = v.iter().sum::<f32>() / smooth_length as f32;
        }

        if self.pause_state == PauseState::PauseRequested {
            self.pause_state = PauseState::Paused(x_cqt_smoothed.clone())
        }
        if let PauseState::Paused(v) = &self.pause_state {
            x_cqt_smoothed = v.clone();
        }

        self.analysis_state
            .x_cqt_afterglow
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| {
                *x *= 0.90 - 0.15 * (i as f32 / (self.octaves * self.buckets_per_octave) as f32);
                if *x < x_cqt_smoothed[i] {
                    *x = x_cqt_smoothed[i];
                }
            });

        // TEST unmodified
        //let x_cqt_smoothed = x_cqt;
        //let x_cqt_afterglow = x_cqt;

        self.analysis_state.peaks = peaks;
        self.analysis_state.x_cqt_smoothed = x_cqt_smoothed;
    }

    fn draw_spectrum(&mut self, window: &mut Window) {
        let x_cqt = &self.analysis_state.x_cqt_afterglow;

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
        for (i, c) in self.cubes.iter_mut().enumerate() {
            let x_cqt = &self.analysis_state.x_cqt_afterglow;

            //c.prepend_to_local_rotation(&rot);
            let (r, g, b) = calculate_color(
                self.buckets_per_octave,
                (i + self.buckets_per_octave - 3 * (self.buckets_per_octave / 12))
                    % self.buckets_per_octave,
            );

            let k_max = arg_max(x_cqt);
            let max = x_cqt[k_max];

            let color_coefficient = 1.0 - (1.0 - x_cqt[i] / max).powf(2.0);

            c.set_color(
                r * color_coefficient,
                g * color_coefficient,
                b * color_coefficient,
            );

            //c.set_local_scale((x_cqt[i] / 10.0).max(0.1), (x_cqt[i] / 10.0).max(0.1), (x_cqt[i] / 10.0).max(0.1));

            //let scale_factor = 1.0 / 30.0 * (0.7 + 0.3 * local_maximum as i32 as f32);

            //let scale_factor = 1.0;
            let scale_factor = 1.0 / 15.0;

            c.set_local_scale(
                x_cqt[i] * scale_factor,
                x_cqt[i] * scale_factor,
                x_cqt[i] * scale_factor,
            );
            //c.set_local_scale(1.0, 1.0, 1.0);
        }
    }

    fn update_cylinders(&mut self) {
        let x_cqt = &self.analysis_state.x_cqt_afterglow;

        // find peaks
        let mut fp = PeakFinder::new(x_cqt);
        fp.with_min_prominence(BASSLINE_PEAK_MIN_PROMINENCE);
        fp.with_min_height(BASSLINE_PEAK_MIN_HEIGHT);
        let peaks = fp.find_peaks();
        let mut peaks = peaks
            .iter()
            .map(|p| p.middle_position())
            .collect::<Vec<usize>>();

        peaks.sort_by(|a, b| {
            if a == b {
                std::cmp::Ordering::Equal
            } else if a < b {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        //let mut color_map: Vec<i32> = vec![-1; self.buckets_per_octave * self.octaves];
        // for (prev, cur) in peaks.iter().tuple_windows() {
        //     color_map[*prev..*cur].fill(*prev as i32);
        // }
        self.cylinders
            .iter_mut()
            .for_each(|c| c.0.set_visible(false));
        if let Some(first_peak) = peaks.first() {
            // color up to lowest note
            for idx in 0..*first_peak {
                let (ref mut c, ref height) = self.cylinders[idx];
                c.set_visible(true);

                let color_map_ref = *first_peak;
                let (r, g, b) = calculate_color(
                    self.buckets_per_octave,
                    (color_map_ref as usize + self.buckets_per_octave
                        - 3 * (self.buckets_per_octave / 12))
                        % self.buckets_per_octave,
                );

                let k_max = arg_max(x_cqt);
                let max = x_cqt[k_max];

                let color_coefficient = 1.0 - (1.0 - x_cqt[color_map_ref as usize] / max).powf(2.0);

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
            .map(|i| {
                //let radius = 1.8 * (0.5 + i as f32 / buckets_per_octave as f32);
                let radius = 1.5 * (0.5 + (i as f32 / buckets_per_octave as f32).powf(0.8));
                let (transl_y, transl_x) =
                    (((i + buckets_per_octave - 3 * (buckets_per_octave / 12)) as f32)
                        / (buckets_per_octave as f32)
                        * 2.0
                        * PI)
                        .sin_cos();
                (transl_x * radius, transl_y * radius, 0.0) //17.0 - radius)
            })
            .collect()
    }
}
