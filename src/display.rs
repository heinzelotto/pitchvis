use kiss3d::camera::ArcBall;
use kiss3d::light::Light;
use kiss3d::nalgebra::Point3;
use kiss3d::window::Window;
use num_complex::Complex32;
use std::f32::consts::PI;

const COLORS: [(f32, f32, f32); 12] = [
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

pub struct Display {
    window: Window,
    cam: ArcBall,
    cubes: Vec<kiss3d::scene::SceneNode>,
    octaves: usize,
    buckets_per_octave: usize,
    history: Vec<Vec<f32>>,
    //x_cqt_smoothed: Vec<f32>,
}

impl Display {
    pub fn new(octaves: usize, buckets_per_octave: usize) -> Self {
        let mut window = Window::new("Kiss3d: cube");
        let cam = kiss3d::camera::ArcBall::new(Point3::new(0.0, 0.0, 19.0), Point3::origin());

        let mut cubes = vec![];
        for i in 0..(buckets_per_octave * octaves) {
            let mut c = window.add_sphere(1.0);
            let radius = 1.3 * (0.5 + i as f32 / buckets_per_octave as f32);
            let (transl_y, transl_x) = (((i + buckets_per_octave - 3 * (buckets_per_octave / 12))
                as f32)
                / (buckets_per_octave as f32)
                * 2.0
                * PI)
                .sin_cos();
            let transl_y = transl_y * radius;
            let transl_x = transl_x * radius;
            c.prepend_to_local_translation(&kiss3d::nalgebra::Translation::from([
                transl_x, transl_y, 0.0,
            ]));
            cubes.push(c);
        }

        window.set_light(Light::StickToCamera);

        // let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.014);

        window.set_line_width(0.7);

        Self {
            window,
            cam,
            cubes,
            octaves,
            buckets_per_octave,
            history: vec![],
            //x_cqt_smoothed : vec![0.0; octaves * buckets_per_octave],
        }
    }

    pub fn render(&mut self, x_cqt: &[f32]) -> bool {
        let num_buckets = self.octaves * self.buckets_per_octave;

        assert!(num_buckets == x_cqt.len());

        let k_min = arg_min(&x_cqt);
        let k_max = arg_max(&x_cqt);
        let min = x_cqt[k_min];
        let max = x_cqt[k_max];
        println!("x_cqt[{k_min}] = {min}, x_cqt[{k_max}] = {max}");

        // smooth by averaging over the history
        let mut x_cqt_smoothed = vec![0.0; num_buckets];
        self.history.push(x_cqt.to_vec());
        // for i in 0..self.history.len() {
        //     self.x_cqt_smoothed[i] += x_cqt[i] / 15.0;
        // }
        if self.history.len() > 3 {
            // for i in 0..self.history.len() {
            //     self.x_cqt_smoothed[i] -= (self.history.last().unwrap())[i] / 15.0;
            // }
            self.history.drain(0..1);
        }
        for i in 0..num_buckets {
            let mut v = vec![];
            for t in 0..self.history.len() {
                v.push(self.history[t][i]);
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
            x_cqt_smoothed[i] = v.iter().sum::<f32>() / 3.0;
        }

        // draw spectrum
        for i in 0..(self.buckets_per_octave * self.octaves - 1) {
            let x = i as f32 / (self.buckets_per_octave * self.octaves) as f32 * 7.0 - 13.5;
            let x_next =
                (i + 1) as f32 / (self.buckets_per_octave * self.octaves) as f32 * 7.0 - 13.5;
            self.window.draw_line(
                &Point3::new(x, x_cqt_smoothed[i] / 10.0 + 3.0, 0.0),
                &Point3::new(x_next, x_cqt_smoothed[i + 1] / 10.0 + 3.0, 0.0),
                &Point3::new(0.7, 0.9, 0.0),
            );
        }

        // draw pitch spider net
        for i in 0..12 {
            let radius = self.octaves as f32 * 2.2;
            let (p_y, p_x) = (i as f32 / 12.0 * 2.0 * PI).sin_cos();
            self.window.draw_line(
                &Point3::new(0.0, 0.0, 0.0),
                &Point3::new(radius * p_x, radius * p_y, 0.0),
                &Point3::new(0.05, 0.05, 0.05),
            );
        }

        for (i, c) in self.cubes.iter_mut().enumerate() {
            //c.prepend_to_local_rotation(&rot);
            let (r, g, b) = calculate_color(
                self.buckets_per_octave,
                (i + self.buckets_per_octave - 3 * (self.buckets_per_octave / 12))
                    % self.buckets_per_octave,
            );
            let local_maximum = (i <= 1
                || (x_cqt_smoothed[i] > x_cqt_smoothed[i - 1]
                    && x_cqt_smoothed[i] > x_cqt_smoothed[i - 2]))
                && (i >= num_buckets - 2
                    || (x_cqt_smoothed[i] > x_cqt_smoothed[i + 1]
                        && x_cqt_smoothed[i] > x_cqt_smoothed[i + 2]));
            let color_coefficient =
                (x_cqt_smoothed[i] / max).powf(if local_maximum { 0.5 } else { 1.5 });
            c.set_color(
                r * color_coefficient,
                g * color_coefficient,
                b * color_coefficient,
            );
            //c.set_local_scale((x_cqt[i] / 10.0).max(0.1), (x_cqt[i] / 10.0).max(0.1), (x_cqt[i] / 10.0).max(0.1));
            let scale_factor = 1.0 / 30.0 * (0.7 + 0.3 * local_maximum as i32 as f32);
            c.set_local_scale(
                x_cqt_smoothed[i] * scale_factor,
                x_cqt_smoothed[i] * scale_factor,
                x_cqt_smoothed[i] * scale_factor,
            );
            //c.set_local_scale(1.0, 1.0, 1.0);
        }

        return self.window.render_with_camera(&mut self.cam);
    }
}
