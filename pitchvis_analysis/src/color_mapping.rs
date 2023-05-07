use lab::LCh;

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

const COLORS: [[f32; 3]; 12] = [
    [0.85, 0.36, 0.36], // C
    [0.01, 0.52, 0.71], // C#
    [0.97, 0.76, 0.05], // D
    //vec![0.37, 0.28, 0.50], // Eb
    [0.45, 0.34, 0.63], // Eb
    [0.47, 0.77, 0.22], // E
    [0.78, 0.32, 0.52], // Fh
    [0.00, 0.64, 0.56], // F#
    [0.95, 0.54, 0.23], // G
    //vec![0.26, 0.31, 0.53], // Ab
    [0.30, 0.37, 0.64], // Ab
    [1.00, 0.96, 0.03], // A
    [0.57, 0.30, 0.55], // Bb
    [0.12, 0.71, 0.34], // H
];

pub fn calculate_color(buckets_per_octave: usize, bucket: f32) -> (f32, f32, f32) {
    const GRAY_LEVEL: f32 = 60.0; // could be the mean lightness of the two neighbors. for now this is good enough.
    const EASING_POW: f32 = 1.3;

    let pitch_continuous = 12.0 * bucket / (buckets_per_octave as f32);
    let base_color =
        COLORS[(pitch_continuous.round() as usize) % 12].map(|rgb| (rgb * 255.0) as u8);
    let inaccuracy_cents = (pitch_continuous - pitch_continuous.round()).abs();

    let mut base_lcha = LCh::from_rgb(&base_color);
    let LCh {
        ref mut l,
        ref mut c,
        h: _,
    } = base_lcha;
    let saturation = 1.0 - (2.0 * inaccuracy_cents).powf(EASING_POW);
    *c *= saturation;
    *l = saturation * *l + (1.0 - saturation) * GRAY_LEVEL;

    let pitch_color = base_lcha.to_rgb().map(|rgb| (rgb as f32 / 255.0));
    (pitch_color[0], pitch_color[1], pitch_color[2])
}
