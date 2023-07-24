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

pub const COLORS: [[f32; 3]; 12] = [
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
pub const GRAY_LEVEL: f32 = 60.0; // could be the mean lightness of the two neighbors. for now this is good enough.
pub const EASING_POW: f32 = 1.3;

/// Calculates a color based on a musical pitch bucket and a predefined color scale.
///
/// This function converts the pitch information into a continuous color representation based on
/// the closeness of the pitch to the central tones of a 12-tone scale. The colors for the 12-tone
/// scale are provided via the `colors` parameter.
///
/// # Parameters:
/// - `buckets_per_octave`: Number of pitch buckets in one octave.
/// - `bucket`: The specific pitch bucket for which the color is calculated.
/// - `colors`: A 12-element array representing the colors for each of the 12 musical pitches,
///             where each color is an RGB triplet of f32 values in [0, 1] range.
/// - `gray_level`: The luminance value for the gray color that the pitch color will interpolate
///                 towards if the pitch falls between two central tones.
/// - `easing_pow`: A power factor for the easing function used in color interpolation.
///
/// # Returns:
/// A tuple of three f32 values representing the RGB color of the given pitch bucket in the [0, 1] range.
///
/// # Notes:
/// - If the pitch falls exactly on one of the 12 tones, the output color will be exactly the
///   corresponding color from the `colors` array.
/// - As the pitch moves away from a central tone, the color will interpolate towards the gray
///   color defined by `gray_level` based on the `easing_pow` factor.
///
/// # Examples
///
/// ```
/// # use color_mappings::calculate_color;
/// let colors = [
///     [0.85, 0.36, 0.36], // Red for C
///     [0.01, 0.52, 0.71], // ... and so on for other pitches
///     // ... fill the other 10 colors
/// ];
/// let result = calculate_color(12, 1.0, colors, 0.5, 2.0);
/// assert_eq!(result, (0.01, 0.52, 0.71)); // Assuming the bucket matches the color for C# rexactly.
/// ```
pub fn calculate_color(
    buckets_per_octave: usize,
    bucket: f32,
    colors: [[f32; 3]; 12],
    gray_level: f32,
    easing_pow: f32,
) -> (f32, f32, f32) {
    let pitch_continuous = 12.0 * bucket / (buckets_per_octave as f32);
    let base_color =
        colors[(pitch_continuous.round() as usize) % 12].map(|rgb| (rgb * 255.0) as u8);
    let inaccuracy_cents = (pitch_continuous - pitch_continuous.round()).abs();

    let mut base_lcha = LCh::from_rgb(&base_color);
    let LCh {
        ref mut l,
        ref mut c,
        h: _,
    } = base_lcha;
    let saturation = 1.0 - (2.0 * inaccuracy_cents).powf(easing_pow);
    *c *= saturation;
    *l = saturation * *l + (1.0 - saturation) * gray_level;

    let pitch_color = base_lcha.to_rgb().map(|rgb| (rgb as f32 / 255.0));
    (pitch_color[0], pitch_color[1], pitch_color[2])
}
