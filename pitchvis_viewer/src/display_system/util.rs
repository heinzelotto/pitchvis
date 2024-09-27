use std::f32::consts::PI;

pub fn calculate_spiral_points(octaves: u8, buckets_per_octave: u16) -> Vec<(f32, f32, f32)> {
    (0..(buckets_per_octave * octaves as u16))
        .map(|i| bin_to_spiral(buckets_per_octave, i as f32))
        .collect()
}

pub fn bin_to_spiral(buckets_per_octave: u16, x: f32) -> (f32, f32, f32) {
    //let radius = 1.5 * (0.5 + (x / buckets_per_octave as f32).powf(0.75));
    let radius = 2.0 * (0.3 + (x / buckets_per_octave as f32).powf(0.75));
    #[allow(clippy::erasing_op)]
    let (transl_y, transl_x) = ((x + (buckets_per_octave - 0 * (buckets_per_octave / 12)) as f32)
        / buckets_per_octave as f32
        * 2.0
        * PI)
        .sin_cos();
    (-1.0 * transl_x * radius, transl_y * radius, 0.0) //17.0 - radius)
}

#[cfg(feature = "ml")]
pub fn vqt_bin_to_midi_pitch(buckets_per_octave: usize, bin: usize) -> Option<usize> {
    let midi_pitch = (bin as f32 / buckets_per_octave as f32 * 12.0).round() as usize
        + crate::FREQ_A1_MIDI_KEY_ID as usize;
    if midi_pitch > 127 {
        None
    } else {
        Some(midi_pitch)
    }
}
