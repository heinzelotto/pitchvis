/// Wrapper for the chord_detector crate
/// This provides an alternative chord detection implementation
use chord_detector::ChordDetector;
use std::collections::HashMap;

/// Result from chord_detector library
#[derive(Debug, Clone)]
pub struct ChordDetectorResult {
    /// Chord name as detected by chord_detector
    pub chord_name: String,
    /// Root note name
    pub root_note: String,
    /// Chord kind/quality
    pub chord_kind: String,
    /// Confidence/quality score
    pub confidence: f32,
    /// Detected pitch classes (0-11)
    pub pitch_classes: Vec<usize>,
}

/// Detect chords using the chord_detector library
///
/// # Arguments
/// * `active_bins` - Map of bin index to strength/amplitude
/// * `buckets_per_octave` - Number of buckets per octave
/// * `min_freq` - The frequency of bin 0 in Hz
pub fn detect_chord_with_external_lib(
    active_bins: &HashMap<usize, f32>,
    buckets_per_octave: u16,
    min_freq: f32,
) -> Option<ChordDetectorResult> {
    if active_bins.is_empty() {
        return None;
    }

    // Convert bins to chromagram (12-tone pitch class histogram)
    let mut chroma = [0.0f32; 12];

    const C4_FREQ: f32 = 261.626;
    let semitones_from_c4 = 12.0 * (min_freq / C4_FREQ).log2();
    let bin_0_pitch_class = ((semitones_from_c4.round() as i32 % 12) + 12) % 12;

    for (&bin, &strength) in active_bins.iter() {
        // Use rounding to nearest semitone (same as chord.rs)
        let semitone = ((bin * 12) as f32 / buckets_per_octave as f32).round() as usize;
        let pitch_class = ((semitone as i32 + bin_0_pitch_class) % 12) as usize;

        chroma[pitch_class] += strength;
    }

    // Normalize chromagram
    let max_val = chroma.iter().cloned().fold(0.0f32, f32::max);
    if max_val > 0.0 {
        for val in &mut chroma {
            *val /= max_val;
        }
    }

    // Use ChordDetector to find the chord
    let mut detector = ChordDetector::default();
    let detected = detector.detect_chord(&chroma).ok()?;

    // Extract pitch classes that are active
    let pitch_classes: Vec<usize> = chroma
        .iter()
        .enumerate()
        .filter(|(_, &val)| val > 0.1) // Threshold for active pitch class
        .map(|(i, _)| i)
        .collect();

    Some(ChordDetectorResult {
        chord_name: format!("{:?}", detected),
        root_note: format!("{:?}", detected.root),
        chord_kind: format!("{:?}", detected.quality),
        confidence: detected.confidence,
        pitch_classes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chord_detector_c_major() {
        let mut active_bins = HashMap::new();
        // C major: C (bin 0), E (bin 28), G (bin 49)
        active_bins.insert(0, 1.0);
        active_bins.insert(28, 0.8);
        active_bins.insert(49, 0.9);

        let result = detect_chord_with_external_lib(&active_bins, 84, 261.626);
        assert!(result.is_some());

        let result = result.unwrap();
        println!("Detected chord: {}", result.chord_name);
        println!("Pitch classes: {:?}", result.pitch_classes);

        // Check that C, E, G are in the pitch classes
        assert!(result.pitch_classes.contains(&0)); // C
        assert!(result.pitch_classes.contains(&4)); // E
        assert!(result.pitch_classes.contains(&7)); // G
    }

    #[test]
    fn test_chord_detector_a_minor() {
        let mut active_bins = HashMap::new();
        // A minor: A (bin 0), C (bin 21), E (bin 49)
        // Starting from A (220 Hz)
        active_bins.insert(0, 1.0);
        active_bins.insert(21, 0.8);
        active_bins.insert(49, 0.9);

        let result = detect_chord_with_external_lib(&active_bins, 84, 220.0);
        assert!(result.is_some());

        let result = result.unwrap();
        println!("Detected chord: {}", result.chord_name);
        println!("Pitch classes: {:?}", result.pitch_classes);

        // A=9, C=0, E=4
        assert!(result.pitch_classes.contains(&9)); // A
        assert!(result.pitch_classes.contains(&0)); // C
        assert!(result.pitch_classes.contains(&4)); // E
    }

    #[test]
    fn test_chord_detector_detuned() {
        let mut active_bins = HashMap::new();
        // C major detuned by +43 cents
        active_bins.insert(3, 1.0);
        active_bins.insert(31, 0.8);
        active_bins.insert(52, 0.9);

        let result = detect_chord_with_external_lib(&active_bins, 84, 261.626);
        assert!(result.is_some());

        let result = result.unwrap();
        println!("Detected detuned chord: {}", result.chord_name);
        println!("Pitch classes: {:?}", result.pitch_classes);

        // Should still detect C, E, G (within tolerance)
        assert!(result.pitch_classes.contains(&0)); // C
        assert!(result.pitch_classes.contains(&4)); // E
        assert!(result.pitch_classes.contains(&7)); // G
    }
}
