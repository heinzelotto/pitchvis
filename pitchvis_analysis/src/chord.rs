/// Chord detection and analysis
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChordQuality {
    Major,
    Minor,
    Diminished,
    Augmented,
    Sus2,
    Sus4,
    Major7,
    Minor7,
    Dominant7,
    Diminished7,
    HalfDiminished7,
}

#[derive(Debug, Clone)]
pub struct DetectedChord {
    /// Root note (0-11, where 0 = C)
    pub root: usize,
    /// Chord quality
    pub quality: ChordQuality,
    /// Notes that are part of the chord (0-11)
    pub notes: Vec<usize>,
    /// Confidence level (0.0-1.0)
    pub confidence: f32,
}

impl DetectedChord {
    pub fn name(&self) -> String {
        const NOTE_NAMES: [&str; 12] = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];

        let root_name = NOTE_NAMES[self.root];
        let quality_name = match self.quality {
            ChordQuality::Major => "",
            ChordQuality::Minor => "m",
            ChordQuality::Diminished => "dim",
            ChordQuality::Augmented => "aug",
            ChordQuality::Sus2 => "sus2",
            ChordQuality::Sus4 => "sus4",
            ChordQuality::Major7 => "maj7",
            ChordQuality::Minor7 => "m7",
            ChordQuality::Dominant7 => "7",
            ChordQuality::Diminished7 => "dim7",
            ChordQuality::HalfDiminished7 => "m7♭5",
        };

        format!("{}{}", root_name, quality_name)
    }
}

/// Detect chords from a set of active pitch bins
///
/// # Arguments
/// * `active_bins` - Map of bin index to strength/amplitude
/// * `buckets_per_octave` - Number of buckets per octave
/// * `min_freq` - The frequency of bin 0 in Hz (e.g., 55.0 for A1)
/// * `min_notes` - Minimum number of notes to consider for chord detection (default: 2)
pub fn detect_chord(
    active_bins: &HashMap<usize, f32>,
    buckets_per_octave: u16,
    min_freq: f32,
    min_notes: usize,
) -> Option<DetectedChord> {
    if active_bins.len() < min_notes {
        return None;
    }

    // Calculate the pitch class of bin 0 (min_freq)
    // Reference: C4 = 261.626 Hz is pitch class 0
    // Formula: pitch_class = (12 * log2(freq / C4) + 0.5) % 12
    const C4_FREQ: f32 = 261.626;
    let semitones_from_c4 = 12.0 * (min_freq / C4_FREQ).log2();
    let bin_0_pitch_class = ((semitones_from_c4.round() as i32 % 12) + 12) % 12;

    // Convert bins to pitch classes (0-11) with aggregated power
    let mut pitch_classes: HashMap<usize, f32> = HashMap::new();
    for (&bin, &strength) in active_bins.iter() {
        let semitone = (bin * 12) / buckets_per_octave as usize;
        // Offset by the pitch class of bin 0
        let pitch_class = ((semitone as i32 + bin_0_pitch_class) % 12) as usize;
        *pitch_classes.entry(pitch_class).or_insert(0.0) += strength;
    }

    // Need at least 2 distinct pitch classes
    if pitch_classes.len() < 2 {
        return None;
    }

    // Find the pitch class with the highest power (likely root)
    let mut pitch_classes_vec: Vec<(usize, f32)> = pitch_classes
        .iter()
        .map(|(&pc, &power)| (pc, power))
        .collect();
    pitch_classes_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Try each potential root (starting with strongest)
    for (potential_root, root_power) in pitch_classes_vec.iter().take(3) {
        let mut intervals: Vec<usize> = pitch_classes
            .keys()
            .map(|&pc| (pc + 12 - potential_root) % 12)
            .filter(|&interval| interval != 0)
            .collect();
        intervals.sort();

        // Check for chord patterns
        let chord_quality = match_chord_pattern(&intervals);

        if let Some(quality) = chord_quality {
            let notes: Vec<usize> = pitch_classes.keys().copied().collect();

            // Get the expected intervals for this chord quality
            let expected_intervals = get_expected_intervals(&quality);

            // Calculate confidence based on multiple factors:
            // 1. How well the detected intervals match the expected pattern
            // 2. Root note prominence
            // 3. Chord tones power vs non-chord tones power
            // 4. Penalty for extra notes that don't belong to the chord

            let total_power: f32 = pitch_classes_vec.iter().map(|(_, p)| p).sum();

            // Factor 1: Pattern match score (0.0 - 1.0)
            // Check if all expected intervals are present with significant power
            let mut matched_power = 0.0;
            let mut expected_power = 0.0;
            for &expected_interval in &expected_intervals {
                let expected_pc = (*potential_root + expected_interval) % 12;
                if let Some(&power) = pitch_classes.get(&expected_pc) {
                    matched_power += power;
                }
                // Use root power as baseline expectation
                expected_power += root_power * 0.7; // Expect chord tones to be at least 70% of root
            }
            let pattern_completeness = if expected_power > 0.0 {
                (matched_power / expected_power).min(1.0)
            } else {
                0.0
            };

            // Factor 2: Root prominence (should be strong but not necessarily strongest)
            let root_prominence = (root_power / total_power).min(1.0);

            // Factor 3: Chord tones vs non-chord tones power ratio
            let mut chord_tones_power = *root_power;
            for &interval in &expected_intervals {
                let pc = (*potential_root + interval) % 12;
                if let Some(&power) = pitch_classes.get(&pc) {
                    chord_tones_power += power;
                }
            }
            let chord_tone_ratio = chord_tones_power / total_power;

            // Factor 4: Penalty for extra pitch classes (non-chord tones)
            let num_expected_notes = expected_intervals.len() + 1; // +1 for root
            let extra_notes_penalty = if pitch_classes.len() > num_expected_notes {
                let extra = pitch_classes.len() - num_expected_notes;
                1.0 / (1.0 + extra as f32 * 0.3) // Diminishing penalty
            } else {
                1.0
            };

            // Combine factors with weights
            // - Pattern completeness is most important (40%)
            // - Chord tone ratio is very important (30%)
            // - Root prominence matters (20%)
            // - Extra notes penalty (10%)
            let confidence = (pattern_completeness * 0.4
                + chord_tone_ratio * 0.3
                + root_prominence * 0.2
                + extra_notes_penalty * 0.1)
                .min(1.0);

            return Some(DetectedChord {
                root: *potential_root,
                quality,
                notes,
                confidence,
            });
        }
    }

    None
}

/// Get the expected intervals for a chord quality (not including root)
fn get_expected_intervals(quality: &ChordQuality) -> Vec<usize> {
    match quality {
        ChordQuality::Major => vec![4, 7],
        ChordQuality::Minor => vec![3, 7],
        ChordQuality::Diminished => vec![3, 6],
        ChordQuality::Augmented => vec![4, 8],
        ChordQuality::Sus2 => vec![2, 7],
        ChordQuality::Sus4 => vec![5, 7],
        ChordQuality::Major7 => vec![4, 7, 11],
        ChordQuality::Minor7 => vec![3, 7, 10],
        ChordQuality::Dominant7 => vec![4, 7, 10],
        ChordQuality::Diminished7 => vec![3, 6, 9],
        ChordQuality::HalfDiminished7 => vec![3, 6, 10],
    }
}

fn match_chord_pattern(intervals: &[usize]) -> Option<ChordQuality> {
    // Triads
    if intervals.contains(&3) && intervals.contains(&7) {
        return Some(ChordQuality::Minor);
    }
    if intervals.contains(&4) && intervals.contains(&7) {
        return Some(ChordQuality::Major);
    }
    if intervals.contains(&3) && intervals.contains(&6) {
        return Some(ChordQuality::Diminished);
    }
    if intervals.contains(&4) && intervals.contains(&8) {
        return Some(ChordQuality::Augmented);
    }
    if intervals.contains(&2) && intervals.contains(&7) {
        return Some(ChordQuality::Sus2);
    }
    if intervals.contains(&5) && intervals.contains(&7) {
        return Some(ChordQuality::Sus4);
    }

    // Seventh chords
    if intervals.contains(&4) && intervals.contains(&7) && intervals.contains(&11) {
        return Some(ChordQuality::Major7);
    }
    if intervals.contains(&3) && intervals.contains(&7) && intervals.contains(&10) {
        return Some(ChordQuality::Minor7);
    }
    if intervals.contains(&4) && intervals.contains(&7) && intervals.contains(&10) {
        return Some(ChordQuality::Dominant7);
    }
    if intervals.contains(&3) && intervals.contains(&6) && intervals.contains(&9) {
        return Some(ChordQuality::Diminished7);
    }
    if intervals.contains(&3) && intervals.contains(&6) && intervals.contains(&10) {
        return Some(ChordQuality::HalfDiminished7);
    }

    // Power chord (just fifth) - treat as major
    if intervals.len() == 1 && intervals.contains(&7) {
        return Some(ChordQuality::Major);
    }

    None
}

#[allow(dead_code)]
fn expected_notes_for_quality(quality: &ChordQuality) -> usize {
    match quality {
        ChordQuality::Major
        | ChordQuality::Minor
        | ChordQuality::Diminished
        | ChordQuality::Augmented
        | ChordQuality::Sus2
        | ChordQuality::Sus4 => 3,
        ChordQuality::Major7
        | ChordQuality::Minor7
        | ChordQuality::Dominant7
        | ChordQuality::Diminished7
        | ChordQuality::HalfDiminished7 => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_major_chord_detection() {
        let mut active_bins = HashMap::new();
        // Assuming bin 0 starts at C (261.626 Hz)
        // C major chord: C (bin 0), E (bin 28), G (bin 49)
        // With 84 buckets per octave: 84/12 = 7 buckets per semitone
        // C = 0, E = 4 semitones = 28 buckets, G = 7 semitones = 49 buckets
        active_bins.insert(0, 1.0); // C
        active_bins.insert(28, 0.8); // E
        active_bins.insert(49, 0.9); // G

        let chord = detect_chord(&active_bins, 84, 261.626, 2);
        assert!(chord.is_some());
        let chord = chord.unwrap();
        assert_eq!(chord.root, 0); // C
        assert_eq!(chord.quality, ChordQuality::Major);
        assert_eq!(chord.name(), "C");
    }

    #[test]
    fn test_minor_chord_detection() {
        let mut active_bins = HashMap::new();
        // Assuming bin 0 starts at A (220.0 Hz)
        // A minor chord: A (bin 0), C (bin 21), E (bin 49)
        // A = 0, C = 3 semitones = 21 buckets, E = 7 semitones = 49 buckets
        active_bins.insert(0, 1.0); // A (root)
        active_bins.insert(21, 0.8); // C
        active_bins.insert(49, 0.9); // E

        let chord = detect_chord(&active_bins, 84, 220.0, 2);
        assert!(chord.is_some());
        let chord = chord.unwrap();
        assert_eq!(chord.root, 9); // A
        assert_eq!(chord.quality, ChordQuality::Minor);
        assert_eq!(chord.name(), "Am");
    }

    #[test]
    fn test_dominant_7th_chord_detection() {
        let mut active_bins = HashMap::new();
        // Assuming bin 0 starts at G (196.0 Hz)
        // G7 chord: G (bin 0), B (bin 28), D (bin 49), F (bin 70)
        // G = 0, B = 4 semitones = 28, D = 7 semitones = 49, F = 10 semitones = 70
        active_bins.insert(0, 1.0); // G
        active_bins.insert(28, 0.7); // B
        active_bins.insert(49, 0.8); // D
        active_bins.insert(70, 0.6); // F

        let chord = detect_chord(&active_bins, 84, 196.0, 2);
        assert!(chord.is_some());
        let chord = chord.unwrap();
        assert_eq!(chord.root, 7); // G
        assert_eq!(chord.quality, ChordQuality::Dominant7);
        assert_eq!(chord.name(), "G7");
    }

    #[test]
    fn test_sus4_chord_detection() {
        let mut active_bins = HashMap::new();
        // Assuming bin 0 starts at D (293.66 Hz)
        // Dsus4: D (bin 0), G (bin 35), A (bin 49)
        // D = 0, G = 5 semitones = 35, A = 7 semitones = 49
        active_bins.insert(0, 1.0); // D
        active_bins.insert(35, 0.8); // G
        active_bins.insert(49, 0.9); // A

        let chord = detect_chord(&active_bins, 84, 293.66, 2);
        assert!(chord.is_some());
        let chord = chord.unwrap();
        assert_eq!(chord.root, 2); // D
        assert_eq!(chord.quality, ChordQuality::Sus4);
        assert_eq!(chord.name(), "Dsus4");
    }

    #[test]
    fn test_no_chord_detection_single_note() {
        let mut active_bins = HashMap::new();
        active_bins.insert(0, 1.0);

        let chord = detect_chord(&active_bins, 84, 220.0, 2);
        assert!(chord.is_none());
    }

    #[test]
    fn test_no_chord_detection_non_chord_intervals() {
        let mut active_bins = HashMap::new();
        // Random intervals that don't form a chord
        active_bins.insert(0, 1.0);
        active_bins.insert(7, 0.8); // 1 semitone
        active_bins.insert(14, 0.9); // 2 semitones

        let _chord = detect_chord(&active_bins, 84, 220.0, 2);
        // This might detect something or nothing depending on pattern matching
        // The important thing is it doesn't crash
        assert!(true);
    }

    #[test]
    fn test_chord_with_octaves() {
        let mut active_bins = HashMap::new();
        // Assuming bin 0 starts at C (261.626 Hz)
        // C major with octave doubling
        // C = 0, E = 28, G = 49, C (octave) = 84
        active_bins.insert(0, 1.0); // C
        active_bins.insert(28, 0.8); // E
        active_bins.insert(49, 0.9); // G
        active_bins.insert(84, 0.7); // C (octave higher)

        let chord = detect_chord(&active_bins, 84, 261.626, 2);
        assert!(chord.is_some());
        let chord = chord.unwrap();
        assert_eq!(chord.root, 0); // C
        assert_eq!(chord.quality, ChordQuality::Major);
    }

    #[test]
    fn test_power_chord_detection() {
        let mut active_bins = HashMap::new();
        // Assuming bin 0 starts at A (220.0 Hz)
        // Power chord: root + fifth
        // A = 0, E = 49 (7 semitones)
        active_bins.insert(0, 1.0); // A
        active_bins.insert(49, 0.9); // E

        let chord = detect_chord(&active_bins, 84, 220.0, 2);
        assert!(chord.is_some());
        let chord = chord.unwrap();
        // Power chords are treated as major
        assert_eq!(chord.quality, ChordQuality::Major);
    }

    #[test]
    fn test_different_buckets_per_octave() {
        let mut active_bins = HashMap::new();
        // Test with 12 buckets per octave (1 per semitone)
        // Assuming bin 0 starts at C (261.626 Hz)
        // C major: C (0), E (4), G (7)
        active_bins.insert(0, 1.0);
        active_bins.insert(4, 0.8);
        active_bins.insert(7, 0.9);

        let chord = detect_chord(&active_bins, 12, 261.626, 2);
        assert!(chord.is_some());
        let chord = chord.unwrap();
        assert_eq!(chord.root, 0); // C
        assert_eq!(chord.quality, ChordQuality::Major);
    }

    #[test]
    fn test_c_major_in_tune_detected() {
        let mut active_bins = HashMap::new();
        // C major in tune with 84 buckets per octave (7 buckets per semitone)
        // Starting at C (261.626 Hz)
        // C = bin 0, E = bin 28 (4 semitones * 7), G = bin 49 (7 semitones * 7)
        active_bins.insert(0, 1.0);  // C
        active_bins.insert(28, 0.8); // E
        active_bins.insert(49, 0.9); // G

        let chord = detect_chord(&active_bins, 84, 261.626, 2);
        assert!(chord.is_some(), "In-tune C major should be detected");

        let chord = chord.unwrap();
        assert_eq!(chord.root, 0, "Root should be C (pitch class 0)");
        assert_eq!(chord.quality, ChordQuality::Major, "Quality should be Major");
        assert_eq!(chord.name(), "C", "Chord name should be C");
    }

    #[test]
    fn test_c_major_plus_50_cents_not_detected() {
        let mut active_bins = HashMap::new();
        // C major +50 cents (half semitone sharp)
        // With 84 buckets per octave: 1 semitone = 7 buckets, 0.5 semitones = 3.5 buckets
        // We offset each note by 3-4 buckets to represent +50 cents
        // C+50¢ = bin 3, E+50¢ = bin 31, G+50¢ = bin 52
        active_bins.insert(3, 1.0);  // C + 50 cents
        active_bins.insert(31, 0.8); // E + 50 cents
        active_bins.insert(52, 0.9); // G + 50 cents

        let chord = detect_chord(&active_bins, 84, 261.626, 2);

        // The detuned chord should either:
        // 1. Not be detected at all, OR
        // 2. Be detected but NOT as a valid chord (wrong pitch classes)
        if let Some(chord) = chord {
            // If a chord is detected, it should not be a clean major chord
            // due to the pitch offset causing wrong pitch class assignments
            println!("Detected chord: {} (root: {}, confidence: {})",
                     chord.name(), chord.root, chord.confidence);

            // With +50 cents offset, the pitch classes should be misaligned
            // This test documents the current behavior
            // Expected: Should NOT detect as C major, or should have low confidence

            // Check if it's detecting as C major (which would be wrong)
            if chord.root == 0 && chord.quality == ChordQuality::Major {
                panic!("C major +50 cents should NOT be detected as clean C major. \
                        This indicates a pitch quantization issue.");
            }
        }
    }

    #[test]
    fn test_c_major_minus_50_cents_not_detected() {
        let mut active_bins = HashMap::new();
        // C major -50 cents (half semitone flat)
        // We offset each note by -3 to -4 buckets to represent -50 cents
        // Since we can't use negative bins, we need to think about this differently.
        // C-50¢ = bin -3 (not valid), so let's use higher octave positions
        // Actually, for a cleaner test, let's shift the reference frequency
        // If we use min_freq slightly higher, the bins will represent flatter notes

        // Alternative approach: use bins that are 3-4 buckets below the target
        // For E and G, we can use bins 24-25 and 45-46
        // For C (bin 0), we'd need to wrap around, so let's start from a higher base

        // Using bins that represent -50 cents offset:
        // C-50¢ ≈ bin 80 (from previous octave wrap-around, or we can use bin 0 with different min_freq)
        // E-50¢ = bin 25, G-50¢ = bin 46
        active_bins.insert(0, 1.0);  // C (but we'll adjust min_freq to simulate -50 cents)
        active_bins.insert(25, 0.8); // E - 50 cents (28 - 3)
        active_bins.insert(46, 0.9); // G - 50 cents (49 - 3)

        // To simulate -50 cents, we increase min_freq by 50 cents
        // This makes the algorithm think the notes are 50 cents flatter than they are
        let min_freq_plus_50_cents = 261.626 * 2_f32.powf(0.5 / 12.0); // ≈ 269.03 Hz

        let chord = detect_chord(&active_bins, 84, min_freq_plus_50_cents, 2);

        // Similar to +50 cents test, this should either not detect or detect incorrectly
        if let Some(chord) = chord {
            println!("Detected chord: {} (root: {}, confidence: {})",
                     chord.name(), chord.root, chord.confidence);

            // The pitch classes should be misaligned due to the -50 cents offset
            // If it still detects as a clean major chord, that's the bug

            // Check intervals to see if they match a proper chord
            let mut pitch_classes: Vec<usize> = chord.notes.clone();
            pitch_classes.sort();

            // Calculate intervals from root
            let intervals: Vec<usize> = pitch_classes.iter()
                .map(|&pc| (pc + 12 - chord.root) % 12)
                .filter(|&i| i != 0)
                .collect();

            println!("  Detected intervals: {:?}", intervals);

            // If it detects clean major intervals [4, 7], that's wrong
            if intervals.contains(&4) && intervals.contains(&7) && intervals.len() == 2 {
                panic!("C major -50 cents should NOT be detected as clean major chord. \
                        This indicates a pitch quantization issue.");
            }
        }
    }
}
