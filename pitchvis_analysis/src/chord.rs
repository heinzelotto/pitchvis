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
/// * `min_notes` - Minimum number of notes to consider for chord detection (default: 2)
pub fn detect_chord(
    active_bins: &HashMap<usize, f32>,
    buckets_per_octave: u16,
    min_notes: usize,
) -> Option<DetectedChord> {
    if active_bins.len() < min_notes {
        return None;
    }

    // Convert bins to pitch classes (0-11) with aggregated power
    let mut pitch_classes: HashMap<usize, f32> = HashMap::new();
    for (&bin, &strength) in active_bins.iter() {
        let semitone = (bin * 12) / buckets_per_octave as usize;
        let pitch_class = semitone % 12;
        *pitch_classes.entry(pitch_class).or_insert(0.0) += strength;
    }

    // Need at least 2 distinct pitch classes
    if pitch_classes.len() < 2 {
        return None;
    }

    // Find the pitch class with the highest power (likely root)
    let mut pitch_classes_vec: Vec<(usize, f32)> = pitch_classes.iter()
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

            // Calculate confidence based on:
            // 1. How many notes match the expected pattern
            // 2. Root note power relative to other notes
            let total_power: f32 = pitch_classes_vec.iter().map(|(_, p)| p).sum();
            let root_ratio = root_power / total_power;
            let pattern_match = intervals.len() as f32 / expected_notes_for_quality(&quality) as f32;
            let confidence = (root_ratio * 0.5 + pattern_match * 0.5).min(1.0);

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
