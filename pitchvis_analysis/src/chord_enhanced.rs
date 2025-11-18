use std::collections::HashMap;
use rust_music_theory::chord::{Chord as RmtChord, Number as ChordNumber, Quality as ChordQuality};
use rust_music_theory::note::{Notes, PitchClass};

use crate::chord::{ChordQuality as OldChordQuality, DetectedChord};

/// Convert a pitch class index (0-11, C=0) to rust-music-theory PitchClass
fn index_to_pitch_class(index: usize) -> PitchClass {
    match index % 12 {
        0 => PitchClass::C,
        1 => PitchClass::Cs,
        2 => PitchClass::D,
        3 => PitchClass::Ds,
        4 => PitchClass::E,
        5 => PitchClass::F,
        6 => PitchClass::Fs,
        7 => PitchClass::G,
        8 => PitchClass::Gs,
        9 => PitchClass::A,
        10 => PitchClass::As,
        11 => PitchClass::B,
        _ => unreachable!(),
    }
}

/// Convert rust-music-theory ChordQuality to our ChordQuality
fn rmt_quality_to_old_quality(quality: &ChordQuality, number: &ChordNumber) -> OldChordQuality {
    match (quality, number) {
        (ChordQuality::Major, ChordNumber::Triad) => OldChordQuality::Major,
        (ChordQuality::Minor, ChordNumber::Triad) => OldChordQuality::Minor,
        (ChordQuality::Augmented, ChordNumber::Triad) => OldChordQuality::Augmented,
        (ChordQuality::Diminished, ChordNumber::Triad) => OldChordQuality::Diminished,
        (ChordQuality::Suspended2, ChordNumber::Triad) => OldChordQuality::Sus2,
        (ChordQuality::Suspended4, ChordNumber::Triad) => OldChordQuality::Sus4,
        (ChordQuality::Major, ChordNumber::Seventh) => OldChordQuality::Major7,
        (ChordQuality::Minor, ChordNumber::Seventh) => OldChordQuality::Minor7,
        (ChordQuality::Dominant, ChordNumber::Seventh) => OldChordQuality::Dominant7,
        (ChordQuality::Diminished, ChordNumber::Seventh) => OldChordQuality::Diminished7,
        (ChordQuality::HalfDiminished, ChordNumber::Seventh) => OldChordQuality::HalfDiminished7,
        _ => OldChordQuality::Major, // Default fallback
    }
}

/// Convert PitchClass back to index
fn pitch_class_to_index(pc: &PitchClass) -> usize {
    match pc {
        PitchClass::C => 0,
        PitchClass::Cs => 1,
        PitchClass::D => 2,
        PitchClass::Ds => 3,
        PitchClass::E => 4,
        PitchClass::F => 5,
        PitchClass::Fs => 6,
        PitchClass::G => 7,
        PitchClass::Gs => 8,
        PitchClass::A => 9,
        PitchClass::As => 10,
        PitchClass::B => 11,
    }
}

#[derive(Debug, Clone)]
struct ChordCandidate {
    root: usize,
    quality: ChordQuality,
    number: ChordNumber,
    matched_notes: usize,
    total_chord_notes: usize,
    extra_notes: usize,
    root_strength: f32,
}

/// Enhanced chord detection using rust-music-theory library
///
/// This function tests all possible chord combinations and finds the best match
/// based on how many detected notes match the chord.
pub fn detect_chord_enhanced(
    active_bins: &HashMap<usize, f32>,
    buckets_per_octave: usize,
    min_notes: usize,
) -> Option<DetectedChord> {
    if active_bins.is_empty() {
        return None;
    }

    // Convert VQT bins to pitch classes (0-11) with their summed amplitudes
    // NOTE: VQT values are in dB (logarithmic scale), so we must convert to linear power
    // before summing across octaves for the same pitch class
    let mut pitch_class_power: HashMap<usize, f32> = HashMap::new();
    for (bin_idx, amplitude_db) in active_bins {
        let pitch_class = (bin_idx % buckets_per_octave) * 12 / buckets_per_octave;
        // Convert from A-based (A=0) to C-based (C=0)
        // Since C is 3 semitones above A, to make C=0 we subtract 3 (or add 9 and mod 12)
        let pitch_class_c = (pitch_class + 9) % 12;

        // Convert dB to linear power: power = 10^(dB/10)
        let linear_power = 10.0_f32.powf(amplitude_db / 10.0);

        // Sum linear power values (correct way to combine power across octaves)
        *pitch_class_power.entry(pitch_class_c).or_insert(0.0) += linear_power;
    }

    let detected_pitch_classes: Vec<usize> = pitch_class_power.keys().cloned().collect();

    if detected_pitch_classes.len() < min_notes {
        return None;
    }

    // Define chord types to test
    let chord_types = vec![
        // Triads
        (ChordQuality::Major, ChordNumber::Triad),
        (ChordQuality::Minor, ChordNumber::Triad),
        (ChordQuality::Diminished, ChordNumber::Triad),
        (ChordQuality::Augmented, ChordNumber::Triad),
        (ChordQuality::Suspended2, ChordNumber::Triad),
        (ChordQuality::Suspended4, ChordNumber::Triad),
        // Seventh chords
        (ChordQuality::Dominant, ChordNumber::Seventh),
        (ChordQuality::Major, ChordNumber::Seventh),
        (ChordQuality::Minor, ChordNumber::Seventh),
        (ChordQuality::Diminished, ChordNumber::Seventh),
        (ChordQuality::HalfDiminished, ChordNumber::Seventh),
    ];

    let mut candidates = Vec::new();

    // Test all possible roots (all 12 pitch classes)
    for root_pc in 0..12 {
        let root_strength = pitch_class_power.get(&root_pc).copied().unwrap_or(0.0);

        // Test each chord type with this root
        for (quality, number) in &chord_types {
            let chord = RmtChord::new(index_to_pitch_class(root_pc), quality.clone(), number.clone());
            let chord_notes = chord.notes();

            // Extract pitch classes from chord
            let chord_pitch_classes: Vec<usize> = chord_notes
                .iter()
                .map(|n| pitch_class_to_index(&n.pitch_class))
                .collect();

            // Count how many detected notes match this chord
            let matched = detected_pitch_classes
                .iter()
                .filter(|pc| chord_pitch_classes.contains(pc))
                .count();

            // Count extra notes (detected but not in chord)
            let extra = detected_pitch_classes
                .iter()
                .filter(|pc| !chord_pitch_classes.contains(pc))
                .count();

            // Only consider chords where we matched at least min_notes
            if matched >= min_notes {
                candidates.push(ChordCandidate {
                    root: root_pc,
                    quality: quality.clone(),
                    number: number.clone(),
                    matched_notes: matched,
                    total_chord_notes: chord_pitch_classes.len(),
                    extra_notes: extra,
                    root_strength,
                });
            }
        }
    }

    if candidates.is_empty() {
        return None;
    }

    // Sort candidates by quality (best first):
    // 1. More matched notes is better
    // 2. Fewer extra notes is better (penalize detecting notes not in the chord)
    // 3. Perfect match (matched == total_chord_notes) is better
    // 4. Stronger root note is better
    candidates.sort_by(|a, b| {
        // First: prefer more matched notes
        let matched_cmp = b.matched_notes.cmp(&a.matched_notes);
        if matched_cmp != std::cmp::Ordering::Equal {
            return matched_cmp;
        }

        // Second: prefer fewer extra notes
        let extra_cmp = a.extra_notes.cmp(&b.extra_notes);
        if extra_cmp != std::cmp::Ordering::Equal {
            return extra_cmp;
        }

        // Third: prefer complete matches
        let a_complete = a.matched_notes == a.total_chord_notes;
        let b_complete = b.matched_notes == b.total_chord_notes;
        let complete_cmp = b_complete.cmp(&a_complete);
        if complete_cmp != std::cmp::Ordering::Equal {
            return complete_cmp;
        }

        // Fourth: prefer stronger root
        b.root_strength.partial_cmp(&a.root_strength).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take the best candidate
    let best = &candidates[0];

    // Calculate confidence score
    let match_ratio = best.matched_notes as f32 / best.total_chord_notes as f32;
    let extra_penalty = 1.0 / (1.0 + best.extra_notes as f32 * 0.2); // Penalize extra notes

    // Normalize root strength relative to total power
    let total_power: f32 = pitch_class_power.values().sum();
    let root_ratio = if total_power > 0.0 {
        best.root_strength / total_power
    } else {
        0.0
    };

    let confidence = (match_ratio * 0.4 + extra_penalty * 0.3 + root_ratio * 0.3).min(1.0);

    Some(DetectedChord {
        root: best.root,
        quality: rmt_quality_to_old_quality(&best.quality, &best.number),
        notes: detected_pitch_classes,
        confidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_major_detection() {
        let mut active_bins = HashMap::new();
        // Simulate C, E, G in a 24-buckets-per-octave VQT
        // VQT is A-based: A=0, so C=3 semitones from A
        // For 24 buckets per octave: 24/12 = 2 buckets per semitone
        // C is at bin 6 (3 semitones * 2 buckets/semitone)
        // E is at bin 14 (7 semitones from A * 2)
        // G is at bin 20 (10 semitones from A * 2)
        active_bins.insert(6, 1.0);  // C
        active_bins.insert(14, 0.8); // E
        active_bins.insert(20, 0.6); // G

        let result = detect_chord_enhanced(&active_bins, 24, 2);
        assert!(result.is_some());

        let chord = result.unwrap();
        // Debug: print what we actually got
        println!("Detected root: {}, quality: {:?}, confidence: {}",
                 chord.root, chord.quality, chord.confidence);
        println!("Detected notes: {:?}", chord.notes);

        // The algorithm should detect C major (root=0, quality=Major)
        assert_eq!(chord.root, 0); // C
        assert!(matches!(chord.quality, OldChordQuality::Major));
    }

    #[test]
    fn test_no_chord_with_one_note() {
        let mut active_bins = HashMap::new();
        active_bins.insert(6, 1.0); // Just C

        let result = detect_chord_enhanced(&active_bins, 24, 2);
        assert!(result.is_none());
    }
}
