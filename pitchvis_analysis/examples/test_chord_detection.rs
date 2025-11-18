use rust_music_theory::chord::{Chord, Number as ChordNumber, Quality as ChordQuality};
use rust_music_theory::note::{Note, Notes, PitchClass};

fn main() {
    println!("=== Testing rust-music-theory chord functionality ===\n");

    // Test 1: Create a C Major chord
    println!("Test 1: C Major chord");
    let c_major = Chord::new(PitchClass::C, ChordQuality::Major, ChordNumber::Triad);
    let notes = c_major.notes();
    println!("  C Major notes: {:?}", notes);
    println!();

    // Test 2: Create various chord types
    println!("Test 2: Various chord types");
    let chords = vec![
        ("C Minor", Chord::new(PitchClass::C, ChordQuality::Minor, ChordNumber::Triad)),
        ("C Dominant 7th", Chord::new(PitchClass::C, ChordQuality::Dominant, ChordNumber::Seventh)),
        ("C Major 7th", Chord::new(PitchClass::C, ChordQuality::Major, ChordNumber::Seventh)),
        ("C Minor 7th", Chord::new(PitchClass::C, ChordQuality::Minor, ChordNumber::Seventh)),
    ];

    for (name, chord) in &chords {
        println!("  {}: {:?}", name, chord.notes());
    }
    println!();

    // Test 3: Check what pitch classes are available
    println!("Test 3: Available PitchClasses");
    let pitch_classes = vec![
        PitchClass::C,
        PitchClass::Cs,
        PitchClass::D,
        PitchClass::Ds,
        PitchClass::E,
        PitchClass::F,
        PitchClass::Fs,
        PitchClass::G,
        PitchClass::Gs,
        PitchClass::A,
        PitchClass::As,
        PitchClass::B,
    ];
    println!("  All 12 pitch classes: {:?}", pitch_classes);
    println!();

    // Test 4: Try to understand how we could match detected notes to chords
    println!("Test 4: Chord matching simulation");
    println!("  If we detect notes C, E, G:");

    // Simulate what we would detect
    let detected_pitch_classes = vec![
        PitchClass::C,  // 0
        PitchClass::E,  // 4
        PitchClass::G,  // 7
    ];

    println!("    Detected pitch classes: {:?}", detected_pitch_classes);

    // Try different chords and see which one matches
    let test_chords = vec![
        ("C Major", Chord::new(PitchClass::C, ChordQuality::Major, ChordNumber::Triad)),
        ("C Minor", Chord::new(PitchClass::C, ChordQuality::Minor, ChordNumber::Triad)),
        ("E Minor", Chord::new(PitchClass::E, ChordQuality::Minor, ChordNumber::Triad)),
    ];

    for (name, chord) in &test_chords {
        let chord_notes = chord.notes();
        let chord_pitch_classes: Vec<PitchClass> = chord_notes.iter()
            .map(|n| n.pitch_class.clone())
            .collect();
        println!("    {} has pitch classes: {:?}", name, chord_pitch_classes);

        // Check if all detected notes are in this chord
        let all_match = detected_pitch_classes.iter()
            .all(|pc| chord_pitch_classes.contains(pc));
        println!("      All detected notes in chord? {}", all_match);
    }
}
