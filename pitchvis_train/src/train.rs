use anyhow::Result;
use hound::WavWriter;
use rustysynth::*;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

fn read_file(file_path: &str) -> Result<Vec<u8>> {
    let path = Path::new(file_path);
    let mut file = File::open(&path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    Ok(buf)
}

pub fn train() -> Result<()> {
    let midi_path = "example.mid";
    let soundfont_path = "MuseScore_General.sf2";
    let output_wav_path = "output_wav_file.wav";
    synthesize_midi_to_wav(midi_path, soundfont_path, output_wav_path)?;

    Ok(())
}

fn synthesize_midi_to_wav(
    midi_path: &str,
    soundfont_path: &str,
    output_wav_path: &str,
) -> Result<()> {
    let mut sf2 = File::open(soundfont_path).unwrap();
    let sound_font = Arc::new(SoundFont::new(&mut sf2).unwrap());

    // Load the MIDI file.
    let mut mid = File::open(midi_path).unwrap();
    let midi_file = Arc::new(MidiFile::new(&mut mid).unwrap());

    // Create the MIDI file sequencer.
    let settings = SynthesizerSettings::new(44100);
    let synthesizer = Synthesizer::new(&sound_font, &settings).unwrap();
    let mut sequencer = MidiFileSequencer::new(synthesizer);

    // Play the MIDI file.
    sequencer.play(&midi_file, false);

    // The output buffer.
    let sample_count = (settings.sample_rate as f64 * midi_file.get_length()) as usize;
    let mut left: Vec<f32> = vec![0_f32; sample_count];
    let mut right: Vec<f32> = vec![0_f32; sample_count];

    // Render the waveform.
    sequencer.render(&mut left[..], &mut right[..]);

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 44100,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = WavWriter::create(output_wav_path, spec).expect("Failed to create WAV file");

    left.iter().zip(right.iter()).for_each(|(l, r)| {
        writer
            .write_sample(*l)
            .expect("Failed to write sample to WAV file");
        writer
            .write_sample(*r)
            .expect("Failed to write sample to WAV file");
    });

    // Finalize the WAV file
    writer.finalize().expect("Failed to finalize WAV file");

    Ok(())
}
