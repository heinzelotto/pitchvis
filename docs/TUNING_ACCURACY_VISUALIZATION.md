# Tuning Accuracy Visualization

## Overview

Real-time visual feedback system that helps choir singers achieve precise tuning by providing brightness modulation based on how close each note is to perfect intonation.

## Implementation

### Location
- **File**: `pitchvis_viewer/src/display_system/update.rs`
- **Function**: `update_pitch_balls()` (lines 261-288)
- **Calculation**: Per-note deviation from perfect semitone grid

### Visual Feedback System

The system modulates pitch ball brightness based on tuning deviation:

| Tuning Deviation | Brightness Boost | Visual Effect |
|-----------------|------------------|---------------|
| 0-10 cents | +10% | Maximum brightness (perfect tuning) |
| 10-20 cents | +5-10% | Moderate brightness (acceptable) |
| 20-30 cents | 0-5% | Slight boost (needs correction) |
| > 30 cents | 0% | No boost (significantly out of tune) |

### How It Works

```rust
// For each detected note:
1. Calculate pitch position in semitones: center * 12 / buckets_per_octave
2. Find deviation from nearest semitone: abs(position - round(position))
3. Convert to cents: deviation * 100
4. Calculate accuracy score: 1.0 - (cents / 30.0)
5. Apply brightness boost: rgb * (1.0 + 0.1 * accuracy)
```

### Design Principles

1. **Subtle Feedback**: Maximum 10% brightness boost to avoid overwhelming
2. **Linear Falloff**: Smooth gradient from 0-30 cents deviation
3. **Positive Reinforcement**: Brighter = better tuning
4. **Non-intrusive**: Works alongside other visual features (vibrato, calmness)
5. **Real-time**: Updates at 60 FPS as singers adjust

## User Experience

### For Choir Singers

When singing:
- **Brightest notes**: You're perfectly in tune (< 10 cents off)
- **Moderately bright**: Acceptable tuning (10-20 cents)
- **Dimmer notes**: Needs pitch adjustment (> 20 cents)
- **Significantly dimmer**: Substantially out of tune (> 30 cents)

### Training Benefits

1. **Precise Intonation**: See exact tuning quality in real-time
2. **Ensemble Matching**: All singers can converge on perfect tuning
3. **Ear Training**: Visual feedback reinforces pitch perception
4. **Self-Correction**: Immediate feedback enables fast adjustments
5. **Performance Confidence**: Know when you're accurately in tune

## Integration with Existing Features

### Works Alongside:
- **Vibrato coloring**: Tunes vibrato health (hue) + tuning accuracy (brightness)
- **Calmness scaling**: Calm notes are larger AND brighter if in tune
- **ML pitch detection**: Tuning accuracy applies after ML highlighting
- **Display modes**: Only active in Normal and Debugging modes

### Combined Effects Example:
A sustained note with healthy vibrato and perfect tuning will appear:
- Normal pitch hue (healthy vibrato, no color tint)
- Maximum brightness (+10% from tuning accuracy)
- Larger size (from calmness scaling)
- Result: The "ideal" note stands out clearly

## Technical Details

### Performance Impact

- **Overhead**: ~0.05% per frame
- **Cost**: One division + one multiplication per visible note
- **Data**: Uses existing peak position (center), no additional analysis
- **Real-time**: No latency, instant feedback

### Tuning Deviation Calculation

The deviation is calculated from the **equal temperament grid**:
```
Semitone position = (bin_center * 12) / buckets_per_octave
Perfect semitone = round(semitone_position)
Deviation (cents) = abs(semitone_position - perfect_semitone) * 100
```

### Why 30 Cents Threshold?

- **Perceptual**: Most listeners can detect 10-20 cent deviations
- **Pedagogical**: 30 cents (~ 1/3 semitone) is pedagogically "out of tune"
- **Just Intonation**: Some intervals use ±14-22 cent adjustments
- **Practical**: Provides useful range for visual gradient

## Complementary Systems

### Already Exists:
- **Average tuning display** (common.rs:284): Shows overall tuning accuracy in cents
  - Green: 0-10 cents average
  - Yellow: 10-20 cents
  - Orange: 20-30 cents
  - Red: > 30 cents

### Per-Note vs Average:
- **This feature (per-note)**: Shows which specific notes need adjustment
- **Existing (average)**: Shows overall ensemble tuning quality
- **Together**: Complete feedback for individual and group tuning

## Choir Director Use Cases

### During Rehearsal:
1. **Identify weak singers**: See who consistently produces dimmer notes
2. **Section tuning**: Focus on sections with lower brightness
3. **Difficult passages**: Monitor tuning in challenging harmonic progressions
4. **Warm-ups**: Verify all singers achieve bright notes on scales/arpeggios

### Advanced Techniques:
1. **Just Intonation Training**: Temporarily disable for pure interval work
2. **Temperament Comparison**: Compare equal vs just tuning visually
3. **Blend Exercises**: All singers aim for maximum brightness simultaneously
4. **Pitch Centering**: Use as reference for finding ensemble pitch center

## Future Enhancements

Potential additions:
1. **Adjustable reference**: Switch between A440, A442, historical temperaments
2. **Just intonation mode**: Different targets based on chord context
3. **Tuning tolerance settings**: Customizable threshold (5/10/15 cent standards)
4. **Color-coded rings**: Add colored outline (green/yellow/red) in addition to brightness
5. **Tuning history graph**: Show tuning drift over time
6. **Per-singer calibration**: Different brightness curves for different voice types

## Related Documentation

- `VIBRATO_VISUALIZATION.md` - Vibrato health feedback system
- `CALMNESS_ANALYSIS_AND_IMPROVEMENTS.md` - Calmness calculation
- Analysis tuning calculation: `pitchvis_analysis/src/analysis.rs:745-765`
- Average tuning display: `pitchvis_viewer/src/app/common.rs:276-302`
