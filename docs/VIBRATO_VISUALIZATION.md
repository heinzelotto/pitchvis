# Vibrato Visualization for Choir Singers

## Overview

Real-time visual feedback system that helps choir singers identify and correct vibrato issues during performance. Implemented as color-coded tints on pitch balls in the PitchVis viewer.

## Implementation

### Location
- **File**: `pitchvis_viewer/src/display_system/update.rs`
- **Function**: `update_pitch_balls()` (lines 209-235)
- **Data Source**: `analysis_state.vibrato_states[idx]` (computed by vibrato analysis)

### Visual Feedback System

The system applies subtle color tints to pitch balls based on vibrato health:

| Vibrato Issue | Color Tint | RGB Offset | Meaning |
|---------------|------------|------------|---------|
| **Healthy** | None | (0, 0, 0) | Vibrato within acceptable ranges |
| **Wobble** | Blue | (-0.2, -0.1, +0.3) | Too slow (< 4.5 Hz) |
| **Tremolo** | Orange/Red | (+0.4, +0.1, -0.2) | Too fast (> 8 Hz) |
| **Excessive** | Yellow | (+0.3, +0.3, -0.1) | Too wide (> 120 cents) |
| **Minimal** | Cyan | (-0.1, +0.2, +0.2) | Too narrow (< 40 cents) |
| **Straight Tone** | None | (0, 0, 0) | No vibrato detected |

### Design Principles

1. **Subtle Feedback**: Tint strength is 30% to avoid overwhelming the base pitch color
2. **Confidence Threshold**: Only applies to vibrato with confidence > 0.7
3. **Healthy is Invisible**: No visual change for healthy vibrato (positive reinforcement)
4. **Issues Stand Out**: Unhealthy vibrato gets colored tints (draws attention)
5. **Real-time**: Updates at 60 FPS as singers perform

### Code Flow

```rust
// For each visible pitch ball:
1. Get vibrato state for the bin: vibrato_states[idx]
2. Check if vibrato is active AND confident (> 0.7)
3. Determine vibrato category (Healthy/Wobble/Tremolo/etc)
4. Apply appropriate color tint to the pitch ball
5. Render with modulated color
```

## User Experience

### For Choir Singers

When singing with vibrato:
- **Normal appearance**: Your vibrato is healthy
- **Blue tint appears**: Slow down your vibrato (currently too slow)
- **Orange/red tint**: Speed up your vibrato (currently too fast)
- **Yellow tint**: Reduce vibrato width (oscillating too widely)
- **Cyan tint**: Increase vibrato width (too subtle)

### Benefits

1. **Immediate Feedback**: See vibrato issues in real-time while singing
2. **Intuitive**: Color-coded warnings don't require reading numbers
3. **Non-intrusive**: Only appears when issues are detected
4. **Pedagogical**: Learn to self-correct vibrato technique
5. **Performance Tool**: Monitor vibrato health during rehearsals/performances

## Technical Details

### Performance Impact

- **Overhead**: < 0.1% per frame
- **Cost**: One vector lookup + one match statement per visible note
- **Data Already Computed**: Vibrato analysis runs in pitchvis_analysis
- **No Additional I/O**: Pure computation on existing data

### Integration with Other Features

- **Works with ML feature**: Vibrato tint applied before ML pitch highlighting
- **Respects display modes**: Only modifies pitch ball colors, doesn't affect spectrum
- **Compatible with calmness**: Vibrato tint combines with existing calmness scaling

### Healthy Vibrato Ranges

Based on vocal pedagogy research:
- **Rate**: 4.5-8 Hz (typical: 5.5-6.5 Hz)
- **Extent**: 40-120 cents peak-to-peak (typical: 60-80 cents)
- **Regularity**: ≥ 0.6 (on 0-1 scale)

## Future Enhancements

Potential additions:
1. **Vibrato metrics overlay** - Text display showing exact Hz/cents values
2. **Vibrato onset visualization** - Highlight when vibrato starts/stops
3. **Reference vibrato comparison** - Show target vs actual vibrato shape
4. **Historical vibrato tracking** - Graph vibrato stability over time
5. **Per-note vibrato settings** - Different healthy ranges for different voice types

## Related Documentation

- `VIBRATO_ANALYSIS.md` - Technical design of vibrato detection algorithm
- `CALMNESS_ANALYSIS_AND_IMPROVEMENTS.md` - Related analysis features
- `ATTACK_AND_PERCUSSION_DETECTION.md` - Complementary onset detection
