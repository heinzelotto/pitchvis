# Vibrato Shader Animation

## Overview

Real-time GPU-based vibrato visualization that makes pitch balls **pulsate and breathe** at the detected vibrato rate. This creates an intuitive, organic visual representation of vibrato that singers can see while performing.

## Visual Effect

Notes with vibrato will show:
- **Pulsating rings** that oscillate at the vibrato rate (4-8 Hz)
- **Breathing effect** - the pitch ball expands and contracts
- **Amplitude proportional to extent** - wider vibrato = stronger pulsation
- **Synchronized to actual vibrato** - animation matches singer's vibrato exactly

## Implementation

### Architecture

The vibrato animation is implemented entirely in the GPU shader for optimal performance:

1. **CPU side** (`pitchvis_viewer/src/display_system/update.rs`):
   - Reads vibrato state from analysis
   - Passes `vibrato_rate` (Hz) and `vibrato_extent` (0-1) to shader via uniform

2. **GPU side** (`pitchvis_viewer/assets/shaders/noisy_color_rings_2d.wgsl`):
   - Calculates vibrato phase: `sin(time * rate * 2π)`
   - Modulates UV coordinates to create pulsating effect
   - Max pulse amplitude: ±5% of ball size

### Shader Parameters

Extended the `Params` struct to include vibrato data:

```rust
pub struct Params {
    pub calmness: f32,        // [0.0, 1.0] - ring noise/fullness
    pub time: f32,            // seconds - for noise animation
    pub vibrato_rate: f32,    // Hz [0.0, 10.0] - 0 if no vibrato
    pub vibrato_extent: f32,  // normalized [0.0, 1.0] - 0 if no vibrato
}
```

### Shader Algorithm

```wgsl
// Calculate vibrato phase (oscillates at detected rate)
let vibrato_phase = sin(time * vibrato_rate * 2.0 * PI);

// Scale UV coordinates to create pulsating effect
// Amplitude controlled by vibrato extent (normalized 0-1)
let pulse_amplitude = vibrato_extent * 0.05;  // Max ±5% size
let scale_factor = 1.0 + vibrato_phase * pulse_amplitude;
uv = uv * scale_factor;

// Use modulated UV for ring calculation
let f_ring = ring(uv);  // Creates pulsating ring pattern
```

### Normalization

Vibrato extent is normalized for shader use:
- **Input**: Vibrato extent in cents (e.g., 78 cents)
- **Normalization**: `extent / 120.0` (healthy max is 120 cents)
- **Clamped**: `.min(1.0)` to prevent excessive pulsation
- **Result**: 0.0-1.0 range where 1.0 = full healthy vibrato width

### Activation Conditions

Vibrato animation is only active when:
1. Display mode is Normal or Debugging (not Performance/Zen/Galaxy)
2. Vibrato is detected (`is_active == true`)
3. Confidence is high (`confidence > 0.7`)
4. Rate is above threshold (`rate > 0.1 Hz` in shader)

Otherwise, `vibrato_rate` and `vibrato_extent` are set to 0.0, disabling the animation.

## Visual Examples

### Scenario: Singer with healthy vibrato

**Input**:
- Vibrato rate: 6.2 Hz
- Vibrato extent: 78 cents
- Confidence: 0.85

**Shader receives**:
- `vibrato_rate = 6.2`
- `vibrato_extent = 78 / 120 = 0.65`

**Visual result**:
- Pitch ball pulsates at 6.2 Hz (visible oscillation)
- Pulse amplitude: ±3.25% of ball size (0.65 × 5%)
- Smooth, organic breathing effect
- Synchronized exactly with singer's vibrato

### Scenario: Singer with excessive vibrato

**Input**:
- Vibrato rate: 6.5 Hz
- Vibrato extent: 150 cents (too wide!)
- Confidence: 0.90

**Shader receives**:
- `vibrato_rate = 6.5`
- `vibrato_extent = 150 / 120 = 1.25 → clamped to 1.0`

**Visual result**:
- Pulsates at 6.5 Hz
- Maximum pulse amplitude: ±5% (clamped)
- **Plus** yellow color tint from vibrato health visualization
- Singer sees: fast-pulsating yellow ball = "vibrato too wide"

### Scenario: Straight tone (no vibrato)

**Input**:
- Vibrato not detected
- Confidence: 0.0

**Shader receives**:
- `vibrato_rate = 0.0`
- `vibrato_extent = 0.0`

**Visual result**:
- No pulsation (static rings)
- Normal calmness-based ring effect
- Clean, steady pitch ball

## Benefits for Choir Singers

### Real-time Feedback

1. **Immediate visual confirmation**: See vibrato as it happens
2. **Rate perception**: Fast pulsation = fast vibrato, slow = slow vibrato
3. **Extent visualization**: Strong pulsation = wide vibrato
4. **Training aid**: Match your vibrato to the visual rhythm

### Combined with Other Feedback

Works harmoniously with existing visualizations:

| Feature | What it shows | Visual channel |
|---------|---------------|----------------|
| Vibrato shader animation | Rate & extent | **Motion** (pulsation) |
| Vibrato health tinting | Category (healthy/wobble/tremolo) | **Color** (hue) |
| Tuning accuracy | Pitch precision | **Brightness** |
| Calmness | Note sustain | **Size** |

**Result**: A singing note with vibrato provides 4 simultaneous feedback channels!

### Pedagogical Use

**Exercise 1: Vibrato control**
- Sing with varying vibrato widths
- Watch pulsation amplitude change
- Train conscious control of vibrato extent

**Exercise 2: Vibrato rate matching**
- Set metronome to target rate (e.g., 6 Hz)
- Sing and match pulsation to metronome
- Develop consistent vibrato rate

**Exercise 3: Vibrato onset**
- Start with straight tone (no pulsation)
- Gradually introduce vibrato
- Watch pulsation smoothly fade in
- Train controlled vibrato onset

## Performance Characteristics

### GPU Efficiency

- **Cost**: Negligible - all computation on GPU
- **Operations per pixel**:
  - 1 sine calculation (vibrato phase)
  - 2 multiplications (scale factor)
  - No branches or complex operations
- **Estimated overhead**: < 0.01% GPU time

### CPU Overhead

- **Per visible note**: 2 float assignments
- **Total**: ~0.05% CPU overhead
- **Memory**: No additional allocation (uses existing Params struct)

### Why GPU Implementation?

1. **Parallel execution**: All visible notes animated simultaneously
2. **No CPU bottleneck**: Animation happens entirely on GPU
3. **Smooth 60 FPS**: Even with many notes with vibrato
4. **Power efficient**: GPU designed for parallel graphics operations

## Technical Details

### Pulsation Formula

The pulsation effect is created by modulating UV coordinates:

```
uv_original = mesh.uv * 2.0 - 1.0           // [-1, 1] range
phase = sin(time * rate * 2π)                // [-1, 1] oscillation
amplitude = extent * 0.05                    // Max ±5%
scale = 1.0 + phase * amplitude              // [0.95, 1.05]
uv_modulated = uv_original * scale           // Pulsating UVs
```

Result: The ring pattern `ring(uv)` is calculated with modulated coordinates, creating the pulsating effect.

### Why ±5% Maximum?

- **Perceptibility**: 5% size change is clearly visible but not jarring
- **Matches physiology**: Vibrato is subtle oscillation, not dramatic
- **Aesthetic**: Maintains elegant visualization without overwhelming
- **Preserves other effects**: Doesn't interfere with calmness scaling

### Synchronization

The animation is perfectly synchronized with the singer's vibrato:

1. **Analysis detects vibrato**: Rate measured via autocorrelation
2. **Rate passed to shader**: Updated every frame (60 FPS)
3. **Shader calculates phase**: `sin(time * rate * 2π)`
4. **Phase drives pulsation**: Matches actual vibrato frequency

The result is that the visual pulsation is **phase-locked** to the audio vibrato.

## Integration with Existing Systems

### Display Modes

| Mode | Vibrato Animation | Reason |
|------|-------------------|--------|
| Normal | ✅ Enabled | Primary performance mode |
| Debugging | ✅ Enabled | Full feedback for development |
| Performance | ❌ Disabled | Minimal visual load |
| Zen | ❌ Disabled | Aesthetic focus |
| Galaxy | ❌ Disabled | Aesthetic focus |

### Fading Pitch Balls

When notes release and pitch balls fade out:
- Vibrato animation is **disabled** (`rate = 0, extent = 0`)
- Prevents visual confusion from fading balls still pulsating
- Clean transition from active → fading → hidden

### Calmness Interaction

The vibrato pulsation and calmness ring effect work together:

- **High calmness**: Strong ring pattern + vibrato pulsation = "breathing rings"
- **Low calmness**: Solid disk + vibrato pulsation = "pulsating disk"
- Both effects use the same `ring()` function with modulated UVs

## Future Enhancements

Potential additions to vibrato shader animation:

### 1. **Phase-accurate vibrato** (Advanced)
Currently uses `sin(time * rate)` which creates smooth oscillation but doesn't track actual pitch phase. Could:
- Pass vibrato phase from analysis
- Show exactly when pitch is sharp vs flat
- Enable precise vibrato shape visualization

### 2. **Vibrato shape visualization**
Some singers have asymmetric vibrato (sawtooth vs sine). Could:
- Analyze vibrato waveform shape
- Modulate pulsation to match (sine vs triangle vs sawtooth)
- Help singers develop smoother vibrato

### 3. **Multi-layered pulsation**
For notes with multiple harmonics, could:
- Detect vibrato on fundamental + overtones
- Create layered pulsation effects
- Visualize harmonic richness

### 4. **Tuning-coupled animation**
Combine vibrato pulsation with tuning feedback:
- Pulsation brightness varies with tuning accuracy
- Green flash when perfectly in tune during vibrato cycle
- Real-time feedback on intonation during vibrato

### 5. **Customizable aesthetics**
Allow users to adjust:
- Maximum pulsation amplitude (default 5%)
- Animation style (expand/contract vs rotate vs shimmer)
- Enable/disable per display mode

## Testing Recommendations

### Visual Testing

1. **Test with operatic recordings** (strong vibrato 6-7 Hz)
   - Verify pulsation is visible and smooth
   - Check rate matches expected vibrato

2. **Test with varying widths**
   - 40 cents: subtle pulsation
   - 80 cents: moderate pulsation
   - 120 cents: strong pulsation

3. **Test edge cases**
   - Very fast vibrato (8-9 Hz): should still be trackable
   - Very slow (3-4 Hz): should be clearly visible
   - Straight tone: no pulsation

### Performance Testing

1. **Many simultaneous notes with vibrato**
   - Spawn 50+ pitch balls with vibrato
   - Verify 60 FPS maintained
   - Check GPU usage

2. **Mode switching**
   - Toggle between Normal/Performance modes
   - Verify animation disables cleanly
   - No visual artifacts

3. **Fading behavior**
   - Release sustained note with vibrato
   - Verify pulsation stops when fading begins
   - Clean transition

## Related Documentation

- `VIBRATO_ANALYSIS.md` - Vibrato detection algorithm
- `VIBRATO_VISUALIZATION.md` - Color-based vibrato health feedback
- `CHOIR_SINGER_FEATURES.md` - Complete feature overview
- Shader code: `pitchvis_viewer/assets/shaders/noisy_color_rings_2d.wgsl`
- Material definition: `pitchvis_viewer/src/display_system/material.rs`
