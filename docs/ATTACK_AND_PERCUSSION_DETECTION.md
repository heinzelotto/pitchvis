# Attack and Percussion Detection

## Overview

This document describes the design and implementation of **attack detection** and **percussion detection** for PitchVis. These features enable:
- Distinguishing newly attacked notes from sustained notes
- Identifying percussive vs. melodic instruments
- Different visualization for attacks (e.g., brightness, particle effects)
- Advanced musical analysis (onset timing, rhythm detection)

## Definitions

### Attack
An **attack** is the onset of a note - the moment when a sound begins or when an existing note is re-articulated.

**Characteristics:**
- Rapid amplitude increase (high dB/s rate of change)
- New peak appears in spectrum
- Or existing peak suddenly jumps in amplitude
- Transient energy burst

**Examples:**
- Piano key press
- Plucked guitar string
- Drum hit
- Staccato violin note
- Re-tongued wind instrument note

### Percussion
**Percussion** refers to sounds with a percussive character - sharp attack with quick decay.

**Characteristics:**
- Very sharp attack (< 20ms rise time)
- Rapid decay (amplitude drops quickly after attack)
- Short sustain (< 200ms total duration)
- Often weak or inharmonic overtones
- High transient-to-sustain ratio

**Examples:**
- Drums, cymbals, percussion instruments
- Pizzicato strings
- Staccato piano (short notes)
- Percussion synthesizers

**Non-percussion for comparison:**
- Sustained organ notes (slow attack, long sustain)
- Bowed strings (medium attack, long sustain)
- Pads (very slow attack, long sustain)
- Legato piano (medium attack, medium sustain)

## Design Challenges

### Challenge 1: Noise vs. Real Attacks

**Problem:** Random amplitude fluctuations can look like attacks.

**Example:**
```
Noise spike: 35 dB → 42 dB → 36 dB (7 dB jump in 30ms)
Real attack:  30 dB → 50 dB → 48 dB (20 dB jump, then sustain)
```

**Solution:**
- Require minimum amplitude threshold (e.g., peak must be > 40 dB)
- Require minimum rate of change (e.g., > 50 dB/s)
- Check for sustain after attack (noise decays immediately)
- Use smoothed VQT to filter jitter

### Challenge 2: Multiple Attacks at Same Frequency

**Problem:** Two notes at the same pitch in quick succession.

**Example:**
```
Repeated piano note at 440 Hz:
  t=0ms:   Attack 1 (50 dB)
  t=100ms: Decay to 35 dB
  t=200ms: Attack 2 (52 dB)  ← Must detect this!
```

**Solution:**
- Track amplitude history per bin
- Detect "re-attack" when amplitude increases after decay
- Don't require peak to disappear completely

### Challenge 3: Distinguishing Percussion from Short Melodic Notes

**Problem:** Staccato melodic notes can look percussive.

**Example:**
```
Staccato piano:  Sharp attack, 100ms sustain, harmonic structure
Snare drum:      Sharp attack, 80ms sustain, inharmonic
```

**Solution:**
- Check harmonic content (percussion is inharmonic)
- Measure decay shape (percussion: exponential, melodic: plateau then decay)
- Consider frequency range (percussion often in specific ranges)

## Implementation Design

### Data Structures

Add to `AnalysisState`:

```rust
pub struct AnalysisState {
    // ... existing fields ...

    /// Per-bin attack state tracking
    pub attack_state: Vec<AttackState>,

    /// Per-bin percussion characteristics
    pub percussion_score: Vec<f32>,  // 0.0 = melodic, 1.0 = percussive

    /// Attack events in current frame (for visualization)
    pub current_attacks: Vec<AttackEvent>,
}

/// Tracks attack state for a single frequency bin
#[derive(Debug, Clone)]
pub struct AttackState {
    /// Time since last attack (in seconds)
    pub time_since_attack: f32,

    /// Peak amplitude at time of last attack (dB)
    pub attack_amplitude: f32,

    /// Amplitude of previous frame (for rate-of-change detection)
    pub previous_amplitude: f32,

    /// Is this bin currently in attack phase? (first ~50ms after onset)
    pub in_attack_phase: bool,

    /// Amplitude history for decay rate calculation (last 10 frames)
    pub amplitude_history: VecDeque<f32>,

    /// Timestamp history (for rate calculations)
    pub time_history: VecDeque<f32>,
}

/// Represents a detected attack event
#[derive(Debug, Clone, Copy)]
pub struct AttackEvent {
    /// Bin index where attack occurred
    pub bin_idx: usize,

    /// Frequency of the attacked note (Hz)
    pub frequency: f32,

    /// Attack amplitude (dB)
    pub amplitude: f32,

    /// Rate of amplitude increase (dB/s)
    pub attack_rate: f32,

    /// Percussion score for this attack (0.0-1.0)
    pub percussion_score: f32,
}
```

### Attack Detection Algorithm

```rust
fn detect_attacks(&mut self, x_vqt: &[f32], frame_time: Duration) {
    let dt = frame_time.as_secs_f32();
    self.current_attacks.clear();

    for (bin_idx, state) in self.attack_state.iter_mut().enumerate() {
        let current_amp = x_vqt[bin_idx];
        let amp_change = current_amp - state.previous_amplitude;
        let rate_of_change = amp_change / dt;  // dB/s

        // Update time since last attack
        state.time_since_attack += dt;

        // ATTACK DETECTION CRITERIA
        let attack_detected =
            rate_of_change > 50.0 &&           // Rapid increase (> 50 dB/s)
            current_amp > 38.0 &&              // Minimum amplitude (above noise floor)
            amp_change > 3.0 &&                // Minimum delta (> 3 dB jump)
            state.time_since_attack > 0.05;   // Cooldown period (> 50ms since last)

        if attack_detected {
            // Record attack event
            state.time_since_attack = 0.0;
            state.attack_amplitude = current_amp;
            state.in_attack_phase = true;

            // Calculate percussion score (will refine later)
            let percussion = calculate_percussion_score(state, bin_idx);

            self.current_attacks.push(AttackEvent {
                bin_idx,
                frequency: self.bin_to_frequency(bin_idx),
                amplitude: current_amp,
                attack_rate: rate_of_change,
                percussion_score: percussion,
            });
        }

        // Exit attack phase after 50ms
        if state.time_since_attack > 0.05 {
            state.in_attack_phase = false;
        }

        // Update amplitude history
        state.amplitude_history.push_back(current_amp);
        state.time_history.push_back(state.time_since_attack);
        if state.amplitude_history.len() > 10 {
            state.amplitude_history.pop_front();
            state.time_history.pop_front();
        }

        // Store for next frame
        state.previous_amplitude = current_amp;
    }
}
```

### Percussion Detection Algorithm

```rust
fn calculate_percussion_score(state: &AttackState, bin_idx: usize) -> f32 {
    let mut percussion_score = 0.0;
    let mut factor_count = 0;

    // FACTOR 1: Decay Rate
    // Percussion decays quickly after attack
    if state.amplitude_history.len() >= 5 {
        let recent_amps: Vec<f32> = state.amplitude_history.iter().rev().take(5).copied().collect();
        let decay_rate = calculate_decay_rate(&recent_amps, &state.time_history);

        // High decay rate = percussive
        // > 100 dB/s = very percussive (drums)
        // < 20 dB/s = sustained (organ, strings)
        let decay_factor = (decay_rate / 100.0).min(1.0);
        percussion_score += decay_factor;
        factor_count += 1;
    }

    // FACTOR 2: Attack Sharpness
    // Very sharp attacks (> 200 dB/s) are percussive
    // Slow attacks (< 50 dB/s) are melodic
    if state.time_since_attack < 0.1 {
        let attack_rate = (state.attack_amplitude - state.previous_amplitude)
                          / state.time_since_attack.max(0.001);
        let attack_factor = ((attack_rate - 50.0) / 150.0).clamp(0.0, 1.0);
        percussion_score += attack_factor;
        factor_count += 1;
    }

    // FACTOR 3: Harmonic Content (if available)
    // Percussion has weak/inharmonic content
    // This could use the harmonic detection already implemented
    if let Some(harmonic_strength) = self.get_harmonic_strength(bin_idx) {
        // Low harmonic strength = more percussive
        let harmonic_factor = 1.0 - harmonic_strength;
        percussion_score += harmonic_factor;
        factor_count += 1;
    }

    // FACTOR 4: Sustain Duration
    // Percussion notes don't sustain long
    // After 200ms, if amplitude hasn't dropped significantly, it's melodic
    if state.time_since_attack > 0.2 {
        let amp_drop = state.attack_amplitude - state.previous_amplitude;
        if amp_drop < 10.0 {
            // Still sustaining after 200ms = melodic
            percussion_score -= 0.5;
            factor_count += 1;
        }
    }

    // Average all factors
    if factor_count > 0 {
        (percussion_score / factor_count as f32).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn calculate_decay_rate(amplitudes: &[f32], times: &VecDeque<f32>) -> f32 {
    if amplitudes.len() < 2 {
        return 0.0;
    }

    // Linear regression on amplitude vs time to get decay rate
    // Negative slope = decay, positive slope = crescendo
    let n = amplitudes.len() as f32;
    let sum_t: f32 = times.iter().rev().take(amplitudes.len()).sum();
    let sum_a: f32 = amplitudes.iter().sum();
    let sum_ta: f32 = times.iter().rev().take(amplitudes.len())
        .zip(amplitudes.iter())
        .map(|(t, a)| t * a)
        .sum();
    let sum_tt: f32 = times.iter().rev().take(amplitudes.len())
        .map(|t| t * t)
        .sum();

    let slope = (n * sum_ta - sum_t * sum_a) / (n * sum_tt - sum_t * sum_t);

    // Return absolute decay rate (negative slope = decay)
    -slope
}
```

### Integration with Existing Systems

#### 1. Use Smoothed VQT for Attack Detection

```rust
pub fn preprocess(&mut self, x_vqt: &[f32], frame_time: Duration) {
    // ... existing smoothing ...

    // Get smoothed VQT for attack detection (reduces noise)
    let x_vqt_smoothed: Vec<f32> = self.x_vqt_smoothed.iter().map(|x| x.get()).collect();

    // Detect attacks using smoothed data
    self.detect_attacks(&x_vqt_smoothed, frame_time);

    // ... existing peak detection, calmness, etc. ...
}
```

#### 2. Filter Attacks by Peak Presence

Only consider attacks at actual peak locations to reduce false positives:

```rust
fn detect_attacks(&mut self, x_vqt: &[f32], frame_time: Duration) {
    // ... attack detection ...

    // Filter: only keep attacks that coincide with detected peaks
    self.current_attacks.retain(|attack| {
        // Check if there's a peak within ±2 bins
        let bin_range = (attack.bin_idx.saturating_sub(2))
            ..=(attack.bin_idx + 2).min(self.range.n_buckets() - 1);

        bin_range.any(|b| self.peaks.contains(&b))
    });
}
```

#### 3. Enhance Harmonic Detection with Percussion Info

Percussive notes should not be promoted by harmonic content (they're often inharmonic):

```rust
fn promote_bass_peaks_with_harmonics(...) {
    for peak in peaks_continuous.iter_mut() {
        // Skip percussion peaks - they're inharmonic
        if self.percussion_score[peak.center as usize] > 0.7 {
            continue;  // Very percussive, don't apply harmonic boost
        }

        // ... normal harmonic promotion ...
    }
}
```

## Parameters

Add to `AnalysisParameters`:

```rust
pub struct AnalysisParameters {
    // ... existing parameters ...

    /// Attack detection parameters
    attack_detection_config: AttackDetectionParameters,
}

#[derive(Debug, Clone)]
pub struct AttackDetectionParameters {
    /// Minimum rate of amplitude increase to detect attack (dB/s)
    pub min_attack_rate: f32,              // Default: 50.0

    /// Minimum amplitude for attack detection (dB)
    pub min_attack_amplitude: f32,         // Default: 38.0

    /// Minimum amplitude jump to detect attack (dB)
    pub min_attack_delta: f32,             // Default: 3.0

    /// Cooldown period between attacks on same bin (seconds)
    pub attack_cooldown: f32,              // Default: 0.05

    /// Duration of attack phase (seconds)
    pub attack_phase_duration: f32,        // Default: 0.05

    /// Decay rate threshold for percussion (dB/s)
    pub percussion_decay_threshold: f32,   // Default: 100.0

    /// Attack rate threshold for percussion (dB/s)
    pub percussion_attack_threshold: f32,  // Default: 200.0

    /// Sustain threshold for non-percussion (seconds)
    pub melodic_sustain_threshold: f32,    // Default: 0.2
}

impl Default for AttackDetectionParameters {
    fn default() -> Self {
        Self {
            min_attack_rate: 50.0,
            min_attack_amplitude: 38.0,
            min_attack_delta: 3.0,
            attack_cooldown: 0.05,
            attack_phase_duration: 0.05,
            percussion_decay_threshold: 100.0,
            percussion_attack_threshold: 200.0,
            melodic_sustain_threshold: 0.2,
        }
    }
}
```

## Visualization Applications

### 1. Attack Visualization

```rust
// In viewer code:
for attack in &analysis.current_attacks {
    // Spawn particle effect at attack location
    spawn_attack_particle(attack.frequency, attack.amplitude);

    // Increase brightness temporarily
    let brightness_boost = 1.0 + (attack.attack_rate / 100.0).min(0.5);
    pitch_ball.brightness *= brightness_boost;

    // Different color for percussion vs melodic
    if attack.percussion_score > 0.6 {
        pitch_ball.color = PERCUSSION_COLOR;  // Red/orange
    } else {
        pitch_ball.color = MELODIC_COLOR;     // Blue/green
    }
}
```

### 2. Attack Age Visualization

```rust
// Fade from bright (new attack) to normal (sustained)
for (bin_idx, state) in analysis.attack_state.iter().enumerate() {
    let age_factor = (state.time_since_attack / 0.5).min(1.0);  // Fade over 500ms
    pitch_ball.brightness = 1.0 - age_factor * 0.5;  // Dim to 50% over time

    // Scale based on attack phase
    if state.in_attack_phase {
        pitch_ball.scale = 1.3;  // Larger during attack
    } else {
        pitch_ball.scale = 1.0;
    }
}
```

### 3. Percussion Indicator

```rust
// Show percussion instruments differently
for (bin_idx, &percussion) in analysis.percussion_score.iter().enumerate() {
    if percussion > 0.7 {
        // Very percussive - use sharp-edged visualization
        pitch_ball.shape = SQUARE;
        pitch_ball.outline = SHARP;
    } else if percussion < 0.3 {
        // Melodic - use smooth visualization
        pitch_ball.shape = CIRCLE;
        pitch_ball.outline = SOFT;
    }
}
```

## Performance Considerations

### Computational Cost

**Attack detection (per frame):**
- Per-bin operations: 588 bins × 10 operations = 5,880 ops
- Attack filtering: 0-10 attacks × 5 operations = 50 ops
- **Total: ~6,000 operations ≈ 2-3 µs**

**Percussion scoring (per attack):**
- Decay rate calculation: 10 samples × linear regression = ~50 ops
- Harmonic check: already computed
- **Total per attack: ~50 ops ≈ 0.2 µs**
- **For 10 attacks: ~2 µs**

**Total overhead: ~5 µs per frame = 0.03% of frame budget at 60 FPS**

### Memory Impact

```rust
struct AttackState {
    time_since_attack: f32,        // 4 bytes
    attack_amplitude: f32,         // 4 bytes
    previous_amplitude: f32,       // 4 bytes
    in_attack_phase: bool,         // 1 byte
    amplitude_history: VecDeque,   // 10 × 4 = 40 bytes
    time_history: VecDeque,        // 10 × 4 = 40 bytes
}
// Total: ~97 bytes per bin
```

**Total memory:**
- 588 bins × 97 bytes = **57 KB** (AttackState)
- Vec<AttackEvent>: typically < 10 attacks × 24 bytes = **240 bytes**
- percussion_score: 588 × 4 bytes = **2.4 KB**

**Grand total: ~60 KB additional memory**

Negligible for modern systems.

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_attack_detection_sharp_onset() {
        // Simulate sharp attack: 30 dB → 55 dB in 20ms
        let mut state = AttackState::new();
        state.previous_amplitude = 30.0;

        let attacks = detect_attack_for_bin(
            55.0,  // current amplitude
            &mut state,
            Duration::from_millis(20),
        );

        assert!(attacks.is_some());
        assert!(attacks.unwrap().attack_rate > 1000.0);  // (55-30)/0.02 = 1250 dB/s
    }

    #[test]
    fn test_attack_detection_noise_rejection() {
        // Simulate noise: 35 dB → 37 dB (too small delta)
        let mut state = AttackState::new();
        state.previous_amplitude = 35.0;

        let attacks = detect_attack_for_bin(
            37.0,
            &mut state,
            Duration::from_millis(20),
        );

        assert!(attacks.is_none());  // Should reject (delta < 3 dB)
    }

    #[test]
    fn test_percussion_detection_drum() {
        // Simulate drum hit: sharp attack, rapid decay
        let history = vec![55.0, 50.0, 42.0, 35.0, 30.0];  // Fast decay
        let decay_rate = calculate_decay_rate(&history, ...);

        assert!(decay_rate > 100.0);  // High decay = percussion
    }

    #[test]
    fn test_percussion_detection_organ() {
        // Simulate organ: slow attack, sustained
        let history = vec![50.0, 49.5, 49.0, 48.5, 48.0];  // Slow decay
        let decay_rate = calculate_decay_rate(&history, ...);

        assert!(decay_rate < 20.0);  // Low decay = melodic
    }
}
```

### Integration Tests

Test with real audio:
1. **Drum loop**: Should detect many attacks, all high percussion scores
2. **Piano melody**: Should detect attacks on each note, medium percussion scores
3. **Sustained pad**: Should detect few/no attacks, low percussion scores
4. **Mixed (drums + bass)**: Should distinguish percussive drums from melodic bass

## Future Enhancements

### 1. Onset Detection Refinement

Use spectral flux for better onset detection:
```rust
// Spectral flux: sum of positive changes across all bins
let spectral_flux: f32 = x_vqt.iter().zip(previous_vqt.iter())
    .map(|(curr, prev)| (curr - prev).max(0.0))
    .sum();

if spectral_flux > threshold {
    // Global onset detected
}
```

### 2. Rhythm Analysis

Track attack timing to extract tempo:
```rust
// Store attack times
pub attack_times: VecDeque<f32>,

// Compute inter-onset intervals (IOI)
let ioi = attack_times.windows(2)
    .map(|w| w[1] - w[0])
    .collect();

// Estimate tempo from most common IOI
let tempo = 60.0 / most_common_ioi;
```

### 3. Instrument Classification

Use percussion score + harmonic content + frequency range to classify:
```rust
fn classify_instrument(attack: &AttackEvent, harmonic_strength: f32) -> InstrumentClass {
    match (attack.percussion_score, harmonic_strength, attack.frequency) {
        (p, _, f) if p > 0.8 && f < 100.0 => InstrumentClass::BassDrum,
        (p, _, f) if p > 0.8 && f > 1000.0 => InstrumentClass::Cymbal,
        (p, h, f) if p < 0.3 && h > 0.7 && f < 200.0 => InstrumentClass::Bass,
        (p, h, _) if p < 0.3 && h > 0.7 => InstrumentClass::Melodic,
        _ => InstrumentClass::Unknown,
    }
}
```

### 4. Articulation Detection

Distinguish legato (smooth transitions) from staccato (separated notes):
```rust
// If attack happens while previous note still sustaining → legato
// If attack happens after silence → staccato
let articulation = if state.previous_amplitude > 35.0 {
    Articulation::Legato
} else {
    Articulation::Staccato
};
```

## Conclusion

This attack and percussion detection system provides:

✅ **Robust attack detection** with noise rejection
✅ **Percussion vs. melodic classification** based on multiple factors
✅ **Low computational overhead** (~5 µs per frame)
✅ **Small memory footprint** (~60 KB)
✅ **Rich data for visualization** (attack events, percussion scores, timing)
✅ **Foundation for rhythm analysis** (onset times, tempo)

The system distinguishes real musical events from noise through:
- Amplitude thresholds
- Rate-of-change thresholds
- Peak presence filtering
- Harmonic content analysis
- Sustain/decay characteristics

This enables sophisticated visualization features like:
- Attack particle effects
- Brightness bursts on onset
- Different colors/shapes for percussion vs melodic
- Scale/size changes during attack phase
- Age-based fading

All while maintaining PitchVis's real-time performance requirements.
