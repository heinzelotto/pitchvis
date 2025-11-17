# Calmness Calculation Analysis and Improvement Proposals

## Current Implementation

### Algorithm (analysis.rs:350-393)

```rust
fn update_calmness(&mut self, x_vqt: &[f32], frame_time: Duration) {
    // 1. Find peaks in UNSMOOTHED VQT
    let peaks = find_peaks(&self.params.peak_config, x_vqt, buckets_per_octave);

    // 2. Mark bins within radius (~1/4 semitone) of each peak
    let radius = buckets_per_octave / 12 / 2;
    for each peak p:
        mark bins [p - radius, p + radius] as having peaks

    // 3. Update per-bin calmness (4.5s EMA time horizon)
    for each bin:
        if has_peak:
            bin_calmness → 1.0  // Sustained note
            add to active bins
        else:
            bin_calmness → 0.0  // Note released

    // 4. Scene calmness = average of active bin calmness (1.1s EMA)
    scene_calmness = average(active_bin_calmness values)
}
```

### Key Characteristics

**What it measures:**
- **Sustainability**: How long notes have been held
- Bins with sustained notes accumulate calmness over 4.5 seconds
- Scene calmness is the average calmness of currently active notes

**Design decisions:**
1. Uses **unsmoothed VQT** for peak detection (intentional - more responsive)
2. **Binary peak detection**: A bin either has a peak or doesn't
3. **Equal weighting**: All active notes contribute equally to scene calmness
4. **Simple average**: No amplitude, frequency, or quality weighting

## Issues and Limitations

### 1. No Amplitude Weighting

**Problem**: A quiet sustained pad contributes the same as a loud bass note.

**Example:**
```
Quiet pad at 200 Hz: amplitude 35 dB, sustained 5s → calmness = 1.0
Loud bass at 55 Hz: amplitude 50 dB, sustained 5s → calmness = 1.0
Scene calmness: average(1.0, 1.0) = 1.0
```

**Reality**: The loud bass dominates perception, so scene calmness should reflect bass characteristics more.

**Impact**: Scene calmness doesn't match perceptual importance.

### 2. No Rate-of-Change Consideration

**Problem**: Doesn't distinguish between stable vs. fluctuating amplitudes.

**Example - Two scenarios with identical scene calmness:**

**Scenario A** (Truly calm):
```
Sustained organ chord, amplitudes stable:
  Time t+0: [50, 48, 47] dB
  Time t+1: [50, 48, 47] dB
  Time t+2: [50, 48, 47] dB
  → Scene calmness: 1.0
```

**Scenario B** (Fluctuating, but notes held):
```
Tremolo strings, amplitudes oscillating:
  Time t+0: [50, 35, 50] dB  (tremolo phase)
  Time t+1: [35, 50, 35] dB
  Time t+2: [50, 35, 50] dB
  → Scene calmness: 1.0 (same!)
```

**Impact**: Fluctuating/tremolo passages incorrectly classified as calm.

### 3. Binary Peak Detection

**Problem**: Peak presence is all-or-nothing, ignoring peak strength.

**Example:**
```
Strong peak: 55 dB (prominence 20 dB) → contributes to calmness
Weak peak: 38 dB (prominence 2.1 dB) → contributes to calmness equally
```

**Impact**: Noise or weak transients contribute as much as strong sustained notes.

### 4. No Spectral Distribution Consideration

**Problem**: Doesn't consider WHERE energy is in the spectrum.

**Musical reality:**
- High treble energy → often energetic (cymbals, bright attacks)
- Bass-heavy → can be calm (organ) or energetic (EDM)
- Spectral centroid is a strong indicator of musical energy

**Example - Same number of peaks, different energy:**

**Scenario A** (Calm bass):
```
3 peaks at 55, 110, 165 Hz
Spectral centroid: ~110 Hz
→ Scene calmness: depends on sustain
```

**Scenario B** (Bright, energetic):
```
3 peaks at 1000, 2000, 4000 Hz
Spectral centroid: ~2000 Hz
→ Scene calmness: depends on sustain (should be lower!)
```

**Impact**: Treble-heavy energetic music can appear calm if sustained.

### 5. No Attack/Transient Detection

**Problem**: Doesn't distinguish smooth vs. percussive onsets.

**Example:**
```
Vibraphone: Sharp attack, quick decay → energetic
Pad synth: Slow attack, long sustain → calm
```

Current algorithm only sees that notes are being held, not HOW they started.

**Impact**: Percussive sustained notes (marimba, vibraphone) classified as calm.

### 6. FIXME Already Noted in Code

From line 355-360:
```rust
// FIXME: we only take into account the notes that are currently being played. But
// the calmness of notes is a function of their history. Should we take into account
// the calmness of notes that are not currently being played, or that have recently
// been released?
// Currently, releasing a note with above average calmness decreases scene calmness.
// Releasing a note with below average increases scene calmness.
```

**Example of the problem:**
```
Long sustained chord (calmness = 1.0)
All notes released simultaneously
Scene calmness immediately drops to 0.0 (no active peaks)
→ Sudden drop, perceptually jarring
```

**Impact**: Calmness can have unrealistic discontinuities.

## Improvement Proposals

### Proposal 1: Amplitude-Weighted Scene Calmness

**Change**: Weight each note's calmness contribution by its amplitude.

**Implementation:**
```rust
// Instead of simple average
let mut weighted_calmness_sum = 0.0;
let mut weight_sum = 0.0;

for (bin_idx, (calmness, has_peak)) in self.calmness.iter_mut().zip(peaks_around.iter()).enumerate() {
    if *has_peak {
        calmness.update_with_timestep(1.0, frame_time);

        // Weight by amplitude (convert dB to power for proper weighting)
        let amplitude_db = x_vqt_smoothed[bin_idx];
        let amplitude_power = 10.0_f32.powf(amplitude_db / 10.0);

        weighted_calmness_sum += calmness.get() * amplitude_power;
        weight_sum += amplitude_power;
    } else {
        calmness.update_with_timestep(0.0, frame_time);
    }
}

if weight_sum > 0.0 {
    self.smoothed_scene_calmness.update_with_timestep(
        weighted_calmness_sum / weight_sum,
        frame_time
    );
}
```

**Benefits:**
- Loud notes have more influence (matches perception)
- Quiet background notes don't dominate metric
- Still uses power domain for physically meaningful weights

**Risks:**
- May be too sensitive to loudest notes
- Could need normalization/capping

### Proposal 2: Incorporate Rate-of-Change

**Change**: Reduce calmness when amplitudes are changing rapidly.

**Concept**: Calm music has stable amplitudes; energetic music has fluctuations.

**Implementation:**
```rust
// Add to AnalysisState struct:
previous_x_vqt_smoothed: Vec<f32>,

// In update_calmness:
let mut weighted_calmness_sum = 0.0;
let mut weight_sum = 0.0;

for (bin_idx, (calmness, has_peak)) in ... {
    if *has_peak {
        let amplitude = x_vqt_smoothed[bin_idx];
        let prev_amplitude = self.previous_x_vqt_smoothed[bin_idx];

        // Rate of change (in dB per second)
        let delta_db = (amplitude - prev_amplitude).abs();
        let delta_db_per_sec = delta_db / frame_time.as_secs_f32();

        // Reduce calmness for rapidly changing amplitudes
        // Threshold: 5 dB/s is considered stable, 20 dB/s very unstable
        let stability_factor = (1.0 - (delta_db_per_sec / 20.0).min(1.0)).max(0.0);

        // Target calmness is modulated by stability
        let target_calmness = stability_factor;
        calmness.update_with_timestep(target_calmness, frame_time);

        // ... rest of weighting ...
    }
}

// Store for next frame
self.previous_x_vqt_smoothed = x_vqt_smoothed.clone();
```

**Benefits:**
- Tremolo/vibrato reduces calmness (correct!)
- Distinguishes sustained vs. fluctuating
- Natural measure of musical stability

**Risks:**
- May need tuning of thresholds
- Slight memory overhead (one vec copy per frame)

### Proposal 3: Spectral Centroid Influence

**Change**: Reduce calmness when energy is concentrated in treble.

**Concept**: High spectral centroid → bright, energetic music.

**Implementation:**
```rust
fn update_calmness(&mut self, x_vqt: &[f32], frame_time: Duration) {
    // ... existing peak detection ...

    // Calculate spectral centroid (weighted by amplitude)
    let mut centroid_num = 0.0;
    let mut centroid_den = 0.0;

    for (bin_idx, amplitude_db) in x_vqt_smoothed.iter().enumerate() {
        let amplitude_power = 10.0_f32.powf(*amplitude_db / 10.0);
        let bin_freq = self.range.min_freq * 2.0_f32.powf(bin_idx as f32 / self.range.buckets_per_octave as f32);

        centroid_num += bin_freq * amplitude_power;
        centroid_den += amplitude_power;
    }

    let spectral_centroid = if centroid_den > 0.0 {
        centroid_num / centroid_den
    } else {
        self.range.min_freq
    };

    // Map centroid to calmness multiplier
    // Bass-heavy (110 Hz): 1.0x
    // Mid (440 Hz): 0.7x
    // Treble (2000 Hz): 0.3x
    let centroid_factor = if spectral_centroid < 110.0 {
        1.0
    } else if spectral_centroid < 2000.0 {
        1.0 - 0.7 * ((spectral_centroid.log2() - 110.0_f32.log2()) / (2000.0_f32.log2() - 110.0_f32.log2()))
    } else {
        0.3
    };

    // Apply to scene calmness
    let base_scene_calmness = weighted_calmness_sum / weight_sum;
    let adjusted_scene_calmness = base_scene_calmness * centroid_factor;

    self.smoothed_scene_calmness.update_with_timestep(adjusted_scene_calmness, frame_time);
}
```

**Benefits:**
- Bright cymbals/hi-hats reduce calmness
- Bass-heavy music maintains calmness
- Matches psychoacoustic perception

**Risks:**
- May need careful tuning of frequency breakpoints
- Could interact unexpectedly with other factors

### Proposal 4: Attack Detection

**Change**: Newly detected peaks temporarily reduce calmness.

**Concept**: Percussive attacks are energetic; smooth sustains are calm.

**Implementation:**
```rust
// Add to AnalysisState:
previous_peaks: HashSet<usize>,
attack_cooldown: Vec<f32>,  // Per-bin cooldown timers

// In update_calmness:
let new_peaks = &peaks - &self.previous_peaks;  // Set difference

for bin_idx in new_peaks {
    // New peak detected = attack!
    self.attack_cooldown[bin_idx] = 1.0;  // Start cooldown
}

// Decay cooldowns
for cooldown in self.attack_cooldown.iter_mut() {
    *cooldown *= 0.9;  // Exponential decay
}

// When calculating per-bin calmness:
if *has_peak {
    // Reduce target calmness during attack cooldown
    let attack_penalty = self.attack_cooldown[bin_idx];
    let target_calmness = 1.0 - attack_penalty * 0.7;  // 70% penalty during attack
    calmness.update_with_timestep(target_calmness, frame_time);
}

self.previous_peaks = peaks.clone();
```

**Benefits:**
- Percussive instruments reduce calmness
- Smooth pads increase calmness
- Distinguishes attack character

**Risks:**
- Peak detection jitter could cause false attacks
- May need smoothing/hysteresis

### Proposal 5: Total Energy Consideration

**Change**: Very high total energy reduces calmness.

**Concept**: Loud, dense music is energetic regardless of note duration.

**Implementation:**
```rust
// Calculate total energy (sum of power across all bins)
let total_energy: f32 = x_vqt_smoothed.iter()
    .map(|db| 10.0_f32.powf(*db / 10.0))
    .sum();

// Map to energy factor
// Low energy (< 10,000): 1.0x
// Medium (10,000 - 100,000): 1.0x - 0.5x
// High (> 100,000): 0.5x
let energy_factor = if total_energy < 10000.0 {
    1.0
} else if total_energy < 100000.0 {
    1.0 - 0.5 * ((total_energy.log10() - 4.0) / 1.0)
} else {
    0.5
};

// Apply to scene calmness
let adjusted_scene_calmness = base_scene_calmness * energy_factor;
```

**Benefits:**
- Wall-of-sound passages reduce calmness
- Sparse arrangements maintain calmness
- Captures musical density

**Risks:**
- Threshold tuning depends on VQT normalization
- May need per-song calibration

### Proposal 6: Fix Release Discontinuity (Address FIXME)

**Change**: Keep calmness history for recently released notes.

**Implementation:**
```rust
// Add to AnalysisState:
released_note_calmness: Vec<EmaMeasurement>,  // Same as calmness but for released notes

// In update_calmness:
for (bin_idx, (calmness, has_peak)) in ... {
    if *has_peak {
        calmness.update_with_timestep(1.0, frame_time);
        // Also update released tracker to current value
        self.released_note_calmness[bin_idx] = *calmness;
    } else {
        calmness.update_with_timestep(0.0, frame_time);
        // Decay released note calmness
        self.released_note_calmness[bin_idx].update_with_timestep(0.0, frame_time);
    }
}

// Include recently released notes in scene calmness
// Weight them by how recently they were released
let mut total_calmness_sum = 0.0;
let mut total_weight = 0.0;

for bin_idx in 0..n_buckets {
    let active_weight = if peaks_around[bin_idx] { 1.0 } else { 0.0 };
    let released_weight = self.released_note_calmness[bin_idx].get() * 0.3;  // 30% weight for released

    total_calmness_sum += self.calmness[bin_idx].get() * (active_weight + released_weight);
    total_weight += active_weight + released_weight;
}

scene_calmness = total_calmness_sum / total_weight;
```

**Benefits:**
- Smooth transitions when notes release
- Calmness "lingers" after sustained passages
- More natural musical flow

**Risks:**
- May slow response to true energy changes
- Complexity increase

## Recommended Implementation Strategy

### Phase 1: High-Impact, Low-Risk (Implement First)

**1. Amplitude-Weighted Scene Calmness** (Proposal 1)
- Clear perceptual benefit
- Low implementation complexity
- No new state needed

**2. Fix Release Discontinuity** (Proposal 6)
- Addresses known FIXME
- Improves smoothness
- Low risk

### Phase 2: Medium-Impact, Medium-Risk (Test Thoroughly)

**3. Rate-of-Change Detection** (Proposal 2)
- Good conceptual fit
- Needs threshold tuning
- Small memory overhead

**4. Attack Detection** (Proposal 4)
- Distinguishes percussive vs sustained
- May need jitter filtering
- Moderate complexity

### Phase 3: Advanced Features (Experimental)

**5. Spectral Centroid** (Proposal 3)
- Strong theoretical basis
- Needs careful tuning
- May interact with other factors

**6. Total Energy** (Proposal 5)
- Captures density
- Threshold depends on normalization
- Most experimental

## Testing Recommendations

Test each improvement with diverse music:

1. **Calm music**:
   - Ambient pads (Brian Eno)
   - Classical largo (Bach slow movements)
   - Solo piano ballads

2. **Energetic music**:
   - Drum solos
   - EDM/dubstep drops
   - Fast percussion (Steve Reich)

3. **Edge cases**:
   - Tremolo strings (should be less calm than sustained)
   - Sustained organ (should be very calm)
   - Staccato pizzicato (should be energetic)
   - Cymbal wash (bright, should reduce calmness)

## Performance Considerations

Current: ~10-20 µs per frame for calmness calculation

Estimated overhead for each proposal:
- **Proposal 1** (Amplitude weighting): +5 µs (power conversion)
- **Proposal 2** (Rate-of-change): +10 µs (vec copy, differencing)
- **Proposal 3** (Spectral centroid): +20 µs (full spectrum iteration)
- **Proposal 4** (Attack detection): +5 µs (set operations)
- **Proposal 5** (Total energy): +15 µs (power sum)
- **Proposal 6** (Release history): +10 µs (additional EMA updates)

**Total worst case**: +65 µs per frame
**At 60 FPS**: 3.9 ms/s = 0.39% CPU

All improvements combined would still be < 0.5% of frame budget.

## Conclusion

The current calmness calculation is simple and functional, but has several limitations:

**Strengths:**
- Fast and efficient
- Captures basic sustain vs. transient character
- Works reasonably well for many use cases

**Weaknesses:**
- No amplitude weighting (all notes equal)
- Doesn't distinguish stable vs. fluctuating
- Ignores spectral distribution
- Binary peak detection
- Release discontinuities

**Recommended quick wins:**
1. Amplitude-weighted scene calmness (Proposal 1)
2. Fix release discontinuity (Proposal 6)

These two changes alone would address the most significant limitations with minimal risk and complexity.

**For comprehensive improvement:**
Implement all Phase 1 and Phase 2 proposals for a robust, perceptually-aligned calmness metric that adapts to diverse musical contexts.
