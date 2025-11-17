# Vibrato Analysis and Detection

## Overview

This document describes the design and implementation of **vibrato analysis** for PitchVis, addressing two critical needs:

1. **Singer feedback**: Analyze vibrato quality (rate, extent, regularity) for vocal technique improvement
2. **Visualization quality**: Fix "double peak" glitches by recognizing vibrato as single oscillating note

## The Double Peak Problem

### Current Behavior

When a singer uses vibrato, the frequency oscillates periodically (e.g., 440 Hz ± 25 Hz at 6 Hz rate). With current peak detection:

```
Time t=0.00s: Peak at 440 Hz (bin 880)
Time t=0.08s: Peak at 465 Hz (bin 930) - shifted up
Time t=0.16s: Peak at 415 Hz (bin 830) - shifted down
Time t=0.24s: Peak at 465 Hz (bin 930) - shifted up again
```

**Problem**: The `min_distance` constraint (0.4 semitones ≈ 40 bins) **doesn't prevent** detecting separate peaks when vibrato oscillation is large:

- Vibrato extent: ±50 cents = ±100 cents total = ±8.3 semitones = ±83 bins (at this bucket density)
- This is >> min_distance, so two separate peaks detected
- Visualization shows two flickering pitch balls instead of one vibrato note
- Visual "glitch" as peaks appear/disappear

### Root Cause

Peak detection operates on **single frame** - no temporal context. Vibrato is inherently **temporal** - it's a pattern over multiple frames.

**Solution**: Track frequency history per note, detect periodic oscillation, consolidate into single "vibrato note."

---

## Vibrato Detection Design

### Core Concepts

**Vibrato characteristics:**
- **Rate**: Oscillations per second (Hz), typically 5-7 Hz for trained singers
- **Extent**: Frequency deviation peak-to-peak (cents), typically 50-100 cents
- **Regularity**: How consistent the oscillation pattern is (0.0-1.0)

**Detection approach:**
1. Track frequency over time (1-2 second window)
2. Detect periodicity using autocorrelation
3. Measure oscillation extent
4. Classify as vibrato if matches expected characteristics

### Data Structures

```rust
/// Per-note vibrato state tracking
#[derive(Debug, Clone)]
pub struct VibratoState {
    /// Frequency history (last 120 samples ≈ 2 seconds at 60 FPS)
    pub frequency_history: VecDeque<f32>,

    /// Time history (for proper rate calculation)
    pub time_history: VecDeque<f32>,

    /// Detected vibrato rate (Hz), 0.0 if no vibrato detected
    pub rate: f32,

    /// Detected vibrato extent (cents peak-to-peak)
    pub extent: f32,

    /// Vibrato regularity score (0.0 = irregular, 1.0 = perfectly regular)
    pub regularity: f32,

    /// Is vibrato currently active?
    pub is_active: bool,

    /// Center frequency of vibrato (Hz)
    pub center_frequency: f32,

    /// Confidence in vibrato detection (0.0-1.0)
    pub confidence: f32,
}

impl VibratoState {
    pub fn new() -> Self {
        Self {
            frequency_history: VecDeque::with_capacity(120),
            time_history: VecDeque::with_capacity(120),
            rate: 0.0,
            extent: 0.0,
            regularity: 0.0,
            is_active: false,
            center_frequency: 0.0,
            confidence: 0.0,
        }
    }

    /// Check if vibrato is healthy (rate and extent in acceptable ranges)
    pub fn is_healthy(&self) -> bool {
        if !self.is_active {
            return true;  // No vibrato is fine
        }

        // Healthy vibrato: 4.5-8 Hz rate, 40-120 cents extent
        let rate_ok = self.rate >= 4.5 && self.rate <= 8.0;
        let extent_ok = self.extent >= 40.0 && self.extent <= 120.0;
        let regular_enough = self.regularity >= 0.6;

        rate_ok && extent_ok && regular_enough
    }

    /// Get vibrato category for feedback
    pub fn get_category(&self) -> VibratoCategory {
        if !self.is_active {
            return VibratoCategory::StraightTone;
        }

        if self.rate < 4.0 {
            VibratoCategory::Wobble  // Too slow
        } else if self.rate > 8.5 {
            VibratoCategory::Tremolo  // Too fast
        } else if self.extent > 120.0 {
            VibratoCategory::Excessive  // Too wide
        } else if self.extent < 30.0 {
            VibratoCategory::Minimal  // Too narrow
        } else {
            VibratoCategory::Healthy
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VibratoCategory {
    StraightTone,   // No vibrato
    Healthy,        // Good vibrato
    Wobble,         // Too slow
    Tremolo,        // Too fast
    Excessive,      // Too wide
    Minimal,        // Too narrow
}

/// Public vibrato analysis for a detected peak/note
#[derive(Debug, Clone, Copy)]
pub struct VibratoAnalysis {
    /// Vibrato rate in Hz (0.0 if none detected)
    pub rate: f32,

    /// Vibrato extent in cents peak-to-peak
    pub extent: f32,

    /// Regularity score (0.0-1.0)
    pub regularity: f32,

    /// Is vibrato present?
    pub is_present: bool,

    /// Center frequency (Hz) of the vibrating note
    pub center_frequency: f32,

    /// Vibrato health category
    pub category: VibratoCategory,
}
```

### Vibrato Detection Algorithm

#### Step 1: Frequency History Tracking

```rust
fn update_vibrato_state(
    &mut self,
    bin_idx: usize,
    current_freq: f32,
    current_time: f32,
    is_peak_active: bool,
) {
    let state = &mut self.vibrato_states[bin_idx];

    if is_peak_active {
        // Add to history
        state.frequency_history.push_back(current_freq);
        state.time_history.push_back(current_time);

        // Keep last 2 seconds (120 samples at 60 FPS)
        if state.frequency_history.len() > 120 {
            state.frequency_history.pop_front();
            state.time_history.pop_front();
        }

        // Need at least 0.5 seconds of data to detect vibrato
        if state.frequency_history.len() >= 30 {
            self.analyze_vibrato(bin_idx);
        }
    } else {
        // Peak not active - decay history
        if !state.frequency_history.is_empty() {
            // Keep history for a short time (0.2s) in case peak re-appears
            if state.time_history.back().unwrap() - current_time > 0.2 {
                state.frequency_history.clear();
                state.time_history.clear();
                state.is_active = false;
            }
        }
    }
}
```

#### Step 2: Autocorrelation for Periodicity Detection

```rust
fn analyze_vibrato(&mut self, bin_idx: usize) {
    let state = &mut self.vibrato_states[bin_idx];

    // Convert frequency history to cents deviation from mean
    let mean_freq: f32 = state.frequency_history.iter().sum::<f32>()
                         / state.frequency_history.len() as f32;
    state.center_frequency = mean_freq;

    let cents_history: Vec<f32> = state.frequency_history.iter()
        .map(|f| 1200.0 * (f / mean_freq).log2())
        .collect();

    // Autocorrelation to find periodicity
    let (vibrato_period, correlation_strength) =
        self.find_vibrato_period(&cents_history, &state.time_history);

    if correlation_strength > 0.5 {
        // Vibrato detected!
        state.is_active = true;
        state.rate = 1.0 / vibrato_period;  // Convert period to Hz
        state.regularity = correlation_strength;

        // Measure extent (peak-to-peak deviation)
        let max_cents = cents_history.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_cents = cents_history.iter().cloned().fold(f32::INFINITY, f32::min);
        state.extent = max_cents - min_cents;

        // Confidence based on history length and correlation
        state.confidence = (state.frequency_history.len() as f32 / 120.0).min(1.0)
                         * correlation_strength;
    } else {
        // No clear periodicity - not vibrato
        state.is_active = false;
        state.rate = 0.0;
        state.extent = 0.0;
    }
}
```

#### Step 3: Autocorrelation Implementation

```rust
fn find_vibrato_period(
    &self,
    cents_history: &[f32],
    time_history: &VecDeque<f32>,
) -> (f32, f32) {
    // Remove DC component (mean)
    let mean: f32 = cents_history.iter().sum::<f32>() / cents_history.len() as f32;
    let centered: Vec<f32> = cents_history.iter().map(|x| x - mean).collect();

    // Expected vibrato period range: 4-8 Hz → 0.125-0.25 seconds
    // At 60 FPS: 7.5-15 samples per cycle
    let min_lag = 7;   // ~8 Hz max rate
    let max_lag = 30;  // ~2 Hz min rate (generous)

    let mut best_lag = 0;
    let mut best_correlation = 0.0;

    for lag in min_lag..=max_lag {
        if lag >= centered.len() {
            break;
        }

        // Autocorrelation at this lag
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..(centered.len() - lag) {
            sum += centered[i] * centered[i + lag];
            count += 1;
        }

        let correlation = sum / count as f32;

        // Normalize by variance
        let variance: f32 = centered.iter().map(|x| x * x).sum::<f32>()
                            / centered.len() as f32;
        let normalized_correlation = correlation / variance.max(1.0);

        if normalized_correlation > best_correlation {
            best_correlation = normalized_correlation;
            best_lag = lag;
        }
    }

    // Convert lag to period in seconds
    let period = if best_lag > 0 && best_correlation > 0.5 {
        let time_start = time_history[0];
        let time_end = time_history[best_lag];
        time_end - time_start
    } else {
        0.0
    };

    (period, best_correlation)
}
```

---

## Vibrato-Aware Peak Consolidation

### Problem: Multiple Peaks for One Vibrato Note

Current peak detection sees vibrato as multiple distinct peaks. Solution: **consolidate** nearby oscillating peaks into single "vibrato note."

### Consolidation Algorithm

```rust
fn consolidate_vibrato_peaks(
    &mut self,
    peaks_continuous: &mut Vec<ContinuousPeak>,
) {
    // Group peaks that might be same note with vibrato
    let mut consolidated = Vec::new();
    let mut used = vec![false; peaks_continuous.len()];

    for i in 0..peaks_continuous.len() {
        if used[i] {
            continue;
        }

        let peak = peaks_continuous[i];
        let bin_idx = peak.center.round() as usize;
        let vibrato = &self.vibrato_states[bin_idx];

        if vibrato.is_active && vibrato.confidence > 0.7 {
            // This is a vibrato note - find all peaks within vibrato extent
            let extent_bins = vibrato.extent / 100.0 * self.range.buckets_per_octave as f32 / 12.0;
            let search_range = extent_bins * 1.2;  // 20% margin

            // Find all peaks within vibrato extent
            let mut group = vec![i];
            for j in (i+1)..peaks_continuous.len() {
                if used[j] {
                    continue;
                }

                let other_peak = peaks_continuous[j];
                let distance = (other_peak.center - peak.center).abs();

                if distance <= search_range {
                    // Check if this peak's frequency is in this vibrato's range
                    let other_freq = self.bin_to_frequency(other_peak.center as usize);
                    let center_freq = vibrato.center_frequency;
                    let freq_deviation = 1200.0 * (other_freq / center_freq).log2();

                    if freq_deviation.abs() <= vibrato.extent / 2.0 {
                        group.push(j);
                        used[j] = true;
                    }
                }
            }

            // Create consolidated peak at vibrato center
            let consolidated_peak = ContinuousPeak {
                center: self.frequency_to_bin(vibrato.center_frequency),
                size: peak.size,  // Keep original amplitude
            };

            consolidated.push(consolidated_peak);
            used[i] = true;
        } else {
            // Not vibrato - keep as-is
            consolidated.push(peak);
            used[i] = true;
        }
    }

    *peaks_continuous = consolidated;
}

fn frequency_to_bin(&self, frequency: f32) -> f32 {
    self.range.buckets_per_octave as f32 * (frequency / self.range.min_freq).log2()
}
```

### Integration into Preprocessing

```rust
pub fn preprocess(&mut self, x_vqt: &[f32], frame_time: Duration) {
    // ... existing smoothing, peak detection ...

    // Update vibrato state for all active peaks
    let current_time = self.accumulated_time;
    for peak in &peaks_continuous {
        let bin_idx = peak.center.round() as usize;
        let freq = self.bin_to_frequency(bin_idx);
        self.update_vibrato_state(bin_idx, freq, current_time, true);
    }

    // Consolidate peaks that are part of same vibrato note
    self.consolidate_vibrato_peaks(&mut peaks_continuous);

    // ... rest of processing ...

    self.accumulated_time += frame_time.as_secs_f32();
}
```

---

## Visualization Enhancements

### 1. Vibrato Indicator

```rust
// In viewer code:
for (bin_idx, vibrato) in analysis.vibrato_states.iter().enumerate() {
    if vibrato.is_active {
        // Draw wavy outline around pitch ball
        let wave_amplitude = vibrato.extent / 100.0;  // Scale to visual units
        let wave_frequency = vibrato.rate;

        draw_wavy_outline(
            pitch_ball,
            wave_amplitude,
            wave_frequency,
            vibrato.regularity,
        );

        // Color code by health
        match vibrato.get_category() {
            VibratoCategory::Healthy => pitch_ball.outline_color = Color::Green,
            VibratoCategory::Wobble => pitch_ball.outline_color = Color::Orange,
            VibratoCategory::Tremolo => pitch_ball.outline_color = Color::Red,
            VibratoCategory::Excessive => pitch_ball.outline_color = Color::Yellow,
            _ => {}
        }
    }
}
```

### 2. Vibrato Waveform Display

```rust
pub struct VibratoWaveform {
    /// Frequency history to display (last 1 second)
    pub frequency_points: Vec<(f32, f32)>,  // (time, frequency_cents)

    /// Center line (target frequency)
    pub center_line: f32,

    /// Upper/lower bounds based on extent
    pub upper_bound: f32,
    pub lower_bound: f32,
}

// Draw scrolling waveform
fn draw_vibrato_waveform(vibrato: &VibratoState) {
    for (i, freq) in vibrato.frequency_history.iter().enumerate() {
        let time = vibrato.time_history[i];
        let cents = 1200.0 * (freq / vibrato.center_frequency).log2();

        draw_point(time, cents, Color::Cyan);
    }

    // Draw center reference line
    draw_line(0.0, 1.0, 0.0, 0.0, Color::White);

    // Draw extent boundaries
    let half_extent = vibrato.extent / 2.0;
    draw_line(0.0, 1.0, half_extent, half_extent, Color::Yellow);
    draw_line(0.0, 1.0, -half_extent, -half_extent, Color::Yellow);
}
```

### 3. Vibrato Meter Display

```rust
pub struct VibratoMeter {
    pub rate_display: String,        // "6.2 Hz"
    pub extent_display: String,      // "75 cents"
    pub regularity_bar: f32,         // 0.0-1.0
    pub health_indicator: Color,     // Green/Yellow/Red
    pub category_label: String,      // "Healthy" / "Wobble" / etc.
}

impl VibratoMeter {
    pub fn from_vibrato(vibrato: &VibratoState) -> Self {
        let health_color = match vibrato.get_category() {
            VibratoCategory::Healthy => Color::Green,
            VibratoCategory::StraightTone => Color::Gray,
            VibratoCategory::Wobble | VibratoCategory::Tremolo => Color::Orange,
            VibratoCategory::Excessive | VibratoCategory::Minimal => Color::Yellow,
        };

        Self {
            rate_display: if vibrato.is_active {
                format!("{:.1} Hz", vibrato.rate)
            } else {
                "No vibrato".to_string()
            },
            extent_display: if vibrato.is_active {
                format!("{:.0} cents", vibrato.extent)
            } else {
                "—".to_string()
            },
            regularity_bar: vibrato.regularity,
            health_indicator: health_color,
            category_label: format!("{:?}", vibrato.get_category()),
        }
    }
}
```

---

## Parameters and Tuning

### AnalysisParameters Addition

```rust
pub struct AnalysisParameters {
    // ... existing parameters ...

    /// Vibrato detection parameters
    vibrato_detection_config: VibratoDetectionParameters,
}

#[derive(Debug, Clone)]
pub struct VibratoDetectionParameters {
    /// Minimum vibrato rate to detect (Hz)
    pub min_rate: f32,              // Default: 2.0

    /// Maximum vibrato rate to detect (Hz)
    pub max_rate: f32,              // Default: 10.0

    /// Minimum correlation for vibrato detection
    pub min_correlation: f32,       // Default: 0.5

    /// Minimum extent to consider as vibrato (cents)
    pub min_extent: f32,            // Default: 20.0

    /// History window duration (seconds)
    pub history_duration: f32,      // Default: 2.0

    /// Enable vibrato peak consolidation?
    pub consolidate_peaks: bool,    // Default: true
}

impl Default for VibratoDetectionParameters {
    fn default() -> Self {
        Self {
            min_rate: 2.0,
            max_rate: 10.0,
            min_correlation: 0.5,
            min_extent: 20.0,
            history_duration: 2.0,
            consolidate_peaks: true,
        }
    }
}
```

---

## Performance Analysis

### Computational Cost

**Per-frame overhead:**

1. **Frequency tracking**: ~30 active notes × 2 push_back operations = 60 ops (~0.2 µs)

2. **Vibrato analysis**: ~10 notes with sufficient history × autocorrelation
   - Autocorrelation: 30 lags × 90 samples = 2,700 multiplications
   - Per note: ~2,700 ops (~1.5 µs)
   - Total: 10 notes × 1.5 µs = **15 µs**

3. **Peak consolidation**: ~5 vibrato notes × 10 peak comparisons = 50 ops (~0.2 µs)

**Total: ~15.5 µs per frame = 0.09% of frame budget at 60 FPS**

**Optimization opportunities:**
- Only analyze vibrato every 2-3 frames (saves 66%)
- Use FFT instead of autocorrelation (faster for long histories)
- Skip analysis if history too short

### Memory Impact

```rust
struct VibratoState {
    frequency_history: VecDeque<f32>,  // 120 × 4 = 480 bytes
    time_history: VecDeque<f32>,       // 120 × 4 = 480 bytes
    // Other fields: ~40 bytes
}
// Total: ~1000 bytes per bin
```

**Total memory**: 588 bins × 1 KB = **588 KB**

This is acceptable but could be optimized:
- Only track vibrato for active peaks (reduce to ~30 states = 30 KB)
- Use smaller history for less important bins

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_vibrato_detection_sine_wave() {
    // Generate perfect vibrato: 440 Hz ± 20 Hz at 6 Hz rate
    let mut state = VibratoState::new();
    let base_freq = 440.0;
    let extent_hz = 20.0;
    let rate = 6.0;

    for i in 0..120 {
        let time = i as f32 / 60.0;  // 60 FPS
        let freq = base_freq + extent_hz * (2.0 * PI * rate * time).sin();
        state.update(freq, time);
    }

    state.analyze();

    assert!(state.is_active);
    assert!((state.rate - 6.0).abs() < 0.5);  // Within 0.5 Hz
    assert!((state.extent - 79.0).abs() < 10.0);  // ~79 cents for ±20 Hz at 440 Hz
    assert!(state.regularity > 0.8);  // Very regular
}

#[test]
fn test_vibrato_consolidation() {
    // Create two peaks that are part of same vibrato
    let peaks = vec![
        ContinuousPeak { center: 880.0, size: 50.0 },  // Low end of vibrato
        ContinuousPeak { center: 920.0, size: 48.0 },  // High end of vibrato
    ];

    let mut analysis = AnalysisState::new(...);
    // Set up vibrato state for this note
    analysis.vibrato_states[880].is_active = true;
    analysis.vibrato_states[880].center_frequency = 450.0;  // Mid-point
    analysis.vibrato_states[880].extent = 75.0;  // Large enough to span both peaks

    let consolidated = analysis.consolidate_vibrato_peaks(&peaks);

    // Should consolidate to single peak
    assert_eq!(consolidated.len(), 1);
    assert!((consolidated[0].center - 900.0).abs() < 10.0);  // At center
}
```

### Integration Tests

**Test scenarios:**
1. **Singer with healthy vibrato**: Verify rate, extent detected correctly
2. **Wobble (too slow)**: Verify categorized as Wobble
3. **Tremolo (too fast)**: Verify categorized as Tremolo
4. **Straight tone**: Verify no vibrato detected
5. **Multiple vibrato notes**: Each tracked independently

---

## Future Enhancements

### 1. Vibrato Onset Detection

Track when vibrato starts/stops:
```rust
pub vibrato_onset_time: Option<f32>,
pub vibrato_offset_time: Option<f32>,
```

Useful for detecting "delayed vibrato" (common technique where vibrato starts after note onset).

### 2. Vibrato Shape Analysis

Analyze waveform shape (sine vs. sawtooth vs. asymmetric):
```rust
pub vibrato_symmetry: f32,  // 0.5 = symmetric, <0.5 = descending bias
pub vibrato_shape: VibratoShape,  // Sinusoidal, Sawtooth, Triangular
```

### 3. Coupled Vibrato Detection

Detect when multiple singers' vibratos are synchronized (choir analysis):
```rust
pub fn detect_coupled_vibrato(
    vibratos: &[VibratoState],
) -> Option<CoupledVibratoAnalysis> {
    // Check if multiple notes have synchronized vibrato
    // Useful for choir blend analysis
}
```

### 4. Vibrato Adaptivity

Use vibrato detection to inform other analysis:
- Disable attack detection during vibrato (not new note, just oscillation)
- Adjust calmness calculation (vibrato is sustained, not energetic)
- Inform harmonic analysis (vibrato can blur harmonics)

---

## Conclusion

Vibrato analysis provides dual benefits:

**For Singers:**
- ✅ Real-time feedback on vibrato rate, extent, regularity
- ✅ Health categorization (Healthy, Wobble, Tremolo, etc.)
- ✅ Visual waveform display for technique improvement
- ✅ Helps develop consistent, controlled vibrato

**For Visualization:**
- ✅ **Fixes double peak glitch** by consolidating vibrato into single note
- ✅ Smoother, more stable visualization
- ✅ Accurate representation of musical intent
- ✅ Enables vibrato-specific visual effects (wavy outline, etc.)

**Performance:**
- Minimal overhead: ~15 µs per frame (0.09%)
- Acceptable memory: ~588 KB (optimizable to ~30 KB)
- Can be further optimized by analyzing every 2-3 frames

**Key Innovation**: Using temporal context to solve spatial problem - recognizing that what appears as "multiple peaks" in single frame is actually "single vibrating note" across time.
