/// Attack detection and percussion classification module.
///
/// This module detects note onsets (attacks) and classifies them as percussive or melodic
/// based on their attack and decay characteristics.

use std::collections::{HashSet, VecDeque};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct AttackDetectionParameters {
    /// Minimum rate of amplitude increase to detect attack (dB/s)
    pub min_attack_rate: f32,
    /// Minimum amplitude for attack detection (dB)
    pub min_attack_amplitude: f32,
    /// Minimum amplitude jump to detect attack (dB)
    pub min_attack_delta: f32,
    /// Cooldown period between attacks on same bin (seconds)
    pub attack_cooldown: f32,
    /// Duration of attack phase after onset (seconds)
    pub attack_phase_duration: f32,
    /// Decay rate threshold for percussion classification (dB/s)
    pub percussion_decay_threshold: f32,
    /// Attack rate threshold for percussion classification (dB/s)
    pub percussion_attack_threshold: f32,
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
        }
    }
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
}

impl AttackState {
    pub fn new() -> Self {
        Self {
            time_since_attack: 1.0, // Start at 1s (no recent attack)
            attack_amplitude: 0.0,
            previous_amplitude: 0.0,
            in_attack_phase: false,
            amplitude_history: VecDeque::with_capacity(10),
        }
    }
}

impl Default for AttackState {
    fn default() -> Self {
        Self::new()
    }
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
    /// Percussion score for this attack (0.0 = melodic, 1.0 = percussive)
    pub percussion_score: f32,
}

/// Detect attack events (note onsets) across all frequency bins
pub fn detect_attacks(
    x_vqt_smoothed: &[f32],
    frame_time: Duration,
    min_freq: f32,
    buckets_per_octave: u16,
    peaks: &HashSet<usize>,
    n_buckets: usize,
    params: &AttackDetectionParameters,
    attack_state: &mut [AttackState],
    percussion_score: &mut [f32],
) -> Vec<AttackEvent> {
    let dt = frame_time.as_secs_f32();
    let mut current_attacks = Vec::new();

    for (bin_idx, state) in attack_state.iter_mut().enumerate() {
        let current_amp = x_vqt_smoothed[bin_idx];
        let amp_change = current_amp - state.previous_amplitude;
        let rate_of_change = amp_change / dt.max(0.001); // dB/s

        // Update time since last attack
        state.time_since_attack += dt;

        // ATTACK DETECTION CRITERIA
        let attack_detected = rate_of_change > params.min_attack_rate &&           // Rapid increase
            current_amp > params.min_attack_amplitude &&         // Above noise floor
            amp_change > params.min_attack_delta &&              // Minimum delta
            state.time_since_attack > params.attack_cooldown; // Cooldown period

        if attack_detected {
            // Record attack event
            state.time_since_attack = 0.0;
            state.attack_amplitude = current_amp;
            state.in_attack_phase = true;

            // Calculate percussion score inline to avoid borrowing issues
            let percussion = calculate_percussion_score(state, rate_of_change, params);
            percussion_score[bin_idx] = percussion;

            // Calculate frequency inline
            let frequency = min_freq * 2.0_f32.powf(bin_idx as f32 / buckets_per_octave as f32);

            current_attacks.push(AttackEvent {
                bin_idx,
                frequency,
                amplitude: current_amp,
                attack_rate: rate_of_change,
                percussion_score: percussion,
            });
        }

        // Exit attack phase after configured duration
        if state.time_since_attack > params.attack_phase_duration {
            state.in_attack_phase = false;
        }

        // Update amplitude history (keep last 10 values)
        state.amplitude_history.push_back(current_amp);
        if state.amplitude_history.len() > 10 {
            state.amplitude_history.pop_front();
        }

        // Store for next frame
        state.previous_amplitude = current_amp;
    }

    // Filter attacks: only keep those that coincide with actual peaks
    // This reduces false positives from noise
    current_attacks.retain(|attack| {
        let start = attack.bin_idx.saturating_sub(2);
        let end = (attack.bin_idx + 2).min(n_buckets - 1);
        (start..=end).any(|b| peaks.contains(&b))
    });

    current_attacks
}

/// Calculate percussion score for an attack
fn calculate_percussion_score(
    state: &AttackState,
    attack_rate: f32,
    params: &AttackDetectionParameters,
) -> f32 {
    let mut percussion_score = 0.0;
    let mut factor_count = 0;

    // FACTOR 1: Decay Rate
    // Percussion decays quickly after attack
    if state.amplitude_history.len() >= 5 {
        let recent_amps: Vec<f32> = state
            .amplitude_history
            .iter()
            .rev()
            .take(5)
            .copied()
            .collect();

        let decay_rate = calculate_decay_rate(&recent_amps);

        // High decay rate = percussive (> 100 dB/s)
        // Low decay rate = sustained (< 20 dB/s)
        let decay_factor = (decay_rate / params.percussion_decay_threshold).min(1.0);
        percussion_score += decay_factor;
        factor_count += 1;
    }

    // FACTOR 2: Attack Sharpness
    // Very sharp attacks (> 200 dB/s) are percussive
    // Slow attacks (< 50 dB/s) are melodic
    let attack_factor = ((attack_rate - params.min_attack_rate)
        / (params.percussion_attack_threshold - params.min_attack_rate))
        .clamp(0.0, 1.0);
    percussion_score += attack_factor;
    factor_count += 1;

    // Average all factors
    if factor_count > 0 {
        (percussion_score / factor_count as f32).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Calculate decay rate from amplitude history (dB/s)
/// Returns positive value for decay (amplitude decreasing)
fn calculate_decay_rate(amplitudes: &[f32]) -> f32 {
    if amplitudes.len() < 2 {
        return 0.0;
    }

    // Simple linear regression: find slope of amplitude vs. sample index
    // We approximate time using frame count (assumes ~60 FPS)
    let n = amplitudes.len() as f32;
    let frame_duration = 1.0 / 60.0; // Approximate frame time

    let sum_a: f32 = amplitudes.iter().sum();
    let sum_t: f32 = (0..amplitudes.len())
        .map(|i| i as f32 * frame_duration)
        .sum();
    let sum_ta: f32 = amplitudes
        .iter()
        .enumerate()
        .map(|(i, a)| i as f32 * frame_duration * a)
        .sum();
    let sum_tt: f32 = (0..amplitudes.len())
        .map(|i| {
            let t = i as f32 * frame_duration;
            t * t
        })
        .sum();

    let slope = (n * sum_ta - sum_t * sum_a) / (n * sum_tt - sum_t * sum_t);

    // Return absolute decay rate (negative slope = decay)
    -slope
}
