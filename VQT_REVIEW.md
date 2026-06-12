# VQT/CQT Code Review — pitchvis_analysis

*Reviewed: `pitchvis_analysis/src/vqt.rs` (core), plus its consumers (`analysis.rs`,
`analysis_modules/`, `pitchvis_viewer/src/vqt_system.rs`) and the legacy copy in
`pitchvis_wasm_kiss3d/src/cqt.rs`. All 8 tests + 3 doc-tests pass in release mode.*

## Executive summary

The implementation is mathematically sound and well-architected. The multi-rate design
(per-group frequency-domain decimation + sparse kernels) is the right approach, the
Parseval-based kernel application is correct, and the magnitude normalization is consistent
across bins (verified by `test_vqt_high_frequencies`). Measured cost is **0.165 ms/frame**
(desktop release, 588 bins, n_fft=32768) — about 1% of a 60 FPS frame budget, so desktop
performance is a non-issue; the optimizations below matter for **Android and WASM**.

Headline findings:

1. **Bug:** the bandwidth-gap validation in `vqt_kernel()` is dead code — it can never fire
   (vqt.rs:786–801).
2. **Perf:** groups sharing the same analysis window recompute the identical FFT — 2× for the
   8192 window, 4× for the 4096 window. Deduplicating cuts FFT work ~57%; switching to a
   real-input FFT halves it again (combined ≈ 4.7× less FFT work).
3. **Theory:** with the default γ=8.64, the transform is effectively *constant-bandwidth*
   (STFT-like) below the αf=γ crossover at ≈1050 Hz, and only approaches constant-Q above
   ~2 kHz. This is a defensible latency trade-off but should be documented — it explains why
   bass-note discrimination depends on the harmonic-promotion heuristic.
4. The γ = 4.8·Q coupling pins the analysis delay at ≈ 1/(2·4.8) s ≈ 104 ms *independent of Q*
   — which means the known FIXME (Q too high for full coverage at high frequencies) can be
   fixed almost latency-free by **lowering** Q to ≈ 1.6.
   *(Correction: the first version of this report said "raising"; that was backwards. Higher
   Q narrows the filters and widens the coverage gaps — the code's FIXME had it right.)*

---

## 1. Correctness

### 1.1 Dead bandwidth-gap check (bug) — `vqt.rs:769–801`

```rust
let mut last_upper_bandwidth = 0.0;
...
if last_upper_bandwidth != 0.0 && filter.bandwidth_3db_in_hz.0 > last_upper_bandwidth {
    error!(...);
    last_upper_bandwidth = filter.bandwidth_3db_in_hz.1;   // only updated INSIDE the if
}
```

`last_upper_bandwidth` starts at 0.0 and is only assigned *inside* the branch guarded by
`last_upper_bandwidth != 0.0`, so it stays 0.0 forever and the error can never log. The doc
comment on `vqt_kernel` ("it is checked that at high frequencies the filters cover the entire
band … errors are logged") is therefore not true. Fix: move the assignment after the `if`,
unconditionally. Note the check is also per-group (reset every group closure) — to validate
across group boundaries, hoist the variable out of the per-group closure.

Once fixed, expect it to actually fire at the top octave (see §3.3) — decide whether that
should stay `error!` or become `warn!` with the measured gap percentage.

### 1.2 `calculated_q` is a placeholder — `vqt.rs:320, 832`

`VqtKernel.calculated_q` is set to the literal `1.0` and never read anywhere in the workspace.
Either compute the real effective Q (see §3.2 — it's ~130, not 1.8, and not 1.0) or delete the
field.

### 1.3 No input-length validation — `vqt.rs:933`

`calculate_vqt_instant_in_db(&self, x: &[f32])` assumes `x.len() == n_fft`. The window
positions (`unscaled_window.window`) are absolute indices relative to an n_fft-sized buffer
whose *end* is "now". A shorter input panics on slicing (acceptable), but a **longer** input
silently analyzes the wrong (older) samples. Add
`assert_eq!(x.len(), self.params.n_fft)` at the top.

### 1.4 Unguarded index arithmetic in filter construction — `vqt.rs:607–609`

`v_frequency_domain[scaled_window_center_rounded - scaled_window_length_rounded / 2 + i]`
can underflow `usize` for parameter combinations where a filter's window doesn't fit before
the window center inside its group window. It's safe under current defaults (verified: the
tightest case is group 0, indices 29..63 of 64), but this is exactly the kind of invariant
that breaks silently during parameter tuning. Add a checked computation or an assert with a
message explaining the constraint.

### 1.5 dB values squared as "power" — `pitch_analysis.rs:57`

`update_tuning_inaccuracy` weights by `p.size * p.size`, but `ContinuousPeak.size` is in dB
(its own doc says `/// The estimated precise amplitude of the peak, in ???`). Squaring dB is
not power weighting — a 40 dB peak gets 4× the weight of a 20 dB peak instead of 100×. The
neighboring modules (`promote_bass_peaks_with_harmonics`, `update_calmness`) do this
correctly via `10^(dB/10)`. Recommend the same here, and fix the `???` doc while at it.

### 1.6 Minor

- `find_3db_points` (vqt.rs:1056) returns one-past-threshold indices and clamps at array
  edges; the comment already admits it's crude. Fine for diagnostics — just don't reuse it
  for anything load-bearing.
- `partial_cmp().unwrap()` in the sparsity sort (vqt.rs:643) → use `total_cmp` (NaN-proof).
- `power_to_db` mixes two normalization behaviors (see §3.6).

---

## 2. Performance

Measured baseline (Linux desktop, release, default params): **0.165 ms/frame**, delay 98 ms.
Per-frame work is dominated by forward FFTs in `resample()`. The kernel mat-vec is negligible
(total nnz = 19,290 ≈ 19k complex MACs).

Actual group structure (from debug logs):

| group | downsample M | window (abs) | unscaled FFT size | filters | scaled bins | nnz | density |
|------:|---:|---|---:|---:|---:|---:|---:|
| 0 | 128 | (24576, 32768) | 8192 | 38 | 64 | 1122 | 46.1% |
| 1 | 64 | (24576, 32768) | 8192 | 84 | 128 | 2959 | 27.5% |
| 2 | 32 | (28537, 32633) | 4096 | 84 | 128 | 1719 | 16.0% |
| 3 | 16 | (28537, 32633) | 4096 | 84 | 256 | 2028 | 9.4% |
| 4 | 8 | (28537, 32633) | 4096 | 84 | 512 | 2661 | 6.2% |
| 5 | 4 | (28537, 32633) | 4096 | 84 | 1024 | 3983 | 4.6% |
| 6 | 2 | (29561, 31609) | 2048 | 84 | 1024 | 3353 | 3.9% |
| 7 | 1 | (30073, 31097) | 1024 | 46 | 1024 | 1465 | 3.1% |

### 2.1 Deduplicate FFTs per distinct window (biggest win)

Groups 0–1 FFT the *identical* 8192-sample slice twice per frame; groups 2–5 FFT the
identical 4096-sample slice four times. Since the frequency-domain decimation is just bin
extraction (`X_decimated[k] = X[k] / M` — your derivation in the comment is correct), all
groups sharing a window size can read from **one** FFT of that window.

Two implementation levels:

- **Easy:** cache the FFT output per distinct window in `calculate_vqt_instant_in_db` and
  run `resample`'s extraction step per group from the cached spectrum.
- **Cleaner:** eliminate the extraction entirely by remapping kernel *column indices* at
  construction time: a decimated bin `j` of group with factor M corresponds to full-spectrum
  bin `j` (positive frequencies) or `N - (N/M - j)` (negative frequencies), with the `1/M`
  factor folded into the kernel values. Then groups 0–1 merge into one sparse matrix over the
  8192-spectrum and groups 2–5 into one over the 4096-spectrum: per frame = 4 FFTs + 4
  mat-vecs, no intermediate vectors at all.

Estimated FFT work (∝ N·log₂N): current ≈ 442k units → deduped ≈ 188k units (**−57%**).

### 2.2 Real-input FFT

The input is real, but `resample()` converts to `Complex32` and runs complex FFTs
(vqt.rs:885–893). Using `realfft` (rustfft's companion crate) halves both the FFT cost and
the conversion/allocation, and yields only the positive half-spectrum — which is exactly what
the kernel needs *if* you also fold the negative-frequency kernel columns onto their positive
conjugate partners (for real input, `X[N−k] = X*[k]`, so the contribution of a negative-
frequency kernel coefficient `c` at column `N−k` equals `(c·X*[k])` — representable by adding
`conj(c)`'s effect; in practice: keep accumulating in complex, then
`coef_pos[k] += conj(coef_neg[N−k])` only works if you take magnitudes at the end, which you
do — but verify with the sweep test). Combined with §2.1: ≈ **4.7× less FFT work**.

### 2.3 Dead work in `resample()` — `vqt.rs:896–899`

The zeroing loop ("low-pass filter for anti-aliasing") zeroes exactly the bins that the
subsequent extraction step never reads. The extraction *is* the low-pass. Delete the loop —
it's O(window) wasted per group per frame, and it's conceptually misleading.

### 2.4 Per-frame allocation churn (~0.5–0.8 MB/frame ≈ 30–50 MB/s at 60 FPS)

Every frame allocates: the n_fft input copy (`vqt_system.rs:52`, 128 KB), per group a complex
conversion buffer + rustfft scratch (`Fft::process()` allocates scratch *on every call* —
use `process_with_scratch` with a preallocated buffer), two result vectors per group
(`resample` builds `result`, then `.map(|z| z / M).collect()` builds a second one —
vqt.rs:906–917), plus `x_vqt`, `power`, `log_spec`. None of this is needed:

- Give `Vqt` (or a separate `VqtScratch` to keep `&self` shareable) preallocated buffers.
- Fold the `1/M` scaling into the kernel values at construction (removes the second vector).
- Let `calculate_vqt_instant_in_db` write into a `&mut [f32]` out-param; the viewer keeps
  `VqtResultResource.x_vqt` allocated already.

This is the difference between "fine on desktop" and "GC-pressure-free on Android/WASM".

### 2.5 Micro

- `z.abs() * z.abs()` → `z.norm_sqr()` (vqt.rs:994): avoids 588 sqrt per frame, free win.
- `power_to_db` makes three passes (map, max, min+clamp); fusable into two. Marginal.
- Init-time only (don't bother unless it annoys): full `sort_by` for the sparsity quantile
  (vqt.rs:643) could be `select_nth_unstable`; the O(G²) `partition` loop in
  `group_window_sizes` (vqt.rs:521–530) could be `slice::chunk_by`.

### 2.6 Pipeline integration

`update_vqt` copies 128 KB out of the ring buffer while holding the same mutex the audio
callback thread blocks on. The memcpy is ~10 µs so it's not a real problem today, but if you
ever see audio overruns on Android, this lock is the first suspect (a triple-buffer or
`arc-swap` snapshot would decouple it).

### 2.7 Longer-term options (only if mobile/WASM profiling demands it)

- **Recursive octave decimation** (Schörkhuber–Klapuri toolbox style): one kernel for the top
  octave, cascade halfband-decimate the signal, reuse the kernel per octave. Cuts kernel
  memory and FFT work further, at the cost of restructuring.
- **GPU compute** (Bevy compute shader for FFT+matvec) — almost certainly not worth it at
  0.165 ms, and hostile to WASM.
- A `criterion` bench in `pitchvis_analysis/benches/` would keep these numbers honest; the
  stale `flamegraph.svg`/`perf.data` (Oct 2024, dominated by `sinf` from the test-signal
  generator, untracked) should be deleted or regenerated against the real pipeline.

---

## 3. Theory

### 3.1 The default parameters make a "constant-bandwidth-then-constant-Q" hybrid

Bandwidth per bin is Δf ≈ 1.44·(αf + γ)/Q (Hann −3 dB main lobe ≈ 1.44/T). With α ≈ 0.00825,
γ = 8.64, Q = 1.8:

| f | Δf | in semitones |
|---:|---:|---:|
| 55 Hz | 7.3 Hz | ≈ 2.2 st |
| 220 Hz | 8.4 Hz | ≈ 0.65 st |
| 440 Hz | 9.8 Hz | ≈ 0.38 st |
| 1048 Hz (αf = γ crossover) | 13.8 Hz | ≈ 0.23 st |
| 6985 Hz | 53 Hz | ≈ 0.13 st |

Below ~1 kHz the γ term dominates: the transform is effectively an STFT with ~8 Hz constant
bandwidth. Adjacent bass *semitones* (3.3 Hz apart at A1) are fundamentally unresolvable as
separate peaks below roughly 300 Hz — consistent with `test_vqt_close_frequencies` starting
at min_freq + 2.5 octaves (~311 Hz). The system compensates with
`promote_bass_peaks_with_harmonics`, which is the right move (it's also how human pitch
perception handles low fundamentals). **Recommendation:** state this crossover explicitly in
the module docs; it's the single most important fact for anyone tuning `gamma`. For
reference, the psychoacoustic (ERB-matched) γ from Schörkhuber et al. would be ≈ 1.9 for
these parameters — the chosen 8.64 is ~4.5× more aggressive, trading bass frequency
resolution for the <100 ms latency target.

### 3.2 γ = 4.8·Q pins latency — changing Q is almost free

Delay = w(f_min)/2sr = Q/(2·(α·f_min + 4.8·Q)) → as the γ term dominates,
delay → 1/(2·4.8) ≈ 104 ms regardless of Q (Q=1.8 → 98 ms, Q=1.6 → 98 ms). Consequence:
the FIXME at vqt.rs ("lower Q factor is needed so that higher frequencies are fully
covered") can be addressed at essentially zero latency cost. The coverage condition is
`1.44·(αf+γ)/Q ≥ f·(2^(1/84)−1)` at the top bin; with γ = 4.8·Q this solves to Q ≤ 1.63,
so **lowering `quality` to 1.6 closes the gaps**. Because γ scales with Q, the *bass*
bandwidth (≈ 1.44·4.8 ≈ 7 Hz) is independent of Q — the price is paid only in mid/high
frequency selectivity (~3% wider filters), which moves the two-notes-a-semitone-apart
resolution limit up from ~310 Hz to ~330 Hz.

*(Correction: the first version of this section recommended raising Q to 2.0–2.2. That was
backwards — higher Q narrows the filters and widens the gaps.)*

### 3.3 Quantifying the high-frequency coverage gap

At the top bin (6985 Hz): bin spacing = f·(2^(1/84) − 1) ≈ 57.8 Hz vs −3 dB width ≈ 53 Hz —
adjacent filters don't quite meet at −3 dB, giving inter-bin ripple. `test_vqt_bandwidths`
bounds the sweep ripple to <3 dB, so this is cosmetic today, but it's exactly what the dead
check in §1.1 was meant to detect. Either lower Q (§3.2) or keep the ripple and rely on the
peak interpolation, which already handles it.

Also note the `quality` doc (vqt.rs:247–254) claims "Q = f_center/bandwidth_3dB … Default
1.8". That's wrong: the parameter is librosa's `filter_scale`; the *actual* Q at high
frequencies is ≈ Q/(1.44·α) ≈ 130. Worth fixing — it will confuse anyone comparing against
the CQT literature.

### 3.4 Normalization: a deliberate and correct choice, with one side effect

L1-normalizing each filter in the time domain makes the response to an on-center unit
sinusoid exactly 1 independent of window length — this is why magnitudes are comparable
across bins (verified within 6 dB by `test_vqt_high_frequencies`; the residual variation is
window overlap, not normalization). Side effect worth documenting: white-noise gain scales
as ‖h‖₂²/‖h‖₁² ∝ 1/window_length, so the noise floor at the top bins sits ~8.6 dB above the
bass bins (window 600 vs 4365 samples). With the frame-relative floor in `power_to_db`, this
biases visual speckle toward the treble in quiet passages. If that's ever visible, a per-bin
noise-floor offset (computable at kernel build time) would flatten it.

The `* params.sr.sqrt()` factor in the kernel fill (vqt.rs:809) is an undocumented magic
gain (≈ ×148.5 ≈ +43 dB) whose only role is to land values in a nice dB range for
`ref_power = 0.09`. Fold it into `ref_power` or name it (`KERNEL_GAIN`) with a comment.

### 3.5 Time alignment

All filters are centered at the same instant (`window_center` = n_fft − max_window/2), so
the spectrum is a temporally coherent snapshot ~99 ms in the past — good for a visualizer
showing chords. The alternative (right-aligning short high-frequency windows at the buffer
end) would show treble transients ~75 ms sooner at the cost of temporal skew across the
spectrum. For a "feels live" instrument-following mode, that's worth an experiment someday;
the current choice is the conservative correct one. Phase is consistent per-bin but the phase
origin is each filter's window start, so absolute phase is not usable across bins — fine
since only magnitudes are consumed, but it rules out phase-vocoder/instantaneous-frequency
refinement without a kernel change. One doc line would prevent someone from trying.

### 3.6 `power_to_db` frame-relative normalization — `vqt.rs:1006–1038`

The function clamps to `max − 60 dB` then either shifts everything down by the minimum (if
min > 0) or clips negatives at 0. Two consequences: (a) a single loud bin lifts the floor and
visually ducks quiet notes for that frame; (b) the output switches between "relative" and
"absolute clipped" regimes depending on the frame minimum, so overall-quiet frames get
boosted. Since dagc already does adaptive gain upstream, there are two interacting AGC
stages. It works in practice, but consider a fixed reference (rely on dagc alone) — it would
also make `min_height`/`min_prominence` in peak detection mean something stable in dB terms.

### 3.7 EMA smoothing (consumer side, brief)

`EmaMeasurement` uses α = 2/(n+1) recomputed per frame — approximately frame-rate-independent
(your own test verifies it converges to 1 − e⁻¹). The exact frame-rate-independent form is
α = 1 − exp(−Δt/τ); switching would resolve the standing TODO ("keep an eye on how consistent
this is across frame rates") outright, with a one-line change and a ~2× reinterpretation of
existing τ constants. Also note smoothing operates on dB values (geometric averaging of
power) — decay is linear in dB, which is perceptually sensible; just document that it's
intentional.

---

## 4. Code quality

Ordered roughly by value:

1. **Doc rot:** `Vqt::new` and `vqt_kernel` doc comments list eight individual arguments
   (vqt.rs:379–394, 676–696) that were long ago folded into `VqtParameters`. The `quality`
   doc is wrong (§3.3). The module-level claim "errors are logged" for coverage gaps is
   false until §1.1 is fixed.
2. **Dead/vestigial code:** commented-out rubato `_resample` (vqt.rs:839–864), test
   scaffolding inside the hot path (vqt.rs:934–939), `power_normalized` (vqt.rs:1040),
   `t_diff` field (vqt.rs:375), `calculated_q` (§1.2), and `AnalysisState._spectrogram_buffer`
   — 940 KB allocated per `AnalysisState` for a buffer that moved to the display system long
   ago (analysis.rs:145, 224).
3. **Editor cruft:** the `//#orange`, `//#blue`, `//#red`, `//#yellow`, `//#pink`, `//#green`
   region markers throughout vqt.rs and util.rs.
4. **Legacy duplicate:** `pitchvis_wasm_kiss3d/src/cqt.rs` is a stale fork of an older VQT
   (single FFT, no multi-rate). Per your notes kiss3d is broken anyway — either delete the
   crate or make it consume `pitchvis_analysis` so the algorithm exists exactly once.
5. **Types/style:** `AnnontatedFilterParams` typo (vqt.rs:454); downscaling factors carried
   as `f32` and compared with `==` (vqt.rs:483, 525–527) — store the exponent `k: u32` or a
   `usize` from the start (`1 << k`); `UnscaledWindow.window: (usize, usize)` →
   `Range<usize>`; `Result<_, String>` → `thiserror` (the workspace convention elsewhere is
   `anyhow`, but a library crate deserves a typed error); `pub const` definitions inside
   `Default::default()` (vqt.rs:280–292) → module-level constants so they're discoverable
   and reusable.
6. **Tests** — current coverage is genuinely good (sweep ripple, close-frequency separation,
   delay bound, FFT-scaling contract). Worth adding:
   - a `resample()` unit test (impulse and pure sine → expected decimated spectrum), since
     it encodes the trickiest math in the file and would be the safety net for §2.1–2.3;
   - a magnitude-continuity test at group boundaries (sweep a tone across 74.9 Hz, 149.8 Hz,
     … and assert no step), which is the failure mode a dedup/refactor could introduce;
   - the in-code TODO tests (noise robustness, beat transients) — even loose bounds beat
     nothing.

---

## 5. Prioritized recommendations

| # | Item | Effort | Impact |
|--:|---|---|---|
| 1 | Fix dead bandwidth-gap check (§1.1) | trivial | correctness of validation |
| 2 | Remove dead zeroing loop; fold 1/M + √sr into kernel (§2.3, §3.4) | small | clarity + minor perf |
| 3 | Deduplicate FFTs per window size (§2.1) | medium | ~57% of per-frame FFT work |
| 4 | Preallocated scratch + `process_with_scratch` + out-param API (§2.4) | medium | zero steady-state allocation |
| 5 | `realfft` for real input (§2.2) | medium | further ~2× on FFT work |
| 6 | Input-length assert; `norm_sqr`; `total_cmp` (§1.3, §2.5) | trivial | robustness |
| 7 | Fix dB²-as-power weighting in tuning inaccuracy (§1.5) | trivial | metric correctness |
| 8 | Doc pass: quality vs real Q, γ crossover at ~1 kHz, phase caveat, magic gains (§3) | small | future-you |
| 9 | Lower `quality` to 1.6 (nearly latency-free, §3.2); re-run sweep test | small | closes the FIXME |
| 10 | Cleanup: markers, dead code, typo, legacy kiss3d cqt.rs, `_spectrogram_buffer` (§4) | small | hygiene |
| 11 | New tests: resample unit test, group-boundary continuity (§4.6) | small | guards the refactors |
| 12 | EMA → 1 − exp(−Δt/τ) (§3.7) | small | kills a standing TODO |

Items 1–6 are independent of each other and of the rest; item 11 should land *before* 3–5.

---

## Addendum: implementation status (2026-06-12)

All items above were implemented, with these notes:

- **Measured result: 0.165 → 0.091 ms/frame (1.8×)**, delay unchanged at 98 ms. Per frame:
  4 real FFTs (8192/4096/2048/1024) + 4 sparse mat-vecs, zero steady-state allocation in
  the hot path (only the returned dB vector).
- **Items 3+5 went further than planned:** instead of dropping the filters' negative-
  frequency coefficients for the half-spectrum path (measured at up to 0.96% of a filter's
  mass — more than estimated), they are handled *exactly* via a small per-group
  conjugate-part matrix (`X[N−k] = conj(X[k])` for real input; 379 of 18k nnz). No
  approximation added.
- **Item 9 (corrected direction):** `quality` lowered 1.8 → 1.6, γ stays 4.8·Q (= 7.68).
  The now-alive gap check confirms full −3 dB coverage (no warnings). Cost: the
  close-frequency resolution limit moved from ~310 Hz to ~330 Hz;
  `test_vqt_close_frequencies` now starts at 2.6 octaves with a comment documenting the
  trade-off.
- **API change:** `calculate_vqt_instant_in_db` takes `&mut self` (internal scratch) and
  asserts `x.len() == n_fft`; `Vqt::new` returns `Result<_, VqtError>` (thiserror).
  `VqtKernel` is now `Vec<WindowGroup>`; default parameters are module-level consts.
- **EMA:** switched to the exact rate-independent form α = 1 − exp(−2Δt/τ) (also fixes
  potential α > 1 overshoot at low frame rates); existing tests pass unchanged.
- **Deliberately NOT done:**
  - Window alignment stays time-centered (right-aligning makes overtones lead during
    glissandi, bending the overtone stack along the spiral radii) — by design, see §3.5.
  - `pitchvis_train` skipped per request: needs a one-line `&mut` at train.rs:248.
  - `pitchvis_wasm_kiss3d` (stale VQT copy) left untouched — crate-level decision
    (delete vs. port) still open.
  - `power_to_db`'s frame-relative normalization (§3.6) kept as-is.
  - `AnalysisParameters::spectrogram_length` kept (debug-UI references it) though it no
    longer controls anything; the dead `_spectrogram_buffer` behind it was removed.
