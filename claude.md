# Claude AI Assistant Guide for PitchVis

This file provides guidance for AI assistants (like Claude) working with the PitchVis codebase.

## Project Overview

PitchVis is a real-time musical pitch analysis and visualization application written in Rust. It captures audio input, performs frequency analysis using Variable Q Transform (VQT), detects musical pitches, and renders beautiful visualizations using the Bevy game engine.

**Target Platforms**: Desktop (Linux/macOS/Windows), Web (WASM), Android

**Primary Use Case**: Real-time visualization of musical pitches for musicians, audio engineers, and music enthusiasts.

## Quick Start

### Building and Running

**Desktop (recommended for development)**:
```bash
cd pitchvis_viewer
cargo run --features bevy/dynamic_linking --bin pitchvis
```

**Quick desktop build check**:
```bash
./check-desktop.sh
```

**Web/WASM**:
```bash
npm install
npm run serve  # Local development
npm run build  # Production build to dist/
```

**Serial output (for LED strips)**:
```bash
cd pitchvis_serial
cargo run --bin pitchvis_serial -- /path/to/serial/device 115200
```

### Prerequisites

System dependencies (Linux):
- `libasound2-dev` (ALSA audio)
- `libudev-dev` (device detection)
- `libwayland-dev` (Wayland display server - for desktop builds)
- `libxkbcommon-dev` (keyboard handling)

**Note**: The `check-desktop.sh` script can be used to verify that all packages compile correctly.

## Architecture Summary

### Crate Organization

```
pitchvis/
├── pitchvis_viewer/      # Main application (desktop/web/Android)
├── pitchvis_audio/       # Audio input (cpal-based)
├── pitchvis_analysis/    # VQT and pitch detection algorithms
├── pitchvis_colors/      # Pitch-to-color mapping (LAB color space)
├── pitchvis_serial/      # LED strip serial output
├── pitchvis_train/       # ML model training (optional)
├── dagc_fork/            # Automatic Gain Control (forked)
└── rustysynth_fork/      # SoundFont synth (forked)
```

### Data Flow

```
Audio Input (cpal) → Ring Buffer (with AGC) → VQT → Peak Detection → Visualization (Bevy)
```

### Key Systems (Bevy ECS)

1. **Audio System**: Manages ring buffer access
2. **VQT System**: Computes frequency spectrum
3. **Analysis System**: Detects peaks and tracks pitches
4. **Display System**: Renders visualization
5. **ML System**: Optional ML inference (feature-gated)

See `ARCHITECTURE.md` and `pitchvis_viewer/ARCHITECTURE.md` for detailed documentation.

## Code Conventions

### Rust Style

- **Edition**: Rust 2021
- **Formatting**: Standard `rustfmt` settings
- **Imports**: Group by `std`, external crates, internal crates
- **Error Handling**: Use `anyhow::Result` for most functions

### Bevy ECS Patterns

**Systems**: Pure functions that take Bevy parameters (`Res`, `ResMut`, `Query`, etc.)
```rust
pub fn update_system(
    mut resource: ResMut<MyResource>,
    query: Query<&Component>,
) {
    // System logic
}
```

**System Ordering**: Use `.after()` to enforce dependencies:
```rust
app.add_systems(Update, (
    update_vqt_system,
    update_analysis_state_system.after(update_vqt_system),
));
```

**Resources**: Wrap external state in Bevy resources:
```rust
#[derive(Resource)]
pub struct MyStateResource(pub MyState);
```

### Platform-Specific Code

Use conditional compilation:
```rust
#[cfg(not(target_arch = "wasm32"))]
fn desktop_only() { }

#[cfg(target_arch = "wasm32")]
fn wasm_only() { }

#[cfg(target_os = "android")]
fn android_only() { }
```

## Common Tasks

### Adding a New Visual Mode

1. Add variant to `display_system::VisualsMode` enum
2. Update `cycle_visuals_mode()` in `app/common.rs`
3. Implement rendering logic in `display_system/update.rs`
4. Update button text in `setup_buttons()`

### Adding a New Display Element

1. Define component in `display_system/mod.rs`
2. Spawn entities in `display_system/setup.rs`
   - Add `spawn_*()` function
   - Call it from `setup_display()`
3. Create update system in `display_system/update.rs`
   - Implement update logic (e.g., `update_*_system()`)
   - Implement show/hide system (e.g., `*_showhide()`)
4. Register systems in `app/common.rs`
   - Add to `register_common_update_systems()`
5. Add visibility controls based on display mode

**Example**: See `SpectrogramDisplay` and `ChromaBox` components for reference implementation of texture-based and UI-based displays.

### Modifying Analysis Parameters

1. Update `VqtParameters` in `pitchvis_analysis/src/vqt.rs`
2. Update `AnalysisParameters` in `pitchvis_analysis/src/analysis.rs`
3. Adjust constants in `pitchvis_viewer/src/app/desktop_app.rs` (e.g., `BUFSIZE`)

### Adding UI Elements

1. Define component marker in `app/common.rs`
2. Create setup system (spawns entities)
3. Create update system (modifies text/visibility)
4. Add systems to `Startup` and `Update` schedules

## Important Files

### Configuration

- **`Cargo.toml`** (workspace): Build profiles, workspace members, patches
- **`pitchvis_viewer/Cargo.toml`**: Features (`ml`), platform-specific deps
- **Settings file**: `{config_dir}/pitchvis/settings.toml` (runtime, created automatically)

### Entry Points

- **`pitchvis_viewer/src/main.rs`**: Desktop binary entry
- **`pitchvis_viewer/src/lib.rs`**: Library exports, module declarations
- **`pitchvis_viewer/src/app/desktop_app.rs`**: Desktop app setup
- **`pitchvis_viewer/src/app/wasm_app.rs`**: WASM app setup
- **`pitchvis_viewer/src/app/android_app.rs`**: Android app setup

### Core Logic

- **`pitchvis_analysis/src/vqt.rs`**: Variable Q Transform implementation
- **`pitchvis_analysis/src/analysis.rs`**: Peak detection and pitch tracking
- **`pitchvis_audio/src/audio_desktop.rs`**: Desktop audio capture (cpal)
- **`pitchvis_audio/src/audio_wasm.rs`**: Web audio capture (WebAudio API)
- **`pitchvis_viewer/src/display_system/setup.rs`**: Display setup (all visual elements)
- **`pitchvis_viewer/src/display_system/update.rs`**: Main rendering loop and debug displays

## Testing

### Running Tests

```bash
cargo test --workspace
```

### Manual Testing

1. Run the viewer: `cargo run --features bevy/dynamic_linking --bin pitchvis`
2. Test inputs:
   - Play music or sing into microphone
   - Use online tone generator (e.g., 440 Hz for A4)
3. Toggle modes:
   - Space/Click: Switch between Normal and Debugging modes
   - Buttons (Debug mode): Cycle visual modes and FPS limits

### Debug Features

When in debugging mode (press Space or click to toggle), additional analysis visualizations are shown:

**FPS Counter & Metrics** (Top-left):
- FPS with color coding (green >120, yellow 60-120, red <30)
- Audio latency and chunk size
- VQT latency
- VQT smoothing duration (varies by frequency and calmness)

**Analysis Parameters Display** (Bottom-left):
- bassline_peak_config (prominence, height)
- highest_bassnote
- vqt_smoothing_duration_base
- vqt_smoothing_calmness range (min-max)
- note_calmness_smoothing_duration
- scene_calmness_smoothing_duration
- tuning_inaccuracy_smoothing_duration
- Current scene_calmness value
- Number of detected peaks

**Spectrogram Display** (Top-middle):
- Real-time scrolling spectrogram showing VQT data over time
- 200 frames of history, scrolls top-to-bottom (newest at top)
- Width: VQT bins (frequency), Height: time frames
- Colors: pitch-based coloring from `pitchvis_colors::COLORS`
- Brightness: VQT magnitude with enhancement for visibility
- Position: `Transform::from_xyz(0.0, 10.0, 10.0)` (adjustable in `display_system/setup.rs:538`)

**Chroma Display** (Bottom, UI overlay):
- 12 colored boxes representing pitch class presence (C, C#, D, etc.)
- Calculates energy per pitch class from smoothed VQT
- Opacity varies with pitch class strength (transparent=absent, opaque=strong)
- Normalized per-frame for relative strength
- Uses power domain: `10^(dB/10)` for energy calculation
- Position: adjustable in `display_system/setup.rs:563`

**Tuning & Chord Info** (Bottom-right):
- Tuning drift in cents (color-coded: green <10¢, yellow 10-20¢, orange 20-30¢, red >30¢)
- Detected chord name (if any)

**Control Buttons** (Left side):
- Visuals Mode: Full / Zen / Performance / Galaxy
- FPS Limit: 30 / 60 / None
- VQT Smoothing: None / Short / Default / Long

**Feature Toggles** (Right side):
- Chord Recognition: On/Off
- Root Note Tinting: On/Off

### Performance Testing

1. Enable debugging mode (press Space)
2. Monitor FPS counter (should be green at 60+ FPS)
3. Check latency metrics:
   - Audio latency: < 50ms ideal
   - VQT latency: < 20ms ideal
4. Test on different platforms (desktop, web, mobile)

## Performance Considerations

### Bottlenecks

1. **VQT Computation**: Most CPU-intensive part
   - Optimize by reducing FFT size or octave range
   - Consider GPU acceleration (future work)

2. **Peak Detection**: Scales with number of frequency bins
   - Already optimized with sparse operations

3. **Rendering**: Usually GPU-bound
   - Use `VisualsMode::Performance` for lower overhead
   - Reduce particle/entity counts

### Optimization Tips

- **Debug builds**: Use `opt-level = 1` for workspace, `3` for dependencies (already configured)
- **Release builds**: Use `--release` for production performance
- **WASM builds**: Profile uses `opt-level = 'z'` for size optimization
- **FPS limiting**: Default 60 FPS reduces unnecessary computation

## Build Profiles

```toml
[profile.dev]
opt-level = 1  # Workspace: Fast compilation, basic optimization
[profile.dev.package."*"]
opt-level = 3  # Dependencies: Full optimization

[profile.release]
opt-level = 2
lto = "thin"   # Link-time optimization
strip = true   # Remove debug symbols

[profile.web-release]
opt-level = 'z'  # Optimize for binary size
lto = "thin"
panic = "abort"
```

## Debugging

### Common Issues

**No audio input detected**:
- Check system audio permissions
- Verify default input device is set correctly
- On Linux, ensure ALSA/PulseAudio is configured

**Low FPS**:
- Check VQT parameters (reduce octaves or buckets per octave)
- Enable `VisualsMode::Performance`
- Reduce buffer size (trade-off: more latency)

**WASM build fails**:
- Ensure `wasm-pack` is installed: `cargo install wasm-pack`
- May need to build from master for latest features

**Android build fails**:
- Requires Android NDK and proper toolchain setup
- See Bevy Android documentation

### Logging

Desktop uses `env_logger`:
```bash
RUST_LOG=debug cargo run --features bevy/dynamic_linking --bin pitchvis
```

Levels: `error`, `warn`, `info`, `debug`, `trace`

## Dependencies

### Key External Crates

- **bevy** (0.16): Game engine, ECS, rendering
- **cpal** (0.16): Cross-platform audio I/O
- **rustfft** (6.1): Fast Fourier Transform
- **dagc**: Digital Automatic Gain Control (forked)
- **nalgebra** (0.34): Linear algebra for transforms
- **bevy-persistent** (0.8): Persistent settings storage

### Patched Crates

```toml
[patch.crates-io]
kiss3d = { git = "https://github.com/heinzelotto/kiss3d" }
```

## Feature Flags

**`ml`**: Enable machine learning inference
```bash
cargo run --features ml,bevy/dynamic_linking --bin pitchvis
```

Requires PyTorch (`tch` crate) and model file (`model.pt`).

## Platform-Specific Notes

### Desktop

- Audio via `cpal` (ALSA on Linux, CoreAudio on macOS, WASAPI on Windows)
- Settings stored in OS config directory
- Full keyboard/mouse support

### Web (WASM)

- Audio via WebAudio API and MediaStream
- Settings stored in browser LocalStorage
- Limited to browser security constraints (HTTPS for microphone access)
- Optimized for binary size

### Android

- Audio via `oboe` (low-latency Android audio)
- Touch-only input
- Settings stored in app data directory
- Lifecycle managed by `android-activity` crate

## Contributing Guidelines

When making changes:

1. **Read architecture docs**: Understand the system you're modifying
2. **Test on target platforms**: Desktop changes should also work on WASM if possible
3. **Maintain performance**: Profile before/after for performance-critical code
4. **Update documentation**: Keep `ARCHITECTURE.md` and this file in sync
5. **Follow Rust conventions**: Run `cargo fmt` and `cargo clippy`
6. **Check compile warnings**: Fix all warnings before committing

## Useful Commands

```bash
# Development
cargo run --features bevy/dynamic_linking --bin pitchvis
cargo test --workspace
cargo clippy --workspace
cargo fmt --all

# Release builds
cargo build --release --bin pitchvis
npm run build  # WASM

# Clean builds
cargo clean
rm -rf dist/  # WASM output

# Documentation
cargo doc --open --no-deps
```

## Resources

- **Bevy Documentation**: https://bevyengine.org/
- **cpal Documentation**: https://docs.rs/cpal/
- **VQT Theory**: Variable Q Transform for audio analysis
- **WebAudio API**: https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API

## Future Improvements

Potential areas for enhancement (see TODO comments in code):

- GPU-accelerated VQT computation
- More visual modes and customization
- MIDI output support
- Advanced tuning detection (microtonal scales)
- Recording and playback features
- Multi-channel audio analysis
- Plugin system for extensibility

## Questions?

If you're unclear about any aspect of the codebase:

1. Check `ARCHITECTURE.md` for high-level overview
2. Check `pitchvis_viewer/ARCHITECTURE.md` for viewer details
3. Search for TODO/FIXME comments in the code
4. Look at git history for context on specific features
5. Examine test cases for usage examples

## AI Assistant Tips

When working with this codebase:

- **System ordering matters**: Bevy systems must run in correct order (VQT → Analysis → Display)
- **Platform conditionals**: Always check `cfg` attributes when modifying code
- **Resource wrapping**: External state must be wrapped in Bevy `Resource` types
- **Visibility toggles**: Display elements are shown/hidden based on display mode, not destroyed
- **Ring buffer is shared**: Audio capture happens on separate thread, synchronized via Arc<Mutex<>>
- **Settings are persistent**: Changes to `SettingsState` are automatically saved to disk

When suggesting changes:

- Prefer editing existing files over creating new ones
- Maintain backward compatibility with existing save files
- Consider performance impact (VQT is expensive)
- Test on multiple platforms when possible
- Keep visual changes consistent with existing aesthetic
