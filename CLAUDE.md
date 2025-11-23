# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PitchVis is a real-time musical pitch analysis and visualization system that analyzes live audio input to detect musical pitches and provides colorful visualizations. The project supports multiple deployment targets including desktop, web (WASM), Android, and serial output for LED hardware controllers.

**Target Platforms**: Desktop (Linux/macOS/Windows), Web (WASM), Android

**Primary Use Case**: Real-time visualization of musical pitches for musicians, audio engineers, and music enthusiasts.

## Multi-Crate Architecture

This is a Rust workspace with 9 main crates:

- **`pitchvis_analysis`** - Core VQT (Variable-Q Transform) and pitch detection algorithms
- **`pitchvis_audio`** - Cross-platform audio input (CPAL for desktop, Web Audio API for WASM)
- **`pitchvis_colors`** - Color space conversion and pitch-to-color mapping
- **`pitchvis_viewer`** - Main visualization app using Bevy engine (desktop, WASM, Android)
- **`pitchvis_serial`** - Serial output for hardware LED controllers
- **`pitchvis_train`** - ML training utilities with TinySOL dataset and MIDI synthesis
- **`pitchvis_android_serial`** - Android-specific serial implementation
- **`pitchvis_wasm_kiss3d`** - Alternative WASM renderer using Kiss3D
- **`dagc_fork`** / **`rustysynth_fork`** - Forked dependencies for audio processing and MIDI synthesis

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

## Quick Start

### Building and Running

PitchVis uses a unified build system powered by `cargo xtask`. This is the recommended way to build and run the project.

**Desktop (recommended for development)**:
```bash
cargo xtask run  # Fast iteration with dynamic linking and file watcher
```

**Desktop production build**:
```bash
cargo xtask build desktop  # Always builds in release mode
```

**Web/WASM**:
```bash
cargo xtask build wasm           # Development build
cargo xtask build wasm --release # Production build
cd pitchvis_viewer/wasm/
npm run serve                    # Local development server
```

**Android**:
```bash
cargo xtask build android           # Debug APK
cargo xtask build android --release # Release AAB
```

**Serial output (for LED strips)**:
```bash
cd pitchvis_serial
cargo run --bin pitchvis_serial -- /path/to/serial/device 115200
```

See the [Cargo xtask Build System](#cargo-xtask-build-system) section below for complete documentation.

### Prerequisites

System dependencies (Linux):
- `libasound2-dev` (ALSA audio)
- `libudev-dev` (device detection)
- `libwayland-dev` (Wayland display server - for desktop builds)
- `libxkbcommon-dev` (keyboard handling)

**Note**: The `check-desktop.sh` script can be used to verify that all packages compile correctly.

## Architecture Summary

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

### Key Technical Concepts

**Variable-Q Transform (VQT)**: The core audio analysis uses VQT algorithms in `pitchvis_analysis` for musical pitch detection. VQT provides better frequency resolution at lower frequencies, making it ideal for musical analysis.

**Real-time Pipeline**: Audio input → Digital AGC → VQT analysis → Peak detection → Color mapping → Visualization

**Serial Protocol Format**: For hardware LED controllers: `0xFF <num_triples> <r1> <g1> <b1> <r2> <g2> <b2> ...`
LED values are in range [0x00; 0xFE], with 0xFF as the marker byte.

## Cargo xtask Build System

PitchVis uses a unified, type-safe build automation system implemented as a Rust crate. This replaces older shell scripts and provides a consistent, cross-platform command interface for all build operations.

### Why xtask?

- **Type-safe**: Commands are checked by the Rust compiler
- **Cross-platform**: Works on Linux, macOS, and Windows
- **Discoverable**: Built-in help via `cargo xtask --help`
- **Unified**: Single interface for all platforms (Desktop, WASM, Android)
- **Clear errors**: Better error messages than shell scripts

### Available Commands

#### Build Command

```bash
cargo xtask build <target> [options]
```

**Desktop**:
```bash
cargo xtask build desktop  # Always builds in release mode (Bevy recommendation)
```
- Always uses `--release` profile for optimal performance
- Does NOT include development features (dynamic_linking, file_watcher)
- Use for production builds

**WASM/Web**:
```bash
cargo xtask build wasm              # Development build
cargo xtask build wasm --release    # Production build (optimized for size)
cargo xtask build wasm --rust-only  # Build Rust only, skip npm
```
- Automatically installs npm dependencies
- Runs wasm-pack and webpack
- Output to `pitchvis_viewer/wasm/dist/`

**Android**:
```bash
cargo xtask build android              # Debug APK
cargo xtask build android --release    # Release AAB (Android App Bundle)
cargo xtask build android --skip-setup # Skip asset/library copying
```
- Automatically copies native libraries (libc++_shared.so)
- Copies shader assets to Android project
- Builds with cargo-ndk for arm64-v8a architecture
- Uses Gradle for final APK/AAB packaging

#### Run Command

```bash
cargo xtask run [-- args...]
```

Runs the desktop viewer optimized for **fast development iteration**:
- Always uses `--release` mode (per Bevy recommendations)
- Enables `bevy/dynamic_linking` for faster compile times
- Enables `bevy/file_watcher` for hot reload of assets
- Equivalent to: `cargo run --release --features bevy/dynamic_linking,bevy/file_watcher --bin pitchvis`

**Pass custom arguments**:
```bash
cargo xtask run -- --your-custom-args
```

**Set logging level**:
```bash
RUST_LOG=debug cargo xtask run
RUST_LOG=warn cargo xtask run
```

#### Clean Command

```bash
cargo xtask clean [target]
```

Clean build artifacts for specific targets or all:
```bash
cargo xtask clean desktop   # Clean desktop builds
cargo xtask clean wasm      # Clean WASM builds and node_modules
cargo xtask clean android   # Clean Android builds with Gradle
cargo xtask clean all       # Clean everything (default)
```

### Platform-Specific Notes

#### Desktop

**Development**: Use `cargo xtask run` for fastest iteration
- Compiles in release mode but with dynamic linking enabled
- File watcher enables hot reloading of shaders and assets
- Typical rebuild time: ~3-10 seconds (depending on changes)

**Production**: Use `cargo xtask build desktop`
- Full release optimization without dev features
- Binary located at: `target/release/pitchvis`

**Why always release mode?**
Bevy (the game engine used) is significantly slower in debug mode. Even for development, release mode with dynamic linking provides the best experience.

#### WASM/Web

**Prerequisites**:
- Node.js and npm
- wasm-pack: `cargo install wasm-pack`

**Build process**:
1. Installs npm dependencies (if needed)
2. Compiles Rust to WebAssembly
3. Bundles with webpack to `dist/`

**Development workflow**:
```bash
cargo xtask build wasm
cd pitchvis_viewer/wasm
npm run serve  # Starts local server with live reload
```

**Production deployment**:
```bash
cargo xtask build wasm --release
# Deploy contents of pitchvis_viewer/wasm/dist/
```

#### Android

**Prerequisites**:
- Android SDK (set `ANDROID_SDK_ROOT` or `ANDROID_HOME`)
- Android NDK (set `ANDROID_NDK_ROOT`)
- cargo-ndk: `cargo install cargo-ndk`

**Build process**:
1. Copies native libraries from NDK
2. Copies shader assets to Android project
3. Builds Rust library with cargo-ndk
4. Packages with Gradle

**Installation**:
Debug builds are automatically installed to connected device.

**Output locations**:
- Debug APK: `android/app/build/outputs/apk/debug/app-debug.apk`
- Release AAB: `android/app/build/outputs/bundle/release/app-release.aab`

### Common Workflows

**Quick development session**:
```bash
cargo xtask run
# Make changes, Ctrl+C to stop, run again
```

**Test all platforms**:
```bash
cargo xtask build desktop
cargo xtask build wasm
cargo xtask build android
```

**Clean rebuild**:
```bash
cargo xtask clean all
cargo xtask build desktop
```

**WASM development with live server**:
```bash
cargo xtask build wasm
cd pitchvis_viewer/wasm
npm run serve
# Edit code, rebuild with cargo xtask build wasm in another terminal
```

### Getting Help

All commands have built-in help:
```bash
cargo xtask --help
cargo xtask build --help
cargo xtask run --help
cargo xtask clean --help
```

### Implementation

The xtask system is located in `/xtask/` and uses:
- **anyhow**: Error handling
- **clap**: CLI argument parsing

All build logic is in Rust, making it type-safe and cross-platform.

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

Tests are primarily in `pitchvis_analysis` for VQT accuracy validation.

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

**Spectrogram Display** (Center):
- Real-time scrolling spectrogram showing VQT data over time
- 200 frames of history displayed horizontally (newest on right, scrolls left)
- Horizontal axis: Time (200 frames), Vertical axis: Frequency (low notes at bottom)
- Rotated 90° counter-clockwise via transform, so the circular buffer "scan line" rotates through texture indices
- Colors: pitch-based coloring from `pitchvis_colors::COLORS`
- Brightness: VQT magnitude with enhancement for visibility
- Position: `Transform::from_xyz(0.0, 4.0, 5.0)` with -90° rotation (adjustable in `display_system/setup.rs:542`)

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

The workspace defines several optimized build profiles:

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

[profile.non-web-release]
# Performance-optimized for desktop/mobile
```

## Training Data

The project includes:
- **TinySOL Dataset**: 2,913 classical instrument samples for ML training
- **Large MIDI Collection**: Thousands of MIDI files in `pitchvis_train/midi/`
- **SoundFont Integration**: Multiple .sf2/.sf3 files for realistic synthesis

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

## Architecture Notes

- Uses Bevy ECS for the main visualization application
- Cross-platform audio via CPAL (desktop) and Web Audio API (WASM)
- Shared analysis crate ensures consistent pitch detection across all platforms
- Modular design allows mixing and matching components for different deployment scenarios

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
cargo xtask run                # Run with dev features (recommended)
cargo test --workspace         # Run tests
cargo clippy --workspace       # Lint code
cargo fmt --all                # Format code

# Production builds
cargo xtask build desktop      # Desktop release
cargo xtask build wasm --release  # WASM release
cargo xtask build android --release  # Android release

# Clean builds
cargo xtask clean all          # Clean all targets
cargo xtask clean desktop      # Clean desktop only

# Documentation
cargo doc --open --no-deps

# Direct cargo commands (if not using xtask)
cargo run --release --features bevy/dynamic_linking,bevy/file_watcher --bin pitchvis
cargo build --release --bin pitchvis
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
