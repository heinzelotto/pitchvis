# Building PitchVis

PitchVis uses a unified build system powered by `cargo xtask` that provides simple, consistent commands for all platforms.

## Quick Start

```bash
# Desktop - Development (fast iteration with dynamic linking)
cargo xtask run

# Desktop - Release build
cargo xtask build desktop --release

# WASM - Development
cargo xtask build wasm

# WASM - Release
cargo xtask build wasm --release

# Android - Debug
cargo xtask build android

# Android - Release
cargo xtask build android --release
```

## Prerequisites

### All Platforms
- Rust toolchain (install from https://rustup.rs/)
- System dependencies for audio (`cpal`):
  - Linux: `libasound2-dev`, `libudev-dev`
  - macOS: No additional dependencies
  - Windows: No additional dependencies

### WASM
- Node.js and npm
- wasm-pack: `cargo install wasm-pack`

### Android
- Android SDK and NDK
- cargo-ndk: `cargo install cargo-ndk`
- Environment variables:
  - `ANDROID_SDK_ROOT` or `ANDROID_HOME`: Path to Android SDK
  - `ANDROID_NDK_ROOT`: Path to Android NDK (e.g., `$ANDROID_SDK_ROOT/ndk/28.2.13676358/`)
- Gradle (included in Android SDK)

For release builds on Android, you also need:
- `CARGO_APK_RELEASE_KEYSTORE`: Path to your release keystore
- `CARGO_APK_RELEASE_KEYSTORE_PASSWORD`: Keystore password

## Detailed Commands

### Desktop

**Run (Development):**
```bash
cargo xtask run
```
This builds and runs with:
- Debug profile with optimizations for dependencies
- Bevy dynamic linking for faster iteration
- Default logging: `RUST_LOG=error,pitchvis_analysis=debug`

**Run (Release):**
```bash
cargo xtask run --profile release
```

**Build only:**
```bash
cargo xtask build desktop           # Debug build with dynamic linking
cargo xtask build desktop --release # Release build
```

**Pass arguments to the binary:**
```bash
cargo xtask run -- --your-args-here
```

### WASM

**Build:**
```bash
cargo xtask build wasm              # Development build
cargo xtask build wasm --release    # Release build
```

This will:
1. Install npm dependencies
2. Build Rust code to WASM with wasm-pack
3. Bundle everything with webpack
4. Output to `pitchvis_viewer/wasm/dist/`

**Build Rust code only (skip npm):**
```bash
cargo xtask build wasm --rust-only
```

**Local development server:**
```bash
cd pitchvis_viewer/wasm
npm run serve
```

### Android

**Build:**
```bash
cargo xtask build android           # Debug APK
cargo xtask build android --release # Release AAB (Android App Bundle)
```

This will:
1. Copy native libraries (libc++_shared.so)
2. Copy shader assets
3. Build Rust library with cargo-ndk for arm64-v8a
4. Build Android app with Gradle

**Skip asset/library setup (if already done):**
```bash
cargo xtask build android --skip-setup
```

**Install debug build to device:**
```bash
cargo xtask build android
cd pitchvis_viewer/android
./gradlew installDebug
```

### Cleaning

```bash
cargo xtask clean desktop  # Clean desktop builds
cargo xtask clean wasm     # Clean WASM builds
cargo xtask clean android  # Clean Android builds
cargo xtask clean all      # Clean everything
```

## Help

Get help on any command:
```bash
cargo xtask --help
cargo xtask build --help
cargo xtask run --help
```

## Architecture

The build system is implemented as a Rust crate (`xtask/`) in the workspace. This provides:
- **Type safety**: Build logic is checked by the Rust compiler
- **Cross-platform**: Works on Linux, macOS, and Windows
- **Simplicity**: One command interface for all platforms
- **Discoverability**: Built-in help and clear error messages

### Project Structure

```
pitchvis/
├── xtask/                  # Build automation crate
│   └── src/main.rs         # Build commands implementation
├── pitchvis_viewer/        # Main viewer application
│   ├── android/            # Android project
│   ├── wasm/               # WASM/web project
│   └── src/                # Rust source code
├── .cargo/
│   └── config.toml         # Cargo aliases
└── Cargo.toml              # Workspace configuration
```

## Migration from Old Build Scripts

Old scripts in `pitchvis_viewer/`:
- `build_wasm.sh` → `cargo xtask build wasm`
- `build_android_debug.sh` → `cargo xtask build android`
- `build_android_release.sh` → `cargo xtask build android --release`
- `run_desktop.sh` → `cargo xtask run`

The old scripts are still present for reference but the new xtask system is recommended.

## Troubleshooting

### WASM build fails
- Ensure wasm-pack is installed: `cargo install wasm-pack`
- Try cleaning first: `cargo xtask clean wasm`

### Android build fails
- Check environment variables are set correctly
- Ensure cargo-ndk is installed: `cargo install cargo-ndk`
- Verify NDK version matches the path in `ANDROID_NDK_ROOT`
- For release builds, ensure keystore variables are set

### Desktop build is slow
- Use `cargo xtask run` (default) which uses dynamic linking for faster iteration
- Only use `--profile release` for final builds

### Permission errors on scripts
The xtask system doesn't require execute permissions since it's pure Rust code.
