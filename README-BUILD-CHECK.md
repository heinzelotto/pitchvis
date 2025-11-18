# Desktop Build Check Without System Dependencies

This document explains how to check the desktop build compiles without having all the required system dependencies (Wayland, ALSA, libudev, etc.) installed.

## Quick Start

Simply run:
```bash
./check-desktop.sh
```

This script will check that all desktop packages compile correctly without attempting to link system libraries.

## How It Works

### 1. Stub pkg-config Files

The solution uses stub `.pc` files that satisfy the build script checks for system dependencies:
- `/tmp/stub-pkgconfig/wayland-client.pc`
- `/tmp/stub-pkgconfig/wayland-cursor.pc`
- `/tmp/stub-pkgconfig/wayland-egl.pc`
- `/tmp/stub-pkgconfig/alsa.pc`
- `/tmp/stub-pkgconfig/libudev.pc`

These files are created automatically by the check script and contain minimal valid pkg-config data.

### 2. Selective Package Checking

The script checks only desktop packages and excludes the WASM package (`pitchvis-wasm-kiss3d`) which has web-specific dependencies.

Packages checked:
- `pitchvis_analysis` - Core analysis algorithms
- `pitchvis_audio` - Audio processing
- `pitchvis_colors` - Color calculations
- `pitchvis_serial` - Serial interface
- `pitchvis_viewer` - Main desktop viewer (Bevy-based)
- `pitchvis_train` - ML training utilities
- `dagc` - Dynamic AGC
- `rustysynth` - MIDI synthesis

## Manual Check

If you prefer to run the checks manually:

```bash
# Set up stub dependencies
mkdir -p /tmp/stub-pkgconfig
for lib in wayland-client wayland-cursor wayland-egl alsa libudev; do
  cat > /tmp/stub-pkgconfig/$lib.pc <<EOF
prefix=/usr
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: $lib
Description: $lib (stub)
Version: 1.0.0
Libs: -L\${libdir}
Cflags: -I\${includedir}
EOF
done

# Run cargo check
PKG_CONFIG_PATH=/tmp/stub-pkgconfig cargo check \
  -p pitchvis_analysis \
  -p pitchvis_audio \
  -p pitchvis_colors \
  -p pitchvis_serial \
  -p pitchvis_viewer \
  -p pitchvis_train \
  --all-targets
```

## Limitations

- This only checks that the Rust code compiles
- It does NOT produce working binaries
- For actual builds, you still need the real system dependencies
- The WASM package is not checked (requires wasm32 target)

## CI/CD Integration

This script is ideal for CI environments where you want to verify code compiles without installing full desktop dependencies. Add to your CI pipeline:

```yaml
- name: Check desktop build
  run: ./check-desktop.sh
```

## Full Build Requirements

For a complete working build, install:
- Wayland development libraries
- ALSA development libraries
- libudev development libraries
- Rust toolchain

On Debian/Ubuntu:
```bash
sudo apt-get install libwayland-dev libasound2-dev libudev-dev
```
