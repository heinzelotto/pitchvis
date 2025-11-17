# PitchVis
live analysis of musical pitches in an audio signal and a nice visualization

Watch a demo video here:

[![PitchVis Demo Video](https://img.youtube.com/vi/7B4tVcVmOgg/sddefault.jpg)](https://youtu.be/7B4tVcVmOgg)

Get the Android App here: https://play.google.com/store/apps/details?id=org.p1graph.pitchvis

More resources on the [Website](https://www.p1graph.org/pitchvis/)

# Building and Running

PitchVis uses a unified build system powered by `cargo xtask`.

## Quick Start

```bash
# Run desktop viewer (development mode with fast iteration)
cargo xtask run

# Build for WASM/web
cargo xtask build wasm

# Build for Android
cargo xtask build android
```

See [BUILD.md](BUILD.md) for complete build documentation, including release builds, prerequisites, and troubleshooting.

## Development

For fast iteration during development, use `cargo xtask run` which enables Bevy's dynamic linking feature for faster compile times.

For the WASM development server:
```bash
cd pitchvis_viewer/wasm
npm run serve
```

## Serial Output

To output to a serial port (e.g., for LED strip control), run from within `pitchvis_serial/`:
```bash
cargo r --features bevy/dynamic_linking --bin pitchvis_serial -- </path/to/serial/fd> <baudrate>
```

The serial output format is: `0xFF <num_triples / 256> <num_triples % 256> <r1> <g1> <b1> <r2> <g2> <b2> ...`
LED values are within [0x00; 0xFE] and 0xFF is the marker byte beginning each sequence.

## Prerequisites

- Rust toolchain (install from https://rustup.rs/)
- System dependencies for audio (`cpal`): `libasound2-dev` and `libudev-dev` on Linux

See [BUILD.md](BUILD.md) for platform-specific prerequisites.


