# PitchVis Architecture

## Overview

PitchVis is a real-time musical pitch analysis and visualization application built in Rust. The project is organized as a Cargo workspace with multiple crates, each handling a specific aspect of the audio analysis and visualization pipeline.

## Workspace Structure

The workspace consists of the following crates:

### Core Crates

- **`pitchvis_viewer`**: Main application crate that ties everything together. Provides desktop, web (WASM), and Android applications using the Bevy game engine for rendering.

- **`pitchvis_audio`**: Audio input handling using `cpal`. Manages audio stream setup, ring buffer for audio data, and automatic gain control (AGC) via the `dagc` crate.

- **`pitchvis_analysis`**: Core signal processing algorithms including:
  - Variable Q Transform (VQT) for frequency analysis
  - Peak detection and pitch tracking
  - Analysis state management and smoothing

- **`pitchvis_colors`**: Color calculation utilities for mapping pitch frequencies to colors in the LAB color space.

### Additional Crates

- **`pitchvis_serial`**: Serial port output for LED strip control (embedded applications).

- **`pitchvis_train`**: Machine learning model training utilities (optional).

- **`pitchvis_wasm_kiss3d`**: Legacy WASM visualization using kiss3d (deprecated in favor of Bevy-based viewer).

- **`pitchvis_android_serial`**: Android-specific serial communication support.

### Forked Dependencies

- **`dagc_fork`**: Forked Digital Automatic Gain Control library.

- **`rustysynth_fork`**: Forked SoundFont synthesizer library.

## Data Flow

The application follows a pipeline architecture:

```
Audio Input → Ring Buffer → VQT → Analysis → Display
     ↓            ↓           ↓        ↓         ↓
  (cpal)    (AGC applied)  (freq)  (peaks)  (Bevy ECS)
```

1. **Audio Capture**: `pitchvis_audio` captures audio from the default input device and stores it in a ring buffer with AGC applied.

2. **Frequency Analysis**: `pitchvis_viewer` reads from the ring buffer and computes the Variable Q Transform (VQT) to convert time-domain audio to frequency-domain representation.

3. **Pitch Analysis**: `pitchvis_analysis` processes the VQT output to detect peaks, track pitches, and compute smoothed values for visualization.

4. **Visualization**: `pitchvis_viewer` uses Bevy's ECS architecture to render the analyzed data as 3D/2D graphics with various visual modes.

## Platform Support

- **Desktop** (Linux, macOS, Windows): Native binary using `cpal` for audio
- **Web** (WASM): Browser-based version with WebAudio API integration
- **Android**: Mobile application using `oboe` for low-latency audio

## Key Technologies

- **Bevy**: Game engine providing ECS, rendering, and cross-platform support
- **cpal**: Cross-platform audio I/O
- **rustfft**: Fast Fourier Transform implementation
- **wgpu**: Modern graphics API for rendering (via Bevy)

## Build Profiles

The workspace defines several build profiles optimized for different targets:

- `dev`: Fast compilation with moderate optimization (opt-level 1 for workspace, 3 for dependencies)
- `release`: Production builds with thin LTO and strip enabled
- `web-release`: WASM-optimized builds with size optimization (opt-level 'z')
