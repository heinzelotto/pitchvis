# PitchVis Viewer Architecture

## Overview

`pitchvis_viewer` is the main application crate for PitchVis. It integrates audio input, signal processing, and visualization using the Bevy game engine. The crate supports multiple platforms (desktop, web, Android) through conditional compilation.

## Module Structure

### Entry Points

- **`main.rs`**: Desktop binary entry point (delegates to `lib.rs`)
- **`lib.rs`**: Library exports and module declarations
- **`app/`**: Platform-specific application setup
  - `desktop_app.rs`: Native desktop application (Linux, macOS, Windows)
  - `wasm_app.rs`: Web browser application (WebAssembly)
  - `android_app.rs`: Android mobile application
  - `common.rs`: Shared UI components and systems (FPS counter, settings, input handling)

### Core Systems

The application is organized into Bevy systems, each handling a specific aspect of the pipeline:

#### 1. Audio System (`audio_system.rs`)

**Purpose**: Bridge between `pitchvis_audio` crate and Bevy resources.

**Resources**:
- `AudioBufferResource`: Wraps the ring buffer from `pitchvis_audio` in a Bevy resource

**Responsibilities**:
- Provides access to the audio ring buffer for other systems
- The actual audio capture happens in `pitchvis_audio` on a separate thread

#### 2. VQT System (`vqt_system.rs`)

**Purpose**: Compute Variable Q Transform on incoming audio data.

**Resources**:
- `VqtResource`: Contains the VQT calculator
- `VqtResultResource`: Stores the computed VQT spectrum and gain

**Key Functions**:
- `update_vqt()`: Reads audio from ring buffer, computes VQT, stores result in dB scale

**Data Flow**:
```
AudioBufferResource → VQT calculation → VqtResultResource
```

#### 3. Analysis System (`analysis_system.rs`)

**Purpose**: Perform pitch analysis on VQT output.

**Resources**:
- `AnalysisStateResource`: Wraps `pitchvis_analysis::AnalysisState`

**Key Functions**:
- `update_analysis_state_to_system()`: Processes VQT data to detect peaks, track pitches, smooth values, and estimate tuning drift

**Processing**:
- Peak detection
- Pitch tracking with continuity
- Temporal smoothing
- Tuning grid alignment

#### 4. Display System (`display_system/`)

**Purpose**: Render the analyzed audio data using Bevy's rendering pipeline.

**Submodules**:
- `mod.rs`: System definitions and types
- `setup.rs`: Initial scene setup (camera, lights, entities)
- `update.rs`: Per-frame rendering updates
- `material.rs`: Custom shader materials (e.g., `NoisyColorMaterial`)
- `util.rs`: Helper functions for visualization

**Resources**:
- `CylinderEntityListResource`: Maintains mapping between frequency bins and rendered entities

**Components**:
- `PitchBall`: Spheres representing detected pitches
- `BassCylinder`: Cylinders for bass visualization
- `SpiderNetSegment`: Grid lines for spatial reference
- `Spectrum`: 2D spectrum display
- `PitchNameText`: Labels for pitch names

**Visual Modes**:
- `Full`: Complete visualization with all elements
- `Zen`: Minimal view without pitch labels
- `Performance`: Optimized for speed and precision
- `Galaxy`: Pitch balls only with dark background

**Display Modes**:
- `Normal`: Standard visualization
- `Debugging`: Shows FPS counter, latency metrics, and debug UI

#### 5. ML System (`ml_system.rs`)

**Purpose**: Optional machine learning inference (enabled with `ml` feature).

**Note**: Currently optional and requires PyTorch (`tch` crate).

### Common UI Systems (`app/common.rs`)

**Settings Management**:
- `SettingsState`: Persistent user settings (display mode, visuals mode, FPS limit)
- Stored in TOML format using `bevy-persistent`

**UI Components**:
- FPS counter with color-coded performance indicators
- Audio latency and chunk size display
- VQT latency monitoring
- Tuning drift indicator
- Bloom effect controls (debugging mode)
- Interactive buttons for mode switching

**Input Handling**:
- Keyboard: Space/Escape for mode switching
- Mouse: Click to toggle display mode
- Touch: Tap to cycle modes (mobile)

## Bevy ECS Architecture

The application uses Bevy's Entity Component System (ECS):

### System Execution Order

```
Startup:
  - setup_display
  - setup_fps_counter
  - setup_buttons
  - setup_bloom_ui
  - setup_analysis_text

Update (each frame):
  - close_on_esc
  - update_vqt_system
  - update_analysis_state_system (after VQT)
  - update_ml_system (after analysis, if enabled)
  - update_display_system (after analysis/ML)
  - update_fps_text_system
  - fps_counter_showhide
  - update_button_system
  - button_showhide
  - user_input_system
  - update_bloom_settings
  - update_analysis_text_system
  - analysis_text_showhide
  - set_frame_limiter_system
```

### Resource Flow

```
AudioBufferResource
    ↓
VqtResource + VqtResultResource
    ↓
AnalysisStateResource
    ↓
Display Components (PitchBall, BassCylinder, etc.)
```

## Configuration

**Default Constants** (desktop_app.rs):
- `BUFSIZE`: 32768 samples (2 * 16384)
- `DEFAULT_FPS`: 60 Hz

**Persistent Settings**:
- Stored in platform-specific config directory
- Location: `{config_dir}/pitchvis/settings.toml`
- Contains display mode, visuals mode, and FPS limit preferences

## Platform-Specific Details

### Desktop

- Uses `cpal` for audio input
- Stores settings in OS config directory (via `dirs` crate)
- Supports window management and keyboard/mouse input

### WASM (Web)

- Uses WebAudio API through `web-sys` bindings
- Audio captured via MediaStream API
- Compiled with `wasm-bindgen`
- Optimized for size (`opt-level = 'z'`)

### Android

- Uses `oboe` for low-latency audio
- Integrates with Android Activity lifecycle
- Touch-based interaction

## Performance Considerations

**Frame Rate Control**:
- Configurable FPS limiting (30, 60, or unlimited)
- Uses Bevy's `WinitSettings` for reactive rendering

**Optimization**:
- VQT computation is the primary bottleneck
- Analysis smoothing adds latency but improves visual stability
- Display system uses entity visibility toggling to reduce rendering load

**Latency Metrics**:
- Audio latency: Time between audio capture and buffer availability
- Audio chunk size: Determines update granularity
- VQT latency: Processing delay in VQT computation

## Rendering Pipeline

1. **VQT Computation**: Convert audio to frequency spectrum
2. **Peak Detection**: Identify prominent frequencies
3. **Color Calculation**: Map pitches to colors via `pitchvis_colors`
4. **Entity Updates**: Update transforms, materials, and visibility
5. **Bevy Rendering**: GPU-accelerated rendering via wgpu

**Material System**:
- `NoisyColorMaterial`: Custom material with procedural noise for visual interest
- Standard `ColorMaterial`: For simpler elements like cylinders
- Bloom post-processing for glow effects (configurable in debug mode)

## Extension Points

- **New Visual Modes**: Add variants to `VisualsMode` enum and implement rendering logic in `display_system/update.rs`
- **Additional Analysis**: Extend `AnalysisState` in `pitchvis_analysis` crate
- **Custom Materials**: Create new material types in `display_system/material.rs`
- **Platform Support**: Add new platform-specific entry points in `app/` directory
