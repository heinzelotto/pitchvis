# dependencies

- keep everything on rubato 0.12 for now; do NOT bump to 0.16. Upgrading pitchvis_audio (and
  anything else) to rubato 0.16 produced garbled audio — the resampler interface appears to have
  changed semantically without changing syntactically (a silent breaking change: same signatures,
  different behavior). pitchvis_viewer's Android system-audio capture is pinned to 0.12 to match.
  Before any future 0.16 upgrade, figure out the semantic difference (chunk framing? channel
  layout? overlap handling?) and fix all call sites together. See the TODO in
  pitchvis_audio/Cargo.toml.

# build system

- discover -laaudio instead of having to hardcode it
- make the NDK sysroot link-search path in pitchvis_viewer/build.rs configurable (e.g. derive from ANDROID_NDK_ROOT) instead of hardcoding the home directory and NDK version

# analysis

- make "short" vqt smoothing shorter, either absolutely or with the calmness influence
