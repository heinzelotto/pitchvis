#!/bin/bash
# Script to check desktop build without WASM dependencies
# Usage: ./check-desktop.sh

set -e

# Set up stub pkg-config files to bypass missing system dependencies
export PKG_CONFIG_PATH=/tmp/stub-pkgconfig

echo "Checking desktop packages..."
cargo check \
  -p pitchvis_analysis \
  -p pitchvis_audio \
  -p pitchvis_colors \
  -p pitchvis_serial \
  -p pitchvis_viewer \
  -p pitchvis_train \
  -p dagc \
  -p rustysynth \
  --all-targets

echo "✓ Desktop build check successful!"
