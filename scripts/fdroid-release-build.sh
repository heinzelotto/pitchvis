#!/usr/bin/env bash
#
# fdroid-release-build.sh — canonical reproducible release build + signing for the
# F-Droid developer-signed (reproducible) distribution of org.p1graph.pitchvis.
#
# This MIRRORS the F-Droid recipe's `prebuild` + `gradle: assembleRelease` steps
# (metadata/org.p1graph.pitchvis.yml). The reproducibility-critical settings — Rust
# 1.94.0, cargo-ndk 4.1.2, NDK r28c, and the --remap-path-prefix flags — must stay
# in sync with the recipe; the recipe is the source of truth that F-Droid CI runs.
#
# For a byte-match against F-Droid's build, run this INSIDE the official buildserver
# image (registry.gitlab.com/fdroid/fdroidserver:buildserver) so the source path,
# CARGO_HOME, NDK path, gradle and AGP are identical. The simplest equivalent is to
# let `fdroid build org.p1graph.pitchvis` produce the unsigned APK and then run only
# the signing step below on its output.
#
# We build UNSIGNED (no keystore.properties present) and sign out-of-band with
# apksigner. F-Droid does the same: it builds unsigned from source and verifies our
# published APK's non-signature content is byte-identical (apksigcopier), then ships
# our-signed binary.
#
# Required env:
#   ANDROID_NDK_ROOT   path to NDK r28c (in the buildserver this is $NDK)
#   KEYSTORE           path to the dedicated F-Droid release keystore (pitchvis-fdroid.jks)
#   KS_ALIAS           key alias inside that keystore (default: pitchvis)
# Optional:
#   OUT                output path for the signed APK (default: ./org.p1graph.pitchvis_<vc>.apk)
#
set -euo pipefail

: "${ANDROID_NDK_ROOT:=${NDK:-}}"
: "${KS_ALIAS:=pitchvis}"
if [ -z "${ANDROID_NDK_ROOT}" ]; then
  echo "ERROR: set ANDROID_NDK_ROOT (or NDK) to the r28c NDK path" >&2; exit 1
fi

# Repo root = two levels up from this script (scripts/ -> repo root).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# --- toolchain (release-only pin; does not touch your everyday default) ----------
rustup default 1.94.0
rustup target add aarch64-linux-android
cargo install cargo-ndk --version 4.1.2 --locked

# --- reproducibility: neutralise machine-specific absolute paths -----------------
# Setting the target rustflags env REPLACES (does not merge with) the
# [target.aarch64-linux-android].rustflags in .cargo/config.toml, so the existing
# `-Clink-arg=-lc++_shared` MUST be re-included here or linking breaks.
export CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS="-Clink-arg=-lc++_shared --remap-path-prefix=${PWD}=/build --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/cargo"

# --- native cdylib via cargo-ndk -------------------------------------------------
cd pitchvis_viewer
ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT}" cargo ndk -t arm64-v8a \
  -o android/app/src/main/jniLibs/ rustc --release --lib --crate-type cdylib

# --- stage libc++_shared.so + shaders that gradle packages -----------------------
cp "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" \
   android/app/src/main/jniLibs/arm64-v8a/
mkdir -p android/app/src/main/assets/shaders
cp assets/shaders/* android/app/src/main/assets/shaders/

# --- unsigned release APK (no keystore.properties => signingConfig null) ---------
# Temporarily move any local keystore.properties aside so gradle's signingConfig is
# null and the APK is built UNSIGNED, then restore it on exit. That file is gitignored
# and holds the local dev/debug-key config; we never want it to sign the F-Droid APK —
# we sign out-of-band below with the dedicated pitchvis-fdroid.jks key instead.
cd android
if [ -f keystore.properties ]; then
  mv keystore.properties keystore.properties.fdroid-bak
  trap 'mv -f keystore.properties.fdroid-bak keystore.properties 2>/dev/null || true' EXIT
fi
./gradlew assembleRelease
UNSIGNED="$(ls app/build/outputs/apk/release/*.apk | head -n1)"

# --- read versionCode to name the artifact the way Binaries: expects -------------
VC="$(grep -oE 'versionCode[[:space:]]+[0-9]+' app/build.gradle | grep -oE '[0-9]+')"
: "${OUT:=${REPO_ROOT}/org.p1graph.pitchvis_${VC}.apk}"

# --- sign out-of-band with the dedicated F-Droid key -----------------------------
if [ -z "${KEYSTORE:-}" ]; then
  echo "Built UNSIGNED: ${PWD}/${UNSIGNED}"
  echo "Set KEYSTORE=/path/to/pitchvis-fdroid.jks to also sign. Skipping signing." >&2
  exit 0
fi
# Sign so the published APK byte-matches F-Droid's unsigned rebuild except for the signature
# (apksigcopier requirement). Per https://f-droid.org/en/docs/Reproducible_Builds/ :
#   * Use apksigner from BUILD-TOOLS 34. Newer apksigner (35+) re-zipaligns the APK with its
#     own scheme, shifting entry offsets a few bytes vs AGP's packaging, and apksigcopier
#     "will fail" to reconcile it. build-tools 34 preserves the existing AGP alignment, so
#     signing only inserts the APK Signing Block. (bt35+ CAN be made to work with
#     `--alignment-preserved true`, but the F-Droid-sanctioned path is bt34.)
#   * --v1-signing-enabled false : v1 (JAR) signing injects META-INF/{MANIFEST.MF,*.SF,*.RSA}
#     zip entries the unsigned rebuild lacks; v2/v3 digests then cover them. v1 is useless at
#     minSdk 29 (v2/v3 cover API 24+/28+).
: "${APKSIGNER:=${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}}/build-tools/34.0.0/apksigner}"
[ -x "${APKSIGNER}" ] || { echo "ERROR: build-tools 34 apksigner not found at ${APKSIGNER} (set APKSIGNER=/path/to/build-tools/34.0.0/apksigner)" >&2; exit 1; }
"${APKSIGNER}" sign --ks "${KEYSTORE}" --ks-key-alias "${KS_ALIAS}" \
  --v1-signing-enabled false \
  --out "${OUT}" "${UNSIGNED}"
"${APKSIGNER}" verify --print-certs "${OUT}" | grep -i 'SHA-256'
echo "Signed APK: ${OUT}"
echo "Upload this as the v$(grep -oE 'versionName \"[0-9.]+\"' app/build.gradle | grep -oE '[0-9.]+') release asset, named org.p1graph.pitchvis_${VC}.apk"
echo "Put the cert SHA-256 above (lowercase hex, no colons) into AllowedAPKSigningKeys in the recipe."
