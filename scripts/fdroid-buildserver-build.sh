#!/usr/bin/env bash
#
# fdroid-buildserver-build.sh — build org.p1graph.pitchvis inside the official F-Droid
# buildserver Docker image (registry.gitlab.com/fdroid/fdroidserver:buildserver), exactly
# reproducing F-Droid CI, and copy the UNSIGNED APK out for out-of-band signing.
#
# This encodes every issue we hit getting the buildserver build to work:
#
#   1. UID mismatch. The container runs as 'vagrant' (uid 1000); the host user is usually
#      not uid 1000, so the bind-mounted fdroiddata fork is not writable by the build.
#      Fix: grant uid 1000 rwX on the fork via ACL (no chown, fully reversible with
#      `setfacl -R -b`). We run AS vagrant (not root) so $HOME/paths match F-Droid CI,
#      which matters for byte-reproducibility.
#
#   2. git "dubious ownership". The mounted .git is owned by the host uid, so git in the
#      container refuses it and `git diff --cached` falls back to --no-index mode, crashing
#      fdroidserver's is_dirty() check. Fix: `git config --global --add safe.directory '*'`
#      inside the container.
#
#   3. --on-server is REQUIRED for this recipe. Only with --on-server does fdroid run the
#      `sudo:` apt step (gcc/libc-dev/rustup) and auto-install NDK r28c (build.py only calls
#      auto_install_ndk + the sudo commands when onserver=True). Plain `fdroid build` skips
#      both and dies instantly with "Android NDK version 'r28c' could not be found!".
#
#   4. Output ownership. Build artifacts are written as uid 1000; the host user can't read
#      them through the 1000-owned intermediate dirs. Fix: copy the APK to the fork root and
#      chmod 644 so the host user can read it.
#
#   5. Two-phase Binaries:. With `Binaries:` set, `fdroid build` downloads the published APK
#      and HARD-FAILS if it 404s (build.py:1216). So:
#        - mode "build"  (default): comment Binaries out, produce the unsigned APK to sign.
#        - mode "verify": leave Binaries in; requires the signed APK already published at the
#          v<ver> GitHub release. Reports the apksigcopier byte-match + AllowedAPKSigningKeys.
#
# Usage:
#   scripts/fdroid-buildserver-build.sh            # build mode -> unsigned APK
#   scripts/fdroid-buildserver-build.sh verify     # reproducibility check vs published APK
#
# Env overrides: FDROIDDATA, FDROIDSERVER, IMAGE, APPID
set -euo pipefail

MODE="${1:-build}"
FDROIDDATA="${FDROIDDATA:-$HOME/Projects/fdroiddata}"
FDROIDSERVER="${FDROIDSERVER:-$HOME/fdroidserver}"
IMAGE="${IMAGE:-registry.gitlab.com/fdroid/fdroidserver:buildserver}"
APPID="${APPID:-org.p1graph.pitchvis}"
VAGRANT_UID=1000

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE="${FDROIDDATA}/metadata/${APPID}.yml"
[ -f "$RECIPE" ] || { echo "ERROR: recipe not found: $RECIPE" >&2; exit 1; }

# versionCode drives the published artifact name (Binaries: ..._%c.apk).
VC="$(grep -oE 'versionCode[[:space:]]+[0-9]+' "${REPO_ROOT}/pitchvis_viewer/android/app/build.gradle" | grep -oE '[0-9]+')"
OUT_NAME="${APPID}_${VC}-unsigned.apk"
GRADLE_APK="build/${APPID}/pitchvis_viewer/android/app/build/outputs/apk/release/app-release-unsigned.apk"

# 1) Make the fork writable by the container's vagrant uid (only if host uid differs).
#    ACLs persist, so this is a ONE-TIME grant. We must not re-run `setfacl -R` afterwards:
#    a previous build leaves build/ logs/ tmp/ and the copied APK owned by uid 1000, which the
#    host user does not own and therefore cannot setfacl (EPERM). So: skip if already granted.
if [ "$(id -u)" != "$VAGRANT_UID" ]; then
  if getfacl -pn "$FDROIDDATA" 2>/dev/null | grep -q "^user:${VAGRANT_UID}:"; then
    echo ">> ACL for uid ${VAGRANT_UID} already present on ${FDROIDDATA}; skipping setfacl"
  else
    echo ">> granting uid ${VAGRANT_UID} write access to ${FDROIDDATA} (ACL, one-time)"
    # First-time grant: no uid-1000 artifacts exist yet, so -R succeeds. The default ACL (d:)
    # makes everything the build later creates inherit the grant.
    setfacl -R -m "u:${VAGRANT_UID}:rwX" -m "d:u:${VAGRANT_UID}:rwX" "$FDROIDDATA"
  fi
  # The recipe gets rewritten by editors/rewritemeta via atomic rename, which drops the
  # directory's inherited ACL, so the container then can't read it. Re-grant on it each run.
  setfacl -m "u:${VAGRANT_UID}:rwX" "$RECIPE" 2>/dev/null || true
fi

# Cleanup on exit: (a) restore the recipe if we backed it up (build mode), and (b) restore HOST
# ownership of everything the container created. fdroid runs git on the bind-mounted fork as
# vagrant (uid 1000) and leaves .git/index + build outputs owned by 1000; the host user (uid != 1000)
# then can't manage the fork — host `git` fails with ".git/index: Permission denied". Only root can
# chown those back, so we do it via a throwaway root container after every run.
cleanup() {
  [ -f "${RECIPE}.bsbak" ] && mv -f "${RECIPE}.bsbak" "${RECIPE}" 2>/dev/null || true
  if [ "$(id -u)" != "$VAGRANT_UID" ]; then
    docker run --rm -u 0 --entrypoint chown -v "${FDROIDDATA}:/build" "$IMAGE" \
      -R "$(id -u):$(id -g)" /build >/dev/null 2>&1 \
      || echo ">> WARN: could not restore host ownership of ${FDROIDDATA}; run:  docker run --rm -u 0 --entrypoint chown -v ${FDROIDDATA}:/build ${IMAGE} -R $(id -u):$(id -g) /build" >&2
  fi
}
trap cleanup EXIT

# 5) Binaries: toggle for build mode (restored by cleanup() on exit).
if [ "$MODE" = "build" ]; then
  cp "$RECIPE" "${RECIPE}.bsbak"
  # Comment out the ENTIRE Binaries: block — the key line PLUS its indented continuation
  # line(s) that hold the URL. Commenting only the `Binaries:` key line orphans the indented
  # URL scalar and yields invalid YAML (ruamel: "expected <block end>"). Read the pristine
  # copy from .bsbak so re-runs are idempotent.
  awk '
    /^Binaries:/ { print "# " $0; inblk=1; next }
    inblk && /^[[:space:]]/ { print "# " $0; next }
    { inblk=0; print }
  ' "${RECIPE}.bsbak" > "$RECIPE"
fi

# Build/verify steps run inside the container.
# - build mode: copy the produced unsigned APK out to a host-readable path.
# - verify mode: remove any previously-built output for this versionCode first, so fdroid
#   build doesn't short-circuit (build.py trybuild returns "not necessary" if the output
#   already exists). This forces a genuine fresh rebuild that is then compared, via
#   apksigcopier, against the downloaded published APK (Binaries:) — the real determinism test.
PRE_STEP=""
COPY_STEP=""
if [ "$MODE" = "build" ]; then
  COPY_STEP="cp '${GRADLE_APK}' '/build/${OUT_NAME}'; chmod 644 '/build/${OUT_NAME}'; echo; echo 'UNSIGNED APK: ${FDROIDDATA}/${OUT_NAME}'; sha256sum '/build/${OUT_NAME}'"
elif [ "$MODE" = "verify" ]; then
  PRE_STEP="rm -f '/build/unsigned/${APPID}_${VC}.apk' '/build/repo/${APPID}_${VC}.apk'"
fi

docker rm -f pitchvis-fdroid-build >/dev/null 2>&1 || true
docker run --rm --name pitchvis-fdroid-build -u vagrant --entrypoint /bin/bash \
  -v "${FDROIDDATA}:/build:z" \
  -v "${FDROIDSERVER}:/home/vagrant/fdroidserver:Z" \
  "$IMAGE" -c "
set -e -o pipefail
. /etc/profile
export PATH=\"/home/vagrant/fdroidserver:\$PATH\"
export PYTHONPATH=\"/home/vagrant/fdroidserver\"
git config --global --add safe.directory '*'
cd /build
${PRE_STEP}
fdroid build --on-server --no-tarball -v -l ${APPID}
${COPY_STEP}
"

if [ "$MODE" = "build" ]; then
  echo
  echo "Next: sign out-of-band with the dedicated F-Droid key (run in a real terminal —"
  echo "apksigner prompts for the passphrase, which must NOT enter scripts/logs):"
  echo
  echo "  apksigner sign --ks ${REPO_ROOT}/pitchvis-fdroid.jks --ks-key-alias pitchvis \\"
  echo "    --out ${REPO_ROOT}/${APPID}_${VC}.apk ${FDROIDDATA}/${OUT_NAME}"
fi
