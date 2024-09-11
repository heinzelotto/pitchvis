This is the main crate of the project. It contains the pitchvis visualization software for desktop, android and as a webapp.

# Desktop app

To run the desktop app, run the pitchvis executable

`cargo r --features bevy/dynamic_linking --release --bin pitchvis`

If you have trained a model.pt using `pitchvis_train`, it can be run with machine learning base frequency emphasis:

`LD_LIBRARY_PATH="$LIBTORCH/lib" LIBTORCH_STATIC=0 cargo r --features bevy/dynamic_linking,ml --release --bin pitchvis`

Press <kbd>Space</kbd> to cycle through view modes.

# Android app

```bash
rustup target add aarch64-linux-android
cargo install cargo-ndk
```

Then adapt and run `build_android_debug.sh` or `build_android_release.sh`.

For a release build modify `app/build.gradle`.

Tap to cycle through view modes.

# Webapp

You can check it out at https://www.p1graph.org/pitchvis/ 

To build:

```bash
npm install
npm run build # or `npm run serve` for a locally hosted
```
