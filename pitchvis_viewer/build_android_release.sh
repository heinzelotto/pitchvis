export CARGO_APK_RELEASE_KEYSTORE=~/.android/my-release-key.keystore
export CARGO_APK_RELEASE_KEYSTORE_PASSWORD=testtest
export ANDROID_SDK_ROOT=~/Android/Sdk
export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/26.3.11579264/
#export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/24.0.8215888/

# TODO: make this into a Makefile

cd android/

cp $ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so app/src/main/jniLibs/arm64-v8a/

mkdir -p app/src/main/assets/shaders
cp ../assets/shaders/* app/src/main/assets/shaders

cargo ndk -t arm64-v8a -o app/src/main/jniLibs/ --manifest-path ../Cargo.toml rustc --lib --crate-type cdylib --profile non-web-release --features bevy/tonemapping_luts && ./gradlew build && ./gradlew bundleRelease

# assembleRelease würde ein APK erzeugen, bundleRelease erzeugt ein AAB (?und apk)
