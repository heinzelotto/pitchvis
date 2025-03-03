export CARGO_APK_RELEASE_KEYSTORE=~/.android/my-release-key.keystore
export CARGO_APK_RELEASE_KEYSTORE_PASSWORD=testtest
export ANDROID_SDK_ROOT=~/Android/Sdk
export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/26.3.11579264/
#export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/24.0.8215888/

# TODO: make this into a Makefile
cargo ndk -t arm64-v8a -o app/src/main/jniLibs/ build --release && ./gradlew build && ./gradlew bundleRelease && adb install app/build/outputs/apk/release/app-release.apk
