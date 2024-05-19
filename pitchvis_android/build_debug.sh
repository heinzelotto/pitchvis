export CARGO_APK_RELEASE_KEYSTORE=~/.android/my-release-key.keystore
export CARGO_APK_RELEASE_KEYSTORE_PASSWORD=testtest
export ANDROID_SDK_ROOT=~/Android/Sdk
export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/26.3.11579264/
#export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/24.0.8215888/

# TODO: make this into a Makefile

cp /home/otheruser/Android/Sdk/ndk/26.3.11579264/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so app/src/main/jniLibs/arm64-v8a/

mkdir -p app/src/main/assets/shaders
cp assets/shaders/* app/src/main/assets/shaders

cargo ndk -t arm64-v8a -o app/src/main/jniLibs/ build && ./gradlew build && ./gradlew installDebug
