export CARGO_APK_RELEASE_KEYSTORE=~/.android/my-release-key.keystore
export CARGO_APK_RELEASE_KEYSTORE_PASSWORD=testtest
export ANDROID_SDK_ROOT=~/Android/Sdk
export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/25.2.9519653/
#export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/24.0.8215888/

cp /home/otheruser/Android/Sdk/ndk/25.2.9519653/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so app/src/main/jniLibs/arm64-v8a/
#cargo ndk -t arm64-v8a -o app/src/main/jniLibs/ build --release && 
./gradlew build && ./gradlew bundleRelease