fn main() {
    println!("cargo:rustc-link-search=native=/home/otheruser/Android/Sdk/ndk/25.2.9519653/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/33/");

    // //println!("cargo:rustc-link-search=native=/home/otheruser/Android/Sdk/ndk/25.2.9519653/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/arm-linux-androideabi/33/");

    //println!("cargo:rustc-link-search=native=/home/otheruser/Android/Sdk/ndk/24.0.8215888/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/31/");

    println!("cargo:rustc-link-lib=c++_shared");
}
