fn main() {
    use std::env;
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    if target_os == "android" {
        // The Android system stub libraries (libaaudio.so, libOpenSLES.so, ...)
        // live in an API-versioned subdir of the NDK sysroot. Derive that path
        // from cargo-ndk's environment and pick the highest available API level,
        // instead of hardcoding a machine-specific absolute NDK path (which is
        // not portable and breaks reproducible builds, e.g. on F-Droid).
        let libs = env::var("CARGO_NDK_SYSROOT_LIBS_PATH")
            .expect("CARGO_NDK_SYSROOT_LIBS_PATH not set; build the Android target via cargo-ndk");
        let api = std::fs::read_dir(&libs)
            .unwrap_or_else(|e| panic!("cannot read NDK sysroot libs dir {libs}: {e}"))
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter_map(|n| n.parse::<u32>().ok())
            .max()
            .expect("no API-level subdir found in NDK sysroot libs dir");
        println!("cargo:rustc-link-search=native={libs}/{api}/");

        println!("cargo:rustc-link-lib=c++_shared");
    }
}
