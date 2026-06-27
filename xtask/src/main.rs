use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::process::Command;

#[derive(Parser)]
#[command(name = "xtask")]
#[command(about = "Build automation for PitchVis", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build the project
    Build {
        #[command(subcommand)]
        target: BuildTarget,
    },
    /// Run the project (desktop only, optimized for development with fast iteration)
    Run {
        /// Extra arguments to pass to the binary
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// Check the project (desktop only)
    Check {
        /// Extra arguments to pass to the binary
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// Clean build artifacts
    Clean {
        /// Target to clean (defaults to all)
        #[arg(value_enum)]
        target: Option<CleanTarget>,
    },
}

#[derive(Subcommand)]
enum BuildTarget {
    /// Build for desktop (native) - always builds in release mode per Bevy recommendations
    Desktop {
        /// Build in release mode (always enabled for desktop builds)
        #[arg(short, long, hide = true)]
        release: bool,
    },
    /// Build for WASM/web
    Wasm {
        /// Build in release mode
        #[arg(short, long)]
        release: bool,

        /// Just build the Rust code, skip npm build
        #[arg(long)]
        rust_only: bool,
    },
    /// Build for Android
    Android {
        /// Build in release mode
        #[arg(short, long)]
        release: bool,

        /// Skip copying assets and native libraries
        #[arg(long)]
        skip_setup: bool,
    },
}

#[derive(Clone, ValueEnum)]
enum CleanTarget {
    Desktop,
    Wasm,
    Android,
    All,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Build { target } => build(target)?,
        Commands::Run { args } => run(args)?,
        Commands::Check { args } => check(args)?,
        Commands::Clean { target } => clean(target.unwrap_or(CleanTarget::All))?,
    }

    Ok(())
}

fn build(target: BuildTarget) -> Result<()> {
    match target {
        BuildTarget::Desktop { release } => build_desktop(release)?,
        BuildTarget::Wasm { release, rust_only } => build_wasm(release, rust_only)?,
        BuildTarget::Android {
            release,
            skip_setup,
        } => build_android(release, skip_setup)?,
    }
    Ok(())
}

fn build_desktop(_release: bool) -> Result<()> {
    println!("🔨 Building pitchvis for desktop...");

    let viewer_dir = project_root().join("pitchvis_viewer");

    // Desktop always builds in release mode per Bevy recommendations
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&viewer_dir)
        .arg("build")
        .arg("--bin")
        .arg("pitchvis")
        .arg("--release");

    let status = cmd.status().context("Failed to execute cargo build")?;

    if !status.success() {
        anyhow::bail!("Build failed");
    }

    let binary_path = viewer_dir.join("../target/release/pitchvis");

    println!("✅ Build complete!");
    println!("   Binary: {}", binary_path.display());

    Ok(())
}

// Post-MVP wasm features the Rust wasm32 target (and the precompiled std) emit by
// default. wasm-opt validates input as MVP unless each is explicitly enabled, so these
// must be passed or it rejects the bulk-memory ops (`memory.copy`) coming out of std.
// `--enable-bulk-memory-opt` requires binaryen >= 119.
const WASM_OPT_FEATURE_ARGS: &[&str] = &[
    "--enable-bulk-memory",
    "--enable-bulk-memory-opt",
    "--enable-sign-ext",
    "--enable-nontrapping-float-to-int",
    "--enable-mutable-globals",
    "--enable-multivalue",
    "--enable-reference-types",
];

fn build_wasm(release: bool, rust_only: bool) -> Result<()> {
    println!("🔨 Building pitchvis for WASM...");

    let viewer_dir = project_root().join("pitchvis_viewer");
    let wasm_dir = viewer_dir.join("wasm");
    let profile = if release { "web-release" } else { "dev" };

    // 1) Compile Rust -> wasm + wasm-bindgen glue. We pass `--no-opt` and run wasm-opt
    //    ourselves below: wasm-pack 0.13 ignores the `wasm-opt` config of *custom* cargo
    //    profiles (such as `web-release`), so it would run a bare `-O` with no feature
    //    flags and fail to validate the bulk-memory ops std emits. `--target bundler
    //    --out-name index` matches what webpack imports (`import('./pkg')`).
    println!("🦀 Compiling Rust to WASM (profile: {profile})...");
    let status = Command::new("wasm-pack")
        .current_dir(&viewer_dir)
        .args([
            "build",
            "--target",
            "bundler",
            "--out-name",
            "index",
            "--out-dir",
            "wasm/pkg",
            "--no-opt",
            "--profile",
            profile,
        ])
        .status()
        .context("Failed to run wasm-pack")?;
    if !status.success() {
        anyhow::bail!("wasm-pack build failed");
    }

    // 2) Optimize with wasm-opt (release only; dev skips it to stay fast). Needs the
    //    feature flags above and a recent binaryen (>= 119) on PATH.
    if release {
        let wasm = wasm_dir.join("pkg").join("index_bg.wasm");
        let wasm_opt_out = wasm_dir.join("pkg").join("index_bg.opt.wasm");
        println!("📦 Optimizing with wasm-opt -Oz...");
        let status = Command::new("wasm-opt")
            .arg(&wasm)
            .arg("-o")
            .arg(&wasm_opt_out)
            .arg("-Oz")
            .args(WASM_OPT_FEATURE_ARGS)
            .status()
            .context("Failed to run wasm-opt (install binaryen >= 119 and put it on PATH)")?;
        if !status.success() {
            anyhow::bail!("wasm-opt failed");
        }
        std::fs::rename(&wasm_opt_out, &wasm).context("Failed to replace wasm with optimized output")?;
    }

    if rust_only {
        println!("✅ Rust WASM build complete!");
        return Ok(());
    }

    // 3) Bundle with webpack (consumes the prebuilt wasm/pkg).
    println!("📦 Installing npm dependencies...");
    let status = Command::new("npm")
        .current_dir(&wasm_dir)
        .arg("install")
        .status()
        .context("Failed to run npm install")?;
    if !status.success() {
        anyhow::bail!("npm install failed");
    }

    println!("🏗️  Bundling with webpack...");
    let status = Command::new("npm")
        .current_dir(&wasm_dir)
        .arg("run")
        .arg("build")
        .status()
        .context("Failed to run npm build")?;
    if !status.success() {
        anyhow::bail!("npm build failed");
    }

    println!("✅ WASM build complete!");
    println!("   Output: {}", wasm_dir.join("dist").display());
    Ok(())
}

fn build_android(release: bool, skip_setup: bool) -> Result<()> {
    println!("🔨 Building pitchvis for Android...");

    let viewer_dir = project_root().join("pitchvis_viewer");
    let android_dir = viewer_dir.join("android");

    // Check for required environment variables
    let android_sdk = std::env::var("ANDROID_SDK_ROOT")
        .or_else(|_| std::env::var("ANDROID_HOME"))
        .context("ANDROID_SDK_ROOT or ANDROID_HOME environment variable not set")?;

    let android_ndk = std::env::var("ANDROID_NDK_ROOT")
        .context("ANDROID_NDK_ROOT environment variable not set")?;

    println!("📱 Using Android SDK: {}", android_sdk);
    println!("🔧 Using Android NDK: {}", android_ndk);

    if !skip_setup {
        println!("📦 Setting up native libraries and assets...");

        // Copy libc++_shared.so
        let ndk_path = PathBuf::from(&android_ndk);
        let libcpp = ndk_path
            .join("toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so");
        let jni_libs = android_dir.join("app/src/main/jniLibs/arm64-v8a");

        std::fs::create_dir_all(&jni_libs).context("Failed to create jniLibs directory")?;

        std::fs::copy(&libcpp, jni_libs.join("libc++_shared.so"))
            .context("Failed to copy libc++_shared.so")?;

        // Copy shaders
        let assets_shaders = android_dir.join("app/src/main/assets/shaders");
        std::fs::create_dir_all(&assets_shaders)
            .context("Failed to create assets/shaders directory")?;

        let shader_dir = viewer_dir.join("assets/shaders");
        if shader_dir.exists() {
            for entry in std::fs::read_dir(&shader_dir)? {
                let entry = entry?;
                let dest = assets_shaders.join(entry.file_name());
                std::fs::copy(entry.path(), dest)?;
            }
        }
    }

    // Build native library with cargo-ndk
    println!("🦀 Building native library...");
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&viewer_dir)
        .arg("ndk")
        .arg("-t")
        .arg("arm64-v8a")
        .arg("-o")
        .arg("android/app/src/main/jniLibs/")
        .arg("rustc")
        .arg("--lib")
        .arg("--crate-type")
        .arg("cdylib");

    if release {
        cmd.arg("--release");
    }

    let status = cmd.status().context(
        "Failed to execute cargo ndk. Make sure cargo-ndk is installed: cargo install cargo-ndk",
    )?;

    if !status.success() {
        anyhow::bail!("cargo ndk build failed");
    }

    // Build Android app with Gradle
    println!("📱 Building Android app with Gradle...");
    let gradle_cmd = if cfg!(windows) {
        "gradlew.bat"
    } else {
        "./gradlew"
    };
    let gradle_task = if release {
        "bundleRelease"
    } else {
        "assembleDebug"
    };

    let status = Command::new(gradle_cmd)
        .current_dir(&android_dir)
        .arg(gradle_task)
        .status()
        .context("Failed to execute gradlew")?;

    if !status.success() {
        anyhow::bail!("Gradle build failed");
    }

    println!("✅ Android build complete!");
    if release {
        let bundle_path = android_dir.join("app/build/outputs/bundle/release/app-release.aab");
        println!("   Bundle: {}", bundle_path.display());
    } else {
        let apk_path = android_dir.join("app/build/outputs/apk/debug/app-debug.apk");
        println!("   APK: {}", apk_path.display());
    }

    Ok(())
}

fn run(extra_args: Vec<String>) -> Result<()> {
    println!("🚀 Running pitchvis...");

    let viewer_dir = project_root().join("pitchvis_viewer");

    // Run in release mode with dev features for fast iteration
    // (dynamic_linking speeds up compile times, file_watcher enables hot reload)
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&viewer_dir)
        .arg("run")
        .arg("--release")
        .arg("--features")
        .arg("bevy/dynamic_linking,bevy/file_watcher")
        .arg("--bin")
        .arg("pitchvis");

    if !extra_args.is_empty() {
        cmd.arg("--").args(&extra_args);
    }

    let status = cmd.status().context("Failed to execute cargo run")?;

    if !status.success() {
        anyhow::bail!("Run failed");
    }

    Ok(())
}

fn check(extra_args: Vec<String>) -> Result<()> {
    println!("Checking pitchvis...");

    let viewer_dir = project_root().join("pitchvis_viewer");

    let mut cmd = Command::new("cargo");
    cmd.current_dir(&viewer_dir)
        .arg("check")
        .arg("--release")
        .arg("--bin")
        .arg("pitchvis");

    if !extra_args.is_empty() {
        cmd.arg("--").args(&extra_args);
    }

    let status = cmd.status().context("Failed to execute cargo run")?;

    if !status.success() {
        anyhow::bail!("Run failed");
    }

    Ok(())
}

fn clean(target: CleanTarget) -> Result<()> {
    match target {
        CleanTarget::Desktop => {
            println!("🧹 Cleaning desktop build artifacts...");
            let status = Command::new("cargo")
                .current_dir(project_root().join("pitchvis_viewer"))
                .arg("clean")
                .status()?;
            if !status.success() {
                anyhow::bail!("Clean failed");
            }
        }
        CleanTarget::Wasm => {
            println!("🧹 Cleaning WASM build artifacts...");
            let wasm_dir = project_root().join("pitchvis_viewer/wasm");
            let dist = wasm_dir.join("dist");
            let pkg = wasm_dir.join("pkg");
            let node_modules = wasm_dir.join("node_modules");

            if dist.exists() {
                std::fs::remove_dir_all(&dist)?;
            }
            if pkg.exists() {
                std::fs::remove_dir_all(&pkg)?;
            }
            if node_modules.exists() {
                std::fs::remove_dir_all(&node_modules)?;
            }
        }
        CleanTarget::Android => {
            println!("🧹 Cleaning Android build artifacts...");
            let android_dir = project_root().join("pitchvis_viewer/android");
            let gradle_cmd = if cfg!(windows) {
                "gradlew.bat"
            } else {
                "./gradlew"
            };

            let status = Command::new(gradle_cmd)
                .current_dir(&android_dir)
                .arg("clean")
                .status()?;
            if !status.success() {
                anyhow::bail!("Gradle clean failed");
            }
        }
        CleanTarget::All => {
            clean(CleanTarget::Desktop)?;
            clean(CleanTarget::Wasm)?;
            clean(CleanTarget::Android)?;
        }
    }

    println!("✅ Clean complete!");
    Ok(())
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}
