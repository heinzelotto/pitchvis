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
    /// Run the project (desktop only)
    Run {
        /// Build profile (dev uses dynamic linking for faster iteration)
        #[arg(short, long, default_value = "dev")]
        profile: Profile,

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
    /// Build for desktop (native)
    Desktop {
        /// Build in release mode
        #[arg(short, long)]
        release: bool,
        // TODO: always build in release, but toggle `--features bevy/dynamic_linking,bevy/file_watcher`
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
enum Profile {
    Dev,
    Release,
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
        Commands::Run { profile, args } => run(profile, args)?,
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

fn build_desktop(release: bool) -> Result<()> {
    println!("🔨 Building pitchvis for desktop...");

    let viewer_dir = project_root().join("pitchvis_viewer");

    let mut cmd = Command::new("cargo");
    cmd.current_dir(&viewer_dir)
        .arg("build")
        .arg("--bin")
        .arg("pitchvis");

    if release {
        cmd.arg("--release");
    } else {
        // Use dynamic linking for faster development builds
        cmd.arg("--features").arg("bevy/dynamic_linking");
    }

    let status = cmd.status().context("Failed to execute cargo build")?;

    if !status.success() {
        anyhow::bail!("Build failed");
    }

    let profile = if release { "release" } else { "debug" };
    let binary_path = viewer_dir.join(format!("../target/{}/pitchvis", profile));

    println!("✅ Build complete!");
    println!("   Binary: {}", binary_path.display());

    Ok(())
}

fn build_wasm(release: bool, rust_only: bool) -> Result<()> {
    println!("🔨 Building pitchvis for WASM...");

    let viewer_dir = project_root().join("pitchvis_viewer");
    let wasm_dir = viewer_dir.join("wasm");

    if !rust_only {
        // Build via npm which handles wasm-pack internally
        println!("📦 Installing npm dependencies...");
        let status = Command::new("npm")
            .current_dir(&wasm_dir)
            .arg("install")
            .status()
            .context("Failed to run npm install")?;

        if !status.success() {
            anyhow::bail!("npm install failed");
        }

        println!("🏗️  Building with npm...");
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
    } else {
        // Just build Rust code
        println!("🦀 Building Rust code only...");
        let profile = if release { "web-release" } else { "dev" };

        let status = Command::new("wasm-pack")
            .current_dir(&viewer_dir)
            .arg("build")
            .arg("--target")
            .arg("web")
            .arg("--out-dir")
            .arg("wasm/pkg")
            .arg("--profile")
            .arg(profile)
            .status()
            .context("Failed to run wasm-pack")?;

        if !status.success() {
            anyhow::bail!("wasm-pack build failed");
        }

        println!("✅ Rust WASM build complete!");
    }

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

fn run(profile: Profile, extra_args: Vec<String>) -> Result<()> {
    println!("🚀 Running pitchvis...");

    let viewer_dir = project_root().join("pitchvis_viewer");

    let mut cmd = Command::new("cargo");
    cmd.current_dir(&viewer_dir)
        .arg("run")
        .arg("--bin")
        .arg("pitchvis");

    match profile {
        Profile::Dev => {
            // Use dynamic linking for faster iteration
            cmd.arg("--features").arg("bevy/dynamic_linking");
        }
        Profile::Release => {
            cmd.arg("--release");
        }
    }

    if !extra_args.is_empty() {
        cmd.arg("--").args(&extra_args);
    }

    // Set default log level
    if std::env::var("RUST_LOG").is_err() {
        cmd.env("RUST_LOG", "error,pitchvis_analysis=debug");
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
