mod analysis_system;
mod app;
mod audio_system;
mod display_system;
#[cfg(feature = "ml")]
mod ml_system;
mod util;
mod vqt_system;

// reexport for main.rs. The entry points for android and wasm do not need to be reexported
#[cfg(not(any(target_os = "android", target_arch = "wasm32")))]
pub use app::desktop_app::main_fun;
