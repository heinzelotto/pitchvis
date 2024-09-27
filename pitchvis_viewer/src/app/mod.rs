#[cfg(target_os = "android")]
pub(crate) mod android_app;
mod common;
#[cfg(not(any(target_os = "android", target_arch = "wasm32")))]
pub(crate) mod desktop_app;
#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm_app;

pub use common::SettingsState;
