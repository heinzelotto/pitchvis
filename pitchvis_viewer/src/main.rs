use anyhow::Result;

#[cfg(not(target_arch = "wasm32"))]
pub fn main() -> Result<()> {
    pitchvis_viewer_lib::main_fun()
}

#[cfg(target_arch = "wasm32")]
pub fn main() -> Result<()> {
    // Nothing. But we need this function to be defined as we can't disable the rlib crate-type based on target.
    Ok(())
}
