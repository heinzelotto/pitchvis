use anyhow::Result;

mod train;

pub fn main() -> Result<()> {
    train::train()?;

    Ok(())
}
