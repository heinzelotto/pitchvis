use anyhow::Result;
pub mod analysis;
pub mod cqt;
pub mod util;

// TODO: make library arguments
pub const SR: usize = 22050;
pub const BUFSIZE: usize = 2 * SR;
pub const N_FFT: usize = 2 * 16384;
pub const FREQ_A1: f32 = 55.0;
pub const BUCKETS_PER_OCTAVE: usize = 12 * 5;
pub const OCTAVES: usize = 6; // TODO: extend to 6
pub const SPARSITY_QUANTILE: f32 = 0.999;
pub const Q: f32 = 1.0;
pub const GAMMA: f32 = 5.0;
