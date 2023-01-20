use num_complex::{Complex32, ComplexFloat};
use rustfft::num_traits::Zero;
use rustfft::FftPlanner;
use std::f32::consts::PI;

pub struct Cqt {
    sr: usize,
    n: usize,
    min_freq: f32,
    buckets_per_octave: usize,
    octaves: usize,
    cqt_kernel: sprs::CsMat<Complex32>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl Cqt {
    pub fn new(
        sr: usize,
        n: usize,
        min_freq: f32,
        buckets_per_octave: usize,
        octaves: usize,
        sparsity_quantile: f32,
    ) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n as usize);

        let cqt_kernel = cqt_kernel(
            sr,
            n,
            min_freq,
            buckets_per_octave,
            octaves,
            sparsity_quantile,
        );

        Self {
            sr,
            n,
            min_freq,
            buckets_per_octave,
            octaves,
            cqt_kernel,
            fft,
        }
    }

    pub fn calculate_cqt_instant_in_db(&self, x: &[f32]) -> Vec<f32> {
        let mut x = x
            .iter()
            .map(|f| rustfft::num_complex::Complex32::new(*f, 0.0))
            .collect::<Vec<rustfft::num_complex::Complex32>>();
        self.fft.process(&mut x);

        // TODO: harmonic/percussion source separation possible with our one-sided spectrum?

        // signal fft'd
        let mut x_cqt = vec![Complex32::zero(); self.buckets_per_octave * self.octaves];

        sprs::prod::mul_acc_mat_vec_csr(self.cqt_kernel.view(), x, &mut x_cqt);
        let x_cqt: Vec<f32> = x_cqt
            .iter()
            .map(|z| (z.abs() * z.abs()).log(10.0).max(-2.0) + 2.0)
            .collect();

        // TODO: proper amp -> db conversion

        // TODO: median filter

        x_cqt
    }
}

fn cqt_kernel(
    sr: usize,
    n: usize,
    min_freq: f32,
    buckets_per_octave: usize,
    octaves: usize,
    sparsity_quantile: f32,
) -> sprs::CsMat<Complex32> {
    let num_buckets = buckets_per_octave * octaves;
    // TODO: use different window lengths for every pitch
    let window = apodize::hanning_iter(n).collect::<Vec<f64>>();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let mut a = sprs::TriMat::new((num_buckets as usize, n as usize));

    // fill a with the 'atoms', i. e. morlet wavelets
    for k in 0..num_buckets {
        let f_k = min_freq * 2.0_f32.powf(k as f32 / buckets_per_octave as f32);
        let mut v = Vec::new();
        for i in 0..n {
            v.push(
                1.0 / (n as f32)
                    * (window[i] as f32)
                    * (-num_complex::Complex32::i() * 2.0 * PI * (i as f32) * f_k / (sr as f32))
                        .exp(),
            );
        }

        // transform wavelets into frequency space
        fft.process(&mut v);

        // the complex conjugate is what we later need
        v = v.iter_mut().map(|z| z.conj()).collect();

        // filter all values smaller than some value and use sparse arrays
        let mut v_abs = v.iter().map(|z| z.abs()).collect::<Vec<f32>>();
        v_abs.sort_by(|a, b| {
            if a == b {
                std::cmp::Ordering::Equal
            } else if a < b {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });
        let cutoff_value = v_abs[(n as f32 * sparsity_quantile) as usize];
        let mut cnt = 0;
        v.iter_mut().for_each(|z| {
            if z.abs() < cutoff_value {
                *z = Complex32::zero();
                cnt += 1;
            }
        });
        println!("for k {k} erased {cnt} points below {cutoff_value}");

        // fill the kernel matrix
        for (i, z) in v.iter().enumerate() {
            if !z.is_zero() {
                a.add_triplet(k, i, *z);
            }
        }
    }

    a.to_csr()
}
