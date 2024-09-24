pub fn arg_min(sl: &[f32]) -> usize {
    // we have no NaNs
    sl.iter()
        .enumerate()
        .fold(
            (0, f32::MAX),
            |cur, x| if *x.1 < cur.1 { (x.0, *x.1) } else { cur },
        )
        .0
}

pub fn arg_max(sl: &[f32]) -> usize {
    // we have no NaNs
    sl.iter()
        .enumerate()
        .fold(
            (0, f32::MIN),
            |cur, x| if *x.1 > cur.1 { (x.0, *x.1) } else { cur },
        )
        .0
}

/// Exponential moving average helper
pub struct EmaMeasurement {
    alpha: f32,
    value: f32,
}

impl EmaMeasurement {
    pub fn new(alpha: f32, value: f32) -> Self {
        Self { alpha, value }
    }

    pub fn update(&mut self, new_value: f32) {
        self.value = self.alpha * new_value + (1.0 - self.alpha) * self.value;
    }

    pub fn get(&self) -> f32 {
        self.value
    }
}
