use std::time::Duration;

use log::debug;

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
///
/// y is the EMA of the series of x put into the update function
///
/// We calculate the decay factor ``alpha`` as 2 / (n + 1) where n is the number of
/// timesteps in the time horizon. Since our FPS is not constant, calculate this
/// factor for each update.
///
/// Note: Running the update with some timestep/2 twice is not the same as
/// running it once with timestep. In fact, in the limit of n, running it n times with
/// timestep/n is the same as running it once with 1 - exp(-timestep).
///
/// TODO: keep an eye on how consistent this is across different frame rates.
///
pub struct EmaMeasurement {
    time_horizon: Duration,
    y: f32,
}

impl EmaMeasurement {
    /// Create a new EmaMeasurement with a given averaging timespan and initial value
    pub fn new(time_horizon: Duration, value: f32) -> Self {
        Self {
            time_horizon,
            // alpha: 0.86, //(2.0 / (time_horizon.as_secs_f32() + 1.0)),
            y: value,
        }
    }

    pub fn update_with_timestep(&mut self, new_value: f32, timestep: Duration) {
        let n_horizon = self.time_horizon.as_secs_f32() / timestep.as_secs_f32();
        let alpha = 2.0 / (n_horizon + 1.0);
        // let alpha = 1.0 / (n_horizon);

        self.update_with_alpha(new_value, alpha);
        println!(
            "alpha: {alpha:.04}, timestep: {:.02}, new value: {:.02},  EMA: {:.02}",
            timestep.as_secs_f32(),
            new_value,
            &self.y
        );
    }

    pub fn update_one_second(&mut self, new_value: f32) {
        self.update_with_timestep(new_value, Duration::from_secs(1));
    }

    pub fn update_with_alpha(&mut self, new_value: f32, alpha: f32) {
        self.y = self.y + alpha * (new_value - self.y);
    }

    pub fn get(&self) -> f32 {
        self.y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_basic() {
        let mut ema_low_fps = EmaMeasurement::new(Duration::from_secs(1), 0.0);
        ema_low_fps.update_with_timestep(1.0, Duration::from_millis(250));
        ema_low_fps.update_with_timestep(1.0, Duration::from_millis(250));
        println!("Low FPS EMA: {:.02}", ema_low_fps.get());
        ema_low_fps.update_with_timestep(2.0, Duration::from_millis(250));
        ema_low_fps.update_with_timestep(2.0, Duration::from_millis(250));
        println!("Low FPS EMA: {:.02}", ema_low_fps.get());
        ema_low_fps.update_with_timestep(3.0, Duration::from_millis(250));
        ema_low_fps.update_with_timestep(3.0, Duration::from_millis(250));
        println!("Low FPS EMA: {:.02}", ema_low_fps.get());
        ema_low_fps.update_with_timestep(4.0, Duration::from_millis(250));
        ema_low_fps.update_with_timestep(4.0, Duration::from_millis(250));
        println!("Low FPS EMA: {:.02}", ema_low_fps.get());

        let mut ema_high_fps = EmaMeasurement::new(Duration::from_secs(1), 0.0);
        ema_high_fps.update_with_timestep(1.0, Duration::from_millis(125));
        ema_high_fps.update_with_timestep(1.0, Duration::from_millis(125));
        ema_high_fps.update_with_timestep(1.0, Duration::from_millis(125));
        ema_high_fps.update_with_timestep(1.0, Duration::from_millis(125));
        println!("High FPS EMA: {:.02}", ema_high_fps.get());
        ema_high_fps.update_with_timestep(2.0, Duration::from_millis(125));
        ema_high_fps.update_with_timestep(2.0, Duration::from_millis(125));
        ema_high_fps.update_with_timestep(2.0, Duration::from_millis(125));
        ema_high_fps.update_with_timestep(2.0, Duration::from_millis(125));
        println!("High FPS EMA: {:.02}", ema_high_fps.get());
        ema_high_fps.update_with_timestep(3.0, Duration::from_millis(125));
        ema_high_fps.update_with_timestep(3.0, Duration::from_millis(125));
        ema_high_fps.update_with_timestep(3.0, Duration::from_millis(125));
        ema_high_fps.update_with_timestep(3.0, Duration::from_millis(125));
        println!("High FPS EMA: {:.02}", ema_high_fps.get());
        ema_high_fps.update_with_timestep(4.0, Duration::from_millis(125));
        ema_high_fps.update_with_timestep(4.0, Duration::from_millis(125));
        ema_high_fps.update_with_timestep(4.0, Duration::from_millis(125));
        ema_high_fps.update_with_timestep(4.0, Duration::from_millis(125));
        println!("High FPS EMA: {:.02}", ema_high_fps.get());

        assert!((ema_low_fps.get() - ema_high_fps.get()).abs() < 0.05);
    }

    #[test]
    fn test_ema_limit() {
        /// Running the update with some timestep/2 twice is not the same as
        /// running it once with timestep. In fact, in the limit of n, running it n times with
        /// timestep/n is the same as running it once with 1 - exp(-timestep).
        const NEW_VALUE: f32 = 1.0;
        const TIME_HORIZON: Duration = Duration::from_secs(1);
        const N_HIGH: usize = 100;
        const TIMESTEP_HIGH_FPS: Duration = Duration::from_millis(500 / N_HIGH as u64);
        const N_MEDIUM: usize = 10;
        const TIMESTEP_MEDIUM_FPS: Duration = Duration::from_millis(500 / N_MEDIUM as u64);
        const N_LOW: usize = 3;
        const TIMESTEP_LOW_FPS: Duration = Duration::from_millis(500 / N_LOW as u64);

        let mut ema_high_fps = EmaMeasurement::new(TIME_HORIZON, 0.0);
        for _ in 0..N_HIGH {
            ema_high_fps.update_with_timestep(NEW_VALUE, TIMESTEP_HIGH_FPS);
        }
        println!("High FPS EMA: {:.02}", ema_high_fps.get());

        let mut ema_medium_fps = EmaMeasurement::new(Duration::from_secs(1), 0.0);
        for _ in 0..N_MEDIUM {
            ema_medium_fps.update_with_timestep(NEW_VALUE, TIMESTEP_MEDIUM_FPS);
        }
        println!("Medium FPS EMA: {:.02}", ema_medium_fps.get());

        let mut ema_low_fps = EmaMeasurement::new(Duration::from_secs(1), 0.0);
        for _ in 0..N_LOW {
            ema_low_fps.update_with_timestep(NEW_VALUE, TIMESTEP_LOW_FPS);
        }
        println!("Low FPS EMA: {:.02}", ema_low_fps.get());

        let mut ema_calculated = EmaMeasurement::new(Duration::from_secs(1), 0.0);
        let n_horizon_high_fps = TIME_HORIZON.as_secs_f32() / TIMESTEP_HIGH_FPS.as_secs_f32();
        let alpha_high_fps = 2.0 / (n_horizon_high_fps + 1.0);
        ema_calculated.update_with_alpha(NEW_VALUE, 1.0 - (-alpha_high_fps * N_HIGH as f32).exp());
        println!("Calculated FPS EMA: {:.02}", ema_calculated.get());

        // both will be approx 1 - 1/e
        assert!((ema_low_fps.get() - ema_high_fps.get()).abs() < 0.02);
        assert!((ema_low_fps.get() - ema_medium_fps.get()).abs() < 0.02);
        assert!((ema_low_fps.get() - ema_calculated.get()).abs() < 0.02);
    }
}
