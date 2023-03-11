#[cfg(target_arch = "wasm32")]
use js_sys::{Function, Promise};

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

#[cfg(target_arch = "wasm32")]
pub async fn sleep(delay: std::time::Duration) {
        let mut cb = |resolve: js_sys::Function, _reject: js_sys::Function| {
            web_sys::window()
                .unwrap()
                .set_timeout_with_callback_and_timeout_and_arguments_0(
                    &resolve,
                    delay.as_millis() as i32,
                )
                .unwrap();
        };
    
        let p = js_sys::Promise::new(&mut cb);
    
        wasm_bindgen_futures::JsFuture::from(p).await.unwrap();
    }
    