use anyhow::Result;
use cpal::traits::*;
use cpal::*;
use itertools::Itertools;
use log::trace;
#[cfg(target_arch = "wasm32")]
use rubato::{FftFixedIn, Resampler};
#[cfg(target_arch = "wasm32")]
use std::f32::EPSILON;
#[cfg(target_arch = "wasm32")]
use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use std::time::Duration;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;
#[cfg(target_arch = "wasm32")]
use web_sys::*;

#[cfg(target_arch = "wasm32")]
const MIN_CHANNELS: u16 = 1;
#[cfg(target_arch = "wasm32")]
const MAX_CHANNELS: u16 = 32;
#[cfg(target_arch = "wasm32")]
const MIN_SAMPLE_RATE: SampleRate = SampleRate(8_000);
#[cfg(target_arch = "wasm32")]
const MAX_SAMPLE_RATE: SampleRate = SampleRate(96_000);
const _DEFAULT_SAMPLE_RATE: SampleRate = SampleRate(44_100);
#[cfg(target_arch = "wasm32")]
const _MIN_BUFFER_SIZE: u32 = 1;
const _MAX_BUFFER_SIZE: u32 = u32::MAX;
const _DEFAULT_BUFFER_SIZE: usize = 2048;
#[cfg(target_arch = "wasm32")]
const SUPPORTED_SAMPLE_FORMAT: SampleFormat = SampleFormat::F32;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmUint8Array(Vec<u8>);

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmUint8Array {
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize) -> Self {
        let buffer = vec![0; size];
        Self { 0: buffer }
    }

    #[wasm_bindgen(getter, js_name = buffer)]
    pub fn buffer(&mut self) -> js_sys::Uint8Array {
        unsafe { js_sys::Uint8Array::view_mut_raw(self.0.as_mut_ptr(), self.0.len()) }
    }
}

pub struct RingBuffer {
    pub buf: Vec<f32>,
    pub gain: f32,
}

pub struct AudioStream {
    pub sr: usize,
    pub ring_buffer: std::sync::Arc<std::sync::Mutex<RingBuffer>>,
    pub stream: Stream,
    //pub agc: dagc::MonoAgc,
}

// Whether or not the given stream configuration is valid for building a stream.
#[cfg(target_arch = "wasm32")]
fn valid_config(conf: &StreamConfig, sample_format: SampleFormat) -> bool {
    conf.channels <= MAX_CHANNELS
        && conf.channels >= MIN_CHANNELS
        && conf.sample_rate <= MAX_SAMPLE_RATE
        && conf.sample_rate >= MIN_SAMPLE_RATE
        && sample_format == SUPPORTED_SAMPLE_FORMAT
}

#[cfg(target_arch = "wasm32")]
pub struct Stream {
    pub ctx: Arc<AudioContext>,
    //pub on_ended_closures: Vec<Arc<RwLock<Option<Closure<dyn FnMut()>>>>>,
    pub config: StreamConfig,
    //pub buffer_size_frames: usize,
}

#[cfg(target_arch = "wasm32")]
pub async fn build_input_stream_raw<D, E>(
    config: &StreamConfig,
    sample_format: SampleFormat,
    mut data_callback: D,
    _error_callback: E,
    _timeout: Option<Duration>,
) -> Result<Stream, BuildStreamError>
where
    D: FnMut(&[f32]) + Send + 'static,
    E: FnMut(StreamError) + Send + 'static,
{
    if !valid_config(config, sample_format) {
        return Err(BuildStreamError::StreamConfigNotSupported);
    }

    let target_sample_rate = config.sample_rate.0 as usize;

    let n_channels = config.channels as usize;

    // let buffer_size_frames = match config.buffer_size {
    //     BufferSize::Fixed(v) => {
    //         if v == 0 {
    //             return Err(BuildStreamError::StreamConfigNotSupported);
    //         } else {
    //             v as usize
    //         }
    //     }
    //     BufferSize::Default => DEFAULT_BUFFER_SIZE,
    // };
    // let _buffer_size_samples = buffer_size_frames * n_channels;
    // let _buffer_time_step_secs = buffer_time_step_secs(buffer_size_frames, config.sample_rate);

    //let data_callback = Arc::new(Mutex::new(Box::new(data_callback)));

    let window = web_sys::window().unwrap();

    let has_firefox = {
        let user_agent = window.navigator().user_agent().expect("user agent");
        let re = regex::Regex::new(r"(firefox)|(fxios)").unwrap();
        re.is_match(&user_agent.to_lowercase())
    };

    web_sys::console::log_1(&JsValue::from_str(&format!("has_firefox: {has_firefox}")));

    let microphone_sample_rate = if has_firefox {
        let contex = AudioContext::new().expect("default context for firefox");
        // There is currently no way to get the chosen microphone input sample rate in firefox. We just assume it is the same as the sample rate for output as chosen by the AudioContext default constructor.
        contex.sample_rate()
    } else {
        config.sample_rate.0 as f32
    };

    // Create the WebAudio stream.
    let mut stream_opts = AudioContextOptions::new();
    stream_opts.sample_rate(microphone_sample_rate);
    let ctx = Arc::new(
        AudioContext::new_with_context_options(&stream_opts).map_err(
            |err| -> BuildStreamError {
                let description = format!("{:?}", err);
                let err = BackendSpecificError { description };
                err.into()
            },
        )?,
    );

    let media_stream_js = JsFuture::from(
        window
            .navigator()
            .media_devices()
            .unwrap()
            .get_user_media_with_constraints(
                web_sys::MediaStreamConstraints::new()
                    .audio(&wasm_bindgen::JsValue::from_str(
                        "{sampleRate: 22050, channelCount: 1}",
                    ))
                    .video(&wasm_bindgen::JsValue::FALSE),
            )
            .unwrap(),
    )
    .await
    .expect("got media stream");
    let user_media = web_sys::MediaStream::unchecked_from_js(media_stream_js);

    web_sys::console::log_1(
        web_sys::MediaStreamTrack::unchecked_from_js(user_media.get_audio_tracks().at(0))
            .get_settings()
            .as_ref(),
    );

    let source_node = ctx
        .create_media_stream_source(&user_media)
        .expect("souce created");

    // TODO: what does subchunks do?
    // TODO: only need this for firefox
    let microphone_sample_rate_rounded = (microphone_sample_rate.round() + EPSILON) as usize;
    let mut resampler: FftFixedIn<f32> = FftFixedIn::new(
        microphone_sample_rate_rounded,
        config.sample_rate.0 as usize,
        128,
        1,
        1,
    )
    .expect("resampler");

    let mut bufs = vec![Vec::new()];
    let cb = Closure::wrap(Box::new(move |s: &JsValue| {
        let message_event = web_sys::MessageEvent::unchecked_from_js_ref(s);
        let array = js_sys::Float32Array::unchecked_from_js(message_event.data());
        let v = array.to_vec();
        let v_sl = vec![&v];

        if microphone_sample_rate_rounded == target_sample_rate {
            data_callback(&v)
        } else {
            // resample to desired frequency
            resampler
                .process_into_buffer(&v_sl, &mut bufs, None)
                .expect("resampling");
            //web_sys::console::log_1(array.as_ref());
            //let ici = cpal::InputCallbackInfo{timestamp: InputStreamTimestamp{callback: {StreamInstant {secs: 0, nanos: 0}}, capture : StreamInstant {secs: 0, nanos: 0}}};
            // web_sys::console::log_1(
            //     &JsValue::from_str(&format!("resample output: {:?}", bufs))
            // );
            if bufs[0].len() > 0 {
                data_callback(&bufs[0]);
            }
        }
    }) as Box<dyn FnMut(&JsValue)>);

    JsFuture::from(
        ctx.audio_worklet()
            .expect("has audio worklet")
            .add_module("basic_processor.js")
            .expect("add worklet"),
    )
    .await
    .expect("blabla");

    let worklet_node =
        AudioWorkletNode::new(&ctx, "basic_processor").expect("worklet node created");

    let port = worklet_node.port().expect("workletnode has port");
    let f = cb.as_ref().unchecked_ref();
    port.set_onmessage(Some(f));

    source_node
        .connect_with_audio_node(&worklet_node)
        .expect("connect1");
    worklet_node
        .connect_with_audio_node(&ctx.destination())
        .expect("connect2");
    std::mem::forget(worklet_node);
    std::mem::forget(cb);

    // A container for managing the lifecycle of the audio callbacks.
    //let mut on_ended_closures: Vec<Arc<RwLock<Option<Closure<dyn FnMut()>>>>> = Vec::new();

    // A cursor keeping track of the current time at which new frames should be scheduled.
    //let time = Arc::new(RwLock::new(0f64));

    Ok(Stream {
        ctx,
        //on_ended_closures: vec![], //on_ended_closures,
        config: config.clone(),
        //buffer_size_frames,
    })
}

impl AudioStream {
    #[cfg(target_arch = "wasm32")]
    pub async fn async_new(sr: usize, buf_size: usize) -> Result<Self> {
        dbg!(cpal::available_hosts());

        let host = cpal::default_host();

        let _device_names = host
            .devices()
            .expect("device query failed")
            .map(|d| d.name().expect("has no name"))
            .join(" ");

        //panic!("{:?}", device_names);

        let device = host.devices().unwrap().next().unwrap();

        //panic!("{}", device.supported_input_configs().unwrap().count());

        //let device = host.default_input_device().expect("no default input device");

        //println!("{}", device.name()?);

        let stream_config = cpal::StreamConfig {
            channels: 1u16,
            sample_rate: cpal::SampleRate(sr as u32),
            buffer_size: cpal::BufferSize::Default,
        };

        let mut ring_buffer = RingBuffer {
            buf: Vec::new(),
            gain: 0.0,
        };
        ring_buffer.buf.resize(buf_size, 0f32);
        let ring_buffer = std::sync::Mutex::from(ring_buffer);
        let ring_buffer = std::sync::Arc::new(ring_buffer);

        let ring_buffer_input_thread_clone = ring_buffer.clone();

        let mut agc = dagc::MonoAgc::new(0.07, 0.0001).expect("mono-agc creation failed");

        let data_callback = move |data: &[f32]| {
            if let Some(x) = data.iter().find(|x| !x.is_finite()) {
                log::warn!("bad audio sample encountered: {x}");
                return;
            }
            let sample_abs_sum = data.iter().map(|x| x.abs()).sum::<f32>();
            agc.freeze_gain(sample_abs_sum < 1e-4);

            let mut rb = ring_buffer_input_thread_clone
                .lock()
                .expect("locking failed");
            rb.buf.drain(..data.len());
            rb.buf.extend_from_slice(&data);
            let begin = rb.buf.len() - data.len();
            let sample_abs_sum = rb.buf[begin..].iter().map(|x| x.abs()).sum::<f32>();
            if sample_abs_sum > std::f32::EPSILON {
                agc.process(&mut rb.buf[begin..]);
                rb.gain = agc.gain();
            }
            trace!("gain: {}", agc.gain());
        };

        let stream = build_input_stream_raw(
            &stream_config,
            cpal::SampleFormat::F32,
            data_callback,
            move |err| panic!("{}", err),
            None,
        )
        .await?;

        Ok(Self {
            sr,
            ring_buffer,
            stream: stream,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(sr: usize, buf_size: usize) -> Result<Self> {
        dbg!(cpal::available_hosts());

        let host = cpal::default_host();

        // let device = host
        //     .devices()?
        //     .find(|d| d.name().unwrap() == "plughw:CARD=USB,DEV=0")
        //     .unwrap();

        let _device_names = host
            .devices()
            .expect("device query failed")
            .map(|d| d.name().expect("has no name"))
            .join(" ");

        //panic!("{:?}", device_names);

        let device = host.devices().unwrap().next().unwrap();

        //panic!("{}", device.supported_input_configs().unwrap().count());

        //let device = host.default_input_device().expect("no default input device");

        //println!("{}", device.name()?);

        let stream_config = cpal::StreamConfig {
            channels: 1u16,
            sample_rate: cpal::SampleRate(sr as u32),
            buffer_size: cpal::BufferSize::Default,
        };

        let mut ring_buffer = RingBuffer {
            buf: Vec::new(),
            gain: 0.0,
        };
        ring_buffer.buf.resize(buf_size, 0f32);
        let ring_buffer = std::sync::Mutex::from(ring_buffer);
        let ring_buffer = std::sync::Arc::new(ring_buffer);

        let ring_buffer_input_thread_clone = ring_buffer.clone();

        let mut agc = dagc::MonoAgc::new(0.07, 0.0001).expect("mono-agc creation failed");

        let stream = device.build_input_stream(
            &stream_config,
            move |data: &[f32], _info: &cpal::InputCallbackInfo| {
                if let Some(x) = data.iter().find(|x| !x.is_finite()) {
                    log::warn!("bad audio sample encountered: {x}");
                    return;
                }
                let sample_sq_sum = data.iter().map(|x| x.powi(2)).sum::<f32>();
                agc.freeze_gain(sample_sq_sum < 1e-6);

                let mut rb = ring_buffer_input_thread_clone
                    .lock()
                    .expect("locking failed");
                rb.buf.drain(..data.len());
                rb.buf.extend_from_slice(&data);
                let begin = rb.buf.len() - data.len();
                agc.process(&mut rb.buf[begin..]);
                rb.gain = agc.gain();
                trace!(
                    "gain: {}, avg_rms: {}",
                    agc.gain(),
                    sample_sq_sum / data.len() as f32
                );
            },
            move |err| panic!("{}", err),
            None,
        )?;

        Ok(Self {
            sr,
            ring_buffer,
            stream,
        })
    }

    pub fn play(&self) -> Result<()> {
        //self.stream.play()?;

        Ok(())
    }
}
