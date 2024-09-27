use super::AudioStream;
use super::RingBuffer;
use anyhow::Result;
use cpal::traits::*;
use itertools::Itertools;
use log::trace;

pub struct DesktopAudioStream {
    pub sr: u32,
    pub ring_buffer: std::sync::Arc<std::sync::Mutex<RingBuffer>>,
    pub stream: cpal::Stream,
    //pub agc: dagc::MonoAgc,
}

impl AudioStream for DesktopAudioStream {
    fn sr(&self) -> u32 {
        self.sr
    }
    fn ring_buffer(&self) -> std::sync::Arc<std::sync::Mutex<RingBuffer>> {
        self.ring_buffer.clone()
    }
    fn play(&self) -> Result<()> {
        //self.stream.play()?;

        Ok(())
    }
}

pub fn new_audio_stream(sr: u32, buf_size: usize) -> Result<DesktopAudioStream> {
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

    const PREFERRED_BUFFER_SIZE: usize = 256;
    let minimum_supported_buffer_size = device
        .supported_input_configs()
        .unwrap()
        .filter_map(|ref sc| {
            if sc.min_sample_rate() <= cpal::SampleRate(sr as u32)
                && sc.max_sample_rate() >= cpal::SampleRate(sr as u32)
            {
                match sc.buffer_size() {
                    cpal::SupportedBufferSize::Range { min, .. } => Some(*min),
                    _ => None,
                }
            } else {
                None
            }
        })
        .min()
        .expect("no supported minimum buffer size");
    let buffer_size = std::cmp::max(minimum_supported_buffer_size, PREFERRED_BUFFER_SIZE as u32);

    //let device = host.default_input_device().expect("no default input device");

    //println!("{}", device.name()?);

    let stream_config = cpal::StreamConfig {
        channels: 1u16,
        sample_rate: cpal::SampleRate(sr as u32),
        buffer_size: cpal::BufferSize::Fixed(buffer_size),
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
            // println!("data len: {}", data.len());

            let mut rb = ring_buffer_input_thread_clone
                .lock()
                .expect("locking failed");
            rb.buf.drain(..data.len());
            rb.buf.extend_from_slice(data);
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

    Ok(DesktopAudioStream {
        sr,
        ring_buffer,
        stream,
    })
}
