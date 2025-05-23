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

    let device = match host.default_input_device() {
        Some(dev) => dev,
        None => {
            let device_names = host
                .devices()
                .expect("device query failed")
                .enumerate()
                .map(|(i, d)| format!("{}: {}", i, d.name().expect("has no name")))
                .join("\n");

            println!("No default input device found. Available devices:");
            println!("{}", device_names);
            panic!("Quitting due to no default device found.");
        }
    };

    println!(
        "Default input device chosen: \"{}\"",
        device.name().unwrap()
    );

    const PREFERRED_BUFFER_SIZE: usize = 256;
    let minimum_supported_buffer_size = device
        .supported_input_configs()
        .unwrap()
        .filter_map(|ref sc| {
            if sc.min_sample_rate() <= cpal::SampleRate(sr)
                && sc.max_sample_rate() >= cpal::SampleRate(sr)
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
        sample_rate: cpal::SampleRate(sr),
        buffer_size: cpal::BufferSize::Fixed(buffer_size),
    };

    let mut ring_buffer = RingBuffer {
        buf: Vec::new(),
        gain: 0.0,
        latency_ms: None,
        chunk_size_ms: 0.0,
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
            rb.chunk_size_ms = data.len() as f32 / sr as f32 * 1000.0;
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

pub fn dump_input_devices() {
    let host = cpal::default_host();
    let devices = host.devices().unwrap();
    for device in devices {
        println!("Device {:?}:", device.name().unwrap());
        dump_supported_input_configs(&device);
    }
}

pub fn dump_supported_input_configs(device: &cpal::Device) {
    let supported_input_configs = device.supported_input_configs().unwrap();
    for supported_input_config in supported_input_configs {
        println!("{:?}", supported_input_config);
    }
}
