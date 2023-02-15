use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

pub struct AudioStream {
    pub sr: usize,
    pub ring_buffer: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
    pub stream: cpal::Stream,
    //pub agc: dagc::MonoAgc,
}

impl AudioStream {
    pub fn new(sr: usize, buf_size: usize) -> Result<Self> {
        dbg!(cpal::available_hosts());

        let host = cpal::default_host();

        // let device = host
        //     .devices()?
        //     .find(|d| d.name().unwrap() == "plughw:CARD=USB,DEV=0")
        //     .unwrap();

        let device = host.default_input_device().unwrap();

        println!("{}", device.name()?);
        let stream_config = cpal::StreamConfig {
            channels: 1u16,
            sample_rate: cpal::SampleRate(sr as u32),
            buffer_size: cpal::BufferSize::Default,
        };

        let mut ring_buffer = Vec::new();
        ring_buffer.resize(buf_size, 0f32);
        let ring_buffer = std::sync::Mutex::from(ring_buffer);
        let ring_buffer = std::sync::Arc::new(ring_buffer);

        let ring_buffer_input_thread_clone = ring_buffer.clone();

        let mut agc = dagc::MonoAgc::new(0.07, 0.0001).unwrap();

        let stream = device.build_input_stream(
            &stream_config,
            move |data: &[f32], info: &cpal::InputCallbackInfo| {
                let mut rb = ring_buffer_input_thread_clone.lock().unwrap();
                rb.drain(..data.len());
                rb.extend_from_slice(&data);
                let begin = rb.len() - data.len();
                agc.process(&mut rb[begin..]);
                println!("gain: {}", agc.gain());
            },
            move |err| panic!("{}", err),
        )?;

        Ok(Self {
            sr,
            ring_buffer,
            stream,
        })
    }

    pub fn play(&self) -> Result<()> {
        self.stream.play()?;

        Ok(())
    }
}
