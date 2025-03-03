use android_activity::{AndroidApp};
//use jni::sys::{jobject, jstring};
use core::panic;
use jni::objects::{GlobalRef, JByteArray, JObject, JString, JValue};
use jni::{
    objects::{JClass, JMethodID, JValueGen},
    JNIEnv, JavaVM,
};
use jni::{
    sys::{jobject, jstring},
};
use log::{info, trace};
use std::ffi::c_void;
use oboe::{
    AudioInputCallback, AudioInputStreamSafe, AudioStream, AudioStreamBuilder, DataCallbackResult,
    Input, InputPreset, Mono, PerformanceMode, SampleRateConversionQuality, SharingMode,
};
//use jni::sys::{JNI_CreateJavaVM, JNI_GetCreatedJavaVMs, JavaVMInitArgs, JavaVMOption, JNI_VERSION_1_6};

// #![no_std]
// #![no_main]

use std::time::Duration;

use pitchvis_analysis::{
    analysis::{AnalysisParameters, AnalysisState},
    util::*,
    vqt::{VqtParameters, VqtRange},
};
use pitchvis_colors::calculate_color;
// use serialport::SerialPort;

// increasing BUCKETS_PER_SEMITONE or Q will improve frequency resolution at cost of time resolution,
// increasing GAMMA will improve time resolution at lower frequencies.
pub const SR: u32 = 22050;
pub const BUFSIZE: usize = 2 * SR as usize;
pub const N_FFT: usize = 2 * 16384;
pub const FREQ_A1: f32 = 55.0;
pub const BUCKETS_PER_SEMITONE: u16 = 3;
pub const BUCKETS_PER_OCTAVE: u16 = 12 * BUCKETS_PER_SEMITONE;
pub const OCTAVES: u8 = 5;
pub const SPARSITY_QUANTILE: f32 = 0.999;
pub const Q: f32 = 1.8;
pub const GAMMA: f32 = 4.8 * Q;

const VQT_PARAMETERS: VqtParameters = VqtParameters {
    sr: SR as f32,
    n_fft: N_FFT,
    range: VqtRange {
        min_freq: FREQ_A1,
        octaves: OCTAVES,
        buckets_per_octave: BUCKETS_PER_OCTAVE,
    },
    sparsity_quantile: SPARSITY_QUANTILE,
    quality: Q,
    gamma: GAMMA,
};

const FPS: u64 = 30;

// color calculation constants
pub const COLORS: [[f32; 3]; 12] = [
    [0.95, 0.10, 0.10], // C
    [0.01, 0.52, 0.71], // C#
    [0.97, 0.79, 0.00], // D
    [0.45, 0.34, 0.63], // Eb
    [0.47, 0.99, 0.02], // E
    [0.88, 0.02, 0.52], // F
    [0.00, 0.80, 0.55], // F#
    [0.99, 0.54, 0.03], // G
    [0.25, 0.30, 0.64], // Ab
    [0.95, 0.99, 0.00], // A
    [0.52, 0.00, 0.60], // Bb
    [0.05, 0.80, 0.15], // H
];
const GRAY_LEVEL: f32 = 5.0;
const EASING_POW: f32 = 2.3;

struct VqtResult {
    pub x_vqt: Vec<f32>,
    pub gain: f32,
}

impl VqtResult {
    pub fn new(range: &VqtRange) -> Self {
        Self {
            x_vqt: vec![0.0; range.n_buckets()],
            gain: 1.0,
        }
    }
}

fn microphone_permission_granted(perm: &str) -> bool {
    let ctx = ndk_context::android_context();
    let jvm = unsafe { JavaVM::from_raw(ctx.vm().cast()) }.unwrap();
    let mut cth = jvm.attach_current_thread().unwrap();

    let j_permission = cth.new_string(perm).unwrap();
    let permission_granted = cth
        .call_method(
            &unsafe { JObject::from_raw(ctx.context() as jni::sys::jobject) },
            "checkSelfPermission",
            "(Ljava/lang/String;)I",
            &[JValue::try_from(&j_permission).unwrap()],
        )
        .unwrap();

    permission_granted.i().unwrap() != -1
}

pub fn request_microphone_permission(android_app: &AndroidApp, permission: &str) {
    let jvm = unsafe { JavaVM::from_raw(android_app.vm_as_ptr() as _) }.unwrap();
    let mut cth = jvm.attach_current_thread().unwrap();

    let j_permission = cth.new_string(permission).unwrap();

    let permissions: Vec<jstring> = vec![j_permission.into_raw()];
    let permissions_array = cth
        .new_object_array(
            permissions.len() as i32,
            "java/lang/String",
            JObject::null(),
        )
        .unwrap();
    for (i, permission) in permissions.into_iter().enumerate() {
        cth.set_object_array_element(&permissions_array, i as i32, unsafe {
            JObject::from_raw(permission)
        })
        .unwrap();
    }

    let activity = unsafe { JObject::from_raw(android_app.activity_as_ptr() as jobject) };
    let _res = cth
        .call_method(
            activity,
            "requestPermissions",
            "([Ljava/lang/String;I)V",
            &[JValue::Object(&permissions_array), JValue::Int(3)],
        )
        .unwrap();
}

/// This enum is used to control the audio stream. It is sent to the audio thread via a channel. This allows us to control the audio stream from the bevy thread.
// enum AudioControl {
//     Play,
//     Pause,
// }

// #[derive(Resource)]
// struct AudioControlChannelResource(std::sync::mpsc::Sender<AudioControl>);

struct AudioCallback {
    ring_buffer: std::sync::Arc<std::sync::Mutex<pitchvis_audio::RingBuffer>>,
    agc: dagc::MonoAgc,
}

impl AudioInputCallback for AudioCallback {
    type FrameType = (f32, Mono);

    fn on_audio_ready(
        &mut self,
        stream: &mut dyn AudioInputStreamSafe,
        data: &[f32],
    ) -> DataCallbackResult {
        // check for invalid samples
        if let Some(x) = data.iter().find(|x| !x.is_finite()) {
            log::warn!("bad audio sample encountered: {x}");
            return DataCallbackResult::Continue;
        }

        // agc processing
        let sample_sq_sum = data.iter().map(|x| x.powi(2)).sum::<f32>();
        self.agc.freeze_gain(sample_sq_sum < 1e-6);

        // update ring buffer
        let mut rb = self.ring_buffer.lock().expect("locking failed");
        rb.buf.drain(..data.len());
        rb.buf.extend_from_slice(data);
        let begin = rb.buf.len() - data.len();
        self.agc.process(&mut rb.buf[begin..]);
        rb.gain = self.agc.gain();

        // calculate latency
        // use nix::time::{clock_gettime, ClockId};
        // if let Ok(ts) = stream.get_timestamp(ClockId::CLOCK_MONOTONIC.into()) {
        //     let app_frame_index = stream.get_frames_read();

        //     // ts is a snapshot. We use the sample rate to calculate the corresponding
        //     // time of the latest sample we are currently handling.
        //     let frame_delta = app_frame_index - ts.position;
        //     let frame_time_delta = frame_delta as f64 / SR as f64 * 1e9;
        //     let app_frame_in_hardware_time = ts.timestamp + frame_time_delta as i64;

        //     // We assume that the frame we are currently handling is being processed
        //     // NOW, and calculate the latency by comparing this to the time it originated
        //     // from the hardware.
        //     let app_frame_time =
        //         std::time::Duration::from(clock_gettime(ClockId::CLOCK_MONOTONIC).unwrap())
        //             .as_nanos();
        //     let latency_nanos = app_frame_time - app_frame_in_hardware_time as u128;

        //     rb.latency_ms = Some((latency_nanos as f64 / 1e6) as f32);
        // } else {
        //     rb.latency_ms = None;
        // }

        rb.latency_ms = stream.calculate_latency_millis().ok().map(|x| x as f32);
        rb.chunk_size_ms = data.len() as f32 / stream.get_sample_rate() as f32 * 1000.0;

        DataCallbackResult::Continue
    }
}

// #[no_mangle]
// pub extern "C" fn JNI_OnLoad(vm: jni::JavaVM, res: *mut std::os::raw::c_void) -> jni::sys::jint {
//     let env = vm.get_env().unwrap();
//     let vm = vm.get_java_vm_pointer() as *mut c_void;
//     unsafe {
//         ndk_context::initialize_android_context(vm, res);
//     }
//     jni::JNIVersion::V6.into()
// }

fn resolve_method_id(jenv: &mut JNIEnv, klass: JClass, name: &str, signature: &str) -> JMethodID {
    trace!("Resolving method {name}...");
    jenv.get_method_id(klass, name, signature)
        .expect("get method id failed")
}

pub struct SerialPortManager {
    jvm_ptr: *mut c_void,
    manager: GlobalRef,
}

impl SerialPortManager {
    pub fn new(
        jvm_ptr: *mut c_void,
        activity_ptr: *mut c_void,
    ) -> Result<Self, jni::errors::Error> {
        // Get the JavaVM from the pointer
        let vm = unsafe { JavaVM::from_raw(jvm_ptr as *mut _) }.expect("JavaVM::from_raw failed");

        // Get the Activity from the pointer
        let activity = unsafe { JObject::from_raw(activity_ptr as *mut _) };

        // Attach the current thread to the JVM
        let mut env = vm
            .attach_current_thread()
            .expect("attach_current_thread failed");

        let activity_class = env
            .get_object_class(activity)
            .expect("get_object_class failed");

        let get_class_loader_method = resolve_method_id(
            &mut env,
            activity_class,
            "getClassLoader",
            "()Ljava/lang/ClassLoader;",
        );

        trace!("Calling activity.getClassLoader()");
        let activity = unsafe { JObject::from_raw(activity_ptr as *mut _) };
        let loader: JObject = match unsafe {
            env.call_method_unchecked(
                activity,
                get_class_loader_method,
                jni::signature::ReturnType::Object,
                &[],
            )
        }
        .expect("bla")
        {
            JValueGen::Object(loader) => loader,
            _ => panic!("Unexpected return type from activity.getClassLoader()"),
        };
        let loader_class: JObject = env
            .find_class("java/lang/ClassLoader")
            .expect("find_class")
            .into();
        let load_class_method = resolve_method_id(
            &mut env,
            loader_class.into(),
            "loadClass",
            "(Ljava/lang/String;)Ljava/lang/Class;",
        );
        let ble_session_class_name: JObject = env
            .new_string("org/p1graph/pitchvis_serial/MainActivity")
            .expect("new_string")
            .into();
        // let session_class: JClass = try_call_object_method(env, loader, load_class_method, &[
        //     JValue::Object(&ble_session_class_name).as_jni()
        // ]).expect("try_call_object_method failed2").into();
        let session_class: JClass = match unsafe {
            env.call_method_unchecked(
                loader,
                load_class_method,
                jni::signature::ReturnType::Object,
                &[JValue::Object(&ble_session_class_name).as_jni()],
            )
        }
        .expect("try_call_object_method failed2")
        {
            JValueGen::Object(loader) => loader.into(),
            _ => panic!("Unexpected return type from activity.getClassLoader()"),
        };

        let class = session_class;

        // Get the class
        //let class = env.find_class("java/lang/String").expect("find_class failed");
        //let classsss = env.find_class("org/p1graph/pitchvis_serial/MainActivity").expect("find_class failed");
        //let class: jni::objects::JClass = activity.into();

        // Call getSerialPortManager to get the SerialPortManager instance
        let manager = env
            .call_static_method(
                class,
                "getSerialPortManager",
                "()Lorg/p1graph/pitchvis_serial/SerialPortManager;",
                &[],
            )
            .expect("call_static_method failed");

        // Get a global reference to the manager
        let manager = env
            .new_global_ref(manager.l().expect("manager.l"))
            .expect("new_global_ref failed");

        Ok(SerialPortManager { jvm_ptr, manager })
    }

    pub fn open_serial_port(&self) -> Result<(), jni::errors::Error> {
        // Get the JavaVM from the pointer
        let vm = unsafe { JavaVM::from_raw(self.jvm_ptr as *mut _) }?;

        let mut env = vm.attach_current_thread()?;
        let output = env.call_method(
            self.manager.as_obj(),
            "openSerialPort",
            "()Ljava/lang/String;",
            &[],
        )?;
        let output_jstring = JString::from(output.l()?);
        let output_str: String = env.get_string(&output_jstring)?.into();
        info!("output_str: {output_str}");
        Ok(())
    }

    pub fn write_data(&self, data: &[u8]) -> Result<(), jni::errors::Error> {
        // Get the JavaVM from the pointer
        let vm = unsafe { JavaVM::from_raw(self.jvm_ptr as *mut _) }?;

        let mut env = vm.attach_current_thread()?;
        let data_jbytearray: JByteArray = env.byte_array_from_slice(data)?;
        env.call_method(
            self.manager.as_obj(),
            "writeData",
            "([B)V",
            &[JValue::from(&JObject::from(data_jbytearray))],
        )?;
        Ok(())
    }
}

fn update_serial(
    range: &VqtRange,
    _vqt_result: &VqtResult,
    analysis_state: &AnalysisState,
    serial_port: &mut SerialPortManager,
) {
    // let vqt = &vqt_result.x_vqt;

    let mut x = vec![0.0_f32; range.n_buckets()];
    for p in analysis_state.peaks_continuous.iter() {
        let lower = p.center.floor() as usize;
        x[lower] = p.size * (1.0 - p.center.fract().powf(1.9));
        if lower < range.n_buckets() - 1 {
            x[lower + 1] = p.size * p.center.fract().powf(1.9);
        }
        // if *p > 0 {
        //     x[*p-1] = vqt[*p-1];
        // }
    }

    let k_max = arg_max(&x);
    let max_size = x[k_max];

    // special value to indicate begin of data
    let mut output: Vec<u8> = vec![0xFF];
    // 16 bit number of RGB triples to follow
    let num_triples: u16 = analysis_state.x_vqt_peakfiltered.len().try_into().unwrap();
    output.push((num_triples / 256) as u8);
    output.push((num_triples % 256) as u8);
    output.extend(x.iter().enumerate().flat_map(|(idx, size)| {
        let (mut r, mut g, mut b) = calculate_color(
            range.buckets_per_octave,
            ((idx + (range.buckets_per_octave - 3 * (range.buckets_per_octave / 12)) as usize)
                as f32)
                % range.buckets_per_octave as f32,
            COLORS,
            GRAY_LEVEL,
            EASING_POW,
        );

        let color_coefficient = 1.0 - (1.0 - size / max_size); //.powf(0.18);
        r *= color_coefficient;
        g *= color_coefficient;
        b *= color_coefficient;

        [(r * 254.0) as u8, (g * 254.0) as u8, (b * 254.0) as u8]
    }));
    println!("output: {:02x?}", &output);

    serial_port
        .write_data(output.as_slice())
        .expect("Write failed!");
}
pub struct RingBuffer {
    pub buf: Vec<f32>,
    pub gain: f32,
}

#[no_mangle]
fn android_main(app: AndroidApp) {
    android_logger::init_once(android_logger::Config::default().with_min_level(log::Level::Info));

    // println!("Serial output format: 0xFF <num_triples (16 bit)> <r1> <g1> <b1> <r2> <g2> <b2> ...");
    // take command line arguments, e. g. `pitchvis_serial /dev/ttyUSB0 9600`
    //let args = std::env::args().collect::<Vec<_>>();
    // let path = String::from("/dev/ttyUSB0");
    // let baud_rate = 115_200;
    // let mut serial_port = serialport::new(path, baud_rate)
    //     .timeout(std::time::Duration::from_secs(10)) // TODO: ???
    //     .open()
    //     .expect("Failed to open port");
    //let audio_stream = pitchvis_audio::audio::AudioStream::new(SR, BUFSIZE).unwrap();
    let mut serial_port = SerialPortManager::new(app.vm_as_ptr(), app.activity_as_ptr()).unwrap();
    serial_port.open_serial_port().unwrap();

    if !microphone_permission_granted("android.permission.RECORD_AUDIO") {
        log::info!("requesting microphone permission");
        request_microphone_permission(
            &app,
            "android.permission.RECORD_AUDIO",
        );
    }
    // wait until permission is granted
    while !microphone_permission_granted("android.permission.RECORD_AUDIO") {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    let vqt = pitchvis_analysis::vqt::Vqt::new(&VQT_PARAMETERS);
    let mut vqt_result = VqtResult::new(&VQT_PARAMETERS.range);
    let mut analysis_state =
        AnalysisState::new(VQT_PARAMETERS.range, AnalysisParameters::default());

    // let (audio_control_channel_tx, audio_control_channel_rx) = mpsc::channel::<AudioControl>();
    let mut ring_buffer = pitchvis_audio::RingBuffer {
        buf: Vec::new(),
        gain: 0.0,
        latency_ms: None,
        chunk_size_ms: 0.0,
    };
    ring_buffer.buf.resize(BUFSIZE, 0f32);
    let ring_buffer = std::sync::Mutex::from(ring_buffer);
    let ring_buffer = std::sync::Arc::new(ring_buffer);

    let agc = dagc::MonoAgc::new(0.07, 0.0001).expect("mono-agc creation failed");

    let callback = AudioCallback {
        ring_buffer: ring_buffer.clone(),
        agc,
    };

    let mut audio_stream = AudioStreamBuilder::default()
        .set_direction::<Input>()
        .set_performance_mode(PerformanceMode::LowLatency)
        .set_sample_rate(VQT_PARAMETERS.sr as i32)
        // TODO: support all microphone channels if `Mono` does not mix them down
        .set_channel_count::<Mono>()
        .set_format::<f32>()
        .set_sharing_mode(SharingMode::Exclusive)
        .set_input_preset(InputPreset::Unprocessed)
        // this will solve bugs on some devices
        // TODO: is `Best` necessary?
        .set_sample_rate_conversion_quality(SampleRateConversionQuality::Best)
        .set_callback(callback)
        .open_stream()
        .unwrap();

    audio_stream.request_start().unwrap();

    let mut start_time = std::time::Instant::now();
    loop {
        let (x, gain) = {
            let mut x = vec![0.0_f32; VQT_PARAMETERS.n_fft];
            let rb = ring_buffer.lock().unwrap();
            x.copy_from_slice(&rb.buf[(BUFSIZE - VQT_PARAMETERS.n_fft)..]);
            (x, rb.gain)
        };
        vqt_result.x_vqt = vqt.calculate_vqt_instant_in_db(&x);
        vqt_result.gain = gain;

        let elapsed = start_time.elapsed();
        start_time = std::time::Instant::now();

        analysis_state.preprocess(&vqt_result.x_vqt, elapsed);
        update_serial(
            &VQT_PARAMETERS.range,
            &vqt_result,
            &analysis_state,
            &mut serial_port,
        );

        let sleep_time = Duration::from_millis(1000 / FPS).saturating_sub(elapsed);
        std::thread::sleep(sleep_time);
    }
}
