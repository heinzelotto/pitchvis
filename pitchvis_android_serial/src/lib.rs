use android_activity::{AndroidApp, InputStatus, MainEvent, PollEvent};
//use jni::sys::{jobject, jstring};
use core::panic;
use jni::objects::{GlobalRef, JByteArray, JObject, JString, JValue};
use jni::{
    objects::{JClass, JMethodID, JValueGen},
    JNIEnv, JavaVM,
};
use log::{error, info, trace};
use std::ffi::c_void;
//use jni::sys::{JNI_CreateJavaVM, JNI_GetCreatedJavaVMs, JavaVMInitArgs, JavaVMOption, JNI_VERSION_1_6};

// #![no_std]
// #![no_main]

use std::time::Duration;

use pitchvis_analysis::{analysis::AnalysisState, util::*};
// use serialport::SerialPort;

// increasing BUCKETS_PER_SEMITONE or Q will improve frequency resolution at cost of time resolution,
// increasing GAMMA will improve time resolution at lower frequencies.
pub const SR: usize = 22050;
pub const BUFSIZE: usize = 2 * SR;
pub const N_FFT: usize = 2 * 16384;
pub const FREQ_A1: f32 = 55.0;
pub const BUCKETS_PER_SEMITONE: usize = 3;
pub const BUCKETS_PER_OCTAVE: usize = 12 * BUCKETS_PER_SEMITONE;
pub const OCTAVES: usize = 5;
pub const SPARSITY_QUANTILE: f32 = 0.999;
pub const Q: f32 = 10.0;
pub const GAMMA: f32 = 5.3 * Q;

const FPS: u64 = 25;

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

struct CqtResult {
    pub x_cqt: Vec<f32>,
    pub gain: f32,
}

impl CqtResult {
    pub fn new(octaves: usize, buckets_per_octave: usize) -> Self {
        Self {
            x_cqt: vec![0.0; octaves * buckets_per_octave],
            gain: 1.0,
        }
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
            .new_string("co/realfit/agdkcpal/MainActivity")
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
        //let classsss = env.find_class("co/realfit/agdkcpal/MainActivity").expect("find_class failed");
        //let class: jni::objects::JClass = activity.into();

        // Call getSerialPortManager to get the SerialPortManager instance
        let manager = env
            .call_static_method(
                class,
                "getSerialPortManager",
                "()Lco/realfit/agdkcpal/SerialPortManager;",
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
    buckets_per_octave: usize,
    analysis_state: &AnalysisState,
    serial_port: &mut SerialPortManager,
) {
    let k_max = arg_max(&analysis_state.x_cqt_peakfiltered);
    let max_size = analysis_state.x_cqt_peakfiltered[k_max];

    // special value to indicate begin of data
    let mut output: Vec<u8> = vec![0xFF];
    // 16 bit number of RGB triples to follow
    let num_triples: u16 = analysis_state.x_cqt_peakfiltered.len().try_into().unwrap();
    output.push((num_triples / 256) as u8);
    output.push((num_triples % 256) as u8);
    output.extend(
        analysis_state
            .x_cqt_peakfiltered
            .iter()
            .enumerate()
            .flat_map(|(idx, size)| {
                let (mut r, mut g, mut b) = pitchvis_analysis::color_mapping::calculate_color(
                    buckets_per_octave,
                    ((idx + (buckets_per_octave - 3 * (buckets_per_octave / 12))) as f32)
                        % buckets_per_octave as f32,
                    COLORS,
                    GRAY_LEVEL,
                    EASING_POW,
                );

                let color_coefficient = 1.0 - (1.0 - size / max_size).powf(0.18);
                r *= color_coefficient;
                g *= color_coefficient;
                b *= color_coefficient;

                [(r * 254.0) as u8, (g * 254.0) as u8, (b * 254.0) as u8]
            }),
    );
    println!("output: {:02x?}", &output);

    // serial_port
    //     .write_all(output.as_slice())
    //     .expect("Write failed!");
    // serial_port.flush().expect("Flush failed!");
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

    let mut cqt = pitchvis_analysis::cqt::Cqt::new(
        SR,
        N_FFT,
        FREQ_A1,
        BUCKETS_PER_OCTAVE,
        OCTAVES,
        SPARSITY_QUANTILE,
        Q,
        GAMMA,
    );
    let mut cqt_result = CqtResult::new(OCTAVES, BUCKETS_PER_OCTAVE);
    let mut analysis_state = AnalysisState::new(
        OCTAVES * BUCKETS_PER_OCTAVE,
        pitchvis_analysis::analysis::SPECTROGRAM_LENGTH,
    );

    //audio_stream.play().unwrap();

    let mut ring_buffer = RingBuffer {
        buf: Vec::new(),
        gain: 0.0,
    };
    ring_buffer.buf.resize(BUFSIZE, 0f32);
    let ring_buffer = std::sync::Mutex::from(ring_buffer);
    let ring_buffer = std::sync::Arc::new(ring_buffer);
    let mut agc = dagc::MonoAgc::new(0.07, 0.0001).expect("mono-agc creation failed");
    let ring_buffer_input_thread_clone = ring_buffer.clone();
    let mut audio_stream = aaudio::AAudioStreamBuilder::new()
        .unwrap()
        .set_direction(aaudio::Direction::Input)
        .set_performance_mode(aaudio::PerformanceMode::LowLatency)
        .set_sample_rate(SR as i32)
        .set_format(aaudio::Format::F32)
        .set_channel_count(1)
        .set_sharing_mode(aaudio::SharingMode::Exclusive)
        .set_callbacks(
            move |_, data: &mut [u8], frames: i32| {
                let data = unsafe {
                    std::slice::from_raw_parts_mut(data.as_ptr() as *mut f32, frames as usize * 1)
                };
                if let Some(x) = data.iter().find(|x| !x.is_finite()) {
                    log::warn!("bad audio sample encountered: {x}");
                    return aaudio::CallbackResult::Continue;
                }
                let sample_sq_sum = data.iter().map(|x| x.powi(2)).sum::<f32>();
                agc.freeze_gain(sample_sq_sum < 1e-6);

                //log::info!("audio callback");

                let mut rb = ring_buffer_input_thread_clone
                    .lock()
                    .expect("locking failed");
                rb.buf.drain(..data.len());
                rb.buf.extend_from_slice(data);
                let begin = rb.buf.len() - data.len();
                agc.process(&mut rb.buf[begin..]);
                rb.gain = agc.gain();

                aaudio::CallbackResult::Continue
            },
            |_, _| {},
        )
        .open_stream()
        .unwrap();

    audio_stream.request_start().unwrap();

    let mut quit = false;
    let mut redraw_pending = true;
    let mut render_state: Option<()> = Default::default();

    while !quit {
        let start_time = std::time::Instant::now();

        let (x, gain) = {
            let mut x = vec![0.0_f32; cqt.n_fft];
            let rb = ring_buffer.lock().unwrap();
            x.copy_from_slice(&rb.buf[(BUFSIZE - cqt.n_fft)..]);
            (x, rb.gain)
        };
        cqt_result.x_cqt = cqt.calculate_cqt_instant_in_db(&x);
        cqt_result.gain = gain;
        analysis_state.preprocess(&cqt_result.x_cqt, OCTAVES, BUCKETS_PER_OCTAVE);
        update_serial(BUCKETS_PER_OCTAVE, &analysis_state, &mut serial_port);

        let elapsed = start_time.elapsed();
        let sleep_time = Duration::from_millis(1000 / FPS).saturating_sub(elapsed);
        std::thread::sleep(sleep_time);

        app.poll_events(
            Some(std::time::Duration::from_millis(5)), /* timeout */
            |event| {
                match event {
                    PollEvent::Wake => {
                        info!("Early wake up");
                    }
                    PollEvent::Timeout => {
                        info!("Timed out");
                        // Real app would probably rely on vblank sync via graphics API...
                        redraw_pending = true;
                    }
                    PollEvent::Main(main_event) => {
                        info!("Main event: {:?}", main_event);
                        match main_event {
                            MainEvent::SaveState { saver, .. } => {
                                saver.store("foo://bar".as_bytes());
                            }
                            MainEvent::Pause => {
                                // if let Err(err) = stream.pause() {
                                //     log::error!("Failed to pause audio playback: {err}");
                                // }
                            }
                            MainEvent::Resume { loader, .. } => {
                                if let Some(state) = loader.load() {
                                    if let Ok(uri) = String::from_utf8(state) {
                                        info!("Resumed with saved state = {uri:#?}");
                                    }
                                }

                                // if let Err(err) = stream.play() {
                                //     log::error!("Failed to start audio playback: {err}");
                                // }
                            }
                            MainEvent::InitWindow { .. } => {
                                render_state = Some(());
                                redraw_pending = true;
                            }
                            MainEvent::TerminateWindow { .. } => {
                                render_state = None;
                            }
                            MainEvent::WindowResized { .. } => {
                                redraw_pending = true;
                            }
                            MainEvent::RedrawNeeded { .. } => {
                                redraw_pending = true;
                            }
                            MainEvent::LowMemory => {}

                            MainEvent::Destroy => quit = true,
                            _ => { /* ... */ }
                        }
                    }
                    _ => {}
                }

                if redraw_pending {
                    if let Some(_rs) = render_state {
                        redraw_pending = false;

                        // Handle input
                        app.input_events(|event| {
                            info!("Input Event: {event:?}");
                            InputStatus::Unhandled
                        });

                        info!("Render...");
                    }
                }
            },
        );
    }
}
