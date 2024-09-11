#[cfg(target_arch = "wasm32")]
use crate::audio_system::AudioBufferResource;
#[cfg(not(target_os = "android"))]
use anyhow::Result;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::input::keyboard::KeyboardInput;
use bevy::input::mouse::MouseButtonInput;
use bevy::input::touch::TouchPhase;
use bevy::prelude::*;
use bevy::sprite::Material2dPlugin;
#[cfg(target_os = "android")]
use bevy::window::AppLifecycle;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
mod analysis_system;
mod audio_system;
mod display_system;
#[cfg(feature = "ml")]
mod ml_system;
mod util;
mod vqt_system;

#[cfg(target_os = "android")]
use android_activity::AndroidApp;
#[cfg(target_os = "android")]
use jni::{
    objects::{JObject, JValue},
    sys::{jobject, jstring},
    JavaVM,
};

#[cfg(target_os = "android")]
#[cfg_attr(not(feature = "ml"), link(name = "c++_shared"))]
extern "C" {}

// increasing BUCKETS_PER_SEMITONE or Q will improve frequency resolution at cost of time resolution,
// increasing GAMMA will improve time resolution at lower frequencies.
pub const SR: usize = 22050;
pub const BUFSIZE: usize = 2 * SR;
pub const N_FFT: usize = 2 * 16384;
pub const FREQ_A1: f32 = 55.0;
pub const FREQ_A1_MIDI_KEY_ID: i32 = 33;
pub const UPSCALE_FACTOR: usize = 1;
pub const BUCKETS_PER_SEMITONE: usize = 3 * UPSCALE_FACTOR;
pub const BUCKETS_PER_OCTAVE: usize = 12 * BUCKETS_PER_SEMITONE;
pub const OCTAVES: usize = 7;
pub const SPARSITY_QUANTILE: f32 = 0.999;
pub const Q: f32 = 10.0 / UPSCALE_FACTOR as f32;
pub const GAMMA: f32 = 5.3 * Q;

#[cfg(target_os = "android")]
const FPS: u64 = 33;
#[cfg(not(target_os = "android"))]
const FPS: u64 = 30;
// TODO: make the animation speed/blurring windows/... independent of the frame rate

use bevy::diagnostic::DiagnosticsStore;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn main_fun() -> Result<(), JsValue> {
    let audio_stream = pitchvis_audio::audio::AudioStream::async_new(SR, BUFSIZE)
        .await
        .unwrap();

    let vqt = pitchvis_analysis::vqt::Vqt::new(
        SR,
        N_FFT,
        FREQ_A1,
        BUCKETS_PER_OCTAVE,
        OCTAVES,
        SPARSITY_QUANTILE,
        Q,
        GAMMA,
    );

    audio_stream.play().unwrap();

    let update_vqt_system = vqt_system::update_vqt_to_system(BUFSIZE);
    let update_analysis_state_system =
        analysis_system::update_analysis_state_to_system(OCTAVES, BUCKETS_PER_OCTAVE);
    let update_display_system =
        display_system::update_display_to_system(BUCKETS_PER_OCTAVE, OCTAVES);

    App::new()
        .add_plugins((
            DefaultPlugins,
            LogDiagnosticsPlugin::default(),
            FrameTimeDiagnosticsPlugin::default(),
            // .add_plugin(MaterialPlugin::<display_system::LineMaterial>::default())
            Material2dPlugin::<display_system::material::NoisyColorMaterial>::default(),
        ))
        .insert_resource(vqt_system::VqtResource(vqt))
        .insert_resource(vqt_system::VqtResultResource::new(
            OCTAVES,
            BUCKETS_PER_OCTAVE,
        ))
        .insert_resource(audio_system::AudioBufferResource(
            audio_stream.ring_buffer.clone(),
        ))
        .insert_resource(analysis_system::AnalysisStateResource(
            pitchvis_analysis::analysis::AnalysisState::new(
                OCTAVES * BUCKETS_PER_OCTAVE,
                pitchvis_analysis::analysis::SPECTROGRAM_LENGTH,
            ),
        ))
        .insert_resource(display_system::CylinderEntityListResource(Vec::new()))
        .insert_resource(display_system::SettingsState {
            display_mode: display_system::DisplayMode::PitchnamesCalmness,
        })
        .add_systems(
            Startup,
            (
                display_system::setup_display_to_system(OCTAVES, BUCKETS_PER_OCTAVE),
                setup_fps_counter,
            ),
        )
        .add_systems(
            Update,
            (
                close_on_esc,
                // frame_limiter_system, // FIXME: this is not working on wasm
                update_vqt_system,
                update_analysis_state_system.after(update_vqt_system),
                update_display_system.after(update_analysis_state_system),
                user_input_system,
                fps_text_update_system,
                fps_counter_showhide,
            ),
        )
        .run();

    Ok(())
}

/// Marker to find the container entity so we can show/hide the FPS counter
#[derive(Component)]
struct FpsRoot;

/// Marker to find the text entity so we can update it
#[derive(Component)]
struct FpsText;

fn setup_fps_counter(mut commands: Commands) {
    // create our UI root node
    // this is the wrapper/container for the text
    let root = commands
        .spawn((
            FpsRoot,
            NodeBundle {
                // give it a dark background for readability
                background_color: BackgroundColor(Color::BLACK.with_alpha(0.5)),
                // make it "always on top" by setting the Z index to maximum
                // we want it to be displayed over all other UI
                z_index: ZIndex::Global(i32::MAX),
                style: Style {
                    position_type: PositionType::Absolute,
                    // position it at the top-right corner
                    // 1% away from the top window edge
                    left: Val::Percent(1.),
                    // right: Val::Percent(1.),
                    top: Val::Percent(1.),
                    // set bottom/left to Auto, so it can be
                    // automatically sized depending on the text
                    bottom: Val::Auto,
                    right: Val::Auto,
                    // give it some padding for readability
                    padding: UiRect::all(Val::Px(4.0)),
                    ..Default::default()
                },
                visibility: Visibility::Visible,
                ..Default::default()
            },
        ))
        .id();
    // create our text
    let text_fps = commands
        .spawn((
            FpsText,
            TextBundle {
                // use two sections, so it is easy to update just the number
                text: Text::from_sections([
                    TextSection {
                        value: "FPS: ".into(),
                        style: TextStyle {
                            font_size: 16.0,
                            color: Color::WHITE,
                            // if you want to use your game's font asset,
                            // uncomment this and provide the handle:
                            // font: my_font_handle
                            ..default()
                        },
                    },
                    TextSection {
                        value: " N/A".into(),
                        style: TextStyle {
                            font_size: 16.0,
                            color: Color::WHITE,
                            // if you want to use your game's font asset,
                            // uncomment this and provide the handle:
                            // font: my_font_handle
                            ..default()
                        },
                    },
                ]),
                ..Default::default()
            },
        ))
        .id();
    commands.entity(root).push_children(&[text_fps]);
}

fn fps_text_update_system(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut Text, With<FpsText>>,
) {
    for mut text in &mut query {
        // try to get a "smoothed" FPS value from Bevy
        if let Some(value) = diagnostics
            .get(&FrameTimeDiagnosticsPlugin::FPS)
            .and_then(|fps| fps.smoothed())
        {
            // Format the number as to leave space for 4 digits, just in case,
            // right-aligned and rounded. This helps readability when the
            // number changes rapidly.
            text.sections[1].value = format!("{value:>4.0}");

            // Let's make it extra fancy by changing the color of the
            // text according to the FPS value:
            text.sections[1].style.color = if value >= 120.0 {
                // Above 120 FPS, use green color
                Color::srgb(0.0, 1.0, 0.0)
            } else if value >= 60.0 {
                // Between 60-120 FPS, gradually transition from yellow to green
                Color::srgb((1.0 - (value - 60.0) / (120.0 - 60.0)) as f32, 1.0, 0.0)
            } else if value >= 30.0 {
                // Between 30-60 FPS, gradually transition from red to yellow
                Color::srgb(1.0, ((value - 30.0) / (60.0 - 30.0)) as f32, 0.0)
            } else {
                // Below 30 FPS, use red color
                Color::srgb(1.0, 0.0, 0.0)
            }
        } else {
            // display "N/A" if we can't get a FPS measurement
            // add an extra space to preserve alignment
            text.sections[1].value = " N/A".into();
            text.sections[1].style.color = Color::WHITE;
        }
    }
}

/// Toggle the FPS counter when pressing F12
fn fps_counter_showhide(
    mut q: Query<&mut Visibility, With<FpsRoot>>,
    settings: Res<display_system::SettingsState>,
) {
    let mut vis = q.single_mut();
    if settings.display_mode == display_system::DisplayMode::Debugging {
        *vis = Visibility::Visible;
    } else {
        *vis = Visibility::Hidden;
    }
}

fn frame_limiter_system() {
    use std::{thread, time};
    thread::sleep(time::Duration::from_millis((1000 / FPS).saturating_sub(5)));
}

fn cycle_display_mode(mode: &display_system::DisplayMode) -> display_system::DisplayMode {
    match mode {
        display_system::DisplayMode::PitchnamesCalmness => display_system::DisplayMode::Calmness,
        display_system::DisplayMode::Calmness => display_system::DisplayMode::Debugging,
        display_system::DisplayMode::Debugging => display_system::DisplayMode::PitchnamesCalmness,
    }
}

fn user_input_system(
    mut touch_events: EventReader<TouchInput>,
    mut keyboard_input_events: EventReader<KeyboardInput>,
    mut mouse_button_input_events: EventReader<MouseButtonInput>,
    mut settings: ResMut<display_system::SettingsState>,
) {
    for touch in touch_events.read() {
        if touch.phase == TouchPhase::Ended {
            settings.display_mode = cycle_display_mode(&settings.display_mode);
        }
    }

    for keyboard_input in keyboard_input_events.read() {
        if keyboard_input.state.is_pressed() {
            match keyboard_input.key_code {
                KeyCode::Space => {
                    settings.display_mode = cycle_display_mode(&settings.display_mode);
                }
                _ => {}
            }
        }
    }

    for mouse_button_input in mouse_button_input_events.read() {
        if mouse_button_input.state.is_pressed() {
            match mouse_button_input.button {
                MouseButton::Left => {
                    settings.display_mode = cycle_display_mode(&settings.display_mode);
                }
                _ => {}
            }
        }
    }
}

pub fn close_on_esc(
    mut commands: Commands,
    focused_windows: Query<(Entity, &Window)>,
    input: Res<ButtonInput<KeyCode>>,
) {
    for (window, focus) in focused_windows.iter() {
        if !focus.focused {
            continue;
        }

        if input.just_pressed(KeyCode::Escape) {
            commands.entity(window).despawn();
        }
    }
}

#[cfg(target_os = "android")]
fn handle_lifetime_events_system(
    mut lifetime_events: EventReader<AppLifecycle>,
    // audio_control_tx: ResMut<AudioControlChannelResource>,
    mut exit: EventWriter<AppExit>,
) {
    use bevy::window::AppLifecycle;

    for event in lifetime_events.read() {
        match event {
            // Upon receiving the `Suspended` event, the application has 1 frame before it is paused
            // As audio happens in an independent thread, it needs to be stopped
            AppLifecycle::WillSuspend | AppLifecycle::Suspended => {
                // FIXME: suspending audio does not seem to be good enough, we still get crashes
                // that block the entire android UI and sometimes seem to require a reboot.
                // audio_control_tx.0.send(AudioControl::Pause).unwrap();

                // log
                log::warn!("Application suspended, exiting.");

                // For now, just quit.
                // FIXME: also doesn't seem to work, the app is not exiting
                exit.send(AppExit::Success);

                // This is a workaround to exit the app, but it's not clean.
                std::process::exit(0);
                panic!("App should have exited");
            }
            // On `Resumed``, audio can continue playing
            AppLifecycle::WillResume => {
                // audio_control_tx.0.send(AudioControl::Play).unwrap();
            }
            // `Started` is the only other event for now, more to come in the next Bevy version
            _ => (),
        }
    }
}

#[cfg(target_os = "android")]
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

#[cfg(target_os = "android")]
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

// desktop main function
#[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
pub fn main_fun() -> Result<()> {
    // env_logger::init();

    use bevy::window::PresentMode;

    let audio_stream = pitchvis_audio::audio::AudioStream::new(SR, BUFSIZE).unwrap();

    let vqt = pitchvis_analysis::vqt::Vqt::new(
        SR,
        N_FFT,
        FREQ_A1,
        BUCKETS_PER_OCTAVE,
        OCTAVES,
        SPARSITY_QUANTILE,
        Q,
        GAMMA,
    );

    audio_stream.play().unwrap();

    let update_vqt_system = vqt_system::update_vqt_to_system(BUFSIZE);
    let update_analysis_state_system =
        analysis_system::update_analysis_state_to_system(OCTAVES, BUCKETS_PER_OCTAVE);
    #[cfg(feature = "ml")]
    let update_ml_system = ml_system::update_ml_to_system();
    let update_display_system =
        display_system::update_display_to_system(BUCKETS_PER_OCTAVE, OCTAVES);

    #[cfg(feature = "ml")]
    let ml_model_resource = ml_system::MlModelResource(ml_system::MlModel::new("model.pt"));

    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins, // .set(WindowPlugin {
        // primary_window: Some(Window {
        //     present_mode: PresentMode::AutoNoVsync,
        //     ..default()
        // }),
        // ..default()})
        LogDiagnosticsPlugin::default(),
        FrameTimeDiagnosticsPlugin,
        //.add_plugin(MaterialPlugin::<display_system::LineMaterial>::default())
        Material2dPlugin::<display_system::material::NoisyColorMaterial>::default(),
    ))
    .insert_resource(vqt_system::VqtResource(vqt))
    .insert_resource(vqt_system::VqtResultResource::new(
        OCTAVES,
        BUCKETS_PER_OCTAVE,
    ))
    .insert_resource(audio_system::AudioBufferResource(audio_stream.ring_buffer))
    .insert_resource(analysis_system::AnalysisStateResource(
        pitchvis_analysis::analysis::AnalysisState::new(
            OCTAVES * BUCKETS_PER_OCTAVE,
            pitchvis_analysis::analysis::SPECTROGRAM_LENGTH,
        ),
    ))
    .insert_resource(display_system::SettingsState {
        display_mode: display_system::DisplayMode::PitchnamesCalmness,
    });

    #[cfg(feature = "ml")]
    app.insert_resource(ml_model_resource);
    app.insert_resource(display_system::CylinderEntityListResource(Vec::new()))
        .add_systems(
            Startup,
            (
                display_system::setup_display_to_system(OCTAVES, BUCKETS_PER_OCTAVE),
                setup_fps_counter,
            ),
        )
        .add_systems(
            Update,
            (
                close_on_esc,
                frame_limiter_system,
                update_vqt_system,
                user_input_system,
                fps_text_update_system,
                fps_counter_showhide,
                update_analysis_state_system.after(update_vqt_system),
            ),
        );
    #[cfg(feature = "ml")]
    app.add_systems(
        Update,
        (
            update_ml_system.after(update_analysis_state_system),
            update_display_system.after(update_ml_system),
        ),
    )
    .run();
    #[cfg(not(feature = "ml"))]
    app.add_systems(
        Update,
        update_display_system.after(update_analysis_state_system),
    )
    .run();

    Ok(())
}

// android main function
#[cfg(target_os = "android")]
#[bevy_main]
fn main() -> AppExit {
    std::env::set_var("RUST_BACKTRACE", "1");

    use std::{process::exit, sync::mpsc};

    use bevy::{
        audio,
        log::LogPlugin,
        render::{
            settings::{WgpuFeatures, WgpuSettings},
            RenderPlugin,
        },
    };

    //env_logger::init();

    if !microphone_permission_granted("android.permission.RECORD_AUDIO") {
        log::info!("requesting microphone permission");
        request_microphone_permission(
            bevy_winit::ANDROID_APP
                .get()
                .expect("Bevy must be setup with the #[bevy_main] macro on Android"),
            "android.permission.RECORD_AUDIO",
        );
    }
    // wait until permission is granted
    while !microphone_permission_granted("android.permission.RECORD_AUDIO") {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // keep screen awake. This is a bitflags! enum, the second argument is an empty bitflags mask
    bevy_winit::ANDROID_APP
        .get()
        .expect("Bevy must be setup with the #[bevy_main] macro on Android")
        .set_window_flags(
            android_activity::WindowManagerFlags::KEEP_SCREEN_ON,
            android_activity::WindowManagerFlags::empty(),
        );

    let vqt = pitchvis_analysis::vqt::Vqt::new(
        SR,
        N_FFT,
        FREQ_A1,
        BUCKETS_PER_OCTAVE,
        OCTAVES,
        SPARSITY_QUANTILE,
        Q,
        GAMMA,
    );

    // let (audio_control_channel_tx, audio_control_channel_rx) = mpsc::channel::<AudioControl>();
    let mut ring_buffer = pitchvis_audio::audio::RingBuffer {
        buf: Vec::new(),
        gain: 0.0,
    };
    ring_buffer.buf.resize(BUFSIZE, 0f32);
    let ring_buffer = std::sync::Mutex::from(ring_buffer);
    let ring_buffer = std::sync::Arc::new(ring_buffer);
    // spawn thread for audio
    let ring_buffer_input_thread_clone = ring_buffer.clone();

    let mut agc = dagc::MonoAgc::new(0.07, 0.0001).expect("mono-agc creation failed");
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

                // log::info!("audio callback");

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

    let update_vqt_system = vqt_system::update_vqt_to_system(BUFSIZE);
    let update_analysis_state_system =
        analysis_system::update_analysis_state_to_system(OCTAVES, BUCKETS_PER_OCTAVE);
    let update_display_system =
        display_system::update_display_to_system(BUCKETS_PER_OCTAVE, OCTAVES);

    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    //resizable: false,
                    mode: bevy::window::WindowMode::BorderlessFullscreen,
                    ..default()
                }),
                ..default()
            })
            .set(LogPlugin {
                filter: "wgpu=debug,debug".to_string(),
                level: bevy::log::Level::DEBUG,
                ..default()
            }),
        LogDiagnosticsPlugin::default(),
        FrameTimeDiagnosticsPlugin::default(),
        Material2dPlugin::<display_system::material::NoisyColorMaterial>::default(),
    ))
    .insert_resource(vqt_system::VqtResource(vqt))
    .insert_resource(vqt_system::VqtResultResource::new(
        OCTAVES,
        BUCKETS_PER_OCTAVE,
    ))
    .insert_resource(audio_system::AudioBufferResource(ring_buffer))
    .insert_resource(analysis_system::AnalysisStateResource(
        pitchvis_analysis::analysis::AnalysisState::new(
            OCTAVES * BUCKETS_PER_OCTAVE,
            pitchvis_analysis::analysis::SPECTROGRAM_LENGTH,
        ),
    ))
    .insert_resource(display_system::CylinderEntityListResource(Vec::new()))
    // .insert_resource(AudioControlChannelResource(audio_control_channel_tx))
    .insert_resource(display_system::SettingsState {
        display_mode: display_system::DisplayMode::PitchnamesCalmness,
    })
    .add_systems(
        Startup,
        (
            display_system::setup_display_to_system(OCTAVES, BUCKETS_PER_OCTAVE),
            setup_fps_counter,
        ),
    )
    .add_systems(
        Update,
        (
            close_on_esc,
            frame_limiter_system,
            update_vqt_system,
            user_input_system,
            handle_lifetime_events_system,
            update_analysis_state_system.after(update_vqt_system),
            update_display_system.after(update_analysis_state_system),
            fps_text_update_system,
            fps_counter_showhide,
        ),
    );

    // MSAA makes some Android devices panic, this is under investigation
    // https://github.com/bevyengine/bevy/issues/8229
    #[cfg(target_os = "android")]
    app.insert_resource(Msaa::Off);

    app.run()
}
