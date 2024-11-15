use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use crate::analysis_system::AnalysisStateResource;
use crate::audio_system::AudioBufferResource;
use crate::display_system;
use crate::vqt_system::VqtResource;
use bevy::core_pipeline::bloom::BloomCompositeMode;
use bevy::core_pipeline::bloom::BloomSettings;
use bevy::diagnostic::DiagnosticsStore;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::input::keyboard::KeyboardInput;
use bevy::input::mouse::MouseButtonInput;
use bevy::input::touch::TouchPhase;
use bevy::prelude::*;

#[derive(Resource)]
pub struct SettingsState {
    pub display_mode: display_system::DisplayMode,
    pub fps_limit: Option<u32>,
}

/// Marker to find the container entity so we can show/hide the FPS counter
#[derive(Component)]
pub struct FpsRoot;

/// Marker to find the text entity so we can update it
#[derive(Component)]
pub struct FpsText;

pub fn setup_fps_counter(mut commands: Commands) {
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
                        value: "FPS (F/long press to cycle): ".into(),
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
                    TextSection {
                        value: "\nAudio latency: ".into(),
                        style: TextStyle {
                            font_size: 16.0,
                            color: Color::WHITE,
                            ..default()
                        },
                    },
                    TextSection {
                        value: "N/A".into(),
                        style: TextStyle {
                            font_size: 16.0,
                            color: Color::WHITE,
                            ..default()
                        },
                    },
                    TextSection {
                        value: "\nVQT latency: ".into(),
                        style: TextStyle {
                            font_size: 16.0,
                            color: Color::WHITE,
                            ..default()
                        },
                    },
                    TextSection {
                        value: "N/A".into(),
                        style: TextStyle {
                            font_size: 16.0,
                            color: Color::WHITE,
                            ..default()
                        },
                    },
                    // TODO: add section for analysis smoothing latency
                ]),
                ..Default::default()
            },
        ))
        .id();
    commands.entity(root).push_children(&[text_fps]);
}

pub fn update_fps_text_system(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut Text, With<FpsText>>,
    settings: Res<SettingsState>,
    audio_buffer: Res<AudioBufferResource>,
    vqt: Res<VqtResource>,
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
        if let Some(fps_limit) = settings.fps_limit {
            text.sections[1].value.push_str(&format!("/{}", fps_limit));
        }
        if let Some(latency_ms) = audio_buffer.0.lock().unwrap().latency_ms {
            text.sections[3].value = format!("{:.2}ms", latency_ms);
        } else {
            text.sections[3].value = "N/A".into();
        }
        text.sections[5].value = format!("{}ms", vqt.0.delay.as_millis());
    }
}

/// Toggle the FPS counter based on the display mode
pub fn fps_counter_showhide(
    mut q: Query<&mut Visibility, With<FpsRoot>>,
    settings: Res<SettingsState>,
) {
    let mut vis = q.single_mut();
    if settings.display_mode == display_system::DisplayMode::Debugging {
        *vis = Visibility::Visible;
    } else {
        *vis = Visibility::Hidden;
    }
}

/// Marker to find the container entity so we can show/hide the FPS counter
#[derive(Component)]
pub struct AnalysisRoot;

/// Marker to find the text entity so we can update it
#[derive(Component)]
pub struct AnalysisText;

pub fn setup_analysis_text(mut commands: Commands) {
    let root = commands
        .spawn((
            AnalysisRoot,
            NodeBundle {
                background_color: BackgroundColor(Color::BLACK.with_alpha(0.5)),
                z_index: ZIndex::Global(i32::MAX),
                style: Style {
                    position_type: PositionType::Absolute,
                    left: Val::Auto,
                    top: Val::Auto,
                    right: Val::Percent(1.),
                    bottom: Val::Percent(1.),
                    // give it some padding for readability
                    padding: UiRect::all(Val::Px(4.0)),
                    ..Default::default()
                },
                visibility: Visibility::Visible,
                ..Default::default()
            },
        ))
        .id();
    let text_analysis = commands
        .spawn((
            AnalysisText,
            TextBundle {
                text: Text::from_sections([
                    TextSection {
                        value: "Tuning drift: ".into(),
                        style: TextStyle {
                            font_size: 16.0,
                            color: Color::WHITE,
                            ..default()
                        },
                    },
                    TextSection {
                        value: " N/A".into(),
                        style: TextStyle {
                            font_size: 16.0,
                            color: Color::WHITE,
                            ..default()
                        },
                    },
                ]),
                ..Default::default()
            },
        ))
        .id();
    commands.entity(root).push_children(&[text_analysis]);
}

pub fn update_analysis_text_system(
    // TODO: replace with Single in bevy 0.15
    mut query: Query<&mut Text, With<AnalysisText>>,
    analysis: Res<AnalysisStateResource>,
) {
    let inaccuracy = analysis.0.smoothed_tuning_grid_inaccuracy.get().round();
    let mut text = query.single_mut();

    // Format the number as to leave space for 3 digits, just in case,
    // right-aligned and rounded. This helps readability when the
    // number changes rapidly.
    text.sections[1].value = format!("{inaccuracy:>3.0}");

    let inaccuracy_abs = inaccuracy.abs();
    text.sections[1].style.color = if inaccuracy_abs <= 10.0 {
        Color::srgb(0.0, 1.0, 0.0)
    } else if inaccuracy_abs <= 20.0 {
        Color::srgb((inaccuracy_abs - 10.0) / (20.0 - 10.0), 1.0, 0.0)
    } else if inaccuracy_abs <= 30.0 {
        Color::srgb(1.0, 1.0 - (inaccuracy_abs - 20.0) / (30.0 - 20.0), 0.0)
    } else {
        Color::srgb(1.0, 0.0, 0.0)
    };
}

/// Toggle the FPS counter based on the display mode
pub fn analysis_text_showhide(
    mut q: Query<&mut Visibility, With<AnalysisRoot>>,
    settings: Res<SettingsState>,
) {
    // TODO: move all showhides into one system that updates the scene based on the display mode
    let mut vis = q.single_mut();
    if settings.display_mode == display_system::DisplayMode::Debugging {
        *vis = Visibility::Visible;
    } else {
        *vis = Visibility::Hidden;
    }
}

/// Marker to find the text entity so we can update it
#[derive(Component)]
pub struct BloomSettingsText;

pub fn update_bloom_settings(
    mut camera: Query<(Entity, Option<&mut BloomSettings>), With<Camera>>,
    mut text: Query<(&mut Text, &mut Visibility), With<BloomSettingsText>>,
    keycode: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    settings: Res<SettingsState>,
) {
    let mut text = text.single_mut();
    if settings.display_mode != display_system::DisplayMode::Debugging {
        *text.1 = Visibility::Hidden;
        return;
    } else {
        *text.1 = Visibility::Visible;
    }

    let bloom_settings = camera.single_mut();
    let text = &mut text.0.sections[0].value;

    if let (_, Some(mut bloom_settings)) = bloom_settings {
        *text = "BloomSettings\n".to_string();
        text.push_str(&format!("(Q/A) Intensity: {}\n", bloom_settings.intensity));
        text.push_str(&format!(
            "(W/S) Low-frequency boost: {}\n",
            bloom_settings.low_frequency_boost
        ));
        text.push_str(&format!(
            "(E/D) Low-frequency boost curvature: {}\n",
            bloom_settings.low_frequency_boost_curvature
        ));
        text.push_str(&format!(
            "(R/F) High-pass frequency: {}\n",
            bloom_settings.high_pass_frequency
        ));
        text.push_str(&format!(
            "(T/G) Mode: {}\n",
            match bloom_settings.composite_mode {
                BloomCompositeMode::EnergyConserving => "Energy-conserving",
                BloomCompositeMode::Additive => "Additive",
            }
        ));
        text.push_str(&format!(
            "(Y/H) Threshold: {}\n",
            bloom_settings.prefilter_settings.threshold
        ));
        text.push_str(&format!(
            "(U/J) Threshold softness: {}\n",
            bloom_settings.prefilter_settings.threshold_softness
        ));

        let dt = time.delta_seconds();

        if keycode.pressed(KeyCode::KeyA) {
            bloom_settings.intensity -= dt / 10.0;
        }
        if keycode.pressed(KeyCode::KeyQ) {
            bloom_settings.intensity += dt / 10.0;
        }
        bloom_settings.intensity = bloom_settings.intensity.clamp(0.0, 1.0);

        if keycode.pressed(KeyCode::KeyS) {
            bloom_settings.low_frequency_boost -= dt / 10.0;
        }
        if keycode.pressed(KeyCode::KeyW) {
            bloom_settings.low_frequency_boost += dt / 10.0;
        }
        bloom_settings.low_frequency_boost = bloom_settings.low_frequency_boost.clamp(0.0, 1.0);

        if keycode.pressed(KeyCode::KeyD) {
            bloom_settings.low_frequency_boost_curvature -= dt / 10.0;
        }
        if keycode.pressed(KeyCode::KeyE) {
            bloom_settings.low_frequency_boost_curvature += dt / 10.0;
        }
        bloom_settings.low_frequency_boost_curvature =
            bloom_settings.low_frequency_boost_curvature.clamp(0.0, 1.0);

        if keycode.pressed(KeyCode::KeyF) {
            bloom_settings.high_pass_frequency -= dt / 10.0;
        }
        if keycode.pressed(KeyCode::KeyR) {
            bloom_settings.high_pass_frequency += dt / 10.0;
        }
        bloom_settings.high_pass_frequency = bloom_settings.high_pass_frequency.clamp(0.0, 1.0);

        if keycode.pressed(KeyCode::KeyG) {
            bloom_settings.composite_mode = BloomCompositeMode::Additive;
        }
        if keycode.pressed(KeyCode::KeyT) {
            bloom_settings.composite_mode = BloomCompositeMode::EnergyConserving;
        }

        if keycode.pressed(KeyCode::KeyH) {
            bloom_settings.prefilter_settings.threshold -= dt;
        }
        if keycode.pressed(KeyCode::KeyY) {
            bloom_settings.prefilter_settings.threshold += dt;
        }
        bloom_settings.prefilter_settings.threshold =
            bloom_settings.prefilter_settings.threshold.max(0.0);

        if keycode.pressed(KeyCode::KeyJ) {
            bloom_settings.prefilter_settings.threshold_softness -= dt / 10.0;
        }
        if keycode.pressed(KeyCode::KeyU) {
            bloom_settings.prefilter_settings.threshold_softness += dt / 10.0;
        }
        bloom_settings.prefilter_settings.threshold_softness = bloom_settings
            .prefilter_settings
            .threshold_softness
            .clamp(0.0, 1.0);
    }
}

pub fn setup_bloom_ui(mut commands: Commands) {
    // UI
    commands.spawn((
        BloomSettingsText,
        TextBundle::from_section(
            "",
            TextStyle {
                font_size: 16.0,
                color: Color::WHITE,
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            bottom: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        }),
    ));
}

#[cfg(not(target_arch = "wasm32"))]
pub fn frame_limiter_system(settings: Res<SettingsState>) {
    if let Some(fps) = settings.fps_limit {
        use std::{thread, time};
        thread::sleep(time::Duration::from_micros(
            (1_000_000 / fps as u64).saturating_sub(5_000),
        ));
    }
}

fn cycle_display_mode(mode: &mut display_system::DisplayMode) {
    *mode = match mode {
        display_system::DisplayMode::PitchnamesCalmness => display_system::DisplayMode::Calmness,
        display_system::DisplayMode::Calmness => display_system::DisplayMode::Debugging,
        display_system::DisplayMode::Debugging => display_system::DisplayMode::PitchnamesCalmness,
    }
}

fn cycle_fps_limit(fps_limit: &mut Option<u32>) {
    *fps_limit = match fps_limit {
        None => Some(30),
        Some(30) => Some(60),
        _ => None,
    };
}

#[derive(Resource, Default)]
pub struct ActiveTouches(Arc<Mutex<HashMap<u64, Instant>>>);

pub fn user_input_system(
    mut touch_events: EventReader<TouchInput>,
    active_touches: Res<ActiveTouches>,
    mut keyboard_input_events: EventReader<KeyboardInput>,
    mut mouse_button_input_events: EventReader<MouseButtonInput>,
    mut settings: ResMut<SettingsState>,
) {
    const LONG_PRESS_DURATION: f32 = 1.0;
    // trigger fps limit change immediately after long press duration even if touch is still held
    active_touches.0.lock().unwrap().retain(|_, touch_start| {
        if touch_start.elapsed().as_secs_f32() >= LONG_PRESS_DURATION {
            cycle_fps_limit(&mut settings.fps_limit);
            false
        } else {
            true
        }
    });

    for touch in touch_events.read() {
        match touch.phase {
            TouchPhase::Started => {
                active_touches
                    .0
                    .lock()
                    .unwrap()
                    .insert(touch.id, Instant::now());
            }
            TouchPhase::Ended => {
                // trigger display mode change on touch release only
                let touch_start = active_touches.0.lock().unwrap().remove(&touch.id);
                if let Some(touch_start) = touch_start {
                    if touch_start.elapsed().as_secs_f32() < LONG_PRESS_DURATION {
                        cycle_display_mode(&mut settings.display_mode);
                    }
                }
            }
            _ => {}
        }
    }

    for keyboard_input in keyboard_input_events.read() {
        if keyboard_input.state.is_pressed() {
            match keyboard_input.key_code {
                KeyCode::Space => {
                    cycle_display_mode(&mut settings.display_mode);
                }
                KeyCode::KeyF => cycle_fps_limit(&mut settings.fps_limit),
                _ => {}
            }
        }
    }

    for mouse_button_input in mouse_button_input_events.read() {
        if mouse_button_input.state.is_pressed() {
            #[allow(clippy::single_match)]
            match mouse_button_input.button {
                MouseButton::Left => {
                    cycle_display_mode(&mut settings.display_mode);
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
