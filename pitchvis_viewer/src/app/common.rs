use bevy::winit::UpdateMode;
use bevy::winit::WinitSettings;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use crate::analysis_system::AnalysisStateResource;
use crate::audio_system::AudioBufferResource;
use crate::display_system;
use crate::vqt_system::VqtResource;
use bevy::core_pipeline::bloom::Bloom;
use bevy::core_pipeline::bloom::BloomCompositeMode;
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
#[derive(Resource)]
pub struct CurrentFpsLimit(pub Option<u32>);

pub fn set_frame_limiter_system(
    mut current_limit: ResMut<CurrentFpsLimit>,
    mut winit_settings: ResMut<WinitSettings>,
    settings: Res<SettingsState>,
) {
    if settings.fps_limit != current_limit.0 {
        current_limit.0 = settings.fps_limit;
        *winit_settings = match settings.fps_limit {
            Some(fps) => WinitSettings {
                focused_mode: UpdateMode::reactive(std::time::Duration::from_secs_f32(
                    1.0 / fps as f32,
                )),
                unfocused_mode: UpdateMode::reactive(std::time::Duration::from_secs_f32(
                    1.0 / fps as f32,
                )),
            },
            None => WinitSettings {
                focused_mode: UpdateMode::Continuous,
                unfocused_mode: UpdateMode::Continuous,
            },
        };
    }
}

/// Marker to find the container entity so we can show/hide the FPS counter
#[derive(Component)]
pub struct FpsRoot;

pub fn setup_fps_counter(mut commands: Commands) {
    let text_font = TextFont {
        font_size: 16.0,
        ..default()
    };

    // create our UI root node
    // this is the wrapper/container for the text
    commands
        .spawn((
            FpsRoot,
            Text::default(),
            text_font.clone(),
            Node {
                position_type: PositionType::Absolute,
                left: Val::Percent(1.),
                top: Val::Percent(1.),
                // set bottom/left to Auto, so it can be
                // automatically sized depending on the text
                // bottom: Val::Auto,
                // right: Val::Auto,
                // give it some padding for readability
                padding: UiRect::all(Val::Px(4.0)),
                ..default()
            },
            // give it a dark background for readability
            BackgroundColor(Color::BLACK.with_alpha(0.5)),
            // make it "always on top" by setting the Z index to maximum
            // we want it to be displayed over all other UI
            ZIndex(i32::MAX),
            Visibility::Visible,
        ))
        .with_children(|builder| {
            builder.spawn((
                TextSpan::new("FPS (F/long press to cycle): "),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new(" N/A"),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new("\nAudio latency: "),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new("N/A"),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new("\nAudio chunk size: "),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new("N/A"),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new("\nVQT latency: "),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new("N/A"),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            // TODO: add section for analysis smoothing latency
        });
}

pub fn update_fps_text_system(
    diagnostics: Res<DiagnosticsStore>,
    query: Query<Entity, With<FpsRoot>>,
    mut writer: TextUiWriter,
    settings: Res<SettingsState>,
    audio_buffer: Res<AudioBufferResource>,
    vqt: Res<VqtResource>,
) {
    let entity = query.single();

    // try to get a "smoothed" FPS value from Bevy
    if let Some(value) = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|fps| fps.smoothed())
    {
        // Format the number as to leave space for 4 digits, just in case,
        // right-aligned and rounded. This helps readability when the
        // number changes rapidly.
        *writer.text(entity, 2) = format!("{value:>4.0}");

        // // Let's make it extra fancy by changing the color of the
        // // text according to the FPS value:
        *writer.color(entity, 2) = if value >= 120.0 {
            // Above 120 FPS, use green color
            TextColor(Color::srgb(0.0, 1.0, 0.0))
        } else if value >= 60.0 {
            // Between 60-120 FPS, gradually transition from yellow to green
            TextColor(Color::srgb(
                (1.0 - (value - 60.0) / (120.0 - 60.0)) as f32,
                1.0,
                0.0,
            ))
        } else if value >= 30.0 {
            // Between 30-60 FPS, gradually transition from red to yellow
            TextColor(Color::srgb(
                1.0,
                ((value - 30.0) / (60.0 - 30.0)) as f32,
                0.0,
            ))
        } else {
            // Below 30 FPS, use red color
            TextColor(Color::srgb(1.0, 0.0, 0.0))
        }
    } else {
        // display "N/A" if we can't get a FPS measurement
        // add an extra space to preserve alignment
        *writer.text(entity, 2) = " N/A".into();
        *writer.color(entity, 2) = TextColor(Color::WHITE);
    }
    if let Some(fps_limit) = settings.fps_limit {
        (*writer.text(entity, 2)).push_str(&format!("/{}", fps_limit));
    } else {
        (*writer.text(entity, 2)).push_str("");
    }
    if let Some(latency_ms) = audio_buffer.0.lock().unwrap().latency_ms {
        *writer.text(entity, 4) = format!("{:.2}ms", latency_ms);
    } else {
        *writer.text(entity, 4) = "N/A".into();
    }
    *writer.text(entity, 6) = format!("{:.2}ms", audio_buffer.0.lock().unwrap().chunk_size_ms);
    *writer.text(entity, 8) = format!("{}ms", vqt.0.delay.as_millis());
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

pub fn setup_analysis_text(mut commands: Commands) {
    let text_font = TextFont {
        font_size: 16.0,
        ..default()
    };

    commands
        .spawn((
            AnalysisRoot,
            Text::default(),
            text_font.clone(),
            Node {
                position_type: PositionType::Absolute,
                right: Val::Percent(1.),
                bottom: Val::Percent(1.),
                // set bottom/left to Auto, so it can be
                // automatically sized depending on the text
                // bottom: Val::Auto,
                // right: Val::Auto,
                // give it some padding for readability
                padding: UiRect::all(Val::Px(4.0)),
                ..default()
            },
            // give it a dark background for readability
            BackgroundColor(Color::BLACK.with_alpha(0.5)),
            // make it "always on top" by setting the Z index to maximum
            // we want it to be displayed over all other UI
            ZIndex(i32::MAX),
            Visibility::Visible,
        ))
        .with_children(|builder| {
            builder.spawn((
                TextSpan::new("Tuning drift: "),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new(" N/A"),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
        });
}

pub fn update_analysis_text_system(
    // TODO: replace with Single in bevy 0.15
    query: Query<Entity, With<AnalysisRoot>>,
    analysis: Res<AnalysisStateResource>,
    mut writer: TextUiWriter,
) {
    let entity = query.single();

    let inaccuracy = analysis.0.smoothed_tuning_grid_inaccuracy.get().round();
    // Format the number as to leave space for 3 digits, just in case,
    // right-aligned and rounded. This helps readability when the
    // number changes rapidly.
    *writer.text(entity, 2) = format!("{inaccuracy:>3.0}");

    let inaccuracy_abs = inaccuracy.abs();
    *writer.color(entity, 2) = TextColor(if inaccuracy_abs <= 10.0 {
        Color::srgb(0.0, 1.0, 0.0)
    } else if inaccuracy_abs <= 20.0 {
        Color::srgb((inaccuracy_abs - 10.0) / (20.0 - 10.0), 1.0, 0.0)
    } else if inaccuracy_abs <= 30.0 {
        Color::srgb(1.0, 1.0 - (inaccuracy_abs - 20.0) / (30.0 - 20.0), 0.0)
    } else {
        Color::srgb(1.0, 0.0, 0.0)
    });
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

pub fn setup_bloom_ui(mut commands: Commands) {
    // UI
    commands.spawn((
        BloomSettingsText,
        Text::new(""),
        TextColor(Color::WHITE),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
        // give it a dark background for readability
        BackgroundColor(Color::BLACK.with_alpha(0.5)),
        // make it "always on top" by setting the Z index to maximum
        // we want it to be displayed over all other UI
        ZIndex(i32::MAX),
    ));
}

pub fn update_bloom_settings(
    mut camera: Query<(Entity, Option<&mut Bloom>), With<Camera>>,
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
    let text = &mut text.0 .0;

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
            bloom_settings.prefilter.threshold
        ));
        text.push_str(&format!(
            "(U/J) Threshold softness: {}\n",
            bloom_settings.prefilter.threshold_softness
        ));

        let dt = time.delta_secs();

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
            bloom_settings.prefilter.threshold -= dt;
        }
        if keycode.pressed(KeyCode::KeyY) {
            bloom_settings.prefilter.threshold += dt;
        }
        bloom_settings.prefilter.threshold = bloom_settings.prefilter.threshold.max(0.0);

        if keycode.pressed(KeyCode::KeyJ) {
            bloom_settings.prefilter.threshold_softness -= dt / 10.0;
        }
        if keycode.pressed(KeyCode::KeyU) {
            bloom_settings.prefilter.threshold_softness += dt / 10.0;
        }
        bloom_settings.prefilter.threshold_softness =
            bloom_settings.prefilter.threshold_softness.clamp(0.0, 1.0);
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
