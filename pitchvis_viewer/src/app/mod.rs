#[cfg(target_os = "android")]
pub(crate) mod android_app;
#[cfg(not(any(target_os = "android", target_arch = "wasm32")))]
pub(crate) mod desktop_app;
#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm_app;

use crate::display_system;
use bevy::core_pipeline::bloom::BloomCompositeMode;
use bevy::core_pipeline::bloom::BloomSettings;
use bevy::diagnostic::DiagnosticsStore;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::input::keyboard::KeyboardInput;
use bevy::input::mouse::MouseButtonInput;
use bevy::input::touch::TouchPhase;
use bevy::prelude::*;

/// Marker to find the container entity so we can show/hide the FPS counter
#[derive(Component)]
struct FpsRoot;

/// Marker to find the text entity so we can update it
#[derive(Component)]
struct FpsText;

#[derive(Resource)]
pub struct SettingsState {
    pub display_mode: display_system::DisplayMode,
    pub fps_limit: Option<u32>,
}

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

/// Toggle the FPS counter based on the display mode
fn fps_counter_showhide(
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

/// Marker to find the text entity so we can update it
#[derive(Component)]
struct BloomSettingsText;

fn update_bloom_settings(
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

fn setup_bloom_ui(mut commands: Commands) {
    // UI
    commands.spawn((
        BloomSettingsText,
        TextBundle::from_section("", TextStyle::default()).with_style(Style {
            position_type: PositionType::Absolute,
            bottom: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        }),
    ));
}

#[cfg(not(target_arch = "wasm32"))]
fn frame_limiter_system(settings: Res<SettingsState>) {
    if let Some(fps) = settings.fps_limit {
        use std::{thread, time};
        thread::sleep(time::Duration::from_micros(
            (1_000_000 / fps as u64).saturating_sub(5_000),
        ));
    }
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
    mut settings: ResMut<SettingsState>,
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
                KeyCode::KeyF => match settings.fps_limit {
                    None => settings.fps_limit = Some(30),
                    Some(30) => settings.fps_limit = Some(60),
                    _ => settings.fps_limit = None,
                },
                _ => {}
            }
        }
    }

    for mouse_button_input in mouse_button_input_events.read() {
        if mouse_button_input.state.is_pressed() {
            #[allow(clippy::single_match)]
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
