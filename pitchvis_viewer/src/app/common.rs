use bevy::winit::UpdateMode;
use bevy::winit::WinitSettings;
use bevy_persistent::Persistent;
use serde::Deserialize;
use serde::Serialize;

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

#[derive(Resource, Serialize, Deserialize)]
pub struct SettingsState {
    pub display_mode: display_system::DisplayMode,
    pub visuals_mode: display_system::VisualsMode,
    pub fps_limit: Option<u32>,
}
#[derive(Resource)]
pub struct CurrentFpsLimit(pub Option<u32>);

pub fn set_frame_limiter_system(
    mut current_limit: ResMut<CurrentFpsLimit>,
    mut winit_settings: ResMut<WinitSettings>,
    settings: Res<Persistent<SettingsState>>,
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
                TextSpan::new("FPS: "),
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
    settings: Res<Persistent<SettingsState>>,
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
    settings: Res<Persistent<SettingsState>>,
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
    settings: Res<Persistent<SettingsState>>,
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
    settings: Res<Persistent<SettingsState>>,
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

/// Marker to find the container entity so we can show/hide the Buttons
#[derive(Component)]
pub struct ButtonRoot;

#[derive(Component)]
pub enum ButtonAction {
    VisualsMode,
    FpsLimit,
}

#[derive(Component)]
struct ConsumesPressEvents;

#[derive(Resource, Default)]
pub struct PressEventConsumed(bool);

pub fn setup_buttons(mut commands: Commands, settings: Res<Persistent<SettingsState>>) {
    let button_node = Node {
        justify_content: JustifyContent::Center,
        align_items: AlignItems::Center,
        border: UiRect::all(Val::Px(5.0)),
        padding: UiRect::all(Val::Px(4.0)),
        margin: UiRect::all(Val::Px(4.0)),
        ..default()
    };
    let text_font = TextFont {
        font_size: 16.0,
        ..default()
    };

    // create our UI root node
    // this is the wrapper/container for the text
    commands
        .spawn((
            ButtonRoot,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Percent(1.),
                top: Val::Percent(35.),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                padding: UiRect::all(Val::Px(4.0)),
                margin: UiRect::all(Val::Px(4.0)),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            ZIndex(i32::MAX),
            Visibility::Visible,
        ))
        .with_children(|parent| {
            parent
                .spawn((
                    Button,
                    Node {
                        ..button_node.clone()
                    },
                    BackgroundColor(Color::srgba(0.0, 0.2, 0.0, 0.5)),
                    BorderColor(Color::srgb(0.0, 0.5, 0.0)),
                    BorderRadius::MAX,
                    ButtonAction::VisualsMode,
                ))
                .insert(ConsumesPressEvents)
                .with_child((
                    Text::new(format!(
                        "Visuals Mode: {}",
                        match settings.visuals_mode {
                            display_system::VisualsMode::Full => "Full",
                            display_system::VisualsMode::Zen => "Zen",
                            display_system::VisualsMode::Galaxy => "Galaxy",
                        }
                    )),
                    TextColor(Color::WHITE),
                    text_font.clone(),
                ));
            parent
                .spawn((
                    Button,
                    Node {
                        ..button_node.clone()
                    },
                    BackgroundColor(Color::srgba(0.0, 0.2, 0.0, 0.5)),
                    BorderColor(Color::srgb(0.0, 0.5, 0.0)),
                    BorderRadius::MAX,
                    ButtonAction::FpsLimit,
                ))
                .insert(ConsumesPressEvents)
                .with_child((
                    Text::new(if let Some(fps_limit) = settings.fps_limit {
                        format!("FPS Limit: {}", fps_limit)
                    } else {
                        format!("FPS Limit: None")
                    }),
                    TextColor(Color::WHITE),
                    text_font.clone(),
                ));
        });
    commands.insert_resource(PressEventConsumed(false));
}

pub fn update_button_system(
    interaction_query: Query<
        (&Interaction, &ButtonAction, &Children),
        (Changed<Interaction>, With<Button>),
    >,
    mut settings: ResMut<Persistent<SettingsState>>,
    mut text_query: Query<&mut Text>,
    mut mouse_consumed: ResMut<PressEventConsumed>,
) {
    for (interaction, button_action, children) in &interaction_query {
        if *interaction == Interaction::Pressed {
            mouse_consumed.0 = true;
            let mut text = text_query.get_mut(children[0]).unwrap();
            match button_action {
                ButtonAction::VisualsMode => {
                    settings
                        .update(|settings| {
                            cycle_visuals_mode(&mut settings.visuals_mode);
                        })
                        .expect("failed to update settings");
                    **text = format!(
                        "Visuals Mode: {}",
                        match settings.visuals_mode {
                            display_system::VisualsMode::Full => "Full",
                            display_system::VisualsMode::Zen => "Zen",
                            display_system::VisualsMode::Galaxy => "Galaxy",
                        }
                    );
                }
                ButtonAction::FpsLimit => {
                    settings
                        .update(|settings| {
                            cycle_fps_limit(&mut settings.fps_limit);
                        })
                        .expect("failed to update settings");
                    if let Some(fps_limit) = settings.fps_limit {
                        **text = format!("FPS Limit: {}", fps_limit);
                    } else {
                        **text = format!("FPS Limit: None");
                    }
                }
            }
        }
    }
}

/// Toggle the FPS counter based on the display mode
pub fn button_showhide(
    mut q: Query<&mut Visibility, With<ButtonRoot>>,
    settings: Res<Persistent<SettingsState>>,
) {
    let mut vis = q.single_mut();
    if settings.display_mode == display_system::DisplayMode::Debugging {
        *vis = Visibility::Visible;
    } else {
        *vis = Visibility::Hidden;
    }
}

fn cycle_display_mode(mode: &mut display_system::DisplayMode) {
    *mode = match mode {
        display_system::DisplayMode::Normal => display_system::DisplayMode::Debugging,
        display_system::DisplayMode::Debugging => display_system::DisplayMode::Normal,
    }
}

fn cycle_visuals_mode(mode: &mut display_system::VisualsMode) {
    *mode = match mode {
        display_system::VisualsMode::Full => display_system::VisualsMode::Zen,
        display_system::VisualsMode::Zen => display_system::VisualsMode::Galaxy,
        display_system::VisualsMode::Galaxy => display_system::VisualsMode::Full,
    }
}

fn cycle_fps_limit(fps_limit: &mut Option<u32>) {
    *fps_limit = match fps_limit {
        None => Some(30),
        Some(30) => Some(60),
        _ => None,
    };
}

pub fn user_input_system(
    mut touch_events: EventReader<TouchInput>,
    mut keyboard_input_events: EventReader<KeyboardInput>,
    mut mouse_button_input_events: EventReader<MouseButtonInput>,
    mut settings: ResMut<Persistent<SettingsState>>,
    mouse_consumed: Res<PressEventConsumed>,
    mut commands: Commands,
) {
    for touch in touch_events.read() {
        match touch.phase {
            TouchPhase::Ended => {
                if !mouse_consumed.0 {
                    settings
                        .update(|settings| {
                            cycle_display_mode(&mut settings.display_mode);
                        })
                        .expect("failed to update settings");
                }
                // reset the consumed state for the next frame
                commands.insert_resource(PressEventConsumed(false));
            }
            _ => {}
        }
    }

    for keyboard_input in keyboard_input_events.read() {
        if keyboard_input.state.is_pressed() {
            match keyboard_input.key_code {
                KeyCode::Space => {
                    settings
                        .update(|settings| {
                            cycle_display_mode(&mut settings.display_mode);
                        })
                        .expect("failed to update settings");
                }
                _ => {}
            }
        }
    }

    for mouse_button_input in mouse_button_input_events.read() {
        if mouse_button_input.state.is_pressed() {
            #[allow(clippy::single_match)]
            match mouse_button_input.button {
                MouseButton::Left => {
                    if !mouse_consumed.0 {
                        settings
                            .update(|settings| {
                                cycle_display_mode(&mut settings.display_mode);
                            })
                            .expect("failed to update settings");
                    }
                    // reset the consumed state for the next frame
                    commands.insert_resource(PressEventConsumed(false));
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
