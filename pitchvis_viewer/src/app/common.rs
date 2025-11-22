use bevy::winit::UpdateMode;
use bevy::winit::WinitSettings;
use bevy_persistent::Persistent;
use bevy_persistent::StorageFormat;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use crate::analysis_system::AnalysisStateResource;
use crate::audio_system::AudioBufferResource;
use crate::display_system;
use crate::vqt_system::{VqtResource, VqtResultResource};
use bevy::asset::AssetMetaCheck;
use bevy::diagnostic::DiagnosticsStore;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::input::keyboard::KeyboardInput;
use bevy::input::mouse::MouseButtonInput;
use bevy::input::touch::TouchPhase;
use bevy::sprite_render::Material2dPlugin;
// use bevy::post_process::bloom::Bloom;
// use bevy::post_process::bloom::BloomCompositeMode;
use bevy::prelude::*;
use pitchvis_analysis::analysis::{AnalysisParameters, AnalysisState};
use pitchvis_analysis::vqt::{Vqt, VqtParameters, VqtRange};
use pitchvis_audio::RingBuffer;

#[derive(Resource, Serialize, Deserialize)]
pub struct SettingsState {
    pub display_mode: display_system::DisplayMode,
    pub visuals_mode: display_system::VisualsMode,
    pub fps_limit: Option<u32>,
    pub vqt_smoothing_mode: display_system::VQTSmoothingMode,
    #[serde(default = "default_true")]
    pub enable_vibrato: bool,
    #[serde(default = "default_true")]
    pub enable_attack_detection: bool,
    #[serde(default = "default_true")]
    pub enable_glissando: bool,
    #[serde(default = "default_true")]
    pub enable_chord_recognition: bool,
    #[serde(default = "default_true")]
    pub enable_harmonic_lines: bool,
    #[serde(default = "default_false")]
    pub enable_root_note_tinting: bool,
}

fn default_false() -> bool {
    false
}

fn default_true() -> bool {
    true
}
#[derive(Resource)]
pub struct CurrentFpsLimit(pub Option<u32>);

#[derive(Resource)]
pub struct CurrentVQTSmoothingMode(pub display_system::VQTSmoothingMode);

pub fn set_vqt_smoothing_system(
    mut current_mode: ResMut<CurrentVQTSmoothingMode>,
    mut analysis_state: ResMut<AnalysisStateResource>,
    settings: Res<Persistent<SettingsState>>,
) {
    if settings.vqt_smoothing_mode != current_mode.0 {
        current_mode.0 = settings.vqt_smoothing_mode;
        let new_duration = settings.vqt_smoothing_mode.to_duration();
        analysis_state.0.update_vqt_smoothing_duration(new_duration);
    }
}

/// Resource to track screen lock state
#[derive(Resource)]
pub struct ScreenLockState(pub bool);

/// Resource to track when the last tap occurred while screen was locked
#[derive(Resource)]
pub struct ScreenLockTapTime(pub Option<Instant>);

/// Resource to track if the screen lock indicator button is currently being pressed
#[derive(Resource)]
pub struct ScreenLockIndicatorPressed(pub bool);

/// Resource to track active touches for long press detection
#[derive(Resource)]
pub struct ActiveTouches(pub Arc<Mutex<HashMap<u64, Instant>>>);

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

pub fn setup_fps_counter(mut commands: Commands, asset_server: Res<AssetServer>) {
    let font = asset_server.load("fonts/DejaVuSans.ttf");
    let text_font = TextFont {
        font: font.clone(),
        font_size: 16.0,
        ..Default::default()
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
            builder.spawn((
                TextSpan::new("\nVQT smoothing: "),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new("N/A"),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
        });
}

pub fn update_fps_text_system(
    diagnostics: Res<DiagnosticsStore>,
    query: Query<Entity, With<FpsRoot>>,
    mut writer: TextUiWriter,
    settings: Res<Persistent<SettingsState>>,
    audio_buffer: Res<AudioBufferResource>,
    vqt: Res<VqtResource>,
    analysis: Res<AnalysisStateResource>,
) {
    let entity = query.single().expect("Failed to get entity.");

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
    *writer.text(entity, 6) = format!(
        "{:>7}",
        format!("{:.2}ms", audio_buffer.0.lock().unwrap().chunk_size_ms)
    );
    *writer.text(entity, 8) = format!("{}ms", vqt.0.delay.as_millis());

    // Display VQT smoothing as min-max range (varies by frequency and calmness)
    if let Some(base_duration) = settings.vqt_smoothing_mode.to_duration() {
        let base_ms = base_duration.as_millis() as f32;

        // Get current calmness from analysis state
        let calmness = analysis.0.smoothed_scene_calmness.get();
        let calmness_multiplier = analysis.0.params.vqt_smoothing_calmness_min
            + (analysis.0.params.vqt_smoothing_calmness_max
                - analysis.0.params.vqt_smoothing_calmness_min)
                * calmness;

        // Frequency multiplier range: bass (1.5x) to treble (1.0x)
        // Calmness multiplier range: energetic (0.6x) to calm (2.0x)
        let min_duration_ms = (base_ms * 1.0 * calmness_multiplier) as u32; // Treble with current calmness
        let max_duration_ms = (base_ms * 1.5 * calmness_multiplier) as u32; // Bass with current calmness

        *writer.text(entity, 10) = format!("{}-{}ms", min_duration_ms, max_duration_ms);
    } else {
        // None mode - no smoothing
        *writer.text(entity, 10) = "None".to_string();
    }
}

/// Toggle the FPS counter based on the display mode
pub fn fps_counter_showhide(
    mut q: Query<&mut Visibility, With<FpsRoot>>,
    settings: Res<Persistent<SettingsState>>,
) -> Result<()> {
    let mut vis = q.single_mut()?;
    if settings.display_mode == display_system::DisplayMode::Debugging {
        *vis = Visibility::Visible;
    } else {
        *vis = Visibility::Hidden;
    }

    Ok(())
}

/// Marker to find the container entity so we can show/hide the analysis text
#[derive(Component)]
pub struct AnalysisRoot;

/// Marker to find the container entity for debug info (bottom left)
#[derive(Component)]
pub struct DebugRoot;

/// Marker to find the screen lock indicator
#[derive(Component)]
pub struct ScreenLockIndicator;

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
                // give it some padding for readability
                // padding: UiRect::all(Val::Px(4.0)),
                ..default()
            },
            TextLayout::new_with_justify(Justify::Center),
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
            // builder.spawn((
            //     TextSpan::new("\nActive attacks: "),
            //     TextColor(Color::WHITE),
            //     text_font.clone(),
            // ));
            // builder.spawn((
            //     TextSpan::new("0"),
            //     TextColor(Color::WHITE),
            //     text_font.clone(),
            // ));
            // builder.spawn((
            //     TextSpan::new("\nVibrato (conf>0.7): "),
            //     TextColor(Color::WHITE),
            //     text_font.clone(),
            // // ));
            // builder.spawn((
            //     TextSpan::new("0"),
            //     TextColor(Color::WHITE),
            //     text_font.clone(),
            // ));
            // builder.spawn((
            //     TextSpan::new("\nGlissandi: "),
            //     TextColor(Color::WHITE),
            //     text_font.clone(),
            // ));
            // builder.spawn((
            //     TextSpan::new("0"),
            //     TextColor(Color::WHITE),
            //     text_font.clone(),
            // ));
            // builder.spawn((
            //     TextSpan::new("\nTracked peaks: "),
            //     TextColor(Color::WHITE),
            //     text_font.clone(),
            // ));
            // builder.spawn((
            //     TextSpan::new("0"),
            //     TextColor(Color::WHITE),
            //     text_font.clone(),
            // ));
            builder.spawn((
                TextSpan::new("\nChord: "),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new("None"),
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
) -> Result<()> {
    let entity = query.single()?;

    // Tuning drift
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

    // Active attacks count
    // let active_attacks = analysis.0.current_attacks.len();
    // *writer.text(entity, 4) = format!("{}", active_attacks);
    // *writer.color(entity, 4) = if active_attacks > 0 {
    //     TextColor(Color::srgb(1.0, 0.8, 0.0)) // Yellow when attacks are happening
    // } else {
    //     TextColor(Color::WHITE)
    // };

    // Vibrato count (confidence > 0.7)
    // let vibrato_count = analysis
    //     .0
    //     .vibrato_states
    //     .iter()
    //     .filter(|v| v.is_active && v.confidence > 0.7)
    //     .count();
    // *writer.text(entity, 6) = format!("{}", vibrato_count);
    // *writer.color(entity, 6) = if vibrato_count > 0 {
    //     TextColor(Color::srgb(0.5, 0.8, 1.0)) // Cyan when vibrato detected
    // } else {
    //     TextColor(Color::WHITE)
    // };

    // Glissandi count
    // let glissando_count = analysis.0.glissandi.len();
    // *writer.text(entity, 8) = format!("{}", glissando_count);
    // *writer.color(entity, 8) = if glissando_count > 0 {
    //     TextColor(Color::srgb(1.0, 0.5, 1.0)) // Magenta when glissandi present
    // } else {
    //     TextColor(Color::WHITE)
    // };

    // Tracked peaks count
    // let tracked_peaks = analysis.0.tracked_peaks.len();
    // *writer.text(entity, 10) = format!("{}", tracked_peaks);

    // Detected chord
    if let Some(ref chord) = analysis.0.detected_chord {
        *writer.text(entity, 4) = format!("{}", chord.name());
        *writer.color(entity, 4) = TextColor(Color::srgb(0.5, 1.0, 0.5)); // Green when chord detected
    } else {
        *writer.text(entity, 4) = "None".to_string();
        *writer.color(entity, 4) = TextColor(Color::srgb(0.6, 0.6, 0.6)); // Gray when no chord
    }

    Ok(())
}

/// Toggle the analysis text based on the display mode
pub fn analysis_text_showhide(
    mut q: Query<&mut Visibility, With<AnalysisRoot>>,
    settings: Res<Persistent<SettingsState>>,
) -> Result<()> {
    // TODO: move all showhides into one system that updates the scene based on the display mode
    let mut vis = q.single_mut()?;
    if settings.display_mode == display_system::DisplayMode::Debugging {
        *vis = Visibility::Visible;
    } else {
        *vis = Visibility::Hidden;
    }

    Ok(())
}

/// Setup debug text (bottom left)
pub fn setup_debug_text(mut commands: Commands) {
    let text_font = TextFont {
        font_size: 14.0,
        ..default()
    };

    commands
        .spawn((
            DebugRoot,
            Text::default(),
            text_font.clone(),
            Node {
                position_type: PositionType::Absolute,
                left: Val::Percent(1.),
                bottom: Val::Percent(1.),
                padding: UiRect::all(Val::Px(4.0)),
                ..default()
            },
            TextLayout::new_with_justify(Justify::Left),
            BackgroundColor(Color::BLACK.with_alpha(0.5)),
            ZIndex(i32::MAX),
            Visibility::Visible,
        ))
        .with_children(|builder| {
            // AnalysisParameters section
            builder.spawn((
                TextSpan::new("=== AnalysisParameters ===\n"),
                TextColor(Color::srgb(0.8, 0.8, 1.0)),
                text_font.clone(),
            ));

            // bassline_peak_config
            builder.spawn((
                TextSpan::new("bassline_peak_config:\n"),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new("  prom: X.X, height: X.X\n"),
                TextColor(Color::srgb(0.9, 0.9, 0.9)),
                text_font.clone(),
            ));

            // highest_bassnote
            builder.spawn((
                TextSpan::new("highest_bassnote: "),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new("N/A\n"),
                TextColor(Color::srgb(0.9, 0.9, 0.9)),
                text_font.clone(),
            ));

            // vqt_smoothing_duration_base
            builder.spawn((
                TextSpan::new("vqt_smoothing_dur_base: "),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new("N/A\n"),
                TextColor(Color::srgb(0.9, 0.9, 0.9)),
                text_font.clone(),
            ));

            // vqt_smoothing_calmness min/max
            builder.spawn((
                TextSpan::new("vqt_smooth_calmness: "),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new("N/A\n"),
                TextColor(Color::srgb(0.9, 0.9, 0.9)),
                text_font.clone(),
            ));

            // note_calmness_smoothing_duration
            builder.spawn((
                TextSpan::new("note_calmness_dur: "),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new("N/A\n"),
                TextColor(Color::srgb(0.9, 0.9, 0.9)),
                text_font.clone(),
            ));

            // scene_calmness_smoothing_duration
            builder.spawn((
                TextSpan::new("scene_calmness_dur: "),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new("N/A\n"),
                TextColor(Color::srgb(0.9, 0.9, 0.9)),
                text_font.clone(),
            ));

            // tuning_inaccuracy_smoothing_duration
            builder.spawn((
                TextSpan::new("tuning_inaccuracy_dur: "),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new("N/A\n"),
                TextColor(Color::srgb(0.9, 0.9, 0.9)),
                text_font.clone(),
            ));

            // Separator
            builder.spawn((
                TextSpan::new("------------------------\n"),
                TextColor(Color::srgb(0.5, 0.5, 0.5)),
                text_font.clone(),
            ));

            // AnalysisState section
            builder.spawn((
                TextSpan::new("=== AnalysisState ===\n"),
                TextColor(Color::srgb(1.0, 0.8, 0.8)),
                text_font.clone(),
            ));

            // smoothed_scene_calmness
            builder.spawn((
                TextSpan::new("scene_calmness: "),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new("N/A\n"),
                TextColor(Color::srgb(0.9, 0.9, 0.9)),
                text_font.clone(),
            ));

            // number of detected peaks
            builder.spawn((
                TextSpan::new("detected peaks: "),
                TextColor(Color::WHITE),
                text_font.clone(),
            ));
            builder.spawn((
                TextSpan::new("N/A"),
                TextColor(Color::srgb(0.9, 0.9, 0.9)),
                text_font.clone(),
            ));
        });
}

pub fn update_debug_text_system(
    query: Query<Entity, With<DebugRoot>>,
    analysis: Res<AnalysisStateResource>,
    mut writer: TextUiWriter,
) -> Result<()> {
    let entity = query.single()?;

    // bassline_peak_config (index 3 - parent is 0, children start at 1)
    let bassline_config = &analysis.0.params.bassline_peak_config;
    *writer.text(entity, 3) = format!(
        "  prom: {:.1}, height: {:.1}\n",
        bassline_config.min_prominence, bassline_config.min_height
    );

    // highest_bassnote (index 5)
    *writer.text(entity, 5) = format!("{}\n", analysis.0.params.highest_bassnote);

    // vqt_smoothing_duration_base (index 7)
    let vqt_smooth_ms = analysis.0.params.vqt_smoothing_duration_base.as_millis();
    *writer.text(entity, 7) = if vqt_smooth_ms > 0 {
        format!("{}ms\n", vqt_smooth_ms)
    } else {
        "None\n".to_string()
    };

    // vqt_smoothing_calmness min/max (index 9)
    *writer.text(entity, 9) = format!(
        "{:.2} - {:.2}\n",
        analysis.0.params.vqt_smoothing_calmness_min, analysis.0.params.vqt_smoothing_calmness_max
    );

    // note_calmness_smoothing_duration (index 11)
    *writer.text(entity, 11) = format!(
        "{}ms\n",
        analysis
            .0
            .params
            .note_calmness_smoothing_duration
            .as_millis()
    );

    // scene_calmness_smoothing_duration (index 13)
    *writer.text(entity, 13) = format!(
        "{}ms\n",
        analysis
            .0
            .params
            .scene_calmness_smoothing_duration
            .as_millis()
    );

    // tuning_inaccuracy_smoothing_duration (index 15)
    *writer.text(entity, 15) = format!(
        "{}ms\n",
        analysis
            .0
            .params
            .tuning_inaccuracy_smoothing_duration
            .as_millis()
    );

    // smoothed_scene_calmness (index 19)
    let scene_calmness = analysis.0.smoothed_scene_calmness.get();
    *writer.text(entity, 19) = format!("{:.3}\n", scene_calmness);

    // Color code the calmness value
    *writer.color(entity, 19) = TextColor(if scene_calmness > 0.7 {
        Color::srgb(0.5, 0.8, 1.0) // Cyan for calm
    } else if scene_calmness > 0.3 {
        Color::srgb(1.0, 1.0, 0.5) // Yellow for medium
    } else {
        Color::srgb(1.0, 0.5, 0.5) // Red for energetic
    });

    // number of detected peaks (index 21)
    let num_peaks = analysis.0.peaks.len();
    *writer.text(entity, 21) = format!("{}", num_peaks);

    // Color code based on number of peaks
    *writer.color(entity, 21) = TextColor(if num_peaks == 0 {
        Color::srgb(0.5, 0.5, 0.5) // Gray for no peaks
    } else if num_peaks <= 3 {
        Color::srgb(0.5, 1.0, 0.5) // Green for few peaks
    } else if num_peaks <= 8 {
        Color::srgb(1.0, 1.0, 0.5) // Yellow for moderate
    } else {
        Color::srgb(1.0, 0.5, 0.5) // Red for many peaks
    });

    Ok(())
}

/// Toggle the debug text based on the display mode
pub fn debug_text_showhide(
    mut q: Query<&mut Visibility, With<DebugRoot>>,
    settings: Res<Persistent<SettingsState>>,
) -> Result<()> {
    let mut vis = q.single_mut()?;
    if settings.display_mode == display_system::DisplayMode::Debugging {
        *vis = Visibility::Visible;
    } else {
        *vis = Visibility::Hidden;
    }

    Ok(())
}

/// Setup screen lock indicator
pub fn setup_screen_lock_indicator(mut commands: Commands) {
    let text_font = TextFont {
        font_size: 20.0,
        ..default()
    };

    commands.spawn((
        ScreenLockIndicator,
        Button,
        Text::new("LONG PRESS HERE TO UNLOCK"),
        TextColor(Color::srgb(0.95, 0.95, 0.65)),
        text_font,
        Node {
            position_type: PositionType::Absolute,
            right: Val::Percent(1.),
            top: Val::Percent(1.),
            padding: UiRect::all(Val::Px(8.0)),
            ..default()
        },
        BackgroundColor(Color::BLACK.with_alpha(0.6)),
        BorderColor::all(Color::srgb(0.7, 0.7, 0.5)),
        ZIndex(i32::MAX),
        Visibility::Hidden,
    ));
}

/// Update screen lock indicator visibility
pub fn update_screen_lock_indicator(
    mut query: Query<&mut Visibility, With<ScreenLockIndicator>>,
    lock_state: Res<ScreenLockState>,
    tap_time: Res<ScreenLockTapTime>,
) {
    const INDICATOR_DISPLAY_DURATION: f32 = 3.0;

    if let Ok(mut visibility) = query.single_mut() {
        // Show indicator only if locked AND a tap was registered recently
        let should_show = lock_state.0
            && tap_time
                .0
                .map(|t| t.elapsed().as_secs_f32() < INDICATOR_DISPLAY_DURATION)
                .unwrap_or(false);

        *visibility = if should_show {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

/// Track when screen lock indicator button is being pressed
pub fn handle_screen_lock_indicator_press(
    interaction_query: Query<&Interaction, With<ScreenLockIndicator>>,
    mut indicator_pressed: ResMut<ScreenLockIndicatorPressed>,
    mut mouse_consumed: ResMut<PressEventConsumed>,
) {
    if let Ok(interaction) = interaction_query.single() {
        let is_pressed =
            *interaction == Interaction::Pressed || *interaction == Interaction::Hovered;

        // Update the pressed state
        indicator_pressed.0 = is_pressed;

        // Consume mouse events when interacting with the indicator
        if is_pressed {
            mouse_consumed.0 = true;
        }
    }
}

// /// Marker to find the text entity so we can update it
// #[derive(Component)]
// pub struct BloomSettingsText;

// pub fn setup_bloom_ui(mut commands: Commands) {
//     // UI
//     commands.spawn((
//         BloomSettingsText,
//         Text::new(""),
//         TextColor(Color::WHITE),
//         TextFont {
//             font_size: 16.0,
//             ..default()
//         },
//         Node {
//             position_type: PositionType::Absolute,
//             bottom: Val::Px(12.0),
//             left: Val::Px(12.0),
//             // padding: UiRect::all(Val::Px(4.0)),
//             ..default()
//         },
//         // give it a dark background for readability
//         BackgroundColor(Color::BLACK.with_alpha(0.5)),
//         // make it "always on top" by setting the Z index to maximum
//         // we want it to be displayed over all other UI
//         ZIndex(i32::MAX),
//     ));
// }

// pub fn update_bloom_settings(
//     mut camera: Query<(Entity, Option<&mut Bloom>), With<Camera>>,
//     mut text: Query<(&mut Text, &mut Visibility), With<BloomSettingsText>>,
//     keycode: Res<ButtonInput<KeyCode>>,
//     time: Res<Time>,
//     settings: Res<Persistent<SettingsState>>,
// ) -> Result<()> {
//     let mut text = text.single_mut()?;
//     if settings.display_mode != display_system::DisplayMode::Debugging {
//         *text.1 = Visibility::Hidden;
//         return Ok(());
//     } else {
//         *text.1 = Visibility::Visible;
//     }

//     let bloom_settings = camera.single_mut()?;
//     let text = &mut text.0 .0;

//     if let (_, Some(mut bloom_settings)) = bloom_settings {
//         *text = "BloomSettings\n".to_string();
//         text.push_str(&format!("(Q/A) Intensity: {}\n", bloom_settings.intensity));
//         text.push_str(&format!(
//             "(W/S) Low-frequency boost: {}\n",
//             bloom_settings.low_frequency_boost
//         ));
//         text.push_str(&format!(
//             "(E/D) Low-frequency boost curvature: {}\n",
//             bloom_settings.low_frequency_boost_curvature
//         ));
//         text.push_str(&format!(
//             "(R/F) High-pass frequency: {}\n",
//             bloom_settings.high_pass_frequency
//         ));
//         text.push_str(&format!(
//             "(T/G) Mode: {}\n",
//             match bloom_settings.composite_mode {
//                 BloomCompositeMode::EnergyConserving => "Energy-conserving",
//                 BloomCompositeMode::Additive => "Additive",
//             }
//         ));
//         text.push_str(&format!(
//             "(Y/H) Threshold: {}\n",
//             bloom_settings.prefilter.threshold
//         ));
//         text.push_str(&format!(
//             "(U/J) Threshold softness: {}\n",
//             bloom_settings.prefilter.threshold_softness
//         ));

//         let dt = time.delta_secs();

//         if keycode.pressed(KeyCode::KeyA) {
//             bloom_settings.intensity -= dt / 10.0;
//         }
//         if keycode.pressed(KeyCode::KeyQ) {
//             bloom_settings.intensity += dt / 10.0;
//         }
//         bloom_settings.intensity = bloom_settings.intensity.clamp(0.0, 1.0);

//         if keycode.pressed(KeyCode::KeyS) {
//             bloom_settings.low_frequency_boost -= dt / 10.0;
//         }
//         if keycode.pressed(KeyCode::KeyW) {
//             bloom_settings.low_frequency_boost += dt / 10.0;
//         }
//         bloom_settings.low_frequency_boost = bloom_settings.low_frequency_boost.clamp(0.0, 1.0);

//         if keycode.pressed(KeyCode::KeyD) {
//             bloom_settings.low_frequency_boost_curvature -= dt / 10.0;
//         }
//         if keycode.pressed(KeyCode::KeyE) {
//             bloom_settings.low_frequency_boost_curvature += dt / 10.0;
//         }
//         bloom_settings.low_frequency_boost_curvature =
//             bloom_settings.low_frequency_boost_curvature.clamp(0.0, 1.0);

//         if keycode.pressed(KeyCode::KeyF) {
//             bloom_settings.high_pass_frequency -= dt / 10.0;
//         }
//         if keycode.pressed(KeyCode::KeyR) {
//             bloom_settings.high_pass_frequency += dt / 10.0;
//         }
//         bloom_settings.high_pass_frequency = bloom_settings.high_pass_frequency.clamp(0.0, 1.0);

//         if keycode.pressed(KeyCode::KeyG) {
//             bloom_settings.composite_mode = BloomCompositeMode::Additive;
//         }
//         if keycode.pressed(KeyCode::KeyT) {
//             bloom_settings.composite_mode = BloomCompositeMode::EnergyConserving;
//         }

//         if keycode.pressed(KeyCode::KeyH) {
//             bloom_settings.prefilter.threshold -= dt;
//         }
//         if keycode.pressed(KeyCode::KeyY) {
//             bloom_settings.prefilter.threshold += dt;
//         }
//         bloom_settings.prefilter.threshold = bloom_settings.prefilter.threshold.max(0.0);

//         if keycode.pressed(KeyCode::KeyJ) {
//             bloom_settings.prefilter.threshold_softness -= dt / 10.0;
//         }
//         if keycode.pressed(KeyCode::KeyU) {
//             bloom_settings.prefilter.threshold_softness += dt / 10.0;
//         }
//         bloom_settings.prefilter.threshold_softness =
//             bloom_settings.prefilter.threshold_softness.clamp(0.0, 1.0);
//     }

//     Ok(())
// }

/// Marker to find the container entity so we can show/hide the Buttons (left side)
#[derive(Component)]
pub struct ButtonRoot;

/// Marker to find the container entity so we can show/hide the feature toggle buttons (right side)
#[derive(Component)]
pub struct ButtonRootRight;

#[derive(Component)]
pub enum ButtonAction {
    VisualsMode,
    FpsLimit,
    VQTSmoothing,
    ToggleVibrato,
    ToggleAttackDetection,
    ToggleGlissando,
    ToggleChordRecognition,
    ToggleHarmonicLines,
    ToggleRootNoteTinting,
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
                justify_content: JustifyContent::Default,
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
                    BorderColor::all(Color::srgb(0.0, 0.5, 0.0)),
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
                            display_system::VisualsMode::Performance => "Performance",
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
                    BorderColor::all(Color::srgb(0.0, 0.5, 0.0)),
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
            parent
                .spawn((
                    Button,
                    Node {
                        ..button_node.clone()
                    },
                    BackgroundColor(Color::srgba(0.0, 0.2, 0.0, 0.5)),
                    BorderColor::all(Color::srgb(0.0, 0.5, 0.0)),
                    BorderRadius::MAX,
                    ButtonAction::VQTSmoothing,
                ))
                .insert(ConsumesPressEvents)
                .with_child((
                    Text::new(format!(
                        "VQT Smoothing: {}",
                        match settings.vqt_smoothing_mode {
                            display_system::VQTSmoothingMode::None => "None",
                            display_system::VQTSmoothingMode::Short => "Short",
                            display_system::VQTSmoothingMode::Default => "Default",
                            display_system::VQTSmoothingMode::Long => "Long",
                        }
                    )),
                    TextColor(Color::WHITE),
                    text_font.clone(),
                ));
        });
    commands.insert_resource(PressEventConsumed(false));
}

pub fn setup_feature_toggle_buttons(
    mut commands: Commands,
    settings: Res<Persistent<SettingsState>>,
) {
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

    // Helper function to get button colors based on enabled state
    let button_colors = |enabled: bool| {
        if enabled {
            (
                BackgroundColor(Color::srgba(0.0, 0.2, 0.2, 0.5)),
                BorderColor::all(Color::srgb(0.0, 0.5, 0.5)),
            )
        } else {
            (
                BackgroundColor(Color::srgba(0.2, 0.0, 0.0, 0.5)),
                BorderColor::all(Color::srgb(0.5, 0.0, 0.0)),
            )
        }
    };

    // create our UI root node for right side
    commands
        .spawn((
            ButtonRootRight,
            Node {
                position_type: PositionType::Absolute,
                right: Val::Percent(1.),
                top: Val::Percent(35.),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Default,
                padding: UiRect::all(Val::Px(4.0)),
                margin: UiRect::all(Val::Px(4.0)),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            ZIndex(i32::MAX),
            Visibility::Visible,
        ))
        .with_children(|parent| {
            // Vibrato toggle
            // let (bg_color, border_color) = button_colors(settings.enable_vibrato);
            // parent
            //     .spawn((
            //         Button,
            //         Node {
            //             ..button_node.clone()
            //         },
            //         bg_color,
            //         border_color,
            //         BorderRadius::MAX,
            //         ButtonAction::ToggleVibrato,
            //     ))
            //     .insert(ConsumesPressEvents)
            //     .with_child((
            //         Text::new(format!(
            //             "Vibrato: {}",
            //             if settings.enable_vibrato { "On" } else { "Off" }
            //         )),
            //         TextColor(Color::WHITE),
            //         text_font.clone(),
            //     ));

            // Attack Detection toggle
            // let (bg_color, border_color) = button_colors(settings.enable_attack_detection);
            // parent
            //     .spawn((
            //         Button,
            //         Node {
            //             ..button_node.clone()
            //         },
            //         bg_color,
            //         border_color,
            //         BorderRadius::MAX,
            //         ButtonAction::ToggleAttackDetection,
            //     ))
            //     .insert(ConsumesPressEvents)
            //     .with_child((
            //         Text::new(format!(
            //             "Attack Det.: {}",
            //             if settings.enable_attack_detection {
            //                 "On"
            //             } else {
            //                 "Off"
            //             }
            //         )),
            //         TextColor(Color::WHITE),
            //         text_font.clone(),
            //     ));

            // Glissando toggle
            // let (bg_color, border_color) = button_colors(settings.enable_glissando);
            // parent
            //     .spawn((
            //         Button,
            //         Node {
            //             ..button_node.clone()
            //         },
            //         bg_color,
            //         border_color,
            //         BorderRadius::MAX,
            //         ButtonAction::ToggleGlissando,
            //     ))
            //     .insert(ConsumesPressEvents)
            //     .with_child((
            //         Text::new(format!(
            //             "Glissando: {}",
            //             if settings.enable_glissando {
            //                 "On"
            //             } else {
            //                 "Off"
            //             }
            //         )),
            //         TextColor(Color::WHITE),
            //         text_font.clone(),
            //     ));

            // Chord Recognition toggle
            let (bg_color, border_color) = button_colors(settings.enable_chord_recognition);
            parent
                .spawn((
                    Button,
                    Node {
                        ..button_node.clone()
                    },
                    bg_color,
                    border_color,
                    BorderRadius::MAX,
                    ButtonAction::ToggleChordRecognition,
                ))
                .insert(ConsumesPressEvents)
                .with_child((
                    Text::new(format!(
                        "Chords: {}",
                        if settings.enable_chord_recognition {
                            "On"
                        } else {
                            "Off"
                        }
                    )),
                    TextColor(Color::WHITE),
                    text_font.clone(),
                ));

            // Harmonic Lines toggle
            // let (bg_color, border_color) = button_colors(settings.enable_harmonic_lines);
            // parent
            //     .spawn((
            //         Button,
            //         Node {
            //             ..button_node.clone()
            //         },
            //         bg_color,
            //         border_color,
            //         BorderRadius::MAX,
            //         ButtonAction::ToggleHarmonicLines,
            //     ))
            //     .insert(ConsumesPressEvents)
            //     .with_child((
            //         Text::new(format!(
            //             "Harmonic Lines: {}",
            //             if settings.enable_harmonic_lines {
            //                 "On"
            //             } else {
            //                 "Off"
            //             }
            //         )),
            //         TextColor(Color::WHITE),
            //         text_font.clone(),
            //     ));

            // Root Note Tinting toggle
            let (bg_color, border_color) = button_colors(settings.enable_root_note_tinting);
            parent
                .spawn((
                    Button,
                    Node {
                        ..button_node.clone()
                    },
                    bg_color,
                    border_color,
                    BorderRadius::MAX,
                    ButtonAction::ToggleRootNoteTinting,
                ))
                .insert(ConsumesPressEvents)
                .with_child((
                    Text::new(format!(
                        "Root Tinting: {}",
                        if settings.enable_root_note_tinting {
                            "On"
                        } else {
                            "Off"
                        }
                    )),
                    TextColor(Color::WHITE),
                    text_font.clone(),
                ));
        });
}

pub fn update_button_system(
    interaction_query: Query<
        (Entity, &Interaction, &ButtonAction, &Children),
        (Changed<Interaction>, With<Button>),
    >,
    mut settings: ResMut<Persistent<SettingsState>>,
    mut text_query: Query<&mut Text>,
    mut bg_color_query: Query<&mut BackgroundColor>,
    mut border_color_query: Query<&mut BorderColor>,
    mut mouse_consumed: ResMut<PressEventConsumed>,
) {
    for (entity, interaction, button_action, children) in &interaction_query {
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
                            display_system::VisualsMode::Performance => "Performance",
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
                ButtonAction::VQTSmoothing => {
                    settings
                        .update(|settings| {
                            cycle_vqt_smoothing_mode(&mut settings.vqt_smoothing_mode);
                        })
                        .expect("failed to update settings");
                    **text = format!(
                        "VQT Smoothing: {}",
                        match settings.vqt_smoothing_mode {
                            display_system::VQTSmoothingMode::None => "None",
                            display_system::VQTSmoothingMode::Short => "Short",
                            display_system::VQTSmoothingMode::Default => "Default",
                            display_system::VQTSmoothingMode::Long => "Long",
                        }
                    );
                }
                ButtonAction::ToggleVibrato => {
                    // settings
                    //     .update(|settings| {
                    //         settings.enable_vibrato = !settings.enable_vibrato;
                    //     })
                    //     .expect("failed to update settings");
                    // **text = format!(
                    //     "Vibrato: {}",
                    //     if settings.enable_vibrato { "On" } else { "Off" }
                    // );
                    // update_toggle_button_colors(
                    //     entity,
                    //     settings.enable_vibrato,
                    //     &mut bg_color_query,
                    //     &mut border_color_query,
                    // );
                }
                ButtonAction::ToggleAttackDetection => {
                    // settings
                    //     .update(|settings| {
                    //         settings.enable_attack_detection = !settings.enable_attack_detection;
                    //     })
                    //     .expect("failed to update settings");
                    // **text = format!(
                    //     "Attack Det.: {}",
                    //     if settings.enable_attack_detection {
                    //         "On"
                    //     } else {
                    //         "Off"
                    //     }
                    // );
                    // update_toggle_button_colors(
                    //     entity,
                    //     settings.enable_attack_detection,
                    //     &mut bg_color_query,
                    //     &mut border_color_query,
                    // );
                }
                ButtonAction::ToggleGlissando => {
                    // settings
                    //     .update(|settings| {
                    //         settings.enable_glissando = !settings.enable_glissando;
                    //     })
                    //     .expect("failed to update settings");
                    // **text = format!(
                    //     "Glissando: {}",
                    //     if settings.enable_glissando {
                    //         "On"
                    //     } else {
                    //         "Off"
                    //     }
                    // );
                    // update_toggle_button_colors(
                    //     entity,
                    //     settings.enable_glissando,
                    //     &mut bg_color_query,
                    //     &mut border_color_query,
                    // );
                }
                ButtonAction::ToggleChordRecognition => {
                    settings
                        .update(|settings| {
                            settings.enable_chord_recognition = !settings.enable_chord_recognition;
                        })
                        .expect("failed to update settings");
                    **text = format!(
                        "Chords: {}",
                        if settings.enable_chord_recognition {
                            "On"
                        } else {
                            "Off"
                        }
                    );
                    update_toggle_button_colors(
                        entity,
                        settings.enable_chord_recognition,
                        &mut bg_color_query,
                        &mut border_color_query,
                    );
                }
                ButtonAction::ToggleHarmonicLines => {
                    // settings
                    //     .update(|settings| {
                    //         settings.enable_harmonic_lines = !settings.enable_harmonic_lines;
                    //     })
                    //     .expect("failed to update settings");
                    // **text = format!(
                    //     "Harmonic Lines: {}",
                    //     if settings.enable_harmonic_lines {
                    //         "On"
                    //     } else {
                    //         "Off"
                    //     }
                    // );
                    // update_toggle_button_colors(
                    //     entity,
                    //     settings.enable_harmonic_lines,
                    //     &mut bg_color_query,
                    //     &mut border_color_query,
                    // );
                }
                ButtonAction::ToggleRootNoteTinting => {
                    settings
                        .update(|settings| {
                            settings.enable_root_note_tinting = !settings.enable_root_note_tinting;
                        })
                        .expect("failed to update settings");
                    **text = format!(
                        "Root Tinting: {}",
                        if settings.enable_root_note_tinting {
                            "On"
                        } else {
                            "Off"
                        }
                    );
                    update_toggle_button_colors(
                        entity,
                        settings.enable_root_note_tinting,
                        &mut bg_color_query,
                        &mut border_color_query,
                    );
                }
            }
        }
    }
}

fn update_toggle_button_colors(
    entity: Entity,
    enabled: bool,
    bg_color_query: &mut Query<&mut BackgroundColor>,
    border_color_query: &mut Query<&mut BorderColor>,
) {
    if let Ok(mut bg_color) = bg_color_query.get_mut(entity) {
        *bg_color = if enabled {
            BackgroundColor(Color::srgba(0.0, 0.2, 0.2, 0.5))
        } else {
            BackgroundColor(Color::srgba(0.2, 0.0, 0.0, 0.5))
        };
    }
    if let Ok(mut border_color) = border_color_query.get_mut(entity) {
        *border_color = if enabled {
            BorderColor::all(Color::srgb(0.0, 0.5, 0.5))
        } else {
            BorderColor::all(Color::srgb(0.5, 0.0, 0.0))
        };
    }
}

/// Toggle the left-side buttons based on the display mode
pub fn button_showhide(
    mut q: Query<&mut Visibility, With<ButtonRoot>>,
    settings: Res<Persistent<SettingsState>>,
) -> Result<()> {
    let mut vis = q.single_mut()?;
    if settings.display_mode == display_system::DisplayMode::Debugging {
        *vis = Visibility::Visible;
    } else {
        *vis = Visibility::Hidden;
    }

    Ok(())
}

/// Toggle the right-side feature toggle buttons based on the display mode
pub fn button_showhide_right(
    mut q: Query<&mut Visibility, With<ButtonRootRight>>,
    settings: Res<Persistent<SettingsState>>,
) -> Result<()> {
    let mut vis = q.single_mut()?;
    if settings.display_mode == display_system::DisplayMode::Debugging {
        *vis = Visibility::Visible;
    } else {
        *vis = Visibility::Hidden;
    }

    Ok(())
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
        display_system::VisualsMode::Zen => display_system::VisualsMode::Performance,
        display_system::VisualsMode::Performance => display_system::VisualsMode::Galaxy,
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

fn cycle_vqt_smoothing_mode(mode: &mut display_system::VQTSmoothingMode) {
    *mode = match mode {
        display_system::VQTSmoothingMode::None => display_system::VQTSmoothingMode::Short,
        display_system::VQTSmoothingMode::Short => display_system::VQTSmoothingMode::Default,
        display_system::VQTSmoothingMode::Default => display_system::VQTSmoothingMode::Long,
        display_system::VQTSmoothingMode::Long => display_system::VQTSmoothingMode::None,
    }
}

pub fn user_input_system(
    mut touch_events: MessageReader<TouchInput>,
    mut keyboard_input_events: MessageReader<KeyboardInput>,
    mut mouse_button_input_events: MessageReader<MouseButtonInput>,
    mut settings: ResMut<Persistent<SettingsState>>,
    mouse_consumed: Res<PressEventConsumed>,
    mut lock_state: ResMut<ScreenLockState>,
    mut tap_time: ResMut<ScreenLockTapTime>,
    active_touches: Res<ActiveTouches>,
    indicator_pressed: Res<ScreenLockIndicatorPressed>,
    mut commands: Commands,
) {
    const LONG_PRESS_DURATION: f32 = 1.5;

    // Check for long presses that are still held
    active_touches.0.lock().unwrap().retain(|_, touch_start| {
        if touch_start.elapsed().as_secs_f32() >= LONG_PRESS_DURATION {
            // When unlocked: long press anywhere locks
            // When locked: long press only unlocks if on the indicator button
            if !lock_state.0 {
                // Unlocked - lock the screen
                lock_state.0 = true;
            } else if indicator_pressed.0 {
                // Locked and pressing indicator - unlock the screen
                lock_state.0 = false;
            }
            false // Remove from tracking
        } else {
            true // Keep tracking
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
                // Check if this was a short press
                let touch_start = active_touches.0.lock().unwrap().remove(&touch.id);
                if let Some(touch_start) = touch_start {
                    if touch_start.elapsed().as_secs_f32() < LONG_PRESS_DURATION {
                        // Short press - only toggle display mode if not locked and not consumed
                        if !lock_state.0 && !mouse_consumed.0 {
                            settings
                                .update(|settings| {
                                    cycle_display_mode(&mut settings.display_mode);
                                })
                                .expect("failed to update settings");
                        } else if lock_state.0 {
                            // Record tap time when screen is locked
                            tap_time.0 = Some(Instant::now());
                        }
                    }
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
                    // Only allow space to toggle display mode if not locked
                    if !lock_state.0 {
                        settings
                            .update(|settings| {
                                cycle_display_mode(&mut settings.display_mode);
                            })
                            .expect("failed to update settings");
                    }
                }
                KeyCode::KeyL => {
                    // Ctrl+L or just L to toggle lock on desktop
                    lock_state.0 = !lock_state.0;
                }
                _ => {}
            }
        }
    }

    for mouse_button_input in mouse_button_input_events.read() {
        if mouse_button_input.state.is_pressed() {
            match mouse_button_input.button {
                MouseButton::Left => {
                    // Only allow left click to toggle display mode if not locked and not consumed
                    if !lock_state.0 && !mouse_consumed.0 {
                        settings
                            .update(|settings| {
                                cycle_display_mode(&mut settings.display_mode);
                            })
                            .expect("failed to update settings");
                    } else if lock_state.0 {
                        // Record tap time when screen is locked
                        tap_time.0 = Some(Instant::now());
                    }
                    // reset the consumed state for the next frame
                    commands.insert_resource(PressEventConsumed(false));
                }
                MouseButton::Right => {
                    // Right click to toggle lock on desktop
                    lock_state.0 = !lock_state.0;
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

// ============================================================================
// Common App Builder Functions
// ============================================================================

/// Configuration for platform-specific app setup
pub struct PlatformConfig {
    /// Directory for storing configuration files
    pub config_dir: PathBuf,
    /// Storage format for persistent settings (Toml or Json)
    pub storage_format: StorageFormat,
    /// Audio buffer ring buffer (platform-specific audio implementation)
    pub audio_buffer: Arc<Mutex<RingBuffer>>,
    /// Window configuration (platform-specific)
    pub window_config: Window,
    /// Platform-specific additional plugins
    pub additional_plugins: Vec<Box<dyn Plugin>>,
}

/// Creates persistent settings with platform-specific configuration
pub fn create_persistent_settings(
    config_dir: PathBuf,
    format: StorageFormat,
    filename: &str,
    default_fps: u32,
) -> Persistent<SettingsState> {
    let mut persistent_settings_state = Persistent::<SettingsState>::builder()
        .name("settings")
        .format(format)
        .path(config_dir.join(filename))
        .default(SettingsState {
            display_mode: display_system::DisplayMode::Normal,
            visuals_mode: display_system::VisualsMode::Full,
            fps_limit: Some(default_fps),
            vqt_smoothing_mode: display_system::VQTSmoothingMode::Default,
            enable_vibrato: false,
            enable_attack_detection: false,
            enable_glissando: false,
            enable_chord_recognition: true,
            enable_harmonic_lines: false,
            enable_root_note_tinting: false,
        })
        .revertible(true)
        .revert_to_default_on_deserialization_errors(true)
        .build()
        .expect("failed to initialize settings");

    // Always start in normal mode
    persistent_settings_state.display_mode = display_system::DisplayMode::Normal;
    persistent_settings_state
}

/// Inserts all common resources into the app
pub fn insert_common_resources(
    app: &mut App,
    vqt: Vqt,
    vqt_range: &VqtRange,
    audio_buffer: Arc<Mutex<RingBuffer>>,
    settings: Persistent<SettingsState>,
    default_fps: u32,
) {
    app.insert_resource(VqtResource(vqt))
        .insert_resource(VqtResultResource::new(vqt_range))
        .insert_resource(AudioBufferResource(audio_buffer))
        .insert_resource(AnalysisStateResource(AnalysisState::new(
            vqt_range.clone(),
            AnalysisParameters::default(),
        )))
        .insert_resource(display_system::CylinderEntityListResource(Vec::new()))
        .insert_resource(display_system::GlissandoCurveEntityListResource(Vec::new()))
        .insert_resource(settings)
        .insert_resource(CurrentFpsLimit(Some(default_fps)))
        .insert_resource(CurrentVQTSmoothingMode(
            display_system::VQTSmoothingMode::Default,
        ))
        .insert_resource(WinitSettings {
            focused_mode: UpdateMode::reactive(std::time::Duration::from_secs_f32(
                1.0 / default_fps as f32,
            )),
            unfocused_mode: UpdateMode::reactive(std::time::Duration::from_secs_f32(
                1.0 / default_fps as f32,
            )),
        })
        .insert_resource(ScreenLockState(false))
        .insert_resource(ScreenLockTapTime(None))
        .insert_resource(ScreenLockIndicatorPressed(false))
        .insert_resource(ActiveTouches(Arc::new(Mutex::new(HashMap::new()))));
}

/// Registers all common startup systems
pub fn register_startup_systems(app: &mut App, vqt_range: &VqtRange) {
    app.add_systems(
        Startup,
        (
            display_system::setup_display_to_system(vqt_range),
            setup_fps_counter,
            setup_buttons,
            setup_feature_toggle_buttons,
            // setup_bloom_ui,
            setup_analysis_text,
            setup_debug_text,
            setup_screen_lock_indicator,
        ),
    );
}

/// Registers all common update systems
pub fn register_common_update_systems<M1, M2, M3>(
    app: &mut App,
    update_vqt_system: impl IntoSystem<(), (), M1>,
    update_analysis_state_system: impl IntoSystem<(), (), M2>,
    update_display_system: impl IntoSystem<(), (), M3>,
) {
    app.add_systems(
        Update,
        (
            close_on_esc,
            update_vqt_system,
            update_analysis_state_system,
            update_fps_text_system,
            fps_counter_showhide,
            update_button_system,
            button_showhide,
            button_showhide_right,
            user_input_system.after(update_button_system),
            update_analysis_text_system,
            analysis_text_showhide,
            update_debug_text_system,
            debug_text_showhide,
            update_screen_lock_indicator,
            handle_screen_lock_indicator_press,
            set_frame_limiter_system,
            set_vqt_smoothing_system,
            update_display_system,
            display_system::update::update_spectrogram_system,
            display_system::update::spectrogram_showhide,
            display_system::update::update_chroma_system,
            display_system::update::chroma_showhide,
        ),
    );
}

/// Builds a complete app with common configuration and platform-specific components
pub fn build_common_app(
    vqt: Vqt,
    vqt_parameters: &VqtParameters,
    config: PlatformConfig,
    default_fps: u32,
) -> App {
    let settings_filename = match config.storage_format {
        StorageFormat::Toml => "settings.toml",
        StorageFormat::Json => "settings.json",
        _ => "settings.toml",
    };

    let persistent_settings = create_persistent_settings(
        config.config_dir,
        config.storage_format,
        settings_filename,
        default_fps,
    );

    let mut app = App::new();

    // Add default plugins with platform-specific window configuration
    app.add_plugins((
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(config.window_config),
                ..default()
            })
            .set(AssetPlugin {
                meta_check: AssetMetaCheck::Never,
                ..default()
            }),
        FrameTimeDiagnosticsPlugin::default(),
        Material2dPlugin::<display_system::material::NoisyColorMaterial>::default(),
    ));

    // Add platform-specific plugins
    for plugin in config.additional_plugins {
        plugin.build(&mut app);
    }

    // Insert common resources
    insert_common_resources(
        &mut app,
        vqt,
        &vqt_parameters.range,
        config.audio_buffer,
        persistent_settings,
        default_fps,
    );

    app
}
