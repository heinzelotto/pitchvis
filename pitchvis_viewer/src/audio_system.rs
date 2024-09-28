use bevy::prelude::*;
use pitchvis_audio::RingBuffer;

#[derive(Resource)]
pub struct AudioBufferResource(pub std::sync::Arc<std::sync::Mutex<RingBuffer>>);
