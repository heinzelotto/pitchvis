use crate::audio::RingBuffer;
use bevy::prelude::*;

#[derive(Resource)]
pub struct AudioBufferResource(pub std::sync::Arc<std::sync::Mutex<RingBuffer>>);
