//! The link to the engine: `None` while the ZMQ handshake runs, then the
//! client or the reason it failed. Health is gated on it.

use std::sync::{Mutex, OnceLock};

use engine_zmq_adapter::ZmqEngineClient;
use engine_zmq_client::EngineLiveness;

/// The engine link: unset while the handshake runs, then the client or the
/// reason it failed.
#[derive(Default)]
pub(crate) struct EngineLink {
    pub(crate) client: OnceLock<ZmqEngineClient>,
    pub(crate) error: Mutex<Option<String>>,
    /// Signs of life the lifecycle owner reports while the handshake runs
    /// (it polls the engine process it launched); they extend the handshake's
    /// silence bound.
    pub(crate) liveness: EngineLiveness,
}

impl EngineLink {
    pub(crate) fn error(&self) -> Option<String> {
        self.error.lock().ok().and_then(|slot| slot.clone())
    }

    /// Record why the connect failed; the first reason wins.
    pub(crate) fn fail(&self, message: String) {
        if let Ok(mut slot) = self.error.lock() {
            slot.get_or_insert(message);
        }
    }
}
