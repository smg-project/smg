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

    /// The engine's silence on its in-flight requests, for the pushed load
    /// record (`EngineLoad.engine_silence_ms`): `None` while the handshake
    /// runs, while nothing is in flight, or right after an output.
    pub(crate) fn output_silence_ms(&self) -> Option<u32> {
        let silence = self.client.get()?.output_silence()?;
        Some(u32::try_from(silence.as_millis()).unwrap_or(u32::MAX))
    }

    /// Whether the engine's wire has shown a scheduler step without any
    /// token (`EngineLoad.engine_reports_steps`): `None` while the handshake
    /// runs.
    pub(crate) fn reports_steps(&self) -> Option<bool> {
        Some(self.client.get()?.reports_steps())
    }

    /// Prompt tokens handed to the engine with no output yet
    /// (`EngineLoad.prefill_pending_tokens`): `None` while the handshake runs.
    pub(crate) fn prefill_pending_tokens(&self) -> Option<u32> {
        let tokens = self.client.get()?.prefill_pending_tokens();
        Some(u32::try_from(tokens).unwrap_or(u32::MAX))
    }
}
