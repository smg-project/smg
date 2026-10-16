//! The link to the engine: `None` while the ZMQ handshake runs, then the
//! client or the reason it failed. Health is gated on it.

use std::sync::{Arc, Mutex, OnceLock};

use engine_zmq_adapter::ZmqEngineClient;
use engine_zmq_client::EngineLiveness;
use llm_tokenizer::traits::Tokenizer;
use tracing::warn;

/// A servicer's tokenizer, set once: by the lifecycle owner that already
/// loaded it, else by [`load_tokenizer_once`]; `None` inside when there is
/// none to load, and string stops are refused.
pub(crate) type TokenizerSlot = OnceLock<Option<Arc<dyn Tokenizer>>>;

/// Load the tokenizer from `tokenizer_dir` into `slot`, unless the slot is
/// already set (a pre-seeded tokenizer wins; `set` is a no-op then). No
/// directory, or a load that fails, leaves `None` in the slot and says once
/// that string stops will be refused.
pub(crate) async fn load_tokenizer_once(slot: &TokenizerSlot, tokenizer_dir: Option<&str>) {
    let tokenizer = match tokenizer_dir {
        _ if slot.get().is_some() => None,
        Some(dir) => match llm_tokenizer::factory::create_tokenizer_async(dir).await {
            Ok(tokenizer) => Some(tokenizer),
            Err(load_error) => {
                warn!(
                    %dir,
                    %load_error,
                    "tokenizer load failed; string stops will be refused"
                );
                None
            }
        },
        None => {
            warn!("no tokenizer directory configured; string stops will be refused");
            None
        }
    };
    let _ = slot.set(tokenizer);
}

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

    /// The engine's silence on the in-flight requests it reports every
    /// step, for the pushed load record (`EngineLoad.engine_silence_ms`):
    /// `None` while the handshake runs, while no such request is in flight,
    /// or right after an output.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn without_a_directory_the_slot_holds_none() {
        let slot = TokenizerSlot::new();
        load_tokenizer_once(&slot, None).await;
        assert!(matches!(slot.get(), Some(None)));
    }

    #[tokio::test]
    async fn a_directory_that_does_not_load_leaves_none() {
        let dir = tempfile::tempdir().unwrap();
        let slot = TokenizerSlot::new();
        load_tokenizer_once(&slot, Some(dir.path().to_str().unwrap())).await;
        assert!(matches!(slot.get(), Some(None)));
    }

    #[tokio::test]
    async fn a_slot_already_set_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let slot = TokenizerSlot::new();
        let _ = slot.set(None);
        // Nothing is loaded for a set slot (the directory is not read).
        load_tokenizer_once(&slot, Some(dir.path().to_str().unwrap())).await;
        assert!(matches!(slot.get(), Some(None)));
    }
}
