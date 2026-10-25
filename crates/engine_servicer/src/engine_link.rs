//! The link to the engine: `None` while the ZMQ handshake runs, then the
//! client or the reason it failed. Health is gated on it.

use std::{
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use engine_zmq_adapter::{connect_with_eos, EosTokenIds, Handshake, WireRevision, ZmqEngineClient};
use engine_zmq_client::{EngineLiveness, StartupBudget};
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::worker::RuntimeType;
use tracing::{error, info, warn};

use crate::{record_error, servicer_config::SilenceBounds, SharedError};

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

/// The words a wire's connect says of its peer: what sits at the other end
/// of the link, as that engine's own documents name it (the connected log
/// line and the failure message start with it), and the party the start-up
/// bound waits on (what the failure text says it heard nothing from).
pub(crate) struct PeerWords {
    pub(crate) peer: &'static str,
    pub(crate) party: &'static str,
}

/// The inputs of one connect, as the server's start hands them to the link.
pub(crate) struct ConnectInputs {
    pub(crate) ipc_base_url: String,
    pub(crate) handshake_address: String,
    pub(crate) engine_count: usize,
    pub(crate) startup_timeout: Duration,
    pub(crate) startup_ceiling: Option<Duration>,
    pub(crate) silence: SilenceBounds,
    pub(crate) last_error: SharedError,
}

/// Connect the engine behind `link`, in the background of a server that is
/// already listening. Failures gate health and surface as the last error;
/// nothing here retries, the lifecycle owner restarts the pair.
///
/// The handshake's silence bound (`startup_timeout`) counts from the latest
/// sign of life: a handshake message, or the lifecycle owner's report that
/// the engine process is alive (`link.liveness`); the ceiling bounds the
/// whole start regardless. `wire` is the wire revision the launcher read of
/// the engine before it started, for a wire with more than one framing (the
/// others pass the default and ignore it); the connection speaks it from
/// its first frame. `configure` runs on the connected client first, with
/// the wire's own setup (its structured-outputs support, the tokenizer's
/// EOS, what it knows of the engine at start); then the engine's silence is
/// bounded by `silence` (with the prefill allowance for the prompt it
/// holds), applied before the first request reaches it, and the client is
/// published.
#[expect(
    clippy::too_many_arguments,
    reason = "one connect: the link, the model, the EOS ids, the wire and its revision, the words, the inputs and the setup"
)]
pub(crate) async fn connect_engine(
    link: &EngineLink,
    model_path: String,
    eos_token_ids: &[u32],
    runtime: RuntimeType,
    wire: WireRevision,
    words: PeerWords,
    inputs: ConnectInputs,
    configure: impl FnOnce(&ZmqEngineClient),
) {
    let ConnectInputs {
        ipc_base_url,
        handshake_address,
        engine_count,
        startup_timeout,
        startup_ceiling,
        silence,
        last_error,
    } = inputs;
    // The engine stops at EOS itself; the ids only inform the adapter.
    let eos = if eos_token_ids.is_empty() {
        EosTokenIds::default()
    } else {
        EosTokenIds::from_ids(eos_token_ids)
    };
    match connect_with_eos(
        &ipc_base_url,
        model_path,
        runtime,
        Handshake::TcpOrIpc(&handshake_address),
        engine_count,
        eos,
        StartupBudget {
            silence: startup_timeout,
            ceiling: startup_ceiling,
            liveness: Some(link.liveness.clone()),
        },
        wire,
    )
    .await
    {
        Ok(client) => {
            configure(&client);
            silence.apply(&client);
            info!(
                handshake = %handshake_address,
                engines = engine_count,
                silence_window = ?silence.window,
                prefill_tokens_per_sec = silence.prefill_tokens_per_sec,
                "{} connected",
                words.peer
            );
            let _ = link.client.set(client);
        }
        Err(connect_error) => {
            let ceiling = startup_ceiling
                .map_or_else(|| "none".to_string(), |ceiling| format!("{ceiling:?}"));
            let message = format!(
                "{} connection failed: {connect_error} (startup bounds: {startup_timeout:?} \
                 without a sign of life from the {}, ceiling {ceiling})",
                words.peer, words.party
            );
            error!(%message);
            link.fail(message.clone());
            record_error(&last_error, message);
        }
    }
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
