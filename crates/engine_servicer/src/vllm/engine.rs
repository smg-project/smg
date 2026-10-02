//! The link to the engine: `None` while the ZMQ handshake runs, then the
//! client or the reason it failed. Health is gated on it.

use std::sync::{Arc, Mutex, OnceLock};

use engine_zmq_adapter::{
    connect_with_eos, structured_outputs_backend_from_config, EosTokenIds, ZmqEngineClient,
};
use openai_protocol::worker::RuntimeType;
use tracing::{error, info, warn};

use super::State;
use crate::{record_error, SharedError};

/// The engine link: unset while the handshake runs, then the client or the
/// reason it failed. Health is gated on it.
#[derive(Default)]
pub(crate) struct EngineLink {
    pub(crate) client: OnceLock<ZmqEngineClient>,
    pub(crate) error: Mutex<Option<String>>,
}

impl EngineLink {
    pub(super) fn error(&self) -> Option<String> {
        self.error.lock().ok().and_then(|slot| slot.clone())
    }
}

/// Load the tokenizer and connect the engine, in the background of a server
/// that is already listening. Failures gate health and surface as the last
/// error; nothing here retries, the lifecycle owner restarts the pair.
pub(super) async fn connect_engine(
    state: Arc<State>,
    ipc_base_url: String,
    handshake_address: String,
    engine_count: usize,
    tokenizer_dir: Option<String>,
    last_error: SharedError,
) {
    let tokenizer = match tokenizer_dir.as_deref() {
        _ if state.tokenizer.get().is_some() => None,
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
    // A pre-seeded tokenizer wins; `set` is a no-op then.
    let _ = state.tokenizer.set(tokenizer);

    let eos = if state.model.eos_token_ids.is_empty() {
        EosTokenIds::default()
    } else {
        EosTokenIds::from_ids(&state.model.eos_token_ids)
    };
    match connect_with_eos(
        &ipc_base_url,
        state.model.model_path.clone(),
        RuntimeType::Vllm,
        Some(&handshake_address),
        engine_count,
        eos,
    )
    .await
    {
        Ok(client) => {
            client.adopt_tokenizer_eos(state.tokenizer());
            client.set_structured_outputs_backend(structured_outputs_backend_from_config(
                &state.model.structured_outputs_backend,
            ));
            info!(
                handshake = %handshake_address,
                engines = engine_count,
                "vLLM engine connected"
            );
            let _ = state.engine.client.set(client);
        }
        Err(connect_error) => {
            let message = format!("vLLM engine connection failed: {connect_error}");
            error!(%message);
            if let Ok(mut slot) = state.engine.error.lock() {
                *slot = Some(message.clone());
            }
            record_error(&last_error, message);
        }
    }
}
