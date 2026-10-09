//! The link to the engine: `None` while the ZMQ handshake runs, then the
//! client or the reason it failed. Health is gated on it.

use std::{sync::Arc, time::Duration};

use engine_zmq_adapter::{
    connect_with_eos, structured_outputs_backend_from_config, EosTokenIds, Handshake,
};
use engine_zmq_client::StartupBudget;
use openai_protocol::worker::RuntimeType;
use tracing::{error, info, warn};

use super::State;
use crate::{record_error, SharedError};

/// Load the tokenizer and connect the engine, in the background of a server
/// that is already listening. Failures gate health and surface as the last
/// error; nothing here retries, the lifecycle owner restarts the pair.
///
/// The handshake's silence bound (`startup_timeout`) counts from the latest
/// sign of life: a handshake message, or the lifecycle owner's report that
/// the engine process is alive (`State::engine::liveness`); the ceiling
/// bounds the whole start regardless.
#[expect(
    clippy::too_many_arguments,
    reason = "the link's inputs: endpoints, engine count, the two startup bounds, the tokenizer and the error slot"
)]
pub(super) async fn connect_engine(
    state: Arc<State>,
    ipc_base_url: String,
    handshake_address: String,
    engine_count: usize,
    startup_timeout: Duration,
    startup_ceiling: Option<Duration>,
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
        Handshake::TcpOrIpc(&handshake_address),
        engine_count,
        eos,
        StartupBudget {
            silence: startup_timeout,
            ceiling: startup_ceiling,
            liveness: Some(state.engine.liveness.clone()),
        },
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
            let ceiling = startup_ceiling
                .map_or_else(|| "none".to_string(), |ceiling| format!("{ceiling:?}"));
            let message = format!(
                "vLLM engine connection failed: {connect_error} (startup bounds: \
                 {startup_timeout:?} without a sign of life from the engine, ceiling {ceiling})"
            );
            error!(%message);
            state.engine.fail(message.clone());
            record_error(&last_error, message);
        }
    }
}
