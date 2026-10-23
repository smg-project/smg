//! Loading the tokenizer and connecting the headless scheduler(s), in the
//! background of a server that is already listening. Failures gate health
//! and surface as the last error; nothing here retries, the lifecycle owner
//! restarts the pair.

use std::{sync::Arc, time::Duration};

use engine_zmq_adapter::{connect_with_eos, EosTokenIds, Handshake};
use engine_zmq_client::StartupBudget;
use openai_protocol::worker::RuntimeType;
use tracing::{error, info};

use super::State;
use crate::{
    engine_link::load_tokenizer_once, record_error, servicer_config::SilenceBounds, SharedError,
};

/// The handshake's silence bound (`startup_timeout`) counts from the latest
/// sign of life: a handshake message, or the lifecycle owner's report that
/// the scheduler process is alive (`State::engine::liveness`); the ceiling
/// bounds the whole start regardless. Once connected, the scheduler's silence
/// is bounded by `silence` (with the prefill allowance for the prompt it
/// holds), applied before the first request reaches it.
#[expect(
    clippy::too_many_arguments,
    reason = "the link's inputs: endpoints, engine count, the startup and silence bounds, the tokenizer and the error slot"
)]
pub(super) async fn connect_engine(
    state: Arc<State>,
    ipc_base_url: String,
    handshake_address: String,
    engine_count: usize,
    startup_timeout: Duration,
    startup_ceiling: Option<Duration>,
    silence: SilenceBounds,
    tokenizer_dir: Option<String>,
    last_error: SharedError,
) {
    load_tokenizer_once(&state.tokenizer, tokenizer_dir.as_deref()).await;

    let eos = if state.model.eos_token_ids.is_empty() {
        EosTokenIds::default()
    } else {
        EosTokenIds::from_ids(&state.model.eos_token_ids)
    };
    match connect_with_eos(
        &ipc_base_url,
        state.model.model_path.clone(),
        RuntimeType::TokenSpeed,
        Handshake::TcpOrIpc(&handshake_address),
        engine_count,
        eos,
        StartupBudget {
            silence: startup_timeout,
            ceiling: startup_ceiling,
            liveness: Some(state.shared.engine.liveness.clone()),
        },
    )
    .await
    {
        Ok(client) => {
            client.adopt_tokenizer_eos(state.tokenizer());
            client.set_structured_outputs_support(super::structured_outputs_support(&state.model));
            silence.apply(&client);
            info!(
                handshake = %handshake_address,
                engines = engine_count,
                silence_window = ?silence.window,
                prefill_tokens_per_sec = silence.prefill_tokens_per_sec,
                "TokenSpeed scheduler connected"
            );
            let _ = state.shared.engine.client.set(client);
        }
        Err(connect_error) => {
            let ceiling = startup_ceiling
                .map_or_else(|| "none".to_string(), |ceiling| format!("{ceiling:?}"));
            let message = format!(
                "TokenSpeed scheduler connection failed: {connect_error} (startup bounds: \
                 {startup_timeout:?} without a sign of life from the scheduler, ceiling {ceiling})"
            );
            error!(%message);
            state.shared.engine.fail(message.clone());
            record_error(&last_error, message);
        }
    }
}
