//! Connecting the headless scheduler(s), in the background of a server that
//! is already listening. Failures gate health and surface as the last error;
//! nothing here retries, the lifecycle owner restarts the pair.

use std::{sync::Arc, time::Duration};

use engine_zmq_adapter::{connect_with_eos, EosTokenIds, Handshake};
use engine_zmq_client::StartupBudget;
use openai_protocol::worker::RuntimeType;
use tracing::{error, info};

use super::State;
use crate::{record_error, SharedError};

/// The handshake's silence bound (`startup_timeout`) counts from the latest
/// sign of life: a handshake message, or the lifecycle owner's report that
/// the scheduler process is alive (`State::engine::liveness`); the ceiling
/// bounds the whole start regardless.
pub(super) async fn connect_engine(
    state: Arc<State>,
    ipc_base_url: String,
    handshake_address: String,
    engine_count: usize,
    startup_timeout: Duration,
    startup_ceiling: Option<Duration>,
    last_error: SharedError,
) {
    // The scheduler stops at EOS itself; the ids only inform the adapter.
    let eos = if state.model.eos_token_ids.is_empty() {
        EosTokenIds::default()
    } else {
        EosTokenIds::from_ids(&state.model.eos_token_ids)
    };
    match connect_with_eos(
        &ipc_base_url,
        state.model.model_path.clone(),
        RuntimeType::Sglang,
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
            info!(
                handshake = %handshake_address,
                engines = engine_count,
                "SGLang scheduler connected"
            );
            let _ = state.engine.client.set(client);
        }
        Err(connect_error) => {
            let ceiling = startup_ceiling
                .map_or_else(|| "none".to_string(), |ceiling| format!("{ceiling:?}"));
            let message = format!(
                "SGLang scheduler connection failed: {connect_error} (startup bounds: \
                 {startup_timeout:?} without a sign of life from the scheduler, ceiling {ceiling})"
            );
            error!(%message);
            state.engine.fail(message.clone());
            record_error(&last_error, message);
        }
    }
}
