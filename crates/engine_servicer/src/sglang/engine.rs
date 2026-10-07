//! Connecting the headless scheduler(s), in the background of a server that
//! is already listening. Failures gate health and surface as the last error;
//! nothing here retries, the lifecycle owner restarts the pair.

use std::{sync::Arc, time::Duration};

use engine_zmq_adapter::{connect_with_eos, EosTokenIds, Handshake};
use openai_protocol::worker::RuntimeType;
use tracing::{error, info};

use super::State;
use crate::{record_error, SharedError};

pub(super) async fn connect_engine(
    state: Arc<State>,
    ipc_base_url: String,
    handshake_address: String,
    engine_count: usize,
    startup_timeout: Duration,
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
        startup_timeout,
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
            let message = format!("SGLang scheduler connection failed: {connect_error}");
            error!(%message);
            state.engine.fail(message.clone());
            record_error(&last_error, message);
        }
    }
}
