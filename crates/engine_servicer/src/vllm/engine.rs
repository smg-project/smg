//! The link to the engine: `None` while the ZMQ handshake runs, then the
//! client or the reason it failed. Health is gated on it.

use std::{sync::Arc, time::Duration};

use engine_zmq_adapter::structured_outputs_backend_from_config;
use openai_protocol::worker::RuntimeType;

use super::State;
use crate::{
    engine_link::{self, load_tokenizer_once, ConnectInputs, PeerWords},
    servicer_config::SilenceBounds,
    SharedError,
};

/// This wire's connect: the tokenizer is loaded first (string stops need
/// it), then [`engine_link::connect_engine`] with this wire's words; the
/// connected engine takes the tokenizer's EOS, the structured-outputs
/// backend its config names and, when its statistics are on, the note that
/// it reports every scheduler step. The start-up and silence bounds are the
/// link's.
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

    engine_link::connect_engine(
        &state.shared.engine,
        state.model.model_path.clone(),
        &state.model.eos_token_ids,
        RuntimeType::Vllm,
        PeerWords {
            peer: "vLLM engine",
            party: "engine",
        },
        ConnectInputs {
            ipc_base_url,
            handshake_address,
            engine_count,
            startup_timeout,
            startup_ceiling,
            silence,
            last_error,
        },
        |client| {
            client.adopt_tokenizer_eos(state.tokenizer());
            client.set_structured_outputs_support(
                structured_outputs_backend_from_config(&state.model.structured_outputs_backend)
                    .into(),
            );
            // Known at start: the wire reports every scheduler step while the
            // engine's statistics are on, before its first batch shows it.
            if state.model.log_stats == Some(true) {
                client.note_reports_steps();
            }
        },
    )
    .await;
}
