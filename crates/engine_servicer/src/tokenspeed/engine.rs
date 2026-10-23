//! Loading the tokenizer and connecting the headless scheduler(s), in the
//! background of a server that is already listening. Failures gate health
//! and surface as the last error; nothing here retries, the lifecycle owner
//! restarts the pair.

use std::{sync::Arc, time::Duration};

use openai_protocol::worker::RuntimeType;

use super::State;
use crate::{
    engine_link::{self, load_tokenizer_once, ConnectInputs, PeerWords},
    servicer_config::SilenceBounds,
    SharedError,
};

/// This wire's connect: the tokenizer is loaded first (string stops need
/// it), then [`engine_link::connect_engine`] with this wire's words; the
/// connected scheduler takes the tokenizer's EOS and the structured-outputs
/// support the launcher's arguments say. The start-up and silence bounds are
/// the link's.
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
        RuntimeType::TokenSpeed,
        PeerWords {
            peer: "TokenSpeed scheduler",
            party: "scheduler",
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
            client.set_structured_outputs_support(super::structured_outputs_support(&state.model));
        },
    )
    .await;
}
