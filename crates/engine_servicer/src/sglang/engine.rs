//! Connecting the headless scheduler(s), in the background of a server that
//! is already listening. Failures gate health and surface as the last error;
//! nothing here retries, the lifecycle owner restarts the pair.

use std::{sync::Arc, time::Duration};

use openai_protocol::worker::RuntimeType;

use super::State;
use crate::{
    engine_link::{self, ConnectInputs, PeerWords},
    servicer_config::SilenceBounds,
    SharedError,
};

/// This wire's connect ([`engine_link::connect_engine`] with its words): the
/// connected scheduler takes the structured-outputs support the launcher's
/// arguments say and the note that it truncates long prompts, when they say
/// so. The start-up and silence bounds are the link's.
#[expect(
    clippy::too_many_arguments,
    reason = "the link's inputs: endpoints, engine count, the startup and silence bounds and the error slot"
)]
pub(super) async fn connect_engine(
    state: Arc<State>,
    ipc_base_url: String,
    handshake_address: String,
    engine_count: usize,
    startup_timeout: Duration,
    startup_ceiling: Option<Duration>,
    silence: SilenceBounds,
    last_error: SharedError,
) {
    engine_link::connect_engine(
        &state.shared.engine,
        state.model.model_path.clone(),
        &state.model.eos_token_ids,
        RuntimeType::Sglang,
        PeerWords {
            peer: "SGLang scheduler",
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
            client.set_structured_outputs_support(super::structured_outputs_support(&state.model));
            if super::truncates_long_prompts(&state.model) {
                client.note_engine_truncates_long_prompts();
            }
        },
    )
    .await;
}
