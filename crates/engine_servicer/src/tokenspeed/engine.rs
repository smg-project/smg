//! Loading the tokenizer and connecting the headless scheduler(s), in the
//! background of a server that is already listening. Failures gate health
//! and surface as the last error; nothing here retries, the lifecycle owner
//! restarts the pair.

use std::sync::Arc;

use openai_protocol::worker::RuntimeType;

use super::State;
use crate::engine_link::{self, load_tokenizer_once, ConnectInputs, PeerWords};

/// This wire's connect: the tokenizer is loaded first (string stops need
/// it), then [`engine_link::connect_engine`] with this wire's words and the
/// wire revision the launcher read; the connected scheduler takes the
/// tokenizer's EOS and the structured-outputs support the launcher's
/// arguments say. The start-up and silence bounds are the link's.
pub(super) async fn connect_engine(state: Arc<State>, inputs: ConnectInputs) {
    load_tokenizer_once(&state.tokenizer, state.shared.tokenizer_dir.as_deref()).await;

    engine_link::connect_engine(
        &state.shared.engine,
        state.model.model_path.clone(),
        &state.model.eos_token_ids,
        RuntimeType::TokenSpeed,
        state.model.wire_revision,
        PeerWords {
            peer: "TokenSpeed scheduler",
            party: "scheduler",
        },
        inputs,
        |client| {
            client.adopt_tokenizer_eos(state.tokenizer());
            client.set_structured_outputs_support(super::structured_outputs_support(&state.model));
        },
    )
    .await;
}
