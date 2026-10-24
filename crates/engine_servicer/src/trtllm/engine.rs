//! Connecting the shim(s), in the background of a server that is already
//! listening. Failures gate health and surface as the last error; nothing
//! here retries, the lifecycle owner restarts the pair.

use std::sync::Arc;

use openai_protocol::worker::RuntimeType;

use super::State;
use crate::engine_link::{self, ConnectInputs, PeerWords};

/// This wire's connect ([`engine_link::connect_engine`] with its words): the
/// connected engine takes the structured-outputs support the launcher's
/// facts say. The start-up and silence bounds are the link's.
pub(super) async fn connect_engine(state: Arc<State>, inputs: ConnectInputs) {
    engine_link::connect_engine(
        &state.shared.engine,
        state.model.model_path.clone(),
        &state.model.eos_token_ids,
        RuntimeType::Trtllm,
        PeerWords {
            peer: "TensorRT-LLM engine",
            party: "engine",
        },
        inputs,
        |client| {
            client.set_structured_outputs_support(super::structured_outputs_support(&state.model));
        },
    )
    .await;
}
