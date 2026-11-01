//! Connecting the headless scheduler(s), in the background of a server that
//! is already listening. Failures gate health and surface as the last error;
//! nothing here retries, the lifecycle owner restarts the pair.

use std::sync::Arc;

use openai_protocol::worker::RuntimeType;

use super::State;
use crate::engine_link::{self, ConnectInputs, PeerWords};

/// This wire's connect ([`engine_link::connect_engine`] with its words): the
/// connected scheduler takes the structured-outputs support the launcher's
/// arguments say, the note that it truncates long prompts and the note that
/// it is a prefill side placing a disaggregated request by its room, when
/// they say so. The start-up and silence bounds are the link's.
pub(super) async fn connect_engine(state: Arc<State>, inputs: ConnectInputs) {
    engine_link::connect_engine(
        &state.shared.engine,
        state.model.model_path.clone(),
        &state.model.eos_token_ids,
        RuntimeType::Sglang,
        PeerWords {
            peer: "SGLang scheduler",
            party: "scheduler",
        },
        inputs,
        |client| {
            client.set_structured_outputs_support(super::structured_outputs_support(&state.model));
            if super::truncates_long_prompts(&state.model) {
                client.note_engine_truncates_long_prompts();
            }
            if super::places_by_rendezvous_room(&state.model) {
                client.note_engine_places_by_rendezvous_room();
            }
            Ok(())
        },
    )
    .await;
}
