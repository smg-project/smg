//! The link to the engine: `None` while the ZMQ handshake runs, then the
//! client or the reason it failed. Health is gated on it.

use std::sync::Arc;

use engine_zmq_adapter::structured_outputs_backend_from_config;
use openai_protocol::worker::RuntimeType;

use super::State;
use crate::engine_link::{self, load_tokenizer_once, ConnectInputs, PeerWords};

/// This wire's connect: the tokenizer is loaded first (string stops need
/// it), then [`engine_link::connect_engine`] with this wire's words; the
/// connected engine takes the tokenizer's EOS, the structured-outputs
/// backend its config names and, when its statistics are on, the note that
/// it reports every scheduler step. The start-up and silence bounds are the
/// link's.
pub(super) async fn connect_engine(state: Arc<State>, inputs: ConnectInputs) {
    load_tokenizer_once(&state.tokenizer, state.shared.tokenizer_dir.as_deref()).await;

    engine_link::connect_engine(
        &state.shared.engine,
        state.model.model_path.clone(),
        &state.model.eos_token_ids,
        RuntimeType::Vllm,
        PeerWords {
            peer: "vLLM engine",
            party: "engine",
        },
        inputs,
        |client| {
            client.adopt_tokenizer_eos(state.tokenizer());
            client.set_structured_outputs_support(
                structured_outputs_backend_from_config(&state.model.structured_outputs_backend)
                    .into(),
            );
            // The model's vocabulary, widened where vLLM's frontend widens
            // it by the tokenizer's largest id (a multimodal placeholder can
            // lie past the model's).
            client.set_vocab_size(
                state.model.vocab_size,
                state
                    .tokenizer()
                    .and_then(|tokenizer| tokenizer.max_token_id()),
            );
            // A PD decode side's prompt KV is pulled only by an engine
            // started as a KV consumer; elsewhere a handoff claim is ignored
            // and the engine computes the prompt itself.
            if matches!(state.model.kv_role.as_str(), "kv_consumer" | "kv_both") {
                client.note_engine_pulls_remote_kv();
            }
            // Known at start: the wire reports every scheduler step while the
            // engine's statistics are on, before its first batch shows it.
            if state.model.log_stats == Some(true) {
                client.note_reports_steps();
            }
            // The stats tick logs the connector's KV-transfer figures
            // (`info::log_engine_stats`): keep the samples from the first
            // batch on.
            client.keep_kv_transfer();
            Ok(())
        },
    )
    .await;
}
