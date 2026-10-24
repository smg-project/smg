//! Read-only RPCs: `HealthCheck`, `GetModelInfo`, `GetServerInfo`. Metadata
//! comes from the launcher's facts (what the engine's Python servicer reads
//! off the LLM API's arguments and tokenizer) and from the handshake.

use smg_grpc_client::trtllm_proto as trt;

use super::{State, HEALTHY};

/// `HealthCheck`: `OK` only with the engine link up and no drain announced,
/// else the shared message (the Python servicer's status is its message too).
pub(super) fn health_check(state: &State) -> trt::HealthCheckResponse {
    trt::HealthCheckResponse {
        status: state.shared.health_message(HEALTHY).to_string(),
    }
}

fn dim<T: TryInto<i32>>(value: T) -> i32 {
    value.try_into().unwrap_or(i32::MAX)
}

/// A launcher fact, else the handshake's figure, else `fallback`.
fn fact_or(fact: i32, handshake: Option<i32>, fallback: i32) -> i32 {
    if fact > 0 {
        fact
    } else {
        handshake.filter(|&figure| figure > 0).unwrap_or(fallback)
    }
}

/// `GetModelInfo`: the launcher's facts, with the handshake's window as the
/// fallback for the bounds a launcher reported as 0.
pub(super) fn model_info(state: &State) -> trt::GetModelInfoResponse {
    let model = &state.model;
    let window = state
        .shared
        .ready()
        .map(|ready| dim(ready.max_model_len))
        .filter(|&window| window > 0);
    let max_seq_len = fact_or(model.max_seq_len, window, model.max_input_len);
    let max_input_len = fact_or(model.max_input_len, window, max_seq_len);
    trt::GetModelInfoResponse {
        model_id: if model.served_model_name.is_empty() {
            model.model_path.clone()
        } else {
            model.served_model_name.clone()
        },
        max_input_len,
        max_seq_len,
        max_batch_size: model.max_batch_size,
        vocab_size: model.vocab_size,
        hidden_size: model.hidden_size,
        num_layers: model.num_layers,
        num_heads: model.num_heads,
        supported_features: model.supported_features.clone(),
    }
}

/// `GetServerInfo`: the engine's version and backend as the launcher reports
/// them (the handshake's version is the fallback) and the parallel sizes,
/// the handshake's figures filling what the launcher left at 0.
pub(super) fn server_info(state: &State) -> trt::GetServerInfoResponse {
    let model = &state.model;
    let ready = state.shared.ready();
    let tensor_parallel_size = fact_or(
        model.tensor_parallel_size,
        ready.map(|r| dim(r.tensor_parallel_size)),
        1,
    );
    let pipeline_parallel_size = fact_or(
        model.pipeline_parallel_size,
        ready.map(|r| dim(r.pipeline_parallel_size)),
        1,
    );
    let context_parallel_size = fact_or(
        model.context_parallel_size,
        ready.map(|r| dim(r.decode_context_parallel_size)),
        1,
    );
    trt::GetServerInfoResponse {
        version: state
            .shared
            .engine_version(&model.trtllm_version, "trtllm-"),
        backend: if model.backend.is_empty() {
            "tensorrt-llm".to_string()
        } else {
            model.backend.clone()
        },
        tensor_parallel_size,
        pipeline_parallel_size,
        context_parallel_size,
        // The product of the launcher's sizes when it reported any; else the
        // handshake's world, else the product of the handshake's sizes.
        world_size: if model.tensor_parallel_size > 0
            || model.pipeline_parallel_size > 0
            || model.context_parallel_size > 0
        {
            world_of(
                tensor_parallel_size,
                pipeline_parallel_size,
                context_parallel_size,
            )
        } else {
            fact_or(
                0,
                ready.map(|r| dim(r.world_size)),
                world_of(
                    tensor_parallel_size,
                    pipeline_parallel_size,
                    context_parallel_size,
                ),
            )
        },
    }
}

fn world_of(tensor: i32, pipeline: i32, context: i32) -> i32 {
    tensor.saturating_mul(pipeline).saturating_mul(context)
}
