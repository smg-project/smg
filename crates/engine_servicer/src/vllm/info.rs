//! Read-only RPCs: `HealthCheck`, `GetModelInfo`, `GetServerInfo`, `GetLoads`.
//! Metadata comes from the launcher's config (what the Python servicer reads
//! off vLLM's `ModelConfig`) and from the engine handshake.

use engine_zmq_adapter::ZmqEngineClient;
use smg_grpc_client::vllm_proto as vllm;
use tonic::Status;

use super::{State, SERVER_TYPE};

/// `HealthCheck`: SERVING only with the engine link up and no drain announced.
pub(super) fn health_check(state: &State) -> vllm::HealthCheckResponse {
    vllm::HealthCheckResponse {
        healthy: state.is_serving(),
        message: state.health_message().to_string(),
    }
}

/// `GetModelInfo`: the launcher's config, with the handshake's context length
/// as the fallback for a launcher that reported none.
pub(super) fn model_info(state: &State) -> vllm::GetModelInfoResponse {
    let model = &state.model;
    // The launcher's context length wins; the handshake's is the fallback
    // for a launcher that reported none.
    let mut max_context_length = model.max_context_length;
    if max_context_length == 0 {
        if let Some(ready) = state
            .engine
            .client
            .get()
            .and_then(ZmqEngineClient::ready_response)
        {
            max_context_length = u32::try_from(ready.max_model_len).unwrap_or(u32::MAX);
        }
    }
    vllm::GetModelInfoResponse {
        model_path: model.model_path.clone(),
        is_generation: model.is_generation,
        max_context_length,
        vocab_size: model.vocab_size,
        supports_vision: model.supports_vision,
        served_model_name: model.served_model_name.clone(),
        tokenizer_path: model.tokenizer_path.clone(),
        model_type: model.model_type.clone(),
        architectures: model.architectures.clone(),
        eos_token_ids: model
            .eos_token_ids
            .iter()
            .map(|&id| i32::try_from(id).unwrap_or(i32::MAX))
            .collect(),
        pad_token_id: model.pad_token_id,
        bos_token_id: model.bos_token_id,
        max_req_input_len: i32::try_from(max_context_length).unwrap_or(i32::MAX),
        default_sampling_params_json: model.default_sampling_params_json.clone(),
    }
}

/// `GetServerInfo`: this implementation's identity plus the handshake facts.
pub(super) fn server_info(state: &State) -> vllm::GetServerInfoResponse {
    let ready = state
        .engine
        .client
        .get()
        .and_then(ZmqEngineClient::ready_response);
    let data_parallel_size = ready
        .map(|ready| i32::try_from(ready.data_parallel_size).unwrap_or(i32::MAX))
        .filter(|&size| size > 0)
        .unwrap_or_else(|| state.model.data_parallel_size.max(1));
    vllm::GetServerInfoResponse {
        active_requests: state.active_requests(),
        uptime_seconds: state.started.elapsed().as_secs_f64(),
        server_type: SERVER_TYPE.to_string(),
        data_parallel_size,
        pairing_protocol: state.model.pairing_protocol.clone(),
        block_size: ready
            .map(|ready| i32::try_from(ready.block_size).unwrap_or(i32::MAX))
            .unwrap_or_default(),
        model_dtype: ready
            .map(|ready| ready.dtype.as_str().to_string())
            .unwrap_or_default(),
        ..Default::default()
    }
}

/// `GetLoads`: the per-rank load piggybacked on engine output, in the gRPC
/// response shape, with the handshake's capacity figures.
pub(super) fn loads(state: &State) -> Result<vllm::GetLoadsResponse, Status> {
    let client = state.engine()?;
    let ready = client.ready_response();
    let max_running_requests = ready
        .map(|ready| i32::try_from(ready.max_num_seqs).unwrap_or(i32::MAX))
        .unwrap_or_default();
    let max_total_num_tokens = ready
        .and_then(|ready| ready.kv_cache_size_tokens)
        .map(|tokens| i32::try_from(tokens).unwrap_or(i32::MAX))
        .unwrap_or_default();
    let snapshot = client.get_loads();
    let loads = snapshot
        .loads
        .into_iter()
        .map(|load| vllm::SchedulerLoad {
            dp_rank: load.dp_rank,
            num_running_reqs: load.num_running_reqs,
            num_waiting_reqs: load.num_waiting_reqs,
            num_total_reqs: load.num_running_reqs.saturating_add(load.num_waiting_reqs),
            token_usage: load.token_usage,
            max_running_requests,
            max_total_num_tokens,
            ..Default::default()
        })
        .collect();
    Ok(vllm::GetLoadsResponse {
        timestamp: chrono::Utc::now().to_rfc3339(),
        version: "1".to_string(),
        dp_rank_count: snapshot.dp_rank_count,
        loads,
    })
}
