//! Read-only RPCs: `HealthCheck`, `GetModelInfo`, `GetServerInfo`, `GetLoads`.
//! Metadata comes from the launcher's config (what the Python servicer reads
//! off TokenSpeed's `ModelConfig` and `ServerArgs`) and from the handshake.

use engine_zmq_adapter::ZmqEngineClient;
use prost_types::Timestamp;
use smg_grpc_client::tokenspeed_proto as ts;
use tonic::Status;

use super::State;
use crate::proto_json::{number, struct_from_json};

/// `HealthCheck`: SERVING only with the engine link up and no drain announced.
pub(super) fn health_check(state: &State) -> ts::HealthCheckResponse {
    ts::HealthCheckResponse {
        healthy: state.is_serving(),
        message: state.health_message().to_string(),
    }
}

/// `GetModelInfo`: the launcher's config, with the handshake's context length
/// as the fallback for a launcher that reported none.
pub(super) fn model_info(state: &State) -> ts::GetModelInfoResponse {
    let model = &state.model;
    let ready = state
        .engine
        .client
        .get()
        .and_then(ZmqEngineClient::ready_response);
    let mut max_context_length = model.max_context_length;
    if max_context_length == 0 {
        if let Some(ready) = ready {
            max_context_length = i32::try_from(ready.max_model_len).unwrap_or(i32::MAX);
        }
    }
    let max_req_input_len = if model.max_req_input_len > 0 {
        model.max_req_input_len
    } else {
        max_context_length
    };
    ts::GetModelInfoResponse {
        model_path: model.model_path.clone(),
        tokenizer_path: model.tokenizer_path.clone(),
        served_model_name: model.served_model_name.clone(),
        model_type: model.model_type.clone(),
        architectures: model.architectures.clone(),
        max_context_length,
        max_req_input_len,
        vocab_size: model.vocab_size,
        eos_token_ids: model
            .eos_token_ids
            .iter()
            .map(|&id| i32::try_from(id).unwrap_or(i32::MAX))
            .collect(),
        pad_token_id: model.pad_token_id,
        bos_token_id: model.bos_token_id,
        weight_version: model.weight_version.clone(),
        default_sampling_params_json: model.default_sampling_params_json.clone(),
        supports_vision: model.supports_vision,
        supports_multimodal: model.supports_multimodal,
        supported_modalities: model.supported_modalities.clone(),
        model_dtype: model.model_dtype.clone(),
        multimodal_encoder_dtype: model.multimodal_encoder_dtype.clone(),
    }
}

/// `GetServerInfo`: the launcher's `server_args` (the Router's label source)
/// and a `scheduler_info` of the handshake's capacity figures over the
/// launcher's entries, as the Python servicer reports them.
pub(super) fn server_info(state: &State) -> ts::GetServerInfoResponse {
    let model = &state.model;
    let ready = state
        .engine
        .client
        .get()
        .and_then(ZmqEngineClient::ready_response);
    let max_total_num_tokens = max_total_num_tokens(state);
    let mut scheduler_info = struct_from_json(&model.scheduler_info_json);
    if let Some(ready) = ready {
        scheduler_info.fields.insert(
            "max_total_num_tokens".to_string(),
            number(max_total_num_tokens),
        );
        scheduler_info.fields.insert(
            "page_size".to_string(),
            number(i32::try_from(ready.block_size).unwrap_or(i32::MAX)),
        );
        scheduler_info.fields.insert(
            "max_running_requests".to_string(),
            number(max_running_requests(state)),
        );
    }
    let version = Some(model.tokenspeed_version.clone())
        .filter(|version| !version.is_empty())
        .or_else(|| {
            ready.map(|ready| {
                ready
                    .vllm_version
                    .strip_prefix("tokenspeed-")
                    .unwrap_or(&ready.vllm_version)
                    .to_string()
            })
        })
        .filter(|version| !version.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    ts::GetServerInfoResponse {
        server_args: Some(struct_from_json(&model.server_args_json)),
        scheduler_info: Some(scheduler_info),
        active_requests: i32::try_from(state.registry.len()).unwrap_or(i32::MAX),
        is_paused: false,
        uptime_seconds: state.started.elapsed().as_secs_f64(),
        max_total_num_tokens,
        tokenspeed_version: version,
        start_time: Some(Timestamp::from(state.started_at)),
    }
}

/// `GetLoads`: the per-rank load piggybacked on engine output, in the gRPC
/// response shape the Python servicer derives from the scheduler's load
/// replies: pages in use become tokens, running is total minus waiting.
pub(super) fn loads(state: &State, dp_rank: Option<i32>) -> Result<ts::GetLoadsResponse, Status> {
    let client = state.engine()?;
    let ready = client.ready_response();
    let max_total_num_tokens = max_total_num_tokens(state);
    let max_running_requests = max_running_requests(state);
    let snapshot = client.get_loads();
    let mut loads: Vec<ts::SchedulerLoad> = snapshot
        .loads
        .into_iter()
        .filter(|load| dp_rank.is_none_or(|rank| rank == load.dp_rank))
        .map(|load| {
            let num_used_tokens = (load.token_usage * f64::from(max_total_num_tokens)).round();
            ts::SchedulerLoad {
                dp_rank: load.dp_rank,
                num_running_reqs: load.num_running_reqs,
                num_waiting_reqs: load.num_waiting_reqs,
                num_total_reqs: load.num_running_reqs.saturating_add(load.num_waiting_reqs),
                num_used_tokens: num_used_tokens.clamp(0.0, f64::from(i32::MAX)) as i32,
                max_total_num_tokens,
                max_running_requests,
                token_usage: load.token_usage,
                ..Default::default()
            }
        })
        .collect();
    if loads.is_empty() && dp_rank.is_none() {
        // No output batch has carried a snapshot yet: a zero-filled entry per
        // rank, as the Python servicer reports an idle scheduler.
        let ranks = ready
            .and_then(|ready| usize::try_from(ready.data_parallel_size).ok())
            .filter(|&ranks| ranks > 0)
            .or_else(|| usize::try_from(state.model.data_parallel_size).ok())
            .unwrap_or(1)
            .max(1);
        loads = (0..ranks)
            .map(|rank| ts::SchedulerLoad {
                dp_rank: i32::try_from(rank).unwrap_or(i32::MAX),
                max_running_requests,
                max_total_num_tokens,
                ..Default::default()
            })
            .collect();
    }
    let total_running: i32 = loads.iter().map(|load| load.num_running_reqs).sum();
    let total_waiting: i32 = loads.iter().map(|load| load.num_waiting_reqs).sum();
    let avg_token_usage = if loads.is_empty() {
        0.0
    } else {
        loads.iter().map(|load| load.token_usage).sum::<f64>() / loads.len() as f64
    };
    Ok(ts::GetLoadsResponse {
        timestamp: chrono::Utc::now().to_rfc3339(),
        version: "tokenspeed".to_string(),
        dp_rank_count: i32::try_from(loads.len()).unwrap_or(i32::MAX),
        loads,
        aggregate: Some(ts::AggregateMetrics {
            total_running_reqs: total_running,
            total_waiting_reqs: total_waiting,
            total_reqs: total_running.saturating_add(total_waiting),
            avg_token_usage,
            ..Default::default()
        }),
    })
}

/// The KV capacity in tokens, from the handshake; 0 before the engines are up.
fn max_total_num_tokens(state: &State) -> i32 {
    state
        .engine
        .client
        .get()
        .and_then(ZmqEngineClient::ready_response)
        .and_then(|ready| ready.kv_cache_size_tokens)
        .map(|tokens| i32::try_from(tokens).unwrap_or(i32::MAX))
        .unwrap_or(0)
}

/// The admission window: the launcher's `max_num_seqs`, else the handshake's.
fn max_running_requests(state: &State) -> i32 {
    if state.model.max_running_requests > 0 {
        return state.model.max_running_requests;
    }
    state
        .engine
        .client
        .get()
        .and_then(ZmqEngineClient::ready_response)
        .map(|ready| i32::try_from(ready.max_num_seqs).unwrap_or(i32::MAX))
        .unwrap_or(0)
}
