//! Read-only RPCs: `HealthCheck`, `GetModelInfo`, `GetServerInfo`, `GetLoads`.
//! Metadata comes from the launcher's config (what the Python servicer reads
//! off SGLang's `ModelConfig` and `ServerArgs`) and from the handshake.

use prost_types::Timestamp;
use smg_grpc_client::sglang_proto as sg;
use tonic::Status;

use super::{State, HEALTHY};
use crate::{
    load_report::{rank_ids, Aggregate, RankFigures, RankLoad},
    proto_json::{boolean, insert_capacity, struct_from_json},
};

/// `HealthCheck`: SERVING only with the engine link up and no drain announced.
pub(super) fn health_check(state: &State) -> sg::HealthCheckResponse {
    sg::HealthCheckResponse {
        healthy: state.shared.is_serving(),
        message: state.shared.health_message(HEALTHY).to_string(),
    }
}

/// `GetModelInfo`: the launcher's config, with the handshake's context length
/// as the fallback for a launcher that reported none.
pub(super) fn model_info(state: &State) -> sg::GetModelInfoResponse {
    let model = &state.model;
    let ready = state.shared.ready();
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
    sg::GetModelInfoResponse {
        model_path: model.model_path.clone(),
        tokenizer_path: model.tokenizer_path.clone(),
        is_generation: model.is_generation,
        preferred_sampling_params: model.preferred_sampling_params.clone(),
        weight_version: model.weight_version.clone(),
        served_model_name: model.served_model_name.clone(),
        max_context_length,
        vocab_size: model.vocab_size,
        // Not the model's capability: this servicer carries no media to the
        // scheduler yet (the wire refuses `mm_inputs`), so it serves none,
        // whatever the launcher reported; see `server_info`, which says so
        // where the Router reads a verdict.
        supports_vision: false,
        model_type: model.model_type.clone(),
        eos_token_ids: model
            .eos_token_ids
            .iter()
            .map(|&id| i32::try_from(id).unwrap_or(i32::MAX))
            .collect(),
        pad_token_id: model.pad_token_id,
        bos_token_id: model.bos_token_id,
        max_req_input_len,
        architectures: model.architectures.clone(),
        id2label_json: model.id2label_json.clone(),
        num_labels: model.num_labels,
        default_sampling_params_json: model.default_sampling_params_json.clone(),
    }
}

/// `GetServerInfo`: the launcher's `server_args` (the Router's label source)
/// and a `scheduler_info` of the handshake's capacity figures over the
/// launcher's entries, as the Python servicer reports them.
pub(super) fn server_info(state: &State) -> sg::GetServerInfoResponse {
    let model = &state.model;
    let ready = state.shared.ready();
    let max_total_num_tokens = state.shared.kv_capacity_tokens();
    let mut server_args = struct_from_json(&model.server_args_json);
    // This servicer does not carry media to the scheduler yet (the wire
    // refuses `mm_inputs`), so it advertises no vision, on purpose and where
    // the Router reads it: `GetModelInfo.supports_vision` is a proto3 bool
    // whose `false` the Router drops as no verdict for this runtime, while a
    // `false` written here is the servicer's own, and the Router refuses
    // image and video input for the worker at its gate instead of after
    // fetching it. The label follows the model's capability again once the
    // wire carries media.
    server_args
        .fields
        .insert("supports_vision".to_string(), boolean(false));
    let mut scheduler_info = struct_from_json(&model.scheduler_info_json);
    if let Some(ready) = ready {
        insert_capacity(
            &mut scheduler_info,
            max_total_num_tokens,
            ready.block_size,
            state.shared.running_window(model.max_running_requests),
        );
    }
    sg::GetServerInfoResponse {
        server_args: Some(server_args),
        scheduler_info: Some(scheduler_info),
        active_requests: i32::try_from(state.shared.registry.len()).unwrap_or(i32::MAX),
        is_paused: false,
        last_receive_timestamp: 0.0,
        uptime_seconds: state.shared.started.elapsed().as_secs_f64(),
        sglang_version: state
            .shared
            .engine_version(&model.sglang_version, "sglang-"),
        // The Router strips this label; the two implementations serve the
        // same contract and are not told apart by it.
        server_type: "grpc".to_string(),
        start_time: Some(Timestamp::from(state.shared.started_at)),
        max_total_num_tokens,
    }
}

/// `GetLoads`: the per-rank load piggybacked on engine output, in the gRPC
/// response shape the Python servicer derives from the scheduler's load
/// snapshots. The `include` sections have no source on this wire.
pub(super) fn loads(state: &State, dp_rank: Option<i32>) -> Result<sg::GetLoadsResponse, Status> {
    let client = state.shared.engine()?;
    let max_total_num_tokens = state.shared.kv_capacity_tokens();
    let max_running_requests = state
        .shared
        .running_window(state.model.max_running_requests);
    let snapshot = client.get_loads();
    let mut loads: Vec<sg::SchedulerLoad> = snapshot
        .loads
        .into_iter()
        .filter(|load| dp_rank.is_none_or(|rank| rank == load.dp_rank))
        .map(|load| {
            let figures = RankFigures::of(&load, max_total_num_tokens);
            sg::SchedulerLoad {
                dp_rank: figures.dp_rank,
                num_running_reqs: figures.running,
                num_waiting_reqs: figures.waiting,
                num_total_reqs: figures.total,
                num_used_tokens: figures.used_tokens,
                max_total_num_tokens,
                max_running_requests,
                token_usage: figures.usage,
                ..Default::default()
            }
        })
        .collect();
    if loads.is_empty() && dp_rank.is_none() {
        // No output batch has carried a snapshot yet: a zero-filled entry per
        // rank, as the Python servicer reports an idle scheduler.
        let ranks = state.shared.rank_count(state.model.data_parallel_size);
        loads = rank_ids(ranks)
            .map(|dp_rank| sg::SchedulerLoad {
                dp_rank,
                max_running_requests,
                max_total_num_tokens,
                ..Default::default()
            })
            .collect();
    }
    let aggregate = Aggregate::of(loads.iter().map(|load| RankLoad {
        running: load.num_running_reqs,
        waiting: load.num_waiting_reqs,
        usage: load.token_usage,
    }));
    Ok(sg::GetLoadsResponse {
        timestamp: chrono::Utc::now().to_rfc3339(),
        version: state
            .shared
            .engine_version(&state.model.sglang_version, "sglang-"),
        dp_rank_count: i32::try_from(loads.len()).unwrap_or(i32::MAX),
        loads,
        aggregate: Some(sg::AggregateMetrics {
            total_running_reqs: aggregate.running,
            total_waiting_reqs: aggregate.waiting,
            total_reqs: aggregate.total,
            avg_token_usage: aggregate.avg_usage,
            ..Default::default()
        }),
    })
}
