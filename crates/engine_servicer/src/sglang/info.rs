//! Read-only RPCs: `HealthCheck`, `GetModelInfo`, `GetServerInfo`, `GetLoads`.
//! Metadata comes from the launcher's config (what the Python servicer reads
//! off SGLang's `ModelConfig` and `ServerArgs`) and from the handshake.

use prost_types::Timestamp;
use smg_grpc_client::sglang_proto as sg;
use tonic::Status;

use super::{State, HEALTHY};
use crate::{
    load_report::{mean, rank_entries},
    proto_json::{insert_capacity, struct_from_json},
    servicer::proto_token_ids,
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
    let max_context_length = state.shared.context_window(model.max_context_length);
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
        supports_vision: model.supports_vision,
        model_type: model.model_type.clone(),
        eos_token_ids: proto_token_ids(&model.eos_token_ids),
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
    if state.media.is_some() {
        for (key, value) in [
            ("mm_processor", "smg"),
            ("mm_media_ref_schemes", "http,https,data"),
        ] {
            server_args.fields.insert(
                key.to_string(),
                prost_types::Value {
                    kind: Some(prost_types::value::Kind::StringValue(value.to_string())),
                },
            );
        }
    }
    if let Some(key) = super::media_identity::key(state) {
        server_args.fields.insert(
            smg_grpc_client::sglang_media_identity::LABEL.to_string(),
            prost_types::Value {
                kind: Some(prost_types::value::Kind::StringValue(key.to_string())),
            },
        );
    }
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
/// snapshots, with the servicer's own bookkeeping ([`crate::load_tracker`])
/// for the queued token-work, the prefill and generation throughput and the
/// hit rate this wire's load tail does not carry, each rank's own. The
/// `include` sections have no source on this wire.
pub(super) fn loads(state: &State, dp_rank: Option<i32>) -> Result<sg::GetLoadsResponse, Status> {
    let client = state.shared.engine()?;
    let max_total_num_tokens = state.shared.kv_capacity_tokens();
    let max_running_requests = state
        .shared
        .running_window(state.model.max_running_requests);
    let ranks = state.shared.rank_count(state.model.data_parallel_size);
    let (loads, aggregate) = rank_entries(
        client.get_loads().loads,
        max_total_num_tokens,
        ranks,
        dp_rank,
        |load, figures| {
            let estimate = state
                .shared
                .loads
                .estimate(load.dp_rank, load.num_waiting_reqs);
            sg::SchedulerLoad {
                dp_rank: figures.dp_rank,
                num_running_reqs: figures.running,
                num_waiting_reqs: figures.waiting,
                num_total_reqs: figures.total,
                num_used_tokens: figures.used_tokens,
                max_total_num_tokens,
                max_running_requests,
                token_usage: figures.usage,
                utilization: figures.usage,
                num_waiting_uncached_tokens: estimate.queued_token_work,
                gen_throughput: estimate.gen_throughput,
                prefill_throughput: estimate.prefill_throughput,
                cache_hit_rate: estimate.cache_hit_rate,
                ..Default::default()
            }
        },
        |dp_rank| {
            let estimate = state.shared.loads.estimate(dp_rank, i32::MAX);
            sg::SchedulerLoad {
                dp_rank,
                max_running_requests,
                max_total_num_tokens,
                num_waiting_uncached_tokens: estimate.queued_token_work,
                gen_throughput: estimate.gen_throughput,
                prefill_throughput: estimate.prefill_throughput,
                cache_hit_rate: estimate.cache_hit_rate,
                ..Default::default()
            }
        },
    );
    let aggregate = sg::AggregateMetrics {
        total_running_reqs: aggregate.running,
        total_waiting_reqs: aggregate.waiting,
        total_reqs: aggregate.total,
        avg_token_usage: aggregate.avg_usage,
        avg_throughput: mean(loads.iter().map(|load| load.gen_throughput)),
        avg_utilization: mean(loads.iter().map(|load| load.utilization)),
    };
    Ok(sg::GetLoadsResponse {
        timestamp: chrono::Utc::now().to_rfc3339(),
        version: state
            .shared
            .engine_version(&state.model.sglang_version, "sglang-"),
        dp_rank_count: i32::try_from(loads.len()).unwrap_or(i32::MAX),
        loads,
        aggregate: Some(aggregate),
    })
}
