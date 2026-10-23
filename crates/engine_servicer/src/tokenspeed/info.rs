//! Read-only RPCs: `HealthCheck`, `GetModelInfo`, `GetServerInfo`, `GetLoads`.
//! Metadata comes from the launcher's config (what the Python servicer reads
//! off TokenSpeed's `ModelConfig` and `ServerArgs`) and from the handshake.

use engine_zmq_adapter::ZmqEngineClient;
use prost_types::Timestamp;
use smg_grpc_client::tokenspeed_proto as ts;
use tonic::Status;

use super::{State, HEALTHY};
use crate::{
    load_report::{rank_ids, Aggregate, RankFigures, RankLoad},
    proto_json::{insert_capacity, struct_from_json},
};

/// `HealthCheck`: SERVING only with the engine link up and no drain announced.
pub(super) fn health_check(state: &State) -> ts::HealthCheckResponse {
    ts::HealthCheckResponse {
        healthy: state.shared.is_serving(),
        message: state.shared.health_message(HEALTHY).to_string(),
    }
}

/// `GetModelInfo`: the launcher's config, with the handshake's context length
/// as the fallback for a launcher that reported none.
pub(super) fn model_info(state: &State) -> ts::GetModelInfoResponse {
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
    let ready = state.shared.ready();
    let max_total_num_tokens = state.shared.kv_capacity_tokens();
    let mut scheduler_info = struct_from_json(&model.scheduler_info_json);
    if let Some(ready) = ready {
        let epochs = state
            .shared
            .engine
            .client
            .get()
            .map(ZmqEngineClient::cache_trace_epochs)
            .unwrap_or_default();
        scheduler_info.fields.insert(
            "cache_trace_epochs".to_string(),
            prost_types::Value {
                kind: Some(prost_types::value::Kind::ListValue(
                    prost_types::ListValue {
                        values: epochs
                            .into_iter()
                            .map(|epoch| prost_types::Value {
                                kind: Some(prost_types::value::Kind::StringValue(epoch)),
                            })
                            .collect(),
                    },
                )),
            },
        );
        insert_capacity(
            &mut scheduler_info,
            max_total_num_tokens,
            ready.block_size,
            state.shared.running_window(model.max_running_requests),
        );
    }
    let version = state
        .shared
        .engine_version(&model.tokenspeed_version, "tokenspeed-");
    ts::GetServerInfoResponse {
        server_args: Some(struct_from_json(&model.server_args_json)),
        scheduler_info: Some(scheduler_info),
        active_requests: i32::try_from(state.shared.registry.len()).unwrap_or(i32::MAX),
        is_paused: false,
        uptime_seconds: state.shared.started.elapsed().as_secs_f64(),
        max_total_num_tokens,
        tokenspeed_version: version,
        start_time: Some(Timestamp::from(state.shared.started_at)),
        constrained_decoding: model.constrained_decoding.clone(),
    }
}

/// `GetLoads`: the per-rank load piggybacked on engine output, in the gRPC
/// response shape the Python servicer derives from the scheduler's load
/// replies: pages in use become tokens, running is total minus waiting.
pub(super) fn loads(state: &State, dp_rank: Option<i32>) -> Result<ts::GetLoadsResponse, Status> {
    let client = state.shared.engine()?;
    let max_total_num_tokens = state.shared.kv_capacity_tokens();
    let max_running_requests = state
        .shared
        .running_window(state.model.max_running_requests);
    let snapshot = client.get_loads();
    let mut loads: Vec<ts::SchedulerLoad> = snapshot
        .loads
        .into_iter()
        .filter(|load| dp_rank.is_none_or(|rank| rank == load.dp_rank))
        .map(|load| {
            let figures = RankFigures::of(&load, max_total_num_tokens);
            ts::SchedulerLoad {
                dp_rank: figures.dp_rank,
                num_running_reqs: figures.running,
                num_waiting_reqs: figures.waiting,
                num_total_reqs: figures.total,
                num_used_tokens: figures.used_tokens,
                max_total_num_tokens,
                max_running_requests,
                token_usage: figures.usage,
                // The direct TokenSpeed wire reports active pages, excluding
                // retained cache (unlike the Python legacy GetLoad projection).
                active_token_usage: Some(figures.usage),
                ..Default::default()
            }
        })
        .collect();
    if loads.is_empty() && dp_rank.is_none() {
        // No output batch has carried a snapshot yet: a zero-filled entry per
        // rank, as the Python servicer reports an idle scheduler.
        let ranks = state.shared.rank_count(state.model.data_parallel_size);
        loads = rank_ids(ranks)
            .map(|dp_rank| ts::SchedulerLoad {
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
    Ok(ts::GetLoadsResponse {
        timestamp: chrono::Utc::now().to_rfc3339(),
        version: "tokenspeed".to_string(),
        dp_rank_count: i32::try_from(loads.len()).unwrap_or(i32::MAX),
        loads,
        aggregate: Some(ts::AggregateMetrics {
            total_running_reqs: aggregate.running,
            total_waiting_reqs: aggregate.waiting,
            total_reqs: aggregate.total,
            avg_token_usage: aggregate.avg_usage,
            ..Default::default()
        }),
    })
}
