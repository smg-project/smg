//! The informational RPCs: `HealthCheck` (the Python servicer's one-token
//! probe of the scheduler), `GetModelInfo`, `GetServerInfo`, `GetLoads`.
//! Metadata comes from the launcher's config (what the Python servicer reads
//! off TokenSpeed's `ModelConfig` and `ServerArgs`) and from the handshake.

use std::{sync::Arc, time::Duration};

use engine_zmq_adapter::ZmqEngineClient;
use futures::StreamExt;
use openai_protocol::worker::SchedulerLoadSnapshot;
use prost_types::Timestamp;
use smg_grpc_client::tokenspeed_proto as ts;
use tonic::Status;

use super::{generate, State};
use crate::{
    health_probe::{self, Verdict},
    load_report::{mean, rank_entries, rank_entry, Capacity, RankFigures},
    proto_json::{insert_capacity, struct_from_json},
    servicer::{proto_token_ids, RequestOrigin},
};

/// What `HealthCheck` says of a scheduler that answered the probe, or
/// pushed anything while it was out (the Python servicer's words).
const PASSED: &str = "Health check passed";

/// `HealthCheck`: unhealthy with the engine link down or the drain
/// announced, as every wire; else the Python servicer's deep probe, a
/// one-token request through [`generate::generate`], healthy once the
/// scheduler pushed anything since the probe started or answered it,
/// unhealthy the moment the probe fails, unhealthy at the servicer's bound
/// with neither ([`health_probe::run`]). A disaggregated role (prefill,
/// decode, encode) answers shallow, as the Python servicer does: its
/// requests pair with a peer, so a lone probe would not reach the scheduler.
pub(super) async fn health_check(state: &Arc<State>) -> ts::HealthCheckResponse {
    health_check_with(state, state.health_probe_timeout).await
}

/// [`health_check`] with the probe's bound given.
pub(super) async fn health_check_with(
    state: &Arc<State>,
    timeout: Duration,
) -> ts::HealthCheckResponse {
    if !state.shared.is_serving() {
        return ts::HealthCheckResponse {
            healthy: false,
            message: state.shared.health_message(PASSED).to_string(),
        };
    }
    if health_probe::disaggregated_role(&state.model.server_args_json) {
        return ts::HealthCheckResponse {
            healthy: true,
            message: health_probe::SHALLOW_MESSAGE.to_string(),
        };
    }
    let client = match state.shared.engine() {
        Ok(client) => client,
        Err(status) => {
            return ts::HealthCheckResponse {
                healthy: false,
                message: status.message().to_string(),
            }
        }
    };
    let verdict = health_probe::run(timeout, || client.last_output_at(), probe(state)).await;
    ts::HealthCheckResponse {
        healthy: verdict.healthy(),
        message: match verdict {
            Verdict::Passed => PASSED.to_string(),
            Verdict::Failed => "Scheduler returned no output".to_string(),
            Verdict::TimedOut => format!("Health check timeout after {}s", timeout.as_secs()),
        },
    }
}

/// The probe, as the Python servicer sends it: one token from the prompt
/// `[0]` at temperature zero under a `HEALTH_CHECK_` id, resolved at its
/// first output. It is the servicer's own, counted neither as a client's
/// request nor as load ([`RequestOrigin::HealthProbe`]). Dropping it before
/// the output aborts the request on the scheduler, as a client's stream
/// does.
async fn probe(state: &Arc<State>) -> Result<(), Status> {
    let request = ts::GenerateRequest {
        request_id: health_probe::probe_request_id(),
        tokenized: Some(ts::TokenizedInput {
            input_ids: vec![0],
            original_text: String::new(),
        }),
        sampling_params: Some(ts::SamplingParams {
            max_new_tokens: Some(1),
            temperature: Some(0.0),
            ..Default::default()
        }),
        stream: false,
        ..Default::default()
    };
    let mut stream = generate::generate(state, request, RequestOrigin::HealthProbe).await?;
    match stream.next().await {
        Some(Ok(_)) => Ok(()),
        Some(Err(status)) => Err(status),
        None => Err(Status::internal(
            "the probe's stream ended without an output",
        )),
    }
}

/// `GetModelInfo`: the launcher's config, with the handshake's context length
/// as the fallback for a launcher that reported none.
pub(super) fn model_info(state: &State) -> ts::GetModelInfoResponse {
    let model = &state.model;
    let max_context_length = state.shared.context_window(model.max_context_length);
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
        eos_token_ids: proto_token_ids(&model.eos_token_ids),
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
/// replies (pages in use become tokens, running is total minus waiting), with
/// the servicer's own bookkeeping ([`crate::load_tracker`]) for the queued
/// token-work, the prefill and generation throughput and the hit rate this
/// wire's load tail does not carry, each rank's own.
pub(super) fn loads(state: &State, dp_rank: Option<i32>) -> Result<ts::GetLoadsResponse, Status> {
    let client = state.shared.engine()?;
    if state.encode_role.is_some() {
        // An encode worker has no scheduler: its engine runs the vision
        // tower alone, with no running batch, no waiting queue and no KV
        // cache, and its encode path feeds no load figure. The answer is
        // the Python servicer's: no rank, an empty aggregate; the load
        // record the relay and the records-only stream read from here is
        // then none, so nothing is pushed for the role either.
        return Ok(ts::GetLoadsResponse {
            timestamp: chrono::Utc::now().to_rfc3339(),
            version: "tokenspeed".to_string(),
            dp_rank_count: 0,
            loads: Vec::new(),
            aggregate: Some(ts::AggregateMetrics::default()),
        });
    }
    let capacity = Capacity::of(&state.shared, state.model.max_running_requests);
    let ranks = state.shared.rank_count(state.model.data_parallel_size);
    let (loads, aggregate) = rank_entries(
        client.get_loads().loads,
        capacity.max_total_num_tokens,
        ranks,
        dp_rank,
        |load, figures| reported_entry(state, load, figures, &capacity),
        |dp_rank| unreported_entry(state, dp_rank, &capacity),
    );
    let aggregate = ts::AggregateMetrics {
        total_running_reqs: aggregate.running,
        total_waiting_reqs: aggregate.waiting,
        total_reqs: aggregate.total,
        avg_token_usage: aggregate.avg_usage,
        avg_throughput: mean(loads.iter().map(|load| load.gen_throughput)),
        avg_utilization: mean(loads.iter().map(|load| load.utilization)),
    };
    Ok(ts::GetLoadsResponse {
        timestamp: chrono::Utc::now().to_rfc3339(),
        version: "tokenspeed".to_string(),
        dp_rank_count: i32::try_from(loads.len()).unwrap_or(i32::MAX),
        loads,
        aggregate: Some(aggregate),
    })
}

/// One rank's entry of [`loads`], built as that answer builds it and read
/// alone: no timestamp, no version, no vector of the other ranks and this
/// rank's estimate only, for the record the relay attaches to every batch
/// and tick. `None` names the answer's first entry, rank 0 (a connected
/// engine makes it a rank); a rank past the connected ones has no entry, and
/// an encode worker has none at all (it answers `GetLoads` with no rank).
pub(super) fn rank_load(state: &State, dp_rank: Option<i32>) -> Option<ts::SchedulerLoad> {
    let client = state.shared.engine().ok()?;
    if state.encode_role.is_some() {
        return None;
    }
    let capacity = Capacity::of(&state.shared, state.model.max_running_requests);
    rank_entry(
        |index| client.rank_load(index),
        capacity.max_total_num_tokens,
        state.shared.rank_count(state.model.data_parallel_size),
        dp_rank,
        |load, figures| reported_entry(state, load, figures, &capacity),
        |rank| unreported_entry(state, rank, &capacity),
    )
}

/// A reported rank's entry: the snapshot's figures with, from this
/// servicer's own bookkeeping ([`crate::load_tracker`]), the queued
/// token-work, prefill/generation throughput and hit rate this wire's load
/// tail does not carry, the rank's own.
fn reported_entry(
    state: &State,
    load: &SchedulerLoadSnapshot,
    figures: &RankFigures,
    capacity: &Capacity,
) -> ts::SchedulerLoad {
    let estimate = state
        .shared
        .loads
        .estimate(load.dp_rank, load.num_waiting_reqs);
    ts::SchedulerLoad {
        dp_rank: figures.dp_rank,
        num_running_reqs: figures.running,
        num_waiting_reqs: figures.waiting,
        num_total_reqs: figures.total,
        num_used_tokens: figures.used_tokens,
        max_total_num_tokens: capacity.max_total_num_tokens,
        max_running_requests: capacity.max_running_requests,
        token_usage: figures.usage,
        utilization: figures.usage,
        // The direct TokenSpeed wire reports active pages, excluding
        // retained cache (unlike the Python legacy GetLoad projection).
        active_token_usage: Some(figures.usage),
        num_waiting_uncached_tokens: estimate.queued_token_work,
        gen_throughput: estimate.gen_throughput,
        prefill_throughput: estimate.prefill_throughput,
        cache_hit_rate: estimate.cache_hit_rate,
        ..Default::default()
    }
}

/// A connected rank no load tail has reported yet: a zero-filled entry with
/// the servicer's own figures for the rank (see [`rank_entries`]).
fn unreported_entry(state: &State, dp_rank: i32, capacity: &Capacity) -> ts::SchedulerLoad {
    let estimate = state.shared.loads.estimate(dp_rank, i32::MAX);
    ts::SchedulerLoad {
        dp_rank,
        max_running_requests: capacity.max_running_requests,
        max_total_num_tokens: capacity.max_total_num_tokens,
        num_waiting_uncached_tokens: estimate.queued_token_work,
        gen_throughput: estimate.gen_throughput,
        prefill_throughput: estimate.prefill_throughput,
        cache_hit_rate: estimate.cache_hit_rate,
        ..Default::default()
    }
}
