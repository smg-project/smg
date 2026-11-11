//! Read-only RPCs: `HealthCheck`, `GetModelInfo`, `GetServerInfo`. Metadata
//! comes from the launcher's facts (what the engine's Python servicer reads
//! off the LLM API's arguments and tokenizer) and from the handshake.

use openai_protocol::worker::SchedulerLoadSnapshot;
use smg_grpc_client::trtllm_proto as trt;
use tonic::Status;

use super::{State, HEALTHY};
use crate::{
    load_report::{rank_entries, rank_entry, Capacity, RankFigures},
    load_tracker::LoadEstimate,
};

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
/// them (the handshake's version is the fallback), the parallel sizes, the
/// handshake's figures filling what the launcher left at 0, and the
/// vocabulary of structural-tag nodes the launcher read off its xgrammar.
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
        structural_tag_vocabulary: model.structural_tag_vocabulary.clone(),
    }
}

fn world_of(tensor: i32, pipeline: i32, context: i32) -> i32 {
    tensor.saturating_mul(pipeline).saturating_mul(context)
}

/// `GetLoads`: the per-rank load from the shims' load tails (the figures the
/// pushed record carries) in the gRPC response shape, with the servicer's
/// own bookkeeping ([`crate::load_tracker`]) for the queued token-work, the
/// prefill and generation throughput and the hit rate the tail does not
/// carry, each rank's own: every connected rank, zero-filled before its
/// first tail, with the window and the KV capacity the servicer knows; the
/// rank the request names alone when it names one.
pub(super) fn loads(state: &State, dp_rank: Option<i32>) -> Result<trt::GetLoadsResponse, Status> {
    let client = state.shared.engine()?;
    let capacity = Capacity::of(&state.shared, state.model.max_running_requests);
    let ranks = state.shared.rank_count(state.model.data_parallel_size);
    let (loads, _) = rank_entries(
        client.get_loads().loads,
        capacity.max_total_num_tokens,
        ranks,
        dp_rank,
        |load, figures| reported_entry(state, load, figures, &capacity),
        |dp_rank| unreported_entry(state, dp_rank, &capacity),
    );
    Ok(trt::GetLoadsResponse {
        timestamp: chrono::Utc::now().to_rfc3339(),
        version: state
            .shared
            .engine_version(&state.model.trtllm_version, "trtllm-"),
        dp_rank_count: i32::try_from(loads.len()).unwrap_or(i32::MAX),
        loads,
    })
}

/// One rank's entry of [`loads`], built as that answer builds it and read
/// alone: no timestamp, no version, no vector of the other ranks and this
/// rank's estimate only, for the record the relay attaches to every batch
/// and tick. `None` names the answer's first entry, rank 0 (a connected
/// shim makes it a rank); a rank past the connected ones has no entry.
pub(super) fn rank_load(state: &State, dp_rank: Option<i32>) -> Option<trt::SchedulerLoad> {
    let client = state.shared.engine().ok()?;
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

/// One rank's entry from its figures and the servicer's own estimate for it.
fn entry(figures: &RankFigures, capacity: &Capacity, estimate: LoadEstimate) -> trt::SchedulerLoad {
    trt::SchedulerLoad {
        dp_rank: figures.dp_rank,
        num_running_reqs: figures.running,
        num_waiting_reqs: figures.waiting,
        num_total_reqs: figures.total,
        num_used_tokens: figures.used_tokens,
        max_total_num_tokens: capacity.max_total_num_tokens,
        token_usage: figures.usage,
        utilization: figures.usage,
        max_running_requests: capacity.max_running_requests,
        num_waiting_uncached_tokens: estimate.queued_token_work,
        gen_throughput: estimate.gen_throughput,
        prefill_throughput: estimate.prefill_throughput,
        cache_hit_rate: estimate.cache_hit_rate,
    }
}

/// A reported rank's entry: the tail's figures with, from this servicer's
/// own bookkeeping ([`crate::load_tracker`]), the queued token-work,
/// prefill/generation throughput and hit rate the tail does not carry, the
/// rank's own.
fn reported_entry(
    state: &State,
    load: &SchedulerLoadSnapshot,
    figures: &RankFigures,
    capacity: &Capacity,
) -> trt::SchedulerLoad {
    entry(
        figures,
        capacity,
        state
            .shared
            .loads
            .estimate(load.dp_rank, load.num_waiting_reqs),
    )
}

/// A connected rank no load tail has reported yet: a zero-filled entry with
/// the servicer's own figures for the rank (see [`rank_entries`]).
fn unreported_entry(state: &State, dp_rank: i32, capacity: &Capacity) -> trt::SchedulerLoad {
    let zero = RankFigures {
        dp_rank,
        running: 0,
        waiting: 0,
        total: 0,
        used_tokens: 0,
        usage: 0.0,
    };
    entry(
        &zero,
        capacity,
        state.shared.loads.estimate(dp_rank, i32::MAX),
    )
}
