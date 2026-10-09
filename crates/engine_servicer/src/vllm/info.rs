//! Read-only RPCs: `HealthCheck`, `GetModelInfo`, `GetServerInfo`, `GetLoads`.
//! Metadata comes from the launcher's config (what the Python servicer reads
//! off vLLM's `ModelConfig`) and from the engine handshake.

use std::{
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use engine_zmq_adapter::ZmqEngineClient;
use smg_grpc_client::vllm_proto as vllm;
use tonic::Status;
use tracing::info;

use super::{State, SERVER_TYPE};

/// How often the engine stats line is logged (vLLM's `VLLM_LOG_STATS_INTERVAL`
/// default).
pub(super) const STATS_INTERVAL: Duration = Duration::from_secs(10);

/// vLLM's periodic stats line. Its AsyncLLM logs one every interval; the
/// headless engine has no frontend to do it, so the servicer does, from the
/// piggybacked scheduler stats and its own token counters. One line marks the
/// transition to idle, then it stays quiet until there is traffic again.
pub(super) async fn log_engine_stats(state: Arc<State>) {
    let mut ticker = tokio::time::interval(STATS_INTERVAL);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let (mut last_prompt, mut last_generation) = (0u64, 0u64);
    let mut idle_logged = false;
    loop {
        ticker.tick().await;
        let Some(client) = state.engine.client.get() else {
            continue;
        };
        let prompt = state.stats.prompt_tokens.load(Ordering::Relaxed);
        let generation = state.stats.generation_tokens.load(Ordering::Relaxed);
        let prompt_delta = prompt.saturating_sub(last_prompt);
        let generation_delta = generation.saturating_sub(last_generation);
        (last_prompt, last_generation) = (prompt, generation);
        let loads = client.get_loads();
        let running: i64 = loads
            .loads
            .iter()
            .map(|load| i64::from(load.num_running_reqs))
            .sum();
        let waiting: i64 = loads
            .loads
            .iter()
            .map(|load| i64::from(load.num_waiting_reqs))
            .sum();
        let kv_usage = loads
            .loads
            .iter()
            .map(|load| load.token_usage)
            .fold(0.0_f64, f64::max);
        let idle = prompt_delta == 0 && generation_delta == 0 && running == 0 && waiting == 0;
        if idle && idle_logged {
            continue;
        }
        idle_logged = idle;
        let secs = STATS_INTERVAL.as_secs_f64();
        info!(
            "Avg prompt throughput: {:.1} tokens/s, Avg generation throughput: {:.1} tokens/s, \
             Running: {running} reqs, Waiting: {waiting} reqs, GPU KV cache usage: {:.1}%",
            prompt_delta as f64 / secs,
            generation_delta as f64 / secs,
            kv_usage * 100.0
        );
    }
}

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

/// `GetServerInfo`: this implementation's identity plus the handshake facts,
/// and the media processor while it is serving.
pub(super) async fn server_info(state: &State) -> vllm::GetServerInfoResponse {
    let mut info = server_facts(state);
    // Advertised only while the backend answers and the engine takes
    // multimodal input (a `--language-model-only` engine must not draw media
    // references), as the Python servicer advertises it.
    if let Some(gate) = state.media.as_ref() {
        if state.model.supports_vision && gate.processor.probe().await {
            info.mm_processor = gate.processor.name().to_string();
            info.mm_media_ref_schemes = gate.processor.schemes();
            info.mm_processor_source = gate.processor.source().to_string();
        }
    }
    info
}

fn server_facts(state: &State) -> vllm::GetServerInfoResponse {
    let ready = state
        .engine
        .client
        .get()
        .and_then(ZmqEngineClient::ready_response);
    let data_parallel_size = ready
        .map(|ready| i32::try_from(ready.data_parallel_size).unwrap_or(i32::MAX))
        .filter(|&size| size > 0)
        .unwrap_or_else(|| state.model.data_parallel_size.max(1));
    let model = &state.model;
    // Block size: the handshake's resolved figure once the engine is up, the
    // launcher's config value before that. Model dtype: the launcher's label
    // (`torch.bfloat16`, the spelling the Python servicer reports and PD
    // pairing compares), the handshake's short form only as a fallback.
    let block_size = ready
        .map(|ready| i32::try_from(ready.block_size).unwrap_or(i32::MAX))
        .filter(|&size| size > 0)
        .unwrap_or(model.block_size);
    let model_dtype = Some(model.model_dtype.clone())
        .filter(|dtype| !dtype.is_empty())
        .or_else(|| ready.map(|ready| ready.dtype.as_str().to_string()))
        .unwrap_or_default();
    vllm::GetServerInfoResponse {
        active_requests: state.active_requests(),
        uptime_seconds: state.started.elapsed().as_secs_f64(),
        server_type: SERVER_TYPE.to_string(),
        data_parallel_size,
        kv_connector: model.kv_connector.clone(),
        kv_role: model.kv_role.clone(),
        kv_engine_id: model.kv_engine_id.clone(),
        kv_cache_dtype: model.kv_cache_dtype.clone(),
        attention_backend: model.attention_backend.clone(),
        pairing_protocol: model.pairing_protocol.clone(),
        block_size,
        model_dtype,
        shm_namespace_id: model.shm_namespace_id.clone(),
        mm_device_do_normalize: model.mm_device_do_normalize,
        mm_item_limits: model.mm_item_limits.clone(),
        max_num_seqs: max_num_seqs(state),
        ..Default::default()
    }
}

/// The scheduler's running window: the launcher's `--max-num-seqs`, else the
/// handshake's; 0 before the engine is up on a launcher that reported none.
fn max_num_seqs(state: &State) -> i32 {
    if state.model.max_num_seqs > 0 {
        return state.model.max_num_seqs;
    }
    state
        .engine
        .client
        .get()
        .and_then(ZmqEngineClient::ready_response)
        .map(|ready| i32::try_from(ready.max_num_seqs).unwrap_or(i32::MAX))
        .unwrap_or_default()
}

/// `GetLoads`: the per-rank load piggybacked on engine output, in the gRPC
/// response shape, with the handshake's capacity figures and, from this
/// servicer's own bookkeeping ([`crate::load_tracker`]), the queued
/// token-work, generation throughput and hit rate that vLLM's stats do not
/// carry. The servicer forwards to one engine process, so those three go on
/// the first rank's entry.
pub(super) fn loads(state: &State) -> Result<vllm::GetLoadsResponse, Status> {
    let client = state.engine()?;
    let ready = client.ready_response();
    let max_running_requests = max_num_seqs(state);
    let max_total_num_tokens = ready
        .and_then(|ready| ready.kv_cache_size_tokens)
        .map(|tokens| i32::try_from(tokens).unwrap_or(i32::MAX))
        .unwrap_or_default();
    let snapshot = client.get_loads();
    let mut dp_rank_count = snapshot.dp_rank_count;
    let mut loads: Vec<vllm::SchedulerLoad> = snapshot
        .loads
        .into_iter()
        .map(|load| {
            let num_used_tokens = (load.token_usage * f64::from(max_total_num_tokens)).round();
            vllm::SchedulerLoad {
                dp_rank: load.dp_rank,
                num_running_reqs: load.num_running_reqs,
                num_waiting_reqs: load.num_waiting_reqs,
                num_total_reqs: load.num_running_reqs.saturating_add(load.num_waiting_reqs),
                num_used_tokens: num_used_tokens.clamp(0.0, f64::from(i32::MAX)) as i32,
                token_usage: load.token_usage,
                utilization: load.token_usage,
                max_running_requests,
                max_total_num_tokens,
                ..Default::default()
            }
        })
        .collect();
    if let Some(first) = loads.first_mut() {
        let estimate = state.loads.estimate(first.num_waiting_reqs);
        first.num_waiting_uncached_tokens = estimate.queued_token_work;
        first.gen_throughput = estimate.gen_throughput;
        first.cache_hit_rate = estimate.cache_hit_rate;
    }
    if loads.is_empty() {
        // No output batch has carried a snapshot yet. The Python servicer
        // reports a zero-filled entry per rank, and the Router reads an
        // empty list as no report at all.
        let ranks = ready
            .and_then(|ready| usize::try_from(ready.data_parallel_size).ok())
            .filter(|&ranks| ranks > 0)
            .or_else(|| usize::try_from(state.model.data_parallel_size).ok())
            .unwrap_or(1)
            .max(1);
        loads = (0..ranks)
            .map(|rank| vllm::SchedulerLoad {
                dp_rank: i32::try_from(rank).unwrap_or(i32::MAX),
                max_running_requests,
                max_total_num_tokens,
                ..Default::default()
            })
            .collect();
        dp_rank_count = loads.len().try_into().unwrap_or_default();
    }
    Ok(vllm::GetLoadsResponse {
        timestamp: chrono::Utc::now().to_rfc3339(),
        // The engine's version, as the Python servicer reports vLLM's.
        version: ready
            .map(|ready| ready.vllm_version.clone())
            .filter(|version| !version.is_empty())
            .unwrap_or_else(|| "1".to_string()),
        dp_rank_count,
        loads,
    })
}
