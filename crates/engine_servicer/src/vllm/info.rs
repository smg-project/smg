//! Read-only RPCs: `HealthCheck`, `GetModelInfo`, `GetServerInfo`, `GetLoads`.
//! Metadata comes from the launcher's config (what the Python servicer reads
//! off vLLM's `ModelConfig`) and from the engine handshake.

use std::{sync::Arc, time::Duration};

use smg_grpc_client::vllm_proto as vllm;
use tonic::Status;
use tracing::info;

use super::{State, HEALTHY, SERVER_TYPE};
use crate::load_report::{rank_ids, RankFigures};

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
        let Some(client) = state.shared.engine.client.get() else {
            continue;
        };
        let prompt = state.shared.metrics.prompt_tokens_total();
        let generation = state.shared.metrics.generation_tokens_total();
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
        healthy: state.shared.is_serving(),
        message: state.shared.health_message(HEALTHY).to_string(),
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
        if let Some(ready) = state.shared.ready() {
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
    let ready = state.shared.ready();
    // The ranks this servicer serves (the gateway expands the worker into
    // that many under `--dp-aware`): the connected engines, as `GetLoads`
    // counts them, not the handshake's figure.
    let data_parallel_size =
        i32::try_from(state.shared.rank_count(state.model.data_parallel_size)).unwrap_or(i32::MAX);
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
        uptime_seconds: state.shared.started.elapsed().as_secs_f64(),
        server_type: SERVER_TYPE.to_string(),
        data_parallel_size,
        kv_connector: model.kv_connector.clone(),
        kv_role: model.kv_role.clone(),
        kv_engine_id: model.kv_engine_id.clone(),
        kv_cache_dtype: model.kv_cache_dtype.clone(),
        attention_backend: model.attention_backend.clone(),
        pairing_protocol: model.pairing_protocol.clone(),
        block_size,
        multimodal_encoder_dtype: match model_dtype.strip_prefix("torch.").unwrap_or(&model_dtype) {
            dtype @ ("float32" | "float16" | "bfloat16") => dtype.to_string(),
            _ => String::new(),
        },
        model_dtype,
        shm_namespace_id: model.shm_namespace_id.clone(),
        mm_device_do_normalize: model.mm_device_do_normalize,
        mm_item_limits: model.mm_item_limits.clone(),
        mm_encoder_budget: model.mm_encoder_budget,
        max_num_seqs: state.shared.running_window(model.max_num_seqs),
        engine_reasoning_parser: model.engine_reasoning_parser.clone(),
        constrained_decoding_start: model.constrained_decoding_start.clone(),
        ..Default::default()
    }
}

/// `GetLoads`: the per-rank load piggybacked on engine output, in the gRPC
/// response shape, with the handshake's capacity figures and, from this
/// servicer's own bookkeeping ([`crate::load_tracker`]), the queued
/// token-work, generation throughput and hit rate that vLLM's stats do not
/// carry, each rank's own: the servicer picks the rank a request goes to
/// and sees what that rank cached and streamed for it.
pub(super) fn loads(state: &State) -> Result<vllm::GetLoadsResponse, Status> {
    let client = state.shared.engine()?;
    let ready = client.ready_response();
    let max_running_requests = state.shared.running_window(state.model.max_num_seqs);
    let max_total_num_tokens = state.shared.kv_capacity_tokens();
    let snapshot = client.get_loads();
    let mut dp_rank_count = snapshot.dp_rank_count;
    let mut loads: Vec<vllm::SchedulerLoad> = snapshot
        .loads
        .into_iter()
        .map(|load| {
            let figures = RankFigures::of(&load, max_total_num_tokens);
            let estimate = state.loads.estimate(load.dp_rank, load.num_waiting_reqs);
            vllm::SchedulerLoad {
                dp_rank: figures.dp_rank,
                num_running_reqs: figures.running,
                num_waiting_reqs: figures.waiting,
                num_total_reqs: figures.total,
                num_used_tokens: figures.used_tokens,
                token_usage: figures.usage,
                utilization: figures.usage,
                max_running_requests,
                max_total_num_tokens,
                num_waiting_uncached_tokens: estimate.queued_token_work,
                gen_throughput: estimate.gen_throughput,
                cache_hit_rate: estimate.cache_hit_rate,
            }
        })
        .collect();
    if loads.is_empty() {
        // No output batch has carried a snapshot yet. The Python servicer
        // reports a zero-filled entry per rank, and the Router reads an
        // empty list as no report at all.
        let ranks = state.shared.rank_count(state.model.data_parallel_size);
        loads = rank_ids(ranks)
            .map(|dp_rank| {
                // The servicer's own figures do not wait for the engine's
                // first snapshot: what this rank cached and streamed so far,
                // and the token-work queued to it. The engine has not said
                // how many requests it holds waiting, and before its first
                // snapshot that is every request submitted to the rank and
                // not started, so all of them count (an overestimate by
                // what is already in prefill, better than a zero that reads
                // as known-empty).
                let estimate = state.loads.estimate(dp_rank, i32::MAX);
                vllm::SchedulerLoad {
                    dp_rank,
                    max_running_requests,
                    max_total_num_tokens,
                    num_waiting_uncached_tokens: estimate.queued_token_work,
                    gen_throughput: estimate.gen_throughput,
                    cache_hit_rate: estimate.cache_hit_rate,
                    ..Default::default()
                }
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
