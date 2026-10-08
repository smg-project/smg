//! The pushed load record (`common::EngineLoad`, on `KvEventBatch.load`)
//! against the engines' `GetLoads` shapes: what a servicer fills from its
//! engine's per-rank report, and what the gateway reads back into its
//! per-rank snapshot, so the `smg_engine_*` gauges and `GET /loads` show the
//! same under pushes as under a poll.
//!
//! The routing core (running and waiting requests, the queued token-work,
//! KV usage, generation rate, the admission window) rides every record; the
//! telemetry (cache hit rate, used and total tokens, and SGLang's memory,
//! queue, speculative, LoRA and disaggregation sections, which TokenSpeed
//! shares for memory and queues and vLLM has none of) is filled when the
//! engine reports it and left unset otherwise. `waiting_uncached_tokens` is
//! the servicer's to set: whether the figure is an estimate for one rank or
//! the engine's own is a per-servicer rule.

use openai_protocol::worker::{
    EngineMemoryMetricsSnapshot, EngineQueueMetricsSnapshot, SchedulerLoadSnapshot,
};

use crate::{common_proto as common, sglang_proto, tokenspeed_proto, vllm_proto};

fn unsigned(value: i32) -> u32 {
    u32::try_from(value).unwrap_or(0)
}

/// The record's core from the figures every engine reports.
fn core(
    running: i32,
    waiting: i32,
    token_usage: f64,
    gen_throughput: f64,
    max_running_requests: i32,
) -> common::EngineLoad {
    common::EngineLoad {
        running_requests: unsigned(running),
        waiting_requests: unsigned(waiting),
        token_usage,
        gen_throughput,
        max_running_requests: unsigned(max_running_requests),
        ..Default::default()
    }
}

/// vLLM's report: the core, the cache hit rate and the token counts; no
/// sections on its wire.
impl From<&vllm_proto::SchedulerLoad> for common::EngineLoad {
    fn from(load: &vllm_proto::SchedulerLoad) -> Self {
        Self {
            cache_hit_rate: Some(load.cache_hit_rate),
            num_used_tokens: Some(load.num_used_tokens),
            max_total_num_tokens: Some(load.max_total_num_tokens),
            ..core(
                load.num_running_reqs,
                load.num_waiting_reqs,
                load.token_usage,
                load.gen_throughput,
                load.max_running_requests,
            )
        }
    }
}

/// SGLang's report: the core, the cache hit rate, the token counts and
/// every optional section it carries.
impl From<&sglang_proto::SchedulerLoad> for common::EngineLoad {
    fn from(load: &sglang_proto::SchedulerLoad) -> Self {
        Self {
            cache_hit_rate: Some(load.cache_hit_rate),
            num_used_tokens: Some(load.num_used_tokens),
            max_total_num_tokens: Some(load.max_total_num_tokens),
            memory: load.memory.as_ref().map(|memory| common::EngineMemory {
                weight_gb: memory.weight_gb,
                kv_cache_gb: memory.kv_cache_gb,
                graph_gb: memory.graph_gb,
                token_capacity: memory.token_capacity,
            }),
            queues: load.queues.as_ref().map(|queues| common::EngineQueues {
                waiting: queues.waiting,
                grammar: queues.grammar,
                paused: queues.paused,
                retracted: queues.retracted,
            }),
            speculative: load
                .speculative
                .as_ref()
                .map(|speculative| common::EngineSpeculative {
                    accept_length: speculative.accept_length,
                    accept_rate: speculative.accept_rate,
                }),
            lora: load.lora.as_ref().map(|lora| common::EngineLora {
                slots_used: lora.slots_used,
                slots_total: lora.slots_total,
                utilization: lora.utilization,
            }),
            disaggregation: load.disaggregation.as_ref().map(|disagg| {
                common::EngineDisaggregation {
                    mode: disagg.mode.clone(),
                    prefill_prealloc_queue_reqs: disagg.prefill_prealloc_queue_reqs,
                    prefill_inflight_queue_reqs: disagg.prefill_inflight_queue_reqs,
                    decode_prealloc_queue_reqs: disagg.decode_prealloc_queue_reqs,
                    decode_transfer_queue_reqs: disagg.decode_transfer_queue_reqs,
                    decode_retracted_queue_reqs: disagg.decode_retracted_queue_reqs,
                    kv_transfer_speed_gb_s: disagg.kv_transfer_speed_gb_s,
                    kv_transfer_latency_ms: disagg.kv_transfer_latency_ms,
                }
            }),
            ..core(
                load.num_running_reqs,
                load.num_waiting_reqs,
                load.token_usage,
                load.gen_throughput,
                load.max_running_requests,
            )
        }
    }
}

/// TokenSpeed's report: the core, the cache hit rate, the token counts, and
/// the memory and queue sections.
impl From<&tokenspeed_proto::SchedulerLoad> for common::EngineLoad {
    fn from(load: &tokenspeed_proto::SchedulerLoad) -> Self {
        Self {
            cache_hit_rate: Some(load.cache_hit_rate),
            num_used_tokens: Some(load.num_used_tokens),
            max_total_num_tokens: Some(load.max_total_num_tokens),
            memory: load.memory.as_ref().map(|memory| common::EngineMemory {
                weight_gb: memory.weight_gb,
                kv_cache_gb: memory.kv_cache_gb,
                graph_gb: memory.graph_gb,
                token_capacity: memory.token_capacity,
            }),
            queues: load.queues.as_ref().map(|queues| common::EngineQueues {
                waiting: queues.waiting,
                grammar: queues.grammar,
                paused: queues.paused,
                retracted: queues.retracted,
            }),
            ..core(
                load.num_running_reqs,
                load.num_waiting_reqs,
                load.token_usage,
                load.gen_throughput,
                load.max_running_requests,
            )
        }
    }
}

/// The per-rank snapshot a poll would have produced from what the record
/// carries, the way the engines' `GetLoads` responses convert: the two
/// canonical disaggregation queue depths roll the per-stage counters up
/// (prefill = prealloc + inflight, decode = prealloc + transfer +
/// retracted), `utilization` is the KV usage, `num_total_reqs` the running
/// and waiting sum. The rank is the caller's; telemetry the record leaves
/// unset stays at the type's default, which the gateway merges with what it
/// knew from the last record that carried it.
impl From<&common::EngineLoad> for SchedulerLoadSnapshot {
    fn from(record: &common::EngineLoad) -> Self {
        let signed = |value: u32| i32::try_from(value).unwrap_or(i32::MAX);
        let disagg = record.disaggregation.as_ref();
        Self {
            num_running_reqs: signed(record.running_requests),
            num_waiting_reqs: signed(record.waiting_requests),
            num_waiting_uncached_tokens: record.waiting_uncached_tokens.map_or(0, signed),
            num_total_reqs: signed(
                record
                    .running_requests
                    .saturating_add(record.waiting_requests),
            ),
            num_used_tokens: record.num_used_tokens.unwrap_or(0),
            max_total_num_tokens: record.max_total_num_tokens.unwrap_or(0),
            token_usage: record.token_usage,
            gen_throughput: record.gen_throughput,
            cache_hit_rate: record.cache_hit_rate.unwrap_or(0.0),
            utilization: record.token_usage,
            max_running_requests: signed(record.max_running_requests),
            memory: record
                .memory
                .as_ref()
                .map(|memory| EngineMemoryMetricsSnapshot {
                    weight_gb: memory.weight_gb,
                    kv_cache_gb: memory.kv_cache_gb,
                    graph_gb: memory.graph_gb,
                    token_capacity: memory.token_capacity,
                }),
            queues: record
                .queues
                .as_ref()
                .map(|queues| EngineQueueMetricsSnapshot {
                    waiting: queues.waiting,
                    grammar: queues.grammar,
                    paused: queues.paused,
                    retracted: queues.retracted,
                }),
            kv_transfer_latency_ms: disagg.map(|d| d.kv_transfer_latency_ms),
            kv_transfer_speed_gb_s: disagg.map(|d| d.kv_transfer_speed_gb_s),
            prefill_queue_reqs: disagg.map(|d| {
                d.prefill_prealloc_queue_reqs
                    .saturating_add(d.prefill_inflight_queue_reqs)
            }),
            decode_queue_reqs: disagg.map(|d| {
                d.decode_prealloc_queue_reqs
                    .saturating_add(d.decode_transfer_queue_reqs)
                    .saturating_add(d.decode_retracted_queue_reqs)
            }),
            disagg_mode: disagg.map(|d| d.mode.clone()),
            ..Default::default()
        }
    }
}

/// Telemetry fields of the record cleared: what an event batch carries
/// between heartbeats.
pub fn core_only(record: &mut common::EngineLoad) {
    record.cache_hit_rate = None;
    record.num_used_tokens = None;
    record.max_total_num_tokens = None;
    record.memory = None;
    record.queues = None;
    record.speculative = None;
    record.lora = None;
    record.disaggregation = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A SGLang rank with every section set, as the Python servicer reports
    /// one. `utilization`, `num_total_reqs` and the queued token-work are
    /// what the record derives (KV usage, the sum, the servicer's own), so
    /// the poll's values are set to those for the comparison.
    fn sglang_rank() -> sglang_proto::SchedulerLoad {
        sglang_proto::SchedulerLoad {
            dp_rank: 2,
            num_running_reqs: 7,
            num_waiting_reqs: 3,
            num_total_reqs: 10,
            num_used_tokens: 12_000,
            max_total_num_tokens: 64_000,
            token_usage: 0.1875,
            gen_throughput: 1_500.5,
            cache_hit_rate: 0.42,
            utilization: 0.1875,
            max_running_requests: 128,
            num_waiting_uncached_tokens: 0,
            memory: Some(sglang_proto::MemoryMetrics {
                weight_gb: 15.0,
                kv_cache_gb: 40.0,
                graph_gb: 1.5,
                token_capacity: 64_000,
            }),
            speculative: Some(sglang_proto::SpeculativeMetrics {
                accept_length: 2.5,
                accept_rate: 0.8,
            }),
            lora: Some(sglang_proto::LoRaMetrics {
                slots_used: 2,
                slots_total: 8,
                utilization: 0.25,
            }),
            disaggregation: Some(sglang_proto::DisaggregationMetrics {
                mode: "prefill".to_string(),
                prefill_prealloc_queue_reqs: 4,
                prefill_inflight_queue_reqs: 5,
                decode_prealloc_queue_reqs: 1,
                decode_transfer_queue_reqs: 2,
                decode_retracted_queue_reqs: 3,
                kv_transfer_speed_gb_s: 12.0,
                kv_transfer_latency_ms: 3.5,
            }),
            queues: Some(sglang_proto::QueueMetrics {
                waiting: 3,
                grammar: 1,
                paused: 0,
                retracted: 2,
            }),
        }
    }

    fn pushed(record: &common::EngineLoad, dp_rank: i32) -> SchedulerLoadSnapshot {
        let mut snapshot = SchedulerLoadSnapshot::from(record);
        snapshot.dp_rank = dp_rank;
        snapshot
    }

    #[test]
    fn a_sglang_rank_reads_the_same_from_the_record_as_from_the_poll() {
        let rank = sglang_rank();
        let polled = SchedulerLoadSnapshot::from(rank.clone());
        let record = common::EngineLoad::from(&rank);
        assert_eq!(record.cache_hit_rate, Some(0.42));
        assert_eq!(
            record.disaggregation.as_ref().map(|d| d.mode.as_str()),
            Some("prefill")
        );
        assert_eq!(
            format!("{polled:?}"),
            format!("{:?}", pushed(&record, rank.dp_rank))
        );
    }

    #[test]
    fn a_tokenspeed_rank_reads_the_same_from_the_record_as_from_the_poll() {
        let rank = tokenspeed_proto::SchedulerLoad {
            dp_rank: 1,
            num_running_reqs: 4,
            num_waiting_reqs: 2,
            num_total_reqs: 6,
            num_used_tokens: 8_192,
            max_total_num_tokens: 32_768,
            max_running_requests: 64,
            num_waiting_uncached_tokens: 0,
            token_usage: 0.25,
            gen_throughput: 900.0,
            cache_hit_rate: 0.6,
            utilization: 0.25,
            memory: Some(tokenspeed_proto::MemoryMetrics {
                weight_gb: 7.0,
                kv_cache_gb: 20.0,
                graph_gb: 0.5,
                token_capacity: 32_768,
            }),
            queues: Some(tokenspeed_proto::QueueMetrics {
                waiting: 2,
                grammar: 0,
                paused: 0,
                retracted: 1,
            }),
        };
        let polled = SchedulerLoadSnapshot::from(rank);
        let record = common::EngineLoad::from(&rank);
        assert!(record.memory.is_some() && record.disaggregation.is_none());
        assert_eq!(
            format!("{polled:?}"),
            format!("{:?}", pushed(&record, rank.dp_rank))
        );
    }

    #[test]
    fn a_vllm_rank_reads_the_same_from_the_record_as_from_the_poll() {
        let rank = vllm_proto::SchedulerLoad {
            dp_rank: 0,
            num_running_reqs: 9,
            num_waiting_reqs: 0,
            num_total_reqs: 9,
            num_used_tokens: 100_000,
            max_total_num_tokens: 676_144,
            token_usage: 0.1479,
            gen_throughput: 2_000.0,
            cache_hit_rate: 0.33,
            utilization: 0.1479,
            max_running_requests: 256,
            num_waiting_uncached_tokens: 0,
        };
        let polled = SchedulerLoadSnapshot::from(rank);
        let record = common::EngineLoad::from(&rank);
        assert!(record.memory.is_none() && record.queues.is_none());
        assert_eq!(
            format!("{polled:?}"),
            format!("{:?}", pushed(&record, rank.dp_rank))
        );
    }

    /// Absent telemetry is absent, not zero: a core-only record converts
    /// with the type's defaults for the caller to merge, never with a
    /// section it did not carry.
    #[test]
    fn a_core_only_record_carries_no_sections() {
        let mut record = common::EngineLoad::from(&sglang_rank());
        core_only(&mut record);
        let snapshot = SchedulerLoadSnapshot::from(&record);
        assert!(record.cache_hit_rate.is_none() && record.memory.is_none());
        assert!(snapshot.memory.is_none() && snapshot.disagg_mode.is_none());
        assert_eq!(
            (snapshot.num_running_reqs, snapshot.max_total_num_tokens),
            (7, 0)
        );
    }
}
