//! The engine load record (`common_proto::EngineLoad`, pushed on
//! `KvEventBatch.load`) read as the per-rank snapshot of a load report, and
//! as the record the worker's load report and its liveness read
//! ([`LoadRecord`]). `smg-grpc-client` holds the other direction, each
//! engine's `GetLoads` shape into the record, and the cases that hold the two
//! together.

use std::time::Duration;

use crate::{
    common_proto as common,
    worker::{
        EngineMemoryMetricsSnapshot, EngineQueueMetricsSnapshot, LoadRecord, LoadTelemetry,
        SchedulerLoadSnapshot,
    },
};

/// The per-rank snapshot a poll would have produced from what the record
/// carries, the way the engines' `GetLoads` responses convert: the two
/// canonical disaggregation queue depths roll the per-stage counters up
/// (prefill = prealloc + inflight, decode = prealloc + transfer +
/// retracted), `utilization` is the KV usage, `num_total_reqs` the running
/// and waiting sum. The rank is the caller's; telemetry the record leaves
/// unset stays at the type's default, which the gateway merges with what it
/// knew from the last record that carried it.
impl From<&common::EngineLoad> for SchedulerLoadSnapshot {
    /// Read input capacity directly, including absence after the samples expire.
    fn from(record: &common::EngineLoad) -> Self {
        let signed = |value: u32| i32::try_from(value).unwrap_or(i32::MAX);
        let disagg = record.disaggregation.as_ref();
        Self {
            active_token_usage: record.active_token_usage,
            num_resident_tokens: record
                .num_resident_tokens
                .map(|tokens| i64::try_from(tokens).unwrap_or(i64::MAX)),
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
            prefill_throughput: record.prefill_throughput,
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

/// The record as the worker's load report and its liveness read it.
///
/// The readers are named apart from the record's fields: a generated
/// accessor of a field's name would shadow the trait's method on this type
/// and answer the field's default where the trait says absent.
impl LoadRecord for common::EngineLoad {
    #[inline]
    fn age(&self) -> Duration {
        Duration::from_millis(u64::from(self.age_ms))
    }

    #[inline]
    fn telemetry(&self) -> LoadTelemetry {
        let mut carried = LoadTelemetry::empty();
        carried.set(LoadTelemetry::CACHE_HIT_RATE, self.cache_hit_rate.is_some());
        carried.set(LoadTelemetry::USED_TOKENS, self.num_used_tokens.is_some());
        carried.set(
            LoadTelemetry::TOTAL_TOKENS,
            self.max_total_num_tokens.is_some(),
        );
        carried.set(LoadTelemetry::MEMORY, self.memory.is_some());
        carried.set(LoadTelemetry::QUEUES, self.queues.is_some());
        carried.set(LoadTelemetry::DISAGGREGATION, self.disaggregation.is_some());
        carried
    }

    #[inline]
    fn snapshot(&self) -> SchedulerLoadSnapshot {
        SchedulerLoadSnapshot::from(self)
    }

    #[inline]
    fn engine_silence(&self) -> Option<Duration> {
        self.engine_silence_ms
            .map(|ms| Duration::from_millis(u64::from(ms)))
    }

    #[inline]
    fn reports_every_step(&self) -> bool {
        self.engine_reports_steps.unwrap_or(false)
    }

    #[inline]
    fn prefill_pending(&self) -> Option<u64> {
        self.prefill_pending_tokens.map(u64::from)
    }
}
