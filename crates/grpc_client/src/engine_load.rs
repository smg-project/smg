//! The pushed load record (`common::EngineLoad`, on `KvEventBatch.load`)
//! against the engines' `GetLoads` shapes: what a servicer fills from its
//! engine's per-rank report. What the gateway reads back into its per-rank
//! snapshot is the worker crate's conversion, beside both of its types; the
//! cases here hold the fixed fields together so those figures agree under
//! pushes and polls.
//!
//! The routing core (running and waiting requests, the queued token-work,
//! KV usage, generation rate, the admission window) rides every record; the
//! telemetry (cache hit rate, used and total tokens, and SGLang's memory,
//! queue, speculative, LoRA and disaggregation sections, which TokenSpeed
//! shares for memory and queues and vLLM has none of) is filled when the
//! engine reports it and left unset otherwise. `waiting_uncached_tokens` is
//! the servicer's to set: whether the figure is an estimate for one rank or
//! the engine's own is a per-servicer rule.

use crate::{common_proto as common, sglang_proto, tokenspeed_proto, trtllm_proto, vllm_proto};

/// The sections a `GetLoads` poll asks for. The optional sections beyond
/// `core` (the disaggregation, queue and memory figures) are dropped by the
/// engines that do not report them, so asking for every one is always safe
/// and the routing core every reader needs is unaffected.
pub const LOAD_SECTIONS: [&str; 4] = ["core", "disagg", "queues", "memory"];

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
    /// Preserve optional input capacity when converting a poll to a pushed record.
    fn from(load: &vllm_proto::SchedulerLoad) -> Self {
        Self {
            values: load.values.clone(),
            prefill_throughput: load.prefill_throughput,
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

/// The TensorRT-LLM service's report: the same shape as vLLM's, no sections.
impl From<&trtllm_proto::SchedulerLoad> for common::EngineLoad {
    fn from(load: &trtllm_proto::SchedulerLoad) -> Self {
        Self {
            prefill_throughput: load.prefill_throughput,
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
            values: load.values.clone(),
            prefill_throughput: load.prefill_throughput,
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
            values: load.values.clone(),
            prefill_throughput: load.prefill_throughput,
            active_token_usage: load.active_token_usage,
            num_resident_tokens: load
                .num_resident_tokens
                .map(|tokens| u64::try_from(tokens).unwrap_or(0)),
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
    use std::time::Duration;

    use openai_protocol::worker::{LoadRecord, LoadTelemetry, NamedValue, SchedulerLoadSnapshot};
    use prost::Message;

    use super::*;

    /// Every engine's poll carries the measured input capacity the same way
    /// into the record and back to the snapshot, present or absent.
    #[test]
    fn every_engine_carries_the_measured_input_capacity_alike() {
        for rate in [Some(4_321.0), None] {
            let sglang = sglang_proto::SchedulerLoad {
                prefill_throughput: rate,
                ..Default::default()
            };
            let tokenspeed = tokenspeed_proto::SchedulerLoad {
                prefill_throughput: rate,
                ..Default::default()
            };
            let vllm = vllm_proto::SchedulerLoad {
                prefill_throughput: rate,
                ..Default::default()
            };
            for record in [
                common::EngineLoad::from(&sglang),
                common::EngineLoad::from(&tokenspeed),
                common::EngineLoad::from(&vllm),
            ] {
                assert_eq!(record.prefill_throughput, rate);
                assert_eq!(
                    SchedulerLoadSnapshot::from(&record).prefill_throughput,
                    rate
                );
            }
            assert_eq!(SchedulerLoadSnapshot::from(sglang).prefill_throughput, rate);
            assert_eq!(
                SchedulerLoadSnapshot::from(tokenspeed).prefill_throughput,
                rate
            );
        }
    }

    fn value(name: &str, unit: &str, value: f64, age_ms: Option<u32>) -> common::NamedValue {
        common::NamedValue {
            name: name.to_string(),
            unit: unit.to_string(),
            value,
            age_ms,
        }
    }

    /// A rank's named values are one list on its polled answer and on its
    /// pushed record: under the record's field number on every wire, and
    /// read alike from either, each entry with its unit and its own age.
    /// An answer with no list states no value.
    #[test]
    fn a_ranks_named_values_read_alike_from_its_poll_and_from_its_pushed_record() {
        let stated = vec![
            value("queue_age", "ms", 6.5, Some(37)),
            value("read_rate", "items/s", 0.125, None),
        ];
        let read = [
            NamedValue {
                name: "queue_age".into(),
                unit: "ms".into(),
                value: 6.5,
                age_ms: Some(37),
            },
            NamedValue {
                name: "read_rate".into(),
                unit: "items/s".into(),
                value: 0.125,
                age_ms: None,
            },
        ];
        let on_the_record = common::EngineLoad {
            values: stated.clone(),
            ..Default::default()
        }
        .encode_to_vec();
        macro_rules! wire {
            ($proto:ident) => {{
                let rank = $proto::SchedulerLoad {
                    values: stated.clone(),
                    ..Default::default()
                };
                assert_eq!(rank.encode_to_vec(), on_the_record);
                let record = common::EngineLoad::from(&rank);
                assert_eq!(record.values, stated);
                assert_eq!(SchedulerLoadSnapshot::from(&record).values, read);
                assert_eq!(SchedulerLoadSnapshot::from(rank).values, read);
                let old = $proto::SchedulerLoad::default();
                assert!(common::EngineLoad::from(&old).values.is_empty());
                assert!(SchedulerLoadSnapshot::from(old).values.is_empty());
            }};
        }
        wire!(sglang_proto);
        wire!(tokenspeed_proto);
        wire!(vllm_proto);
    }

    /// A polled list is read whole, and the bounds a report holds it to keep
    /// what a pushed record of the same entries keeps, so what leaves the
    /// polled list is counted as what the record's read leaves out is. The
    /// bound list holds no more room than the record's.
    #[test]
    fn a_polled_list_past_the_bounds_keeps_what_its_pushed_record_keeps() {
        let mut stated: Vec<_> = (0..70)
            .map(|at| value(&format!("ready_{at}"), "items", f64::from(at), None))
            .collect();
        stated[3] = value("", "items", 1.0, None);
        stated[5] = value("ready_4", "items", 2.0, None);
        stated[7] = value("ready_7", "items", f64::INFINITY, None);
        stated[9] = value(&"n".repeat(64), "items", 3.0, None);
        stated[11] = value("ready_11", &"u".repeat(32), 4.0, None);
        macro_rules! wire {
            ($proto:ident) => {{
                let rank = $proto::SchedulerLoad {
                    values: stated.clone(),
                    ..Default::default()
                };
                let record = common::EngineLoad::from(&rank);
                let pushed = SchedulerLoadSnapshot::from(&record).values;
                assert_eq!(pushed.len(), 59);
                let mut polled = SchedulerLoadSnapshot::from(rank);
                assert_eq!(polled.values.len(), 70);
                assert_eq!(polled.bound_values(), 11);
                assert_eq!(polled.values, pushed);
                assert!(polled.values.capacity() <= pushed.capacity());
                assert_eq!(record.values_sent() - pushed.len(), 11);
            }};
        }
        wire!(sglang_proto);
        wire!(tokenspeed_proto);
        wire!(vllm_proto);
    }

    /// Polling and core-only push records preserve optional input capacity.
    #[test]
    fn prefill_capacity_survives_poll_push_and_core_only_and_clears_when_absent() {
        for rate in [Some(1234.0), None] {
            let load = vllm_proto::SchedulerLoad {
                prefill_throughput: rate,
                gen_throughput: 0.5,
                ..Default::default()
            };
            let mut pushed = common::EngineLoad::from(&load);
            let polled = SchedulerLoadSnapshot::from(load);
            core_only(&mut pushed);
            let pushed = SchedulerLoadSnapshot::from(&pushed);
            assert_eq!(polled.prefill_throughput, rate);
            assert_eq!(pushed.prefill_throughput, rate);
            assert_eq!(pushed.gen_throughput, 0.5);
        }
    }

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
            prefill_throughput: None,
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
            values: Vec::new(),
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
        assert!(polled.values.is_empty() && record.values.is_empty());
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
            active_token_usage: Some(0.125),
            gen_throughput: 900.0,
            prefill_throughput: None,
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
            num_resident_tokens: Some(70_000),
            values: Vec::new(),
        };
        let polled = SchedulerLoadSnapshot::from(rank.clone());
        let record = common::EngineLoad::from(&rank);
        assert!(polled.values.is_empty() && record.values.is_empty());
        assert!(record.memory.is_some() && record.disaggregation.is_none());
        assert_eq!(record.num_resident_tokens, Some(70_000));
        assert_eq!(polled.num_resident_tokens, Some(70_000));
        assert_eq!(
            format!("{polled:?}"),
            format!("{:?}", pushed(&record, rank.dp_rank))
        );
    }

    /// Legacy vLLM loads without measured capacity remain identical on both wires.
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
            prefill_throughput: None,
            cache_hit_rate: 0.33,
            utilization: 0.1479,
            max_running_requests: 256,
            num_waiting_uncached_tokens: 0,
            values: Vec::new(),
        };
        let polled = SchedulerLoadSnapshot::from(rank.clone());
        let record = common::EngineLoad::from(&rank);
        assert!(polled.values.is_empty() && record.values.is_empty());
        assert!(record.memory.is_none() && record.queues.is_none());
        assert_eq!(
            format!("{polled:?}"),
            format!("{:?}", pushed(&record, rank.dp_rank))
        );
    }

    /// A TensorRT-LLM rank reads the same on both wires, the measured
    /// capacity included.
    #[test]
    fn a_trtllm_rank_reads_the_same_from_the_record_as_from_the_poll() {
        let rank = trtllm_proto::SchedulerLoad {
            dp_rank: 1,
            num_running_reqs: 3,
            num_waiting_reqs: 2,
            num_total_reqs: 5,
            num_used_tokens: 640,
            max_total_num_tokens: 6_400,
            token_usage: 0.1,
            gen_throughput: 800.0,
            cache_hit_rate: 0.5,
            utilization: 0.1,
            max_running_requests: 32,
            num_waiting_uncached_tokens: 0,
            prefill_throughput: Some(12_000.0),
        };
        let polled = SchedulerLoadSnapshot::from(rank);
        let record = common::EngineLoad::from(&rank);
        assert!(record.memory.is_none() && record.queues.is_none());
        assert!(polled.values.is_empty() && record.values.is_empty());
        assert_eq!(record.prefill_throughput, Some(12_000.0));
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
        assert!(record.values.is_empty() && snapshot.values.is_empty());
        assert!(record.cache_hit_rate.is_none() && record.memory.is_none());
        assert!(snapshot.memory.is_none() && snapshot.disagg_mode.is_none());
        assert_eq!(
            (snapshot.num_running_reqs, snapshot.max_total_num_tokens),
            (7, 0)
        );
    }

    /// Under the worker's reader each telemetry section a record carries is
    /// its own flag, a record cut to its core carries none, and the age and
    /// the liveness figures are the wire's fields.
    #[test]
    fn a_record_reads_as_the_load_report_reads_it() {
        let only = |set: fn(&mut common::EngineLoad)| {
            let mut record = common::EngineLoad::default();
            set(&mut record);
            record.telemetry()
        };
        assert_eq!(
            only(|r| r.cache_hit_rate = Some(0.5)),
            LoadTelemetry::CACHE_HIT_RATE
        );
        assert_eq!(
            only(|r| r.num_used_tokens = Some(1)),
            LoadTelemetry::USED_TOKENS
        );
        assert_eq!(
            only(|r| r.max_total_num_tokens = Some(1)),
            LoadTelemetry::TOTAL_TOKENS
        );
        assert_eq!(
            only(|r| r.memory = Some(common::EngineMemory::default())),
            LoadTelemetry::MEMORY
        );
        assert_eq!(
            only(|r| r.queues = Some(common::EngineQueues::default())),
            LoadTelemetry::QUEUES
        );
        assert_eq!(
            only(|r| r.disaggregation = Some(common::EngineDisaggregation::default())),
            LoadTelemetry::DISAGGREGATION
        );
        let mut full = common::EngineLoad {
            cache_hit_rate: Some(0.5),
            num_used_tokens: Some(1),
            max_total_num_tokens: Some(1),
            memory: Some(common::EngineMemory::default()),
            queues: Some(common::EngineQueues::default()),
            disaggregation: Some(common::EngineDisaggregation::default()),
            ..Default::default()
        };
        assert_eq!(full.telemetry(), LoadTelemetry::all());
        core_only(&mut full);
        assert_eq!(full.telemetry(), LoadTelemetry::empty());

        let record = common::EngineLoad {
            running_requests: 3,
            age_ms: 40,
            engine_silence_ms: Some(1_500),
            engine_reports_steps: Some(true),
            prefill_pending_tokens: Some(9),
            ..Default::default()
        };
        assert_eq!(record.age(), Duration::from_millis(40));
        assert_eq!(record.engine_silence(), Some(Duration::from_millis(1_500)));
        assert!(record.reports_every_step());
        assert_eq!(record.prefill_pending(), Some(9));
        assert_eq!(record.snapshot().num_running_reqs, 3);
        let quiet = common::EngineLoad::default();
        assert_eq!(
            (quiet.age(), quiet.engine_silence(), quiet.prefill_pending()),
            (Duration::ZERO, None, None)
        );
        assert!(!quiet.reports_every_step());
    }
}
