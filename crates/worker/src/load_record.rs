//! The engine load record (`common_proto::EngineLoad`, pushed on
//! `KvEventBatch.load`) read as the per-rank snapshot of a load report, and
//! as the record the worker's load report and its liveness read
//! ([`LoadRecord`]). `smg-grpc-client` holds the other direction, each
//! engine's `GetLoads` shape into the record, and the cases that hold the two
//! together.

use std::{sync::Arc, time::Duration};

use crate::{
    common_proto as common,
    value_bounds::{stands, MAX_VALUES},
    worker::{
        EngineMemoryMetricsSnapshot, EngineQueueMetricsSnapshot, LoadRecord, LoadTelemetry,
        NamedValue, SchedulerLoadSnapshot,
    },
};

/// The record's named values under the report's bounds: of the record's
/// first `MAX_VALUES` entries, those that stand by the test every list is
/// held to, in a list sized for them once. What is left out here is
/// counted where the record is taken: the record's list less this one
/// ([`LoadRecord::values_sent`]).
fn read_values(values: &[common::NamedValue]) -> Vec<NamedValue> {
    let mut named: Vec<NamedValue> = Vec::with_capacity(values.len().min(MAX_VALUES));
    for sample in values.iter().take(MAX_VALUES) {
        if stands(&sample.name, &sample.unit, sample.value, &named) {
            named.push(NamedValue::from(sample));
        }
    }
    named
}

/// A named value as a wire states it, field for field: the one read of an
/// entry, for a pushed record's list and for a polled answer's. A polled
/// answer's list is read whole by it, and the load monitor holds that list
/// to the report's bounds and counts what leaves it, as it does the list
/// of a poll over HTTP.
impl From<&common::NamedValue> for NamedValue {
    fn from(stated: &common::NamedValue) -> Self {
        Self {
            name: Arc::from(stated.name.as_str()),
            unit: Arc::from(stated.unit.as_str()),
            value: stated.value,
            age_ms: stated.age_ms,
        }
    }
}

/// A record's fixed fields take the shape of a poll's. The two
/// disaggregation queue depths roll up the per-stage counters, utilization
/// is KV usage, and the request total is the sum of running and waiting
/// requests. The caller sets the rank and can merge fixed telemetry the
/// record omitted with the last figures it received.
///
/// The fields alone: a part the record left out is at the type's default
/// and bears no mark, so the named values of this shape offer a count the
/// record did not state. A reader of named values takes the record's entry
/// from [`LoadRecord::snapshot`], which marks it.
impl From<&common::EngineLoad> for SchedulerLoadSnapshot {
    /// Read input capacity directly, including absence after the samples expire.
    fn from(record: &common::EngineLoad) -> Self {
        // Read first: no fixed field below is then held across this
        // read's calls, and each is written to its place once.
        let values = read_values(&record.values);
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
            values,
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
    fn fields(&self) -> SchedulerLoadSnapshot {
        SchedulerLoadSnapshot::from(self)
    }

    #[inline]
    fn values_sent(&self) -> usize {
        self.values.len()
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

#[cfg(test)]
mod tests {
    use prost::Message;

    use super::*;

    #[test]
    fn a_pushed_record_carries_each_named_value_with_its_own_age() {
        let wire = b"\xc2\x01\x1a\x0a\x09queue_age\x12\x02ms\x19\x00\x00\x00\x00\x00\x00\x1a\x40\x20\x25\
                     \xc2\x01\x1d\x0a\x09read_rate\x12\x07items/s\x19\x00\x00\x00\x00\x00\x00\xc0\x3f";
        let record = common::EngineLoad::decode(wire.as_slice()).unwrap();
        assert_eq!(record.encode_to_vec().as_slice(), wire);
        let snapshot = record.snapshot();
        assert_eq!(
            snapshot.values,
            [
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
            ]
        );
        let json = serde_json::to_value(snapshot).unwrap();
        assert_eq!(json["values"][0]["age_ms"], 37);
        assert!(json["values"][1].get("age_ms").is_none());
    }

    #[test]
    fn an_unnamed_record_keeps_the_old_snapshot_json() {
        let record = common::EngineLoad {
            running_requests: 13,
            num_used_tokens: Some(493),
            max_total_num_tokens: Some(8_317),
            ..Default::default()
        };
        let legacy = concat!(
            r#"{"dp_rank":0,"num_running_reqs":13,"num_waiting_reqs":0,"num_waiting_uncached_tokens":0,"#,
            r#""num_total_reqs":13,"num_used_tokens":493,"max_total_num_tokens":8317,"token_usage":0.0,"#,
            r#""gen_throughput":0.0,"cache_hit_rate":0.0,"utilization":0.0,"max_running_requests":0}"#,
        );
        let snapshot = record.snapshot();
        assert!(snapshot.values.is_empty());
        assert_eq!(serde_json::to_string(&snapshot).unwrap(), legacy);
        let decoded: SchedulerLoadSnapshot = serde_json::from_str(legacy).unwrap();
        assert!(decoded.values.is_empty());
    }

    /// The record's own entry bears the mark of what the record did not
    /// state: a count it left out is not offered as a zero, and neither
    /// count is where the record states no capacity.
    #[test]
    fn a_records_snapshot_is_marked_with_what_the_record_did_not_state() {
        let offered = |snapshot: &SchedulerLoadSnapshot| -> Vec<(String, f64)> {
            snapshot
                .named_values()
                .map(|value| (value.name.to_string(), value.value))
                .collect()
        };
        let record = |held: Option<i32>, capacity: Option<i32>| common::EngineLoad {
            running_requests: 2,
            num_used_tokens: held,
            max_total_num_tokens: capacity,
            ..Default::default()
        };
        let capacity = ("kv_tokens_capacity".to_owned(), 8_192.0);
        // Both stated beside a capacity: both offered, the zero held too.
        let idle = record(Some(0), Some(8_192)).snapshot();
        assert_eq!(
            offered(&idle),
            [("kv_tokens_used".to_owned(), 0.0), capacity.clone()]
        );
        // The sections the record left out are marked with the counts.
        assert!(idle
            .kept
            .contains(LoadTelemetry::MEMORY | LoadTelemetry::QUEUES));
        // The held count left out: the capacity alone.
        let core = record(None, Some(8_192)).snapshot();
        assert_eq!(offered(&core), [capacity]);
        // No capacity: neither, stated as zeros or beside a held count.
        for (held, capacity) in [
            (Some(0), Some(0)),
            (Some(211), Some(-1)),
            (Some(211), None),
            (None, None),
        ] {
            let snapshot = record(held, capacity).snapshot();
            assert!(offered(&snapshot).is_empty(), "{held:?} {capacity:?}");
        }
    }

    fn sample(name: &str, unit: &str, value: f64) -> common::NamedValue {
        common::NamedValue {
            name: name.into(),
            unit: unit.into(),
            value,
            age_ms: None,
        }
    }

    #[test]
    fn a_pushed_record_drops_and_counts_each_nonfinite_value() {
        for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let values = [
                sample("invalid", "items", nonfinite),
                sample("finite", "items", -3.25),
            ];
            let kept = read_values(&values);
            assert_eq!(kept.len(), 1);
            assert_eq!((kept[0].name.as_ref(), kept[0].value), ("finite", -3.25));
            let record = common::EngineLoad {
                values: values.into(),
                ..Default::default()
            };
            assert_eq!(record.snapshot().values, kept);
            assert_eq!(record.values_sent() - kept.len(), 1);
        }
    }

    #[test]
    fn a_pushed_record_drops_and_counts_values_past_each_bound() {
        let dropped = |values: &[common::NamedValue]| values.len() - read_values(values).len();
        let mut values: Vec<_> = (0..=64)
            .map(|index| sample(&format!("ready_{index}"), "items", f64::from(index)))
            .collect();
        let kept = read_values(&values);
        assert_eq!(kept.len(), 64);
        assert_eq!(kept.last().unwrap().value, 63.0);
        assert_eq!(dropped(&values), 1);
        // An entry dropped among the first 64 gives no later one its place.
        values[0].value = f64::NAN;
        assert_eq!(read_values(&values).len(), 63);
        assert_eq!(dropped(&values), 2);

        let name = "é".repeat(31) + "e";
        let unit = "ø".repeat(15) + "o";
        let values = [
            sample(&"é".repeat(32), "items", 19.0),
            sample("ready", &"ø".repeat(16), 23.0),
            sample(&name, &unit, 27.0),
        ];
        let kept = read_values(&values);
        assert_eq!(kept.len(), 1);
        assert_eq!(
            (kept[0].name.as_ref(), kept[0].unit.as_ref()),
            (name.as_str(), unit.as_str())
        );
        assert_eq!(dropped(&values), 2);

        let record = common::EngineLoad {
            running_requests: 17,
            values: values.to_vec(),
            ..Default::default()
        };
        let snapshot = record.snapshot();
        assert_eq!(snapshot.num_running_reqs, 17);
        assert_eq!(snapshot.values, kept);
        assert_eq!(record.values_sent() - snapshot.values.len(), 2);
    }

    #[test]
    fn a_pushed_record_drops_an_empty_name_and_a_name_it_states_again() {
        let values = [
            sample("", "items", 1.0),
            sample("depth", "items", 2.0),
            sample("depth", "items", 3.0),
            sample("rate", "items/s", 4.0),
        ];
        let kept = read_values(&values);
        let read: Vec<_> = kept
            .iter()
            .map(|value| (value.name.as_ref(), value.value))
            .collect();
        assert_eq!(read, [("depth", 2.0), ("rate", 4.0)]);
        // An entry dropped at a bound keeps no name: the next of its name stands.
        let values = [
            sample("depth", "items", f64::NAN),
            sample("depth", "items", 5.0),
        ];
        let kept = read_values(&values);
        assert_eq!((kept.len(), kept[0].value), (1, 5.0));
    }

    /// A polled answer's list and a pushed record's are held to one test:
    /// of the same entries the same ones stand, in the same order, and as
    /// many leave.
    #[test]
    fn a_polled_list_keeps_what_a_pushed_record_of_its_entries_keeps() {
        let mut values: Vec<_> = (0..70)
            .map(|index| sample(&format!("ready_{index}"), "items", f64::from(index)))
            .collect();
        values[3] = sample("", "items", 1.0);
        values[5] = sample("ready_4", "items", 2.0);
        values[7] = sample("ready_7", "items", f64::INFINITY);
        values[9] = sample(&"n".repeat(64), "items", 3.0);
        values[11] = sample("ready_11", &"u".repeat(32), 4.0);
        let pushed = read_values(&values);
        assert_eq!(pushed.len(), 59);
        let mut polled = SchedulerLoadSnapshot {
            values: values
                .iter()
                .map(|value| NamedValue {
                    name: value.name.as_str().into(),
                    unit: value.unit.as_str().into(),
                    value: value.value,
                    age_ms: value.age_ms,
                })
                .collect(),
            ..Default::default()
        };
        assert_eq!(polled.bound_values(), values.len() - pushed.len());
        assert_eq!(polled.values, pushed);
    }
}
