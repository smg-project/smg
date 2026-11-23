//! The worker's own metric writers: a worker's state and health, its requests
//! in flight, its circuit breaker, the engine load it reports and the KV
//! events of its stream, with the handles those records keep and the string
//! interner every writer of the gateway shares.
//!
//! `describe_series` holds the describing line of each series written
//! here; the gateway's list calls it. `series` is the store the series
//! labelled `worker` live in, so they leave `/metrics` with their worker.

use std::{
    fmt,
    sync::{Arc, OnceLock, Weak},
};

use arc_swap::{ArcSwap, Guard};
use dashmap::DashMap;
use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use once_cell::sync::Lazy;
use rustc_hash::FxBuildHasher;

use super::series;
use crate::{identity::basic::Worker, metrics::intern_model_label};

/// `smg_kv_event_lag_seconds`, named for the exporter's bucket list: without
/// buckets of its own the recorder renders a histogram as a summary.
pub const KV_EVENT_LAG_SECONDS: &str = "smg_kv_event_lag_seconds";
/// `smg_kv_event_apply_seconds`, named for the exporter's bucket list.
pub const KV_EVENT_APPLY_SECONDS: &str = "smg_kv_event_apply_seconds";

// Interned strings are never freed; only intern low-cardinality, server-controlled
// labels (model IDs, worker URLs, normalized paths), never user-controlled input.

/// Global string interner for metric labels.
/// Uses DashMap (sharded read-write locks) for concurrent access; the keys are the server's own
/// labels, so it hashes with [`FxBuildHasher`] as the registry's maps do.
static STRING_INTERNER: Lazy<DashMap<String, Arc<str>, FxBuildHasher>> =
    Lazy::new(|| DashMap::with_hasher(FxBuildHasher));

/// Intern a string, returning a cheaply-cloneable `Arc<str>`.
///
/// This function is designed for high-throughput scenarios where the same
/// strings (model IDs, worker URLs) appear repeatedly. The first call allocates,
/// subsequent calls just clone the Arc (very cheap - just a ref count increment).
pub fn intern_string(s: &str) -> Arc<str> {
    // Fast path: check if already interned
    if let Some(entry) = STRING_INTERNER.get(s) {
        return Arc::clone(entry.value());
    }

    // Slow path: intern the string
    // Use entry API to avoid TOCTOU race
    STRING_INTERNER
        .entry(s.to_string())
        .or_insert_with(|| Arc::from(s))
        .clone()
}

/// The ranks whose label [`rank_label`] keeps in its table.
const TABLED_RANKS: usize = 16;

/// The `dp_rank` label of a data-parallel rank, the same text for every
/// writer of a rank's series. The first ranks come from a table built
/// once: a record that builds its own keys formats nothing and interns
/// nothing for its rank. A rank past the table is formatted and interned.
fn rank_label(dp_rank: i32) -> Arc<str> {
    static FIRST: Lazy<[Arc<str>; TABLED_RANKS]> =
        Lazy::new(|| std::array::from_fn(|rank| Arc::from(rank.to_string())));
    usize::try_from(dp_rank)
        .ok()
        .and_then(|rank| FIRST.get(rank))
        .map_or_else(|| intern_string(&dp_rank.to_string()), Arc::clone)
}

#[cfg(any(test, feature = "test-util"))]
pub fn interner_size() -> usize {
    STRING_INTERNER.len()
}

/// Whether `s` is a label the interner holds. A test that must prove a
/// value never became a label asks this, which no neighbouring test's
/// interning can disturb, where the interner's size moves with every
/// label any test records.
#[cfg(any(test, feature = "test-util"))]
pub fn is_interned(s: &str) -> bool {
    STRING_INTERNER.contains_key(s)
}

/// The handles a worker's records of its requests, its circuit breaker and
/// its engine load go through, held by what writes them (the worker object,
/// its breaker, its routing-key load) where each thread once kept a map by
/// worker address: a gateway keeps one set a writer, whatever the number of
/// its threads, and a record hashes no address. A set dies with its holder.
pub struct WorkerHandles {
    /// The worker URL, the `worker` label of every series of the set.
    url: String,
    kept: ArcSwap<handles::Kept>,
}

impl WorkerHandles {
    /// An empty set for the worker at `url`: a handle is built by the first
    /// record that needs it, so the series that exist are the ones written.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            kept: ArcSwap::new(handles::Kept::none()),
        }
    }

    /// The worker URL the set's series carry.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// The handles kept under the current worker set, or `None` while
    /// handles are not kept and a record builds its key.
    fn kept(&self) -> Option<Guard<Arc<handles::Kept>>> {
        handles::keep_handles().then(|| handles::Kept::current(&self.kept))
    }
}

impl fmt::Debug for WorkerHandles {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerHandles")
            .field("url", &self.url)
            .finish_non_exhaustive()
    }
}

/// The kept handles of the worker's records: the sets the worker's parts
/// hold ([`WorkerHandles`]), and this thread's cache of the KV-event set,
/// which a batch of the event stream records. The helpers and the
/// installed flag are the worker crate's (`crate::metrics::handles`), so
/// every writer records the same way.
mod handles {
    use std::{
        cell::RefCell,
        sync::{
            atomic::{AtomicU64, Ordering},
            Arc, OnceLock,
        },
    };

    use arc_swap::{ArcSwap, Guard};
    use metrics::{Counter, Gauge, Histogram};
    use once_cell::sync::Lazy;
    use parking_lot::Mutex;
    use rustc_hash::FxHashMap;

    use super::{intern_model_label, intern_string, rank_label};
    #[cfg(test)]
    pub(super) use crate::metrics::handles::force_on_this_thread;
    pub(super) use crate::metrics::handles::{is_sentinel, keep_handles, kept, nested};

    /// The `smg_engine_*` handles of one rank of a worker under one model
    /// label, each built the first time a load has a figure for it, so the
    /// series that exist are exactly the ones recorded to: the core gauges
    /// with the rank's first load, the capacity gauge with the first that
    /// carries the capacity, both under the worker's registered role; the
    /// PD gauges under the role the engine reports with the first that
    /// carries a `disagg` section.
    #[derive(Default)]
    pub(super) struct EngineRankHandles {
        pub(super) core: Option<EngineCoreHandles>,
        /// `smg_engine_kv_capacity_tokens`, under the core gauges' role.
        pub(super) kv_capacity: Option<Gauge>,
        /// The PD gauges, under the role label the engine last reported.
        pd: Option<EnginePdHandles>,
    }

    /// The KV-event handles of one worker that a batch of its stream moves,
    /// each built the first time a batch has something for it, so the
    /// series that exist are exactly the ones recorded to.
    #[derive(Default)]
    pub(super) struct KvEventHandles {
        /// `smg_kv_event_batches_total` by disposition.
        pub(super) batches: FxHashMap<&'static str, Counter>,
        /// `smg_kv_event_blocks_total` by op.
        pub(super) blocks: FxHashMap<&'static str, Counter>,
        /// `smg_kv_index_blocks`.
        pub(super) index_blocks: Option<Gauge>,
        /// `smg_kv_event_apply_seconds`.
        pub(super) apply: Option<Histogram>,
        /// `smg_kv_event_lag_seconds`.
        pub(super) lag: Option<Histogram>,
    }

    /// The labels of one rank's core series, interned once per record that
    /// builds a handle (a rank's first sight, or every record while handles
    /// are not kept) and shared by the handles it builds; a record on the
    /// kept path interns nothing.
    pub(super) struct EngineLabels {
        worker: Arc<str>,
        model: Arc<str>,
        dp_rank: Arc<str>,
    }

    impl EngineLabels {
        fn new(worker_url: &str, model_id: &str, dp_rank: i32) -> Self {
            Self {
                worker: intern_string(worker_url),
                model: intern_model_label(model_id),
                dp_rank: rank_label(dp_rank),
            }
        }
    }

    /// The labels in `slot`, built on the first call of a record.
    pub(super) fn engine_labels<'l>(
        slot: &'l mut Option<EngineLabels>,
        worker_url: &str,
        model_id: &str,
        dp_rank: i32,
    ) -> &'l EngineLabels {
        slot.get_or_insert_with(|| EngineLabels::new(worker_url, model_id, dp_rank))
    }

    /// The gauges every load sets for its rank, under one role label.
    pub(super) struct EngineCoreHandles {
        /// The role these series carry, their worker's registered one,
        /// interned once per role.
        pub(super) role: Arc<str>,
        /// The static `role` was interned from: a record of the same writer
        /// hands in the same one, and its test is then two words.
        pub(super) registered: &'static str,
        /// `smg_engine_running_requests`.
        pub(super) running: Gauge,
        /// `smg_engine_waiting_requests`.
        pub(super) waiting: Gauge,
        /// `smg_engine_token_usage`.
        pub(super) token_usage: Gauge,
        /// `smg_engine_gen_throughput`.
        pub(super) gen_throughput: Gauge,
        /// `smg_engine_cache_hit_rate`.
        pub(super) cache_hit_rate: Gauge,
    }

    /// The gauges of a rank's `disagg` section under one role label, each
    /// built when a load first carries its figure.
    pub(super) struct EnginePdHandles {
        /// The role label the engine reported, interned once per role.
        pub(super) role: Arc<str>,
        /// `smg_engine_pd_kv_transfer_latency_ms`.
        pub(super) kv_transfer_latency: Option<Gauge>,
        /// `smg_engine_pd_kv_transfer_speed_gb_s`.
        pub(super) kv_transfer_speed: Option<Gauge>,
        /// `smg_engine_pd_prefill_queue_reqs`.
        pub(super) prefill_queue: Option<Gauge>,
        /// `smg_engine_pd_decode_queue_reqs`.
        pub(super) decode_queue: Option<Gauge>,
    }

    impl EngineRankHandles {
        /// The PD handles under `role`: the kept set while the engine keeps
        /// reporting that role, a fresh set when it reports another (the
        /// old role's series stay as they were, as they always have).
        pub(super) fn pd(&mut self, role: &str) -> &mut EnginePdHandles {
            let kept = self
                .pd
                .get_or_insert_with(|| EnginePdHandles::new(intern_string(role)));
            if kept.role.as_ref() != role {
                *kept = EnginePdHandles::new(intern_string(role));
            }
            kept
        }
    }

    impl EngineCoreHandles {
        /// The five core series of the rank `labels` name, under `role`.
        pub(super) fn new(labels: &EngineLabels, registered: &'static str) -> Self {
            let role = intern_string(registered);
            Self {
                running: engine_gauge("smg_engine_running_requests", labels, &role),
                waiting: engine_gauge("smg_engine_waiting_requests", labels, &role),
                token_usage: engine_gauge("smg_engine_token_usage", labels, &role),
                gen_throughput: engine_gauge("smg_engine_gen_throughput", labels, &role),
                cache_hit_rate: engine_gauge("smg_engine_cache_hit_rate", labels, &role),
                role,
                registered,
            }
        }
    }

    impl EnginePdHandles {
        fn new(role: Arc<str>) -> Self {
            Self {
                role,
                kv_transfer_latency: None,
                kv_transfer_speed: None,
                prefill_queue: None,
                decode_queue: None,
            }
        }
    }

    /// The series `name` of the rank `labels` name, by the key the
    /// engine-load record has always built: the series is the same one
    /// whether it is reached through a kept handle or a key built per call,
    /// and the scrape reads the same.
    pub(super) fn engine_gauge(
        name: &'static str,
        labels: &EngineLabels,
        role: &Arc<str>,
    ) -> Gauge {
        metrics::gauge!(
            name,
            "worker" => Arc::clone(&labels.worker),
            "model" => Arc::clone(&labels.model),
            "dp_rank" => Arc::clone(&labels.dp_rank),
            "role" => Arc::clone(role),
        )
    }

    /// A PD series: labelled by the role the engine reported, not the model.
    pub(super) fn engine_pd_gauge(
        name: &'static str,
        labels: &EngineLabels,
        role: &Arc<str>,
    ) -> Gauge {
        metrics::gauge!(
            name,
            "worker" => Arc::clone(&labels.worker),
            "role" => Arc::clone(role),
            "dp_rank" => Arc::clone(&labels.dp_rank),
        )
    }

    /// The kept handles of rank `dp_rank` under `model_id`'s label for one
    /// worker, created on first sight: the model map's key is the interner's
    /// own allocation, as the request counters' is, and a label the interner
    /// refused (its sentinel) keeps nothing (`None`).
    pub(super) fn engine_rank<'m>(
        by_model: &'m mut FxHashMap<Arc<str>, FxHashMap<i32, EngineRankHandles>>,
        model_id: &str,
        dp_rank: i32,
    ) -> Option<&'m mut EngineRankHandles> {
        if !by_model.contains_key(model_id) {
            let label = intern_model_label(model_id);
            if !kept(by_model.len(), &label) {
                return None;
            }
            by_model.insert(label, FxHashMap::default());
        }
        let ranks = by_model.get_mut(model_id)?;
        Some(ranks.entry(dp_rank).or_default())
    }

    /// Every removal of a worker moves this on, and so does a registration
    /// when a kept handle may be a no-op one (its address was retired, or the
    /// upkeep forgot a retired address since the last registration), and a
    /// holder whose kept per-worker handles are from an earlier state (a
    /// [`super::WorkerHandles`] set, a thread's KV-event cache) drops them
    /// at its next record: a removal drops the address's series from the
    /// per-worker store, so a handle kept from before feeds nothing, and a
    /// handle built while the address was retired is a no-op one, which a
    /// re-registration would otherwise leave in place.
    static WORKER_SET_GENERATION: AtomicU64 = AtomicU64::new(0);

    /// A worker was removed, or registered while a kept handle may be a
    /// no-op one: the kept per-worker handles are stale.
    pub(super) fn worker_set_changed() {
        WORKER_SET_GENERATION.fetch_add(1, Ordering::AcqRel);
    }

    /// Held by a registration from the store's answer to the move it asks
    /// for: the store answers once, so no other registration may end, and
    /// its worker be written for through a kept no-op handle, in between.
    pub(super) static REGISTRATION: Mutex<()> = Mutex::new(());

    /// What one [`super::WorkerHandles`] keeps, with the worker-set
    /// generation it was kept under: a set whose handles are from an
    /// earlier state drops them at its next record. A handle of a request
    /// is a write-once cell, so a record that finds it built is one load
    /// and the update of the value; the engine-load handles sit behind a
    /// lock that the load record alone takes.
    #[derive(Default)]
    pub(super) struct Kept {
        generation: u64,
        /// `smg_worker_requests_total`, under the worker's model label.
        pub(super) requests: OnceLock<Counter>,
        /// `smg_worker_requests_active`.
        pub(super) requests_active: OnceLock<Gauge>,
        /// `smg_worker_routing_keys_active`.
        pub(super) routing_keys_active: OnceLock<Gauge>,
        /// `smg_worker_cb_consecutive_failures`.
        pub(super) cb_failures: OnceLock<Gauge>,
        /// `smg_worker_cb_consecutive_successes`.
        pub(super) cb_successes: OnceLock<Gauge>,
        /// `smg_worker_cb_outcomes_total` of a success.
        pub(super) cb_outcome_success: OnceLock<Counter>,
        /// `smg_worker_cb_outcomes_total` of a failure.
        pub(super) cb_outcome_failure: OnceLock<Counter>,
        /// The `smg_engine_*` set: by the model label as the interner gave
        /// it, then the DP rank. The model is the one a worker registered
        /// with, the gateway's own string, so its level hashes with FxHash.
        pub(super) engine_loads: Mutex<FxHashMap<Arc<str>, FxHashMap<i32, EngineRankHandles>>>,
    }

    impl Kept {
        /// What a set holds before its first record: one shared value of a
        /// generation no worker set has, so a set costs no allocation
        /// until it keeps a handle.
        pub(super) fn none() -> Arc<Self> {
            static NONE: Lazy<Arc<Kept>> = Lazy::new(|| {
                Arc::new(Kept {
                    generation: u64::MAX,
                    ..Kept::default()
                })
            });
            Arc::clone(&NONE)
        }

        /// The handles `slot` keeps while they are of the current worker
        /// set, an empty set in their place when they are not.
        pub(super) fn current(slot: &ArcSwap<Self>) -> Guard<Arc<Self>> {
            let generation = WORKER_SET_GENERATION.load(Ordering::Acquire);
            let kept = slot.load();
            // A set kept under a later generation than this read is newer,
            // not stale (`none()`'s `u64::MAX` is neither).
            if kept.generation != u64::MAX && kept.generation >= generation {
                return kept;
            }
            // The first record of the new generation installs its set; a
            // record that raced it uses the winner's, so no handle it built
            // is thrown away.
            let _ = slot.compare_and_swap(
                &kept,
                Arc::new(Self {
                    generation,
                    ..Self::default()
                }),
            );
            slot.load()
        }
    }

    /// This thread's kept handles of one per-worker family, by worker URL,
    /// with the worker-set generation they were kept under. The rule for
    /// every family is here: the map is cleared when the generation moves.
    pub(super) struct ByWorker<V> {
        generation: u64,
        by_worker: FxHashMap<String, V>,
    }

    impl<V> Default for ByWorker<V> {
        fn default() -> Self {
            Self {
                generation: 0,
                by_worker: FxHashMap::default(),
            }
        }
    }

    impl<V> ByWorker<V> {
        /// The handles by worker, after dropping every kept handle if the
        /// worker set changed since they were kept.
        pub(super) fn current(&mut self) -> &mut FxHashMap<String, V> {
            let generation = WORKER_SET_GENERATION.load(Ordering::Acquire);
            if self.generation != generation {
                self.by_worker.clear();
                self.generation = generation;
            }
            &mut self.by_worker
        }
    }

    impl<V: Default> ByWorker<V> {
        /// The kept handles under `worker_url`, created on first sight.
        pub(super) fn for_worker(&mut self, worker_url: &str) -> &mut V {
            nested(self.current(), worker_url)
        }
    }

    thread_local! {
        /// The KV-event set a batch moves.
        pub(super) static KV_EVENTS: RefCell<ByWorker<KvEventHandles>> = RefCell::new(ByWorker::default());
    }

    /// For tests: this thread's kept handles of the KV events, dropped, and
    /// the installed flag given its say back.
    #[cfg(test)]
    pub(super) fn reset() {
        force_on_this_thread(None);
        KV_EVENTS.with_borrow_mut(|kept| *kept = ByWorker::default());
    }

    #[cfg(test)]
    mod tests {
        use std::collections::HashMap;

        use super::*;
        use crate::metrics::handles::MAX_KEPT_PER_MAP;

        #[test]
        fn kept_handles_go_when_the_worker_set_changes() {
            // The generation is process-wide and other tests register and
            // remove workers: the steady state is read off the set one
            // `current` call handed out, with no second call in between
            // that a bump elsewhere could clear.
            let slot = ArcSwap::new(Kept::none());
            let kept = Kept::current(&slot);
            assert!(kept.requests.set(Counter::noop()).is_ok());
            worker_set_changed();
            assert!(Kept::current(&slot).requests.get().is_none());
        }

        #[test]
        fn engine_rank_handles_are_made_once_per_model_and_rank() {
            let mut by_model = FxHashMap::default();
            for (model, rank) in [("m", 0), ("m", 1), ("m", 0), ("m2", 0), ("m", 1)] {
                assert!(engine_rank(&mut by_model, model, rank).is_some());
            }
            assert_eq!(by_model.len(), 2, "one map per model label");
            assert_eq!(by_model.get("m").map(HashMap::len), Some(2));
            assert_eq!(by_model.get("m2").map(HashMap::len), Some(1));
        }

        /// A model map at its cap keeps no further label: the rank's handles
        /// are then built and used by the caller without a place here.
        #[test]
        fn engine_rank_keeps_nothing_past_the_cap() {
            let mut by_model = FxHashMap::default();
            for held in 0..MAX_KEPT_PER_MAP {
                by_model.insert(Arc::from(format!("held-{held}")), FxHashMap::default());
            }
            assert!(engine_rank(&mut by_model, "one-more-model", 0).is_none());
            assert_eq!(by_model.len(), MAX_KEPT_PER_MAP);
        }
    }
}

/// Describes the series this module writes. The gateway's list calls it.
pub fn describe_series() {
    describe_gauge!(
        "smg_worker_pool_size",
        "Current worker pool size by worker_type, connection_mode and the worker's primary model"
    );
    describe_counter!(
        "smg_pd_rendezvous_failures_total",
        "PD rendezvous failures by side (the worker type of the side whose rendezvous failed) and phase: a bootstrap port that refused the registration probe (probe) or the same probe run when a decode side's engine failed a request without a reason (request, on the prefill side whose port it is), and the failures the engines report on the gRPC path when the sides never met (bootstrap) or the KV handoff after they met failed (transfer)"
    );
    describe_gauge!(
        "smg_worker_requests_active",
        "Currently running requests per worker"
    );
    describe_counter!(
        "smg_worker_requests_total",
        "Requests dispatched to a worker by worker (URL) and model, counted at each send"
    );
    describe_gauge!(
        "smg_worker_health",
        "Worker health status (1=healthy, 0=unhealthy)"
    );
    describe_gauge!(
        "smg_worker_http2",
        "Whether the router speaks HTTP/2 to the worker (1=HTTP/2, 0=HTTP/1.1)"
    );
    describe_counter!(
        "smg_worker_health_checks_total",
        "Health check results by worker_type and result"
    );
    describe_counter!(
        "smg_kv_event_subscription_failures_total",
        "KV event subscription task failures by worker and reason \
         (panic, join_error, intern_failed)"
    );
    describe_counter!(
        "smg_kv_event_subscriptions_total",
        "KV event streams connected, by worker; a reconnect counts again"
    );
    describe_counter!(
        "smg_kv_event_stream_errors_total",
        "KV event streams that failed after connecting, by worker and error (the gRPC \
         status code: out_of_range, unavailable, data_loss, ...)"
    );
    describe_gauge!(
        "smg_kv_events_unavailable",
        "1 while a worker's backend serves no KV events (SubscribeKvEvents answered \
         UNIMPLEMENTED: the engine's publisher is off), 0 once the worker leaves, by \
         worker. Cache-aware routing is blind to such a worker's cache"
    );
    describe_counter!(
        "smg_kv_event_batches_total",
        "KV event batches by worker and disposition (applied, stale, snapshot, load_only, \
         other_rank, not_indexed: a batch of events on a stream opened for load records alone, \
         or one whose rank fits no rank)"
    );
    describe_counter!(
        "smg_engine_load_polls_total",
        "Load-monitor tick decisions by worker and mode: poll (no pushed load record on file), \
         fallback (pushed records older than the tick interval), skipped_fresh_push (the \
         worker's KV-event stream pushed its load within the interval; no GetLoads RPC)"
    );
    describe_counter!(
        "smg_engine_load_values_dropped_total",
        "Named values of pushed load records and polled load answers dropped at the report's \
         bounds, by worker: an entry past the first 64 of a list, an empty name, a name over \
         63 bytes, a unit over 31 bytes, a value that is not finite, a name the list states \
         again, an entry of an answer that is no named value"
    );
    describe_counter!(
        "smg_kv_event_gaps_total",
        "KV event sequence gaps by worker and outcome (replay_requested, \
         unrecovered_kept, unrecovered_cleared)"
    );
    describe_counter!(
        "smg_kv_event_missed_batches_total",
        "KV event batches the publisher skipped and could not replay, by worker"
    );
    describe_counter!(
        "smg_kv_event_resyncs_total",
        "KV event rank resyncs by worker and reason (out_of_range, data_loss, \
         publisher_restart, gap_cleared, snapshot)"
    );
    describe_histogram!(
        KV_EVENT_LAG_SECONDS,
        "Age of a KV event batch when applied: now minus the publisher timestamp, by worker"
    );
    describe_counter!(
        "smg_kv_event_parentless_stores_total",
        "Stores whose parent block the index did not hold for the worker, placed as a \
         new chain from the root instead (a parent evicted, dropped, or cleared), by worker"
    );
    describe_counter!(
        "smg_kv_event_parentless_blocks_total",
        "Blocks of the parent-less stores, by worker"
    );
    describe_counter!(
        "smg_kv_event_sub_block_stores_total",
        "Stores of blocks narrower than the worker's block of record (a hybrid engine's \
         prefix entries at its hash granularity), left out of the index, by worker"
    );
    describe_counter!(
        "smg_kv_event_sub_block_blocks_total",
        "Blocks of the sub-block stores, by worker"
    );
    describe_counter!(
        "smg_kv_event_cap_refused_stores_total",
        "Stores the KV index declined because a run's child table or hash array would outgrow \
         an arena chunk (the fan-out and run-length caps), by worker"
    );
    describe_counter!(
        "smg_kv_event_cap_refused_blocks_total",
        "Blocks the declined stores left out of the index, by worker"
    );
    describe_counter!(
        "smg_kv_event_blocks_total",
        "Blocks named by applied KV events, by worker and op (stored, removed; snapshot: \
         the live set a relay state snapshot listed)"
    );
    describe_histogram!(
        KV_EVENT_APPLY_SECONDS,
        "Time to apply one KV event batch to the index, by worker"
    );
    describe_gauge!(
        "smg_kv_event_degraded_ranks",
        "KV event ranks whose index may be stale after an unreplayed gap, by worker"
    );
    describe_gauge!(
        "smg_kv_index_memberships",
        "Blocks the KV index holds across a model's workers (a block two workers hold \
         counts twice), by model; published every 30 s"
    );
    describe_gauge!(
        "smg_kv_index_entries",
        "Distinct entries in the KV index, by model: distinct blocks on a chain"
    );
    describe_gauge!(
        "smg_kv_index_runs_live",
        "Runs linked in the chain index, by model (blocks_live over runs_live is the \
         mean run length; a falling ratio under steady traffic is fragmentation)"
    );
    describe_gauge!(
        "smg_kv_index_blocks_live",
        "Content hashes held by the chain index's live runs, by model"
    );
    describe_gauge!(
        "smg_kv_index_arena_bytes",
        "Bytes the chain index's word arena has handed out (hash arrays, child tables, \
         free lists included), by model"
    );
    describe_gauge!(
        "smg_kv_index_arena_free_bytes",
        "Bytes of the chain index's word arena sitting in free lists, by model"
    );
    describe_gauge!(
        "smg_kv_index_arena_chunk_bytes",
        "Bytes the chain index's word arena holds from the allocator: whole 8 MiB chunks, \
         each kept once taken, by model"
    );
    describe_gauge!(
        "smg_kv_index_slab_bytes",
        "Bytes the chain index's run slab holds from the allocator, by model"
    );
    describe_gauge!(
        "smg_kv_index_moved_hashes",
        "Stores that moved a held engine hash to another place in the chain index \
         (cumulative), by model"
    );
    describe_gauge!(
        "smg_kv_index_engine_conflicts",
        "Stored blocks whose engine hash differed from the one the chain index holds for \
         the block, a fleet whose engines do not name content alike (cumulative), by model"
    );
    describe_gauge!(
        "smg_kv_index_cap_refusals",
        "Stores the chain index declined because a run's child table or hash array would \
         outgrow an arena chunk (the fan-out and run-length caps; cumulative), by model"
    );
    describe_gauge!(
        "smg_kv_index_blocks",
        "Blocks the index holds for a worker, as the index counts them; \
         set when a KV event batch is applied, when the worker's state is reset \
         and when the worker is removed"
    );
    describe_gauge!(
        "smg_workers_overloaded",
        "Workers currently flagged overloaded and excluded from routing, by model"
    );
    describe_gauge!(
        "smg_worker_stalled",
        "Liveness veto on a worker: 1 while routing skips it, by worker and reason"
    );
    describe_gauge!(
        "smg_worker_state",
        "Routing state of a worker: 1 on the series of its current state and reason, 0 on a series it left; state is available or excluded, reason the first fact of the routing decision that excludes the worker (pending, not_ready, failed, draining: its status; overloaded: the load monitor's veto; unreachable, wedged: the liveness veto; circuit_open: its breaker) or none while it is available"
    );
    describe_counter!(
        "smg_worker_stall_transitions_total",
        "Times the liveness veto was set on a worker, by worker and reason"
    );
    describe_gauge!(
        "smg_worker_routing_keys_active",
        "Sticky routing keys currently held by a worker"
    );
    describe_gauge!(
        "smg_worker_cb_state",
        "Circuit breaker state per worker (0=closed, 1=open, 2=half_open)"
    );
    describe_counter!(
        "smg_worker_cb_transitions_total",
        "Circuit breaker state transitions by worker, from, to"
    );
    describe_counter!(
        "smg_worker_cb_outcomes_total",
        "Circuit breaker outcomes by worker and outcome (success/failure)"
    );
    describe_gauge!(
        "smg_worker_cb_consecutive_failures",
        "Current consecutive failure count per worker"
    );
    describe_gauge!(
        "smg_worker_cb_consecutive_successes",
        "Current consecutive success count per worker"
    );
    describe_counter!(
        "smg_worker_cb_health_readmissions_total",
        "Times a worker's open circuit breaker went half-open because the worker's health \
         returned (its promotion to Ready) rather than at the breaker's timeout, by worker"
    );
    describe_gauge!(
        "smg_engine_running_requests",
        "Engine-reported running requests by worker, model, dp_rank, role"
    );
    describe_gauge!(
        "smg_engine_waiting_requests",
        "Engine-reported waiting requests by worker, model, dp_rank, role"
    );
    describe_gauge!(
        "smg_engine_token_usage",
        "Engine-reported KV token usage ratio (0.0-1.0) by worker, model, dp_rank, role"
    );
    describe_gauge!(
        "smg_engine_gen_throughput",
        "Engine-reported generation throughput (tokens/s) by worker, model, dp_rank, role"
    );
    describe_gauge!(
        "smg_engine_cache_hit_rate",
        "Engine-reported prefix cache hit rate (0.0-1.0) by worker, model, dp_rank, role"
    );
    describe_gauge!(
        "smg_engine_kv_capacity_tokens",
        "Engine-reported KV cache capacity in tokens by worker, model, dp_rank, role: the denominator of smg_engine_token_usage; absent while the engine has not reported it (a record that carries only the core figures keeps the last reported value); divided by smg_kv_block_size_tokens it is the engine's total KV blocks"
    );
    describe_gauge!(
        "smg_kv_block_size_tokens",
        "Tokens per KV block of a model as the gateway knows it: seeded from a worker's registration, replaced by the size of the first block its KV event stream stores, 0 once the model's last subscription ended and no size is known"
    );
    describe_gauge!(
        "smg_engine_pd_kv_transfer_latency_ms",
        "Engine-reported PD KV transfer latency (ms) by worker, role, dp_rank"
    );
    describe_gauge!(
        "smg_engine_pd_kv_transfer_speed_gb_s",
        "Engine-reported PD KV transfer speed (GB/s) by worker, role, dp_rank"
    );
    describe_gauge!(
        "smg_engine_pd_prefill_queue_reqs",
        "Engine-reported PD prefill queue depth by worker, role, dp_rank"
    );
    describe_gauge!(
        "smg_engine_pd_decode_queue_reqs",
        "Engine-reported PD decode queue depth by worker, role, dp_rank"
    );
}

/// The chain index's shape and memory as
/// [`Metrics::set_kv_index_chain_stats`] publishes them. The index's holder
/// copies them from the index's own counters, so the writers name no index
/// type.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvIndexChainStats {
    /// Runs linked in the tree.
    pub runs_live: usize,
    /// Content hashes held by live runs.
    pub blocks_live: usize,
    /// Bytes of the word arena handed out so far.
    pub arena_bytes: usize,
    /// Bytes of the word arena sitting in free lists.
    pub arena_free_bytes: usize,
    /// Bytes the word arena holds from the process allocator.
    pub arena_chunk_bytes: usize,
    /// Bytes the run slab holds from the process allocator.
    pub slab_bytes: usize,
    /// Blocks a store moved to another place for their worker.
    pub moved_hashes: usize,
    /// Stored blocks whose engine hash differed from the index's own.
    pub engine_conflicts: usize,
    /// Stores declined at the arena's caps.
    pub cap_refusals: usize,
}

/// The worker's metric writers. A dynamic label is interned once per value,
/// and the records a request or a stream repeats go through kept handles.
pub struct Metrics;

impl Metrics {
    /// Record a PD rendezvous failure by `side` (the worker type of the side
    /// whose rendezvous failed) and `phase` (`probe` at registration and
    /// `request` on the request path, the prefill side's port probed by the
    /// gateway; `bootstrap` or `transfer` as the engine reported it). Its
    /// own family: the injection failures keep their unlabelled series,
    /// `smg_pd_bootstrap_failures_total`, so a bare selector on it still
    /// reads them alone.
    pub fn record_pd_rendezvous_failure(side: &'static str, phase: &'static str) {
        counter!(
            "smg_pd_rendezvous_failures_total",
            "leg" => side,
            "phase" => phase
        )
        .increment(1);
    }

    /// Set worker pool size
    pub fn set_worker_pool_size(
        worker_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        size: usize,
    ) {
        let model = intern_model_label(model_id);
        gauge!(
            "smg_worker_pool_size",
            "worker_type" => worker_type,
            "connection_mode" => connection_mode,
            "model" => model
        )
        .set(size as f64);
    }

    /// Record health check result
    pub fn record_worker_health_check(worker_type: &'static str, result: &'static str) {
        counter!(
            "smg_worker_health_checks_total",
            "worker_type" => worker_type,
            "result" => result
        )
        .increment(1);
    }

    /// Set the count of workers a model currently has vetoed by the absolute
    /// overload guard. Written only when a worker's flag transitions.
    pub fn set_workers_overloaded(model_id: &str, count: usize) {
        let model = intern_model_label(model_id);
        gauge!("smg_workers_overloaded", "model" => model).set(count as f64);
    }

    /// A request dispatched to a worker (each attempt, each PD side), counted
    /// at the send. The per-worker gauges miss short requests and cannot be
    /// summed over a window, so this is how the share of traffic per worker
    /// is read from the scrape.
    pub fn record_worker_request(worker: &dyn Worker) {
        let (worker_url, model_id) = (worker.url(), worker.model_id());
        let Some(kept) = worker.metric_handles().and_then(WorkerHandles::kept) else {
            Self::worker_requests(worker_url, intern_model_label(model_id)).increment(1);
            return;
        };
        if let Some(counter) = kept.requests.get() {
            counter.increment(1);
            return;
        }
        // One lookup gives the label the counter is built with and the
        // label that is judged: a label the interner refused is used and
        // not kept, so its sentinel's series is not pinned to this worker
        // for its life.
        let model = intern_model_label(model_id);
        let refused = handles::is_sentinel(&model);
        let counter = Self::worker_requests(worker_url, model);
        counter.increment(1);
        if !refused {
            let _ = kept.requests.set(counter);
        }
    }

    /// A worker registered: its request counter exists from zero, so a
    /// worker that gets no traffic shows as such instead of being absent.
    pub fn init_worker_requests(worker_url: &str, model_id: &str) {
        Self::worker_requests(worker_url, intern_model_label(model_id)).increment(0);
    }

    fn worker_requests(worker_url: &str, model: Arc<str>) -> metrics::Counter {
        counter!(
            "smg_worker_requests_total",
            "worker" => intern_string(worker_url),
            "model" => model
        )
    }

    pub fn set_worker_requests_active(worker: &WorkerHandles, count: usize) {
        Self::set_worker_gauge(
            worker,
            |kept| &kept.requests_active,
            count as f64,
            || {
                gauge!(
                    "smg_worker_requests_active",
                    "worker" => intern_string(&worker.url)
                )
            },
        );
    }

    /// Sets a per-worker gauge that moves on every request: through the
    /// handle `worker` keeps in `slot` (built by `build` on first use and
    /// dropped when the worker set changes), or through a key built per call
    /// when handles are not kept.
    fn set_worker_gauge(
        worker: &WorkerHandles,
        slot: impl FnOnce(&handles::Kept) -> &OnceLock<metrics::Gauge>,
        value: f64,
        build: impl FnOnce() -> metrics::Gauge,
    ) {
        match worker.kept() {
            Some(kept) => slot(&kept).get_or_init(build).set(value),
            None => build().set(value),
        }
    }

    /// Set active routing keys per worker
    pub fn set_worker_routing_keys_active(worker: &WorkerHandles, count: usize) {
        Self::set_worker_gauge(
            worker,
            |kept| &kept.routing_keys_active,
            count as f64,
            || {
                gauge!(
                    "smg_worker_routing_keys_active",
                    "worker" => intern_string(&worker.url)
                )
            },
        );
    }

    /// Flip the liveness veto gauge for a worker; count the transition when
    /// the veto is set.
    pub fn set_worker_stalled(worker_url: &str, reason: &'static str, stalled: bool) {
        let worker = intern_string(worker_url);
        gauge!(
            "smg_worker_stalled",
            "worker" => Arc::clone(&worker),
            "reason" => reason
        )
        .set(if stalled { 1.0 } else { 0.0 });
        if stalled {
            counter!(
                "smg_worker_stall_transitions_total",
                "worker" => worker,
                "reason" => reason
            )
            .increment(1);
        }
    }

    /// The tokens per KV block of `model_id` as the gateway knows it now
    /// (`smg_kv_block_size_tokens`): seeded at a worker's registration,
    /// replaced by the first block its event stream stores, 0 when the
    /// model's last subscription ended and no size is known.
    pub fn set_kv_block_size(model_id: &str, tokens: usize) {
        gauge!(
            "smg_kv_block_size_tokens",
            "model" => intern_model_label(model_id)
        )
        .set(tokens as f64);
    }

    /// Set one `smg_worker_state` series of a worker: 1 on the state and
    /// reason it is on, 0 on the pair it left.
    pub fn set_worker_state(
        worker_url: &str,
        state: &'static str,
        reason: &'static str,
        current: bool,
    ) {
        gauge!(
            "smg_worker_state",
            "worker" => intern_string(worker_url),
            "state" => state,
            "reason" => reason
        )
        .set(if current { 1.0 } else { 0.0 });
    }

    /// Set worker health status
    pub fn set_worker_health(worker_url: &str, healthy: bool) {
        let worker_interned = intern_string(worker_url);
        gauge!(
            "smg_worker_health",
            "worker" => worker_interned
        )
        .set(if healthy { 1.0 } else { 0.0 });
    }

    pub fn set_worker_http2(worker_url: &str, http2: bool) {
        let worker_interned = intern_string(worker_url);
        gauge!(
            "smg_worker_http2",
            "worker" => worker_interned
        )
        .set(if http2 { 1.0 } else { 0.0 });
    }

    /// A worker's KV event subscription starts: its integrity families
    /// exist from zero, so a scrape tells "no gap, no resync yet" from "no
    /// series". Every label value the recording paths emit is enumerated
    /// here (the gap outcomes and the failure reasons are the call sites'
    /// literals, the resync reasons `ResyncReason::as_str`). An
    /// `increment(0)` creates a series and leaves an existing one alone,
    /// so a worker added again keeps its counts.
    pub fn init_kv_event_series(worker_url: &str) {
        let worker = intern_string(worker_url);
        for outcome in [
            "replay_requested",
            "unrecovered_kept",
            "unrecovered_cleared",
        ] {
            counter!(
                "smg_kv_event_gaps_total",
                "worker" => Arc::clone(&worker),
                "outcome" => outcome
            )
            .increment(0);
        }
        counter!("smg_kv_event_missed_batches_total", "worker" => Arc::clone(&worker)).increment(0);
        for reason in [
            "out_of_range",
            "data_loss",
            "publisher_restart",
            "gap_cleared",
            "snapshot",
        ] {
            counter!(
                "smg_kv_event_resyncs_total",
                "worker" => Arc::clone(&worker),
                "reason" => reason
            )
            .increment(0);
        }
        for reason in ["panic", "join_error", "intern_failed"] {
            counter!(
                "smg_kv_event_subscription_failures_total",
                "worker" => Arc::clone(&worker),
                "reason" => reason
            )
            .increment(0);
        }
        counter!("smg_kv_event_parentless_stores_total", "worker" => Arc::clone(&worker))
            .increment(0);
        counter!("smg_kv_event_parentless_blocks_total", "worker" => Arc::clone(&worker))
            .increment(0);
        counter!("smg_kv_event_sub_block_stores_total", "worker" => Arc::clone(&worker))
            .increment(0);
        counter!("smg_kv_event_sub_block_blocks_total", "worker" => Arc::clone(&worker))
            .increment(0);
        counter!("smg_kv_event_cap_refused_stores_total", "worker" => Arc::clone(&worker))
            .increment(0);
        counter!("smg_kv_event_cap_refused_blocks_total", "worker" => Arc::clone(&worker))
            .increment(0);
        gauge!("smg_kv_event_degraded_ranks", "worker" => worker).set(0.0);
    }

    /// Record a KV event subscription task failure (panic, join error, or
    /// worker-id intern failure)
    pub fn record_kv_event_subscription_failure(worker_url: &str, reason: &'static str) {
        let worker_interned = intern_string(worker_url);
        counter!(
            "smg_kv_event_subscription_failures_total",
            "worker" => worker_interned,
            "reason" => reason
        )
        .increment(1);
    }

    /// Count a KV event stream connected: the first time and every reconnect,
    /// so a drill can time a resubscription without reading the log.
    pub fn record_kv_event_subscription(worker_url: &str) {
        counter!(
            "smg_kv_event_subscriptions_total",
            "worker" => intern_string(worker_url)
        )
        .increment(1);
    }

    /// Count a KV event stream that failed after it connected, by the
    /// error's gRPC status code: a stream that dies on its first message at
    /// every reconnect shows here, where the connect counter alone reads as
    /// a healthy subscription.
    pub fn record_kv_event_stream_error(worker_url: &str, error: &'static str) {
        counter!(
            "smg_kv_event_stream_errors_total",
            "worker" => intern_string(worker_url),
            "error" => error
        )
        .increment(1);
    }

    /// Whether a worker's backend serves no KV events: its subscription ended
    /// on `UNIMPLEMENTED`, and until the worker leaves cache-aware routing is
    /// blind to its cache. A gauge per worker, so a dashboard or an alert sees
    /// a blind cache-aware router where the log had one WARN line.
    pub fn set_kv_events_unavailable(worker_url: &str, unavailable: bool) {
        gauge!("smg_kv_events_unavailable", "worker" => intern_string(worker_url))
            .set(if unavailable { 1.0 } else { 0.0 });
    }

    /// Named values a pushed load record or a polled answer stated and the
    /// report dropped at a bound.
    pub fn record_engine_load_values_dropped(worker_url: &str, dropped: usize) {
        counter!(
            "smg_engine_load_values_dropped_total",
            "worker" => intern_string(worker_url)
        )
        .increment(dropped as u64);
    }

    /// One load-monitor tick decision for a worker (see `PollMode`).
    pub fn record_engine_load_poll(worker_url: &str, mode: &'static str) {
        counter!(
            "smg_engine_load_polls_total",
            "worker" => intern_string(worker_url),
            "mode" => mode
        )
        .increment(1);
    }

    /// The KV-event records of a batch go through this thread's kept
    /// handles for the worker (dropped when the worker set changes, as every
    /// kept per-worker handle is), or through handles built for the one
    /// call when handles are not kept.
    fn with_kv_event_handles(worker_url: &str, record: impl FnOnce(&mut handles::KvEventHandles)) {
        if handles::keep_handles() {
            handles::KV_EVENTS.with_borrow_mut(|kept| record(kept.for_worker(worker_url)));
        } else {
            record(&mut handles::KvEventHandles::default());
        }
    }

    /// Count a KV event batch by what the subscriber did with it.
    pub fn record_kv_event_batch(worker_url: &str, disposition: &'static str) {
        Self::with_kv_event_handles(worker_url, |kept| {
            kept.batches
                .entry(disposition)
                .or_insert_with(|| {
                    counter!(
                        "smg_kv_event_batches_total",
                        "worker" => intern_string(worker_url),
                        "disposition" => disposition
                    )
                })
                .increment(1);
        });
    }

    /// Count a sequence gap and the batches it skipped.
    pub fn record_kv_event_gap(worker_url: &str, outcome: &'static str, missed: u64) {
        let worker_interned = intern_string(worker_url);
        counter!(
            "smg_kv_event_gaps_total",
            "worker" => Arc::clone(&worker_interned),
            "outcome" => outcome
        )
        .increment(1);
        if missed > 0 {
            counter!("smg_kv_event_missed_batches_total", "worker" => worker_interned)
                .increment(missed);
        }
    }

    /// Count the batches of a gap no replay will bring: the pending replay
    /// was dropped, because the resume the subscriber asked for was refused
    /// (the server's history is gone), because the publisher started over
    /// before it answered, or because another rank's unrecoverable gap
    /// cleared the worker and reset this rank's cursor.
    pub fn record_kv_event_missed_batches(worker_url: &str, missed: u64) {
        if missed > 0 {
            counter!("smg_kv_event_missed_batches_total", "worker" => intern_string(worker_url))
                .increment(missed);
        }
    }

    /// Count a rank resync (its index state dropped) by reason.
    pub fn record_kv_event_resync(worker_url: &str, reason: &'static str) {
        counter!(
            "smg_kv_event_resyncs_total",
            "worker" => intern_string(worker_url),
            "reason" => reason
        )
        .increment(1);
    }

    /// Observe how old a batch was when it was applied.
    pub fn record_kv_event_lag(worker_url: &str, seconds: f64) {
        Self::with_kv_event_handles(worker_url, |kept| {
            kept.lag
                .get_or_insert_with(
                    || histogram!(KV_EVENT_LAG_SECONDS, "worker" => intern_string(worker_url)),
                )
                .record(seconds);
        });
    }

    /// Count the blocks an applied batch's events named (`stored` or
    /// `removed`), before the monitor's tier and group filters: the rate the
    /// index is offered.
    pub fn record_kv_event_blocks(worker_url: &str, op: &'static str, blocks: usize) {
        Self::with_kv_event_handles(worker_url, |kept| {
            kept.blocks
                .entry(op)
                .or_insert_with(|| {
                    counter!(
                        "smg_kv_event_blocks_total",
                        "worker" => intern_string(worker_url),
                        "op" => op
                    )
                })
                .increment(blocks as u64);
        });
    }

    /// Count the stores a batch placed without their parent (as new chains
    /// from the root) and the blocks they carried.
    pub fn record_kv_event_parentless(worker_url: &str, stores: u64, blocks: u64) {
        let worker = intern_string(worker_url);
        counter!("smg_kv_event_parentless_stores_total", "worker" => worker.clone())
            .increment(stores);
        counter!("smg_kv_event_parentless_blocks_total", "worker" => worker).increment(blocks);
    }

    /// Count the stores a batch had declined at an arena cap (a run's fan-out or
    /// length) and the blocks they left out of the index.
    pub fn record_kv_event_cap_refused(worker_url: &str, stores: u64, blocks: u64) {
        let worker = intern_string(worker_url);
        counter!("smg_kv_event_cap_refused_stores_total", "worker" => worker.clone())
            .increment(stores);
        counter!("smg_kv_event_cap_refused_blocks_total", "worker" => worker).increment(blocks);
    }

    /// Count the stores a batch left out of the index as narrower than the
    /// worker's block of record, and the blocks they carried.
    pub fn record_kv_event_sub_block(worker_url: &str, stores: u64, blocks: u64) {
        let worker = intern_string(worker_url);
        counter!("smg_kv_event_sub_block_stores_total", "worker" => worker.clone())
            .increment(stores);
        counter!("smg_kv_event_sub_block_blocks_total", "worker" => worker).increment(blocks);
    }

    /// Time to apply one batch to the index.
    pub fn record_kv_event_apply(worker_url: &str, seconds: f64) {
        Self::with_kv_event_handles(worker_url, |kept| {
            kept.apply
                .get_or_insert_with(
                    || histogram!(KV_EVENT_APPLY_SECONDS, "worker" => intern_string(worker_url)),
                )
                .record(seconds);
        });
    }

    /// Ranks of this worker whose index may be stale.
    pub fn set_kv_event_degraded_ranks(worker_url: &str, count: usize) {
        gauge!("smg_kv_event_degraded_ranks", "worker" => intern_string(worker_url))
            .set(count as f64);
    }

    /// Publish the blocks the index holds for a worker. Called from
    /// the KV event subscriber where it already counts applied batches, never
    /// from the lookup path, so routing reads nothing that writes.
    pub fn set_kv_index_blocks(worker_url: &str, blocks: usize) {
        Self::with_kv_event_handles(worker_url, |kept| {
            kept.index_blocks
                .get_or_insert_with(
                    || gauge!("smg_kv_index_blocks", "worker" => intern_string(worker_url)),
                )
                .set(blocks as f64);
        });
    }

    /// Publish a model's KV index size: memberships across workers and
    /// distinct entries. Called from the monitor's periodic stats task, never
    /// from the lookup path.
    pub fn set_kv_index_size(model_id: &str, memberships: usize, entries: usize) {
        let model = intern_string(model_id);
        gauge!("smg_kv_index_memberships", "model" => model.clone()).set(memberships as f64);
        gauge!("smg_kv_index_entries", "model" => model).set(entries as f64);
    }

    /// Publish the chain index's shape and memory for a model, from the
    /// counters its holder copied: live runs and blocks, arena and slab bytes,
    /// moved hashes and engine conflicts.
    pub fn set_kv_index_chain_stats(model_id: &str, stats: &KvIndexChainStats) {
        let model = intern_string(model_id);
        gauge!("smg_kv_index_runs_live", "model" => model.clone()).set(stats.runs_live as f64);
        gauge!("smg_kv_index_blocks_live", "model" => model.clone()).set(stats.blocks_live as f64);
        gauge!("smg_kv_index_arena_bytes", "model" => model.clone()).set(stats.arena_bytes as f64);
        gauge!("smg_kv_index_arena_free_bytes", "model" => model.clone())
            .set(stats.arena_free_bytes as f64);
        gauge!("smg_kv_index_arena_chunk_bytes", "model" => model.clone())
            .set(stats.arena_chunk_bytes as f64);
        gauge!("smg_kv_index_slab_bytes", "model" => model.clone()).set(stats.slab_bytes as f64);
        gauge!("smg_kv_index_moved_hashes", "model" => model.clone())
            .set(stats.moved_hashes as f64);
        gauge!("smg_kv_index_engine_conflicts", "model" => model.clone())
            .set(stats.engine_conflicts as f64);
        gauge!("smg_kv_index_cap_refusals", "model" => model).set(stats.cap_refusals as f64);
    }

    // ========================================================================
    // Layer 3: Worker resilience metrics (circuit breaker)
    // ========================================================================

    /// Set circuit breaker state (0=closed, 1=open, 2=half_open)
    pub fn set_worker_cb_state(worker: &str, state_code: u8) {
        let worker_interned = intern_string(worker);
        gauge!(
            "smg_worker_cb_state",
            "worker" => worker_interned
        )
        .set(state_code as f64);
    }

    /// Record circuit breaker state transition
    pub fn record_worker_cb_transition(worker: &str, from: &'static str, to: &'static str) {
        let worker_interned = intern_string(worker);
        counter!(
            "smg_worker_cb_transitions_total",
            "worker" => worker_interned,
            "from" => from,
            "to" => to
        )
        .increment(1);
    }

    /// Publish a circuit breaker transition counter at zero, so the series
    /// exists before the breaker first makes that transition.
    pub fn init_worker_cb_transition(worker: &str, from: &'static str, to: &'static str) {
        let worker_interned = intern_string(worker);
        counter!(
            "smg_worker_cb_transitions_total",
            "worker" => worker_interned,
            "from" => from,
            "to" => to
        )
        .absolute(0);
    }

    /// Record circuit breaker outcome: through the handle `worker` keeps
    /// for the outcome (dropped when the worker set changes, as every kept
    /// per-worker handle is), or through a key built per call when handles
    /// are not kept.
    pub fn record_worker_cb_outcome(worker: &WorkerHandles, outcome: &'static str) {
        let build = || {
            counter!(
                "smg_worker_cb_outcomes_total",
                "worker" => intern_string(&worker.url),
                "outcome" => outcome
            )
        };
        match (worker.kept(), outcome) {
            (Some(kept), crate::metrics_labels::CB_SUCCESS) => {
                kept.cb_outcome_success.get_or_init(build).increment(1);
            }
            (Some(kept), crate::metrics_labels::CB_FAILURE) => {
                kept.cb_outcome_failure.get_or_init(build).increment(1);
            }
            _ => build().increment(1),
        }
    }

    /// Record an open circuit breaker put on trial (half-open) by the
    /// worker's return to health rather than by its timeout.
    pub fn record_worker_cb_health_readmission(worker: &str) {
        let worker_interned = intern_string(worker);
        counter!(
            "smg_worker_cb_health_readmissions_total",
            "worker" => worker_interned
        )
        .increment(1);
    }

    /// Publish the health-readmission counter at zero, so the series exists
    /// before the worker's first return.
    pub fn init_worker_cb_health_readmissions(worker: &str) {
        let worker_interned = intern_string(worker);
        counter!(
            "smg_worker_cb_health_readmissions_total",
            "worker" => worker_interned
        )
        .absolute(0);
    }

    /// Set circuit breaker consecutive failures
    pub fn set_worker_cb_consecutive_failures(worker: &WorkerHandles, count: u32) {
        Self::set_worker_gauge(
            worker,
            |kept| &kept.cb_failures,
            f64::from(count),
            || {
                gauge!(
                    "smg_worker_cb_consecutive_failures",
                    "worker" => intern_string(&worker.url)
                )
            },
        );
    }

    /// Set circuit breaker consecutive successes
    pub fn set_worker_cb_consecutive_successes(worker: &WorkerHandles, count: u32) {
        Self::set_worker_gauge(
            worker,
            |kept| &kept.cb_successes,
            f64::from(count),
            || {
                gauge!(
                    "smg_worker_cb_consecutive_successes",
                    "worker" => intern_string(&worker.url)
                )
            },
        );
    }

    // ========================================================================
    // Layer 3: Engine load re-export
    // ========================================================================

    /// Re-export a worker's `GetLoads` snapshot as `smg_engine_*` gauges:
    /// every rank the report carries, as a poll's does. A pushed record
    /// names one rank and goes through [`Self::record_engine_rank_load`].
    ///
    /// Core gauges are per DP rank (`dp_rank` bounded by dp_size) and carry
    /// `role`, the role their worker is registered in
    /// ([`registered_engine_role`]), so a pool's load splits by role whether
    /// or not its engines report one. PD gauges are emitted only for ranks
    /// that carry a `disagg` section, labeled by the engine-reported role
    /// (`prefill`/`decode`/`null`).
    pub fn record_engine_load(
        worker: &dyn Worker,
        model_id: &str,
        role: &'static str,
        response: &crate::worker::WorkerLoadResponse,
    ) {
        for load in &response.loads {
            Self::record_engine_rank_load(worker, model_id, role, load);
        }
    }

    /// Re-export one rank's figures as its `smg_engine_*` gauges: through
    /// the handles `worker` keeps for the model and rank (built on the
    /// rank's first sight, dropped when the worker set changes), or
    /// through keys built per call when handles are not kept. A record
    /// pushed at the engine's batch rate is then a few atomic stores, with
    /// nothing built or allocated, and touches no other rank's series.
    pub fn record_engine_rank_load(
        worker: &dyn Worker,
        model_id: &str,
        role: &'static str,
        load: &crate::worker::SchedulerLoadSnapshot,
    ) {
        let worker_url = worker.url();
        if let Some(kept) = worker.metric_handles().and_then(WorkerHandles::kept) {
            let mut kept_loads = kept.engine_loads.lock();
            let by_model = &mut *kept_loads;
            // A model seen before is found by one lookup, and a new rank
            // under it is made here as `engine_rank` makes it. `engine_rank`
            // looks twice, to insert on a first sight, which still goes
            // through it.
            let rank = match by_model.get_mut(model_id) {
                Some(ranks) => Some(ranks.entry(load.dp_rank).or_default()),
                None => handles::engine_rank(by_model, model_id, load.dp_rank),
            };
            let recorded = rank
                .map(|rank| {
                    Self::set_engine_rank_gauges(rank, worker_url, model_id, role, load);
                })
                .is_some();
            if recorded {
                return;
            }
        }
        // Nothing kept (no recorder installed yet, a worker that holds no
        // handles, or a model label the interner refused): the keys are
        // built for this call alone.
        Self::set_engine_rank_gauges(
            &mut handles::EngineRankHandles::default(),
            worker_url,
            model_id,
            role,
            load,
        );
    }

    /// Set a rank's gauges through `rank`'s handles, each built the first
    /// time the rank has a figure for its series; the labels are interned
    /// once per record that builds a handle and not at all otherwise.
    fn set_engine_rank_gauges(
        rank: &mut handles::EngineRankHandles,
        worker_url: &str,
        model_id: &str,
        role: &'static str,
        load: &crate::worker::SchedulerLoadSnapshot,
    ) {
        let dp_rank = load.dp_rank;
        // The role is the worker's, and a worker of another role at the URL
        // is another registration, which drops the kept handles: one that
        // still names another role here is not this writer's.
        if rank
            .core
            .as_ref()
            .is_some_and(|core| !std::ptr::eq(core.registered, role) && core.role.as_ref() != role)
        {
            rank.core = None;
            rank.kv_capacity = None;
        }
        let mut labels = None;
        let core = rank.core.get_or_insert_with(|| {
            handles::EngineCoreHandles::new(
                handles::engine_labels(&mut labels, worker_url, model_id, dp_rank),
                role,
            )
        });
        core.running.set(load.num_running_reqs as f64);
        core.waiting.set(load.num_waiting_reqs as f64);
        core.token_usage.set(load.token_usage);
        core.gen_throughput.set(load.gen_throughput);
        core.cache_hit_rate.set(load.cache_hit_rate);
        // The capacity behind the usage ratio, when the engine reports
        // it (a snapshot read off an HTTP metrics page carries the
        // ratio alone and leaves it 0).
        if load.max_total_num_tokens > 0 {
            rank.kv_capacity
                .get_or_insert_with(|| {
                    handles::engine_gauge(
                        "smg_engine_kv_capacity_tokens",
                        handles::engine_labels(&mut labels, worker_url, model_id, dp_rank),
                        &core.role,
                    )
                })
                .set(f64::from(load.max_total_num_tokens));
        }

        // PD gauges only when the engine reported a disagg section. Labeled
        // by dp_rank too, so DP ranks sharing a role don't overwrite.
        let Some(role) = load.disagg_mode.as_deref() else {
            return;
        };
        let pd = rank.pd(role);
        if let Some(latency) = load.kv_transfer_latency_ms {
            pd.kv_transfer_latency
                .get_or_insert_with(|| {
                    handles::engine_pd_gauge(
                        "smg_engine_pd_kv_transfer_latency_ms",
                        handles::engine_labels(&mut labels, worker_url, model_id, dp_rank),
                        &pd.role,
                    )
                })
                .set(latency);
        }
        if let Some(speed) = load.kv_transfer_speed_gb_s {
            pd.kv_transfer_speed
                .get_or_insert_with(|| {
                    handles::engine_pd_gauge(
                        "smg_engine_pd_kv_transfer_speed_gb_s",
                        handles::engine_labels(&mut labels, worker_url, model_id, dp_rank),
                        &pd.role,
                    )
                })
                .set(speed);
        }
        if let Some(reqs) = load.prefill_queue_reqs {
            pd.prefill_queue
                .get_or_insert_with(|| {
                    handles::engine_pd_gauge(
                        "smg_engine_pd_prefill_queue_reqs",
                        handles::engine_labels(&mut labels, worker_url, model_id, dp_rank),
                        &pd.role,
                    )
                })
                .set(reqs as f64);
        }
        if let Some(reqs) = load.decode_queue_reqs {
            pd.decode_queue
                .get_or_insert_with(|| {
                    handles::engine_pd_gauge(
                        "smg_engine_pd_decode_queue_reqs",
                        handles::engine_labels(&mut labels, worker_url, model_id, dp_rank),
                        &pd.role,
                    )
                })
                .set(reqs as f64);
        }
    }

    // ========================================================================
    // Worker cleanup
    // ========================================================================

    /// A worker registered at `worker_url`: its series record again if the
    /// address was retired by an earlier removal. Only a retirement leaves a
    /// no-op handle kept, so the kept handles go only when the store says one
    /// may be: this address was retired, or the upkeep forgot a retired
    /// address since the last registration.
    pub fn worker_registered(worker_url: &str) {
        let _registering = handles::REGISTRATION.lock();
        if series::installed().is_some_and(|store| store.activate(worker_url)) {
            handles::worker_set_changed();
        }
    }

    /// `worker`, replaced at `worker_url` by another instance, may still
    /// write for the address from the handles that hold it (a load guard
    /// taken before the replacement): the store keeps its witness, so a
    /// later removal of the address drops those writes until it is gone too.
    pub fn worker_replaced(worker_url: &str, worker: Weak<dyn Worker>) {
        if let Some(store) = series::installed() {
            store.replaced(worker_url, worker);
        }
    }

    /// The registration at `worker_url` was removed: drop the address's
    /// series (every family labelled `worker`, see `series`), so the
    /// scrape and the time-series database stop growing with pod churn, and
    /// drop the writes that arrive after the removal until the address
    /// registers again or `worker`, the removed one, is gone from every
    /// handle that could still write for it. The interned label goes with
    /// them.
    pub fn remove_worker_metrics(worker_url: &str, worker: Weak<dyn Worker>) {
        if let Some(store) = series::installed() {
            store.retire(worker_url, worker);
        }
        handles::worker_set_changed();
        STRING_INTERNER.remove(worker_url);
    }

    /// Sentinel-out one rank's `smg_engine_*` series at a worker's URL: the
    /// six load gauges under each of `models`, which carry the worker, the
    /// model and the rank, and the four pair gauges under every role
    /// (prefill/decode/null), which carry the worker, the engine's role
    /// and the rank, and no model. The collector sets them for a worker
    /// whose load feed ended while its URL stays registered: its status
    /// left `Ready`, or a same-URL replacement took its place (see
    /// `WorkerMonitor::clear_series`). The worker series store keeps those
    /// series, so the last load would read as live. Each gauge is set to
    /// -1, an impossible value for gauges whose 0 is meaningful. A removed
    /// worker needs no sentinel: the store drops every `worker`-labelled
    /// series with the registration ([`Self::remove_worker_metrics`]) and
    /// no-ops the writes that follow, this one included.
    ///
    /// A series a live writer writes is left. The writer asks `live`,
    /// series by series: does a live writer at the URL hold an entry of
    /// this rank that passes the series' test, under a model's label for
    /// the load gauges and under any (`None`) for the pair gauges? A load
    /// series carries its model's label, and two models share one past
    /// the label cap ([`intern_model_label`]). The tests are
    /// [`Self::record_engine_rank_load`]'s own conditions: any entry of a
    /// writer registered in the same role for the five core gauges, one
    /// that carries the capacity for the capacity gauge, one that reports
    /// the role and carries the figure for a pair gauge. A load series
    /// carries its worker's registered role, so `role` is the ended
    /// worker's and the test of a live writer's entry is told that
    /// writer's ([`LoadTest`]): a successor registered in another role
    /// writes other series and keeps no sentinel from these. The collector
    /// answers over every entry of the rank its live writers hold: a
    /// rollup's row names a rank once a downstream worker, and a poll
    /// writes each. The label sets must
    /// match that writer's exactly, or a fresh series is created instead
    /// of overwriting the live one. One call a rank: no rank between two
    /// is written, and the rank's label is built once.
    pub(crate) fn remove_engine_rank_metrics(
        worker_url: &str,
        models: &[String],
        dp_rank: i32,
        role: &'static str,
        live: impl Fn(Option<&str>, &LoadTest<'_>) -> bool,
    ) {
        let worker = intern_string(worker_url);
        let dp_rank = rank_label(dp_rank);
        for model_id in models {
            let model = intern_model_label(model_id);
            let core = live(Some(&model), &|_, writer| writer == role);
            let capacity = live(Some(&model), &|load, writer| {
                writer == role && load.max_total_num_tokens > 0
            });
            if core && capacity {
                continue;
            }
            for (name, written) in [
                ("smg_engine_running_requests", core),
                ("smg_engine_waiting_requests", core),
                ("smg_engine_token_usage", core),
                ("smg_engine_gen_throughput", core),
                ("smg_engine_cache_hit_rate", core),
                ("smg_engine_kv_capacity_tokens", capacity),
            ] {
                if !written {
                    gauge!(
                        name,
                        "worker" => Arc::clone(&worker),
                        "model" => Arc::clone(&model),
                        "dp_rank" => Arc::clone(&dp_rank),
                        "role" => role,
                    )
                    .set(-1.0);
                }
            }
        }
        for role in ["prefill", "decode", "null"] {
            let carries = |figure: fn(&crate::worker::SchedulerLoadSnapshot) -> bool| {
                live(None, &|load, _writer| {
                    load.disagg_mode.as_deref() == Some(role) && figure(load)
                })
            };
            for (name, written) in [
                (
                    "smg_engine_pd_kv_transfer_latency_ms",
                    carries(|load| load.kv_transfer_latency_ms.is_some()),
                ),
                (
                    "smg_engine_pd_kv_transfer_speed_gb_s",
                    carries(|load| load.kv_transfer_speed_gb_s.is_some()),
                ),
                (
                    "smg_engine_pd_prefill_queue_reqs",
                    carries(|load| load.prefill_queue_reqs.is_some()),
                ),
                (
                    "smg_engine_pd_decode_queue_reqs",
                    carries(|load| load.decode_queue_reqs.is_some()),
                ),
            ] {
                if !written {
                    gauge!(
                        name,
                        "worker" => Arc::clone(&worker),
                        "role" => role,
                        "dp_rank" => Arc::clone(&dp_rank),
                    )
                    .set(-1.0);
                }
            }
        }
    }
}

/// A test of one entry of a load, for the sentinel's guard
/// ([`Metrics::remove_engine_rank_metrics`]): whether the entry carries
/// the figure of the series in question. The second argument is the
/// registered role of the live writer that holds the entry
/// ([`registered_engine_role`]), the role that writer's load series carry.
pub(crate) type LoadTest<'t> =
    dyn Fn(&crate::worker::SchedulerLoadSnapshot, &'static str) -> bool + 't;

/// The `role` label of a worker's engine load series: the type the worker is
/// registered under. `prefill` and `decode` for the two sides of a
/// disaggregated pool, and `null`, the engines' own spelling of a worker
/// that is neither, for any other, so the label reads as the pair series'
/// does. It is the registration's and not the load's: every engine's load
/// then splits by role, one that reports none included, and a worker's
/// series stand under one role for as long as it is registered.
pub fn registered_engine_role(worker_type: crate::worker::WorkerType) -> &'static str {
    match worker_type {
        crate::worker::WorkerType::Prefill => "prefill",
        crate::worker::WorkerType::Decode => "decode",
        crate::worker::WorkerType::Regular | crate::worker::WorkerType::Encode => "null",
    }
}

#[cfg(test)]
mod tests;
