//! Worker load monitoring service.
//!
//! `WorkerMonitor` consolidates the previous `LoadMonitor` (per-group
//! polling loops) and `WorkerLoadManager` (DP rank load cache) into a
//! single coordinator that subscribes to `WorkerRegistry` events.
//!
//! ## Lifecycle
//!
//! - `new()` creates the cache and watch channel without spawning any
//!   tasks. The factory layer reads `worker_load_manager` immediately so
//!   policies that need the DP cache can be wired before workers exist.
//! - `start_event_loop()` subscribes to the registry, runs a synchronous
//!   bootstrap reconcile, and spawns the background event-handling task.
//!   It must be called after the initial worker population (mesh replay,
//!   K8s discovery, etc.) has finished — same ordering rule as
//!   `WorkerManager::start`.
//! - `Drop` aborts the event task and every per-group polling loop.
//!
//! ## Group lifecycle (event-driven)
//!
//! Polling is keyed by `WorkerGroupKey = (model_id, worker_type,
//! connection_mode)`. The event loop reacts to:
//!
//! - `Registered` / `Replaced`: reconcile every group the worker
//!   participates in. New groups start a polling loop; existing groups
//!   restart with a new interval if the per-worker override changed.
//! - `Removed`: reconcile every group the removed worker participated
//!   in. Empty groups stop their loops and evict cached state.
//! - `StatusChanged`: workers leaving `Ready` are evicted from the
//!   watch-channel snapshot and the DP cache (the group loop is left
//!   alone and will skip non-Ready workers on its next tick).
//! - `RecvError::Lagged`: stop every loop, clear shared state, and
//!   rebuild from the current registry snapshot. Monitoring state is
//!   derived data; full rebuild is the recovery mechanism.
//!
//! ## Polling
//!
//! Each group runs a single `tokio::time::interval` loop. Every tick:
//!
//! 1. Fetch the group's `Ready` workers. Polling is unconditional by
//!    default; with `--disable-load-monitoring` the tick is skipped
//!    unless a load-aware policy, engine-metrics re-export, or overload
//!    protection on a group member needs the data (the original
//!    conditional gate — never "never poll").
//! 2. Fetch loads concurrently from every `Ready` worker in the group.
//! 3. Update load-aware policies and the DP cache.
//! 4. Atomically clear stale entries for the group from the watch
//!    channel and merge in the fresh loads.

use std::{
    collections::HashMap,
    fmt::Debug,
    sync::{Arc, Weak},
    time::Duration,
};

use dashmap::DashSet;
use futures::future;
use openai_protocol::worker::{
    RuntimeType, SchedulerLoadSnapshot, WorkerGroupKey, WorkerLoadResponse, WorkerStatus,
};
use parking_lot::{Mutex, RwLock};
use reqwest::StatusCode;
use tokio::{
    sync::{broadcast, watch},
    task::JoinHandle,
};
use tracing::{debug, info, warn};

use crate::{
    observability::metrics::Metrics,
    policies::PolicyRegistry,
    worker::{event::WorkerEvent, ConnectionMode, Worker, WorkerRegistry},
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Minimal Prometheus text-format scraper: metric name -> sample values
/// (one per label-set). Only the flat `name{labels} value` / `name value`
/// gauge lines that the engine load fetchers look up are collected;
/// `#` comment lines and unparsable values are skipped, and histogram or
/// summary buckets are simply never queried by name.
struct PromScrape {
    samples: HashMap<String, Vec<f64>>,
}

impl PromScrape {
    fn parse(text: &str) -> Self {
        let mut samples: HashMap<String, Vec<f64>> = HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // Line shape: `name{labels} value [timestamp]`. Split the head
            // (`name{labels}`) from the tail at the end of the label block so
            // spaces inside quoted label values don't confuse the split; fall
            // back to the first whitespace for unlabeled metrics. Then take
            // the FIRST tail token as the value, ignoring any optional
            // trailing timestamp.
            let (head, tail) = match line.rfind('}') {
                Some(close) => (&line[..=close], &line[close + 1..]),
                None => match line.split_once(char::is_whitespace) {
                    Some(split) => split,
                    None => continue,
                },
            };
            let Some(value) = tail.split_whitespace().next() else {
                continue;
            };
            let Ok(value) = value.parse::<f64>() else {
                continue;
            };
            let name = head.split('{').next().unwrap_or(head).trim();
            samples.entry(name.to_string()).or_default().push(value);
        }
        Self { samples }
    }

    /// Sum across all label-set samples for `name` (0.0 if absent). Right for
    /// additive counts like running/waiting requests across DP ranks.
    fn sum(&self, name: &str) -> f64 {
        self.samples
            .get(name)
            .map(|v| v.iter().sum())
            .unwrap_or(0.0)
    }

    /// Mean across all label-set samples for `name` (0.0 if absent). Right for
    /// ratios like KV-cache usage that must not be double-counted.
    fn mean(&self, name: &str) -> f64 {
        match self.samples.get(name) {
            Some(v) if !v.is_empty() => v.iter().sum::<f64>() / v.len() as f64,
            _ => 0.0,
        }
    }

    /// True when at least one sample exists for `name`.
    fn has(&self, name: &str) -> bool {
        self.samples.get(name).is_some_and(|v| !v.is_empty())
    }
}

/// DP rank load cache used by load-aware routing policies.
///
/// Pure in-memory data structure with no I/O. `WorkerMonitor` owns the
/// shared `Arc<WorkerLoadManager>` and updates it on every successful
/// poll; routing policies read from it via
/// `select_and_increment_lowest_dp_load`.
#[derive(Debug, Default)]
pub struct WorkerLoadManager {
    /// `<worker_url, <dp_rank, load>>`
    dp_cached_loads: RwLock<HashMap<String, HashMap<isize, isize>>>,
}

impl WorkerLoadManager {
    pub fn new() -> Self {
        Self {
            dp_cached_loads: RwLock::new(HashMap::new()),
        }
    }

    pub fn update_dp_loads(&self, loads: &HashMap<String, HashMap<isize, isize>>) {
        debug!("WorkerLoadManager update_dp_loads map:{:?}", loads);
        let mut cached = self.dp_cached_loads.write();
        cached.extend(loads.iter().map(|(k, v)| (k.clone(), v.clone())));
    }

    pub fn select_and_increment_lowest_dp_load(
        &self,
        worker: &dyn Worker,
        increment: isize,
    ) -> Option<isize> {
        let mut cached = self.dp_cached_loads.write();
        let loads = cached.get_mut(worker.url())?;
        let (&dp_rank, _) = loads.iter().min_by_key(|&(rank, load)| (*load, *rank))?;
        if let Some(v) = loads.get_mut(&dp_rank) {
            *v += increment;
        }
        Some(dp_rank)
    }

    pub fn remove_workers(&self, urls: &[String]) {
        let mut cached = self.dp_cached_loads.write();
        for url in urls {
            cached.remove(url);
        }
    }

    /// Drop a single worker's cached DP loads. Avoids the
    /// `&[String]` allocation that `remove_workers` requires when the
    /// caller only has a single `&str`.
    pub fn remove_worker(&self, url: &str) {
        self.dp_cached_loads.write().remove(url);
    }

    /// Drop every cached DP load entry. Used by `WorkerMonitor` during
    /// `stop_all_groups` and lag-recovery rebuilds so the cache cannot
    /// hand out stale per-rank loads after a full reconcile.
    pub fn clear(&self) {
        self.dp_cached_loads.write().clear();
    }
}

/// Per-group polling loop state.
struct GroupState {
    handle: JoinHandle<()>,
    interval: Duration,
}

/// Outcome of probing a worker's `/v1/loads` endpoint.
enum NativeLoads {
    Available(WorkerLoadResponse),
    /// The backend answered, and the endpoint is not there. Memoizable.
    Absent,
    /// Nothing was learned about the endpoint — retry on the next tick.
    Inconclusive,
}

/// Load monitoring service that subscribes to `WorkerRegistry` events.
pub struct WorkerMonitor {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    pub worker_load_manager: Arc<WorkerLoadManager>,
    client: reqwest::Client,
    default_interval: Duration,
    /// When set, poll loads and re-export `smg_engine_*` gauges even if no
    /// load-aware routing policy is active (`--engine-metrics`).
    engine_metrics: bool,
    /// `--disable-load-monitoring`: restore the conditional poll gate (a
    /// load-aware/dp-rank policy, engine metrics, or overload protection on a
    /// group member still forces the poll) instead of polling every group
    /// unconditionally from registration onward.
    conditional_polling: bool,
    load_tx: watch::Sender<HashMap<String, WorkerLoadResponse>>,
    load_rx: watch::Receiver<HashMap<String, WorkerLoadResponse>>,
    /// Worker URLs that answered definitively that they do not serve
    /// `/v1/loads`, so the probe is not repeated every tick. Cleared by
    /// [`Self::evict_worker_loads`], which also runs on `Replaced` — an
    /// in-place image upgrade that adds the endpoint is re-probed.
    native_loads_absent: Arc<DashSet<String>>,
    group_handles: Mutex<HashMap<WorkerGroupKey, GroupState>>,
    event_task: Mutex<Option<JoinHandle<()>>>,
}

impl Debug for WorkerMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerMonitor")
            .field("default_interval", &self.default_interval)
            .finish_non_exhaustive()
    }
}

impl WorkerMonitor {
    /// Construct a `WorkerMonitor` without spawning any background work.
    ///
    /// The caller must invoke [`Self::start_event_loop`] once initial
    /// workers have been registered. Until then, `worker_load_manager`
    /// is still safe to read; it just stays empty.
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        default_interval_secs: u64,
        engine_metrics: bool,
        conditional_polling: bool,
    ) -> Self {
        let (load_tx, load_rx) = watch::channel(HashMap::new());
        Self {
            worker_registry,
            policy_registry,
            worker_load_manager: Arc::new(WorkerLoadManager::new()),
            client,
            default_interval: Duration::from_secs(default_interval_secs.max(1)),
            engine_metrics,
            conditional_polling,
            load_tx,
            load_rx,
            native_loads_absent: Arc::new(DashSet::new()),
            group_handles: Mutex::new(HashMap::new()),
            event_task: Mutex::new(None),
        }
    }

    /// The shared `/v1/loads` probe memo, so on-demand load sweeps skip
    /// backends the polling loop has already found to lack the endpoint.
    pub(crate) fn native_loads_absent(&self) -> &DashSet<String> {
        &self.native_loads_absent
    }

    /// Subscribe to the snapshot of per-worker loads.
    ///
    /// The watch receiver returns the most recent fully merged map;
    /// stale entries are pruned on each tick of the relevant group.
    pub fn subscribe(&self) -> watch::Receiver<HashMap<String, WorkerLoadResponse>> {
        self.load_rx.clone()
    }

    /// Subscribe to registry events, run a synchronous bootstrap
    /// reconcile, and spawn the background event task.
    ///
    /// Subscribing **before** the bootstrap reconcile guarantees the
    /// "registered between subscribe and reconcile" race resolves
    /// idempotently: any registration that lands after this call either
    /// is already in the snapshot (because it ran on this thread) or
    /// arrives as a `Registered` event that the loop applies on top of
    /// the snapshot.
    pub fn start_event_loop(self: &Arc<Self>) {
        // Capture the receiver before reconciling so events that fire
        // during reconcile are buffered, not lost.
        let events_rx = self.worker_registry.subscribe_events();

        // Synchronous bootstrap: build initial group set from the
        // registry snapshot and start polling loops.
        self.reconcile_from_registry();

        // Hand the spawned task a `Weak<Self>` so it does not pin the
        // monitor in memory across drops. If we passed an `Arc<Self>`
        // the spawned task would form a cycle: the monitor stores the
        // task's `JoinHandle`, and the task closure holds an `Arc`
        // back to the monitor. The strong count would never reach
        // zero, `Drop` would never run, and the abort calls would be
        // unreachable outside of full runtime shutdown.
        let monitor = Arc::downgrade(self);
        #[expect(
            clippy::disallowed_methods,
            reason = "WorkerMonitor event loop runs for the lifetime of the registry; the JoinHandle is stored on the monitor and aborted in Drop"
        )]
        let handle = tokio::spawn(async move {
            run_event_loop(monitor, events_rx).await;
        });

        *self.event_task.lock() = Some(handle);
    }

    /// Stop every per-group polling loop and clear the shared load
    /// snapshot + DP-rank cache.
    ///
    /// Called from `Drop`, the bootstrap reconcile path, and the
    /// `RecvError::Lagged` rebuild. Always clears both caches even
    /// when no groups were running, so a lag-recovery rebuild cannot
    /// inherit stale per-rank loads from a previous incarnation.
    /// Idempotent.
    pub fn stop_all_groups(&self) {
        let drained: Vec<(WorkerGroupKey, GroupState)> = {
            let mut handles = self.group_handles.lock();
            handles.drain().collect()
        };

        if !drained.is_empty() {
            info!("Stopping all {} load monitor groups", drained.len());
            for (key, state) in drained {
                debug!("Stopping load monitor group: {key}");
                state.handle.abort();
            }
        }

        // Always clear every cache. Skipping when `drained.is_empty()`
        // would leave stale per-rank loads behind for any caller that
        // seeded the cache without going through a group loop, and
        // makes the function harder to reason about as a "reset".
        //
        // The probe memo is dropped with the rest rather than retained
        // per live URL: workers removed during the lag window would
        // otherwise leak entries, and the cost is one re-probe per
        // worker on a path that only runs on lag recovery.
        self.load_tx.send_modify(|map| map.clear());
        self.worker_load_manager.clear();
        self.native_loads_absent.clear();
        // The feed that would clear the vetoes is being torn down: fail open,
        // but say so — a wiped gauge is otherwise indistinguishable from a
        // genuine recovery. With no thresholds anywhere no flag was ever
        // written, so this clears nothing and stays silent.
        let cleared = self.worker_registry.clear_all_overload_flags();
        if cleared > 0 {
            warn!(
                vetoes_cleared = cleared,
                "Overload vetoes cleared with the load feed; every worker is routable again \
                 until its next poll"
            );
        }
    }

    /// Recompute the polling state for every currently-known group.
    ///
    /// Used as the synchronous bootstrap path and as the lag-recovery
    /// rebuild after `RecvError::Lagged`. Stops every existing loop,
    /// clears the cached snapshot, then walks the registry to start
    /// fresh loops for each non-empty group.
    fn reconcile_from_registry(self: &Arc<Self>) {
        // Stop everything first so the rebuild starts from a clean slate.
        self.stop_all_groups();

        // Walk every worker once and bucket them by group.
        let workers = self.worker_registry.get_all();
        let mut group_keys: HashMap<WorkerGroupKey, ()> = HashMap::new();
        for worker in workers {
            for key in group_keys_for_worker(&worker) {
                group_keys.insert(key, ());
            }
        }

        for key in group_keys.into_keys() {
            self.reconcile_group(&key);
        }
    }

    /// Bring a single group's polling state into sync with the registry.
    ///
    /// - Empty group: stop the loop (if any), evict the group's cached
    ///   loads from the watch channel and DP cache.
    /// - Non-empty group, no current loop: spawn one with the group's
    ///   desired interval.
    /// - Non-empty group, loop exists with a stale interval: stop and
    ///   restart so the new interval takes effect.
    /// - Non-empty group, loop exists with the correct interval: no-op.
    fn reconcile_group(self: &Arc<Self>, key: &WorkerGroupKey) {
        let workers = self.worker_registry.get_workers_filtered(
            Some(&key.model_id),
            Some(key.worker_type),
            Some(key.connection_mode),
            None,
            false,
        );

        if workers.is_empty() {
            self.stop_group(key);
            return;
        }

        let desired_interval = group_interval(&workers, self.default_interval);

        let needs_start = {
            let mut handles = self.group_handles.lock();
            match handles.get(key) {
                Some(state) if state.interval == desired_interval => false,
                Some(_) => {
                    // Interval changed — stop the old loop and fall
                    // through to start a fresh one below.
                    if let Some(old) = handles.remove(key) {
                        debug!("Restarting load monitor group {key} with new interval {desired_interval:?}");
                        old.handle.abort();
                    }
                    true
                }
                None => true,
            }
        };

        if needs_start {
            self.spawn_group_loop(key.clone(), desired_interval);
        }
    }

    /// Stop a single group's polling loop.
    ///
    /// Does **not** evict per-worker cached loads from the watch
    /// channel or DP cache. Callers that know the affected URLs (e.g.
    /// the event loop on `Removed` / `Replaced` / `StatusChanged`)
    /// must invoke [`Self::evict_worker_loads`] separately. The
    /// per-group polling loop itself prunes URLs on the next tick when
    /// the worker is gone, but a stopped loop never gets a next tick.
    fn stop_group(&self, key: &WorkerGroupKey) {
        let removed = {
            let mut handles = self.group_handles.lock();
            handles.remove(key)
        };

        if let Some(state) = removed {
            info!("Stopping load monitor for empty group {key}");
            state.handle.abort();
        }
    }

    /// Evict a single worker's cached loads from both the watch
    /// channel snapshot and the DP cache. Used by the event loop on
    /// `Removed`, `Replaced`, and `StatusChanged` away from `Ready`.
    ///
    /// Also sentinels the worker's `smg_engine_*` series when engine-metrics
    /// re-export is on, since metrics-rs cannot delete series.
    fn evict_worker_loads(&self, worker: &Arc<dyn Worker>) {
        let url = worker.url();
        // No load feed, no verdict: a flag left set would strand the worker
        // out of routing with nothing to ever clear it.
        self.worker_registry.set_worker_overloaded(worker, false);
        self.load_tx.send_modify(|map| {
            map.remove(url);
        });
        self.worker_load_manager.remove_worker(url);
        self.native_loads_absent.remove(url);
        if self.engine_metrics {
            // A worker can serve multiple models (one load group per model),
            // so sentinel every model's series — not just the primary.
            let dp_size = worker.dp_size().unwrap_or(1);
            for model_id in WorkerRegistry::worker_model_ids(worker) {
                Metrics::remove_engine_load_metrics(url, &model_id, dp_size);
            }
        }
    }

    /// Spawn the polling loop for a single group.
    fn spawn_group_loop(self: &Arc<Self>, key: WorkerGroupKey, interval: Duration) {
        info!("Starting load monitor for group {key} with interval {interval:?}");

        // Same `Weak<Self>` rationale as `start_event_loop`: the
        // group loop is owned by `group_handles` on the monitor, so
        // it must not own a strong reference back to the monitor.
        let monitor = Arc::downgrade(self);
        let group_key = key.clone();

        #[expect(
            clippy::disallowed_methods,
            reason = "Group polling loop runs for the lifetime of the group; the JoinHandle is stored in group_handles and aborted on group removal or monitor drop"
        )]
        let handle = tokio::spawn(async move {
            group_monitor_loop(monitor, group_key, interval).await;
        });

        let mut handles = self.group_handles.lock();
        handles.insert(key, GroupState { handle, interval });
    }

    /// Fetch load over HTTP, preferring `/v1/loads` on any backend that
    /// serves it and falling back to the runtime's Prometheus gauges.
    ///
    /// `/v1/loads` is the canonical load schema: it reports the same
    /// scheduler state the `/metrics` arms reconstruct from gauges, in two
    /// orders of magnitude fewer bytes, and carries fields — queued tokens,
    /// generation throughput, the disagg section — that no gauge exposes.
    /// Preferring it is both cheaper and more informative wherever it exists.
    ///
    /// Every backend is normalized into a single-rank [`WorkerLoadResponse`]
    /// whose `token_usage` field drives the load-aware policies. Returns
    /// `None` on failure so the caller records the load as unavailable (`-1`).
    ///
    /// `native_loads_absent` is the shared probe memo, or `None` for callers
    /// with no monitor to borrow it from (they simply always probe).
    pub(crate) async fn fetch_http_load(
        client: &reqwest::Client,
        worker: &Arc<dyn Worker>,
        native_loads_absent: Option<&DashSet<String>>,
    ) -> Option<WorkerLoadResponse> {
        // Only workers already `Ready` are polled, so a definitive "no such
        // endpoint" cannot be a warm-up artifact and is safe to memoize.
        let known_absent = native_loads_absent.is_some_and(|memo| memo.contains(worker.url()));
        if !known_absent {
            match Self::probe_native_loads(client, worker).await {
                NativeLoads::Available(response) => return Some(response),
                NativeLoads::Absent => {
                    if let Some(memo) = native_loads_absent {
                        memo.insert(worker.url().to_string());
                    }
                }
                NativeLoads::Inconclusive => {}
            }
        }

        match worker.metadata().spec.runtime_type {
            RuntimeType::Vllm => Self::fetch_http_load_vllm(client, worker).await,
            RuntimeType::Sglang => Self::fetch_http_load_sglang(client, worker).await,
            // Custom engines expose no gauge schema we can parse; `/v1/loads`
            // above was their only path.
            _ => None,
        }
    }

    /// Probe `GET /v1/loads?include=core,disagg,queues,memory`.
    ///
    /// Extra sections beyond `core` degrade gracefully: engines that do not
    /// report them simply omit the fields, which deserialize to `None`.
    ///
    /// Distinguishing [`NativeLoads::Absent`] from [`NativeLoads::Inconclusive`]
    /// is what makes the memo safe. Collapsing both into "unsupported" would
    /// let one timeout demote a healthy backend to the expensive `/metrics`
    /// path permanently.
    async fn probe_native_loads(client: &reqwest::Client, worker: &Arc<dyn Worker>) -> NativeLoads {
        let url = format!(
            "{}/v1/loads?include=core,disagg,queues,memory",
            worker.url()
        );
        let resp = match Self::authed_request(client, worker, &url).send().await {
            Ok(resp) => resp,
            // Transport error or timeout: says nothing about the route.
            Err(_) => return NativeLoads::Inconclusive,
        };

        let status = resp.status();
        if matches!(
            status,
            StatusCode::NOT_FOUND | StatusCode::METHOD_NOT_ALLOWED | StatusCode::NOT_IMPLEMENTED
        ) {
            return NativeLoads::Absent;
        }
        if !status.is_success() {
            // 5xx and auth failures are conditions, not routing facts.
            return NativeLoads::Inconclusive;
        }

        match resp.json::<WorkerLoadResponse>().await {
            Ok(response) if !response.loads.is_empty() => NativeLoads::Available(response),
            // Our schema, no ranks reported yet — keep probing.
            Ok(_) => NativeLoads::Inconclusive,
            // A 200 that is not a load response means something else is
            // mounted here; that is as definitive as a 404.
            Err(_) => NativeLoads::Absent,
        }
    }

    /// vLLM HTTP: derive load from the Prometheus `/metrics` endpoint.
    /// The KV-cache usage ratio (0.0–1.0) maps onto `token_usage`; it is
    /// exposed as `vllm:gpu_cache_usage_perc` in vLLM v0 and renamed to
    /// `vllm:kv_cache_usage_perc` in vLLM v1, so accept either.
    async fn fetch_http_load_vllm(
        client: &reqwest::Client,
        worker: &Arc<dyn Worker>,
    ) -> Option<WorkerLoadResponse> {
        let url = format!("{}/metrics", worker.url());
        let body = Self::authed_get(client, worker, &url)
            .await?
            .text()
            .await
            .ok()?;
        let m = PromScrape::parse(&body);

        // Require the KV-usage gauge: it is the signal load-aware routing acts
        // on, and without it `token_usage` would default to `0.0` and make a
        // worker of unknown pressure look idle. v0 and v1 name it differently.
        let kv_usage = ["vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc"]
            .into_iter()
            .find(|name| m.has(name))?;

        Some(Self::single_rank(SchedulerLoadSnapshot {
            num_running_reqs: m.sum("vllm:num_requests_running") as i32,
            num_waiting_reqs: m.sum("vllm:num_requests_waiting") as i32,
            token_usage: m.mean(kv_usage),
            cache_hit_rate: m.mean("vllm:gpu_prefix_cache_hit_rate"),
            ..Default::default()
        }))
    }

    /// SGLang HTTP: try the custom `/v1/loads` endpoint first (some builds
    /// serve it), then fall back to the Prometheus `/metrics` gauges. The
    /// KV-usage ratio (0.0–1.0) is `<prefix>token_usage`, where SGLang used
    /// the `sglang:` metric prefix through v0.5.3 and switched to `sglang_`
    /// in v0.5.4+, so detect whichever is present and use it throughout.
    async fn fetch_http_load_sglang(
        client: &reqwest::Client,
        worker: &Arc<dyn Worker>,
    ) -> Option<WorkerLoadResponse> {
        let url = format!("{}/metrics", worker.url());
        let body = Self::authed_get(client, worker, &url)
            .await?
            .text()
            .await
            .ok()?;
        let m = PromScrape::parse(&body);

        // Require the KV-usage gauge — the load signal routing acts on.
        // Without it `token_usage` would default to 0.0 and hide real KV
        // pressure, so report unavailable instead. Pick the prefix off it.
        let prefix = ["sglang:", "sglang_"]
            .into_iter()
            .find(|p| m.has(&format!("{p}token_usage")))?;

        Some(Self::single_rank(SchedulerLoadSnapshot {
            num_running_reqs: m.sum(&format!("{prefix}num_running_reqs")) as i32,
            num_waiting_reqs: m.sum(&format!("{prefix}num_queue_reqs")) as i32,
            token_usage: m.mean(&format!("{prefix}token_usage")),
            gen_throughput: m.mean(&format!("{prefix}gen_throughput")),
            cache_hit_rate: m.mean(&format!("{prefix}cache_hit_rate")),
            utilization: m.mean(&format!("{prefix}utilization")),
            ..Default::default()
        }))
    }

    /// Shared authenticated GET builder with the standard timeout.
    fn authed_request(
        client: &reqwest::Client,
        worker: &Arc<dyn Worker>,
        url: &str,
    ) -> reqwest::RequestBuilder {
        let req = client.get(url).timeout(REQUEST_TIMEOUT);
        match worker.api_key() {
            Some(key) => req.bearer_auth(key),
            None => req,
        }
    }

    /// Shared authenticated GET with the standard timeout. Returns `None` on
    /// transport error or non-success status.
    async fn authed_get(
        client: &reqwest::Client,
        worker: &Arc<dyn Worker>,
        url: &str,
    ) -> Option<reqwest::Response> {
        match Self::authed_request(client, worker, url).send().await {
            Ok(r) if r.status().is_success() => Some(r),
            _ => None,
        }
    }

    /// Wrap a single scheduler snapshot as a one-rank `WorkerLoadResponse`.
    /// HTTP `/metrics` already aggregates across DP ranks, so a single
    /// synthetic rank is the correct shape.
    ///
    /// Such a snapshot carries the KV-usage ratio (`token_usage`) but no
    /// absolute token counts (`max_total_num_tokens`/`num_used_tokens` stay
    /// `0`), so `WorkerLoadResponse::has_absolute_token_data` reports `false`
    /// for it — keeping it out of the DP-rank cache and the `/get_loads`
    /// absolute-token scalar.
    fn single_rank(snapshot: SchedulerLoadSnapshot) -> WorkerLoadResponse {
        WorkerLoadResponse {
            dp_rank_count: 1,
            loads: vec![snapshot],
            ..Default::default()
        }
    }

    /// Fetch load via the backend client's `GetLoads` (gRPC RPC or the ZMQ
    /// engine's pushed scheduler stats). Returns `None` on missing client,
    /// error, or empty `loads` array.
    pub(crate) async fn fetch_backend_load(worker: &Arc<dyn Worker>) -> Option<WorkerLoadResponse> {
        let backend_client = match worker.get_backend_client().await {
            Ok(Some(client)) => client,
            Ok(None) => {
                debug!("No backend client for worker {}", worker.url());
                return None;
            }
            Err(e) => {
                debug!("Failed to get backend client for {}: {e}", worker.url());
                return None;
            }
        };

        match backend_client.get_loads().await {
            Ok(load) if !load.loads.is_empty() => Some(load),
            Ok(_) => None,
            Err(e) => {
                debug!("backend GetLoads failed for {}: {e}", worker.url());
                None
            }
        }
    }
}

impl Drop for WorkerMonitor {
    fn drop(&mut self) {
        if let Some(handle) = self.event_task.get_mut().take() {
            handle.abort();
        }
        for (_, state) in self.group_handles.get_mut().drain() {
            state.handle.abort();
        }
    }
}

/// Compute the set of `WorkerGroupKey`s a worker participates in.
///
/// A worker can serve multiple models (multimodel deployments), so it
/// can belong to multiple groups simultaneously. Each `(model, type,
/// connection)` triple is one group.
fn group_keys_for_worker(worker: &Arc<dyn Worker>) -> Vec<WorkerGroupKey> {
    WorkerRegistry::worker_model_ids(worker)
        .into_iter()
        .map(|model_id| WorkerGroupKey {
            model_id,
            worker_type: *worker.worker_type(),
            connection_mode: *worker.connection_mode(),
        })
        .collect()
}

/// Compute the polling interval for a group from per-worker overrides.
///
/// Uses the smallest `load_monitor_interval_secs` across the group so
/// the fastest worker's polling cadence wins. Falls back to
/// `default_interval` when no worker sets an override. Always floored
/// to one second to prevent tight-loop DoS.
fn group_interval(workers: &[Arc<dyn Worker>], default_interval: Duration) -> Duration {
    let override_secs = workers
        .iter()
        .filter_map(|w| w.metadata().spec.load_monitor_interval_secs)
        .min();

    let interval = override_secs
        .map(|s| Duration::from_secs(s.max(1)))
        .unwrap_or(default_interval);
    interval.max(Duration::from_secs(1))
}

/// Background event handler. Lives as long as the `WorkerMonitor`.
///
/// Holds a `Weak<WorkerMonitor>` rather than an `Arc<WorkerMonitor>`
/// so the spawned task does not pin the monitor in memory after every
/// strong reference has been dropped. Each iteration upgrades the
/// `Weak` only for the duration of the work and drops the temporary
/// `Arc` before the next `recv().await`, so the monitor's `Drop` can
/// fire as soon as the owning `AppContext` goes away.
async fn run_event_loop(
    monitor: Weak<WorkerMonitor>,
    mut events_rx: broadcast::Receiver<WorkerEvent>,
) {
    loop {
        let event = events_rx.recv().await;
        let Some(monitor) = monitor.upgrade() else {
            debug!("WorkerMonitor was dropped; exiting event loop");
            return;
        };
        match event {
            Ok(WorkerEvent::Registered { worker, .. }) => {
                for key in group_keys_for_worker(&worker) {
                    monitor.reconcile_group(&key);
                }
            }
            Ok(WorkerEvent::Removed { worker, .. }) => {
                // Evict the worker from both caches BEFORE reconciling
                // the group. If this was the last worker in its group,
                // `reconcile_group` will stop the polling loop, and a
                // stopped loop cannot prune the stale entry on a later
                // tick — it would persist forever otherwise.
                monitor.evict_worker_loads(&worker);
                for key in group_keys_for_worker(&worker) {
                    monitor.reconcile_group(&key);
                }
            }
            Ok(WorkerEvent::Replaced { old, new, .. }) => {
                // Registry guarantees `old.url() == new.url()` (replace
                // rejects URL changes), but the model list may have
                // shrunk so some old groups disappear. Evict the URL
                // from caches before reconciling so the disappearing
                // groups do not leak the entry; the surviving / new
                // groups will repopulate it on their next poll tick.
                monitor.evict_worker_loads(&old);
                let mut keys = group_keys_for_worker(&old);
                keys.extend(group_keys_for_worker(&new));
                keys.sort_by(|a, b| {
                    a.model_id
                        .cmp(&b.model_id)
                        .then_with(|| (a.worker_type as u8).cmp(&(b.worker_type as u8)))
                        .then_with(|| (a.connection_mode as u8).cmp(&(b.connection_mode as u8)))
                });
                keys.dedup();
                for key in keys {
                    monitor.reconcile_group(&key);
                }
            }
            Ok(WorkerEvent::StatusChanged {
                worker,
                new_status,
                old_status: _,
                ..
            }) => {
                if new_status != WorkerStatus::Ready {
                    monitor.evict_worker_loads(&worker);
                }
                // No action needed when transitioning *into* Ready: the
                // group's polling loop reads from the registry on every
                // tick and will pick the worker up automatically.
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                warn!(
                    skipped = n,
                    "WorkerMonitor lagged behind registry events; rebuilding from snapshot"
                );
                monitor.reconcile_from_registry();
            }
            Err(broadcast::error::RecvError::Closed) => {
                debug!("WorkerMonitor event channel closed; exiting event loop");
                return;
            }
        }
        // Drop the temporary strong reference so we do not keep the
        // monitor alive while parked on the next `recv().await`.
        drop(monitor);
    }
}

/// Polling loop body for a single worker group.
///
/// Holds a `Weak<WorkerMonitor>` for the same cycle-breaking reason
/// as `run_event_loop`. The temporary `Arc` is upgraded after the
/// timer tick and dropped before the next tick so the monitor's
/// `Drop` is reachable.
async fn group_monitor_loop(
    monitor: Weak<WorkerMonitor>,
    group_key: WorkerGroupKey,
    interval: Duration,
) {
    let mut interval_timer = tokio::time::interval(interval);

    loop {
        interval_timer.tick().await;

        let Some(monitor) = monitor.upgrade() else {
            debug!("WorkerMonitor was dropped; exiting group loop for {group_key}");
            return;
        };

        // Only poll Ready workers — Pending/NotReady/Failed do not
        // serve traffic and should not contribute load samples.
        let workers: Vec<Arc<dyn Worker>> = monitor
            .worker_registry
            .get_workers_filtered(
                Some(&group_key.model_id),
                Some(group_key.worker_type),
                Some(group_key.connection_mode),
                None,
                false,
            )
            .into_iter()
            .filter(|w| w.status() == WorkerStatus::Ready)
            .collect();

        if workers.is_empty() {
            debug!("No Ready workers in group {group_key}, skipping");
            drop(monitor);
            continue;
        }

        // Polling is unconditional by default so registration alone gives a
        // worker live load state. `--disable-load-monitoring` restores the
        // conditional gate: a load-aware policy, engine-metrics re-export, or
        // overload protection on any group member still forces the poll —
        // never "never poll".
        let load_aware_policies = monitor.policy_registry.get_all_load_aware_policies();
        if monitor.conditional_polling {
            let routing_needs_load = !load_aware_policies.is_empty()
                || monitor.policy_registry.get_dp_rank_policy().is_some();
            let overload_needs_load = workers.iter().any(|w| w.metadata().overload.is_enabled());
            if !routing_needs_load && !monitor.engine_metrics && !overload_needs_load {
                debug!("Load monitoring disabled and nothing needs the data, skipping load fetch for group {group_key}");
                drop(monitor);
                continue;
            }
        }

        let futures: Vec<_> = workers
            .iter()
            .map(|worker| {
                let client = monitor.client.clone();
                let native_loads_absent = Arc::clone(&monitor.native_loads_absent);
                let worker = Arc::clone(worker);
                let connection_mode = group_key.connection_mode;
                async move {
                    let response = match connection_mode {
                        ConnectionMode::Http => {
                            WorkerMonitor::fetch_http_load(
                                &client,
                                &worker,
                                Some(&native_loads_absent),
                            )
                            .await
                        }
                        ConnectionMode::Grpc | ConnectionMode::Zmq => {
                            WorkerMonitor::fetch_backend_load(&worker).await
                        }
                    };
                    (worker, response)
                }
            })
            .collect();

        let results = future::join_all(futures).await;

        let mut group_loads: HashMap<String, WorkerLoadResponse> = HashMap::new();
        let mut group_dp_loads: HashMap<String, HashMap<isize, isize>> = HashMap::new();
        let mut dp_evict: Vec<String> = Vec::new();
        for (worker, response) in results {
            let url = worker.url().to_string();
            // The overload predicate runs exactly here, once per report, never
            // on a request path, against the worker's effective thresholds
            // (resolved at registration). A failed fetch means no fresh
            // signal, which clears the flag — absent means no opinion.
            let overload = worker.metadata().overload;
            if overload.is_enabled() {
                let verdict = response
                    .as_ref()
                    .is_some_and(|load| overload.is_overloaded(load));
                monitor
                    .worker_registry
                    .set_worker_overloaded(&worker, verdict);
            }
            if let Some(load) = response {
                // Only feed the DP-rank cache from responses that carry real
                // absolute per-rank token counts. Ratio-only snapshots,
                // which would otherwise poison with a fake `{0: 0}`
                // entry and collapse DP routing onto rank 0.
                if load.has_absolute_token_data() {
                    group_dp_loads.insert(url.clone(), load.dp_rank_loads());
                } else {
                    dp_evict.push(url.clone());
                }
                group_loads.insert(url, load);
            }
        }

        // Compute the URL set up front so both the success and
        // empty-fetch branches can prune stale entries from the watch
        // snapshot. Without the empty-fetch prune, a group that
        // starts timing out keeps publishing its previous tick's
        // loads forever — subscribers see a stale snapshot indefinitely.
        let all_group_urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();

        if group_loads.is_empty() {
            debug!("No loads fetched for group {group_key}, pruning stale entries");
            monitor.load_tx.send_modify(|map| {
                for url in &all_group_urls {
                    map.remove(url);
                }
            });
            // The DP cache deliberately keeps last-known-good entries
            // so routing decisions still have a hint to fall back to
            // when the upstream is briefly unreachable.
            drop(monitor);
            continue;
        }

        debug!(
            "Fetched loads from {}/{} workers in group {group_key}",
            group_loads.len(),
            workers.len()
        );

        for policy in &load_aware_policies {
            policy.update_loads(&group_loads);
        }
        monitor.worker_load_manager.update_dp_loads(&group_dp_loads);

        if !dp_evict.is_empty() {
            monitor.worker_load_manager.remove_workers(&dp_evict);
        }

        // Re-export the freshly fetched loads as `smg_engine_*` gauges. Reuses
        // this poll's data; no extra fetch. Model label comes from the group.
        if monitor.engine_metrics {
            for (url, load) in &group_loads {
                Metrics::record_engine_load(url, &group_key.model_id, load);
            }
        }

        // Atomically merge into the shared watch channel: clear stale
        // entries for *this group's* URLs first, then insert the fresh
        // loads. Workers that failed this tick get their stale entries
        // pruned along with the rest.
        monitor.load_tx.send_modify(|map| {
            for url in &all_group_urls {
                map.remove(url);
            }
            map.extend(group_loads);
        });

        // Drop the temporary strong reference so we do not keep the
        // monitor alive across the next `interval_timer.tick().await`.
        drop(monitor);
    }
}

#[cfg(test)]
mod worker_load_manager_tests {
    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    #[test]
    fn test_new_dp_load_manager_instance() {
        let dp_load_manager = WorkerLoadManager::new();
        let cached = dp_load_manager.dp_cached_loads.read();
        assert!(cached.is_empty());
    }

    #[test]
    fn test_update_dp_load() {
        let manager = WorkerLoadManager::new();
        let mut loads = HashMap::new();

        let mut worker1_load = HashMap::new();
        worker1_load.insert(0, 2);
        worker1_load.insert(1, 1);
        loads.insert("http://worker1:8080".to_string(), worker1_load);

        let mut worker2_load = HashMap::new();
        worker2_load.insert(0, 3);
        loads.insert("http://worker2:8080".to_string(), worker2_load);

        manager.update_dp_loads(&loads);

        let cached = manager.dp_cached_loads.read();
        assert_eq!(cached.len(), 2);

        let worker2_cache = cached.get("http://worker2:8080").unwrap();
        assert_eq!(worker2_cache.get(&0), Some(&3));
    }

    #[test]
    fn test_select_and_increment_lowest_dp_load_multiple() {
        let worker = BasicWorkerBuilder::new("http://worker:8080")
            .worker_type(WorkerType::Regular)
            .api_key("test_key")
            .build();

        let manager = WorkerLoadManager::new();
        let mut loads = HashMap::new();
        let mut worker_load = HashMap::new();
        worker_load.insert(0, 10);
        worker_load.insert(1, 3);
        worker_load.insert(2, 7);
        loads.insert(worker.url().to_string(), worker_load);
        manager.update_dp_loads(&loads);

        let selected = manager.select_and_increment_lowest_dp_load(&worker, 4);

        assert_eq!(selected, Some(1));
        let cached = manager.dp_cached_loads.read();
        assert_eq!(*cached.get(worker.url()).unwrap().get(&1).unwrap(), 3 + 4);
    }

    #[test]
    fn test_select_and_increment_lowest_dp_load_none_worker() {
        let worker = BasicWorkerBuilder::new("http://nonexist:8080")
            .worker_type(WorkerType::Regular)
            .api_key("test")
            .build();

        let manager = WorkerLoadManager::new();
        let result = manager.select_and_increment_lowest_dp_load(&worker, 1);
        assert_eq!(result, None);
    }
}

#[cfg(test)]
mod worker_monitor_tests {
    use std::collections::HashMap;

    use openai_protocol::{
        model_card::ModelCard,
        worker::{HealthCheckConfig, WorkerStatus},
    };

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        policies::PolicyRegistry,
        worker::{BasicWorkerBuilder, ConnectionMode, WorkerType},
    };

    fn ready_worker(url: &str, model: &str) -> Arc<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Http)
            .model(ModelCard::new(model))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        worker.set_status(WorkerStatus::Ready);
        Arc::new(worker)
    }

    fn build_monitor() -> (Arc<WorkerRegistry>, Arc<WorkerMonitor>) {
        let registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        let monitor = Arc::new(WorkerMonitor::new(
            registry.clone(),
            policy_registry,
            reqwest::Client::new(),
            5,
            false,
            false,
        ));
        (registry, monitor)
    }

    #[tokio::test]
    async fn bootstrap_reconcile_starts_loops_for_existing_workers() {
        let (registry, monitor) = build_monitor();
        registry
            .register(ready_worker("http://w1:8080", "llama-3"))
            .unwrap();
        registry
            .register(ready_worker("http://w2:8080", "llama-3"))
            .unwrap();
        registry
            .register(ready_worker("http://w3:8080", "gpt-4"))
            .unwrap();

        monitor.start_event_loop();

        // Two model groups should now have polling loops.
        let handles = monitor.group_handles.lock();
        assert_eq!(handles.len(), 2);
        let keys: Vec<&WorkerGroupKey> = handles.keys().collect();
        assert!(keys.iter().any(|k| k.model_id == "llama-3"));
        assert!(keys.iter().any(|k| k.model_id == "gpt-4"));
    }

    #[tokio::test]
    async fn registered_event_starts_a_new_group() {
        let (registry, monitor) = build_monitor();
        monitor.start_event_loop();

        // Registry was empty at bootstrap, so no groups yet.
        assert!(monitor.group_handles.lock().is_empty());

        registry
            .register(ready_worker("http://w:8080", "llama-3"))
            .unwrap();

        // Give the event loop a moment to process the broadcast.
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(50)).await;

        let handles = monitor.group_handles.lock();
        assert_eq!(handles.len(), 1);
        assert!(handles
            .keys()
            .any(|k| k.model_id == "llama-3" && k.worker_type == WorkerType::Regular));
    }

    #[tokio::test]
    async fn removed_event_stops_empty_group() {
        let (registry, monitor) = build_monitor();
        let id = registry
            .register(ready_worker("http://w:8080", "llama-3"))
            .unwrap();
        monitor.start_event_loop();
        assert_eq!(monitor.group_handles.lock().len(), 1);

        registry.remove(&id);
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert!(monitor.group_handles.lock().is_empty());
    }

    #[tokio::test]
    async fn removed_event_evicts_cached_loads() {
        // Regression test for the bug where a removed worker's last
        // known load would persist forever in `load_tx` and the DP
        // cache because the polling loop that pruned it had been
        // stopped by the same removal.
        let (registry, monitor) = build_monitor();
        let worker = ready_worker("http://w:8080", "llama-3");
        let url = worker.url().to_string();
        let id = registry.register(worker).unwrap();
        monitor.start_event_loop();

        monitor.load_tx.send_modify(|map| {
            map.insert(url.clone(), WorkerLoadResponse::default());
        });
        monitor.native_loads_absent.insert(url.clone());
        let mut dp_loads: HashMap<String, HashMap<isize, isize>> = HashMap::new();
        let mut inner = HashMap::new();
        inner.insert(0, 5);
        dp_loads.insert(url.clone(), inner);
        monitor.worker_load_manager.update_dp_loads(&dp_loads);

        registry.remove(&id);
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(50)).await;

        let snapshot = monitor.load_rx.borrow().clone();
        assert!(
            !snapshot.contains_key(&url),
            "load_tx must not retain entries for removed workers"
        );
        let cached = monitor.worker_load_manager.dp_cached_loads.read();
        assert!(
            !cached.contains_key(&url),
            "DP cache must not retain entries for removed workers"
        );
        assert!(
            !monitor.native_loads_absent.contains(&url),
            "probe memo must not retain entries for removed workers"
        );
    }

    #[tokio::test]
    async fn stop_all_groups_clears_native_probe_memo() {
        // A worker removed during a `RecvError::Lagged` window never
        // fires an eviction, so the reset path must drop the memo too.
        let (_registry, monitor) = build_monitor();
        monitor
            .native_loads_absent
            .insert("http://w1:8080".to_string());

        monitor.stop_all_groups();

        assert!(monitor.native_loads_absent.is_empty());
    }

    #[tokio::test]
    async fn stop_all_groups_clears_dp_cache() {
        // Regression test for the bug where a `RecvError::Lagged`
        // rebuild would leave stale DP cache entries because
        // stop_all_groups only cleared the watch snapshot.
        let (registry, monitor) = build_monitor();
        registry
            .register(ready_worker("http://w1:8080", "llama-3"))
            .unwrap();

        let mut dp_loads: HashMap<String, HashMap<isize, isize>> = HashMap::new();
        let mut inner = HashMap::new();
        inner.insert(0, 7);
        dp_loads.insert("http://w1:8080".to_string(), inner);
        monitor.worker_load_manager.update_dp_loads(&dp_loads);

        monitor.stop_all_groups();

        assert!(monitor.load_rx.borrow().is_empty());
        assert!(monitor
            .worker_load_manager
            .dp_cached_loads
            .read()
            .is_empty());
    }

    #[tokio::test]
    async fn status_changed_away_from_ready_evicts_worker() {
        let (registry, monitor) = build_monitor();
        let worker = ready_worker("http://w:8080", "llama-3");
        let url = worker.url().to_string();
        let id = registry.register(worker).unwrap();
        monitor.start_event_loop();

        // Seed the watch channel + DP cache as if a poll had succeeded.
        monitor.load_tx.send_modify(|map| {
            map.insert(url.clone(), WorkerLoadResponse::default());
        });
        let mut dp_loads: HashMap<String, HashMap<isize, isize>> = HashMap::new();
        let mut inner = HashMap::new();
        inner.insert(0, 5);
        dp_loads.insert(url.clone(), inner);
        monitor.worker_load_manager.update_dp_loads(&dp_loads);

        registry.transition_status(&id, WorkerStatus::NotReady);
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Watch channel entry pruned.
        let snapshot = monitor.load_rx.borrow().clone();
        assert!(!snapshot.contains_key(&url));

        // DP cache entry pruned.
        let cached = monitor.worker_load_manager.dp_cached_loads.read();
        assert!(!cached.contains_key(&url));
    }

    #[tokio::test]
    async fn spawned_tasks_use_weak_references() {
        // Regression test for the Arc cycle bug: the spawned event
        // loop and the per-group polling loops must hold
        // `Weak<WorkerMonitor>` rather than `Arc<WorkerMonitor>`. If
        // they held strong references, the cycle through
        // `event_task` / `group_handles` would keep the strong count
        // pinned and `Drop` would never fire outside of full runtime
        // shutdown.
        let (registry, monitor) = build_monitor();
        registry
            .register(ready_worker("http://w:8080", "llama-3"))
            .unwrap();
        monitor.start_event_loop();

        // Bootstrap should have started a polling loop for the
        // worker's group, on top of the event task itself.
        assert_eq!(monitor.group_handles.lock().len(), 1);

        // After spawning, only the test's `Arc` should be strong:
        // both the event task and the group loop downgraded to
        // `Weak<Self>`.
        assert_eq!(
            Arc::strong_count(&monitor),
            1,
            "spawned tasks must not hold strong references to the monitor"
        );
        assert!(
            Arc::weak_count(&monitor) >= 2,
            "expected at least one Weak from the event task and one from the group loop"
        );
    }
}

#[cfg(test)]
mod prom_scrape_tests {
    use super::*;

    // Trimmed sample mirroring the fields observed on a live vLLM v0.7.3
    // `/metrics` endpoint (dev ORD cluster).
    const VLLM_METRICS: &str = r#"
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="llama"} 3.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="llama"} 5.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="llama"} 0.75
vllm:generation_tokens_total{model_name="llama"} 123456.0
"#;

    const SGLANG_METRICS: &str = r#"
sglang:num_running_reqs{model="llama"} 2.0
sglang:num_queue_reqs{model="llama"} 4.0
sglang:token_usage{model="llama"} 0.42
sglang:utilization{model="llama"} 0.9
"#;

    #[test]
    fn parses_gauges_ignoring_comments_and_labels() {
        let m = PromScrape::parse(VLLM_METRICS);
        assert!(m.has("vllm:num_requests_running"));
        assert_eq!(m.sum("vllm:num_requests_running"), 3.0);
        assert_eq!(m.sum("vllm:num_requests_waiting"), 5.0);
        assert_eq!(m.mean("vllm:gpu_cache_usage_perc"), 0.75);
        assert!(!m.has("vllm:does_not_exist"));
        assert_eq!(m.sum("vllm:does_not_exist"), 0.0);
        assert_eq!(m.mean("vllm:does_not_exist"), 0.0);
    }

    #[test]
    fn sum_adds_ranks_mean_averages_ratios() {
        // Two DP-rank series for the same gauge name.
        let text = "g{dp=\"0\"} 10\ng{dp=\"1\"} 30\nr{dp=\"0\"} 0.2\nr{dp=\"1\"} 0.8\n";
        let m = PromScrape::parse(text);
        assert_eq!(m.sum("g"), 40.0); // counts add
        assert_eq!(m.mean("r"), 0.5); // ratios average
    }

    #[test]
    fn vllm_metrics_map_onto_token_usage_snapshot() {
        let m = PromScrape::parse(VLLM_METRICS);
        let snap = SchedulerLoadSnapshot {
            num_running_reqs: m.sum("vllm:num_requests_running") as i32,
            num_waiting_reqs: m.sum("vllm:num_requests_waiting") as i32,
            token_usage: m.mean("vllm:gpu_cache_usage_perc"),
            ..Default::default()
        };
        let resp = WorkerMonitor::single_rank(snap);
        assert_eq!(resp.dp_rank_count, 1);
        assert_eq!(resp.effective_token_usage(), 0.75);
        assert_eq!(resp.loads[0].num_running_reqs, 3);
        assert_eq!(resp.loads[0].num_waiting_reqs, 5);
    }

    #[test]
    fn sglang_metrics_map_onto_token_usage_snapshot() {
        let m = PromScrape::parse(SGLANG_METRICS);
        assert_eq!(m.mean("sglang:token_usage"), 0.42);
        assert_eq!(m.sum("sglang:num_running_reqs"), 2.0);
        assert_eq!(m.sum("sglang:num_queue_reqs"), 4.0);
    }

    #[test]
    fn sglang_v054_underscore_prefix_is_recognized() {
        // SGLang v0.5.4+ renamed the metric prefix `sglang:` -> `sglang_`.
        let v054 = "sglang_token_usage{model=\"llama\"} 0.5\n\
                    sglang_num_running_reqs{model=\"llama\"} 7\n\
                    sglang_num_queue_reqs{model=\"llama\"} 1\n";
        let m = PromScrape::parse(v054);
        assert!(!m.has("sglang:token_usage"));
        let prefix = ["sglang:", "sglang_"]
            .into_iter()
            .find(|p| m.has(&format!("{p}token_usage")));
        assert_eq!(prefix, Some("sglang_"));
        assert_eq!(m.mean("sglang_token_usage"), 0.5);
        assert_eq!(m.sum("sglang_num_running_reqs"), 7.0);
    }

    #[test]
    fn parse_ignores_trailing_timestamp_and_label_spaces() {
        // Optional trailing timestamp must not be read as the value, and a
        // space inside a quoted label value must not split the head early.
        let text = "vllm:gpu_cache_usage_perc{model_name=\"my model\"} 0.75 1719849600000\n";
        let m = PromScrape::parse(text);
        assert_eq!(m.mean("vllm:gpu_cache_usage_perc"), 0.75);
    }

    #[test]
    fn vllm_v1_kv_cache_usage_metric_is_recognized() {
        // vLLM v1 renamed the gauge from `gpu_cache_usage_perc` to
        // `kv_cache_usage_perc`; the fetcher accepts either name.
        let v1 = "vllm:kv_cache_usage_perc{model_name=\"llama\"} 0.6\n";
        let m = PromScrape::parse(v1);
        assert!(!m.has("vllm:gpu_cache_usage_perc"));
        let kv_usage = ["vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc"]
            .into_iter()
            .find(|name| m.has(name));
        assert_eq!(kv_usage, Some("vllm:kv_cache_usage_perc"));
        assert_eq!(m.mean("vllm:kv_cache_usage_perc"), 0.6);
    }

    #[test]
    fn metric_derived_snapshot_has_no_absolute_token_data() {
        // A ratio-only snapshot (from `/metrics`) must not be treated as
        // carrying absolute token counts: it stays out of the DP cache and
        // the `/get_loads` scalar, even at high KV usage.
        let resp = WorkerMonitor::single_rank(SchedulerLoadSnapshot {
            token_usage: 0.75,
            num_running_reqs: 3,
            ..Default::default()
        });
        assert!(!resp.has_absolute_token_data());
        assert_eq!(resp.total_used_tokens(), 0);
        assert_eq!(resp.effective_token_usage(), 0.75);
    }

    #[test]
    fn real_snapshot_with_capacity_has_absolute_token_data() {
        // A snapshot from gRPC/`/v1/loads` reports KV capacity, so its
        // absolute token fields are meaningful (even when idle).
        let resp = WorkerMonitor::single_rank(SchedulerLoadSnapshot {
            max_total_num_tokens: 8192,
            num_used_tokens: 1024,
            ..Default::default()
        });
        assert!(resp.has_absolute_token_data());
        assert_eq!(resp.total_used_tokens(), 1024);
    }
}

#[cfg(test)]
mod native_loads_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use openai_protocol::{
        model_card::ModelCard,
        worker::{HealthCheckConfig, OverloadUpdate},
    };

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        worker::{BasicWorkerBuilder, ConnectionMode, WorkerType},
    };

    const VLLM_METRICS: &str = "vllm:num_requests_running{m=\"a\"} 7.0\n\
         vllm:num_requests_waiting{m=\"a\"} 11.0\n\
         vllm:kv_cache_usage_perc{m=\"a\"} 0.5\n";

    /// Carries `num_waiting_uncached_tokens`, which the `/metrics` arm has no
    /// gauge for — so its presence identifies which path answered.
    const NATIVE_BODY: &str = r#"{"loads":[{"dp_rank":0,"num_running_reqs":3,
        "num_waiting_reqs":4,"num_waiting_uncached_tokens":900,"token_usage":0.25}]}"#;

    struct Stub {
        url: String,
        probes: Arc<AtomicUsize>,
    }

    /// Loopback engine stub. `/v1/loads` answers with the given status and
    /// body and counts its hits; `/metrics` always serves the vLLM gauges.
    async fn spawn_engine(status: StatusCode, body: &'static str) -> Stub {
        let probes = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&probes);
        let app = axum::Router::new()
            .route(
                "/v1/loads",
                axum::routing::get(move || {
                    let counter = Arc::clone(&counter);
                    async move {
                        counter.fetch_add(1, Ordering::SeqCst);
                        (status, body)
                    }
                }),
            )
            .route("/metrics", axum::routing::get(|| async { VLLM_METRICS }));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind stub");
        let addr = listener.local_addr().expect("stub address");
        #[expect(
            clippy::disallowed_methods,
            reason = "test stub server lives for the duration of the test process"
        )]
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("stub serve");
        });

        Stub {
            url: format!("http://{addr}"),
            probes,
        }
    }

    fn vllm_worker(url: &str) -> Arc<dyn Worker> {
        vllm_worker_with_overload(url, OverloadUpdate::default())
    }

    /// Worker carrying a per-worker overload block — protection for it is
    /// enabled by the spec alone, with no gateway thresholds anywhere.
    fn vllm_worker_with_overload(url: &str, overload: OverloadUpdate) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Http)
                .runtime_type(RuntimeType::Vllm)
                .model(ModelCard::new("a"))
                .overload(overload)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    fn cache_aware_policy_config(overlap_decay: f32) -> PolicyConfig {
        PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 0,
            max_tree_size: 4096,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            overlap_decay,
            selection_temperature: 0.0,
            cache_index: Default::default(),
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
        }
    }

    fn monitor_with_policy(config: PolicyConfig) -> (Arc<WorkerRegistry>, Arc<WorkerMonitor>) {
        monitor_with(config, false)
    }

    fn monitor_with(
        config: PolicyConfig,
        conditional_polling: bool,
    ) -> (Arc<WorkerRegistry>, Arc<WorkerMonitor>) {
        let registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(config));
        let monitor = Arc::new(WorkerMonitor::new(
            registry.clone(),
            policy_registry,
            reqwest::Client::new(),
            1,
            false,
            conditional_polling,
        ));
        (registry, monitor)
    }

    /// Poll `condition` until it holds or the timeout expires. The group loop
    /// writes the flag out of band, so there is nothing to await directly.
    async fn wait_until(label: &str, mut condition: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(10), async {
            while !condition() {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {label}"));
    }

    /// `NATIVE_BODY` reports 4 waiting requests, so a threshold of 4 vetoes and
    /// a threshold of 5 does not — the ingestion path must latch the verdict on
    /// the worker itself, under a policy (round-robin) that reads no loads at
    /// all. The threshold rides the worker's spec block, so this also proves a
    /// spec block alone enables protection with no gateway flags anywhere.
    #[tokio::test]
    async fn load_ingestion_flags_worker_over_waiting_threshold() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let (registry, monitor) = monitor_with(PolicyConfig::RoundRobin, false);
        let worker = vllm_worker_with_overload(
            &stub.url,
            OverloadUpdate {
                waiting_requests: Some(4),
                token_usage: None,
            },
        );
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        monitor.start_event_loop();

        wait_until("worker to be flagged overloaded", || worker.is_overloaded()).await;
        assert_eq!(registry.overloaded_worker_count("a"), 1);
        assert!(
            !worker.is_available(),
            "an overloaded worker is not routable"
        );
    }

    /// A worker with real queueing that stays under the threshold must keep
    /// serving: the veto is absolute, not comparative.
    #[tokio::test]
    async fn busy_worker_under_threshold_stays_eligible() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let (registry, monitor) = monitor_with(PolicyConfig::RoundRobin, false);
        let worker = vllm_worker_with_overload(
            &stub.url,
            OverloadUpdate {
                waiting_requests: Some(5),
                token_usage: None,
            },
        );
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        monitor.start_event_loop();

        // Wait for the feed to actually land, then assert the verdict.
        let mut rx = monitor.subscribe();
        tokio::time::timeout(Duration::from_secs(10), async {
            while !rx.borrow().contains_key(&stub.url) {
                rx.changed().await.expect("watch sender alive");
            }
        })
        .await
        .expect("load feed must reach the watch channel");

        assert!(!worker.is_overloaded());
        assert!(worker.is_available());
        assert_eq!(registry.overloaded_worker_count("a"), 0);
    }

    /// Overload protection alone must open the poll gate even under
    /// `--disable-load-monitoring`: without it the feature would silently
    /// never engage under a load-blind policy.
    #[tokio::test]
    async fn overload_thresholds_alone_start_load_polling() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let (registry, monitor) = monitor_with(PolicyConfig::RoundRobin, true);
        let worker = vllm_worker_with_overload(
            &stub.url,
            OverloadUpdate {
                waiting_requests: None,
                token_usage: Some(0.2),
            },
        );
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        monitor.start_event_loop();

        wait_until("the engine to be polled", || {
            stub.probes.load(Ordering::SeqCst) > 0
        })
        .await;
        wait_until("token usage 0.25 to trip the 0.2 ceiling", || {
            worker.is_overloaded()
        })
        .await;
    }

    /// With protection off and no spec block the feature is off: an engine
    /// reporting a load that would trip either signal must still leave the
    /// flag, the counters and every routing verdict exactly where they were.
    #[tokio::test]
    async fn unconfigured_thresholds_never_write_the_flag() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        // cache_aware with overlap_decay on, so loads are polled and ingested
        // for reasons unrelated to overload protection.
        let (registry, monitor) = monitor_with(cache_aware_policy_config(1.0), false);
        let worker = vllm_worker(&stub.url);
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        monitor.start_event_loop();

        let mut rx = monitor.subscribe();
        tokio::time::timeout(Duration::from_secs(10), async {
            while !rx.borrow().contains_key(&stub.url) {
                rx.changed().await.expect("watch sender alive");
            }
        })
        .await
        .expect("load feed must reach the watch channel");
        // A second tick, so this is not just "the first poll had not landed".
        tokio::time::sleep(Duration::from_millis(1200)).await;

        assert!(!worker.is_overloaded());
        assert!(worker.is_available());
        assert_eq!(registry.overloaded_worker_count("a"), 0);
    }

    /// A worker whose feed goes away has no verdict at all, so it must be
    /// re-admitted rather than stranded out of routing forever.
    #[tokio::test]
    async fn losing_the_load_feed_clears_the_veto() {
        let (registry, monitor) = monitor_with(PolicyConfig::RoundRobin, false);
        // Never-registered address: every fetch fails, so the report is absent.
        let worker = vllm_worker_with_overload(
            "http://127.0.0.1:1",
            OverloadUpdate {
                waiting_requests: Some(1),
                token_usage: None,
            },
        );
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        registry.set_worker_overloaded(&worker, true);
        assert_eq!(registry.overloaded_worker_count("a"), 1);

        monitor.start_event_loop();
        wait_until("the absent feed to clear the veto", || {
            !worker.is_overloaded()
        })
        .await;
        assert_eq!(registry.overloaded_worker_count("a"), 0);
    }

    /// `stop_all_groups` tears down the only writer that could clear the flag,
    /// so it must clear it itself — same rule the load snapshot follows.
    #[tokio::test]
    async fn stop_all_groups_clears_overload_flags() {
        let (registry, monitor) = monitor_with(PolicyConfig::RoundRobin, false);
        let worker = vllm_worker_with_overload(
            "http://127.0.0.1:1",
            OverloadUpdate {
                waiting_requests: Some(1),
                token_usage: None,
            },
        );
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        registry.set_worker_overloaded(&worker, true);

        monitor.stop_all_groups();

        assert!(!worker.is_overloaded());
        assert_eq!(registry.overloaded_worker_count("a"), 0);
    }

    /// A load-aware policy must still be fed with the opt-out set — the
    /// conditional gate is "poll when needed", never "never poll".
    #[tokio::test]
    async fn pressure_configured_cache_aware_starts_load_polling() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let (registry, monitor) = monitor_with(cache_aware_policy_config(1.0), true);
        let worker = vllm_worker(&stub.url);
        worker.set_status(WorkerStatus::Ready);
        registry.register(worker).unwrap();
        monitor.start_event_loop();

        let mut rx = monitor.subscribe();
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let waiting = rx
                    .borrow()
                    .get(&stub.url)
                    .map(|load| load.total_waiting_uncached_tokens());
                if let Some(waiting) = waiting {
                    assert_eq!(waiting, 900);
                    break;
                }
                rx.changed().await.expect("watch sender alive");
            }
        })
        .await
        .expect("cache_aware with overlap_decay must trigger load polling");
    }

    /// Load monitoring is on by default: a load-blind policy with no flags and
    /// no spec blocks still gets its workers polled from registration onward.
    #[tokio::test]
    async fn load_blind_policy_polls_by_default() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let (registry, monitor) = monitor_with_policy(cache_aware_policy_config(0.0));
        let worker = vllm_worker(&stub.url);
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        monitor.start_event_loop();

        wait_until("default-on monitoring to poll the engine", || {
            stub.probes.load(Ordering::SeqCst) > 0
        })
        .await;
        // Polling alone must not write the overload flag.
        assert!(!worker.is_overloaded());
    }

    /// `--disable-load-monitoring` restores the old conditional gate: a
    /// load-blind policy with nothing needing the data is never polled.
    #[tokio::test]
    async fn disable_load_monitoring_restores_the_conditional_gate() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let (registry, monitor) = monitor_with(cache_aware_policy_config(0.0), true);
        let worker = vllm_worker(&stub.url);
        worker.set_status(WorkerStatus::Ready);
        registry.register(worker).unwrap();
        monitor.start_event_loop();

        // Covers the immediate first tick plus one full 1s interval.
        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert_eq!(stub.probes.load(Ordering::SeqCst), 0);
        assert!(monitor.load_rx.borrow().is_empty());
    }

    #[tokio::test]
    async fn native_loads_preferred_over_metrics_on_vllm() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let worker = vllm_worker(&stub.url);
        let memo = DashSet::new();

        let resp = WorkerMonitor::fetch_http_load(&reqwest::Client::new(), &worker, Some(&memo))
            .await
            .expect("load response");

        // `/metrics` cannot report queued tokens; only the native path can.
        assert_eq!(resp.loads[0].num_waiting_uncached_tokens, 900);
        assert_eq!(resp.loads[0].num_running_reqs, 3);
        assert!(memo.is_empty(), "a serving endpoint must not be memoized");
    }

    #[tokio::test]
    async fn absent_native_endpoint_falls_back_and_is_probed_once() {
        let stub = spawn_engine(StatusCode::NOT_FOUND, "").await;
        let worker = vllm_worker(&stub.url);
        let memo = DashSet::new();
        let client = reqwest::Client::new();

        for _ in 0..3 {
            let resp = WorkerMonitor::fetch_http_load(&client, &worker, Some(&memo))
                .await
                .expect("metrics fallback");
            assert_eq!(resp.loads[0].num_running_reqs, 7);
            assert_eq!(resp.loads[0].num_waiting_reqs, 11);
        }

        assert!(memo.contains(worker.url()));
        assert_eq!(
            stub.probes.load(Ordering::SeqCst),
            1,
            "a 404 is definitive; the probe must not repeat every tick"
        );
    }

    #[tokio::test]
    async fn transient_native_failure_is_not_memoized() {
        // 503 says the backend is unwell, not that the route is missing.
        // Memoizing it would demote a healthy engine to `/metrics` forever.
        let stub = spawn_engine(StatusCode::SERVICE_UNAVAILABLE, "").await;
        let worker = vllm_worker(&stub.url);
        let memo = DashSet::new();
        let client = reqwest::Client::new();

        for _ in 0..3 {
            WorkerMonitor::fetch_http_load(&client, &worker, Some(&memo))
                .await
                .expect("metrics fallback");
        }

        assert!(memo.is_empty());
        assert_eq!(stub.probes.load(Ordering::SeqCst), 3, "probe must retry");
    }

    #[tokio::test]
    async fn non_load_body_on_200_is_treated_as_absent() {
        let stub = spawn_engine(StatusCode::OK, "<html>nope</html>").await;
        let worker = vllm_worker(&stub.url);
        let memo = DashSet::new();

        WorkerMonitor::fetch_http_load(&reqwest::Client::new(), &worker, Some(&memo))
            .await
            .expect("metrics fallback");

        assert!(memo.contains(worker.url()));
    }

    #[tokio::test]
    async fn empty_loads_array_keeps_probing() {
        // Our schema, no ranks yet — the endpoint exists, so keep asking.
        let stub = spawn_engine(StatusCode::OK, r#"{"loads":[]}"#).await;
        let worker = vllm_worker(&stub.url);
        let memo = DashSet::new();
        let client = reqwest::Client::new();

        for _ in 0..2 {
            WorkerMonitor::fetch_http_load(&client, &worker, Some(&memo))
                .await
                .expect("metrics fallback");
        }

        assert!(memo.is_empty());
        assert_eq!(stub.probes.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn absent_memo_is_bypassed_when_caller_has_none() {
        let stub = spawn_engine(StatusCode::NOT_FOUND, "").await;
        let worker = vllm_worker(&stub.url);
        let client = reqwest::Client::new();

        for _ in 0..2 {
            WorkerMonitor::fetch_http_load(&client, &worker, None)
                .await
                .expect("metrics fallback");
        }

        assert_eq!(stub.probes.load(Ordering::SeqCst), 2);
    }
}
