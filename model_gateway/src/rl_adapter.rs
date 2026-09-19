//! Glue between the RL control plane crate and the gateway. This file is the
//! whole of coupling surfaces (a), (b), and the stamping half of (c); see
//! `crates/rl/COUPLING.md`.

use std::sync::{Arc, Weak};

use axum::{
    body::{to_bytes, Body},
    extract::{Request, State},
    middleware::Next,
    response::{IntoResponse, Response},
};
use http::{header::CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use http_body::Body as _;
use openai_protocol::worker::ConnectionMode;
use smg_rl::{
    stamp::generate_is_mixed, table::base_url_of, RlError, RlState, RlWorkerInfo, RlWorkerView,
    VersionEvictionSink, VersionPolicy,
};
use tokio::sync::broadcast::error::RecvError;
use tracing::warn;

use crate::{
    config::RouterConfig,
    policies::{CandidateFilter, PolicyRegistry},
    routers::{common::header_utils::RoutedWorker, error},
    worker::{event::WorkerEvent, registry::WorkerId, Worker, WorkerRegistry},
};

/// Read-only registry view for the RL crate.
pub struct RegistryRlView {
    registry: Arc<WorkerRegistry>,
}

impl RegistryRlView {
    pub fn new(registry: Arc<WorkerRegistry>) -> Self {
        Self { registry }
    }

    fn info(&self, worker: &Arc<dyn Worker>) -> Option<RlWorkerInfo> {
        let id = self.registry.get_id_by_url(worker.url())?;
        let spec = &worker.metadata().spec;
        // Borrow the client the gateway negotiated for this worker, as the
        // admin ops do; a worker without one is not spoken to over HTTP.
        let http_client = (*worker.connection_mode() == ConnectionMode::Http).then(|| {
            worker
                .http_client_handle_if_initialized()
                .unwrap_or_else(|| Arc::new(worker.http_client().clone()))
        });
        Some(RlWorkerInfo {
            id: id.as_str().to_string(),
            url: worker.url().to_string(),
            base_url: worker.base_url().to_string(),
            api_key: worker.api_key().cloned(),
            model_id: worker.model_id().to_string(),
            runtime: spec.runtime_type,
            worker_type: *worker.worker_type(),
            connection_mode: *worker.connection_mode(),
            status: worker.status(),
            is_dp_aware: worker.is_dp_aware(),
            dp_size: worker.dp_size(),
            labels: spec.labels.clone(),
            http_client,
        })
    }
}

impl RlWorkerView for RegistryRlView {
    fn list(&self) -> Vec<RlWorkerInfo> {
        self.registry
            .get_all()
            .iter()
            .filter_map(|w| self.info(w))
            .collect()
    }

    fn get(&self, id: &str) -> Option<RlWorkerInfo> {
        let worker = self.registry.get(&WorkerId::from_string(id.to_string()))?;
        self.info(&worker)
    }

    /// One engine is one table entry but can be many registry entries (a
    /// DP-aware group registers one worker per rank), so the engine's
    /// inflight count is the sum over every worker sharing its base URL.
    fn inflight(&self, base_url: &str) -> usize {
        self.registry
            .get_all()
            .iter()
            .filter(|w| w.base_url() == base_url)
            .map(|w| w.load())
            .sum()
    }
}

/// Coupling (a): a version change resets every rank's cache-aware entries.
///
/// Called with the RL table's write mutex held, so it must never write back
/// into the table — resetting policy caches is the whole of its work.
///
/// Both handles are weak. The policy registry owns the candidate filter,
/// which owns the [`RlState`] this sink lives in, so a strong handle here
/// would close an ownership cycle and leak both registries (and the
/// maintainer task with them) for the life of the process.
struct RegistryEvictionSink {
    registry: Weak<WorkerRegistry>,
    policy_registry: Weak<PolicyRegistry>,
}

impl VersionEvictionSink for RegistryEvictionSink {
    fn on_version_changed(&self, _model_id: &str, base_url: &str) {
        // Either one gone means the gateway is tearing down: there is no
        // cache left to reset.
        let (Some(registry), Some(policy_registry)) =
            (self.registry.upgrade(), self.policy_registry.upgrade())
        else {
            return;
        };
        for worker in registry
            .get_all()
            .iter()
            .filter(|w| w.base_url() == base_url)
        {
            policy_registry.reset_worker_cache(worker.as_ref());
        }
    }
}

/// Coupling (b): the RL pre-filter installed on the policy registry.
pub struct RlCandidateFilter {
    rl: Arc<RlState>,
}

impl CandidateFilter for RlCandidateFilter {
    fn eligible(
        &self,
        workers: &[Arc<dyn Worker>],
        headers: Option<&HeaderMap>,
    ) -> Option<Vec<usize>> {
        let policy = self.rl.policy_for(headers);
        self.rl.table().eligible(
            workers.iter().map(|w| (w.model_id(), w.base_url())),
            &policy,
        )
    }
}

/// The weight version a worker's discovered labels report, if any.
fn version_label(worker: &Arc<dyn Worker>) -> Option<&str> {
    worker
        .metadata()
        .spec
        .labels
        .get("weight_version")
        .map(String::as_str)
}

fn seed(rl: &RlState, worker: &Arc<dyn Worker>) {
    rl.table()
        .seed(worker.base_url(), worker.model_id(), version_label(worker));
}

/// Re-derive the table from the registry after a lagged event stream.
fn resync(rl: &RlState, registry: &WorkerRegistry) {
    let workers = registry.get_all();
    for worker in &workers {
        seed(rl, worker);
    }
    rl.table()
        .retain(|base_url| workers.iter().any(|w| w.base_url() == base_url));
}

/// Keep the table in step with registrations: seed on `Registered`, reseed
/// on `Replaced` only when the discovered label changed (a property patch
/// must not reset an observed version), drop on the last `Removed` for an
/// engine. Runs only when a Tokio runtime is current.
fn spawn_table_maintainer(rl: Weak<RlState>, registry: Arc<WorkerRegistry>) {
    if tokio::runtime::Handle::try_current().is_err() {
        warn!("RL table maintainer not started: no async runtime; versions seed lazily");
        return;
    }
    // Subscribe before the initial pass so a registration racing this call is
    // either already in `get_all()` or still queued on the receiver.
    let mut rx = registry.subscribe_events();
    // The caller still holds the state, so the initial pass runs against a
    // strong handle. It happens before the loop, so workers registered before
    // this call are in the table by the time it returns.
    if let Some(state) = rl.upgrade() {
        resync(&state, &registry);
    }
    // Every handle the task keeps is weak, so it never keeps the gateway
    // alive: it exits when the registry drops (`Closed`) or the state does.
    let registry = Arc::downgrade(&registry);
    #[expect(
        clippy::disallowed_methods,
        reason = "runs for the lifetime of the gateway and exits when the registry or the RL state is dropped"
    )]
    tokio::spawn(async move {
        loop {
            let event = rx.recv().await;
            let Some(state) = rl.upgrade() else {
                break;
            };
            match event {
                Ok(WorkerEvent::Registered { worker, .. }) => seed(&state, &worker),
                Ok(WorkerEvent::Replaced { old, new, .. }) => {
                    if version_label(&old) == version_label(&new) {
                        seed(&state, &new);
                    } else {
                        state
                            .table()
                            .reseed(new.base_url(), new.model_id(), version_label(&new));
                    }
                }
                Ok(WorkerEvent::Removed { worker, .. }) => {
                    let Some(registry) = registry.upgrade() else {
                        break;
                    };
                    let base_url = worker.base_url();
                    if !registry.get_all().iter().any(|w| w.base_url() == base_url) {
                        state.table().remove(base_url);
                    }
                }
                Ok(WorkerEvent::StatusChanged { .. }) => {}
                Err(RecvError::Lagged(n)) => {
                    warn!(missed = n, "RL table maintainer lagged; resyncing");
                    let Some(registry) = registry.upgrade() else {
                        break;
                    };
                    resync(&state, &registry);
                }
                Err(RecvError::Closed) => break,
            }
        }
    });
}

/// Build the RL state when `config.rl.enabled`, install the candidate
/// filter, and start the table maintainer; `None` otherwise, so the
/// disabled path constructs nothing.
pub fn build_rl_state(
    registry: &Arc<WorkerRegistry>,
    policy_registry: &Arc<PolicyRegistry>,
    config: &RouterConfig,
) -> Option<Arc<RlState>> {
    if !config.rl.enabled {
        return None;
    }
    let view = Arc::new(RegistryRlView::new(Arc::clone(registry)));
    let sink = Arc::new(RegistryEvictionSink {
        registry: Arc::downgrade(registry),
        policy_registry: Arc::downgrade(policy_registry),
    });
    let rl = Arc::new(RlState::with_sink(view, config.rl.clone(), sink));
    if !policy_registry.set_candidate_filter(Arc::new(RlCandidateFilter {
        rl: Arc::clone(&rl),
    })) {
        warn!("a candidate filter was already installed; RL routing filter not active");
    }
    spawn_table_maintainer(Arc::downgrade(&rl), Arc::clone(registry));
    Some(rl)
}

/// The version SMG believed the routed engine held when it answered.
static WEIGHT_VERSION: HeaderName = HeaderName::from_static(smg_rl::stamp::WEIGHT_VERSION_HEADER);
/// Set when a buffered `/generate` body spanned more than one version.
static MIXED_VERSION: HeaderName = HeaderName::from_static(smg_rl::stamp::MIXED_VERSION_HEADER);

fn is_json(headers: &HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("json"))
}

/// Coupling (c): validate the per-request version policy, stamp every served
/// response with the engine's version, and flag buffered `/generate` bodies
/// that spanned two versions. Mounted only when `--enable-rl` is on.
///
/// A streamed body is never read: `size_hint().exact()` is `None` for it, so
/// the mixed-version probe is skipped rather than buffering a generation.
pub async fn rl_middleware(
    State(rl): State<Arc<RlState>>,
    request: Request,
    next: Next,
) -> Response {
    if let Err(e) = VersionPolicy::from_headers(request.headers()) {
        return RlError::InvalidVersionPolicy(e.to_string()).into_response();
    }
    // The request is consumed by `next`, so decide on the path now.
    let inspect_body = request.uri().path() == "/generate";
    let mut response = next.run(request).await;
    // Set by the gateway alone, so it names the engine that actually served
    // this response — no upstream header can forge it.
    let Some(routed) = response.extensions().get::<RoutedWorker>().cloned() else {
        return response;
    };
    let version = rl.table().version_of(base_url_of(&routed.0));
    if let Some(value) = version
        .as_ref()
        .and_then(|v| HeaderValue::from_str(v.as_str()).ok())
    {
        response.headers_mut().insert(WEIGHT_VERSION.clone(), value);
    }
    let buffered = response.body().size_hint().exact().is_some();
    if !(inspect_body && response.status().is_success() && is_json(response.headers()) && buffered)
    {
        return response;
    }
    let (parts, body) = response.into_parts();
    let limit = body.size_hint().exact().map_or(0, |n| n as usize);
    match to_bytes(body, limit).await {
        Ok(bytes) => {
            let mixed = generate_is_mixed(&bytes, version.as_ref());
            let mut response = Response::from_parts(parts, Body::from(bytes));
            if mixed {
                response
                    .headers_mut()
                    .insert(MIXED_VERSION.clone(), HeaderValue::from_static("true"));
                smg_rl::metrics::record_mixed_version();
            }
            response
        }
        Err(e) => {
            warn!(error = %e, "RL middleware could not re-read a buffered generate body");
            error::internal_error("read_response_body_failed", "Failed to read response body")
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::worker::ConnectionMode;
    use smg_rl::{RlConfig, Version, VersionSource};

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        policies::CacheAwarePolicy,
        worker::{BasicWorkerBuilder, WorkerType},
    };

    fn registry_with(worker: Arc<dyn Worker>) -> Arc<WorkerRegistry> {
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(worker);
        registry
    }

    fn config_with_rl(rl: RlConfig) -> RouterConfig {
        RouterConfig {
            rl,
            ..Default::default()
        }
    }

    fn rl_enabled_config() -> RouterConfig {
        config_with_rl(RlConfig {
            enabled: true,
            ..RlConfig::default()
        })
    }

    /// Control calls must use the client the gateway negotiated for the
    /// worker (HTTP version, TLS identity, pool tuning), not a second one.
    #[test]
    fn view_hands_out_the_worker_negotiated_http_client() {
        let client = Arc::new(reqwest::Client::new());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://engine:30000")
                .connection_mode(ConnectionMode::Http)
                .http_client(Arc::clone(&client))
                .build(),
        );
        let view = RegistryRlView::new(registry_with(worker));

        let info = view.list().pop().expect("one worker");
        let handed = info.http_client.expect("HTTP worker carries a client");
        assert!(Arc::ptr_eq(&handed, &client));
    }

    /// A worker the gateway does not speak HTTP to has no client to hand out.
    #[test]
    fn view_gives_no_http_client_for_grpc_workers() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://engine:30000")
                .connection_mode(ConnectionMode::Grpc)
                .build(),
        );
        let view = RegistryRlView::new(registry_with(worker));

        let info = view.list().pop().expect("one worker");
        assert!(info.http_client.is_none());
    }

    /// The mixed-version accounting is per engine, so a DP group's ranks —
    /// separate registry entries sharing one base URL — must add up, and a
    /// second engine's load must not leak in.
    #[test]
    fn inflight_sums_ranks_sharing_a_base_url() {
        let registry = Arc::new(WorkerRegistry::new());
        let rank0: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://engine:30000")
                .dp_config(0, 2)
                .build(),
        );
        let rank1: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://engine:30000")
                .dp_config(1, 2)
                .build(),
        );
        let other: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("http://other:30000").build());
        for worker in [&rank0, &rank1, &other] {
            registry.register(Arc::clone(worker));
        }
        rank0.increment_load();
        rank1.increment_load();
        other.increment_load();

        let view = RegistryRlView::new(Arc::clone(&registry));
        assert_eq!(view.inflight("http://engine:30000"), 2);
        assert_eq!(view.inflight("http://other:30000"), 1);
        assert_eq!(view.inflight("http://absent:30000"), 0);
    }

    /// Registration labels seed the table and the last removal for an engine
    /// drops the entry, so the filter never judges a worker the registry no
    /// longer has.
    #[tokio::test]
    async fn maintainer_seeds_and_drops_entries() {
        let registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        let rl = build_rl_state(&registry, &policy_registry, &rl_enabled_config())
            .expect("rl is enabled");
        assert!(policy_registry.has_candidate_filter());

        let versioned: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://engine-a:30000")
                .label("weight_version", "5")
                .build(),
        );
        let unversioned: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("http://engine-b:30000").build());
        registry.register(Arc::clone(&versioned));
        registry.register(Arc::clone(&unversioned));
        settle().await;

        let entry = rl
            .table()
            .get("http://engine-a:30000")
            .expect("the label seeded an entry");
        assert_eq!(entry.version, Some(Version::parse("5")));
        assert_eq!(entry.version_source, Some(VersionSource::Registration));
        let entry = rl
            .table()
            .get("http://engine-b:30000")
            .expect("an unlabeled worker still gets an entry");
        assert_eq!(entry.version, None);
        assert_eq!(entry.version_source, None);

        registry.remove_by_url("http://engine-a:30000");
        settle().await;
        assert!(rl.table().get("http://engine-a:30000").is_none());
        assert!(rl.table().get("http://engine-b:30000").is_some());
    }

    /// One engine, many registry entries: a DP group's ranks share a base
    /// URL and one table entry, so a single rank leaving must not take the
    /// engine's version with it — only the last one does.
    #[tokio::test]
    async fn removing_one_dp_rank_keeps_the_engines_entry() {
        let registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        let rl = build_rl_state(&registry, &policy_registry, &rl_enabled_config())
            .expect("rl is enabled");

        for rank in 0..2 {
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new("http://engine:30000")
                    .dp_config(rank, 2)
                    .label("weight_version", "5")
                    .build(),
            );
            registry.register(worker);
        }
        settle().await;
        let entry = rl
            .table()
            .get("http://engine:30000")
            .expect("both ranks seed the one engine entry");
        assert_eq!(entry.version, Some(Version::parse("5")));

        registry.remove_by_url("http://engine:30000@0");
        settle().await;
        let entry = rl
            .table()
            .get("http://engine:30000")
            .expect("rank 1 still serves this engine");
        assert_eq!(
            entry.version,
            Some(Version::parse("5")),
            "a departing rank must not reset the engine's version"
        );

        registry.remove_by_url("http://engine:30000@1");
        settle().await;
        assert!(
            rl.table().get("http://engine:30000").is_none(),
            "the last rank leaving drops the engine"
        );
    }

    /// Let the maintainer task drain the event stream.
    async fn settle() {
        for _ in 0..16 {
            tokio::task::yield_now().await;
        }
    }

    /// Coupling (a): a refit invalidates the engine's KV cache, so the
    /// cache-aware trees must forget what it held or the next request with
    /// that prefix is routed to a worker that no longer has it.
    #[tokio::test]
    async fn version_change_resets_cache_aware_entries() {
        let registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 0,
            max_tree_size: 4096,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            overlap_decay: 0.0,
            selection_temperature: 0.0,
            cache_index: Default::default(),
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
        }));
        let rl = build_rl_state(&registry, &policy_registry, &rl_enabled_config())
            .expect("rl is enabled");

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://engine:30000")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        let id = registry
            .register(Arc::clone(&worker))
            .expect("registration returns an id");
        let model = worker.model_id().to_string();
        policy_registry.on_worker_added(&model, None);
        policy_registry.init_cache_aware_policy(&model, &[Arc::clone(&worker)]);
        let policy = policy_registry
            .get_policy(&model)
            .expect("the model has a policy");
        let cache_aware = policy
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()
            .expect("cache-aware policy");

        let text = "a long shared instruction block this engine has served before";
        cache_aware.insert_text_for_test(&model, text, worker.url());
        assert_eq!(
            cache_aware.string_prefix_for_tenant(&model, text, worker.url()),
            text,
            "precondition: the tree holds the prefix"
        );

        let view = RegistryRlView::new(Arc::clone(&registry));
        let info = view.get(id.as_str()).expect("the worker is registered");
        assert!(rl.apply_version(&info, Version::parse("2"), VersionSource::Api));

        assert_eq!(
            cache_aware.string_prefix_for_tenant(&model, text, worker.url()),
            "",
            "a refit must clear what the engine's cache no longer holds"
        );
    }

    /// The gateway must not outlive itself: the policy registry owns the
    /// candidate filter, which owns the state, which owns the sink — so the
    /// sink's handles back to both registries have to be weak or nothing is
    /// ever freed and the maintainer task never exits.
    #[tokio::test]
    async fn dropping_the_context_frees_the_registries() {
        let registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        let rl = build_rl_state(&registry, &policy_registry, &rl_enabled_config())
            .expect("rl is enabled");

        let weak_rl = Arc::downgrade(&rl);
        let weak_registry = Arc::downgrade(&registry);
        let weak_policies = Arc::downgrade(&policy_registry);

        drop(rl);
        drop(registry);
        drop(policy_registry);
        settle().await;

        assert!(
            weak_policies.upgrade().is_none(),
            "the policy registry outlived the context"
        );
        assert!(
            weak_rl.upgrade().is_none(),
            "the RL state outlived the context"
        );
        assert!(
            weak_registry.upgrade().is_none(),
            "the worker registry outlived the context"
        );
    }

    /// The disabled path constructs nothing: no state, and no filter on the
    /// policy registry to slow the hot path down.
    #[test]
    fn disabled_builds_nothing_and_installs_no_filter() {
        let registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        let config = RouterConfig::default();
        assert!(!config.rl.enabled, "precondition: the flag defaults off");

        assert!(build_rl_state(&registry, &policy_registry, &config).is_none());
        assert!(!policy_registry.has_candidate_filter());
    }

    /// The state carries the gateway's own RL settings, not the crate
    /// defaults, so the control deadline a deployment configured applies.
    #[test]
    fn build_uses_the_configured_rl_settings() {
        let registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        let config = config_with_rl(RlConfig {
            enabled: true,
            control_timeout_secs: 7,
            ..RlConfig::default()
        });

        let rl = build_rl_state(&registry, &policy_registry, &config).expect("rl is enabled");
        assert_eq!(rl.config().control_timeout_secs, 7);
    }
}
