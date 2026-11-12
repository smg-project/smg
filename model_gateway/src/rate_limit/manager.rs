//! [`RateLimitManager`]: the facade routers/middleware call into.
//!
//! Built once at startup from the optional `--tenant-rate-limit-config`
//! YAML. `Ok(None)` when disabled; `Err` when enabled but the YAML is
//! invalid (or there is no Tokio runtime for the in-memory backend's idle
//! sweep) — callers must fail startup on `Err`, not run unlimited.

use std::sync::Arc;

use tracing::info;
use uuid::Uuid;

use super::{
    backend::{RateLimitBackend, ReserveOutcome, ReserveRequest, UsageSettlement},
    config::RateLimitYaml,
    document::{PolicyChanges, PolicySummary},
    local_backend::{LocalRateLimitBackend, PolicyReplacement},
    policy::CompiledPolicySet,
    reservation::SharedReservationHandle,
};
use crate::{
    admin::document::{etag, AdminDocument, Applied},
    config::types::RouterConfig,
};

/// Outcome of [`RateLimitManager::reserve`] — bundles admission with the
/// handle, since every admitted caller needs one immediately.
/// [`ReserveOutcome`] is the lighter backend-trait-level result.
pub enum Reservation {
    Admitted(Arc<SharedReservationHandle>),
    Denied { retry_after_secs: u64 },
}

/// The backend a manager and its handles call. The in-memory one is called
/// by static dispatch: none of its calls waits, so a reservation taken and
/// settled on it builds no boxed future. Any other implementation of the
/// [`RateLimitBackend`] contract is called through the trait object.
#[derive(Clone)]
pub(crate) enum Backend {
    Local(Arc<LocalRateLimitBackend>),
    Dyn(Arc<dyn RateLimitBackend>),
}

impl Backend {
    pub(crate) async fn reserve(&self, request: ReserveRequest<'_>) -> ReserveOutcome {
        match self {
            Self::Local(backend) => backend.reserve_now(request),
            Self::Dyn(backend) => backend.reserve(request).await,
        }
    }

    pub(crate) async fn settle_success(&self, request_charge_id: Uuid, usage: UsageSettlement) {
        match self {
            Self::Local(backend) => backend.settle_success_now(request_charge_id, usage),
            Self::Dyn(backend) => backend.settle_success(request_charge_id, usage).await,
        }
    }

    pub(crate) async fn close_reserved_only(&self, request_charge_id: Uuid) {
        match self {
            Self::Local(backend) => backend.close_reserved_only_now(request_charge_id),
            Self::Dyn(backend) => backend.close_reserved_only(request_charge_id).await,
        }
    }

    pub(crate) async fn abandon(&self, request_charge_id: Uuid) {
        match self {
            Self::Local(backend) => backend.abandon_now(request_charge_id),
            Self::Dyn(backend) => backend.abandon(request_charge_id).await,
        }
    }
}

impl From<Arc<dyn RateLimitBackend>> for Backend {
    fn from(backend: Arc<dyn RateLimitBackend>) -> Self {
        Self::Dyn(backend)
    }
}

/// Call [`Self::reserve`] once per logical request, before any internal
/// retry loop, and resolve the returned handle exactly once at the end.
/// Each call builds an independent handle, so calling it again mid-retry
/// risks one handle abandoning a reservation another still intends to
/// settle.
pub struct RateLimitManager {
    backend: Backend,
    /// The policy document in force with its revision: the start's file,
    /// then each document [`Self::replace_document`] applied. `None` for a
    /// manager over a backend of its own, which took no document.
    document: Option<AdminDocument<RateLimitYaml>>,
    /// Whether `header:` tenant keys can match a request, as the start
    /// checked its file; a replacement document is held to the same rule.
    trusts_tenant_header: bool,
}

/// What [`RateLimitManager::replace_document`] did.
#[derive(Debug)]
pub struct DocumentOutcome {
    /// The revision in force after the call: the new one when applied, the
    /// live one a dry run was checked against.
    pub revision: Arc<Applied<RateLimitYaml>>,
    /// `false` for a dry run: nothing was swapped and the revision stands.
    pub applied: bool,
    /// The shape of the document given.
    pub summary: PolicySummary,
    /// What it changes against the revision it was checked against.
    pub changes: PolicyChanges,
    /// What the application did to the live buckets; `None` for a dry run.
    pub buckets: Option<PolicyReplacement>,
}

/// Why [`RateLimitManager::replace_document`] refused.
#[derive(Debug)]
pub enum DocumentRefusal {
    /// The manager took no document: it runs over a backend of its own.
    NoDocument,
    /// The document fails the checks the start applies to its file; the
    /// message says where.
    Invalid(String),
    /// The ETag given is not the live revision's, carried here.
    Stale(Arc<Applied<RateLimitYaml>>),
    /// The backend in use takes no new policy while running.
    BackendFixed(String),
    /// The document has no JSON form to hash; nothing was applied.
    Record(serde_json::Error),
}

impl RateLimitManager {
    /// A manager over `backend`, an implementation of the
    /// [`RateLimitBackend`] contract other than the in-memory one (a
    /// test's, an external one's); [`Self::from_config`] builds the
    /// in-memory one.
    pub fn new(backend: Arc<dyn RateLimitBackend>) -> Self {
        Self {
            backend: Backend::Dyn(backend),
            document: None,
            trusts_tenant_header: false,
        }
    }

    pub fn from_config(rc: &RouterConfig) -> Result<Option<Arc<Self>>, String> {
        if !rc.tenant_rate_limit_enabled {
            return Ok(None);
        }
        let manager = Self::try_build(rc)?;
        info!("tenant rate limiter enabled");
        Ok(Some(Arc::new(manager)))
    }

    fn try_build(rc: &RouterConfig) -> Result<Self, String> {
        let yaml = load_yaml(rc.tenant_rate_limit_config.as_deref())?;
        let trusts_tenant_header = rc.tenant_resolution.trust_tenant_header;
        header_keys_reachable(&yaml, trusts_tenant_header)?;
        let compiled = CompiledPolicySet::compile(&yaml).map_err(|e| e.to_string())?;
        let backend = LocalRateLimitBackend::start(compiled)?;
        let document = AdminDocument::from_startup(yaml)
            .map_err(|e| format!("recording the tenant rate-limit document: {e}"))?;
        Ok(Self {
            backend: Backend::Local(backend),
            document: Some(document),
            trusts_tenant_header,
        })
    }

    /// The policy document in force with its revision and ETag; `None` for
    /// a manager over a backend of its own, which took no document.
    pub fn document(&self) -> Option<Arc<Applied<RateLimitYaml>>> {
        self.document.as_ref().map(AdminDocument::load)
    }

    /// `document` checked as the start checks its file, compiled and, unless
    /// `dry_run`, put in force as the next revision under `principal`; the
    /// live buckets follow the backend's `replace_policy` rule. `expected_etag`
    /// is the ETag of the revision the caller read: a write against a
    /// revision another writer replaced is refused with the live one. One
    /// writer at a time runs from the check to the swap.
    pub fn replace_document(
        &self,
        expected_etag: &str,
        document: RateLimitYaml,
        principal: &str,
        dry_run: bool,
    ) -> Result<DocumentOutcome, DocumentRefusal> {
        let Some(holder) = &self.document else {
            return Err(DocumentRefusal::NoDocument);
        };
        header_keys_reachable(&document, self.trusts_tenant_header)
            .map_err(DocumentRefusal::Invalid)?;
        let compiled = CompiledPolicySet::compile(&document)
            .map_err(|error| DocumentRefusal::Invalid(error.to_string()))?;
        let etag = etag(&document).map_err(DocumentRefusal::Record)?;
        let writer = holder.write();
        let live = writer.live();
        if live.etag != expected_etag {
            return Err(DocumentRefusal::Stale(live));
        }
        let summary = PolicySummary::of(&document);
        let changes = PolicyChanges::between(&live.document, &document);
        if dry_run {
            return Ok(DocumentOutcome {
                revision: live,
                applied: false,
                summary,
                changes,
                buckets: None,
            });
        }
        let buckets = self
            .replace_policy(compiled)
            .map_err(DocumentRefusal::BackendFixed)?;
        Ok(DocumentOutcome {
            revision: writer.apply(document, etag, principal),
            applied: true,
            summary,
            changes,
            buckets: Some(buckets),
        })
    }

    /// Puts `policy` in force on the in-memory backend (its
    /// `replace_policy`). The caller holds the document the set was compiled
    /// from to the start's checks (a `header:` tenant key needs the trusted
    /// tenant header), since a compiled set carries no such rule. A backend
    /// behind the trait object takes no new policy while running: `Err`
    /// says so. [`Self::replace_document`] alone calls it: a policy put in
    /// force past the document would leave the revision and its ETag behind.
    fn replace_policy(&self, policy: CompiledPolicySet) -> Result<PolicyReplacement, String> {
        match &self.backend {
            Backend::Local(backend) => Ok(backend.replace_policy(policy)),
            Backend::Dyn(_) => {
                Err("the rate-limit backend in use takes no new policy while running".to_string())
            }
        }
    }

    pub async fn reserve(&self, request: ReserveRequest<'_>) -> Reservation {
        let request_charge_id = request.request_charge_id;
        match self.backend.reserve(request).await {
            ReserveOutcome::Admitted => Reservation::Admitted(Arc::new(
                SharedReservationHandle::new(self.backend.clone(), request_charge_id),
            )),
            ReserveOutcome::Denied { retry_after_secs } => Reservation::Denied { retry_after_secs },
        }
    }
}

/// `header:` tenant keys match a request only when the tenant header is
/// trusted; `tenant_resolution` never builds that key form otherwise, so a
/// policy naming one would never apply. The start checks its file by this
/// rule and a replacement document is held to it too.
fn header_keys_reachable(yaml: &RateLimitYaml, trusts_tenant_header: bool) -> Result<(), String> {
    if trusts_tenant_header {
        return Ok(());
    }
    match yaml.tenants.iter().find_map(|t| {
        t.tenant_key
            .as_deref()
            .filter(|key| key.starts_with("header:"))
    }) {
        Some(tenant_key) => Err(format!(
            "tenant_rate_limit_config has tenant_key '{tenant_key}' but tenant_resolution.trust_tenant_header is not set -- this override can never match a real request"
        )),
        None => Ok(()),
    }
}

fn load_yaml(path: Option<&str>) -> Result<RateLimitYaml, String> {
    let path = path.ok_or_else(|| {
        "tenant_rate_limit_enabled is set but tenant_rate_limit_config is not".to_string()
    })?;
    let contents = std::fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
    serde_yaml::from_str(&contents).map_err(|e| format!("parsing {path}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rate_limit::{
        config::TenantPolicySpec,
        test_support::{reserve_request, RecordingBackend},
    };

    fn manager_with_limits(tpm: u32, rpm: u32) -> RateLimitManager {
        let yaml = RateLimitYaml {
            default_policy: TenantPolicySpec {
                tenant_key: None,
                tokens_per_minute: tpm,
                requests_per_minute: rpm,
                model_rules: Vec::new(),
            },
            tenants: Vec::new(),
        };
        let compiled = CompiledPolicySet::compile(&yaml).unwrap();
        RateLimitManager {
            backend: Backend::Local(Arc::new(LocalRateLimitBackend::new(compiled))),
            document: None,
            trusts_tenant_header: false,
        }
    }

    #[tokio::test]
    async fn admitted_reservation_carries_a_handle() {
        let manager = manager_with_limits(100, 5);
        match manager.reserve(reserve_request("auth:team-red", 50)).await {
            Reservation::Admitted(handle) => {
                // The handle is independently usable -- settle through it,
                // not through the backend directly.
                handle.settle_success(UsageSettlement::default()).await;
            }
            Reservation::Denied { .. } => panic!("expected admission"),
        }
    }

    #[tokio::test]
    async fn denied_reservation_carries_no_handle() {
        let manager = manager_with_limits(100, 5);
        assert!(matches!(
            manager.reserve(reserve_request("auth:team-red", 200)).await,
            Reservation::Denied { .. }
        ));
    }

    fn router_config(enabled: bool, config_path: Option<String>) -> RouterConfig {
        RouterConfig::builder()
            .worker_startup_timeout_secs(1)
            .tenant_rate_limit_enabled(enabled)
            .tenant_rate_limit_config(config_path)
            .build_unchecked()
    }

    #[test]
    fn disabled_returns_ok_none() {
        let rc = router_config(false, None);
        assert!(matches!(RateLimitManager::from_config(&rc), Ok(None)));
    }

    #[test]
    fn enabled_without_config_path_fails_startup() {
        let rc = router_config(true, None);
        assert!(RateLimitManager::from_config(&rc).is_err());
    }

    #[test]
    fn enabled_with_missing_file_fails_startup() {
        let rc = router_config(true, Some("/nonexistent/path/rate_limit.yaml".to_string()));
        assert!(RateLimitManager::from_config(&rc).is_err());
    }

    #[test]
    fn enabled_with_invalid_policy_fails_startup() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rate_limit.yaml");
        // default_policy must not set tenant_key -- this is invalid.
        std::fs::write(
            &path,
            "default_policy:\n  tenant_key: auth:oops\n  tokens_per_minute: 1000\n  requests_per_minute: 60\n",
        )
        .unwrap();

        let rc = router_config(true, Some(path.to_str().unwrap().to_string()));
        assert!(RateLimitManager::from_config(&rc).is_err());
    }

    #[tokio::test]
    async fn enabled_with_valid_policy_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rate_limit.yaml");
        std::fs::write(
            &path,
            "default_policy:\n  tokens_per_minute: 1000\n  requests_per_minute: 60\n",
        )
        .unwrap();

        let rc = router_config(true, Some(path.to_str().unwrap().to_string()));
        assert!(matches!(RateLimitManager::from_config(&rc), Ok(Some(_))));
    }

    /// The in-memory backend's idle sweep needs a runtime to run on: a
    /// valid policy built outside one fails startup rather than panicking
    /// or running without the sweep.
    #[test]
    fn enabled_outside_a_runtime_fails_startup() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rate_limit.yaml");
        std::fs::write(
            &path,
            "default_policy:\n  tokens_per_minute: 1000\n  requests_per_minute: 60\n",
        )
        .unwrap();

        let rc = router_config(true, Some(path.to_str().unwrap().to_string()));
        assert!(RateLimitManager::from_config(&rc).is_err());
    }

    #[test]
    fn header_tenant_key_without_trust_tenant_header_fails_startup() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rate_limit.yaml");
        std::fs::write(
            &path,
            "default_policy:\n  tokens_per_minute: 1000\n  requests_per_minute: 60\ntenants:\n  - tenant_key: header:team-red\n    tokens_per_minute: 500\n    requests_per_minute: 30\n",
        )
        .unwrap();

        // trust_tenant_header defaults to false, so this override could
        // never match a real request.
        let rc = RouterConfig::builder()
            .worker_startup_timeout_secs(1)
            .tenant_rate_limit_enabled(true)
            .tenant_rate_limit_config(Some(path.to_str().unwrap().to_string()))
            .build_unchecked();
        assert!(RateLimitManager::from_config(&rc).is_err());
    }

    #[tokio::test]
    async fn header_tenant_key_with_trust_tenant_header_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rate_limit.yaml");
        std::fs::write(
            &path,
            "default_policy:\n  tokens_per_minute: 1000\n  requests_per_minute: 60\ntenants:\n  - tenant_key: header:team-red\n    tokens_per_minute: 500\n    requests_per_minute: 30\n",
        )
        .unwrap();

        let rc = RouterConfig::builder()
            .worker_startup_timeout_secs(1)
            .tenant_rate_limit_enabled(true)
            .tenant_rate_limit_config(Some(path.to_str().unwrap().to_string()))
            .trust_tenant_header(true)
            .build_unchecked();
        assert!(matches!(RateLimitManager::from_config(&rc), Ok(Some(_))));
    }

    /// A manager from a file with one tenant beside the default.
    fn documented_manager(dir: &std::path::Path) -> Arc<RateLimitManager> {
        let path = dir.join("rate_limit.yaml");
        std::fs::write(
            &path,
            "default_policy:\n  tokens_per_minute: 1000\n  requests_per_minute: 60\ntenants:\n  - tenant_key: auth:team-red\n    tokens_per_minute: 500\n    requests_per_minute: 30\n",
        )
        .unwrap();
        RateLimitManager::from_config(&router_config(
            true,
            Some(path.to_str().unwrap().to_string()),
        ))
        .unwrap()
        .unwrap()
    }

    fn policy(tenant_key: Option<&str>, tokens_per_minute: u32) -> TenantPolicySpec {
        TenantPolicySpec {
            tenant_key: tenant_key.map(str::to_string),
            tokens_per_minute,
            requests_per_minute: 30,
            model_rules: Vec::new(),
        }
    }

    /// A document over the start file's default policy (1000 tokens, 60
    /// requests a minute).
    fn document(tenants: Vec<TenantPolicySpec>) -> RateLimitYaml {
        RateLimitYaml {
            default_policy: TenantPolicySpec {
                requests_per_minute: 60,
                ..policy(None, 1000)
            },
            tenants,
        }
    }

    #[tokio::test]
    async fn the_start_file_is_revision_one_under_the_startup_principal() {
        let dir = tempfile::tempdir().unwrap();
        let manager = documented_manager(dir.path());
        let revision = manager.document().expect("the start read a document");
        assert_eq!(revision.revision, 1);
        assert_eq!(revision.principal, "startup");
        assert_eq!(revision.document.tenants.len(), 1);
        assert!(
            revision.etag.starts_with('"') && revision.etag.ends_with('"'),
            "{}",
            revision.etag
        );
    }

    #[tokio::test]
    async fn a_write_against_another_revisions_etag_is_refused_with_the_live_one() {
        let dir = tempfile::tempdir().unwrap();
        let manager = documented_manager(dir.path());
        match manager.replace_document("\"stale\"", document(Vec::new()), "ops", false) {
            Err(DocumentRefusal::Stale(live)) => assert_eq!(live.revision, 1),
            other => panic!("expected the stale refusal, got {other:?}"),
        }
        assert_eq!(manager.document().unwrap().revision, 1);
    }

    /// Two writers on one ETag: the second waits for the first's turn and
    /// is refused with the revision the first applied. The holder's writer
    /// turn, held from the ETag check to the swap, is what makes it so; a
    /// turn per writer would let both apply.
    #[tokio::test]
    async fn a_second_writer_on_the_same_etag_waits_its_turn_and_is_refused() {
        use std::{sync::mpsc, thread, time::Duration};

        let dir = tempfile::tempdir().unwrap();
        let manager = documented_manager(dir.path());
        let first = manager.document().unwrap();
        let holder = manager.document.as_ref().unwrap();

        // The first writer's turn, held while the second writer calls in.
        let writer = holder.write();
        let (ready, at_the_turn) = mpsc::channel();
        let second = {
            let manager = Arc::clone(&manager);
            let etag = first.etag.clone();
            thread::spawn(move || {
                ready.send(()).unwrap();
                manager.replace_document(
                    &etag,
                    document(vec![policy(Some("auth:team-blue"), 100)]),
                    "second",
                    false,
                )
            })
        };
        at_the_turn.recv().unwrap();
        thread::sleep(Duration::from_millis(200));
        assert_eq!(
            manager.document().unwrap().revision,
            1,
            "the second writer waits for the turn"
        );

        let next = document(vec![policy(Some("auth:team-red"), 800)]);
        let tag = etag(&next).unwrap();
        assert_eq!(writer.apply(next, tag, "first").revision, 2);
        drop(writer);

        match second.join().unwrap() {
            Err(DocumentRefusal::Stale(live)) => {
                assert_eq!(live.revision, 2);
                assert_eq!(live.principal, "first");
            }
            other => panic!("expected the stale refusal, got {other:?}"),
        }
        assert_eq!(manager.document().unwrap().revision, 2);
    }

    #[tokio::test]
    async fn a_dry_run_reports_the_changes_and_leaves_the_revision() {
        let dir = tempfile::tempdir().unwrap();
        let manager = documented_manager(dir.path());
        let live = manager.document().unwrap();
        let next = document(vec![
            policy(Some("auth:team-red"), 600),
            policy(Some("auth:team-blue"), 100),
        ]);
        let outcome = manager
            .replace_document(&live.etag, next, "ops", true)
            .unwrap();
        assert!(!outcome.applied);
        assert_eq!(outcome.revision.revision, 1);
        assert_eq!(outcome.changes.tenants_added, vec!["auth:team-blue"]);
        assert_eq!(outcome.changes.tenants_changed, vec!["auth:team-red"]);
        assert!(outcome.changes.tenants_removed.is_empty());
        assert!(!outcome.changes.default_changed);
        assert_eq!(outcome.summary.tenants, 2);
        assert!(outcome.buckets.is_none());
        assert_eq!(manager.document().unwrap().revision, 1);
        // The start's limit stands: 600 tokens are over team-red's 500.
        assert!(matches!(
            manager.reserve(reserve_request("auth:team-red", 600)).await,
            Reservation::Denied { .. }
        ));
    }

    #[tokio::test]
    async fn an_applied_document_is_the_next_revision_and_in_force() {
        let dir = tempfile::tempdir().unwrap();
        let manager = documented_manager(dir.path());
        let live = manager.document().unwrap();
        let next = document(vec![policy(Some("auth:team-red"), 800)]);
        let outcome = manager
            .replace_document(&live.etag, next, "ops", false)
            .unwrap();
        assert!(outcome.applied);
        assert_eq!(outcome.revision.revision, 2);
        assert_eq!(outcome.revision.principal, "ops");
        assert_ne!(outcome.revision.etag, live.etag);
        assert_eq!(outcome.changes.tenants_changed, vec!["auth:team-red"]);
        assert_eq!(
            outcome.buckets,
            Some(PolicyReplacement::default()),
            "no bucket was live"
        );
        let now = manager.document().unwrap();
        assert_eq!(now.revision, 2);
        assert_eq!(now.document.tenants[0].tokens_per_minute, 800);
        // The new limit is in force: 600 tokens fit under 800.
        assert!(matches!(
            manager.reserve(reserve_request("auth:team-red", 600)).await,
            Reservation::Admitted(_)
        ));
    }

    #[tokio::test]
    async fn a_document_the_start_would_refuse_is_refused_and_nothing_moves() {
        let dir = tempfile::tempdir().unwrap();
        let manager = documented_manager(dir.path());
        let live = manager.document().unwrap();
        let twice = document(vec![
            policy(Some("auth:team-red"), 500),
            policy(Some("auth:team-red"), 400),
        ]);
        assert!(matches!(
            manager.replace_document(&live.etag, twice, "ops", false),
            Err(DocumentRefusal::Invalid(_))
        ));
        // A header: key without a trusted tenant header, as at the start.
        let header = document(vec![policy(Some("header:team-red"), 500)]);
        match manager.replace_document(&live.etag, header, "ops", false) {
            Err(DocumentRefusal::Invalid(message)) => {
                assert!(message.contains("trust_tenant_header"), "{message}");
            }
            other => panic!("expected the invalid refusal, got {other:?}"),
        }
        assert_eq!(manager.document().unwrap().revision, 1);
    }

    #[test]
    fn a_manager_over_a_backend_of_its_own_took_no_document() {
        let backend = Arc::new(RecordingBackend::default());
        let manager = RateLimitManager::new(backend as Arc<dyn RateLimitBackend>);
        assert!(manager.document().is_none());
        assert!(matches!(
            manager.replace_document("\"any\"", document(Vec::new()), "ops", false),
            Err(DocumentRefusal::NoDocument)
        ));
    }
}
