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
#[path = "../../tests/unit/rate_limit/manager/tests.rs"]
mod tests;
