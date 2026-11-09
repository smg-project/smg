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
    local_backend::{LocalRateLimitBackend, PolicyReplacement},
    policy::CompiledPolicySet,
    reservation::SharedReservationHandle,
};
use crate::config::types::RouterConfig;

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
}

impl RateLimitManager {
    /// A manager over `backend`, an implementation of the
    /// [`RateLimitBackend`] contract other than the in-memory one (a
    /// test's, an external one's); [`Self::from_config`] builds the
    /// in-memory one.
    pub fn new(backend: Arc<dyn RateLimitBackend>) -> Self {
        Self {
            backend: Backend::Dyn(backend),
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
        // header: overrides are unreachable without trust_tenant_header --
        // tenant_resolution never constructs that key form otherwise.
        if !rc.tenant_resolution.trust_tenant_header {
            if let Some(tenant_key) = yaml.tenants.iter().find_map(|t| {
                t.tenant_key
                    .as_deref()
                    .filter(|key| key.starts_with("header:"))
            }) {
                return Err(format!(
                    "tenant_rate_limit_config has tenant_key '{tenant_key}' but tenant_resolution.trust_tenant_header is not set -- this override can never match a real request"
                ));
            }
        }
        let compiled = CompiledPolicySet::compile(&yaml).map_err(|e| e.to_string())?;
        let backend = LocalRateLimitBackend::start(compiled)?;
        Ok(Self {
            backend: Backend::Local(backend),
        })
    }

    /// Puts `policy` in force on the in-memory backend (its
    /// `replace_policy`). The caller holds the document the set was compiled
    /// from to the start's checks (a `header:` tenant key needs the trusted
    /// tenant header), since a compiled set carries no such rule. A backend
    /// behind the trait object takes no new policy while running: `Err`
    /// says so.
    pub fn replace_policy(&self, policy: CompiledPolicySet) -> Result<PolicyReplacement, String> {
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
    use crate::rate_limit::{config::TenantPolicySpec, test_support::reserve_request};

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
}
