//! Per-tenant and per-model in-flight caps: concurrency bulkheads.
//!
//! The admission bound ([`super::concurrency`]) caps the gateway's standing
//! concurrency as a whole; nothing stops one tenant with many parallel
//! streams, or one slow model, from holding every permit while staying under
//! its per-minute rate limits. A bulkhead caps the in-flight requests of one
//! key: the caller's tenant key, or the served model's id. The gateway takes
//! both at dispatch, once the model is known, after the admission bound and
//! before the router runs, and attaches the permits to the response body, so
//! a streaming response holds its slots to its last frame or the disconnect.
//!
//! The hot path is one sharded-map lookup and one compare-and-swap on the
//! key's counter: no lock every request shares. A key's entry goes when its
//! count returns to zero, so keys that come and go (callers known by their
//! address, model names the gateway does not serve) never pin memory. The
//! removal re-checks the count under the shard's write lock, which no
//! acquisition (made under the shard's read lock) can interleave with, so the
//! count stays exact across it.

use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

use axum::{http::StatusCode, response::Response};
use dashmap::DashMap;
use tracing::{info, warn};

use super::concurrency::shed_response;
use crate::{
    config::RouterConfig,
    observability::metrics::{metrics_labels, Metrics},
};

/// The scope whose cap turned a request away.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BulkheadScope {
    /// The caller's tenant key: the caller's own concurrency, answered as its
    /// rate limit is, with 429.
    Tenant,
    /// The served model: the model's capacity, answered as a saturated
    /// admission queue is, with 503.
    Model,
}

impl BulkheadScope {
    /// The `reason` of `smg_admission_queue_rejected_total`.
    #[must_use]
    pub fn reason(self) -> &'static str {
        match self {
            Self::Tenant => metrics_labels::ADMISSION_REJECTED_TENANT_CONCURRENCY,
            Self::Model => metrics_labels::ADMISSION_REJECTED_MODEL_CONCURRENCY,
        }
    }

    /// The answer to a request beyond the cap, counted under the scope's
    /// reason: the standard error envelope with `Retry-After`, as every
    /// admission shed.
    #[must_use]
    pub fn reject(self) -> Response {
        Metrics::record_admission_rejected(self.reason());
        match self {
            Self::Tenant => shed_response(
                StatusCode::TOO_MANY_REQUESTS,
                "tenant_concurrency_exceeded",
                "concurrency limit reached for this tenant",
            ),
            Self::Model => shed_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "model_concurrency_exceeded",
                "concurrency limit reached for this model",
            ),
        }
    }
}

/// One key's in-flight count.
struct Slot {
    key: Arc<str>,
    inflight: AtomicU32,
}

/// A cap on the in-flight requests of each key.
pub struct Bulkhead {
    cap: u32,
    slots: DashMap<Arc<str>, Arc<Slot>>,
}

impl Bulkhead {
    /// A bulkhead admitting at most `cap` (at least 1) requests per key.
    #[must_use]
    pub fn new(cap: u32) -> Arc<Self> {
        Arc::new(Self {
            cap: cap.max(1),
            slots: DashMap::new(),
        })
    }

    /// The most requests one key may have in flight.
    #[must_use]
    pub fn cap(&self) -> u32 {
        self.cap
    }

    /// The requests of `key` in flight.
    #[must_use]
    pub fn inflight(&self, key: &str) -> u32 {
        self.slots
            .get(key)
            .map_or(0, |slot| slot.inflight.load(Ordering::Acquire))
    }

    /// The keys with a request in flight.
    #[must_use]
    pub fn active_keys(&self) -> usize {
        self.slots.len()
    }

    /// A permit for one more request of `key`, or `None` at the cap.
    #[must_use]
    pub fn try_acquire(self: &Arc<Self>, key: &str) -> Option<BulkheadPermit> {
        // The common case: the key is in flight already. The increment is
        // made while `slot` borrows the map (the shard's read lock), which is
        // what keeps the removal at zero exact.
        if let Some(slot) = self.slots.get(key) {
            return self
                .take(slot.value())
                .then(|| self.permit(Arc::clone(slot.value())));
        }
        let key: Arc<str> = Arc::from(key);
        let slot = self.slots.entry(Arc::clone(&key)).or_insert_with(|| {
            Arc::new(Slot {
                key,
                inflight: AtomicU32::new(0),
            })
        });
        self.take(slot.value())
            .then(|| self.permit(Arc::clone(slot.value())))
    }

    /// One more request of `slot`'s key, unless it is at the cap.
    fn take(&self, slot: &Slot) -> bool {
        slot.inflight
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.cap).then_some(n + 1)
            })
            .is_ok()
    }

    fn permit(self: &Arc<Self>, slot: Arc<Slot>) -> BulkheadPermit {
        BulkheadPermit {
            bulkhead: Arc::clone(self),
            slot,
        }
    }
}

/// One request's slot under a [`Bulkhead`], returned when dropped.
pub struct BulkheadPermit {
    bulkhead: Arc<Bulkhead>,
    slot: Arc<Slot>,
}

impl Drop for BulkheadPermit {
    fn drop(&mut self) {
        if self.slot.inflight.fetch_sub(1, Ordering::AcqRel) == 1 {
            // Possibly the key's last request: the entry goes if the count is
            // still zero under the shard's write lock, which no acquisition
            // can interleave with.
            self.bulkhead.slots.remove_if(&self.slot.key, |_, slot| {
                slot.inflight.load(Ordering::Acquire) == 0
            });
        }
    }
}

/// The gateway's bulkheads: the per-tenant and the per-model cap, either of
/// them off. Built once at start-up from the configuration.
#[derive(Clone, Default)]
pub struct Bulkheads {
    tenant: Option<Arc<Bulkhead>>,
    model: Option<Arc<Bulkhead>>,
}

impl Bulkheads {
    /// No caps: every request passes, at the cost of two branches.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            tenant: None,
            model: None,
        }
    }

    /// Caps of `per_tenant` and `per_model` in-flight requests, 0 for none.
    #[must_use]
    pub fn new(per_tenant: usize, per_model: usize) -> Self {
        let cap = |n: usize| (n > 0).then(|| Bulkhead::new(u32::try_from(n).unwrap_or(u32::MAX)));
        Self {
            tenant: cap(per_tenant),
            model: cap(per_model),
        }
    }

    /// The caps `max_concurrent_requests_per_tenant` and
    /// `max_concurrent_requests_per_model` set, logged at start-up.
    #[must_use]
    pub fn from_config(config: &RouterConfig) -> Self {
        let bulkheads = Self::new(
            config.max_concurrent_requests_per_tenant,
            config.max_concurrent_requests_per_model,
        );
        if let Some(tenant) = &bulkheads.tenant {
            info!(
                "Concurrency bulkhead: at most {} in-flight requests per tenant (max_concurrent_requests_per_tenant); one beyond answers 429",
                tenant.cap()
            );
            if tenant_cap_has_one_key_for_every_caller(config) {
                warn!(
                    "Concurrency bulkhead: without tenant API keys or a trusted tenant header every caller resolves to one tenant key (its client address, or the shared API key), so behind a proxy the per-tenant cap is a gateway-wide cap"
                );
            }
        }
        if let Some(model) = &bulkheads.model {
            info!(
                "Concurrency bulkhead: at most {} in-flight requests per model (max_concurrent_requests_per_model); one beyond answers 503",
                model.cap()
            );
        }
        bulkheads
    }

    /// Whether neither cap is set.
    #[must_use]
    pub fn is_off(&self) -> bool {
        self.tenant.is_none() && self.model.is_none()
    }

    /// The per-tenant bulkhead, when set.
    #[must_use]
    pub fn tenant(&self) -> Option<&Arc<Bulkhead>> {
        self.tenant.as_ref()
    }

    /// The per-model bulkhead, when set.
    #[must_use]
    pub fn model(&self) -> Option<&Arc<Bulkhead>> {
        self.model.as_ref()
    }

    /// The permits of a request by `tenant` for `model`, the tenant's first
    /// (a tenant at its cap takes no model slot), or the scope at its cap. A
    /// scope without a key, or without a cap, takes no permit.
    pub fn try_acquire(
        &self,
        tenant: Option<&str>,
        model: Option<&str>,
    ) -> Result<BulkheadPermits, BulkheadScope> {
        let tenant = match (&self.tenant, tenant) {
            (Some(bulkhead), Some(key)) => {
                Some(bulkhead.try_acquire(key).ok_or(BulkheadScope::Tenant)?)
            }
            _ => None,
        };
        let model = match (&self.model, model) {
            (Some(bulkhead), Some(key)) => {
                Some(bulkhead.try_acquire(key).ok_or(BulkheadScope::Model)?)
            }
            _ => None,
        };
        Ok(BulkheadPermits { tenant, model })
    }
}

/// Whether the per-tenant cap would have one key for every caller: with no
/// tenant API keys and the tenant header untrusted, a caller's tenant key is
/// its client address (one address for everyone behind a proxy) or the shared
/// API key's identity.
#[must_use]
pub fn tenant_cap_has_one_key_for_every_caller(config: &RouterConfig) -> bool {
    config.max_concurrent_requests_per_tenant > 0
        && config.tenant_api_keys.is_empty()
        && !config.tenant_resolution.trust_tenant_header
}

/// A request's permits, returned together when dropped: the gateway attaches
/// them to the response body.
pub struct BulkheadPermits {
    tenant: Option<BulkheadPermit>,
    model: Option<BulkheadPermit>,
}

impl BulkheadPermits {
    /// Whether the request holds a tenant slot.
    #[must_use]
    pub fn holds_tenant_slot(&self) -> bool {
        self.tenant.is_some()
    }

    /// Whether the request holds a model slot.
    #[must_use]
    pub fn holds_model_slot(&self) -> bool {
        self.model.is_some()
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::atomic::AtomicUsize, thread};

    use axum::http::header::RETRY_AFTER;
    use metrics_exporter_prometheus::PrometheusBuilder;

    use super::*;
    use crate::{
        config::TenantApiKeyEntry, middleware::SHED_RETRY_AFTER_SECS,
        routers::error::HEADER_X_SMG_ERROR_CODE,
    };

    #[test]
    fn a_key_is_capped_and_keys_are_independent() {
        let bulkhead = Bulkhead::new(2);
        let first = bulkhead.try_acquire("a").expect("the first of a");
        let second = bulkhead.try_acquire("a").expect("the second of a");
        assert!(bulkhead.try_acquire("a").is_none(), "a is at its cap");
        assert_eq!(bulkhead.inflight("a"), 2);
        let other = bulkhead.try_acquire("b").expect("b is not bounded by a");
        assert_eq!(bulkhead.inflight("b"), 1);
        assert_eq!(bulkhead.active_keys(), 2);
        drop((first, second, other));
    }

    #[test]
    fn a_dropped_permit_frees_its_slot_and_an_idle_key_leaves_the_map() {
        let bulkhead = Bulkhead::new(1);
        let held = bulkhead.try_acquire("a").expect("the first");
        assert!(bulkhead.try_acquire("a").is_none());
        drop(held);
        assert_eq!(bulkhead.inflight("a"), 0);
        assert_eq!(bulkhead.active_keys(), 0, "an idle key holds no entry");
        let again = bulkhead.try_acquire("a").expect("the slot is free again");
        assert_eq!(bulkhead.active_keys(), 1);
        drop(again);
        assert_eq!(bulkhead.active_keys(), 0);
    }

    #[test]
    fn the_tenant_is_taken_before_the_model_and_a_refusal_holds_nothing() {
        let bulkheads = Bulkheads::new(1, 1);
        let held = bulkheads
            .try_acquire(Some("tenant-a"), Some("model-1"))
            .expect("the first request");
        assert!(held.holds_tenant_slot() && held.holds_model_slot());

        // The tenant at its cap is refused as a tenant, before the model's
        // slot is looked at.
        assert_eq!(
            bulkheads
                .try_acquire(Some("tenant-a"), Some("model-2"))
                .err(),
            Some(BulkheadScope::Tenant)
        );
        // Another tenant on the saturated model is refused as a model, and
        // its tenant slot, taken first, is given back.
        assert_eq!(
            bulkheads
                .try_acquire(Some("tenant-b"), Some("model-1"))
                .err(),
            Some(BulkheadScope::Model)
        );
        assert_eq!(bulkheads.tenant().unwrap().inflight("tenant-b"), 0);
        assert_eq!(bulkheads.tenant().unwrap().active_keys(), 1);
        let other = bulkheads
            .try_acquire(Some("tenant-b"), Some("model-2"))
            .expect("tenant-b holds nothing, model-2 is free");
        drop((held, other));
        assert_eq!(bulkheads.tenant().unwrap().active_keys(), 0);
        assert_eq!(bulkheads.model().unwrap().active_keys(), 0);
    }

    #[test]
    fn a_scope_without_a_key_or_a_cap_takes_no_permit() {
        let off = Bulkheads::new(0, 0);
        assert!(off.is_off());
        assert!(Bulkheads::none().is_off());
        assert!(Bulkheads::default().is_off());
        let permits = off
            .try_acquire(Some("tenant-a"), Some("model-1"))
            .expect("no caps");
        assert!(!permits.holds_tenant_slot() && !permits.holds_model_slot());

        let tenant_only = Bulkheads::new(1, 0);
        assert!(!tenant_only.is_off());
        let permits = tenant_only
            .try_acquire(Some("tenant-a"), None)
            .expect("the tenant's first");
        assert!(permits.holds_tenant_slot() && !permits.holds_model_slot());
        let no_tenant = tenant_only
            .try_acquire(None, Some("model-1"))
            .expect("no tenant key, no tenant slot");
        assert!(!no_tenant.holds_tenant_slot());
        drop((permits, no_tenant));
    }

    #[test]
    fn the_configuration_sets_the_caps() {
        let config = RouterConfig {
            max_concurrent_requests_per_tenant: 8,
            max_concurrent_requests_per_model: 0,
            ..Default::default()
        };
        let bulkheads = Bulkheads::from_config(&config);
        assert_eq!(bulkheads.tenant().map(|b| b.cap()), Some(8));
        assert!(bulkheads.model().is_none());
        assert!(Bulkheads::from_config(&RouterConfig::default()).is_off());
    }

    /// The per-tenant cap is flagged at start-up when nothing tells the
    /// callers apart: no tenant API keys and the tenant header untrusted.
    #[test]
    fn a_per_tenant_cap_without_tenant_identities_is_flagged() {
        let capped = RouterConfig {
            max_concurrent_requests_per_tenant: 8,
            ..Default::default()
        };
        assert!(tenant_cap_has_one_key_for_every_caller(&capped));
        assert!(!tenant_cap_has_one_key_for_every_caller(
            &RouterConfig::default()
        ));
        let mut trusted = capped.clone();
        trusted.tenant_resolution.trust_tenant_header = true;
        assert!(!tenant_cap_has_one_key_for_every_caller(&trusted));
        let mut keyed = capped.clone();
        keyed.tenant_api_keys.push(TenantApiKeyEntry {
            tenant_id: "team-red".to_string(),
            key: "team-red-secret".to_string(),
        });
        assert!(!tenant_cap_has_one_key_for_every_caller(&keyed));
    }

    /// Many threads taking and returning one key's slots: the count in
    /// flight never passes the cap, and the key's entry is gone at the end.
    #[test]
    fn concurrent_acquisitions_never_exceed_the_cap() {
        const CAP: u32 = 4;
        const THREADS: usize = 8;
        const ROUNDS: usize = 5_000;
        let bulkhead = Bulkhead::new(CAP);
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let admitted = Arc::new(AtomicUsize::new(0));
        let workers: Vec<_> = (0..THREADS)
            .map(|_| {
                let bulkhead = Arc::clone(&bulkhead);
                let in_flight = Arc::clone(&in_flight);
                let peak = Arc::clone(&peak);
                let admitted = Arc::clone(&admitted);
                thread::spawn(move || {
                    for _ in 0..ROUNDS {
                        let Some(permit) = bulkhead.try_acquire("shared") else {
                            thread::yield_now();
                            continue;
                        };
                        let now = in_flight.fetch_add(1, Ordering::AcqRel) + 1;
                        peak.fetch_max(now, Ordering::AcqRel);
                        admitted.fetch_add(1, Ordering::Relaxed);
                        in_flight.fetch_sub(1, Ordering::AcqRel);
                        drop(permit);
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().expect("a worker thread");
        }
        assert!(
            peak.load(Ordering::Acquire) <= CAP as usize,
            "peak {} over the cap {CAP}",
            peak.load(Ordering::Acquire)
        );
        assert!(admitted.load(Ordering::Relaxed) > 0);
        assert_eq!(bulkhead.inflight("shared"), 0);
        assert_eq!(bulkhead.active_keys(), 0, "the idle key left the map");
    }

    #[test]
    fn a_rejection_has_the_shed_shape_and_is_counted_by_scope() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let (tenant, model) = metrics::with_local_recorder(&recorder, || {
            (
                BulkheadScope::Tenant.reject(),
                BulkheadScope::Model.reject(),
            )
        });

        assert_eq!(tenant.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            tenant.headers().get(HEADER_X_SMG_ERROR_CODE).unwrap(),
            "tenant_concurrency_exceeded"
        );
        assert_eq!(
            tenant.headers().get(RETRY_AFTER).unwrap(),
            &SHED_RETRY_AFTER_SECS.to_string()
        );
        assert_eq!(model.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            model.headers().get(HEADER_X_SMG_ERROR_CODE).unwrap(),
            "model_concurrency_exceeded"
        );
        assert_eq!(
            model.headers().get(RETRY_AFTER).unwrap(),
            &SHED_RETRY_AFTER_SECS.to_string()
        );

        let rendered = handle.render();
        for line in [
            "smg_admission_queue_rejected_total{reason=\"tenant_concurrency\"} 1",
            "smg_admission_queue_rejected_total{reason=\"model_concurrency\"} 1",
        ] {
            assert!(
                rendered.lines().any(|l| l == line),
                "expected `{line}`; rendered:\n{rendered}"
            );
        }
    }
}
