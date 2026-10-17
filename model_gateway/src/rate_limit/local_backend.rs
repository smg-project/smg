//! In-memory [`RateLimitBackend`]: per-tenant (+ optional per-model-rule)
//! token/request buckets, reserved once per logical request and settled
//! from final usage.
//!
//! Per-instance only: a tenant's true limit across N independent SMG
//! instances is roughly N× the configured value. Exact cluster-wide
//! enforcement is out of scope for this backend — that's what a future
//! `External` (distributed) backend behind the same trait is for.

use std::sync::Arc;

use async_trait::async_trait;
use dashmap::{mapref::entry::Entry, DashMap};
use uuid::Uuid;

use super::{
    backend::{RateLimitBackend, ReserveOutcome, ReserveRequest, UsageSettlement},
    bucket::ScopeBucket,
    policy::{CompiledPolicySet, ScopeLimits},
};
use crate::tenant::TenantKey;

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
enum ScopeKey {
    TenantGlobal(TenantKey),
    ModelRule(TenantKey, String),
}

/// What was actually debited for a reservation, kept around so
/// settle/close/abandon know which scopes to touch (and, for settlement,
/// how much to true up).
struct ActiveReservation {
    scopes: Vec<ScopeKey>,
    reserved_tokens: u32,
}

pub struct LocalRateLimitBackend {
    // Hot-reload lands in a later phase; the ArcSwap seam is in place now
    // so that phase is additive rather than an API change.
    policy: arc_swap::ArcSwap<CompiledPolicySet>,
    buckets: DashMap<ScopeKey, Arc<ScopeBucket>>,
    active: DashMap<Uuid, ActiveReservation>,
}

impl LocalRateLimitBackend {
    pub fn new(policy: CompiledPolicySet) -> Self {
        Self {
            policy: arc_swap::ArcSwap::from(Arc::new(policy)),
            buckets: DashMap::new(),
            active: DashMap::new(),
        }
    }

    fn bucket_for(&self, key: ScopeKey, limits: ScopeLimits) -> Arc<ScopeBucket> {
        self.buckets
            .entry(key)
            .or_insert_with(|| Arc::new(ScopeBucket::new(limits)))
            .clone()
    }
}

#[async_trait]
impl RateLimitBackend for LocalRateLimitBackend {
    async fn reserve(&self, request: ReserveRequest) -> ReserveOutcome {
        // Idempotent per request_charge_id: a retry reuses the same
        // charge_id and must not be debited twice. Claim the slot before
        // any bucket work so a duplicate (even concurrent) call just
        // confirms the existing reservation. Holding entry()'s lock here
        // is safe since nothing below awaits.
        let active_entry = match self.active.entry(request.request_charge_id) {
            Entry::Occupied(_) => return ReserveOutcome::Admitted,
            Entry::Vacant(entry) => entry,
        };

        let policy = self.policy.load();
        let tenant_policy = policy.policy_for(&request.tenant_key);

        let mut scopes: Vec<(ScopeKey, Arc<ScopeBucket>)> = Vec::with_capacity(2);
        let global_key = ScopeKey::TenantGlobal(request.tenant_key.clone());
        let global_bucket = self.bucket_for(global_key.clone(), tenant_policy.limits);
        scopes.push((global_key, global_bucket));

        if let Some(model_id) = request.model_id.as_deref() {
            if let Some(rule) = tenant_policy.matching_rule(model_id) {
                let rule_key =
                    ScopeKey::ModelRule(request.tenant_key.clone(), rule.rule_id.clone());
                let rule_bucket = self.bucket_for(rule_key.clone(), rule.limits);
                scopes.push((rule_key, rule_bucket));
            }
        }

        // Fixed lock order (tenant-global, then model-rule): every
        // reservation for the same tenant locks in this same order, so
        // this can never deadlock. Different tenants never share a bucket,
        // so cross-tenant ordering doesn't matter.
        let mut locked: Vec<_> = scopes.iter().map(|(_, b)| (b, b.lock())).collect();
        for (bucket, guard) in &mut locked {
            bucket.refill(guard);
        }

        let tokens_needed = f64::from(request.estimated_input_tokens);
        let mut denied_retry_after: Option<u64> = None;
        for (bucket, guard) in &locked {
            let dry = bucket.dry_run(guard, tokens_needed, 1.0);
            if !dry.allowed {
                denied_retry_after =
                    Some(denied_retry_after.unwrap_or(0).max(dry.retry_after_secs));
            }
        }

        if let Some(retry_after_secs) = denied_retry_after {
            return ReserveOutcome::Denied { retry_after_secs };
        }

        for (bucket, guard) in &mut locked {
            bucket.debit(guard, tokens_needed, 1.0);
        }
        drop(locked);

        active_entry.insert(ActiveReservation {
            scopes: scopes.into_iter().map(|(k, _)| k).collect(),
            reserved_tokens: request.estimated_input_tokens,
        });

        ReserveOutcome::Admitted
    }

    async fn settle_success(&self, request_charge_id: Uuid, usage: UsageSettlement) {
        let Some((_, reservation)) = self.active.remove(&request_charge_id) else {
            return;
        };
        let actual_total =
            i64::from(usage.actual_input_tokens) + i64::from(usage.completion_tokens);
        let delta = (actual_total - i64::from(reservation.reserved_tokens)) as f64;
        for key in &reservation.scopes {
            if let Some(bucket) = self.buckets.get(key) {
                let mut guard = bucket.lock();
                // Refill first, or a later deferred refill can overshoot
                // capacity and clamp, silently erasing this delta.
                bucket.refill(&mut guard);
                bucket.settle_tokens(&mut guard, delta);
            }
        }
    }

    async fn close_reserved_only(&self, request_charge_id: Uuid) {
        // The reserved-input debit already happened at reserve() time;
        // nothing further to debit. Just drop the bookkeeping entry.
        self.active.remove(&request_charge_id);
    }

    async fn abandon(&self, request_charge_id: Uuid) {
        // Same as close_reserved_only: the reserved-input debit is kept,
        // deliberately not refunded (see
        // `SharedReservationHandle::abandon_if_open`).
        self.active.remove(&request_charge_id);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::rate_limit::config::{
        ModelMatcherSpec, ModelRuleSpec, RateLimitYaml, TenantPolicySpec,
    };

    fn policy_set(tpm: u32, rpm: u32) -> CompiledPolicySet {
        let yaml = RateLimitYaml {
            default_policy: TenantPolicySpec {
                tenant_key: None,
                tokens_per_minute: tpm,
                requests_per_minute: rpm,
                model_rules: Vec::new(),
            },
            tenants: Vec::new(),
        };
        CompiledPolicySet::compile(&yaml).unwrap()
    }

    fn request(tenant: &str, tokens: u32) -> ReserveRequest {
        ReserveRequest {
            request_charge_id: Uuid::now_v7(),
            tenant_key: TenantKey::new(tenant),
            model_id: None,
            estimated_input_tokens: tokens,
        }
    }

    #[tokio::test]
    async fn reserve_admits_within_budget_and_denies_over() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        let first = backend.reserve(request("auth:team-red", 60)).await;
        assert_eq!(first, ReserveOutcome::Admitted);

        let second = backend.reserve(request("auth:team-red", 60)).await;
        assert!(matches!(second, ReserveOutcome::Denied { .. }));
    }

    #[tokio::test]
    async fn different_tenants_have_independent_budgets() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        assert_eq!(
            backend.reserve(request("auth:team-red", 90)).await,
            ReserveOutcome::Admitted
        );
        assert_eq!(
            backend.reserve(request("auth:team-blue", 90)).await,
            ReserveOutcome::Admitted
        );
    }

    #[tokio::test]
    async fn repeated_reserve_with_same_charge_id_does_not_double_charge() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        let req = request("auth:team-red", 60);
        let charge_id = req.request_charge_id;

        // First attempt debits 60 of 100; the retry (same charge_id)
        // must not debit again.
        assert_eq!(backend.reserve(req.clone()).await, ReserveOutcome::Admitted);
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);

        // Remaining 40 still fits -- would be denied if the retry had
        // double-debited.
        assert_eq!(
            backend.reserve(request("auth:team-red", 40)).await,
            ReserveOutcome::Admitted
        );

        // Settlement must see the original bookkeeping, not have it lost
        // to the retry overwriting it.
        backend
            .settle_success(
                charge_id,
                UsageSettlement {
                    actual_input_tokens: 60,
                    completion_tokens: 0,
                    ..UsageSettlement::default()
                },
            )
            .await;
        // 100 - 60 (settled exactly) - 40 (still open) = 0 remaining.
        assert!(matches!(
            backend.reserve(request("auth:team-red", 1)).await,
            ReserveOutcome::Denied { .. }
        ));
    }

    #[tokio::test]
    async fn settle_with_more_usage_than_reserved_creates_debt() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        let req = request("auth:team-red", 50);
        let charge_id = req.request_charge_id;
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);

        backend
            .settle_success(
                charge_id,
                UsageSettlement {
                    actual_input_tokens: 50,
                    completion_tokens: 60,
                    ..UsageSettlement::default()
                },
            )
            .await;

        // 100 capacity - 50 reserved - 60 extra from settlement = -10 (debt).
        let next = backend.reserve(request("auth:team-red", 1)).await;
        assert!(matches!(next, ReserveOutcome::Denied { .. }));
    }

    /// The cached and reasoning split of a settlement is carried, not
    /// charged: the true-up reads the input and completion counts alone.
    #[tokio::test]
    async fn settle_charges_the_counts_and_not_their_split() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        let req = request("auth:team-red", 50);
        let charge_id = req.request_charge_id;
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);

        backend
            .settle_success(
                charge_id,
                UsageSettlement {
                    actual_input_tokens: 50,
                    completion_tokens: 10,
                    cached_input_tokens: 40,
                    reasoning_tokens: 8,
                },
            )
            .await;

        // 100 - 60 = 40 remain: the split neither charges twice nor refunds.
        assert_eq!(
            backend.reserve(request("auth:team-red", 40)).await,
            ReserveOutcome::Admitted
        );
        assert!(matches!(
            backend.reserve(request("auth:team-red", 1)).await,
            ReserveOutcome::Denied { .. }
        ));
    }

    #[tokio::test]
    async fn settle_refills_before_charging_so_idle_time_cannot_erase_the_charge() {
        // Near-full bucket so a short idle period saturates the refill at
        // settle time. Without refilling before the charge, that
        // overshoot gets replayed later, clawing back part of the charge.
        let backend = LocalRateLimitBackend::new(policy_set(6000, 1000));
        let req = request("auth:team-red", 2);
        let charge_id = req.request_charge_id;
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);

        tokio::time::sleep(Duration::from_millis(600)).await;
        backend
            .settle_success(
                charge_id,
                UsageSettlement {
                    actual_input_tokens: 2,
                    completion_tokens: 500,
                    ..UsageSettlement::default()
                },
            )
            .await;

        // 6000 - 500 charged = 5500 available; the bug would admit this.
        // Assert well above 5500 (not the exact boundary) so a few extra
        // tokens of scheduling-jitter refill can't flip the outcome.
        assert!(matches!(
            backend.reserve(request("auth:team-red", 5530)).await,
            ReserveOutcome::Denied { .. }
        ));
    }

    #[tokio::test]
    async fn close_reserved_only_keeps_the_debit() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        let req = request("auth:team-red", 100);
        let charge_id = req.request_charge_id;
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);

        backend.close_reserved_only(charge_id).await;

        let next = backend.reserve(request("auth:team-red", 1)).await;
        assert!(matches!(next, ReserveOutcome::Denied { .. }));
    }

    #[tokio::test]
    async fn abandon_keeps_the_debit() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        let req = request("auth:team-red", 100);
        let charge_id = req.request_charge_id;
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);

        backend.abandon(charge_id).await;

        let next = backend.reserve(request("auth:team-red", 1)).await;
        assert!(matches!(next, ReserveOutcome::Denied { .. }));
    }

    #[tokio::test]
    async fn model_rule_scope_is_independently_limited() {
        let yaml = RateLimitYaml {
            default_policy: TenantPolicySpec {
                tenant_key: None,
                tokens_per_minute: 10_000,
                requests_per_minute: 1_000,
                model_rules: vec![ModelRuleSpec {
                    rule_id: "gpt4".to_string(),
                    matcher: ModelMatcherSpec::Exact {
                        value: "gpt-4".to_string(),
                    },
                    tokens_per_minute: 50,
                    requests_per_minute: 5,
                }],
            },
            tenants: Vec::new(),
        };
        let backend = LocalRateLimitBackend::new(CompiledPolicySet::compile(&yaml).unwrap());

        let req = ReserveRequest {
            request_charge_id: Uuid::now_v7(),
            tenant_key: TenantKey::new("auth:team-red"),
            model_id: Some("gpt-4".to_string()),
            estimated_input_tokens: 40,
        };
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);

        // The tenant-global budget (10,000) has plenty left, but the
        // gpt-4 rule budget (50) is nearly exhausted — the rule scope
        // denies even though the request has a small ask.
        let second = ReserveRequest {
            request_charge_id: Uuid::now_v7(),
            tenant_key: TenantKey::new("auth:team-red"),
            model_id: Some("gpt-4".to_string()),
            estimated_input_tokens: 20,
        };
        assert!(matches!(
            backend.reserve(second).await,
            ReserveOutcome::Denied { .. }
        ));

        // A different model on the same tenant isn't affected by the rule
        // scope, only the tenant-global one, which still has budget.
        let other_model = ReserveRequest {
            request_charge_id: Uuid::now_v7(),
            tenant_key: TenantKey::new("auth:team-red"),
            model_id: Some("claude-3".to_string()),
            estimated_input_tokens: 20,
        };
        assert_eq!(backend.reserve(other_model).await, ReserveOutcome::Admitted);
    }
}
