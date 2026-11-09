//! In-memory [`RateLimitBackend`]: per-tenant (+ optional per-model-rule)
//! token/request buckets, reserved once per logical request and settled
//! from final usage.
//!
//! Per-instance only: a tenant's true limit across N independent SMG
//! instances is roughly N× the configured value. Exact cluster-wide
//! enforcement is out of scope for this backend — that's what a future
//! `External` (distributed) backend behind the same trait is for.
//!
//! A bucket is created on a tenant key's first request and every client IP
//! is its own tenant key, so a periodic sweep drops buckets nobody is using
//! (see `LocalRateLimitBackend::sweep_idle_buckets`); without it the map
//! grows with every distinct client for the life of the process.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use dashmap::{mapref::entry::Entry, DashMap};
use tokio::time::MissedTickBehavior;
use tracing::{debug, warn};
use uuid::Uuid;

use super::{
    backend::{RateLimitBackend, ReserveOutcome, ReserveRequest, UsageSettlement},
    bucket::ScopeBucket,
    policy::{CompiledPolicySet, ScopeLimits},
};
use crate::tenant::TenantKey;

/// How long a bucket must go untouched before the sweep may drop it. The
/// sweep also requires the bucket to have refilled to capacity, so this is
/// not what keeps eviction from forgiving usage; it keeps a tenant that
/// comes back every few minutes on the bucket it already has. At least one
/// refill window (60s), the time an emptied bucket takes to refill.
const IDLE_BUCKET_TTL: Duration = Duration::from_secs(5 * 60);

/// How often the sweep runs.
const IDLE_SWEEP_INTERVAL: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
enum ScopeKey {
    TenantGlobal(TenantKey),
    /// The tenant and the rule's id, the id shared with the compiled policy
    /// so a reservation under a rule copies nothing to name its bucket.
    ModelRule(TenantKey, Arc<str>),
}

/// What was actually debited for a reservation, kept around so
/// settle/close/abandon know which buckets to touch (and, for settlement,
/// how much to true up): the tenant-global bucket, and the model rule's
/// when one matched. Holding the buckets' `Arc`s is also what keeps the
/// idle sweep off them until the reservation ends.
struct ActiveReservation {
    global: Arc<ScopeBucket>,
    rule: Option<Arc<ScopeBucket>>,
    reserved_tokens: u32,
}

impl ActiveReservation {
    /// The buckets debited, in lock order.
    fn buckets(&self) -> impl Iterator<Item = &Arc<ScopeBucket>> {
        std::iter::once(&self.global).chain(self.rule.iter())
    }
}

/// What the in-memory backend's `replace_policy` did to the live buckets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PolicyReplacement {
    /// Buckets whose scope kept its limits, kept with their fill.
    pub kept: usize,
    /// Buckets replaced at new limits, their spending carried.
    pub replaced: usize,
    /// Buckets of a scope the policy no longer names, dropped.
    pub dropped: usize,
}

pub struct LocalRateLimitBackend {
    /// The policy in force, read once per reservation and swapped whole
    /// by [`Self::replace_policy`].
    policy: arc_swap::ArcSwap<CompiledPolicySet>,
    buckets: DashMap<ScopeKey, Arc<ScopeBucket>>,
    active: DashMap<Uuid, ActiveReservation>,
    /// One replacement at a time: two interleaved would leave a bucket at
    /// the limits of the policy that lost. The reserve path never takes it.
    replacement: parking_lot::Mutex<()>,
}

impl LocalRateLimitBackend {
    pub fn new(policy: CompiledPolicySet) -> Self {
        Self {
            policy: arc_swap::ArcSwap::from(Arc::new(policy)),
            buckets: DashMap::new(),
            active: DashMap::new(),
            replacement: parking_lot::Mutex::new(()),
        }
    }

    /// The backend, with its idle sweep running every
    /// `IDLE_SWEEP_INTERVAL` on the current Tokio runtime. `Err` outside
    /// a runtime.
    pub fn start(policy: CompiledPolicySet) -> Result<Arc<Self>, String> {
        Self::start_with_sweep(policy, IDLE_SWEEP_INTERVAL, IDLE_BUCKET_TTL)
    }

    fn start_with_sweep(
        policy: CompiledPolicySet,
        interval: Duration,
        idle_ttl: Duration,
    ) -> Result<Arc<Self>, String> {
        let runtime = tokio::runtime::Handle::try_current().map_err(|error| {
            format!("tenant rate limiting needs a Tokio runtime for its idle-bucket sweep: {error}")
        })?;
        let backend = Arc::new(Self::new(policy));
        let weak = Arc::downgrade(&backend);
        // Holds only a Weak<Self> between sweeps and exits within one
        // interval of the backend being dropped. Nothing waits for the task
        // at shutdown; a sweep already on the blocking pool delays the
        // runtime's drop by at most its own run.
        runtime.spawn(async move {
            let mut tick = tokio::time::interval(interval);
            tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
            tick.tick().await; // the first tick completes immediately
            loop {
                tick.tick().await;
                let Some(backend) = weak.upgrade() else {
                    break;
                };
                // The scan is synchronous and grows with the number of
                // buckets, so it runs on the blocking pool, not on a worker
                // that serves requests.
                let swept = tokio::task::spawn_blocking(move || {
                    let evicted = backend.sweep_idle_buckets(idle_ttl);
                    (evicted, backend.buckets.len())
                })
                .await;
                match swept {
                    Ok((0, _)) => {}
                    Ok((evicted, remaining)) => {
                        debug!(
                            evicted,
                            remaining, "tenant rate limiter dropped idle buckets"
                        );
                    }
                    Err(error) => warn!(%error, "tenant rate limiter's idle-bucket sweep failed"),
                }
            }
        });
        Ok(backend)
    }

    /// Puts `policy` in force for every reservation after this call and
    /// brings the live buckets to its limits. A bucket whose scope keeps
    /// its limits stays, fill included. One whose limits changed is replaced
    /// by a bucket carrying its spending ([`ScopeBucket::with_limits`]). One
    /// whose scope the policy no longer names (a model rule that is gone)
    /// is dropped; a tenant the policy no longer names falls to the default
    /// policy's limits. A reservation in flight settles its true-up on the
    /// bucket that replaced the one it holds, so the change forgives none
    /// of its completion tokens. A reserve that cloned a bucket's `Arc` just
    /// before its replacement debits the old one, which the new one does
    /// not see: one debit at the swap instant, cheaper than a lock on the
    /// reserve path. One replacement runs at a time.
    pub fn replace_policy(&self, policy: CompiledPolicySet) -> PolicyReplacement {
        let _turn = self.replacement.lock();
        let policy = Arc::new(policy);
        self.policy.store(Arc::clone(&policy));
        let mut report = PolicyReplacement::default();
        let keys: Vec<ScopeKey> = self
            .buckets
            .iter()
            .map(|entry| entry.key().clone())
            .collect();
        for key in keys {
            let limits = match &key {
                ScopeKey::TenantGlobal(tenant) => Some(policy.policy_for(tenant).limits),
                ScopeKey::ModelRule(tenant, rule_id) => policy
                    .policy_for(tenant)
                    .rule_by_id(rule_id)
                    .map(|rule| rule.limits),
            };
            let Some(limits) = limits else {
                if self.buckets.remove(&key).is_some() {
                    report.dropped += 1;
                }
                continue;
            };
            // Under the entry's shard write lock, as the sweep's eviction is: a
            // reserve that reaches the slot takes what is there.
            if let Entry::Occupied(mut entry) = self.buckets.entry(key) {
                if entry.get().limits() == limits {
                    report.kept += 1;
                } else {
                    let replacement = entry.get().with_limits(limits);
                    entry.insert(replacement);
                    report.replaced += 1;
                }
            }
        }
        report
    }

    fn bucket_for(&self, key: ScopeKey, limits: ScopeLimits) -> Arc<ScopeBucket> {
        // A shard read lock first: an existing bucket is the common case,
        // and the sweep's scan holds shard read locks, which never make a
        // reader wait.
        if let Some(bucket) = self.buckets.get(&key) {
            return Arc::clone(&bucket);
        }
        self.buckets
            .entry(key)
            .or_insert_with(|| Arc::new(ScopeBucket::new(limits)))
            .clone()
    }

    /// Drop every bucket that no reservation holds, that has gone untouched
    /// for at least `idle_ttl`, and that has refilled to capacity; returns
    /// how many were dropped. A dropped bucket is recreated full on its key's
    /// next request, which is the state it was dropped in, so no usage is
    /// forgiven (a bucket still paying off settlement debt is kept).
    ///
    /// Two passes, so no shard's write lock is held for more than one
    /// entry. The first scans under shard read locks and only reads two
    /// atomics per bucket to pick out the ones that look evictable: a
    /// reserve of an existing bucket never waits on it, and one that
    /// creates a bucket waits for at most the scan of its own shard. The
    /// second evicts each candidate with `remove_if`, which re-checks
    /// everything under the entry's shard write lock.
    ///
    /// That re-check is what makes the sweep race-free against `reserve`:
    /// `bucket_for` clones the map's `Arc` under the same shard's read or
    /// write lock, so a strong count of one there means no reserve holds the
    /// bucket and none can take it before it is gone. A reserve that comes
    /// after finds the slot vacant and starts a fresh bucket.
    fn sweep_idle_buckets(&self, idle_ttl: Duration) -> usize {
        let candidates = self.idle_bucket_candidates(idle_ttl);
        self.evict_idle_candidates(&candidates, idle_ttl)
    }

    fn idle_bucket_candidates(&self, idle_ttl: Duration) -> Vec<ScopeKey> {
        let now = Instant::now();
        self.buckets
            .iter()
            .filter(|entry| {
                Arc::strong_count(entry.value()) == 1 && entry.value().idle_at(now) >= idle_ttl
            })
            .map(|entry| entry.key().clone())
            .collect()
    }

    fn evict_idle_candidates(&self, candidates: &[ScopeKey], idle_ttl: Duration) -> usize {
        candidates
            .iter()
            .filter(|key| {
                self.buckets
                    .remove_if(*key, |_, bucket| {
                        // `is_full` takes the bucket's lock, which nothing
                        // else can hold once the count is one.
                        Arc::strong_count(bucket) == 1
                            && bucket.idle_at(Instant::now()) >= idle_ttl
                            && bucket.is_full()
                    })
                    .is_some()
            })
            .count()
    }
}

/// The contract's four calls, answered without waiting: nothing in this
/// backend awaits, so the manager and the handles call these directly and
/// build no boxed future per reservation. The trait's methods below wrap
/// them for a caller that holds the backend as a trait object.
impl LocalRateLimitBackend {
    pub(crate) fn reserve_now(&self, request: ReserveRequest<'_>) -> ReserveOutcome {
        // Idempotent per request_charge_id: a retry reuses the same
        // charge_id and must not be debited twice. Claim the slot before
        // any bucket work so a duplicate (even concurrent) call just
        // confirms the existing reservation. Holding entry()'s lock here
        // is safe since nothing below waits.
        let active_entry = match self.active.entry(request.request_charge_id) {
            Entry::Occupied(_) => return ReserveOutcome::Admitted,
            Entry::Vacant(entry) => entry,
        };

        let policy = self.policy.load();
        let tenant_policy = policy.policy_for(&request.tenant_key);

        // The tenant-global bucket, and the model rule's when one matches:
        // at most two, held as a pair.
        let global_key = ScopeKey::TenantGlobal(request.tenant_key.clone());
        let global = self.bucket_for(global_key, tenant_policy.limits);
        let rule = request
            .model_id
            .and_then(|model_id| tenant_policy.matching_rule(model_id))
            .map(|rule| {
                let rule_key =
                    ScopeKey::ModelRule(request.tenant_key.clone(), Arc::clone(&rule.rule_id));
                self.bucket_for(rule_key, rule.limits)
            });

        // Fixed lock order (tenant-global, then model-rule): every
        // reservation for the same tenant locks in this same order, so
        // this can never deadlock. Different tenants never share a bucket,
        // so cross-tenant ordering doesn't matter. Each lock refills its
        // bucket from the clock read that stamps the touch.
        let mut global_guard = global.lock_and_refill();
        let mut rule_guard = rule.as_ref().map(|bucket| bucket.lock_and_refill());

        let tokens_needed = f64::from(request.estimated_input_tokens);
        let mut denied_retry_after: Option<u64> = None;
        let locked = std::iter::once((&global, &global_guard)).chain(rule.iter().zip(&rule_guard));
        for (bucket, guard) in locked {
            let dry = bucket.dry_run(guard, tokens_needed, 1.0);
            if !dry.allowed {
                denied_retry_after =
                    Some(denied_retry_after.unwrap_or(0).max(dry.retry_after_secs));
            }
        }

        if let Some(retry_after_secs) = denied_retry_after {
            return ReserveOutcome::Denied { retry_after_secs };
        }

        global.debit(&mut global_guard, tokens_needed, 1.0);
        if let (Some(bucket), Some(guard)) = (&rule, rule_guard.as_mut()) {
            bucket.debit(guard, tokens_needed, 1.0);
        }
        drop(global_guard);
        drop(rule_guard);

        active_entry.insert(ActiveReservation {
            global,
            rule,
            reserved_tokens: request.estimated_input_tokens,
        });

        ReserveOutcome::Admitted
    }

    pub(crate) fn settle_success_now(&self, request_charge_id: Uuid, usage: UsageSettlement) {
        let Some((_, reservation)) = self.active.remove(&request_charge_id) else {
            return;
        };
        let actual_total =
            i64::from(usage.actual_input_tokens) + i64::from(usage.completion_tokens);
        let delta = (actual_total - i64::from(reservation.reserved_tokens)) as f64;
        for bucket in reservation.buckets() {
            // The bucket in force for the scope, should a replacement have
            // succeeded the one the reservation holds.
            bucket.settle_live(delta);
        }
    }

    pub(crate) fn close_reserved_only_now(&self, request_charge_id: Uuid) {
        // The reserved-input debit already happened at reserve() time;
        // nothing further to debit. Just drop the bookkeeping entry.
        self.active.remove(&request_charge_id);
    }

    pub(crate) fn abandon_now(&self, request_charge_id: Uuid) {
        // Same as close_reserved_only: the reserved-input debit is kept,
        // deliberately not refunded (see
        // `SharedReservationHandle::abandon_if_open`).
        self.active.remove(&request_charge_id);
    }
}

#[async_trait]
impl RateLimitBackend for LocalRateLimitBackend {
    async fn reserve(&self, request: ReserveRequest<'_>) -> ReserveOutcome {
        self.reserve_now(request)
    }

    async fn settle_success(&self, request_charge_id: Uuid, usage: UsageSettlement) {
        self.settle_success_now(request_charge_id, usage);
    }

    async fn close_reserved_only(&self, request_charge_id: Uuid) {
        self.close_reserved_only_now(request_charge_id);
    }

    async fn abandon(&self, request_charge_id: Uuid) {
        self.abandon_now(request_charge_id);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::rate_limit::{
        config::{ModelMatcherSpec, ModelRuleSpec, RateLimitYaml, TenantPolicySpec},
        test_support::reserve_request,
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

    fn policy_yaml(default: (u32, u32), tenants: Vec<TenantPolicySpec>) -> RateLimitYaml {
        RateLimitYaml {
            default_policy: TenantPolicySpec {
                tenant_key: None,
                tokens_per_minute: default.0,
                requests_per_minute: default.1,
                model_rules: Vec::new(),
            },
            tenants,
        }
    }

    fn tenant_policy(
        tenant: &str,
        tpm: u32,
        rpm: u32,
        rules: Vec<ModelRuleSpec>,
    ) -> TenantPolicySpec {
        TenantPolicySpec {
            tenant_key: Some(tenant.to_string()),
            tokens_per_minute: tpm,
            requests_per_minute: rpm,
            model_rules: rules,
        }
    }

    fn exact_rule(rule_id: &str, model: &str, tpm: u32, rpm: u32) -> ModelRuleSpec {
        ModelRuleSpec {
            rule_id: rule_id.to_string(),
            matcher: ModelMatcherSpec::Exact {
                value: model.to_string(),
            },
            tokens_per_minute: tpm,
            requests_per_minute: rpm,
        }
    }

    fn compiled(yaml: &RateLimitYaml) -> CompiledPolicySet {
        CompiledPolicySet::compile(yaml).unwrap()
    }

    fn denied(outcome: ReserveOutcome) -> u64 {
        match outcome {
            ReserveOutcome::Denied { retry_after_secs } => retry_after_secs,
            ReserveOutcome::Admitted => panic!("admitted where a refusal was expected"),
        }
    }

    #[tokio::test]
    async fn replace_keeps_a_bucket_whose_limits_stand_with_its_fill() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        assert_eq!(
            backend.reserve(reserve_request("auth:team-red", 60)).await,
            ReserveOutcome::Admitted
        );

        let report = backend.replace_policy(policy_set(100, 5));
        assert_eq!(
            report,
            PolicyReplacement {
                kept: 1,
                replaced: 0,
                dropped: 0
            }
        );
        // 40 of the 100 are left: the fill carried with the bucket.
        denied(backend.reserve(reserve_request("auth:team-red", 50)).await);
        assert_eq!(
            backend.reserve(reserve_request("auth:team-red", 40)).await,
            ReserveOutcome::Admitted
        );
    }

    #[tokio::test]
    async fn replace_with_a_raised_limit_opens_headroom_at_once() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        assert_eq!(
            backend.reserve(reserve_request("auth:team-red", 100)).await,
            ReserveOutcome::Admitted
        );
        denied(backend.reserve(reserve_request("auth:team-red", 1)).await);

        let report = backend.replace_policy(policy_set(300, 5));
        assert_eq!(report.replaced, 1);
        // The 100 spent carry over: 200 of the 300 are open now.
        assert_eq!(
            backend.reserve(reserve_request("auth:team-red", 150)).await,
            ReserveOutcome::Admitted
        );
        denied(backend.reserve(reserve_request("auth:team-red", 60)).await);
        assert_eq!(
            backend.reserve(reserve_request("auth:team-red", 50)).await,
            ReserveOutcome::Admitted
        );
    }

    #[tokio::test]
    async fn replace_with_a_limit_below_the_spending_admits_again_within_one_window() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        assert_eq!(
            backend.reserve(reserve_request("auth:team-red", 100)).await,
            ReserveOutcome::Admitted
        );

        backend.replace_policy(policy_set(10, 5));
        // 100 spent against a capacity of 10: the counter starts empty, not
        // 90 below zero, so the smallest ask waits for its refill and no
        // longer than the window (the old window's spending at the new rate
        // would be nine minutes).
        let retry_after = denied(backend.reserve(reserve_request("auth:team-red", 1)).await);
        assert!(retry_after > 0 && retry_after <= 60, "{retry_after}");
    }

    #[tokio::test]
    async fn replace_with_a_lowered_limit_carries_the_settlement_debt_in_full() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        let req = reserve_request("auth:team-red", 100);
        let charge_id = req.request_charge_id;
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);
        // 50 tokens of debt: the completion used more than was reserved.
        backend
            .settle_success(
                charge_id,
                UsageSettlement {
                    actual_input_tokens: 100,
                    completion_tokens: 50,
                    ..UsageSettlement::default()
                },
            )
            .await;

        backend.replace_policy(policy_set(10, 5));
        // The debt is carried as it was: 50 below zero against a capacity
        // of 10 waits longer than one window, as it would have at the old
        // rate for a smaller bucket.
        let retry_after = denied(backend.reserve(reserve_request("auth:team-red", 1)).await);
        assert!(retry_after > 60 && retry_after < u64::MAX, "{retry_after}");
    }

    #[tokio::test]
    async fn replace_drops_a_removed_tenant_to_the_default_policy() {
        let generous = policy_yaml(
            (100, 5),
            vec![tenant_policy("auth:team-red", 1000, 50, vec![])],
        );
        let backend = LocalRateLimitBackend::new(compiled(&generous));
        assert_eq!(
            backend.reserve(reserve_request("auth:team-red", 500)).await,
            ReserveOutcome::Admitted
        );

        let report = backend.replace_policy(compiled(&policy_yaml((100, 5), vec![])));
        assert_eq!(
            report.replaced, 1,
            "the bucket now carries the default's limits"
        );
        // 500 spent against the default's 100: refused until the refill;
        // another tenant is untouched.
        denied(backend.reserve(reserve_request("auth:team-red", 1)).await);
        assert_eq!(
            backend.reserve(reserve_request("auth:team-blue", 90)).await,
            ReserveOutcome::Admitted
        );
    }

    #[tokio::test]
    async fn replace_drops_the_bucket_of_a_rule_the_policy_no_longer_has() {
        let with_rule = policy_yaml(
            (1000, 50),
            vec![tenant_policy(
                "auth:team-red",
                1000,
                50,
                vec![exact_rule("gpt4", "gpt-4", 100, 5)],
            )],
        );
        let backend = LocalRateLimitBackend::new(compiled(&with_rule));
        let mut req = reserve_request("auth:team-red", 100);
        req.model_id = Some("gpt-4");
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);
        assert_eq!(
            backend.buckets.len(),
            2,
            "the tenant's bucket and the rule's"
        );

        let without_rule = policy_yaml(
            (1000, 50),
            vec![tenant_policy("auth:team-red", 1000, 50, vec![])],
        );
        let report = backend.replace_policy(compiled(&without_rule));
        assert_eq!(
            report,
            PolicyReplacement {
                kept: 1,
                replaced: 0,
                dropped: 1
            }
        );
        assert_eq!(backend.buckets.len(), 1);
        // The model is limited by the tenant's bucket alone now: 900 are left there.
        let mut req = reserve_request("auth:team-red", 900);
        req.model_id = Some("gpt-4");
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);
    }

    #[tokio::test]
    async fn a_reservation_open_across_a_replacement_settles_on_the_live_bucket() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        let req = reserve_request("auth:team-red", 50);
        let charge_id = req.request_charge_id;
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);

        backend.replace_policy(policy_set(200, 5));
        // The live bucket carries the 50 reserved; the true-up of 60
        // completion tokens reaches it through the replaced bucket's
        // successor, so 90 of the 200 are left.
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
        assert!(backend.active.is_empty(), "the reservation is closed");
        denied(backend.reserve(reserve_request("auth:team-red", 91)).await);
        assert_eq!(
            backend.reserve(reserve_request("auth:team-red", 90)).await,
            ReserveOutcome::Admitted
        );
        denied(backend.reserve(reserve_request("auth:team-red", 1)).await);
    }

    #[tokio::test]
    async fn a_settlement_follows_two_replacements_to_the_live_bucket() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        let req = reserve_request("auth:team-red", 50);
        let charge_id = req.request_charge_id;
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);

        backend.replace_policy(policy_set(200, 5));
        backend.replace_policy(policy_set(300, 5));
        // 50 reserved, carried twice; the true-up of 60 lands on the third
        // bucket: 190 of the 300 are left.
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
        denied(backend.reserve(reserve_request("auth:team-red", 191)).await);
        assert_eq!(
            backend.reserve(reserve_request("auth:team-red", 190)).await,
            ReserveOutcome::Admitted
        );
    }

    #[tokio::test]
    async fn reserve_admits_within_budget_and_denies_over() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        let first = backend.reserve(reserve_request("auth:team-red", 60)).await;
        assert_eq!(first, ReserveOutcome::Admitted);

        let second = backend.reserve(reserve_request("auth:team-red", 60)).await;
        assert!(matches!(second, ReserveOutcome::Denied { .. }));
    }

    #[tokio::test]
    async fn different_tenants_have_independent_budgets() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        assert_eq!(
            backend.reserve(reserve_request("auth:team-red", 90)).await,
            ReserveOutcome::Admitted
        );
        assert_eq!(
            backend.reserve(reserve_request("auth:team-blue", 90)).await,
            ReserveOutcome::Admitted
        );
    }

    #[tokio::test]
    async fn repeated_reserve_with_same_charge_id_does_not_double_charge() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        let req = reserve_request("auth:team-red", 60);
        let charge_id = req.request_charge_id;

        // First attempt debits 60 of 100; the retry (same charge_id)
        // must not debit again.
        assert_eq!(backend.reserve(req.clone()).await, ReserveOutcome::Admitted);
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);

        // Remaining 40 still fits -- would be denied if the retry had
        // double-debited.
        assert_eq!(
            backend.reserve(reserve_request("auth:team-red", 40)).await,
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
            backend.reserve(reserve_request("auth:team-red", 1)).await,
            ReserveOutcome::Denied { .. }
        ));
    }

    #[tokio::test]
    async fn settle_with_more_usage_than_reserved_creates_debt() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        let req = reserve_request("auth:team-red", 50);
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
        let next = backend.reserve(reserve_request("auth:team-red", 1)).await;
        assert!(matches!(next, ReserveOutcome::Denied { .. }));
    }

    /// The cached and reasoning split of a settlement is carried, not
    /// charged: the true-up reads the input and completion counts alone.
    #[tokio::test]
    async fn settle_charges_the_counts_and_not_their_split() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        let req = reserve_request("auth:team-red", 50);
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
            backend.reserve(reserve_request("auth:team-red", 40)).await,
            ReserveOutcome::Admitted
        );
        assert!(matches!(
            backend.reserve(reserve_request("auth:team-red", 1)).await,
            ReserveOutcome::Denied { .. }
        ));
    }

    #[tokio::test]
    async fn settle_refills_before_charging_so_idle_time_cannot_erase_the_charge() {
        // Near-full bucket so a short idle period saturates the refill at
        // settle time. Without refilling before the charge, that
        // overshoot gets replayed later, clawing back part of the charge.
        let backend = LocalRateLimitBackend::new(policy_set(6000, 1000));
        let req = reserve_request("auth:team-red", 2);
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
            backend
                .reserve(reserve_request("auth:team-red", 5530))
                .await,
            ReserveOutcome::Denied { .. }
        ));
    }

    #[tokio::test]
    async fn close_reserved_only_keeps_the_debit() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        let req = reserve_request("auth:team-red", 100);
        let charge_id = req.request_charge_id;
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);

        backend.close_reserved_only(charge_id).await;

        let next = backend.reserve(reserve_request("auth:team-red", 1)).await;
        assert!(matches!(next, ReserveOutcome::Denied { .. }));
    }

    #[tokio::test]
    async fn abandon_keeps_the_debit() {
        let backend = LocalRateLimitBackend::new(policy_set(100, 5));
        let req = reserve_request("auth:team-red", 100);
        let charge_id = req.request_charge_id;
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);

        backend.abandon(charge_id).await;

        let next = backend.reserve(reserve_request("auth:team-red", 1)).await;
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
            model_id: Some("gpt-4"),
            estimated_input_tokens: 40,
        };
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);

        // The tenant-global budget (10,000) has plenty left, but the
        // gpt-4 rule budget (50) is nearly exhausted — the rule scope
        // denies even though the request has a small ask.
        let second = ReserveRequest {
            request_charge_id: Uuid::now_v7(),
            tenant_key: TenantKey::new("auth:team-red"),
            model_id: Some("gpt-4"),
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
            model_id: Some("claude-3"),
            estimated_input_tokens: 20,
        };
        assert_eq!(backend.reserve(other_model).await, ReserveOutcome::Admitted);
    }

    /// Refills a token in ~0.1ms and a request in ~1ms, so a bucket a
    /// one-token reservation drew on is full again a few milliseconds later.
    fn fast_refill_policy() -> CompiledPolicySet {
        policy_set(600_000, 60_000)
    }

    #[tokio::test]
    async fn sweep_drops_an_idle_full_bucket_and_keeps_a_recent_one() {
        let backend = LocalRateLimitBackend::new(fast_refill_policy());
        let req = reserve_request("ip:203.0.113.7", 1);
        let charge_id = req.request_charge_id;
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);
        backend.close_reserved_only(charge_id).await;

        tokio::time::sleep(Duration::from_millis(50)).await;
        // Full again, but touched more recently than the TTL.
        assert_eq!(backend.sweep_idle_buckets(Duration::from_secs(3600)), 0);
        assert_eq!(backend.buckets.len(), 1);

        assert_eq!(backend.sweep_idle_buckets(Duration::from_millis(10)), 1);
        assert!(backend.buckets.is_empty());
    }

    #[tokio::test]
    async fn sweep_never_drops_a_bucket_an_open_reservation_holds() {
        let yaml = RateLimitYaml {
            default_policy: TenantPolicySpec {
                tenant_key: None,
                tokens_per_minute: 600_000,
                requests_per_minute: 60_000,
                model_rules: vec![ModelRuleSpec {
                    rule_id: "gpt4".to_string(),
                    matcher: ModelMatcherSpec::Exact {
                        value: "gpt-4".to_string(),
                    },
                    tokens_per_minute: 600_000,
                    requests_per_minute: 60_000,
                }],
            },
            tenants: Vec::new(),
        };
        let backend = LocalRateLimitBackend::new(CompiledPolicySet::compile(&yaml).unwrap());
        let gpt4 = |tokens| ReserveRequest {
            model_id: Some("gpt-4"),
            ..reserve_request("ip:203.0.113.7", tokens)
        };
        let req = gpt4(1);
        let charge_id = req.request_charge_id;
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);

        // Both scopes are idle and refilled: only the open reservation
        // keeps them.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(backend.sweep_idle_buckets(Duration::ZERO), 0);
        assert_eq!(backend.buckets.len(), 2);

        // The settlement lands on the buckets the map still serves: twice
        // the capacity over the reservation leaves both scopes in debt.
        backend
            .settle_success(
                charge_id,
                UsageSettlement {
                    actual_input_tokens: 1,
                    completion_tokens: 1_200_000,
                    ..UsageSettlement::default()
                },
            )
            .await;
        assert!(matches!(
            backend.reserve(gpt4(1)).await,
            ReserveOutcome::Denied { .. }
        ));
        assert!(matches!(
            backend.reserve(reserve_request("ip:203.0.113.7", 1)).await,
            ReserveOutcome::Denied { .. }
        ));
    }

    #[tokio::test]
    async fn sweep_keeps_a_bucket_in_settlement_debt() {
        let backend = LocalRateLimitBackend::new(fast_refill_policy());
        let req = reserve_request("ip:203.0.113.7", 1);
        let charge_id = req.request_charge_id;
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);
        // A full refill window of debt beyond capacity: dropping the bucket
        // now would forgive it.
        backend
            .settle_success(
                charge_id,
                UsageSettlement {
                    actual_input_tokens: 1,
                    completion_tokens: 1_200_000,
                    ..UsageSettlement::default()
                },
            )
            .await;

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(backend.sweep_idle_buckets(Duration::ZERO), 0);
        assert_eq!(backend.buckets.len(), 1);
        assert!(matches!(
            backend.reserve(reserve_request("ip:203.0.113.7", 1)).await,
            ReserveOutcome::Denied { .. }
        ));
    }

    #[tokio::test]
    async fn evicted_bucket_comes_back_at_full_capacity() {
        // 100 tokens/s: refills a one-token draw in ~10ms, but not a
        // hundred tokens in the few milliseconds the rest of the test takes.
        let backend = LocalRateLimitBackend::new(policy_set(6000, 60_000));
        let req = reserve_request("ip:203.0.113.7", 1);
        let charge_id = req.request_charge_id;
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);
        backend.close_reserved_only(charge_id).await;

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(backend.sweep_idle_buckets(Duration::from_millis(10)), 1);
        assert!(backend.buckets.is_empty());

        assert_eq!(
            backend
                .reserve(reserve_request("ip:203.0.113.7", 6000))
                .await,
            ReserveOutcome::Admitted
        );
        assert_eq!(backend.buckets.len(), 1);
        assert!(matches!(
            backend
                .reserve(reserve_request("ip:203.0.113.7", 100))
                .await,
            ReserveOutcome::Denied { .. }
        ));
    }

    /// A reserve racing the sweep either takes the bucket first (and so
    /// keeps the sweep off it) or finds it gone and starts a fresh one: an
    /// admitted reservation always holds the bucket the map serves for its
    /// key, never one the sweep removed out from under it.
    ///
    /// In rounds, so the sweep gets its evictions on any host: the reserve
    /// threads hammer the key while the sweep passes back to back, then
    /// park past their last close, and a pass over the parked bucket must
    /// leave the map empty; the next round's reserves start it afresh
    /// under the sweep. A free-running sweep has to find the bucket free
    /// at its scan and again at its `remove_if`, which three reserve
    /// threads allow a few times in a hundred passes on a many-core host,
    /// where a pass over the map's many shards takes tens of microseconds,
    /// so a floor of a thousand evictions it must win on its own is a
    /// floor on the host's scheduling, not on the backend.
    #[test]
    fn a_reserve_racing_the_sweep_never_debits_a_removed_bucket() {
        use std::{
            sync::atomic::{AtomicBool, AtomicUsize, Ordering},
            time::Instant,
        };

        use futures::FutureExt;

        const TENANT: &str = "ip:203.0.113.7";
        const RESERVE_THREADS: usize = 3;
        const ROUNDS: usize = 1000;
        const RESERVES_PER_ROUND: usize = 100;
        // Refills a one-token draw in nanoseconds, so the bucket is full,
        // and sweepable, almost whenever no reserve holds it.
        let backend = LocalRateLimitBackend::new(policy_set(u32::MAX, u32::MAX));
        let key = ScopeKey::TenantGlobal(TenantKey::new(TENANT));
        let stop = AtomicBool::new(false);
        let parked = AtomicUsize::new(0);
        let rounds = AtomicUsize::new(0);
        // The rounds are a fixed amount of work; this bounds a starved run.
        let deadline = Instant::now() + Duration::from_secs(10);

        std::thread::scope(|scope| {
            scope.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    backend.sweep_idle_buckets(Duration::ZERO);
                    if parked.load(Ordering::Acquire) == RESERVE_THREADS {
                        // Every reserve thread is past its close, so nothing
                        // holds the bucket and it has long refilled: this
                        // pass, if not the one just made, drops it.
                        backend.sweep_idle_buckets(Duration::ZERO);
                        assert!(
                            backend.buckets.is_empty(),
                            "the sweep kept a full bucket no reservation holds"
                        );
                        parked.store(0, Ordering::Relaxed);
                        rounds.fetch_add(1, Ordering::Release);
                    }
                    // A pass, then the core: on a host with fewer cores than
                    // threads the reserve threads run between passes rather
                    // than after the sweep's whole slice.
                    std::thread::yield_now();
                }
            });
            for _ in 0..RESERVE_THREADS {
                scope.spawn(|| {
                    while !stop.load(Ordering::Relaxed) {
                        let round = rounds.load(Ordering::Acquire);
                        for _ in 0..RESERVES_PER_ROUND {
                            let req = reserve_request(TENANT, 1);
                            let charge_id = req.request_charge_id;
                            // reserve/close never await, so one poll finishes them.
                            let outcome = backend.reserve(req).now_or_never().unwrap();
                            assert_eq!(outcome, ReserveOutcome::Admitted);
                            {
                                let held = backend.active.get(&charge_id).unwrap();
                                let served = backend.buckets.get(&key);
                                assert!(
                                    served.is_some_and(|b| Arc::ptr_eq(&held.global, b.value())),
                                    "an admitted reservation holds a bucket the map no longer serves"
                                );
                            }
                            backend
                                .close_reserved_only(charge_id)
                                .now_or_never()
                                .unwrap();
                        }
                        parked.fetch_add(1, Ordering::Release);
                        while rounds.load(Ordering::Acquire) == round
                            && !stop.load(Ordering::Relaxed)
                        {
                            std::thread::yield_now();
                        }
                    }
                });
            }
            while rounds.load(Ordering::Relaxed) < ROUNDS && Instant::now() < deadline {
                std::thread::yield_now();
            }
            stop.store(true, Ordering::Relaxed);
        });

        assert!(
            rounds.load(Ordering::Relaxed) >= ROUNDS,
            "the reserve threads and the sweep did not get through {ROUNDS} rounds in 10 s"
        );
    }

    /// Force a reserve after the sweep collects an idle candidate, before
    /// its final Arc-count recheck. Held reservations must keep that bucket
    /// served; after all closes the next sweep must evict it.
    #[test]
    fn a_reserve_between_sweep_passes_never_debits_a_removed_bucket() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Barrier,
        };

        use futures::FutureExt;

        const TENANT: &str = "ip:203.0.113.7";
        const ROUNDS: usize = 128;
        // Refills a one-token draw in nanoseconds, so the bucket is full,
        // and sweepable, almost whenever no reserve holds it.
        let backend = LocalRateLimitBackend::new(policy_set(u32::MAX, u32::MAX));
        let key = ScopeKey::TenantGlobal(TenantKey::new(TENANT));
        let phases = Barrier::new(4);
        let bad_reserves = AtomicUsize::new(0);
        let bad_candidates = AtomicUsize::new(0);
        let wrong_bucket = AtomicUsize::new(0);
        let bad_sweeps = AtomicUsize::new(0);
        let seed = || {
            drop(backend.bucket_for(
                key.clone(),
                ScopeLimits {
                    tokens_per_minute: u32::MAX,
                    requests_per_minute: u32::MAX,
                },
            ));
        };
        seed();

        std::thread::scope(|scope| {
            scope.spawn(|| {
                for _ in 0..ROUNDS {
                    // Pause the actual two-pass sweep after its count-of-one
                    // observation, then let all three reserves clone that Arc.
                    let candidates = backend.idle_bucket_candidates(Duration::ZERO);
                    if candidates.as_slice() != std::slice::from_ref(&key) {
                        bad_candidates.fetch_add(1, Ordering::Relaxed);
                    }
                    phases.wait(); // first pass finished; let reserves run
                    phases.wait(); // all reservations now hold their bucket
                    if backend.evict_idle_candidates(&candidates, Duration::ZERO) != 0 {
                        bad_sweeps.fetch_add(1, Ordering::Relaxed);
                    }
                    phases.wait(); // re-check finished; let readers check identity
                    phases.wait(); // all identity checks finished; let reserves close
                    phases.wait(); // all reservation-held Arcs have been dropped
                    if backend.sweep_idle_buckets(Duration::ZERO) != 1 {
                        bad_sweeps.fetch_add(1, Ordering::Relaxed);
                    }
                    seed();
                    phases.wait(); // prepare the next race before releasing readers
                }
            });
            for _ in 0..3 {
                scope.spawn(|| {
                    for _ in 0..ROUNDS {
                        phases.wait();
                        let req = reserve_request(TENANT, 1);
                        let charge_id = req.request_charge_id;
                        // reserve/close never await, so one poll finishes them.
                        if backend.reserve(req).now_or_never() != Some(ReserveOutcome::Admitted) {
                            bad_reserves.fetch_add(1, Ordering::Relaxed);
                        }
                        phases.wait();
                        phases.wait();
                        {
                            let matches =
                                match (backend.active.get(&charge_id), backend.buckets.get(&key)) {
                                    (Some(held), Some(served)) => held
                                        .buckets()
                                        .next()
                                        .is_some_and(|first| Arc::ptr_eq(first, served.value())),
                                    _ => false,
                                };
                            if !matches {
                                wrong_bucket.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        phases.wait();
                        if backend
                            .close_reserved_only(charge_id)
                            .now_or_never()
                            .is_none()
                        {
                            bad_reserves.fetch_add(1, Ordering::Relaxed);
                        }
                        phases.wait();
                        phases.wait();
                    }
                });
            }
        });

        // Assert after joining, so an invariant failure cannot strand another
        // participant at a barrier. Each round checks both held and idle states.
        assert_eq!(bad_reserves.load(Ordering::Relaxed), 0);
        assert_eq!(bad_candidates.load(Ordering::Relaxed), 0);
        assert_eq!(
            wrong_bucket.load(Ordering::Relaxed),
            0,
            "an admitted reservation holds a bucket the map no longer serves"
        );
        assert_eq!(bad_sweeps.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn started_backend_sweeps_on_its_own_and_does_not_outlive_its_owner() {
        let backend = LocalRateLimitBackend::start_with_sweep(
            fast_refill_policy(),
            Duration::from_millis(20),
            Duration::from_millis(10),
        )
        .unwrap();
        let req = reserve_request("ip:203.0.113.7", 1);
        let charge_id = req.request_charge_id;
        assert_eq!(backend.reserve(req).await, ReserveOutcome::Admitted);
        backend.close_reserved_only(charge_id).await;

        tokio::time::timeout(Duration::from_secs(5), async {
            while !backend.buckets.is_empty() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the sweep task never dropped the idle bucket");

        // The task holds only a Weak between sweeps, so the backend goes
        // with its last owner once a sweep in flight on the blocking pool
        // (which can outlast the emptying of the map) has returned.
        let weak = Arc::downgrade(&backend);
        drop(backend);
        tokio::time::timeout(Duration::from_secs(5), async {
            while weak.strong_count() > 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the sweep task kept the backend alive after its owner dropped it");
    }
}
