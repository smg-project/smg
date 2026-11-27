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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
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
    /// reserve path. A reserve that read the policy before the store and
    /// finds no bucket of its scope after the key snapshot creates none
    /// (`bucket_for`): it reads the new policy and comes again, so no bucket
    /// at the old limits escapes the walk. One replacement runs at a time.
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

    /// The bucket of `key`, created at `limits` when it has none. `limits`
    /// is the caller's reading of `read`; a creation that finds another
    /// policy in force, stored by a replacement since that reading, inserts
    /// nothing and answers `None`, for the caller to read the live policy
    /// again. The check runs under the shard's write lock, which a
    /// replacement's key snapshot excludes: a bucket in the map before the
    /// snapshot is walked to the new limits, and one created after it was
    /// checked against the policy the walk serves.
    fn bucket_for(
        &self,
        key: ScopeKey,
        read: &Arc<CompiledPolicySet>,
        limits: ScopeLimits,
    ) -> Option<Arc<ScopeBucket>> {
        // A shard read lock first: an existing bucket is the common case,
        // and the sweep's scan holds shard read locks, which never make a
        // reader wait.
        if let Some(bucket) = self.buckets.get(&key) {
            return Some(Arc::clone(&bucket));
        }
        match self.buckets.entry(key) {
            Entry::Occupied(entry) => Some(Arc::clone(entry.get())),
            Entry::Vacant(entry) => {
                let live = self.policy.load();
                if !Arc::ptr_eq(&live, read) {
                    return None;
                }
                Some(Arc::clone(
                    &entry.insert(Arc::new(ScopeBucket::new(limits))),
                ))
            }
        }
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

        // The tenant-global bucket, and the model rule's when one matches:
        // at most two, held as a pair, both resolved under one reading of
        // the policy. A creation that finds a replacement's policy in force
        // since that reading sends the reserve back to read it: the
        // replacement's walk never sees the bucket it would have made. A
        // pass ends unless a replacement lands inside it, and replacements
        // run one at a time.
        let (global, rule) = loop {
            let policy = self.policy.load();
            let tenant_policy = policy.policy_for(&request.tenant_key);
            let global_key = ScopeKey::TenantGlobal(request.tenant_key.clone());
            let Some(global) = self.bucket_for(global_key, &policy, tenant_policy.limits) else {
                continue;
            };
            let Some(rule) = request
                .model_id
                .and_then(|model_id| tenant_policy.matching_rule(model_id))
            else {
                break (global, None);
            };
            let rule_key =
                ScopeKey::ModelRule(request.tenant_key.clone(), Arc::clone(&rule.rule_id));
            match self.bucket_for(rule_key, &policy, rule.limits) {
                Some(bucket) => break (global, Some(bucket)),
                None => continue,
            }
        };

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
#[path = "../../tests/unit/rate_limit/local_backend/tests.rs"]
mod tests;
