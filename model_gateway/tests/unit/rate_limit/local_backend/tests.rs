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

fn tenant_policy(tenant: &str, tpm: u32, rpm: u32, rules: Vec<ModelRuleSpec>) -> TenantPolicySpec {
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

#[test]
fn a_creation_that_finds_a_replacement_in_force_sends_the_reserve_back_to_the_policy() {
    let backend = LocalRateLimitBackend::new(policy_set(100, 5));
    let tenant = TenantKey::new("auth:team-red");
    let key = ScopeKey::TenantGlobal(tenant.clone());
    // A reserve's reading of the policy and the limits it took from it.
    let read = backend.policy.load_full();
    let limits = read.policy_for(&tenant).limits;
    // The replacement lands between that reading and the creation: its
    // key snapshot holds no bucket of the tenant for the walk.
    backend.replace_policy(policy_set(10, 1));
    assert!(
        backend.bucket_for(key.clone(), &read, limits).is_none(),
        "a creation from the superseded reading must insert nothing"
    );
    assert!(backend.buckets.get(&key).is_none());
    // The reserve reads again and creates at the limits in force.
    let live = backend.policy.load_full();
    let bucket = backend
        .bucket_for(key, &live, live.policy_for(&tenant).limits)
        .expect("the live reading creates");
    assert_eq!(bucket.limits(), live.policy_for(&tenant).limits);
    assert_ne!(bucket.limits(), limits);
    // Through the reserve: the budget of 10 tokens a minute refuses 60.
    assert!(matches!(
        backend.reserve_now(reserve_request("auth:team-red", 60)),
        ReserveOutcome::Denied { .. }
    ));
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
                    while rounds.load(Ordering::Acquire) == round && !stop.load(Ordering::Relaxed) {
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
        let policy = backend.policy.load_full();
        drop(backend.bucket_for(
            key.clone(),
            &policy,
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
