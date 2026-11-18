use std::{
    collections::HashMap,
    sync::{Arc, Barrier},
    thread,
};

use kv_index::{
    compute_request_content_hashes, salt::request_content_hashes_with_seed, SequenceHash,
    StoredBlock,
};
use openai_protocol::common::CachePartition;

use super::*;
use crate::{
    policies::{
        prefill_cost::reservation::{Reservation, ReservationLedger, ReservationSlot},
        test_support::make_workers,
        CacheAwareConfig, CacheAwarePolicy, CacheNamespace,
    },
    worker::{
        load::state::{one_rank_load, TestFeed},
        KvIndex, WorkerBlocks,
    },
};

const COLD: &str = "http://cold:8000";
const HOLDER: &str = "http://holder:8000";
const BLOCK: usize = 16;
const PROMPT: usize = 16_384;

/// Both actual policies see the same native index, load feed, worker order, and
/// arrivals. Cold comes first, so cache credit cannot pass as a first-row tie.
struct Host {
    policy: PrefillCostPolicy,
    legacy: CacheAwarePolicy,
    workers: Vec<Arc<dyn Worker>>,
    monitor: Arc<KvEventMonitor>,
    feed: TestFeed,
    prefix: Vec<u32>,
}

impl Host {
    fn new(cached: usize) -> Self {
        let workers = make_workers(&[COLD, HOLDER]);
        let policy = PrefillCostPolicy::default();
        let legacy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            block_size: BLOCK,
            // In particular, preserve the default cache threshold of 0.5.
            ..Default::default()
        });
        legacy.init_workers(&workers);
        let prefix: Vec<u32> = (1..=cached as u32).collect();
        let index = Arc::new(KvIndex::chain());
        let holder = index.intern_worker(HOLDER).unwrap();
        index.intern_worker(COLD).unwrap();
        let blocks: Vec<_> = compute_request_content_hashes(&prefix, BLOCK)
            .into_iter()
            .enumerate()
            .map(|(i, content_hash)| StoredBlock {
                seq_hash: SequenceHash(i as u64 + 1),
                content_hash,
            })
            .collect();
        index
            .apply_stored(holder, &blocks, None, &mut WorkerBlocks::default())
            .unwrap();
        let model = normalize_model_key(workers[0].model_id()).to_owned();
        let monitor = Arc::new(KvEventMonitor::new());
        monitor.indexers.insert(model.clone(), index);
        monitor.set_block_size(&model, BLOCK);
        policy.set_kv_event_monitor(Some(Arc::clone(&monitor)));
        legacy.set_kv_event_monitor(Some(Arc::clone(&monitor)));
        let feed = TestFeed::attached_to(|rx| {
            policy.set_load_receiver(rx.clone());
            legacy.set_load_receiver(rx);
        });
        Self {
            policy,
            legacy,
            workers,
            monitor,
            feed,
            prefix,
        }
    }

    fn prompt(&self, suffix: u32) -> Vec<u32> {
        let mut tokens = self.prefix.clone();
        tokens.extend((0..PROMPT - tokens.len()).map(|i| suffix + i as u32));
        tokens
    }

    fn reports(&self, holder_queue: i32, input: [Option<f64>; 2], output: [f64; 2]) {
        let mut cold = one_rank_load(0, 0.0, output[0]);
        cold.loads[0].prefill_throughput = input[0];
        let mut holder = one_rank_load(holder_queue, 0.0, output[1]);
        holder.loads[0].prefill_throughput = input[1];
        holder.loads[0].num_waiting_reqs = i32::from(holder_queue > 0);
        self.feed.publish(&HashMap::from([
            (COLD.to_owned(), cold),
            (HOLDER.to_owned(), holder),
        ]));
    }

    fn share_prefix_with_cold(&self) {
        let model = normalize_model_key(self.workers[0].model_id());
        let index = self.monitor.indexers.get(model).unwrap();
        let cold = index.worker_id(COLD).unwrap();
        let blocks: Vec<_> = compute_request_content_hashes(&self.prefix, BLOCK)
            .into_iter()
            .enumerate()
            .map(|(i, content_hash)| StoredBlock {
                seq_hash: SequenceHash(1_000_000 + i as u64),
                content_hash,
            })
            .collect();
        index
            .apply_stored(cold, &blocks, None, &mut WorkerBlocks::default())
            .unwrap();
    }

    fn select(&self, tokens: &[u32], ledger: &Arc<ReservationLedger>) -> (usize, Reservation) {
        let (idx, lease) = self.select_using(&self.policy, tokens, tokens.len(), None, ledger);
        (idx, lease.expect("prefill_cost hands off its reservation"))
    }

    fn select_using(
        &self,
        policy: &dyn LoadBalancingPolicy,
        tokens: &[u32],
        total_tokens: usize,
        namespace: Option<CacheNamespace>,
        ledger: &Arc<ReservationLedger>,
    ) -> (usize, Option<Reservation>) {
        let slot = ReservationSlot::new(Arc::clone(ledger)).with_total_tokens(total_tokens);
        // Match the registry: legacy policies do not create cost reservations.
        if policy.needs_reservation() {
            slot.require(total_tokens as u64);
        }
        let idx = policy
            .select_worker(
                &self.workers,
                &SelectWorkerInfo {
                    tokens: Some(tokens),
                    cache_namespace: namespace,
                    reservation: Some(&slot),
                    ..Default::default()
                },
            )
            .expect("an eligible worker must be selected");
        let lease = slot.take_for(self.workers[idx].url());
        (idx, lease)
    }
}

#[test]
fn native_partial_hit_keeps_cache_credit_then_spreads_long_siblings() {
    let host = Host::new(4_096);
    host.reports(0, [Some(2_000.0); 2], [100.0; 2]);
    let ledger = Arc::new(ReservationLedger::default());
    let first = host.select(&host.prompt(1_000_000), &ledger);
    let second = host.select(&host.prompt(2_000_000), &ledger);
    assert_eq!((first.0, second.0), (1, 0));
    assert_eq!(ledger.pending(HOLDER), 12_288);
    assert_eq!(ledger.pending(COLD), 16_384);
    drop((first, second));
    assert_eq!(ledger.active(HOLDER) + ledger.active(COLD), 0);
}

#[test]
fn simultaneous_actual_policy_calls_book_before_the_next_choice() {
    for _ in 0..8 {
        let host = Arc::new(Host::new(4_096));
        host.reports(0, [Some(2_000.0); 2], [100.0; 2]);
        let ledger = Arc::new(ReservationLedger::default());
        let barrier = Arc::new(Barrier::new(2));
        let threads: Vec<_> = (0..2)
            .map(|i| {
                let (host, ledger, barrier) =
                    (Arc::clone(&host), Arc::clone(&ledger), Arc::clone(&barrier));
                thread::spawn(move || {
                    let tokens = host.prompt(1_000_000 + i * 100_000);
                    barrier.wait();
                    host.select(&tokens, &ledger)
                })
            })
            .collect();
        // Returned leases remain owned until both choices have completed.
        let picks: Vec<_> = threads
            .into_iter()
            .map(|task| task.join().unwrap())
            .collect();
        assert_ne!(picks[0].0, picks[1].0);
        assert_eq!(ledger.active(HOLDER), 1);
        assert_eq!(ledger.active(COLD), 1);
        assert_eq!(ledger.pending(HOLDER), 12_288);
        assert_eq!(ledger.pending(COLD), 16_384);
    }
}

#[test]
fn paired_native_burst_cache_aware_keeps_both_on_holder_prefill_cost_splits() {
    let host = Host::new(4_096);
    host.reports(0, [Some(2_000.0); 2], [100.0; 2]);
    let arrivals = [host.prompt(1_000_000), host.prompt(2_000_000)];
    let legacy_ledger = Arc::new(ReservationLedger::default());
    let legacy: Vec<_> = arrivals
        .iter()
        .map(|tokens| host.select_using(&host.legacy, tokens, tokens.len(), None, &legacy_ledger))
        .collect();
    let ledger = Arc::new(ReservationLedger::default());
    let revised: Vec<_> = arrivals
        .iter()
        .map(|tokens| host.select(tokens, &ledger))
        .collect();
    assert_eq!((legacy[0].0, legacy[1].0), (1, 1));
    assert_eq!((revised[0].0, revised[1].0), (1, 0));
    assert_eq!(legacy_ledger.active(HOLDER), 0);
    assert_eq!(ledger.active(HOLDER), 1);
    assert_eq!(ledger.active(COLD), 1);
}

#[test]
fn paired_native_tiny_suffix_preserves_the_expensive_warm_prefix() {
    let host = Host::new(PROMPT - 64);
    host.reports(0, [Some(2_000.0); 2], [100.0; 2]);
    let arrivals = [host.prompt(1_000_000), host.prompt(2_000_000)];
    for policy in [&host.legacy as &dyn LoadBalancingPolicy, &host.policy] {
        let ledger = Arc::new(ReservationLedger::default());
        let picks: Vec<_> = arrivals
            .iter()
            .map(|tokens| host.select_using(policy, tokens, tokens.len(), None, &ledger))
            .collect();
        assert_eq!((picks[0].0, picks[1].0), (1, 1), "{}", policy.name());
        if policy.needs_reservation() {
            assert_eq!(ledger.pending(HOLDER), 128);
        }
    }
}

#[test]
fn fully_cached_requests_still_spread_while_the_first_is_decoding() {
    let host = Host::new(PROMPT);
    host.share_prefix_with_cold();
    host.reports(0, [Some(2_000.0); 2], [100.0; 2]);
    let tokens = host.prompt(1_000_000);
    let ledger = Arc::new(ReservationLedger::default());
    let mut first = host.select(&tokens, &ledger);
    first.1.dispatched();
    first.1.prefill_complete();
    let second = host.select(&tokens, &ledger);
    assert_ne!(first.0, second.0);
    assert_eq!(ledger.pending(COLD) + ledger.pending(HOLDER), 0);
    assert_eq!(ledger.active(COLD) + ledger.active(HOLDER), 2);
    drop((first, second));
}

#[test]
fn first_token_does_not_hide_an_active_request_behind_a_stale_idle_report() {
    let host = Host::new(PROMPT / 2);
    host.share_prefix_with_cold();
    host.reports(0, [Some(2_000.0); 2], [100.0; 2]);
    let ledger = Arc::new(ReservationLedger::default());
    let mut first = host.select(&host.prompt(1_000_000), &ledger);
    first.1.dispatched();
    first.1.prefill_complete();
    let second = host.select(&host.prompt(2_000_000), &ledger);
    assert_ne!(first.0, second.0);
    drop((first, second));
}

#[test]
fn backend_waiting_work_can_outweigh_native_cache_even_without_local_inflight() {
    let host = Host::new(12_288);
    host.reports(100_000, [Some(2_000.0); 2], [100.0; 2]);
    assert_eq!(host.workers[1].load(), 0);
    let ledger = Arc::new(ReservationLedger::default());
    let pick = host.select(&host.prompt(1_000_000), &ledger);
    assert_eq!(pick.0, 0);
    assert_eq!(ledger.pending(COLD), PROMPT as u64);
}

#[test]
fn invalid_input_rates_use_nominal_capacity_and_never_generation_throughput() {
    for rate in [
        None,
        Some(f64::NAN),
        Some(0.0),
        Some(-1.0),
        Some(f64::INFINITY),
    ] {
        let host = Host::new(4_096);
        host.reports(1_000_000, [Some(1_500.0), rate], [0.001, 1e12]);
        let ledger = Arc::new(ReservationLedger::default());
        assert_eq!(
            host.select(&host.prompt(1_000_000), &ledger).0,
            0,
            "{rate:?}"
        );
    }
}

#[test]
fn measured_input_capacity_can_outweigh_a_larger_queue() {
    let host = Host::new(4_096);
    host.reports(
        1_000_000,
        [Some(1_500.0), Some(100_000_000.0)],
        [1e12, 0.001],
    );
    let ledger = Arc::new(ReservationLedger::default());
    let pick = host.select(&host.prompt(1_000_000), &ledger);
    assert_eq!(
        pick.0, 1,
        "input capacity, not inverse output rate, prices prefill"
    );
    assert_eq!(ledger.pending(HOLDER), 12_288);
}

#[test]
fn no_native_events_still_balances_and_books_full_prompt_work() {
    let host = Host::new(4_096);
    host.policy.set_kv_event_monitor(None);
    host.reports(0, [None; 2], [1e12, 0.001]);
    let ledger = Arc::new(ReservationLedger::default());
    let first = host.select(&host.prompt(1_000_000), &ledger);
    let second = host.select(&host.prompt(2_000_000), &ledger);
    assert_eq!((first.0, second.0), (0, 1));
    assert_eq!(ledger.pending(COLD), PROMPT as u64);
    assert_eq!(ledger.pending(HOLDER), PROMPT as u64);
}

#[test]
fn native_cache_namespace_cannot_borrow_another_salts_prefix() {
    let host = Host::new(12_288);
    host.reports(0, [Some(2_000.0); 2], [100.0; 2]);
    let tokens = host.prompt(1_000_000);
    let namespace = CacheNamespace::derive(&CachePartition {
        cache_salt: Some("tenant-a"),
        ..Default::default()
    })
    .unwrap();
    let ledger = Arc::new(ReservationLedger::default());
    assert_eq!(host.select(&tokens, &ledger).0, 1);
    let salted = host.select_using(
        &host.policy,
        &tokens,
        tokens.len(),
        Some(namespace),
        &ledger,
    );
    assert_eq!(salted.0, 0, "plain blocks do not credit a salted request");
    drop(salted);

    // Replace the event index with the engine-equivalent salted block hashes.
    let index = Arc::new(KvIndex::chain());
    let holder = index.intern_worker(HOLDER).unwrap();
    let blocks: Vec<_> =
        request_content_hashes_with_seed(&host.prefix, BLOCK, namespace.event_seed())
            .into_iter()
            .enumerate()
            .map(|(i, content_hash)| StoredBlock {
                seq_hash: SequenceHash(i as u64 + 1),
                content_hash,
            })
            .collect();
    index
        .apply_stored(holder, &blocks, None, &mut WorkerBlocks::default())
        .unwrap();
    host.monitor.indexers.insert(
        normalize_model_key(host.workers[0].model_id()).to_owned(),
        index,
    );
    let salted = host.select_using(
        &host.policy,
        &tokens,
        tokens.len(),
        Some(namespace),
        &ledger,
    );
    assert_eq!(salted.0, 1);
    assert_eq!(ledger.pending(HOLDER), 4_096);
}

#[test]
fn batch_tail_is_charged_even_when_the_first_prompt_is_almost_cached() {
    let host = Host::new(PROMPT - 64);
    host.reports(0, [Some(2_000.0); 2], [100.0; 2]);
    let ledger = Arc::new(ReservationLedger::default());
    let first = host.select_using(
        &host.policy,
        &host.prompt(1_000_000),
        PROMPT * 2,
        None,
        &ledger,
    );
    let second = host.select_using(
        &host.policy,
        &host.prompt(2_000_000),
        PROMPT * 2,
        None,
        &ledger,
    );
    assert_eq!((first.0, second.0), (1, 0));
    assert_eq!(ledger.pending(HOLDER), (PROMPT + 64) as u64);
    assert_eq!(ledger.pending(COLD), (PROMPT * 2) as u64);
}

#[test]
fn engine_block_size_overrides_router_fallback() {
    let mut host = Host::new(4_096);
    host.policy = PrefillCostPolicy::with_params(64, 0.15, 1_024, 2_000.0);
    host.policy
        .set_kv_event_monitor(Some(Arc::clone(&host.monitor)));
    let ledger = Arc::new(ReservationLedger::default());
    let pick = host.select(&host.prompt(1_000_000), &ledger);
    assert_eq!(pick.0, 1);
    assert_eq!(ledger.pending(HOLDER), 12_288);
}

#[test]
fn a_policy_requiring_lease_handoff_does_not_silently_preview_without_it() {
    let host = Host::new(4_096);
    assert_eq!(host.policy.name(), "prefill_cost");
    assert!(host.policy.needs_reservation());
    assert!(host.policy.needs_backend_loads());
    assert!(host.policy.needs_kv_events());
    assert_eq!(
        host.policy.select_worker(
            &host.workers,
            &SelectWorkerInfo {
                tokens: Some(&host.prompt(1_000_000)),
                ..Default::default()
            }
        ),
        None
    );
}

#[test]
fn dynamic_prefill_cost_hint_cannot_bypass_the_validated_gateway_config() {
    use crate::{
        config::PolicyConfig,
        policies::PolicyRegistry,
        routers::common::placement::{select_from, PlacementInputs},
        worker::test_support::registry_with,
    };

    let workers = make_workers(&[COLD, HOLDER]);
    let registry = registry_with(workers.clone());
    let model = workers[0].model_id();
    // A registration hint on an otherwise supported HTTP gateway must not
    // publish a policy that returns None on every HTTP request (no lease Slot).
    let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
    policies.on_worker_added(model, Some("prefill_cost"));
    assert_eq!(policies.get_policy(model).unwrap().name(), "round_robin");
    assert!(select_from(
        &registry,
        &policies,
        model,
        &workers,
        PlacementInputs::default(),
    )
    .is_some());

    // A matching hint still inherits the operator's explicitly configured type
    // and tunables, rather than manufacturing an unrelated default instance.
    let policies = PolicyRegistry::new(PolicyConfig::PrefillCost {
        block_size: BLOCK,
        kv_pressure_weight: 0.25,
        mean_prefill_tokens: 2_048,
        default_throughput: 8_000.0,
    });
    policies.on_worker_added(model, Some("prefill_cost"));
    let policy = policies.get_policy(model).unwrap();
    assert_eq!(policy.name(), "prefill_cost");
    assert_eq!(policy.reservation_active_tokens(), Some(2_048));
}

#[test]
fn actual_placement_all_vetoed_fallback_still_hands_off_a_full_batch_lease() {
    use crate::{
        config::PolicyConfig,
        policies::PolicyRegistry,
        routers::common::placement::{select_from, PlacementInputs},
        worker::test_support::registry_with,
    };

    let workers = make_workers(&[COLD, HOLDER]);
    let registry = registry_with(workers.clone());
    for worker in &workers {
        registry.set_worker_overloaded(worker, true);
    }
    // Default steering serves the least-loaded vetoed worker when shedding
    // is off. That route bypasses the successful policy result, but it must
    // still reach registry marking and conservative final-worker booking.
    let policies = PolicyRegistry::new(PolicyConfig::PrefillCost {
        block_size: BLOCK,
        kv_pressure_weight: 0.15,
        mean_prefill_tokens: 1_024,
        default_throughput: 2_000.0,
    });
    let slot = policies
        .reservation_slot()
        .with_total_tokens(PROMPT * 2)
        .with_request_count(2);
    let tokens = vec![1u32; PROMPT];
    let selected = select_from(
        &registry,
        &policies,
        workers[0].model_id(),
        &workers,
        PlacementInputs {
            tokens: Some(&tokens),
            reservation: Some(&slot),
            ..Default::default()
        },
    )
    .expect("all-overloaded pool is steered when shedding is off");
    let lease = slot
        .take_for(selected.url())
        .expect("fallback dispatch has a lease");
    assert_eq!(slot.ledger.active(selected.url()), 1);
    assert_eq!(slot.ledger.pending(selected.url()), (PROMPT * 2) as u64);
    drop(lease);
    assert_eq!(slot.ledger.active(selected.url()), 0);
}
