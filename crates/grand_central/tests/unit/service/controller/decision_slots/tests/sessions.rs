//! The cases of a session key at the service: a request a sessions rule
//! keeps or pins, a key's count under a cap, under no cap and under no
//! rule, and the upkeep of the session table.

use super::*;

/// The service tells no stage the issue of a request its sessions rule
/// keeps from the profile. A key's first request is placed by the
/// profile's picker, which hears its issue. The key's next request
/// follows its pin: under `untold` its lease is filed as kept, and the
/// picker hears nothing of it ahead of its release. Under `told`, the
/// default, the picker hears that issue as any other.
#[tokio::test]
async fn the_service_tells_no_stage_the_issue_of_a_request_its_rule_keeps() {
    use crate::{
        document::{Assignment, RulePlaced},
        service::controller::lifecycle::tests::{a_dispatched, a_released},
    };
    for kept in [RulePlaced::Untold, RulePlaced::Told] {
        let told = Arc::new(Mutex::new(Calls::new()));
        let hears = Arc::clone(&told);
        let mut catalog = StageRegistry::builtin();
        catalog
            .picker("told", move || Box::new(Told(Arc::clone(&hears))))
            .expect("a free name");
        // The manual preset's rule over a profile that picks through
        // the case's stage: a new key is the profile's to place.
        let mut document = PolicySpec::preset("manual").expect("a preset of this build");
        document.generation = 1;
        let profile = document.profiles.get_mut("manual");
        profile.expect("the preset's profile").pick = "told".into();
        let rule = document.bindings[0].sessions.as_mut();
        let rule = rule.expect("the preset's rule");
        (rule.assignment, rule.rule_placed) = (Assignment::Delegate, kept);
        let digest = document.digest().expect("the document digests");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
            .expect("the catalog holds the stage");
        let mut bed = bed(Some(spec), true, (1, 8), []);
        let at = bed.script.taken;
        let untold = |bed: &Bed, answer: &proto::DecideAnswer| {
            let id = LeaseId(lease_of(answer).id);
            let filed = LeasesStore::get(&*bed.store, GATEWAY, id);
            filed.expect("the lease is on file").untold
        };
        let first = bed.ask(keyed(1, 1_001)).await;
        assert!(!untold(&bed, &first), "{kept:?}: a new key");
        let hand = Hand::new(GATEWAY);
        lifecycle::dispatched(&bed.store, &hand, &a_dispatched(lease_of(&first).id), at)
            .expect("the frame reads");
        let pinned = bed.ask(keyed(1, 1_002)).await;
        assert_eq!(picked(&pinned), picked(&first), "{kept:?}: its pin");
        assert_eq!(untold(&bed, &pinned), kept == RulePlaced::Untold);
        let (first, pinned) = (lease_of(&first).id, lease_of(&pinned).id);
        lifecycle::released(&bed.store, &hand, &a_released(pinned)).expect("the frame reads");
        let mut heard = vec![(first, None)];
        if kept == RulePlaced::Told {
            heard.push((pinned, None));
        }
        heard.push((pinned, Some(ReleaseReason::ClientGone)));
        // The hand leaves the release to a decider still at work
        // behind its answer, which tells it as it leaves that work.
        settled(|| told.lock().len() >= heard.len()).await;
        assert_eq!(*told.lock(), heard, "{kept:?}");
    }
}

/// A pinned request is none of the requests its policy sent, through
/// the service as in a gateway's own process: the case of the same name
/// in `tests/classic/sticky_override.rs`, asked by frames, with its
/// counts. Two workers each hold one request, a lease on file, and the
/// second reports 5,000 tokens waiting. A key's first request of 1,000
/// tokens goes to the first worker, placed by the policy, and its next
/// request of 10,000 tokens follows the pin: its lease is filed as kept
/// from the profile. The policy then prices the first worker at the
/// 1,000 tokens it placed there and at two requests more on file, at
/// the mean prompt of 1,024 tokens: 3,048, under the second worker's
/// 6,024. So a request with no key goes to the first worker. Counted as
/// a send of the policy's, the pinned request would price that worker
/// at 12,024.
///
/// The pinned request still weighs what a lease on file weighs: with
/// 1,500 tokens waiting on the second worker, 2,524 in all, the request
/// with no key goes there. Once the pinned request ends it weighs
/// nothing: the first worker stands at 2,024, and the request with no
/// key goes back to it.
#[tokio::test]
async fn a_served_pinned_request_is_none_of_the_requests_its_policy_sent() {
    use crate::{
        document::{
            Assignment, LoadMetricSetting, PolicySettings, RoutingKeyOverrideSettings,
            RoutingSettings,
        },
        engine::sticky_sessions::SessionVerdict,
        service::controller::lifecycle::tests::{a_completed, a_dispatched, a_released},
        store::{LeaseRecord, PickRecord},
    };
    let policies = [
        PolicySettings::LeastLoad {
            kv_pressure_weight: 0.15,
            mean_prefill_tokens: 1024,
            default_throughput: 2000.0,
        },
        PolicySettings::PowerOfTwo {
            load_metric: LoadMetricSetting::ExpectedWait,
        },
    ];
    for policy in policies {
        let name = policy.name();
        // The settings' document for the policy under the sticky
        // override: its rule keeps what it places from the profile.
        let settings = RoutingSettings {
            policy,
            routing_key_override: RoutingKeyOverrideSettings {
                enabled: true,
                headers: vec!["x-smg-routing-key".into()],
                eviction_interval_secs: 60,
                max_idle_secs: 14_400,
                assignment_mode: Assignment::Delegate,
            },
            ..RoutingSettings::default()
        };
        let mut document = settings.policy_document().expect("the settings' document");
        document.generation = 1;
        let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
        let at = bed.script.taken;
        let (near, far, aside) = (bed.workers[0], bed.workers[1], bed.workers[2]);
        // Two workers take requests: the third is set not ready.
        stand(
            &bed.reads.objects,
            &bed.store,
            aside,
            WorkerStatus::NotReady,
        );
        // One request on each of the two, held by a lease on file that
        // no decision of the policy's filed.
        for (id, target) in [(1, near), (2, far)] {
            let held = LeaseRecord {
                lease: Lease {
                    id: LeaseId(1 << 63 | id),
                    ..Lease::default()
                },
                gateway: GATEWAY,
                request: id,
                target,
                picked: PickRecord::default(),
                decode: None,
                untold: false,
                term: 1,
                issued_at: at,
                dispatch_deadline: at + DISPATCH_TTL,
                state: LeaseState::ISSUED,
            };
            assert_eq!(LeasesStore::issue(&*bed.store, held), Ok(()));
        }
        // The two workers' reports, the second with `waiting` tokens in
        // its queue, sampled ahead of every send: none reflects one.
        let reports = |waiting: i32| {
            let loads = [("w0", 0), ("w1", waiting)].map(|(worker, waiting)| {
                let mut report = one_rank_load(waiting, 0.1, 1_000.0);
                report.sampled_at = Some(at - Duration::from_millis(1));
                (format!("http://{worker}:8000"), report)
            });
            LoadSnapshot::from_loads_for_test(loads.into_iter().collect())
        };
        let sized = |mut frame: proto::DecideRequest, tokens: u32| {
            let facts = frame.facts.as_mut().expect("the case's frame has facts");
            facts.prompt_tokens = tokens;
            frame
        };
        let filed = |bed: &Bed, answer: &proto::DecideAnswer| {
            let id = LeaseId(lease_of(answer).id);
            let filed = LeasesStore::get(&*bed.store, GATEWAY, id);
            let filed = filed.expect("the lease is on file");
            (filed.lease.session_verdict, filed.untold)
        };
        let hand = Hand::new(GATEWAY);
        let sent = |bed: &Bed, answer: &proto::DecideAnswer| {
            let frame = a_dispatched(lease_of(answer).id);
            lifecycle::dispatched(&bed.store, &hand, &frame, at).expect("the frame reads");
        };
        let given_up = |bed: &Bed, answer: &proto::DecideAnswer| {
            let frame = a_released(lease_of(answer).id);
            lifecycle::released(&bed.store, &hand, &frame).expect("the frame reads");
        };
        bed.loads.send_replace(reports(5_000));
        let first = bed.ask(sized(keyed(1, 1_001), 1_000)).await;
        assert_eq!(picked(&first), near, "{name}: a new key");
        assert_eq!(filed(&bed, &first), (SessionVerdict::Claimed(near), false));
        sent(&bed, &first);
        let pinned = bed.ask(sized(keyed(1, 1_002), 10_000)).await;
        assert_eq!(picked(&pinned), near, "{name}: its pin");
        assert_eq!(filed(&bed, &pinned), (SessionVerdict::Pinned, true));
        sent(&bed, &pinned);
        let plain = bed.ask(sized(frame(3, 1_003), 1_000)).await;
        assert_eq!(
            picked(&plain),
            near,
            "{name}: the pinned request is no send of the policy's"
        );
        given_up(&bed, &plain);
        bed.loads.send_replace(reports(1_500));
        let plain = bed.ask(sized(frame(4, 1_004), 1_000)).await;
        assert_eq!(
            picked(&plain),
            far,
            "{name}: the pinned request weighs as one on file"
        );
        given_up(&bed, &plain);
        let ended = a_completed(lease_of(&pinned).id, near);
        lifecycle::completed(&bed.store, &hand, &ended).expect("the frame reads");
        let plain = bed.ask(sized(frame(5, 1_005), 1_000)).await;
        assert_eq!(
            picked(&plain),
            near,
            "{name}: an ended pinned request weighs nothing"
        );
    }
}

/// The upkeep of a served engine rides the presence sweep's task: that
/// engine runs no task of its own. A session key a decision read leaves
/// the engine's table by the task's own tick once its idle time ended,
/// and no sooner, with no request behind the one that read it.
#[tokio::test]
async fn the_sweeps_task_takes_a_session_key_out_once_its_idle_time_ended() {
    let mut document = PolicySpec::preset("manual").expect("a preset of this build");
    document.generation = 1;
    let rule = document.bindings[0].sessions.as_mut();
    let rule = rule.expect("the preset's rule");
    (rule.ttl_s, rule.sweep_s) = (1, 1);
    let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
    let keys = |bed: &Bed| {
        bed.store
            .deciding(|world| world.spec().engine().sessions().len())
    };
    bed.ask(keyed(1, 1_001)).await;
    assert_eq!(keys(&bed), 1, "the key its decision read");

    let grace = Duration::from_millis(60);
    let fence = bed.elector.fence();
    let gateways = Arc::new(Gateways::new(grace, 1_024, fence, Arc::clone(&bed.store)));
    let (stop, stopped) = watch::channel(false);
    let mut task = JoinSet::new();
    task.spawn(async move { gateways.run(stopped).await });
    // The decision is dated at the bed's take.
    let read = bed.script.taken;
    while keys(&bed) > 0 {
        assert!(read.elapsed() < PATIENCE, "the task took no key out");
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert!(
        read.elapsed() >= Duration::from_secs(1),
        "the key left inside its idle time"
    );
    stop.send_replace(true);
    timeout(PATIENCE, task.join_next())
        .await
        .expect("the task ends at its stop")
        .expect("one task")
        .expect("it ended by itself");
}

/// A served engine's session table hears the store's word of a worker
/// that leaves and of its return. The worker a key is pinned on is
/// removed through the store: the table lists it absent. Registered
/// again under its id, it is listed no more.
#[tokio::test]
async fn the_session_table_lists_a_worker_the_store_removed_until_it_returns() {
    let mut document = PolicySpec::preset("manual").expect("a preset of this build");
    document.generation = 1;
    let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
    let absent = |bed: &Bed| {
        bed.store
            .deciding(|world| world.spec().engine().sessions().absent())
    };
    let pinned = picked(&bed.ask(keyed(1, 1_001)).await);
    assert_eq!(absent(&bed), 0, "every worker stands");
    let row = WorkersStore::get(&*bed.store, pinned).expect("the row");
    let objects = &bed.reads.objects;
    let removed = objects.remove(&bed.store, pinned, row.revision);
    assert!(removed.is_some(), "the store took the worker out");
    assert_eq!(absent(&bed), 1, "the worker its key is pinned on left");
    assert_eq!(rewritten(&bed, row.facts), pinned, "its id again");
    assert_eq!(absent(&bed), 0, "back under its id");
}

/// The bed's frame under a session key in the clear.
fn keyed_in_the_clear(key: &str, seed: u64) -> proto::DecideRequest {
    let mut frame = frame(u8::try_from(seed % 251).expect("under 251"), seed);
    frame
        .facts
        .as_mut()
        .expect("the case's frame has facts")
        .session_key = Some(proto::Identity {
        form: Some(proto::identity::Form::Plain(key.to_owned())),
    });
    frame
}

/// A keyed session keeps its worker across an apply of a new
/// generation. The store's engine keeps one session table for its
/// life, and each generation's stages read it
/// (`AppliedSpec::share_sessions`): a decision of a dispatched key is
/// made on its pin under either generation.
#[tokio::test]
async fn a_keyed_session_keeps_its_worker_across_an_apply() {
    use crate::{
        engine::sticky_sessions::SessionVerdict,
        service::controller::lifecycle::{self, tests::a_dispatched},
    };
    const KEYS: u64 = 24;
    let mut bed = bed(Some(preset("manual", 1)), true, (1, 8), []);
    let at = bed.script.taken;
    let verdict = |bed: &Bed, answer: &proto::DecideAnswer| {
        LeasesStore::get(&*bed.store, GATEWAY, LeaseId(lease_of(answer).id))
            .expect("the lease is on file")
            .lease
            .session_verdict
    };
    // The key's value as its decision made it, read off the lease.
    let session = |bed: &Bed, answer: &proto::DecideAnswer| {
        LeasesStore::get(&*bed.store, GATEWAY, LeaseId(lease_of(answer).id))
            .expect("the lease is on file")
            .lease
            .session
    };
    let pins = |bed: &Bed, session: view::Session| {
        bed.store
            .deciding(|world| world.spec().engine().sessions().pins(session))
    };
    // Each key's first decision claims a worker, and its gateway's
    // Dispatched frame commits it: the engine's pin.
    let mut first = Vec::new();
    for key in 1..=KEYS {
        let answer = bed.ask(keyed(key, 1_000 + key)).await;
        let worker = picked(&answer);
        assert_eq!(verdict(&bed, &answer), SessionVerdict::Claimed(worker));
        let hand = Hand::new(GATEWAY);
        lifecycle::dispatched(&bed.store, &hand, &a_dispatched(lease_of(&answer).id), at)
            .expect("the frame reads");
        first.push((key, worker, session(&bed, &answer)));
    }
    // The keys stand on more than one worker: a bed that kept every
    // key on one would hold nothing.
    let stood = bed
        .workers
        .iter()
        .filter(|member| first.iter().any(|&(_, worker, _)| worker == **member))
        .count();
    assert!(stood > 1, "the keys stand on {stood} worker");
    for &(key, worker, _) in &first {
        let answer = bed.ask(keyed(key, 2_000 + key)).await;
        assert_eq!(picked(&answer), worker);
        assert_eq!(verdict(&bed, &answer), SessionVerdict::Pinned);
    }
    // A new generation: the pins of the one before stand.
    let held = pins(&bed, first[0].2);
    assert!(held.is_some());
    bed.store
        .apply(preset("manual", 2))
        .expect("a later generation");
    assert_eq!(pins(&bed, first[0].2), held);
    for &(key, worker, session) in &first {
        let answer = bed.ask(keyed(key, 3_000 + key)).await;
        assert_eq!(
            picked(&answer),
            worker,
            "key {key} left its worker at the apply"
        );
        assert_eq!(verdict(&bed, &answer), SessionVerdict::Pinned);
        assert_eq!(pins(&bed, session), Some([worker, WorkerId(0)]));
    }
}

/// The controller's own service counts a key's requests. Under a rule
/// with an in-flight cap the request that finds the cap met on its
/// key's worker is placed again and lands on a worker with fewer
/// requests. The pin moves there where that request is decided, ahead
/// of its gateway's Dispatched frame, so the key's next request follows
/// it, and the one after stays once one of the two ended. The frames
/// carry the key as its hash.
#[tokio::test]
async fn the_service_hands_a_key_on_at_its_rules_cap() {
    handed_on_at_the_cap(|seed| keyed(1, seed)).await;
}

/// The same of a key in the clear, the form a gateway sends a key it
/// holds as text in. The lease carries the session's number alone, so
/// the count stands under that number whichever form the frame took.
#[tokio::test]
async fn the_service_hands_a_key_in_the_clear_on_at_its_rules_cap() {
    handed_on_at_the_cap(|seed| keyed_in_the_clear("talk-1", seed)).await;
}

/// One key's five requests under a rule whose in-flight cap is two,
/// each made by `frame` from its seed: the first claims a worker, the
/// second stays, the third is placed again and moves the pin, the
/// fourth follows the move, and the fifth stays behind an end.
async fn handed_on_at_the_cap(frame: impl Fn(u64) -> proto::DecideRequest) {
    use crate::{
        engine::sticky_sessions::SessionVerdict,
        service::controller::lifecycle::{self, tests::a_dispatched},
    };
    let mut document = PolicySpec::preset("manual").expect("a preset of this build");
    let rule = document.bindings[0].sessions.as_mut();
    let rule = rule.expect("the preset's rule");
    rule.inflight_cap = 2;
    // The rule places a key on the worker with the fewest requests: a
    // request placed again at the cap leaves the key's worker.
    rule.assignment = crate::document::Assignment::MinLoad;
    let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
    let at = bed.script.taken;
    let verdict = |bed: &Bed, answer: &proto::DecideAnswer| {
        LeasesStore::get(&*bed.store, GATEWAY, LeaseId(lease_of(answer).id))
            .expect("the lease is on file")
            .lease
            .session_verdict
    };
    // The key's first request claims a worker, and its gateway's
    // Dispatched frame pins the key there.
    let first = bed.ask(frame(1_001)).await;
    let owner = picked(&first);
    assert_eq!(verdict(&bed, &first), SessionVerdict::Claimed(owner));
    let hand = Hand::new(GATEWAY);
    lifecycle::dispatched(&bed.store, &hand, &a_dispatched(lease_of(&first).id), at)
        .expect("the frame reads");
    // One request of the key in flight: the second stays.
    let second = bed.ask(frame(1_002)).await;
    assert_eq!(picked(&second), owner);
    assert_eq!(verdict(&bed, &second), SessionVerdict::Pinned);
    // Two in flight on the key's worker: the third is placed again,
    // on a worker with fewer requests. Its decision moves the pin
    // there, and no Dispatched frame has come for it.
    let third = bed.ask(frame(1_003)).await;
    let moved = picked(&third);
    assert_ne!(moved, owner);
    assert_eq!(verdict(&bed, &third), SessionVerdict::Respill);
    // The fourth follows the move, as a gateway's sticky map sends it:
    // that worker carries two of the key's requests now.
    let fourth = bed.ask(frame(1_004)).await;
    assert_eq!(picked(&fourth), moved);
    assert_eq!(verdict(&bed, &fourth), SessionVerdict::Pinned);
    // One of the two ended: the key's next request is under the cap.
    let ended = LeaseId(lease_of(&third).id);
    let gone = ReleaseReason::ClientGone;
    assert!(LeasesStore::released(
        &*bed.store,
        GATEWAY,
        ended,
        None,
        gone
    ));
    let fifth = bed.ask(frame(1_005)).await;
    assert_eq!(picked(&fifth), moved);
    assert_eq!(verdict(&bed, &fifth), SessionVerdict::Pinned);
}

/// The keys the bed's worker objects count, over all of them: a key
/// counts once on each worker that carries a request of it.
fn keys_counted(bed: &Bed) -> usize {
    let objects = bed.reads.objects.registry().get_all();
    let keys = objects.iter().map(|object| object.routing_key_load());
    keys.sum()
}

/// The service counts a keyed single request for its key under any
/// sessions rule and under none: its lease counts on the worker object
/// it books, as a gateway's load guard counts every request that
/// carries a key (`crates/worker/src/load/inflight.rs:196-199` at main
/// `5e609dc2d`). The stock rule's cap is disabled, and the `random`
/// preset has no sessions rule. A request with no key counts for none.
#[tokio::test]
async fn the_service_counts_a_keyed_request_under_any_rule_and_under_none() {
    let capped = {
        let mut document = PolicySpec::preset("manual").expect("a preset of this build");
        let rule = document.bindings[0].sessions.as_mut();
        rule.expect("the preset's rule").inflight_cap = 2;
        applied(&document)
    };
    let ruleless = PolicySpec::preset("random").expect("a preset of this build");
    assert!(ruleless
        .bindings
        .iter()
        .all(|binding| binding.sessions.is_none()));
    let documents = [
        ("a rule with no cap", preset("manual", 1)),
        ("a rule with a cap", capped),
        ("no rule", applied(&ruleless)),
    ];
    for (under, document) in documents {
        let mut bed = bed(Some(document), true, (1, 8), []);
        bed.ask(frame(1, 1_000)).await;
        assert_eq!(keys_counted(&bed), 0, "{under}: a request with no key");
        bed.ask(keyed(1, 1_001)).await;
        assert_eq!(keys_counted(&bed), 1, "{under}: a keyed request");
    }
}

/// A rule that places a new key on the worker with the fewest keys
/// reads the keys of the leases the service holds, under no cap. One
/// worker carries three keys' requests and the others none: no new key
/// takes it. Each new key's count leaves at its lease's release, so
/// the others stay at none. The gateway's classic sticky policy
/// answers the same: a new key takes a worker with the fewest keys in
/// flight (`model_gateway/src/policies/manual.rs:479-480` at main
/// `5e609dc2d`), which its load guard counts for every request that
/// carries a key (`crates/worker/src/load/inflight.rs:196-199`).
#[tokio::test]
async fn the_service_places_a_new_key_by_the_keys_its_leases_count_under_no_cap() {
    let mut document = PolicySpec::preset("manual").expect("a preset of this build");
    let rule = document.bindings[0].sessions.as_mut();
    rule.expect("the preset's rule").assignment = crate::document::Assignment::MinGroup;
    let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
    let (busy, others) = (bed.workers[0], [bed.workers[1], bed.workers[2]]);
    // Three keys' requests on one worker, the only one ready.
    for other in others {
        set_status(&bed, other, WorkerStatus::NotReady);
    }
    for key in 1..=3 {
        let answer = bed.ask(keyed(key, 1_000 + key)).await;
        assert_eq!(picked(&answer), busy);
    }
    for other in others {
        set_status(&bed, other, WorkerStatus::Ready);
    }
    // Each new key takes a worker that carries no key, and its request
    // ends before the next key's.
    for key in 4..=27 {
        let answer = bed.ask(keyed(key, 1_000 + key)).await;
        assert_ne!(picked(&answer), busy, "key {key}");
        let ended = LeaseId(lease_of(&answer).id);
        let gone = ReleaseReason::ClientGone;
        let store: &dyn LeasesStore = bed.store.as_ref();
        assert!(store.released(GATEWAY, ended, None, gone));
    }
    // The three keys of the first worker alone: each new key's count
    // left with its lease.
    assert_eq!(keys_counted(&bed), 3);
}

/// Under a rule with no cap a key's count leaves its worker's object
/// with the key's lease, at a release.
#[tokio::test]
async fn a_key_under_a_rule_with_no_cap_is_counted_until_its_lease_is_released() {
    let mut bed = bed(Some(preset("manual", 1)), true, (1, 8), []);
    let answer = bed.ask(keyed(1, 1_001)).await;
    assert_eq!(keys_counted(&bed), 1);
    let ended = LeaseId(lease_of(&answer).id);
    let store: &dyn LeasesStore = bed.store.as_ref();
    assert!(store.released(GATEWAY, ended, None, ReleaseReason::ClientGone));
    assert_eq!(keys_counted(&bed), 0);
}

/// Under a rule with no cap a key's count leaves its worker's object
/// where no dispatch names its lease: a decision past the lease's
/// dispatch deadline ends it as abandoned.
#[tokio::test]
async fn a_key_under_a_rule_with_no_cap_is_counted_until_its_dispatch_deadline() {
    let mut bed = bed(Some(preset("manual", 1)), true, (1, 8), []);
    bed.ask(keyed(1, 1_001)).await;
    assert_eq!(keys_counted(&bed), 1);
    // A request with no key, decided past that deadline.
    let late = DISPATCH_TTL + Duration::from_secs(1);
    bed.script.steps.lock().push_back(Step::At(late));
    assert_eq!(bed.submit_at(frame(2, 1_002), GATEWAY, late), None);
    bed.answer().await;
    let store: &dyn LeasesStore = bed.store.as_ref();
    assert_eq!(store.live(GATEWAY), 1, "the late request's lease alone");
    assert_eq!(keys_counted(&bed), 0);
}

/// A cap that a new document sets counts the key's requests that stood
/// before it. Two requests of a key stand on its worker, decided under
/// a rule with no cap, and a third is decided and dispatched there
/// under the document that sets a cap of two. The key's next request
/// finds three on its pin and is placed again, on another worker.
#[tokio::test]
async fn a_cap_a_new_document_sets_counts_the_requests_that_stood_before_it() {
    use crate::{
        engine::sticky_sessions::SessionVerdict,
        service::controller::lifecycle::{self, tests::a_dispatched},
    };
    let ruled = |cap: Option<u16>, generation: u64| {
        let mut document = PolicySpec::preset("manual").expect("a preset of this build");
        document.generation = generation;
        let rule = document.bindings[0].sessions.as_mut();
        let rule = rule.expect("the preset's rule");
        // A request placed again at the cap leaves the key's worker.
        rule.assignment = crate::document::Assignment::MinLoad;
        if let Some(cap) = cap {
            rule.inflight_cap = cap;
        }
        applied(&document)
    };
    let mut bed = bed(Some(ruled(None, 1)), true, (1, 8), []);
    let at = bed.script.taken;
    let (owner, others) = (bed.workers[0], [bed.workers[1], bed.workers[2]]);
    let dispatched = |bed: &Bed, answer: &proto::DecideAnswer| {
        let (hand, frame) = (Hand::new(GATEWAY), a_dispatched(lease_of(answer).id));
        lifecycle::dispatched(&bed.store, &hand, &frame, at).expect("the frame reads");
    };
    // The key's worker is the only one ready until its last request.
    for other in others {
        set_status(&bed, other, WorkerStatus::NotReady);
    }
    // Two requests of the key under the rule with no cap.
    for seed in [1_001, 1_002] {
        let answer = bed.ask(keyed(1, seed)).await;
        assert_eq!(picked(&answer), owner);
        dispatched(&bed, &answer);
    }
    // The cap, and the key's third request under it: its dispatch pins
    // the key on its worker under the new document.
    bed.store
        .apply(ruled(Some(2), 2))
        .expect("a later generation");
    let third = bed.ask(keyed(1, 1_003)).await;
    assert_eq!(picked(&third), owner);
    dispatched(&bed, &third);
    for other in others {
        set_status(&bed, other, WorkerStatus::Ready);
    }
    // Three of the key's requests on its pin, two from before the cap.
    let fourth = bed.ask(keyed(1, 1_004)).await;
    assert_ne!(picked(&fourth), owner);
    let filed = LeasesStore::get(&*bed.store, GATEWAY, LeaseId(lease_of(&fourth).id));
    let verdict = filed.expect("the lease is on file").lease.session_verdict;
    assert_eq!(verdict, SessionVerdict::Respill);
}
