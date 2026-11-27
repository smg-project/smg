//! The lifecycle: the events a plugin hears, once, by its row's kinds and
//! under its exposure, and the membership feed an instance hears before it
//! decides.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use smg_grand_central::{
    engine::{
        Constraints, Decision, DecisionSlot, Engine, Identity, RequestFacts, Scratch, Target,
        WorkerId,
    },
    plugins::{
        shared::{BreakerSpec, OpenDecisions, Reveal},
        wasm::{Plugin, PluginSpec, Runtime},
    },
    stages::Lifecycle,
    tracking::{EventKind, EventSet, Lease, Side},
};

use crate::{
    bound_clock, bound_profile, counter_value, decide, facts, four, load, loaded, made_again,
    off_cpu_since, pair_decide, pair_engine, pick_ids, plugin_engine, plugin_profile, pool,
    profile_that, runtime, spec,
    support::wasm_guest::{self as guest, counter, mode},
    tried, warmed, Inside, FRAME_MICROS, INIT_FRAME, TRIES,
};

#[test]
fn the_lifecycle_reaches_the_plugin_once_per_profile_on_the_hosts_clock() {
    let runtime =
        Arc::new(Runtime::with_clock(&OpenDecisions::new(), || 424_242).expect("a runtime"));
    let plugin = load(
        &runtime,
        "observer",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let engine = plugin_engine(&plugin);
    let lease = Lease::default();
    engine.on_dispatched(&lease, &Target::default(), Side::Single, 3);
    assert_eq!(
        counter_value(&plugin, counter::OBSERVED),
        1,
        "three adapters of one profile, one observe"
    );
    assert_eq!(counter_value(&plugin, counter::OBSERVE_NOW), 424_242);
    engine.on_released(
        &lease,
        smg_grand_central::tracking::ReleaseReason::ClientGone,
    );
    engine.on_membership(&smg_grand_central::tracking::Membership::Snapshot(&[
        WorkerId(1),
        WorkerId(2),
    ]));
    engine.on_signal(&smg_grand_central::tracking::Signal::gateway_veto(
        WorkerId(1),
        500_000,
    ));
    assert_eq!(counter_value(&plugin, counter::OBSERVED), 4);
    assert_eq!(plugin.stats().observed, 4);
}

/// The exposure gates the lifecycle as it gates the request: a withheld
/// session key reads zero in every lease the plugin observes.
#[test]
fn a_withheld_session_key_is_withheld_from_the_lifecycle_too() {
    let runtime = runtime();
    for (name, component) in [
        ("record", guest::record_guest()),
        ("packed", guest::packed_guest()),
    ] {
        let mut withheld = spec(name, component.clone(), mode::LEAST_IN_FLIGHT);
        withheld.view.session_key = Reveal::None;
        let withheld = loaded(&runtime, withheld);
        let hashed = load(&runtime, name, component, mode::LEAST_IN_FLIGHT);
        // A lease carries a request's session, which the request's facts
        // alone make.
        let mut facts = RequestFacts::default();
        facts.model = "m";
        facts.session_key = Identity::hashed(0xBEEF);
        let session = facts.session();
        let mut lease = Lease::default();
        lease.session = session;
        for plugin in [&withheld, &hashed] {
            plugin_engine(plugin).on_dispatched(&lease, &Target::default(), Side::Single, 3);
        }
        assert_eq!(
            counter_value(&withheld, counter::OBSERVE_SESSION),
            0,
            "{name}: withheld"
        );
        assert_eq!(
            counter_value(&hashed, counter::OBSERVE_SESSION),
            session.get(),
            "{name}: the session's value, as in the request"
        );
    }
}

/// A plugin bound to two profiles of one build observes each lifecycle
/// event once: the engine broadcasts to every profile, and the bound sets
/// the build took from its one binder share one observer, the first adapter
/// made.
#[test]
fn a_plugin_bound_to_two_profiles_observes_each_event_once() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "twice-bound",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let binder = plugin.binder();
    let prefill = bound_profile(binder.bind(), None, None, None);
    let decode = bound_profile(binder.bind(), None, None, None);
    let engine = pair_engine(vec![prefill, decode], 1);
    engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    assert_eq!(plugin.stats().observed, 1);
    assert_eq!(counter_value(&plugin, counter::OBSERVED), 1);
}

/// A loaded plugin outlives the engine that bound it first. The engine
/// built to replace it, while the first still stands, forwards the
/// lifecycle for itself from its first event: each build has its own
/// observer, so an event reaches the plugin once per engine that hears it,
/// while the first engine stands and once it is gone.
#[test]
fn a_plugin_bound_into_a_later_engine_observes_through_it() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "rebound",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let dispatch = |engine: &Engine| {
        engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
        plugin.stats().observed
    };
    let first = plugin_engine(&plugin);
    assert_eq!(dispatch(&first), 1);
    let second = plugin_engine(&plugin);
    assert_eq!(
        dispatch(&second),
        2,
        "the later engine forwards beside the first"
    );
    assert_eq!(dispatch(&first), 3, "and the first still does");
    drop(first);
    assert_eq!(dispatch(&second), 4, "and once the first is gone");
    assert_eq!(counter_value(&plugin, counter::OBSERVED), 4);
}

/// A row lists the kinds of event its module hears, none by default. A
/// module whose row lists none is never told: no call is made, and a thread
/// that only tells makes no instance. One whose row lists some is told
/// those kinds alone. A loaded module says the kinds its row lists.
#[test]
fn a_module_hears_the_kinds_of_event_its_row_lists_and_no_other() {
    let tell = |engine: &Engine| {
        let lease = Lease::default();
        engine.on_dispatched(&lease, &Target::default(), Side::Single, 3);
        engine.on_released(
            &lease,
            smg_grand_central::tracking::ReleaseReason::ClientGone,
        );
        engine.on_membership(&smg_grand_central::tracking::Membership::Snapshot(&[
            WorkerId(1),
            WorkerId(2),
        ]));
        engine.on_signal(&smg_grand_central::tracking::Signal::gateway_veto(
            WorkerId(1),
            500_000,
        ));
    };
    let runtime = runtime();
    // The default row lists nothing.
    let mut unlisted = spec("unlisted", guest::record_guest(), mode::LEAST_IN_FLIGHT);
    unlisted.observes = PluginSpec::new("default", Vec::new()).observes;
    assert_eq!(unlisted.observes, EventSet::NONE);
    let plugin = loaded(&runtime, unlisted);
    assert_eq!(plugin.observes(), EventSet::NONE);
    let engine = plugin_engine(&plugin);
    assert!(!engine.needs_lifecycle());
    // Told on a thread that never decided, then on the loading thread.
    std::thread::scope(|scope| {
        scope
            .spawn(|| tell(&engine))
            .join()
            .expect("the telling thread");
    });
    tell(&engine);
    let stats = plugin.stats();
    assert_eq!(
        (stats.observed, stats.lazy_instantiations),
        (0, 0),
        "nothing told, and no instance made by a thread that only tells: {stats:?}"
    );
    assert_eq!(counter_value(&plugin, counter::OBSERVED), 0);

    let mut listed = spec("listed", guest::record_guest(), mode::LEAST_IN_FLIGHT);
    listed.observes = EventSet::of([EventKind::Released, EventKind::Signal]);
    let plugin = loaded(&runtime, listed);
    assert_eq!(
        plugin.observes(),
        EventSet::of([EventKind::Released, EventKind::Signal])
    );
    let engine = plugin_engine(&plugin);
    assert!(engine.needs_lifecycle());
    tell(&engine);
    assert_eq!(plugin.stats().observed, 2);
    assert_eq!(counter_value(&plugin, counter::OBSERVED), 2);
}

/// A lifecycle event's trap, outside any decision, drops the instance; the
/// next decision on the thread arrives with its budget gone and touches no
/// slot, so its close rebuilds nothing: the slot stays cold for the next
/// event or open to make lazily.
#[test]
fn an_observe_fault_leaves_no_mark_for_a_decision_that_never_touched_the_slot() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "observe-trap",
        guest::record_guest(),
        mode::TRAP_IN_OBSERVE,
    );
    let engine = plugin_engine(&plugin);
    let decide = |budget_micros: u32, seed: u64| {
        let mut request_facts = facts(seed);
        request_facts.budget_micros = budget_micros;
        engine.decide(
            &request_facts,
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        )
    };
    assert!(matches!(decide(FRAME_MICROS, 1), Decision::Pick(_)));
    assert!(plugin.is_warm(), "a decision ran on the kept instance");
    engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    let s = plugin.stats();
    assert_eq!(s.traps, 1, "the event's trap: {s:?}");
    assert!(!plugin.is_warm(), "the trap dropped the instance");
    assert!(
        matches!(decide(0, 2), Decision::Pick(_)),
        "the fallbacks decide"
    );
    let s = plugin.stats();
    assert!(
        !plugin.is_warm() && s.instantiations == 1 && s.budget_gone == 1,
        "a decision that touched no slot rebuilt none: {s:?}"
    );
}

/// The membership is fed to every instance of a plugin whose row lists it.
/// A change is told once, on a thread that only tells, and makes no guest
/// call there and no instance; each decider's instance hears it before its
/// next decision, one call a change. An instance made after a telling hears
/// the list as it stands first.
///
/// The second decider's instance is made by its first decision: a lazy
/// `init`, on the wall clock. A runner that holds that thread for the init
/// frame has the host end the `init`. The decision then falls back, to the
/// very worker the plugin names there, and the instance the decider's next
/// decision makes hears the changes told so far as one snapshot: one call
/// fewer, on one instance more. A destructor's time costs an instance the
/// same way, and the close rebuilds it. So a try that counts another
/// instance than the two the case makes, where a decision's thread was off
/// the CPU half the init frame, says so and is made again ([`tried`]). Each
/// try reads its picks and its counts before it asserts any, and a try no
/// hold reached asserts what it read.
#[test]
fn a_membership_told_once_reaches_every_deciders_instance_before_it_decides() {
    use smg_grand_central::tracking::Membership;
    const CASE: &str = "a_membership_told_once_reaches_every_deciders_instance_before_it_decides";
    let runtime = runtime();
    tried(|made| {
        let plugin = load(&runtime, "fed", guest::record_guest(), mode::COUNT_MEMBERS);
        let engine = plugin_engine(&plugin);
        // The guest's pick names the row at the count of members its
        // instance heard: on a thread of its own, acting as the decider of
        // that number. The longest any such thread was off the CPU is kept.
        let worst = std::cell::Cell::new(std::time::Duration::ZERO);
        let acting = |number: u8| {
            let (pick, held) = std::thread::scope(|scope| {
                scope
                    .spawn(|| {
                        let (wall, cpu) = (std::time::Instant::now(), bound_clock());
                        let pick = {
                            let _turn = DecisionSlot::new(number).enter();
                            pick_ids(&decide(&engine, &pool(4), 1)).first().copied()
                        };
                        (pick, off_cpu_since(wall, cpu))
                    })
                    .join()
                    .expect("the acting thread")
            });
            worst.set(worst.get().max(held));
            pick
        };
        // Told on a thread that names no decider and never decides.
        let tell = |change: Membership<'_>| {
            std::thread::scope(|scope| {
                scope
                    .spawn(|| engine.on_membership(&change))
                    .join()
                    .expect("the telling thread");
            });
        };
        let untold = acting(0);
        tell(Membership::Snapshot(&[WorkerId(100), WorkerId(101)]));
        let told = plugin.stats();
        let heard = acting(0);
        // The second decider's first decision, and the instances made
        // lazily by its end: its own.
        let late = (acting(1), plugin.stats().lazy_instantiations);
        tell(Membership::Added(WorkerId(102)));
        let added = (acting(1), acting(0));
        tell(Membership::Removed(WorkerId(100)));
        tell(Membership::Removed(WorkerId(101)));
        let removed = (acting(0), acting(1));
        let stats = plugin.stats();
        // The witness: the load's instance and the second decider's are the
        // two a try makes. Another one, made or only begun, in a try where
        // a decision's thread was off the CPU half the init frame, is the
        // host's answer to that hold. With no such hold it is the host's
        // own, and the try asserts.
        let another = (stats.instantiations, stats.lazy_instantiations) != (2, 1);
        if another && worst.get() >= INIT_FRAME / 2 && made < TRIES {
            made_again(
                CASE,
                "a decision's thread was off the CPU half the init frame, the bound of a lazy init and of a close's rebuild",
            );
            return false;
        }
        assert_eq!(untold, Some(100), "no member told yet");
        assert_eq!(
            (told.observed, told.instantiations, told.events_dropped),
            (0, 1, 0),
            "a telling calls no guest, makes no instance and drops nothing: {told:?}"
        );
        assert_eq!(heard, Some(102), "the first decider's instance heard");
        assert_eq!(
            late,
            (Some(102), 1),
            "an instance made after the telling, by its decider's first decision, hears the list as it stands"
        );
        assert_eq!(added, (Some(103), Some(103)));
        assert_eq!(removed, (Some(101), Some(101)));
        assert_eq!(
            (
                stats.observed,
                stats.instantiations,
                stats.lazy_instantiations
            ),
            (8, 2, 1),
            "one call a decider a change, the snapshot among them, on the two instances: {stats:?}"
        );
        true
    });
}

/// An instance new to the feed hears the membership as it stands, as one
/// snapshot, whatever the host told: the first instance where only changes
/// were told, an instance remade after a fault, which then counts what one
/// that heard every change counts, and an instance made at the warm-up,
/// which hears there and not at its first decision. A change told before a
/// lease's event is heard before it.
#[test]
fn an_instance_made_anew_hears_the_membership_as_it_stands() {
    use smg_grand_central::tracking::Membership;
    let runtime = runtime();
    let plugin = load(&runtime, "anew", guest::record_guest(), mode::COUNT_MEMBERS);
    let engine = plugin_engine(&plugin);
    let pick = || pick_ids(&decide(&engine, &pool(4), 1)).first().copied();
    // Five workers come, one change each, and no snapshot was ever told: an
    // instance new to the feed still hears them as one snapshot.
    for id in 100..105 {
        engine.on_membership(&Membership::Added(WorkerId(id)));
    }
    // Five members over four rows: the guest's pick names no row, an answer
    // outside the contract, and its instance is dropped.
    let _ = pick();
    let faulted = plugin.stats();
    assert_eq!((faulted.invalid, faulted.observed), (1, 1), "{faulted:?}");
    engine.on_membership(&Membership::Removed(WorkerId(104)));
    engine.on_membership(&Membership::Removed(WorkerId(103)));
    assert_eq!(pick(), Some(103), "three members stand");
    let remade = plugin.stats();
    assert_eq!(
        remade.observed - faulted.observed,
        1,
        "one snapshot, not the two changes: {remade:?}"
    );
    let (at_the_warm_up, at_its_decision) = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                warmed(&plugin);
                let warmed = plugin.stats().observed;
                assert_eq!(pick(), Some(103), "the warmed instance counts three");
                (warmed - remade.observed, plugin.stats().observed - warmed)
            })
            .join()
            .expect("the warming thread")
    });
    assert_eq!((at_the_warm_up, at_its_decision), (1, 0));
    // A change, then a lease's event with no decision between: the instance
    // hears the change first, then the event.
    let before = plugin.stats().observed;
    engine.on_membership(&Membership::Removed(WorkerId(102)));
    engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    assert_eq!(plugin.stats().observed - before, 2);
    assert_eq!(pick(), Some(102), "two members stand");
}

/// The feed keeps the models a worker left: an instance level with the feed
/// hears one call for the word, and counts the members it counted.
#[test]
fn a_guest_level_with_the_feed_hears_the_models_a_worker_left() {
    use smg_grand_central::tracking::Membership;
    let runtime = runtime();
    let plugin = load(&runtime, "left", guest::record_guest(), mode::COUNT_MEMBERS);
    let engine = plugin_engine(&plugin);
    let pick = || pick_ids(&decide(&engine, &pool(4), 1)).first().copied();
    engine.on_membership(&Membership::Snapshot(&[WorkerId(100), WorkerId(101)]));
    assert_eq!(pick(), Some(102), "two members stand");
    let level = plugin.stats().observed;
    let models = ["a".to_owned()];
    engine.on_membership(&Membership::left(WorkerId(100), &models));
    assert_eq!(pick(), Some(102), "the same two members");
    assert_eq!(plugin.stats().observed, level + 1, "one call for the word");
}

/// The feed keeps its last 64 changes. An instance behind that many hears
/// each, one call a change; one further behind, or behind a snapshot, hears
/// the list as it stands, as one snapshot.
#[test]
fn an_instance_behind_more_than_the_feed_keeps_hears_one_snapshot() {
    use smg_grand_central::tracking::Membership;
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "behind",
        guest::record_guest(),
        mode::COUNT_MEMBERS,
    );
    let engine = plugin_engine(&plugin);
    // The row the guest picks, and the `observe` calls the decision's
    // catch-up made.
    let pick = || {
        let before = plugin.stats().observed;
        let row = pick_ids(&decide(&engine, &pool(4), 1)).first().copied();
        (row, plugin.stats().observed - before)
    };
    let come_and_go = |times: u32| {
        for _ in 0..times {
            engine.on_membership(&Membership::Added(WorkerId(300)));
            engine.on_membership(&Membership::Removed(WorkerId(300)));
        }
    };
    engine.on_membership(&Membership::Snapshot(&[WorkerId(100)]));
    assert_eq!(pick(), (Some(101), 1), "new to the feed: the snapshot");
    come_and_go(32);
    assert_eq!(pick(), (Some(101), 64), "sixty-four behind: each change");
    come_and_go(32);
    engine.on_membership(&Membership::Added(WorkerId(301)));
    assert_eq!(pick(), (Some(102), 1), "sixty-five behind: one snapshot");
    engine.on_membership(&Membership::Added(WorkerId(302)));
    engine.on_membership(&Membership::Snapshot(&[WorkerId(100), WorkerId(101)]));
    engine.on_membership(&Membership::Added(WorkerId(303)));
    assert_eq!(
        pick(),
        (Some(103), 1),
        "behind a snapshot: the list as it stands"
    );
    assert_eq!(pick(), (Some(103), 0), "level: no call");
    // A replacement is folded into the list: an instance made after it
    // counts what stands.
    engine.on_membership(&Membership::replaced(WorkerId(303), WorkerId(304)));
    engine.on_membership(&Membership::Removed(WorkerId(304)));
    plugin.forget_instance();
    assert_eq!(
        pick(),
        (Some(102), 1),
        "the two that stand, as one snapshot"
    );
}

/// An instance hears the feed once a decision, at the first open that
/// reaches it. A change handed between a pair's two opens is not told
/// inside the pair, where the first side's resource is live on the
/// instance: it is heard at the next decision's first open.
#[test]
fn a_change_handed_between_a_pairs_opens_is_heard_at_the_next_decision() {
    use smg_grand_central::tracking::Membership;
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "between-sides",
        guest::record_guest(),
        mode::COUNT_MEMBERS,
    );
    let plain = Arc::new(plugin_engine(&plugin));
    // The first side's profile: the plugin's scope opens, then the host's
    // scope hands a change. The second side's open follows it.
    let prefill = profile_that(&plugin, &plain, Inside::HandsAMember(WorkerId(900)));
    let decode = plugin_profile(&plugin, None, None, None);
    let engine = pair_engine(vec![prefill, decode], 1);
    plain.on_membership(&Membership::Snapshot(&[WorkerId(100), WorkerId(101)]));
    plugin
        .warm()
        .expect("it warms, and hears the two that stand");
    let warmed = plugin.stats().observed;
    let _ = pair_decide(&engine);
    let paired = plugin.stats().observed;
    assert_eq!(paired - warmed, 0, "no telling inside the pair");
    let _ = pair_decide(&engine);
    assert_eq!(
        plugin.stats().observed - paired,
        1,
        "the first pair's change, at the second pair's first open"
    );
}

/// The open that reaches the instance may be a pair's second. A pair whose
/// first side meets the open breaker and whose second side opens at the
/// cool-down's end makes its instance there: that instance hears the list
/// as it stands before the side reads it.
#[test]
fn a_pair_that_straddles_the_cool_down_hears_the_feed_at_the_open_that_reaches_it() {
    use smg_grand_central::tracking::Membership;
    const START: u64 = 1_000_000;
    const COOL_DOWN: u64 = 1_000_000;
    // The host's clock is the case's own: the cool-down ends when the first
    // side's scope says so, whatever the runner's clock reads.
    static CLOCK: AtomicU64 = AtomicU64::new(START);
    let runtime = Arc::new(
        Runtime::with_clock(&OpenDecisions::new(), || CLOCK.load(Ordering::Relaxed))
            .expect("a runtime"),
    );
    let mut row = spec(
        "straddles",
        guest::record_guest(),
        mode::TRAP_ON_FIVE_MEMBERS,
    );
    row.breaker = BreakerSpec::default();
    row.breaker.failures = 1;
    row.breaker.window_micros = 60_000_000;
    row.breaker.cooldown_micros = COOL_DOWN;
    let plugin = loaded(&runtime, row);
    let plain = Arc::new(plugin_engine(&plugin));
    // The first side's profile: the plugin's open, then the host's scope
    // moves the clock to the cool-down's end. The second side's open follows.
    let ends = Inside::MovesTheClock(&CLOCK, START + COOL_DOWN);
    let prefill = profile_that(&plugin, &plain, ends);
    let decode = plugin_profile(&plugin, None, None, None);
    let engine = pair_engine(vec![prefill, decode], 1);
    // The warm-up tells the snapshot of five and this guest traps on it: the
    // one failure opens the breaker, and the slot is cold.
    let five: Vec<WorkerId> = (100..105).map(WorkerId).collect();
    plain.on_membership(&Membership::Snapshot(&five));
    plugin.warm().expect("the warm-up's own work is done");
    let warmed = plugin.stats();
    assert_eq!((warmed.traps, warmed.breaker_opens), (1, 1), "{warmed:?}");
    // Four go and one stands.
    for id in [104, 103, 102, 101] {
        plain.on_membership(&Membership::Removed(WorkerId(id)));
    }
    let decision = pair_decide(&engine);
    let paired = plugin.stats();
    assert_eq!(
        (
            paired.bypassed - warmed.bypassed,
            paired.lazy_instantiations - warmed.lazy_instantiations
        ),
        (1, 1),
        "the first side met the open breaker, the second made an instance: {paired:?}"
    );
    assert_eq!(
        (paired.observed - warmed.observed, paired.invalid),
        (1, 0),
        "the instance made at the second side's open hears one snapshot: {paired:?}"
    );
    // The second side's pick is the plugin's own: the row at one member.
    let decode_id = decision
        .pick()
        .and_then(|pick| pick.targets.first())
        .and_then(|lead| lead.pair)
        .map(|pair| pair.decode_id);
    assert_eq!(decode_id, Some(WorkerId(201)), "{decision:?}");
}

/// The pick of one decision over the pool of four that the plugin itself
/// answered: its calls answered and its pick within the pool. A decision
/// that went by the plugin, or took an answer out of the pool, names a row
/// too, the fallback's.
fn own_pick(plugin: &Plugin, engine: &Engine) -> Option<u64> {
    let before = plugin.stats();
    let picked = pick_ids(&decide(engine, &pool(4), 1)).first().copied();
    let after = plugin.stats();
    assert!(
        after.answered > before.answered && after.invalid == before.invalid,
        "the plugin's own answer: {before:?}, then {after:?}"
    );
    picked
}

/// In a told event the feed comes first: a change handed ahead of a lease's
/// event is heard ahead of it, so what the event's call reads of the
/// membership is the membership as the host told it.
#[test]
fn a_change_told_ahead_of_a_leases_event_is_heard_ahead_of_it() {
    use smg_grand_central::tracking::Membership;
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "ahead",
        guest::record_guest(),
        mode::MEMBERS_AT_A_DISPATCH,
    );
    let engine = plugin_engine(&plugin);
    engine.on_membership(&Membership::Snapshot(&[WorkerId(100), WorkerId(101)]));
    plugin.warm().expect("it warms, and hears the two");
    // One comes, then a dispatch with no decision between: the guest keeps
    // the count as it stands at the dispatch. Told the other way round the
    // dispatch reads two, and the pick names the row before.
    engine.on_membership(&Membership::Added(WorkerId(102)));
    engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    let picked = own_pick(&plugin, &engine);
    let stats = plugin.stats();
    assert_eq!(picked, Some(103), "three stood at the dispatch: {stats:?}");
}

/// An instance behind the feed hears a relabelling and a resize as the host
/// told them, each by its own kind.
#[test]
fn an_instance_behind_hears_a_relabelling_and_a_resize_each_as_told() {
    use smg_grand_central::tracking::Membership;
    let runtime = runtime();
    let plugin = load(&runtime, "kinds", guest::record_guest(), mode::COUNT_KINDS);
    let engine = plugin_engine(&plugin);
    engine.on_membership(&Membership::Snapshot(&[WorkerId(100), WorkerId(101)]));
    plugin.warm().expect("it warms");
    let warmed = plugin.stats().observed;
    engine.on_membership(&Membership::Relabelled(WorkerId(100)));
    engine.on_membership(&Membership::Resized(WorkerId(101)));
    // One up the relabelling and two up the resize: the row at three.
    let picked = own_pick(&plugin, &engine);
    let stats = plugin.stats();
    assert_eq!(picked, Some(103), "{stats:?}");
    assert_eq!(stats.observed - warmed, 2, "one call a change: {stats:?}");
}

/// The feed's list holds a worker once, and a replacement of a worker it
/// does not hold brings the new one, once: an instance new to the feed
/// hears the list as it stands.
#[test]
fn the_feeds_list_holds_a_worker_once_and_takes_an_unknown_ones_replacement() {
    use smg_grand_central::tracking::Membership;
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "folds",
        guest::record_guest(),
        mode::COUNT_MEMBERS,
    );
    let engine = plugin_engine(&plugin);
    for _ in 0..2 {
        engine.on_membership(&Membership::Added(WorkerId(100)));
    }
    for old in [900, 901] {
        engine.on_membership(&Membership::replaced(WorkerId(old), WorkerId(101)));
    }
    // Two members: one told twice, and one for a worker the list never
    // held, told twice. A list that held either twice, or left the second
    // out, names another row of the pool.
    let picked = own_pick(&plugin, &engine);
    let stats = plugin.stats();
    assert_eq!(picked, Some(102), "{stats:?}");
}

/// The models a worker left reach a guest as one word, with its names, ahead
/// of the relabelling of the same write: on either world an instance behind
/// the feed hears the two in the host's order, one call each.
#[test]
fn an_instance_behind_hears_the_models_a_worker_left_ahead_of_its_relabelling() {
    use smg_grand_central::tracking::Membership;
    let models = ["m-chat".to_owned(), "m-embed".to_owned()];
    for (world, component) in [
        ("left-record", guest::record_guest()),
        ("left-packed", guest::packed_guest()),
    ] {
        let runtime = runtime();
        let plugin = load(&runtime, world, component, mode::COUNT_LEFT);
        let engine = plugin_engine(&plugin);
        engine.on_membership(&Membership::Snapshot(&[WorkerId(100), WorkerId(101)]));
        plugin.warm().expect("it warms");
        let warmed = plugin.stats().observed;
        engine.on_membership(&Membership::left(WorkerId(100), &models));
        engine.on_membership(&Membership::Relabelled(WorkerId(100)));
        // One up each name of the word and one more up the relabelling
        // behind it: the row at three. A word not told, or told behind its
        // relabelling, names another row; one told twice names none of the
        // pool; a name the guest cannot read is its fault.
        let picked = own_pick(&plugin, &engine);
        let stats = plugin.stats();
        assert_eq!(picked, Some(103), "{world}: {stats:?}");
        assert_eq!(stats.observed - warmed, 2, "{world}: {stats:?}");
        assert_eq!(stats.traps, 0, "{world}: {stats:?}");
    }
}

/// A snapshot names ids and no pool. An instance further behind than the
/// feed keeps hears the list as it stands, in one call, and no word of the
/// models a worker left among the changes it missed.
#[test]
fn an_instance_that_hears_a_snapshot_for_what_it_missed_hears_no_left_word() {
    use smg_grand_central::tracking::Membership;
    let models = ["m-chat".to_owned(), "m-embed".to_owned()];
    let runtime = runtime();
    let plugin = load(&runtime, "behind", guest::record_guest(), mode::COUNT_LEFT);
    let engine = plugin_engine(&plugin);
    engine.on_membership(&Membership::Snapshot(&[WorkerId(100), WorkerId(101)]));
    plugin.warm().expect("it warms");
    let warmed = plugin.stats().observed;
    engine.on_membership(&Membership::left(WorkerId(100), &models));
    // The feed keeps 64 changes, and the word is the oldest of 65.
    for _ in 0..64 {
        engine.on_membership(&Membership::Resized(WorkerId(101)));
    }
    let picked = own_pick(&plugin, &engine);
    let stats = plugin.stats();
    assert_eq!(picked, Some(100), "no name was counted: {stats:?}");
    assert_eq!(stats.observed - warmed, 1, "one snapshot: {stats:?}");
}
