//! The lifecycle: the calls and the events a plugin hears.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use smg_grand_central::{
    document::Needs,
    engine::{
        Candidate, Engine, Plan, Profile, RequestFacts, Scorer, Target, TieBand, WeightedScorer,
        WorkerId,
    },
    plugins::{
        grpc::{GrpcPicker, GrpcPluginSpec, GrpcScorer, Plugin, Runtime},
        shared::{BreakerSpec, OpenDecisions},
    },
    service::proto,
    stages::{Best, LeastRequests, Lifecycle},
    tracking::{EventKind, EventSet, Lease, Membership, ReleaseReason, Side, Signal},
};

use crate::{
    binding, bound_profile, decide, lead, load, made_again, pool, profile, runtime, single_engine,
    spec,
    support::{
        grpc_plugin::{eventually, Behaviour, TestPlugin, ALL},
        said::{lines, panic_at, said, trap_spent},
    },
    unix, BUDGET_MICROS,
};

/// A scorer whose state the lifecycle's calls keep.
#[derive(Debug)]
struct Kept;

impl Lifecycle for Kept {
    fn needs_lifecycle(&self) -> bool {
        true
    }
}

impl Scorer for Kept {
    fn name(&self) -> &'static str {
        "kept"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(1.0);
        TieBand::NORMALIZED
    }
}

/// A plugin whose row lists no event declares no need of the lifecycle's
/// calls: an adapter needs them where its fallback does, and an engine built
/// in code says so. A plugin whose row lists an event needs them, whatever
/// its fallback and whichever adapter forwards.
#[test]
fn an_adapter_needs_the_lifecycle_where_its_row_or_its_fallback_does() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut unlisted = spec(&served, "kept");
    unlisted.observes = EventSet::NONE;
    let plugin = Plugin::load(&runtime, unlisted).expect("the plugin loads");
    let binder = plugin.binder();
    let adapter = |fallback: Option<Box<dyn Scorer>>| GrpcScorer::new(binder.bind(), fallback);
    assert!(adapter(Some(Box::new(Kept))).needs_lifecycle());
    assert!(!adapter(Some(Box::new(LeastRequests))).needs_lifecycle());
    assert!(!adapter(None).needs_lifecycle());

    let engine_of = |fallback: Option<Box<dyn Scorer>>| {
        let mut profile = Profile::of("adapted".into(), Box::new(Best));
        profile.filters = vec![];
        profile.scorers = vec![WeightedScorer::of(Box::new(adapter(fallback)), 1.0)];
        single_engine(profile)
    };
    assert!(engine_of(Some(Box::new(Kept))).needs_lifecycle());
    assert!(!engine_of(None).needs_lifecycle());

    let mut listed = spec(&served, "hearing");
    listed.observes = EventSet::of([EventKind::Completed]);
    let hearing = Plugin::load(&runtime, listed).expect("the plugin loads");
    let binder = hearing.binder();
    // The build's observer and the adapter that is not: each says so.
    assert!(GrpcScorer::new(binder.bind(), None).needs_lifecycle());
    assert!(GrpcPicker::new(binder.bind(), None).needs_lifecycle());
}

/// The models a worker left ride protocol 2. A plugin whose streams settled
/// on 2 hears the word once, with its names, ahead of the relabelling of the
/// same write. One that settled on 1 is not sent it and hears the
/// relabelling alone, numbered as the first event: the word takes no number
/// on a stream that does not carry it.
#[test]
fn the_models_a_worker_left_reach_a_plugin_on_protocol_2_alone() {
    use proto::{lifecycle_event::Event, membership_change::Kind};
    let models = ["m-chat".to_owned(), "m-embed".to_owned()];
    for version in [2, 1] {
        let served = TestPlugin::serve();
        served.script(|s| s.ack_protocol = version);
        let runtime = runtime();
        let plugin = load(&runtime, &served, "leaver");
        let engine = single_engine(profile(&plugin, ALL));
        engine.on_membership(&Membership::left(WorkerId(7), &models));
        engine.on_membership(&Membership::Relabelled(WorkerId(7)));
        // While a stream stands on 1 the left word is not raised.
        let raised = if version == 2 { 2 } else { 1 };
        assert_eq!(plugin.stats().observed, raised, "protocol {version}");
        // The relabelling is the last event of either stream.
        let relabelled = |events: &[proto::LifecycleEvent]| {
            events.last().is_some_and(|last| {
                matches!(&last.event, Some(Event::Membership(m)) if m.kind == Kind::Relabelled as i32)
            })
        };
        assert!(
            eventually(|| relabelled(&served.seen().events)),
            "protocol {version}"
        );
        let seen = served.seen();
        let told: Vec<(u64, i32, u64, &[String])> = seen
            .events
            .iter()
            .map(|event| match &event.event {
                Some(Event::Membership(m)) => (event.seq, m.kind, m.worker_id, &m.models[..]),
                other => panic!("a membership word, not {other:?}"),
            })
            .collect();
        let left = (1, Kind::Left as i32, 7, &models[..]);
        if version == 2 {
            assert_eq!(told, [left, (2, Kind::Relabelled as i32, 7, &[][..])]);
        } else {
            assert_eq!(told, [(1, Kind::Relabelled as i32, 7, &[][..])]);
        }
    }
}

/// A stream on protocol 1 hears what it heard before the word came, at the
/// queue's bound too. While a stream stands on 1 a left word is not raised:
/// twice the queue's worth of them takes no place and counts no loss, and
/// the relabelling behind them is the stream's first event.
#[test]
fn left_words_past_the_queues_bound_cost_a_stream_on_protocol_1_nothing() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "older");
    let engine = single_engine(profile(&plugin, ALL));
    let models = ["m-chat".to_owned()];
    // The queue holds 1,024 events.
    for _ in 0..2_048 {
        engine.on_membership(&Membership::left(WorkerId(7), &models));
    }
    engine.on_membership(&Membership::Relabelled(WorkerId(7)));
    assert!(
        eventually(|| !served.seen().events.is_empty()),
        "the relabelling reaches the plugin"
    );
    let stats = plugin.stats();
    assert_eq!(
        (stats.observed, stats.observe_dropped),
        (1, 0),
        "no left word took a place: {stats:?}"
    );
    let seen = served.seen();
    assert_eq!(seen.events.len(), 1, "the relabelling alone");
    assert_eq!(seen.events[0].seq, 1, "no loss is numbered");
}

/// A left word raised while no stream stands waits for the stream that
/// settles next: the events call that still stands yields at it, and the
/// word is kept with the events behind it. Where that stream settles on
/// protocol 1 the word is not written: the plugin hears the relabelling
/// behind it alone, as the first event, and the word takes no number on a
/// stream that does not carry it. Here the load's stream has ended as the
/// two words are raised.
#[test]
fn a_left_word_that_waited_for_a_stream_is_not_written_on_protocol_1() {
    use proto::{lifecycle_event::Event, membership_change::Kind};
    let served = TestPlugin::serve();
    served.behave(Behaviour::EndStream);
    let runtime = runtime();
    let mut spec = spec(&served, "waited");
    // The ended stream counts on the breaker: far from its threshold.
    spec.breaker = BreakerSpec {
        failures: 100,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    // The plugin ends the load's stream at its first frame.
    decide(&engine, &pool(4), 1);
    assert!(
        eventually(|| plugin.stats().failed_streams == 1),
        "the load's stream ended: no stream stands"
    );
    let models = ["m-chat".to_owned()];
    engine.on_membership(&Membership::left(WorkerId(7), &models));
    engine.on_membership(&Membership::Relabelled(WorkerId(7)));
    assert_eq!(
        plugin.stats().observed,
        2,
        "no stream stands, so both words are raised"
    );
    // The next stream settles on protocol 1, the fixture's own.
    served.behave(Behaviour::Honest);
    plugin.warm().expect("a stream settles");
    assert!(
        eventually(|| !served.seen().events.is_empty()),
        "the relabelling reaches the plugin"
    );
    let seen = served.seen();
    let told: Vec<(u64, i32, u64)> = seen
        .events
        .iter()
        .map(|event| match &event.event {
            Some(Event::Membership(m)) => (event.seq, m.kind, m.worker_id),
            other => panic!("a membership word, not {other:?}"),
        })
        .collect();
    assert_eq!(
        told,
        [(1, Kind::Relabelled as i32, 7)],
        "the left word was not written, and took no number"
    );
}

#[test]
fn the_lifecycle_reaches_the_plugin_once_in_order() {
    let served = TestPlugin::serve();
    let runtime = Runtime::with_clock(&OpenDecisions::new(), || 400_000).expect("a runtime");
    let plugin = load(&runtime, &served, "observer");
    // Three adapters of one profile: one of them forwards.
    let engine = single_engine(profile(&plugin, ALL));
    let lease = Lease::default();
    engine.on_dispatched(&lease, &Target::default(), Side::Single, 3);
    engine.on_released(&lease, ReleaseReason::ClientGone);
    engine.on_membership(&Membership::Snapshot(&[WorkerId(1), WorkerId(2)]));
    engine.on_signal(&Signal::gateway_veto(WorkerId(1), 500_000));
    assert_eq!(plugin.stats().observed, 4);
    assert!(eventually(|| served.seen().events.len() == 4));
    let seen = served.seen();
    assert_eq!(seen.observe_hellos.len(), 1);
    assert_eq!(seen.observe_hellos[0].plugin, "observer");
    let seqs: Vec<u64> = seen.events.iter().map(|e| e.seq).collect();
    assert_eq!(seqs, [1, 2, 3, 4], "numbered in the order they were raised");
    use proto::lifecycle_event::Event;
    match &seen.events[0].event {
        Some(Event::Dispatched(d)) => assert_eq!(d.since_decision_micros, 3),
        other => panic!("dispatched, not {other:?}"),
    }
    match &seen.events[1].event {
        Some(Event::Released(r)) => {
            assert_eq!(r.reason, proto::released::Reason::ClientGone as i32);
        }
        other => panic!("released, not {other:?}"),
    }
    match &seen.events[2].event {
        Some(Event::Membership(m)) => assert_eq!(m.ids, [1, 2]),
        other => panic!("membership, not {other:?}"),
    }
    match &seen.events[3].event {
        Some(Event::Signal(s)) => assert_eq!(
            s.kind,
            Some(proto::signal::Kind::GatewayVetoMicros(100_000)),
            "what the veto has left on the runtime's clock"
        ),
        other => panic!("a signal, not {other:?}"),
    }
}

/// A row lists the kinds of event its plugin hears, none by default. A
/// plugin whose row lists none is told nothing, and no `Observe` stream is
/// opened for it. One whose row lists some is sent those kinds alone,
/// numbered with no gap for the kinds it does not hear. A loaded plugin
/// says the kinds its row lists.
#[test]
fn a_plugin_hears_the_kinds_of_event_its_row_lists_and_no_other() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let tell = |engine: &Engine| {
        let lease = Lease::default();
        engine.on_dispatched(&lease, &Target::default(), Side::Single, 3);
        engine.on_released(&lease, ReleaseReason::ClientGone);
        engine.on_membership(&Membership::Snapshot(&[WorkerId(1), WorkerId(2)]));
        engine.on_signal(&Signal::gateway_veto(WorkerId(1), 500_000));
    };
    // The default row lists nothing.
    let mut unlisted = GrpcPluginSpec::new("unlisted", unix(&served));
    unlisted.connect_micros = BUDGET_MICROS;
    unlisted.budget_micros = BUDGET_MICROS;
    assert_eq!(unlisted.observes, EventSet::NONE);
    let deaf = Plugin::load(&runtime, unlisted).expect("the plugin loads");
    assert_eq!(deaf.observes(), EventSet::NONE);
    let deaf_engine = single_engine(profile(&deaf, ALL));
    assert!(!deaf_engine.needs_lifecycle());
    assert_eq!(lead(&decide(&deaf_engine, &pool(4), 1)), Some(103));
    tell(&deaf_engine);
    let stats = deaf.stats();
    assert_eq!((stats.observed, stats.observe_dropped), (0, 0), "{stats:?}");

    let mut listed = spec(&served, "listed");
    listed.observes = EventSet::of([EventKind::Released, EventKind::Membership]);
    let hearing = Plugin::load(&runtime, listed).expect("the plugin loads");
    assert_eq!(
        hearing.observes(),
        EventSet::of([EventKind::Released, EventKind::Membership])
    );
    let hearing_engine = single_engine(profile(&hearing, ALL));
    assert!(hearing_engine.needs_lifecycle());
    tell(&hearing_engine);
    assert_eq!(hearing.stats().observed, 2);
    assert!(eventually(|| served.seen().events.len() == 2));
    let seen = served.seen();
    // One stream, the hearing plugin's: the other plugin's `Decide` stream
    // settled first, answered a decision and opened none.
    let hellos: Vec<&str> = seen
        .observe_hellos
        .iter()
        .map(|hello| hello.plugin.as_str())
        .collect();
    assert_eq!(hellos, ["listed"]);
    let seqs: Vec<u64> = seen.events.iter().map(|e| e.seq).collect();
    assert_eq!(seqs, [1, 2], "no gap for the kinds it does not hear");
    use proto::lifecycle_event::Event;
    assert!(
        matches!(seen.events[0].event, Some(Event::Released(_))),
        "{:?}",
        seen.events[0]
    );
    assert!(
        matches!(seen.events[1].event, Some(Event::Membership(_))),
        "{:?}",
        seen.events[1]
    );
}

/// A dispatch on another target of its pick is told with a lease that names
/// that target's index, and the event's index is the lease's.
#[test]
fn a_dispatch_tells_the_plugin_the_index_its_lease_names() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "second-target");
    let engine = single_engine(profile(&plugin, ALL));
    let mut lease = Lease::default();
    lease.target_index = 1;
    engine.on_dispatched(&lease, &Target::default(), Side::Single, 3);
    assert!(eventually(|| served.seen().events.len() == 1));
    let seen = served.seen();
    match &seen.events[0].event {
        Some(proto::lifecycle_event::Event::Dispatched(d)) => assert_eq!(d.target_index, 1),
        other => panic!("dispatched, not {other:?}"),
    }
}

/// The events' numbers are the stream's: an event is numbered where it
/// enters the stream, on the I/O thread, so threads that raise events at
/// once cannot put a later number ahead of an earlier one.
#[test]
fn events_raised_on_many_threads_are_numbered_in_the_order_they_are_sent() {
    const THREADS: usize = 16;
    const EACH: usize = 60;
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "many-observers");
    let engine = single_engine(profile(&plugin, ALL));
    let start = std::sync::Barrier::new(THREADS);
    std::thread::scope(|scope| {
        for _ in 0..THREADS {
            scope.spawn(|| {
                start.wait();
                for _ in 0..EACH {
                    engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
                }
            });
        }
    });
    // Fewer than the queue holds: none is dropped.
    assert_eq!(plugin.stats().observed, (THREADS * EACH) as u64);
    assert!(eventually(|| served.seen().events.len() == THREADS * EACH));
    let seen = served.seen();
    let seqs: Vec<u64> = seen.events.iter().map(|e| e.seq).collect();
    let in_order: Vec<u64> = (1..=(THREADS * EACH) as u64).collect();
    assert_eq!(seqs, in_order, "the numbers follow the order on the stream");
}

/// An event the queue cannot take is dropped and counted, never a wait on
/// the thread that raised it. The I/O thread is held off the queue here,
/// so nothing leaves it while 2,000 events are raised.
#[test]
fn events_past_the_queue_are_dropped_and_counted() {
    for _ in 0..FLOODS {
        let served = TestPlugin::serve();
        let runtime = runtime();
        let plugin = load(&runtime, &served, "flooded-observer");
        let engine = single_engine(profile(&plugin, ALL));
        if !flooded_under_a_hold(&runtime, &engine) {
            made_again("a flood", HELD_PAST_THE_HOLD);
            continue;
        }
        let stats = plugin.stats();
        assert_eq!(stats.observed + stats.observe_dropped, 2_000, "{stats:?}");
        assert!(
            stats.observed >= 1_024 && stats.observe_dropped > 0,
            "the queue holds 1,024 and the rest is dropped: {stats:?}"
        );
        return;
    }
    panic!("the runner held every flood");
}

/// The floods a case makes at most: one whose test thread a held runner
/// paused past the hold of the I/O thread is made again.
const FLOODS: u32 = 5;

/// How long a flood holds the I/O thread off the queue.
const FLOOD_HOLD: Duration = Duration::from_millis(300);

/// What a flood that is made again says of itself.
const HELD_PAST_THE_HOLD: &str = "its thread was held past the hold of the I/O thread";

/// Raises 2,000 events while the I/O thread is held off the queue, and says
/// whether the hold outlasted them. A test thread paused past the hold lets
/// the queue drain under the events, and what is raised after it carries the
/// count of refusals: that flood asserts nothing.
fn flooded_under_a_hold(runtime: &Arc<Runtime>, engine: &Engine) -> bool {
    let asked = Instant::now();
    runtime.hold_io(FLOOD_HOLD);
    for _ in 0..2_000 {
        engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    }
    asked.elapsed() < FLOOD_HOLD
}

/// A refused event leaves its mark on the numbers, where the loss fell. An
/// event's number counts the events sent before it and the events the queue
/// had refused when it entered the queue. Three events go first and arrive;
/// then the I/O thread is held off the queue while it fills and refuses.
/// The events the queue took were raised before the loss: they arrive
/// gapless. The next event raised skips by what was lost, so a plugin that
/// keeps a ledger of leases reads the gap behind everything raised before
/// it.
#[test]
fn a_refused_event_leaves_a_gap_in_the_numbers() {
    for _ in 0..FLOODS {
        let served = TestPlugin::serve();
        let runtime = runtime();
        let plugin = load(&runtime, &served, "ledger-keeper");
        let engine = single_engine(profile(&plugin, ALL));
        let dispatch =
            || engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
        for _ in 0..3 {
            dispatch();
        }
        assert!(eventually(|| served.seen().events.len() == 3));
        if !flooded_under_a_hold(&runtime, &engine) {
            made_again("a refused event", HELD_PAST_THE_HOLD);
            continue;
        }
        let stats = plugin.stats();
        let (sent, refused) = (stats.observed, stats.observe_dropped);
        assert!(refused > 0, "the queue refused some: {stats:?}");
        assert!(eventually(|| served.seen().events.len() as u64 == sent));
        let seqs: Vec<u64> = served.seen().events.iter().map(|e| e.seq).collect();
        assert!(
            seqs.windows(2).all(|pair| pair[0] + 1 == pair[1]),
            "an event raised before the loss arrived behind a gap"
        );
        assert_eq!(
            seqs.last().copied(),
            Some(sent),
            "the events raised before the loss read gapless"
        );
        dispatch();
        assert!(eventually(|| served.seen().events.len() as u64 == sent + 1));
        assert_eq!(
            served.seen().events.last().map(|e| e.seq),
            Some(sent + 1 + refused),
            "the next event raised skips by what the queue refused"
        );

        return;
    }
    panic!("the runner held every flood");
}

/// The numbers go on across two `Observe` streams: the first server goes, a
/// second takes its socket's path, and what the second is sent is numbered
/// above everything the first was.
#[test]
fn the_numbers_go_on_across_two_observe_streams() {
    let mut first = TestPlugin::serve();
    let runtime = runtime();
    let mut row = spec(&first, "two-streams");
    row.connect_micros = 1_000_000;
    let plugin = Plugin::load(&runtime, row).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    for _ in 0..40 {
        engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    }
    assert!(eventually(|| first.seen().events.len() == 40));
    let last = first.seen().events.last().map(|event| event.seq);
    assert_eq!(last, Some(40));
    let path = first.socket.clone();
    first.stop();
    let second = TestPlugin::serve();
    std::fs::rename(&second.socket, &path).expect("the second server takes the first's path");
    let mut raised = 0_u32;
    while second.seen().events.len() < 20 {
        raised += 1;
        assert!(raised <= 15_000, "no event reached the second server");
        engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
        std::thread::sleep(Duration::from_millis(1));
    }
    let seen = second.seen();
    let numbers: Vec<u64> = seen.events.iter().map(|event| event.seq).collect();
    assert!(
        numbers.first().copied() > last,
        "the numbers went down across two streams: {numbers:?}"
    );
    assert!(
        numbers.windows(2).all(|pair| pair[1] > pair[0]),
        "{numbers:?}"
    );
}

/// The numbers are kept across a turn of the `Observe` task that panicked.
/// The first server goes, and the host's line of that end panics where it
/// is said, once. The task is started again a pause later, and what the
/// second server is sent is numbered above everything the first was.
#[test]
fn the_numbers_go_on_across_a_turn_that_panicked() {
    let mut first = TestPlugin::serve();
    let runtime = runtime();
    let mut row = spec(&first, "numbers-over-a-panic");
    row.connect_micros = 1_000_000;
    let plugin = Plugin::load(&runtime, row).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    for _ in 0..40 {
        engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    }
    assert!(eventually(|| first.seen().events.len() == 40));
    let last = first.seen().events.last().map(|event| event.seq);
    assert_eq!(last, Some(40));
    let path = first.socket.clone();
    panic_at("numbers-over-a-panic", "observe stream failed", 1);
    first.stop();
    // No event is raised while the first server goes: the call's own end is
    // then all the task sees, and its line panics where it is said. With an
    // event queued at that instant the send can see the end first, and that
    // end has no line.
    assert!(
        eventually(|| trap_spent("numbers-over-a-panic")),
        "the line of the first stream's end did not panic"
    );
    let second = TestPlugin::serve();
    std::fs::rename(&second.socket, &path).expect("the second server takes the first's path");
    let mut raised = 0_u32;
    while second.seen().events.len() < 20 {
        raised += 1;
        assert!(raised <= 15_000, "no event reached the second server");
        engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        !lines("numbers-over-a-panic", "observe task panicked").is_empty(),
        "the host said nothing of the task that panicked"
    );
    let seen = second.seen();
    let numbers: Vec<u64> = seen.events.iter().map(|event| event.seq).collect();
    assert!(
        numbers.first().copied() > last,
        "the numbers started again with the turn: {numbers:?}"
    );
    assert!(
        numbers.windows(2).all(|pair| pair[1] > pair[0]),
        "{numbers:?}"
    );
}

/// A telling that panics is told once an outage: the outage's line is held
/// outside the task's turn. Here the host's line of a failed stream is armed
/// to panic three times, and the first server goes with no event raised: a
/// task that has sent nothing says its first panic. A second server takes
/// the path, is sent events and goes, and a third is reached: the task went
/// on, no call stood, and an event sent arms no line. The outage's line and
/// the task's were each said once.
#[test]
fn a_telling_that_panics_is_told_once_an_outage() {
    let name = "panics-in-a-row";
    let mut first = TestPlugin::serve();
    let runtime = runtime();
    let mut row = spec(&first, name);
    row.connect_micros = 1_000_000;
    let plugin = Plugin::load(&runtime, row).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    assert!(eventually(|| first.seen().observe_hellos.len() == 1));
    let path = first.socket.clone();
    panic_at(name, "observe stream failed", 3);
    first.stop();
    assert!(
        eventually(|| lines(name, "observe task panicked").len() == 1),
        "the first panic of a task that sent nothing was not said"
    );
    // A stopped server stays to the case's end: the path is in the first
    // one's directory.
    let reach = |served: &TestPlugin| {
        std::fs::rename(&served.socket, &path).expect("the next server takes the path");
        let mut raised = 0_u32;
        while served.seen().events.is_empty() {
            raised += 1;
            assert!(raised <= 15_000, "no event reached the next server");
            engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    let mut second = TestPlugin::serve();
    reach(&second);
    second.stop();
    let third = TestPlugin::serve();
    reach(&third);
    assert!(!trap_spent(name), "the outage was told again");
    assert_eq!(
        lines(name, "observe stream failed").len(),
        1,
        "{:?}",
        lines(name, "observe stream failed")
    );
    assert_eq!(
        lines(name, "observe task panicked").len(),
        1,
        "{:?}",
        lines(name, "observe task panicked")
    );
}

/// The ends of calls that did not stand are one outage, whichever side of
/// the host's task saw each: the call's own outcome, or a send that found
/// the stream gone. Here four servers go one after another while events are
/// raised, so every call carried events and a send can meet an end: an
/// event carried arms no line, and the four ends say one.
#[test]
fn ends_of_calls_that_did_not_stand_say_one_line() {
    const ENDS: usize = 4;
    said();
    let name = "ends-under-sends";
    let mut served = TestPlugin::serve();
    let runtime = runtime();
    let mut row = spec(&served, name);
    row.connect_micros = 1_000_000;
    let plugin = Plugin::load(&runtime, row).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let path = served.socket.clone();
    let mut stopped = Vec::new();
    // The server after the last end is reached too: the task saw that end.
    for server in 1..=ENDS + 1 {
        let mut raised = 0_u32;
        while served.seen().events.is_empty() {
            raised += 1;
            assert!(raised <= 15_000, "no event reached server {server}");
            engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
            std::thread::sleep(Duration::from_millis(1));
        }
        if server <= ENDS {
            served.stop();
            // A stopped server stays to the case's end: the path is in the
            // first one's directory.
            stopped.push(std::mem::replace(&mut served, TestPlugin::serve()));
            std::fs::rename(&served.socket, &path).expect("the next server takes the path");
        }
    }
    assert_eq!(
        lines(name, "observe stream failed").len(),
        1,
        "{:?}",
        lines(name, "observe stream failed")
    );
}

/// A plugin that ends every `Observe` call is one outage, with events
/// waiting or with none: a call that ends at its hello did not stand, and an
/// event the host sent into it arms no line. The server's word is set before
/// the load, so the first call ends at its hello and says the line. Events
/// are raised from then on: the next calls end at theirs, a connect frame
/// apart, and say nothing.
#[test]
fn calls_that_end_at_their_hello_say_one_line() {
    said();
    let name = "ends-at-the-hello";
    let served = TestPlugin::serve();
    served.script(|s| s.observe_ack = "enough".to_owned());
    let runtime = runtime();
    let mut row = spec(&served, name);
    row.connect_micros = 1_000_000;
    let plugin = Plugin::load(&runtime, row).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let mut raised = 0_u32;
    while served.seen().observe_hellos.len() < 4 {
        raised += 1;
        assert!(raised <= 15_000, "no fourth call: {:?}", plugin.stats());
        engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        lines(name, "observe stream failed").len(),
        1,
        "{:?}",
        lines(name, "observe stream failed")
    );
}

/// The `Observe` call's answer is bounded in bytes like a decision's: an Ack
/// past the row's bound ends the call as a status about its length, and the
/// host's line carries no more of the status than its own bound.
#[test]
fn an_observe_answer_past_the_message_bound_is_refused_by_its_length() {
    said();
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "wordy-ack");
    let engine = single_engine(profile(&plugin, ALL));
    served.script(|s| s.observe_ack = "x".repeat(300_000));
    engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    assert!(
        eventually(|| !lines("wordy-ack", "observe stream failed").is_empty()),
        "no line of the observe stream: {:?}",
        plugin.stats()
    );
    let line = lines("wordy-ack", "observe stream failed").remove(0);
    assert!(
        line.contains("too large") || line.contains("OutOfRange"),
        "an Ack past the bound was read: {line}"
    );
    assert!(line.len() < 600, "the line is {} bytes", line.len());
}

/// A build that binds the plugin to two profiles takes both bound sets from
/// one binder: the engine hands an event to every profile and the plugin
/// hears it once. An engine built later over the same plugin has an observer
/// of its own, beside the first engine and once it is gone.
#[test]
fn the_lifecycle_reaches_the_plugin_once_per_engine_build() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "rebound");
    let dispatch = |engine: &Engine| {
        engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
        plugin.stats().observed
    };
    let binder = plugin.binder();
    let first = Engine::new(
        vec![
            bound_profile(binder.bind(), ALL, true),
            bound_profile(binder.bind(), ALL, true),
        ],
        vec![binding(Plan::PdPair, Some(1))],
        1,
        Needs::default(),
    );
    assert_eq!(
        dispatch(&first),
        1,
        "two bound sets of one build, one observer"
    );
    let later = single_engine(profile(&plugin, ALL));
    assert_eq!(
        dispatch(&later),
        2,
        "the later engine forwards beside the first"
    );
    assert_eq!(dispatch(&first), 3, "and the first still does");
    drop(first);
    assert_eq!(dispatch(&later), 4, "and once the first is gone");
}
