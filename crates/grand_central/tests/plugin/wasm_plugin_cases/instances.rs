//! The instances kept per decider: a thread's own and a named decider's, a
//! warm-up, a place left lent, a new build, and which instance hears what a
//! thread tells.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    mpsc, Arc,
};

use smg_grand_central::{
    engine::{
        Candidate, Constraints, Decision, DecisionSlot, Engine, Profile, Scratch, Target,
        WeightedScorer, WorkerId,
    },
    plugins::{
        shared::{BreakerSpec, OpenDecisions},
        wasm::{Plugin, PluginSpec, Runtime, WasmScorer},
    },
    stages::{Best, Lifecycle},
    tracking::{Lease, Side},
};

use crate::{
    bound_clock, decide, facts, four, load, loaded, made_again, off_cpu_since, pair_decide,
    pair_engine, pick_ids, plugin_engine, plugin_profile, pool, profile_that, runtime, said,
    single_engine, spec,
    support::wasm_guest::{self as guest, mode},
    the_close_rebuilt, tried, warmed, warnings, Inside, HELD_NAME, INIT_FRAME, PANICS, TRIES,
    TRY_BEGAN, WARMED_AGAIN,
};

#[test]
fn a_decision_on_a_kept_instance_opens_one_resource_and_leaves_no_instance_behind() {
    let runtime = runtime();
    let plugin = load(&runtime, "p", guest::record_guest(), mode::LEAST_IN_FLIGHT);
    let engine = plugin_engine(&plugin);
    let candidates = pool(4);
    for seed in 0..50 {
        decide(&engine, &candidates, seed);
    }
    let stats = plugin.stats();
    assert_eq!(
        stats.instantiations, 1,
        "one kept instance for fifty decisions"
    );
    assert_eq!(stats.answered, 200);
    assert!(plugin.is_warm());
}

/// Panics inside `plugin`'s next log line, once. The guest's log line is a
/// host call, so the panic unwinds through a guest call and leaves its
/// instance where it stood: a clock that panics behind a failed call finds
/// the instance dropped.
fn panic_in_a_line_of(plugin: &str) {
    said();
    PANICS
        .lock()
        .expect("the panics")
        .push(format!("plugin=\"{plugin}\""));
}

#[test]
fn warm_creates_the_threads_instance_and_a_cold_thread_counts_its_first_decision() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "warm",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    // Both threads live at once: the slot crate recycles the id of a thread
    // that exited, and a later thread would adopt its instance.
    let gate = Arc::new(std::sync::Barrier::new(3));
    let warmer = {
        let (plugin, gate) = (Arc::clone(&plugin), Arc::clone(&gate));
        std::thread::spawn(move || {
            let before = plugin.is_warm();
            // The verdict leaves through the gate: a warm that panicked
            // here would leave the other two waiting at it.
            let warmed = plugin.warm();
            let after = plugin.is_warm();
            gate.wait();
            warmed.map(|()| (before, after))
        })
    };
    let cold = {
        let (plugin, gate) = (Arc::clone(&plugin), Arc::clone(&gate));
        std::thread::spawn(move || {
            let cold = plugin.is_warm();
            gate.wait();
            cold
        })
    };
    gate.wait();
    assert_eq!(warmer.join().expect("joins").expect("warms"), (false, true));
    assert!(
        !cold.join().expect("joins"),
        "a thread that neither warmed nor decided holds nothing"
    );
    let engine = plugin_engine(&plugin);
    plugin.forget_instance();
    decide(&engine, &pool(4), 1);
    let stats = plugin.stats();
    assert_eq!(stats.instantiations, 3, "the load, the warm, the lazy one");
    assert_eq!(stats.lazy_instantiations, 1);
}

/// One loaded plugin in two engine builds, as a spec applied again leaves
/// it: a thread decides under the old build and the plugin fails that
/// decision, then the thread's next decision is the new build's first. It is
/// another decision to the plugin: its stages are asked, not bypassed for
/// the old build's failure. A plugin keys its part in a decision by the
/// decision's number alone, so the numbers come from the open decisions its
/// runtime was built over, as wide as the plugin's sharing: counted per
/// build, the new build's first decision on the thread would carry the
/// number of the old build's.
#[test]
fn a_new_builds_first_decision_on_a_thread_is_not_the_old_builds_last() {
    const CASE: &str = "a new build's first decision on a thread";
    tried(|made| {
        let runtime = runtime();
        let plugin = load(
            &runtime,
            "four-trapper",
            guest::record_guest(),
            mode::TRAP_ON_FOUR_CANDIDATES,
        );
        let old = plugin_engine(&plugin);
        let new = plugin_engine(&plugin);
        // A thread of its own: its one decision under the old build, a lazy
        // `init` among what that decision paid, then its first under the new.
        let (failed, asked, took, held) = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let (wall, cpu) = (std::time::Instant::now(), bound_clock());
                    decide(&old, &pool(4), 1);
                    let (took, held) = (wall.elapsed(), off_cpu_since(wall, cpu));
                    let failed = plugin.stats();
                    decide(&new, &pool(3), 2);
                    (failed, plugin.stats(), took, held)
                })
                .join()
                .expect("the deciding thread")
        });
        // Both instances the old build's decision makes come of an `init`, on
        // the wall clock: the cold slot's, and the close's rebuild. An `init`
        // the host ended took its frame on that clock, and the decision with
        // it. The thread's own clock can count that time the thread's
        // ([`loaded`] says where), so the witness is read on the clock of the
        // bound.
        let made_both = (failed.traps, failed.lazy_instantiations);
        if made_both != (1, 2) && took >= INIT_FRAME && made < TRIES {
            made_again(
                CASE,
                &format!(
                    "the old build's decision took {took:?} on the wall clock, the frame of an init or more, its thread off the CPU {held:?} of it"
                ),
            );
            return false;
        }
        assert_eq!(
            made_both,
            (1, 2),
            "the old build's decision made its instance, trapped in the filter and was rebuilt (try {made}, {took:?} on the wall clock, the thread off the CPU {held:?}): {failed:?}"
        );
        assert_eq!(
            (asked.traps, asked.answered - failed.answered),
            (1, 4),
            "the new build's first decision asked the constructor and the three stages: {asked:?}"
        );
        true
    });
}

/// How many pairs of threads a case of a taken place makes for one pair to
/// share a place.
const PLACES: u32 = 64;

/// A plugin one of whose working places a thread left lent to decider 1 at
/// its exit, with the engine that decides on it. The thread made a decision
/// as decider 1 whose close the host stopped half-way, by a panic inside
/// the destructor's log line: the close did not reach the give-back, and
/// the thread exited with the decider's instance in its place.
///
/// `thread_local` hands a new thread the lowest id that is free, and the
/// first thread's is free at its join, which returns behind the thread's
/// destructors, that crate's among them. The suite's other threads take and
/// free ids all the while, and that crate keys every map of its kind by the
/// one id, so an entry of the case's own says whether a later thread stands
/// in the first's place. A pair that shared none is made again, [`PLACES`]
/// pairs at most, and a case asserts that one pair did. The calling thread
/// takes its id ahead of the first thread: its place is its own.
struct LeftLent {
    plugin: Arc<Plugin>,
    engine: Engine,
    candidates: Vec<Candidate>,
    place: thread_local::ThreadLocal<std::cell::Cell<bool>>,
}

impl LeftLent {
    fn by_a_thread_that_exited(name: &str) -> Self {
        let row = spec(name, guest::record_guest(), mode::LOG_IN_DESTRUCTOR);
        Self::of(&runtime(), row)
    }

    /// As [`Self::by_a_thread_that_exited`], the plugin loaded from `row` on
    /// `runtime`: a guest of the row's mode writes a log line in its
    /// destructor.
    fn of(runtime: &Arc<Runtime>, row: PluginSpec) -> Self {
        let name = row.name.clone();
        let plugin = loaded(runtime, row);
        let mut profile = Profile::of("one-stage".into(), Box::new(Best));
        profile.filters = vec![];
        profile.scorers = vec![WeightedScorer::of(
            Box::new(WasmScorer::new(plugin.bind(), None)),
            1.0,
        )];
        let engine = single_engine(profile);
        let left = Self {
            plugin,
            engine,
            candidates: pool(3),
            place: thread_local::ThreadLocal::new(),
        };
        left.place.get_or_default().set(false);
        panic_in_a_line_of(&name);
        let ended = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    left.place.get_or_default().set(true);
                    let _turn = DecisionSlot::new(1).enter();
                    decide(&left.engine, &left.candidates, 1)
                })
                .join()
        });
        assert!(ended.is_err(), "the first thread's close panicked");
        left
    }

    /// Whether a new thread stood in the first thread's place, and what
    /// `act` said there.
    fn taken<T: Send>(&self, act: impl FnOnce(&Self) -> T + Send) -> (bool, T) {
        std::thread::scope(|scope| {
            scope
                .spawn(|| (self.place.get_or_default().get(), act(self)))
                .join()
                .expect("the second thread")
        })
    }

    /// Decider 1's decision on the calling thread, and the `observe` calls
    /// its catch-up made.
    fn heard_as_decider_1(&self) -> u64 {
        let before = self.plugin.stats().observed;
        let _turn = DecisionSlot::new(1).enter();
        assert!(
            matches!(decide(&self.engine, &self.candidates, 1), Decision::Pick(_)),
            "decider 1 decides"
        );
        self.plugin.stats().observed - before
    }

    /// Decider 1's two turns on the calling thread, with what each heard:
    /// the first makes an instance, and the second hears the list, one
    /// member told between them. Two more members are told behind them.
    fn two_turns_then_two_members(&self) -> [u64; 2] {
        use smg_grand_central::tracking::Membership;
        let anew = self.heard_as_decider_1();
        self.engine.on_membership(&Membership::Added(WorkerId(102)));
        let listed = self.heard_as_decider_1();
        self.engine.on_membership(&Membership::Added(WorkerId(103)));
        self.engine.on_membership(&Membership::Added(WorkerId(104)));
        [anew, listed]
    }
}

/// A thread under no name holds no instance but its own, where it takes the
/// working place of a thread that exited with a named decider's state lent
/// ([`LeftLent`]). The thread that takes the place reads the mark the first
/// left there: under no name it is not warm, and decider 1's state is at
/// its rest again for the thread that next acts as decider 1.
#[test]
fn a_thread_under_no_name_gives_back_the_state_its_place_was_left_with() {
    // One pair: whether the second thread stood in the first's place, and
    // whether it was warm there under no name, then as decider 1.
    let pair = || {
        LeftLent::by_a_thread_that_exited("left-lent").taken(|left| {
            let own = left.plugin.is_warm();
            let _turn = DecisionSlot::new(1).enter();
            (own, left.plugin.is_warm())
        })
    };
    let shared = (0..PLACES).map(|_| pair()).find(|(shared, _)| *shared);
    assert_eq!(
        shared,
        Some((true, (false, true))),
        "a second thread stood in the first's place: under no name it held no instance there, and decider 1's state was at its rest behind it"
    );
}

/// A state that comes back late does not replace the one that rests. The
/// first thread exits with decider 1's instance in its place ([`LeftLent`]).
/// Decider 1 then takes its turns on the case's own thread: its rest is
/// empty, so its first decision makes an instance. One member is then told,
/// and its second decision hears the list as it stands. The thread that
/// takes the first thread's place gives back what it finds there, late. The
/// rest keeps the instance of the later turns: it is level with the feed,
/// and decider 1's third decision hears nothing. The instance the first
/// thread left never heard the feed: put over the newer one, it would hear
/// the list there.
#[test]
fn a_state_given_back_late_does_not_replace_the_one_at_rest() {
    // One pair: whether the second thread stood in the first's place and
    // was warm there under no name, the `observe` calls of decider 1's three
    // decisions on this thread, and the instances made.
    let pair = || {
        let left = LeftLent::by_a_thread_that_exited("came-late");
        let anew = left.heard_as_decider_1();
        left.engine
            .on_membership(&smg_grand_central::tracking::Membership::Added(WorkerId(
                102,
            )));
        let behind = left.heard_as_decider_1();
        let (shared, own) = left.taken(|left| left.plugin.is_warm());
        (
            shared,
            own,
            [anew, behind, left.heard_as_decider_1()],
            left.plugin.stats().instantiations,
        )
    };
    let shared = (0..PLACES).map(|_| pair()).find(|(shared, ..)| *shared);
    assert_eq!(
        shared,
        Some((true, false, [0, 1, 0], 2)),
        "decider 1's later instance heard the list once and nothing behind the late give-back: it rests"
    );
}

/// A turn under the decider's own name in a place left lent runs on the
/// state at rest. The first thread exits with decider 1's instance in its
/// place ([`LeftLent`]). Decider 1 takes two turns on the case's own
/// thread, and two more members are told. The thread that takes the first
/// thread's place then acts as decider 1 itself. The hold
/// it finds there is stale, since the rest names the other place: it drops
/// it and decides on the state at rest, which hears the two changes. That
/// state rests behind the turn: decider 1's next decision on the case's
/// thread hears nothing. On the instance the first thread left, the turn
/// would hear the list as one snapshot and its close would come late: the
/// next decision would hear the two changes again.
#[test]
fn a_turn_under_the_same_name_in_a_place_left_lent_runs_on_the_state_at_rest() {
    // One pair: whether the second thread stood in the first's place, the
    // `observe` calls of decider 1's decisions (two on this thread, the
    // second thread's, one more on this thread), and the instances made.
    let pair = || {
        let left = LeftLent::by_a_thread_that_exited("same-name");
        let before = left.two_turns_then_two_members();
        // Behind its turn the second thread reads, under no name, whether
        // an instance is its own: the stale one is dropped, not set aside.
        let (shared, turn) = left.taken(|left| (left.heard_as_decider_1(), left.plugin.is_warm()));
        (
            shared,
            before,
            turn,
            left.heard_as_decider_1(),
            left.plugin.stats().instantiations,
        )
    };
    let shared = (0..PLACES).map(|_| pair()).find(|(shared, ..)| *shared);
    assert_eq!(
        shared,
        Some((true, [0, 1], (2, false), 0, 2)),
        "the second thread's turn as decider 1 heard the two changes on the state at rest, kept no instance of the stale hold, and left that state at rest"
    );
}

/// A hold of the same decider is checked by the open that reaches the bring,
/// also where the open breaker turned the decision's first open away. The
/// first thread exits with decider 1's instance in its place ([`LeftLent`]),
/// and decider 1 takes a turn on the case's own thread: the rest names that
/// place. A warm-up then faults on a snapshot of five and opens the breaker,
/// and four members go. The thread that takes the first thread's place
/// decides a pair as decider 1. Its prefill side meets the open breaker, the
/// cool-down ends inside that side's scope, and the decode side's open is
/// the first to reach the bring. The hold it finds is stale: it drops it and
/// decides on the state at rest, which hears the list as it stands and rests
/// behind the decision, so decider 1's next decision on the case's thread
/// hears nothing. On the instance the first thread left, the close would
/// come late, and that next decision would hear the list again.
#[test]
fn a_hold_is_checked_by_the_open_that_reaches_the_bring_behind_an_open_breaker() {
    use smg_grand_central::tracking::Membership;
    const START: u64 = 1_000_000;
    const COOL_DOWN: u64 = 1_000_000;
    // The host's clock is the case's own: the cool-down ends when the
    // prefill side's scope says so.
    static CLOCK: AtomicU64 = AtomicU64::new(START);
    // One pair: whether the second thread stood in the first's place, what
    // decider 1's first turn here heard, the warm-up's fault, what the pair
    // decision heard and bypassed, and what decider 1's next turn here heard.
    let pair = || {
        CLOCK.store(START, Ordering::Relaxed);
        let runtime = Arc::new(
            Runtime::with_clock(&OpenDecisions::new(), || CLOCK.load(Ordering::Relaxed))
                .expect("a runtime"),
        );
        let mode = mode::TRAP_ON_FIVE_AND_LOG_IN_DESTRUCTOR;
        let mut row = spec("corner", guest::record_guest(), mode);
        row.breaker = BreakerSpec::default();
        row.breaker.failures = 1;
        row.breaker.window_micros = 60_000_000;
        row.breaker.cooldown_micros = COOL_DOWN;
        let left = LeftLent::of(&runtime, row);
        let anew = left.heard_as_decider_1();
        let five: Vec<WorkerId> = (100..105).map(WorkerId).collect();
        left.engine.on_membership(&Membership::Snapshot(&five));
        left.plugin.warm().expect("the warm-up's own work is done");
        let warmed = left.plugin.stats();
        for id in [104, 103, 102, 101] {
            left.engine
                .on_membership(&Membership::Removed(WorkerId(id)));
        }
        let plain = Arc::new(plugin_engine(&left.plugin));
        let ends = Inside::MovesTheClock(&CLOCK, START + COOL_DOWN);
        let prefill = profile_that(&left.plugin, &plain, ends);
        let decode = plugin_profile(&left.plugin, None, None, None);
        let engine = pair_engine(vec![prefill, decode], 1);
        let (shared, paired) = left.taken(|left| {
            let before = left.plugin.stats();
            let _turn = DecisionSlot::new(1).enter();
            let _ = pair_decide(&engine);
            let after = left.plugin.stats();
            (
                after.observed - before.observed,
                after.bypassed - before.bypassed,
            )
        });
        let next = left.heard_as_decider_1();
        (
            shared,
            anew,
            (warmed.traps, warmed.breaker_opens),
            paired,
            next,
        )
    };
    let shared = (0..PLACES).map(|_| pair()).find(|(shared, ..)| *shared);
    assert_eq!(
        shared,
        Some((true, 0, (1, 1), (1, 1), 0)),
        "the decode side's open checked the hold: the state at rest heard the list and rests behind the pair"
    );
}

/// An event told under the decider's own name in a place left lent reaches
/// the state at rest: the start of a told event checks the hold as a
/// decision's first open does. The staging is the case's above. The thread
/// that takes the first thread's place tells one dispatch as decider 1: the
/// state at rest comes, hears the two changes and then the event, and rests
/// behind it, so decider 1's next decision hears nothing. On the instance
/// the first thread left, the event would be heard behind one snapshot and
/// stay in the place: the next decision would hear the two changes.
#[test]
fn an_event_told_under_the_same_name_in_a_place_left_lent_reaches_the_state_at_rest() {
    // One pair, as above: the second thread's `observe` calls are its told
    // event's.
    let pair = || {
        let left = LeftLent::by_a_thread_that_exited("same-name-told");
        let before = left.two_turns_then_two_members();
        let (shared, told) = left.taken(|left| {
            let before = left.plugin.stats().observed;
            let _turn = DecisionSlot::new(1).enter();
            left.engine
                .on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
            left.plugin.stats().observed - before
        });
        (
            shared,
            before,
            [told, left.heard_as_decider_1()],
            left.plugin.stats().instantiations,
        )
    };
    let shared = (0..PLACES).map(|_| pair()).find(|(shared, ..)| *shared);
    assert_eq!(
        shared,
        Some((true, [0, 1], [3, 0], 2)),
        "the second thread's event as decider 1 was heard behind the two changes on the state at rest, and that state rests behind it"
    );
}

/// A warm-up under a name holds the decider's state for its own span: at
/// its end the state is back at the decider's rest, for whichever thread
/// acts as the decider next. The first thread lives until the second has
/// read, so the second stands in a place of its own. The warm-up takes the
/// apply's own instance: no thread here makes one.
#[test]
fn a_named_warm_up_leaves_the_state_at_its_deciders_rest() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "warmed-as-one",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let (warmed, first_warmed) = mpsc::channel();
    let (read, second_read) = mpsc::channel::<()>();
    let warm_behind_it = std::thread::scope(|scope| {
        let plugin = &plugin;
        scope.spawn(move || {
            {
                let _turn = DecisionSlot::new(1).enter();
                plugin.warm().expect("decider 1 warms");
            }
            warmed.send(()).expect("the case waits for the warm-up");
            // Alive, under no name, until the second thread has read.
            let _ = second_read.recv();
        });
        first_warmed.recv().expect("the first thread warmed");
        let warm = scope
            .spawn(move || {
                let _turn = DecisionSlot::new(1).enter();
                plugin.is_warm()
            })
            .join();
        drop(read);
        warm.expect("the second thread")
    });
    assert!(
        warm_behind_it,
        "decider 1 is warm on the second thread: the first left its state at the rest"
    );
    assert_eq!(
        plugin.stats().instantiations,
        1,
        "the warm-up took the load's own instance"
    );
}

/// A named decider's kept instance is its name's, on whichever thread acts
/// as it. Three threads act as one decider in turn: the first decides, the
/// second is told a dispatch and decides, the third decides, and each pick
/// names the row at the count the instance heard. A second name has an
/// instance of its own, which heard nothing. The apply's own instance went
/// to the first decider that needed one: the instances made are the deciders.
#[test]
fn a_named_deciders_instance_is_its_names_on_whichever_thread_acts_as_it() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "named",
        guest::record_guest(),
        mode::COUNT_DISPATCHES,
    );
    let engine = plugin_engine(&plugin);
    // One thread a call, acting as the decider of that number.
    let acting = |number: u8, dispatch: bool| {
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let _turn = DecisionSlot::new(number).enter();
                    if dispatch {
                        engine.on_dispatched(
                            &Lease::default(),
                            &Target::default(),
                            Side::Single,
                            3,
                        );
                    }
                    pick_ids(&decide(&engine, &pool(4), 1)).first().copied()
                })
                .join()
                .expect("the acting thread")
        })
    };
    assert_eq!(acting(0, false), Some(100), "nothing told yet");
    assert_eq!(
        acting(0, true),
        Some(101),
        "told on a second thread, to the instance the first decided on"
    );
    assert_eq!(
        acting(0, false),
        Some(101),
        "and a third thread decides on it"
    );
    assert_eq!(
        acting(1, false),
        Some(100),
        "another name's instance heard nothing"
    );
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.instantiations,
            stats.lazy_instantiations,
            stats.observed,
            stats.events_dropped
        ),
        (2, 1, 1, 0),
        "two deciders, two instances, the apply's own among them: {stats:?}"
    );
    assert!(!plugin.is_warm(), "the loading thread kept none");
}

/// A thread that decides under no name and also acts as a named decider
/// keeps the two apart: its own instance stands aside while it acts, and is
/// as it left it after.
#[test]
fn a_threads_own_instance_stands_aside_while_it_acts_as_a_named_decider() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "aside",
        guest::record_guest(),
        mode::COUNT_DISPATCHES,
    );
    let engine = plugin_engine(&plugin);
    let pick = || pick_ids(&decide(&engine, &pool(4), 1)).first().copied();
    let dispatch = || engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    dispatch();
    assert_eq!(pick(), Some(101), "the thread's own instance heard it");
    {
        let _turn = DecisionSlot::new(5).enter();
        assert_eq!(pick(), Some(100), "the named decider's heard nothing");
        dispatch();
        dispatch();
        assert_eq!(pick(), Some(102), "and hears what is told as it");
    }
    assert_eq!(pick(), Some(101), "the thread's own is as it left it");
    assert_eq!(plugin.stats().instantiations, 2);
}

/// A thread that names no decider and never decided with the plugin only
/// tells: its lease event has no instance to reach and makes none. The
/// event is dropped and counted, and the plugin's first drop writes its one
/// warning. The thread that loaded is told as before: the apply's instance
/// is its own.
#[test]
fn a_thread_that_only_tells_makes_no_instance_and_its_events_are_dropped() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "only-told",
        guest::record_guest(),
        mode::COUNT_DISPATCHES,
    );
    let engine = plugin_engine(&plugin);
    let tell = || engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                tell();
                tell();
            })
            .join()
            .expect("the telling thread");
    });
    let stats = plugin.stats();
    assert_eq!(
        (stats.events_dropped, stats.observed, stats.instantiations),
        (2, 0, 1),
        "dropped, and no instance made: {stats:?}"
    );
    let said = warnings(plugin.name());
    assert_eq!(
        said.iter()
            .filter(|line| line.contains("names no decider"))
            .count(),
        1,
        "one warning a plugin, at its first drop: {said:?}"
    );
    tell();
    let stats = plugin.stats();
    assert_eq!(
        (stats.events_dropped, stats.observed, stats.instantiations),
        (2, 1, 1),
        "the loading thread is told: {stats:?}"
    );
    assert_eq!(
        pick_ids(&decide(&engine, &pool(4), 1)).first().copied(),
        Some(101),
        "and decides on the instance that heard"
    );
}

/// A thread that decided or warmed under no name is a decider of its own:
/// what it tells reaches the instance it decides on.
#[test]
fn a_thread_that_decided_or_warmed_under_no_name_hears_what_it_tells() {
    let runtime = runtime();
    // On a thread that did not load: the pick before the dispatch (none
    // where the thread warmed first), the pick after it, the drops.
    let heard_by = |name: &str, warm_first: bool| {
        let plugin = load(
            &runtime,
            name,
            guest::record_guest(),
            mode::COUNT_DISPATCHES,
        );
        let engine = plugin_engine(&plugin);
        let picks = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let pick = || pick_ids(&decide(&engine, &pool(4), 1)).first().copied();
                    let before = if warm_first {
                        warmed(&plugin);
                        None
                    } else {
                        pick()
                    };
                    engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
                    (before, pick())
                })
                .join()
                .expect("the thread")
        });
        (picks, plugin.stats().events_dropped)
    };
    assert_eq!(
        heard_by("own-decided", false),
        ((Some(100), Some(101)), 0),
        "it decided first"
    );
    assert_eq!(
        heard_by("own-warmed", true),
        ((None, Some(101)), 0),
        "it warmed first"
    );
}

/// A pair decided as a named decider keeps the decider's instance on the
/// thread until the plugin's last close of the decision: each side's
/// resource is dropped on it. So two pairs decided by two threads that act
/// as one decider use the fuel two pairs use under no name, to the unit, on
/// one instance.
#[test]
fn a_named_pair_decision_keeps_its_instance_to_the_plugins_last_close() {
    let runtime = runtime();
    // The fuel of two pair decisions, one after the other, and the
    // instances made: on this thread under no name, or on two threads that
    // act as one decider in turn.
    let two_pairs = |name: &str, named: bool| {
        let plugin = load(&runtime, name, guest::record_guest(), mode::LEAST_IN_FLIGHT);
        let prefill = plugin_profile(&plugin, None, None, None);
        let decode = plugin_profile(&plugin, None, None, None);
        let engine = pair_engine(vec![prefill, decode], 1);
        let loaded = plugin.stats().fuel_used;
        let pair = || {
            let decision = pair_decide(&engine);
            assert!(decision.pick().is_some(), "a pair pick, not {decision:?}");
            plugin.stats().fuel_used
        };
        let acting = || {
            std::thread::scope(|scope| {
                scope
                    .spawn(|| {
                        let _turn = DecisionSlot::new(4).enter();
                        pair()
                    })
                    .join()
                    .expect("the acting thread")
            })
        };
        let first = if named { acting() } else { pair() };
        let second = if named { acting() } else { pair() };
        (
            first - loaded,
            second - first,
            plugin.stats().instantiations,
        )
    };
    let under_no_name = two_pairs("pair-own", false);
    assert_eq!(under_no_name.2, 1);
    assert_eq!(
        two_pairs("pair-named", true),
        under_no_name,
        "the same calls, the two destructors of each pair among them, on one instance"
    );
}

/// A warm-up whose telling faults the instance leaves no mark for a decision:
/// the next decision on that thread makes an instance at its open and is the
/// plugin's own, not the fallback's.
#[test]
fn a_warm_up_whose_telling_faults_leaves_the_next_decision_the_plugins_own() {
    use smg_grand_central::tracking::Membership;
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "cold-warm",
        guest::record_guest(),
        mode::TRAP_ON_FIVE_MEMBERS,
    );
    let engine = plugin_engine(&plugin);
    let five: Vec<WorkerId> = (100..105).map(WorkerId).collect();
    engine.on_membership(&Membership::Snapshot(&five));
    // The warm-up tells the snapshot of five and this guest traps on it: the
    // fault is the breaker's, and the slot is cold.
    plugin.warm().expect("the warm-up's own work is done");
    let warmed = plugin.stats();
    assert_eq!((warmed.traps, warmed.observed), (1, 1), "{warmed:?}");
    assert!(!plugin.is_warm(), "the faulted instance is gone");
    // Two members go. The decision makes an instance at its open, which
    // hears the three that stand, and its pick names the row at that count:
    // not the row the fallback picks for this pool and seed.
    for id in [104, 103] {
        engine.on_membership(&Membership::Removed(WorkerId(id)));
    }
    let picked = pick_ids(&decide(&engine, &pool(4), 1)).first().copied();
    let decided = plugin.stats();
    assert_eq!(picked, Some(103), "the plugin's own pick: {decided:?}");
    assert_eq!(
        decided.observed - warmed.observed,
        1,
        "the instance made at the open hears one snapshot: {decided:?}"
    );
    assert!(decided.answered > warmed.answered, "{decided:?}");
}

/// A thread under no name that came by its first instance at a close's
/// rebuild (its first `init` ended by the decision's clamp) is a decider of
/// its own from there: a lease's event it tells is heard, not dropped.
#[test]
fn a_thread_whose_first_instance_the_close_rebuilt_hears_what_it_tells() {
    tried(a_thread_whose_first_instance_the_close_rebuilt_hears_what_it_tells_in_a_try);
}

/// One try of the case above, told its number: whether it stood.
fn a_thread_whose_first_instance_the_close_rebuilt_hears_what_it_tells_in_a_try(made: u32) -> bool {
    const CASE: &str = "a first instance the close rebuilt";
    // The host's warnings are kept from here on: a rebuild that fails says why.
    said();
    let began = TRY_BEGAN.get();
    let runtime = runtime();
    let heavy = load(
        &runtime,
        "rebuilt-first",
        guest::record_guest(),
        mode::BURN_IN_INIT,
    );
    let engine = plugin_engine(&heavy);
    // A thread that did not load and was not warmed: the load's instance is
    // not its to take, and its first `init` runs under a decision's clamp.
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                TRY_BEGAN.set(began);
                let (wall, cpu) = (std::time::Instant::now(), bound_clock());
                let mut request_facts = facts(1);
                request_facts.budget_micros = 200;
                let _ = engine.decide(
                    &request_facts,
                    &four(),
                    &pool(4),
                    &Constraints::default(),
                    &mut Scratch::new(),
                );
                let held = off_cpu_since(wall, cpu);
                if !the_close_rebuilt(CASE, &heavy, held, made) {
                    return false;
                }
                let rebuilt = heavy.stats();
                assert_eq!(rebuilt.budget_spent, 1, "the clamp ended it: {rebuilt:?}");
                // Three tellings: after the rebuild, after a second decision
                // on that instance, and after a warm-up there.
                let tell = || {
                    engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
                };
                tell();
                let _ = decide(&engine, &pool(4), 2);
                tell();
                heavy.warm().expect("the thread is warm already");
                tell();
                let told = heavy.stats();
                assert_eq!(
                    (told.observed - rebuilt.observed, told.events_dropped),
                    (3, 0),
                    "the thread is a decider of its own: {told:?}"
                );
                true
            })
            .join()
            .expect("the thread that did not load")
    })
}

/// An engine over the plugin whose decision, after the plugin's scope
/// opened, runs a scope of the host that acts inside it.
fn engine_that(plugin: &Arc<Plugin>, tells: &Arc<Engine>, does: Inside) -> Engine {
    single_engine(profile_that(plugin, tells, does))
}

/// What a thread still holds of one named decider goes back to that
/// decider's rest before the thread acts as another: the first decider's
/// instance, with what it heard, is there for whoever acts as it next.
#[test]
fn a_thread_that_turns_to_another_name_gives_the_held_one_back_first() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "turns",
        guest::record_guest(),
        mode::COUNT_DISPATCHES,
    );
    let plain = Arc::new(plugin_engine(&plugin));
    let turning = engine_that(&plugin, &plain, Inside::WarmsAs(DecisionSlot::new(6)));
    let first = DecisionSlot::new(5);
    {
        let _turn = first.enter();
        for _ in 0..2 {
            plain.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
        }
        // The plugin's scope brings the first decider's instance, and the
        // host's scope then acts as the second, inside the decision.
        let _ = decide(&turning, &pool(4), 1);
    }
    let picked = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let _turn = first.enter();
                pick_ids(&decide(&plain, &pool(4), 2)).first().copied()
            })
            .join()
            .expect("the thread that acts as the first decider next")
    });
    let stats = plugin.stats();
    assert_eq!(picked, Some(102), "the instance that heard two: {stats:?}");
    // The warm-up inside the decision ran on this thread: one made again is
    // an instance more.
    assert_eq!(
        stats.instantiations,
        2 + WARMED_AGAIN.get(),
        "one a decider: {stats:?}"
    );
}

/// A thread that still holds a named decider's instance when its name ends
/// gives it back before it acts under no name: what it then tells reaches
/// its own instance and not the decider's.
#[test]
fn a_thread_whose_name_ends_gives_the_held_decider_back_before_it_acts_as_itself() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "ends",
        guest::record_guest(),
        mode::COUNT_DISPATCHES,
    );
    let plain = Arc::new(plugin_engine(&plugin));
    let ending = engine_that(&plugin, &plain, Inside::TellsUnderNoName);
    let pick = |seed| pick_ids(&decide(&plain, &pool(4), seed)).first().copied();
    // The thread is a decider of its own: its instance heard nothing.
    assert_eq!(pick(1), Some(100));
    let first = DecisionSlot::new(5);
    HELD_NAME.set(Some(first.enter()));
    for _ in 0..2 {
        plain.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    }
    // The plugin's scope brings the first decider's instance. The host's
    // scope then ends the name and tells a dispatch under none.
    let _ = decide(&ending, &pool(4), 2);
    assert_eq!(DecisionSlot::acting(), None, "the scope ended the name");
    assert_eq!(pick(3), Some(101), "the thread's own instance heard it");
    let as_first = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let _turn = first.enter();
                pick(4)
            })
            .join()
            .expect("the thread that acts as the first decider")
    });
    assert_eq!(as_first, Some(102), "its two dispatches and no more");
}

/// Both opens of a decision run on the instance its bring brought. A
/// thread acts as a decider whose instance heard two dispatches, and
/// decides a pair. Between the pair's two opens another thread takes a
/// whole turn as that decider: the rest then names that thread's place and
/// keeps an instance that heard none. The pair's second side still decides
/// on the instance its first side brought. An open that checked that hold
/// would find it stale, drop it and decide on the instance at rest.
#[test]
fn both_opens_of_a_pair_decide_on_the_instance_its_bring_brought() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "one-pair",
        guest::record_guest(),
        mode::COUNT_DISPATCHES,
    );
    let plain = Arc::new(plugin_engine(&plugin));
    let first = DecisionSlot::new(5);
    // The first side's profile: the plugin's scope opens and brings the
    // decider's instance, then the host's scope lets another thread take a
    // turn as that decider. The second side's open follows it.
    let prefill = profile_that(&plugin, &plain, Inside::TurnsElsewhereAs(first));
    let decode = plugin_profile(&plugin, None, None, None);
    let engine = pair_engine(vec![prefill, decode], 1);
    let _turn = first.enter();
    for _ in 0..2 {
        plain.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    }
    let decision = pair_decide(&engine);
    let Some(pick) = decision.pick() else {
        panic!("a pair pick, not {decision:?}")
    };
    let lead = pick.targets.first().expect("a target");
    let stats = plugin.stats();
    assert_eq!(
        (lead.id, lead.pair.map(|p| p.decode_id)),
        (WorkerId(102), Some(WorkerId(202))),
        "each side counted the two dispatches its instance heard: {stats:?}"
    );
    assert_eq!(
        stats.instantiations, 2,
        "the other thread's turn made an instance of its own: {stats:?}"
    );
}

/// A thread that acted under a name alone is no decider of its own: what it
/// tells under no name afterwards is dropped and counted, and makes no
/// instance.
#[test]
fn a_thread_that_acted_under_a_name_alone_only_tells_under_none() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "named-alone",
        guest::record_guest(),
        mode::COUNT_DISPATCHES,
    );
    let engine = plugin_engine(&plugin);
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                {
                    let _turn = DecisionSlot::new(3).enter();
                    let _ = decide(&engine, &pool(4), 1);
                }
                let decided = plugin.stats();
                engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
                let told = plugin.stats();
                assert_eq!(
                    (
                        told.events_dropped,
                        told.observed - decided.observed,
                        told.instantiations
                    ),
                    (1, 0, 1),
                    "dropped and counted, no instance made: {told:?}"
                );
            })
            .join()
            .expect("the thread that acted under a name alone");
    });
}

/// A named decider's latency window is one, on whichever thread acts as it:
/// the ring rests with the instance. Sixty-three slow samples taken on one
/// thread and the sixty-fourth, a warm decision's on another, read the
/// window slow: the ring came with the instance and went back with it at
/// every turn.
#[test]
fn a_named_deciders_latency_window_is_one_on_whichever_thread_acts_as_it() {
    const CALIBRATION: u64 = 10;
    let runtime = runtime();
    let mut row = spec("ring-rests", guest::record_guest(), mode::BURN_IN_INIT);
    row.fuel_per_micro = CALIBRATION;
    let plugin = loaded(&runtime, row);
    let init_micros = u32::try_from(plugin.stats().fuel_used.div_ceil(CALIBRATION)).expect("fits");
    let engine = plugin_engine(&plugin);
    // A cold decision's `init` is slow by its own fuel. A warm decision
    // brings the ring once and gives it back once.
    let decide_as_eight = |seed: u64, cold: bool| {
        let _turn = DecisionSlot::new(8).enter();
        if cold {
            plugin.forget_instance();
        }
        let mut request_facts = facts(seed);
        request_facts.budget_micros = init_micros / 3 * 4;
        let decision = engine.decide(
            &request_facts,
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert!(matches!(decision, Decision::Pick(_)), "decision {seed}");
    };
    for seed in 1..=63 {
        decide_as_eight(seed, true);
    }
    let short = plugin.stats();
    assert_eq!(short.breaker_opens, 0, "one sample short: {short:?}");
    std::thread::scope(|scope| {
        scope
            .spawn(|| decide_as_eight(64, false))
            .join()
            .expect("the second thread");
    });
    let s = plugin.stats();
    assert_eq!(
        s.breaker_opens, 1,
        "the sixty-fourth sample, on another thread, completes the window: {s:?}"
    );
}

/// The thread that loaded is told as before, also after it forgot the
/// load's instance: the next event it tells makes an instance and is heard.
#[test]
fn the_thread_that_loaded_hears_what_it_tells_after_it_forgot_its_instance() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "forgot",
        guest::record_guest(),
        mode::COUNT_DISPATCHES,
    );
    let engine = plugin_engine(&plugin);
    plugin.forget_instance();
    engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    let told = plugin.stats();
    assert_eq!(
        (told.observed, told.events_dropped, told.lazy_instantiations),
        (1, 0, 1),
        "an instance made at the telling, and told: {told:?}"
    );
}

/// Under a name the load's instance warms no decider by waiting: it is the
/// first asker's. A warm-up that asks before it warms makes an instance for
/// every decider, the first by taking the load's.
#[test]
fn a_warm_up_that_asks_first_warms_every_named_decider() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "asks-first",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    assert!(plugin.is_warm(), "the thread that loaded, by the load's");
    for name in 0..3 {
        let _turn = DecisionSlot::new(name).enter();
        assert!(!plugin.is_warm(), "decider {name} holds none yet");
        plugin.warm().expect("it warms");
        assert!(plugin.is_warm(), "decider {name} holds its own");
    }
    let stats = plugin.stats();
    assert_eq!(
        stats.instantiations, 3,
        "one a decider, the load's among them: {stats:?}"
    );
    assert!(!plugin.is_warm(), "the load's instance went to a decider");
}
