//! The scopes on a decision's way out, the unwinding one included: a scope
//! that panics at its open or its close, on a single decision and on a
//! pair's two sides, a caught payload, and the plugin's own open, close and
//! undo that stop half way.

use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc,
};

use smg_grand_central::{
    engine::{
        Candidate, Constraints, Engine, PoolFacts, Profile, RequestFacts, Scratch, WeightedScorer,
    },
    plugins::{
        shared::{BreakerSpec, OpenDecisions},
        wasm::{Plugin, Runtime, WasmScorer},
    },
    stages::{Best, LeastRequests, Lifecycle},
};

use crate::{
    bound_clock, decide, facts, load, loaded, made_again, off_cpu_since, pair_decide, pair_engine,
    pick_ids, plugin_profile, pool, runtime, single_engine, spec,
    support::wasm_guest::{self as guest, mode},
    tried, INIT_FRAME, SLOW_CALIBRATION, SLOW_FRAME_MICROS, TRIES,
};

/// A stage of the host's own whose scope panics at its open, once.
#[derive(Debug)]
struct PanicsAtItsOpen(AtomicBool);

impl smg_grand_central::engine::Filter for PanicsAtItsOpen {
    fn name(&self) -> &'static str {
        "panics-at-its-open"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn smg_grand_central::engine::DecisionScope> {
        Some(self)
    }
}

impl smg_grand_central::engine::DecisionScope for PanicsAtItsOpen {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {
        assert!(
            !self.0.swap(false, Ordering::Relaxed),
            "this scope's open panics, once"
        );
    }

    fn close(&self) {}
}

/// The engine closes a decision on every path, the unwinding one included:
/// a scope that panics at its open leaves the scopes opened before it
/// closed. A host that catches the panic keeps its thread, and the thread's
/// next decision is a decision of its own, numbered and seeded anew. Left
/// open, the first decision would hand the next its number, and the plugin
/// that failed it would read every later decision as failed.
#[test]
fn a_scope_that_panics_at_its_open_leaves_the_threads_next_decision_its_own() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "beside-a-panic",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    // The plugin's scope opens first, then the one that panics.
    let mut profile = plugin_profile(&plugin, None, None, None);
    profile
        .filters
        .push(Box::new(PanicsAtItsOpen(AtomicBool::new(true))));
    let engine = single_engine(profile);
    let candidates = pool(3);
    let mut pool_facts = PoolFacts::default();
    pool_facts.pool_size = 3;
    pool_facts.generation = 1;
    // A budget gone on arrival: the plugin fails this decision at its open.
    let mut gone = facts(1);
    gone.budget_micros = 0;
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.decide(
            &gone,
            &pool_facts,
            &candidates,
            &Constraints::default(),
            &mut Scratch::new(),
        )
    }));
    assert!(caught.is_err(), "the second scope's open panicked");
    let failed = plugin.stats();
    assert_eq!(failed.budget_gone, 1, "{failed:?}");
    let next = decide(&engine, &candidates, 2);
    let asked = plugin.stats();
    assert_eq!(
        (asked.budget_gone, asked.answered - failed.answered),
        (1, 4),
        "the next decision asked the constructor and the three stages: {asked:?}"
    );
    assert_eq!(
        pick_ids(&next).first(),
        Some(&102),
        "the plugin's pick leads: the least in flight"
    );
}

/// What a scope of the host's own saw: its opens and its closes.
#[derive(Debug, Default)]
struct ScopeCounts {
    opens: AtomicU32,
    closes: AtomicU32,
}

impl ScopeCounts {
    fn read(&self) -> (u32, u32) {
        (
            self.opens.load(Ordering::Relaxed),
            self.closes.load(Ordering::Relaxed),
        )
    }
}

/// A stage of the host's own whose scope stores at its open and takes what
/// it stored at its close: a close with nothing stored panics. Told to, its
/// open panics before its store, once. It counts its opens and its closes.
#[derive(Debug)]
struct StoresAtItsOpen {
    counts: Arc<ScopeCounts>,
    stored: AtomicBool,
    panics_once: AtomicBool,
}

impl StoresAtItsOpen {
    fn stage(panics_once: bool) -> (Box<Self>, Arc<ScopeCounts>) {
        let counts = Arc::new(ScopeCounts::default());
        let stage = Box::new(Self {
            counts: Arc::clone(&counts),
            stored: AtomicBool::new(false),
            panics_once: AtomicBool::new(panics_once),
        });
        (stage, counts)
    }
}

impl smg_grand_central::engine::Filter for StoresAtItsOpen {
    fn name(&self) -> &'static str {
        "stores-at-its-open"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn smg_grand_central::engine::DecisionScope> {
        Some(self)
    }
}

impl smg_grand_central::engine::DecisionScope for StoresAtItsOpen {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {
        self.counts.opens.fetch_add(1, Ordering::Relaxed);
        assert!(
            !self.panics_once.swap(false, Ordering::Relaxed),
            "this scope's open panics before its store, once"
        );
        self.stored.store(true, Ordering::Relaxed);
    }

    fn close(&self) {
        self.counts.closes.fetch_add(1, Ordering::Relaxed);
        assert!(
            self.stored.swap(false, Ordering::Relaxed),
            "a close with nothing stored"
        );
    }
}

/// The engine closes a scope only after its open returned. When a scope
/// panics at its open, the scopes whose open returned are closed, and
/// neither that scope nor any after it: an open that stops half-way undoes
/// itself. So a host's scope whose close takes what its open stored is never
/// closed with nothing stored. Such a close would panic inside the first
/// panic's unwinding and end the process; here the first panic reaches the
/// host's catch.
#[test]
fn a_scope_is_closed_only_after_its_open_returned() {
    let (before, opened_before) = StoresAtItsOpen::stage(false);
    let (panics, panicked) = StoresAtItsOpen::stage(true);
    let (after, never_reached) = StoresAtItsOpen::stage(false);
    let filters: Vec<Box<dyn smg_grand_central::engine::Filter>> = vec![before, panics, after];
    let mut profile = Profile::of("host-scopes".into(), Box::new(Best));
    profile.filters = filters;
    profile.scorers = vec![WeightedScorer::of(Box::new(LeastRequests), 1.0)];
    let engine = single_engine(profile);
    let candidates = pool(3);
    let mut pool_facts = PoolFacts::default();
    pool_facts.pool_size = 3;
    pool_facts.generation = 1;
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.decide(
            &facts(1),
            &pool_facts,
            &candidates,
            &Constraints::default(),
            &mut Scratch::new(),
        )
    }));
    assert!(
        caught.is_err(),
        "the second scope's panic reached the catch"
    );
    assert_eq!(opened_before.read(), (1, 1), "its open returned: closed");
    assert_eq!(
        panicked.read(),
        (1, 0),
        "its open did not return: not closed"
    );
    assert_eq!(never_reached.read(), (0, 0), "never opened: not closed");
    // The next decision opens and closes all three.
    let next = decide(&engine, &candidates, 2);
    assert!(!pick_ids(&next).is_empty(), "the next decision picks");
    assert_eq!(opened_before.read(), (2, 2));
    assert_eq!(panicked.read(), (2, 1));
    assert_eq!(never_reached.read(), (1, 1));
}

/// A stage of the host's own whose scope's close panics, once. It counts its
/// opens and its closes.
#[derive(Debug)]
struct PanicsAtItsClose {
    counts: Arc<ScopeCounts>,
    panics_once: AtomicBool,
}

impl smg_grand_central::engine::Filter for PanicsAtItsClose {
    fn name(&self) -> &'static str {
        "panics-at-its-close"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn smg_grand_central::engine::DecisionScope> {
        Some(self)
    }
}

impl smg_grand_central::engine::DecisionScope for PanicsAtItsClose {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {
        self.counts.opens.fetch_add(1, Ordering::Relaxed);
    }

    fn close(&self) {
        self.counts.closes.fetch_add(1, Ordering::Relaxed);
        assert!(
            !self.panics_once.swap(false, Ordering::Relaxed),
            "this scope's close panics, once"
        );
    }
}

/// The engine closes the scopes whose open returned inside the unwinding of
/// a later scope's open, and a close that panics there stops at the engine's
/// own catch: the scopes after it are still closed, and the open's panic is
/// the one the host catches. A second panic let out inside that unwinding
/// would end the process.
#[test]
fn a_close_that_panics_inside_an_opens_unwinding_stops_at_the_engine() {
    let close_counts = Arc::new(ScopeCounts::default());
    let close_panics = Box::new(PanicsAtItsClose {
        counts: Arc::clone(&close_counts),
        panics_once: AtomicBool::new(true),
    });
    let (after_it, closed_after_it) = StoresAtItsOpen::stage(false);
    let (panics, panicked) = StoresAtItsOpen::stage(true);
    let filters: Vec<Box<dyn smg_grand_central::engine::Filter>> =
        vec![close_panics, after_it, panics];
    let mut profile = Profile::of("host-scopes".into(), Box::new(Best));
    profile.filters = filters;
    profile.scorers = vec![WeightedScorer::of(Box::new(LeastRequests), 1.0)];
    let engine = single_engine(profile);
    let candidates = pool(3);
    let mut pool_facts = PoolFacts::default();
    pool_facts.pool_size = 3;
    pool_facts.generation = 1;
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.decide(
            &facts(1),
            &pool_facts,
            &candidates,
            &Constraints::default(),
            &mut Scratch::new(),
        )
    }));
    let payload = caught.expect_err("the third scope's panic reached the catch");
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("this scope's open panics before its store, once"),
        "the open's panic is the one the host sees"
    );
    assert_eq!(
        close_counts.read(),
        (1, 1),
        "closed once, and its close panicked"
    );
    assert_eq!(
        closed_after_it.read(),
        (1, 1),
        "the scope after it is still closed"
    );
    assert_eq!(
        panicked.read(),
        (1, 0),
        "its open did not return: not closed"
    );
    // The next decision opens and closes all three.
    let next = decide(&engine, &candidates, 2);
    assert!(!pick_ids(&next).is_empty(), "the next decision picks");
    assert_eq!(close_counts.read(), (2, 2));
    assert_eq!(closed_after_it.read(), (2, 2));
    assert_eq!(panicked.read(), (2, 1));
}

/// A profile of the host's own stages, in their order.
fn host_profile(filters: Vec<Box<dyn smg_grand_central::engine::Filter>>) -> Profile {
    let mut profile = Profile::of("host-scopes".into(), Box::new(Best));
    profile.filters = filters;
    profile.scorers = vec![WeightedScorer::of(Box::new(LeastRequests), 1.0)];
    profile
}

/// A scope whose close panics, once, with its counts.
fn close_panics_once() -> (Box<PanicsAtItsClose>, Arc<ScopeCounts>) {
    let counts = Arc::new(ScopeCounts::default());
    let stage = Box::new(PanicsAtItsClose {
        counts: Arc::clone(&counts),
        panics_once: AtomicBool::new(true),
    });
    (stage, counts)
}

/// A stage of the host's own whose scope's close panics once with its
/// number as the payload, so a case can tell two such panics apart.
#[derive(Debug)]
struct ClosePanicsWith(u32, AtomicBool);

impl smg_grand_central::engine::Filter for ClosePanicsWith {
    fn name(&self) -> &'static str {
        "close-panics-with"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn smg_grand_central::engine::DecisionScope> {
        Some(self)
    }
}

impl smg_grand_central::engine::DecisionScope for ClosePanicsWith {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}

    fn close(&self) {
        if self.1.swap(false, Ordering::Relaxed) {
            // A panic with a payload of the case's own type; the hook stays quiet.
            std::panic::resume_unwind(Box::new(self.0));
        }
    }
}

/// A stage of the host's own, with no scope, whose `keep` panics once.
#[derive(Debug)]
struct PanicsAtItsKeep(AtomicBool);

impl smg_grand_central::engine::Filter for PanicsAtItsKeep {
    fn name(&self) -> &'static str {
        "panics-at-its-keep"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        assert!(
            !self.0.swap(false, Ordering::Relaxed),
            "this stage's keep panics, once"
        );
        true
    }
}

/// One decision of three candidates, its panic caught.
fn decide_caught(engine: &Engine) -> Box<dyn std::any::Any + Send> {
    let candidates = pool(3);
    let mut pool_facts = PoolFacts::default();
    pool_facts.pool_size = 3;
    pool_facts.generation = 1;
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.decide(
            &facts(1),
            &pool_facts,
            &candidates,
            &Constraints::default(),
            &mut Scratch::new(),
        )
    }))
    .expect_err("the decision panicked")
}

/// A close that panics on a decision's own way out keeps no scope after it
/// open: the engine runs every close it owes, and the first close's panic is
/// the one the host sees, once the rest are closed.
#[test]
fn a_close_that_panics_alone_keeps_no_scope_after_it_open() {
    let first = Box::new(ClosePanicsWith(1, AtomicBool::new(true)));
    let (after_it, closed_after_it) = StoresAtItsOpen::stage(false);
    let second = Box::new(ClosePanicsWith(2, AtomicBool::new(true)));
    let engine = single_engine(host_profile(vec![first, after_it, second]));
    let payload = decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<u32>().copied(),
        Some(1),
        "the first close's panic is the one the host sees"
    );
    assert_eq!(
        closed_after_it.read(),
        (1, 1),
        "the scope after the first panic is closed"
    );
    // Both closes that panic did so in that decision: the next one closes
    // clean, so the third scope's close ran too.
    let next = decide(&engine, &pool(3), 2);
    assert!(!pick_ids(&next).is_empty(), "the next decision picks");
    assert_eq!(closed_after_it.read(), (2, 2));
}

/// A close that panics while a stage's panic unwinds stops at the engine:
/// the scopes after it are still closed, and the stage's panic is the one
/// the host sees. Let out inside that unwinding, it would end the process.
#[test]
fn a_close_that_panics_while_a_stages_panic_unwinds_stops_at_the_engine() {
    let (close_panics, close_counts) = close_panics_once();
    let (after_it, closed_after_it) = StoresAtItsOpen::stage(false);
    let keep_panics = Box::new(PanicsAtItsKeep(AtomicBool::new(true)));
    let engine = single_engine(host_profile(vec![close_panics, after_it, keep_panics]));
    let payload = decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("this stage's keep panics, once"),
        "the stage's panic is the one the host sees"
    );
    assert_eq!(
        close_counts.read(),
        (1, 1),
        "closed, and its close panicked"
    );
    assert_eq!(
        closed_after_it.read(),
        (1, 1),
        "the scope after it is still closed"
    );
    let next = decide(&engine, &pool(3), 2);
    assert!(!pick_ids(&next).is_empty(), "the next decision picks");
    assert_eq!(close_counts.read(), (2, 2));
    assert_eq!(closed_after_it.read(), (2, 2));
}

/// The payload of a pair decision that panics.
fn pair_decide_caught(engine: &Engine) -> Box<dyn std::any::Any + Send> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pair_decide(engine)))
        .expect_err("the pair decision panicked")
}

/// A pair opens its prefill side whole before its decode side. When a decode
/// scope panics at its open, the prefill side's closes run inside that
/// unwinding, by the decision's own guard: one that panics there stops at
/// the engine, the scopes after it are still closed, and the open's panic is
/// the one the host sees.
#[test]
fn a_pairs_prefill_close_that_panics_while_its_decode_side_opens_stops_at_the_engine() {
    let (close_panics, close_counts) = close_panics_once();
    let (after_it, closed_after_it) = StoresAtItsOpen::stage(false);
    let (decode_panics, panicked) = StoresAtItsOpen::stage(true);
    let engine = pair_engine(
        vec![
            host_profile(vec![close_panics, after_it]),
            host_profile(vec![decode_panics]),
        ],
        1,
    );
    let payload = pair_decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("this scope's open panics before its store, once"),
        "the open's panic is the one the host sees"
    );
    assert_eq!(
        close_counts.read(),
        (1, 1),
        "the prefill scope is closed, and its close panicked"
    );
    assert_eq!(
        closed_after_it.read(),
        (1, 1),
        "the prefill scope after it is still closed"
    );
    assert_eq!(
        panicked.read(),
        (1, 0),
        "its open did not return: not closed"
    );
    // The next pair decision opens and closes all three.
    let next = pair_decide(&engine);
    assert!(next.pick().is_some(), "the next decision picks");
    assert_eq!(close_counts.read(), (2, 2));
    assert_eq!(closed_after_it.read(), (2, 2));
    assert_eq!(panicked.read(), (2, 1));
}

/// The same with a decode scope already open when the panic starts: the
/// opening guard closes that one, the decision's guard then runs the prefill
/// side's closes, and the one that panics there stops at the engine.
#[test]
fn a_pairs_prefill_close_that_panics_after_a_decode_scope_opened_stops_at_the_engine() {
    let (close_panics, close_counts) = close_panics_once();
    let (after_it, closed_after_it) = StoresAtItsOpen::stage(false);
    let (decode_opened, decode_counts) = StoresAtItsOpen::stage(false);
    let (decode_panics, panicked) = StoresAtItsOpen::stage(true);
    let engine = pair_engine(
        vec![
            host_profile(vec![close_panics, after_it]),
            host_profile(vec![decode_opened, decode_panics]),
        ],
        1,
    );
    let payload = pair_decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("this scope's open panics before its store, once"),
        "the open's panic is the one the host sees"
    );
    assert_eq!(
        decode_counts.read(),
        (1, 1),
        "the decode scope that opened is closed, once"
    );
    assert_eq!(
        close_counts.read(),
        (1, 1),
        "the prefill scope is closed, and its close panicked"
    );
    assert_eq!(
        closed_after_it.read(),
        (1, 1),
        "the prefill scope after it is still closed"
    );
    assert_eq!(
        panicked.read(),
        (1, 0),
        "its open did not return: not closed"
    );
    let next = pair_decide(&engine);
    assert!(next.pick().is_some(), "the next decision picks");
    assert_eq!(decode_counts.read(), (2, 2));
    assert_eq!(close_counts.read(), (2, 2));
    assert_eq!(closed_after_it.read(), (2, 2));
    assert_eq!(panicked.read(), (2, 1));
}

/// The close that panics on the decode side itself, inside the unwinding of
/// its own profile's open: the opening guard stops it, and the prefill side,
/// open whole, is closed by the decision's guard.
#[test]
fn a_pairs_decode_close_that_panics_inside_its_own_opens_unwinding_stops_at_the_engine() {
    let (prefill, prefill_counts) = StoresAtItsOpen::stage(false);
    let (close_panics, close_counts) = close_panics_once();
    let (decode_panics, panicked) = StoresAtItsOpen::stage(true);
    let engine = pair_engine(
        vec![
            host_profile(vec![prefill]),
            host_profile(vec![close_panics, decode_panics]),
        ],
        1,
    );
    let payload = pair_decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("this scope's open panics before its store, once"),
        "the open's panic is the one the host sees"
    );
    assert_eq!(
        close_counts.read(),
        (1, 1),
        "the decode scope is closed, and its close panicked"
    );
    assert_eq!(prefill_counts.read(), (1, 1), "the prefill side is closed");
    assert_eq!(
        panicked.read(),
        (1, 0),
        "its open did not return: not closed"
    );
    let next = pair_decide(&engine);
    assert!(next.pick().is_some(), "the next decision picks");
    assert_eq!(close_counts.read(), (2, 2));
    assert_eq!(prefill_counts.read(), (2, 2));
    assert_eq!(panicked.read(), (2, 1));
}

/// Both sides' closes panic on a pair decision's own way out. The decode
/// side closes first, so the panic of its close goes on once that side is
/// closed: the prefill side's closes run inside it, the one that panics
/// there stops at the engine, and the decode close's panic is the one the
/// host sees.
#[test]
fn a_pairs_prefill_close_that_panics_while_its_decode_sides_close_unwinds_stops_at_the_engine() {
    let (prefill_panics, prefill_counts) = close_panics_once();
    let (after_prefill, closed_after_prefill) = StoresAtItsOpen::stage(false);
    let decode_panics = Box::new(ClosePanicsWith(2, AtomicBool::new(true)));
    let (after_decode, closed_after_decode) = StoresAtItsOpen::stage(false);
    let engine = pair_engine(
        vec![
            host_profile(vec![prefill_panics, after_prefill]),
            host_profile(vec![decode_panics, after_decode]),
        ],
        1,
    );
    let payload = pair_decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<u32>().copied(),
        Some(2),
        "the decode close's panic is the one the host sees"
    );
    assert_eq!(
        closed_after_decode.read(),
        (1, 1),
        "the decode scope after the close that panicked is closed"
    );
    assert_eq!(
        prefill_counts.read(),
        (1, 1),
        "the prefill scope is closed, and its close panicked"
    );
    assert_eq!(
        closed_after_prefill.read(),
        (1, 1),
        "the prefill scope after it is still closed"
    );
    // The next pair decision opens and closes all four, and no close panics.
    let next = pair_decide(&engine);
    assert!(next.pick().is_some(), "the next decision picks");
    assert_eq!(closed_after_decode.read(), (2, 2));
    assert_eq!(prefill_counts.read(), (2, 2));
    assert_eq!(closed_after_prefill.read(), (2, 2));
}

/// A panic payload whose own destructor panics, with a payload whose
/// destructor panics in turn for as many drops as it counts.
struct DropPanics(u8);

impl Drop for DropPanics {
    fn drop(&mut self) {
        match self.0 {
            0 => std::panic::resume_unwind(Box::new("this payload's drop panics")),
            more => std::panic::resume_unwind(Box::new(DropPanics(more - 1))),
        }
    }
}

/// A stage of the host's own whose scope's close panics, once, with a
/// payload whose own drop panics, as does the drop of that panic's payload.
#[derive(Debug)]
struct ClosePanicsWithADrop(AtomicBool);

impl smg_grand_central::engine::Filter for ClosePanicsWithADrop {
    fn name(&self) -> &'static str {
        "close-panics-with-a-drop"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn smg_grand_central::engine::DecisionScope> {
        Some(self)
    }
}

impl smg_grand_central::engine::DecisionScope for ClosePanicsWithADrop {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}

    fn close(&self) {
        if self.0.swap(false, Ordering::Relaxed) {
            std::panic::resume_unwind(Box::new(DropPanics(1)));
        }
    }
}

/// The engine drops a payload it caught and does not resume under a catch
/// of its own: the payload's destructor can panic, and so can the
/// destructor of that panic's payload, which is forgotten. Let out inside
/// an unwinding, either panic would end the process; on a decision's own
/// way out it would take the first panic's place and leave the scopes
/// after it open.
#[test]
fn a_caught_payload_whose_drop_panics_stops_at_the_engine() {
    // Inside an open's unwinding: the closes of the scopes that had opened.
    let with_a_drop = Box::new(ClosePanicsWithADrop(AtomicBool::new(true)));
    let (after_it, closed_after_it) = StoresAtItsOpen::stage(false);
    let (open_panics, _) = StoresAtItsOpen::stage(true);
    let engine = single_engine(host_profile(vec![with_a_drop, after_it, open_panics]));
    let payload = decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("this scope's open panics before its store, once")
    );
    assert_eq!(closed_after_it.read(), (1, 1), "inside an open's unwinding");

    // Inside a stage's unwinding: the decision's own closes.
    let with_a_drop = Box::new(ClosePanicsWithADrop(AtomicBool::new(true)));
    let (after_it, closed_after_it) = StoresAtItsOpen::stage(false);
    let keep_panics = Box::new(PanicsAtItsKeep(AtomicBool::new(true)));
    let engine = single_engine(host_profile(vec![with_a_drop, after_it, keep_panics]));
    let payload = decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("this stage's keep panics, once")
    );
    assert_eq!(closed_after_it.read(), (1, 1), "inside a stage's unwinding");

    // On a decision's own way out: the first close's panic goes on, and the
    // second close's payload is released.
    let first = Box::new(ClosePanicsWith(1, AtomicBool::new(true)));
    let with_a_drop = Box::new(ClosePanicsWithADrop(AtomicBool::new(true)));
    let (after_it, closed_after_it) = StoresAtItsOpen::stage(false);
    let engine = single_engine(host_profile(vec![first, with_a_drop, after_it]));
    let payload = decide_caught(&engine);
    assert_eq!(payload.downcast_ref::<u32>().copied(), Some(1));
    assert_eq!(
        closed_after_it.read(),
        (1, 1),
        "on a decision's own way out"
    );
}

/// A panic payload that counts its drops, so a case can read whether the
/// engine freed it.
struct CountsItsDrop(Arc<AtomicU32>);

impl Drop for CountsItsDrop {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

/// A stage of the host's own whose scope's close panics, once, with a
/// payload that counts its drops.
#[derive(Debug)]
struct ClosePanicsWithACount(AtomicBool, Arc<AtomicU32>);

impl ClosePanicsWithACount {
    fn stage() -> (Box<Self>, Arc<AtomicU32>) {
        let drops = Arc::new(AtomicU32::new(0));
        let stage = Box::new(Self(AtomicBool::new(true), Arc::clone(&drops)));
        (stage, drops)
    }
}

impl smg_grand_central::engine::Filter for ClosePanicsWithACount {
    fn name(&self) -> &'static str {
        "close-panics-with-a-count"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn smg_grand_central::engine::DecisionScope> {
        Some(self)
    }
}

impl smg_grand_central::engine::DecisionScope for ClosePanicsWithACount {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}

    fn close(&self) {
        if self.0.swap(false, Ordering::Relaxed) {
            std::panic::resume_unwind(Box::new(CountsItsDrop(Arc::clone(&self.1))));
        }
    }
}

/// The engine frees a payload it caught and does not resume. Forgotten, a
/// fault that repeats would leave one payload allocated a decision, with
/// nothing to bound it.
#[test]
fn a_caught_payload_the_engine_does_not_resume_is_freed() {
    // Inside an open's unwinding: the closes of the scopes that had opened.
    let (with_a_count, drops) = ClosePanicsWithACount::stage();
    let (open_panics, _) = StoresAtItsOpen::stage(true);
    let engine = single_engine(host_profile(vec![with_a_count, open_panics]));
    let payload = decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("this scope's open panics before its store, once")
    );
    assert_eq!(
        drops.load(Ordering::Relaxed),
        1,
        "inside an open's unwinding"
    );

    // Inside a stage's unwinding: the decision's own closes.
    let (with_a_count, drops) = ClosePanicsWithACount::stage();
    let keep_panics = Box::new(PanicsAtItsKeep(AtomicBool::new(true)));
    let engine = single_engine(host_profile(vec![with_a_count, keep_panics]));
    let payload = decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("this stage's keep panics, once")
    );
    assert_eq!(
        drops.load(Ordering::Relaxed),
        1,
        "inside a stage's unwinding"
    );

    // On a decision's own way out: the first close's panic goes on, and the
    // second close's payload is freed.
    let first = Box::new(ClosePanicsWith(1, AtomicBool::new(true)));
    let (with_a_count, drops) = ClosePanicsWithACount::stage();
    let engine = single_engine(host_profile(vec![first, with_a_count]));
    let payload = decide_caught(&engine);
    assert_eq!(payload.downcast_ref::<u32>().copied(), Some(1));
    assert_eq!(
        drops.load(Ordering::Relaxed),
        1,
        "on a decision's own way out"
    );
}

/// The plugin's own open marks its phase open and counts the decision
/// before its instance's steps, and the engine does not close a scope whose
/// open did not return. So an open that stops half-way undoes itself: here
/// the host's clock panics inside it, and the thread's next decision is a
/// decision of its own that asks the plugin. Left half-open, the phase would
/// read as open to every later decision, and none would ask the plugin
/// again.
#[test]
fn a_plugins_open_that_stops_half_way_undoes_itself() {
    let clock_panics = Arc::new(AtomicBool::new(false));
    let told = Arc::clone(&clock_panics);
    let runtime = Arc::new(
        Runtime::with_clock(&OpenDecisions::new(), move || {
            assert!(
                !told.swap(false, Ordering::Relaxed),
                "the host's clock panics, once"
            );
            1_000_000
        })
        .expect("a runtime"),
    );
    let plugin = load(
        &runtime,
        "half-open",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let engine = single_engine(plugin_profile(&plugin, None, None, None));
    let candidates = pool(3);
    // A whole decision first: the slot is warm, and what stops the next open
    // is the clock alone.
    let first = decide(&engine, &candidates, 1);
    assert_eq!(pick_ids(&first).first(), Some(&102));
    let before = plugin.stats();
    clock_panics.store(true, Ordering::Relaxed);
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        decide(&engine, &candidates, 2)
    }));
    assert!(caught.is_err(), "the plugin's open panicked at the clock");
    assert_eq!(
        plugin.stats().answered,
        before.answered,
        "the open stopped before any call"
    );
    let next = decide(&engine, &candidates, 3);
    let asked = plugin.stats();
    assert_eq!(
        asked.answered - before.answered,
        4,
        "the next decision asked the constructor and the three stages: {asked:?}"
    );
    assert_eq!(
        pick_ids(&next).first(),
        Some(&102),
        "the plugin's pick leads: the least in flight"
    );
}

/// What the case below shares between its picker and the host's clock.
#[derive(Debug, Default)]
struct PastThePick {
    /// The case wants a close's read of the clock to panic.
    armed: AtomicBool,
    /// The decision's pick ran: its thread's next read of the clock is a
    /// close's.
    picked: AtomicBool,
}

/// The built-in best picker, telling the clock when it ran.
#[derive(Debug)]
struct TellsThePick(Arc<PastThePick>);

impl Lifecycle for TellsThePick {}

impl smg_grand_central::engine::Picker for TellsThePick {
    fn name(&self) -> &'static str {
        "tells-the-pick"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: smg_grand_central::engine::TieBand,
        out: &mut smg_grand_central::engine::Targets,
    ) {
        if self.0.armed.load(Ordering::Relaxed) {
            self.0.picked.store(true, Ordering::Relaxed);
        }
        smg_grand_central::engine::Picker::pick(&Best, facts, candidates, kept, scores, band, out);
    }
}

/// The plugin's own close runs each phase's destructor, the guest's code
/// under the host's, and a destructor that runs out reads the host's clock
/// for the breaker. So a close that stops half-way leaves its scope closed:
/// every phase stops reading as open, and its place in the thread's decision
/// is given back, before the first destructor runs, and a walk that does not
/// return forgets the slot it marked for the rebuild. Here the clock panics
/// at that read, and every later decision of the thread is a decision of its
/// own that asks the plugin. Left open, the phase would read as open to the
/// next decision: it would make no resource and ask the plugin nothing. With
/// its place not given back, the later decisions would be one decision to
/// the thread, under one number and one budget. The destructor runs out in
/// each, and a plugin that failed a decision is not asked again in it: the
/// second later decision would ask nothing.
///
/// The plugin is bound at one stage, so a decision closes its scope once.
/// Bound at two, the engine closes the same bound set twice, and the second
/// close forgets what the first left open.
///
/// A decision makes its instance under the init frame on the wall clock
/// when the close before it could not. A try in which a decision asked
/// nothing while its thread, or the thread of the decision before it, was
/// off the CPU half that frame is made again, [`TRIES`] tries at most: the
/// last try asserts whatever it read.
#[test]
fn a_plugins_close_that_stops_half_way_leaves_its_scope_closed() {
    const LATER: u64 = 8;
    let mut made = 0;
    loop {
        made += 1;
        let shared = Arc::new(PastThePick::default());
        let told = Arc::clone(&shared);
        let deciding = std::thread::current().id();
        let runtime = Arc::new(
            Runtime::with_clock(&OpenDecisions::new(), move || {
                // Past an armed pick the deciding thread reads the clock in
                // a close alone: that read panics, once.
                assert!(
                    std::thread::current().id() != deciding
                        || !told.picked.swap(false, Ordering::Relaxed),
                    "the host's clock panics inside a close"
                );
                1_000_000
            })
            .expect("a runtime"),
        );
        // A destructor that fills without end runs out of its frame, and
        // the fault's read of the clock is inside the close. At the slow
        // calibration that frame is a hundred units of fuel: the fill runs
        // out of fuel, not of time.
        let mut filler = spec(
            &format!("half-closed-{made}"),
            guest::record_guest(),
            mode::FILL_IN_DESTRUCTOR,
        );
        filler.fuel_per_micro = SLOW_CALIBRATION;
        filler.budget_micros = SLOW_FRAME_MICROS;
        // Every decision's destructor runs out: the failure rule is set out
        // of reach, so the breaker stays shut through the later decisions.
        filler.breaker = BreakerSpec::default();
        filler.breaker.failures = u32::MAX;
        let plugin = loaded(&runtime, filler);
        let mut profile = Profile::of(
            "one-stage".into(),
            Box::new(TellsThePick(Arc::clone(&shared))),
        );
        profile.filters = vec![];
        profile.scorers = vec![WeightedScorer::of(
            Box::new(WasmScorer::new(plugin.bind(), None)),
            1.0,
        )];
        let engine = single_engine(profile);
        let candidates = pool(3);
        shared.armed.store(true, Ordering::Relaxed);
        let stopped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            decide(&engine, &candidates, 1)
        }));
        shared.armed.store(false, Ordering::Relaxed);
        shared.picked.store(false, Ordering::Relaxed);
        assert!(
            stopped.is_err(),
            "the plugin's close panicked at the clock: {:?}",
            plugin.stats()
        );
        let mut silent = None;
        let mut before_it = std::time::Duration::ZERO;
        for later in 0..LATER {
            let answered = plugin.stats().answered;
            let (wall, cpu) = (std::time::Instant::now(), bound_clock());
            decide(&engine, &candidates, 2 + later);
            let off_the_cpu = off_cpu_since(wall, cpu);
            if plugin.stats().answered == answered {
                silent = Some((later, off_the_cpu.max(before_it)));
                break;
            }
            before_it = off_the_cpu;
        }
        let Some((later, off_the_cpu)) = silent else {
            break;
        };
        assert!(
            off_the_cpu >= INIT_FRAME / 2 && made < TRIES,
            "every later decision asks the plugin as a decision of its own: decision {later} after the stop asked nothing (try {made}, a thread off the CPU {off_the_cpu:?}): {:?}",
            plugin.stats()
        );
        made_again(
            "a close that stops half-way",
            &format!(
                "decision {later} after the stop asked nothing with a thread off the CPU {off_the_cpu:?}"
            ),
        );
    }
}

/// The plugin's own undo runs inside its open's unwinding too, by the walk
/// its close does. If that walk panics, it stops at the plugin's own catch
/// and the open's panic is the one the host catches: a second panic let out
/// inside that unwinding would end the process. Here the constructor traps,
/// which empties the slot, and the host's clock panics from then on: at the
/// fault's read inside the open, and again at the undo's rebuild of the slot.
#[test]
fn an_undo_that_panics_inside_its_opens_unwinding_stops_at_the_plugin() {
    tried(an_undo_that_panics_inside_its_opens_unwinding_stops_at_the_plugin_in_a_try);
}

/// One try of the case above, told its number: whether it stood. The
/// stopped undo leaves the slot empty, so the next decision makes its
/// instance inside its own frame, and a hold of the runner can end it there.
fn an_undo_that_panics_inside_its_opens_unwinding_stops_at_the_plugin_in_a_try(made: u32) -> bool {
    const CASE: &str = "an undo that panics inside its open's unwinding";
    let clock_panics = Arc::new(AtomicBool::new(true));
    let told = Arc::clone(&clock_panics);
    let loaded: Arc<std::sync::OnceLock<std::sync::Weak<Plugin>>> = Arc::default();
    let seen = Arc::clone(&loaded);
    let runtime = Arc::new(
        Runtime::with_clock(&OpenDecisions::new(), move || {
            let trapped = seen
                .get()
                .and_then(std::sync::Weak::upgrade)
                .is_some_and(|plugin| plugin.stats().traps > 0);
            assert!(
                !(trapped && told.load(Ordering::Relaxed)),
                "the host's clock panics"
            );
            1_000_000
        })
        .expect("a runtime"),
    );
    let plugin = load(
        &runtime,
        "undo-panics",
        guest::record_guest(),
        mode::TRAP_IN_CONSTRUCTOR,
    );
    loaded
        .set(Arc::downgrade(&plugin))
        .expect("the plugin is set once");
    let engine = single_engine(plugin_profile(&plugin, None, None, None));
    let candidates = pool(3);
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        decide(&engine, &candidates, 1)
    }));
    let payload = caught.expect_err("the clock's panic inside the open reached the catch");
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("the host's clock panics"),
        "the open's panic is the one the host sees"
    );
    assert_eq!(plugin.stats().traps, 1, "the constructor trapped once");
    // The clock mended, the thread's next decision is its own: it asks the
    // plugin, whose constructor traps again, and the fallbacks decide.
    clock_panics.store(false, Ordering::Relaxed);
    let (wall, cpu) = (std::time::Instant::now(), bound_clock());
    let next = decide(&engine, &candidates, 2);
    let held = off_cpu_since(wall, cpu);
    if plugin.stats().traps != 2 && held >= INIT_FRAME / 2 && made < TRIES {
        made_again(
            CASE,
            "the next decision's thread was off the CPU half the init frame, the bound of its cold slot's init",
        );
        return false;
    }
    assert!(!pick_ids(&next).is_empty(), "the next decision picks");
    assert_eq!(
        plugin.stats().traps,
        2,
        "the next decision asked the plugin"
    );
    true
}
