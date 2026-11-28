use std::{cell::Cell, collections::BTreeSet};

use openai_protocol::worker::WorkerStatus;
use smg_worker::{runtime::StallReason, test_support::ready_worker};

use super::*;
use crate::{
    document::spec::{Needs, PolicySpec},
    engine::{framework::InTheWay, list_cell::PublishedList},
};

const MODEL: &str = "model-a";

fn engine_of(document: &PolicySpec) -> InProcess {
    InProcess::new(AppliedSpec::validate(document).expect("the document validates"))
}

fn preset(name: &str) -> InProcess {
    let mut document = PolicySpec::preset(name).expect("a built-in preset");
    document.generation = 1;
    engine_of(&document)
}

fn pool(workers: usize) -> Arc<[Arc<dyn Worker>]> {
    (0..workers)
        .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
        .collect()
}

/// The profile of the model's single placements, which the engine picks
/// alone.
fn alone(engine: &InProcess) -> AloneProfile {
    match engine.spec().place_for(MODEL, Phase::Single) {
        Place::Alone(profile) => profile,
        other => panic!("not picked alone: {other:?}"),
    }
}

/// Round robin's preset with `filter` added to its profile's filters.
fn round_robin_with(filter: &str) -> InProcess {
    let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
    let profile = document
        .profiles
        .get_mut("round_robin")
        .expect("the preset's profile");
    profile.filter.push(filter.into());
    engine_of(&document)
}

/// What a host says of its workers when it hands their list is in the
/// list's records, each worker's at its place: the worker it named
/// takes images and no other does. The host is asked once a worker, in
/// the list's order, at the call. A list handed with nothing said says
/// of no worker that it takes images.
#[test]
fn a_lists_records_take_the_image_bit_its_host_says() {
    let engine = preset("round_robin");
    let (list, asked) = (pool(4), Cell::new(0));
    let members = engine
        .loaded()
        .pool_members_with_fixed_facts(&list, |worker| {
            asked.set(asked.get() + 1);
            FixedFacts {
                vision: Arc::ptr_eq(worker, &list[1]),
                ..FixedFacts::default()
            }
        });
    assert_eq!(asked.get(), 4, "asked at the call, once a worker");
    let sees = |members: &PoolMembers| {
        let fixed = members
            .records(engine.standing_spec(), &engine.clock)
            .fixed();
        fixed.iter().map(|row| row.vision).collect::<Vec<_>>()
    };
    assert_eq!(sees(&members), [false, true, false, false]);
    assert_eq!(asked.get(), 4, "and not again where the records are read");
    assert_eq!(sees(&engine.pool_members(&list)), [false; 4]);
}

/// The value made of a pool holds the shared list itself, and making it
/// reads no worker: a pool of workers that are not ready is made as any
/// other, and each pool made draws a start of its own.
#[test]
fn the_members_hold_the_shared_list() {
    let engine = preset("round_robin");
    let list = pool(5);
    for worker in list.iter() {
        worker.set_status(WorkerStatus::NotReady);
    }
    let members = engine.pool_members(&list);
    assert!(Arc::ptr_eq(members.workers(), &list));
    let again = engine.pool_members(&list);
    assert!(Arc::ptr_eq(again.workers(), &list));
    assert_ne!(
        members.rotation.0.load(Ordering::Relaxed),
        again.rotation.0.load(Ordering::Relaxed),
        "each pool made draws the place its turns start at"
    );
}

/// A place is found by the document's bindings, a model and a phase. A
/// profile's name finds nothing: a document that names its profile
/// anything is answered as the preset is. A model no binding names is
/// unbound, and so is a policy that serves every model (the empty id)
/// under a document that binds some models only.
#[test]
fn a_place_is_found_by_its_model_and_phase() {
    let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
    let profile = document
        .profiles
        .remove("round_robin")
        .expect("the preset's profile");
    document.profiles.insert("anything".to_string(), profile);
    document.bindings[0].profile = Some("anything".to_string());
    document.bindings[0].matcher.model = "served-*".to_string();
    let engine = engine_of(&document);
    assert!(matches!(
        engine.spec().place_for("served-a", Phase::Single),
        Place::Alone(_)
    ));
    assert_eq!(
        engine.spec().place_for("other", Phase::Single),
        Place::Unbound
    );
    assert_eq!(engine.spec().place_for("", Phase::Single), Place::Unbound);
    // The preset binds every model, the empty id among them.
    let every = preset("round_robin");
    assert_eq!(
        every.spec().place_for("", Phase::Single),
        every.spec().place_for(MODEL, Phase::Single)
    );
    assert!(matches!(
        every.spec().place_for("", Phase::Single),
        Place::Alone(_)
    ));
}

/// The places one pick decides: the picker places by itself, no stage
/// scores, and the filters are the two tests of a worker's health.
/// Round robin's preset is one, random's and passthrough's. No other preset
/// is one, and each names its own profile and the stage of it that asks more.
#[test]
fn a_place_is_picked_alone_where_its_picker_decides_by_itself() {
    for name in ["round_robin", "random", "passthrough"] {
        let engine = preset(name);
        let document = engine.spec();
        let place = document.place_for(MODEL, Phase::Single);
        assert!(matches!(place, Place::Alone(_)), "{place:?}");
    }
    for name in [
        "power_of_two",
        "least_load",
        "consistent_hashing",
        "prefix_hash",
        "manual",
        "cache_aware",
        "bucket",
    ] {
        let engine = preset(name);
        let document = engine.spec();
        let Place::NotAlone {
            profile,
            in_the_way: stage,
        } = document.place_for(MODEL, Phase::Single)
        else {
            panic!("{name} is picked alone");
        };
        assert_eq!(profile, name);
        assert!(!stage.to_string().is_empty(), "{name}: {stage}");
    }
    assert_eq!(
        preset("manual").spec().place_for(MODEL, Phase::Single),
        Place::NotAlone {
            profile: "manual",
            in_the_way: InTheWay::Picker("manual"),
        }
    );
    assert!(matches!(
        preset("least_load").spec().place_for(MODEL, Phase::Single),
        Place::NotAlone {
            in_the_way: InTheWay::Scorer(_),
            ..
        }
    ));
}

/// A side of a pair holds one test beyond a worker's health: the prefill
/// side its window's slot, the decode side its room. A gateway's pass
/// holds each at its own side's place and at no other. So each side's
/// profile is picked alone at its own place, and the same filter stands
/// in the way at every other place.
#[test]
fn a_pairs_side_is_picked_alone_at_its_own_place_only() {
    let mut document =
        PolicySpec::pair_presets("round_robin", "round_robin").expect("built-in presets");
    document.generation = 1;
    let engine = engine_of(&document);
    assert!(matches!(
        engine.spec().place_for(MODEL, Phase::Prefill),
        Place::Alone(_)
    ));
    assert!(matches!(
        engine.spec().place_for(MODEL, Phase::Decode),
        Place::Alone(_)
    ));
    // The pair plan has no side for a single request, and none for an
    // encode one: a binding names the model, so the place is not
    // unbound.
    assert_eq!(engine.spec().place_for(MODEL, Phase::Single), Place::NoSide);
    assert_eq!(engine.spec().place_for(MODEL, Phase::Encode), Place::NoSide);

    for (filter, own) in [
        ("has-capacity", Phase::Prefill),
        ("has-room", Phase::Decode),
    ] {
        let engine = round_robin_with(filter);
        for phase in [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode] {
            let document = engine.spec();
            let place = document.place_for(MODEL, phase);
            if phase == own {
                assert!(matches!(place, Place::Alone(_)), "{filter} at {phase:?}");
            } else {
                let stage = InTheWay::Filter(filter);
                let profile = "round_robin";
                let named = Place::NotAlone {
                    profile,
                    in_the_way: stage,
                };
                assert_eq!(place, named, "{phase:?}");
            }
        }
    }
}

/// Every request of a document is decided whole where each of its
/// places is picked alone or its stages read nothing the host does not
/// feed. Round robin asks nothing of a host. Random and passthrough
/// read a worker's record and no more. Least load reads the workers'
/// reports, and the default policy who holds the prompt beside them.
/// A host that feeds less than a profile declares, a profile that
/// reads a hash ring, one under a sessions rule and a pair's plan are
/// not decided so.
#[test]
fn a_document_is_decided_whole_where_each_place_is_picked_alone_or_fed() {
    let whole = |engine: &InProcess, fed: Needs| engine.spec().engine().single_and_fed(fed);
    let nothing = Needs::default();
    for name in ["round_robin", "random", "passthrough"] {
        assert!(whole(&preset(name), nothing), "{name}");
    }
    for name in ["power_of_two", "least_load", "bucket"] {
        let engine = preset(name);
        let declared = engine.spec().engine().needs_for(MODEL, Phase::Single);
        assert!(!declared.affinity && !declared.ring, "{name}");
        assert!(whole(&engine, declared), "{name}");
    }
    assert!(
        !whole(&preset("least_load"), nothing),
        "its scorer reads reports"
    );
    let default = preset("cache_aware");
    let declared = default.spec().engine().needs_for(MODEL, Phase::Single);
    assert!(declared.affinity && declared.loads && declared.sent_prefixes);
    assert!(whole(&default, declared));
    assert!(!whole(&default, nothing));
    let unfed = Needs {
        loads: false,
        ..declared
    };
    assert!(!whole(&default, unfed), "its load stages read reports");
    // A ring is no part of what the default policy's host feeds, and a
    // sessions rule is decided by no whole pass here yet.
    for name in ["consistent_hashing", "prefix_hash", "manual"] {
        assert!(!whole(&preset(name), declared), "{name}");
    }
    let mut pair =
        PolicySpec::pair_presets("round_robin", "round_robin").expect("built-in presets");
    pair.generation = 1;
    assert!(!whole(&engine_of(&pair), declared));
}

/// A pick tests a worker's health whole, as the pass does. A profile
/// that leaves one of the two health tests out would gain it by the
/// pick, so it is not picked alone, and the answer names the test.
#[test]
fn a_profile_without_a_health_test_is_not_picked_alone() {
    for (at, left_out) in [(0, "healthy"), (1, "not-overloaded")] {
        let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
        let profile = document
            .profiles
            .get_mut("round_robin")
            .expect("the preset's profile");
        assert_eq!(profile.filter[at].name(), left_out);
        profile.filter.remove(at);
        let stage = InTheWay::NoFilter(left_out);
        let profile = "round_robin";
        assert_eq!(
            engine_of(&document).spec().place_for(MODEL, Phase::Single),
            Place::NotAlone {
                profile,
                in_the_way: stage
            }
        );
    }
}

/// With every worker ready a pick is the pool's own turn: one strict
/// rotation at every pool size, from the position the pool drew.
#[test]
fn a_pick_turns_the_pools_own_rotation() {
    for workers in [1, 2, 3, 8, 16, 64] {
        let engine = preset("round_robin");
        let profile = alone(&engine);
        let members = PublishedList::new(pool(workers));
        let first = engine
            .pick(
                &engine.spec(),
                profile,
                members.lent(),
                RoutingState::eligible,
            )
            .expect("a worker");
        for step in 1..2 * workers {
            let at = engine.pick(
                &engine.spec(),
                profile,
                members.lent(),
                RoutingState::eligible,
            );
            assert_eq!(at, Some((first + step) % workers), "{workers} workers");
        }
    }
}

/// The `random` preset's pick is one draw of its picker over the pool:
/// every ready worker takes a near even share and a worker that is not
/// ready takes none, by the pool's own draws and by a pass over a list
/// alike. A pool whose draws miss is passed over, and the pass names
/// its ready worker.
#[test]
fn a_random_pick_is_one_draw_over_the_ready_workers() {
    let engine = preset("random");
    let document = engine.spec();
    let profile = alone(&engine);
    let list = pool(8);
    list[3].set_status(WorkerStatus::NotReady);
    let members = PublishedList::new(Arc::clone(&list));
    let (mut picks, mut passes) = ([0u32; 8], [0u32; 8]);
    for _ in 0..4_000 {
        let at = engine.pick(&document, profile, members.lent(), RoutingState::eligible);
        picks[at.expect("a ready worker")] += 1;
        let at = engine.pick_among(&document, profile, &list, RoutingState::eligible);
        passes[at.expect("a ready worker")] += 1;
    }
    for shares in [picks, passes] {
        assert_eq!(shares[3], 0, "{shares:?}");
        let even = |&share: &u32| (350..800).contains(&share);
        assert_eq!(
            shares.iter().filter(|share| even(share)).count(),
            7,
            "{shares:?}"
        );
    }

    // One ready worker of 64: eight draws miss it nearly nine times in
    // ten, and the pass over the pool then names it.
    let wide = pool(64);
    for worker in wide.iter().skip(1) {
        worker.set_status(WorkerStatus::NotReady);
    }
    let members = PublishedList::new(wide);
    for _ in 0..200 {
        let at = engine.pick(&document, profile, members.lent(), RoutingState::eligible);
        assert_eq!(at, Some(0));
    }
}

/// A pick runs under the handle its profile was asked of, whatever
/// document the engine holds in force by then: a rotation's profile
/// keeps turning its pool while a document that draws is in force.
#[test]
fn a_pick_runs_under_the_handle_its_profile_was_asked_of() {
    let engine = preset("round_robin");
    let handle = engine.spec();
    let profile = alone(&engine);
    let mut next = PolicySpec::preset("random").expect("a built-in preset");
    next.generation = 2;
    let next = AppliedSpec::validate(&next).expect("the document validates");
    engine.apply(next).expect("a number above the standing one");

    let members = PublishedList::new(pool(8));
    let pick = || engine.pick(&handle, profile, members.lent(), RoutingState::eligible);
    let first = pick().expect("a worker");
    for step in 1..16 {
        assert_eq!(pick(), Some((first + step) % 8), "the handle's rotation");
    }
}

/// A profile asked of one document and picked under another: where that
/// document holds no profile at the position, a pick names no worker, by
/// the pool's own turn and by a pass over a list alike.
#[test]
fn a_pick_under_a_document_with_no_profile_at_its_position_names_no_worker() {
    let mut wide = PolicySpec::preset("round_robin").expect("a built-in preset");
    let spare = wide.profiles["round_robin"].clone();
    wide.profiles.insert("a_spare".to_owned(), spare);
    let wide = engine_of(&wide);
    // The spare stands first in the document's order: the model's profile
    // is the second of two.
    let profile = alone(&wide);
    assert_eq!(profile, AloneProfile(1));
    let list = pool(4);
    let own = wide.pick_among(&wide.spec(), profile, &list, RoutingState::eligible);
    assert!(own.is_some(), "its own document picks");

    let engine = preset("round_robin");
    let handle = engine.spec();
    let members = PublishedList::new(Arc::clone(&list));
    let turn = engine.pick(&handle, profile, members.lent(), RoutingState::eligible);
    assert_eq!(turn, None);
    let pass = engine.pick_among(&handle, profile, &list, RoutingState::eligible);
    assert_eq!(pass, None);
}

/// The rotation is the pool's own word: two pools of one engine turn a
/// rotation each, and a turn of one moves no turn of the other.
#[test]
fn each_pool_turns_its_own_word() {
    let engine = preset("round_robin");
    let profile = alone(&engine);
    let (one, other) = (pool(5), pool(5));
    let (one, other) = (
        PublishedList::new(Arc::clone(&one)),
        PublishedList::new(Arc::clone(&other)),
    );
    let first = engine
        .pick(&engine.spec(), profile, one.lent(), RoutingState::eligible)
        .expect("a worker");
    let apart = engine
        .pick(
            &engine.spec(),
            profile,
            other.lent(),
            RoutingState::eligible,
        )
        .expect("a worker");
    for step in 1..=10 {
        assert_eq!(
            engine.pick(&engine.spec(), profile, one.lent(), RoutingState::eligible),
            Some((first + step) % 5)
        );
        assert_eq!(
            engine.pick(
                &engine.spec(),
                profile,
                other.lent(),
                RoutingState::eligible
            ),
            Some((apart + step) % 5)
        );
    }
}

/// Workers that are not ready take no request, and the ready ones take
/// exact shares. At every size a turn on such a worker goes to the
/// pass, and the ready workers' own rotation places the request, as
/// the classic router's turn does
/// (`model_gateway/src/policies/round_robin.rs:151-153`, `:164-167`):
/// the turn is not handed to the list's next worker. Its own case of
/// four workers holds the shares
/// (`a_refused_row_falls_to_the_admitted_set_and_the_spread_stays_even` in
/// `model_gateway/tests/unit/policies/round_robin/tests.rs`).
#[test]
fn a_pick_keeps_exact_shares_past_workers_not_ready() {
    let engine = preset("round_robin");
    let profile = alone(&engine);
    let shares = |members: &PublishedList, picks: usize| {
        let mut times = vec![0usize; members.workers().len()];
        for _ in 0..picks {
            times[engine
                .pick(
                    &engine.spec(),
                    profile,
                    members.lent(),
                    RoutingState::eligible,
                )
                .expect("a worker")] += 1;
        }
        times
    };
    let leave = |list: &[Arc<dyn Worker>], ready: &dyn Fn(usize) -> bool| {
        for (at, worker) in list.iter().enumerate() {
            if !ready(at) {
                worker.set_status(WorkerStatus::NotReady);
            }
        }
    };

    // Eight workers, one not ready: seven turns of eight place by the
    // pool's turn and one by the seven ready workers' own rotation.
    let list = pool(8);
    leave(&list, &|at| at != 5);
    let members = PublishedList::new(Arc::clone(&list));
    let times = shares(&members, 8 * 7 * 3);
    for (at, &times) in times.iter().enumerate() {
        assert_eq!(times, if at == 5 { 0 } else { 24 }, "worker {at} of 8");
    }

    // Sixty-four workers, one in four not ready, none beside another.
    // Over three turns of the pool each ready worker takes three by
    // the pool's turn, and the 48 turns on a worker not ready go once
    // round the ready workers' own rotation.
    let list = pool(64);
    leave(&list, &|at| at % 4 != 1);
    let members = PublishedList::new(Arc::clone(&list));
    let times = shares(&members, 64 * 3);
    for (at, &times) in times.iter().enumerate() {
        assert_eq!(times, if at % 4 == 1 { 0 } else { 4 }, "worker {at} of 64");
    }

    // Sixty-four workers, sixteen in a row not ready, from the pool's
    // turn on the first of the run: the same shares. The worker behind
    // the run takes no more than its peers.
    let list = pool(64);
    leave(&list, &|at| !(20..36).contains(&at));
    let members = PublishedList::new(Arc::clone(&list));
    let made = engine.members_of(members.lent());
    made.rotation.0.store(20, Ordering::Relaxed);
    let times = shares(&members, 64 * 3);
    for (at, &times) in times.iter().enumerate() {
        let expected = if (20..36).contains(&at) { 0 } else { 4 };
        assert_eq!(times, expected, "worker {at} of 64, a run not ready");
    }
}

/// A worker over its overload thresholds takes a request only where the
/// caller lifted that veto, and a worker that is not ready, doubted or
/// behind an open breaker under neither word. A pool no worker of which
/// passes names nobody: its caller decides such a pool.
#[test]
fn a_pick_tests_a_worker_as_the_worker_tests_itself() {
    let engine = preset("round_robin");
    let profile = alone(&engine);
    let list = pool(4);
    list[0].set_status(WorkerStatus::NotReady);
    list[1].set_overloaded(true);
    list[2].set_stall(Some(StallReason::Wedged));
    let members = PublishedList::new(Arc::clone(&list));
    for _ in 0..8 {
        assert_eq!(
            engine.pick(
                &engine.spec(),
                profile,
                members.lent(),
                RoutingState::eligible
            ),
            Some(3)
        );
    }
    let lifted: BTreeSet<usize> = (0..8)
        .filter_map(|_| {
            engine.pick(
                &engine.spec(),
                profile,
                members.lent(),
                RoutingState::eligible_ignoring_overload,
            )
        })
        .collect();
    assert_eq!(lifted, BTreeSet::from([1, 3]));

    list[3].set_overloaded(true);
    assert_eq!(
        engine.pick(
            &engine.spec(),
            profile,
            members.lent(),
            RoutingState::eligible
        ),
        None
    );
    assert_eq!(
        engine.pick_among(&engine.spec(), profile, &list, RoutingState::eligible),
        None
    );
    assert!(engine
        .pick(
            &engine.spec(),
            profile,
            members.lent(),
            RoutingState::eligible_ignoring_overload
        )
        .is_some());
    // A pool that holds no worker.
    let empty = PublishedList::new(pool(0));
    assert_eq!(
        engine.pick(
            &engine.spec(),
            profile,
            empty.lent(),
            RoutingState::eligible
        ),
        None
    );
    assert_eq!(
        engine.pick_among(&engine.spec(), profile, &[], RoutingState::eligible),
        None
    );
}

/// A pick that asked every worker of a pool and kept none has the pool's
/// answer: a scan reads each worker once and no pass reads them again.
/// A rotation's one turn stops short of the pool, which is then passed
/// over.
#[test]
fn a_pick_that_asked_every_worker_is_not_followed_by_a_pass() {
    for workers in [1, 2, 8, 64] {
        let list = pool(workers);
        for worker in list.iter() {
            worker.set_overloaded(true);
        }
        let reads_under = |name: &str| {
            let engine = preset(name);
            let members = PublishedList::new(Arc::clone(&list));
            let reads = Cell::new(0);
            let counted = |state: RoutingState| {
                reads.set(reads.get() + 1);
                state.eligible()
            };
            assert_eq!(
                engine.pick(&engine.spec(), alone(&engine), members.lent(), counted),
                None
            );
            reads.get()
        };
        assert_eq!(reads_under("passthrough"), workers, "each worker once");
        assert!(reads_under("round_robin") > workers, "turns, then a pass");
    }
}

/// A list the engine keeps nothing of is passed over once, and the
/// workers that pass take turns by their set's own rotation: exact
/// shares, named by their place in the list handed.
#[test]
fn a_pick_among_a_list_turns_the_rotation_of_the_workers_that_pass() {
    let engine = preset("round_robin");
    let profile = alone(&engine);
    let list = pool(9);
    // A part of the pool, as a caller narrows it: every third worker
    // left out, and one of the others not ready.
    let part: Vec<Arc<dyn Worker>> = (0..9)
        .filter(|at| at % 3 != 0)
        .map(|at| Arc::clone(&list[at]))
        .collect();
    part[2].set_status(WorkerStatus::NotReady);
    let mut times = [0usize; 6];
    for _ in 0..5 * 7 {
        times[engine
            .pick_among(&engine.spec(), profile, &part, RoutingState::eligible)
            .expect("a worker")] += 1;
    }
    assert_eq!(times, [7, 7, 0, 7, 7, 7]);
}

/// The maker receives the document that governs the single request.
/// The pick records the generation the maker reads.
#[test]
fn a_request_is_made_from_the_document_its_decision_is_made_under() {
    use super::single::tests::{request, Asks};

    struct ReadsDocument<'a> {
        asks: Asks<'a>,
        expected: &'a AppliedSpec,
        generation: Cell<Option<u64>>,
    }

    impl RequestMaker for ReadsDocument<'_> {
        fn model(&self) -> &str {
            self.asks.model()
        }

        fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
            assert!(
                ptr::eq(spec, self.expected),
                "the maker receives the engine's own document"
            );
            self.generation.set(Some(spec.generation()));
            self.asks.make(spec)
        }

        fn workers(&self) -> WorkerLists<'_> {
            self.asks.workers()
        }
    }

    let mut document = PolicySpec::preset("random").expect("a built-in preset");
    document.generation = 17;
    let engine = engine_of(&document);
    let members = PublishedList::new(pool(3));
    let maker = ReadsDocument {
        asks: Asks::of(request(), Some(&members)),
        expected: engine.standing_spec(),
        generation: Cell::new(None),
    };

    let (outcome, _) = engine
        .decide_at(&maker, Instant::now)
        .expect("a decision")
        .whole()
        .expect("a whole decision");
    let pick = outcome.decision.pick().expect("a pick");
    let generation = maker.generation.get().expect("the maker ran");
    assert_eq!(generation, document.generation);
    assert_eq!(pick.trace.generation, generation);
}

/// The unit tests' way to a decision asked ahead with a seed they state.
/// The placing call's seed is drawn fresh, so a test that compares such a
/// decision with the made request's states the seed here, and the pass
/// reads it in a test build alone (`seed_ahead`).
pub(in crate::engine::in_process) mod seed_hook {
    use std::cell::Cell;

    thread_local! {
        static SEED: Cell<Option<u64>> = const { Cell::new(None) };
    }

    /// The seed stated on this thread, if a test states one now.
    pub(in crate::engine::in_process) fn seed() -> Option<u64> {
        SEED.get()
    }

    /// Runs `run` with `seed` stated on this thread: a pass inside it
    /// may decide its request ahead of the making, whoever stated the
    /// request, and such a decision draws from `seed`.
    pub(in crate::engine::in_process) fn stated<T>(seed: u64, run: impl FnOnce() -> T) -> T {
        struct Restored(Option<u64>);
        impl Drop for Restored {
            fn drop(&mut self) {
                SEED.set(self.0);
            }
        }
        let _restored = Restored(SEED.replace(Some(seed)));
        run()
    }
}
