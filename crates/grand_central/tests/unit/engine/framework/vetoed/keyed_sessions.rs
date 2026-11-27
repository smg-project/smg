//! A keyed request under the rule: the ring's owner kept over the veto, a pin
//! that yields, the key's mark, a claim followed, and a pin at the rule's cap.

use super::*;

/// A profile that places by the session key keeps a keyed request on
/// its key's worker over the veto, where the binding steers: the owner
/// over its thresholds takes the key beside free workers, and takes it
/// when every worker is over, whoever holds the fewest requests.
#[test]
fn a_key_keeps_its_ring_owner_over_the_veto() {
    let engine = engine_of("consistent_hashing", false);
    let request = keyed("k1");
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&engine, &request, &pool);
    let owner = ring[0] as usize;
    pool[owner].health.overloaded = true;
    pool[owner].in_flight = 9;
    assert_eq!(
        order(&decide(&engine, &request, &pool))[0],
        ring[0],
        "beside free workers"
    );
    for worker in &mut pool {
        worker.health.overloaded = true;
    }
    let decision = decide(&engine, &request, &pool);
    assert_eq!(order(&decision)[0], ring[0], "every worker over");
    let pick = decision.pick().expect("a pick");
    assert_eq!(pick.trace.survivors.get(), Some(4));
}

/// A binding that sheds keeps the veto for a keyed request: an owner
/// over its thresholds passes the key to the ring's next member, and a
/// pool over on every worker is shed as saturated.
#[test]
fn a_binding_that_sheds_keeps_the_veto_for_a_key() {
    let engine = engine_of("consistent_hashing", true);
    let request = keyed("k1");
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&engine, &request, &pool);
    pool[ring[0] as usize].health.overloaded = true;
    assert_eq!(order(&decide(&engine, &request, &pool))[0], ring[1]);
    for worker in &mut pool {
        worker.health.overloaded = true;
    }
    assert_eq!(
        shed_of(&decide(&engine, &request, &pool)),
        (ShedReason::Saturated, "not-overloaded")
    );
}

/// A kept key's pick says the veto it stands by. With every worker over
/// its thresholds the veto as it stands leaves the pool nobody, and the
/// trace carries the mark the rule's own pick carries: a host counts
/// both from it. Beside a worker under its thresholds the owner over its
/// own takes the key as a session keeps its owner, and the trace says no
/// veto. So it does beside a worker under its thresholds that the retry's
/// list passes over: the veto did not empty that pool.
#[test]
fn a_kept_keys_pick_says_the_veto_over_an_all_vetoed_pool_alone() {
    let engine = engine_of("consistent_hashing", false);
    let request = keyed("k1");
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&engine, &request, &pool);
    let (owner, next) = (ring[0] as usize, ring[1] as usize);
    pool[owner].health.overloaded = true;
    let beside = decide(&engine, &request, &pool);
    assert_eq!(order(&beside)[0], ring[0]);
    assert_eq!(
        beside.trace().lifted,
        LiftedVeto::None,
        "beside free workers"
    );
    for worker in &mut pool {
        worker.health.overloaded = true;
    }
    let whole = decide(&engine, &request, &pool);
    assert_eq!(order(&whole)[0], ring[0]);
    assert_eq!(
        whole.trace().lifted,
        LiftedVeto::Overload,
        "every worker over"
    );
    pool[next].health.overloaded = false;
    let tried = [pool[next].id];
    let retry = RequestFacts {
        tried: &tried,
        attempt: 1,
        ..request
    };
    let passed = decide(&engine, &retry, &pool);
    assert_eq!(order(&passed)[0], ring[0]);
    assert_eq!(
        passed.trace().lifted,
        LiftedVeto::None,
        "a tried worker under"
    );
}

/// No pass of a kept key is put to the all-vetoed rule: its lifted run
/// stands in for it. With every worker over its thresholds the profile
/// runs once with the veto lifted, and the pass that asks whether the
/// veto emptied the pool stops at the veto: a filter listed behind it is
/// asked about each worker once.
#[test]
fn a_kept_keys_pool_is_put_to_no_rule() {
    /// Keeps every row and counts the rows it was asked about.
    #[derive(Debug)]
    struct Counts(Arc<AtomicUsize>);
    impl Filter for Counts {
        fn name(&self) -> &'static str {
            "counts"
        }
        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed);
            true
        }
    }
    let asked = Arc::new(AtomicUsize::new(0));
    let mut engine = Engine::new(
        vec![Profile {
            name: "counted".into(),
            filters: vec![
                Box::new(NotOverloaded),
                Box::new(Counts(Arc::clone(&asked))),
            ],
            scorers: vec![],
            picker: Box::new(Best),
        }],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::Single,
            profile: 0,
            decode_profile: None,
            shadow: None,
            decode_shadow: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    );
    engine.places_by_key(0);
    let pool = [over(0, 1), over(1, 2)];
    let decision = decide(&engine, &keyed("k1"), &pool);
    assert_eq!(decision.trace().lifted, LiftedVeto::Overload);
    assert_eq!(asked.load(Ordering::Relaxed), 2);
}

/// The lift is the veto's alone: an owner that is not ready, or that an
/// earlier attempt tried, passes the key to the ring's next member under
/// either word of the binding.
#[test]
fn a_key_whose_owner_is_unusable_walks_the_ring() {
    for shed in [false, true] {
        let engine = engine_of("consistent_hashing", shed);
        let request = keyed("k1");
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&engine, &request, &pool);
        let owner = ring[0] as usize;
        pool[owner].health.overloaded = true;
        let tried = [pool[owner].id];
        let retry = RequestFacts {
            tried: &tried,
            attempt: 1,
            ..request
        };
        assert_eq!(
            order(&decide(&engine, &retry, &pool))[0],
            ring[1],
            "tried, shed {shed}"
        );
        let excluded = Constraints {
            excluded: &tried,
            ..Constraints::default()
        };
        let decision = engine.decide(
            &request,
            &PoolFacts::default(),
            &pool,
            &excluded,
            &mut Scratch::new(),
        );
        assert_eq!(order(&decision)[0], ring[1], "unreached, shed {shed}");
        pool[owner].health.healthy = false;
        assert_eq!(
            order(&decide(&engine, &request, &pool))[0],
            ring[1],
            "not ready, shed {shed}"
        );
    }
}

/// The lift is the mark's alone: a request with no key under the ring's
/// profile, and a keyed request under a profile with no ring, meet the
/// veto as every request does. A pool over on every worker goes to the
/// fewest in flight, and a worker under its thresholds takes the request
/// when there is one.
#[test]
fn a_request_the_ring_does_not_place_gets_no_lift() {
    for (preset, request) in [
        ("consistent_hashing", facts(1)),
        ("round_robin", keyed("k1")),
        ("least_load", keyed("k1")),
    ] {
        let engine = engine_of(preset, false);
        let mut pool = [over(0, 3), over(1, 1), over(2, 2), over(3, 4)];
        assert_eq!(order(&decide(&engine, &request, &pool))[0], 1, "{preset}");
        pool[3].health.overloaded = false;
        assert_eq!(
            order(&decide(&engine, &request, &pool)),
            vec![3],
            "{preset}"
        );
    }
}

/// The lift is the single plan's: an encode item of the `epd` plan is
/// keyed by its content hash, and under the ring's profile it meets the
/// veto as every item does. Encoders over on every worker take it by
/// the fewest in flight, whichever of them the ring names.
#[test]
fn an_encode_item_under_the_ring_keeps_the_veto() {
    let engine = PolicySpec::epd_presets("consistent_hashing", "round_robin", "round_robin")
        .expect("the presets' form")
        .validate()
        .expect("the presets build");
    let mut pool = [over(0, 3), over(1, 1), over(2, 2), over(3, 4)];
    for worker in &mut pool {
        worker.ec_compatible = true;
    }
    for key in ["k1", "k2", "k3", "k4"] {
        let item = RequestFacts {
            phase: Phase::Encode,
            session_key: Identity::plain(key),
            ..facts(1)
        };
        assert_eq!(order(&decide(&engine, &item, &pool))[0], 1, "{key}");
    }
}

/// An engine built by hand marks no profile: its ring stage meets the
/// veto, and the host's lift keeps the key on its worker.
#[test]
fn an_engine_built_by_hand_lifts_nothing_for_a_key() {
    let engine = Engine::new(
        vec![Profile {
            name: "ring".into(),
            filters: vec![Box::new(NotOverloaded)],
            scorers: vec![WeightedScorer {
                scorer: Box::new(ConsistentHash::new(None)),
                weight: 1.0,
            }],
            picker: Box::new(Best),
        }],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::Single,
            profile: 0,
            decode_profile: None,
            shadow: None,
            decode_shadow: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    );
    let request = keyed("k1");
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let owner = order(&decide(&engine, &request, &pool))[0];
    pool[owner as usize].health.overloaded = true;
    assert_ne!(order(&decide(&engine, &request, &pool))[0], owner);
    let lifted = Constraints {
        overload_lifted: true,
        ..Constraints::default()
    };
    let decision = engine.decide(
        &request,
        &PoolFacts::default(),
        &pool,
        &lifted,
        &mut Scratch::new(),
    );
    assert_eq!(order(&decision)[0], owner);
}

/// A pin does not borrow the ring's lift. Under a sessions rule over the
/// ring's profile the pin's test reads the veto as it stands: a pinned
/// worker over its thresholds yields, and the ring's own worker for the
/// key takes the request over its thresholds, beside free workers.
#[test]
fn a_pin_over_its_thresholds_yields_and_the_rings_worker_takes_the_key() {
    let engine = ring_under_a_rule(Assignment::Delegate);
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    // The key's ring order, read on an engine of the same profile with
    // no rule.
    let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
    let (owner, next) = (ring[0] as usize, ring[1] as usize);
    // The key's worker is not ready: the ring's next member takes the
    // key and is pinned.
    pool[owner].health.healthy = false;
    assert_eq!(dispatched(&engine, &pool, 1_000), ring[1], "the pin");
    // The worker is back and free: the pin stands.
    pool[owner].health.healthy = true;
    assert_eq!(dispatched(&engine, &pool, 1_500), ring[1], "the pin stands");
    // The worker and the pin are both over their thresholds beside two
    // free workers.
    pool[owner].health.overloaded = true;
    pool[next].health.overloaded = true;
    assert_eq!(
        dispatched(&engine, &pool, 2_000),
        ring[0],
        "the pin yields and the ring's worker takes the key"
    );
}

/// A pool over on every worker keeps no pin standing: the pin's test
/// reads the veto as it stands, under which nobody is kept, and no rule
/// answers that pass. The ring's own worker takes the key, and the trace
/// says the lifted veto. The pin stands ahead of that worker's claim, and
/// once the pool is under its thresholds the key returns to it (the
/// manual preset's rule recovers the original).
#[test]
fn a_pin_yields_in_a_pool_over_on_every_worker() {
    let engine = ring_under_a_rule(Assignment::Delegate);
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
    let owner = ring[0] as usize;
    // The key's worker is not ready: the ring's next member is pinned.
    pool[owner].health.healthy = false;
    assert_eq!(dispatched(&engine, &pool, 1_000), ring[1], "the pin");
    pool[owner].health.healthy = true;
    for worker in &mut pool {
        worker.health.overloaded = true;
    }
    let request = RequestFacts {
        now_micros: 2_000,
        ..keyed("k1")
    };
    let whole = decide(&engine, &request, &pool);
    assert_eq!(order(&whole)[0], ring[0], "the pin yields");
    assert_eq!(whole.trace().lifted, LiftedVeto::Overload);
    assert_eq!(
        engine
            .sessions()
            .pins(request.session())
            .map(|pins| pins[0]),
        Some(pool[ring[1] as usize].id),
        "the pin is the key's first"
    );
    // The pool is under its thresholds again: the key is its pin's.
    for worker in &mut pool {
        worker.health.overloaded = false;
    }
    let later = RequestFacts {
        now_micros: 3_000,
        ..keyed("k1")
    };
    let back = decide(&engine, &later, &pool);
    assert_eq!(order(&back)[0], ring[1], "the pinned worker");
    assert_eq!(
        back.pick().expect("a pick").session_verdict,
        SessionVerdict::Pinned
    );
}

/// Under a sessions rule the lifted placement says the veto the same
/// way: a key no pin takes, placed in a pool over on every worker,
/// carries the mark, and beside a worker under its thresholds it carries
/// none.
#[test]
fn a_key_placed_under_a_rule_says_the_veto_over_an_all_vetoed_pool_alone() {
    let mut pool = [over(0, 4), over(1, 4), over(2, 4), over(3, 4)];
    let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
    let request = RequestFacts {
        now_micros: 1_000,
        ..keyed("k1")
    };
    let whole = decide(&ring_under_a_rule(Assignment::Delegate), &request, &pool);
    assert_eq!(order(&whole)[0], ring[0]);
    assert_eq!(
        whole.trace().lifted,
        LiftedVeto::Overload,
        "every worker over"
    );
    pool[ring[1] as usize].health.overloaded = false;
    let beside = decide(&ring_under_a_rule(Assignment::Delegate), &request, &pool);
    assert_eq!(order(&beside)[0], ring[0]);
    assert_eq!(
        beside.trace().lifted,
        LiftedVeto::None,
        "beside a free worker"
    );
}

/// The lift is the ring's: a rule whose own assignment places a key no
/// pin takes asks no stage where the key goes, and nothing is lifted.
/// The key's worker over its thresholds is passed over for the fewest
/// in flight under theirs.
#[test]
fn a_rule_that_places_a_key_itself_lifts_nothing() {
    let engine = ring_under_a_rule(Assignment::MinLoad);
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
    pool[ring[0] as usize].health.overloaded = true;
    pool[ring[3] as usize].in_flight = 1;
    assert_eq!(dispatched(&engine, &pool, 1_000), ring[3]);
}

/// The key's mark moves a profile's scope word to the key's own, where
/// both plans still find a scope to open, and it leaves the marks of a
/// profile placed by assignment as they are.
#[test]
fn the_keys_mark_keeps_a_scope_to_open() {
    let keyed = Marks(Marks::SCOPED).with_key();
    assert!(keyed.scoped() && keyed.keyed_scope());
    assert!(!keyed.opens_first() && !keyed.asks_nothing_first());
    let bare = Marks(Marks::DRAWS_FIRST).with_key();
    assert!(!bare.scoped() && !bare.keyed_scope());
    assert_eq!(bare.ahead(), Marks::DRAWS_FIRST);
    assert!(!bare.asks_nothing_first());
    let placed = Marks(Marks::SCOPED | Marks::BY_ASSIGNMENT);
    assert_eq!(placed.with_key(), placed);
}

/// Keeps every row and holds a scope.
#[derive(Debug)]
struct Scoped(Arc<Scope>);

impl Filter for Scoped {
    fn name(&self) -> &'static str {
        "scoped"
    }
    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }
    fn scope(&self) -> Option<&dyn DecisionScope> {
        Some(self.0.as_ref())
    }
}

/// The ring's profile built by hand with a scope to open and the
/// picker given, marked as the validator marks the ring's profile.
fn scoped_ring(picker: Box<dyn Picker>) -> (Engine, Arc<Scope>) {
    let scope = Arc::new(Scope::default());
    let mut engine = Engine::new(
        vec![Profile {
            name: "ring".into(),
            filters: vec![
                Box::new(NotOverloaded),
                Box::new(Scoped(Arc::clone(&scope))),
            ],
            scorers: vec![WeightedScorer {
                scorer: Box::new(ConsistentHash::new(None)),
                weight: 1.0,
            }],
            picker,
        }],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::Single,
            profile: 0,
            decode_profile: None,
            shadow: None,
            decode_shadow: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    );
    engine.places_by_key(0);
    (engine, scope)
}

/// The key's mark keeps the profile's scope: each decision opens and
/// closes it once, a keyed request's lifted run and a request with no
/// key alike.
#[test]
fn a_profile_under_the_keys_mark_opens_its_scope_once() {
    let (engine, scope) = scoped_ring(Box::new(Best));
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let owner = order(&decide(&engine, &keyed("k1"), &pool))[0];
    pool[owner as usize].health.overloaded = true;
    assert_eq!(
        order(&decide(&engine, &keyed("k1"), &pool))[0],
        owner,
        "the lift"
    );
    assert!(decide(&engine, &facts(1), &pool).pick().is_some());
    assert_eq!(scope.opened.load(Ordering::Relaxed), 3);
    assert_eq!(scope.closed.load(Ordering::Relaxed), 3);
}

/// A profile under the key's mark that shows its rows declines what
/// the scope's own arm declines: a mask beside the rows says
/// they were not written for the decision, and the key's arm opens no
/// scope over them, for a request with a key and for one with none.
#[test]
fn a_profile_under_the_keys_mark_declines_rows_not_written_for_it() {
    let (engine, scope) = scoped_ring(Box::new(Best));
    let pool = [free(0, 4), free(1, 4)];
    let key = keyed("k1");
    let none = facts(1);
    let masked = Constraints {
        stalled: &[0],
        ..Constraints::default()
    };
    let written = Constraints::default();
    let run = |facts: &RequestFacts<'_>, cx: &Constraints<'_>| {
        engine.decide(facts, &PoolFacts::default(), &pool, cx, &mut Scratch::new())
    };
    for (facts, cx, beside) in [
        (&key, &masked, "a mask"),
        (&none, &masked, "a mask and no key"),
    ] {
        assert_eq!(
            shed_of(&run(facts, cx)),
            (ShedReason::PolicyDeclined, WRITTEN_ROWS),
            "{beside}"
        );
    }
    assert_eq!(scope.opened.load(Ordering::Relaxed), 0);
    assert!(run(&key, &written).pick().is_some());
    assert_eq!(scope.opened.load(Ordering::Relaxed), 1);
}

/// A picker that leaves the placement to its rule takes no mark: its
/// profile's scope is opened as it is without a ring.
#[test]
fn a_profile_placed_by_assignment_takes_no_mark() {
    let (engine, scope) = scoped_ring(Box::new(Manual));
    let pool = [free(0, 4), free(1, 4)];
    assert!(decide(&engine, &keyed("k1"), &pool).pick().is_some());
    assert_eq!(scope.opened.load(Ordering::Relaxed), 1);
    assert_eq!(scope.closed.load(Ordering::Relaxed), 1);
}

/// A pool over on every worker keeps no pin standing, and the ring's
/// own worker takes a key under a rule there: the lifted profile answers
/// where the all-vetoed rule answers a profile with no ring. A pool the
/// lifted pass leaves nobody is shed as it is under no rule.
#[test]
fn a_key_under_a_rule_goes_to_the_rings_worker_over_a_pool_over_on_every_worker() {
    let engine = ring_under_a_rule(Assignment::Delegate);
    let ringed = engine_of("consistent_hashing", false);
    let mut pool = [over(0, 4), over(1, 4), over(2, 4), over(3, 4)];
    let ring = ring_order(&ringed, &keyed("k1"), &pool);
    // The fewest in flight is not the key's worker.
    pool[ring[2] as usize].in_flight = 1;
    assert_eq!(dispatched(&engine, &pool, 1_000), ring[0]);
    for worker in &mut pool {
        worker.health.healthy = false;
    }
    let request = RequestFacts {
        now_micros: 2_000,
        ..keyed("k1")
    };
    assert_eq!(
        shed_of(&decide(&engine, &request, &pool)),
        shed_of(&decide(&ringed, &request, &pool))
    );
}

/// Two decisions of one key agree before either is dispatched: the
/// second reads the first one's claim as a pin. With the ring's worker
/// over its thresholds that pin yields, and the ring sends the key to
/// the same worker again: the decision follows the pin it read and
/// claims nothing, so only the first decision's own dispatch elsewhere
/// withdraws the claim.
#[test]
fn a_second_decision_follows_the_first_ones_claim_under_the_ring() {
    let engine = ring_under_a_rule(Assignment::Delegate);
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
    let request = RequestFacts {
        now_micros: 1_000,
        ..keyed("k1")
    };
    let owner = pool[ring[0] as usize].id;
    let verdicts: Vec<_> = [false, false, true, true]
        .into_iter()
        .map(|over_its_thresholds| {
            pool[ring[0] as usize].health.overloaded = over_its_thresholds;
            let decision = decide(&engine, &request, &pool);
            let pick = *decision.pick().expect("a pick");
            assert_eq!(
                pick.targets.first().map(|target| target.position),
                Some(ring[0])
            );
            pick.session_verdict
        })
        .collect();
    let (claimed, pinned) = (SessionVerdict::Claimed(owner), SessionVerdict::Pinned);
    assert_eq!(verdicts, [claimed, pinned, pinned, pinned]);
}

/// The validator marks the profile that holds the ring wherever it
/// stands in the document, and no other. A keyed request of the ring's
/// binding keeps its worker over the veto; one of a binding whose
/// profile scores without the ring meets the veto, though the worker
/// over its thresholds holds the fewest requests.
#[test]
fn the_mark_lands_on_the_rings_profile_and_no_other() {
    let mut spec = PolicySpec::preset("consistent_hashing").expect("the preset's form");
    let scored = PolicySpec::preset("least_load").expect("the preset's form");
    // First by name, so the ring's profile is not the document's first.
    spec.profiles
        .insert("a_least_load".into(), scored.profiles["least_load"].clone());
    let mut binding = spec
        .bindings
        .first()
        .cloned()
        .expect("the preset's binding");
    binding.matcher.model = "scored".into();
    binding.profile = Some("a_least_load".into());
    spec.bindings.insert(0, binding);
    let engine = spec.validate().expect("the document builds");
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let owner = ring_order(&engine, &keyed("k1"), &pool)[0];
    pool[owner as usize].health.overloaded = true;
    pool[owner as usize].in_flight = 1;
    assert_eq!(order(&decide(&engine, &keyed("k1"), &pool))[0], owner);
    let other = RequestFacts {
        model: "scored",
        ..keyed("k1")
    };
    assert_ne!(order(&decide(&engine, &other, &pool))[0], owner);
}

/// A client's key places like a session's and keeps nothing: the request
/// goes to the ring's worker for the key, and off it where that worker
/// is over its thresholds, which a session's key stays on.
#[test]
fn a_client_key_places_on_the_ring_and_lifts_no_veto() {
    let engine = PolicySpec::preset("consistent_hashing")
        .expect("the preset's form")
        .validate()
        .expect("the document builds");
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let as_client = |session: &RequestFacts<'static>| RequestFacts {
        session_key: Identity::NONE,
        client_key: session.session_key.hash,
        ..*session
    };
    // Eight keys: one key alone passes a tie's draw one time in four.
    for key in ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"] {
        let session = keyed(key);
        let owner = ring_order(&engine, &session, &pool)[0];
        let placed = order(&decide(&engine, &as_client(&session), &pool))[0];
        assert_eq!(placed, owner, "{key}");
    }
    let session = keyed("k1");
    let client = as_client(&session);
    let owner = ring_order(&engine, &session, &pool)[0];
    pool[owner as usize].health.overloaded = true;
    pool[owner as usize].in_flight = 1;
    assert_eq!(order(&decide(&engine, &session, &pool))[0], owner);
    assert_ne!(order(&decide(&engine, &client, &pool))[0], owner);
}

/// The ring's preset with its one profile's scorers and picker replaced,
/// under the prefix hash preset's view, which admits the prefix form.
fn ring_shaped(score: serde_json::Value, pick: serde_json::Value) -> Engine {
    let mut spec = PolicySpec::preset("consistent_hashing").expect("the preset's form");
    spec.view = PolicySpec::preset("prefix_hash")
        .expect("the preset's form")
        .view;
    for profile in spec.profiles.values_mut() {
        profile.score = serde_json::from_value(score.clone()).expect("a score list");
        profile.pick = serde_json::from_value(pick.clone()).expect("a picker");
    }
    spec.validate().expect("the document builds")
}

/// The mark is the plain ring's only where the ring alone decides the
/// pick: the profile's only scorer, with no bound, under the picker that
/// takes the best score. A ring under a bound, in its prefix form,
/// beside another scorer, under a picker that turns or under one that
/// samples takes no mark: a keyed request of such a profile meets the
/// veto, and its key's worker over its thresholds is never its pick,
/// though that worker holds the fewest requests.
#[test]
fn a_ring_that_does_not_decide_the_pick_alone_takes_no_mark() {
    use serde_json::json;
    let plain = json!(["consistent-hash"]);
    let shapes = [
        ("the preset's own shape", plain.clone(), json!("best"), true),
        (
            "a bound",
            json!([{"consistent-hash": {"bounded_load": {"factor": 1.0, "abs_margin": 0}}}]),
            json!("best"),
            false,
        ),
        (
            "the prefix form",
            json!([{"consistent-hash": {"prefix_tokens": 256}}]),
            json!("best"),
            false,
        ),
        (
            "another scorer beside it",
            json!(["consistent-hash", "least-load"]),
            json!("best"),
            false,
        ),
        (
            "a picker that turns",
            plain.clone(),
            json!("round-robin"),
            false,
        ),
        (
            "a picker that samples",
            plain,
            json!({"best": {"sample": 8}}),
            false,
        ),
    ];
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let owner = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool)[0];
    pool[owner as usize].health.overloaded = true;
    pool[owner as usize].in_flight = 0;
    for (shape, score, pick, marked) in shapes {
        let engine = ring_shaped(score, pick);
        for seed in 1..=8 {
            let request = RequestFacts {
                seed,
                ..keyed("k1")
            };
            let first = order(&decide(&engine, &request, &pool))[0];
            assert_eq!(first == owner, marked, "{shape}, seed {seed}");
        }
    }
}

/// A ring under a bound keeps the veto for a keyed request. Its owner
/// past the bound is passed over and no row is ranked, so the picker
/// draws among the survivors: a worker over its thresholds is not one
/// of them, whatever it holds, and the pick says no lifted veto.
#[test]
fn a_bounded_rings_draw_leaves_out_a_worker_over_its_thresholds() {
    use serde_json::json;
    let engine = ring_shaped(
        json!([{"consistent-hash": {"bounded_load": {"factor": 1.0, "abs_margin": 0}}}]),
        json!("best"),
    );
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
    // The owner holds requests past the bound; the ring's next member is
    // over its thresholds and holds the fewest.
    pool[ring[0] as usize].in_flight = 20;
    pool[ring[1] as usize].health.overloaded = true;
    pool[ring[1] as usize].in_flight = 2;
    for seed in 1..=32 {
        let request = RequestFacts {
            seed,
            ..keyed("k1")
        };
        let decision = decide(&engine, &request, &pool);
        assert_ne!(order(&decision)[0], ring[1], "seed {seed}");
        assert_eq!(decision.trace().lifted, LiftedVeto::None, "seed {seed}");
    }
}

/// A kept key the lifted pass puts back on its pin follows that pin and
/// claims nothing. The pin over its thresholds yields under the veto as
/// it stands, the ring names the same worker with the veto lifted, and
/// the pin stands as it was read: when the host dispatches the decision
/// to its alternate, the key's worker is still its first pin, and the
/// key returns there (the manual preset's rule recovers the original).
#[test]
fn a_kept_key_put_back_on_its_pin_claims_nothing() {
    let engine = ring_under_a_rule(Assignment::Delegate);
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
    let owner = ring[0] as usize;
    // The key is served on its ring worker: the pin is confirmed.
    assert_eq!(dispatched(&engine, &pool, 1_000), ring[0], "the pin");
    // The pin is over its thresholds beside free workers.
    pool[owner].health.overloaded = true;
    let request = RequestFacts {
        now_micros: 2_000,
        ..keyed("k1")
    };
    let decision = decide(&engine, &request, &pool);
    let pick = *decision.pick().expect("a pick");
    assert_eq!(order(&decision)[..2], ring[..2], "the ring's order");
    assert_eq!(pick.session_verdict, SessionVerdict::Pinned, "no claim");
    // The host dispatches the decision to its alternate.
    let alternate = *pick.targets.iter().nth(1).expect("an alternate");
    let lease = Lease {
        issued_micros: 2_000,
        session: request.session(),
        session_verdict: pick.session_verdict,
        ..Lease::default()
    };
    engine.on_dispatched(&lease, &alternate, Side::Single, 0);
    // The worker is under its thresholds again: the key is still its.
    pool[owner].health.overloaded = false;
    assert_eq!(dispatched(&engine, &pool, 3_000), ring[0], "the pin stood");
}

/// A pin the lifted pass lands on leads no worker that carries the cap's
/// requests of the key. A dispatch on an alternate writes that worker
/// among the key's pins, and the cap bounds where a pin may go. The free
/// alternates stay behind the pin.
#[test]
fn a_kept_key_on_its_pin_names_no_alternate_at_the_rules_cap() {
    let engine = ring_under_a_cap(2);
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
    let owner = ring[0] as usize;
    assert_eq!(dispatched(&engine, &pool, 1_000), ring[0], "the pin");
    let request = RequestFacts {
        now_micros: 2_000,
        ..keyed("k1")
    };
    // The pin over its thresholds and under the cap. The ring's next
    // worker carries the cap's requests of the key.
    pool[owner].health.overloaded = true;
    pool[owner].session_in_flight = 1;
    pool[ring[1] as usize].session_in_flight = 2;
    let one = decide(&engine, &request, &pool);
    let verdict = one.pick().expect("a pick").session_verdict;
    assert_eq!(verdict, SessionVerdict::Pinned);
    assert_eq!(
        order(&one),
        [ring[0], ring[2], ring[3]],
        "the free alternates alone"
    );
    // Every other worker carries them: the pin alone.
    for &other in &ring[1..] {
        pool[other as usize].session_in_flight = 2;
    }
    assert_eq!(order(&decide(&engine, &request, &pool)), [ring[0]]);
}

/// No dispatch of a kept key's decision writes a worker at the cap as
/// the key's failover. Every worker but the pin carries the cap's
/// requests of the key, and the host dispatches the decision's last
/// target. Once the pin is down and the requests have ended, the key
/// follows no failover: the ring places it anew.
#[test]
fn a_kept_keys_dispatch_writes_no_failover_at_the_rules_cap() {
    let engine = ring_under_a_cap(2);
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
    let owner = ring[0] as usize;
    assert_eq!(dispatched(&engine, &pool, 1_000), ring[0], "the pin");
    let request = RequestFacts {
        now_micros: 2_000,
        ..keyed("k1")
    };
    pool[owner].health.overloaded = true;
    pool[owner].session_in_flight = 1;
    for &other in &ring[1..] {
        pool[other as usize].session_in_flight = 2;
    }
    let decision = decide(&engine, &request, &pool);
    let pick = *decision.pick().expect("a pick");
    assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
    let last = *pick.targets.as_slice().last().expect("a target");
    let lease = Lease {
        issued_micros: 2_000,
        session: request.session(),
        session_verdict: pick.session_verdict,
        ..Lease::default()
    };
    engine.on_dispatched(&lease, &last, Side::Single, 0);
    pool[owner].health.healthy = false;
    for candidate in &mut pool {
        candidate.session_in_flight = 0;
    }
    assert_eq!(
        dispatched(&engine, &pool, 3_000),
        ring[1],
        "no failover was written at the cap"
    );
}

/// The rule's in-flight cap holds for a pin the lifted pass lands on as
/// it holds for a pin read first: the lift is of the overload veto, and
/// of no other limit. A pin that carries the cap's requests of the key
/// puts the request to the ring again. Under its thresholds the ring
/// places the key on its pin. Over them the veto as it stands drops
/// the pin, and the ring's next worker takes the key. With every
/// worker over its thresholds the key stays on its pin, by the lift.
#[test]
fn a_followed_pin_at_the_rules_cap_puts_the_key_to_the_ring_again() {
    let engine = ring_under_a_cap(2);
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
    let owner = ring[0] as usize;
    assert_eq!(dispatched(&engine, &pool, 1_000), ring[0], "the pin");
    let request = RequestFacts {
        now_micros: 2_000,
        ..keyed("k1")
    };
    let verdict = |decision: &Decision| decision.pick().expect("a pick").session_verdict;
    // Under the cap a pin over its thresholds keeps the key.
    pool[owner].session_in_flight = 1;
    pool[owner].health.overloaded = true;
    let kept = decide(&engine, &request, &pool);
    assert_eq!(order(&kept)[0], ring[0], "under the cap");
    assert_eq!(verdict(&kept), SessionVerdict::Pinned);
    // The pin carries two requests of the key, the rule's cap.
    pool[owner].session_in_flight = 2;
    pool[owner].health.overloaded = false;
    // Under its thresholds the ring places the key on its pin again.
    let under = decide(&engine, &request, &pool);
    assert_eq!(order(&under)[0], ring[0], "under its thresholds");
    assert_eq!(verdict(&under), SessionVerdict::Pinned);
    // Over them the veto as it stands drops the pin: the ring's next
    // worker takes the key.
    pool[owner].health.overloaded = true;
    let over = decide(&engine, &request, &pool);
    assert_eq!(order(&over)[0], ring[1], "over its thresholds");
    assert_eq!(verdict(&over), SessionVerdict::Respill);
    assert_eq!(over.trace().lifted, LiftedVeto::None);
    // Every worker over its thresholds: the lifted pass places the key
    // on its pin again, and the placement stands by the lift.
    for candidate in &mut pool {
        candidate.health.overloaded = true;
    }
    let all = decide(&engine, &request, &pool);
    assert_eq!(order(&all)[0], ring[0], "every worker over");
    assert_eq!(verdict(&all), SessionVerdict::Pinned);
    assert_eq!(all.trace().lifted, LiftedVeto::Overload);
    // Every worker at the cap as well: the key stays on its pin.
    for candidate in &mut pool {
        candidate.session_in_flight = 2;
    }
    let stays = decide(&engine, &request, &pool);
    assert_eq!(order(&stays)[0], ring[0], "every worker at the cap");
    assert_eq!(verdict(&stays), SessionVerdict::Pinned);
}

/// A key handed on at the cap reads the veto as it stands: from a pin
/// over its thresholds it passes the ring's next worker where that
/// worker is over its thresholds beside free ones. Where the free
/// workers carry the cap too, the first of them still takes the
/// request and no pin moves: the lifted pass's survivors decide only
/// where the standing veto keeps no worker.
#[test]
fn a_key_handed_on_at_the_cap_passes_a_worker_over_its_thresholds() {
    let engine = ring_under_a_cap(2);
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
    let (owner, next) = (ring[0] as usize, ring[1] as usize);
    assert_eq!(dispatched(&engine, &pool, 1_000), ring[0], "the pin");
    let request = RequestFacts {
        now_micros: 2_000,
        ..keyed("k1")
    };
    // The pin carries the cap's two requests of the key, and the ring's
    // next worker is over its thresholds beside two free ones.
    pool[owner].session_in_flight = 2;
    pool[next].health.overloaded = true;
    // Under its thresholds the ring places the key on its pin again.
    let under = decide(&engine, &request, &pool);
    assert_eq!(order(&under)[0], ring[0], "the pin under its thresholds");
    pool[owner].health.overloaded = true;
    // The free workers carry the cap too. The veto as it stands still
    // keeps them: the ring's first of them takes the request, no veto
    // is lifted, and no pin moves to a worker at the cap. Asked ahead
    // of the respill below, which moves the pin where it is decided.
    pool[ring[2] as usize].session_in_flight = 2;
    pool[ring[3] as usize].session_in_flight = 2;
    let standing = decide(&engine, &request, &pool);
    assert_eq!(order(&standing)[0], ring[2], "a free worker at the cap");
    let pick = standing.pick().expect("a pick");
    assert_eq!(pick.session_verdict, SessionVerdict::None);
    assert_eq!(standing.trace().lifted, LiftedVeto::None);
    // The free workers under the cap: the key passes the ring's next
    // worker, over its thresholds, to the first of them.
    pool[ring[2] as usize].session_in_flight = 0;
    pool[ring[3] as usize].session_in_flight = 0;
    let over = decide(&engine, &request, &pool);
    assert_eq!(order(&over)[0], ring[2], "the pin over its thresholds");
    assert_eq!(over.trace().lifted, LiftedVeto::None);
}

/// A kept key handed on from the pin its lifted pass landed on moves
/// that pin where it is decided, as a respill from a pin read first
/// does: the worker taken leads the key's pins ahead of its dispatch,
/// and the key's next request, decided ahead of that dispatch, follows
/// it there as a pin.
#[test]
fn a_kept_key_handed_on_at_the_cap_fronts_its_worker_where_it_is_decided() {
    let engine = ring_under_a_cap(2);
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
    let (owner, next) = (ring[0] as usize, ring[1] as usize);
    assert_eq!(dispatched(&engine, &pool, 1_000), ring[0], "the pin");
    let request = RequestFacts {
        now_micros: 2_000,
        ..keyed("k1")
    };
    let pins = || engine.sessions().pins(request.session());
    assert_eq!(pins(), Some([pool[owner].id, WorkerId(0)]), "the pin read");
    // The pin carries the cap's two requests of the key, over its
    // thresholds: the veto as it stands drops it, the lifted pass lands
    // on it, and the ring's next worker takes the key.
    pool[owner].session_in_flight = 2;
    pool[owner].health.overloaded = true;
    let handed = decide(&engine, &request, &pool);
    assert_eq!(order(&handed)[0], ring[1]);
    let pick = handed.pick().expect("a pick");
    assert_eq!(pick.session_verdict, SessionVerdict::Respill);
    assert_eq!(
        pins(),
        Some([pool[next].id, pool[owner].id]),
        "ahead of its dispatch"
    );
    // The key's next request, decided ahead of that dispatch.
    let later = decide(&engine, &request, &pool);
    assert_eq!(order(&later)[0], ring[1]);
    let pick = later.pick().expect("a pick");
    assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
}

/// A kept key leaves its worker at the rule's cap alone, so a capped
/// rule keeps no key where the host writes no count of the key's
/// requests: the veto stands for the key as for any request, and a pin
/// over its thresholds yields it to the ring's next worker. The same
/// request with its count written is kept, and a rule with no cap reads
/// no count.
#[test]
fn a_capped_rule_keeps_no_key_where_the_host_writes_no_count() {
    let free_pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(
        &engine_of("consistent_hashing", false),
        &keyed("k1"),
        &free_pool,
    );
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    pool[ring[0] as usize].health.overloaded = true;
    // The key's pin on its ring worker, then a request of the key with
    // that worker over its thresholds beside three free ones.
    let asked = |engine: &Engine, counted: bool| {
        assert_eq!(dispatched(engine, &free_pool, 1_000), ring[0], "the pin");
        let request = RequestFacts {
            now_micros: 2_000,
            session_counted: counted,
            ..keyed("k1")
        };
        let decision = decide(engine, &request, &pool);
        (order(&decision)[0], decision.trace().lifted)
    };
    assert_eq!(
        asked(&ring_under_a_cap(2), false),
        (ring[1], LiftedVeto::None),
        "no count: the veto stands for the key"
    );
    assert_eq!(
        asked(&ring_under_a_cap(2), true),
        (ring[0], LiftedVeto::None),
        "the count written: the key is kept"
    );
    assert_eq!(
        asked(&ring_under_a_rule(Assignment::Delegate), false),
        (ring[0], LiftedVeto::None),
        "no cap: no count is read"
    );
}

/// A profile the engine does not hold cannot be marked: the index is
/// the validator's own, and a debug build refuses any other.
#[test]
#[cfg_attr(debug_assertions, should_panic(expected = "is not built"))]
fn a_profile_the_engine_does_not_hold_is_refused_in_a_debug_build() {
    let mut engine = engine_of("consistent_hashing", false);
    engine.places_by_key(7);
}
