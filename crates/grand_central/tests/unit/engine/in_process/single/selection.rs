//! The cases of how a single request is picked: the place that answers a
//! plain request, the pick that is noted where a stage takes a word, the
//! request the one pick does not decide, the ids a list names its workers by,
//! a list's rotation and its cell, a request that is declined by name, and a
//! request held to places.

use super::*;

/// The preset `name` with no filter in any of its profiles: a valid
/// document that weighs no worker's health.
fn unfiltered(name: &str) -> InProcess {
    let mut document = PolicySpec::preset(name).expect("a built-in preset");
    document.generation = 1;
    for profile in document.profiles.values_mut() {
        profile.filter.clear();
    }
    InProcess::new(AppliedSpec::validate(&document).expect("the document validates"))
}

/// Round robin's plain single request is one turn of the list's
/// rotation, answered as the place in the list the turn lands on: no
/// pass over the pool ran, no request was made, and no lease took an
/// id, since no stage of the document takes a word of one. The decision
/// a host writes of the place names that worker alone.
#[test]
fn round_robin_is_one_turn_of_the_list_answered_as_a_place() {
    let engine = preset("round_robin");
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let mut sized = request();
    sized.facts.prompt_tokens = 321;
    sized.facts.session_key = Identity::plain("a-key");
    let asks = Asks::of(sized, Some(&members));
    let mut seen = [0_u32; 4];
    for _ in 0..8 {
        let pick = placed(&engine, &asks);
        assert_eq!(pick.generation, 1);
        seen[pick.place as usize] += 1;
        let id = worker_id(list[pick.place as usize].as_ref());
        let decision = pick.decision(id, 3);
        let named = decision.pick().expect("a pick");
        assert_eq!(named.targets.len(), 1);
        assert!(named.trace.drew_first(), "no pass over the pool ran");
        assert_eq!(named.trace.seed, 0, "no seed was drawn");
        let target = named.targets.first().expect("the target");
        assert_eq!(
            (target.position, target.id, target.dp_rank),
            (pick.place, id, 3)
        );
    }
    assert_eq!(seen, [2; 4], "every worker takes its turn");
    assert_eq!(asks.made.get(), 0, "the pick read no fact of a request");
    assert_eq!(engine.leases.0.load(Ordering::Relaxed), 0, "no id taken");
}

/// Passthrough's plain single request is answered as a place: the first
/// worker of the list its host's test admits. That is the gateway's
/// classic rule, the position of the first worker whose routing state
/// the selection admits (`model_gateway/src/policies/passthrough.rs:48`
/// at main `882be6f4f`). No pass over the pool ran, no request was made
/// and no lease took an id. A list that admits no worker is the whole
/// decision's: the request is made once.
#[test]
fn passthrough_is_the_first_admitted_worker_answered_as_a_place() {
    let engine = preset("passthrough");
    for workers in [1, 2, 8, 64] {
        let list = pool(workers);
        let members = PublishedList::new(Arc::clone(&list));
        let asks = Asks::of(request(), Some(&members));
        for refused in 0..workers.min(12) {
            // Classic's rule, asked of the same worker objects.
            let classic = list.iter().position(|w| w.routing_state().eligible());
            assert_eq!(classic, Some(refused));
            let place = placed(&engine, &asks).place;
            assert_eq!(place as usize, refused, "{workers} workers");
            // The first is refused by its status alone, the others by overload.
            list[refused].set_overloaded(refused > 0);
            list[0].set_status(WorkerStatus::NotReady);
        }
        assert_eq!(asks.made.get(), 0, "the pick read no fact of a request");
        if workers <= 12 {
            // The whole decision's rule, not the first worker: a worker
            // that is not ready is never named, and the overload veto
            // is lifted off the workers it alone refuses.
            match decided(&engine, &asks).decision {
                Decision::Shed(shed) => assert_eq!((workers, shed.emptied_by), (1, "healthy")),
                Decision::Pick(pick) => {
                    assert_eq!(pick.trace.lifted, LiftedVeto::Overload);
                    assert!(pick.targets.iter().all(|target| target.position > 0));
                }
                Decision::Queue(_) => panic!("no queue is planned"),
            }
            assert_eq!(asks.made.get(), 1, "no worker is admitted: made once");
        }
    }
    assert_eq!(engine.leases.0.load(Ordering::Relaxed), 0, "no id taken");
}

/// Under a document a stage of which takes a word on a lease, a plain
/// request of a place one pick decides is still never made, and its
/// pick is noted: a whole decision of one target under a lease that
/// takes the next id, states the size the plain word states and no
/// session, and is told to the stage that takes the issue.
#[test]
fn a_place_picked_under_a_document_that_takes_a_word_is_noted() {
    let (engine, heard) = hearing_as(false, true, EventSet::LEASE);
    let document = engine.spec();
    let place = document.place_for(MODEL, Phase::Single);
    assert!(matches!(place, Place::Alone(_)), "{place:?}");
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let mut sized = request();
    (sized.facts.prompt_tokens, sized.facts.prompt_chars) = (321, 77);
    sized.facts.prompt_bytes = 154;
    sized.facts.session_key = Identity::plain("a-key");
    let asks = Asks::of(sized, Some(&members));
    let (mut seen, mut last) = ([0_u32; 4], 0);
    for _ in 0..8 {
        let decided = decided(&engine, &asks);
        let pick = decided.decision.pick().expect("a pick");
        assert_eq!(pick.targets.len(), 1);
        assert!(pick.trace.drew_first(), "no pass over the pool ran");
        assert_eq!(pick.trace.seed, 0, "no seed was drawn");
        let target = pick.targets.first().expect("the target");
        let worker = &list[target.position as usize];
        assert_eq!(target.id, worker_id(worker.as_ref()));
        seen[target.position as usize] += 1;
        let lease = decided.lease.expect("a lease").lease;
        assert_eq!(
            (lease.side, lease.booked, lease.target_index),
            (Side::Single, target.id, 0)
        );
        assert_eq!((lease.tokens, lease.chars, lease.bytes), (321, 77, 154));
        assert_eq!(lease.session, Session::NONE);
        assert_eq!(lease.session_verdict, SessionVerdict::None);
        assert!(lease.id.0 > last, "each pick takes the next id");
        last = lease.id.0;
        assert_eq!(decided.decode_session, Session::NONE);
        assert_eq!(heard.lock().last(), Some(&("issued", target.id, 0)));
    }
    assert_eq!(seen, [2; 4], "every worker takes its turn");
    assert_eq!(heard.lock().len(), 8, "one issue a pick");
    let taken = engine.leases.0.load(Ordering::Relaxed);
    assert_eq!(taken, ID_RUN, "eight ids, one write of the shared word");
    assert_eq!(asks.made.get(), 0, "the pick read no fact of a request");
}

/// Under a document whose session table alone takes a word on a lease,
/// a plain request of a place one pick decides is answered the place,
/// as under a document that takes no word: its binding states no rule,
/// so the table has no word for its pick. No request is made and no
/// lease takes an id.
#[test]
fn a_place_picked_under_a_table_that_alone_takes_a_word_is_the_whole_answer() {
    let document = PolicySpec::preset("round_robin").expect("a built-in preset");
    let spec = AppliedSpec::validate(&ruled_ahead(document)).expect("the document validates");
    let engine = InProcess::new(spec);
    let hears = engine.in_force().hears.intersects(EventSet::LEASE);
    let beside = engine.in_force().hears_beside_table;
    assert!(
        hears && !beside.intersects(EventSet::LEASE),
        "the session table alone takes a word"
    );
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let mut plain = request();
    plain.facts.session_key = Identity::plain("a-key");
    let asks = Asks::of(plain, Some(&members));
    placed(&engine, &asks);
    assert_eq!(asks.made.get(), 0, "the pick read no fact of a request");
    assert_eq!(engine.leases.0.load(Ordering::Relaxed), 0, "no id taken");
}

/// A request that is not plain, and a plain one the turn finds no
/// worker for, is made once and put to the whole pass. Under a document
/// that takes no word its pick has no lease either.
#[test]
fn a_request_the_one_pick_does_not_decide_is_made_once() {
    let engine = preset("round_robin");
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));

    // A retry is not plain: it is made, once a decision.
    let tried = [worker_id(list[0].as_ref())];
    let mut retry = request();
    retry.facts.tried = &tried;
    let asks = Asks::of(retry, Some(&members));
    let whole = decided(&engine, &asks);
    assert!(whole.decision.pick().is_some());
    assert!(whole.lease.is_none(), "no stage takes a word of it");
    assert_eq!(asks.made.get(), 1);
    assert_eq!(engine.leases.0.load(Ordering::Relaxed), 0, "no id taken");

    // A plain request whose turns land on no eligible worker is made
    // for the whole pass, which answers by the binding's own rule.
    for worker in list.iter() {
        worker.set_overloaded(true);
    }
    let asks = Asks::of(request(), Some(&members));
    let _ = decided(&engine, &asks);
    assert_eq!(asks.made.get(), 1);
}

/// A program that keeps ids of its own states each worker's with its
/// list's fixed facts. The list's records carry those ids whole, and
/// a record's key stays the low half of its address's fingerprint. A
/// whole decision names every target by the stated id, and passes
/// over the worker a retry names by it. A list handed with nothing
/// said names each worker by its address's fingerprint.
#[test]
fn a_list_names_each_worker_by_the_id_its_host_states() {
    use crate::engine::view::FixedFacts;

    // Ids past 32 bits: no half of one stands for it.
    let wide = |at: u64| WorkerId((1 << 40) + at);
    let engine = preset("round_robin");
    let list = pool(4);
    let place = |worker: &Arc<dyn Worker>| list.iter().position(|one| Arc::ptr_eq(one, worker));
    let members = PublishedList::with_facts(Arc::clone(&list), |worker| FixedFacts {
        id: place(worker).map(|at| wide(at as u64)),
        ..FixedFacts::default()
    });
    let records = engine
        .members_of(members.lent())
        .records(engine.standing_spec(), &engine.clock);
    let stated: Vec<WorkerId> = (0..4).map(wide).collect();
    assert_eq!(records.ids(), stated);
    for (row, worker) in records.fixed().iter().zip(list.iter()) {
        assert_eq!(row.key, worker_id(worker.as_ref()).0 as u32);
    }

    let tried = [stated[1]];
    let mut retry = request();
    retry.facts.tried = &tried;
    let asks = Asks::of(retry, Some(&members));
    let mut seen = [false; 4];
    for _ in 0..12 {
        let whole = decided(&engine, &asks);
        let pick = whole.decision.pick().expect("a pick");
        for target in pick.targets.iter() {
            assert_eq!(target.id, stated[target.position as usize]);
            assert_ne!(target.id, tried[0], "the tried one is no alternate");
        }
        let target = pick.targets.first().expect("a target");
        seen[target.position as usize] = true;
    }
    assert_eq!(seen, [true, false, true, true], "the tried one is passed");

    let unsaid = PublishedList::new(Arc::clone(&list));
    let prints: Vec<WorkerId> = list
        .iter()
        .map(|worker| worker_id(worker.as_ref()))
        .collect();
    let unsaid = engine.members_of(unsaid.lent());
    let records = unsaid.records(engine.standing_spec(), &engine.clock);
    assert_eq!(records.ids(), prints);
}

/// Zero is no worker's id: it reads as absent wherever an id travels,
/// and a session's table keeps no pin on it. A program that states
/// zero for a worker states none: the list's records name the worker
/// by its address's fingerprint, and a session placed on that worker
/// keeps its pin.
#[test]
fn a_list_names_a_worker_by_its_address_where_its_host_states_zero() {
    use crate::engine::view::FixedFacts;

    let engine = preset("manual");
    let list = pool(4);
    let members = PublishedList::with_facts(Arc::clone(&list), |_| FixedFacts {
        id: Some(WorkerId(0)),
        ..FixedFacts::default()
    });
    let prints: Vec<WorkerId> = list
        .iter()
        .map(|worker| worker_id(worker.as_ref()))
        .collect();
    let records = engine
        .members_of(members.lent())
        .records(engine.standing_spec(), &engine.clock);
    assert_eq!(records.ids(), prints);

    // The stock sessions rule: a key's first request is placed and
    // its dispatch pins the key, and the key's next request follows
    // the pin.
    let mut keyed = request();
    keyed.facts.session_key = Identity::hashed(41);
    let asks = Asks::of(keyed, Some(&members));
    let placed = served(&engine, &asks, Instant::now());
    assert_eq!(placed.id, prints[placed.position as usize]);
    let followed = decided(&engine, &asks).decision;
    let pick = followed.pick().expect("a pick");
    assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
    let again = pick.targets.first().expect("a target");
    assert_eq!(again.id, placed.id);
}

/// Round robin under a sessions rule that leaves a new session to the
/// profile (`delegate`) turns one rotation for a request with no routing
/// key and for a new session: in turn they take the list's workers one
/// after the other, under the draw's floor, at it and above it. The
/// gateway's classic override asks its one policy for both
/// (`model_gateway/src/policies/round_robin.rs:142-167`): for a request
/// with no key at `model_gateway/src/policies/registry.rs:471` and for a
/// new session at `:581`.
#[test]
fn a_plain_request_and_a_new_session_share_one_rotation_under_delegate() {
    let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
    let manual = PolicySpec::preset("manual").expect("a built-in preset");
    document.view.headers.clone_from(&manual.view.headers);
    let mut rule = manual.bindings[0].sessions.clone().expect("its rule");
    rule.assignment = Assignment::Delegate;
    document.bindings[0].sessions = Some(rule);
    let keys = ["key-a", "key-b", "key-c", "key-d", "key-e", "key-f"];
    for workers in [DRAW_FROM - 1, DRAW_FROM, 64] {
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        let engine = InProcess::new(spec);
        let list = pool(workers);
        let members = PublishedList::new(Arc::clone(&list));
        let mut places = Vec::new();
        for turn in 0..12_usize {
            let mut ask = request();
            if turn % 2 == 1 {
                ask.facts.session_key = Identity::plain(keys[turn / 2]);
            }
            let asks = Asks::of(ask, Some(&members));
            let decided = decided(&engine, &asks);
            let pick = decided.decision.pick().expect("a pick");
            places.push(pick.targets.first().expect("the target").position as usize);
        }
        let in_turn: Vec<usize> = (0..12).map(|turn| (places[0] + turn) % workers).collect();
        assert_eq!(places, in_turn, "{workers} workers");
    }
}

/// A whole decision turns the list's own word, the one a pick alone
/// turns, at every size: under the draw's floor, at it and above it.
/// Two lists of one model and one size so turn a rotation each: asked
/// in turn, a request with no routing key and a new session at each,
/// they take each list's workers one after the other. And over one
/// list a pick alone and a new session's whole decision share one
/// rotation: the pick is asked for a model the same profile is bound
/// for with no rule, since a place under a rule is not one pick's.
/// The gateway's classic policy keeps its rotation by the list it is
/// handed whole, whatever its size
/// (`model_gateway/src/policies/round_robin.rs:95-99`, `:151`).
#[test]
fn each_list_turns_its_own_rotation_for_every_request() {
    let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
    let manual = PolicySpec::preset("manual").expect("a built-in preset");
    document.view.headers.clone_from(&manual.view.headers);
    let mut rule = manual.bindings[0].sessions.clone().expect("its rule");
    rule.assignment = Assignment::Delegate;
    let mut unruled = document.bindings[0].clone();
    unruled.matcher.model = "unruled-*".to_owned();
    document.bindings.insert(0, unruled);
    document.bindings[1].sessions = Some(rule);
    let keys: Vec<String> = (0..48).map(|key| format!("key-{key}")).collect();
    let mut keys = keys.iter();
    for workers in [3, DRAW_FROM - 1, DRAW_FROM, 64] {
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        let engine = InProcess::new(spec);
        let Place::Alone(profile) = engine.spec().place_for("unruled-1", Phase::Single) else {
            panic!("one pick of round robin decides the place");
        };
        let (one, other) = (pool(workers), pool(workers));
        let lists = [
            PublishedList::new(Arc::clone(&one)),
            PublishedList::new(Arc::clone(&other)),
        ];
        let mut whole = |members: &PublishedList, keyed: bool| {
            let mut ask = request();
            if keyed {
                ask.facts.session_key = Identity::plain(keys.next().expect("a key"));
            }
            let asks = Asks::of(ask, Some(members));
            let decided = decided(&engine, &asks);
            let pick = decided.decision.pick().expect("a pick");
            pick.targets.first().expect("the target").position as usize
        };
        let mut places = [Vec::new(), Vec::new()];
        for turn in 0..8 {
            for (list, members) in lists.iter().enumerate() {
                places[list].push(whole(members, turn % 2 == 1));
            }
        }
        for places in &places {
            let in_turn: Vec<usize> = (0..8).map(|turn| (places[0] + turn) % workers).collect();
            assert_eq!(*places, in_turn, "{workers} workers, two lists");
        }
        // The first list's rotation goes on, a pick alone and a new
        // session's whole decision in turn.
        let mut shared = Vec::new();
        for turn in 0..8 {
            shared.push(if turn % 2 == 0 {
                let alone = engine.pick(
                    &engine.spec(),
                    profile,
                    lists[0].lent(),
                    RoutingState::eligible,
                );
                alone.expect("a worker")
            } else {
                whole(&lists[0], true)
            });
        }
        let in_turn: Vec<usize> = (8..16)
            .map(|turn| (places[0][0] + turn) % workers)
            .collect();
        assert_eq!(shared, in_turn, "{workers} workers, one list");
    }
}

/// Two callers of one engine ask in strict turn over one list, and each
/// is held to the places of the list it may take
/// (`Constraints::places`): every record but two. Each caller is held
/// away from a record at each parity, so whichever parity its turns
/// land on, a record it may not take stands there. Each caller's 112
/// requests hand each of its fourteen records eight and its two held
/// records none: a record one caller is held away from keeps the other
/// caller's share. The picks are counted a caller.
#[test]
fn two_callers_in_strict_turn_leave_each_others_held_records_their_share() {
    let away: [[u32; 2]; 2] = [[0, 1], [2, 3]];
    let places = away.map(|held| {
        (0..16)
            .filter(|at| !held.contains(at))
            .collect::<Vec<u32>>()
    });
    let engine = preset("round_robin");
    let list = pool(16);
    let members = PublishedList::new(Arc::clone(&list));
    let mut picked = [[0_u32; 16]; 2];
    for turn in 0..224_usize {
        let mut ask = request();
        ask.constraints.places = Some(&places[turn % 2]);
        let asks = Asks::of(ask, Some(&members));
        let decided = decided(&engine, &asks);
        let pick = decided.decision.pick().expect("a ready record");
        picked[turn % 2][pick.targets.first().expect("a target").position as usize] += 1;
    }
    for (caller, held) in away.iter().enumerate() {
        let mut want = [8_u32; 16];
        for &at in held {
            want[at as usize] = 0;
        }
        assert_eq!(picked[caller], want, "caller {caller}");
    }
}

/// A request that hands no list is decided over no worker. One that
/// states what no stage here runs is declined under that stage's name.
#[test]
fn what_no_stage_runs_is_answered_by_name() {
    let engine = preset("round_robin");
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let shed = |request: Request<'_>, members: Option<&PublishedList>| {
        let asks = Asks::of(request, members);
        match decided(&engine, &asks).decision {
            Decision::Shed(shed) => (shed.reason, shed.emptied_by),
            other => panic!("a shed, not {other:?}"),
        }
    };
    assert_eq!(shed(request(), None), (ShedReason::NoEndpoints, NO_STAGE));
    let mut hinted = request();
    hinted.constraints.subset_hint = Some(1);
    assert_eq!(
        shed(hinted, Some(&members)),
        (ShedReason::PolicyDeclined, SUBSET_HINT)
    );
    // The list's workers are dialed over HTTP.
    let mut dialed = request();
    dialed.constraints.wire_modes = WireModes::GRPC;
    assert_eq!(
        shed(dialed, Some(&members)),
        (ShedReason::PolicyDeclined, WIRES)
    );
    dialed.constraints.wire_modes = WireModes::HTTP.union(WireModes::GRPC);
    let asks = Asks::of(dialed, Some(&members));
    assert!(matches!(answered(&engine, &asks), Decided::Alone(_)));
}

/// A list's cell is asked where the engine decides over the list, and
/// nowhere else. A request the engine refuses by its phase, a pair's
/// decode sent on its own, asks no
/// cell, plain or not, at either entry: its list's cell stays empty. A
/// maker that names its list while it makes the request is decided over
/// that list. So is a plain maker that lends one list ahead of its
/// making and another behind it: the pick ahead of the making found no
/// ready worker in the first, and the value it asked there is not the
/// one the request is decided over.
#[test]
fn a_cell_is_asked_where_its_list_is_decided_over() {
    /// A maker that names its list while it makes the request.
    struct Late<'a>(Request<'a>, &'a PublishedList, Cell<bool>);

    impl RequestMaker for Late<'_> {
        fn model(&self) -> &str {
            self.0.facts.model
        }

        fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
            self.2.set(true);
            self.0
        }
    }

    impl LendsLists for Late<'_> {
        fn workers(&self) -> WorkerLists<'_> {
            if self.2.get() {
                WorkerLists::Single(self.1.lent())
            } else {
                WorkerLists::None
            }
        }
    }

    /// A plain maker that lends its first list ahead of its making
    /// and its second behind it.
    struct Moved<'a>(Request<'a>, [&'a PublishedList; 2], Cell<bool>);

    impl RequestMaker for Moved<'_> {
        fn model(&self) -> &str {
            self.0.facts.model
        }

        fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
            self.2.set(true);
            self.0
        }

        fn plain(&self) -> Option<Plain> {
            self.0.plain()
        }
    }

    impl LendsLists for Moved<'_> {
        fn workers(&self) -> WorkerLists<'_> {
            WorkerLists::Single(self.1[usize::from(self.2.get())].lent())
        }
    }

    let engine = preset("round_robin");
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let now = Instant::now();
    // Under a pair plan a decode is its pair's side: sent on its own
    // over one list it is refused, and the list's cell is not asked.
    let document =
        PolicySpec::pair_presets("round_robin", "round_robin").expect("the pair presets");
    let paired = InProcess::new(AppliedSpec::validate(&document).expect("a valid document"));
    let mut side = request();
    side.facts.phase = Phase::Decode;
    // A plain request, and the same one from a maker that does not say
    // so.
    let plain = Asks::of(side, Some(&members));
    assert!(plain.plain().is_some());
    let leased = paired.decide_under(GatewayId::LOCAL, &plain, now);
    assert_eq!(leased.err(), Some(DecisionRefused::PairSide));
    let whole = Whole(Asks::of(side, Some(&members)));
    let leased = paired.decide_under(GatewayId::LOCAL, &whole, now);
    assert_eq!(leased.err(), Some(DecisionRefused::PairSide));
    let bare = paired.decide_without_lease(&whole, now);
    assert_eq!(bare.err(), Some(DecisionRefused::PairSide));
    assert_eq!(paired.values_made(), 0, "no cell was asked");
    assert!(!members.filled(), "the list's cell stays empty");

    let late = Late(request(), &members, Cell::new(false));
    let leased = engine.decide_under(GatewayId::LOCAL, &late, now);
    let (whole, _) = leased.expect("an answer").whole().expect("no place");
    assert!(whole.decision.pick().is_some(), "over the list it named");
    let late = Late(request(), &members, Cell::new(false));
    let (bare, _) = engine.decide_without_lease(&late, now).expect("an answer");
    assert!(bare.pick().is_some(), "over the list it named");
    assert_eq!(engine.values_made(), 1, "one value of the one list");

    let idle = pool(2);
    for worker in idle.iter() {
        worker.set_status(WorkerStatus::NotReady);
    }
    let ahead = PublishedList::new(Arc::clone(&idle));
    let moved = Moved(request(), [&ahead, &members], Cell::new(false));
    assert!(moved.plain().is_some());
    let leased = engine.decide_under(GatewayId::LOCAL, &moved, now);
    let (whole, _) = leased.expect("an answer").whole().expect("no place");
    assert!(whole.decision.pick().is_some(), "over the list lent behind");
    assert!(moved.2.get() && ahead.filled(), "the pick asked the first");
}

/// A request its host holds to places of its list is made and decided
/// over the list's own records: every target stands at one of the
/// places, and each of them takes requests. Under a rotation, two draws
/// and a profile that scores, each as it ships and with no filter, over
/// a list under the draw's floor and one over it.
#[test]
fn a_request_held_to_places_is_handed_no_other_place_under_any_document() {
    for name in ["round_robin", "random", "power_of_two", "least_load"] {
        for bare in [false, true] {
            let engine = if bare { unfiltered(name) } else { preset(name) };
            for workers in [8, 32] {
                let list = pool(workers);
                let members = PublishedList::new(Arc::clone(&list));
                let mut asks = Asks::of(request(), Some(&members));
                asks.request.constraints.places = Some(&[3, 6]);
                let mut first = [0_u32; 8];
                for seed in 0..32 {
                    asks.request.facts.seed = seed;
                    let decided = decided(&engine, &asks);
                    let pick = decided.decision.pick().expect("a pick");
                    let mut places = pick.targets.iter().map(|target| target.position);
                    assert!(places.all(|place| place == 3 || place == 6), "{name}");
                    let lead = pick.targets.first().expect("a target").position;
                    first[lead as usize] += 1;
                }
                assert!(first[3] > 0 && first[6] > 0, "{name}: {first:?}");
                assert_eq!(first[3] + first[6], 32, "{name}: {first:?}");
                assert_eq!(asks.made.get(), 32, "{name}: such a request is made");
            }
        }
    }
}

/// A keyed request held to places is handed the next members of the
/// stock ring. Held to every member but its key's owner, it is handed
/// the members a request held to nothing is named behind the owner, in
/// that order: as many targets, and never the owner.
#[test]
fn a_keyed_request_held_to_places_takes_the_rings_next_members() {
    for engine in [
        preset("consistent_hashing"),
        unfiltered("consistent_hashing"),
    ] {
        let list = pool(16);
        let members = PublishedList::new(Arc::clone(&list));
        let places = |decided: &DecideOutcome| -> Vec<u32> {
            let pick = decided.decision.pick().expect("a pick");
            pick.targets.iter().map(|target| target.position).collect()
        };
        for key in 1..=40 {
            let mut keyed = request();
            keyed.facts.seed = key;
            keyed.facts.session_key = Identity::hashed(key);
            let whole = places(&decided(&engine, &Asks::of(keyed, Some(&members))));
            assert_eq!(whole.len(), MAX_TARGETS, "key {key}");
            let others: Vec<u32> = (0..16).filter(|&place| place != whole[0]).collect();
            let mut held = Asks::of(keyed, Some(&members));
            held.request.constraints.places = Some(&others);
            let part = places(&decided(&engine, &held));
            assert_eq!(part.len(), MAX_TARGETS, "key {key}");
            assert_eq!(part[..MAX_TARGETS - 1], whole[1..], "key {key}");
            assert!(!part.contains(&whole[0]), "key {key}: the owner is handed");
        }
    }
}
