//! Cases of a decode call behind a pair, under no sessions rule. A decode
//! call's ask [`decide_decode`] behind the pair its lease holds, and
//! those that issue its side ask [`issue_decode`] behind it. One case
//! holds a `Dispatched` frame on a decode side that stands chosen: a call
//! goes behind the first of its three pairs. A decode call under a
//! sessions rule has its cases in `sessions`, and the one behind an
//! encode request in `encode::sessions`.

use super::*;

/// The `n`th pair of [`GATEWAY`]'s leased with its decode asked for later,
/// and its prefill dispatched to the pick's target at `index`: the pair's
/// decision and the prefill side's lease.
fn held_pair_at(
    store: &MemoryStore,
    objects: &Objects<Members>,
    (n, index): (u8, u8),
    now: Instant,
) -> (Decision, Lease) {
    let mut frame = at_prefill(n, WireModes::NONE, None);
    frame.decode_later = true;
    let (decision, prefill, decode) = leased_pair(store, objects, frame, now);
    assert_eq!(decode, None, "the decode side stands chosen");
    let sent = Dispatch {
        side: Side::Prefill,
        target_index: index,
        since_decision_micros: 0,
    };
    assert!(LeasesStore::dispatched(
        store, GATEWAY, prefill.id, sent, now
    ));
    (decision, prefill)
}

/// A decode call is decided over its model's two lists, and the decode
/// side's list alone is the asking gateway's to take, by that side's reach
/// and standing vetoes. Another gateway alone dials one decode and the
/// asking gateway holds another: every call's pick names the one decode
/// left. The held prefill was sent already and is found wherever it stands:
/// a hold of the asking gateway on it sheds a pair at its prefill and takes
/// nothing from the call, whose every target names it.
#[test]
fn a_decode_call_takes_the_decode_side_alone_and_finds_its_prefill_under_a_hold() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(stock_pair_plan())
        .expect("above the unbound start");
    let p = side_member(&store, &objects, "p0", Role::Prefill, None);
    let decodes = [
        side_member(&store, &objects, "d0", Role::Decode, None),
        side_member(&store, &objects, "d1", Role::Decode, None),
        side_member(&store, &objects, "d2", Role::Decode, Some(OTHER)),
    ];
    let now = Instant::now();
    let until = Some(now + Duration::from_secs(60));
    store.gateway_veto(GATEWAY, p, until);
    store.gateway_veto(GATEWAY, decodes[0], until);
    let frame = at_prefill(1, WireModes::NONE, None);
    let pair = served_pair(&store, &objects, frame, GATEWAY, now);
    let at_health = Some((ShedReason::NoEndpoints, "healthy"));
    assert_eq!(shed_of(&pair), at_health, "a pair at its prefill");
    let unread = Arc::new(LoadSnapshot::default());
    let named = |gateway: GatewayId| -> BTreeSet<(WorkerId, WorkerId)> {
        let mut named = BTreeSet::new();
        for n in 1..=32 {
            let frame = at_call(n, LeaseId(1), WireModes::NONE, None);
            let (asked, given) = ((gateway, now), (&unread, None));
            let answer = served_call(&store, &objects, frame, asked, (p, None), given);
            let Decision::Pick(pick) = &answer else {
                panic!("call {n} is answered no pick: {answer:?}");
            };
            let pairs = pairs_of(&answer);
            assert_eq!(pairs.len(), pick.targets.len(), "a decode a target");
            named.extend(pairs);
        }
        named
    };
    assert_eq!(named(GATEWAY), BTreeSet::from([(p, decodes[1])]));
    let every: BTreeSet<(WorkerId, WorkerId)> = decodes.iter().map(|&d| (p, d)).collect();
    assert_eq!(named(OTHER), every);
}

/// The answers of a decode call come in a pair's order, and the decode side
/// alone states them. A model with no decode and a model nobody serves are
/// decided over no worker, whatever the gateway states. Else decodes the
/// gateway dials, whose every worker its wires or a hint of 0 take, shed
/// the call under that stage. Else decodes it dials none of are decided
/// over no worker. What the gateway dials of the prefills says nothing: the
/// held prefill was sent.
#[test]
fn a_decode_call_answers_no_decode_then_the_decodes_emptied_then_none_dialed() {
    let now = Instant::now();
    let unread = Arc::new(LoadSnapshot::default());
    // GATEWAY's call over a fleet of one prefill, which `prefill` alone
    // dials, and of `decodes`, with the engine's own answer over no worker
    // for the same call.
    type Fleet<'a> = (Option<GatewayId>, &'a [Option<GatewayId>]);
    let answered = |(prefill, decodes): Fleet<'_>, wires, hint: Option<u32>| {
        let store = Arc::new(MemoryStore::new());
        let objects = holder(&store);
        store
            .apply(stock_pair_plan())
            .expect("above the unbound start");
        let p = side_member(&store, &objects, "p0", Role::Prefill, prefill);
        for (n, &only) in decodes.iter().enumerate() {
            side_member(&store, &objects, &format!("d{n}"), Role::Decode, only);
        }
        let frame = at_call(1, LeaseId(1), wires, hint);
        let request = Checked::new(frame.clone()).expect("a frame the record holds");
        let (over_none, traced) = store.deciding(|world| {
            let mut lists = Lists::default();
            let read = lists.of(&request);
            let maker = FrameRequest::new(&request, &read, BUDGET, WorkerLists::None);
            let held = Target {
                id: p,
                ..Target::default()
            };
            let decided = world.front().decide_decode_without_lease(
                &maker,
                &held,
                Default::default(),
                0,
                now,
            );
            let decided = decided.expect("the engine decides");
            let trace = decided.trace();
            (shed_of(&decided), (trace.seed, trace.generation))
        });
        assert!(over_none.is_some(), "no worker is no pick");
        assert_eq!(traced.0, 1, "the frame's own seed");
        let (asked, given) = ((GATEWAY, now), (&unread, None));
        let answer = served_call(&store, &objects, frame, asked, (p, None), given);
        // A shed the thread makes itself states what the engine's states:
        // the frame's seed and the document's generation.
        if shed_of(&answer).is_some() {
            let trace = answer.trace();
            assert_eq!((trace.seed, trace.generation), traced);
        }
        (answer, over_none)
    };
    let (http, grpc) = (WireModes::HTTP, WireModes::GRPC);
    let stated = [(WireModes::NONE, None), (grpc, None), (http, Some(0))];
    // No decode at all: over no worker, whatever the gateway states.
    for (wires, hint) in stated {
        let (answer, over_none) = answered((None, &[]), wires, hint);
        assert_eq!(shed_of(&answer), over_none, "{wires:?}, a hint of {hint:?}");
    }
    // A decode the gateway dials and its wires or its hint take: shed under
    // that stage, whoever dials the prefill.
    let wires = Some((ShedReason::NoEndpoints, WIRES));
    let hinted = Some((ShedReason::NoEndpoints, SUBSET_HINT));
    for prefill in [None, Some(OTHER)] {
        let fleet: Fleet<'_> = (prefill, &[None]);
        assert_eq!(shed_of(&answered(fleet, grpc, None).0), wires);
        assert_eq!(shed_of(&answered(fleet, http, Some(0)).0), hinted);
    }
    // A decode another gateway alone dials: over no worker, whatever the
    // gateway states.
    for (wires, hint) in stated {
        let (answer, over_none) = answered((None, &[Some(OTHER)]), wires, hint);
        assert_eq!(shed_of(&answer), over_none, "{wires:?}, a hint of {hint:?}");
    }
    // A decode the gateway dials: a pick behind the held prefill, though
    // another gateway alone dials that prefill.
    for prefill in [None, Some(OTHER)] {
        let (answer, _) = answered((prefill, &[None]), http, None);
        assert!(matches!(answer, Decision::Pick(_)), "{answer:?}");
    }
    // A model nobody serves: over no worker.
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(stock_pair_plan())
        .expect("above the unbound start");
    let frame = at_call(1, LeaseId(1), WireModes::NONE, None);
    let (asked, given) = ((GATEWAY, now), (&unread, None));
    let held = (WorkerId(7), None);
    let answer = served_call(&store, &objects, frame, asked, held, given);
    assert!(shed_of(&answer).is_some(), "{answer:?}");
}

/// The pair plan's document whose sides go by the least load, the best
/// first, among the workers the health filter keeps.
fn least_load_pair_plan() -> AppliedSpec {
    let document = r#"{
      "apiVersion": "smg.policy/v1",
      "generation": 1,
      "profiles": {
        "side": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" }
      },
      "bindings": [
        { "match": { "phase": ["prefill", "decode"] }, "plan": "pd-pair",
          "profiles": { "prefill": "side", "decode": "side" } }
      ]
    }"#;
    let document: PolicySpec = serde_json::from_str(document).expect("a document of this build");
    staged(&document, &StageRegistry::builtin())
}

/// A decode call's request is given the load monitor's snapshot as its
/// pair's first decision is: under a plan whose sides go by the least load,
/// every call's pick names first the decode whose report shows no queue.
/// Under a snapshot that holds no report no decode stands out, and the
/// calls name more than one first.
#[test]
fn a_decode_calls_request_reads_the_reports_of_the_snapshot_it_is_given() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(least_load_pair_plan())
        .expect("above the unbound start");
    let p = side_member(&store, &objects, "p0", Role::Prefill, None);
    let decodes =
        ["d0", "d1", "d2"].map(|name| side_member(&store, &objects, name, Role::Decode, None));
    let now = Instant::now();
    // The second decode reports no queue, and every other worker is deep
    // in its own.
    let reports = ["p0", "d0", "d1", "d2"].map(|name| {
        let load = if name == "d1" {
            one_rank_load(0, 0.1, 10.0)
        } else {
            one_rank_load(900_000, 0.9, 10.0)
        };
        (format!("http://{name}:8000"), load)
    });
    let published = LoadSnapshot::from_loads_for_test(reports.into_iter().collect());
    let unread = Arc::new(LoadSnapshot::default());
    let firsts = |loads: &Arc<LoadSnapshot>| -> BTreeSet<WorkerId> {
        let calls = (1..=32).map(|n| at_call(n, LeaseId(1), WireModes::NONE, None));
        let picks = calls.map(|frame| {
            let (asked, given) = ((GATEWAY, now), (loads, None));
            served_call(&store, &objects, frame, asked, (p, None), given)
        });
        let firsts = picks.map(|pick| pairs_of(&pick).first().expect("a pick").1);
        firsts.collect()
    };
    assert_eq!(firsts(&published), BTreeSet::from([decodes[1]]));
    assert!(firsts(&unread).len() > 1, "no report, and one decode alone");
}

/// A decode call whose decode profile holds a stage that may refuse by a
/// report of leases is decided behind an apply of what waits at the other
/// files, where its decider lent its place. The prefill profile's stages
/// say nothing of it: the call runs none of them, and a frame left under
/// another file's watch waits for that file's look.
#[test]
fn a_decode_call_whose_decode_side_may_refuse_by_reports_applies_what_waits_at_the_other_files() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("by-reports", || Box::new(ByReports))
        .expect("a free name");
    let cases = [
        (["random", "by-reports"], LookOutcome::Quiet),
        (["by-reports", "random"], LookOutcome::Fed),
        (["random", "random"], LookOutcome::Fed),
    ];
    for (picks, waited) in cases {
        let store = Arc::new(MemoryStore::new().with_decision_slots(2));
        let objects = holder(&store);
        store
            .apply(pair_plan(&catalog, "[]", picks))
            .expect("above the unbound start");
        let p = side_member(&store, &objects, "p0", Role::Prefill, None);
        side_member(&store, &objects, "d0", Role::Decode, None);
        LeasesStore::set_term(&*store, 1);
        let mut watch = a_frame_under_the_second_files_watch(&store, p);
        // The first file's decision, its place lent.
        let _acting = DecisionSlot::new(0).enter();
        let place = store.at_work();
        let frame = at_call(1, LeaseId(1), WireModes::NONE, None);
        let unread = Arc::new(LoadSnapshot::default());
        let (asked, given) = ((GATEWAY, Instant::now()), (&unread, Some(&place)));
        let decision = served_call(&store, &objects, frame, asked, (p, None), given);
        assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
        assert_eq!(watch.look(), waited, "{picks:?}");
    }
}

/// A decode call starts from the decode its pair's first decision chose for
/// the target the prefill's dispatch took: the store's answer for the lease
/// reaches the engine. Under a decode profile that takes turns, and so
/// keeps its pick, the call answers that decode again, the pick's first or
/// the one the dispatch named, where a new turn names another. The side is
/// issued on it under the pair's own id.
#[test]
fn a_decode_call_starts_from_the_decode_of_the_target_its_prefills_dispatch_took() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    let turns = ["random", "round-robin"];
    store
        .apply(pair_plan(&StageRegistry::builtin(), "[]", turns))
        .expect("above the unbound start");
    let p = side_member(&store, &objects, "p0", Role::Prefill, None);
    for name in ["d0", "d1", "d2"] {
        side_member(&store, &objects, name, Role::Decode, None);
    }
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    for (n, index) in [(1, 0), (2, 1)] {
        let (first, prefill) = held_pair_at(&store, &objects, (n, index), now);
        let chosen = pairs_of(&first);
        assert!(chosen.len() > 1 && chosen[0].1 != chosen[1].1, "{first:?}");
        let took = chosen[usize::from(index)].1;
        let held = store.held_pair(GATEWAY, prefill.id);
        assert_eq!(
            held.map(|(prefill, decode, ..)| (prefill, decode.map(|at| at.0))),
            Ok((p, Some(took)))
        );
        let frame = at_call(n + 10, prefill.id, WireModes::NONE, None);
        let (again, issued) = leased_call(&store, &objects, frame, now);
        let named = pairs_of(&again);
        assert_eq!(named.first(), Some(&(p, took)), "the target at {index}");
        let (side, _) = issued.expect("a pick's issue");
        let lease = (side.id, side.side, side.booked);
        assert_eq!(lease, (prefill.id, Side::Decode, took));
        // The side keeps the call's own pick, for a dispatch by its index.
        let on_file = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
        let targets = &again.pick().expect("a pick").targets;
        assert!(targets.len() > 1, "a target behind the first: {again:?}");
        for (at, target) in (0u8..).zip(targets.iter()) {
            let pair = target.pair.expect("a decode a target");
            let kept = on_file.target_of(Side::Decode, at);
            let named = (pair.decode_id, pair.decode_dp_rank);
            assert_eq!(kept, Some(named), "the call's target at {at}");
        }
    }
}

/// A hold of the asking gateway on the only decode sheds a decode call
/// under the health filter: no rule lifts a hold, where a doubt of every
/// decode is lifted as the last resort. Another gateway holds no veto, and
/// the call is its pick.
#[test]
fn a_hold_on_the_only_decode_sheds_a_decode_call_under_the_health_filter() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(stock_pair_plan())
        .expect("above the unbound start");
    let p = side_member(&store, &objects, "p0", Role::Prefill, None);
    let d = side_member(&store, &objects, "d0", Role::Decode, None);
    let now = Instant::now();
    let unread = Arc::new(LoadSnapshot::default());
    let answered = |gateway| {
        let frame = at_call(1, LeaseId(1), WireModes::NONE, None);
        let (asked, given) = ((gateway, now), (&unread, None));
        served_call(&store, &objects, frame, asked, (p, None), given)
    };
    assert_eq!(pairs_of(&answered(GATEWAY)), [(p, d)]);
    store.gateway_veto(GATEWAY, d, Some(now + Duration::from_secs(60)));
    let shed = Some((ShedReason::NoEndpoints, "healthy"));
    assert_eq!(shed_of(&answered(GATEWAY)), shed);
    assert_eq!(pairs_of(&answered(OTHER)), [(p, d)]);
}

/// A decode call's pick is issued under its pair's own lease: the decode
/// side on the pick's first decode, under the session the frame's reader
/// states of a request of the decode phase and with the call's count of
/// tokens, and that decode's worker is booked. The stages hear the issue
/// dated at the call, three seconds behind the pair's own, and the
/// lease's dispatch deadline counts from the call's instant. A call that
/// sheds files nothing. The issue's undo puts the side and the deadline
/// back as the call found them and frees the booking, and the prefill's
/// stands throughout.
#[test]
fn a_decode_calls_pick_is_issued_under_its_pairs_lease_and_told_dated_at_the_call() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    let told = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&told);
    let mut catalog = StageRegistry::builtin();
    let stage = move || -> Box<dyn Picker> { Box::new(Told(Arc::clone(&kept))) };
    catalog.picker("told", stage).expect("a free name");
    store
        .apply(pair_plan(&catalog, r#"["healthy"]"#, ["random", "told"]))
        .expect("above the unbound start");
    let p = side_member(&store, &objects, "p0", Role::Prefill, None);
    let d = side_member(&store, &objects, "d0", Role::Decode, None);
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    let (_, prefill) = held_pair_at(&store, &objects, (1, 0), now);
    assert_eq!(*told.lock(), [prefill], "the prefill side's issue alone");
    // What the leases on file book on the prefill and on the decode.
    let booked = || {
        let load = |name: &str| {
            let url = format!("http://{name}:8000");
            let object = objects.registry().get_by_url(&url).expect("its object");
            object.load()
        };
        [load("p0"), load("d0")]
    };
    let side_on_file = || {
        let sides = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
        sides.decode.map(|side| (side.state, side.target))
    };
    let chosen = Some((LeaseState::Chosen, d));
    assert_eq!((side_on_file(), booked()), (chosen, [1, 0]));
    // The dispatch deadline of the lease on file.
    let deadline = || {
        let sides = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
        sides.record.dispatch_deadline
    };
    let pairs = deadline();
    // The call's frame of one conversation, at `phase`: its key goes plain,
    // so its reader makes the session of the phase it takes the frame at.
    let framed = |phase: proto::Phase| {
        let mut frame = at_call(2, prefill.id, WireModes::NONE, None);
        let facts = frame.facts.as_mut().expect("the case's frame has facts");
        (facts.phase, facts.prompt_tokens) = (phase.into(), 300);
        facts.session_key = Some(proto::Identity {
            form: Some(proto::identity::Form::Plain("conversation-7".to_owned())),
        });
        frame
    };
    // The session of a request of the decode phase, as the reader of a
    // frame that states that phase and names no lease makes it.
    let session = {
        let mut stated = framed(proto::Phase::Decode);
        stated.decode_of = None;
        let request = Checked::new(stated).expect("a frame the record holds");
        let mut lists = Lists::default();
        let read = lists.of(&request);
        request.facts(&read, 0, 0).session()
    };
    let call = now + Duration::from_secs(3);

    // A call that sheds files nothing, and no stage hears of it.
    let hinted = at_call(3, prefill.id, WireModes::HTTP, Some(0));
    let (decision, issued) = leased_call(&store, &objects, hinted, call);
    let at_hint = Some((ShedReason::NoEndpoints, SUBSET_HINT));
    assert_eq!((shed_of(&decision), issued.is_none()), (at_hint, true));
    assert_eq!((side_on_file(), booked()), (chosen, [1, 0]));
    assert_eq!(told.lock().len(), 1);
    assert_eq!(deadline(), pairs, "a shed starts no deadline");

    // The call's frame states the prefill phase: a frame that names a
    // lease is read at the decode phase whatever it states.
    let (decision, issued) = leased_call(&store, &objects, framed(proto::Phase::Prefill), call);
    assert_eq!(pairs_of(&decision), [(p, d)]);
    let (side, issue) = issued.expect("a pick's issue");
    let beside = Lease {
        side: Side::Decode,
        booked: d,
        session,
        tokens: 300,
        ..prefill
    };
    assert_eq!(side, beside, "the pair's own id, generation and instant");
    let dated = Lease {
        issued_micros: prefill.issued_micros + 3_000_000,
        ..side
    };
    assert_eq!(*told.lock(), [prefill, dated], "dated at the call");
    let issued = Some((LeaseState::ISSUED, d));
    assert_eq!((side_on_file(), booked()), (issued, [1, 1]));
    // The side's five seconds count from the call, not from the pair's
    // first decision three seconds earlier.
    assert_eq!(deadline(), call + DISPATCH_TTL);
    assert_ne!(deadline(), pairs);

    assert!(store.unissue_decode(GATEWAY, prefill.id, issue));
    assert_eq!((side_on_file(), booked()), (chosen, [1, 0]));
    assert_eq!(deadline(), pairs, "the deadline the call found");
}

/// A stage that panics at the issue a decode call tells it is held and
/// counted, as under a report: the side stands issued on file and the call
/// still gets its lease. The panic's payload panics again at its own drop,
/// and that is held too. An issue whose telling returns ends the streak of
/// faults: the next fault is the first of a streak again.
#[test]
fn a_stage_that_panics_at_a_decode_calls_issue_is_held_and_the_side_stands_issued() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("faulty", || Box::new(Faulty))
        .expect("a free name");
    store
        .apply(pair_plan(&catalog, "[]", ["random", "faulty"]))
        .expect("above the unbound start");
    side_member(&store, &objects, "p0", Role::Prefill, None);
    let d = side_member(&store, &objects, "d0", Role::Decode, None);
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    let (_, prefill) = held_pair_at(&store, &objects, (1, 0), now);
    assert_eq!(store.lifecycle_faults(), 0);
    // One call of the lease, whose frame states `tokens`.
    let called = |n: u8, tokens: u32| {
        let mut frame = at_call(n, prefill.id, WireModes::NONE, None);
        let facts = frame.facts.as_mut().expect("the case's frame has facts");
        facts.prompt_tokens = tokens;
        let (_, issued) = leased_call(&store, &objects, frame, now);
        issued.expect("the pick the file holds")
    };
    let (side, issue) = called(2, FAULTY_TOKENS);
    assert_eq!((side.side, side.booked), (Side::Decode, d));
    assert_eq!(store.lifecycle_faults(), 1);
    let sides = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
    let on_file = sides.decode.map(|side| (side.state, side.target));
    assert_eq!(on_file, Some((LeaseState::ISSUED, d)));
    // The next call's issue is told and returns: the streak ends.
    assert!(store.unissue_decode(GATEWAY, prefill.id, issue));
    called(3, 1);
    assert_eq!(store.lifecycle_faults(), 1);
    assert!(store.call_faulted(), "the first fault of a new streak");
}

/// A `Dispatched` frame at an index the pick holds no target at names no
/// worker. On a decode side that stands chosen it moves nothing, and no
/// stage hears of it: no issue, no dispatch and, where the gateway then
/// reports the side's end or gives the lease up, no end of that side. A
/// decode call behind the frame is decided as behind no frame, and its
/// pick is issued under the pair's lease. A frame behind it that names a
/// target is told as the side's issue and then its dispatch, once each.
#[test]
fn a_dispatched_frame_at_an_index_its_pick_does_not_hold_leaves_a_chosen_side_as_it_stood() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    let told = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&told);
    let mut catalog = StageRegistry::builtin();
    let stage = move || -> Box<dyn Picker> { Box::new(SideWords(Arc::clone(&kept))) };
    catalog.picker("side-words", stage).expect("a free name");
    store
        .apply(pair_plan(
            &catalog,
            r#"["healthy"]"#,
            ["random", "side-words"],
        ))
        .expect("above the unbound start");
    let p = side_member(&store, &objects, "p0", Role::Prefill, None);
    let d = side_member(&store, &objects, "d0", Role::Decode, None);
    LeasesStore::set_term(&*store, 1);
    // The word to the engine of a frame applied to a lease, as a replica
    // gives it at the start of its deciders, and a session's hand at the
    // files: no decider is at work here, so the hand applies each frame.
    lifecycle::tells(&store, &objects);
    let hand = Hand::new(GATEWAY);
    let now = Instant::now();
    // A `Dispatched` frame of the decode side of `lease`, at `index`.
    let sent = |lease: LeaseId, index: u32| {
        let frame = proto::Dispatched {
            lease_id: lease.0,
            target_index: index,
            side: proto::Side::Decode.into(),
            since_decision_micros: 0,
        };
        lifecycle::dispatched(&store, &hand, &frame, now).expect("the frame reads");
    };
    // What the decode side's stage heard since the last read.
    let heard = || mem::take(&mut *told.lock());
    // What the leases on file book on the prefill and on the decode.
    let booked = || {
        let load = |name: &str| {
            let url = format!("http://{name}:8000");
            let object = objects.registry().get_by_url(&url).expect("its object");
            object.load()
        };
        [load("p0"), load("d0")]
    };
    let side_on_file = |lease: LeaseId| {
        let sides = LeasesStore::sides(&*store, GATEWAY, lease).expect("on file");
        sides.decode.map(|side| (side.state, side.target))
    };
    let chosen = Some((LeaseState::Chosen, d));

    // The pick holds one target. A frame past it, at the next index and at
    // the last a byte names, moves nothing and reaches no stage.
    let (_, first) = held_pair_at(&store, &objects, (1, 0), now);
    assert_eq!(heard(), [("issued", Side::Prefill, p)]);
    for index in [1, 255] {
        sent(first.id, index);
    }
    assert_eq!((side_on_file(first.id), booked()), (chosen, [1, 0]));
    assert!(
        heard().is_empty(),
        "no stage hears a frame that names no worker"
    );
    // A decode call behind it finds the side chosen: its pick is issued
    // under the pair's lease, and the stages hear that issue.
    let call = at_call(2, first.id, WireModes::NONE, None);
    let (decision, issued) = leased_call(&store, &objects, call, now);
    assert_eq!(pairs_of(&decision), [(p, d)]);
    let (side, _) = issued.expect("a pick's issue");
    assert_eq!(
        (side.id, side.side, side.booked),
        (first.id, Side::Decode, d)
    );
    assert_eq!(heard(), [("issued", Side::Decode, d)]);
    let issued = Some((LeaseState::ISSUED, d));
    assert_eq!((side_on_file(first.id), booked()), (issued, [1, 1]));

    // A second pair, whose decode is reported complete behind such a
    // frame and whose lease is then given up whole: the prefill's release
    // is told, and no end of a side no stage heard issued.
    let (_, second) = held_pair_at(&store, &objects, (3, 0), now);
    told.lock().clear();
    sent(second.id, 1);
    let ended = proto::Completed {
        lease_id: second.id.0,
        side: proto::Side::Decode.into(),
        served: d.0,
        status: 200,
        class: proto::completed::Class::Ok.into(),
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    };
    lifecycle::completed(&store, &hand, &ended).expect("the frame reads");
    assert!(heard().is_empty(), "a side no stage heard issued");
    let complete = Some((LeaseState::Completed, d));
    assert_eq!((side_on_file(second.id), booked()), (complete, [2, 1]));
    let gone = proto::Released {
        what: Some(proto::released::What::LeaseId(second.id.0)),
        reason: proto::released::Reason::ClientGone.into(),
        side: proto::Side::Unspecified.into(),
    };
    lifecycle::released(&store, &hand, &gone).expect("the frame reads");
    assert_eq!(heard(), [("released", Side::Prefill, p)]);
    assert_eq!(booked(), [1, 1]);

    // A third: a frame behind it that names the pick's target is the
    // side's issue and then its dispatch, and the same frame again tells
    // nothing.
    let (_, third) = held_pair_at(&store, &objects, (4, 0), now);
    told.lock().clear();
    for index in [1, 0] {
        sent(third.id, index);
    }
    let words = [("issued", Side::Decode, d), ("dispatched", Side::Decode, d)];
    assert_eq!(heard(), words);
    sent(third.id, 0);
    assert!(heard().is_empty(), "the same frame again");
    let sent_to = Some((LeaseState::Dispatched { at: now }, d));
    assert_eq!((side_on_file(third.id), booked()), (sent_to, [2, 2]));
}
