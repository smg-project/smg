//! The cases of a pair's frame at its prefill. A pair's ask [`decide_pair`]
//! for a pair at its prefill, and the two that file its leases ask
//! [`issue_pair`] behind it. One case holds the document's pairing at a
//! pair's first decision and at its decode call.

use super::*;

/// A pair at its prefill is decided over its model's two lists, and each
/// side's list is the asking gateway's to take by that side's own places
/// and vetoes. On each side another gateway alone dials one worker and the
/// asking gateway vetoes another, at places that differ between the sides.
/// Every frame is answered a pick, and every target of it names the one
/// prefill left with the one decode left: a place of one list says nothing
/// of the other. The other gateway dials all six and vetoes none: its
/// picks name every pair.
#[test]
fn a_pair_at_its_prefill_takes_each_side_by_that_sides_own_places_and_vetoes() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(stock_pair_plan())
        .expect("above the unbound start");
    let member = |name: &str, role, only| side_member(&store, &objects, name, role, only);
    let (prefill, decode) = (Role::Prefill, Role::Decode);
    let prefills = [
        member("p0", prefill, None),
        member("p1", prefill, Some(OTHER)),
        member("p2", prefill, None),
    ];
    let decodes = [
        member("d0", decode, None),
        member("d1", decode, None),
        member("d2", decode, Some(OTHER)),
    ];
    let now = Instant::now();
    let until = Some(now + Duration::from_secs(60));
    store.gateway_veto(GATEWAY, prefills[2], until);
    store.gateway_veto(GATEWAY, decodes[0], until);
    let named = |gateway: GatewayId| -> BTreeSet<(WorkerId, WorkerId)> {
        let mut named = BTreeSet::new();
        for n in 1..=64 {
            let frame = at_prefill(n, WireModes::NONE, None);
            let answer = served_pair(&store, &objects, frame, gateway, now);
            let Decision::Pick(pick) = &answer else {
                panic!("frame {n} is answered no pick: {answer:?}");
            };
            let pairs = pairs_of(&answer);
            assert_eq!(pairs.len(), pick.targets.len(), "a decode a target");
            named.extend(pairs);
        }
        named
    };
    assert_eq!(named(GATEWAY), BTreeSet::from([(prefills[0], decodes[1])]));
    let every = prefills
        .iter()
        .flat_map(|&p| decodes.iter().map(move |&d| (p, d)));
    let every: BTreeSet<(WorkerId, WorkerId)> = every.collect();
    assert_eq!(named(OTHER), every);
}

/// A pair's request is given the load monitor's snapshot as a single
/// request's is, and both sides read it: under a plan whose sides go by
/// the least load, every pick names first the prefill and the decode whose
/// reports show no queue. Under a snapshot that holds no report no worker
/// stands out, and the picks name more than one pair first.
#[test]
fn a_pairs_sides_read_the_reports_of_the_snapshot_their_request_is_given() {
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
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(staged(&document, &StageRegistry::builtin()))
        .expect("above the unbound start");
    let prefills =
        ["p0", "p1"].map(|name| side_member(&store, &objects, name, Role::Prefill, None));
    let decodes = ["d0", "d1"].map(|name| side_member(&store, &objects, name, Role::Decode, None));
    let now = Instant::now();
    // The second prefill and the first decode report no queue, and the
    // other two are deep in theirs.
    let reports = ["p0", "p1", "d0", "d1"].map(|name| {
        let load = if name == "p1" || name == "d0" {
            one_rank_load(0, 0.1, 10.0)
        } else {
            one_rank_load(900_000, 0.9, 10.0)
        };
        (format!("http://{name}:8000"), load)
    });
    let published = LoadSnapshot::from_loads_for_test(reports.into_iter().collect());
    let unread = Arc::new(LoadSnapshot::default());
    let firsts = |loads: &Arc<LoadSnapshot>| -> BTreeSet<(WorkerId, WorkerId)> {
        let frames = (1..=32).map(|n| at_prefill(n, WireModes::NONE, None));
        let asked = (GATEWAY, now);
        let picks = frames.map(|frame| served_pair_reading(&store, &objects, frame, asked, loads));
        let firsts = picks.map(|pick| *pairs_of(&pick).first().expect("a pick"));
        firsts.collect()
    };
    let idle = (prefills[1], decodes[0]);
    assert_eq!(firsts(&published), BTreeSet::from([idle]));
    assert!(firsts(&unread).len() > 1, "no report, and one pair alone");
}

/// A hold of the asking gateway on a side's only worker sheds the pair's
/// frame under the health filter, on either side: no rule lifts a hold,
/// where a doubt of a side's every worker is lifted as the last resort.
/// Another gateway holds no veto, and the pair is its pick.
#[test]
fn a_hold_on_the_only_worker_of_either_side_sheds_a_pairs_frame_under_the_health_filter() {
    for held in [Role::Prefill, Role::Decode] {
        let store = Arc::new(MemoryStore::new());
        let objects = holder(&store);
        store
            .apply(stock_pair_plan())
            .expect("above the unbound start");
        let p = side_member(&store, &objects, "p0", Role::Prefill, None);
        let d = side_member(&store, &objects, "d0", Role::Decode, None);
        let now = Instant::now();
        let answered = |gateway| {
            let frame = at_prefill(1, WireModes::NONE, None);
            served_pair(&store, &objects, frame, gateway, now)
        };
        assert_eq!(pairs_of(&answered(GATEWAY)), [(p, d)]);
        let on = if held == Role::Prefill { p } else { d };
        store.gateway_veto(GATEWAY, on, Some(now + Duration::from_secs(60)));
        let shed = Some((ShedReason::NoEndpoints, "healthy"));
        assert_eq!(shed_of(&answered(GATEWAY)), shed, "a hold on the {held:?}");
        assert_eq!(pairs_of(&answered(OTHER)), [(p, d)]);
    }
}

/// The answers of a pair's frame come in one order. A side with no worker
/// at all is decided over no worker, whatever the gateway's wires take of
/// the other side: one side alone is no pair. Else a side the gateway dials
/// and can take nothing of sheds the frame under that stage, though the
/// gateway dials no worker of the other side. Else a side it dials no
/// worker of is decided over no worker. A model nobody serves is too.
#[test]
fn a_pairs_frame_answers_a_side_with_no_worker_then_one_emptied_then_one_not_dialed() {
    let now = Instant::now();
    // The answer of GATEWAY's frame over a fleet of `decodes`, beside one
    // prefill every gateway dials where `prefill` says so, with the
    // engine's own answer over no worker for the same frame.
    let answered = |prefill: bool, decodes: &[Option<GatewayId>], wires, hint: Option<u32>| {
        let store = Arc::new(MemoryStore::new());
        let objects = holder(&store);
        store
            .apply(stock_pair_plan())
            .expect("above the unbound start");
        if prefill {
            side_member(&store, &objects, "p0", Role::Prefill, None);
        }
        for (n, &only) in decodes.iter().enumerate() {
            side_member(&store, &objects, &format!("d{n}"), Role::Decode, only);
        }
        let frame = at_prefill(1, wires, hint);
        let request = Checked::new(frame.clone()).expect("a frame the record holds");
        let over_none = store.deciding(|world| {
            let mut lists = Lists::default();
            let read = lists.of(&request);
            let maker = FrameRequest::new(&request, &read, BUDGET, WorkerLists::None);
            let decided = world.front().decide_without_lease(&maker, now);
            shed_of(&decided.expect("the engine decides").0)
        });
        assert!(over_none.is_some(), "no worker is no pick");
        let answer = served_pair(&store, &objects, frame, GATEWAY, now);
        (answer, over_none)
    };
    let (http, grpc) = (WireModes::HTTP, WireModes::GRPC);
    // A side with no worker at all: over no worker, though the gateway's
    // wires take every worker of the other side and a hint of 0 takes them
    // too.
    let lone: [(bool, &[Option<GatewayId>]); 2] = [(true, &[]), (false, &[None])];
    for (prefill, decodes) in lone {
        for (wires, hint) in [(WireModes::NONE, None), (grpc, None), (http, Some(0))] {
            let (answer, over_none) = answered(prefill, decodes, wires, hint);
            assert_eq!(shed_of(&answer), over_none, "{wires:?}, a hint of {hint:?}");
        }
    }
    // A decode another gateway alone dials, beside a prefill the gateway's
    // wires or its hint take: the emptied side sheds.
    let theirs = [Some(OTHER)];
    let wires = Some((ShedReason::NoEndpoints, WIRES));
    assert_eq!(shed_of(&answered(true, &theirs, grpc, None).0), wires);
    let hinted = Some((ShedReason::NoEndpoints, SUBSET_HINT));
    assert_eq!(shed_of(&answered(true, &theirs, http, Some(0)).0), hinted);
    // The same decode beside a prefill the gateway takes: over no worker.
    let (answer, over_none) = answered(true, &theirs, http, None);
    assert_eq!(shed_of(&answer), over_none);
    // And with a decode it dials, the pair is picked.
    let (answer, _) = answered(true, &[None], http, None);
    assert!(matches!(answer, Decision::Pick(_)), "{answer:?}");
    // A model nobody serves: over no worker.
    let (answer, over_none) = answered(false, &[], WireModes::NONE, None);
    assert_eq!(shed_of(&answer), over_none);
}

/// A pair's pick is filed a lease a side under one id: the prefill side's
/// on the pick's prefill, the decode side's on its decode, each under the
/// session the frame's own reader states of a request of that side's phase,
/// and each side's worker is booked for the decisions behind it. A frame
/// that states the later decode files its decode side chosen: the answer
/// carries the prefill's lease alone, and the decode's worker is booked no
/// more.
#[test]
fn a_pairs_pick_is_filed_a_lease_a_side_and_a_later_decode_stands_chosen() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    let read = Arc::new(Mutex::new(Vec::new()));
    let mut catalog = StageRegistry::builtin();
    let (kept, own) = (Arc::clone(&read), Arc::downgrade(&store));
    let stage = move || -> Box<dyn Filter> {
        Box::new(Late {
            store: Weak::clone(&own),
            late: Arc::default(),
            read: Arc::clone(&kept),
        })
    };
    catalog.filter("late", stage).expect("a free name");
    let late = r#"["healthy", "late"]"#;
    store
        .apply(pair_plan(&catalog, late, ["random", "random"]))
        .expect("above the unbound start");
    let p = side_member(&store, &objects, "p0", Role::Prefill, None);
    let d = side_member(&store, &objects, "d0", Role::Decode, None);
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    // The `n`th frame of one conversation, at `phase`: its key goes plain,
    // so each phase's reader makes the session of its own phase.
    let framed = |n: u8, phase: proto::Phase| {
        let mut frame = at_prefill(n, WireModes::NONE, None);
        let facts = frame.facts.as_mut().expect("the case's frame has facts");
        (facts.phase, facts.prompt_tokens) = (phase.into(), 300);
        facts.session_key = Some(proto::Identity {
            form: Some(proto::identity::Form::Plain("conversation-7".to_owned())),
        });
        frame
    };
    let leased = |n: u8, later: bool| {
        let mut frame = framed(n, proto::Phase::Prefill);
        frame.decode_later = later;
        let (_, prefill, decode) = leased_pair(&store, &objects, frame, now);
        (prefill, decode)
    };
    // A side's session, as the frame's own reader states it of a request
    // of that phase.
    let session_at = |phase: proto::Phase| {
        let request = Checked::new(framed(9, phase)).expect("a frame the record holds");
        let mut lists = Lists::default();
        let read = lists.of(&request);
        request.facts(&read, 0, 0).session()
    };
    // What the two sides of the newest decision read of their workers.
    let counts = || -> Counts {
        let read = read.lock();
        let sides = read.iter().rev().take(2);
        let mut counts: Counts = sides.flat_map(|side| side[0].clone()).collect();
        counts.sort_unstable();
        counts
    };
    let side_on_file = |lease: &Lease| {
        let sides = LeasesStore::sides(&*store, GATEWAY, lease.id).expect("on file");
        (sides.record.decode, sides.decode.map(|side| side.state))
    };

    let (prefill, decode) = leased(1, false);
    let decode = decode.expect("issued with its prefill");
    assert_eq!(
        (prefill.side, prefill.booked, prefill.tokens),
        (Side::Prefill, p, 300)
    );
    let sessions = [proto::Phase::Prefill, proto::Phase::Decode].map(session_at);
    assert_ne!(sessions[0], sessions[1], "a session a side");
    assert_eq!(prefill.session, sessions[0]);
    let beside = Lease {
        side: Side::Decode,
        booked: d,
        session: sessions[1],
        ..prefill
    };
    assert_eq!(decode, beside, "one id, and the prompt's tokens on both");
    let issued = (Some(DecodeMark), Some(LeaseState::ISSUED));
    assert_eq!(side_on_file(&prefill), issued);

    // The next pair reads both workers booked, and states the later decode.
    let (prefill, decode) = leased(2, true);
    assert_eq!(counts(), by_id(&[p, d], &[1, 1]));
    assert_eq!(decode, None, "the answer carries the prefill's lease alone");
    let chosen = (Some(DecodeMark), Some(LeaseState::Chosen));
    assert_eq!(side_on_file(&prefill), chosen);
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 2);

    // A third reads the prefill booked twice and the decode once.
    leased(3, false);
    assert_eq!(counts(), by_id(&[p, d], &[2, 1]));
}

/// A pair's lease keeps each target of its pick for a dispatch that names
/// one by its index: the prefill side's record the target's prefill, and
/// the decode side's its decode.
#[test]
fn a_pairs_lease_keeps_each_targets_prefill_and_decode_for_a_dispatch_by_index() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(stock_pair_plan())
        .expect("above the unbound start");
    side_member(&store, &objects, "p0", Role::Prefill, None);
    for name in ["d0", "d1", "d2"] {
        side_member(&store, &objects, name, Role::Decode, None);
    }
    LeasesStore::set_term(&*store, 1);
    let frame = at_prefill(1, WireModes::NONE, None);
    let (decision, prefill, _) = leased_pair(&store, &objects, frame, Instant::now());
    let targets = &decision.pick().expect("a pick").targets;
    assert!(targets.len() > 1, "a target behind the first: {decision:?}");
    let on_file = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
    for (index, target) in (0u8..).zip(targets.iter()) {
        let pair = target.pair.expect("a decode a target");
        let sides = [Side::Prefill, Side::Decode].map(|side| on_file.target_of(side, index));
        let named = [
            (target.id, target.dp_rank),
            (pair.decode_id, pair.decode_dp_rank),
        ];
        assert_eq!(sides, named.map(Some), "the target at {index}");
    }
    let past = u8::try_from(targets.len()).expect("a few targets");
    assert_eq!(on_file.target_of(Side::Decode, past), None);
}

/// A pair either of whose sides holds a stage that may refuse by a report
/// of leases is decided behind an apply of what waits at the other files:
/// a frame left under another file's watch is applied by the decision,
/// where its decider lent its place. A pair whose stages refuse by none
/// takes no turn there, and the frame waits for its own file's look.
#[test]
fn a_pair_either_of_whose_sides_may_refuse_by_reports_applies_what_waits_at_the_other_files() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("by-reports", || Box::new(ByReports))
        .expect("a free name");
    let cases = [
        (["by-reports", "random"], LookOutcome::Quiet),
        (["random", "by-reports"], LookOutcome::Quiet),
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
        let now = Instant::now();
        let frame = at_prefill(1, WireModes::NONE, None);
        let request = Checked::new(frame).expect("a frame the record holds");
        let taken = taken_at(&request, GATEWAY, (now, now), Some(&place));
        let (mut frames, own) = (FrameBuffers::default(), Part::default());
        let loads = Arc::new(LoadSnapshot::default());
        let decided =
            store.deciding(|world| decide_pair(world, &objects, &taken, &mut frames, &loads, &own));
        let (decision, _) = decided.expect("the engine decides");
        assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
        assert_eq!(watch.look(), waited, "{picks:?}");
    }
}

/// A ready member `name` of [`PAIRED`] on `role`'s side, as [`side_member`]
/// lays one, that offers a partner the KV transport `transport`.
fn transport_member(
    store: &MemoryStore,
    objects: &Objects<Members>,
    name: &str,
    role: Role,
    transport: &str,
) -> WorkerId {
    let key = format!("http://{name}:8000");
    let descriptor = PairingDescriptor {
        transport: Some(transport.to_owned()),
        ..PairingDescriptor::default()
    };
    let facts = WorkerFacts {
        key: key.clone(),
        engine: key,
        incarnation: 1,
        role,
        wire_modes: WireModes::HTTP,
        models: vec![PAIRED.to_owned()],
        dp_size: 1,
        max_running: Some(4),
        pairing: Some(Arc::new(descriptor)),
        ..WorkerFacts::default()
    };
    written_ready(store, objects, facts)
}

/// A pair's two sides pair by the classes of the document's own pairing, at
/// the pair's first decision and at its decode call. Under the pairing of
/// the stock plan a decode on another KV transport than the prefill's is no
/// partner: no pick and no call names it. Under a plan that pairs on
/// nothing every decode is a partner.
#[test]
fn a_pair_and_its_decode_call_pair_by_the_classes_of_the_documents_pairing() {
    let plan = |pairing: Option<&str>| {
        let plans = pairing.map_or_else(String::new, |mode| {
            format!(r#""plans": {{ "pd-pair": {{ "pairing": "{mode}" }} }},"#)
        });
        let document = format!(
            r#"{{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              {plans}
              "profiles": {{ "side": {{ "pick": "random" }} }},
              "bindings": [
                {{ "match": {{ "phase": ["prefill", "decode"] }}, "plan": "pd-pair",
                  "profiles": {{ "prefill": "side", "decode": "side" }} }}
              ]
            }}"#
        );
        let document: PolicySpec =
            serde_json::from_str(&document).expect("a document of this build");
        staged(&document, &StageRegistry::builtin())
    };
    let now = Instant::now();
    let unread = Arc::new(LoadSnapshot::default());
    // The decodes the pair's first decisions name, and those its decode
    // calls name behind the one prefill.
    let named = |pairing: Option<&str>| {
        let store = Arc::new(MemoryStore::new());
        let objects = holder(&store);
        store.apply(plan(pairing)).expect("above the unbound start");
        let p = transport_member(&store, &objects, "p0", Role::Prefill, "a");
        let same = transport_member(&store, &objects, "d0", Role::Decode, "a");
        let other = transport_member(&store, &objects, "d1", Role::Decode, "b");
        let mut firsts = BTreeSet::new();
        let mut calls = BTreeSet::new();
        for n in 1..=32 {
            let frame = at_prefill(n, WireModes::NONE, None);
            let first = served_pair(&store, &objects, frame, GATEWAY, now);
            firsts.extend(pairs_of(&first).into_iter().map(|pair| pair.1));
            let frame = at_call(n, LeaseId(1), WireModes::NONE, None);
            let (asked, given) = ((GATEWAY, now), (&unread, None));
            let call = served_call(&store, &objects, frame, asked, (p, None), given);
            calls.extend(pairs_of(&call).into_iter().map(|pair| pair.1));
        }
        ([firsts, calls], same, other)
    };
    let (sides, same, _) = named(None);
    assert_eq!(sides, [BTreeSet::from([same]), BTreeSet::from([same])]);
    let (sides, same, other) = named(Some("off"));
    let both = BTreeSet::from([same, other]);
    assert_eq!(sides, [both.clone(), both]);
}

/// A ready member `name` of [`PAIRED`] on `role`'s side, on the gRPC wire,
/// written through `objects`, whose registration names the engine kind
/// `kind`: none where it is empty.
fn kind_member(
    store: &MemoryStore,
    objects: &Objects<Members>,
    name: &str,
    role: Role,
    kind: &str,
) -> WorkerId {
    let key = format!("grpc://{name}:8000");
    let facts = WorkerFacts {
        key: key.clone(),
        engine: key,
        incarnation: 1,
        role,
        wire_modes: WireModes::GRPC,
        engine_kind: kind.to_owned(),
        models: vec![PAIRED.to_owned()],
        dp_size: 1,
        max_running: Some(4),
        ..WorkerFacts::default()
    };
    written_ready(store, objects, facts)
}

/// A controller's pair classes hold a model's workers of both wires in one
/// table, and state which pairs go first: every pair but one that, on the
/// wire that carries one runtime's request, has a worker of unknown
/// runtime. One prefill and one decode stand on the plain wire here, and
/// on the other wire a prefill and a decode of one known runtime beside a
/// prefill and a decode of none. Every worker of a wire pairs with every
/// other of it under the stock plan's pairing.
///
/// A frame held to one wire reads the gateway's own order on it. On the
/// runtime-specific wire every target names the pair on the known runtime
/// while it is open, and a pair with a worker of unknown runtime once the
/// asking gateway holds the known decode back
/// (`model_gateway/src/routers/common/placement.rs:2754-2771`,
/// `:2884-2887`, on main `4b41f39a7`). On the plain wire the one pair
/// there is made: that wire has no such order.
///
/// A frame of a gateway that speaks both wires, or states none, is decided
/// over both. No replay on main shows this: no request of a gateway's own
/// pair placement is placed over two wires. The plain pair and the pair on
/// the known runtime are both made, and a pair with a worker of unknown
/// runtime is made only while neither is open.
///
/// A decode call behind the prefill of the known runtime names the decode
/// of its runtime alone, and one behind the prefill of unknown runtime
/// names either decode of its wire.
#[test]
fn a_served_pair_with_a_worker_of_unknown_runtime_goes_behind_every_other_pair() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    let applied = store.apply(stock_pair_plan());
    applied.expect("above the unbound start");
    let nameless = kind_member(&store, &objects, "p0", Role::Prefill, "");
    let known = kind_member(&store, &objects, "p1", Role::Prefill, "runtime-a");
    let any = kind_member(&store, &objects, "d0", Role::Decode, "");
    let same = kind_member(&store, &objects, "d1", Role::Decode, "runtime-a");
    let plain = (
        side_member(&store, &objects, "p2", Role::Prefill, None),
        side_member(&store, &objects, "d2", Role::Decode, None),
    );
    let now = Instant::now();
    // One gateway holds the known decode back, and another the plain
    // decode too.
    let until = Some(now + Duration::from_secs(60));
    let (one_held, both_held) = (OTHER, GatewayId(13));
    store.gateway_veto(one_held, same, until);
    store.gateway_veto(both_held, same, until);
    store.gateway_veto(both_held, plain.1, until);
    let made = |gateway: GatewayId, wires: WireModes| -> BTreeSet<(WorkerId, WorkerId)> {
        let mut made = BTreeSet::new();
        for n in 1..=64 {
            let frame = at_prefill(n, wires, None);
            let answer = served_pair(&store, &objects, frame, gateway, now);
            assert!(answer.pick().is_some(), "frame {n}: {answer:?}");
            made.extend(pairs_of(&answer));
        }
        made
    };
    let first = (known, same);
    let behind = BTreeSet::from([(known, any), (nameless, any)]);
    assert_eq!(made(GATEWAY, WireModes::GRPC), BTreeSet::from([first]));
    assert_eq!(made(one_held, WireModes::GRPC), behind);
    assert_eq!(made(GATEWAY, WireModes::HTTP), BTreeSet::from([plain]));
    for wires in [WireModes::HTTP.union(WireModes::GRPC), WireModes::NONE] {
        let both = BTreeSet::from([first, plain]);
        assert_eq!(made(GATEWAY, wires), both, "{wires:?}");
        assert_eq!(made(one_held, wires), BTreeSet::from([plain]), "{wires:?}");
        assert_eq!(made(both_held, wires), behind, "{wires:?}");
    }

    let unread = Arc::new(LoadSnapshot::default());
    let mut calls = [BTreeSet::new(), BTreeSet::new()];
    for n in 1..=32 {
        for (prefill, named) in [known, nameless].into_iter().zip(&mut calls) {
            let frame = at_call(n, LeaseId(1), WireModes::GRPC, None);
            let (asked, given) = ((GATEWAY, now), (&unread, None));
            let call = served_call(&store, &objects, frame, asked, (prefill, None), given);
            named.extend(pairs_of(&call).into_iter().map(|pair| pair.1));
        }
    }
    let either = BTreeSet::from([any, same]);
    assert_eq!(calls, [BTreeSet::from([same]), either]);
}
