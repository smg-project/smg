//! The cases of the gateway a decision is made for: whose request it is,
//! the vetoes it holds, and the wires and the hint it states.

use super::*;

#[tokio::test]
async fn a_decision_is_made_for_the_gateway_its_request_came_from() {
    let mut bed = leading([]);
    let vetoed = bed.workers[0];
    let until = bed.script.taken + Duration::from_secs(60);
    bed.store.gateway_veto(GATEWAY, vetoed, Some(until));
    let mut theirs = Vec::new();
    for (id, seed) in (1..=48).zip(1..) {
        let own = bed.ask(frame(id, seed)).await;
        assert_ne!(picked(&own), vetoed, "seed {seed}");
        assert_eq!(bed.submit_at(frame(id, seed), OTHER, Duration::ZERO), None);
        theirs.push(picked(&bed.answer().await));
    }
    // The veto is the asking gateway's alone.
    assert!(theirs.contains(&vetoed), "{theirs:?}");
}

/// The kinds of a gateway's vetoes reach its decisions. A pool it doubts
/// for liveness on every worker is steered: a doubt is not an outage. A
/// breaker's open window on each worker refuses the pool, as the gateway
/// refuses it. One doubted worker among held ones takes every request,
/// a kind this build has no name for holds as a breaker does, and
/// another gateway reads none of it.
#[tokio::test]
async fn a_pool_behind_the_gateways_open_breakers_is_refused_and_one_it_doubts_is_steered() {
    const BREAKER: i32 = proto::gateway_veto::Kind::Breaker as i32;
    const LIVENESS: i32 = proto::gateway_veto::Kind::Liveness as i32;
    let mut bed = leading([]);
    let now = bed.script.taken;
    let until = now + Duration::from_secs(60);
    let workers = bed.workers.clone();
    let fence = bed.elector.fence();
    let gateways = Arc::new(Gateways::new(GRACE, 1_024, fence, Arc::clone(&bed.store)));
    let seat = gateways
        .welcome(GATEWAY, "gateway-a", 1, now)
        .expect("the table takes the Hello");
    let veto = |worker: WorkerId, kind: i32, until: Option<Instant>| {
        assert_eq!(seat.veto(worker, kind, until, now), Vetoed::Shown);
    };
    let shed_at = |answer: &proto::DecideAnswer| match outcome(answer) {
        Outcome::Shed(shed) => (shed.reason(), shed.emptied_by.clone()),
        other => panic!("a shed: {other:?}"),
    };
    let at_health = (proto::ShedReason::NoEndpoints, "healthy".to_owned());
    for &worker in &workers {
        veto(worker, LIVENESS, Some(until));
    }
    let steered = bed.ask(frame(1, 1)).await;
    assert!(workers.contains(&picked(&steered)));
    for &worker in &workers {
        veto(worker, BREAKER, Some(until));
    }
    assert_eq!(shed_at(&bed.ask(frame(2, 2)).await), at_health);
    // One breaker closes: its worker is doubted alone.
    let doubted = workers[1];
    veto(doubted, BREAKER, None);
    for (id, seed) in (3..=10).zip(3..) {
        let answer = bed.ask(frame(id, seed)).await;
        assert_eq!(picked(&answer), doubted, "seed {seed}");
    }
    veto(doubted, 7, Some(until));
    assert_eq!(shed_at(&bed.ask(frame(11, 11)).await), at_health);
    assert_eq!(bed.submit_at(frame(12, 12), OTHER, Duration::ZERO), None);
    assert!(workers.contains(&picked(&bed.answer().await)));
}

/// Every worker a pick names, its alternates with it.
fn targets(answer: &proto::DecideAnswer) -> Vec<WorkerId> {
    match outcome(answer) {
        Outcome::Pick(pick) => pick.targets.iter().map(|t| WorkerId(t.id)).collect(),
        other => panic!("a pick: {other:?}"),
    }
}

/// How many times each worker of `picked` stands in it, in the
/// workers' order.
fn turns(mut picked: Vec<WorkerId>) -> Vec<usize> {
    picked.sort_unstable();
    picked
        .chunk_by(|one, next| one == next)
        .map(<[WorkerId]>::len)
        .collect()
}

/// One ready member that speaks HTTP alone, asked for under each thing
/// a gateway states of itself. A gateway that states no wire, or one
/// the member speaks, is answered the member. One whose wires the
/// member shares none of is shed under the wires' stage, and so is one
/// that states only a wire this build does not know. A hint of 0 is a
/// decision over no candidates, shed under the hint's stage whatever
/// the wires, and a positive hint cuts nothing and names no stage: the
/// wires that share none are the stage under one. Then a member the
/// gateway cannot dial and a second one it can, beside the first, which
/// it just failed to reach: the second is its whole answer.
#[tokio::test]
async fn a_gateway_is_answered_no_member_it_cannot_dial_and_none_under_a_hint_of_0() {
    const SPOKEN: &str = "model-http";
    let mut bed = leading([]);
    let store = Arc::clone(&bed.store);
    let first = ready(
        &store,
        &bed.reads.objects,
        "h0",
        SPOKEN,
        Role::Single,
        WireModes::HTTP,
    );
    let answer = bed.ask(frame_for(SPOKEN, 100, 100)).await;
    assert_eq!(targets(&answer), [first], "no constraints");
    // The wires a gateway states, its hint, and the stage its decision
    // is shed under where it is shed.
    let cells = [
        (0b001, None, None),
        (0b011, None, None),
        (0b010, None, Some("wires")),
        (0b110, None, Some("wires")),
        (0b1000, None, Some("wires")),
        (0b010, Some(1), Some("wires")),
        (0b001, Some(0), Some("subset-hint")),
        (0b010, Some(0), Some("subset-hint")),
        (0, Some(0), Some("subset-hint")),
        (0b001, Some(1), None),
        (0, Some(7), None),
    ];
    for (id, (wires, hint, stage)) in (1..).zip(cells) {
        let frame = constrained(frame_for(SPOKEN, id, u64::from(id)), wires, hint);
        let answer = bed.ask(frame).await;
        let cell = format!("wires {wires:#b}, a hint of {hint:?}");
        match stage {
            Some(stage) => assert_eq!(
                outcome(&answer),
                &declined(proto::ShedReason::NoEndpoints, stage),
                "{cell}"
            ),
            None => assert_eq!(targets(&answer), [first], "{cell}"),
        }
    }
    // The wires took the member in four decisions: under a hint of 0
    // they had none to take.
    assert_eq!(bed.counts.undialed(), 4);
    ready(
        &store,
        &bed.reads.objects,
        "g0",
        SPOKEN,
        Role::Single,
        WireModes::GRPC,
    );
    let second = ready(
        &store,
        &bed.reads.objects,
        "h1",
        SPOKEN,
        Role::Single,
        WireModes::HTTP,
    );
    for seed in 0..32_u8 {
        let frame = frame_for(SPOKEN, 40 + seed, u64::from(seed));
        let mut frame = constrained(frame, 0b001, None);
        frame.constraints.get_or_insert_default().excluded = vec![first.0];
        let answer = bed.ask(frame).await;
        assert_eq!(targets(&answer), [second], "seed {seed}");
    }
    assert_eq!(bed.counts.undialed(), 4 + 32);
}

/// A gateway that states its wires is answered members that share one
/// with it, its alternates too, and each one its wires took is counted.
/// A member that states no wire is no candidate of a gateway that
/// states some: silence is no waiver. A gateway that states none is not
/// narrowed, and a positive hint, which no stage reads yet, cuts
/// nothing.
#[tokio::test]
async fn a_gateways_wires_narrow_its_decision_to_the_members_that_share_one() {
    let mut bed = leading([]);
    // The fleet's three members state no wire.
    let answer = bed.ask(constrained(frame(1, 1), 0b111, None)).await;
    let shed = declined(proto::ShedReason::NoEndpoints, "wires");
    assert_eq!(outcome(&answer), &shed);
    assert_eq!(bed.counts.undialed(), 3);
    let store = Arc::clone(&bed.store);
    let both = WireModes::HTTP.union(WireModes::GRPC);
    let [_, grpc, either] = [
        ("w0", WireModes::HTTP),
        ("w1", WireModes::GRPC),
        ("w2", both),
    ]
    .map(|(name, wires)| ready(&store, &bed.reads.objects, name, MODEL, Role::Single, wires));
    let mut dialed = [grpc, either];
    dialed.sort_unstable();
    let mut named = BTreeSet::new();
    for (id, seed) in (2..=33).zip(100..) {
        let answer = bed.ask(constrained(frame(id, seed), 0b010, None)).await;
        named.extend(targets(&answer));
    }
    assert_eq!(named.into_iter().collect::<Vec<_>>(), dialed);
    // The member that speaks HTTP alone, in each of the 32 decisions.
    assert_eq!(bed.counts.undialed(), 3 + 32);
    for (id, hint) in (40..).zip([None, Some(1), Some(2), Some(64)]) {
        let answer = bed.ask(constrained(frame(id, 40), 0, hint)).await;
        assert_eq!(targets(&answer).len(), 3, "a hint of {hint:?}");
    }
    assert_eq!(bed.counts.undialed(), 3 + 32);
}

/// The rows a gateway's wires leave are a list of their own and turn a
/// rotation of their own. Sixteen members speak HTTP alone and sixteen
/// gRPC alone under `round_robin`, and the asks alternate between the
/// two wires: each list is sixteen rows cut out of one build, and every
/// member is the pick once in its list's sixteen asks. Under one
/// rotation between the two each list was drawn at every other row.
/// Wires that take no row leave the build's own list and its rotation:
/// asks that state the one wire a pool speaks, in turn with asks that
/// state none, walk its sixteen rows once.
#[tokio::test]
async fn the_rows_a_gateways_wires_leave_turn_a_rotation_of_their_own() {
    const TWO: &str = "model-two-wires";
    const ONE: &str = "model-one-wire";
    let mut bed = bed(Some(preset("round_robin", 1)), true, (2, 8), []);
    let store = Arc::clone(&bed.store);
    for row in 0..16 {
        ready(
            &store,
            &bed.reads.objects,
            &format!("h{row}"),
            TWO,
            Role::Single,
            WireModes::HTTP,
        );
        ready(
            &store,
            &bed.reads.objects,
            &format!("g{row}"),
            TWO,
            Role::Single,
            WireModes::GRPC,
        );
        ready(
            &store,
            &bed.reads.objects,
            &format!("o{row}"),
            ONE,
            Role::Single,
            WireModes::HTTP,
        );
    }
    let mut picked = Vec::new();
    for ask in 0..64_u8 {
        let wires = if ask.is_multiple_of(2) { 0b001 } else { 0b010 };
        let frame = constrained(frame_for(TWO, ask, u64::from(ask)), wires, None);
        picked.push(targets(&bed.ask(frame).await)[0]);
    }
    assert_eq!(turns(picked), [2; 32], "two wires over one pool");
    let mut picked = Vec::new();
    for ask in 64..80_u8 {
        let frame = frame_for(ONE, ask, u64::from(ask));
        let frame = if ask.is_multiple_of(2) {
            constrained(frame, 0b001, None)
        } else {
            frame
        };
        picked.push(targets(&bed.ask(frame).await)[0]);
    }
    assert_eq!(turns(picked), [1; 16], "wires that take no row");
}
