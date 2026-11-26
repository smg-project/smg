use super::*;

/// The constraints of a host that holds the request to the rows `at` of a
/// pool.
fn holding(at: &[u32]) -> Constraints<'_> {
    Constraints {
        places: Some(at),
        ..Constraints::default()
    }
}

/// The workers of a pick's first target: the prefill's and the decode's.
fn workers_of(decision: &Decision) -> (WorkerId, WorkerId) {
    let Decision::Pick(Pick { targets, .. }) = decision else {
        panic!("a pick, not {decision:?}");
    };
    let target = targets.first().expect("a target");
    (target.id, target.pair.expect("a pair").decode_id)
}

/// The places a side's host holds the request to number the rows of the
/// pool it is handed with. Beside rows this side numbered anew they are
/// read by where each row stood: a place whose row was left out opens none,
/// and the pair is picked among the rows admitted that stand at a place. So
/// a side answers as it does for a host that hands the rows admitted
/// itself, with its places among them. Beside a whole pool the places are
/// read as they are handed.
#[test]
fn a_sides_places_are_read_by_where_each_row_stood() {
    let engine = engine();
    let open = Constraints::default();
    // The first prefill's class has no encoder and the first decode takes
    // no image: each side is cut to its other two rows, the last the
    // fuller.
    let mut cut = Bed::new();
    cut.prefills = (0..3)
        .map(|at| row(at, Role::Prefill, 1, 9 * (at / 2)))
        .collect();
    cut.prefill_classes = vec![2, 1, 1];
    cut.decodes = (0..3)
        .map(|at| row(at, Role::Decode, 1, 9 * (at / 2)))
        .collect();
    cut.decodes[0].vision = false;
    cut.decode_images = true;
    let (facts, _) = cut.asked();
    let decide =
        |sides: [&Constraints<'_>; 3]| decide_under(&cut, &engine, &facts, cut.pools(), sides).0;
    let pair = |sides: [&Constraints<'_>; 3]| pair_of(&decide(sides));
    assert_eq!(pair([&open; 3]), (1, 1));
    // The fuller row alone stands at a place: the idle one is not the
    // pair's.
    let last = holding(&[2]);
    assert_eq!(pair([&last, &open, &open]), (2, 1));
    assert_eq!(pair([&open, &last, &open]), (1, 2));
    // A place whose row was left out opens none: no other row stands in
    // its stead.
    let ends = holding(&[0, 2]);
    assert_eq!(pair([&ends, &open, &open]), (2, 1));
    assert_eq!(pair([&open, &ends, &open]), (1, 2));
    assert_eq!(pair([&ends, &ends, &open]), (2, 2));
    // Both rows admitted stand at a place: the idle one.
    let admitted = holding(&[1, 2]);
    assert_eq!(pair([&admitted, &admitted, &open]), (1, 1));
    // A mask beside the places is numbered with them: the idle row at a
    // place is masked, and the fuller one is the pair's.
    let masked = Constraints {
        stalled: &[1],
        ..holding(&[0, 1, 2])
    };
    assert_eq!(pair([&masked, &masked, &open]), (2, 2));

    // A host that hands the rows admitted itself states its places among
    // them, and the same workers answer.
    let moved = |row: &Candidate, position: u32| Candidate { position, ..*row };
    let mut handed = Bed::new();
    handed.prefills = vec![moved(&cut.prefills[1], 0), moved(&cut.prefills[2], 1)];
    handed.prefill_classes = vec![1, 1];
    handed.decodes = vec![moved(&cut.decodes[1], 0), moved(&cut.decodes[2], 1)];
    handed.decode_images = true;
    let theirs = holding(&[1]);
    for (ours, theirs) in [
        ([&ends, &open, &open], [&theirs, &open, &open]),
        ([&open, &ends, &open], [&open, &theirs, &open]),
        ([&ends, &ends, &open], [&theirs, &theirs, &open]),
    ] {
        let (answer, _) = decide_under(&handed, &engine, &facts, handed.pools(), theirs);
        assert_eq!(workers_of(&decide(ours)), workers_of(&answer));
    }

    // No cut: beside a whole pool the places are read as they are handed.
    let mut whole = Bed::new();
    whole.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 9)];
    whole.prefill_classes = vec![1, 1];
    whole.decodes = vec![row(0, Role::Decode, 1, 0), row(1, Role::Decode, 1, 9)];
    let second = holding(&[1]);
    for (sides, pair) in [
        ([&second, &open, &open], (1, 0)),
        ([&open, &second, &open], (0, 1)),
    ] {
        let (decision, _) = decide_under(&whole, &engine, &facts, whole.pools(), sides);
        assert_eq!(pair_of(&decision), pair);
    }
}

/// Places numbered anew are the decision's own: the next decision of one
/// scratch reads its own places and none of the last.
#[test]
fn places_numbered_anew_are_forgotten_by_the_next_decision() {
    let engine = engine();
    // The first prefill's class has no encoder: the pool is cut to the
    // other two, the last the fuller.
    let mut bed = Bed::new();
    bed.prefills = (0..3)
        .map(|at| row(at, Role::Prefill, 1, 9 * (at / 2)))
        .collect();
    bed.prefill_classes = vec![2, 1, 1];
    let (facts, pool) = bed.asked();
    let encode = EncodePool {
        encoders: &bed.encoders,
        facts: &bed.facts,
        prefill_classes: &bed.prefill_classes,
        affinity: &bed.encode_affinity,
        ..EncodePool::default()
    };
    let items = EncodeItems {
        keys: &bed.keys,
        one: bed.one,
        references: bed.references,
        decode_images: bed.decode_images,
        ring: None,
    };
    let open = Constraints::default();
    let (mut scratch, mut placed) = (Scratch::new(), EncodePlaced::new());
    let mut prefill = |cx: &Constraints<'_>| {
        let decision = engine.decide_encode_pair(
            &facts,
            &pool,
            bed.pools(),
            encode,
            items,
            cx,
            &open,
            &open,
            &mut scratch,
            &mut placed,
        );
        pair_of(&decision).0
    };
    assert_eq!(prefill(&holding(&[2])), 2);
    assert_eq!(prefill(&open), 1);
    assert_eq!(prefill(&masking(&[1])), 2);
    assert_eq!(prefill(&holding(&[1, 2])), 1);
}

/// An item is placed on an encoder at one of the places the encoders' host
/// holds the request to. An encoder outside them takes no key and opens no
/// class, so a prefill it alone hands to is no candidate of the pair. A
/// key's run reads the mask numbered with the encoders offered, and no
/// place.
#[test]
fn an_item_is_placed_on_an_encoder_at_one_of_its_requests_places() {
    let engine = engine();
    let open = Constraints::default();
    // Two encoders of the pair's class, the second the fuller, and one of
    // another class.
    let mut bed = Bed::new();
    bed.encoders = (0..3)
        .map(|at| row(at, Role::Encode, 0, 9 * (at % 2)))
        .collect();
    bed.facts = vec![pushing(1), pushing(1), pushing(2)];
    let (facts, _) = bed.asked();
    let encoder = |held: &Constraints<'_>| {
        let sides = [&open, &open, held];
        let (decision, placed) = decide_under(&bed, &engine, &facts, bed.pools(), sides);
        assert!(decision.pick().is_some(), "a pick, not {decision:?}");
        placed.encoders().first().map(|target| target.position)
    };
    assert_eq!(encoder(&open), Some(0));
    assert_eq!(encoder(&holding(&[1])), Some(1));
    assert_eq!(encoder(&holding(&[1, 2])), Some(1));
    assert_eq!(encoder(&holding(&[0, 1, 2])), Some(0));
    // A place past the pool's end names no encoder.
    assert_eq!(encoder(&holding(&[1, 7])), Some(1));
    // The idle encoder of the class stands outside the places, and a mask
    // names it: the mask is read by where each encoder stood.
    let masked = Constraints {
        stalled: &[0],
        ..holding(&[1, 2])
    };
    assert_eq!(encoder(&masked), Some(1));

    // The idle prefill's one encoder stands outside the places: the pair is
    // the busy prefill, and its encoder takes the item. So it is for two
    // classes the sets' bits hold, and for two far above the rows handed.
    for [idle, busy] in [[1, 2], [u32::MAX - 1, u32::MAX]] {
        let mut classes = Bed::new();
        classes.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 5)];
        classes.prefill_classes = vec![idle, busy];
        classes.encoders = vec![row(0, Role::Encode, 0, 0), row(1, Role::Encode, 0, 0)];
        classes.facts = vec![pushing(idle), pushing(busy)];
        let placed = |held: &Constraints<'_>| {
            let sides = [&open, &open, held];
            let (decision, placed) =
                decide_under(&classes, &engine, &facts, classes.pools(), sides);
            let encoders: Vec<u32> = placed.encoders().iter().map(|t| t.position).collect();
            (pair_of(&decision), encoders)
        };
        assert_eq!(placed(&open), ((0, 0), vec![0]));
        assert_eq!(placed(&holding(&[1])), ((1, 0), vec![1]));
        assert_eq!(placed(&holding(&[0, 1])), ((0, 0), vec![0]));
    }
}

/// A request whose places on a list name workers and none that takes it,
/// though the list holds one, is held off every worker that takes it: it is
/// shed under the places' name. No gap of the membership says it, since the
/// model takes the request. A gap is read over every worker, with places as
/// without. Where a worker at the places takes the request and is not open,
/// the shed is the list's own, by the worker's state.
///
/// Places that name no worker of a list are no such case: nobody stands at
/// them for the pass to leave out. Such places of a side shed the request
/// as they shed its pair alone, beside the rows admitted as beside a whole
/// list. Such places of the encoders shed it as a pool with no open encoder
/// does.
#[test]
fn a_request_held_off_every_worker_that_takes_it_is_shed_under_the_places_name() {
    let stock = engine();
    let open = Constraints::default();
    let held_off = (ShedReason::NoEndpoints, "places");
    let first = holding(&[0]);
    // No place, the place at the end of a list of two, and a place past
    // it.
    let nobody = [holding(&[]), holding(&[2]), holding(&[7])];
    // The first place, which names a worker, and a place that names none.
    let first_and_nobody = holding(&[0, 7]);
    let shed = |bed: &Bed, sides: [&Constraints<'_>; 3]| {
        let (facts, _) = bed.asked();
        shed_of(&decide_under(bed, &stock, &facts, bed.pools(), sides).0)
    };
    // The pair entry's own answer under each side's constraints.
    let alone = |bed: &Bed, [cx, decode_cx]: [&Constraints<'_>; 2]| {
        let (facts, pool) = bed.asked();
        let mut scratch = Scratch::new();
        shed_of(&stock.decide_pair(&facts, &pool, bed.pools(), cx, decode_cx, &mut scratch))
    };

    // The first decode takes no image, and the request asks images of its
    // decode.
    let mut bed = Bed::new();
    bed.decodes = vec![row(0, Role::Decode, 1, 0), row(1, Role::Decode, 1, 0)];
    bed.decodes[0].vision = false;
    bed.decode_images = true;
    assert_eq!(shed(&bed, [&open, &first, &open]), held_off);
    assert_eq!(shed(&bed, [&open, &first_and_nobody, &open]), held_off);
    for nobody in &nobody {
        let pair = alone(&bed, [&open, nobody]);
        assert_ne!(pair.1, "places");
        assert_eq!(shed(&bed, [&open, nobody, &open]), pair);
    }
    // No decode takes images: the gap is the membership's.
    bed.decodes[1].vision = false;
    assert_eq!(shed(&bed, [&open, &first, &open]).1, "decode-vision");
    assert_eq!(shed(&bed, [&open; 3]).1, "decode-vision");

    // The first prefill's class has no encoder.
    let mut bed = Bed::new();
    bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 0)];
    bed.prefill_classes = vec![2, 1];
    assert_eq!(shed(&bed, [&first, &open, &open]), held_off);
    assert_eq!(shed(&bed, [&first_and_nobody, &open, &open]), held_off);
    for nobody in &nobody {
        let pair = alone(&bed, [nobody, &open]);
        assert_ne!(pair.1, "places");
        assert_eq!(shed(&bed, [nobody, &open, &open]), pair);
    }
    // No prefill's class has one.
    bed.prefill_classes = vec![2, 2];
    assert_eq!(shed(&bed, [&first, &open, &open]).1, "ec-compatible");
    assert_eq!(shed(&bed, [&open; 3]).1, "ec-compatible");

    // The first encoder takes no image, and the request states one.
    let mut bed = Bed::new();
    bed.encoders = vec![row(0, Role::Encode, 0, 0), row(1, Role::Encode, 0, 0)];
    bed.encoders[0].vision = false;
    bed.facts = vec![pushing(1), pushing(1)];
    assert_eq!(shed(&bed, [&open, &open, &first]), held_off);
    let by_state = (ShedReason::NoEndpoints, "encoders");
    for nobody in &nobody {
        assert_eq!(shed(&bed, [&open, &open, nobody]), by_state);
    }
    // No encoder takes images.
    bed.encoders[1].vision = false;
    assert_eq!(shed(&bed, [&open, &open, &first]).1, "encode-vision");
    assert_eq!(shed(&bed, [&open; 3]).1, "encode-vision");
    // The encoder at the place states no end: it takes no request,
    // whatever its row says, and the one outside the place takes this one.
    bed.encoders[1].vision = true;
    bed.encoders[0].vision = true;
    bed.facts = vec![pushing(0), pushing(1)];
    assert_eq!(shed(&bed, [&open, &open, &first]), held_off);

    // A side that admits nobody leaves nobody out: its places are the
    // pair's to answer, as without them. The prefill's class has one
    // encoder, which is not ready, beside a ready encoder of a class no
    // prefill names.
    let mut bed = Bed::new();
    bed.encoders = vec![
        unready(row(0, Role::Encode, 0, 0)),
        row(1, Role::Encode, 0, 0),
    ];
    bed.facts = vec![pushing(1), pushing(2)];
    let unheld = shed(&bed, [&open; 3]);
    assert_ne!(unheld.1, "places");
    assert_eq!(shed(&bed, [&first, &open, &open]), unheld);

    // The encoder at the place takes the request and is not ready: the
    // shed is the encode side's, by state. Over its thresholds it reads
    // the same under a binding that steers, and is a saturated fleet of
    // one under a binding that sheds, whatever the encoder outside the
    // place is.
    let mut bed = Bed::new();
    bed.encoders = vec![
        unready(row(0, Role::Encode, 0, 0)),
        row(1, Role::Encode, 0, 0),
    ];
    bed.facts = vec![pushing(1), pushing(1)];
    assert_eq!(shed(&bed, [&open, &open, &first]), by_state);
    bed.encoders[0] = overloaded(row(0, Role::Encode, 0, 0));
    assert_eq!(shed(&bed, [&open, &open, &first]), by_state);
    let saturated = (ShedReason::Saturated, "not-overloaded");
    let mut shedding = engine();
    shedding.shed_when_all_vetoed(0);
    let (facts, _) = bed.asked();
    let held = decide_under(&bed, &shedding, &facts, bed.pools(), [&open, &open, &first]);
    assert_eq!(shed_of(&held.0), saturated);
    // A hard filter behind the veto refuses the encoder outside the place.
    // Over every encoder that is a missing endpoint: the request held to
    // the place reads the encoder there alone.
    let mut behind = engine();
    behind.shed_when_all_vetoed(0);
    for profile in behind.profiles.iter_mut().map(own) {
        profile
            .filters
            .push(Box::new(RefusesOne(bed.encoders[1].id)));
    }
    let under = |sides: [&Constraints<'_>; 3]| {
        shed_of(&decide_under(&bed, &behind, &facts, bed.pools(), sides).0)
    };
    assert_eq!(under([&open; 3]), by_state);
    assert_eq!(under([&open, &open, &first]), saturated);
}

/// A gap the pass reads of the pairs' membership is answered ahead of the
/// encoders' places: the model's workers say it whatever the places are,
/// and no other place closes it. So a request that its encoders' places
/// hold off every encoder that takes it is shed under the gap where one
/// stands, as it is with no place and with a side's places. An encoder
/// outside the places that is not ready changes nothing: the places hold
/// the request off it. What the pair's own decision answers stays behind
/// the pass, under places as under none.
#[test]
fn a_gap_of_the_pairs_membership_stands_ahead_of_the_encoders_places() {
    let stock = engine();
    let open = Constraints::default();
    let first = holding(&[0]);
    let shed = |bed: &Bed, sides: [&Constraints<'_>; 3]| {
        let (facts, _) = bed.asked();
        shed_of(&decide_under(bed, &stock, &facts, bed.pools(), sides).0)
    };
    // The first encoder takes no image, and the request states one: the
    // encoders' place holds it off the one encoder that takes it.
    let held_off = || {
        let mut bed = Bed::new();
        bed.encoders = vec![row(0, Role::Encode, 0, 0), row(1, Role::Encode, 0, 0)];
        bed.encoders[0].vision = false;
        bed.facts = vec![pushing(1), pushing(1)];
        bed
    };
    let mut bed = held_off();
    assert_eq!(shed(&bed, [&open, &open, &first]).1, "places");
    // No decode takes images, and the request asks images of its decode.
    bed.decodes[0].vision = false;
    bed.decode_images = true;
    for sides in [[&open; 3], [&open, &first, &open], [&open, &open, &first]] {
        assert_eq!(shed(&bed, sides).1, "decode-vision");
    }
    // The encoders hand to a class that no prefill names.
    let mut bed = held_off();
    bed.prefill_classes = vec![2];
    for sides in [[&open; 3], [&first, &open, &open], [&open, &open, &first]] {
        assert_eq!(shed(&bed, sides).1, "ec-compatible");
    }
    // The one encoder that takes the request is not ready. With no place
    // the request has no open encoder, and under the place it is held off
    // the encoder all the same.
    let mut bed = held_off();
    bed.encoders[1] = unready(bed.encoders[1]);
    assert_eq!(shed(&bed, [&open; 3]).1, "encoders");
    assert_eq!(shed(&bed, [&open, &open, &first]).1, "places");
    // No prefill pairs with a decode: the pair's own decision says so, and
    // it is asked behind the pass.
    let mut bed = held_off();
    bed.decodes = vec![row(0, Role::Decode, 2, 0)];
    assert_eq!(shed(&bed, [&open; 3]).1, "partnered");
    assert_eq!(shed(&bed, [&open, &open, &first]).1, "places");
}

/// The encoders' places are numbered anew by each decision, as a side's
/// are: the next decision of one scratch holds its item to its own places
/// and to none of the last decision's.
#[test]
fn the_encoders_places_are_forgotten_by_the_next_decision() {
    let engine = engine();
    // Two encoders of the pair's class.
    let mut bed = Bed::new();
    bed.encoders = vec![row(0, Role::Encode, 0, 0), row(1, Role::Encode, 0, 0)];
    bed.facts = vec![pushing(1), pushing(1)];
    let (facts, pool) = bed.asked();
    let encode = EncodePool {
        encoders: &bed.encoders,
        facts: &bed.facts,
        prefill_classes: &bed.prefill_classes,
        affinity: &bed.encode_affinity,
        ..EncodePool::default()
    };
    let items = EncodeItems {
        keys: &bed.keys,
        one: bed.one,
        references: bed.references,
        decode_images: bed.decode_images,
        ring: None,
    };
    let open = Constraints::default();
    let (mut scratch, mut placed) = (Scratch::new(), EncodePlaced::new());
    let mut encoder = |held: &Constraints<'_>| {
        let decision = engine.decide_encode_pair(
            &facts,
            &pool,
            bed.pools(),
            encode,
            items,
            &open,
            &open,
            held,
            &mut scratch,
            &mut placed,
        );
        assert!(decision.pick().is_some(), "a pick, not {decision:?}");
        placed.encoders().first().map(|target| target.position)
    };
    assert_eq!(encoder(&holding(&[1])), Some(1));
    assert_eq!(encoder(&holding(&[0])), Some(0));
    assert_eq!(encoder(&holding(&[1])), Some(1));
}

/// A debug build refuses places that do not ascend where the encode pass
/// numbers them beside a list it narrowed. Read by one walk beside rows
/// that ascend, the places of the last two rows stated backwards would
/// hold the request to the last row alone and say nothing.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "the places are ascending, each once")]
fn a_debug_build_refuses_a_narrowed_sides_places_out_of_order() {
    // The first prefill's class has no encoder: the side is narrowed.
    let mut bed = Bed::new();
    bed.prefills = (0..3).map(|at| row(at, Role::Prefill, 1, 0)).collect();
    bed.prefill_classes = vec![2, 1, 1];
    let (facts, _) = bed.asked();
    let open = Constraints::default();
    let sides = [&holding(&[2, 1]), &open, &open];
    let _ = decide_under(&bed, &engine(), &facts, bed.pools(), sides);
}

/// The encoders' places are refused the same way where the pass holds the
/// open encoders to them.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "the places are ascending, each once")]
fn a_debug_build_refuses_the_encoders_places_out_of_order() {
    let mut bed = Bed::new();
    bed.encoders = vec![row(0, Role::Encode, 0, 0), row(1, Role::Encode, 0, 0)];
    bed.facts = vec![pushing(1), pushing(1)];
    let (facts, _) = bed.asked();
    let open = Constraints::default();
    let sides = [&open, &open, &holding(&[1, 0])];
    let _ = decide_under(&bed, &engine(), &facts, bed.pools(), sides);
}
