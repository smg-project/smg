//! A pair's first decision: the partners the classes name, a prefill's
//! capacity and a decode's room, the places and the demands each side is
//! held to, the ring and the reports a side reads, the ranks a retry
//! leaves, and what a pair that cannot be picked is shed under.

use super::*;

#[test]
fn the_capacity_filters_name_is_the_one_a_full_list_is_read_by() {
    assert_eq!(HasCapacity.name(), HAS_CAPACITY);
}

#[test]
fn a_stamp_writes_each_records_class_and_its_sides_admission_bit() {
    let (mut prefills, mut decodes) = lists(3, 2);
    let classes = PartnerClasses::number(2, [&[0u32][..], &[1][..]]);
    let gates = Gates(|at| at != 1, |at| at == 1);
    stamp(&mut prefills, &mut decodes, &classes, &gates);
    let class = |rows: &[Candidate]| rows.iter().map(|row| row.pair_class).collect::<Vec<_>>();
    // The third prefill stands past the classes: it pairs with nobody.
    assert_eq!(class(&prefills), [1, 2, 0]);
    assert_eq!(class(&decodes), [1, 2]);
    let capacity: Vec<bool> = prefills.iter().map(|row| row.has_capacity).collect();
    assert_eq!(capacity, [true, false, true]);
    let room: Vec<bool> = decodes.iter().map(|row| row.has_room).collect();
    assert_eq!(room, [false, true]);
}

/// Two groups that never pair: whichever prefill the rotation lands
/// on, its decode is one of its own group.
#[test]
fn a_pair_is_decided_among_the_partners_the_classes_name() {
    let (spec, _) = hearing();
    let (mut prefills, mut decodes) = lists(2, 4);
    let classes = PartnerClasses::number(4, [&[0u32, 1][..], &[2, 3][..]]);
    stamp(&mut prefills, &mut decodes, &classes, &OPEN);
    let mut seen = [false; 2];
    for seed in 0..16 {
        let facts = RequestFacts { seed, ..facts() };
        let decision = decide_over(&spec, &facts, &prefills, &decodes, &classes);
        let (prefill, decode) = picked(&decision);
        let group = if decode <= 102 { 1 } else { 2 };
        assert_eq!(group, prefill, "prefill {prefill} with decode {decode}");
        seen[prefill as usize - 1] = true;
    }
    assert_eq!(seen, [true; 2], "the rotation reaches both prefills");
}

/// The capacity bit is the host's own read: a full prefill takes no
/// request while another has room, and a list of full prefills is not
/// a refusal but a request that waits.
#[test]
fn a_full_prefill_is_passed_over_and_a_list_of_full_prefills_reads_as_full() {
    let (spec, _) = hearing();
    let (mut prefills, mut decodes) = lists(2, 1);
    let classes = every(2, 1);
    stamp(
        &mut prefills,
        &mut decodes,
        &classes,
        &Gates(|at| at == 1, |_| true),
    );
    for seed in 0..4 {
        let facts = RequestFacts { seed, ..facts() };
        let decision = decide_over(&spec, &facts, &prefills, &decodes, &classes);
        assert_eq!(picked(&decision), (2, 101));
        assert!(!prefills_full(&decision));
    }
    stamp(
        &mut prefills,
        &mut decodes,
        &classes,
        &Gates(|_| false, |_| true),
    );
    let decision = decide_over(&spec, &facts(), &prefills, &decodes, &classes);
    assert!(prefills_full(&decision), "{decision:?}");
}

/// The room bit is soft: a decode with no room is passed over while
/// another has one, and where none has the list stands and the
/// dispatch waits at its pick's gate.
#[test]
fn a_decode_with_no_room_is_passed_over_and_stands_where_none_has_one() {
    let (spec, _) = hearing();
    let (mut prefills, mut decodes) = lists(1, 2);
    let classes = every(1, 2);
    stamp(
        &mut prefills,
        &mut decodes,
        &classes,
        &Gates(|_| true, |at| at == 1),
    );
    for seed in 0..4 {
        let facts = RequestFacts { seed, ..facts() };
        let decision = decide_over(&spec, &facts, &prefills, &decodes, &classes);
        assert_eq!(picked(&decision), (1, 102));
    }
    stamp(
        &mut prefills,
        &mut decodes,
        &classes,
        &Gates(|_| true, |_| false),
    );
    let none_has = decide_over(&spec, &facts(), &prefills, &decodes, &classes);
    assert_eq!(picked(&none_has).0, 1, "the list stands: {none_has:?}");
}

#[test]
fn a_request_that_states_what_a_pair_is_not_narrowed_by_is_shed_under_that_stage() {
    let (spec, _) = hearing();
    let (mut prefills, mut decodes) = lists(1, 1);
    let classes = every(1, 1);
    stamp(&mut prefills, &mut decodes, &classes, &OPEN);
    let shed_under = |cx: &Constraints<'_>| {
        let decision = decide_records(
            spec.engine(),
            &facts(),
            (cx, [unvetoed(None); 2]),
            &pool(&prefills),
            pools(&prefills, &decodes, &classes),
            &mut Scratch::default(),
        );
        match decision {
            Decision::Shed(shed) => (shed.reason, shed.emptied_by),
            other => panic!("not shed: {other:?}"),
        }
    };
    let demanded = Constraints {
        demands: Demands {
            vision: true,
            ..Demands::default()
        },
        ..Constraints::default()
    };
    assert_eq!(shed_under(&demanded), (ShedReason::NoEndpoints, WIRES));
    let none = Constraints {
        subset_hint: Some(0),
        ..Constraints::default()
    };
    assert_eq!(shed_under(&none), (ShedReason::NoEndpoints, SUBSET_HINT));
    let some = Constraints {
        subset_hint: Some(2),
        ..Constraints::default()
    };
    let declined = (ShedReason::PolicyDeclined, SUBSET_HINT);
    assert_eq!(shed_under(&some), declined);
    let held = Constraints {
        places: Some(&[0]),
        ..Constraints::default()
    };
    let declined = (ShedReason::PolicyDeclined, single::PLACES);
    assert_eq!(shed_under(&held), declined);
}

/// The refusal names the side of the list that holds no record, the
/// prefills' first: the gateway's own pair selection refuses a pool
/// with no worker for that pool's role, and reads the prefills first
/// (`model_gateway/src/routers/common/placement.rs:2860-2867`, on main
/// `d43c7b2a9`).
#[test]
fn one_side_alone_is_no_pair() {
    let (spec, _) = hearing();
    let sides = [
        (2, 0, PairSide::Decode),
        (0, 2, PairSide::Prefill),
        (0, 0, PairSide::Prefill),
    ];
    for (prefills, decodes, side) in sides {
        let classes = every(prefills as usize, decodes as u32);
        let (mut prefills, mut decodes) = lists(prefills, decodes);
        stamp(&mut prefills, &mut decodes, &classes, &OPEN);
        let decision = decide_over(&spec, &facts(), &prefills, &decodes, &classes);
        // A pair's first decision over no candidate is emptied by no
        // stage.
        let Decision::Shed(shed) = &decision else {
            panic!("not shed: {decision:?}");
        };
        assert_eq!((shed.emptied_by, shed.side), ("none", Some(side)));
        assert!(!prefills_full(&decision));
    }
}

/// A decision names each worker by its place in the list its host
/// handed and by its id, and reads the workers and the host's admission
/// as they stand at the call.
#[test]
fn a_pair_is_decided_over_the_hosts_two_lists_as_they_stand_at_the_call() {
    let host = Host::of(2, 3);
    let mut seen = [false; 2];
    for seed in 0..16 {
        seen[host.picked(&host.decide(&OPEN, seed, None)).0] = true;
    }
    assert_eq!(seen, [true; 2], "the rotation reaches both prefills");
    // A prefill that stopped being ready and a decode with no room: the
    // next decision reads both where the host keeps them.
    host.prefills[0].set_status(WorkerStatus::NotReady);
    let one_room = Gates(|_| true, |at| at == 2);
    assert_eq!(host.picked(&host.decide(&one_room, 1, None)), (1, 2));
    // The one prefill left is at its limit: the request waits.
    let full = Gates(|_| false, |_| true);
    assert!(prefills_full(&host.decide(&full, 1, None)));
}

/// A pair its host holds to places is decided over the pools' own lists,
/// and each side opens with its own places: no prefill and no decode
/// outside them is picked, under the stock document and under one whose
/// profiles list no filter. One side may be held alone. A side with no
/// place kept picks no pair. The decode asked again is held to the
/// decodes' places of its own call.
#[test]
fn a_pair_held_to_places_is_handed_no_other_place_on_either_side() {
    for host in [Host::of(3, 4), Host::unfiltered(3, 4)] {
        // Both sides held: every pick is of the places, and the turns
        // reach each of them.
        let mut seen = ([false; 3], [false; 4]);
        for seed in 0..64 {
            let both = [Some(&[1][..]), Some(&[0, 3][..])];
            let picked =
                host.picked(&host.decide_among(&OPEN, seed, None, Constraints::default(), both));
            seen.0[picked.0] = true;
            seen.1[picked.1] = true;
        }
        assert_eq!(seen, ([false, true, false], [true, false, false, true]));
        for seed in 0..32 {
            let prefills = [Some(&[0, 2][..]), None];
            let picked = host.picked(&host.decide_among(
                &OPEN,
                seed,
                None,
                Constraints::default(),
                prefills,
            ));
            assert_ne!(picked.0, 1, "seed {seed}");
            let decodes = [None, Some(&[2][..])];
            let picked =
                host.picked(&host.decide_among(&OPEN, seed, None, Constraints::default(), decodes));
            assert_eq!(picked.1, 2, "seed {seed}");
        }
        // No decode kept: no pair is picked, whatever is ready.
        let none = [None, Some(&[][..])];
        assert!(host
            .decide_among(&OPEN, 1, None, Constraints::default(), none)
            .pick()
            .is_none());
        // The decode asked again behind a sent prefill.
        let (sent, _) = host.picked(&host.decide(&OPEN, 3, None));
        let held = prefill_alone(worker_id(host.prefills[sent].as_ref()));
        for seed in 0..16 {
            let decodes = [None, Some(&[1][..])];
            let again =
                host.decide_among(&OPEN, seed, Some(&held), Constraints::default(), decodes);
            assert_eq!(host.picked(&again), (sent, 1), "seed {seed}");
        }
    }
}

/// A decode among the decode places is handed only with the prefill it
/// partners. Two prefills, each with its own decode, both decode places
/// stated and the prefill list held to the second: the pair is the
/// second prefill with the second decode at every turn. So it is where
/// a place of either side stands past its list's end, which names no
/// row. With the prefill list free and the first decode not ready, the
/// second decode place alone opens a prefill: the same pair. The
/// gateway's classic router picks a pair's decode among the picked
/// prefill's own partners that are open for the request
/// (`model_gateway/src/routers/common/placement.rs:1301-1306`,
/// `:1389-1394`, `:1548-1553`, at main `b8c15d40a`).
#[test]
fn a_decode_among_the_places_is_handed_only_with_the_prefill_it_partners() {
    let (spec, _) = hearing();
    let own: Vec<[u32; 1]> = (0..2).map(|at| [at]).collect();
    let classes = PartnerClasses::number(2, own.iter().map(|one| one.as_slice()));
    let free = Constraints::default();
    let decide = |seed: u64, first_ready: bool, places: [Option<&[u32]>; 2]| {
        let (mut prefills, mut decodes) = lists(2, 2);
        stamp(&mut prefills, &mut decodes, &classes, &OPEN);
        decodes[0].health.healthy = first_ready;
        let facts = RequestFacts { seed, ..facts() };
        decide_records(
            spec.engine(),
            &facts,
            (&free, places.map(unvetoed)),
            &pool(&prefills),
            pools(&prefills, &decodes, &classes),
            &mut Scratch::default(),
        )
    };
    for seed in 0..24 {
        let held = [Some(&[1][..]), Some(&[0, 1][..])];
        assert_eq!(picked(&decide(seed, true, held)), (2, 102), "seed {seed}");
        let past = [Some(&[1, 7][..]), Some(&[0, 1, 9][..])];
        assert_eq!(picked(&decide(seed, true, past)), (2, 102), "seed {seed}");
        let second = [None, Some(&[0, 1][..])];
        assert_eq!(
            picked(&decide(seed, false, second)),
            (2, 102),
            "seed {seed}"
        );
    }
}

/// A side's places stand beside what the request states, and hold the
/// other side to nothing. Three prefills and four decodes, every pair
/// open. With the prefill list held to its second place the turns reach
/// every decode, and with the decode list held to its fourth they reach
/// every prefill. A decode its host just failed to reach steps aside
/// among the decode places while another of them is left. A prefill
/// list held to no place picks no pair, as a pair with no prefill picks
/// none. Behind the third prefill the
/// decode asked again reads the decodes' places alone: the prefills'
/// places, which leave that prefill out, are not read, and decode
/// places that keep none pick none. The gateway's classic router steps
/// aside from the engines a request tried on each side of a pair, among
/// the workers its request's rule admits
/// (`model_gateway/src/routers/common/placement.rs:1301-1306`,
/// `:1492-1494`, `:1568-1570`, at main `b8c15d40a`).
#[test]
fn a_sides_places_stand_beside_the_requests_word_and_hold_the_other_side_to_nothing() {
    let host = Host::of(3, 4);
    let free = Constraints::default;
    let mut decodes = [false; 4];
    for seed in 0..64 {
        let prefill = [Some(&[1][..]), None];
        let picked = host.picked(&host.decide_among(&OPEN, seed, None, free(), prefill));
        assert_eq!(picked.0, 1, "seed {seed}");
        decodes[picked.1] = true;
    }
    assert_eq!(decodes, [true; 4]);
    let mut prefills = [false; 3];
    for seed in 0..64 {
        let decode = [None, Some(&[3][..])];
        let picked = host.picked(&host.decide_among(&OPEN, seed, None, free(), decode));
        assert_eq!(picked.1, 3, "seed {seed}");
        prefills[picked.0] = true;
    }
    assert_eq!(prefills, [true; 3]);
    let tried = [worker_id(host.decodes[0].as_ref())];
    for seed in 0..16 {
        let excluding = Constraints {
            excluded: &tried,
            ..Constraints::default()
        };
        let among = [None, Some(&[0, 3][..])];
        let picked = host.picked(&host.decide_among(&OPEN, seed, None, excluding, among));
        assert_eq!(picked.1, 3, "seed {seed}");
    }
    let none = [Some(&[][..]), None];
    let unheld = host.decide_among(&OPEN, 1, None, free(), none);
    assert_eq!(shed_for(&unheld), (ShedReason::NoEndpoints, "none"));
    let held = prefill_alone(worker_id(host.prefills[2].as_ref()));
    for seed in 0..16 {
        let other = [Some(&[0][..]), Some(&[1][..])];
        let again = host.decide_among(&OPEN, seed, Some(&held), free(), other);
        assert_eq!(host.picked(&again), (2, 1), "seed {seed}");
    }
    let none = [None, Some(&[][..])];
    let again = host.decide_among(&OPEN, 1, Some(&held), free(), none);
    assert_eq!(shed_for(&again), (ShedReason::NoEndpoints, "partners"));
}

/// A pair that states a hard demand is decided over its two lists,
/// each side held to the workers that meet the demand: every pick
/// names one of them on each side, and each of them takes its turn.
/// With no demand the other workers take theirs.
///
/// The classic router holds both of a pair's pools to the request's
/// own rule (`model_gateway/src/routers/common/placement.rs:1227`, the
/// rule's read at `:1262-1266`; its cases at `:3272` and `:3324`, at
/// main `b8c15d40a`).
#[test]
fn a_pair_that_states_a_demand_is_decided_among_the_workers_of_each_side_that_meet_it() {
    // The second prefill takes images, and so do the first and the
    // third decode.
    let host = stating(2, 3, |side, at| FixedFacts {
        vision: if side == 0 { at == 1 } else { at != 1 },
        ..FixedFacts::default()
    });
    let pixels = demanding(Demands {
        vision: true,
        ..Demands::default()
    });
    let seeing = (BTreeSet::from([1]), BTreeSet::from([0, 2]));
    assert_eq!(places_named(&host, pixels, [None, None]), seeing);
    let all = (BTreeSet::from([0, 1]), BTreeSet::from([0, 1, 2]));
    assert_eq!(
        places_named(&host, Constraints::default(), [None, None]),
        all
    );
}

/// A demand is judged among the places a pair's host states for a
/// side: the side is held to those of them whose worker meets it, and
/// a side none of whose places meets it sheds the pair, whatever a
/// worker outside them states.
#[test]
fn a_pairs_demand_is_judged_among_the_places_its_host_states() {
    // The second and the third worker of each list take images.
    let host = stating(3, 3, |_, at| FixedFacts {
        vision: at != 0,
        ..FixedFacts::default()
    });
    let pixels = demanding(Demands {
        vision: true,
        ..Demands::default()
    });
    let held = places_named(&host, pixels, [Some(&[0, 1]), Some(&[1, 2])]);
    assert_eq!(held, (BTreeSet::from([1]), BTreeSet::from([1, 2])));
    let blind = host.decide_among(&OPEN, 0, None, pixels, [None, Some(&[0])]);
    assert_eq!(shed_of(&blind), (ShedReason::NoEndpoints, "vision"));
}

/// A pair's request that states both a hard demand and places of its
/// own is declined under the places' name, whether its sides meet the
/// demand or not: a pair's places are its lists' to state, and that
/// answer comes ahead of the demand's.
#[test]
fn a_pair_that_states_a_demand_and_places_of_its_own_is_declined_under_the_places() {
    let pixels = demanding(Demands {
        vision: true,
        ..Demands::default()
    });
    let stated = Constraints {
        places: Some(&[0]),
        ..pixels
    };
    let declined = (ShedReason::PolicyDeclined, single::PLACES);
    // Every worker takes images, then none does.
    for sees in [true, false] {
        let host = stating(1, 1, |_, _| FixedFacts {
            vision: sees,
            ..FixedFacts::default()
        });
        let decision = host.decide_among(&OPEN, 0, None, stated, [None, None]);
        assert_eq!(shed_of(&decision), declined, "takes images: {sees}");
    }
}

/// Both sides of a pair take the hash ring their caller lends, each
/// over its own list. The gateway hands a pair its model's one ring
/// where either side's policy reads it
/// (`model_gateway/src/routers/common/placement.rs`, lines 1574 to 1585
/// at 882be6f4f), and each side's policy places by its two rules
/// (`model_gateway/src/policies/consistent_hashing.rs`, lines 99 to
/// 140). A side takes the ring's worker only where its own list holds
/// an eligible one: the walk passes the other side's workers, and a
/// worker of its own that is not ready. A value no worker of the side
/// takes goes by its position over that side's eligible workers.
#[test]
fn each_side_of_a_pair_places_on_the_ring_its_caller_lends() {
    let document = PolicySpec::pair_presets("consistent_hashing", "consistent_hashing");
    let spec = AppliedSpec::validate(&document.expect("the pair presets"));
    let host = Host::under((spec.expect("a valid document"), Arc::default()), 3, 3);
    let lent = |workers: &str| {
        let address = |at: &str| format!("grpc://{at}:9000");
        InOrder(workers.split(' ').map(address).collect())
    };
    // The pair a keyed request is placed on where it stands at
    // `position` of the ring lent.
    let placed = |ring: &InOrder, position: u64| {
        let [prefill, decode] = &host.lists;
        let classes = every(3, 3);
        let lists = PairLists {
            prefill: prefill.lent(),
            decode: decode.lent(),
            classes: &classes,
            admission: &OPEN,
            prefill_places: None,
            decode_places: None,
            prefill_stalled: &[],
            prefill_held: &[],
            decode_stalled: &[],
            decode_held: &[],
        };
        let positions = [position];
        let facts = RequestFacts {
            ring: Some(ring),
            ring_positions: &positions,
            ..keyed()
        };
        let mut request = Request {
            id: b"",
            facts,
            constraints: Constraints::default(),
        };
        let now = Instant::now();
        host.picked(&lists.decide(&host.engine, None, now, &mut request, None))
    };
    // The model's one ring holds both lists' workers: each side takes
    // the first worker of its own list that the walk reaches.
    let model = lent("decode-1 prefill-2 decode-0 prefill-0 prefill-1 decode-2");
    assert_eq!(placed(&model, 0), (2, 1));
    assert_eq!(placed(&model, 3), (0, 2));
    // A worker that is not ready is passed over on its side.
    host.prefills[2].set_status(WorkerStatus::NotReady);
    host.decodes[1].set_status(WorkerStatus::NotReady);
    assert_eq!(placed(&model, 0), (0, 0));
    // A ring that reaches no eligible worker of a side: the value goes
    // by its position over that side's eligible workers, two here.
    let no_prefill = lent("decode-2 decode-0 prefill-2");
    assert_eq!(placed(&no_prefill, 3), (1, 2));
    let no_decode = lent("prefill-1 decode-1");
    assert_eq!(placed(&no_decode, 4), (1, 0));
}

/// Under the prefix form each side of a pair is placed on the lent ring
/// too, from the first place its caller states whose owner is within
/// the bound: the first eligible worker of its own list that the walk
/// reaches there, as the gateway's own policy takes it from the pair's
/// one ring (`model_gateway/src/policies/prefix_hash.rs`, lines 291 to
/// 315 at 882be6f4f). Every worker here is idle: the first place is
/// the one taken.
#[test]
fn each_side_of_a_pair_under_the_prefix_form_places_on_the_lent_ring() {
    let document = PolicySpec::pair_presets("prefix_hash", "prefix_hash");
    let spec = AppliedSpec::validate(&document.expect("the pair presets"));
    let host = Host::under((spec.expect("a valid document"), Arc::default()), 3, 3);
    let address = |at: &str| format!("grpc://{at}:9000");
    let model = "decode-1 prefill-2 decode-0 prefill-0 prefill-1 decode-2";
    let ring = InOrder(model.split(' ').map(address).collect());
    let placed = |position: u64| {
        let classes = every(3, 3);
        let positions = [position];
        // The host's own request over its two lists, every pair open.
        let Asks(mut request, lists, _) = host.asks(&classes, false);
        request.facts = RequestFacts {
            ring: Some(&ring),
            ring_positions: &positions,
            ..keyed()
        };
        let now = Instant::now();
        host.picked(&lists.decide(&host.engine, None, now, &mut request, None))
    };
    assert_eq!(placed(0), (2, 1));
    assert_eq!(placed(3), (0, 2));
}

/// Each side of a pair whose two sides read the lent ring by two words
/// is placed from its own places: the prefill side from the request's
/// list, the decode side from the list stated for it
/// ([`RequestFacts::decode_ring_positions`]), as each of the gateway's
/// two policies walks the pair's one ring from its own place
/// (`model_gateway/src/routers/common/placement.rs`, lines 3016 to 3031
/// and 3098 to 3101 at a06805492). The plain ring ahead of the prefix
/// form, and the reverse. A decode side stated no list of its own reads
/// the request's one list. One stated an empty list is placed by its
/// stage's own ring, and the prefill side keeps its place.
#[test]
fn each_side_of_a_pair_of_two_ring_words_places_from_its_own_places() {
    let address = |at: &str| format!("grpc://{at}:9000");
    let model = "decode-1 prefill-2 decode-0 prefill-0 prefill-1 decode-2";
    let ring = InOrder(model.split(' ').map(address).collect());
    let (plain, prefix) = ("consistent_hashing", "prefix_hash");
    for (prefill, decode) in [(plain, prefix), (prefix, plain)] {
        let document = PolicySpec::pair_presets(prefill, decode);
        let spec = AppliedSpec::validate(&document.expect("the pair presets"));
        let host = Host::under((spec.expect("a valid document"), Arc::default()), 3, 3);
        let placed = |position: u64, decode_ring_positions: Option<&[u64]>| {
            let classes = every(3, 3);
            let positions = [position];
            // The host's own request over its two lists, every pair open.
            let Asks(mut request, lists, _) = host.asks(&classes, false);
            request.facts = RequestFacts {
                ring: Some(&ring),
                ring_positions: &positions,
                decode_ring_positions,
                ..keyed()
            };
            let now = Instant::now();
            host.picked(&lists.decide(&host.engine, None, now, &mut request, None))
        };
        // The prefill from the first place and the decode from the
        // fourth, then the reverse.
        assert_eq!(placed(0, Some(&[3])), (2, 2), "{prefill}, {decode}");
        assert_eq!(placed(3, Some(&[0])), (0, 1), "{prefill}, {decode}");
        assert_eq!(placed(0, None), (2, 1), "{prefill}, {decode}");
        assert_eq!(placed(0, Some(&[])).0, 2, "{prefill}, {decode}");
    }
}

/// The engine's own task, turned once: it takes what stands ready for
/// it, here the newest snapshot on the channel its host handed.
async fn turned(engine: &InProcess) {
    let step = Duration::from_millis(1);
    assert!(tokio::time::timeout(step, engine.run()).await.is_err());
}

/// Each side of a pair that scores by load reads its workers' reports in
/// the snapshot its host hands for the call: the prefill and the decode
/// whose queue is empty are picked at every seed. The decode decided
/// again behind its prefill reads the snapshot of its own call, and
/// keeps the prefill that was sent whatever that snapshot says of the
/// prefills. Handed no snapshot, the same engine reads no report: no
/// worker here holds a request in flight, so each scores alike and the
/// picks spread over both lists. The gateway's classic router reads the
/// reports where a side's policy has workers to choose between: each
/// side of a pair asks its policy
/// (`model_gateway/src/routers/common/placement.rs:1521` and `:1583`),
/// and `least_load` reads its snapshot and each worker's report in it
/// (`model_gateway/src/policies/least_load.rs:918` and `:450`, at main
/// `ddd8a44cb`), behind its answer for a side's one healthy worker
/// (`:1119-1120`). A decode pool all over its overload thresholds is not
/// asked of its policy there: its pick is the least loaded partner
/// (`placement.rs:1575-1578`). No worker of this case is over one.
#[test]
fn a_pair_that_scores_by_load_reads_the_reports_its_host_hands() {
    let (host, classes) = of_presets("least_load", "least_load");
    let mut unread = [[0_u32; 4]; 2];
    for seed in 0..64 {
        let fed = host.asks_by(&classes, (seed, false), Some(queues(2, 1)));
        assert_eq!(host.first(&fed), (2, 1), "seed {seed}");
        let (prefill, decode) = host.first(&host.asks_by(&classes, (seed, false), None));
        unread[0][prefill] += 1;
        unread[1][decode] += 1;
    }
    assert!(
        unread.iter().flatten().all(|&picks| picks > 0),
        "{unread:?}"
    );
    // The decode decided again: in its call's snapshot another decode
    // has the empty queue, and another prefill too. The prefill that
    // was sent stays the pair's.
    for seed in 0..16 {
        let now = Instant::now();
        let sent = host.asks_by(&classes, (seed, true), Some(queues(2, 1)));
        let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &sent, now)
            .expect("a decision");
        assert_eq!(host.picked(guard.decision()), (2, 1), "seed {seed}");
        guard.dispatched(now);
        let call = host.asks_by(&classes, (seed, true), Some(queues(0, 3)));
        let again = *guard.decide_decode_under(&call, now).expect("a decision");
        assert_eq!(host.picked(&again), (2, 3), "seed {seed}");
    }
}

/// The snapshot reaches a side that scores by load beside a side that
/// reads no report. With round robin prefills the decode whose queue is
/// empty is picked at every seed, and with round robin decodes the
/// prefill whose queue is empty.
#[test]
fn a_side_that_scores_by_load_reads_the_reports_beside_a_side_that_reads_none() {
    for (prefill, decode) in [("round_robin", "least_load"), ("least_load", "round_robin")] {
        let (host, classes) = of_presets(prefill, decode);
        for seed in 0..64 {
            let fed = host.asks_by(&classes, (seed, false), Some(queues(2, 1)));
            let picked = host.first(&fed);
            if decode == "least_load" {
                assert_eq!(picked.1, 1, "the decode at seed {seed}");
            } else {
                assert_eq!(picked.0, 2, "the prefill at seed {seed}");
            }
        }
    }
}

/// A pair whose request gives no snapshot reads the newest the engine's
/// own task keeps at hand, as a single request does. At its first
/// decision and at its decode decided again, the worker whose queue is
/// empty in the reports the load monitor published last is picked at
/// every seed, by the entries that issue a lease and by those that
/// issue none. Until the task has read the channel the engine holds no
/// snapshot, and the picks spread. A snapshot the request gives is read
/// in place of the engine's.
#[tokio::test(start_paused = true)]
async fn a_pair_whose_request_gives_no_snapshot_reads_the_engines_newest() {
    let (host, classes) = of_presets("least_load", "least_load");
    let mut channel = None;
    let feed = TestFeed::attached_to(|loads| channel = loads);
    // The load monitor's side: the reports published again.
    let publish = |prefill, decode| {
        let published = reports(prefill, decode);
        feed.publish(&published.into_iter().collect());
    };
    publish(2, 1);
    host.engine.set_loads(channel.expect("the feed's channel"));
    let bare = |seed, later| host.asks_by(&classes, (seed, later), None);
    let mut unread = [[0_u32; 4]; 2];
    for seed in 0..64 {
        let (prefill, decode) = host.first(&bare(seed, false));
        unread[0][prefill] += 1;
        unread[1][decode] += 1;
    }
    assert!(
        unread.iter().flatten().all(|&picks| picks > 0),
        "{unread:?}"
    );
    turned(&host.engine).await;
    for seed in 0..32 {
        assert_eq!(host.first(&bare(seed, false)), (2, 1), "seed {seed}");
        let given = host.asks_by(&classes, (seed, false), Some(queues(0, 3)));
        assert_eq!(host.first(&given), (0, 3), "seed {seed}");
    }
    // The decode decided again reads what the monitor published since
    // the first decision, and keeps the prefill that was sent.
    for seed in 0..16 {
        publish(2, 1);
        turned(&host.engine).await;
        let now = Instant::now();
        let sent = bare(seed, true);
        let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &sent, now)
            .expect("a decision");
        assert_eq!(host.picked(guard.decision()), (2, 1), "seed {seed}");
        guard.dispatched(now);
        let unleased = host.engine.decide_without_lease(&sent, now);
        let (unleased, _) = unleased.expect("a decision");
        assert_eq!(host.picked(&unleased), (2, 1), "seed {seed}");
        publish(0, 3);
        turned(&host.engine).await;
        let call = bare(seed, true);
        let again = *guard.decide_decode_under(&call, now).expect("a decision");
        assert_eq!(host.picked(&again), (2, 3), "seed {seed}");
        let held = target_at(&unleased, 0);
        let again = host.engine.decide_decode_without_lease(&call, &held, now);
        assert_eq!(
            host.picked(&again.expect("a decision")),
            (2, 3),
            "seed {seed}"
        );
    }
}

/// The entries that issue no lease read the reports as the leasing
/// entries do ([`InProcess::decide_without_lease`],
/// [`InProcess::decide_decode_without_lease`]). At every seed a
/// pair's first decision picks the two workers whose queues are empty
/// in the snapshot its request gives. Its decode decided again picks
/// the decode whose queue is empty in the snapshot of its own call,
/// behind the prefill the first decision named.
#[test]
fn a_pair_decided_without_a_lease_reads_the_reports_its_host_hands() {
    let (host, classes) = of_presets("least_load", "least_load");
    let now = Instant::now();
    for seed in 0..32 {
        let asks = host.asks_by(&classes, (seed, true), Some(queues(2, 1)));
        let first = host.engine.decide_without_lease(&asks, now);
        let (first, _) = first.expect("a decision");
        assert_eq!(host.picked(&first), (2, 1), "seed {seed}");
        let held = target_at(&first, 0);
        let call = host.asks_by(&classes, (seed, true), Some(queues(0, 3)));
        let again = host.engine.decide_decode_without_lease(&call, &held, now);
        let again = again.expect("a decision");
        assert_eq!(host.picked(&again), (2, 3), "seed {seed}");
    }
}

/// A pair that exists among the places is found. Six prefills, each
/// with its own decode, and the decode list held to the sixth decode's
/// place: the sixth prefill is picked at every turn. A pick among all
/// six names four of them a turn and would miss it on some. The
/// gateway's classic router does the same by the request's own rule: a
/// prefill is a candidate only where one of its partners is open for
/// the request (`model_gateway/src/routers/common/placement.rs:1301-1306`,
/// `:1389-1394`, `:1423-1437`, at main `c738c8d3b`).
#[test]
fn a_prefill_whose_partners_stand_outside_the_decode_places_is_no_candidate() {
    let (spec, _) = hearing();
    let (mut prefills, mut decodes) = lists(6, 6);
    let own: Vec<[u32; 1]> = (0..6).map(|at| [at]).collect();
    let classes = PartnerClasses::number(6, own.iter().map(|one| one.as_slice()));
    stamp(&mut prefills, &mut decodes, &classes, &OPEN);
    let free = Constraints::default();
    for seed in 0..24 {
        let facts = RequestFacts { seed, ..facts() };
        let decision = decide_records(
            spec.engine(),
            &facts,
            (&free, [unvetoed(None), unvetoed(Some(&[5]))]),
            &pool(&prefills),
            pools(&prefills, &decodes, &classes),
            &mut Scratch::default(),
        );
        assert_eq!(picked(&decision), (6, 106), "seed {seed}");
    }
}

/// What a pair's shed says is said of the prefills the request may take.
/// The one prefill place holds a worker that is not ready, the prefills
/// outside the places are ready, and every decode is over its
/// thresholds: the shed names the prefill side's stage, not the decodes'
/// veto, which stands against prefills the request may not take. The
/// gateway's classic router names the prefill role there: no prefill
/// its request's rule admits is available
/// (`model_gateway/src/routers/common/placement.rs:1301-1306`,
/// `:1342-1346`, `:1395-1413`, at main `c738c8d3b`).
#[test]
fn a_pair_with_no_ready_prefill_among_its_places_is_shed_at_the_prefill_stage() {
    let (spec, _) = hearing();
    let (mut prefills, mut decodes) = lists(3, 2);
    let classes = every(3, 2);
    stamp(&mut prefills, &mut decodes, &classes, &OPEN);
    prefills[0].health.healthy = false;
    for decode in &mut decodes {
        decode.health.overloaded = true;
    }
    let free = Constraints::default();
    let decision = decide_records(
        spec.engine(),
        &facts(),
        (&free, [unvetoed(Some(&[0])), unvetoed(None)]),
        &pool(&prefills),
        pools(&prefills, &decodes, &classes),
        &mut Scratch::default(),
    );
    match decision {
        Decision::Shed(shed) => assert_eq!(
            (shed.reason, shed.emptied_by),
            (ShedReason::NoEndpoints, "healthy")
        ),
        other => panic!("not shed: {other:?}"),
    }
}

/// The all-vetoed rule keeps to a side's places. Two prefills, each
/// with its own decode, and the decode list held to the second decode's
/// place, which is over its thresholds. Under a binding that steers the
/// pair is the second prefill with that decode, whatever stands outside
/// the places: a first decode over its thresholds too opens no prefill,
/// and a free one does not keep the pool from the rule. Under a binding
/// that sheds both are refused as saturated. Where the one place holds
/// a decode that is not ready, a decode outside it that is over its
/// thresholds or doubted takes nothing: the rule looks at every row of
/// the side and the pair is refused at the place's own stage, under
/// either word. The gateway's classic router counts only the decodes
/// its request's rule admits when it lifts a decode pool, and pairs
/// among those
/// (`model_gateway/src/routers/common/placement.rs:1301-1306`,
/// `:1315-1321`, `:1341`, `:1389-1394`, `:1575-1578`, at main
/// `ebbcd9c6d`).
#[test]
fn the_all_vetoed_rule_keeps_to_a_sides_places() {
    use crate::{document::spec::OverloadSpec, engine::decision::LiftedVeto};
    let spec_of = |shed: bool| {
        let presets = PolicySpec::pair_presets("passthrough", "passthrough");
        let mut document = presets.expect("the pair presets");
        document.generation = 3;
        document.bindings[0].overload = Some(OverloadSpec {
            shed_when_all_vetoed: shed,
            ..OverloadSpec::default()
        });
        AppliedSpec::validate(&document).expect("a valid document")
    };
    let (steers, sheds) = (spec_of(false), spec_of(true));
    let own: Vec<[u32; 1]> = (0..2).map(|at| [at]).collect();
    let classes = PartnerClasses::number(2, own.iter().map(|one| one.as_slice()));
    let free = Constraints::default();
    // The two lists as `health` leaves the decodes, decided with the
    // decode list held to its second place.
    let decide = |spec: &AppliedSpec, health: &dyn Fn(&mut [Candidate])| {
        let (mut prefills, mut decodes) = lists(2, 2);
        stamp(&mut prefills, &mut decodes, &classes, &OPEN);
        health(&mut decodes);
        decide_records(
            spec.engine(),
            &facts(),
            (&free, [unvetoed(None), unvetoed(Some(&[1]))]),
            &pool(&prefills),
            pools(&prefills, &decodes, &classes),
            &mut Scratch::default(),
        )
    };
    let shed_of = |decision: Decision| match decision {
        Decision::Shed(shed) => (shed.reason, shed.emptied_by),
        other => panic!("not shed: {other:?}"),
    };
    for first_over in [true, false] {
        let over = |decodes: &mut [Candidate]| {
            decodes[0].health.overloaded = first_over;
            decodes[1].health.overloaded = true;
        };
        let steered = decide(&steers, &over);
        assert_eq!(picked(&steered), (2, 102), "first over {first_over}");
        assert_eq!(steered.trace().lifted, LiftedVeto::Overload);
        assert_eq!(
            shed_of(decide(&sheds, &over)),
            (ShedReason::Saturated, "not-overloaded"),
            "first over {first_over}"
        );
    }
    for doubted in [false, true] {
        let outside = |decodes: &mut [Candidate]| {
            decodes[0].health.overloaded = !doubted;
            decodes[0].health.stalled = doubted;
            decodes[1].health.healthy = false;
        };
        for spec in [&steers, &sheds] {
            assert_eq!(
                shed_of(decide(spec, &outside)),
                (ShedReason::NoEndpoints, "healthy"),
                "doubted {doubted}"
            );
        }
    }
}

/// The lifted rule opens no prefill for a decode outside the decode
/// places. Six prefills, each with its own decode, every decode over
/// its thresholds, and the decode list held to the sixth decode's
/// place. Under a binding that steers the pair is the sixth prefill
/// with that decode at every turn, the overload veto lifted. Were the
/// other five decodes to open their prefills too, the prefills' pick,
/// which names four of the six a turn, would miss the sixth on some
/// turns and no pair would be found. The gateway's classic router lifts
/// a decode pool over the workers its request's rule admits, and takes
/// a prefill as a candidate only where one of its partners is open
/// among those
/// (`model_gateway/src/routers/common/placement.rs:1301-1306`,
/// `:1315-1321`, `:1341-1345`, `:1389-1394`, `:1423-1437`, at main
/// `9f303447f`).
#[test]
fn the_lifted_rule_opens_no_prefill_for_a_decode_outside_the_decode_places() {
    use crate::{document::spec::OverloadSpec, engine::decision::LiftedVeto};
    let presets = PolicySpec::pair_presets("round_robin", "round_robin");
    let mut document = presets.expect("the pair presets");
    document.generation = 3;
    document.bindings[0].overload = Some(OverloadSpec {
        shed_when_all_vetoed: false,
        ..OverloadSpec::default()
    });
    let steers = AppliedSpec::validate(&document).expect("a valid document");
    let (mut prefills, mut decodes) = lists(6, 6);
    let own: Vec<[u32; 1]> = (0..6).map(|at| [at]).collect();
    let classes = PartnerClasses::number(6, own.iter().map(|one| one.as_slice()));
    stamp(&mut prefills, &mut decodes, &classes, &OPEN);
    for decode in &mut decodes {
        decode.health.overloaded = true;
    }
    let free = Constraints::default();
    for seed in 0..24 {
        let facts = RequestFacts { seed, ..facts() };
        let decision = decide_records(
            steers.engine(),
            &facts,
            (&free, [unvetoed(None), unvetoed(Some(&[5]))]),
            &pool(&prefills),
            pools(&prefills, &decodes, &classes),
            &mut Scratch::default(),
        );
        assert_eq!(picked(&decision), (6, 106), "seed {seed}");
        assert_eq!(decision.trace().lifted, LiftedVeto::Overload, "seed {seed}");
    }
}

/// A caller held to the decodes it reaches (`decode_places`) is paired
/// by the table of the full lists. The first prefill, a partner of
/// every decode, takes each decode its caller reaches and no other.
/// The second, a partner of the one decode both callers reach, takes
/// that one. The decode records are written whole, each with its class.
#[test]
fn each_gateway_pairs_its_reachable_decodes_by_the_full_lists_table() {
    let all = [0_u32, 1, 2, 3, 4, 5, 6, 7, 8];
    let shared = [1];
    let host = Host::of(2, all.len());
    let classes = PartnerClasses::number(all.len(), [&all[..], &shared[..]]);
    let full_records: Vec<_> = host
        .decodes
        .iter()
        .map(|worker| worker_id(worker.as_ref()))
        .zip(classes.decodes().iter().copied())
        .collect();
    let now = Instant::now();

    let partners = |prefill: usize, places: Option<&[u32]>| {
        let other = worker_id(host.prefills[1 - prefill].as_ref());
        let mut excluded = vec![other];
        let mut drawn = Vec::new();
        loop {
            let mut ask = host.asks(&classes, false);
            ask.1.decode_places = places;
            ask.0.constraints.excluded = &excluded;
            let (decision, _) = host
                .engine
                .decide_without_lease(&ask, now)
                .expect("a pair decision");
            let (chosen, decode) = host.picked(&decision);
            assert_eq!(chosen, prefill, "decode places {places:?}");

            // A soft exclusion allows a repeat once every partner has been drawn.
            let id = full_records[decode].0;
            if excluded.contains(&id) {
                break;
            }
            excluded.push(id);
            drawn.push(decode as u32);
        }

        BUFFERS.with_borrow(|buffers| {
            let written: Vec<_> = buffers.rows[1]
                .iter()
                .map(|row| (row.id, row.pair_class))
                .collect();
            assert_eq!(written, full_records, "decode classes for {places:?}");
        });
        drawn.sort_unstable();
        drawn
    };

    assert_eq!(partners(0, None), all);
    assert_eq!(partners(1, None), shared);

    // Place 1 is shared. Each other place is local to one gateway.
    let gateway_a = [0, 1, 3, 5, 7];
    let gateway_b = [1, 2, 4, 6, 8];
    for places in [gateway_a, gateway_b] {
        assert_eq!(partners(0, Some(&places)), places, "reachable {places:?}");
        assert_eq!(partners(1, Some(&places)), shared, "reachable {places:?}");
    }
}

/// A pair's retry leaves every rank of the engines it tried, a side at
/// a time: a tried prefill takes its engine's other rank out of the
/// prefills and leaves the decodes as they were, a tried decode does
/// the same on its side, and both together move both sides. Named the
/// tried rank alone, each side kept its engine's other rank. That is
/// the gateway's classic answer for a pair: each side steps aside from
/// the engines the request tried
/// (`model_gateway/src/routers/common/placement.rs:1486-1493` and
/// `:1562-1569` at main `75d279287`). The list such a retry is decided by
/// adds each side's ranks in the order of the pair's lists.
#[test]
fn a_pairs_retry_leaves_every_rank_of_the_engines_it_tried() {
    use std::collections::BTreeSet;

    // On each side two ranks of one engine, then two of another.
    let (prefills, prefill_engine) = ranked("prefill", WorkerType::Prefill, &[2, 2]);
    let (decodes, decode_engine) = ranked("decode", WorkerType::Decode, &[2, 2]);
    let classes = every(prefills.len(), decodes.len() as u32);
    let tried_prefill = worker_id(prefills[1].as_ref());
    let tried_decode = worker_id(decodes[0].as_ref());
    let host = Host::over(hearing(), prefills, decodes);
    // The engines a retry's targets stand on, a side each, over the
    // turns of both lists.
    let landed = |tried: &[WorkerId]| {
        let (mut on_prefills, mut on_decodes) = (BTreeSet::new(), BTreeSet::new());
        for seed in 0..32 {
            let mut asks = host.asks(&classes, false);
            asks.0.facts.seed = seed;
            asks.0.facts.attempt = 1;
            asks.0.facts.tried = tried;
            let outcome = leased(&host, &asks, Instant::now());
            for target in named(&outcome.decision) {
                let pair = target.pair.expect("a pair");
                on_prefills.insert(prefill_engine[target.position as usize]);
                on_decodes.insert(decode_engine[pair.decode_position as usize]);
            }
        }
        (on_prefills, on_decodes)
    };
    let (both, second) = (BTreeSet::from([0, 1]), BTreeSet::from([1]));
    assert_eq!(landed(&[]), (both.clone(), both.clone()), "nobody tried");
    assert_eq!(
        landed(&[tried_prefill]),
        (second.clone(), both.clone()),
        "the tried prefill's engine steps aside, and no decode does"
    );
    assert_eq!(
        landed(&[tried_decode]),
        (both, second.clone()),
        "the tried decode's engine steps aside, and no prefill does"
    );
    assert_eq!(
        landed(&[tried_prefill, tried_decode]),
        (second.clone(), second),
        "a tried pair moves on both sides"
    );

    // The list such a retry is decided by, in the order a stage reads it:
    // the tried workers as the host named them, the prefills' other ranks
    // behind them, and the decodes' behind those.
    let prefill = host.engine.pool_members(&host.prefills);
    let decode = host.engine.pool_members(&host.decodes);
    let tried = [tried_prefill, tried_decode];
    let whole = tried_engines(host.engine.loaded(), &tried, &[&prefill, &decode]);
    assert_eq!(
        whole.expect("each list holds ranks"),
        [
            tried_prefill,
            tried_decode,
            worker_id(host.prefills[0].as_ref()),
            worker_id(host.decodes[1].as_ref()),
        ]
    );
}
