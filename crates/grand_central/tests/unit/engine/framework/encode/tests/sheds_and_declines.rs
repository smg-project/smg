use super::*;

/// The position of the encode profile among `engine`'s profiles.
fn encode_profile(engine: &Engine) -> Option<u16> {
    let mut names = engine
        .profiles()
        .iter()
        .map(|profile| profile.name.as_str());
    u16::try_from(names.position(|name| name == "encode")?).ok()
}

/// An encode decision says, in the scratch it is made with, the list it
/// sheds its request over as saturated: the encoders where the overload
/// veto left none of them open, and else the list its pair's decision
/// says, where that decision was handed the pair's two lists whole. Where
/// the pass handed it a part of either list, the decision says no list,
/// of the list that was cut or of the other: the pair's decision judged
/// the part. Each decision takes back the word of the one before where it
/// begins, so an answer that is no such shed leaves none.
#[test]
fn an_encode_decision_says_the_list_it_sheds_over() {
    let mut shedding = engine();
    shedding.shed_when_all_vetoed(0);
    let saturated = (ShedReason::Saturated, "not-overloaded");
    let mut scratch = Scratch::new();
    // The one encoder is over its thresholds.
    let mut bed = Bed::new();
    bed.encoders[0] = overloaded(bed.encoders[0]);
    assert_eq!(shed_of(&bed.decide_in(&shedding, &mut scratch)), saturated);
    assert_eq!(scratch.shed_over(), Some(ShedOver::Encoders));
    // No encoder is ready: a missing endpoint, which says no list.
    bed.encoders[0] = unready(bed.encoders[0]);
    let missing = (ShedReason::NoEndpoints, "encoders");
    assert_eq!(shed_of(&bed.decide_in(&shedding, &mut scratch)), missing);
    assert_eq!(scratch.shed_over(), None, "behind a missing endpoint");
    // The one prefill is over its thresholds: the pair's decision is
    // handed both lists whole, and says the prefills.
    let mut bed = Bed::new();
    bed.prefills[0] = overloaded(bed.prefills[0]);
    assert_eq!(shed_of(&bed.decide_in(&shedding, &mut scratch)), saturated);
    assert_eq!(scratch.shed_over(), Some(ShedOver::Prefills));
    // A second prefill under its thresholds hands to no encoder, so the
    // pair's decision is handed the first alone: its word of the prefills
    // says nothing of the list the caller lent, and goes.
    bed.prefills.push(row(1, Role::Prefill, 1, 0));
    bed.prefill_classes.push(2);
    assert_eq!(shed_of(&bed.decide_in(&shedding, &mut scratch)), saturated);
    assert_eq!(scratch.shed_over(), None, "a part of the prefills");
    // A pick says no list either.
    bed.prefills[0] = row(0, Role::Prefill, 1, 0);
    let decision = bed.decide_in(&shedding, &mut scratch);
    assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
    assert_eq!(scratch.shed_over(), None, "behind a pick");
    // The one decode is over its thresholds behind whole lists: the
    // pair's decision says the decodes.
    let mut bed = Bed::new();
    bed.decodes[0] = overloaded(bed.decodes[0]);
    assert_eq!(shed_of(&bed.decide_in(&shedding, &mut scratch)), saturated);
    assert_eq!(scratch.shed_over(), Some(ShedOver::Decodes));
    // Behind a part of the prefills that word goes too: the decision made
    // its shed for the prefills it was handed.
    bed.prefills.push(row(1, Role::Prefill, 1, 0));
    bed.prefill_classes.push(2);
    assert_eq!(shed_of(&bed.decide_in(&shedding, &mut scratch)), saturated);
    assert_eq!(scratch.shed_over(), None, "the prefills cut, the decodes");
    // The request's decode must take images, and a second decode under
    // its thresholds takes none: a part of the decodes, behind a shed over
    // the decodes and behind one over the prefills.
    let mut bed = Bed::new();
    bed.decode_images = true;
    bed.decodes[0] = overloaded(bed.decodes[0]);
    bed.decodes.push(Candidate {
        vision: false,
        ..row(1, Role::Decode, 1, 0)
    });
    assert_eq!(shed_of(&bed.decide_in(&shedding, &mut scratch)), saturated);
    assert_eq!(scratch.shed_over(), None, "the decodes cut, the decodes");
    bed.decodes[0] = row(0, Role::Decode, 1, 0);
    bed.prefills[0] = overloaded(bed.prefills[0]);
    assert_eq!(shed_of(&bed.decide_in(&shedding, &mut scratch)), saturated);
    assert_eq!(scratch.shed_over(), None, "the decodes cut, the prefills");
}

/// Under a binding that sheds a pool the overload veto empties, encoders
/// the veto left none of are a saturated fleet: the shed names the veto's
/// stage and its reason is saturation, as a pair's side sheds under that
/// binding. A stage ahead of the veto that leaves none is a missing endpoint
/// as before, and so is one behind it that refuses an encoder the veto
/// kept. An encoder that does not take the request is not counted, and
/// under a veto its host lifted the encoders serve.
#[test]
fn encoders_the_veto_left_none_of_shed_as_saturation_under_a_binding_that_sheds() {
    let mut shedding = engine();
    shedding.shed_when_all_vetoed(0);
    let saturated = (ShedReason::Saturated, "not-overloaded");
    let missing = (ShedReason::NoEndpoints, "encoders");
    // Every encoder that takes the request is over its thresholds.
    let mut bed = Bed::new();
    bed.encoders = vec![
        overloaded(row(0, Role::Encode, 0, 0)),
        overloaded(row(1, Role::Encode, 0, 0)),
    ];
    bed.facts = vec![pushing(1), pushing(1)];
    let shed = bed.decide_with(&shedding).0;
    assert_eq!(shed_of(&shed), saturated);
    // The veto is a stage of the encode profile, and the trace says so.
    assert_eq!(shed.trace().made_under(), encode_profile(&shedding));
    assert!(shed.trace().made_under().is_some());
    // A pair's decode side says the same of its own rows under that
    // binding.
    let mut side = Bed::new();
    side.decodes = vec![overloaded(row(0, Role::Decode, 1, 0))];
    assert_eq!(shed_of(&side.pair(&shedding)), saturated);
    // The host lifted the veto on the encode side: the item is placed.
    let (facts, _) = bed.asked();
    let cx = Constraints::default();
    let lifted = Constraints {
        overload_lifted: true,
        ..Constraints::default()
    };
    let sides = [&cx, &cx, &lifted];
    let (decision, placed) = decide_under(&bed, &shedding, &facts, bed.pools(), sides);
    assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
    assert_eq!(placed.encoders().len(), 1);

    // An unready encoder beside one over its thresholds was refused ahead
    // of the veto: the veto still left none.
    bed.encoders[1] = unready(row(1, Role::Encode, 0, 0));
    assert_eq!(shed_of(&bed.decide_with(&shedding).0), saturated);
    // No encoder reached the veto: a missing endpoint.
    bed.encoders[0] = unready(row(0, Role::Encode, 0, 0));
    assert_eq!(shed_of(&bed.decide_with(&shedding).0), missing);

    // An encoder the veto kept, refused by a hard filter listed behind it:
    // a missing endpoint, as a pair's side names rows so filtered.
    bed.encoders = vec![
        overloaded(row(0, Role::Encode, 0, 0)),
        row(1, Role::Encode, 0, 0),
    ];
    let mut behind = engine();
    behind.shed_when_all_vetoed(0);
    for profile in behind.profiles.iter_mut().map(own) {
        profile
            .filters
            .push(Box::new(RefusesOne(bed.encoders[1].id)));
    }
    assert_eq!(shed_of(&bed.decide_with(&behind).0), missing);

    // An encoder over its thresholds that takes no image is not counted:
    // the one that takes it was refused ahead of the veto.
    bed.encoders = vec![
        unready(row(0, Role::Encode, 0, 0)),
        Candidate {
            vision: false,
            ..overloaded(row(1, Role::Encode, 0, 0))
        },
    ];
    assert_eq!(shed_of(&bed.decide_with(&shedding).0), missing);
    // A video alone asks the same of an encoder as an image does.
    (bed.images, bed.video) = (0, 1);
    assert_eq!(shed_of(&bed.decide_with(&shedding).0), missing);
    (bed.images, bed.video) = (1, 0);
    // Nor is one over its thresholds that states no end.
    bed.encoders[1] = overloaded(row(1, Role::Encode, 0, 0));
    bed.facts[1] = pushing(0);
    assert_eq!(shed_of(&bed.decide_with(&shedding).0), missing);
    // Nor is one over its thresholds dialed on no wire the request states.
    bed.encoders = vec![
        unready(dialed_on(WireModes::HTTP, 0, 0)),
        overloaded(dialed_on(WireModes::GRPC, 1, 0)),
    ];
    bed.facts[1] = pushing(1);
    bed.wires = WireModes::HTTP;
    assert_eq!(shed_of(&bed.decide_with(&shedding).0), missing);
    // With no wire stated the veto left none.
    bed.wires = WireModes::NONE;
    assert_eq!(shed_of(&bed.decide_with(&shedding).0), saturated);
    // Nor is one over its thresholds that states another runtime than the
    // request stays on.
    let on = |runtime: u32| DemandFacts {
        runtime,
        ..DemandFacts::default()
    };
    bed.demands = Demands {
        runtime: 7,
        ..Demands::default()
    };
    bed.stated = vec![on(7), on(8)];
    assert_eq!(shed_of(&bed.decide_with(&shedding).0), missing);
    // Both on the request's runtime: the veto left none.
    bed.stated = vec![on(7), on(7)];
    assert_eq!(shed_of(&bed.decide_with(&shedding).0), saturated);
}

/// Under a binding that steers, encoders the overload veto left none of are
/// a missing endpoint under `encoders`: no encoder enters the all-vetoed
/// rule, so none is steered to, and the request is answered as one whose
/// encoders are not ready. The shed is made under the encode profile all
/// the same. A pair's decode side picks among its rows under that binding,
/// and under a veto its host lifted the encoders serve.
#[test]
fn encoders_the_veto_left_none_of_are_a_missing_endpoint_under_a_binding_that_steers() {
    let stock = engine();
    let missing = (ShedReason::NoEndpoints, "encoders");
    // Every encoder that takes the request is over its thresholds.
    let mut bed = Bed::new();
    bed.encoders = vec![
        overloaded(row(0, Role::Encode, 0, 0)),
        overloaded(row(1, Role::Encode, 0, 0)),
    ];
    bed.facts = vec![pushing(1), pushing(1)];
    let shed = bed.decide().0;
    assert_eq!(shed_of(&shed), missing);
    assert_eq!(shed.trace().made_under(), encode_profile(&stock));
    assert!(shed.trace().made_under().is_some());
    // A pair's decode side over its thresholds is picked among under that
    // binding: the rule steers there, and it is asked of no encoder.
    let mut side = Bed::new();
    side.decodes = vec![overloaded(row(0, Role::Decode, 1, 0))];
    let pair = side.pair(&stock);
    assert!(matches!(pair, Decision::Pick(_)), "{pair:?}");
    // The host lifted the veto on the encode side: the item is placed.
    let (facts, _) = bed.asked();
    let cx = Constraints::default();
    let lifted = Constraints {
        overload_lifted: true,
        ..Constraints::default()
    };
    let sides = [&cx, &cx, &lifted];
    let (decision, placed) = decide_under(&bed, &stock, &facts, bed.pools(), sides);
    assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
    assert_eq!(placed.encoders().len(), 1);
    // An unready encoder beside one over its thresholds reads the same.
    bed.encoders[1] = unready(row(1, Role::Encode, 0, 0));
    assert_eq!(shed_of(&bed.decide().0), missing);
}

#[test]
fn a_shed_of_the_pair_is_returned_as_the_pair_gave_it() {
    let mut bed = Bed::new();
    bed.prefills = vec![unready(row(0, Role::Prefill, 1, 0))];
    assert_eq!(bed.stage(), "healthy");
    // A class with no open encoder is no candidate of the pair: beside an
    // open encoder of another class the prefill pool is empty.
    let mut bed = Bed::new();
    bed.encoders = vec![
        unready(row(0, Role::Encode, 0, 0)),
        row(1, Role::Encode, 0, 0),
    ];
    bed.facts = vec![pushing(1), pushing(2)];
    assert_eq!(bed.stage(), "none");
}

#[test]
fn a_column_of_another_length_and_a_binding_of_another_plan_decline() {
    // A shed made once the encode profile is resolved says that profile.
    // One made where no binding gives a profile says none.
    let declined = |bed: &Bed, engine: &Engine| match bed.decide_with(engine).0 {
        Decision::Shed(shed) => (shed.reason, shed.emptied_by, shed.trace.made_under()),
        other => panic!("a shed, not {other:?}"),
    };
    let encode = encode_profile(&engine());
    assert!(encode.is_some(), "the plan has an encode profile");
    let column = (ShedReason::PolicyDeclined, "encode-column", encode);
    let mut bed = Bed::new();
    bed.facts.clear();
    assert_eq!(declined(&bed, &engine()), column);
    let mut bed = Bed::new();
    bed.prefill_classes.push(1);
    assert_eq!(declined(&bed, &engine()), column);
    let mut bed = Bed::new();
    bed.encode_affinity = vec![AffinityFacts::default(); 3];
    assert_eq!(declined(&bed, &engine()), column);

    let pair = PolicySpec::pair_presets("power_of_two", "power_of_two")
        .expect("the presets")
        .validate()
        .expect("the form validates");
    assert_eq!(
        declined(&Bed::new(), &pair),
        (ShedReason::PolicyDeclined, "binding", None)
    );
}

/// An encode request names no worker: a decision that states a number on
/// its decode side ([`Constraints::target_worker`]) is declined under
/// `target-worker`, beside places of that side and beside none, with an
/// item and with none. The pair entry counts such a number.
#[test]
fn a_number_on_the_decode_side_declines_an_encode_request() {
    let (bed, engine) = (Bed::new(), engine());
    let (facts, pool) = bed.asked();
    let shed = |keys: &[Identity<'static>], decode_cx: &Constraints<'_>| {
        let encode = EncodePool {
            encoders: &bed.encoders,
            facts: &bed.facts,
            prefill_classes: &bed.prefill_classes,
            affinity: &bed.encode_affinity,
            ..EncodePool::default()
        };
        let items = EncodeItems {
            keys,
            one: None,
            references: false,
            decode_images: false,
            ring: None,
        };
        let cx = Constraints::default();
        let decision = engine.decide_encode_pair(
            &facts,
            &pool,
            bed.pools(),
            encode,
            items,
            &cx,
            decode_cx,
            &cx,
            &mut Scratch::new(),
            &mut EncodePlaced::new(),
        );
        match decision {
            Decision::Shed(shed) => Some((shed.reason, shed.emptied_by)),
            Decision::Pick(_) | Decision::Queue(_) => None,
        }
    };
    let numbered = |places| Constraints {
        places,
        target_worker: Some(0),
        ..Constraints::default()
    };
    let declined = Some((ShedReason::PolicyDeclined, "target-worker"));
    for keys in [&bed.keys[..], &[]] {
        assert_eq!(shed(keys, &numbered(Some(&[0]))), declined);
        assert_eq!(shed(keys, &numbered(None)), declined);
        assert_eq!(shed(keys, &Constraints::default()), None, "a pick");
    }
}

/// What the pair entry answers, this entry answers the same when it admits
/// every row: a pick with its alternates, a shed, and the overload veto,
/// with an item and without. No built-in stage issues a queue ticket yet,
/// so none is read here.
#[test]
fn the_pairs_own_answers_read_the_same_through_both_entries() {
    let engine = engine();
    let mut alternates = Bed::new();
    alternates.decodes = vec![row(0, Role::Decode, 1, 2), row(1, Role::Decode, 1, 1)];
    let mut shed = Bed::new();
    shed.prefills = vec![unready(row(0, Role::Prefill, 1, 0))];
    let mut vetoed = Bed::new();
    vetoed.decodes = vec![overloaded(row(0, Role::Decode, 1, 0))];
    // The beds are what they are named for.
    assert!(
        matches!(alternates.pair(&engine), Decision::Pick(pick) if pick.targets.len() == 2),
        "{:?}",
        alternates.pair(&engine)
    );
    assert!(
        matches!(shed.pair(&engine), Decision::Shed(shed) if shed.emptied_by == "healthy"),
        "{:?}",
        shed.pair(&engine)
    );
    assert!(
        matches!(vetoed.pair(&engine), Decision::Pick(pick) if pick.trace.lifted == LiftedVeto::Overload),
        "{:?}",
        vetoed.pair(&engine)
    );
    for (name, mut bed) in [
        ("alternates", alternates),
        ("a shed", shed),
        ("the veto", vetoed),
    ] {
        let pair = bed.pair(&engine);
        assert_eq!(bed.decide_with(&engine).0, pair, "{name}, with an item");
        bed.keys.clear();
        assert_eq!(bed.decide_with(&engine).0, pair, "{name}, without items");
    }
}

/// A stage that shows its rows to a plugin and counts each time it is
/// asked: a scope to open, a row to keep.
#[derive(Debug)]
struct ShowsItsRows(Arc<AtomicU32>);

impl framework::Lifecycle for ShowsItsRows {}

impl framework::DecisionScope for ShowsItsRows {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    fn close(&self) {}
}

impl framework::Filter for ShowsItsRows {
    fn name(&self) -> &'static str {
        "shows-its-rows"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        self.0.fetch_add(1, Ordering::Relaxed);
        true
    }

    fn scope(&self) -> Option<&dyn framework::DecisionScope> {
        Some(self)
    }
}

/// A request with items under an encode profile that shows its rows to a
/// plugin is declined before a stage of the profile is asked. Without items
/// the profile is not read, and the pair is decided.
#[test]
fn an_encode_profile_that_shows_its_rows_declines_a_request_with_items() {
    let asked = Arc::new(AtomicU32::new(0));
    let mut engine = engine();
    let encode = engine
        .binding("m", Phase::Prefill)
        .and_then(|binding| match binding.plan {
            Plan::Epd { encode } => Some(usize::from(encode)),
            Plan::Single | Plan::PdPair => None,
        })
        .expect("the epd plan");
    let shows = ShowsItsRows(Arc::clone(&asked));
    own(&mut engine.profiles[encode])
        .filters
        .push(Box::new(shows));
    engine.marks[encode] = framework::Marks::of(&engine.profiles[encode], engine.orders[encode]);

    let (decision, placed) = Bed::new().decide_with(&engine);
    assert_eq!(
        shed_of(&decision),
        (ShedReason::PolicyDeclined, "encode-scope")
    );
    assert!(placed.encoders().is_empty());
    let mut bare = Bed::new();
    bare.keys.clear();
    let (decision, _) = bare.decide_with(&engine);
    assert!(decision.pick().is_some(), "a pick, not {decision:?}");
    assert_eq!(asked.load(Ordering::Relaxed), 0, "no stage is asked");
}

/// The pair entry declines a mask beside the rows a side shows a plugin
/// (`written-rows`), and it reads the mask as this entry hands it: numbered
/// with the rows a cut leaves. A mask on a row the cut left out masks none
/// of them and declines nothing.
#[test]
fn a_mask_on_a_row_cut_out_declines_nothing_for_a_side_that_shows_its_rows() {
    let mut engine = engine();
    let binding = engine.binding("m", Phase::Prefill).expect("the binding");
    let prefill = usize::from(binding.profile);
    let shows = ShowsItsRows(Arc::new(AtomicU32::new(0)));
    own(&mut engine.profiles[prefill])
        .filters
        .push(Box::new(shows));
    engine.marks[prefill] = framework::Marks::of(&engine.profiles[prefill], engine.orders[prefill]);
    // The first prefill's class holds no encoder: the pair is decided over
    // the second alone.
    let mut bed = Bed::new();
    bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 0)];
    bed.prefill_classes = vec![2, 1];
    let (facts, _) = bed.asked();
    let cx = Constraints::default();
    let decide = |mask: &[u32]| {
        let masked = masking(mask);
        decide_under(&bed, &engine, &facts, bed.pools(), [&masked, &cx, &cx]).0
    };
    // The mask names the row the cut left out.
    assert_eq!(pair_of(&decide(&[0])), (1, 0));
    // It names the row admitted: a mask beside the rows shown.
    assert_eq!(
        shed_of(&decide(&[1])),
        (ShedReason::PolicyDeclined, "written-rows")
    );
}

/// A pair column of another length than its pool declines as the pair
/// entry declines it, where a cut would leave a column that fits.
#[test]
fn a_pair_column_of_another_length_declines_before_a_cut() {
    let engine = engine();
    // The second prefill's class has no encoder and the second decode takes
    // no image: each side is cut to its first row.
    let mut bed = Bed::new();
    bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 0)];
    bed.prefill_classes = vec![1, 2];
    bed.decodes = vec![
        row(0, Role::Decode, 1, 0),
        Candidate {
            vision: false,
            ..row(1, Role::Decode, 1, 0)
        },
    ];
    bed.decode_images = true;
    let (facts, _) = bed.asked();
    let cx = Constraints::default();
    let none: &[AffinityFacts] = &[];
    let short = [AffinityFacts::default(); 1];
    let long = [AffinityFacts::default(); 3];
    for (prefill_affinity, decode_affinity) in [
        (&short[..], none),
        (none, &short[..]),
        (&long[..], none),
        (none, &long[..]),
    ] {
        let pools = PairPools {
            prefill_affinity,
            decode_affinity,
            ..bed.pools()
        };
        let (decision, _) = decide_under(&bed, &engine, &facts, pools, [&cx; 3]);
        assert_eq!(
            shed_of(&decision),
            (ShedReason::PolicyDeclined, "affinity-column")
        );
        assert_eq!(decision.trace().made_under(), None, "as the pair entry");
    }
}

/// A hard filter that counts what it is asked and keeps every row.
#[derive(Debug)]
struct CountsItsAsks(Arc<AtomicU32>);

impl framework::Filter for CountsItsAsks {
    fn name(&self) -> &'static str {
        "counts-its-asks"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        self.0.fetch_add(1, Ordering::Relaxed);
        true
    }
}

/// A request that carries a column of its own, or names rows, declines as
/// the pair entry declines it, before a stage of the encode side is asked.
#[test]
fn a_request_with_a_column_or_named_rows_declines_before_a_stage_is_asked() {
    let asked = Arc::new(AtomicU32::new(0));
    let mut engine = engine();
    for profile in engine.profiles.iter_mut().map(own) {
        let counts = CountsItsAsks(Arc::clone(&asked));
        profile.filters.push(Box::new(counts));
    }
    let bed = Bed::new();
    let cx = Constraints::default();
    let holders = [crate::engine::AffinityHolder {
        row: 1,
        overlap_blocks: 8,
        predicted_blocks: 0,
    }];
    let named = RequestFacts {
        named: Some(crate::engine::AffinityRows {
            holders: &holders,
            ..crate::engine::AffinityRows::default()
        }),
        ..bed.asked().0
    };
    let column = [AffinityFacts::default()];
    let carried = RequestFacts {
        affinity: &column,
        ..bed.asked().0
    };
    for facts in [&named, &carried] {
        let (decision, placed) = decide_under(&bed, &engine, facts, bed.pools(), [&cx; 3]);
        assert_eq!(
            shed_of(&decision),
            (ShedReason::PolicyDeclined, "affinity-column")
        );
        assert_eq!(decision.trace().made_under(), None, "as the pair entry");
        assert!(placed.encoders().is_empty());
        assert_eq!(asked.load(Ordering::Relaxed), 0, "no stage is asked");
    }
    // The same engine decides a request that carries neither, and asks.
    let (decision, _) = bed.decide_with(&engine);
    assert!(decision.pick().is_some(), "a pick, not {decision:?}");
    assert_ne!(asked.load(Ordering::Relaxed), 0);
}
