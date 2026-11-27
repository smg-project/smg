use super::*;

/// The assignment's path is the single plan's alone: under `epd` an
/// encode item of a `manual` profile is not placed by it, so the picker
/// is asked and declines by its name, as on a side of a pair.
#[test]
fn the_manual_picker_declines_an_encode_item_under_the_epd_plan() {
    let engine = Engine::new(
        vec![strict_profile(), manual_profile()],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::Epd { encode: 1 },
            profile: 0,
            decode_profile: Some(0),
            shadow: None,
            decode_shadow: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    );
    let mut scratch = Scratch::new();
    match engine.decide(
        &encode_facts(1),
        &PoolFacts::default(),
        &[encode_candidate(0, true, true)],
        &Constraints::default(),
        &mut scratch,
    ) {
        Decision::Shed(shed) => {
            assert_eq!(shed.reason, ShedReason::PolicyDeclined);
            assert_eq!(shed.emptied_by, "manual");
        }
        other => panic!("a manual encode profile answered {other:?}"),
    }
}

/// Profile 0 is the strict pair side, profile 1 the encode profile.
fn epd_engine() -> Engine {
    Engine::new(
        vec![strict_profile(), least_requests_profile()],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::Epd { encode: 1 },
            profile: 0,
            decode_profile: Some(0),
            shadow: None,
            decode_shadow: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    )
}

/// A single or an encode request under the pair plan, and a single one
/// under `epd`, resolve to no profile and are declined by the binding,
/// as a pair decision under a single binding is; the pair phases and
/// `epd`'s encode phase resolve to their sides.
#[test]
fn a_single_or_encode_request_under_the_pair_plan_is_declined_by_the_binding() {
    let pair = Engine::new(
        vec![strict_profile()],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    for phase in [Phase::Single, Phase::Encode] {
        assert!(pair.profile_for("m", phase).is_none(), "{phase:?}");
        let facts = RequestFacts {
            model: "m",
            phase,
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        match pair.decide(
            &facts,
            &PoolFacts::default(),
            &[candidate(0, 0)],
            &Constraints::default(),
            &mut scratch,
        ) {
            Decision::Shed(shed) => {
                assert_eq!(shed.reason, ShedReason::PolicyDeclined, "{phase:?}");
                assert_eq!(shed.emptied_by, "binding", "{phase:?}");
            }
            other => panic!("{phase:?} under the pair plan answered {other:?}"),
        }
    }
    assert!(pair.profile_for("m", Phase::Prefill).is_some());
    assert!(pair.profile_for("m", Phase::Decode).is_some());
    let epd = epd_engine();
    assert!(epd.profile_for("m", Phase::Single).is_none());
    let side = |phase: Phase| {
        epd.profile_for("m", phase)
            .map(|(i, p)| (i, p.name.clone()))
    };
    assert_eq!(side(Phase::Encode), Some((1, "least".to_string())));
    assert_eq!(side(Phase::Prefill), Some((0, "strict".to_string())));
    assert_eq!(side(Phase::Decode), Some((0, "strict".to_string())));
}

/// The `epd` plan reads the rows' compatibility and vision bits before
/// the encode profile's own stages, and a pool they empty names them; a
/// single-plan binding never reads the bits.
#[test]
fn the_epd_plan_filters_encoders_by_ec_compatibility_and_vision_and_names_the_stage() {
    let epd = epd_engine();
    let decide = |facts: &RequestFacts<'_>, pool: &[Candidate]| {
        let mut scratch = Scratch::new();
        epd.decide(
            facts,
            &PoolFacts::default(),
            pool,
            &Constraints::default(),
            &mut scratch,
        )
    };
    let pool = [
        encode_candidate(0, false, true),
        encode_candidate(1, true, true),
        encode_candidate(2, true, false),
    ];
    let text = decide(&encode_facts(0), &pool);
    let pick = text.pick().expect("a pick");
    assert!(pick.targets.iter().all(|t| t.position != 0), "{pick:?}");
    assert_eq!(
        pick.trace.survivors.get(),
        Some(2),
        "the incompatible encoder is out"
    );
    let image = decide(&encode_facts(1), &pool);
    let pick = image.pick().expect("a pick");
    let order: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
    assert_eq!(order, [1], "an image takes the vision encoder alone");
    let shed_of = |facts: &RequestFacts<'_>, pool: &[Candidate]| match decide(facts, pool) {
        Decision::Shed(shed) => (shed.emptied_by, shed.reason),
        other => panic!("an emptied pool answered {other:?}"),
    };
    assert_eq!(
        shed_of(&encode_facts(0), &[encode_candidate(0, false, true)]),
        ("ec-compatible", ShedReason::NoEndpoints)
    );
    assert_eq!(
        shed_of(&encode_facts(1), &[encode_candidate(0, true, false)]),
        ("vision", ShedReason::NoEndpoints)
    );
    assert!(
        decide(&encode_facts(0), &[encode_candidate(0, true, false)])
            .pick()
            .is_some(),
        "a request without images takes an encoder without a vision encoder"
    );
    let mut sick = encode_candidate(0, true, true);
    sick.health.healthy = false;
    assert_eq!(
        shed_of(&encode_facts(0), &[sick, encode_candidate(1, false, true)]).0,
        "healthy",
        "the profile's stage is named after the plan's"
    );
    let single = engine();
    let mut scratch = Scratch::new();
    let regular = [encode_candidate(0, false, false)];
    assert!(
        single
            .decide(
                &encode_facts(1),
                &PoolFacts::default(),
                &regular,
                &Constraints::default(),
                &mut scratch,
            )
            .pick()
            .is_some(),
        "a single-plan binding reads no compatibility bit"
    );
}

/// A video asks vision as an image does: a request of the encode phase
/// whose media is video alone takes the vision encoder alone, and a
/// pool with none is shed under the rule's stage. Audio alone asks
/// none. The classic router asks vision for any pixel item, an image
/// or a video (`is_pixel_modality` and `request_needs_vision`,
/// `model_gateway/src/routers/grpc/multimodal/capability.rs:325-344`
/// on main d6517a4fe).
#[test]
fn a_video_alone_takes_the_vision_encoder_under_the_epd_plan() {
    let epd = epd_engine();
    let decide = |media: MediaCounts, pool: &[Candidate]| {
        let facts = RequestFacts {
            media,
            ..encode_facts(0)
        };
        let cx = Constraints::default();
        epd.decide(
            &facts,
            &PoolFacts::default(),
            pool,
            &cx,
            &mut Scratch::new(),
        )
    };
    let video = MediaCounts {
        video: 1,
        ..MediaCounts::default()
    };
    let audio = MediaCounts {
        audio: 1,
        ..MediaCounts::default()
    };
    let pool = [
        encode_candidate(0, true, false),
        encode_candidate(1, true, true),
    ];
    let seen = decide(video, &pool);
    let pick = seen.pick().expect("a pick");
    let order: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
    assert_eq!(order, [1], "a video takes the vision encoder alone");
    match decide(video, &pool[..1]) {
        Decision::Shed(shed) => assert_eq!(
            (shed.emptied_by, shed.reason),
            ("vision", ShedReason::NoEndpoints)
        ),
        other => panic!("a pool with no vision encoder answered {other:?}"),
    }
    assert!(
        decide(audio, &pool[..1]).pick().is_some(),
        "audio alone takes an encoder without a vision encoder"
    );
}

#[test]
fn the_epd_binding_serves_its_pair_half_through_decide_pair() {
    let epd = epd_engine();
    let mut prefills = [candidate(0, 0)];
    prefills[0].pair_class = 7;
    let mut decodes = [candidate(0, 0)];
    decodes[0].pair_class = 7;
    let decision = decide_pair(&epd, &prefills, &decodes);
    let pick = decision
        .pick()
        .expect("the epd binding's pair half decides");
    assert_eq!(
        pick.targets
            .first()
            .and_then(|t| t.pair)
            .map(|p| p.decode_position),
        Some(0)
    );
}
