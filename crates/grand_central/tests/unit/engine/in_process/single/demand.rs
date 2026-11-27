//! The cases of a demand a request states: the workers that meet it, over a
//! list long enough for the passes that draw and under a pinned key, and the
//! shed of a request whose demand no worker meets.

use super::*;

/// The places sixteen decisions of one request that states `demands`
/// name over `members`, each as a place or as a whole decision's first
/// target, or what the request is shed for.
fn places_named(
    engine: &InProcess,
    members: &PublishedList,
    demands: Demands,
) -> Result<BTreeSet<u32>, (ShedReason, &'static str)> {
    places_named_among(engine, members, demands, None)
}

/// [`places_named`], for a request its host holds to `among`.
fn places_named_among(
    engine: &InProcess,
    members: &PublishedList,
    demands: Demands,
    among: Option<&[u32]>,
) -> Result<BTreeSet<u32>, (ShedReason, &'static str)> {
    let mut stated = request();
    stated.constraints.demands = demands;
    stated.constraints.places = among;
    let asks = Asks::of(stated, Some(members));
    let mut places = BTreeSet::new();
    for _ in 0..16 {
        let place = match answered(engine, &asks) {
            Decided::Alone(pick) => pick.place,
            Decided::Whole(decided, _) => match decided.decision {
                Decision::Pick(pick) => pick.targets.first().expect("a target").position,
                Decision::Shed(shed) => return Err((shed.reason, shed.emptied_by)),
                Decision::Queue(_) => panic!("a queue ticket"),
            },
        };
        places.insert(place);
    }
    Ok(places)
}

/// A request that states a hard demand is decided among the workers of
/// its list that meet it, by what the list's host said of each worker
/// when it published the list:
/// every pick lands on one of them, and each of them takes its turn. A
/// demand no worker meets sheds the list with no endpoint, under that
/// demand's stage. Of a list its host said nothing of, no worker takes
/// images and each holds a request of any size.
#[test]
fn a_request_that_states_a_demand_is_decided_among_the_workers_that_meet_it() {
    let engine = preset("round_robin");
    let list = pool(4);
    // The second and the fourth worker take images; the fourth states
    // a window, each of the others a runtime.
    let (sees, bounded) = (Arc::clone(&list[1]), Arc::clone(&list[3]));
    let said = PublishedList::with_facts(Arc::clone(&list), |worker| {
        let bounded = Arc::ptr_eq(worker, &bounded);
        // Each fact behind the rest of the type: a further fact of a
        // worker adds its field.
        let seeing = FixedFacts {
            vision: bounded || Arc::ptr_eq(worker, &sees),
            ..FixedFacts::default()
        };
        FixedFacts {
            demands: DemandFacts {
                window: if bounded { 4_096 } else { 0 },
                runtime: if bounded { 0 } else { 7 },
                ..DemandFacts::default()
            },
            ..seeing
        }
    });
    let silent = PublishedList::new(Arc::clone(&list));
    let asked = |members, demands| places_named(&engine, members, demands);
    let pixels = Demands {
        vision: true,
        ..Demands::default()
    };
    let long = Demands {
        input_window: 5_000,
        ..Demands::default()
    };
    assert_eq!(asked(&said, pixels), Ok(BTreeSet::from([1, 3])));
    assert_eq!(asked(&said, long), Ok(BTreeSet::from([0, 1, 2])));
    let both = Demands {
        vision: true,
        ..long
    };
    assert_eq!(asked(&said, both), Ok(BTreeSet::from([1])));
    let stays = Demands {
        runtime: 7,
        ..Demands::default()
    };
    assert_eq!(asked(&said, stays), Ok(BTreeSet::from([0, 1, 2])));
    // Held to places by its host too: the places that meet the demand,
    // every place where each meets it, and the demand's stage where
    // none does. The classic router's tests of a retry are one test of
    // a candidate too
    // (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:779-782`
    // at main `f55a0a6bf`).
    let among = |places| places_named_among(&engine, &said, pixels, Some(places));
    assert_eq!(among(&[0, 1, 2]), Ok(BTreeSet::from([1])));
    assert_eq!(among(&[1, 3]), Ok(BTreeSet::from([1, 3])));
    let blind = Err((ShedReason::NoEndpoints, "vision"));
    assert_eq!(among(&[0, 2]), blind);
    // Nobody meets both: the second worker alone takes images and is
    // too small, and the others take none. The list is shed under the
    // later of the two demands in a host's order, which is neither
    // the first worker's failure nor the last one's.
    let narrow = PublishedList::with_facts(Arc::clone(&list), |worker| {
        let small = Arc::ptr_eq(worker, &sees);
        let seeing = FixedFacts {
            vision: small,
            ..FixedFacts::default()
        };
        FixedFacts {
            demands: DemandFacts {
                window: if small { 4_096 } else { 0 },
                ..DemandFacts::default()
            },
            ..seeing
        }
    });
    let too_small = Err((ShedReason::NoEndpoints, "context-window"));
    assert_eq!(asked(&narrow, both), too_small);
    // Nothing said of the list: no worker takes images. The classic
    // router reads a worker with no label as taking them
    // (`model_gateway/src/routers/grpc/multimodal/capability.rs:98-105`
    // at main `c22acc3b9`): a host that follows it says so of each such
    // worker when it makes the list's value.
    assert_eq!(asked(&silent, long), Ok(BTreeSet::from([0, 1, 2, 3])));
    assert_eq!(asked(&silent, pixels), blind);
}

/// Over a list long enough for the passes that draw, a request that
/// states a demand is answered a worker that meets it at every
/// decision, whatever the picker: those passes step aside for a request
/// held to places.
#[test]
fn a_demand_holds_under_a_pick_that_draws_over_a_long_list() {
    for name in [
        "random",
        "power_of_two",
        "round_robin",
        "consistent_hashing",
    ] {
        let engine = preset(name);
        let list = pool(32);
        // Every fourth worker takes images.
        let sees: Vec<_> = list.iter().step_by(4).cloned().collect();
        let members = PublishedList::with_facts(Arc::clone(&list), |worker| FixedFacts {
            vision: sees.iter().any(|seeing| Arc::ptr_eq(seeing, worker)),
            ..FixedFacts::default()
        });
        let (mut landed, mut firsts) = (BTreeSet::new(), BTreeSet::new());
        for seed in 0..64 {
            let mut stated = request();
            stated.facts.seed = seed;
            // The ring places a request by its key.
            if name == "consistent_hashing" {
                stated.facts.session_key = Identity::hashed(seed + 1);
            }
            stated.constraints.demands = Demands {
                vision: true,
                ..Demands::default()
            };
            let asks = Asks::of(stated, Some(&members));
            let outcome = decided(&engine, &asks);
            let pick = outcome.decision.pick().expect("a pick");
            firsts.insert(pick.targets.first().expect("a target").position);
            let places = pick.targets.iter().map(|target| target.position);
            landed.extend(places);
        }
        // Every target takes images, the alternates too, and the first
        // targets are more than one worker.
        let seeing = landed.iter().all(|place| place % 4 == 0);
        assert!(seeing, "{name}: {landed:?} take images");
        assert!(firsts.len() > 1, "{name}: {firsts:?}");
    }
}

/// Where a binding of the request's phase pins a key, a request that
/// states a hard demand is decided among the workers that meet it, as
/// a request held to places is: a pin on a worker that meets the
/// demand is followed, and a pin on one that does not is not. That is
/// the gateway's classic answer: it narrows its candidates by the
/// request's own test before its policy runs
/// (`model_gateway/src/routers/common/placement.rs:508-509` at main
/// `f55a0a6bf`), and its sticky policy looks a key's pinned workers up
/// among the workers it is handed and picks another of them where none
/// is there (`model_gateway/src/policies/manual.rs:303-308`).
#[test]
fn a_demand_under_a_pinned_key_is_decided_among_the_workers_that_meet_it() {
    // The stock sessions rule: its binding pins a key.
    let engine = preset("manual");
    let list = pool(8);
    // Every other worker takes images.
    let sees: Vec<_> = list.iter().step_by(2).cloned().collect();
    let members = PublishedList::with_facts(Arc::clone(&list), |worker| FixedFacts {
        vision: sees.iter().any(|seeing| Arc::ptr_eq(seeing, worker)),
        ..FixedFacts::default()
    });
    // One request of `key` that states `demands`, served by its first
    // target: that target's place.
    let sent = |key: u64, demands: Demands| {
        let mut keyed = request();
        keyed.facts.seed = key;
        keyed.facts.session_key = Identity::hashed(key);
        keyed.constraints.demands = demands;
        let asks = Asks::of(keyed, Some(&members));
        served(&engine, &asks, Instant::now()).position
    };
    let pixels = Demands {
        vision: true,
        ..Demands::default()
    };
    let (mut followed, mut left) = (0, 0);
    for key in 1..=16 {
        // The key's first request states nothing and is pinned.
        let pin = sent(key, Demands::default());
        assert_eq!(sent(key, Demands::default()), pin, "key {key}: its pin");
        let went = sent(key, pixels);
        assert_eq!(went % 2, 0, "key {key}: place {went} takes images");
        if pin % 2 == 0 {
            assert_eq!(went, pin, "key {key}: its pin meets the demand");
            followed += 1;
        } else {
            left += 1;
        }
    }
    assert!(
        followed > 0 && left > 0,
        "{followed} pins followed, {left} left"
    );
}

/// One worker states runtime 7 and the other states no runtime.
/// A request for runtime 9 is shed under `wires` with no endpoint.
/// The shed takes no lease ID and sends no issue event.
#[test]
fn a_runtime_demand_no_member_meets_is_shed_without_a_lease() {
    let (engine, heard) = hearing_as(false, true, EventSet::LEASE);
    let list = pool(2);
    let members = PublishedList::with_facts(Arc::clone(&list), |worker| FixedFacts {
        demands: DemandFacts {
            runtime: if Arc::ptr_eq(worker, &list[0]) { 7 } else { 0 },
            ..DemandFacts::default()
        },
        ..FixedFacts::default()
    });
    let mut asked = request();
    asked.constraints.demands.runtime = 9;

    let outcome = decided(&engine, &Asks::of(asked, Some(&members)));
    let Decision::Shed(shed) = outcome.decision else {
        panic!("expected a shed for runtime 9, got {:?}", outcome.decision);
    };
    assert_eq!(
        (shed.reason, shed.emptied_by),
        (ShedReason::NoEndpoints, "wires")
    );
    assert!(outcome.lease.is_none());
    assert_eq!(
        engine.leases.0.load(Ordering::Relaxed),
        0,
        "no lease ID taken"
    );
    assert!(heard.lock().is_empty(), "no lease issue reaches the picker");
}

/// The workers state decoding values 5 and 0.
/// A request for 7 is shed under `decoding`.
/// A request for 5 picks the worker that states 5.
#[test]
fn a_constraint_no_member_decides_alike_is_shed_under_its_own_stage() {
    let engine = preset("random");
    let list = pool(2);
    let members = PublishedList::with_facts(Arc::clone(&list), |worker| FixedFacts {
        demands: DemandFacts {
            decoding: if Arc::ptr_eq(worker, &list[0]) { 5 } else { 0 },
            ..DemandFacts::default()
        },
        ..FixedFacts::default()
    });
    let mut asks = Asks::of(request(), Some(&members));
    asks.request.constraints.demands.decoding = 7;
    let missed = decided(&engine, &asks).decision;
    let Decision::Shed(shed) = missed else {
        panic!("expected a shed for decoding 7, got {missed:?}");
    };
    assert_eq!(
        (shed.reason, shed.emptied_by),
        (ShedReason::NoEndpoints, "decoding")
    );

    asks.request.constraints.demands.decoding = 5;
    let matched = decided(&engine, &asks).decision;
    let pick = matched.pick().expect("a worker states decoding 5");
    let first = pick.targets.first().expect("a first target");
    assert_eq!(first.id, worker_id(list[0].as_ref()));
}

/// Vision and reference demands select the workers that meet them.
/// Each excluded worker fails at its first unmet demand.
/// The latest of those demands names the shed when the pool empties.
#[test]
fn a_media_demand_keeps_the_members_that_take_it_and_names_its_shed() {
    use crate::engine::References;

    let engine = preset("random");
    let members_with = |states: &[(bool, bool, bool)]| {
        let list = pool(states.len());
        let mut states = states.iter().copied();
        PublishedList::with_facts(Arc::clone(&list), |_| {
            let (vision, references, host_pipeline) =
                states.next().expect("media facts for each worker");
            FixedFacts {
                vision,
                demands: DemandFacts {
                    runtime: 7,
                    references,
                    host_pipeline,
                    ..DemandFacts::default()
                },
                ..FixedFacts::default()
            }
        })
    };
    let mixed = members_with(&[
        (true, false, false),
        (true, true, true),
        (false, true, false),
    ]);
    let id = |at: usize| worker_id(mixed.workers()[at].as_ref());
    let (plain, caller, own) = (id(0), id(1), id(2));
    let media = |vision, references| Demands {
        vision,
        references,
        ..Demands::default()
    };

    for (vision, references, expected) in [
        (true, References::None, BTreeSet::from([plain, caller])),
        (false, References::Any, BTreeSet::from([caller, own])),
        (false, References::Caller, BTreeSet::from([caller])),
        (false, References::Worker, BTreeSet::from([own])),
    ] {
        let mut asks = Asks::of(request(), Some(&mixed));
        asks.request.constraints.demands = media(vision, references);
        let mut firsts = BTreeSet::new();
        for seed in 0..48_u64 {
            asks.request.facts.seed = seed;
            let outcome = decided(&engine, &asks);
            let pick = outcome.decision.pick().expect("a matching worker");
            firsts.insert(pick.targets.first().expect("a first target").id);
        }
        assert_eq!(
            firsts, expected,
            "vision {vision}, references {references:?}"
        );
    }

    let other_runtime = Demands {
        runtime: 9,
        ..media(true, References::Caller)
    };
    let blind = members_with(&[(false, true, true), (false, false, false)]);
    for (members, demands, stage) in [
        (&mixed, media(true, References::Worker), "pipeline"),
        (&mixed, other_runtime, "wires"),
        (&blind, media(true, References::Any), "vision"),
    ] {
        assert_eq!(
            places_named(&engine, members, demands),
            Err((ShedReason::NoEndpoints, stage)),
            "{demands:?}"
        );
    }
}
