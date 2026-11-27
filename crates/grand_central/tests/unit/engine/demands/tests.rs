use super::{DemandFacts, Demands, References, Role, UnmetDemand};

/// A request that demands nothing is met by every worker, and one that
/// demands a runtime by the workers that registered its number alone: a
/// worker that states no runtime meets no demand for one.
#[test]
fn a_runtime_demand_is_met_by_the_same_number_alone() {
    let of = |runtime: u32| DemandFacts {
        runtime,
        ..DemandFacts::default()
    };
    let (first, other, none) = (of(7), of(8), DemandFacts::default());
    let nothing = Demands::default();
    assert!(nothing.is_none());
    assert!([first, other, none]
        .iter()
        .all(|facts| nothing.met_by(false, *facts)));
    let stays = Demands {
        runtime: 7,
        ..Demands::default()
    };
    assert!(!stays.is_none());
    assert_eq!(
        [first, other, none].map(|facts| stays.met_by(false, facts)),
        [true, false, false]
    );
}

/// Each media demand is a demand, and a worker is judged by the one it
/// fails first: its runtime, then vision, then media by reference, then
/// the pipeline that processes it. Vision is judged by the bit of the
/// worker's row, and the rest by what it states beside it.
#[test]
fn a_worker_is_judged_by_the_demand_it_fails_first() {
    let worker = |vision: bool, references: bool, host_pipeline: bool| {
        let facts = DemandFacts {
            runtime: 7,
            references,
            host_pipeline,
            ..DemandFacts::default()
        };
        (vision, facts)
    };
    let asks = |vision: bool, references: References| Demands {
        runtime: 7,
        vision,
        references,
        ..Demands::default()
    };
    for demands in [
        asks(true, References::None),
        asks(false, References::Any),
        asks(false, References::Caller),
        asks(false, References::Worker),
    ] {
        assert!(!demands.is_none(), "{demands:?}");
    }
    let (plain, host, own) = (
        worker(true, false, false),
        worker(true, true, true),
        worker(false, true, false),
    );
    let table = [
        (asks(true, References::None), plain, None),
        (asks(true, References::None), own, Some(UnmetDemand::Vision)),
        (
            asks(false, References::Any),
            plain,
            Some(UnmetDemand::References),
        ),
        (asks(false, References::Any), host, None),
        (asks(false, References::Any), own, None),
        (asks(false, References::Caller), host, None),
        (
            asks(false, References::Caller),
            own,
            Some(UnmetDemand::Pipeline),
        ),
        (
            asks(false, References::Worker),
            host,
            Some(UnmetDemand::Pipeline),
        ),
        (asks(false, References::Worker), own, None),
        (
            asks(false, References::Worker),
            plain,
            Some(UnmetDemand::References),
        ),
        // Vision is judged before media by reference.
        (asks(true, References::Any), own, Some(UnmetDemand::Vision)),
        (asks(true, References::Caller), host, None),
    ];
    for (demands, (vision, facts), unmet) in table {
        let judged = demands.unmet_by(vision, facts);
        assert_eq!(judged, unmet, "{demands:?} {vision} {facts:?}");
        assert_eq!(demands.met_by(vision, facts), unmet.is_none());
    }
    // The runtime is judged before any of them.
    let elsewhere = DemandFacts {
        runtime: 8,
        ..DemandFacts::default()
    };
    let every = asks(true, References::Caller);
    assert_eq!(every.unmet_by(true, elsewhere), Some(UnmetDemand::Runtime));
}

/// A window demand is two bounds, the input's and the completion's, and
/// a worker's window holds each alone: the two are not added. A worker
/// that states no window holds a request of any size. The window is
/// judged last, behind the pipeline of media by reference.
#[test]
fn a_window_holds_each_bound_alone_and_none_stated_holds_any_request() {
    let window = |window: u32| DemandFacts {
        window,
        ..DemandFacts::default()
    };
    let asks = |input_window: u32, output_window: u32| Demands {
        input_window,
        output_window,
        ..Demands::default()
    };
    assert!(asks(0, 0).is_none());
    assert!(!asks(1, 0).is_none() && !asks(0, 1).is_none());
    // The request's two bounds, the worker's window, whether it holds.
    let table = [
        ((100, 0), 100, true),
        ((101, 0), 100, false),
        ((0, 100), 100, true),
        ((0, 101), 100, false),
        ((100, 100), 100, true),
        ((60, 60), 100, true),
        ((150, 150), 200, true),
        ((150, 150), 100, false),
        ((u32::MAX, u32::MAX), 0, true),
    ];
    for ((input, output), stated, held) in table {
        let unmet = asks(input, output).unmet_by(false, window(stated));
        let judged = if held {
            None
        } else {
            Some(UnmetDemand::Window)
        };
        assert_eq!(unmet, judged, "{input} {output} {stated}");
    }
    let both = Demands {
        references: References::Caller,
        ..asks(101, 0)
    };
    let own = DemandFacts {
        references: true,
        ..window(100)
    };
    assert_eq!(both.unmet_by(false, own), Some(UnmetDemand::Pipeline));
    let host = DemandFacts {
        host_pipeline: true,
        ..own
    };
    assert_eq!(both.unmet_by(false, host), Some(UnmetDemand::Window));
}

/// A pool is shed under the largest of the demands its workers fail
/// first. A worker of another runtime that holds the request, beside
/// one of the request's runtime whose window does not: the pool is
/// shed under the window.
#[test]
fn a_pool_is_shed_under_the_largest_demand_its_workers_fail_first() {
    let long = Demands {
        runtime: 7,
        input_window: 101,
        ..Demands::default()
    };
    let elsewhere = DemandFacts {
        runtime: 8,
        ..DemandFacts::default()
    };
    let narrow = DemandFacts {
        runtime: 7,
        window: 100,
        ..DemandFacts::default()
    };
    let mix = [
        long.unmet_by(false, elsewhere),
        long.unmet_by(false, narrow),
    ];
    assert_eq!(mix, [Some(UnmetDemand::Runtime), Some(UnmetDemand::Window)]);
    assert_eq!(mix.into_iter().max(), Some(Some(UnmetDemand::Window)));
}

/// A retry's constraint is met by the workers that state its number
/// alone: a worker that states another way, or none, meets no demand
/// for one. It is judged last, behind the window.
#[test]
fn a_decided_constraint_is_met_by_the_same_number_alone() {
    let decides = |decoding: u8| DemandFacts {
        decoding,
        ..DemandFacts::default()
    };
    let retry = Demands {
        decoding: 5,
        ..Demands::default()
    };
    assert!(!retry.is_none());
    assert_eq!(
        [decides(5), decides(7), decides(0)].map(|facts| retry.unmet_by(false, facts)),
        [
            None,
            Some(UnmetDemand::Decoding),
            Some(UnmetDemand::Decoding)
        ]
    );
    assert!(Demands::default().met_by(false, decides(7)));
    let long = Demands {
        input_window: 101,
        ..retry
    };
    let narrow = DemandFacts {
        window: 100,
        ..decides(7)
    };
    assert_eq!(long.unmet_by(false, narrow), Some(UnmetDemand::Window));
}

/// The test of a request that demands nothing reads every field: the
/// default demands nothing, and a request that states one demand alone
/// demands something, whichever it is.
#[test]
fn a_request_that_states_one_demand_alone_demands_something() {
    let nothing = Demands::default();
    assert!(nothing.is_none());
    let mut alone = [nothing; 8];
    alone[0].runtime = 1;
    alone[1].vision = true;
    alone[2].references = References::Any;
    alone[3].references = References::Caller;
    alone[4].references = References::Worker;
    alone[5].input_window = 1;
    alone[6].output_window = 1;
    alone[7].decoding = 1;
    for demands in alone {
        assert!(!demands.is_none(), "{demands:?}");
    }
}

/// A prefill worker is held to the input alone: the completion's
/// bound is its decode's to hold, and so is the way a constraint is
/// decided. Every other role is held to all of it, and no other demand
/// moves with the role.
#[test]
fn a_prefill_is_held_to_the_input_alone() {
    let asked = Demands {
        runtime: 7,
        vision: true,
        references: References::Caller,
        input_window: 40,
        output_window: 400,
        decoding: 5,
    };
    let input_alone = Demands {
        output_window: 0,
        decoding: 0,
        ..asked
    };
    assert_eq!(asked.held_by(Role::Prefill), input_alone);
    for role in [Role::Single, Role::Decode, Role::Encode] {
        assert_eq!(asked.held_by(role), asked, "{role:?}");
    }
    let budget = Demands {
        output_window: 400,
        ..Demands::default()
    };
    assert!(budget.held_by(Role::Prefill).is_none());
}

/// Where an encoder takes a request's media beside a pair, the demand
/// is split between them. The encoder is held to neither window and to
/// no way of deciding a constraint: it is held to the runtime and to
/// the media's demands. A side of the pair is held to no demand of the
/// media, and to every other as it was stated. An encode request that
/// stands alone has no pair beside it: its role is held to all of it.
#[test]
fn an_encoder_and_a_pair_beside_it_split_a_requests_demand() {
    let asked = Demands {
        runtime: 7,
        vision: true,
        references: References::Worker,
        input_window: 40,
        output_window: 400,
        decoding: 5,
    };
    let the_media = Demands {
        input_window: 0,
        output_window: 0,
        decoding: 0,
        ..asked
    };
    assert_eq!(asked.beside_pair(), the_media);
    let the_rest = Demands {
        vision: false,
        references: References::None,
        ..asked
    };
    assert_eq!(asked.beside_encoders(), the_rest);
    assert_eq!(asked.held_by(Role::Encode), asked);
    let media = Demands {
        vision: true,
        references: References::Any,
        ..Demands::default()
    };
    assert!(media.beside_encoders().is_none());
    assert_eq!(media.beside_pair(), media);
    let bounds = Demands {
        input_window: 40,
        output_window: 400,
        decoding: 5,
        ..Demands::default()
    };
    assert!(bounds.beside_pair().is_none());
    assert_eq!(bounds.beside_encoders(), bounds);
}

/// The demands stand in the order a host answers for them, and each has
/// the stage its pool is shed under.
#[test]
fn the_demands_stand_in_a_hosts_order_each_with_its_stage() {
    let order = [
        UnmetDemand::Runtime,
        UnmetDemand::Vision,
        UnmetDemand::References,
        UnmetDemand::Pipeline,
        UnmetDemand::Window,
        UnmetDemand::Decoding,
    ];
    assert!(order.is_sorted());
    assert_eq!(
        order.map(UnmetDemand::stage),
        [
            "wires",
            "vision",
            "references",
            "pipeline",
            "context-window",
            "decoding"
        ]
    );
}
