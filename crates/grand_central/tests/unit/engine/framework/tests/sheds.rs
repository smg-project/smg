use super::*;

fn decide_single(engine: &Engine, pool: &[Candidate]) -> Decision {
    let facts = RequestFacts {
        model: "m",
        ..RequestFacts::default()
    };
    let mut scratch = Scratch::new();
    engine.decide(
        &facts,
        &PoolFacts::default(),
        pool,
        &Constraints::default(),
        &mut scratch,
    )
}

#[test]
fn an_all_overloaded_pool_sheds_as_saturated() {
    let mut engine = strict_engine();
    engine.shed_when_all_vetoed(0);
    let mut pool = [candidate(0, 1), candidate(1, 1)];
    for c in &mut pool {
        c.health.overloaded = true;
    }
    let facts = RequestFacts {
        model: "m",
        ..RequestFacts::default()
    };
    let mut scratch = Scratch::new();
    match engine.decide(
        &facts,
        &PoolFacts::default(),
        &pool,
        &Constraints::default(),
        &mut scratch,
    ) {
        Decision::Shed(shed) => {
            assert_eq!(shed.reason, ShedReason::Saturated);
            assert_eq!(
                shed.emptied_by, "not-overloaded",
                "the stage that emptied the pool is named"
            );
        }
        other => panic!("an overloaded pool answered {other:?}"),
    }
}

#[test]
fn a_pool_emptied_behind_the_healthy_filter_names_the_later_stage() {
    let mut engine = strict_engine();
    engine.shed_when_all_vetoed(0);
    let mut pool = [candidate(0, 1), candidate(1, 1)];
    pool[0].health.healthy = false;
    pool[1].health.overloaded = true;
    let facts = RequestFacts {
        model: "m",
        ..RequestFacts::default()
    };
    let mut scratch = Scratch::new();
    match engine.decide(
        &facts,
        &PoolFacts::default(),
        &pool,
        &Constraints::default(),
        &mut scratch,
    ) {
        Decision::Shed(shed) => {
            assert_eq!(shed.emptied_by, "not-overloaded");
            assert_eq!(
                shed.reason,
                ShedReason::Saturated,
                "every routable row was over the thresholds"
            );
        }
        other => panic!("a pool with no routable row answered {other:?}"),
    }
}

#[test]
fn a_draining_overloaded_pool_is_emptied_by_healthy_and_not_saturated() {
    let mut pool = [candidate(0, 1), candidate(1, 1)];
    for c in &mut pool {
        c.health.overloaded = true;
        c.health.draining = true;
    }
    match decide_single(&strict_engine(), &pool) {
        Decision::Shed(shed) => {
            assert_eq!(shed.emptied_by, "healthy");
            assert_eq!(shed.reason, ShedReason::NoEndpoints);
        }
        other => panic!("a draining pool answered {other:?}"),
    }
}

#[test]
fn an_empty_pool_is_emptied_by_no_stage() {
    match decide_single(&strict_engine(), &[]) {
        Decision::Shed(shed) => {
            assert_eq!(shed.emptied_by, "none");
            assert_eq!(shed.reason, ShedReason::NoEndpoints);
        }
        other => panic!("an empty pool answered {other:?}"),
    }
}

/// The prefills' classes are 7 and 8. Class 7's decode is over its
/// thresholds, and class 9's healthy decode partners no prefill. The
/// decode list is not wholly over its thresholds, so the request is not
/// shed as saturated, under either word of the binding: it has no
/// endpoint, and the shed names `partnered`, for the decode with room
/// that pairs with nobody. With that decode over its thresholds too, a
/// binding that sheds answers saturation.
#[test]
fn a_class_over_its_thresholds_beside_a_decode_apart_is_no_saturation() {
    let steering = strict_pair_engine();
    let mut shedding = strict_pair_engine();
    shedding.shed_when_all_vetoed(0);
    let mut prefills = [candidate(0, 0), candidate(1, 0)];
    prefills[0].pair_class = 7;
    prefills[1].pair_class = 8;
    let mut decodes = [candidate(0, 0), candidate(1, 0)];
    decodes[0].pair_class = 7;
    decodes[0].health.overloaded = true;
    decodes[1].pair_class = 9;
    for engine in [&steering, &shedding] {
        assert_eq!(
            shed_of(&decide_pair(engine, &prefills, &decodes)),
            Some((ShedReason::NoEndpoints, PARTNERED))
        );
    }
    decodes[1].health.overloaded = true;
    assert_eq!(
        shed_of(&decide_pair(&shedding, &prefills, &decodes)),
        Some((ShedReason::Saturated, "not-overloaded"))
    );
}

#[test]
fn an_unroutable_prefills_partners_do_not_name_the_shed() {
    // A routable prefill of class 1 with an unhealthy partner, a draining
    // prefill of class 2 with an overloaded one: the routable prefill's
    // partner names the shed, not the draining prefill's.
    let engine = Engine::new(
        vec![strict_profile()],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    let mut prefills = [candidate(0, 0), candidate(1, 0)];
    prefills[0].pair_class = 1;
    prefills[1].pair_class = 2;
    prefills[1].health.draining = true;
    let mut decodes = [candidate(0, 0), candidate(1, 0)];
    decodes[0].pair_class = 1;
    decodes[0].health.healthy = false;
    decodes[1].pair_class = 2;
    decodes[1].health.overloaded = true;
    match decide_pair(&engine, &prefills, &decodes) {
        Decision::Shed(shed) => {
            assert_eq!(shed.emptied_by, "healthy");
            assert_eq!(shed.reason, ShedReason::NoEndpoints);
        }
        other => panic!("a pool with one routable prefill answered {other:?}"),
    }
    // Every prefill draining over a saturated decode fleet: the prefill
    // side's own stage is named.
    prefills[0].health.draining = true;
    decodes[0].health.healthy = true;
    decodes[0].health.overloaded = true;
    match decide_pair(&engine, &prefills, &decodes) {
        Decision::Shed(shed) => {
            assert_eq!(shed.emptied_by, "healthy");
            assert_eq!(shed.reason, ShedReason::NoEndpoints);
        }
        other => panic!("a draining prefill fleet answered {other:?}"),
    }
}

#[test]
fn a_routable_prefills_partner_over_its_thresholds_beside_a_free_decode_is_no_saturation() {
    // The mirror case: the draining prefill's partner is healthy, and
    // the routable prefill's partner is over its thresholds. The decode
    // list is not wholly over them: the request has no endpoint, and
    // the shed names "partnered", for the healthy decode, which pairs
    // with no prefill that stands.
    let engine = Engine::new(
        vec![strict_profile()],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    let mut prefills = [candidate(0, 0), candidate(1, 0)];
    prefills[0].pair_class = 1;
    prefills[1].pair_class = 2;
    prefills[1].health.draining = true;
    let mut decodes = [candidate(0, 0), candidate(1, 0)];
    decodes[0].pair_class = 1;
    decodes[0].health.overloaded = true;
    decodes[1].pair_class = 2;
    match decide_pair(&engine, &prefills, &decodes) {
        Decision::Shed(shed) => {
            assert_eq!(shed.emptied_by, PARTNERED);
            assert_eq!(shed.reason, ShedReason::NoEndpoints);
        }
        other => panic!("a routable prefill beside a free decode apart answered {other:?}"),
    }
}

#[test]
fn a_saturated_decode_fleet_sheds_as_saturated() {
    let mut engine = Engine::new(
        vec![strict_profile()],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    engine.shed_when_all_vetoed(0);
    let mut prefills = [candidate(0, 0)];
    prefills[0].pair_class = 7;
    let mut decodes = [candidate(0, 0), candidate(1, 0)];
    for d in &mut decodes {
        d.pair_class = 7;
        d.health.overloaded = true;
    }
    match decide_pair(&engine, &prefills, &decodes) {
        Decision::Shed(shed) => {
            assert_eq!(shed.emptied_by, "not-overloaded");
            assert_eq!(shed.reason, ShedReason::Saturated);
        }
        other => panic!("a saturated decode fleet answered {other:?}"),
    }
}
