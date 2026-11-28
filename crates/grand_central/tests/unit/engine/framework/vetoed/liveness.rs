//! The rule's second tier, where the liveness veto is lifted: doubted workers,
//! rows the host holds, and the passes each costs.

use super::*;

/// Keeps every row and counts the rows it was asked about.
#[derive(Debug)]
struct Counts(Arc<AtomicUsize>);
impl Filter for Counts {
    fn name(&self) -> &'static str {
        "counts"
    }
    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        self.0.fetch_add(1, Ordering::Relaxed);
        true
    }
}

/// A pool doubted on every ready worker goes to the fewest requests in
/// flight, under every preset and under either word of the binding: a
/// doubt is not an outage.
#[test]
fn a_pool_doubted_on_every_worker_goes_to_the_fewest_in_flight() {
    let pool = [doubted(0, 3), doubted(1, 1), doubted(2, 2)];
    for preset in [
        "random",
        "round_robin",
        "passthrough",
        "power_of_two",
        "least_load",
    ] {
        for shed in [false, true] {
            let decision = decide(&engine_of(preset, shed), &facts(1), &pool);
            assert_eq!(order(&decision), vec![1, 2, 0], "{preset}, shed {shed}");
            let pick = decision.pick().expect("a pick");
            assert_eq!(pick.session_verdict, SessionVerdict::None, "{preset}");
        }
    }
}

/// The overload veto is lifted first: a worker over its thresholds
/// alone takes the request before any doubted one, whatever it holds.
/// With no such worker the doubted ones are taken by their load, over
/// their thresholds or not.
#[test]
fn a_worker_over_alone_is_taken_before_the_doubted_ones() {
    let engine = engine_of("least_load", false);
    let pool = [doubted_over(0, 0), doubted(1, 3), over(2, 5)];
    assert_eq!(order(&decide(&engine, &facts(1), &pool)), vec![2]);
    let pool = [doubted_over(0, 2), doubted(1, 1)];
    assert_eq!(order(&decide(&engine, &facts(1), &pool)), vec![1, 0]);
    let pool = [doubted_over(0, 0), doubted(1, 5)];
    assert_eq!(order(&decide(&engine, &facts(1), &pool)), vec![0, 1]);
}

/// A binding that sheds keeps the overload veto in the lifted pass: a
/// doubted worker under its thresholds takes the request, a doubted one
/// over them does not, and a pool over on every ready worker is shed as
/// saturated, doubted or not.
#[test]
fn a_binding_that_sheds_takes_a_doubted_worker_under_its_thresholds() {
    let engine = engine_of("least_load", true);
    let pool = [over(0, 0), doubted(1, 3), doubted(2, 1)];
    assert_eq!(order(&decide(&engine, &facts(1), &pool)), vec![2, 1]);
    let pool = [doubted_over(0, 0), doubted(1, 3)];
    assert_eq!(order(&decide(&engine, &facts(1), &pool)), vec![1]);
    for pool in [
        vec![over(0, 3), doubted_over(1, 1)],
        vec![doubted_over(0, 3), doubted_over(1, 1), doubted_over(2, 2)],
    ] {
        assert_eq!(
            shed_of(&decide(&engine, &facts(1), &pool)),
            (ShedReason::Saturated, "not-overloaded")
        );
    }
}

/// A worker that is down, draining or behind an open breaker is not
/// usable with both vetoes lifted: it is never the steer, and a pool of
/// such workers alone is shed at the health stage.
#[test]
fn a_doubted_pool_never_steers_to_a_worker_another_veto_refuses() {
    let engine = engine_of("least_load", false);
    let mut pool = [doubted(0, 0), doubted(1, 0), doubted(2, 0), doubted(3, 7)];
    pool[0].health.healthy = false;
    pool[1].health.draining = true;
    pool[2].health.breaker_open = true;
    assert_eq!(order(&decide(&engine, &facts(1), &pool)), vec![3]);
    pool[3].health.healthy = false;
    assert_eq!(
        shed_of(&decide(&engine, &facts(1), &pool)),
        (ShedReason::NoEndpoints, "healthy")
    );
}

/// Equal loads among doubted workers are drawn by the decision's seed,
/// over the tied rows alone.
#[test]
fn a_tie_among_the_doubted_is_drawn_by_the_seed() {
    let engine = engine_of("round_robin", false);
    let pool = [doubted(0, 2), doubted(1, 2), doubted(2, 5)];
    let mut firsts = [0u32; 3];
    for seed in 1..=600 {
        let first = order(&decide(&engine, &facts(seed), &pool))[0];
        firsts[first as usize] += 1;
    }
    assert_eq!(firsts[2], 0, "{firsts:?}");
    assert!(firsts[0] > 200 && firsts[1] > 200, "{firsts:?}");
}

/// A host's own lift of the overload veto ends the overload tier alone:
/// a doubted pool under it is still steered. The lifted pass is as its
/// host made it, so under a binding that sheds a host's lift stands
/// there too, and a doubted worker over its thresholds is taken.
#[test]
fn a_hosts_lift_leaves_the_liveness_tier_its_own() {
    let pool = [doubted_over(0, 3), doubted_over(1, 1), doubted(2, 2)];
    let lifted = Constraints {
        overload_lifted: true,
        ..Constraints::default()
    };
    for shed in [false, true] {
        let decision = engine_of("least_load", shed).decide(
            &facts(1),
            &PoolFacts::default(),
            &pool,
            &lifted,
            &mut Scratch::new(),
        );
        assert_eq!(order(&decision), vec![1, 2, 0], "shed {shed}");
    }
    // With no lift of the host's, a binding that sheds keeps the veto.
    let shedding = engine_of("least_load", true);
    assert_eq!(order(&decide(&shedding, &facts(1), &pool)), vec![2]);
}

/// Affinity outlives a load signal, not a doubt. A key whose ring
/// worker is doubted passes to the ring's next member, over its
/// thresholds or not. With every worker doubted the key goes to the
/// fewest requests in flight, whichever worker the ring names, and no
/// pin is written: the key is its ring's again when the doubt ends.
/// Under a binding that sheds, a doubted ring worker under its
/// thresholds takes its key once every other worker is over.
#[test]
fn a_key_with_every_worker_doubted_goes_to_the_fewest_in_flight() {
    let engine = engine_of("consistent_hashing", false);
    let request = keyed("k1");
    let mut pool = [free(0, 5), free(1, 4), free(2, 3), free(3, 2)];
    let ring = ring_order(&engine, &request, &pool);
    let (owner, next) = (ring[0] as usize, ring[1] as usize);
    pool[owner].health.stalled = true;
    let first = order(&decide(&engine, &request, &pool))[0];
    assert_eq!(first, ring[1], "beside free workers");
    for worker in &mut pool {
        worker.health.overloaded = !worker.health.stalled;
    }
    let first = order(&decide(&engine, &request, &pool))[0];
    assert_eq!(first, ring[1], "beside workers over their thresholds");
    for worker in &mut pool {
        worker.health.overloaded = false;
        worker.health.stalled = true;
    }
    pool[owner].in_flight = 9;
    pool[next].in_flight = 8;
    let fewest = (0..pool.len())
        .min_by_key(|&i| pool[i].in_flight)
        .expect("a worker");
    let decision = decide(&engine, &request, &pool);
    assert_eq!(order(&decision)[0] as usize, fewest, "every worker doubted");
    let pick = decision.pick().expect("a pick");
    assert_eq!(pick.session_verdict, SessionVerdict::None);

    let shedding = engine_of("consistent_hashing", true);
    let mut pool = [over(0, 5), over(1, 4), over(2, 3), over(3, 2)];
    let ring = ring_order(&shedding, &request, &pool);
    let owner = ring[0] as usize;
    pool[owner].health.overloaded = false;
    pool[owner].health.stalled = true;
    assert_eq!(order(&decide(&shedding, &request, &pool)), vec![ring[0]]);
}

/// A shed no doubt had a part in is not filtered again for the liveness
/// veto: a pool with no doubted row, and a doubted pool emptied before
/// the health stage ran, cost the one pass they cost without the tier.
/// A doubted pool behind the health stage costs the tier's one pass
/// more, and its shed names the stage that refused the rows with no
/// doubt on them.
#[test]
fn a_shed_no_doubt_had_a_part_in_costs_no_further_pass() {
    // No doubted row: the health stage keeps both and the next refuses.
    let pool = [free(0, 1), free(1, 2)];
    let (engine, asked) = counted(Box::new(Healthy), true);
    assert_eq!(
        shed_of(&decide(&engine, &facts(1), &pool)),
        (ShedReason::NoEndpoints, "refuses")
    );
    assert_eq!(asked.load(Ordering::Relaxed), pool.len());
    // Doubted rows refused ahead of the health stage: it never ran.
    let pool = [doubted(0, 1), doubted(1, 2)];
    let (engine, asked) = counted(Box::new(Healthy), false);
    assert_eq!(
        shed_of(&decide(&engine, &facts(1), &pool)),
        (ShedReason::NoEndpoints, "refuses")
    );
    assert_eq!(asked.load(Ordering::Relaxed), pool.len());
    // Doubted rows behind it: the first pass ends at the health stage
    // and the tier's pass asks the next stage about each row once.
    let (engine, asked) = counted(Box::new(Healthy), true);
    assert_eq!(
        shed_of(&decide(&engine, &facts(1), &pool)),
        (ShedReason::NoEndpoints, "refuses")
    );
    assert_eq!(asked.load(Ordering::Relaxed), pool.len());
}

/// The rule is asked once a decision: its own passes are put to no rule.
/// A pool over and doubted on every worker ends its first pass at the
/// overload veto, and the overload lift's pass at the health stage. The
/// liveness tier's pass alone reaches the filter listed behind both,
/// and asks it about each row once.
#[test]
fn the_rules_own_passes_are_put_to_no_rule() {
    let pool = [doubted_over(0, 1), doubted_over(1, 2)];
    let (engine, asked) =
        counted_among(|refuses| vec![Box::new(NotOverloaded), Box::new(Healthy), refuses]);
    assert_eq!(
        shed_of(&decide(&engine, &facts(1), &pool)),
        (ShedReason::NoEndpoints, "refuses")
    );
    assert_eq!(asked.load(Ordering::Relaxed), pool.len());
}

/// A pair's prefill pool doubted on every worker, with nobody usable in
/// it, is shed at the health stage. The pair's default lifts the doubt
/// in its last pass alone: the filter listed behind that stage is asked
/// about both prefills there, and about none before.
#[test]
fn a_pairs_prefill_pool_doubted_with_no_usable_worker_sheds_at_the_health_stage() {
    let none: (&[u32], &[u32]) = (&[], &[]);
    let side = pairs_side_shed(Box::new(Healthy), [doubted(0, 1), doubted(1, 2)], none);
    assert_eq!(side, ((ShedReason::NoEndpoints, "healthy"), 2));
}

/// A hold stands in every pass of a pair's default. A prefill pool the
/// host holds on every row is shed at the health stage, and the filter
/// listed behind that stage is asked about no prefill. With a doubted
/// row among held ones it is asked about that row alone.
#[test]
fn a_pairs_prefill_pool_under_a_hold_sheds_at_the_health_stage() {
    for (held, asked) in [(&[0, 1][..], 0), (&[0], 1)] {
        let masks: (&[u32], &[u32]) = (&[0, 1], held);
        let side = pairs_side_shed(Box::new(Healthy), [free(0, 1), free(1, 2)], masks);
        assert_eq!(side, ((ShedReason::NoEndpoints, "healthy"), asked));
    }
}

/// The liveness tier follows the pass the overload lift ended. Where a
/// document lists the overload filter ahead of the health stage, a pool
/// over and doubted on every worker ends its first pass before the
/// health stage ran; the overload lift's pass ends at the health stage,
/// and the tier takes it from there. A binding that sheds lifts no
/// overload veto, and that pool is shed as saturated.
#[test]
fn the_liveness_tier_follows_the_pass_the_overload_lift_ended() {
    let filters = ["not-overloaded", "healthy", "not-tried"];
    let pool = [doubted_over(0, 3), doubted_over(1, 1), doubted_over(2, 2)];
    let steering = engine_filtering(&filters, false);
    assert_eq!(order(&decide(&steering, &facts(1), &pool)), vec![1, 2, 0]);
    let shedding = engine_filtering(&filters, true);
    assert_eq!(
        shed_of(&decide(&shedding, &facts(1), &pool)),
        (ShedReason::Saturated, "not-overloaded")
    );
}

/// The retry's tried list does not narrow the liveness tier either: a
/// doubted worker an earlier attempt tried is taken again when it holds
/// the fewest requests, whether its own bit doubts it or the host's
/// mask. A worker the host just failed to reach is passed over while
/// another doubted one stands.
#[test]
fn a_doubted_worker_tried_is_taken_again_when_it_holds_the_fewest() {
    let engine = engine_of("round_robin", true);
    let pool = [doubted(0, 3), doubted(1, 1), doubted(2, 2)];
    let tried = [WorkerId(101)];
    let retry = RequestFacts {
        tried: &tried,
        attempt: 1,
        ..facts(1)
    };
    assert_eq!(order(&decide(&engine, &retry, &pool)), vec![1, 2, 0]);
    // The pass that leaves the host's mask out sets the list aside too.
    let masked = Constraints {
        stalled: &[0, 1, 2],
        ..Constraints::default()
    };
    let bare = [free(0, 3), free(1, 1), free(2, 2)];
    let decision = engine.decide(
        &retry,
        &PoolFacts::default(),
        &bare,
        &masked,
        &mut Scratch::new(),
    );
    assert_eq!(order(&decision), vec![1, 2, 0]);
    let excluded = Constraints {
        excluded: &tried,
        ..Constraints::default()
    };
    let decision = engine.decide(
        &facts(1),
        &PoolFacts::default(),
        &pool,
        &excluded,
        &mut Scratch::new(),
    );
    assert_eq!(order(&decision), vec![2, 0]);
}

/// The pass that lifts the liveness veto hands its rows back to the
/// scratch: the next doubted decision finds them sized.
#[test]
fn the_lifted_pass_hands_its_rows_back_to_the_scratch() {
    let engine = engine_of("least_load", true);
    let pool = [doubted(0, 3), doubted(1, 1), doubted(2, 2)];
    let mut scratch = Scratch::new();
    let decision = engine.decide(
        &facts(1),
        &PoolFacts::default(),
        &pool,
        &Constraints::default(),
        &mut scratch,
    );
    assert_eq!(order(&decision), vec![1, 2, 0]);
    assert!(scratch.single.undoubted.capacity() >= pool.len());
}

/// The trace says which veto the rule lifted: the overload veto for a
/// worker over its thresholds alone, the liveness veto for a doubted
/// one, under either word of the binding, and none for a pick no lift
/// made.
#[test]
fn a_steered_pick_says_which_veto_was_lifted() {
    let mark = |engine: &Engine, pool: &[Candidate]| decide(engine, &facts(1), pool).trace().lifted;
    let steering = engine_of("least_load", false);
    assert_eq!(
        mark(&steering, &[over(0, 1), doubted(1, 0)]),
        LiftedVeto::Overload
    );
    assert_eq!(
        mark(&steering, &[doubted(0, 1), doubted_over(1, 0)]),
        LiftedVeto::Liveness
    );
    assert_eq!(
        mark(&steering, &[free(0, 1), doubted(1, 0)]),
        LiftedVeto::None
    );
    let shedding = engine_of("least_load", true);
    assert_eq!(
        mark(&shedding, &[over(0, 1), doubted(1, 3)]),
        LiftedVeto::Liveness
    );
}

/// A doubt the host hands beside the rows is the same doubt: a pool its
/// mask covers whole is steered as one the rows' own bits doubt, and no
/// row is copied for it. A row with its own bit beside masked rows is
/// read with no doubt too, and a row the mask leaves out stands with no
/// lift.
#[test]
fn a_pool_the_hosts_mask_doubts_is_steered_too() {
    let engine = engine_of("least_load", true);
    let ask = |pool: &[Candidate], stalled: &[u32], scratch: &mut Scratch| {
        let masked = Constraints {
            stalled,
            ..Constraints::default()
        };
        engine.decide(&facts(1), &PoolFacts::default(), pool, &masked, scratch)
    };
    let mut scratch = Scratch::new();
    let decision = ask(
        &[free(0, 3), free(1, 1), free(2, 2)],
        &[0, 1, 2],
        &mut scratch,
    );
    assert_eq!(order(&decision), vec![1, 2, 0]);
    assert_eq!(decision.trace().lifted, LiftedVeto::Liveness);
    assert_eq!(scratch.single.undoubted.capacity(), 0, "no row is copied");
    let decision = ask(
        &[free(0, 3), doubted(1, 1), free(2, 2)],
        &[0, 2],
        &mut scratch,
    );
    assert_eq!(order(&decision), vec![1, 2, 0]);
    assert_eq!(decision.trace().lifted, LiftedVeto::Liveness);
    let decision = ask(&[free(0, 0), free(1, 5)], &[0], &mut scratch);
    assert_eq!(order(&decision), vec![1]);
    assert_eq!(decision.trace().lifted, LiftedVeto::None);
}

/// A row a hold of the host stands on is not doubted: the tier lifts the
/// host's mask and leaves its held rows masked. A pool held on every
/// row is shed at the health stage under either word of the binding, as
/// with no tier. One doubted row among held ones takes the request
/// whatever it holds. So it does where the host wrote its rows: there a
/// held row carries the breaker bit beside the stalled one.
#[test]
fn a_row_the_host_holds_stays_refused_through_the_liveness_tier() {
    let pool = [free(0, 0), free(1, 1), free(2, 9), free(3, 2)];
    let ask = |engine: &Engine, pool: &[Candidate], stalled: &[u32], held: &[u32]| {
        let masked = Constraints {
            stalled,
            held,
            ..Constraints::default()
        };
        let mut scratch = Scratch::new();
        engine.decide(
            &facts(1),
            &PoolFacts::default(),
            pool,
            &masked,
            &mut scratch,
        )
    };
    let mut written = pool;
    for row in &mut written {
        row.health.stalled = true;
        row.health.breaker_open = row.position != 2;
    }
    for shed in [false, true] {
        let engine = engine_of("least_load", shed);
        let whole = ask(&engine, &pool, &[0, 1, 2, 3], &[0, 1, 2, 3]);
        assert_eq!(shed_of(&whole), (ShedReason::NoEndpoints, "healthy"));
        for one in [
            ask(&engine, &pool, &[0, 1, 2, 3], &[0, 1, 3]),
            ask(&engine, &written, &[], &[]),
        ] {
            assert_eq!(order(&one), vec![2], "shed {shed}");
            assert_eq!(one.trace().lifted, LiftedVeto::Liveness);
        }
    }
}

/// Keeps every row, and writes down what a stage that knows no lift
/// reads of each: whether the row's stalled bit or the host's mask
/// refuses it.
#[derive(Debug)]
struct Shown(Arc<Mutex<Vec<bool>>>);
impl Filter for Shown {
    fn name(&self) -> &'static str {
        "shown"
    }
    fn keep(&self, _: &RequestFacts<'_>, row: &Candidate, cx: &Constraints<'_>) -> bool {
        let mut shown = self.0.lock().expect("no reader panics");
        shown.push(row.health.stalled || cx.masks(row.position));
        true
    }
}

/// A stage of a host's own that reads the stalled bit and the mask, and
/// knows no lift, refuses a held row in every pass: the first, and the
/// lifted one, where a doubted row reads clear. Rows the host wrote and
/// rows it lent beside its lists show the stage the same.
#[test]
fn a_held_row_reads_as_vetoed_to_a_stage_in_every_pass() {
    let shown = Arc::new(Mutex::new(Vec::new()));
    let stage: Box<dyn Filter> = Box::new(Shown(Arc::clone(&shown)));
    let (engine, _) = counted_among(|_| vec![stage, Box::new(Healthy)]);
    let lent = [free(0, 0), free(1, 1)];
    let mut written = lent;
    written[0].health.stalled = true;
    written[0].health.breaker_open = true;
    written[1].health.stalled = true;
    let hosts: [(&[Candidate], &[u32], &[u32]); 2] = [(&lent, &[0, 1], &[0]), (&written, &[], &[])];
    for (pool, stalled, held) in hosts {
        let masked = Constraints {
            stalled,
            held,
            ..Constraints::default()
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide(
            &facts(1),
            &PoolFacts::default(),
            pool,
            &masked,
            &mut scratch,
        );
        assert_eq!(order(&decision), vec![1]);
        assert_eq!(decision.trace().lifted, LiftedVeto::Liveness);
        // Row by row: the first pass, then the lifted one.
        let passes = std::mem::take(&mut *shown.lock().expect("no reader panics"));
        assert_eq!(passes, [true, true, true, false]);
    }
}

/// The tier runs where it would gain a row and nowhere else: over a mask
/// of held rows alone it makes no pass. A filter listed ahead of the
/// health stage is asked about each row once there, and once more by the
/// tier's pass where one row of the mask is doubted.
#[test]
fn a_mask_of_held_rows_alone_costs_no_further_pass() {
    let pool = [free(0, 0), free(1, 1), free(2, 9), free(3, 2)];
    let asked = Arc::new(AtomicUsize::new(0));
    let counts: Box<dyn Filter> = Box::new(Counts(Arc::clone(&asked)));
    let (engine, _) = counted_among(|_| vec![counts, Box::new(Healthy)]);
    let ask = |held: &[u32]| {
        let masked = Constraints {
            stalled: &[0, 1, 2, 3],
            held,
            ..Constraints::default()
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide(
            &facts(1),
            &PoolFacts::default(),
            &pool,
            &masked,
            &mut scratch,
        );
        (decision, asked.swap(0, Ordering::Relaxed))
    };
    let (whole, passes) = ask(&[0, 1, 2, 3]);
    assert_eq!(shed_of(&whole), (ShedReason::NoEndpoints, "healthy"));
    assert_eq!(passes, pool.len());
    let (one, passes) = ask(&[0, 1, 3]);
    assert_eq!(order(&one), vec![2]);
    assert_eq!(passes, 2 * pool.len());
}

/// A held row that carries a doubt of its own gains nobody either: the
/// hold stands through the tier, on a lent row with its leader's bit
/// as on any other. A pool held on every row costs its one pass
/// whether or not a row carries the bit. A row that carries it with no
/// hold on it still opens the tier and takes the request.
#[test]
fn a_held_row_that_carries_its_own_doubt_costs_no_further_pass() {
    let mut pool = [free(0, 0), free(1, 1), free(2, 9)];
    pool[0].health.stalled = true;
    pool[1].health.stalled = true;
    let asked = Arc::new(AtomicUsize::new(0));
    let counts: Box<dyn Filter> = Box::new(Counts(Arc::clone(&asked)));
    let (engine, _) = counted_among(|_| vec![counts, Box::new(Healthy)]);
    let ask = |stalled: &[u32], held: &[u32]| {
        let masked = Constraints {
            stalled,
            held,
            ..Constraints::default()
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide(
            &facts(1),
            &PoolFacts::default(),
            &pool,
            &masked,
            &mut scratch,
        );
        (decision, asked.swap(0, Ordering::Relaxed))
    };
    let (whole, passes) = ask(&[0, 1, 2], &[0, 1, 2]);
    assert_eq!(shed_of(&whole), (ShedReason::NoEndpoints, "healthy"));
    assert_eq!(passes, pool.len());
    // Row 1 carries the bit and no hold: the tier opens for it alone.
    let (one, passes) = ask(&[0, 2], &[0, 2]);
    assert_eq!(order(&one), vec![1]);
    assert_eq!(one.trace().lifted, LiftedVeto::Liveness);
    assert_eq!(passes, 2 * pool.len());
}

/// A doubt on a row a harder veto refuses too opens no tier: lifting it
/// gains nobody. A pool with no ready worker but the doubted one, which
/// is not ready, draining or behind an open breaker, costs the one pass
/// it costs with no doubt in it, by the host's mask and by the row's own
/// bit alike. A doubt alone on a ready row costs the tier's pass, and
/// that row takes the request.
#[test]
fn a_doubt_on_a_row_a_harder_veto_refuses_costs_no_further_pass() {
    let asked = Arc::new(AtomicUsize::new(0));
    let counts: Box<dyn Filter> = Box::new(Counts(Arc::clone(&asked)));
    let (engine, _) = counted_among(|_| vec![counts, Box::new(Healthy)]);
    let ask = |pool: &[Candidate], stalled: &[u32]| {
        let masked = Constraints {
            stalled,
            ..Constraints::default()
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide(
            &facts(1),
            &PoolFacts::default(),
            pool,
            &masked,
            &mut scratch,
        );
        (decision, asked.swap(0, Ordering::Relaxed))
    };
    // No row is ready but the second, which each case refuses its way.
    let pool_of = |refuse: fn(&mut Health)| {
        let mut pool = [free(0, 0), free(1, 0), free(2, 0)];
        pool[0].health.healthy = false;
        pool[2].health.healthy = false;
        refuse(&mut pool[1].health);
        pool
    };
    let harder: [fn(&mut Health); 3] = [
        |health| health.healthy = false,
        |health| health.draining = true,
        |health| health.breaker_open = true,
    ];
    for refuse in harder {
        let mut pool = pool_of(refuse);
        let (masked, passes) = ask(&pool, &[1]);
        assert_eq!(shed_of(&masked), (ShedReason::NoEndpoints, "healthy"));
        assert_eq!(passes, pool.len());
        pool[1].health.stalled = true;
        let (own, passes) = ask(&pool, &[]);
        assert_eq!(shed_of(&own), (ShedReason::NoEndpoints, "healthy"));
        assert_eq!(passes, pool.len());
    }
    let mut pool = pool_of(|_| {});
    let (masked, passes) = ask(&pool, &[1]);
    assert_eq!(order(&masked), vec![1]);
    assert_eq!(passes, 2 * pool.len());
    pool[1].health.stalled = true;
    let (own, passes) = ask(&pool, &[]);
    assert_eq!(order(&own), vec![1]);
    assert_eq!(passes, 2 * pool.len());
}

/// A kept key in a pool doubted on every worker is steered to the fewest
/// requests in flight, under a sessions rule too: the pick says the
/// liveness veto and claims no pin. So it does where every worker is
/// over its thresholds as well: the lift that kept a row there is the
/// liveness veto's, and the kept run's own mark does not write over it.
#[test]
fn a_kept_key_in_a_doubted_pool_is_steered_and_claims_nothing() {
    let request = keyed("k1");
    let under = [doubted(0, 5), doubted(1, 4), doubted(2, 1), doubted(3, 2)];
    let over = [
        doubted_over(0, 5),
        doubted_over(1, 4),
        doubted_over(2, 1),
        doubted_over(3, 2),
    ];
    for pool in [under, over] {
        for engine in [
            engine_of("consistent_hashing", false),
            ring_under_a_rule(Assignment::Delegate),
        ] {
            let decision = decide(&engine, &request, &pool);
            assert_eq!(order(&decision)[0], 2);
            assert_eq!(decision.trace().lifted, LiftedVeto::Liveness);
            let pick = decision.pick().expect("a pick");
            assert_eq!(pick.session_verdict, SessionVerdict::None);
        }
    }
}

/// A kept key reaches the liveness tier over a pool whose host lists its
/// engines as over one that lists none. The ring's walk places the key
/// while a row it reads stands. With every worker doubted the walk
/// keeps no row and the whole pass decides: the key goes to the fewest
/// requests in flight by the lift of the liveness veto and claims no
/// pin, with the workers under their thresholds and over them.
#[test]
fn a_kept_key_in_a_doubted_pool_is_steered_past_the_rings_walk() {
    let engine = engine_of("consistent_hashing", false);
    let request = keyed("k1");
    let (listed, ends) = ([0, 1, 2, 3], [1, 2, 3, 4]);
    let listing = PoolFacts {
        pool_size: 4,
        membership_epoch: 1,
        generation: 1,
        engines: Engines::new(0, &listed, &ends),
        ..PoolFacts::default()
    };
    let decide = |pool: &[Candidate]| {
        let cx = Constraints::default();
        engine.decide(&request, &listing, pool, &cx, &mut Scratch::new())
    };
    // No worker doubted: the walk alone decides, and counts no survivor.
    let calm = [free(0, 5), free(1, 4), free(2, 1), free(3, 2)];
    let walked = decide(&calm);
    assert!(walked.trace().survivors.get().is_none(), "by the walk");
    let under = [doubted(0, 5), doubted(1, 4), doubted(2, 1), doubted(3, 2)];
    let over = [
        doubted_over(0, 5),
        doubted_over(1, 4),
        doubted_over(2, 1),
        doubted_over(3, 2),
    ];
    for pool in [under, over] {
        let decision = decide(&pool);
        assert_eq!(order(&decision)[0], 2);
        assert_eq!(decision.trace().lifted, LiftedVeto::Liveness);
        let pick = decision.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::None);
    }
}

/// A kept key no pin took is answered by the lifted pass where that pass
/// leaves nobody: with the liveness tier's pick, or with a shed. Neither
/// pins the key. Under a sessions rule without a TTL the key keeps no row
/// behind either answer; under a TTL its row stays to its end.
#[test]
fn a_kept_key_the_lifted_pass_answers_keeps_no_row_under_no_ttl() {
    let request = RequestFacts {
        now_micros: 10,
        ..keyed("k1")
    };
    let steered = [doubted(0, 5), doubted(1, 4), doubted(2, 1)];
    let mut unready = [free(0, 5), free(1, 4), free(2, 1)];
    for worker in &mut unready {
        worker.health.healthy = false;
    }
    for (ttl_s, rows, site) in [(u32::MAX, 0, "no TTL"), (60, 1, "a TTL")] {
        let engine = ring_under_a_ttl(ttl_s);
        let decision = decide(&engine, &request, &steered);
        assert_eq!(order(&decision)[0], 2, "{site}");
        assert_eq!(decision.trace().lifted, LiftedVeto::Liveness, "{site}");
        let pick = decision.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::None, "{site}");
        assert_eq!(engine.sessions().len(), rows, "{site}: behind the pick");

        let engine = ring_under_a_ttl(ttl_s);
        let shed = decide(&engine, &request, &unready);
        assert_eq!(shed_of(&shed), (ShedReason::NoEndpoints, "healthy"));
        assert_eq!(engine.sessions().len(), rows, "{site}: behind the shed");
    }
}

/// An encode item of the `epd` plan is put to the liveness tier as it is
/// put to the overload tier, behind the plan's own rules: encoders
/// doubted on every worker take it by the fewest requests in flight, and
/// an encoder the pair cannot use stays unusable.
#[test]
fn an_encode_item_over_doubted_encoders_is_steered_among_those_the_plan_admits() {
    let engine = PolicySpec::epd_presets("round_robin", "round_robin", "round_robin")
        .expect("the presets' form")
        .validate()
        .expect("the presets build");
    let mut pool = [doubted(0, 0), doubted(1, 4), doubted(2, 2)];
    pool[1].ec_compatible = true;
    pool[2].ec_compatible = true;
    let item = RequestFacts {
        phase: Phase::Encode,
        ..facts(1)
    };
    let decision = decide(&engine, &item, &pool);
    assert_eq!(order(&decision), vec![2, 1]);
    assert_eq!(decision.trace().lifted, LiftedVeto::Liveness);
}
