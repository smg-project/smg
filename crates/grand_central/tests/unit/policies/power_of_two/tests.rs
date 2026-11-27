use super::*;
use crate::{
    document::spec::ScoreSpec,
    engine::framework::Scorer,
    stages::{
        pickers::CountFirst,
        scorers::{
            tests::{candidate, kept, picked, picked_within},
            LeastRequests,
        },
    },
};

#[test]
fn power_of_two_takes_the_better_scored_of_two_draws() {
    let pool = [candidate(0, 9), candidate(1, 9), candidate(2, 0)];
    let least_requests = |rows: &[u32]| {
        let mut scores = vec![0.0f32; rows.len()];
        LeastRequests.score(&RequestFacts::default(), &pool, rows, &mut scores);
        scores
    };
    let mut hits = 0;
    for seed in 0..100u64 {
        let facts = RequestFacts {
            seed,
            ..RequestFacts::default()
        };
        let order = picked(&PowerOfTwo, &facts, &pool, &least_requests);
        assert_eq!(order.len(), 2);
        if order[0] == 2 {
            hits += 1;
        }
    }
    // Worker 2 is drawn in two of three pairs and wins each time.
    assert!((50..=85).contains(&hits), "{hits}");
}

/// The review's case: counts 0, 1 and 2000 with the idle worker's wait
/// the worst of the three. The count decides before any score: the idle
/// worker leads, the next by count follows; a drawn pair is led by its
/// fewer in flight, so the busiest worker never leads one.
#[test]
fn count_first_takes_the_fewest_in_flight_before_any_score() {
    let pool = [candidate(0, 0), candidate(1, 1), candidate(2, 2000)];
    let by_position = |rows: &[u32]| -> Vec<f32> {
        rows.iter()
            .map(|&row| [0.0f32, 1.0, 0.49][row as usize])
            .collect()
    };
    for seed in 0..20u64 {
        let facts = RequestFacts {
            seed,
            ..RequestFacts::default()
        };
        let order = picked(&CountFirst, &facts, &pool, &by_position);
        assert_eq!(order, [0, 1, 2], "seed {seed}");
        let drawn = picked(&PowerOfTwoCountFirst, &facts, &pool, &by_position);
        assert_eq!(drawn.len(), 2, "seed {seed}");
        assert!(
            pool[drawn[0] as usize].in_flight <= pool[drawn[1] as usize].in_flight,
            "seed {seed}: {drawn:?}"
        );
        assert_ne!(drawn[0], 2, "seed {seed}: the busiest worker never leads");
    }
}

/// A drawn pair of equal scores is led by an even draw: the third value
/// of the seed's stream, between the two rows as the pool lists them.
/// Over many seeds every worker leads as often as another, wherever it
/// sits in the pool. (The stream's first value read again over the two
/// rows would favor a row by where it sits.)
#[test]
fn a_tie_of_two_draws_is_broken_evenly_wherever_the_rows_sit() {
    let pool = [
        candidate(0, 1),
        candidate(1, 1),
        candidate(2, 1),
        candidate(3, 1),
    ];
    let level = |rows: &[u32]| vec![0.5f32; rows.len()];
    for picker in [&PowerOfTwo as &dyn Picker, &PowerOfTwoCountFirst] {
        let mut leads = [0u32; 4];
        for seed in 0..4_000u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let order = picked(picker, &facts, &pool, &level);
            let mut stream = Draw::new(seed);
            let a = stream.below(4) as u32;
            let mut b = stream.below(3) as u32;
            if b >= a {
                b += 1;
            }
            let lead = if stream.below(2) == 0 {
                a.min(b)
            } else {
                a.max(b)
            };
            assert_eq!(order, [lead, a + b - lead], "seed {seed}");
            leads[order[0] as usize] += 1;
        }
        assert!(
            leads.iter().all(|n| (850..=1150).contains(n)),
            "{}: {leads:?}",
            picker.name()
        );
    }
}

/// The band is the scorers' word on what they cannot tell apart, and
/// every picker that compares scores reads it: two workers of one count
/// whose scores lie within the band are a tie and drawn evenly, and two
/// further apart are not.
#[test]
fn scores_within_the_band_are_a_tie_for_every_picker_that_compares_them() {
    let pair = [candidate(0, 2), candidate(1, 2)];
    let near = |rows: &[u32]| -> Vec<f32> {
        rows.iter()
            .map(|&row| [0.50f32, 0.56][row as usize])
            .collect()
    };
    for picker in [
        &PowerOfTwo as &dyn Picker,
        &PowerOfTwoCountFirst,
        &CountFirst,
    ] {
        let mut leads = [0u32; 2];
        for seed in 0..2_000u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let within = picked_within(picker, &facts, &pair, &near, TieBand(0.1));
            leads[within[0] as usize] += 1;
            let apart = picked_within(picker, &facts, &pair, &near, TieBand(0.05));
            assert_eq!(
                apart,
                [1, 0],
                "{}, seed {seed}: the higher score",
                picker.name()
            );
        }
        assert!(
            leads.iter().all(|n| (850..=1150).contains(n)),
            "{}: {leads:?}",
            picker.name()
        );
    }
}

/// A two-draw pick handed rows that were never narrowed draws its own
/// two, the two its narrowing would have put in front. A plugin picker's
/// fallback is handed every survivor with their scores: it picks as the
/// engine's order does over the same scores.
#[test]
fn a_two_draw_pick_over_rows_that_were_not_narrowed_draws_its_own_two() {
    let pool = [
        candidate(0, 4),
        candidate(1, 2),
        candidate(2, 4),
        candidate(3, 2),
        candidate(4, 3),
    ];
    // The scores tie two by two, so the tie's draw is in the case.
    let scores = [0.3f32, 0.3, 0.7, 0.7, 0.5];
    let by_position =
        |rows: &[u32]| -> Vec<f32> { rows.iter().map(|&row| scores[row as usize]).collect() };
    let mut leads = std::collections::BTreeSet::new();
    for picker in [&PowerOfTwo as &dyn Picker, &PowerOfTwoCountFirst] {
        for seed in 0..200u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let mut out = Targets::new();
            picker.pick(
                &facts,
                &pool,
                &kept(5),
                &scores,
                TieBand::NORMALIZED,
                &mut out,
            );
            let whole: Vec<u32> = out.iter().map(|t| t.position).collect();
            assert_eq!(
                whole,
                picked(picker, &facts, &pool, &by_position),
                "{}, seed {seed}",
                picker.name()
            );
            leads.insert(whole[0]);
        }
    }
    assert_eq!(leads.len(), 5, "the draw reaches every row: {leads:?}");
}

/// Over a pool under the draw's size the two draws give up at the small
/// pool's bound of refused rows, and from that size on at the pickers'
/// own. A pool that refuses every row is asked about that many rows and
/// no more. A pool that keeps two rows has both handed out wherever
/// fewer rows were refused, and nobody where the bound was met.
#[test]
fn the_two_draws_give_up_sooner_over_a_pool_under_the_draws_size() {
    let from = DRAW_FROM as u32;
    for (rows, bound) in [
        (3, SMALL_POOL_BOUND),
        (from - 1, SMALL_POOL_BOUND),
        (from, DRAW_BOUND),
    ] {
        let pool: Vec<Candidate> = (0..rows).map(|row| candidate(row, 0)).collect();
        let (facts, shared) = (RequestFacts::default(), PoolFacts::default());
        for picker in [&PowerOfTwo as &dyn Picker, &PowerOfTwoCountFirst] {
            let (mut gave_up, mut drew_two) = (0u32, 0u32);
            for seed in 0..4096u64 {
                let at = format!("{}, {rows} rows, seed {seed}", picker.name());
                let (mut asked, mut kept) = (0u32, Vec::new());
                let mut refuse = |_: &Candidate| {
                    asked += 1;
                    false
                };
                let mut draw = Draw::new(seed);
                let drew =
                    picker.draw_first(&facts, &shared, &pool, &mut refuse, &mut kept, &mut draw);
                assert!(!drew && kept.is_empty(), "{at}: nobody is kept");
                assert_eq!(asked, bound, "{at}: the rows asked about");

                let mut refused = 0u32;
                let mut first_two = |row: &Candidate| {
                    refused += u32::from(row.position >= 2);
                    row.position < 2
                };
                let mut draw = Draw::new(seed);
                let drew =
                    picker.draw_first(&facts, &shared, &pool, &mut first_two, &mut kept, &mut draw);
                assert_eq!(drew, refused < bound, "{at}: {refused} refused");
                kept.sort_unstable();
                if drew {
                    assert_eq!(kept, [0, 1], "{at}: the two rows kept");
                    drew_two += 1;
                } else {
                    assert!(kept.is_empty(), "{at}: nobody behind the bound");
                    gave_up += 1;
                }
            }
            let name = picker.name();
            assert!(drew_two > 0, "{name}, {rows} rows: no draw held its two");
            assert!(gave_up > 0, "{name}, {rows} rows: no draw met its bound");
        }
    }
}

#[test]
fn the_power_of_two_preset_has_four_forms_by_its_load_metric() {
    let form = |metric| {
        let spec = PolicySpec::power_of_two(metric).expect("a preset");
        let engine = spec.validate().expect("validates");
        assert!(engine.needs().loads, "{metric:?}");
        let profile = spec.profiles["power_of_two"].clone();
        (profile.score, profile.pick.name().to_string())
    };
    let least_load: Vec<ScoreSpec> = vec!["least-load".into()];
    assert_eq!(
        form(PowerOfTwoMetric::ExpectedWait),
        (least_load.clone(), "power-of-two".to_string())
    );
    assert_eq!(
        form(PowerOfTwoMetric::Requests),
        (least_load.clone(), "power-of-two-count-first".to_string())
    );
    assert_eq!(
        form(PowerOfTwoMetric::LeastRequests),
        (least_load, "count-first".to_string())
    );
    let resident: Vec<ScoreSpec> = vec![ScoreStage::ResidentWork(ResidentWorkParams {
        weight: 1.0,
        request_tokens: 2048,
    })
    .into()];
    assert_eq!(
        form(PowerOfTwoMetric::ResidentTokens {
            request_tokens: 2048
        }),
        (resident, "best".to_string())
    );
    // A request priced at nothing is a form too: the tokens alone.
    PolicySpec::power_of_two(PowerOfTwoMetric::ResidentTokens { request_tokens: 0 })
        .expect("a preset")
        .validate()
        .expect("the tokens alone");
}
