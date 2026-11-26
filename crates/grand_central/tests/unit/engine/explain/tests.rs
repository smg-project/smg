use super::*;
use crate::{
    document::{Needs, PolicySpec},
    engine::{
        decision::{Queue, Shed, ShedReason, Trace},
        framework::{Binding, Profile, WeightedScorer},
    },
    stages::{
        filters::{Healthy, NotTried},
        pickers::Best,
        scorers::{FewestSessions, LeastRequests},
    },
};

fn naming_engine(generation: u64) -> Engine {
    let profiles = ["other", "prefill", "decode", "encode"].map(|name| Profile {
        name: name.into(),
        filters: vec![Box::new(NotTried), Box::new(Healthy)],
        scorers: vec![
            WeightedScorer {
                scorer: Box::new(LeastRequests),
                weight: 1.0,
            },
            WeightedScorer {
                scorer: Box::new(FewestSessions),
                weight: 1.0,
            },
        ],
        picker: Box::new(Best),
    });
    let bindings = [
        ("single", Plan::Single, None),
        ("pair", Plan::PdPair, Some(2)),
        ("encode-pair", Plan::Epd { encode: 3 }, Some(2)),
    ]
    .map(|(model, plan, decode_profile)| Binding {
        model: model.into(),
        phases: vec![],
        plan,
        profile: 1,
        decode_profile,
        shadow: None,
        decode_shadow: None,
        sessions: None,
    });
    Engine::new(
        profiles.into(),
        bindings.into(),
        generation,
        Needs::default(),
    )
}

fn record(generation: u64, binding: Option<u32>, profile: u16) -> DecisionRecord {
    DecisionRecord {
        decision: Decision::Shed(Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by: "view",
            retry_after_secs: 1,
            wait_allowed: false,
            trace: Trace {
                profile,
                generation,
                ..Trace::default()
            },
        }),
        binding,
        at: Instant::now(),
        shadow: None,
        shadow_target: None,
        decode_shadow: None,
        decode_shadow_target: None,
        gateway: GatewayId(11),
        request: fingerprint(b"shed"),
    }
}

#[test]
fn a_records_names_are_those_of_the_generation_that_decided() {
    let engine = naming_engine(7);
    let next = naming_engine(8);
    for (binding, profile, plan, name) in [
        (0, 1, "single", "prefill"),
        (1, 2, "pd-pair", "decode"),
        (2, 3, "epd", "encode"),
    ] {
        let record = record(7, Some(binding), profile);
        let names = record.names(&engine).expect("the standing names");
        assert_eq!(names.plan, plan);
        assert_eq!(names.profile, name);
        assert_eq!(names.picker, "best");
        assert_eq!(names.filters, ["not-tried", "healthy"]);
        assert_eq!(names.scorers, ["least-requests", "fewest-sessions"]);
        assert_eq!(record.names(&next), None);
    }
}

#[test]
fn a_trace_for_a_profile_outside_its_binding_names_the_plan_alone() {
    let engine = naming_engine(7);
    let names = record(7, Some(2), 0).names(&engine).expect("a binding");
    assert_eq!(names.plan, "epd");
    assert!(names.profile.is_empty());
    assert!(names.picker.is_empty());
    assert!(names.filters.is_empty());
    assert!(names.scorers.is_empty());
}

#[test]
fn a_record_with_no_binding_has_no_names() {
    let engine = naming_engine(7);
    assert_eq!(record(7, None, 1).names(&engine), None);
    assert_eq!(record(7, Some(3), 1).names(&engine), None);
}

#[test]
fn a_queue_ticket_does_not_count_toward_a_sample() {
    let mut document = PolicySpec::preset("random").expect("a preset");
    document.explain.sample = 2;
    let engine = document.validate().expect("a valid document");
    let ticket = Decision::Queue(Queue {
        ticket: 1,
        max_wait_micros: 2,
        trace: Trace::default(),
    });
    let shed = record(engine.generation(), None, 0).decision;
    let mut count = SampleCount::default();
    assert!(!count.keeps(0, &engine, &ticket));
    assert!(!count.keeps(0, &engine, &shed));
    assert!(count.keeps(0, &engine, &shed));
}

/// The next push to a full ring drops its oldest record.
/// The second and last request ids still find their records.
#[test]
fn the_ring_loses_its_oldest_record_when_it_wraps() {
    let mut engine = naming_engine(7);
    engine.set_explain_sample(1);
    let shed = record(engine.generation(), None, 0);
    let mut count = SampleCount::default();
    let mut samples = Samples::default();

    for n in 0..=SLOTS {
        assert!(
            count.keeps(0, &engine, &shed.decision),
            "every shed is sampled"
        );
        let id = n.to_le_bytes();
        let taken = (shed.at, shed.gateway, &id[..]);
        samples.push(0, None, taken, &shed.decision, None, [None; 2]);
    }
    for (n, present) in [(0, false), (1, true), (SLOTS, true)] {
        let found = samples.find(0, shed.gateway, fingerprint(&n.to_le_bytes()));
        assert_eq!(found.is_some(), present, "request {n}");
    }
}

/// The old generation stops one shed short of a sample.
/// The new generation skips its first shed and keeps its second.
#[test]
fn the_count_starts_over_under_a_new_generation() {
    let mut document = PolicySpec::preset("random").expect("a preset");
    document.explain.sample = 2;
    let first = document.validate().expect("a valid document");
    document.generation += 1;
    let next = document.validate().expect("a valid document");
    let mut count = SampleCount::default();
    let mut samples = Samples::default();

    for (engine, id, kept) in [
        (&first, b"old", false),
        (&next, b"one", false),
        (&next, b"two", true),
    ] {
        let shed = record(engine.generation(), None, 0);
        if count.keeps(0, engine, &shed.decision) {
            let taken = (shed.at, shed.gateway, &id[..]);
            samples.push(0, None, taken, &shed.decision, None, [None; 2]);
        }
        let found = samples.find(0, shed.gateway, fingerprint(id));
        assert_eq!(found.is_some(), kept, "request {id:?}");
    }
}

/// Each count keeps every third shed, and interleaving two counts does
/// not shift either.
#[test]
fn each_decider_keeps_every_nth_answer_by_its_own_count() {
    let mut document = PolicySpec::preset("random").expect("a preset");
    document.explain.sample = 3;
    let engine = document.validate().expect("a valid document");
    let shed = record(engine.generation(), None, 0);
    let mut counts = [SampleCount::default(), SampleCount::default()];
    let mut samples = Samples::default();

    for base in [0, 3] {
        for (seat, turn) in [(0_u8, 1), (0, 2), (1, 1), (0, 3), (1, 2), (1, 3)] {
            let n = base + turn;
            let id = [seat, n];
            if counts[usize::from(seat)].keeps(0, &engine, &shed.decision) {
                let taken = (shed.at, shed.gateway, &id[..]);
                samples.push(0, None, taken, &shed.decision, None, [None; 2]);
            }
            let found = samples.find(0, shed.gateway, fingerprint(&id));
            assert_eq!(found.is_some(), n % 3 == 0, "decider {seat}, answer {n}");
        }
    }
}
