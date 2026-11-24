//! A seat's holder counts picks and sheds before it takes the ring lock.
//! The lock holds a push or a scan of the seat's recent decisions.

#![cfg_attr(
    not(any(test, feature = "controller")),
    expect(
        dead_code,
        reason = "Only the served controller keeps sampled records."
    )
)]

use std::{collections::VecDeque, time::Instant};

use crate::{
    document::spec::PlanName,
    engine::{
        decision::{Decision, ShadowKind},
        framework::{Engine, Plan},
        shadow::Shadowed,
        view::{fingerprint, WorkerId},
    },
    store::GatewayId,
};

/// The maximum records a decider holds.
const SLOTS: usize = 1_024;

/// The decision a door returned, kept by gateway and request fingerprint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecisionRecord {
    /// The decision and its trace, which carries the generation.
    pub decision: Decision,
    /// The position in that generation of the binding the decision ran
    /// under, if one matched. A decode decided after its prefill ran
    /// under its pair's binding, whatever phase the call stated.
    pub binding: Option<u32>,
    /// The instant the request was taken.
    pub at: Instant,
    /// What the binding's shadow said beside the decision, where one
    /// was due.
    pub shadow: Option<ShadowKind>,
    /// The first target of the shadow's pick.
    pub shadow_target: Option<WorkerId>,
    /// What the decode side's shadow said beside a pair's decode, where
    /// one was due. A pair's prefill side reads in `shadow`.
    pub decode_shadow: Option<ShadowKind>,
    /// The first target of the decode side's shadow.
    pub decode_shadow_target: Option<WorkerId>,
    gateway: GatewayId,
    request: u64,
}

/// The names of a sampled decision, from the generation that decided.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecisionNames {
    /// The binding's plan, spelled as in the policy document.
    pub plan: &'static str,
    /// The profile named by the trace. This and the stage names are empty
    /// where the decision was made ahead of every profile, or the trace's
    /// profile is outside the binding.
    pub profile: String,
    /// The profile's picker.
    pub picker: String,
    /// The profile's filters, in their declared order.
    pub filters: Vec<String>,
    /// The profile's scorers, in their declared order.
    pub scorers: Vec<String>,
}

impl DecisionRecord {
    /// Reads names if `engine` has the recorded generation number and binding.
    #[must_use]
    pub fn names(&self, engine: &Engine) -> Option<DecisionNames> {
        let trace = self.decision.trace();
        if trace.generation != engine.generation() {
            return None;
        }
        let binding = engine.binding_at(self.binding?)?;
        let plan = match binding.plan {
            Plan::Single => PlanName::Single,
            Plan::PdPair => PlanName::PdPair,
            Plan::Epd { .. } => PlanName::Epd,
        };
        let mut names = DecisionNames {
            plan: plan.name(),
            ..DecisionNames::default()
        };
        let Some(at) = trace.made_under() else {
            return Some(names);
        };
        let bound = at == binding.profile
            || binding.decode_profile == Some(at)
            || matches!(binding.plan, Plan::Epd { encode } if encode == at);
        if let Some(profile) = engine.profiles().get(usize::from(at)).filter(|_| bound) {
            names.profile.clone_from(&profile.name);
            profile.picker.name().clone_into(&mut names.picker);
            names.filters = profile
                .filters
                .iter()
                .map(|f| f.name().to_owned())
                .collect();
            names.scorers = profile
                .scorers
                .iter()
                .map(|w| w.scorer.name().to_owned())
                .collect();
        }
        Some(names)
    }
}

/// A decider's count toward its next sample, kept under one term and
/// generation.
#[derive(Default)]
pub(crate) struct SampleCount {
    term: u64,
    generation: u64,
    count: u32,
}

impl SampleCount {
    /// Counts a pick or shed, restarting under a new term or generation.
    /// Returns true when the count reaches the sampling rate.
    pub(crate) fn keeps(&mut self, term: u64, engine: &Engine, decision: &Decision) -> bool {
        if !matches!(decision, Decision::Pick(_) | Decision::Shed(_)) {
            return false;
        }
        let generation = engine.generation();
        if self.term != term || self.generation != generation {
            *self = Self {
                term,
                generation,
                count: 0,
            };
        }
        self.count += 1;
        if self.count != engine.explain_sample() {
            return false;
        }
        self.count = 0;
        true
    }
}

/// A seat's recent decisions. The seat's holder counts separately and
/// takes the ring lock to push. A reader takes it for one scan.
#[derive(Default)]
pub(crate) struct Samples {
    records: VecDeque<DecisionRecord>,
    term: u64,
}

impl Samples {
    /// Pushes one record, allocating the ring on its first write. A new
    /// lead starts with an empty ring. `binding` is the position of the
    /// binding the decision ran under. `taken` is the instant the request
    /// was taken, the gateway that asked and the request's id as stated.
    #[cold]
    pub(crate) fn push(
        &mut self,
        term: u64,
        binding: Option<u32>,
        (at, gateway, id): (Instant, GatewayId, &[u8]),
        decision: &Decision,
        shadow: Option<Shadowed>,
        sides: [Option<Shadowed>; 2],
    ) {
        debug_assert!(
            matches!(decision, Decision::Pick(_) | Decision::Shed(_)),
            "a record is a pick's or a shed's: the count keeps no other"
        );
        if self.term != term {
            *self = Self {
                term,
                ..Self::default()
            };
        }
        if self.records.capacity() == 0 {
            self.records = VecDeque::with_capacity(SLOTS);
        }
        if self.records.len() == SLOTS {
            self.records.pop_front();
        }
        // A single's shadow, or a pair's by its two sides.
        let [prefill, decode] = sides;
        self.records.push_back(DecisionRecord {
            decision: *decision,
            binding,
            at,
            shadow: shadow.or(prefill).map(|said| said.kind),
            shadow_target: shadow.or(prefill).and_then(|said| said.first),
            decode_shadow: decode.map(|said| said.kind),
            decode_shadow_target: decode.and_then(|said| said.first),
            gateway,
            request: fingerprint(id),
        });
    }

    /// The last record of this key at the seat, from the lead being read.
    pub(crate) fn find(
        &self,
        term: u64,
        gateway: GatewayId,
        request: u64,
    ) -> Option<DecisionRecord> {
        if self.term != term {
            return None;
        }
        self.records
            .iter()
            .rev()
            .find(|record| record.gateway == gateway && record.request == request)
            .copied()
    }
}

#[cfg(test)]
mod tests {
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
            overload: None,
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
}
