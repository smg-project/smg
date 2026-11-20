//! The recent decisions of one seat. The seat's holder counts and writes,
//! and a reader copies a matching record under the seat's ring lock.

use std::{collections::VecDeque, time::Instant};

use super::{Asked, Taken};
use crate::{
    decision::Decision,
    framework::{Engine, Plan},
    spec::PlanName,
    store::GatewayId,
    view::fingerprint,
};

/// The maximum records a decider holds.
const SLOTS: usize = 1_024;

/// The decision a door returned, kept by gateway and request fingerprint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecisionRecord {
    /// The decision and its trace, which carries the generation.
    pub decision: Decision,
    /// The binding's position in that generation, if one matched.
    pub binding: Option<u32>,
    /// The instant the request was taken.
    pub at: Instant,
    gateway: GatewayId,
    request: u64,
}

/// The names of a sampled decision, from the generation that decided.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Names {
    /// The binding's plan, spelled as in the policy document.
    pub plan: &'static str,
    /// The profile named by the trace. This and the stage names are empty
    /// when no profile ran or the trace's profile is outside the binding.
    pub profile: String,
    /// The profile's picker.
    pub picker: String,
    /// The profile's filters, in their declared order.
    pub filters: Vec<String>,
    /// The profile's scorers, in their declared order.
    pub scorers: Vec<String>,
}

/// A sampled decision with the names that can still be read for it.
#[derive(Clone, Debug, PartialEq)]
pub struct Explained {
    /// The sampled decision.
    pub record: DecisionRecord,
    /// `None` when the decision's generation or binding is unavailable.
    pub names: Option<Names>,
}

impl DecisionRecord {
    /// Reads names if `engine` has the recorded generation number and binding.
    #[must_use]
    pub fn names(&self, engine: &Engine) -> Option<Names> {
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
        let mut names = Names {
            plan: plan.name(),
            ..Names::default()
        };
        let Some(at) = trace.profile_ran() else {
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

#[derive(Default)]
pub(crate) struct Samples {
    records: VecDeque<DecisionRecord>,
    /// A served decider's lead. A door in process uses zero throughout.
    term: u64,
    generation: u64,
    count: u32,
}

impl Samples {
    /// Counts a pick or shed under a sampling generation, allocating the
    /// ring on the first kept decision. A new generation starts its count
    /// over. A new lead starts with an empty ring.
    #[cold]
    pub(crate) fn keep<A: Asked>(
        &mut self,
        term: u64,
        engine: &Engine,
        taken: &Taken<'_, A>,
        decision: &Decision,
    ) {
        if !matches!(decision, Decision::Pick(_) | Decision::Shed(_)) {
            return;
        }
        if self.term != term {
            *self = Self {
                term,
                ..Self::default()
            };
        }
        let generation = engine.generation();
        if self.generation != generation {
            self.generation = generation;
            self.count = 0;
        }
        self.count += 1;
        if self.count != engine.explain_sample() {
            return;
        }
        self.count = 0;
        if self.records.capacity() == 0 {
            self.records = VecDeque::with_capacity(SLOTS);
        }
        if self.records.len() == SLOTS {
            self.records.pop_front();
        }
        self.records.push_back(DecisionRecord {
            decision: *decision,
            binding: engine.binding_position(taken.request.model(), taken.request.phase()),
            at: taken.now,
            gateway: taken.gateway,
            request: fingerprint(taken.request.id()),
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

    #[cfg(test)]
    pub(super) fn slots(&self) -> usize {
        self.records.capacity()
    }

    #[cfg(test)]
    pub(super) fn counted(&self) -> u32 {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builtins::{Best, FewestSessions, Healthy, LeastRequests, NotTried},
        decision::{Queue, Shed, ShedReason, Trace},
        door::tests::{request, GATEWAY},
        framework::{Binding, Profile, Weighted},
        Needs, PolicySpec,
    };

    fn naming_engine(generation: u64) -> Engine {
        let profiles = ["other", "prefill", "decode", "encode"].map(|name| Profile {
            name: name.into(),
            filters: vec![Box::new(NotTried), Box::new(Healthy)],
            scorers: vec![
                Weighted {
                    scorer: Box::new(LeastRequests),
                    weight: 1.0,
                },
                Weighted {
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
            gateway: GATEWAY,
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
        document.explain.sample = 1;
        let engine = document.validate().expect("a valid document");
        let taken = Taken {
            request: &request(b"queued", 7),
            gateway: GATEWAY,
            now: Instant::now(),
            left: std::time::Duration::MAX,
            pins: None,
            place: None,
        };
        let ticket = Decision::Queue(Queue {
            ticket: 1,
            max_wait_micros: 2,
            trace: Trace::default(),
        });
        let mut samples = Samples::default();
        samples.keep(0, &engine, &taken, &ticket);
        assert_eq!(samples.counted(), 0);
        assert_eq!(samples.slots(), 0);
        assert!(samples.find(0, GATEWAY, fingerprint(b"queued")).is_none());
    }
}
