//! The recent decisions of one seat. The seat's holder counts and writes,
//! and a reader copies a matching record while holding that seat.

use std::{collections::VecDeque, time::Instant};

use super::{Asked, Taken};
use crate::{decision::Decision, framework::Engine, store::GatewayId, view::fingerprint};

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
        decision::{Queue, Trace},
        door::tests::{request, GATEWAY},
        PolicySpec,
    };

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
