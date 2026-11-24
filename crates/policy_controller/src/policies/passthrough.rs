//! The `passthrough` policy: how it is composed, and the picker only it uses.

use crate::{
    document::spec::ProfileSpec,
    engine::{
        decision::{Targets, MAX_TARGETS},
        framework::{Picker, PoolWord, TieBand, DRAW_BOUND},
        view::{Candidate, PoolFacts, RequestFacts},
    },
    policies::standard_filters,
    stages::{lifecycle::Lifecycle, pickers::target_of},
    Draw,
};

/// How the `passthrough` policy is composed: its filters, its scorers and its
/// picker.
pub(crate) fn profile() -> ProfileSpec {
    ProfileSpec {
        filter: standard_filters(),
        score: vec![],
        pick: "passthrough".into(),
        sent_ttl_ms: None,
        sent_report_delay_ms: None,
    }
}

/// The first survivor, as the host lists them.
#[derive(Debug)]
pub struct Passthrough;

impl Lifecycle for Passthrough {}

impl Picker for Passthrough {
    fn name(&self) -> &'static str {
        "passthrough"
    }

    // Picks the rows in the order handed.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn draws_first(&self) -> bool {
        true
    }

    // The pool's order is its whole rule: it reads no word of the pool.
    fn picks_alone(&self) -> bool {
        true
    }

    /// The first worker of the pool the caller's test keeps, in the pool's
    /// order: the gateway's classic `passthrough` rule. `None` where the
    /// test keeps no worker of the pool.
    fn pick_alone(
        &self,
        pool: &PoolWord<'_>,
        keep: &mut dyn FnMut(usize) -> bool,
    ) -> Option<usize> {
        (0..pool.rows).find(|&at| keep(at))
    }

    // The scan ends past the pool's last worker: it asked them all.
    fn asks_every_worker(&self) -> bool {
        true
    }

    /// The pick is the first row every filter keeps, in the host's order,
    /// and the walk goes on to the last target: the whole pass's targets,
    /// row for row. A walk that meets [`DRAW_BOUND`] refused rows stops
    /// there: with a pick in hand the alternates are the pass's as far as
    /// the walk went, and with none the pass decides.
    fn draw_first(
        &self,
        _: &RequestFacts<'_>,
        _: &PoolFacts<'_>,
        candidates: &[Candidate],
        keep: &mut dyn FnMut(&Candidate) -> bool,
        kept: &mut Vec<u32>,
        _: &mut Draw,
    ) -> bool {
        let mut refused = 0;
        for (i, c) in candidates.iter().enumerate() {
            if keep(c) {
                kept.push(i as u32);
                if kept.len() == MAX_TARGETS {
                    break;
                }
            } else {
                refused += 1;
                if refused == DRAW_BOUND {
                    break;
                }
            }
        }
        !kept.is_empty()
    }

    fn pick(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        for &i in kept.iter().take(MAX_TARGETS) {
            out.push(target_of(&candidates[i as usize]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::scorers::tests::candidate;

    #[test]
    fn passthrough_keeps_the_hosts_order() {
        let pool = [candidate(0, 9), candidate(1, 0)];
        let mut out = Targets::new();
        Passthrough.pick(
            &RequestFacts::default(),
            &pool,
            &[1, 0],
            &[0.0; 2],
            TieBand::NORMALIZED,
            &mut out,
        );
        let order: Vec<u32> = out.iter().map(|t| t.position).collect();
        assert_eq!(order, vec![1, 0]);
    }
}
