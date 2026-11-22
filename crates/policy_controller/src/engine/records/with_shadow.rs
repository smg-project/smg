//! One single decision over a gateway's own data, followed by the decision
//! of its binding's shadow profile.
//!
//! The live decision is the one [`super::decide`] makes, at the engine's
//! entry that keeps its working state ([`Engine::decide_keeping`]). The
//! binding's shadow profile then decides over the same records and the same
//! scratch ([`Engine::shadow`]), before the live decision is returned: the
//! rows, the counts and the session reading are those the live decision
//! read, and no stage has heard of the request. The shadow writes its own
//! stages and the scratch, and nothing else: it issues no lease, sends no
//! event to a stage, moves no session pin and reads no worker a second
//! time. What is returned is the live decision's value.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
    time::Duration,
};

use smg_worker::identity::basic::Worker;

use super::{with_buffers, Buffers, Live, PoolRecords};
use crate::engine::{
    decision::{Decision, ShadowKind},
    framework::{release, Engine, ShadowRows},
    shadow::{Shadow, Shadowed, SHADOW_FLOOR},
    view::{Constraints, RequestFacts},
};

/// [`super::decide`] for a request whose binding names a shadow profile:
/// the live decision, and the shadow's outcome. `shadow` is what the engine
/// says of the binding ([`Engine::shadow_for`]): its position, and how its
/// shadow profile reads the affinity rows the live decision is handed.
///
/// Every worker's record is written, once ([`PoolRecords::write`]), for the
/// two decisions: the shadow profile reads the pool, so a live profile that
/// draws is not handed the drawn workers' records alone. Its draw is made
/// over the whole write, which gives the pick the short write gives.
///
/// A standing veto stated beside the records is written into them where
/// the live profile shows its rows to a plugin, as [`super::decide`]
/// writes it ([`super::vetoes::written`]). Both decisions then read the
/// same records under the same constraints.
///
/// A shadow profile is not run in two cases, and its decision is counted
/// as skipped ([`ShadowKind::Skipped`]). A request with less than
/// [`SHADOW_FLOOR`] of its stated budget left keeps that for its answer.
/// And the affinity rows the live decision was handed are not the shadow
/// profile's where that profile would be handed others: rows of another
/// warm-up ruling or another record ([`ShadowRows::Apart`]), or rows of
/// its own ([`ShadowRows::Own`]), which no caller builds for a shadow
/// here. A shadow profile that reads no affinity column runs without the
/// rows named for the live one ([`ShadowRows::Unnamed`]).
///
/// A shadow profile that panics is caught here: the live decision stands
/// as decided. The thread's buffers are kept, as behind a live decision
/// that panics: each entry writes the scratch it reads, and the keeping
/// entry forgets the kept session reading first.
pub(crate) fn decide(
    engine: &Engine,
    pool: &PoolRecords,
    workers: &[Arc<dyn Worker>],
    live: &Live<'_>,
    facts: &mut RequestFacts<'_>,
    cx: &Constraints<'_>,
    shadow: (u32, ShadowRows),
) -> (Decision, Shadow) {
    facts.now_micros = live.now_micros;
    let facts = &*facts;
    let (binding, reads) = shadow;
    let counted = |kind, first| {
        Shadow::Said(Shadowed {
            binding,
            kind,
            first,
        })
    };
    with_buffers(|buffers: &mut Buffers| {
        let Buffers { rows, scratch } = buffers;
        let (rows, pool_facts) = pool.write(workers, live, rows);
        let written;
        let cx = if cx.stalled.is_empty() {
            cx
        } else {
            written = super::vetoes::written(engine, facts.model, facts.phase, rows, cx);
            &written
        };
        let decided = engine.decide_keeping(facts, &pool_facts, rows, cx, scratch);
        // The request's own budget, as its host states it: none stated is
        // no floor to hold.
        let left = match facts.budget_micros {
            0 => Duration::MAX,
            micros => Duration::from_micros(micros.into()),
        };
        let reads = match reads {
            ShadowRows::Same | ShadowRows::Unnamed if left >= SHADOW_FLOOR => reads,
            ShadowRows::Same | ShadowRows::Unnamed | ShadowRows::Own | ShadowRows::Apart => {
                return (decided, counted(ShadowKind::Skipped, None));
            }
            // A pair's sides are no single's shadow.
            ShadowRows::Sides => return (decided, Shadow::NotDue),
        };
        let ran = catch_unwind(AssertUnwindSafe(|| {
            if reads == ShadowRows::Unnamed {
                // A profile that reads no affinity column is named no row,
                // live or as a shadow: the names would turn its pick to a
                // draw.
                let unnamed = RequestFacts {
                    named: None,
                    ..*facts
                };
                engine.shadow(&unnamed, &pool_facts, rows, cx, scratch)
            } else {
                engine.shadow(facts, &pool_facts, rows, cx, scratch)
            }
        }));
        let outcome = match ran {
            Ok(Some(decision)) => {
                let first = decision.pick().and_then(|pick| pick.targets.first());
                counted(
                    ShadowKind::of(&decided, &decision),
                    first.map(|target| target.id),
                )
            }
            Ok(None) => Shadow::NotDue,
            Err(payload) => {
                // A payload's own drop can panic: the engine's helper
                // frees it under a catch of its own.
                release(payload);
                Shadow::Faulted
            }
        };
        (decided, outcome)
    })
}
