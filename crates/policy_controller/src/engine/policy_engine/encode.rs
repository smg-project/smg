//! An encode request's decision over three lists of records written for
//! it: its pair and the encoder of each of its media items, in the one call
//! the engine decides them in
//! ([`Engine::decide_encode_pair`](crate::engine::Engine::decide_encode_pair)).
//!
//! No affinity column and no signal cell stands beside the encoders, and
//! no ring is lent for them: an encode profile's stage that places by a
//! ring places by its own.

use std::{sync::Arc, time::Instant};

use smg_worker::load::state::LoadSnapshot;

use super::{
    pair::{not_narrowed_by, stamp, Buffers, PairLists, BUFFERS},
    PoolMembers,
};
use crate::{
    document::spec::AppliedSpec,
    engine::{
        decision::Decision,
        decision_engine::Request,
        framework::{EncodeItems, EncodePlaced, EncodePool, EncoderFacts, PairPools},
        records::{DecisionClock, Live},
    },
};

#[cfg(test)]
mod tests;

/// An encode request's three lists as their host holds them for one
/// decision: the pair's two lists, the encoders, and what the encode side
/// reads of the workers beside their records. A decision names each placed
/// key's encoder by its place in the encoder list, so the host finds it
/// with no lookup by id.
#[derive(Clone, Copy, Debug)]
pub struct EncodeLists<'a> {
    /// The pair's two lists, with their pair classes and the host's reads
    /// of its admission.
    pub pair: PairLists<'a>,
    /// The encoders.
    pub encoders: &'a PoolMembers,
    /// What the encode side reads of each encoder beside its record, by its
    /// place in the encoder list
    /// ([`EncodeClasses::encoder_facts`](crate::store::EncodeClasses::encoder_facts)).
    pub facts: &'a [EncoderFacts],
    /// The encode class each prefill names, one entry a prefill, by its
    /// place in the prefill list
    /// ([`EncodeClasses::prefill_class`](crate::store::EncodeClasses::prefill_class)).
    pub prefill_classes: &'a [u32],
}

impl EncodeLists<'_> {
    /// One decision for an encode request at `now`: its pair, and the
    /// encoder of each key of `items`, which `placed` takes. One record is
    /// written for each worker of the three lists, in its list's order, as
    /// the workers stand at the call, with `loads` as a pair's decision
    /// reads them; the pair's records are stamped as a pair's
    /// ([`PairLists::decide`]). Whether an encoder or a decode takes images
    /// is on its record as its list's records were made.
    ///
    /// `request` is the pair's, as its host made it for the prefill phase.
    /// This call's instant is written into its facts: `now` on `clock`.
    ///
    /// A pair's lists are narrowed by no demand, no subset hint and no
    /// places: a request that states one is shed or declined as its pair's
    /// own decision answers it, and no key is placed. One side of the pair
    /// alone is no pair, and the engine says so behind the encoders: a
    /// request that no encoder is ready for is shed under the encode side's
    /// stage. Facts or classes of another length than their list are
    /// declined by the engine under the column's name.
    ///
    /// Nothing is issued for the decision and no stage hears of it.
    #[expect(
        clippy::too_many_arguments,
        reason = "a pair's inputs and the encode side's"
    )]
    pub fn decide(
        &self,
        spec: &AppliedSpec,
        loads: Option<&Arc<LoadSnapshot>>,
        clock: &DecisionClock,
        now: Instant,
        request: &mut Request<'_>,
        items: EncodeItems<'_>,
        placed: &mut EncodePlaced,
    ) -> Decision {
        let pair = &self.pair;
        if not_narrowed_by(&request.constraints).is_some() {
            placed.clear();
            return pair.decide(spec, loads, clock, now, request, None);
        }
        // One value for the three lists' records, as a pair's decision
        // takes one for its two.
        let live = Live::at(spec.engine(), clock, now, loads, None);
        request.facts.now_micros = live.now_micros();
        let (facts, cx) = (&request.facts, &request.constraints);
        let mut decide = |buffers: &mut Buffers| {
            let [prefills, decodes] = &mut buffers.rows;
            let (workers, records) = (pair.prefill.workers(), pair.prefill.records(spec, clock));
            let (prefills, pool) = records.write(workers, &live, prefills);
            let (workers, records) = (pair.decode.workers(), pair.decode.records(spec, clock));
            let (decodes, _) = records.write(workers, &live, decodes);
            stamp(prefills, decodes, pair.classes, pair.admission);
            let (workers, records) = (self.encoders.workers(), self.encoders.records(spec, clock));
            let (encoders, _) = records.write(workers, &live, &mut buffers.encoders);
            let pools = PairPools {
                prefills,
                decodes,
                table: pair.classes.table(),
                ..PairPools::default()
            };
            let encode = EncodePool {
                encoders,
                facts: self.facts,
                prefill_classes: self.prefill_classes,
                ..EncodePool::default()
            };
            let (engine, scratch) = (spec.engine(), &mut buffers.scratch);
            engine.decide_encode_pair(
                facts, &pool, pools, encode, items, cx, cx, cx, scratch, placed,
            )
        };
        BUFFERS.with(|buffers| match buffers.try_borrow_mut() {
            Ok(mut buffers) => decide(&mut buffers),
            // A stage asked for a decision from inside its own, on this
            // thread: that decision takes buffers of its own.
            Err(_) => decide(&mut Buffers::default()),
        })
    }
}
