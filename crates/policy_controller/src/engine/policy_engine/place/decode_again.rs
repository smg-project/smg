//! A pair's decode decided again and placed: one decode decision behind the
//! prefill that was sent, and the first of the decodes it names that can be
//! sent to ([`Placement::decode_again`](crate::tracking::Placement::decode_again)).
//!
//! The decision is the engine's decode entry, asked under the guard that
//! holds the pair's first decision
//! ([`DecisionGuard::decode_placed_under`]). Each of its targets names the
//! prefill that was sent, and a decode by its place in the decode list its
//! caller hands at this call. The targets are walked best first, and a
//! decode is read once, by the one read a worker gets at a first placement
//! ([`sendable`]): ready, and admitted by the route's own rule as the
//! request states it at this call. A decision that keeps where the prompt
//! went asks that of its decodes itself ([`Made`]), and the walk takes its
//! answers.

use std::{panic::resume_unwind, sync::Arc, time::Instant};

use smg_worker::identity::basic::Worker;

use super::{picked, sendable, Made, PlaceRequest, Unplaced};
use crate::{
    engine::{
        decision_engine::{DecisionRefused, DecodeRefused, WorkerLists},
        policy_engine::{encode::EncodeLists, PolicyEngine},
        records::worker_id,
    },
    tracking::DecisionGuard,
};

impl PolicyEngine {
    /// A pair's decode decided again for the placement whose guard, `held`,
    /// holds the pair's first decision: one decode decision of the engine
    /// for `request`, made again, and the first of the decodes it names
    /// that can be sent to, as the caller's decode list holds it at this
    /// call. The guard is then on that decode.
    ///
    /// # Errors
    ///
    /// The decision's shed or its queue, by its own name. A decision none
    /// of whose decodes can be sent to answers [`Unplaced::NoWorkers`], and
    /// the side it issued is given up by name. After each of them the
    /// guard stands as it stood before the call. A call out of turn is
    /// refused by the guard's word, with no decision made.
    pub(crate) fn decode_placed(
        held: &mut DecisionGuard<&'static Self>,
        request: &PlaceRequest<'_>,
    ) -> Result<Arc<dyn Worker>, Unplaced> {
        // A guard over a link asks no decode later: none is placed here.
        let Some(&engine) = held.door() else {
            let refused = DecisionRefused::Decode(DecodeRefused::NotChosen);
            return Err(Unplaced::Refused(refused));
        };
        let mut made = Made {
            decode_alone: true,
            ..Made::new(engine, request, &[])
        };
        // A request that hands no pair's lists is decided over no worker.
        // An encode request's three lists are its pair's two for a decode
        // decided again, as they are for the engine's decode entry.
        let decodes = match request.lists {
            WorkerLists::Pair(lists) | WorkerLists::Encode(EncodeLists { pair: lists, .. }) => {
                Some(lists.decode)
            }
            WorkerLists::None | WorkerLists::Single(_) => None,
        };
        let mut taken = None;
        let placed = held.decode_placed_under(&made, Instant::now(), |target| {
            let place = target.pair.map(|pair| pair.decode_position);
            let worker = place.and_then(|place| decodes?.workers.get(place as usize));
            debug_assert_eq!(
                worker.map(|worker| worker_id(worker.as_ref())),
                worker.and(target.pair).map(|pair| pair.decode_id),
                "the decode stands at the target's position"
            );
            // A decode the decision asked is taken by the answer it was
            // given there. A rule that panicked there holds no decode.
            let kept = place.and_then(|place| made.decodes.get().answer(place));
            let sends = |worker: &&Arc<dyn Worker>| {
                made.panicked.get().is_none()
                    && kept.unwrap_or_else(|| sendable(request, worker.as_ref()))
            };
            taken = worker.filter(sends).cloned();
            taken.is_some()
        });
        // The route's rule panicked inside the decision, at the question
        // of where the prompt goes. The walk held no decode, the guard
        // stands as it stood before the call, and the panic is the
        // caller's.
        if let Some(panic) = made.panicked.take() {
            resume_unwind(panic);
        }
        match placed.map_err(Unplaced::Refused)? {
            None => taken.ok_or(Unplaced::NoWorkers),
            // A pick none of whose decodes holds names workers and no
            // reason.
            Some(unplaced) => Err(picked(&unplaced).err().unwrap_or(Unplaced::NoWorkers)),
        }
    }
}
