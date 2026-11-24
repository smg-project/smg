//! Telling a document's stages of a request's dispatch and end.
//!
//! An engine that keeps no file of leases is lent back, at each thing said
//! of a lease ([`Said`]), what the decision's holder kept of it
//! ([`HeldDecision`]): the pick, the lease as issued and the target taken.
//! The functions here make, from that, the lease and the target each stage
//! is told under: [`single`] for a single request and [`pair`] for a pair,
//! a side at a time. They take no lock and keep nothing.
//!
//! A single request its binding's sessions rule keeps from the profile is
//! told to the session table at its dispatch, and to no stage ahead of its
//! end ([`stages_hear`]).

use std::time::Instant;

use super::{
    decision::{Decision, Target},
    decision_engine::HeldDecision,
    framework::Engine,
    view::Session,
};
use crate::{
    stages::Lifecycle,
    tracking::lease::{Lease, Said, Side},
};

/// Whether the stages hear the issue and the dispatch of a single
/// request's `decision`, and the profile counts the request among its
/// sends: of every pick but one its binding's sessions rule placed alone
/// and keeps from the profile ([`RulePlaced::Untold`]). The rule places a
/// request alone where it follows a pin of the key, and where its own
/// assignment places the key.
///
/// Under that word no stage placed such a request, and none counts it
/// among its sends: the session table hears its dispatch, where the key's
/// pin moves, and no stage hears a word of it ahead of its end. A
/// gateway's sticky map keeps a request it places from the gateway's
/// policy in the same way. A stage that prices what it sent a worker
/// still reads such a request in the count of requests its host holds on
/// that worker. Under the rule's other word, the default, the request is
/// the profile's own and every word of it is told.
///
/// What is kept of where a request's prompt went asks the same: the
/// record of sent prefixes takes no entry, and a lent token tree stores no
/// prompt, behind a pick the stages do not hear of
/// (`policy_engine::overlap`). A pin the keyed entry followed is handed to
/// that lookup only where the stages hear of it (`policy_engine::single`).
/// So the rule's word has this one reader, at the words of a lease and at
/// those writes.
///
/// A controller asks where it files a pick's lease, and keeps the answer on
/// the lease's record: the dispatch it tells off its file later follows it
/// (`LeaseRecord::untold`).
///
/// A pair's sides are not read here: each side keeps a verdict of its own,
/// and a pick's one word does not say which side the rule placed.
///
/// [`RulePlaced::Untold`]: crate::document::spec::RulePlaced::Untold
#[inline]
pub(crate) fn stages_hear(decision: &Decision) -> bool {
    decision.pick().is_none_or(|pick| !pick.untold())
}

/// Tells `engine`'s stages one word of a single's lease, from what the
/// request's guard lends back, as a file of leases tells it: a dispatch
/// under the lease as issued with the index the word names, on the target
/// taken, by its id and rank; an end under the lease moved to that target;
/// a release under the lease as issued. The dispatch of a pick its
/// sessions rule keeps from the profile is told to the session table and
/// to no stage ([`stages_hear`]).
pub(crate) fn single(engine: &Engine, held: &HeldDecision<'_>, said: Said) {
    match said {
        Said::Dispatched(dispatch, _) => {
            let lease = Lease {
                target_index: dispatch.target_index,
                ..*held.lease
            };
            let target = Target {
                id: held.target.id,
                dp_rank: held.target.dp_rank,
                ..Target::default()
            };
            if stages_hear(held.decision) {
                engine.on_dispatched(
                    &lease,
                    &target,
                    dispatch.side,
                    dispatch.since_decision_micros,
                );
            } else {
                engine.dispatched_table(&lease, &target, dispatch.side);
            }
        }
        Said::Completed(outcome) => {
            let lease = Lease {
                target_index: held.index,
                booked: held.target.id,
                ..*held.lease
            };
            engine.on_completed(&lease, &outcome);
        }
        Said::Released(_, reason) => engine.on_released(held.lease, reason),
    }
}

/// The decode side's lease as issued beside `prefill`, the prefill side's:
/// it books the decode of `first`, the pick's first target, and states the
/// decode side's own session verdict, which that target's pair keeps
/// ([`Pair::decode_verdict`](super::decision::Pair::decode_verdict)).
/// `None` where that target names no pair.
pub(crate) fn decode_as_issued(prefill: &Lease, first: &Target, session: Session) -> Option<Lease> {
    let pair = first.pair?;
    Some(Lease {
        side: Side::Decode,
        target_index: 0,
        booked: pair.decode_id,
        session,
        session_verdict: pair.decode_verdict.on(pair.decode_id),
        ..*prefill
    })
}

/// Tells the stages what was `said` of a pair's lease, a side at a time.
///
/// - A dispatch is told on the side it names: the send counts, the session
///   table, then the stages. The lease told is the side's as issued, under
///   the index the dispatch took, and the target is the worker the side
///   went to. A decode that stood chosen
///   ([`HeldDecode::chosen`](super::decision_engine::HeldDecode::chosen))
///   goes out as its pick's with no call:
///   its issue is told first, dated at the dispatch's instant by `micros`,
///   under the decode it goes to.
/// - A completion ends the side its outcome names. The prefill's is told
///   under its lease as issued: every target of a pair's pick names the one
///   prefill. The decode's names the decode it went to.
/// - A release ends the side it names, as issued, and with none both sides:
///   the prefill's first.
///
/// A word on a side a pair's lease does not hold tells nothing, and so
/// does an end of a decode that stood chosen: no stage heard of that side.
/// A send count takes a send at a rank it does not know yet, as it does of
/// any host that tells a dispatch whole.
///
/// Every word behind a side's issue carries the instant of the pair's first
/// decision, as a lease on file does. The decode's end names the index its
/// guard holds, which is the one its dispatch took.
pub(crate) fn pair(
    engine: &Engine,
    held: &HeldDecision<'_>,
    said: Said,
    micros: impl Fn(Instant) -> u64,
) {
    let prefill = held.lease;
    let first = held.decision.pick().and_then(|pick| pick.targets.first());
    // The decode side as its host lends it with a pair's lease, under the
    // three sizes it was issued under.
    let decode = first
        .and_then(|first| decode_as_issued(prefill, first, held.decode.session))
        .map(|issued| Lease {
            tokens: held.decode.tokens,
            chars: held.decode.chars,
            bytes: held.decode.bytes,
            ..issued
        });
    let chosen = held.decode.chosen;
    match said {
        Said::Dispatched(dispatch, at) => {
            let taken = held.target;
            let (lease, target) = match dispatch.side {
                Side::Prefill => (
                    *prefill,
                    Target {
                        id: taken.id,
                        dp_rank: taken.dp_rank,
                        ..Target::default()
                    },
                ),
                Side::Decode => {
                    let (Some(decode), Some(pair)) = (decode, taken.pair) else {
                        return;
                    };
                    let target = Target {
                        id: pair.decode_id,
                        dp_rank: pair.decode_dp_rank,
                        ..Target::default()
                    };
                    (decode, target)
                }
                Side::Single | Side::Encode => return,
            };
            let lease = Lease {
                target_index: dispatch.target_index,
                ..lease
            };
            let since = dispatch.since_decision_micros;
            engine.dispatched_sent(&lease, &target, dispatch.side, since, || true);
            engine.dispatched_table(&lease, &target, dispatch.side);
            if dispatch.side == Side::Decode && chosen {
                // No call issued the side: this dispatch books it, so the
                // stages hear its issue first, under the worker that serves.
                let sent = Lease {
                    booked: target.id,
                    ..lease
                };
                let issued = Lease {
                    issued_micros: micros(at),
                    ..sent
                };
                engine.on_decision_issued(&issued, held.decision);
                engine.dispatched_stages(&sent, &target, dispatch.side, since);
            } else {
                engine.dispatched_stages(&lease, &target, dispatch.side, since);
            }
        }
        Said::Completed(outcome) => match outcome.side {
            Side::Prefill => engine.on_completed(prefill, &outcome),
            Side::Decode => {
                let (Some(decode), Some(pair)) = (decode, held.target.pair) else {
                    return;
                };
                if chosen {
                    return;
                }
                let ended = Lease {
                    target_index: held.index,
                    booked: pair.decode_id,
                    ..decode
                };
                engine.on_completed(&ended, &outcome);
            }
            Side::Single | Side::Encode => {}
        },
        Said::Released(side, reason) => {
            if matches!(side, None | Some(Side::Prefill)) {
                engine.on_released(prefill, reason);
            }
            let ends_decode = matches!(side, None | Some(Side::Decode)) && !chosen;
            if let Some(decode) = decode.filter(|_| ends_decode) {
                engine.on_released(&decode, reason);
            }
        }
    }
}

#[cfg(test)]
mod tests;
