//! The host's side of the decide and lifecycle wire. An ask carries the
//! request's facts and constraints, and the answer supplies the targets and
//! the dispatch span. The controller supplies the view used to decide and
//! writes the reports it receives.

use std::time::{Duration, Instant};

use crate::{
    engine::{
        decision::{
            Decision, LiftedVeto, Pair, Pick, Queue, Shed, ShedReason, Survivors, Target, Targets,
            Trace, MAX_TARGETS, TRACE_TOP,
        },
        decision_engine::{DecideOutcome, Leased, Request},
        framework::{EncodeItems, EncodePlaced},
        view::{HashKind, Identity, Phase, RequestFacts, Session, WorkerId},
    },
    service::{
        client::MAX_HASHES,
        proto::{
            self,
            decision::Outcome,
            gateway_frame::Frame,
            identity::Form,
            released::{Reason, What},
        },
    },
    tracking::lease::{self, Lease, LeaseId, LeaseKind, Said, Side},
};

/// A request as the wire carries it, and the block hashes left out of the
/// copy. The controller builds the view it decides over.
///
/// The list is copied no further than one frame can hold. The link cuts a
/// frame to its size from the list's tail, so a longer copy would wait in
/// the link's queue for nothing. The link counts the hashes left out with
/// its own cut.
///
/// `named` is what the target source of the request's binding answered for
/// it ([`KeySources::target`](crate::document::KeySources::target)): the
/// number the request names its worker by, `Some(None)` for a text that is
/// no number, and `None` for a request that names no worker. The frame
/// states the number (`DecideRequest.named_worker`), and a text that is no
/// number as the widest one: neither reaches a worker of any list. A
/// request that names none states nothing, as the frame of a host older
/// than the field does.
pub(crate) fn request(
    request: &Request<'_>,
    named: Option<Option<u32>>,
) -> (proto::DecideRequest, usize) {
    let (facts, constraints) = (&request.facts, &request.constraints);
    let held = facts.block_hashes.len().min(MAX_HASHES);
    let asked = proto::DecideRequest {
        request_id: request.id.to_vec(),
        facts: Some(proto::RequestFacts {
            model: facts.model.to_owned(),
            namespace_seed: facts.namespace_seed,
            tenant: identity(facts.tenant, || facts.tenant.hash),
            priority: facts.priority,
            streaming: facts.streaming,
            max_output_tokens: facts.max_output_tokens,
            // A frame states one count: a request of text alone states
            // its text's bytes over four, the number a frame stated before
            // a request told its sizes apart.
            prompt_tokens: facts.prompt_size(),
            block_hashes: facts.block_hashes[..held].to_vec(),
            hash_kind: match facts.hash_kind {
                HashKind::Token => proto::HashKind::Token,
                HashKind::Text => proto::HashKind::Text,
            }
            .into(),
            session_key: identity(facts.session_key, || facts.session().get()),
            phase: match facts.phase {
                Phase::Single => proto::Phase::Single,
                Phase::Prefill => proto::Phase::Prefill,
                Phase::Decode => proto::Phase::Decode,
                Phase::Encode => proto::Phase::Encode,
            }
            .into(),
            attempt: u32::from(facts.attempt),
            tried: facts.tried.iter().map(|id| id.0).collect(),
            budget_micros: facts.budget_micros,
            headers: facts
                .headers
                .iter()
                .map(|header| proto::Header {
                    name: u32::from(header.name),
                    value: header.value.to_owned(),
                })
                .collect(),
            media: Some(proto::MediaCounts {
                images: u32::from(facts.media.images),
                audio: u32::from(facts.media.audio),
                video: u32::from(facts.media.video),
                bytes: facts.media.bytes,
            }),
            seed: facts.seed,
            prompt_text: Vec::new(),
            block_size: facts.block_size,
            client_key: facts.client_key,
            session_key_from_rid: facts.session_key_from_rid,
        }),
        constraints: Some(proto::Constraints {
            excluded: constraints.excluded.iter().map(|id| id.0).collect(),
            wire_modes: u32::from(constraints.wire_modes.bits()),
            overload_lifted: constraints.overload_lifted,
            subset_hint: constraints.subset_hint,
        }),
        again: false,
        // No link asks its decode later: the controller issues both sides.
        decode_later: false,
        decode_of: None,
        // An encode request's frame states its items ([`encode_request`]).
        encode: None,
        named_worker: named.map(|number| number.unwrap_or(u32::MAX)),
    };
    (asked, facts.block_hashes.len() - held)
}

/// An encode request as the wire carries it: its pair's one request, with
/// its `items` stated (`DecideRequest.encode`), and the block hashes left
/// out of the copy ([`request`]).
///
/// `request` is the request as the door in process takes it, at the
/// prefill phase. The frame states the encode phase, where the contract
/// reads items. A session key in its hash form crosses as the session's
/// value under that phase: the controller reads the key back under the
/// phase the facts state and decides at the prefill phase, so both doors
/// hold one session. The one formula makes the value
/// ([`RequestFacts::session`]), on the facts as the frame states them.
///
/// Each item's key crosses by the rule of a frame's key, with the key's
/// own fingerprint as its hash form. A key that is none keeps its place,
/// as an identity with no form: the answer's encoders stand in the keys'
/// order. The items' one key crosses by the same rule where the host
/// states it. One that is none is stated with no form too: it still says
/// that the items are taken as one, and its form is only what that one
/// placement is placed by.
pub(crate) fn encode_request(
    request: &Request<'_>,
    items: EncodeItems<'_>,
) -> (proto::DecideRequest, usize) {
    let stated = Request {
        facts: RequestFacts {
            phase: Phase::Encode,
            ..request.facts
        },
        ..*request
    };
    let (mut asked, left_out) = self::request(&stated, None);
    asked.encode = Some(proto::EncodeItems {
        keys: items
            .keys
            .iter()
            .map(|key| proto::Identity {
                form: form(*key, || key.hash),
            })
            .collect(),
        one: items.one.map(|one| proto::Identity {
            form: form(one, || one.hash),
        }),
        references: items.references,
        decode_images: items.decode_images,
    });
    (asked, left_out)
}

/// The hash form carries a tenant's fingerprint or the value of a session
/// within its model and phase.
fn identity(key: Identity<'_>, hash: impl FnOnce() -> u64) -> Option<proto::Identity> {
    form(key, hash).map(|form| proto::Identity { form: Some(form) })
}

/// The form a key crosses in; none for the absent key.
fn form(key: Identity<'_>, hash: impl FnOnce() -> u64) -> Option<Form> {
    if key.is_none() {
        return None;
    }
    // The controller reads an empty text as no key, and the door in process
    // reads it as a key. So the empty key crosses as its hash, and the two
    // doors decide alike.
    Some(match key.plain {
        Some(plain) if !plain.is_empty() => Form::Plain(plain.to_owned()),
        _ => Form::Hash(hash()),
    })
}

/// Why an answer cannot be read, and where in the answer it failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{path}: {why}")]
pub(crate) struct Unread {
    path: &'static str,
    why: &'static str,
}

const fn unread(path: &'static str, why: &'static str) -> Unread {
    Unread { path, why }
}

fn narrow<T: TryFrom<u32>>(number: u32, path: &'static str) -> Result<T, Unread> {
    T::try_from(number).map_err(|_| unread(path, "out of range"))
}

/// The controller's decision as the door's own record, told at `now`.
/// A pair gives the handle a prefill and a decode to report, under the
/// one id and dispatch span the door keeps in its prefill lease.
pub(crate) fn decided(told: &proto::Decision, now: Instant) -> Result<DecideOutcome, Unread> {
    let trace = told
        .trace
        .as_ref()
        .map(trace)
        .transpose()?
        .unwrap_or_default();
    let decision = match told
        .outcome
        .as_ref()
        .ok_or_else(|| unread("outcome", "missing"))?
    {
        Outcome::Pick(pick) => Decision::Pick(Pick {
            targets: targets(&pick.targets)?,
            trace,
            ..Pick::default()
        }),
        Outcome::Shed(shed) => Decision::Shed(Shed {
            reason: shed_reason(shed.reason)?,
            emptied_by: "controller",
            retry_after_secs: narrow(shed.retry_after_secs, "shed.retry_after_secs")?,
            wait_allowed: shed.wait_allowed,
            // The message has no field for the side of a pair.
            side: None,
            trace,
        }),
        Outcome::Queue(queue) => Decision::Queue(Queue {
            ticket: queue.ticket,
            max_wait_micros: queue.max_wait_micros,
            trace,
        }),
    };
    let lease = match decision.pick() {
        Some(pick) => pick_lease(&told.leases, &pick.targets, now)?,
        None if told.leases.is_empty() => None,
        None => return Err(unread("leases", "only a pick carries a lease")),
    };
    // A controller's file of leases holds the decode side whole.
    Ok(DecideOutcome {
        decision,
        lease,
        decode_session: Session::NONE,
    })
}

/// Where an encode request's answer puts its items, read onto `placed`,
/// the host's room, emptied first: the encoder of each placed key in the
/// keys' order, as the engine's entry writes them in process
/// ([`EncodePlaced::encoders`]). Only a pick places items: beside any
/// other decision `placed` stays empty and `told` is not read.
///
/// By the contract the encoders are one where the answer says `one`, else
/// one a key, and an encoder names no pair. An answer that says otherwise
/// cannot be read. Neither can one that says `one` to a request that
/// stated no key for its items as one, or that stated no item: the one key
/// is placed in the stead of items alone. Nor can a pick of a request with
/// keys that names no encoders: its host could send no item. `placed` is
/// left empty then.
pub(crate) fn placed(
    told: Option<&proto::EncodePlaced>,
    decision: &Decision,
    items: &EncodeItems<'_>,
    placed: &mut EncodePlaced,
) -> Result<(), Unread> {
    placed.clear();
    if decision.pick().is_none() {
        return Ok(());
    }
    let Some(told) = told else {
        return match items.keys {
            [] => Ok(()),
            _ => Err(unread("encode", "a pick names no encoder for its keys")),
        };
    };
    if told.one && (items.one.is_none() || items.keys.is_empty()) {
        return Err(unread("encode.one", "the request stated no items as one"));
    }
    let count = if told.one { 1 } else { items.keys.len() };
    if told.encoders.len() != count {
        return Err(unread("encode.encoders", "encoder count out of range"));
    }
    let room = placed.room(told.one);
    let read = told
        .encoders
        .iter()
        .try_for_each(|told| encoder(told).map(|encoder| room.push(encoder)));
    if read.is_err() {
        placed.clear();
    }
    read
}

/// One key's encoder as an answer names it: its worker and its rank. Its
/// row in the encode pool stays with the controller.
fn encoder(told: &proto::Target) -> Result<Target, Unread> {
    if told.pair.is_some() {
        return Err(unread("encode.encoders.pair", "an encoder names no pair"));
    }
    Ok(Target {
        position: 0,
        id: WorkerId(told.id),
        dp_rank: narrow(told.dp_rank, "encode.encoders.dp_rank")?,
        pair: None,
    })
}

fn targets(told: &[proto::Target]) -> Result<Targets, Unread> {
    if !(1..=MAX_TARGETS).contains(&told.len()) {
        return Err(unread("pick.targets", "target count out of range"));
    }
    let mut targets = Targets::new();
    for target in told {
        let pair = target
            .pair
            .map(|pair| {
                Ok(Pair {
                    decode_id: WorkerId(pair.decode_id),
                    decode_dp_rank: narrow(
                        pair.decode_dp_rank,
                        "pick.targets.pair.decode_dp_rank",
                    )?,
                    ..Pair::default()
                })
            })
            .transpose()?;
        targets.push(Target {
            position: 0,
            id: WorkerId(target.id),
            dp_rank: narrow(target.dp_rank, "pick.targets.dp_rank")?,
            pair,
        });
    }
    Ok(targets)
}

/// Checks the leases a pick carries before keeping its own side's record.
fn pick_lease(
    told: &[proto::Lease],
    targets: &Targets,
    now: Instant,
) -> Result<Option<Leased>, Unread> {
    let paired = targets.first().is_some_and(|target| target.pair.is_some());
    if targets.iter().any(|target| target.pair.is_some() != paired) {
        return Err(unread("pick.targets.pair", "mixed single and pair targets"));
    }
    match (paired, told) {
        (false, []) => Ok(None),
        (false, [single]) => leased(single, Side::Single, targets, now).map(Some),
        (true, [prefill, decode]) => {
            let prefill = leased(prefill, Side::Prefill, targets, now)?;
            let decode = leased(decode, Side::Decode, targets, now)?;
            if prefill.lease.id != decode.lease.id {
                return Err(unread("leases.id", "pair sides must share an id"));
            }
            if prefill.due != decode.due {
                return Err(unread(
                    "leases.dispatch_ttl_ms",
                    "pair sides must share a deadline",
                ));
            }
            Ok(Some(prefill))
        }
        _ => Err(unread("leases", "wrong number of leases for the pick")),
    }
}

fn leased(
    told: &proto::Lease,
    side: Side,
    targets: &Targets,
    now: Instant,
) -> Result<Leased, Unread> {
    if told.side != i32::from(proto::Side::from(side)) {
        return Err(unread("leases.side", "unexpected side for the pick"));
    }
    let target_index = narrow(told.target_index, "leases.target_index")?;
    let target = targets
        .as_slice()
        .get(usize::from(target_index))
        .ok_or_else(|| unread("leases.target_index", "no target at this index"))?;
    let booked = target
        .pair
        .filter(|_| side == Side::Decode)
        .map_or(target.id, |pair| pair.decode_id);
    let kind = match proto::LeaseKind::try_from(told.kind) {
        Ok(proto::LeaseKind::Credit) => LeaseKind::Credit,
        Ok(proto::LeaseKind::Reservation) => LeaseKind::Reservation,
        Ok(proto::LeaseKind::Unspecified) | Err(_) => {
            return Err(unread("leases.kind", "unknown lease kind"))
        }
    };
    let due = now
        .checked_add(Duration::from_millis(u64::from(told.dispatch_ttl_ms)))
        .ok_or_else(|| unread("leases.dispatch_ttl_ms", "deadline out of range"))?;
    Ok(Leased {
        lease: Lease {
            id: LeaseId(told.id),
            kind,
            side,
            target_index,
            booked,
            generation: told.generation,
            session: Session::from_raw(told.session_key),
            tokens: told.tokens,
            ..Lease::default()
        },
        due,
    })
}

fn shed_reason(number: i32) -> Result<ShedReason, Unread> {
    match proto::ShedReason::try_from(number) {
        Ok(proto::ShedReason::Saturated) => Ok(ShedReason::Saturated),
        Ok(proto::ShedReason::FlowLimit) => Ok(ShedReason::FlowLimit),
        Ok(proto::ShedReason::NoEndpoints) => Ok(ShedReason::NoEndpoints),
        Ok(proto::ShedReason::PolicyDeclined) => Ok(ShedReason::PolicyDeclined),
        Ok(proto::ShedReason::Deadline) => Ok(ShedReason::Deadline),
        Ok(proto::ShedReason::Unspecified) | Err(_) => {
            Err(unread("shed.reason", "unknown shed reason"))
        }
    }
}

fn trace(told: &proto::Trace) -> Result<Trace, Unread> {
    if told.top.len() > TRACE_TOP {
        return Err(unread("trace.top", "score count out of range"));
    }
    let mut top = [(WorkerId(0), 0.0); TRACE_TOP];
    for (entry, scored) in top.iter_mut().zip(&told.top) {
        *entry = (WorkerId(scored.worker_id), scored.score);
    }
    let survivors = match told.survivors {
        Some(u32::MAX) => return Err(unread("trace.survivors", "count out of range")),
        Some(count) => Survivors::counted(count),
        None => Survivors::NOT_COUNTED,
    };
    let lifted = match proto::LiftedVeto::try_from(told.lifted) {
        Ok(proto::LiftedVeto::None) => LiftedVeto::None,
        Ok(proto::LiftedVeto::Overload) => LiftedVeto::Overload,
        Ok(proto::LiftedVeto::Liveness) => LiftedVeto::Liveness,
        Err(_) => return Err(unread("trace.lifted", "unknown veto")),
    };
    // The mark is read first: ahead of every profile the position on the
    // wire stays zero. The writer sends neither the mark over a position
    // nor the mark's own number as a position, so both are refused.
    let profile = match (
        told.no_profile,
        narrow::<u16>(told.profile, "trace.profile")?,
    ) {
        (true, 0) => Trace::NO_PROFILE,
        (true, _) | (false, Trace::NO_PROFILE) => {
            return Err(unread("trace.profile", "out of range"));
        }
        (false, at) => at,
    };
    Ok(Trace {
        profile,
        survivors,
        top,
        top_len: u8::try_from(told.top.len()).unwrap_or_default(),
        seed: told.seed,
        generation: told.generation,
        lifted,
    })
}

/// What a host says of a lease, as the session's frame. An unnamed side
/// releases a whole pair. A single's release names its one side.
pub(crate) fn frame(lease: LeaseId, side: Side, said: Said) -> Frame {
    match said {
        Said::Dispatched(dispatch, _) => Frame::Dispatched(proto::Dispatched {
            lease_id: lease.0,
            target_index: u32::from(dispatch.target_index),
            side: proto::Side::from(dispatch.side).into(),
            since_decision_micros: dispatch.since_decision_micros,
        }),
        Said::Completed(outcome) => Frame::Completed(lease::wire::completed(lease, &outcome)),
        Said::Released(released, reason) => Frame::Released(proto::Released {
            what: Some(What::LeaseId(lease.0)),
            reason: Reason::from(reason).into(),
            side: match (released, side) {
                (Some(released), _) => proto::Side::from(released),
                (None, Side::Single) => proto::Side::Single,
                (None, Side::Prefill | Side::Decode | Side::Encode) => proto::Side::Unspecified,
            }
            .into(),
        }),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/service/client/wire/tests.rs"]
mod tests;
