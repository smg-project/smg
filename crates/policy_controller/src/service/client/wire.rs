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
        view::{HashKind, Identity, Phase, Session, WorkerId},
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
pub(crate) fn request(request: &Request<'_>) -> (proto::DecideRequest, usize) {
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
    };
    (asked, facts.block_hashes.len() - held)
}

/// The hash form carries a tenant's fingerprint or the value of a session
/// within its model and phase.
fn identity(key: Identity<'_>, hash: impl FnOnce() -> u64) -> Option<proto::Identity> {
    if key.is_none() {
        return None;
    }
    // The controller reads an empty text as no key, and the door in process
    // reads it as a key. So the empty key crosses as its hash, and the two
    // doors decide alike.
    let form = match key.plain {
        Some(plain) if !plain.is_empty() => Form::Plain(plain.to_owned()),
        _ => Form::Hash(hash()),
    };
    Some(proto::Identity { form: Some(form) })
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
mod tests {
    use std::time::{Duration, Instant};

    use super::{decided, frame, request};
    use crate::{
        engine::{
            decision,
            decision::{
                Decision, LiftedVeto, Pair, Pick, Queue, Shed, ShedReason, Survivors, Target,
                Targets, Trace,
            },
            decision_engine::Request,
            view,
            view::{
                Constraints, HashKind, Header, Identity, MediaCounts, Phase, RequestFacts, WorkerId,
            },
        },
        service::{
            client::MAX_HASHES,
            controller::wire::{tests::made, Beside, Checked, Decided, Leased, Lists},
            proto::{
                self, decide_answer::Answer, decision::Outcome, gateway_frame::Frame,
                identity::Form,
            },
        },
        tracking::{
            lease,
            lease::{Lease, LeaseId, ReleaseReason, Said, Side, TransportClass},
        },
    };

    /// What a controller has left of a request's budget where it makes the
    /// request of a frame.
    const LEFT: Duration = Duration::from_micros(1_500);

    /// Every field a host states of a request crosses the wire, and the
    /// controller makes its request of the frame. Four things of that
    /// request are the controller's own. Its budget is what the controller
    /// has left. It says no wire: the wires the host states are read on
    /// the frame, where it is checked. Its block hashes are cut at the
    /// depth the deciding document reads, and of a subset hint it states
    /// a 0 alone. The case's three hashes and its hint of 0 are inside
    /// both, and the frame is read for each as it crossed.
    #[test]
    fn the_controller_reads_every_request_field_as_the_host_stated_it() {
        let mut asked = Request {
            id: b"a request",
            facts: RequestFacts {
                model: "model-a",
                namespace_seed: 11,
                tenant: Identity::plain("tenant-a"),
                priority: -100,
                streaming: true,
                max_output_tokens: 12,
                prompt_tokens: 13,
                block_hashes: &[14, 15, 16],
                hash_kind: HashKind::Text,
                session_key: Identity::hashed(17),
                phase: Phase::Decode,
                attempt: 2,
                tried: &[WorkerId(18), WorkerId(19)],
                budget_micros: 2_000,
                headers: &[
                    Header {
                        name: 255,
                        value: "last",
                    },
                    Header {
                        name: 0,
                        value: "first",
                    },
                ],
                media: MediaCounts {
                    images: 20,
                    audio: 21,
                    video: 22,
                    bytes: 23,
                },
                seed: 24,
                block_size: 25,
                client_key: 29,
                ..RequestFacts::default()
            },
            constraints: Constraints {
                excluded: &[WorkerId(26), WorkerId(27)],
                wire_modes: crate::engine::WireModes::HTTP.union(crate::engine::WireModes::GRPC),
                overload_lifted: true,
                subset_hint: Some(0),
                ..Constraints::default()
            },
        };
        for (tenant, session, phase, hash_kind) in [
            (
                Identity::plain("tenant-a"),
                Identity::hashed(17),
                Phase::Decode,
                HashKind::Text,
            ),
            (
                Identity::hashed(28),
                Identity::plain("session-a"),
                Phase::Single,
                HashKind::Token,
            ),
            (
                Identity::NONE,
                Identity::NONE,
                Phase::Prefill,
                HashKind::Token,
            ),
            (
                Identity::NONE,
                Identity::NONE,
                Phase::Encode,
                HashKind::Text,
            ),
        ] {
            asked.facts.tenant = tenant;
            asked.facts.session_key = session;
            asked.facts.phase = phase;
            asked.facts.hash_kind = hash_kind;
            let want = &asked.facts;
            let (sent, left_out) = request(&asked);
            assert_eq!(left_out, 0);
            let facts = sent.facts.as_ref().expect("request facts");
            assert_eq!(facts.client_key, want.client_key);
            if tenant.is_none() {
                assert!(facts.tenant.is_none());
            }
            if session.is_none() {
                assert!(facts.session_key.is_none());
            }
            if session.plain.is_none() && !session.is_none() {
                let key = facts.session_key.as_ref();
                assert_eq!(
                    key.and_then(|key| key.form.as_ref()),
                    Some(&Form::Hash(want.session().get()))
                );
            }
            let checked = Checked::new(sent).expect("the controller reads the request");
            assert_eq!(checked.id(), asked.id);
            assert_eq!(checked.budget().as_micros(), u128::from(want.budget_micros));
            assert!(!checked.has_text());
            assert!(!checked.again());
            assert_eq!(checked.wires(), Some(asked.constraints.wire_modes));
            assert_eq!(checked.block_hashes(), want.block_hashes);
            assert_eq!(checked.subset(), asked.constraints.subset_hint);
            made(
                &checked,
                &mut Lists::default(),
                LEFT,
                |facts, constraints| {
                    assert_eq!(facts.model, want.model);
                    assert_eq!(facts.namespace_seed, want.namespace_seed);
                    assert_eq!(facts.tenant, want.tenant);
                    assert_eq!(facts.priority, want.priority);
                    assert_eq!(facts.streaming, want.streaming);
                    assert_eq!(facts.max_output_tokens, want.max_output_tokens);
                    assert_eq!(facts.prompt_tokens, want.prompt_tokens);
                    assert_eq!(facts.block_hashes, want.block_hashes);
                    assert_eq!(facts.hash_kind, want.hash_kind);
                    assert_eq!(facts.session_key, want.session_key);
                    assert_eq!(facts.phase, want.phase);
                    assert_eq!(facts.attempt, want.attempt);
                    assert_eq!(facts.tried, want.tried);
                    assert_eq!(u128::from(facts.budget_micros), LEFT.as_micros());
                    assert_eq!(facts.headers, want.headers);
                    assert_eq!(facts.media, want.media);
                    assert_eq!(facts.seed, want.seed);
                    assert_eq!(facts.block_size, want.block_size);
                    assert_eq!(facts.client_key, want.client_key);
                    assert_eq!(constraints.excluded, asked.constraints.excluded);
                    assert_eq!(constraints.wire_modes, crate::engine::WireModes::NONE);
                    assert_eq!(
                        constraints.overload_lifted,
                        asked.constraints.overload_lifted
                    );
                    assert_eq!(constraints.subset_hint, asked.constraints.subset_hint);
                },
            );
        }
    }

    /// A frame states one count. A request of text alone states its text's
    /// length and no count of tokens: it is sent as its bytes over four,
    /// the count a frame stated before a request told its sizes apart. A
    /// text of two-byte characters is sent by its bytes, and so is a text
    /// whose characters nothing counted. A stated count of tokens is sent
    /// as stated, whatever length stands beside it.
    #[test]
    fn a_frame_states_the_length_of_a_request_of_text_as_its_count() {
        let sent = |prompt_tokens, (prompt_chars, prompt_bytes): (u32, u32)| {
            let asked = Request {
                id: b"a request",
                facts: RequestFacts {
                    model: "model-a",
                    prompt_tokens,
                    prompt_chars,
                    prompt_bytes,
                    ..RequestFacts::default()
                },
                constraints: Constraints::default(),
            };
            let facts = request(&asked).0.facts.expect("the frame's facts");
            facts.prompt_tokens
        };
        // 52 one-byte characters, 26 two-byte characters, a text of 52
        // bytes under a document that counts no characters, a count of
        // tokens beside a text, and nothing.
        let texts = [sent(0, (52, 52)), sent(0, (26, 52)), sent(0, (0, 52))];
        assert_eq!(texts, [13, 13, 13]);
        assert_eq!((sent(13, (400, 400)), sent(0, (0, 0))), (13, 0));
    }

    /// A list longer than a frame holds is copied no further than a frame's
    /// worth, and the rest is counted for the link's cut.
    #[test]
    fn a_request_copies_no_more_hashes_than_a_frame_holds() {
        let blocks = vec![7; MAX_HASHES + 1_000];
        let asked = Request {
            id: b"a request",
            facts: RequestFacts {
                model: "model-a",
                block_hashes: &blocks,
                budget_micros: 2_000,
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        };
        let (sent, left_out) = request(&asked);
        let copied = sent.facts.expect("request facts").block_hashes;
        assert_eq!((copied.len(), copied.capacity()), (MAX_HASHES, MAX_HASHES));
        assert_eq!(left_out, 1_000);
    }

    /// An empty text is a key at the door in process, and the controller
    /// reads an empty text as no key. The empty key crosses as its hash, so
    /// the controller decides over the same tenant and session.
    #[test]
    fn an_empty_plain_key_reads_over_the_link_as_the_key_it_is() {
        let asked = Request {
            id: b"a request",
            facts: RequestFacts {
                model: "model-a",
                tenant: Identity::plain(""),
                session_key: Identity::plain(""),
                budget_micros: 2_000,
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        };
        let want = &asked.facts;
        assert!(!want.tenant.is_none() && !want.session_key.is_none());
        let (sent, _) = request(&asked);
        let checked = Checked::new(sent).expect("the controller reads the request");
        made(&checked, &mut Lists::default(), LEFT, |facts, _| {
            assert_eq!(facts.tenant.hash, want.tenant.hash);
            assert_eq!(facts.session().get(), want.session().get());
        });
    }

    fn trace(survivors: Survivors) -> Trace {
        Trace {
            profile: 7,
            survivors,
            top: [
                (WorkerId(58), 0.8),
                (WorkerId(42), 0.3),
                (WorkerId(17), 0.1),
            ],
            top_len: 3,
            seed: 27,
            generation: 13,
            lifted: LiftedVeto::Overload,
        }
    }

    fn pick(count: usize, paired: bool, survivors: Survivors) -> Decision {
        let mut targets = Targets::new();
        for (id, rank) in [(58, 7), (42, 3), (17, 1), (20, 5)].into_iter().take(count) {
            targets.push(Target {
                id: WorkerId(id),
                dp_rank: rank,
                pair: paired.then_some(Pair {
                    decode_id: WorkerId(id + 100),
                    decode_dp_rank: rank + 10,
                    ..Pair::default()
                }),
                ..Target::default()
            });
        }
        Decision::Pick(Pick {
            targets,
            trace: trace(survivors),
            ..Pick::default()
        })
    }

    fn told(decision: Decision, lease: Option<Leased>) -> proto::Decision {
        let answer = Decided::Decision {
            id: vec![7],
            term: 3,
            decision,
            lease,
        }
        .frame();
        let Some(Answer::Decision(told)) = answer.answer else {
            panic!("expected a decision, got {answer:?}");
        };
        told
    }

    #[test]
    fn picks_keep_their_target_order_and_book_the_target_the_lease_names() {
        let now = Instant::now();
        for (count, paired, index, booked, survivors) in [
            (1, false, 0, 58, Survivors::NOT_COUNTED),
            (decision::MAX_TARGETS, false, 2, 17, Survivors::counted(4)),
            (2, true, 1, 42, Survivors::counted(2)),
        ] {
            let decision = pick(count, paired, survivors);
            let lease = Lease {
                id: LeaseId(17),
                side: if paired { Side::Prefill } else { Side::Single },
                target_index: index,
                booked: WorkerId(booked),
                generation: 13,
                session: RequestFacts {
                    model: "model-a",
                    session_key: Identity::hashed(17),
                    ..RequestFacts::default()
                }
                .session(),
                tokens: 165,
                ..Lease::default()
            };
            let told = told(
                decision,
                Some(Leased {
                    lease,
                    beside: if paired {
                        Beside::Decode(lease.session)
                    } else {
                        Beside::Alone
                    },
                    left: Duration::from_millis(17),
                    due: now + Duration::from_secs(1),
                }),
            );
            assert_eq!(told.leases.len(), if paired { 2 } else { 1 });
            let read = decided(&told, now).expect("the wire's pick fits the record");
            assert_eq!(read.decision, decision);
            let leased = read.lease.expect("a lease was told");
            assert_eq!(leased.lease, lease);
            assert_eq!(leased.due, now + Duration::from_millis(17));
            if paired {
                let targets = &read.decision.pick().expect("a pair pick").targets;
                let decode = super::leased(&told.leases[1], Side::Decode, targets, now)
                    .expect("the decode lease");
                assert_eq!(decode.lease.booked, WorkerId(booked + 100));
            }
        }
    }

    #[test]
    fn a_pick_keeps_the_lifted_liveness_veto() {
        let Decision::Pick(mut pick) = pick(1, false, Survivors::counted(1)) else {
            panic!("a pick");
        };
        pick.trace.lifted = LiftedVeto::Liveness;
        let decision = Decision::Pick(pick);
        let read = decided(&told(decision, None), Instant::now()).expect("a pick");
        assert_eq!(read.decision, decision);
        assert!(read.lease.is_none());
    }

    #[test]
    fn a_shed_keeps_its_reason_and_pacing_and_a_queue_keeps_its_ticket() {
        let now = Instant::now();
        let shed = Shed {
            reason: ShedReason::Saturated,
            emptied_by: "not-overloaded",
            retry_after_secs: 3,
            wait_allowed: true,
            trace: trace(Survivors::counted(0)),
        };
        for reason in [
            ShedReason::Saturated,
            ShedReason::FlowLimit,
            ShedReason::NoEndpoints,
            ShedReason::PolicyDeclined,
            ShedReason::Deadline,
        ] {
            let shed = Shed { reason, ..shed };
            let read = decided(&told(Decision::Shed(shed), None), now).expect("a shed");
            assert_eq!(
                read.decision,
                Decision::Shed(Shed {
                    emptied_by: "controller",
                    ..shed
                })
            );
            assert!(read.lease.is_none());
        }
        let queue = Decision::Queue(Queue {
            ticket: 75,
            max_wait_micros: 15_000,
            trace: Trace {
                lifted: LiftedVeto::None,
                ..shed.trace
            },
        });
        let read = decided(&told(queue, None), now).expect("a queue");
        assert_eq!(read.decision, queue);
        assert!(read.lease.is_none());
    }

    /// A shed made ahead of every profile goes out with the mark over a
    /// zero position. Its record names no profile, and a decision under the
    /// first profile still names that one.
    #[test]
    fn a_shed_made_ahead_of_every_profile_names_none_over_a_link() {
        let now = Instant::now();
        let ahead = decision::shed(ShedReason::NoEndpoints, "wires", 0, 6);
        let wire = told(ahead, None);
        let sent = wire.trace.as_ref().expect("a trace");
        assert!(sent.no_profile);
        assert_eq!(sent.profile, 0);
        let read = decided(&wire, now).expect("a shed");
        assert_eq!(read.decision.trace().made_under(), None);

        let first = Decision::Shed(Shed {
            reason: ShedReason::Saturated,
            emptied_by: "not-overloaded",
            retry_after_secs: 3,
            wait_allowed: true,
            trace: Trace {
                profile: 0,
                ..trace(Survivors::counted(0))
            },
        });
        let read = decided(&told(first, None), now).expect("a shed");
        assert_eq!(read.decision.trace().made_under(), Some(0));
    }

    #[test]
    fn a_lease_keeps_the_kind_it_carries() {
        use crate::tracking::lease::LeaseKind;

        let now = Instant::now();
        let mut told = told(pick(1, false, Survivors::counted(1)), None);
        told.leases = vec![proto::Lease {
            id: 73,
            side: proto::Side::Single.into(),
            kind: proto::LeaseKind::Reservation.into(),
            ..proto::Lease::default()
        }];
        let leased = decided(&told, now).expect("a pick").lease.expect("a lease");
        assert_eq!(leased.lease.kind, LeaseKind::Reservation);
    }

    #[test]
    fn a_pair_requires_a_prefill_and_decode_under_one_id_and_span() {
        let now = Instant::now();
        let good = told(
            pick(2, true, Survivors::counted(2)),
            Some(Leased {
                lease: Lease {
                    id: LeaseId(17),
                    side: Side::Prefill,
                    ..Lease::default()
                },
                beside: Beside::Decode(view::Session::NONE),
                left: Duration::from_secs(1),
                due: now + Duration::from_secs(1),
            }),
        );
        let checks: [(_, fn(&mut proto::Decision)); 7] = [
            ("leases", |bad| {
                bad.leases.pop();
            }),
            ("leases", |bad| bad.leases.push(bad.leases[0])),
            ("leases.side", |bad| {
                bad.leases[0].side = proto::Side::Encode.into();
            }),
            ("leases.side", |bad| {
                bad.leases[1].side = proto::Side::Single.into();
            }),
            ("leases.id", |bad| bad.leases[1].id += 1),
            ("leases.dispatch_ttl_ms", |bad| {
                bad.leases[1].dispatch_ttl_ms += 1;
            }),
            ("pick.targets.pair", |bad| {
                let Some(Outcome::Pick(pick)) = &mut bad.outcome else {
                    panic!("a pick");
                };
                pick.targets[1].pair = None;
            }),
        ];
        for (path, edit) in checks {
            let mut bad = good.clone();
            edit(&mut bad);
            let error = decided(&bad, now).expect_err("an unreadable pair");
            assert_eq!(error.path, path);
        }
    }

    #[test]
    fn a_dispatch_deadline_outside_the_hosts_clock_is_unreadable() {
        let now = Instant::now();
        let mut last = now;
        for bit in (0..64).rev() {
            if let Some(further) = last.checked_add(Duration::from_secs(1 << bit)) {
                last = further;
            }
        }
        assert!(last.checked_add(Duration::from_secs(1)).is_none());
        let told = told(
            pick(1, false, Survivors::counted(1)),
            Some(Leased {
                lease: Lease {
                    id: LeaseId(17),
                    ..Lease::default()
                },
                beside: Beside::Alone,
                left: Duration::from_secs(1),
                due: now + Duration::from_secs(1),
            }),
        );
        let error = decided(&told, last).expect_err("the span runs past the clock's range");
        assert_eq!(error.path, "leases.dispatch_ttl_ms");
    }

    #[test]
    fn unreadable_fields_are_named_by_path() {
        let now = Instant::now();
        let good = told(
            pick(2, false, Survivors::counted(2)),
            Some(Leased {
                lease: Lease {
                    id: LeaseId(17),
                    ..Lease::default()
                },
                beside: Beside::Alone,
                left: Duration::from_secs(1),
                due: now + Duration::from_secs(1),
            }),
        );
        let refuses = |bad: proto::Decision, path| {
            let error = decided(&bad, now).expect_err("the field does not fit the record");
            assert_eq!(error.path, path);
            assert!(!error.why.is_empty());
        };
        let mut bad = good.clone();
        bad.leases[0].target_index = 256;
        refuses(bad, "leases.target_index");
        let mut bad = good.clone();
        bad.leases[0].target_index = 2;
        refuses(bad, "leases.target_index");
        let mut bad = good.clone();
        bad.leases[0].kind = 0;
        refuses(bad, "leases.kind");
        let mut bad = good.clone();
        bad.leases[0].side = 0;
        refuses(bad, "leases.side");
        let mut bad = good.clone();
        bad.outcome = None;
        refuses(bad, "outcome");
        let mut bad = good.clone();
        bad.outcome = Some(Outcome::Queue(proto::Queue::default()));
        refuses(bad, "leases");

        let checks: [(_, fn(&mut proto::Pick)); 4] = [
            ("pick.targets", |pick| {
                pick.targets.resize(5, proto::Target::default());
            }),
            ("pick.targets", |pick| pick.targets.clear()),
            ("pick.targets.dp_rank", |pick| {
                pick.targets[0].dp_rank = 65_536;
            }),
            ("pick.targets.pair.decode_dp_rank", |pick| {
                pick.targets[0].pair = Some(proto::Pair {
                    decode_dp_rank: 65_536,
                    ..proto::Pair::default()
                });
            }),
        ];
        for (path, edit) in checks {
            let mut bad = good.clone();
            if let Some(Outcome::Pick(pick)) = bad.outcome.as_mut() {
                edit(pick);
            }
            refuses(bad, path);
        }
        let checks: [(_, fn(&mut proto::Trace)); 6] = [
            ("trace.profile", |trace| trace.profile = 65_536),
            // The mark's own number is no position, and the mark stands
            // over no position but zero.
            ("trace.profile", |trace| trace.profile = 65_535),
            ("trace.profile", |trace| {
                trace.no_profile = true;
                trace.profile = 1;
            }),
            ("trace.top", |trace| {
                trace.top.resize(4, proto::Scored::default());
            }),
            ("trace.survivors", |trace| trace.survivors = Some(u32::MAX)),
            ("trace.lifted", |trace| trace.lifted = 100),
        ];
        for (path, edit) in checks {
            let mut bad = good.clone();
            edit(bad.trace.as_mut().expect("a trace"));
            refuses(bad, path);
        }
        refuses(
            proto::Decision {
                outcome: Some(Outcome::Shed(proto::Shed::default())),
                ..proto::Decision::default()
            },
            "shed.reason",
        );
        refuses(
            proto::Decision {
                outcome: Some(Outcome::Shed(proto::Shed {
                    reason: proto::ShedReason::Saturated.into(),
                    retry_after_secs: 65_536,
                    ..proto::Shed::default()
                })),
                ..proto::Decision::default()
            },
            "shed.retry_after_secs",
        );
    }

    #[test]
    fn reports_preserve_each_side_class_and_release_reason() {
        use proto::{completed::Class, released::Reason};

        let lease = LeaseId(17);
        for (side, wire) in [
            (Side::Single, proto::Side::Single),
            (Side::Prefill, proto::Side::Prefill),
            (Side::Decode, proto::Side::Decode),
            (Side::Encode, proto::Side::Encode),
        ] {
            let dispatch = Said::Dispatched(
                lease::Dispatch {
                    target_index: 2,
                    side,
                    since_decision_micros: 250,
                },
                Instant::now(),
            );
            let report = frame(lease, Side::Single, dispatch);
            assert_eq!(
                report,
                Frame::Dispatched(proto::Dispatched {
                    lease_id: 17,
                    target_index: 2,
                    side: wire.into(),
                    since_decision_micros: 250
                })
            );
        }
        for (class, wire) in [
            (TransportClass::Ok, Class::Ok),
            (TransportClass::ClientError, Class::ClientError),
            (TransportClass::ServerError, Class::ServerError),
            (TransportClass::ConnectFailed, Class::ConnectFailed),
            (TransportClass::Timeout, Class::Timeout),
            (TransportClass::Cancelled, Class::Cancelled),
        ] {
            let outcome = lease::Outcome {
                side: Side::Decode,
                served: WorkerId(42),
                status: 503,
                class,
                ttft_micros: 15,
                tokens_in: 8,
                tokens_out: 9,
                duration_micros: 38,
            };
            let report = frame(lease, Side::Prefill, Said::Completed(outcome));
            assert_eq!(
                report,
                Frame::Completed(proto::Completed {
                    lease_id: 17,
                    side: proto::Side::Decode.into(),
                    served: 42,
                    status: 503,
                    class: wire.into(),
                    ttft_micros: 15,
                    tokens_in: 8,
                    tokens_out: 9,
                    duration_micros: 38
                })
            );
        }
        for (reason, wire) in [
            (ReleaseReason::Abandoned, Reason::Abandoned),
            (ReleaseReason::LateDecision, Reason::LateDecision),
            (ReleaseReason::ClientGone, Reason::ClientGone),
            (ReleaseReason::CommitFailed, Reason::CommitFailed),
            (ReleaseReason::Expired, Reason::Expired),
            (ReleaseReason::WorkerRemoved, Reason::WorkerRemoved),
        ] {
            for side in [Some(Side::Prefill), Some(Side::Decode), None] {
                let report = frame(lease, Side::Prefill, Said::Released(side, reason));
                let side = side.map_or(proto::Side::Unspecified, proto::Side::from);
                assert_eq!(
                    report,
                    Frame::Released(proto::Released {
                        what: Some(proto::released::What::LeaseId(17)),
                        reason: wire.into(),
                        side: side.into()
                    })
                );
            }
        }
    }
}
