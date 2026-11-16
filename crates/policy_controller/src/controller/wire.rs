//! The decide wire (design 5.2, appendix A): a `DecideRequest` as the
//! engine reads a request, and the engine's decision as the wire's answer.
//!
//! A frame is checked once, where it is read ([`Checked::new`]): what the
//! wire can say and the engine's record cannot hold is refused there, by
//! the field's name, before the request takes a place in a queue. The read
//! after it ([`Checked::read`]) borrows the frame and cannot fail: the
//! model, the hashes, the keys and the header values are read where the
//! frame holds them, and the three lists the record holds in another form
//! (the tried and the excluded ids, the headers) are written into buffers
//! the reading decider keeps ([`Lists`]).
//!
//! The record is narrower than the wire in three kinds of place, each with
//! one rule. An enum the frame leaves unset reads as the record's default
//! (a single-phase request, token hashes), and a number that names no
//! value of it is refused. A count the wire carries wider than the record
//! (the attempt, the media counts) saturates at the record's width. A
//! header's name is a position, so one past the record's 256 is refused:
//! saturated, it would read as another header's.
//!
//! An answer carries no lease here: a decision is written as the engine
//! made it, and the call that issues leases adds them.

use std::{mem, time::Duration};

use crate::{
    decision::{Decision, ShedReason, Target, Trace},
    proto::{self, busy::Reason, decide_answer::Answer, decision::Outcome, identity::Form},
    view::{
        AffinityFacts, Constraints, HashKind, Header, Identity, MediaCounts, ModelFingerprint,
        Phase, RequestFacts, WireModes, WorkerId,
    },
};

/// Why a frame is refused before any decision: the field, by its path in
/// the contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum Refusal {
    /// The field holds a number its record has no place for.
    #[error("{0} is out of its range in this protocol version")]
    OutOfRange(&'static str),
    /// The field asks for what no step of this build serves.
    #[error("{0} is not served by this build")]
    Unserved(&'static str),
}

/// A `DecideRequest` the engine's record holds whole.
#[derive(Debug)]
pub(crate) struct Checked {
    id: Vec<u8>,
    facts: proto::RequestFacts,
    constraints: proto::Constraints,
    phase: Phase,
    hash_kind: HashKind,
}

impl Checked {
    /// Checks `request`, or refuses it by the first field the record cannot
    /// hold. A frame with no facts, or with no constraints, reads as one
    /// that leaves every field of them unset.
    pub(crate) fn new(request: proto::DecideRequest) -> Result<Self, Refusal> {
        let facts = request.facts.unwrap_or_default();
        let phase = match proto::Phase::try_from(facts.phase) {
            Ok(proto::Phase::Unspecified | proto::Phase::Single) => Phase::Single,
            Ok(proto::Phase::Prefill) => Phase::Prefill,
            Ok(proto::Phase::Decode) => Phase::Decode,
            Ok(proto::Phase::Encode) => Phase::Encode,
            Err(_) => return Err(Refusal::OutOfRange("facts.phase")),
        };
        let hash_kind = match proto::HashKind::try_from(facts.hash_kind) {
            Ok(proto::HashKind::Unspecified | proto::HashKind::Token) => HashKind::Token,
            Ok(proto::HashKind::Text) => HashKind::Text,
            Err(_) => return Err(Refusal::OutOfRange("facts.hash_kind")),
        };
        if facts
            .headers
            .iter()
            .any(|header| u8::try_from(header.name).is_err())
        {
            return Err(Refusal::OutOfRange("facts.headers.name"));
        }
        // The controller is to hash the text into block hashes before any
        // stage reads the view; no step does yet, and a decision made as if
        // the request carried no prefix would be another request's.
        if !facts.prompt_text.is_empty() {
            return Err(Refusal::Unserved("facts.prompt_text"));
        }
        Ok(Self {
            id: request.request_id,
            facts,
            constraints: request.constraints.unwrap_or_default(),
            phase,
            hash_kind,
        })
    }

    /// The model the request is for: its deciders' key.
    pub(crate) fn model(&self) -> &str {
        &self.facts.model
    }

    pub(crate) const fn phase(&self) -> Phase {
        self.phase
    }

    /// The decision's seed, which its trace carries.
    pub(crate) const fn seed(&self) -> u64 {
        self.facts.seed
    }

    /// The time the request gives its decision, counted from its frame's
    /// receipt.
    pub(crate) fn budget(&self) -> Duration {
        Duration::from_micros(u64::from(self.facts.budget_micros))
    }

    /// The request's id, for its answer.
    pub(crate) fn into_id(self) -> Vec<u8> {
        self.id
    }

    /// Hands `decide` the request as the engine reads it, with what the
    /// decision's `view` adds to the frame, and returns its answer. The
    /// request borrows the frame and `lists`, for the time of `decide`.
    pub(crate) fn read<R>(
        &self,
        lists: &mut Lists,
        view: Viewed<'_>,
        decide: impl FnOnce(&RequestFacts<'_>, &Constraints<'_>) -> R,
    ) -> R {
        let (facts, constraints) = (&self.facts, &self.constraints);
        lists.tried.clear();
        lists
            .tried
            .extend(facts.tried.iter().copied().map(WorkerId));
        lists.excluded.clear();
        lists
            .excluded
            .extend(constraints.excluded.iter().copied().map(WorkerId));
        let mut headers = emptied(mem::take(&mut lists.headers));
        headers.extend(facts.headers.iter().map(|header| Header {
            // Checked: a name is a position below 256.
            name: u8::try_from(header.name).unwrap_or(u8::MAX),
            value: &header.value,
        }));
        let media = facts
            .media
            .as_ref()
            .map(|media| MediaCounts {
                images: narrowed(media.images),
                audio: narrowed(media.audio),
                video: narrowed(media.video),
                bytes: media.bytes,
            })
            .unwrap_or_default();
        let answer = decide(
            &RequestFacts {
                model: &facts.model,
                namespace_seed: facts.namespace_seed,
                tenant: identity(facts.tenant.as_ref()),
                priority: facts.priority,
                streaming: facts.streaming,
                max_output_tokens: facts.max_output_tokens,
                prompt_tokens: facts.prompt_tokens,
                block_hashes: &facts.block_hashes,
                hash_kind: self.hash_kind,
                session_key: identity(facts.session_key.as_ref()),
                phase: self.phase,
                attempt: u8::try_from(facts.attempt).unwrap_or(u8::MAX),
                tried: &lists.tried,
                budget_micros: view.budget_micros,
                headers: &headers,
                media,
                seed: facts.seed,
                block_size: facts.block_size,
                now_micros: view.now_micros,
                affinity: view.affinity,
                // In-process only: a host names rows where it holds them.
                named: None,
                // The wire carries the name: each read of the request's
                // session fingerprints it.
                model_fingerprint: ModelFingerprint::NONE,
            },
            &Constraints {
                excluded: &lists.excluded,
                // A bit this build does not know names no wire it could
                // match: dropped, as the record drops it.
                wire_modes: WireModes::from_bits(
                    u8::try_from(constraints.wire_modes & 0b111).unwrap_or_default(),
                ),
                overload_lifted: constraints.overload_lifted,
                subset_hint: constraints.subset_hint,
            },
        );
        lists.headers = emptied(headers);
        answer
    }
}

/// What a decision's view adds to its request: the facts that are the
/// deciding host's own, never the frame's.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Viewed<'a> {
    /// The decision's instant in the rows' clock.
    pub(crate) now_micros: u64,
    /// What is left of the request's budget at the decision.
    pub(crate) budget_micros: u32,
    /// The candidates' affinity column, as the view's build filled it.
    pub(crate) affinity: &'a [AffinityFacts],
}

/// The lists of one request in the engine's form, on buffers their decider
/// keeps across requests: a read allocates only while they grow.
#[derive(Debug, Default)]
pub(crate) struct Lists {
    tried: Vec<WorkerId>,
    excluded: Vec<WorkerId>,
    /// Empty between reads: a request's headers borrow its frame.
    headers: Vec<Header<'static>>,
}

/// `spent`'s buffer, emptied, for the headers of another frame. Nothing is
/// left to map: the collect keeps the buffer and names the other frame's
/// lifetime.
fn emptied<'a>(mut spent: Vec<Header<'_>>) -> Vec<Header<'a>> {
    spent.clear();
    spent
        .into_iter()
        .map(|header| Header {
            name: header.name,
            value: "",
        })
        .collect()
}

/// A key in the form its frame carries: the fingerprint, or the plain key,
/// fingerprinted here. Absent when the frame carries none, a fingerprint of
/// zero or an empty name.
fn identity(wire: Option<&proto::Identity>) -> Identity<'_> {
    match wire.and_then(|identity| identity.form.as_ref()) {
        Some(Form::Hash(hash)) => Identity::hashed(*hash),
        Some(Form::Plain(key)) if !key.is_empty() => Identity::plain(key),
        Some(Form::Plain(_)) | None => Identity::NONE,
    }
}

/// A count at the record's width: one past it reads as the widest.
fn narrowed(count: u32) -> u16 {
    u16::try_from(count).unwrap_or(u16::MAX)
}

/// The answer that carries `decision`, made under `term`.
pub(crate) fn decided(id: Vec<u8>, term: u64, decision: &Decision) -> proto::DecideAnswer {
    proto::DecideAnswer {
        request_id: id,
        term,
        answer: Some(Answer::Decision(self::decision(decision))),
    }
}

/// The answer of a request this controller does not decide, for `reason`:
/// the gateway decides it locally at once. No pacing hint rides it yet:
/// when the gateway asks again is its own interval's to say.
pub(crate) fn busy(id: Vec<u8>, term: u64, reason: Reason) -> proto::DecideAnswer {
    proto::DecideAnswer {
        request_id: id,
        term,
        answer: Some(Answer::Busy(proto::Busy {
            reason: reason.into(),
            retry_after_micros: 0,
        })),
    }
}

/// A decision as the wire carries it, with no lease.
fn decision(decision: &Decision) -> proto::Decision {
    let outcome = match decision {
        Decision::Pick(pick) => Outcome::Pick(proto::Pick {
            targets: pick.targets.iter().map(target).collect(),
        }),
        Decision::Shed(shed) => Outcome::Shed(proto::Shed {
            reason: shed_reason(shed.reason).into(),
            emptied_by: shed.emptied_by.to_owned(),
            retry_after_secs: u32::from(shed.retry_after_secs),
            wait_allowed: shed.wait_allowed,
        }),
        Decision::Queue(queue) => Outcome::Queue(proto::Queue {
            ticket: queue.ticket,
            max_wait_micros: queue.max_wait_micros,
        }),
    };
    proto::Decision {
        leases: Vec::new(),
        outcome: Some(outcome),
        trace: Some(trace(decision.trace())),
    }
}

fn target(target: &Target) -> proto::Target {
    proto::Target {
        id: target.id.0,
        dp_rank: u32::from(target.dp_rank),
        pair: target.pair.map(|pair| proto::Pair {
            decode_id: pair.decode_id.0,
            decode_dp_rank: u32::from(pair.decode_dp_rank),
        }),
    }
}

const fn shed_reason(reason: ShedReason) -> proto::ShedReason {
    match reason {
        ShedReason::Saturated => proto::ShedReason::Saturated,
        ShedReason::FlowLimit => proto::ShedReason::FlowLimit,
        ShedReason::NoEndpoints => proto::ShedReason::NoEndpoints,
        ShedReason::PolicyDeclined => proto::ShedReason::PolicyDeclined,
        ShedReason::Deadline => proto::ShedReason::Deadline,
    }
}

/// The record's trace: the five facts it holds. The observability contract
/// fills the wire's other fields; they stay unset until then.
fn trace(trace: &Trace) -> proto::Trace {
    proto::Trace {
        profile: u32::from(trace.profile),
        survivors: trace.survivors,
        top: trace
            .top
            .iter()
            .take(usize::from(trace.top_len))
            .map(|&(id, score)| proto::Scored {
                worker_id: id.0,
                score,
                ..proto::Scored::default()
            })
            .collect(),
        seed: trace.seed,
        generation: trace.generation,
        ..proto::Trace::default()
    }
}

#[cfg(test)]
mod tests {
    use super::{busy, decided, emptied, Checked, Lists, Refusal, Viewed};
    use crate::{
        decision::{Decision, Pair, Pick, Queue, Shed, ShedReason, Target, Targets, Trace},
        proto::{self, busy::Reason, decide_answer::Answer, decision::Outcome, identity::Form},
        view::{
            fingerprint, AffinityFacts, Constraints, HashKind, Header, Identity, MediaCounts,
            ModelFingerprint, Phase, RequestFacts, WireModes, WorkerId,
        },
    };

    fn key(form: Form) -> proto::Identity {
        proto::Identity { form: Some(form) }
    }

    /// A frame whose every field holds a value of its own.
    fn full() -> proto::DecideRequest {
        proto::DecideRequest {
            request_id: vec![7, 8, 9],
            facts: Some(proto::RequestFacts {
                model: "model-a".to_owned(),
                namespace_seed: 11,
                tenant: Some(key(Form::Plain("tenant-a".to_owned()))),
                priority: -100,
                streaming: true,
                max_output_tokens: 12,
                prompt_tokens: 13,
                block_hashes: vec![14, 15, 16],
                hash_kind: proto::HashKind::Text.into(),
                session_key: Some(key(Form::Hash(17))),
                phase: proto::Phase::Decode.into(),
                attempt: 2,
                tried: vec![18, 19],
                budget_micros: 2_000,
                headers: vec![
                    proto::Header {
                        name: 0,
                        value: "first".to_owned(),
                    },
                    proto::Header {
                        name: 255,
                        value: "last".to_owned(),
                    },
                ],
                media: Some(proto::MediaCounts {
                    images: 20,
                    audio: 21,
                    video: 22,
                    bytes: 23,
                }),
                seed: 24,
                prompt_text: Vec::new(),
                block_size: 25,
            }),
            constraints: Some(proto::Constraints {
                excluded: vec![26, 27, 28],
                wire_modes: 0b110,
                overload_lifted: true,
                subset_hint: Some(0),
            }),
        }
    }

    /// The full frame with its facts edited.
    fn with(edit: impl FnOnce(&mut proto::RequestFacts)) -> proto::DecideRequest {
        let mut frame = full();
        edit(frame.facts.as_mut().expect("the full frame has facts"));
        frame
    }

    /// Reads `frame` over a view of the case's own (the rows' clock at 77,
    /// 1,500 microseconds left, a column of two) and hands `check` the
    /// request as the engine reads it.
    fn reading<R>(
        frame: proto::DecideRequest,
        lists: &mut Lists,
        check: impl FnOnce(&RequestFacts<'_>, &Constraints<'_>) -> R,
    ) -> R {
        let checked = Checked::new(frame).expect("a frame the record holds");
        let affinity = [AffinityFacts::default(); 2];
        let view = Viewed {
            now_micros: 77,
            budget_micros: 1_500,
            affinity: &affinity,
        };
        checked.read(lists, view, check)
    }

    #[test]
    fn every_field_of_a_frame_reaches_the_engines_request() {
        let checked = Checked::new(full()).expect("a frame the record holds");
        assert_eq!(
            (checked.model(), checked.phase(), checked.seed()),
            ("model-a", Phase::Decode, 24)
        );
        assert_eq!(checked.budget().as_micros(), 2_000);
        assert_eq!(checked.into_id(), [7, 8, 9]);
        reading(full(), &mut Lists::default(), |facts, constraints| {
            assert_eq!(facts.model, "model-a");
            assert_eq!(facts.namespace_seed, 11);
            assert_eq!(facts.tenant, Identity::plain("tenant-a"));
            assert_eq!(facts.tenant.hash, fingerprint(b"tenant-a"));
            assert_eq!(facts.priority, -100);
            assert!(facts.streaming);
            assert_eq!((facts.max_output_tokens, facts.prompt_tokens), (12, 13));
            assert_eq!(facts.block_hashes, [14, 15, 16]);
            assert_eq!(facts.hash_kind, HashKind::Text);
            assert_eq!(facts.session_key, Identity::hashed(17));
            assert_eq!(facts.phase, Phase::Decode);
            assert_eq!(facts.attempt, 2);
            assert_eq!(facts.tried, [WorkerId(18), WorkerId(19)]);
            assert_eq!(
                facts.headers,
                [
                    Header {
                        name: 0,
                        value: "first"
                    },
                    Header {
                        name: 255,
                        value: "last"
                    },
                ]
            );
            assert_eq!(
                facts.media,
                MediaCounts {
                    images: 20,
                    audio: 21,
                    video: 22,
                    bytes: 23
                }
            );
            assert_eq!((facts.seed, facts.block_size), (24, 25));
            // The view's three facts are the decision's own, not the
            // frame's: the frame's budget of 2,000 is not what the engine
            // is told is left.
            assert_eq!(
                (facts.now_micros, facts.budget_micros, facts.affinity.len()),
                (77, 1_500, 2)
            );
            assert!(facts.named.is_none());
            assert_eq!(facts.model_fingerprint, ModelFingerprint::NONE);
            assert_eq!(
                constraints.excluded,
                [WorkerId(26), WorkerId(27), WorkerId(28)]
            );
            assert_eq!(
                constraints.wire_modes,
                WireModes::GRPC.union(WireModes::ZMQ)
            );
            assert!(constraints.overload_lifted);
            assert_eq!(constraints.subset_hint, Some(0));
        });
    }

    #[test]
    fn a_frame_with_nothing_set_reads_as_the_records_defaults() {
        let checked = Checked::new(proto::DecideRequest::default()).expect("an empty frame");
        assert_eq!(
            (checked.model(), checked.phase(), checked.seed()),
            ("", Phase::Single, 0)
        );
        assert!(checked.budget().is_zero());
        let empty = proto::DecideRequest::default();
        reading(empty, &mut Lists::default(), |facts, constraints| {
            assert_eq!(
                (facts.phase, facts.hash_kind),
                (Phase::Single, HashKind::Token)
            );
            assert_eq!(
                (facts.tenant, facts.session_key),
                (Identity::NONE, Identity::NONE)
            );
            assert_eq!(facts.media, MediaCounts::default());
            assert!(facts.headers.is_empty() && facts.tried.is_empty());
            assert!(constraints.excluded.is_empty());
            assert_eq!(constraints.wire_modes, WireModes::NONE);
            assert_eq!(constraints.subset_hint, None);
        });
    }

    #[test]
    fn an_enum_reads_by_its_number_and_a_number_that_names_no_value_is_refused_by_its_field() {
        for (wire, record) in [
            (proto::Phase::Unspecified, Phase::Single),
            (proto::Phase::Single, Phase::Single),
            (proto::Phase::Prefill, Phase::Prefill),
            (proto::Phase::Decode, Phase::Decode),
            (proto::Phase::Encode, Phase::Encode),
        ] {
            let frame = with(|facts| facts.phase = wire.into());
            let read = reading(frame, &mut Lists::default(), |facts, _| facts.phase);
            assert_eq!(read, record, "{wire:?}");
        }
        for (wire, record) in [
            (proto::HashKind::Unspecified, HashKind::Token),
            (proto::HashKind::Token, HashKind::Token),
            (proto::HashKind::Text, HashKind::Text),
        ] {
            let frame = with(|facts| facts.hash_kind = wire.into());
            let read = reading(frame, &mut Lists::default(), |facts, _| facts.hash_kind);
            assert_eq!(read, record, "{wire:?}");
        }
        let refused = |frame| Checked::new(frame).expect_err("refused");
        assert_eq!(
            refused(with(|facts| facts.phase = 5)),
            Refusal::OutOfRange("facts.phase")
        );
        assert_eq!(
            refused(with(|facts| facts.hash_kind = 3)),
            Refusal::OutOfRange("facts.hash_kind")
        );
        assert_eq!(
            refused(with(|facts| facts.phase = 5)).to_string(),
            "facts.phase is out of its range in this protocol version"
        );
    }

    #[test]
    fn a_key_is_read_in_the_form_its_frame_carries() {
        let tenant = |form: Option<Form>| {
            let frame = with(|facts| facts.tenant = Some(proto::Identity { form }));
            reading(frame, &mut Lists::default(), |facts, _| {
                (facts.tenant.hash, facts.tenant.plain.map(str::to_owned))
            })
        };
        assert_eq!(tenant(Some(Form::Hash(5))), (5, None));
        assert_eq!(
            tenant(Some(Form::Plain("a".to_owned()))),
            (fingerprint(b"a"), Some("a".to_owned()))
        );
        // No form, a fingerprint of zero and an empty name are no key.
        for absent in [None, Some(Form::Hash(0)), Some(Form::Plain(String::new()))] {
            assert_eq!(tenant(absent.clone()), (0, None), "{absent:?}");
        }
        let unset = with(|facts| facts.session_key = None);
        let read = reading(unset, &mut Lists::default(), |facts, _| {
            facts.session_key.hash
        });
        assert_eq!(read, 0);
    }

    #[test]
    fn a_count_past_its_records_width_saturates_and_a_header_past_256_names_is_refused() {
        let frame = with(|facts| {
            facts.attempt = 256;
            facts.media = Some(proto::MediaCounts {
                images: 65_536,
                audio: u32::MAX,
                video: 65_535,
                bytes: u32::MAX,
            });
        });
        let (attempt, media) = reading(frame, &mut Lists::default(), |facts, _| {
            (facts.attempt, facts.media)
        });
        assert_eq!(attempt, u8::MAX);
        assert_eq!(
            media,
            MediaCounts {
                images: u16::MAX,
                audio: u16::MAX,
                video: 65_535,
                bytes: u32::MAX
            }
        );
        let frame = with(|facts| facts.headers[1].name = 256);
        assert_eq!(
            Checked::new(frame).expect_err("a name past the record's"),
            Refusal::OutOfRange("facts.headers.name")
        );
        // A wire-mode bit this build does not know is dropped, wherever in
        // the wire's 32 it stands.
        let mut frame = full();
        frame
            .constraints
            .as_mut()
            .expect("the full frame has constraints")
            .wire_modes = 0b1_0000_1001;
        let modes = reading(frame, &mut Lists::default(), |_, constraints| {
            constraints.wire_modes
        });
        assert_eq!(modes, WireModes::HTTP);
    }

    #[test]
    fn a_frame_that_carries_its_prompt_as_text_is_refused_as_unserved() {
        let refused = Checked::new(with(|facts| facts.prompt_text = b"hello".to_vec()))
            .expect_err("no step hashes the text yet");
        assert_eq!(refused, Refusal::Unserved("facts.prompt_text"));
        assert_eq!(
            refused.to_string(),
            "facts.prompt_text is not served by this build"
        );
    }

    #[test]
    fn a_decider_reads_frame_after_frame_on_the_same_buffers() {
        let mut lists = Lists::default();
        let first = reading(full(), &mut lists, |facts, constraints| {
            (
                facts.tried.len(),
                facts.headers.len(),
                constraints.excluded.len(),
            )
        });
        assert_eq!(first, (2, 2, 3));
        let held = (
            lists.tried.capacity(),
            lists.excluded.capacity(),
            lists.headers.capacity(),
        );
        assert!(held.0 >= 2 && held.1 >= 3 && held.2 >= 2, "{held:?}");
        assert!(
            lists.headers.is_empty(),
            "a frame's headers end with its read"
        );
        // A frame with shorter lists reads its own, and nothing of the last.
        let frame = with(|facts| {
            facts.tried = vec![1];
            facts.headers.truncate(1);
        });
        let second = reading(frame, &mut lists, |facts, _| {
            let headers: Vec<(u8, String)> = facts
                .headers
                .iter()
                .map(|header| (header.name, header.value.to_owned()))
                .collect();
            (facts.tried.to_vec(), headers)
        });
        assert_eq!(second, (vec![WorkerId(1)], vec![(0, "first".to_owned())]));
        assert_eq!(
            held,
            (
                lists.tried.capacity(),
                lists.excluded.capacity(),
                lists.headers.capacity()
            ),
            "the buffers are kept"
        );
        // The buffer one frame's headers leave is the one the next frame's
        // take: the emptying keeps it.
        let spent: Vec<Header<'_>> = Vec::with_capacity(8);
        assert!(emptied(spent).capacity() >= 8);
    }

    fn traced() -> Trace {
        Trace {
            profile: 3,
            survivors: 5,
            // The third entry is past the trace's length: not a part of it.
            top: [
                (WorkerId(31), 0.5),
                (WorkerId(32), 0.25),
                (WorkerId(33), 9.0),
            ],
            top_len: 2,
            seed: 41,
            generation: 6,
        }
    }

    fn wire_trace() -> proto::Trace {
        proto::Trace {
            profile: 3,
            survivors: 5,
            top: vec![
                proto::Scored {
                    worker_id: 31,
                    score: 0.5,
                    ..proto::Scored::default()
                },
                proto::Scored {
                    worker_id: 32,
                    score: 0.25,
                    ..proto::Scored::default()
                },
            ],
            seed: 41,
            generation: 6,
            ..proto::Trace::default()
        }
    }

    /// The decision an answer carries: its outcome, its trace and how many
    /// leases ride it.
    fn carried(answer: proto::DecideAnswer) -> (Outcome, Option<proto::Trace>, usize) {
        match answer.answer {
            Some(Answer::Decision(decision)) => (
                decision.outcome.expect("a decision has an outcome"),
                decision.trace,
                decision.leases.len(),
            ),
            other => panic!("a decision's answer: {other:?}"),
        }
    }

    #[test]
    fn a_pick_is_its_targets_in_order_with_the_trace_and_no_lease() {
        let mut targets = Targets::new();
        targets.push(Target {
            position: 4,
            id: WorkerId(51),
            dp_rank: 2,
            pair: Some(Pair {
                decode_position: 1,
                decode_id: WorkerId(52),
                decode_dp_rank: 3,
            }),
        });
        targets.push(Target {
            position: 0,
            id: WorkerId(53),
            dp_rank: 0,
            pair: None,
        });
        let pick = Decision::Pick(Pick {
            targets,
            trace: traced(),
            ..Pick::default()
        });
        let answer = decided(vec![1, 2], 9, &pick);
        assert_eq!(
            (answer.request_id.as_slice(), answer.term),
            (&[1, 2][..], 9)
        );
        let (outcome, trace, leases) = carried(answer);
        assert_eq!(
            outcome,
            Outcome::Pick(proto::Pick {
                targets: vec![
                    proto::Target {
                        id: 51,
                        dp_rank: 2,
                        pair: Some(proto::Pair {
                            decode_id: 52,
                            decode_dp_rank: 3
                        }),
                    },
                    proto::Target {
                        id: 53,
                        dp_rank: 0,
                        pair: None
                    },
                ],
            })
        );
        assert_eq!(trace, Some(wire_trace()));
        assert_eq!(leases, 0);
    }

    #[test]
    fn a_shed_names_its_reason_and_its_stage_and_a_ticket_its_wait() {
        for (record, wire) in [
            (ShedReason::Saturated, proto::ShedReason::Saturated),
            (ShedReason::FlowLimit, proto::ShedReason::FlowLimit),
            (ShedReason::NoEndpoints, proto::ShedReason::NoEndpoints),
            (
                ShedReason::PolicyDeclined,
                proto::ShedReason::PolicyDeclined,
            ),
            (ShedReason::Deadline, proto::ShedReason::Deadline),
        ] {
            let shed = Decision::Shed(Shed {
                reason: record,
                emptied_by: "healthy",
                retry_after_secs: 7,
                wait_allowed: true,
                trace: traced(),
            });
            let (outcome, trace, leases) = carried(decided(Vec::new(), 1, &shed));
            assert_eq!(
                outcome,
                Outcome::Shed(proto::Shed {
                    reason: wire.into(),
                    emptied_by: "healthy".to_owned(),
                    retry_after_secs: 7,
                    wait_allowed: true,
                }),
                "{record:?}"
            );
            assert_eq!((trace, leases), (Some(wire_trace()), 0));
        }
        let queue = Decision::Queue(Queue {
            ticket: 61,
            max_wait_micros: 62,
            trace: traced(),
        });
        let (outcome, trace, leases) = carried(decided(Vec::new(), 1, &queue));
        assert_eq!(
            outcome,
            Outcome::Queue(proto::Queue {
                ticket: 61,
                max_wait_micros: 62
            })
        );
        assert_eq!((trace, leases), (Some(wire_trace()), 0));
    }

    #[test]
    fn a_busy_answer_carries_its_requests_id_its_term_and_its_reason_alone() {
        for reason in [
            Reason::Unspecified,
            Reason::QueueDepth,
            Reason::NoCredit,
            Reason::BudgetGoneOnArrival,
            Reason::NotLeader,
        ] {
            assert_eq!(
                busy(vec![5], 3, reason),
                proto::DecideAnswer {
                    request_id: vec![5],
                    term: 3,
                    answer: Some(Answer::Busy(proto::Busy {
                        reason: reason.into(),
                        retry_after_micros: 0,
                    })),
                }
            );
        }
    }
}
