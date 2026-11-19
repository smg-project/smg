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
//! saturated, it would read as another header's. A frame's headers are as
//! many as those positions at most ([`MAX_HEADERS`]): a longer list repeats
//! a name, and is refused by its count.
//!
//! A key is read in the form its frame carries, and the two keys' hash
//! forms are two things (the contract's `Identity`). A tenant's is its
//! key's fingerprint, which the record carries as it is. A session key's
//! is the session's value, the key within the request's model and phase
//! already, while the record carries a key's fingerprint and scopes it
//! itself: the fingerprint is read back from the value
//! ([`key_of_session`]), once, at the check. So one key names one session
//! in either form, and a stage that reads the key itself reads one key.
//! The one value that would give the fingerprint no key has is refused.
//!
//! A range the contract fixes refuses a frame, its id's length among
//! them, and the transport refuses one over its size as received; a
//! bound a generation's document sets never does,
//! since the gateway may be a generation behind. The block hashes past what
//! the deciding generation reads of the request are cut at the read, the
//! first ones kept: a prefix cut short is still a prefix.
//!
//! A request that carries its prompt as text is checked like any other and
//! says so ([`Checked::has_text`]): the controller is to hash the text into
//! block hashes before any stage reads the view, and its decider has no
//! decision for it until a step does. Until one reads the text it is
//! dropped at the check, so a frame that waits in a queue holds none.
//!
//! A pick's answer carries the lease its decider had the store file for it
//! ([`decided`]): one a decision, minted under the answer's term, a credit
//! on the pick's first target, with the time its gateway has to name the
//! dispatch. A shed and a queue ticket carry none.

use std::{
    mem,
    time::{Duration, Instant},
};

pub(crate) use crate::door::Viewed;
use crate::{
    decision::{Decision, ShedReason, Target, Trace},
    door::Asked,
    lease::{Lease, Side},
    proto::{self, busy::Reason, decide_answer::Answer, decision::Outcome, identity::Form},
    spec::{AppliedSpec, MAX_HEADERS},
    view::{
        fingerprint, key_of_session, Constraints, HashKind, Header, Identity, MediaCounts,
        ModelFingerprint, Phase, RequestFacts, Session, WireModes, WorkerId,
    },
};

/// Why a frame is refused before any decision: the field, by its path in
/// the contract. A frame's size is the transport's to refuse
/// ([`MAX_FRAME`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum Refusal {
    /// The field holds a number its record has no place for.
    #[error("{0} is out of its range in this protocol version")]
    OutOfRange(&'static str),
}

/// The longest `request_id` a frame carries, in bytes; the shortest is one.
/// The id is its request's key and rides back on every answer, so its
/// length is a range of the contract.
pub(crate) const MAX_REQUEST_ID: usize = 256;

/// The largest `DecideRequest`, in bytes as received: the transport refuses
/// a longer one on the `Decide` call at its length, before it reads it, so
/// a field this build does not know counts too. A stream's window and a
/// decider's queue hold frames by the place, so the frame's size is what
/// bounds their bytes: as read, a frame holds no more than it was received
/// as and one record for each of its headers, [`MAX_HEADERS`] of them at
/// most. A frame over the bound is refused whole, whatever its lists hold:
/// a sender cuts its hashes at the tail until its frame fits, and the
/// spec's range for their cap ends at what one frame carries
/// ([`MAX_BLOCK_HASHES`](crate::spec::MAX_BLOCK_HASHES)).
pub(crate) const MAX_FRAME: usize = 64 * 1024;

/// The bit of a [`Checked`] frame's rare words for one that carried its
/// prompt as text.
const TEXT: u8 = 1;

/// The bit for a frame whose gateway asks its request again
/// (`DecideRequest.again`).
const AGAIN: u8 = 1 << 1;

/// What the ranges of a frame's facts read as: its phase, the kind of its
/// block hashes and its session key's fingerprint (`0` for a key carried
/// plain, and for none). Refuses by the first field the record cannot
/// hold. The check of a frame received reads its facts here
/// ([`Checked::new`]), and so does a sender whose callers share one
/// stream: a frame out of its ranges ends the stream it is sent on.
/// Inline in both, so the check of a frame received pays no call for it.
#[inline(always)]
pub(crate) fn ranges(facts: &proto::RequestFacts) -> Result<(Phase, HashKind, u64), Refusal> {
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
    // The headers by their count first: an empty one is two bytes as
    // encoded and a record of its own as read, so the frame's size
    // does not bound them, and a list longer than a view's names
    // repeats one.
    if facts.headers.len() > MAX_HEADERS {
        return Err(Refusal::OutOfRange("facts.headers"));
    }
    if facts
        .headers
        .iter()
        .any(|header| u8::try_from(header.name).is_err())
    {
        return Err(Refusal::OutOfRange("facts.headers.name"));
    }
    // A session key's hash form is the session's value: the key's
    // fingerprint is read back from it within the request's model and
    // phase. A value of zero is no session.
    let session = facts.session_key.as_ref();
    let session_key = match session.and_then(|key| key.form.as_ref()) {
        Some(Form::Hash(value @ 1..)) => {
            let model = fingerprint(facts.model.as_bytes());
            match key_of_session(*value, model, phase) {
                0 => return Err(Refusal::OutOfRange("facts.session_key.hash")),
                key => key,
            }
        }
        _ => 0,
    };
    Ok((phase, hash_kind, session_key))
}

/// A `DecideRequest` the engine's record holds whole.
#[derive(Debug)]
pub(crate) struct Checked {
    id: Vec<u8>,
    facts: proto::RequestFacts,
    constraints: proto::Constraints,
    phase: Phase,
    hash_kind: HashKind,
    /// The session key's fingerprint, for a frame that carries the key as
    /// the session's value; `0` for one that carries it plain or carries
    /// none.
    session_key: u64,
    /// What the frame says that few frames say, a bit a word ([`TEXT`],
    /// [`AGAIN`]): `0` for a frame that says neither, so a decider reads
    /// both by one test.
    rare: u8,
}

impl Checked {
    /// Checks `request`, or refuses it by the first field the record cannot
    /// hold. Its size is not asked here: the transport held the frame to
    /// [`MAX_FRAME`] as received, and a frame reads to no more than that.
    /// A frame with no facts, or with no constraints, reads as one that
    /// leaves every field of them unset.
    pub(crate) fn new(request: proto::DecideRequest) -> Result<Self, Refusal> {
        if !(1..=MAX_REQUEST_ID).contains(&request.request_id.len()) {
            return Err(Refusal::OutOfRange("request_id"));
        }
        let mut facts = request.facts.unwrap_or_default();
        // The prompt as text is said and not kept: no step reads it yet.
        let text = !mem::take(&mut facts.prompt_text).is_empty();
        let (phase, hash_kind, session_key) = ranges(&facts)?;
        let rare = (u8::from(text) * TEXT) | (u8::from(request.again) * AGAIN);
        Ok(Self {
            id: request.request_id,
            facts,
            constraints: request.constraints.unwrap_or_default(),
            phase,
            hash_kind,
            session_key,
            rare,
        })
    }

    /// The model the request is for: its deciders' key.
    pub(crate) fn model(&self) -> &str {
        &self.facts.model
    }

    pub(crate) const fn phase(&self) -> Phase {
        self.phase
    }

    /// The decision's seed, which its trace carries and its naming draws from.
    pub(crate) const fn seed(&self) -> u64 {
        self.facts.seed
    }

    /// The time the request gives its decision, counted from its frame's
    /// receipt.
    pub(crate) fn budget(&self) -> Duration {
        Duration::from_micros(u64::from(self.facts.budget_micros))
    }

    /// Whether the frame says a word few frames say: it carried its prompt
    /// as text, or its gateway asks its request again. A decider asks this
    /// of every request, once, and which word it is of those that say one.
    pub(crate) const fn says_more(&self) -> bool {
        self.rare != 0
    }

    /// Whether the request carried its prompt as text, for the controller
    /// to hash.
    pub(crate) const fn has_text(&self) -> bool {
        self.rare & TEXT != 0
    }

    /// Whether the request's gateway sent it before under this id and
    /// holds no answer to it (`DecideRequest.again`).
    pub(crate) const fn again(&self) -> bool {
        self.rare & AGAIN != 0
    }

    /// How many block hashes the frame carries: a read hands the engine
    /// as many as its generation reads of them ([`Self::hashes_read`]).
    pub(crate) fn hashes(&self) -> usize {
        self.facts.block_hashes.len()
    }

    /// How many of the frame's block hashes a decision under `spec` reads:
    /// the depth of the generation's stages, in blocks as the frame's host
    /// hashed them ([`AppliedSpec::block_hashes_read`]).
    pub(crate) fn hashes_read(&self, spec: &AppliedSpec) -> usize {
        spec.block_hashes_read(self.hash_kind, self.facts.block_size)
    }

    /// The request's id as its frame carried it: its lease's record keeps
    /// the id's fingerprint.
    pub(crate) fn id(&self) -> &[u8] {
        &self.id
    }

    /// The request's id, for its answer.
    pub(crate) fn into_id(self) -> Vec<u8> {
        self.id
    }

    /// The wires the request's gateway states it speaks: `None` where it
    /// states none. A bit this build does not know is dropped, as the
    /// record drops it: a gateway that states such bits alone states wires,
    /// and none that a member of this build shares.
    pub(crate) fn wires(&self) -> Option<WireModes> {
        let stated = self.constraints.wire_modes;
        (stated != 0)
            .then(|| WireModes::from_bits(u8::try_from(stated & 0b111).unwrap_or_default()))
    }

    /// The request's subset hint, absent where its gateway gives none: a
    /// present 0 is a decision over no candidates.
    pub(crate) const fn subset(&self) -> Option<u32> {
        self.constraints.subset_hint
    }

    /// The request's block hashes, in order from its first block.
    pub(crate) fn block_hashes(&self) -> &[u64] {
        &self.facts.block_hashes
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
                // The first hashes, as deep as the generation reads.
                block_hashes: facts
                    .block_hashes
                    .get(..view.max_block_hashes)
                    .unwrap_or(&facts.block_hashes),
                hash_kind: self.hash_kind,
                session_key: match facts.session_key.as_ref().and_then(|key| key.form.as_ref()) {
                    Some(Form::Plain(key)) if !key.is_empty() => Identity::plain(key),
                    Some(Form::Hash(_)) => Identity::hashed(self.session_key),
                    Some(Form::Plain(_)) | None => Identity::NONE,
                },
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
                // In-process only: the rows the deciding host named.
                named: view.named,
                // The wire carries the name: each read of the request's
                // session fingerprints it.
                model_fingerprint: ModelFingerprint::NONE,
                // No decider asks its view for a pinned worker's row.
                pinned_at: 0,
                // What the leases book, beside rows a view lent.
                lent_in_flight: view.in_flight.as_ref(),
                // The store lends no count of a key's requests beside a
                // decision's rows yet.
                lent_session_in_flight: None,
                session_counted: false,
            },
            &Constraints {
                excluded: &lists.excluded,
                wire_modes: self.wires().unwrap_or_default(),
                overload_lifted: constraints.overload_lifted,
                subset_hint: constraints.subset_hint,
                // The asking gateway's standing vetoes beside lent rows,
                // and nothing where each row carries its own bit.
                stalled: view.stalled,
                held: view.held,
                pins: view.pins,
            },
        );
        lists.headers = emptied(headers);
        answer
    }
}

impl Asked for Checked {
    type Lists = Lists;

    fn model(&self) -> &str {
        Self::model(self)
    }

    fn phase(&self) -> Phase {
        Self::phase(self)
    }

    fn id(&self) -> &[u8] {
        Self::id(self)
    }

    fn wires(&self) -> Option<WireModes> {
        Self::wires(self)
    }

    fn subset(&self) -> Option<u32> {
        Self::subset(self)
    }

    fn hashes(&self) -> usize {
        Self::hashes(self)
    }

    fn hashes_read(&self, spec: &AppliedSpec) -> usize {
        Self::hashes_read(self, spec)
    }

    fn block_hashes(&self) -> &[u64] {
        Self::block_hashes(self)
    }

    fn seed(&self) -> u64 {
        Self::seed(self)
    }

    fn read<R>(
        &self,
        lists: &mut Lists,
        viewed: Viewed<'_>,
        decide: impl FnOnce(&RequestFacts<'_>, &Constraints<'_>) -> R,
    ) -> R {
        Self::read(self, lists, viewed, decide)
    }
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

/// A tenant in the form its frame carries: the fingerprint, or the plain
/// key, fingerprinted here. Absent when the frame carries none, a
/// fingerprint of zero or an empty name.
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

/// A pick's lease as its decider hands it over: the lease the store filed,
/// what its decision left of the lease's dispatch span, and when that span
/// ends.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Leased {
    /// The lease as the store filed it.
    pub(crate) lease: Lease,
    /// The session key of the decode side's lease of a pair's pick; none
    /// for a single's. That lease is the prefill side's under this key:
    /// one id, one record, and at the answer one index and one count of
    /// tokens. The key alone keeps the queue's entry small.
    pub(crate) decode: Option<Session>,
    /// What is left of its dispatch span: at the decision's last read as
    /// its decider hands it over, at the send as its stream sends it.
    pub(crate) left: Duration,
    /// When the span ends: the dispatch deadline the store filed.
    pub(crate) due: Instant,
}

impl Leased {
    /// The least of a dispatch span a lease is handed over with: the wire
    /// tells the span in whole milliseconds, and under one is none.
    pub(crate) const LEAST: Duration = Duration::from_millis(1);

    /// The lease as its stream sends it at `now`, a read of the clock
    /// behind its decider's last: with what is left of the span there.
    pub(crate) fn sent_at(self, now: Instant) -> Self {
        Self {
            left: self.due.saturating_duration_since(now),
            ..self
        }
    }

    /// Whether too little of the span is left to hand the lease over.
    pub(crate) fn spent(&self) -> bool {
        self.left < Self::LEAST
    }
}

/// A decider's answer as its stream's queue holds it. A decision waits as
/// it was made, with the lease issued for its pick, and its frame is built
/// where its stream sends it, behind that send's read of the fence and of
/// the lease's span: the lists a frame carries (a pick's targets, the
/// trace's top, the lease) are made and freed on the thread that encodes
/// them, and a deciding thread makes none.
#[derive(Debug)]
pub(crate) enum Decided {
    /// A decision made under `term` for the request `id`, and the lease
    /// its decider hands over with it.
    Decision {
        id: Vec<u8>,
        term: u64,
        decision: Decision,
        lease: Option<Leased>,
    },
    /// An answer that carries no list: a `Busy`.
    Frame(proto::DecideAnswer),
}

#[cfg(test)]
impl Decided {
    /// The frame the wire carries, as a stream builds it that sends at its
    /// decider's last read: what a case of the deciders reads.
    pub(crate) fn frame(self) -> proto::DecideAnswer {
        match self {
            Self::Decision {
                id,
                term,
                decision,
                lease,
            } => decided(id, term, &decision, lease.as_ref()),
            Self::Frame(frame) => frame,
        }
    }
}

/// The answer that carries `decision`, made under `term`, with the lease
/// the store filed for it where it is a pick that was issued one, and
/// what that lease's dispatch span has left where the answer is sent. A pair's pick
/// carries two leases of one id, the prefill side's first.
pub(crate) fn decided(
    id: Vec<u8>,
    term: u64,
    decision: &Decision,
    lease: Option<&Leased>,
) -> proto::DecideAnswer {
    let mut decision = self::decision(decision);
    if let Some(leased) = lease {
        let mut leases = Vec::with_capacity(1 + usize::from(leased.decode.is_some()));
        leases.push(self::lease(&leased.lease, term, leased.left));
        if let Some(session) = leased.decode {
            let decode = Lease {
                side: Side::Decode,
                session,
                ..leased.lease
            };
            leases.push(self::lease(&decode, term, leased.left));
        }
        decision.leases = leases;
    }
    proto::DecideAnswer {
        request_id: id,
        term,
        answer: Some(Answer::Decision(decision)),
    }
}

/// A duration as a wire field's milliseconds: the field's largest value
/// for a longer one.
pub(super) fn millis(duration: Duration) -> u32 {
    u32::try_from(duration.as_millis()).unwrap_or(u32::MAX)
}

/// A lease this controller minted, as the wire carries it under the term
/// that minted it, with `left`: what is left, where it is sent, of the time
/// its gateway has to name the dispatch. The decide call issues one kind of
/// lease, a credit: on a single request's target, or on each side of a
/// pair under one id.
fn lease(lease: &Lease, term: u64, left: Duration) -> proto::Lease {
    let side = match lease.side {
        Side::Single => proto::Side::Single,
        Side::Prefill => proto::Side::Prefill,
        Side::Decode => proto::Side::Decode,
        Side::Encode => proto::Side::Encode,
    };
    proto::Lease {
        id: lease.id.0,
        kind: proto::LeaseKind::Credit.into(),
        side: side.into(),
        target_index: u32::from(lease.target_index),
        generation: lease.generation,
        session_key: lease.session.get(),
        tokens: lease.tokens,
        term,
        dispatch_ttl_ms: millis(left),
        origin: proto::LeaseOrigin::Controller.into(),
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

/// The frame that grants its stream `window` more requests.
pub(crate) fn credit(term: u64, window: u32) -> proto::DecideAnswer {
    proto::DecideAnswer {
        request_id: Vec::new(),
        term,
        answer: Some(Answer::Credit(proto::Credit { window })),
    }
}

/// A decision as the wire carries it; its lease is its answer's to add
/// ([`decided`]).
pub(super) fn decision(decision: &Decision) -> proto::Decision {
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

/// The record's trace: the five facts it holds. Its count fills two of the
/// wire's fields: `survivors` where a pass counted, and `drew_first` where
/// the decision drew and counted none. The observability contract fills
/// the wire's other fields; they stay unset until then.
fn trace(trace: &Trace) -> proto::Trace {
    proto::Trace {
        profile: u32::from(trace.profile),
        survivors: trace.survivors.get(),
        drew_first: trace.drew_first(),
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
        lifted: proto::LiftedVeto::from(trace.lifted) as i32,
        ..proto::Trace::default()
    }
}

#[cfg(test)]
mod tests {
    use super::{busy, credit, decided, emptied, Checked, Lists, Refusal, Viewed, MAX_FRAME};
    use crate::{
        decision::{
            Decision, Lifted, Pair, Pick, Queue, Shed, ShedReason, Survivors, Target, Targets,
            Trace,
        },
        proto::{self, busy::Reason, decide_answer::Answer, decision::Outcome, identity::Form},
        spec::MAX_BLOCK_HASHES,
        view::{
            fingerprint, AffinityFacts, Constraints, HashKind, Header, Identity, MediaCounts,
            ModelFingerprint, Phase, RequestFacts, WireModes, WorkerId,
        },
    };

    fn key_form(form: Form) -> proto::Identity {
        proto::Identity { form: Some(form) }
    }

    /// A frame whose every field holds a value of its own.
    fn full() -> proto::DecideRequest {
        proto::DecideRequest {
            request_id: vec![7, 8, 9],
            facts: Some(proto::RequestFacts {
                model: "model-a".to_owned(),
                namespace_seed: 11,
                tenant: Some(key_form(Form::Plain("tenant-a".to_owned()))),
                priority: -100,
                streaming: true,
                max_output_tokens: 12,
                prompt_tokens: 13,
                block_hashes: vec![14, 15, 16],
                hash_kind: proto::HashKind::Text.into(),
                session_key: Some(key_form(Form::Hash(17))),
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
            again: false,
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
            stalled: &[],
            held: &[],
            in_flight: None,
            max_block_hashes: 256,
            named: None,
            pins: None,
        };
        checked.read(lists, view, check)
    }

    #[test]
    fn every_field_of_a_frame_reaches_the_engines_request() {
        let checked = Checked::new(full()).expect("a frame the record holds");
        assert_eq!(
            (checked.model(), checked.phase(), checked.facts.seed),
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
            // The hash form is the session's value, as it was sent.
            assert_eq!(facts.session().get(), 17);
            assert_eq!(facts.session_key.plain, None);
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
        // Nothing but its id, which the contract's range asks of every frame.
        let bare = || proto::DecideRequest {
            request_id: vec![1],
            ..proto::DecideRequest::default()
        };
        let checked = Checked::new(bare()).expect("a frame of its id alone");
        assert_eq!(
            (checked.model(), checked.phase(), checked.facts.seed),
            ("", Phase::Single, 0)
        );
        assert!(checked.budget().is_zero());
        reading(bare(), &mut Lists::default(), |facts, constraints| {
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

    /// The contract's own numbers: the key "conversation-7" under the model
    /// "model-a" names one session a phase, and names it in either form.
    /// The hash form is the session's value: it is never read as a key's
    /// fingerprint and scoped again.
    #[test]
    fn a_session_key_names_the_contracts_session_in_its_plain_form_and_in_its_hash_form() {
        let sessions: [(proto::Phase, u64); 4] = [
            (proto::Phase::Single, 0x6e88_ff71_cc98_2a42),
            (proto::Phase::Prefill, 0x619f_dbb0_c262_9a55),
            (proto::Phase::Decode, 0x3e92_fbbd_183d_f4b6),
            (proto::Phase::Encode, 0x3780_ba54_336a_57d0),
        ];
        let named = |phase: proto::Phase, form: Option<Form>| {
            let frame = with(|facts| {
                facts.phase = phase.into();
                facts.session_key = Some(proto::Identity { form });
            });
            reading(frame, &mut Lists::default(), |facts, _| {
                let key = facts.session_key;
                (
                    facts.session().get(),
                    key.hash,
                    key.plain.map(str::to_owned),
                )
            })
        };
        let key = fingerprint(b"conversation-7");
        for (phase, session) in sessions {
            let plain = "conversation-7".to_owned();
            assert_eq!(
                named(phase, Some(Form::Plain(plain.clone()))),
                (session, key, Some(plain)),
                "{phase:?}"
            );
            // The same session, and the same key for a stage that reads it.
            assert_eq!(
                named(phase, Some(Form::Hash(session))),
                (session, key, None),
                "{phase:?}"
            );
        }
        // No form, a value of zero and an empty key are no session.
        for absent in [None, Some(Form::Hash(0)), Some(Form::Plain(String::new()))] {
            let read = named(proto::Phase::Single, absent.clone());
            assert_eq!(read, (0, 0, None), "{absent:?}");
        }
        // The one value that would give the fingerprint no key has is no
        // key's session.
        let keyless = with(|facts| {
            facts.phase = proto::Phase::Single.into();
            facts.session_key = Some(key_form(Form::Hash(0xc771_96d2_b9cf_9497)));
        });
        let refused = Checked::new(keyless).expect_err("no key's session");
        assert_eq!(refused, Refusal::OutOfRange("facts.session_key.hash"));
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
        // A frame's headers by their count: 256 are read, each a record of
        // its own, and one more is refused by the list's path. An empty
        // header is two bytes as encoded.
        let headers = |count| with(|facts| facts.headers = vec![proto::Header::default(); count]);
        let read = reading(headers(256), &mut Lists::default(), |facts, _| {
            facts.headers.len()
        });
        assert_eq!(read, 256);
        assert_eq!(
            Checked::new(headers(257)).expect_err("a header past a view's names"),
            Refusal::OutOfRange("facts.headers")
        );
        let size = |count| {
            let facts = headers(count).facts.expect("the full frame has facts");
            prost::Message::encoded_len(&facts)
        };
        assert_eq!(size(257) - size(0), 2 * 257);
        // A wire-mode bit this build does not know is dropped, wherever in
        // the wire's 32 it stands. A gateway that states such bits alone
        // still states wires: none this build knows, and the record reads
        // that as it reads a gateway that states none.
        let stating = |bits: u32| {
            let mut frame = full();
            frame
                .constraints
                .as_mut()
                .expect("the full frame has constraints")
                .wire_modes = bits;
            let stated = Checked::new(frame.clone())
                .expect("a frame the record holds")
                .wires();
            let read = reading(frame, &mut Lists::default(), |_, constraints| {
                constraints.wire_modes
            });
            (stated, read)
        };
        assert_eq!(
            stating(0b1_0000_1001),
            (Some(WireModes::HTTP), WireModes::HTTP)
        );
        assert_eq!(
            stating(0b1_0000_0000),
            (Some(WireModes::NONE), WireModes::NONE)
        );
        assert_eq!(stating(0), (None, WireModes::NONE));
    }

    /// The text is said and dropped at the check: the frame a queue holds
    /// keeps none of it, its buffer neither.
    #[test]
    fn a_frame_that_carries_its_prompt_as_text_is_checked_and_says_so() {
        let plain = Checked::new(full()).expect("a frame the record holds");
        assert!(!plain.has_text());
        let text = Checked::new(with(|facts| facts.prompt_text = b"hello".to_vec()))
            .expect("text is no field out of its range");
        assert!(text.has_text());
        assert_eq!(text.facts.prompt_text.capacity(), 0);
    }

    /// The two words few frames say are a bit each of one byte: a frame
    /// that says neither reads as none, and each word reads apart from
    /// the other.
    #[test]
    fn a_frame_that_is_asked_again_says_so_beside_its_text() {
        let said = |frame: proto::DecideRequest| {
            let frame = Checked::new(frame).expect("no word is a field out of its range");
            (frame.says_more(), frame.has_text(), frame.again())
        };
        let texted = |text: &[u8], again| {
            let mut frame = with(|facts| facts.prompt_text = text.to_vec());
            frame.again = again;
            frame
        };
        assert_eq!(said(full()), (false, false, false));
        assert_eq!(said(texted(b"", true)), (true, false, true));
        assert_eq!(said(texted(b"hello", false)), (true, true, false));
        assert_eq!(said(texted(b"hello", true)), (true, true, true));
    }

    #[test]
    fn the_hashes_past_what_the_generation_reads_are_cut_at_the_read_and_the_first_ones_kept() {
        let frame = with(|facts| facts.block_hashes = (1..=6).collect());
        let checked = Checked::new(frame).expect("a frame the record holds");
        assert_eq!(checked.hashes(), 6);
        let read = |max_block_hashes| {
            let view = Viewed {
                now_micros: 0,
                budget_micros: 1,
                affinity: &[],
                stalled: &[],
                held: &[],
                in_flight: None,
                max_block_hashes,
                named: None,
                pins: None,
            };
            checked.read(&mut Lists::default(), view, |facts, _| {
                facts.block_hashes.to_vec()
            })
        };
        assert_eq!(read(4), [1, 2, 3, 4]);
        assert_eq!(read(6), [1, 2, 3, 4, 5, 6]);
        assert_eq!(read(7), [1, 2, 3, 4, 5, 6]);
    }

    /// The spec's range for `view.max_block_hashes` ends where the frame's
    /// bound does. A frame of that many hashes and an id of one byte is
    /// read, five bytes under the bound, and one hash more carries it over.
    #[test]
    fn the_specs_highest_cap_of_hashes_is_what_one_frame_carries() {
        let frame = |hashes: u32| proto::DecideRequest {
            request_id: vec![1],
            facts: Some(proto::RequestFacts {
                block_hashes: (0..u64::from(hashes)).collect(),
                ..proto::RequestFacts::default()
            }),
            constraints: None,
            again: false,
        };
        let at = frame(MAX_BLOCK_HASHES);
        assert_eq!(prost::Message::encoded_len(&at), MAX_FRAME - 5);
        let read = Checked::new(at).expect("a frame under the bound");
        assert_eq!(u32::try_from(read.hashes()), Ok(MAX_BLOCK_HASHES));
        let over = frame(MAX_BLOCK_HASHES + 1);
        assert_eq!(prost::Message::encoded_len(&over), MAX_FRAME + 3);
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

    /// The wire's trace carries the veto the all-vetoed rule lifted, and
    /// none for a pick no rule made.
    #[test]
    fn the_wire_carries_the_lifted_veto() {
        let steered = Trace {
            lifted: Lifted::Overload,
            ..Trace::default()
        };
        assert_eq!(
            super::trace(&steered).lifted,
            proto::LiftedVeto::Overload as i32
        );
        assert_eq!(
            super::trace(&Trace::default()).lifted,
            proto::LiftedVeto::None as i32
        );
    }

    fn traced() -> Trace {
        Trace {
            profile: 3,
            survivors: Survivors::counted(5),
            // The third entry is past the trace's length: not a part of it.
            top: [
                (WorkerId(31), 0.5),
                (WorkerId(32), 0.25),
                (WorkerId(33), 9.0),
            ],
            top_len: 2,
            seed: 41,
            generation: 6,
            ..Trace::default()
        }
    }

    fn wire_trace() -> proto::Trace {
        proto::Trace {
            profile: 3,
            survivors: Some(5),
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
        let answer = decided(vec![1, 2], 9, &pick, None);
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

    /// A decision that drew counted no survivor, and the wire says so in
    /// the contract's two words: `survivors` stays unset, which is no
    /// count and no zero, and `drew_first` is set. The trace's other facts
    /// leave as a counted trace's do.
    #[test]
    fn a_drawn_decisions_trace_says_drew_first_and_carries_no_count() {
        let mut targets = Targets::new();
        targets.push(Target {
            position: 0,
            id: WorkerId(53),
            dp_rank: 0,
            pair: None,
        });
        let pick = Decision::Pick(Pick {
            targets,
            trace: Trace {
                survivors: Survivors::NOT_COUNTED,
                ..traced()
            },
            ..Pick::default()
        });
        let (_, trace, _) = carried(decided(Vec::new(), 1, &pick, None));
        let drawn = proto::Trace {
            survivors: None,
            drew_first: true,
            ..wire_trace()
        };
        assert_eq!(trace, Some(drawn));
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
            let (outcome, trace, leases) = carried(decided(Vec::new(), 1, &shed, None));
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
        let (outcome, trace, leases) = carried(decided(Vec::new(), 1, &queue, None));
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
    fn a_credit_frame_carries_its_term_and_its_grant_and_no_requests_id() {
        assert_eq!(
            credit(3, 64),
            proto::DecideAnswer {
                request_id: Vec::new(),
                term: 3,
                answer: Some(Answer::Credit(proto::Credit { window: 64 })),
            }
        );
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
