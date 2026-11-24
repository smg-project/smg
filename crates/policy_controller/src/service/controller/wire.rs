//! The decide wire (design 5.2, appendix A): a `DecideRequest` as the
//! engine reads a request, and the engine's decision as the wire's answer.
//!
//! A frame is checked once, where it is read ([`Checked::new`]): what the
//! wire can say and the engine's record cannot hold is refused there, by
//! the field's name, before the request takes a place in a queue. The read
//! after it ([`Lists::of`]) borrows the frame and cannot fail: the
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
//! ([`decided`]), with the time its gateway has to name the dispatch.
//! A pair carries two sides under one id, prefill first. A shed and a
//! queue ticket carry no lease.

use std::{
    mem,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crate::{
    document::{
        operator::{Document, NeedsRow, Status},
        spec::{AppliedSpec, MAX_HEADERS},
    },
    engine::{
        decision::{Decision, ShedReason, Target},
        demands::Demands,
        view::{
            fingerprint, key_of_session, Constraints, HashKind, Header, Identity, MediaCounts,
            ModelFingerprint, Phase, RequestFacts, Session, WireModes, WorkerId,
        },
    },
    service::proto::{
        self, busy::Reason, decide_answer::Answer, decision::Outcome, identity::Form,
    },
    store::{DecodeIssue, Hand, MemoryStore},
    tracking::lease::{Lease, LeaseId, ReleaseReason, Said, Side},
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
/// ([`MAX_BLOCK_HASHES`](crate::document::spec::MAX_BLOCK_HASHES)).
pub(crate) const MAX_FRAME: usize = 64 * 1024;

/// The bit of a [`Checked`] frame's rare words for one that carried its
/// prompt as text.
const TEXT: u8 = 1;

/// The bit for a frame whose gateway asks its request again
/// (`DecideRequest.again`).
const AGAIN: u8 = 1 << 1;

/// The bit for a pair's frame whose decode is sent after its prefill has
/// run (`DecideRequest.decode_later`).
const DECODE_LATER: u8 = 1 << 2;

/// The bit for a frame that asks the decode side of a lease
/// (`DecideRequest.decode_of`).
const DECODE_OF: u8 = 1 << 3;

/// The lease a frame asks the decode side of (`DecideRequest.decode_of`),
/// `0` where it asks none. An id of 0 is no lease's and out of the field's
/// range. The check of a frame received reads the field here, and so does
/// a sender whose callers share one stream, as both read [`ranges`].
#[inline(always)]
pub(crate) fn asked_decode(decode_of: Option<u64>) -> Result<u64, Refusal> {
    match decode_of {
        Some(0) => Err(Refusal::OutOfRange("decode_of")),
        asked => Ok(asked.unwrap_or(0)),
    }
}

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
        // A text request states its window: the window is its host's
        // choice, and the view's block size is in tokens. One that carries
        // hashes and states none cannot be priced. One with no hash prices
        // nothing and is taken.
        Ok(proto::HashKind::Text) if facts.block_size == 0 && !facts.block_hashes.is_empty() => {
            return Err(Refusal::OutOfRange("facts.block_size"));
        }
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
    /// [`AGAIN`], [`DECODE_LATER`], [`DECODE_OF`]): `0` for a frame that
    /// says none, so a decider reads them by one test.
    rare: u8,
    /// The lease whose decode side the frame asks; `0` for a frame that
    /// asks none.
    decode_of: u64,
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
        let decode_of = asked_decode(request.decode_of)?;
        // A frame that asks a lease's decode side is read at the decode
        // phase, whatever its facts say, and is no pair's first request:
        // its `decode_later` is not read.
        let phase = if decode_of == 0 { phase } else { Phase::Decode };
        let later = request.decode_later && decode_of == 0;
        let rare = (u8::from(text) * TEXT)
            | (u8::from(request.again) * AGAIN)
            | (u8::from(later) * DECODE_LATER)
            | (u8::from(decode_of != 0) * DECODE_OF);
        Ok(Self {
            id: request.request_id,
            facts,
            constraints: request.constraints.unwrap_or_default(),
            phase,
            hash_kind,
            session_key,
            rare,
            decode_of,
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
    /// as text, its gateway asks its request again, its pair's decode comes
    /// later, or it asks a lease's decode side. A decider asks this of
    /// every request, once, and which word it is of those that say one.
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

    /// Whether the pair's decode is sent after its prefill has run
    /// (`DecideRequest.decode_later`): its first decision files the decode
    /// side chosen.
    pub(crate) const fn decode_later(&self) -> bool {
        self.rare & DECODE_LATER != 0
    }

    /// The lease whose decode side the frame asks
    /// (`DecideRequest.decode_of`); `None` for a frame that asks none.
    pub(crate) fn decode_of(&self) -> Option<LeaseId> {
        (self.rare & DECODE_OF != 0).then_some(LeaseId(self.decode_of))
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

    /// The request's block hashes, in order from its first block. Its one
    /// reader is a case of the client's wire.
    #[cfg(all(test, feature = "client"))]
    pub(crate) fn block_hashes(&self) -> &[u64] {
        &self.facts.block_hashes
    }

    /// The frame's facts in the engine's form: every fact the frame
    /// carries, its block hashes as deep as `depth` and its budget as
    /// `budget_micros`, what is left of it at the decision. The request
    /// borrows the frame and `lists`, the frame's own as its decider read
    /// them ([`Lists::of`]).
    ///
    /// A fact that is the deciding program's own stands at rest: the
    /// decision's instant, the rows the program names and what it lends
    /// beside its candidates. A program that holds one writes it over this.
    ///
    /// The one reader of a frame's facts: whichever engine decides, its
    /// request is made of this.
    #[inline]
    pub(crate) fn facts<'a>(
        &'a self,
        lists: &'a FrameLists<'_>,
        depth: usize,
        budget_micros: u32,
    ) -> RequestFacts<'a> {
        let facts = &self.facts;
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
        RequestFacts {
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
                .get(..depth)
                .unwrap_or(&facts.block_hashes),
            hash_kind: self.hash_kind,
            session_key: match facts.session_key.as_ref().and_then(|key| key.form.as_ref()) {
                Some(Form::Plain(key)) if !key.is_empty() => Identity::plain(key),
                Some(Form::Hash(_)) => Identity::hashed(self.session_key),
                Some(Form::Plain(_)) | None => Identity::NONE,
            },
            phase: self.phase,
            attempt: u8::try_from(facts.attempt).unwrap_or(u8::MAX),
            tried: &lists.lists.tried,
            budget_micros,
            headers: &lists.headers,
            media,
            seed: facts.seed,
            block_size: facts.block_size,
            // A frame states a count and no text's length.
            prompt_chars: 0,
            prompt_bytes: 0,
            // The decision's instant and the candidates' column are its
            // deciding program's own.
            now_micros: 0,
            affinity: &[],
            // In-process only: the rows the deciding host names.
            named: None,
            // The wire carries the name: each read of the request's
            // session fingerprints it.
            model_fingerprint: ModelFingerprint::NONE,
            // No decider asks its view for a pinned worker's row.
            pinned_at: 0,
            // What the leases book stands beside rows a view lent: no
            // frame carries it.
            lent_in_flight: None,
            session_counted: false,
            // The frame carries no signal cells: they stand beside the
            // candidates of the program that decides.
            signal_cells: None,
            client_key: facts.client_key,
            // No frame carries a ring: the ring stage places by its own,
            // as it learns it from the rows.
            ring: None,
            ring_positions: &[],
            // No frame states where a key came from: a served controller's
            // prefix ring places by any session key.
            session_key_from_rid: false,
        }
    }

    /// The frame's constraints in the engine's form: the workers its
    /// gateway just missed, the wires it states, the lifted veto, the later
    /// decode and the subset hint. What the deciding program adds of its
    /// own stands at rest: no position is held back and the request may
    /// take every place of its list.
    ///
    /// The one reader of a frame's constraints, as [`Self::facts`] is of
    /// its facts.
    #[inline]
    pub(crate) fn constraints<'a>(&'a self, lists: &'a FrameLists<'_>) -> Constraints<'a> {
        let constraints = &self.constraints;
        Constraints {
            excluded: &lists.lists.excluded,
            wire_modes: self.wires().unwrap_or_default(),
            overload_lifted: constraints.overload_lifted,
            decode_later: self.decode_later(),
            subset_hint: constraints.subset_hint,
            demands: Demands::default(),
            stalled: &[],
            held: &[],
            // No frame holds a request to places.
            places: None,
        }
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

impl Lists {
    /// Reads `frame`'s three lists onto these buffers, in the engine's
    /// form: the tried and the excluded ids, and the headers, which borrow
    /// the frame. The lists stand for as long as the answer is held
    /// ([`FrameLists`]): what an earlier frame left here is gone.
    #[inline]
    pub(crate) fn of<'a>(&'a mut self, frame: &'a Checked) -> FrameLists<'a> {
        let (facts, constraints) = (&frame.facts, &frame.constraints);
        self.tried.clear();
        self.tried.extend(facts.tried.iter().copied().map(WorkerId));
        self.excluded.clear();
        self.excluded
            .extend(constraints.excluded.iter().copied().map(WorkerId));
        let mut headers = emptied(mem::take(&mut self.headers));
        headers.extend(facts.headers.iter().map(|header| Header {
            // Checked: a name is a position below 256.
            name: u8::try_from(header.name).unwrap_or(u8::MAX),
            value: &header.value,
        }));
        FrameLists {
            lists: self,
            headers,
        }
    }
}

/// One frame's lists in the engine's form ([`Lists::of`]), lent to the
/// frame's request for as long as the request is read
/// ([`Checked::facts`], [`Checked::constraints`]). The headers borrow the
/// frame. Their buffer goes back to the decider's where this is dropped,
/// emptied: the next frame's headers take it.
#[derive(Debug)]
pub(crate) struct FrameLists<'a> {
    lists: &'a mut Lists,
    headers: Vec<Header<'a>>,
}

impl Drop for FrameLists<'_> {
    fn drop(&mut self) {
        self.lists.headers = emptied(mem::take(&mut self.headers));
    }
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

/// What a pick's lease is handed over with, beside the lease. One word of
/// the three: the queue's entry stays the size it was.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Beside {
    /// A single's lease: nothing.
    Alone,
    /// The session key of the decode side's lease of a pair's pick. That
    /// lease is the prefill side's under this key: one id, one record, and
    /// at the answer one index and one count of tokens.
    Decode(Session),
    /// The issue of a decode call's pick. Where the answer does not leave,
    /// its sender takes this one issue back: the prefill that runs under
    /// the same lease keeps its booking ([`Leased::given_up`]).
    Issue(DecodeIssue),
}

/// A pick's lease as its decider hands it over: the lease the store filed,
/// what its decision left of the lease's dispatch span, and when that span
/// ends.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Leased {
    /// The lease as the store filed it.
    pub(crate) lease: Lease,
    /// What the lease is handed over with.
    pub(crate) beside: Beside,
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

    /// Gives the lease up for an answer that does not leave, under
    /// `reason`, by a word handed to its file through `hand`, as a
    /// `Released` frame's word is: whoever may act at the file next applies
    /// it and tells its engine, so no stage hears it beside its decider at
    /// work. A decode call's answer takes its one issue back, and the side
    /// stands chosen for the first pick's dispatch
    /// ([`MemoryStore::hand_taken_back`]).
    /// Any other lease is released whole.
    pub(crate) fn given_up(&self, store: &MemoryStore, hand: &Hand, reason: ReleaseReason) {
        match self.beside {
            Beside::Issue(issue) => {
                store.hand_taken_back(hand, self.lease.id, issue, reason);
            }
            Beside::Alone | Beside::Decode(_) => {
                store.hand(hand, self.lease.id, Said::Released(None, reason));
            }
        }
    }

    /// Whether too little of the span is left to hand the lease over.
    pub(crate) fn spent(&self) -> bool {
        self.left < Self::LEAST
    }

    /// Whether the lease's id was minted for this answer. A decode call's
    /// was not: its lease is its pair's, minted for that pair's first
    /// answer, under that answer's term.
    pub(crate) const fn minted(&self) -> bool {
        !matches!(self.beside, Beside::Issue(_))
    }
}

/// A decider's answer as its stream's queue holds it. A decision waits as
/// it was made, with the lease issued for its pick, and its frame is built
/// where its stream sends it, behind that send's read of the fence and of
/// the lease's span: the lists a frame carries (a pick's targets, the
/// trace's top, the lease) are made and freed on the thread that encodes
/// them, and a deciding thread makes none.
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "a decision waits in the queue as it was made; a box would be an allocation on the deciding thread"
)]
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
/// carries two leases of one id, the prefill side's first, or the prefill
/// side's alone where its decode side was not issued.
pub(crate) fn decided(
    id: Vec<u8>,
    term: u64,
    decision: &Decision,
    lease: Option<&Leased>,
) -> proto::DecideAnswer {
    let mut decision = self::decision(decision);
    if let Some(leased) = lease {
        let decode = match leased.beside {
            Beside::Decode(session) => Some(session),
            Beside::Alone | Beside::Issue(_) => None,
        };
        let mut leases = Vec::with_capacity(1 + usize::from(decode.is_some()));
        leases.push(self::lease(&leased.lease, leased.left));
        if let Some(session) = decode {
            let decode = Lease {
                side: Side::Decode,
                session,
                ..leased.lease
            };
            leases.push(self::lease(&decode, leased.left));
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

/// Where a controller's lease id holds the term that minted it: the bits
/// from this one up, over the term's counter.
const MINTED_UNDER: u32 = 32;

/// A lease this controller minted, as the wire carries it under the term
/// that minted it, with `left`: what is left, where it is sent, of the time
/// its gateway has to name the dispatch. The term is the one in the id's
/// own bits ([`MINTED_UNDER`]): a decode call answered under a later term
/// names a lease an earlier one minted. The decide call issues one kind of
/// lease, a credit: on a single request's target, or on each side of a
/// pair under one id.
fn lease(lease: &Lease, left: Duration) -> proto::Lease {
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
        term: lease.id.0 >> MINTED_UNDER,
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
        trace: Some(decision.trace().wire()),
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

/// A redacted snapshot row, with its age at the take's clock.
pub(super) fn snapshot_row(row: &crate::store::WorkerRow, now: Instant) -> proto::WorkerRow {
    let mut told = super::worker_wire::row(row);
    told.admitted_age_ms = super::replication::age_ms(now, row.admitted_at);
    told
}

/// The operator's status in the contract's form.
pub(super) fn spec_status(status: Status) -> proto::SpecStatus {
    let mode = if status.applied.is_some() {
        proto::ApplyMode::Enforce
    } else {
        proto::ApplyMode::Unspecified
    };
    proto::SpecStatus {
        generation: status.generation,
        digest: status
            .digest
            .map_or_else(String::new, |digest| digest.to_string()),
        applied_unix_micros: status.applied.map_or(0, unix_micros),
        mode: mode.into(),
        retained: status.retained,
        source: if status.writable { "api" } else { "file" }.to_owned(),
    }
}

/// The source's document as the spec read call answers it.
pub(super) fn spec_document(document: Document) -> proto::PolicySpecDocument {
    proto::PolicySpecDocument {
        generation: document.generation,
        digest: document.digest.to_string(),
        document: document.document,
        format: document.format.as_str().to_owned(),
        applied_unix_micros: unix_micros(document.applied),
        mode: proto::ApplyMode::Enforce.into(),
        acks: Vec::new(),
    }
}

/// The source's document and its status as a standby is sent them.
pub(super) fn generation_event(document: Document) -> proto::GenerationEvent {
    let digest = document.digest.to_string();
    proto::GenerationEvent {
        status: Some(proto::SpecStatus {
            generation: document.generation,
            digest: digest.clone(),
            applied_unix_micros: unix_micros(document.applied),
            mode: proto::ApplyMode::Enforce.into(),
            ..proto::SpecStatus::default()
        }),
        document: Some(proto::SpecPush {
            generation: document.generation,
            digest,
            document: document.document,
            format: document.format.as_str().to_owned(),
            needs: None,
        }),
    }
}

/// The feeds before and after a document's verdict.
pub(super) fn needs_diff((before, after): (Vec<NeedsRow>, Vec<NeedsRow>)) -> proto::NeedsDiff {
    proto::NeedsDiff {
        before: Some(needs_summary(before)),
        after: Some(needs_summary(after)),
    }
}

/// A time as the contract carries it, or zero where it cannot be stated.
pub(super) fn unix_micros(at: SystemTime) -> i64 {
    at.duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|since| i64::try_from(since.as_micros()).ok())
        .unwrap_or(0)
}

/// What a host must feed for these rows, as the contract writes it: a row
/// for each of the engine's rows that has a flag to set
/// ([`crate::engine::Engine::needs_rows`]), in their order. A need the contract's row has
/// no flag for is not told here: a worker's requests in flight, the
/// sessions in flight, a key's in-flight count, the affinity column and
/// the signal cells are the controller's own to hold.
pub(super) fn needs_summary(rows: Vec<NeedsRow>) -> proto::NeedsSummary {
    let models = rows
        .into_iter()
        .filter(|NeedsRow { needs, .. }| {
            needs.text
                || needs.token_hashes
                || needs.text_hashes
                || needs.loads
                || needs.events
                || needs.ring
                || needs.headers
                || needs.media
        })
        .map(
            |NeedsRow {
                 model,
                 phase,
                 needs,
             }| proto::ModelNeeds {
                model,
                role: role(phase).into(),
                text: needs.text,
                token_hashes: needs.token_hashes,
                text_hashes: needs.text_hashes,
                loads: needs.loads,
                events: needs.events,
                ring: needs.ring,
                headers: needs.headers,
                media: needs.media,
            },
        )
        .collect();
    proto::NeedsSummary { models }
}

/// The role a phase's decision places on, as the contract writes it.
const fn role(phase: Phase) -> proto::Role {
    match phase {
        Phase::Single => proto::Role::Single,
        Phase::Prefill => proto::Role::Prefill,
        Phase::Decode => proto::Role::Decode,
        Phase::Encode => proto::Role::Encode,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{busy, credit, decided, Checked, Lists, Refusal, MAX_FRAME};
    use crate::{
        document::spec::{AppliedSpec, MAX_BLOCK_HASHES},
        engine::{
            decision::{
                self, Decision, LiftedVeto, Pair, Pick, Queue, Shed, ShedReason, Survivors, Target,
                Targets, Trace,
            },
            decision_engine::{Request, RequestMaker, WorkerLists},
            view::{Constraints, Phase, RequestFacts, WorkerId},
        },
        service::{
            controller::frame_request::FrameRequest,
            proto::{self, busy::Reason, decide_answer::Answer, decision::Outcome, identity::Form},
        },
    };

    #[test]
    fn plain_spec_answers_keep_each_wire_field() {
        use std::time::{Duration, UNIX_EPOCH};

        use prost::Message as _;

        use crate::document::{
            operator::{Document, Format, NeedsRow, Status},
            spec::PolicySpec,
            Needs,
        };

        let digest = PolicySpec::preset("random")
            .expect("a policy")
            .digest()
            .expect("a digest");
        let applied = UNIX_EPOCH + Duration::from_micros(123_456);
        let status = super::spec_status(Status {
            generation: 7,
            digest: Some(digest),
            applied: Some(applied),
            retained: vec![2, 4],
            writable: true,
        });
        assert_eq!(
            status.encode_to_vec(),
            proto::SpecStatus {
                generation: 7,
                digest: digest.to_string(),
                applied_unix_micros: 123_456,
                mode: proto::ApplyMode::Enforce.into(),
                retained: vec![2, 4],
                source: "api".to_owned(),
            }
            .encode_to_vec()
        );
        let empty = super::spec_status(Status {
            generation: 0,
            digest: None,
            applied: None,
            retained: Vec::new(),
            writable: false,
        });
        assert_eq!(
            empty,
            proto::SpecStatus {
                source: "file".to_owned(),
                ..proto::SpecStatus::default()
            }
        );
        let document = super::spec_document(Document {
            generation: 7,
            digest,
            document: b"the source's bytes\n".to_vec(),
            format: Format::Yaml,
            applied,
        });
        assert_eq!(
            document.encode_to_vec(),
            proto::PolicySpecDocument {
                generation: 7,
                digest: digest.to_string(),
                document: b"the source's bytes\n".to_vec(),
                format: "yaml".to_owned(),
                applied_unix_micros: 123_456,
                mode: proto::ApplyMode::Enforce.into(),
                acks: Vec::new(),
            }
            .encode_to_vec()
        );
        let row = |model: &str, phase, needs| NeedsRow {
            model: model.to_owned(),
            phase,
            needs,
        };
        let diff = super::needs_diff((
            vec![
                row(
                    "local",
                    Phase::Single,
                    Needs {
                        affinity: true,
                        ..Needs::default()
                    },
                ),
                row(
                    "paired",
                    Phase::Prefill,
                    Needs {
                        headers: true,
                        media: true,
                        ..Needs::default()
                    },
                ),
            ],
            vec![row(
                "paired",
                Phase::Decode,
                Needs {
                    text: true,
                    token_hashes: true,
                    text_hashes: true,
                    loads: true,
                    events: true,
                    ring: true,
                    ..Needs::default()
                },
            )],
        ));
        assert_eq!(
            diff.encode_to_vec(),
            proto::NeedsDiff {
                before: Some(proto::NeedsSummary {
                    models: vec![proto::ModelNeeds {
                        model: "paired".to_owned(),
                        role: proto::Role::Prefill.into(),
                        headers: true,
                        media: true,
                        ..proto::ModelNeeds::default()
                    }],
                }),
                after: Some(proto::NeedsSummary {
                    models: vec![proto::ModelNeeds {
                        model: "paired".to_owned(),
                        role: proto::Role::Decode.into(),
                        text: true,
                        token_hashes: true,
                        text_hashes: true,
                        loads: true,
                        events: true,
                        ring: true,
                        ..proto::ModelNeeds::default()
                    }],
                }),
            }
            .encode_to_vec(),
        );
    }

    use crate::{
        document::{
            operator::{needs_rows, test_support::LEAST_LOAD},
            spec::PolicySpec,
        },
        engine::{Candidate, Filter},
        stages::StageRegistry,
    };

    /// A need the contract's row has no flag for writes no row: the engine
    /// reads the affinity column of a host's own stage in every phase, and
    /// the column is the controller's to hold.
    #[test]
    fn a_need_with_no_flag_in_the_contract_writes_no_row() {
        #[derive(Debug)]
        struct Held;

        impl Filter for Held {
            fn name(&self) -> &'static str {
                "held"
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }

            fn needs(&self) -> crate::document::Needs {
                crate::document::Needs {
                    affinity: true,
                    ..crate::document::Needs::default()
                }
            }
        }

        let mut stages = StageRegistry::builtin();
        stages
            .filter("held", || Box::new(Held))
            .expect("a free name");
        let document = LEAST_LOAD.replace("[healthy]", "[held]");
        let spec: PolicySpec = serde_yaml::from_str(&document).expect("a document");
        let engine = spec.validate_in(&stages, &[]).expect("validates");
        let rows = super::needs_summary(needs_rows(&engine)).models;
        // The profile's scorer reads the loads: its rows stand, and say
        // nothing of the column.
        assert_eq!(rows.len(), 4, "{rows:?}");
        let quiet = document
            .replace("    score: [least-load]\n", "")
            .replace("best", "random");
        let spec: PolicySpec = serde_yaml::from_str(&quiet).expect("a document");
        let engine = spec.validate_in(&stages, &[]).expect("validates");
        assert_eq!(engine.needs_rows().len(), 4, "the column in every phase");
        assert_eq!(
            super::needs_summary(needs_rows(&engine)),
            proto::NeedsSummary::default()
        );
    }

    /// The summary tells each row under its own role and each need under
    /// its own flag: a pair's prefill row and its decode row part, a row of
    /// media alone is told, and a sessions rule's key is a row of its
    /// binding in every phase it covers.
    #[test]
    fn the_summary_tells_each_role_and_each_flag_apart() {
        /// Keeps every worker and reads what it is built to read.
        #[derive(Debug)]
        struct Reads(&'static str, crate::document::Needs);

        impl Filter for Reads {
            fn name(&self) -> &'static str {
                self.0
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }

            fn needs(&self) -> crate::document::Needs {
                self.1
            }
        }

        let tokens = crate::document::Needs {
            token_hashes: true,
            ..crate::document::Needs::default()
        };
        let media = crate::document::Needs {
            media: true,
            ..crate::document::Needs::default()
        };
        let mut stages = StageRegistry::builtin();
        stages
            .filter("tokens", move || Box::new(Reads("tokens", tokens)))
            .expect("a free name")
            .filter("media", move || Box::new(Reads("media", media)))
            .expect("a free name");
        let document = r#"{
          "apiVersion": "smg.policy/v1",
          "view": { "headers": ["x-session-id"] },
          "profiles": {
            "prefill": { "filter": ["tokens"], "pick": "random" },
            "decode": { "filter": ["media"], "pick": "random" },
            "quiet": { "pick": "random" }
          },
          "bindings": [
            { "match": { "model": "pd/*", "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "prefill", "decode": "decode" } },
            { "match": { "model": "chat/*" }, "profile": "quiet", "sessions": { "sticky_header": "x-session-id", "ttl_s": 600 } },
            { "match": { "model": "*" }, "profile": "quiet" }
          ]
        }"#;
        let spec: PolicySpec = serde_json::from_str(document).expect("a document");
        let engine = spec.validate_in(&stages, &[]).expect("validates");
        let rows = super::needs_summary(needs_rows(&engine)).models;
        let row = |model: &str, role: proto::Role| {
            rows.iter()
                .find(|row| row.model == model && row.role == i32::from(role))
                .cloned()
        };
        let told = |model: &str, role: proto::Role| proto::ModelNeeds {
            model: model.to_owned(),
            role: role.into(),
            ..proto::ModelNeeds::default()
        };
        assert_eq!(
            row("pd/*", proto::Role::Prefill),
            Some(proto::ModelNeeds {
                token_hashes: true,
                ..told("pd/*", proto::Role::Prefill)
            })
        );
        assert_eq!(
            row("pd/*", proto::Role::Decode),
            Some(proto::ModelNeeds {
                media: true,
                ..told("pd/*", proto::Role::Decode)
            })
        );
        // The sessions rule reads its key under the view's headers, in
        // every phase its binding covers.
        for role in [
            proto::Role::Single,
            proto::Role::Prefill,
            proto::Role::Decode,
            proto::Role::Encode,
        ] {
            assert_eq!(
                row("chat/*", role),
                Some(proto::ModelNeeds {
                    headers: true,
                    ..told("chat/*", role)
                })
            );
        }
        assert_eq!(rows.len(), 6, "{rows:?}");
    }

    fn key_form(form: Form) -> proto::Identity {
        proto::Identity { form: Some(form) }
    }

    /// A frame whose every field holds a value of its own.
    pub(crate) fn full() -> proto::DecideRequest {
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
                client_key: 29,
            }),
            constraints: Some(proto::Constraints {
                excluded: vec![26, 27, 28],
                wire_modes: 0b110,
                overload_lifted: true,
                subset_hint: Some(0),
            }),
            again: false,
            ..Default::default()
        }
    }

    /// The room of `lists`' three buffers, and how many headers stand in
    /// theirs: what a case reads of a decider's buffers between two frames.
    pub(crate) fn kept(lists: &Lists) -> ([usize; 3], usize) {
        let room = [
            lists.tried.capacity(),
            lists.excluded.capacity(),
            lists.headers.capacity(),
        ];
        (room, lists.headers.len())
    }

    /// Makes `frame`'s request as a controller's maker makes it
    /// ([`FrameRequest`]), on `lists` and with `left` of the frame's
    /// budget, and hands `check` its facts and its constraints. The
    /// request bears the frame's id whole: that is held here, for every
    /// frame a case makes a request of. The document it is made under has
    /// a ring keyed by the prompt: it reads a prompt's hashes of either
    /// kind 256 tokens deep, past the three a case's frame carries.
    pub(crate) fn made<R>(
        frame: &Checked,
        lists: &mut Lists,
        left: std::time::Duration,
        check: impl FnOnce(&RequestFacts<'_>, &Constraints<'_>) -> R,
    ) -> R {
        let document = PolicySpec::preset("prefix_hash").expect("a built-in preset");
        let spec = AppliedSpec::validate(&document).expect("the preset validates");
        let lists = lists.of(frame);
        let maker = FrameRequest::new(frame, &lists, left, WorkerLists::None);
        let Request {
            id,
            facts,
            constraints,
        } = maker.make(&spec);
        assert_eq!(id, frame.id(), "a request bears its frame's id whole");
        check(&facts, &constraints)
    }

    /// The full frame with its facts edited.
    fn with(edit: impl FnOnce(&mut proto::RequestFacts)) -> proto::DecideRequest {
        let mut frame = full();
        edit(frame.facts.as_mut().expect("the full frame has facts"));
        frame
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

    /// A pair's two words of its decode ride the same byte. A frame keeps
    /// its `decode_later` bit, and its `decode_of` id as it came. A frame
    /// that asks a lease's decode side is read at the decode phase, whatever
    /// its facts say, and is no pair's first request: its bit is not read.
    #[test]
    fn a_frame_keeps_its_decode_later_bit_and_its_decode_of_id() {
        use crate::engine::view::Phase;
        let said = |later: bool, of: Option<u64>| {
            let mut frame = with(|facts| facts.phase = proto::Phase::Prefill.into());
            (frame.decode_later, frame.decode_of) = (later, of);
            let frame = Checked::new(frame).expect("no word is a field out of its range");
            let of = frame.decode_of().map(|lease| lease.0);
            (frame.says_more(), frame.decode_later(), of, frame.phase())
        };
        let lease = 1 << 32 | 7;
        assert_eq!(said(false, None), (false, false, None, Phase::Prefill));
        assert_eq!(said(true, None), (true, true, None, Phase::Prefill));
        let call = (true, false, Some(lease), Phase::Decode);
        assert_eq!(said(false, Some(lease)), call);
        assert_eq!(said(true, Some(lease)), call);
        let last = Some(u64::MAX);
        assert_eq!(said(false, last), (true, false, last, Phase::Decode));
    }

    /// No lease has the id 0: a frame that names it is refused by its
    /// field, as any field out of its range is.
    #[test]
    fn a_decode_of_zero_is_refused_by_its_field() {
        let mut frame = full();
        frame.decode_of = Some(0);
        let refused = Checked::new(frame).err();
        assert_eq!(refused, Some(Refusal::OutOfRange("decode_of")));
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
            ..Default::default()
        };
        let at = frame(MAX_BLOCK_HASHES);
        assert_eq!(prost::Message::encoded_len(&at), MAX_FRAME - 5);
        let read = Checked::new(at).expect("a frame under the bound");
        assert_eq!(u32::try_from(read.hashes()), Ok(MAX_BLOCK_HASHES));
        let over = frame(MAX_BLOCK_HASHES + 1);
        assert_eq!(prost::Message::encoded_len(&over), MAX_FRAME + 3);
    }

    /// The wire's trace carries the veto the all-vetoed rule lifted, and
    /// none for a pick no rule made.
    #[test]
    fn the_wire_carries_the_lifted_veto() {
        let steered = Trace {
            lifted: LiftedVeto::Overload,
            ..Trace::default()
        };
        assert_eq!(steered.wire().lifted, proto::LiftedVeto::Overload as i32);
        assert_eq!(
            Trace::default().wire().lifted,
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
    fn a_first_profile_pick_keeps_its_trace_bytes_and_an_early_shed_adds_one_field() {
        use prost::Message as _;

        let pick = Decision::Pick(Pick {
            trace: Trace {
                profile: 0,
                survivors: Survivors::counted(1),
                generation: 6,
                ..Trace::default()
            },
            ..Pick::default()
        });
        let shed = decision::shed(ShedReason::NoEndpoints, "wires", 0, 6);
        let encoded = |decision| {
            super::decision(decision)
                .trace
                .expect("a trace")
                .encode_to_vec()
        };
        assert_eq!(encoded(&pick), [0x10, 1, 0x28, 6]);
        assert_eq!(encoded(&shed), [0x10, 0, 0x28, 6, 0x70, 1]);
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
                ..Pair::default()
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
