//! The typed decision (design 4.2): an ordered list of targets the host walks
//! at dispatch, a shed with its reason and the stage that emptied the pool, or
//! a queue ticket. A busy answer is not a decision: a remote decider gives it
//! on its stream and the host's fallback takes over at once, so the enum
//! never carries one.

use std::fmt;

#[cfg(any(test, feature = "in-process"))]
use crate::engine::view::AffinityFacts;
#[cfg(feature = "controller")]
use crate::service::proto;
use crate::{
    document::spec::RulePlaced,
    engine::{
        sticky_sessions::{DecodeVerdict, SessionVerdict},
        view::{Candidate, WorkerId},
    },
};

/// Targets a decision carries at most: the pick and its alternates.
pub const MAX_TARGETS: usize = 4;

/// Candidates a trace names with their scores.
pub const TRACE_TOP: usize = 3;

/// One target: a candidate by position and id, with the decode side when the
/// decision is a pair.
///
/// The record gains fields: code outside this crate that builds one writes the
/// rest as `..Target::default()`, and a pattern of it ends with `..`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Target {
    /// Position in the host's candidate slice (the prefill's for a pair).
    pub position: u32,
    pub id: WorkerId,
    pub dp_rank: u16,
    /// The decode side of a pair; the target itself is then the prefill.
    pub pair: Option<Pair>,
}

/// The decode side of a pair target, by position in the host's decode slice.
/// The rendezvous the two sides meet at is the host's at dispatch.
///
/// The record gains fields: code outside this crate that builds one writes the
/// rest as `..Pair::default()`, and a pattern of it ends with `..`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pair {
    pub decode_position: u32,
    pub decode_id: WorkerId,
    pub decode_dp_rank: u16,
    /// What the decision did with the decode side's session key, as the
    /// pair of a pick's first target states it: a claim names that pair's
    /// decode. The pick's own verdict ([`Pick::session_verdict`]) is the
    /// prefill side's. It is one byte the pair had spare.
    pub decode_verdict: DecodeVerdict,
}

impl Target {
    /// The target that names `record`, on no pair: its position, its id and
    /// its rank.
    #[inline]
    #[must_use]
    pub fn of(record: &Candidate) -> Self {
        Self {
            position: record.position,
            id: record.id,
            dp_rank: record.dp_rank,
            pair: None,
        }
    }

    /// The pair target of `prefill` with `decode` as its decode side
    /// ([`Pair::of`]).
    #[must_use]
    pub fn paired(prefill: &Candidate, decode: &Candidate) -> Self {
        Self {
            pair: Some(Pair::of(decode)),
            ..Self::of(prefill)
        }
    }
}

impl Pair {
    /// The decode side that names `decode`: its position in the decodes,
    /// its id and its rank, with no verdict on a session key.
    #[must_use]
    pub fn of(decode: &Candidate) -> Self {
        Self {
            decode_position: decode.position,
            decode_id: decode.id,
            decode_dp_rank: decode.dp_rank,
            decode_verdict: DecodeVerdict::NONE,
        }
    }
}

/// The ordered targets of a pick: the best first. For a pair the list reads
/// "pairs before the prefill went out, decodes of the chosen prefill after",
/// so an alternate taken after the prefill dispatched changes the decode only.
#[derive(Clone, Copy, Default)]
pub struct Targets {
    items: [Target; MAX_TARGETS],
    len: u8,
    /// Who placed the pick ([`Pick::placed_by`]) and, of a rule's pick,
    /// whether the rule keeps it from the profile (`Pick::untold`), in
    /// the byte the count and the overlap left spare between them.
    placing: Placing,
    /// What the rows the decision was given say the first target holds
    /// ([`Pick::overlap`]), in the bytes the count left spare.
    overlap: Overlap,
}

/// Two lists are equal where their targets, what they state of the first
/// and who placed them are. The entries past a count are no part of a list:
/// [`Targets::clear`] leaves them as they stood.
impl PartialEq for Targets {
    fn eq(&self, other: &Self) -> bool {
        self.placing == other.placing
            && self.overlap == other.overlap
            && self.as_slice() == other.as_slice()
    }
}

impl Eq for Targets {}

/// A list prints what its equality reads: its targets and what it states
/// of the first.
impl fmt::Debug for Targets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Targets")
            .field("items", &self.as_slice())
            .field("placing", &self.placing)
            .field("overlap", &self.overlap)
            .finish()
    }
}

impl Targets {
    pub const fn new() -> Self {
        Self {
            items: [Target {
                position: 0,
                id: WorkerId(0),
                dp_rank: 0,
                pair: None,
            }; MAX_TARGETS],
            len: 0,
            placing: Placing::Stages,
            overlap: Overlap::NONE,
        }
    }

    /// No target, of a pick the binding's sessions rule placed alone and
    /// keeps from the profile (`Pick::untold`): the pick a file of leases
    /// lends with the words of such a request's lease, which lists no
    /// target (`Sides::with_held_single`).
    pub(crate) const fn untold() -> Self {
        Self {
            placing: Placing::RuleUntold,
            ..Self::new()
        }
    }

    /// States what the first target holds of the request. A list led anew
    /// ([`Self::lead`]) says nothing of its new first target.
    #[cfg(any(test, feature = "in-process"))]
    pub(crate) fn state(&mut self, overlap: Overlap) {
        self.overlap = overlap;
    }

    /// States that the binding's sessions rule placed the pick, with no
    /// stage of the profile asked ([`PlacedBy::Rule`]), and whether that
    /// rule keeps what it places from the profile (`kept`, the rule's own
    /// word). Both are the decision's: a list led anew, emptied or turned
    /// into a pair's keeps them.
    pub(crate) fn by_rule(&mut self, kept: RulePlaced) {
        self.placing = match kept {
            RulePlaced::Told => Placing::Rule,
            RulePlaced::Untold => Placing::RuleUntold,
        };
    }

    /// Append a target; a list already full keeps what it has.
    pub fn push(&mut self, target: Target) {
        if usize::from(self.len) < MAX_TARGETS {
            self.items[usize::from(self.len)] = target;
            self.len += 1;
        }
    }

    /// Turns a decode side's targets into a pair's where they stand: each
    /// names `prefill` with the decode it held, in the order they had. The
    /// first pair states `decode`, the decode side's own session verdict,
    /// and no pair behind it states one. Nothing is stated of a pair's
    /// first target ([`Pick::overlap`]). The entries past the count are not
    /// rewritten, as [`Self::clear`] leaves them. No accessor goes past the
    /// count.
    pub(crate) fn pair_under(&mut self, prefill: &Target, decode: SessionVerdict) {
        // Taken by the first pair: the pairs behind it read none.
        let mut verdict = DecodeVerdict::of(decode);
        for target in &mut self.items[..usize::from(self.len)] {
            *target = Target {
                position: prefill.position,
                id: prefill.id,
                dp_rank: prefill.dp_rank,
                pair: Some(Pair {
                    decode_position: target.position,
                    decode_id: target.id,
                    decode_dp_rank: target.dp_rank,
                    decode_verdict: std::mem::take(&mut verdict),
                }),
            };
        }
        self.overlap = Overlap::NONE;
    }

    /// Empties the list. What was stated of its first target goes with it
    /// ([`Pick::overlap`]).
    pub fn clear(&mut self) {
        self.len = 0;
        self.overlap = Overlap::NONE;
    }

    /// Puts `target` first and keeps the others behind it in their order; a
    /// copy of `target` among them leaves, and so does the last one when the
    /// list is full.
    pub fn lead(&mut self, target: Target) {
        let mut led = Self::new();
        led.placing = self.placing;
        led.push(target);
        for t in self.as_slice() {
            if *t != target {
                led.push(*t);
            }
        }
        *self = led;
    }

    pub fn len(&self) -> usize {
        usize::from(self.len)
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn first(&self) -> Option<&Target> {
        self.as_slice().first()
    }

    pub fn as_slice(&self) -> &[Target] {
        &self.items[..usize::from(self.len)]
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Target> {
        self.as_slice().iter()
    }
}

/// Why a decision shed the request. The host maps each reason to its status
/// code and `Retry-After`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShedReason {
    /// Every candidate is over the overload thresholds and shedding is on.
    Saturated,
    /// A queue at its limit.
    FlowLimit,
    /// Nothing routable is left after the filters.
    NoEndpoints,
    /// The picker declined every surviving candidate.
    PolicyDeclined,
    /// The decision budget passed.
    Deadline,
}

/// How many candidates the filters left, where the decision counted them. A
/// decision that drew first ([`Trace::drew_first`]) asked the filters about
/// the rows it drew alone and holds no count of the pool: [`Survivors::get`]
/// answers `None` there, never a zero, and a reader prints the word, not a
/// number. The default is a count of zero.
///
/// The count and its absence are one word of the record, written by one
/// store as the bare count was: a trace that says whether it counted is the
/// size of one that could not, and a decision that draws nothing pays
/// nothing for the word.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Survivors(u32);

impl Survivors {
    /// No count: the decision drew first.
    pub const NOT_COUNTED: Self = Self(u32::MAX);

    /// A count of `kept` candidates. The top value is the mark of no count,
    /// so a count that large is held one under it: a count never reads as
    /// none.
    pub const fn counted(kept: u32) -> Self {
        Self(if kept == Self::NOT_COUNTED.0 {
            kept - 1
        } else {
            kept
        })
    }

    /// The count, where the decision made one.
    pub const fn get(self) -> Option<u32> {
        if self.0 == Self::NOT_COUNTED.0 {
            None
        } else {
            Some(self.0)
        }
    }
}

impl fmt::Debug for Survivors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(f)
    }
}

/// The soft veto lifted to make a pick in a pool that veto left nobody, in
/// the order the all-vetoed rule lifts them: the mark a host counts such
/// picks from. The rule's own pick writes it, and so does the pick of a
/// session key kept on its worker where the veto as it stands kept no
/// worker. A key handed on at its sessions rule's in-flight cap writes it
/// too, where the veto as it stands kept no worker. Every
/// other pick reads `None`, the default. One mark is the decision's, since
/// the pick and its alternates are of the one lifted pool. The mark sits in
/// a byte the record had spare: the trace is no larger for it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum LiftedVeto {
    /// No veto was lifted.
    #[default]
    None,
    /// The overload veto: every usable worker was over its thresholds.
    Overload,
    /// The liveness veto: no worker stood until it was lifted, and the pick
    /// is of the workers the liveness check doubts.
    Liveness,
}

/// The contract's word for the mark: the one mapper, for every writer of a
/// trace.
#[cfg(feature = "controller")]
impl From<LiftedVeto> for proto::LiftedVeto {
    fn from(mark: LiftedVeto) -> Self {
        match mark {
            LiftedVeto::None => Self::None,
            LiftedVeto::Overload => Self::Overload,
            LiftedVeto::Liveness => Self::Liveness,
        }
    }
}

/// The decision's trace: what the trace header and an explain call print.
/// This is the first step's subset of the record; per-filter drops,
/// per-scorer contributions, the plan, the picker and the shadow pick join it
/// with the observability contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trace {
    /// The position of the profile the decision was made under, or
    /// [`Self::NO_PROFILE`] where it was made ahead of every profile.
    pub profile: u16,
    /// Candidates after the filters, where the decision counted them
    /// ([`Survivors`]).
    pub survivors: Survivors,
    /// The top candidates and their combined scores, best first: of the set
    /// the picker compared, which is every candidate the filters left unless
    /// the picker narrows them. Where a profile's ranks put rows in front
    /// ([`Scorer::score_front`](crate::engine::Scorer::score_front)) the scores are
    /// those rows' sums among themselves: a stage asked of them alone
    /// normalizes over them, so a score need not be the one the whole pass
    /// gives its row. The rows are the whole pass's, and their order too
    /// wherever its sums tell them apart: rows it rounds to one sum it
    /// draws among, and the front's scores order them. Where a tie of its
    /// sums runs past the trace's last row, the rows listed can differ
    /// too: each is a row of that tie.
    pub top: [(WorkerId, f32); TRACE_TOP],
    pub top_len: u8,
    pub seed: u64,
    pub generation: u64,
    /// The soft veto lifted for this pick ([`LiftedVeto`]): `None` unless that
    /// veto as it stands left the pool nobody, or left a key handed on at
    /// its sessions rule's in-flight cap nobody.
    pub lifted: LiftedVeto,
}

impl Default for Trace {
    #[inline]
    fn default() -> Self {
        Self {
            profile: Self::NO_PROFILE,
            survivors: Survivors::default(),
            top: Default::default(),
            top_len: 0,
            seed: 0,
            generation: 0,
            lifted: LiftedVeto::default(),
        }
    }
}

impl Trace {
    /// The value of `profile` where the decision was made ahead of every
    /// profile: no binding resolved one. An engine holds at most this many
    /// profiles, so the value is no profile's position.
    pub const NO_PROFILE: u16 = u16::MAX;

    /// The position of the profile the decision was made under: none where
    /// it was made ahead of every profile. A shed made once a profile is
    /// resolved names it, though no stage of it ran.
    #[must_use]
    pub const fn made_under(&self) -> Option<u16> {
        if self.profile == Self::NO_PROFILE {
            None
        } else {
            Some(self.profile)
        }
    }

    /// Whether the decision drew first: its picker drew the rows it hands
    /// out and every filter of the profile was asked about each of them
    /// alone ([`Picker::draw_first`](crate::engine::Picker::draw_first)). No pass
    /// over the pool ran, so the trace counts nothing of the pool:
    /// `survivors` holds no count, which is the mark read here, and `top`
    /// is of the rows drawn. A decision by the rows its key names carries
    /// the mark with no draw, a pin followed or a ring walked
    /// ([`Engine::decide_keyed`](crate::engine::Engine::decide_keyed)): no
    /// pass ran for it either, and `top` is of the rows it hands out.
    pub const fn drew_first(&self) -> bool {
        self.survivors.get().is_none()
    }

    /// The trace on the wire. Ahead of every profile the position stays zero.
    #[cfg(feature = "controller")]
    pub(crate) fn wire(&self) -> proto::Trace {
        let profile = self.made_under();
        proto::Trace {
            profile: u32::from(profile.unwrap_or_default()),
            no_profile: profile.is_none(),
            survivors: self.survivors.get(),
            drew_first: self.drew_first(),
            top: self
                .top
                .iter()
                .take(usize::from(self.top_len))
                .map(|&(id, score)| proto::Scored {
                    worker_id: id.0,
                    score,
                    ..proto::Scored::default()
                })
                .collect(),
            seed: self.seed,
            generation: self.generation,
            lifted: proto::LiftedVeto::from(self.lifted) as i32,
            ..proto::Trace::default()
        }
    }
}

/// Where the blocks a pick's first target holds were read ([`Overlap`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum OverlapSource {
    /// Nothing is held, or nothing is stated.
    #[default]
    None,
    /// The event index.
    Index,
    /// The sent-prefix predictor.
    Predictor,
}

/// What a pick states of its first target: how many leading blocks of the
/// request the worker holds, of how many, and where they were read. A host
/// observes its cache series from it: the share held is `blocks` over `of`,
/// and a miss reads zero of `of`. `of` is zero only where nothing is stated
/// ([`Overlap::NONE`]): a request counts as one block at the least.
///
/// The engine over a host's worker lists (`PolicyEngine`) states it behind
/// a single decision, from the first target's entry of the rows that
/// decision was given: the rows the engine named from the host's own data,
/// a caller's column, or the rows a caller named. No stage is asked. A
/// decision given no rows states nothing. Neither does a pair, a request
/// decided inside a caller's token tree, or a pick the binding's sessions
/// rule keeps from the profile. [`Engine::decide`](crate::engine::Engine::decide)
/// itself states nothing.
///
/// Six bytes, in bytes a pick's targets had spare behind their count: a
/// pick is no larger for it ([`Pick::overlap`]).
///
/// The record gains fields: code outside this crate that builds one writes the
/// rest as `..Overlap::default()`, and a pattern of it ends with `..`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Overlap {
    /// The leading blocks the worker holds: the larger of its entry's two
    /// readings, the index's and the sent-prefix predictor's. Halved with
    /// `of` where the request's count is past sixteen bits.
    pub blocks: u16,
    /// The request's blocks as the prefix stage counts them: its tokens
    /// over the block size, one at the least, and never under the block
    /// hashes its host sent. A request of more than 65,535 halves this
    /// count and `blocks` together until it fits, so their share stays
    /// true.
    pub of: u16,
    /// Where `blocks` was read: the index on a tie.
    pub source: OverlapSource,
    /// The held prefix's price kept the worker in front: a deepest holder
    /// that stood over the swing's margin and under the priced one, at the
    /// mean its pass judged by. It is a prefix stage's own word
    /// ([`Scorer::said`](crate::engine::Scorer::said)): the engine over
    /// worker lists asks no stage and states `false`.
    pub margin_held: bool,
}

impl Overlap {
    /// Nothing is stated.
    pub const NONE: Self = Self {
        blocks: 0,
        of: 0,
        source: OverlapSource::None,
        margin_held: false,
    };

    /// `blocks` held of a request of `request_blocks` blocks, read from the
    /// index where `indexed` and else from the predictor, with a margin's
    /// word. Both counts are halved together, by one shift, until the
    /// request's fits sixteen bits: the share they say stays true. Whoever
    /// states an overlap ends here: a prefix stage's own word
    /// ([`Scorer::said`](crate::engine::Scorer::said)) and the engine over
    /// worker lists ([`Self::held`]), so the two say one thing of one count.
    pub(crate) fn of(blocks: u32, indexed: bool, request_blocks: u32, margin_held: bool) -> Self {
        let source = if blocks == 0 {
            OverlapSource::None
        } else if indexed {
            OverlapSource::Index
        } else {
            OverlapSource::Predictor
        };
        let shift = (u32::BITS - request_blocks.leading_zeros()).saturating_sub(u16::BITS);
        Self {
            blocks: u16::try_from(blocks >> shift).unwrap_or(u16::MAX),
            of: u16::try_from(request_blocks >> shift).unwrap_or(u16::MAX),
            source,
            margin_held,
        }
    }

    /// What a worker's entry of a decision's rows says it holds of a request
    /// of `request_blocks` blocks: the larger of the entry's two readings,
    /// the index's on a tie, and no word of a margin. No stage's gate is
    /// applied: a prefix stage counts a predicted prefix only past its
    /// threshold and reads its own source alone, so its word of the same
    /// worker may be less.
    #[cfg(any(test, feature = "in-process"))]
    pub(crate) fn held(entry: AffinityFacts, request_blocks: u32) -> Self {
        let blocks = entry.overlap_blocks.max(entry.predicted_blocks);
        Self::of(
            blocks,
            entry.overlap_blocks == blocks,
            request_blocks,
            false,
        )
    }
}

/// Who placed a pick's first target. It answers one question: did the
/// binding's sessions rule place it alone? The rule does, with no stage of
/// the profile asked, where it follows a pin of the request's key, and
/// where its own assignment places the key.
///
/// What a host does with a pick the rule placed is the rule's own word
/// beside this one ([`RulePlaced`]), read through one function
/// (`lease_events::stages_hear`): by the front in a gateway's own process,
/// and by a controller where it files the pick's lease. Under `untold` no
/// stage hears the issue or the dispatch of such a single request, the
/// record of sent prompts takes no entry behind it and a lent token tree
/// stores no prompt for it, as a gateway's sticky map keeps a request it
/// places from the gateway's policy. Under `told`, the default, the stages
/// hear both and where the prompt went is kept as behind any pick. The
/// session table hears the dispatch under either.
///
/// Who placed a pick does not cross the crate's frame: no wire message
/// carries it, since the record is written where the decision is made.
///
/// It shares one byte with the rule's word (`Pick::untold`), the one a
/// pick's targets had spare between their count and what is stated of the
/// first: a pick is no larger for it ([`Pick::placed_by`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum PlacedBy {
    /// The profile's stages. Every pick no sessions rule placed reads this:
    /// a request with no key is never the rule's, whoever placed it.
    #[default]
    Stages,
    /// The binding's sessions rule alone: a pin of the key followed, or the
    /// rule's own assignment.
    Rule,
}

/// A pick's one byte of its placing: who placed it ([`PlacedBy`]), and of
/// a pick the binding's sessions rule placed, whether that rule keeps it
/// from the profile ([`RulePlaced::Untold`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
enum Placing {
    #[default]
    Stages,
    Rule,
    RuleUntold,
}

/// A pick: ordered targets, the trace, and what a sessions rule did with
/// the request's key (`None` without one). The host copies that verdict
/// onto the lease it commits, and fills the lease's session from the
/// request ([`crate::engine::RequestFacts::session`]) for every keyed request, rule
/// or no rule: the verdict alone says whether the commit moves a pin.
///
/// The record gains fields: code outside this crate that builds one writes the
/// rest as `..Pick::default()`, and a pattern of it ends with `..`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pick {
    pub targets: Targets,
    pub trace: Trace,
    pub session_verdict: SessionVerdict,
}

impl Pick {
    /// What the pick states of its first target ([`Overlap`]): what the
    /// rows its decision was given say the worker holds of the request.
    /// [`Overlap::NONE`] where nothing is stated.
    pub const fn overlap(&self) -> Overlap {
        self.targets.overlap
    }

    /// Who placed the pick ([`PlacedBy`]): the profile's stages, unless the
    /// binding's sessions rule placed it alone.
    pub const fn placed_by(&self) -> PlacedBy {
        match self.targets.placing {
            Placing::Stages => PlacedBy::Stages,
            Placing::Rule | Placing::RuleUntold => PlacedBy::Rule,
        }
    }

    /// Whether the pick is kept from the profile: the binding's sessions
    /// rule placed it alone, and the rule says such a request is none of
    /// the profile's own ([`RulePlaced::Untold`]). It has one reader
    /// (`lease_events::stages_hear`): the words of a lease and what is kept
    /// of where a prompt went both ask it, and so does a controller where
    /// it files a pick's lease.
    #[cfg(any(test, feature = "in-process"))]
    pub(crate) const fn untold(&self) -> bool {
        matches!(self.targets.placing, Placing::RuleUntold)
    }
}

/// A shed: the reason, the stage that emptied the pool, and the pacing hint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shed {
    pub reason: ShedReason,
    /// The name of the filter or picker that left nothing.
    pub emptied_by: &'static str,
    pub retry_after_secs: u16,
    /// The binding's scale-from-zero budget lets the host hold the request.
    pub wait_allowed: bool,
    pub trace: Trace,
}

/// A queue ticket: the request waits for capacity in the binding's admission
/// order and is decided again when a signal frees some. Specified in v1; no
/// built-in issues one until the admission step, and a host that meets one
/// refuses the request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Queue {
    pub ticket: u64,
    /// The longest the ticket may wait, within the request's admission
    /// budget.
    pub max_wait_micros: u32,
    pub trace: Trace,
}

/// What a decision answers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Decision {
    Pick(Pick),
    Shed(Shed),
    Queue(Queue),
}

impl Decision {
    pub fn pick(&self) -> Option<&Pick> {
        match self {
            Self::Pick(pick) => Some(pick),
            Self::Shed(_) | Self::Queue(_) => None,
        }
    }

    pub fn trace(&self) -> &Trace {
        match self {
            Self::Pick(pick) => &pick.trace,
            Self::Shed(shed) => &shed.trace,
            Self::Queue(queue) => &queue.trace,
        }
    }
}

/// How a shadow's decision stands beside the live one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ShadowKind {
    /// The same first target, or the same outcome without a pick.
    Agreed,
    /// Both picked, with different first targets.
    OtherTarget,
    /// One picked, shed or queued where the other did something else.
    OtherOutcome,
    /// No shadow decision was made.
    Skipped,
}

impl ShadowKind {
    /// Compare the outcomes and, for two picks, their first targets.
    pub fn of(live: &Decision, shadow: &Decision) -> Self {
        use Decision::{Pick, Queue, Shed};

        match (live, shadow) {
            (Pick(a), Pick(b)) => {
                if a.targets.first() == b.targets.first() {
                    Self::Agreed
                } else {
                    Self::OtherTarget
                }
            }
            (Shed(_), Shed(_)) | (Queue(_), Queue(_)) => Self::Agreed,
            (Pick(_), Shed(_) | Queue(_))
            | (Shed(_), Pick(_) | Queue(_))
            | (Queue(_), Pick(_) | Shed(_)) => Self::OtherOutcome,
        }
    }

    /// How many kinds there are: the width of a count by kind.
    pub(crate) const COUNT: usize = 4;

    /// Every kind, in the order of its place in a count by kind: what a
    /// host walks to show such a count.
    pub const ALL: [Self; Self::COUNT] = [
        Self::Agreed,
        Self::OtherTarget,
        Self::OtherOutcome,
        Self::Skipped,
    ];

    /// The kind's place in a count by kind, under [`Self::COUNT`]. Every
    /// kind is named here: one added to the enum does not build until it
    /// takes a place.
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Agreed => 0,
            Self::OtherTarget => 1,
            Self::OtherOutcome => 2,
            Self::Skipped => 3,
        }
    }

    /// The comparison's label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Agreed => "agreed",
            Self::OtherTarget => "other-target",
            Self::OtherOutcome => "other-outcome",
            Self::Skipped => "skipped",
        }
    }
}

/// The stage a decision is shed under when its pool has rows and its
/// gateway shares a wire with none of them, or its request's runtime is
/// none of theirs
/// ([`UnmetDemand::Runtime`](crate::engine::demands::UnmetDemand::Runtime)).
pub(crate) const WIRES: &str = "wires";

/// The stage a decision is shed under when its gateway hints a subset of 0
/// of a pool that has rows: a decision over no candidates, as it asked.
#[cfg_attr(
    not(feature = "in-process"),
    expect(
        dead_code,
        reason = "the deciders that shed under it are built under `in-process`"
    )
)]
pub(crate) const SUBSET_HINT: &str = "subset-hint";

/// The stage a request is answered under for the places it is held to
/// ([`Constraints::places`](crate::engine::view::Constraints::places)). A
/// pair whose own request states places is declined under it: one set of
/// places cannot hold a pair's two lists. An encode request is shed under
/// it where a list's places name workers and none of them takes the
/// request, though a worker outside them does: it is held off every worker
/// that takes it, which no gap of the membership says.
pub(crate) const PLACES: &str = "places";

/// A shed the deciders say in the engine's place, in the shape of the
/// engine's own.
#[cfg_attr(
    not(feature = "in-process"),
    expect(
        dead_code,
        reason = "the deciders that say it are built under `in-process`"
    )
)]
pub(crate) fn shed(
    reason: ShedReason,
    emptied_by: &'static str,
    seed: u64,
    generation: u64,
) -> Decision {
    Decision::Shed(Shed {
        reason,
        emptied_by,
        retry_after_secs: 1,
        wait_allowed: false,
        trace: Trace {
            seed,
            generation,
            ..Trace::default()
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::view::Role;

    /// A target made of a record names the record's position, id and rank
    /// and no pair. A pair's target names its prefill so and its decode in
    /// the pair, with no verdict on the decode's key: the value the engine
    /// makes where it puts a decode's target under its prefill's.
    #[test]
    fn a_target_names_the_records_it_is_made_of() {
        let prefill = Candidate {
            dp_rank: 2,
            ..Candidate::ready(4, WorkerId(7), Role::Prefill)
        };
        let decode = Candidate {
            dp_rank: 1,
            ..Candidate::ready(6, WorkerId(9), Role::Decode)
        };
        let alone = Target {
            position: 4,
            id: WorkerId(7),
            dp_rank: 2,
            pair: None,
        };
        assert_eq!(Target::of(&prefill), alone);
        let side = Pair {
            decode_position: 6,
            decode_id: WorkerId(9),
            decode_dp_rank: 1,
            decode_verdict: DecodeVerdict::NONE,
        };
        assert_eq!(Pair::of(&decode), side);
        let pair = Target {
            pair: Some(side),
            ..alone
        };
        assert_eq!(Target::paired(&prefill, &decode), pair);

        let mut targets = Targets::new();
        targets.push(Target::of(&decode));
        targets.pair_under(&Target::of(&prefill), SessionVerdict::None);
        assert_eq!(targets.first(), Some(&pair));
    }

    #[test]
    fn every_shadow_kind_takes_a_place_of_its_own_under_the_count() {
        use ShadowKind::{Agreed, OtherOutcome, OtherTarget, Skipped};
        let places = [Agreed, OtherTarget, OtherOutcome, Skipped].map(ShadowKind::index);
        assert_eq!(places, [0, 1, 2, 3]);
        assert_eq!(places.len(), ShadowKind::COUNT);
        assert_eq!(ShadowKind::ALL.map(ShadowKind::index), places);
    }

    #[test]
    fn shadow_kind_compares_the_outcome_and_first_target() {
        let first = Target {
            id: WorkerId(73),
            ..Target::default()
        };
        let second = Target {
            id: WorkerId(91),
            position: 1,
            ..Target::default()
        };
        let rank = Target {
            dp_rank: 5,
            ..first
        };
        let pick = |items: &[Target]| {
            let mut targets = Targets::new();
            for &target in items {
                targets.push(target);
            }
            Decision::Pick(Pick {
                targets,
                ..Pick::default()
            })
        };
        let shed = Decision::Shed(Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by: "healthy",
            retry_after_secs: 0,
            wait_allowed: false,
            trace: Trace::default(),
        });
        let queue = |ticket| {
            Decision::Queue(Queue {
                ticket,
                max_wait_micros: 113,
                trace: Trace::default(),
            })
        };
        for (a, b, kind) in [
            (pick(&[first]), pick(&[first, second]), ShadowKind::Agreed),
            (
                pick(&[first, second]),
                pick(&[second, first]),
                ShadowKind::OtherTarget,
            ),
            (pick(&[first]), pick(&[rank]), ShadowKind::OtherTarget),
            (pick(&[first]), shed, ShadowKind::OtherOutcome),
            (pick(&[first]), queue(67), ShadowKind::OtherOutcome),
            (shed, queue(67), ShadowKind::OtherOutcome),
            (
                shed,
                Decision::Shed(Shed {
                    reason: ShedReason::Saturated,
                    emptied_by: "not-overloaded",
                    retry_after_secs: 3,
                    wait_allowed: false,
                    trace: Trace::default(),
                }),
                ShadowKind::Agreed,
            ),
            (queue(67), queue(79), ShadowKind::Agreed),
        ] {
            assert_eq!(ShadowKind::of(&a, &b), kind);
            assert_eq!(ShadowKind::of(&b, &a), kind);
        }
        for (kind, label) in [
            (ShadowKind::Agreed, "agreed"),
            (ShadowKind::OtherTarget, "other-target"),
            (ShadowKind::OtherOutcome, "other-outcome"),
            (ShadowKind::Skipped, "skipped"),
        ] {
            assert_eq!(kind.as_str(), label);
        }
    }

    #[test]
    fn a_default_trace_marks_no_profile_and_keeps_its_other_defaults() {
        let trace = Trace::default();
        assert_eq!(
            trace,
            Trace {
                profile: u16::MAX,
                survivors: Survivors::counted(0),
                top: [(WorkerId(0), 0.0); TRACE_TOP],
                top_len: 0,
                seed: 0,
                generation: 0,
                lifted: LiftedVeto::None,
            }
        );
        assert_eq!(trace.made_under(), None);
        let last = Trace {
            profile: u16::MAX - 1,
            ..trace
        };
        assert_eq!(last.made_under(), Some(u16::MAX - 1));
    }

    /// The mark of a lifted veto costs the trace no byte: it sits in one the
    /// record had spare, and a trace no rule wrote reads none.
    #[test]
    fn the_mark_of_a_lifted_veto_costs_the_trace_no_byte() {
        assert_eq!(size_of::<LiftedVeto>(), 1);
        assert_eq!(size_of::<Trace>(), 72);
        assert_eq!(Trace::default().lifted, LiftedVeto::None);
    }

    /// The overlap costs the pick no byte: it shares the eight bytes the
    /// targets' count stood alone in, a pick nothing is stated of reads
    /// none, and a list led anew or cleared says nothing of a first target.
    #[test]
    fn the_overlap_costs_the_pick_no_byte() {
        assert_eq!(size_of::<Overlap>(), 6);
        assert_eq!(size_of::<Targets>(), MAX_TARGETS * size_of::<Target>() + 8);
        assert_eq!(
            size_of::<Pick>(),
            size_of::<Targets>() + size_of::<Trace>() + size_of::<SessionVerdict>()
        );
        assert_eq!(Pick::default().overlap(), Overlap::NONE);
        assert_eq!(Overlap::default(), Overlap::NONE);
        let held = Overlap {
            blocks: 3,
            of: 4,
            source: OverlapSource::Index,
            margin_held: false,
        };
        let mut targets = Targets::new();
        for position in 0..2u32 {
            targets.push(Target {
                position,
                id: WorkerId(u64::from(position) + 1),
                dp_rank: 0,
                pair: None,
            });
        }
        targets.state(held);
        let mut pick = Pick {
            targets,
            ..Pick::default()
        };
        assert_eq!(pick.overlap(), held);
        let second = *pick.targets.as_slice().last().expect("two targets");
        pick.targets.lead(second);
        assert_eq!(pick.overlap(), Overlap::NONE);
        pick.targets.state(held);
        pick.targets.clear();
        assert_eq!(pick.overlap(), Overlap::NONE);
    }

    /// An entry of a decision's rows is stated by the larger of its two
    /// readings, the index's on a tie, with no word of a margin. An entry
    /// that holds nothing is a miss of the request's blocks, not "nothing
    /// stated". A request past sixteen bits halves both counts together.
    #[test]
    fn an_entry_is_stated_by_its_larger_reading_of_the_requests_blocks() {
        let entry = |index: u32, predicted: u32| AffinityFacts {
            overlap_blocks: index,
            predicted_blocks: predicted,
            ..AffinityFacts::default()
        };
        let stated = |blocks: u16, of: u16, source: OverlapSource| Overlap {
            blocks,
            of,
            source,
            margin_held: false,
        };
        let cases = [
            (entry(3, 0), 8, stated(3, 8, OverlapSource::Index)),
            (entry(0, 5), 8, stated(5, 8, OverlapSource::Predictor)),
            (entry(2, 5), 8, stated(5, 8, OverlapSource::Predictor)),
            (entry(5, 2), 8, stated(5, 8, OverlapSource::Index)),
            // A tie reads as the index's.
            (entry(4, 4), 8, stated(4, 8, OverlapSource::Index)),
            // A miss: zero of the request's blocks, which is not `NONE`.
            (entry(0, 0), 8, stated(0, 8, OverlapSource::None)),
            // 65,536 blocks are one bit past sixteen: one halving.
            (
                entry(40_000, 0),
                65_536,
                stated(20_000, 32_768, OverlapSource::Index),
            ),
            // 262,143 blocks are two bits past: two halvings, rounded down.
            (
                entry(0, 262_143),
                262_143,
                stated(65_535, 65_535, OverlapSource::Predictor),
            ),
        ];
        for (held, request_blocks, said) in cases {
            assert_eq!(
                Overlap::held(held, request_blocks),
                said,
                "{held:?} of {request_blocks}"
            );
        }
        assert_ne!(Overlap::held(entry(0, 0), 8), Overlap::NONE);
    }

    /// The word of who placed a pick costs the pick no byte: it sits in the
    /// one a list's count and overlap left spare between them, with the
    /// rule's own word of a pick it placed. A pick no rule placed reads the
    /// stages, and is not kept from them. Both words are the decision's: a
    /// list led anew, turned into a pair's or emptied keeps them.
    #[test]
    fn the_word_of_who_placed_a_pick_costs_the_pick_no_byte() {
        assert_eq!(size_of::<Placing>(), 1);
        assert_eq!(size_of::<Targets>(), MAX_TARGETS * size_of::<Target>() + 8);
        assert_eq!(size_of::<Pick>(), 256);
        assert_eq!(Pick::default().placed_by(), PlacedBy::Stages);
        let target = |position: u32| Target {
            position,
            id: WorkerId(u64::from(position) + 1),
            dp_rank: 0,
            pair: None,
        };
        let mut pick = Pick::default();
        pick.targets.push(target(0));
        pick.targets.push(target(1));
        let by_stages = pick;
        assert!(!by_stages.untold());
        for kept in [RulePlaced::Told, RulePlaced::Untold] {
            let untold = kept == RulePlaced::Untold;
            let mut pick = by_stages;
            pick.targets.by_rule(kept);
            assert_eq!((pick.placed_by(), pick.untold()), (PlacedBy::Rule, untold));
            assert_ne!(pick, by_stages);
            pick.targets.lead(target(1));
            assert_eq!((pick.placed_by(), pick.untold()), (PlacedBy::Rule, untold));
            pick.targets.pair_under(&target(2), SessionVerdict::None);
            assert_eq!((pick.placed_by(), pick.untold()), (PlacedBy::Rule, untold));
            pick.targets.clear();
            assert_eq!((pick.placed_by(), pick.untold()), (PlacedBy::Rule, untold));
        }
        // The two words of a rule's pick are two picks.
        let (mut told, mut kept) = (by_stages, by_stages);
        told.targets.by_rule(RulePlaced::Told);
        kept.targets.by_rule(RulePlaced::Untold);
        assert_ne!(told, kept);
    }

    /// The decode side's verdict costs a pair no byte: it sits in one the
    /// pair had spare. A target and a pick are the size they were, and a
    /// pair no decision spoke of reads none.
    #[test]
    fn the_decode_sides_verdict_costs_the_pair_no_byte() {
        assert_eq!(size_of::<DecodeVerdict>(), 1);
        assert_eq!((size_of::<Pair>(), size_of::<Target>()), (16, 40));
        assert_eq!(size_of::<Pick>(), 256);
        assert_eq!(Pair::default().decode_verdict, DecodeVerdict::NONE);
    }

    #[test]
    fn targets_keep_order_and_refuse_a_fifth() {
        let mut targets = Targets::new();
        for position in 0..6u32 {
            targets.push(Target {
                position,
                id: WorkerId(u64::from(position)),
                dp_rank: 0,
                pair: None,
            });
        }
        assert_eq!(targets.len(), MAX_TARGETS);
        let positions: Vec<u32> = targets.iter().map(|t| t.position).collect();
        assert_eq!(positions, vec![0, 1, 2, 3]);
        assert_eq!(targets.first().map(|t| t.position), Some(0));
        targets.clear();
        assert!(targets.is_empty());
    }

    #[test]
    fn every_outcome_carries_its_trace_and_only_a_pick_has_targets() {
        let trace = Trace {
            seed: 9,
            ..Trace::default()
        };
        let shed = Decision::Shed(Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by: "healthy",
            retry_after_secs: 1,
            wait_allowed: false,
            trace,
        });
        let queue = Decision::Queue(Queue {
            ticket: 1,
            max_wait_micros: 50_000,
            trace,
        });
        assert!(shed.pick().is_none());
        assert!(queue.pick().is_none());
        assert_eq!(shed.trace().seed, 9);
        assert_eq!(queue.trace().seed, 9);
        let pick = Decision::Pick(Pick {
            targets: Targets::new(),
            trace,
            session_verdict: SessionVerdict::None,
        });
        assert!(pick.pick().is_some());
    }

    #[test]
    fn a_list_cleared_and_filled_again_equals_a_new_one() {
        let target = |position: u32| Target {
            position,
            id: WorkerId(u64::from(position) + 1),
            dp_rank: 0,
            pair: None,
        };
        let mut used = Targets::new();
        used.push(target(0));
        used.push(target(1));
        used.clear();
        used.push(target(2));
        let mut new = Targets::new();
        new.push(target(2));
        assert_eq!(used, new);
        // A count and what a list states of its first target are parts
        // of it still.
        let mut longer = new;
        longer.push(target(1));
        assert_ne!(used, longer);
        let mut stated = new;
        stated.state(Overlap {
            blocks: 1,
            of: 2,
            source: OverlapSource::Index,
            margin_held: false,
        });
        assert_ne!(used, stated);
        // A list prints what its equality reads.
        assert_eq!(format!("{used:?}"), format!("{new:?}"));
        assert_ne!(format!("{used:?}"), format!("{longer:?}"));
    }

    #[test]
    fn lead_puts_a_target_first_and_drops_its_copy_or_the_last_one() {
        let target = |id: u64| Target {
            position: id as u32,
            id: WorkerId(id),
            dp_rank: 0,
            pair: None,
        };
        let mut targets = Targets::new();
        for id in 1..=3 {
            targets.push(target(id));
        }
        targets.lead(target(3));
        let ids = |t: &Targets| t.as_slice().iter().map(|t| t.id.0).collect::<Vec<_>>();
        assert_eq!(ids(&targets), [3, 1, 2], "a copy among them leaves");
        let mut full = Targets::new();
        for id in 1..=MAX_TARGETS as u64 {
            full.push(target(id));
        }
        full.lead(target(9));
        assert_eq!(full.len(), MAX_TARGETS);
        assert_eq!(ids(&full)[0], 9, "the newcomer leads");
        assert_eq!(
            *ids(&full).last().expect("full"),
            MAX_TARGETS as u64 - 1,
            "the last one left"
        );
    }

    #[test]
    fn lead_removes_only_the_equal_target_not_another_rank() {
        let x_rank_0 = Target {
            position: 0,
            id: WorkerId(1),
            dp_rank: 0,
            pair: None,
        };
        let w_rank_1 = Target {
            position: 1,
            id: WorkerId(2),
            dp_rank: 1,
            pair: None,
        };
        let w_rank_0 = Target {
            position: 2,
            dp_rank: 0,
            ..w_rank_1
        };
        let mut targets = Targets::new();
        for target in [x_rank_0, w_rank_1, w_rank_0] {
            targets.push(target);
        }

        targets.lead(w_rank_0);

        assert_eq!(targets.as_slice(), [w_rank_0, x_rank_0, w_rank_1]);
    }

    /// A decode side's targets turn into a pair's where they stand. Each
    /// names the prefill with the decode it held, in the order they had,
    /// and the first pair alone states the decode side's verdict. The list
    /// equals the one a new list takes push by push, and nothing is stated
    /// of its first target.
    #[test]
    fn a_decodes_targets_turn_into_a_pairs_where_they_stand() {
        let prefill = Target {
            position: 7,
            id: WorkerId(70),
            dp_rank: 3,
            pair: None,
        };
        let decode = |position: u32| Target {
            position,
            id: WorkerId(u64::from(position) + 1),
            dp_rank: position as u16 + 5,
            pair: None,
        };
        let mut turned = Targets::new();
        let mut pushed = Targets::new();
        for position in 0..2u32 {
            turned.push(decode(position));
            pushed.push(Target {
                pair: Some(Pair {
                    decode_position: position,
                    decode_id: decode(position).id,
                    decode_dp_rank: decode(position).dp_rank,
                    decode_verdict: [DecodeVerdict::CLAIMED, DecodeVerdict::NONE]
                        [position as usize],
                }),
                ..prefill
            });
        }
        turned.state(Overlap {
            blocks: 3,
            of: 4,
            source: OverlapSource::Index,
            margin_held: false,
        });
        turned.pair_under(&prefill, SessionVerdict::Claimed(decode(0).id));
        assert_eq!(turned, pushed);
        // The first pair reads its side's whole verdict back with its decode.
        let first = turned.as_slice()[0].pair.expect("a pair");
        assert_eq!(
            first.decode_verdict.on(first.decode_id),
            SessionVerdict::Claimed(decode(0).id)
        );
        let pick = Pick {
            targets: turned,
            ..Pick::default()
        };
        assert_eq!(pick.overlap(), Overlap::NONE);
        // Each verdict of the decode side is read back whole from the
        // first pair, with the decode it names.
        for verdict in [
            SessionVerdict::None,
            SessionVerdict::Pinned,
            SessionVerdict::Respill,
            SessionVerdict::Claimed(decode(0).id),
        ] {
            let mut list = Targets::new();
            list.push(decode(0));
            list.pair_under(&prefill, verdict);
            let pair = list.as_slice()[0].pair.expect("a pair");
            assert_eq!(pair.decode_verdict.on(pair.decode_id), verdict);
        }
        // An empty list stays empty.
        let mut none = Targets::new();
        none.pair_under(&prefill, SessionVerdict::None);
        assert_eq!(none, Targets::new());
    }

    /// The count and its absence are one word: a trace is the size it was
    /// when it held a bare count, and no count is never a zero.
    #[test]
    fn survivors_hold_a_count_or_none_in_one_word() {
        assert_eq!(size_of::<Survivors>(), size_of::<u32>());
        assert_eq!(Survivors::default().get(), Some(0), "the default counted");
        assert_eq!(Survivors::counted(5).get(), Some(5));
        assert_eq!(Survivors::NOT_COUNTED.get(), None);
        assert_eq!(
            Survivors::counted(u32::MAX).get(),
            Some(u32::MAX - 1),
            "a count is never the mark of none"
        );
        assert_ne!(Survivors::NOT_COUNTED, Survivors::counted(0));
        assert_eq!(format!("{:?}", Survivors::counted(5)), "Some(5)");
        assert_eq!(format!("{:?}", Survivors::NOT_COUNTED), "None");
        let counted = Trace::default();
        assert!(!counted.drew_first(), "a trace that counted did not draw");
        let drew = Trace {
            survivors: Survivors::NOT_COUNTED,
            ..counted
        };
        assert!(drew.drew_first());
    }
}
