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
/// The record gains fields: code outside this crate starts from the default,
/// or from [`Target::of`], and assigns what it knows, and a pattern of it
/// ends with `..`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
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
/// The record gains fields: code outside this crate starts from the default,
/// or from [`Pair::of`], and assigns what it knows, and a pattern of it
/// ends with `..`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Pair {
    pub decode_position: u32,
    pub decode_id: WorkerId,
    pub decode_dp_rank: u16,
    /// What the decision did with the decode side's session key, as the
    /// pair of a pick's first target states it: a claim names that pair's
    /// decode. The pick's own verdict ([`Pick::session_verdict`]) is the
    /// prefill side's. It is one byte the pair had spare.
    pub decode_verdict: DecodeVerdict,
    /// Who placed the decode side, and whether a sessions rule that placed
    /// it alone keeps it from the profile: the word of the decision that
    /// made the pick's decodes, which every pair of the pick states. The
    /// pick's own word ([`Pick::placed_by`]) is the prefill side's. It is
    /// the last byte the pair had spare.
    pub decode_placing: DecodePlacing,
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
    /// its id and its rank, with no verdict on a session key and the
    /// stages as its placer.
    #[must_use]
    pub fn of(decode: &Candidate) -> Self {
        Self {
            decode_position: decode.position,
            decode_id: decode.id,
            decode_dp_rank: decode.dp_rank,
            decode_verdict: DecodeVerdict::NONE,
            decode_placing: DecodePlacing::STAGES,
        }
    }
}

/// One side of a pair decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PairSide {
    Prefill,
    Decode,
}

impl PairSide {
    /// Both sides, the prefill first: the order a pair decides them in.
    pub const BOTH: [Self; 2] = [Self::Prefill, Self::Decode];

    /// The side's place in a table of the two ([`Self::BOTH`]).
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Prefill => 0,
            Self::Decode => 1,
        }
    }

    /// The side's label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
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
    /// the byte the count and the overlap left spare between them. A
    /// pair's list states its prefill side's here, and its decode side's
    /// in its pairs ([`Pair::decode_placing`]).
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
    /// word). Both are the decision's: a list led anew or emptied keeps
    /// them, and a list turned into a pair's hands them to its pairs
    /// ([`Self::pair_under`]).
    pub(crate) fn by_rule(&mut self, kept: RulePlaced) {
        self.placing = match kept {
            RulePlaced::Told => Placing::Rule,
            RulePlaced::Untold => Placing::RuleUntold,
        };
    }

    /// The list's word of who placed it: a single's, or the prefill side's
    /// of a pair.
    pub(crate) const fn placing(&self) -> Placing {
        self.placing
    }

    /// Takes the word of `held`, a pair's list, for the prefill side: a
    /// decode asked again decides no prefill, and its pick keeps the word
    /// the pair's first decision stated of that side. The word is on no
    /// lease, so the pick alone carries it to the side's later words.
    #[cfg(any(test, feature = "in-process"))]
    pub(crate) fn placed_as(&mut self, held: &Self) {
        self.placing = held.placing;
    }

    /// Takes the decode side's word of `held`, the pair a decode asked
    /// again is held by, where the list names that pair's decode first.
    /// The decision that put the request on that decode stands: the call
    /// found the decode again, by the claim that decision made for the key
    /// or by its own stages, and placed nothing. A list that names another
    /// decode first placed the side anew, and keeps its own word.
    #[cfg(any(test, feature = "in-process"))]
    pub(crate) fn decode_placed_as(&mut self, held: &Pair) {
        let named = |pair: Pair| (pair.decode_id, pair.decode_dp_rank);
        let first = self.first().and_then(|first| first.pair);
        if first.map(named) != Some(named(*held)) {
            return;
        }
        for target in &mut self.items[..usize::from(self.len)] {
            if let Some(pair) = &mut target.pair {
                pair.decode_placing = held.decode_placing;
            }
        }
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
    /// and no pair behind it states one. Every pair states the list's own
    /// word of who placed it, which is the decode side's: one decision
    /// made them all. The list then states `placed`, the prefill side's
    /// word, as a pair's pick states the prefill side's verdict. So each
    /// side of a pair keeps the word of its own decision: a sessions rule
    /// may place one side alone and leave the other to its stages. What
    /// the decode side's list stated of its first target goes
    /// ([`Pick::overlap`]): the engine over worker lists states what a
    /// pair's prefill holds behind this turn. The entries past the count
    /// are not rewritten, as [`Self::clear`] leaves them. No accessor goes
    /// past the count.
    pub(crate) fn pair_under(&mut self, prefill: &Target, decode: SessionVerdict, placed: Placing) {
        // Taken by the first pair: the pairs behind it read none.
        let mut verdict = DecodeVerdict::of(decode);
        let placing = DecodePlacing(self.placing as u8);
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
                    decode_placing: placing,
                }),
            };
        }
        self.placing = placed;
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
#[expect(clippy::exhaustive_enums, reason = "a caller answers every case")]
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
/// too, where the veto as it stands kept no worker. So does a pair placed
/// by its prefill profile that weighs load against affinity, where the veto
/// as it stands kept no prefill. Every
/// other pick reads `None`, the default. One mark is the decision's, since
/// the pick and its alternates are of the one lifted pool. The mark sits in
/// a byte the record had spare: the trace is no larger for it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
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
#[non_exhaustive]
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
    /// How many rows of `top` are set, which can be none where the picker
    /// compared workers: the decision the placing call keeps for a single
    /// request reads no best row of its scores where no stage of the
    /// document takes its issue and the document keeps no sample.
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

/// Where the two counts a pick states of its first target were read, and
/// so what they count ([`Overlap`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum OverlapSource {
    /// Nothing is held, or nothing is stated.
    #[default]
    None,
    /// The event index: whole blocks of the request's blocks.
    Index,
    /// What the worker was sent, by the sent-prefix predictor: whole blocks
    /// of the request's blocks.
    Predictor,
    /// The match of a tree the host lends: the match's size of the
    /// request's, in the tree's own unit and not in blocks. The unit is
    /// tokens for a request hashed by its token ids and characters for one
    /// hashed by its text, a host's marker in neither.
    Tree,
}

/// What a pick states of its first target: how much of the request the
/// worker holds, of how much, and where the two counts were read. Under
/// every source but a tree they are leading blocks of the request's
/// blocks. A host observes its cache series from them: the share held is
/// `blocks` over `of` under every source, and a miss reads zero of `of`.
/// `of` is zero only where nothing is stated ([`Overlap::NONE`]): a request
/// counts as one block at the least.
///
/// The engine over a host's worker lists (`InProcess`) states it behind
/// a single decision, from the first target's entry of the rows that
/// decision was given: the rows the engine named from the host's own data,
/// a caller's column, or the rows a caller named. No stage is asked. A
/// pair's first decision states the same of its prefill, which is its first
/// target's worker, from the prefills' column: a pick has room for one
/// statement, so nothing is stated of the decode. A decision given no rows
/// states nothing. Neither does a pair's decode decided again, an encode
/// request's pair, or a pick the binding's sessions rule keeps from the
/// profile.
/// [`Engine::decide`](crate::engine::Engine::decide) itself states nothing.
///
/// A request decided inside a tree its host lends is given the rows the
/// tree's match names. For a worker named among the match's holders its
/// pick states the match's own two numbers ([`OverlapSource::Tree`]): the
/// match's size of the request's, in the tree's unit, which are the two
/// the classic policy's series divides. No count is rounded to a block, so
/// a match under one block is stated too. A pick of any other worker states
/// a miss of the request's blocks. Past the profile's sample the holders
/// named are a draw.
///
/// Six bytes, in bytes a pick's targets had spare behind their count: a
/// pick is no larger for it ([`Pick::overlap`]).
///
/// The record gains fields: code outside this crate starts one from
/// [`Overlap::NONE`] or `Overlap::default()` and assigns the fields it sets,
/// and a pattern of it ends with `..`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Overlap {
    /// The leading blocks the worker holds: the larger of its entry's two
    /// readings, the index's and the sent-prefix predictor's. Under
    /// [`OverlapSource::Tree`] it is the size of the tree's match in the
    /// tree's unit. Halved with `of` where the request's count is past
    /// sixteen bits.
    pub blocks: u16,
    /// The request's blocks as the prefix stage counts them: its tokens
    /// over the block size, one at the least, and never under the block
    /// hashes its host sent. Under [`OverlapSource::Tree`] it is the
    /// request's size in the tree's unit. A count of more than 65,535
    /// halves this count and `blocks` together until it fits, so their
    /// share stays true to one part in 32,768.
    pub of: u16,
    /// Where `blocks` was read, which says what the two counts count: the
    /// index on a tie.
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
    /// word. Whoever states blocks ends here: a prefix stage's own word
    /// ([`Scorer::said`](crate::engine::Scorer::said)) and the engine over
    /// worker lists ([`Self::held`]), so the two name one source for one
    /// count. The counts are halved where every statement ends
    /// ([`Self::halved`]).
    pub(crate) fn of(blocks: u32, indexed: bool, request_blocks: u32, margin_held: bool) -> Self {
        let source = if blocks == 0 {
            OverlapSource::None
        } else if indexed {
            OverlapSource::Index
        } else {
            OverlapSource::Predictor
        };
        Self::halved(blocks, request_blocks, source, margin_held)
    }

    /// The match of a tree a host lends, as a pick of a worker the match
    /// named states it ([`OverlapSource::Tree`]): `matched` of the
    /// request's `of`, both in the tree's unit, with a margin's word.
    /// Halved as blocks are ([`Self::halved`]). A match of nothing is a
    /// miss. A match smaller than the shift's unit states zero under the
    /// source [`OverlapSource::Tree`]: 1 of 100,000 states 0 of 50,000.
    pub(crate) fn in_tree(matched: u32, of: u32, margin_held: bool) -> Self {
        let source = if matched == 0 {
            OverlapSource::None
        } else {
            OverlapSource::Tree
        };
        Self::halved(matched, of, source, margin_held)
    }

    /// `held` of `of` read from `source`, both counts halved together, by
    /// one shift, until the request's fits sixteen bits: the share they
    /// say stays true to one part in 32,768. Every statement of an overlap
    /// ends here, in blocks ([`Self::of`]) or in a tree's unit
    /// ([`Self::in_tree`]), so a count is halved in one place.
    fn halved(held: u32, of: u32, source: OverlapSource, margin_held: bool) -> Self {
        let shift = (u32::BITS - of.leading_zeros()).saturating_sub(u16::BITS);
        Self {
            blocks: u16::try_from(held >> shift).unwrap_or(u16::MAX),
            of: u16::try_from(of >> shift).unwrap_or(u16::MAX),
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
/// record of sent prompts takes no entry behind it and a lent tree
/// stores no prompt for it, as a gateway's sticky map keeps a request it
/// places from the gateway's policy. Under `told`, the default, the stages
/// hear both and where the prompt went is kept as behind any pick. The
/// session table hears the dispatch under either. Each side of a pair is
/// placed by a decision of its own and states its own word: the pick the
/// prefill side's, and its pairs the decode side's
/// ([`Pair::decode_placing`]).
///
/// Who placed a pick does not cross the crate's frame: no wire message
/// carries it, since the record is written where the decision is made.
///
/// It shares one byte with the rule's word (`Pick::untold`), the one a
/// pick's targets had spare between their count and what is stated of the
/// first: a pick is no larger for it ([`Pick::placed_by`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
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
pub(crate) enum Placing {
    #[default]
    Stages,
    Rule,
    RuleUntold,
}

/// What a pair keeps of its decode side's placing, in the last byte the
/// pair had spare ([`Pair::decode_placing`]): who placed that side, and
/// whether a sessions rule that placed it alone keeps it from the profile.
/// The decode side of a pair is decided on its own, so the rule may place
/// it alone beside a prefill its stages placed, and the other way round.
///
/// It is a byte with a name for each word, and no enum: every value of
/// the byte is a word's or reads as the stages', so the byte tells nothing
/// of whether a target names a pair, and a pick is laid out as it was.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodePlacing(u8);

impl DecodePlacing {
    /// The profile's stages placed the side ([`PlacedBy::Stages`]).
    pub const STAGES: Self = Self(Placing::Stages as u8);
    /// The binding's sessions rule placed the side alone
    /// ([`PlacedBy::Rule`]), and the profile is told of it.
    pub const RULE: Self = Self(Placing::Rule as u8);
    /// The rule placed the side alone and keeps it from the profile
    /// ([`RulePlaced::Untold`]).
    pub const RULE_UNTOLD: Self = Self(Placing::RuleUntold as u8);

    /// Who placed the side ([`PlacedBy`]).
    #[must_use]
    pub const fn placed_by(self) -> PlacedBy {
        match self {
            Self::RULE | Self::RULE_UNTOLD => PlacedBy::Rule,
            _ => PlacedBy::Stages,
        }
    }

    /// Whether the side is kept from the profile, as `Pick::untold` says
    /// it of a single.
    #[cfg(any(test, feature = "in-process"))]
    pub(crate) const fn untold(self) -> bool {
        self.0 == Self::RULE_UNTOLD.0
    }
}

/// A pick: ordered targets, the trace, and what a sessions rule did with
/// the request's key (`None` without one). The host copies that verdict
/// onto the lease it commits, and fills the lease's session from the
/// request ([`crate::engine::RequestFacts::session`]) for every keyed request, rule
/// or no rule: the verdict alone says whether the commit moves a pin.
///
/// The record gains fields: code outside this crate starts one from
/// `Pick::default()` and assigns the fields it sets, and a pattern of it
/// ends with `..`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
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
    /// binding's sessions rule placed it alone. A pair's pick says it of
    /// its prefill side, and its pairs of its decode side
    /// ([`DecodePlacing::placed_by`]). The pick of a decode asked again
    /// says the first decision's word of the prefill side to its guard and
    /// its host. The stages that hear that decode's issue are handed the
    /// pick ahead of that word: they read the decode side's word there,
    /// and none of the prefill side.
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
    /// it files a pick's lease. A pair's pick says it of its prefill side
    /// ([`Self::decode_untold`] of the other).
    #[cfg(any(test, feature = "in-process"))]
    pub(crate) const fn untold(&self) -> bool {
        matches!(self.targets.placing, Placing::RuleUntold)
    }

    /// [`Self::untold`] of a pair's decode side: the word the pick's pairs
    /// state ([`Pair::decode_placing`]), read off the first. A pick whose
    /// first target names no pair has one side, and that side's word.
    #[cfg(any(test, feature = "in-process"))]
    #[inline]
    pub(crate) fn decode_untold(&self) -> bool {
        match self.targets.first().and_then(|first| first.pair) {
            Some(pair) => pair.decode_placing.untold(),
            None => self.untold(),
        }
    }
}

/// A shed: the reason, the stage that emptied the pool, and the pacing hint.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Shed {
    pub reason: ShedReason,
    /// The name of the filter or picker that left nothing.
    pub emptied_by: &'static str,
    pub retry_after_secs: u16,
    /// The binding's scale-from-zero budget lets the host hold the request.
    pub wait_allowed: bool,
    /// The side of a pair this refusal is for: a host answers for that
    /// role, as the gateway's own pair selection names a role in every
    /// refusal. A decision fills it by its own list, the one its stages
    /// left with nobody. The prefill side where no prefill passed its own
    /// stages, where the prefill's own run declined, and where the
    /// prefills stand at their admission limit. The decode side where the
    /// prefills that passed had no decode to pair with, and where the
    /// decode's run refused. A decode decided again names the decode
    /// side, and the prefill side where the held prefill left its list.
    /// Two refusals name the side of the list that selection judges
    /// instead. A pair whose named worker's number reaches nobody
    /// (`target-worker`) names the side it reaches nobody on, the
    /// prefills' first. A call's shed that stands behind a route's rule
    /// names the prefill side where none of the rule's prefills can take
    /// the request, whichever list the decision said. An encode request
    /// names the side of its pair that a hard demand left nobody, as its
    /// pair alone does. `None` for a single decision, for a pair refused
    /// before either side was read, for an encode request's other
    /// refusals of its own pass (for its encoders, for the pairs they
    /// serve, for a list's places), and for a refusal that came over a
    /// link: the protocol's message has no field for it.
    pub side: Option<PairSide>,
    pub trace: Trace,
}

/// A queue ticket: the request waits for capacity in the binding's admission
/// order and is decided again when a signal frees some. Specified in v1; no
/// built-in issues one until the admission step, and a host that meets one
/// refuses the request.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Queue {
    pub ticket: u64,
    /// The longest the ticket may wait, within the request's admission
    /// budget.
    pub max_wait_micros: u32,
    pub trace: Trace,
}

/// What a decision answers.
#[derive(Clone, Copy, Debug, PartialEq)]
#[expect(clippy::exhaustive_enums, reason = "a caller answers every case")]
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

    /// This decision, its refusal naming the side of a pair that was left
    /// with nobody ([`Shed::side`]). A pick and a queue ticket pass as they
    /// are.
    #[must_use]
    pub(crate) fn of_side(mut self, side: PairSide) -> Self {
        if let Self::Shed(shed) = &mut self {
            shed.side = Some(side);
        }
        self
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
/// that takes it, which no gap of the membership says. A list's places are
/// its host's, or the places its request's demand leaves. A gap of the
/// pairs' membership the encode pass reads (`ec-compatible`,
/// `decode-vision`) is answered ahead of every list's places.
pub(crate) const PLACES: &str = "places";

/// The stage a request that names its worker is shed under where that
/// worker cannot take it: by the one call in a host's own process, and by
/// a controller's served decision. A pair's decision is shed under it
/// where the number its decode side states reaches no row
/// ([`Constraints::target_worker`](crate::engine::view::Constraints::target_worker)),
/// and declined under it where the number stands beside no places.
pub(crate) const TARGET_WORKER: &str = "target-worker";

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
        side: None,
        trace: Trace {
            seed,
            generation,
            ..Trace::default()
        },
    })
}

#[cfg(test)]
#[path = "../../tests/unit/engine/decision/tests.rs"]
mod tests;
