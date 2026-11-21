//! The typed decision (design 4.2): an ordered list of targets the host walks
//! at dispatch, a shed with its reason and the stage that emptied the pool, or
//! a queue ticket. A busy answer is not a decision: a remote decider gives it
//! on its stream and the host's fallback takes over at once, so the enum
//! never carries one.

use std::fmt;

#[cfg(feature = "controller")]
use crate::service::proto::{self, LiftedVeto};
use crate::{engine::view::WorkerId, policies::sticky_sessions::SessionVerdict};

/// Targets a decision carries at most: the pick and its alternates.
pub const MAX_TARGETS: usize = 4;

/// Candidates a trace names with their scores.
pub const TRACE_TOP: usize = 3;

/// One target: a candidate by position and id, with the decode side when the
/// decision is a pair.
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pair {
    pub decode_position: u32,
    pub decode_id: WorkerId,
    pub decode_dp_rank: u16,
}

/// The ordered targets of a pick: the best first. For a pair the list reads
/// "pairs before the prefill went out, decodes of the chosen prefill after",
/// so an alternate taken after the prefill dispatched changes the decode only.
#[derive(Clone, Copy, Default)]
pub struct Targets {
    items: [Target; MAX_TARGETS],
    len: u8,
    /// What the profile's prefix stage says of the first target
    /// ([`Pick::overlap`]), in the bytes the count left spare.
    overlap: Overlap,
}

/// Two lists are equal where their targets and what they state of the first
/// are. The entries past a count are no part of a list:
/// [`Targets::clear`] leaves them as they stood.
impl PartialEq for Targets {
    fn eq(&self, other: &Self) -> bool {
        self.overlap == other.overlap && self.as_slice() == other.as_slice()
    }
}

impl Eq for Targets {}

/// A list prints what its equality reads: its targets and what it states
/// of the first.
impl fmt::Debug for Targets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Targets")
            .field("items", &self.as_slice())
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
            overlap: Overlap::NONE,
        }
    }

    /// States what the first target holds of the request. A list led anew
    /// ([`Self::lead`]) says nothing of its new first target.
    pub(crate) fn state(&mut self, overlap: Overlap) {
        self.overlap = overlap;
    }

    /// Append a target; a list already full keeps what it has.
    pub fn push(&mut self, target: Target) {
        if usize::from(self.len) < MAX_TARGETS {
            self.items[usize::from(self.len)] = target;
            self.len += 1;
        }
    }

    /// Turns a decode side's targets into a pair's where they stand: each
    /// names `prefill` with the decode it held, in the order they had.
    /// Nothing is stated of a pair's first target ([`Pick::overlap`]). The
    /// entries past the count are not rewritten, as [`Self::clear`] leaves
    /// them. No accessor goes past the count.
    pub(crate) fn pair_under(&mut self, prefill: &Target) {
        for target in &mut self.items[..usize::from(self.len)] {
            *target = Target {
                position: prefill.position,
                id: prefill.id,
                dp_rank: prefill.dp_rank,
                pair: Some(Pair {
                    decode_position: target.position,
                    decode_id: target.id,
                    decode_dp_rank: target.dp_rank,
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
pub enum Lifted {
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
impl From<Lifted> for LiftedVeto {
    fn from(mark: Lifted) -> Self {
        match mark {
            Lifted::None => Self::None,
            Lifted::Overload => Self::Overload,
            Lifted::Liveness => Self::Liveness,
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
    /// The soft veto lifted for this pick ([`Lifted`]): `None` unless that
    /// veto as it stands left the pool nobody, or left a key handed on at
    /// its sessions rule's in-flight cap nobody.
    pub lifted: Lifted,
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
            lifted: Lifted::default(),
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
    /// is of the rows drawn.
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
            lifted: LiftedVeto::from(self.lifted) as i32,
            ..proto::Trace::default()
        }
    }
}

/// Where the blocks a pick's first target holds were read ([`Overlap`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum OverlapSource {
    /// Nothing is held, or no stage said.
    #[default]
    None,
    /// The event index.
    Index,
    /// The sent-prefix predictor.
    Predictor,
}

/// What a profile's prefix stage says of a pick's first target: how many
/// leading blocks of the request the worker holds, of how many, and where
/// they were read. A host observes its cache series from it: the share
/// held is `blocks` over `of`, and a miss reads zero of `of`. `of` is zero
/// only where no stage said: the profile holds no such stage, or the
/// decision was made at an entry that asks none
/// ([`Engine::decide_stated`](crate::engine::Engine::decide_stated) asks). A
/// stage counts a request as one block at the least.
///
/// Six bytes, in bytes a pick's targets had spare behind their count: a
/// pick is no larger for it ([`Pick::overlap`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Overlap {
    /// The leading blocks the worker holds, by the stage's source: the
    /// index's, the predictor's past the stage's threshold, the larger of
    /// the two under `merged`. Halved with `of` where the request's count
    /// is past sixteen bits.
    pub blocks: u16,
    /// The request's blocks as the stage counts them. A request of more
    /// than 65,535 halves this count and `blocks` together until it fits,
    /// so their share stays true.
    pub of: u16,
    /// Where `blocks` was read: the index on a tie under `merged`.
    pub source: OverlapSource,
    /// The held prefix's price kept the worker in front: a deepest holder
    /// that stood over the swing's margin and under the priced one, at the
    /// mean its pass judged by. `false` under a document that prices no
    /// margin, and for a decode phase.
    pub margin_held: bool,
}

impl Overlap {
    /// No stage said.
    pub const NONE: Self = Self {
        blocks: 0,
        of: 0,
        source: OverlapSource::None,
        margin_held: false,
    };
}

/// A pick: ordered targets, the trace, and what a sessions rule did with
/// the request's key (`None` without one). The host copies that verdict
/// onto the lease it commits, and fills the lease's session from the
/// request ([`crate::engine::RequestFacts::session`]) for every keyed request, rule
/// or no rule: the verdict alone says whether the commit moves a pin.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pick {
    pub targets: Targets,
    pub trace: Trace,
    pub session_verdict: SessionVerdict,
}

impl Pick {
    /// What the profile's prefix stage says of the first target
    /// ([`Overlap`]); nothing where no stage said.
    pub const fn overlap(&self) -> Overlap {
        self.targets.overlap
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

#[cfg(test)]
mod tests {
    use super::*;

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
                lifted: Lifted::None,
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
        assert_eq!(size_of::<Lifted>(), 1);
        assert_eq!(size_of::<Trace>(), 72);
        assert_eq!(Trace::default().lifted, Lifted::None);
    }

    /// The overlap costs the pick no byte: it shares the eight bytes the
    /// targets' count stood alone in, a pick no stage spoke of reads none,
    /// and a list led anew or cleared says nothing of a first target.
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
    /// names the prefill with the decode it held, in the order they had.
    /// The list equals the one a new list takes push by push, and nothing
    /// is stated of its first target.
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
        turned.pair_under(&prefill);
        assert_eq!(turned, pushed);
        let pick = Pick {
            targets: turned,
            ..Pick::default()
        };
        assert_eq!(pick.overlap(), Overlap::NONE);
        // An empty list stays empty.
        let mut none = Targets::new();
        none.pair_under(&prefill);
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
