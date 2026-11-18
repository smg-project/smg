//! The typed decision (design 4.2): an ordered list of targets the host walks
//! at commit, a shed with its reason and the stage that emptied the pool, or
//! a queue ticket. A busy answer is not a decision: a remote decider gives it
//! on its stream and the host's fallback takes over at once, so the enum
//! never carries one.

use std::fmt;

#[cfg(feature = "controller")]
use crate::proto::LiftedVeto;
use crate::{sessions::SessionVerdict, view::WorkerId};

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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Targets {
    items: [Target; MAX_TARGETS],
    len: u8,
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
        }
    }

    /// Append a target; a list already full keeps what it has.
    pub fn push(&mut self, target: Target) {
        if usize::from(self.len) < MAX_TARGETS {
            self.items[usize::from(self.len)] = target;
            self.len += 1;
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Puts `target` first and keeps the others behind it in their order; a
    /// copy of `target` among them leaves, and so does the last one when the
    /// list is full.
    pub fn lead(&mut self, target: Target) {
        let mut led = Self::new();
        led.push(target);
        for t in self.as_slice() {
            if t.id != target.id {
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
/// worker. Every other pick reads `None`, the default. One mark is the
/// decision's, since the pick and its alternates are of the one lifted
/// pool. The mark sits in a byte the record had spare: the trace is no
/// larger for it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Lifted {
    /// No veto was lifted.
    #[default]
    None,
    /// The overload veto: every usable worker was over its thresholds.
    Overload,
}

/// The contract's word for the mark: the one mapper, for every writer of a
/// trace.
#[cfg(feature = "controller")]
impl From<Lifted> for LiftedVeto {
    fn from(mark: Lifted) -> Self {
        match mark {
            Lifted::None => Self::None,
            Lifted::Overload => Self::Overload,
        }
    }
}

/// The decision's trace: what the trace header and an explain call print.
/// This is the first step's subset of the record; per-filter drops,
/// per-scorer contributions, the plan, the picker and the shadow pick join it
/// with the observability contract.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Trace {
    /// The profile that decided.
    pub profile: u16,
    /// Candidates after the filters, where the decision counted them
    /// ([`Survivors`]).
    pub survivors: Survivors,
    /// The top candidates and their combined scores, best first: of the set
    /// the picker compared, which is every candidate the filters left unless
    /// the picker narrows them. Where a profile's ranks put rows in front
    /// ([`Scorer::score_front`](crate::Scorer::score_front)) the scores are
    /// those rows' sums among themselves: a stage asked of them alone
    /// normalizes over them, so a score need not be the one the whole pass
    /// gives its row. The rows are the whole pass's, and their order too
    /// wherever its sums tell them apart: rows it rounds to one sum it
    /// draws among, and the front's scores order them.
    pub top: [(WorkerId, f32); TRACE_TOP],
    pub top_len: u8,
    pub seed: u64,
    pub generation: u64,
    /// The soft veto lifted for this pick ([`Lifted`]): `None` unless that
    /// veto as it stands left the pool nobody.
    pub lifted: Lifted,
}

impl Trace {
    /// Whether the decision drew first: its picker drew the rows it hands
    /// out and every filter of the profile was asked about each of them
    /// alone ([`Picker::draw_first`](crate::Picker::draw_first)). No pass
    /// over the pool ran, so the trace counts nothing of the pool:
    /// `survivors` holds no count, which is the mark read here, and `top`
    /// is of the rows drawn.
    pub const fn drew_first(&self) -> bool {
        self.survivors.get().is_none()
    }
}

/// A pick: ordered targets, the trace, and what a sessions rule did with
/// the request's key (`None` without one). The host copies that verdict
/// onto the lease it commits, and fills the lease's session from the
/// request ([`crate::RequestFacts::session`]) for every keyed request, rule
/// or no rule: the verdict alone says whether the commit moves a pin.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pick {
    pub targets: Targets,
    pub trace: Trace,
    pub session_verdict: SessionVerdict,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The mark of a lifted veto costs the trace no byte: it sits in one the
    /// record had spare, and a trace no rule wrote reads none.
    #[test]
    fn the_mark_of_a_lifted_veto_costs_the_trace_no_byte() {
        assert_eq!(size_of::<Lifted>(), 1);
        assert_eq!(size_of::<Trace>(), 72);
        assert_eq!(Trace::default().lifted, Lifted::None);
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
