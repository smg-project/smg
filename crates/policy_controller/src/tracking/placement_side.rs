//! A placement's sides: one placed request as the values its host keeps
//! with each send, from that send to that side's end.
//!
//! A host that sends a request keeps something with the send until the
//! response ends: a gateway keeps the worker's load guard. A kept
//! placement gives such a host one owned value a side
//! ([`Placement::sides`]): a single request's one, a pair's prefill and
//! decode. A side says its dispatch by one call ([`PlacementSide::sent`])
//! and its end at its drop. With no outcome reported, that end is said in
//! the words a dropped [`Placement`] says: a release where the side never
//! went out, an end of class cancelled where it did. Its host writes no
//! drop of its own and shares nothing itself. A side also answers the
//! label of the profile that picked it ([`PlacementSide::profile`]), so a
//! host counts each side by the value it keeps.
//!
//! **A side's outcome.** The value a host keeps with a send is often not
//! where the answer is read: a streamed response's guard rides in the
//! response's body, and another task reads the worker's answer. So a side
//! hands out handles on its end ([`PlacementSide::side_end`], a
//! [`SideEnd`]), and whoever reads the answer reports its outcome through
//! one ([`SideEnd::reported`]). The side's end is then said once the side
//! and every handle on it are gone: with the outcome last reported, or in
//! the words above where none was. A host that asks no handle changes
//! nothing: the end is said at the side's drop.
//!
//! A handle reaches a stage only under a document one of whose stages
//! takes an end ([`EventKind::Completed`]). Under any other document a
//! side keeps its placement by itself, as it does with no handle: the
//! handle it hands out is empty, allocates nothing and takes no lock, and
//! a report through it says nothing. Under a document that takes an end,
//! a side shares its placement with its handles: one allocation a side,
//! and a lock of the side's own at its send, at each report and at its
//! drop.
//!
//! **A pair's two sides** share the pair's one placement behind a lock.
//! The lock is one request's own: its two sides alone take it, each for
//! the time of one call or of its end, and no other request ever waits
//! on it. The decode side holds it through its second decision
//! ([`PlacementSide::decode_again`]), so whatever the request lends that
//! decision is asked under it: its route's rule
//! ([`PlaceRequest::admits`]), its headers ([`PlaceRequest::headers`]),
//! its ring ([`PlaceRequest::ring`]) and its lists' admission. None of
//! them may send or drop a side of the pair, or report through or drop a
//! handle on a side's end: it would wait on a lock it is asked under and
//! never return. A single request's side owns its placement and takes no
//! lock, but the one it shares the placement with its handles behind.
//!
//! **The order of a pair's words** is kept here, in whatever order its
//! host lets the two sides go. A decode that still stands chosen leaves
//! with its prefill's last word: the prefill's end
//! ([`DecisionGuard::completed`](super::DecisionGuard::completed)), or its
//! release where it never went out. The decode's send would then say
//! nothing. So a prefill side let go while the decode still stands chosen
//! holds its word back until the decode side speaks: at its dispatch, at
//! the call that takes its decode again, or at its own end. The word is
//! said then as it stood when it was held. An end is said with the outcome
//! a handle reported, or with the instant the prefill side was dropped at,
//! so the span a stage reads is the prefill's own. A prefill that never
//! went out is released then: a host that sends a pair's decode and no
//! prefill lets the prefill side go when it will.
//!
//! **A decode decided again** is asked of the pair's decode side
//! ([`PlacementSide::decode_again`]), behind its prefill's answer. Its host
//! may let the prefill side go before that call or behind it.

use std::{fmt, sync::Arc, time::Instant};

use parking_lot::Mutex;
use smg_worker::identity::basic::Worker;

use super::{
    lease::{EventKind, Outcome, ReleaseReason, Side},
    placement::Placement,
};
use crate::engine::{
    decision_engine::{DecisionRefused, DecodeRefused},
    PlaceRequest, Unplaced,
};

/// One side of a placed request, kept by its host from that side's send to
/// its end ([`Placement::sides`]).
///
/// Dropped, it says the side's end, once, where no handle on that end is
/// left ([`Self::side_end`]): the last of them says it otherwise. With no
/// outcome reported, a side that never went out is released as abandoned,
/// and a side that went out ends as class cancelled, with the span from
/// its dispatch to its drop. A side of a placement that owes no event says
/// nothing and reads no clock.
///
/// A pair's sides say two of those words otherwise (the module's words on
/// the order). A prefill side let go while the decode still stands chosen
/// says its word later, behind the decode side's: its end where it went
/// out, its release where it never did. A decode side that stands chosen
/// and never went out says nothing at its drop: nothing was booked for it.
pub struct PlacementSide {
    side: Side,
    /// The label of the profile that picked this side.
    profile: &'static str,
    kept: Kept,
}

/// How a side keeps the placement it speaks of.
#[expect(
    clippy::large_enum_variant,
    reason = "a single request's side is its placement: its host boxes the side once, and a box here would be a second one"
)]
enum Kept {
    /// By itself: no stage takes an end, so no handle reports one.
    Own(Of),
    /// With the handles on its end ([`SideEnd`]), behind the side's lock.
    Told(Arc<Mutex<Told>>),
}

/// The placement a side speaks of.
#[expect(
    clippy::large_enum_variant,
    reason = "a single request's side is its placement: its host boxes the side once, and a box here would be a second one"
)]
enum Of {
    /// A single request's, which its one side owns.
    Whole(Placement<'static>),
    /// A pair's, which its two sides share.
    Shared(Arc<Mutex<Pair>>),
}

/// A pair's one placement, as its two sides share it.
struct Pair {
    placement: Placement<'static>,
    /// The prefill side's last word, where it is held back for the decode
    /// side's: its end, or the release of a prefill that never went out.
    held_end: Option<End>,
}

/// A side's end, as it is said: the outcome a handle reported and the
/// instant of that report, or no outcome and the instant the side was
/// dropped at.
#[derive(Clone, Copy)]
struct End {
    at: Instant,
    outcome: Option<Outcome>,
}

/// A side as it shares its placement with the handles on its end. Its
/// drop, when the side and every handle are gone, says the side's end.
struct Told {
    side: Side,
    of: Of,
    /// What a handle last reported.
    reported: Option<End>,
    /// The instant the side was dropped at, where its placement still owed
    /// an event then.
    dropped: Option<Instant>,
}

/// Says `side`'s end of `placement` as `end` states it. An outcome goes
/// the way [`Placement::ended`] takes. With none the side is given up in
/// the words of a dropped placement, a single request's one side as a
/// pair's prefill ([`DecisionGuard::released_prefill`](super::DecisionGuard)).
fn say(placement: &mut Placement<'static>, side: Side, end: End) {
    if let Some(outcome) = end.outcome {
        placement.reported(side, outcome, end.at);
        return;
    }
    let Some(held) = &mut placement.held else {
        return;
    };
    match side {
        Side::Decode => held.released_decode(ReleaseReason::Abandoned, end.at),
        _ => held.released_prefill(ReleaseReason::Abandoned, end.at),
    }
}

impl Pair {
    /// Says the prefill's word where it is held back, as it stood when it
    /// was held.
    fn end_held_prefill(&mut self) {
        if let Some(end) = self.held_end.take() {
            say(&mut self.placement, Side::Prefill, end);
        }
    }

    /// `side`'s end. A prefill's is held back where saying it now would
    /// take the chosen decode with it, and one that waited is said behind
    /// the decode's. The release of a prefill that never went out takes a
    /// chosen decode as an end does, and is held back the same.
    ///
    /// An outcome says that its side went out: a send never said is said
    /// here first, at the report's instant, as a placement's own end says
    /// it ([`Placement::ended`]). So a prefill reported with no send is
    /// held back like any other while its decode stands chosen.
    fn ended(&mut self, side: Side, end: End) {
        if let (Some(_), Some(held)) = (end.outcome, &mut self.placement.held) {
            match side {
                Side::Decode => held.dispatched_decode(end.at),
                _ => held.dispatched(end.at),
            }
        }
        let lease = self.placement.held.as_ref();
        if side != Side::Decode && lease.is_some_and(|held| held.end_takes_decode()) {
            self.held_end = Some(end);
            return;
        }
        say(&mut self.placement, side, end);
        if side == Side::Decode {
            self.end_held_prefill();
        }
    }
}

impl Of {
    /// [`PlacementSide::sent`], for `side`.
    fn sent(&mut self, side: Side) {
        match self {
            Self::Whole(placement) => placement.sent(side),
            Self::Shared(pair) => {
                let mut pair = pair.lock();
                pair.placement.sent(side);
                // The decode no longer stands chosen: a prefill's end or
                // release that waited for this word is said behind it.
                if side == Side::Decode {
                    pair.end_held_prefill();
                }
            }
        }
    }

    /// [`PlacementSide::decode_again`], for `side`.
    fn decode_again(
        &mut self,
        side: Side,
        request: &PlaceRequest<'_>,
    ) -> Result<Arc<dyn Worker>, Unplaced> {
        let (Side::Decode, Self::Shared(pair)) = (side, &*self) else {
            let refused = DecisionRefused::Decode(DecodeRefused::NotChosen);
            return Err(Unplaced::Refused(refused));
        };
        let mut pair = pair.lock();
        let decode = pair.placement.decode_taken(request)?;
        // The decode no longer stands chosen: a prefill's end that waited
        // for this side's word is said behind the decode the call took.
        pair.end_held_prefill();
        Ok(decode)
    }

    /// Whether the placement still owes an event.
    fn owes(&self) -> bool {
        match self {
            Self::Whole(placement) => placement.owes_events(),
            Self::Shared(pair) => pair.lock().placement.owes_events(),
        }
    }

    /// `side`'s end, as `end` states it.
    fn ended(&mut self, side: Side, end: End) {
        match self {
            Self::Whole(placement) => say(placement, side, end),
            Self::Shared(pair) => pair.lock().ended(side, end),
        }
    }
}

impl Drop for Told {
    /// The side's end, once the side and every handle on it are gone: with
    /// the outcome last reported, or with none at the instant the side was
    /// dropped at. Nothing is said where the placement owed no event at
    /// that drop.
    fn drop(&mut self) {
        let Some(at) = self.dropped else {
            return;
        };
        let end = self.reported.unwrap_or(End { at, outcome: None });
        self.of.ended(self.side, end);
    }
}

impl Placement<'static> {
    /// The placement as the sides its host sends, each an owned value to
    /// keep with that side's send: the side of [`Placement::worker`], which
    /// is a single request's one side or a pair's prefill, and a pair's
    /// decode side. A placement is a pair's where its lease is a pair's: a
    /// pair always holds one.
    ///
    /// Its host reads the placement's workers before this call and leaves
    /// them as placed: a side names none. A debug build stops on a pair's
    /// placement whose decode worker was taken out, and on any other that
    /// a decode worker was written into.
    ///
    /// Where a stage of the engine's document takes an end, each side
    /// shares the placement with the handles on its end
    /// ([`PlacementSide::side_end`]): one allocation a side. Under any
    /// other document a side keeps it by itself.
    #[must_use]
    pub fn sides(self) -> (PlacementSide, Option<PlacementSide>) {
        let profile = self.profile;
        let lease = self.held.as_ref();
        let pair = lease.is_some_and(|held| held.target().pair.is_some());
        debug_assert_eq!(
            pair,
            self.decode.is_some(),
            "a placement names a decode worker where it is a pair's, and nowhere else"
        );
        let told = lease
            .and_then(|held| held.door())
            .is_some_and(|engine| engine.loaded().takes(EventKind::Completed));
        if !pair {
            if !told {
                // One expression, so that a build can write the side where
                // it is returned. Behind the two arms of `new` it copies
                // the placement at each step on the way.
                let whole = PlacementSide {
                    side: Side::Single,
                    profile,
                    kept: Kept::Own(Of::Whole(self)),
                };
                return (whole, None);
            }
            let whole = PlacementSide::new(Side::Single, profile, Of::Whole(self), told);
            return (whole, None);
        }
        let decode_profile = self.decode_profile.unwrap_or(profile);
        let pair = Arc::new(Mutex::new(Pair {
            placement: self,
            held_end: None,
        }));
        let shared = Of::Shared(Arc::clone(&pair));
        let prefill = PlacementSide::new(Side::Prefill, profile, shared, told);
        let decode = PlacementSide::new(Side::Decode, decode_profile, Of::Shared(pair), told);
        (prefill, Some(decode))
    }
}

impl PlacementSide {
    /// `side` of the placement `of`, shared with the handles on its end
    /// where a report through one is `told` to a stage.
    ///
    /// A single request's side kept by itself is built in
    /// [`Placement::sides`] directly, so its placement is not moved
    /// through here.
    fn new(side: Side, profile: &'static str, of: Of, told: bool) -> Self {
        let kept = if told {
            Kept::Told(Arc::new(Mutex::new(Told {
                side,
                of,
                reported: None,
                dropped: None,
            })))
        } else {
            Kept::Own(of)
        };
        Self {
            side,
            profile,
            kept,
        }
    }

    /// The label of the profile that picked this side, as its placement
    /// names it: [`Placement::profile`], and for a pair's decode side
    /// [`Placement::decode_profile`]. What a host counts the side under.
    #[must_use]
    pub fn profile(&self) -> &'static str {
        self.profile
    }

    /// A handle on this side's end, for whoever reads the side's answer
    /// ([`SideEnd::reported`]). A host asks one for each party that may
    /// learn how the side ended, or clones one. The side's end waits for
    /// every handle: it is said when the side and the last of them are
    /// gone.
    ///
    /// The handle is empty under a document none of whose stages takes an
    /// end, and for a side of a placement that holds no lease: it
    /// allocates nothing and takes no lock.
    #[must_use]
    pub fn side_end(&self) -> SideEnd {
        match &self.kept {
            Kept::Own(_) => SideEnd(None),
            Kept::Told(told) => SideEnd(Some(Arc::clone(told))),
        }
    }

    /// This side of the request went out: its dispatch, said once
    /// ([`Placement::sent`]).
    pub fn sent(&mut self) {
        match &mut self.kept {
            Kept::Own(of) => of.sent(self.side),
            Kept::Told(told) => told.lock().of.sent(self.side),
        }
    }

    /// Decides the pair's decode again behind its prefill's answer
    /// ([`Placement::decode_again`]), for a pair whose first decision
    /// booked its prefill alone: the call of the pair's decode side. It
    /// answers the decode the request now goes to. Its host sends the
    /// decode there, and says this side's [`Self::sent`] at that send.
    ///
    /// `request` states the request again, as [`Placement::decode_again`]
    /// reads it. What it lends the decision (its rule, its headers, its
    /// ring and its lists' admission) is asked under the pair's lock and
    /// must leave the pair's sides and the handles on their ends alone
    /// (the module's words on the lock).
    ///
    /// **The order.** The call is made once the prefill side's send was
    /// said. Its host may let the prefill side go before the call or
    /// behind it: a prefill's end that waits for this side's word is said
    /// behind the decode the call took, as it stood when it was held.
    /// Behind a call that took no decode nothing more is said, and that
    /// end still waits for this side's word.
    ///
    /// # Errors
    ///
    /// As [`Placement::decode_again`]: the placement stands as it stood
    /// before the call. A side that is not a pair's decode side, which is
    /// a pair's prefill side or a single request's one side, is refused
    /// with no decision made: [`Unplaced::Refused`] with
    /// [`DecisionRefused::Decode`] and [`DecodeRefused::NotChosen`].
    pub fn decode_again(
        &mut self,
        request: &PlaceRequest<'_>,
    ) -> Result<Arc<dyn Worker>, Unplaced> {
        match &mut self.kept {
            Kept::Own(of) => of.decode_again(self.side, request),
            Kept::Told(told) => told.lock().of.decode_again(self.side, request),
        }
    }
}

impl Drop for PlacementSide {
    /// The side's end, with no outcome, of a side that keeps its placement
    /// by itself. A single request's placement says its own at its drop.
    /// A side that shares its placement with handles leaves the instant of
    /// this drop, and its end is said when the last of them is gone: here,
    /// where none is left.
    fn drop(&mut self) {
        match &mut self.kept {
            Kept::Own(Of::Whole(_)) => {}
            Kept::Own(Of::Shared(pair)) => {
                let mut pair = pair.lock();
                // The common drop of a side that ended: nothing is owed on
                // either side, so no word is said and no clock is read.
                if pair.placement.owes_events() {
                    let at = Instant::now();
                    pair.ended(self.side, End { at, outcome: None });
                }
            }
            Kept::Told(told) => {
                let mut told = told.lock();
                if told.of.owes() {
                    told.dropped = Some(Instant::now());
                }
            }
        }
    }
}

impl fmt::Debug for PlacementSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlacementSide")
            .field("side", &self.side)
            .field("profile", &self.profile)
            .finish_non_exhaustive()
    }
}

/// A handle on one side's end ([`PlacementSide::side_end`]): whoever reads
/// the side's answer reports through it how the side ended.
///
/// A handle is cheap to clone and travels with its holder: to the task
/// that reads a streamed answer, and past the side itself. The side's end
/// is said once the side and every handle on it are gone. A handle dropped
/// with no report changes nothing: with none reported the end is said in
/// the words of a side with no handle, with the instant that side was
/// dropped at.
///
/// The default handle is empty, as is the one a side hands out where no
/// stage takes an end: a report through it says nothing, and it holds
/// nothing.
#[derive(Clone, Default)]
pub struct SideEnd(Option<Arc<Mutex<Told>>>);

impl SideEnd {
    /// Whether the handle is empty: a report through it reaches nobody.
    /// Its holder asks before it makes an outcome for the handle alone.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    /// The side ended with `outcome`, as this handle's holder read it:
    /// the status, the class and what it measured
    /// ([`Outcome::reported`]). The clock is read here.
    ///
    /// The outcome is said when the side and every handle on it are gone,
    /// the way [`Placement::ended`] says one. The side writes three things
    /// into it, whatever its holder wrote there: the side, the worker that
    /// served, and the span from the side's dispatch to this report. An
    /// outcome reported of a side whose send was never said says the send
    /// first, as [`Placement::ended`] does. A pair's prefill reported so is
    /// held back for its decode's word as one that was sent: the decode
    /// that stands chosen stays, and may still be sent or asked again.
    ///
    /// The last report of a side's handles stands: a later one writes the
    /// earlier over, with its own instant.
    pub fn reported(&self, outcome: Outcome) {
        let Some(told) = &self.0 else {
            return;
        };
        let at = Instant::now();
        told.lock().reported = Some(End {
            at,
            outcome: Some(outcome),
        });
    }
}

impl fmt::Debug for SideEnd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SideEnd")
            .field("empty", &self.is_empty())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, thread, time::Duration};

    use smg_worker::{
        identity::{basic::Worker, builder::BasicWorkerBuilder},
        test_support::no_health_check,
        worker::{WorkerSpec, WorkerStatus, WorkerType},
    };

    use super::*;
    use crate::{
        document::spec::{AppliedSpec, DecodePick, PolicySpec},
        engine::{
            decision::{Decision, Target, Targets},
            decision_engine::{DecisionEngine, Request, RequestMaker, WorkerLists},
            framework::{Picker, TieBand},
            list_cell::tests::PublishedList,
            policy_engine::PolicyEngine,
            records::worker_id,
            view::{Candidate, Constraints, Phase, RequestFacts, WorkerId},
            PairAdmission, PairLists, PartnerClasses,
        },
        stages::{pickers::push_drawn, registry::StageRegistry, Lifecycle},
        store::GatewayId,
        tracking::lease::{EventSet, Lease, TransportClass},
    };

    const MODEL: &str = "model-a";

    /// One word a stage heard of a lease: the word, the side its lease
    /// books, and the span an end states, in microseconds.
    type Word = (&'static str, Side, u32);

    thread_local! {
        /// The words of a lease the stage of this thread's next hosts
        /// takes: every one, but for a case run by [`without_an_end`].
        static TAKES: Cell<EventSet> = const { Cell::new(EventSet::LEASE) };
    }

    /// Every word of a lease but its end.
    const NO_END: EventSet = EventSet::NONE
        .with(EventKind::Issued)
        .with(EventKind::Dispatched)
        .with(EventKind::Released);

    /// Runs `case` with the stage of each host it makes taking no end: under
    /// such a document a side keeps its placement by itself.
    fn without_an_end(case: impl FnOnce()) {
        TAKES.set(NO_END);
        case();
        TAKES.set(EventSet::LEASE);
    }

    /// A picker that names the workers the filters keep in their list's
    /// order, takes the words of a lease it is made to take, and keeps
    /// every word it hears, with each end's outcome beside. A release under
    /// another reason than abandoned fails its hearing. An end of another
    /// class than cancelled or with a status fails the read of the words
    /// ([`Host::words`]), unless its case read the outcomes first
    /// ([`Host::ends`]).
    #[derive(Debug)]
    struct Hears {
        takes: EventSet,
        words: Arc<Mutex<Vec<Word>>>,
        ends: Arc<Mutex<Vec<Outcome>>>,
    }

    impl Lifecycle for Hears {
        fn hears(&self) -> EventSet {
            self.takes
        }

        fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
            self.words.lock().push(("issued", lease.side, 0));
        }

        fn on_dispatched(&self, lease: &Lease, _: &Target, _: Side, _: u32) {
            self.words.lock().push(("sent", lease.side, 0));
        }

        fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
            let span = outcome.duration_micros;
            self.words.lock().push(("ended", lease.side, span));
            self.ends.lock().push(*outcome);
        }

        fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
            assert_eq!(reason, ReleaseReason::Abandoned);
            self.words.lock().push(("released", lease.side, 0));
        }
    }

    impl Picker for Hears {
        fn name(&self) -> &'static str {
            "hears"
        }

        fn pick(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            _: &[f32],
            _: TieBand,
            out: &mut Targets,
        ) {
            push_drawn(candidates, kept, out);
        }
    }

    /// A host that admits every worker of a pair.
    #[derive(Debug)]
    struct Open;

    impl PairAdmission for Open {
        fn has_capacity(&self, _: usize) -> bool {
            true
        }

        fn has_room(&self, _: usize) -> bool {
            true
        }
    }

    /// A request stated ahead, with the lists its host hands for it.
    struct Asks<'a>(Request<'a>, WorkerLists<'a>);

    impl RequestMaker for Asks<'_> {
        fn model(&self) -> &str {
            self.0.facts.model
        }

        fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
            self.0
        }

        fn workers(&self) -> WorkerLists<'_> {
            self.1
        }
    }

    /// The placement of one whole decision of `engine` over `lists`, its
    /// first target taken, on `workers`: the worker the request goes to,
    /// and a pair's decode.
    fn placed(
        engine: &'static PolicyEngine,
        workers: &[Arc<dyn Worker>; 2],
        lists: WorkerLists<'_>,
        later: bool,
    ) -> Placement<'static> {
        let pair = matches!(lists, WorkerLists::Pair(_));
        let facts = RequestFacts {
            model: MODEL,
            phase: if pair { Phase::Prefill } else { Phase::Single },
            ..RequestFacts::default()
        };
        let constraints = Constraints {
            decode_later: later,
            ..Constraints::default()
        };
        let request = Request {
            id: b"",
            facts,
            constraints,
        };
        let now = Instant::now();
        let asks = Asks(request, lists);
        let [worker, decode] = workers;
        let worker = Arc::clone(worker);
        let decode = pair.then(|| Arc::clone(decode));
        let decided = engine.decide_under(GatewayId::LOCAL, &asks, now);
        // The maker says of no request that it is plain: the answer is whole.
        let (decided, said_later) = decided
            .expect("an answer")
            .whole()
            .expect("a whole decision");
        assert_eq!(said_later, later);
        Placement::taken(engine.loaded(), decided, later, now, 0, worker, decode)
            .expect("a placement")
            .kept()
    }

    /// A ready worker of `worker_type` at an address of `name`.
    fn worker(name: &str, worker_type: WorkerType) -> Arc<dyn Worker> {
        let worker = <BasicWorkerBuilder>::from_spec(WorkerSpec {
            worker_type,
            ..WorkerSpec::new(format!("grpc://{name}:9000"))
        })
        .health_config(no_health_check())
        .build();
        worker.set_status(WorkerStatus::Ready);
        Arc::new(worker)
    }

    /// A host: its engine for the life of the process, under a document
    /// whose `profile` picks through [`Hears`], the words that stage heard
    /// and the outcomes of the ends among them, and one prefill and one
    /// decode worker, each one's list published with its cell.
    struct Host {
        engine: &'static PolicyEngine,
        heard: Arc<Mutex<Vec<Word>>>,
        ends: Arc<Mutex<Vec<Outcome>>>,
        workers: [Arc<dyn Worker>; 2],
        lists: [PublishedList; 2],
    }

    impl Host {
        fn under(mut document: PolicySpec, profile: &str) -> Self {
            let heard = Arc::new(Mutex::new(Vec::new()));
            let ends = Arc::new(Mutex::new(Vec::new()));
            let takes = TAKES.get();
            let (words, outcomes) = (Arc::clone(&heard), Arc::clone(&ends));
            let mut stages = StageRegistry::builtin();
            let added = stages.picker("hears", move || {
                Box::new(Hears {
                    takes,
                    words: Arc::clone(&words),
                    ends: Arc::clone(&outcomes),
                })
            });
            added.expect("a free name");
            document.generation = 1;
            let picks = document.profiles.get_mut(profile);
            picks.expect("the preset's profile").pick = "hears".into();
            let spec = AppliedSpec::validate_in(&document, &stages, &[]);
            let engine = PolicyEngine::new(spec.expect("the stages hold the picker"));
            let engine: &'static PolicyEngine = Box::leak(Box::new(engine));
            let workers = [
                worker("prefill", WorkerType::Prefill),
                worker("decode", WorkerType::Decode),
            ];
            let lists = workers
                .each_ref()
                .map(|worker| PublishedList::new(Arc::from([Arc::clone(worker)])));
            Self {
                engine,
                heard,
                ends,
                workers,
                lists,
            }
        }

        /// A host of pairs, under the stock pair document.
        fn of_pairs() -> Self {
            let document = PolicySpec::pair_presets("round_robin", "round_robin");
            Self::under(document.expect("the pair presets"), "decode")
        }

        /// A host of pairs whose document picks a pair's decode after its
        /// prefill: a request that asks its decode again is placed with
        /// its decode standing chosen.
        fn of_later_pairs() -> Self {
            let document = PolicySpec::pair_presets("round_robin", "round_robin");
            let mut document = document.expect("the pair presets");
            document.plans.pd_pair.decode = DecodePick::AfterPrefill;
            Self::under(document, "decode")
        }

        /// A host of single requests whose document's stage takes every
        /// word of a lease, so each request is leased.
        fn of_singles() -> Self {
            let document = PolicySpec::preset("round_robin");
            Self::under(document.expect("a built-in preset"), "round_robin")
        }

        /// The host's two lists as it hands them for a pair, under
        /// `classes`: every worker is admitted, and no side is held to
        /// places.
        fn lists<'a>(&'a self, classes: &'a PartnerClasses) -> WorkerLists<'a> {
            let [prefill, decode] = &self.lists;
            WorkerLists::Pair(PairLists {
                prefill: prefill.lent(),
                decode: decode.lent(),
                classes,
                admission: &Open,
                prefill_places: None,
                decode_places: None,
                prefill_stalled: &[],
                prefill_held: &[],
                decode_stalled: &[],
                decode_held: &[],
            })
        }

        /// A pair placed on the host's two workers: its decode booked with
        /// its prefill, or standing chosen where `later`. Its two sides.
        fn pair(&self, later: bool) -> (PlacementSide, PlacementSide) {
            let classes = PartnerClasses::number(1, [&[0_u32][..]]);
            let lists = self.lists(&classes);
            let placement = placed(self.engine, &self.workers, lists, later);
            assert!(placement.owes_events());
            let (prefill, decode) = placement.sides();
            (prefill, decode.expect("a pair's decode side"))
        }

        /// A single request placed on the host's first worker, leased.
        fn single(&self) -> PlacementSide {
            let lists = WorkerLists::Single(self.lists[0].lent());
            let placement = placed(self.engine, &self.workers, lists, false);
            assert!(placement.owes_events());
            let (side, none) = placement.sides();
            assert!(none.is_none(), "a single request has one side");
            side
        }

        /// The outcomes of the ends the stage heard since the last read, in
        /// their order: a case that reports one reads them ahead of the
        /// words.
        fn ends(&self) -> Vec<Outcome> {
            std::mem::take(&mut *self.ends.lock())
        }

        /// Every end heard since the last read said no outcome: class
        /// cancelled, with no status.
        fn no_outcome(&self) {
            for end in self.ends() {
                let plain = (TransportClass::Cancelled, 0);
                assert_eq!((end.class, end.status), plain);
            }
        }

        /// What the stage heard since the last call: each word and its
        /// side. No word went unheard, no decision faulted, and no end
        /// said an outcome its case did not read ([`Self::ends`]).
        fn words(&self) -> Vec<(&'static str, Side)> {
            assert_eq!((self.engine.faults(), self.engine.unheard()), (0, 0));
            self.no_outcome();
            let heard = std::mem::take(&mut *self.heard.lock());
            heard
                .into_iter()
                .map(|(word, side, _)| (word, side))
                .collect()
        }

        /// The span the prefill's end states, in microseconds: the last
        /// word the stage heard since the last call, behind `ahead`.
        fn prefills_end(&self, ahead: &[(&'static str, Side)]) -> u128 {
            self.no_outcome();
            let heard = std::mem::take(&mut *self.heard.lock());
            let [before @ .., ("ended", P, span)] = heard.as_slice() else {
                panic!("the prefill's end is heard last: {heard:?}");
            };
            let before: Vec<_> = before.iter().map(|&(word, side, _)| (word, side)).collect();
            assert_eq!(before, ahead);
            u128::from(*span)
        }
    }

    /// Whether the placement `side` speaks of still owes an event.
    fn owes(side: &PlacementSide) -> bool {
        match &side.kept {
            Kept::Own(of) => of.owes(),
            Kept::Told(told) => told.lock().of.owes(),
        }
    }

    const P: Side = Side::Prefill;
    const D: Side = Side::Decode;

    /// Each side of a pair says its own dispatch, and its own end at its
    /// drop, in whichever order its host lets the two go: an end of class
    /// cancelled with no status, under its own side's lease.
    #[test]
    fn each_side_of_a_pair_says_its_dispatch_and_its_end_at_its_drop() {
        let host = Host::of_pairs();
        let issued = [("issued", P), ("issued", D)];
        let (mut prefill, mut decode) = host.pair(false);
        assert_eq!(host.words(), issued);
        prefill.sent();
        decode.sent();
        // A second word of a side says nothing.
        prefill.sent();
        assert_eq!(host.words(), [("sent", P), ("sent", D)]);
        drop(prefill);
        assert_eq!(host.words(), [("ended", P)]);
        drop(decode);
        assert_eq!(host.words(), [("ended", D)]);

        // The decode side let go first.
        let (mut prefill, mut decode) = host.pair(false);
        decode.sent();
        prefill.sent();
        drop(decode);
        drop(prefill);
        let told = [("sent", D), ("sent", P), ("ended", D), ("ended", P)];
        assert_eq!(host.words(), [&issued[..], &told[..]].concat());
    }

    /// A side that never went out is released by name at its drop, and the
    /// other side lives on: its dispatch and its end are still its own.
    #[test]
    fn a_side_dropped_unsent_is_released_by_name() {
        let host = Host::of_pairs();
        let (mut prefill, decode) = host.pair(false);
        host.words();
        drop(decode);
        assert_eq!(host.words(), [("released", D)]);
        prefill.sent();
        drop(prefill);
        assert_eq!(host.words(), [("sent", P), ("ended", P)]);

        let (prefill, mut decode) = host.pair(false);
        host.words();
        drop(prefill);
        assert_eq!(host.words(), [("released", P)]);
        decode.sent();
        drop(decode);
        assert_eq!(host.words(), [("sent", D), ("ended", D)]);

        // Neither side went out.
        let (prefill, decode) = host.pair(false);
        host.words();
        drop(prefill);
        drop(decode);
        assert_eq!(host.words(), [("released", P), ("released", D)]);
    }

    /// A prefill side dropped behind its send, while the decode still
    /// stands chosen, holds its end back for the decode side's word. A
    /// stage hears what it hears of a host that sends the decode first:
    /// the decode's issue and its dispatch, then the prefill's end, then
    /// the decode's. The end names the span to the prefill side's drop,
    /// not to the decode's later word.
    #[test]
    fn a_prefill_side_dropped_first_holds_its_end_for_the_decodes_word() {
        let host = Host::of_pairs();
        let in_order = {
            let (mut prefill, mut decode) = host.pair(true);
            prefill.sent();
            decode.sent();
            drop(prefill);
            drop(decode);
            host.words()
        };
        let told = [("issued", P), ("sent", P), ("issued", D), ("sent", D)];
        let ended = [("ended", P), ("ended", D)];
        assert_eq!(in_order, [&told[..], &ended[..]].concat());

        let (mut prefill, mut decode) = host.pair(true);
        let start = Instant::now();
        prefill.sent();
        drop(prefill);
        let dropped = Instant::now();
        assert_eq!(host.words(), told[..2], "the end waits");
        thread::sleep(Duration::from_millis(20));
        decode.sent();
        let span = host.prefills_end(&told[2..]);
        let longest = dropped.duration_since(start).as_micros();
        assert!(span <= longest, "{span} over {longest}");
        drop(decode);
        assert_eq!(host.words(), [("ended", D)]);
    }

    /// A decode side dropped with its decode never sent ends the pair
    /// behind a prefill's held end: a chosen decode was booked nowhere and
    /// says nothing, and the prefill's end is said then, with the span to
    /// the prefill side's drop. Let go first, it leaves the prefill's end
    /// to its own drop. A prefill that never went out holds its release
    /// the same way: the chosen decode stays, and the release is said at
    /// the decode side's drop, one issue and one release in all.
    #[test]
    fn a_decode_side_dropped_unsent_ends_the_pair_behind_a_held_prefill() {
        let host = Host::of_pairs();
        let (mut prefill, decode) = host.pair(true);
        let start = Instant::now();
        prefill.sent();
        drop(prefill);
        let dropped = Instant::now();
        assert_eq!(host.words(), [("issued", P), ("sent", P)]);
        thread::sleep(Duration::from_millis(20));
        drop(decode);
        let span = host.prefills_end(&[]);
        let longest = dropped.duration_since(start).as_micros();
        assert!(span <= longest, "{span} over {longest}");

        let (mut prefill, decode) = host.pair(true);
        prefill.sent();
        drop(decode);
        assert_eq!(host.words(), [("issued", P), ("sent", P)]);
        drop(prefill);
        assert_eq!(host.words(), [("ended", P)]);

        let (prefill, decode) = host.pair(true);
        drop(prefill);
        assert_eq!(host.words(), [("issued", P)], "the release waits");
        assert!(owes(&decode), "the chosen decode stays with its side");
        drop(decode);
        assert_eq!(host.words(), [("released", P)]);
    }

    /// A prefill side dropped with its prefill never sent, while the decode
    /// still stands chosen, holds its release back for the decode side's
    /// word, as one that went out holds its end: the release would take
    /// the chosen decode with it. So a host that sends a pair's decode and
    /// no prefill is heard whole, whichever side it lets go first: the
    /// decode's issue and its dispatch, then the prefill's release, then
    /// the decode's end.
    #[test]
    fn a_prefill_side_dropped_unsent_holds_its_release_for_the_decodes_word() {
        let host = Host::of_pairs();
        let (prefill, mut decode) = host.pair(true);
        drop(prefill);
        assert_eq!(host.words(), [("issued", P)], "the release waits");
        decode.sent();
        let told = [("issued", D), ("sent", D), ("released", P)];
        assert_eq!(host.words(), told);
        drop(decode);
        assert_eq!(host.words(), [("ended", D)]);
    }

    /// A pair's request that asks its decode again, over `lists`.
    fn later(lists: WorkerLists<'_>) -> PlaceRequest<'_> {
        PlaceRequest {
            decode_again: true,
            ..PlaceRequest::of(MODEL, lists)
        }
    }

    /// The two sides of the pair `host`'s engine places for `request`.
    fn sides_of(host: &Host, request: &PlaceRequest<'_>) -> (PlacementSide, PlacementSide) {
        let placement = host.engine.place(request).expect("a placement");
        let (prefill, decode) = placement.kept().sides();
        (prefill, decode.expect("a pair's decode side"))
    }

    /// A pair's decode side asks its decode again once the prefill went
    /// out, and is answered the decode the request now goes to. A stage
    /// hears the decode's issue at that call, and the decode's dispatch at
    /// its send. A prefill side kept past the call says its end at its own
    /// drop. One let go ahead of the call holds its end back for the
    /// decode the call takes: the end is said behind that decode's issue,
    /// with the span to the prefill side's drop.
    #[test]
    fn a_decode_side_asks_its_decode_again_behind_its_prefills_send() {
        let host = Host::of_later_pairs();
        let classes = PartnerClasses::number(1, [&[0_u32][..]]);
        let request = later(host.lists(&classes));
        let (mut prefill, mut decode) = sides_of(&host, &request);
        prefill.sent();
        assert_eq!(host.words(), [("issued", P), ("sent", P)]);
        let taken = decode.decode_again(&request).expect("a decode");
        assert_eq!(taken.url(), host.workers[1].url());
        assert_eq!(host.words(), [("issued", D)]);
        drop(prefill);
        assert_eq!(host.words(), [("ended", P)]);
        decode.sent();
        drop(decode);
        assert_eq!(host.words(), [("sent", D), ("ended", D)]);

        // The prefill side let go ahead of the call.
        let (mut prefill, mut decode) = sides_of(&host, &request);
        let start = Instant::now();
        prefill.sent();
        drop(prefill);
        let dropped = Instant::now();
        assert_eq!(host.words(), [("issued", P), ("sent", P)], "the end waits");
        thread::sleep(Duration::from_millis(20));
        decode.decode_again(&request).expect("a decode");
        let span = host.prefills_end(&[("issued", D)]);
        let longest = dropped.duration_since(start).as_micros();
        assert!(span <= longest, "{span} over {longest}");
        decode.sent();
        drop(decode);
        assert_eq!(host.words(), [("sent", D), ("ended", D)]);
    }

    /// Behind a call that took no decode nothing is said and the placement
    /// stands: a prefill's end that is held back still waits, and is said
    /// behind the dispatch of the first decision's decode.
    #[test]
    fn a_second_ask_that_takes_no_decode_leaves_a_held_end_waiting() {
        let host = Host::of_later_pairs();
        let classes = PartnerClasses::number(1, [&[0_u32][..]]);
        let request = later(host.lists(&classes));
        let (mut prefill, mut decode) = sides_of(&host, &request);
        prefill.sent();
        drop(prefill);
        assert_eq!(host.words(), [("issued", P), ("sent", P)]);

        // At this call the host's classes pair the prefill with no decode:
        // the decode's own decision sheds, and issues nothing.
        let alone = PartnerClasses::number(1, [&[0_u32; 0][..]]);
        let answer = decode.decode_again(&later(host.lists(&alone))).err();
        assert!(matches!(answer, Some(Unplaced::Shed { .. })), "{answer:?}");
        assert!(host.words().is_empty(), "the held end still waits");
        decode.sent();
        assert_eq!(host.words(), [("issued", D), ("sent", D), ("ended", P)]);
        drop(decode);
        assert_eq!(host.words(), [("ended", D)]);
    }

    /// A decode asked again behind a prefill that never went out is refused
    /// by name, with nothing said: the ask comes once the prefill's send was
    /// said. The release that prefill side holds back still waits, and is
    /// said behind the dispatch of the first decision's decode.
    #[test]
    fn a_refused_second_ask_leaves_an_unsent_prefills_release_waiting() {
        let host = Host::of_later_pairs();
        let classes = PartnerClasses::number(1, [&[0_u32][..]]);
        let request = later(host.lists(&classes));
        let (prefill, mut decode) = sides_of(&host, &request);
        drop(prefill);
        assert_eq!(host.words(), [("issued", P)], "the release waits");

        let refused = DecisionRefused::Decode(DecodeRefused::PrefillNotHeld);
        let answer = decode.decode_again(&request).err();
        assert_eq!(answer, Some(Unplaced::Refused(refused)));
        assert!(host.words().is_empty(), "the held release still waits");
        decode.sent();
        let told = [("issued", D), ("sent", D), ("released", P)];
        assert_eq!(host.words(), told);
        drop(decode);
        assert_eq!(host.words(), [("ended", D)]);
    }

    /// The side answers the decode its second decision took, and not the
    /// one that stood chosen: by that call the host's decode list holds
    /// another worker alone.
    #[test]
    fn a_decode_side_answers_the_decode_its_second_decision_took() {
        let host = Host::of_later_pairs();
        let classes = PartnerClasses::number(1, [&[0_u32][..]]);
        let request = later(host.lists(&classes));
        let (mut prefill, mut decode) = sides_of(&host, &request);
        prefill.sent();
        assert_eq!(host.words(), [("issued", P), ("sent", P)]);

        let other = worker("decode-b", WorkerType::Decode);
        let moved = PublishedList::new(Arc::from([Arc::clone(&other)]));
        let WorkerLists::Pair(lists) = host.lists(&classes) else {
            panic!("a pair's lists");
        };
        let lists = PairLists {
            decode: moved.lent(),
            ..lists
        };
        let taken = decode.decode_again(&later(WorkerLists::Pair(lists)));
        assert_eq!(taken.expect("a decode").url(), other.url());
        assert_eq!(host.words(), [("issued", D)]);
    }

    /// A side that is not a pair's decode side is refused its second ask
    /// by one name, with no decision made and nothing said: a pair's
    /// prefill side, and a single request's one side.
    #[test]
    fn a_side_that_is_no_pairs_decode_side_is_refused_a_second_ask_by_name() {
        let refused = Unplaced::Refused(DecisionRefused::Decode(DecodeRefused::NotChosen));
        let host = Host::of_later_pairs();
        let classes = PartnerClasses::number(1, [&[0_u32][..]]);
        let request = later(host.lists(&classes));
        let (mut prefill, decode) = sides_of(&host, &request);
        prefill.sent();
        host.words();
        assert_eq!(prefill.decode_again(&request).err(), Some(refused));
        assert!(host.words().is_empty());
        drop((prefill, decode));
        host.words();

        let singles = Host::of_singles();
        let mut single = singles.single();
        single.sent();
        singles.words();
        assert_eq!(single.decode_again(&request).err(), Some(refused));
        assert!(singles.words().is_empty());
    }

    /// A single request's one side owns its placement: its dispatch, and
    /// at its drop its end, or its release where it never went out. A
    /// placement that owes no event gives a side that says nothing.
    #[test]
    fn a_single_requests_one_side_says_its_placements_words() {
        let host = Host::of_singles();
        let mut side = host.single();
        side.sent();
        drop(side);
        let single = Side::Single;
        let told = [("issued", single), ("sent", single), ("ended", single)];
        assert_eq!(host.words(), told);
        drop(host.single());
        assert_eq!(host.words(), [("issued", single), ("released", single)]);

        // Under a document none of whose stages takes a word of a request.
        let plain = PolicySpec::preset("round_robin").expect("a built-in preset");
        let spec = AppliedSpec::validate(&plain).expect("the document validates");
        let engine: &'static PolicyEngine = Box::leak(Box::new(PolicyEngine::new(spec)));
        let list: Arc<[Arc<dyn Worker>]> = Arc::from([Arc::clone(&host.workers[0])]);
        let members = PublishedList::new(list);
        let placement = placed(
            engine,
            &host.workers,
            WorkerLists::Single(members.lent()),
            false,
        );
        assert!(!placement.owes_events());
        let (mut side, none) = placement.sides();
        assert!(none.is_none());
        side.sent();
        drop(side);
        assert!(host.words().is_empty());
    }

    /// A side answers the label of the profile that picked it, as the call
    /// that placed the request wrote it: the pair's first profile for its
    /// prefill, the binding's decode profile for its decode. Where no call
    /// wrote a decode's label, both sides answer the first. A single
    /// request's side answers its placement's one label.
    #[test]
    fn a_side_answers_the_label_of_the_profile_that_picked_it() {
        let host = Host::of_pairs();
        let classes = PartnerClasses::number(1, [&[0_u32][..]]);
        let request = PlaceRequest::of(MODEL, host.lists(&classes));
        let placement = host.engine.place(&request).expect("a placement");
        let labels = (placement.profile, placement.decode_profile);
        assert_eq!(labels, ("prefill", Some("decode")));
        let (prefill, decode) = placement.kept().sides();
        let decode = decode.expect("a pair's decode side");
        assert_eq!((prefill.profile(), decode.profile()), ("prefill", "decode"));

        let (prefill, decode) = host.pair(false);
        assert_eq!(
            (prefill.profile(), decode.profile()),
            ("prefill", "prefill")
        );

        let host = Host::of_singles();
        assert_eq!(host.single().profile(), "round_robin");
    }

    /// A pair's placement whose host took the decode worker out stops a
    /// debug build where its sides are asked: the lease is a pair's, and
    /// the placement names no decode.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "names a decode worker where it is a pair's, and nowhere else")]
    fn a_pairs_placement_with_its_decode_taken_out_stops_a_debug_build() {
        let host = Host::of_pairs();
        let classes = PartnerClasses::number(1, [&[0_u32][..]]);
        let lists = host.lists(&classes);
        let mut placement = placed(host.engine, &host.workers, lists, false);
        drop(placement.decode.take());
        drop(placement.sides());
    }

    /// A single request's placement that its host wrote a decode worker
    /// into stops a debug build under the same words: its lease is no
    /// pair's.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "names a decode worker where it is a pair's, and nowhere else")]
    fn a_single_requests_placement_with_a_decode_written_in_stops_a_debug_build() {
        let host = Host::of_singles();
        let lists = WorkerLists::Single(host.lists[0].lent());
        let mut placement = placed(host.engine, &host.workers, lists, false);
        placement.decode = Some(Arc::clone(&host.workers[1]));
        drop(placement.sides());
    }

    /// A placement that holds no lease is no pair's, whatever its host
    /// wrote into it: one with a decode worker stops a debug build under
    /// the same words. The engine makes no such placement: a pair always
    /// holds a lease.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "names a decode worker where it is a pair's, and nowhere else")]
    fn a_placement_that_holds_no_lease_with_a_decode_written_in_stops_a_debug_build() {
        let host = Host::of_pairs();
        let [worker, decode] = &host.workers;
        let mut placement = Placement::alone(Arc::clone(worker), "prefill");
        placement.decode = Some(Arc::clone(decode));
        drop(placement.sides());
    }

    /// A side travels with its send: to the task that reads the response,
    /// and behind a pointer several threads share. A pair's two sides say
    /// their words from two threads at once, each side's in its order,
    /// and none is lost or said twice.
    #[test]
    fn a_pairs_two_sides_speak_from_two_threads() {
        fn travels<T: Send + Sync + 'static>() {}
        travels::<PlacementSide>();

        let host = Host::of_pairs();
        for round in 0..64 {
            let (mut prefill, mut decode) = host.pair(round % 2 == 0);
            let first = thread::spawn(move || prefill.sent());
            let second = thread::spawn(move || decode.sent());
            first.join().expect("the prefill's thread");
            second.join().expect("the decode's thread");
            let words = host.words();
            let of = |side| words.iter().filter(move |word| word.1 == side);
            for side in [P, D] {
                let told: Vec<_> = of(side).map(|word| word.0).collect();
                assert_eq!(told, ["issued", "sent", "ended"], "{side:?}: {words:?}");
            }
        }
    }

    /// A pair's sides say the same words in the same order where no stage
    /// of the document takes an end, and each side keeps the pair's
    /// placement by itself: every case of a pair's words above, run again
    /// under such a document. A pair's every word is told, so its stage
    /// still hears the ends it did not name.
    #[test]
    fn a_pairs_words_stand_where_no_stage_takes_an_end() {
        let cases: [fn(); 9] = [
            each_side_of_a_pair_says_its_dispatch_and_its_end_at_its_drop,
            a_side_dropped_unsent_is_released_by_name,
            a_prefill_side_dropped_first_holds_its_end_for_the_decodes_word,
            a_decode_side_dropped_unsent_ends_the_pair_behind_a_held_prefill,
            a_prefill_side_dropped_unsent_holds_its_release_for_the_decodes_word,
            a_decode_side_asks_its_decode_again_behind_its_prefills_send,
            a_second_ask_that_takes_no_decode_leaves_a_held_end_waiting,
            a_refused_second_ask_leaves_an_unsent_prefills_release_waiting,
            a_pairs_two_sides_speak_from_two_threads,
        ];
        for case in cases {
            without_an_end(case);
        }
        without_an_end(|| {
            let (prefill, decode) = Host::of_pairs().pair(false);
            let own = |side: &PlacementSide| matches!(side.kept, Kept::Own(_));
            assert!(own(&prefill) && own(&decode));
        });
        let (prefill, decode) = Host::of_pairs().pair(false);
        let told = |side: &PlacementSide| matches!(side.kept, Kept::Told(_));
        assert!(told(&prefill) && told(&decode), "the cases above share it");
    }

    /// An outcome as the reader of an answer states it: the status, the
    /// class and what it measured. It names a side, a served worker and a
    /// span that are not its side's: the side writes all three over.
    fn read(status: u16, class: TransportClass) -> Outcome {
        Outcome {
            side: Side::Encode,
            served: WorkerId(9),
            ttft_micros: 11,
            tokens_in: 12,
            tokens_out: 13,
            duration_micros: 7,
            ..Outcome::reported(status, class)
        }
    }

    /// The one end the stage of `host` heard since the last read.
    fn one_end(host: &Host) -> Outcome {
        let ends = host.ends();
        let [end] = ends[..] else {
            panic!("one end is heard: {ends:?}");
        };
        end
    }

    /// A side ends with the outcome a handle on its end reported, said
    /// once, when the side and the handle are both gone: the status, the
    /// class and the measures as reported, under the side's own name and
    /// worker, with the span from its dispatch to the report. The reader
    /// of a streamed answer reports while the side is kept, and the reader
    /// of an answer read whole behind the side's drop.
    #[test]
    fn a_side_ends_with_the_outcome_its_handle_reported() {
        let host = Host::of_singles();
        let single = Side::Single;
        let served = worker_id(host.workers[0].as_ref());
        let sent = [("issued", single), ("sent", single)];

        let mut side = host.single();
        let end = side.side_end();
        assert!(!end.is_empty());
        let start = Instant::now();
        side.sent();
        thread::sleep(Duration::from_millis(20));
        end.reported(read(200, TransportClass::Ok));
        let reported = Instant::now();
        thread::sleep(Duration::from_millis(20));
        drop(side);
        assert_eq!(host.words(), sent, "the end waits for the handle");
        drop(end);
        let heard = one_end(&host);
        let span = heard.duration_micros;
        let longest = reported.duration_since(start).as_micros();
        assert!(span >= 20_000, "{span}: the span from the dispatch");
        assert!(u128::from(span) <= longest, "{span} over {longest}");
        let stated = Outcome {
            side: single,
            served,
            duration_micros: span,
            ..read(200, TransportClass::Ok)
        };
        assert_eq!(heard, stated);
        assert_eq!(host.words(), [("ended", single)]);

        // Reported behind the side's drop.
        let mut side = host.single();
        side.sent();
        let end = side.side_end();
        drop(side);
        end.reported(read(503, TransportClass::ServerError));
        assert_eq!(host.words(), sent, "the end waits for the handle");
        drop(end);
        let heard = one_end(&host);
        assert_eq!(
            (heard.status, heard.class),
            (503, TransportClass::ServerError)
        );
        assert_eq!(host.words(), [("ended", single)]);

        // The handle let go first: the end is said at the side's drop.
        let mut side = host.single();
        side.sent();
        let end = side.side_end();
        end.reported(read(404, TransportClass::ClientError));
        drop(end);
        assert_eq!(host.words(), sent, "the end waits for the side");
        drop(side);
        assert_eq!(one_end(&host).status, 404);
        assert_eq!(host.words(), [("ended", single)]);
    }

    /// The last report of a side's handles stands, and the end is said
    /// once, behind the last of them. A clone is a handle as any other,
    /// and one dropped with no report leaves another's report standing.
    #[test]
    fn the_last_report_of_a_sides_handles_stands() {
        let host = Host::of_singles();
        let single = Side::Single;
        let mut side = host.single();
        side.sent();
        let first = side.side_end();
        let second = first.clone();
        let silent = side.side_end();
        first.reported(read(502, TransportClass::ServerError));
        second.reported(read(200, TransportClass::Ok));
        drop(side);
        drop(first);
        drop(silent);
        assert_eq!(host.words(), [("issued", single), ("sent", single)]);
        drop(second);
        assert_eq!(one_end(&host).status, 200);
        assert_eq!(host.words(), [("ended", single)]);
    }

    /// An outcome reported of a side whose send was never said says the
    /// send first, as a placement's own end does, and states a span of
    /// zero.
    #[test]
    fn an_outcome_reported_of_an_unsent_side_says_its_send_first() {
        let host = Host::of_singles();
        let single = Side::Single;
        let side = host.single();
        let end = side.side_end();
        end.reported(read(502, TransportClass::ConnectFailed));
        drop(side);
        drop(end);
        let heard = one_end(&host);
        assert_eq!((heard.status, heard.duration_micros), (502, 0));
        let told = [("issued", single), ("sent", single), ("ended", single)];
        assert_eq!(host.words(), told);
    }

    /// A handle dropped with no report leaves a side's words as they are
    /// with no handle, said where the last handle goes: an end of class
    /// cancelled with no status and the span to the side's own drop, or a
    /// release where the side never went out.
    #[test]
    fn a_handle_dropped_with_no_report_leaves_a_dropped_sides_words() {
        let host = Host::of_singles();
        let single = Side::Single;
        let mut side = host.single();
        let start = Instant::now();
        side.sent();
        let end = side.side_end();
        let twin = end.clone();
        drop(side);
        let dropped = Instant::now();
        drop(end);
        let sent = [("issued", single), ("sent", single)];
        assert_eq!(host.words(), sent, "a handle is left");
        thread::sleep(Duration::from_millis(20));
        drop(twin);
        let heard = one_end(&host);
        let plain = Outcome {
            side: single,
            served: worker_id(host.workers[0].as_ref()),
            duration_micros: heard.duration_micros,
            ..Outcome::reported(0, TransportClass::Cancelled)
        };
        assert_eq!(heard, plain);
        let longest = dropped.duration_since(start).as_micros();
        let span = u128::from(heard.duration_micros);
        assert!(span <= longest, "{span} over {longest}");
        assert_eq!(host.words(), [("ended", single)]);

        let side = host.single();
        let end = side.side_end();
        drop(side);
        assert_eq!(host.words(), [("issued", single)], "a handle is left");
        drop(end);
        assert_eq!(host.words(), [("released", single)]);
    }

    /// A prefill's end with a handle left keeps the pair's order. Where
    /// the handle goes while the decode still stands chosen, the end is
    /// held back as at a side's drop, and said behind the decode's word
    /// with the span to the prefill side's drop. Where the decode's word
    /// comes first, the end is said where the handle goes.
    #[test]
    fn a_prefills_end_with_a_handle_left_keeps_the_pairs_order() {
        let host = Host::of_pairs();
        let told = [("issued", P), ("sent", P), ("issued", D), ("sent", D)];

        let (mut prefill, mut decode) = host.pair(true);
        let start = Instant::now();
        prefill.sent();
        let end = prefill.side_end();
        drop(prefill);
        let dropped = Instant::now();
        thread::sleep(Duration::from_millis(20));
        drop(end);
        assert_eq!(host.words(), told[..2], "the end waits");
        decode.sent();
        let span = host.prefills_end(&told[2..]);
        let longest = dropped.duration_since(start).as_micros();
        assert!(span <= longest, "{span} over {longest}");
        drop(decode);
        assert_eq!(host.words(), [("ended", D)]);

        let (mut prefill, mut decode) = host.pair(true);
        prefill.sent();
        let end = prefill.side_end();
        drop(prefill);
        decode.sent();
        assert_eq!(host.words(), told, "a handle is left");
        drop(end);
        assert_eq!(host.words(), [("ended", P)]);
        drop(decode);
        assert_eq!(host.words(), [("ended", D)]);
    }

    /// Each side of a pair ends with what its own handle reported, under
    /// its own side's lease and its own worker, with the span from its own
    /// dispatch.
    #[test]
    fn each_side_of_a_pair_ends_with_its_own_handles_report() {
        let host = Host::of_pairs();
        let ids = host.workers.each_ref().map(|worker| worker_id(&**worker));
        let (mut prefill, mut decode) = host.pair(false);
        prefill.sent();
        thread::sleep(Duration::from_millis(20));
        let start = Instant::now();
        decode.sent();
        let (of_prefill, of_decode) = (prefill.side_end(), decode.side_end());
        drop((prefill, decode));
        let told = [("issued", P), ("issued", D), ("sent", P), ("sent", D)];
        assert_eq!(host.words(), told, "each end waits for its handle");
        of_decode.reported(read(200, TransportClass::Ok));
        of_prefill.reported(read(500, TransportClass::ServerError));
        let longest = start.elapsed().as_micros();
        drop(of_decode);
        drop(of_prefill);
        let ends = host.ends();
        let spans = ends.iter().map(|end| u128::from(end.duration_micros));
        let [of_decodes, of_prefills] = spans.collect::<Vec<_>>()[..] else {
            panic!("two ends are heard: {ends:?}");
        };
        assert!(of_decodes <= longest, "{of_decodes} over {longest}");
        assert!(
            of_prefills >= 20_000,
            "{of_prefills}: the prefill's own span"
        );
        let spanless = |end| Outcome {
            duration_micros: 7,
            ..end
        };
        let heard: Vec<_> = ends.into_iter().map(spanless).collect();
        let stated = [
            Outcome {
                side: D,
                served: ids[1],
                ..read(200, TransportClass::Ok)
            },
            Outcome {
                side: P,
                served: ids[0],
                ..read(500, TransportClass::ServerError)
            },
        ];
        assert_eq!(heard, stated);
        assert_eq!(host.words(), [("ended", D), ("ended", P)]);
    }

    /// A prefill's end a handle reported is held back as an end with no
    /// outcome is, while the decode still stands chosen. It is said behind
    /// the decode side's word, with the outcome and the span of its
    /// report: at the decode's dispatch, or at the drop of a decode side
    /// that never went out.
    #[test]
    fn a_reported_prefills_end_is_held_for_the_decodes_word() {
        let host = Host::of_pairs();
        let told = [("issued", P), ("sent", P), ("issued", D), ("sent", D)];

        let (mut prefill, mut decode) = host.pair(true);
        let start = Instant::now();
        prefill.sent();
        let end = prefill.side_end();
        end.reported(read(200, TransportClass::Ok));
        let reported = Instant::now();
        drop(prefill);
        drop(end);
        assert_eq!(host.words(), told[..2], "the end waits");
        thread::sleep(Duration::from_millis(20));
        decode.sent();
        let heard = one_end(&host);
        assert_eq!((heard.side, heard.status), (P, 200));
        let longest = reported.duration_since(start).as_micros();
        let span = u128::from(heard.duration_micros);
        assert!(span <= longest, "{span} over {longest}");
        let behind = [("issued", D), ("sent", D), ("ended", P)];
        assert_eq!(host.words(), behind);
        drop(decode);
        assert_eq!(host.words(), [("ended", D)]);

        let (mut prefill, decode) = host.pair(true);
        prefill.sent();
        let end = prefill.side_end();
        end.reported(read(200, TransportClass::Ok));
        drop((prefill, end));
        assert_eq!(host.words(), told[..2], "the end waits");
        drop(decode);
        let heard = one_end(&host);
        assert_eq!((heard.side, heard.status), (P, 200));
        assert_eq!(host.words(), [("ended", P)]);
    }

    /// A prefill reported with no send, while its decode stands chosen,
    /// keeps the pair whole: its send is said first, and its end is held
    /// back for the decode side's word as a sent prefill's is. The decode
    /// still goes out under its lease, and both ends are said in the
    /// pair's order.
    #[test]
    fn a_prefill_reported_with_no_send_leaves_its_chosen_decode() {
        let host = Host::of_pairs();
        let (prefill, mut decode) = host.pair(true);
        let end = prefill.side_end();
        end.reported(read(502, TransportClass::ConnectFailed));
        drop((prefill, end));
        let sent = [("issued", P), ("sent", P)];
        assert_eq!(host.words(), sent, "the send is said, and the end waits");
        assert!(owes(&decode), "the chosen decode stays");
        decode.sent();
        let heard = one_end(&host);
        let stated = (P, 502, TransportClass::ConnectFailed, 0);
        let told = (heard.side, heard.status, heard.class, heard.duration_micros);
        assert_eq!(told, stated);
        let behind = [("issued", D), ("sent", D), ("ended", P)];
        assert_eq!(host.words(), behind);
        drop(decode);
        assert_eq!(host.words(), [("ended", D)]);

        // The decode asked again behind such a prefill is decided and sent.
        let host = Host::of_later_pairs();
        let classes = PartnerClasses::number(1, [&[0_u32][..]]);
        let request = later(host.lists(&classes));
        let (prefill, mut decode) = sides_of(&host, &request);
        let end = prefill.side_end();
        end.reported(read(200, TransportClass::Ok));
        drop((prefill, end));
        assert_eq!(host.words(), sent, "the send is said, and the end waits");
        decode.decode_again(&request).expect("a decode");
        assert_eq!(one_end(&host).status, 200);
        assert_eq!(host.words(), [("issued", D), ("ended", P)]);
        decode.sent();
        drop(decode);
        assert_eq!(host.words(), [("sent", D), ("ended", D)]);
    }

    /// Under a document none of whose stages takes an end a side keeps its
    /// placement by itself: the handle it hands out is empty, and a report
    /// through it says nothing. The side says the words it says with no
    /// handle: its dispatch, and at its drop an end no stage takes, or its
    /// release. A side of a placement that holds no lease hands out an
    /// empty handle under any document, and the default handle is empty.
    #[test]
    fn a_handle_is_empty_where_no_stage_takes_an_end() {
        assert!(SideEnd::default().is_empty());
        without_an_end(|| {
            let host = Host::of_singles();
            let single = Side::Single;
            let mut side = host.single();
            assert!(matches!(side.kept, Kept::Own(_)));
            let end = side.side_end();
            assert!(end.is_empty());
            side.sent();
            end.reported(read(200, TransportClass::Ok));
            drop(side);
            drop(end);
            assert_eq!(host.words(), [("issued", single), ("sent", single)]);
            drop(host.single());
            let released = [("issued", single), ("released", single)];
            assert_eq!(host.words(), released);

            let (prefill, decode) = Host::of_pairs().pair(false);
            assert!(prefill.side_end().is_empty() && decode.side_end().is_empty());
        });

        let host = Host::of_singles();
        let alone = Placement::alone(Arc::clone(&host.workers[0]), "round_robin");
        let (side, none) = alone.sides();
        assert!(none.is_none() && side.side_end().is_empty());
    }

    /// A handle travels with its holder: to the task that reads the
    /// answer, and behind a pointer several threads share. Two handles of
    /// one side report from two threads beside the side's drop: the side
    /// ends once, with one of the two outcomes.
    #[test]
    fn a_sides_handles_report_from_two_threads() {
        fn travels<T: Send + Sync + 'static>() {}
        travels::<SideEnd>();

        let host = Host::of_singles();
        for round in 0..64_u16 {
            let mut side = host.single();
            side.sent();
            let (first, second) = (side.side_end(), side.side_end());
            let reports = [(first, 200 + round), (second, 300 + round)].map(|(end, status)| {
                thread::spawn(move || end.reported(read(status, TransportClass::Ok)))
            });
            drop(side);
            for report in reports {
                report.join().expect("a reporting thread");
            }
            let status = one_end(&host).status;
            assert!([200 + round, 300 + round].contains(&status), "{status}");
            assert_eq!(host.words().len(), 3, "round {round}");
        }
    }
}
