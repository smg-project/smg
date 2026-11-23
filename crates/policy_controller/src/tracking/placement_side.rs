//! A placement's sides: one placed request as the values its host keeps
//! with each send, from that send to that side's end.
//!
//! A host that sends a request keeps something with the send until the
//! response ends: a gateway keeps the worker's load guard. A kept
//! placement gives such a host one owned value a side
//! ([`Placement::sides`]): a single request's one, a pair's prefill and
//! decode. A side says its dispatch by one call ([`PlacementSide::sent`])
//! and its end at its drop, with no outcome, in the words a dropped
//! [`Placement`] says: a release where the side never went out, an end of
//! class cancelled where it did. Its host writes no drop of its own and
//! shares nothing itself. A side also answers the label of the profile
//! that picked it ([`PlacementSide::profile`]), so a host counts each side
//! by the value it keeps.
//!
//! **A pair's two sides** share the pair's one placement behind a lock.
//! The lock is one request's own: its two sides alone take it, each for
//! the time of one call or of its drop, and no other request ever waits
//! on it. The decode side holds it through its second decision
//! ([`PlacementSide::decode_again`]), so whatever the request lends that
//! decision is asked under it: its route's rule
//! ([`PlaceRequest::admits`]), its headers ([`PlaceRequest::headers`]),
//! its ring ([`PlaceRequest::ring`]) and its lists' admission. None of
//! them may send or drop a side of the pair: it would wait on the lock it
//! is asked under and never return. A single request's side owns its
//! placement and takes no lock.
//!
//! **The order of a pair's words** is kept here, in whatever order its
//! host lets the two sides go. A decode that still stands chosen leaves
//! with its prefill's end
//! ([`DecisionGuard::completed`](super::DecisionGuard::completed)), and
//! its send would then say nothing. So a prefill side dropped behind its
//! send, while the decode still stands chosen, holds its end back until
//! the decode side speaks: at its dispatch, at the call that takes its
//! decode again, or at its own drop. The end is said then with the instant
//! the prefill side was dropped at, so the span a stage reads is the
//! prefill's own.
//!
//! **A decode decided again** is asked of the pair's decode side
//! ([`PlacementSide::decode_again`]), behind its prefill's answer. Its host
//! may let the prefill side go before that call or behind it.

use std::{fmt, sync::Arc, time::Instant};

use parking_lot::Mutex;
use smg_worker::identity::basic::Worker;

use super::{
    lease::{ReleaseReason, Side},
    placement::Placement,
};
use crate::engine::{
    decision_engine::{DecisionRefused, DecodeRefused},
    PlaceRequest, Unplaced,
};

/// One side of a placed request, kept by its host from that side's send to
/// its end ([`Placement::sides`]).
///
/// Dropped, it says the side's end with no outcome, once. A side that
/// never went out is released as abandoned. A side that went out ends as
/// class cancelled, with the span from its dispatch. A side of a placement
/// that owes no event says nothing and reads no clock.
///
/// A pair's sides say two of those words otherwise (the module's words on
/// the order). A prefill side dropped behind its send, while the decode
/// still stands chosen, says its end later: behind the decode side's word.
/// A decode side that stands chosen and never went out says nothing at its
/// drop: nothing was booked for it.
pub struct PlacementSide {
    side: Side,
    /// The label of the profile that picked this side.
    profile: &'static str,
    of: Of,
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
    /// The instant the prefill side was dropped at, where its end is held
    /// back for the decode side's word.
    prefill_dropped: Option<Instant>,
}

impl Pair {
    /// Says the prefill's end where it is held back, with the instant its
    /// side was dropped at.
    fn end_held_prefill(&mut self) {
        let Some(at) = self.prefill_dropped.take() else {
            return;
        };
        if let Some(held) = &mut self.placement.held {
            held.released_prefill(ReleaseReason::Abandoned, at);
        }
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
        if !pair {
            let whole = PlacementSide {
                side: Side::Single,
                profile,
                of: Of::Whole(self),
            };
            return (whole, None);
        }
        let decode_profile = self.decode_profile.unwrap_or(profile);
        let pair = Arc::new(Mutex::new(Pair {
            placement: self,
            prefill_dropped: None,
        }));
        let prefill = PlacementSide {
            side: Side::Prefill,
            profile,
            of: Of::Shared(Arc::clone(&pair)),
        };
        let decode = PlacementSide {
            side: Side::Decode,
            profile: decode_profile,
            of: Of::Shared(pair),
        };
        (prefill, Some(decode))
    }
}

impl PlacementSide {
    /// The label of the profile that picked this side, as its placement
    /// names it: [`Placement::profile`], and for a pair's decode side
    /// [`Placement::decode_profile`]. What a host counts the side under.
    #[must_use]
    pub fn profile(&self) -> &'static str {
        self.profile
    }

    /// This side of the request went out: its dispatch, said once
    /// ([`Placement::sent`]).
    pub fn sent(&mut self) {
        match &mut self.of {
            Of::Whole(placement) => placement.sent(self.side),
            Of::Shared(pair) => {
                let mut pair = pair.lock();
                pair.placement.sent(self.side);
                // The decode no longer stands chosen: a prefill's end that
                // waited for this word is said behind it.
                if self.side == Side::Decode {
                    pair.end_held_prefill();
                }
            }
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
    /// must leave the pair's sides alone (the module's words on the lock).
    ///
    /// **The order.** The call is made once the prefill side's send was
    /// said. Its host may let the prefill side go before the call or
    /// behind it: a prefill's end that waits for this side's word is said
    /// behind the decode the call took, with the instant the prefill side
    /// was dropped at. Behind a call that took no decode nothing more is
    /// said, and that end still waits for this side's word.
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
        let (Side::Decode, Of::Shared(pair)) = (self.side, &self.of) else {
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
}

impl Drop for PlacementSide {
    /// The side's end, with no outcome. A single request's placement says
    /// its own at its drop.
    fn drop(&mut self) {
        let Of::Shared(pair) = &self.of else {
            return;
        };
        let mut pair = pair.lock();
        let pair = &mut *pair;
        let Some(held) = &mut pair.placement.held else {
            return;
        };
        // The common drop of a side that ended: nothing is owed on either
        // side, so no word is said and no clock is read.
        if !held.owes() {
            return;
        }
        let now = Instant::now();
        match self.side {
            Side::Decode => {
                held.released_decode(ReleaseReason::Abandoned, now);
                pair.end_held_prefill();
            }
            // Said now, this end would take the chosen decode with it.
            _ if held.end_takes_decode() => pair.prefill_dropped = Some(now),
            _ => held.released_prefill(ReleaseReason::Abandoned, now),
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

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

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
            view::{Candidate, Constraints, Phase, RequestFacts},
            PairAdmission, PairLists, PartnerClasses,
        },
        stages::{pickers::push_drawn, registry::StageRegistry, Lifecycle},
        store::GatewayId,
        tracking::lease::{EventSet, Lease, Outcome, TransportClass},
    };

    const MODEL: &str = "model-a";

    /// One word a stage heard of a lease: the word, the side its lease
    /// books, and the span an end states, in microseconds.
    type Word = (&'static str, Side, u32);

    /// A picker that names the workers the filters keep in their list's
    /// order, and keeps every word of a lease it hears. An end of another
    /// class than cancelled or with a status, and a release under another
    /// reason than abandoned, fail its hearing.
    #[derive(Debug)]
    struct Hears(Arc<Mutex<Vec<Word>>>);

    impl Lifecycle for Hears {
        fn hears(&self) -> EventSet {
            EventSet::LEASE
        }

        fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
            self.0.lock().push(("issued", lease.side, 0));
        }

        fn on_dispatched(&self, lease: &Lease, _: &Target, _: Side, _: u32) {
            self.0.lock().push(("sent", lease.side, 0));
        }

        fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
            let plain = (TransportClass::Cancelled, 0);
            assert_eq!((outcome.class, outcome.status), plain);
            let span = outcome.duration_micros;
            self.0.lock().push(("ended", lease.side, span));
        }

        fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
            assert_eq!(reason, ReleaseReason::Abandoned);
            self.0.lock().push(("released", lease.side, 0));
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
        Placement::taken(engine, decided, later, now, 0, worker, decode)
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
    /// whose `profile` picks through [`Hears`], the words that stage heard,
    /// and one prefill and one decode worker, each one's list published
    /// with its cell.
    struct Host {
        engine: &'static PolicyEngine,
        heard: Arc<Mutex<Vec<Word>>>,
        workers: [Arc<dyn Worker>; 2],
        lists: [PublishedList; 2],
    }

    impl Host {
        fn under(mut document: PolicySpec, profile: &str) -> Self {
            let heard = Arc::new(Mutex::new(Vec::new()));
            let kept = Arc::clone(&heard);
            let mut stages = StageRegistry::builtin();
            let added = stages.picker("hears", move || Box::new(Hears(Arc::clone(&kept))));
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

        /// What the stage heard since the last call: each word and its
        /// side. No word went unheard, and no decision faulted.
        fn words(&self) -> Vec<(&'static str, Side)> {
            assert_eq!((self.engine.faults(), self.engine.unheard()), (0, 0));
            let heard = std::mem::take(&mut *self.heard.lock());
            heard
                .into_iter()
                .map(|(word, side, _)| (word, side))
                .collect()
        }

        /// The span the prefill's end states, in microseconds: the last
        /// word the stage heard since the last call, behind `ahead`.
        fn prefills_end(&self, ahead: &[(&'static str, Side)]) -> u128 {
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
        match &side.of {
            Of::Whole(placement) => placement.owes_events(),
            Of::Shared(pair) => pair.lock().placement.owes_events(),
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
    /// to its own drop. A prefill that never went out is released at once,
    /// and the chosen decode leaves with it: the placement owes nothing
    /// more.
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
        assert_eq!(host.words(), [("issued", P), ("released", P)]);
        assert!(!owes(&decode), "the chosen decode left with its prefill");
        drop(decode);
        assert!(host.words().is_empty());
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
}
