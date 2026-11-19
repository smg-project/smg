//! The handle: a decision as its host holds it, from the answer to the end
//! of the request.
//!
//! A host in process asks its door for a decision ([`Handle::decide`]) and
//! keeps the handle while it commits a target, sends the request and reads
//! its end. Through it the host says of the decision's lease what a
//! gateway's session says to a served controller in frames: its dispatch,
//! its completion or its release. No frame is built. Each report goes
//! through the door ([`Door::report`]) to the file the lease's id names,
//! as the served controller hands a frame's. The thread that says it is
//! never held for a decision, and whoever applies it tells the engine
//! that issued the lease.
//!
//! **Once.** A lease is reported once on every path. A handle knows what
//! it still owes its lease: a dispatch, a completion, or nothing. A report
//! that is not owed says nothing, and a handle dropped with a report owed
//! says it.
//!
//! **Before a dispatch** a lease ends by a release, under its host's
//! reason: the commit refused every target
//! ([`ReleaseReason::CommitFailed`]), or the client went away
//! ([`ReleaseReason::ClientGone`]). A handle dropped there knows no
//! reason: a client left, a timeout fired or an error path returned
//! early. It gives the lease up as abandoned
//! ([`ReleaseReason::Abandoned`]), the word for a lease that ends before
//! a dispatch with no cause given. A host that knows its client left
//! says so before it lets the handle go.
//!
//! **After a dispatch** a lease lives until its completion. A client that
//! leaves then is a completion of class cancelled
//! ([`TransportClass::Cancelled`]), and so are a release said then and a
//! handle dropped with no outcome.
//!
//! **The instants.** A dispatch, a completion and a release each take the
//! instant their host read, and a span is counted between two of them. A
//! handle dropped with a report owed is handed no instant: it reads the
//! clock, so a host whose instants are not the clock's says the release
//! itself.
//!
//! **The targets not taken.** A pick's lease books the pick's first
//! target, and a dispatch that names another moves the booking there
//! ([`Handle::take`]). A target the host passes over holds nothing on
//! file: its release is the commit itself, and no report.
//!
//! A decision that picks no worker has no lease, and its handle owes
//! nothing.
//!
//! **A pair.** A pair's prefill and decode share a lease id, and the
//! handle reports each side. Before a dispatch, [`Handle::take`] takes a
//! target for both sides. After the prefill's dispatch,
//! [`Handle::take_decode`] can take another decode of that prefill.
//! [`Handle::dispatched`] says the prefill's dispatch, and
//! [`Handle::dispatched_decode`] the decode's. A completion names the
//! side that ended. A release or a drop gives up both sides in one word
//! while both still owe a dispatch. After a dispatch it completes that
//! side as cancelled and releases a waiting side by name.
//! [`Handle::released_decode`] gives up the decode alone.

use std::{
    fmt,
    ops::Deref,
    time::{Duration, Instant},
};

use crate::{
    decision::{Decision, Target},
    door::{self, Decided, Door, Leased, Refused, Request},
    lease::{LeaseId, Outcome, ReleaseReason, Side, TransportClass},
    store::{Dispatch, GatewayId, Said},
    view::WorkerId,
};

/// What a handle still owes its lease.
#[derive(Clone, Copy, Debug)]
enum Owed {
    /// Nothing: the lease's end was said, or the decision picked no
    /// worker.
    Nothing,
    /// A dispatch, or a release in its place: the lease stands as issued
    /// since `decided`, the decision's instant.
    Dispatch {
        lease: LeaseId,
        side: Side,
        decided: Instant,
    },
    /// A completion: the lease was dispatched to `worker` at `sent`.
    Completion {
        lease: LeaseId,
        side: Side,
        sent: Instant,
        worker: WorkerId,
    },
}

/// A span in whole microseconds, saturating.
fn micros(span: Duration) -> u32 {
    u32::try_from(span.as_micros()).unwrap_or(u32::MAX)
}

/// A decision as its host holds it: one a decision, from the answer to
/// the end of the request.
///
/// `D` is how the host holds its door: a borrow where the handle stays in
/// its caller's frame, a shared pointer where it travels with the request.
/// The handle keeps it for its reports and for its drop.
pub struct Handle<D: Deref<Target = Door>> {
    door: D,
    /// The gateway the decision was made for: its reports are said under
    /// it.
    gateway: GatewayId,
    decided: Decided,
    /// The index to name at dispatch: the first until the host takes
    /// another. After the prefill dispatch it can move for the decode.
    index: u8,
    /// The target at that index.
    target: Target,
    owed: Owed,
    /// What the handle still owes the decode side's lease of a pair.
    /// Nothing for a single.
    decode: Owed,
}

impl<D: Deref<Target = Door>> Handle<D> {
    /// One decision of `door` for `gateway`'s `request` at `now`, on the
    /// calling thread ([`Door::decide`]), held.
    ///
    /// # Errors
    ///
    /// [`Refused`]: the door made no decision for the request.
    pub fn decide(
        door: D,
        gateway: GatewayId,
        request: &Request<'_>,
        now: Instant,
    ) -> Result<Self, Refused> {
        let decided = door.decide(gateway, request, now)?;
        let first = decided
            .decision
            .pick()
            .and_then(|pick| pick.targets.first());
        let target = first.copied().unwrap_or_default();
        let owed = decided
            .lease
            .map_or(Owed::Nothing, |leased| Owed::Dispatch {
                lease: leased.lease.id,
                side: leased.lease.side,
                decided: now,
            });
        let decode = match owed {
            Owed::Dispatch { lease, .. } if decided.pair() => Owed::Dispatch {
                lease,
                side: Side::Decode,
                decided: now,
            },
            _ => Owed::Nothing,
        };
        Ok(Self {
            door,
            gateway,
            decided,
            index: 0,
            target,
            owed,
            decode,
        })
    }

    /// The engine's decision.
    #[must_use]
    pub fn decision(&self) -> &Decision {
        &self.decided.decision
    }

    /// The lease the door filed for a pick, the prefill side for a pair.
    /// `None` for a decision that picks no worker.
    #[must_use]
    pub fn lease(&self) -> Option<&Leased> {
        self.decided.lease.as_ref()
    }

    /// Takes the target at `index` of the pick, best first: the one the
    /// host's commit holds, which its dispatch then names. A host that
    /// takes none dispatches to the first. A take says nothing to the
    /// store: the lease holds one credit, on the pick's first target, and
    /// the dispatch names the index taken, where the file moves the
    /// record. Nothing stands on a target not taken, so there is nothing
    /// to release. `None` where the pick has no target at `index`, and
    /// where no dispatch is owed any more: what was taken before stands.
    /// A pair takes the target for each side that still owes dispatch.
    pub fn take(&mut self, index: usize) -> Option<Target> {
        let Owed::Dispatch { .. } = self.owed else {
            return None;
        };
        let (taken, target) = self.target_at(index)?;
        (self.index, self.target) = (taken, target);
        Some(target)
    }

    /// Takes the decode of target `index` after the prefill dispatch.
    /// The target must name the prefill already sent. The decode's
    /// dispatch moves its booking to the decode of that target.
    /// `None` if the target cannot be taken or the decode owes no dispatch.
    pub fn take_decode(&mut self, index: usize) -> Option<Target> {
        if matches!(self.owed, Owed::Dispatch { .. })
            || !matches!(self.decode, Owed::Dispatch { .. })
        {
            return None;
        }
        let (taken, target) = self.target_at(index)?;
        if target.id != self.target.id {
            return None;
        }
        (self.index, self.target) = (taken, target);
        Some(target)
    }

    fn target_at(&self, index: usize) -> Option<(u8, Target)> {
        let taken = u8::try_from(index).ok()?;
        let pick = self.decided.decision.pick()?;
        let target = *pick.targets.as_slice().get(index)?;
        Some((taken, target))
    }

    /// Says through the door what the host says of `lease`: the calling
    /// thread is never held for a decision ([`Door::report`]).
    fn say(&self, lease: LeaseId, said: Said) {
        self.door.report(self.gateway, lease, said);
    }

    /// The host sent the request to the target it took, at `now`: the
    /// lease is dispatched on that target, and no dispatch deadline ends
    /// it from here. A dispatch said after the door's clock ended the
    /// lease past that deadline (a sweep, or another decision's issue at
    /// its file) meets no lease: the request runs unbooked, and its host
    /// is told nothing. Said once: a second call says nothing.
    /// For a pair this says the prefill's dispatch.
    pub fn dispatched(&mut self, now: Instant) {
        self.owed = self.dispatch(self.owed, self.target.id, now);
    }

    /// The host sent the decode to the target it took, at `now`. Said
    /// once, for a pair's decode side alone.
    pub fn dispatched_decode(&mut self, now: Instant) {
        if let Some(pair) = self.target.pair {
            self.decode = self.dispatch(self.decode, pair.decode_id, now);
        }
    }

    /// Says a dispatch and keeps the worker it named for a cancelled end.
    fn dispatch(&self, owed: Owed, worker: WorkerId, now: Instant) -> Owed {
        let Owed::Dispatch {
            lease,
            side,
            decided,
        } = owed
        else {
            return owed;
        };
        let dispatch = Dispatch {
            target_index: self.index,
            side,
            since_decision_micros: micros(now.saturating_duration_since(decided)),
        };
        self.say(lease, Said::Dispatched(dispatch, now));
        Owed::Completion {
            lease,
            side,
            sent: now,
            worker,
        }
    }

    /// The request ended with `outcome`, as its host read it at `now`: the
    /// lease ends, and the completion is written on the served worker for
    /// the health precedence, as a served controller writes a gateway's.
    /// A completion said with no dispatch says the dispatch first, at
    /// `now`: a request whose send failed was sent, and the engine hears a
    /// lease's events in their order. That dispatch carries the
    /// completion's instant, so a stage that times a request from its
    /// dispatch reads a span of zero for it. A decode outcome ends the
    /// decode side. Other outcomes end the handle's own side. Said once.
    pub fn completed(&mut self, outcome: &Outcome, now: Instant) {
        let owed = if outcome.side == Side::Decode {
            self.dispatched_decode(now);
            &mut self.decode
        } else {
            self.dispatched(now);
            &mut self.owed
        };
        let Owed::Completion { lease, side, .. } = *owed else {
            return;
        };
        *owed = Owed::Nothing;
        let outcome = Outcome {
            side: if side == Side::Prefill {
                side
            } else {
                outcome.side
            },
            ..*outcome
        };
        self.say_completion(lease, &outcome, now);
    }

    fn say_completion(&self, lease: LeaseId, outcome: &Outcome, now: Instant) {
        door::completed(self.door.store(), self.gateway, outcome, now, |said| {
            self.say(lease, said);
        });
    }

    /// Gives up every side at `now`, under `reason`, in one release if
    /// both sides of a pair still owe dispatch. A dispatched side completes
    /// as cancelled on the worker it was sent to, with the span from its
    /// dispatch to `now`. A waiting side is released. Said once per side.
    pub fn released(&mut self, reason: ReleaseReason, now: Instant) {
        if let (Owed::Dispatch { lease, .. }, Owed::Dispatch { .. }) = (self.owed, self.decode) {
            self.say(lease, Said::Released(None, reason));
            self.owed = Owed::Nothing;
            self.decode = Owed::Nothing;
            return;
        }
        self.owed = self.release(self.owed, reason, now);
        self.released_decode(reason, now);
    }

    /// Gives up the decode side alone. A sent decode completes as
    /// cancelled. A waiting decode is released by name under `reason`.
    pub fn released_decode(&mut self, reason: ReleaseReason, now: Instant) {
        self.decode = self.release(self.decode, reason, now);
    }

    fn release(&self, owed: Owed, reason: ReleaseReason, now: Instant) -> Owed {
        match owed {
            Owed::Nothing => {}
            Owed::Dispatch { lease, side, .. } => {
                let side = (side != Side::Single).then_some(side);
                self.say(lease, Said::Released(side, reason));
            }
            Owed::Completion {
                lease,
                side,
                sent,
                worker,
            } => {
                let cancelled = Outcome {
                    side,
                    served: worker,
                    status: 0,
                    class: TransportClass::Cancelled,
                    ttft_micros: 0,
                    tokens_in: 0,
                    tokens_out: 0,
                    duration_micros: micros(now.saturating_duration_since(sent)),
                };
                self.say_completion(lease, &cancelled, now);
            }
        }
        Owed::Nothing
    }
}

impl<D: Deref<Target = Door>> Drop for Handle<D> {
    /// A handle dropped with a report owed speaks for a lease nobody
    /// spoke for, in the word of its phase that names no cause: a release
    /// as abandoned before any dispatch, a completion of class cancelled
    /// after one. A drop is handed no instant and reads the clock.
    fn drop(&mut self) {
        self.released(ReleaseReason::Abandoned, Instant::now());
    }
}

impl<D: Deref<Target = Door>> fmt::Debug for Handle<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Handle")
            .field("decided", &self.decided)
            .field("index", &self.index)
            .field("owed", &self.owed)
            .field("decode", &self.decode)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        error::Error,
        mem,
        sync::{mpsc, Arc, Barrier},
        thread,
        time::{Duration, Instant},
    };

    use parking_lot::Mutex;

    use super::Handle;
    use crate::{
        builtins::Random,
        decision::{Decision, Target, Targets},
        door::{
            tests::{
                counted, fleet, held as deciding, pair_counts, pair_fleet, picking, request, Ends,
                GATEWAY, PATIENCE,
            },
            Door, Refused,
        },
        framework::{Picker, TieBand},
        lease::{Lease, LeaseId, Lifecycle, Outcome, ReleaseReason, Side, TransportClass},
        store::{Dispatch, HealthSink, LeaseState, LeasesStore, MemoryStore, Said, WorkersStore},
        view::{Candidate, Phase, RequestFacts, WorkerId},
    };

    /// What a stage heard of a lease.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Call {
        Issued,
        /// The target the dispatch named, its index on the lease told, and
        /// the microseconds since the decision.
        Dispatched(WorkerId, u8, u32),
        /// The served worker, the class and the microseconds the dispatch
        /// took.
        Completed(WorkerId, TransportClass, u32),
        Released(ReleaseReason),
    }

    type Heard = Arc<Mutex<Vec<Call>>>;

    /// The `random` picker, keeping what the engine tells it of leases.
    #[derive(Debug)]
    struct Hears(Heard);

    impl Lifecycle for Hears {
        fn on_decision_issued(&self, _: &Lease, _: &Decision) {
            self.0.lock().push(Call::Issued);
        }

        fn on_dispatched(&self, lease: &Lease, target: &Target, _: Side, since: u32) {
            let call = Call::Dispatched(target.id, lease.target_index, since);
            self.0.lock().push(call);
        }

        fn on_completed(&self, _: &Lease, outcome: &Outcome) {
            let call = Call::Completed(outcome.served, outcome.class, outcome.duration_micros);
            self.0.lock().push(call);
        }

        fn on_released(&self, _: &Lease, reason: ReleaseReason) {
            self.0.lock().push(Call::Released(reason));
        }
    }

    impl Picker for Hears {
        fn name(&self) -> &'static str {
            "hears"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            Random.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// A door over the fleet's three ready members, whose picker keeps
    /// what the engine tells it of leases, and what it heard.
    fn bed() -> (Door, Heard) {
        let store = fleet(1);
        let heard = Heard::default();
        let kept = Arc::clone(&heard);
        picking(&store, 2, "hears", move || {
            Box::new(Hears(Arc::clone(&kept)))
        });
        (Door::new(store), heard)
    }

    /// One decision of `door` for the request `id`, held.
    fn held<'a>(door: &'a Door, id: &[u8], now: Instant) -> Handle<&'a Door> {
        let decided = Handle::decide(door, GATEWAY, &request(id, 7), now);
        decided.expect("a decision")
    }

    /// An outcome of `class` on `served`, as a host reads one.
    fn outcome(served: WorkerId, class: TransportClass) -> Outcome {
        Outcome {
            side: Side::Single,
            served,
            status: 200,
            class,
            ttft_micros: 1_000,
            tokens_in: 12,
            tokens_out: 34,
            duration_micros: 5_000,
        }
    }

    /// What was heard, with a cancelled completion's time read as 0: a
    /// clock made it, and it is `least` microseconds at least.
    fn clocked(heard: &Heard, least: u32) -> Vec<Call> {
        let heard = heard.lock();
        let calls = heard.iter().map(|call| match *call {
            Call::Completed(served, TransportClass::Cancelled, took) => {
                assert!(took >= least, "a cancelled request took {took}");
                Call::Completed(served, TransportClass::Cancelled, 0)
            }
            said => said,
        });
        calls.collect()
    }

    /// A request that is served: its lease is booked at the issue, moves
    /// to dispatched at the host's word and ends at its completion, which
    /// is written on the served worker. Each is told once: what the host
    /// says after the end, and the drop, say nothing.
    #[test]
    fn a_served_request_tells_its_dispatch_and_its_completion_once_each() {
        let (door, heard) = bed();
        let store = door.store();
        let now = Instant::now();
        let mut handle = held(&door, b"r-1", now);
        let lease = handle.lease().expect("a pick's lease").lease.id;
        let first = handle.take(0).expect("the pick's first target");
        assert_eq!(counted(store), [(first.id, 1)], "booked at the issue");
        let sent = now + Duration::from_micros(250);
        handle.dispatched(sent);
        let on_file = LeasesStore::get(&**store, GATEWAY, lease);
        let state = on_file.expect("the lease is on file").state;
        assert_eq!(state, LeaseState::Dispatched { at: sent });
        let ended = now + Duration::from_millis(5);
        handle.completed(&outcome(first.id, TransportClass::Ok), ended);
        assert_eq!(counted(store), []);
        assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
        assert!(store.completed_ok_since(GATEWAY, first.id, now));
        // Nothing more is owed: what its host says behind the end is told
        // to nobody and written nowhere.
        let later = ended + Duration::from_secs(1);
        handle.dispatched(later);
        handle.completed(&outcome(first.id, TransportClass::Ok), later);
        handle.released(ReleaseReason::ClientGone, Instant::now());
        drop(handle);
        let again = store.completed_ok_since(GATEWAY, first.id, later);
        assert!(!again, "one completion is written");
        let told = [
            Call::Issued,
            Call::Dispatched(first.id, 0, 250),
            Call::Completed(first.id, TransportClass::Ok, 5_000),
        ];
        assert_eq!(*heard.lock(), told);
    }

    /// A host whose commit passes over the first target takes another: a
    /// take says nothing to the store, and the dispatch moves the lease's
    /// booking to the target taken. A take of no target of the pick, and
    /// one behind the dispatch, change nothing.
    #[test]
    fn a_dispatch_on_an_alternate_moves_the_leases_booking_there() {
        let (door, heard) = bed();
        let store = door.store();
        let now = Instant::now();
        let mut handle = held(&door, b"r-1", now);
        let first = handle.take(0).expect("the pick's first target");
        let second = handle.take(1).expect("an alternate of the pick");
        assert_ne!(first.id, second.id);
        assert_eq!(handle.take(9), None, "no such target");
        assert_eq!(counted(store), [(first.id, 1)], "a take books nothing");
        // Two hours behind the decision: more microseconds than the span's
        // field holds, so it says its most.
        handle.dispatched(now + Duration::from_secs(7_200));
        assert_eq!(counted(store), [(second.id, 1)]);
        assert_eq!(handle.take(0), None, "the dispatch named its target");
        handle.completed(&outcome(second.id, TransportClass::Ok), now);
        assert_eq!(counted(store), []);
        let told = [
            Call::Issued,
            Call::Dispatched(second.id, 1, u32::MAX),
            Call::Completed(second.id, TransportClass::Ok, 5_000),
        ];
        assert_eq!(*heard.lock(), told);
    }

    /// A host whose commit refuses every target releases the lease under
    /// that reason, once: nothing it says behind the release is told.
    #[test]
    fn a_commit_that_refuses_every_target_releases_the_lease_once() {
        let (door, heard) = bed();
        let store = door.store();
        let now = Instant::now();
        let mut handle = held(&door, b"r-1", now);
        let first = handle.take(0).expect("the pick's first target");
        handle.released(ReleaseReason::CommitFailed, Instant::now());
        assert_eq!(counted(store), []);
        assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
        handle.released(ReleaseReason::ClientGone, Instant::now());
        handle.completed(&outcome(first.id, TransportClass::Ok), now);
        drop(handle);
        let written = store.completed_ok_since(GATEWAY, first.id, now);
        assert!(!written, "no completion is written behind a release");
        let told = [Call::Issued, Call::Released(ReleaseReason::CommitFailed)];
        assert_eq!(*heard.lock(), told);
    }

    /// A handle dropped with no call knows no reason for its drop: before
    /// any dispatch its lease is given up as abandoned, the word for a
    /// lease nobody spoke for.
    #[test]
    fn a_handle_dropped_with_no_call_gives_its_lease_up_as_abandoned() {
        let (door, heard) = bed();
        let store = door.store();
        let handle = held(&door, b"r-1", Instant::now());
        assert_eq!(LeasesStore::live(&**store, GATEWAY), 1);
        drop(handle);
        assert_eq!(counted(store), []);
        assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
        let told = [Call::Issued, Call::Released(ReleaseReason::Abandoned)];
        assert_eq!(*heard.lock(), told);
    }

    /// A host that knows its client left before any dispatch says so, and
    /// the handle's drop behind it says nothing more.
    #[test]
    fn a_host_that_knows_its_client_left_says_so_once() {
        let (door, heard) = bed();
        let store = door.store();
        let mut handle = held(&door, b"r-1", Instant::now());
        handle.released(ReleaseReason::ClientGone, Instant::now());
        assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
        drop(handle);
        let told = [Call::Issued, Call::Released(ReleaseReason::ClientGone)];
        assert_eq!(*heard.lock(), told);
    }

    /// After a dispatch a lease lives until its completion: a handle
    /// dropped with no outcome, and a release said there, are told as one
    /// completion of class cancelled on the dispatched target, which says
    /// the time since the dispatch. It is no completion of class ok for
    /// the health precedence.
    #[test]
    fn behind_a_dispatch_a_drop_and_a_release_are_a_cancelled_completion() {
        for released in [false, true] {
            let (door, heard) = bed();
            let store = door.store();
            let now = Instant::now();
            let mut handle = held(&door, b"r-1", now);
            let second = handle.take(1).expect("an alternate of the pick");
            handle.dispatched(now);
            thread::sleep(Duration::from_millis(1));
            if released {
                handle.released(ReleaseReason::ClientGone, Instant::now());
                assert_eq!(counted(store), [], "told where it was said");
            }
            drop(handle);
            assert_eq!(counted(store), []);
            assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
            assert!(!store.completed_ok_since(GATEWAY, second.id, now));
            let told = [
                Call::Issued,
                Call::Dispatched(second.id, 1, 0),
                Call::Completed(second.id, TransportClass::Cancelled, 0),
            ];
            assert_eq!(clocked(&heard, 1_000), told, "released: {released}");
        }
    }

    /// A request whose send failed was sent: a completion said with no
    /// dispatch says the dispatch first, on the target taken and at the
    /// completion's instant, so the engine hears the lease's events in
    /// their order.
    #[test]
    fn a_completion_said_with_no_dispatch_says_the_dispatch_first() {
        let (door, heard) = bed();
        let now = Instant::now();
        let mut handle = held(&door, b"r-1", now);
        let second = handle.take(1).expect("an alternate of the pick");
        let failed = outcome(second.id, TransportClass::ConnectFailed);
        handle.completed(&failed, now + Duration::from_micros(400));
        assert_eq!(counted(door.store()), []);
        let told = [
            Call::Issued,
            Call::Dispatched(second.id, 1, 400),
            Call::Completed(second.id, TransportClass::ConnectFailed, 5_000),
        ];
        assert_eq!(*heard.lock(), told);
    }

    /// A decision that picks no worker has no lease: its handle takes no
    /// target and tells nothing, whatever its host says.
    #[test]
    fn a_decision_that_picks_no_worker_owes_no_report() {
        let (door, heard) = bed();
        let store = door.store();
        let now = Instant::now();
        // A member of the fleet, by a decision that picks one: a
        // completion on a worker the store holds no row of is written
        // for nobody.
        let mut picked = held(&door, b"r-0", now);
        let member = picked.take(0).expect("the pick's first target").id;
        picked.released(ReleaseReason::CommitFailed, Instant::now());
        heard.lock().clear();
        let mut asked = request(b"r-1", 7);
        asked.facts.model = "model-none";
        let decided = Handle::decide(&door, GATEWAY, &asked, now);
        let mut handle = decided.expect("a decision");
        assert!(handle.decision().pick().is_none(), "no pool, no pick");
        assert!(handle.lease().is_none());
        assert_eq!(handle.take(0), None);
        handle.dispatched(now);
        handle.completed(&outcome(member, TransportClass::Ok), now);
        handle.released(ReleaseReason::CommitFailed, Instant::now());
        drop(handle);
        assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
        assert!(!store.completed_ok_since(GATEWAY, member, now));
        assert_eq!(*heard.lock(), []);
    }

    /// A host that takes no target dispatches to the pick's first, and a
    /// handle let go behind that dispatch completes the lease as cancelled
    /// on that target. The completion's span counts from the dispatch: one
    /// said ahead of the clock has no span yet.
    #[test]
    fn a_host_that_takes_no_target_dispatches_to_the_picks_first() {
        let (door, heard) = bed();
        let store = door.store();
        let now = Instant::now();
        let mut handle = held(&door, b"r-1", now);
        let pick = handle.decision().pick();
        let first = pick.and_then(|pick| pick.targets.first().copied());
        let first = first.expect("the pick's first target");
        handle.dispatched(now + Duration::from_secs(3_600));
        assert_eq!(counted(store), [(first.id, 1)]);
        thread::sleep(Duration::from_millis(1));
        drop(handle);
        assert_eq!(counted(store), []);
        let told = [
            Call::Issued,
            Call::Dispatched(first.id, 0, 3_600_000_000),
            Call::Completed(first.id, TransportClass::Cancelled, 0),
        ];
        assert_eq!(*heard.lock(), told);
    }

    /// A handle's report holds no thread for a decision. While one
    /// decision stands, more handles than a file's ring takes are let go:
    /// every one returns, and their leases stay on file, the deciding
    /// thread's to end. As that caller leaves, every lease is ended.
    #[test]
    fn handles_let_go_at_a_full_ring_hold_no_thread_for_a_decision() {
        let store = fleet(1);
        let door = Arc::new(Door::new(Arc::clone(&store)));
        let more = MemoryStore::HANDOVER + 8;
        let now = Instant::now();
        let asks = (0..more).map(|n| {
            let id = n.to_le_bytes();
            let asked = request(&id, 7);
            let decided = Handle::decide(Arc::clone(&door), GATEWAY, &asked, now);
            decided.expect("a decision")
        });
        let handles: Vec<_> = asks.collect();
        let ends = Ends::over(&store);
        let held = deciding(&door);
        ends.stands("pick");
        let (done, returned) = mpsc::channel();
        thread::spawn(move || {
            drop(handles);
            done.send(()).ok();
        });
        let returned = returned.recv_timeout(PATIENCE).is_ok();
        let on_file = LeasesStore::live(&*store, GATEWAY);
        ends.go();
        ends.stands("issue");
        ends.go();
        let lease = held.join().expect("the held decision's thread");
        assert!(returned, "every handle was let go while the decision stood");
        assert_eq!(on_file, more, "a report is the deciding thread's to apply");
        assert!(lease.is_some(), "the held request has its lease");
        assert_eq!(LeasesStore::live(&*store, GATEWAY), 1, "its lease alone");
    }

    /// A handle that owns its door travels with its request: dropped on
    /// another thread behind its dispatch, it ends the lease from there.
    #[test]
    fn a_handle_that_owns_its_door_ends_its_lease_from_another_thread() {
        let (door, heard) = bed();
        let door = Arc::new(door);
        let now = Instant::now();
        let asked = request(b"r-1", 7);
        let decided = Handle::decide(Arc::clone(&door), GATEWAY, &asked, now);
        let mut handle = decided.expect("a decision");
        let first = handle.take(0).expect("the pick's first target");
        handle.dispatched(now);
        let ended = thread::spawn(move || drop(handle));
        ended.join().expect("the request's thread");
        assert_eq!(counted(door.store()), []);
        let told = [
            Call::Issued,
            Call::Dispatched(first.id, 0, 0),
            Call::Completed(first.id, TransportClass::Cancelled, 0),
        ];
        assert_eq!(clocked(&heard, 0), told);
    }

    /// Sixty-four hosts at once on one model, each holding its requests
    /// from the decision to the end, by turns served, dropped behind the
    /// dispatch, refused at the commit and dropped before any dispatch:
    /// every lease is ended once, by its own path's word, none is left on
    /// file and no count is left on a member.
    #[test]
    fn sixty_four_hosts_at_once_end_every_lease_once() {
        let (door, heard) = bed();
        let (hosts, each) = (64_u8, 32_u8);
        let start = Barrier::new(usize::from(hosts));
        thread::scope(|scope| {
            for host in 0..hosts {
                let (door, start) = (&door, &start);
                scope.spawn(move || {
                    start.wait();
                    for turn in 0..each {
                        let now = Instant::now();
                        let mut handle = held(door, &[host, turn], now);
                        match turn % 4 {
                            0 => {
                                let first = handle.take(0).expect("the first target");
                                handle.dispatched(now);
                                handle.completed(&outcome(first.id, TransportClass::Ok), now);
                            }
                            1 => handle.dispatched(now),
                            2 => handle.released(ReleaseReason::CommitFailed, Instant::now()),
                            _ => {}
                        }
                    }
                });
            }
        });
        let store = door.store();
        assert_eq!(counted(store), []);
        assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
        let heard = heard.lock();
        let told = |want: fn(&Call) -> bool| heard.iter().filter(|call| want(call)).count();
        let quarter = usize::from(hosts) * usize::from(each) / 4;
        let ok = |call: &Call| matches!(call, Call::Completed(_, TransportClass::Ok, _));
        let cancelled =
            |call: &Call| matches!(call, Call::Completed(_, TransportClass::Cancelled, _));
        let refused = |call: &Call| matches!(call, Call::Released(ReleaseReason::CommitFailed));
        let dropped = |call: &Call| matches!(call, Call::Released(ReleaseReason::Abandoned));
        assert_eq!(told(|call| matches!(call, Call::Issued)), 4 * quarter);
        assert_eq!(
            told(|call| matches!(call, Call::Dispatched(..))),
            2 * quarter
        );
        assert_eq!(
            (told(ok), told(cancelled), told(refused), told(dropped)),
            (quarter, quarter, quarter, quarter)
        );
    }

    /// A completion is written on the worker its host names, which need
    /// not be the target the handle took: the health precedence reads the
    /// worker that served.
    #[test]
    fn a_completion_is_written_on_the_worker_its_host_names() {
        let (door, _) = bed();
        let store = door.store();
        let now = Instant::now();
        let mut handle = held(&door, b"r-1", now);
        let other = handle.take(1).expect("the pick's second target");
        let taken = handle.take(0).expect("the pick's first target");
        handle.dispatched(now);
        handle.completed(&outcome(other.id, TransportClass::Ok), now);
        assert!(store.completed_ok_since(GATEWAY, other.id, now));
        assert!(!store.completed_ok_since(GATEWAY, taken.id, now));
    }

    /// A completion is written at the instant its host read: a host whose
    /// instants run ahead of the clock writes there, as its dispatch did.
    #[test]
    fn a_completion_is_written_at_its_hosts_instant() {
        let (door, _) = bed();
        let store = door.store();
        let now = Instant::now();
        let mut handle = held(&door, b"r-1", now);
        let first = handle.take(0).expect("the pick's first target");
        let ahead = now + Duration::from_secs(2);
        handle.completed(&outcome(first.id, TransportClass::Ok), ahead);
        assert!(store.completed_ok_since(GATEWAY, first.id, ahead));
    }

    /// A release behind a dispatch spans its host's two instants, the
    /// dispatch's and its own: no clock is read.
    #[test]
    fn a_release_behind_a_dispatch_spans_its_hosts_instants() {
        let (door, heard) = bed();
        let now = Instant::now();
        let mut handle = held(&door, b"r-1", now);
        let first = handle.take(0).expect("the pick's first target");
        let sent = now + Duration::from_secs(3_600);
        handle.dispatched(sent);
        handle.released(ReleaseReason::ClientGone, sent + Duration::from_millis(7));
        let told = [
            Call::Issued,
            Call::Dispatched(first.id, 0, 3_600_000_000),
            Call::Completed(first.id, TransportClass::Cancelled, 7_000),
        ];
        assert_eq!(*heard.lock(), told);
    }

    thread_local! {
        static WORDS: RefCell<Vec<Said>> = const { RefCell::new(Vec::new()) };
    }

    fn words() -> Vec<Said> {
        WORDS.with_borrow_mut(mem::take)
    }

    fn pair_door(prefills: usize, decodes: usize) -> Result<Door, Box<dyn Error>> {
        let store = pair_fleet(prefills, decodes)?;
        store.tell_through(|_, _, _, said, _| WORDS.with_borrow_mut(|words| words.push(said)));
        Ok(Door::new(store))
    }

    fn paired(door: &Door, now: Instant) -> Result<Handle<&Door>, Refused> {
        let mut asked = request(b"pair", 7);
        asked.facts.phase = Phase::Prefill;
        Handle::decide(door, GATEWAY, &asked, now)
    }

    fn states(store: &MemoryStore, lease: LeaseId) -> Option<[LeaseState; 2]> {
        let filed = store.sides(GATEWAY, lease)?;
        Some([filed.record.state, filed.decode?.state])
    }

    #[test]
    fn each_side_of_a_pair_is_dispatched_and_ended_by_its_own_word() {
        for explicit in [true, false] {
            let door = pair_door(1, 2).expect("a pair door");
            let store = door.store();
            let now = Instant::now();
            let mut handle = paired(&door, now).expect("a pair's handle");
            let lease = handle.lease().expect("the pair's lease").lease.id;
            let first = handle.take(0).expect("the first target");
            let second = handle.take(1).expect("the alternate target");
            let d0 = first.pair.expect("the first decode").decode_id;
            let d1 = second.pair.expect("the alternate decode").decode_id;
            assert_ne!(d0, d1);
            let workers = [first.id, d0, d1];
            let (prefill_at, decode_at) = (
                now + Duration::from_micros(100),
                now + Duration::from_micros(200),
            );
            let prefill = LeaseState::Dispatched { at: prefill_at };
            let decode = LeaseState::Dispatched { at: decode_at };
            let issued = LeaseState::ISSUED;
            assert_eq!(states(store, lease), Some([issued, issued]));
            assert_eq!(pair_counts(store, workers), [1, 1, 0]);

            handle.dispatched(prefill_at);
            assert_eq!(states(store, lease), Some([prefill, issued]));
            assert_eq!(pair_counts(store, workers), [1, 1, 0]);
            if explicit {
                handle.dispatched_decode(decode_at);
                handle.dispatched_decode(decode_at);
                assert_eq!(states(store, lease), Some([prefill, decode]));
                assert_eq!(pair_counts(store, workers), [1, 0, 1]);
            }

            let decoded = Outcome {
                side: Side::Decode,
                ..outcome(d1, TransportClass::Ok)
            };
            handle.completed(&decoded, decode_at);
            handle.completed(&decoded, decode_at);
            assert_eq!(states(store, lease), Some([prefill, LeaseState::Completed]));
            assert_eq!(pair_counts(store, workers), [1, 0, 0]);
            let prefilled = Outcome {
                side: Side::Prefill,
                ..outcome(first.id, TransportClass::Ok)
            };
            let said = if explicit {
                prefilled
            } else {
                outcome(first.id, TransportClass::Ok)
            };
            handle.completed(&said, decode_at);
            drop(handle);
            assert!(store.sides(GATEWAY, lease).is_none());
            assert_eq!(pair_counts(store, workers), [0, 0, 0]);
            assert_eq!(
                words(),
                [
                    Said::Dispatched(
                        Dispatch {
                            target_index: 1,
                            side: Side::Prefill,
                            since_decision_micros: 100,
                        },
                        prefill_at
                    ),
                    Said::Dispatched(
                        Dispatch {
                            target_index: 1,
                            side: Side::Decode,
                            since_decision_micros: 200,
                        },
                        decode_at
                    ),
                    Said::Completed(decoded),
                    Said::Completed(prefilled),
                ]
            );
        }
    }

    #[test]
    fn a_decode_taken_after_the_prefill_went_out_moves_the_decode_alone() {
        let door = pair_door(2, 2).expect("a pair door");
        let store = door.store();
        let now = Instant::now();
        let mut handle = paired(&door, now).expect("a pair's handle");
        let lease = handle.lease().expect("the pair's lease").lease.id;
        let first = handle.take(0).expect("the first target");
        let alternate = handle.take(1).expect("the alternate target");
        assert_eq!(handle.take(0), Some(first));
        assert_eq!(
            handle.take_decode(1),
            None,
            "the prefill still owes a dispatch"
        );
        let other = ["http://p0:8000", "http://p1:8000"]
            .into_iter()
            .filter_map(|key| store.by_key(key))
            .find(|worker| worker.id != first.id)
            .expect("another prefill");
        // The picker lists one prefill. Add a target to exercise the handle's check.
        let Decision::Pick(pick) = &mut handle.decided.decision else {
            panic!("a pair pick");
        };
        let different = pick.targets.as_slice().len();
        pick.targets.push(Target {
            id: other.id,
            ..alternate
        });

        handle.dispatched(now);
        assert_eq!(handle.take(1), None, "the prefill has dispatched");
        assert_eq!(handle.take_decode(different), None);
        assert_eq!(handle.take_decode(usize::MAX), None);
        assert_eq!(handle.take_decode(1), Some(alternate));
        let d0 = first.pair.expect("the first decode").decode_id;
        let d1 = alternate.pair.expect("the alternate decode").decode_id;
        let workers = [first.id, other.id, d0, d1];
        assert_eq!(
            pair_counts(store, workers),
            [1, 0, 1, 0],
            "a take books nothing"
        );
        handle.dispatched_decode(now);
        assert_eq!(handle.take_decode(0), None);
        assert_eq!(pair_counts(store, workers), [1, 0, 0, 1]);
        let filed = store.sides(GATEWAY, lease).expect("both sides on file");
        let decode = filed.decode.expect("the decode side");
        assert_eq!(filed.record.target, first.id);
        assert_eq!((decode.target, decode.target_index), (d1, 1));
        drop(handle);
        let _ = words();
    }

    #[test]
    fn a_pair_given_up_before_any_send_ends_both_sides_in_one_word() {
        let door = pair_door(1, 1).expect("a pair door");
        let store = door.store();
        let now = Instant::now();
        let mut handle = paired(&door, now).expect("a pair's handle");
        let first = handle.take(0).expect("the first target");
        let decode = first.pair.expect("the first decode").decode_id;
        let reason = ReleaseReason::CommitFailed;
        handle.released(reason, now);
        assert_eq!(pair_counts(store, [first.id, decode]), [0, 0]);
        assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
        drop(handle);
        assert_eq!(words(), [Said::Released(None, reason)]);
    }

    #[test]
    fn a_pair_dropped_with_one_side_sent_completes_it_and_gives_the_other_up() {
        for sent in [Side::Prefill, Side::Decode] {
            let door = pair_door(1, 2).expect("a pair door");
            let store = door.store();
            let now = Instant::now();
            let mut handle = paired(&door, now).expect("a pair's handle");
            let taken = handle.take(1).expect("the alternate target");
            let decode = taken.pair.expect("the alternate decode").decode_id;
            let (worker, waiting) = if sent == Side::Prefill {
                handle.dispatched(now);
                (taken.id, Side::Decode)
            } else {
                handle.dispatched_decode(now);
                assert!(handle.take(0).is_some());
                (decode, Side::Prefill)
            };
            assert_eq!(words().len(), 1);
            drop(handle);
            let [first, second]: [Said; 2] = words().try_into().expect("two reports");
            let (completed, released) = if sent == Side::Prefill {
                (first, second)
            } else {
                (second, first)
            };
            let Said::Completed(cancelled) = completed else {
                panic!("the sent side completed");
            };
            assert_eq!(
                (cancelled.side, cancelled.served, cancelled.class),
                (sent, worker, TransportClass::Cancelled)
            );
            assert_eq!(
                released,
                Said::Released(Some(waiting), ReleaseReason::Abandoned)
            );
            assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
            assert_eq!(pair_counts(store, [taken.id, decode]), [0, 0]);

            let mut handle = paired(&door, now).expect("another pair's handle");
            let lease = handle.lease().expect("the pair's lease").lease.id;
            let first = handle.take(0).expect("the first target");
            let decode = first.pair.expect("the first decode").decode_id;
            let reason = ReleaseReason::ClientGone;
            let (state, report) = if sent == Side::Decode {
                handle.dispatched_decode(now);
                assert_eq!(words().len(), 1);
                let cancelled = Outcome {
                    side: Side::Decode,
                    served: decode,
                    class: TransportClass::Cancelled,
                    status: 0,
                    ttft_micros: 0,
                    tokens_in: 0,
                    tokens_out: 0,
                    duration_micros: 0,
                };
                (LeaseState::Completed, Said::Completed(cancelled))
            } else {
                (
                    LeaseState::Released(reason),
                    Said::Released(Some(Side::Decode), reason),
                )
            };
            handle.released_decode(reason, now);
            handle.released_decode(reason, now);
            assert_eq!(states(store, lease), Some([LeaseState::ISSUED, state]));
            assert_eq!(pair_counts(store, [first.id, decode]), [1, 0]);
            assert_eq!(words(), [report]);
            drop(handle);
            assert_eq!(
                words(),
                [Said::Released(
                    Some(Side::Prefill),
                    ReleaseReason::Abandoned
                )]
            );
        }
    }

    #[test]
    fn a_single_handle_says_nothing_of_a_decode_side() {
        let (door, heard) = bed();
        let store = door.store();
        let now = Instant::now();
        let mut handle = held(&door, b"single", now);
        let lease = handle.lease().expect("the single's lease").lease.id;
        let first = handle.take(0).expect("the first target");
        handle.dispatched_decode(now);
        assert_eq!(handle.take_decode(1), None);
        handle.released_decode(ReleaseReason::ClientGone, now);
        let decoded = Outcome {
            side: Side::Decode,
            ..outcome(first.id, TransportClass::Ok)
        };
        handle.completed(&decoded, now);
        let state = LeasesStore::get(&**store, GATEWAY, lease).map(|filed| filed.state);
        assert_eq!(state, Some(LeaseState::ISSUED));
        assert_eq!(counted(store), [(first.id, 1)]);
        assert_eq!(*heard.lock(), [Call::Issued]);
    }
}
