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
    /// A completion: the lease was dispatched at `sent`.
    Completion {
        lease: LeaseId,
        side: Side,
        sent: Instant,
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
    /// The target the host took, by its index among the pick's: the
    /// first, until the host takes another.
    index: u8,
    /// That target: the worker a completion the host does not say is
    /// written on.
    target: Target,
    owed: Owed,
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
        Ok(Self {
            door,
            gateway,
            decided,
            index: 0,
            target,
            owed,
        })
    }

    /// The engine's decision.
    #[must_use]
    pub fn decision(&self) -> &Decision {
        &self.decided.decision
    }

    /// The lease the door filed for a pick; `None` for a decision that
    /// picks no worker.
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
    pub fn take(&mut self, index: usize) -> Option<Target> {
        let Owed::Dispatch { .. } = self.owed else {
            return None;
        };
        let taken = u8::try_from(index).ok()?;
        let pick = self.decided.decision.pick()?;
        let target = *pick.targets.as_slice().get(index)?;
        (self.index, self.target) = (taken, target);
        Some(target)
    }

    /// Says through the door what the host says of `lease`: the calling
    /// thread is never held for a decision ([`Door::report`]).
    fn say(&self, lease: LeaseId, said: Said) {
        self.door.report(self.gateway, lease, said);
    }

    /// The host sent the request to the target it took, at `now`: the
    /// lease is dispatched on that target, and no dispatch deadline ends
    /// it from here. A dispatch said after that deadline meets no lease:
    /// the request runs unbooked, and its host is told nothing. Said once:
    /// a second call says nothing.
    pub fn dispatched(&mut self, now: Instant) {
        let Owed::Dispatch {
            lease,
            side,
            decided,
        } = self.owed
        else {
            return;
        };
        let dispatch = Dispatch {
            target_index: self.index,
            side,
            since_decision_micros: micros(now.saturating_duration_since(decided)),
        };
        self.say(lease, Said::Dispatched(dispatch, now));
        self.owed = Owed::Completion {
            lease,
            side,
            sent: now,
        };
    }

    /// The request ended with `outcome`, as its host read it at `now`: the
    /// lease ends, and the completion is written on the served worker for
    /// the health precedence, as a served controller writes a gateway's.
    /// A completion said with no dispatch says the dispatch first, at
    /// `now`: a request whose send failed was sent, and the engine hears a
    /// lease's events in their order. That dispatch carries the
    /// completion's instant, so a stage that times a request from its
    /// dispatch reads a span of zero for it. Said once.
    pub fn completed(&mut self, outcome: &Outcome, now: Instant) {
        self.dispatched(now);
        let Owed::Completion { lease, .. } = self.owed else {
            return;
        };
        door::completed(self.door.store(), self.gateway, outcome, now, |said| {
            self.say(lease, said);
        });
        self.owed = Owed::Nothing;
    }

    /// The host gives the lease up at `now`, under `reason`. Before any
    /// dispatch its commit refused every target, or its client went away.
    /// After a dispatch a lease lives until its completion: a release said
    /// then is told as a completion of class cancelled, on the target the
    /// dispatch named, with no status and no tokens, since none were read.
    /// Its span runs from the dispatch's instant to `now`, both the host's.
    /// Said once.
    pub fn released(&mut self, reason: ReleaseReason, now: Instant) {
        match self.owed {
            Owed::Nothing => {}
            Owed::Dispatch { lease, .. } => {
                self.say(lease, Said::Released(None, reason));
                self.owed = Owed::Nothing;
            }
            Owed::Completion { side, sent, .. } => {
                let cancelled = Outcome {
                    side,
                    served: self.target.id,
                    status: 0,
                    class: TransportClass::Cancelled,
                    ttft_micros: 0,
                    tokens_in: 0,
                    tokens_out: 0,
                    duration_micros: micros(now.saturating_duration_since(sent)),
                };
                self.completed(&cancelled, now);
            }
        }
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
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::{
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
            tests::{counted, fleet, held as deciding, picking, request, Ends, GATEWAY, PATIENCE},
            Door,
        },
        framework::{Picker, TieBand},
        lease::{Lease, Lifecycle, Outcome, ReleaseReason, Side, TransportClass},
        store::{HealthSink, LeaseState, LeasesStore, MemoryStore},
        view::{Candidate, RequestFacts, WorkerId},
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
}
