//! The lifecycle frames of a `Session` (design 4.3 and 5.2): `Dispatched`,
//! `Completed` and `Released`, a gateway's report of each request it sent
//! to a worker.
//!
//! The gateway reports every request on every worker, so no arm asks a
//! worker's row what else reports for it. The gateway is the stream's
//! principal: no field of a frame names one, and a lease is keyed by its
//! gateway, so a gateway reaches no lease of another.
//!
//! A frame is read whole before its first write. One that reads wrong is
//! refused by its field's name and writes nothing. A lease the store does
//! not hold is no error: its dispatch deadline may have ended it, or the
//! gateway decided that request itself. Nor is a ticket: this build
//! issues none, so the one a `Released` names is one it does not hold. A
//! completion is written for a worker the store holds a row for and for
//! no other, under the stream's own gateway, and a gateway's completions
//! end with its presence: the health table grows with the rows and with
//! who is present, never with what a frame names.
//!
//! What a frame says of a lease is handed to the leases' file, never
//! written by its session ([`MemoryStore::hand`]): whoever holds the
//! file's turn next applies it, a decider at work or the session itself
//! where none is, in the order the frames were read. A `Decide` stream
//! hands the release of a decision it stops at its send through a hand
//! of its own ([`open`]).
//!
//! A frame whose lease is on file tells the engine that issued it, the
//! generation's own, what became of the decision: its dispatch, its
//! completion or its release. A generation that no longer stands is told
//! nothing: the stages of a later one never saw the issue.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    time::Instant,
};

use tonic::Status;
use tracing::error;

use crate::{
    decider::Decider,
    decision::Target,
    framework::{release as release_payload, Engine},
    lease::{Lease, LeaseId, Lifecycle, Outcome, ReleaseReason, Side, TransportClass},
    proto::{
        self,
        completed::Class,
        released::{Reason, What},
    },
    sessions::{Forgets, SessionVerdict},
    store::{
        Dispatch, GatewayId, Hand, HealthSink, LeaseRecord, MemoryStore, Said, SessionsStore, World,
    },
    view::{Session, WorkerId},
};

/// A `Dispatched` frame: the gateway's lease moves to dispatched at `now`.
///
/// # Errors
///
/// The frame names no side, or a target index past a byte.
pub(crate) fn dispatched(
    store: &MemoryStore,
    hand: &Hand,
    frame: &proto::Dispatched,
    now: Instant,
) -> Result<(), Status> {
    let (lease, dispatch) = dispatch(frame)?;
    store.hand(hand, lease, Said::Dispatched(dispatch, now));
    Ok(())
}

/// A `Completed` frame: the lease's side ends with its outcome, and the
/// gateway's completion on the worker is written for the health
/// precedence, whether or not the lease is on file.
///
/// # Errors
///
/// The frame names no side or no class, or a status past sixteen bits.
pub(crate) fn completed(
    store: &MemoryStore,
    hand: &Hand,
    frame: &proto::Completed,
    now: Instant,
) -> Result<(), Status> {
    let (lease, outcome) = outcome(frame)?;
    store.hand(hand, lease, Said::Completed(outcome));
    // The sink writes for a worker it holds a row for and for no other.
    HealthSink::completed(store, hand.gateway(), outcome.served, outcome.class, now);
    Ok(())
}

/// A `Released` frame: the gateway gives its lease up, on one side or on
/// every side. A ticket is no error and writes nothing: a ticket this
/// build never issued is a ticket not on file.
///
/// # Errors
///
/// The frame names neither a lease nor a ticket, no reason, or a side
/// this version does not know.
pub(crate) fn released(
    store: &MemoryStore,
    hand: &Hand,
    frame: &proto::Released,
) -> Result<(), Status> {
    let (lease, side, reason) = release(frame)?;
    if let Some(lease) = lease {
        store.hand(hand, lease, Said::Released(side, reason));
    }
    Ok(())
}

/// Opens a hand at the leases' files, a session's or a `Decide` stream's:
/// what it says of `gateway`'s leases is handed through it, and the store
/// is given the word to the engine that whoever applies it tells with
/// ([`tells`]).
pub(crate) fn open(store: &MemoryStore, gateway: GatewayId) -> Hand {
    tells(store);
    Hand::new(gateway)
}

/// Gives the store the word to the engine of a frame applied to a lease.
pub(crate) fn tells(store: &MemoryStore) {
    store.tell_through(told);
}

/// Tells the engine what a frame said of a lease: one function of the
/// record found on file and the frame, whoever calls it.
///
/// The engine told is the one of the lease's own generation, where that
/// generation still stands: a later generation's stages never saw the
/// issue and hear nothing of it. A dispatch is told on the target its
/// frame names by index, as the record holds the pick, and the lease told
/// names that index: the record was read before the door wrote it. A
/// frame that names no target of the pick tells nothing.
///
/// Its caller is whoever applies the frame, under the turn of the leases'
/// file: what it calls is bounded and takes no door of the leases. It
/// enters as the decider whose file the lease is on, on whichever thread
/// applies.
///
/// A stage's call that panics is held here, whoever applies: a decider
/// inside a decision or as it takes up or leaves its work, a session, the
/// gateways' sweep. The lease's event stands on file as applied, the
/// fault is counted on the store ([`MemoryStore::lifecycle_faults`]) and
/// a streak of faults writes one line. It costs no request its decision
/// and ends no thread.
fn told(store: &MemoryStore, world: &World, record: &LeaseRecord, said: Said, by: Option<Decider>) {
    // The lease's own decider acts while its event is told, whichever
    // thread applies the frame.
    let _acting = by.map(Decider::enter);
    let Err(payload) = catch_unwind(AssertUnwindSafe(|| tell(store, world, record, said))) else {
        store.call_returned();
        return;
    };
    // A payload's own drop can panic: the engine's helper frees it under a
    // catch of its own.
    release_payload(payload);
    if store.call_faulted() {
        error!(
            lease = record.lease.id.0,
            "a lease's lifecycle call panicked: held, and the lease's event stands as applied"
        );
    }
}

/// What [`told`] tells, under its catch.
fn tell(store: &MemoryStore, world: &World, record: &LeaseRecord, said: Said) {
    let spec = world.spec();
    let engine = (spec.generation() == record.lease.generation).then(|| spec.engine());
    match (said, engine) {
        (Said::Dispatched(dispatch, at), engine) => {
            let Some((id, dp_rank)) = record.target_at(dispatch.target_index) else {
                return;
            };
            let target = Target {
                id,
                dp_rank,
                ..Target::default()
            };
            // The record was read before the door wrote the dispatched
            // index: the lease told carries it, for a stage that reads
            // the index off the lease.
            let lease = Lease {
                target_index: dispatch.target_index,
                ..record.lease
            };
            // The table hears first and the pin of record reads it.
            // The stages hear last, so nothing is held across their
            // loop: a value kept for the pin behind it cost every
            // lease a store and a reload a profile.
            if let Some(engine) = engine {
                engine.dispatched_key(&lease, &target, dispatch.side);
            }
            pinned(store, record, engine, id, at);
            if let Some(engine) = engine {
                engine.dispatched_stages(
                    &lease,
                    &target,
                    dispatch.side,
                    dispatch.since_decision_micros,
                );
            }
        }
        (Said::Completed(outcome), Some(engine)) => {
            engine.on_completed(&record.lease, &outcome);
        }
        (Said::Released(_, reason), Some(engine)) => engine.on_released(&record.lease, reason),
        (Said::Completed(_) | Said::Released(..), None) => {}
    }
}

/// Writes a keyed lease's pin of record at its dispatch (design 5.1, the
/// sessions row: written at the commit, never at selection). The pin names
/// the worker the key's rule prefers once the engine was told, as its
/// table holds it: a dispatch on the failover leaves the preferred worker
/// pinned. Where no engine was told it names the dispatched worker. A
/// lease no sessions rule governed pins nothing.
///
/// The pin ends where the engine's table forgets the key, and never under
/// a rule without a TTL. Where the table holds no such key, or no engine
/// was told, the lease's own rule ends it: the TTL its decision was made
/// under, as its record on file carries it, counted from the lease's
/// issue as the table counted it then. So it does where the dispatch
/// itself made the table's entry anew and no decision has read it: that
/// entry holds no instant yet. An end the lease's rule counts never moves
/// the pin's end earlier: a lease dispatched behind a newer one of its
/// key, or behind a dispatch whose table ended the pin later, leaves the
/// later end standing. Either end stands on the decision clock: a
/// dispatch converts no clock. The table's lock is let go before the
/// store's pin is written: no lock is held across the two.
fn pinned(
    store: &MemoryStore,
    record: &LeaseRecord,
    engine: Option<&Engine>,
    dispatched: WorkerId,
    at: Instant,
) {
    let lease = &record.lease;
    if lease.session == Session::NONE || lease.session_verdict == SessionVerdict::None {
        return;
    }
    let held = engine.and_then(|engine| engine.sessions().held(lease.session));
    let preferred = held
        .map(|(pins, _)| pins[0])
        .filter(|pin| *pin != WorkerId(0));
    let (key, worker) = (lease.session.get(), preferred.unwrap_or(dispatched));
    match held {
        // The table's word is the key's rule as it stands: it moves the
        // pin's end either way.
        Some((_, Forgets::At(forgotten))) => {
            SessionsStore::pin(store, key, worker, at, Some(forgotten));
        }
        Some((_, Forgets::Never)) => SessionsStore::pin(store, key, worker, at, None),
        // The lease's own rule is the word of a decision that may be older
        // than the pin's last writer: it never moves the end earlier.
        Some((_, Forgets::Unread)) | None => {
            let ends = record
                .session_ttl_micros
                .map(|ttl| lease.issued_micros.saturating_add(ttl));
            SessionsStore::pin_no_earlier(store, key, worker, at, ends);
        }
    }
}

/// What a `Dispatched` frame says, read whole.
fn dispatch(frame: &proto::Dispatched) -> Result<(LeaseId, Dispatch), Status> {
    Ok((
        LeaseId(frame.lease_id),
        Dispatch {
            target_index: u8::try_from(frame.target_index).map_err(|_| {
                past(
                    "Dispatched.target_index",
                    frame.target_index,
                    u8::MAX.into(),
                )
            })?,
            side: side("Dispatched.side", frame.side)?,
            since_decision_micros: frame.since_decision_micros,
        },
    ))
}

/// What a `Completed` frame says, read whole.
fn outcome(frame: &proto::Completed) -> Result<(LeaseId, Outcome), Status> {
    Ok((
        LeaseId(frame.lease_id),
        Outcome {
            side: side("Completed.side", frame.side)?,
            served: WorkerId(frame.served),
            status: u16::try_from(frame.status)
                .map_err(|_| past("Completed.status", frame.status, u16::MAX.into()))?,
            class: class(frame.class)?,
            ttft_micros: frame.ttft_micros,
            tokens_in: frame.tokens_in,
            tokens_out: frame.tokens_out,
            duration_micros: frame.duration_micros,
        },
    ))
}

/// What a `Released` frame says, read whole: the lease (`None` is a
/// ticket), the side given up (`None` is the frame's unspecified side,
/// every side) and the reason.
fn release(
    frame: &proto::Released,
) -> Result<(Option<LeaseId>, Option<Side>, ReleaseReason), Status> {
    let lease = match frame.what {
        Some(What::LeaseId(lease)) => Some(LeaseId(lease)),
        Some(What::Ticket(_)) => None,
        None => {
            return Err(Status::invalid_argument(
                "Released.what names neither a lease nor a ticket",
            ));
        }
    };
    let side = if frame.side == i32::from(proto::Side::Unspecified) {
        None
    } else {
        Some(side("Released.side", frame.side)?)
    };
    Ok((lease, side, reason(frame.reason)?))
}

/// The side a frame names.
fn side(field: &'static str, number: i32) -> Result<Side, Status> {
    match proto::Side::try_from(number) {
        Ok(proto::Side::Single) => Ok(Side::Single),
        Ok(proto::Side::Prefill) => Ok(Side::Prefill),
        Ok(proto::Side::Decode) => Ok(Side::Decode),
        Ok(proto::Side::Encode) => Ok(Side::Encode),
        Ok(proto::Side::Unspecified) | Err(_) => Err(none(field, number)),
    }
}

/// The transport class a `Completed` names.
fn class(number: i32) -> Result<TransportClass, Status> {
    match Class::try_from(number) {
        Ok(Class::Ok) => Ok(TransportClass::Ok),
        Ok(Class::ClientError) => Ok(TransportClass::ClientError),
        Ok(Class::ServerError) => Ok(TransportClass::ServerError),
        Ok(Class::ConnectFailed) => Ok(TransportClass::ConnectFailed),
        Ok(Class::Timeout) => Ok(TransportClass::Timeout),
        Ok(Class::Cancelled) => Ok(TransportClass::Cancelled),
        Ok(Class::Unspecified) | Err(_) => Err(none("Completed.class", number)),
    }
}

/// The reason a `Released` names.
fn reason(number: i32) -> Result<ReleaseReason, Status> {
    match Reason::try_from(number) {
        Ok(Reason::Abandoned) => Ok(ReleaseReason::Abandoned),
        Ok(Reason::LateDecision) => Ok(ReleaseReason::LateDecision),
        Ok(Reason::ClientGone) => Ok(ReleaseReason::ClientGone),
        Ok(Reason::CommitFailed) => Ok(ReleaseReason::CommitFailed),
        Ok(Reason::Expired) => Ok(ReleaseReason::Expired),
        Ok(Reason::WorkerRemoved) => Ok(ReleaseReason::WorkerRemoved),
        Ok(Reason::Unspecified) | Err(_) => Err(none("Released.reason", number)),
    }
}

/// The refusal of a field whose number names none of its kind in this
/// version.
fn none(field: &'static str, number: i32) -> Status {
    Status::invalid_argument(format!(
        "{field} is {number}, which names none this version knows"
    ))
}

/// The refusal of a number past its field's range.
fn past(field: &'static str, number: u32, most: u32) -> Status {
    Status::invalid_argument(format!("{field} is {number}: {most} is the most it holds"))
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };

    use parking_lot::Mutex;
    use tonic::{Code, Status};

    use super::{
        class, completed, dispatch, dispatched, open, outcome, reason, release, released, side,
    };
    use crate::{
        builtins::Random,
        catalog::Catalog,
        controller::gateways::tests::lease,
        decision::{Target, Targets},
        framework::{Picker, TieBand},
        lease::{Lease, LeaseId, Lifecycle, Outcome, ReleaseReason, Side, TransportClass},
        proto::{
            self,
            completed::Class,
            released::{Reason, What},
        },
        sessions::{SessionRule, SessionVerdict},
        spec::{AppliedSpec, Assignment, PolicySpec, Recovery},
        store::{
            Dispatch, GatewayId, HealthSink, LeaseState, LeasesStore, MemoryStore, Picked,
            SessionsStore, WorkerFacts, WorkersStore,
        },
        view::{Candidate, Health, Phase, RequestFacts, Session, WorkerId},
        Constraints, PoolFacts, Scratch,
    };

    const A: GatewayId = GatewayId(11);
    const B: GatewayId = GatewayId(12);
    /// The session of the keyed leases.
    const KEY: Session = Session::from_raw(41);
    /// The lease whose release the hearing stage panics at.
    const BURSTS: u64 = 99;

    /// A worker the store holds a row for.
    pub(crate) fn worker(store: &MemoryStore, key: &str) -> WorkerId {
        WorkersStore::upsert(
            store,
            WorkerFacts {
                key: key.to_owned(),
                engine: key.to_owned(),
                incarnation: 1,
                models: vec!["m".to_owned()],
                dp_size: 1,
                ..WorkerFacts::default()
            },
        )
        .id()
    }

    pub(crate) fn a_dispatched(lease: u64) -> proto::Dispatched {
        proto::Dispatched {
            lease_id: lease,
            target_index: 0,
            side: proto::Side::Single.into(),
            since_decision_micros: 250,
        }
    }

    pub(crate) fn a_completed(lease: u64, served: WorkerId) -> proto::Completed {
        proto::Completed {
            lease_id: lease,
            side: proto::Side::Single.into(),
            served: served.0,
            status: 200,
            class: Class::Ok.into(),
            ttft_micros: 1_000,
            tokens_in: 12,
            tokens_out: 34,
            duration_micros: 5_000,
        }
    }

    pub(crate) fn a_released(lease: u64) -> proto::Released {
        proto::Released {
            what: Some(What::LeaseId(lease)),
            reason: Reason::ClientGone.into(),
            side: proto::Side::Unspecified.into(),
        }
    }

    fn state(store: &MemoryStore, gateway: GatewayId, lease: u64) -> Option<LeaseState> {
        LeasesStore::get(store, gateway, LeaseId(lease)).map(|record| record.state)
    }

    /// What the engine told a stage of leases: the call, the lease's id and
    /// the worker the call named.
    type Heard = Vec<(&'static str, u64, WorkerId)>;

    /// The index each dispatch's lease named and the rank of the target it
    /// was told on, as a stage reads them off what it is handed.
    type Indexes = Arc<Mutex<Vec<(u8, u16)>>>;

    /// The `random` picker, keeping what the engine tells it of leases.
    #[derive(Debug)]
    struct Hears(Arc<Mutex<Heard>>, Indexes);

    impl Lifecycle for Hears {
        fn on_dispatched(&self, lease: &Lease, target: &Target, _: Side, _: u32) {
            self.0.lock().push(("dispatched", lease.id.0, target.id));
            self.1.lock().push((lease.target_index, target.dp_rank));
        }

        fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
            self.0
                .lock()
                .push(("completed", lease.id.0, outcome.served));
        }

        fn on_released(&self, lease: &Lease, _: ReleaseReason) {
            self.0.lock().push(("released", lease.id.0, WorkerId(0)));
            assert_ne!(lease.id.0, BURSTS, "the stage's call panics");
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

    /// Applies `generation` of a document whose picker keeps what it hears,
    /// and hands back what that generation's stage heard.
    fn hearing(store: &MemoryStore, generation: u64) -> Arc<Mutex<Heard>> {
        hearing_indexes(store, generation).0
    }

    /// [`hearing`], with the index each dispatch's lease named and the rank
    /// it was told.
    fn hearing_indexes(store: &MemoryStore, generation: u64) -> (Arc<Mutex<Heard>>, Indexes) {
        let heard = Arc::new(Mutex::new(Vec::new()));
        let indexes = Indexes::default();
        let kept = (Arc::clone(&heard), Arc::clone(&indexes));
        let mut catalog = Catalog::builtin();
        catalog
            .picker("hears", move || {
                Box::new(Hears(Arc::clone(&kept.0), Arc::clone(&kept.1)))
            })
            .expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        document.generation = generation;
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = "hears".into();
        let digest = document.digest().expect("the document digests");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog)
            .expect("the catalog holds the stage");
        store.apply(spec).expect("a later generation");
        (heard, indexes)
    }

    /// Files the lease `id` of [`A`] as `generation` issued it.
    fn issued(store: &MemoryStore, id: u64, generation: u64, now: Instant) {
        let mut record = lease(A, id, now);
        record.lease.generation = generation;
        store.issue(record).expect("a lease");
    }

    #[test]
    fn a_frame_tells_the_engine_that_issued_its_lease() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let heard = hearing(&store, 1);
        let served = worker(&store, "http://a:8000");
        for id in 1..=2 {
            issued(&store, id, 1, t0);
        }
        dispatched(&store, &open(&store, A), &a_dispatched(1), t0).expect("the frame reads");
        // The dispatch names the worker the lease books.
        assert_eq!(*heard.lock(), [("dispatched", 1, WorkerId(7))]);
        completed(&store, &open(&store, A), &a_completed(1, served), t0).expect("the frame reads");
        released(&store, &open(&store, A), &a_released(2)).expect("the frame reads");
        assert_eq!(
            heard.lock()[1..],
            [("completed", 1, served), ("released", 2, WorkerId(0))]
        );
    }

    /// A stage's call that panics is held where the engine is told,
    /// whoever applies the frame: here the thread that hands it, as a
    /// session does with no decider at work. The lease left the file
    /// ahead of the call, the fault is counted, and the next frame is
    /// told.
    #[test]
    fn a_stages_call_that_panics_is_held_and_the_next_frame_is_told() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let heard = hearing(&store, 1);
        for id in [BURSTS, 2] {
            issued(&store, id, 1, t0);
        }
        let hand = open(&store, A);
        released(&store, &hand, &a_released(BURSTS)).expect("the frame reads");
        assert_eq!(state(&store, A, BURSTS), None);
        assert_eq!(store.lifecycle_faults(), 1);
        released(&store, &hand, &a_released(2)).expect("the frame reads");
        let told = [BURSTS, 2].map(|lease| ("released", lease, WorkerId(0)));
        assert_eq!(*heard.lock(), told);
        assert_eq!(store.lifecycle_faults(), 1);
        assert_eq!(store.live(A), 0);
    }

    #[test]
    fn a_lease_is_told_dispatched_once_and_nothing_once_it_left_the_file() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let heard = hearing(&store, 1);
        issued(&store, 1, 1, t0);
        for _ in 0..2 {
            dispatched(&store, &open(&store, A), &a_dispatched(1), t0).expect("the frame reads");
        }
        assert_eq!(heard.lock().len(), 1, "a second dispatch of one lease");
        // A lease that was never on file, and one that left it.
        dispatched(&store, &open(&store, A), &a_dispatched(2), t0).expect("no error");
        released(&store, &open(&store, A), &a_released(2)).expect("no error");
        released(&store, &open(&store, A), &a_released(1)).expect("the frame reads");
        released(&store, &open(&store, A), &a_released(1)).expect("no error");
        assert_eq!(heard.lock()[1..], [("released", 1, WorkerId(0))]);
    }

    #[test]
    fn a_lease_of_a_generation_that_left_tells_no_engine() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let first = hearing(&store, 1);
        issued(&store, 1, 1, t0);
        // A later generation's stages never saw the issue.
        let second = hearing(&store, 2);
        dispatched(&store, &open(&store, A), &a_dispatched(1), t0).expect("the frame reads");
        released(&store, &open(&store, A), &a_released(1)).expect("the frame reads");
        assert!(first.lock().is_empty() && second.lock().is_empty());
        assert_eq!(
            state(&store, A, 1),
            None,
            "the file is written all the same"
        );
    }

    #[test]
    fn a_dispatch_that_names_no_target_of_the_pick_is_not_told() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let heard = hearing(&store, 1);
        issued(&store, 1, 1, t0);
        let mut second_target = a_dispatched(1);
        second_target.target_index = 1;
        dispatched(&store, &open(&store, A), &second_target, t0).expect("the frame reads");
        assert!(heard.lock().is_empty());
        assert_eq!(state(&store, A, 1), Some(LeaseState::Dispatched { at: t0 }));
    }

    /// A pick of the workers 7 and 9, the second at rank 2.
    fn two_targets() -> Picked {
        let mut targets = Targets::new();
        for (id, dp_rank) in [(7, 0), (9, 2)] {
            targets.push(Target {
                id: WorkerId(id),
                dp_rank,
                ..Target::default()
            });
        }
        Picked::of(&targets)
    }

    #[test]
    fn a_dispatch_on_another_target_of_the_pick_is_told_on_that_target() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let (heard, indexes) = hearing_indexes(&store, 1);
        let session = Session::from_raw(41);
        let mut record = lease(A, 1, t0);
        record.lease.generation = 1;
        record.lease.session = session;
        record.lease.session_verdict = SessionVerdict::Claimed(WorkerId(7));
        record.picked = two_targets();
        store.issue(record).expect("a lease");
        let mut second_target = a_dispatched(1);
        second_target.target_index = 1;
        dispatched(&store, &open(&store, A), &second_target, t0).expect("the frame reads");
        assert_eq!(*heard.lock(), [("dispatched", 1, WorkerId(9))]);
        // The lease told names the dispatched index, which the door wrote
        // after the record was read: a stage that reads the index off the
        // lease is told the second target, at that target's rank.
        assert_eq!(*indexes.lock(), [(1, 2)]);
        // The key's pin follows the dispatch, and so does the lease.
        let pins = store.with_world(|world| world.spec().engine().sessions().pins(session));
        assert_eq!(pins, Some([WorkerId(9), WorkerId(0)]));
        let filed = LeasesStore::get(&store, A, LeaseId(1)).expect("on file");
        assert_eq!((filed.target, filed.lease.target_index), (WorkerId(9), 1));
        assert_eq!(filed.target_at(1), Some((WorkerId(9), 2)));
    }

    /// The worker the bucket stage leads a request of `tokens` with, beside
    /// the two workers of its view as a first pick ordered them, after a
    /// lease of that size was issued on that pick's first target and its
    /// gateway dispatched on the pick's second: the other worker, or
    /// `another_rank` of the first target's own.
    fn led_after_a_dispatch_on_the_second_target(
        tokens: u32,
        another_rank: bool,
    ) -> (WorkerId, [WorkerId; 2]) {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let mut document = PolicySpec::preset("bucket").expect("a preset of this build");
        document.generation = 1;
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        store.apply(spec).expect("a later generation");
        // Two workers of one model, as a view shows them, and a prefill
        // request decided by the standing engine.
        let view: Vec<Candidate> = [7, 9]
            .iter()
            .zip(0..)
            .map(|(&id, position)| Candidate {
                position,
                id: WorkerId(id),
                health: Health {
                    healthy: true,
                    ..Health::default()
                },
                ..Candidate::default()
            })
            .collect();
        let mut scratch = Scratch::new();
        let mut decide = |seed: u64| {
            let facts = RequestFacts {
                model: "m",
                phase: Phase::Prefill,
                prompt_tokens: tokens,
                seed,
                now_micros: 1_000_000,
                ..RequestFacts::default()
            };
            let pool = PoolFacts {
                pool_size: 2,
                generation: 1,
                ..PoolFacts::default()
            };
            store.with_world(|world| {
                let engine = world.spec().engine();
                engine.decide(&facts, &pool, &view, &Constraints::default(), &mut scratch)
            })
        };
        let decision = decide(1);
        let pick = decision.pick().expect("a pick");
        let [owner, other] = [0, 1].map(|at| pick.targets.as_slice()[at].id);
        assert_ne!(owner, other);
        // What the record keeps of the pick: its second target as the
        // engine listed it, or the first target's worker again at its
        // second rank, as a pick over a worker of two ranks lists it.
        let mut targets = pick.targets;
        if another_rank {
            targets = Targets::new();
            for dp_rank in 0..2 {
                targets.push(Target {
                    id: owner,
                    dp_rank,
                    ..Target::default()
                });
            }
        }
        // The lease is issued on the pick's first target, which it names as
        // the worker it books, and the engine that decided is told: the
        // stage books that worker.
        let mut record = lease(A, 1, t0);
        record.lease.generation = 1;
        record.lease.tokens = tokens;
        record.lease.issued_micros = 1_000_000;
        record.lease.booked = owner;
        record.target = owner;
        record.picked = Picked::of(&targets);
        let issued = record.lease;
        store.with_world(|world| {
            let engine = world.spec().engine();
            engine.on_decision_issued(&issued, &decision);
        });
        store.issue(record).expect("a lease");
        // Its gateway dispatches on the pick's second target.
        let mut second_target = a_dispatched(1);
        second_target.target_index = 1;
        dispatched(&store, &open(&store, A), &second_target, t0).expect("the frame reads");
        let next = decide(2);
        let led = next.pick().expect("a pick").targets.as_slice()[0].id;
        (led, [owner, other])
    }

    /// A stage that books at the issue books the worker that serves: the
    /// bucket stage hears a `Dispatched` frame that names its pick's second
    /// target and books the lease's tokens on that worker too.
    #[test]
    fn a_dispatch_on_another_target_is_booked_by_a_stage_that_books_at_the_issue() {
        // Both workers carry the lease's tokens: the next request of that
        // size goes to its range's owner again. With the first target alone
        // booked, the balance rule would hand it to the other.
        let (led, [owner, _]) = led_after_a_dispatch_on_the_second_target(3_000, false);
        assert_eq!(led, owner);
    }

    /// A dispatch that names the worker the issue booked books nothing
    /// more: the frame names another rank of the lease's own worker.
    #[test]
    fn a_dispatch_on_another_rank_of_the_leases_worker_is_booked_once() {
        // Booked once, the worker carries 50 tokens, under the balance
        // rule's 64: the next request of that size goes to its range's
        // owner again. Booked twice it would carry 100, and the rule would
        // hand the request to the other worker.
        let (led, [owner, _]) = led_after_a_dispatch_on_the_second_target(50, true);
        assert_eq!(led, owner);
    }

    #[test]
    fn a_dispatched_frame_books_a_keyed_leases_pin_in_its_engines_table() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let _heard = hearing(&store, 1);
        let session = Session::from_raw(41);
        let pins = || store.with_world(|world| world.spec().engine().sessions().pins(session));
        let mut record = lease(A, 1, t0);
        record.lease.generation = 1;
        record.lease.session = session;
        record.lease.session_verdict = SessionVerdict::Claimed(WorkerId(7));
        store.issue(record).expect("a lease");
        assert_eq!(pins(), None, "an issue books no pin");
        dispatched(&store, &open(&store, A), &a_dispatched(1), t0).expect("the frame reads");
        assert_eq!(pins(), Some([WorkerId(7), WorkerId(0)]));
    }

    /// Files the keyed lease `id` of [`A`] on `target`, as generation 1
    /// issued it under `verdict`.
    fn keyed(store: &MemoryStore, id: u64, target: u64, verdict: SessionVerdict, now: Instant) {
        let mut record = lease(A, id, now);
        record.lease.generation = 1;
        record.lease.session = KEY;
        record.lease.session_verdict = verdict;
        record.target = WorkerId(target);
        store.issue(record).expect("a lease");
    }

    /// The worker the pin of record of [`KEY`] names.
    fn pin_of_record(store: &MemoryStore) -> Option<WorkerId> {
        SessionsStore::lookup(store, KEY.get()).map(|pin| pin.worker)
    }

    #[test]
    fn a_dispatched_frame_writes_a_keyed_leases_pin_of_record() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let _heard = hearing(&store, 1);
        keyed(&store, 1, 7, SessionVerdict::Claimed(WorkerId(7)), t0);
        assert_eq!(pin_of_record(&store), None, "an issue pins nothing");
        let at = t0 + Duration::from_millis(3);
        dispatched(&store, &open(&store, A), &a_dispatched(1), at).expect("the frame reads");
        let pin = SessionsStore::lookup(&store, KEY.get()).expect("a pin");
        assert_eq!((pin.worker, pin.last_used), (WorkerId(7), at));
        // A lease no sessions rule governed, and one with no key: no pin.
        // The first has a key of its own: under the first lease's key the
        // worker its rule prefers would be written again and hide a write.
        let mut ungoverned = lease(A, 2, t0);
        ungoverned.lease.generation = 1;
        ungoverned.lease.session = Session::from_raw(77);
        ungoverned.lease.session_verdict = SessionVerdict::None;
        store.issue(ungoverned).expect("a lease");
        issued(&store, 3, 1, t0);
        for id in 2..=3 {
            dispatched(&store, &open(&store, A), &a_dispatched(id), at).expect("the frame reads");
        }
        assert_eq!(pin_of_record(&store), Some(WorkerId(7)));
        assert_eq!(SessionsStore::len(&store), 1);
    }

    #[test]
    fn a_dispatch_on_the_failover_leaves_the_preferred_worker_pinned() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let _heard = hearing(&store, 1);
        // The key is claimed on 1, respills to 2 and is then sent to 1
        // again as its failover: 2 stays the worker the rule prefers.
        let sent = [
            (1, SessionVerdict::Claimed(WorkerId(1)), 1),
            (2, SessionVerdict::Respill, 2),
            (1, SessionVerdict::Pinned, 2),
        ];
        for (id, (target, verdict, preferred)) in (1..).zip(sent) {
            keyed(&store, id, target, verdict, t0);
            dispatched(&store, &open(&store, A), &a_dispatched(id), t0).expect("the frame reads");
            assert_eq!(pin_of_record(&store), Some(WorkerId(preferred)), "{id}");
        }
    }

    #[test]
    fn a_lease_of_a_generation_that_left_still_writes_its_pin() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let _first = hearing(&store, 1);
        keyed(&store, 1, 7, SessionVerdict::Claimed(WorkerId(7)), t0);
        let _second = hearing(&store, 2);
        dispatched(&store, &open(&store, A), &a_dispatched(1), t0).expect("the frame reads");
        // No engine is told, and the pin names the dispatched worker.
        let table = store.with_world(|world| world.spec().engine().sessions().len());
        assert_eq!((table, pin_of_record(&store)), (0, Some(WorkerId(7))));
    }

    /// A decision's read of [`KEY`] at `now` under a rule of `ttl_micros`,
    /// in the table of the engine that stands.
    fn read_under(store: &MemoryStore, ttl_micros: Option<u64>, now: Instant) {
        let rule = SessionRule {
            assignment: Assignment::Random,
            inflight_cap: 0,
            ttl_micros,
            sweep_micros: 0,
            recovery: Recovery::Original,
        };
        store.with_world(|world| {
            let table = world.spec().engine().sessions();
            table.lookup(KEY, store.micros(now), &rule);
        });
    }

    /// The pin of record ends where the engine's table forgets its key: the
    /// instant the key's rule set at its last decision, and no instant
    /// under a rule without a TTL.
    #[test]
    fn a_pin_of_record_ends_where_the_engines_table_forgets_its_key() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let _heard = hearing(&store, 1);
        read_under(&store, Some(60_000_000), t0);
        keyed(&store, 1, 7, SessionVerdict::Claimed(WorkerId(7)), t0);
        let at = t0 + Duration::from_millis(3);
        dispatched(&store, &open(&store, A), &a_dispatched(1), at).expect("the frame reads");
        let pin = SessionsStore::lookup(&store, KEY.get()).expect("a pin");
        let forgotten = store.micros(t0) + 60_000_000;
        assert_eq!((pin.worker, pin.ends), (WorkerId(7), Some(forgotten)));
        // The key read again under a rule without a TTL: its next dispatch
        // moves the pin's end with it.
        read_under(&store, None, t0);
        keyed(&store, 2, 7, SessionVerdict::Pinned, t0);
        dispatched(&store, &open(&store, A), &a_dispatched(2), at).expect("the frame reads");
        let pin = SessionsStore::lookup(&store, KEY.get()).expect("a pin");
        assert_eq!((pin.worker, pin.ends), (WorkerId(7), None));
        // And read once more under a rule of half a minute: the table's
        // word moves the pin's end earlier as well.
        read_under(&store, Some(30_000_000), t0);
        keyed(&store, 3, 7, SessionVerdict::Pinned, t0);
        dispatched(&store, &open(&store, A), &a_dispatched(3), at).expect("the frame reads");
        let pin = SessionsStore::lookup(&store, KEY.get()).expect("a pin");
        let sooner = forgotten - 30_000_000;
        assert_eq!((pin.worker, pin.ends), (WorkerId(7), Some(sooner)));
    }

    /// Files the keyed lease `id` of [`A`] on worker 7, as generation 1
    /// issued it at `now` under a rule of `ttl_micros`.
    fn keyed_under(store: &MemoryStore, id: u64, ttl_micros: Option<u64>, now: Instant) {
        let mut record = lease(A, id, now);
        record.lease.generation = 1;
        record.lease.session = KEY;
        record.lease.session_verdict = SessionVerdict::Claimed(WorkerId(7));
        record.session_ttl_micros = ttl_micros;
        record.lease.issued_micros = store.micros(now);
        record.target = WorkerId(7);
        store.issue(record).expect("a lease");
    }

    /// A dispatch whose key left the table since its decision makes the
    /// entry anew, and no decision has read it: the pin of record ends by
    /// the lease's own rule, not at the unread entry's instant.
    #[test]
    fn a_dispatch_that_makes_its_keys_entry_anew_ends_the_pin_by_its_leases_rule() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let _heard = hearing(&store, 1);
        keyed_under(&store, 1, Some(60_000_000), t0);
        let at = t0 + Duration::from_millis(3);
        dispatched(&store, &open(&store, A), &a_dispatched(1), at).expect("the frame reads");
        // The engine was told: its table holds the key again.
        let table = store.with_world(|world| world.spec().engine().sessions().len());
        let pin = SessionsStore::lookup(&store, KEY.get()).expect("a pin");
        let ends = store.micros(t0) + 60_000_000;
        assert_eq!((table, pin.worker, pin.ends), (1, WorkerId(7), Some(ends)));
    }

    /// A lease whose generation left tells no engine: its pin ends by the
    /// TTL its own decision was made under, counted from its issue,
    /// whatever the standing rules keep. Here one of them keeps its keys
    /// for ever and the other an hour, and the lease's rule kept a minute.
    #[test]
    fn a_pin_no_engine_was_told_of_ends_by_its_leases_own_rule() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let _first = hearing(&store, 1);
        keyed_under(&store, 1, Some(60_000_000), t0);
        let mut document = PolicySpec::preset("manual").expect("a preset of this build");
        document.generation = 2;
        let mut for_ever = document.bindings[0].clone();
        for_ever.matcher.model = "kept-for-ever".into();
        let rule = for_ever.sessions.as_mut().expect("the preset's rule");
        (rule.ttl_s, rule.sweep_s) = (u32::MAX, u32::MAX);
        let rule = document.bindings[0].sessions.as_mut();
        rule.expect("the preset's rule").ttl_s = 3_600;
        document.bindings.insert(0, for_ever);
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        store.apply(spec).expect("a later generation");
        let at = t0 + Duration::from_millis(3);
        dispatched(&store, &open(&store, A), &a_dispatched(1), at).expect("the frame reads");
        let pin = SessionsStore::lookup(&store, KEY.get()).expect("a pin");
        let ends = store.micros(t0) + 60_000_000;
        assert_eq!((pin.worker, pin.ends), (WorkerId(7), Some(ends)));
    }

    /// A lease of a rule with no TTL pins with no end, though every
    /// standing rule has one: the key was that rule's, and a standing
    /// rule's TTL would end its pin of record early.
    #[test]
    fn a_pin_no_engine_was_told_of_has_no_end_under_a_lease_of_a_rule_with_no_ttl() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let _first = hearing(&store, 1);
        keyed_under(&store, 1, None, t0);
        // The manual preset stands as generation 2: its rule keeps a key
        // an hour.
        let mut document = PolicySpec::preset("manual").expect("a preset of this build");
        document.generation = 2;
        let rule = document.bindings[0].sessions.as_mut();
        rule.expect("the preset's rule").ttl_s = 3_600;
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        store.apply(spec).expect("a later generation");
        let at = t0 + Duration::from_millis(3);
        dispatched(&store, &open(&store, A), &a_dispatched(1), at).expect("the frame reads");
        let pin = SessionsStore::lookup(&store, KEY.get()).expect("a pin");
        assert_eq!((pin.worker, pin.ends), (WorkerId(7), None));
    }

    /// Two leases of one key under a rule of a second, in flight at an
    /// apply, are dispatched in the reverse of their issues. The older
    /// one, dispatched last, leaves the pin's end where the newer one's
    /// rule counted it: its own count ended a tenth of a second before
    /// its dispatch.
    #[test]
    fn an_older_lease_dispatched_last_never_ends_the_pin_earlier() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let ms = Duration::from_millis;
        let _first = hearing(&store, 1);
        keyed_under(&store, 1, Some(1_000_000), t0);
        keyed_under(&store, 2, Some(1_000_000), t0 + ms(800));
        let _second = hearing(&store, 2);
        let hand = open(&store, A);
        dispatched(&store, &hand, &a_dispatched(2), t0 + ms(1_000)).expect("the frame reads");
        let ends = Some(store.micros(t0) + 1_800_000);
        let pin = SessionsStore::lookup(&store, KEY.get()).expect("a pin");
        assert_eq!(pin.ends, ends);
        dispatched(&store, &hand, &a_dispatched(1), t0 + ms(1_100)).expect("the frame reads");
        let pin = SessionsStore::lookup(&store, KEY.get()).expect("a pin");
        assert_eq!((pin.last_used, pin.ends), (t0 + ms(1_100), ends));
    }

    /// A lease no engine is told of, dispatched behind a newer lease of
    /// its key that the engine in place was told of: the end the table
    /// gave the pin stands, though the older lease's own rule, the shorter
    /// one, counts an earlier end.
    #[test]
    fn an_end_the_table_gave_stands_behind_an_older_lease_of_a_shorter_rule() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let ms = Duration::from_millis;
        let _first = hearing(&store, 1);
        keyed_under(&store, 1, Some(1_000_000), t0);
        let _second = hearing(&store, 2);
        // The generation in place reads the key under a rule of a minute,
        // and its lease is dispatched first.
        read_under(&store, Some(60_000_000), t0 + ms(1));
        let mut newer = lease(A, 2, t0 + ms(1));
        newer.lease.generation = 2;
        newer.lease.session = KEY;
        newer.lease.session_verdict = SessionVerdict::Claimed(WorkerId(7));
        newer.target = WorkerId(7);
        store.issue(newer).expect("a lease");
        let hand = open(&store, A);
        dispatched(&store, &hand, &a_dispatched(2), t0 + ms(2)).expect("the frame reads");
        dispatched(&store, &hand, &a_dispatched(1), t0 + ms(3)).expect("the frame reads");
        let pin = SessionsStore::lookup(&store, KEY.get()).expect("a pin");
        let forgotten = store.micros(t0 + ms(1)) + 60_000_000;
        assert_eq!((pin.last_used, pin.ends), (t0 + ms(3), Some(forgotten)));
    }

    #[test]
    fn a_dispatched_frame_moves_its_gateways_issued_lease_to_dispatched() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        store.issue(lease(A, 1, t0)).expect("a lease");
        let at = t0 + Duration::from_millis(3);
        dispatched(&store, &open(&store, A), &a_dispatched(1), at).expect("the frame reads");
        assert_eq!(state(&store, A, 1), Some(LeaseState::Dispatched { at }));
    }

    #[test]
    fn a_completed_frame_ends_the_lease_and_writes_the_gateways_completion() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let served = worker(&store, "http://a:8000");
        store.issue(lease(A, 1, t0)).expect("a lease");
        completed(&store, &open(&store, A), &a_completed(1, served), t0).expect("the frame reads");
        assert_eq!(state(&store, A, 1), None, "the lease ended");
        assert!(store.completed_ok_since(A, served, t0));
        assert!(
            !store.completed_ok_since(B, served, t0),
            "a completion is its gateway's own"
        );
    }

    /// The gateway reports every request, the ones it decided itself among
    /// them: the completion is a fact of the worker and the gateway.
    #[test]
    fn a_completed_frame_for_a_lease_not_on_file_still_writes_the_completion() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let served = worker(&store, "http://a:8000");
        completed(&store, &open(&store, A), &a_completed(9, served), t0).expect("no error");
        assert!(store.completed_ok_since(A, served, t0));
    }

    /// The health table grows with the rows, never with what a frame names.
    #[test]
    fn a_completion_for_a_worker_on_no_file_writes_nothing() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let gone = WorkerId(999_999);
        completed(&store, &open(&store, A), &a_completed(9, gone), t0).expect("no error");
        assert!(!store.completed_ok_since(A, gone, t0));
    }

    #[test]
    fn a_released_frame_ends_the_lease() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        store.issue(lease(A, 1, t0)).expect("a lease");
        released(&store, &open(&store, A), &a_released(1)).expect("the frame reads");
        assert_eq!(state(&store, A, 1), None);
    }

    #[test]
    fn a_lease_not_on_file_is_no_error_and_changes_nothing() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        store.issue(lease(A, 1, t0)).expect("a lease");
        dispatched(&store, &open(&store, A), &a_dispatched(2), t0).expect("no error");
        completed(&store, &open(&store, A), &a_completed(2, WorkerId(7)), t0).expect("no error");
        released(&store, &open(&store, A), &a_released(2)).expect("no error");
        assert_eq!(state(&store, A, 1), Some(LeaseState::Issued));
        assert_eq!(store.live(A), 1);
    }

    /// A ticket this build never issued is a ticket not on file: no error
    /// and nothing written, a lease under the ticket's number included.
    /// The frame is read whole all the same.
    #[test]
    fn a_released_ticket_is_no_error_and_writes_nothing() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        store.issue(lease(A, 1, t0)).expect("a lease");
        let ticket = proto::Released {
            what: Some(What::Ticket(1)),
            ..a_released(1)
        };
        assert_eq!(
            release(&ticket).expect("the frame reads"),
            (None, None, ReleaseReason::ClientGone)
        );
        released(&store, &open(&store, A), &ticket).expect("no error");
        assert_eq!(state(&store, A, 1), Some(LeaseState::Issued));

        let no_reason = proto::Released {
            reason: 0,
            ..ticket
        };
        let status =
            released(&store, &open(&store, A), &no_reason).expect_err("the frame is read whole");
        assert_eq!(
            (status.code(), status.message()),
            (
                Code::InvalidArgument,
                "Released.reason is 0, which names none this version knows"
            )
        );
    }

    #[test]
    fn a_gateway_reaches_no_lease_of_another() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        store.issue(lease(B, 1, t0)).expect("a lease");
        dispatched(&store, &open(&store, A), &a_dispatched(1), t0).expect("no error");
        completed(&store, &open(&store, A), &a_completed(1, WorkerId(7)), t0).expect("no error");
        released(&store, &open(&store, A), &a_released(1)).expect("no error");
        assert_eq!(state(&store, B, 1), Some(LeaseState::Issued));
    }

    #[test]
    fn a_frame_reads_whole_into_what_the_store_is_handed() {
        let read = dispatch(&proto::Dispatched {
            lease_id: 5,
            target_index: 255,
            side: proto::Side::Decode.into(),
            since_decision_micros: 250,
        })
        .expect("the frame reads");
        assert_eq!(
            read,
            (
                LeaseId(5),
                Dispatch {
                    target_index: 255,
                    side: Side::Decode,
                    since_decision_micros: 250,
                }
            )
        );
        let read = outcome(&proto::Completed {
            lease_id: 6,
            side: proto::Side::Prefill.into(),
            served: 7,
            status: 65_535,
            class: Class::Timeout.into(),
            ttft_micros: 1,
            tokens_in: 2,
            tokens_out: 3,
            duration_micros: 4,
        })
        .expect("the frame reads");
        assert_eq!(
            read,
            (
                LeaseId(6),
                Outcome {
                    side: Side::Prefill,
                    served: WorkerId(7),
                    status: 65_535,
                    class: TransportClass::Timeout,
                    ttft_micros: 1,
                    tokens_in: 2,
                    tokens_out: 3,
                    duration_micros: 4,
                }
            )
        );
        // A release of no side names every side; a named side, its own.
        assert_eq!(
            release(&a_released(8)).expect("the frame reads"),
            (Some(LeaseId(8)), None, ReleaseReason::ClientGone)
        );
        let one = proto::Released {
            side: proto::Side::Prefill.into(),
            reason: Reason::Expired.into(),
            ..a_released(8)
        };
        assert_eq!(
            release(&one).expect("the frame reads"),
            (
                Some(LeaseId(8)),
                Some(Side::Prefill),
                ReleaseReason::Expired
            )
        );
    }

    #[test]
    fn every_side_class_and_reason_reads_as_its_own() {
        for (wire, read) in [
            (proto::Side::Single, Side::Single),
            (proto::Side::Prefill, Side::Prefill),
            (proto::Side::Decode, Side::Decode),
            (proto::Side::Encode, Side::Encode),
        ] {
            assert_eq!(side("a side", wire.into()).expect("a side"), read);
        }
        for (wire, read) in [
            (Class::Ok, TransportClass::Ok),
            (Class::ClientError, TransportClass::ClientError),
            (Class::ServerError, TransportClass::ServerError),
            (Class::ConnectFailed, TransportClass::ConnectFailed),
            (Class::Timeout, TransportClass::Timeout),
            (Class::Cancelled, TransportClass::Cancelled),
        ] {
            assert_eq!(class(wire.into()).expect("a class"), read);
        }
        for (wire, read) in [
            (Reason::Abandoned, ReleaseReason::Abandoned),
            (Reason::LateDecision, ReleaseReason::LateDecision),
            (Reason::ClientGone, ReleaseReason::ClientGone),
            (Reason::CommitFailed, ReleaseReason::CommitFailed),
            (Reason::Expired, ReleaseReason::Expired),
            (Reason::WorkerRemoved, ReleaseReason::WorkerRemoved),
        ] {
            assert_eq!(reason(wire.into()).expect("a reason"), read);
        }
    }

    #[test]
    fn a_frame_that_reads_wrong_is_refused_by_its_fields_name_and_writes_nothing() {
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let served = worker(&store, "http://a:8000");
        store.issue(lease(A, 1, t0)).expect("a lease");
        let refused = |result: Result<(), Status>, code: Code, message: &str| {
            let status = result.expect_err("the frame is refused");
            assert_eq!((status.code(), status.message()), (code, message));
        };
        let none = |field: &str, number: i32| {
            format!("{field} is {number}, which names none this version knows")
        };
        for number in [0, 99] {
            let frame = proto::Dispatched {
                side: number,
                ..a_dispatched(1)
            };
            refused(
                dispatched(&store, &open(&store, A), &frame, t0),
                Code::InvalidArgument,
                &none("Dispatched.side", number),
            );
            let frame = proto::Completed {
                side: number,
                ..a_completed(1, served)
            };
            refused(
                completed(&store, &open(&store, A), &frame, t0),
                Code::InvalidArgument,
                &none("Completed.side", number),
            );
            let frame = proto::Completed {
                class: number,
                ..a_completed(1, served)
            };
            refused(
                completed(&store, &open(&store, A), &frame, t0),
                Code::InvalidArgument,
                &none("Completed.class", number),
            );
            let frame = proto::Released {
                reason: number,
                ..a_released(1)
            };
            refused(
                released(&store, &open(&store, A), &frame),
                Code::InvalidArgument,
                &none("Released.reason", number),
            );
        }
        let frame = proto::Released {
            side: 99,
            ..a_released(1)
        };
        refused(
            released(&store, &open(&store, A), &frame),
            Code::InvalidArgument,
            &none("Released.side", 99),
        );
        let frame = proto::Dispatched {
            target_index: 256,
            ..a_dispatched(1)
        };
        refused(
            dispatched(&store, &open(&store, A), &frame, t0),
            Code::InvalidArgument,
            "Dispatched.target_index is 256: 255 is the most it holds",
        );
        let frame = proto::Completed {
            status: 65_536,
            ..a_completed(1, served)
        };
        refused(
            completed(&store, &open(&store, A), &frame, t0),
            Code::InvalidArgument,
            "Completed.status is 65536: 65535 is the most it holds",
        );
        let frame = proto::Released {
            what: None,
            ..a_released(1)
        };
        refused(
            released(&store, &open(&store, A), &frame),
            Code::InvalidArgument,
            "Released.what names neither a lease nor a ticket",
        );
        // Nothing was written: the lease stands as issued and no
        // completion is on file.
        assert_eq!(state(&store, A, 1), Some(LeaseState::Issued));
        assert!(!store.completed_ok_since(A, served, t0));
    }
}
