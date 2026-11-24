//! A worker object's own count of requests in flight, kept by the leases
//! the store holds on file.
//!
//! The engine over worker lists reads each worker's count off its object.
//! In a gateway's own process a request's load guard moves that count. A
//! controller serves no request itself: what it knows of a request is the
//! lease it filed. So the holder hands the store each row's object as that
//! row's count ([`HostCount`]), and the store steps the object's count
//! where it books and frees the worker for a lease.
//!
//! A host that serves requests itself counts each by its load guard and
//! files no lease for it: a lease filed for a request a guard counts is
//! counted twice on the object.
//!
//! The object's count of one session key's requests moves the same way.
//! A guard counts a request under its key's text. A lease carries its
//! session as a number, so the store's step counts it under that number
//! written as text. One object stands in one process, which counts by
//! guards or by leases: the two texts never meet on it.

use std::sync::Arc;

use smg_worker::{identity::basic::Worker, registry::Generation};

use super::Objects;
use crate::{
    engine::view::{Session, WorkerId},
    store::WorkersStore,
    tracking::HostCount,
};

/// The count of the object laid for a row: the object itself.
///
/// The store keeps it while the row stands. An object laid in another's
/// place takes the standing object's running state, its count with it:
/// the store hands the new object the leases that stand and takes them
/// off the old one, which leaves the number the two share as it stood.
/// An object with a running state of its own reads them from there on.
struct ObjectCount(Arc<dyn Worker>);

impl HostCount for ObjectCount {
    fn booked(&self) {
        self.0.increment_load();
    }

    fn freed(&self) {
        self.0.decrement_load();
    }

    /// The object counts a key's requests under a text. A lease names its
    /// session by a number, so the count stands under that number as
    /// [`Session::text`] writes it, the text the engine's reader asks
    /// under for a request whose held decisions count it.
    fn session_booked(&self, session: Session) {
        let mut text = [0; Session::TEXT_LEN];
        self.0.increment_routing_key_load(session.text(&mut text));
    }

    fn session_freed(&self, session: Session) {
        let mut text = [0; Session::TEXT_LEN];
        self.0.decrement_routing_key_load(session.text(&mut text));
    }
}

impl<G: Generation> Objects<G> {
    /// Hands `store` the object held for `row` as the row's count, where
    /// one is held. The holder calls it behind each lay of an object and
    /// for each row a whole read finds standing: a store that took
    /// another's rows gives each a new place in its counts.
    pub(super) fn counted<S: WorkersStore + ?Sized>(&self, store: &S, row: WorkerId) {
        let object = {
            let held = self.held.read();
            let laid = held.objects.get(&row).and_then(|id| held.rows.get(id));
            laid.map(|laid| Arc::clone(&laid.object))
        };
        if let Some(object) = object {
            store.host_count(row, Box::new(ObjectCount(object)));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use smg_worker::registry::WorkerRegistry;

    use super::*;
    use crate::{
        engine::view::{Identity, RequestFacts, SessionCountedUnder},
        store::{
            GatewayId, LeaseRecord, LeaseState, LeasesStore, MemoryStore, PickRecord, WorkerFacts,
            WorkerPatch,
        },
        tracking::lease::{Lease, LeaseId, Outcome, TransportClass},
        worker_host::Members,
    };

    const A: &str = "http://a:8000";
    const B: &str = "http://b:8000";
    const GATEWAY: GatewayId = GatewayId(1);

    /// A holder with the store it follows, under the term its leases state.
    struct Replica {
        store: Arc<MemoryStore>,
        objects: Objects<Members>,
    }

    impl Replica {
        fn new() -> Self {
            let store = Arc::new(MemoryStore::new());
            store.set_term(1);
            let objects = Objects::new(Arc::new(WorkerRegistry::new()));
            Self { store, objects }
        }

        /// Have the holder read the store.
        fn follow(&self) {
            self.objects.follow(self.store.as_ref());
        }

        /// Write the first life of the row at `url` and have the holder
        /// follow: the row's id.
        fn write(&self, url: &str) -> WorkerId {
            self.life(url, 1)
        }

        /// Write the life `incarnation` of the row at `url` and have the
        /// holder follow: the row's id.
        fn life(&self, url: &str, incarnation: u64) -> WorkerId {
            let row = self.row(url, incarnation);
            self.follow();
            row
        }

        /// Write the life `incarnation` of the row at `url`, which the
        /// holder has not read yet: the row's id.
        fn row(&self, url: &str, incarnation: u64) -> WorkerId {
            let facts = WorkerFacts {
                key: url.to_string(),
                engine: url.to_string(),
                incarnation,
                models: vec!["m".to_string()],
                dp_size: 1,
                ..WorkerFacts::default()
            };
            self.store.upsert(facts).id()
        }

        /// The object the registry holds under `url`.
        fn at(&self, url: &str) -> Arc<dyn Worker> {
            self.objects.registry().get_by_url(url).expect("an object")
        }

        /// File a single's lease under `id` that books `target`.
        fn file(&self, id: u64, target: WorkerId) {
            self.file_of(id, target, Session::NONE);
        }

        /// File a single's lease of `session` under `id` that books
        /// `target`.
        fn file_of(&self, id: u64, target: WorkerId, session: Session) {
            let now = Instant::now();
            let record = LeaseRecord {
                lease: Lease {
                    id: LeaseId(id),
                    session,
                    ..Lease::default()
                },
                gateway: GATEWAY,
                request: id,
                target,
                picked: PickRecord::default(),
                decode: None,
                untold: false,
                term: 1,
                issued_at: now,
                dispatch_deadline: now + Duration::from_secs(1),
                state: LeaseState::ISSUED,
            };
            assert_eq!(self.store.issue(record), Ok(()));
        }

        /// End the lease under `id`.
        fn end(&self, id: u64) {
            let outcome = Outcome::reported(200, TransportClass::Ok);
            assert!(self.store.completed(GATEWAY, LeaseId(id), &outcome));
        }

        /// Move the facts of the row at `url`: one label more.
        fn relabel(&self, url: &str) {
            let row = self.store.by_key(url).expect("a row");
            let labels = [("tier".to_string(), "gold".to_string())];
            let patch = WorkerPatch {
                labels: Some(labels.into_iter().collect()),
                ..WorkerPatch::default()
            };
            assert!(self.store.patch(row.id, patch, row.revision).is_some());
        }
    }

    /// A lease on file counts on its worker's object from its filing to
    /// its end, and on no other object.
    #[test]
    fn a_lease_on_file_counts_on_its_workers_object() {
        let replica = Replica::new();
        let (a, b) = (replica.write(A), replica.write(B));
        let loads = || (replica.at(A).load(), replica.at(B).load());
        assert_eq!(loads(), (0, 0));
        replica.file(1, a);
        replica.file(2, a);
        replica.file(3, b);
        assert_eq!(loads(), (2, 1));
        replica.end(1);
        assert_eq!(loads(), (1, 1));
        replica.end(2);
        replica.end(3);
        assert_eq!(loads(), (0, 0));
    }

    /// The two sessions of the keyed cases.
    const ONE: Session = Session::from_raw(0x1f);
    const TWO: Session = Session::from_raw(u64::MAX);

    /// What `object` counts of the two sessions' requests, each under its
    /// number as text, and how many keys it counts requests of.
    fn keyed(object: &Arc<dyn Worker>) -> (usize, usize, usize) {
        let (mut one, mut two) = ([0; Session::TEXT_LEN], [0; Session::TEXT_LEN]);
        (
            object.routing_key_inflight(ONE.text(&mut one)),
            object.routing_key_inflight(TWO.text(&mut two)),
            object.routing_key_load(),
        )
    }

    /// A lease that names a session counts on its worker's object under
    /// the session's number as text, from its filing to its end: the count
    /// a capped sessions rule reads there. A lease with no session counts
    /// under no key.
    #[test]
    fn a_keyed_lease_counts_on_its_workers_object_under_its_sessions_text() {
        let replica = Replica::new();
        let (a, b) = (replica.write(A), replica.write(B));
        let (at_a, at_b) = (replica.at(A), replica.at(B));
        replica.file_of(1, a, ONE);
        replica.file_of(2, a, ONE);
        replica.file_of(3, a, TWO);
        replica.file_of(4, b, ONE);
        replica.file(5, a);
        assert_eq!((keyed(&at_a), keyed(&at_b)), ((2, 1, 2), (1, 0, 1)));
        assert_eq!((at_a.load(), at_b.load()), (4, 1));
        replica.end(1);
        replica.end(3);
        assert_eq!(
            keyed(&at_a),
            (1, 0, 1),
            "a key leaves with its last request"
        );
        for id in [2, 4, 5] {
            replica.end(id);
        }
        assert_eq!((keyed(&at_a), keyed(&at_b)), ((0, 0, 0), (0, 0, 0)));
    }

    /// The sessions of the leases that stand follow their row's object: a
    /// lease filed ahead of the object is on it once it is laid, and an
    /// object laid in its place reads the same sessions, whose ends are
    /// taken off it.
    #[test]
    fn the_sessions_that_stand_follow_their_rows_object() {
        let replica = Replica::new();
        let a = replica.row(A, 1);
        replica.file_of(1, a, ONE);
        replica.file_of(2, a, ONE);
        replica.file_of(3, a, TWO);
        replica.follow();
        let before = replica.at(A);
        assert_eq!(keyed(&before), (2, 1, 2), "the leases that stood");

        // The row's facts move: the holder builds another object for it.
        replica.relabel(A);
        replica.follow();
        let after = replica.at(A);
        assert!(!Arc::ptr_eq(&before, &after), "another object");
        assert_eq!(keyed(&after), (2, 1, 2), "the sessions that stand");
        replica.end(1);
        replica.end(3);
        assert_eq!(keyed(&after), (1, 0, 1));
        replica.end(2);
        assert_eq!((keyed(&after), keyed(&before)), ((0, 0, 0), (0, 0, 0)));
    }

    /// A row that leaves takes the sessions of the leases that book it off
    /// its object: their ends step nothing.
    #[test]
    fn a_row_that_leaves_takes_its_sessions_off_its_object() {
        let replica = Replica::new();
        let a = replica.write(A);
        let object = replica.at(A);
        replica.file_of(1, a, ONE);
        replica.file_of(2, a, TWO);
        assert_eq!(keyed(&object), (1, 1, 2));
        let row = replica.store.by_key(A).expect("a row");
        assert!(replica.store.remove(row.id, row.revision).is_some());
        assert_eq!(keyed(&object), (0, 0, 0), "at the store's write");
        replica.end(1);
        replica.end(2);
        assert_eq!(keyed(&object), (0, 0, 0));
    }

    /// What moves a lease's count onto its worker's object and what reads
    /// a record's count write one text. A lease of a request's session is
    /// read under the name the request's facts make for a count that
    /// stands under the session's number, with the key in either form. The
    /// key's own text names no count of a lease.
    #[test]
    fn a_leases_session_is_read_under_the_name_its_requests_facts_make() {
        let replica = Replica::new();
        let a = replica.write(A);
        let object = replica.at(A);
        let plain = RequestFacts {
            model: "model-a",
            session_key: Identity::plain("conversation-7"),
            ..RequestFacts::default()
        };
        let hashed = RequestFacts {
            session_key: Identity::hashed(plain.session_key.hash),
            ..plain
        };
        replica.file_of(1, a, plain.session());
        for facts in [plain, hashed] {
            let mut name = [0; Session::TEXT_LEN];
            let name = SessionCountedUnder::SessionNumber.name(&facts, &mut name);
            let read = name.map(|name| object.routing_key_inflight(name));
            assert_eq!(read, Some(1));
        }
        assert_eq!(object.routing_key_inflight("conversation-7"), 0);
        replica.end(1);
        assert_eq!(object.routing_key_load(), 0);
    }

    /// A lease filed between a row's write and its object's lay is on the
    /// object once it is laid: the first count handed takes it.
    #[test]
    fn a_lease_filed_ahead_of_its_workers_object_counts_on_it() {
        let replica = Replica::new();
        let a = replica.row(A, 1);
        replica.file(1, a);
        replica.follow();
        assert_eq!(replica.at(A).load(), 1, "the lease that stood");
        replica.end(1);
        assert_eq!(replica.at(A).load(), 0);
    }

    /// An object laid in another's place under its id reads the leases
    /// filed over the one before it, and their ends are taken off it.
    #[test]
    fn an_object_laid_in_anothers_place_reads_the_leases_that_stand() {
        let replica = Replica::new();
        let a = replica.write(A);
        let before = replica.at(A);
        replica.file(1, a);
        replica.file(2, a);
        assert_eq!(before.load(), 2);

        // The row's facts move: the holder builds another object for it.
        replica.relabel(A);
        replica.follow();
        let after = replica.at(A);
        assert!(!Arc::ptr_eq(&before, &after), "another object");
        assert_eq!(after.load(), 2, "the leases that stand");

        replica.end(1);
        assert_eq!(after.load(), 1);
        replica.file(3, a);
        assert_eq!(after.load(), 2);
        replica.end(2);
        replica.end(3);
        assert_eq!((after.load(), before.load()), (0, 0));
    }

    /// A row that leaves takes the leases that book it off its object,
    /// and the store keeps the object no longer: their ends step nothing.
    #[test]
    fn a_row_that_leaves_takes_its_leases_off_its_object() {
        let replica = Replica::new();
        let a = replica.write(A);
        let object = replica.at(A);
        replica.file(1, a);
        replica.file(2, a);
        assert_eq!(object.load(), 2);
        let row = replica.store.by_key(A).expect("a row");
        assert!(replica.store.remove(row.id, row.revision).is_some());
        assert_eq!(object.load(), 0, "at the store's write");
        replica.follow();
        assert!(replica.objects.registry().get_by_url(A).is_none());
        assert_eq!(Arc::strong_count(&object), 1, "the store lets it go");

        replica.end(1);
        replica.end(2);
        assert_eq!(object.load(), 0);
    }

    /// A worker's new life reads its own row's leases alone: the leases of
    /// the life before it left the count with that life's row.
    #[test]
    fn a_workers_new_life_reads_its_own_rows_leases() {
        let replica = Replica::new();
        let old = replica.write(A);
        replica.file(1, old);
        replica.file(2, old);
        assert_eq!(replica.at(A).load(), 2);

        let new = replica.life(A, 2);
        assert_ne!(old, new);
        let object = replica.at(A);
        assert_eq!(object.load(), 0, "no lease of the life before it");
        replica.file(3, new);
        assert_eq!(object.load(), 1);
        replica.end(1);
        assert_eq!(object.load(), 1, "an end of the life before it");
        replica.end(3);
        assert_eq!(object.load(), 0);
    }

    /// A lease filed on a worker's new life before its object is laid is
    /// on that object, and no lease of the life before it is.
    #[test]
    fn a_lease_filed_ahead_of_a_new_lifes_object_counts_on_it() {
        let replica = Replica::new();
        let old = replica.write(A);
        replica.file(1, old);
        let new = replica.row(A, 2);
        replica.file(2, new);
        replica.follow();
        assert_eq!(replica.at(A).load(), 1, "the new life's lease");
        replica.end(2);
        assert_eq!(replica.at(A).load(), 0);
    }

    /// A lease filed ahead of an object that takes a standing one's place
    /// is on the new object: the count is handed behind the registry's
    /// write, on the running state the object keeps.
    #[test]
    fn a_lease_filed_ahead_of_an_object_laid_in_anothers_place_counts_on_it() {
        let leader = Replica::new();
        let a = leader.write(A);
        let follower = Replica::new();
        let cut = || leader.store.checkpoint();
        assert!(follower.store.restore_from(cut()));
        follower.follow();
        let before = follower.at(A);

        // The leader's row moves and the follower takes its rows again:
        // the row stands at a new place, with another object to lay.
        leader.relabel(A);
        assert!(follower.store.restore_from(cut()));
        follower.store.set_term(1);
        follower.file(1, a);
        follower.follow();
        let after = follower.at(A);
        assert!(!Arc::ptr_eq(&before, &after), "another object");
        assert_eq!(after.load(), 1, "the lease filed ahead of it");
        follower.end(1);
        assert_eq!(after.load(), 0);
    }

    /// An object with a running state of its own, laid by another holder
    /// over the same store, takes the leases that stand, and the object
    /// before it gives them up.
    #[test]
    fn an_object_of_another_running_state_takes_the_leases_that_stand() {
        let replica = Replica::new();
        let a = replica.write(A);
        replica.file(1, a);
        replica.file(2, a);
        let first = replica.at(A);
        assert_eq!(first.load(), 2);

        let other = Objects::<Members>::new(Arc::new(WorkerRegistry::new()));
        other.follow(replica.store.as_ref());
        let second = other.registry().get_by_url(A).expect("an object");
        assert_eq!((second.load(), first.load()), (2, 0));
        replica.end(1);
        assert_eq!((second.load(), first.load()), (1, 0));
    }

    /// A store that took another's rows gives each row a new place in its
    /// counts: the holder's next whole read hands each object again, so a
    /// lease filed at the new place counts on it.
    #[test]
    fn a_whole_read_hands_each_standing_object_again() {
        let leader = Replica::new();
        let a = leader.write(A);
        let follower = Replica::new();
        let cut = || leader.store.checkpoint();
        assert!(follower.store.restore_from(cut()));
        follower.follow();
        let object = follower.at(A);
        let held = Arc::strong_count(&object);

        // The same rows again: every row takes a new place.
        assert!(follower.store.restore_from(cut()));
        let kept = Arc::strong_count(&object);
        assert_eq!(kept, held - 1, "the old place lets the object go");
        follower.store.set_term(1);
        follower.file(1, a);
        assert_eq!(object.load(), 0, "no count stands at the new place yet");
        follower.follow();
        assert!(Arc::ptr_eq(&object, &follower.at(A)), "the object stands");
        assert_eq!(object.load(), 1, "the lease filed at the new place");
        follower.file(2, a);
        assert_eq!(object.load(), 2);
        follower.end(1);
        follower.end(2);
        assert_eq!(object.load(), 0);
    }

    /// The engine over worker lists reads the count the leases on file
    /// moved: its next decision goes to the worker fewer leases book.
    #[cfg(feature = "in-process")]
    #[test]
    fn the_engine_over_worker_lists_reads_the_count_at_its_next_decision() {
        use crate::{
            document::spec::{AppliedSpec, PolicySpec},
            engine::{
                Constraints, Decided, LentList, PolicyEngine, Request, RequestFacts, RequestMaker,
                Role, WorkerLists,
            },
            store::WorkerStatus,
        };

        /// A single request with the list its host hands for it.
        struct Asks<'a>(Request<'a>, LentList<'a>);

        impl RequestMaker for Asks<'_> {
            fn model(&self) -> &str {
                self.0.facts.model
            }

            fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
                self.0
            }

            fn workers(&self) -> WorkerLists<'_> {
                WorkerLists::Single(self.1)
            }
        }

        let mut replica = Replica::new();
        replica.objects = replica.objects.with_status_followed();
        let rows = [replica.write(A), replica.write(B)];
        for url in [A, B] {
            let row = replica.store.by_key(url).expect("a row");
            let ready = WorkerStatus::Ready;
            assert!(replica
                .store
                .set_status(row.id, ready, row.revision)
                .is_some());
        }
        replica.follow();
        let registry = replica.objects.registry();
        let generation = registry.model_generation("m").expect("a generation");
        let list = replica.objects.lent(&generation, Role::Single);
        let stated = list.facts.expect("the holder lends facts");
        let ids: Vec<WorkerId> = stated.iter().filter_map(|fact| fact.id).collect();
        assert_eq!(ids, rows, "the list is the two rows' workers");

        // A profile that takes the worker with the fewest requests in flight.
        let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
        let fewest = r#"{"score":["least-requests"],"pick":"best"}"#;
        let profile = serde_json::from_str(fewest).expect("a profile");
        document.profiles.insert("round_robin".to_string(), profile);
        let engine = PolicyEngine::new(AppliedSpec::validate(&document).expect("it validates"));
        let request = Request {
            id: b"request",
            facts: RequestFacts {
                model: "m",
                seed: 7,
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        };
        let asks = Asks(request, list);
        let taken = || match engine.decide_at(&asks, Instant::now).expect("an answer") {
            Decided::Alone(pick) => ids[pick.place as usize],
            Decided::Whole(decided, _) => {
                let pick = decided.decision.pick().expect("a pick");
                pick.targets.first().expect("a target").id
            }
        };

        let [a, b] = rows;
        replica.file(1, a);
        assert_eq!(taken(), b, "a lease books the first worker");
        replica.file(2, b);
        replica.file(3, b);
        assert_eq!(taken(), a, "two leases book the second");
        replica.end(1);
        replica.end(2);
        assert_eq!(taken(), a, "one lease books the second yet");
    }
}
