//! A store that follows a controller's watch of its workers: a gateway's
//! side of `WatchWorkers` (design 5.2), for a host that decides against a
//! standalone controller and keeps what it was last told.
//!
//! A [`Watcher`] takes a snapshot's rows at its last page
//! ([`MemoryStore::watch_from`]). The store keeps its leases and session
//! pins, and the workers still present keep their gateway vetoes.
//! Each membership and status event after the snapshot lands by the door a
//! standby's applier uses for a row ([`MemoryStore::follow`]). A
//! row holds the controller's id and revision, and the store's epoch is
//! the controller's. The watcher keeps the place of the last event it
//! took, and its next stream goes on after it ([`Watcher::open`],
//! [`Watcher::open_on`]).
//!
//! **Through the worker objects.** The watcher writes its store through
//! the holder of its host's worker objects ([`Objects`]), as a standby's
//! applier writes its own. The holder builds the object of the row an
//! event tells, hands the store the event, and puts the object in the
//! registry where the store took it. A snapshot's last page is the
//! store's write and the read of its rows in one turn of the holder. So
//! the object of each row stands when the event that told it returns, and
//! the host follows nothing behind the watch.
//!
//! **One stream at a time.** A stream is followed through the watcher
//! ([`WatchStream`]), which borrows it: the watcher takes that stream's
//! events and no other's, and opens its next stream only once this one is
//! dropped. A snapshot's pages carry no page number, so a page of a stream let
//! go would read as a first page.
//!
//! **No journal.** A watch tells no start of a term and no generation, so
//! the places it is told skip, and no log continues over a gap. The store a
//! watcher writes holds no journal: the place is the watcher's own.
//!
//! **Every model.** A followed write takes the next epoch alone. A watch
//! that names models is told some changes only, so the watcher's request
//! names none.
//!
//! **A lost place.** The watcher loses its place ([`WatchPositionLost`]) at the
//! controller's own word (`SnapshotRequired`), at an event the wire's
//! conversion refuses, at one out of its order (a page where the tail is
//! due, a place not past the last) and at one that does not land where the
//! store stands. The stream that told it is let go: the
//! watcher takes nothing more of it. The next stream starts from a
//! snapshot. What the store held stands until that snapshot's last page.

use std::{future::Future, mem, sync::Arc, time::Instant};

use smg_worker::registry::Generation;

use crate::{
    service::{
        client::{self, Client, Link, Watch},
        controller::worker_wire::{self, Refused},
        proto::{self, worker_event::Event},
    },
    store::{MemoryStore, Position, StoreCheckpoint, WorkerRow, WorkersStore},
    worker_host::Objects,
};

/// Why a watcher lost its place in the watch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WatchPositionLost {
    /// The controller's word: the watch cannot go on from the place asked.
    Required,
    /// A field the wire's conversion refuses.
    Unread {
        /// The field's path in the event.
        path: &'static str,
        /// Why it is refused.
        why: &'static str,
    },
    /// A page of another snapshot or at a place held, an event of the tail
    /// where a snapshot's page is due or at a place not past the last, or
    /// a page or a change of a stream that lost the place.
    Order,
    /// The event did not land: the store does not stand where the
    /// controller's stood before it.
    Apart,
}

/// The snapshot whose pages are arriving: its place, its epoch and the rows
/// told so far.
#[derive(Debug)]
struct Staged {
    at: Position,
    epoch: u64,
    rows: Vec<WorkerRow>,
}

/// Where a watcher stands in the stream it reads.
#[derive(Debug)]
enum Stands {
    /// Nowhere: a snapshot's first page is due.
    Nowhere,
    /// Inside a snapshot, with its pages so far.
    Inside(Staged),
    /// At the place of the last event taken.
    At(Position),
    /// The stream lost the place: nothing more of it is taken.
    LetGo,
}

/// One store and its host's worker objects, written as a controller's
/// watch tells them.
#[derive(Debug)]
pub struct Watcher<G> {
    store: Arc<MemoryStore>,
    /// The holder each event and each snapshot is handed to.
    objects: Arc<Objects<G>>,
    stands: Stands,
    /// The snapshots applied so far.
    snapshots: u64,
}

/// One stream of the watch as its watcher follows it. It also writes each
/// event into the watcher's store. It borrows the
/// watcher: the watcher takes this stream's events and no other's, and
/// opens its next stream only once this one is dropped.
#[derive(Debug)]
pub struct WatchStream<'w, G> {
    watcher: &'w mut Watcher<G>,
    watch: Watch,
}

/// Why a stream is followed no further.
#[derive(Debug)]
pub enum WatchStreamEnd {
    /// The watcher lost its place: its next stream starts from a snapshot.
    Lost(WatchPositionLost),
    /// The client's error on the stream: what the watcher held stands, and
    /// its next stream goes on after its place, or starts the snapshot
    /// again where the stream was cut inside one.
    Client(client::Error),
}

impl<G: Generation> WatchStream<'_, G> {
    /// Takes the stream's next event into the store, at the instant it is
    /// read. `Ok(true)`: an event was taken. `Ok(false)`: the controller
    /// ended the stream with no error, and the place stands. The wait
    /// holds no bound: a fleet that does not change tells nothing.
    ///
    /// # Errors
    ///
    /// [`WatchStreamEnd`]: the watcher lost its place, or the stream ended with an
    /// error.
    pub async fn next(&mut self) -> Result<bool, WatchStreamEnd> {
        match self.watch.next().await {
            Ok(Some(event)) => self
                .watcher
                .take(&event, Instant::now())
                .map(|()| true)
                .map_err(WatchStreamEnd::Lost),
            Ok(None) => Ok(false),
            Err(error) => Err(WatchStreamEnd::Client(error)),
        }
    }

    /// The watcher that follows the stream.
    #[must_use]
    pub fn watcher(&self) -> &Watcher<G> {
        self.watcher
    }
}

fn unread(refused: Refused) -> WatchPositionLost {
    WatchPositionLost::Unread {
        path: refused.path,
        why: refused.why,
    }
}

impl<G: Generation> Watcher<G> {
    /// A watcher of `store`, whose worker rows come from the controller,
    /// and of `objects`, the holder of the host's worker objects over that
    /// store. The watch alone writes the store's rows, and through the
    /// holder: the object of a row stands when its event returns. The
    /// watch tells each status, so the holder is one whose objects follow
    /// their rows' status ([`Objects::with_status_followed`]).
    /// A snapshot keeps the store's leases and session pins, and gateway
    /// vetoes for workers still present. Use a store with no journal, since
    /// the places in a watch can skip. A journaled store takes a snapshot
    /// only at a place its journal accepts ([`WatchPositionLost::Apart`]
    /// otherwise).
    #[must_use]
    pub fn new(store: Arc<MemoryStore>, objects: Arc<Objects<G>>) -> Self {
        Self {
            store,
            objects,
            stands: Stands::Nowhere,
            snapshots: 0,
        }
    }

    /// The store the watcher writes.
    #[must_use]
    pub fn store(&self) -> &Arc<MemoryStore> {
        &self.store
    }

    /// The holder of the worker objects the watcher lays.
    #[must_use]
    pub fn objects(&self) -> &Arc<Objects<G>> {
        &self.objects
    }

    /// The place of the last event taken: none before a snapshot's last
    /// page, and after a lost place.
    #[must_use]
    pub fn at(&self) -> Option<Position> {
        match self.stands {
            Stands::At(at) => Some(at),
            Stands::Nowhere | Stands::Inside(_) | Stands::LetGo => None,
        }
    }

    /// The snapshots applied so far: one where the watcher never lost its
    /// place, since a stream opened at a place that stands is told the
    /// tail alone.
    #[must_use]
    pub fn snapshots(&self) -> u64 {
        self.snapshots
    }

    /// Opens the watch's next stream on `client` and follows it: the tail
    /// after the last event taken, or a snapshot first where the watcher
    /// holds no place.
    ///
    /// # Errors
    ///
    /// The client's: the open had no answer inside its bound, or the
    /// controller refused it.
    pub async fn open(&mut self, client: &mut Client) -> Result<WatchStream<'_, G>, client::Error> {
        self.open_with(move |request| client.watch_workers(request))
            .await
    }

    /// Opens the watch as [`Self::open`] does, on the link's channel.
    pub async fn open_on(&mut self, link: &Link) -> Result<WatchStream<'_, G>, client::Error> {
        self.open_with(|request| link.watch_workers(request)).await
    }

    async fn open_with<F: Future<Output = Result<Watch, client::Error>>>(
        &mut self,
        open: impl FnOnce(proto::WatchWorkersRequest) -> F,
    ) -> Result<WatchStream<'_, G>, client::Error> {
        let watch = open(self.request()).await?;
        Ok(WatchStream {
            watcher: self,
            watch,
        })
    }

    /// The request that opens the watch's next stream: every model, and the
    /// tail after the last event taken, or a snapshot first where the
    /// watcher holds no place. A stream cut inside a snapshot starts that
    /// snapshot again: the pages taken so far are let go. The stream it
    /// opens borrows the watcher ([`WatchStream`]), so the events taken next
    /// are that stream's.
    fn request(&mut self) -> proto::WatchWorkersRequest {
        let at = self.at().unwrap_or_default();
        if !matches!(self.stands, Stands::At(_)) {
            self.stands = Stands::Nowhere;
        }
        proto::WatchWorkersRequest {
            models: Vec::new(),
            resume_seq: at.seq,
            with_needs: false,
            resume_term: at.term,
        }
    }

    /// One event of the watch, taken at `now`: a snapshot's page is kept
    /// until its last, which applies the rows, and an event of the tail
    /// is written where the store stands. A generation's needs, which the
    /// watcher does not ask for, write no row.
    ///
    /// # Errors
    ///
    /// [`WatchPositionLost`]: the watcher holds no place now and lets the stream go.
    /// Every later event of it is refused until the next stream, which
    /// starts from a snapshot.
    fn take(&mut self, told: &proto::WorkerEvent, now: Instant) -> Result<(), WatchPositionLost> {
        let taken = self.taken(told, now);
        if taken.is_err() {
            self.stands = Stands::LetGo;
        }
        taken
    }

    fn taken(&mut self, told: &proto::WorkerEvent, now: Instant) -> Result<(), WatchPositionLost> {
        let at = Position {
            seq: told.seq,
            term: told.term,
        };
        match told.event.as_ref() {
            Some(Event::Snapshot(page)) => self.page(at, told.membership_epoch, page, now),
            Some(Event::SnapshotRequired(_)) => Err(WatchPositionLost::Required),
            Some(event) => {
                // The tail follows a snapshot's last page, never a page,
                // and each of its places is past the last.
                let Stands::At(held) = self.stands else {
                    return Err(WatchPositionLost::Order);
                };
                if at.seq <= held.seq || at.term < held.term {
                    return Err(WatchPositionLost::Order);
                }
                if !matches!(event, Event::Needs(_)) {
                    let (epoch, change, row) =
                        worker_wire::told_change(told, now).map_err(unread)?;
                    let before = WorkersStore::epoch(&*self.store);
                    self.objects.follow_at(&self.store, at, epoch, change, row);
                    // A change that does not fit writes nothing and says
                    // nothing: it landed where the epoch moved to its own.
                    // One at the epoch the store held already moves none.
                    if before == epoch || WorkersStore::epoch(&*self.store) != epoch {
                        return Err(WatchPositionLost::Apart);
                    }
                }
                self.stands = Stands::At(at);
                Ok(())
            }
            None => Err(WatchPositionLost::Unread {
                path: "event",
                why: "an event this build does not know",
            }),
        }
    }

    /// Keep a snapshot's pages until the last, then apply its rows together.
    fn page(
        &mut self,
        at: Position,
        epoch: u64,
        page: &proto::WorkerSnapshot,
        now: Instant,
    ) -> Result<(), WatchPositionLost> {
        // A snapshot starts where the watcher stands nowhere. A page at a
        // place held is out of its order: a stream that resumes is told
        // the tail or the controller's word, and no page follows the tail.
        let mut staged = match mem::replace(&mut self.stands, Stands::Nowhere) {
            // Every page of one snapshot stands at its place.
            Stands::Inside(staged) if staged.at == at && staged.epoch == epoch => staged,
            Stands::Inside(_) | Stands::At(_) | Stands::LetGo => {
                return Err(WatchPositionLost::Order)
            }
            Stands::Nowhere => Staged {
                at,
                epoch,
                rows: Vec::new(),
            },
        };
        for row in &page.workers {
            staged
                .rows
                .push(worker_wire::told_row(row, now).map_err(unread)?);
        }
        if !page.last {
            self.stands = Stands::Inside(staged);
            return Ok(());
        }
        let cut = StoreCheckpoint {
            at,
            epoch,
            rows: staged.rows,
        };
        if !self.objects.watch_from(&self.store, cut) {
            return Err(WatchPositionLost::Apart);
        }
        self.snapshots += 1;
        self.stands = Stands::At(at);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        sync::Arc,
        time::{Duration, Instant},
    };

    use smg_worker::registry::WorkerRegistry;

    use super::{WatchPositionLost, Watcher};
    use crate::{
        engine::view::WorkerId,
        service::{
            controller::worker_wire,
            proto::{self, worker_event::Event},
            replication::Log,
        },
        store::{
            GatewayId, LeaseRecord, LeaseState, LeasesStore, MembershipChange, MemoryStore,
            PickRecord, Position, WorkerFacts, WorkerPatch, WorkerRow, WorkerStatus, WorkersStore,
        },
        tracking::lease::{Lease, LeaseId},
        worker_host::{Members, Objects},
    };

    const TERM: u64 = 1;

    fn facts(key: &str, incarnation: u64) -> WorkerFacts {
        WorkerFacts {
            key: key.to_owned(),
            engine: key.to_owned(),
            incarnation,
            models: vec!["m".to_owned()],
            dp_size: 1,
            ..WorkerFacts::default()
        }
    }

    /// A controller's side of a watch: its store, and the place of the last
    /// event it told.
    struct Told {
        store: MemoryStore,
        seq: u64,
    }

    impl Told {
        fn new() -> Self {
            Self {
                store: MemoryStore::new(),
                seq: 100,
            }
        }

        fn row(&self, id: WorkerId) -> WorkerRow {
            WorkersStore::get(&self.store, id).expect("the row")
        }

        /// The watch's event for the write the store just took. `before` is
        /// the row the write replaced or took out.
        fn event(
            &mut self,
            change: MembershipChange,
            before: Option<&WorkerRow>,
        ) -> proto::WorkerEvent {
            // Writes a watch is not told take places between two of its
            // events: the places a watcher is told skip.
            self.seq += 3;
            let event = match change {
                MembershipChange::Registered { id } => worker_wire::registered(&self.row(id)),
                MembershipChange::Relabelled { id } | MembershipChange::Resized { id } => {
                    worker_wire::relabelled(&self.row(id))
                }
                MembershipChange::Replaced { new, .. } => {
                    worker_wire::replaced(before.expect("the old row"), &self.row(new), false)
                }
                MembershipChange::Removed { .. } => {
                    worker_wire::removed(before.expect("the last row"))
                }
                MembershipChange::StatusChanged { id, from, to } => {
                    worker_wire::status_changed(id, from, to)
                }
            };
            worker_wire::event(event, self.seq, WorkersStore::epoch(&self.store), TERM)
        }

        /// The snapshot of the store as it stands, `rows` a page.
        fn pages(&self, rows: usize) -> Vec<proto::WorkerEvent> {
            let cut = self.store.checkpoint();
            let mut chunks: Vec<&[WorkerRow]> = cut.rows.chunks(rows).collect();
            if chunks.is_empty() {
                chunks.push(&[]);
            }
            let last = chunks.len() - 1;
            chunks
                .iter()
                .enumerate()
                .map(|(n, chunk)| {
                    let page = proto::WorkerSnapshot {
                        workers: chunk.iter().map(worker_wire::row).collect(),
                        last: n == last,
                    };
                    worker_wire::event(Event::Snapshot(page), self.seq, cut.epoch, TERM)
                })
                .collect()
        }
    }

    /// What two stores must agree on: the epoch, and each row's id, facts,
    /// status and revision. An admission instant crosses as an age.
    fn held(store: &MemoryStore) -> (u64, Vec<(WorkerId, WorkerFacts, WorkerStatus, u64)>) {
        let (epoch, mut rows) = WorkersStore::all_rows(store);
        rows.sort_by_key(|row| row.id.0);
        let rows = rows
            .into_iter()
            .map(|row| (row.id, row.facts, row.status, row.revision))
            .collect();
        (epoch, rows)
    }

    /// A watcher of `store` over a holder of its own, whose objects follow
    /// their rows' status as a gateway's do.
    fn over(store: Arc<MemoryStore>) -> Watcher<Members> {
        let objects = Objects::new(Arc::new(WorkerRegistry::new())).with_status_followed();
        Watcher::new(store, Arc::new(objects))
    }

    /// The watcher's worker objects state the rows its store holds, each
    /// read back from its object, and no other object stands. Nothing
    /// follows the store here: the watcher's own writes laid them.
    #[track_caller]
    fn laid(watcher: &Watcher<Members>) {
        let (_, mut rows) = WorkersStore::all_rows(&**watcher.store());
        rows.sort_by_key(|row| row.id.0);
        let objects = watcher.objects();
        let read = |row: &WorkerRow| {
            let object = objects.object(row.id)?;
            objects.row_of(&object)
        };
        let stated: Vec<_> = rows.iter().map(read).collect();
        let rows: Vec<_> = rows.into_iter().map(Some).collect();
        assert_eq!(stated, rows, "each row's object states it");
        assert_eq!(objects.len(), rows.len(), "no other object stands");
    }

    fn watching(told: &Told, rows: usize) -> Watcher<Members> {
        let mut watcher = over(Arc::new(MemoryStore::new()));
        let request = watcher.request();
        assert_eq!((request.resume_seq, request.resume_term), (0, 0));
        assert!(request.models.is_empty(), "every model");
        assert!(!request.with_needs, "no generation's needs");
        for page in told.pages(rows) {
            assert_eq!(watcher.take(&page, Instant::now()), Ok(()));
        }
        assert_eq!(held(watcher.store()), held(&told.store));
        laid(&watcher);
        watcher
    }

    /// A watcher holds the controller's rows after a snapshot in pages and
    /// after each kind of change the tail tells, and its next request asks
    /// for the tail after the last place it took. Its worker objects state
    /// the same rows when the snapshot's last page and each change return.
    #[test]
    fn a_watcher_follows_a_fleet_from_a_snapshot_through_every_kind_of_change() {
        let mut told = Told::new();
        let a = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.1:8000", 1)).id();
        let b = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.2:8000", 1)).id();
        let mut watcher = watching(&told, 1);
        let now = Instant::now();
        let mut follows =
            |told: &mut Told, change: MembershipChange, before: Option<&WorkerRow>| {
                let event = told.event(change, before);
                assert_eq!(watcher.take(&event, now), Ok(()), "{change:?}");
                assert_eq!(held(watcher.store()), held(&told.store), "{change:?}");
                laid(&watcher);
            };

        let change = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.3:8000", 1));
        follows(&mut told, change, None);

        let revision = told.row(a).revision;
        let change = WorkersStore::set_status(&told.store, a, WorkerStatus::Ready, revision);
        follows(&mut told, change.expect("a status change"), None);

        let patch = WorkerPatch {
            labels: Some(BTreeMap::from([("zone".to_owned(), "a".to_owned())])),
            priority: None,
            overload: None,
        };
        let revision = told.row(b).revision;
        let change = WorkersStore::patch(&told.store, b, patch, revision);
        follows(&mut told, change.expect("a patch"), None);

        let old = told.row(a);
        let change = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.1:8000", 2));
        assert!(matches!(change, MembershipChange::Replaced { .. }));
        follows(&mut told, change, Some(&old));

        let last = told.row(b);
        let change = WorkersStore::remove(&told.store, b, last.revision);
        follows(&mut told, change.expect("a removal"), Some(&last));

        // A generation's needs write no row, and their place is taken.
        told.seq += 2;
        let needs = Event::Needs(proto::NeedsSummary::default());
        let needs = worker_wire::event(needs, told.seq, WorkersStore::epoch(&told.store), TERM);
        assert_eq!(watcher.take(&needs, now), Ok(()));
        assert_eq!(held(watcher.store()), held(&told.store));

        assert_eq!(watcher.snapshots(), 1, "the tail alone behind one snapshot");
        let request = watcher.request();
        assert_eq!((request.resume_seq, request.resume_term), (told.seq, TERM));
        assert!(request.models.is_empty(), "every model");
    }

    /// An event that does not land where the store stands loses the
    /// watcher its place, and the next snapshot restores it. The event
    /// lays no object, and the snapshot keeps the object of a row that
    /// stands through it.
    #[test]
    fn a_change_that_does_not_land_loses_the_place_and_a_snapshot_restores_it() {
        let mut told = Told::new();
        let first = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.1:8000", 1)).id();
        let mut watcher = watching(&told, 8);
        let object = |watcher: &Watcher<Members>| {
            let objects = watcher.objects();
            let id = objects.object(first).expect("the first row's object");
            objects.registry().get(&id).expect("it stands")
        };
        let stood = object(&watcher);
        let now = Instant::now();
        // The watcher is not told this one.
        let change = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.2:8000", 1));
        let _untold = told.event(change, None);
        let change = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.3:8000", 1));
        let event = told.event(change, None);
        assert_eq!(watcher.take(&event, now), Err(WatchPositionLost::Apart));
        assert_eq!(watcher.at(), None);
        laid(&watcher);
        assert_eq!(watcher.request().resume_seq, 0);
        for page in told.pages(2) {
            assert_eq!(watcher.take(&page, now), Ok(()));
        }
        assert_eq!(held(watcher.store()), held(&told.store));
        laid(&watcher);
        assert!(
            Arc::ptr_eq(&stood, &object(&watcher)),
            "the object that stood"
        );
        assert_eq!(watcher.request().resume_seq, told.seq);
        assert_eq!(watcher.snapshots(), 2);
    }

    #[test]
    fn a_lease_filed_between_watch_snapshots_stays_on_file() {
        let told = Told::new();
        let id = told.store.upsert(facts("grpc://10.0.0.1:8000", 1)).id();
        let mut watcher = watching(&told, 8);
        let store = Arc::clone(watcher.store());
        let leases: &dyn LeasesStore = &*store;
        let (gateway, now) = (GatewayId(1), Instant::now());
        let lease = LeaseId(1);
        leases
            .issue(LeaseRecord {
                lease: Lease {
                    id: lease,
                    ..Lease::default()
                },
                gateway,
                request: 1,
                target: id,
                picked: PickRecord::default(),
                decode: None,
                untold: false,
                term: 0,
                issued_at: now,
                dispatch_deadline: now + Duration::from_secs(1),
                state: LeaseState::ISSUED,
            })
            .expect("filed");
        assert_eq!(leases.live(gateway), 1);

        let word = worker_wire::start_again(Position::default());
        assert_eq!(watcher.take(&word, now), Err(WatchPositionLost::Required));
        assert_eq!(watcher.request().resume_seq, 0);
        assert_eq!(watcher.take(&told.pages(8)[0], now), Ok(()));
        assert_eq!(watcher.snapshots(), 2);
        assert_eq!(leases.live(gateway), 1);
        assert!(leases.get(gateway, lease).is_some());
    }

    /// The controller's word, an event of the tail with no snapshot behind
    /// it or inside one, a page of another snapshot or at a place held, and
    /// an event of no known kind each lose the place.
    #[test]
    fn the_controllers_word_and_an_event_out_of_its_order_lose_the_place() {
        let mut told = Told::new();
        WorkersStore::upsert(&told.store, facts("grpc://10.0.0.1:8000", 1));
        WorkersStore::upsert(&told.store, facts("grpc://10.0.0.2:8000", 1));
        let now = Instant::now();
        let pages = told.pages(1);
        let change = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.3:8000", 1));
        let event = told.event(change, None);

        let mut watcher = over(Arc::new(MemoryStore::new()));
        assert_eq!(
            watcher.take(&event, now),
            Err(WatchPositionLost::Order),
            "no snapshot"
        );
        assert_eq!(watcher.request().resume_seq, 0);
        assert_eq!(watcher.take(&pages[0], now), Ok(()));
        assert_eq!(
            watcher.take(&event, now),
            Err(WatchPositionLost::Order),
            "inside one"
        );
        // A page of a snapshot at another place, and one of another epoch.
        let (seq, epoch) = (pages[0].seq, pages[0].membership_epoch);
        for (seq, epoch) in [(seq + 1, epoch), (seq, epoch + 1)] {
            assert_eq!(watcher.request().resume_seq, 0);
            assert_eq!(watcher.take(&pages[0], now), Ok(()));
            let page = proto::WorkerSnapshot {
                workers: Vec::new(),
                last: true,
            };
            let other = worker_wire::event(Event::Snapshot(page), seq, epoch, TERM);
            assert_eq!(
                watcher.take(&other, now),
                Err(WatchPositionLost::Order),
                "another's"
            );
        }
        // A page at a place held.
        let mut watcher = watching(&told, 8);
        let again = told.pages(8);
        assert_eq!(
            watcher.take(&again[0], now),
            Err(WatchPositionLost::Order),
            "held"
        );
        // An event of a kind this build does not know.
        let mut watcher = watching(&told, 8);
        let unknown = proto::WorkerEvent {
            seq: told.seq + 1,
            term: TERM,
            ..proto::WorkerEvent::default()
        };
        let lost = watcher.take(&unknown, now);
        assert!(
            matches!(lost, Err(WatchPositionLost::Unread { path: "event", .. })),
            "{lost:?}"
        );

        // The answer of a registration that states several ranks rides no
        // watch: a watcher that meets it lets the stream go by its name.
        let mut watcher = watching(&told, 8);
        let ranks = worker_wire::event(
            Event::Ranks(proto::RankRows::default()),
            told.seq + 1,
            WorkersStore::epoch(&told.store),
            TERM,
        );
        assert_eq!(
            watcher.take(&ranks, now),
            Err(WatchPositionLost::Unread {
                path: "membership.ranks",
                why: "a registration's answer rides no watch",
            })
        );
        assert_eq!(watcher.at(), None);

        let mut watcher = watching(&told, 8);
        let word = worker_wire::start_again(Position::default());
        assert_eq!(watcher.take(&word, now), Err(WatchPositionLost::Required));
        assert_eq!(watcher.at(), None);
        assert_eq!(held(watcher.store()), held(&told.store), "the rows stand");
    }

    /// A stream cut inside a snapshot starts the snapshot again, and a row
    /// or a change the conversion refuses loses the place.
    #[test]
    fn a_cut_snapshot_starts_again_and_a_refused_row_loses_the_place() {
        let mut told = Told::new();
        WorkersStore::upsert(&told.store, facts("grpc://10.0.0.1:8000", 1));
        WorkersStore::upsert(&told.store, facts("grpc://10.0.0.2:8000", 1));
        let now = Instant::now();
        let mut watcher = over(Arc::new(MemoryStore::new()));
        assert_eq!(watcher.take(&told.pages(1)[0], now), Ok(()));
        assert_eq!(watcher.request().resume_seq, 0);
        let change = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.3:8000", 1));
        let _told_to_others = told.event(change, None);
        for page in told.pages(2) {
            assert_eq!(watcher.take(&page, now), Ok(()));
        }
        assert_eq!(held(watcher.store()), held(&told.store));

        let blank = proto::WorkerSnapshot {
            workers: vec![proto::WorkerRow::default()],
            last: true,
        };
        let page = worker_wire::event(Event::Snapshot(blank), told.seq + 1, 9, TERM);
        let mut watcher = watching(&told, 8);
        let word = worker_wire::start_again(Position::default());
        assert_eq!(watcher.take(&word, now), Err(WatchPositionLost::Required));
        assert_eq!(watcher.request().resume_seq, 0);
        let lost = watcher.take(&page, now);
        assert!(
            matches!(lost, Err(WatchPositionLost::Unread { .. })),
            "{lost:?}"
        );
        assert_eq!(held(watcher.store()), held(&told.store), "the rows stand");
        let later = told.pages(8);
        assert_eq!(
            watcher.take(&later[0], now),
            Err(WatchPositionLost::Order),
            "let go"
        );

        let mut watcher = watching(&told, 8);
        let blank = Event::Registered(proto::WorkerRow::default());
        let blank = worker_wire::event(blank, told.seq + 1, 9, TERM);
        let lost = watcher.take(&blank, now);
        assert!(
            matches!(lost, Err(WatchPositionLost::Unread { .. })),
            "{lost:?}"
        );
        assert_eq!(watcher.at(), None);
        assert_eq!(held(watcher.store()), held(&told.store), "the rows stand");
    }

    /// A stream that lost the place is let go: the pages it goes on to
    /// tell are refused, so no part of a snapshot changes the store, and
    /// the stream the next request opens is taken from its first page.
    #[test]
    fn a_stream_that_lost_the_place_tells_nothing_more() {
        let told = Told::new();
        for n in 1..=3 {
            WorkersStore::upsert(&told.store, facts(&format!("grpc://10.0.0.{n}:8000"), 1));
        }
        let now = Instant::now();
        let mut watcher = over(Arc::new(MemoryStore::new()));
        WorkersStore::upsert(&told.store, facts("grpc://10.0.0.4:8000", 1));
        let before = held(watcher.store());
        let pages = told.pages(1);
        let blank = proto::WorkerSnapshot {
            workers: vec![proto::WorkerRow::default()],
            last: false,
        };
        let (seq, epoch) = (pages[0].seq, pages[0].membership_epoch);
        let blank = worker_wire::event(Event::Snapshot(blank), seq, epoch, TERM);
        assert_eq!(watcher.take(&pages[0], now), Ok(()));
        let lost = watcher.take(&blank, now);
        assert!(
            matches!(lost, Err(WatchPositionLost::Unread { .. })),
            "{lost:?}"
        );
        for page in &pages[2..] {
            assert_eq!(watcher.take(page, now), Err(WatchPositionLost::Order));
        }
        assert_eq!(watcher.at(), None);
        assert_eq!(held(watcher.store()), before, "the rows stand");

        assert_eq!(watcher.request().resume_seq, 0);
        for page in &pages {
            assert_eq!(watcher.take(page, now), Ok(()));
        }
        assert_eq!(held(watcher.store()), held(&told.store));
    }

    /// The watcher loses its place when the store's journal refuses a
    /// snapshot. The store keeps its rows.
    #[test]
    fn a_snapshot_a_stores_journal_refuses_loses_the_place() {
        let told = Told::new();
        WorkersStore::upsert(&told.store, facts("grpc://10.0.0.1:8000", 1));
        let store = Arc::new(MemoryStore::journaled(Arc::new(Log::new(8))));
        let mut watcher = over(Arc::clone(&store));
        // A place under no term, which a log takes from nobody.
        let mut page = told.pages(8).remove(0);
        page.term = 0;
        assert_eq!(
            watcher.take(&page, Instant::now()),
            Err(WatchPositionLost::Apart)
        );
        assert_eq!(watcher.at(), None);
        assert_eq!(held(&store).1, Vec::new(), "the store stays empty");
        assert!(watcher.objects().is_empty(), "and no object is laid");
    }

    /// A watcher that followed one registration behind its snapshot, with
    /// the controller's side and that event.
    fn followed() -> (Told, Watcher<Members>, proto::WorkerEvent) {
        let mut told = Told::new();
        WorkersStore::upsert(&told.store, facts("grpc://10.0.0.1:8000", 1));
        let mut watcher = watching(&told, 8);
        let change = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.2:8000", 1));
        let event = told.event(change, None);
        assert_eq!(watcher.take(&event, Instant::now()), Ok(()));
        (told, watcher, event)
    }

    /// An event of the tail at a place not past the last loses the place:
    /// the last event told again, a generation's needs behind it, and an
    /// event under a term below the place's. So does a change at a new
    /// place stamped with the epoch the store holds already: it moves no
    /// epoch, so it did not land.
    #[test]
    fn a_place_not_past_the_last_and_an_epoch_held_already_lose_the_place() {
        let now = Instant::now();
        let lost =
            |watcher: &mut Watcher<Members>, event: &proto::WorkerEvent, by: WatchPositionLost| {
                let rows = held(watcher.store());
                assert_eq!(watcher.take(event, now), Err(by));
                assert_eq!(watcher.at(), None);
                assert_eq!(held(watcher.store()), rows, "the rows stand");
                laid(watcher);
                assert_eq!(watcher.request().resume_seq, 0);
            };

        let (_, mut watcher, event) = followed();
        lost(&mut watcher, &event, WatchPositionLost::Order);

        let (told, mut watcher, _) = followed();
        let epoch = WorkersStore::epoch(&told.store);
        let needs = Event::Needs(proto::NeedsSummary::default());
        let needs = worker_wire::event(needs, told.seq - 1, epoch, TERM);
        lost(&mut watcher, &needs, WatchPositionLost::Order);

        let (mut told, mut watcher, _) = followed();
        let change = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.3:8000", 1));
        let mut older = told.event(change, None);
        older.term = TERM - 1;
        lost(&mut watcher, &older, WatchPositionLost::Order);

        let (_, mut watcher, event) = followed();
        let mut again = event;
        again.seq += 1;
        lost(&mut watcher, &again, WatchPositionLost::Apart);
    }
}
