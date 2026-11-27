//! A store that follows a controller's watch of its workers: a gateway's
//! side of `WatchWorkers` (design 5.2), for a host that decides against a
//! standalone controller and keeps what it was last told.
//!
//! A [`Watcher`] takes a snapshot's rows at its last page
//! ([`MemoryStore::watch_from`]). The store keeps its leases and session
//! pins, and the workers still present keep their gateway vetoes.
//! Each membership and status event after the snapshot lands as a
//! standby's applier lands one: through the holder of the worker objects
//! (below), by the body of [`MemoryStore::follow`]. A
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
//! store's write and the lay of the snapshot's rows in one turn of the
//! holder. So
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
    store::{MemoryStore, Position, StoreCheckpoint, WorkerRow},
    worker_host::Objects,
};

/// Why a watcher lost its place in the watch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
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
#[non_exhaustive]
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
    /// watch tells each status, and the holder lays it on the row's
    /// object: it needs no option for that.
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
                    // A change that does not fit writes nothing, and one
                    // at the epoch the store held already moves none: the
                    // holder says whether the store took it.
                    if !self.objects.follow_at(&self.store, at, epoch, change, row) {
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
#[path = "../../tests/unit/service/watcher/tests.rs"]
mod tests;
