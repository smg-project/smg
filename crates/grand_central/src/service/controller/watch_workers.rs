//! `WatchWorkers` (design 5.2): the store's membership as one history. A
//! watcher is sent the rows as one write of the log left them, then every
//! membership and status event after that write in the log's order, or it
//! is told to start again: never a gap, never a replay.
//!
//! No writer waits for a watcher. A stream has no task and no queue of its
//! own: the transport polls it, on the task of its call, and each poll
//! reads the stream's place. It starts from the cut of its place (taken
//! once at a place, from the worker objects in their holder's turn, and
//! shared), then reads the log, at most [`HELD`] entries a read, when
//! a watched write landed or the role's word was told, never for a load
//! row. An event is built when the transport takes it: what a watcher has
//! not taken waits in the log, and one the log's ring let go is told to
//! start again. A watcher that takes nothing for [`STALL`] after an event
//! is ended at its stream's next poll: no timer runs for a watcher. A
//! stream ends when its replica stops leading under the watch's term: the
//! role's word and the fence are read before the cut is taken and before
//! each page and each entry is handed over.
//!
//! No write wakes a watcher either. A stream the log holds nothing for
//! parks its waker with the feed. A watched write wakes one task of the
//! feed, once its store has let its locks go, and that task wakes the
//! parked streams, as it does at the role's word: a write pays one wake
//! however many watch, and none of it under the membership lock. The task
//! is the feed's. It is started with the feed's first watch, on that
//! watch's runtime, and ended with the last feed; the serve does not tell
//! it the stop and the drain does not join it, since it holds nothing a
//! drain waits on. A write on a feed whose watches have all left still
//! wakes it once, outside the locks.
//!
//! A watch of some models is told a write when a row it names serves one of
//! them: the row the write left in the table, or the row it took out. A
//! row's models change with its incarnation at the door, so a row leaves a
//! watcher's models by a replacement, which names both rows. A status
//! entry names no row. It holds the models its row served at the write,
//! and the same rule reads them.
//!
//! A membership entry the wire has no form for is told to no watch. The
//! store writes none; one that lands is logged once a replica.
//!
//! No peer sets a size here. A principal holds at most [`WATCHES_MAX`]
//! watches open at a replica, and as many principals watch a replica as may
//! be present at a leader: an open past either bound is refused by name. A
//! watch's seat is its stream's, given back when the transport has let the
//! stream go: a watcher that takes nothing keeps its stream, and its seat
//! with it, so the ceiling of watches bounds what a principal leaves
//! unread.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, Waker},
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use prost::encoding::message;
use rustc_hash::FxHashMap;
use tokio::{
    sync::{watch, Notify},
    task::JoinSet,
    time,
};
use tonic::{codegen::tokio_stream::Stream, Status};
use tracing::{info, warn};

use super::{principal::Principal, worker_wire};
pub(super) use crate::document::operator::snapshot::PAGE_BYTES;
use crate::{
    document::operator::{
        snapshot::{serves, serves_models},
        SerialTurn,
    },
    service::{
        elector::Fence,
        proto,
        replication::{Entry, Event, Log, ReadStamp},
    },
    store::{GatewayId, MemoryStore, Position, StoreCheckpoint, WorkerRow},
    worker_host::{Members, Objects},
};

/// The rows of one snapshot page.
const PAGE: usize = 128;

/// The log entries a stream takes at one read: what it holds of the log
/// until its watcher took them.
const HELD: usize = 64;

/// How long a watcher may take nothing after an event. No timer runs for
/// it: the bound is read when the transport next polls the stream, which
/// ends there, and the watcher resumes by the place of the last event it
/// took.
const STALL: Duration = Duration::from_secs(30);

/// The watches one principal holds open at a replica. A gateway follows a
/// replica with one watch and holds one more across a reconnect: twice
/// that.
pub(crate) const WATCHES_MAX: usize = 4;

/// What a watch reads: the store's log, and where a stream waits for it to
/// move.
#[derive(Clone, Debug)]
pub(crate) struct Feed {
    pub(crate) log: Arc<Log>,
    /// The wake of the feed's one task: one a watched write, issued once
    /// the store has let the write's locks go.
    landed: Arc<Notify>,
    /// The streams that wait on the log: that task wakes them.
    parked: Arc<Mutex<Parked>>,
    /// The feed's one task ([`Feed::start`]), ended with the last feed.
    teller: Arc<Mutex<JoinSet<()>>>,
    /// The place of the last membership entry the wire had no form for
    /// ([`Feed::untold`]): such an entry is logged once a replica.
    unwired: Arc<Mutex<Position>>,
    /// The rows at one place of the log: the cut the first read took there,
    /// kept for every other that starts there ([`Feed::rows_at_head`]). It
    /// is let go when a write moves the log past that place, when the last
    /// watch leaves, and by the last snapshot call out with no watch open:
    /// a copy of every row is held only while a reader can share it.
    rows: Arc<Mutex<Option<Arc<StoreCheckpoint>>>>,
}

impl Feed {
    /// The feed of `log`, whose watched writes wake the feed's one task
    /// through `landed`, each once its store has let its locks go: one wake
    /// of one task, whatever the count of watches, and while that task is
    /// busy or not started the wake is kept, once. The log's listener is
    /// made where the log is, beside the store that journals to it
    /// ([`journaled_at`](super::replication::journaled_at)).
    pub(crate) fn of(log: Arc<Log>, landed: Arc<Notify>) -> Self {
        Self {
            log,
            landed,
            parked: Arc::default(),
            teller: Arc::default(),
            unwired: Arc::default(),
            rows: Arc::default(),
        }
    }

    /// Starts the feed's one task where none lives: with the feed's first
    /// watch, on that watch's runtime, and again with the next watch where
    /// the task has ended (its runtime went away). A watch that lives on
    /// another runtime is woken by no write from that end until that next
    /// watch. The task wakes the parked streams after each watched write
    /// and at each word of `role`, off the writer and under no lock of the
    /// store: of the feed a write wakes this task alone, however many
    /// watch.
    ///
    /// The task takes nothing of a watch. `role` is the replica's word, one
    /// for every watch: the watch that starts the task hands it a handle of
    /// its own.
    fn start(&self, mut role: watch::Receiver<Option<u64>>) {
        let mut teller = self.teller.lock();
        // A live task alone counts. The set keeps one that ended until it
        // is joined: those are taken out first.
        while teller.try_join_next().is_some() {}
        if !teller.is_empty() {
            return;
        }
        let (landed, parked) = (Arc::clone(&self.landed), Arc::clone(&self.parked));
        teller.spawn(async move {
            // The wakers of one turn, and whether the role is told still.
            let (mut woken, mut told) = (Vec::new(), true);
            loop {
                tokio::select! {
                    () = landed.notified() => {}
                    word = role.changed(), if told => told = word.is_ok(),
                }
                // Taken out under the list's lock and woken outside it: a
                // stream that parks meanwhile stands behind no wake.
                parked.lock().take(&mut woken);
                for waker in woken.drain(..) {
                    waker.wake();
                }
            }
        });
    }

    /// A slot for one more stream among the feed's parked streams. The
    /// stream parks its waker there ([`Slot::park`]), and the slot is the
    /// next stream's once this one lets it go.
    fn slot(&self) -> Slot {
        let parked = Arc::clone(&self.parked);
        let at = parked.lock().slot();
        Slot { parked, at }
    }

    /// Lets the kept cut go once a write has moved the log past its place:
    /// no watch that starts now shares it. The lock is tried and never
    /// waited on: a watch that holds it is taking or sharing a cut, and a
    /// tail parks no thread of the runtime behind that copy of the rows. A
    /// stale cut it leaves goes at the next write, or with the last seat.
    fn let_go(&self) {
        let Some(mut kept) = self.rows.try_lock() else {
            return;
        };
        if kept.as_ref().is_some_and(|cut| cut.at != self.log.head()) {
            *kept = None;
        }
    }

    /// Lets the kept cut go whatever its place: no reader is left to share
    /// it. The feed's readers call it, at the last seat given back and at
    /// the last snapshot call out with no watch open ([`Readers`]). It
    /// waits for the lock, where [`Feed::let_go`] tries it and lets go a
    /// cut the log has passed and no other.
    fn keep_none(&self) {
        *self.rows.lock() = None;
    }

    /// Whether `other` is this feed: a clone of a feed shares its kept cut,
    /// and a feed made apart keeps one of its own.
    fn shares_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.rows, &other.rows)
    }

    /// The rows at the log's last watched write. A cut is a write's, not a
    /// reader's: the first read that starts at a place cuts it from
    /// `objects`, in their holder's turn for the copy
    /// ([`Objects::checkpoint`]), and every read that starts there shares
    /// it: a watch's start, and the operators' snapshot. Reads that come
    /// together or in a loop take that turn once a write, and the feed
    /// keeps one copy of the rows while a reader can share it
    /// ([`Feed::let_go`], [`Readers`]). The kept cut is named by its place
    /// with its term, never by a sequence alone: a restore can leave one
    /// sequence over other rows.
    fn rows_at_head(
        &self,
        objects: &Objects<Members>,
        store: &MemoryStore,
    ) -> Arc<StoreCheckpoint> {
        let mut kept = self.rows.lock();
        let head = self.log.head();
        if let Some(cut) = kept.as_ref().filter(|cut| cut.at == head) {
            return Arc::clone(cut);
        }
        let cut = Arc::new(objects.checkpoint(store));
        *kept = Some(Arc::clone(&cut));
        cut
    }

    /// The kept cut where it stands at the log's last watched write: what a
    /// read shares with no copy. None where a write moved the log past it,
    /// and where none is kept.
    fn kept_at_head(&self) -> Option<Arc<StoreCheckpoint>> {
        let kept = self.rows.lock();
        let head = self.log.head();
        kept.as_ref().filter(|cut| cut.at == head).cloned()
    }

    /// A membership entry the wire has no form for: no watch is told it,
    /// so a watcher's rows miss that write. The store writes none. One
    /// that lands is logged once a replica, with its change, by the first
    /// stream that comes to it ([`Asked::tells`]), whatever the count of
    /// streams that read it.
    ///
    /// The mark orders places by term, then by number: an entry of a later
    /// term is logged where a restart of the log left its number under the
    /// mark's. An entry under no term after a marked term is not.
    fn untold(&self, entry: &Entry) {
        let Event::Member { change, .. } = &entry.event else {
            return;
        };
        let at = entry.at;
        {
            let mut last = self.unwired.lock();
            if (at.term, at.seq) <= (last.term, last.seq) {
                return;
            }
            *last = at;
        }
        warn!(
            seq = at.seq,
            term = at.term,
            ?change,
            "a membership entry has no wire form: no watch is told it"
        );
    }
}

/// Where a watch starts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Start {
    /// The rows as the log's last watched write left them, then the tail.
    Snapshot,
    /// The tail after the place of the last event the watcher took.
    After(Position),
}

/// The start a request asks for, or the refusal of a field by its name.
pub(crate) fn start(request: &proto::WatchWorkersRequest) -> Result<Start, Status> {
    if request.with_needs {
        return Err(Status::invalid_argument(
            "with_needs: no generation's needs are told by this build",
        ));
    }
    // The list is kept for the stream's life and read at each row: it is
    // held to a row's own bounds before a seat is taken.
    worker_wire::watched(&request.models).map_err(|refused| {
        Status::invalid_argument(format!("{}: {}", refused.path, refused.why))
    })?;
    match (request.resume_seq, request.resume_term) {
        (0, 0) => Ok(Start::Snapshot),
        (0, _) => Err(Status::invalid_argument(
            "resume_term: a term comes with the position it was written under",
        )),
        (seq, term) => Ok(Start::After(Position { seq, term })),
    }
}

/// What a stream reads.
#[derive(Debug)]
pub(crate) struct Source {
    pub(crate) store: Arc<MemoryStore>,
    /// The holder of the replica's worker objects: a watch's first copy
    /// is cut from them.
    pub(crate) objects: Arc<Objects<Members>>,
    pub(crate) feed: Feed,
    pub(crate) fence: Arc<Fence>,
    /// The term this replica leads, if it leads one. A term is told at its
    /// start; that the replica leads none is told again at every move of
    /// the serve, since the word a stream found at its start may be the one
    /// before the lead it was admitted under. A stream ends at the word
    /// that this replica does not lead, whatever the word before it was.
    pub(crate) role: watch::Receiver<Option<u64>>,
    /// The term the watch was admitted under.
    pub(crate) term: u64,
}

impl Source {
    /// Whether this replica leads under the watch's term at this instant:
    /// the fence alone. The open was admitted where the store's log led
    /// that term. The log itself would follow a later term's event or a
    /// snapshot's place; while the fence holds the term no other replica
    /// leads a later one, so there is no leader to send either.
    fn leads(&self) -> bool {
        self.fence.leader_term(Instant::now()) == Some(self.term)
    }

    /// The rows a watch starts from: the feed's cut at the log's last
    /// watched write ([`Feed::rows_at_head`]).
    fn rows(&self) -> Arc<StoreCheckpoint> {
        self.feed.rows_at_head(&self.objects, &self.store)
    }
}

type Told = Result<proto::WorkerEvent, Status>;

/// The streams that wait on the log: a slot a stream, which holds its
/// waker while it is parked. The slot of a stream that is gone is the next
/// stream's: the list is as long as the most watches open at once.
#[derive(Debug, Default)]
struct Parked {
    wakers: Vec<Option<Waker>>,
    free: Vec<usize>,
}

impl Parked {
    /// A slot for one more stream.
    fn slot(&mut self) -> usize {
        self.free.pop().unwrap_or_else(|| {
            self.wakers.push(None);
            self.wakers.len() - 1
        })
    }

    /// Parks `waker` in `slot`, for one wake of the feed's task.
    fn park(&mut self, slot: usize, waker: &Waker) {
        match self.wakers.get_mut(slot) {
            Some(Some(parked)) if parked.will_wake(waker) => {}
            Some(parked) => *parked = Some(waker.clone()),
            None => {}
        }
    }

    /// Gives `slot` back, with whatever it held.
    fn leave(&mut self, slot: usize) {
        if let Some(parked) = self.wakers.get_mut(slot) {
            *parked = None;
        }
        self.free.push(slot);
    }

    /// Takes every parked waker out, into `woken`.
    fn take(&mut self, woken: &mut Vec<Waker>) {
        woken.extend(self.wakers.iter_mut().filter_map(Option::take));
    }
}

/// One stream's slot among a feed's parked streams ([`Feed::slot`]). It is
/// given back once, when the stream that holds it is let go.
#[derive(Debug)]
struct Slot {
    parked: Arc<Mutex<Parked>>,
    at: usize,
}

impl Slot {
    /// Parks `waker` here, for one wake of the feed's task.
    fn park(&self, waker: &Waker) {
        self.parked.lock().park(self.at, waker);
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        self.parked.lock().leave(self.at);
    }
}

/// Where a stream stands.
#[derive(Debug)]
enum At {
    /// Before its first poll: the start its watcher asked for.
    Start(Start),
    /// In the snapshot: the cut, and the first of its rows not told yet.
    Snapshot {
        cut: Arc<StoreCheckpoint>,
        next: usize,
    },
    /// In the tail: the place of the last entry it took out of a read,
    /// and the stamp of its last read of the log.
    Tail(Position, ReadStamp),
    /// Told its last word, or ended with none: the next poll ends it.
    Ended,
}

/// One watcher's stream. It has no task and no queue of its own: the
/// transport polls it, and each poll reads the role's word and the fence,
/// then the stream's place of the log.
#[derive(Debug)]
pub(crate) struct Watching {
    /// The stream's slot among the feed's parked streams, given back when
    /// the transport lets the stream go. It is the first field, so it goes
    /// back ahead of what the stream read: a stream that is going is woken
    /// no more.
    slot: Slot,
    source: Source,
    asked: Asked,
    at: At,
    /// One read of the log, and the first of its entries not looked at.
    entries: Vec<Entry>,
    seen: usize,
    /// When the transport took the stream's last event, until the stream
    /// parks: a watcher that takes nothing for [`STALL`] from then is ended.
    handed: Option<time::Instant>,
    /// The watch's seat, given back when the transport lets the stream go.
    _seat: WatchPlace,
}

impl Watching {
    /// Whether the word was told, since the watch's open or the stream's
    /// last poll, that this replica does not lead; or nobody is left to
    /// tell it. The word ends the stream whatever the word before it was.
    fn left(&mut self) -> bool {
        match self.source.role.has_changed() {
            Ok(false) => false,
            Ok(true) => self.source.role.borrow_and_update().is_none(),
            Err(_) => true,
        }
    }

    /// The stream's last word: the next poll ends it.
    fn last(&mut self, word: Told) -> Poll<Option<Told>> {
        self.at = At::Ended;
        Poll::Ready(Some(word))
    }

    /// The word to a watcher that cannot continue from its place.
    fn start_again(&mut self) -> Poll<Option<Told>> {
        let floor = self.source.feed.log.floor();
        self.last(Ok(worker_wire::start_again(floor)))
    }
}

/// Who reads the rows of one feed and can share its kept cut: each
/// principal that holds a watch open, with how many, and the operators'
/// snapshot calls let in to their turn. A copy of every row is held only
/// while one of them can share it, and the readers let it go themselves:
/// at the last seat given back, and at the last snapshot call out with no
/// watch open. They are built over their feed, so the copy they let go is
/// the one their reads share.
///
/// It holds three bounds, and a peer sets none: [`WATCHES_MAX`] watches a
/// principal, the rows it was built for, and the snapshot calls that may
/// wait their turn.
#[derive(Debug)]
pub(crate) struct Readers {
    /// The feed these readers read: its kept cut is theirs to let go.
    feed: Feed,
    /// A principal's open watches, under the key a `Session` seats it by.
    /// No row stands at zero.
    held: Mutex<FxHashMap<GatewayId, usize>>,
    /// The ceiling of rows.
    principals: usize,
    /// The turn of the operators' snapshots: one is taken at a time, and a
    /// call let in is a reader until its place goes back.
    snapshots: SerialTurn,
}

/// One open watch's place in the table, held by its stream and given back
/// when the transport lets the stream go. The last place given back lets
/// the feed's kept cut go: no watch shares it.
#[derive(Debug)]
struct WatchPlace {
    readers: Arc<Readers>,
    of: GatewayId,
}

impl Readers {
    /// The readers of `feed`: a table of at most `principals` rows, and a
    /// turn that lets `waiting` snapshot calls wait behind the one taken.
    pub(crate) fn of(feed: Feed, principals: usize, waiting: usize) -> Self {
        Self {
            feed,
            held: Mutex::new(FxHashMap::default()),
            principals,
            snapshots: SerialTurn::new(waiting),
        }
    }

    /// A seat for one more watch of `by`, or the refusal by name. The table
    /// has no gauge yet: the refusal and its line are its signal.
    fn seat(self: &Arc<Self>, by: &Principal) -> Result<WatchPlace, Status> {
        let of = by.gateway();
        let mut held = self.held.lock();
        let open = held.get(&of).copied().unwrap_or(0);
        if open >= WATCHES_MAX {
            warn!(
                refused = %by.subject,
                ceiling = WATCHES_MAX,
                "a watch past its principal's ceiling of open watches: refused"
            );
            return Err(Status::resource_exhausted(format!(
                "this principal holds its ceiling of {WATCHES_MAX} open watches: end one, then watch again"
            )));
        }
        if open == 0 && held.len() >= self.principals {
            warn!(
                refused = %by.subject,
                ceiling = self.principals,
                "a watch would make a new row past the ceiling of watching principals: refused"
            );
            return Err(Status::resource_exhausted(format!(
                "this replica holds its ceiling of {} watching principals: watch again later",
                self.principals
            )));
        }
        held.insert(of, open + 1);
        Ok(WatchPlace {
            readers: Arc::clone(self),
            of,
        })
    }

    /// A snapshot call's turn, with its place among the readers: both are
    /// held until what this returns is dropped. None where as many calls
    /// wait as may. The last call out lets the feed's kept cut go where no
    /// watch is open to share it ([`Readers::unshared`]), whether its take
    /// ended or its caller left while it waited.
    pub(crate) async fn snapshot(self: &Arc<Self>) -> Option<impl Send + use<>> {
        let last = Arc::clone(self);
        self.snapshots.take_with(move || last.unshared()).await
    }

    /// The cut a snapshot's take shares, or none where the take reads the
    /// rows it may tell and keeps nothing: the cheapest road that keeps
    /// calls that come together at one copy a write. A cut kept at the
    /// log's head is shared with no copy. Where another call is let in or
    /// a watch is open the feed's cut is taken and kept: somebody can
    /// share it. A take of `every` row takes it too: it would copy every
    /// row anyway, and a call let in while it runs shares the copy.
    pub(crate) fn cut(
        &self,
        every: bool,
        objects: &Objects<Members>,
        store: &MemoryStore,
    ) -> Option<Arc<StoreCheckpoint>> {
        if let Some(cut) = self.feed.kept_at_head() {
            return Some(cut);
        }
        let shared = every || self.snapshots.let_in() > 1 || self.open();
        shared.then(|| self.feed.rows_at_head(objects, store))
    }

    /// Lets the feed's kept cut go where no watch is open: what the last
    /// snapshot call out does, once no other call is let in to share it.
    /// The last seat given back does the same ([`WatchPlace`]): a copy of every
    /// row is held only while a reader can share it. The table's lock is
    /// given back first, as there: a watch that opens now takes a cut of
    /// its own at worst.
    fn unshared(&self) {
        if !self.open() {
            self.feed.keep_none();
        }
    }

    /// Whether a watch holds a seat: a reader that can share a kept cut.
    fn open(&self) -> bool {
        !self.held.lock().is_empty()
    }

    /// Whether these are the readers of `feed`: a clone of a feed shares its
    /// kept cut, so a clone of the feed they were built over is theirs too.
    fn are_of(&self, feed: &Feed) -> bool {
        self.feed.shares_with(feed)
    }
}

impl Drop for WatchPlace {
    fn drop(&mut self) {
        let last = {
            let mut held = self.readers.held.lock();
            match held.get_mut(&self.of) {
                Some(open) if *open > 1 => *open -= 1,
                _ => {
                    held.remove(&self.of);
                }
            }
            held.is_empty()
        };
        // The table's lock is given back first: a watch that opens now
        // takes a cut of its own at worst.
        if last {
            self.readers.feed.keep_none();
        }
    }
}

/// Opens the watch `by` asked for: from `start`, of the rows that serve one
/// of `models`, or of every row when none is named. It takes one of the
/// principal's seats among `readers`, or is refused by name. The source
/// reads the feed those readers were built over, or the cut its watch
/// keeps would not be the one its seat lets go: a debug build stops here.
pub(crate) fn open(
    source: Source,
    readers: &Arc<Readers>,
    by: &Principal,
    start: Start,
    models: Vec<String>,
) -> Result<Watching, Status> {
    debug_assert!(
        readers.are_of(&source.feed),
        "a watch's source reads the feed its readers were built over"
    );
    let seat = readers.seat(by)?;
    // The role's word is the replica's, one for every watch: the feed's
    // task wakes the parked streams by it, and takes nothing else of this
    // watch.
    source.feed.start(source.role.clone());
    info!(
        by = %by.subject,
        ?start,
        models = models.len(),
        term = source.term,
        "watch opened"
    );
    let slot = source.feed.slot();
    Ok(Watching {
        source,
        asked: Asked { models },
        at: At::Start(start),
        entries: Vec::new(),
        seen: 0,
        slot,
        handed: None,
        _seat: seat,
    })
}

/// The models a watcher asked for; none is every model.
#[derive(Debug)]
struct Asked {
    models: Vec<String>,
}

impl Asked {
    fn serves(&self, row: &WorkerRow) -> bool {
        serves(&self.models, row)
    }

    /// Whether a watch comes to the write: a row it names serves an asked
    /// model. A membership entry that names no row has no wire form, and
    /// every watch comes to it: the first logs it ([`Feed::untold`]). A
    /// status entry is read by its kind: it holds the models its row
    /// serves, which is all the rule reads of a row.
    fn tells(&self, entry: &Entry) -> bool {
        match &entry.event {
            Event::Member { row, left, .. } => {
                (row.is_none() && left.is_none())
                    || [row, left]
                        .into_iter()
                        .flatten()
                        .any(|row| self.serves(row))
            }
            Event::Status { models, .. } => serves_models(&self.models, models),
            Event::Term | Event::Generation { .. } => false,
        }
    }

    /// The page of `cut` that starts at its row `from`, of the rows that
    /// serve: the page, the row after its last, and whether it ends the
    /// snapshot. An empty fleet is one empty page, which ends it.
    fn page(&self, cut: &StoreCheckpoint, from: usize) -> (Vec<proto::WorkerRow>, usize, bool) {
        // One row more than a page holds: whether a page follows.
        let mut rows = Vec::with_capacity(PAGE + 1);
        let mut after = Vec::with_capacity(PAGE + 1);
        for (index, row) in cut.rows.iter().enumerate().skip(from) {
            if self.serves(row) {
                rows.push(row);
                after.push(index + 1);
                if rows.len() > PAGE {
                    break;
                }
            }
        }
        let page = fill(rows.iter().copied(), worker_wire::row);
        let last = page.len() == rows.len();
        let next = match page.len().checked_sub(1).and_then(|at| after.get(at)) {
            Some(next) if !last => *next,
            _ => cut.rows.len(),
        };
        (page, next, last)
    }
}

/// The page that starts `rows`, each as `told` tells it: at most [`PAGE`]
/// of them while they take at most [`PAGE_BYTES`] on the wire. A page takes
/// its first row whatever it weighs: a row past a page's bytes goes alone.
/// A watch's snapshot and a standby's are paged here alike.
pub(super) fn fill<'a>(
    rows: impl IntoIterator<Item = &'a WorkerRow>,
    told: impl Fn(&WorkerRow) -> proto::WorkerRow,
) -> Vec<proto::WorkerRow> {
    fill_within(PAGE, rows, told)
}

/// [`fill`] under a count of the caller's: at most `most` of `rows` while
/// they take at most [`PAGE_BYTES`] on the wire, the first whatever it
/// weighs.
pub(super) fn fill_within<'a>(
    most: usize,
    rows: impl IntoIterator<Item = &'a WorkerRow>,
    told: impl Fn(&WorkerRow) -> proto::WorkerRow,
) -> Vec<proto::WorkerRow> {
    let mut page = Vec::new();
    let mut bytes = 0;
    for row in rows.into_iter().take(most) {
        let told = told(row);
        // A row in a page: its bytes under the page's tag and a length
        // (`WorkerSnapshot.workers`, `ReplicationSnapshot.workers`,
        // `Snapshot.workers`: a tag of one byte each).
        let framed = message::encoded_len(1, &told);
        if !page.is_empty() && bytes + framed > PAGE_BYTES {
            break;
        }
        bytes += framed;
        page.push(told);
    }
    page
}

impl Watching {
    /// Steps at `now` until the stream has something for its watcher: each
    /// step reads the watcher's bound, the role's word and the fence, then
    /// the stream's place. `Pending` where the log holds nothing for the
    /// stream: no waker is parked here ([`Watching::park`]).
    fn step(&mut self, now: time::Instant) -> Poll<Option<Told>> {
        loop {
            if matches!(self.at, At::Ended) {
                return Poll::Ready(None);
            }
            // The transport took an event and did not come back for the
            // next: its watcher takes nothing. The stream ends with no
            // error, and the watcher resumes after the last event it took.
            if self
                .handed
                .is_some_and(|handed| now.duration_since(handed) >= STALL)
            {
                self.at = At::Ended;
                return Poll::Ready(None);
            }
            // The word that this replica does not lead ends the stream
            // whatever the fence reads at this instant: a leader that
            // leaves says so before its fence closes. The fence is read
            // before the cut is taken and before each page and each entry
            // is handed over.
            if self.left() || !self.source.leads() {
                let status = Status::aborted("this replica no longer leads under the watch's term");
                return self.last(Err(status));
            }
            match &mut self.at {
                At::Ended => return Poll::Ready(None),
                At::Start(Start::After(place)) => {
                    let place = *place;
                    if !self.source.feed.log.holds(place) {
                        return self.start_again();
                    }
                    self.at = At::Tail(place, ReadStamp::default());
                }
                At::Start(Start::Snapshot) => {
                    self.at = At::Snapshot {
                        cut: self.source.rows(),
                        next: 0,
                    };
                }
                At::Snapshot { cut, next } => {
                    let (page, reached, last) = self.asked.page(cut, *next);
                    let (at, epoch) = (cut.at, cut.epoch);
                    if last {
                        self.at = At::Tail(at, ReadStamp::default());
                    } else {
                        *next = reached;
                    }
                    return Poll::Ready(Some(Ok(worker_wire::page(page, last, at, epoch))));
                }
                At::Tail(place, stamp) => {
                    while let Some(entry) = self.entries.get(self.seen) {
                        self.seen += 1;
                        *place = entry.at;
                        if !self.asked.tells(entry) {
                            continue;
                        }
                        // An entry with no wire form is told to no watcher
                        // and logged once a replica.
                        let Some(told) = worker_wire::member(entry) else {
                            self.source.feed.untold(entry);
                            continue;
                        };
                        return Poll::Ready(Some(Ok(told)));
                    }
                    self.seen = 0;
                    // A stream comes back to the log after each event and
                    // after each park. Where the log stands as its last
                    // read left it, the read takes nothing and no lock.
                    let log = &self.source.feed.log;
                    if log
                        .read_if_moved(stamp, *place, HELD, &mut self.entries)
                        .is_err()
                    {
                        return self.start_again();
                    }
                    if self.entries.is_empty() {
                        return Poll::Pending;
                    }
                    // The log moved past the kept cut's place: no watch
                    // that starts now shares it.
                    self.source.feed.let_go();
                }
            }
        }
    }
}

impl Stream for Watching {
    type Item = Told;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let now = time::Instant::now();
        match this.step(now) {
            Poll::Ready(told) => this.hand(told, now),
            Poll::Pending => this.park(cx.waker(), now),
        }
    }
}

impl Watching {
    /// Parks the stream's waker with the feed, then steps once more. A
    /// write that landed or a word that was told between the stream's last
    /// read and its park is found by that step, whether the feed's task
    /// took the parked wakers before the park or after it.
    fn park(&mut self, waker: &Waker, now: time::Instant) -> Poll<Option<Told>> {
        self.slot.park(waker);
        match self.step(now) {
            Poll::Ready(told) => self.hand(told, now),
            Poll::Pending => {
                self.handed = None;
                Poll::Pending
            }
        }
    }

    /// Hands the watcher `told`. An event starts the watcher's bound: the
    /// transport comes back for the next within [`STALL`], or the stream
    /// ends.
    fn hand(&mut self, told: Option<Told>, now: time::Instant) -> Poll<Option<Told>> {
        self.handed = matches!(told, Some(Ok(_))).then_some(now);
        Poll::Ready(told)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/service/controller/watch_workers/tests.rs"]
pub(crate) mod tests;
