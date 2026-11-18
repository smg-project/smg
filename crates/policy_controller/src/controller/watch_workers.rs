//! `WatchWorkers` (design 5.2): the store's membership as one history. A
//! watcher is sent the rows as one write of the log left them, then every
//! membership and status event after that write in the log's order, or it
//! is told to start again: never a gap, never a replay.
//!
//! No writer waits for a watcher. A stream has no task and no queue of its
//! own: the transport polls it, on the task of its call, and each poll
//! reads the stream's place. It starts from the cut of its place (taken
//! once at a place, under the membership lock for the copy of the rows,
//! and shared), then reads the log, at most [`HELD`] entries a read, when
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
//! watcher's models by a replacement, which names both rows.
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
use crate::{
    elector::Fence,
    proto,
    replication::{Entry, Event, Log, Stamp},
    store::{Cut, GatewayId, Kinds, MemoryStore, Position, WorkerRow},
};

/// The rows of one snapshot page.
const PAGE: usize = 128;

/// The bytes of one snapshot page's rows on the wire: half a frame of 4 MiB.
/// A page of [`PAGE`] rows at the door's bound of a row is cut here.
pub(super) const PAGE_BYTES: usize = 2 * 1024 * 1024;

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
    /// The rows at one place of the log: the cut the first watch took
    /// there, kept for every other that starts there. It is let go when a
    /// write moves the log past that place and when the last watch leaves:
    /// a copy of every row is held only while a watch can share it.
    rows: Arc<Mutex<Option<Arc<Cut>>>>,
}

impl Feed {
    /// The feed of `log`, whose watched writes wake the feed's one task,
    /// each once its store has let its locks go. The store that journals
    /// to the log is built beside it
    /// ([`journaled`](super::replication::journaled)).
    pub(crate) fn of(log: Log) -> Self {
        let landed = Arc::new(Notify::new());
        let log = {
            let landed = Arc::clone(&landed);
            // One wake of one task, whatever the count of watches. While
            // that task is busy or not started the wake is kept, once.
            Arc::new(log.on_watched(move || landed.notify_one()))
        };
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
    /// store: a write wakes this task alone, however many watch.
    ///
    /// The task takes nothing of a watch. `role` is the replica's word, one
    /// for every watch: the watch that starts the task hands it a handle of
    /// its own.
    fn start(&self, mut role: watch::Receiver<bool>) {
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

    /// Lets the kept cut go once a write has moved the log past its place:
    /// no watch that starts now shares it. The lock is tried and never
    /// waited on: a watch that holds it is taking or sharing a cut, and a
    /// tail parks no thread of the runtime behind that copy of the rows. A
    /// stale cut it leaves goes at the next write, or with the last seat.
    fn let_go(&self) {
        let Some(mut kept) = self.rows.try_lock() else {
            return;
        };
        if kept
            .as_ref()
            .is_some_and(|cut| cut.at != self.log.head(Kinds::Watched))
        {
            *kept = None;
        }
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
    pub(crate) feed: Feed,
    pub(crate) fence: Arc<Fence>,
    /// Whether this replica leads. That it does is told at the change; that
    /// it does not is told again at every move of the serve, since the word
    /// a stream found at its start may be the one before the lead it was
    /// admitted under. A stream ends at the word that this replica does not
    /// lead, whatever the word before it was.
    pub(crate) role: watch::Receiver<bool>,
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

    /// The rows at the log's last watched write. A cut is a write's, not a
    /// watcher's: the first watch that starts at a place takes it, under
    /// the membership lock for the copy, and every watch that starts there
    /// shares it. Watchers that open together or in a loop hold that lock
    /// once a write, and the feed keeps one copy of the rows while a watch
    /// can share it ([`Feed::let_go`], [`Seat`]). The kept cut is named by
    /// its place with its term, never by a sequence alone: a restore can
    /// leave one sequence over other rows.
    fn rows(&self) -> Arc<Cut> {
        let mut kept = self.feed.rows.lock();
        let head = self.feed.log.head(Kinds::Watched);
        if let Some(cut) = kept.as_ref().filter(|cut| cut.at == head) {
            return Arc::clone(cut);
        }
        let cut = Arc::new(self.store.cut(Kinds::Watched));
        *kept = Some(Arc::clone(&cut));
        cut
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

/// Where a stream stands.
#[derive(Debug)]
enum At {
    /// Before its first poll: the start its watcher asked for.
    Start(Start),
    /// In the snapshot: the cut, and the first of its rows not told yet.
    Snapshot { cut: Arc<Cut>, next: usize },
    /// In the tail: the place of the last entry it took out of a read,
    /// and the stamp of its last read of the log.
    Tail(Position, Stamp),
    /// Told its last word, or ended with none: the next poll ends it.
    Ended,
}

/// One watcher's stream. It has no task and no queue of its own: the
/// transport polls it, and each poll reads the role's word and the fence,
/// then the stream's place of the log.
#[derive(Debug)]
pub(crate) struct Watching {
    source: Source,
    asked: Asked,
    at: At,
    /// One read of the log, and the first of its entries not looked at.
    entries: Vec<Entry>,
    seen: usize,
    /// The stream's slot among the feed's parked streams.
    slot: usize,
    /// When the transport took the stream's last event, until the stream
    /// parks: a watcher that takes nothing for [`STALL`] from then is ended.
    handed: Option<time::Instant>,
    /// The watch's seat, given back when the transport lets the stream go.
    _seat: Seat,
}

impl Watching {
    /// Whether the word was told, since the watch's open or the stream's
    /// last poll, that this replica does not lead; or nobody is left to
    /// tell it. The word ends the stream whatever the word before it was.
    fn left(&mut self) -> bool {
        match self.source.role.has_changed() {
            Ok(false) => false,
            Ok(true) => !*self.source.role.borrow_and_update(),
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
        let floor = self.source.feed.log.floor(Kinds::Watched);
        self.last(Ok(worker_wire::start_again(floor)))
    }
}

/// Who watches this replica: each principal that holds a watch open, with
/// how many. It holds two bounds, and a peer sets neither: [`WATCHES_MAX`]
/// watches a principal, and the rows it was built for.
#[derive(Debug)]
pub(crate) struct Watchers {
    /// A principal's open watches, under the key a `Session` seats it by.
    /// No row stands at zero.
    held: Mutex<FxHashMap<GatewayId, usize>>,
    /// The ceiling of rows.
    principals: usize,
}

/// One open watch's seat in the table, held by its stream and given back
/// when the transport lets the stream go. The last seat given back lets
/// the feed's kept cut go: no watch shares it.
#[derive(Debug)]
struct Seat {
    watchers: Arc<Watchers>,
    of: GatewayId,
    kept: Arc<Mutex<Option<Arc<Cut>>>>,
}

impl Watchers {
    /// A table of at most `principals` rows.
    pub(crate) fn new(principals: usize) -> Self {
        Self {
            held: Mutex::new(FxHashMap::default()),
            principals,
        }
    }

    /// A seat for one more watch of `by`, or the refusal by name. The table
    /// has no gauge yet: the refusal and its line are its signal.
    fn seat(self: &Arc<Self>, by: &Principal, feed: &Feed) -> Result<Seat, Status> {
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
        Ok(Seat {
            watchers: Arc::clone(self),
            of,
            kept: Arc::clone(&feed.rows),
        })
    }

    /// The watches `by` holds open.
    #[cfg(test)]
    fn open_of(&self, by: &Principal) -> usize {
        self.held.lock().get(&by.gateway()).copied().unwrap_or(0)
    }
}

impl Drop for Seat {
    fn drop(&mut self) {
        let last = {
            let mut held = self.watchers.held.lock();
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
            *self.kept.lock() = None;
        }
    }
}

/// Opens the watch `by` asked for: from `start`, of the rows that serve one
/// of `models`, or of every row when none is named. It takes one of the
/// principal's seats among `watchers`, or is refused by name.
pub(crate) fn open(
    source: Source,
    watchers: &Arc<Watchers>,
    by: &Principal,
    start: Start,
    models: Vec<String>,
) -> Result<Watching, Status> {
    let seat = watchers.seat(by, &source.feed)?;
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
    let slot = source.feed.parked.lock().slot();
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
        self.models.is_empty()
            || row
                .facts
                .models
                .iter()
                .any(|model| self.models.contains(model))
    }

    /// Whether a watch comes to the write: a row it names serves an asked
    /// model. A membership entry that names no row has no wire form, and
    /// every watch comes to it: the first logs it ([`Feed::untold`]).
    fn tells(&self, entry: &Entry) -> bool {
        match &entry.event {
            Event::Member { row, left, .. } => {
                (row.is_none() && left.is_none())
                    || [row, left]
                        .into_iter()
                        .flatten()
                        .any(|row| self.serves(row))
            }
            Event::Load { .. } | Event::Term => false,
        }
    }

    /// The page of `cut` that starts at its row `from`, of the rows that
    /// serve: the page, the row after its last, and whether it ends the
    /// snapshot. An empty fleet is one empty page, which ends it.
    fn page(&self, cut: &Cut, from: usize) -> (Vec<proto::WorkerRow>, usize, bool) {
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
    let mut page = Vec::new();
    let mut bytes = 0;
    for row in rows.into_iter().take(PAGE) {
        let told = told(row);
        // A row in a page: its bytes under the page's tag and a length
        // (`WorkerSnapshot.workers`, `ReplicationSnapshot.workers`: a tag
        // of one byte each).
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
                    if !self.source.feed.log.holds(Kinds::Watched, place) {
                        return self.start_again();
                    }
                    self.at = At::Tail(place, Stamp::default());
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
                        self.at = At::Tail(at, Stamp::default());
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
        self.source.feed.parked.lock().park(self.slot, waker);
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

impl Drop for Watching {
    fn drop(&mut self) {
        self.source.feed.parked.lock().leave(self.slot);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        collections::BTreeMap,
        pin::Pin,
        sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc, Arc,
        },
        task::{Context, Poll, Wake, Waker},
        thread,
        time::{Duration, Instant},
    };

    use prost::encoding::message;
    use tokio::{runtime::Builder, sync::watch, time};
    use tonic::{
        codegen::tokio_stream::{Stream, StreamExt},
        Code, Status,
    };

    use super::{
        fill, open, start, Feed, Source, Start, Watchers, Watching, HELD, PAGE, PAGE_BYTES, STALL,
        WATCHES_MAX,
    };
    use crate::{
        controller::{
            election::LeasePlace,
            gateways::tests::PATIENT,
            membership::tests::Logged,
            principal::tests::told,
            replication::{journaled, journaled_from, FIRST_PLACES, WATCHED_EVENTS},
            worker_wire,
        },
        elector::{Elector, Tick},
        proto::{self, worker_event::Event},
        store::{
            Journal, Kinds, LeasesStore, LoadSink, LoadSource, MembershipChange, MemoryStore,
            Position, WorkerFacts, WorkerPatch, WorkerRow, WorkerStatus, WorkersStore,
        },
        view::{LoadRow, WorkerId},
    };

    /// A replica that leads under term 1, its store standing in that term.
    struct Leader {
        store: Arc<MemoryStore>,
        feed: Feed,
        elector: Elector,
        role: watch::Sender<bool>,
        /// Who watches it: a table of two principals.
        watchers: Arc<Watchers>,
    }

    fn leader() -> Leader {
        leading(journaled_from(&LeasePlace::Process, 0))
    }

    /// A replica over a store that takes the lease now: the store enters
    /// term 1 after whatever was written to it before. The candidate is
    /// alone over its lease, so each take of the kit leads at once.
    fn leading((store, feed): (Arc<MemoryStore>, Feed)) -> Leader {
        // A lease no case outlasts: the fence closes when a case lets the
        // lease go, never by the clock of a held host.
        let mut elector =
            Elector::alone("controller-0@127.0.0.1:1", PATIENT).expect("the settings are in order");
        assert_eq!(
            elector.tick(Instant::now()).expect("a tick"),
            Tick::Acquired { term: 1 }
        );
        store.set_term(1);
        let (role, _) = watch::channel(true);
        Leader {
            store,
            feed,
            elector,
            role,
            watchers: Arc::new(Watchers::new(2)),
        }
    }

    fn key(n: usize) -> String {
        format!("grpc://10.0.{}.{}:8000", n / 250, n % 250)
    }

    impl Leader {
        fn watch(&self, start: Start, models: &[&str]) -> Watching {
            self.watch_as("gateway-a", start, models)
                .expect("a seat for the watch")
        }

        fn watch_as(&self, by: &str, start: Start, models: &[&str]) -> Result<Watching, Status> {
            let source = Source {
                store: Arc::clone(&self.store),
                feed: self.feed.clone(),
                fence: self.elector.fence(),
                role: self.role.subscribe(),
                term: 1,
            };
            let models = models.iter().map(|model| (*model).to_owned()).collect();
            open(source, &self.watchers, &told(by), start, models)
        }

        /// `by`'s open watches read `want` at this instant: a stream's seat
        /// goes back at its drop itself, with no turn of the runtime.
        fn holds(&self, by: &str, want: usize) {
            assert_eq!(self.watchers.open_of(&told(by)), want, "{by}'s watches");
        }

        fn register(&self, n: usize, incarnation: u64, model: &str) -> WorkerId {
            self.store
                .upsert(WorkerFacts {
                    key: key(n),
                    engine: key(n),
                    incarnation,
                    models: vec![model.to_owned()],
                    dp_size: 1,
                    ..WorkerFacts::default()
                })
                .id()
        }

        /// A row whose URL alone passes half a page's bytes: it goes alone
        /// in a snapshot's page.
        fn register_wide(&self, n: usize) -> WorkerId {
            let url = format!("grpc://{n:03}.{}:8000", "u".repeat(PAGE_BYTES / 4));
            self.store
                .upsert(WorkerFacts {
                    key: url.clone(),
                    engine: url,
                    incarnation: 1,
                    models: vec!["model-a".to_owned()],
                    dp_size: 1,
                    ..WorkerFacts::default()
                })
                .id()
        }

        fn head(&self) -> Position {
            self.feed.log.head(Kinds::Watched)
        }
    }

    /// A waker that counts its wakes.
    #[derive(Default)]
    pub(crate) struct Woken(AtomicUsize);

    impl Wake for Woken {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl Woken {
        /// A counting waker, and its count.
        pub(crate) fn waker() -> (Arc<Self>, Waker) {
            let woken = Arc::new(Self::default());
            (Arc::clone(&woken), Waker::from(woken))
        }

        pub(crate) fn wakes(&self) -> usize {
            self.0.load(Ordering::SeqCst)
        }

        /// The count reads `want` within a bound of the runtime's turns:
        /// the feed's task wakes the parked streams at its next one.
        async fn reads(&self, want: usize) {
            for _ in 0..10_000 {
                if self.wakes() == want {
                    return;
                }
                tokio::task::yield_now().await;
            }
            assert_eq!(self.wakes(), want, "the wakes of the parked stream");
        }
    }

    /// One poll of `stream` under `waker`: what it has for its watcher at
    /// this instant, with no turn of the runtime.
    fn poll(
        stream: &mut Watching,
        waker: &Waker,
    ) -> Poll<Option<Result<proto::WorkerEvent, Status>>> {
        Pin::new(stream).poll_next(&mut Context::from_waker(waker))
    }

    /// A hundred turns of the runtime: every task that could run has run.
    /// A feed's task starts with one wake kept from the writes before its
    /// first watch, and spends it here.
    async fn turns() {
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
    }

    /// `stream` has nothing for its watcher at this instant: it parks
    /// `waker` with its feed.
    fn parks(stream: &mut Watching, waker: &Waker) {
        assert!(
            poll(stream, waker).is_pending(),
            "the stream had something for its watcher"
        );
    }

    /// The stream's next event, which the case expects within five seconds.
    async fn next(stream: &mut Watching) -> proto::WorkerEvent {
        time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("an event within five seconds")
            .expect("the stream is open")
            .expect("an event, not a status")
    }

    /// The status that ends the stream, which the case expects within five
    /// seconds.
    async fn last_word(stream: &mut Watching) -> Status {
        time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("a last word within five seconds")
            .expect("the stream is open")
            .expect_err("a status ends the stream")
    }

    #[test]
    fn a_request_names_its_start_or_is_refused_by_the_field() {
        let request = |resume_seq, resume_term, with_needs| proto::WatchWorkersRequest {
            models: Vec::new(),
            resume_seq,
            with_needs,
            resume_term,
        };
        assert_eq!(start(&request(0, 0, false)).ok(), Some(Start::Snapshot));
        assert_eq!(
            start(&request(7, 2, false)).ok(),
            Some(Start::After(Position { seq: 7, term: 2 }))
        );
        // A place under no term is asked of the log, which holds none.
        assert_eq!(
            start(&request(7, 0, false)).ok(),
            Some(Start::After(Position { seq: 7, term: 0 }))
        );
        for (refused, field) in [
            (request(0, 2, false), "resume_term: "),
            (request(0, 0, true), "with_needs: "),
        ] {
            let status = start(&refused).expect_err("refused");
            assert_eq!(status.code(), Code::InvalidArgument);
            assert!(status.message().starts_with(field), "{}", status.message());
        }
    }

    #[test]
    fn a_page_is_cut_at_its_bytes_as_well_as_at_its_rows() {
        // A row whose URL holds `bytes`: the wire tells a URL twice.
        fn row(bytes: usize) -> WorkerRow {
            WorkerRow {
                id: WorkerId(7),
                facts: WorkerFacts {
                    key: "u".repeat(bytes),
                    ..WorkerFacts::default()
                },
                status: WorkerStatus::Ready,
                revision: 1,
                admitted_at: Instant::now(),
            }
        }
        fn paged(fleet: &[WorkerRow]) -> Vec<usize> {
            let rows: Vec<&WorkerRow> = fleet.iter().collect();
            let (mut told, mut pages) = (0, Vec::new());
            loop {
                let page = fill(rows[told..].iter().copied(), worker_wire::row);
                assert!(told == rows.len() || !page.is_empty(), "a page takes a row");
                told += page.len();
                pages.push(page.len());
                if told == rows.len() {
                    break pages;
                }
            }
        }
        assert_eq!(paged(&[]), [0], "an empty fleet is one empty page");
        assert_eq!(paged(&vec![row(16); PAGE + 2]), [PAGE, 2]);
        // Rows past the door's bound of 16,384 bytes: a full page of them
        // would pass a page's bytes, so the page ends where they fit.
        let fat = row(8_192);
        let framed = message::encoded_len(1, &worker_wire::row(&fat));
        assert!(framed > 16_384 && PAGE * framed > PAGE_BYTES);
        let held = PAGE_BYTES / framed;
        assert_eq!(paged(&vec![fat; PAGE]), [held, PAGE - held]);
        // A row past a page's bytes goes alone.
        assert_eq!(paged(&[row(16), row(PAGE_BYTES), row(16)]), [1, 1, 1]);
    }

    #[tokio::test]
    async fn a_snapshot_is_paged_at_one_place_and_its_last_page_is_marked() {
        let leader = leader();
        // An empty fleet is one empty page, marked last, at the term's start.
        let mut stream = leader.watch(Start::Snapshot, &[]);
        let page = next(&mut stream).await;
        assert_eq!((page.seq, page.term, page.membership_epoch), (1, 1, 0));
        assert_eq!(
            page.event,
            Some(Event::Snapshot(proto::WorkerSnapshot {
                workers: Vec::new(),
                last: true,
            }))
        );
        drop(stream);

        for n in 0..PAGE + 2 {
            leader.register(n, 1, "model-a");
        }
        let at = leader.head();
        let mut stream = leader.watch(Start::Snapshot, &[]);
        let (first, second) = (next(&mut stream).await, next(&mut stream).await);
        for page in [&first, &second] {
            assert_eq!(
                (page.seq, page.term, page.membership_epoch),
                (at.seq, at.term, leader.store.epoch())
            );
        }
        let (Some(Event::Snapshot(first)), Some(Event::Snapshot(second))) =
            (first.event, second.event)
        else {
            panic!("a snapshot is pages");
        };
        assert_eq!((first.workers.len(), first.last), (PAGE, false));
        assert_eq!((second.workers.len(), second.last), (2, true));
        // Every row of the store once, by registration key.
        let (_, rows) = leader.store.snapshot();
        let mut keys: Vec<String> = rows.into_iter().map(|row| row.facts.key).collect();
        keys.sort();
        let paged: Vec<String> = first
            .workers
            .into_iter()
            .chain(second.workers)
            .map(|row| row.url)
            .collect();
        assert_eq!(paged, keys);
    }

    #[tokio::test]
    async fn the_tail_tells_every_write_after_the_cut_in_the_logs_order() {
        let leader = leader();
        let mut stream = leader.watch(Start::Snapshot, &[]);
        let cut = next(&mut stream).await;

        let id = leader.register(0, 1, "model-a");
        let registered = next(&mut stream).await;
        let Some(Event::Registered(row)) = &registered.event else {
            panic!("a registration: {registered:?}");
        };
        assert_eq!(
            (row.id, row.revision, row.status),
            (id.0, 1, proto::WorkerStatus::Pending as i32)
        );

        leader
            .store
            .set_status(id, WorkerStatus::Ready, 1)
            .expect("ready");
        let ready = next(&mut stream).await;
        assert_eq!(
            ready.event,
            Some(Event::Status(proto::StatusChanged {
                worker_id: id.0,
                old: proto::WorkerStatus::Pending as i32,
                new: proto::WorkerStatus::Ready as i32,
                health: proto::HealthFlag::Healthy as u32,
            }))
        );

        let labels = BTreeMap::from([("tier".to_owned(), "gold".to_owned())]);
        let patch = WorkerPatch {
            labels: Some(labels),
            ..WorkerPatch::default()
        };
        // The status change moved no revision: the patch names the one the
        // registration gave.
        leader.store.patch(id, patch, 1).expect("patched");
        let relabelled = next(&mut stream).await;
        let Some(Event::Relabelled(row)) = &relabelled.event else {
            panic!("a change in place: {relabelled:?}");
        };
        assert_eq!(
            (row.revision, row.labels.get("tier").map(String::as_str)),
            (2, Some("gold"))
        );

        leader.store.remove(id, 2).expect("removed");
        let removed = next(&mut stream).await;
        assert_eq!(
            removed.event,
            Some(Event::Removed(proto::Removed {
                worker_id: id.0,
                registration_key: key(0),
                revision: 2,
                draining: false,
            }))
        );

        // One position a write, in the store's order, under the term; the
        // epochs are the store's.
        let told = [&cut, &registered, &ready, &relabelled, &removed];
        assert!(told.windows(2).all(|pair| pair[1].seq == pair[0].seq + 1));
        assert!(told.iter().all(|event| event.term == 1));
        assert_eq!(
            told.map(|event| event.membership_epoch),
            [0, 1, 2, 3, 4],
            "an epoch a write"
        );
    }

    /// A membership entry the wire has no form for is told to no watch. The
    /// store writes none; one that lands is logged once a replica, whatever
    /// the count of streams that read it, and the tail goes on with the
    /// write after it. One that names no row is logged as one that names
    /// one is, and so is one of a later term whose number a restart of the
    /// log left under the last one logged.
    #[tokio::test]
    async fn a_membership_entry_with_no_wire_form_is_logged_once_and_told_to_no_watch() {
        // The case's own subscriber keeps this thread's lines, a second one
        // registered beside it, as in the door's audit case.
        let lines = Logged::default();
        let writer = lines.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let _kept = tracing::subscriber::set_default(subscriber);

        let leader = leader();
        let id = leader.register(0, 1, "model-a");
        let row = WorkersStore::get(leader.store.as_ref(), id).expect("the row");
        let mut streams = [
            leader.watch(Start::Snapshot, &[]),
            leader.watch(Start::Snapshot, &[]),
        ];
        for stream in &mut streams {
            next(stream).await;
        }
        // Two shapes the store never tells, written here by the journal's
        // own calls: a removal that names no row at all, and a replacement
        // that names no row it took out.
        let journal: &dyn Journal = leader.feed.log.as_ref();
        let logged = || {
            let text = String::from_utf8(lines.0.lock().clone()).expect("the lines as text");
            text.matches("has no wire form").count()
        };
        journal.member(None, 2, MembershipChange::Removed { id }, None, None);
        let replaced = MembershipChange::Replaced { old: id, new: id };
        journal.member(None, 3, replaced, Some(&row), None);
        journal.released();
        let unwired = leader.head();
        leader.register(1, 1, "model-a");
        for stream in &mut streams {
            let told = next(stream).await;
            assert_eq!(told.seq, unwired.seq + 1, "the write after the two entries");
        }
        assert_eq!(logged(), 2, "one line an entry, whatever the streams");

        // The log starts again at a lower number and the store takes a
        // later term: an entry there is under the last one logged by its
        // number and past it by its term.
        leader
            .feed
            .log
            .restart(Position { seq: 1, term: 2 })
            .expect("a place the log takes");
        leader.store.set_term(3);
        let mut later = leader.watch(Start::Snapshot, &[]);
        next(&mut later).await;
        journal.member(None, 4, MembershipChange::Removed { id }, None, None);
        journal.released();
        let again = leader.head();
        assert!(again.seq < unwired.seq && again.term == 3, "{again:?}");
        leader.register(2, 1, "model-a");
        let told = next(&mut later).await;
        assert_eq!(told.seq, again.seq + 1, "the write after the entry");
        assert_eq!(logged(), 3, "the entry of the later term");
    }

    /// Where the lease lives in the process a term is taken again at each
    /// start, and each run's log starts at a place of its own. A watcher
    /// that resumes at a place of the run before is no reader of this
    /// run's log, whether that place lies under this run's first or ahead
    /// of its head: it is told to start again at once, never held.
    #[tokio::test]
    async fn a_place_of_the_run_before_is_told_to_start_again() {
        for (earlier, later) in [(1_000, 9_000), (9_000, 1_000)] {
            let before = leading(journaled_from(&LeasePlace::Process, earlier));
            before.register(0, 1, "model-a");
            let held = before.head();
            let run = leading(journaled_from(&LeasePlace::Process, later));
            run.register(1, 1, "model-a");
            run.register(2, 1, "model-a");
            assert_eq!(held.term, run.head().term, "one term, two runs");
            let floor = run.feed.log.floor(Kinds::Watched);
            let mut stream = run.watch(Start::After(held), &[]);
            let word = next(&mut stream).await;
            assert_eq!(
                word.event,
                Some(Event::SnapshotRequired(proto::SnapshotRequired {
                    oldest_seq: floor.seq + 1,
                })),
                "a run that starts after {later}, a place of one that started after {earlier}"
            );
            assert!(stream.next().await.is_none(), "one word, then the end");
        }
    }

    #[test]
    fn a_runs_log_starts_at_a_place_drawn_below_the_bound_and_apart_from_another_runs() {
        let starts: Vec<u64> = (0..64)
            .map(|_| {
                journaled(&LeasePlace::Process)
                    .1
                    .log
                    .head(Kinds::Watched)
                    .seq
            })
            .collect();
        assert!(
            starts.iter().all(|&first| first < FIRST_PLACES),
            "{starts:?}"
        );
        let mut apart = starts.clone();
        apart.sort_unstable();
        apart.dedup();
        assert_eq!(apart.len(), starts.len(), "{starts:?}");
    }

    #[test]
    fn a_watchs_models_are_held_to_a_rows_bounds() {
        let asked = |models: Vec<String>| proto::WatchWorkersRequest {
            models,
            ..Default::default()
        };
        let names = |n: usize| (0..n).map(|i| format!("model-{i}")).collect::<Vec<_>>();
        let unnamed = "models: a model id is not empty and holds no blank and no control character";
        assert_eq!(
            start(&asked(names(128))).expect("at the bound"),
            Start::Snapshot
        );
        assert_eq!(
            start(&asked(vec!["m".repeat(256)])).expect("at the bound"),
            Start::Snapshot
        );
        for (models, why) in [
            (names(129), "models: a watch names at most 128 model ids"),
            (
                vec!["m".repeat(257)],
                "models: a model id is at most 256 bytes",
            ),
            (vec![String::new()], unnamed),
            (vec!["model a".to_owned()], unnamed),
        ] {
            let refused = start(&asked(models)).expect_err("refused");
            assert_eq!(
                (refused.code(), refused.message()),
                (Code::InvalidArgument, why)
            );
        }
    }

    #[tokio::test]
    async fn a_resume_at_a_held_place_gets_the_tail_alone_and_any_other_starts_again() {
        let leader = leader();
        leader.register(0, 1, "model-a");
        let held = leader.head();
        let second = leader.register(1, 1, "model-a");

        let mut stream = leader.watch(Start::After(held), &[]);
        let event = next(&mut stream).await;
        let Some(Event::Registered(row)) = &event.event else {
            panic!("the write after the held place: {event:?}");
        };
        assert_eq!((row.id, event.seq), (second.0, held.seq + 1));
        drop(stream);

        let floor = leader.feed.log.floor(Kinds::Watched);
        let lost = [
            // Another term at the position.
            Position {
                term: held.term + 1,
                ..held
            },
            // Past the head.
            Position {
                seq: leader.head().seq + 5,
                term: held.term,
            },
            // No term names a history.
            Position { term: 0, ..held },
        ];
        for place in lost {
            let mut stream = leader.watch(Start::After(place), &[]);
            let word = next(&mut stream).await;
            assert_eq!(
                word.event,
                Some(Event::SnapshotRequired(proto::SnapshotRequired {
                    oldest_seq: floor.seq + 1,
                })),
                "{place:?}"
            );
            assert!(stream.next().await.is_none(), "one word, then the end");
        }

        // A place written under no term names no history, though the log
        // has a position there: a store written before its first lease.
        let (store, feed) = journaled_from(&LeasePlace::Process, 0);
        let facts = |n: usize| WorkerFacts {
            key: key(n),
            engine: key(n),
            incarnation: 1,
            models: vec!["model-a".to_owned()],
            dp_size: 1,
            ..WorkerFacts::default()
        };
        assert_ne!(store.upsert(facts(0)).id(), store.upsert(facts(1)).id());
        let leader = leading((store, feed));
        assert_eq!(leader.head(), Position { seq: 3, term: 1 });
        let mut stream = leader.watch(Start::After(Position { seq: 1, term: 0 }), &[]);
        let word = next(&mut stream).await;
        assert!(
            matches!(word.event, Some(Event::SnapshotRequired(_))),
            "{word:?}"
        );
    }

    #[tokio::test]
    async fn a_watcher_the_ring_let_go_is_told_to_start_again() {
        let leader = leader();
        let id = leader.register(0, 1, "model-a");
        let mut stream = leader.watch(Start::Snapshot, &[]);
        next(&mut stream).await;
        // The watcher takes nothing while the log turns over more writes
        // than it retains beyond all the stream can hold.
        for flip in 0..WATCHED_EVENTS + 2 * HELD {
            let status = if flip % 2 == 0 {
                WorkerStatus::Ready
            } else {
                WorkerStatus::NotReady
            };
            leader
                .store
                .set_status(id, status, 1)
                .expect("a status write");
        }
        // It takes again: what its channel held, the word, the end.
        let word = loop {
            let event = next(&mut stream).await;
            if !matches!(event.event, Some(Event::Status(_))) {
                break event;
            }
        };
        assert!(
            matches!(word.event, Some(Event::SnapshotRequired(_))),
            "{word:?}"
        );
        assert!(stream.next().await.is_none(), "one word, then the end");
    }

    #[tokio::test]
    async fn the_word_that_its_replica_left_ends_a_stream_before_the_fence_closes() {
        let leader = leader();
        let mut stream = leader.watch(Start::Snapshot, &[]);
        next(&mut stream).await;
        // The fence is open still: the word alone ends the stream.
        leader.role.send_replace(false);
        assert_eq!(last_word(&mut stream).await.code(), Code::Aborted);
        assert!(stream.next().await.is_none(), "the end");
    }

    #[tokio::test]
    async fn a_write_that_lands_after_the_fence_closed_is_not_told() {
        let mut leader = leader();
        let mut stream = leader.watch(Start::Snapshot, &[]);
        next(&mut stream).await;
        // The fence closes and nobody says so. The next write wakes the
        // stream, which reads the fence before it reads the log.
        assert!(leader.elector.release().expect("a release"));
        leader.register(0, 1, "model-a");
        assert_eq!(last_word(&mut stream).await.code(), Code::Aborted);
        assert!(stream.next().await.is_none(), "the end");
    }

    #[tokio::test]
    async fn a_stream_of_an_earlier_term_is_ended_under_a_later_lead() {
        let mut leader = leader();
        let mut stream = leader.watch(Start::Snapshot, &[]);
        next(&mut stream).await;
        // The lease is let go and taken again, and nobody says so: the fence
        // is open, under term 2. The next write wakes the stream, which
        // reads the fence for its own term and tells nothing of a later one.
        assert!(leader.elector.release().expect("a release"));
        assert_eq!(
            leader.elector.tick(Instant::now()).expect("a tick"),
            Tick::Acquired { term: 2 }
        );
        leader.store.set_term(2);
        leader.register(0, 1, "model-a");
        assert_eq!(last_word(&mut stream).await.code(), Code::Aborted);
        assert!(stream.next().await.is_none(), "the end");
    }

    #[tokio::test]
    async fn a_watch_of_one_model_is_told_the_rows_that_serve_it() {
        let leader = leader();
        let mine = leader.register(0, 1, "model-a");
        let other = leader.register(1, 1, "model-b");
        let mut stream = leader.watch(Start::Snapshot, &["model-a"]);
        let page = next(&mut stream).await;
        let Some(Event::Snapshot(page)) = page.event else {
            panic!("a snapshot: {page:?}");
        };
        assert_eq!(
            page.workers.iter().map(|row| row.id).collect::<Vec<_>>(),
            [mine.0]
        );

        // A write about a row of another model is not told; the next about
        // its own is.
        for id in [other, mine] {
            leader
                .store
                .set_status(id, WorkerStatus::Ready, 1)
                .expect("ready");
        }
        let status = next(&mut stream).await;
        let Some(Event::Status(status)) = status.event else {
            panic!("a status: {status:?}");
        };
        assert_eq!(status.worker_id, mine.0);

        // A replacement that takes the row out of the watcher's models
        // names both rows, and the new incarnation's later writes are not
        // told.
        let moved = leader.register(0, 2, "model-b");
        let replaced = next(&mut stream).await;
        let Some(Event::Replaced(replaced)) = replaced.event else {
            panic!("a replacement: {replaced:?}");
        };
        assert_eq!(replaced.old.map(|row| row.id), Some(mine.0));
        assert_eq!(
            replaced.new.map(|row| row.models),
            Some(vec!["model-b".to_owned()])
        );
        leader.store.remove(moved, 1).expect("removed");
        let again = leader.register(2, 1, "model-a");
        let registered = next(&mut stream).await;
        let Some(Event::Registered(row)) = registered.event else {
            panic!("the next write of the watcher's model: {registered:?}");
        };
        assert_eq!(row.id, again.0);
    }

    #[tokio::test]
    async fn a_load_write_is_no_place_of_the_log_and_wakes_no_watcher() {
        let leader = leader();
        let id = leader.register(0, 1, "model-a");
        let mut stream = leader.watch(Start::Snapshot, &[]);
        next(&mut stream).await;
        turns().await;
        let (woken, waker) = Woken::waker();
        parks(&mut stream, &waker);
        let before = (leader.head(), leader.feed.log.head(Kinds::All));
        let now = Instant::now();
        for _ in 0..10_000 {
            leader
                .store
                .record(id, 0, LoadRow::default(), now, LoadSource::Stream);
        }
        // The journal's kinds were fixed where the store was built, the
        // watched kinds alone: a load row is no position of this log, wakes
        // no task and hands the watcher nothing.
        assert_eq!((leader.head(), leader.feed.log.head(Kinds::All)), before);
        turns().await;
        assert_eq!(woken.wakes(), 0, "a load row woke the stream");
        parks(&mut stream, &waker);
    }

    /// A watched write wakes no watch itself. Once its store has let its
    /// locks go it wakes the feed's one task, and that task wakes the
    /// parked streams: a write pays one wake however many watch.
    #[tokio::test]
    async fn a_watched_write_wakes_the_feeds_one_task_and_that_task_the_watches() {
        let leader = leader();
        let id = leader.register(0, 1, "model-a");
        let mut stream = leader.watch(Start::Snapshot, &[]);
        next(&mut stream).await;
        turns().await;
        let (woken, waker) = Woken::waker();
        parks(&mut stream, &waker);
        leader
            .store
            .set_status(id, WorkerStatus::Ready, 1)
            .expect("ready");
        // The write is over, and on this one thread no task has had a turn
        // since: the stream is parked still.
        assert_eq!(woken.wakes(), 0, "the write woke the watch itself");
        woken.reads(1).await;
        // One wake a park: the task took the stream's waker out, and a
        // write beside a stream that has not parked again wakes nothing.
        leader.register(1, 1, "model-a");
        turns().await;
        assert_eq!(woken.wakes(), 1, "a stream woken that had not parked");
        let told = next(&mut stream).await;
        assert!(
            matches!(told.event, Some(Event::Status(_))),
            "the status write: {told:?}"
        );
    }

    /// A write that lands before the feed's task first parks is kept for
    /// it, once: the task wakes the parked streams at its first turn. A
    /// wake of a parked task alone would lose that write's telling.
    #[tokio::test]
    async fn a_write_before_the_feeds_task_parks_is_told_at_its_first_turn() {
        let leader = leader();
        // The first watch starts the task, and on this one thread the task
        // has no turn before the write: its stream parks first.
        let mut stream = leader.watch(Start::Snapshot, &[]);
        let (woken, waker) = Woken::waker();
        assert!(poll(&mut stream, &waker).is_ready(), "the snapshot");
        parks(&mut stream, &waker);
        assert_eq!(leader.feed.teller.lock().len(), 1);
        leader.register(0, 1, "model-a");
        assert_eq!(woken.wakes(), 0, "the write woke the watch itself");
        // The task's first turn takes the kept wake.
        woken.reads(1).await;
        let told = next(&mut stream).await;
        assert!(
            matches!(told.event, Some(Event::Registered(_))),
            "the registration: {told:?}"
        );
    }

    /// The watches of one feed share its one task, started with the first
    /// of them, and the task ends with the last feed: what it held of the
    /// feed is given back.
    #[tokio::test]
    async fn a_feeds_watches_share_its_one_task_and_the_last_feed_ends_it() {
        let leader = leader();
        assert!(leader.feed.teller.lock().is_empty(), "before any watch");
        let streams: Vec<Watching> = (0..3).map(|_| leader.watch(Start::Snapshot, &[])).collect();
        assert_eq!(leader.feed.teller.lock().len(), 1);
        let parked = Arc::clone(&leader.feed.parked);
        drop((streams, leader));
        // Turns and no clock: the last feed went with the streams, and its
        // task, which held the list of parked streams, ends at the
        // runtime's next turn.
        for _ in 0..10_000 {
            if Arc::strong_count(&parked) == 1 {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("{} still hold the list", Arc::strong_count(&parked) - 1);
    }

    /// A feed counts a live task alone. One whose runtime went away has
    /// ended, though the feed's set still holds it: the feed's next watch
    /// starts another, which wakes that watch at a write.
    #[test]
    fn a_feed_whose_task_ended_with_its_runtime_starts_another_at_its_next_watch() {
        let leader = leader();
        let runtime = || {
            Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("a runtime")
        };
        // The feed's first watch, and its task with it, on a runtime that
        // goes away.
        let first = runtime();
        first.block_on(async {
            let mut stream = leader.watch(Start::Snapshot, &[]);
            next(&mut stream).await;
        });
        drop(first);
        runtime().block_on(async {
            let mut stream = leader.watch(Start::Snapshot, &[]);
            next(&mut stream).await;
            turns().await;
            // Parked ahead of the write: a stream that polls reads a write
            // with no task, and a live task alone wakes one that waits.
            let (woken, waker) = Woken::waker();
            parks(&mut stream, &waker);
            leader.register(0, 1, "model-a");
            woken.reads(1).await;
            let told = next(&mut stream).await;
            assert!(
                matches!(told.event, Some(Event::Registered(_))),
                "the registration: {told:?}"
            );
        });
    }

    /// A write that lands between a stream's read and its park is told by
    /// that poll: the stream steps once more after it parked. The feed's
    /// task took the parked wakers before the park here, the order that
    /// would leave a stream parked beside its entry.
    #[tokio::test]
    async fn a_write_between_a_streams_read_and_its_park_is_told_by_that_poll() {
        let leader = leader();
        let mut stream = leader.watch(Start::Snapshot, &[]);
        next(&mut stream).await;
        let now = time::Instant::now();
        assert!(stream.step(now).is_pending(), "the log holds nothing yet");
        // The write lands and the feed's task has its turn: nobody is
        // parked for it to wake.
        leader.register(0, 1, "model-a");
        turns().await;
        let (woken, waker) = Woken::waker();
        let told = stream.park(&waker, now);
        assert!(
            matches!(told, Poll::Ready(Some(Ok(_)))),
            "the stream parked beside its entry: {told:?}"
        );
        assert_eq!(woken.wakes(), 0);
    }

    /// A stream that comes back to a log that stands as its last read left
    /// it takes no lock of the log: its poll after an event does not wait
    /// for a lock another holds.
    #[tokio::test]
    async fn a_stream_takes_no_lock_of_a_log_that_has_not_moved() {
        let leader = leader();
        let mut stream = leader.watch(Start::Snapshot, &[]);
        next(&mut stream).await;
        leader.register(0, 1, "model-a");
        next(&mut stream).await;
        let (polled, parked) = mpsc::channel();
        let parked = thread::scope(|scope| {
            let held = leader.feed.log.held();
            scope.spawn(|| {
                let _ = polled.send(poll(&mut stream, Waker::noop()).is_pending());
            });
            let parked = parked.recv_timeout(Duration::from_secs(10));
            drop(held);
            parked
        });
        assert_eq!(parked, Ok(true), "the poll waited for the log's lock");
    }

    /// A stream that is gone gives its slot among the parked streams to
    /// the next: the list is as long as the most watches open at once.
    #[tokio::test]
    async fn a_stream_that_is_gone_gives_its_slot_to_the_next() {
        let leader = leader();
        for _ in 0..3 {
            let streams: Vec<Watching> = (0..WATCHES_MAX)
                .map(|_| leader.watch(Start::Snapshot, &[]))
                .collect();
            drop(streams);
        }
        assert_eq!(leader.feed.parked.lock().wakers.len(), WATCHES_MAX);
    }

    /// A burst past what a read takes, beside a watcher that takes nothing
    /// meanwhile, ends no watch and loses no event: what a stream has not
    /// read waits in the log, and the stream reads on from its place.
    #[tokio::test]
    async fn a_burst_beside_a_slow_watcher_ends_no_watch_and_loses_no_event() {
        let leader = leader();
        let mut stream = leader.watch(Start::Snapshot, &[]);
        let cut = next(&mut stream).await;
        let burst = 3 * HELD + 7;
        for n in 0..burst {
            leader.register(n, 1, "model-a");
        }
        let mut last = cut.seq;
        for _ in 0..burst {
            let told = next(&mut stream).await;
            assert!(
                matches!(told.event, Some(Event::Registered(_))),
                "a registration: {told:?}"
            );
            assert_eq!(told.seq, last + 1, "an event skipped or told twice");
            last = told.seq;
        }
        parks(&mut stream, Waker::noop());
    }

    /// The word that its replica left reaches a parked stream with no
    /// write to wake it, and is told ahead of an entry that landed since.
    #[tokio::test]
    async fn the_word_that_its_replica_left_wakes_a_parked_stream_ahead_of_its_next_entry() {
        let leader = leader();
        let mut stream = leader.watch(Start::Snapshot, &[]);
        next(&mut stream).await;
        turns().await;
        let (woken, waker) = Woken::waker();
        parks(&mut stream, &waker);
        // The fence is open still, and nothing was written: the word alone
        // wakes the stream, by the feed's task.
        leader.role.send_replace(false);
        woken.reads(1).await;
        leader.register(0, 1, "model-a");
        assert_eq!(last_word(&mut stream).await.code(), Code::Aborted);
        assert!(stream.next().await.is_none(), "the end");
    }

    /// The bound runs from an event the transport took, never from a park:
    /// a stream that waits on the log is its watcher's fault at no time.
    #[tokio::test(start_paused = true)]
    async fn a_stream_that_waits_on_the_log_is_not_ended_at_the_bound() {
        let leader = leader();
        let mut stream = leader.watch(Start::Snapshot, &[]);
        next(&mut stream).await;
        parks(&mut stream, Waker::noop());
        time::sleep(10 * STALL).await;
        leader.register(0, 1, "model-a");
        let told = next(&mut stream).await;
        assert!(
            matches!(told.event, Some(Event::Registered(_))),
            "a registration: {told:?}"
        );
        // And an event taken just under the bound after the last is told.
        leader.register(1, 1, "model-a");
        time::sleep(STALL - Duration::from_secs(1)).await;
        next(&mut stream).await;
    }

    #[tokio::test]
    async fn watches_that_start_at_one_place_share_one_cut_and_a_write_ends_it() {
        let leader = leader();
        leader.register(0, 1, "model-a");
        let kept = || leader.feed.rows.lock().clone().expect("a cut is kept");
        let mut first = leader.watch(Start::Snapshot, &[]);
        let page = next(&mut first).await;
        let cut = kept();
        assert_eq!(cut.at, leader.head());
        assert_eq!((page.seq, page.term), (cut.at.seq, cut.at.term));
        // A second watch at the same place takes no cut of its own.
        let mut second = leader.watch(Start::Snapshot, &[]);
        assert_eq!(next(&mut second).await, page);
        assert!(Arc::ptr_eq(&cut, &kept()), "the cut is shared");
        // A cut at the log's head is kept: a watch that starts now shares it.
        leader.feed.let_go();
        assert!(Arc::ptr_eq(&cut, &kept()), "a cut at the head is kept");
        // A write moves the place, and the tail that tells it lets the
        // passed cut go: no watch that starts now shares it.
        leader.register(1, 1, "model-a");
        next(&mut first).await;
        assert!(leader.feed.rows.lock().is_none(), "a passed cut is let go");
        // The next watch starts from the rows that write left, at its place.
        let mut third = leader.watch(Start::Snapshot, &[]);
        let later = next(&mut third).await;
        assert_eq!(later.seq, leader.head().seq);
        let Some(Event::Snapshot(rows)) = later.event else {
            panic!("a snapshot first");
        };
        assert_eq!(rows.workers.len(), 2);
        assert!(!Arc::ptr_eq(&cut, &kept()), "a cut of the later place");
    }

    #[tokio::test]
    async fn the_last_watch_to_leave_lets_the_kept_cut_go() {
        let leader = leader();
        leader.register(0, 1, "model-a");
        let mut first = leader.watch(Start::Snapshot, &[]);
        let mut second = leader
            .watch_as("gateway-b", Start::Snapshot, &[])
            .expect("another principal's seat");
        next(&mut first).await;
        next(&mut second).await;
        // A watch is still open: the cut stays for one that starts here.
        drop(first);
        leader.holds("gateway-a", 0);
        assert!(leader.feed.rows.lock().is_some(), "a watch is open");
        drop(second);
        leader.holds("gateway-b", 0);
        assert!(leader.feed.rows.lock().is_none(), "the last watch left");
    }

    #[tokio::test]
    async fn a_kept_cut_is_named_by_its_place_with_its_term() {
        let leader = leader();
        leader.register(0, 1, "model-a");
        let mut first = leader.watch(Start::Snapshot, &[]);
        let page = next(&mut first).await;
        let kept = leader.head();
        assert_eq!((page.seq, page.term), (kept.seq, kept.term));
        // A restore leaves the log at the same sequence under another term,
        // over other rows: the kept cut is not that place's.
        leader.register(1, 1, "model-a");
        let restored = Position {
            seq: kept.seq,
            term: kept.term + 1,
        };
        leader
            .feed
            .log
            .restart(restored)
            .expect("a place the log takes");
        let mut second = leader.watch(Start::Snapshot, &[]);
        let later = next(&mut second).await;
        assert_eq!((later.seq, later.term), (restored.seq, restored.term));
        let Some(Event::Snapshot(rows)) = later.event else {
            panic!("a snapshot first");
        };
        assert_eq!(rows.workers.len(), 2);
    }

    #[tokio::test]
    async fn a_principal_holds_its_ceiling_of_watches_and_a_dropped_one_gives_its_seat_back() {
        let leader = leader();
        let mut streams: Vec<Watching> = (0..WATCHES_MAX)
            .map(|_| leader.watch(Start::Snapshot, &[]))
            .collect();
        let refused = leader
            .watch_as("gateway-a", Start::Snapshot, &[])
            .expect_err("one watch past the ceiling");
        assert_eq!(
            (refused.code(), refused.message()),
            (
                Code::ResourceExhausted,
                "this principal holds its ceiling of 4 open watches: end one, then watch again"
            )
        );
        // Another principal's watches are its own.
        let other = leader
            .watch_as("gateway-b", Start::Snapshot, &[])
            .expect("another principal's seat");
        // A watcher that leaves gives its seat back, and the next is seated.
        drop(streams.pop());
        leader.holds("gateway-a", WATCHES_MAX - 1);
        let mut again = leader
            .watch_as("gateway-a", Start::Snapshot, &[])
            .expect("the seat given back");
        next(&mut again).await;
        drop((streams, other, again));
        leader.holds("gateway-a", 0);
        leader.holds("gateway-b", 0);
    }

    #[tokio::test]
    async fn the_table_holds_its_ceiling_of_watching_principals() {
        // The bed's table holds two principals.
        let leader = leader();
        let first = leader
            .watch_as("gateway-a", Start::Snapshot, &[])
            .expect("a seat");
        let _second = leader
            .watch_as("gateway-b", Start::Snapshot, &[])
            .expect("a seat");
        let refused = leader
            .watch_as("gateway-c", Start::Snapshot, &[])
            .expect_err("a third principal");
        assert_eq!(
            (refused.code(), refused.message()),
            (
                Code::ResourceExhausted,
                "this replica holds its ceiling of 2 watching principals: watch again later"
            )
        );
        // A principal that watches already takes no new row.
        let again = leader
            .watch_as("gateway-a", Start::Snapshot, &[])
            .expect("a seat in a standing row");
        // A principal's last watch takes its row out with it.
        drop(first);
        leader.holds("gateway-a", 1);
        leader
            .watch_as("gateway-c", Start::Snapshot, &[])
            .expect_err("the row stands while one watch does");
        drop(again);
        leader.holds("gateway-a", 0);
        leader
            .watch_as("gateway-c", Start::Snapshot, &[])
            .expect("the row given back");
    }

    /// A watcher that takes nothing for the bound after an event is ended
    /// with no error, at its stream's next poll, and no writer waited for
    /// it. The seat stays the stream's until its watcher lets it go.
    #[tokio::test(start_paused = true)]
    async fn a_watcher_that_takes_nothing_is_ended_at_its_bound_and_no_writer_waits() {
        let leader = leader();
        let mut stream = leader.watch(Start::Snapshot, &[]);
        next(&mut stream).await;
        // Twice what a read takes, each write made and done while the
        // watcher takes nothing.
        for n in 0..2 * HELD {
            leader.register(n, 1, "model-a");
        }
        assert_eq!(
            leader.head().seq,
            1 + 2 * u64::try_from(HELD).expect("a count")
        );
        time::sleep(STALL + Duration::from_secs(1)).await;
        assert!(
            stream.next().await.is_none(),
            "the stream ended at its bound, with no error"
        );
        leader.holds("gateway-a", 1);
        // The stream let go gives the seat back.
        drop(stream);
        leader.holds("gateway-a", 0);
    }

    /// The events `stream` tells ahead of its last word, counted, and the
    /// word. A stream that tells more than `most` events fails here.
    async fn to_the_last_word(stream: &mut Watching, most: usize) -> (usize, Status) {
        for taken in 0..=most {
            let told = time::timeout(Duration::from_secs(5), stream.next())
                .await
                .expect("an answer within five seconds")
                .expect("the stream is open");
            if let Err(status) = told {
                return (taken, status);
            }
        }
        panic!("more than {most} events ahead of the last word");
    }

    /// A watcher that takes nothing keeps its stream, and the seat with it,
    /// for as long as it keeps its end: a principal's ceiling of watches is
    /// the ceiling of the streams it leaves unread.
    #[tokio::test(start_paused = true)]
    async fn a_stream_left_unread_holds_its_seat_until_its_watcher_lets_it_go() {
        let leader = leader();
        let mut streams: Vec<Watching> = (0..WATCHES_MAX)
            .map(|_| leader.watch(Start::Snapshot, &[]))
            .collect();
        for stream in &mut streams {
            next(stream).await;
        }
        for n in 0..2 * HELD {
            leader.register(n, 1, "model-a");
        }
        time::sleep(STALL + Duration::from_secs(1)).await;
        // Past the bound and unread still: each stream holds its seat.
        let refused = leader
            .watch_as("gateway-a", Start::Snapshot, &[])
            .expect_err("a watch past four streams left unread");
        assert_eq!(refused.code(), Code::ResourceExhausted);
        // A stream let go gives its seat back.
        drop(streams.pop());
        leader.holds("gateway-a", WATCHES_MAX - 1);
        leader
            .watch_as("gateway-a", Start::Snapshot, &[])
            .expect("the seat given back");
    }

    /// The fence is read before each hand-over. A watcher inside a read of
    /// the log is told that its replica stopped leading ahead of the read's
    /// next entry, though that entry was written before the fence closed.
    #[tokio::test]
    async fn a_batchs_next_entry_is_not_told_after_the_fence_closed() {
        let mut leader = leader();
        let mut stream = leader.watch(Start::Snapshot, &[]);
        next(&mut stream).await;
        let written = 8;
        for n in 0..written {
            leader.register(n, 1, "model-a");
        }
        // One read took the eight, and the watcher takes three of them.
        for _ in 0..3 {
            next(&mut stream).await;
        }
        // The fence closes and nobody says so.
        assert!(leader.elector.release().expect("a release"));
        let (taken, word) = to_the_last_word(&mut stream, written).await;
        assert_eq!(word.code(), Code::Aborted);
        assert_eq!(taken, 0, "an entry in hand told after the fence closed");
        assert!(stream.next().await.is_none(), "the end");
    }

    /// And before each page of a snapshot. A row past half a page of bytes
    /// goes alone, so a few rows make as many pages.
    #[tokio::test]
    async fn a_snapshots_next_page_is_not_handed_after_the_fence_closed() {
        let mut leader = leader();
        let pages = 4;
        for n in 0..pages {
            leader.register_wide(n);
        }
        let mut stream = leader.watch(Start::Snapshot, &[]);
        next(&mut stream).await;
        assert!(leader.elector.release().expect("a release"));
        let (taken, word) = to_the_last_word(&mut stream, pages).await;
        assert_eq!(word.code(), Code::Aborted);
        assert_eq!(taken, 0, "a page handed after the fence closed");
        assert!(stream.next().await.is_none(), "the end");
    }

    /// A watch whose term ended between its admission and its stream's
    /// first poll is sent no snapshot, and no cut is taken for it.
    #[tokio::test]
    async fn a_watch_whose_term_ended_before_its_start_is_sent_no_snapshot() {
        let mut leader = leader();
        leader.register(0, 1, "model-a");
        assert!(leader.elector.release().expect("a release"));
        let mut stream = leader.watch(Start::Snapshot, &[]);
        assert_eq!(last_word(&mut stream).await.code(), Code::Aborted);
        assert!(stream.next().await.is_none(), "the end");
        assert!(leader.feed.rows.lock().is_none(), "no cut was taken for it");
    }

    /// The word that its replica left is told ahead of an entry in hand:
    /// the read's next entry is not told.
    #[tokio::test]
    async fn the_word_that_its_replica_left_is_told_ahead_of_an_entry_in_hand() {
        let leader = leader();
        let mut stream = leader.watch(Start::Snapshot, &[]);
        next(&mut stream).await;
        let written = 8;
        for n in 0..written {
            leader.register(n, 1, "model-a");
        }
        // One of the eight is taken: seven are in hand.
        next(&mut stream).await;
        // The fence is open still: the word alone ends the stream.
        leader.role.send_replace(false);
        let (taken, word) = to_the_last_word(&mut stream, written).await;
        assert_eq!(word.code(), Code::Aborted);
        assert_eq!(taken, 0, "an entry in hand told after the word");
        assert!(stream.next().await.is_none(), "the end");
    }

    /// The last word waits on its watcher as an event does, and no longer:
    /// a watcher that takes nothing for [`STALL`] is ended with no error at
    /// its next poll, though the word that its replica left was told
    /// meanwhile. The bound is read ahead of the word, and the word is not
    /// told.
    #[tokio::test(start_paused = true)]
    async fn the_last_word_waits_on_a_watcher_that_takes_nothing_to_its_bound_and_no_longer() {
        let leader = leader();
        let mut stream = leader.watch(Start::Snapshot, &[]);
        next(&mut stream).await;
        leader.register(0, 1, "model-a");
        // The word is told while the watcher takes nothing, the fence open
        // still.
        leader.role.send_replace(false);
        time::sleep(STALL + Duration::from_secs(1)).await;
        assert!(
            stream.next().await.is_none(),
            "the stream ended at its bound, with no word"
        );
    }

    /// A watcher that comes back inside [`STALL`] is told the word that its
    /// replica left at that poll, ahead of every entry the log holds for
    /// it: the word waits behind nothing.
    #[tokio::test(start_paused = true)]
    async fn the_last_word_is_told_to_a_watcher_that_comes_back_inside_its_bound() {
        let leader = leader();
        let mut stream = leader.watch(Start::Snapshot, &[]);
        next(&mut stream).await;
        let written = HELD + 8;
        for n in 0..written {
            leader.register(n, 1, "model-a");
        }
        // The word is told while the watcher takes nothing, the fence open
        // still.
        leader.role.send_replace(false);
        time::sleep(STALL / 2).await;
        let (taken, word) = to_the_last_word(&mut stream, written).await;
        assert_eq!(word.code(), Code::Aborted);
        assert_eq!(taken, 0, "an entry told after the word");
        assert!(stream.next().await.is_none(), "the end");
    }

    /// A tail that lets the kept cut go waits behind no watch that holds
    /// the lock for its cut: the lock is tried.
    #[test]
    fn a_tail_waits_behind_no_watch_that_takes_its_cut() {
        let leader = leader();
        let cutting = leader.feed.rows.lock();
        let (done, came_back) = mpsc::channel();
        let feed = leader.feed.clone();
        let tail = thread::spawn(move || {
            feed.let_go();
            done.send(()).ok();
        });
        came_back
            .recv_timeout(Duration::from_secs(5))
            .expect("the tail came back while a watch held the lock");
        drop(cutting);
        tail.join().expect("the tail's thread");
    }

    /// The draw is made wherever the lease lives. A later run takes a term
    /// again at each start where the lease lives in the process, and where
    /// a shared lease's record was lost.
    #[test]
    fn the_draw_is_made_wherever_the_lease_lives() {
        for place in [LeasePlace::Process, LeasePlace::File("lease".into())] {
            let first = journaled(&place).1.log.head(Kinds::Watched);
            assert!(
                first.seq != 0 && first.term == 0,
                "a drawn start: {first:?}"
            );
        }
    }

    /// The fence is read at each turn of the tail, before the log: once it
    /// closed, a write that is none of the watcher's ends its stream too.
    #[tokio::test]
    async fn a_write_of_another_model_ends_a_stream_after_the_fence_closed() {
        let mut leader = leader();
        let mut stream = leader.watch(Start::Snapshot, &["model-b"]);
        next(&mut stream).await;
        assert!(leader.elector.release().expect("a release"));
        leader.register(0, 1, "model-a");
        assert_eq!(last_word(&mut stream).await.code(), Code::Aborted);
        assert!(stream.next().await.is_none(), "the end");
    }

    /// The word that its replica left, told between a watch's open and its
    /// task's first turn, is not lost: the stream ends at it.
    #[tokio::test]
    async fn the_word_told_before_a_streams_first_turn_ends_it() {
        let leader = leader();
        let mut stream = leader.watch(Start::Snapshot, &[]);
        // The task has not run yet, and the fence is open still.
        leader.role.send_replace(false);
        assert_eq!(last_word(&mut stream).await.code(), Code::Aborted);
        assert!(stream.next().await.is_none(), "the end");
    }

    /// A stream may find at its start the word of the time before the lead
    /// it was admitted under: that its replica does not lead. That word told
    /// again ends the stream, though it changes nothing.
    #[tokio::test]
    async fn the_word_told_again_ends_a_stream_that_started_under_it() {
        let leader = leader();
        leader.role.send_replace(false);
        let mut stream = leader.watch(Start::Snapshot, &[]);
        // The fence is open: the stream is served under the word it found.
        let page = next(&mut stream).await;
        assert!(matches!(page.event, Some(Event::Snapshot(_))), "{page:?}");
        leader.role.send_replace(false);
        assert_eq!(last_word(&mut stream).await.code(), Code::Aborted);
        assert!(stream.next().await.is_none(), "the end");
    }

    /// The tail starts at the snapshot's place, not where the log stands
    /// when the last page is handed over: a write that lands while the
    /// pages wait on the watcher is told after them.
    #[tokio::test]
    async fn a_write_that_lands_while_a_snapshot_is_handed_over_is_told_after_it() {
        let leader = leader();
        let pages = 4;
        for n in 0..pages {
            leader.register_wide(n);
        }
        let mut stream = leader.watch(Start::Snapshot, &[]);
        // The stream stands inside its snapshot, past its cut.
        let first = next(&mut stream).await;
        assert!(
            matches!(first.event, Some(Event::Snapshot(_))),
            "a page of the snapshot"
        );
        let id = leader.register(0, 1, "model-a");
        for _ in 1..pages {
            let page = next(&mut stream).await;
            assert!(
                matches!(page.event, Some(Event::Snapshot(_))),
                "a page of the snapshot"
            );
        }
        let told = next(&mut stream).await;
        let Some(Event::Registered(row)) = &told.event else {
            panic!("the write after the cut: {told:?}");
        };
        assert_eq!(row.id, id.0);
    }
}
