//! `WatchWorkers` (design 5.2): the store's membership as one history. A
//! watcher is sent the rows as one write of the log left them, then every
//! membership and status event after that write in the log's order, or it
//! is told to start again: never a gap, never a replay.
//!
//! No writer waits for a watcher. A stream has one task. It starts from the
//! cut of its place (taken once at a place, under the membership lock for
//! the copy of the rows, and shared), then reads the log when
//! a watched write landed or the role changed, never for a load row, and
//! hands its events to a bounded channel the transport drains. A watcher
//! that takes nothing fills its own channel and is ended at [`STALL`]; one
//! the log's ring let go is told to start again.
//!
//! A watch of some models is told a write when a row it names serves one of
//! them: the row the write left in the table, or the row it took out. A
//! row's models change with its incarnation at the door, so a row leaves a
//! watcher's models by a replacement, which names both rows.
//!
//! No peer sets a size here. A principal holds at most [`WATCHES_MAX`]
//! watches open at a replica, and as many principals watch a replica as may
//! be present at a leader: an open past either bound is refused by name. A
//! watch's seat is its task's, so it is given back when the stream ends, by
//! any path.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use prost::encoding::message;
use rustc_hash::FxHashMap;
use tokio::{
    sync::{mpsc, watch},
    task::JoinSet,
};
use tonic::{codegen::tokio_stream::Stream, Status};
use tracing::{info, warn};

use super::{principal::Principal, worker_wire};
use crate::{
    elector::Fence,
    proto,
    replication::{Entry, Event, Log},
    store::{Cut, GatewayId, Kinds, MemoryStore, Position, WorkerRow},
};

/// The rows of one snapshot page.
const PAGE: usize = 128;

/// The bytes of one snapshot page's rows on the wire: half a frame of 4 MiB.
/// A page of [`PAGE`] rows at the door's bound of a row is cut here.
const PAGE_BYTES: usize = 2 * 1024 * 1024;

/// The log entries one read takes: the log's own bound.
const BATCH: usize = Log::BATCH;

/// The messages a stream holds for its watcher.
const HELD: usize = 64;

/// How long a stream waits on a watcher that takes nothing. Then it ends,
/// and the watcher resumes by the place of the last event it took.
const STALL: Duration = Duration::from_secs(30);

/// The watches one principal holds open at a replica. A gateway follows a
/// replica with one watch and holds one more across a reconnect: twice
/// that.
pub(crate) const WATCHES_MAX: usize = 4;

/// What a watch reads: the store's log and the wake of its watched writes.
#[derive(Clone, Debug)]
pub(crate) struct Feed {
    pub(crate) log: Arc<Log>,
    /// The place of the last watched write; a stream's task waits on its
    /// change.
    wake: watch::Receiver<Position>,
    /// The rows at one place of the log: the cut the first watch took
    /// there, kept for every other that starts there. It is let go when a
    /// write moves the log past that place and when the last watch leaves:
    /// a copy of every row is held only while a watch can share it.
    rows: Arc<Mutex<Option<Arc<Cut>>>>,
}

impl Feed {
    /// The feed of `log`, which tells its wake at each watched write. The
    /// store that journals to the log is built beside it
    /// ([`journaled`](super::replication::journaled)).
    pub(crate) fn of(log: Log) -> Self {
        let (told, wake) = watch::channel(Position::default());
        let log = Arc::new(log.on_watched(move |at| {
            told.send_replace(at);
        }));
        Self {
            log,
            wake,
            rows: Arc::default(),
        }
    }

    /// Lets the kept cut go once a write has moved the log past its place:
    /// no watch that starts now shares it.
    fn let_go(&self) {
        let mut kept = self.rows.lock();
        if kept
            .as_ref()
            .is_some_and(|cut| cut.at != self.log.head(Kinds::Watched))
        {
            *kept = None;
        }
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

/// What a stream's task reads.
#[derive(Debug)]
pub(crate) struct Source {
    pub(crate) store: Arc<MemoryStore>,
    pub(crate) feed: Feed,
    pub(crate) fence: Arc<Fence>,
    /// Whether this replica leads, told at a change of role alone. A
    /// stream ends at the word that it does not; the word it found at its
    /// start may be the one before the take the stream was admitted under.
    pub(crate) role: watch::Receiver<bool>,
    /// The term the watch was admitted under.
    pub(crate) term: u64,
}

impl Source {
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

/// One watcher's stream: the events its task hands over.
#[derive(Debug)]
pub(crate) struct Watching {
    events: mpsc::Receiver<Told>,
    /// The stream's one task, aborted when the stream is dropped.
    _task: JoinSet<()>,
}

impl Stream for Watching {
    type Item = Told;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.events.poll_recv(cx)
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

/// One open watch's seat in the table, given back when it is dropped. The
/// last seat given back lets the feed's kept cut go: no watch shares it.
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
    info!(
        by = %by.subject,
        ?start,
        models = models.len(),
        term = source.term,
        "watch opened"
    );
    let (hand, events) = mpsc::channel(HELD);
    let mut task = JoinSet::new();
    task.spawn(run(source, start, Asked { models }, hand, seat));
    Ok(Watching {
        events,
        _task: task,
    })
}

/// The models a watcher asked for; none is every model.
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

    /// Whether the write is told: a row it names serves an asked model.
    fn tells(&self, entry: &Entry) -> bool {
        match &entry.event {
            Event::Member { row, left, .. } => [row, left]
                .into_iter()
                .flatten()
                .any(|row| self.serves(row)),
            Event::Load { .. } | Event::Term => false,
        }
    }
}

/// Hands `told` to the watcher. `false` when the watcher left, or took
/// nothing for [`STALL`].
async fn hand_over(hand: &mpsc::Sender<Told>, told: Told) -> bool {
    hand.send_timeout(told, STALL).await.is_ok()
}

/// The word to a watcher that cannot continue from its place, the stream's
/// last.
async fn start_again(source: &Source, hand: &mpsc::Sender<Told>) {
    let floor = source.feed.log.floor(Kinds::Watched);
    hand_over(hand, Ok(worker_wire::start_again(floor))).await;
}

/// The status of a stream whose replica stopped leading under its term, the
/// stream's last: the watcher finds the leader through `Leader`.
async fn fenced(hand: &mpsc::Sender<Told>) {
    let status = Status::aborted("this replica no longer leads under the watch's term");
    hand_over(hand, Err(status)).await;
}

/// The page that starts `rows`: fewer than [`PAGE`] of them while they take
/// at most [`PAGE_BYTES`] on the wire. A page takes its first row whatever
/// it weighs: a row past a page's bytes goes alone.
fn fill(rows: &[&WorkerRow]) -> Vec<proto::WorkerRow> {
    let mut page = Vec::new();
    let mut bytes = 0;
    for row in rows.iter().take(PAGE) {
        let told = worker_wire::row(row);
        // A row in a page: its bytes under the page's tag and a length
        // (`WorkerSnapshot.workers`).
        let framed = message::encoded_len(1, &told);
        if !page.is_empty() && bytes + framed > PAGE_BYTES {
            break;
        }
        bytes += framed;
        page.push(told);
    }
    page
}

/// A stream's task: the start, then the tail until the watcher leaves, the
/// log lets its place go or this replica stops leading under the term. It
/// holds the watch's seat to its end.
async fn run(
    mut source: Source,
    start: Start,
    asked: Asked,
    hand: mpsc::Sender<Told>,
    _seat: Seat,
) {
    // Seen from here: a write or a change of role after this line wakes the
    // tail, whether it landed before the cut or after it.
    source.feed.wake.mark_unchanged();
    source.role.mark_unchanged();
    let mut place = match start {
        Start::After(place) => {
            if !source.feed.log.holds(Kinds::Watched, place) {
                start_again(&source, &hand).await;
                return;
            }
            place
        }
        Start::Snapshot => {
            let cut = source.rows();
            let rows: Vec<&WorkerRow> = cut.rows.iter().filter(|row| asked.serves(row)).collect();
            let mut told = 0;
            // An empty fleet is one empty page: the watcher is told the
            // snapshot ended, and where.
            loop {
                let page = fill(&rows[told..]);
                told += page.len();
                let last = told == rows.len();
                if !hand_over(&hand, Ok(worker_wire::page(page, last, cut.at, cut.epoch))).await {
                    return;
                }
                if last {
                    break;
                }
            }
            cut.at
        }
    };
    let mut entries = Vec::with_capacity(BATCH);
    loop {
        if source.fence.leader_term(Instant::now()) != Some(source.term) {
            fenced(&hand).await;
            return;
        }
        if source
            .feed
            .log
            .read(Kinds::Watched, place, BATCH, &mut entries)
            .is_err()
        {
            start_again(&source, &hand).await;
            return;
        }
        if entries.is_empty() {
            tokio::select! {
                woke = source.feed.wake.changed() => {
                    if woke.is_err() {
                        return;
                    }
                }
                role = source.role.changed() => {
                    // The word that this replica does not lead ends the
                    // stream, whatever the fence reads at this instant: a
                    // leader that leaves says so before its fence closes.
                    if role.is_err() || !*source.role.borrow_and_update() {
                        fenced(&hand).await;
                        return;
                    }
                }
                () = hand.closed() => return,
            }
            continue;
        }
        // The log moved past the kept cut's place: no watch that starts now
        // shares it.
        source.feed.let_go();
        for entry in entries.drain(..) {
            place = entry.at;
            if !asked.tells(&entry) {
                continue;
            }
            if let Some(told) = worker_wire::member(&entry) {
                if !hand_over(&hand, Ok(told)).await {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        sync::Arc,
        time::{Duration, Instant},
    };

    use prost::encoding::message;
    use tokio::sync::{mpsc::error::TryRecvError, watch};
    use tonic::{codegen::tokio_stream::StreamExt, Code, Status};

    use super::{
        fill, open, start, Feed, Source, Start, Watchers, Watching, BATCH, HELD, PAGE, PAGE_BYTES,
        STALL, WATCHES_MAX,
    };
    use crate::{
        controller::{
            election::LeasePlace,
            principal::tests::told,
            replication::{journaled, journaled_from, FIRST_PLACES, WATCHED_EVENTS},
            worker_wire,
        },
        elector::{ElectionSettings, Elector, Tick},
        proto::{self, worker_event::Event},
        store::{
            Kinds, LeasesStore, LoadSink, LoadSource, MemoryStore, Position, WorkerFacts,
            WorkerPatch, WorkerRow, WorkerStatus, WorkersStore,
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
        let mut elector = Elector::alone("controller-0@127.0.0.1:1", ElectionSettings::default())
            .expect("the default settings are in order");
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

        /// `by`'s open watches read `want` within a bound of the runtime's
        /// turns: the task of a dropped stream ends at the next one. Turns
        /// and no clock: a paused clock stands still under a task that
        /// yields, and a count that never comes would wait for good.
        async fn holds(&self, by: &str, want: usize) {
            for _ in 0..10_000 {
                if self.watchers.open_of(&told(by)) == want {
                    return;
                }
                tokio::task::yield_now().await;
            }
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

        fn head(&self) -> Position {
            self.feed.log.head(Kinds::Watched)
        }
    }

    /// The stream's next event, which the case expects within five seconds.
    async fn next(stream: &mut Watching) -> proto::WorkerEvent {
        tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("an event within five seconds")
            .expect("the stream is open")
            .expect("an event, not a status")
    }

    /// The status that ends the stream, which the case expects within five
    /// seconds.
    async fn last_word(stream: &mut Watching) -> Status {
        tokio::time::timeout(Duration::from_secs(5), stream.next())
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
                let page = fill(&rows[told..]);
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
        for flip in 0..WATCHED_EVENTS + BATCH + HELD {
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
        // The stream's task waits on the wake; this handle of it counts:
        // the wake's value moves once a watched write.
        let mut wakes = leader.feed.wake.clone();
        wakes.mark_unchanged();
        let before = (leader.head(), leader.feed.log.head(Kinds::All));
        let now = Instant::now();
        for _ in 0..10_000 {
            leader
                .store
                .record(id, 0, LoadRow::default(), now, LoadSource::Stream);
        }
        // The journal's kinds were fixed where the store was built, the
        // watched kinds alone: a load row is no position of this log, moves
        // no wake and hands the watcher nothing.
        assert_eq!((leader.head(), leader.feed.log.head(Kinds::All)), before);
        assert!(!wakes.has_changed().expect("the log lives"));
        tokio::task::yield_now().await;
        assert_eq!(stream.events.try_recv().err(), Some(TryRecvError::Empty));
        // A watched write is one wake.
        leader
            .store
            .set_status(id, WorkerStatus::Ready, 1)
            .expect("ready");
        assert!(wakes.has_changed().expect("the log lives"));
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
        leader.holds("gateway-a", 0).await;
        assert!(leader.feed.rows.lock().is_some(), "a watch is open");
        drop(second);
        leader.holds("gateway-b", 0).await;
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
        leader.feed.log.restart(restored);
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
        leader.holds("gateway-a", WATCHES_MAX - 1).await;
        let mut again = leader
            .watch_as("gateway-a", Start::Snapshot, &[])
            .expect("the seat given back");
        next(&mut again).await;
        drop((streams, other, again));
        leader.holds("gateway-a", 0).await;
        leader.holds("gateway-b", 0).await;
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
        leader.holds("gateway-a", 1).await;
        leader
            .watch_as("gateway-c", Start::Snapshot, &[])
            .expect_err("the row stands while one watch does");
        drop(again);
        leader.holds("gateway-a", 0).await;
        leader
            .watch_as("gateway-c", Start::Snapshot, &[])
            .expect("the row given back");
    }

    #[tokio::test(start_paused = true)]
    async fn a_watcher_that_takes_nothing_is_ended_at_its_bound_and_no_writer_waits() {
        let leader = leader();
        let mut stream = leader.watch(Start::Snapshot, &[]);
        next(&mut stream).await;
        // Twice what the stream holds, each write made and done while the
        // watcher takes nothing.
        for n in 0..2 * HELD {
            leader.register(n, 1, "model-a");
        }
        assert_eq!(
            leader.head().seq,
            1 + 2 * u64::try_from(HELD).expect("a count")
        );
        tokio::time::sleep(STALL + Duration::from_secs(1)).await;
        // The stream ended at its bound and its seat is given back, with the
        // watcher still holding its end.
        leader.holds("gateway-a", 0).await;
        // What its channel held is still the watcher's; then the stream has
        // ended, and the watcher resumes by its place.
        let taken = tokio::time::timeout(Duration::from_secs(60), async {
            let mut taken = 0;
            while let Some(told) = stream.next().await {
                told.expect("an event");
                taken += 1;
            }
            taken
        })
        .await
        .expect("the stream ended at its bound");
        assert_eq!(taken, HELD);
    }

    /// The draw is made where the lease lives in the process, whose log no
    /// standby follows: there a term is taken again at each start. Under a
    /// shared lease the term is the guard, and the log starts at the start,
    /// where a standby joins it.
    #[test]
    fn the_draw_is_made_where_the_lease_lives_in_the_process_alone() {
        let alone = journaled(&LeasePlace::Process).1.log.head(Kinds::Watched);
        assert!(
            alone.seq != 0 && alone.term == 0,
            "a drawn start: {alone:?}"
        );
        let shared = journaled(&LeasePlace::File("lease".into())).1;
        assert_eq!(shared.log.head(Kinds::Watched), Position::default());
    }
}
