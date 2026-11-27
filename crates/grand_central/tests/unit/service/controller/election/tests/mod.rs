use std::{
    collections::VecDeque,
    io::Write,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use tokio::sync::watch;

use super::{
    elector, give_up, identity, publish_end, run_loop, split_identity, start, ElectionConfig,
    ElectionNotifier, ElectionObservation, LeaseError, LeasePlace, Running, Word,
};
use crate::{
    service::{
        controller::replication::tests::journaled_from,
        elector::{
            ConditionalWriteOutcome, ElectError, ElectionSettings, Elector, Fence, LeaseBackend,
            LeaseRecord, MemoryLease, TickOutcome, TERM_CEILING,
        },
    },
    store::{LeasesStore, MemoryStore, Position},
};

mod followed_term;
mod lease_place;
mod log_lines;
mod slow_store;
mod standing;
mod stop;
mod word_to_leave;

/// A door no thread listens behind, for a case of the door's caller. It
/// stands in for the handle too: the door says while this lives.
pub(in crate::service::controller) struct Said {
    /// The handle's part: the door holds these weakly.
    _words: Arc<mpsc::Sender<Word>>,
    heard: mpsc::Receiver<Word>,
}

impl Said {
    /// The door, and what is said through it.
    pub(in crate::service::controller) fn door() -> (ElectionNotifier, Self) {
        let (words, heard) = mpsc::channel();
        let words = Arc::new(words);
        let door = ElectionNotifier {
            words: Arc::downgrade(&words),
        };
        let said = Self {
            _words: words,
            heard,
        };
        (door, said)
    }

    /// The terms said since the last call, in order.
    pub(in crate::service::controller) fn terms(&self) -> Vec<u64> {
        self.heard
            .try_iter()
            .filter_map(|word| match word {
                Word::StepDown(above) => above,
                Word::HandOver(_) | Word::Stop => None,
            })
            .collect()
    }
}

/// Runs the loop through its normal cleanup for cases that call it directly.
fn run(
    mut elector: Elector,
    store: &MemoryStore,
    standing: &watch::Sender<ElectionObservation>,
    words: &mpsc::Receiver<Word>,
) {
    let seen = run_loop(&mut elector, store, standing, words);
    give_up(&mut elector);
    publish_end(&elector, standing, seen);
}

/// Polls `met` every few milliseconds. No instant has two open fences:
/// at every poll the first fence is read, then the second, then the
/// first again, each at its own instant, and the first open under one
/// term on both sides of an open second is two open fences at the
/// second's instant. One instant for both reads would call a hand-over
/// between them two leaders: the taker's stamp lies after that instant.
fn wait_for(what: &str, fences: [&Fence; 2], met: impl Fn() -> bool) {
    let started = Instant::now();
    loop {
        let before = fences[0].leader_term(Instant::now());
        let between = fences[1].leader_term(Instant::now());
        let after = fences[0].leader_term(Instant::now());
        assert!(
            before.is_none() || between.is_none() || before != after,
            "two open fences while waiting for {what}"
        );
        if met() {
            return;
        }
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "{what} did not happen in 20 s"
        );
        thread::sleep(Duration::from_millis(2));
    }
}

/// Polls until `met`, bounded.
pub(in crate::service::controller) fn until(what: &str, mut met: impl FnMut() -> bool) {
    let started = Instant::now();
    while !met() {
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "{what} did not happen in 20 s"
        );
        thread::sleep(Duration::from_millis(2));
    }
}

/// Waits, bounded, for an election thread to signal that it is gone.
pub(in crate::service::controller) fn ended(running: Running) {
    let mut end = running.end.expect("the end was not seen yet");
    let started = Instant::now();
    while end.try_recv().is_err() {
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the election thread did not signal its end in 20 s"
        );
        thread::sleep(Duration::from_millis(2));
    }
}

/// How many times a case that bounds the loop on the real clock is made.
/// A try whose bound fails where its witness says the host held this
/// process says so and is made again. No witness passes a run: the last
/// try stands whatever its witness says, so a loop that fails by itself
/// fails each try, and the case.
const TRIES: u32 = 3;

/// A bound on the real clock that did not hold: the one failure a hold
/// of the host explains, so the one [`unheld`] makes a try again for.
struct Late(String);

/// `assert!` for a bound on the real clock: the try unwinds as [`Late`]
/// with no line of the panic hook's, and [`unheld`] says its words.
macro_rules! bound {
        ($met:expr, $($said:tt)+) => {
            if !$met {
                let said = format!("{}:{}: {}", file!(), line!(), format_args!($($said)+));
                std::panic::resume_unwind(Box::new(Late(said)));
            }
        };
    }
// The cases stand in child modules declared above: this line gives the
// macro a path, and their `use super::*` takes it.
use bound;

/// A witness of the host's holds beside such a case: a thread that naps
/// 5 ms at a time and keeps the longest it overslept. A host that holds
/// this process holds its witness too.
struct Witness {
    end: mpsc::Sender<()>,
    naps: thread::JoinHandle<Duration>,
}

impl Witness {
    fn start() -> Self {
        let (end, ended) = mpsc::channel();
        // The first nap begins here, before the case can run, and each
        // nap begins where the one before it ended: no instant of the
        // case is outside a nap, the witness thread's own start with it.
        let mut began = Instant::now();
        let naps = thread::spawn(move || {
            let nap = Duration::from_millis(5);
            let mut longest = Duration::ZERO;
            loop {
                let over = ended.recv_timeout(nap) != Err(mpsc::RecvTimeoutError::Timeout);
                let woke = Instant::now();
                // The nap its end cut short counts too: a case the host
                // held can end before its witness wakes.
                longest = longest.max(woke.duration_since(began).saturating_sub(nap));
                if over {
                    return longest;
                }
                began = woke;
            }
        });
        Self { end, naps }
    }

    /// Ends the witness: the longest hold it saw.
    fn held(self) -> Duration {
        drop(self.end);
        self.naps.join().expect("the witness returns")
    }
}

/// Runs `case`, whose bounds on the real clock ([`bound!`]) leave the
/// loop `room`. A host that holds this process longer breaks such a
/// bound with no fault of the loop's, so a try whose bound fails where
/// the witness saw a hold of half the room is made again, [`TRIES`] at
/// most. No hold explains any other failure: it fails the case at once.
fn unheld(name: &str, room: Duration, case: fn()) {
    for tried in 1..=TRIES {
        let witness = Witness::start();
        let failed = std::panic::catch_unwind(case);
        let held = witness.held();
        let Err(failure) = failed else {
            return;
        };
        let late = match failure.downcast::<Late>() {
            Ok(late) => late,
            Err(failure) => std::panic::resume_unwind(failure),
        };
        assert!(held >= room / 2 && tried < TRIES, "{}", late.0);
        writeln!(
            std::io::stdout(),
            "{name}: made again in this run: the host held the test {held:?}: {}",
            late.0
        )
        .expect("the test's output");
    }
}

/// A lease whose first `failing` releases fail, the first of them after
/// `stalled`; every release is stamped as its write begins.
struct FailsReleases {
    lease: MemoryLease,
    failing: u32,
    stalled: Duration,
    releases: Arc<Mutex<Vec<Instant>>>,
}

impl LeaseBackend for FailsReleases {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        self.lease.read()
    }

    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<ConditionalWriteOutcome, ElectError> {
        if !record.holder.is_empty() {
            return self.lease.write(expected, record);
        }
        self.releases.lock().push(Instant::now());
        if self.failing == 0 {
            return self.lease.write(expected, record);
        }
        self.failing -= 1;
        thread::sleep(std::mem::take(&mut self.stalled));
        Err(ElectError::Io {
            place: self.place(),
            source: std::io::Error::other("the store is away"),
        })
    }

    fn place(&self) -> String {
        self.lease.place()
    }
}

/// What the loop logged: the writer of a subscriber the case sets on the
/// thread it runs the loop on.
#[derive(Clone)]
struct Logged(Arc<Mutex<Vec<u8>>>);

impl Write for Logged {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Waits through `count` more looks at the least: the loop publishes once
/// a look.
fn looks(stood: &mut watch::Receiver<ElectionObservation>, count: u32) {
    stood.borrow_and_update();
    let mut seen = 0;
    until("the looks", || {
        if stood.has_changed().expect("the loop runs") {
            stood.borrow_and_update();
            seen += 1;
        }
        seen >= count
    });
}

/// A store whose log followed `term` before any take.
fn followed(term: u64) -> Arc<MemoryStore> {
    let (store, feed) = journaled_from(&LeasePlace::Process, 0);
    feed.log
        .restart(Position { seq: 9, term })
        .expect("a place the log takes");
    store
}

/// What the leave's line says of a replica alone over its lease: its next
/// step takes the lease again. A stop read first leaves no next step.
const RETAKES: &str = "alone over its lease, takes it again at its next step";

/// How many lines of `text` hold `what`.
fn lines_with(text: &str, what: &str) -> usize {
    text.lines().filter(|line| line.contains(what)).count()
}

/// What each line of `text` that holds `stepped down` says behind its last
/// ` above=`, a line each: `None` for a line with no such text, else the
/// run of digits there as a term. A line with no digit there fails the
/// case. The identities of the cases that ask this hold no ` above=`.
fn left_under(text: &str) -> Vec<Option<u64>> {
    text.lines()
        .filter(|line| line.contains("stepped down"))
        .map(|line| {
            let (_, said) = line.rsplit_once(" above=")?;
            let digits = said.split(|at: char| !at.is_ascii_digit()).next()?;
            Some(digits.parse().expect("a number behind the line's `above`"))
        })
        .collect()
}

/// A lease whose every write takes `takes` to store while `slow` is set,
/// and when each began.
struct WritesSlowly {
    lease: MemoryLease,
    takes: Duration,
    slow: Arc<AtomicBool>,
    began: Arc<Mutex<Vec<Instant>>>,
}

impl LeaseBackend for WritesSlowly {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        self.lease.read()
    }

    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<ConditionalWriteOutcome, ElectError> {
        self.began.lock().push(Instant::now());
        if self.slow.load(Ordering::SeqCst) {
            thread::sleep(self.takes);
        }
        self.lease.write(expected, record)
    }

    fn place(&self) -> String {
        self.lease.place()
    }
}

/// The loop over `elector` on a thread of the case's own, under a
/// subscriber that keeps its lines. A second subscriber stays registered
/// beside it, for the reason the fallen term's case gives.
struct Looping {
    lines: Logged,
    stop: mpsc::Sender<Word>,
    thread: thread::JoinHandle<()>,
    _beside: tracing::Dispatch,
}

fn looping(elector: Elector, standing: watch::Sender<ElectionObservation>) -> Looping {
    looping_over(elector, Arc::new(MemoryStore::new()), standing)
}

/// The same over `store`: the loop tells it each take's term.
fn looping_over(
    elector: Elector,
    store: Arc<MemoryStore>,
    standing: watch::Sender<ElectionObservation>,
) -> Looping {
    looping_behind(elector, store, standing, Vec::new())
}

/// The same with `ahead` on the loop's channel before its thread starts: the
/// loop reads those words in their order, and no clock decides which step
/// meets them.
fn looping_behind(
    elector: Elector,
    store: Arc<MemoryStore>,
    standing: watch::Sender<ElectionObservation>,
    ahead: Vec<Word>,
) -> Looping {
    looping_on(elector, store, standing, ahead, false)
}

/// The same, with the channel `closed` behind `ahead` or left open. Closed,
/// no sender is left when the thread starts: a loop that reads past those
/// words finds the channel's end, as it does where the election's handle
/// is dropped and no notifier holds the channel. Nothing can be sent to
/// such a loop: [`Looping::ended`] waits it out.
fn looping_on(
    elector: Elector,
    store: Arc<MemoryStore>,
    standing: watch::Sender<ElectionObservation>,
    ahead: Vec<Word>,
    closed: bool,
) -> Looping {
    let lines = Logged(Arc::new(Mutex::new(Vec::new())));
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let (stop, stopped) = mpsc::channel();
    for word in ahead {
        stop.send(word)
            .expect("the channel's other end stands here");
    }
    // Closed, the sender the words went through is gone before the thread
    // starts: the one kept beside the loop reaches nobody.
    let stop = if closed {
        drop(stop);
        mpsc::channel().0
    } else {
        stop
    };
    let thread = thread::spawn(move || {
        tracing::subscriber::with_default(subscriber, || {
            run(elector, &store, &standing, &stopped);
        });
    });
    Looping {
        lines,
        stop,
        thread,
        _beside: beside,
    }
}

impl Looping {
    /// The lines so far hold `words`.
    fn said(&self, words: &str) -> bool {
        String::from_utf8_lossy(&self.lines.0.lock()).contains(words)
    }

    /// How many of the lines so far hold `words`.
    fn count(&self, words: &str) -> usize {
        lines_with(&String::from_utf8_lossy(&self.lines.0.lock()), words)
    }

    /// Waits out, bounded, the end of a loop that a stop on its channel
    /// ended, or the channel's end: the whole text.
    fn ended(self) -> String {
        until("the loop's end at its stop or its channel's end", || {
            self.thread.is_finished()
        });
        self.thread.join().expect("the loop ends");
        String::from_utf8(self.lines.0.lock().clone()).expect("the lines are text")
    }

    /// Stops the loop, waits its end out, bounded, and counts the lines
    /// that hold `words`; the whole text comes back for a failure's
    /// message.
    fn stopped_with(self, words: &str) -> (usize, String) {
        self.stop.send(Word::Stop).expect("the loop runs");
        until("the loop's end after its stop", || {
            self.thread.is_finished()
        });
        self.thread.join().expect("the loop ends");
        let text = String::from_utf8(self.lines.0.lock().clone()).expect("the lines are text");
        (lines_with(&text, words), text)
    }
}

/// A lease in this process that a case can lose: after `lose` it holds
/// no record, as a deleted object or a deleted file. Every write that
/// lands is counted. A case can also say what the next looks find: one
/// record stored before each read.
#[derive(Clone, Default)]
struct Losable {
    lease: MemoryLease,
    landed: Arc<AtomicU32>,
    ahead: Arc<Mutex<VecDeque<Ahead>>>,
    fails: Arc<AtomicBool>,
}

/// What another hand does to a [`Losable`] store ahead of the
/// candidate's next calls, in order: the first waits for its call.
enum Ahead {
    /// Before a read: a record stored, or nothing at this read.
    Read(Option<LeaseRecord>),
    /// Before a write: a record stored, so the write comes back moved.
    Write(LeaseRecord),
}

impl Losable {
    fn lose(&self) {
        self.lease.lose();
    }

    /// A record no candidate wrote, stored over the one there. A live
    /// holder's renewal can land between the read and the write, and
    /// the write is then turned away: both are made again.
    fn put(&self, record: &LeaseRecord) {
        let mut lease = self.lease.clone();
        until("a record stored over the one there", || {
            let stored = lease.read().expect("a read");
            lease.write(stored.as_ref(), record).expect("a write") == ConditionalWriteOutcome::Done
        });
    }

    /// Records no candidate wrote, stored one before each of the next
    /// reads, in order.
    fn put_before_the_next_looks(&self, records: impl IntoIterator<Item = LeaseRecord>) {
        self.ahead(records.into_iter().map(|record| Ahead::Read(Some(record))));
    }

    /// What another hand does ahead of the next calls, in order.
    fn ahead(&self, steps: impl IntoIterator<Item = Ahead>) {
        self.ahead.lock().extend(steps);
    }

    /// Every write fails from here as a store that is away, and stores
    /// nothing, until a case says otherwise: but a write the script
    /// names ([`Ahead::Write`]) is answered by the store it left.
    fn fail_writes(&self, fail: bool) {
        self.fails.store(fail, Ordering::SeqCst);
    }

    fn landed(&self) -> u32 {
        self.landed.load(Ordering::SeqCst)
    }
}

impl LeaseBackend for Losable {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        let next = {
            let mut ahead = self.ahead.lock();
            match ahead.front() {
                Some(Ahead::Read(_)) => ahead.pop_front(),
                _ => None,
            }
        };
        if let Some(Ahead::Read(Some(record))) = next {
            self.put(&record);
        }
        self.lease.read()
    }

    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<ConditionalWriteOutcome, ElectError> {
        let next = {
            let mut ahead = self.ahead.lock();
            match ahead.front() {
                Some(Ahead::Write(_)) => ahead.pop_front(),
                _ => None,
            }
        };
        // A write the script names is answered by the store it left,
        // whatever the failing mode: a case scripts a step in one call.
        if let Some(Ahead::Write(rival)) = next {
            self.put(&rival);
        } else if self.fails.load(Ordering::SeqCst) {
            return Err(ElectError::Io {
                place: self.place(),
                source: std::io::Error::other("the store is away"),
            });
        }
        let written = self.lease.write(expected, record)?;
        if written == ConditionalWriteOutcome::Done {
            self.landed.fetch_add(1, Ordering::SeqCst);
        }
        Ok(written)
    }

    fn place(&self) -> String {
        "a store that can lose its record".to_owned()
    }
}

/// The order of the design's settings at a fifth of their size, with a
/// short period.
fn brisk() -> ElectionSettings {
    ElectionSettings {
        duration: Duration::from_millis(600),
        renew_period: Duration::from_millis(20),
        renew_deadline: Duration::from_millis(400),
    }
}

/// A renew period of a minute, past the bound of every wait of a case:
/// after its first step nothing reads the lease but the case's own
/// word or stop.
fn unhurried() -> ElectionSettings {
    ElectionSettings {
        duration: Duration::from_secs(180),
        renew_period: Duration::from_secs(60),
        renew_deadline: Duration::from_secs(120),
    }
}
