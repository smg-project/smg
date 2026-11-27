use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;
use tokio::task::JoinSet;
use tonic::Code;

use super::*;
use crate::{
    engine::view::WorkerId,
    service::{
        controller::decision_slots::tests::holder,
        elector::{ElectionSettings, Elector, TickOutcome},
        proto::replication_event::Event as Sent,
        replication,
    },
    stages::registry,
    store::{
        LeadWord, LeasesStore, MembershipChange, Registrant, WorkerFacts, WorkerRow, WorkerStatus,
        WorkersStore,
    },
    worker_host::tests::stand,
};

/// The same over one file of leases with the log's places starting after
/// `first`: zero in the cases, whose tables read a log's places from one.
pub(in crate::service::controller) fn journaled_from(
    place: &LeasePlace,
    first: u64,
) -> (Arc<MemoryStore>, watch_workers::Feed) {
    journaled_at(place, first, 1)
}

impl Replication {
    /// A restore's hold as a stream that took no part since `at` leaves
    /// it, over the cut at the head with no admission: a served case's own.
    pub(in crate::service::controller) fn stalled_since(&self, at: Instant) {
        let cut = self.cuts.at_head(&self.objects, &self.store, &self.log);
        self.cuts.stalled(cut, at);
    }

    /// How many restores this leader holds: a served case's own.
    pub(in crate::service::controller) fn held(&self) -> usize {
        self.cuts.held()
    }
}

impl Hold {
    /// A hold of `cut` from `now`, with no room taken first: a case's own.
    fn of(cut: Arc<StoreCheckpoint>, now: Instant) -> Self {
        Self(Mutex::new(Held::Cut(cut, now)))
    }
}

impl Kept {
    /// A hold of `cut` as a stream that took no part since `at` leaves
    /// it, with no room counted: a case's own.
    fn stalled(&self, cut: Arc<StoreCheckpoint>, at: Instant) {
        self.holds.lock().push(Arc::new(Hold::of(cut, at)));
    }

    /// How many holds stand: a case's own.
    fn held(&self) -> usize {
        self.holds.lock().len()
    }
}

impl Parts {
    /// The parts of `cut` from `now`, counted by no leader: a case's own.
    fn of(cut: Arc<StoreCheckpoint>, now: Instant) -> Self {
        Self::held(Arc::new(Hold::of(cut, now)))
    }
}

/// Every part of the snapshot of `cut`, as sent at `now`: the cases'
/// leader.
pub(in crate::service::controller) fn parts(
    cut: Arc<StoreCheckpoint>,
    now: Instant,
) -> Vec<proto::ReplicationBatch> {
    let mut parts = Parts::of(cut, now);
    let mut sent = Vec::new();
    while let Some((batch, ends)) = parts.next(now) {
        sent.push(batch);
        if ends.is_some() {
            break;
        }
    }
    sent
}

const IDENTITY: &str = "controller-0@10.0.0.7:30100";

/// The standby the cases' feeds name.
const STANDBY: &str = "controller-1";

/// A replica that took a free lease of its own at `at`, and a store
/// that journals as a run's does under a shared lease.
struct Leader {
    elector: Elector,
    store: Arc<MemoryStore>,
    /// The holder of the leader's worker objects: the cases write its
    /// rows through it, and its cuts are taken from it.
    objects: Arc<Objects<Members>>,
    log: Arc<Log>,
    term: u64,
    at: Instant,
}

impl Leader {
    /// The take, not yet told to the store, over a log whose places
    /// start at the start: the cases' tables read them from one.
    fn elected() -> Self {
        Self::over(journaled_from(&LeasePlace::File("shared".into()), 0))
    }

    /// The same over a store and its feed as a run builds them.
    fn over((store, feed): (Arc<MemoryStore>, watch_workers::Feed)) -> Self {
        let mut elector = Elector::alone(IDENTITY, ElectionSettings::default())
            .expect("the default settings are in order");
        let at = Instant::now();
        let Ok(TickOutcome::Acquired { term }) = elector.tick(at) else {
            panic!("a free lease is taken at the first tick");
        };
        Self {
            elector,
            objects: holder(&store),
            store,
            log: feed.log,
            term,
            at,
        }
    }

    /// The take with its write: the store is under the leader's term.
    fn leading() -> Self {
        let leader = Self::elected();
        leader.store.set_term(leader.term);
        leader
    }

    /// The holder of the leader's worker objects, for a cut.
    fn objects(&self) -> Arc<Objects<Members>> {
        Arc::clone(&self.objects)
    }

    /// `facts` written through the leader's holder, as its registration
    /// call writes a row: the row's id.
    fn registers(&self, facts: WorkerFacts) -> WorkerId {
        let written = self.objects.upsert(&self.store, facts, Registrant::Control);
        written.expect("the store takes the write").id()
    }

    fn replication(&self) -> Replication {
        Replication::new(
            Arc::clone(&self.store),
            Arc::clone(&self.log),
            self.elector.fence(),
            self.objects(),
        )
    }

    /// A standby's feed from `place`.
    fn feed(&self, place: Position, backlog: u64) -> Feed {
        Feed {
            term: self.term,
            place,
            acked: place.seq,
            backlog,
            full: None,
            beat: self.at,
            entries: Vec::new(),
            source: self.replication(),
            standby: STANDBY.to_owned(),
            restore: Restore::No,
            restored: None,
        }
    }

    /// One worker and `more` watched writes behind it: generations by
    /// their numbers alone, each one small entry of the log.
    fn writes(&self, more: u32) -> WorkerId {
        let id = self.registers(WorkerFacts {
            key: "http://a:8000".to_owned(),
            engine: "http://a:8000".to_owned(),
            incarnation: 1,
            models: vec!["m".to_owned()],
            dp_size: 1,
            ..WorkerFacts::default()
        });
        for generation in 1..=more {
            Journal::generation(&*self.log, None, u64::from(generation));
        }
        id
    }

    /// Rows for a snapshot of two parts: one more than a page of
    /// rows holds, which is 128.
    fn writes_a_page_and_one(&self) {
        for n in 0..129 {
            let key = format!("http://w{n:03}:8000");
            let facts = WorkerFacts {
                key: key.clone(),
                engine: key,
                incarnation: 1,
                models: vec!["m".to_owned()],
                dp_size: 1,
                ..WorkerFacts::default()
            };
            let _id = self.registers(facts);
        }
    }
}

const START: Position = Position { seq: 0, term: 0 };
const RING: u64 = WATCHED_EVENTS as u64;

fn batch(step: Step) -> proto::ReplicationBatch {
    match step {
        Step::Batch(batch) => batch,
        other => panic!("a batch, not {other:?}"),
    }
}

/// Why the look at `at` says the word to re-sync: a batch that says
/// it, the log's head and nothing else.
fn resync(feed: &mut Feed, at: Instant) -> Why {
    let word = proto::ReplicationBatch {
        resync_required: true,
        head_seq: feed.source.log.head().seq,
        ..proto::ReplicationBatch::default()
    };
    match (feed.look(at), &feed.restore) {
        (Step::Batch(said), Restore::Cut(why)) if said == word => *why,
        other => panic!("the word to re-sync and its reason, not {other:?}"),
    }
}

fn cases(batch: &proto::ReplicationBatch) -> Vec<&'static str> {
    batch
        .events
        .iter()
        .map(|event| match &event.event {
            Some(Sent::Membership(_)) => "membership",
            Some(Sent::Load(_)) => "load",
            Some(Sent::TermStart(_)) => "term",
            Some(Sent::Generation(_)) => "generation",
            Some(_) => "another",
            None => "unset",
        })
        .collect()
}

#[test]
fn a_batch_follows_its_readers_place_one_position_each() {
    let at = |seq| Entry {
        at: Position { seq, term: 3 },
        event: replication::Event::Term,
    };
    let place = Position { seq: 7, term: 3 };
    assert!(follows(place, &[]));
    assert!(follows(place, &[at(8), at(9), at(10)]));
    assert!(
        !follows(place, &[at(7), at(8)]),
        "the reader's own place again"
    );
    assert!(
        !follows(place, &[at(9), at(10)]),
        "a position was passed over"
    );
    assert!(!follows(place, &[at(8), at(10)]), "a gap inside the batch");
    assert!(!follows(place, &[at(8), at(8)]), "a position twice");
}

#[test]
fn a_burst_above_one_batch_arrives_in_consecutive_batches_with_no_idle_look_between() {
    let leader = Leader::leading();
    // The take, the registration and a thousand generations.
    leader.writes(1_000);
    let mut feed = leader.feed(START, RING);
    let mut sizes = Vec::new();
    let mut next = 1;
    // Four batches and the idle look; a feed that never rested would
    // be looked at without end.
    for _ in 0..8 {
        match feed.look(leader.at) {
            Step::Batch(batch) => {
                assert_eq!(batch.first_seq, next, "a batch starts where the last ended");
                assert_eq!(batch.term, leader.term);
                next += batch.events.len() as u64;
                sizes.push(batch.events.len());
            }
            Step::Idle => break,
            other @ Step::Close(_) => panic!("a batch or the end of the burst, not {other:?}"),
        }
    }
    assert_eq!(sizes, [256, 256, 256, 234]);
    assert_eq!(feed.place, leader.log.head());
}

#[test]
fn a_catch_up_across_a_terms_start_is_two_batches_each_under_its_write_term() {
    let leader = Leader::leading();
    leader.writes(2);
    // The store writes on under a later term: the log holds both.
    let later = leader.term + 3;
    leader.store.set_term(later);
    leader.writes(1);
    let mut feed = leader.feed(START, RING);
    let first = batch(feed.look(leader.at));
    assert_eq!((first.first_seq, first.term), (1, leader.term));
    assert_eq!(
        cases(&first),
        ["term", "membership", "generation", "generation"]
    );
    let second = batch(feed.look(leader.at));
    assert_eq!((second.first_seq, second.term), (5, later));
    // The same registration again is a relabel: a membership event.
    assert_eq!(
        cases(&second),
        ["term", "membership", "generation"],
        "the start is the first event of its batch"
    );
    assert!(matches!(feed.look(leader.at), Step::Idle));
}

#[test]
fn a_feed_is_closed_once_the_fence_no_longer_says_leader() {
    let leader = Leader::leading();
    leader.writes(1);
    let mut feed = leader.feed(START, RING);
    let past = leader.at + ElectionSettings::default().renew_deadline + Duration::from_millis(1);
    match feed.look(past) {
        Step::Close(status) => assert_eq!(status.code(), Code::Aborted),
        other => panic!("the stream's end, not {other:?}"),
    }
    assert_eq!(feed.place, START, "nothing was sent past the fence");
}

#[test]
fn events_follow_a_place_at_the_positions_after_it_one_each_or_not_at_all() {
    let leader = Leader::leading();
    leader.writes(3);
    let mut entries = Vec::new();
    leader
        .log
        .read(START, Log::BATCH, &mut entries)
        .expect("the log holds its start");
    assert!(entries.len() > 2, "{}", entries.len());
    assert!(follows(START, &entries));
    assert!(follows(START, &[]), "no event leaves no gap");
    assert!(follows(entries[0].at, &entries[1..]));
    // A gap, a repeat and a start past the place: each is another
    // history, which a batch's first position alone would not show.
    let mut gap = entries.clone();
    gap.remove(1);
    assert!(!follows(START, &gap));
    let mut twice = entries.clone();
    twice.insert(1, entries[0].clone());
    assert!(!follows(START, &twice));
    assert!(!follows(START, &entries[1..]));
}

#[test]
fn a_replica_deposed_while_it_awaited_the_hello_joins_no_feed() {
    let leader = Leader::leading();
    leader.writes(4);
    let replication = leader.replication();
    let held = leader.log.head();
    let lost = Position {
        seq: held.seq + 1,
        ..held
    };
    let says = |replica: &str, place: Position, seen: u64| proto::ReplicaHello {
        replica_id: replica.to_owned(),
        last_applied_seq: place.seq,
        term_seen: seen,
        max_backlog: 0,
        last_applied_term: place.term,
    };
    let joins = |hello: &proto::ReplicaHello, now| {
        replication
            .joins(leader.term, STANDBY, hello, now)
            .map(|feed| feed.restore)
            .map_err(|status| status.code())
    };
    // While the fence stands each hello meets its own rule.
    let term = held.term;
    let stands = leader.at;
    assert!(matches!(
        joins(&says(STANDBY, held, term), stands),
        Ok(Restore::No)
    ));
    assert!(matches!(
        joins(&says(STANDBY, lost, term), stands),
        Ok(Restore::Word(_))
    ));
    assert!(matches!(
        joins(&says(STANDBY, held, term + 1), stands),
        Err(Code::FailedPrecondition)
    ));
    // Once it closed, inside the wait for the hello, each is refused
    // alike: no feed is joined and no standby is told to re-sync.
    let past = leader.at + ElectionSettings::default().renew_deadline + Duration::from_millis(1);
    for (place, seen) in [(held, term), (lost, term), (held, term + 1)] {
        assert!(
            matches!(joins(&says(STANDBY, place, seen), past), Err(Code::Aborted)),
            "{place:?} under a floor of {seen}"
        );
    }
    // The principal is judged before the fence is read.
    for now in [stands, past] {
        assert!(matches!(
            joins(&says("controller-2", held, term), now),
            Err(Code::Unauthenticated)
        ));
    }
}

#[test]
fn a_standby_is_fed_no_further_than_its_backlog_and_cut_a_beat_after_it_fills() {
    let leader = Leader::leading();
    leader.writes(1_000);
    let mut feed = leader.feed(START, 300);
    // One batch, then the 44 events of room the bound leaves.
    assert_eq!(batch(feed.look(leader.at)).events.len(), 256);
    let rest = batch(feed.look(leader.at));
    assert_eq!((rest.first_seq, rest.events.len()), (257, 44));
    // A full backlog is read no further. It waits one beat for an
    // acknowledgement, then the standby is told to re-sync.
    assert!(matches!(feed.look(leader.at), Step::Idle));
    let short = leader.at + BEAT - Duration::from_millis(1);
    assert!(matches!(feed.look(short), Step::Idle));
    assert_eq!(feed.place.seq, 300);
    assert!(matches!(
        resync(&mut feed, leader.at + BEAT),
        Why::Backlog(_)
    ));

    // An acknowledgement makes room, and the wait starts again at the
    // look that finds the backlog full.
    let mut feed = leader.feed(START, 300);
    assert_eq!(batch(feed.look(leader.at)).events.len(), 256);
    assert_eq!(batch(feed.look(leader.at)).events.len(), 44);
    assert!(matches!(feed.look(leader.at), Step::Idle));
    feed.ack(100);
    let later = leader.at + BEAT;
    let more = batch(feed.look(later));
    assert_eq!((more.first_seq, more.events.len()), (301, 100));
    assert!(matches!(feed.look(later), Step::Idle));
    let short = later + BEAT - Duration::from_millis(1);
    assert!(matches!(feed.look(short), Step::Idle));
    assert!(matches!(resync(&mut feed, later + BEAT), Why::Backlog(_)));

    // One event acknowledged is room for one.
    let mut feed = leader.feed(START, 300);
    assert_eq!(batch(feed.look(leader.at)).events.len(), 256);
    assert_eq!(batch(feed.look(leader.at)).events.len(), 44);
    feed.ack(1);
    let one = batch(feed.look(leader.at));
    assert_eq!((one.first_seq, one.events.len()), (301, 1));
    assert!(matches!(feed.look(leader.at), Step::Idle));

    // An ack never runs past what was sent.
    let mut feed = leader.feed(START, 300);
    assert_eq!(batch(feed.look(leader.at)).events.len(), 256);
    feed.ack(u64::MAX);
    assert_eq!(feed.acked, 256);
    assert_eq!(batch(feed.look(leader.at)).first_seq, 257);
    // An older acknowledgement moves nothing back.
    feed.ack(100);
    assert_eq!(feed.acked, 256);
}

#[test]
fn a_bound_under_one_batch_is_fed_in_its_own_steps_to_the_head() {
    let leader = Leader::leading();
    // The take, the registration and a thousand generations against a
    // bound of 64, each batch acknowledged.
    leader.writes(1_000);
    let mut feed = leader.feed(START, 64);
    let mut sent = 0;
    for _ in 0..64 {
        match feed.look(leader.at) {
            Step::Batch(batch) => {
                assert_eq!(batch.first_seq, sent + 1);
                assert!(batch.events.len() <= 64, "{}", batch.events.len());
                sent += batch.events.len() as u64;
                feed.ack(sent);
            }
            Step::Idle => break,
            other @ Step::Close(_) => panic!("a batch or the head, not {other:?}"),
        }
    }
    assert_eq!(sent, 1_002);
}

#[test]
fn a_batch_of_wide_replacements_is_cut_at_one_message() {
    let leader = Leader::leading();
    // 257 registrations of one worker, each a later incarnation of a row
    // that holds three labels at a value's bound: an event carries the
    // row it replaced beside the row it wrote.
    for incarnation in 1..=257 {
        leader.registers(WorkerFacts {
            key: "http://a:8000".to_owned(),
            engine: "http://a:8000".to_owned(),
            incarnation,
            models: vec!["m".to_owned()],
            dp_size: 1,
            labels: ["a", "b", "c"]
                .map(|key| (key.to_owned(), "v".repeat(4_096)))
                .into(),
            ..WorkerFacts::default()
        });
    }
    let mut feed = leader.feed(START, RING);
    let mut sizes = Vec::new();
    let mut next = 1;
    for _ in 0..8 {
        match feed.look(leader.at) {
            Step::Batch(batch) => {
                assert_eq!(batch.first_seq, next, "a batch starts where the last ended");
                next += batch.events.len() as u64;
                sizes.push(batch.encoded_len());
            }
            Step::Idle => break,
            other @ Step::Close(_) => panic!("a batch or the head, not {other:?}"),
        }
    }
    // Every event was sent. Together they weigh more than one message
    // and less than two: two batches, each one the transport reads.
    assert_eq!(next, 259);
    let whole: usize = sizes.iter().sum();
    assert!(
        (MESSAGE_BYTES..2 * MESSAGE_BYTES).contains(&whole),
        "{whole}"
    );
    assert_eq!(sizes.len(), 2, "{sizes:?}");
    assert!(sizes.iter().all(|size| *size <= MESSAGE_BYTES), "{sizes:?}");
}

#[test]
fn a_batch_stops_at_its_bound_to_the_byte_and_holds_its_first_event_whatever_it_weighs() {
    let leader = Leader::leading();
    // The take, the registration and three generations: five events.
    leader.writes(3);
    let mut entries = Vec::new();
    let read = leader.log.read(START, Log::BATCH, &mut entries);
    assert!(read.is_ok());
    let first = entries.first().expect("the take's event").at;
    let head = leader.log.head();
    let (all, last) = fitting(first, head.seq, &entries, usize::MAX, None);
    assert_eq!((all.len(), last), (5, head));
    // What a batch of the first `count` events weighs on the wire.
    let weighs = |count: usize| {
        proto::ReplicationBatch {
            first_seq: first.seq,
            term: first.term,
            events: all.iter().take(count).cloned().collect(),
            resync_required: false,
            snapshot: None,
            head_seq: head.seq,
        }
        .encoded_len()
    };
    for count in 2..=5 {
        let (fits, last) = fitting(first, head.seq, &entries, weighs(count), None);
        assert_eq!(fits.len(), count);
        assert_eq!(last.seq, first.seq + count as u64 - 1);
        let (short, _) = fitting(first, head.seq, &entries, weighs(count) - 1, None);
        assert_eq!(short.len(), count - 1, "one byte under {count} events");
    }
    // No bound is under the first event: a batch always moves the place.
    let (alone, last) = fitting(first, head.seq, &entries, 0, None);
    assert_eq!((alone.len(), last), (1, first));
}

#[test]
fn a_standby_that_joins_far_along_owes_nothing_it_applied() {
    let leader = Leader::leading();
    leader.writes(1_000);
    // 900 events applied and a bound of 64: the bound counts from the
    // standby's place.
    let place = Position {
        seq: 900,
        term: leader.term,
    };
    let mut feed = Feed::join(
        &leader.replication(),
        STANDBY,
        leader.term,
        place,
        64,
        leader.at,
    );
    assert!(matches!(feed.restore, Restore::No));
    assert_eq!(feed.acked, 900);
    // The room is the bound's own: 64 events, not the batch's 256.
    let first = batch(feed.look(leader.at));
    assert_eq!((first.first_seq, first.events.len()), (901, 64));
}

#[test]
fn a_hello_joins_where_the_log_holds_its_place_under_a_term() {
    // Written before the take: positions under no term, which the log
    // holds and no standby may stand at.
    let leader = Leader::elected();
    leader.writes(1);
    leader.store.set_term(leader.term);
    let head = leader.log.head();
    let join = |place| {
        Feed::join(
            &leader.replication(),
            STANDBY,
            leader.term,
            place,
            0,
            leader.at,
        )
    };
    assert!(matches!(join(head).restore, Restore::No));
    assert!(matches!(join(START).restore, Restore::No));
    for place in [
        Position { seq: 1, term: 0 },
        Position {
            seq: head.seq + 1,
            ..head
        },
        Position {
            term: head.term + 1,
            ..head
        },
    ] {
        // The word is owed, and the feed's first look says it.
        let mut feed = join(place);
        assert!(
            matches!(feed.restore, Restore::Word(why) if why == Why::Place(place)),
            "{place:?} is no place to join at"
        );
        assert_eq!(resync(&mut feed, leader.at), Why::Place(place));
    }
}

/// A shared lease whose record was lost counts its terms again, so two
/// runs lead under one term. Each run's log starts after a number of
/// its own: a standby's place of the run before, or a watcher's, names
/// no write of this run and is told to start again, where one count of
/// places would have continued it.
#[test]
fn a_place_of_the_run_before_under_an_equal_term_is_told_to_start_again() {
    let run = || {
        let leader = Leader::over(journaled(&LeasePlace::File("shared".into()), 1));
        leader.store.set_term(leader.term);
        leader.writes(3);
        leader
    };
    let (before, after) = (run(), run());
    let held = before.log.head();
    let head = after.log.head();
    assert_eq!(held.term, head.term, "one term, two runs");
    let join = |place| {
        Feed::join(
            &after.replication(),
            STANDBY,
            after.term,
            place,
            0,
            after.at,
        )
    };
    match join(held).restore {
        Restore::Word(why) => assert_eq!(why, Why::Place(held)),
        _ => panic!("{held:?} is a place of the run before, this run's head is {head:?}"),
    }
    assert!(matches!(join(head).restore, Restore::No));
    // The log is a watcher's judge as well, and a generation's place is
    // one it holds: this run's head, and no place of the run before.
    assert!(after.log.holds(head), "{head:?}");
    assert!(!after.log.holds(held), "{held:?}");

    // A standby that applied nothing names no place of either run: it
    // stands at this run's first place and reads its writes from there.
    let first = after.log.first().expect("every write is retained");
    assert_eq!(first.term, 0, "{first:?}");
    let mut feed = join(START);
    assert!(matches!(feed.restore, Restore::No));
    assert_eq!(feed.place, first);
    let sent = batch(feed.look(after.at));
    assert_eq!((sent.first_seq, sent.term), (first.seq + 1, after.term));
    assert_eq!(
        cases(&sent),
        [
            "term",
            "membership",
            "generation",
            "generation",
            "generation"
        ]
    );
    // Position zero under a named term states a place, and no run wrote
    // one there: it is no standby that applied nothing.
    let named = Position {
        seq: 0,
        term: head.term,
    };
    assert!(matches!(join(named).restore, Restore::Word(_)));
    // A log that restarted at a snapshot's place is no longer its
    // store's whole history: that standby starts from a snapshot.
    after
        .log
        .restart(Position {
            seq: head.seq + 9,
            term: head.term + 1,
        })
        .expect("a place the log takes");
    assert!(matches!(join(START).restore, Restore::Word(_)));
}

#[test]
fn a_bound_of_zero_is_the_logs_own_and_no_bound_is_above_it() {
    let leader = Leader::leading();
    let bound = |asked| {
        Feed::join(
            &leader.replication(),
            STANDBY,
            leader.term,
            START,
            asked,
            leader.at,
        )
        .backlog
    };
    assert_eq!(bound(0), RING);
    assert_eq!(bound(300), 300);
    assert_eq!(bound(u32::MAX), RING);
}

#[test]
fn a_feed_ends_when_its_replica_leads_again_under_a_later_term() {
    let mut leader = Leader::leading();
    leader.writes(1);
    let mut feed = leader.feed(START, RING);
    // The lease is given up and taken again: a later term.
    assert!(leader.elector.release().expect("the lease's own store"));
    let Ok(TickOutcome::Acquired { term }) = leader.elector.tick(leader.at) else {
        panic!("a released lease is taken at the next tick");
    };
    assert!(term > leader.term);
    match feed.look(leader.at) {
        Step::Close(status) => assert_eq!(status.code(), Code::Aborted),
        other => panic!("the stream's end, not {other:?}"),
    }
}

#[test]
fn every_frame_states_the_logs_head_as_its_look_found_it() {
    let leader = Leader::leading();
    // The take, the registration and 300 generations: more than a batch.
    leader.writes(300);
    let head = leader.log.head();
    let ends = |batch: &proto::ReplicationBatch| batch.first_seq + batch.events.len() as u64 - 1;
    let mut feed = leader.feed(START, RING);
    // A batch short of the head states a head past its last event: a
    // standby that applied it is not there yet.
    let first = batch(feed.look(leader.at));
    assert_eq!(first.events.len(), Log::BATCH);
    assert!(ends(&first) < head.seq);
    assert_eq!(first.head_seq, head.seq);
    // The batch that ends at the head states its own last position,
    let second = batch(feed.look(leader.at));
    assert_eq!((ends(&second), second.head_seq), (head.seq, head.seq));
    // and the heartbeat the position before its own.
    let beat = batch(feed.look(leader.at + BEAT));
    assert!(beat.events.is_empty());
    assert_eq!((beat.first_seq - 1, beat.head_seq), (head.seq, head.seq));
    // The word to re-sync and a snapshot's part read no event of the
    // log: each states the head of its own instant.
    let nowhere = Position {
        seq: head.seq + 9,
        term: head.term,
    };
    let mut lost = leader.feed(nowhere, RING);
    let word = batch(lost.look(leader.at));
    assert!(word.resync_required);
    assert_eq!(word.head_seq, head.seq);
    let part = batch(lost.look(leader.at));
    assert!(part.snapshot.is_some());
    assert_eq!(part.head_seq, head.seq);
}

#[test]
fn a_batch_puts_the_heartbeat_off_a_second() {
    let leader = Leader::leading();
    let mut feed = leader.feed(START, RING);
    // The take's own event, sent 600 ms after the hello.
    let sent = leader.at + Duration::from_millis(600);
    assert!(!batch(feed.look(sent)).events.is_empty());
    // A second after the hello, 600 ms after the batch: silent still.
    assert!(matches!(
        feed.look(leader.at + Duration::from_millis(1_200)),
        Step::Idle
    ));
    assert!(batch(feed.look(sent + BEAT)).events.is_empty());
}

#[test]
fn a_store_above_one_message_is_sent_in_parts_of_a_page_each() {
    let leader = Leader::leading();
    // 400 workers whose rows hold three labels at a value's bound: more
    // rows than one message holds.
    for n in 0..400 {
        let key = format!("http://w{n:03}:8000");
        let facts = WorkerFacts {
            key: key.clone(),
            engine: key,
            incarnation: 1,
            models: vec!["m".to_owned()],
            dp_size: 1,
            labels: ["a", "b", "c"]
                .map(|key| (key.to_owned(), "v".repeat(4_096)))
                .into(),
            ..WorkerFacts::default()
        };
        let _id = leader.registers(facts);
    }
    let cut = leader.replication().cut(leader.term).expect("a cut");
    let sent = parts(Arc::clone(&cut), leader.at);
    let sizes: Vec<usize> = sent.iter().map(|part| part.encoded_len()).collect();
    let whole_bytes: usize = sizes.iter().sum();
    assert!(whole_bytes > MESSAGE_BYTES, "{whole_bytes}");
    // Each part is a page of rows at most, under the batch's own fields:
    // half of what the transport reads.
    let page = watch_workers::PAGE_BYTES + 64;
    assert!(sizes.iter().all(|size| *size <= page), "{sizes:?}");
    // Every part says the cut's place and its epoch, in order, and the
    // one that ends the rows is the last.
    let at = (cut.at.seq + 1, cut.at.term);
    for (number, batch) in sent.iter().enumerate() {
        let part = batch.snapshot.as_ref().expect("a part");
        assert_eq!((batch.first_seq, batch.term), at, "part {number}");
        assert_eq!(
            (part.part as usize, part.membership_epoch),
            (number, cut.epoch)
        );
        assert_eq!(part.last, number + 1 == sent.len(), "part {number}");
    }
    // The parts are the cut whole, in its order: the rows, and no load
    // row.
    let (rows, loads) = whole(&sent);
    let told: Vec<u64> = rows.iter().map(|row| row.id).collect();
    let held: Vec<u64> = cut.rows.iter().map(|row| row.id.0).collect();
    assert_eq!(told, held);
    assert_eq!((rows.len(), loads.len()), (400, 0));
}

#[test]
fn a_store_that_holds_nothing_is_one_empty_part() {
    let leader = Leader::leading();
    let cut = leader.replication().cut(leader.term).expect("a cut");
    let sent = parts(cut, leader.at);
    assert_eq!(sent.len(), 1);
    let part = sent[0].snapshot.as_ref().expect("a part");
    assert!(part.last && part.workers.is_empty() && part.loads.is_empty());
    let head = leader.log.head();
    assert_eq!((sent[0].first_seq, sent[0].term), (head.seq + 1, head.term));
}

#[test]
fn a_cut_is_shared_by_the_streams_restored_at_its_place_and_kept_by_none() {
    let leader = Leader::leading();
    leader.writes(1);
    let replication = leader.replication();
    let first = replication.cut(leader.term).expect("a cut");
    let second = replication.cut(leader.term).expect("a cut");
    assert!(Arc::ptr_eq(&first, &second), "one copy at one place");
    // A stream holds a clone of the call: the same kept cut.
    let third = replication.clone().cut(leader.term).expect("a cut");
    assert!(Arc::ptr_eq(&first, &third));
    // A write moves the head: the next stream takes its own cut there.
    leader.writes(1);
    let later = replication.cut(leader.term).expect("a cut");
    assert!(!Arc::ptr_eq(&first, &later));
    assert_eq!(later.at, leader.log.head());
    // Nothing is kept past the last stream that holds a cut: the next
    // at the same place copies the rows again.
    let held = Arc::downgrade(&later);
    drop(later);
    assert!(held.upgrade().is_none(), "the rows are let go");
    let again = replication.cut(leader.term).expect("a cut");
    assert_eq!(again.at, leader.log.head());
}

#[test]
fn a_place_the_log_does_not_hold_is_told_to_resync() {
    let leader = Leader::leading();
    leader.writes(4);
    let held = leader.log.head();
    for place in [
        Position {
            term: held.term + 1,
            ..held
        },
        Position {
            seq: held.seq + 1,
            ..held
        },
    ] {
        let mut feed = leader.feed(place, RING);
        assert_eq!(resync(&mut feed, leader.at), Why::Place(place));
    }
    assert!(matches!(
        leader.feed(held, RING).look(leader.at),
        Step::Idle
    ));
}

/// A place past the head of `leader`'s log: one its log does not hold.
fn past_the_head(leader: &Leader) -> Position {
    let head = leader.log.head();
    Position {
        seq: head.seq + 9,
        ..head
    }
}

#[test]
fn a_standby_the_log_does_not_continue_is_restored_by_the_looks_after_the_word() {
    let leader = Leader::leading();
    leader.writes(3);
    let head = leader.log.head();
    let lost = past_the_head(&leader);
    let mut feed = leader.feed(lost, RING);
    assert_eq!(resync(&mut feed, leader.at), Why::Place(lost));
    // The snapshot is one part, the row's, and the last.
    let rows = batch(feed.look(leader.at));
    let part = rows.snapshot.as_ref().expect("the rows' part");
    assert_eq!((rows.first_seq, rows.term), (head.seq + 1, head.term));
    assert_eq!((part.part, part.last, part.workers.len()), (0, true, 1));
    assert!(part.loads.is_empty(), "a leader sends no load row");
    // The standby stands at the snapshot's place and owes nothing: the
    // log follows from there.
    assert_eq!((feed.place, feed.acked), (head, head.seq));
    assert!(matches!(feed.look(leader.at), Step::Idle));
    Journal::generation(&*leader.log, None, 9);
    let next = batch(feed.look(leader.at));
    assert_eq!((next.first_seq, next.events.len()), (head.seq + 1, 1));
    assert!(next.snapshot.is_none());
}

#[test]
fn a_repeat_restore_on_a_stream_waits_a_beat_from_the_end_of_the_last() {
    let leader = Leader::leading();
    leader.writes(3);
    let lost = past_the_head(&leader);
    let at = leader.at;
    let mut feed = leader.feed(lost, RING);
    // The stream's first restore: the word, and its cut at the next look.
    assert_eq!(resync(&mut feed, at), Why::Place(lost));
    while !matches!(feed.restore, Restore::No) {
        assert!(batch(feed.look(at)).snapshot.is_some());
    }
    // Writes outran the snapshot's send, and the log no longer holds
    // the place it left the standby at: the word again,
    feed.place = lost;
    assert_eq!(resync(&mut feed, at), Why::Place(lost));
    // and the cut a beat from the first restore's last part, no sooner.
    assert!(matches!(feed.look(at), Step::Idle));
    let short = at + BEAT - Duration::from_millis(1);
    assert!(matches!(feed.look(short), Step::Idle));
    assert!(batch(feed.look(at + BEAT)).snapshot.is_some());
}

#[test]
fn a_leader_holds_four_restores_and_the_next_waits_for_one_to_end() {
    let leader = Leader::leading();
    let source = leader.replication();
    let (term, at) = (leader.term, leader.at);
    let restore = |now| {
        let admitted = source.restore(term, now).expect("the term is served");
        admitted.map(|(parts, _)| parts)
    };
    // A cut that is refused gives its room back: five refusals leave
    // the leader's four rooms free.
    for _ in 0..5 {
        assert!(source.restore(term + 1, at).is_err());
    }
    // Four restores under way, each at a head of its own: four by the
    // case's own count, the number the contract tells a standby.
    let mut held = Vec::new();
    for _ in 0..4 {
        leader.writes(1);
        held.push(restore(at).expect("room for a restore"));
    }
    // One more waits, and the leader cuts nothing for it. A term that
    // is not served is refused there as its cut would be.
    assert!(restore(at).is_none());
    assert!(source.restore(term + 1, at).is_err());
    // A stream that ends inside its restore gives its room up,
    drop(held.pop());
    let mut fifth = restore(at).expect("room once a restore ended");
    assert!(restore(at).is_none());
    // and so does one that took its last part.
    while let Some((_, ends)) = fifth.next(at) {
        if ends.is_some() {
            break;
        }
    }
    assert!(fifth.next(at).is_none(), "nothing follows the last part");
    assert!(
        restore(at).is_some(),
        "room once a restore's last part left"
    );
}

/// A restore past the four waits on its stream, and the word is said
/// again a beat after the stream's last frame: the standby's quiet
/// bound, three beats, does not lapse while the cut waits.
#[test]
fn a_restore_that_waits_for_room_is_told_the_word_again_each_beat() {
    let leader = Leader::leading();
    leader.writes(3);
    let (term, at) = (leader.term, leader.at);
    let lost = past_the_head(&leader);
    let mut feed = leader.feed(lost, RING);
    let source = feed.source.clone();
    // Four restores under way: the leader holds as many as it may.
    let mut held = Vec::new();
    for _ in 0..4 {
        held.push(source.restore(term, at).expect("the term is served"));
    }
    assert!(held.iter().all(Option::is_some));
    // The fifth standby is told the word half a beat after its
    // stream's last frame, and its cut waits in silence short of a
    // beat from the word: the word is the frame the beat counts from.
    let said = at + BEAT / 2;
    assert_eq!(resync(&mut feed, said), Why::Place(lost));
    assert!(matches!(feed.look(said), Step::Idle));
    let short = said + BEAT - Duration::from_millis(1);
    assert!(matches!(feed.look(short), Step::Idle));
    // A beat from the word it is said again, and the cut waits on. The
    // looks stay inside the lead's window, which ends two beats after
    // the take.
    assert_eq!(resync(&mut feed, said + BEAT), Why::Place(lost));
    assert!(matches!(feed.look(said + BEAT), Step::Idle));
    // Room: the next look makes the cut and sends its first part.
    held.pop();
    assert!(batch(feed.look(said + BEAT)).snapshot.is_some());
}

/// A room taken with no cut made yet is a hold all the same: a sweep
/// leaves it, and a restore beside four rooms waits.
#[test]
fn a_room_with_no_cut_yet_is_held_through_a_sweep() {
    let leader = Leader::leading();
    leader.writes(3);
    let source = leader.replication();
    source
        .cuts
        .holds
        .lock()
        .extend((0..RESTORES).map(|_| Arc::new(Hold::default())));
    // A room dates no part: no instant lets it go.
    source.cuts.sweep(leader.at + STALLED + STALLED);
    assert_eq!(source.cuts.holds.lock().len(), RESTORES);
    let fifth = source
        .restore(leader.term, leader.at)
        .expect("the term is served");
    assert!(fifth.is_none(), "a restore beside four rooms waits");
}

#[test]
fn a_restore_whose_stream_took_no_part_for_the_bound_lets_its_cut_go() {
    let leader = Leader::leading();
    leader.writes_a_page_and_one();
    let at = leader.at;
    let lost = past_the_head(&leader);
    // A stream inside its restore: the word, the cut and its first part.
    let mut feed = leader.feed(lost, RING);
    let source = feed.source.clone();
    assert_eq!(resync(&mut feed, at), Why::Place(lost));
    assert!(batch(feed.look(at)).snapshot.is_some());
    assert!(matches!(feed.restore, Restore::Parts(_)));
    let held = || source.cuts.holds.lock().len();
    // Short of the bound the leader's clock leaves the hold: 60 s by
    // the case's own number, the one the contract tells a standby,
    let bound = Duration::from_secs(60);
    source.cuts.sweep(at + bound - Duration::from_millis(1));
    assert_eq!(held(), 1);
    // and at the bound it lets the cut go, though no look came.
    source.cuts.sweep(at + bound);
    assert_eq!(held(), 0);
    // The stream ends at its next look, where the transport asks one.
    match feed.look(at) {
        Step::Close(status) => assert_eq!(status.code(), Code::DeadlineExceeded),
        other => panic!("the stream's end, not {other:?}"),
    }
}

/// A part taken dates its hold anew: a stream that takes its parts a
/// beat short of the bound apart is never let go for its age.
#[test]
fn a_part_taken_renews_its_hold() {
    let leader = Leader::leading();
    leader.writes(3);
    let at = leader.at;
    let cut = leader.replication().cut(leader.term).expect("a cut");
    let hold = Hold::of(cut, at);
    let taken = at + STALLED - BEAT;
    assert!(hold.take(taken).is_some());
    // The bound counts from the part, not from the hold's start.
    assert!(hold.keeps(at + STALLED));
    assert!(hold.keeps(taken + STALLED - Duration::from_millis(1)));
    assert!(!hold.keeps(taken + STALLED));
    assert!(hold.take(taken + STALLED).is_none(), "the cut was let go");
}

#[tokio::test(start_paused = true)]
async fn the_restores_clock_lets_a_stalled_cut_go_with_no_look() {
    let leader = Leader::leading();
    leader.writes(3);
    let source = leader.replication();
    let now = tokio::time::Instant::now().into_std();
    let (mut parts, _) = source
        .restore(leader.term, now)
        .expect("the term is served")
        .expect("room");
    let (stop, stopped) = watch::channel(false);
    let mut task = JoinSet::new();
    let clock = source.clone();
    task.spawn(async move { clock.run(stopped).await });
    // A beat short of the bound the hold stands: its stream took a part.
    tokio::time::sleep(STALLED - BEAT).await;
    let taken = tokio::time::Instant::now().into_std();
    assert!(parts.next(taken).is_some());
    // The bound from that part on, and a beat for the clock's pass.
    tokio::time::sleep(STALLED + BEAT).await;
    let late = tokio::time::Instant::now().into_std();
    assert!(parts.next(late).is_none(), "the cut was let go");
    stop.send_replace(true);
    task.join_next()
        .await
        .expect("one task")
        .expect("the clock ends at the stop");
}

#[test]
fn a_replica_deposed_inside_a_restore_cuts_and_sends_no_further() {
    let leader = Leader::leading();
    leader.writes_a_page_and_one();
    let lost = past_the_head(&leader);
    let past = leader.at + ElectionSettings::default().renew_deadline + Duration::from_millis(1);
    let closed = |step: Step| match step {
        Step::Close(status) => assert_eq!(status.code(), Code::Aborted),
        other => panic!("the stream's end, not {other:?}"),
    };
    // Deposed between the word and the cut.
    let mut feed = leader.feed(lost, RING);
    assert_eq!(resync(&mut feed, leader.at), Why::Place(lost));
    closed(feed.look(past));
    // Deposed between two parts.
    let mut feed = leader.feed(lost, RING);
    assert_eq!(resync(&mut feed, leader.at), Why::Place(lost));
    assert!(batch(feed.look(leader.at)).snapshot.is_some());
    closed(feed.look(past));
    assert_eq!(feed.place, lost, "no standby is told it stands there");
}

#[test]
fn a_standby_cut_at_its_backlog_owes_nothing_after_its_restore() {
    let leader = Leader::leading();
    leader.writes(8);
    let mut feed = leader.feed(START, 4);
    // Four events sent and none acknowledged: the word a beat later.
    assert_eq!(batch(feed.look(leader.at)).events.len(), 4);
    assert!(matches!(feed.look(leader.at), Step::Idle));
    let later = leader.at + BEAT;
    assert_eq!(resync(&mut feed, later), Why::Backlog(4));
    // The snapshot, then the head: the backlog is empty again.
    let mut parts = 0;
    while batch(feed.look(later))
        .snapshot
        .is_some_and(|part| !part.last)
    {
        parts += 1;
    }
    assert_eq!(parts, 0, "one part, the rows', and it is the last");
    let head = leader.log.head();
    assert_eq!((feed.place, feed.acked), (head, head.seq));
    assert!(matches!(feed.look(later), Step::Idle));
}

#[test]
fn a_restore_this_replica_cuts_nothing_for_ends_the_stream() {
    // The store took a later term after the word: a cut would stand
    // under that term, not the feed's.
    let leader = Leader::leading();
    leader.writes(1);
    let lost = past_the_head(&leader);
    let mut feed = leader.feed(lost, RING);
    assert_eq!(resync(&mut feed, leader.at), Why::Place(lost));
    leader.store.set_term(leader.term + 1);
    match feed.look(leader.at) {
        Step::Close(status) => assert_eq!(status.code(), Code::Unavailable),
        other => panic!("the stream's end, not {other:?}"),
    }
}

#[test]
fn events_written_under_no_term_are_never_sent() {
    let leader = Leader::elected();
    // The store wrote before the take reached it.
    leader.writes(1);
    leader.store.set_term(leader.term);
    let mut feed = leader.feed(START, RING);
    assert_eq!(resync(&mut feed, leader.at), Why::NoTerm);
}

/// The rows and the load rows the parts hold together, in their order.
fn whole(sent: &[proto::ReplicationBatch]) -> (Vec<proto::WorkerRow>, Vec<proto::LoadSample>) {
    let parts = sent.iter().filter_map(|batch| batch.snapshot.as_ref());
    let rows = parts.clone().flat_map(|part| part.workers.clone());
    let loads = parts.flat_map(|part| part.loads.clone());
    (rows.collect(), loads.collect())
}

#[test]
fn a_snapshots_parts_are_the_stores_cut_at_its_place() {
    let leader = Leader::leading();
    let id = leader.writes(3);
    let head = leader.log.head();
    let cut = leader.replication().cut(leader.term).expect("a cut");
    assert_eq!((cut.at, cut.epoch), (head, leader.store.epoch()));
    // One part, the row's, and the last. It says the place as a
    // heartbeat says the head, and the epoch.
    let sent = parts(Arc::clone(&cut), leader.at);
    assert_eq!(sent.len(), 1);
    for (number, batch) in sent.iter().enumerate() {
        assert_eq!((batch.first_seq, batch.term), (head.seq + 1, head.term));
        assert!(batch.events.is_empty() && !batch.resync_required);
        let part = batch.snapshot.as_ref().expect("a part");
        assert_eq!((part.part as usize, part.last), (number, number == 0));
        assert_eq!(part.membership_epoch, leader.store.epoch());
    }
    let (rows, loads) = whole(&sent);
    assert_eq!(rows.iter().map(|row| row.id).collect::<Vec<_>>(), [id.0]);
    assert!(loads.is_empty(), "a leader cuts no load row into a part");

    // A row's admission crosses as its age at the part's send.
    let later = parts(cut, Instant::now() + Duration::from_secs(5));
    let age = whole(&later).0[0].admitted_age_ms;
    assert!((5_000..65_000).contains(&age), "{age} ms");
}

/// The leader's log as its store's journal, and one write that lands
/// as a cut reads its place: where a write of another thread can.
struct AtTheCut {
    log: Arc<Log>,
    lands: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

impl Journal for AtTheCut {
    fn member(
        &self,
        at: Option<Position>,
        term: Option<u64>,
        epoch: u64,
        change: MembershipChange,
        row: Option<&WorkerRow>,
        left: Option<&WorkerRow>,
    ) -> bool {
        self.log.member(at, term, epoch, change, row, left)
    }

    fn status(
        &self,
        at: Option<Position>,
        term: Option<u64>,
        epoch: u64,
        id: WorkerId,
        from: WorkerStatus,
        to: WorkerStatus,
        models: Arc<[String]>,
    ) -> bool {
        self.log.status(at, term, epoch, id, from, to, models)
    }

    fn term_rose(&self, term: u64) {
        self.log.term_rose(term);
    }

    fn generation(&self, at: Option<Position>, generation: u64) {
        self.log.generation(at, generation);
    }

    fn term_followed(&self, at: Position) {
        self.log.term_followed(at);
    }

    fn head(&self) -> Position {
        // Taken out before it runs: the write reads no head itself.
        let lands = self.lands.lock().take();
        if let Some(lands) = lands {
            lands();
        }
        self.log.head()
    }

    fn followed(&self) -> Option<u64> {
        self.log.followed()
    }

    fn released(&self) {
        self.log.released();
    }

    fn restart(&self, at: Position) -> bool {
        self.log.restart(at).is_ok()
    }
}

#[test]
fn a_snapshot_a_take_landed_inside_is_refused() {
    let leader = Leader::elected();
    let told = Arc::new(AtomicUsize::new(0));
    let log = {
        let told = Arc::clone(&told);
        Arc::new(Log::new(WATCHED_EVENTS).on_watched(move || {
            told.fetch_add(1, Ordering::Relaxed);
        }))
    };
    let journal = Arc::new(AtTheCut {
        log: Arc::clone(&log),
        lands: Mutex::default(),
    });
    let store = Arc::new(MemoryStore::journaled(Arc::clone(&journal) as _));
    store.set_term(leader.term);
    // The journal hands the store's release on to its log, or the
    // log's listener hears of no write: the term's start is told.
    assert_eq!(
        told.load(Ordering::Relaxed),
        1,
        "the release was not handed on"
    );
    let cut_from = holder(&store);
    let replication = Replication::new(Arc::clone(&store), log, leader.elector.fence(), cut_from);
    // No stream holds the cut past this line: the next call takes its
    // own.
    let served = replication.cut(leader.term).expect("a cut").at;
    assert_eq!(served.term, leader.term);

    // The store takes the next term after the call's admission, as the
    // cut reads its place: the place is that take's.
    let later = leader.term + 1;
    *journal.lands.lock() = Some(Box::new({
        let store = Arc::clone(&store);
        move || store.set_term(later)
    }));
    let refused = replication
        .cut(leader.term)
        .expect_err("a term beside a later term's place");
    assert_eq!(refused.code(), Code::Unavailable);
    assert_eq!(store.term(), later, "the take landed inside the call");
    // The later term's call is served at that place.
    let served = replication.cut(later).expect("the later term's snapshot");
    assert_eq!(served.at.term, later);
}

#[test]
fn a_silent_second_is_answered_with_the_heartbeat_at_the_head() {
    let leader = Leader::leading();
    leader.writes(1);
    let mut feed = leader.feed(START, RING);
    let sent = batch(feed.look(leader.at));
    let head = leader.log.head();
    assert_eq!(sent.first_seq + sent.events.len() as u64 - 1, head.seq);
    // Silent under a second: nothing. At the second: the heartbeat.
    let second = leader.at + BEAT;
    assert!(matches!(
        feed.look(second - Duration::from_millis(1)),
        Step::Idle
    ));
    let beat = batch(feed.look(second));
    assert_eq!((beat.first_seq, beat.term), (head.seq + 1, head.term));
    assert!(beat.events.is_empty() && !beat.resync_required);
    assert_eq!(feed.place, head, "a heartbeat moves no place");
    // The next is a second after it, and a write is sent at once.
    assert!(matches!(
        feed.look(second + Duration::from_millis(1)),
        Step::Idle
    ));
    leader.writes(0);
    let next = batch(feed.look(second + Duration::from_millis(2)));
    assert_eq!((next.first_seq, next.events.len()), (head.seq + 1, 1));
}

#[test]
fn a_leader_that_has_not_written_under_its_term_is_not_yet_served() {
    let leader = Leader::elected();
    let refused = leader
        .replication()
        .cut(leader.term)
        .expect_err("the take has not reached the store");
    assert_eq!(refused.code(), Code::Unavailable);
    assert!(
        refused.message().contains("not yet written under term 1"),
        "{}",
        refused.message()
    );
    // A replica that led before: its log leads the earlier term still,
    // and has seen no term as high as the later take's.
    leader.store.set_term(leader.term);
    let refused = leader
        .replication()
        .cut(leader.term + 1)
        .expect_err("the later take has not reached the store");
    assert_eq!(refused.code(), Code::Unavailable);
    assert!(
        refused.message().contains("not yet written under term 2"),
        "{}",
        refused.message()
    );
}

#[test]
fn a_leader_whose_take_started_no_term_in_its_log_serves_no_standby() {
    // The log followed term 7 before the take of term 1. The answer is
    // one before the store is told the take and after it: no write of
    // that take starts the term.
    let leader = Leader::elected();
    leader
        .log
        .restart(Position { seq: 9, term: 7 })
        .expect("a place the log takes");
    for told in [false, true] {
        if told {
            leader.store.set_term(leader.term);
        }
        let refused = leader
            .replication()
            .cut(leader.term)
            .expect_err("its writes stand under no term");
        assert_eq!(refused.code(), Code::Unavailable);
        let said = refused.message();
        assert!(
            said.contains("took term 1") && said.contains("followed term 7"),
            "the store told its take: {told}: {said}"
        );
    }

    // The same where the store holds the followed term, as a standby
    // that applied that term's start does: a take does not lower it,
    // and the refusal names both terms, never a write still to come.
    let leader = Leader::elected();
    leader
        .log
        .restart(Position { seq: 9, term: 7 })
        .expect("a place the log takes");
    leader.store.set_term(7);
    leader.store.set_term(leader.term);
    assert_eq!(leader.store.term(), 7);
    let refused = leader
        .replication()
        .cut(leader.term)
        .expect_err("its writes stand under no term");
    assert_eq!(refused.code(), Code::Unavailable);
    let said = refused.message();
    assert!(
        said.contains("took term 1") && said.contains("followed term 7"),
        "{said}"
    );
}

#[test]
fn a_store_under_a_later_term_than_the_calls_serves_no_call() {
    // The store took term 7 after the take of term 1 and its log
    // followed neither: a call admitted under term 1 is told so.
    let leader = Leader::leading();
    leader.writes(1);
    leader.store.set_term(7);
    let replication = leader.replication();
    let refused = replication.serves(leader.term).expect_err("a later take");
    assert_eq!(refused.code(), Code::Unavailable);
    let said = refused.message();
    assert!(
        said.contains("under term 7") && said.contains("a later take than term 1"),
        "{said}"
    );
    let refused = replication.cut(leader.term).expect_err("a later take");
    assert_eq!(refused.code(), Code::Unavailable);
    // The later term's own call is served, at that take's place.
    let served = replication.cut(7).expect("the later term's snapshot");
    assert_eq!(served.at.term, 7);
}

#[test]
fn a_controller_whose_lease_lives_in_its_process_serves_no_standby() {
    let leader = Leader::elected();
    let (store, feed) = journaled_from(&LeasePlace::Process, 0);
    store.set_term(leader.term);
    assert!(feed.log.alone());
    let cut_from = holder(&store);
    let refused = Replication::new(store, feed.log, leader.elector.fence(), cut_from)
        .cut(leader.term)
        .expect_err("no standby follows it");
    assert_eq!(refused.code(), Code::FailedPrecondition);
    assert_eq!(
        refused.message(),
        "this controller's lease lives in its own process: no standby follows it"
    );
}

#[test]
fn a_controller_under_a_shared_lease_serves_a_standby_from_a_log_of_the_watched_kinds() {
    let leader = Leader::elected();
    let (store, feed) = journaled_from(&LeasePlace::File("shared".into()), 0);
    store.set_term(leader.term);
    // The lease's place alone says whether a standby may follow the
    // log.
    assert!(!feed.log.alone());
    let head = feed.log.head();
    let cut_from = holder(&store);
    let cut = Replication::new(store, feed.log, leader.elector.fence(), cut_from)
        .cut(leader.term)
        .expect("a standby may follow it");
    assert_eq!(cut.at, head);
}

/// The stock door over the leader's store.
fn door(leader: &Leader) -> Arc<Generations> {
    let catalog = Arc::new(registry::StageRegistry::builtin());
    Arc::new(Generations::new(Arc::clone(&leader.store), catalog))
}

/// The generation an event states.
fn generation(event: &proto::ReplicationEvent) -> proto::GenerationEvent {
    match &event.event {
        Some(Sent::Generation(told)) => told.clone(),
        other => panic!("a generation, not {other:?}"),
    }
}

#[test]
fn a_stream_sends_a_generation_with_the_document_its_door_holds() {
    use crate::service::controller::generations::{
        tests::{LEAST_LOAD, RANDOM},
        Format,
    };

    let leader = Leader::leading();
    let documents = door(&leader);
    let apply = |document: &str| documents.apply(document.into(), Format::Yaml);
    apply(RANDOM).expect("a valid document");
    apply(LEAST_LOAD).expect("a changed document");
    let mut feed = leader.feed(START, RING);
    feed.source = feed.source.telling(Arc::clone(&documents));
    let sent = batch(feed.look(leader.at));
    assert_eq!(cases(&sent), ["term", "generation", "generation"]);
    // The first stood no longer at the send: its number alone.
    assert_eq!(generation(&sent.events[1]), worker_wire::generation_told(1));
    assert_eq!(Some(generation(&sent.events[2])), documents.standing_sent());
    // A source with no door sends each by its number alone.
    let sent = batch(leader.feed(START, RING).look(leader.at));
    assert_eq!(generation(&sent.events[2]), worker_wire::generation_told(2));
}

#[test]
fn a_snapshots_first_part_states_the_generation_that_stands_at_its_send() {
    use crate::service::controller::generations::{tests::LEAST_LOAD, Format};

    let leader = Leader::leading();
    let documents = door(&leader);
    leader.writes_a_page_and_one();
    let lost = past_the_head(&leader);
    let mut feed = leader.feed(lost, RING);
    feed.source = feed.source.telling(Arc::clone(&documents));
    assert_eq!(resync(&mut feed, leader.at), Why::Place(lost));
    // No document is held: the part states none.
    let rows = batch(feed.look(leader.at));
    let part = rows.snapshot.as_ref().expect("the rows' part");
    assert_eq!((part.part, part.generation.as_ref()), (0, None));
    // Applied inside the restore: the parts behind the first state none.
    documents
        .apply(LEAST_LOAD.into(), Format::Yaml)
        .expect("a valid document");
    let rest = batch(feed.look(leader.at));
    let part = rest.snapshot.as_ref().expect("the rows' second part");
    assert_eq!((part.part, part.generation.as_ref()), (1, None));
    // The log follows from the cut's place, the generation's event in it.
    let next = batch(feed.look(leader.at));
    assert_eq!(cases(&next), ["generation"]);
    assert_eq!(Some(generation(&next.events[0])), documents.standing_sent());

    // A restore that starts behind the apply states it on its first part.
    let mut feed = leader.feed(past_the_head(&leader), RING);
    feed.source = feed.source.telling(Arc::clone(&documents));
    resync(&mut feed, leader.at);
    let rows = batch(feed.look(leader.at));
    let part = rows.snapshot.as_ref().expect("the rows' part");
    assert_eq!(part.part, 0);
    assert_eq!(
        part.generation.as_deref(),
        documents.standing_sent().as_ref()
    );
}

/// A leader's first copy as the holder of its worker objects cuts it:
/// its place, which is the log's head, and its rows.
fn first_copy(leader: &Leader, objects: &Objects<Members>) -> (Position, Vec<WorkerRow>) {
    let cut = objects.checkpoint(&leader.store);
    assert_eq!(cut.at, leader.log.head(), "a copy stands at the log's head");
    (cut.at, cut.rows)
}

/// A first copy is cut in the turn of the worker objects' holder, which
/// every writer of a row's facts holds to the journal's telling. A
/// generation's apply moves the log's place with no turn of the holder.
/// It changes no row: a copy cut behind it holds every row, at the new
/// place.
#[test]
fn a_copy_behind_a_generations_apply_holds_every_row_at_the_new_place() {
    use crate::service::controller::generations::{tests::LEAST_LOAD, Format};

    let leader = Leader::leading();
    leader.writes_a_page_and_one();
    let objects = leader.objects();
    let (before, rows) = first_copy(&leader, &objects);
    assert_eq!(rows.len(), 129);
    door(&leader)
        .apply(LEAST_LOAD.into(), Format::Yaml)
        .expect("a valid document");
    let (behind, held) = first_copy(&leader, &objects);
    assert!(behind.seq > before.seq, "the apply moved the place");
    assert_eq!(held, rows);
}

/// A term's start moves the log's place with no turn of the holder, and
/// changes no row either.
#[test]
fn a_copy_behind_a_terms_start_holds_every_row_under_the_new_term() {
    let leader = Leader::leading();
    leader.writes_a_page_and_one();
    let objects = leader.objects();
    let (before, rows) = first_copy(&leader, &objects);
    let later = leader.term + 1;
    leader.store.set_term(later);
    let (behind, held) = first_copy(&leader, &objects);
    assert_eq!((before.term, behind.term), (leader.term, later));
    assert_eq!(held, rows);
}

/// A status tell takes no turn of the holder. It changes no fact of a
/// row, and a copy reads the told statuses with the place in one hold of
/// the store's lock: a copy cut behind the tell stands at the tell's
/// place and holds its status.
#[test]
fn a_copy_behind_a_status_tell_holds_the_told_status_at_its_place() {
    let leader = Leader::leading();
    let id = leader.writes(0);
    let objects = leader.objects();
    let (before, rows) = first_copy(&leader, &objects);
    let [row] = rows.as_slice() else {
        panic!("one row was written");
    };
    assert_eq!(row.status, WorkerStatus::Pending);
    // The store's entry a holder's tell ends in, with no turn here: the
    // models are the row's, as a holder reads them off its object.
    let (ready, models) = (WorkerStatus::Ready, row.facts.models.as_slice().into());
    let word = LeadWord::FREE;
    let told = leader
        .store
        .status_told(id, ready, row.revision, word, Some(models));
    assert!(told.moved().is_some(), "the store took the status");
    let (behind, held) = first_copy(&leader, &objects);
    assert!(behind.seq > before.seq, "the tell moved the place");
    let told = WorkerRow {
        status: ready,
        ..row.clone()
    };
    assert_eq!(held, [told]);
}

/// A status entry names the models its worker object served at the
/// write, and no row: the holder reads them under the object's lock
/// and hands them ahead of the store's. A row whose models change in
/// place behind a told status leaves that entry as it was told, and
/// the row's next status names the new ones.
#[test]
fn a_status_entry_names_the_models_its_row_served_at_the_write() {
    let leader = Leader::leading();
    let id = leader.writes(0);
    stand(&leader.objects, &leader.store, id, WorkerStatus::Ready);
    // The row's models change in place, and its object moves again.
    let stood = WorkersStore::get(leader.store.as_ref(), id).expect("the row");
    let mut wider = stood.facts;
    wider.models.push("n".to_owned());
    assert_eq!(leader.registers(wider), id, "a change in place");
    stand(&leader.objects, &leader.store, id, WorkerStatus::NotReady);
    let mut entries = Vec::new();
    leader
        .log
        .read(START, Log::BATCH, &mut entries)
        .expect("the log holds its start");
    let statuses = entries.iter().filter_map(|entry| match &entry.event {
        Written::Status { to, models, .. } => Some((*to, models.to_vec())),
        Written::Member { .. } | Written::Term | Written::Generation { .. } => None,
    });
    let (m, n) = ("m".to_owned(), "n".to_owned());
    let told = [
        (WorkerStatus::Ready, vec![m.clone()]),
        (WorkerStatus::NotReady, vec![m, n]),
    ];
    assert_eq!(statuses.collect::<Vec<_>>(), told);
}
