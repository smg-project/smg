//! A send told through a decision's handle, over the engine that decides
//! over worker lists.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use serde_json::json;
use smg_worker::{
    identity::basic::Worker,
    load::state::{one_rank_load, LoadSnapshot},
    test_support::ready_worker,
};

use super::{catalog, chosen, clock, document, one, send, validated, Readers, DELAY, POOL, TOKENS};
use crate::{
    document::{AppliedSpec, PolicySpec, API_VERSION},
    engine::{
        decision_engine::{Request, RequestMaker},
        in_process::{lists::PublishedList, InProcess, LendsLists, WorkerLists},
        Candidate, Constraints, Live, RequestFacts,
    },
    store::{GatewayId, StaleGeneration},
    tracking::{decision_guard::DecisionGuard, CreditedDispatches},
};

/// A request stated ahead, with the list its host hands for it.
struct Asks<'a>(Request<'a>, &'a PublishedList);

impl RequestMaker for Asks<'_> {
    fn model(&self) -> &str {
        self.0.facts.model
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        self.0
    }
}

impl LendsLists for Asks<'_> {
    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Single(self.1.lent())
    }
}

/// The engine of the standing document, two ready workers as it reads
/// their list, and the document's readers.
fn bed() -> (InProcess, PublishedList, Readers) {
    let (catalog, readers) = catalog();
    let engine = InProcess::new(validated(&document(None), &catalog));
    let list: Arc<[Arc<dyn Worker>]> = ["http://first:8000", "http://second:8000"]
        .iter()
        .map(|url| ready_worker(url, POOL))
        .collect();
    let members = PublishedList::new(list);
    (engine, members, readers)
}

/// A decision for a request of [`TOKENS`] tokens over `members`, held
/// by its handle.
fn decide<'a>(
    engine: &'a InProcess,
    members: &PublishedList,
    id: &[u8],
    now: Instant,
) -> DecisionGuard<&'a InProcess> {
    let request = Request {
        id,
        facts: RequestFacts {
            model: POOL,
            prompt_tokens: TOKENS,
            ..RequestFacts::default()
        },
        constraints: Constraints::default(),
    };
    let asks = Asks(request, members);
    DecisionGuard::decide_under(engine, GatewayId::LOCAL, &asks, now).expect("a decision")
}

/// A send told through a decision's handle is read by the profile's
/// reader, and the engine's next decision goes to the other worker.
#[test]
fn a_handle_dispatch_changes_the_next_in_process_pick() {
    let (engine, members, readers) = bed();
    let now = Instant::now();
    let mut first = decide(&engine, &members, b"first", now);
    let sent_to = chosen(&first);
    let sent = Arc::clone(&readers.lock()[0]);
    assert_eq!(
        sent.read(&clock(0), &sent_to),
        CreditedDispatches::default()
    );
    first.dispatched(now);
    let second = decide(&engine, &members, b"second", now);
    assert_ne!(chosen(&second).id, sent_to.id);
    assert_eq!(sent.read(&clock(0), &sent_to), one(TOKENS));
}

/// A send counted under one document is read under the next: where a
/// new document is put in force on the running engine, a profile's
/// count takes over the standing count.
#[test]
fn a_send_counted_under_one_document_is_read_under_the_next() {
    let (engine, members, _) = bed();
    let now = Instant::now();
    let mut first = decide(&engine, &members, b"first", now);
    let sent_to = chosen(&first);
    first.dispatched(now);

    let (catalog, later) = catalog();
    let mut next = document(None);
    next.generation += 1;
    let applied = engine.apply(validated(&next, &catalog));
    applied.expect("a number above the standing one");
    assert_eq!(later.lock()[0].read(&clock(0), &sent_to), one(TOKENS));
}

/// [`document`] under the number behind the standing one's.
fn next_document(limit: Option<u32>) -> PolicySpec {
    let mut next = document(limit);
    next.generation += 1;
    next
}

/// `next` put in force on `engine`, validated over a catalog of its
/// own: the readers of that document.
fn put(engine: &InProcess, next: &PolicySpec) -> Readers {
    let (catalog, readers) = catalog();
    let applied = engine.apply(validated(next, &catalog));
    applied.expect("a number above the standing one");
    readers
}

/// The engine with one send told under its standing document: the
/// worker the send went to, the list, and the standing readers.
fn sent_once() -> (InProcess, PublishedList, Readers, Candidate) {
    let (engine, members, readers) = bed();
    let now = Instant::now();
    let sent_to = {
        let mut first = decide(&engine, &members, b"before", now);
        first.dispatched(now);
        chosen(&first)
    };
    (engine, members, readers, sent_to)
}

/// The next document's time limit applies to the sends its profile
/// took over: a send ends at that limit for the new reader and stays
/// for the standing one.
#[test]
fn the_new_profiles_limit_applies_to_the_sends_it_took_over() {
    let (engine, members, readers) = bed();
    let now = Instant::now();
    let mut first = decide(&engine, &members, b"before", now);
    let sent_to = chosen(&first);
    let issued = first.lease().expect("the pick's lease").lease.issued_micros;
    first.dispatched(now);

    let limit = 29;
    let later = put(&engine, &next_document(Some(limit)));
    let edge = issued + u64::from(limit) * 1000;
    let (standing, next) = (&readers.lock()[0], &later.lock()[0]);
    assert_eq!(next.read(&clock(edge - 1), &sent_to), one(TOKENS));
    assert_eq!(standing.read(&clock(edge), &sent_to), one(TOKENS));
    let nothing = CreditedDispatches::default();
    assert_eq!(next.read(&clock(edge), &sent_to), nothing);
}

/// A document whose number is not above the standing one is refused,
/// and its profile took no send from the standing one.
#[test]
fn a_refused_document_takes_no_sends_from_the_standing_profile() {
    let (engine, _members, readers, sent_to) = sent_once();
    let (catalog, later) = catalog();
    let stale = document(None);
    let refused = StaleGeneration {
        standing: stale.generation,
        offered: stale.generation,
    };
    assert_eq!(engine.apply(validated(&stale, &catalog)), Err(refused));
    let nothing = CreditedDispatches::default();
    assert_eq!(later.lock()[0].read(&clock(0), &sent_to), nothing);
    assert_eq!(readers.lock()[0].read(&clock(0), &sent_to), one(TOKENS));
}

/// The standing profile and the one that took its count over read one
/// entry for a worker both knew. A send told to that worker under the
/// next document is read by the standing profile's reader too, which a
/// decision under way may still ask.
#[test]
fn a_send_under_the_next_document_is_read_by_the_standing_profiles_reader() {
    let (engine, _members, readers, sent_to) = sent_once();
    let later = put(&engine, &next_document(None));
    send(engine.standing_spec().engine(), &sent_to, 1, TOKENS);
    let both = CreditedDispatches {
        requests: 2,
        tokens: 2 * u64::from(TOKENS),
        ..CreditedDispatches::default()
    };
    for reader in [&readers.lock()[0], &later.lock()[0]] {
        assert_eq!(reader.read(&clock(0), &sent_to), both);
    }
}

/// A document with no reader keeps no count, so the document put in
/// force behind it starts with no sends.
#[test]
fn a_document_with_no_reader_leaves_the_sends_behind() {
    let (engine, _members, readers, sent_to) = sent_once();
    let unread: PolicySpec = serde_json::from_value(json!({
        "apiVersion": API_VERSION,
        "generation": 5,
        "profiles": { (POOL): { "pick": "random" } }
    }))
    .expect("a document");
    assert!(put(&engine, &unread).lock().is_empty());
    let mut next = document(None);
    next.generation = 6;
    let later = put(&engine, &next);
    let nothing = CreditedDispatches::default();
    assert_eq!(later.lock()[0].read(&clock(0), &sent_to), nothing);
    assert_eq!(readers.lock()[0].read(&clock(0), &sent_to), one(TOKENS));
}

/// A send said behind a new document, for a decision made under the
/// replaced one, is counted by the count in force: a send is a fact of
/// its worker, whatever document decided it. The later document was
/// validated over a catalog of its own, so its count is built new.
#[test]
fn a_send_said_behind_a_new_document_is_counted_by_the_count_in_force() {
    let (engine, members, _readers) = bed();
    let now = Instant::now();
    let mut first = decide(&engine, &members, b"first", now);
    let sent_to = chosen(&first);
    let later = put(&engine, &next_document(None));
    let nothing = CreditedDispatches::default();
    assert_eq!(later.lock()[0].read(&clock(0), &sent_to), nothing);

    first.dispatched(now);
    assert_eq!(later.lock()[0].read(&clock(0), &sent_to), one(TOKENS));
}

/// A send said behind a new document, for a decision the document
/// before made, is counted where the new document kept the profile's
/// count: both were validated over one catalog, and the profile's text
/// is the same. The reader the later validation built is let go, and
/// the count it was handed follows the standing one: it reads the same
/// send.
#[test]
fn a_send_said_behind_a_document_that_kept_its_count_is_counted() {
    let (catalog, readers) = catalog();
    let engine = InProcess::new(validated(&document(None), &catalog));
    let (_, members, _) = bed();
    let now = Instant::now();
    let mut first = decide(&engine, &members, b"first", now);
    let sent_to = chosen(&first);
    let applied = engine.apply(validated(&next_document(None), &catalog));
    applied.expect("a number above the standing one");

    first.dispatched(now);
    let readers = readers.lock();
    assert_eq!(readers[0].read(&clock(0), &sent_to), one(TOKENS));
    assert_eq!(readers[1].read(&clock(0), &sent_to), one(TOKENS));
}

/// A document validated and never put in force leaves the standing
/// count as it was, over one catalog and under the same text, where
/// an apply would keep the profile. A count follows another only
/// where an apply keeps its reader: the count that validation built
/// reads no send, and the standing count reads the send once.
#[test]
fn a_document_validated_and_never_applied_leaves_the_standing_count_as_it_was() {
    let (catalog, readers) = catalog();
    let engine = InProcess::new(validated(&document(None), &catalog));
    let (_, members, _) = bed();
    let now = Instant::now();
    let mut first = decide(&engine, &members, b"first", now);
    let sent_to = chosen(&first);
    let unapplied = validated(&next_document(None), &catalog);

    first.dispatched(now);
    let nothing = CreditedDispatches::default();
    let read = |at: usize| readers.lock()[at].read(&clock(0), &sent_to);
    assert_eq!((read(0), read(1)), (one(TOKENS), nothing));
    drop(unapplied);
    assert_eq!(read(0), one(TOKENS), "as it was");
}

/// A handle says its dispatch once, so a second word of the same send
/// counts nothing. The engine files no lease: the handle is the one
/// that tells it a word, and no entry takes a report by a lease's id.
#[test]
fn a_second_dispatch_word_of_one_handle_counts_nothing() {
    let (engine, members, readers) = bed();
    let now = Instant::now();
    let mut first = decide(&engine, &members, b"first", now);
    let sent_to = chosen(&first);
    first.dispatched(now);
    first.dispatched(now);
    assert_eq!(readers.lock()[0].read(&clock(0), &sent_to), one(TOKENS));
}

/// A report sampled behind a send ends the send's credit. A report
/// sampled ahead of the send, read later, leaves the credit ended: no
/// older report brings a send back.
#[test]
fn a_released_send_stays_retired_when_an_older_report_arrives() {
    let (engine, members, readers) = bed();
    let sent = Arc::clone(&readers.lock()[0]);
    let url = "http://first:8000";
    let time = engine.clock();
    let records = engine
        .members_of(members.lent())
        .records(engine.standing_spec(), time);
    let stages = engine.standing_spec().engine();

    let issued = Instant::now();
    let dispatched = issued + Duration::from_micros(u64::from(DELAY));
    let newer = dispatched + Duration::from_millis(13);
    let later = newer + Duration::from_millis(1);
    send(stages, &records.fixed()[0], time.micros(issued), TOKENS);

    // The first worker's report, sampled at `sampled_at` and read at
    // `now`: what the profile's reader counts for the worker then.
    let read = |sampled_at, running: u16, now| {
        let mut report = one_rank_load(0, 0.0, 0.0);
        report.loads[0].num_running_reqs = i32::from(running);
        report.sampled_at = Some(sampled_at);
        let loads = LoadSnapshot::from_loads_for_test(vec![(url.to_owned(), report)]);
        let live = Live::at(stages, time, now, Some(&loads), None);
        let row = records
            .read(members.workers(), &live, 0)
            .expect("the worker record");
        assert_eq!(row.load.expect("the report").running, u32::from(running));
        assert_eq!(row.load_sampled_at_micros, time.micros(sampled_at));
        sent.read(&clock(live.now_micros()), &row)
    };

    let empty = CreditedDispatches::default();
    assert_eq!(read(issued, 19, dispatched), one(TOKENS));
    assert_eq!(read(newer, 7, newer), empty);
    assert_eq!(read(issued, 19, later), empty, "the send stays retired");
}
