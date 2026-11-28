//! A take over a store whose log followed a term: a take under that term, at
//! it and above it, a store that followed it as a standby, the last term and
//! the one below it, and the next take of a replica alone.

use super::*;

/// A lease that reads, at each release's write, whom the standing names
/// and the term the released record states.
struct SeesStanding {
    lease: MemoryLease,
    stood: watch::Receiver<ElectionObservation>,
    releases: Arc<Mutex<Vec<(String, u64)>>>,
}

impl LeaseBackend for SeesStanding {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        self.lease.read()
    }

    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<ConditionalWriteOutcome, ElectError> {
        if record.holder.is_empty() {
            let named = self.stood.borrow().holder.clone();
            self.releases.lock().push((named, record.term));
        }
        self.lease.write(expected, record)
    }

    fn place(&self) -> String {
        self.lease.place()
    }
}

/// A store that followed `term` as a standby's does: the start of a
/// leader's term, applied at the place the applier holds for it. The log
/// follows the term and the store's own term rises to it, where a restart
/// of the log alone leaves the store's term to the take.
fn followed_as_a_standby(term: u64) -> Arc<MemoryStore> {
    let (store, feed) = journaled_from(&LeasePlace::Process, 0);
    let at = Position { seq: 1, term };
    {
        let _place = feed.log.adopt(at).expect("a place the log takes");
        store.follow_term(at);
    }
    assert_eq!((store.term(), store.followed()), (term, Some(term)));
    store
}

/// A take of term 1 over `store`, whose log followed `followed_term`, a
/// term at or above it: another replica led that term, and the take starts
/// none in the log. The loop leaves the lease in the take's own step, ahead
/// of the standing, and the release states the higher of the two terms. The
/// next take is `next`, above the followed term, and the log leads it. One
/// warning says the take, and the leave's line says the term `said`.
fn a_take_under_a_followed_term_is_left(
    store: Arc<MemoryStore>,
    followed_term: u64,
    said: Option<u64>,
    next: u64,
) {
    let name = "b@127.0.0.1:2";
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let releases = Arc::new(Mutex::new(Vec::new()));
    let lease = SeesStanding {
        lease: MemoryLease::new(),
        stood: stood.clone(),
        releases: Arc::clone(&releases),
    };
    let candidate =
        Elector::new(Box::new(lease), name, brisk()).expect("the settings are in order");
    let looping = looping_over(candidate, Arc::clone(&store), standing);
    // No standing names the take of term 1: the first one that names this
    // replica is the next take's, one lease duration after the leave.
    until("the take above the followed term", || {
        stood.borrow().holder == name
    });
    assert_eq!(stood.borrow().term, next);
    assert_eq!(store.term(), next);
    assert_eq!(store.followed(), None);
    // One release so far, the leave's: under the followed term, written
    // while the standing named nobody.
    assert_eq!(*releases.lock(), [(String::new(), followed_term)]);
    until("the line of the lead", || looping.said(": leading"));
    let (warned, text) = looping.stopped_with("started no term");
    assert_eq!(warned, 1, "{text}");
    let count = |what: &str| lines_with(&text, what);
    assert_eq!(count(&format!("followed={followed_term}")), 1, "{text}");
    // One leave, and its line names the term said or none.
    assert_eq!(left_under(&text), [said], "{text}");
    // The take that was left says no lead and no wait for one: the next
    // take leads, once.
    assert_eq!(count(": leading"), 1, "{text}");
    assert_eq!(count("no holder to wait for"), 0, "{text}");
}

#[test]
fn a_take_under_a_term_its_log_followed_is_left_for_a_take_above_that_term() {
    a_take_under_a_followed_term_is_left(followed(5), 5, Some(5), 6);
}

#[test]
fn a_take_at_the_term_its_log_followed_is_left_with_no_term_said() {
    a_take_under_a_followed_term_is_left(followed(1), 1, None, 2);
}

/// A take under a term its store followed as a standby is left as one over
/// a log that was restarted is. The store stands at the followed term, and
/// a lower take does not lower it: the loop reads the followed term off the
/// log, whatever term the store holds.
#[test]
fn a_take_under_a_term_its_store_followed_as_a_standby_is_left() {
    a_take_under_a_followed_term_is_left(followed_as_a_standby(5), 5, Some(5), 6);
}

/// A take under the term below the last one, which its log followed, is
/// left: the last term is still there to take, and the next take is it. The
/// loop asks the elector's rule for the followed term itself, never for the
/// term above it.
#[test]
fn a_take_under_the_term_below_the_last_one_is_left_for_the_last_term() {
    let below = TERM_CEILING - 2;
    a_take_under_a_followed_term_is_left(followed(below), below, Some(below), below + 1);
}

/// No other candidate can take a lease the elector made itself. So the
/// step behind a take that started no term in the log follows at once, a
/// renew period of a minute ahead of the clock's next step, and takes
/// above the term the log followed.
#[test]
fn a_replica_alone_takes_again_at_once_above_the_term_its_log_followed() {
    let name = "a@127.0.0.1:1";
    let store = followed(5);
    let elector = Elector::alone(name, unhurried()).expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let looping = looping_over(elector, Arc::clone(&store), standing);
    until("the retake above the followed term", || {
        store.term() == 6 && stood.borrow().holder == name
    });
    assert_eq!(stood.borrow().term, 6);
    assert_eq!(store.followed(), None);
    until("the line of the lead", || looping.said(": leading"));
    let (said, text) = looping.stopped_with(RETAKES);
    assert_eq!(said, 1, "{text}");
    assert_eq!(text.matches("started no term").count(), 1, "{text}");
    assert_eq!(text.matches(": leading").count(), 1, "{text}");
}

/// The same at the take's own term. The log followed term 1, the first
/// take's: no term is said to the elector, and the leave is the replica's
/// own. The next step follows at once all the same, a renew period of a
/// minute ahead of the clock's next step, and takes term 2.
#[test]
fn a_replica_alone_takes_again_at_once_at_the_term_its_log_followed() {
    let name = "a@127.0.0.1:1";
    let store = followed(1);
    let elector = Elector::alone(name, unhurried()).expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let looping = looping_over(elector, Arc::clone(&store), standing);
    until("the retake above the followed term", || {
        store.term() == 2 && stood.borrow().holder == name
    });
    assert_eq!(stood.borrow().term, 2);
    assert_eq!(store.followed(), None);
    until("the line of the lead", || looping.said(": leading"));
    let (said, text) = looping.stopped_with(RETAKES);
    assert_eq!(said, 1, "{text}");
    assert_eq!(left_under(&text), [None], "{text}");
    assert_eq!(text.matches("started no term").count(), 1, "{text}");
    assert_eq!(text.matches(": leading").count(), 1, "{text}");
}

/// A take over a store whose log followed the last term: the next take
/// would stand above that term, and none is left. The election refuses the
/// followed term by the elector's rule, as a word of that term is refused.
/// The replica keeps the lease as it stands: the standing names it, its
/// log leads no term, and a second warning says so. No release is written
/// before the stop's. `released` is the term the lease was released under
/// before the take: the take is the term above it, and term 1 over no
/// record.
fn a_take_under_the_last_term_keeps_its_lease(released: Option<u64>) {
    let name = "b@127.0.0.1:2";
    let last = TERM_CEILING - 1;
    let store = followed(last);
    let mut held = MemoryLease::new();
    if let Some(term) = released {
        let released = LeaseRecord {
            term,
            ..LeaseRecord::default()
        };
        assert_eq!(
            held.write(None, &released).expect("a write"),
            ConditionalWriteOutcome::Done
        );
    }
    let taken = released.map_or(1, |term| term + 1);
    let (standing, mut stood) = watch::channel(ElectionObservation::default());
    let releases = Arc::new(Mutex::new(Vec::new()));
    let lease = SeesStanding {
        lease: held,
        stood: stood.clone(),
        releases: Arc::clone(&releases),
    };
    let candidate =
        Elector::new(Box::new(lease), name, brisk()).expect("the settings are in order");
    let looping = looping_over(candidate, Arc::clone(&store), standing);
    until("the take", || stood.borrow().holder == name);
    looks(&mut stood, 3);
    assert_eq!(
        *stood.borrow(),
        ElectionObservation {
            holder: name.to_owned(),
            term: taken,
            looked: true,
        }
    );
    assert_eq!(store.term(), taken);
    assert_eq!(store.followed(), Some(last));
    assert!(releases.lock().is_empty());
    until("the line of the lead", || looping.said(": leading"));
    let (said, text) = looping.stopped_with("the election refused the term the log followed");
    assert_eq!(said, 1, "{text}");
    assert_eq!(text.matches("started no term").count(), 1, "{text}");
    assert_eq!(text.matches("stepped down").count(), 0, "{text}");
    assert_eq!(text.matches(": leading").count(), 1, "{text}");
}

/// A followed term above the take's that leaves the election none to take
/// keeps the replica on its lease.
#[test]
fn a_followed_term_that_leaves_none_to_take_keeps_the_replica_on_its_lease() {
    a_take_under_the_last_term_keeps_its_lease(None);
}

/// A take at the last term, the very term its log followed, keeps the
/// replica on its lease too. The elector is said no term at the take's
/// own, so the loop asks the elector's rule itself: a lease left there
/// could be taken by no replica again.
#[test]
fn a_take_at_the_last_term_its_log_followed_keeps_the_replica_on_its_lease() {
    a_take_under_the_last_term_keeps_its_lease(Some(TERM_CEILING - 2));
}

/// A take above the term its log followed starts its term in the log: the
/// loop leaves nothing, and says nothing of the log. The lease was released
/// under term 7 before the take, so the take is term 8.
#[test]
fn a_take_above_the_term_its_log_followed_leaves_nothing() {
    let name = "b@127.0.0.1:2";
    let store = followed(5);
    let mut held = MemoryLease::new();
    let released = LeaseRecord {
        term: 7,
        ..LeaseRecord::default()
    };
    assert_eq!(
        held.write(None, &released).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let (standing, mut stood) = watch::channel(ElectionObservation::default());
    let releases = Arc::new(Mutex::new(Vec::new()));
    let lease = SeesStanding {
        lease: held,
        stood: stood.clone(),
        releases: Arc::clone(&releases),
    };
    let candidate =
        Elector::new(Box::new(lease), name, brisk()).expect("the settings are in order");
    let looping = looping_over(candidate, Arc::clone(&store), standing);
    until("the take", || stood.borrow().holder == name);
    looks(&mut stood, 3);
    assert_eq!(stood.borrow().term, 8);
    assert_eq!(store.term(), 8);
    assert_eq!(store.followed(), None);
    assert!(releases.lock().is_empty());
    let (said, text) = looping.stopped_with("started no term");
    assert_eq!(said, 0, "{text}");
    assert_eq!(text.matches("stepped down").count(), 0, "{text}");
}
