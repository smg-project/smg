use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Barrier,
    },
    thread,
};

use super::{InFlightCounts, WorkerSlot};
use crate::engine::view::Session;

/// Two files book one worker where it serves two models: a write is one
/// atomic step, so no booking of either writer is lost under the
/// other's, and nothing they give back.
#[test]
fn two_writers_of_one_cell_lose_no_count() {
    const EACH: u32 = 400_000;
    let counts = InFlightCounts::default();
    let slot = counts.claim();
    let start = Barrier::new(2);
    let both = |write: fn(&InFlightCounts, WorkerSlot, Session)| {
        thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| {
                    start.wait();
                    for _ in 0..EACH {
                        write(&counts, slot, Session::NONE);
                    }
                });
            }
        });
        counts.in_flight(slot)
    };
    assert_eq!(
        both(InFlightCounts::book),
        2 * EACH,
        "every booking of both"
    );
    assert_eq!(both(InFlightCounts::free), 0, "and each one given back");
}

/// A step the cell does not take is no step of the host's count: an
/// end of a cell at zero, and a booking of a cell that holds every
/// lease it can. A slot that names no cell keeps no count.
#[test]
fn a_step_the_cell_does_not_take_steps_no_hosts_count() {
    use std::sync::{atomic::AtomicU32, Arc};

    use super::HostCount;

    /// The bookings and the ends a count was told.
    #[derive(Default)]
    struct Told {
        bookings: AtomicU32,
        ends: AtomicU32,
    }

    struct Count(Arc<Told>);

    impl HostCount for Count {
        fn booked(&self) {
            self.0.bookings.fetch_add(1, Ordering::Relaxed);
        }

        fn freed(&self) {
            self.0.ends.fetch_add(1, Ordering::Relaxed);
        }
    }

    let counts = InFlightCounts::default();
    let slot = counts.claim();
    let told = Arc::new(Told::default());
    let count = || Box::new(Count(Arc::clone(&told)));
    assert!(!counts.beside(WorkerSlot::NONE, count()), "no cell");
    assert!(counts.beside(slot, count()));
    let read = || {
        let bookings = told.bookings.load(Ordering::Relaxed);
        (bookings, told.ends.load(Ordering::Relaxed))
    };
    counts.free(slot, Session::NONE);
    assert_eq!((counts.in_flight(slot), read()), (0, (0, 0)), "at zero");
    let cell = counts.cell(slot).expect("the slot's cell");
    cell.store(u32::MAX, Ordering::Relaxed);
    counts.book(slot, Session::NONE);
    assert_eq!(read(), (0, 0), "a cell that holds every lease it can");
    counts.free(slot, Session::NONE);
    assert_eq!((counts.in_flight(slot), read()), (u32::MAX - 1, (0, 1)));
    counts.book(slot, Session::NONE);
    assert_eq!(read(), (1, 1), "the last booking the cell takes");
}

/// Writers of one cell with no file's turn between them, while the
/// cell's count is handed and then while its member leaves: the count
/// takes what the cell holds at the hand-over and gives up what it
/// holds at the leaving, and no step reaches it half made.
#[test]
fn a_count_handed_and_taken_among_writers_holds_what_the_cell_holds() {
    use std::sync::{atomic::AtomicI64, Arc};

    use super::HostCount;

    /// The leases a count holds: never an end with no booking under it.
    struct Held(Arc<AtomicI64>);

    impl HostCount for Held {
        fn booked(&self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }

        fn freed(&self) {
            let held = self.0.fetch_sub(1, Ordering::SeqCst);
            assert!(held > 0, "an end with no booking under it");
        }
    }

    /// Says a writer ended, however it ended.
    struct Ended<'a>(&'a AtomicUsize);

    impl Drop for Ended<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    const WRITERS: usize = 4;
    const EACH: usize = 50_000;
    let counts = InFlightCounts::default();
    let slot = counts.claim();
    let held = Arc::new(AtomicI64::new(0));
    // Four writers book twice and free once, `EACH` times: `among`
    // runs once they are under way, or once each ended.
    let among_writers = |among: &dyn Fn()| {
        let (wrote, ended) = (AtomicUsize::new(0), AtomicUsize::new(0));
        thread::scope(|scope| {
            for _ in 0..WRITERS {
                scope.spawn(|| {
                    let _ended = Ended(&ended);
                    for _ in 0..EACH {
                        counts.book(slot, Session::NONE);
                        counts.book(slot, Session::NONE);
                        counts.free(slot, Session::NONE);
                        wrote.fetch_add(1, Ordering::SeqCst);
                    }
                });
            }
            let under_way = || wrote.load(Ordering::SeqCst) >= 1_000;
            while !under_way() && ended.load(Ordering::SeqCst) < WRITERS {
                thread::yield_now();
            }
            among();
        });
    };
    let wrote = i64::try_from(WRITERS * EACH).expect("a small count");

    among_writers(&|| {
        let count = Box::new(Held(Arc::clone(&held)));
        assert!(counts.beside(slot, count));
    });
    assert_eq!(i64::from(counts.in_flight(slot)), wrote);
    assert_eq!(held.load(Ordering::SeqCst), wrote, "what the cell holds");

    among_writers(&|| counts.left(slot));
    assert_eq!(i64::from(counts.in_flight(slot)), 2 * wrote);
    assert_eq!(held.load(Ordering::SeqCst), 0, "every lease given up");
}

/// A member that leaves gives its place's table back, with the
/// sessions of the leases that outlive it, and its place keeps no
/// session of a lease that books it behind the leaving. No count is
/// handed for such a member again, so nothing would take them, and a
/// place is never another member's: what it kept would stand for the
/// store's life. A member that stands keeps its table's room when its
/// last session leaves.
#[test]
fn a_member_that_left_keeps_no_session_and_no_table() {
    let counts = InFlightCounts::default();
    let slot = counts.claim();
    let kept = || (counts.sessions_kept(slot), counts.sessions_room(slot));
    for session in 1..=64 {
        counts.book(slot, Session::from_raw(session));
    }
    let (sessions, room) = kept();
    assert_eq!(sessions, 64);
    assert!(room >= 64, "room for the sessions that stand: {room}");
    for session in 1..=64 {
        counts.free(slot, Session::from_raw(session));
    }
    assert_eq!(kept(), (0, room), "a standing member keeps the room");
    for session in 1..=64 {
        counts.book(slot, Session::from_raw(session));
    }
    counts.left(slot);
    assert_eq!(kept(), (0, 0), "the table is given back at the leaving");
    // A lease that outlives the member ends, and a late one books it.
    counts.free(slot, Session::from_raw(1));
    counts.book(slot, Session::from_raw(65));
    assert_eq!(kept(), (0, 0), "no session is kept behind the leaving");
    assert_eq!(counts.in_flight(slot), 64, "the cell steps as it did");
}

/// A member's place keeps a session while a lease of it books the
/// member, and the session's entry leaves with its last lease. A lease
/// that counts for no key is kept under no session, and an end of a
/// session the place does not keep takes nothing from it.
#[test]
fn a_place_keeps_a_session_until_its_last_lease_ends() {
    let counts = InFlightCounts::default();
    let slot = counts.claim();
    let kept = || (counts.in_flight(slot), counts.sessions_kept(slot));
    let (one, two) = (Session::from_raw(7), Session::from_raw(9));
    counts.book(slot, Session::NONE);
    assert_eq!(kept(), (1, 0), "a lease that counts for no key");
    counts.book(slot, one);
    counts.book(slot, one);
    counts.book(slot, two);
    assert_eq!(kept(), (4, 2));
    counts.free(slot, one);
    assert_eq!(kept(), (3, 2), "a lease of the first still books it");
    counts.free(slot, two);
    counts.free(slot, Session::NONE);
    assert_eq!(kept(), (1, 1));
    counts.free(slot, one);
    assert_eq!(kept(), (0, 0), "the last keyed lease ended");
    // A move takes the session with the lease.
    let next = counts.claim();
    counts.book(slot, two);
    counts.moved(Some(slot), Some(next), two);
    assert_eq!((kept(), counts.sessions_kept(next)), ((0, 0), 1));
    counts.free(next, one);
    assert_eq!(counts.sessions_kept(next), 1, "a session it does not keep");
    counts.free(next, two);
    assert_eq!(counts.sessions_kept(next), 0);
}

/// A count handed in the place of another takes every step before the
/// one before it gives any up, the leases' and each session's alike: a
/// number the two counts share reads high between the two and never
/// low. A member that leaves gives up the same steps, and a session's
/// end after it reaches no count.
#[test]
fn a_count_handed_takes_its_steps_before_the_one_before_gives_them_up() {
    use std::sync::Arc;

    use parking_lot::Mutex;

    use super::HostCount;

    /// What the counts of one member were told, in order, each line
    /// under its count's name.
    struct Says(&'static str, Arc<Mutex<Vec<String>>>);

    impl Says {
        fn say(&self, step: &str) {
            self.1.lock().push(format!("{} {step}", self.0));
        }
    }

    impl HostCount for Says {
        fn booked(&self) {
            self.say("booked");
        }

        fn freed(&self) {
            self.say("freed");
        }

        fn session_booked(&self, session: Session) {
            self.say(&format!("booked {}", session.get()));
        }

        fn session_freed(&self, session: Session) {
            self.say(&format!("freed {}", session.get()));
        }
    }

    let counts = InFlightCounts::default();
    let slot = counts.claim();
    let said = Arc::new(Mutex::new(Vec::new()));
    let count = |name| Box::new(Says(name, Arc::clone(&said)));
    let told = || std::mem::take(&mut *said.lock());
    // A lease filed ahead of the member's count, then the count.
    counts.book(slot, Session::from_raw(7));
    assert!(counts.beside(slot, count("first")));
    assert_eq!(told(), ["first booked", "first booked 7"]);
    assert!(counts.beside(slot, count("second")));
    let handed = [
        "second booked",
        "second booked 7",
        "first freed",
        "first freed 7",
    ];
    assert_eq!(told(), handed);
    // An end of a session the place does not keep reaches no count:
    // its lease's own step alone does.
    counts.book(slot, Session::NONE);
    counts.free(slot, Session::from_raw(9));
    assert_eq!(told(), ["second booked", "second freed"]);
    counts.left(slot);
    assert_eq!(told(), ["second freed", "second freed 7"]);
    // The lease outlives its member: its end steps the place alone.
    counts.free(slot, Session::from_raw(7));
    assert_eq!((told().len(), counts.sessions_kept(slot)), (0, 0));
}

/// A booking that moves is added before it is taken: a reader beside the
/// mover that reads the worker it leaves and then the worker it goes to
/// counts the request on one of the two or on both, never on neither.
/// Read where another thread sees a thread's writes in their order: a
/// machine that orders them more loosely may show the taking first. On
/// one core a reader seldom lands between the two steps, and the case
/// never fails a right order.
#[cfg(target_arch = "x86_64")]
#[test]
fn a_moved_booking_is_never_read_on_neither_worker() {
    const MOVES: usize = 50_000;
    let counts = InFlightCounts::default();
    // A pair of workers a move, the booking on the first of them.
    let pairs: Vec<[WorkerSlot; 2]> = (0..MOVES)
        .map(|_| [counts.claim(), counts.claim()])
        .collect();
    for [from, _] in &pairs {
        counts.book(*from, Session::NONE);
    }
    let read = |[from, to]: &[WorkerSlot; 2]| (counts.in_flight(*from), counts.in_flight(*to));
    let at = AtomicUsize::new(0);
    thread::scope(|scope| {
        scope.spawn(|| {
            for (pair, [from, to]) in pairs.iter().enumerate() {
                at.store(pair, Ordering::Relaxed);
                counts.moved(Some(*from), Some(*to), Session::NONE);
            }
            at.store(MOVES, Ordering::Relaxed);
        });
        while let Some(pair) = pairs.get(at.load(Ordering::Relaxed)) {
            let (left, came) = read(pair);
            assert!(left + came >= 1, "a moved booking read on neither worker");
        }
    });
    assert!(
        pairs.iter().all(|pair| read(pair) == (0, 1)),
        "each moved once"
    );
}
