//! The accounting of leases: the count a row reads of the leases that book
//! its worker, the slot that holds a member's count across memberships and
//! builds, the doors that move a count or give it back, and the decode
//! sides on file beside the records that carry a pair's mark.

use super::*;

#[test]
fn a_rows_count_reads_the_leases_on_file_that_book_its_worker() {
    let store = MemoryStore::new();
    let (a, b) = (ready(&store, "http://a:1"), ready(&store, "http://b:1"));
    let (first, second) = (a.min(b), a.max(b));
    let (gateway, now) = (GatewayId(1), Instant::now());
    store.set_term(1);
    assert_eq!(in_flight(&store), [(first, 0), (second, 0)]);
    for (id, target) in [(1, a), (2, a), (3, b)] {
        LeasesStore::issue(&store, booking(gateway, id, target, now)).expect("filed");
    }
    let of = |a_holds, b_holds| {
        let mut rows = [(a, a_holds), (b, b_holds)];
        rows.sort_unstable();
        rows
    };
    assert_eq!(in_flight(&store), of(2, 1));
    // A dispatch moves no count: the lease stands on file.
    assert!(LeasesStore::dispatched(
        &store,
        gateway,
        LeaseId(1),
        Dispatch::default(),
        now
    ));
    assert_eq!(in_flight(&store), of(2, 1));
    // A count is its worker's through a membership write and a status.
    let c = ready(&store, "http://c:1");
    let mut rows = in_flight(&store);
    rows.retain(|&(id, _)| id != c);
    assert_eq!(rows, of(2, 1), "a member that joined moved no count");
    assert!(in_flight(&store).contains(&(c, 0)));
    // A lease of a worker that left ends on its own count, and no other
    // member's moves.
    let revision = WorkersStore::get(&store, b).expect("the row").revision;
    assert!(store.remove(b, revision).is_some());
    let leases: &dyn LeasesStore = &store;
    assert!(leases.released(gateway, LeaseId(3), None, ReleaseReason::WorkerRemoved));
    let mut left = [(a, 2), (c, 0)];
    left.sort_unstable();
    assert_eq!(in_flight(&store), left);
}

/// The counts are one table in pieces that never move: every slot names a
/// cell of its own in whichever piece holds it, a run of slots is read in
/// its order across the pieces, and the slot that names no cell reads zero
/// and books nothing.
#[test]
fn a_count_is_found_by_its_slot_in_whichever_piece_holds_it() {
    let counts = InFlightCounts::default();
    // The first piece holds 1,024 cells and the second 2,048: one slot
    // more stands in the third.
    let slots: Vec<WorkerSlot> = (0..3_073).map(|_| counts.claim()).collect();
    for &slot in &slots {
        counts.book(slot, Session::NONE);
    }
    assert!(counts.of(&slots).all(|count| count == 1), "a cell a slot");
    for &slot in &slots {
        counts.free(slot, Session::NONE);
    }
    // Both ends of each piece, a count of its own on each.
    let marked = [0, 1, 1_023, 1_024, 3_071, 3_072];
    for (times, &at) in (1..).zip(&marked) {
        for _ in 0..times {
            counts.book(slots[at], Session::NONE);
        }
    }
    counts.book(WorkerSlot::NONE, Session::NONE);
    let ends = marked.iter().rev().map(|&at| slots[at]);
    let run: Vec<WorkerSlot> = ends.chain([WorkerSlot::NONE, slots[2]]).collect();
    assert_eq!(
        counts.of(&run).collect::<Vec<_>>(),
        [6, 5, 4, 3, 2, 1, 0, 0]
    );
    // A lease that leaves gives one back, and a count at zero stays there.
    for slot in [slots[1_024], slots[2], WorkerSlot::NONE] {
        counts.free(slot, Session::NONE);
    }
    let read: Vec<u32> = counts.of(&[slots[1_024], slots[2]]).collect();
    assert_eq!(read, [3, 0]);
}

/// A member keeps its slot whichever member leaves or joins beside it, and
/// the slot of a worker that left is given to no member after it: a worker
/// that joins reads zero, and the leaver's leases end on the count they
/// booked.
#[test]
fn a_member_keeps_its_slot_and_a_leavers_slot_is_given_to_no_one() {
    let keys = ["http://a:1", "http://b:1", "http://c:1"];
    for leaver in 0..keys.len() {
        let store = MemoryStore::new();
        let ids: Vec<WorkerId> = keys.iter().map(|key| ready(&store, key)).collect();
        let leases: &dyn LeasesStore = &store;
        let (gateway, now) = (GatewayId(1), Instant::now());
        leases.set_term(1);
        // One lease on the first worker, two on the second, three on the
        // third: the leaver's are the ones its turn filed.
        let (mut id, mut its) = (0, Vec::new());
        for (count, &target) in (1..).zip(&ids) {
            for _ in 0..count {
                id += 1;
                let filed = leases.issue(booking(gateway, id, target, now));
                assert_eq!(filed, Ok(()));
                if target == ids[leaver] {
                    its.push(LeaseId(id));
                }
            }
        }
        let revision = WorkersStore::get(&store, ids[leaver]).expect("the row");
        assert!(store.remove(ids[leaver], revision.revision).is_some());
        let joined = ready(&store, "http://d:1");
        let stayers = (1..).zip(&ids).filter(|&(_, &id)| id != ids[leaver]);
        let mut stand: Vec<(WorkerId, u32)> = stayers.map(|(count, &id)| (id, count)).collect();
        stand.push((joined, 0));
        stand.sort_unstable();
        assert_eq!(in_flight(&store), stand, "worker {leaver} left");
        for lease in its {
            assert!(leases.released(gateway, lease, None, ReleaseReason::WorkerRemoved));
        }
        assert_eq!(in_flight(&store), stand, "worker {leaver}'s leases ended");
    }
}

/// Two leases filed through `file`, one on a member that is built and one
/// on a member whose first build is owed to the window. Each books its
/// worker's count and asks for no build: its host's count reads the lease
/// inside the window and behind the window's build, and each lease ends
/// on the count it booked.
fn filed_ahead_of_a_first_build(file: fn(&dyn LeasesStore, GatewayId, &[LeaseRecord; 2])) {
    let store = MemoryStore::new();
    let built = ready(&store, "http://a:1");
    // No build is asked for the second worker: its registration's is owed
    // to the window when its lease is filed.
    let owed = store.upsert(facts("http://b:1", 1)).id();
    let revision = WorkersStore::get(&store, owed).expect("the row").revision;
    store
        .set_status(owed, WorkerStatus::Ready, revision)
        .expect("ready");
    assert!(store.publish_due().is_some(), "a build is owed");
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let records = [
        booking(gateway, 1, built, now),
        booking(gateway, 2, owed, now),
    ];
    let builds = store.membership_builds();
    file(leases, gateway, &records);
    assert_eq!(
        (store.membership_builds(), store.publish_due().is_some()),
        (builds, true),
        "a lease asked for the build"
    );
    let of = |count| {
        let mut rows = [(built, count), (owed, count)];
        rows.sort_unstable();
        rows
    };
    assert_eq!(in_flight(&store), of(1), "inside the window");
    store.publish_now();
    assert_eq!(in_flight(&store), of(1));
    for id in [1, 2] {
        assert!(leases.released(gateway, LeaseId(id), None, ReleaseReason::ClientGone));
    }
    assert_eq!(in_flight(&store), of(0));
}

/// The issue door books the slot a member's write gave it, inside the
/// member's registration window too.
#[test]
fn a_lease_issued_ahead_of_its_workers_first_build_books_its_count() {
    filed_ahead_of_a_first_build(|leases, _, records| {
        for record in records {
            assert_eq!(leases.issue(*record), Ok(()));
        }
    });
}

/// The restore door reads the same slot: a gateway's pages may list a
/// lease on a member inside its registration window.
#[test]
fn a_lease_restored_ahead_of_its_workers_first_build_books_its_count() {
    filed_ahead_of_a_first_build(|leases, gateway, records| {
        assert_eq!(leases.restore(gateway, records), 2);
    });
}

/// No lease asks for a build: one on a worker the standing world lists
/// and one on a worker inside its registration window both leave the owed
/// build to its window.
#[test]
fn a_lease_issued_leaves_an_owed_build_to_its_window() {
    let store = MemoryStore::new();
    let built = ready(&store, "http://a:1");
    let owed = store.upsert(facts("http://b:1", 1)).id();
    assert!(store.publish_due().is_some(), "a build is owed");
    let builds = store.membership_builds();
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    assert_eq!(leases.issue(booking(gateway, 1, built, now)), Ok(()));
    assert_eq!(
        (store.membership_builds(), store.publish_due().is_some()),
        (builds, true),
        "a lease on a listed worker asked for the build"
    );
    assert_eq!(leases.issue(booking(gateway, 2, owed, now)), Ok(()));
    assert_eq!(
        (store.membership_builds(), store.publish_due().is_some()),
        (builds, true),
        "a lease on a worker inside its window asked for the build"
    );
    for id in [1, 2] {
        assert!(leases.released(gateway, LeaseId(id), None, ReleaseReason::ClientGone));
    }
}

/// A first dispatch that moves a lease onto a worker inside its
/// registration window books that worker's count and asks for no build:
/// its host's count reads the lease inside the window and behind the
/// window's build, and the lease ends on the count it booked.
#[test]
fn a_lease_dispatched_onto_a_worker_inside_its_window_books_its_count() {
    let store = MemoryStore::new();
    let built = ready(&store, "http://a:1");
    let owed = store.upsert(facts("http://b:1", 1)).id();
    let revision = WorkersStore::get(&store, owed).expect("the row").revision;
    store
        .set_status(owed, WorkerStatus::Ready, revision)
        .expect("ready");
    let builds = store.membership_builds();
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let mut targets = Targets::new();
    for id in [built, owed] {
        targets.push(Target {
            id,
            ..Target::default()
        });
    }
    let record = LeaseRecord {
        picked: PickRecord::of(&targets),
        ..booking(gateway, 1, built, now)
    };
    assert_eq!(leases.issue(record), Ok(()));
    let on = Dispatch {
        target_index: 1,
        ..Dispatch::default()
    };
    assert!(leases.dispatched(gateway, LeaseId(1), on, now));
    assert_eq!(
        (store.membership_builds(), store.publish_due().is_some()),
        (builds, true),
        "the dispatch asked for the build"
    );
    let of = |count| {
        let mut rows = [(built, 0), (owed, count)];
        rows.sort_unstable();
        rows
    };
    assert_eq!(in_flight(&store), of(1), "inside the window");
    store.publish_now();
    assert_eq!(in_flight(&store), of(1));
    assert!(leases.released(gateway, LeaseId(1), None, ReleaseReason::ClientGone));
    assert_eq!(in_flight(&store), of(0));
}

/// A member that leaves inside its registration window takes its slot
/// with it: the lease filed on it ends on the count it booked, and the
/// member that joins next is given another.
#[test]
fn a_member_that_leaves_inside_its_window_takes_its_slot_with_it() {
    let store = MemoryStore::new();
    let built = ready(&store, "http://a:1");
    let left = store.upsert(facts("http://b:1", 1)).id();
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    assert_eq!(leases.issue(booking(gateway, 1, left, now)), Ok(()));
    let revision = WorkersStore::get(&store, left).expect("the row").revision;
    assert!(store.remove(left, revision).is_some());
    let joined = ready(&store, "http://c:1");
    let mut rows = [(built, 0), (joined, 0)];
    rows.sort_unstable();
    assert_eq!(in_flight(&store), rows, "the lease of the one that left");
    assert!(leases.released(gateway, LeaseId(1), None, ReleaseReason::WorkerRemoved));
    assert_eq!(in_flight(&store), rows);
}

/// The table holds a member from the write that files it to the build
/// that lists it: behind a membership build it holds none, however many
/// joined inside the window.
#[test]
fn a_membership_build_leaves_no_member_in_the_table() {
    let store = MemoryStore::new();
    let first = ready(&store, "http://a:1");
    assert_eq!(store.filed_members(), 0, "the first build listed it");
    let second = store.upsert(facts("http://b:1", 1)).id();
    let third = store.upsert(facts("http://c:1", 1)).id();
    assert!(first != second && second != third);
    assert_eq!(store.filed_members(), 2, "two joined inside the window");
    store.publish_now();
    assert_eq!(store.filed_members(), 0, "the build listed both");
}

/// A generation of the standing pairing mode builds no membership: the
/// table stands behind its publication, and a lease on a member inside
/// its window books its count.
#[cfg(feature = "controller")]
#[test]
fn a_generation_that_builds_no_membership_leaves_a_joiners_slot_on_file() {
    let store = MemoryStore::new();
    store.apply(manual_ruled(1, 60, 30)).expect("generation 1");
    let built = ready(&store, "http://a:1");
    let owed = store.upsert(facts("http://b:1", 1)).id();
    let revision = WorkersStore::get(&store, owed).expect("the row").revision;
    store
        .set_status(owed, WorkerStatus::Ready, revision)
        .expect("ready");
    store.apply(manual_ruled(2, 60, 30)).expect("generation 2");
    assert_eq!(
        (store.publish_due().is_some(), store.filed_members()),
        (true, 1),
        "the generation built no membership"
    );
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    assert_eq!(leases.issue(booking(gateway, 1, owed, now)), Ok(()));
    store.publish_now();
    let mut rows = [(built, 0), (owed, 1)];
    rows.sort_unstable();
    assert_eq!(in_flight(&store), rows);
}

/// A member filed with no model stands in no pool, and the world of the
/// build behind its write names its slot all the same: a lease on it
/// inside the window of the refresh that gives it a model books its
/// count.
#[test]
fn a_member_given_its_first_model_books_a_lease_inside_that_window() {
    let store = MemoryStore::new();
    let built = ready(&store, "http://a:1");
    let bare = WorkerFacts {
        models: Vec::new(),
        ..facts("http://b:1", 1)
    };
    let owed = store.upsert(bare).id();
    store.publish_now();
    assert_eq!(store.filed_members(), 0, "the build named it, in no pool");
    // The same incarnation with a model: the row and its slot stay.
    assert_eq!(store.upsert(facts("http://b:1", 1)).id(), owed);
    let revision = WorkersStore::get(&store, owed).expect("the row").revision;
    store
        .set_status(owed, WorkerStatus::Ready, revision)
        .expect("ready");
    assert!(store.publish_due().is_some(), "the refresh owes its build");
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    assert_eq!(leases.issue(booking(gateway, 1, owed, now)), Ok(()));
    store.publish_now();
    let mut rows = [(built, 0), (owed, 1)];
    rows.sort_unstable();
    assert_eq!(in_flight(&store), rows);
}

/// A lease on a member that offers no model books its count all the same:
/// its host's count reads the lease while the member stands in no pool,
/// and through the write that gives it its first model.
#[test]
fn a_lease_on_a_member_in_no_pool_books_its_count() {
    let store = MemoryStore::new();
    let built = ready(&store, "http://a:1");
    let bare = WorkerFacts {
        models: Vec::new(),
        ..facts("http://b:1", 1)
    };
    let owed = store.upsert(bare).id();
    store.publish_now();
    assert_eq!(store.filed_members(), 0, "the build named it, in no pool");
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    assert_eq!(leases.issue(booking(gateway, 1, owed, now)), Ok(()));
    assert_eq!(in_flight(&store), [(built, 0)], "it serves no model");
    assert_eq!(booked_on(&store, owed), Some(1), "booked, in no pool");
    assert_eq!(store.upsert(facts("http://b:1", 1)).id(), owed);
    let revision = WorkersStore::get(&store, owed).expect("the row").revision;
    store
        .set_status(owed, WorkerStatus::Ready, revision)
        .expect("ready");
    store.publish_now();
    let mut rows = [(built, 0), (owed, 1)];
    rows.sort_unstable();
    assert_eq!(in_flight(&store), rows);
}

/// A world names the slot of a member in no pool from the build behind
/// its write to the build behind its leave, a slot of its own, and of
/// no worker beside the members its build read.
#[test]
fn a_world_names_a_member_in_no_pool_until_it_leaves() {
    let store = MemoryStore::new();
    let pooled = ready(&store, "http://a:1");
    let bare = WorkerFacts {
        models: Vec::new(),
        ..facts("http://b:1", 1)
    };
    let owed = store.upsert(bare).id();
    let listed = store.snapshot().booked_of(pooled);
    assert!(listed.is_some());
    assert!(
        store.snapshot().booked_of(owed).is_none(),
        "inside its window"
    );
    store.publish_now();
    let named = store.snapshot().booked_of(owed);
    assert!(named.is_some(), "named, in no pool");
    assert_ne!(named, listed, "a slot of its own");
    assert_eq!(store.snapshot().booked_of(pooled), listed);
    let revision = WorkersStore::get(&store, owed).expect("the row").revision;
    assert!(store.remove(owed, revision).is_some());
    assert!(store.snapshot().booked_of(owed).is_none(), "it left");
    assert_eq!(store.snapshot().booked_of(pooled), listed);
}

/// A generation of the standing pairing mode builds no membership: the
/// world it publishes still names the slot of a member in no pool.
#[cfg(feature = "controller")]
#[test]
fn a_generation_that_builds_no_membership_still_names_a_member_in_no_pool() {
    let store = MemoryStore::new();
    store.apply(manual_ruled(1, 60, 30)).expect("generation 1");
    ready(&store, "http://a:1");
    let bare = WorkerFacts {
        models: Vec::new(),
        ..facts("http://b:1", 1)
    };
    let owed = store.upsert(bare).id();
    store.publish_now();
    let named = store.snapshot().booked_of(owed);
    assert!(named.is_some());
    store.apply(manual_ruled(2, 60, 30)).expect("generation 2");
    assert_eq!(
        store.snapshot().booked_of(owed),
        named,
        "the generation built no membership"
    );
}

/// A standby follows the refresh as its leader wrote it: its restored
/// world names the slot of the member given its first model, the
/// followed refresh files nothing, and a lease on the member before the
/// standby's build books its count.
#[test]
fn a_lease_books_on_a_standby_behind_a_followed_refresh() {
    let leader = MemoryStore::new();
    let bare = WorkerFacts {
        models: Vec::new(),
        ..facts("http://b:1", 1)
    };
    let owed = leader.upsert(bare).id();
    let revision = WorkersStore::get(&leader, owed).expect("the row").revision;
    leader
        .set_status(owed, WorkerStatus::Ready, revision)
        .expect("ready");
    leader.publish_now();
    let standby = MemoryStore::new();
    assert!(standby.restore_from(leader.checkpoint()));
    assert_eq!(standby.filed_members(), 0, "the restore built its world");
    // The leader gives the member its first model and the standby follows.
    let unjournaled = Position::default();
    let refresh = leader.upsert(facts("http://b:1", 1));
    let row = WorkersStore::get(&leader, owed);
    standby.follow(unjournaled, WorkersStore::epoch(&leader), refresh, row);
    assert_eq!(
        (standby.publish_due().is_some(), standby.filed_members()),
        (true, 0),
        "the followed refresh owes its build and files nothing"
    );
    let leases: &dyn LeasesStore = &standby;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    assert_eq!(leases.issue(booking(gateway, 1, owed, now)), Ok(()));
    standby.publish_now();
    assert_eq!(in_flight(&standby), [(owed, 1)]);
}

/// A member back under its key and incarnation is one that joined: it is
/// given the table's next slot, and the leases on file for the one that
/// left end on the count they booked.
#[test]
fn a_member_back_under_its_id_is_given_the_next_slot() {
    let store = MemoryStore::new();
    let (stays, leaves) = (ready(&store, "http://a:1"), ready(&store, "http://b:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    for id in 1..=2 {
        assert_eq!(leases.issue(booking(gateway, id, leaves, now)), Ok(()));
    }
    let row = WorkersStore::get(&store, leaves).expect("the row");
    assert!(store.remove(leaves, row.revision).is_some());
    // The same key and incarnation after the leave: the same id, and the
    // two leases of the registration that left still on file.
    let back = ready(&store, "http://b:1");
    assert_eq!(back, leaves);
    let mut stand = [(stays, 0), (back, 0)];
    stand.sort_unstable();
    assert_eq!(in_flight(&store), stand);
    for id in 1..=2 {
        assert!(leases.released(gateway, LeaseId(id), None, ReleaseReason::WorkerRemoved));
    }
    assert_eq!(in_flight(&store), stand);
}

/// A record filed over one on file under its key gives the first's count
/// back: a restore whose page names one lease twice keeps one lease and
/// one count.
#[test]
fn a_record_filed_over_one_on_file_gives_the_firsts_count_back() {
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let page = booking(gateway, 7, a, now);
    assert_eq!(leases.restore(gateway, &[page, page]), 1);
    assert_eq!(in_flight(&store), [(a, 1)]);
}

/// A sweep's one write ends the leases of every gateway that goes: each
/// gives its count back, and the leases of a gateway that stays still book
/// their workers.
#[test]
fn a_sweeps_one_write_gives_back_the_counts_of_every_gateway_that_goes() {
    let store = MemoryStore::new();
    let (a, b) = (ready(&store, "http://a:1"), ready(&store, "http://b:1"));
    let leases: &dyn LeasesStore = &store;
    let now = Instant::now();
    leases.set_term(1);
    // Gateways 1 and 3 go, gateway 2 stays with its two leases.
    for (gateway, id, target) in [(1, 1, a), (1, 2, b), (2, 3, a), (2, 4, a), (3, 5, b)] {
        let filed = leases.issue(booking(GatewayId(gateway), id, target, now));
        filed.expect("the key is free");
    }
    let of = |a_holds, b_holds| {
        let mut rows = [(a, a_holds), (b, b_holds)];
        rows.sort_unstable();
        rows
    };
    assert_eq!(in_flight(&store), of(3, 2));
    assert_eq!(leases.drop_gateways(&[GatewayId(3), GatewayId(1)]), [1, 2]);
    assert_eq!(in_flight(&store), of(2, 0));
    assert_eq!(leases.live(GatewayId(2)), 2);
}

#[test]
fn a_dispatch_on_another_target_of_the_pick_moves_the_leases_count() {
    let store = MemoryStore::new();
    let (a, b) = (ready(&store, "http://a:1"), ready(&store, "http://b:1"));
    let (gateway, now) = (GatewayId(1), Instant::now());
    store.set_term(1);
    let of = |a_holds, b_holds| {
        let mut rows = [(a, a_holds), (b, b_holds)];
        rows.sort_unstable();
        rows
    };
    // A pick of a, then b: the lease books a.
    let mut targets = Targets::new();
    for id in [a, b] {
        targets.push(Target {
            id,
            ..Target::default()
        });
    }
    let picked = |id: u64| LeaseRecord {
        picked: PickRecord::of(&targets),
        ..booking(gateway, id, a, now)
    };
    let leases: &dyn LeasesStore = &store;
    leases.issue(picked(1)).expect("filed");
    assert_eq!(in_flight(&store), of(1, 0));
    // The gateway dispatched to the pick's second target: the count is b's.
    let on = |target_index| Dispatch {
        target_index,
        ..Dispatch::default()
    };
    assert!(leases.dispatched(gateway, LeaseId(1), on(1), now));
    assert_eq!(in_flight(&store), of(0, 1));
    let filed = leases.get(gateway, LeaseId(1)).expect("on file");
    assert_eq!((filed.target, filed.lease.target_index), (b, 1));
    // A second dispatch of one lease, on the first target, moves nothing.
    assert!(leases.dispatched(gateway, LeaseId(1), on(0), now));
    assert_eq!(in_flight(&store), of(0, 1));
    // The lease ends on the count it books now.
    assert!(leases.released(gateway, LeaseId(1), None, ReleaseReason::ClientGone));
    assert_eq!(in_flight(&store), of(0, 0));
    // An index the pick holds no target at moves the state, and neither a
    // count nor the record's index.
    leases.issue(picked(2)).expect("filed");
    assert!(leases.dispatched(gateway, LeaseId(2), on(2), now));
    assert_eq!(in_flight(&store), of(1, 0));
    let filed = leases.get(gateway, LeaseId(2)).expect("on file");
    assert_eq!(
        (filed.state, filed.lease.target_index),
        (LeaseState::Dispatched { at: now }, 0)
    );
}

/// A lease moves at its first dispatch alone: a second one that names a
/// third target of the pick, at a later instant, leaves the instant, the
/// target and the counts as the first set them.
#[test]
fn a_second_dispatch_on_a_third_target_of_the_pick_moves_nothing() {
    let store = MemoryStore::new();
    let workers = ["http://a:1", "http://b:1", "http://c:1"].map(|key| ready(&store, key));
    let [a, b, c] = workers;
    let (gateway, now) = (GatewayId(1), Instant::now());
    store.set_term(1);
    let of = |held: [u32; 3]| {
        let mut rows = [(a, held[0]), (b, held[1]), (c, held[2])];
        rows.sort_unstable();
        rows
    };
    let mut targets = Targets::new();
    for id in workers {
        targets.push(Target {
            id,
            ..Target::default()
        });
    }
    let record = LeaseRecord {
        picked: PickRecord::of(&targets),
        ..booking(gateway, 1, a, now)
    };
    let leases: &dyn LeasesStore = &store;
    leases.issue(record).expect("filed");
    let on = |target_index| Dispatch {
        target_index,
        ..Dispatch::default()
    };
    assert!(leases.dispatched(gateway, LeaseId(1), on(1), now));
    assert_eq!(in_flight(&store), of([0, 1, 0]));
    let later = now + Duration::from_secs(1);
    assert!(leases.dispatched(gateway, LeaseId(1), on(2), later));
    assert_eq!(
        in_flight(&store),
        of([0, 1, 0]),
        "the second target's count"
    );
    let filed = leases.get(gateway, LeaseId(1)).expect("on file");
    assert_eq!(
        (filed.target, filed.lease.target_index, filed.state),
        (b, 1, LeaseState::Dispatched { at: now })
    );
}

/// One worker at two ranks of a pick: a dispatch on the second names the
/// lease's own worker, so no count moves, and the record's index follows
/// the dispatch all the same.
#[test]
fn a_dispatch_on_another_rank_of_the_leases_worker_moves_the_index_alone() {
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    let (gateway, now) = (GatewayId(1), Instant::now());
    store.set_term(1);
    let mut targets = Targets::new();
    for dp_rank in [0, 3] {
        targets.push(Target {
            id: a,
            dp_rank,
            ..Target::default()
        });
    }
    let record = LeaseRecord {
        picked: PickRecord::of(&targets),
        ..booking(gateway, 1, a, now)
    };
    let leases: &dyn LeasesStore = &store;
    leases.issue(record).expect("filed");
    let on = Dispatch {
        target_index: 1,
        ..Dispatch::default()
    };
    assert!(leases.dispatched(gateway, LeaseId(1), on, now));
    assert_eq!(in_flight(&store), [(a, 1)], "the worker's count stands");
    let filed = leases.get(gateway, LeaseId(1)).expect("on file");
    assert_eq!((filed.target, filed.lease.target_index), (a, 1));
    assert_eq!(filed.target_at(1), Some((a, 3)), "the rank dispatched to");
}

/// The file's records and its places of decode sides stand and fall
/// together: at every check of a walk over the leases' doors a decode
/// side is on file for each record that carries the mark of a pair's
/// lease and for no other, and both agree with the walk's own plain map
/// of what it filed.
#[test]
fn a_decode_side_is_on_file_for_each_record_that_carries_the_mark_and_for_no_other() {
    let store = MemoryStore::new();
    let [p, d0, d1] = ["p", "d0", "d1"].map(|name| ready(&store, &format!("http://{name}:1")));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    let gone = ReleaseReason::ClientGone;
    leases.set_term(1);
    // The walk's own word: each id it filed, and whether a pair's lease
    // stands under it; an id that left the file is not a key.
    let mut filed = std::collections::BTreeMap::<u64, bool>::new();
    let mut seen = Vec::new();
    let mut read = |door: &str, filed: &std::collections::BTreeMap<u64, bool>| {
        let [sides, marks] = store.paired();
        assert_eq!(sides, marks, "{door}: a decode side for each mark");
        let pairs = filed.iter().filter(|(_, pair)| **pair);
        let pairs: Vec<_> = pairs.map(|(id, _)| (gateway, *id)).collect();
        assert_eq!(sides, pairs, "{door}: the pairs on file");
        seen.extend(filed.keys().copied());
        for id in &seen {
            let found = leases.sides(gateway, LeaseId(*id));
            let pair = found.map(|found| found.decode.is_some());
            assert_eq!(pair, filed.get(id).copied(), "{door}: lease {id}");
        }
    };
    // Filed by each door that files: a single's, a pair's, and a record
    // alone that carries a mark, which is a single's.
    leases.issue(booking(gateway, 1, p, now)).expect("free");
    issue_pair(leases, pair(2, p, &[d0], now)).expect("free");
    issue_pair(leases, pair(3, p, &[d0, d1], now)).expect("free");
    let alone = LeaseRecord {
        decode: Some(DecodeMark),
        ..booking(gateway, 4, p, now)
    };
    leases.issue(alone).expect("free");
    filed.extend([(1, false), (2, true), (3, true), (4, false)]);
    read("an issue", &filed);
    assert_eq!(
        leases.get(gateway, LeaseId(4)).expect("on file").decode,
        None
    );
    assert_eq!(
        leases.get(gateway, LeaseId(2)).expect("on file").decode,
        Some(DecodeMark)
    );
    // A side's dispatch and a side's end leave both tables as they stand;
    // the last side's end takes both.
    assert!(leases.dispatched(gateway, LeaseId(2), on(Side::Prefill, 0), now));
    assert!(leases.dispatched(gateway, LeaseId(3), on(Side::Decode, 1), now));
    read("a side's dispatch", &filed);
    assert!(leases.completed(gateway, LeaseId(2), &done(Side::Prefill)));
    read("a side's completion", &filed);
    assert!(leases.completed(gateway, LeaseId(2), &done(Side::Decode)));
    filed.remove(&2);
    read("the last side's completion", &filed);
    assert!(leases.released(gateway, LeaseId(3), Some(Side::Decode), gone));
    read("a side's release", &filed);
    assert!(leases.released(gateway, LeaseId(3), Some(Side::Prefill), gone));
    filed.remove(&3);
    read("the last side's release", &filed);
    // A decision's issue of a single, under an id the store mints, beside
    // a pair; a release of the pair whole.
    let slot = store.snapshot().booked_of(p);
    let single = store.issue_decided(booking(gateway, 0, p, now), slot, |_| ());
    let single = single.expect("minted under the store's term").id;
    issue_pair(leases, pair(10, p, &[d0], now)).expect("free");
    filed.extend([(single.0, false), (10, true)]);
    read("a decision's issue", &filed);
    assert!(leases.released(gateway, LeaseId(10), None, gone));
    filed.remove(&10);
    read("a pair's release whole", &filed);
    // The dispatch deadline: a pair that waits whole leaves, and one with
    // a dispatched decode stands with its prefill side ended.
    issue_pair(leases, pair(5, p, &[d0], now)).expect("free");
    issue_pair(leases, pair(6, p, &[d0], now)).expect("free");
    assert!(leases.dispatched(gateway, LeaseId(6), on(Side::Decode, 0), now));
    filed.extend([(5, true), (6, true)]);
    read("two pairs more", &filed);
    leases.expire(now + Duration::from_secs(2));
    filed.retain(|id, _| *id == 6);
    read("the dispatch deadline", &filed);
    // A frame applied through a hand, by whoever holds the turn next.
    let hand = Hand::new(gateway);
    store.hand(&hand, LeaseId(6), Said::Completed(done(Side::Decode)));
    store.hand_back(hand);
    filed.remove(&6);
    read("a handed frame", &filed);
    // A restore over a key on file: its pages list records alone, so the
    // pair under that key leaves whole and a single's lease stands there.
    issue_pair(leases, pair(7, p, &[d0], now)).expect("free");
    issue_pair(leases, pair(8, p, &[d0], now)).expect("free");
    filed.extend([(7, true), (8, true)]);
    read("two pairs for a restore", &filed);
    assert_eq!(leases.restore(gateway, &[pair(7, p, &[d0], now).0]), 1);
    filed = [(7, false)].into();
    read("a restore", &filed);
    // The gateway's end takes every lease, a pair's two parts together.
    issue_pair(leases, pair(9, p, &[d1], now)).expect("free");
    filed.insert(9, true);
    read("a pair beside a restored lease", &filed);
    assert_eq!(leases.drop_gateways(&[gateway]), [2]);
    filed.clear();
    read("the gateway's end", &filed);
    assert_eq!(booked(&store, &[p, d0, d1]), [0, 0, 0]);
}

#[test]
fn every_door_that_takes_a_lease_off_the_file_gives_its_count_back() {
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let ok = Outcome {
        side: Side::Single,
        served: a,
        status: 200,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    };
    let key = LeaseId(1);
    let doors: [(&str, &dyn Fn()); 5] = [
        ("a completion", &|| {
            assert!(leases.completed(gateway, key, &ok));
        }),
        ("a release", &|| {
            assert!(leases.released(gateway, key, None, ReleaseReason::ClientGone));
        }),
        ("the dispatch deadline", &|| {
            assert_eq!(leases.expire(now + Duration::from_secs(2)).len(), 1);
        }),
        ("a restore that omits it", &|| {
            assert_eq!(leases.restore(gateway, &[]), 0);
        }),
        ("its gateway's end", &|| {
            assert_eq!(leases.drop_gateways(&[gateway]), [1]);
        }),
    ];
    for (door, through) in doors {
        leases
            .issue(booking(gateway, 1, a, now))
            .expect("the key is free");
        assert_eq!(in_flight(&store), [(a, 1)], "{door}: the lease books");
        through();
        assert_eq!(in_flight(&store), [(a, 0)], "{door}: the count is back");
        assert_eq!(leases.get(gateway, key), None, "{door}");
    }
    // A restore lays the pages' leases on file, each on its worker's count,
    // and a lease written again under its key books once.
    let pages = [booking(gateway, 1, a, now), booking(gateway, 2, a, now)];
    assert_eq!(leases.restore(gateway, &pages), 2);
    assert_eq!(in_flight(&store), [(a, 2)]);
    assert_eq!(leases.restore(gateway, &pages[..1]), 1);
    assert_eq!(in_flight(&store), [(a, 1)]);
}
