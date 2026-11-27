//! A key's session pins across its worker's lives: what a new life or a
//! return takes of them, what stands through an absence and when it ends,
//! and what a follower and a snapshot do with them.

use super::*;

/// Pins `key` to `worker` in the session table of `store`'s standing
/// engine, as a keyed dispatch does.
fn pinned(store: &MemoryStore, key: u64, worker: WorkerId) {
    let world = store.snapshot();
    let table = world.spec().engine().sessions();
    table.dispatched(Session::from_raw(key), SessionVerdict::Pinned, worker, 0);
}

/// The worker `key`'s first pin names in that table.
fn pin_of(store: &MemoryStore, key: u64) -> Option<WorkerId> {
    let world = store.snapshot();
    let table = world.spec().engine().sessions();
    table.pins(Session::from_raw(key)).map(|pins| pins[0])
}

/// How many keys that table holds.
fn pins_held(store: &MemoryStore) -> usize {
    store.snapshot().spec().engine().sessions().len()
}

/// A removed worker's session pins stand through its absence, and the
/// worker it was takes them back where its key is registered again, under
/// the id the registration makes. A new life of the same worker in place
/// keeps its own. Another worker under the key takes none.
#[test]
fn a_key_registered_again_as_the_worker_it_was_takes_its_pins() {
    let store = MemoryStore::new();
    let (a, b) = (
        ready(&store, "http://a:8000"),
        ready(&store, "http://b:8000"),
    );
    pinned(&store, 11, a);
    pinned(&store, 12, b);
    pinned(&store, 13, a);
    let of = |key: u64| pin_of(&store, key);
    let revision = WorkersStore::get(&store, a).expect("the row").revision;
    store.remove(a, revision).expect("removed");
    // The removal drops none: the key may be registered again.
    assert_eq!((of(11), of(13)), (Some(a), Some(a)));
    let again = store.upsert(facts("http://a:8000", 2)).id();
    assert_ne!(again, a);
    assert_eq!((of(11), of(13)), (Some(again), Some(again)));
    assert_eq!(of(12), Some(b));
    let life = store.upsert(facts("http://b:8000", 2)).id();
    assert_ne!(life, b);
    assert_eq!(of(12), Some(life), "a new life of the same worker");
    // Removed again, then registered in another role: another worker.
    let revision = WorkersStore::get(&store, again).expect("the row").revision;
    store.remove(again, revision).expect("removed");
    store.upsert(WorkerFacts {
        role: Role::Decode,
        ..facts("http://a:8000", 3)
    });
    assert_eq!((of(11), of(13)), (None, None));
    assert_eq!(pins_held(&store), 1);
    // A new life in another role, in place: another worker too.
    store.upsert(WorkerFacts {
        role: Role::Prefill,
        ..facts("http://b:8000", 3)
    });
    assert_eq!(pins_held(&store), 0, "the old worker's pin");
}

/// What [`AtTheRelease`] runs for a case.
type Hook = Box<dyn FnOnce()>;

thread_local! {
    /// The hook a case armed on this thread for the journal's next
    /// release.
    static AT_RELEASE: RefCell<Option<Hook>> = const { RefCell::new(None) };
}

/// A journal that takes every write. Where the store has let a write's
/// locks go it runs the hook a case armed on this thread, once: the hook
/// is taken out first, so a write the hook itself makes meets none.
struct AtTheRelease;

impl Journal for AtTheRelease {
    fn member(
        &self,
        _: Option<Position>,
        _: Option<u64>,
        _: u64,
        _: MembershipChange,
        _: Option<&WorkerRow>,
        _: Option<&WorkerRow>,
    ) -> bool {
        true
    }

    fn status(
        &self,
        _: Option<Position>,
        _: Option<u64>,
        _: u64,
        _: WorkerId,
        _: WorkerStatus,
        _: WorkerStatus,
        _: Arc<[String]>,
    ) -> bool {
        true
    }

    fn term_rose(&self, _: u64) {}

    fn generation(&self, _: Option<Position>, _: u64) {}

    fn term_followed(&self, _: Position) {}

    fn head(&self) -> Position {
        Position::default()
    }

    fn followed(&self) -> Option<u64> {
        None
    }

    fn released(&self) {
        let hook = AT_RELEASE.with(|held| held.borrow_mut().take());
        if let Some(hook) = hook {
            hook();
        }
    }

    fn restart(&self, _: Position) -> bool {
        true
    }
}

/// Two registrations of one key rename its session pins in their order.
/// The second lands where the first has let the workers lock go, and the
/// pin names the last life: in place, and for a key that left.
#[test]
fn registrations_of_one_key_rename_its_pins_in_their_order() {
    use std::rc::Rc;

    let store = Rc::new(MemoryStore::journaled(Arc::new(AtTheRelease)));
    let of = |key: u64| pin_of(&store, key);
    // The third life of `key`, registered behind the next registration:
    // the journal hears there that the write's locks are let go.
    let third_behind = |key: &'static str| {
        let third = Rc::new(Cell::new(None));
        let (behind, seen) = (Rc::clone(&store), Rc::clone(&third));
        let register = move || seen.set(Some(behind.upsert(facts(key, 3)).id()));
        AT_RELEASE.set(Some(Box::new(register)));
        third
    };
    // In place: the key's second life is replaced by its third at once.
    let a = ready(&store, "http://a:8000");
    pinned(&store, 11, a);
    let third = third_behind("http://a:8000");
    store.upsert(facts("http://a:8000", 2));
    assert!(third.get().is_some(), "the hook ran");
    assert_eq!(of(11), third.get());
    // A key that left: registered again, and that life replaced at once.
    let b = ready(&store, "http://b:8000");
    pinned(&store, 12, b);
    let revision = WorkersStore::get(&*store, b).expect("the row").revision;
    store.remove(b, revision).expect("removed");
    let third = third_behind("http://b:8000");
    store.upsert(facts("http://b:8000", 2));
    assert!(third.get().is_some(), "the hook ran");
    assert_eq!(of(12), third.get());
}

/// A key that comes back as another worker takes no pin, and the pins of
/// the worker it was are forgotten with the workers lock let go: where the
/// journal hears of the release, the old pin still stands.
#[test]
fn a_returned_keys_old_pins_are_forgotten_behind_the_workers_lock() {
    use std::rc::Rc;

    let store = Rc::new(MemoryStore::journaled(Arc::new(AtTheRelease)));
    let a = ready(&store, "http://a:8000");
    pinned(&store, 11, a);
    let revision = WorkersStore::get(&*store, a).expect("the row").revision;
    store.remove(a, revision).expect("removed");
    let at_release = Rc::new(Cell::new(None));
    let (behind, seen) = (Rc::clone(&store), Rc::clone(&at_release));
    AT_RELEASE.set(Some(Box::new(move || seen.set(Some(pin_of(&behind, 11))))));
    store.upsert(WorkerFacts {
        role: Role::Decode,
        ..facts("http://a:8000", 2)
    });
    assert_eq!(at_release.get(), Some(Some(a)), "forgotten under the lock");
    assert_eq!(pin_of(&store, 11), None, "another worker takes no pin");
}

/// A key that stays away longer than the absence's time takes no pin
/// back: the time is the longest TTL among the document's session rules.
#[test]
fn a_key_registered_again_past_the_absences_time_takes_no_pin() {
    let store = MemoryStore::new();
    let ruled = manual_ruled(1, 1, 1);
    store.apply(ruled).expect("above the unbound start");
    let a = ready(&store, "http://a:8000");
    pinned(&store, 11, a);
    let revision = WorkersStore::get(&store, a).expect("the row").revision;
    store.remove(a, revision).expect("removed");
    assert_eq!(pin_of(&store, 11), Some(a), "inside its time");
    std::thread::sleep(Duration::from_millis(1_100));
    store.upsert(facts("http://a:8000", 2));
    assert_eq!(pin_of(&store, 11), None);
}

/// A key that stays away is forgotten at the store's tick, in the engine's
/// table: no membership write and no decision has to follow.
#[test]
fn the_tick_ends_the_absence_of_a_key_that_stays_away() {
    let store = MemoryStore::new();
    let ruled = manual_ruled(1, 1, 1);
    store.apply(ruled).expect("above the unbound start");
    let a = ready(&store, "http://a:8000");
    pinned(&store, 21, a);
    let revision = WorkersStore::get(&store, a).expect("the row").revision;
    store.remove(a, revision).expect("removed");
    store.end_gone(Instant::now());
    assert_eq!(pin_of(&store, 21), Some(a), "inside its time");
    store.end_gone(Instant::now() + Duration::from_millis(1_100));
    assert_eq!(pin_of(&store, 21), None);
}

/// A follower takes its leader's steps for a key's session pins, in its
/// engine's table. A followed removal drops none. The worker it was takes
/// them back at its followed registration, under the id that registration
/// made. A followed new life of the same worker keeps its own. Another
/// worker under the key takes none.
#[test]
fn a_followed_key_registered_again_as_the_worker_it_was_takes_its_pins() {
    let (leader, follower) = (MemoryStore::new(), MemoryStore::new());
    let join = |facts: WorkerFacts| {
        let change = leader.upsert(facts);
        followed(&leader, &follower, change);
        change.id()
    };
    let leave = |id: WorkerId| {
        let revision = WorkersStore::get(&leader, id).expect("the row").revision;
        let change = leader.remove(id, revision).expect("removed");
        followed(&leader, &follower, change);
    };
    let (a, b) = (
        join(facts("http://a:8000", 1)),
        join(facts("http://b:8000", 1)),
    );
    pinned(&follower, 11, a);
    pinned(&follower, 12, b);
    let of = |key: u64| pin_of(&follower, key);
    leave(a);
    assert_eq!(of(11), Some(a), "none is dropped");
    let again = join(facts("http://a:8000", 2));
    assert_ne!(again, a);
    assert_eq!(of(11), Some(again));
    assert_eq!(of(12), Some(b));
    let life = join(facts("http://b:8000", 2));
    assert_ne!(life, b);
    assert_eq!(of(12), Some(life), "a new life of the same worker");
    // Removed again, then registered in another role: another worker.
    leave(again);
    join(WorkerFacts {
        role: Role::Decode,
        ..facts("http://a:8000", 3)
    });
    assert_eq!(of(11), None);
    assert_eq!(pins_held(&follower), 1);
    // A new life in another role, in place: another worker too.
    join(WorkerFacts {
        role: Role::Prefill,
        ..facts("http://b:8000", 3)
    });
    assert_eq!(pins_held(&follower), 0, "the old worker's pin");
}

/// A follower counts a key's absence by its own clock, from the removal
/// it followed. A key registered again past that time takes no pin, at
/// its followed registration and at a snapshot that holds it again.
#[test]
fn a_followed_key_registered_again_past_the_absences_time_takes_no_pin() {
    let leader = MemoryStore::new();
    let followers = [MemoryStore::new(), MemoryStore::new()];
    let joined = leader.upsert(facts("http://a:8000", 1));
    let revision = WorkersStore::get(&leader, joined.id()).expect("the row");
    for follower in &followers {
        follower
            .apply(manual_ruled(1, 1, 1))
            .expect("above the unbound start");
        followed(&leader, follower, joined);
        pinned(follower, 11, joined.id());
    }
    let removed = leader.remove(joined.id(), revision.revision);
    let removed = removed.expect("removed");
    for follower in &followers {
        followed(&leader, follower, removed);
        assert_eq!(pin_of(follower, 11), Some(joined.id()), "inside its time");
    }
    std::thread::sleep(Duration::from_millis(1_100));
    let again = leader.upsert(facts("http://a:8000", 2));
    followed(&leader, &followers[0], again);
    assert!(followers[1].watch_from(leader.checkpoint()));
    for follower in &followers {
        assert_eq!(pin_of(follower, 11), None);
    }
}

/// A follower files the key of a removal it followed. Where it leads
/// next, its own registration of the key takes the pins.
#[test]
fn a_key_a_follower_saw_leave_takes_its_pins_at_the_followers_own_write() {
    let (leader, follower) = (MemoryStore::new(), MemoryStore::new());
    let joined = leader.upsert(facts("http://a:8000", 1));
    followed(&leader, &follower, joined);
    pinned(&follower, 11, joined.id());
    let revision = WorkersStore::get(&leader, joined.id()).expect("the row");
    let removed = leader.remove(joined.id(), revision.revision);
    followed(&leader, &follower, removed.expect("removed"));
    let again = follower.upsert(facts("http://a:8000", 2)).id();
    assert_eq!(pin_of(&follower, 11), Some(again));
}

/// A snapshot's rows move a key's session pins as the writes it stands
/// for would have, in the engine's table, at a gateway's watch and at a
/// standby's restore. A key the cut holds under a new id
/// takes its pins where it is the worker it was. Another worker's pins
/// leave. A key the cut drops keeps its pins on file, and takes them at
/// its followed registration.
#[test]
fn a_snapshot_moves_the_pins_of_the_rows_it_drops() {
    for restore in [false, true] {
        let (leader, follower) = (MemoryStore::new(), MemoryStore::new());
        let lay = || {
            let cut = leader.checkpoint();
            if restore {
                follower.restore_from(cut)
            } else {
                follower.watch_from(cut)
            }
        };
        let keys = [
            "http://a:8000",
            "http://b:8000",
            "http://c:8000",
            "http://d:8000",
        ];
        let [a, b, c, d] = keys.map(|key| leader.upsert(facts(key, 1)).id());
        assert!(lay());
        for (key, worker) in [(11, a), (12, b), (13, c), (14, d)] {
            pinned(&follower, key, worker);
        }
        let of = |key: u64| pin_of(&follower, key);
        // The follower hears none of these writes.
        let leave = |id: WorkerId| {
            let revision = WorkersStore::get(&leader, id).expect("the row").revision;
            leader.remove(id, revision).expect("removed");
        };
        leave(a);
        let again = leader.upsert(facts("http://a:8000", 2)).id();
        leave(b);
        leader.upsert(WorkerFacts {
            role: Role::Decode,
            ..facts("http://b:8000", 2)
        });
        leave(c);
        assert!(lay());
        assert_eq!(of(11), Some(again));
        assert_eq!(of(12), None, "another worker under the key");
        assert_eq!((of(13), of(14)), (Some(c), Some(d)), "none is dropped");
        let back = leader.upsert(facts("http://c:8000", 2));
        followed(&leader, &follower, back);
        assert_eq!(of(13), Some(back.id()), "the dropped key was on file");
    }
}

/// A key on file when a snapshot comes stays there where the cut does not
/// hold it. Where the cut holds it again as the worker it was, it takes
/// its pins.
#[test]
fn a_snapshot_returns_a_key_that_had_left_to_its_pins() {
    let (leader, follower) = (MemoryStore::new(), MemoryStore::new());
    let [a, b] = ["http://a:8000", "http://b:8000"].map(|key| {
        let change = leader.upsert(facts(key, 1));
        followed(&leader, &follower, change);
        change.id()
    });
    pinned(&follower, 11, a);
    pinned(&follower, 12, b);
    let of = |key: u64| pin_of(&follower, key);
    for id in [a, b] {
        let revision = WorkersStore::get(&leader, id).expect("the row").revision;
        let change = leader.remove(id, revision).expect("removed");
        followed(&leader, &follower, change);
    }
    assert!(follower.watch_from(leader.checkpoint()));
    assert_eq!((of(11), of(12)), (Some(a), Some(b)), "both stay on file");
    let again = leader.upsert(facts("http://a:8000", 2)).id();
    assert!(follower.watch_from(leader.checkpoint()));
    assert_eq!((of(11), of(12)), (Some(again), Some(b)));
    let back = leader.upsert(facts("http://b:8000", 2));
    followed(&leader, &follower, back);
    assert_eq!(of(12), Some(back.id()), "still on file behind two cuts");
}

/// The rows a snapshot drops are filed while the list has room. Past it
/// a row's pins leave at once.
#[test]
fn a_snapshot_files_no_more_keys_than_the_list_holds() {
    use super::super::memory::GONE_KEYS;
    let (leader, follower) = (MemoryStore::new(), MemoryStore::new());
    let ids: Vec<WorkerId> = (0..GONE_KEYS + 2)
        .map(|at| leader.upsert(facts(&format!("http://w{at}:8000"), 1)).id())
        .collect();
    assert!(follower.watch_from(leader.checkpoint()));
    for (key, &id) in (1..).zip(&ids) {
        pinned(&follower, key, id);
    }
    assert!(follower.watch_from(MemoryStore::new().checkpoint()));
    assert_eq!(pins_held(&follower), GONE_KEYS);
}
