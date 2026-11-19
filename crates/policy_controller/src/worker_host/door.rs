//! The writes a provider can make through a host's membership door.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::{
    store::{
        membership::{self, Past},
        HealthSink, Laid, Maker, MembershipChange, MemoryStore, WorkerFacts, WorkerRow,
        WorkerStatus, WorkersStore,
    },
    view::WorkerId,
};

/// What a write of the door made of the row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wrote {
    Registered(WorkerId),
    InPlace(WorkerId),
    Another { old: WorkerId, new: WorkerId },
    Unmoved(WorkerId),
    Draining(WorkerId),
    Removed(WorkerId),
}

/// A write the door refused. Nothing was written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    /// Another maker's row, named by its maker.
    NotMaker(String),
    /// A fact past its bound, or the row past its size.
    Past(Past),
    /// No row stands under the URL.
    Absent,
    /// The host does not lead now, with its reason.
    NotLeading(String),
    /// Another writer moved the row under the call.
    Moved,
}

/// The membership door a host hands the steps that write a found worker.
pub trait Membership: Send + Sync + 'static {
    /// Register a worker, or refresh the maker's row. The callback states
    /// facts for `url` during the door's turn and must leave writes to the door.
    fn found(
        &self,
        maker: &Maker,
        url: &str,
        state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
    ) -> Result<Wrote, Refused>;

    /// Mark the maker's row as draining.
    fn drained(&self, maker: &Maker, url: &str) -> Result<Wrote, Refused>;

    /// Remove the maker's row under the registered URL.
    fn gone(&self, maker: &Maker, url: &str) -> Result<Wrote, Refused>;
}

/// What a door owns of a write, beside the rules every door shares.
pub(crate) trait Turn {
    fn store(&self) -> &MemoryStore;

    /// The number of the life a registration makes now.
    fn life(&self) -> Result<u64, Refused>;

    /// A size rule of the door's own, behind the store's measure.
    fn room(&self, facts: &WorkerFacts, maker: &Maker) -> Result<(), Refused>;
}

fn made_by(row: &WorkerRow, maker: &Maker) -> Result<(), Refused> {
    row.made_by(maker)
        .map_err(|not| Refused::NotMaker(not.to_string()))
}

/// Lay a registration over the standing facts while the door holds its turn.
pub(crate) fn lay(
    turn: &impl Turn,
    maker: &Maker,
    url: &str,
    state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
) -> Result<Wrote, Refused> {
    let store = turn.store();
    let standing = store.by_key(url);
    if let Some(row) = &standing {
        made_by(row, maker)?;
    }

    let mut facts = state(standing.as_ref().map(|row| &row.facts));
    debug_assert!(facts.key == url, "the facts name the URL of the call");
    if let Some(row) = &standing {
        facts.incarnation = row.facts.incarnation;
    }
    facts.bounded(maker).map_err(Refused::Past)?;
    turn.room(&facts, maker)?;

    match &standing {
        None => facts.incarnation = turn.life()?,
        Some(row) => match facts.laid_over(&row.facts) {
            Laid::Unmoved => return Ok(Wrote::Unmoved(row.id)),
            Laid::InPlace => {}
            Laid::Another => facts.incarnation = turn.life()?,
        },
    }
    let probes = facts.health.probes;
    let change = store.upsert_by(facts, maker.clone());
    let wrote = match change {
        MembershipChange::Registered { id } => Wrote::Registered(id),
        MembershipChange::Relabelled { id } | MembershipChange::Resized { id } => {
            Wrote::InPlace(id)
        }
        MembershipChange::Replaced { old, new } => Wrote::Another { old, new },
        MembershipChange::Removed { .. } | MembershipChange::StatusChanged { .. } => {
            return Err(Refused::Moved);
        }
    };
    if !probes {
        if let Some(row) = store.get(change.id()) {
            store.connected(row.id, row.revision);
        }
    }
    Ok(wrote)
}

fn changed(
    change: Option<MembershipChange>,
    wrote: fn(WorkerId) -> Wrote,
) -> Result<Wrote, Refused> {
    change
        .map(|change| wrote(change.id()))
        .ok_or(Refused::Moved)
}

/// Mark a maker's row as draining while the door holds its turn.
pub(crate) fn drain(store: &MemoryStore, maker: &Maker, url: &str) -> Result<Wrote, Refused> {
    let row = store.by_key(url).ok_or(Refused::Absent)?;
    made_by(&row, maker)?;
    if row.status == WorkerStatus::Draining {
        return Ok(Wrote::Unmoved(row.id));
    }
    changed(
        store.set_status(row.id, WorkerStatus::Draining, row.revision),
        Wrote::Draining,
    )
}

/// Remove the row under the registered URL while the door holds its turn.
/// The maker is checked whole and the removal uses the revision just read.
/// A provider may treat [`Refused::Absent`] and [`Refused::NotMaker`] as done.
/// Another spelling of the URL names no row. A late or repeated removal
/// leaves workers under another key alone. Reusing a key lets the old
/// removal take the later worker.
/// Control calls remove by id and revision through their own door.
pub(crate) fn take(store: &MemoryStore, maker: &Maker, url: &str) -> Result<Wrote, Refused> {
    let row = store.by_key(url).ok_or(Refused::Absent)?;
    made_by(&row, maker)?;
    changed(store.remove(row.id, row.revision), Wrote::Removed)
}

/// The membership door over a store alone, for a host with no election or
/// log. A host shares one door for its store so its writes share a turn.
pub struct StoreDoor {
    store: Arc<MemoryStore>,
    base: u64,
    turn: Mutex<()>,
    told: Option<Box<dyn Fn() + Send + Sync>>,
}

impl StoreDoor {
    /// The one term such a host writes under.
    pub const TERM: u64 = 0;

    /// Open a door over the store, with a base for this run's lives.
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self {
            store,
            base: membership::base(),
            turn: Mutex::new(()),
            told: None,
        }
    }

    /// Call `told` once per write, after the turn is released. A refusal
    /// or an unmoved row leaves the listener alone.
    #[must_use]
    pub fn telling(mut self, told: impl Fn() + Send + Sync + 'static) -> Self {
        self.told = Some(Box::new(told));
        self
    }

    fn call(&self, write: impl FnOnce() -> Result<Wrote, Refused>) -> Result<Wrote, Refused> {
        let turn = self.turn.lock();
        let answer = write();
        drop(turn);
        if !matches!(answer, Err(_) | Ok(Wrote::Unmoved(_))) {
            if let Some(told) = &self.told {
                told();
            }
        }
        answer
    }
}

impl Turn for StoreDoor {
    fn store(&self) -> &MemoryStore {
        &self.store
    }

    fn life(&self) -> Result<u64, Refused> {
        Ok(membership::life(Self::TERM, self.base, self.store.epoch()))
    }

    fn room(&self, _: &WorkerFacts, _: &Maker) -> Result<(), Refused> {
        Ok(())
    }
}

impl Membership for StoreDoor {
    fn found(
        &self,
        maker: &Maker,
        url: &str,
        state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
    ) -> Result<Wrote, Refused> {
        self.call(|| lay(self, maker, url, state))
    }

    fn drained(&self, maker: &Maker, url: &str) -> Result<Wrote, Refused> {
        self.call(|| drain(&self.store, maker, url))
    }

    fn gone(&self, maker: &Maker, url: &str) -> Result<Wrote, Refused> {
        self.call(|| take(&self.store, maker, url))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc, Arc, Barrier, Weak,
        },
        thread,
        time::Duration,
    };

    use super::{changed, lay, Membership, Refused, StoreDoor, Turn, Wrote};
    use crate::{
        store::{
            membership::{self, Past},
            Maker, MemoryStore, WorkerFacts, WorkerRow, WorkerStatus, WorkersStore,
        },
        view::{Role, WireModes},
    };

    const URL: &str = "http://worker:8000";

    fn door() -> (Arc<MemoryStore>, StoreDoor) {
        let store = Arc::new(MemoryStore::default());
        let door = StoreDoor::new(Arc::clone(&store));
        (store, door)
    }

    fn stated() -> WorkerFacts {
        WorkerFacts {
            key: URL.to_owned(),
            engine: URL.to_owned(),
            incarnation: 17,
            wire_modes: WireModes::HTTP,
            dp_size: 1,
            ..WorkerFacts::default()
        }
    }

    #[test]
    fn a_found_worker_is_registered_under_its_maker() {
        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
        let wrote = door.found(&maker, URL, &|standing| {
            assert!(standing.is_none());
            stated()
        });
        let row = store.by_key(URL).expect("the registered row");
        assert_eq!(wrote, Ok(Wrote::Registered(row.id)));
        assert_eq!(row.maker, maker);
        assert_eq!(
            row.facts,
            WorkerFacts {
                incarnation: membership::life(0, door.base, 0),
                ..stated()
            }
        );
        assert_eq!(
            (row.status, row.revision, store.epoch()),
            (WorkerStatus::Pending, 1, 1)
        );
    }

    #[test]
    fn equal_facts_write_nothing_and_tell_nobody() {
        let store = Arc::new(MemoryStore::default());
        let count = Arc::new(AtomicUsize::new(0));
        let heard = Arc::clone(&count);
        let door = StoreDoor::new(Arc::clone(&store)).telling(move || {
            heard.fetch_add(1, Ordering::Relaxed);
        });
        let maker = Maker::provider("pool", "first").expect("a maker");
        door.found(&maker, URL, &|_| stated()).expect("registered");
        let row = store.by_key(URL).expect("the registered row");
        assert_eq!(
            door.found(&maker, URL, &|_| stated()),
            Ok(Wrote::Unmoved(row.id))
        );
        assert_eq!(store.by_key(URL), Some(row));
        assert_eq!(store.epoch(), 1);
        assert_eq!(count.load(Ordering::Relaxed), 1);

        assert!(matches!(
            door.gone(&Maker::Control, URL),
            Err(Refused::NotMaker(_))
        ));
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_changed_label_is_written_in_place_and_a_changed_role_is_another_life() {
        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
        door.found(&maker, URL, &|_| stated()).expect("registered");
        let old = store.by_key(URL).expect("the registered row");
        let labelled = WorkerFacts {
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            ..stated()
        };
        assert_eq!(
            door.found(&maker, URL, &|_| labelled.clone()),
            Ok(Wrote::InPlace(old.id))
        );
        let standing = store.by_key(URL).expect("the refreshed row");
        assert_eq!(standing.id, old.id);
        assert_eq!(standing.facts.incarnation, old.facts.incarnation);
        assert_eq!(standing.facts.labels, labelled.labels);
        assert_eq!(standing.revision, 2);

        let before = store.epoch();
        let wrote = door.found(&maker, URL, &|_| WorkerFacts {
            role: Role::Decode,
            ..labelled.clone()
        });
        let row = store.by_key(URL).expect("the replaced row");
        assert_eq!(
            wrote,
            Ok(Wrote::Another {
                old: old.id,
                new: row.id
            })
        );
        assert_ne!(row.id, old.id);
        assert_eq!(store.get(old.id), None);
        assert_eq!(row.maker, maker);
        assert_eq!(
            row.facts.incarnation,
            membership::life(0, door.base, before)
        );
        assert_eq!((row.facts.role, row.revision), (Role::Decode, 1));
    }

    #[test]
    fn another_makers_row_refuses_every_write_and_stands() {
        let caller = Maker::provider("pool", "second").expect("a maker");
        for (maker, why) in [
            (
                Maker::Control,
                "the row was made by a control call: its maker alone changes it",
            ),
            (
                Maker::provider("pool", "first").expect("a maker"),
                "the row is provider pool's under another key: that key's caller changes it",
            ),
        ] {
            let (store, door) = door();
            door.found(&maker, URL, &|_| stated()).expect("registered");
            let row = store.by_key(URL).expect("the registered row");
            let expected = Err(Refused::NotMaker(why.to_owned()));
            assert_eq!(
                door.found(&caller, URL, &|_| panic!("only the maker states facts")),
                expected
            );
            assert_eq!(door.drained(&caller, URL), expected);
            assert_eq!(door.gone(&caller, URL), expected);
            assert_eq!(store.by_key(URL), Some(row));
            assert_eq!(store.epoch(), 1);
        }
    }

    #[test]
    fn a_fact_past_its_bound_writes_nothing() {
        struct Shut<'a>(&'a MemoryStore);

        impl Turn for Shut<'_> {
            fn store(&self) -> &MemoryStore {
                self.0
            }

            fn life(&self) -> Result<u64, Refused> {
                Err(Refused::NotLeading("no life".to_owned()))
            }

            fn room(&self, _: &WorkerFacts, _: &Maker) -> Result<(), Refused> {
                Err(Refused::Past(Past {
                    fact: "row",
                    why: "no room",
                }))
            }
        }

        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
        let invalid = WorkerFacts {
            models: vec!["m".repeat(257)],
            ..stated()
        };
        let no = Err(Refused::Past(Past {
            fact: "models",
            why: "a model id is at most 256 bytes",
        }));
        assert_eq!(door.found(&maker, URL, &|_| invalid.clone()), no);
        assert_eq!(lay(&Shut(&store), &maker, URL, &|_| invalid.clone()), no);
        assert_eq!(
            lay(&Shut(&store), &maker, URL, &|_| stated()),
            Err(Refused::Past(Past {
                fact: "row",
                why: "no room"
            }))
        );
        assert_eq!(store.epoch(), 0);
        assert_eq!(store.by_key(URL), None);

        store.upsert_by(invalid.clone(), maker.clone());
        let standing = store.by_key(URL);
        assert_eq!(door.found(&maker, URL, &|_| invalid.clone()), no);
        assert_eq!(store.by_key(URL), standing);
        assert_eq!(store.epoch(), 1);
    }

    #[test]
    fn a_drain_marks_the_makers_row_once_and_a_removal_takes_it() {
        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
        door.found(&maker, URL, &|_| stated()).expect("registered");
        let row = store.by_key(URL).expect("the registered row");
        let stale = row.revision - 1;
        assert_eq!(
            changed(
                store.set_status(row.id, WorkerStatus::Draining, stale),
                Wrote::Draining
            ),
            Err(Refused::Moved)
        );
        assert_eq!(
            changed(store.remove(row.id, stale), Wrote::Removed),
            Err(Refused::Moved)
        );
        assert_eq!(door.drained(&maker, URL), Ok(Wrote::Draining(row.id)));
        assert_eq!(
            store.by_key(URL),
            Some(WorkerRow {
                status: WorkerStatus::Draining,
                ..row.clone()
            })
        );
        assert_eq!(store.epoch(), 2);
        assert_eq!(door.drained(&maker, URL), Ok(Wrote::Unmoved(row.id)));
        assert_eq!(store.epoch(), 2);
        assert_eq!(door.gone(&maker, URL), Ok(Wrote::Removed(row.id)));
        assert_eq!(store.epoch(), 3);
        assert_eq!(store.by_key(URL), None);
    }

    #[test]
    fn no_row_under_the_url_is_absent() {
        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
        assert_eq!(door.drained(&maker, URL), Err(Refused::Absent));
        assert_eq!(door.gone(&maker, URL), Err(Refused::Absent));
        assert_eq!(store.epoch(), 0);
    }

    #[test]
    fn the_listener_is_told_once_a_write_with_the_turn_released() {
        let store = Arc::new(MemoryStore::default());
        let count = Arc::new(AtomicUsize::new(0));
        let heard = Arc::clone(&count);
        let maker = Maker::provider("pool", "first").expect("a maker");
        let door = Arc::new_cyclic(|door: &Weak<StoreDoor>| {
            let door = door.clone();
            let maker = maker.clone();
            StoreDoor::new(store).telling(move || {
                let door = door.upgrade().expect("the calling door");
                assert_eq!(door.gone(&maker, "http://absent"), Err(Refused::Absent));
                heard.fetch_add(1, Ordering::Relaxed);
            })
        });
        let (done, finished) = mpsc::channel();
        let writer = thread::spawn(move || {
            let found = door.found(&maker, URL, &|_| stated());
            let drained = door.drained(&maker, URL);
            let gone = door.gone(&maker, URL);
            done.send([found, drained, gone]).expect("the answers");
        });
        let [found, drained, gone] = finished
            .recv_timeout(Duration::from_secs(5))
            .expect("a listener can call the door");
        let Ok(Wrote::Registered(id)) = found else {
            panic!("the row was not registered: {found:?}")
        };
        assert_eq!(drained, Ok(Wrote::Draining(id)));
        assert_eq!(gone, Ok(Wrote::Removed(id)));
        assert_eq!(count.load(Ordering::Relaxed), 3);
        writer.join().expect("the writer finished");
    }

    #[test]
    fn a_worker_whose_checks_are_off_is_ready_at_once() {
        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
        door.found(&maker, URL, &|_| stated()).expect("registered");
        assert_eq!(
            store.by_key(URL).map(|row| row.status),
            Some(WorkerStatus::Pending)
        );
        let wrote = door.found(&maker, URL, &|_| {
            let mut facts = stated();
            facts.health.probes = false;
            facts
        });
        let row = store.by_key(URL).expect("the row with its checks off");
        assert_eq!(wrote, Ok(Wrote::InPlace(row.id)));
        assert_eq!((row.status, row.revision), (WorkerStatus::Ready, 2));

        door.gone(&maker, URL).expect("removed");
        let wrote = door.found(&maker, URL, &|_| row.facts.clone());
        let fresh = store.by_key(URL).expect("the newly registered row");
        assert_eq!(wrote, Ok(Wrote::Registered(fresh.id)));
        assert_eq!((fresh.status, fresh.revision), (WorkerStatus::Ready, 1));
    }

    #[test]
    fn calls_at_once_take_their_turn() {
        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
        let ready = Barrier::new(4);
        let answers = thread::scope(|scope| {
            let callers = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        ready.wait();
                        door.found(&maker, URL, &|_| {
                            assert!(
                                door.turn.is_locked(),
                                "the facts are stated in the door's turn"
                            );
                            stated()
                        })
                    })
                })
                .collect::<Vec<_>>();
            callers
                .into_iter()
                .map(|caller| caller.join().expect("the call finished"))
                .collect::<Vec<_>>()
        });
        let row = store.by_key(URL).expect("the registered row");
        assert_eq!(
            answers
                .iter()
                .filter(|answer| **answer == Ok(Wrote::Registered(row.id)))
                .count(),
            1
        );
        assert_eq!(
            answers
                .iter()
                .filter(|answer| **answer == Ok(Wrote::Unmoved(row.id)))
                .count(),
            3
        );
        assert_eq!(store.epoch(), 1);
    }

    #[test]
    fn the_facts_are_stated_over_the_row_that_stands() {
        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
        door.found(&maker, URL, &|_| WorkerFacts {
            zone: Some("east".to_owned()),
            ..stated()
        })
        .expect("registered");
        let old = store.by_key(URL).expect("the registered row");
        let wrote = door.found(&maker, URL, &|standing| {
            assert_eq!(standing, Some(&old.facts));
            WorkerFacts {
                zone: standing.and_then(|facts| facts.zone.clone()),
                priority: 5,
                ..stated()
            }
        });
        assert_eq!(wrote, Ok(Wrote::InPlace(old.id)));
        let row = store.by_key(URL).expect("the refreshed row");
        assert_eq!(row.facts.zone.as_deref(), Some("east"));
        assert_eq!(row.facts.priority, 5);

        #[cfg(debug_assertions)]
        {
            use std::panic::{catch_unwind, AssertUnwindSafe};

            let mismatch = catch_unwind(AssertUnwindSafe(|| {
                door.found(&maker, URL, &|_| WorkerFacts {
                    key: "http://another".to_owned(),
                    ..stated()
                })
            }));
            assert!(mismatch.is_err());
            assert_eq!(store.by_key(URL), Some(row));
            assert_eq!(store.by_key("http://another"), None);
        }
    }
}
