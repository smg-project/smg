//! The membership call: a membership write tells the engine of the world
//! it published, and a generation is first told every member.

use super::{facts, member, MODEL};
use crate::{
    store::{MemoryStore, WorkerFacts, WorkersStore},
    view::{Role, Session, WorkerId},
    SessionVerdict,
};

/// The revision `id`'s row stands at.
fn revision(store: &MemoryStore, id: WorkerId) -> u64 {
    WorkersStore::get(store, id).expect("the row").revision
}

/// A worker that leaves by the store's doors leaves the engine's session
/// table with it: a removal drops the id's pins, and so does the
/// replacement of its incarnation.
#[test]
fn a_worker_that_leaves_by_the_store_s_doors_leaves_the_engine_s_pins() {
    let store = MemoryStore::new();
    let removed = member(&store, "http://a:8000");
    let replaced = member(&store, "http://b:8000");
    let pins = |key: u64| {
        store.deciding(|world| {
            let table = world.spec().engine().sessions();
            table.pins(Session::from_raw(key))
        })
    };
    store.deciding(|world| {
        let table = world.spec().engine().sessions();
        table.dispatched(Session::from_raw(1), SessionVerdict::Pinned, removed);
        table.dispatched(Session::from_raw(2), SessionVerdict::Pinned, replaced);
    });
    assert!(pins(1).is_some() && pins(2).is_some());
    store
        .remove(removed, revision(&store, removed))
        .expect("removed");
    assert!(pins(1).is_none(), "the removed worker kept its pin");
    store.upsert(WorkerFacts {
        incarnation: 2,
        ..facts("http://b:8000", Role::Single, &[MODEL])
    });
    assert!(pins(2).is_none(), "the replaced incarnation kept its pin");
}

/// The cases that hold a stage of their own: a generation over a host's
/// catalog is made at the controller's door.
#[cfg(feature = "controller")]
mod held {
    use std::{
        sync::{Arc, Weak},
        time::Instant,
    };

    use parking_lot::Mutex;

    use super::{
        super::{ask, facts, member, status, MODEL},
        revision,
    };
    use crate::{
        framework::{Scorer, TieBand},
        lease::{Lifecycle, Membership},
        spec::{BindingSpec, Needs},
        store::{
            world::ViewBuffers, Kinds, MembershipChange, MemoryStore, WorkerFacts, WorkerPatch,
            WorkerStatus, WorkersStore,
        },
        view::{Candidate, Constraints, Identity, RequestFacts, Role, WorkerId},
        AppliedSpec, Catalog, ConsistentHash, PolicySpec, Scratch,
    };

    /// What the store told a stage: each call as it prints, with the epoch
    /// of the world the store showed when the call came.
    type Told = Arc<Mutex<Vec<(u64, String)>>>;

    /// The stock ring scorer under a name of the case's own: the case holds
    /// the instance its generation scores with, and every membership call
    /// that stage took.
    #[derive(Debug)]
    struct HeldRing {
        ring: Arc<ConsistentHash>,
        told: Told,
        store: Weak<MemoryStore>,
    }

    impl Lifecycle for HeldRing {
        fn on_membership(&self, change: &Membership<'_>) {
            let store = self.store.upgrade();
            if let Some(store) = &store {
                // The call comes under the membership lock, so the writes
                // arrive in order, and with the load lock let go, so a load
                // write waits for no stage.
                assert!(
                    !store.workers_lock_free(),
                    "told outside the membership lock"
                );
                assert!(store.load_lock_free(), "told under the load lock");
            }
            let shown = store.map_or(0, |store| store.world().epoch());
            self.told.lock().push((shown, format!("{change:?}")));
            self.ring.on_membership(change);
        }

        fn needs_lifecycle(&self) -> bool {
            self.ring.needs_lifecycle()
        }
    }

    impl Scorer for HeldRing {
        fn name(&self) -> &'static str {
            "held-ring"
        }

        fn score(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            self.ring.score(facts, candidates, kept, out)
        }

        fn needs(&self) -> Needs {
            self.ring.needs()
        }
    }

    /// Generation `number` of one profile for every model: the held ring
    /// scores and the best row is picked.
    fn generation(
        number: u64,
        store: &Arc<MemoryStore>,
        ring: &Arc<ConsistentHash>,
        told: &Told,
    ) -> AppliedSpec {
        let (ring, told, store) = (Arc::clone(ring), Arc::clone(told), Arc::downgrade(store));
        let mut catalog = Catalog::builtin();
        catalog
            .scorer("held-ring", move || {
                Box::new(HeldRing {
                    ring: Arc::clone(&ring),
                    told: Arc::clone(&told),
                    store: Weak::clone(&store),
                })
            })
            .expect("a free name");
        let mut document: PolicySpec = serde_json::from_str(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":["held-ring"],"pick":"best"}}}"#,
        )
        .expect("a document");
        document.generation = number;
        document.bindings = vec![BindingSpec {
            profile: Some("p".to_string()),
            ..BindingSpec::default()
        }];
        let digest = document.digest().expect("a digest");
        AppliedSpec::validate_digested(&document, digest, &catalog).expect("the document validates")
    }

    /// The worker a decision keyed `key` goes to at this instant; its view
    /// stays in `out`.
    fn decided(
        store: &MemoryStore,
        key: u64,
        out: &mut ViewBuffers,
        scratch: &mut Scratch,
    ) -> WorkerId {
        store.deciding(|world| {
            let built = world
                .build(MODEL, Role::Single, &ask(Instant::now()), out)
                .expect("the pool");
            let request = RequestFacts {
                model: MODEL,
                session_key: Identity::hashed(key),
                now_micros: built.now_micros,
                ..RequestFacts::default()
            };
            let decision = world.spec().engine().decide(
                &request,
                &built.pool_facts(),
                &out.candidates,
                &Constraints::default(),
                scratch,
            );
            let pick = decision.pick().expect("a member");
            pick.targets.iter().next().expect("a target").id
        })
    }

    /// The row `scorer` ranks first for `key` among four.
    fn leader(scorer: &ConsistentHash, key: u64, pool: &[Candidate]) -> Option<usize> {
        let request = RequestFacts {
            model: MODEL,
            session_key: Identity::hashed(key),
            ..RequestFacts::default()
        };
        let mut scores = [0.0_f32; 4];
        scorer.score(&request, pool, &[0, 1, 2, 3], &mut scores);
        (0..4).max_by(|a, b| scores[*a].total_cmp(&scores[*b]))
    }

    /// One of four workers is replaced 2,000 times through the store's
    /// doors, by a new incarnation and by a removal and a registration in
    /// turn, a keyed decision after each. The ring the generation scores
    /// with holds the four live workers, and ranks first for each key the
    /// worker a ring of those four alone ranks first. A store that tells
    /// its engine nothing leaves that ring 2,004 members, and every
    /// decision's walk the longer.
    #[test]
    fn a_ring_holds_the_live_workers_after_two_thousand_replacements() {
        let store = Arc::new(MemoryStore::new());
        let (ring, told) = (Arc::new(ConsistentHash::new(None)), Told::default());
        store
            .apply(generation(1, &store, &ring, &told))
            .expect("applied");
        let join = |slot: usize, incarnation: u64| {
            let key = format!("http://w{slot}:8000");
            store
                .upsert(WorkerFacts {
                    incarnation,
                    ..facts(&key, Role::Single, &[MODEL])
                })
                .id()
        };
        let mut live: Vec<WorkerId> = (0..4).map(|slot| join(slot, 1)).collect();
        let (mut out, mut scratch) = (ViewBuffers::default(), Scratch::new());
        decided(&store, 77, &mut out, &mut scratch);
        assert_eq!(ring.ring().len(), 4);
        for life in 0..2_000_u64 {
            let slot = (life % 4) as usize;
            if life % 8 >= 4 {
                store
                    .remove(live[slot], revision(&store, live[slot]))
                    .expect("removed");
            }
            live[slot] = join(slot, 2 + life);
            let target = decided(&store, 77, &mut out, &mut scratch);
            assert!(live.contains(&target), "life {life}: a worker that left");
        }
        assert_eq!(ring.ring().len(), 4, "the ring kept workers that left");
        let fresh = ConsistentHash::new(None);
        for key in 1..=500 {
            let (held, alone) = (
                leader(&ring, key, &out.candidates),
                leader(&fresh, key, &out.candidates),
            );
            assert_eq!(held, alone, "key {key}");
        }
    }

    /// A generation is first told every member once, by id, then each
    /// membership write in the writes' order, each when the store shows
    /// the world that write made; a status change is no membership event.
    /// The generation after it is told the members that stand then, and
    /// the one it replaced hears no more.
    #[test]
    fn a_generation_is_told_its_members_once_then_each_write_in_order() {
        let store = Arc::new(MemoryStore::new());
        let mut first: Vec<WorkerId> = (0..8)
            .map(|n| member(&store, &format!("http://w{n}:8000")))
            .collect();
        let (a, b) = (first[0], first[1]);
        let (ring, told) = (Arc::new(ConsistentHash::new(None)), Told::default());
        store
            .apply(generation(1, &store, &ring, &told))
            .expect("applied");
        let at = WorkersStore::epoch(&*store);
        let c = member(&store, "http://c:8000");
        status(&store, c, WorkerStatus::Ready);
        let priority = WorkerPatch {
            priority: Some(3),
            ..WorkerPatch::default()
        };
        store
            .patch(a, priority, revision(&store, a))
            .expect("patched");
        store.upsert(WorkerFacts {
            dp_size: 2,
            ..facts("http://w1:8000", Role::Single, &[MODEL])
        });
        let d = store
            .upsert(WorkerFacts {
                incarnation: 2,
                ..facts("http://c:8000", Role::Single, &[MODEL])
            })
            .id();
        store.remove(a, revision(&store, a)).expect("removed");
        first.sort_unstable();
        // The status change took the epoch between the second and the third.
        assert_eq!(
            *told.lock(),
            [
                (at, format!("{:?}", Membership::Snapshot(&first))),
                (at + 1, format!("{:?}", Membership::Added(c))),
                (at + 3, format!("{:?}", Membership::Relabelled(a))),
                (at + 4, format!("{:?}", Membership::Resized(b))),
                (
                    at + 5,
                    format!("{:?}", Membership::Replaced { old: c, new: d })
                ),
                (at + 6, format!("{:?}", Membership::Removed(a))),
            ]
        );
        let (next_ring, next_told) = (Arc::new(ConsistentHash::new(None)), Told::default());
        store
            .apply(generation(2, &store, &next_ring, &next_told))
            .expect("applied");
        store.remove(b, revision(&store, b)).expect("removed");
        let mut second: Vec<WorkerId> = first
            .iter()
            .copied()
            .filter(|id| *id != a)
            .chain([d])
            .collect();
        second.sort_unstable();
        assert_eq!(
            *next_told.lock(),
            [
                (at + 6, format!("{:?}", Membership::Snapshot(&second))),
                (at + 7, format!("{:?}", Membership::Removed(b))),
            ]
        );
        assert_eq!(told.lock().len(), 6, "a generation that left was told");
    }

    /// A standby's two doors tell the engine as a leader's writes do: a
    /// restore tells one snapshot, the restored members alone, and a
    /// followed write the change it applied. The ring keeps no row the
    /// restore replaced.
    #[test]
    fn a_restore_tells_one_snapshot_and_a_followed_write_its_change() {
        let leader = MemoryStore::new();
        let mut members = vec![
            member(&leader, "http://a:8000"),
            member(&leader, "http://b:8000"),
        ];
        members.sort_unstable();
        let standby = Arc::new(MemoryStore::new());
        let (ring, told) = (Arc::new(ConsistentHash::new(None)), Told::default());
        standby
            .apply(generation(1, &standby, &ring, &told))
            .expect("applied");
        // A row of the standby's own joins the ring at a decision.
        member(&standby, "http://own:8000");
        let (mut out, mut scratch) = (ViewBuffers::default(), Scratch::new());
        decided(&standby, 77, &mut out, &mut scratch);
        assert_eq!(ring.ring().len(), 1);
        told.lock().clear();
        assert!(standby.restore_from(leader.cut(Kinds::All)));
        let snapshot = format!("{:?}", Membership::Snapshot(&members));
        assert_eq!(*told.lock(), [(2, snapshot)]);
        let target = decided(&standby, 77, &mut out, &mut scratch);
        assert!(members.contains(&target), "a row the restore replaced");
        assert_eq!(ring.ring().len(), 2, "the ring kept a replaced row");
        // A followed registration is told as the leader's own is.
        let joined = member(&leader, "http://c:8000");
        let row = WorkersStore::get(&leader, joined).expect("the row");
        standby.follow(3, MembershipChange::Registered { id: joined }, Some(row));
        let added = format!("{:?}", Membership::Added(joined));
        assert_eq!(told.lock().last(), Some(&(3, added)));
    }
}
