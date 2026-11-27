//! A worker's ingress: a registration and its replacement, a row's revision,
//! status, health and maker, a gateway's veto, the snapshot of the rows, and
//! what the store holds its locks for.

use super::*;

#[test]
fn a_registration_is_keyed_and_a_new_incarnation_replaces_it_under_a_new_id() {
    let store = MemoryStore::new();
    let workers = &store;
    let first = workers.upsert(facts("http://w:8000", 1));
    let MembershipChange::Registered { id } = first else {
        panic!("a new key registers: {first:?}");
    };
    assert_eq!(id, identity("http://w:8000", 1));
    assert_eq!(
        workers.upsert(facts("http://w:8000", 1)),
        MembershipChange::Relabelled { id },
        "the same incarnation refreshes in place"
    );
    let mut wider = facts("http://w:8000", 1);
    wider.dp_size = 4;
    assert_eq!(
        workers.upsert(wider),
        MembershipChange::Resized { id },
        "a changed width is a resize"
    );
    let replaced = workers.upsert(facts("http://w:8000", 2));
    let MembershipChange::Replaced { old, new } = replaced else {
        panic!("a new incarnation replaces: {replaced:?}");
    };
    assert_eq!(old, id);
    assert_eq!(new, identity("http://w:8000", 2));
    assert_ne!(old, new);
    assert!(WorkersStore::get(workers, old).is_none(), "the old id left");
    assert_eq!(workers.by_key("http://w:8000").map(|row| row.id), Some(new));
    assert_eq!(
        replaced.as_lifecycle(),
        Some(Membership::Replaced { old, new })
    );
}

#[test]
fn a_replacement_carries_a_matching_registrations_status_and_a_changed_one_starts_pending() {
    let store = MemoryStore::new();
    let id = ready(&store, "http://w:8000");
    let same = store.upsert(facts("http://w:8000", 2)).id();
    assert_eq!(
        WorkersStore::get(&store, same).expect("the row").status,
        WorkerStatus::Ready
    );
    // What a registration states for a hard demand is none of the three
    // facts: its next life is handed the status too.
    let mut other_runtime = facts("http://w:8000", 3);
    other_runtime.demand_facts = DemandFacts {
        runtime: 9,
        ..DemandFacts::default()
    };
    let stated = store.upsert(other_runtime).id();
    assert_eq!(
        WorkersStore::get(&store, stated).expect("the row").status,
        WorkerStatus::Ready
    );
    let mut other_role = facts("http://w:8000", 4);
    other_role.role = Role::Prefill;
    let changed = store.upsert(other_role).id();
    assert_eq!(
        WorkersStore::get(&store, changed).expect("the row").status,
        WorkerStatus::Pending
    );
    assert_ne!(id, changed);
}

#[test]
fn a_write_under_a_moved_revision_is_refused_and_never_reaches_a_successor() {
    let store = MemoryStore::new();
    let workers = &store;
    let id = workers.upsert(facts("http://w:8000", 1)).id();
    let stale = WorkersStore::get(workers, id).expect("the row").revision;
    assert!(workers
        .patch(id, WorkerPatch::default(), stale)
        .is_some_and(|change| change == MembershipChange::Relabelled { id }));
    assert!(
        workers.remove(id, stale).is_none(),
        "the patch moved the revision"
    );
    assert!(workers.set_status(id, WorkerStatus::Ready, stale).is_none());
    assert!(WorkersStore::get(workers, id).is_some(), "the row stands");
    let current = WorkersStore::get(workers, id).expect("the row").revision;
    let status = workers.set_status(id, WorkerStatus::Ready, current);
    assert_eq!(
        status,
        Some(MembershipChange::StatusChanged {
            id,
            from: WorkerStatus::Pending,
            to: WorkerStatus::Ready
        })
    );
    assert!(
        status.and_then(|change| change.as_lifecycle()).is_none(),
        "a status change is a health signal, not a membership event"
    );
    let current = WorkersStore::get(workers, id).expect("the row").revision;
    assert!(
        workers
            .set_status(id, WorkerStatus::Ready, current)
            .is_none(),
        "a status that already stood emits nothing"
    );
    assert_eq!(
        workers.remove(id, current),
        Some(MembershipChange::Removed { id })
    );
    assert!(WorkersStore::get(workers, id).is_none());
}

#[test]
fn a_snapshot_holds_every_row_at_its_epoch_ordered_by_key() {
    let store = MemoryStore::new();
    let workers = &store;
    let (epoch, rows) = workers.all_rows();
    assert_eq!((epoch, rows.len()), (0, 0));
    let a = workers.upsert(facts("http://a:8000", 1)).id();
    let (epoch, rows) = workers.all_rows();
    assert_eq!((epoch, rows.len()), (1, 1));
    workers.upsert(facts("http://b:8000", 1));
    let (epoch, rows) = workers.all_rows();
    assert_eq!((epoch, rows.len()), (2, 2));
    assert_eq!(rows[0].id, a, "the snapshot is ordered by key");
}

#[test]
fn snapshot_sort_does_not_hold_the_workers_lock() {
    let store = Arc::new(MemoryStore::new());
    let id = store.upsert(facts("http://c:8000", 1)).id();
    store.upsert(facts("http://a:8000", 1));
    store.upsert(facts("http://b:8000", 1));
    let revision = WorkersStore::get(store.as_ref(), id)
        .expect("the row")
        .revision;
    let epoch = store.epoch();
    let (cloned, resume) = store.pause_snapshot_after_clone();

    let snapshot_store = Arc::clone(&store);
    let snapshot = std::thread::spawn(move || WorkersStore::all_rows(snapshot_store.as_ref()));
    cloned
        .recv_timeout(PATIENCE)
        .expect("snapshot reached the sort");

    let writer_store = Arc::clone(&store);
    let (written_tx, written_rx) = mpsc::channel();
    let writer = std::thread::spawn(move || {
        let change = writer_store.set_status(id, WorkerStatus::Ready, revision);
        written_tx
            .send(change)
            .expect("the test receives the write");
    });
    let written_while_paused = written_rx.recv_timeout(PATIENCE);

    resume.send(()).expect("resume the snapshot");
    let (snapshot_epoch, rows) = snapshot.join().expect("snapshot thread");
    writer.join().expect("writer thread");
    assert_eq!(snapshot_epoch, epoch, "epoch was read with the rows");
    assert_eq!(
        rows.iter()
            .map(|row| row.facts.key.as_str())
            .collect::<Vec<_>>(),
        ["http://a:8000", "http://b:8000", "http://c:8000"],
        "snapshot rows are ordered by key"
    );
    assert_eq!(
        rows.iter().find(|row| row.id == id).map(|row| row.status),
        Some(WorkerStatus::Pending),
        "the snapshot keeps the row copied before the status write"
    );
    assert!(
        matches!(
            written_while_paused,
            Ok(Some(MembershipChange::StatusChanged {
                id: changed,
                from: WorkerStatus::Pending,
                to: WorkerStatus::Ready,
            })) if changed == id
        ),
        "status write waited for snapshot sorting"
    );
}

/// A status its holder tells says what happened to it, and the status that
/// stands where it was judged: it moved, it stood, its writer's last word
/// said no, the row left the revision it was told under, or no row stands.
#[cfg(feature = "worker-host")]
#[test]
fn a_told_status_says_what_happened_and_the_status_that_stands() {
    use super::super::{
        HealthWrite::{Moved, NoRow, Stale, Stood},
        LeadWord,
    };

    let store = MemoryStore::new();
    let id = store.upsert(facts("http://w:8000", 1)).id();
    let revision = || WorkersStore::get(&store, id).expect("the row").revision;
    // The store has no journal: its holder hands it no models.
    let told = |status, revision, leads: bool| {
        store.status_told(id, status, revision, LeadWord::of(&|| leads), None)
    };
    // A status the row holds already stands, and nothing is written.
    let epoch = store.epoch();
    assert_eq!(
        told(WorkerStatus::Pending, revision(), true),
        Stood(WorkerStatus::Pending)
    );
    assert_eq!(store.epoch(), epoch);
    assert_eq!(
        told(WorkerStatus::Ready, revision(), true),
        Moved {
            from: WorkerStatus::Pending,
            to: WorkerStatus::Ready
        }
    );
    // A writer whose last word says no writes nothing.
    assert_eq!(told(WorkerStatus::NotReady, revision(), false), Stale);
    // A status told under a revision the row left is stale: it says no
    // status, and the row stands as it stood.
    let left = revision();
    store
        .patch(id, WorkerPatch::default(), left)
        .expect("the patch");
    assert_eq!(told(WorkerStatus::NotReady, left, true), Stale);
    let stands = WorkersStore::get(&store, id).expect("the row").status;
    assert_eq!((stands, store.epoch()), (WorkerStatus::Ready, epoch + 2));
    // A status told for a worker whose row is gone says no row.
    let last = revision();
    store.remove(id, last).expect("removed");
    assert_eq!(told(WorkerStatus::NotReady, last, true), NoRow);
}

/// A status told under a term goes to the journal ahead of the table's
/// move. Under the term the log leads, the event tells the epoch the
/// table then takes, and the status the row takes. Under a term the
/// log left, the journal takes no event and nothing moved: the status,
/// the table's epoch and the log's head stand. A word that names no term
/// writes there, as before.
#[cfg(feature = "worker-host")]
#[test]
fn a_status_under_a_term_is_journaled_under_it_or_moves_nothing() {
    use super::super::{
        HealthWrite::{Moved, Stale},
        LeadWord,
    };
    use crate::service::replication::{Entry, Event};

    let log = Arc::new(Log::new(8));
    let journal: Arc<Log> = Arc::clone(&log);
    let store = MemoryStore::journaled(journal);
    store.set_term(3);
    let id = store.upsert(facts("http://w:8000", 1)).id();
    let revision = WorkersStore::get(&store, id).expect("the row").revision;
    let under = |status: WorkerStatus, term: u64| {
        let word = LeadWord {
            leads: &|| true,
            term: Some(term),
        };
        // This writer hands no models: the event names the table's row's.
        store.status_told(id, status, revision, word, None)
    };
    let stands = || WorkersStore::get(&store, id).expect("the row").status;
    // Under the term the log leads.
    let before = log.head();
    let ready = Moved {
        from: WorkerStatus::Pending,
        to: WorkerStatus::Ready,
    };
    assert_eq!(under(WorkerStatus::Ready, 3), ready);
    let mut told: Vec<Entry> = Vec::new();
    log.read(before, 2, &mut told).expect("the place stands");
    let [Entry {
        at,
        event: Event::Status {
            epoch, from, to, ..
        },
    }] = told.as_slice()
    else {
        panic!("one status event: {told:?}");
    };
    assert_eq!((at.term, *epoch), (3, store.epoch()));
    assert_eq!(
        (*from, *to, stands()),
        (
            WorkerStatus::Pending,
            WorkerStatus::Ready,
            WorkerStatus::Ready
        )
    );
    // The log leaves the term by a later take.
    store.set_term(4);
    let (epoch, head) = (store.epoch(), log.head());
    assert_eq!(under(WorkerStatus::NotReady, 3), Stale);
    assert_eq!(
        (stands(), store.epoch(), log.head()),
        (WorkerStatus::Ready, epoch, head)
    );
    // A word that names no term is journaled as the store's own.
    let told = store.status_told(id, WorkerStatus::NotReady, revision, LeadWord::FREE, None);
    assert!(told.moved().is_some());
    assert_eq!(
        (stands(), store.epoch()),
        (WorkerStatus::NotReady, epoch + 1)
    );
    let next = Position {
        seq: head.seq + 1,
        term: 4,
    };
    assert_eq!(log.head(), next);
}

#[test]
fn a_health_writes_two_readers_say_the_transition_and_the_status_that_stands() {
    use super::super::HealthWrite::{Moved, NoRow, Stale, Stood};

    let moved = Moved {
        from: WorkerStatus::Pending,
        to: WorkerStatus::Ready,
    };
    assert_eq!(
        moved.moved(),
        Some((WorkerStatus::Pending, WorkerStatus::Ready))
    );
    assert_eq!(moved.stands(), Some(WorkerStatus::Ready));
    let stood = Stood(WorkerStatus::Draining);
    assert_eq!(stood.moved(), None);
    assert_eq!(stood.stands(), Some(WorkerStatus::Draining));
    for unjudged in [Stale, NoRow] {
        assert_eq!(unjudged.moved(), None);
        assert_eq!(unjudged.stands(), None);
    }
}

#[test]
fn a_gateway_veto_masks_the_worker_for_that_gateway_until_its_end_or_its_clear() {
    let store = MemoryStore::new();
    // A registered worker: the health writers refuse an id without a row.
    let id = ready(&store, "http://v:8000");
    let health: &dyn HealthSink = &store;
    let (g1, g2) = (GatewayId(1), GatewayId(2));
    let now = Instant::now();
    health.gateway_veto(g1, id, Some(now + Duration::from_secs(60)));
    assert!(health.vetoed(g1, id, now));
    assert!(!health.vetoed(g2, id, now), "another gateway's view");
    assert!(
        !health.vetoed(g1, id, now + Duration::from_secs(61)),
        "the veto ends"
    );
    health.gateway_veto(g1, id, None);
    assert!(!health.vetoed(g1, id, now), "an explicit clear");
}

#[test]
fn a_removed_worker_leaves_the_health_tables_and_takes_no_late_write() {
    let store = MemoryStore::new();
    let id = ready(&store, "http://d:8000");
    let health: &dyn HealthSink = &store;
    let (g1, g2) = (GatewayId(1), GatewayId(2));
    let now = Instant::now();
    health.gateway_veto(g1, id, Some(now + Duration::from_secs(60)));
    health.gateway_veto(g2, id, Some(now + Duration::from_secs(1)));
    assert_eq!(store.health_entries(), 2);
    let revision = WorkersStore::get(&store, id).expect("the row").revision;
    store.remove(id, revision).expect("removed");
    assert!(
        !health.vetoed(g1, id, now),
        "a standing veto on a removed worker is gone with it"
    );
    assert_eq!(store.health_entries(), 0);
    // A gateway's in-flight work on the worker ends after the removal: a
    // late veto writes nothing under the gone id.
    health.gateway_veto(g2, id, Some(now + Duration::from_secs(90)));
    assert!(
        !health.vetoed(g2, id, now + Duration::from_secs(3)),
        "a veto after the removal masks nothing"
    );
    assert_eq!(store.health_entries(), 0, "and the tables stay empty");
}

#[test]
fn a_replaced_incarnation_leaves_the_health_tables_with_its_row() {
    let store = MemoryStore::new();
    let old = ready(&store, "http://w:8000");
    let health: &dyn HealthSink = &store;
    let g = GatewayId(1);
    let now = Instant::now();
    health.gateway_veto(g, old, Some(now + Duration::from_secs(60)));
    assert_eq!(store.health_entries(), 1);
    let new = store.upsert(facts("http://w:8000", 2)).id();
    assert_ne!(old, new, "a new incarnation under the key");
    assert!(
        !health.vetoed(g, old, now),
        "the old incarnation's veto left with its row"
    );
    assert_eq!(store.health_entries(), 0, "and the new id starts clean");
}

#[test]
fn a_new_row_takes_its_maker_and_a_standing_row_keeps_it() {
    use super::super::Registrant;

    let store = MemoryStore::new();
    let found = Registrant::provider("cluster", "pods/a").expect("a whole maker");
    let id = store
        .upsert_by(facts("http://w:8000", 1), found.clone())
        .id();
    assert_eq!(WorkersStore::get(&store, id).expect("the row").maker, found);
    // A refresh and a new incarnation keep the maker, whoever calls: the
    // store tests no caller.
    let mut relabelled = facts("http://w:8000", 1);
    relabelled.zone = Some("b".to_string());
    assert_eq!(store.upsert(relabelled).id(), id);
    assert_eq!(WorkersStore::get(&store, id).expect("the row").maker, found);
    let next = store.upsert(facts("http://w:8000", 2)).id();
    assert_ne!(next, id);
    assert_eq!(
        WorkersStore::get(&store, next).expect("the row").maker,
        found
    );
    // A row the plain write makes is a control call's.
    let by_hand = store.upsert(facts("http://x:8000", 1)).id();
    let held = WorkersStore::get(&store, by_hand).expect("the row");
    assert_eq!(held.maker, Registrant::Control);
}

#[test]
fn a_providers_maker_is_made_inside_the_rules_of_its_two_parts() {
    use super::super::{ProviderKeyError, Registrant};

    let Ok(Registrant::Provider(found)) = Registrant::provider("cluster", "pods/a") else {
        panic!("a maker of both parts is made");
    };
    assert_eq!((found.provider(), found.key()), ("cluster", "pods/a"));
    // Each part at its bound is taken, and the key is its provider's own
    // text, blanks and all.
    let name = "p".repeat(Registrant::NAME_BYTES);
    let key = "k ".repeat(Registrant::KEY_BYTES / 2);
    assert!(Registrant::provider(name.as_str(), key.as_str()).is_ok());
    // The part that breaks its rule is named, the name's rules first.
    let unmade =
        |provider: &str, key: &str| Registrant::provider(provider, key).expect_err("no maker");
    // An empty name is refused in its own words, ahead of the token's.
    assert_eq!(
        unmade("", "pods/a"),
        ProviderKeyError::Provider("a provider's row names its provider")
    );
    assert!(matches!(unmade("", ""), ProviderKeyError::Provider(_)));
    assert!(matches!(
        unmade(&format!("{name}p"), "pods/a"),
        ProviderKeyError::Provider(_)
    ));
    // A name is one token: a blank, a control character, a comma, a letter
    // outside ASCII and a character of no width each make none.
    for name in ["a b", "a\nb", "a\u{7}b", "a,b", "\u{e9}", "a\u{200b}b"] {
        let refused = unmade(name, "pods/a");
        assert!(matches!(refused, ProviderKeyError::Provider(_)), "{name:?}");
    }
    assert!(matches!(unmade("cluster", ""), ProviderKeyError::Key(_)));
    assert!(matches!(
        unmade("cluster", &format!("{key}k")),
        ProviderKeyError::Key(_)
    ));
}

/// Formatting the store takes each table's guard once.
#[test]
fn the_store_formats_without_holding_a_table_twice() {
    let store = MemoryStore::new();
    let shown = format!("{store:?}");
    assert!(shown.starts_with("MemoryStore"), "{shown}");
}

#[test]
fn the_identity_rule_reads_a_key_through_its_fingerprint() {
    assert_eq!(
        identity("http://w:8000", 3),
        identity_of(fingerprint(b"http://w:8000"), 3),
        "the wrapper and the rule agree"
    );
    assert_ne!(identity_of(0, 0), WorkerId(0), "an id is never 0");
}

/// What an apply holds the store's two locks for, measured. Three instants
/// are stamped inside the call. The first is the last word the store asks
/// under both locks, the `admit` of [`MemoryStore::apply_admitted`]. The
/// second is the first word a stage built new hears, which the store says
/// under the workers lock once the load lock is let go: a scorer of the
/// case's own stamps it. The third is the journal's word of the generation,
/// the last step under the workers lock. A store of 64 members. Specs of 1,
/// 8 and 32 profiles, each validated ahead of the timed call: 200
/// generations that change the text of the stamping profile alone, then 200
/// that change every profile's.
#[cfg(all(feature = "controller", feature = "in-process"))]
#[test]
#[ignore = "a measurement for the Numbers table, run by name"]
fn an_applys_time_under_the_stores_locks_measured() {
    use std::io::Write as _;

    use crate::{
        document::{AppliedSpec, PolicySpec},
        engine::{
            view::{Candidate, RequestFacts},
            Scorer, TieBand,
        },
        stages::{Lifecycle, StageRegistry},
    };

    /// A scorer that stamps the instant it hears the list of members.
    #[derive(Debug)]
    struct Stamps(Arc<Mutex<Option<Instant>>>);

    impl Lifecycle for Stamps {
        fn on_membership(&self, word: &Membership<'_>) {
            if matches!(word, Membership::Snapshot(_)) {
                *self.0.lock().expect("the case's own lock") = Some(Instant::now());
            }
        }
    }

    impl Scorer for Stamps {
        fn name(&self) -> &'static str {
            "stamps"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            out.fill(0.0);
            TieBand::NORMALIZED
        }
    }

    /// A journal that stamps the instant it is told a generation.
    struct Told(Arc<Mutex<Option<Instant>>>);

    impl Journal for Told {
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

        fn generation(&self, _: Option<Position>, _: u64) {
            *self.0.lock().expect("the case's own lock") = Some(Instant::now());
        }

        fn term_followed(&self, _: Position) {}

        fn head(&self) -> Position {
            Position::default()
        }

        fn followed(&self) -> Option<u64> {
            None
        }

        fn restart(&self, _: Position) -> bool {
            true
        }
    }

    const ROUNDS: usize = 200;
    const MEMBERS: usize = 64;
    let heard = Arc::new(Mutex::new(None));
    let cell = Arc::clone(&heard);
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer("stamps", move || Box::new(Stamps(Arc::clone(&cell))))
        .expect("a free name");
    // A generation of `profiles` profiles, each bound to a model of its
    // own and the first to every other model. The first holds the
    // stamping scorer and a stock one whose weight changes at every
    // generation, so its stages are built new each time. Each other holds
    // a stock scorer whose weight is the text that `turn` changes.
    let spec = |generation: u64, profiles: usize, turn: u64| {
        let own = 1 + generation % 2;
        let mut text = format!(
            r#"{{"apiVersion":"smg.policy/v1","generation":{generation},"profiles":{{"p0":{{"score":["stamps",{{"least-load":{{"weight":{own}}}}}],"pick":"best"}}"#
        );
        for at in 1..profiles {
            let weight = 1 + turn;
            text.push_str(&format!(
                r#","p{at}":{{"score":[{{"least-load":{{"weight":{weight}}}}}],"pick":"best"}}"#
            ));
        }
        text.push_str(r#"},"bindings":["#);
        for at in 1..profiles {
            text.push_str(&format!(
                r#"{{"match":{{"model":"m{at}"}},"profiles":{{"single":"p{at}"}}}},"#
            ));
        }
        text.push_str(r#"{"match":{"model":"*"},"profiles":{"single":"p0"}}]}"#);
        let document: PolicySpec = serde_json::from_str(&text).expect("a policy document");
        AppliedSpec::validate_in(&document, &catalog, &[]).expect("a generation")
    };
    let said = Arc::new(Mutex::new(None));
    let store = MemoryStore::journaled(Arc::new(Told(Arc::clone(&said))));
    for member in 0..MEMBERS {
        ready(&store, &format!("http://w{member}:1"));
    }
    let mut out = std::io::stdout();
    let mut generation = 0_u64;
    for profiles in [1_usize, 8, 32] {
        for changed in [false, true] {
            // Six spans a round, in nanoseconds: to the last word, under
            // both locks, the telling, the three together (the workers
            // lock), behind the locks, and the whole call.
            let mut spans: [Vec<u128>; 6] = Default::default();
            // One round more than is kept: the first of a kind follows
            // another kind's last.
            for round in 0..=ROUNDS {
                generation += 1;
                let turn = if changed { generation % 2 } else { 0 };
                let next = spec(generation, profiles, turn);
                let admitted = Cell::new(None);
                let began = Instant::now();
                let applied = store.apply_admitted(next, || {
                    admitted.set(Some(Instant::now()));
                    true
                });
                let ended = Instant::now();
                assert_eq!(applied, Ok(true), "a number above the standing one");
                let admitted = admitted.get().expect("the store asked its last word");
                let heard = heard.lock().expect("the case's own lock").take();
                let heard = heard.expect("the stamping scorer heard the members");
                let told = said.lock().expect("the case's own lock").take();
                let told = told.expect("the journal was told the generation");
                if round > 0 {
                    for (span, nanos) in spans.iter_mut().zip([
                        admitted.duration_since(began).as_nanos(),
                        heard.duration_since(admitted).as_nanos(),
                        told.duration_since(heard).as_nanos(),
                        told.duration_since(began).as_nanos(),
                        ended.duration_since(told).as_nanos(),
                        ended.duration_since(began).as_nanos(),
                    ]) {
                        span.push(nanos);
                    }
                }
            }
            let [ahead, both, telling, workers, behind, whole] = spans.map(|mut span| {
                span.sort_unstable();
                (span[span.len() / 2], span[span.len() * 9 / 10])
            });
            let kind = if changed {
                "every text changed"
            } else {
                "one text changed"
            };
            writeln!(
                out,
                "an apply, {profiles} profiles, {kind}, {MEMBERS} members, ns (middle, ninth tenth): \
                 to the last word {} {}, under both locks {} {}, the telling {} {}, \
                 the workers lock {} {}, behind the locks {} {}, whole {} {}",
                ahead.0,
                ahead.1,
                both.0,
                both.1,
                telling.0,
                telling.1,
                workers.0,
                workers.1,
                behind.0,
                behind.1,
                whole.0,
                whole.1
            )
            .expect("stdout");
        }
    }
}
