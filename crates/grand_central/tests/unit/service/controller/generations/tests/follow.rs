//! A leader's generation at a replica that follows: which one it runs,
//! which one it refuses, when the source's document stood, and the lock a
//! followed generation enters under.

use std::{
    cell::Cell,
    sync::{mpsc, Arc},
    thread,
    time::Duration,
};

use super::{
    super::{Generations, NOT_ABOVE, NUMBER},
    applied, followed, generations, sent, stored, switched, yaml,
};
use crate::{
    document::{
        applied::AppliedSpec,
        operator::{
            test_support::{faulting, standing, stock, with_line, LEAST_LOAD, RANDOM},
            ApplyOutcome, Format,
        },
        spec::{Digest, PolicySpec, SpecError},
    },
    service::{
        controller::{wire, worker_wire},
        proto,
    },
    stages::registry::StageRegistry,
    store::MemoryStore,
};

#[test]
fn a_standby_runs_its_leaders_generation_under_the_leaders_number() {
    let (led, leader) = generations();
    assert_eq!(leader.standing_sent(), None, "no document is held yet");
    yaml(&leader, RANDOM).expect("a valid document");
    yaml(&leader, LEAST_LOAD).expect("a changed document");
    let told = sent(&leader);
    assert_eq!(told, leader.sent(2), "the standing one, by its number");
    // A generation that stands no longer is sent by its number alone.
    assert_eq!(leader.sent(1), worker_wire::generation_told(1));

    let (store, standby) = generations();
    followed(&standby, &told).expect("above the unbound one");
    assert_eq!(standing(&store), standing(&led));
    // The read call answers the leader's bytes, form and time.
    assert_eq!(
        wire::spec_document(standby.document(0).expect("held")),
        wire::spec_document(leader.document(0).expect("held"))
    );
    assert_eq!(standby.apply_counts().applied, 1);
    // Told again, it changes nothing.
    followed(&standby, &told).expect("the standing one again");
    // Nor does an event with no document, whatever its number.
    followed(&standby, &worker_wire::generation_told(9)).expect("no document");
    assert_eq!(standing(&store), standing(&led));
    assert_eq!(standby.apply_counts().applied, 1);
}

#[test]
fn a_replica_that_read_the_same_file_under_a_lower_number_takes_its_leaders() {
    let (_, leader) = generations();
    yaml(&leader, with_line(LEAST_LOAD, 5)).expect("a claim above none");
    let (store, standby) = generations();
    yaml(&standby, LEAST_LOAD).expect("a valid document");
    let (own, digest) = standing(&store);
    assert_eq!(own, 1);
    followed(&standby, &sent(&leader)).expect("the leader's number is above its own");
    assert_eq!(
        standing(&store),
        (5, digest),
        "one document, the leader's number"
    );
    assert_eq!(standby.document(0).expect("held").generation, 5);
}

#[test]
fn a_generation_that_is_not_above_the_standing_one_is_not_followed() {
    let (_, leader) = generations();
    yaml(&leader, RANDOM).expect("a valid document");
    let (store, standby) = generations();
    yaml(&standby, LEAST_LOAD).expect("a valid document");
    // Its own number over another document.
    let own = standing(&store);
    let refused = followed(&standby, &sent(&leader)).expect_err("another document");
    assert_eq!((refused.path, refused.why), (NUMBER, NOT_ABOVE));
    assert_eq!(standing(&store), own);
    // A number under its own, whatever the document.
    yaml(&standby, RANDOM).expect("a changed document");
    let own = standing(&store);
    assert_eq!(own.0, 2);
    let refused = followed(&standby, &sent(&leader)).expect_err("under its own");
    assert_eq!((refused.path, refused.why), (NUMBER, NOT_ABOVE));
    assert_eq!(standing(&store), own);
    assert_eq!(standby.document(0).expect("held").generation, 2);
    // Its own number over its own bytes, in another form.
    let mut told = sent(&standby);
    told.document.as_mut().expect("a document").format = "json".to_owned();
    let refused = followed(&standby, &told).expect_err("another form");
    assert_eq!((refused.path, refused.why), (NUMBER, NOT_ABOVE));
    assert_eq!(standing(&store), own);
}

#[test]
fn an_event_this_replica_cannot_run_is_refused_by_its_path() {
    let (store, standby) = generations();
    yaml(&standby, RANDOM).expect("a valid document");
    let own = standing(&store);
    let push = |document: &str, format: &str| {
        Some(proto::SpecPush {
            generation: 2,
            document: document.as_bytes().to_vec(),
            format: format.to_owned(),
            ..proto::SpecPush::default()
        })
    };
    let mut told = worker_wire::generation_told(2);
    // A stage this build does not hold.
    told.document = push(&LEAST_LOAD.replace("healthy", "no-such-stage"), "yaml");
    let refused = followed(&standby, &told).expect_err("a stage this build lacks");
    assert_eq!(refused.path, "generation.document");
    // A form it does not know, and an event that names no number.
    told.document = push(LEAST_LOAD, "toml");
    let refused = followed(&standby, &told).expect_err("no such form");
    assert_eq!(refused.path, "generation.document.format");
    let refused = followed(&standby, &proto::GenerationEvent::default());
    assert_eq!(refused.expect_err("no number").path, NUMBER);
    assert_eq!(standing(&store), own);
    assert_eq!(standby.apply_counts().applied, 1);
    // The same event with a document that runs is followed.
    told.document = push(LEAST_LOAD, "yaml");
    followed(&standby, &told).expect("above its own");
    assert_eq!(standing(&store).0, 2);
}

/// A standby does not follow a leader's generation whose bindings
/// differ on the overload switch. A leader of an older build took the
/// document. This build's pass refuses it, and the standby keeps its
/// own generation. A later generation that says one word is followed.
#[test]
fn a_standby_does_not_follow_a_generation_whose_bindings_differ_on_the_overload_switch() {
    // The older build's door: the stock pass, which takes the document.
    let led = Arc::new(MemoryStore::new());
    let stock_pass = AppliedSpec::validate_digested;
    let leader = Generations::validating(Arc::clone(&led), stock(), stock_pass, None);
    yaml(&leader, RANDOM).expect("a valid document");
    yaml(&leader, switched(Some(false), None)).expect("the older build takes it");
    assert_eq!(standing(&led).0, 2);

    let (store, standby) = generations();
    yaml(&standby, RANDOM).expect("a valid document");
    let own = standing(&store);
    let refused = followed(&standby, &sent(&leader)).expect_err("its bindings differ");
    assert_eq!(refused.path, "generation.document");
    assert_eq!(standing(&store), own, "the standby keeps its own");

    yaml(&leader, switched(Some(false), Some(false))).expect("every binding off");
    followed(&standby, &sent(&leader)).expect("above its own");
    assert_eq!(standing(&store).0, 3);
}

/// Held bytes of another generation are not the standing one's. Where
/// another writer's generation runs, an event at the standing number
/// over the bytes this door holds is no word of what stands: it is
/// refused as a number that is not above.
#[test]
fn an_event_over_the_held_bytes_of_another_generation_is_not_the_standing_one() {
    let (store, door) = generations();
    assert_eq!(yaml(&door, RANDOM), Ok(ApplyOutcome::Generation(1)));
    let mut next: PolicySpec = serde_yaml::from_str(LEAST_LOAD).expect("a document");
    next.generation = 2;
    store
        .apply(AppliedSpec::validate(&next).expect("a valid document"))
        .expect("above the standing one");
    // A leader whose generation 2 runs the bytes this door holds under 1.
    let (_, leader) = generations();
    yaml(&leader, LEAST_LOAD).expect("a valid document");
    yaml(&leader, RANDOM).expect("a changed document");
    followed(&door, &sent(&leader)).expect_err("not above the standing one");
    assert_eq!(standing(&store).0, 2);
}

/// The document the source handed last stood on its replica once it was
/// applied here or told by a leader, and at no other time: not while
/// it only passed, and not while a leader told another one.
#[test]
fn the_sources_document_stood_once_it_was_applied_or_told() {
    let (_, alone) = generations();
    assert!(alone.source_stood(), "no source handed one");
    yaml(&alone, RANDOM).expect("a valid document");
    assert!(alone.source_stood(), "applied here");

    let (_, door) = generations();
    door.defer(RANDOM.into(), Format::Yaml)
        .expect("a valid document");
    assert!(!door.source_stood(), "passed, never applied");
    // A leader tells another document, then the source's own.
    let (_, leader) = generations();
    yaml(&leader, LEAST_LOAD).expect("a valid document");
    followed(&door, &sent(&leader)).expect("above no generation");
    assert!(!door.source_stood(), "another document than the source's");
    yaml(&leader, RANDOM).expect("a changed document");
    followed(&door, &sent(&leader)).expect("above its own");
    assert!(door.source_stood(), "told while it followed");

    let (_, taken) = generations();
    taken
        .defer(RANDOM.into(), Format::Yaml)
        .expect("a valid document");
    taken.at_take();
    assert!(taken.source_stood(), "applied at its first take");

    // The source's later word replaces its start's: a file that moves
    // to the document its replica follows holds what stands.
    let (_, moved) = generations();
    moved
        .defer(LEAST_LOAD.into(), Format::Yaml)
        .expect("a valid document");
    followed(&moved, &sent(&leader)).expect("above no generation");
    assert!(!moved.source_stood(), "another document than the source's");
    assert!(matches!(
        yaml(&moved, RANDOM),
        Ok(ApplyOutcome::Unchanged(2))
    ));
    assert!(moved.source_stood(), "the source handed what stands");
}

/// What the door knows of a file that a replica that follows shows it,
/// by the order of the hand and of the generations. Shown ahead of the
/// generation that brings its document, the document stands once that
/// generation enters, and stays stood when its leader replaces it: a
/// mount that lags. Shown behind the generation that replaced it, it
/// has not stood since: a rollback. A file with no document never
/// stood. The door applies nothing of a hand and keeps no byte of it.
#[test]
fn a_shown_document_stood_by_the_order_of_its_hand_and_the_generations() {
    let (store, door) = generations();
    let (_, leader) = generations();
    door.shown(Some(RANDOM.as_bytes()), Format::Yaml);
    assert!(!door.source_stood(), "ahead of its generation");
    yaml(&leader, RANDOM).expect("a valid document");
    followed(&door, &sent(&leader)).expect("above no generation");
    assert!(door.source_stood(), "its generation entered");
    yaml(&leader, LEAST_LOAD).expect("a changed document");
    followed(&door, &sent(&leader)).expect("above its own");
    assert!(
        door.source_stood(),
        "replaced by its leader: a mount that lags"
    );
    door.shown(Some(RANDOM.as_bytes()), Format::Yaml);
    assert!(
        !door.source_stood(),
        "behind the generation that replaced it"
    );
    door.shown(Some(LEAST_LOAD.as_bytes()), Format::Yaml);
    assert!(door.source_stood(), "the document that stands");
    door.shown(None, Format::Yaml);
    assert!(!door.source_stood(), "a file no look could read");
    door.shown(Some(LEAST_LOAD.as_bytes()), Format::Yaml);
    assert!(door.source_stood());
    door.shown(Some(b"{ no policy"), Format::Yaml);
    assert!(!door.source_stood(), "a file that does not parse");
    assert_eq!(standing(&store).0, 2, "a hand applies nothing");
    assert_eq!(door.apply_counts(), applied(2, 0));
    assert_eq!(
        door.document(0).expect("held").document,
        LEAST_LOAD.as_bytes()
    );
}

/// A leader's generation holds the door's lock from its pass until it
/// lands: held behind the pass, and let go behind the entry. No apply
/// comes between the two. An event that changes nothing holds it the
/// same way.
#[test]
fn a_followed_generation_holds_the_doors_lock_from_its_pass_until_it_lands() {
    let (_, leader) = generations();
    yaml(&leader, RANDOM).expect("a valid document");
    let told = sent(&leader);
    let (store, standby) = generations();
    let locked = || standby.applying_lock().try_lock().is_none();

    let following = standby.following(&told).expect("above no generation");
    assert!(locked(), "held behind the pass");
    following.land(None).expect("it lands");
    assert!(!locked(), "let go behind the entry");
    assert_eq!(standing(&store).0, 1);

    let again = standby.following(&told).expect("the standing one again");
    assert!(locked(), "held by an event that changes nothing");
    again.land(None).expect("nothing to land");
    assert!(!locked());
    // An event the door refuses holds nothing behind its refusal.
    yaml(&standby, LEAST_LOAD).expect("its own second generation");
    assert!(standby.following(&told).is_err(), "under its own");
    assert!(!locked());
}

/// The hand of a replica that follows waits for the lock a generation
/// enters under, the one of the bytes held: a hand and an entry stand
/// in one order, whichever thread runs first.
#[test]
fn a_hand_waits_for_the_lock_a_generation_enters_under() {
    let (_, door) = generations();
    let entry = door.held_lock().lock();
    thread::scope(|scope| {
        let (done, lands) = mpsc::channel();
        let door = &door;
        scope.spawn(move || {
            door.shown(Some(RANDOM.as_bytes()), Format::Yaml);
            done.send(()).expect("the case waits");
        });
        let early = lands.recv_timeout(Duration::from_millis(200));
        assert!(early.is_err(), "the hand waits for the lock");
        assert!(door.source_stood(), "nothing is written ahead of it");
        drop(entry);
        lands
            .recv_timeout(Duration::from_secs(30))
            .expect("the hand lands behind the entry");
    });
    assert!(!door.source_stood());
}

/// A door with a spec store bound follows no document: no standby
/// follows beside a store, and the refusal names the document.
#[test]
fn a_door_beside_a_spec_store_follows_no_document() {
    let (_, leader) = generations();
    yaml(&leader, RANDOM).expect("a valid document");
    let (_dir, store, standby) = stored();
    let own = standing(&store);
    let refused = followed(&standby, &sent(&leader)).expect_err("a store is bound here");
    assert_eq!(refused.path, "generation.document");
    assert_eq!(
        refused.why,
        "a spec store is bound here, and no standby follows beside one"
    );
    assert_eq!(standing(&store), own);
}

/// A leader's generation this replica cannot follow is passed once: the
/// same event sent again at a retry is refused from the door's memory,
/// by the same words. Another generation is passed anew.
#[test]
fn a_refused_generation_sent_again_is_refused_from_memory() {
    thread_local! {
        static PASSES: Cell<u32> = const { Cell::new(0) };
    }
    fn counting(
        document: &PolicySpec,
        digest: Digest,
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<AppliedSpec, SpecError> {
        PASSES.with(|passes| passes.set(passes.get() + 1));
        faulting(document, digest, catalog, denied_headers)
    }
    let (_, leader) = generations();
    yaml(&leader, RANDOM).expect("a valid document");
    let store = Arc::new(MemoryStore::new());
    let standby = Generations::validating(Arc::clone(&store), stock(), counting, None);
    let told = sent(&leader);
    let first = followed(&standby, &told).expect_err("a fault of this build's validator");
    assert_eq!(first.path, "generation.document");
    assert_eq!(PASSES.with(Cell::get), 1);
    assert_eq!(followed(&standby, &told), Err(first));
    assert_eq!(PASSES.with(Cell::get), 1, "no second pass");
    yaml(&leader, LEAST_LOAD).expect("a changed document");
    followed(&standby, &sent(&leader)).expect("a document that runs");
    assert_eq!(PASSES.with(Cell::get), 2);
    assert_eq!(standing(&store).0, 2);
}
