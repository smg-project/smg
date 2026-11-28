//! A start's document: the one source a door takes, a document deferred
//! to its replica's first take, and what an unbound start waits for.

use std::{
    cell::{Cell, RefCell},
    sync::Arc,
    time::{Duration, Instant},
};

use super::{
    super::Generations, applied, candidate_under, election, followed, generations,
    generations_under, sent, stored, yaml,
};
use crate::{
    document::{
        applied::AppliedSpec,
        operator::{
            test_support::{standing, stock, LEAST_LOAD, RANDOM},
            ApplyOutcome, DocumentManager, Format, SpecRejected,
        },
        spec::{Digest, PolicySpec, SpecError},
    },
    service::controller::spec_store::{SpecStore, StoreRefused},
    stages::registry::StageRegistry,
    store::MemoryStore,
};

#[test]
fn a_door_takes_one_source() {
    let (_dir, _store, generations) = stored();
    // A second store, on a directory of its own: the door holds its
    // first one's.
    let other = tempfile::tempdir().expect("a temp dir");
    let open = || SpecStore::open(other.path().to_owned()).expect("the directory opens");
    assert_eq!(
        generations.bind_store(open()),
        Err(StoreRefused::SecondSource)
    );

    // A file's document applied first is a source too.
    let (_store, filed) = self::generations();
    assert_eq!(yaml(&filed, LEAST_LOAD), Ok(ApplyOutcome::Generation(1)));
    assert_eq!(filed.bind_store(open()), Err(StoreRefused::SecondSource));
    assert!(!filed.writable());

    // So is a file's document that waits for its replica's first take.
    let (_store, deferred) = self::generations();
    deferred
        .defer(LEAST_LOAD.into(), Format::Yaml)
        .expect("a valid document");
    assert!(deferred.bound(), "a source, with nothing applied yet");
    assert_eq!(deferred.bind_store(open()), Err(StoreRefused::SecondSource));

    // And a file's document is not applied beside a store.
    assert_eq!(
        yaml(&generations, LEAST_LOAD),
        Err(SpecRejected::SecondSource)
    );
    assert_eq!(generations.apply_counts(), applied(0, 0));
    // Nor is one passed there for a later take.
    assert_eq!(
        generations.defer(LEAST_LOAD.into(), Format::Yaml),
        Err(SpecRejected::SecondSource)
    );
    assert!(generations.deferred.lock().is_none(), "nothing is kept");
}

#[test]
fn a_deferred_document_runs_at_the_first_take_and_at_no_later_one() {
    let (store, door) = generations();
    door.defer(RANDOM.into(), Format::Yaml)
        .expect("a valid document");
    assert_eq!(standing(&store), (0, None), "passed, not applied");
    assert_eq!(door.apply_counts().applied, 0);
    door.at_take();
    assert_eq!(standing(&store).0, 1);
    assert_eq!(door.document(0).expect("held").document, RANDOM.as_bytes());
    assert!(
        door.deferred.lock().is_none(),
        "the start's bytes are let go"
    );
    // A later take applies nothing: what was applied since stands.
    yaml(&door, LEAST_LOAD).expect("a changed document");
    door.at_take();
    assert_eq!(standing(&store).0, 2);
    assert_eq!(door.apply_counts().applied, 2);
}

#[test]
fn a_deferred_document_that_does_not_run_refuses_the_start() {
    let (store, door) = generations();
    let unrun = LEAST_LOAD.replace("healthy", "no-such-stage");
    door.defer(unrun.into_bytes(), Format::Yaml)
        .expect_err("a stage this build lacks");
    // A refused start waits for no document: the store takes no mark.
    assert!(
        !store.snapshot().spec().awaits_document(),
        "a refused start marked the store"
    );
    // Nothing is kept for a take, and the door has no source.
    assert!(!door.bound());
    door.at_take();
    assert_eq!(standing(&store), (0, None));
    let counts = door.apply_counts();
    assert_eq!((counts.applied, counts.rejected), (0, 0));
}

#[test]
fn a_replica_that_deferred_its_document_follows_a_leader_under_any_number() {
    // The leader runs its first generation over another document: a
    // replica that had applied its own at its start could not follow.
    let (_, leader) = generations();
    yaml(&leader, RANDOM).expect("a valid document");
    let (store, standby) = generations();
    standby
        .defer(LEAST_LOAD.into(), Format::Yaml)
        .expect("a valid document");
    followed(&standby, &sent(&leader)).expect("above no generation");
    assert_eq!(standing(&store).0, 1);
    assert_eq!(
        standby.document(0).expect("held").document,
        RANDOM.as_bytes()
    );
    // Promoted behind a followed generation, the start's bytes are not
    // applied: they are as old as the start, and the file's watch reads
    // the file as it stands then.
    standby.at_take();
    assert_eq!(standing(&store).0, 1);
    assert_eq!(standby.apply_counts().applied, 1);
    assert_eq!(
        standby.document(0).expect("held").document,
        RANDOM.as_bytes()
    );
}

/// A replica's source applies its document through the replica. The
/// check its caller hands in is asked ahead of the pass and behind it,
/// and the pass's notice runs between the two: at the second asking the
/// document the source handed has passed and has not stood. It stands
/// behind the apply.
#[test]
fn a_sources_apply_tells_its_pass_between_its_callers_two_checks() {
    let (_, replica) = generations();
    let admissions = Cell::new(0);
    let applied = replica.apply_where(
        replica.lead(),
        LEAST_LOAD.into(),
        Format::Yaml,
        true,
        |_| {
            assert_eq!(
                replica.source_stood(),
                admissions.get() == 0,
                "nothing handed at the first asking, passed and not stood at the second"
            );
            admissions.set(admissions.get() + 1);
            None
        },
    );
    assert_eq!(applied, Ok(ApplyOutcome::Generation(1)));
    assert_eq!(admissions.get(), 2, "asked ahead of the pass and behind it");
    assert!(replica.source_stood(), "the source's document stands");
}

/// A host can publish and serve its start's document with no replica
/// beside its door: the door applies it, holds it and answers it.
#[test]
fn a_local_start_serves_its_document_with_no_replica() {
    let door = DocumentManager::new(Arc::new(MemoryStore::new()), stock(), Vec::new());
    assert_eq!(
        door.apply(RANDOM.into(), Format::Yaml),
        Ok(ApplyOutcome::Generation(1))
    );
    let held = door.document(0).expect("the start's document");
    let status = door.status().expect("the start's status");
    assert_eq!(
        (held.generation, held.document.as_slice()),
        (1, RANDOM.as_bytes())
    );
    assert_eq!((status.generation, status.digest), (1, Some(held.digest)));
}

/// A start that defers its document passes it under the door's lock,
/// as an apply does. It lets the lock go behind the start. So no second
/// source comes between its pass and its mark.
#[test]
fn a_deferred_start_passes_its_document_under_the_doors_lock() {
    thread_local! {
        static DOOR: RefCell<Option<Arc<Generations>>> = const { RefCell::new(None) };
        static LOCKED: Cell<Option<bool>> = const { Cell::new(None) };
    }
    fn looking(
        document: &PolicySpec,
        digest: Digest,
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<AppliedSpec, SpecError> {
        let locked = DOOR.with_borrow(|door| {
            let door = door.as_ref().expect("the case's door");
            door.applying_lock().try_lock().is_none()
        });
        LOCKED.set(Some(locked));
        AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
    }
    let store = Arc::new(MemoryStore::new());
    let door = Arc::new(Generations::validating(store, stock(), looking, None));
    DOOR.set(Some(Arc::clone(&door)));
    let deferred = door.defer(RANDOM.into(), Format::Yaml);
    DOOR.set(None);
    deferred.expect("a valid document");
    assert_eq!(LOCKED.get(), Some(true), "the pass ran under the lock");
    assert!(
        door.applying_lock().try_lock().is_some(),
        "let go behind the start"
    );
}

/// The unbound start waits for a document a start deferred, and for no
/// other: a controller with no source and a replica alone that applied
/// its file carry no mark. The mark ends at the first publication, by a
/// take's road or by a followed generation's.
#[test]
fn the_unbound_start_awaits_a_deferred_document_alone() {
    let awaits = |store: &MemoryStore| store.snapshot().spec().awaits_document();
    let (store, alone) = generations();
    assert!(!awaits(&store), "no source");
    yaml(&alone, RANDOM).expect("a valid document");
    assert!(!awaits(&store), "a replica alone applied its file");

    let (store, door) = generations();
    door.defer(RANDOM.into(), Format::Yaml)
        .expect("a valid document");
    assert!(awaits(&store), "the start's document waits");
    door.at_take();
    assert_eq!(standing(&store).0, 1);
    assert!(!awaits(&store), "it landed at the take");

    let (_, leader) = generations();
    yaml(&leader, LEAST_LOAD).expect("a valid document");
    let (store, standby) = generations();
    standby
        .defer(RANDOM.into(), Format::Yaml)
        .expect("a valid document");
    assert!(awaits(&store));
    followed(&standby, &sent(&leader)).expect("above no generation");
    assert!(!awaits(&store), "a followed generation stands");
    // A start that passes its document behind a generation it followed
    // marks nothing: a generation stands.
    let (store, late) = generations();
    followed(&late, &sent(&leader)).expect("above no generation");
    late.defer(RANDOM.into(), Format::Yaml)
        .expect("a valid document");
    assert!(!awaits(&store), "no mark on a generation that stands");
}

/// The watch's sleep where the start's document waits: no longer than
/// to the opening of a lead that is still ahead, no time once the fence
/// shows a term, and the whole of it where nothing waits.
#[test]
fn the_watch_sleeps_no_longer_than_to_the_opening_a_starts_document_waits_for() {
    let most = Duration::from_secs(3_600);
    let deadline = Duration::from_secs(600);
    // A replica alone has no fence, and no opening to wait for.
    let (_, alone) = generations();
    assert_eq!(alone.nap(most), most);

    // A take with no holder to wait for opens one renew deadline
    // later, and its first renewal says when.
    let mut elector = candidate_under(deadline);
    let (store, door) = generations_under(elector.fence());
    assert_eq!(door.nap(most), most, "no document waits");
    door.defer(RANDOM.into(), Format::Yaml)
        .expect("a valid document");
    assert_eq!(door.nap(most), most, "no lead is in sight");
    elector.tick(Instant::now()).expect("a take");
    elector.tick(Instant::now()).expect("a renewal");
    let nap = door.nap(most);
    assert!(
        nap <= deadline && deadline - nap < Duration::from_secs(60),
        "to the opening: {nap:?}"
    );
    assert_eq!(standing(&store), (0, None), "the lead is not open");

    // Under an open fence it is no time, and the whole once the
    // document stands.
    let mut elector = election();
    let (store, door) = generations_under(elector.fence());
    door.defer(RANDOM.into(), Format::Yaml)
        .expect("a valid document");
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(door.nap(most), Duration::ZERO);
    door.deferred_now();
    assert_eq!(standing(&store).0, 1);
    assert_eq!(door.nap(most), most);
}

/// A start's document that its take's pass refuses is counted and let
/// go, and the door keeps its source: the spec calls are served on.
#[test]
fn a_starts_document_refused_at_its_take_leaves_the_door_its_source() {
    thread_local! {
        static PASSES: Cell<u32> = const { Cell::new(0) };
    }
    // The start's pass runs, and the take's faults.
    fn faulting_again(
        document: &PolicySpec,
        digest: Digest,
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<AppliedSpec, SpecError> {
        let passed = PASSES.with(|passes| passes.replace(passes.get() + 1));
        assert_eq!(passed, 0, "a fault of the validator");
        AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
    }
    let store = Arc::new(MemoryStore::new());
    let door = Generations::validating(Arc::clone(&store), stock(), faulting_again, None);
    door.defer(RANDOM.into(), Format::Yaml)
        .expect("a valid document");
    door.at_take();
    assert_eq!(standing(&store), (0, None));
    assert_eq!(door.apply_counts(), applied(0, 1));
    assert!(door.deferred.lock().is_none(), "the bytes are let go");
    assert!(door.bound(), "a source all the same");
}
