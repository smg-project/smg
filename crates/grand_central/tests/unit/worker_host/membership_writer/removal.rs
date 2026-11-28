//! A removal by each asker: the row it saw and no other, the host's own
//! source under any key, and health inside its scope. A drain leaves the
//! row's maker as it was.

use super::*;

#[test]
fn a_removal_takes_only_the_row_its_asker_saw() {
    check_guard(local_door).expect("both registrations stand in turn");
}

pub(crate) fn check_guard(
    open: impl FnOnce() -> (Arc<MemoryStore>, Box<dyn Rows>),
) -> Result<(), &'static str> {
    let (store, door) = open();
    let maker = Registrant::Control;
    let by = Remover::Registrant(&maker);
    assert!(door.found_one(&maker, URL, &|_| stated()).is_ok());
    let first = store.by_key(URL).ok_or("the first row")?;
    assert!(door
        .found_one(&maker, URL, &|_| WorkerFacts {
            role: Role::Decode,
            ..stated()
        })
        .is_ok());
    let replacement = store.by_key(URL).ok_or("the replacement row")?;
    assert_ne!(first.id, replacement.id);
    assert_eq!(first.revision, replacement.revision);
    let epoch = store.epoch();
    let wrong_revision = RowRevision {
        revision: replacement.revision + 1,
        ..standing(&replacement)
    };
    for seen in [wrong_revision, standing(&first)] {
        assert_eq!(
            door.drained(by, URL, seen),
            Err(MembershipWriteRefused::Moved)
        );
        assert_eq!(door.gone(by, URL, seen), Err(MembershipWriteRefused::Moved));
        assert_eq!(
            door.gone(Remover::Provider("other"), URL, seen),
            Err(MembershipWriteRefused::Moved)
        );
    }
    assert_eq!(store.by_key(URL), Some(replacement));
    assert_eq!(store.epoch(), epoch);
    Ok(())
}

#[test]
fn the_hosts_own_source_removes_its_rows_under_any_key() {
    check_provider(local_door).expect("the makers and their rows");
}

pub(crate) fn check_provider(
    open: impl Fn() -> (Arc<MemoryStore>, Box<dyn Rows>),
) -> Result<(), &'static str> {
    let own = Registrant::provider("pool", "private/key").map_err(|_| "the source's maker")?;
    let another =
        Registrant::provider("other", "private/key").map_err(|_| "another source's maker")?;
    for (maker, refusal) in [
        (own, None),
        (
            Registrant::Control,
            Some("the row was made by a control call: its maker alone changes it"),
        ),
        (
            another,
            Some("the row is provider other's: its worker is changed at the provider's source"),
        ),
    ] {
        let (store, door) = open();
        assert!(door.found_one(&maker, URL, &|_| stated()).is_ok());
        let row = store.by_key(URL).ok_or("the registered row")?;
        let by = Remover::Provider("pool");
        let seen = standing(&row);
        if let Some(refusal) = refusal {
            let expected = Err(MembershipWriteRefused::NotRegistrant(refusal.to_owned()));
            assert_eq!(door.drained(by, URL, seen), expected);
            assert_eq!(door.gone(by, URL, seen), expected);
            assert_eq!(store.by_key(URL), Some(row));
        } else {
            assert_eq!(
                door.drained(by, URL, seen),
                Ok(MembershipWriteOutcome::Draining(row.id))
            );
            assert_eq!(
                door.gone(by, URL, seen),
                Ok(MembershipWriteOutcome::Removed(row.id))
            );
            assert_eq!(store.by_key(URL), None);
        }
    }
    Ok(())
}

#[test]
fn health_removes_inside_its_scope_alone() {
    check_health_scope(local_door).expect("the makers and their rows");
}

pub(crate) fn check_health_scope(
    open: impl Fn() -> (Arc<MemoryStore>, Box<dyn Rows>),
) -> Result<(), &'static str> {
    let provider = Registrant::provider("pool", "private/key").map_err(|_| "the source's maker")?;
    for (scope, control, discovered) in [
        (UnhealthyRemovalScope::Disabled, false, false),
        (UnhealthyRemovalScope::DiscoveryOwned, false, true),
        (UnhealthyRemovalScope::AllLocal, true, true),
    ] {
        for (maker, allowed) in [(&Registrant::Control, control), (&provider, discovered)] {
            let (store, door) = open();
            assert!(door.found_one(maker, URL, &|_| stated()).is_ok());
            let row = store.by_key(URL).ok_or("the registered row")?;
            let by = Remover::Health(scope);
            let seen = standing(&row);
            if allowed {
                assert_eq!(
                    door.drained(by, URL, seen),
                    Ok(MembershipWriteOutcome::Draining(row.id))
                );
                assert_eq!(
                    door.gone(by, URL, seen),
                    Ok(MembershipWriteOutcome::Removed(row.id))
                );
                assert_eq!(store.by_key(URL), None);
            } else {
                assert_eq!(
                    door.drained(by, URL, seen),
                    Err(MembershipWriteRefused::OutOfScope)
                );
                assert_eq!(
                    door.gone(by, URL, seen),
                    Err(MembershipWriteRefused::OutOfScope)
                );
                assert_eq!(store.by_key(URL), Some(row));
            }
        }
    }
    Ok(())
}

#[test]
fn a_removal_leaves_the_rows_maker_as_it_was() {
    check_maker(local_door).expect("the maker and its row");
}

pub(crate) fn check_maker(
    open: impl FnOnce() -> (Arc<MemoryStore>, Box<dyn Rows>),
) -> Result<(), &'static str> {
    let (store, door) = open();
    let maker = Registrant::provider("pool", "private/key").map_err(|_| "the source's maker")?;
    assert!(door.found_one(&maker, URL, &|_| stated()).is_ok());
    let row = store.by_key(URL).ok_or("the registered row")?;
    assert_eq!(
        door.drained(Remover::Provider("pool"), URL, standing(&row)),
        Ok(MembershipWriteOutcome::Draining(row.id))
    );
    assert_eq!(
        store.by_key(URL),
        Some(WorkerRow {
            status: WorkerStatus::Draining,
            ..row
        })
    );
    Ok(())
}
