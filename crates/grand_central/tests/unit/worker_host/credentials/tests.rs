use std::{sync::mpsc, thread, time::Duration};

use super::{Credentials, KeptCredential, RegistrationInFlight};
use crate::{engine::view::WorkerId, store::WorkerFacts};

const ADDRESS: &str = "http://a:8000";
const OTHER_ADDRESS: &str = "http://b:8000";
const KEY_A: &str = "key-a";
const KEY_B: &str = "key-b";
const KEY_C: &str = "key-c";
const ROW: WorkerId = WorkerId(1);
const OTHER_ROW: WorkerId = WorkerId(2);

/// The facts of a row that dials `address`.
fn facts(address: &str) -> WorkerFacts {
    WorkerFacts {
        key: address.to_owned(),
        ..WorkerFacts::default()
    }
}

/// A registration's keep at an address where none is in flight.
fn keep(credentials: &Credentials, address: &str, credential: Option<&str>) -> KeptCredential {
    credentials
        .keep(&facts(address), credential.map(str::to_owned))
        .expect("no registration in flight at the address")
}

/// A registration whose write was taken, with the row its door
/// answered. Return whether the address is to be laid again.
fn register(
    credentials: &Credentials,
    address: &str,
    row: WorkerId,
    credential: Option<&str>,
) -> bool {
    keep(credentials, address, credential).accepted(row)
}

/// Whether the object of `row`, built at the address now, takes
/// `credential`. The read marks a flight, as a build does. No failure
/// prints a credential.
fn builds_with(
    credentials: &Credentials,
    address: &str,
    row: WorkerId,
    credential: Option<&str>,
) -> bool {
    credentials.of(address, row).as_deref() == credential
}

/// A forgetting whose read of the store begins now and finds no row.
fn forget(credentials: &Credentials) {
    credentials.forget_left(credentials.taken(), |_| false);
}

fn has_entry(credentials: &Credentials, address: &str) -> bool {
    credentials.held.read().entries.contains_key(address)
}

/// A registration of several rows at one address is one flight. Its
/// acceptance binds every row its door answered, and the object of
/// each takes the credential.
#[test]
fn an_acceptance_of_several_rows_binds_each_of_them() {
    let credentials = Credentials::new();
    let rows = [ROW, OTHER_ROW];
    assert!(keep(&credentials, ADDRESS, Some(KEY_A)).accepted_rows(&rows));
    for row in rows {
        assert!(builds_with(&credentials, ADDRESS, row, Some(KEY_A)));
    }
    // Stated again with no change, nothing is to be laid again.
    assert!(!keep(&credentials, ADDRESS, Some(KEY_A)).accepted_rows(&rows));
    // A row beside them that no registration bound takes nothing.
    assert!(builds_with(&credentials, ADDRESS, WorkerId(3), None));
}

#[test]
fn neither_the_keeper_nor_a_token_prints_an_address_or_a_credential() {
    let credentials = Credentials::new();
    // An address as a registration writes it can carry a userinfo.
    let token = keep(&credentials, "http://pebble:cedar@a:8000", Some(KEY_A));
    for text in [
        format!("{token:?}"),
        format!("{token:#?}"),
        format!("{credentials:?}"),
        format!("{credentials:#?}"),
    ] {
        for secret in ["pebble", "cedar", KEY_A] {
            assert!(!text.contains(secret), "{text}");
        }
    }
    assert!(!token.put_back());
}

#[test]
fn the_keeper_holds_a_credential_at_the_address_the_rows_object_dials() {
    let credentials = Credentials::new();
    // A row that names an engine dials it. One that names none dials
    // its key.
    let named = WorkerFacts {
        engine: ADDRESS.to_owned(),
        ..facts("http://a:8000@0")
    };
    let token = credentials
        .keep(&named, Some(KEY_A.to_owned()))
        .expect("no registration in flight at the address");
    assert!(token.accepted(ROW));
    assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));
    assert!(!has_entry(&credentials, "http://a:8000@0"));
}

#[test]
fn an_acceptance_says_whether_the_address_is_to_be_laid_again() {
    let credentials = Credentials::new();
    // A row bound for the first time, with or without a credential.
    assert!(register(&credentials, ADDRESS, ROW, None));
    assert!(register(
        &credentials,
        OTHER_ADDRESS,
        OTHER_ROW,
        Some(KEY_A)
    ));

    // The same row again: only where the credential changed.
    assert!(!register(&credentials, ADDRESS, ROW, None));
    assert!(register(&credentials, ADDRESS, ROW, Some(KEY_A)));
    assert!(!register(&credentials, ADDRESS, ROW, Some(KEY_A)));
    assert!(register(&credentials, ADDRESS, ROW, Some(KEY_B)));
    assert!(register(&credentials, ADDRESS, ROW, None));

    // Another row of the address is bound beside the first.
    assert!(register(&credentials, ADDRESS, OTHER_ROW, None));
    assert!(!register(&credentials, ADDRESS, ROW, None));

    // An object built while the registration was in flight.
    let token = keep(&credentials, ADDRESS, None);
    assert!(builds_with(&credentials, ADDRESS, ROW, None));
    assert!(token.accepted(ROW));

    // A put-back restores what stood: the next keep is held against it.
    assert!(register(&credentials, ADDRESS, ROW, Some(KEY_B)));
    assert!(!keep(&credentials, ADDRESS, None).put_back());
    assert!(!register(&credentials, ADDRESS, ROW, Some(KEY_B)));
}

#[test]
fn a_row_no_registration_here_stated_takes_no_credential() {
    let credentials = Credentials::new();
    assert!(register(&credentials, ADDRESS, ROW, Some(KEY_A)));
    assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));
    assert!(builds_with(&credentials, ADDRESS, OTHER_ROW, None));
    assert!(builds_with(&credentials, OTHER_ADDRESS, ROW, None));

    // A registration in flight states its credential for the address:
    // its row is not answered yet. Its answer binds its row alone.
    let flight = keep(&credentials, ADDRESS, Some(KEY_B));
    assert!(builds_with(&credentials, ADDRESS, OTHER_ROW, Some(KEY_B)));
    assert!(flight.accepted(ROW));
    assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_B)));
    assert!(builds_with(&credentials, ADDRESS, OTHER_ROW, None));
}

#[test]
fn a_registration_taken_leaves_its_credential_standing() {
    let credentials = Credentials::new();
    assert!(credentials.is_empty());
    register(&credentials, ADDRESS, ROW, Some(KEY_A));
    assert_eq!(credentials.len(), 1);
    assert!(!credentials.is_empty());

    register(&credentials, ADDRESS, ROW, Some(KEY_B));
    register(&credentials, OTHER_ADDRESS, OTHER_ROW, None);
    assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_B)));
    assert_eq!(credentials.len(), 1);

    register(&credentials, ADDRESS, ROW, None);
    assert!(credentials.is_empty());
    assert!(builds_with(&credentials, ADDRESS, ROW, None));
    forget(&credentials);
    assert!(!has_entry(&credentials, ADDRESS));
    assert!(!has_entry(&credentials, OTHER_ADDRESS));
}

#[test]
fn the_keeper_prints_how_many_it_keeps_and_no_credential() {
    let credentials = Credentials::new();
    register(&credentials, ADDRESS, ROW, Some(KEY_A));
    let shown = format!("{credentials:?}");
    assert!(!shown.contains(KEY_A));
    assert_eq!(shown, "Credentials { held: 1 }");
}

#[test]
fn a_keep_at_an_address_in_flight_takes_nothing() {
    let credentials = Credentials::new();
    let first = keep(&credentials, ADDRESS, Some(KEY_A));
    let second = credentials.keep(&facts(ADDRESS), Some(KEY_B.to_owned()));
    assert_eq!(second.err(), Some(RegistrationInFlight));
    assert_eq!(
        RegistrationInFlight.to_string(),
        "a registration is in flight at the address"
    );
    assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));
    // Another address is its own turn.
    register(&credentials, OTHER_ADDRESS, OTHER_ROW, Some(KEY_C));

    assert!(first.accepted(ROW));
    register(&credentials, ADDRESS, ROW, Some(KEY_B));
    assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_B)));
}

#[test]
fn two_refused_registrations_at_once_leave_what_stood() {
    let credentials = Credentials::new();
    register(&credentials, ADDRESS, ROW, Some(KEY_A));
    assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));

    let first = keep(&credentials, ADDRESS, Some(KEY_B));
    let second = credentials.keep(&facts(ADDRESS), Some(KEY_C.to_owned()));
    assert!(second.is_err());
    assert!(!first.put_back());
    // The second asks again and is refused too.
    assert!(!keep(&credentials, ADDRESS, Some(KEY_C)).put_back());
    assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));
    assert_eq!(credentials.len(), 1);
}

#[test]
fn a_put_back_says_that_an_object_was_built_with_the_refused_credential() {
    let credentials = Credentials::new();
    register(&credentials, ADDRESS, ROW, Some(KEY_A));

    let refused = keep(&credentials, ADDRESS, Some(KEY_B));
    assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_B)));
    assert!(refused.put_back());
    // The next flight starts with no object built in it.
    assert!(!keep(&credentials, ADDRESS, Some(KEY_C)).put_back());
    assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));
}

#[test]
fn a_put_back_of_a_first_registration_leaves_no_entry() {
    let credentials = Credentials::new();
    assert!(!keep(&credentials, ADDRESS, Some(KEY_A)).put_back());
    assert!(!has_entry(&credentials, ADDRESS));

    let refused = keep(&credentials, ADDRESS, Some(KEY_A));
    assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));
    assert!(refused.put_back());
    assert!(!has_entry(&credentials, ADDRESS));
    assert!(credentials.is_empty());
}

#[test]
fn a_registration_that_states_none_can_be_put_back() {
    let credentials = Credentials::new();
    register(&credentials, ADDRESS, ROW, Some(KEY_A));

    let refused = keep(&credentials, ADDRESS, None);
    // What it displaced is still held.
    assert_eq!(credentials.len(), 1);
    assert!(builds_with(&credentials, ADDRESS, ROW, None));
    assert!(refused.put_back());
    assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));
}

#[test]
fn a_token_dropped_with_no_answer_is_put_back() {
    let credentials = Credentials::new();
    register(&credentials, ADDRESS, ROW, Some(KEY_A));

    drop(keep(&credentials, ADDRESS, Some(KEY_B)));
    assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));
    // The address takes the next registration.
    register(&credentials, ADDRESS, ROW, Some(KEY_B));
    assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_B)));
}

#[test]
fn a_forgetting_drops_the_rows_that_left_and_an_entry_with_its_last() {
    let credentials = Credentials::new();
    register(&credentials, ADDRESS, ROW, Some(KEY_A));
    register(&credentials, ADDRESS, OTHER_ROW, Some(KEY_A));
    let read = credentials.taken();

    // One row left: the other keeps the address's credential.
    credentials.forget_left(read, |row| row == OTHER_ROW);
    assert!(builds_with(&credentials, ADDRESS, ROW, None));
    assert!(builds_with(&credentials, ADDRESS, OTHER_ROW, Some(KEY_A)));
    assert_eq!(credentials.len(), 1);

    // The last row left: the entry goes.
    credentials.forget_left(read, |_| false);
    assert!(!has_entry(&credentials, ADDRESS));
    assert!(credentials.is_empty());
}

#[test]
fn a_forgetting_keeps_a_row_bound_behind_its_read() {
    let credentials = Credentials::new();
    let read = credentials.taken();
    // Its row may have been written behind the holder's read.
    register(&credentials, ADDRESS, ROW, Some(KEY_A));
    credentials.forget_left(read, |_| false);
    assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));

    // A put-back leaves the binding as it was taken.
    assert!(!keep(&credentials, ADDRESS, Some(KEY_B)).put_back());
    credentials.forget_left(read, |_| false);
    assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));

    // A read that begins behind the answer forgets it.
    forget(&credentials);
    assert!(!has_entry(&credentials, ADDRESS));
}

#[test]
fn a_forgetting_in_flight_drops_the_rows_and_keeps_what_was_stated() {
    let credentials = Credentials::new();
    register(&credentials, ADDRESS, ROW, Some(KEY_A));

    // The row left while the next registration is in flight.
    let next = keep(&credentials, ADDRESS, Some(KEY_B));
    forget(&credentials);
    assert_eq!(credentials.len(), 1);
    assert!(next.accepted(OTHER_ROW));
    assert!(builds_with(&credentials, ADDRESS, OTHER_ROW, Some(KEY_B)));
    assert!(builds_with(&credentials, ADDRESS, ROW, None));

    // Refused instead: no row is bound any more, so nothing is left.
    let refused = keep(&credentials, ADDRESS, Some(KEY_C));
    forget(&credentials);
    assert!(!refused.put_back());
    assert!(!has_entry(&credentials, ADDRESS));
}

#[test]
fn a_build_reads_beside_another_reader_of_the_keeper() {
    let credentials = Credentials::new();
    register(&credentials, ADDRESS, ROW, Some(KEY_A));

    let keeper = &credentials;
    let held = keeper.held.read();
    let (send, receive) = mpsc::channel();
    thread::scope(|scope| {
        scope.spawn(move || {
            let _ = send.send(keeper.of(ADDRESS, ROW).as_deref() == Some(KEY_A));
        });
        let read = receive.recv_timeout(Duration::from_secs(5));
        drop(held);
        assert_eq!(read, Ok(true));
    });
}

#[test]
fn the_token_of_a_keep_prints_no_credential() {
    let credentials = Credentials::new();
    register(&credentials, ADDRESS, ROW, Some(KEY_A));
    let kept = keep(&credentials, ADDRESS, Some(KEY_B));

    let shown = format!("{kept:?}");
    assert!(!shown.contains(KEY_A));
    assert!(!shown.contains(KEY_B));
    assert_eq!(shown, "KeptCredential { .. }");
    assert!(kept.accepted(ROW));
}
