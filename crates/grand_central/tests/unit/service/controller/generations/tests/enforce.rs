//! The write calls: an enforce on file under the number it runs under,
//! and the turn and the places the write calls take.

use std::{
    collections::BTreeSet,
    sync::Arc,
    time::{Duration, Instant},
};

use tokio::task::JoinSet;

use super::{
    super::{Generations, Unanswered, DIGEST_TEXT_LEN, WRITES_WAITING},
    applied, stored,
};
use crate::{
    document::operator::{
        test_support::{bound_to, needs_rows, padded, standing, LEAST_LOAD, RANDOM},
        EnforceRefused, Format, SpecRejected, MAX_DOCUMENT_BYTES,
    },
    service::{
        controller::{
            spec_store::tests::{name_of, names},
            wire,
        },
        proto,
    },
};

/// An enforce under a spec store: the document is on file under the
/// number it runs under, as its caller sent it, and the read call
/// answers those bytes.
#[test]
fn an_enforce_is_persisted_and_runs_as_the_next_generation() {
    let (dir, store, generations) = stored();
    assert!(generations.writable() && generations.bound());
    assert_eq!(standing(&store), (0, None));

    let first = generations
        .enforce("", LEAST_LOAD.into(), Format::Yaml)
        .expect("no guard");
    let (number, digest) = first.ran.expect("it runs");
    assert_eq!((number, first.standing), (1, (0, None)));
    assert_eq!(standing(&store), (1, Some(digest)));
    assert_eq!(names(dir.path()), BTreeSet::from([name_of(1, "yaml")]));
    let on_file = std::fs::read(dir.path().join(name_of(1, "yaml"))).expect("the file");
    assert_eq!(on_file, LEAST_LOAD.as_bytes());
    let read = generations.document(0).expect("the standing document");
    assert_eq!(
        (read.generation, read.document.as_slice()),
        (1, LEAST_LOAD.as_bytes())
    );
    // The feed before and after: an enforce answers the diff a dry run
    // does.
    let needs = first.needs.map(wire::needs_diff).expect("the diff");
    assert_eq!(needs.before, Some(proto::NeedsSummary::default()));
    let runs = wire::needs_summary(needs_rows(store.snapshot().spec().engine()));
    assert_eq!(needs.after, Some(runs));

    // A JSON document is on file under its own form.
    let json = r#"{"apiVersion":"smg.policy/v1","profiles":{"default":{"pick":"random"}}}"#;
    let second = generations
        .enforce("", json.into(), Format::Json)
        .expect("no guard");
    assert_eq!(second.ran.map(|(number, _)| number), Ok(2));
    let both = BTreeSet::from([name_of(1, "yaml"), name_of(2, "json")]);
    assert_eq!(names(dir.path()), both);
    assert_eq!(generations.apply_counts(), applied(2, 0));
}

/// The bound of what waits: no more write calls are let in than the
/// one that applies and the ones behind it.
#[tokio::test]
async fn a_write_call_past_the_waiting_ones_is_refused() {
    let (_dir, store, generations) = stored();
    let taken = generations.write_turn.fill();
    let places = taken.len();
    assert_eq!(places, WRITES_WAITING + 1, "every place is free");
    let enforce = || generations.enforce_in_turn(String::new(), LEAST_LOAD.into(), Format::Yaml);

    let refused = enforce().await;
    assert!(matches!(refused, Err(Unanswered::Waiting)), "{refused:?}");
    // A rollback waits in the same turn.
    let refused = generations.roll_back_in_turn(1).await;
    assert!(matches!(refused, Err(Unanswered::Waiting)), "{refused:?}");
    assert_eq!(standing(&store), (0, None));
    drop(taken);
    // A call gives its place back at its end: as many calls again as
    // there are places each find one.
    for _ in 0..=places {
        let ran = enforce().await.expect("a place is free").ran;
        assert_eq!(ran.map(|(number, _)| number), Ok(1));
    }
}

/// A document over the bound and a guard longer than a digest's text
/// take no place: each is refused though every place is taken.
#[tokio::test]
async fn a_document_over_the_bound_and_a_long_guard_wait_on_no_apply() {
    let (dir, _store, generations) = stored();
    let taken = generations.write_turn.fill();
    assert_eq!(taken.len(), WRITES_WAITING + 1, "every place is free");
    let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");

    let over = generations
        .enforce_in_turn(String::new(), padded(LEAST_LOAD, bound + 1), Format::Yaml)
        .await
        .expect("refused by its size");
    assert_eq!(
        over.ran,
        Err(SpecRejected::TooLarge(MAX_DOCUMENT_BYTES + 1))
    );
    let long = "x".repeat(DIGEST_TEXT_LEN + 1);
    let guarded = generations
        .enforce_in_turn(long, LEAST_LOAD.into(), Format::Yaml)
        .await;
    assert!(
        matches!(
            guarded,
            Err(Unanswered::Manager(EnforceRefused::Guarded(0, None)))
        ),
        "{guarded:?}"
    );
    assert!(names(dir.path()).is_empty());
    assert_eq!(generations.apply_counts(), applied(0, 1));
}

/// Write calls take the door one at a time, in the order they came: a
/// call awaits its turn before its thread starts, so while the turn is
/// held none runs, and then each runs as the next number in its order.
#[tokio::test]
async fn write_calls_take_their_turn_in_the_order_they_came() {
    let (_dir, store, generations) = stored();
    let held = generations.write_turn.hold().await;
    let mut calls = JoinSet::new();
    for n in 1..=4_u64 {
        let door = Arc::clone(&generations);
        calls.spawn(async move {
            let document = bound_to(n).into_bytes();
            let call = door.enforce_in_turn(String::new(), document, Format::Yaml);
            (n, call.await)
        });
        // The call reaches its wait before the next one is made.
        tokio::task::yield_now().await;
    }
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(standing(&store).0, 0, "a write ran beside the turn held");
    drop(held);
    while let Some(call) = calls.join_next().await {
        let (n, answer) = call.expect("the call ends");
        let ran = answer.expect("a place").ran;
        assert_eq!(ran.map(|(number, _)| number), Ok(n));
    }
}

/// Waits until `free` of the write calls' places are free.
async fn places_free(door: &Generations, free: usize) {
    let started = Instant::now();
    while door.write_turn.available() != free {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "the write calls' places did not come to {free} free"
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

/// A call holds its place until its write ends, and the one whose
/// thread started holds the turn as long. With the door's lock held, as
/// a write in flight holds it, the calls that fill the places stay in
/// them, the turn stays taken and one more call is refused; once the
/// door is free each of them runs and gives its place back.
#[tokio::test]
#[expect(
    clippy::await_holding_lock,
    reason = "the case holds the door's lock, as a write in flight does, while the calls take their places"
)]
async fn a_write_call_holds_its_place_until_its_write_ends() {
    let (_dir, store, generations) = stored();
    let enforce = |door: Arc<Generations>, n: usize| async move {
        let n = u64::try_from(n).expect("a small count");
        let document = bound_to(n).into_bytes();
        door.enforce_in_turn(String::new(), document, Format::Yaml)
            .await
    };
    let applying = generations.applying_lock().lock();
    let mut calls = JoinSet::new();
    for n in 0..=WRITES_WAITING {
        calls.spawn(enforce(Arc::clone(&generations), n));
    }
    places_free(&generations, 0).await;
    // The call whose thread waits on the door holds the turn: no other
    // call's thread starts behind it, however long the door is held.
    for _ in 0..50 {
        assert!(
            generations.write_turn.is_held(),
            "the turn was given back before the write ended"
        );
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let past = enforce(Arc::clone(&generations), 0).await;
    assert!(matches!(past, Err(Unanswered::Waiting)), "{past:?}");
    assert_eq!(standing(&store).0, 0);
    drop(applying);
    while let Some(call) = calls.join_next().await {
        let answer = call.expect("the call ends").expect("a place");
        answer.ran.expect("it runs");
    }
    let all = u64::try_from(WRITES_WAITING).expect("a small count") + 1;
    assert_eq!(standing(&store).0, all);
    assert_eq!(generations.write_turn.available(), WRITES_WAITING + 1);
}

/// A caller that leaves while its call waits for the turn writes
/// nothing and frees its place: the next call runs as the first
/// generation, and its document alone is on file.
#[tokio::test]
async fn a_write_call_whose_caller_left_while_it_waited_writes_nothing() {
    let (dir, store, generations) = stored();
    let held = generations.write_turn.hold().await;
    let mut calls = JoinSet::new();
    let door = Arc::clone(&generations);
    let left = calls.spawn(async move {
        door.enforce_in_turn(String::new(), RANDOM.into(), Format::Yaml)
            .await
    });
    places_free(&generations, WRITES_WAITING).await;
    left.abort();
    let gone = calls.join_next().await.expect("the one call");
    let gone = gone.expect_err("its caller left");
    assert!(gone.is_cancelled(), "{gone}");
    drop(held);
    places_free(&generations, WRITES_WAITING + 1).await;
    assert_eq!((standing(&store).0, names(dir.path()).len()), (0, 0));

    let ran = generations
        .enforce_in_turn(String::new(), LEAST_LOAD.into(), Format::Yaml)
        .await
        .expect("a place")
        .ran;
    assert_eq!(ran.map(|(number, _)| number), Ok(1));
    let on_file = std::fs::read(dir.path().join(name_of(1, "yaml"))).expect("the file");
    assert_eq!(on_file, LEAST_LOAD.as_bytes());
}
