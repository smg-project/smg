//! The cases of the door's turn: a call reads the fence once it holds the
//! turn, a call that lost the lead while it waited is refused, and of the
//! copies of an add, a patch or a removal that writes, let go at once, one
//! lands, the others answered as behind it.

use super::*;

/// The copies of one call that [`at_once`] lets go together.
const CALLERS: usize = 4;

#[test]
fn a_call_reads_the_fence_once_it_holds_the_doors_turn() {
    let (_store, door) = door();
    // The fence's reader finds the turn taken: the call that reads it
    // holds it.
    let held = || {
        assert!(door.turn.try_lock().is_none(), "read outside the turn");
        Ok(TERM)
    };
    let row = row_of(
        &door
            .register(&operator(), held, &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let patch = proto::PatchWorkerRequest {
        worker_id: row.id,
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        revision: row.revision,
        ..Default::default()
    };
    let row = row_of(&door.patch(&operator(), held, &patch).expect("patched"));
    let removal = proto::DeregisterWorkerRequest {
        worker_id: row.id,
        revision: row.revision,
        drain: false,
    };
    door.deregister(&operator(), held, &removal)
        .expect("removed");
}

#[test]
fn a_call_that_lost_the_lead_while_it_waited_its_turn_is_refused_and_writes_nothing() {
    let (store, door) = door();
    let leads = AtomicBool::new(true);
    let fence = || {
        if leads.load(Ordering::SeqCst) {
            Ok(TERM)
        } else {
            Err(Status::failed_precondition("this replica does not lead"))
        }
    };
    let request = ask(spec(URL), Mode::Add);
    // Another call holds the turn. This one waits behind it, and the
    // lead is lost before its turn comes.
    let turn = door.turn.lock();
    let answer = thread::scope(|scope| {
        let waiting = scope.spawn(|| door.register(&operator(), fence, &request));
        leads.store(false, Ordering::SeqCst);
        drop(turn);
        waiting.join().expect("the call returns")
    });
    let refused = answer.expect_err("the lead was lost before the call's turn");
    assert_eq!(refused.code(), Code::FailedPrecondition);
    assert_eq!((store.epoch(), store.by_key(URL)), (0, None));
}

/// The answers of [`CALLERS`] copies of one call let go at one instant.
fn at_once<T: Send>(call: impl Fn() -> Result<T, Status> + Sync) -> Vec<Result<T, Status>> {
    let (ready, go) = (Barrier::new(CALLERS + 1), AtomicBool::new(false));
    thread::scope(|scope| {
        let calls: Vec<_> = (0..CALLERS)
            .map(|_| {
                scope.spawn(|| {
                    ready.wait();
                    while !go.load(Ordering::Acquire) {
                        std::hint::spin_loop();
                    }
                    call()
                })
            })
            .collect();
        ready.wait();
        go.store(true, Ordering::Release);
        calls
            .into_iter()
            .map(|call| call.join().expect("a caller"))
            .collect()
    })
}

/// What the callers that were refused were told.
fn refusals<T>(answers: &[Result<T, Status>]) -> Vec<Code> {
    answers
        .iter()
        .filter_map(|answer| answer.as_ref().err().map(Status::code))
        .collect()
}

#[test]
fn calls_at_once_take_their_turn_at_the_door() {
    // A call's read, its write and its read back are one step under the
    // door's lock. Of the copies of a call let go together one lands and
    // each other is told what it would be told after it, round after
    // round: an ADD that the URL stands, a patch that the row moved, a
    // removal that the row left.
    for round in 0..300 {
        let (store, door) = door();
        let added = at_once(|| door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add)));
        assert_eq!(
            refusals(&added),
            [Code::AlreadyExists; CALLERS - 1],
            "round {round}: {added:?}"
        );
        let row = store.by_key(URL).expect("the row");
        assert_eq!((store.epoch(), row.revision), (1, 1), "round {round}");

        let patch = proto::PatchWorkerRequest {
            worker_id: row.id.0,
            labels: [("lane".to_owned(), "b".to_owned())].into(),
            revision: row.revision,
            ..Default::default()
        };
        let patched = at_once(|| door.patch(&operator(), under(TERM), &patch));
        assert_eq!(
            refusals(&patched),
            [Code::FailedPrecondition; CALLERS - 1],
            "round {round}: {patched:?}"
        );

        let removal = proto::DeregisterWorkerRequest {
            worker_id: row.id.0,
            revision: 2,
            drain: false,
        };
        let removed = at_once(|| door.deregister(&operator(), under(TERM), &removal));
        assert_eq!(
            refusals(&removed),
            [Code::NotFound; CALLERS - 1],
            "round {round}: {removed:?}"
        );
        assert_eq!(
            (store.epoch(), store.by_key(URL)),
            (3, None),
            "round {round}"
        );
    }
}
