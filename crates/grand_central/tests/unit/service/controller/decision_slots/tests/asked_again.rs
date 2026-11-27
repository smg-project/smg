//! The cases of a request its gateway asks again, and of one it sends twice
//! without saying so.

use super::*;

/// One lease stands for a request its gateway asks again. The lease
/// the first ask was issued ends as abandoned where it waits, and the
/// engine that issued it is told, ahead of the repeat's own lease. A
/// lease of another request stands.
#[tokio::test]
async fn a_request_asked_again_ends_the_lease_its_first_ask_was_issued() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let mut bed = bed(Some(hearing(&told, 1)), true, (2, 8), []);
    let id = |count: u64| 1 << 32 | count;
    for request in 1..=2 {
        bed.ask(frame(request, 1)).await;
    }
    // The answer to the first never arrived: its gateway asks again.
    let answer = bed.ask(again(frame(1, 1))).await;
    assert_eq!(lease_of(&answer).id, id(3));
    let abandoned = (id(1), Some(ReleaseReason::Abandoned));
    assert_eq!(
        *told.lock(),
        [(id(1), None), (id(2), None), abandoned, (id(3), None)]
    );
    let store: &dyn LeasesStore = bed.store.as_ref();
    assert_eq!(store.get(GATEWAY, LeaseId(id(1))), None);
    assert_eq!(store.live(GATEWAY), 2, "another request's and the repeat's");
    assert_eq!(asked_again(&bed), [0, 1]);
}

/// A request sent twice under one id without the word is two requests:
/// each is issued a lease and neither ends the other's.
#[tokio::test]
async fn a_request_sent_twice_without_the_word_is_issued_two_leases() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let mut bed = bed(Some(hearing(&told, 1)), true, (2, 8), []);
    let id = |count: u64| 1 << 32 | count;
    for _ in 0..2 {
        bed.ask(frame(1, 1)).await;
    }
    assert_eq!(*told.lock(), [(id(1), None), (id(2), None)]);
    assert_eq!(bed.store.live(GATEWAY), 2);
    assert_eq!(asked_again(&bed), [0, 0]);
}

/// A repeat ends a lease that waits and no other: a lease its gateway
/// named a dispatch for stands, and a request no decider took has
/// none. Each such repeat is decided as a first ask.
#[tokio::test]
async fn a_request_asked_again_ends_no_lease_that_was_dispatched() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let mut bed = bed(Some(hearing(&told, 1)), true, (2, 8), []);
    let id = |count: u64| 1 << 32 | count;
    bed.ask(frame(1, 1)).await;
    let store: &dyn LeasesStore = bed.store.as_ref();
    let first = LeaseId(id(1));
    assert!(store.dispatched(GATEWAY, first, Dispatch::default(), bed.script.taken));
    // The gateway holds the first answer and asks again all the same,
    // and it asks again for a request no decider ever took.
    for request in [1, 9] {
        bed.ask(again(frame(request, 1))).await;
    }
    assert_eq!(*told.lock(), [(id(1), None), (id(2), None), (id(3), None)]);
    assert_eq!(bed.store.live(GATEWAY), 3);
    assert_eq!(asked_again(&bed), [2, 0]);
}

/// The word reaches the asking gateway's own leases alone: a lease
/// another gateway holds under the same id stands.
#[tokio::test]
async fn a_request_asked_again_ends_no_lease_of_another_gateway() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let mut bed = bed(Some(hearing(&told, 1)), true, (2, 8), []);
    let id = |count: u64| 1 << 32 | count;
    bed.ask(frame(1, 1)).await;
    let repeat = again(frame(1, 1));
    assert_eq!(bed.submit_at(repeat, OTHER, Duration::ZERO), None);
    bed.answer().await;
    assert_eq!(*told.lock(), [(id(1), None), (id(2), None)]);
    assert_eq!((bed.store.live(GATEWAY), bed.store.live(OTHER)), (1, 1));
    assert_eq!(asked_again(&bed), [1, 0]);
}

/// A repeat ends its earlier lease whatever becomes of its own
/// decision: one answered `Busy` for a budget that is gone leaves no
/// lease of its request on file.
#[tokio::test]
async fn a_request_asked_again_past_its_budget_ends_its_earlier_lease_all_the_same() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let mut bed = bed(Some(hearing(&told, 1)), true, (2, 8), []);
    let id = |count: u64| 1 << 32 | count;
    bed.ask(frame(1, 1)).await;
    // The repeat's budget is gone at its decider's read of the clock,
    // well inside the first lease's dispatch span.
    let (budget, read) = (Duration::from_millis(1), Duration::from_millis(2));
    bed.script.steps.lock().push_back(Step::At(read));
    let answer = bed.ask(budgeted(again(frame(1, 1)), budget)).await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    let abandoned = (id(1), Some(ReleaseReason::Abandoned));
    assert_eq!(*told.lock(), [(id(1), None), abandoned]);
    assert_eq!(bed.store.live(GATEWAY), 0);
    assert_eq!(asked_again(&bed), [0, 1]);
}

/// A dispatch handed ahead of the repeat stands: what was handed to
/// the file is applied before the repeat looks for a lease that waits.
#[tokio::test]
async fn a_dispatch_handed_ahead_of_a_request_asked_again_is_applied_first() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let mut bed = bed(Some(hearing(&told, 1)), true, (1, 8), []);
    let id = |count: u64| 1 << 32 | count;
    bed.ask(frame(1, 1)).await;
    // The repeat's decider is held at its first read of the clock: it
    // has the job at hand, so a frame handed now is left to it.
    let (reached, held) = std_mpsc::sync_channel(1);
    let (go, word) = std_mpsc::sync_channel(1);
    let hold = Step::Hold { reached, go: word };
    bed.script.steps.lock().push_back(hold);
    assert_eq!(bed.submit(again(frame(1, 1))), None);
    held.recv_timeout(PATIENCE)
        .expect("the decider reads the clock");
    let hand = Hand::new(GATEWAY);
    let dispatched = Said::Dispatched(Dispatch::default(), bed.script.taken);
    bed.store.hand(&hand, LeaseId(id(1)), dispatched);
    go.send(()).expect("the decider holds the other end");
    bed.answer().await;
    assert_eq!(*told.lock(), [(id(1), None), (id(2), None)]);
    assert_eq!(
        bed.store.live(GATEWAY),
        2,
        "the dispatched lease and the repeat's"
    );
    assert_eq!(asked_again(&bed), [1, 0]);
}
