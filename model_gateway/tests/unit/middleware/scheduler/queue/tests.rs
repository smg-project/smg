use std::time::Duration;

use tokio_util::sync::CancellationToken;

use super::*;
use crate::middleware::scheduler::Class;

fn waiter(class: Class) -> Waiter {
    let (tx, _rx) = oneshot::channel();
    Waiter::new(class, CancellationToken::new(), RequestId("t".into()), tx)
}

fn waiter_with_cancel(class: Class, cancel: CancellationToken) -> Waiter {
    let (tx, _rx) = oneshot::channel();
    Waiter::new(class, cancel, RequestId("t".into()), tx)
}

#[test]
fn test_try_enqueue_fills_to_capacity_then_rejects() {
    let q = FifoClassQueue::new(4);
    for _ in 0..4 {
        assert!(q.try_enqueue(waiter(Class::Default)).is_ok());
    }
    let rejected = q
        .try_enqueue(waiter(Class::Default))
        .expect_err("queue full");
    assert_eq!(rejected.class, Class::Default);
}

#[test]
fn test_pop_eligible_is_fifo() {
    // Distinguish waiters by class to verify insertion order is preserved.
    let q = FifoClassQueue::new(4);
    q.try_enqueue(waiter(Class::Bulk)).unwrap();
    q.try_enqueue(waiter(Class::Default)).unwrap();
    q.try_enqueue(waiter(Class::Interactive)).unwrap();
    assert_eq!(q.pop_eligible().unwrap().class, Class::Bulk);
    assert_eq!(q.pop_eligible().unwrap().class, Class::Default);
    assert_eq!(q.pop_eligible().unwrap().class, Class::Interactive);
    assert!(q.pop_eligible().is_none());
}

#[test]
fn test_pop_eligible_returns_none_when_empty() {
    let q = FifoClassQueue::new(4);
    assert!(q.pop_eligible().is_none());
}

#[test]
fn test_head_age_is_some_after_enqueue() {
    let q = FifoClassQueue::new(4);
    assert!(q.head_age().is_none());
    q.try_enqueue(waiter(Class::Default)).unwrap();
    std::thread::sleep(Duration::from_millis(5));
    let age = q.head_age().expect("head present");
    assert!(age >= Duration::from_millis(5));
}

#[test]
fn test_depth_reflects_enqueue_and_pop() {
    let q = FifoClassQueue::new(4);
    assert_eq!(q.depth(), 0);
    q.try_enqueue(waiter(Class::Default)).unwrap();
    q.try_enqueue(waiter(Class::Default)).unwrap();
    assert_eq!(q.depth(), 2);
    q.pop_eligible();
    assert_eq!(q.depth(), 1);
}

#[test]
fn test_drop_cancelled_head_removes_cancelled_head_only() {
    let q = FifoClassQueue::new(4);
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    q.try_enqueue(waiter_with_cancel(Class::Default, cancelled))
        .unwrap();
    q.try_enqueue(waiter(Class::Interactive)).unwrap();
    assert_eq!(q.depth(), 2);

    q.drop_cancelled_head();
    assert_eq!(q.depth(), 1, "cancelled head dropped");
    assert_eq!(
        q.pop_eligible().unwrap().class,
        Class::Interactive,
        "live waiter exposed as new head"
    );
}

#[test]
fn test_drop_cancelled_head_leaves_live_head_alone() {
    let q = FifoClassQueue::new(4);
    q.try_enqueue(waiter(Class::Default)).unwrap();
    q.drop_cancelled_head();
    assert_eq!(q.depth(), 1, "non-cancelled head retained");
}

#[test]
fn test_drop_cancelled_head_noop_on_empty_queue() {
    let q = FifoClassQueue::new(4);
    q.drop_cancelled_head();
    assert_eq!(q.depth(), 0);
}

#[test]
fn test_drop_cancelled_head_drains_run_in_single_call() {
    // When several clients in a row have walked away, a single call
    // should clear the whole run rather than requiring the caller to
    // loop. This keeps the dispatcher's GC pass to one lock cycle
    // instead of N.
    let q = FifoClassQueue::new(8);
    for _ in 0..3 {
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        q.try_enqueue(waiter_with_cancel(Class::Default, cancelled))
            .unwrap();
    }
    q.try_enqueue(waiter(Class::Interactive)).unwrap();
    assert_eq!(q.depth(), 4);

    q.drop_cancelled_head();
    assert_eq!(
        q.depth(),
        1,
        "all three cancelled heads dropped in one call"
    );
    assert_eq!(
        q.pop_eligible().unwrap().class,
        Class::Interactive,
        "first live waiter is now the head"
    );
}
