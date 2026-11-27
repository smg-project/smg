use std::{convert::Infallible, sync::Arc};

use super::*;

fn lease(release: ReleasePoint) -> (RequestLease<Arc<()>>, std::sync::Weak<()>) {
    let probe = Arc::new(());
    let weak = Arc::downgrade(&probe);
    (
        RequestLease::new(probe, RoutingDerivatives::default(), release),
        weak,
    )
}

#[test]
fn after_dispatch_release_frees_request_and_memo() {
    let (lease, probe) = lease(ReleasePoint::AfterDispatch);
    let body = lease
        .serialize_with(|_| Ok::<_, Infallible>(b"abcd".to_vec()))
        .unwrap();
    assert_eq!(probe.strong_count(), 1);
    assert_eq!(lease.body(), Some(body.clone()));

    lease.release_dispatch();

    assert_eq!(probe.strong_count(), 0, "release must free the request");
    assert_eq!(lease.body(), None, "release must drop the memo handle");
    assert_eq!(body.as_ref(), b"abcd", "caller handles stay valid");
}

#[test]
fn at_retry_close_keeps_request_until_drop() {
    let (lease, probe) = lease(ReleasePoint::AtRetryClose);
    lease
        .serialize_with(|_| Ok::<_, Infallible>(b"abcd".to_vec()))
        .unwrap();

    lease.release_dispatch();

    assert_eq!(probe.strong_count(), 1, "request must survive for replay");
    assert!(lease.body().is_some());

    drop(lease);
    assert_eq!(probe.strong_count(), 0, "drop closes the lease");
}

#[test]
fn serialize_refreshes_the_memo_per_attempt() {
    let (lease, _probe) = lease(ReleasePoint::AtRetryClose);
    lease
        .serialize_with(|_| Ok::<_, Infallible>(b"first".to_vec()))
        .unwrap();
    lease
        .serialize_with(|_| Ok::<_, Infallible>(b"second".to_vec()))
        .unwrap();
    assert_eq!(lease.body().as_deref(), Some(b"second".as_slice()));
}

#[test]
fn sides_serialize_from_one_view_and_release_together() {
    let (lease, probe) = lease(ReleasePoint::AfterDispatch);
    let (prefill, decode) = lease
        .serialize_sides_with(|_| Ok::<_, Infallible>((b"pp".to_vec(), b"dddd".to_vec())))
        .unwrap();
    assert_eq!(lease.body(), None, "sides are not a single-body memo");

    lease.release_dispatch();

    assert_eq!(probe.strong_count(), 0);
    assert_eq!(prefill.as_ref(), b"pp");
    assert_eq!(decode.as_ref(), b"dddd");
}
