//! The scheduler's permit on the edge's one body wrapper: TTFT on the
//! first data frame, the slot released with the body.
use std::sync::Arc;

use axum::{body::Body, http::StatusCode, response::Response};
use http_body::Body as HttpBody;
use http_body_util::BodyExt;
use smg_auth::RequestId;

use super::*;
use crate::middleware::{
    edge_body::at_the_edge,
    scheduler::{PriorityScheduler, SchedulerPermit, SchedulerSettings},
};

fn scheduler() -> Arc<PriorityScheduler> {
    let settings = SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, None).unwrap();
    // Default reservations sum to 160 (Interactive 128 + System 32),
    // so capacity must be >= 160 for new() to accept them.
    PriorityScheduler::new(&settings, 256).unwrap()
}

fn permit(sched: &Arc<PriorityScheduler>, id: &str) -> SchedulerPermit {
    sched
        .acquire_inflight(Class::Default, RequestId(id.to_string()))
        .expect("slot available")
}

/// `body` with `permit` on it, as the admission layer leaves it, at the
/// edge.
fn guarded(body: Body, permit: SchedulerPermit) -> Body {
    let mut response = Response::new(body);
    BodyDuties::hand(&mut response, |duties| {
        duties.permit = Some(Box::new(permit));
    });
    at_the_edge(response).into_body()
}

#[tokio::test]
async fn the_first_data_frame_marks_ttft() {
    let sched = scheduler();
    let p = permit(&sched, "req-ttft");
    let handle = Arc::clone(p.handle());
    let mut body = guarded(Body::from("hello world"), p);

    assert!(handle.is_preemptible(), "pre-TTFT before first frame");
    let frame = body.frame().await.expect("a frame").expect("ok frame");
    assert!(frame.is_data());
    assert!(
        !handle.is_preemptible(),
        "first data frame must mark TTFT (no longer preemptible)"
    );
    // A request that has emitted its first byte must reject preemption.
    assert!(!handle.try_mark_preempted());
}

#[tokio::test]
async fn a_request_preempted_before_its_first_byte_ends_the_stream_without_it() {
    let sched = scheduler();
    let p = permit(&sched, "req-preempt");
    let handle = Arc::clone(p.handle());
    // Scheduler wins the preempt CAS before any byte is polled.
    assert!(handle.try_mark_preempted());

    let mut body = guarded(Body::from("should not surface"), p);
    // The first poll must end the stream rather than yield the data.
    assert!(
        body.frame().await.is_none(),
        "preempted body must end the stream on first frame"
    );
    // is_end_stream must agree with the forced None, and a re-poll must
    // stay ended.
    assert!(
        HttpBody::is_end_stream(&body),
        "force-ended stream must report is_end_stream() == true"
    );
    assert!(
        body.frame().await.is_none(),
        "re-polling a force-ended stream must stay ended"
    );
    assert_eq!(
        sched.inflight_for_test(Class::Default),
        0,
        "the slot is released with the ended body"
    );
}

#[tokio::test]
async fn an_empty_body_never_marks_ttft() {
    let sched = scheduler();
    let p = permit(&sched, "req-empty");
    let handle = Arc::clone(p.handle());
    // Empty body: no data frames at all.
    let body = guarded(Body::empty(), p);
    let _ = body.collect().await;
    assert!(
        handle.is_preemptible(),
        "no data frame emitted -> TTFT never marked"
    );
}

#[tokio::test]
async fn dropping_the_body_releases_the_slot() {
    let sched = scheduler();
    assert_eq!(sched.inflight_for_test(Class::Default), 0);
    let p = permit(&sched, "req-drop");
    assert_eq!(sched.inflight_for_test(Class::Default), 1);
    let body = guarded(Body::from("x"), p);
    drop(body);
    assert_eq!(
        sched.inflight_for_test(Class::Default),
        0,
        "dropping the guarded body releases the slot"
    );
}

/// The preemption 503 the handler's guard wrote carries no permit: the
/// request is over, the slot goes with the response head, and the body
/// goes out whole, since no first-frame check rides on it.
#[tokio::test]
async fn a_preemption_503_carries_no_permit_and_goes_out_whole() {
    let sched = scheduler();
    let p = permit(&sched, "req-503");
    let handle = Arc::clone(p.handle());
    assert!(handle.try_mark_preempted(), "the scheduler won the CAS");
    let mut response = Response::builder()
        .status(StatusCode::SERVICE_UNAVAILABLE)
        .header(HEADER_X_SMG_PREEMPTED, "true")
        .body(Body::from("{\"error\":\"overloaded\"}"))
        .expect("response");
    carry_permit(&mut response, p);
    assert_eq!(
        sched.inflight_for_test(Class::Default),
        0,
        "the slot goes with the 503, not with its body"
    );
    let body = at_the_edge(response)
        .into_body()
        .collect()
        .await
        .expect("whole")
        .to_bytes();
    assert_eq!(&body[..], b"{\"error\":\"overloaded\"}");
}

/// The preemption header a worker's response carries through the relays
/// is not this scheduler's word: an admitted stream keeps its permit,
/// the slot stays held and the first frame marks TTFT.
#[tokio::test]
async fn a_relayed_preemption_header_on_an_admitted_stream_keeps_the_permit() {
    let sched = scheduler();
    let p = permit(&sched, "req-relayed");
    let handle = Arc::clone(p.handle());
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header(HEADER_X_SMG_PREEMPTED, "true")
        .body(Body::from("data: x\n\n"))
        .expect("response");
    carry_permit(&mut response, p);
    assert_eq!(
        sched.inflight_for_test(Class::Default),
        1,
        "the slot stays held: the header is the worker's, not the scheduler's"
    );
    let mut body = at_the_edge(response).into_body();
    let frame = body.frame().await.expect("a frame").expect("frame ok");
    assert!(frame.is_data());
    assert!(!handle.is_preemptible(), "the first frame marked TTFT");
    drop(body);
    assert_eq!(sched.inflight_for_test(Class::Default), 0);
}
