use smg_worker::health::resilience::{
    DEFAULT_CAPACITY_STATUS_CODES, DEFAULT_RETRYABLE_STATUS_CODES,
};

use super::*;

/// A status tonic made of an HTTP/2 fault under the call (the h2 error in
/// its source chain) is the worker's connection failing, whatever code
/// tonic picked from the frame's reason: a 502 the breaker samples and
/// the retry layer replays elsewhere, never the 429 a GOAWAY
/// ENHANCE_YOUR_CALM reads as (a rate limit the client never hit) nor the
/// 500 of a worker's own answer. The worker's own RESOURCE_EXHAUSTED, in
/// its trailers, stays capacity pushback.
#[test]
fn an_http2_fault_under_the_call_is_a_retryable_502_whatever_its_code() {
    use smg_grpc_client::h2::{Error, Reason};

    for (reason, code) in [
        (Reason::ENHANCE_YOUR_CALM, Code::ResourceExhausted),
        (Reason::PROTOCOL_ERROR, Code::Internal),
        (Reason::FLOW_CONTROL_ERROR, Code::Internal),
        (Reason::REFUSED_STREAM, Code::Unavailable),
        (Reason::CANCEL, Code::Cancelled),
    ] {
        let status = tonic::Status::from_error(Box::new(Error::from(reason)));
        assert_eq!(status.code(), code, "tonic's own code for {reason:?}");
        assert_eq!(status.http_status(), StatusCode::BAD_GATEWAY, "{reason:?}");
        let result: Result<(), tonic::Status> = Err(status);
        assert_eq!(result.breaker_status(), Some(502), "{reason:?}");
    }
    assert!(DEFAULT_RETRYABLE_STATUS_CODES.contains(&502));
    assert_eq!(
        tonic::Status::resource_exhausted("busy").http_status(),
        StatusCode::TOO_MANY_REQUESTS
    );
}

/// A worker's report that the engine could not continue the request under
/// its constraint is answered for the constraint: a 422 under its own
/// code with the worker's words, terminal for the retry layer, sampled by
/// no breaker; the same code without the report keeps its 400.
#[test]
fn a_constraint_failure_is_a_terminal_422_the_breaker_does_not_sample() {
    use smg_grpc_client::constraint_failed_status;

    use crate::routers::{common::retry::is_retryable_response, error};

    let status = constraint_failed_status("the scheduler aborted the request: Aborted");
    assert_eq!(status.http_status(), StatusCode::UNPROCESSABLE_ENTITY);
    let result: Result<(), tonic::Status> = Err(status.clone());
    assert_eq!(result.breaker_status(), None);
    let response = status
        .constraint_failure_response()
        .expect("the report is answered");
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        CONSTRAINT_FAILED_CODE
    );
    assert!(!is_retryable_response(&response));
    let plain = tonic::Status::failed_precondition("the scheduler aborted the request");
    assert_eq!(plain.http_status(), StatusCode::BAD_REQUEST);
    assert!(plain.constraint_failure_response().is_none());
}

/// A worker's refusal of a constraint its engine cannot enforce is the
/// request's 400 under its own code with the worker's words, terminal for
/// the retry layer and sampled by no breaker; an `INVALID_ARGUMENT`
/// without the trailer is no refusal of the kind.
#[test]
fn a_constraint_refusal_is_a_terminal_400_the_breaker_does_not_sample() {
    use smg_grpc_client::constraint_refused_status;

    use crate::routers::{common::retry::NonRetryable, error};

    let status = constraint_refused_status("The JSON schema constraint cannot be enforced");
    assert_eq!(status.http_status(), StatusCode::BAD_REQUEST);
    let result: Result<(), tonic::Status> = Err(status.clone());
    assert_eq!(result.breaker_status(), None);
    let response = status
        .constraint_refusal_response()
        .expect("the refusal is answered");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        UNSUPPORTED_CONSTRAINT_CODE
    );
    // A 400 is not replayed by status alone; the refusal is marked
    // terminal whatever statuses the operator makes retryable.
    assert!(response.extensions().get::<NonRetryable>().is_some());
    let plain = tonic::Status::invalid_argument("The JSON schema constraint cannot be enforced");
    assert!(plain.constraint_refusal_response().is_none());
    assert_eq!(plain.breaker_status(), Some(400));
}

/// The reason a status names for the router error family, by the same
/// predicates that class it: the worker's connection failing under the
/// call (an HTTP/2 fault of any code, a socket fault, the worker's side
/// cancelling), its deadline, its capacity pushback, its unavailability,
/// the request's own fault, the engine's own report, a failure inside
/// the worker, a constraint, the request's media; a code outside them is
/// `other`.
#[test]
fn a_status_names_the_reason_behind_its_class() {
    use metrics_labels::*;
    use smg_grpc_client::{constraint_failed_status, h2::Reason};

    for reason in [
        Reason::ENHANCE_YOUR_CALM,
        Reason::PROTOCOL_ERROR,
        Reason::REFUSED_STREAM,
        Reason::CANCEL,
    ] {
        let status = tonic::Status::from_error(Box::new(smg_grpc_client::h2::Error::from(reason)));
        assert_eq!(
            status.error_reason(),
            REASON_TRANSPORT_DISCONNECT,
            "{reason:?}"
        );
    }
    for (status, reason) in [
        (
            tonic::Status::cancelled("operation was canceled"),
            REASON_TRANSPORT_DISCONNECT,
        ),
        (
            tonic::Status::deadline_exceeded("too slow"),
            REASON_DEADLINE,
        ),
        (
            tonic::Status::resource_exhausted("kv cache full"),
            REASON_OVERLOADED,
        ),
        (
            tonic::Status::unavailable("draining"),
            REASON_ENGINE_UNAVAILABLE,
        ),
        (
            tonic::Status::invalid_argument("bad"),
            REASON_INVALID_REQUEST,
        ),
        (
            tonic::Status::failed_precondition("bad"),
            REASON_INVALID_REQUEST,
        ),
        (tonic::Status::out_of_range("bad"), REASON_INVALID_REQUEST),
        (
            tonic::Status::internal(ENGINE_ERROR_MESSAGE),
            REASON_ENGINE_ERROR,
        ),
        (
            tonic::Status::internal("the servicer failed"),
            REASON_ENGINE_INTERNAL,
        ),
        (tonic::Status::unknown("no idea"), REASON_ENGINE_INTERNAL),
        (tonic::Status::data_loss("lost"), REASON_ENGINE_INTERNAL),
        (
            constraint_failed_status("the scheduler aborted the request: Aborted"),
            REASON_CONSTRAINT_FAILED,
        ),
        (
            tonic::Status::unavailable(
                "HTTP error while fetching media from https://example.invalid/a.png",
            ),
            REASON_MEDIA_FAULT,
        ),
        (tonic::Status::not_found("gone"), REASON_OTHER),
        (tonic::Status::unimplemented("no"), REASON_OTHER),
        (tonic::Status::permission_denied("no"), REASON_OTHER),
    ] {
        assert_eq!(status.error_reason(), reason, "{status:?}");
    }
}

/// The HTTP answer made of a status carries the status's reason, so the
/// router that counts the answer (which sees no status) counts the cause
/// the status named.
#[test]
fn the_http_answer_of_a_status_carries_its_reason() {
    use super::super::metrics::ErrorReason;

    let response = tonic::Status::deadline_exceeded("too slow").to_http_error(
        "start_generation_failed",
        "Failed to start generation".to_string(),
    );
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(
        response.extensions().get::<ErrorReason>().map(|r| r.0),
        Some(metrics_labels::REASON_DEADLINE)
    );
    let plain = error::internal_error("no_response_produced", "No response produced");
    assert!(plain.extensions().get::<ErrorReason>().is_none());
}

/// A status that is no answer of the worker's: an HTTP/2 fault under
/// the call, the worker's side cancelling, a dial that failed, a
/// keep-alive timeout. A status the worker answered with is one.
#[tokio::test]
async fn a_status_says_whether_the_worker_answered() {
    use smg_grpc_client::h2::{Error, Reason};

    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("a loopback port");
    let closed = format!("grpc://{}", listener.local_addr().expect("its address"));
    drop(listener);
    let Err(refused) = smg_grpc_client::connect_channel(&closed).await else {
        panic!("nothing listens on {closed}");
    };
    for (status, no_answer) in [
        (
            tonic::Status::from_error(Box::new(Error::from(Reason::PROTOCOL_ERROR))),
            true,
        ),
        (tonic::Status::cancelled("operation was canceled"), true),
        (tonic::Status::from_error(refused), true),
        (smg_grpc_client::keep_alive_timeout_status(), true),
        (tonic::Status::internal("the servicer failed"), false),
        (tonic::Status::unavailable("draining"), false),
        (tonic::Status::unimplemented("no"), false),
        (tonic::Status::deadline_exceeded("too slow"), false),
        (tonic::Status::invalid_argument("bad"), false),
        (tonic::Status::resource_exhausted("kv cache full"), false),
    ] {
        assert_eq!(status.is_no_answer(), no_answer, "{status:?}");
    }
}

/// What a failure tells Grand Central where it ends a request: the
/// status the breaker samples for it and whether the worker answered
/// it. A failure the breaker takes no sample of tells nothing.
#[test]
fn a_failure_tells_the_engine_what_the_breaker_samples_and_who_answered() {
    assert_eq!(
        tonic::Status::internal("the engine failed").end_status(),
        Some((500, true))
    );
    assert_eq!(
        tonic::Status::invalid_argument("a bad request").end_status(),
        Some((400, true))
    );
    assert_eq!(
        tonic::Status::cancelled("the worker's side ended the call").end_status(),
        Some((502, false))
    );
    let constraint = smg_grpc_client::constraint_failed_status("no token fits the grammar");
    assert_eq!(constraint.breaker_status(), None);
    assert_eq!(constraint.end_status(), None);
}

#[test]
fn invalid_argument_maps_to_400_and_is_not_a_circuit_breaker_failure() {
    for status in [
        tonic::Status::invalid_argument("bad"),
        tonic::Status::failed_precondition("bad"),
        tonic::Status::out_of_range("bad"),
    ] {
        assert_eq!(status.http_status(), StatusCode::BAD_REQUEST);
        let result: Result<(), tonic::Status> = Err(status);
        assert_eq!(result.breaker_status(), Some(400));
    }
    assert!(!DEFAULT_RETRYABLE_STATUS_CODES.contains(&400));
}

/// A call the worker's side cancelled is the worker's failure: a 502 the
/// retry layer replays elsewhere and the breaker samples, never a 400.
#[test]
fn a_cancelled_call_maps_to_502_and_is_a_retryable_circuit_breaker_failure() {
    let status = tonic::Status::cancelled("operation was canceled");
    assert_eq!(status.http_status(), StatusCode::BAD_GATEWAY);
    let result: Result<(), tonic::Status> = Err(status);
    assert_eq!(result.breaker_status(), Some(502));
    assert!(DEFAULT_RETRYABLE_STATUS_CODES.contains(&502));
    assert!(!DEFAULT_RETRYABLE_STATUS_CODES.contains(&400));
}

#[test]
fn internal_and_unknown_map_to_500() {
    for status in [
        tonic::Status::internal("boom"),
        tonic::Status::unknown("boom"),
        tonic::Status::data_loss("boom"),
    ] {
        assert_eq!(status.http_status(), StatusCode::INTERNAL_SERVER_ERROR);
        let result: Result<(), tonic::Status> = Err(status);
        assert_eq!(result.breaker_status(), Some(500));
    }
    assert!(DEFAULT_RETRYABLE_STATUS_CODES.contains(&500));
}

#[test]
fn resource_exhausted_maps_to_429_capacity_pushback() {
    assert_eq!(
        tonic::Status::resource_exhausted("busy").http_status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    assert_eq!(
        tonic::Status::unavailable("down").http_status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert!(DEFAULT_CAPACITY_STATUS_CODES.contains(&429));
    let ok: Result<(), tonic::Status> = Ok(());
    assert_eq!(ok.breaker_status(), Some(200));
}

/// A worker's report that the request's media failed is answered for the
/// media (a 400 for the request's own fault, a 502 for the media host's)
/// and is no circuit-breaker sample, whichever code carries it. The API's
/// code is the one the worker names, `media_fetch_failed` when it names
/// none or one this router does not know.
#[test]
fn a_media_fault_answers_for_the_media_and_is_no_breaker_sample() {
    let client = WorkerMediaFault::Client.stamp(tonic::Status::invalid_argument(
        "HTTP error while fetching media: HTTP status client error (404 Not Found) for \
             url (https://media.example/missing.png)",
    ));
    assert_eq!(client.media_fault(), Some(WorkerMediaFault::Client));
    assert_eq!(client.http_status(), StatusCode::BAD_REQUEST);
    assert_eq!(client.media_code(), MediaErrorCode::FetchFailed);
    assert_eq!(client.breaker_status(), None);
    // The client reads the gateway's sentence for a fetch that failed,
    // by whose fault it is; the worker's own words stay on the log line.
    assert_eq!(
        client.media_message(WorkerMediaFault::Client),
        MEDIA_FETCH_FAILED_CLIENT
    );

    let transient = WorkerMediaFault::Transient.stamp(tonic::Status::unavailable(
        "media fetch of https://media.example/a.png timed out after 10s",
    ));
    assert_eq!(transient.media_fault(), Some(WorkerMediaFault::Transient));
    assert_eq!(transient.http_status(), StatusCode::BAD_GATEWAY);
    assert_eq!(transient.media_code(), MediaErrorCode::FetchFailed);
    assert_eq!(
        transient.media_message(WorkerMediaFault::Transient),
        MEDIA_FETCH_FAILED_TRANSIENT
    );
    let result: Result<(), tonic::Status> = Err(transient);
    assert_eq!(result.breaker_status(), None);

    let too_large = smg_grpc_client::stamp_media_code(
        WorkerMediaFault::Client.stamp(tonic::Status::invalid_argument(
            "decoded video RGB payload would be 2208384000 bytes, exceeding \
                 SMG_VIDEO_MAX_DECODED_BYTES=1073741824",
        )),
        "media_too_large",
    );
    assert_eq!(too_large.media_fault(), Some(WorkerMediaFault::Client));
    assert_eq!(too_large.http_status(), StatusCode::BAD_REQUEST);
    assert_eq!(too_large.media_code(), MediaErrorCode::TooLarge);
    assert_eq!(too_large.breaker_status(), None);
    // A size refusal and media refused as sent describe the request's
    // payload: the worker's words reach the client.
    assert_eq!(
        too_large.media_message(WorkerMediaFault::Client),
        too_large.message()
    );

    let refused = smg_grpc_client::stamp_media_code(
        WorkerMediaFault::Client.stamp(tonic::Status::invalid_argument(
            "data URL parse error: missing comma in data url",
        )),
        "invalid_multimodal_request",
    );
    assert_eq!(refused.media_fault(), Some(WorkerMediaFault::Client));
    assert_eq!(refused.http_status(), StatusCode::BAD_REQUEST);
    assert_eq!(refused.media_code(), MediaErrorCode::InvalidRequest);
    assert_eq!(refused.breaker_status(), None);
    assert_eq!(
        refused.media_message(WorkerMediaFault::Client),
        "data URL parse error: missing comma in data url"
    );

    // A fetch the worker names by its code reads the sentence, whatever
    // words the worker has for it.
    let named = smg_grpc_client::stamp_media_code(
        WorkerMediaFault::Client.stamp(tonic::Status::invalid_argument(
            "the reference could not be followed",
        )),
        "media_fetch_failed",
    );
    assert_eq!(named.media_code(), MediaErrorCode::FetchFailed);
    assert_eq!(
        named.media_message(WorkerMediaFault::Client),
        MEDIA_FETCH_FAILED_CLIENT
    );

    // The gRPC framing's refusal of a message longer than its four-byte
    // length prefix carries (tonic's words; this gateway's own client
    // when the request's pixel tensors ride in the Generate request): the
    // request's own size, a 400 under `media_too_large` and no breaker
    // sample, where a RESOURCE_EXHAUSTED otherwise reads as the worker's
    // capacity pushback, a 429 the retry layer replays.
    let framing = tonic::Status::resource_exhausted(
        "Cannot return body with more than 4GB of data but got 5065026759 bytes",
    );
    assert_eq!(framing.media_fault(), Some(WorkerMediaFault::Client));
    assert_eq!(framing.http_status(), StatusCode::BAD_REQUEST);
    assert_eq!(framing.media_code(), MediaErrorCode::TooLarge);
    assert_eq!(framing.breaker_status(), None);
    // The framing's refusal names no fetch: its own words reach the client.
    assert_eq!(
        framing.media_message(WorkerMediaFault::Client),
        framing.message()
    );
    let pushback = tonic::Status::resource_exhausted("the engine is at capacity");
    assert_eq!(pushback.media_fault(), None);
    assert_eq!(pushback.http_status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(pushback.breaker_status(), Some(429));
    // The same encoder's refusal under a configured limit below the
    // framing's cap (`max_encoding_message_size`) reads the same way;
    // any other OUT_OF_RANGE stays the plain 400 it was.
    let limited = tonic::Status::out_of_range(
        "Error, encoded message length too large: found 5065026759 bytes, the limit is: 4194304 bytes",
    );
    assert_eq!(limited.media_fault(), Some(WorkerMediaFault::Client));
    assert_eq!(limited.http_status(), StatusCode::BAD_REQUEST);
    assert_eq!(limited.media_code(), MediaErrorCode::TooLarge);
    assert_eq!(limited.breaker_status(), None);
    let plain_range = tonic::Status::out_of_range("position 9 is past the end");
    assert_eq!(plain_range.media_fault(), None);
    assert_eq!(plain_range.http_status(), StatusCode::BAD_REQUEST);

    let unknown = smg_grpc_client::stamp_media_code(
        WorkerMediaFault::Client.stamp(tonic::Status::invalid_argument("new words")),
        "media_from_the_future",
    );
    assert_eq!(unknown.media_code(), MediaErrorCode::FetchFailed);
    // A code this router does not know names no fetch: the worker's words
    // reach the client under the fallback code.
    assert_eq!(unknown.media_message(WorkerMediaFault::Client), "new words");
}

/// A worker's message above the receive bound is refused by the gateway's
/// own decoder, `OUT_OF_RANGE` in the words of tonic 0.14's
/// `codec::decode`. The request was valid and the answer could not be
/// read: an upstream fault, as an HTTP router answers for its size bound.
#[test]
fn a_worker_message_above_the_receive_bound_is_an_upstream_fault() {
    use crate::routers::common::retry::is_retryable_response;

    let refused = tonic::Status::out_of_range(
        "Error, decoded message length too large: found 9886731 bytes, the limit is: 4194304 bytes",
    );
    assert_eq!(refused.http_status(), StatusCode::BAD_GATEWAY);
    assert_eq!(refused.error_reason(), metrics_labels::REASON_BAD_RESPONSE);
    // The worker did nothing wrong, and another worker's answer would
    // be as large: no breaker sample, no report, no replay; the code a
    // client knows stays.
    assert_eq!(refused.breaker_status(), None);
    assert_eq!(refused.end_status(), None);
    let answer = refused.to_http_error("worker_stream_failed", "too large".to_string());
    assert_eq!(answer.status(), StatusCode::BAD_GATEWAY);
    assert!(!is_retryable_response(&answer));
    // Any other `OUT_OF_RANGE` stays the 400 it was, sampled and final.
    let plain = tonic::Status::out_of_range("position 9 is past the end");
    assert_eq!(plain.http_status(), StatusCode::BAD_REQUEST);
    assert_eq!(plain.error_reason(), metrics_labels::REASON_INVALID_REQUEST);
    assert_eq!(plain.breaker_status(), Some(400));
}

/// The gateway's sentence stands only for a status that names a fetch. A
/// worker that predates the code trailer names the fault alone: where its
/// words describe the request's payload they reach the client, as the
/// same words do under a code.
#[test]
fn a_fault_without_a_code_keeps_the_words_that_name_no_fetch() {
    for message in [
        "data URL parse error: missing comma in data url",
        "image decode error: The image format could not be determined",
        "video input payload exceeds the maximum size of 536870912 bytes",
    ] {
        let payload = WorkerMediaFault::Client.stamp(tonic::Status::invalid_argument(message));
        assert_eq!(payload.media_fault(), Some(WorkerMediaFault::Client));
        assert_eq!(payload.http_status(), StatusCode::BAD_REQUEST);
        assert_eq!(payload.breaker_status(), None);
        assert_eq!(
            payload.media_message(WorkerMediaFault::Client),
            message,
            "{message}"
        );
    }
}

/// A worker that predates the trailer reports a fetch failure on
/// `UNAVAILABLE` in the connector's words; it is read as the media host's
/// fault and answered for the media (a 502). The worker's own trouble, and
/// a caller fault the worker already answers 4xx, are read as before.
#[test]
fn a_fetch_failure_in_the_connectors_words_is_read_without_the_trailer() {
    for message in [
        "Failed to finalize multimodal tracker: HTTP error while fetching media: error \
             sending request for url (https://media.example/a.png)",
        "Failed to finalize multimodal tracker: media fetch timed out after 10s",
        "Failed to finalize multimodal tracker: media fetch of https://media.example/a.png \
             timed out after 30s",
    ] {
        let legacy = tonic::Status::unavailable(message);
        assert_eq!(
            legacy.media_fault(),
            Some(WorkerMediaFault::Transient),
            "{message}"
        );
        assert_eq!(legacy.http_status(), StatusCode::BAD_GATEWAY);
        assert_eq!(legacy.media_code(), MediaErrorCode::FetchFailed);
        assert_eq!(
            legacy.media_message(WorkerMediaFault::Transient),
            MEDIA_FETCH_FAILED_TRANSIENT
        );
        assert_eq!(legacy.breaker_status(), None);
    }
    let sidecar = tonic::Status::unavailable("sidecar_timeout: no result");
    assert_eq!(sidecar.media_fault(), None);
    assert_eq!(sidecar.breaker_status(), Some(503));
    let engine = tonic::Status::unavailable("engine is restarting");
    assert_eq!(engine.breaker_status(), Some(503));
    let decode = tonic::Status::invalid_argument(
        "Failed to finalize multimodal tracker: image decode error: The image format \
             could not be determined",
    );
    assert_eq!(decode.media_fault(), None);
    assert_eq!(decode.breaker_status(), Some(400));
}
