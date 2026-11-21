//! MiniMax's wire (MiniMax-Provider-Verifier): its stream-QoS contract
//! bounds the size of each SSE event's delta (almost none may be shorter
//! than five characters or longer than two hundred), so a chat stream is
//! re-sliced on its way out by [`RECHUNK`] (`routers::common::sse_rechunk`):
//! on the gRPC lane, where the gateway encodes the frames itself, and on the
//! regular HTTP relay, where the worker's body passes through. And a MiniMax
//! M3 answer whose tool call the reader could not read is not delivered: the
//! request ends as [`InvalidToolCallFormat`]. Everything else is the OpenAI
//! wire.

use std::time::Duration;

use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use serde::Serialize;
use symphony::formats::Table;
use tracing::warn;

use crate::{
    observability::metrics::metrics_labels,
    routers::{
        common::sse_rechunk::RechunkPolicy,
        error,
        grpc::{regular::symphony::UnreadableCall, utils::mark_error_reason},
    },
};

/// The contract's numbers: slices of at most 160 characters, cut once 80
/// are buffered, never leaving a tail under 5, and buffered text flushed
/// after 250 ms of upstream silence.
pub(super) const RECHUNK: RechunkPolicy = RechunkPolicy {
    slice_chars: 160,
    emit_threshold: 80,
    min_tail_chars: 5,
    idle_flush: Duration::from_millis(250),
};

/// The table whose answers the contract covers: MiniMax M3's, which reads
/// MiniMax-M3.1-Flash too. Another table on this wire (MiniMax M2's) reads
/// its answers as every wire's are read.
const M3_TABLE: &str = Table::MinimaxM3.name();

/// Whether an answer read by `table` fails its request on a tool call the
/// reader could not read (`Wire::fails_on_unreadable_tool_call`): the M3
/// table's answers alone.
pub(super) fn fails_on_unreadable_tool_call(table: &str) -> bool {
    table == M3_TABLE
}

/// The error `type` and `code` of [`InvalidToolCallFormat`], the
/// `X-SMG-Error-Code` of its whole response and the `error_type` label the
/// request counts under: the contract's name for the failure, spelled once.
pub(crate) const INVALID_TOOL_CALL_FORMAT: &str = metrics_labels::ERROR_INVALID_TOOL_CALL_FORMAT;

/// How a MiniMax M3 request ends when the model wrote a tool call the reader
/// could not read and the request expects calls
/// (`ChatResponseSpec::fails_on_unreadable_call`): the request's outcome, not
/// the worker's failure, in place of an answer with a partial or repaired
/// call. A whole response is a 400 ([`Self::response`]). A stream, whose
/// status is out, ends with the usage chunk when the request asked for one,
/// then one `event: error` frame ([`Self::sse_frame`]), with no finish chunk
/// ahead of it and no `[DONE]` after it. Both carry the same body:
/// `{"error":{"message":<the sentence>,"type":"invalid_tool_call_format",
/// "code":"invalid_tool_call_format","finish_reason":"length"|"error"}}`.
///
/// Nothing of the call that failed reaches the client on either surface: the
/// reader holds a call's events until its end is known and whole
/// (`Reader::failing_unreadable_calls`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InvalidToolCallFormat {
    /// The reader's verdict.
    unreadable: UnreadableCall,
    /// The token limit cut the call: the engine ended the output with the
    /// `length` finish while the call was incomplete.
    cut_by_limit: bool,
}

/// The body both surfaces carry, in the member order the contract writes.
#[derive(Serialize)]
struct Envelope {
    error: Detail,
}

#[derive(Serialize)]
struct Detail {
    message: &'static str,
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'static str,
    finish_reason: &'static str,
}

impl InvalidToolCallFormat {
    /// The ending for the reader's verdict on an output the engine ended
    /// with the `length` finish when `ended_at_limit`. The token limit is
    /// blamed only for a call it cut: a malformed block under `length`, or
    /// an incomplete call under the model's own stop, is not its doing.
    pub(crate) fn new(unreadable: UnreadableCall, ended_at_limit: bool) -> Self {
        Self {
            unreadable,
            cut_by_limit: ended_at_limit && unreadable == UnreadableCall::Incomplete,
        }
    }

    /// `error.finish_reason`: `length` for a call the token limit cut, where
    /// the request may succeed with room for the call; `error` otherwise.
    pub(crate) const fn finish_reason(self) -> &'static str {
        if self.cut_by_limit {
            "length"
        } else {
            "error"
        }
    }

    /// The sentence the client is told: the response failed, why, and that
    /// the request is to be retried.
    pub(crate) const fn message(self) -> &'static str {
        match (self.cut_by_limit, self.unreadable) {
            (true, _) => "Treat this response as failed. The model stopped due to max_tokens before it completed the tool call. Retry the request.",
            (false, UnreadableCall::Incomplete) => "Treat this response as failed. The model generated an incomplete tool call. Retry the request.",
            (false, UnreadableCall::Malformed) => "Treat this response as failed. The model generated a tool call with malformed structure. Retry the request.",
        }
    }

    /// The verdict's name in the log line.
    const fn category(self) -> &'static str {
        match self.unreadable {
            UnreadableCall::Incomplete => "incomplete_tool_call",
            UnreadableCall::Malformed => "malformed_structure",
        }
    }

    fn envelope(self) -> Envelope {
        Envelope {
            error: Detail {
                message: self.message(),
                error_type: INVALID_TOOL_CALL_FORMAT,
                code: INVALID_TOOL_CALL_FORMAT,
                finish_reason: self.finish_reason(),
            },
        }
    }

    /// The 400 a whole response ends with, counted under the contract's
    /// label (`error_type_for_response`) and the `bad_response` reason. The
    /// gateway's builder mints the status, the code's header and the metrics
    /// extension; its envelope's `type` is the status's and it carries no
    /// finish, so the contract's body replaces it.
    pub(crate) fn response(self) -> Response {
        let mut response = error::bad_request(INVALID_TOOL_CALL_FORMAT, self.message());
        *response.body_mut() = axum::Json(self.envelope()).into_response().into_body();
        mark_error_reason(&mut response, metrics_labels::REASON_BAD_RESPONSE);
        response
    }

    /// The frame a stream ends with: `event: error`, then the body. The
    /// caller sends no `[DONE]` after it.
    pub(crate) fn sse_frame(self) -> Bytes {
        let mut frame = Vec::with_capacity(256);
        frame.extend_from_slice(b"event: error\ndata: ");
        // A `Vec` writer never fails and the body is static text, so the
        // write cannot fail.
        let _ = serde_json::to_writer(&mut frame, &self.envelope());
        frame.extend_from_slice(b"\n\n");
        Bytes::from(frame)
    }

    /// The one WARN of a request that ends this way: the request, the model,
    /// the category and the finish the client is told. The model's bytes are
    /// never logged.
    pub(crate) fn warn(self, request_id: &str, model: &str) {
        warn!(
            request_id,
            model,
            error_type = INVALID_TOOL_CALL_FORMAT,
            failure_category = self.category(),
            finish_reason = self.finish_reason(),
            "The model wrote a tool call the reader could not read; the request ends as invalid_tool_call_format"
        );
    }
}

#[cfg(test)]
mod tests {
    use axum::{
        body::to_bytes,
        http::{header::CONTENT_TYPE, StatusCode},
    };

    use super::*;
    use crate::routers::{
        common::retry::is_retryable_response,
        grpc::utils::{error_reason_for_response, error_type_for_response},
    };

    const CUT_BY_THE_LIMIT: &str = "{\"error\":{\"message\":\"Treat this response as failed. The model stopped due to max_tokens before it completed the tool call. Retry the request.\",\"type\":\"invalid_tool_call_format\",\"code\":\"invalid_tool_call_format\",\"finish_reason\":\"length\"}}";

    /// The contract covers the answers the M3 table reads, and no other
    /// table's: MiniMax M2's answers on the same wire are read as before.
    #[test]
    fn only_the_m3_tables_answers_fail_on_an_unreadable_tool_call() {
        for table in Table::ALL {
            assert_eq!(
                fails_on_unreadable_tool_call(table.name()),
                table == Table::MinimaxM3,
                "{table:?}"
            );
        }
        assert!(!fails_on_unreadable_tool_call("other"));
    }

    /// The token limit is blamed only for a call it cut: an incomplete call
    /// under `length` is `length` with the max_tokens sentence; under the
    /// model's own stop, `error` with the incomplete-call sentence; a
    /// malformed block is `error` whatever the finish.
    #[test]
    fn the_finish_blames_the_token_limit_only_for_an_incomplete_call_it_cut() {
        let cut = InvalidToolCallFormat::new(UnreadableCall::Incomplete, true);
        assert_eq!(cut.finish_reason(), "length");
        assert_eq!(cut.category(), "incomplete_tool_call");
        assert_eq!(
            cut.message(),
            "Treat this response as failed. The model stopped due to max_tokens before it completed the tool call. Retry the request."
        );
        let stopped = InvalidToolCallFormat::new(UnreadableCall::Incomplete, false);
        assert_eq!(stopped.finish_reason(), "error");
        assert_eq!(
            stopped.message(),
            "Treat this response as failed. The model generated an incomplete tool call. Retry the request."
        );
        for ended_at_limit in [true, false] {
            let malformed = InvalidToolCallFormat::new(UnreadableCall::Malformed, ended_at_limit);
            assert_eq!(malformed.finish_reason(), "error");
            assert_eq!(malformed.category(), "malformed_structure");
            assert_eq!(
                malformed.message(),
                "Treat this response as failed. The model generated a tool call with malformed structure. Retry the request."
            );
        }
    }

    /// The whole response: a 400 under the code, the contract's body byte
    /// for byte, counted under the contract's label and the `bad_response`
    /// reason, not retried.
    #[tokio::test]
    async fn the_whole_response_is_a_400_with_the_contracts_body() {
        let response = InvalidToolCallFormat::new(UnreadableCall::Incomplete, true).response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response
                .headers()
                .get(error::HEADER_X_SMG_ERROR_CODE)
                .and_then(|v| v.to_str().ok()),
            Some(INVALID_TOOL_CALL_FORMAT)
        );
        assert_eq!(
            response
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
        assert_eq!(
            error_type_for_response(&response),
            "invalid_tool_call_format"
        );
        assert_eq!(error_reason_for_response(&response), "bad_response");
        assert!(!is_retryable_response(&response));
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(std::str::from_utf8(&body).unwrap(), CUT_BY_THE_LIMIT);
    }

    /// The stream's frame: a named event with the same body as the whole
    /// response.
    #[test]
    fn the_streams_frame_is_a_named_error_event_with_the_same_body() {
        let frame = InvalidToolCallFormat::new(UnreadableCall::Incomplete, true).sse_frame();
        assert_eq!(
            std::str::from_utf8(&frame).unwrap(),
            format!("event: error\ndata: {CUT_BY_THE_LIMIT}\n\n")
        );
    }
}
