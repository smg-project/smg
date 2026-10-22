//! The one vocabulary a disaggregated side's failure reaches the client in,
//! whichever transport carried the side.
//!
//! A prefill or decode side fails in one of three phases: it never started
//! (the call to its worker failed below HTTP, or the worker answered a status
//! before the side's first byte), it produced nothing within its liveness
//! bound, or its stream ended with an error after it began. The client's
//! `error.code` names the side and the phase; the status is the failure's own
//! class (a status the worker answered is kept as it is, so the retry layer
//! and a client's back-off read what the worker said; a call that failed
//! below HTTP is a bad gateway); the message opens with the same words. The
//! HTTP prefill-decode router and the gRPC prefill-decode executors build
//! their answers here, so a client, an alert or a runbook that keys on the
//! code reads one failure one way on an HTTP-mode and a gRPC-mode gateway
//! behind the same front door, and the two routers cannot drift apart again.
//!
//! A stream that fails after the response began cannot change its status:
//! both routers end it with the typed `stream_error` event (see the routers'
//! `stream_failure` modules); the codes here are for the answers that are
//! still whole.

use std::fmt::Display;

use axum::{http::StatusCode, response::Response};

use crate::{observability::metrics::metrics_labels, routers::error};

/// Which side of a disaggregated pair a failure belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PdSide {
    Prefill,
    Decode,
}

/// The phase a side failed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SidePhase {
    /// The side never started: the call to its worker failed below HTTP, or
    /// the worker answered a status before the side's first byte.
    Start,
    /// The side produced nothing within its liveness bound.
    Silent,
    /// The side's stream ended with an error after it began.
    Stream,
}

impl PdSide {
    /// The side's `worker_type` label, as its metrics and log lines name it.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Prefill => metrics_labels::WORKER_PREFILL,
            Self::Decode => metrics_labels::WORKER_DECODE,
        }
    }

    /// The other side of the pair.
    pub(crate) fn partner(self) -> Self {
        match self {
            Self::Prefill => Self::Decode,
            Self::Decode => Self::Prefill,
        }
    }

    /// The client's `error.code` for a failure of this side in `phase`.
    pub(crate) fn code(self, phase: SidePhase) -> &'static str {
        match (self, phase) {
            (Self::Prefill, SidePhase::Start) => "prefill_worker_failed_to_start",
            (Self::Decode, SidePhase::Start) => "decode_worker_failed_to_start",
            (Self::Prefill, SidePhase::Silent) => "prefill_worker_silent",
            (Self::Decode, SidePhase::Silent) => "decode_worker_silent",
            (Self::Prefill, SidePhase::Stream) => "prefill_stream_error",
            (Self::Decode, SidePhase::Stream) => "decode_stream_error",
        }
    }

    /// The words the message of a failure of this side in `phase` opens
    /// with; the detail (the worker's own words, the transport's reason)
    /// follows after a colon.
    pub(crate) fn description(self, phase: SidePhase) -> &'static str {
        match (self, phase) {
            (Self::Prefill, SidePhase::Start) => "Prefill worker failed to start",
            (Self::Decode, SidePhase::Start) => "Decode worker failed to start",
            (Self::Prefill, SidePhase::Silent) => "Prefill worker silent",
            (Self::Decode, SidePhase::Silent) => "Decode worker silent",
            (Self::Prefill, SidePhase::Stream) => "Prefill stream error",
            (Self::Decode, SidePhase::Stream) => "Decode stream error",
        }
    }

    /// The client's answer for a failure of this side in `phase` under
    /// `status`: the envelope every router answers its errors in, with the
    /// side's code, its words and `detail` after them.
    pub(crate) fn answer(
        self,
        phase: SidePhase,
        status: StatusCode,
        detail: impl Display,
    ) -> Response {
        error::create_error(
            status,
            self.code(phase),
            format!("{}: {detail}", self.description(phase)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vocabulary, row by row: the code names the side and the phase,
    /// the words match the code, and no two rows share a code.
    #[test]
    fn every_side_and_phase_has_its_own_code_and_words() {
        let table = [
            (
                PdSide::Prefill,
                SidePhase::Start,
                "prefill_worker_failed_to_start",
            ),
            (
                PdSide::Decode,
                SidePhase::Start,
                "decode_worker_failed_to_start",
            ),
            (PdSide::Prefill, SidePhase::Silent, "prefill_worker_silent"),
            (PdSide::Decode, SidePhase::Silent, "decode_worker_silent"),
            (PdSide::Prefill, SidePhase::Stream, "prefill_stream_error"),
            (PdSide::Decode, SidePhase::Stream, "decode_stream_error"),
        ];
        let mut codes = Vec::new();
        for side in [PdSide::Prefill, PdSide::Decode] {
            for phase in [SidePhase::Start, SidePhase::Silent, SidePhase::Stream] {
                let expected = table
                    .iter()
                    .find(|(s, p, _)| *s == side && *p == phase)
                    .map(|(_, _, code)| *code)
                    .expect("every side and phase is in the table");
                let code = side.code(phase);
                assert_eq!(code, expected);
                assert!(
                    code.starts_with(side.name()),
                    "{code} names {}",
                    side.name()
                );
                let words = side.description(phase);
                assert!(
                    words.to_ascii_lowercase().starts_with(side.name()),
                    "{words:?} names {}",
                    side.name()
                );
                codes.push(code);
            }
        }
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), 6, "six distinct codes");
    }

    /// The answer is the shared envelope under the given status, with the
    /// code in its header and body and the words before the detail.
    #[tokio::test]
    async fn the_answer_is_the_shared_envelope_under_the_failures_own_status() {
        let response = PdSide::Decode.answer(
            SidePhase::Start,
            StatusCode::SERVICE_UNAVAILABLE,
            "the engine is busy",
        );
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response
                .headers()
                .get(error::HEADER_X_SMG_ERROR_CODE)
                .and_then(|value| value.to_str().ok()),
            Some("decode_worker_failed_to_start")
        );
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let body: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(body["error"]["code"], "decode_worker_failed_to_start");
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(
            body["error"]["message"],
            "Decode worker failed to start: the engine is busy"
        );
    }
}
