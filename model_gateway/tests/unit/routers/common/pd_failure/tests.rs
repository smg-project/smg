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
