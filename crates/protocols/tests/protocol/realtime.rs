//! Realtime request bodies, as the gateway writes them back for the
//! upstream on `/v1/realtime/client_secrets`, `/sessions` and
//! `/transcription_sessions`.

use openai_protocol::{
    realtime_response::RealtimeResponseCreateParams,
    realtime_session::{
        RealtimeSessionCreateRequest, RealtimeSessionCreateResponse,
        RealtimeTranscriptionSessionCreateRequest,
    },
};
use serde_json::json;

/// A realtime session keeps its reasoning effort and `parallel_tool_calls`,
/// and its input transcription's languages, keywords and delay, as the
/// public API takes them: the session is written back as sent.
#[test]
fn a_realtime_session_keeps_its_reasoning_and_transcription_hints() {
    let session = json!({
        "type": "realtime",
        "model": "gpt-realtime-2",
        "output_modalities": ["audio"],
        "parallel_tool_calls": false,
        "reasoning": {"effort": "high"},
        "audio": {"input": {"transcription": {
            "model": "gpt-4o-transcribe",
            "languages": ["en", "fr"],
            "keywords": ["Shepherd", "realtime"],
            "delay": "low"
        }}}
    });
    let request: RealtimeSessionCreateRequest = serde_json::from_value(session.clone()).unwrap();
    assert_eq!(serde_json::to_value(&request).unwrap(), session);

    let refused = serde_json::from_value::<RealtimeSessionCreateRequest>(json!({
        "type": "realtime",
        "model": "gpt-realtime-2",
        "reasoning": {"effort": "an_effort_of_no_api"}
    }));
    assert!(refused.is_err(), "an effort the public API does not name");
}

/// A transcription session shares the input transcription config, so it
/// keeps the hints too.
#[test]
fn a_transcription_session_keeps_its_transcription_hints() {
    let session = json!({
        "type": "transcription",
        "model": "gpt-4o-transcribe",
        "audio": {"input": {"transcription": {
            "model": "gpt-4o-transcribe",
            "languages": ["de"],
            "keywords": ["Kubernetes"],
            "delay": "xhigh"
        }}}
    });
    let request: RealtimeTranscriptionSessionCreateRequest =
        serde_json::from_value(session.clone()).unwrap();
    assert_eq!(serde_json::to_value(&request).unwrap(), session);
}

/// A `response.create` keeps its reasoning effort and `parallel_tool_calls`,
/// and an MCP tool its `tunnel_id`, `allowed_callers` and `defer_loading`,
/// as the public API takes them.
#[test]
fn a_response_create_keeps_its_reasoning_and_an_mcp_tools_keys() {
    let params = json!({
        "parallel_tool_calls": true,
        "reasoning": {"effort": "minimal"},
        "tools": [{
            "type": "mcp",
            "server_label": "internal-docs",
            "server_url": "https://mcp.internal.example.com/mcp",
            "tunnel_id": "tun_1",
            "allowed_callers": ["direct"],
            "defer_loading": true
        }]
    });
    let request: RealtimeResponseCreateParams = serde_json::from_value(params.clone()).unwrap();
    assert_eq!(serde_json::to_value(&request).unwrap(), params);
}

/// The session object the public API answers echoes the reasoning effort
/// (SDK `RealtimeSessionCreateResponse.reasoning`): it is read and written
/// back.
#[test]
fn a_session_answer_keeps_its_reasoning() {
    let answer: RealtimeSessionCreateResponse = serde_json::from_value(json!({
        "client_secret": {"value": "ek_test", "expires_at": 1_700_000_000},
        "type": "realtime",
        "model": "gpt-realtime-2",
        "reasoning": {"effort": "low"}
    }))
    .unwrap();
    assert_eq!(
        serde_json::to_value(&answer).unwrap()["reasoning"],
        json!({"effort": "low"})
    );
}
