use axum::{extract::WebSocketUpgrade, routing::any, Router};
use tokio::net::TcpListener;

use super::*;

/// A provider's `error` event as its realtime WebSocket sends it: the
/// message names the operator's organization and project.
const ERROR_FRAME: &str = r#"{"type":"error","event_id":"event_1","error":{"type":"invalid_request_error","code":"rate_limit_exceeded","message":"Rate limit reached for realtime-model in organization org-abc123 project proj_xyz789 on tokens per min (TPM): Limit 40000, Used 39500, Requested 1000.","param":null,"event_id":null}}"#;

/// The same event as the client gets it: the message without the two
/// identifiers (the sanitizer's replacement word for word: the
/// preposition before the organization stays, as on the REST paths),
/// everything else in place.
const RELAYED_ERROR_FRAME: &str = r#"{"type":"error","event_id":"event_1","error":{"type":"invalid_request_error","code":"rate_limit_exceeded","message":"Rate limit reached for realtime-model in on tokens per min (TPM): Limit 40000, Used 39500, Requested 1000.","param":null,"event_id":null}}"#;

/// An `error` event whose message names nothing of the operator's.
const PLAIN_ERROR_FRAME: &str = r#"{"type":"error","event_id":"event_3","error":{"type":"invalid_request_error","code":"invalid_event","message":"The 'type' field is missing.","param":null,"event_id":"event_567"}}"#;

/// A transcription failure as the upstream sends it: the same `error`
/// object under another event type, its message naming the operator's
/// organization and project.
const TRANSCRIPTION_FAILED_FRAME: &str = r#"{"type":"conversation.item.input_audio_transcription.failed","event_id":"event_4","item_id":"item_2","content_index":0,"error":{"type":"transcription_error","code":"rate_limit_exceeded","message":"Rate limit reached for transcription-model in organization org-abc123 project proj_xyz789 on requests per min (RPM): Limit 50, Used 50, Requested 1.","param":null}}"#;

/// The same failure as the client gets it.
const RELAYED_TRANSCRIPTION_FAILED_FRAME: &str = r#"{"type":"conversation.item.input_audio_transcription.failed","event_id":"event_4","item_id":"item_2","content_index":0,"error":{"type":"transcription_error","code":"rate_limit_exceeded","message":"Rate limit reached for transcription-model in on requests per min (RPM): Limit 50, Used 50, Requested 1.","param":null}}"#;

/// A text delta as the upstream sends it; the text spells the very
/// patterns an error's message loses.
const DELTA_FRAME: &str = r#"{"type":"response.output_text.delta","event_id":"event_2","response_id":"resp_1","item_id":"item_1","output_index":0,"content_index":0,"delta":"in organization org-abc123 project proj_xyz789"}"#;

/// An `error` event reaches the client without the operator's
/// organization and project; its type, its id and the error's code and
/// words stay.
#[test]
fn an_error_event_loses_the_operators_identifiers_and_keeps_the_rest() {
    let relayed = relayed_text("s", ERROR_FRAME);
    assert!(!relayed.contains("org-abc123"), "{relayed}");
    assert!(!relayed.contains("proj_xyz789"), "{relayed}");
    assert_eq!(relayed, RELAYED_ERROR_FRAME);
}

/// An `error` event with nothing to remove reaches the client byte for
/// byte.
#[test]
fn an_error_event_without_identifiers_goes_as_it_came() {
    assert_eq!(relayed_text("s", PLAIN_ERROR_FRAME), PLAIN_ERROR_FRAME);
}

/// A transcription failure carries the same `error` object and loses
/// the operator's identifiers the same way.
#[test]
fn a_transcription_failure_loses_the_operators_identifiers_too() {
    let relayed = relayed_text("s", TRANSCRIPTION_FAILED_FRAME);
    assert!(!relayed.contains("org-abc123"), "{relayed}");
    assert!(!relayed.contains("proj_xyz789"), "{relayed}");
    assert_eq!(relayed, RELAYED_TRANSCRIPTION_FAILED_FRAME);
}

/// A frame that is not an `error` event is the upstream's text, borrowed
/// (read for its type alone, never re-serialized), whatever it spells;
/// so is a frame without a readable type.
#[test]
fn every_other_frame_goes_as_it_came() {
    for frame in [
        DELTA_FRAME,
        r#"{"type":"session.created","event_id":"event_0","session":{"id":"sess_1"}}"#,
        r#"{"event_id":"event_9"}"#,
        "not json",
    ] {
        let relayed = relayed_text("s", frame);
        assert!(matches!(relayed, Cow::Borrowed(_)), "{frame}");
        assert_eq!(relayed, frame);
    }
}

/// A recorded upstream session through the proxy itself: the upstream
/// WebSocket sends an `error` event, a binary frame, a transcription
/// failure and a text delta, then closes; the client behind the proxy
/// gets the two error-carrying events sanitized and the other two
/// frames as they came, in the upstream's order, then the close.
#[tokio::test]
async fn the_proxy_relays_an_error_event_sanitized_and_every_other_frame_as_it_came() {
    const AUDIO: &[u8] = &[0x00, 0x7f, 0x80, 0xff];

    let upstream = TcpListener::bind("127.0.0.1:0").await.expect("a port");
    let upstream_url = format!(
        "ws://{}/v1/realtime",
        upstream.local_addr().expect("an address")
    );
    #[expect(
        clippy::disallowed_methods,
        reason = "the test's upstream ends with the test"
    )]
    let upstream = tokio::spawn(async move {
        let (stream, _) = upstream.accept().await.expect("a connection");
        let mut ws = tokio_tungstenite::accept_async(stream)
            .await
            .expect("a handshake");
        for frame in [
            TungsteniteMessage::Text(ERROR_FRAME.into()),
            TungsteniteMessage::Binary(AUDIO.into()),
            TungsteniteMessage::Text(TRANSCRIPTION_FAILED_FRAME.into()),
            TungsteniteMessage::Text(DELTA_FRAME.into()),
        ] {
            ws.send(frame).await.expect("a sent frame");
        }
        let _ = ws.close(None).await;
    });

    let registry = Arc::new(RealtimeRegistry::new());
    let app = Router::new().route(
        "/ws",
        any(move |ws: WebSocketUpgrade| {
            let registry = Arc::clone(&registry);
            let upstream_url = upstream_url.clone();
            async move {
                ws.on_upgrade(move |socket| async move {
                    let _ = run_ws_proxy(
                        socket,
                        &upstream_url,
                        "Bearer k",
                        registry,
                        "s".into(),
                        CancellationToken::new(),
                    )
                    .await;
                })
            }
        }),
    );
    let gateway = TcpListener::bind("127.0.0.1:0").await.expect("a port");
    let gateway_url = format!("ws://{}/ws", gateway.local_addr().expect("an address"));
    #[expect(
        clippy::disallowed_methods,
        reason = "the test's gateway ends with the test"
    )]
    let gateway = tokio::spawn(async move {
        axum::serve(gateway, app).await.expect("a served gateway");
    });

    let (mut client, _) = tokio_tungstenite::connect_async(gateway_url)
        .await
        .expect("a connection");
    let mut frames = Vec::new();
    while let Some(frame) = client.next().await {
        match frame.expect("a frame") {
            TungsteniteMessage::Close(_) => break,
            frame => frames.push(frame),
        }
    }
    let [TungsteniteMessage::Text(error_event), TungsteniteMessage::Binary(audio), TungsteniteMessage::Text(failure), TungsteniteMessage::Text(delta)] =
        frames.as_slice()
    else {
        panic!("{frames:?}");
    };
    assert_eq!(error_event.as_str(), RELAYED_ERROR_FRAME);
    assert_eq!(audio.as_ref(), AUDIO);
    assert_eq!(failure.as_str(), RELAYED_TRANSCRIPTION_FAILED_FRAME);
    assert_eq!(delta.as_str(), DELTA_FRAME);

    gateway.abort();
    upstream.await.expect("the upstream sent its frames");
}
