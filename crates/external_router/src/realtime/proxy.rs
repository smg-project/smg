//! Bidirectional WebSocket proxy between client (axum) and upstream (tungstenite).

use std::{borrow::Cow, sync::Arc};

use axum::extract::ws::{Message as AxumMessage, WebSocket};
use futures::{
    stream::{SplitSink, SplitStream},
    SinkExt, StreamExt,
};
use serde::Deserialize;
use tokio::net::TcpStream;
use tokio_tungstenite::{tungstenite::Message as TungsteniteMessage, MaybeTlsStream};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, trace, warn};

use super::registry::{ConnectionState, RealtimeRegistry};
use crate::error;

type UpstreamWs = tokio_tungstenite::WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Lightweight deserializer that extracts only the `"type"` field from a
/// realtime event JSON payload. Avoids the cost of fully deserializing
/// `ClientEvent`/`ServerEvent` (which box large variants like `SessionConfig`
/// at 624 B) on every frame — critical for high-frequency audio streams.
///
/// Uses `Cow<str>` so serde can borrow zero-copy for plain strings while
/// still handling JSON escape sequences (e.g. `\u002E`) that require allocation.
#[derive(Deserialize)]
struct EventTypeOnly<'a> {
    #[serde(rename = "type", borrow)]
    event_type: Cow<'a, str>,
}

/// Run a bidirectional WebSocket proxy between a client and upstream.
///
/// Returns when either side closes or an error occurs.
pub async fn run_ws_proxy(
    client_ws: WebSocket,
    upstream_url: &str,
    auth_header: &str,
    registry: Arc<RealtimeRegistry>,
    session_id: String,
    cancel_token: CancellationToken,
) -> anyhow::Result<()> {
    // Connect to upstream WebSocket with auth.
    // Let tungstenite auto-add WebSocket handshake headers (Connection, Upgrade,
    // Sec-WebSocket-Version, Sec-WebSocket-Key); we only add app-specific headers.
    //
    // Do not send `OpenAI-Beta: realtime=v1` — OpenAI's GA Realtime API rejects
    // it with `beta_api_shape_disabled` ("The Realtime Beta API is no longer
    // supported. Please use /v1/realtime for the GA API.").
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    let mut request = upstream_url.into_client_request()?;
    request
        .headers_mut()
        .insert("Authorization", auth_header.parse()?);

    // Build an explicit rustls TLS connector so we don't depend on the
    // process-level CryptoProvider being installed.
    let connector = get_tls_connector();

    let (upstream_ws, _response) = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio_tungstenite::connect_async_tls_with_config(request, None, false, Some(connector)),
    )
    .await
    .map_err(|_| anyhow::anyhow!("upstream WebSocket connect timed out after 10s"))??;

    registry.set_session_state(&session_id, ConnectionState::Connected);
    debug!(session_id, "Upstream WebSocket connected");

    let (client_sink, client_stream) = client_ws.split();
    let (upstream_sink, upstream_stream) = upstream_ws.split();

    let cancel_c2u = cancel_token.clone();
    let cancel_u2c = cancel_token.clone();
    let session_id_c2u = session_id.clone();
    let session_id_u2c = session_id.clone();

    #[expect(
        clippy::disallowed_methods,
        reason = "forward tasks cancelled via token on session end"
    )]
    let client_to_upstream = tokio::spawn(forward_client_to_upstream(
        client_stream,
        upstream_sink,
        cancel_c2u,
        session_id_c2u,
    ));

    #[expect(
        clippy::disallowed_methods,
        reason = "forward tasks cancelled via token on session end"
    )]
    let upstream_to_client = tokio::spawn(forward_upstream_to_client(
        upstream_stream,
        client_sink,
        cancel_u2c,
        session_id_u2c,
    ));

    // Wait for either task to finish (or cancellation)
    tokio::select! {
        result = client_to_upstream => {
            cancel_token.cancel();
            debug!(session_id, "Client→upstream task ended");
            if let Err(e) = result {
                error!(session_id, error = %e, "Client→upstream task panicked");
            }
        }
        result = upstream_to_client => {
            cancel_token.cancel();
            debug!(session_id, "Upstream→client task ended");
            if let Err(e) = result {
                error!(session_id, error = %e, "Upstream→client task panicked");
            }
        }
        () = cancel_token.cancelled() => {
            debug!(session_id, "Session cancelled via token");
        }
    }

    registry.set_session_state(&session_id, ConnectionState::Disconnected);
    debug!(session_id, "WebSocket proxy session ended");
    Ok(())
}

/// Forward messages from client (axum) to upstream (tungstenite).
async fn forward_client_to_upstream(
    mut client_stream: SplitStream<WebSocket>,
    mut upstream_sink: SplitSink<UpstreamWs, TungsteniteMessage>,
    cancel: CancellationToken,
    session_id: String,
) {
    loop {
        tokio::select! {
            msg = client_stream.next() => {
                match msg {
                    Some(Ok(axum_msg)) => {
                        let tungstenite_msg = match axum_msg {
                            AxumMessage::Text(text) => {
                                if let Ok(ev) = serde_json::from_str::<EventTypeOnly>(&text) {
                                    let et: &str = &ev.event_type;
                                    match et {
                                        "input_audio_buffer.append" => {
                                            trace!(session_id, event_type = et, "Client→Upstream");
                                        }
                                        _ => {
                                            debug!(session_id, event_type = et, "Client→Upstream");
                                        }
                                    }
                                }

                                TungsteniteMessage::Text(text.to_string().into())
                            }
                            AxumMessage::Binary(data) => TungsteniteMessage::Binary(data),
                            AxumMessage::Ping(data) => TungsteniteMessage::Ping(data),
                            AxumMessage::Pong(data) => TungsteniteMessage::Pong(data),
                            AxumMessage::Close(_) => {
                                let _ = upstream_sink.close().await;
                                return;
                            }
                        };
                        if let Err(e) = upstream_sink.send(tungstenite_msg).await {
                            warn!(session_id, error = %e, "Failed to send to upstream");
                            return;
                        }
                    }
                    Some(Err(e)) => {
                        warn!(session_id, error = %e, "Client WebSocket error");
                        return;
                    }
                    None => {
                        debug!(session_id, "Client WebSocket closed");
                        let _ = upstream_sink.close().await;
                        return;
                    }
                }
            }
            () = cancel.cancelled() => return,
        }
    }
}

/// Forward messages from upstream (tungstenite) to client (axum).
async fn forward_upstream_to_client(
    mut upstream_stream: SplitStream<UpstreamWs>,
    mut client_sink: SplitSink<WebSocket, AxumMessage>,
    cancel: CancellationToken,
    session_id: String,
) {
    loop {
        tokio::select! {
            msg = upstream_stream.next() => {
                match msg {
                    Some(Ok(tungstenite_msg)) => {
                        match tungstenite_msg {
                            TungsteniteMessage::Text(text) => {
                                let text = relayed_text(&session_id, &text);
                                if let Err(e) = client_sink.send(AxumMessage::Text(text.into_owned().into())).await {
                                    warn!(session_id, error = %e, "Failed to send to client");
                                    return;
                                }
                            }
                            TungsteniteMessage::Binary(data) => {
                                if let Err(e) = client_sink.send(AxumMessage::Binary(data)).await {
                                    warn!(session_id, error = %e, "Failed to send binary to client");
                                    return;
                                }
                            }
                            TungsteniteMessage::Ping(data) => {
                                if let Err(e) = client_sink.send(AxumMessage::Ping(data)).await {
                                    warn!(session_id, error = %e, "Failed to send ping to client");
                                    return;
                                }
                            }
                            TungsteniteMessage::Pong(data) => {
                                if let Err(e) = client_sink.send(AxumMessage::Pong(data)).await {
                                    warn!(session_id, error = %e, "Failed to send pong to client");
                                    return;
                                }
                            }
                            TungsteniteMessage::Close(_) => {
                                let _ = client_sink.close().await;
                                return;
                            }
                            TungsteniteMessage::Frame(_) => {
                                // Raw frames — ignore
                            }
                        }
                    }
                    Some(Err(e)) => {
                        warn!(session_id, error = %e, "Upstream WebSocket error");
                        return;
                    }
                    None => {
                        debug!(session_id, "Upstream WebSocket closed");
                        let _ = client_sink.close().await;
                        return;
                    }
                }
            }
            () = cancel.cancelled() => return,
        }
    }
}

/// The client's copy of an upstream text frame. The event's type is read
/// alone, as it was to pick the log level; an `error` event, and a
/// transcription failure, which carries the same `error` object, are the
/// frames whose `error.message` may name the provider's organization and
/// project (the operator's, not the caller's), so those frames alone are
/// parsed and lose them through the replacement the REST relays apply
/// ([`error::sanitize_error_body`]). Every other frame is the upstream's
/// text as it came, never re-serialized.
fn relayed_text<'a>(session_id: &str, text: &'a str) -> Cow<'a, str> {
    let Ok(ev) = serde_json::from_str::<EventTypeOnly>(text) else {
        return Cow::Borrowed(text);
    };
    let et: &str = &ev.event_type;
    match et {
        "response.output_audio.delta"
        | "response.output_text.delta"
        | "response.output_audio_transcript.delta"
        | "response.function_call_arguments.delta" => {
            trace!(session_id, event_type = et, "Upstream→Client");
        }
        "error" => {
            info!(session_id, event_type = et, "Upstream→Client");
            return Cow::Owned(error::sanitize_error_body(text));
        }
        "conversation.item.input_audio_transcription.failed" => {
            debug!(session_id, event_type = et, "Upstream→Client");
            return Cow::Owned(error::sanitize_error_body(text));
        }
        "session.created"
        | "session.updated"
        | "response.created"
        | "response.done"
        | "response.function_call_arguments.done" => {
            info!(session_id, event_type = et, "Upstream→Client");
        }
        _ => {
            debug!(session_id, event_type = et, "Upstream→Client");
        }
    }
    Cow::Borrowed(text)
}

/// Return a cached rustls-backed TLS connector for upstream WebSocket connections.
///
/// The `ClientConfig` is built once and reused across all connections. Uses
/// the `ring` crypto provider explicitly so we don't depend on the
/// process-level `CryptoProvider::install_default()` having been called.
fn get_tls_connector() -> tokio_tungstenite::Connector {
    static TLS_CONFIG: std::sync::OnceLock<Arc<rustls::ClientConfig>> = std::sync::OnceLock::new();

    let config = TLS_CONFIG.get_or_init(|| {
        use rustls::{crypto::ring, ClientConfig};

        let root_store = rustls::RootCertStore {
            roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
        };
        // INVARIANT: `ring::default_provider()` supports rustls default TLS protocol versions.
        #[expect(
            clippy::expect_used,
            reason = "ring infallibly supports default TLS versions"
        )]
        let builder = ClientConfig::builder_with_provider(Arc::new(ring::default_provider()))
            .with_safe_default_protocol_versions()
            .expect("ring supports default TLS protocol versions");
        Arc::new(
            builder
                .with_root_certificates(root_store)
                .with_no_client_auth(),
        )
    });

    tokio_tungstenite::Connector::Rustls(Arc::clone(config))
}

#[cfg(test)]
mod tests {
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
}
