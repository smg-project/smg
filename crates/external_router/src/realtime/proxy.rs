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
                                if let Ok(ev) = serde_json::from_str::<EventTypeOnly>(&text) {
                                    let et: &str = &ev.event_type;
                                    match et {
                                        "response.output_audio.delta"
                                        | "response.output_text.delta"
                                        | "response.output_audio_transcript.delta"
                                        | "response.function_call_arguments.delta" => {
                                            trace!(session_id, event_type = et, "Upstream→Client");
                                        }
                                        "session.created" | "session.updated"
                                        | "response.created" | "response.done"
                                        | "response.function_call_arguments.done"
                                        | "error" => {
                                            info!(session_id, event_type = et, "Upstream→Client");
                                        }
                                        _ => {
                                            debug!(session_id, event_type = et, "Upstream→Client");
                                        }
                                    }
                                }

                                if let Err(e) = client_sink.send(AxumMessage::Text(text.to_string().into())).await {
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
