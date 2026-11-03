//! WebSocket transport helpers for `/v1/realtime`.

use std::sync::Arc;

use axum::{
    extract::{
        ws::{WebSocket, WebSocketUpgrade},
        FromRequestParts,
    },
    http::{request::Parts, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use tracing::{debug, error};

use super::{proxy, RealtimeLabels, RealtimeRegistry};
use crate::{
    header_utils::extract_auth_header,
    metrics::{error_reason_for_refused_selection, metrics_labels, Metrics},
    worker::{redacted_url, ExternalWorker},
};

#[derive(Debug, Deserialize)]
pub struct RealtimeQueryParams {
    pub model: Option<String>,
}

/// Handle a realtime WebSocket upgrade request.
///
/// The caller is responsible for extracting the model from the query string
/// and selecting a worker. This function handles WS upgrade, auth resolution,
/// and proxy spawning.
pub async fn handle_realtime_ws(
    labels: RealtimeLabels,
    mut parts: Parts,
    model: String,
    worker: Result<Arc<dyn ExternalWorker>, Response>,
    auth_header: Option<HeaderValue>,
    realtime_registry: Arc<RealtimeRegistry>,
) -> Response {
    let worker = match worker {
        Ok(w) => w,
        Err(response) => {
            Metrics::record_router_error(
                labels.router,
                labels.backend,
                metrics_labels::CONNECTION_WEBSOCKET,
                &model,
                metrics_labels::ENDPOINT_REALTIME,
                metrics_labels::ERROR_NO_WORKERS,
                error_reason_for_refused_selection(response.status()),
            );
            return response;
        }
    };

    let worker_url = worker.url().to_string();
    let upstream_ws_url = build_upstream_ws_url(&worker_url, &model);

    // Use user auth if available, fall back to worker's API key
    let effective_auth = auth_header.or_else(|| extract_auth_header(None, worker.api_key()));
    let auth_str = match effective_auth {
        Some(v) => match v.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                Metrics::record_router_error(
                    labels.router,
                    labels.backend,
                    metrics_labels::CONNECTION_WEBSOCKET,
                    &model,
                    metrics_labels::ENDPOINT_REALTIME,
                    metrics_labels::ERROR_VALIDATION,
                    metrics_labels::REASON_INVALID_REQUEST,
                );
                return (
                    StatusCode::BAD_REQUEST,
                    "Authorization header contains invalid UTF-8 characters",
                )
                    .into_response();
            }
        },
        None => {
            Metrics::record_router_error(
                labels.router,
                labels.backend,
                metrics_labels::CONNECTION_WEBSOCKET,
                &model,
                metrics_labels::ENDPOINT_REALTIME,
                metrics_labels::ERROR_VALIDATION,
                metrics_labels::REASON_INVALID_REQUEST,
            );
            return StatusCode::UNAUTHORIZED.into_response();
        }
    };

    let ws = match WebSocketUpgrade::from_request_parts(&mut parts, &()).await {
        Ok(ws) => ws,
        Err(e) => {
            Metrics::record_router_error(
                labels.router,
                labels.backend,
                metrics_labels::CONNECTION_WEBSOCKET,
                &model,
                metrics_labels::ENDPOINT_REALTIME,
                metrics_labels::ERROR_VALIDATION,
                metrics_labels::REASON_INVALID_REQUEST,
            );
            return e.into_response();
        }
    };

    let session_id = uuid::Uuid::now_v7().to_string();
    let entry =
        realtime_registry.register_session(session_id.clone(), model.clone(), worker_url.clone());
    let cancel_token = entry.cancel_token.clone();

    tracing::info!(
        session_id,
        model,
        worker_url = %redacted_url(&worker_url),
        "Upgrading to realtime WebSocket"
    );

    ws.on_upgrade(move |socket: WebSocket| async move {
        match proxy::run_ws_proxy(
            socket,
            &upstream_ws_url,
            &auth_str,
            realtime_registry.clone(),
            session_id.clone(),
            cancel_token,
        )
        .await
        {
            Ok(()) => worker.record_outcome(200),
            Err(e) => {
                worker.record_outcome(502);
                error!(session_id, error = %e, "Realtime WebSocket proxy error");
            }
        }

        realtime_registry.remove_session(&session_id);
        debug!(session_id, "Realtime session cleaned up");
    })
}

/// Build the upstream WebSocket URL for the realtime endpoint.
///
/// Worker URLs use `http(s)://` but tungstenite requires `ws(s)://`,
/// e.g. `https://api.openai.com` → `wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview`
pub fn build_upstream_ws_url(worker_url: &str, model: &str) -> String {
    let base = worker_url.trim_end_matches('/');
    let ws_base = if let Some(rest) = base.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = base.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        base.to_string()
    };
    let query = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("model", model)
        .finish();
    format!("{ws_base}/v1/realtime?{query}")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::{extract::Request, http::HeaderValue, routing::any, Router};
    use futures_util::StreamExt;
    use tokio::net::TcpListener;
    use tokio_tungstenite::tungstenite::Message as TungsteniteMessage;
    use tracing::Instrument;

    use super::{build_upstream_ws_url, handle_realtime_ws, RealtimeLabels, RealtimeRegistry};
    use crate::test_support::{StubWorker, CREDENTIALED_WORKER_URL, MASKED_WORKER_URL};

    #[test]
    fn https_becomes_wss() {
        assert_eq!(
            build_upstream_ws_url("https://api.openai.com", "gpt-4o-realtime-preview"),
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
        );
    }

    #[test]
    fn http_becomes_ws() {
        assert_eq!(
            build_upstream_ws_url("http://127.0.0.1:8000", "qwen3-asr"),
            "ws://127.0.0.1:8000/v1/realtime?model=qwen3-asr"
        );
    }

    #[test]
    fn trailing_slash_trimmed() {
        assert_eq!(
            build_upstream_ws_url("http://worker:9000/", "m"),
            "ws://worker:9000/v1/realtime?model=m"
        );
    }

    #[test]
    fn model_is_query_encoded() {
        assert_eq!(
            build_upstream_ws_url("https://x", "gpt 4o"),
            "wss://x/v1/realtime?model=gpt+4o"
        );
    }

    /// The upgrade line names the worker masked, read through a served
    /// route: a client upgrades, the proxy's connection to a worker
    /// registered with credentials at a port nobody listens on fails, and no
    /// line the handler wrote carries the credentials. The handler runs on
    /// the server's task, so it borrows the test's span for the capture to
    /// read its lines.
    #[tracing_test::traced_test]
    #[tokio::test]
    async fn the_upgrade_line_names_the_worker_masked() {
        let registry = Arc::new(RealtimeRegistry::new());
        let span = tracing::Span::current();
        let app = Router::new().route(
            "/ws",
            any(move |request: Request| {
                let registry = Arc::clone(&registry);
                let span = span.clone();
                async move {
                    let (parts, _body) = request.into_parts();
                    handle_realtime_ws(
                        RealtimeLabels::OPENAI,
                        parts,
                        "m".to_string(),
                        Ok(StubWorker::at(CREDENTIALED_WORKER_URL)),
                        Some(HeaderValue::from_static("Bearer k")),
                        registry,
                    )
                    .await
                }
                .instrument(span)
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
            .expect("an upgraded connection");
        // The proxy cannot reach the worker and ends the client's socket.
        while let Some(frame) = client.next().await {
            if matches!(frame, Err(_) | Ok(TungsteniteMessage::Close(_))) {
                break;
            }
        }
        gateway.abort();

        assert!(!logs_contain("secret"));
        assert!(!logs_contain("user:"));
        assert!(logs_contain("Upgrading to realtime WebSocket"));
        assert!(logs_contain(MASKED_WORKER_URL));
    }
}
