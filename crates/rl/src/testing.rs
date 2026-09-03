//! Shared fakes for unit tests: an in-memory registry view and an in-process
//! fake engine that records what it receives.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use axum::{
    body::Bytes,
    extract::{Request, State},
    http::{HeaderMap, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::any,
    Json, Router,
};
use openai_protocol::worker::{ConnectionMode, RuntimeType, WorkerStatus, WorkerType};
use serde_json::json;
use tokio::net::TcpListener;

use crate::view::{RlWorkerInfo, RlWorkerView};

pub struct FakeView(pub Vec<RlWorkerInfo>);

impl RlWorkerView for FakeView {
    fn list(&self) -> Vec<RlWorkerInfo> {
        self.0.clone()
    }
    fn get(&self, id: &str) -> Option<RlWorkerInfo> {
        self.0.iter().find(|w| w.id == id).cloned()
    }
}

pub fn worker(id: &str, url: &str, runtime: RuntimeType) -> RlWorkerInfo {
    let mut labels = HashMap::new();
    labels.insert("tp_size".to_string(), "1".to_string());
    labels.insert("dp_size".to_string(), "1".to_string());
    labels.insert("pp_size".to_string(), "1".to_string());
    labels.insert("weight_version".to_string(), "default".to_string());
    RlWorkerInfo {
        id: id.to_string(),
        url: url.to_string(),
        base_url: url.split('@').next().unwrap_or(url).to_string(),
        api_key: None,
        model_id: "mock-model".to_string(),
        runtime,
        worker_type: WorkerType::Regular,
        connection_mode: ConnectionMode::Http,
        status: WorkerStatus::Ready,
        is_dp_aware: url.contains('@'),
        dp_size: None,
        labels,
    }
}

/// One recorded request seen by a fake engine.
#[derive(Debug, Clone)]
pub struct Seen {
    pub method: Method,
    pub path: String,
    pub query: Option<String>,
    pub headers: HeaderMap,
    pub body: Bytes,
}

#[derive(Clone)]
struct EngineState {
    seen: Arc<Mutex<Vec<Seen>>>,
    status: StatusCode,
    body: serde_json::Value,
    delay_ms: u64,
    in_flight: Arc<Mutex<(usize, usize)>>, // (current, max observed)
}

/// An in-process engine that answers every route with a fixed status/body,
/// records requests, and tracks peak concurrency.
pub struct FakeEngine {
    pub url: String,
    state: EngineState,
}

impl FakeEngine {
    #[expect(clippy::disallowed_methods, reason = "test server task")]
    pub async fn start(status: StatusCode, body: serde_json::Value, delay_ms: u64) -> Self {
        let state = EngineState {
            seen: Arc::new(Mutex::new(Vec::new())),
            status,
            body,
            delay_ms,
            in_flight: Arc::new(Mutex::new((0, 0))),
        };
        let app = Router::new()
            .route("/{*path}", any(handle))
            .with_state(state.clone());
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        Self {
            url: format!("http://{addr}"),
            state,
        }
    }

    pub fn seen(&self) -> Vec<Seen> {
        self.state.seen.lock().expect("seen").clone()
    }

    pub fn peak_concurrency(&self) -> usize {
        self.state.in_flight.lock().expect("in_flight").1
    }
}

async fn handle(State(st): State<EngineState>, req: Request) -> Response {
    {
        let mut g = st.in_flight.lock().expect("in_flight");
        g.0 += 1;
        g.1 = g.1.max(g.0);
    }
    let (parts, body) = req.into_parts();
    let uri: Uri = parts.uri;
    let bytes = axum::body::to_bytes(body, 1 << 20)
        .await
        .unwrap_or_default();
    st.seen.lock().expect("seen").push(Seen {
        method: parts.method,
        path: uri.path().to_string(),
        query: uri.query().map(str::to_string),
        headers: parts.headers,
        body: bytes,
    });
    if st.delay_ms > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(st.delay_ms)).await;
    }
    st.in_flight.lock().expect("in_flight").0 -= 1;
    (st.status, Json(json!(st.body))).into_response()
}
