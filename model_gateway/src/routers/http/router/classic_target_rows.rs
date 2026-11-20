//! The recorder of the gateway's explicit target: a request that names a
//! worker by `x-smg-target-worker` goes to that worker, or to none. The
//! consistent hashing policy holds the rule, and the sheet is
//! `explicit_target`.
//!
//! Each row is a pool of regular HTTP workers, each with its state, whether
//! an all-overloaded pool is shed, and a list of requests, each with the
//! header's value. The row records what the typed line's own selection
//! answered at each request: the worker it placed the request on, or its
//! refusal. The policy controller crate replays the rows through its own
//! decision.
//!
//! The header's value is an index into the workers the selection hands the
//! policy, counted when the request arrives. Beside a worker's state the
//! sheet writes what the gateway reads of it for that list (`available`).
//! The pool runs under `consistent_hashing`, and a request is placed sixteen
//! times the pool's size, as the other demand rows are: `placed` shows the
//! one worker a named request lands on. A refusal is the response's status,
//! its code, whether it says when to come back, and its message.
//!
//! This file is an ignored test of the gateway crate, kept beside the code
//! it runs. It records from the checkout it is built in and writes the sheet
//! under `CLASSIC_ROWS_OUT`, naming `CLASSIC_ROWS_COMMIT` as the commit it
//! recorded from. The sheets the policy controller crate replays are in
//! `crates/policy_controller/tests/classic/sheets`; to record this one again:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg --lib router::classic_target_rows -- --ignored --nocapture
//! ```
//!
//! Every row is recorded twenty times, each time from a fresh pool and a
//! fresh policy, and the recorder stops if two recordings differ.

use std::{collections::BTreeSet, fs, path::PathBuf};

use axum::http::header::RETRY_AFTER;
use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use serde_json::{json, Value};
use smg_worker::{identity::basic::ConnectionMode, test_support::no_health_check};

use super::*;
use crate::{
    app_context::testing::app_context_for_tests,
    config::{types::PolicyConfig, RouterConfig},
    routers::{
        common::request_lease::test_probe::DropProbeRequest,
        grpc::backend_client::BasicWorkerBuilder,
    },
};

const MODEL: &str = "m";
/// A request is placed this many times the pool's size.
const ROUNDS: usize = 16;

/// What the router holds of a worker.
#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    /// Ready and under its overload thresholds.
    Free,
    /// Ready and over its overload thresholds.
    Over,
    /// Not ready.
    Down,
}

impl State {
    const fn name(self) -> &'static str {
        match self {
            Self::Free => "free",
            Self::Over => "over",
            Self::Down => "down",
        }
    }
}

/// One row as it is written down: its workers' states, and the header's
/// value at each request.
struct Row {
    name: &'static str,
    /// Whether an all-overloaded pool is shed (`--worker-overload-shed`).
    shed: bool,
    workers: Vec<State>,
    targets: Vec<&'static str>,
}

fn row(name: &'static str, workers: &[State], targets: &[&'static str]) -> Row {
    Row {
        name,
        shed: false,
        workers: workers.to_vec(),
        targets: targets.to_vec(),
    }
}

impl Row {
    fn shedding(self) -> Self {
        Self { shed: true, ..self }
    }
}

/// A fresh pool in the row's state, under a fresh policy.
struct Bed {
    router: Router,
    workers: Vec<Arc<dyn Worker>>,
}

impl Bed {
    async fn of(row: &Row) -> Self {
        let worker_registry = Arc::new(WorkerRegistry::new());
        worker_registry.set_overload_shed(row.shed);
        let mut workers = Vec::new();
        for (at, state) in row.workers.iter().enumerate() {
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(format!("http://127.0.0.1:{}", 9600 + at))
                    .model(ModelCard::new(MODEL))
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Http)
                    .health_config(no_health_check())
                    .build(),
            );
            worker.set_status(WorkerStatus::Ready);
            worker_registry
                .register(Arc::clone(&worker))
                .expect("a worker of the row registers");
            worker.set_status(if *state == State::Down {
                WorkerStatus::NotReady
            } else {
                WorkerStatus::Ready
            });
            worker_registry.set_worker_overloaded(&worker, *state == State::Over);
            workers.push(worker);
        }
        // The router is built as the gateway builds it, by its own
        // constructor over a context: a field the router gains is filled
        // there, and this file names none of them.
        let config = RouterConfig {
            policy: PolicyConfig::ConsistentHashing,
            ..RouterConfig::default()
        };
        let context = app_context_for_tests(config, worker_registry, None);
        let router = Router::new(&context)
            .await
            .expect("the router builds over a test context");
        Self { router, workers }
    }

    /// The typed line's answer to one request that names `target`: the
    /// workers its selection placed the request on over the rounds, or its
    /// refusal.
    async fn answer(&self, target: &str) -> Value {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-smg-target-worker",
            HeaderValue::from_str(target).expect("a header's value"),
        );
        let request = DropProbeRequest {
            text: "a prompt".to_string(),
            _probe: Arc::new(()),
        };
        let routing = self.router.routing_derivatives(Some(&headers), &request);
        let lease = RequestLease::new(
            request,
            routing,
            ReleasePoint::from_retry_config(&self.router.retry_config),
        );
        let tried = parking_lot::Mutex::new(Vec::new());
        let tried_rows = parking_lot::Mutex::new(Vec::new());
        let ledger = AttemptLedger::default();
        let attempt = TypedAttempt {
            headers: Some(&headers),
            lease: &lease,
            route: "/generate",
            model_id: MODEL,
            canonical_model: None,
            is_stream: false,
            ledger: &ledger,
            tried: &tried,
            tried_rows: &tried_rows,
            rid: None,
        };
        let mut placed = BTreeSet::new();
        for _ in 0..ROUNDS * self.workers.len().max(1) {
            match self.router.select_worker_for_attempt(&attempt) {
                Ok(worker) => {
                    let at = self
                        .workers
                        .iter()
                        .position(|w| w.url() == worker.url())
                        .expect("a worker of the row");
                    placed.insert(at);
                }
                Err(refusal) => {
                    assert!(placed.is_empty(), "a request placed once is placed again");
                    return refused(refusal).await;
                }
            }
        }
        json!({"placed": placed.into_iter().collect::<Vec<_>>()})
    }
}

/// A refusal as the sheet writes it.
async fn refused(refusal: Response) -> Value {
    let status = refusal.status().as_u16();
    let code = extract_error_code_from_response(&refusal).to_string();
    let retry_after = refusal.headers().contains_key(RETRY_AFTER);
    let body = to_bytes(refusal.into_body(), usize::MAX)
        .await
        .expect("a refusal's body is read");
    let message = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|body| body["error"]["message"].as_str().map(str::to_string))
        .unwrap_or_default();
    json!({"refused": {"status": status, "code": code, "retry_after": retry_after, "message": message}})
}

/// One recording of a row: its workers as the gateway reads them, and the
/// answer to each request.
async fn recorded(row: &Row) -> Value {
    let bed = Bed::of(row).await;
    let workers: Vec<Value> = row
        .workers
        .iter()
        .zip(&bed.workers)
        .map(|(state, worker)| json!({"state": state.name(), "available": worker.is_available()}))
        .collect();
    let requests: Vec<Value> = row
        .targets
        .iter()
        .map(|target| json!({"target": target}))
        .collect();
    let mut answers = Vec::new();
    for target in &row.targets {
        answers.push(bed.answer(target).await);
    }
    json!({
        "name": row.name,
        "shed": row.shed,
        "workers": workers,
        "requests": requests,
        "answers": answers,
    })
}

fn rows() -> Vec<Row> {
    use State::{Down, Free, Over};
    vec![
        row(
            "a-named-worker-takes-the-request",
            &[Free, Free, Free],
            &["0", "1", "2"],
        ),
        row(
            "an-index-past-the-list-names-nobody",
            &[Free, Free],
            &["2", "7"],
        ),
        row(
            "a-value-that-is-no-index-names-nobody",
            &[Free, Free],
            &["x", "-1"],
        ),
        row(
            "a-worker-that-is-out-moves-the-indexes-behind-it",
            &[Free, Down, Free],
            &["0", "1", "2"],
        ),
        row("the-one-worker-is-down", &[Down], &["0"]),
        row("a-model-no-worker-serves", &[], &["0"]),
        row(
            "a-named-worker-over-its-thresholds-beside-one-under-them",
            &[Over, Free],
            &["0", "1"],
        ),
        row(
            "every-worker-is-over-its-thresholds",
            &[Over, Over],
            &["0", "1", "2"],
        ),
        row(
            "every-worker-is-over-and-shedding-is-on",
            &[Over, Over],
            &["0", "1"],
        )
        .shedding(),
    ]
}

#[tokio::test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
async fn record_the_classic_target_rows() {
    let out = PathBuf::from(
        std::env::var_os("CLASSIC_ROWS_OUT").expect("CLASSIC_ROWS_OUT names the output directory"),
    );
    let commit =
        std::env::var("CLASSIC_ROWS_COMMIT").expect("CLASSIC_ROWS_COMMIT names the commit");
    fs::create_dir_all(&out).expect("the output directory");
    let mut written = Vec::new();
    for row in rows() {
        let first = recorded(&row).await;
        for _ in 1..20 {
            assert_eq!(
                recorded(&row).await,
                first,
                "{}: two recordings differ",
                row.name
            );
        }
        println!("ROW {}: {}", row.name, first["answers"]);
        written.push(first);
    }
    let path = out.join("explicit_target.json");
    let text = serde_json::to_string_pretty(&json!({
        "recorded_from": commit,
        "policy": "explicit-target",
        "rows": written,
    }))
    .expect("the sheet serializes");
    fs::write(&path, text + "\n").expect("the sheet is written");
    println!("{} rows -> {}", written.len(), path.display());
}
