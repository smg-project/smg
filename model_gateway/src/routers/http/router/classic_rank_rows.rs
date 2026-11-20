//! The recorder of the gateway's rank rule: a request on a route that cannot
//! pin a data-parallel rank goes to no worker that registered with one. The
//! route is the multipart transcription, the one place the routers hold the
//! rule. The sheet is `rank_pin`.
//!
//! Each row is a pool of regular HTTP workers, each with its state and the
//! rank and group size it registered with, if any, whether an all-overloaded
//! pool is shed, and a list of requests. The row records what the route's
//! own selection answered at each request: the workers it placed the request
//! on, or its refusal. The policy controller crate replays the rows through
//! its own decision.
//!
//! The pool runs under `round_robin`, and a request is placed sixteen times
//! the pool's size: a rotation walks every worker the rule leaves it, and a
//! pool over on every such worker, steered by a draw among the fewest in
//! flight, reaches each of them. So `placed` is the whole set a request may
//! land on, by each worker's place in the row's list. A refusal is the
//! response's status, its code, whether it says when to come back, and its
//! message. Beside a worker's registration the sheet writes what the gateway
//! reads of it (`rank_registered`): a group of one still registers a rank.
//!
//! This file is an ignored test of the gateway crate, kept beside the code
//! it runs. It records from the checkout it is built in and writes the sheet
//! under `CLASSIC_ROWS_OUT`, naming `CLASSIC_ROWS_COMMIT` as the commit it
//! recorded from. The sheets the policy controller crate replays are in
//! `crates/policy_controller/tests/classic/sheets`; to record this one again:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg --lib router::classic_rank_rows -- --ignored --nocapture
//! ```
//!
//! Every row is recorded twenty times, each time from a fresh pool and a
//! fresh policy, and the recorder stops if two recordings differ.

use std::{collections::BTreeSet, fs, path::PathBuf};

use axum::http::header::RETRY_AFTER;
use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use serde_json::{json, Map, Value};
use smg_worker::{identity::basic::ConnectionMode, test_support::no_health_check};

use super::*;
use crate::{
    app_context::testing::app_context_for_tests,
    config::{types::PolicyConfig, RouterConfig},
    routers::grpc::backend_client::BasicWorkerBuilder,
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

/// One worker of a row, as it registers: its state, and the rank and the
/// group size of its registration, if it names a rank.
#[derive(Clone, Copy)]
struct Member {
    state: State,
    rank: Option<(usize, usize)>,
}

const PLAIN: Member = Member {
    state: State::Free,
    rank: None,
};

const fn ranked(rank: usize, of: usize) -> Member {
    Member {
        state: State::Free,
        rank: Some((rank, of)),
    }
}

impl Member {
    const fn state(self, state: State) -> Self {
        Self { state, ..self }
    }
}

/// One row as it is written down. A row's requests carry nothing the rule
/// reads: each is one transcription of the row's model.
struct Row {
    name: &'static str,
    /// Whether an all-overloaded pool is shed (`--worker-overload-shed`).
    shed: bool,
    workers: Vec<Member>,
    requests: usize,
}

fn row(name: &'static str, workers: &[Member]) -> Row {
    Row {
        name,
        shed: false,
        workers: workers.to_vec(),
        requests: 1,
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
        for (at, member) in row.workers.iter().enumerate() {
            let mut builder = BasicWorkerBuilder::new(format!("http://127.0.0.1:{}", 9500 + at))
                .model(ModelCard::new(MODEL))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Http)
                .health_config(no_health_check());
            if let Some((rank, of)) = member.rank {
                builder = builder.dp_config(rank, of);
            }
            let worker: Arc<dyn Worker> = Arc::new(builder.build());
            worker.set_status(WorkerStatus::Ready);
            worker_registry
                .register(Arc::clone(&worker))
                .expect("a worker of the row registers");
            worker.set_status(if member.state == State::Down {
                WorkerStatus::NotReady
            } else {
                WorkerStatus::Ready
            });
            worker_registry.set_worker_overloaded(&worker, member.state == State::Over);
            workers.push(worker);
        }
        // The router is built as the gateway builds it, by its own
        // constructor over a context: a field the router gains is filled
        // there, and this file names none of them.
        let config = RouterConfig {
            policy: PolicyConfig::RoundRobin,
            ..RouterConfig::default()
        };
        let context = app_context_for_tests(config, worker_registry, None);
        let router = Router::new(&context)
            .await
            .expect("the router builds over a test context");
        Self { router, workers }
    }

    /// A worker as the sheet writes it: its registration's words, and what
    /// the gateway reads of them for the rule.
    fn worker_json(&self, at: usize, member: &Member) -> Value {
        let mut out = Map::new();
        out.insert("state".into(), json!(member.state.name()));
        if let Some((rank, of)) = member.rank {
            out.insert("dp_rank".into(), json!(rank));
            out.insert("dp_size".into(), json!(of));
        }
        out.insert(
            "rank_registered".into(),
            json!(self.workers[at].is_dp_aware()),
        );
        Value::Object(out)
    }

    /// The route's answer to one request: the workers its selection placed
    /// the request on over the rounds, or its refusal.
    async fn answer(&self) -> Value {
        let routing = RoutingDerivatives {
            tokens: None,
            text: None,
            rid_key: None,
            cache_namespace: None,
        };
        let mut placed = BTreeSet::new();
        for _ in 0..ROUNDS * self.workers.len().max(1) {
            match self
                .router
                .select_transcription_worker(None, MODEL, &routing)
            {
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
        .enumerate()
        .map(|(at, member)| bed.worker_json(at, member))
        .collect();
    let mut answers = Vec::new();
    for _ in 0..row.requests {
        answers.push(bed.answer().await);
    }
    json!({
        "name": row.name,
        "shed": row.shed,
        "workers": workers,
        "answers": answers,
    })
}

fn rows() -> Vec<Row> {
    vec![
        row(
            "a-request-no-rank-can-be-pinned-for-leaves-the-workers-that-registered-one",
            &[PLAIN, ranked(0, 2), ranked(1, 2), PLAIN],
        ),
        row(
            "a-group-of-one-still-registered-a-rank",
            &[ranked(0, 1), PLAIN],
        ),
        row(
            "a-pool-with-no-rank-takes-the-request-whole",
            &[PLAIN, PLAIN],
        ),
        row(
            "every-worker-registered-a-rank",
            &[ranked(0, 2), ranked(1, 2)],
        ),
        row("one-worker-and-it-registered-a-rank", &[ranked(0, 1)]),
        row("a-transcription-of-a-model-no-worker-serves", &[]),
        row(
            "the-worker-with-no-rank-is-down",
            &[PLAIN.state(State::Down), ranked(0, 2), ranked(1, 2)],
        ),
        row(
            "a-worker-with-no-rank-over-its-thresholds-yields-to-one-under-them",
            &[PLAIN.state(State::Over), PLAIN, ranked(0, 1)],
        ),
        row(
            "the-workers-with-no-rank-are-over-their-thresholds",
            &[
                PLAIN.state(State::Over),
                PLAIN.state(State::Over),
                ranked(0, 1),
            ],
        ),
        row(
            "the-workers-with-no-rank-are-over-and-shedding-is-on",
            &[
                PLAIN.state(State::Over),
                PLAIN.state(State::Over),
                ranked(0, 1),
            ],
        )
        .shedding(),
        row(
            "a-worker-with-a-rank-that-is-down-still-counts-as-one",
            &[ranked(0, 2).state(State::Down), ranked(1, 2)],
        ),
    ]
}

#[tokio::test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
async fn record_the_classic_rank_rows() {
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
    let path = out.join("rank_pin.json");
    let text = serde_json::to_string_pretty(&json!({
        "recorded_from": commit,
        "policy": "rank-pin",
        "rows": written,
    }))
    .expect("the sheet serializes");
    fs::write(&path, text + "\n").expect("the sheet is written");
    println!("{} rows -> {}", written.len(), path.display());
}
