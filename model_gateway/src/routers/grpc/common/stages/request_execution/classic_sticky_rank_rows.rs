//! The recorder of the gateway's sticky rank: each row is one grouped
//! data-parallel worker, several engines behind one connection and no rank
//! of its own, and the rank each request is pinned to under hash placement
//! (`--grouped-dp-sticky hash`). A row of steps lists its requests by their
//! sticky key and records the rank each carries after the gateway's pin,
//! `null` where the gateway sets none and the connection places the request
//! on any engine. A counted row asks `keys.count` keys, `key-0` upward: a
//! pairs row asks each twice and counts the keys whose two requests took one
//! rank, then the others; a spread row counts what each rank took, the
//! largest count first. The policy controller crate replays the rows over
//! the rank rows of one worker
//! (`crates/policy_controller/tests/classic/sticky_rank.rs`).
//!
//! This file is an ignored test of the gateway crate, kept beside the code
//! it runs. It records from the checkout it is built in and writes the sheet
//! under `CLASSIC_ROWS_OUT`, naming `CLASSIC_ROWS_COMMIT` as the commit it
//! recorded from. The sheets the policy controller crate replays are in
//! `crates/policy_controller/tests/classic/sheets`; to record this one again:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg --lib routers::grpc::common::stages::request_execution::classic_sticky_rank_rows -- --ignored --nocapture
//! ```
//!
//! Nothing in a row draws: a key's rank is its hash. The recorder runs each
//! row twice, each time with a fresh worker, and stops if the two runs
//! differ.

use std::{fs, path::PathBuf};

use serde::Serialize;
use smg_grpc_client::tokenspeed_proto as ts;
use smg_worker::identity::basic::{ConnectionMode, RuntimeType};

use super::*;
use crate::routers::grpc::backend_client::BasicWorkerBuilder;

const URL: &str = "ipc:///tmp/classic-sticky-rank.ipc";

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// A request with this sticky key, or with none.
    Pick { key: Option<&'static str> },
}

const fn pick(key: &'static str) -> Step {
    Step::Pick { key: Some(key) }
}

/// A counted row's keys, `key-0` to `key-<count - 1>`: asked twice each in a
/// pairs row, once each in a spread row.
#[derive(Clone, Copy, Serialize)]
struct Keys {
    count: u32,
    pairs: bool,
}

#[derive(PartialEq, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    /// The rank each request carries, in order: `null` where the gateway
    /// sets none.
    Sequence(Vec<Option<usize>>),
    /// A pairs row: the keys whose two requests took one rank, then the
    /// other keys. A spread row: what each rank took, the largest count
    /// first.
    Counts { draws: u32, counts: Vec<u32> },
}

#[derive(Serialize)]
struct Row {
    name: &'static str,
    /// The engines the worker names; `null` where it names no count.
    ranks: Option<usize>,
    /// The ranks the worker's connection places a pin on; `null` where the
    /// connection does not say.
    pinnable: Option<usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    steps: Vec<Step>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keys: Option<Keys>,
    outcome: Outcome,
}

#[derive(Serialize)]
struct Sheet {
    recorded_from: String,
    policy: &'static str,
    rows: Vec<Row>,
}

/// A fresh grouped worker of `ranks` engines, or one that names no count.
fn worker(ranks: Option<usize>) -> WorkerSelection {
    let builder = BasicWorkerBuilder::new(URL)
        .runtime_type(RuntimeType::TokenSpeed)
        .connection_mode(ConnectionMode::Zmq);
    let builder = match ranks {
        Some(ranks) => builder.zmq_engine_group(ranks),
        None => builder,
    };
    WorkerSelection::Single {
        worker: Arc::new(builder.build()).into(),
    }
}

/// The rank one single generate request with `key` carries after the
/// gateway's pin under hash placement.
fn pinned(worker: &WorkerSelection, pinnable: Option<usize>, key: Option<&str>) -> Option<usize> {
    let mut plan = ExecutionPlan::Single(ProtoRequest::Generate(ProtoGenerateRequest::TokenSpeed(
        Box::new(ts::GenerateRequest {
            request_id: "req".to_string(),
            ..Default::default()
        }),
    )));
    pin_grouped_dp_rank(&mut plan, worker, || pinnable, key, GroupedDpSticky::Hash);
    let ExecutionPlan::Single(ProtoRequest::Generate(ProtoGenerateRequest::TokenSpeed(request))) =
        &plan
    else {
        panic!("the plan stays a single generate request");
    };
    request
        .data_parallel_rank
        .map(|rank| usize::try_from(rank).expect("a rank is no negative number"))
}

fn sequence(ranks: Option<usize>, pinnable: Option<usize>, steps: &[Step]) -> Outcome {
    let worker = worker(ranks);
    Outcome::Sequence(
        steps
            .iter()
            .map(|&Step::Pick { key }| pinned(&worker, pinnable, key))
            .collect(),
    )
}

fn counted(ranks: usize, keys: Keys) -> Outcome {
    let worker = worker(Some(ranks));
    let rank = |i: u32| pinned(&worker, None, Some(&format!("key-{i}")));
    let counts = if keys.pairs {
        let together = (0..keys.count)
            .filter(|&i| rank(i).is_some_and(|first| rank(i) == Some(first)))
            .count();
        let together = u32::try_from(together).expect("no more keys than the row's count");
        vec![together, keys.count - together]
    } else {
        let mut counts = vec![0u32; ranks];
        for i in 0..keys.count {
            counts[rank(i).expect("a key of a grouped worker takes a rank")] += 1;
        }
        counts.sort_unstable_by(|a, b| b.cmp(a));
        counts
    };
    Outcome::Counts {
        draws: keys.count,
        counts,
    }
}

fn steps_row(
    name: &'static str,
    ranks: Option<usize>,
    pinnable: Option<usize>,
    steps: Vec<Step>,
) -> Row {
    let outcome = sequence(ranks, pinnable, &steps);
    assert_eq!(
        outcome,
        sequence(ranks, pinnable, &steps),
        "{name}: two runs of one row differ"
    );
    Row {
        name,
        ranks,
        pinnable,
        steps,
        keys: None,
        outcome,
    }
}

fn counted_row(name: &'static str, ranks: usize, keys: Keys) -> Row {
    let outcome = counted(ranks, keys);
    assert_eq!(
        outcome,
        counted(ranks, keys),
        "{name}: two runs of one row differ"
    );
    Row {
        name,
        ranks: Some(ranks),
        pinnable: None,
        steps: Vec::new(),
        keys: Some(keys),
        outcome,
    }
}

fn rows() -> Vec<Row> {
    let unkeyed = Step::Pick { key: None };
    vec![
        steps_row(
            "a-key-takes-one-rank-on-every-request",
            Some(8),
            None,
            vec![
                pick("conv-0"),
                pick("conv-1"),
                pick("conv-2"),
                pick("conv-0"),
                pick("conv-1"),
                pick("conv-2"),
            ],
        ),
        steps_row(
            "a-request-with-no-key-takes-no-rank",
            Some(8),
            None,
            vec![unkeyed, pick("conv-0"), unkeyed],
        ),
        steps_row(
            "a-worker-of-one-engine-takes-no-rank",
            Some(1),
            None,
            vec![pick("conv-0"), pick("conv-1")],
        ),
        steps_row(
            "a-worker-that-names-no-count-takes-no-rank-where-its-connection-says-none",
            None,
            None,
            vec![pick("conv-0"), pick("conv-1")],
        ),
        steps_row(
            "the-connection-s-ranks-stand-where-the-worker-names-no-count",
            None,
            Some(4),
            vec![
                pick("conv-0"),
                pick("conv-2"),
                pick("conv-0"),
                pick("conv-2"),
            ],
        ),
        steps_row(
            "a-key-whose-rank-the-connection-places-no-pin-on-takes-none",
            Some(8),
            Some(4),
            vec![
                pick("conv-0"),
                pick("conv-1"),
                pick("conv-2"),
                pick("conv-3"),
                pick("conv-4"),
                pick("conv-5"),
                pick("conv-6"),
                pick("conv-7"),
            ],
        ),
        counted_row(
            "every-key-keeps-one-rank",
            8,
            Keys {
                count: 512,
                pairs: true,
            },
        ),
        counted_row(
            "the-keys-spread-over-the-ranks",
            8,
            Keys {
                count: 4_096,
                pairs: false,
            },
        ),
    ]
}

#[test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
fn record_the_classic_sticky_rank_rows() {
    let out = PathBuf::from(
        std::env::var_os("CLASSIC_ROWS_OUT").expect("CLASSIC_ROWS_OUT names the output directory"),
    );
    let commit =
        std::env::var("CLASSIC_ROWS_COMMIT").expect("CLASSIC_ROWS_COMMIT names the commit");
    let sheet = Sheet {
        recorded_from: commit,
        policy: "sticky_rank",
        rows: rows(),
    };
    fs::create_dir_all(&out).expect("the output directory");
    let path = out.join("sticky_rank.json");
    let text = serde_json::to_string_pretty(&sheet).expect("the sheet serializes");
    fs::write(&path, text + "\n").expect("the sheet is written");
    println!("{} rows -> {}", sheet.rows.len(), path.display());
}
