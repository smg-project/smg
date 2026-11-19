//! The recorder of the classic pair index's outcomes: each row is a fleet of
//! prefill and decode workers, what each one offers a partner as the gateway
//! derives it from the worker's registration, and, on each of the gateway's
//! two pair wires under each of the three pairing modes, the verdict on every
//! prefill and decode of the fleet, the partners the index lists for each
//! prefill and the refusal it records when no pair is left. The gRPC wire's
//! requests are runtime-specific, so it holds two known runtimes apart in
//! every mode; the HTTP wire has no such fact. The policy controller crate
//! replays the rows through its own pair classes.
//!
//! This file is an ignored test of the gateway crate, kept beside the code
//! it runs. It records from the checkout it is built in and writes the sheet
//! under `CLASSIC_ROWS_OUT`, naming `CLASSIC_ROWS_COMMIT` as the commit it
//! recorded from. The sheets the policy controller crate replays are in
//! `crates/policy_controller/tests/classic/sheets`; to record this one again:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg --lib routers::common::pair_index::classic_pair_rows -- --ignored --nocapture
//! ```
//!
//! It is a child of the pair index's module and reads the descriptor the
//! gateway derived through the descriptor's own readers, field by field,
//! never a second reading of the labels.
//! Nothing in a row draws: the recorder builds each row twice and stops if the
//! two builds differ.

use std::{fs, path::PathBuf, sync::Arc};

use openai_protocol::{model_card::ModelCard, worker::WorkerSpec};
use serde::Serialize;
use smg_worker::{
    identity::basic::{ConnectionMode, RuntimeType, Worker, WorkerType},
    pd::pairing::{PdPairing, PdPairingMode},
    test_support::no_health_check,
};

use crate::routers::{
    common::pair_index::{PdPairIndex, PdWire},
    grpc::backend_client::BasicWorkerBuilder,
};

/// The label a transfer backend is reported under by the runtimes that name
/// one.
const BACKEND: &str = "disaggregation_transfer_backend";

/// What a worker's registration says of it: the facts the gateway derives its
/// pairing descriptor from.
#[derive(Clone, Copy)]
struct Reported {
    runtime: RuntimeType,
    /// The operator's pairing protocol on the worker's spec.
    protocol: Option<&'static str>,
    /// The KV connector on the worker's spec.
    connector: Option<&'static str>,
    /// The labels discovery read from the engine.
    labels: &'static [(&'static str, &'static str)],
}

impl Reported {
    const fn of(runtime: RuntimeType) -> Self {
        Self {
            runtime,
            protocol: None,
            connector: None,
            labels: &[],
        }
    }

    const fn protocol(mut self, protocol: &'static str) -> Self {
        self.protocol = Some(protocol);
        self
    }

    const fn connector(mut self, connector: &'static str) -> Self {
        self.connector = Some(connector);
        self
    }

    const fn labels(mut self, labels: &'static [(&'static str, &'static str)]) -> Self {
        self.labels = labels;
        self
    }
}

/// A worker of the first runtime that names its transport by a connector.
const fn first(connector: &'static str) -> Reported {
    Reported::of(RuntimeType::Vllm).connector(connector)
}

/// A worker of the first runtime that names no transport.
const fn first_silent() -> Reported {
    Reported::of(RuntimeType::Vllm)
}

/// A worker of the second runtime with its labels (the transfer backend is
/// one of them).
const fn second(labels: &'static [(&'static str, &'static str)]) -> Reported {
    Reported::of(RuntimeType::Sglang).labels(labels)
}

const NIXL: &str = "NixlConnector";
const MOONCAKE: &str = "MooncakeConnector";
const MOONCAKE_STORE: &str = "MooncakeStoreConnector";

/// The descriptor the gateway derived for one worker, in the classic's own
/// fields. `runtime` is absent for a worker whose runtime was not detected.
/// The layout facts stand in the order the classic compares them, each under
/// its canonical label. `key` is the group the classic names the worker by in
/// a refusal.
#[derive(Clone, PartialEq, Eq, Serialize)]
struct Descriptor {
    runtime: Option<&'static str>,
    protocol: Option<String>,
    transport: Option<String>,
    version: Option<String>,
    kv_layout: Vec<(&'static str, String)>,
    key: String,
}

impl Descriptor {
    fn of(pairing: &PdPairing) -> Self {
        Self {
            runtime: pairing
                .runtime()
                .is_specified()
                .then(|| pairing.runtime().as_str()),
            protocol: pairing.explicit().map(str::to_owned),
            transport: pairing.transport().map(str::to_owned),
            version: pairing.version().map(str::to_owned),
            kv_layout: pairing
                .kv_layout()
                .map(|(label, value)| (label, value.to_owned()))
                .collect(),
            key: pairing.key(),
        }
    }
}

/// The classic's refusal: each pool's distinct keys and the components the
/// pools disagreed on.
#[derive(Clone, PartialEq, Eq, Serialize)]
struct Refusal {
    prefill: Vec<String>,
    decode: Vec<String>,
    mismatches: Vec<String>,
}

/// What the classic decides for a fleet on one wire under one pairing mode.
#[derive(Clone, PartialEq, Eq, Serialize)]
struct Outcome {
    /// For every prefill, in the row's order, the verdict on each decode in
    /// the row's order: the first component the two disagree on, or nothing
    /// when they pair.
    verdicts: Vec<Vec<Option<String>>>,
    /// For every prefill, in the row's order, the decodes the index lists for
    /// it, each by its place in the row; empty for a prefill the index
    /// leaves out.
    partners: Vec<Vec<usize>>,
    /// Set when both pools have workers and the index holds no pair.
    refusal: Option<Refusal>,
}

/// A fleet's outcome on one wire under each pairing mode.
#[derive(Clone, PartialEq, Eq, Serialize)]
struct Modes {
    strict: Outcome,
    lenient: Outcome,
    off: Outcome,
}

/// A fleet's outcome on each of the gateway's pair wires.
#[derive(Clone, PartialEq, Eq, Serialize)]
struct Wires {
    /// The wire whose requests are runtime-specific.
    grpc: Modes,
    /// The wire that forwards one body to both sides.
    http: Modes,
}

#[derive(Serialize)]
struct Row {
    name: &'static str,
    prefills: Vec<Descriptor>,
    decodes: Vec<Descriptor>,
    outcome: Wires,
}

#[derive(Serialize)]
struct Sheet {
    recorded_from: String,
    policy: &'static str,
    rows: Vec<Row>,
}

fn worker(place: usize, role: WorkerType, reported: Reported) -> Arc<dyn Worker> {
    let side = if role == WorkerType::Prefill {
        "p"
    } else {
        "d"
    };
    let mut spec = WorkerSpec::new(format!("grpc://{side}{place}:9000"));
    spec.runtime_type = reported.runtime;
    spec.pairing_protocol = reported.protocol.map(str::to_string);
    spec.kv_connector = reported.connector.map(str::to_string);
    spec.labels = reported
        .labels
        .iter()
        .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
        .collect();
    Arc::new(
        BasicWorkerBuilder::from_spec(spec)
            .model(ModelCard::new("m"))
            .worker_type(role)
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check())
            .build(),
    )
}

fn pool(role: WorkerType, reported: &[Reported]) -> Vec<Arc<dyn Worker>> {
    reported
        .iter()
        .enumerate()
        .map(|(place, reported)| worker(place, role, *reported))
        .collect()
}

fn descriptors(pool: &[Arc<dyn Worker>]) -> Vec<Descriptor> {
    pool.iter()
        .map(|worker| Descriptor::of(worker.pd_pairing()))
        .collect()
}

/// A listed worker's place in its pool of the row.
fn place(pool: &[Arc<dyn Worker>], worker: &Arc<dyn Worker>) -> usize {
    pool.iter()
        .position(|candidate| Arc::ptr_eq(candidate, worker))
        .expect("the index lists the row's own workers")
}

/// The wire's verdict on one prefill and one decode under `mode`, read off
/// the index of that pair alone: its refusal names the one component the two
/// cannot pair on.
fn verdict(
    prefill: &Arc<dyn Worker>,
    decode: &Arc<dyn Worker>,
    wire: PdWire,
    mode: PdPairingMode,
) -> Option<String> {
    let pair = PdPairIndex::build(
        Arc::from([Arc::clone(prefill)]),
        Arc::from([Arc::clone(decode)]),
        wire,
        mode,
    );
    let refusal = pair.refusal?;
    assert_eq!(refusal.mismatches.len(), 1, "one pair, one component");
    refusal.mismatches.into_iter().next()
}

fn outcome(
    prefills: &[Arc<dyn Worker>],
    decodes: &[Arc<dyn Worker>],
    wire: PdWire,
    mode: PdPairingMode,
) -> Outcome {
    let verdicts = prefills
        .iter()
        .map(|prefill| {
            decodes
                .iter()
                .map(|decode| verdict(prefill, decode, wire, mode))
                .collect()
        })
        .collect();
    let index = PdPairIndex::build(
        Arc::from(prefills.to_vec()),
        Arc::from(decodes.to_vec()),
        wire,
        mode,
    );
    let mut partners = vec![Vec::new(); prefills.len()];
    for (listed, list) in index.prefill.iter().zip(&index.partners) {
        partners[place(prefills, listed)] =
            list.iter().map(|decode| place(decodes, decode)).collect();
    }
    let refusal = index.refusal.as_ref().map(|refusal| Refusal {
        prefill: refusal.prefill.clone(),
        decode: refusal.decode.clone(),
        mismatches: refusal.mismatches.clone(),
    });
    Outcome {
        verdicts,
        partners,
        refusal,
    }
}

/// One build of a row from fresh workers: the two pools' descriptors and the
/// outcome on each wire under each mode.
fn build(prefills: &[Reported], decodes: &[Reported]) -> (Vec<Descriptor>, Vec<Descriptor>, Wires) {
    let prefill_pool = pool(WorkerType::Prefill, prefills);
    let decode_pool = pool(WorkerType::Decode, decodes);
    let on = |wire| {
        let under = |mode| outcome(&prefill_pool, &decode_pool, wire, mode);
        Modes {
            strict: under(PdPairingMode::Strict),
            lenient: under(PdPairingMode::Lenient),
            off: under(PdPairingMode::Off),
        }
    };
    (
        descriptors(&prefill_pool),
        descriptors(&decode_pool),
        Wires {
            grpc: on(PdWire::Grpc),
            http: on(PdWire::Http),
        },
    )
}

fn row(name: &'static str, prefills: &[Reported], decodes: &[Reported]) -> Row {
    let (prefill_descriptors, decode_descriptors, outcome) = build(prefills, decodes);
    assert!(
        build(prefills, decodes)
            == (
                prefill_descriptors.clone(),
                decode_descriptors.clone(),
                outcome.clone()
            ),
        "{name}: two builds of the row differ"
    );
    Row {
        name,
        prefills: prefill_descriptors,
        decodes: decode_descriptors,
        outcome,
    }
}

fn rows() -> Vec<Row> {
    let undetected = Reported::of(RuntimeType::Unspecified);
    vec![
        // One descriptor on each side: every prefill lists every decode.
        row(
            "pair-one-descriptor-lists-every-decode",
            &[first(NIXL), first(NIXL)],
            &[first(NIXL), first(NIXL)],
        ),
        // Two transports: each prefill lists the decodes of its own.
        row(
            "pair-two-transports-list-apart",
            &[first(NIXL), first(MOONCAKE)],
            &[first(NIXL), first(MOONCAKE), first(NIXL)],
        ),
        // A prefill that names no transport: both transports' decodes
        // leniently, left out strictly.
        row(
            "pair-a-prefill-that-names-no-transport",
            &[first(NIXL), first(MOONCAKE), first_silent()],
            &[first(NIXL), first(MOONCAKE)],
        ),
        // A decode that names no transport stands on the list of each
        // transport's prefill: the lists overlap and neither holds the other.
        row(
            "pair-a-decode-that-names-no-transport-is-on-two-lists",
            &[first(NIXL), first(MOONCAKE)],
            &[first_silent(), first(NIXL), first(MOONCAKE)],
        ),
        // Nobody names a transport: a pair leniently, none strictly.
        row(
            "pair-nobody-names-a-transport",
            &[first_silent()],
            &[first_silent()],
        ),
        // No transport is shared: the refusal.
        row(
            "pair-no-shared-transport-is-refused",
            &[first(NIXL)],
            &[first(MOONCAKE)],
        ),
        // A store variant of a transport is its own transport.
        row(
            "pair-a-store-transport-pairs-with-its-own-kind-alone",
            &[first(MOONCAKE_STORE)],
            &[first(MOONCAKE), first(MOONCAKE_STORE)],
        ),
        // Two runtimes, each with a pair of its own: never a pair across
        // them while the mode compares anything.
        row(
            "pair-two-runtimes-each-list-their-own",
            &[first(NIXL), second(&[(BACKEND, "nixl")])],
            &[first(NIXL), second(&[(BACKEND, "nixl")])],
        ),
        // Two runtimes and no pair on one of them: refused, and paired
        // across the runtimes with pairing off.
        row(
            "pair-two-runtimes-and-no-pair-on-one-runtime",
            &[first(NIXL)],
            &[second(&[(BACKEND, "nixl")])],
        ),
        // A runtime that was not detected is unknown, not another runtime.
        row(
            "pair-an-undetected-runtime-on-one-side",
            &[first_silent()],
            &[undetected, first_silent()],
        ),
        row(
            "pair-an-undetected-runtime-on-both-sides",
            &[undetected],
            &[undetected],
        ),
        // Equal protocols settle the pair whatever the transports say, and
        // two protocols that differ refuse it.
        row(
            "pair-equal-protocols-settle-the-pair-over-the-transport",
            &[first(NIXL).protocol("blue")],
            &[
                first(MOONCAKE).protocol("blue"),
                first(NIXL).protocol("red"),
            ],
        ),
        // A protocol on one side alone: the facts decide leniently, the
        // pair is refused strictly.
        row(
            "pair-a-protocol-on-one-side-alone",
            &[first(NIXL).protocol("blue")],
            &[first(NIXL), first(MOONCAKE)],
        ),
        // The runtime is judged before the protocol.
        row(
            "pair-equal-protocols-do-not-cross-runtimes",
            &[first_silent().protocol("blue")],
            &[second(&[]).protocol("blue")],
        ),
        // One layout fact: equal pairs, another value never does, a side
        // that omits it pairs leniently alone.
        row(
            "pair-one-layout-fact",
            &[second(&[(BACKEND, "nixl"), ("kv_cache_dtype", "auto")])],
            &[
                second(&[(BACKEND, "nixl"), ("kv_cache_dtype", "auto")]),
                second(&[(BACKEND, "nixl"), ("kv_cache_dtype", "fp8_e5m2")]),
                second(&[(BACKEND, "nixl")]),
            ],
        ),
        // A layout fact neither side reports is not compared, strictly
        // either; one that a single side reports is unknown.
        row(
            "pair-a-layout-fact-one-side-reports",
            &[second(&[
                (BACKEND, "nixl"),
                ("kv_cache_dtype", "auto"),
                ("attention_backend", "flash_attn"),
            ])],
            &[
                second(&[(BACKEND, "nixl"), ("kv_cache_dtype", "auto")]),
                second(&[
                    (BACKEND, "nixl"),
                    ("kv_cache_dtype", "auto"),
                    ("attention_backend", "flash_attn"),
                ]),
            ],
        ),
        // Two layout facts differ in one pair: the verdict names one, the
        // first in the order the classic compares them.
        row(
            "pair-two-layout-facts-differ-and-one-is-named",
            &[second(&[
                (BACKEND, "nixl"),
                ("kv_cache_dtype", "auto"),
                ("page_size", "16"),
                ("attention_backend", "flash_attn"),
                ("model_dtype", "bfloat16"),
            ])],
            &[
                second(&[
                    (BACKEND, "nixl"),
                    ("kv_cache_dtype", "fp8_e5m2"),
                    ("page_size", "16"),
                    ("attention_backend", "xformers"),
                    ("model_dtype", "bfloat16"),
                ]),
                second(&[
                    (BACKEND, "nixl"),
                    ("kv_cache_dtype", "auto"),
                    ("page_size", "64"),
                    ("attention_backend", "flash_attn"),
                    ("model_dtype", "float16"),
                ]),
            ],
        ),
        // A version: compared strictly alone, and only where both report
        // one.
        row(
            "pair-a-version-counts-strictly-where-both-report-one",
            &[second(&[(BACKEND, "nixl"), ("version", "1")])],
            &[
                second(&[(BACKEND, "nixl"), ("version", "1")]),
                second(&[(BACKEND, "nixl"), ("version", "0")]),
                second(&[(BACKEND, "nixl")]),
            ],
        ),
        // Two prefills whose descriptors differ by the version alone list
        // the same decodes.
        row(
            "pair-two-versions-of-prefill-list-the-same-decodes",
            &[
                second(&[(BACKEND, "nixl"), ("version", "1")]),
                second(&[(BACKEND, "nixl"), ("version", "2")]),
            ],
            &[second(&[(BACKEND, "nixl")])],
        ),
        // A refusal names each component the pools disagreed on once.
        row(
            "pair-a-refusal-names-each-component-once",
            &[first(NIXL), second(&[(BACKEND, "nixl")])],
            &[first(MOONCAKE), Reported::of(RuntimeType::TokenSpeed)],
        ),
        // One pool alone is no refusal and no pair, in every mode.
        row("pair-prefills-and-no-decode", &[first(NIXL)], &[]),
        row("pair-decodes-and-no-prefill", &[], &[first(NIXL)]),
        // The lists keep the pools' order, and a prefill with no partner
        // leaves no gap in the others' lists.
        row(
            "pair-the-lists-keep-the-pools-order",
            &[
                first(MOONCAKE),
                first(MOONCAKE_STORE),
                first(NIXL),
                first_silent(),
                first(MOONCAKE),
            ],
            &[
                first(NIXL),
                first(MOONCAKE),
                first_silent(),
                first(NIXL),
                first(MOONCAKE),
            ],
        ),
    ]
}

#[test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
fn record_the_classic_pair_rows() {
    let out = PathBuf::from(
        std::env::var_os("CLASSIC_ROWS_OUT").expect("CLASSIC_ROWS_OUT names the output directory"),
    );
    let commit =
        std::env::var("CLASSIC_ROWS_COMMIT").expect("CLASSIC_ROWS_COMMIT names the commit");
    let sheet = Sheet {
        recorded_from: commit,
        policy: "pair-index",
        rows: rows(),
    };
    fs::create_dir_all(&out).expect("the output directory");
    let path = out.join("pair_index.json");
    let text = serde_json::to_string_pretty(&sheet).expect("the sheet serializes");
    fs::write(&path, text + "\n").expect("the sheet is written");
    println!("{} rows -> {}", sheet.rows.len(), path.display());
}
