//! The recorder of the gateway's answer to a request of the encode, prefill
//! and decode form. Each row is a fleet (encoders, prefill workers and decode
//! workers, with what the gateway reads of each from its registration and the
//! requests it holds on each), the policies the fleet runs under and a list
//! of requests. The row records what the gateway's own worker selection
//! answered at each request: the pair it picked, the encoders it offered the
//! request's items, the key each item was placed by and the encoder that took
//! it, or the name of its refusal. The policy controller crate replays the
//! rows through its own decision.
//!
//! A worker is named by its place in its list. An encoder's `handoff` is the
//! form its runtime hands an encoded item to a prefill worker in, as the
//! gateway's table has it: a push to the prefill worker, a meeting at a
//! rendezvous, or none. `offered` and `keys` are written down by a wrapper
//! around the encode policy, which passes every call on unchanged: they are
//! what the selection handed that policy, in its order.
//!
//! No row draws. Each side of the pair is picked by `power_of_two` under its
//! `least_requests` metric (the metric follows the policy's name after a
//! colon), and no row leaves that policy more than two workers to pick from,
//! or two that hold the same number of requests: of two workers it takes the
//! one that holds fewer. An item's encoder is picked the same way, or by
//! `consistent_hashing`, the gateway's own choice for encode items when none
//! is configured. The selection holds no request on a worker, so the numbers
//! a row states stand through its requests. Every row is recorded twenty
//! times, each time from a fresh fleet and fresh policies, and the recorder
//! stops if two recordings differ.
//!
//! This file is an ignored test of the gateway crate, kept beside the code
//! it runs. It records from the checkout it is built in and writes the sheet
//! under `CLASSIC_ROWS_OUT`, naming `CLASSIC_ROWS_COMMIT` as the commit it
//! recorded from. The sheets the policy controller crate replays are in
//! `crates/policy_controller/tests/classic/sheets`; to record this one again:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg --lib worker_selection::classic_encode_pair_rows -- --ignored --nocapture
//! ```

use std::{
    fs,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use serde::{Serialize, Serializer};
use smg_worker::{identity::basic::ConnectionMode, test_support::no_health_check};

use super::*;
use crate::{
    config::types::PolicyConfig,
    policies::PolicyFactory,
    routers::{common::snapshot::WorkerRegistryExt, grpc::backend_client::BasicWorkerBuilder},
};

const MODEL: &str = "m";

/// The runtime whose encoder pushes an encoded item to the prefill worker.
const PUSH: RuntimeType = RuntimeType::Vllm;
/// The runtime whose encoder and prefill worker meet at a rendezvous.
const MEET: RuntimeType = RuntimeType::TokenSpeed;
/// A runtime that serves a prefill and decode pair and states no encode
/// handoff.
const BARE: RuntimeType = RuntimeType::Sglang;

/// The policies, in their configuration spelling; the load metric follows
/// `power_of_two` after a colon.
const BY_COUNT: &str = "power_of_two:least_requests";
const RING: &str = "consistent_hashing";

const C1: &str = "c1";
const C2: &str = "c2";

/// The policy's configuration, as the gateway's own configuration file
/// spells it.
fn config_of(policy: &str) -> PolicyConfig {
    let text = match policy.split_once(':') {
        Some((kind, metric)) => format!(r#"{{"type":"{kind}","load_metric":"{metric}"}}"#),
        None => format!(r#"{{"type":"{policy}"}}"#),
    };
    serde_json::from_str(&text).expect("the policy's configuration")
}

/// What a worker's registration says of it (its runtime and the labels the
/// selection reads), whether the worker is ready and the requests the router
/// holds on it.
#[derive(Clone)]
struct Reg {
    runtime: RuntimeType,
    labels: Vec<(&'static str, &'static str)>,
    open: bool,
    in_flight: u32,
}

impl Reg {
    fn of(runtime: RuntimeType, labels: &[(&'static str, &'static str)]) -> Self {
        Self {
            runtime,
            labels: labels.to_vec(),
            open: true,
            in_flight: 0,
        }
    }

    /// The same worker with `label` set to `value`.
    fn with(mut self, label: &'static str, value: &'static str) -> Self {
        self.labels.retain(|(name, _)| *name != label);
        self.labels.push((label, value));
        self
    }

    /// The same worker without `label`.
    fn without(mut self, label: &'static str) -> Self {
        self.labels.retain(|(name, _)| *name != label);
        self
    }

    /// The same worker, not ready.
    fn closed(mut self) -> Self {
        self.open = false;
        self
    }

    /// The same worker, holding `in_flight` requests.
    fn holding(mut self, in_flight: u32) -> Self {
        self.in_flight = in_flight;
        self
    }
}

/// An encoder of the pushing runtime: a producer on `connector` that takes
/// images and processes references itself.
fn push_encoder(connector: &'static str) -> Reg {
    Reg::of(
        PUSH,
        &[
            ("supports_ec_transfer", "true"),
            ("ec_role", "ec_producer"),
            ("ec_connector", connector),
            ("mm_processor", "inprocess"),
            ("supports_vision", "true"),
        ],
    )
}

/// A prefill worker of the pushing runtime: a consumer on `connector`.
fn push_prefill(connector: &'static str) -> Reg {
    Reg::of(
        PUSH,
        &[
            ("supports_ec_transfer", "true"),
            ("ec_role", "ec_consumer"),
            ("ec_connector", connector),
            ("kv_connector", "NixlConnector"),
        ],
    )
}

/// A decode worker of the pushing runtime.
fn push_decode() -> Reg {
    Reg::of(PUSH, &[("kv_connector", "NixlConnector")])
}

/// A worker of the rendezvous runtime, in any role: it reports no label.
fn meet() -> Reg {
    Reg::of(MEET, &[])
}

/// A worker of the runtime with no encode handoff, in any role.
fn bare() -> Reg {
    Reg::of(BARE, &[])
}

#[derive(Clone)]
struct Fleet {
    encoders: Vec<Reg>,
    prefills: Vec<Reg>,
    decodes: Vec<Reg>,
}

fn fleet(encoders: &[Reg], prefills: &[Reg], decodes: &[Reg]) -> Fleet {
    Fleet {
        encoders: encoders.to_vec(),
        prefills: prefills.to_vec(),
        decodes: decodes.to_vec(),
    }
}

/// `fleet` with the listed encoders not ready.
fn closing(fleet: &Fleet, encoders: &[usize]) -> Fleet {
    let mut fleet = fleet.clone();
    for &at in encoders {
        fleet.encoders[at].open = false;
    }
    fleet
}

/// What the gateway reads of an encoder.
#[derive(Clone, PartialEq, Eq, Serialize)]
struct Encoder {
    runtime: &'static str,
    /// The form the runtime hands an encoded item over in: `push`,
    /// `rendezvous`, or nothing.
    handoff: Option<&'static str>,
    /// The registration states the producer role of a connector.
    ec_producer: bool,
    connector: Option<String>,
    /// The encoder takes images.
    images: bool,
    /// The encoder fetches and processes references itself.
    references: bool,
    /// The worker is ready.
    open: bool,
    /// The requests the router holds on the worker.
    in_flight: usize,
}

/// What the gateway reads of a prefill worker.
#[derive(Clone, PartialEq, Eq, Serialize)]
struct Prefill {
    runtime: &'static str,
    /// The registration states the consumer role of a connector.
    ec_consumer: bool,
    connector: Option<String>,
    open: bool,
    in_flight: usize,
    /// The decode workers the gateway pairs it with, by their places.
    partners: Vec<usize>,
}

/// What the gateway reads of a decode worker.
#[derive(Clone, PartialEq, Eq, Serialize)]
struct Decode {
    runtime: &'static str,
    /// The registration does not say the worker runs without a vision
    /// encoder.
    images: bool,
    open: bool,
    in_flight: usize,
}

/// One request: its encode items by their keys and what it asks of the
/// workers.
#[derive(Clone, Serialize)]
struct Request {
    /// The key of each encode item (its content hash), in hexadecimal.
    #[serde(serialize_with = "in_hex")]
    items: Vec<Vec<u8>>,
    /// The items are references an encoder fetches and processes itself.
    references: bool,
    /// The items carry images.
    images: bool,
    /// The decode side must take images.
    decode_images: bool,
    /// Each item is encoded on its own. Without it a pushing runtime takes
    /// the request's items as one.
    fanout: bool,
}

fn in_hex<S: Serializer>(items: &[Vec<u8>], serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_seq(items.iter().map(|item| hex_encode(item)))
}

/// A request whose items have the one-byte `keys`: it carries images, each
/// item encoded on its own.
fn items(keys: &[u8]) -> Request {
    Request {
        items: keys.iter().map(|&key| vec![key]).collect(),
        references: false,
        images: true,
        decode_images: false,
        fanout: true,
    }
}

impl Request {
    fn references(mut self) -> Self {
        self.references = true;
        self
    }

    fn no_images(mut self) -> Self {
        self.images = false;
        self
    }

    fn decode_images(mut self) -> Self {
        self.decode_images = true;
        self
    }

    #[expect(
        clippy::wrong_self_convention,
        reason = "a builder step, named for what the request becomes"
    )]
    fn as_one(mut self) -> Self {
        self.fanout = false;
        self
    }
}

/// What the gateway answered a request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Answer {
    Placed {
        prefill: usize,
        decode: usize,
        /// The encoders the items were placed over, in the order the encode
        /// policy was handed them.
        offered: Vec<usize>,
        /// The key each item was placed by, in hexadecimal.
        keys: Vec<String>,
        /// The encoder each key was placed on.
        encoders: Vec<usize>,
    },
    Refused {
        name: &'static str,
        /// The side of the pair a refused placement names.
        #[serde(skip_serializing_if = "Option::is_none")]
        role: Option<String>,
        /// Why that side was refused.
        #[serde(skip_serializing_if = "Option::is_none")]
        why: Option<&'static str>,
    },
}

fn refusal(failure: EpdSelectionFailure) -> Answer {
    let named = |name: &'static str| Answer::Refused {
        name,
        role: None,
        why: None,
    };
    match failure {
        EpdSelectionFailure::Unavailable => named("unavailable"),
        EpdSelectionFailure::EcIncompatible => named("ec_incompatible"),
        EpdSelectionFailure::DecodeVisionUnsupported => named("decode_vision_unsupported"),
        EpdSelectionFailure::EncodeNotSupported => named("encode_not_supported"),
        EpdSelectionFailure::EncodeVisionUnsupported => named("encode_vision_unsupported"),
        // The recorder starts no policy engine, so no row of its sheet
        // records this refusal. It has its name all the same.
        EpdSelectionFailure::EncodeUnderPolicyEngine => named("encode_under_policy_engine"),
        EpdSelectionFailure::Placement(failure) => Answer::Refused {
            name: "placement",
            role: Some(format!("{:?}", failure.role).to_lowercase()),
            why: Some(match failure.verdict {
                PlacementFailure::NoCandidates => "no_candidates",
                PlacementFailure::AllOverloaded(_) => "overloaded",
                PlacementFailure::Unavailable => "unavailable",
                PlacementFailure::PolicyDeclined(_) => "declined",
                PlacementFailure::NoCompatiblePair { .. } => "no_compatible_pair",
                PlacementFailure::PrefillAtCapacity => "at_capacity",
            }),
        },
    }
}

/// What a row recorded: the answer to each of its requests, in order. Every
/// row of this sheet is a sequence row.
#[derive(Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    Sequence(Vec<Answer>),
}

/// One call of the encode policy as the selection made it.
#[derive(Debug)]
struct Call {
    /// The encoders it was handed, by address.
    offered: Vec<String>,
    /// The routing key it was handed.
    key: String,
    /// The place it answered in `offered`.
    picked: Option<usize>,
}

/// The encode policy behind a wrapper that writes down each call and passes
/// it on unchanged.
#[derive(Debug)]
struct Watched {
    inner: Arc<dyn LoadBalancingPolicy>,
    calls: Mutex<Vec<Call>>,
}

impl LoadBalancingPolicy for Watched {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let key = info
            .headers
            .and_then(|headers| headers.get("x-smg-routing-key"))
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let picked = self.inner.select_worker(workers, info);
        self.calls.lock().expect("the calls").push(Call {
            offered: workers.iter().map(|w| w.url().to_string()).collect(),
            key,
            picked,
        });
        picked
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn honors_routing_key(&self) -> bool {
        self.inner.honors_routing_key()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn url(role: WorkerType, at: usize) -> String {
    let base = match role {
        WorkerType::Encode => 9100,
        WorkerType::Prefill => 9200,
        _ => 9300,
    };
    format!("grpc://127.0.0.1:{}", base + at)
}

fn place_of(workers: &[Arc<dyn Worker>], url: &str) -> usize {
    workers
        .iter()
        .position(|worker| worker.url() == url)
        .expect("a worker of the row")
}

/// A fresh registry holding a row's fleet, under fresh policies.
struct Bed {
    stage: WorkerSelectionStage,
    watched: Arc<Watched>,
    encoders: Vec<Arc<dyn Worker>>,
    prefills: Vec<Arc<dyn Worker>>,
    decodes: Vec<Arc<dyn Worker>>,
}

impl Bed {
    fn new(encode_policy: &str, fleet: &Fleet) -> Self {
        let registry = Arc::new(WorkerRegistry::new());
        let pool = |role: WorkerType, workers: &[Reg]| -> Vec<Arc<dyn Worker>> {
            workers
                .iter()
                .enumerate()
                .map(|(at, reg)| {
                    let address = url(role, at);
                    let mut builder = BasicWorkerBuilder::new(address.clone())
                        .model(ModelCard::new(MODEL))
                        .worker_type(role)
                        .runtime_type(reg.runtime)
                        .connection_mode(ConnectionMode::Grpc)
                        .health_config(no_health_check());
                    for &(label, value) in &reg.labels {
                        builder = builder.label(label, value);
                    }
                    registry
                        .register(Arc::new(builder.build()))
                        .expect("the worker registers");
                    let worker = registry.get_by_url(&address).expect("the worker is held");
                    for _ in 0..reg.in_flight {
                        worker.increment_load();
                    }
                    if !reg.open {
                        worker.set_status(WorkerStatus::NotReady);
                    }
                    worker
                })
                .collect()
        };
        let encoders = pool(WorkerType::Encode, &fleet.encoders);
        let prefills = pool(WorkerType::Prefill, &fleet.prefills);
        let decodes = pool(WorkerType::Decode, &fleet.decodes);
        let watched = Arc::new(Watched {
            inner: PolicyFactory::create_from_config(&config_of(encode_policy)),
            calls: Mutex::new(Vec::new()),
        });
        let policies = PolicyRegistry::new(config_of("random"));
        policies.set_prefill_policy(PolicyFactory::create_from_config(&config_of(BY_COUNT)));
        policies.set_decode_policy(PolicyFactory::create_from_config(&config_of(BY_COUNT)));
        policies.set_encode_policy(Arc::clone(&watched) as Arc<dyn LoadBalancingPolicy>);
        let stage = WorkerSelectionStage::new(
            registry,
            Arc::new(policies),
            WorkerSelectionMode::EncodePrefillDecode,
            None,
        );
        Self {
            stage,
            watched,
            encoders,
            prefills,
            decodes,
        }
    }

    /// What the gateway reads of each worker of the fleet.
    fn facts(&self) -> (Vec<Encoder>, Vec<Prefill>, Vec<Decode>) {
        let runtime = |worker: &Arc<dyn Worker>| worker.metadata().spec.runtime_type;
        let connector = |worker: &Arc<dyn Worker>| {
            worker
                .metadata()
                .spec
                .labels
                .get("ec_connector")
                .filter(|value| !value.is_empty())
                .cloned()
        };
        let pairs = self
            .stage
            .worker_registry
            .get_routing_snapshot(MODEL)
            .pd_pairs(PdWire::Grpc, self.stage.policy_registry.pd_pairing_mode());
        let encoders = self
            .encoders
            .iter()
            .map(|worker| Encoder {
                runtime: runtime(worker).as_str(),
                handoff: match PdProtocol::encode_handoff(runtime(worker)) {
                    Some(EncodeHandoff::Push) => Some("push"),
                    Some(EncodeHandoff::Rendezvous) => Some("rendezvous"),
                    None => None,
                },
                ec_producer: ec_role(worker.as_ref(), "ec_producer"),
                connector: connector(worker),
                images: vision_capable(worker.as_ref()),
                references: accepts_media_refs(worker.as_ref()),
                open: worker.is_available(),
                in_flight: worker.load(),
            })
            .collect();
        let prefills = self
            .prefills
            .iter()
            .map(|worker| Prefill {
                runtime: runtime(worker).as_str(),
                ec_consumer: ec_role(worker.as_ref(), "ec_consumer"),
                connector: connector(worker),
                open: worker.is_available(),
                in_flight: worker.load(),
                partners: pairs
                    .prefill
                    .iter()
                    .position(|listed| listed.url() == worker.url())
                    .map(|at| {
                        pairs.partners[at]
                            .iter()
                            .map(|decode| place_of(&self.decodes, decode.url()))
                            .collect()
                    })
                    .unwrap_or_default(),
            })
            .collect();
        let decodes = self
            .decodes
            .iter()
            .map(|worker| Decode {
                runtime: runtime(worker).as_str(),
                images: vision_capable(worker.as_ref()),
                open: worker.is_available(),
                in_flight: worker.load(),
            })
            .collect();
        (encoders, prefills, decodes)
    }

    /// One request through the gateway's selection of an encode, prefill and
    /// decode set.
    fn answer(&self, request: &Request) -> Answer {
        self.watched.calls.lock().expect("the calls").clear();
        let picked = self.stage.select_encode_prefill_decode_workers(
            MODEL,
            PlacementInputs::default(),
            &request.items,
            EpdDemands {
                wire: None,
                encode: MediaDemands {
                    media_refs: request.references,
                    vision: request.images,
                    pipeline: PipelinePin::Any,
                },
                decode_vision: request.decode_images,
                image_fanout: request.fanout,
            },
            None,
        );
        let calls = std::mem::take(&mut *self.watched.calls.lock().expect("the calls"));
        let (assignments, prefill, decode, _runtime) = match picked {
            Ok(picked) => picked,
            Err(failure) => return refusal(failure),
        };
        let offered = calls
            .first()
            .map(|call| call.offered.clone())
            .unwrap_or_default();
        assert_eq!(
            calls.len(),
            assignments.len(),
            "one call of the encode policy for each key placed"
        );
        let mut keys = Vec::with_capacity(calls.len());
        let mut encoders = Vec::with_capacity(calls.len());
        for (at, (call, assignment)) in calls.iter().zip(&assignments).enumerate() {
            assert_eq!(assignment.item_index, at, "the keys are placed in order");
            assert_eq!(call.offered, offered, "one request, one list of encoders");
            assert_eq!(
                call.picked.map(|picked| call.offered[picked].as_str()),
                Some(assignment.worker.url()),
                "the encoder is the encode policy's pick"
            );
            keys.push(call.key.clone());
            encoders.push(place_of(&self.encoders, assignment.worker.url()));
        }
        Answer::Placed {
            prefill: place_of(&self.prefills, prefill.url()),
            decode: place_of(&self.decodes, decode.url()),
            offered: offered
                .iter()
                .map(|url| place_of(&self.encoders, url))
                .collect(),
            keys,
            encoders,
        }
    }
}

#[derive(Serialize)]
struct Row {
    name: &'static str,
    /// The policy each side of the pair is picked by, in its configuration
    /// spelling.
    prefill_policy: &'static str,
    decode_policy: &'static str,
    /// The policy an item's encoder is picked by.
    encode_policy: &'static str,
    encoders: Vec<Encoder>,
    prefills: Vec<Prefill>,
    decodes: Vec<Decode>,
    requests: Vec<Request>,
    outcome: Outcome,
}

#[derive(Serialize)]
struct Sheet {
    recorded_from: String,
    policy: &'static str,
    rows: Vec<Row>,
}

fn row(
    name: &'static str,
    encode_policy: &'static str,
    fleet: &Fleet,
    requests: Vec<Request>,
) -> Row {
    let record = || {
        let bed = Bed::new(encode_policy, fleet);
        let facts = bed.facts();
        let outcome: Vec<Answer> = requests.iter().map(|request| bed.answer(request)).collect();
        (facts, outcome)
    };
    let (facts, outcome) = record();
    for _ in 0..19 {
        let again = record();
        assert!(
            again.0 == facts && again.1 == outcome,
            "{name}: two recordings of a row differ"
        );
    }
    let (encoders, prefills, decodes) = facts;
    Row {
        name,
        prefill_policy: BY_COUNT,
        decode_policy: BY_COUNT,
        encode_policy,
        encoders,
        prefills,
        decodes,
        requests,
        outcome: Outcome::Sequence(outcome),
    }
}

/// The encoder the ring gives the one item of key `key` in `fleet`.
fn owner_of(fleet: &Fleet, key: u8) -> usize {
    match Bed::new(RING, fleet).answer(&items(&[key])) {
        Answer::Placed { encoders, .. } => encoders[0],
        other @ Answer::Refused { .. } => panic!("a key of the search is placed: {other:?}"),
    }
}

/// The first `n` one-byte keys the ring gives `n` different encoders of
/// `fleet`.
fn keys_of_different_owners(fleet: &Fleet, n: usize) -> Vec<u8> {
    let mut owners = Vec::new();
    let mut keys = Vec::new();
    for key in 1..=u8::MAX {
        let owner = owner_of(fleet, key);
        if !owners.contains(&owner) {
            owners.push(owner);
            keys.push(key);
            if keys.len() == n {
                return keys;
            }
        }
    }
    panic!("fewer than {n} ring owners among the one-byte keys")
}

fn rows() -> Vec<Row> {
    let mut rows = Vec::new();

    // A pair on each of two runtimes. The pair is picked where an encoder is
    // ready, and the items go to the encoders of the runtime it was picked
    // on, whatever the other runtime's encoders hold.
    let two = |prefills: [u32; 2], encoders: [u32; 4]| {
        fleet(
            &[
                push_encoder(C1).holding(encoders[0]),
                push_encoder(C1).holding(encoders[1]),
                meet().holding(encoders[2]),
                meet().holding(encoders[3]),
            ],
            &[
                push_prefill(C1).holding(prefills[0]),
                meet().holding(prefills[1]),
            ],
            &[push_decode(), meet()],
        )
    };
    let on_push = two([0, 4], [3, 1, 2, 0]);
    let on_meet = two([4, 0], [0, 1, 3, 2]);
    rows.push(row(
        "two-runtimes-the-items-go-to-the-encoders-of-the-pushing-pair",
        BY_COUNT,
        &on_push,
        vec![items(&[1, 2, 3])],
    ));
    rows.push(row(
        "two-runtimes-the-items-go-to-the-encoders-of-the-rendezvous-pair",
        BY_COUNT,
        &on_meet,
        vec![items(&[1, 2, 3])],
    ));
    rows.push(row(
        "two-runtimes-a-key-goes-to-an-encoder-of-the-pushing-pair",
        RING,
        &on_push,
        vec![items(&[1, 2, 3])],
    ));
    rows.push(row(
        "two-runtimes-a-key-goes-to-an-encoder-of-the-rendezvous-pair",
        RING,
        &on_meet,
        vec![items(&[1, 2, 3])],
    ));
    rows.push(row(
        "two-runtimes-the-pair-is-picked-where-an-encoder-is-ready-the-rendezvous-encoders-are-not",
        BY_COUNT,
        &closing(&on_meet, &[2, 3]),
        vec![items(&[1, 2])],
    ));
    rows.push(row(
        "two-runtimes-the-pair-is-picked-where-an-encoder-is-ready-the-pushing-encoders-are-not",
        BY_COUNT,
        &closing(&on_push, &[0, 1]),
        vec![items(&[1, 2])],
    ));
    rows.push(row(
        "two-runtimes-a-request-without-items-is-not-held-to-the-encoders",
        BY_COUNT,
        &closing(&on_meet, &[2, 3]),
        vec![items(&[]), items(&[1])],
    ));
    rows.push(row(
        "two-runtimes-references-go-to-the-pushing-runtime",
        BY_COUNT,
        &on_meet,
        vec![items(&[1, 2]).references(), items(&[1, 2])],
    ));
    rows.push(row(
        "two-runtimes-the-items-are-placed-as-one-key-on-the-pushing-pair",
        BY_COUNT,
        &on_push,
        vec![items(&[1, 2, 3]).as_one()],
    ));
    rows.push(row(
        "two-runtimes-the-items-keep-their-keys-on-the-rendezvous-pair",
        BY_COUNT,
        &on_meet,
        vec![items(&[1, 2, 3]).as_one()],
    ));

    // A push: the producer and the consumer roles and their connector.
    let one = fleet(&[push_encoder(C1)], &[push_prefill(C1)], &[push_decode()]);
    rows.push(row(
        "push-a-producer-and-a-consumer-on-one-connector-pair",
        BY_COUNT,
        &one,
        vec![items(&[1, 2])],
    ));
    let by_connector = |prefills: [u32; 2]| {
        fleet(
            &[
                push_encoder(C1).holding(2),
                push_encoder(C2),
                push_encoder(C2).holding(1),
            ],
            &[
                push_prefill(C1).holding(prefills[0]),
                push_prefill(C2).holding(prefills[1]),
            ],
            &[push_decode(), push_decode().holding(1)],
        )
    };
    rows.push(row(
        "push-a-prefill-takes-the-encoder-of-its-connector",
        BY_COUNT,
        &by_connector([0, 4]),
        vec![items(&[1, 2, 3])],
    ));
    rows.push(row(
        "push-a-prefill-takes-the-encoders-of-its-connector-the-one-that-holds-fewer",
        BY_COUNT,
        &by_connector([4, 0]),
        vec![items(&[1, 2, 3])],
    ));
    rows.push(row(
        "push-a-producer-and-a-consumer-on-two-connectors-are-refused",
        BY_COUNT,
        &fleet(&[push_encoder(C1)], &[push_prefill(C2)], &[push_decode()]),
        vec![items(&[1])],
    ));
    rows.push(row(
        "push-an-encoder-without-the-producer-role-is-refused",
        BY_COUNT,
        &fleet(
            &[push_encoder(C1).with("ec_role", "ec_consumer")],
            &[push_prefill(C1)],
            &[push_decode()],
        ),
        vec![items(&[1])],
    ));
    rows.push(row(
        "push-a-prefill-without-the-consumer-role-is-refused",
        BY_COUNT,
        &fleet(
            &[push_encoder(C1)],
            &[push_prefill(C1).with("ec_role", "ec_producer")],
            &[push_decode()],
        ),
        vec![items(&[1])],
    ));
    rows.push(row(
        "push-a-worker-in-both-roles-is-a-producer-and-a-consumer",
        BY_COUNT,
        &fleet(
            &[push_encoder(C1).with("ec_role", "ec_both")],
            &[push_prefill(C1).with("ec_role", "ec_both")],
            &[push_decode()],
        ),
        vec![items(&[1, 2])],
    ));
    rows.push(row(
        "push-a-role-without-a-connector-is-no-role",
        BY_COUNT,
        &fleet(
            &[push_encoder(C1).without("ec_connector")],
            &[push_prefill(C1).without("ec_connector")],
            &[push_decode()],
        ),
        vec![items(&[1])],
    ));
    rows.push(row(
        "push-a-role-on-a-connector-without-a-name-is-no-role",
        BY_COUNT,
        &fleet(&[push_encoder("")], &[push_prefill("")], &[push_decode()]),
        vec![items(&[1])],
    ));
    rows.push(row(
        "push-a-producer-without-the-transfer-label-is-no-producer",
        BY_COUNT,
        &fleet(
            &[push_encoder(C1).without("supports_ec_transfer")],
            &[push_prefill(C1)],
            &[push_decode()],
        ),
        vec![items(&[1])],
    ));
    rows.push(row(
        "push-a-consumer-without-the-transfer-label-is-no-consumer",
        BY_COUNT,
        &fleet(
            &[push_encoder(C1)],
            &[push_prefill(C1).without("supports_ec_transfer")],
            &[push_decode()],
        ),
        vec![items(&[1])],
    ));
    rows.push(row(
        "push-an-encoder-without-the-producer-role-is-refused-ready-or-not",
        BY_COUNT,
        &fleet(
            &[push_encoder(C1).with("ec_role", "ec_consumer").closed()],
            &[push_prefill(C1)],
            &[push_decode()],
        ),
        vec![items(&[1])],
    ));
    rows.push(row(
        "push-the-pair-is-picked-on-the-prefill-that-has-a-producer",
        BY_COUNT,
        &fleet(
            &[push_encoder(C1)],
            &[push_prefill(C2), push_prefill(C1).holding(4)],
            &[push_decode()],
        ),
        vec![items(&[1])],
    ));
    let two_connectors = fleet(
        &[push_encoder(C1), push_encoder(C2)],
        &[push_prefill(C1), push_prefill(C2).holding(4)],
        &[push_decode(), push_decode().holding(1)],
    );
    rows.push(row(
        "push-the-pair-is-picked-on-the-connector-whose-encoder-is-ready",
        BY_COUNT,
        &closing(&two_connectors, &[0]),
        vec![items(&[1, 2])],
    ));
    let mut crossed = closing(&two_connectors, &[1]);
    crossed.prefills[0].open = false;
    rows.push(row(
        "push-a-ready-encoder-on-one-connector-and-a-ready-prefill-on-the-other-are-refused",
        BY_COUNT,
        &crossed,
        vec![items(&[1])],
    ));

    // A rendezvous, and references.
    let meeting = fleet(
        &[meet().holding(2), meet().holding(1)],
        &[meet()],
        &[meet()],
    );
    rows.push(row(
        "rendezvous-the-items-are-placed-over-its-encoders",
        BY_COUNT,
        &meeting,
        vec![items(&[1, 2, 3])],
    ));
    rows.push(row(
        "rendezvous-references-are-not-supported",
        BY_COUNT,
        &meeting,
        vec![
            items(&[1]).references(),
            items(&[1]).references().no_images(),
        ],
    ));
    rows.push(row(
        "rendezvous-a-processor-label-does-not-make-an-encoder-take-references",
        BY_COUNT,
        &fleet(
            &[meet().with("mm_processor", "inprocess")],
            &[meet()],
            &[meet()],
        ),
        vec![items(&[1]).references(), items(&[1])],
    ));
    rows.push(row(
        "push-references-go-to-the-encoders-that-process-them",
        BY_COUNT,
        &fleet(
            &[
                push_encoder(C1).without("mm_processor"),
                push_encoder(C1).holding(3),
            ],
            &[push_prefill(C1)],
            &[push_decode()],
        ),
        vec![items(&[1, 2]).references(), items(&[1, 2])],
    ));
    rows.push(row(
        "push-references-without-an-encoder-that-processes-them-are-not-supported",
        BY_COUNT,
        &fleet(
            &[push_encoder(C1).without("mm_processor")],
            &[push_prefill(C1)],
            &[push_decode()],
        ),
        vec![items(&[1]).references()],
    ));

    // Images: an encoder that takes them, and a decode that takes them.
    let blind = || push_encoder(C1).with("supports_vision", "false");
    let text_decode = || push_decode().with("supports_vision", "false");
    rows.push(row(
        "images-go-to-the-encoders-that-take-them",
        BY_COUNT,
        &fleet(
            &[blind(), push_encoder(C1).holding(3)],
            &[push_prefill(C1)],
            &[push_decode()],
        ),
        vec![items(&[1, 2, 3]), items(&[1, 2, 3]).no_images()],
    ));
    rows.push(row(
        "images-without-an-encoder-that-takes-them-are-refused",
        BY_COUNT,
        &fleet(&[blind()], &[push_prefill(C1)], &[push_decode()]),
        vec![items(&[1]), items(&[1]).no_images()],
    ));
    let two_decodes = fleet(
        &[push_encoder(C1)],
        &[push_prefill(C1)],
        &[text_decode(), push_decode().holding(4)],
    );
    let text_alone = fleet(&[push_encoder(C1)], &[push_prefill(C1)], &[text_decode()]);
    rows.push(row(
        "decode-images-take-the-decode-that-takes-them",
        BY_COUNT,
        &two_decodes,
        vec![items(&[1]).decode_images(), items(&[1])],
    ));
    rows.push(row(
        "decode-images-without-a-decode-that-takes-them-are-refused",
        BY_COUNT,
        &text_alone,
        vec![items(&[1]).decode_images(), items(&[1])],
    ));
    rows.push(row(
        "references-with-images-prefer-a-decode-that-takes-images",
        BY_COUNT,
        &two_decodes,
        vec![
            items(&[1]).references(),
            items(&[1]).references().no_images(),
        ],
    ));
    rows.push(row(
        "references-with-images-keep-a-decode-without-them-when-no-other-is-registered",
        BY_COUNT,
        &text_alone,
        vec![items(&[1]).references()],
    ));
    rows.push(row(
        "rendezvous-a-decode-is-not-held-to-the-vision-label",
        BY_COUNT,
        &fleet(
            &[meet()],
            &[meet()],
            &[meet().with("supports_vision", "false")],
        ),
        vec![items(&[1]).decode_images()],
    ));

    // What is ready against what is registered.
    rows.push(row(
        "no-ready-encoder-beside-a-registered-one-is-unavailable",
        BY_COUNT,
        &closing(&one, &[0]),
        vec![items(&[1])],
    ));
    rows.push(row(
        "a-request-without-items-needs-no-encoder",
        BY_COUNT,
        &closing(&one, &[0]),
        vec![items(&[])],
    ));
    rows.push(row(
        "a-request-without-items-is-not-held-to-the-encode-roles",
        BY_COUNT,
        &fleet(&[push_encoder(C1)], &[push_prefill(C2)], &[push_decode()]),
        vec![items(&[])],
    ));
    rows.push(row(
        "no-encoder-registered-is-unavailable",
        BY_COUNT,
        &fleet(&[], &[push_prefill(C1)], &[push_decode()]),
        vec![items(&[1])],
    ));
    rows.push(row(
        "a-ready-encoder-of-another-runtime-does-not-open-the-pair",
        BY_COUNT,
        &fleet(
            &[push_encoder(C1).closed(), meet()],
            &[push_prefill(C1)],
            &[push_decode()],
        ),
        vec![items(&[1])],
    ));
    rows.push(row(
        "the-pair-s-prefill-not-ready-is-the-pair-s-refusal",
        BY_COUNT,
        &fleet(
            &[push_encoder(C1)],
            &[push_prefill(C1).closed()],
            &[push_decode()],
        ),
        vec![items(&[1])],
    ));
    rows.push(row(
        "an-encoder-whose-runtime-states-no-handoff-serves-no-item",
        BY_COUNT,
        &fleet(&[bare()], &[bare()], &[bare()]),
        vec![items(&[1]), items(&[1]).no_images(), items(&[])],
    ));

    // The items and their keys: one item, three ring owners, the items as
    // one key.
    let three = fleet(
        &[push_encoder(C1), push_encoder(C1), push_encoder(C1)],
        &[push_prefill(C1)],
        &[push_decode()],
    );
    let keys = keys_of_different_owners(&three, 3);
    rows.push(row(
        "ring-one-item-goes-to-its-key-s-owner",
        RING,
        &three,
        vec![items(&keys[..1])],
    ));
    rows.push(row(
        "ring-three-items-of-three-owners-go-to-three-encoders",
        RING,
        &three,
        vec![items(&keys); 2],
    ));
    rows.push(row(
        "ring-equal-keys-go-to-one-encoder",
        RING,
        &three,
        vec![items(&[keys[0], keys[1], keys[0]])],
    ));
    rows.push(row(
        "ring-the-items-of-a-request-without-image-fan-out-are-placed-as-one-key",
        RING,
        &three,
        vec![
            items(&keys).as_one(),
            items(&keys[..2]).as_one(),
            items(&keys[..1]).as_one(),
        ],
    ));
    rows.push(row(
        "ring-a-key-whose-owner-is-not-ready-goes-to-another-encoder",
        RING,
        &closing(&three, &[owner_of(&three, keys[0])]),
        vec![items(&keys)],
    ));
    rows.push(row(
        "ring-forty-eight-keys-over-three-encoders",
        RING,
        &three,
        vec![items(&(1..=48).collect::<Vec<u8>>())],
    ));
    let meeting_three = fleet(&[meet(), meet(), meet()], &[meet()], &[meet()]);
    let meeting_keys = keys_of_different_owners(&meeting_three, 3);
    rows.push(row(
        "ring-a-rendezvous-never-places-the-items-as-one-key",
        RING,
        &meeting_three,
        vec![items(&meeting_keys).as_one(), items(&meeting_keys)],
    ));
    rows.push(row(
        "push-the-items-of-a-request-without-image-fan-out-take-one-encoder",
        BY_COUNT,
        &fleet(
            &[push_encoder(C1).holding(1), push_encoder(C1)],
            &[push_prefill(C1)],
            &[push_decode()],
        ),
        vec![items(&[1, 2, 3]).as_one(), items(&[1, 2, 3])],
    ));
    rows
}

#[test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
fn record_the_classic_encode_pair_rows() {
    let out = PathBuf::from(
        std::env::var_os("CLASSIC_ROWS_OUT").expect("CLASSIC_ROWS_OUT names the output directory"),
    );
    let commit =
        std::env::var("CLASSIC_ROWS_COMMIT").expect("CLASSIC_ROWS_COMMIT names the commit");
    let sheet = Sheet {
        recorded_from: commit,
        policy: "encode_pair",
        rows: rows(),
    };
    fs::create_dir_all(&out).expect("the output directory");
    let path = out.join("encode_pair.json");
    let text = serde_json::to_string_pretty(&sheet).expect("the sheet serializes");
    fs::write(&path, text + "\n").expect("the sheet is written");
    for row in &sheet.rows {
        println!(
            "ROW {}: {}",
            row.name,
            serde_json::to_string(&row.outcome).expect("an outcome serializes")
        );
    }
    println!("{} rows -> {}", sheet.rows.len(), path.display());
}
