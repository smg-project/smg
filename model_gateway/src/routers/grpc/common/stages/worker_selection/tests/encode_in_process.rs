//! An encode request where the policy engine of this process decides: its
//! pair, then one encoder for each key its items are placed by. The request
//! may use what the selection with no policy engine leaves it. So the two
//! place alike wherever the policies leave no choice, and a request no
//! selection can serve reads that selection's own answer.

use openai_protocol::worker::{WorkerLoadResponse, WorkerStatus};
use parking_lot::Mutex;
use smg_policy_controller::{
    document::{AppliedSpec, Needs, PolicySpec},
    engine::{
        Candidate, Decision, PolicyEngine as OwnEngine, RequestFacts, Scorer, TieBand, WorkerId,
    },
    stages::{Lifecycle, StageRegistry},
    tracking::{EventKind, EventSet, Lease},
};
use smg_worker::load::state::{one_rank_load, TestFeed};

use super::{
    in_process::{answer_of, own_engine},
    *,
};

const MODEL: &str = "test-model";

/// A request of router-processed images with no grids, each image placed by
/// its own key.
const IMAGES: EpdDemands = EpdDemands {
    wire: None,
    encode: MediaDemands {
        media_refs: false,
        vision: true,
        pipeline: PipelinePin::Any,
    },
    decode_vision: false,
    image_fanout: true,
};

/// The encode stage of `registry` with no policy engine: `encode` on its
/// encode side and round robin on its pair's two sides.
fn classic_stage(registry: &Arc<WorkerRegistry>, encode: &PolicyConfig) -> WorkerSelectionStage {
    let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
    let round_robin = || PolicyFactory::create_from_config(&PolicyConfig::RoundRobin);
    policies.set_encode_policy(PolicyFactory::create_from_config(encode));
    policies.set_prefill_policy(round_robin());
    policies.set_decode_policy(round_robin());
    WorkerSelectionStage::new(
        Arc::clone(registry),
        Arc::new(policies),
        WorkerSelectionMode::EncodePrefillDecode,
        None,
    )
}

/// The same stage over `engine`, the policy engine of this process.
fn stage_under(registry: &Arc<WorkerRegistry>, engine: Arc<PolicyEngine>) -> WorkerSelectionStage {
    classic_stage(registry, &PolicyConfig::RoundRobin).with_policy_engine(Some(engine))
}

/// The same stage over the engine of this process under the stock document
/// of three sides: `encode` on the encode side, round robin on the pair's.
fn own_stage(registry: &Arc<WorkerRegistry>, encode: &str) -> WorkerSelectionStage {
    let document = PolicySpec::epd_presets(encode, "round_robin", "round_robin");
    let document = document.expect("the presets of three sides");
    stage_under(registry, own_engine(registry, document))
}

/// A registry of [`MODEL`]'s workers of `runtime`, each with what `fleet`
/// names for it: its port, its role, its EC connector, its EC role and
/// whether it takes vision input.
fn fleet_of(
    runtime: RuntimeType,
    fleet: &[(u16, WorkerType, &str, &str, bool)],
) -> Arc<WorkerRegistry> {
    let registry = Arc::new(WorkerRegistry::new());
    for &(port, role, connector, ec_role, vision) in fleet {
        let worker = BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .model(ModelCard::new(MODEL))
            .worker_type(role)
            .runtime_type(runtime)
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check())
            .label("supports_ec_transfer", "true")
            .label("mm_processor", "inprocess")
            .label("mm_media_ref_schemes", "http,https,data")
            .label("ec_connector", connector)
            .label("ec_role", ec_role)
            .label("supports_vision", vision.to_string())
            .label("kv_connector", "NixlConnector")
            .build();
        registry.register(Arc::new(worker)).expect("a free address");
    }
    registry
}

/// The worker of `registry` at `port`.
fn at(registry: &WorkerRegistry, port: u16) -> Arc<dyn Worker> {
    let url = format!("grpc://127.0.0.1:{port}");
    registry.get_by_url(&url).expect("a worker of the fleet")
}

/// A selection's workers by address: the encoder of each key, the prefill
/// and the decode.
type Chosen = (Vec<String>, String, String);

fn chosen((encoders, prefill, decode, _): &EncodePrefillDecodeWorkerSelection) -> Chosen {
    let encoders = encoders.iter().map(|assigned| assigned.worker.url().into());
    (
        encoders.collect(),
        prefill.url().into(),
        decode.url().into(),
    )
}

/// What the engine's arm answers a first attempt whose items are placed by
/// `keys`.
async fn placed(
    stage: &WorkerSelectionStage,
    keys: &[Vec<u8>],
    demand: EpdDemands,
) -> Result<Chosen, Response> {
    let ctx = prepared_ctx(Arc::clone(&stage.worker_registry), None);
    let engine = stage.policy_engine.as_ref().expect("the stage's engine");
    let inputs = PlacementInputs::default();
    let first = Attempt::First(&ctx);
    let attempt = stage.place_encode_pair(engine, MODEL, inputs, None, keys, demand, first);
    let ((selection, placed), _guard) = attempt.await?;
    let placed = placed.expect("the engine's placement of the pair");
    assert!(placed.decode().is_some(), "a pair's placement");
    Ok(chosen(&selection))
}

/// What the selection with no policy engine answers the same attempt.
fn selected(
    stage: &WorkerSelectionStage,
    keys: &[Vec<u8>],
    demand: EpdDemands,
) -> Result<Chosen, Response> {
    let inputs = PlacementInputs::default();
    let selection = stage.select_encode_prefill_decode_workers(MODEL, inputs, keys, demand, None);
    let refused = |failure| stage.epd_selection_failure(MODEL, failure, !keys.is_empty());
    Ok(chosen(&selection.map_err(refused)?))
}

/// One attempt asked of both stages over one fleet: placed on the same
/// workers, or refused with the same answer whole. Says whether it was
/// placed.
async fn alike(
    stages: &[WorkerSelectionStage; 2],
    keys: &[Vec<u8>],
    demand: EpdDemands,
    case: &str,
) -> bool {
    let [classic, own] = stages;
    match (
        selected(classic, keys, demand),
        placed(own, keys, demand).await,
    ) {
        (Ok(selected), Ok(placed)) => {
            assert_eq!(placed, selected, "{case}");
            true
        }
        (Err(selected), Err(placed)) => {
            assert_eq!(answer_of(placed).await, answer_of(selected).await, "{case}");
            false
        }
        (selected, placed) => panic!(
            "{case}: the selection with no policy engine placed the request: {}, the engine: {}",
            selected.is_ok(),
            placed.is_ok()
        ),
    }
}

/// One encoder, one prefill and one decode, so no policy has a choice, and a
/// request of them.
struct Trio {
    /// The EC connector of the encoder and of the prefill.
    connectors: [&'static str; 2],
    /// The EC role of the encoder and of the prefill.
    roles: [&'static str; 2],
    /// Whether the encoder and the decode take vision input.
    vision: [bool; 2],
    /// Whether the request carries pixels and grids.
    carries: [bool; 2],
}

const fn trio(
    connectors: [&'static str; 2],
    roles: [&'static str; 2],
    vision: [bool; 2],
    carries: [bool; 2],
) -> Trio {
    Trio {
        connectors,
        roles,
        vision,
        carries,
    }
}

const PRODUCES: [&str; 2] = ["ec_producer", "ec_consumer"];
const SAME: [&str; 2] = ["ECCPU", "ECCPU"];
const ALL: [bool; 2] = [true, true];

/// The fleets and requests the two selections are compared over: one that
/// pairs and serves, then one difference at a time. Two connectors, an
/// encoder stated a consumer, a prefill stated a producer, both stated both
/// roles, no connector, an encoder with no vision input, a request of no
/// pixels, and a decode with no vision input under a request with grids
/// and under one without.
const TRIOS: [Trio; 10] = [
    trio(SAME, PRODUCES, ALL, ALL),
    trio(["ECCPU", "ECOther"], PRODUCES, ALL, ALL),
    trio(SAME, ["ec_consumer", "ec_consumer"], ALL, ALL),
    trio(SAME, ["ec_producer", "ec_producer"], ALL, ALL),
    trio(SAME, ["ec_both", "ec_both"], ALL, ALL),
    trio(["", ""], PRODUCES, ALL, ALL),
    trio(SAME, PRODUCES, [false, true], [true, false]),
    trio(SAME, PRODUCES, [false, false], [false, false]),
    trio(SAME, PRODUCES, [true, false], ALL),
    trio(SAME, PRODUCES, [true, false], [true, false]),
];

/// Over every fleet of one encoder, one prefill and one decode, under a push
/// handoff and under a rendezvous one, the engine answers an encode request
/// as the selection with no policy engine does: the same three workers, or
/// the same refusal, its status, its code and its message. So does it with
/// the encoder not ready, for a request of media references, and for a
/// request with no item.
#[tokio::test]
async fn an_encode_request_is_answered_as_the_selection_with_no_policy_engine_answers_it() {
    let (mut placed, mut refused) = (0, 0);
    let mut count = |was_placed: bool| {
        if was_placed {
            placed += 1;
        } else {
            refused += 1;
        }
    };
    for runtime in [RuntimeType::Vllm, RuntimeType::TokenSpeed] {
        for (at_row, row) in TRIOS.iter().enumerate() {
            let [e_connector, p_connector] = row.connectors;
            let [e_role, p_role] = row.roles;
            let [e_vision, d_vision] = row.vision;
            let [pixels, grids] = row.carries;
            let registry = fleet_of(
                runtime,
                &[
                    (8810, WorkerType::Encode, e_connector, e_role, e_vision),
                    (8811, WorkerType::Prefill, p_connector, p_role, true),
                    (8812, WorkerType::Decode, "", "", d_vision),
                ],
            );
            let round_robin = PolicyConfig::RoundRobin;
            let stages = [
                classic_stage(&registry, &round_robin),
                own_stage(&registry, "round_robin"),
            ];
            let demand = |media_refs: bool, decode_vision: bool| EpdDemands {
                wire: None,
                encode: MediaDemands {
                    media_refs,
                    vision: pixels,
                    pipeline: PipelinePin::Any,
                },
                decode_vision,
                image_fanout: false,
            };
            let case = |what: &str| format!("{runtime:?} fleet {at_row}, {what}");
            let one = [vec![1]];
            count(alike(&stages, &one, demand(false, grids), &case("one item")).await);
            count(alike(&stages, &one, demand(true, false), &case("references")).await);
            count(alike(&stages, &[], demand(false, grids), &case("no item")).await);
            at(&registry, 8810).set_status(WorkerStatus::NotReady);
            count(
                alike(
                    &stages,
                    &one,
                    demand(false, grids),
                    &case("no encoder ready"),
                )
                .await,
            );
            count(
                alike(
                    &stages,
                    &[],
                    demand(false, grids),
                    &case("no item, no encoder"),
                )
                .await,
            );
        }
    }
    // Both answers are met many times over the fleets.
    assert!(
        placed >= 20 && refused >= 20,
        "{placed} placed, {refused} refused"
    );
}

/// Two prefills on two EC connectors, an encoder on each: the pair takes the
/// prefills in turn, and every item of a request goes to the encoder that
/// reaches the prefill chosen for it, as the selection with no policy engine
/// narrows its encoders ahead of its encode policy.
#[tokio::test]
async fn each_item_takes_an_encoder_that_reaches_the_chosen_prefill() {
    for encode in ["round_robin", "random", "passthrough"] {
        let registry = ec_registry(
            MODEL,
            &[
                (8910, WorkerType::Encode, "ECCPUConnector"),
                (8911, WorkerType::Encode, "ECOtherConnector"),
                (8912, WorkerType::Prefill, "ECCPUConnector"),
                (8913, WorkerType::Prefill, "ECOtherConnector"),
                (8914, WorkerType::Decode, "ECCPUConnector"),
            ],
        );
        let stage = own_stage(&registry, encode);
        let mut prefills = Vec::new();
        for _ in 0..8 {
            let keys = [vec![1], vec![2], vec![3]];
            let (encoders, prefill, _) = placed(&stage, &keys, IMAGES).await.expect("placed");
            let reaching = match prefill.as_str() {
                "grpc://127.0.0.1:8912" => "grpc://127.0.0.1:8910",
                "grpc://127.0.0.1:8913" => "grpc://127.0.0.1:8911",
                other => panic!("no prefill of the fleet: {other}"),
            };
            assert_eq!(encoders, [reaching; 3], "{encode}");
            prefills.push(prefill);
        }
        prefills.sort();
        prefills.dedup();
        assert_eq!(prefills.len(), 2, "{encode}: both prefills take a request");
    }
}

/// Three encoders that all reach the prefill. Under round robin the items of
/// a request take them in turn, and under each policy the engine decides,
/// every item lands on an encoder of the pool. An encoder that is not ready,
/// and one over its overload thresholds, takes no item under either arm.
#[tokio::test]
async fn the_items_are_placed_over_the_encoders_that_can_take_them() {
    let fleet = [
        (8920, WorkerType::Encode, "ECCPUConnector"),
        (8921, WorkerType::Encode, "ECCPUConnector"),
        (8922, WorkerType::Encode, "ECCPUConnector"),
        (8923, WorkerType::Prefill, "ECCPUConnector"),
        (8924, WorkerType::Decode, "ECCPUConnector"),
    ];
    let encoders_of = |ports: &[u16]| -> Vec<String> {
        let url = |port: &u16| format!("grpc://127.0.0.1:{port}");
        ports.iter().map(url).collect()
    };
    let keys = [vec![1], vec![2], vec![3]];
    for encode in ["round_robin", "random", "passthrough"] {
        let registry = ec_registry(MODEL, &fleet);
        let stage = own_stage(&registry, encode);
        let mut seen = Vec::new();
        for _ in 0..16 {
            let (mut encoders, _, _) = placed(&stage, &keys, IMAGES).await.expect("placed");
            assert_eq!(encoders.len(), 3, "{encode}: one encoder a key");
            if encode == "round_robin" {
                encoders.sort();
                assert_eq!(encoders, encoders_of(&[8920, 8921, 8922]), "a turn each");
            }
            seen.extend(encoders);
        }
        seen.sort();
        seen.dedup();
        let whole = if encode == "passthrough" { 1 } else { 3 };
        assert_eq!(seen.len(), whole, "{encode}: {seen:?}");

        // One encoder is not ready and one is over its thresholds: the
        // third takes every item, as with no policy engine.
        at(&registry, 8920).set_status(WorkerStatus::NotReady);
        registry.set_worker_overloaded(&at(&registry, 8921), true);
        let round_robin = PolicyConfig::RoundRobin;
        let stages = [classic_stage(&registry, &round_robin), stage];
        for _ in 0..4 {
            assert!(alike(&stages, &keys, IMAGES, encode).await);
        }
        let (encoders, _, _) = placed(&stages[1], &keys, IMAGES).await.expect("placed");
        assert_eq!(encoders, encoders_of(&[8922, 8922, 8922]), "{encode}");
    }
}

/// Under a push handoff with no image fan-out the items of a request are
/// placed as one, by one key, and under a rendezvous handoff each by its
/// own: the count of encoders asked is the selection's with no policy
/// engine.
#[tokio::test]
async fn a_push_handoff_places_the_items_of_a_request_as_one() {
    for (runtime, asked) in [(RuntimeType::Vllm, 1), (RuntimeType::TokenSpeed, 2)] {
        let registry = fleet_of(
            runtime,
            &[
                (8930, WorkerType::Encode, "ECCPU", "ec_producer", true),
                (8931, WorkerType::Prefill, "ECCPU", "ec_consumer", true),
                (8932, WorkerType::Decode, "", "", true),
            ],
        );
        let round_robin = PolicyConfig::RoundRobin;
        let stages = [
            classic_stage(&registry, &round_robin),
            own_stage(&registry, "round_robin"),
        ];
        let merged = EpdDemands {
            image_fanout: false,
            ..IMAGES
        };
        let keys = [vec![1], vec![2]];
        assert!(alike(&stages, &keys, merged, "merged").await);
        let (encoders, _, _) = placed(&stages[1], &keys, merged).await.expect("placed");
        assert_eq!(encoders.len(), asked, "{runtime:?}");
        let (encoders, _, _) = placed(&stages[1], &keys, IMAGES).await.expect("placed");
        assert_eq!(
            encoders.len(),
            2,
            "{runtime:?}: a key an image under fan-out"
        );
    }
}

/// A whole first attempt through the stage: the engine places the pair and
/// the item's encoder, and the request holds its pair's placement and its
/// prefill's slot. A retry places no item again: one whose plan replays no
/// handoff asks its pair alone of the engine.
#[tokio::test]
async fn the_stage_places_an_encode_request_and_its_retry_by_the_engine() {
    let registry = ec_registry(
        MODEL,
        &[
            (8940, WorkerType::Encode, "ECCPUConnector"),
            (8941, WorkerType::Encode, "ECCPUConnector"),
            (8942, WorkerType::Prefill, "ECCPUConnector"),
            (8943, WorkerType::Decode, "ECCPUConnector"),
        ],
    );
    let stage = own_stage(&registry, "round_robin");
    let mut encoders = Vec::new();
    for _ in 0..4 {
        let mut ctx = prepared_ctx(Arc::clone(&registry), None);
        ctx.state.multimodal_intermediate = Some(image_intermediate(None));
        stage.execute(&mut ctx).await.expect("a placed request");
        let Some(WorkerSelection::Disaggregated {
            encode_assignments: Some(assigned),
            prefill,
            decode,
            ..
        }) = &ctx.state.workers
        else {
            panic!("an encode request's selection with its item's encoder");
        };
        assert_eq!(prefill.url(), "grpc://127.0.0.1:8942");
        assert_eq!(decode.url(), "grpc://127.0.0.1:8943");
        assert_eq!(assigned.len(), 1, "one image, one encoder");
        encoders.push(assigned[0].worker.url().to_owned());
        let placement = ctx.state.placed.as_ref().expect("the pair's placement");
        assert_eq!(placement.worker().url(), prefill.url());
        assert!(ctx.state.pd_prefill_guard.is_some());
    }
    encoders.sort();
    encoders.dedup();
    assert_eq!(encoders.len(), 2, "the encoders take the requests in turn");

    // A request with no item is its pair.
    let mut ctx = prepared_ctx(Arc::clone(&registry), None);
    stage.execute(&mut ctx).await.expect("a placed pair");
    assert!(matches!(
        ctx.state.workers,
        Some(WorkerSelection::Disaggregated {
            encode_assignments: None,
            ..
        })
    ));
    assert!(ctx.state.placed.is_some());
    drop(ctx);

    let mut retry = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Vllm));
    assert!(stage.places_retry(&retry));
    let reselected = stage.reselect(&mut retry, &retained_plan(&[1])).await;
    reselected.expect("the pair again");
    assert!(retry.placed.is_some(), "the engine placed the retry's pair");
    assert!(matches!(
        retry.workers,
        Some(WorkerSelection::Disaggregated {
            encode_assignments: None,
            ..
        })
    ));
}

/// What a stage heard of a lease: the word and the worker the lease books.
type Heard = Arc<Mutex<Vec<(&'static str, WorkerId)>>>;

/// A scorer that scores every candidate alike and keeps what it hears of a
/// lease's issue and its release.
#[derive(Debug)]
struct Hears(Heard);

impl Lifecycle for Hears {
    fn hears(&self) -> EventSet {
        EventSet::NONE
            .with(EventKind::Issued)
            .with(EventKind::Released)
    }

    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        self.0.lock().push(("issued", lease.booked));
    }

    fn on_released(&self, lease: &Lease, _: ReleaseReason) {
        self.0.lock().push(("released", lease.booked));
    }
}

impl Scorer for Hears {
    fn name(&self) -> &'static str {
        "hears"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(1.0);
        TieBand(0.0)
    }
}

/// A scorer whose score panics: the decision it is asked for is one the
/// engine does not make.
#[derive(Debug)]
struct Panics;

impl Lifecycle for Panics {}

impl Scorer for Panics {
    fn name(&self) -> &'static str {
        "panics"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], _: &mut [f32]) -> TieBand {
        panic!("this scorer panics");
    }
}

/// An item whose decision the engine does not make gives the request's pair
/// up: each lease the pair was issued is released by name, so a stage that
/// heard its issue hears its end, the prefill's slot is given back, and the
/// answer is the engine's own.
#[tokio::test]
async fn an_item_the_engine_does_not_place_gives_the_pair_up() {
    let registry = ec_registry(
        MODEL,
        &[
            (8950, WorkerType::Encode, "ECCPUConnector"),
            (8951, WorkerType::Prefill, "ECCPUConnector"),
            (8952, WorkerType::Decode, "ECCPUConnector"),
        ],
    );
    let heard = Heard::default();
    let kept = Arc::clone(&heard);
    let mut stages = StageRegistry::builtin();
    let hears = move || -> Box<dyn Scorer> { Box::new(Hears(Arc::clone(&kept))) };
    stages.scorer("hears", hears).expect("a free name");
    let panics = || -> Box<dyn Scorer> { Box::new(Panics) };
    stages.scorer("panics", panics).expect("a free name");
    let document = PolicySpec::epd_presets("random", "random", "round_robin");
    let mut document = document.expect("the presets of three sides");
    document.generation = 1;
    for (side, scorer) in [("prefill", "hears"), ("encode", "panics")] {
        let profile = document.profiles.get_mut(side).expect("the side's profile");
        profile.score = vec![scorer.into()];
        profile.pick = "best".into();
    }
    let spec = AppliedSpec::validate_in(&document, &stages, &[]).expect("the document validates");
    let engine = PolicyEngine::own(Arc::new(OwnEngine::new(spec)), Arc::clone(&registry));
    let stage = stage_under(&registry, Arc::new(engine));
    let (prefill, decode) = (at(&registry, 8951), at(&registry, 8952));

    // With no item the pair is placed, and holds its prefill's slot.
    let pair = placed(&stage, &[], IMAGES).await;
    assert!(pair.is_ok_and(|(encoders, _, _)| encoders.is_empty()));
    assert_eq!(
        prefill.routing_state().load,
        0,
        "the slot went with the guard"
    );
    heard.lock().clear();

    let refused = placed(&stage, &[vec![1]], IMAGES).await;
    let (status, body) = answer_of(refused.expect_err("no encoder was decided")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "policy_controller_unavailable");
    let booked = [prefill, decode].map(|worker| smg_policy_controller::engine::worker_id(&*worker));
    let told = |word: &'static str| booked.map(|worker| (word, worker));
    assert_eq!(*heard.lock(), [told("issued"), told("released")].concat());
    let prefill = at(&registry, 8951);
    assert_eq!(
        prefill.routing_state().load,
        0,
        "the prefill's slot is given back"
    );
}

/// What a stage read at each decision: the request's size, and how many of
/// the candidates it was asked of came with a load report.
type Read = Arc<Mutex<Vec<(u32, usize)>>>;

/// A scorer that reads the request's size and each candidate's load report,
/// declares both, and keeps what it read.
#[derive(Debug)]
struct Reads(Read);

impl Lifecycle for Reads {}

impl Scorer for Reads {
    fn name(&self) -> &'static str {
        "reads"
    }

    fn needs(&self) -> Needs {
        Needs {
            prompt_size: true,
            loads: true,
            ..Needs::default()
        }
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let reported = |row: &&u32| candidates[**row as usize].load.is_some();
        let reported = kept.iter().filter(reported).count();
        self.0.lock().push((facts.prompt_tokens, reported));
        out.fill(1.0);
        TieBand(0.0)
    }
}

/// An item's decision is fed what its pair's is. Its ask states the
/// request's prompt size, and its list's records read the reports the load
/// monitor publishes: a stage of the encode side that declares both reads
/// the prompt's size and a report for each encoder, at each item.
#[tokio::test]
async fn an_items_decision_reads_the_prompts_size_and_the_load_reports() {
    let registry = ec_registry(
        MODEL,
        &[
            (8960, WorkerType::Encode, "ECCPUConnector"),
            (8961, WorkerType::Encode, "ECCPUConnector"),
            (8962, WorkerType::Prefill, "ECCPUConnector"),
            (8963, WorkerType::Decode, "ECCPUConnector"),
        ],
    );
    let read = Read::default();
    let kept = Arc::clone(&read);
    let mut stages = StageRegistry::builtin();
    let reads = move || -> Box<dyn Scorer> { Box::new(Reads(Arc::clone(&kept))) };
    stages.scorer("reads", reads).expect("a free name");
    let document = PolicySpec::epd_presets("random", "round_robin", "round_robin");
    let mut document = document.expect("the presets of three sides");
    document.generation = 1;
    let profile = document.profiles.get_mut("encode");
    let profile = profile.expect("the encode side's profile");
    profile.score = vec!["reads".into()];
    profile.pick = "best".into();
    let spec = AppliedSpec::validate_in(&document, &stages, &[]).expect("the document validates");
    let engine = PolicyEngine::own(Arc::new(OwnEngine::new(spec)), Arc::clone(&registry));
    let feed = TestFeed::attached_to(|receiver| {
        engine.read_loads(receiver.expect("the feed's receiver"));
    });
    let report = |port: u16| {
        let url = format!("grpc://127.0.0.1:{port}");
        (url, one_rank_load(0, 0.1, 1_000.0))
    };
    let reports: HashMap<String, WorkerLoadResponse> = [8960, 8961].map(report).into();
    feed.publish(&reports);
    let stage = stage_under(&registry, Arc::new(engine));

    let tokens: Vec<u32> = (0..600).collect();
    let inputs = PlacementInputs {
        tokens: Some(&tokens),
        ..PlacementInputs::default()
    };
    let ctx = prepared_ctx(Arc::clone(&registry), None);
    let engine = stage.policy_engine.as_ref().expect("the stage's engine");
    let keys = [vec![1], vec![2]];
    let first = Attempt::First(&ctx);
    let attempt = stage.place_encode_pair(engine, MODEL, inputs, None, &keys, IMAGES, first);
    let ((selection, _placed), _guard) = attempt.await.expect("a placed request");
    assert_eq!(chosen(&selection).0.len(), 2, "one encoder a key");
    assert_eq!(*read.lock(), [(600, 2); 2], "each item's decision");
}
