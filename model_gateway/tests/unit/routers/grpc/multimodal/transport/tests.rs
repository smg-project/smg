use openai_protocol::model_card::ModelCard;
use smg_worker::{identity::basic::WorkerType, test_support::no_health_check};

use super::*;
use crate::routers::grpc::backend_client::BasicWorkerBuilder;

fn worker(url: &str, runtime: RuntimeType, labels: &[(&str, &str)]) -> Arc<dyn Worker> {
    let mut builder = BasicWorkerBuilder::new(url)
        .model(ModelCard::new("qwen3-vl"))
        .worker_type(WorkerType::Regular)
        .runtime_type(runtime)
        .health_config(no_health_check());
    for (key, value) in labels {
        builder = builder.label(*key, *value);
    }
    Arc::new(builder.build())
}

/// The dtype `workers` agree on, the one fold over a slice.
fn shared_image_wire_dtype(
    workers: &[Arc<dyn Worker>],
    overrides: &DtypeOverrides,
) -> Option<EncoderDtype> {
    let mut agreed = None;
    agree_on_image_wire_dtype(&mut agreed, workers.iter(), overrides)?;
    agreed
}

/// Pool planning agrees with assembly, including SGLang worker labels.
#[test]
fn a_worker_image_wire_dtype_is_what_its_assembly_writes() {
    let plain = DtypeOverrides::default();
    let bf16 = || Some("bfloat16");
    assert_eq!(
        image_wire_dtype(RuntimeType::Sglang, bf16(), false, &plain),
        Some(EncoderDtype::BFloat16)
    );
    assert_eq!(
        image_wire_dtype(RuntimeType::Vllm, None, false, &plain),
        Some(EncoderDtype::Float32)
    );
    assert_eq!(
        image_wire_dtype(RuntimeType::Vllm, bf16(), false, &plain),
        Some(EncoderDtype::BFloat16)
    );
    assert_eq!(
        image_wire_dtype(RuntimeType::Vllm, bf16(), true, &plain),
        None,
        "an engine normalizing on device takes the pixels' own bytes"
    );
    assert_eq!(
        image_wire_dtype(RuntimeType::Vllm, Some("uint8"), false, &plain),
        None
    );
    assert_eq!(
        image_wire_dtype(RuntimeType::TokenSpeed, None, false, &plain),
        Some(EncoderDtype::BFloat16)
    );
    assert_eq!(
        image_wire_dtype(RuntimeType::TokenSpeed, Some("uint8"), false, &plain),
        Some(EncoderDtype::Float32),
        "that wire reads float widths only"
    );
    assert_eq!(
        image_wire_dtype(RuntimeType::Trtllm, bf16(), false, &plain),
        None
    );
    let overridden = DtypeOverrides {
        vllm: Some("float16"),
        image: Some("float32"),
        default: bf16(),
    };
    assert_eq!(
        image_wire_dtype(RuntimeType::Vllm, bf16(), false, &overridden),
        Some(EncoderDtype::Float16)
    );
    assert_eq!(
        image_wire_dtype(RuntimeType::TokenSpeed, bf16(), false, &overridden),
        Some(EncoderDtype::Float32)
    );
    assert_eq!(
        image_wire_dtype(RuntimeType::Sglang, bf16(), false, &overridden),
        Some(EncoderDtype::BFloat16)
    );
}

#[test]
fn the_shared_image_wire_dtype_needs_every_worker_to_agree() {
    let plain = DtypeOverrides::default();
    let bf16 = |url| {
        worker(
            url,
            RuntimeType::Vllm,
            &[("multimodal_encoder_dtype", "bfloat16")],
        )
    };
    assert_eq!(shared_image_wire_dtype(&[], &plain), None);
    assert_eq!(
        shared_image_wire_dtype(&[bf16("http://a"), bf16("http://b")], &plain),
        Some(EncoderDtype::BFloat16)
    );
    assert_eq!(
        shared_image_wire_dtype(
            &[
                bf16("http://a"),
                worker(
                    "http://c",
                    RuntimeType::TokenSpeed,
                    &[("multimodal_encoder_dtype", "bfloat16")],
                ),
            ],
            &plain
        ),
        Some(EncoderDtype::BFloat16),
        "two runtimes reading the same width agree"
    );
    assert_eq!(
        shared_image_wire_dtype(
            &[bf16("http://a"), worker("http://d", RuntimeType::Vllm, &[])],
            &plain
        ),
        None,
        "a worker without the label reads float32"
    );
    assert_eq!(
        shared_image_wire_dtype(
            &[
                bf16("http://a"),
                worker(
                    "http://e",
                    RuntimeType::Vllm,
                    &[
                        ("multimodal_encoder_dtype", "bfloat16"),
                        (MM_DEVICE_DO_NORMALIZE_LABEL, "true"),
                    ],
                ),
            ],
            &plain
        ),
        None
    );
    assert_eq!(
        shared_image_wire_dtype(&[worker("http://f", RuntimeType::Sglang, &[])], &plain),
        Some(EncoderDtype::Float32)
    );
}

/// Disagreeing SGLang workers must defer the wire write until selection.
#[test]
fn sglang_mixed_precision_fleet_defers_the_wire_write() {
    let narrow = worker(
        "http://half",
        RuntimeType::Sglang,
        &[("multimodal_encoder_dtype", "float16")],
    );
    let legacy = worker("http://legacy", RuntimeType::Sglang, &[]);
    let policy = DtypeOverrides::default();
    assert_eq!(
        shared_image_wire_dtype(std::slice::from_ref(&narrow), &policy),
        Some(EncoderDtype::Float16)
    );
    assert_eq!(shared_image_wire_dtype(&[narrow, legacy], &policy), None);
}

/// A real ZMQ handshake supplies SGLang's dtype without an explicit label;
/// overrides win and old or disagreeing schedulers retain float32. Before the handshake,
/// the one-pass planner must not freeze a layout in the legacy precision.
#[tokio::test]
async fn sglang_zmq_dtype_comes_from_the_handshake_unless_overridden() {
    use engine_zmq_client::{
        mock_engine::{connect_to_frontend, default_ready_response},
        protocol::handshake::EngineCoreReadyResponse,
        EngineId,
    };

    use crate::routers::grpc::zmq_client::{EosTokenIds, ZmqEngineClient};

    let pending: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("ipc:///pending")
            .runtime_type(RuntimeType::Sglang)
            .connection_mode(ConnectionMode::Zmq)
            .health_config(no_health_check())
            .build(),
    );
    let policy = DtypeOverrides::default();
    let pending_backend = connected_backend_client(pending.as_ref());
    assert_eq!(shared_image_wire_dtype(&[pending], &policy), None);

    for (advertised, explicit, expected) in [
        (vec![Some("bfloat16")], None, EncoderDtype::BFloat16),
        (vec![Some("float16"); 2], None, EncoderDtype::Float16),
        (vec![Some("bfloat16"); 2], None, EncoderDtype::BFloat16),
        (vec![None; 2], None, EncoderDtype::Float32),
        (
            vec![Some("bfloat16"), Some("float32")],
            None,
            EncoderDtype::Float32,
        ),
        (
            vec![Some("float16"), Some("bfloat16")],
            None,
            EncoderDtype::Float32,
        ),
        (vec![Some("bfloat16"), None], None, EncoderDtype::Float32),
        (vec![None, Some("bfloat16")], None, EncoderDtype::Float32),
        (
            vec![Some("bfloat16"), Some("float32")],
            Some("float16"),
            EncoderDtype::Float16,
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs"), ep("in"), ep("out"));
        let count = advertised.len();
        let ranks = advertised.into_iter().enumerate().map(|(rank, dtype)| {
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(rank as u32),
                EngineCoreReadyResponse {
                    multimodal_encoder_dtype: dtype.map(str::to_string),
                    ..default_ready_response()
                },
            )
        });
        let (client, engines) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                count,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::Sglang,
                Duration::from_secs(10)
            ),
            futures::future::join_all(ranks),
        );
        let _engines: Vec<_> = engines.into_iter().map(Result::unwrap).collect();
        let mut builder = BasicWorkerBuilder::new(ep("worker"))
            .runtime_type(RuntimeType::Sglang)
            .connection_mode(ConnectionMode::Zmq)
            .health_config(no_health_check())
            .engine_connection(BackendClient::Zmq(client.unwrap()));
        if let Some(dtype) = explicit {
            builder = builder.label("multimodal_encoder_dtype", dtype);
        }
        let worker: Arc<dyn Worker> = Arc::new(builder.build());
        let workers = WorkerSelection::Single {
            worker: Arc::clone(&worker).into(),
        };
        assert_eq!(mm_sglang_encoder_input_dtype(Some(&workers)), expected);
        // Retain the snapshot from before initialization while the worker
        // now has a live backend. The planner must not resample that backend.
        assert!(worker.engine_connection_if_initialized().is_some());
        assert_eq!(
            image_wire_dtype_with_backend(worker.as_ref(), &policy, pending_backend.as_deref()),
            explicit.map(|_| expected),
        );
        assert_eq!(shared_image_wire_dtype(&[worker], &policy), Some(expected));
    }
}

/// The stages see the fleet's wire dtype by default: the one-pass wire
/// write is on unless its switch turns it off, and then they get no
/// dtype whatever the fleet agrees on (every request takes the two-pass
/// path) while the resolution itself still answers. The default is read
/// off the settings' built-in values; the wrapper is checked against the
/// process's own setting, so the test holds whether or not the
/// environment it runs in switches the write off. The switch's spellings
/// are the settings module's tests.
#[test]
fn the_one_pass_wire_write_is_the_default_and_its_switch_turns_it_off() {
    use smg_worker::identity::basic::ConnectionMode;

    use crate::routers::{
        common::snapshot::WorkerRegistry,
        grpc::{mode::Mode, multimodal::MultimodalSettings},
    };

    let registry = WorkerRegistry::new();
    for url in ["grpc://regular-1:8000", "grpc://regular-2:8000"] {
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new(url)
                    .model(ModelCard::new("qwen3-vl"))
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(RuntimeType::TokenSpeed)
                    .health_config(no_health_check())
                    .build(),
            ))
            .expect("register");
    }
    let pools = Mode::Regular.multimodal_wire_pools();
    assert_eq!(
        shared_image_wire_dtype_over(&registry, "qwen3-vl", pools, &DtypeOverrides::default()),
        Some(EncoderDtype::BFloat16),
        "the fleet agrees on one dtype"
    );
    assert!(
        MultimodalSettings::default().one_pass_wire.value,
        "the one-pass wire write is on by default"
    );
    let expected = if mm_settings().one_pass_wire.value {
        Some(EncoderDtype::BFloat16)
    } else {
        None
    };
    assert_eq!(
        mm_shared_image_wire_dtype_over(&registry, "qwen3-vl", pools),
        expected,
        "the stages see the fleet's wire dtype unless the switch turns the one-pass write off"
    );
}

/// The up-front pixel bound holds only where every worker the selection
/// can pick takes the tensors inline in the gateway's gRPC message: one
/// worker on the SHM transport, a TRT-LLM pool or a ZMQ wire drops it.
#[test]
fn the_image_wire_cap_holds_only_for_an_inline_grpc_fleet() {
    use openai_protocol::worker::WorkerSpec;
    use smg_worker::identity::basic::ConnectionMode;

    use crate::routers::{
        common::snapshot::WorkerRegistry,
        grpc::{mode::Mode, proto_wrapper::mm_shm_dev_writable},
    };

    let pools = Mode::Regular.multimodal_wire_pools();
    let fleet = |specs: Vec<WorkerSpec>, rdma_lane_stages: &dyn Fn() -> bool| {
        let registry = WorkerRegistry::new();
        for spec in specs {
            registry
                .register(Arc::new(
                    BasicWorkerBuilder::from_spec(spec)
                        .model(ModelCard::new("qwen3-vl"))
                        .worker_type(WorkerType::Regular)
                        .health_config(no_health_check())
                        .build(),
                ))
                .expect("register");
        }
        image_wire_cap_over(&registry, "qwen3-vl", pools, rdma_lane_stages)
    };
    let grpc = |url: &str, runtime: RuntimeType, transport: Option<TransportMode>| WorkerSpec {
        connection_mode: ConnectionMode::Grpc,
        runtime_type: runtime,
        multimodal_tensor_transport: transport,
        ..WorkerSpec::new(url)
    };
    let inline_fleet = || {
        vec![
            grpc(
                "grpc://inline-1:8000",
                RuntimeType::Vllm,
                Some(TransportMode::Inline),
            ),
            grpc(
                "grpc://inline-2:8000",
                RuntimeType::TokenSpeed,
                Some(TransportMode::Inline),
            ),
            grpc("grpc://inline-3:8000", RuntimeType::Sglang, None),
        ]
    };
    assert_eq!(
        fleet(inline_fleet(), &|| false),
        Some(WIRE_MESSAGE_CAP),
        "an inline gRPC fleet keeps the bound"
    );
    assert_eq!(
        fleet(inline_fleet(), &|| true),
        None,
        "a live RDMA exporter takes every TokenSpeed input over NIXL, whatever the worker's mode"
    );
    assert_eq!(
        fleet(
            vec![
                grpc(
                    "grpc://inline-1:8000",
                    RuntimeType::Vllm,
                    Some(TransportMode::Inline),
                ),
                grpc("grpc://rdma-1:8000", RuntimeType::Vllm, Some(TransportMode::Rdma)),
            ],
            &|| true,
        ),
        Some(WIRE_MESSAGE_CAP),
        "vLLM never pulls RDMA: its pixels ride inline under a live exporter and the rdma mode alike"
    );
    if mm_shm_dev_writable() {
        assert_eq!(
            fleet(
                vec![
                    grpc(
                        "grpc://inline-1:8000",
                        RuntimeType::Vllm,
                        Some(TransportMode::Inline)
                    ),
                    grpc(
                        "grpc://shm-1:8000",
                        RuntimeType::Vllm,
                        Some(TransportMode::Shm)
                    ),
                ],
                &|| false
            ),
            None,
            "one worker on the SHM transport drops it"
        );
    }
    assert_eq!(
        fleet(
            vec![grpc("grpc://trt-1:8000", RuntimeType::Trtllm, None)],
            &|| false
        ),
        None,
        "a TRT-LLM pool takes the files' bytes"
    );
    assert_eq!(
        fleet(
            vec![WorkerSpec {
                connection_mode: ConnectionMode::Zmq,
                runtime_type: RuntimeType::Vllm,
                ..WorkerSpec::new("ipc:///vllm-1")
            }],
            &|| false
        ),
        None,
        "a ZMQ wire has no such framing"
    );
    assert_eq!(fleet(vec![], &|| false), None, "no candidate, no bound");
}

/// A build without the lane (the default build: the exporter constructs but
/// exports nothing, so every byte rides inline) never reads the lane as
/// staging, whatever the lane's switches and listener say, and so keeps
/// the bound for TokenSpeed; the build is asked before any exporter is
/// built. The build's own answer is pinned at compile time (a const
/// block, as clippy asks of an assertion on a constant), the reading's
/// at run time.
#[cfg(not(feature = "mm-rdma"))]
#[test]
fn a_build_without_the_rdma_lane_keeps_the_bound() {
    const {
        assert!(
            !STAGES_OVER_RDMA,
            "the default build carries the no-op exporter"
        );
    };
    assert!(
        !rdma_lane_stages(),
        "the inert exporter is never read as a staging lane"
    );
}

/// The dtype is resolved over the pools a mode's request can reach:
/// the regular workers for the regular pipeline, the prefill and decode
/// pools for a disaggregated one. A fleet whose pools read another width
/// than its regular workers keeps the per-destination write (`None`)
/// over the union, never a wire for the wrong pool.
#[test]
fn the_wire_dtype_is_resolved_over_the_pools_a_modes_pixels_can_reach() {
    use smg_worker::identity::basic::ConnectionMode;

    use crate::routers::{common::snapshot::WorkerRegistry, grpc::mode::Mode};

    let plain = DtypeOverrides::default();
    let registry = WorkerRegistry::new();
    let worker = |url: &str, worker_type: WorkerType, runtime: RuntimeType| -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("qwen3-vl"))
                .worker_type(worker_type)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(runtime)
                .health_config(no_health_check())
                .build(),
        )
    };
    // Regular workers reading bfloat16 (the TokenSpeed default) next to
    // prefill and decode workers pinned to float32 (SGLang).
    for worker in [
        worker(
            "grpc://regular-1:8000",
            WorkerType::Regular,
            RuntimeType::TokenSpeed,
        ),
        worker(
            "grpc://regular-2:8000",
            WorkerType::Regular,
            RuntimeType::TokenSpeed,
        ),
        worker(
            "grpc://prefill-1:8000",
            WorkerType::Prefill,
            RuntimeType::Sglang,
        ),
        worker(
            "grpc://decode-1:8000",
            WorkerType::Decode,
            RuntimeType::Sglang,
        ),
    ] {
        registry.register(worker).expect("register");
    }
    let over =
        |pools: &[RoutingPool]| shared_image_wire_dtype_over(&registry, "qwen3-vl", pools, &plain);
    assert_eq!(
        over(Mode::Regular.multimodal_wire_pools()),
        Some(EncoderDtype::BFloat16)
    );
    assert_eq!(
        over(Mode::PrefillDecode.multimodal_wire_pools()),
        Some(EncoderDtype::Float32)
    );
    assert_eq!(
        over(Mode::EncodePrefillDecode.multimodal_wire_pools()),
        Some(EncoderDtype::Float32),
        "no encode worker registered: the prefill and decode workers decide"
    );
    assert_eq!(
        over(&[
            RoutingPool::GrpcPipelineRegular,
            RoutingPool::GrpcPrefill,
            RoutingPool::GrpcDecode,
        ]),
        None,
        "pools that disagree with the regular workers keep the per-destination write"
    );
    assert_eq!(
        over(&[RoutingPool::GrpcEncode]),
        None,
        "an empty pool has nothing to agree on"
    );
}

#[test]
fn modality_dtype_precedes_global_then_worker_with_bfloat16_fallback() {
    assert_eq!(
        resolve_mm_encoder_input_dtype(Some("float16"), Some("float32"), Some("bfloat16")),
        ResolvedDtype::Process("float16")
    );
    assert_eq!(
        resolve_mm_encoder_input_dtype(None, Some("float32"), Some("bfloat16")),
        ResolvedDtype::Process("float32")
    );
    assert_eq!(
        resolve_mm_encoder_input_dtype(None, None, Some("bfloat16")),
        ResolvedDtype::Worker("bfloat16")
    );
    assert_eq!(
        resolve_mm_encoder_input_dtype(None, None, None),
        ResolvedDtype::Process("bfloat16")
    );
}

/// An assembly keeps the process's own names borrowed and a worker's
/// label as its one copy.
#[test]
fn an_assembly_keeps_a_workers_label_as_its_one_copy() {
    assert!(matches!(
        ResolvedDtype::Process("float32").kept(),
        Cow::Borrowed("float32")
    ));
    assert!(matches!(
        ResolvedDtype::Worker("bfloat16").kept(),
        Cow::Owned(dtype) if dtype == "bfloat16"
    ));
}

/// The vLLM wire starts from the widest dtype and narrows only when asked,
/// the opposite of the TokenSpeed default above. A worker reads the bytes at
/// whatever width it knows, so guessing a narrower one turns every media
/// request against an older worker into a failure or worse.
#[test]
fn vllm_dtype_stays_float32_until_something_asks_for_narrower() {
    assert_eq!(
        resolve_mm_vllm_encoder_input_dtype(None, None),
        ResolvedDtype::Process("float32")
    );
    assert_eq!(
        resolve_mm_vllm_encoder_input_dtype(None, Some("bfloat16")),
        ResolvedDtype::Worker("bfloat16")
    );
    assert_eq!(
        resolve_mm_vllm_encoder_input_dtype(Some("float16"), Some("bfloat16")),
        ResolvedDtype::Process("float16")
    );
}

/// The derived slot TTL must strictly exceed the worker's max hold, so the
/// reaper can never reclaim a slot the worker could still be reading (a late
/// READ against a recycled slot cross-wires images). The crate's `SlotPool`
/// tests cover the mechanics; this pins the gateway's TTL-derivation policy.
#[test]
fn derived_rdma_slot_ttl_exceeds_worker_max_hold() {
    assert!(
        derive_rdma_slot_ttl() > worker_max_hold(),
        "slot_ttl {:?} must exceed worker_max_hold {:?} or a late READ cross-wires",
        derive_rdma_slot_ttl(),
        worker_max_hold()
    );
}

/// The `SMG_RDMA_SLOT_TTL_S` override is honored only when it exceeds the worker
/// hold; a too-small (or absent) value falls back to the derived `hold + slack`,
/// so an operator can never silently reintroduce the recycled-under-READ bug.
#[test]
fn slot_ttl_override_must_exceed_hold() {
    let hold = Duration::from_secs(180);
    // Override above the hold is honored verbatim.
    assert_eq!(resolve_slot_ttl(Some(600), hold), Duration::from_secs(600));
    // Override at or below the hold is rejected -> derived hold + slack.
    assert_eq!(
        resolve_slot_ttl(Some(180), hold),
        hold + RDMA_SLOT_TTL_SLACK
    );
    assert_eq!(resolve_slot_ttl(Some(10), hold), hold + RDMA_SLOT_TTL_SLACK);
    // No override -> derived.
    assert_eq!(resolve_slot_ttl(None, hold), hold + RDMA_SLOT_TTL_SLACK);
}

/// A slot count that would blow past the arena cap is reduced to fit (>= 1),
/// while a normal request passes through untouched.
#[test]
fn pool_slots_capped_to_arena_max() {
    let slot_bytes = 1024 * 1024 * 1024; // 1 GiB
    let capped = clamp_pool_slots(1_000_000, slot_bytes);
    assert_eq!(capped, MAX_RDMA_ARENA_BYTES / slot_bytes);
    assert!(capped >= 1, "must keep at least one slot");
    assert_eq!(clamp_pool_slots(64, 32 * 1024 * 1024), 64);
}

#[test]
#[cfg(target_os = "linux")]
fn local_shm_namespace_id_resolves_on_linux() {
    // /proc/.../boot_id and /dev/shm both exist on the Linux CI/runtime
    // image, so the token must resolve to `<boot_id>:<st_dev>`. If it ever
    // returned None, `auto` would silently never enable SHM.
    let id = local_shm_namespace_id().expect("shm namespace id should resolve on Linux");
    assert!(
        id.contains(':'),
        "token must be <boot_id>:<st_dev>, got {id:?}"
    );
    let dev = id.rsplit(':').next().unwrap();
    assert!(
        dev.parse::<u64>().is_ok(),
        "st_dev component must be numeric, got {id:?}"
    );
}
