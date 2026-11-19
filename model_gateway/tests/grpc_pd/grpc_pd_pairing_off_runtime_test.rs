//! A prefill and a decode on two runtimes cannot be served on the gRPC wire:
//! the pair's one request is in the prefill runtime's message type, and the
//! decode's client sends only its own. With pairing off such a fleet is
//! refused by name, in the lenient mode's words, and a pair on one runtime
//! is served as before, whatever the handoff's facts say.

use std::{sync::Arc, time::Duration};

use common::grpc_harness::{
    build_grpc_router, grpc_router_config, grpc_worker, serve_mock_worker_with,
};
use mock_worker::config::{Config, SchedulerWire};
use openai_protocol::model_card::ModelCard;
use smg::{middleware::TenantRequestMeta, tenant::TenantKey};
use smg_worker::{
    identity::basic::{RuntimeType, Worker, WorkerType},
    pd::pairing::PdPairingMode,
};

use super::completion_request;
use crate::common;

const MODEL: &str = "pd-pairing-off-runtime-test-model";

/// The wires the mock engine speaks, each with the runtime its worker
/// registers as.
const WIRES: [(SchedulerWire, RuntimeType); 3] = [
    (SchedulerWire::TokenSpeed, RuntimeType::TokenSpeed),
    (SchedulerWire::Vllm, RuntimeType::Vllm),
    (SchedulerWire::Sglang, RuntimeType::Sglang),
];

/// One mock engine on `wire`, answering three tokens at once.
async fn engine(wire: SchedulerWire) -> u16 {
    serve_mock_worker_with(|port| Config {
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        grpc_wire: wire,
        model_id: MODEL.to_string(),
        tokenizer_path: MODEL.to_string(),
        gen_delay: Duration::ZERO,
        output_tokens: 3,
        realistic: false,
        ..Config::default()
    })
    .await
}

/// One completion through a PD gRPC router under `mode`, round robin on both
/// roles, over one prefill and one decode, each on its wire with its
/// runtime and under the pairing protocol its operator named in
/// `protocols`, if any: the status and the body's `error` member (null on a
/// success).
async fn answer(
    mode: PdPairingMode,
    (prefill_wire, prefill_runtime): (SchedulerWire, RuntimeType),
    (decode_wire, decode_runtime): (SchedulerWire, RuntimeType),
    protocols: Option<(&str, &str)>,
) -> (http::StatusCode, serde_json::Value) {
    let (prefill_port, decode_port) = (engine(prefill_wire).await, engine(decode_wire).await);
    let config = grpc_router_config()
        .prefill_decode_mode(vec![], vec![])
        .round_robin_policy()
        .pd_pairing_mode(mode)
        .build_unchecked();
    let worker = |port: u16,
                  worker_type: WorkerType,
                  runtime: RuntimeType,
                  protocol: Option<&str>|
     -> Arc<dyn Worker> {
        let builder = grpc_worker(port, worker_type, runtime, ModelCard::new(MODEL));
        Arc::new(match protocol {
            Some(protocol) => builder.label("pairing_protocol", protocol).build(),
            None => builder.build(),
        })
    };
    let workers = vec![
        worker(
            prefill_port,
            WorkerType::Prefill,
            prefill_runtime,
            protocols.map(|(prefill, _)| prefill),
        ),
        worker(
            decode_port,
            WorkerType::Decode,
            decode_runtime,
            protocols.map(|(_, decode)| decode),
        ),
    ];
    let router = build_grpc_router(config, MODEL, workers).await;
    let tenant = TenantRequestMeta::new(TenantKey::new("pd-pairing-off-runtime-test-tenant"));
    let response = router
        .route_completion(None, &tenant, completion_request(MODEL, 8), MODEL)
        .await;
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap_or_default();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_default();
    (status, body["error"].clone())
}

/// Every ordered pair of the three wires under pairing off: a pair on one
/// runtime is served; a pair across two is refused with the code and the
/// message the lenient mode gives the same fleet, naming the runtime and
/// both pools' keys. The request's task used to panic at the dispatch there.
#[tokio::test]
async fn pairing_off_refuses_a_pair_across_two_runtimes_by_name() {
    for prefill in WIRES {
        for decode in WIRES {
            let cell = format!("{:?} to {:?}", prefill.1, decode.1);
            let (status, error) = answer(PdPairingMode::Off, prefill, decode, None).await;
            if prefill.1 == decode.1 {
                assert_eq!(status, http::StatusCode::OK, "{cell}: {error}");
                continue;
            }
            assert_eq!(
                status,
                http::StatusCode::SERVICE_UNAVAILABLE,
                "{cell}: {error}"
            );
            assert_eq!(error["code"], "no_compatible_pd_pair", "{cell}: {error}");
            let message = error["message"].as_str().unwrap_or_default();
            assert!(
                message.contains(r#"mismatch on ["runtime"]"#),
                "{cell}: {error}"
            );
            for runtime in [prefill.1, decode.1] {
                assert!(message.contains(runtime.as_str()), "{cell}: {error}");
            }
            assert_eq!(
                (status, error),
                answer(PdPairingMode::Lenient, prefill, decode, None).await,
                "{cell}: one cause, one answer in both modes"
            );
        }
    }
}

/// Off still compares nothing of the handoff. A pair on one runtime whose
/// operators named two pairing protocols is refused under the lenient mode
/// and served under off: the router reads the mode its configuration names.
#[tokio::test]
async fn pairing_off_serves_a_pair_on_one_runtime_that_the_lenient_mode_refuses() {
    let protocols = Some(("protocol-a", "protocol-b"));
    let (status, error) = answer(PdPairingMode::Lenient, WIRES[0], WIRES[0], protocols).await;
    assert_eq!(status, http::StatusCode::SERVICE_UNAVAILABLE, "{error}");
    assert_eq!(error["code"], "no_compatible_pd_pair", "{error}");
    let message = error["message"].as_str().unwrap_or_default();
    assert!(
        message.contains(r#"mismatch on ["pairing_protocol"]"#),
        "{error}"
    );

    let (status, error) = answer(PdPairingMode::Off, WIRES[0], WIRES[0], protocols).await;
    assert_eq!(status, http::StatusCode::OK, "{error}");
}
