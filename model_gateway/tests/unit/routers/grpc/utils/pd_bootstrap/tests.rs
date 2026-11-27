use std::net::TcpListener;

use metrics_exporter_prometheus::PrometheusBuilder;
use openai_protocol::worker::{RuntimeType, WorkerType};
use smg_worker::test_support::RefusedAddress;

use super::*;
use crate::routers::grpc::{
    backend_client::BasicWorkerBuilder, proto_wrapper::ENGINE_ERROR_MESSAGE,
};

/// A prefill side on the loopback host whose partners are handed `port`.
fn prefill_on(port: u16) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new("grpc://127.0.0.1:30000")
            .worker_type(WorkerType::Prefill)
            .runtime_type(RuntimeType::Sglang)
            .bootstrap_port(Some(port))
            .build(),
    )
}

/// The rendered rendezvous series.
fn rendezvous_series(rendered: &str) -> Vec<&str> {
    rendered
        .lines()
        .filter(|line| line.starts_with("smg_pd_rendezvous_failures_total{"))
        .collect()
}

/// The engine's own report behind a prefill side whose bootstrap port
/// nothing listens on is the dead rendezvous: counted under the request
/// phase on the prefill side, worded for the client without the address.
#[tokio::test]
async fn the_engines_own_report_behind_a_dead_bootstrap_port_is_a_rendezvous_failure() {
    let refused = RefusedAddress::reserve();
    let port = refused.port();
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let dead = {
        let _guard = metrics::set_default_local_recorder(&recorder);
        dead_rendezvous_of(
            Some(&prefill_on(port)),
            &tonic::Status::internal(ENGINE_ERROR_MESSAGE),
        )
        .await
    }
    .expect("the dead port is the rendezvous failure");
    assert_eq!((dead.0.host.as_str(), dead.0.port), ("127.0.0.1", port));
    let words = dead.to_string();
    assert!(
        words.starts_with("the prefill side's bootstrap port did not accept a connection (")
            && words.ends_with("), so the sides could not rendezvous"),
        "{words}"
    );
    assert!(
        !words.contains("127.0.0.1"),
        "the address stays out of the client's words: {words}"
    );
    let rendered = handle.render();
    let counted = rendezvous_series(&rendered);
    assert_eq!(counted.len(), 1, "{rendered}");
    assert!(
        counted[0].contains(r#"leg="prefill""#)
            && counted[0].contains(r#"phase="request""#)
            && counted[0].ends_with(" 1"),
        "{rendered}"
    );
}

/// Nothing is probed or counted without both a report that carries no
/// reason and a dead port: a status with the engine's words (classed by
/// phase above), a port that accepts, no prefill side, or a prefill side
/// no partner is handed a port for.
#[tokio::test]
async fn nothing_is_probed_without_a_reasonless_report_and_a_dead_port() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let live = listener.local_addr().expect("address").port();
    let side_channel: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://127.0.0.1:30000")
            .worker_type(WorkerType::Prefill)
            .runtime_type(RuntimeType::Vllm)
            .kv_connector("NixlConnector")
            .build(),
    );
    let dead = RefusedAddress::reserve();
    let report = tonic::Status::internal(ENGINE_ERROR_MESSAGE);
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let verdicts = {
        let _guard = metrics::set_default_local_recorder(&recorder);
        [
            dead_rendezvous_of(Some(&prefill_on(live)), &report).await,
            dead_rendezvous_of(
                Some(&prefill_on(dead.port())),
                &tonic::Status::internal(
                    "Decode handshake failed for request rank=0 req.rid='r2' \
                         req.bootstrap_room=42",
                ),
            )
            .await,
            dead_rendezvous_of(
                Some(&prefill_on(dead.port())),
                &tonic::Status::unavailable("connection reset"),
            )
            .await,
            dead_rendezvous_of(None, &report).await,
            dead_rendezvous_of(Some(&side_channel), &report).await,
        ]
    };
    assert!(verdicts.iter().all(Option::is_none), "{verdicts:?}");
    assert!(
        rendezvous_series(&handle.render()).is_empty(),
        "{}",
        handle.render()
    );
}

/// The texts a disaggregated engine's scheduler and transfer layer write
/// for a side they retire, as their sources spell them, and the phase
/// each names.
#[test]
fn an_engine_reported_rendezvous_failure_is_classed_by_phase() {
    let cases = [
        // The prefill waited for the decode's KV indices past its deadline.
        (
            "Prefill bootstrap failed for request rank=0 req.rid='r1' req.bootstrap_room=41 \
                 with exception KVTransferError(bootstrap_room=41): Request 41 timed out after \
                 120.0s in KVPoll.Bootstrapping",
            RendezvousPhase::Bootstrap,
        ),
        // The decode never reached the prefill's bootstrap server.
        (
            "Decode handshake failed for request rank=0 req.rid='r2' req.bootstrap_room=42 \
                 with exception KVTransferError(bootstrap_room=42): Failed to get kvcache from \
                 prefill instance, it might be dead",
            RendezvousPhase::Bootstrap,
        ),
        (
            "Some requests timed out when bootstrapping, which means prefill instances fail \
                 to receive the KV indices from the decode instance of this request.",
            RendezvousPhase::Bootstrap,
        ),
        // The sides met; the handoff itself failed.
        (
            "Decode transfer failed for request rank=0 req.rid='r3' req.bootstrap_room=43 \
                 with exception KVTransferError(bootstrap_room=43): KV transfer failed",
            RendezvousPhase::Transfer,
        ),
        (
            "Prefill transfer failed for request rank=0 req.rid='r4' req.bootstrap_room=44",
            RendezvousPhase::Transfer,
        ),
        (
            "Failed to get kvcache from prefill instance, it might be dead",
            RendezvousPhase::Transfer,
        ),
    ];
    for (message, phase) in cases {
        assert_eq!(
            RendezvousPhase::of(&tonic::Status::internal(message)),
            Some(phase),
            "{message}"
        );
        assert_eq!(
            RendezvousPhase::of(&tonic::Status::unavailable(message)),
            Some(phase),
            "{message}"
        );
    }
}

#[test]
fn a_failure_that_does_not_name_the_rendezvous_is_not_classed() {
    for status in [
        tonic::Status::internal("engine finished the request with an error (see engine logs)"),
        tonic::Status::unavailable("connection reset"),
        tonic::Status::deadline_exceeded("decode step took too long"),
        tonic::Status::internal("h2 protocol error: error reading a body from connection"),
    ] {
        assert_eq!(RendezvousPhase::of(&status), None, "{status:?}");
    }
}

/// A request's own fault, and capacity pushback, are not rendezvous
/// failures whatever they mention.
#[test]
fn a_request_fault_is_not_a_rendezvous_failure_whatever_it_mentions() {
    for status in [
        tonic::Status::invalid_argument("bootstrap_room must be set for a disaggregated request"),
        tonic::Status::resource_exhausted("bootstrap queue full"),
    ] {
        assert_eq!(RendezvousPhase::of(&status), None, "{status:?}");
    }
}

#[test]
fn a_side_failure_that_names_the_rendezvous_is_counted_under_the_rendezvous_family() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        account_pd_side_failure(
            metrics_labels::WORKER_PREFILL,
            &tonic::Status::internal(
                "Prefill bootstrap failed for request rank=0 req.rid='r1' req.bootstrap_room=41",
            ),
        );
        account_pd_side_failure(
            metrics_labels::WORKER_DECODE,
            &tonic::Status::internal(
                "Decode transfer failed for request rank=0 req.rid='r3' req.bootstrap_room=43",
            ),
        );
        account_pd_side_failure(
            metrics_labels::WORKER_DECODE,
            &tonic::Status::unavailable("connection reset"),
        );
    });
    let rendered = handle.render();
    let counted: Vec<&str> = rendered
        .lines()
        .filter(|line| line.starts_with("smg_pd_rendezvous_failures_total{"))
        .collect();
    assert_eq!(counted.len(), 2, "{rendered}");
    assert!(
        counted.iter().any(|line| line.contains(r#"leg="prefill""#)
            && line.contains(r#"phase="bootstrap""#)
            && line.ends_with(" 1")),
        "{rendered}"
    );
    assert!(
        counted.iter().any(|line| line.contains(r#"leg="decode""#)
            && line.contains(r#"phase="transfer""#)
            && line.ends_with(" 1")),
        "{rendered}"
    );
}
