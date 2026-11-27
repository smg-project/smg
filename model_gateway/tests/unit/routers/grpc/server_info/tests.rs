use axum::{body::to_bytes, http::StatusCode, response::Response};
use mock_worker::config::SchedulerWire;
use openai_protocol::worker::{RuntimeType, WorkerType};
use prost_types::{value::Kind, ListValue, Struct};
use serde_json::{json, Value};
use smg_grpc_client::{engine_labels::ServerInfo, mlx_proto, trtllm_proto};
use smg_worker::identity::basic::{ConnectionMode, Worker};

use super::{answer, body_of, json_of, record};
use crate::routers::grpc::{backend_client::BasicWorkerBuilder, test_engine::mock_engine_on_wire};

/// The status and the JSON body of `response`.
async fn parts(response: Response) -> (StatusCode, Value) {
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body is read");
    (
        status,
        serde_json::from_slice(&body).expect("the body is JSON"),
    )
}

/// The status and the JSON body the route gives for `worker`.
async fn read(worker: &dyn Worker) -> (StatusCode, Value) {
    parts(answer(worker).await).await
}

/// A value of a struct as an engine sends it.
fn wire(kind: Kind) -> prost_types::Value {
    prost_types::Value { kind: Some(kind) }
}

/// The route answers `200` with the engine's message on every gRPC wire
/// the mock engine speaks, each asked over the gateway's own client.
#[tokio::test]
async fn the_route_answers_the_engines_message_on_every_wire() {
    for (spoken, runtime, field) in [
        (SchedulerWire::Vllm, RuntimeType::Vllm, None),
        (
            SchedulerWire::TokenSpeed,
            RuntimeType::TokenSpeed,
            Some("tokenspeed_version"),
        ),
        (
            SchedulerWire::Sglang,
            RuntimeType::Sglang,
            Some("sglang_version"),
        ),
    ] {
        let (url, engine) = mock_engine_on_wire("m", spoken).await;
        let worker = BasicWorkerBuilder::new(url.replacen("http://", "grpc://", 1))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(runtime)
            .build();
        let (status, body) = read(&worker).await;
        engine.abort();
        assert_eq!(status, StatusCode::OK, "{runtime:?}: {body}");
        assert!(body.is_object(), "{runtime:?}: {body}");
        if let Some(field) = field {
            assert!(
                body.get(field).is_some(),
                "{runtime:?} names {field}: {body}"
            );
        }
    }
}

/// The two wires the mock engine does not speak: each one's message
/// reads `200` under its own field names.
#[tokio::test]
async fn a_message_of_flat_fields_reads_under_its_own_names() {
    let trtllm = ServerInfo::Trtllm(trtllm_proto::GetServerInfoResponse {
        version: "1.2".to_string(),
        world_size: 2,
        ..Default::default()
    });
    let (status, body) = parts(body_of(&trtllm)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body.get("version"), Some(&json!("1.2")), "{body}");
    assert_eq!(body.get("world_size"), Some(&json!(2)), "{body}");

    let mlx = ServerInfo::Mlx(mlx_proto::GetServerInfoResponse {
        server_type: "mlx-grpc".to_string(),
        active_requests: 3,
        ..Default::default()
    });
    let (status, body) = parts(body_of(&mlx)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body.get("server_type"), Some(&json!("mlx-grpc")), "{body}");
    assert_eq!(body.get("active_requests"), Some(&json!(3)), "{body}");
}

/// A worker the route does not ask reads `501`: one on the ZMQ link and
/// an HTTP worker.
#[tokio::test]
async fn a_worker_that_is_not_asked_reads_501() {
    for mode in [ConnectionMode::Zmq, ConnectionMode::Http] {
        let worker = BasicWorkerBuilder::new("http://127.0.0.1:9")
            .worker_type(WorkerType::Regular)
            .connection_mode(mode)
            .build();
        let (status, body) = read(&worker).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{mode:?}: {body}");
        assert_eq!(
            body,
            json!({"error": {
                "type": "server_error",
                "code": "server_info_not_implemented",
                "message": "Server info not implemented",
                "param": null,
            }}),
            "{mode:?}"
        );
    }
}

/// A struct of the engine reads as a JSON object, each value in its JSON
/// type: a whole number as an integer, any other as a double.
#[test]
fn a_struct_of_the_engine_reads_as_a_json_object() {
    let inner = Struct {
        fields: [("depth".to_string(), wire(Kind::NumberValue(2.0)))].into(),
    };
    let fields = Struct {
        fields: [
            ("none".to_string(), prost_types::Value { kind: None }),
            ("null".to_string(), wire(Kind::NullValue(0))),
            ("flag".to_string(), wire(Kind::BoolValue(true))),
            ("count".to_string(), wire(Kind::NumberValue(4096.0))),
            ("ratio".to_string(), wire(Kind::NumberValue(0.25))),
            ("name".to_string(), wire(Kind::StringValue("m".to_string()))),
            (
                "list".to_string(),
                wire(Kind::ListValue(ListValue {
                    values: vec![wire(Kind::NumberValue(1.0)), wire(Kind::BoolValue(false))],
                })),
            ),
            ("record".to_string(), wire(Kind::StructValue(inner))),
        ]
        .into(),
    };
    assert_eq!(
        record(&fields),
        json!({
            "none": null,
            "null": null,
            "flag": true,
            "count": 4096,
            "ratio": 0.25,
            "name": "m",
            "list": [1, false],
            "record": {"depth": 2},
        })
    );
    // A whole number no double holds exactly stays a double; one that is
    // not finite reads null.
    assert_eq!(json_of(&wire(Kind::NumberValue(1e300))), json!(1e300));
    assert_eq!(json_of(&wire(Kind::NumberValue(f64::NAN))), Value::Null);
}
