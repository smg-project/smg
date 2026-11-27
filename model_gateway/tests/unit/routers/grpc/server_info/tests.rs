use axum::{body::to_bytes, http::StatusCode, response::Response};
use mock_worker::{config::SchedulerWire, overrides::InfoAnswer};
use openai_protocol::worker::{RuntimeType, WorkerType};
use prost_types::{value::Kind, ListValue, Struct};
use serde_json::{json, Value};
use smg_grpc_client::{
    engine_labels::ServerInfo, mlx_proto, sglang_proto, tokenspeed_proto, trtllm_proto,
};
use smg_worker::identity::basic::{ConnectionMode, Worker};

use super::{answer, body_of, json_of, record, within};
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

/// A gRPC worker on a mock engine of the TokenSpeed wire, with the engine's
/// port and its server task.
async fn asked_worker() -> (impl Worker, u16, tokio::task::JoinHandle<()>) {
    let (url, engine) = mock_engine_on_wire("m", SchedulerWire::TokenSpeed).await;
    let port = url
        .rsplit(':')
        .next()
        .and_then(|port| port.parse().ok())
        .expect("the engine has a port");
    let worker = BasicWorkerBuilder::new(url.replacen("http://", "grpc://", 1))
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Grpc)
        .runtime_type(RuntimeType::TokenSpeed)
        .build();
    (worker, port, engine)
}

/// An engine that does not offer the call reads `501`, as a worker that is
/// not asked does.
#[tokio::test]
async fn an_engine_without_the_call_reads_501() {
    let (worker, port, engine) = asked_worker().await;
    mock_worker::overrides::for_port(port).set_server_info(InfoAnswer::Unimplemented);
    let (status, body) = read(&worker).await;
    engine.abort();
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    assert_eq!(
        body["error"]["code"], "server_info_not_implemented",
        "{body}"
    );
}

/// An engine that fails the call reads `502`: the worker was asked, and its
/// failure is not the answer of a worker without the call.
#[tokio::test]
async fn an_engine_that_fails_the_call_reads_502() {
    let (worker, port, engine) = asked_worker().await;
    mock_worker::overrides::for_port(port).set_server_info(InfoAnswer::Failing);
    let (status, body) = read(&worker).await;
    engine.abort();
    assert_eq!(status, StatusCode::BAD_GATEWAY, "{body}");
    assert_eq!(body["error"]["code"], "server_info_failed", "{body}");
}

/// A worker that sends no answer within the bound reads `504`: the listener
/// takes the connection and never answers, so the bound alone ends the wait.
#[tokio::test]
async fn no_answer_within_the_bound_reads_504() {
    let silent = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a listener binds");
    let address = silent.local_addr().expect("the listener has an address");
    let worker = BasicWorkerBuilder::new(format!("grpc://{address}"))
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Grpc)
        .runtime_type(RuntimeType::TokenSpeed)
        .build();
    let bound = std::time::Duration::from_millis(20);
    let (status, body) = parts(within(&worker, bound).await).await;
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
    assert_eq!(body["error"]["code"], "server_info_timeout", "{body}");
}

/// A struct of `fields` as an engine sends it.
fn sent<const N: usize>(fields: [(&str, Kind); N]) -> Struct {
    Struct {
        fields: fields
            .into_iter()
            .map(|(name, kind)| (name.to_string(), wire(kind)))
            .collect(),
    }
}

/// The body leaves out every field of a struct whose name marks a secret,
/// at every depth, on both wires that carry a struct, and keeps the others:
/// a name that only holds the word `token` is no secret's.
#[tokio::test]
async fn a_field_named_as_a_secret_is_left_out_at_every_depth() {
    let text = |value: &str| Kind::StringValue(value.to_string());
    let args = sent([
        ("api_key", text("k-1")),
        ("ADMIN_API_KEY", text("k-2")),
        ("hf_token", text("t-1")),
        ("token", text("t-2")),
        ("tp_size", Kind::NumberValue(2.0)),
        ("max_total_tokens", Kind::NumberValue(4096.0)),
        ("tokenizer", text("m")),
        (
            "store",
            Kind::StructValue(sent([("db_password", text("p-1")), ("host", text("h"))])),
        ),
        (
            "peers",
            Kind::ListValue(ListValue {
                values: vec![wire(Kind::StructValue(sent([
                    ("client_secret", text("s-1")),
                    ("name", text("n")),
                ])))],
            }),
        ),
    ]);
    let kept = json!({
        "tp_size": 2,
        "max_total_tokens": 4096,
        "tokenizer": "m",
        "store": {"host": "h"},
        "peers": [{"name": "n"}],
    });
    let facts = sent([
        ("rl_control_api_key", text("k-3")),
        ("waiting", Kind::NumberValue(1.0)),
    ]);
    for info in [
        ServerInfo::TokenSpeed(Box::new(tokenspeed_proto::GetServerInfoResponse {
            server_args: Some(args.clone()),
            scheduler_info: Some(facts.clone()),
            ..Default::default()
        })),
        ServerInfo::Sglang(Box::new(sglang_proto::GetServerInfoResponse {
            server_args: Some(args.clone()),
            scheduler_info: Some(facts.clone()),
            ..Default::default()
        })),
    ] {
        let (status, body) = parts(body_of(&info)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["server_args"], kept, "{body}");
        assert_eq!(body["scheduler_info"], json!({"waiting": 1}), "{body}");
    }
}

/// The route leaves such a field out of what an engine sends over either
/// wire that carries a struct, and keeps the argument beside it.
#[tokio::test]
async fn the_route_leaves_out_a_field_named_as_a_secret() {
    for (spoken, runtime) in [
        (SchedulerWire::TokenSpeed, RuntimeType::TokenSpeed),
        (SchedulerWire::Sglang, RuntimeType::Sglang),
    ] {
        let (url, engine) = mock_engine_on_wire("m", spoken).await;
        let port: u16 = url
            .rsplit(':')
            .next()
            .and_then(|port| port.parse().ok())
            .expect("the engine has a port");
        let patch = json!({"api_key": "k-1", "shown_arg": "v"});
        mock_worker::overrides::for_port(port)
            .patch_info(patch.as_object().expect("an object of arguments"));
        let worker = BasicWorkerBuilder::new(url.replacen("http://", "grpc://", 1))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(runtime)
            .build();
        let (status, body) = read(&worker).await;
        engine.abort();
        assert_eq!(status, StatusCode::OK, "{runtime:?}: {body}");
        let args = &body["server_args"];
        assert_eq!(args["shown_arg"], "v", "{runtime:?}: {body}");
        assert!(args.get("api_key").is_none(), "{runtime:?}: {body}");
    }
}
