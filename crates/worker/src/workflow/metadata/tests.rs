//! The metadata read's cases.

use std::sync::Arc;

use super::*;
use crate::{
    test_support::{labels_of, RefusedAddress, ScriptedAnswers, ScriptedConnector},
    workflow::{
        host::testing::{serve_json_routes, worker_client},
        runtimes::HTTP_RUNTIMES,
    },
};

/// The path of the table's row whose metadata is the model list and its own
/// path.
const ROW_PATH: &str = "/version";

/// The info routes as registration reads them: a failed read is an empty
/// map.
async fn info_routes_labels(
    client: &Client,
    url: &str,
    api_key: Option<&str>,
) -> HashMap<String, String> {
    read_info_routes(client, url, api_key, false)
        .await
        .unwrap_or_default()
}

/// The model list and the row's path as registration reads them: a failed
/// read is an empty map.
async fn model_list_labels(
    client: &Client,
    url: &str,
    api_key: Option<&str>,
) -> HashMap<String, String> {
    read_model_list_and_path(client, url, api_key, false, ROW_PATH)
        .await
        .unwrap_or_default()
}

/// Every engine's spelling of the parallelism widths folds into the
/// canonical labels, and an existing canonical label is never overwritten.
#[test]
fn normalize_grpc_keys_canonicalises_every_parallelism_spelling() {
    let mut labels = labels_of(&[
        ("attn_tp_size", "2"),
        ("pipeline_parallel_size", "1"),
        ("data_parallel_size", "4"),
        ("context_parallel_size", "1"),
        ("block_size", "16"),
        ("uptime_seconds", "12.5"),
    ]);
    normalize_grpc_keys(&mut labels);
    assert_eq!(labels.get("tp_size").map(String::as_str), Some("2"));
    assert_eq!(labels.get("pp_size").map(String::as_str), Some("1"));
    assert_eq!(labels.get("dp_size").map(String::as_str), Some("4"));
    assert_eq!(labels.get("cp_size").map(String::as_str), Some("1"));
    assert!(!labels.contains_key("attn_tp_size"));
    assert_eq!(labels.get("page_size").map(String::as_str), Some("16"));
    assert!(!labels.contains_key("block_size"));
    assert!(!labels.contains_key("uptime_seconds"));

    // An engine that reports both keeps its own `tp_size`.
    let mut attn = labels_of(&[("tp_size", "8"), ("attn_tp_size", "2")]);
    normalize_grpc_keys(&mut attn);
    assert_eq!(attn.get("tp_size").map(String::as_str), Some("8"));
    assert!(!attn.contains_key("attn_tp_size"));

    let mut both = labels_of(&[("tp_size", "8"), ("tensor_parallel_size", "2")]);
    normalize_grpc_keys(&mut both);
    assert_eq!(both.get("tp_size").map(String::as_str), Some("8"));
}

fn label<'a>(labels: &'a HashMap<String, String>, key: &str) -> Option<&'a str> {
    labels.get(key).map(String::as_str)
}

/// The info routes' read asks `/server_info`, `/model_info` and
/// `/v1/models` and combines them: the engine's own labels first, the
/// model list filling in what they left out.
#[tokio::test]
async fn the_info_routes_read_combines_the_three_endpoints_with_the_engines_own_labels_first() {
    use http::StatusCode;

    let base = serve_json_routes(vec![
        (
            "/server_info",
            StatusCode::OK,
            serde_json::json!({
                "model_path": "org/model-a", "tp_size": 4, "dp_size": 1,
                "max_running_requests": 256, "version": "0.5.0",
            }),
        ),
        (
            "/model_info",
            StatusCode::OK,
            serde_json::json!({
                "model_path": "org/model-a", "tokenizer_path": "org/model-a-tokenizer",
                "is_generation": true,
            }),
        ),
        (
            "/v1/models",
            StatusCode::OK,
            serde_json::json!({"data": [
                {"id": "served-a", "root": "org/model-a-root", "max_model_len": 32768}
            ]}),
        ),
    ])
    .await;

    let labels = info_routes_labels(&worker_client(), &base, None).await;
    assert_eq!(
        label(&labels, "model_path"),
        Some("org/model-a"),
        "the engine's own model path, not the model list's root: {labels:?}"
    );
    assert_eq!(
        label(&labels, "tokenizer_path"),
        Some("org/model-a-tokenizer")
    );
    assert_eq!(label(&labels, "tp_size"), Some("4"));
    assert_eq!(label(&labels, "max_running_requests"), Some("256"));
    assert_eq!(label(&labels, "version"), Some("0.5.0"));
    assert_eq!(label(&labels, "is_generation"), Some("true"));
    assert_eq!(
        label(&labels, "served_model_name"),
        Some("served-a"),
        "the model list fills in the served name the engine's info left out"
    );
    assert_eq!(label(&labels, "max_model_len"), Some("32768"));
}

/// An endpoint gone to 404 is read at its older `/get_` spelling; one
/// that answers 404 under both spellings is one the engine does not
/// offer, tolerated by the strict read as by registration.
#[tokio::test]
async fn the_info_routes_read_takes_the_older_spelling_and_tolerates_an_endpoint_not_offered() {
    use http::StatusCode;

    let base = serve_json_routes(vec![
        (
            "/get_server_info",
            StatusCode::OK,
            serde_json::json!({"model_path": "org/model-b", "tp_size": 2}),
        ),
        (
            "/v1/models",
            StatusCode::OK,
            serde_json::json!({"data": [{"id": "served-b"}]}),
        ),
    ])
    .await;
    let client = worker_client();

    let tolerant = info_routes_labels(&client, &base, None).await;
    assert_eq!(label(&tolerant, "model_path"), Some("org/model-b"));
    assert_eq!(label(&tolerant, "tp_size"), Some("2"));
    assert_eq!(label(&tolerant, "served_model_name"), Some("served-b"));
    assert_eq!(
        label(&tolerant, "tokenizer_path"),
        None,
        "an endpoint not offered leaves no label: {tolerant:?}"
    );

    let strict = read_info_routes(&client, &base, None, true)
        .await
        .expect("an endpoint the engine does not offer fails no read");
    assert_eq!(strict, tolerant);
}

/// An endpoint that fails leaves a gap in the registration read and
/// fails a strict read, named.
#[tokio::test]
async fn an_endpoint_that_fails_leaves_a_gap_for_registration_and_fails_a_strict_read_by_name() {
    use http::StatusCode;

    let base = serve_json_routes(vec![
        (
            "/server_info",
            StatusCode::OK,
            serde_json::json!({"model_path": "org/model-c"}),
        ),
        (
            "/model_info",
            StatusCode::INTERNAL_SERVER_ERROR,
            serde_json::json!({"error": "down"}),
        ),
        (
            "/v1/models",
            StatusCode::OK,
            serde_json::json!({"data": []}),
        ),
    ])
    .await;
    let client = worker_client();

    let tolerant = info_routes_labels(&client, &base, None).await;
    assert_eq!(label(&tolerant, "model_path"), Some("org/model-c"));
    assert_eq!(label(&tolerant, "tokenizer_path"), None, "{tolerant:?}");

    let error = read_info_routes(&client, &base, None, true)
        .await
        .expect_err("a strict read fails on a failing endpoint");
    assert!(
        error.starts_with("model_info: status 500"),
        "the error names the endpoint and its answer: {error}"
    );
}

/// The model list's read takes the model's identity and window from
/// `/v1/models` and the build from the row's path; a path the engine
/// does not offer leaves no label and fails no read.
#[tokio::test]
async fn the_model_list_read_takes_the_model_list_and_the_rows_path() {
    use http::StatusCode;

    let models = serde_json::json!({"data": [
        {"id": "served-d", "root": "org/model-d", "max_model_len": 8192}
    ]});
    let client = worker_client();

    let base = serve_json_routes(vec![
        ("/v1/models", StatusCode::OK, models.clone()),
        (
            "/version",
            StatusCode::OK,
            serde_json::json!({"version": "0.31.0"}),
        ),
    ])
    .await;
    let labels = model_list_labels(&client, &base, None).await;
    assert_eq!(label(&labels, "model_path"), Some("org/model-d"));
    assert_eq!(label(&labels, "served_model_name"), Some("served-d"));
    assert_eq!(label(&labels, "max_model_len"), Some("8192"));
    assert_eq!(label(&labels, "version"), Some("0.31.0"));

    let without_version = serve_json_routes(vec![("/v1/models", StatusCode::OK, models)]).await;
    let labels = read_model_list_and_path(&client, &without_version, None, true, ROW_PATH)
        .await
        .expect("a version endpoint the engine does not offer fails no read");
    assert_eq!(label(&labels, "model_path"), Some("org/model-d"));
    assert_eq!(label(&labels, "version"), None, "{labels:?}");
}

/// An engine that refuses the connection leaves registration with no
/// labels and fails a strict read on the first endpoint, named.
#[tokio::test]
async fn an_engine_that_refuses_the_connection_leaves_registration_empty_and_fails_a_strict_read() {
    let refused = RefusedAddress::reserve();
    let base = refused.http_url();
    let client = worker_client();

    assert!(model_list_labels(&client, &base, None).await.is_empty());
    assert!(info_routes_labels(&client, &base, None).await.is_empty());

    let error = read_model_list_and_path(&client, &base, None, true, ROW_PATH)
        .await
        .expect_err("a strict read fails on an engine it cannot reach");
    assert!(error.starts_with("/v1/models: "), "{error}");
    let error = read_info_routes(&client, &base, None, true)
        .await
        .expect_err("a strict read fails on an engine it cannot reach");
    assert!(error.starts_with("server_info: "), "{error}");
}

#[test]
fn the_server_info_surfaces_max_running_requests_as_a_label() {
    // Subset of an actual /server_info response. The full payload has
    // many more fields; `serde(deny_unknown_fields)` is off by default so
    // they're silently ignored.
    for (dp_size, attn_dp_size) in [(4, None), (1, Some(4))] {
        let body = serde_json::json!({
            "model_path": "Qwen/Qwen3-8B",
            "tp_size": 4,
            "dp_size": dp_size,
            "attn_dp_size": attn_dp_size,
            "max_running_requests": 256,
            "context_length": 32768,
        });
        let info: ServerInfo = serde_json::from_value(body).expect("deserialize ServerInfo");
        assert_eq!(info.max_running_requests, Some(256));
        assert_eq!(info.attn_dp_size, attn_dp_size);

        let labels = flat_labels(&info);
        assert_eq!(
            labels.get("max_running_requests").map(String::as_str),
            Some("256")
        );
    }
}

/// The attention-DP split reaches the labels on this path too, so a
/// registration of one worker a rank over HTTP gives each its share of
/// the window as the gRPC path does.
#[test]
fn the_server_info_surfaces_the_attention_dp_split() {
    let body = serde_json::json!({
        "model_path": "Qwen/Qwen3-8B",
        "tp_size": 16,
        "dp_size": 8,
        "enable_dp_attention": true,
        "attn_dp_size": 8,
        "max_running_requests": 16384,
    });
    let info: ServerInfo = serde_json::from_value(body).expect("deserialize ServerInfo");
    let labels = flat_labels(&info);
    assert_eq!(
        labels.get("enable_dp_attention").map(String::as_str),
        Some("true")
    );
    assert_eq!(labels.get("attn_dp_size").map(String::as_str), Some("8"));

    // Absent fields make no label, so a rank worker of an engine that
    // reports neither keeps its window as it is.
    let plain: ServerInfo = serde_json::from_value(serde_json::json!({
        "model_path": "Qwen/Qwen3-8B",
        "tp_size": 1,
        "dp_size": 1,
        "max_running_requests": 256,
    }))
    .expect("deserialize ServerInfo");
    let labels = flat_labels(&plain);
    assert!(
        !labels.contains_key("enable_dp_attention") && !labels.contains_key("attn_dp_size"),
        "{labels:?}"
    );
}

#[test]
fn the_server_infos_max_running_requests_is_optional() {
    // Older engine versions or special configurations may omit the field.
    let body = serde_json::json!({
        "model_path": "Qwen/Qwen3-8B",
        "tp_size": 1,
    });
    let info: ServerInfo = serde_json::from_value(body).expect("deserialize ServerInfo");
    assert_eq!(info.max_running_requests, None);

    let labels = flat_labels(&info);
    assert!(!labels.contains_key("max_running_requests"));
}

#[tokio::test]
async fn the_info_routes_read_uses_the_model_lists_identity() {
    use http::StatusCode;
    use serde_json::json;

    let base = serve_json_routes(vec![(
        "/v1/models",
        StatusCode::OK,
        json!({
            "data": [{
                "id": "test-model",
                "max_model_len": 4096,
                "object": "model",
                "owned_by": "nvidia",
                "root": "test-root"
            }],
            "object": "list"
        }),
    )])
    .await;

    let labels = info_routes_labels(&worker_client(), &base, None).await;

    assert_eq!(
        labels.get("served_model_name").map(String::as_str),
        Some("test-model")
    );
    assert_eq!(
        labels.get("model_path").map(String::as_str),
        Some("test-root")
    );
    assert_eq!(
        labels.get("max_model_len").map(String::as_str),
        Some("4096")
    );
}

/// A registration reads tolerantly: a server info that fails to read leaves
/// a gap, the engine's data-parallel size in it, so the worker registers as
/// one worker. The failure is said in one warn line that names the route.
/// A server info the engine does not offer is no failure, and no such line
/// is written for it.
#[tokio::test]
#[tracing_test::traced_test]
async fn a_tolerant_read_says_a_server_info_that_failed_to_read() {
    let said = "the labels are read without it";
    let not_offered = serve_json_routes(vec![(
        "/model_info",
        http::StatusCode::OK,
        serde_json::json!({"model_path": "org/model-a"}),
    )])
    .await;
    let labels = info_routes_labels(&worker_client(), &not_offered, None).await;
    assert_eq!(
        labels.get("model_path").map(String::as_str),
        Some("org/model-a")
    );
    assert!(!logs_contain(said));

    let failing = serve_json_routes(vec![
        (
            "/server_info",
            http::StatusCode::INTERNAL_SERVER_ERROR,
            serde_json::json!({"dp_size": 4}),
        ),
        (
            "/model_info",
            http::StatusCode::OK,
            serde_json::json!({"model_path": "org/model-a"}),
        ),
    ])
    .await;
    let labels = info_routes_labels(&worker_client(), &failing, None).await;
    assert_eq!(labels.get("dp_size"), None);
    assert_eq!(
        labels.get("model_path").map(String::as_str),
        Some("org/model-a")
    );
    assert!(logs_contain("server_info: status 500"));
    assert!(logs_contain(said));
}

/// A credentialed worker that answers 401 (a wrong password): the error
/// text names the address masked, never the secret. The status error is
/// the likeliest failure of such a worker, and it ends in a registration's
/// warning and a refresh's failure message.
#[tokio::test]
async fn a_status_error_names_the_address_masked() {
    use axum::{http::StatusCode, Router};
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    #[expect(
        clippy::disallowed_methods,
        reason = "test-only server answering 401; handle is aborted at test end"
    )]
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new().fallback(|| async { (StatusCode::UNAUTHORIZED, "no") }),
        )
        .await
        .unwrap();
    });

    let err = read_info_routes(
        &worker_client(),
        &format!("http://user:secret@127.0.0.1:{port}"),
        None,
        true,
    )
    .await
    .expect_err("401 is a failure");
    server.abort();

    assert!(!err.contains("secret"), "{err}");
    assert!(!err.contains("user:"), "{err}");
    assert!(
        err.contains(&format!(
            "401 Unauthorized from http://***@127.0.0.1:{port}/server_info"
        )),
        "{err}"
    );
}

/// An address registered with a query, at a port nobody listens on: the
/// failure does not repeat the query. The HTTP client's own error text
/// ends with the request URL, the query kept.
#[tokio::test]
async fn a_refused_connection_goes_without_the_clients_url() {
    let refused = RefusedAddress::reserve();
    let url = format!("{}?token=s3cret", refused.http_url());

    let err = read_info_routes(&worker_client(), &url, None, true)
        .await
        .expect_err("nothing listens at the address");

    assert!(!err.contains("s3cret"), "{err}");
}

/// An address registered with a query that answers a body that is no
/// JSON: the failure does not repeat the query. The HTTP client's text of
/// a failed body read ends with the request URL too.
#[tokio::test]
async fn an_unparsed_answer_goes_without_the_clients_url() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    #[expect(
        clippy::disallowed_methods,
        reason = "test-only server answering no JSON; handle is aborted at test end"
    )]
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            axum::Router::new().fallback(|| async { "not json" }),
        )
        .await
        .unwrap();
    });

    let err = read_info_routes(
        &worker_client(),
        &format!("http://127.0.0.1:{port}?token=s3cret"),
        None,
        true,
    )
    .await
    .expect_err("a body that is no JSON is a failure");
    server.abort();

    assert!(err.contains("Failed to parse"), "{err}");
    assert!(!err.contains("s3cret"), "{err}");
}

/// The row of a runtime says how its metadata is read: the first row's
/// runtime by the model list and the row's path, every other runtime by the
/// info routes, a runtime with no row among them.
#[tokio::test]
async fn the_row_of_the_runtime_says_how_its_metadata_is_read() {
    use http::StatusCode;

    let base = serve_json_routes(vec![
        (
            "/server_info",
            StatusCode::OK,
            serde_json::json!({"model_path": "org/by-the-info-routes"}),
        ),
        (
            "/v1/models",
            StatusCode::OK,
            serde_json::json!({"data": [{"id": "served-a", "root": "org/by-the-model-list"}]}),
        ),
        (
            ROW_PATH,
            StatusCode::OK,
            serde_json::json!({"version": "9.9"}),
        ),
    ])
    .await;
    let client = worker_client();
    let connector = ScriptedConnector::default();
    let rows = [
        (HTTP_RUNTIMES[0].runtime.as_str(), "org/by-the-model-list"),
        (HTTP_RUNTIMES[1].runtime.as_str(), "org/by-the-info-routes"),
        ("a-runtime-with-no-row", "org/by-the-info-routes"),
    ];
    for (runtime, model_path) in rows {
        let (labels, detected) = discover_engine_labels(
            &client,
            &connector,
            &base,
            ConnectionMode::Http,
            runtime,
            None,
            true,
        )
        .await
        .unwrap();
        assert_eq!(label(&labels, "model_path"), Some(model_path), "{runtime}");
        assert_eq!(detected, None, "an HTTP read confirms no runtime");
    }
    assert_eq!(connector.connects(), 0);
}

fn describing(
    model: Result<&[(&str, &str)], tonic::Code>,
    server: Result<&[(&str, &str)], tonic::Code>,
) -> ScriptedConnector {
    ScriptedConnector::default().with_engine(
        GRPC_RUNTIME,
        Arc::new(ScriptedAnswers::describing(model, server)),
    )
}

/// The runtime the gRPC cases' engine answers as.
const GRPC_RUNTIME: RuntimeType = HTTP_RUNTIMES[0].runtime;
const GRPC_URL: &str = "grpc://worker-a:9000";

/// The gRPC read joins the model labels and the server labels under the
/// canonical keys, drops what is the engine's passing state, and says the
/// runtime it read as.
#[tokio::test]
async fn the_grpc_read_joins_the_model_and_the_server_labels_under_canonical_keys() {
    let connector = describing(
        Ok(&[("model_path", "org/model-a"), ("tensor_parallel_size", "4")]),
        Ok(&[("version", "1.2"), ("active_requests", "3")]),
    );

    let (labels, runtime) = fetch_grpc_metadata(&connector, GRPC_URL, GRPC_RUNTIME.as_str(), true)
        .await
        .unwrap();

    assert_eq!(runtime, GRPC_RUNTIME.as_str());
    assert_eq!(label(&labels, "model_path"), Some("org/model-a"));
    assert_eq!(label(&labels, "tp_size"), Some("4"));
    assert_eq!(label(&labels, "tensor_parallel_size"), None);
    assert_eq!(label(&labels, "version"), Some("1.2"));
    assert_eq!(label(&labels, "active_requests"), None);
}

/// An engine that offers no server labels keeps its model labels on a plain
/// read and on a strict one.
#[tokio::test]
async fn an_engine_without_server_labels_keeps_its_model_labels_on_both_reads() {
    let connector = describing(
        Ok(&[("model_path", "org/model-a")]),
        Err(tonic::Code::Unimplemented),
    );

    for strict in [false, true] {
        let (labels, _) = fetch_grpc_metadata(&connector, GRPC_URL, GRPC_RUNTIME.as_str(), strict)
            .await
            .unwrap();
        assert_eq!(
            label(&labels, "model_path"),
            Some("org/model-a"),
            "{strict}"
        );
    }
}

/// A server read that fails leaves a gap for registration and fails a strict
/// read by name. A model read that fails fails both reads by name.
#[tokio::test]
async fn a_failing_read_is_a_gap_or_a_failure_by_name() {
    let connector = describing(
        Ok(&[("model_path", "org/model-a")]),
        Err(tonic::Code::Unavailable),
    );
    let (labels, _) = fetch_grpc_metadata(&connector, GRPC_URL, GRPC_RUNTIME.as_str(), false)
        .await
        .unwrap();
    assert_eq!(label(&labels, "model_path"), Some("org/model-a"));
    let error = fetch_grpc_metadata(&connector, GRPC_URL, GRPC_RUNTIME.as_str(), true)
        .await
        .unwrap_err();
    assert!(
        error.starts_with("Failed to fetch gRPC server info: "),
        "{error}"
    );

    let connector = describing(Err(tonic::Code::Unavailable), Ok(&[("version", "1.2")]));
    for strict in [false, true] {
        let error = fetch_grpc_metadata(&connector, GRPC_URL, GRPC_RUNTIME.as_str(), strict)
            .await
            .unwrap_err();
        assert!(
            error.starts_with("Failed to fetch gRPC model info: "),
            "{strict}: {error}"
        );
    }
}

/// A text that names no runtime is refused before any connect, in the words
/// the read had before it moved.
#[tokio::test]
async fn a_text_that_names_no_runtime_is_refused_before_a_connect() {
    let connector = ScriptedConnector::default();

    let error = fetch_grpc_metadata(&connector, GRPC_URL, "no-such-runtime", false)
        .await
        .unwrap_err();

    assert_eq!(
        error,
        "Failed to connect to gRPC: Unknown runtime type: no-such-runtime"
    );
    assert_eq!(connector.connects(), 0);
}
