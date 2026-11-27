use smg_worker::test_support::RefusedAddress;

use super::fetch_models;
use crate::workflow::steps::local::test_support::serve_json_routes;

/// A model id the provider lists is refused like one a request carries:
/// the discovery fails with the door's sentence, which does not repeat
/// the id.
#[tokio::test]
async fn a_listed_model_id_with_a_control_character_fails_the_discovery() {
    let listed = serde_json::json!({ "data": [{ "id": "m\nINJECTED" }] });
    let url = serve_json_routes(vec![("/v1/models", http::StatusCode::OK, listed)]).await;

    let err = fetch_models(&url, None, None)
        .await
        .expect_err("the listed id is refused");

    assert_eq!(err, "models[].id must not contain control characters");
}

/// An address registered with a query, at a port nobody listens on: the
/// failure does not repeat the query. The HTTP client's own error text
/// ends with the request URL, the query kept.
#[tokio::test]
async fn a_refused_connection_goes_without_the_clients_url() {
    let refused = RefusedAddress::reserve();
    let url = format!("{}?token=s3cret", refused.http_url());

    let err = fetch_models(&url, None, None)
        .await
        .expect_err("nothing listens at the address");

    assert!(!err.contains("s3cret"), "{err}");
}

/// An address registered with a query that answers a body that is no
/// JSON: the failure names the address masked and does not repeat the
/// query. The HTTP client's text of a failed body read ends with the
/// request URL too.
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

    let err = fetch_models(&format!("http://127.0.0.1:{port}?token=s3cret"), None, None)
        .await
        .expect_err("a body that is no JSON is a failure");
    server.abort();

    assert!(
        err.contains(&format!(
            "Failed to parse models response from http://127.0.0.1:{port}: "
        )),
        "{err}"
    );
    assert!(!err.contains("s3cret"), "{err}");
}
