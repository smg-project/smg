//! What the local workflow steps' tests build a worker's labels from, and
//! the engine metadata endpoints a test serves.

use std::{collections::HashMap, net::Ipv4Addr};

use tokio::net::TcpListener;

/// The label map `pairs` spell, each name and value owned.
pub(crate) fn labels_of(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect()
}

/// An engine's HTTP metadata endpoints as a test serves them: each route
/// answers every GET with its status and JSON body, any other path a 404.
/// Returns the base URL; the stub serves for as long as the runtime that
/// spawned it runs.
pub(crate) async fn serve_json_routes(
    routes: Vec<(&'static str, http::StatusCode, serde_json::Value)>,
) -> String {
    let mut router = axum::Router::new();
    for (path, status, body) in routes {
        router = router.route(
            path,
            axum::routing::get(move || {
                let body = body.clone();
                async move { (status, axum::Json(body)) }
            }),
        );
    }
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .expect("a loopback port of the system's choice");
    let addr = listener.local_addr().expect("the bound address");
    #[expect(
        clippy::disallowed_methods,
        reason = "a test fixture: the stub lives as long as the test's runtime"
    )]
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    format!("http://{addr}")
}
