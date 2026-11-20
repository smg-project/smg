//! What the local workflow steps' tests build a worker's labels from, and
//! the engine metadata endpoints a test serves.

use std::{collections::HashMap, net::Ipv4Addr, sync::Arc};

use parking_lot::Mutex;
use tokio::net::TcpListener;

/// The routes a test engine serves: a path, its status and its JSON body.
type JsonRoutes = Vec<(&'static str, http::StatusCode, serde_json::Value)>;

/// The body of a route a test moves between two reads.
pub(crate) type MovingBody = Arc<Mutex<serde_json::Value>>;

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
pub(crate) async fn serve_json_routes(routes: JsonRoutes) -> String {
    serve(json_router(routes)).await
}

/// As [`serve_json_routes`], with one more route at `path` whose body the
/// test moves between two reads. Returns the base URL and the body's cell.
pub(crate) async fn serve_json_routes_and_a_moving_one(
    routes: JsonRoutes,
    path: &'static str,
    body: serde_json::Value,
) -> (String, MovingBody) {
    let cell = Arc::new(Mutex::new(body));
    let read = Arc::clone(&cell);
    let moving = axum::routing::get(move || {
        let body = read.lock().clone();
        async move { axum::Json(body) }
    });
    let base = serve(json_router(routes).route(path, moving)).await;
    (base, cell)
}

/// The router that answers each of `routes`, and any other path a 404.
fn json_router(routes: JsonRoutes) -> axum::Router {
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
    router
}

/// Serve `router` on a loopback port and return its base URL.
async fn serve(router: axum::Router) -> String {
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
