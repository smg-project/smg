use axum::{routing::get, Json, Router};
use serde_json::json;
use tokio::net::TcpListener;

/// An OIDC discovery endpoint over loopback. Returns the server's origin,
/// the issuer a document served there is requested for; the document names
/// `document_issuer(origin)` as its issuer and the origin's `/jwks.json` as
/// its key set.
#[expect(
    clippy::disallowed_methods,
    reason = "the server lives exactly as long as the test"
)]
pub(crate) async fn discovery_server(document_issuer: impl FnOnce(&str) -> String) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let document = json!({
        "issuer": document_issuer(&origin),
        "jwks_uri": format!("{origin}/jwks.json"),
    });
    let app = Router::new().route(
        "/.well-known/openid-configuration",
        get(move || {
            let document = document.clone();
            async move { Json(document) }
        }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    origin
}
