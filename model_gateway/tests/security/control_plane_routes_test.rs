//! The control-plane route list the API document is generated from, against
//! the production route table: every listed route sits behind the
//! control-plane authentication and answers its method, and every route the
//! table registers is listed. A route added to one side without the other
//! fails here.

use std::collections::BTreeSet;

use axum::http::StatusCode;
use openapi_gen::CONTROL_PLANE_ROUTES;
use smg::server::GATED_ROUTES;

use super::control_plane_roles_test::{control_plane_auth, status, ADMIN_KEY};
use crate::common::{
    test_app::create_test_app_with_context_and_auth, AppTestContext, TestRouterConfig,
    TestWorkerConfig,
};

/// A listed path with its parameters filled, so the request matches the route.
fn concrete(path: &str) -> String {
    path.replace("{worker_id}", "no-such-worker")
        .replace("{module_uuid}", "no-such-module")
        .replace("{tokenizer_id}", "no-such-tokenizer")
}

#[tokio::test]
async fn every_listed_route_sits_behind_the_control_plane_authentication_and_answers_its_method() {
    let config = TestRouterConfig::round_robin(4354);
    let ctx = AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20354)]).await;
    let app = create_test_app_with_context_and_auth(
        ctx.router.clone(),
        ctx.app_context.clone(),
        Some(control_plane_auth()),
    );

    for (method, path) in CONTROL_PLANE_ROUTES {
        let path = concrete(path);
        // Without a credential the control-plane layer answers 401; an
        // unknown path answers 404 from the fallback, outside the layer.
        assert_eq!(
            status(&app, method, &path, None).await,
            StatusCode::UNAUTHORIZED,
            "{method} {path} is not behind the control-plane authentication"
        );
        // With the admin key a method the path does not take answers 405.
        assert_ne!(
            status(&app, method, &path, Some(ADMIN_KEY)).await,
            StatusCode::METHOD_NOT_ALLOWED,
            "{method} {path} is listed and not registered"
        );
    }

    ctx.shutdown().await;
}

/// The server source the route groups are read from.
const SERVER_SOURCE: &str = include_str!("../../src/server.rs");

/// The region of the server source from one marker to the next, the first
/// one included.
#[expect(
    clippy::expect_used,
    reason = "test helper: a missing marker in the server source is a test bug"
)]
fn source_between(start_marker: &str, end_marker: &str) -> &'static str {
    let start = SERVER_SOURCE.find(start_marker).expect(start_marker);
    let end = SERVER_SOURCE[start..].find(end_marker).expect(end_marker);
    &SERVER_SOURCE[start..start + end]
}

/// The (method, path) pairs a region of the server source registers: every
/// `.route("<path>", <methods>)` in it.
#[expect(
    clippy::expect_used,
    reason = "test helper: a route call without a path is a test bug"
)]
fn routes_in(region: &str) -> BTreeSet<(String, String)> {
    let mut routes = BTreeSet::new();
    for call in region.split(".route(").skip(1) {
        let path = call.split('"').nth(1).expect("a route call names its path");
        for method in ["get", "post", "put", "patch", "delete"] {
            let builder = &call[call.find('"').map_or(0, |at| at + path.len() + 2)..];
            let registered = builder.match_indices(method).any(|(at, _)| {
                let before = builder[..at].chars().last();
                let after = builder[at + method.len()..].chars().next();
                after == Some('(') && !before.is_some_and(|c| c.is_alphanumeric() || c == '_')
            });
            if registered {
                routes.insert((method.to_uppercase(), path.to_string()));
            }
        }
    }
    routes
}

/// The (method, path) pairs `build_apps` registers on its control-plane
/// groups: between the admin group's start and the authentication closure.
fn registered_routes() -> BTreeSet<(String, String)> {
    routes_in(source_between(
        "let admin_routes = Router::new()",
        "let apply_control_plane_auth",
    ))
}

/// The (method, path) pairs `gated_control_plane_routes` registers: the
/// function's body, up to its closing brace.
fn gated_group_routes() -> BTreeSet<(String, String)> {
    routes_in(source_between("fn gated_control_plane_routes()", "\n}\n"))
}

#[test]
fn every_route_the_gated_group_registers_is_listed() {
    let listed: BTreeSet<(String, String)> = GATED_ROUTES
        .iter()
        .map(|(method, path)| (method.to_string(), path.to_string()))
        .collect();
    assert_eq!(
        gated_group_routes(),
        listed,
        "the gated group's routes and GATED_ROUTES differ"
    );
}

#[test]
fn every_route_the_table_registers_is_listed() {
    // The gated group is a function of its own, outside the parsed region;
    // the test above reads it against its list both ways, so the list stands
    // in for the group here.
    let mut registered = registered_routes();
    registered.extend(
        GATED_ROUTES
            .iter()
            .map(|(method, path)| (method.to_string(), path.to_string())),
    );
    let listed: BTreeSet<(String, String)> = CONTROL_PLANE_ROUTES
        .iter()
        .map(|(method, path)| (method.to_string(), path.to_string()))
        .collect();

    assert!(
        registered.len() >= 20,
        "the route groups were read: {registered:?}"
    );
    let unlisted: Vec<_> = registered.difference(&listed).collect();
    assert!(
        unlisted.is_empty(),
        "registered on the control plane and missing from the API document's list: {unlisted:?}"
    );
    let unregistered: Vec<_> = listed.difference(&registered).collect();
    assert!(
        unregistered.is_empty(),
        "listed for the API document and not registered: {unregistered:?}"
    );
}
