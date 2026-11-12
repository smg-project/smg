//! The control-plane route list the API document is generated from, against
//! the production route table: every listed route sits behind the
//! control-plane authentication and answers its method, and every route the
//! table registers is listed. A route added to one side without the other
//! fails here. The RL list is read against the RL crate's router the same
//! way: the gateway nests that router under `/v1/rl` only with the RL flag,
//! so the route table above never carries it.

use std::collections::BTreeSet;

use axum::http::StatusCode;
use openapi_gen::{CONTROL_PLANE_ROUTES, RL_CONTROL_PLANE_ROUTES};
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

/// The RL crate's source: the gateway nests its router under `/v1/rl`,
/// behind the control-plane authentication and outside the groups above.
const RL_SOURCE: &str = include_str!("../../../crates/rl/src/lib.rs");

/// The region of a source from one marker to the next, the first one
/// included. A missing marker is a test bug; the panic names it.
#[expect(
    clippy::panic,
    reason = "test helper: a missing marker in a source is a test bug, named in the panic"
)]
fn between(source: &'static str, start_marker: &str, end_marker: &str) -> &'static str {
    let start = source
        .find(start_marker)
        .unwrap_or_else(|| panic!("the source has no {start_marker:?}"));
    let end = source[start..]
        .find(end_marker)
        .unwrap_or_else(|| panic!("the source has no {end_marker:?} after {start_marker:?}"));
    &source[start..start + end]
}

/// The region of the server source from one marker to the next, the first
/// one included.
fn source_between(start_marker: &str, end_marker: &str) -> &'static str {
    between(SERVER_SOURCE, start_marker, end_marker)
}

/// `source` without its `//` line comments (outside string literals), so a
/// route call a comment holds or mentions is not read as a registered one.
fn without_line_comments(source: &str) -> String {
    let mut kept = String::with_capacity(source.len());
    for line in source.lines() {
        let mut in_string = false;
        let mut escaped = false;
        let mut previous = None;
        let mut cut = line.len();
        for (at, c) in line.char_indices() {
            match c {
                '\\' if in_string && !escaped => {
                    escaped = true;
                    continue;
                }
                '"' if !escaped => in_string = !in_string,
                '/' if !in_string && previous == Some('/') => {
                    cut = at - 1;
                    break;
                }
                _ => {}
            }
            escaped = false;
            previous = Some(c);
        }
        kept.push_str(&line[..cut]);
        kept.push('\n');
    }
    kept
}

/// The (method, path) pairs a region of a source registers: every
/// `.route("<path>", <methods>)` in it outside its comments.
#[expect(
    clippy::expect_used,
    reason = "test helper: a route call without a path is a test bug"
)]
fn routes_in(region: &str) -> BTreeSet<(String, String)> {
    let mut routes = BTreeSet::new();
    let region = without_line_comments(region);
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

/// The RL router's path parameters by the names the API document uses: the
/// router names them for its handlers, the document for its clients, and the
/// list test reads the router's paths through this table.
const RL_PARAMETER_NAMES: &[(&str, &str)] = &[("{id}", "{worker_id}"), ("{*path}", "{path}")];

/// The (method, path) pairs `smg_rl::router` registers, as the gateway mounts
/// them: under `/v1/rl`, with the parameter names the API document uses.
fn rl_router_routes() -> BTreeSet<(String, String)> {
    routes_in(between(RL_SOURCE, "pub fn router<", "\n}\n"))
        .into_iter()
        .map(|(method, path)| {
            let path = RL_PARAMETER_NAMES
                .iter()
                .fold(path, |path, (router, document)| {
                    path.replace(router, document)
                });
            (method, format!("/v1/rl{path}"))
        })
        .collect()
}

#[test]
fn a_route_call_in_a_comment_is_not_a_registered_route() {
    let region = r#"
        Router::new()
            // .route("/retired", get(handler))
            .route("/live", get(handler)) // once .route("/noted", post(handler))
            .route("/paths//doubled", delete(handler))
            .route("/quoted", post(handler))
    "#;
    let registered = |method: &str, path: &str| (method.to_string(), path.to_string());

    assert_eq!(
        routes_in(region),
        BTreeSet::from([
            registered("DELETE", "/paths//doubled"),
            registered("GET", "/live"),
            registered("POST", "/quoted"),
        ])
    );
}

#[test]
fn every_route_the_rl_router_registers_is_listed() {
    let listed: BTreeSet<(String, String)> = RL_CONTROL_PLANE_ROUTES
        .iter()
        .map(|(method, path)| (method.to_string(), path.to_string()))
        .collect();
    assert_eq!(
        rl_router_routes(),
        listed,
        "the RL router's routes and RL_CONTROL_PLANE_ROUTES differ"
    );
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
