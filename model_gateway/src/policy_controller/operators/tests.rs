//! The policy document's routes through the gateway's control-plane router.

use std::{convert::Infallible, error::Error, sync::Arc};

use axum::{
    body::{to_bytes, Body, Bytes},
    http::{header, request::Builder, HeaderMap, Request, StatusCode},
    Router,
};
use serde_json::{json, Value};
use smg_auth::{ApiKeyEntry, ControlPlaneAuthConfig, ControlPlaneAuthState, Role};
use smg_policy_controller::PolicySpec;
use tokio::task::JoinSet;
use tower::ServiceExt;

use super::{document_bytes, DRY_RUNS_WAITING};
use crate::{
    app_context::AppContext,
    config::RouterConfig,
    middleware::AuthConfig,
    routers::gateway::Gateway,
    server::{build_app, health::ProbeState, AppState},
};

type Case = Result<(), Box<dyn Error>>;

const SPEC: &str = "/policy/spec";
const VALIDATE: &str = "/policy/spec:validate";

/// A document apart from the one a routing configuration means.
const OTHER: &str =
    "apiVersion: smg.policy/v1\nprofiles:\n  preview:\n    score: [least-load]\n    pick: best\n";

/// A routing configuration with the policy controller picked.
fn picked() -> RouterConfig {
    RouterConfig {
        policy_controller: true,
        ..RouterConfig::default()
    }
}

/// The server's state as a start leaves it under `config`.
async fn state(config: RouterConfig) -> Result<Arc<AppState>, String> {
    let context = Arc::new(AppContext::from_config(config, 60, None, None).await?);
    Ok(Arc::new(AppState {
        router: Arc::new(Gateway::new(Arc::clone(&context.worker_registry))),
        probe_state: ProbeState::new(Arc::clone(&context.inflight_tracker)),
        context,
        admission_queue: None,
        gateway: None,
        mesh_handler: None,
        mesh_adapters: None,
        secret_sources: Default::default(),
        secrets: None,
    }))
}

/// The gateway's app over `state` with control-plane authentication, which
/// mounts the gated routes, and a payload limit above a document's bound.
fn app(state: Arc<AppState>) -> Result<Router, header::InvalidHeaderName> {
    let keys = ControlPlaneAuthConfig {
        jwt: None,
        api_keys: vec![ApiKeyEntry::new("admin", "Admin", "admin-key", Role::Admin)],
        audit_enabled: false,
    };
    build_app(
        state,
        AuthConfig::new(None),
        AuthConfig::new(None),
        Some(ControlPlaneAuthState::new(keys, None)),
        2 * document_bytes(),
        vec![],
        vec![],
    )
}

/// The app under `config`.
async fn app_under(config: RouterConfig) -> Result<Router, Box<dyn Error>> {
    Ok(app(state(config).await?)?)
}

/// A request by the admin key.
fn by_admin(request: Builder) -> Builder {
    request.header(header::AUTHORIZATION, "Bearer admin-key")
}

/// A read of the document in force.
fn read() -> Result<Request<Body>, http::Error> {
    by_admin(Request::get(SPEC)).body(Body::empty())
}

/// A dry run of `document`, before its headers and its body.
fn dry_run() -> Builder {
    by_admin(Request::post(VALIDATE))
}

/// What `request` is answered: the status, the headers, and the body as
/// JSON where it is JSON.
async fn answered(
    app: &Router,
    request: Request<Body>,
) -> Result<(StatusCode, HeaderMap, Value), Box<dyn Error>> {
    let (parts, body) = app.clone().oneshot(request).await?.into_parts();
    let bytes = to_bytes(body, usize::MAX).await?;
    let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    Ok((parts.status, parts.headers, body))
}

/// The digest of the document `config` means: the standing one after a
/// start.
fn standing(config: &RouterConfig) -> Result<String, Box<dyn Error>> {
    Ok(config.policy_document()?.digest()?.to_string())
}

#[tokio::test]
async fn under_the_classic_policies_both_routes_refuse_by_one_code() -> Case {
    let state = state(RouterConfig::default()).await?;
    assert!(state.context.policy_operators.is_none());
    let app = app(state)?;
    for request in [read()?, dry_run().body(Body::from(OTHER))?] {
        let (status, _, refused) = answered(&app, request).await?;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(refused["code"], "POLICY_CONTROLLER_OFF");
    }
    Ok(())
}

#[tokio::test]
async fn the_document_in_force_is_the_one_the_print_option_writes() -> Case {
    let config = picked();
    let printed = serde_yaml::to_string(&config.policy_document()?)?;
    let digest = standing(&config)?;
    let (status, headers, in_force) = answered(&app_under(config).await?, read()?).await?;
    assert_eq!(status, StatusCode::OK);
    let etag = headers
        .get(header::ETAG)
        .and_then(|etag| etag.to_str().ok());
    assert_eq!(etag, Some(format!("\"{digest}\"").as_str()));
    let applied_at = in_force["applied_at"].as_str().ok_or("no time")?;
    chrono::DateTime::parse_from_rfc3339(applied_at)?;
    assert_eq!(in_force["generation"], 1);
    assert_eq!(in_force["digest"], digest);
    assert_eq!(in_force["format"], "yaml");
    assert_eq!(in_force["document"], printed);
    Ok(())
}

#[tokio::test]
async fn a_dry_run_tells_the_next_number_and_applies_nothing() -> Case {
    let config = picked();
    let stands = json!({"generation": 1, "digest": standing(&config)?});
    let offered = serde_yaml::from_str::<PolicySpec>(OTHER)?.digest()?;
    let app = app_under(config).await?;
    let (status, _, would_run) = answered(&app, dry_run().body(Body::from(OTHER))?).await?;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(would_run["standing"], stands);
    let next = json!({"generation": 2, "digest": offered.to_string()});
    assert_eq!(would_run["would_run"], next);
    assert_eq!(would_run["needs"]["before"], json!([]));
    let fed = &would_run["needs"]["after"][0];
    assert_eq!(
        (&fed["model"], &fed["phase"]),
        (&json!("*"), &json!("single"))
    );
    assert_eq!(fed["needs"]["loads"], true);

    let (_, _, in_force) = answered(&app, read()?).await?;
    assert_eq!(in_force["generation"], 1);
    Ok(())
}

/// The OpenAPI document gives each fact of a dry run's needs the type the
/// answer writes it in: the picker's sample is a number among flags.
#[tokio::test]
async fn a_dry_runs_needs_are_of_the_types_the_openapi_document_gives() -> Case {
    let app = app_under(picked()).await?;
    let (_, _, would_run) = answered(&app, dry_run().body(Body::from(OTHER))?).await?;
    let fed = would_run["needs"]["after"][0]["needs"]
        .as_object()
        .ok_or("no needs")?;
    assert!(fed.contains_key("sample") && fed.contains_key("loads"));
    let document = serde_json::to_value(openapi_gen::document()?)?;
    let answer = &document["paths"][VALIDATE]["post"]["responses"]["200"]["content"]
        ["application/json"]["schema"];
    let schema =
        &answer["properties"]["needs"]["properties"]["after"]["items"]["properties"]["needs"];
    for (name, value) in fed {
        let declared = schema["properties"]
            .get(name)
            .unwrap_or_else(|| &schema["additionalProperties"]);
        let written = match value {
            Value::Bool(_) => "boolean",
            Value::Number(_) => "integer",
            _ => "another type",
        };
        assert_eq!(declared["type"], written, "{name}");
    }
    Ok(())
}

/// A document that would not run is refused with the path of its field and
/// the generation that stands. The gateway's own header is denied to a dry
/// run's view as the start denies it.
#[tokio::test]
async fn a_document_that_would_not_run_names_its_field() -> Case {
    let mut config = picked();
    config.tenant_resolution.tenant_header_name = "x-case-tenant".to_owned();
    let stands = json!({"generation": 1, "digest": standing(&config)?});
    let app = app_under(config).await?;
    let unknown = OTHER.replace("least-load", "no-such-scorer");
    let denied = format!("{OTHER}view:\n  headers: [x-case-tenant]\n");
    for (document, field_path, error) in [
        (
            unknown,
            "profiles.preview.score",
            "profile \"preview\": unknown scorer \"no-such-scorer\"",
        ),
        (
            denied,
            "view.headers[0]",
            "view.headers[0]: header \"x-case-tenant\" is on the deny-list",
        ),
    ] {
        let (status, _, refused) = answered(&app, dry_run().body(Body::from(document))?).await?;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(refused["code"], "INVALID_DOCUMENT");
        assert_eq!(refused["field_path"], field_path);
        assert_eq!(refused["error"], error);
        assert_eq!(refused["standing"], stands);
    }
    Ok(())
}

/// The guard is one ETag, quoted as the read writes it or bare. `*` guards
/// nothing. Another digest is refused with the standing one, and a form
/// the door cannot compare is refused as such.
#[tokio::test]
async fn a_guard_is_one_etag_and_another_digest_is_refused_with_the_standing_one() -> Case {
    let config = picked();
    let digest = standing(&config)?;
    let app = app_under(config).await?;
    let another = format!("\"sha256:{}\"", "0".repeat(64));
    for (guard, expected) in [
        (format!("\"{digest}\""), StatusCode::OK),
        (digest.clone(), StatusCode::OK),
        ("*".to_owned(), StatusCode::OK),
        (another.clone(), StatusCode::PRECONDITION_FAILED),
        (format!("\"{digest}\", {another}"), StatusCode::BAD_REQUEST),
        (format!("W/\"{digest}\""), StatusCode::BAD_REQUEST),
    ] {
        let request = dry_run().header(header::IF_MATCH, &guard);
        let (status, _, answer) = answered(&app, request.body(Body::from(OTHER))?).await?;
        assert_eq!(status, expected, "If-Match: {guard}");
        if status == StatusCode::PRECONDITION_FAILED {
            assert_eq!(answer["code"], "PRECONDITION_FAILED");
            assert_eq!(answer["standing"]["digest"], digest);
        }
    }
    Ok(())
}

/// The body's form is its `Content-Type`'s, YAML where none is given: the
/// standing document as JSON is the standing document again, YAML read as
/// JSON is no document, and a type that names no form is refused unread.
#[tokio::test]
async fn a_documents_form_is_its_content_types() -> Case {
    let config = picked();
    let as_json = serde_json::to_string(&config.policy_document()?)?;
    let app = app_under(config).await?;
    for (content_type, document, expected) in [
        (
            "application/json; charset=utf-8",
            as_json.as_str(),
            StatusCode::OK,
        ),
        ("application/x-yaml", OTHER, StatusCode::OK),
        ("application/json", OTHER, StatusCode::UNPROCESSABLE_ENTITY),
        ("text/plain", OTHER, StatusCode::UNSUPPORTED_MEDIA_TYPE),
    ] {
        let request = dry_run().header(header::CONTENT_TYPE, content_type);
        let body = Body::from(document.to_owned());
        let (status, _, answer) = answered(&app, request.body(body)?).await?;
        assert_eq!(status, expected, "{content_type}");
        if content_type.starts_with("application/json;") {
            assert_eq!(answer["would_run"], answer["standing"]);
        }
    }
    Ok(())
}

/// A document's bound stands at the edge: a body at the bound reaches the
/// door, and one byte more is refused, declared or streamed.
#[tokio::test]
async fn a_body_past_a_documents_bound_is_refused_at_the_edge() -> Case {
    let app = app_under(picked()).await?;
    let at_bound = format!("{OTHER}#{}", "x".repeat(document_bytes() - OTHER.len() - 1));
    let (status, _, _) = answered(&app, dry_run().body(Body::from(at_bound.clone()))?).await?;
    assert_eq!(status, StatusCode::OK);

    let past = format!("{at_bound}x");
    let declared = dry_run().header(header::CONTENT_LENGTH, past.len());
    let (status, _, _) = answered(&app, declared.body(Body::from(past))?).await?;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "a declared length");
    let frames = [Bytes::from(at_bound), Bytes::from_static(b"x")].map(Ok::<_, Infallible>);
    let streamed = Body::from_stream(futures::stream::iter(frames));
    let (status, _, _) = answered(&app, dry_run().body(streamed)?).await?;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "a streamed body");
    Ok(())
}

/// One dry run runs and a bounded number wait: a call past them is refused
/// at once, by name.
#[tokio::test]
async fn a_dry_run_past_the_ones_that_may_wait_is_refused() -> Case {
    let state = state(picked()).await?;
    let operators = state.context.policy_operators.clone().ok_or("no door")?;
    let _running = operators.dry_runs.take().await.ok_or("a free turn")?;
    // The set ends its waiters when the case does.
    let mut waiting = JoinSet::new();
    for _ in 0..DRY_RUNS_WAITING {
        let operators = Arc::clone(&operators);
        waiting.spawn(async move { operators.dry_runs.take().await.is_some() });
    }
    // Each waiter takes its place at its first poll, ahead of this task.
    tokio::task::yield_now().await;

    let app = app(state)?;
    let (status, _, refused) = answered(&app, dry_run().body(Body::from(OTHER))?).await?;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(refused["code"], "TOO_MANY_DRY_RUNS");
    Ok(())
}
