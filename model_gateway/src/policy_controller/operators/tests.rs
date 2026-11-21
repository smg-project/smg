//! The policy document's routes through the gateway's control-plane router.

use std::{convert::Infallible, error::Error, sync::Arc};

use axum::{
    body::{to_bytes, Body, Bytes},
    http::{header, request::Builder, HeaderMap, Request, StatusCode},
    Router,
};
use serde_json::{json, Value};
use smg_auth::{ApiKeyEntry, ControlPlaneAuthConfig, ControlPlaneAuthState, Role};
use smg_policy_controller::{
    document::{operator::Turn, PolicySpec},
    engine::WireModes,
    store::{
        EncodeOffer, GatewayId, Maker, Overload, PairingDescriptor, Reachability, WorkerFacts,
        WorkersStore,
    },
};
use tower::ServiceExt;

use super::{document_bytes, DRY_RUNS_WAITING, SNAPSHOTS_WAITING, SNAPSHOT_ROWS};
use crate::{
    app_context::AppContext,
    config::RouterConfig,
    middleware::{AuthConfig, SHED_RETRY_AFTER_SECS},
    routers::gateway::Gateway,
    server::{build_app, health::ProbeState, AppState},
};

type Case = Result<(), Box<dyn Error>>;

/// The turn of one kind of call, read from the operators.
type TurnOf = fn(&super::Operators) -> &Turn;

const SPEC: &str = "/policy/spec";
const VALIDATE: &str = "/policy/spec:validate";
const SNAPSHOT: &str = "/policy/snapshot";

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
        api_keys: vec![
            ApiKeyEntry::new("admin", "Admin", "admin-key", Role::Admin),
            ApiKeyEntry::new("user", "User", "user-key", Role::User),
        ],
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

/// A snapshot of the rows under `query`, a query string or none.
fn snapshot(query: &str) -> Result<Request<Body>, http::Error> {
    by_admin(Request::get(format!("{SNAPSHOT}{query}"))).body(Body::empty())
}

/// A worker's facts under `key`, serving `models`.
fn serving(key: &str, models: &[&str]) -> WorkerFacts {
    WorkerFacts {
        key: key.to_owned(),
        engine: key.to_owned(),
        incarnation: 1,
        dp_size: 1,
        models: models.iter().map(|model| (*model).to_owned()).collect(),
        ..WorkerFacts::default()
    }
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

/// The first place where `answer` is not what `schema`, a schema of the
/// OpenAPI document, gives it: a value of another type or outside its
/// names, a key the schema does not list, or a required key left out.
fn off_schema(schema: &Value, answer: &Value, path: &str) -> Option<String> {
    let kind = match answer {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "integer",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    };
    let (kinds, names) = (&schema["type"], &schema["enum"]);
    let typed = kinds
        .as_array()
        .map_or(kinds == kind, |kinds| kinds.iter().any(|of| of == kind));
    let named = names.as_array().is_none_or(|names| names.contains(answer));
    if !(typed && named) {
        return Some(format!("{path}: {answer} is not of {schema}"));
    }
    match answer {
        Value::Array(items) => items
            .iter()
            .find_map(|item| off_schema(&schema["items"], item, &format!("{path}[]"))),
        Value::Object(fields) => {
            let required = schema["required"].as_array().into_iter().flatten();
            let left_out = required
                .filter_map(Value::as_str)
                .find(|name| !fields.contains_key(*name));
            if let Some(name) = left_out {
                return Some(format!("{path}: no {name}"));
            }
            fields.iter().find_map(|(name, field)| {
                let of = schema["properties"]
                    .get(name)
                    .unwrap_or_else(|| &schema["additionalProperties"]);
                off_schema(of, field, &format!("{path}.{name}"))
            })
        }
        _ => None,
    }
}

/// The digest of the document `config` means: the standing one after a
/// start.
fn standing(config: &RouterConfig) -> Result<String, Box<dyn Error>> {
    Ok(config.policy_document()?.digest()?.to_string())
}

#[tokio::test]
async fn under_the_classic_policies_each_route_refuses_by_one_code() -> Case {
    let state = state(RouterConfig::default()).await?;
    assert!(state.context.policy_operators.is_none());
    let app = app(state)?;
    for request in [read()?, dry_run().body(Body::from(OTHER))?, snapshot("")?] {
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

/// A document given to the start, by `--policy-spec` or by a host, is the
/// one in force: the read tells the gateway's YAML of it, under its digest.
#[tokio::test]
async fn a_given_document_is_told_as_the_one_in_force() -> Case {
    let config = RouterConfig {
        policy_spec: Some(serde_yaml::from_str(OTHER)?),
        ..picked()
    };
    let given = config.policy_document_in_force()?;
    let (status, _, in_force) = answered(&app_under(config).await?, read()?).await?;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(in_force["digest"], given.digest()?.to_string());
    let told = in_force["document"].as_str().ok_or("no document")?;
    assert_eq!(told, serde_yaml::to_string(&given)?);
    assert!(told.contains("preview"), "the given profile is told");
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
    let fed = &would_run["needs"]["after"][0]["needs"];
    assert!(fed.get("sample").is_some() && fed.get("loads").is_some());
    let document = serde_json::to_value(openapi_gen::document()?)?;
    let answer = &document["paths"][VALIDATE]["post"]["responses"]["200"]["content"]
        ["application/json"]["schema"];
    let schema =
        &answer["properties"]["needs"]["properties"]["after"]["items"]["properties"]["needs"];
    assert_eq!(off_schema(schema, fed, "needs"), None);
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
        // A form's name under another type, or under none, names no form.
        ("image/json", OTHER, StatusCode::UNSUPPORTED_MEDIA_TYPE),
        ("/yaml", OTHER, StatusCode::UNSUPPORTED_MEDIA_TYPE),
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

/// One call of a kind runs and a bounded number wait: a dry run and a
/// snapshot past them are each refused at once, by name.
#[tokio::test]
async fn a_call_past_the_ones_that_may_wait_is_refused() -> Case {
    let state = state(picked()).await?;
    let operators = state.context.policy_operators.clone().ok_or("no door")?;
    let app = app(state)?;
    let kinds: [(TurnOf, usize, _, &str); 2] = [
        (
            |operators| &operators.dry_runs,
            DRY_RUNS_WAITING,
            dry_run().body(Body::from(OTHER))?,
            "TOO_MANY_DRY_RUNS",
        ),
        (
            |operators| &operators.snapshots,
            SNAPSHOTS_WAITING,
            snapshot("")?,
            "TOO_MANY_SNAPSHOTS",
        ),
    ];
    for (turn, may_wait, request, code) in kinds {
        let _running = turn(&operators).take().await.ok_or("a free turn")?;
        // Each wait takes its place at its first poll and stays behind the
        // running call. Polled here by hand: a runtime gives a yielding
        // case no word on how many spawned waiters ran before it.
        let mut waits: Vec<_> = (0..may_wait)
            .map(|_| Box::pin(turn(&operators).take()))
            .collect();
        for wait in &mut waits {
            assert!(futures::poll!(wait.as_mut()).is_pending());
        }

        let (status, headers, refused) = answered(&app, request).await?;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(refused["code"], code);
        let again = headers
            .get(header::RETRY_AFTER)
            .and_then(|again| again.to_str().ok());
        assert_eq!(again, Some(SHED_RETRY_AFTER_SECS.to_string().as_str()));
    }
    Ok(())
}

/// A snapshot tells the store's rows in the order of their keys as
/// registered, each address without its userinfo, under the generation
/// that runs. `limit` and `models` cut the page, and a misspelt key is
/// refused, never read as no filter.
#[tokio::test]
async fn a_snapshot_tells_the_stores_rows_by_key_without_their_userinfo() -> Case {
    let state = state(picked()).await?;
    let store = state.context.policy_store.clone().ok_or("no store")?;
    // The key with a userinfo sorts last as registered, and second as told.
    store.upsert(serving("http://z.example", &["c"]));
    store.upsert(serving("http://zz:secret@b.example", &["b"]));
    store.upsert(serving("http://a.example", &["a", "c"]));
    let app = app(state)?;
    let (a, b, z) = ("http://a.example", "http://b.example", "http://z.example");
    for (query, keys, truncated) in [
        ("", &[a, z, b][..], false),
        ("?limit=1", &[a][..], true),
        ("?models=b,absent", &[b][..], false),
        ("?models=c", &[a, z][..], false),
    ] {
        let (status, _, page) = answered(&app, snapshot(query)?).await?;
        assert_eq!(status, StatusCode::OK, "{query}");
        assert_eq!(page["generation"], 1);
        assert_eq!(page["truncated"], truncated, "{query}");
        let rows = page["rows"].as_array().ok_or("no rows")?;
        let told = rows
            .iter()
            .filter_map(|row| row["registration_key"].as_str());
        assert_eq!(told.collect::<Vec<_>>(), keys, "{query}");
    }
    let (status, _, _) = answered(&app, snapshot("?model=b")?).await?;
    assert_eq!(status, StatusCode::BAD_REQUEST, "a misspelt key");
    Ok(())
}

/// A snapshot tells at most its cap of rows, whatever `limit` asks.
#[tokio::test]
async fn a_snapshot_stops_at_its_cap_of_rows() -> Case {
    let state = state(picked()).await?;
    let store = state.context.policy_store.clone().ok_or("no store")?;
    for n in 0..SNAPSHOT_ROWS + 2 {
        store.upsert(serving(&format!("http://w{n:05}.example"), &["m"]));
    }
    let (status, _, page) = answered(&app(state)?, snapshot("?limit=5000")?).await?;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(page["rows"].as_array().map(Vec::len), Some(SNAPSHOT_ROWS));
    assert_eq!(page["truncated"], true);
    Ok(())
}

/// A snapshot's ask keeps the bounds a served snapshot holds its own to: at
/// most 128 model ids, each of at most 256 bytes and with no blank, and a
/// limit of at least one. An ask outside them is refused by its key's name.
/// An answer says that it is JSON.
#[tokio::test]
async fn a_snapshots_ask_keeps_the_served_calls_bounds() -> Case {
    let app = app_under(picked()).await?;
    let most = vec!["m"; 128].join(",");
    let longest = "m".repeat(256);
    let within = [
        format!("?models={most}"),
        format!("?models={longest}"),
        "?limit=1".to_owned(),
    ];
    for query in within {
        let (status, headers, _) = answered(&app, snapshot(&query)?).await?;
        assert_eq!(status, StatusCode::OK, "{query}");
        let form = headers.get(header::CONTENT_TYPE);
        assert_eq!(
            form.and_then(|form| form.to_str().ok()),
            Some("application/json")
        );
    }
    let outside = [
        (format!("?models={most},m"), "models"),
        (format!("?models={longest}m"), "models"),
        ("?models=a%20b".to_owned(), "models"),
        ("?limit=0".to_owned(), "limit"),
    ];
    for (query, key) in outside {
        let (status, _, refused) = answered(&app, snapshot(&query)?).await?;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{query}");
        assert_eq!(refused["code"], "BAD_REQUEST");
        let said = refused["error"]
            .as_str()
            .ok_or("the refusal says nothing")?;
        assert!(said.starts_with(key), "{query}: {said}");
    }
    Ok(())
}

/// The OpenAPI document gives a snapshot's page as the route writes it,
/// for a row that states the least and for one that states every nullable
/// fact: each value of its listed type and names, each key listed, and no
/// required key left out.
#[tokio::test]
async fn a_snapshots_page_is_as_the_openapi_document_gives_it() -> Case {
    let state = state(picked()).await?;
    let store = state.context.policy_store.clone().ok_or("no store")?;
    store.upsert(serving("http://least.example", &[]));
    let stated = WorkerFacts {
        wire_modes: WireModes::from_bits(3),
        labels: [("tier".to_owned(), "a".to_owned())].into(),
        zone: Some("zone-a".to_owned()),
        pairing: Some(Arc::new(PairingDescriptor {
            protocol: Some("p1".to_owned()),
            transport: Some("t1".to_owned()),
            version: Some("v1".to_owned()),
            kv_layout: [Some("float16".to_owned()), None, None, None],
        })),
        encode: EncodeOffer {
            connector: Some("c1".to_owned()),
            ..EncodeOffer::default()
        },
        max_running: Some(8),
        kv_block_size: Some(16),
        bootstrap_port: Some(9000),
        reachable_from: Reachability::Gateway(GatewayId(0x1f)),
        overload: Overload {
            kv_usage_permille: Some(900),
            waiting_requests: Some(64),
        },
        ..serving("http://stated.example", &["m"])
    };
    store.upsert_by(stated, Maker::provider("finder", "http://host/x")?);
    let (status, _, page) = answered(&app(state)?, snapshot("")?).await?;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(page["rows"].as_array().map(Vec::len), Some(2));
    let document = serde_json::to_value(openapi_gen::document()?)?;
    let schema = &document["paths"][SNAPSHOT]["get"]["responses"]["200"]["content"]
        ["application/json"]["schema"];
    assert_eq!(off_schema(schema, &page, "page"), None);
    Ok(())
}

/// The document and the rows name a fleet: the `user` role reads neither.
#[tokio::test]
async fn the_user_role_reads_neither_the_document_nor_the_rows() -> Case {
    let app = app_under(picked()).await?;
    for path in [SPEC, SNAPSHOT] {
        let by_user = Request::get(path).header(header::AUTHORIZATION, "Bearer user-key");
        let (status, _, _) = answered(&app, by_user.body(Body::empty())?).await?;
        assert_eq!(status, StatusCode::FORBIDDEN, "{path}");
    }
    Ok(())
}
