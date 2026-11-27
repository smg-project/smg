use super::*;

#[tokio::test]
async fn failed_jwt_initialization_retains_required_authentication() {
    use crate::config::{ApiKeyEntry, JwtConfig};

    for with_api_key in [false, true] {
        let config = ControlPlaneAuthConfig {
            jwt: Some(
                JwtConfig::new("https://issuer.example.com", "audience")
                    .with_jwks_uri("https://127.0.0.1/jwks"),
            ),
            api_keys: if with_api_key {
                vec![ApiKeyEntry::new(
                    "admin",
                    "Admin",
                    "control-plane-key",
                    Role::Admin,
                )]
            } else {
                vec![]
            },
            audit_enabled: false,
        };
        let state = ControlPlaneAuthState::try_init(Some(&config)).await;
        assert!(
            state.is_some(),
            "failed JWT setup must retain the configured auth boundary"
        );
        let state = state.unwrap();
        assert!(state.is_auth_required());
        assert!(state.jwt_validator.is_none());
        assert_eq!(state.config.has_api_keys(), with_api_key);
    }
}

#[tokio::test]
async fn optional_auth_initialization_preserves_disabled_and_api_key_only_modes() {
    use crate::config::ApiKeyEntry;

    assert!(ControlPlaneAuthState::try_init(None).await.is_none());
    assert!(
        ControlPlaneAuthState::try_init(Some(&ControlPlaneAuthConfig::default()))
            .await
            .is_none()
    );
    let config = ControlPlaneAuthConfig {
        jwt: None,
        api_keys: vec![ApiKeyEntry::new(
            "admin",
            "Admin",
            "control-plane-key",
            Role::Admin,
        )],
        audit_enabled: false,
    };
    let state = ControlPlaneAuthState::try_init(Some(&config))
        .await
        .unwrap();
    assert!(state.is_auth_required());
    assert!(state.config.find_api_key("control-plane-key").is_some());
}

#[test]
fn test_auth_method_display() {
    let jwt = AuthMethod::Jwt {
        issuer: "https://example.com".to_string(),
    };
    assert_eq!(jwt.to_string(), "jwt:https://example.com");

    let api_key = AuthMethod::ApiKey {
        key_id: "key-123".to_string(),
    };
    assert_eq!(api_key.to_string(), "api_key:key-123");
}

#[test]
fn bearer_token_matches_the_scheme_case_insensitively() {
    assert_eq!(bearer_token("Bearer k1"), Some("k1"));
    assert_eq!(bearer_token("bearer k1"), Some("k1"));
    assert_eq!(bearer_token("BEARER k1"), Some("k1"));
    assert_eq!(bearer_token("Bearer   k1"), Some("k1"));
}

#[test]
fn bearer_token_rejects_other_schemes_and_empty_tokens() {
    assert_eq!(bearer_token("Basic k1"), None);
    assert_eq!(bearer_token("Bearerk1"), None);
    assert_eq!(bearer_token("Bearer"), None);
    assert_eq!(bearer_token("Bearer "), None);
    assert_eq!(bearer_token("k1"), None);
}

#[test]
fn admin_may_use_every_method() {
    for method in ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"] {
        assert!(role_allows(Role::Admin, method, false), "{method}");
        assert!(
            role_allows(Role::Admin, method, true),
            "{method}, admin-only"
        );
    }
}

#[test]
fn user_may_only_read_and_not_an_admin_only_read() {
    assert!(role_allows(Role::User, "GET", false));
    assert!(role_allows(Role::User, "HEAD", false));
    assert!(!role_allows(Role::User, "GET", true), "an admin-only read");
    assert!(!role_allows(Role::User, "HEAD", true), "an admin-only read");
    for method in ["POST", "PUT", "PATCH", "DELETE"] {
        assert!(!role_allows(Role::User, method, false), "{method}");
    }
}

#[test]
fn test_control_plane_auth_state_no_config() {
    let config = ControlPlaneAuthConfig::default();
    let state = ControlPlaneAuthState::new(config, None);
    assert!(!state.is_auth_required());
}

#[test]
fn test_control_plane_auth_state_with_api_keys() {
    use crate::config::ApiKeyEntry;

    let config = ControlPlaneAuthConfig {
        jwt: None,
        api_keys: vec![ApiKeyEntry::new("test", "Test Key", "secret", Role::Admin)],
        audit_enabled: true,
    };
    let state = ControlPlaneAuthState::new(config, None);
    assert!(state.is_auth_required());
}

/// The handler the middleware admits to answers with the principal it
/// finds in the request, so the test sees what reached it.
async fn whoami(request: Request<Body>) -> String {
    request
        .principal()
        .map(|p| format!("{} via {} as {:?}", p.id, p.auth_method, p.role))
        .unwrap_or_else(|| "nobody".to_string())
}

async fn call(
    app: &axum::Router,
    method: &str,
    key: Option<&str>,
) -> (StatusCode, Option<String>, String) {
    use tower::ServiceExt;

    let mut request = Request::builder().method(method).uri("/thing");
    if let Some(key) = key {
        request = request.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    let response = app
        .clone()
        .oneshot(request.body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let challenge = response
        .headers()
        .get("www-authenticate")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let body = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();
    (status, challenge, String::from_utf8(body.to_vec()).unwrap())
}

/// The API-key credential through the one admission path: a missing or
/// unknown token is refused with the realm's challenge, a role short of
/// the method is forbidden, and an admitted caller reaches the handler as
/// the principal its key names.
#[tokio::test]
async fn an_api_key_caller_is_refused_forbidden_or_admitted_through_one_path() {
    use axum::{middleware::from_fn_with_state, routing::get, Router};

    use crate::config::ApiKeyEntry;

    let state = ControlPlaneAuthState::new(
        ControlPlaneAuthConfig {
            jwt: None,
            api_keys: vec![ApiKeyEntry::new(
                "reader",
                "Reader",
                "reader-key",
                Role::User,
            )],
            audit_enabled: false,
        },
        None,
    );
    let app = Router::new()
        .route("/thing", get(whoami).post(whoami))
        .layer(from_fn_with_state(state, control_plane_auth_middleware));
    let challenge = Some("Bearer realm=\"control-plane\"".to_string());

    assert_eq!(
        call(&app, "GET", None).await,
        (
            StatusCode::UNAUTHORIZED,
            challenge.clone(),
            "Missing or invalid Authorization header".to_string()
        )
    );
    assert_eq!(
        call(&app, "GET", Some("not-a-key")).await,
        (
            StatusCode::UNAUTHORIZED,
            challenge,
            "Invalid authentication token".to_string()
        )
    );
    let (status, _, body) = call(&app, "POST", Some("reader-key")).await;
    assert_eq!(
        (status, body),
        (
            StatusCode::FORBIDDEN,
            "Admin role required for this control plane operation".to_string()
        )
    );
    let (status, _, body) = call(&app, "GET", Some("reader-key")).await;
    assert_eq!(
        (status, body),
        (
            StatusCode::OK,
            "reader via api_key:reader as User".to_string()
        )
    );
}

/// The JWT credential through the same path: a signed token is admitted
/// as the principal its claims name, and a token that is a JWT in shape
/// but does not verify is refused at once with the challenge and the
/// reason, with no fall-back to the API keys.
#[tokio::test]
async fn a_jwt_caller_is_admitted_or_refused_through_the_same_path() {
    use axum::{middleware::from_fn_with_state, routing::get, Router};

    use crate::{
        config::ApiKeyEntry,
        jwt::test_support::{token, validator, ISSUER},
    };

    let validator = validator().await;
    let state = ControlPlaneAuthState::new(
        ControlPlaneAuthConfig {
            jwt: Some(validator.config().clone()),
            api_keys: vec![ApiKeyEntry::new("ops", "Ops", "ops-key", Role::Admin)],
            audit_enabled: false,
        },
        Some(Arc::new(validator)),
    );
    let app = Router::new()
        .route("/thing", get(whoami))
        .layer(from_fn_with_state(state, control_plane_auth_middleware));

    let signed = token(None);
    let (status, _, body) = call(&app, "GET", Some(&signed)).await;
    assert_eq!(
        (status, body),
        (StatusCode::OK, format!("alice via jwt:{ISSUER} as Admin"))
    );

    let mut tampered = signed.clone();
    let last = tampered.pop().expect("a signed token has a signature");
    tampered.push(if last == 'A' { 'B' } else { 'A' });
    let (status, challenge, body) = call(&app, "GET", Some(&tampered)).await;
    assert_eq!(
        (status, challenge),
        (
            StatusCode::UNAUTHORIZED,
            Some("Bearer realm=\"control-plane\"".to_string())
        )
    );
    assert!(body.starts_with("Invalid JWT: "), "{body}");

    // A token that is not a JWT in shape still reaches the API keys.
    let (status, _, body) = call(&app, "GET", Some("ops-key")).await;
    assert_eq!(
        (status, body),
        (StatusCode::OK, "ops via api_key:ops as Admin".to_string())
    );
}

/// The audit lines one call through the middleware writes, with its
/// status: `method` on `/workers` with `key`, carrying the request id the
/// gateway's edge layer would have set.
async fn audited_call(
    audit_enabled: bool,
    method: &str,
    key: &str,
) -> (StatusCode, Vec<Vec<(String, String)>>) {
    use axum::{middleware::from_fn_with_state, routing::post, Router};
    use tower::ServiceExt;
    use tracing::instrument::WithSubscriber;

    use crate::{audit::tests::FieldCapture, config::ApiKeyEntry};

    async fn register() -> Response {
        AuditResource::name(StatusCode::ACCEPTED.into_response(), "worker-1")
    }
    async fn refuse() -> Response {
        StatusCode::BAD_REQUEST.into_response()
    }
    async fn list() -> Response {
        StatusCode::OK.into_response()
    }
    async fn give_up() -> Response {
        panic!("the handler gave up")
    }
    let state = ControlPlaneAuthState::new(
        ControlPlaneAuthConfig {
            jwt: None,
            api_keys: vec![
                ApiKeyEntry::new("admin", "Admin", "admin-key", Role::Admin),
                ApiKeyEntry::new("reader", "Reader", "reader-key", Role::User),
            ],
            audit_enabled,
        },
        None,
    );
    let app = Router::new()
        .route("/workers", post(register).get(list))
        .route("/refused", post(refuse))
        .route("/broken", post(give_up))
        .layer(from_fn_with_state(state, control_plane_auth_middleware));
    let (path, method) = match method {
        "REFUSED" => ("/refused", "POST"),
        "BROKEN" => ("/broken", "POST"),
        _ => ("/workers", method),
    };
    let mut request = Request::builder()
        .method(method)
        .uri(path)
        .header(header::AUTHORIZATION, format!("Bearer {key}"))
        .body(Body::empty())
        .unwrap();
    request
        .extensions_mut()
        .insert(RequestId("req-7".to_string()));

    let capture = Arc::new(FieldCapture::default());
    let response = app
        .oneshot(request)
        .with_subscriber(capture.clone())
        .await
        .unwrap();
    let events = capture.events.lock().unwrap().clone();
    (response.status(), events)
}

#[tokio::test]
async fn a_mutation_writes_an_outcome_line_after_the_admission_line() {
    use crate::audit::tests::field;

    let (status, events) = audited_call(true, "POST", "admin-key").await;

    assert_eq!(status, StatusCode::ACCEPTED);
    assert_eq!(
        events.len(),
        2,
        "the admission line, then the outcome line: {events:?}"
    );
    let (admission, outcome) = (&events[0], &events[1]);
    assert_eq!(
        field(admission, "message"),
        Some("control_plane_audit"),
        "{admission:?}"
    );
    assert_eq!(
        field(admission, "status"),
        None,
        "the admission line is written before the handler: {admission:?}"
    );
    assert_eq!(
        field(outcome, "message"),
        Some("control_plane_mutation"),
        "{outcome:?}"
    );
    assert_eq!(field(outcome, "status"), Some("202"), "{outcome:?}");
    assert_eq!(field(outcome, "outcome"), Some("success"), "{outcome:?}");
    assert_eq!(field(outcome, "resource"), Some("worker-1"), "{outcome:?}");
    for line in [admission, outcome] {
        assert_eq!(field(line, "principal"), Some("admin"), "{line:?}");
        assert_eq!(field(line, "auth_method"), Some("api_key"), "{line:?}");
        assert_eq!(field(line, "role"), Some("admin"), "{line:?}");
        assert_eq!(field(line, "method"), Some("POST"), "{line:?}");
        assert_eq!(field(line, "path"), Some("/workers"), "{line:?}");
        assert_eq!(
            field(line, "request_id"),
            Some("req-7"),
            "one request id correlates both lines: {line:?}"
        );
    }
}

#[tokio::test]
async fn a_mutation_the_handler_refused_is_logged_as_failed_with_its_status() {
    use crate::audit::tests::field;

    let (status, events) = audited_call(true, "REFUSED", "admin-key").await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(events.len(), 2, "{events:?}");
    let outcome = &events[1];
    assert_eq!(field(outcome, "status"), Some("400"), "{outcome:?}");
    assert_eq!(field(outcome, "outcome"), Some("failed"), "{outcome:?}");
    assert_eq!(
        field(outcome, "resource"),
        None,
        "a refused mutation names no resource: {outcome:?}"
    );
}

#[tokio::test]
async fn a_read_writes_the_admission_line_alone() {
    use crate::audit::tests::field;

    let (status, events) = audited_call(true, "GET", "reader-key").await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(
        field(&events[0], "message"),
        Some("control_plane_audit"),
        "{events:?}"
    );
}

/// The audit lines of a mutation whose client leaves after the first
/// poll: `handler` waits at a gate, the middleware's future is polled
/// once and dropped, the gate opens, and the lines are read once the
/// task has written its outcome (or after a thousand yields).
async fn lines_after_the_client_left(
    handler: axum::routing::MethodRouter<Arc<tokio::sync::Notify>>,
) -> Vec<Vec<(String, String)>> {
    use std::{
        future::Future,
        task::{Context, Waker},
    };

    use axum::{middleware::from_fn_with_state, Router};
    use tokio::sync::Notify;
    use tower::ServiceExt;
    use tracing::instrument::WithSubscriber;

    use crate::{audit::tests::FieldCapture, config::ApiKeyEntry};

    let gate = Arc::new(Notify::new());
    let state = ControlPlaneAuthState::new(
        ControlPlaneAuthConfig {
            jwt: None,
            api_keys: vec![ApiKeyEntry::new("admin", "Admin", "admin-key", Role::Admin)],
            audit_enabled: true,
        },
        None,
    );
    let app = Router::new()
        .route("/workers", handler)
        .layer(from_fn_with_state(state, control_plane_auth_middleware))
        .with_state(Arc::clone(&gate));
    let mut request = Request::builder()
        .method("POST")
        .uri("/workers")
        .header(header::AUTHORIZATION, "Bearer admin-key")
        .body(Body::empty())
        .unwrap();
    request
        .extensions_mut()
        .insert(RequestId("req-7".to_string()));
    let capture = Arc::new(FieldCapture::default());

    let mut call = Box::pin(app.oneshot(request).with_subscriber(capture.clone()));
    let mut cx = Context::from_waker(Waker::noop());
    assert!(
        call.as_mut().poll(&mut cx).is_pending(),
        "the handler waits at its gate"
    );
    drop(call);
    gate.notify_one();
    for _ in 0..1_000 {
        if capture.events.lock().unwrap().len() == 2 {
            break;
        }
        tokio::task::yield_now().await;
    }
    let events = capture.events.lock().unwrap().clone();
    events
}

/// The client of a mutation leaves after the first poll: the handler's
/// gate opens, and the outcome line still reaches the audit log under the
/// request's subscriber.
#[tokio::test]
async fn a_mutation_whose_client_leaves_still_writes_its_outcome_line() {
    use axum::routing::post;
    use tokio::sync::Notify;

    use crate::audit::tests::field;

    async fn register(State(gate): State<Arc<Notify>>) -> Response {
        gate.notified().await;
        AuditResource::name(StatusCode::ACCEPTED.into_response(), "worker-1")
    }
    let events = lines_after_the_client_left(post(register)).await;

    assert_eq!(
        events.len(),
        2,
        "the admission line, then the outcome line: {events:?}"
    );
    let outcome = &events[1];
    assert_eq!(
        field(outcome, "message"),
        Some("control_plane_mutation"),
        "{outcome:?}"
    );
    assert_eq!(field(outcome, "status"), Some("202"), "{outcome:?}");
    assert_eq!(field(outcome, "outcome"), Some("success"), "{outcome:?}");
    assert_eq!(field(outcome, "resource"), Some("worker-1"), "{outcome:?}");
    assert_eq!(field(outcome, "request_id"), Some("req-7"), "{outcome:?}");
}

/// The client of a mutation leaves after the first poll: the record keeps
/// the hold the state hands out until its outcome line is written and
/// releases it after the line, whether the handler answers or panics,
/// so a shutdown drain that waits on the hold waits for the line.
#[tokio::test]
async fn a_departed_clients_mutation_keeps_its_hold_until_the_outcome_line() {
    use std::{
        future::Future,
        sync::atomic::{AtomicUsize, Ordering},
        task::{Context, Waker},
    };

    use axum::{middleware::from_fn_with_state, routing::post, Router};
    use tokio::sync::Notify;
    use tower::ServiceExt;
    use tracing::instrument::WithSubscriber;

    use crate::{
        audit::tests::{field, FieldCapture},
        config::ApiKeyEntry,
    };

    /// A hold that counts itself while it lives and notes, when it goes,
    /// how many audit lines the capture held by then.
    struct Held {
        count: Arc<AtomicUsize>,
        lines: Arc<FieldCapture>,
        lines_at_release: Arc<AtomicUsize>,
    }
    impl Drop for Held {
        fn drop(&mut self) {
            let lines = self.lines.events.lock().unwrap().len();
            self.lines_at_release.store(lines, Ordering::Release);
            self.count.fetch_sub(1, Ordering::AcqRel);
        }
    }

    async fn register(State(gate): State<Arc<Notify>>) -> Response {
        gate.notified().await;
        AuditResource::name(StatusCode::ACCEPTED.into_response(), "worker-1")
    }
    async fn give_up(State(gate): State<Arc<Notify>>) -> Response {
        gate.notified().await;
        panic!("the handler gave up after the client left")
    }

    for (handler, status) in [(post(register), "202"), (post(give_up), "500")] {
        let held = Arc::new(AtomicUsize::new(0));
        let lines_at_release = Arc::new(AtomicUsize::new(usize::MAX));
        let capture = Arc::new(FieldCapture::default());
        let gate = Arc::new(Notify::new());
        let state = ControlPlaneAuthState::new(
            ControlPlaneAuthConfig {
                jwt: None,
                api_keys: vec![ApiKeyEntry::new("admin", "Admin", "admin-key", Role::Admin)],
                audit_enabled: true,
            },
            None,
        )
        .with_mutation_hold({
            let (count, lines, lines_at_release) = (
                Arc::clone(&held),
                Arc::clone(&capture),
                Arc::clone(&lines_at_release),
            );
            move || {
                count.fetch_add(1, Ordering::AcqRel);
                Box::new(Held {
                    count: Arc::clone(&count),
                    lines: Arc::clone(&lines),
                    lines_at_release: Arc::clone(&lines_at_release),
                })
            }
        });
        let app = Router::new()
            .route("/workers", handler)
            .layer(from_fn_with_state(state, control_plane_auth_middleware))
            .with_state(Arc::clone(&gate));
        let request = Request::builder()
            .method("POST")
            .uri("/workers")
            .header(header::AUTHORIZATION, "Bearer admin-key")
            .body(Body::empty())
            .unwrap();

        let mut call = Box::pin(app.oneshot(request).with_subscriber(capture.clone()));
        let mut cx = Context::from_waker(Waker::noop());
        assert!(
            call.as_mut().poll(&mut cx).is_pending(),
            "the handler waits at its gate"
        );
        assert_eq!(held.load(Ordering::Acquire), 1, "the record took its hold");
        // The client leaves; the task runs on to its gate with the hold.
        drop(call);
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            held.load(Ordering::Acquire),
            1,
            "the departed client's record keeps its hold"
        );
        gate.notify_one();
        for _ in 0..1_000 {
            if held.load(Ordering::Acquire) == 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(
            held.load(Ordering::Acquire),
            0,
            "the hold goes with the record"
        );
        let events = capture.events.lock().unwrap().clone();
        assert_eq!(
            events.len(),
            2,
            "the admission line, then the outcome line: {events:?}"
        );
        assert_eq!(field(&events[1], "status"), Some(status), "{events:?}");
        assert_eq!(
            lines_at_release.load(Ordering::Acquire),
            2,
            "the outcome line was written before the hold went ({status})"
        );
    }
}

/// The client leaves, then the handler panics: the task's record still
/// writes the outcome line, at 500 with no resource.
#[tokio::test]
async fn a_handler_that_panics_after_the_client_left_still_writes_its_outcome_line() {
    use axum::routing::post;
    use tokio::sync::Notify;

    use crate::audit::tests::field;

    async fn give_up(State(gate): State<Arc<Notify>>) -> Response {
        gate.notified().await;
        panic!("the handler gave up after the client left")
    }
    let events = lines_after_the_client_left(post(give_up)).await;

    assert_eq!(
        events.len(),
        2,
        "the admission line, then the outcome line: {events:?}"
    );
    let outcome = &events[1];
    assert_eq!(field(outcome, "status"), Some("500"), "{outcome:?}");
    assert_eq!(field(outcome, "outcome"), Some("failed"), "{outcome:?}");
    assert_eq!(field(outcome, "resource"), None, "{outcome:?}");
}

/// The runtime drops the task of a mutation in flight (the process stops
/// with the handler at work, no panic): the record writes no outcome
/// line. No client was answered a 500, so the audit trail keeps the
/// admission line alone.
#[test]
fn a_task_the_runtime_drops_writes_no_outcome_line() {
    use std::{
        future::Future,
        task::{Context, Waker},
    };

    use axum::{middleware::from_fn_with_state, routing::post, Router};
    use tokio::sync::Notify;
    use tower::ServiceExt;
    use tracing::instrument::WithSubscriber;

    use crate::{
        audit::tests::{field, FieldCapture},
        config::ApiKeyEntry,
    };

    async fn wait_at_the_gate(State(gate): State<Arc<Notify>>) -> Response {
        gate.notified().await;
        StatusCode::ACCEPTED.into_response()
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let capture = Arc::new(FieldCapture::default());
    let gate = Arc::new(Notify::new());
    runtime.block_on(async {
        let state = ControlPlaneAuthState::new(
            ControlPlaneAuthConfig {
                jwt: None,
                api_keys: vec![ApiKeyEntry::new("admin", "Admin", "admin-key", Role::Admin)],
                audit_enabled: true,
            },
            None,
        );
        let app = Router::new()
            .route("/workers", post(wait_at_the_gate))
            .layer(from_fn_with_state(state, control_plane_auth_middleware))
            .with_state(Arc::clone(&gate));
        let request = Request::builder()
            .method("POST")
            .uri("/workers")
            .header(header::AUTHORIZATION, "Bearer admin-key")
            .body(Body::empty())
            .unwrap();
        let mut call = Box::pin(app.oneshot(request).with_subscriber(capture.clone()));
        let mut cx = Context::from_waker(Waker::noop());
        assert!(
            call.as_mut().poll(&mut cx).is_pending(),
            "the handler waits at its gate"
        );
        // The client leaves; the task runs on to its gate.
        drop(call);
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
    });
    // The runtime goes with the task at its gate: the task's future is
    // dropped on this thread, under the capture, with no panic.
    tracing::subscriber::with_default(capture.clone(), || drop(runtime));

    let events = capture.events.lock().unwrap().clone();
    assert_eq!(events.len(), 1, "the admission line alone: {events:?}");
    assert_eq!(
        field(&events[0], "message"),
        Some("control_plane_audit"),
        "{events:?}"
    );
}

/// A handler that ends without an answer: the request answers 500 and
/// the outcome line says so, with no resource.
#[tokio::test]
async fn a_handler_that_ends_without_an_answer_is_recorded_as_a_500() {
    use crate::audit::tests::field;

    let (status, events) = audited_call(true, "BROKEN", "admin-key").await;

    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(events.len(), 2, "{events:?}");
    let outcome = &events[1];
    assert_eq!(field(outcome, "status"), Some("500"), "{outcome:?}");
    assert_eq!(field(outcome, "outcome"), Some("failed"), "{outcome:?}");
    assert_eq!(field(outcome, "resource"), None, "{outcome:?}");
}

/// A read the gateway named admin-only: the `user` role is refused by
/// the gate with the denial line and no admission line, as for a
/// mutation; the `admin` role reads it with the admission line alone.
#[tokio::test]
async fn an_admin_only_read_refuses_the_user_role_with_the_denial_line() {
    use axum::{middleware::from_fn_with_state, routing::get, Router};
    use tower::ServiceExt;
    use tracing::instrument::WithSubscriber;

    use crate::{
        audit::tests::{field, FieldCapture},
        config::ApiKeyEntry,
    };

    async fn view() -> Response {
        StatusCode::OK.into_response()
    }
    let state = ControlPlaneAuthState::new(
        ControlPlaneAuthConfig {
            jwt: None,
            api_keys: vec![
                ApiKeyEntry::new("admin", "Admin", "admin-key", Role::Admin),
                ApiKeyEntry::new("reader", "Reader", "reader-key", Role::User),
            ],
            audit_enabled: true,
        },
        None,
    )
    .with_admin_only_reads(&["/config"]);
    let app = Router::new()
        .route("/config", get(view))
        .route("/workers", get(view))
        .layer(from_fn_with_state(state, control_plane_auth_middleware));

    let call = |path: &'static str, key: &'static str| {
        let app = app.clone();
        async move {
            let request = Request::builder()
                .uri(path)
                .header(header::AUTHORIZATION, format!("Bearer {key}"))
                .body(Body::empty())
                .unwrap();
            let capture = Arc::new(FieldCapture::default());
            let response = app
                .oneshot(request)
                .with_subscriber(capture.clone())
                .await
                .unwrap();
            let events = capture.events.lock().unwrap().clone();
            (response.status(), events)
        }
    };

    let (status, events) = call("/config", "reader-key").await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(field(&events[0], "outcome"), Some("denied"), "{events:?}");

    let (status, events) = call("/config", "admin-key").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(field(&events[0], "outcome"), Some("success"), "{events:?}");

    let (status, _) = call("/workers", "reader-key").await;
    assert_eq!(status, StatusCode::OK, "a plain read stays the user's");
}

/// An admin-only read is named by the template of its route, so a route
/// with a parameter can be one: the `user` role is refused on every path
/// the template matches and the `admin` role reads it. An entry without a
/// parameter is its own template, and a path no route matched is asked by
/// the path, both as before.
#[tokio::test]
async fn an_admin_only_read_is_known_by_its_route_template() {
    use axum::{middleware::from_fn_with_state, routing::get, Router};
    use tower::ServiceExt;

    use crate::config::ApiKeyEntry;

    async fn view() -> Response {
        StatusCode::OK.into_response()
    }
    let state = ControlPlaneAuthState::new(
        ControlPlaneAuthConfig {
            jwt: None,
            api_keys: vec![
                ApiKeyEntry::new("admin", "Admin", "admin-key", Role::Admin),
                ApiKeyEntry::new("reader", "Reader", "reader-key", Role::User),
            ],
            audit_enabled: false,
        },
        None,
    )
    .with_admin_only_reads(&["/config", "/workers/{worker_id}/server_info", "/unrouted"]);
    let app = Router::new()
        .route("/config", get(view))
        .route("/workers", get(view))
        .route("/workers/{worker_id}", get(view))
        .route("/workers/{worker_id}/server_info", get(view))
        .layer(from_fn_with_state(state, control_plane_auth_middleware));

    let status = |path: &'static str, key: &'static str| {
        let app = app.clone();
        async move {
            let request = Request::builder()
                .uri(path)
                .header(header::AUTHORIZATION, format!("Bearer {key}"))
                .body(Body::empty())
                .unwrap();
            app.oneshot(request).await.unwrap().status()
        }
    };

    for path in ["/workers/w-1/server_info", "/workers/another/server_info"] {
        assert_eq!(
            status(path, "reader-key").await,
            StatusCode::FORBIDDEN,
            "{path}"
        );
        assert_eq!(status(path, "admin-key").await, StatusCode::OK, "{path}");
    }

    assert_eq!(status("/config", "reader-key").await, StatusCode::FORBIDDEN);
    assert_eq!(status("/config", "admin-key").await, StatusCode::OK);

    for path in ["/workers", "/workers/w-1"] {
        assert_eq!(status(path, "reader-key").await, StatusCode::OK, "{path}");
    }

    assert_eq!(
        status("/unrouted", "reader-key").await,
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        status("/unrouted", "admin-key").await,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        status("/elsewhere", "reader-key").await,
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn a_mutation_the_role_may_not_make_writes_the_denial_line_alone() {
    use crate::audit::tests::field;

    let (status, events) = audited_call(true, "POST", "reader-key").await;

    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(field(&events[0], "outcome"), Some("denied"), "{events:?}");
    assert_eq!(
        field(&events[0], "status"),
        None,
        "no handler answered: {events:?}"
    );
}

#[tokio::test]
async fn a_disabled_audit_logger_writes_neither_line() {
    let (status, events) = audited_call(false, "POST", "admin-key").await;

    assert_eq!(status, StatusCode::ACCEPTED);
    assert!(events.is_empty(), "{events:?}");
}
