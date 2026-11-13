//! `GET` and `PUT /rate-limits/tenants` on the production route table: the
//! tenant rate-limit document is read with its revision and ETag by the
//! `admin` role alone, replaced only against the ETag read and checked as
//! the start checks its file, and a dry run applies nothing.

use axum::{
    body::Body,
    extract::Request,
    http::{
        header::{AUTHORIZATION, CONTENT_TYPE, ETAG, IF_MATCH},
        HeaderName, StatusCode,
    },
    response::Response,
    Router,
};
use serde_json::{json, Value};
use smg::config::RouterConfig;
use smg_auth::{ApiKeyEntry, AuditResource, ControlPlaneAuthConfig, ControlPlaneAuthState, Role};
use tower::ServiceExt;

use crate::common::{
    test_app::create_test_app_with_context_and_auth, AppTestContext, TestRouterConfig,
    TestWorkerConfig,
};

const ADMIN_KEY: &str = "rate-limit-admin-key";
const USER_KEY: &str = "rate-limit-user-key";
const PATH: &str = "/rate-limits/tenants";
const REVISION: HeaderName = HeaderName::from_static("x-smg-config-revision");

fn control_plane_auth() -> ControlPlaneAuthState {
    ControlPlaneAuthState::new(
        ControlPlaneAuthConfig {
            jwt: None,
            api_keys: vec![
                ApiKeyEntry::new("admin", "Admin", ADMIN_KEY, Role::Admin),
                ApiKeyEntry::new("reader", "Reader", USER_KEY, Role::User),
            ],
            audit_enabled: false,
        },
        None,
    )
}

/// A gateway with a tenant rate limiter from a file in `dir`: the default
/// policy and one tenant.
#[expect(
    clippy::unwrap_used,
    reason = "test helper: the policy file is written by the test"
)]
fn limited_config(port: u16, dir: &std::path::Path) -> RouterConfig {
    let path = dir.join("rate_limit.yaml");
    std::fs::write(
        &path,
        "default_policy:\n  tokens_per_minute: 1000\n  requests_per_minute: 60\ntenants:\n  - tenant_key: auth:team-red\n    tokens_per_minute: 500\n    requests_per_minute: 30\n",
    )
    .unwrap();
    let mut config = TestRouterConfig::round_robin(port);
    config.tenant_rate_limit_enabled = true;
    config.tenant_rate_limit_config = Some(path.to_string_lossy().into_owned());
    config
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper: a malformed request is a test bug"
)]
async fn call(
    app: &Router,
    method: &str,
    path: &str,
    key: Option<&str>,
    if_match: Option<&str>,
    body: Option<Value>,
) -> Response {
    let mut request = Request::builder()
        .method(method)
        .uri(path)
        .header(CONTENT_TYPE, "application/json");
    if let Some(key) = key {
        request = request.header(AUTHORIZATION, format!("Bearer {key}"));
    }
    if let Some(etag) = if_match {
        request = request.header(IF_MATCH, etag);
    }
    let body = match body {
        Some(body) => Body::from(body.to_string()),
        None => Body::empty(),
    };
    app.clone()
        .oneshot(request.body(body).unwrap())
        .await
        .unwrap()
}

#[expect(clippy::unwrap_used, reason = "test helper: the answer is JSON")]
async fn json(response: Response) -> Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn header(response: &Response, name: &HeaderName) -> Option<String> {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
}

/// A document over the start's default policy: team-red at `red_tokens`,
/// and team-blue beside it when asked.
fn document(red_tokens: u32, blue: bool) -> Value {
    let mut tenants = vec![json!({
        "tenant_key": "auth:team-red",
        "tokens_per_minute": red_tokens,
        "requests_per_minute": 30
    })];
    if blue {
        tenants.push(json!({
            "tenant_key": "auth:team-blue",
            "tokens_per_minute": 100,
            "requests_per_minute": 10
        }));
    }
    json!({
        "default_policy": {"tokens_per_minute": 1000, "requests_per_minute": 60},
        "tenants": tenants
    })
}

#[tokio::test]
async fn the_document_is_read_and_replaced_against_its_etag() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let ctx = AppTestContext::new_with_config(
        limited_config(4356, dir.path()),
        vec![TestWorkerConfig::healthy(20356)],
    )
    .await;
    let app = create_test_app_with_context_and_auth(
        ctx.router.clone(),
        ctx.app_context.clone(),
        Some(control_plane_auth()),
    );

    // The read: the start's file as revision 1, the ETag in the body and
    // the headers, the admin role alone.
    assert_eq!(
        call(&app, "GET", PATH, Some(USER_KEY), None, None)
            .await
            .status(),
        StatusCode::FORBIDDEN,
        "the user role may not read the document"
    );
    let read = call(&app, "GET", PATH, Some(ADMIN_KEY), None, None).await;
    assert_eq!(read.status(), StatusCode::OK);
    let etag = header(&read, &ETAG).expect("the ETag header");
    assert_eq!(header(&read, &REVISION), Some("1".to_string()));
    let live = json(read).await;
    assert_eq!(live["revision"], 1, "{live}");
    assert_eq!(live["etag"], etag, "{live}");
    assert_eq!(live["principal"], "startup", "{live}");
    assert_eq!(
        live["document"]["tenants"][0]["tenant_key"], "auth:team-red",
        "{live}"
    );

    // A write without If-Match, or with `*`, is asked for the ETag.
    for if_match in [None, Some("*")] {
        let refused = call(
            &app,
            "PUT",
            PATH,
            Some(ADMIN_KEY),
            if_match,
            Some(document(600, false)),
        )
        .await;
        assert_eq!(
            refused.status(),
            StatusCode::PRECONDITION_REQUIRED,
            "If-Match {if_match:?}"
        );
        assert_eq!(header(&refused, &REVISION), Some("1".to_string()));
        assert_eq!(json(refused).await["code"], "PRECONDITION_REQUIRED");
    }

    // A write against another revision is refused with the live one.
    let stale = call(
        &app,
        "PUT",
        PATH,
        Some(ADMIN_KEY),
        Some("\"stale\""),
        Some(document(600, false)),
    )
    .await;
    assert_eq!(stale.status(), StatusCode::PRECONDITION_FAILED);
    assert_eq!(header(&stale, &ETAG), Some(etag.clone()));
    assert_eq!(header(&stale, &REVISION), Some("1".to_string()));
    assert_eq!(json(stale).await["code"], "PRECONDITION_FAILED");

    // A document the start would refuse: 400, and nothing moves.
    let twice = json!({
        "default_policy": {"tokens_per_minute": 1000, "requests_per_minute": 60},
        "tenants": [
            {"tenant_key": "auth:team-red", "tokens_per_minute": 1, "requests_per_minute": 1},
            {"tenant_key": "auth:team-red", "tokens_per_minute": 2, "requests_per_minute": 2}
        ]
    });
    let invalid = call(&app, "PUT", PATH, Some(ADMIN_KEY), Some(&etag), Some(twice)).await;
    assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
    let invalid = json(invalid).await;
    assert_eq!(invalid["code"], "INVALID_DOCUMENT", "{invalid}");
    assert!(
        invalid["error"]
            .as_str()
            .is_some_and(|error| error.contains("duplicate tenant_key")),
        "{invalid}"
    );
    // A body that is not the document: the control plane's 400 envelope.
    let not_a_document = call(
        &app,
        "PUT",
        PATH,
        Some(ADMIN_KEY),
        Some(&etag),
        Some(json!({})),
    )
    .await;
    assert_eq!(not_a_document.status(), StatusCode::BAD_REQUEST);
    assert_eq!(json(not_a_document).await["code"], "BAD_REQUEST");

    // The dry run: the changes against revision 1, nothing applied, the
    // revision standing, no resource named for the audit line.
    let dry = call(
        &app,
        "PUT",
        &format!("{PATH}?dry_run=true"),
        Some(ADMIN_KEY),
        Some(&etag),
        Some(document(600, true)),
    )
    .await;
    assert_eq!(dry.status(), StatusCode::OK);
    assert!(
        dry.extensions().get::<AuditResource>().is_none(),
        "a dry run names no resource"
    );
    assert_eq!(header(&dry, &REVISION), Some("1".to_string()));
    let dry = json(dry).await;
    assert_eq!(dry["applied"], false, "{dry}");
    assert_eq!(dry["revision"], 1, "{dry}");
    assert_eq!(dry["etag"], etag, "{dry}");
    assert_eq!(
        dry["changes"]["tenants_added"],
        json!(["auth:team-blue"]),
        "{dry}"
    );
    assert_eq!(
        dry["changes"]["tenants_changed"],
        json!(["auth:team-red"]),
        "{dry}"
    );
    assert_eq!(dry["changes"]["tenants_removed"], json!([]), "{dry}");
    assert_eq!(dry["changes"]["default_changed"], false, "{dry}");
    assert_eq!(dry["summary"]["tenants"], 2, "{dry}");
    assert!(dry.get("buckets").is_none(), "{dry}");
    let still = json(call(&app, "GET", PATH, Some(ADMIN_KEY), None, None).await).await;
    assert_eq!(still["revision"], 1, "{still}");

    // The write: the admin role alone; revision 2 under the key's id, the
    // resource named for the audit line, the new ETag in the headers.
    assert_eq!(
        call(
            &app,
            "PUT",
            PATH,
            Some(USER_KEY),
            Some(&etag),
            Some(document(600, true))
        )
        .await
        .status(),
        StatusCode::FORBIDDEN,
        "the user role may not write the document"
    );
    let applied = call(
        &app,
        "PUT",
        PATH,
        Some(ADMIN_KEY),
        Some(&etag),
        Some(document(600, true)),
    )
    .await;
    assert_eq!(applied.status(), StatusCode::OK);
    assert_eq!(
        applied.extensions().get::<AuditResource>(),
        Some(&AuditResource("rate-limits/tenants".to_string()))
    );
    let new_etag = header(&applied, &ETAG).expect("the new ETag");
    assert_ne!(new_etag, etag);
    assert_eq!(header(&applied, &REVISION), Some("2".to_string()));
    let applied = json(applied).await;
    assert_eq!(applied["applied"], true, "{applied}");
    assert_eq!(applied["revision"], 2, "{applied}");
    assert_eq!(applied["etag"], new_etag, "{applied}");
    assert_eq!(applied["principal"], "admin", "{applied}");
    assert_eq!(
        applied["buckets"],
        json!({"kept": 0, "replaced": 0, "dropped": 0}),
        "no bucket was live: {applied}"
    );

    // The ETag read first is stale now; the read shows the new document.
    assert_eq!(
        call(
            &app,
            "PUT",
            PATH,
            Some(ADMIN_KEY),
            Some(&etag),
            Some(document(700, true))
        )
        .await
        .status(),
        StatusCode::PRECONDITION_FAILED
    );
    let now = json(call(&app, "GET", PATH, Some(ADMIN_KEY), None, None).await).await;
    assert_eq!(now["revision"], 2, "{now}");
    assert_eq!(now["etag"], new_etag, "{now}");
    assert_eq!(now["principal"], "admin", "{now}");
    assert_eq!(
        now["document"]["tenants"][0]["tokens_per_minute"], 600,
        "{now}"
    );
    assert_eq!(
        now["document"]["tenants"][1]["tenant_key"], "auth:team-blue",
        "{now}"
    );
    ctx.shutdown().await;
}

#[tokio::test]
async fn without_a_tenant_rate_limiter_there_is_no_document() {
    let ctx = AppTestContext::new_with_config(
        TestRouterConfig::round_robin(4357),
        vec![TestWorkerConfig::healthy(20357)],
    )
    .await;
    let app = create_test_app_with_context_and_auth(
        ctx.router.clone(),
        ctx.app_context.clone(),
        Some(control_plane_auth()),
    );
    let read = call(&app, "GET", PATH, Some(ADMIN_KEY), None, None).await;
    assert_eq!(read.status(), StatusCode::CONFLICT);
    assert_eq!(json(read).await["code"], "RATE_LIMITER_DISABLED");
    let write = call(
        &app,
        "PUT",
        PATH,
        Some(ADMIN_KEY),
        Some("\"any\""),
        Some(document(600, false)),
    )
    .await;
    assert_eq!(write.status(), StatusCode::CONFLICT);
    assert_eq!(json(write).await["code"], "RATE_LIMITER_DISABLED");
    ctx.shutdown().await;
}
