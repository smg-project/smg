use axum::{
    body::Body,
    extract::connect_info::ConnectInfo,
    http::{header, HeaderValue, Request, StatusCode},
    middleware::from_fn_with_state,
    response::IntoResponse,
    routing::get,
    Router,
};
use tower::ServiceExt;

use super::*;
use crate::{
    config::{PolicyConfig, RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    tenant::DEFAULT_TENANT_HEADER_NAME,
};

fn resolution_state() -> TenantResolutionState {
    TenantResolutionState::new(&RouterConfig::new(
        RoutingMode::Regular {
            worker_urls: vec!["http://worker1:8000".to_string()],
        },
        PolicyConfig::Random,
    ))
    .unwrap()
}

#[tokio::test]
async fn request_meta_prefers_authenticated_data_plane_identity() {
    let state = resolution_state();
    let mut request = Request::builder().uri("/").body(Body::empty()).unwrap();
    request
        .extensions_mut()
        .insert(DataPlaneCaller::new(TenantKey::from("auth:b3c2")));
    request
        .extensions_mut()
        .insert(ConnectInfo("127.0.0.1:8080".parse::<SocketAddr>().unwrap()));

    let request_meta = resolve_route_request_meta(&state, &request).expect("a tenant resolves");
    assert_eq!(request_meta.tenant_key().as_str(), "auth:b3c2");
}

/// The lean layer does what the `from_fn` form did: the route sees the
/// request's meta (the trusted header's tenant), and an unusable header
/// is answered with 400 before the route runs.
#[tokio::test]
async fn the_route_meta_layer_attaches_the_tenant_and_refuses_an_unusable_header() {
    let mut config = RouterConfig::new(
        RoutingMode::Regular {
            worker_urls: vec!["http://worker1:8000".to_string()],
        },
        PolicyConfig::Random,
    );
    config.tenant_resolution.trust_tenant_header = true;
    let state = TenantResolutionState::new(&config).unwrap();
    let app = Router::new()
        .route(
            "/",
            get(|request: Request<Body>| async move {
                request
                    .extensions()
                    .get::<RouteRequestMeta>()
                    .map(|meta| meta.tenant_key().as_str().to_string())
                    .unwrap_or_else(|| "no meta".to_string())
            }),
        )
        .route_layer(state.layer());
    let call = |header: String| {
        let app = app.clone();
        async move {
            let request = Request::builder()
                .uri("/")
                .header(DEFAULT_TENANT_HEADER_NAME, header)
                .body(Body::empty())
                .unwrap();
            let response = app.oneshot(request).await.unwrap();
            let status = response.status();
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            (status, String::from_utf8_lossy(&body).to_string())
        }
    };
    let (status, body) = call("team-red".to_string()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, "header:team-red");
    let (status, body) = call("x".repeat(MAX_TENANT_ID_LEN + 1)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.contains("invalid_tenant_header"), "{body}");
}

#[tokio::test]
async fn request_meta_uses_trusted_header_when_enabled() {
    let mut config = RouterConfig::new(
        RoutingMode::Regular {
            worker_urls: vec!["http://worker1:8000".to_string()],
        },
        PolicyConfig::Random,
    );
    config.tenant_resolution.trust_tenant_header = true;
    let state = TenantResolutionState::new(&config).unwrap();

    let request = Request::builder()
        .uri("/")
        .header(DEFAULT_TENANT_HEADER_NAME, "team-red")
        .body(Body::empty())
        .unwrap();

    let request_meta = resolve_route_request_meta(&state, &request).expect("a tenant resolves");
    assert_eq!(request_meta.tenant_key().as_str(), "header:team-red");
}

/// A trusted tenant header longer than the bound is refused with 400
/// rather than stored, dropped or truncated.
#[tokio::test]
async fn an_over_long_trusted_tenant_header_is_refused() {
    let mut config = RouterConfig::new(
        RoutingMode::Regular {
            worker_urls: vec!["http://worker1:8000".to_string()],
        },
        PolicyConfig::Random,
    );
    config.tenant_resolution.trust_tenant_header = true;
    let state = TenantResolutionState::new(&config).unwrap();
    let longest = "t".repeat(MAX_TENANT_ID_LEN);
    let request = Request::builder()
        .uri("/")
        .header(DEFAULT_TENANT_HEADER_NAME, longest.as_str())
        .body(Body::empty())
        .unwrap();
    let meta = resolve_route_request_meta(&state, &request).expect("at the bound");
    assert_eq!(meta.tenant_key().as_str(), format!("header:{longest}"));

    let request = Request::builder()
        .uri("/")
        .header(
            DEFAULT_TENANT_HEADER_NAME,
            "t".repeat(MAX_TENANT_ID_LEN + 1),
        )
        .body(Body::empty())
        .unwrap();
    let refused = resolve_route_request_meta(&state, &request).expect_err("over the bound");
    assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        refused
            .headers()
            .get(error::HEADER_X_SMG_ERROR_CODE)
            .unwrap(),
        "invalid_tenant_header"
    );
}

#[tokio::test]
async fn request_meta_falls_back_to_client_ip() {
    let state = resolution_state();
    let mut request = Request::builder().uri("/").body(Body::empty()).unwrap();
    request.extensions_mut().insert(ConnectInfo(
        "203.0.113.42:443".parse::<SocketAddr>().unwrap(),
    ));

    let request_meta = resolve_route_request_meta(&state, &request).expect("a tenant resolves");
    assert_eq!(request_meta.tenant_key().as_str(), "ip:203.0.113.42");
}

#[tokio::test]
async fn request_meta_carries_middleware_request_id() {
    let state = resolution_state();
    let mut request = Request::builder().uri("/").body(Body::empty()).unwrap();
    request
        .extensions_mut()
        .insert(RequestId("chatcmpl-abc123".to_string()));

    let request_meta = resolve_route_request_meta(&state, &request).expect("a tenant resolves");
    assert_eq!(
        request_meta
            .extension::<RequestId>()
            .map(|request_id| request_id.0.as_str()),
        Some("chatcmpl-abc123")
    );
}

#[tokio::test]
async fn request_meta_is_stamped_with_its_acceptance_time() {
    let state = resolution_state();
    let before = Instant::now();
    let request = Request::builder().uri("/").body(Body::empty()).unwrap();

    let request_meta = resolve_route_request_meta(&state, &request).expect("a tenant resolves");
    let accepted = request_meta
        .extension::<AcceptedAt>()
        .expect("the meta carries when the gateway accepted the request");
    assert!(accepted.instant() >= before && accepted.instant() <= Instant::now());
}

/// The first selection takes the stamp; a later pipeline run of the same
/// request (a clone of the meta, as the tool loops pass it) finds it spent.
#[test]
fn the_acceptance_stamp_is_taken_once_across_clones() {
    let accepted = AcceptedAt::now();
    let shared = accepted.clone();
    assert!(accepted.take_for_selection().is_some());
    assert_eq!(shared.take_for_selection(), None, "spent on the clone too");
    assert_eq!(accepted.take_for_selection(), None);
}

/// An admission grant replaces the stamp with a fresh, unspent one.
#[test]
fn an_admission_grant_restamps_the_meta() {
    let earlier = Instant::now()
        .checked_sub(std::time::Duration::from_secs(5))
        .expect("the clock has been running for five seconds");
    let meta = RouteRequestMeta::new(TenantKey::new("t")).with_extension(AcceptedAt::at(earlier));
    meta.extension::<AcceptedAt>()
        .unwrap()
        .take_for_selection()
        .expect("the first stamp is fresh");
    let mut extensions = Extensions::new();
    extensions.insert(meta);

    assert!(restamp_accepted_at(&mut extensions));

    let stamp = extensions
        .get::<RouteRequestMeta>()
        .unwrap()
        .extension::<AcceptedAt>()
        .unwrap();
    assert!(stamp.instant() > earlier, "a fresh stamp");
    assert!(stamp.take_for_selection().is_some(), "and an unspent one");
    assert!(
        !restamp_accepted_at(&mut Extensions::new()),
        "no meta, nothing to stamp"
    );
}

#[tokio::test]
async fn request_meta_falls_back_to_anonymous_without_identity_sources() {
    let state = resolution_state();
    let request = Request::builder().uri("/").body(Body::empty()).unwrap();

    let request_meta = resolve_route_request_meta(&state, &request).expect("a tenant resolves");
    assert_eq!(request_meta.tenant_key().as_str(), "anonymous");
}

#[tokio::test]
async fn middleware_attaches_request_meta_extension() {
    async fn handler(request: Request<Body>) -> impl IntoResponse {
        request
            .extensions()
            .get::<TenantRequestMeta>()
            .map(|meta| meta.tenant_key().to_string())
            .unwrap_or_else(|| "missing".to_string())
    }

    let app = Router::new()
        .route("/", get(handler))
        .route_layer(from_fn_with_state(
            resolution_state(),
            route_request_meta_middleware,
        ));

    let response = app
        .oneshot(
            Request::builder()
                .uri("/")
                .header(
                    header::AUTHORIZATION,
                    HeaderValue::from_static("Bearer ignored"),
                )
                .extension(DataPlaneCaller::new(TenantKey::from("auth:tenant-a")))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(std::str::from_utf8(&body).unwrap(), "auth:tenant-a");
}
