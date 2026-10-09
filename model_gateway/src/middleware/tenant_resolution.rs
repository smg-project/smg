//! Tenant resolution and request-meta insertion for serving paths.

use std::{net::SocketAddr, sync::Arc, time::Instant};

use axum::{
    body::Body,
    extract::{connect_info::ConnectInfo, Request, State},
    http::{header::InvalidHeaderName, HeaderMap, HeaderName},
    middleware::Next,
    response::Response,
};

use super::request_id::RequestId;
use crate::{
    config::{RouterConfig, TenantResolutionConfig},
    tenant::{canonical_tenant_key, DataPlaneCaller, RouteRequestMeta, TenantIdentity, TenantKey},
};

/// When the gateway accepted the request: stamped on the route request meta
/// as it is built, before admission, so every later stage can bound the time
/// a request has already spent inside the gateway.
#[derive(Clone, Copy, Debug)]
pub struct AcceptedAt(pub Instant);

#[derive(Clone)]
pub struct TenantResolutionState {
    trust_tenant_header: bool,
    trusted_tenant_header_name: HeaderName,
}

impl TenantResolutionState {
    pub fn new(config: &RouterConfig) -> Result<Self, InvalidHeaderName> {
        Self::from_config(&config.tenant_resolution)
    }

    pub fn from_config(config: &TenantResolutionConfig) -> Result<Self, InvalidHeaderName> {
        let trusted_tenant_header_name: HeaderName = config.tenant_header_name.parse()?;

        Ok(Self {
            trust_tenant_header: config.trust_tenant_header,
            trusted_tenant_header_name,
        })
    }
}

fn resolve_raw_tenant_key(state: &TenantResolutionState, request: &Request<Body>) -> TenantKey {
    if let Some(caller) = request.extensions().get::<DataPlaneCaller>() {
        return caller.tenant_key().clone();
    }
    if state.trust_tenant_header {
        if let Some(tenant_id) = extract_trusted_tenant_id(state, request.headers()) {
            return canonical_tenant_key(TenantIdentity::Header(Arc::from(tenant_id)));
        }
    }

    if let Some(ConnectInfo(addr)) = request.extensions().get::<ConnectInfo<SocketAddr>>() {
        return canonical_tenant_key(TenantIdentity::IpAddress(addr.ip()));
    }

    canonical_tenant_key(TenantIdentity::Anonymous)
}

fn extract_trusted_tenant_id<'a>(
    state: &TenantResolutionState,
    headers: &'a HeaderMap,
) -> Option<&'a str> {
    headers
        .get(&state.trusted_tenant_header_name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

pub fn resolve_route_request_meta(
    state: &TenantResolutionState,
    request: &Request<Body>,
) -> RouteRequestMeta {
    let meta = RouteRequestMeta::new(resolve_raw_tenant_key(state, request))
        .with_extension(AcceptedAt(Instant::now()));
    // Carry the middleware request id so backend request ids derive from it
    // (RequestIdLayer runs outside this middleware).
    match request.extensions().get::<RequestId>() {
        Some(request_id) => meta.with_extension(request_id.clone()),
        None => meta,
    }
}

pub async fn route_request_meta_middleware(
    State(state): State<TenantResolutionState>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    let request_meta = resolve_route_request_meta(&state, &request);
    request.extensions_mut().insert(request_meta);
    next.run(request).await
}

/// Backward-compatible alias for [`route_request_meta_middleware`].
pub async fn ordinary_tenant_resolution_middleware(
    state: State<TenantResolutionState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    route_request_meta_middleware(state, request, next).await
}

#[cfg(test)]
mod tests {
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

        let request_meta = resolve_route_request_meta(&state, &request);
        assert_eq!(request_meta.tenant_key().as_str(), "auth:b3c2");
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

        let request_meta = resolve_route_request_meta(&state, &request);
        assert_eq!(request_meta.tenant_key().as_str(), "header:team-red");
    }

    #[tokio::test]
    async fn request_meta_falls_back_to_client_ip() {
        let state = resolution_state();
        let mut request = Request::builder().uri("/").body(Body::empty()).unwrap();
        request.extensions_mut().insert(ConnectInfo(
            "203.0.113.42:443".parse::<SocketAddr>().unwrap(),
        ));

        let request_meta = resolve_route_request_meta(&state, &request);
        assert_eq!(request_meta.tenant_key().as_str(), "ip:203.0.113.42");
    }

    #[tokio::test]
    async fn request_meta_carries_middleware_request_id() {
        let state = resolution_state();
        let mut request = Request::builder().uri("/").body(Body::empty()).unwrap();
        request
            .extensions_mut()
            .insert(RequestId("chatcmpl-abc123".to_string()));

        let request_meta = resolve_route_request_meta(&state, &request);
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

        let request_meta = resolve_route_request_meta(&state, &request);
        let accepted = request_meta
            .extension::<AcceptedAt>()
            .expect("the meta carries when the gateway accepted the request");
        assert!(accepted.0 >= before && accepted.0 <= Instant::now());
    }

    #[tokio::test]
    async fn request_meta_falls_back_to_anonymous_without_identity_sources() {
        let state = resolution_state();
        let request = Request::builder().uri("/").body(Body::empty()).unwrap();

        let request_meta = resolve_route_request_meta(&state, &request);
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
}
