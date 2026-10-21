//! Tenant resolution and request-meta insertion for serving paths.

use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::Instant,
};

use axum::{
    body::Body,
    extract::{connect_info::ConnectInfo, Request, State},
    http::{header::InvalidHeaderName, Extensions, HeaderMap, HeaderName},
    middleware::Next,
    response::Response,
};
use futures_util::future::{ready, Either, Ready};
use tower::{Layer, Service};

use super::{logging::AccessLogRecord, request_id::RequestId};
use crate::{
    config::{RouterConfig, TenantResolutionConfig},
    routers::error,
    tenant::{
        canonical_tenant_key, DataPlaneCaller, RouteRequestMeta, TenantIdentity, TenantKey,
        MAX_TENANT_ID_LEN,
    },
};

/// When the gateway took the request in: stamped on the route request meta as
/// it is built and stamped again when an admission permit is granted
/// ([`restamp_accepted_at`]), so the bound on the wait ahead of worker
/// selection measures the time after admission, not the admission queue's
/// own, bounded wait. Read once: the first worker selection of a request
/// takes the stamp ([`Self::take_for_selection`]); a later pipeline run of the
/// same request (the Responses tool loops run the pipeline once per iteration
/// with the same meta) finds it spent and is not bounded by it.
#[derive(Clone, Debug)]
pub struct AcceptedAt {
    at: Instant,
    spent: Arc<AtomicBool>,
}

impl AcceptedAt {
    /// A stamp taken now, on the clock the bound is read on
    /// ([`Self::clock_now`]).
    pub fn now() -> Self {
        Self::at(Self::clock_now())
    }

    /// The clock an acceptance is stamped on and the bound ahead of worker
    /// selection is read on, both ends the same: the runtime's, which is the
    /// wall clock in production and the paused clock of a test that advances
    /// time itself.
    pub fn clock_now() -> Instant {
        tokio::time::Instant::now().into_std()
    }

    /// A stamp taken at `at`.
    pub fn at(at: Instant) -> Self {
        Self {
            at,
            spent: Arc::new(AtomicBool::new(false)),
        }
    }

    /// When the request was accepted or admitted.
    pub fn instant(&self) -> Instant {
        self.at
    }

    /// The stamp, for the one selection that bounds the request's wait; `None`
    /// once a selection has taken it, on this meta or any clone of it.
    pub fn take_for_selection(&self) -> Option<Instant> {
        (!self.spent.swap(true, Ordering::AcqRel)).then_some(self.at)
    }
}

/// Stamp the request's route meta with a fresh [`AcceptedAt`]: called when an
/// admission permit is granted, so the wait in the admission queue (bounded by
/// its own timeout) does not count against the bound ahead of worker
/// selection. `false` when the request carries no route meta.
pub fn restamp_accepted_at(extensions: &mut Extensions) -> bool {
    match extensions.get_mut::<RouteRequestMeta>() {
        Some(meta) => {
            meta.insert_extension(AcceptedAt::now());
            true
        }
        None => false,
    }
}

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

/// The caller's tenant key, or the 400 a trusted tenant header earns when its
/// value is longer than [`MAX_TENANT_ID_LEN`] (the key is stored with every
/// response and conversation; a longer one would fail on the SQL backends
/// only, and silently dropping it would leave the request without a tenant).
fn resolve_raw_tenant_key(
    state: &TenantResolutionState,
    request: &Request<Body>,
) -> Result<TenantKey, Response> {
    if let Some(caller) = request.extensions().get::<DataPlaneCaller>() {
        return Ok(caller.tenant_key().clone());
    }
    if state.trust_tenant_header {
        if let Some(tenant_id) = extract_trusted_tenant_id(state, request.headers()) {
            if tenant_id.len() > MAX_TENANT_ID_LEN {
                return Err(error::bad_request(
                    "invalid_tenant_header",
                    format!(
                        "The {} header value is longer than {MAX_TENANT_ID_LEN} characters",
                        state.trusted_tenant_header_name
                    ),
                ));
            }
            return Ok(canonical_tenant_key(TenantIdentity::Header(Arc::from(
                tenant_id,
            ))));
        }
    }

    if let Some(ConnectInfo(addr)) = request.extensions().get::<ConnectInfo<SocketAddr>>() {
        return Ok(canonical_tenant_key(TenantIdentity::IpAddress(addr.ip())));
    }

    Ok(canonical_tenant_key(TenantIdentity::Anonymous))
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

/// The request's route meta, or the answer a request with an unusable tenant
/// header gets (see [`resolve_raw_tenant_key`]).
pub fn resolve_route_request_meta(
    state: &TenantResolutionState,
    request: &Request<Body>,
) -> Result<RouteRequestMeta, Response> {
    let meta = RouteRequestMeta::new(resolve_raw_tenant_key(state, request)?)
        .with_extension(AcceptedAt::now());
    // Carry the middleware request id so backend request ids derive from it
    // (RequestIdLayer runs outside this middleware).
    Ok(match request.extensions().get::<RequestId>() {
        Some(request_id) => meta.with_extension(request_id.clone()),
        None => meta,
    })
}

pub async fn route_request_meta_middleware(
    State(state): State<TenantResolutionState>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    if let Err(response) = attach_route_request_meta(&state, &mut request) {
        return response;
    }
    next.run(request).await
}

/// The request's route meta resolved and attached (the access log told the
/// tenant), or the response a request with an unusable tenant header gets.
fn attach_route_request_meta(
    state: &TenantResolutionState,
    request: &mut Request<Body>,
) -> Result<(), Response> {
    let request_meta = resolve_route_request_meta(state, request)?;
    // The access log, when it carries the tenant.
    if let Some(record) = request.extensions().get::<AccessLogRecord>() {
        record.tenant(request_meta.tenant_key().as_str());
    }
    request.extensions_mut().insert(request_meta);
    Ok(())
}

impl TenantResolutionState {
    /// [`route_request_meta_middleware`] as a lean route layer: the meta is
    /// resolved and attached before the inner service is called, and the
    /// inner service's own future is returned (no clone, no box per
    /// request, which the `from_fn` form made: 6 allocation calls and 1.9 KiB
    /// per request, measured 2026-10-21); a request with an unusable tenant
    /// header gets its response at once.
    pub fn layer(self) -> RouteRequestMetaLayer {
        RouteRequestMetaLayer { state: self }
    }
}

/// [`TenantResolutionState::layer`].
#[derive(Clone)]
pub struct RouteRequestMetaLayer {
    state: TenantResolutionState,
}

impl<S> Layer<S> for RouteRequestMetaLayer {
    type Service = RouteRequestMetaService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RouteRequestMetaService {
            inner,
            state: self.state.clone(),
        }
    }
}

/// The service [`RouteRequestMetaLayer`] wraps a route in.
#[derive(Clone)]
pub struct RouteRequestMetaService<S> {
    inner: S,
    state: TenantResolutionState,
}

impl<S> Service<Request<Body>> for RouteRequestMetaService<S>
where
    S: Service<Request<Body>, Response = Response, Error = Infallible>,
{
    type Response = Response;
    type Error = Infallible;
    type Future = Either<Ready<Result<Response, Infallible>>, S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut request: Request<Body>) -> Self::Future {
        match attach_route_request_meta(&self.state, &mut request) {
            Ok(()) => Either::Right(self.inner.call(request)),
            Err(response) => Either::Left(ready(Ok(response))),
        }
    }
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
        let meta =
            RouteRequestMeta::new(TenantKey::new("t")).with_extension(AcceptedAt::at(earlier));
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
}
