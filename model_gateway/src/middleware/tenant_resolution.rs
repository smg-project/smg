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

/// When the gateway accepted the request: stamped as the route request meta
/// is built (after the serving auth and the wasm hooks, so their time is not
/// the request's wait), and stamped again when an admission permit is granted
/// (by the scheduler on every admit, by the legacy limiter after a wait in its
/// queue; [`restamp_accepted_at`]), so the bound on the wait ahead of worker
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
    /// time itself (the clock of the edge's entry stamp, `request_start`).
    pub fn clock_now() -> Instant {
        super::request_start::clock_now()
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
/// admission permit is granted (by the scheduler on every admit, by the legacy
/// limiter only after a wait in its queue), so a wait in the admission queue
/// (bounded by its own timeout) does not count against the bound ahead of
/// worker selection; a legacy permit granted at once leaves the stamp taken at
/// tenant resolution as the acceptance. `false` when the request carries no
/// route meta.
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
/// header gets (see `resolve_raw_tenant_key`).
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
#[path = "../../tests/unit/middleware/tenant_resolution/tests.rs"]
mod tests;
