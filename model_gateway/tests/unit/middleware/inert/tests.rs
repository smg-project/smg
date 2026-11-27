use axum::{
    body::Body,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, Router},
};
use futures_util::future::{ready, Ready};
use tower::ServiceExt;

use super::*;

/// A layer whose service refuses every request, for the active arm.
#[derive(Clone)]
struct Refuse;

#[derive(Clone)]
struct RefuseService<S>(S);

impl<S> Layer<S> for Refuse {
    type Service = RefuseService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RefuseService(inner)
    }
}

impl<S> Service<Request> for RefuseService<S>
where
    S: Service<Request, Response = Response, Error = Infallible>,
{
    type Response = Response;
    type Error = Infallible;
    type Future = Ready<Result<Response, Infallible>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
        self.0.poll_ready(cx)
    }

    fn call(&mut self, _request: Request) -> Self::Future {
        ready(Ok(StatusCode::IM_A_TEAPOT.into_response()))
    }
}

async fn status_of(app: Router) -> StatusCode {
    let request = Request::builder().uri("/").body(Body::empty()).unwrap();
    app.oneshot(request).await.unwrap().status()
}

#[tokio::test]
async fn an_inert_member_passes_the_request_through() {
    let app = Router::new()
        .route("/", get(|| async { StatusCode::OK }))
        .route_layer(tower::ServiceBuilder::new().layer(InertLayer::<Refuse>(None)));
    assert_eq!(status_of(app).await, StatusCode::OK);
}

#[tokio::test]
async fn an_active_member_acts() {
    let app = Router::new()
        .route("/", get(|| async { StatusCode::OK }))
        .route_layer(tower::ServiceBuilder::new().layer(InertLayer(Some(Refuse))));
    assert_eq!(status_of(app).await, StatusCode::IM_A_TEAPOT);
}
