//! `grpc.health.v1` for a servicer: SERVING is a single predicate over the
//! servicer's state, reported for the service names it hosts.

use std::{pin::Pin, sync::Arc, time::Duration};

use futures::Stream;
use tonic::{Request, Response, Status};
use tonic_health::pb::{
    health_check_response::ServingStatus, health_server::Health, HealthCheckRequest,
    HealthCheckResponse,
};

/// `Health.Watch` poll interval.
const HEALTH_WATCH_INTERVAL: Duration = Duration::from_millis(250);

type HealthWatchStream = Pin<Box<dyn Stream<Item = Result<HealthCheckResponse, Status>> + Send>>;

/// Serving status for a fixed set of service names, read from one predicate.
#[derive(Clone)]
pub(crate) struct HealthReporter {
    services: Vec<String>,
    serving: Arc<dyn Fn() -> bool + Send + Sync>,
}

impl HealthReporter {
    /// `services` are the names answered (the empty name is the whole server);
    /// `serving` is evaluated per probe.
    pub(crate) fn new(services: &[&str], serving: Arc<dyn Fn() -> bool + Send + Sync>) -> Self {
        Self {
            services: services.iter().map(|s| s.to_string()).collect(),
            serving,
        }
    }

    fn status(&self, service: &str) -> Result<ServingStatus, Status> {
        if !self.services.iter().any(|s| s == service) {
            return Err(Status::not_found(format!("unknown service {service:?}")));
        }
        Ok(if (self.serving)() {
            ServingStatus::Serving
        } else {
            ServingStatus::NotServing
        })
    }
}

#[tonic::async_trait]
impl Health for HealthReporter {
    type WatchStream = HealthWatchStream;

    async fn check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let status = self.status(&request.into_inner().service)?;
        Ok(Response::new(HealthCheckResponse {
            status: status as i32,
        }))
    }

    async fn watch(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<Self::WatchStream>, Status> {
        let service = request.into_inner().service;
        let stream = futures::stream::unfold(
            (self.clone(), service, None),
            |(reporter, service, last)| async move {
                loop {
                    let status = match reporter.status(&service) {
                        Ok(status) => status as i32,
                        Err(status) => return Some((Err(status), (reporter, service, last))),
                    };
                    if last != Some(status) {
                        return Some((
                            Ok(HealthCheckResponse { status }),
                            (reporter, service, Some(status)),
                        ));
                    }
                    tokio::time::sleep(HEALTH_WATCH_INTERVAL).await;
                }
            },
        );
        Ok(Response::new(Box::pin(stream)))
    }
}
