//! A request's deadline, the gateway's `--request-timeout-secs`: the standard
//! `grpc-timeout` header on the generate, embed and encode requests to a worker
//! ([`deadline_request`]), which the worker's server honours by cancelling
//! the call, and the gateway's own end of a stream whose frames stopped past
//! it ([`crate::AbortOnDropStream`]). tonic's client bounds the call that
//! opens the stream from the header too, as `CANCELLED` "Timeout expired";
//! [`is_call_deadline`] reads both. The deadline is one setting for the
//! process ([`set_request_timeout`]), read per request; none set means none,
//! as before. The long-lived KV-events subscription carries no deadline.

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use tokio::time::Instant;
use tonic::{Code, Request, Status};

/// How the deadline's status message begins (what tells it from a worker's).
const MESSAGE_PREFIX: &str = "request timed out after ";
/// tonic's own words for the header's bound it enforces on the opening call.
const TONIC_EXPIRY: &str = "Timeout expired";
/// The header holds eight digits of hours at most; tonic panics beyond.
const MAX_HEADER_TIMEOUT: Duration = Duration::from_secs(99_999_999 * 60 * 60);
/// The configured timeout in seconds; zero: none.
static REQUEST_TIMEOUT_SECS: AtomicU64 = AtomicU64::new(0);

/// Make every generate, embed and encode request to a worker from now on
/// carry `timeout` as its deadline (`None`, or zero: no deadline).
pub fn set_request_timeout(timeout: Option<Duration>) {
    REQUEST_TIMEOUT_SECS.store(timeout.map_or(0, |t| t.as_secs()), Ordering::Relaxed);
}

/// The deadline a request to a worker carries, when one is configured.
pub fn request_timeout() -> Option<Duration> {
    match REQUEST_TIMEOUT_SECS.load(Ordering::Relaxed) {
        0 => None,
        secs => Some(Duration::from_secs(secs)),
    }
}

/// A request for `req` under the configured deadline: the `grpc-timeout`
/// header set when one is configured, and the deadline for the stream the
/// call opens (`None` when none is configured).
pub fn deadline_request<T>(req: T) -> (Option<CallDeadline>, Request<T>) {
    let mut request = Request::new(req);
    let deadline = request_timeout().map(|timeout| {
        request.set_timeout(timeout.min(MAX_HEADER_TIMEOUT));
        CallDeadline::after(timeout)
    });
    (deadline, request)
}

/// The deadline of one call to a worker: `timeout` after the call was issued.
#[derive(Clone, Copy, Debug)]
pub struct CallDeadline {
    at: Instant,
    timeout: Duration,
}

impl CallDeadline {
    /// The deadline `timeout` from now; one the clock cannot add is a year away.
    pub fn after(timeout: Duration) -> Self {
        let now = Instant::now();
        let at = now
            .checked_add(timeout)
            .unwrap_or_else(|| now + Duration::from_secs(365 * 24 * 60 * 60));
        Self { at, timeout }
    }

    /// The instant the deadline passes.
    pub fn at(self) -> Instant {
        self.at
    }

    /// The status a stream ends with when the deadline passes.
    pub fn status(self) -> Status {
        Status::deadline_exceeded(format!(
            "{MESSAGE_PREFIX}{:?}: the worker did not finish the request within the call's deadline",
            self.timeout
        ))
    }
}

/// Whether `status` is the call's own deadline (the stream's timer, or tonic's
/// expiry of the header's bound on the opening call), not a worker's
/// `DEADLINE_EXCEEDED`.
pub fn is_call_deadline(status: &Status) -> bool {
    (status.code() == Code::DeadlineExceeded && status.message().starts_with(MESSAGE_PREFIX))
        || (status.code() == Code::Cancelled && status.message() == TONIC_EXPIRY)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, PoisonError};

    use tokio::sync::mpsc;
    use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
    use tonic::{transport::Server, Response};

    use super::*;
    use crate::{common_proto, vllm_engine::proto, VllmEngineClient};

    fn header_of<T>(request: &Request<T>) -> Option<String> {
        let value = request.metadata().get("grpc-timeout")?;
        Some(value.to_str().ok()?.to_owned())
    }

    /// An engine that records, per RPC, whether its request carried the
    /// `grpc-timeout` header; its streams end at once.
    #[derive(Clone, Default)]
    struct Recorder {
        seen: Arc<Mutex<Vec<(&'static str, bool)>>>,
    }

    impl Recorder {
        fn note<T>(&self, rpc: &'static str, request: &Request<T>) {
            self.seen
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push((rpc, request.metadata().contains_key("grpc-timeout")));
        }
    }

    fn ended<T: Send + 'static>() -> ReceiverStream<Result<T, Status>> {
        let (_tx, rx) = mpsc::channel(1);
        ReceiverStream::new(rx)
    }

    #[tonic::async_trait]
    impl proto::vllm_engine_server::VllmEngine for Recorder {
        type GenerateStream = ReceiverStream<Result<proto::GenerateResponse, Status>>;
        type GetTokenizerStream = ReceiverStream<Result<common_proto::GetTokenizerChunk, Status>>;
        type SubscribeKvEventsStream = ReceiverStream<Result<common_proto::KvEventBatch, Status>>;

        async fn generate(
            &self,
            request: Request<proto::GenerateRequest>,
        ) -> Result<Response<Self::GenerateStream>, Status> {
            self.note("generate", &request);
            Ok(Response::new(ended()))
        }

        async fn embed(
            &self,
            request: Request<proto::EmbedRequest>,
        ) -> Result<Response<proto::EmbedResponse>, Status> {
            self.note("embed", &request);
            Err(Status::unimplemented("recorder"))
        }

        async fn subscribe_kv_events(
            &self,
            request: Request<common_proto::SubscribeKvEventsRequest>,
        ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
            self.note("subscribe_kv_events", &request);
            Ok(Response::new(ended()))
        }

        async fn abort(
            &self,
            _: Request<proto::AbortRequest>,
        ) -> Result<Response<proto::AbortResponse>, Status> {
            Ok(Response::new(proto::AbortResponse::default()))
        }

        async fn flush_cache(
            &self,
            _: Request<common_proto::FlushCacheRequest>,
        ) -> Result<Response<common_proto::FlushCacheResponse>, Status> {
            Err(Status::unimplemented("recorder"))
        }

        async fn health_check(
            &self,
            _: Request<proto::HealthCheckRequest>,
        ) -> Result<Response<proto::HealthCheckResponse>, Status> {
            Err(Status::unimplemented("recorder"))
        }

        async fn get_model_info(
            &self,
            _: Request<proto::GetModelInfoRequest>,
        ) -> Result<Response<proto::GetModelInfoResponse>, Status> {
            Err(Status::unimplemented("recorder"))
        }

        async fn get_server_info(
            &self,
            _: Request<proto::GetServerInfoRequest>,
        ) -> Result<Response<proto::GetServerInfoResponse>, Status> {
            Err(Status::unimplemented("recorder"))
        }

        async fn get_loads(
            &self,
            _: Request<proto::GetLoadsRequest>,
        ) -> Result<Response<proto::GetLoadsResponse>, Status> {
            Err(Status::unimplemented("recorder"))
        }

        async fn get_tokenizer(
            &self,
            _: Request<common_proto::GetTokenizerRequest>,
        ) -> Result<Response<Self::GetTokenizerStream>, Status> {
            Err(Status::unimplemented("recorder"))
        }
    }

    /// One test for the one process-wide setting: the header through the
    /// helper, then through a real client against a recording engine, where
    /// the generate and embed requests carry it and the long-lived KV-events
    /// subscription does not; and the two shapes of the call's own deadline.
    #[tokio::test]
    async fn generate_and_embed_carry_the_deadline_and_the_kv_events_subscription_does_not() {
        set_request_timeout(Some(Duration::from_secs(15)));
        let (deadline, request) = deadline_request(());
        assert!(deadline.is_some());
        assert_eq!(header_of(&request).as_deref(), Some("15000000u"));
        set_request_timeout(Some(Duration::from_secs(u64::MAX)));
        assert_eq!(
            header_of(&deadline_request(()).1).as_deref(),
            Some("99999999H")
        );
        set_request_timeout(None);
        assert!(deadline_request(()).0.is_none());
        assert_eq!(header_of(&deadline_request(()).1), None);

        let recorder = Recorder::default();
        let seen = Arc::clone(&recorder.seen);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a port");
        let addr = listener.local_addr().expect("the address");
        #[expect(
            clippy::disallowed_methods,
            reason = "the recording engine lives for the test process"
        )]
        tokio::spawn(
            Server::builder()
                .add_service(proto::vllm_engine_server::VllmEngineServer::new(recorder))
                .serve_with_incoming(TcpListenerStream::new(listener)),
        );
        set_request_timeout(Some(Duration::from_secs(15)));
        let client = VllmEngineClient::connect(&format!("http://{addr}"))
            .await
            .expect("connect");
        let stream = client
            .generate(proto::GenerateRequest::default())
            .await
            .expect("a stream");
        stream.mark_completed();
        drop(stream);
        let _ = client.embed(proto::EmbedRequest::default()).await;
        drop(client.subscribe_kv_events(0).await.expect("a subscription"));
        set_request_timeout(None);
        let seen = seen.lock().unwrap_or_else(PoisonError::into_inner).clone();
        assert_eq!(
            seen,
            vec![
                ("generate", true),
                ("embed", true),
                ("subscribe_kv_events", false)
            ]
        );

        let timer = CallDeadline::after(Duration::from_secs(15)).status();
        assert!(timer.message().starts_with("request timed out after 15s"));
        assert!(is_call_deadline(&timer));
        assert_eq!(tonic::TimeoutExpired(()).to_string(), TONIC_EXPIRY);
        assert!(is_call_deadline(&Status::cancelled(TONIC_EXPIRY)));
        assert!(!is_call_deadline(&Status::deadline_exceeded(
            "engine scheduling deadline exceeded"
        )));
        assert!(!is_call_deadline(&Status::cancelled("stream reset")));
        assert!(CallDeadline::after(Duration::from_secs(u64::MAX)).at() > Instant::now());
    }
}
