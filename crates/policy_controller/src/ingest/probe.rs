//! The probe the controller dials the engines with: one gRPC client per
//! engine family (the client crate's public clients, the same `GetLoads`
//! the gateway asks, answers normalised by the same conversions), kept per
//! endpoint and family across polls and dropped on a transport failure so
//! the next poll redials, and one HTTP client for the scrape, built at the
//! first one.

use std::{collections::HashMap, sync::OnceLock, time::Duration};

use openai_protocol::worker::{RuntimeType, WorkerLoadResponse};
use parking_lot::Mutex;
use rustls::crypto::ring;
use smg_grpc_client::{
    engine_load::LOAD_SECTIONS, is_transport_failure, SglangSchedulerClient,
    TokenSpeedSchedulerClient, TrtllmServiceClient, VllmEngineClient,
};
use tokio::time::timeout;
use tonic::Code;
use tracing::error;

use super::loads::{LoadProbe, ReadError, Target};

/// The engines' probe.
pub struct EngineProbe {
    clients: Mutex<HashMap<String, Family>>,
    /// The scrape's client, built at the first scrape: building it reads the
    /// system's certificates, a few milliseconds and their memory that a
    /// fleet of servicers alone never needs. The build is tried once: a
    /// failure is kept as the text every later scrape fails with.
    http: OnceLock<Result<reqwest::Client, String>>,
}

/// One connected client, by family.
#[derive(Clone)]
enum Family {
    Sglang(SglangSchedulerClient),
    Vllm(VllmEngineClient),
    Trtllm(TrtllmServiceClient),
    TokenSpeed(TokenSpeedSchedulerClient),
}

impl Family {
    /// A connected client of the target's family, or the reason there is
    /// none: a family without the call, or a failed dial.
    async fn connect(target: &Target) -> Result<Self, ReadError> {
        let endpoint = target.endpoint.as_str();
        let dial = |error: Box<dyn std::error::Error + Send + Sync>| {
            ReadError::Transport(error.to_string())
        };
        match target.kind {
            RuntimeType::Sglang => SglangSchedulerClient::connect(endpoint)
                .await
                .map(Self::Sglang)
                .map_err(dial),
            RuntimeType::Vllm => VllmEngineClient::connect(endpoint)
                .await
                .map(Self::Vllm)
                .map_err(dial),
            RuntimeType::Trtllm => TrtllmServiceClient::connect(endpoint)
                .await
                .map(Self::Trtllm)
                .map_err(dial),
            RuntimeType::TokenSpeed => TokenSpeedSchedulerClient::connect(endpoint)
                .await
                .map(Self::TokenSpeed)
                .map_err(dial),
            kind => Err(ReadError::Unsupported(kind.as_str().to_owned())),
        }
    }

    /// The family this client speaks.
    fn kind(&self) -> RuntimeType {
        match self {
            Self::Sglang(_) => RuntimeType::Sglang,
            Self::Vllm(_) => RuntimeType::Vllm,
            Self::Trtllm(_) => RuntimeType::Trtllm,
            Self::TokenSpeed(_) => RuntimeType::TokenSpeed,
        }
    }

    /// The engine's `GetLoads` answer, normalised.
    async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        let include = LOAD_SECTIONS
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        match self {
            Self::Sglang(client) => client
                .get_loads(include)
                .await
                .map(WorkerLoadResponse::from),
            Self::Vllm(client) => client
                .get_loads(include)
                .await
                .map(WorkerLoadResponse::from),
            Self::Trtllm(client) => client
                .get_loads(include)
                .await
                .map(WorkerLoadResponse::from),
            Self::TokenSpeed(client) => client
                .get_loads(include)
                .await
                .map(WorkerLoadResponse::from),
        }
    }
}

/// Installs the process TLS crypto provider (ring) when none is installed
/// yet: the HTTP client below and the listeners' TLS both take the process
/// one, and whichever comes first installs it. The gateway, the servicer and
/// the mock worker install the same provider.
pub fn install_crypto_provider() {
    let _ = ring::default_provider().install_default();
}

/// The crate's HTTP client factory: the provider first, then the client,
/// whose timeouts are per call.
pub fn http_client() -> Result<reqwest::Client, reqwest::Error> {
    install_crypto_provider();
    #[expect(
        clippy::disallowed_methods,
        reason = "the crate's client factory: the process TLS crypto provider is installed on the line above"
    )]
    reqwest::Client::builder().build()
}

/// The HTTP client's error as a failed read. A timeout is the deadline's,
/// whether the headers never came or the body stalled past it. Anything else
/// is the transport's and carries the error with the request's URL off (the
/// client's own text ends with it, and no log line gets an address whole)
/// and the cause kept, as the gateway writes the same error.
fn failed(error: reqwest::Error, timeout: Duration) -> ReadError {
    if error.is_timeout() {
        ReadError::Deadline(timeout)
    } else {
        ReadError::Transport(format!("{:?}", error.without_url()))
    }
}

/// A refused `GetLoads` as a failed read. A failure below gRPC and an engine
/// that could not be reached (the channel's `UNAVAILABLE` over a refused or
/// timed-out dial) are the transport's; any other status is the engine's
/// answer. The text is the status's own and names no address.
fn refused(status: &tonic::Status) -> ReadError {
    if is_transport_failure(status) || status.code() == Code::Unavailable {
        ReadError::Transport(status.message().to_owned())
    } else {
        ReadError::Answer(format!("{:?}: {}", status.code(), status.message()))
    }
}

impl Default for EngineProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineProbe {
    /// A probe with no connection and no HTTP client yet.
    pub fn new() -> Self {
        Self {
            clients: Mutex::new(HashMap::new()),
            http: OnceLock::new(),
        }
    }

    /// The scrape's client, built by the first scrape, once a process. A
    /// client that cannot be built is a fault of this host (its
    /// certificates, its TLS provider), not of an engine, and it does not
    /// pass by itself: where the client was built at the start, the
    /// controller refused to start. Built here, the failure writes one error
    /// line with its cause and is kept: every scrape after it is a transport
    /// failure with that cause, and none builds again.
    fn http(&self) -> Result<&reqwest::Client, ReadError> {
        self.http_built_by(http_client)
    }

    /// [`Self::http`] over the function that builds the client.
    fn http_built_by(
        &self,
        build: impl FnOnce() -> Result<reqwest::Client, reqwest::Error>,
    ) -> Result<&reqwest::Client, ReadError> {
        self.http
            .get_or_init(|| {
                build().map_err(|failure| {
                    let cause = format!("{failure:?}");
                    error!(
                        %cause,
                        "the scrape's HTTP client could not be built: scraping is off for the life of this process"
                    );
                    format!("no HTTP client: {cause}")
                })
            })
            .as_ref()
            .map_err(|kept| ReadError::Transport(kept.clone()))
    }

    /// The client of the target's endpoint, dialed on the first poll and
    /// kept until a failure drops it. The family is part of what is kept: an
    /// engine re-registered at one address under another kind is asked
    /// through its own family's client, never the one the old registration
    /// left (whose refusals are answers and would keep it for good).
    async fn client(&self, target: &Target) -> Result<Family, ReadError> {
        if let Some(client) = self.clients.lock().get(&target.endpoint) {
            if client.kind() == target.kind {
                return Ok(client.clone());
            }
        }
        // No client of this family at the endpoint: another family's goes
        // before the dial, so a failed dial leaves none behind.
        self.clients.lock().remove(&target.endpoint);
        let client = Family::connect(target).await?;
        self.clients
            .lock()
            .insert(target.endpoint.clone(), client.clone());
        Ok(client)
    }

    /// How many endpoints hold a connection.
    pub fn connections(&self) -> usize {
        self.clients.lock().len()
    }
}

impl LoadProbe for EngineProbe {
    async fn poll(
        &self,
        target: &Target,
        deadline: Duration,
    ) -> Result<WorkerLoadResponse, ReadError> {
        let call = async {
            let client = self.client(target).await?;
            client.get_loads().await.map_err(|status| refused(&status))
        };
        let outcome = match timeout(deadline, call).await {
            Ok(outcome) => outcome,
            Err(_elapsed) => Err(ReadError::Deadline(deadline)),
        };
        // A connection that failed or stalled is not asked again: the next
        // poll redials. An answer, even a refusal, keeps it.
        if matches!(
            outcome,
            Err(ReadError::Transport(_) | ReadError::Deadline(_))
        ) {
            self.forget(&target.endpoint);
        }
        outcome
    }

    async fn scrape(&self, url: &str, timeout: Duration) -> Result<String, ReadError> {
        let response = self
            .http()?
            .get(format!("{}/metrics", url.trim_end_matches('/')))
            .timeout(timeout)
            .send()
            .await
            .map_err(|error| failed(error, timeout))?;
        let status = response.status();
        if !status.is_success() {
            return Err(ReadError::Answer(format!("HTTP {status}")));
        }
        response
            .text()
            .await
            .map_err(|error| failed(error, timeout))
    }

    fn forget(&self, endpoint: &str) {
        self.clients.lock().remove(endpoint);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::ErrorKind,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use tokio::{net::TcpListener, task::JoinSet};

    use super::*;

    /// A metrics server that answers its headers at once and never finishes
    /// the body: the address to scrape, and the set whose task holds the
    /// socket open until the set is dropped.
    async fn stalled_body() -> (String, JoinSet<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a loopback port");
        let url = format!(
            "http://{}",
            listener.local_addr().expect("the bound address")
        );
        let mut held = JoinSet::new();
        held.spawn(async move {
            let (socket, _) = listener.accept().await.expect("the scrape's connection");
            // The request first (a readiness that finds nothing is asked
            // again), then the headers and a first part of the body.
            let mut request = [0_u8; 1024];
            loop {
                socket.readable().await.expect("the request");
                match socket.try_read(&mut request) {
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {}
                    read => {
                        read.expect("the request's bytes");
                        break;
                    }
                }
            }
            let mut answer: &[u8] = b"HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\ncontent-length: 64\r\n\r\nvllm:num_requests_running";
            while !answer.is_empty() {
                socket.writable().await.expect("a writable socket");
                match socket.try_write(answer) {
                    Ok(written) => answer = &answer[written..],
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {}
                    Err(error) => panic!("the answer: {error}"),
                }
            }
            // The rest of the body never comes.
            std::future::pending::<()>().await;
        });
        (url, held)
    }

    #[tokio::test]
    async fn a_body_that_stalls_past_the_timeout_is_a_deadline_like_headers_that_never_came() {
        let (url, _held) = stalled_body().await;
        let probe = EngineProbe::new();
        let timeout = Duration::from_millis(300);

        let read = probe.scrape(&url, timeout).await;

        assert!(
            matches!(read, Err(ReadError::Deadline(bound)) if bound == timeout),
            "{read:?}"
        );
    }

    /// A metrics server that takes the connection and never answers: the
    /// address to scrape, and the set whose task holds the socket open until
    /// the set is dropped.
    async fn silent() -> (String, JoinSet<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a loopback port");
        let url = format!(
            "http://{}",
            listener.local_addr().expect("the bound address")
        );
        let mut held = JoinSet::new();
        held.spawn(async move {
            let (_socket, _) = listener.accept().await.expect("the scrape's connection");
            std::future::pending::<()>().await;
        });
        (url, held)
    }

    #[tokio::test]
    async fn headers_that_never_come_within_the_timeout_are_a_deadline() {
        let (url, _held) = silent().await;
        let probe = EngineProbe::new();
        let timeout = Duration::from_millis(300);

        let read = probe.scrape(&url, timeout).await;

        assert!(
            matches!(read, Err(ReadError::Deadline(bound)) if bound == timeout),
            "{read:?}"
        );
    }

    #[tokio::test]
    async fn the_http_client_is_built_by_the_first_scrape_not_with_the_probe() {
        let probe = EngineProbe::new();
        assert!(probe.http.get().is_none(), "a new probe has no client");

        // A closed port: the scrape fails on the transport, and has built the
        // client to find that out.
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a loopback port");
        let url = format!(
            "http://{}",
            listener.local_addr().expect("the bound address")
        );
        drop(listener);
        let read = probe.scrape(&url, Duration::from_secs(5)).await;

        assert!(matches!(read, Err(ReadError::Transport(_))), "{read:?}");
        assert!(matches!(probe.http.get(), Some(Ok(_))), "the client, built");
    }

    /// A client that cannot be built is tried once: the failure is kept, and
    /// every scrape after it is a transport failure with the same cause that
    /// asks the builder nothing.
    #[tokio::test]
    async fn a_client_that_cannot_be_built_is_tried_once_and_its_failure_kept() {
        let probe = EngineProbe::new();
        let builds = AtomicUsize::new(0);
        // A user agent that is no header value: the builder refuses it.
        #[expect(
            clippy::disallowed_methods,
            reason = "a builder the test refuses: no client is built"
        )]
        let refused = || {
            builds.fetch_add(1, Ordering::Relaxed);
            reqwest::Client::builder().user_agent("no\nheader").build()
        };

        let first = probe.http_built_by(refused).map(|_client| ());
        let second = probe.http_built_by(refused).map(|_client| ());

        assert!(
            matches!(&first, Err(ReadError::Transport(text)) if text.starts_with("no HTTP client: ")),
            "{first:?}"
        );
        assert_eq!(
            format!("{first:?}"),
            format!("{second:?}"),
            "the kept failure"
        );
        assert_eq!(builds.load(Ordering::Relaxed), 1, "built once");

        // A scrape answers the kept failure: it reaches no builder and no
        // engine.
        let read = probe
            .scrape("http://127.0.0.1:9", Duration::from_secs(5))
            .await;
        assert!(
            matches!(&read, Err(ReadError::Transport(text)) if text.starts_with("no HTTP client: ")),
            "{read:?}"
        );
    }

    #[test]
    fn a_refused_call_is_written_in_the_status_own_words() {
        let unreachable = refused(&tonic::Status::unavailable("tcp connect error"));
        assert!(
            matches!(&unreachable, ReadError::Transport(text) if text == "tcp connect error"),
            "{unreachable:?}"
        );
        let answered = refused(&tonic::Status::unimplemented("no such method"));
        assert!(
            matches!(&answered, ReadError::Answer(text) if text == "Unimplemented: no such method"),
            "{answered:?}"
        );
    }
}
