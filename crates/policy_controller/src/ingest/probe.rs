//! The probe the controller dials the engines with: one gRPC client per
//! engine family (the client crate's public clients, the same `GetLoads`
//! the gateway asks, answers normalised by the same conversions), kept per
//! endpoint and family across polls and dropped on a transport failure so
//! the next poll redials, and one HTTP client for the scrape, built by
//! the first scrape: a controller that scrapes no engine builds none.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use openai_protocol::worker::{RuntimeType, WorkerLoadResponse};
use parking_lot::Mutex;
use rustls::crypto::ring;
use smg_grpc_client::{
    engine_load::LOAD_SECTIONS, is_transport_failure, SglangSchedulerClient,
    TokenSpeedSchedulerClient, TrtllmServiceClient, VllmEngineClient,
};
use tokio::{
    sync::OnceCell,
    time::{timeout, Instant},
};
use tonic::Code;
use tracing::warn;

use super::loads::{LoadProbe, ReadError, Target};

/// The engines' probe.
pub struct EngineProbe {
    clients: Mutex<HashMap<String, Family>>,
    /// The scrape's HTTP client, built by the first scrape and kept. Its
    /// build reads the platform's root certificates from file: a start
    /// does not pay for them, and a controller that scrapes no engine
    /// never does.
    http: OnceCell<reqwest::Client>,
    /// What builds that client: the crate's factory.
    build: fn() -> Result<reqwest::Client, reqwest::Error>,
    /// Whether a build that failed was said: the first one is, once.
    said: AtomicBool,
}

impl Default for EngineProbe {
    fn default() -> Self {
        Self::building(http_client)
    }
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

/// A client's build that fails, for the cases of a probe whose client
/// cannot be built: a user agent no header takes.
#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "a build that fails stands in for the crate's client factory"
)]
pub(crate) fn unbuildable() -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder().user_agent("\n").build()
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

impl EngineProbe {
    /// A probe with no connection yet, and no HTTP client: the first
    /// scrape builds it.
    pub fn new() -> Self {
        Self::default()
    }

    /// A probe whose HTTP client `build` makes.
    pub(crate) fn building(build: fn() -> Result<reqwest::Client, reqwest::Error>) -> Self {
        Self {
            clients: Mutex::default(),
            http: OnceCell::new(),
            build,
            said: AtomicBool::new(false),
        }
    }

    /// The scrape's HTTP client, built at the first call and kept. The
    /// build reads files, so it runs off the runtime's workers, and its
    /// time is the first scrape's own. A build that fails is that
    /// scrape's transport failure, and the next scrape builds again. The
    /// first such failure is said once, as a warning: it is the
    /// controller's own, where the driver's count reads it as an engine's.
    async fn http(&self) -> Result<&reqwest::Client, ReadError> {
        let built = || async {
            match tokio::task::spawn_blocking(self.build).await {
                Ok(Ok(client)) => Ok(client),
                Ok(Err(error)) => {
                    let failed = format!("{error:?}");
                    if !self.said.swap(true, Ordering::Relaxed) {
                        warn!(
                            error = %failed,
                            "the engine probe's HTTP client could not be built: no engine is scraped until a later scrape builds it"
                        );
                    }
                    Err(ReadError::Transport(failed))
                }
                Err(join) => Err(ReadError::Transport(join.to_string())),
            }
        };
        self.http.get_or_try_init(built).await
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

    /// Whether the scrape's HTTP client is built: the first scrape builds
    /// it.
    pub fn http_built(&self) -> bool {
        self.http.initialized()
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
            // A failure below gRPC and an engine that could not be reached
            // (the channel's `UNAVAILABLE` over a refused or timed-out dial)
            // are the transport's; any other status is the engine's answer.
            client.get_loads().await.map_err(|status| {
                if is_transport_failure(&status) || status.code() == Code::Unavailable {
                    ReadError::Transport(status.message().to_owned())
                } else {
                    ReadError::Answer(format!("{:?}: {}", status.code(), status.message()))
                }
            })
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
        // The client's build is the first scrape's own time: the request
        // has what the build left of the timeout, so the scrape ends by
        // its deadline. A build that outlasted the timeout sends nothing.
        let asked = Instant::now();
        let http = self.http().await?;
        let left = timeout.saturating_sub(asked.elapsed());
        if left.is_zero() {
            return Err(ReadError::Deadline(timeout));
        }
        let response = http
            .get(format!("{}/metrics", url.trim_end_matches('/')))
            .timeout(left)
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
    use std::{io::ErrorKind, net, sync::atomic::AtomicUsize, thread};

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

    /// The scrape's HTTP client is built by the first scrape and kept: a
    /// probe that has scraped nothing holds none, so a controller with no
    /// engine to scrape reads no root certificate of the platform.
    #[tokio::test]
    async fn the_http_client_is_built_by_the_first_scrape() {
        let (url, _held) = stalled_body().await;
        let probe = EngineProbe::new();
        assert!(!probe.http_built(), "no scrape yet");

        let read = probe.scrape(&url, Duration::from_millis(50)).await;

        assert!(matches!(read, Err(ReadError::Deadline(_))), "{read:?}");
        assert!(probe.http_built(), "the first scrape builds it");
    }

    /// A client that cannot be built fails its scrape as the transport's
    /// and holds none, and the next scrape builds again: a controller
    /// whose platform gains its certificates later scrapes from then on.
    #[tokio::test]
    async fn a_client_that_cannot_be_built_is_built_again_by_the_next_scrape() {
        static BUILDS: AtomicUsize = AtomicUsize::new(0);
        fn broken_once() -> Result<reqwest::Client, reqwest::Error> {
            if BUILDS.fetch_add(1, Ordering::Relaxed) == 0 {
                unbuildable()
            } else {
                http_client()
            }
        }
        let (url, _held) = stalled_body().await;
        let probe = EngineProbe::building(broken_once);

        let first = probe.scrape(&url, Duration::from_millis(50)).await;
        assert!(matches!(first, Err(ReadError::Transport(_))), "{first:?}");
        assert!(!probe.http_built(), "a build that fails holds no client");

        let second = probe.scrape(&url, Duration::from_millis(50)).await;
        assert!(matches!(second, Err(ReadError::Deadline(_))), "{second:?}");
        assert!(probe.http_built(), "the next scrape builds it");
    }

    /// The build's time is the first scrape's own: a build that outlasts
    /// the timeout fails that scrape by its deadline with nothing sent,
    /// and the client is kept for the next.
    #[tokio::test]
    async fn a_build_that_outlasts_the_timeout_fails_its_scrape_unsent() {
        fn slow() -> Result<reqwest::Client, reqwest::Error> {
            thread::sleep(Duration::from_millis(60));
            http_client()
        }
        let listener = net::TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        listener
            .set_nonblocking(true)
            .expect("a listener that does not wait");
        let url = format!(
            "http://{}",
            listener.local_addr().expect("the bound address")
        );
        let probe = EngineProbe::building(slow);
        let timeout = Duration::from_millis(20);

        let read = probe.scrape(&url, timeout).await;

        assert!(
            matches!(read, Err(ReadError::Deadline(bound)) if bound == timeout),
            "{read:?}"
        );
        assert!(probe.http_built(), "the build is kept");
        let came = listener.accept();
        assert!(
            matches!(&came, Err(error) if error.kind() == ErrorKind::WouldBlock),
            "the scrape sent nothing: {came:?}"
        );
    }
}
