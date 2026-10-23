//! Consul worker discovery: the instances of one Consul service whose health
//! checks all pass.
//!
//! The provider asks Consul's health API for them,
//! `GET /v1/health/service/<service>?passing=true`, as a blocking query:
//! Consul holds the request until the service's instances or their checks
//! change, or for up to [`WAIT`], then answers with its full current list and
//! an `X-Consul-Index` the next request waits past. A change is applied as
//! soon as Consul reports it. The check interval reconciles the last good list
//! again in between, retrying what an earlier pass left undone, such as an
//! engine that was not yet listening when its instance first passed; it never
//! interrupts a query in flight.
//!
//! An index that goes backwards (a Consul restored or rebuilt) is read afresh
//! from index 0, and an index of 0 is waited past as 1, as Consul's
//! documentation advises. A query that fails is retried after a delay that
//! doubles up to [`RETRY_MAX`], from index 0 again so a restarted Consul is
//! read at once, and leaves the last good list in place. Requests start at
//! most once every [`MIN_GAP`], however fast the index moves.
//!
//! A worker listens at its service's address, or its node's when the service
//! registers none, on the service's port. Its id is the datacenter, the node
//! and the service ID, since a service ID is unique only on its node. The
//! instance's service metadata configures it:
//!
//! | `Service.Meta` key | Setting |
//! |---|---|
//! | `smg_worker_type` | `regular`, `encode`, `prefill` or `decode`; required in disaggregated routing, `regular` otherwise |
//! | `smg_bootstrap_port` | the bootstrap port of an encode or prefill worker |
//! | `smg_model_id` | the model id, instead of the one the worker reports |
//! | `smg_kv_connector`, `smg_kv_role`, `smg_kv_engine_id` | the KV-transfer settings |
//!
//! An empty value counts as unset. Any other `smg_` key is refused, so a
//! misspelled one is seen rather than ignored; keys without the prefix are
//! other tools' and ignored. Changing an instance's metadata updates its
//! worker in place.
//!
//! Each instance stands or falls alone: one that cannot be a worker here (no
//! address or port, metadata that does not parse or does not fit the routing
//! mode) is left out and the rest of the answer applies. An address two
//! instances claim stays with the one registered first (the lower
//! `CreateIndex`), since only one server listens there, and is left out for
//! the other. What an answer leaves out is counted on
//! `smg_discovery_skipped_total` and logged when it first appears. A query
//! that fails, or an answer that does not parse, applies nothing.
//!
//! Consul leaves out of an answer whatever the ACL token cannot read. An
//! answer it marks as filtered (`X-Consul-Results-Filtered-By-ACLs`) is
//! rejected rather than applied, since instances it holds are missing from
//! it. One it does not mark is applied as it stands: a token that cannot read
//! the service at all gets an empty answer, like a service with no instances.
//! An empty answer from a Consul that denies by default, to a gateway with no
//! token, is the likeliest case, and is warned about while it lasts; it is
//! still applied, because the agent's default token may be what reads the
//! service. The token needs `service:read` on the service and `node:read` on
//! its nodes.
//!
//! Configuration follows Consul's own clients. The address is
//! `--consul-address`, else `CONSUL_HTTP_ADDR`, else `127.0.0.1:8500`. It is
//! spoken to over HTTP unless it names `https`, or `CONSUL_HTTP_SSL` is true:
//! once that asks for TLS, an address that names `http` is spoken to over
//! HTTPS too, as Consul's clients never fall back to plaintext.
//! The token is read from the file `CONSUL_HTTP_TOKEN_FILE` names, else
//! `CONSUL_HTTP_TOKEN`, and never taken as a flag. Over HTTPS, `CONSUL_CACERT`
//! adds a CA bundle and `CONSUL_CLIENT_CERT` with `CONSUL_CLIENT_KEY` a client
//! certificate. Settings this provider does not implement (`CONSUL_HTTP_AUTH`,
//! `CONSUL_CAPATH`, `CONSUL_TLS_SERVER_NAME`, `CONSUL_NAMESPACE`,
//! `CONSUL_PARTITION`, and `CONSUL_HTTP_SSL_VERIFY=false`) refuse to start
//! rather than be ignored.

use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    future::Future,
    pin::Pin,
    sync::Arc,
    time::Duration,
};

use openai_protocol::worker::WorkerType;
use reqwest::{header::HeaderMap, StatusCode, Url};
use serde::Deserialize;
use tokio::{task, time};
use tracing::{info, warn};

use super::{
    provider::{endpoint_at, DiscoveredWorker, DiscoveryKind, Roles},
    reconciler::DesiredState,
    snapshot::{LastGood, Skip, SkipLog, SnapshotError},
};
use crate::{
    app_context::AppContext,
    config::{ConfigError, ConfigResult, ConsulDiscoveryConfig, RoutingMode},
    observability::metrics::metrics_labels,
    worker::{http_client::build_client, EndpointKey},
};

/// How long Consul may hold a blocking query before answering anyway.
const WAIT: Duration = Duration::from_secs(300);

/// How long a request may take beyond the time Consul may hold it.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// The least time between the starts of two requests.
const MIN_GAP: Duration = Duration::from_secs(1);

/// The delay before the first retry of a failed query, which doubles with
/// each failure in a row up to [`RETRY_MAX`].
const RETRY_MIN: Duration = Duration::from_secs(1);
const RETRY_MAX: Duration = Duration::from_secs(30);

/// Consul's address when neither the flag nor `CONSUL_HTTP_ADDR` gives one.
/// It names no scheme, so `CONSUL_HTTP_SSL` decides it, as it does for
/// Consul's clients.
const DEFAULT_ADDRESS: &str = "127.0.0.1:8500";

/// The prefix of the service metadata keys this provider reads.
const METADATA_PREFIX: &str = "smg_";

/// How much of an error answer's body an error repeats.
const BODY_EXCERPT: usize = 200;

/// Consul's environment variables whose meaning this provider does not
/// implement: each would change what is read or how it is trusted, so one
/// that is set refuses to start rather than be ignored.
const UNSUPPORTED_ENV: [(&str, &str); 5] = [
    (
        "CONSUL_HTTP_AUTH",
        "HTTP basic authentication is not supported; give an ACL token in \
         CONSUL_HTTP_TOKEN or CONSUL_HTTP_TOKEN_FILE",
    ),
    (
        "CONSUL_CAPATH",
        "a CA directory is not supported; give a CA bundle in CONSUL_CACERT",
    ),
    (
        "CONSUL_TLS_SERVER_NAME",
        "overriding the TLS server name is not supported; address Consul by a name its \
         certificate holds",
    ),
    (
        "CONSUL_NAMESPACE",
        "Consul namespaces are not supported; the default namespace is read",
    ),
    (
        "CONSUL_PARTITION",
        "Consul admin partitions are not supported; the default partition is read",
    ),
];

/// Consul discovery as the running gateway needs it.
#[derive(Debug, Clone)]
pub struct ConsulProviderConfig {
    /// The service's health endpoint, `<address>/v1/health/service/<service>`.
    url: Url,
    service: String,
    datacenter: Option<String>,
    token: Option<Token>,
    tls: Tls,
    roles: Roles,
    /// Interval between reconciles of the last good list.
    pub check_interval: Duration,
    /// How long Consul may hold a query: [`WAIT`], shorter in tests.
    wait: Duration,
}

/// An ACL token, kept out of `Debug` output and errors.
#[derive(Clone)]
struct Token(String);

impl fmt::Debug for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Token(..)")
    }
}

/// What an HTTPS connection to Consul trusts and presents, as PEM.
#[derive(Clone, Default)]
struct Tls {
    ca_bundle: Option<Vec<u8>>,
    /// The client certificate followed by its key.
    identity: Option<Vec<u8>>,
}

impl fmt::Debug for Tls {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tls")
            .field("ca_bundle", &self.ca_bundle.is_some())
            .field("identity", &self.identity.is_some())
            .finish()
    }
}

impl ConsulProviderConfig {
    /// Resolve the configuration with the process environment. A mode with no
    /// worker pools to fill is refused, as is a setting this provider cannot
    /// honor.
    pub fn new(config: &ConsulDiscoveryConfig, mode: &RoutingMode) -> ConfigResult<Self> {
        Self::with_env(config, mode, |name| std::env::var(name).ok())
    }

    /// [`Self::new`] with `env` as the environment.
    fn with_env(
        config: &ConsulDiscoveryConfig,
        mode: &RoutingMode,
        env: impl Fn(&str) -> Option<String>,
    ) -> ConfigResult<Self> {
        let roles = Roles::of(mode).ok_or_else(|| ConfigError::ValidationFailed {
            reason: "Consul discovery needs regular, prefill-decode or encode-prefill-decode \
                     routing"
                .to_string(),
        })?;
        // Consul's clients treat an empty variable as unset.
        let env = |name: &str| env(name).filter(|value| !value.is_empty());
        let refuse = |name: &str, value: String, reason: &str| ConfigError::InvalidValue {
            field: name.to_string(),
            value,
            reason: reason.to_string(),
        };

        // Named, not repeated: `CONSUL_HTTP_AUTH` holds a password.
        for (name, reason) in UNSUPPORTED_ENV {
            if env(name).is_some() {
                return Err(refuse(name, "(set)".to_string(), reason));
            }
        }
        let flag = |name: &str| {
            env(name)
                .map(|value| {
                    parse_bool(&value).ok_or_else(|| refuse(name, value, "Must be true or false"))
                })
                .transpose()
        };
        if flag("CONSUL_HTTP_SSL_VERIFY")? == Some(false) {
            return Err(refuse(
                "CONSUL_HTTP_SSL_VERIFY",
                "false".to_string(),
                "Consul's certificate is always verified; give its CA in CONSUL_CACERT",
            ));
        }
        let https = flag("CONSUL_HTTP_SSL")?.unwrap_or(false);

        let (address, source) = match (&config.address, env("CONSUL_HTTP_ADDR")) {
            (Some(address), _) => (address.clone(), "discovery.address"),
            (None, Some(address)) => (address, "CONSUL_HTTP_ADDR"),
            (None, None) => (DEFAULT_ADDRESS.to_string(), "the default Consul address"),
        };
        // An address with credentials is refused, and never repeated.
        let shown = if address.contains('@') {
            "(an address with credentials)".to_string()
        } else {
            address.clone()
        };
        let mut url = parse_address(&address, https)
            .map_err(|reason| refuse(source, shown.clone(), &reason))?;
        // An http or https URL always has path segments.
        url.path_segments_mut()
            .map_err(|()| refuse(source, shown, "Must be an http or https URL"))?
            .pop_if_empty()
            .extend(["v1", "health", "service", &config.service]);

        let token = match env("CONSUL_HTTP_TOKEN_FILE") {
            Some(path) => {
                let read = std::fs::read_to_string(&path).map_err(|error| {
                    refuse(
                        "CONSUL_HTTP_TOKEN_FILE",
                        path.clone(),
                        &format!("Cannot be read: {error}"),
                    )
                })?;
                Some(read.trim().to_string()).filter(|token| !token.is_empty())
            }
            None => env("CONSUL_HTTP_TOKEN"),
        };

        let tls = if url.scheme() == "https" {
            let read = |name: &str| {
                env(name)
                    .map(|path| {
                        std::fs::read(&path).map_err(|error| {
                            refuse(name, path, &format!("Cannot be read: {error}"))
                        })
                    })
                    .transpose()
            };
            let identity = match (read("CONSUL_CLIENT_CERT")?, read("CONSUL_CLIENT_KEY")?) {
                (None, None) => None,
                (Some(mut cert), Some(key)) => {
                    // rustls reads the certificate, then its key, from one PEM.
                    if !cert.ends_with(b"\n") {
                        cert.push(b'\n');
                    }
                    cert.extend_from_slice(&key);
                    Some(cert)
                }
                (cert, _) => {
                    let (given, missing) = if cert.is_some() {
                        ("CONSUL_CLIENT_CERT", "CONSUL_CLIENT_KEY")
                    } else {
                        ("CONSUL_CLIENT_KEY", "CONSUL_CLIENT_CERT")
                    };
                    return Err(refuse(
                        given,
                        String::new(),
                        &format!("Needs {missing} as well"),
                    ));
                }
            };
            Tls {
                ca_bundle: read("CONSUL_CACERT")?,
                identity,
            }
        } else {
            Tls::default()
        };

        let config = Self {
            url,
            service: config.service.clone(),
            datacenter: config.datacenter.clone(),
            token: token.map(Token),
            tls,
            roles,
            check_interval: Duration::from_secs(config.check_interval_secs),
            wait: WAIT,
        };
        // Building the client now reports bad certificates at startup.
        config
            .client()
            .map_err(|reason| ConfigError::ValidationFailed {
                reason: format!("Consul discovery: {reason}"),
            })?;
        Ok(config)
    }

    /// The HTTP client of every query.
    fn client(&self) -> Result<reqwest::Client, String> {
        let https = self.url.scheme() == "https";
        let ca_certificates = match &self.tls.ca_bundle {
            Some(pem) => {
                let certificates = reqwest::Certificate::from_pem_bundle(pem)
                    .map_err(|error| format!("CONSUL_CACERT does not parse: {error}"))?;
                if certificates.is_empty() {
                    return Err("CONSUL_CACERT holds no certificate".to_string());
                }
                certificates
            }
            None => Vec::new(),
        };
        let identity = self
            .tls
            .identity
            .as_deref()
            .map(reqwest::Identity::from_pem)
            .transpose()
            .map_err(|error| {
                format!("CONSUL_CLIENT_CERT and CONSUL_CLIENT_KEY do not parse: {error}")
            })?;
        let make_builder = || {
            let mut builder = crate::clients::http_client_builder()
                .connect_timeout(Duration::from_secs(10))
                .redirect(reqwest::redirect::Policy::none());
            if https {
                builder = builder.use_rustls_tls();
                for certificate in &ca_certificates {
                    builder = builder.add_root_certificate(certificate.clone());
                }
                if let Some(identity) = &identity {
                    builder = builder.identity(identity.clone());
                }
            }
            builder
        };
        build_client(make_builder, https, "Consul discovery client")
    }

    /// The health query of the service past `index`, which blocks unless it
    /// is 0.
    fn request(&self, client: &reqwest::Client, index: u64) -> reqwest::RequestBuilder {
        let mut url = self.url.clone();
        {
            let mut query = url.query_pairs_mut();
            query.append_pair("passing", "true");
            if let Some(datacenter) = &self.datacenter {
                query.append_pair("dc", datacenter);
            }
            if index > 0 {
                query.append_pair("index", &index.to_string());
                query.append_pair("wait", &format!("{}s", self.wait.as_secs()));
            }
        }
        // Consul adds up to a sixteenth of the wait, so that many waiting
        // clients do not all return at once.
        let timeout = if index > 0 {
            self.wait + self.wait / 16 + REQUEST_TIMEOUT
        } else {
            REQUEST_TIMEOUT
        };
        let request = client.get(url).timeout(timeout);
        match &self.token {
            Some(Token(token)) => request.header("X-Consul-Token", token),
            None => request,
        }
    }

    /// The service and its address, as logs name them.
    fn describe(&self) -> String {
        let mut address = self.url.clone();
        address.set_path("");
        format!("Consul service {} at {address}", self.service)
    }
}

/// An address as Consul's clients read it: a URL, or a `host:port` that HTTP
/// is spoken to. With `https`, HTTPS is spoken to an `http` URL too, on the
/// port it names. It names no path, query or credentials.
fn parse_address(address: &str, https: bool) -> Result<Url, String> {
    if address.starts_with("unix://") {
        return Err("A Unix socket address is not supported; give an http or https URL".into());
    }
    let mut url = if address.contains("://") {
        Url::parse(address)
    } else {
        let scheme = if https { "https" } else { "http" };
        Url::parse(&format!("{scheme}://{address}"))
    }
    .map_err(|error| format!("Not an address: {error}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(format!(
            "Must be an http or https URL, not {}",
            url.scheme()
        ));
    }
    if url.host_str().is_none_or(str::is_empty) {
        return Err("Names no host".into());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("Credentials in the address are not supported".into());
    }
    if !matches!(url.path(), "" | "/") || url.query().is_some() || url.fragment().is_some() {
        return Err("Must name only a scheme, a host and a port".into());
    }
    // Consul's clients never fall back to HTTP once TLS is asked for.
    if https && url.scheme() == "http" {
        let port = url.port_or_known_default();
        url.set_scheme("https")
            .and_then(|()| url.set_port(port))
            .map_err(|()| "Cannot be spoken to over HTTPS".to_string())?;
    }
    Ok(url)
}

/// A boolean as Go's `strconv.ParseBool` reads one, which is how Consul's
/// clients read their flags.
fn parse_bool(value: &str) -> Option<bool> {
    match value {
        "1" | "t" | "T" | "true" | "TRUE" | "True" => Some(true),
        "0" | "f" | "F" | "false" | "FALSE" | "False" => Some(false),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Consul's answer, and the workers it holds.
// ---------------------------------------------------------------------------

/// One answer of the health endpoint.
#[derive(Debug)]
struct Answer {
    /// `X-Consul-Index`: the next query waits for a change past it.
    index: u64,
    entries: Vec<Entry>,
    /// Consul left out what the token cannot read, and said so.
    filtered: bool,
    /// Consul denies by default whatever no token allows.
    default_deny: bool,
}

/// An instance of the service: its node, and the service as registered there.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Entry {
    node: EntryNode,
    service: EntryService,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct EntryNode {
    node: String,
    #[serde(default)]
    address: String,
    #[serde(default)]
    datacenter: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct EntryService {
    #[serde(rename = "ID")]
    id: String,
    #[serde(default)]
    address: String,
    #[serde(default)]
    port: i64,
    /// `null` for a service registered with none.
    #[serde(default)]
    meta: Option<BTreeMap<String, String>>,
    /// The Raft index the registration was created at: the lower, the older.
    #[serde(default)]
    create_index: u64,
}

impl Entry {
    /// The worker's id: a service ID is unique only on its node.
    fn id(&self) -> String {
        format!(
            "{}/{}/{}",
            self.node.datacenter, self.node.node, self.service.id
        )
    }

    /// The worker this instance is, or why it cannot be one in `roles`.
    fn worker(&self, roles: Roles) -> Result<DiscoveredWorker, String> {
        let host = [&self.service.address, &self.node.address]
            .into_iter()
            .find(|host| !host.is_empty())
            .ok_or_else(|| {
                format!(
                    "neither the service nor its node {} has an address",
                    self.node.node
                )
            })?;
        let port = u16::try_from(self.service.port)
            .ok()
            .filter(|port| *port != 0)
            .ok_or_else(|| format!("port {} is not a port", self.service.port))?;
        let endpoint = endpoint_at(host, port)
            .map_err(|error| format!("no worker can be reached at {host}: {error}"))?;
        let meta = Metadata::read(self.service.meta.as_ref())?;

        let worker_type = match meta.worker_type {
            Some(worker_type) if roles.allows(worker_type) => worker_type,
            Some(worker_type) => {
                return Err(format!(
                    "smg_worker_type '{worker_type}' is not used in {} routing",
                    roles.name()
                ))
            }
            None => roles.default_type().ok_or_else(|| {
                format!("smg_worker_type is required in {} routing", roles.name())
            })?,
        };
        if meta.bootstrap_port.is_some()
            && !matches!(worker_type, WorkerType::Encode | WorkerType::Prefill)
        {
            return Err(format!(
                "smg_bootstrap_port applies only to encode and prefill workers, not {worker_type}"
            ));
        }

        Ok(DiscoveredWorker {
            discovery_id: self.id(),
            endpoint,
            worker_type,
            bootstrap_port: meta.bootstrap_port,
            model_id_override: meta.model_id,
            kv_connector: meta.kv_connector,
            kv_role: meta.kv_role,
            kv_engine_id: meta.kv_engine_id,
            compat_labels: BTreeMap::new(),
        })
    }
}

/// The worker settings in an instance's `smg_` metadata.
#[derive(Debug, Default)]
struct Metadata {
    worker_type: Option<WorkerType>,
    bootstrap_port: Option<u16>,
    model_id: Option<String>,
    kv_connector: Option<String>,
    kv_role: Option<String>,
    kv_engine_id: Option<String>,
}

impl Metadata {
    fn read(meta: Option<&BTreeMap<String, String>>) -> Result<Self, String> {
        let mut read = Self::default();
        for (key, value) in meta.into_iter().flatten() {
            let Some(name) = key.strip_prefix(METADATA_PREFIX) else {
                continue;
            };
            let value = Some(value).filter(|value| !value.is_empty());
            match name {
                "worker_type" => {
                    read.worker_type = value
                        .map(|value| {
                            value.parse().map_err(|_| {
                                format!(
                                    "smg_worker_type '{value}' is not regular, encode, prefill \
                                     or decode"
                                )
                            })
                        })
                        .transpose()?;
                }
                "bootstrap_port" => {
                    read.bootstrap_port = value
                        .map(|value| {
                            value
                                .parse::<u16>()
                                .ok()
                                .filter(|port| *port != 0)
                                .ok_or_else(|| {
                                    format!("smg_bootstrap_port '{value}' is not a port")
                                })
                        })
                        .transpose()?;
                }
                "model_id" => read.model_id = value.cloned(),
                "kv_connector" => read.kv_connector = value.cloned(),
                "kv_role" => read.kv_role = value.cloned(),
                "kv_engine_id" => read.kv_engine_id = value.cloned(),
                _ => return Err(format!("unknown metadata key {key}")),
            }
        }
        Ok(read)
    }
}

/// What an answer applies: the workers to publish, and what it left out.
#[derive(Debug, Default)]
struct Outcome {
    workers: Vec<DiscoveredWorker>,
    skipped: Vec<Skip>,
}

/// The workers of an answer, or why the answer is not applied.
///
/// One instance that cannot be a worker is left out alone. An address two
/// instances claim stays with the one registered first (the lower
/// `CreateIndex`, then the lower id), and is left out for the others.
fn read(answer: Answer, config: &ConsulProviderConfig) -> Result<Outcome, ConsulError> {
    if answer.filtered {
        return Err(ConsulError::Invalid {
            reason: format!(
                "Consul left out instances of {} the ACL token cannot read; it needs \
                 service:read on the service and node:read on its nodes",
                config.service
            ),
        });
    }
    let mut skipped = Vec::new();
    let mut placed = Vec::new();
    for entry in &answer.entries {
        match entry.worker(config.roles) {
            Ok(worker) => placed.push((entry.service.create_index, worker)),
            Err(why) => skipped.push(Skip {
                reason: metrics_labels::DISCOVERY_SKIP_INVALID,
                message: format!("instance {}: {why}", entry.id()),
            }),
        }
    }

    // Two instances at one address: only one server listens there, most
    // likely the one registered first.
    let mut holders: HashMap<EndpointKey, (u64, String)> = HashMap::new();
    for (created, worker) in &placed {
        let claim = (*created, worker.discovery_id.clone());
        holders
            .entry(worker.endpoint.key())
            .and_modify(|holder| {
                if claim < *holder {
                    *holder = claim.clone();
                }
            })
            .or_insert(claim);
    }
    let mut workers = Vec::new();
    for (_, worker) in placed {
        let (_, holder) = &holders[&worker.endpoint.key()];
        if *holder == worker.discovery_id {
            workers.push(worker);
            continue;
        }
        skipped.push(Skip {
            reason: metrics_labels::DISCOVERY_SKIP_CONFLICT,
            message: format!(
                "instance {}: {} is {holder}'s, registered first",
                worker.discovery_id,
                worker.endpoint.redacted()
            ),
        });
    }
    Ok(Outcome { workers, skipped })
}

/// Why an answer produced no snapshot.
#[derive(Debug, thiserror::Error)]
enum ConsulError {
    /// The query failed, timed out or was refused.
    #[error("Consul query failed: {reason}")]
    Unreadable { reason: String },
    /// Consul answered, but with what cannot be applied.
    #[error("Consul answer rejected: {reason}")]
    Invalid { reason: String },
}

impl SnapshotError for ConsulError {
    fn metric_reason(&self) -> &'static str {
        match self {
            Self::Unreadable { .. } => metrics_labels::SNAPSHOT_ERROR_UNREADABLE,
            Self::Invalid { .. } => metrics_labels::SNAPSHOT_ERROR_INVALID,
        }
    }
}

/// An error with the causes `reqwest` keeps behind its top-level message.
fn describe(error: &reqwest::Error) -> String {
    let mut description = error.to_string();
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        description.push_str(": ");
        description.push_str(&cause.to_string());
        source = cause.source();
    }
    description
}

/// Send one query and read its answer.
async fn query(request: reqwest::RequestBuilder) -> Result<Answer, ConsulError> {
    let unreadable = |reason: String| ConsulError::Unreadable { reason };
    let response = request
        .send()
        .await
        .map_err(|error| unreadable(describe(&error)))?;
    let status = response.status();
    let headers = response.headers().clone();
    let body = response
        .bytes()
        .await
        .map_err(|error| unreadable(describe(&error)))?;
    if status != StatusCode::OK {
        let text = String::from_utf8_lossy(&body);
        let line = text.lines().map(str::trim).find(|line| !line.is_empty());
        let excerpt: String = line.unwrap_or("").chars().take(BODY_EXCERPT).collect();
        return Err(unreadable(format!("Consul answered {status}: {excerpt}")));
    }
    let invalid = |reason: String| ConsulError::Invalid { reason };
    let index = header(&headers, "x-consul-index")
        .and_then(|index| index.parse::<u64>().ok())
        .ok_or_else(|| invalid("the answer has no X-Consul-Index".to_string()))?;
    let entries = serde_json::from_slice(&body)
        .map_err(|error| invalid(format!("the answer does not parse: {error}")))?;
    Ok(Answer {
        index,
        entries,
        filtered: header(&headers, "x-consul-results-filtered-by-acls")
            .is_some_and(|value| value.eq_ignore_ascii_case("true")),
        default_deny: header(&headers, "x-consul-default-acl-policy")
            .is_some_and(|value| value.eq_ignore_ascii_case("deny")),
    })
}

fn header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name)?.to_str().ok()
}

/// The index the next query waits past, after one answered at `answered`
/// when the last waited past `last`. 0 would not block, so it counts as 1,
/// before anything else: a Consul that keeps answering 0 is then waited on,
/// not read afresh every time. One that went backwards is read afresh from 0.
fn next_index(last: u64, answered: u64) -> u64 {
    let answered = answered.max(1);
    if answered < last {
        0
    } else {
        answered
    }
}

// ---------------------------------------------------------------------------
// The provider's loop.
// ---------------------------------------------------------------------------

/// Query Consul continuously, applying each answer as it comes, and reconcile
/// the last good list on every answer and every tick.
pub(super) fn start_consul_discovery(
    config: ConsulProviderConfig,
    app_context: Arc<AppContext>,
) -> Result<task::JoinHandle<()>, String> {
    let client = config.client()?;
    info!(
        "Starting Consul service discovery | {}{} | apply changes as Consul reports them, \
         reconcile every {}s",
        config.describe(),
        config
            .datacenter
            .as_ref()
            .map(|datacenter| format!(" | datacenter: {datacenter}"))
            .unwrap_or_default(),
        config.check_interval.as_secs_f64()
    );

    #[expect(
        clippy::disallowed_methods,
        reason = "worker discovery runs for the lifetime of the server; shutdown aborts the handle"
    )]
    let handle = task::spawn(ConsulDiscovery::new(config, app_context, client).run());

    Ok(handle)
}

type Query = Pin<Box<dyn Future<Output = Result<Answer, ConsulError>> + Send>>;

/// The provider's loop and its state.
struct ConsulDiscovery {
    config: ConsulProviderConfig,
    app_context: Arc<AppContext>,
    client: reqwest::Client,
    last_good: LastGood,
    skipped: SkipLog,
    /// The index the next query waits past; 0 reads at once.
    index: u64,
    /// The delay before the next retry, should the next query fail too.
    retry: Duration,
    /// When the query in flight starts, or started.
    query_starts: time::Instant,
    /// Whether the empty answers of a default-deny Consul to no token, which
    /// may be hiding the instances, have been warned about.
    warned_tokenless: bool,
}

impl ConsulDiscovery {
    fn new(
        config: ConsulProviderConfig,
        app_context: Arc<AppContext>,
        client: reqwest::Client,
    ) -> Self {
        let last_good = LastGood::new(DiscoveryKind::Consul, config.describe(), "instance list");
        Self {
            config,
            app_context,
            client,
            last_good,
            skipped: SkipLog::new(DiscoveryKind::Consul, "Consul discovery"),
            index: 0,
            retry: RETRY_MIN,
            query_starts: time::Instant::now(),
            warned_tokenless: false,
        }
    }

    async fn run(mut self) {
        let mut tick = time::interval(self.config.check_interval);
        tick.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
        let mut query = self.next_query(Duration::ZERO);
        loop {
            tokio::select! {
                answer = &mut query => {
                    let started_at = time::Instant::now();
                    let delay = self.take(answer);
                    query = self.next_query(delay);
                    self.last_good.reconcile(&self.app_context, started_at).await;
                }
                _ = tick.tick() => {
                    let started_at = time::Instant::now();
                    self.last_good.reconcile(&self.app_context, started_at).await;
                }
            }
        }
    }

    /// The next query, started after `delay`. It owns everything it uses, so
    /// a reconcile that runs meanwhile never holds it up or restarts it.
    fn next_query(&mut self, delay: Duration) -> Query {
        self.query_starts = time::Instant::now() + delay;
        let request = self.config.request(&self.client, self.index);
        Box::pin(async move {
            time::sleep(delay).await;
            query(request).await
        })
    }

    /// Take an answer into the last good list, and say how long to wait
    /// before the next query.
    fn take(&mut self, answer: Result<Answer, ConsulError>) -> Duration {
        let read = answer.and_then(|answer| {
            let index = answer.index;
            let tokenless_deny =
                answer.entries.is_empty() && answer.default_deny && self.config.token.is_none();
            read(answer, &self.config).map(|outcome| (index, tokenless_deny, outcome))
        });
        match read {
            Ok((index, tokenless_deny, outcome)) => {
                if tokenless_deny && !self.warned_tokenless {
                    warn!(
                        "Consul answered no instance of {}, and it denies by default while no \
                         ACL token is set (CONSUL_HTTP_TOKEN or CONSUL_HTTP_TOKEN_FILE): unless \
                         the agent's default token reads the service, its instances are hidden",
                        self.config.service
                    );
                }
                self.warned_tokenless = tokenless_deny;
                let next = next_index(self.index, index);
                if next == 0 {
                    info!(
                        "Consul's index went back from {} to {index}; reading {} afresh",
                        self.index, self.config.service
                    );
                }
                self.index = next;
                self.retry = RETRY_MIN;
                self.skipped.record(outcome.skipped);
                self.last_good
                    .update::<ConsulError>(Ok(DesiredState::from_workers(outcome.workers)));
                (self.query_starts + MIN_GAP).saturating_duration_since(time::Instant::now())
            }
            Err(error) => {
                // Read afresh next time: a Consul that restarted may answer
                // with a lower index, which a query past the old one would
                // wait out in full before seeing.
                self.index = 0;
                let delay = self.retry;
                self.retry = (self.retry * 2).min(RETRY_MAX);
                self.last_good.update(Err(error));
                delay
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::VecDeque, sync::Mutex};

    use super::*;
    use crate::service_discovery::testing::{create_test_app_context, live_app_context};

    const PASSING_2_0_4: &str = include_str!("testdata/consul/health-passing-2.0.4.json");
    const PASSING_1_22_7: &str = include_str!("testdata/consul/health-passing-1.22.7.json");

    fn regular() -> RoutingMode {
        RoutingMode::Regular {
            worker_urls: vec![],
        }
    }

    fn pd() -> RoutingMode {
        RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        }
    }

    fn discovery(service: &str, address: Option<&str>) -> ConsulDiscoveryConfig {
        ConsulDiscoveryConfig {
            service: service.to_string(),
            address: address.map(str::to_string),
            datacenter: None,
            check_interval_secs: 30,
        }
    }

    /// The configuration of `address` (the flag), with `env` as the whole
    /// environment.
    fn resolve(address: Option<&str>, env: &[(&str, &str)]) -> ConfigResult<ConsulProviderConfig> {
        resolve_service("smg-worker", address, env)
    }

    fn resolve_service(
        service: &str,
        address: Option<&str>,
        env: &[(&str, &str)],
    ) -> ConfigResult<ConsulProviderConfig> {
        let env: HashMap<String, String> = env
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        ConsulProviderConfig::with_env(&discovery(service, address), &regular(), |name| {
            env.get(name).cloned()
        })
    }

    fn config_in(mode: &RoutingMode) -> ConsulProviderConfig {
        ConsulProviderConfig::with_env(&discovery("smg-worker", None), mode, |_| None).unwrap()
    }

    fn answer(entries: &str) -> Answer {
        Answer {
            index: 1,
            entries: serde_json::from_str(entries).unwrap(),
            filtered: false,
            default_deny: false,
        }
    }

    fn entry(
        node: &str,
        node_address: &str,
        id: &str,
        address: &str,
        port: i64,
        meta: &[(&str, &str)],
        created: u64,
    ) -> Entry {
        Entry {
            node: EntryNode {
                node: node.to_string(),
                address: node_address.to_string(),
                datacenter: "dc1".to_string(),
            },
            service: EntryService {
                id: id.to_string(),
                address: address.to_string(),
                port,
                meta: Some(
                    meta.iter()
                        .map(|(key, value)| (key.to_string(), value.to_string()))
                        .collect(),
                ),
                create_index: created,
            },
        }
    }

    const DECODE: &[(&str, &str)] = &[("smg_worker_type", "decode")];

    fn ids(outcome: &Outcome) -> Vec<&str> {
        let mut ids: Vec<&str> = outcome
            .workers
            .iter()
            .map(|worker| worker.discovery_id.as_str())
            .collect();
        ids.sort_unstable();
        ids
    }

    // ---- What the recordings say. ----

    /// Both releases answer alike and read alike: every passing instance is a
    /// worker of its type at its service's address, or its node's (IPv6
    /// bracketed), and two nodes' `worker-a` stay apart. In PD routing the
    /// instance with no worker type is left out.
    #[test]
    fn recorded_answers_read_into_workers() {
        for recorded in [PASSING_2_0_4, PASSING_1_22_7] {
            let outcome = read(answer(recorded), &config_in(&pd())).unwrap();
            let mut workers: Vec<(&str, String, WorkerType)> = outcome
                .workers
                .iter()
                .map(|worker| {
                    (
                        worker.discovery_id.as_str(),
                        worker.endpoint.render(),
                        worker.worker_type,
                    )
                })
                .collect();
            workers.sort_by(|a, b| a.0.cmp(b.0));
            assert_eq!(
                workers,
                [
                    (
                        "dc1/agent-node/decode-1",
                        "127.0.0.1:8001".to_string(),
                        WorkerType::Decode
                    ),
                    (
                        "dc1/agent-node/prefill-1",
                        "10.0.0.21:8000".to_string(),
                        WorkerType::Prefill
                    ),
                    (
                        "dc1/gpu-07/worker-a",
                        "10.0.1.7:8000".to_string(),
                        WorkerType::Decode
                    ),
                    (
                        "dc1/gpu-08/worker-a",
                        "[fd00::8]:8000".to_string(),
                        WorkerType::Decode
                    ),
                ]
            );
            let prefill = outcome
                .workers
                .iter()
                .find(|worker| worker.worker_type == WorkerType::Prefill)
                .unwrap();
            assert_eq!(prefill.bootstrap_port, Some(8998));
            assert_eq!(prefill.model_id_override.as_deref(), Some("Qwen/Qwen3-8B"));
            assert_eq!(prefill.kv_connector.as_deref(), Some("MooncakeConnector"));
            assert_eq!(prefill.kv_role.as_deref(), Some("kv_producer"));
            assert_eq!(prefill.kv_engine_id.as_deref(), Some("engine-1"));
            assert_eq!(
                outcome.skipped,
                [Skip {
                    reason: metrics_labels::DISCOVERY_SKIP_INVALID,
                    message: "instance dc1/gpu-09/worker-b: smg_worker_type is required in \
                              prefill-decode routing"
                        .to_string(),
                }]
            );
        }
    }

    /// In regular routing the untyped instance is a regular worker, and the
    /// typed ones are left out, each saying why.
    #[test]
    fn regular_routing_takes_the_untyped_instance_alone() {
        let outcome = read(answer(PASSING_2_0_4), &config_in(&regular())).unwrap();
        assert_eq!(ids(&outcome), ["dc1/gpu-09/worker-b"]);
        assert_eq!(outcome.workers[0].worker_type, WorkerType::Regular);
        assert_eq!(outcome.skipped.len(), 4, "{:?}", outcome.skipped);
        assert!(outcome
            .skipped
            .iter()
            .all(|skip| skip.message.ends_with("is not used in regular routing")));
    }

    // ---- One instance at a time. ----

    #[test]
    fn metadata_configures_the_worker_or_says_why_not() {
        let read = |pairs: &[(&str, &str)]| {
            let meta: BTreeMap<String, String> = pairs
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect();
            Metadata::read(Some(&meta))
        };
        // An empty value is unset, other tools' keys are ignored, and the
        // type is read in any case.
        let meta = read(&[
            ("smg_worker_type", "Prefill"),
            ("smg_model_id", ""),
            ("smg_kv_role", "kv_producer"),
            ("version", "3"),
        ])
        .unwrap();
        assert_eq!(meta.worker_type, Some(WorkerType::Prefill));
        assert_eq!(meta.model_id, None);
        assert_eq!(meta.kv_role.as_deref(), Some("kv_producer"));
        assert!(Metadata::read(None).unwrap().worker_type.is_none());

        for (pairs, why) in [
            (
                &[("smg_worker_typ", "prefill")][..],
                "unknown metadata key smg_worker_typ",
            ),
            (
                &[("smg_worker_type", "verifier")][..],
                "smg_worker_type 'verifier' is not regular, encode, prefill or decode",
            ),
            (
                &[("smg_bootstrap_port", "0")][..],
                "smg_bootstrap_port '0' is not a port",
            ),
            (
                &[("smg_bootstrap_port", "70000")][..],
                "smg_bootstrap_port '70000' is not a port",
            ),
        ] {
            assert_eq!(read(pairs).unwrap_err(), why, "{pairs:?}");
        }
    }

    #[test]
    fn an_instance_that_cannot_be_a_worker_says_why() {
        let with = |meta: &[(&str, &str)]| entry("n1", "10.0.0.1", "w", "", 8000, meta, 1);
        for (instance, why) in [
            (
                entry("n1", "", "w", "", 8000, DECODE, 1),
                "neither the service nor its node n1 has an address",
            ),
            (
                entry("n1", "10.0.0.1", "w", "", 0, DECODE, 1),
                "port 0 is not a port",
            ),
            (
                entry("n1", "10.0.0.1", "w", "", 70000, DECODE, 1),
                "port 70000 is not a port",
            ),
            (
                with(&[
                    ("smg_worker_type", "decode"),
                    ("smg_bootstrap_port", "8998"),
                ]),
                "smg_bootstrap_port applies only to encode and prefill workers, not decode",
            ),
            (
                with(&[("smg_worker_type", "encode")]),
                "smg_worker_type 'encode' is not used in prefill-decode routing",
            ),
        ] {
            assert_eq!(
                instance.worker(Roles::PrefillDecode).unwrap_err(),
                why,
                "{instance:?}"
            );
        }
        let unreachable = entry("n1", "10.0.0.1", "w", "10.0.0.1:9000", 8000, DECODE, 1);
        let why = unreachable.worker(Roles::PrefillDecode).unwrap_err();
        assert!(
            why.starts_with("no worker can be reached at 10.0.0.1:9000"),
            "{why}"
        );
    }

    /// One server listens at an address, so the instance registered first
    /// keeps it, and a later one is left out there; within one index, the
    /// lower id keeps it.
    #[test]
    fn an_address_two_instances_claim_stays_with_the_first_registered() {
        let config = config_in(&pd());
        let answer_of = |entries: Vec<Entry>| Answer {
            index: 1,
            entries,
            filtered: false,
            default_deny: false,
        };
        let outcome = read(
            answer_of(vec![
                entry("n2", "10.0.0.1", "late", "", 8000, DECODE, 9),
                entry("n1", "10.0.0.1", "early", "", 8000, DECODE, 4),
                entry("n3", "10.0.0.3", "own", "", 8000, DECODE, 5),
            ]),
            &config,
        )
        .unwrap();
        assert_eq!(ids(&outcome), ["dc1/n1/early", "dc1/n3/own"]);
        assert_eq!(
            outcome.skipped,
            [Skip {
                reason: metrics_labels::DISCOVERY_SKIP_CONFLICT,
                message: "instance dc1/n2/late: 10.0.0.1:8000 is dc1/n1/early's, registered \
                          first"
                    .to_string(),
            }]
        );

        let outcome = read(
            answer_of(vec![
                entry("n2", "10.0.0.1", "b", "", 8000, DECODE, 4),
                entry("n1", "10.0.0.1", "a", "", 8000, DECODE, 4),
            ]),
            &config,
        )
        .unwrap();
        assert_eq!(ids(&outcome), ["dc1/n1/a"]);
    }

    /// Consul empties an answer of what the token cannot read. An answer it
    /// says it filtered is rejected, whatever the token; any other answer is
    /// applied, an empty one as an empty fleet.
    #[test]
    fn an_answer_consul_says_it_filtered_is_rejected() {
        let config = config_in(&pd());
        let empty = |filtered, default_deny| Answer {
            index: 3,
            entries: vec![],
            filtered,
            default_deny,
        };
        let rejected = |answer, config: &ConsulProviderConfig| match read(answer, config) {
            Err(ConsulError::Invalid { reason }) => reason,
            other => panic!("expected a rejection, got {other:?}"),
        };
        let mut with_token = config.clone();
        with_token.token = Some(Token("t".to_string()));
        assert!(rejected(empty(true, false), &config).contains("node:read"));
        assert!(rejected(empty(true, true), &with_token).contains("node:read"));

        for (answer, config) in [
            (empty(false, true), &config),
            (empty(false, true), &with_token),
            (empty(false, false), &config),
        ] {
            assert!(read(answer, config).unwrap().workers.is_empty());
        }
        let mut granted = answer(PASSING_2_0_4);
        granted.default_deny = true;
        assert_eq!(read(granted, &config).unwrap().workers.len(), 4);
    }

    /// With no token, a default-deny Consul may read through the agent's
    /// default token, so an empty answer after instances is the last one
    /// leaving: it empties the fleet rather than keeping the instance that
    /// left, and the warning about a missing token stands while it lasts.
    #[tokio::test]
    async fn the_last_instance_leaves_a_default_deny_consul_read_with_no_token() {
        let config = config_in(&pd());
        let client = config.client().unwrap();
        let mut discovery = ConsulDiscovery::new(config, create_test_app_context(), client);
        let deny = |mut answer: Answer| {
            answer.default_deny = true;
            answer
        };
        discovery.take(Ok(deny(answer(PASSING_2_0_4))));
        assert_eq!(listed(&discovery).map(|ids| ids.len()), Some(4));
        assert!(!discovery.warned_tokenless);
        discovery.take(Ok(deny(answer("[]"))));
        assert_eq!(listed(&discovery).map(|ids| ids.len()), Some(0));
        assert!(discovery.warned_tokenless);
        discovery.take(Ok(deny(answer(PASSING_2_0_4))));
        assert!(!discovery.warned_tokenless);
    }

    #[test]
    fn the_next_query_waits_past_the_answered_index() {
        assert_eq!(next_index(0, 27), 27);
        assert_eq!(next_index(27, 27), 27);
        assert_eq!(next_index(27, 31), 31);
        // Backwards: read afresh.
        assert_eq!(next_index(27, 5), 0);
        // 0 would not block, so it counts as 1, also after a 1.
        assert_eq!(next_index(0, 0), 1);
        assert_eq!(next_index(1, 0), 1);
    }

    // ---- Configuration. ----

    #[test]
    fn the_address_is_read_as_consul_clients_read_it() {
        let health = "/v1/health/service/smg-worker";
        for (flag, env, url) in [
            (None, &[][..], "http://127.0.0.1:8500"),
            (
                None,
                &[("CONSUL_HTTP_SSL", "true")][..],
                "https://127.0.0.1:8500",
            ),
            (
                None,
                &[("CONSUL_HTTP_ADDR", "")][..],
                "http://127.0.0.1:8500",
            ),
            (
                None,
                &[("CONSUL_HTTP_ADDR", "consul.internal:8500")][..],
                "http://consul.internal:8500",
            ),
            (
                None,
                &[
                    ("CONSUL_HTTP_ADDR", "consul.internal:8501"),
                    ("CONSUL_HTTP_SSL", "true"),
                ][..],
                "https://consul.internal:8501",
            ),
            // Asked for, TLS stands for an address that names http too.
            (
                Some("http://flag:8500"),
                &[("CONSUL_HTTP_ADDR", "env:8500"), ("CONSUL_HTTP_SSL", "1")][..],
                "https://flag:8500",
            ),
            (
                None,
                &[
                    ("CONSUL_HTTP_ADDR", "http://consul.internal"),
                    ("CONSUL_HTTP_SSL", "true"),
                ][..],
                "https://consul.internal:80",
            ),
            (Some("http://flag:8500"), &[][..], "http://flag:8500"),
            (
                Some("https://consul.internal/"),
                &[][..],
                "https://consul.internal",
            ),
            (Some("[fd00::1]:8500"), &[][..], "http://[fd00::1]:8500"),
        ] {
            let resolved = resolve(flag, env).unwrap();
            assert_eq!(
                resolved.url.as_str(),
                format!("{url}{health}"),
                "{flag:?} {env:?}"
            );
        }
        let escaped = resolve_service("smg worker/a", None, &[]).unwrap();
        assert_eq!(
            escaped.url.as_str(),
            "http://127.0.0.1:8500/v1/health/service/smg%20worker%2Fa"
        );
    }

    /// What the provider cannot honor refuses to start, naming the setting.
    #[test]
    fn settings_it_cannot_honor_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let cert = dir.path().join("cert.pem");
        std::fs::write(&cert, "not a certificate").unwrap();
        let cert = cert.to_str().unwrap();
        let https = Some("https://consul:8501");
        for (flag, env, field) in [
            (
                Some("unix:///run/consul.sock"),
                &[][..],
                "discovery.address",
            ),
            (
                Some("http://consul:8500/prefix"),
                &[][..],
                "discovery.address",
            ),
            (
                Some("http://user:secret-pw@consul:8500"),
                &[][..],
                "discovery.address",
            ),
            (Some("ftp://consul:21"), &[][..], "discovery.address"),
            (
                None,
                &[("CONSUL_HTTP_ADDR", "http://consul:8500?x=1")][..],
                "CONSUL_HTTP_ADDR",
            ),
            (
                None,
                &[("CONSUL_HTTP_AUTH", "user:secret-pw")][..],
                "CONSUL_HTTP_AUTH",
            ),
            (
                None,
                &[("CONSUL_CAPATH", "/etc/consul/ca")][..],
                "CONSUL_CAPATH",
            ),
            (
                None,
                &[("CONSUL_TLS_SERVER_NAME", "consul")][..],
                "CONSUL_TLS_SERVER_NAME",
            ),
            (
                None,
                &[("CONSUL_NAMESPACE", "team")][..],
                "CONSUL_NAMESPACE",
            ),
            (None, &[("CONSUL_PARTITION", "p")][..], "CONSUL_PARTITION"),
            (
                None,
                &[("CONSUL_HTTP_SSL_VERIFY", "false")][..],
                "CONSUL_HTTP_SSL_VERIFY",
            ),
            (None, &[("CONSUL_HTTP_SSL", "yes")][..], "CONSUL_HTTP_SSL"),
            (
                None,
                &[("CONSUL_HTTP_TOKEN_FILE", "/no/such/token")][..],
                "CONSUL_HTTP_TOKEN_FILE",
            ),
            (
                https,
                &[("CONSUL_CACERT", "/no/such/ca.pem")][..],
                "CONSUL_CACERT",
            ),
            (
                https,
                &[("CONSUL_CLIENT_CERT", cert)][..],
                "CONSUL_CLIENT_CERT",
            ),
            (
                https,
                &[("CONSUL_CLIENT_KEY", cert)][..],
                "CONSUL_CLIENT_KEY",
            ),
        ] {
            match resolve(flag, env) {
                Err(ConfigError::InvalidValue {
                    field: actual,
                    value,
                    ..
                }) => {
                    assert_eq!(actual, field, "{flag:?} {env:?}");
                    assert!(
                        !value.contains("secret-pw"),
                        "the password was repeated: {value}"
                    );
                }
                other => panic!("{flag:?} {env:?}: expected a refusal, got {other:?}"),
            }
        }

        // A CA file that holds no certificate refuses to start too.
        match resolve(https, &[("CONSUL_CACERT", cert)]) {
            Err(ConfigError::ValidationFailed { reason }) => {
                assert!(reason.contains("CONSUL_CACERT"), "{reason}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        // Verification on is the default, and the TLS settings are read
        // over HTTPS only.
        assert!(resolve(None, &[("CONSUL_HTTP_SSL_VERIFY", "true")]).is_ok());
        let plain = resolve(None, &[("CONSUL_CACERT", "/no/such/ca.pem")]).unwrap();
        assert!(plain.tls.ca_bundle.is_none());
    }

    /// A self-signed certificate and its key, in PEM.
    fn self_signed() -> (Vec<u8>, Vec<u8>) {
        use openssl::{
            asn1::Asn1Time, bn::BigNum, hash::MessageDigest, pkey::PKey, rsa::Rsa,
            x509::X509NameBuilder,
        };
        let key = PKey::from_rsa(Rsa::generate(2048).unwrap()).unwrap();
        let mut name = X509NameBuilder::new().unwrap();
        name.append_entry_by_text("CN", "consul").unwrap();
        let name = name.build();
        let mut cert = openssl::x509::X509Builder::new().unwrap();
        cert.set_version(2).unwrap();
        cert.set_serial_number(&BigNum::from_u32(1).unwrap().to_asn1_integer().unwrap())
            .unwrap();
        cert.set_subject_name(&name).unwrap();
        cert.set_issuer_name(&name).unwrap();
        cert.set_pubkey(&key).unwrap();
        cert.set_not_before(&Asn1Time::days_from_now(0).unwrap())
            .unwrap();
        cert.set_not_after(&Asn1Time::days_from_now(1).unwrap())
            .unwrap();
        cert.sign(&key, MessageDigest::sha256()).unwrap();
        (
            cert.build().to_pem().unwrap(),
            key.private_key_to_pem_pkcs8().unwrap(),
        )
    }

    /// Over HTTPS, the CA bundle and the client certificate with its key are
    /// read, the certificate first; the client builds with them.
    #[test]
    fn https_reads_the_ca_and_the_client_certificate() {
        let dir = tempfile::tempdir().unwrap();
        let (cert, key) = self_signed();
        let path = |name: &str, contents: &[u8]| {
            let path = dir.path().join(name);
            std::fs::write(&path, contents).unwrap();
            path.to_str().unwrap().to_string()
        };
        let (ca, cert_path, key_path) = (
            path("ca.pem", &cert),
            path("cert.pem", &cert),
            path("key.pem", &key),
        );
        let config = resolve(
            Some("https://consul:8501"),
            &[
                ("CONSUL_CACERT", &ca),
                ("CONSUL_CLIENT_CERT", &cert_path),
                ("CONSUL_CLIENT_KEY", &key_path),
            ],
        )
        .unwrap();
        assert_eq!(config.tls.ca_bundle.as_deref(), Some(&cert[..]));
        let identity = config.tls.identity.as_deref().unwrap();
        assert!(identity.starts_with(&cert) && identity.ends_with(&key));
        assert!(config.client().is_ok());
    }

    /// The token is read from its file before the variable, trimmed, and
    /// never shown.
    #[test]
    fn the_token_comes_from_its_file_first_and_never_shows() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("token");
        std::fs::write(&file, " file-token\n").unwrap();
        let file = file.to_str().unwrap();
        let token =
            |env: &[(&str, &str)]| resolve(None, env).unwrap().token.map(|Token(token)| token);
        assert_eq!(
            token(&[
                ("CONSUL_HTTP_TOKEN_FILE", file),
                ("CONSUL_HTTP_TOKEN", "env-token")
            ]),
            Some("file-token".to_string())
        );
        assert_eq!(
            token(&[("CONSUL_HTTP_TOKEN", "env-token")]),
            Some("env-token".to_string())
        );
        assert_eq!(token(&[]), None);

        let config = resolve(None, &[("CONSUL_HTTP_TOKEN", "env-token")]).unwrap();
        assert!(!format!("{config:?}").contains("env-token"));
    }

    /// A query asks for passing instances in the datacenter, with the token;
    /// past an index it blocks for the wait, and may take that long and
    /// Consul's spread on top.
    #[test]
    fn a_query_asks_for_passing_instances_past_its_index() {
        let mut config = resolve(
            Some("http://consul:8500"),
            &[("CONSUL_HTTP_TOKEN", "secret")],
        )
        .unwrap();
        config.datacenter = Some("dc2".to_string());
        let client = config.client().unwrap();

        let first = config.request(&client, 0).build().unwrap();
        assert_eq!(
            first.url().as_str(),
            "http://consul:8500/v1/health/service/smg-worker?passing=true&dc=dc2"
        );
        assert_eq!(first.timeout(), Some(&REQUEST_TIMEOUT));
        assert_eq!(first.headers()["X-Consul-Token"], "secret");

        let blocking = config.request(&client, 42).build().unwrap();
        assert_eq!(
            blocking.url().as_str(),
            "http://consul:8500/v1/health/service/smg-worker?passing=true&dc=dc2&index=42&wait=300s"
        );
        assert_eq!(
            blocking.timeout(),
            Some(&(WAIT + WAIT / 16 + REQUEST_TIMEOUT))
        );
    }

    // ---- Against a scripted Consul. ----

    /// One scripted answer.
    #[derive(Clone)]
    struct Reply {
        status: u16,
        headers: Vec<(&'static str, String)>,
        body: String,
        /// How long the answer is held, as a blocking query is.
        hold: Duration,
    }

    impl Reply {
        fn ok(index: u64, body: &str) -> Self {
            Self {
                status: 200,
                headers: vec![("X-Consul-Index", index.to_string())],
                body: body.to_string(),
                hold: Duration::ZERO,
            }
        }

        fn status(status: u16, body: &str) -> Self {
            Self {
                status,
                headers: vec![],
                body: body.to_string(),
                hold: Duration::ZERO,
            }
        }

        fn with(mut self, name: &'static str, value: &str) -> Self {
            self.headers.push((name, value.to_string()));
            self
        }

        fn held(mut self, hold: Duration) -> Self {
            self.hold = hold;
            self
        }
    }

    /// A local server answering health queries from a script, and keeping
    /// each request's query. Past the script, it holds a request for good.
    struct FakeConsul {
        address: String,
        seen: Arc<Mutex<Vec<HashMap<String, String>>>>,
    }

    impl FakeConsul {
        async fn start(replies: Vec<Reply>) -> Self {
            let replies = Arc::new(Mutex::new(VecDeque::from(replies)));
            let seen = Arc::new(Mutex::new(Vec::new()));
            let app = axum::Router::new().fallback({
                let seen = seen.clone();
                move |uri: http::Uri| {
                    let query = url::form_urlencoded::parse(uri.query().unwrap_or("").as_bytes())
                        .into_owned()
                        .collect();
                    seen.lock().unwrap().push(query);
                    let reply = replies.lock().unwrap().pop_front();
                    async move {
                        let Some(reply) = reply else {
                            time::sleep(Duration::from_secs(3600)).await;
                            return axum::response::Response::new(axum::body::Body::empty());
                        };
                        time::sleep(reply.hold).await;
                        let mut response = axum::response::Response::builder().status(reply.status);
                        for (name, value) in &reply.headers {
                            response = response.header(*name, value);
                        }
                        response.body(axum::body::Body::from(reply.body)).unwrap()
                    }
                }
            });
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = format!("http://{}", listener.local_addr().unwrap());
            #[expect(
                clippy::disallowed_methods,
                reason = "the test server lives for the duration of the test"
            )]
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            Self { address, seen }
        }

        fn config(&self, mode: &RoutingMode) -> ConsulProviderConfig {
            ConsulProviderConfig::with_env(
                &discovery("smg-worker", Some(&self.address)),
                mode,
                |_| None,
            )
            .unwrap()
        }

        /// The `index` each request asked past, in order.
        fn indexes(&self) -> Vec<Option<String>> {
            self.seen
                .lock()
                .unwrap()
                .iter()
                .map(|query| query.get("index").cloned())
                .collect()
        }

        fn requests(&self) -> usize {
            self.seen.lock().unwrap().len()
        }
    }

    #[tokio::test]
    async fn an_answer_is_read_with_its_index_and_acl_headers() {
        let consul = FakeConsul::start(vec![Reply::ok(27, PASSING_2_0_4)
            .with("X-Consul-Default-Acl-Policy", "deny")
            .with("X-Consul-Results-Filtered-By-Acls", "true")])
        .await;
        let config = consul.config(&pd());
        let answer = query(config.request(&config.client().unwrap(), 0))
            .await
            .unwrap();
        assert_eq!(answer.index, 27);
        assert_eq!(answer.entries.len(), 5);
        assert!(answer.filtered && answer.default_deny);
    }

    /// Each way a query can fail says what Consul answered, and how the
    /// failure is counted.
    #[tokio::test]
    async fn a_query_that_fails_says_why() {
        let consul = FakeConsul::start(vec![
            Reply::status(403, "ACL not found\n"),
            Reply::status(500, "No path to datacenter"),
            Reply::status(200, "[]"),
            Reply::ok(3, "not json"),
        ])
        .await;
        let config = consul.config(&pd());
        let client = config.client().unwrap();
        let mut errors = Vec::new();
        for _ in 0..4 {
            let error = query(config.request(&client, 0)).await.unwrap_err();
            errors.push((error.metric_reason(), error.to_string()));
        }
        let (unreadable, invalid) = (
            metrics_labels::SNAPSHOT_ERROR_UNREADABLE,
            metrics_labels::SNAPSHOT_ERROR_INVALID,
        );
        assert_eq!(
            errors[..3],
            [
                (
                    unreadable,
                    "Consul query failed: Consul answered 403 Forbidden: ACL not found".to_string()
                ),
                (
                    unreadable,
                    "Consul query failed: Consul answered 500 Internal Server Error: No path to \
                     datacenter"
                        .to_string()
                ),
                (
                    invalid,
                    "Consul answer rejected: the answer has no X-Consul-Index".to_string()
                ),
            ]
        );
        assert_eq!(errors[3].0, invalid);
        assert!(
            errors[3].1.contains("the answer does not parse"),
            "{}",
            errors[3].1
        );

        // Nothing listening at all.
        let closed = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap()
        };
        let config = ConsulProviderConfig::with_env(
            &discovery("smg-worker", Some(&format!("http://{closed}"))),
            &pd(),
            |_| None,
        )
        .unwrap();
        let error = query(config.request(&config.client().unwrap(), 0))
            .await
            .unwrap_err();
        assert_eq!(error.metric_reason(), unreadable);
    }

    fn listed(discovery: &ConsulDiscovery) -> Option<Vec<String>> {
        discovery.last_good.desired().map(|desired| {
            desired
                .workers
                .values()
                .map(|worker| worker.discovery_id.clone())
                .collect()
        })
    }

    /// Each answer moves the index the next query waits past. One that went
    /// backwards, or a failure, reads afresh from 0, and one of 0 waits past
    /// 1. A failure keeps the last good list.
    #[tokio::test]
    async fn the_index_follows_the_answers() {
        let one = r#"[{"Node":{"Node":"n1","Address":"10.0.0.1","Datacenter":"dc1"},
                       "Service":{"ID":"w1","Port":8000,"Meta":{"smg_worker_type":"decode"},"CreateIndex":1}}]"#;
        let two = r#"[{"Node":{"Node":"n1","Address":"10.0.0.1","Datacenter":"dc1"},
                       "Service":{"ID":"w1","Port":8000,"Meta":{"smg_worker_type":"decode"},"CreateIndex":1}},
                      {"Node":{"Node":"n2","Address":"10.0.0.2","Datacenter":"dc1"},
                       "Service":{"ID":"w2","Port":8000,"Meta":{"smg_worker_type":"decode"},"CreateIndex":2}}]"#;
        let consul = FakeConsul::start(vec![
            Reply::ok(10, one),
            Reply::ok(12, two),
            Reply::ok(5, one),
            Reply::ok(5, one),
            Reply::status(500, "No cluster leader"),
            Reply::ok(0, "[]"),
            Reply::ok(0, "[]"),
        ])
        .await;
        let config = consul.config(&pd());
        let client = config.client().unwrap();
        let mut discovery = ConsulDiscovery::new(config, create_test_app_context(), client);

        let mut seen = Vec::new();
        for _ in 0..7 {
            let answer = discovery.next_query(Duration::ZERO).await;
            discovery.take(answer);
            seen.push((discovery.index, listed(&discovery).map(|ids| ids.len())));
        }
        assert_eq!(
            seen,
            [
                (10, Some(1)),
                (12, Some(2)),
                (0, Some(1)),
                (5, Some(1)),
                (0, Some(1)),
                (1, Some(0)),
                (1, Some(0)),
            ]
        );
        let index = |index: &str| Some(index.to_string());
        assert_eq!(
            consul.indexes(),
            [
                None,
                index("10"),
                index("12"),
                None,
                index("5"),
                None,
                index("1")
            ]
        );
    }

    /// Failures in a row wait longer each time, up to the most; an answer
    /// resets the wait.
    #[tokio::test]
    async fn failures_back_off_and_an_answer_resets_the_backoff() {
        let config = config_in(&pd());
        let client = config.client().unwrap();
        let mut discovery = ConsulDiscovery::new(config, create_test_app_context(), client);
        let failure = || {
            Err(ConsulError::Unreadable {
                reason: "connection refused".to_string(),
            })
        };
        let delays: Vec<u64> = (0..7)
            .map(|_| discovery.take(failure()).as_secs())
            .collect();
        assert_eq!(delays, [1, 2, 4, 8, 16, 30, 30]);

        assert!(discovery.take(Ok(answer("[]"))) <= MIN_GAP);
        assert_eq!(discovery.take(failure()), RETRY_MIN);
    }

    /// The tick reconciles between answers without touching the query in
    /// flight: a query Consul holds is still the one waiting when it answers.
    #[tokio::test]
    async fn ticks_never_restart_the_query_in_flight() {
        let consul = FakeConsul::start(vec![
            Reply::ok(10, "[]"),
            Reply::ok(11, "[]").held(Duration::from_millis(2000)),
        ])
        .await;
        let mut config = consul.config(&pd());
        config.check_interval = Duration::from_millis(50);
        let client = config.client().unwrap();
        #[expect(
            clippy::disallowed_methods,
            reason = "the loop under test is aborted at the end of the test"
        )]
        let handle =
            task::spawn(ConsulDiscovery::new(config, create_test_app_context(), client).run());

        // The first answer at once; the second query a gap later, held.
        time::sleep(Duration::from_millis(1500)).await;
        assert_eq!(consul.requests(), 2);
        time::sleep(Duration::from_millis(1000)).await;
        assert_eq!(consul.requests(), 2, "a tick restarted the held query");
        // Answered, the next query follows.
        time::sleep(Duration::from_millis(1500)).await;
        assert_eq!(consul.requests(), 3);
        assert_eq!(
            consul.indexes(),
            [None, Some("10".to_string()), Some("11".to_string())]
        );
        handle.abort();
    }

    // ---- A change that lands while the instance still registers. ----

    /// An engine for one instance of the scripted service: `/health` after
    /// `delay`, then the metadata a registration reads; 404 elsewhere, as the
    /// metadata fallbacks expect.
    async fn fake_engine(delay: Duration) -> (u16, task::JoinHandle<()>) {
        use axum::{routing::get, Json, Router};
        let app = Router::new()
            .route(
                "/health",
                get(move || async move {
                    time::sleep(delay).await;
                    Json(serde_json::json!({"status": "ok"}))
                }),
            )
            .route(
                "/model_info",
                get(|| async {
                    Json(serde_json::json!({"model_path": "m", "is_generation": true}))
                }),
            )
            .route(
                "/server_info",
                get(|| async { Json(serde_json::json!({"version": "test"})) }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        #[expect(
            clippy::disallowed_methods,
            reason = "the engine lives for the duration of the test"
        )]
        let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (port, handle)
    }

    /// An instance Consul drops while the gateway is still registering it is
    /// not removed by the answer that drops it: that answer's pass finds no
    /// worker of the instance yet, and the registration lands afterwards.
    /// The next periodic pass removes it, and nothing before that does; the
    /// e2e keeps its pass on for this reason.
    #[tokio::test]
    async fn an_instance_dropped_while_it_registers_is_removed_by_the_next_pass() {
        let (a, _engine_a) = fake_engine(Duration::ZERO).await;
        // Its registration waits on this answer twice (the classification
        // probe, then connection detection): past Consul's second answer, a
        // gap after the first, and the query that follows it.
        let (b, _engine_b) = fake_engine(Duration::from_millis(1500)).await;
        let instance = |node: &str, id: &str, port: u16, created: u64| {
            serde_json::json!({
                "Node": {"Node": node, "Address": "127.0.0.1", "Datacenter": "dc1"},
                "Service": {"ID": id, "Port": port, "Meta": {}, "CreateIndex": created}
            })
        };
        let both = serde_json::json!([
            instance("node-a", "worker-a", a, 15),
            instance("node-b", "worker-b", b, 17)
        ])
        .to_string();
        let only_a = serde_json::json!([instance("node-a", "worker-a", a, 15)]).to_string();
        let consul = FakeConsul::start(vec![
            Reply::ok(19, &both),
            // Woken by worker-b's check going critical, a gap after the first.
            Reply::ok(20, &only_a).held(Duration::from_millis(250)),
        ])
        .await;
        let mut config = consul.config(&regular());
        config.check_interval = Duration::from_secs(4);
        let client = config.client().unwrap();
        let app_context = live_app_context();
        let registry = Arc::clone(&app_context.worker_registry);
        let registered = move |port: u16| {
            registry
                .get_all()
                .iter()
                .any(|worker| worker.url().ends_with(&format!(":{port}")))
        };
        let started = time::Instant::now();
        #[expect(
            clippy::disallowed_methods,
            reason = "the loop under test is aborted at the end of the test"
        )]
        let handle = task::spawn(ConsulDiscovery::new(config, app_context, client).run());

        // worker-b registers once its engine answers, after Consul dropped it.
        while !registered(b) {
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "worker-b never registered"
            );
            time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(
            consul.requests(),
            3,
            "the answer that dropped worker-b was taken before it registered"
        );
        assert!(registered(a));

        // Nothing but a pass removes it: until the next one it stays.
        time::sleep_until(started + Duration::from_millis(3500)).await;
        assert!(registered(b), "removed before the next pass");

        // The pass at 4 s removes it.
        while registered(b) {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "worker-b not removed by the next pass"
            );
            time::sleep(Duration::from_millis(20)).await;
        }
        assert!(registered(a));
        handle.abort();
    }

    #[test]
    fn modes_without_worker_pools_cannot_use_consul_discovery() {
        let openai = RoutingMode::OpenAI {
            worker_urls: vec![],
        };
        assert!(matches!(
            ConsulProviderConfig::with_env(&discovery("smg-worker", None), &openai, |_| None),
            Err(ConfigError::ValidationFailed { .. })
        ));
    }
}
