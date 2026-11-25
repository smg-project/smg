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
//! An empty answer from a Consul that denies by default, to a host with no
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
use smg_worker::{
    connection::{client_factory::http_client_builder, http_client::build_client},
    identity::endpoint::EndpointKey,
};
use tokio::{task, time};
use tracing::{info, warn};

use super::{
    config::{ConsulDiscoveryConfig, DiscoveryConfigError, DiscoveryConfigResult},
    host::DiscoveryHost,
    provider::{endpoint_at, DiscoveredWorker, DiscoveryKind, Roles},
    reconciler::DesiredState,
    series,
    snapshot::{LastGood, Skip, SkipLog, SnapshotError},
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

/// Consul discovery as the running host needs it.
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
    pub fn new(
        config: &ConsulDiscoveryConfig,
        roles: Option<Roles>,
    ) -> DiscoveryConfigResult<Self> {
        Self::with_env(config, roles, |name| std::env::var(name).ok())
    }

    /// [`Self::new`] with `env` as the environment.
    fn with_env(
        config: &ConsulDiscoveryConfig,
        roles: Option<Roles>,
        env: impl Fn(&str) -> Option<String>,
    ) -> DiscoveryConfigResult<Self> {
        let roles = roles.ok_or_else(|| DiscoveryConfigError::ValidationFailed {
            reason: "Consul discovery needs regular, prefill-decode or encode-prefill-decode \
                     routing"
                .to_string(),
        })?;
        // Consul's clients treat an empty variable as unset.
        let env = |name: &str| env(name).filter(|value| !value.is_empty());
        let refuse = |name: &str, value: String, reason: &str| DiscoveryConfigError::InvalidValue {
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
            .map_err(|reason| DiscoveryConfigError::ValidationFailed {
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
            let mut builder = http_client_builder()
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
            // Instance metadata carries no pairing protocol: only the one an
            // engine reports about itself applies.
            pairing_protocol: None,
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
                reason: series::DISCOVERY_SKIP_INVALID,
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
            reason: series::DISCOVERY_SKIP_CONFLICT,
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
            Self::Unreadable { .. } => series::SNAPSHOT_ERROR_UNREADABLE,
            Self::Invalid { .. } => series::SNAPSHOT_ERROR_INVALID,
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
    host: Arc<dyn DiscoveryHost>,
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
    let handle = task::spawn(ConsulDiscovery::new(config, host, client).run());

    Ok(handle)
}

type Query = Pin<Box<dyn Future<Output = Result<Answer, ConsulError>> + Send>>;

/// The provider's loop and its state.
struct ConsulDiscovery {
    config: ConsulProviderConfig,
    host: Arc<dyn DiscoveryHost>,
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
        host: Arc<dyn DiscoveryHost>,
        client: reqwest::Client,
    ) -> Self {
        let last_good = LastGood::new(DiscoveryKind::Consul, config.describe(), "instance list");
        Self {
            config,
            host,
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
                    self.last_good.reconcile(self.host.as_ref(), started_at).await;
                }
                _ = tick.tick() => {
                    let started_at = time::Instant::now();
                    self.last_good.reconcile(self.host.as_ref(), started_at).await;
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
mod tests;
