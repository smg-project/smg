use std::{collections::VecDeque, sync::Mutex};

use smg_worker::test_support::RefusedAddress;

use super::*;
use crate::testing::TestHost;

const PASSING_2_0_4: &str = include_str!("../testdata/consul/health-passing-2.0.4.json");
const PASSING_1_22_7: &str = include_str!("../testdata/consul/health-passing-1.22.7.json");

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
fn resolve(
    address: Option<&str>,
    env: &[(&str, &str)],
) -> DiscoveryConfigResult<ConsulProviderConfig> {
    resolve_service("smg-worker", address, env)
}

fn resolve_service(
    service: &str,
    address: Option<&str>,
    env: &[(&str, &str)],
) -> DiscoveryConfigResult<ConsulProviderConfig> {
    let env: HashMap<String, String> = env
        .iter()
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect();
    ConsulProviderConfig::with_env(&discovery(service, address), Some(Roles::Regular), |name| {
        env.get(name).cloned()
    })
}

fn config_in(roles: Option<Roles>) -> ConsulProviderConfig {
    ConsulProviderConfig::with_env(&discovery("smg-worker", None), roles, |_| None).unwrap()
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
        let outcome = read(answer(recorded), &config_in(Some(Roles::PrefillDecode))).unwrap();
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
                reason: series::DISCOVERY_SKIP_INVALID,
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
    let outcome = read(answer(PASSING_2_0_4), &config_in(Some(Roles::Regular))).unwrap();
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
    let config = config_in(Some(Roles::PrefillDecode));
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
            reason: series::DISCOVERY_SKIP_CONFLICT,
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
    let config = config_in(Some(Roles::PrefillDecode));
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
    let config = config_in(Some(Roles::PrefillDecode));
    let client = config.client().unwrap();
    let mut discovery = ConsulDiscovery::new(config, TestHost::without_a_queue(), client);
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
            Err(DiscoveryConfigError::InvalidValue {
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
        Err(DiscoveryConfigError::ValidationFailed { reason }) => {
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
    let token = |env: &[(&str, &str)]| resolve(None, env).unwrap().token.map(|Token(token)| token);
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

    fn config(&self, roles: Option<Roles>) -> ConsulProviderConfig {
        ConsulProviderConfig::with_env(&discovery("smg-worker", Some(&self.address)), roles, |_| {
            None
        })
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
    let config = consul.config(Some(Roles::PrefillDecode));
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
    let config = consul.config(Some(Roles::PrefillDecode));
    let client = config.client().unwrap();
    let mut errors = Vec::new();
    for _ in 0..4 {
        let error = query(config.request(&client, 0)).await.unwrap_err();
        errors.push((error.metric_reason(), error.to_string()));
    }
    let (unreadable, invalid) = (
        series::SNAPSHOT_ERROR_UNREADABLE,
        series::SNAPSHOT_ERROR_INVALID,
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
    let closed = RefusedAddress::reserve();
    let config = ConsulProviderConfig::with_env(
        &discovery("smg-worker", Some(&closed.http_url())),
        Some(Roles::PrefillDecode),
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
    let config = consul.config(Some(Roles::PrefillDecode));
    let client = config.client().unwrap();
    let mut discovery = ConsulDiscovery::new(config, TestHost::without_a_queue(), client);

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
    let config = config_in(Some(Roles::PrefillDecode));
    let client = config.client().unwrap();
    let mut discovery = ConsulDiscovery::new(config, TestHost::without_a_queue(), client);
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
    let mut config = consul.config(Some(Roles::PrefillDecode));
    config.check_interval = Duration::from_millis(50);
    let client = config.client().unwrap();
    #[expect(
        clippy::disallowed_methods,
        reason = "the loop under test is aborted at the end of the test"
    )]
    let handle =
        task::spawn(ConsulDiscovery::new(config, TestHost::without_a_queue(), client).run());

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

/// An instance Consul drops while its host is still registering it is not
/// removed by the answer that drops it: that answer's pass finds no worker
/// of the instance yet, and the registration lands afterwards. The next
/// periodic pass removes it, and nothing before that does; the e2e keeps
/// its pass on for this reason.
#[tokio::test]
async fn an_instance_dropped_while_it_registers_is_removed_by_the_next_pass() {
    // The host's queue registers what it is handed and dials no worker, so
    // nothing listens on either port.
    let (a, b) = (18101, 18102);
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
    let pass = Duration::from_secs(4);
    let mut config = consul.config(Some(Roles::Regular));
    config.check_interval = pass;
    let client = config.client().unwrap();
    // worker-b's registration waits in the host's queue, its address held,
    // until the test releases it: after Consul has dropped the instance.
    let (host, jobs) = TestHost::applying();
    let address_b = endpoint_at("127.0.0.1", b).unwrap().render();
    jobs.register_at_release(&address_b);
    let registry = Arc::clone(&host.registry);
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
    let handle = task::spawn(ConsulDiscovery::new(config, host, client).run());

    // Consul drops worker-b while its registration waits. The query after
    // the dropping answer starts once that answer's pass is done.
    let guard = Duration::from_secs(30);
    while consul.requests() < 3 {
        assert!(
            started.elapsed() < guard,
            "the query after the dropping answer never came"
        );
        time::sleep(Duration::from_millis(20)).await;
    }
    assert!(registered(a));
    assert!(!registered(b), "registered before its registration landed");

    // worker-b's registration lands, after Consul dropped it.
    jobs.release(&address_b);
    assert!(registered(b), "worker-b never registered");
    assert_eq!(
        consul.requests(),
        3,
        "the answer that dropped worker-b was taken before it registered"
    );

    // Nothing but a pass removes it, and the first pass comes one interval
    // after the start: a removal before that is not the pass's.
    while registered(b) {
        assert!(
            started.elapsed() < guard,
            "worker-b not removed by the next pass"
        );
        time::sleep(Duration::from_millis(20)).await;
    }
    assert!(started.elapsed() >= pass, "removed before the next pass");
    assert!(registered(a));
    handle.abort();
}

#[test]
fn modes_without_worker_pools_cannot_use_consul_discovery() {
    assert!(matches!(
        ConsulProviderConfig::with_env(&discovery("smg-worker", None), None, |_| None),
        Err(DiscoveryConfigError::ValidationFailed { .. })
    ));
}
