//! What the crate's unit tests and its integration binaries share: fixtures
//! with no home in one subsystem. Compiled for the crate's own test targets
//! only (`cfg(test)`, or the `test-util` feature the self dev-dependency
//! turns on for them), never into a production build.

/// A test's downloads from the model hub. The file sits in the test harness's
/// source tree but is compiled only here, so the crate's unit tests fetch as
/// its integration binaries do (the harness depends on this crate and is not
/// linked into its unit tests); the harness re-exports this module, so a
/// binary linking both has one module and one download lock.
#[path = "../test_harness/src/hub_fetch.rs"]
pub mod hub_fetch;

use std::{
    collections::BTreeMap,
    fs,
    net::{Ipv4Addr, SocketAddr},
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

use k8s_openapi::{
    api::core::v1::{Pod, PodCondition, PodSpec, PodStatus},
    apimachinery::pkg::apis::meta::v1::ObjectMeta,
};
use llm_tokenizer::{
    traits::{Encoding, Tokenizer},
    SpecialTokens,
};
use tokio::net::{TcpListener, TcpSocket};

/// A loopback address nothing answers at, refused for as long as this value
/// lives.
///
/// The socket behind it is bound and never listening: the host's own stack
/// refuses every connection at once, the way a dead worker's port does, and
/// the port stays taken, so no other test's `bind("127.0.0.1:0")` can draw
/// it in the meantime. A listener bound and dropped for the same purpose
/// gives its port back, and a neighbouring test's stub then answers in the
/// unreachable side's place: a `404` or `405` where a transport failure was
/// expected, or a request held until the test's bound.
///
/// Keep the value in a named binding for the whole case: a `_` pattern or a
/// temporary drops it, and the port is free again.
#[must_use = "the address is refused only while this value lives"]
pub struct RefusedAddress {
    _reservation: TcpSocket,
    addr: SocketAddr,
}

impl RefusedAddress {
    /// Reserve a loopback port of the system's choice, and refuse it.
    #[expect(
        clippy::expect_used,
        reason = "a test fixture: a loopback socket the system cannot give is a test failure"
    )]
    pub fn reserve() -> Self {
        let socket = TcpSocket::new_v4().expect("a TCP socket");
        socket
            .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .expect("a loopback port of the system's choice");
        let addr = socket.local_addr().expect("the bound address");
        Self {
            _reservation: socket,
            addr,
        }
    }

    /// The address, `127.0.0.1:<port>`.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The port alone, for a worker registered with a bootstrap port.
    pub fn port(&self) -> u16 {
        self.addr.port()
    }

    /// The address as a plain HTTP URL, `http://127.0.0.1:<port>`.
    pub fn http_url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

/// An engine's HTTP metadata endpoints as a test serves them: each route
/// answers every GET with its status and JSON body, any other path a 404.
/// Returns the base URL; the stub serves for as long as the runtime that
/// spawned it runs.
#[expect(
    clippy::expect_used,
    reason = "a test fixture: a loopback port the system cannot give is a test failure"
)]
pub async fn serve_json_routes(
    routes: Vec<(&'static str, http::StatusCode, serde_json::Value)>,
) -> String {
    let mut router = axum::Router::new();
    for (path, status, body) in routes {
        router = router.route(
            path,
            axum::routing::get(move || {
                let body = body.clone();
                async move { (status, axum::Json(body)) }
            }),
        );
    }
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .expect("a loopback port of the system's choice");
    let addr = listener.local_addr().expect("the bound address");
    #[expect(
        clippy::disallowed_methods,
        reason = "a test fixture: the stub lives as long as the test's runtime"
    )]
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    format!("http://{addr}")
}

/// A running, ready pod named `name` at `ip`, carrying `labels`: the shape
/// the Kubernetes watchers' tests feed to their label selectors.
pub fn make_labeled_pod(name: &str, ip: &str, labels: &[(&str, &str)]) -> Pod {
    let mut label_map = BTreeMap::new();
    for &(k, v) in labels {
        label_map.insert(k.to_string(), v.to_string());
    }
    Pod {
        metadata: ObjectMeta {
            name: Some(name.to_string()),
            uid: Some(format!("uid-{name}")),
            labels: Some(label_map),
            ..Default::default()
        },
        spec: Some(PodSpec::default()),
        status: Some(PodStatus {
            pod_ip: Some(ip.to_string()),
            phase: Some("Running".to_string()),
            conditions: Some(vec![PodCondition {
                type_: "Ready".to_string(),
                status: "True".to_string(),
                last_probe_time: None,
                last_transition_time: None,
                message: None,
                reason: None,
                observed_generation: None,
            }]),
            ..Default::default()
        }),
    }
}

/// Names a directory that holds the DeepSeek-V4.1 tokenizer files; set,
/// nothing is fetched.
pub const DEEPSEEK_V41_MODEL_DIR_ENV: &str = "DEEPSEEK_V41_MODEL_DIR";
/// Names a directory that holds the Kimi-K3 tokenizer files; set, nothing is
/// fetched.
pub const KIMI_K3_TOKENIZER_DIR_ENV: &str = "SMG_KIMI_K3_TOKENIZER_DIR";

/// The tokenizer files of a public checkpoint, fetched once into the
/// package's `.tokenizer_cache/` for the tests that render through the real
/// tokenizer, as the tokenizer crate's parity tests fetch theirs: the cases
/// run wherever the hub is reachable instead of passing without running.
struct CheckpointTokenizer {
    /// The checkpoint, as the log names it.
    label: &'static str,
    /// The variable naming a directory that already holds the files.
    override_env: &'static str,
    /// The public repository the files are fetched from.
    repo: &'static str,
    /// Where the files are cached, under the package directory the tests
    /// run in.
    cache_dir: &'static str,
    /// The files fetched, each with the least size a whole one has.
    files: &'static [(&'static str, usize)],
    /// The `config.json` written beside them: renderer detection reads the
    /// architecture from it, the multimodal registry the image token.
    config_json: &'static str,
}

const DEEPSEEK_V41: CheckpointTokenizer = CheckpointTokenizer {
    label: "DeepSeek-V4.1",
    override_env: DEEPSEEK_V41_MODEL_DIR_ENV,
    repo: "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main",
    cache_dir: ".tokenizer_cache/deepseek_v41",
    files: &[
        ("tokenizer.json", 1_000_000),
        ("tokenizer_config.json", 500),
    ],
    config_json: r#"{"architectures":["DeepseekV41ForCausalLM"],"model_type":"deepseek_v41","image_token_id":129264}"#,
};

const KIMI_K3: CheckpointTokenizer = CheckpointTokenizer {
    label: "Kimi-K3",
    override_env: KIMI_K3_TOKENIZER_DIR_ENV,
    repo: "https://huggingface.co/moonshotai/Kimi-K3/resolve/main",
    cache_dir: ".tokenizer_cache/kimi_k3",
    files: &[
        ("tiktoken.model", 1_000_000),
        ("tokenizer_config.json", 500),
    ],
    config_json: r#"{"architectures":["KimiK3ForConditionalGeneration"],"model_type":"kimi_k3"}"#,
};

/// The DeepSeek-V4.1 tokenizer directory: `DEEPSEEK_V41_MODEL_DIR`, else
/// `tokenizer.json` and `tokenizer_config.json` fetched once from the public
/// checkpoint, with a `config.json` naming the architecture and the image
/// token. `None` only when the hub never answered and no override is set; a
/// hub that answers wrongly or a cache that cannot be written panics, so the
/// cases cannot be skipped silently.
pub fn ensure_deepseek_v41_cached() -> Option<PathBuf> {
    DEEPSEEK_V41.ensure_cached()
}

/// The Kimi-K3 tokenizer directory: `SMG_KIMI_K3_TOKENIZER_DIR`, else
/// `tiktoken.model` and `tokenizer_config.json` fetched once from the public
/// checkpoint, with a `config.json` naming the architecture. `None` only
/// when the hub never answered and no override is set; a hub that answers
/// wrongly or a cache that cannot be written panics, so the case cannot be
/// skipped silently.
pub fn ensure_kimi_k3_cached() -> Option<PathBuf> {
    KIMI_K3.ensure_cached()
}

impl CheckpointTokenizer {
    /// The directory holding the files: the override as given, else the
    /// cache, each missing file fetched under the download lock and written
    /// whole or not at all. `None` only when the hub never answered a GET (no
    /// connection, a timeout, the request cut short at every try) and no
    /// override is set, so a box offline skips the cases; a hub that answers
    /// wrongly (a non-success status, a short body, a body broken at every
    /// try) or a cache that cannot be written panics with the reason.
    #[expect(
        clippy::panic,
        clippy::print_stdout,
        reason = "a test helper: a download that goes wrong fails loudly, a skipped one is said"
    )]
    fn ensure_cached(&self) -> Option<PathBuf> {
        if let Some(dir) = std::env::var_os(self.override_env) {
            return Some(PathBuf::from(dir));
        }
        let _guard = hub_fetch::download_lock();
        let dir = PathBuf::from(self.cache_dir);
        fs::create_dir_all(&dir)
            .unwrap_or_else(|error| panic!("cannot create {}: {error}", dir.display()));
        let client = crate::clients::blocking_http_client();
        for &(file, min_bytes) in self.files {
            let path = dir.join(file);
            if path.is_file() {
                continue;
            }
            let url = format!("{}/{file}", self.repo);
            println!("Downloading {} {file} from {url}...", self.label);
            let answer = match hub_fetch::fetch(&client, &url) {
                Ok(answer) => answer,
                Err(error) if !hub_fetch::answered(&error) => {
                    println!(
                        "{} {file}: the hub did not answer ({error}); nothing cached",
                        self.label
                    );
                    return None;
                }
                Err(error) => panic!("GET {url}: {error}"),
            };
            assert!(
                answer.status.is_success(),
                "GET {url}: HTTP {}",
                answer.status
            );
            assert!(
                answer.body.len() >= min_bytes,
                "GET {url}: {} bytes, expected at least {min_bytes}",
                answer.body.len()
            );
            // Named for this process, as `hub_fetch::cached_download` names its
            // part file: a second process fetching into the same directory
            // renames its own whole file into place.
            let mut part = path.clone().into_os_string();
            part.push(format!(".{}.part", std::process::id()));
            let part = PathBuf::from(part);
            fs::write(&part, &answer.body)
                .unwrap_or_else(|error| panic!("cannot write {}: {error}", part.display()));
            fs::rename(&part, &path).unwrap_or_else(|error| {
                panic!("cannot rename {} into place: {error}", part.display())
            });
            println!("Cached {} ({} bytes)", path.display(), answer.body.len());
        }
        let config = dir.join("config.json");
        if !config.is_file() {
            fs::write(&config, self.config_json)
                .unwrap_or_else(|error| panic!("cannot write {}: {error}", config.display()));
        }
        Some(dir)
    }
}

/// A tokenizer whose tokens are characters (`id = char as u32`): the
/// streaming tests' and the stream benchmark's, so an engine stream scripted
/// as text decodes with no model's vocabulary behind it.
#[derive(Default)]
pub struct CharacterTokenizer {
    special_tokens: SpecialTokens,
    /// How many times the eager decode step ran (a test reads it).
    pub(crate) eager_steps: AtomicUsize,
    /// The ids `eos_token_ids` reports (none by default).
    eos: Vec<u32>,
}

impl CharacterTokenizer {
    /// A tokenizer whose EOS ids are `eos`: the gateway's stop decoder takes
    /// them as stops, as it does a model's.
    pub fn with_eos(eos: Vec<u32>) -> Self {
        Self {
            eos,
            ..Self::default()
        }
    }
}

impl llm_tokenizer::traits::Encoder for CharacterTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> anyhow::Result<Encoding> {
        Ok(Encoding::Plain(text.chars().map(u32::from).collect()))
    }

    fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> anyhow::Result<Vec<Encoding>> {
        texts
            .iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }
}

impl llm_tokenizer::traits::Decoder for CharacterTokenizer {
    fn decode_step_eager(
        &self,
        token_id: u32,
        ids: &mut Vec<u32>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
    ) -> anyhow::Result<Option<String>> {
        self.eager_steps.fetch_add(1, Ordering::Relaxed);
        self.decode_step(token_id, ids, prefix, prefix_index, skip_special_tokens)
    }

    fn decode(&self, ids: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
        ids.iter()
            .map(|id| char::from_u32(*id).ok_or_else(|| anyhow::anyhow!("invalid character token")))
            .collect()
    }
}

impl Tokenizer for CharacterTokenizer {
    fn vocab_size(&self) -> usize {
        128
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn eos_token_ids(&self) -> &[u32] {
        &self.eos
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        token.chars().next().map(u32::from)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        char::from_u32(id).map(|character| character.to_string())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::io::ErrorKind;

    use super::RefusedAddress;

    /// A connection to the address is refused at once, and no other bind
    /// takes the port while the value lives.
    #[tokio::test]
    async fn the_address_refuses_connections_and_keeps_its_port() {
        let refused = RefusedAddress::reserve();
        let error = tokio::net::TcpStream::connect(refused.addr())
            .await
            .expect_err("nothing answers there");
        assert_eq!(error.kind(), ErrorKind::ConnectionRefused, "{error}");
        let taken = std::net::TcpListener::bind(refused.addr()).expect_err("the port is held");
        assert_eq!(taken.kind(), ErrorKind::AddrInUse, "{taken}");
        assert_eq!(
            refused.http_url(),
            format!("http://127.0.0.1:{}", refused.port())
        );
    }
}
