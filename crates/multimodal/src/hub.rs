//! HuggingFace Hub integration for downloading model config files.
//!
//! When the model source is a HuggingFace model ID (e.g. `Qwen/Qwen3-VL-8B-Instruct`)
//! rather than a local directory, this module downloads `config.json` and
//! `preprocessor_config.json` / video processor config files to the local HF
//! cache and returns the resolved path.

#[cfg(unix)]
use std::os::unix::fs::{symlink, PermissionsExt};
#[cfg(unix)]
use std::path::Component;
use std::{
    fmt, fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::Context;
use hf_hub::{
    api::tokio::{Api, ApiBuilder, ApiError, ApiRepo},
    Cache, CacheRepo,
};

const HF_TOKEN_ENV: &str = "HF_TOKEN";

/// Bound on one Hub object fetch of this crate (`config.json` or a processor
/// config on a cache miss: the metadata request and the file's download),
/// the one bound for every such fetch. The Hub client has no timeout of its
/// own, so an endpoint that drops packets would hold the fetch until the
/// connect gives up, and with nothing cached after a failure every request of
/// an outage would wait that long before its answer. It is of the cold-fetch
/// class (the order of the gateway's start-up and health bounds) rather than
/// a few seconds: a failure is not cached, so a bound under a slow Hub's or
/// mirror's cold fetch would refuse every request of that model for as long
/// as the Hub stays slow.
///
/// An elapsed bound ends the whole fetch: each of its requests is awaited
/// inline ([`HubRepo::fetch`]), so dropping the fetch drops the request in
/// flight and closes its connection, and a Hub that answers the metadata
/// request and then withholds the body holds nothing of this process past
/// the bound.
const HUB_FETCH_TIMEOUT: Duration = Duration::from_secs(30);

/// Resolve a model source to a local directory containing `config.json`.
///
/// If the source is already a local directory with `config.json`, returns it as-is.
/// Otherwise, treats the source as a HuggingFace model ID and downloads
/// `config.json` plus the processor configs the checkpoint ships to the local
/// HF cache. A processor config the Hub answers 404 for is one the checkpoint
/// does not ship, and the directory is complete without it; a fetch that fails
/// any other way (no answer, a 5xx, a denied token) fails the resolution, so
/// the caller never reads an incomplete directory as a checkpoint without the
/// file.
pub async fn resolve_model_config_dir(source: &str) -> anyhow::Result<PathBuf> {
    resolve_within(source, HUB_FETCH_TIMEOUT).await
}

/// [`resolve_model_config_dir`] with each Hub fetch under `bound`.
async fn resolve_within(source: &str, bound: Duration) -> anyhow::Result<PathBuf> {
    let path = Path::new(source);
    if path.join("config.json").exists() {
        return Ok(path.to_path_buf());
    }

    let hub = HubRepo::open(source)?;

    let config_path = match tokio::time::timeout(bound, hub.fetch("config.json")).await {
        Ok(Ok(path)) => path,
        // The Hub's 404, or the 401/403 it answers without access for a
        // repository that does not exist or is gated, is a checkpoint this
        // caller cannot read: the request's error, not a failure of the Hub,
        // so it keeps its plain mapping.
        Ok(Err(error)) if is_absent(&error) || is_refused(&error) => {
            return Err(unreadable(source, "config.json", error));
        }
        Ok(Err(error)) => return Err(HubFetchError::failed(source, "config.json", &error).into()),
        Err(_) => return Err(HubFetchError::elapsed(source, "config.json", bound).into()),
    };

    // The processor configs beside config.json: a checkpoint ships the ones of
    // its modalities, and the Hub's 404 is its answer that this one is not
    // among them, which the processor takes as absent. A 401/403 is the
    // caller's missing access (the Hub answers it for every file of a gated
    // repository, whether or not the file exists, and a cached config.json
    // proves nothing about access), the request's error as on config.json.
    // Any other failure, a fetch that outlives its bound included, leaves the
    // directory incomplete, so it fails the resolution rather than hand the
    // caller a directory it would read as a checkpoint without the file; the
    // next resolution fetches the file again, the ones already fetched being
    // served from the cache.
    for config_file in [
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "processor_config.json",
    ] {
        match tokio::time::timeout(bound, hub.fetch(config_file)).await {
            Ok(Ok(_)) => {}
            Ok(Err(error)) if is_absent(&error) => {}
            Ok(Err(error)) if is_refused(&error) => {
                return Err(unreadable(source, config_file, error));
            }
            Ok(Err(error)) => return Err(HubFetchError::failed(source, config_file, &error).into()),
            Err(_) => return Err(HubFetchError::elapsed(source, config_file, bound).into()),
        }
    }

    config_path
        .parent()
        .map(|p| p.to_path_buf())
        .ok_or_else(|| anyhow::anyhow!("Invalid HF cache path for model '{source}'"))
}

/// A checkpoint's repository on the Hub and in the local Hub cache, as this
/// crate's resolution reads it.
struct HubRepo {
    api: Api,
    repo: ApiRepo,
    cache: CacheRepo,
}

impl HubRepo {
    /// The Hub client from the environment (`HF_ENDPOINT`, `HF_HOME`, the
    /// cache's token file, `HF_TOKEN` over it) and the checkpoint `source`'s
    /// repository on it and in the cache.
    fn open(source: &str) -> anyhow::Result<Self> {
        let mut builder = ApiBuilder::from_env();
        if let Ok(token) = std::env::var(HF_TOKEN_ENV) {
            if !token.is_empty() {
                builder = builder.with_token(Some(token));
            }
        }
        let api = builder
            .build()
            .context("Failed to build HuggingFace API client")?;
        let repo = api.model(source.to_string());
        // The client keeps its cache to itself; this one is the same cache,
        // read from `HF_HOME` the way the client reads it.
        let cache = Cache::from_env().model(source.to_string());
        Ok(Self { api, repo, cache })
    }

    /// The local path of `filename`: the cached file when the cache has it (no
    /// request), else the file fetched into the cache as the client lays it
    /// out (the blob under its etag, the snapshot's link to it, the revision's
    /// ref). The metadata request and the body's GET are both awaited here
    /// rather than left to the client's own download, which spawns a task per
    /// chunk and detaches it when the download is dropped: a fetch dropped at
    /// its bound ends with no task reading a body that never comes. A body
    /// of another length than the metadata request announced is a failed
    /// fetch, nothing written: a close-delimited answer cut short reads as a
    /// complete body, and cached under the file's etag it would be served as
    /// the file from then on.
    async fn fetch(&self, filename: &str) -> Result<PathBuf, ApiError> {
        if let Some(path) = self.cache.get(filename) {
            return Ok(path);
        }
        let url = self.repo.url(filename);
        let metadata = self.api.metadata(&url).await?;
        let body = self
            .api
            .client()
            .get(&url)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?;
        if body.len() != metadata.size() {
            return Err(io::Error::other(format!(
                "{filename}: the Hub sent {} of {} bytes",
                body.len(),
                metadata.size()
            ))
            .into());
        }
        let blob = self.cache.blob_path(metadata.etag());
        write_whole(&blob, &body)?;
        let pointer = self
            .cache
            .pointer_path(metadata.commit_hash())
            .join(filename);
        link_pointer(&pointer, &blob)?;
        self.cache.create_ref(metadata.commit_hash())?;
        Ok(pointer)
    }
}

/// Writes `content` to `path` whole or not at all: a temporary file beside
/// it, renamed into place, so a reader of the cache (another process
/// resolving the same checkpoint) never sees a partial blob.
fn write_whole(path: &Path, content: &[u8]) -> io::Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| io::Error::other("a blob path has no directory"))?;
    fs::create_dir_all(dir)?;
    // The mode a created file gets (0o666 under the umask), as the client's
    // own blobs have, not a temporary file's owner-only one: the cache is
    // read by every user of it.
    #[cfg(unix)]
    let mut file = tempfile::Builder::new()
        .permissions(fs::Permissions::from_mode(0o666))
        .tempfile_in(dir)?;
    #[cfg(not(unix))]
    let mut file = tempfile::Builder::new().tempfile_in(dir)?;
    file.write_all(content)?;
    file.persist(path).map_err(|error| error.error)?;
    Ok(())
}

/// Links `pointer`, a file of a snapshot directory, to `blob` by a relative
/// path, as the Hub cache keeps its snapshots; a pointer already there (the
/// file linked by an earlier fetch of this snapshot, or by another process
/// now) stays.
fn link_pointer(pointer: &Path, blob: &Path) -> io::Result<()> {
    let dir = pointer
        .parent()
        .ok_or_else(|| io::Error::other("a snapshot path has no directory"))?;
    fs::create_dir_all(dir)?;
    if pointer.exists() {
        return Ok(());
    }
    #[cfg(unix)]
    let linked = symlink(relative_path(blob, dir), pointer);
    #[cfg(not(unix))]
    let linked = fs::hard_link(blob, pointer);
    match linked {
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(()),
        linked => linked,
    }
}

/// `target` relative to the directory `from`: a step up for each component
/// of `from` below their common ancestor, then the rest of `target`.
#[cfg(unix)]
fn relative_path(target: &Path, from: &Path) -> PathBuf {
    let mut target = target.components().peekable();
    let mut from = from.components().peekable();
    while target.peek().is_some() && target.peek() == from.peek() {
        target.next();
        from.next();
    }
    let mut relative: PathBuf = from.map(|_| Component::ParentDir).collect();
    relative.extend(target);
    relative
}

/// A file of a checkpoint the Hub failed to serve: no answer, a 5xx, a 429, a
/// transport failure; never its 404, the answer that the checkpoint does not
/// ship the file, nor a 401/403, a checkpoint this caller cannot read (see
/// `is_refused`). The message names the model, the file and the cause (the Hub's
/// answer or the transport failure, the file's URL included) for the log; a
/// client is told only that the model's configuration could not be fetched,
/// since the fault is on the Hub's side and the next resolution asks again.
#[derive(Debug)]
pub struct HubFetchError {
    /// The checkpoint, as the source named it.
    pub model: String,
    /// The file the Hub did not serve.
    pub file: &'static str,
    /// The failure with its causes, each written once.
    pub cause: String,
}

impl HubFetchError {
    /// The Hub answered the fetch of `file` with `error`, which is not its 404
    /// nor a 401/403.
    fn failed(model: &str, file: &'static str, error: &ApiError) -> Self {
        Self {
            model: model.to_string(),
            file,
            cause: describe(error),
        }
    }

    /// The fetch of `file` outlived `bound`.
    fn elapsed(model: &str, file: &'static str, bound: Duration) -> Self {
        Self {
            model: model.to_string(),
            file,
            cause: format!("no answer within {}s", bound.as_secs()),
        }
    }

    /// Whether `error` is, or wraps, a file the Hub failed to serve.
    pub fn is(error: &anyhow::Error) -> bool {
        error.chain().any(|cause| cause.is::<HubFetchError>())
    }
}

impl fmt::Display for HubFetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Failed to download {} for model '{}': {}",
            self.file, self.model, self.cause
        )
    }
}

impl std::error::Error for HubFetchError {}

/// `true` when the Hub answered that the file is not in the repository (HTTP
/// 404): the expected answer of a checkpoint without that processor config.
/// A 401/403 is [`is_refused`]'s; every other failure (no answer, a 5xx, a
/// proxy error) is a fetch that should have succeeded.
fn is_absent(error: &ApiError) -> bool {
    matches!(
        error,
        ApiError::RequestError(request)
            if request.status().is_some_and(|status| status.as_u16() == 404)
    )
}

/// `true` when the Hub refused the request (HTTP 401 or 403): without access
/// the Hub answers every file of a repository that does not exist or is gated
/// with a 401 or a 403, whether or not the file exists, so this is a
/// checkpoint the caller cannot read, not a failure of the Hub, and no retry
/// mends it. (The client keeps the status but not the header that would tell
/// a missing repository from a bad token.)
fn is_refused(error: &ApiError) -> bool {
    matches!(
        error,
        ApiError::RequestError(request)
            if request
                .status()
                .is_some_and(|status| matches!(status.as_u16(), 401 | 403))
    )
}

/// The plain error of a file this caller cannot read (the Hub's 404 on
/// `config.json`, a 401/403 on any file): the request's error, with the
/// Hub's answer in its chain.
fn unreadable(model: &str, file: &str, error: ApiError) -> anyhow::Error {
    anyhow::Error::new(error).context(format!("Failed to download {file} for model '{model}'"))
}

/// The error's own message followed by the causes below it that the message
/// does not already end with: each `ApiError` variant embeds its source's
/// message in its own (`request error: {0}`), as `io::Error` does for the
/// error it wraps, so a plain walk of the chain would write those twice while
/// the deeper causes (the connect error, the OS error) are what the
/// description is for.
fn describe(error: &ApiError) -> String {
    let mut message = error.to_string();
    let mut cause = std::error::Error::source(error);
    while let Some(next) = cause {
        let text = next.to_string();
        if !message.ends_with(&text) {
            message.push_str(": ");
            message.push_str(&text);
        }
        cause = next.source();
    }
    message
}

#[cfg(test)]
mod tests {
    use std::{
        ffi::OsString,
        io::{self, Read, Write},
        net::{TcpListener, TcpStream},
        path::Path,
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc, Mutex, MutexGuard,
        },
        thread,
        time::{Duration, Instant},
    };

    use hf_hub::api::tokio::ApiError;

    #[cfg(unix)]
    use super::relative_path;
    use super::{describe, is_absent, is_refused, resolve_within, HubFetchError};

    /// The error `repo.get` returns when the Hub answers `status` to the
    /// file's request (hf-hub runs `error_for_status` on the answer).
    fn hub_answer(status: u16) -> ApiError {
        let answer = http::Response::builder()
            .status(status)
            .body("")
            .expect("a valid status code");
        let error = reqwest_hub::Response::from(answer)
            .error_for_status()
            .expect_err("a 4xx or 5xx answer is an error");
        ApiError::RequestError(error)
    }

    #[test]
    fn a_404_answer_is_an_absent_file() {
        assert!(is_absent(&hub_answer(404)));
    }

    /// The refusal that picks the plain error over the retryable typed one on
    /// every file: the 401 of a repository that does not exist or is gated,
    /// the 403 of a gate without access; nothing else.
    #[test]
    fn a_401_or_403_is_a_refusal_and_nothing_else_is() {
        for status in [401, 403] {
            assert!(is_refused(&hub_answer(status)), "status {status}");
            assert!(!is_absent(&hub_answer(status)), "status {status}");
        }
        for status in [404, 429, 500, 502, 503, 504] {
            assert!(!is_refused(&hub_answer(status)), "status {status}");
        }
        assert!(!is_refused(&ApiError::IoError(io::Error::other(
            "connection reset"
        ))));
    }

    #[test]
    fn every_other_failure_is_a_fetch_that_failed() {
        for status in [429, 500, 502, 503, 504] {
            assert!(!is_absent(&hub_answer(status)), "status {status}");
            assert!(!is_refused(&hub_answer(status)), "status {status}");
        }
        assert!(!is_absent(&ApiError::IoError(io::Error::other(
            "connection reset"
        ))));
        assert!(!is_absent(&ApiError::TooManyRetries(Box::new(hub_answer(
            503
        )))));
    }

    /// The typed failure is found through the contexts a caller wraps it in,
    /// and its message names the file, the model and the cause.
    #[test]
    fn a_file_the_hub_failed_to_serve_is_found_through_the_chain() {
        let failed = anyhow::Error::from(HubFetchError {
            model: "org/name".to_string(),
            file: "preprocessor_config.json",
            cause: "request error: HTTP status server error (503 Service Unavailable)".to_string(),
        })
        .context("Failed to resolve model config directory for 'org/name'");
        assert!(HubFetchError::is(&failed));
        assert_eq!(
            failed.root_cause().to_string(),
            "Failed to download preprocessor_config.json for model 'org/name': request error: \
             HTTP status server error (503 Service Unavailable)"
        );
        assert!(!HubFetchError::is(&anyhow::anyhow!("decode failed")));
    }

    /// An error whose message is its own and whose cause is the next one
    /// down: the shape of a connect failure's chain, without a network.
    #[derive(Debug)]
    struct Layered(&'static str, Option<Box<Layered>>);

    impl std::fmt::Display for Layered {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.0)
        }
    }

    impl std::error::Error for Layered {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            self.1
                .as_deref()
                .map(|cause| cause as &(dyn std::error::Error + 'static))
        }
    }

    /// The description writes each message of the chain once: hf-hub's
    /// variant embeds its source's message, `io::Error` embeds the error it
    /// wraps, and a connect error embeds the OS error, so the causes already
    /// at the end of the message are not written again while every deeper
    /// cause is.
    #[test]
    fn the_description_writes_each_cause_once() {
        let status = describe(&hub_answer(503));
        assert!(status.starts_with("request error: "), "{status}");
        assert_eq!(
            status.matches("HTTP status server error (503").count(),
            1,
            "{status}"
        );

        let retries = describe(&ApiError::TooManyRetries(Box::new(hub_answer(503))));
        assert!(
            retries.starts_with("Too many retries: request error: "),
            "{retries}"
        );
        assert_eq!(retries.matches("503").count(), 1, "{retries}");

        let os = Layered("Connection refused (os error 111)", None);
        let connect = Layered(
            "tcp connect error: Connection refused (os error 111)",
            Some(Box::new(os)),
        );
        let client = Layered("client error (Connect)", Some(Box::new(connect)));
        let io = ApiError::IoError(io::Error::other(client));
        assert_eq!(
            describe(&io),
            "I/O error client error (Connect): tcp connect error: Connection refused (os error 111)"
        );
    }

    /// The relative link a snapshot's file gets to its blob: up out of the
    /// snapshot directory (and the file's own directories) to the
    /// repository's, then into its blobs.
    #[cfg(unix)]
    #[test]
    fn a_pointer_links_to_its_blob_relative_to_its_snapshot() {
        let repo = Path::new("/cache/hub/models--org--name");
        assert_eq!(
            relative_path(&repo.join("blobs/e1"), &repo.join("snapshots/c1")),
            Path::new("../../blobs/e1")
        );
        assert_eq!(
            relative_path(&repo.join("blobs/e1"), &repo.join("snapshots/c1/sub")),
            Path::new("../../../blobs/e1")
        );
    }

    /// The commit and etag the stub Hub names, and the length it announces
    /// for the file (never sent whole).
    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
    const ETAG: &str = "etag-of-the-withheld-file";
    const ANNOUNCED_LEN: usize = 42;

    /// What the stub Hub makes of the file's own request.
    #[derive(Clone, Copy)]
    enum Body {
        /// The status line and the announced length, then nothing: the
        /// connection stays open until the client closes it.
        Withheld,
        /// A close-delimited answer (no length) cut after a quarter of the
        /// announced bytes, then closed: what the client reads as a complete
        /// body.
        Cut,
    }

    /// A Hub on loopback that answers a file's metadata request (`Range:
    /// bytes=0-0`) with the commit, etag and content range the client reads,
    /// and the file's own request as `body` says; it records the file's
    /// request arriving and, for a withheld body, the client closing the
    /// connection.
    struct StubHub {
        endpoint: String,
        /// The file's request arrived.
        requested: Arc<AtomicBool>,
        /// The client closed the connection a withheld body was on.
        closed: Arc<AtomicBool>,
    }

    impl StubHub {
        fn start(body: Body) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
            let endpoint = format!("http://{}", listener.local_addr().expect("local addr"));
            let requested = Arc::new(AtomicBool::new(false));
            let closed = Arc::new(AtomicBool::new(false));
            let flags = (requested.clone(), closed.clone());
            thread::spawn(move || {
                for stream in listener.incoming().flatten() {
                    let (requested, closed) = flags.clone();
                    thread::spawn(move || answer(stream, body, &requested, &closed));
                }
            });
            Self {
                endpoint,
                requested,
                closed,
            }
        }
    }

    /// One connection's answer: the metadata, or the file's `body`.
    fn answer(mut stream: TcpStream, body: Body, requested: &AtomicBool, closed: &AtomicBool) {
        let head = request_head(&mut stream);
        if head.contains("bytes=0-0\r\n") {
            let answer = format!(
                "HTTP/1.1 206 Partial Content\r\nx-repo-commit: {COMMIT}\r\netag: \"{ETAG}\"\r\n\
                 content-range: bytes 0-0/{ANNOUNCED_LEN}\r\ncontent-length: 1\r\n\
                 connection: close\r\n\r\n{{"
            );
            let _ = stream.write_all(answer.as_bytes());
            return;
        }
        requested.store(true, Ordering::SeqCst);
        match body {
            Body::Withheld => {
                let answer = format!("HTTP/1.1 200 OK\r\ncontent-length: {ANNOUNCED_LEN}\r\n\r\n");
                let _ = stream.write_all(answer.as_bytes());
                let mut buf = [0u8; 256];
                while matches!(stream.read(&mut buf), Ok(read) if read > 0) {}
                closed.store(true, Ordering::SeqCst);
            }
            Body::Cut => {
                let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nconnection: close\r\n\r\n");
                let _ = stream.write_all(&[b'{'; ANNOUNCED_LEN / 4]);
            }
        }
    }

    /// A request's head off `stream`, up to and including the blank line.
    fn request_head(stream: &mut TcpStream) -> String {
        let mut head = Vec::new();
        let mut buf = [0u8; 1024];
        while !head.windows(4).any(|window| window == b"\r\n\r\n") {
            match stream.read(&mut buf) {
                Ok(read) if read > 0 => head.extend_from_slice(&buf[..read]),
                _ => break,
            }
        }
        String::from_utf8_lossy(&head).into_owned()
    }

    /// `HF_HOME` and `HF_ENDPOINT` pointed at a test's cache and Hub while
    /// this lives, then put back. The variables are process-wide: the tests
    /// that set them take turns.
    struct HubEnv {
        previous: Vec<(&'static str, Option<OsString>)>,
        _turn: MutexGuard<'static, ()>,
    }

    static HUB_ENV_TURN: Mutex<()> = Mutex::new(());

    impl HubEnv {
        fn set(home: &Path, endpoint: &str) -> Self {
            let turn = HUB_ENV_TURN
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let previous = ["HF_HOME", "HF_ENDPOINT"]
                .into_iter()
                .map(|name| (name, std::env::var_os(name)))
                .collect();
            std::env::set_var("HF_HOME", home);
            std::env::set_var("HF_ENDPOINT", endpoint);
            Self {
                previous,
                _turn: turn,
            }
        }
    }

    impl Drop for HubEnv {
        fn drop(&mut self) {
            for (name, value) in self.previous.drain(..) {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }

    /// Whether `condition` holds within `wait`, the runtime running meanwhile
    /// (the client closes a dropped request's connection from a task of its
    /// own, which needs the runtime).
    async fn holds_within(wait: Duration, mut condition: impl FnMut() -> bool) -> bool {
        let until = Instant::now() + wait;
        loop {
            if condition() {
                return true;
            }
            if Instant::now() >= until {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// A Hub that answers the metadata request and then withholds the file's
    /// body holds the fetch for its bound and nothing past it: the request is
    /// dropped with the fetch, so the Hub sees the connection close, the
    /// runtime keeps no task of the fetch, and nothing of the file is cached.
    /// (The client's own download would leave its chunk task behind, reading
    /// the body that never comes for as long as the Hub holds the connection,
    /// one such task per fetch that ran out of its bound.)
    #[tokio::test]
    async fn a_fetch_whose_body_the_hub_withholds_ends_at_its_bound_with_nothing_behind() {
        let hub = StubHub::start(Body::Withheld);
        let home = tempfile::tempdir().expect("a cache directory");
        let _env = HubEnv::set(home.path(), &hub.endpoint);
        let metrics = tokio::runtime::Handle::current().metrics();
        let tasks = metrics.num_alive_tasks();

        let bound = Duration::from_secs(2);
        let failed = resolve_within("cell/withheld", bound)
            .await
            .expect_err("the body never comes");
        let message = format!("{failed:#}");
        assert!(
            hub.requested.load(Ordering::SeqCst),
            "the Hub had the file's request before the bound elapsed: {message}"
        );
        assert!(HubFetchError::is(&failed), "{message}");
        assert!(
            message.contains("config.json") && message.contains("no answer within 2s"),
            "the error names the file and the elapsed bound: {message}"
        );

        let grace = Duration::from_secs(5);
        let closed = holds_within(grace, || hub.closed.load(Ordering::SeqCst)).await;
        assert!(
            closed,
            "the Hub still holds the fetch's connection {grace:?} after the bound, \
             with {} task(s) alive on the runtime against {tasks} before the fetch",
            metrics.num_alive_tasks()
        );
        assert!(
            holds_within(grace, || metrics.num_alive_tasks() == tasks).await,
            "{} task(s) alive on the runtime {grace:?} after the bound, {tasks} before the fetch",
            metrics.num_alive_tasks()
        );
        assert!(
            !home
                .path()
                .join("hub/models--cell--withheld/snapshots")
                .exists(),
            "nothing of the file is in the cache; the next resolution fetches it again"
        );
    }

    /// A Hub whose answer to the file's request ends before the size its
    /// metadata announced (a close-delimited answer cut short, which the
    /// client reads as a complete body) is a failed fetch naming the sizes:
    /// nothing is written, so the next resolution asks again instead of the
    /// cache serving the cut file as `config.json` from then on.
    #[tokio::test]
    async fn a_body_cut_short_of_the_announced_size_is_not_cached() {
        let hub = StubHub::start(Body::Cut);
        let home = tempfile::tempdir().expect("a cache directory");
        let _env = HubEnv::set(home.path(), &hub.endpoint);

        let failed = resolve_within("cell/cut", Duration::from_secs(30))
            .await
            .expect_err("a cut body is not the file");
        let message = format!("{failed:#}");
        assert!(hub.requested.load(Ordering::SeqCst), "{message}");
        assert!(HubFetchError::is(&failed), "{message}");
        assert!(
            message.contains("config.json") && message.contains("the Hub sent 10 of 42 bytes"),
            "the error names the file and the sizes: {message}"
        );
        let repo = home.path().join("hub/models--cell--cut");
        assert!(
            !repo.join("blobs").exists() && !repo.join("snapshots").exists(),
            "nothing of the file is in the cache"
        );
    }
}
