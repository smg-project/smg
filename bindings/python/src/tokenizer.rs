//! The tokenizer SMG serves with, for Python: `llm-tokenizer`'s load, encode
//! and decode, and the gateway's own chat rendering, so that a training
//! pipeline gets the token ids serving feeds the model.
//!
//! A chat renders through the functions serving calls, from the request a
//! client would send to `/v1/chat/completions`: deserialized, normalized and
//! validated as the HTTP layer does, refused where the gRPC router refuses
//! it, its tools narrowed by `tool_choice`, rendered (content format,
//! tool-call arguments, template kwargs, the thinking toggle,
//! `continue_final_message`), and encoded as the gateway's tokenize step
//! does: the rendered text without special tokens, or the encode a
//! segment-aware renderer prepared, which a flat encode of the text does not
//! reproduce. The request renders under the tokenizer's model id, as the
//! router renders it under the served model's id: that id selects the
//! provider profile whose rewrites apply.
//!
//! Short work runs on the calling thread with the GIL held. Work that can
//! reach rayon (loading, batches, long texts) runs on a pool this module
//! owns, one per process (see [`pool`]), with the GIL released, and
//! `os.fork()` waits until no such call is running (see [`before_fork`]).

use std::{
    cell::{Cell, RefCell},
    env,
    fmt::{self, Write as _},
    fs,
    panic::{self, AssertUnwindSafe},
    path::{self, Component, Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, LazyLock, Mutex, PoisonError,
    },
    thread,
    time::Duration,
};

use llm_tokenizer::{
    factory::{create_tokenizer_with_chat_template_blocking, is_likely_openai_model},
    hub::{download_tokenizer_from_hf, is_tokenizer_file},
    traits::Tokenizer as TokenizerTrait,
    Encoding, HuggingFaceTokenizer, PromptEncoding, TiktokenTokenizer,
};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    common::ContentPart,
    profile::ProviderProfile,
    validated::Normalizable,
};
use pyo3::{
    exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use serde_json::Value;
use smg::{
    routers::grpc::{
        is_harmony_card, text_only_output_refusal,
        utils::{process_chat_messages_with_encoding, request_as_rendered},
        ProcessedMessages,
    },
    version,
};
use tracing::{
    field::{Field, Visit},
    level_filters::LevelFilter,
    span,
    subscriber::Interest,
    Dispatch, Event, Level, Metadata, Subscriber,
};
use validator::Validate;

/// Texts shorter than this encode on the calling thread, with the GIL held.
/// llm-tokenizer's native encoder splits a text of 64 KiB or more across
/// rayon threads, so a text that might reach that runs on the pool instead
/// (see [`pool`]); a quarter of that size keeps the inline path clear of it
/// with room to spare. The GIL stays held for the short work: a thread that
/// releases it takes it back only after the thread running Python in the
/// meantime gives it up, up to the interpreter's switch interval (5 ms)
/// later, a thousand times a short encode's own time; a long one releases it
/// for the pool.
const INLINE_ENCODE_MAX_BYTES: usize = 16 * 1024;

/// Id sequences shorter than this decode with the GIL held (see
/// [`INLINE_ENCODE_MAX_BYTES`]); a longer one decodes with it released. A
/// single decode reaches no rayon at any length.
const INLINE_DECODE_MAX_IDS: usize = 4096;

/// Why a call failed: its input (`ValueError`), or SMG and its environment
/// (`RuntimeError`: a checkpoint that does not load, a pool that cannot
/// start, a panic).
enum Failure {
    Input(String),
    Internal(String),
}

impl Failure {
    /// The failure of item `index` of the batch argument `argument`.
    fn of_item(self, argument: &str, index: usize) -> Self {
        match self {
            Failure::Input(message) => Failure::Input(format!("{argument}[{index}]: {message}")),
            Failure::Internal(message) => {
                Failure::Internal(format!("{argument}[{index}]: {message}"))
            }
        }
    }
}

impl From<Failure> for PyErr {
    fn from(failure: Failure) -> Self {
        match failure {
            Failure::Input(message) => PyValueError::new_err(message),
            Failure::Internal(message) => PyRuntimeError::new_err(message),
        }
    }
}

/// `work`, with a panic caught as its failure. pyo3 raises an uncaught panic
/// as a `PanicException`, which `except Exception` does not catch, so one
/// request that trips a renderer would end a data job; serving fails that
/// request alone, and so does this.
fn caught<T>(work: impl FnOnce() -> Result<T, Failure>) -> Result<T, Failure> {
    panic::catch_unwind(AssertUnwindSafe(work)).unwrap_or_else(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .map(|message| (*message).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "a panic without a message".to_string());
        Err(Failure::Internal(format!(
            "SMG's tokenizer panicked: {message}"
        )))
    })
}

// ---------------------------------------------------------------------------
// Fork safety: the calls in flight, and the hooks around `os.fork()`
// ---------------------------------------------------------------------------

/// The tokenizer calls running with the GIL released, and the process they
/// run in: the process id in the high half, the count in the low one. A
/// count another process left behind (copied into a child forked from C,
/// past the hooks below) counts for nothing.
static IN_FLIGHT: AtomicU64 = AtomicU64::new(0);

/// The `os.fork()`s between [`before_fork`] and the hook after the fork,
/// kept as [`IN_FLIGHT`] keeps its count: while one is, a call keeps the GIL
/// (see [`detached`]).
static FORKING: AtomicU64 = AtomicU64::new(0);

/// The count `state` keeps for process `pid`.
fn count_for(state: u64, pid: u32) -> u64 {
    if state >> 32 == u64::from(pid) {
        state & u64::from(u32::MAX)
    } else {
        0
    }
}

/// One more in the count `counter` keeps for process `pid`.
fn count_up(counter: &AtomicU64, pid: u32) {
    counter
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |state| {
            Some(u64::from(pid) << 32 | (count_for(state, pid) + 1))
        })
        .ok();
}

/// One fewer in the count `counter` keeps for process `pid`, never below
/// none.
fn count_down(counter: &AtomicU64, pid: u32) {
    counter
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |state| {
            Some(u64::from(pid) << 32 | count_for(state, pid).saturating_sub(1))
        })
        .ok();
}

/// A tokenizer call running with the GIL released, counted from before the
/// GIL is released to the end of its work (see [`before_fork`]).
struct InFlight {
    pid: u32,
}

impl InFlight {
    /// Taken while the GIL is held.
    fn start(_py: Python<'_>) -> Self {
        let pid = std::process::id();
        count_up(&IN_FLIGHT, pid);
        Self { pid }
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        count_down(&IN_FLIGHT, self.pid);
    }
}

/// `work`, run with the GIL released and counted in flight until it ends,
/// before the GIL is taken back; or, while an `os.fork()` runs its hooks (see
/// [`before_fork`]), run with the GIL held, so that it ends before the fork,
/// which the thread holding the GIL makes. `work` never calls into Python.
fn detached<T: Send>(py: Python<'_>, work: impl FnOnce() -> T + Send) -> T {
    if count_for(FORKING.load(Ordering::SeqCst), std::process::id()) != 0 {
        return work();
    }
    let call = InFlight::start(py);
    py.detach(move || {
        let _call = call;
        work()
    })
}

/// Before `os.fork()`, with the GIL held: wait until no tokenizer call is
/// running, and until the fork is made, keep the GIL in every call that
/// starts. A call holds locks (the pool's slot, the gateway's tool cache, a
/// lazily built table half initialized) that a child forked in its middle
/// would inherit held, with no thread left to release them, and the child's
/// next call would wait forever. A call starts only while holding the GIL
/// and needs no GIL to finish, so the calls running drain. The `before`
/// hooks registered ahead of this one (logging's and `concurrent.futures`',
/// at their import) run after it, and one that releases the GIL lets another
/// thread start a call before the fork: that call keeps the GIL (see
/// [`detached`]), so it ends before the fork, which the thread holding the
/// GIL makes. A fork made from C, past `os.fork()`, does not wait.
#[pyfunction]
fn before_fork() {
    let pid = std::process::id();
    count_up(&FORKING, pid);
    while count_for(IN_FLIGHT.load(Ordering::SeqCst), pid) != 0 {
        thread::sleep(Duration::from_micros(100));
    }
}

/// In the parent after `os.fork()`, which calls it whether or not the fork
/// was made: calls release the GIL again.
#[pyfunction]
fn after_fork_in_parent() {
    count_down(&FORKING, std::process::id());
}

/// Forks this process went through by `os.fork()`, counted in the child: a
/// pool built before one has no threads after it, even when the child's
/// process id is a recycled one of its parent's line.
static FORKS: AtomicU64 = AtomicU64::new(0);

/// In the child of `os.fork()`, whose one thread, the one that forked, is
/// in no call and in no other fork.
#[pyfunction]
fn after_fork_in_child() {
    IN_FLIGHT.store(0, Ordering::SeqCst);
    FORKING.store(0, Ordering::SeqCst);
    FORKS.fetch_add(1, Ordering::SeqCst);
}

/// Whether this process registered [`before_fork`],
/// [`after_fork_in_parent`] and [`after_fork_in_child`].
static FORK_HOOKS: AtomicBool = AtomicBool::new(false);

/// Register [`before_fork`], [`after_fork_in_parent`] and
/// [`after_fork_in_child`] with `os.register_at_fork`, once per process, when
/// its first tokenizer loads. Checked and registered with the GIL held
/// throughout, so that no fork falls in between: a once-cell would release
/// the GIL while it initializes, and a child forked then would find the cell
/// initializing forever. Two threads that race to the first load register at
/// worst twice, which waits twice and counts a fork twice.
fn register_fork_hooks(py: Python<'_>) -> PyResult<()> {
    if FORK_HOOKS.load(Ordering::SeqCst) {
        return Ok(());
    }
    let os = py.import("os")?;
    if os.hasattr("register_at_fork")? {
        let hooks = PyDict::new(py);
        hooks.set_item("before", wrap_pyfunction!(before_fork, py)?)?;
        hooks.set_item(
            "after_in_parent",
            wrap_pyfunction!(after_fork_in_parent, py)?,
        )?;
        hooks.set_item("after_in_child", wrap_pyfunction!(after_fork_in_child, py)?)?;
        os.call_method("register_at_fork", (), Some(&hooks))?;
    }
    FORK_HOOKS.store(true, Ordering::SeqCst);
    Ok(())
}

// ---------------------------------------------------------------------------
// The thread pool
// ---------------------------------------------------------------------------

/// The process a pool was built in: its id and its count of forks.
type Owner = (u32, u64);

thread_local! {
    /// This thread's copy of the current process's pool, so that only the
    /// first call on a thread (and the first after a fork) takes [`POOL`].
    static THREAD_POOL: Cell<Option<(Owner, &'static ThreadPool)>> = const { Cell::new(None) };
}

/// The current process's pool.
static POOL: Mutex<Option<(Owner, &'static ThreadPool)>> = Mutex::new(None);

/// The rayon pool for every call that can reach rayon, one per process.
///
/// llm-tokenizer runs rayon work when a native-path tokenizer loads (its
/// vocabulary seed), when the native encoder meets a text of 64 KiB or more,
/// and in the `tokenizers` pipeline's batch encode. Started from a thread
/// outside any pool, that work creates rayon's global pool, and a child
/// forked afterwards (a torch DataLoader or `datasets.map` worker) does not
/// inherit its threads: the child's next rayon call waits forever. Run here,
/// the work never creates the global pool, and a forked child, finding the
/// pool built in another process, builds its own. The parent's pool is left
/// alone, never dropped: its threads are not in the child, and shutting it
/// down would take their sleep locks, which one of them may have held at the
/// fork.
fn pool() -> Result<&'static ThreadPool, Failure> {
    let owner = (std::process::id(), FORKS.load(Ordering::SeqCst));
    if let Some((built_in, pool)) = THREAD_POOL.get() {
        if built_in == owner {
            return Ok(pool);
        }
    }
    let current = *POOL.lock().unwrap_or_else(PoisonError::into_inner);
    let pool = match current {
        Some((built_in, pool)) if built_in == owner => pool,
        _ => {
            // Built outside the lock, which a fork must never find held; of
            // two threads that build at once, the second drops its own pool,
            // idle.
            let built = build_pool()?;
            let mut slot = POOL.lock().unwrap_or_else(PoisonError::into_inner);
            match *slot {
                Some((built_in, pool)) if built_in == owner => pool,
                _ => {
                    let pool: &'static ThreadPool = Box::leak(Box::new(built));
                    *slot = Some((owner, pool));
                    pool
                }
            }
        }
    };
    THREAD_POOL.set(Some((owner, pool)));
    Ok(pool)
}

/// A pool of `SMG_TOKENIZER_THREADS` threads; of one when
/// `TOKENIZERS_PARALLELISM` is off, read as Hugging Face's tokenizers read
/// it; else of rayon's default size (`RAYON_NUM_THREADS`, else the CPU
/// count).
fn build_pool() -> Result<ThreadPool, Failure> {
    let threads = match env::var("SMG_TOKENIZER_THREADS") {
        Ok(value) => match value.trim().parse::<usize>() {
            Ok(threads) if threads > 0 => threads,
            _ => {
                return Err(Failure::Internal(format!(
                    "SMG_TOKENIZER_THREADS must be a positive number of threads, not {value:?}"
                )))
            }
        },
        Err(_) => {
            let parallelism_off = env::var("TOKENIZERS_PARALLELISM").is_ok_and(|value| {
                matches!(
                    value.to_ascii_lowercase().as_str(),
                    "" | "off" | "false" | "f" | "no" | "n" | "0"
                )
            });
            // One thread, or 0: rayon's default.
            usize::from(parallelism_off)
        }
    };
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|index| format!("smg-tokenizer-{index}"))
        .build()
        .map_err(|error| {
            Failure::Internal(format!(
                "starting the tokenizer's thread pool: {error} (SMG_TOKENIZER_THREADS sets its \
                 size)"
            ))
        })?;
    // A worker names its thread when it first runs, so right after `build`
    // a reader of the process's threads can find a worker still under its
    // parent's name. One broadcast returns once every worker has run.
    pool.broadcast(|_| ());
    Ok(pool)
}

/// Run `work` on the process's pool, where its rayon calls stay.
fn in_pool<T: Send>(work: impl FnOnce() -> T + Send) -> Result<T, Failure> {
    Ok(pool()?.install(work))
}

// ---------------------------------------------------------------------------
// The gateway's warnings during a render
// ---------------------------------------------------------------------------

/// The warnings the gateway logs while it renders a request (tool-call
/// `arguments` that are not a JSON object rendered as `{}`, another
/// provider's message extensions dropped, deprecated fields migrated),
/// which serving writes to its log and a Python caller would never see:
/// collected for the render running on the current thread (see
/// [`captured`]).
struct WarningCapture;

thread_local! {
    /// The warnings of the render running on this thread, while one does.
    static CAPTURED: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
}

/// One dispatcher for the process: creating one per render would rebuild
/// every callsite's interest under tracing's global lock each time.
static WARNINGS: LazyLock<Dispatch> = LazyLock::new(|| Dispatch::new(WarningCapture));

impl Subscriber for WarningCapture {
    fn register_callsite(&self, metadata: &'static Metadata<'static>) -> Interest {
        if self.enabled(metadata) {
            Interest::sometimes()
        } else {
            Interest::never()
        }
    }

    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.is_event() && *metadata.level() <= Level::WARN
    }

    fn max_level_hint(&self) -> Option<LevelFilter> {
        Some(LevelFilter::WARN)
    }

    fn new_span(&self, _: &span::Attributes<'_>) -> span::Id {
        span::Id::from_u64(1)
    }

    fn record(&self, _: &span::Id, _: &span::Record<'_>) {}

    fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}

    fn event(&self, event: &Event<'_>) {
        let mut line = WarningLine::default();
        event.record(&mut line);
        CAPTURED.with_borrow_mut(|captured| {
            if let Some(captured) = captured {
                captured.push(line.finish());
            }
        });
    }

    fn enter(&self, _: &span::Id) {}

    fn exit(&self, _: &span::Id) {}
}

/// A warning as one line: its message, then its fields in parentheses.
#[derive(Default)]
struct WarningLine {
    message: String,
    fields: String,
}

impl WarningLine {
    fn field(&mut self, field: &Field, value: fmt::Arguments<'_>) {
        if field.name() == "message" {
            self.message.write_fmt(value).ok();
        } else {
            if !self.fields.is_empty() {
                self.fields.push_str(", ");
            }
            write!(self.fields, "{}={value}", field.name()).ok();
        }
    }

    fn finish(self) -> String {
        if self.fields.is_empty() {
            self.message
        } else {
            format!("{} ({})", self.message, self.fields)
        }
    }
}

impl Visit for WarningLine {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.field(field, format_args!("{value}"));
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.field(field, format_args!("{value:?}"));
    }
}

/// `work`, run with the gateway's warnings collected (see
/// [`WarningCapture`]); `work` must not panic (see [`caught`]).
fn captured<T>(work: impl FnOnce() -> T) -> (T, Vec<String>) {
    let outer = CAPTURED.replace(Some(Vec::new()));
    let result = tracing::dispatcher::with_default(&WARNINGS, work);
    let warnings = CAPTURED.replace(outer).unwrap_or_default();
    (result, warnings)
}

// ---------------------------------------------------------------------------
// Encode, decode, load
// ---------------------------------------------------------------------------

/// The ids of an encoding, without copying the plain variants' vectors.
fn into_ids(encoding: Encoding) -> Vec<u32> {
    match encoding {
        Encoding::Plain(ids) | Encoding::Tiktoken(ids) => ids,
        Encoding::Hf(encoding) => encoding.get_ids().to_vec(),
    }
}

fn encode_ids(
    tokenizer: &dyn TokenizerTrait,
    text: &str,
    add_special_tokens: bool,
) -> Result<Vec<u32>, Failure> {
    caught(|| {
        tokenizer
            .encode(text, add_special_tokens)
            .map(into_ids)
            .map_err(|error| Failure::Input(format!("{error:#}")))
    })
}

fn decode_ids(
    tokenizer: &dyn TokenizerTrait,
    token_ids: &[u32],
    skip_special_tokens: bool,
) -> Result<String, Failure> {
    caught(|| {
        tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| Failure::Input(format!("{error:#}")))
    })
}

/// A path made absolute. `path::absolute` keeps symlinks, so a file in a
/// Hugging Face cache snapshot keeps its sibling files; made absolute, a
/// directory named `mock` or `test` loads as the checkpoint it is (the
/// factory reads those two literal paths as its mock tokenizer).
fn absolute(path: &Path) -> Result<String, Failure> {
    path::absolute(path)
        .map_err(|error| Failure::Input(format!("resolving {}: {error}", path.display())))?
        .into_os_string()
        .into_string()
        .map_err(|path| Failure::Input(format!("{} is not valid UTF-8", path.display())))
}

/// The chat template file, made absolute; the gateway's `--chat-template`
/// takes an existing file too.
fn resolve_chat_template(path: &str) -> Result<String, Failure> {
    if !Path::new(path).is_file() {
        return Err(Failure::Input(format!(
            "chat template file not found: {path}"
        )));
    }
    absolute(Path::new(path))
}

/// Where a path is in a snapshot of the Hugging Face cache
/// (`.../models--{org}--{name}/snapshots/{revision}`, or a path in one): the
/// Hub model id, as the cache spells it (a Hub name holds no `--`, so the
/// first one is the `/` between the owner and the name), and the path inside
/// the snapshot, empty for the snapshot itself. `..` counts by name, as
/// [`path::absolute`] leaves it: a snapshot is a real directory.
fn hub_snapshot(path: &Path) -> Option<(String, String)> {
    let mut components: Vec<&str> = Vec::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if components.last().is_some_and(|last| *last != "/") {
                    components.pop();
                }
            }
            component => components.extend(component.as_os_str().to_str()),
        }
    }
    (0..components.len().saturating_sub(2))
        .rev()
        .find_map(|start| match &components[start..start + 3] {
            [repository, "snapshots", _] => {
                let id = repository
                    .strip_prefix("models--")
                    .filter(|id| !id.is_empty())?;
                Some((id.replacen("--", "/", 1), components[start + 3..].join("/")))
            }
            _ => None,
        })
}

/// A Hub model id's tokenizer files, downloaded as the gateway downloads
/// them (`HF_TOKEN`, `HF_HOME` and `HF_ENDPOINT` honoured, the revision
/// `main`): the directory of the snapshot.
fn download(model_id: &str) -> Result<String, Failure> {
    let failed = |error: &dyn fmt::Display| {
        Failure::Internal(format!(
            "loading the tokenizer from {model_id}: Failed to download tokenizer from \
             HuggingFace: {error}"
        ))
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| failed(&error))?;
    let dir = runtime
        .block_on(download_tokenizer_from_hf(model_id))
        .map_err(|error| failed(&format_args!("{error:#}")))?;
    absolute(&dir)
}

/// The checkpoint fields serving's Harmony rule reads off a worker's model
/// card, from the `config.json` beside the tokenizer: its architectures and
/// its model type. A source that is no path (a model name) has none.
fn read_card(path: &Path) -> (Vec<String>, Option<String>) {
    let dir = if path.is_dir() {
        Some(path)
    } else if path.is_file() {
        path.parent()
    } else {
        None
    };
    let config = dir
        .and_then(|dir| fs::read_to_string(dir.join("config.json")).ok())
        .and_then(|text| serde_json::from_str::<Value>(&text).ok());
    let Some(config) = config else {
        return (Vec::new(), None);
    };
    let architectures = config
        .get("architectures")
        .and_then(Value::as_array)
        .map(|architectures| {
            architectures
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .map(str::to_string);
    (architectures, model_type)
}

/// A digest of what a tokenizer's behaviour follows from: the smg build, the
/// model id requests render under, the chat template file, the file a file
/// source names, and, by name and content, the checkpoint's files the
/// gateway's own Hub download fetches (its tokenizer files, `config.json`,
/// `generation_config.json`) with `encoding/encoding.py`, which DeepSeek
/// V4.1's renderer reads. Two tokenizers with one fingerprint render and
/// encode alike; a file edited in place changes it.
fn fingerprint(
    location: &str,
    chat_template: Option<&str>,
    model: Option<&str>,
) -> Result<String, Failure> {
    let read = |path: &Path| {
        fs::read(path)
            .map_err(|error| Failure::Internal(format!("reading {}: {error}", path.display())))
    };
    let mut hasher = blake3::Hasher::new();
    let mut field = |name: &str, bytes: &[u8]| {
        for part in [name.as_bytes(), bytes] {
            hasher.update(&(part.len() as u64).to_le_bytes());
            hasher.update(part);
        }
    };
    field("smg_build", PyTokenizer::smg_build().as_bytes());
    if let Some(model) = model {
        field("model", model.as_bytes());
    }
    if let Some(path) = chat_template {
        field("chat_template", &read(Path::new(path))?);
    }
    let path = Path::new(location);
    let (dir, source_file) = if path.is_dir() {
        (path, None)
    } else if let (true, Some(dir), Some(name)) = (
        path.is_file(),
        path.parent(),
        path.file_name().and_then(|name| name.to_str()),
    ) {
        // The file loads whatever its name; and unlike a directory source, a
        // file source discovers no chat template file beside it.
        field("source_file", name.as_bytes());
        (dir, Some(name))
    } else {
        // A model name tiktoken loads: no files.
        field("source", location.as_bytes());
        return Ok(hasher.finalize().to_hex().to_string());
    };
    let entries = fs::read_dir(dir)
        .map_err(|error| Failure::Internal(format!("reading {}: {error}", dir.display())))?;
    let mut files: Vec<(String, PathBuf)> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| Some((entry.file_name().into_string().ok()?, entry.path())))
        .filter(|(name, path)| {
            path.is_file()
                && (is_tokenizer_file(name)
                    || name == "config.json"
                    || name == "generation_config.json"
                    || source_file == Some(name.as_str()))
        })
        .collect();
    let encoding = dir.join("encoding").join("encoding.py");
    if encoding.is_file() {
        files.push(("encoding/encoding.py".to_string(), encoding));
    }
    files.sort();
    for (name, path) in files {
        field(&name, &read(&path)?);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

/// What chat rendering reads beyond the tokenizer: the model id requests
/// render under, and the checkpoint's card for serving's Harmony rule.
struct Chat {
    model: Option<String>,
    architectures: Vec<String>,
    model_type: Option<String>,
}

/// The request as the HTTP layer reads it (`ValidatedJson`): parsed into the
/// typed request directly, and parsed again with the failing field's path
/// tracked only when that fails. It renders under `model` when that is set,
/// as the router renders a request under the served model's id; a request
/// that names no model takes it, one that names another is rewritten to it.
fn parse_request(text: &str, model: Option<&str>) -> Result<ChatCompletionRequest, Failure> {
    let mut request = match serde_json::from_str::<ChatCompletionRequest>(text) {
        Ok(request) => request,
        Err(_) if names_no_model(text) => {
            let Some(model) = model else {
                return Err(Failure::Input(
                    "the request names no model, and this tokenizer has no model id to render \
                     it under: pass Tokenizer(..., model=...), the id SMG serves this checkpoint \
                     as (it selects the provider profile, DeepSeek, Kimi, MiniMax or GLM, whose \
                     rewrites apply), or name the model in the request"
                        .to_string(),
                ));
            };
            let text = with_model(text, model);
            serde_json::from_str(&text).map_err(|_| refused(&text))?
        }
        Err(_) => return Err(refused(text)),
    };
    if let Some(model) = model {
        request.model = model.to_string();
    }
    Ok(request)
}

/// Whether `text` is a JSON object without a `model` member.
fn names_no_model(text: &str) -> bool {
    matches!(
        serde_json::from_str::<Value>(text),
        Ok(Value::Object(members)) if !members.contains_key("model")
    )
}

/// `text`, a JSON object, with `model` as its first member: the rest of the
/// text parses as it was, duplicate members refused as before.
fn with_model(text: &str, model: &str) -> String {
    let rest = text.trim_start();
    let rest = rest.strip_prefix('{').unwrap_or(rest);
    let separator = if rest.trim_start().starts_with('}') {
        ""
    } else {
        ","
    };
    format!("{{\"model\":{}{separator}{rest}", Value::from(model))
}

/// Why the typed request refuses `text`, with the path of the failing field
/// (the HTTP layer's second parse).
fn refused(text: &str) -> Failure {
    let deserializer = &mut serde_json::Deserializer::from_str(text);
    let error =
        match serde_path_to_error::deserialize::<_, ChatCompletionRequest>(&mut *deserializer) {
            Ok(_) => match deserializer.end() {
                Ok(()) => "the request does not parse".to_string(),
                Err(error) => error.to_string(),
            },
            Err(error) => error.to_string(),
        };
    Failure::Input(format!("SMG does not accept this request: {error}"))
}

/// Refuse media content: SMG replaces each image, video or audio part with
/// the model's own tokens for that item when it serves the request, which a
/// tokenizer alone cannot reproduce. Every media part is refused, those
/// serving renders as text included (an assistant message's, a template's
/// that writes media as text): a superset of serving's expansion, and the
/// match on the part types fails to build when a new one appears.
fn refuse_media(messages: &[ChatMessage]) -> Result<(), Failure> {
    for (index, message) in messages.iter().enumerate() {
        let content = match message {
            ChatMessage::System { content, .. }
            | ChatMessage::User { content, .. }
            | ChatMessage::Tool { content, .. }
            | ChatMessage::Developer { content, .. }
            | ChatMessage::Root { content, .. } => Some(content),
            ChatMessage::Assistant { content, .. } => content.as_ref(),
            ChatMessage::Function { .. } => None,
        };
        let Some(MessageContent::Parts(parts)) = content else {
            continue;
        };
        for part in parts {
            let kind = match part {
                ContentPart::ImageUrl { .. } => "image_url",
                ContentPart::AudioUrl { .. } => "audio_url",
                ContentPart::InputAudio { .. } => "input_audio",
                ContentPart::VideoUrl { .. } => "video_url",
                ContentPart::Text { .. }
                | ContentPart::Refusal { .. }
                | ContentPart::Unknown(_) => continue,
            };
            return Err(Failure::Input(format!(
                "messages[{index}] has {kind} content; SMG expands media into the model's own \
                 tokens when it serves a request, which this tokenizer does not reproduce, so \
                 it renders text-only conversations"
            )));
        }
    }
    Ok(())
}

/// Render one chat completion request (JSON text) as serving does, up to the
/// tokenize step: the HTTP layer's steps (deserialize, normalize: the
/// provider profile's rewrites, the deprecated fields, the `tool_choice`
/// default; validate), the gRPC router's refusal of output other than text,
/// this binding's own refusals of what a tokenizer cannot reproduce (a model
/// serving renders with Harmony, media content), then the chat preparation
/// stage's: the tools `tool_choice` renders, and the rendering.
fn prepare(
    tokenizer: &dyn TokenizerTrait,
    chat: &Chat,
    text: &str,
) -> Result<(ProcessedMessages, PromptEncoding, bool), Failure> {
    let mut request = parse_request(text, chat.model.as_deref())?;
    request.normalize();
    request
        .validate()
        .map_err(|error| Failure::Input(format!("SMG rejects this request: {error}")))?;
    if let Some((_, message)) = text_only_output_refusal(&request) {
        return Err(Failure::Input(format!(
            "SMG rejects this request: {message}"
        )));
    }
    if is_harmony_card(
        &chat.architectures,
        chat.model_type.as_deref(),
        &request.model,
    ) {
        return Err(Failure::Input(format!(
            "SMG renders gpt-oss models with Harmony, not the chat template, so this tokenizer \
             cannot render requests for them (model {})",
            request.model
        )));
    }
    refuse_media(&request.messages)?;
    let continues_final_assistant = request.continues_final_assistant();
    let request = request_as_rendered(&request);
    let (processed, encoding) =
        process_chat_messages_with_encoding(&request, tokenizer, None).map_err(Failure::Input)?;
    Ok((processed, encoding, continues_final_assistant))
}

/// The gateway's tokenize step: the text without special tokens (the
/// template wrote those it renders), or the renderer's own encode.
fn encode_prompt(
    tokenizer: &dyn TokenizerTrait,
    text: &str,
    encoding: PromptEncoding,
) -> Result<Vec<u32>, Failure> {
    match encoding {
        PromptEncoding::FromText => encode_ids(tokenizer, text, false),
        PromptEncoding::Deferred(job) => caught(|| {
            job.run()
                .map(into_ids)
                .map_err(|error| Failure::Input(format!("{error:#}")))
        }),
    }
}

/// A rendered chat, as `render_chat` hands it to Python: the prompt text,
/// its token ids (`None` when not asked for), whether those are the text's
/// flat encode, whether the renderer wrote its generation header, whether
/// it kept the final message open (a native continuation), the warnings the
/// gateway logged, and whether the request continues a final assistant
/// message (`continue_final_message` with the assistant's message last).
type Rendered = (
    String,
    Option<Vec<u32>>,
    bool,
    bool,
    bool,
    Vec<String>,
    bool,
);

fn rendered(
    processed: ProcessedMessages,
    token_ids: Option<Vec<u32>>,
    from_text: bool,
    warnings: Vec<String>,
    continues_final_assistant: bool,
) -> Rendered {
    (
        processed.text,
        token_ids,
        from_text,
        processed.generation_prompt,
        processed.continued_final_message,
        warnings,
        continues_final_assistant,
    )
}

/// Render and encode one request on the current thread (a batch's item, on
/// the pool).
fn render_one(
    tokenizer: &dyn TokenizerTrait,
    chat: &Chat,
    text: &str,
    tokenize: bool,
) -> Result<Rendered, Failure> {
    let (result, warnings) = captured(|| {
        caught(|| {
            let (processed, encoding, continues) = prepare(tokenizer, chat, text)?;
            let from_text = matches!(encoding, PromptEncoding::FromText);
            let token_ids = tokenize
                .then(|| encode_prompt(tokenizer, &processed.text, encoding))
                .transpose()?;
            Ok((processed, token_ids, from_text, continues))
        })
    });
    let (processed, token_ids, from_text, continues) = result?;
    Ok(rendered(
        processed, token_ids, from_text, warnings, continues,
    ))
}

/// The name of the provider profile a model id selects.
fn profile_name(model: &str) -> &'static str {
    match ProviderProfile::for_model(model) {
        ProviderProfile::Generic => "generic",
        ProviderProfile::OpenAi => "openai",
        ProviderProfile::Kimi => "kimi",
        ProviderProfile::Minimax => "minimax",
        ProviderProfile::Zai => "zai",
        ProviderProfile::DeepSeek => "deepseek",
    }
}

/// The tokenizer tiktoken builds for an OpenAI model name, built on the pool
/// as the factory builds it; `None` for any other name. That includes a name
/// the factory's guess ([`is_likely_openai_model`]) takes for OpenAI's but
/// tiktoken does not know (`org/gpt-2-small`), which the factory would
/// download from the Hub itself, on the pool, and keep as the source: here it
/// is a Hub id like any other.
fn openai_tokenizer(py: Python<'_>, name: &str) -> PyResult<Option<Arc<dyn TokenizerTrait>>> {
    if !is_likely_openai_model(name) {
        return Ok(None);
    }
    let tokenizer = detached(py, || {
        in_pool(|| caught(|| Ok(TiktokenTokenizer::from_model_name(name).ok())))
    })??;
    Ok(tokenizer.map(|tokenizer| Arc::new(tokenizer) as Arc<dyn TokenizerTrait>))
}

/// The Rust half of `smg.tokenizer.Tokenizer`; see that class for the API.
#[pyclass(name = "Tokenizer", module = "smg.smg_rs", frozen)]
pub struct PyTokenizer {
    inner: Arc<dyn TokenizerTrait>,
    source: String,
    chat_template: Option<String>,
    hub: Option<(String, String)>,
    fingerprint: String,
    chat: Chat,
}

#[pymethods]
impl PyTokenizer {
    /// Load the tokenizer of `source` (a checkpoint directory, a tokenizer
    /// file, or a Hugging Face model id) as the gateway does, with the chat
    /// template file `chat_template` instead of the checkpoint's own when
    /// given. `model` is the model id chat requests render under; it
    /// defaults to the Hub id of a Hub source or of a Hugging Face cache
    /// snapshot, and to none for any other path.
    #[new]
    #[pyo3(signature = (source, chat_template=None, model=None))]
    fn new(
        py: Python<'_>,
        source: &str,
        chat_template: Option<&str>,
        model: Option<String>,
    ) -> PyResult<Self> {
        register_fork_hooks(py)?;
        let chat_template = chat_template.map(resolve_chat_template).transpose()?;
        let path = Path::new(source);
        // A path, made absolute; a name that can only be a path (absolute,
        // relative to `.` or `~`, or of more than two segments), refused when
        // it is not there, rather than read as a model name; an OpenAI model
        // name tiktoken knows, as given (see [`openai_tokenizer`]); any other
        // name, a Hub id, downloaded here and loaded from its snapshot, which a
        // pickle then names. The download releases the GIL but counts as no
        // call in flight: it holds none of this module's locks, and a stalled
        // one must not stall every fork; it stays off the pool, whose threads
        // it must not hold.
        let (location, hub, openai) = if path.exists() {
            // Read off the absolute path: `.` inside a snapshot is one too.
            let location = absolute(path)?;
            let hub = hub_snapshot(Path::new(&location));
            (location, hub, None)
        } else if path.is_absolute()
            || source.starts_with(['.', '~'])
            || source.matches('/').count() > 1
        {
            return Err(PyFileNotFoundError::new_err(format!(
                "no such file or directory: {source}"
            )));
        } else if let Some(tokenizer) = openai_tokenizer(py, source)? {
            (source.to_string(), None, Some(tokenizer))
        } else {
            let dir = py.detach(|| caught(|| download(source)))?;
            (dir, Some((source.to_string(), String::new())), None)
        };
        let model = model
            .or_else(|| hub.as_ref().map(|(id, _)| id.clone()))
            .or_else(|| (!Path::new(&location).exists()).then(|| location.clone()));
        let (inner, fingerprint) = detached(py, || {
            let inner = match openai {
                Some(tokenizer) => tokenizer,
                None => in_pool(|| {
                    caught(|| {
                        create_tokenizer_with_chat_template_blocking(
                            &location,
                            chat_template.as_deref(),
                        )
                        .map_err(|error| {
                            Failure::Internal(format!(
                                "loading the tokenizer from {location}: {error:#}"
                            ))
                        })
                    })
                })??,
            };
            let fingerprint =
                caught(|| fingerprint(&location, chat_template.as_deref(), model.as_deref()))?;
            Ok::<_, Failure>((inner, fingerprint))
        })?;
        let (architectures, model_type) = read_card(Path::new(&location));
        Ok(Self {
            inner,
            source: location,
            chat_template,
            hub,
            fingerprint,
            chat: Chat {
                model,
                architectures,
                model_type,
            },
        })
    }

    /// The source the tokenizer was loaded from: the absolute path of a
    /// local checkpoint or of the snapshot a Hub id downloaded to, or the
    /// OpenAI model name tiktoken loaded.
    #[getter]
    fn source(&self) -> &str {
        &self.source
    }

    /// The absolute path of the chat template file given at load, if any.
    #[getter]
    fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    /// The Hub model id of the snapshot the source is in, and the source's
    /// path inside it (empty for the snapshot itself), if it is in one.
    #[getter]
    fn hub_snapshot(&self) -> Option<(&str, &str)> {
        self.hub
            .as_ref()
            .map(|(id, inside)| (id.as_str(), inside.as_str()))
    }

    /// A digest of the smg build, the model id, the chat template file, a
    /// file source itself and the checkpoint's tokenizer files: tokenizers
    /// with one fingerprint render and encode alike (see [`fingerprint`]).
    #[getter]
    fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// The model id chat requests render under, if any.
    #[getter]
    fn model(&self) -> Option<&str> {
        self.chat.model.as_deref()
    }

    /// The provider profile the model id selects, if there is one.
    #[getter]
    fn provider_profile(&self) -> Option<&'static str> {
        self.chat.model.as_deref().map(profile_name)
    }

    /// The smg release and commit this binding was built from.
    #[classattr]
    fn smg_build() -> String {
        format!("{} ({})", version::VERSION, version::GIT_COMMIT)
    }

    /// The base vocabulary's size; for a Hugging Face tokenizer, without the
    /// added tokens, as `transformers` reports `vocab_size`.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// The vocabulary's size with the added tokens, as `len()` of a
    /// `transformers` tokenizer.
    fn __len__(&self) -> usize {
        match self.inner.as_any().downcast_ref::<HuggingFaceTokenizer>() {
            Some(tokenizer) => tokenizer.tokenizers_pipeline().get_vocab_size(true),
            None => self.inner.vocab_size(),
        }
    }

    /// The special tokens the checkpoint names, by role; the additional ones
    /// sorted by id.
    fn special_tokens<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let special = self.inner.get_special_tokens();
        let tokens = PyDict::new(py);
        tokens.set_item("bos_token", &special.bos_token)?;
        tokens.set_item("eos_token", &special.eos_token)?;
        tokens.set_item("unk_token", &special.unk_token)?;
        tokens.set_item("sep_token", &special.sep_token)?;
        tokens.set_item("pad_token", &special.pad_token)?;
        tokens.set_item("cls_token", &special.cls_token)?;
        tokens.set_item("mask_token", &special.mask_token)?;
        // The tokenizer keeps them in hash order, which differs between runs.
        let mut additional = special.additional_special_tokens.clone();
        additional.sort_by_cached_key(|token| {
            (
                self.inner.token_to_id(token).unwrap_or(u32::MAX),
                token.clone(),
            )
        });
        tokens.set_item("additional_special_tokens", additional)?;
        Ok(tokens)
    }

    /// The ids that end generation: `config.json` and
    /// `generation_config.json`'s `eos_token_id` with the tokenizer's own EOS.
    #[getter]
    fn eos_token_ids(&self) -> Vec<u32> {
        self.inner.eos_token_ids().to_vec()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, token_id: u32) -> Option<String> {
        self.inner.id_to_token(token_id)
    }

    fn encode(&self, py: Python<'_>, text: &str, add_special_tokens: bool) -> PyResult<Vec<u32>> {
        let tokenizer = &*self.inner;
        if text.len() < INLINE_ENCODE_MAX_BYTES {
            return Ok(encode_ids(tokenizer, text, add_special_tokens)?);
        }
        let ids = detached(py, || {
            in_pool(|| encode_ids(tokenizer, text, add_special_tokens))?
        })?;
        Ok(ids)
    }

    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        add_special_tokens: bool,
    ) -> PyResult<Vec<Vec<u32>>> {
        let tokenizer = &*self.inner;
        let ids = detached(py, || {
            in_pool(|| {
                texts
                    .par_iter()
                    .enumerate()
                    .map(|(index, text)| {
                        encode_ids(tokenizer, text, add_special_tokens)
                            .map_err(|failure| failure.of_item("texts", index))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })?
        })?;
        Ok(ids)
    }

    fn decode(
        &self,
        py: Python<'_>,
        token_ids: Vec<u32>,
        skip_special_tokens: bool,
    ) -> PyResult<String> {
        let tokenizer = &*self.inner;
        if token_ids.len() < INLINE_DECODE_MAX_IDS {
            return Ok(decode_ids(tokenizer, &token_ids, skip_special_tokens)?);
        }
        Ok(detached(py, || {
            decode_ids(tokenizer, &token_ids, skip_special_tokens)
        })?)
    }

    fn decode_batch(
        &self,
        py: Python<'_>,
        sequences: Vec<Vec<u32>>,
        skip_special_tokens: bool,
    ) -> PyResult<Vec<String>> {
        let tokenizer = &*self.inner;
        let texts = detached(py, || {
            in_pool(|| {
                sequences
                    .par_iter()
                    .enumerate()
                    .map(|(index, token_ids)| {
                        decode_ids(tokenizer, token_ids, skip_special_tokens)
                            .map_err(|failure| failure.of_item("sequences", index))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })?
        })?;
        Ok(texts)
    }

    /// Render a chat completion request, given as JSON text, as the gateway
    /// does; see `smg.tokenizer.Tokenizer.render_chat`.
    #[pyo3(signature = (request, tokenize=true))]
    fn render_chat(&self, py: Python<'_>, request: &str, tokenize: bool) -> PyResult<Rendered> {
        let tokenizer = &*self.inner;
        let chat = &self.chat;
        // The rendering reaches no rayon: a short request renders here with
        // the GIL held, a long one with it released.
        let render = || captured(|| caught(|| prepare(tokenizer, chat, request)));
        let (prepared, mut warnings) = if request.len() < INLINE_ENCODE_MAX_BYTES {
            render()
        } else {
            detached(py, render)
        };
        let (processed, encoding, continues) = prepared?;
        let from_text = matches!(encoding, PromptEncoding::FromText);
        let token_ids = if tokenize {
            // The prompt's encode can reach rayon by the prompt's length,
            // which a template can make many times the request's.
            let text = processed.text.as_str();
            let encode = || captured(|| encode_prompt(tokenizer, text, encoding));
            let (token_ids, more) = if text.len() < INLINE_ENCODE_MAX_BYTES {
                encode()
            } else {
                detached(py, || in_pool(encode))?
            };
            warnings.extend(more);
            Some(token_ids?)
        } else {
            // The renderer's encode is not needed: dropped unrun.
            drop(encoding);
            None
        };
        Ok(rendered(
            processed, token_ids, from_text, warnings, continues,
        ))
    }

    /// Render chat completion requests, given as JSON texts, in parallel.
    #[pyo3(signature = (requests, tokenize=true))]
    fn render_chat_batch(
        &self,
        py: Python<'_>,
        requests: Vec<String>,
        tokenize: bool,
    ) -> PyResult<Vec<Rendered>> {
        let tokenizer = &*self.inner;
        let chat = &self.chat;
        let rendered = detached(py, || {
            in_pool(|| {
                requests
                    .par_iter()
                    .enumerate()
                    .map(|(index, request)| {
                        render_one(tokenizer, chat, request, tokenize)
                            .map_err(|failure| failure.of_item("requests", index))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })?
        })?;
        Ok(rendered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hub_cache_snapshot_names_its_model_id() {
        let snapshot = Path::new("/home/u/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/0123abc");
        let deepseek = |inside: &str| {
            Some((
                "deepseek-ai/DeepSeek-V4-Flash".to_string(),
                inside.to_string(),
            ))
        };
        assert_eq!(hub_snapshot(snapshot), deepseek(""));
        assert_eq!(
            hub_snapshot(&snapshot.join("tokenizer.json")),
            deepseek("tokenizer.json")
        );
        assert_eq!(
            hub_snapshot(&snapshot.join("original/tokenizer.model")),
            deepseek("original/tokenizer.model")
        );
        assert_eq!(
            hub_snapshot(Path::new("/cache/models--gpt2/snapshots/abc")),
            Some(("gpt2".to_string(), String::new()))
        );
        assert_eq!(hub_snapshot(Path::new("/models/Qwen3-8B")), None);
        assert_eq!(hub_snapshot(Path::new("/models/snapshots/abc")), None);
        // `..` counts by name: out of the snapshot, no Hub id; into another
        // revision, that one's.
        assert_eq!(
            hub_snapshot(&snapshot.join("../../../../models/my-ckpt")),
            None
        );
        assert_eq!(hub_snapshot(&snapshot.join("../4567def")), deepseek(""));
    }

    #[test]
    fn a_missing_chat_template_is_refused() {
        let Err(Failure::Input(error)) = resolve_chat_template("/no/such/template.jinja") else {
            panic!("refused as an input error");
        };
        assert_eq!(
            error,
            "chat template file not found: /no/such/template.jinja"
        );
    }

    #[test]
    fn a_checkpoint_card_is_read_from_its_config() {
        let dir = tempfile::tempdir().expect("temp dir");
        let config = dir.path().join("config.json");
        fs::write(
            &config,
            r#"{"architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss"}"#,
        )
        .expect("config");
        let card = (
            vec!["GptOssForCausalLM".to_string()],
            Some("gpt_oss".to_string()),
        );
        assert_eq!(read_card(dir.path()), card);
        let file = dir.path().join("tokenizer.json");
        fs::write(&file, "{}").expect("tokenizer file");
        assert_eq!(read_card(&file), card);
        fs::write(&config, "not json").expect("config");
        assert_eq!(read_card(dir.path()), (Vec::new(), None));
        // A name that is no path reads no config.json, not even the one in
        // the directory it would sit in.
        fs::write(&config, r#"{"architectures": ["GptOssForCausalLM"]}"#).expect("config");
        assert_eq!(read_card(&dir.path().join("gpt-4o")), (Vec::new(), None));
    }

    #[test]
    fn the_fingerprint_follows_the_tokenizer_files_and_nothing_else() {
        let dir = tempfile::tempdir().expect("temp dir");
        let source = dir.path().to_str().expect("utf-8");
        fs::write(dir.path().join("tokenizer.json"), "{}").expect("tokenizer");
        fs::write(dir.path().join("config.json"), "{}").expect("config");
        let digest = |model: Option<&str>| fingerprint(source, None, model).ok().expect("digest");
        let first = digest(Some("m"));
        assert_eq!(digest(Some("m")), first);
        assert_ne!(digest(Some("n")), first);
        // Weights and other files are not the tokenizer's.
        fs::write(dir.path().join("model.safetensors"), "weights").expect("weights");
        fs::write(dir.path().join("README.md"), "readme").expect("readme");
        assert_eq!(digest(Some("m")), first);
        // A tokenizer file edited in place, or one added, is.
        fs::write(dir.path().join("config.json"), r#"{"eos_token_id": 2}"#).expect("config");
        let edited = digest(Some("m"));
        assert_ne!(edited, first);
        fs::create_dir(dir.path().join("encoding")).expect("encoding dir");
        fs::write(dir.path().join("encoding").join("encoding.py"), "x").expect("encoding");
        assert_ne!(digest(Some("m")), edited);
        // A file source is fingerprinted by its directory's files and by
        // itself, whatever its name: an in-place edit of `tok_v2.json` changes
        // it, and a file source is not its directory.
        let file = |name: &str| {
            let path = dir.path().join(name);
            fingerprint(path.to_str().expect("utf-8"), None, Some("m"))
                .ok()
                .expect("digest")
        };
        let tokenizer_json = file("tokenizer.json");
        assert_ne!(tokenizer_json, digest(Some("m")));
        fs::write(dir.path().join("tok_v2.json"), "{}").expect("tokenizer");
        let tok_v2 = file("tok_v2.json");
        assert_ne!(tok_v2, tokenizer_json);
        fs::write(dir.path().join("tok_v2.json"), r#"{"edited": 1}"#).expect("tokenizer");
        assert_ne!(file("tok_v2.json"), tok_v2);
        assert_eq!(file("tokenizer.json"), tokenizer_json);
    }

    #[test]
    fn media_parts_are_refused_and_text_parts_are_not() {
        let messages: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
        ]))
        .expect("messages");
        assert!(refuse_media(&messages[..1]).is_ok());
        let Err(Failure::Input(error)) = refuse_media(&messages) else {
            panic!("refused as an input error");
        };
        assert!(error.starts_with("messages[1] has image_url content;"));
    }

    #[test]
    fn a_request_renders_under_the_tokenizer_model() {
        let request = r#"{"model": "alias", "messages": [{"role": "user", "content": "hi"}]}"#;
        let parsed = parse_request(request, Some("served")).ok().expect("parsed");
        assert_eq!(parsed.model, "served");
        let parsed = parse_request(request, None).ok().expect("parsed");
        assert_eq!(parsed.model, "alias");
        let unnamed = r#" { "messages": [{"role": "user", "content": "hi"}]}"#;
        let parsed = parse_request(unnamed, Some("served")).ok().expect("parsed");
        assert_eq!(parsed.model, "served");
        let Err(Failure::Input(error)) = parse_request(unnamed, None) else {
            panic!("refused as an input error");
        };
        assert!(error.starts_with("the request names no model"));
        assert_eq!(with_model("{}", "m"), r#"{"model":"m"}"#);
        assert_eq!(with_model(" { } ", "m"), r#"{"model":"m" } "#);
    }

    #[test]
    fn a_duplicate_member_is_refused_with_or_without_a_model() {
        for request in [
            r#"{"model": "m", "messages": [], "messages": []}"#,
            r#"{"messages": [], "messages": []}"#,
        ] {
            let Err(Failure::Input(error)) = parse_request(request, Some("served")) else {
                panic!("refused as an input error: {request}");
            };
            assert!(error.contains("duplicate field `messages`"), "{error}");
        }
    }

    #[test]
    fn a_panic_is_an_internal_failure() {
        let Err(Failure::Internal(error)) = caught::<()>(|| panic!("boom")) else {
            panic!("caught as an internal failure");
        };
        assert_eq!(error, "SMG's tokenizer panicked: boom");
    }

    #[test]
    fn warnings_are_captured_for_the_work_that_logs_them() {
        let ((), warnings) = captured(|| {
            tracing::warn!(tool = "search", bytes = 3, "arguments rendered as {{}}");
            tracing::info!("not a warning");
        });
        assert_eq!(
            warnings,
            ["arguments rendered as {} (tool=search, bytes=3)"]
        );
        let ((), outside) = captured(|| ());
        assert!(outside.is_empty());
    }

    #[test]
    fn a_count_left_by_another_process_counts_for_nothing() {
        let pid = std::process::id();
        assert_eq!(count_for(u64::from(pid) << 32 | 2, pid), 2);
        assert_eq!(count_for(u64::from(pid + 1) << 32 | 2, pid), 0);
        // A count taken over from another process starts from none, and
        // never goes below it.
        let counter = AtomicU64::new(u64::from(pid + 1) << 32 | 2);
        count_up(&counter, pid);
        assert_eq!(count_for(counter.load(Ordering::SeqCst), pid), 1);
        count_down(&counter, pid);
        count_down(&counter, pid);
        assert_eq!(counter.load(Ordering::SeqCst), u64::from(pid) << 32);
    }
}
