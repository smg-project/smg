//! The tokenizer SMG serves with, for Python: `llm-tokenizer`'s load, encode
//! and decode, so that a training pipeline gets the token ids serving feeds
//! the model.
//!
//! Short work runs on the calling thread with the GIL held. Work that can
//! reach rayon (loading, batches, long texts) runs on a pool this module
//! owns, one per process (see [`pool`]), with the GIL released.

use std::{
    cell::Cell,
    env, fmt,
    panic::{self, AssertUnwindSafe},
    path::{self, Path},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, PoisonError,
    },
};

use llm_tokenizer::{
    factory::{create_tokenizer_with_chat_template_blocking, is_likely_openai_model},
    hub::download_tokenizer_from_hf,
    traits::Tokenizer as TokenizerTrait,
    Encoding, HuggingFaceTokenizer, TiktokenTokenizer,
};
use pyo3::{
    exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use smg::version;

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
/// text that trips the tokenizer would end a data job; serving fails that
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
// Forks: the hook that counts them
// ---------------------------------------------------------------------------

/// Forks this process went through by `os.fork()`, counted in the child: a
/// pool built before one has no threads after it, even when the child's
/// process id is a recycled one of its parent's line.
static FORKS: AtomicU64 = AtomicU64::new(0);

/// In the child of `os.fork()`.
#[pyfunction]
fn after_fork_in_child() {
    FORKS.fetch_add(1, Ordering::SeqCst);
}

/// Whether this process registered [`after_fork_in_child`].
static FORK_HOOKS: AtomicBool = AtomicBool::new(false);

/// Register [`after_fork_in_child`] with `os.register_at_fork`, once per
/// process, when its first tokenizer loads. Checked and registered with the
/// GIL held throughout, so that no fork falls in between: a once-cell would
/// release the GIL while it initializes, and a child forked then would find
/// the cell initializing forever. Two threads that race to the first load
/// register at worst twice, which counts a fork twice.
fn register_fork_hooks(py: Python<'_>) -> PyResult<()> {
    if FORK_HOOKS.load(Ordering::SeqCst) {
        return Ok(());
    }
    let os = py.import("os")?;
    if os.hasattr("register_at_fork")? {
        let hooks = PyDict::new(py);
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
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|index| format!("smg-tokenizer-{index}"))
        .build()
        .map_err(|error| {
            Failure::Internal(format!(
                "starting the tokenizer's thread pool: {error} (SMG_TOKENIZER_THREADS sets its \
                 size)"
            ))
        })
}

/// Run `work` on the process's pool, where its rayon calls stay.
fn in_pool<T: Send>(work: impl FnOnce() -> T + Send) -> Result<T, Failure> {
    Ok(pool()?.install(work))
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
    let tokenizer =
        py.detach(|| in_pool(|| caught(|| Ok(TiktokenTokenizer::from_model_name(name).ok()))))??;
    Ok(tokenizer.map(|tokenizer| Arc::new(tokenizer) as Arc<dyn TokenizerTrait>))
}

/// The Rust half of `smg.tokenizer.Tokenizer`; see that class for the API.
#[pyclass(name = "Tokenizer", module = "smg.smg_rs", frozen)]
pub struct PyTokenizer {
    inner: Arc<dyn TokenizerTrait>,
    source: String,
}

#[pymethods]
impl PyTokenizer {
    /// Load the tokenizer of `source` (a checkpoint directory, a tokenizer
    /// file, or a Hugging Face model id) as the gateway does.
    #[new]
    fn new(py: Python<'_>, source: &str) -> PyResult<Self> {
        register_fork_hooks(py)?;
        let path = Path::new(source);
        // A path, made absolute; a name that can only be a path (absolute,
        // relative to `.` or `~`, or of more than two segments), refused when
        // it is not there, rather than read as a model name; an OpenAI model
        // name tiktoken knows, as given (see [`openai_tokenizer`]); any other
        // name, a Hub id, downloaded here and loaded from its snapshot. The
        // download releases the GIL, and stays off the pool, whose threads it
        // must not hold.
        let (location, openai) = if path.exists() {
            (absolute(path)?, None)
        } else if path.is_absolute()
            || source.starts_with(['.', '~'])
            || source.matches('/').count() > 1
        {
            return Err(PyFileNotFoundError::new_err(format!(
                "no such file or directory: {source}"
            )));
        } else if let Some(tokenizer) = openai_tokenizer(py, source)? {
            (source.to_string(), Some(tokenizer))
        } else {
            (py.detach(|| caught(|| download(source)))?, None)
        };
        let inner = match openai {
            Some(tokenizer) => tokenizer,
            None => py.detach(|| {
                in_pool(|| {
                    caught(|| {
                        create_tokenizer_with_chat_template_blocking(&location, None).map_err(
                            |error| {
                                Failure::Internal(format!(
                                    "loading the tokenizer from {location}: {error:#}"
                                ))
                            },
                        )
                    })
                })?
            })?,
        };
        Ok(Self {
            inner,
            source: location,
        })
    }

    /// The source the tokenizer was loaded from: the absolute path of a
    /// local checkpoint or of the snapshot a Hub id downloaded to, or the
    /// OpenAI model name tiktoken loaded.
    #[getter]
    fn source(&self) -> &str {
        &self.source
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
        let ids = py.detach(|| in_pool(|| encode_ids(tokenizer, text, add_special_tokens))?)?;
        Ok(ids)
    }

    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        add_special_tokens: bool,
    ) -> PyResult<Vec<Vec<u32>>> {
        let tokenizer = &*self.inner;
        let ids = py.detach(|| {
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
        Ok(py.detach(|| decode_ids(tokenizer, &token_ids, skip_special_tokens))?)
    }

    fn decode_batch(
        &self,
        py: Python<'_>,
        sequences: Vec<Vec<u32>>,
        skip_special_tokens: bool,
    ) -> PyResult<Vec<String>> {
        let tokenizer = &*self.inner;
        let texts = py.detach(|| {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panic_is_an_internal_failure() {
        let Err(Failure::Internal(error)) = caught::<()>(|| panic!("boom")) else {
            panic!("caught as an internal failure");
        };
        assert_eq!(error, "SMG's tokenizer panicked: boom");
    }
}
