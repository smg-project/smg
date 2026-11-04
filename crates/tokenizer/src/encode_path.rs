//! Which encoder serves a Hugging Face tokenizer's encodes.
//!
//! The choice is made once per load: the `vllm-tokenizer` backend takes the
//! tokenizers whose `tokenizer.json` it reads as the wrapper's pipeline
//! reads it, and declines the rest with a reason (`Declined`), which
//! `tokenizers` 0.23 itself encodes. The load logs the path it ended on
//! and the reason, once.
//!
//! The process-lifetime totals here have the shape of `cache::activity`:
//! fixed-size snapshots, no locks, never reset, read through
//! [`encode_path_loads`] whether or not a metrics recorder exists.

use std::{
    borrow::Cow,
    fmt,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

/// The path a Hugging Face tokenizer's encodes take, chosen at load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodePath {
    /// The `vllm-tokenizer` backend, the vLLM Rust frontend's.
    Vllm,
    /// `tokenizers` 0.23, the reference implementation.
    Tokenizers,
}

impl EncodePath {
    /// Every path, in order of preference.
    pub const ALL: [Self; 2] = [Self::Vllm, Self::Tokenizers];

    /// The path's name as the metrics and the log write it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vllm => "vllm",
            Self::Tokenizers => "tokenizers",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Vllm => 0,
            Self::Tokenizers => 1,
        }
    }
}

impl fmt::Display for EncodePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why the backend did not take a tokenizer at load: the configuration it
/// found outside what it reproduces, in the words the load's log line
/// carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Declined(Cow<'static, str>);

impl From<&'static str> for Declined {
    fn from(reason: &'static str) -> Self {
        Self(Cow::Borrowed(reason))
    }
}

impl From<String> for Declined {
    fn from(reason: String) -> Self {
        Self(Cow::Owned(reason))
    }
}

impl fmt::Display for Declined {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// What the load hands the `vllm-tokenizer` backend:
/// the `tokenizer.json` to read, where the wrapper's pipeline is the file's,
/// else why the backend declines (see `vllm`).
pub(crate) type VllmSource<'a> = Result<&'a Path, Declined>;

static LOADS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Count one load that ended on `path`.
pub(crate) fn note_load(path: EncodePath) {
    LOADS[path.index()].fetch_add(1, Ordering::Relaxed);
}

/// Cumulative loads of one path across all tokenizer instances.
#[derive(Debug, Clone, Copy)]
pub struct EncodePathLoads {
    pub path: EncodePath,
    /// Hugging Face tokenizers loaded whose encodes take `path`; a
    /// tokenizer loaded twice counts twice.
    pub loads: u64,
}

/// Read process-lifetime totals, including activity before a metrics recorder exists.
pub fn encode_path_loads() -> [EncodePathLoads; 2] {
    EncodePath::ALL.map(|path| EncodePathLoads {
        path,
        loads: LOADS[path.index()].load(Ordering::Relaxed),
    })
}
