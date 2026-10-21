//! Which encoder serves a Hugging Face tokenizer's encodes.
//!
//! The choice is made once per load, in order of preference: the opt-in
//! `tokenizers` v1 pipeline (feature `tokenizers-v1`), the crate's direct
//! byte-level encoder (`native`), and `tokenizers` 0.23 itself. Each of the
//! first two takes the tokenizers whose configuration it reproduces and
//! declines the rest with a reason ([`Declined`]); the load logs the path it
//! ended on and those reasons, once. After the choice an input can still
//! leave the direct encoder for `tokenizers` ([`Fallback`]), which costs the
//! work done up to that point.
//!
//! The process-lifetime totals here have the shape of `cache::activity`:
//! fixed-size snapshots, no locks, never reset, read through
//! [`encode_path_loads`] and [`encode_fallbacks`] whether or not a metrics
//! recorder exists.

use std::{
    borrow::Cow,
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};

/// The path a Hugging Face tokenizer's encodes take, chosen at load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodePath {
    /// The opt-in `tokenizers` v1 pipeline.
    V1,
    /// The crate's direct byte-level encoder.
    Native,
    /// `tokenizers` 0.23, the reference implementation.
    Tokenizers,
}

impl EncodePath {
    /// Every path, in order of preference.
    pub const ALL: [Self; 3] = [Self::V1, Self::Native, Self::Tokenizers];

    /// The path's name as the metrics and the log write it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::Native => "native",
            Self::Tokenizers => "tokenizers",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::V1 => 0,
            Self::Native => 1,
            Self::Tokenizers => 2,
        }
    }
}

impl fmt::Display for EncodePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why a path did not take a tokenizer at load: the configuration it found
/// outside what it reproduces, in the words the load's log line carries.
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

/// Why one input left the direct encoder for `tokenizers` after the load's
/// choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fallback {
    /// The text is not in NFC under an NFC normalizer; found before any
    /// work, by the quick check or a normalizing pass.
    NotNfc,
    /// The model declined a piece (a vocabulary whose `unk_token` is not in
    /// it, in `tokenizers` 0.23); found after the pieces before it were
    /// tokenized, so that the error surfaces from the `tokenizers` encode.
    PieceDeclined,
}

impl Fallback {
    /// The reason's name as the metrics write it.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::NotNfc => "not_nfc",
            Self::PieceDeclined => "piece_declined",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::NotNfc => 0,
            Self::PieceDeclined => 1,
        }
    }
}

static LOADS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static FALLBACKS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Count one load that ended on `path`.
pub(crate) fn note_load(path: EncodePath) {
    LOADS[path.index()].fetch_add(1, Ordering::Relaxed);
}

/// Count one input that left the direct encoder.
pub(crate) fn note_fallback(fallback: Fallback) {
    FALLBACKS[fallback.index()].fetch_add(1, Ordering::Relaxed);
}

/// Cumulative loads of one path across all tokenizer instances.
#[derive(Debug, Clone, Copy)]
pub struct EncodePathLoads {
    pub path: EncodePath,
    /// Hugging Face tokenizers loaded whose encodes take `path`; a
    /// tokenizer loaded twice counts twice.
    pub loads: u64,
}

/// Cumulative inputs that left the direct encoder for one reason across all
/// tokenizer instances.
#[derive(Debug, Clone, Copy)]
pub struct EncodeFallbacks {
    /// `not_nfc` or `piece_declined`, see [`Fallback`].
    pub reason: &'static str,
    /// Inputs encoded by `tokenizers` after the direct encoder gave them
    /// up; one per input, a batch counting each of its inputs.
    pub encodes: u64,
}

/// Read process-lifetime totals, including activity before a metrics recorder exists.
pub fn encode_path_loads() -> [EncodePathLoads; 3] {
    EncodePath::ALL.map(|path| EncodePathLoads {
        path,
        loads: LOADS[path.index()].load(Ordering::Relaxed),
    })
}

/// Read process-lifetime totals, including activity before a metrics recorder exists.
pub fn encode_fallbacks() -> [EncodeFallbacks; 2] {
    [Fallback::NotNfc, Fallback::PieceDeclined].map(|fallback| EncodeFallbacks {
        reason: fallback.as_str(),
        encodes: FALLBACKS[fallback.index()].load(Ordering::Relaxed),
    })
}
