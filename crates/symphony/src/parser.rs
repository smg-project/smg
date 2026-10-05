//! The one trait, with its one method.

use crate::{event::Events, input::Input};

/// A parser for one model output. One instance per generated choice; never shared.
///
/// The lifecycle is the sequence of [`Input`]s: one `Prompt`, then `Delta`s as the engine produces
/// them, then one `End`. Whole-output parsing is the same sequence with a single `Delta`, so
/// streaming and non-streaming share one implementation.
///
/// On `Err`, everything already pushed into `out` is final. After a `BufferOverflow` or
/// `Internal` error the caller feeds `End`, and the parser reports what it still held as a
/// `Malformed` event, so the bytes are returned rather than lost. After a `Lifecycle` error the
/// stream is already over or was never started; the parser holds nothing and emits nothing more.
pub trait Parser: Send {
    /// Feed one lifecycle step and append the resulting events to `out`.
    fn feed(&mut self, input: Input<'_>, out: &mut Events) -> Result<(), ParseError>;
}

/// Why a parser could not continue.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    /// The parser would have to hold more bytes than its limit allows.
    #[error("buffered output exceeded {limit} bytes")]
    BufferOverflow {
        /// The limit in bytes.
        limit: usize,
    },
    /// The caller fed inputs out of order (a `Delta` after `End`, a second `Prompt`).
    #[error("lifecycle misuse: {0}")]
    Lifecycle(String),
    /// A defect in a parser, described for the report; never expected in production.
    #[error("internal parser error: {0}")]
    Internal(String),
}
