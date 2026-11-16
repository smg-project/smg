//! The one trait, with its one method.

use crate::{event::Events, input::Input};

/// A parser for one model output. One instance per generated choice; never shared.
///
/// The lifecycle is the sequence of [`Input`]s: one `Prompt`, then `Delta`s as the engine produces
/// them, then one `End`. Whole-output parsing is the same sequence with a single `Delta`, so
/// streaming and non-streaming share one implementation.
///
/// On `Err`, everything already pushed into `out` is final. The one error, a `Lifecycle` misuse,
/// rejects a step fed out of order (a `Prompt` after the first step, any step after `End`): the
/// rejected step emits nothing and leaves the parser as it was, so an open stream goes on and an
/// ended one stays closed. A parser never fails on the output's bytes: what it cannot read it
/// returns as `Malformed` events, and the end of the stream closes what it still held.
pub trait Parser: Send {
    /// Feed one lifecycle step and append the resulting events to `out`. The events may borrow
    /// the input's text for the life of `out` (`'d`): a text run the parser passed through
    /// without holding it is a slice of the delta, not a copy; see `Event`.
    fn feed<'d>(&mut self, input: Input<'d>, out: &mut Events<'d>) -> Result<(), ParseError>;
}

/// Why a parser refused a step: the caller's order of steps, never the output's bytes.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    /// The caller fed inputs out of order (a `Delta` after `End`, a second `Prompt`).
    #[error("lifecycle misuse: {0}")]
    Lifecycle(String),
}
