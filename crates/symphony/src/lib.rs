//! Symphony: one parser for everything a model says.
//!
//! A model's output stream carries content, reasoning and tool calls interleaved in a format the
//! model vendor chose. Symphony turns that stream into typed [`Event`]s through one object with one
//! method, [`Parser::feed`], whose lifecycle is expressed as [`Input`]: the prompt the engine was
//! given, each decoded delta, and the end of the stream. Every byte of output ends up in exactly
//! one event, including bytes that were dropped or could not be parsed, so nothing disappears
//! silently.
//!
//! This crate replaces the separate tool-call and reasoning parser crates over time. Until the
//! gateway is switched over, it is developed and verified on its own: it changes nothing outside
//! `crates/symphony/`, and its formats are proven against recorded fixtures before they are wired
//! in. The design and the delivery plan live with the maintainers; the short version is in this
//! crate's `README.md`.
//!
//! The standard this crate is held to is craftsmanship: code that reads as well as it runs, with no
//! compromise kept for convenience. A change whose only reason is "this can be better" is welcome.
//!
//! Status: the public types below are the contract, [`adapt`] renders them for the Chat
//! Completions, Responses and Messages APIs, streamed and whole, and [`Engine`] runs a [`Format`]
//! table: the tables in [`formats`] are the families recorded so far, and more follow.

#![forbid(unsafe_code)]

pub mod adapt;
pub mod engine;
pub mod event;
pub mod format;
pub mod formats;
pub mod grammar;
pub mod input;
pub mod json;
pub mod markers;
pub mod parser;
pub mod pythonic;
pub mod tagged;
pub mod tokens;

pub use engine::Engine;
pub use event::{DropReason, Event, Events, FinishReason, MalformedReason, Text};
pub use format::{CallSyntax, Emits, Format};
pub use formats::{
    deepseek_v4_1, glm, hy4, iquest, kimi_k3, lfm2_5, ling, minimax_m3, olmo3, plain, qwen2_5,
    qwen3, seed_oss, xlam,
};
pub use grammar::{Grammar, Tag};
pub use input::{EngineFinish, Input, TokenSpan};
pub use markers::{Piece, Scanner};
pub use parser::{ParseError, Parser};
pub use tagged::Declared;
pub use tokens::Ledger;
