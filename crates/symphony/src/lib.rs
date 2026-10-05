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
//! Status: skeleton. The public types below are the contract; the engine, the format definitions
//! and the protocol adapters follow in later changes.

#![forbid(unsafe_code)]

pub mod event;
pub mod format;
pub mod input;
pub mod parser;

pub use event::{DropReason, Event, Events, FinishReason, MalformedReason, Text};
pub use format::Format;
pub use input::{EngineFinish, Input, TokenSpan};
pub use parser::{ParseError, Parser};
