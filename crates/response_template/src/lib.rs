//! Response templates: a Rust port of `transformers.utils.chat_parsing`
//! (transformers 5.17.0). A checkpoint can describe how its output splits into
//! message fields (`thinking`, `content`, `tool_calls`, ...) with the
//! `response_template` in its `tokenizer_config.json`; this crate loads the
//! template and parses model output with it, returning what transformers
//! returns, value for value. A template this crate would not parse exactly as
//! transformers does is refused when it loads (`LoadError::Unsupported`).
//!
//! Portions are ported from Hugging Face transformers (Apache-2.0). The files
//! follow its layout (`response_templates.rs`, `response_parser.rs`,
//! `content_parsers.rs`); the README maps each transformers name to its Rust
//! counterpart and lists what is not supported.

mod content_parsers;
mod error;
mod py;
mod pyre;
mod response_parser;
mod response_templates;
#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod test_fixtures;
#[cfg(test)]
mod unit_tests;

pub use error::{LoadError, ParseError, PyErrorKind, Unsupported};
pub use response_parser::{parse_response, Event, Message, ResponseParser};
pub use response_templates::{load_response_template, ResponseTemplate};
