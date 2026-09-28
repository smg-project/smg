//! Capability-driven parsing for tokenizer-provided response templates.

mod compiled;
mod config;
mod error;
mod output;
mod parser;
mod schema;

pub use compiled::{DelimiterBound, DelimiterMetadata};
pub use config::{ParserConfig, DEFAULT_BYTE_LIMIT};
pub use error::ResponseTemplateError;
pub use output::{ParseOutput, ToolCall};
pub use parser::{FinishMode, ResponseTemplateParser, StreamingParser};
pub use schema::{
    ClosePattern, ContentArgs, FieldTemplate, ResponseTemplate, Transform, ValueParser,
    ValueParserArgs,
};
