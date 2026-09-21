//! Shared runtime and global resources for FFI

use once_cell::sync::Lazy;
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use tokio::runtime::Runtime;
use tool_parser::ParserFactory;

/// Global tokio runtime for all async FFI operations
#[expect(
    clippy::expect_used,
    reason = "runtime creation is infallible in practice and failure is unrecoverable"
)]
pub static RUNTIME: Lazy<Runtime> =
    Lazy::new(|| Runtime::new().expect("Failed to create tokio runtime for FFI"));

/// Global parser factory (initialized once)
pub static PARSER_FACTORY: Lazy<ParserFactory> = Lazy::new(ParserFactory::new);

/// Global reasoning parser factory: resolves the model's parser so the
/// rendered prompt can be read for its reasoning markers.
pub static REASONING_PARSER_FACTORY: Lazy<ReasoningParserFactory> =
    Lazy::new(ReasoningParserFactory::new);
