//! Shared runtime and global resources for FFI

use once_cell::sync::Lazy;
use tokio::runtime::Runtime;

/// Global tokio runtime for all async FFI operations
#[expect(
    clippy::expect_used,
    reason = "runtime creation is infallible in practice and failure is unrecoverable"
)]
pub static RUNTIME: Lazy<Runtime> =
    Lazy::new(|| Runtime::new().expect("Failed to create tokio runtime for FFI"));
