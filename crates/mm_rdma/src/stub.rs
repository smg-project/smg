//! No-op RDMA exporter used when the `nixl` feature is disabled, so the gateway
//! compiles and links identically in both build configs. Every `export` returns
//! the bytes back for the caller's inline fallback.

use crate::RdmaConfig;

/// Construction error, mirroring the real impl's signature. The stub never fails.
#[derive(Debug)]
pub struct RdmaError(());

impl std::fmt::Display for RdmaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RDMA transport not compiled in (build with the `nixl` feature)")
    }
}

impl std::error::Error for RdmaError {}

/// Inert exporter: constructs but stages nothing.
pub struct RdmaExporter;

impl RdmaExporter {
    /// Always succeeds; the returned exporter is inert.
    #[expect(
        clippy::unnecessary_wraps,
        reason = "signature must mirror the real nixl RdmaExporter::new so the gateway compiles unchanged in both build configs"
    )]
    pub fn new(_cfg: RdmaConfig) -> Result<Self, RdmaError> {
        Ok(RdmaExporter)
    }

    /// No-op: hands `bytes` straight back so the caller ships them inline.
    #[expect(
        clippy::unused_self,
        reason = "signature must mirror the real nixl RdmaExporter::export"
    )]
    pub fn export(&self, _slot_key: i64, bytes: Vec<u8>) -> Result<Vec<u8>, Vec<u8>> {
        Err(bytes)
    }

    /// No-op: there is no arena in a build without NIXL.
    #[expect(clippy::unused_self, reason = "match the NIXL exporter")]
    pub fn export_slice(&self, _slot_key: i64, _bytes: &[u8]) -> Option<Vec<u8>> {
        None
    }
}
