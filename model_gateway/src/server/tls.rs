//! The server's TLS half: the start-up check of who can read a private key
//! file.
//!
//! The process-level rustls crypto provider and the HTTP client constructors
//! that need it live in the worker crate (`smg_worker::connection::client_factory`),
//! which builds the worker-directed clients; [`crate::server`] re-exports
//! them.

use std::path::Path;

use tracing::warn;

/// The permission bits of a private key file at `path` when more of the
/// system than its owner can read or write it (`Some(0o644)` for a 0644
/// file); `None` when the file is private enough, cannot be read (the loader
/// reports that), or the platform has no Unix modes. A file the gateway owns
/// is private only without group and other bits; a file another user owns
/// (a secret volume with a group, a `root:ssl-cert` key) may be readable by
/// the gateway's group, since that is how the gateway reads it at all, but
/// not writable by it and not open to others (see [`shared_mode`]).
fn shared_private_key_mode(path: &Path) -> Option<u32> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let metadata = std::fs::metadata(path).ok()?;
        let owned = metadata.uid() == rustix::process::geteuid().as_raw();
        shared_mode(metadata.permissions().mode() & 0o777, owned)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

/// Whether `mode` leaves a private key file open to more than its owner:
/// any group or other bit on a file the gateway owns (`owned`), group write
/// or any other bit on a file another user owns (group read is how the
/// gateway reads such a file). Answers the mode when it does.
fn shared_mode(mode: u32, owned: bool) -> Option<u32> {
    let shared = if owned { 0o077 } else { 0o027 };
    (mode & shared != 0).then_some(mode)
}

/// One WARN when the private key file at `path` (the value of `flag`) is
/// readable or writable by more than its owner (see
/// [`shared_private_key_mode`]). The gateway still starts: a secret manager
/// may tighten the mode after the file lands, and the operator decides. The
/// line names the file and its mode, never its content.
pub(crate) fn warn_if_private_key_is_shared(path: &Path, flag: &str) {
    if let Some(mode) = shared_private_key_mode(path) {
        warn!(
            path = %path.display(),
            flag,
            "the TLS private key file is readable or writable by its group or by others (mode {mode:04o}); expected 0600 or 0400 (0640 when another user owns the file)"
        );
    }
}

#[cfg(test)]
#[cfg(unix)]
#[path = "../../tests/unit/server/tls/tests.rs"]
mod tests;
