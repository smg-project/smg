//! The process-level rustls crypto provider, and the start-up check of who
//! can read a private key file.

use std::path::Path;

use rustls::crypto::{ring, CryptoProvider};
use tracing::warn;

/// Installs `ring` as the process-level rustls crypto provider, unless one
/// is installed already.
///
/// `ring` is the only backend compiled into this crate's binaries. The bare
/// rustls builders (the HTTPS listener, the mesh listener) pick it up on
/// their own, but the HTTP client is built with reqwest's
/// `rustls-no-provider` feature and uses whatever provider the process
/// installed, panicking when there is none. Installing is idempotent: a
/// provider installed earlier, by an embedding process or a previous start in
/// the same process, stays. [`crate::server::startup`] calls it first thing,
/// and so do the functions of [`crate::http_client`], through which this
/// crate, its bindings and its tests build every HTTP client; nothing else
/// needs to.
pub fn install_crypto_provider() {
    if CryptoProvider::get_default().is_none() {
        // An error here means another thread installed one in between,
        // which is the outcome wanted.
        let _ = ring::default_provider().install_default();
    }
}

/// The permission bits of a private key file at `path` when its group or
/// others can read or write it (`Some(0o644)` for a 0644 file); `None` when
/// the file is private to its owner, cannot be read (the loader reports
/// that), or the platform has no Unix modes.
pub(crate) fn shared_private_key_mode(path: &Path) -> Option<u32> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(path).ok()?.permissions().mode() & 0o777;
        (mode & 0o077 != 0).then_some(mode)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

/// One WARN when the private key file at `path` (the value of `flag`) is
/// readable or writable by its group or by others. The gateway still starts:
/// a secret manager may tighten the mode after the file lands, and the
/// operator decides. The line names the file and its mode, never its content.
pub(crate) fn warn_if_private_key_is_shared(path: &Path, flag: &str) {
    if let Some(mode) = shared_private_key_mode(path) {
        warn!(
            path = %path.display(),
            flag,
            "the TLS private key file is readable by its group or by others (mode {mode:04o}); expected 0600 or 0400"
        );
    }
}

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use std::{
        io::Write,
        os::unix::fs::PermissionsExt,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
    };

    use tracing_subscriber::{layer::SubscriberExt, Layer, Registry};

    use super::*;

    fn key_file(mode: u32) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(b"-----BEGIN PRIVATE KEY-----\nnot a key\n-----END PRIVATE KEY-----\n")
            .expect("write");
        std::fs::set_permissions(file.path(), std::fs::Permissions::from_mode(mode))
            .expect("set the mode");
        file
    }

    /// Counts the WARN events on this thread's subscriber.
    struct Warnings(Arc<AtomicUsize>);

    impl<S: tracing::Subscriber> Layer<S> for Warnings {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if *event.metadata().level() == tracing::Level::WARN {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[test]
    fn a_key_file_readable_by_its_group_or_by_others_is_shared() {
        assert_eq!(shared_private_key_mode(key_file(0o644).path()), Some(0o644));
        assert_eq!(shared_private_key_mode(key_file(0o640).path()), Some(0o640));
        assert_eq!(shared_private_key_mode(key_file(0o600).path()), None);
        assert_eq!(shared_private_key_mode(key_file(0o400).path()), None);
        assert_eq!(
            shared_private_key_mode(Path::new("/nonexistent/private-key.pem")),
            None,
            "a file that cannot be read is the loader's error, not a mode finding"
        );
    }

    #[test]
    fn a_shared_key_file_warns_once_and_a_private_one_not_at_all() {
        let warnings = Arc::new(AtomicUsize::new(0));
        let _guard =
            tracing::subscriber::set_default(Registry::default().with(Warnings(warnings.clone())));
        warn_if_private_key_is_shared(key_file(0o600).path(), "--tls-key-path");
        assert_eq!(warnings.load(Ordering::Relaxed), 0, "a 0600 key file");
        warn_if_private_key_is_shared(key_file(0o644).path(), "--tls-key-path");
        assert_eq!(warnings.load(Ordering::Relaxed), 1, "a 0644 key file");
    }
}
