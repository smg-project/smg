//! The start-up check of who can read a private key file.

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
pub(crate) fn shared_private_key_mode(path: &Path) -> Option<u32> {
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
pub(crate) fn shared_mode(mode: u32, owned: bool) -> Option<u32> {
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
        // The mode is what the check reads; the content is not a key (and
        // not shaped like one, for the pre-commit hook that looks for one).
        file.write_all(b"not a key\n").expect("write");
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

    /// A file the gateway owns is shared by any group or other bit; a file
    /// another user owns is shared by group write or any other bit, not by
    /// group read (the way the gateway reads a secret volume's key).
    #[test]
    fn a_key_file_open_to_more_than_its_owner_is_shared() {
        assert_eq!(shared_mode(0o644, true), Some(0o644));
        assert_eq!(shared_mode(0o640, true), Some(0o640));
        assert_eq!(shared_mode(0o600, true), None);
        assert_eq!(shared_mode(0o400, true), None);
        assert_eq!(
            shared_mode(0o640, false),
            None,
            "group read of another user's key"
        );
        assert_eq!(shared_mode(0o440, false), None);
        assert_eq!(shared_mode(0o660, false), Some(0o660), "group write");
        assert_eq!(shared_mode(0o644, false), Some(0o644), "others read");
        assert_eq!(shared_mode(0o602, false), Some(0o602), "others write");
    }

    #[test]
    fn a_key_file_readable_by_its_group_or_by_others_is_shared() {
        // The files below are the test's own, so every group or other bit counts.
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
