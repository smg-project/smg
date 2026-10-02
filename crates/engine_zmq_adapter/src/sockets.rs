//! Socket addresses for the same-host ZMQ topology: the `ipc://` data plane
//! SMG binds, the `tcp://` handshake the engine dials, and the owner-only
//! socket directory they live in.

use std::{path::Path, time::Duration};

/// Loopback host for the same-host ZMQ transport (TCP handshake and local
/// binds). Shared with the worker-side socket derivation.
pub(crate) const ZMQ_LOOPBACK_HOST: &str = "127.0.0.1";

/// Time to wait for a ZMQ engine to complete the startup handshake. Generous:
/// the engine loads the model and profiles KV cache between INIT and READY.
pub(crate) const ZMQ_CONNECT_TIMEOUT: Duration = Duration::from_secs(600);

/// Derive a deterministic TCP handshake port from the ipc data-plane path.
///
/// vLLM's headless engine dials a *TCP* handshake (`--data-parallel-address` +
/// `--data-parallel-rpc-port`); making the port a pure function of the worker
/// URL lets the operator compute the same `--data-parallel-rpc-port` without a
/// side channel. FNV-1a keeps it stable across processes and builds. Mapped
/// into 20000..=29999 to avoid well-known and typical ephemeral ranges.
///
/// `_zmq_handshake_port` in `bindings/python/src/smg/serve.py` mirrors this
/// function — keep them in sync.
pub(crate) fn derive_handshake_port(path: &str) -> u16 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in path.as_bytes() {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // Map into 20000..=29999: below the Linux default ephemeral range
    // (`net.ipv4.ip_local_port_range` = 32768..60999) so an outbound socket
    // can't already hold the port. `hash % 10000` always fits u16.
    20000 + (hash % 10000) as u16
}

/// Derive the ZMQ socket addresses for a worker from its base URL.
///
/// Mirrors vLLM's headless topology: the **handshake is TCP** (the engine dials
/// it, so it matches `vllm serve --headless --data-parallel-rpc-port`), while
/// the **data plane is `ipc://`** for the same-host fast path (SMG chooses these
/// and hands them to the engine during the handshake INIT). The operator gives a
/// single `ipc://<path>` base; SMG binds the ipc input/output at
/// `<path>-in.sock` / `-out.sock` and derives the TCP handshake port from the
/// path. A `WorkerSpec.zmq_handshake_address` override replaces the derived
/// handshake address verbatim (it must be `tcp://`), for engines that dial a
/// fixed, pre-agreed address — e.g. TokenSpeed's default dial target is
/// `tcp://127.0.0.1:30500` (its `--data-parallel-address`/
/// `--data-parallel-rpc-port` defaults, outside the derived 20000..=29999
/// band), so setting the override to that value pairs a bare
/// `ts serve --headless` with a manually registered worker.
/// Returns `(handshake, input, output)`.
///
/// [`zmq_handshake_address`] exposes just the handshake half, for the
/// registration-time validation of that address.
pub(crate) fn zmq_socket_addresses(
    base_url: &str,
    handshake_override: Option<&str>,
) -> Result<(String, String, String), String> {
    let path = base_url
        .strip_prefix("ipc://")
        .ok_or_else(|| format!("ZMQ worker URL must be ipc://<path>, got '{base_url}'"))?;
    let handshake = match handshake_override {
        Some(address) => {
            if !address.starts_with("tcp://") {
                return Err(format!(
                    "zmq_handshake_address must be a tcp:// address \
                     (the engine dials a TCP handshake), got '{address}'"
                ));
            }
            address.to_string()
        }
        None => format!("tcp://{ZMQ_LOOPBACK_HOST}:{}", derive_handshake_port(path)),
    };
    let input = format!("ipc://{path}-in.sock");
    let output = format!("ipc://{path}-out.sock");
    Ok((handshake, input, output))
}

/// The TCP handshake address a ZMQ worker will bind.
///
/// Carries [`zmq_socket_addresses`]'s error verbatim rather than flattening it
/// to "no address": an unusable `ipc://` base or a non-`tcp://` override is a
/// misconfiguration registration must reject, not a value to skip past and
/// rediscover at every later connect attempt.
pub fn zmq_handshake_address(
    base_url: &str,
    handshake_override: Option<&str>,
) -> Result<String, String> {
    zmq_socket_addresses(base_url, handshake_override).map(|(handshake, _, _)| handshake)
}

/// Create the parent directory for a worker's `ipc://` sockets. Kept off the
/// address computation (which is pure) and async so it doesn't block a runtime
/// thread.
///
/// The ipc:// data-plane sockets SMG binds here carry no authentication, so the
/// directory must be owner-controlled: when this call creates it, it is created
/// 0700 (mode applied at mkdir time — no chmod window); when it already exists,
/// its permissions are left untouched (never chmod a shared dir like `/tmp`)
/// and it is rejected unless it is a real directory owned by the current user.
pub(crate) async fn ensure_ipc_socket_dir(base_url: &str) -> Result<(), String> {
    let path = base_url.strip_prefix("ipc://").unwrap_or(base_url);
    let Some(parent) = Path::new(path).parent() else {
        return Ok(());
    };
    // symlink_metadata: a symlinked parent must not redirect the checks (or the
    // sockets) into a directory we did not verify.
    let meta = match tokio::fs::symlink_metadata(parent).await {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let mut builder = tokio::fs::DirBuilder::new();
            builder.recursive(true);
            #[cfg(unix)]
            builder.mode(0o700);
            builder
                .create(parent)
                .await
                .map_err(|e| format!("failed to create ipc socket dir for {path}: {e}"))?;
            tokio::fs::symlink_metadata(parent)
                .await
                .map_err(|e| format!("failed to stat ipc socket dir for {path}: {e}"))?
        }
        Err(e) => return Err(format!("failed to stat ipc socket dir for {path}: {e}")),
    };
    if !meta.is_dir() {
        return Err(format!(
            "ipc socket dir {} exists but is not a directory",
            parent.display()
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let uid = rustix::process::geteuid().as_raw();
        if meta.uid() != uid {
            return Err(format!(
                "ipc socket dir {} is owned by uid {} (expected {uid}); refusing to bind \
                 unauthenticated ZMQ sockets in a directory owned by another user",
                parent.display(),
                meta.uid()
            ));
        }
    }
    Ok(())
}

/// Remove a stale ipc socket file left by a previous gateway process, if any.
/// Only ever unlinks sockets (never a regular file at the path), inside the
/// owner-verified socket dir.
#[cfg(unix)]
pub(crate) async fn unlink_stale_socket(address: &str) -> Result<(), String> {
    let Some(path) = address.strip_prefix("ipc://") else {
        return Ok(());
    };
    match tokio::fs::symlink_metadata(path).await {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("failed to stat ipc socket path {path}: {e}")),
        Ok(meta) => {
            use std::os::unix::fs::FileTypeExt;
            if !meta.file_type().is_socket() {
                return Err(format!(
                    "ipc socket path {path} exists but is not a socket; refusing to unlink"
                ));
            }
            tracing::info!("Removing stale ipc socket {path} from a previous gateway run");
            tokio::fs::remove_file(path)
                .await
                .map_err(|e| format!("failed to remove stale ipc socket {path}: {e}"))
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[cfg(unix)]
    #[tokio::test]
    async fn unlink_stale_socket_removes_sockets_and_refuses_files() {
        use std::os::unix::net::UnixListener;

        let dir = tempfile::tempdir().unwrap();
        let sock_path = dir.path().join("stale.sock");
        // A bound-then-dropped listener leaves the socket file behind, exactly
        // like a dead gateway does.
        drop(UnixListener::bind(&sock_path).unwrap());
        assert!(sock_path.exists());
        let addr = format!("ipc://{}", sock_path.display());
        unlink_stale_socket(&addr).await.unwrap();
        assert!(!sock_path.exists(), "stale socket must be removed");

        // Missing file: fine.
        unlink_stale_socket(&addr).await.unwrap();

        // A regular file at the path is not ours to delete.
        std::fs::write(&sock_path, b"not a socket").unwrap();
        let err = unlink_stale_socket(&addr).await.unwrap_err();
        assert!(err.contains("not a socket"), "{err}");
        assert!(sock_path.exists(), "regular file must survive");
    }

    #[test]
    fn derive_handshake_port_matches_pinned_vectors() {
        // Fixed vectors shared with `_zmq_handshake_port` in
        // bindings/python/src/smg/serve.py — a change on either side breaks the
        // engine/router port agreement, so these must stay in sync.
        assert_eq!(derive_handshake_port("/tmp/smg-zmq/ts0.ipc"), 25152);
        assert_eq!(derive_handshake_port("/tmp/smg-zmq/engine-31000"), 22714);
        // Range invariant: every path maps into 20000..=29999.
        for p in ["", "a", "/x/y/z.ipc", "very/long/path/with/segments.sock"] {
            let port = derive_handshake_port(p);
            assert!(
                (20000..=29999).contains(&port),
                "port {port} out of band for {p:?}"
            );
        }
    }

    #[test]
    fn zmq_socket_addresses_derive_handshake_by_default() {
        let (handshake, input, output) =
            zmq_socket_addresses("ipc:///tmp/smg-zmq/ts0.ipc", None).unwrap();
        assert_eq!(handshake, "tcp://127.0.0.1:25152");
        assert_eq!(input, "ipc:///tmp/smg-zmq/ts0.ipc-in.sock");
        assert_eq!(output, "ipc:///tmp/smg-zmq/ts0.ipc-out.sock");
    }

    #[test]
    fn zmq_socket_addresses_honor_handshake_override() {
        // TokenSpeed's default dial target — outside the derived band; the
        // override must be bound verbatim while the data plane stays derived.
        let (handshake, input, output) =
            zmq_socket_addresses("ipc:///tmp/smg-zmq/ts0.ipc", Some("tcp://127.0.0.1:30500"))
                .unwrap();
        assert_eq!(handshake, "tcp://127.0.0.1:30500");
        assert_eq!(input, "ipc:///tmp/smg-zmq/ts0.ipc-in.sock");
        assert_eq!(output, "ipc:///tmp/smg-zmq/ts0.ipc-out.sock");
    }

    #[test]
    fn zmq_socket_addresses_reject_non_tcp_override() {
        // The engine dials a TCP handshake; a non-tcp override is a config
        // error and must fail loudly rather than bind something unexpected.
        let err = zmq_socket_addresses("ipc:///tmp/smg-zmq/ts0.ipc", Some("ipc:///tmp/hs.sock"))
            .unwrap_err();
        assert!(
            err.contains("tcp://"),
            "error must name the required scheme: {err}"
        );
    }

    #[tokio::test]
    async fn ensure_ipc_socket_dir_creates_a_private_owner_only_dir() {
        let base = tempfile::tempdir().unwrap();
        let dir = base.path().join("sockets");
        let url = format!("ipc://{}/x.ipc", dir.display());
        ensure_ipc_socket_dir(&url).await.unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o700, "created socket dir must be 0700");
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn ensure_ipc_socket_dir_leaves_an_existing_owned_dir_untouched() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        let url = format!("ipc://{}/x.ipc", dir.path().display());
        ensure_ipc_socket_dir(&url).await.unwrap();
        let mode = std::fs::metadata(dir.path()).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o755, "an existing dir must not be chmod'd");
    }

    #[tokio::test]
    async fn ensure_ipc_socket_dir_rejects_a_non_directory_parent() {
        let base = tempfile::tempdir().unwrap();
        let file = base.path().join("not-a-dir");
        std::fs::write(&file, b"x").unwrap();
        let url = format!("ipc://{}/x.ipc", file.display());
        assert!(ensure_ipc_socket_dir(&url).await.is_err());
    }
}
