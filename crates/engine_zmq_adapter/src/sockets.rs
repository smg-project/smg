//! Socket addresses for the ZMQ topology: the data plane SMG binds (`ipc://`
//! for a same-host engine, `tcp://` for one on another host), the `tcp://`
//! handshake the engine dials, and the owner-only socket directory the
//! `ipc://` sockets live in.

use std::path::Path;

/// Loopback host for the same-host ZMQ transport (TCP handshake and local
/// binds). Shared with the worker-side socket derivation.
pub(crate) const ZMQ_LOOPBACK_HOST: &str = "127.0.0.1";

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

/// The handshake endpoint a ZMQ link binds, as its caller names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handshake<'a> {
    /// The loopback `tcp://` address derived from the worker's `ipc://` base.
    Derived,
    /// A registered worker's `zmq_handshake_address` override: `tcp://` only,
    /// since the engine dials a TCP handshake and can reach nothing else.
    Tcp(&'a str),
    /// A servicer's own engine link: `tcp://`, or an `ipc://` socket for a
    /// loopback engine (the test harness binds one per test instead of a
    /// probed port two tests can share).
    TcpOrIpc(&'a str),
}

impl<'a> Handshake<'a> {
    /// A registered worker's handshake: its override, or the derived address.
    pub fn registered(handshake_override: Option<&'a str>) -> Self {
        handshake_override.map_or(Self::Derived, Self::Tcp)
    }
}

/// Derive the ZMQ socket addresses for a worker from its base URL.
///
/// A `tcp://<host>:<port>` base serves engines on other hosts: the gateway
/// binds its handshake at that address itself and its input/output sockets
/// on ephemeral ports of the same host, which the INIT hands the engines, so
/// the host must be one they can reach (not a wildcard). The engines dial
/// `--data-parallel-address <host> --data-parallel-rpc-port <port>`. These
/// sockets carry no authentication: such a worker belongs on a trusted
/// network. The rest of this describes the same-host `ipc://` base.
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
/// `ts serve --headless` with a manually registered worker. A servicer's own
/// link may name an `ipc://` handshake instead ([`Handshake::TcpOrIpc`]): its
/// engine is on the same host, and the test harness binds one socket per test.
/// Returns `(handshake, input, output)`.
///
/// [`zmq_handshake_address`] exposes just the handshake half, for the
/// registration-time validation of that address.
pub(crate) fn zmq_socket_addresses(
    base_url: &str,
    handshake: Handshake<'_>,
) -> Result<(String, String, String), String> {
    if let Some(authority) = base_url.strip_prefix("tcp://") {
        return tcp_socket_addresses(base_url, authority, handshake);
    }
    let path = base_url.strip_prefix("ipc://").ok_or_else(|| {
        format!("ZMQ worker URL must be ipc://<path> or tcp://<host>:<port>, got '{base_url}'")
    })?;
    let handshake = match handshake {
        Handshake::Derived => format!("tcp://{ZMQ_LOOPBACK_HOST}:{}", derive_handshake_port(path)),
        Handshake::Tcp(address) => {
            if !address.starts_with("tcp://") {
                return Err(format!(
                    "zmq_handshake_address must be a tcp:// address \
                     (the engine dials a TCP handshake), got '{address}'"
                ));
            }
            address.to_string()
        }
        Handshake::TcpOrIpc(address) => {
            if !address.starts_with("tcp://") && !address.starts_with("ipc://") {
                return Err(format!(
                    "handshake address must be tcp://host:port or ipc://<path>, got '{address}'"
                ));
            }
            address.to_string()
        }
    };
    let input = format!("ipc://{path}-in.sock");
    let output = format!("ipc://{path}-out.sock");
    Ok((handshake, input, output))
}

/// [`zmq_socket_addresses`] for a `tcp://<host>:<port>` base (`authority` is
/// what follows the scheme): the handshake at the base itself, the data plane
/// on ephemeral ports of its host.
fn tcp_socket_addresses(
    base_url: &str,
    authority: &str,
    handshake: Handshake<'_>,
) -> Result<(String, String, String), String> {
    if !matches!(handshake, Handshake::Derived) {
        return Err(format!(
            "ZMQ worker {base_url} is the handshake address its engines dial; it takes no \
             separate handshake address"
        ));
    }
    let (host, _port) = authority
        .rsplit_once(':')
        .filter(|(host, port)| {
            // An IPv6 literal rides in brackets: a bare one cannot be told
            // apart from its port, since the split is at the last colon.
            let bracketed = host.starts_with('[') && host.ends_with(']');
            !host.is_empty()
                && (bracketed || !host.contains(':'))
                && port.parse::<u16>().is_ok_and(|p| p != 0)
        })
        .ok_or_else(|| format!("ZMQ worker URL must be tcp://<host>:<port>, got '{base_url}'"))?;
    let unbracketed = host.trim_start_matches('[').trim_end_matches(']');
    if unbracketed
        .parse::<std::net::IpAddr>()
        .is_ok_and(|ip| ip.is_unspecified())
    {
        return Err(format!(
            "ZMQ worker {base_url} names a wildcard host; the gateway hands its engines the \
             addresses it binds, so name a host they can reach"
        ));
    }
    let data_plane = format!("tcp://{host}:0");
    Ok((base_url.to_string(), data_plane.clone(), data_plane))
}

/// The TCP handshake address a ZMQ worker will bind.
///
/// Carries `zmq_socket_addresses`'s error verbatim rather than flattening it
/// to "no address": an unusable `ipc://` base or a non-`tcp://` override is a
/// misconfiguration registration must reject, not a value to skip past and
/// rediscover at every later connect attempt.
pub fn zmq_handshake_address(
    base_url: &str,
    handshake_override: Option<&str>,
) -> Result<String, String> {
    zmq_socket_addresses(base_url, Handshake::registered(handshake_override))
        .map(|(handshake, _, _)| handshake)
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
    // A `tcp://` worker binds no socket files.
    let Some(path) = base_url.strip_prefix("ipc://") else {
        return Ok(());
    };
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
/// owner-verified socket dir. A socket that disappears between the stat and
/// the unlink counts as removed: the step's job is a free path, and it is.
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
            match tokio::fs::remove_file(path).await {
                Ok(()) => Ok(()),
                // Gone between the stat and the unlink (a previous bind's
                // teardown task, or a sweep beside this one, took it): the
                // path is free, which is all the bind needs.
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(e) => Err(format!("failed to remove stale ipc socket {path}: {e}")),
            }
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

    /// The step's job is a free path, so a socket that vanishes between the
    /// stat and the unlink counts as removed. A previous bind's teardown
    /// unlinks its socket files on their own tasks, and a connect retry sweeps
    /// the same paths while those tasks may still be running: the sweep used
    /// to fail the whole connect with NotFound although the path was free.
    #[cfg(unix)]
    #[tokio::test]
    async fn unlink_stale_socket_counts_a_socket_that_vanishes_under_it_as_removed() {
        use std::{
            os::unix::net::UnixListener,
            sync::{
                atomic::{AtomicBool, Ordering},
                Arc,
            },
        };

        let dir = tempfile::tempdir().unwrap();
        let sock_path = dir.path().join("vanishing.sock");
        let addr = format!("ipc://{}", sock_path.display());
        // A thread creates and unlinks the socket as fast as it can, so the
        // helper's stat keeps finding a socket that its unlink then misses.
        let stop = Arc::new(AtomicBool::new(false));
        let churn = std::thread::spawn({
            let (path, stop) = (sock_path.clone(), Arc::clone(&stop));
            move || {
                let mut cycles = 0u32;
                while !stop.load(Ordering::Relaxed) {
                    // Only this thread creates the file, and it is unlinked
                    // (here or by the helper) before the next bind, so the
                    // bind never finds the path taken.
                    drop(UnixListener::bind(&path).expect("bind the churned socket"));
                    let _ = std::fs::remove_file(&path);
                    cycles += 1;
                }
                cycles
            }
        });
        let mut verdict = Ok(());
        for round in 0..1_000 {
            if let Err(e) = unlink_stale_socket(&addr).await {
                verdict = Err(format!("round {round}: {e}"));
                break;
            }
        }
        stop.store(true, Ordering::Relaxed);
        let cycles = churn.join().expect("the churn thread");
        assert!(cycles > 0, "the churn thread never cycled the socket");
        verdict.expect("a socket that vanishes between the stat and the unlink counts as removed");
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
            zmq_socket_addresses("ipc:///tmp/smg-zmq/ts0.ipc", Handshake::Derived).unwrap();
        assert_eq!(handshake, "tcp://127.0.0.1:25152");
        assert_eq!(input, "ipc:///tmp/smg-zmq/ts0.ipc-in.sock");
        assert_eq!(output, "ipc:///tmp/smg-zmq/ts0.ipc-out.sock");
    }

    #[test]
    fn a_servicers_handshake_may_be_ipc_and_nothing_else_beyond_tcp() {
        // The servicer's own link binds an ipc:// socket for a loopback engine
        // (one per test); any other scheme is still a misconfiguration.
        let (handshake, input, _) = zmq_socket_addresses(
            "ipc:///tmp/smg-zmq/ts0.ipc",
            Handshake::TcpOrIpc("ipc:///tmp/t/handshake"),
        )
        .unwrap();
        assert_eq!(handshake, "ipc:///tmp/t/handshake");
        assert_eq!(input, "ipc:///tmp/smg-zmq/ts0.ipc-in.sock");
        let refused = zmq_socket_addresses(
            "ipc:///tmp/smg-zmq/ts0.ipc",
            Handshake::TcpOrIpc("udp://x:1"),
        )
        .unwrap_err();
        assert!(refused.contains("tcp://"), "{refused}");
    }

    #[test]
    fn zmq_socket_addresses_reject_non_tcp_override() {
        // The engine dials a TCP handshake; a non-tcp override on a registered
        // worker is a config error and must fail loudly rather than bind
        // something no engine can reach.
        let err = zmq_socket_addresses(
            "ipc:///tmp/smg-zmq/ts0.ipc",
            Handshake::Tcp("ipc:///tmp/hs.sock"),
        )
        .unwrap_err();
        assert!(
            err.contains("tcp://"),
            "error must name the required scheme: {err}"
        );
        assert!(
            zmq_handshake_address("ipc:///tmp/smg-zmq/ts0.ipc", Some("ipc:///tmp/hs.sock"))
                .is_err(),
            "registration validation keeps the tcp-only rule"
        );
    }

    #[test]
    fn zmq_socket_addresses_honor_handshake_override() {
        // TokenSpeed's default dial target — outside the derived band; the
        // override must be bound verbatim while the data plane stays derived.
        let (handshake, input, output) = zmq_socket_addresses(
            "ipc:///tmp/smg-zmq/ts0.ipc",
            Handshake::Tcp("tcp://127.0.0.1:30500"),
        )
        .unwrap();
        assert_eq!(handshake, "tcp://127.0.0.1:30500");
        assert_eq!(input, "ipc:///tmp/smg-zmq/ts0.ipc-in.sock");
        assert_eq!(output, "ipc:///tmp/smg-zmq/ts0.ipc-out.sock");
    }

    /// A `tcp://` worker binds its handshake at its own address and its data
    /// plane on ephemeral ports of the same host, for engines on other hosts.
    #[test]
    fn a_tcp_worker_binds_its_handshake_at_its_url_and_its_data_plane_beside_it() {
        let (handshake, input, output) =
            zmq_socket_addresses("tcp://10.0.0.5:31000", Handshake::Derived).unwrap();
        assert_eq!(handshake, "tcp://10.0.0.5:31000");
        assert_eq!(input, "tcp://10.0.0.5:0");
        assert_eq!(output, "tcp://10.0.0.5:0");
        let (_, input, _) =
            zmq_socket_addresses("tcp://[fd00::5]:31000", Handshake::Derived).unwrap();
        assert_eq!(input, "tcp://[fd00::5]:0");
        assert_eq!(
            zmq_handshake_address("tcp://node0:31000", None).unwrap(),
            "tcp://node0:31000"
        );
    }

    /// A `tcp://` worker named by hostname keeps the name on its data plane:
    /// the engines resolve it as they resolved the handshake address.
    #[test]
    fn a_tcp_worker_named_by_hostname_keeps_the_name_on_its_data_plane() {
        let (handshake, input, output) =
            zmq_socket_addresses("tcp://node0:31000", Handshake::Derived).unwrap();
        assert_eq!(handshake, "tcp://node0:31000");
        assert_eq!(input, "tcp://node0:0");
        assert_eq!(output, "tcp://node0:0");
    }

    /// The INIT hands the engines the bound addresses, so a wildcard host (no
    /// engine can dial it), a missing port, a bare IPv6 host without brackets
    /// (it cannot be told apart from its port), and a second handshake address
    /// are each refused.
    #[test]
    fn a_tcp_worker_refuses_a_wildcard_host_a_missing_port_and_an_override() {
        for url in ["tcp://0.0.0.0:31000", "tcp://[::]:31000"] {
            let err = zmq_socket_addresses(url, Handshake::Derived).unwrap_err();
            assert!(err.contains("wildcard"), "{err}");
        }
        for url in [
            "tcp://10.0.0.5",
            "tcp://10.0.0.5:0",
            "tcp://:31000",
            "tcp://10.0.0.5:x",
            "tcp://fd00::5",
            "tcp://fd00::5:31000",
        ] {
            let err = zmq_socket_addresses(url, Handshake::Derived).unwrap_err();
            assert!(err.contains("tcp://<host>:<port>"), "{url}: {err}");
        }
        let err = zmq_handshake_address("tcp://10.0.0.5:31000", Some("tcp://10.0.0.5:30500"))
            .unwrap_err();
        assert!(err.contains("takes no separate handshake address"), "{err}");
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
