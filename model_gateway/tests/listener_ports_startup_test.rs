//! Two of the gateway's listeners on one port are refused before anything
//! binds, naming both flags; nothing binds before a configuration the
//! gateway could read but not use (TLS material, a storage hook) is refused;
//! and a listener port another process holds is refused without the mesh
//! server having joined the cluster. All used to be partial starts: a
//! listener came up, answered, and the process died. The gateway runs as a
//! process on purpose: the checks belong before the runtime and the
//! listeners exist.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::allow_attributes
)]

#[path = "common/gateway_launch.rs"]
mod gateway_launch;

use std::{fs, net::TcpListener};

use gateway_launch::{base_args, free_ports, launch_with, serial, Launch, Until, EXIT_TIMEOUT, IP};

/// The mesh server on `mesh_port`, bound and advertised on the loopback.
fn mesh_args(mesh_port: u16) -> Vec<String> {
    let ip = IP.to_string();
    [
        "--enable-mesh",
        "--mesh-host",
        &ip,
        "--mesh-port",
        &mesh_port.to_string(),
    ]
    .map(str::to_string)
    .to_vec()
}

/// A launch that must end in a refusal: no watched port ever answered, a
/// non-zero exit, no panic, and `named` in the error.
fn assert_refused(name: &str, launch: &Launch, named: &[&str]) {
    let log = &launch.log;
    assert!(
        launch.bound.is_none(),
        "{name}: port {:?} answered although the configuration is unusable\n{log}",
        launch.bound
    );
    let status = launch.status.unwrap_or_else(|| {
        panic!("{name}: the gateway neither exited nor listened within {EXIT_TIMEOUT:?}\n{log}")
    });
    assert!(
        !status.success(),
        "{name}: the gateway exited successfully\n{log}"
    );
    assert!(
        !log.contains("panicked"),
        "{name}: a panic is not a configuration error\n{log}"
    );
    for text in named {
        assert!(
            log.contains(text),
            "{name}: the error does not name {text:?}\n{log}"
        );
    }
}

/// Every pair of the HTTP, metrics and probe ports on one port, and the mesh
/// port on the HTTP port: refused naming both flags, with the port never
/// answering.
#[test]
fn equal_listener_ports_refuse_to_start() {
    let _serial = serial();
    let dir = tempfile::tempdir().expect("a temp dir");
    let [port, metrics_port, _] = free_ports();
    let p = port.to_string();
    let m = metrics_port.to_string();

    let same_metrics_port = base_args(port, port);
    let mut probe_on_http_port = base_args(port, metrics_port);
    probe_on_http_port.extend(["--health-check-port", &p].map(str::to_string));
    let mut probe_on_metrics_port = base_args(port, metrics_port);
    probe_on_metrics_port.extend(["--health-check-port", &m].map(str::to_string));
    let mut mesh_on_http_port = base_args(port, metrics_port);
    mesh_on_http_port.extend(mesh_args(port));

    let cases: [(&str, Vec<String>, [&str; 2]); 4] = [
        (
            "prometheus-port-eq-port",
            same_metrics_port,
            ["--prometheus-port", "--port"],
        ),
        (
            "health-check-port-eq-port",
            probe_on_http_port,
            ["--health-check-port", "--port"],
        ),
        (
            "health-check-port-eq-prometheus-port",
            probe_on_metrics_port,
            ["--health-check-port", "--prometheus-port"],
        ),
        (
            "mesh-port-eq-port",
            mesh_on_http_port,
            ["--mesh-port", "--port"],
        ),
    ];
    for (name, args, flags) in cases {
        let launch = launch_with(
            &args,
            &[port, metrics_port],
            Until::Bound,
            &dir.path().join(format!("{name}.log")),
        );
        assert_refused(name, &launch, &flags);
        assert!(
            launch.log.contains("each listener needs its own port"),
            "{name}: the error does not say what is wrong\n{}",
            launch.log
        );
    }
}

/// Configuration the gateway reads after its validation and could not use
/// (TLS files without a key, a storage hook that is not there) is refused
/// before any listener binds: the metrics port never answers either.
#[test]
fn a_late_configuration_error_binds_nothing() {
    let _serial = serial();
    let dir = tempfile::tempdir().expect("a temp dir");
    let empty = dir.path().join("empty.pem");
    fs::write(&empty, "").expect("write the file");
    let empty = empty.to_str().expect("a UTF-8 path").to_string();
    let missing_hook = dir
        .path()
        .join("missing.wasm")
        .to_str()
        .expect("a UTF-8 path")
        .to_string();

    let [port, metrics_port, _] = free_ports();
    let mut tls = base_args(port, metrics_port);
    tls.extend(["--tls-cert-path", &empty, "--tls-key-path", &empty].map(str::to_string));
    let mut hook = base_args(port, metrics_port);
    hook.extend(["--enable-wasm", "--storage-hook-wasm-path", &missing_hook].map(str::to_string));

    let cases: [(&str, Vec<String>, Vec<&str>); 2] = [
        ("tls-material-without-a-key", tls, vec!["TLS"]),
        ("storage-hook-missing", hook, vec![missing_hook.as_str()]),
    ];
    for (name, args, named) in cases {
        let launch = launch_with(
            &args,
            &[metrics_port, port],
            Until::Bound,
            &dir.path().join(format!("{name}.log")),
        );
        assert_refused(name, &launch, &named);
    }
}

/// A listener port another process holds is refused with the mesh server
/// never started: a member that joined the cluster and vanished with the
/// process is the partial start the checks exist to prevent. The metrics
/// port held (the first bind), then the HTTP port (the last one).
#[test]
fn a_held_listener_port_starts_no_mesh_member() {
    let _serial = serial();
    let dir = tempfile::tempdir().expect("a temp dir");
    for hold_http_port in [false, true] {
        let held = TcpListener::bind((IP, 0)).expect("a port to hold");
        let held_port = held.local_addr().expect("the held address").port();
        let [free_port, mesh_port, _] = free_ports();
        let (name, port, metrics_port, watch) = if hold_http_port {
            // The metrics and probe listeners are up when the HTTP bind
            // fails (they bind earlier, by design), and so is the mesh
            // listener's socket, bound just before it and dropped with the
            // failed start: no port is watched here; the log shows below
            // that the mesh server never served from that socket.
            ("port-held", held_port, free_port, vec![])
        } else {
            (
                "metrics-port-held",
                free_port,
                held_port,
                vec![mesh_port, free_port],
            )
        };
        let mut args = base_args(port, metrics_port);
        args.extend(mesh_args(mesh_port));

        let launch = launch_with(
            &args,
            &watch,
            Until::Bound,
            &dir.path().join(format!("{name}.log")),
        );
        assert_refused(name, &launch, &["Address already in use"]);
        assert!(
            !launch.log.contains("Mesh server listening on"),
            "{name}: the mesh server started although a listener port was held\n{}",
            launch.log
        );
        drop(held);
    }
}

/// The mesh port another process holds is refused like the other listeners'
/// ports, with nothing served: the mesh listener binds with them, before the
/// HTTP listener, where its failed bind used to be one error line from the
/// mesh server's task while the gateway served outside the mesh.
#[test]
fn a_held_mesh_port_refuses_to_start() {
    let _serial = serial();
    let dir = tempfile::tempdir().expect("a temp dir");
    let held = TcpListener::bind((IP, 0)).expect("a port to hold");
    let held_port = held.local_addr().expect("the held address").port();
    let [port, metrics_port, _] = free_ports();
    let mut args = base_args(port, metrics_port);
    args.extend(mesh_args(held_port));
    // The metrics and probe listeners are up when the mesh bind fails (they
    // bind earlier, by design): the HTTP port is watched, the one a gateway
    // outside the mesh used to serve on.
    let launch = launch_with(
        &args,
        &[port],
        Until::Bound,
        &dir.path().join("mesh-port-held.log"),
    );
    assert_refused(
        "mesh-port-held",
        &launch,
        &["--mesh-port", "Address already in use"],
    );
    drop(held);
}
