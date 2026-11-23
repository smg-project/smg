//! `--dp-aware` is deprecated, accepted and ignored: a gateway started with
//! it serves and says once, at warn level, that the option does nothing, and
//! a gateway started without it says no such line. The gateway runs as a
//! process on purpose: a start line that carries the option must still start
//! the binary, and the line is the binary's own word.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::allow_attributes
)]

use std::{net::SocketAddr, path::Path};

use gateway_launch::{base_args, free_ports, log_with_line, serial, Gateway, EXIT_TIMEOUT, IP};

use crate::gateway_launch;

/// The deprecation line's own words: no other line of a start-up holds them.
const DEPRECATION: &str = "dp_aware (--dp-aware) is deprecated and does nothing";

/// The log of a gateway started with `flags` beside the common ones, read
/// once it serves and has logged its bind. The bind follows every warning
/// start-up says of the configuration and the lines are written in order, so
/// the log holds by then all that start-up said of it. The caller holds
/// [`serial`].
fn startup_log(flags: &[&str], log: &Path) -> String {
    let [port, metrics_port] = free_ports();
    let mut args = base_args(port, metrics_port);
    args.extend(flags.iter().map(|flag| (*flag).to_string()));
    let addr = SocketAddr::from((IP, port));
    let gateway = Gateway::start(&args, addr, log);
    let bound = format!("Starting server on {addr}");
    match log_with_line(&gateway.log, &bound, EXIT_TIMEOUT) {
        Ok(log) => log,
        Err(log) => panic!("{flags:?}: the gateway serves and did not log its bind:\n{log}"),
    }
}

#[test]
fn dp_aware_starts_the_gateway_and_is_said_once_to_do_nothing() {
    let _serial = serial();
    let dir = tempfile::tempdir().expect("a temp dir");

    let given = startup_log(&["--dp-aware"], &dir.path().join("given.log"));
    let said: Vec<&str> = given
        .lines()
        .filter(|line| line.contains(DEPRECATION))
        .collect();
    let [line] = said.as_slice() else {
        panic!("a gateway given --dp-aware says once that it does nothing:\n{given}");
    };
    assert!(line.contains("WARN"), "the line is a warning: {line}");

    let not_given = startup_log(&[], &dir.path().join("not-given.log"));
    assert!(
        !not_given.contains(DEPRECATION),
        "a gateway not given the option says nothing of it:\n{not_given}"
    );
}
