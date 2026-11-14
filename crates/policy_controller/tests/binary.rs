//! The binary as a process: what a start writes when nothing renders it.
//! The output here is a pipe, as it is under a container runtime that
//! collects it or a shell that redirects it into a file.

use std::process::{Command, Output, Stdio};

const BIN: &str = env!("CARGO_BIN_EXE_smg-policy-controller");

/// A start the controller refuses: the settle window is over half the grace
/// period. It writes one line through the logger, binds nothing and exits 1.
const REFUSED_START: [&str; 8] = [
    "--controller-id",
    "controller-0",
    "--runtime-worker-threads",
    "1",
    "--shutdown-settle-secs",
    "16",
    "--shutdown-grace-period-secs",
    "30",
];

/// The refused start's output into a pipe, with the color variables and
/// `RUST_LOG` of the test's own environment out of the way and `set` added.
fn refused_start(set: &[(&str, &str)]) -> std::io::Result<Output> {
    let mut command = Command::new(BIN);
    command.args(REFUSED_START).stdin(Stdio::null());
    for unset in ["NO_COLOR", "FORCE_COLOR", "CLICOLOR_FORCE", "RUST_LOG"] {
        command.env_remove(unset);
    }
    command.envs(set.iter().copied()).output()
}

/// The one line of a refused start, checked to be that line.
fn log_of(output: Output) -> Vec<u8> {
    assert_eq!(output.status.code(), Some(1), "a refused start exits 1");
    let log = [output.stdout, output.stderr].concat();
    let text = String::from_utf8_lossy(&log);
    assert!(
        text.contains("start refused"),
        "the refusal is logged: {text:?}"
    );
    log
}

/// An escape sequence in a pipe is bytes on every line that nothing renders
/// and a log collector stores: the text log is plain off a terminal, and a
/// force variable set to `0` does not force.
#[test]
fn the_text_log_is_plain_off_a_terminal() {
    for set in [&[][..], &[("FORCE_COLOR", "0")], &[("CLICOLOR_FORCE", "0")]] {
        let log = log_of(refused_start(set).expect("the binary runs"));
        assert!(
            !log.contains(&0x1b),
            "an escape byte off a terminal under {set:?}: {:?}",
            String::from_utf8_lossy(&log)
        );
    }
}

/// An explicit ask wins over the pipe, as it does in the gateway.
#[test]
fn a_forced_color_reaches_a_pipe() {
    for variable in ["FORCE_COLOR", "CLICOLOR_FORCE"] {
        let log = log_of(refused_start(&[(variable, "1")]).expect("the binary runs"));
        assert!(
            log.contains(&0x1b),
            "no escape byte under {variable}=1: {:?}",
            String::from_utf8_lossy(&log)
        );
    }
}
