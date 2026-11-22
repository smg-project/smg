//! The binary as a process: what a start writes when nothing renders it.
//! The output here is a pipe, as it is under a container runtime that
//! collects it or a shell that redirects it into a file.

use std::process::{Command, Output, Stdio};

const BIN: &str = env!("CARGO_BIN_EXE_smg-policy-controller");

/// A start the controller refuses: the settle window is over half the grace
/// period. It writes one line through the logger, binds nothing and exits 1.
/// It names a spec source, as every start does, and opens none.
const REFUSED_START: [&str; 10] = [
    "--controller-id",
    "controller-0",
    "--spec-store",
    "spec",
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

/// The binary runs on the allocator the gateway's binary registers, with a
/// copy of its settings. A burst of registrations builds the store's world
/// again at every write, and the system allocator keeps resident what those
/// builds free. The allocator is asked for its statistics at the process's
/// end, an ask the system allocator does not hear, and the statistics name
/// the settings it runs on.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
#[test]
fn the_binary_runs_on_the_gateways_allocator() {
    let ask = [("_RJEM_MALLOC_CONF", "stats_print:true")];
    let output = refused_start(&ask).expect("the binary runs");
    let written = String::from_utf8_lossy(&output.stderr);
    assert!(
        written.contains("jemalloc statistics"),
        "no statistics of the allocator at the end: {written:?}"
    );
    for setting in [
        // The allocator compiles no background thread for an Apple target.
        #[cfg(not(target_vendor = "apple"))]
        "opt.background_thread: true",
        "opt.dirty_decay_ms: 10000 (",
        "opt.muzzy_decay_ms: 0 (",
        "opt.narenas: 16\n",
        "opt.oversize_threshold: 33554432\n",
        // The workspace's cargo configuration builds the allocator of an
        // aarch64 binary for 64 KiB pages, the largest its kernels run with.
        #[cfg(all(target_arch = "aarch64", target_env = "gnu"))]
        "Page size: 65536\n",
    ] {
        assert!(
            written.contains(setting),
            "the allocator does not run on {setting:?}: {written:?}"
        );
    }
}

/// The binary's settings are a copy of the gateway's constant, which this
/// crate cannot import. The gateway's source is read by path: the copy
/// moves with the constant, or this case fails.
#[test]
fn the_binarys_settings_are_the_gateways_constant() {
    let gateway = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../model_gateway/src/observability/metrics.rs"
    ));
    let binary = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/bin/smg-policy-controller.rs"
    ));
    let theirs =
        settings_after(gateway, "pub const SERVER_MALLOC_CONF").expect("the gateway's constant");
    let ours = settings_after(binary, "const SETTINGS").expect("the binary's copy");
    assert_eq!(ours, theirs, "the binary's settings are not the gateway's");
}

/// The example controller's settings are a copy of this binary's, in a file
/// a user copies out of the tree: no case of the copy can read this one.
/// The example's source is read by path: its copy moves with the binary's,
/// or this case fails.
#[test]
fn the_examples_settings_are_the_binarys() {
    let binary = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/bin/smg-policy-controller.rs"
    ));
    let example = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/policy-controller/src/main.rs"
    ));
    let ours = settings_after(binary, "const SETTINGS").expect("the binary's settings");
    let theirs = settings_after(example, "pub static MALLOC_CONF").expect("the example's copy");
    assert_eq!(theirs, ours, "the example's settings are not the binary's");
}

/// The served bench counts over the allocator this binary registers, under
/// a copy of its settings: a row read under other settings is not the
/// binary's row. The bench's source is read by path: its copy moves with
/// the binary's, or this case fails.
#[test]
fn the_served_benchs_settings_are_the_binarys() {
    let binary = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/bin/smg-policy-controller.rs"
    ));
    let bench = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/benches/decide_stream.rs"
    ));
    let ours = settings_after(binary, "const SETTINGS").expect("the binary's settings");
    let theirs = settings_after(bench, "pub static MALLOC_CONF").expect("the bench's copy");
    assert_eq!(theirs, ours, "the bench's settings are not the binary's");
    // The in-process engine's bench counts under a copy of them too.
    let in_process = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/benches/in_process.rs"
    ));
    let theirs = settings_after(in_process, "pub static MALLOC_CONF").expect("its copy");
    assert_eq!(theirs, ours, "the in-process bench's settings differ");
}

/// The byte string a source assigns after `name`, without its quotes.
fn settings_after<'a>(source: &'a str, name: &str) -> Option<&'a str> {
    let (_, rest) = source.split_once(name)?;
    let (_, rest) = rest.split_once("b\"")?;
    rest.split_once('"').map(|(settings, _)| settings)
}
