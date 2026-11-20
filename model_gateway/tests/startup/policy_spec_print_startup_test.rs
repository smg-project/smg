//! `--print-policy-spec` answers with the policy document of a command line
//! and starts nothing: the document alone on stdout and a zero exit, or one
//! sentence on stderr and a failing exit. It reads the routing part of the
//! line and no file.

use std::{
    io,
    process::{Command, Output},
};

use smg_policy_controller::PolicySpec;

/// The binary's whole answer to `args` and the print option.
fn print(args: &[&str]) -> io::Result<Output> {
    Command::new(env!("CARGO_BIN_EXE_smg"))
        .args(args)
        .arg("--print-policy-spec")
        .output()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn the_print_is_the_document_alone_on_stdout_and_a_zero_exit() {
    let workers = ["http://worker-a:8000", "http://worker-b:8000"];
    let answer = print(&[
        "launch",
        "--worker-urls",
        workers[0],
        workers[1],
        "--policy",
        "cache_aware",
    ])
    .expect("the binary runs");
    assert_eq!(answer.status.code(), Some(0), "{}", text(&answer.stderr));
    assert_eq!(text(&answer.stderr), "");
    let document: PolicySpec =
        serde_yaml::from_slice(&answer.stdout).expect("stdout is a document and nothing else");
    assert_eq!(
        document,
        PolicySpec::preset("cache_aware").expect("a preset")
    );
}

#[test]
fn a_setting_no_document_says_is_one_sentence_on_stderr_and_a_failing_exit() {
    let answer = print(&["--dp-minimum-tokens-scheduler"]).expect("the binary runs");
    assert_eq!(answer.status.code(), Some(1));
    assert_eq!(text(&answer.stdout), "");
    assert_eq!(
        text(&answer.stderr),
        "error: dp_minimum_tokens_scheduler = true: the document has no rank policy: under the \
         controller a rank is a candidate of the profile\n"
    );
}

#[test]
fn the_sticky_override_prints_the_bindings_session_rule() {
    let answer = print(&["--routing-key-override"]).expect("the binary runs");
    assert_eq!(answer.status.code(), Some(0), "{}", text(&answer.stderr));
    assert_eq!(text(&answer.stderr), "");
    let document: PolicySpec =
        serde_yaml::from_slice(&answer.stdout).expect("stdout is a document and nothing else");
    let binding = document.bindings.first().expect("the document's binding");
    assert!(binding.sessions.is_some(), "the override's rule");
    assert!(binding.routing_key.is_some(), "the key's sources");
}

#[test]
fn a_document_this_build_does_not_run_prints_with_the_controllers_word() {
    let answer = print(&["--overlap-decay", "0.5"]).expect("the binary runs");
    assert_eq!(answer.status.code(), Some(1));
    let document: PolicySpec =
        serde_yaml::from_slice(&answer.stdout).expect("stdout is the document");
    assert_ne!(
        document,
        PolicySpec::preset("cache_aware").expect("a preset")
    );
    assert_eq!(
        text(&answer.stderr),
        "error: profiles.cache_aware.score[1].prefix-affinity: the prefix-affinity decay is \
         not in this build\n"
    );
}

#[test]
fn the_print_reads_no_secret_file() {
    // A start refuses this line at the key file it cannot read. The print
    // comes before the secrets and has its document.
    let answer = print(&["--api-key-file", "/nonexistent/smg-api-key"]).expect("the binary runs");
    assert_eq!(answer.status.code(), Some(0), "{}", text(&answer.stderr));
    assert!(!answer.stdout.is_empty());
}

#[test]
fn the_print_reads_no_file_a_start_reads() {
    // A start refuses each of these lines: the store has no address, and the
    // certificate, the key and the MCP configuration are not there. The
    // print reads none of those flags and no file, and has its document.
    let lines: [&[&str]; 3] = [
        &["--history-backend", "postgres"],
        &[
            "--tls-cert-path",
            "/nonexistent/cert.pem",
            "--tls-key-path",
            "/nonexistent/key.pem",
        ],
        &["--mcp-config-path", "/nonexistent/mcp.yaml"],
    ];
    for line in lines {
        let answer = print(line).expect("the binary runs");
        assert_eq!(text(&answer.stderr), "", "{line:?}");
        assert_eq!(answer.status.code(), Some(0), "{line:?}");
        assert!(!answer.stdout.is_empty(), "{line:?}");
    }
}

#[test]
fn the_tenant_header_as_a_routing_key_prints_with_the_gateways_refusal() {
    // The gateway is the controller's host and denies its tenant header to
    // the view. The document prints, with the refusal a hosted start gives.
    let answer = print(&[
        "--policy",
        "manual",
        "--routing-key-headers",
        "x-tenant-id",
        "--tenant-header-name",
        "x-tenant-id",
    ])
    .expect("the binary runs");
    assert_eq!(answer.status.code(), Some(1));
    assert!(!answer.stdout.is_empty());
    assert_eq!(
        text(&answer.stderr),
        "error: view.headers[0]: header \"x-tenant-id\" is on the deny-list\n"
    );
}
