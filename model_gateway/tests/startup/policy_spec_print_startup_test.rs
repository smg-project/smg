//! `--print-policy-spec` answers with the policy document of a command line
//! and starts nothing: the document alone on stdout and a zero exit, or one
//! sentence on stderr and a failing exit. It reads the routing part of the
//! line and no file.

use std::{
    io,
    process::{Command, Output},
};

use smg_grand_central::document::PolicySpec;

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

/// Whether `stderr` is the binary's one sentence of error that begins with
/// `head`: one line, ended by one line break. The words behind the head are
/// the Grand Central crate's, and no case of the gateway holds them.
fn is_one_sentence_headed(stderr: &str, head: &str) -> bool {
    stderr
        .strip_prefix("error: ")
        .and_then(|sentence| sentence.strip_prefix(head))
        .and_then(|words| words.strip_suffix('\n'))
        .is_some_and(|words| !words.is_empty() && !words.contains('\n'))
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

    // A pair mode under the sticky override has its document too: the
    // override's rule stands on the pair's binding and on the binding of
    // single requests behind it.
    let answer = print(&[
        "--pd-disaggregation",
        "--prefill",
        "http://prefill-a:8000",
        "--decode",
        "http://decode-a:8000",
        "--routing-key-override",
    ])
    .expect("the binary runs");
    assert_eq!(answer.status.code(), Some(0), "{}", text(&answer.stderr));
    assert_eq!(text(&answer.stderr), "");
    let document: PolicySpec =
        serde_yaml::from_slice(&answer.stdout).expect("stdout is a document and nothing else");
    let rules: Vec<bool> = document
        .bindings
        .iter()
        .map(|binding| binding.sessions.is_some())
        .collect();
    assert_eq!(rules, [true, true]);

    // Beside a prefill under `consistent_hashing` the rule on the pair's
    // binding names the decode side, which it covers alone.
    let answer = print(&[
        "--pd-disaggregation",
        "--prefill",
        "http://prefill-a:8000",
        "--decode",
        "http://decode-a:8000",
        "--prefill-policy",
        "consistent_hashing",
        "--routing-key-override",
    ])
    .expect("the binary runs");
    assert_eq!(answer.status.code(), Some(0), "{}", text(&answer.stderr));
    assert_eq!(text(&answer.stderr), "");
    let document: PolicySpec =
        serde_yaml::from_slice(&answer.stdout).expect("stdout is a document and nothing else");
    // Each of the two bindings holds a rule, and the pair's alone names a
    // side.
    assert_eq!(document.bindings.len(), 2);
    let sides: Vec<Option<Vec<String>>> = document
        .bindings
        .iter()
        .filter_map(|binding| Some(binding.sessions.as_ref()?.sides.clone()))
        .collect();
    assert_eq!(sides, [Some(vec!["decode".to_string()]), None]);

    // A pair whose main policy is the hash policy itself, beside a decode
    // side the override places. No binding of the override stands behind
    // the pair's: its rule is the document's one, and the view lists no
    // client header of the hash side.
    let answer = print(&[
        "--pd-disaggregation",
        "--prefill",
        "http://prefill-a:8000",
        "--decode",
        "http://decode-a:8000",
        "--policy",
        "consistent_hashing",
        "--decode-policy",
        "round_robin",
        "--routing-key-override",
    ])
    .expect("the binary runs");
    assert_eq!(answer.status.code(), Some(0), "{}", text(&answer.stderr));
    assert_eq!(text(&answer.stderr), "");
    let document: PolicySpec =
        serde_yaml::from_slice(&answer.stdout).expect("stdout is a document and nothing else");
    let rules: Vec<bool> = document
        .bindings
        .iter()
        .map(|binding| binding.sessions.is_some())
        .collect();
    assert_eq!(rules, [true, false]);
    assert!(!document.view.headers.iter().any(|name| name == "cookie"));
}

#[test]
fn a_setting_no_document_says_is_one_sentence_on_stderr_and_a_failing_exit() {
    // The sticky override's rule has no document in a gateway of many
    // routers yet.
    let answer = print(&["--enable-igw", "--routing-key-override"]).expect("the binary runs");
    assert_eq!(answer.status.code(), Some(1));
    assert_eq!(text(&answer.stdout), "");
    let stderr = text(&answer.stderr);
    assert!(
        is_one_sentence_headed(&stderr, "routing_key_override.enabled = true: "),
        "{stderr}"
    );
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
    let stderr = text(&answer.stderr);
    assert!(
        is_one_sentence_headed(&stderr, "profiles.cache_aware.score[1].prefix-affinity: "),
        "{stderr}"
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
