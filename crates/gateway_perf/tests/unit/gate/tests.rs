use std::collections::BTreeMap;

use super::*;
use crate::cells::tests::results_of;

const GATES: &str = r#"
repeats = 3

[[gate]]
pattern = "/allocation_calls_per_request$"
threshold = 0.02

[[gate]]
pattern = "/user_instructions_per_request_M$"
threshold = 0.02
"#;

fn verdicts(results: &Results, gates: &Gates) -> BTreeMap<String, String> {
    compare(results, gates)
        .unwrap()
        .into_iter()
        .map(|r| (r.bench_id, r.verdict))
        .collect()
}

fn gated(pattern: &str) -> Gates {
    parse_gates(&format!(
        "[[gate]]\npattern = {pattern:?}\nthreshold = 0.02\n"
    ))
    .unwrap()
}

#[test]
fn gate_fails_only_a_separated_change_beyond_the_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.toml");
    fs::write(&path, GATES).unwrap();
    let separated = results_of(
        &[("chat/allocation_calls_per_request", &[300.0, 301.0, 300.0])],
        &[("chat/allocation_calls_per_request", &[320.0, 321.0, 319.0])],
    );
    let (md, rc) = gate_report(&separated, Some(&path)).unwrap();
    assert_eq!(rc, 1);
    assert!(md.contains("FAIL regression"), "{md}");
    assert!(md.contains("### Benchmark gate: FAIL"), "{md}");
    assert!(
        md.contains("| 300 (300 - 301) | 320 (319 - 321) | +6.7% |"),
        "{md}"
    );
    let overlapping = results_of(
        &[("chat/allocation_calls_per_request", &[300.0, 330.0, 300.0])],
        &[("chat/allocation_calls_per_request", &[320.0, 321.0, 319.0])],
    );
    let (md, rc) = gate_report(&overlapping, Some(&path)).unwrap();
    assert_eq!(rc, 0);
    assert!(md.contains("suspect (runs overlap)"), "{md}");
    let small = results_of(
        &[("chat/allocation_calls_per_request", &[300.0, 301.0, 300.0])],
        &[("chat/allocation_calls_per_request", &[303.0, 303.0, 302.0])],
    );
    let (md, rc) = gate_report(&small, Some(&path)).unwrap();
    assert_eq!(rc, 0);
    assert!(md.contains("| ok |"), "{md}");
}

#[test]
fn gate_reports_ungated_ids_and_improvements_without_failing() {
    let gates = parse_gates(GATES).unwrap();
    let res = results_of(
        &[
            ("chat/user_instructions_per_request_M", &[0.50, 0.50, 0.50]),
            ("chat/cpu_us_per_request", &[700.0, 720.0, 690.0]),
        ],
        &[
            ("chat/user_instructions_per_request_M", &[0.45, 0.45, 0.45]),
            ("chat/cpu_us_per_request", &[900.0, 910.0, 905.0]),
        ],
    );
    let rows = compare(&res, &gates).unwrap();
    assert!(!rows.iter().any(Row::fails));
    let md = markdown(&res, &rows);
    assert!(md.contains("faster improvement"), "{md}");
    assert!(
        md.contains("| chat/cpu_us_per_request | 700 (690 - 720) | 905 (900 - 910) | +29.3% |"),
        "{md}"
    );
    assert!(md.contains("report only"), "{md}");
}

#[test]
fn gate_without_a_file_is_report_only() {
    let res = results_of(&[("a/x", &[1.0, 1.0, 1.0])], &[("a/x", &[2.0, 2.0, 2.0])]);
    let (md, rc) = gate_report(&res, None).unwrap();
    assert_eq!(rc, 0);
    assert!(md.contains("report only") && md.contains("0 gated"), "{md}");
}

#[test]
fn a_zero_base_median_is_judged_on_the_absolute_values() {
    let gates = gated("encode/");
    let zero = results_of(
        &[("encode/zero", &[0.0, 0.0, 0.0])],
        &[("encode/zero", &[0.0, 0.0, 0.0])],
    );
    assert_eq!(verdicts(&zero, &gates)["encode/zero"], "ok");
    let grown = results_of(
        &[("encode/zero", &[0.0, 0.0, 0.0])],
        &[("encode/zero", &[2.0, 2.0, 2.0])],
    );
    assert_eq!(verdicts(&grown, &gates)["encode/zero"], "regression");
    let text = markdown(&grown, &compare(&grown, &gates).unwrap());
    assert!(text.contains("| - |"), "{text}");
    let overlapping = results_of(
        &[("encode/zero", &[0.0, 0.0, 0.0])],
        &[("encode/zero", &[2.0, 0.0, 2.0])],
    );
    assert_eq!(
        verdicts(&overlapping, &gates)["encode/zero"],
        "suspect (runs overlap)"
    );
    let below = results_of(
        &[("encode/zero", &[0.0, 0.0, 0.0])],
        &[("encode/zero", &[-1.0, -1.0, -1.0])],
    );
    assert_eq!(verdicts(&below, &gates)["encode/zero"], "improvement");
}

#[test]
fn a_negative_base_median_keeps_the_direction() {
    let gates = gated("encode/");
    let rose = results_of(
        &[("encode/neg", &[-2.0, -2.0, -2.0])],
        &[("encode/neg", &[-1.0, -1.0, -1.0])],
    );
    assert!((compare(&rose, &gates).unwrap()[0].change().unwrap() - 0.5).abs() < 1e-12);
    assert_eq!(verdicts(&rose, &gates)["encode/neg"], "regression");
    let fell = results_of(
        &[("encode/neg", &[-1.0, -1.0, -1.0])],
        &[("encode/neg", &[-2.0, -2.0, -2.0])],
    );
    assert_eq!(verdicts(&fell, &gates)["encode/neg"], "improvement");
    let noisy = results_of(
        &[("encode/neg", &[-2.2, -2.0, -1.8])],
        &[("encode/neg", &[-2.0, -2.0, -2.0])],
    );
    assert!((compare(&noisy, &gates).unwrap()[0].spread().unwrap() - 0.2).abs() < 1e-12);
}

#[test]
fn a_results_file_that_names_its_unit_prints_plain_values() {
    let gates = gated("encode/");
    let mut res = results_of(
        &[("encode/calls", &[300.0, 301.0])],
        &[("encode/calls", &[290.0, 289.0])],
    );
    res.unit = Some(String::new());
    let text = markdown(&res, &compare(&res, &gates).unwrap());
    assert!(
        text.contains("| 300.5 (300 - 301) | 289.5 (289 - 290) |"),
        "{text}"
    );
    res.unit = Some("KiB".to_string());
    let text = markdown(&res, &compare(&res, &gates).unwrap());
    assert!(text.contains("| 300.5 KiB (300 KiB - 301 KiB) |"), "{text}");
    let mut timed = results_of(
        &[("encode/t", &[1500.0, 1600.0])],
        &[("encode/t", &[1550.0, 1650.0])],
    );
    timed.unit = None;
    let text = markdown(&timed, &compare(&timed, &gates).unwrap());
    assert!(text.contains("1.55 µs (1.5 µs - 1.6 µs)"), "{text}");
}

#[test]
fn a_missing_gated_id_fails_and_is_explained() {
    let gates = gated("^a/");
    let res = results_of(
        &[("a/x", &[1.0, 1.0]), ("a/y", &[2.0, 2.0])],
        &[("a/x", &[1.0, 1.0])],
    );
    let rows = compare(&res, &gates).unwrap();
    let md = markdown(&res, &rows);
    assert!(rows
        .iter()
        .any(|r| r.bench_id == "a/y" && r.verdict == "missing"));
    assert!(
        md.contains("`a/y` is gated but the pull request's run has no such id"),
        "{md}"
    );
    let regressed = results_of(&[("a/x", &[1.0, 1.0, 1.0])], &[("a/x", &[2.0, 2.0, 2.0])]);
    let md = markdown(&regressed, &compare(&regressed, &gates).unwrap());
    assert!(
        md.contains("base runs [1.0, 1.0, 1.0], pull request runs [2.0, 2.0, 2.0]."),
        "{md}"
    );
}

#[test]
fn a_group_pattern_excludes_a_renamed_id_with_a_negative_lookahead() {
    let gates = parse_gates(
        "[[gate]]\npattern = \"^(?!latency/encode_4KB$)(encode_throughput|latency)/\"\nthreshold = 0.05\n",
    )
    .unwrap();
    assert!(gates.threshold_for("latency/encode_1KB").unwrap().is_some());
    assert!(gates
        .threshold_for("encode_throughput/x")
        .unwrap()
        .is_some());
    assert!(gates.threshold_for("latency/encode_4KB").unwrap().is_none());
    assert!(gates.threshold_for("decode/x").unwrap().is_none());
}

/// A back-reference pattern over nested repeats runs past the backtracking limit on
/// a long id: the match fails, and the failure is an error naming the pattern and the
/// id, not a miss that would leave the id report-only.
#[test]
fn a_pattern_that_fails_at_match_time_is_an_error_not_a_skipped_gate() {
    let gates = gated("^(a+)+\\1$");
    let id = format!("{}b", "a".repeat(32));
    let err = gates.threshold_for(&id).unwrap_err().to_string();
    assert!(
        err.starts_with("pattern \"^(a+)+") && err.contains(&id),
        "{err}"
    );
    let res = results_of(&[(&id, &[1.0])], &[(&id, &[1.0])]);
    let err = compare(&res, &gates).unwrap_err().to_string();
    assert!(err.contains(&id), "{err}");
    assert!(gates.threshold_for("a/short").unwrap().is_none());
}

#[test]
fn the_gate_file_is_checked() {
    let gates = parse_gates(GATES).unwrap();
    assert_eq!(gates.repeats, 3);
    assert_eq!(gates.gates.len(), 2);
    assert!(gates
        .threshold_for("chat/allocation_calls_per_request")
        .unwrap()
        .is_some());
    assert!(gates
        .threshold_for("chat/cpu_us_per_request")
        .unwrap()
        .is_none());
    for (text, what) in [
        ("nonsense = 1\n", "unknown keys"),
        ("repeats = 0\n", "at least 1"),
        ("repeats = \"x\"\n", "must be an integer"),
        ("[[gate]]\npattern = \"x\"\n", "needs pattern and threshold"),
        (
            "[[gate]]\npattern = \"x\"\nthreshold = 1.5\n",
            "fraction in (0, 1)",
        ),
        (
            "[[gate]]\npattern = \"(\"\nthreshold = 0.1\n",
            "does not compile",
        ),
        (
            "[[gate]]\npattern = \"x\"\nthreshold = 0.1\nodd = 1\n",
            "unknown keys",
        ),
        ("filter = \"(\"\n", "does not compile"),
    ] {
        let err = parse_gates(text).unwrap_err().to_string();
        assert!(err.contains(what), "{text:?}: {err}");
    }
    let int = parse_gates("[[gate]]\npattern = \"x\"\nthreshold = 0.5\nnote = \"n\"\n").unwrap();
    assert_eq!(int.gates[0].note, "n");
}
