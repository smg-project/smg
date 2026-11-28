use super::*;
use crate::paths::path_named;

fn slo() -> Slo {
    Slo {
        ttft_p99_ms: 100.0,
        e2e_p99_ms: 1000.0,
        error_rate: 0.0,
        achieved_fraction: 0.95,
    }
}

fn step(offered: f64, ttft_p99: f64, achieved: Option<f64>, failed: u64) -> Step {
    let summary = json!({
        "sent": 1_000,
        "ok": 1_000 - failed,
        "failed": failed,
        "skipped": 0,
        "achieved_rps": achieved.unwrap_or(offered),
        "ttft_ms": {"p50": 1.0, "p99": ttft_p99},
        "e2e_ms": {"p50": 2.0, "p99": 10.0},
    });
    judge_step(&summary, 0.5, offered, &slo())
}

#[test]
fn capacity_is_the_highest_holding_step_with_the_strict_reading_beside() {
    let steps = vec![
        step(200.0, 5.0, None, 0),
        step(400.0, 500.0, None, 0),
        step(800.0, 5.0, None, 0),
        step(1200.0, 500.0, None, 0),
    ];
    let cap = capacity(steps);
    assert_eq!(cap.hold, Some(800.0));
    assert_eq!(cap.strict, Some(200.0));
    assert_eq!(cap.failing_steps_below_hold, [400.0]);
    assert!(!cap.at_ceiling);
    assert!(cap.rule.contains("highest offered step"));
    assert!(capacity(vec![step(200.0, 5.0, None, 0), step(400.0, 5.0, None, 0)]).at_ceiling);
    assert_eq!(capacity(vec![step(100.0, 900.0, None, 0)]).hold, None);
}

#[test]
fn a_step_fails_on_errors_or_an_achieved_shortfall() {
    assert!(!step(400.0, 5.0, None, 1).holds);
    let short = step(400.0, 5.0, Some(300.0), 0);
    assert!(!short.holds);
    assert!(
        short.reasons[0].starts_with("achieved 300 < 95% of offered 400 req/s"),
        "{:?}",
        short.reasons
    );
    let good = step(400.0, 5.0, None, 0);
    assert!(good.holds);
    assert!((good.cpu_ms_per_request.unwrap() - 0.5).abs() < 1e-12);
    let slow = step(400.0, 500.0, None, 0);
    assert_eq!(slow.reasons, ["TTFT p99 500.0 > 100.0 ms"]);
    let empty = judge_step(&json!({}), 0.0, 100.0, &slo());
    assert_eq!(empty.cpu_ms_per_request, None);
    assert!(
        empty
            .reasons
            .iter()
            .any(|r| r == "TTFT p99 None > 100.0 ms"),
        "{:?}",
        empty.reasons
    );
}

#[test]
fn a_request_the_drain_cut_off_counts_as_an_error() {
    let summary = json!({
        "sent": 1_000,
        "ok": 999,
        "failed": 0,
        "skipped": 0,
        "unfinished": 1,
        "achieved_rps": 400.0,
        "ttft_ms": {"p50": 1.0, "p99": 5.0},
        "e2e_ms": {"p50": 2.0, "p99": 10.0},
    });
    let cut = judge_step(&summary, 0.5, 400.0, &slo());
    assert!(!cut.holds);
    assert_eq!(cut.reasons.len(), 1, "{:?}", cut.reasons);
    assert!(
        cut.reasons[0].starts_with("errors 1/1000 > "),
        "{:?}",
        cut.reasons
    );
}

#[test]
fn the_markdown_carries_the_table_and_the_capacity_lines() {
    let ramp = Ramp {
        rates: vec![200.0, 400.0],
        step_seconds: 20,
        step_warmup: 3,
        slo: slo(),
    };
    let spec = path_named("chat").unwrap();
    let cap = capacity(vec![step(200.0, 5.0, None, 0), step(400.0, 500.0, None, 0)]);
    let text = live_markdown(spec, &ramp, &[("pr", cap)], "");
    assert!(
        text.starts_with("## Live: open-loop ramp on the chat path\n"),
        "{text}"
    );
    assert!(
        text.contains("| pr | 200 | 200 | 1.0 / 5.0 | 2.0 / 10.0 | 0/1000 | 0.5 | 0 | yes |"),
        "{text}"
    );
    assert!(text.contains("| pr | 400 | 400 | 1.0 / 500.0 | 2.0 / 10.0 | 0/1000 | 0.5 | 0 | no: TTFT p99 500.0 > 100.0 ms |"), "{text}");
    assert!(text.contains("- pr: capacity (hold) 200.0 req/s; strict first-break reading 200.0 req/s. Rule: hold:"), "{text}");
}
