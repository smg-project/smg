//! `live`: an open-loop ramp of offered rates on a fresh gateway per step, with the
//! capacity rule.

use std::{fs, thread, time::Duration};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    bed::{enqueue, Bed},
    cells::{run_client, ClientRun, Settings},
    counters::snapshot,
    hms,
    paths::{answers, PathSpec},
    py_float, py_g, round3, SIDES,
};

/// The service-level objective a step is judged against.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Slo {
    pub ttft_p99_ms: f64,
    pub e2e_p99_ms: f64,
    pub error_rate: f64,
    pub achieved_fraction: f64,
}

/// One step of the ramp.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Step {
    pub offered: f64,
    pub achieved: f64,
    pub sent: u64,
    pub ok: u64,
    pub failed: u64,
    pub ttft_p50_ms: Option<f64>,
    pub ttft_p99_ms: Option<f64>,
    pub e2e_p50_ms: Option<f64>,
    pub e2e_p99_ms: Option<f64>,
    pub cpu_ms_per_request: Option<f64>,
    pub holds: bool,
    pub reasons: Vec<String>,
    pub rss_mb_end: f64,
    pub loadavg: f64,
}

fn opt(v: Option<f64>) -> String {
    v.map_or_else(|| "None".to_string(), py_float)
}

fn count(v: &Value, key: &str) -> u64 {
    v.get(key).and_then(Value::as_u64).unwrap_or(0)
}

fn quantile(v: &Value, group: &str, q: &str) -> Option<f64> {
    v.get(group).and_then(|g| g.get(q)).and_then(Value::as_f64)
}

/// Whether a step holds the SLO, with the reasons it does not. A request the client
/// skipped or cut off at its drain counts as an error with the failed ones: it has no
/// latency sample, so the quantiles alone would judge the step on the survivors.
pub fn judge_step(summary: &Value, cpu_s: f64, offered: f64, slo: &Slo) -> Step {
    let sent = count(summary, "sent");
    let failed =
        count(summary, "failed") + count(summary, "skipped") + count(summary, "unfinished");
    let ttft = quantile(summary, "ttft_ms", "p99");
    let e2e = quantile(summary, "e2e_ms", "p99");
    let achieved = summary
        .get("achieved_rps")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let mut reasons = Vec::new();
    if ttft.is_none_or(|t| t > slo.ttft_p99_ms) {
        reasons.push(format!(
            "TTFT p99 {} > {} ms",
            opt(ttft),
            py_float(slo.ttft_p99_ms)
        ));
    }
    if e2e.is_none_or(|t| t > slo.e2e_p99_ms) {
        reasons.push(format!(
            "e2e p99 {} > {} ms",
            opt(e2e),
            py_float(slo.e2e_p99_ms)
        ));
    }
    if sent > 0 && failed as f64 / sent as f64 > slo.error_rate {
        reasons.push(format!(
            "errors {failed}/{sent} > {}",
            py_g(slo.error_rate, 6)
        ));
    }
    if achieved < offered * slo.achieved_fraction {
        reasons.push(format!(
            "achieved {achieved:.0} < {:.0}% of offered {} req/s",
            slo.achieved_fraction * 100.0,
            py_g(offered, 6)
        ));
    }
    Step {
        offered,
        achieved,
        sent,
        ok: count(summary, "ok"),
        failed,
        ttft_p50_ms: quantile(summary, "ttft_ms", "p50"),
        ttft_p99_ms: ttft,
        e2e_p50_ms: quantile(summary, "e2e_ms", "p50"),
        e2e_p99_ms: e2e,
        cpu_ms_per_request: (sent > 0).then(|| cpu_s * 1000.0 / sent as f64),
        holds: reasons.is_empty(),
        reasons,
        rss_mb_end: 0.0,
        loadavg: 0.0,
    }
}

/// The capacity over the steps.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Capacity {
    pub rule: String,
    pub hold: Option<f64>,
    pub strict: Option<f64>,
    pub failing_steps_below_hold: Vec<f64>,
    pub at_ceiling: bool,
    pub steps: Vec<Step>,
}

/// The capacity rule over steps sorted by offered rate: HOLD = the highest offered
/// step at which the SLO holds; STRICT = the last holding step before the first failing
/// one (the first-break reading). Both are reported; HOLD is the capacity.
pub fn capacity(mut steps: Vec<Step>) -> Capacity {
    steps.sort_by(|a, b| a.offered.total_cmp(&b.offered));
    let hold = steps
        .iter()
        .filter(|s| s.holds)
        .map(|s| s.offered)
        .reduce(f64::max);
    let strict = steps
        .iter()
        .take_while(|s| s.holds)
        .last()
        .map(|s| s.offered);
    let failing_steps_below_hold = match hold {
        Some(hold) => steps
            .iter()
            .filter(|s| !s.holds && s.offered < hold)
            .map(|s| s.offered)
            .collect(),
        None => Vec::new(),
    };
    Capacity {
        rule: "hold: the highest offered step at which the SLO holds; \
               strict = the last holding step before the first failing one"
            .to_string(),
        hold,
        strict,
        failing_steps_below_hold,
        // The top step held: the capacity is at least the hold, the ladder was too short.
        at_ceiling: steps.last().is_some_and(|top| hold == Some(top.offered)),
        steps,
    }
}

/// The ramp's parameters.
#[derive(Clone, Debug)]
pub struct Ramp {
    pub rates: Vec<f64>,
    pub step_seconds: u64,
    pub step_warmup: u64,
    pub slo: Slo,
}

fn run_step(
    settings: &Settings,
    bed: &Bed,
    spec: &PathSpec,
    offered: f64,
    ramp: &Ramp,
    sdir: &std::path::Path,
) -> Result<Step> {
    enqueue(
        bed,
        &answers(
            spec,
            (offered * (ramp.step_seconds + ramp.step_warmup + 2) as f64) as usize + 256,
            settings.bed.fleet.wire(),
        ),
    )?;
    thread::sleep(Duration::from_secs(1));
    let s0 = snapshot(bed)?;
    let summary = run_client(
        settings,
        bed,
        &ClientRun {
            spec,
            rate: offered,
            seconds: ramp.step_seconds,
            out: sdir,
            extra: &[],
            warmup: ramp.step_warmup,
            threads: 8,
        },
    )?;
    let s1 = snapshot(bed)?;
    let mut step = judge_step(&summary, s1.cpu_s - s0.cpu_s, offered, &ramp.slo);
    step.rss_mb_end = s1.status.get("VmRSS").copied().unwrap_or(0) as f64 / 1024.0;
    step.loadavg = s1.loadavg;
    Ok(step)
}

/// The ramp over the sides that have a binary; `live.json` and `live.md` under the run
/// directory, the table printed.
pub fn run_live(settings: &Settings, spec: &PathSpec, ramp: &Ramp) -> Result<i32> {
    let sides: Vec<&str> = SIDES
        .iter()
        .copied()
        .filter(|s| settings.binaries.contains_key(*s))
        .collect();
    let mut steps: Vec<(&str, Vec<Step>)> = sides.iter().map(|s| (*s, Vec::new())).collect();
    for offered in &ramp.rates {
        for (side, side_steps) in &mut steps {
            let side: &str = side;
            let sdir = settings
                .out
                .join("live")
                .join(side)
                .join(format!("rate-{}", py_g(*offered, 6)));
            fs::create_dir_all(&sdir)?;
            let mut bed = Bed::new(
                &settings.bed,
                settings.binary(side)?,
                &sdir.join("bed"),
                false,
            );
            println!(
                "[{}] live {side} {} at {} req/s for {} s",
                hms(),
                spec.name,
                py_g(*offered, 6),
                ramp.step_seconds
            );
            bed.up()?;
            let measured = run_step(settings, &bed, spec, *offered, ramp, &sdir);
            bed.down();
            let step = measured?;
            println!(
                "    achieved {:.0} req/s, TTFT p50/p99 {}/{} ms, e2e p99 {} ms, CPU {} ms/req, {}",
                step.achieved,
                opt(step.ttft_p50_ms),
                opt(step.ttft_p99_ms),
                opt(step.e2e_p99_ms),
                opt(step.cpu_ms_per_request.map(round3)),
                if step.holds {
                    "holds".to_string()
                } else {
                    format!("FAILS: {}", step.reasons.join("; "))
                }
            );
            side_steps.push(step);
        }
    }
    let report: Vec<(&str, Capacity)> = steps
        .into_iter()
        .map(|(side, s)| (side, capacity(s)))
        .collect();
    let sides_json: serde_json::Map<String, Value> = report
        .iter()
        .map(|(side, cap)| Ok(((*side).to_string(), serde_json::to_value(cap)?)))
        .collect::<Result<_>>()?;
    fs::write(
        settings.out.join("live.json"),
        serde_json::to_string_pretty(
            &json!({"path": spec.name, "slo": ramp.slo, "sides": sides_json}),
        )? + "\n",
    )?;
    let text = live_markdown(spec, ramp, &report);
    fs::write(settings.out.join("live.md"), &text)?;
    println!("\n{text}");
    if let Ok(summary) = std::env::var("GITHUB_STEP_SUMMARY") {
        if !summary.is_empty() {
            use std::io::Write;
            let mut file = fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(summary)?;
            file.write_all(text.as_bytes())?;
        }
    }
    Ok(0)
}

/// The ramp's table and the capacity lines.
pub fn live_markdown(spec: &PathSpec, ramp: &Ramp, report: &[(&str, Capacity)]) -> String {
    let slo = &ramp.slo;
    let mut lines = vec![
        format!("## Live: open-loop ramp on the {} path", spec.name),
        String::new(),
        format!(
            "SLO: TTFT p99 <= {} ms, e2e p99 <= {} ms, error rate <= {}, achieved >= {:.0}% of offered; \
             {} s per step after {} s warm-up, a fresh gateway per step.",
            py_float(slo.ttft_p99_ms),
            py_float(slo.e2e_p99_ms),
            py_g(slo.error_rate, 6),
            slo.achieved_fraction * 100.0,
            ramp.step_seconds,
            ramp.step_warmup
        ),
        String::new(),
        "| side | offered req/s | achieved | TTFT p50 / p99 ms | e2e p50 / p99 ms | errors \
         | gateway CPU ms/req | RSS MB | holds |"
            .to_string(),
        "|---|---|---|---|---|---|---|---|---|".to_string(),
    ];
    for (side, cap) in report {
        for s in &cap.steps {
            lines.push(format!(
                "| {side} | {} | {:.0} | {} / {} | {} / {} | {}/{} | {} | {:.0} | {} |",
                py_g(s.offered, 6),
                s.achieved,
                opt(s.ttft_p50_ms),
                opt(s.ttft_p99_ms),
                opt(s.e2e_p50_ms),
                opt(s.e2e_p99_ms),
                s.failed,
                s.sent,
                opt(s.cpu_ms_per_request.map(round3)),
                s.rss_mb_end,
                if s.holds {
                    "yes".to_string()
                } else {
                    format!("no: {}", s.reasons.join("; "))
                }
            ));
        }
    }
    lines.push(String::new());
    for (side, r) in report {
        let ceiling = if r.at_ceiling {
            " (the top step of the ladder held: the capacity is at least this; extend --rates)"
        } else {
            ""
        };
        let failing = if r.failing_steps_below_hold.is_empty() {
            String::new()
        } else {
            let items: Vec<String> = r
                .failing_steps_below_hold
                .iter()
                .map(|v| py_float(*v))
                .collect();
            format!("; failing steps below the hold: [{}]", items.join(", "))
        };
        lines.push(format!(
            "- {side}: capacity (hold) {} req/s{ceiling}; strict first-break reading {} req/s{failing}. Rule: {}.",
            opt(r.hold),
            opt(r.strict),
            r.rule
        ));
    }
    lines.join("\n") + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let spec = crate::paths::path_named("chat").unwrap();
        let cap = capacity(vec![step(200.0, 5.0, None, 0), step(400.0, 500.0, None, 0)]);
        let text = live_markdown(spec, &ramp, &[("pr", cap)]);
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
}
