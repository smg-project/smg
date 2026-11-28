//! The verdict rule of `scripts/bench_gate.py`, applied to a run's results: a gated id
//! regresses only when its change is worse than the threshold AND every head repeat is
//! worse than every base repeat; a change beyond the threshold whose repeats overlap is
//! suspect, never a failure; an id no gate pattern matches is reported, never failed; a
//! pattern that fails at match time is an error, never a skipped gate.
//! The gate file is the same TOML (`repeats`, `filter`, `bench_args`, `[[gate]]` with
//! `pattern`, `threshold`, `note`), checked the same way; the patterns are matched by
//! `fancy-regex`, which takes the look-around the Python patterns use to exclude a
//! renamed id from a group (`^(?!latency/encode_4KB$)(encode_throughput|latency)/`).

use std::{fs, path::Path};

use anyhow::{anyhow, bail, Context, Result};
use fancy_regex::Regex;

use crate::{cells::Results, median, py_float, py_g, report::per_side, round3};

#[derive(Clone, Debug)]
pub struct Gate {
    pub pattern: String,
    pub threshold: f64,
    pub note: String,
    regex: Regex,
}

impl Gate {
    pub fn new(pattern: &str, threshold: f64, note: &str) -> Result<Self> {
        let regex = Regex::new(pattern)
            .map_err(|err| anyhow!("pattern {pattern:?} does not compile: {err}"))?;
        Ok(Self {
            pattern: pattern.to_string(),
            threshold,
            note: note.to_string(),
            regex,
        })
    }

    /// Whether the pattern matches the id. A look-around or back-reference pattern can
    /// fail at match time (past the backtracking limit); that is an error naming the
    /// pattern and the id, never a miss, because a miss would skip the gate.
    pub fn matches(&self, bench_id: &str) -> Result<bool> {
        self.regex
            .is_match(bench_id)
            .map_err(|err| anyhow!("pattern {:?} on {bench_id:?}: {err}", self.pattern))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Gates {
    pub repeats: usize,
    pub filter: Option<String>,
    pub bench_args: Vec<String>,
    pub gates: Vec<Gate>,
}

impl Gates {
    /// The first gate whose pattern matches wins; None means report only.
    pub fn threshold_for(&self, bench_id: &str) -> Result<Option<&Gate>> {
        for gate in &self.gates {
            if gate.matches(bench_id)? {
                return Ok(Some(gate));
            }
        }
        Ok(None)
    }
}

const GATE_FILE_KEYS: [&str; 4] = ["repeats", "filter", "bench_args", "gate"];
const GATE_ENTRY_KEYS: [&str; 3] = ["pattern", "threshold", "note"];

/// The gate file, checked: an unknown key, a value of the wrong type, a `[[gate]]`
/// without `pattern` or `threshold`, a pattern or filter that does not compile, a
/// threshold outside (0, 1) or a repeat count under 1 is an error (exit 2, a usage
/// error), never a silent default.
pub fn load_gates(path: &Path) -> Result<Gates> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    parse_gates(&text).map_err(|err| anyhow!("{}: {err}", path.display()))
}

pub fn parse_gates(text: &str) -> Result<Gates> {
    let raw: toml::Table = toml::from_str(text).map_err(|err| anyhow!("not valid TOML: {err}"))?;
    let mut unknown: Vec<&String> = raw
        .keys()
        .filter(|k| !GATE_FILE_KEYS.contains(&k.as_str()))
        .collect();
    unknown.sort();
    if !unknown.is_empty() {
        let mut known = GATE_FILE_KEYS.to_vec();
        known.sort_unstable();
        bail!("unknown keys {unknown:?}; known: {known:?}");
    }
    let repeats = match raw.get("repeats") {
        None => 3,
        Some(toml::Value::Integer(n)) if *n >= 1 => usize::try_from(*n).unwrap_or(usize::MAX),
        Some(toml::Value::Integer(_)) => bail!("repeats must be at least 1"),
        Some(other) => bail!("repeats must be an integer, got {other}"),
    };
    let filter = match raw.get("filter") {
        None => None,
        Some(toml::Value::String(s)) => Some(s.clone()),
        Some(other) => bail!("filter must be a string, got {other}"),
    };
    let bench_args = match raw.get("bench_args") {
        None => Vec::new(),
        Some(toml::Value::Array(items)) => items
            .iter()
            .map(|v| match v {
                toml::Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .collect(),
        Some(other) => bail!("bench_args must be a list of strings, got {other}"),
    };
    let mut gates = Gates {
        repeats,
        filter,
        bench_args,
        gates: Vec::new(),
    };
    if let Some(filter) = &gates.filter {
        Regex::new(filter).map_err(|err| anyhow!("filter {filter:?} does not compile: {err}"))?;
    }
    let entries = match raw.get("gate") {
        None => Vec::new(),
        Some(toml::Value::Array(items)) => items.clone(),
        Some(other) => bail!(
            "gate must be an array of tables ([[gate]]), got a {}",
            other.type_str()
        ),
    };
    for (index, entry) in entries.iter().enumerate() {
        let index = index + 1;
        let Some(table) = entry.as_table() else {
            bail!("[[gate]] entry {index} needs pattern and threshold");
        };
        if !table.contains_key("pattern") || !table.contains_key("threshold") {
            bail!("[[gate]] entry {index} needs pattern and threshold");
        }
        let mut unknown: Vec<&String> = table
            .keys()
            .filter(|k| !GATE_ENTRY_KEYS.contains(&k.as_str()))
            .collect();
        unknown.sort();
        if !unknown.is_empty() {
            bail!("[[gate]] entry {index}: unknown keys {unknown:?}");
        }
        let pattern = match &table["pattern"] {
            toml::Value::String(s) => s.clone(),
            other => bail!("[[gate]] entry {index}: pattern must be a string, got {other}"),
        };
        let threshold = match &table["threshold"] {
            toml::Value::Float(f) => *f,
            toml::Value::Integer(n) => *n as f64,
            other => bail!("[[gate]] entry {index}: threshold must be a number, got {other}"),
        };
        if !(threshold > 0.0 && threshold < 1.0) {
            bail!("threshold {threshold} must be a fraction in (0, 1)");
        }
        let note = table.get("note").map(|v| match v {
            toml::Value::String(s) => s.clone(),
            other => other.to_string(),
        });
        gates.gates.push(Gate::new(
            &pattern,
            threshold,
            note.as_deref().unwrap_or(""),
        )?);
    }
    Ok(gates)
}

/// One id's two samples and its verdict.
#[derive(Clone, Debug)]
pub struct Row {
    pub bench_id: String,
    pub base: Vec<f64>,
    pub pr: Vec<f64>,
    pub gate: Option<Gate>,
    pub verdict: String,
}

impl Row {
    pub fn base_median(&self) -> Option<f64> {
        median(&self.base)
    }

    pub fn pr_median(&self) -> Option<f64> {
        median(&self.pr)
    }

    /// The relative change of the medians against the base median's magnitude (an
    /// idle-subtracted count can sit below zero: a rise stays a rise); None without both
    /// sides, or when the base median is zero (a count that was zero before: the absolute
    /// values tell).
    pub fn change(&self) -> Option<f64> {
        let (bm, pm) = (self.base_median()?, self.pr_median()?);
        if bm == 0.0 {
            return None;
        }
        Some((pm - bm) / bm.abs())
    }

    /// The larger of the two sides' (max - min) / |median|: the noise of this job.
    pub fn spread(&self) -> Option<f64> {
        let mut spreads = Vec::new();
        for values in [&self.base, &self.pr] {
            let Some(med) = median(values) else {
                continue;
            };
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            if med != 0.0 {
                spreads.push((max - min) / med.abs());
            } else if max != min {
                return None;
            } else {
                spreads.push(0.0);
            }
        }
        spreads.into_iter().reduce(f64::max)
    }

    /// Every pull-request run worse than every base run (or the reverse).
    pub fn separated(&self) -> bool {
        if self.base.is_empty() || self.pr.is_empty() {
            return false;
        }
        let (bmin, bmax) = extremes(&self.base);
        let (pmin, pmax) = extremes(&self.pr);
        pmin > bmax || pmax < bmin
    }

    pub fn fails(&self) -> bool {
        self.verdict == "regression" || self.verdict == "missing"
    }
}

fn extremes(values: &[f64]) -> (f64, f64) {
    (
        values.iter().copied().fold(f64::INFINITY, f64::min),
        values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}

pub fn judge(row: &Row) -> String {
    if row.pr.is_empty() {
        return if row.gate.is_some() {
            "missing"
        } else {
            "missing (report only)"
        }
        .to_string();
    }
    if row.base.is_empty() {
        return "new".to_string();
    }
    let Some(gate) = &row.gate else {
        return "report only".to_string();
    };
    let Some(change) = row.change() else {
        // A zero base median: the absolute head values tell. Above zero, every head
        // run above every base run is a regression and an overlap is suspect, as for a
        // change beyond the threshold; below zero and separated, an improvement.
        let pm = row.pr_median().unwrap_or(0.0);
        if pm > 0.0 {
            return if row.separated() {
                "regression"
            } else {
                "suspect (runs overlap)"
            }
            .to_string();
        }
        if pm < 0.0 && row.separated() {
            return "improvement".to_string();
        }
        return "ok".to_string();
    };
    if change > gate.threshold {
        return if row.separated() {
            "regression"
        } else {
            "suspect (runs overlap)"
        }
        .to_string();
    }
    if change < -gate.threshold && row.separated() {
        return "improvement".to_string();
    }
    "ok".to_string()
}

pub fn compare(results: &Results, gates: &Gates) -> Result<Vec<Row>> {
    let sides = per_side(results);
    let empty = Default::default();
    let base = sides.get("base").unwrap_or(&empty);
    let pr = sides.get("pr").unwrap_or(&empty);
    let mut ids: Vec<&String> = base.keys().chain(pr.keys()).collect();
    ids.sort();
    ids.dedup();
    ids.into_iter()
        .map(|bench_id| {
            let mut row = Row {
                bench_id: bench_id.clone(),
                base: base.get(bench_id).cloned().unwrap_or_default(),
                pr: pr.get(bench_id).cloned().unwrap_or_default(),
                gate: gates.threshold_for(bench_id)?.cloned(),
                verdict: String::new(),
            };
            row.verdict = judge(&row);
            Ok(row)
        })
        .collect()
}

fn fmt_time(ns: f64, unit_ns: f64) -> String {
    let (unit, scale) = [("s", 1e9), ("ms", 1e6), ("µs", 1e3)]
        .into_iter()
        .find(|(_, candidate)| unit_ns >= *candidate)
        .unwrap_or(("ns", 1.0));
    format!("{} {unit}", py_g(ns / scale, 4))
}

/// A criterion median is nanoseconds and is scaled to a time unit; a results file
/// that names its own `unit` (an empty string for a bare count: the ids then carry the
/// unit in their names) prints its values as they are, with that unit.
fn fmt_value(value: f64, unit_ns: f64, unit: Option<&str>) -> String {
    match unit {
        None => fmt_time(value, unit_ns),
        Some(unit) => format!("{} {unit}", py_g(value, 4)).trim_end().to_string(),
    }
}

fn fmt_side(values: &[f64], unit_ns: f64, unit: Option<&str>) -> String {
    let Some(med) = median(values) else {
        return "-".to_string();
    };
    if values.len() == 1 {
        return fmt_value(med, unit_ns, unit);
    }
    let (min, max) = extremes(values);
    format!(
        "{} ({} - {})",
        fmt_value(med, unit_ns, unit),
        fmt_value(min, unit_ns, unit),
        fmt_value(max, unit_ns, unit)
    )
}

fn fmt_pct(value: Option<f64>, signed: bool) -> String {
    match value {
        None => "-".to_string(),
        Some(v) if signed => format!("{:+.1}%", v * 100.0),
        Some(v) => format!("{:.1}%", v * 100.0),
    }
}

fn py_list(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|v| py_float(round3(*v))).collect();
    format!("[{}]", items.join(", "))
}

/// The verdict table, as `bench_gate.py` prints it.
pub fn markdown(results: &Results, rows: &[Row]) -> String {
    let repeats = results.repeats;
    let unit = results.unit.as_deref();
    let failed: Vec<&Row> = rows.iter().filter(|r| r.fails()).collect();
    let suspect = rows
        .iter()
        .filter(|r| r.verdict.starts_with("suspect"))
        .count();
    let verdict = if failed.is_empty() { "PASS" } else { "FAIL" };
    let mut lines = vec![
        format!("### Benchmark gate: {verdict}"),
        String::new(),
        format!(
            "Medians of {repeats} interleaved runs per side on this runner, (min - max) per side; \
             spread = the larger side's (max - min) / median. A gated id fails when its change is \
             worse than the threshold and every pull-request run is worse (higher) than every base run."
        ),
        String::new(),
        "| benchmark | base | pull request | change | spread | threshold | verdict |".to_string(),
        "|---|---|---|---|---|---|---|".to_string(),
    ];
    for row in rows {
        let unit_ns = row
            .base_median()
            .filter(|m| *m != 0.0)
            .or_else(|| row.pr_median().filter(|m| *m != 0.0))
            .unwrap_or(1.0);
        let threshold = row
            .gate
            .as_ref()
            .map_or_else(|| "-".to_string(), |g| fmt_pct(Some(g.threshold), false));
        let mark = match row.verdict.as_str() {
            "regression" | "missing" => "FAIL ",
            "improvement" => "faster ",
            _ => "",
        };
        lines.push(format!(
            "| {} | {} | {} | {} | {} | {threshold} | {mark}{} |",
            row.bench_id,
            fmt_side(&row.base, unit_ns, unit),
            fmt_side(&row.pr, unit_ns, unit),
            fmt_pct(row.change(), true),
            fmt_pct(row.spread(), false),
            row.verdict
        ));
    }
    lines.push(String::new());
    let gated = rows.iter().filter(|r| r.gate.is_some()).count();
    lines.push(format!(
        "{} ids, {gated} gated, {} failed, {suspect} suspect (beyond the threshold, runs overlap: not failed).",
        rows.len(),
        failed.len()
    ));
    let unit_suffix = match unit {
        None => " ns".to_string(),
        Some("") => String::new(),
        Some(u) => format!(" {u}"),
    };
    for row in failed {
        if row.verdict == "missing" {
            lines.push(format!(
                "- `{}` is gated but the pull request's run has no such id: restore the benchmark, \
                 or exclude the old id from the gate pattern that matches it (a negative lookahead \
                 in the group pattern).",
                row.bench_id
            ));
        } else {
            let threshold = row.gate.as_ref().map_or(0.0, |g| g.threshold);
            lines.push(format!(
                "- `{}`: {} against a {} threshold; base runs {}{unit_suffix}, pull request runs {}{unit_suffix}.",
                row.bench_id,
                fmt_pct(row.change(), true),
                fmt_pct(Some(threshold), false),
                py_list(&row.base),
                py_list(&row.pr)
            ));
        }
    }
    lines.join("\n") + "\n"
}

/// The verdict table over a run's results (report only without a gate file); returns
/// the markdown and the exit code the gate would give.
pub fn gate_report(results: &Results, gates_path: Option<&Path>) -> Result<(String, i32)> {
    let gates = match gates_path {
        Some(path) => load_gates(path)?,
        None => Gates {
            repeats: 3,
            ..Gates::default()
        },
    };
    let rows = compare(results, &gates)?;
    let text = markdown(results, &rows);
    let rc = i32::from(rows.iter().any(Row::fails));
    Ok((text, rc))
}

#[cfg(test)]
#[path = "../tests/unit/gate/tests.rs"]
mod tests;
