//! The gateway's cost per request, base against head, on the CPU mock bed.
//!
//! One release gateway in front of mock gRPC engines that answer from a script (the
//! answer is served by the wire, so no engine time enters the numbers), an open-loop
//! client at a fixed rate for a fixed number of requests, and the gateway process's own
//! counters over the window, read from outside the process:
//!
//! - `cpu`: instructions per request (user, and user+kernel) and cycles from `perf stat`
//!   on the gateway pid when hardware counters are available; CPU time per request
//!   (utime+stime from /proc) always.
//! - `alloc`: allocation calls and KiB requested per request, exact, from jemalloc's exit
//!   statistics (`_RJEM_MALLOC_CONF=stats_print:true`; the gateway prints them at a clean
//!   shutdown; no special build).
//! - `mem`: RSS after each of several windows on one gateway, the growth per 10,000
//!   requests, the peak (VmHWM) over a burst window, the allocator's live bytes.
//! - `live`: an open-loop ramp of offered rates, a fresh gateway per step, with the
//!   capacity rule: the highest offered step at which the SLO holds, and the strict
//!   reading (the last holding step before the first failing one) beside it.
//!
//! Every cell runs on a FRESH gateway. An idle cell per binary (no requests) gives the
//! per-second idle rates, which the per-request numbers subtract. Base and head cells are
//! interleaved (base, head, head, base, ...) so a drift of the host lands on both sides,
//! and the numbers of each (path, metric) are the medians over the repeats with the
//! spread (max - min) / median beside them. `results.json` has the shape
//! `scripts/bench_gate.py compare` reads, and the [`gate`] module applies the same
//! noise-aware rule: a gated id regresses only when its change is worse than the threshold
//! AND every head repeat is worse than every base repeat. Exit codes: 0; 1 when a gated id
//! regressed; 2 when the run failed or a cell carries a warning (failed requests, script
//! misses, missing allocator statistics): a job reads 2 as "no measurement".
//!
//! The crate drives what the repository already has and re-implements none of it: the bed
//! is `scripts/bench/grpc_stress/bed.sh`, the engines are `mock-worker` with their admin
//! API, the client is `stress_load` (both binaries of `crates/mock_worker`).

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "the driver reports its progress and its tables on the standard streams"
)]

pub mod base;
pub mod bed;
pub mod cells;
pub mod counters;
pub mod gate;
pub mod http;
pub mod live;
pub mod paths;
pub mod report;
pub mod tokenizer;

/// The two sides of an A/B, in the order the results file and the gate name them.
pub const SIDES: [&str; 2] = ["base", "pr"];

/// The median of a sample (the mean of the two middle values for an even count).
pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    Some(if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    })
}

/// `HH:MM:SS` (UTC) for the progress lines.
pub fn hms() -> String {
    chrono::Utc::now().format("%H:%M:%S").to_string()
}

/// `YYYYMMDDTHHMMSSZ`: the stamp of a run directory.
pub fn stamp() -> String {
    chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string()
}

/// Seconds since the epoch, as a float (the cell's clock).
pub fn now_seconds() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// The machine's 1-minute load average (0 when /proc/loadavg cannot be read).
pub fn loadavg() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|text| text.split_whitespace().next().and_then(|v| v.parse().ok()))
        .unwrap_or(0.0)
}

/// A float the way Python's `repr` prints it: integral values keep a `.0`, the rest is
/// the shortest round-trip text. The reports print raw values with it.
pub fn py_float(value: f64) -> String {
    if value.is_finite() && value.fract() == 0.0 && value.abs() < 1e16 {
        format!("{value:.1}")
    } else {
        format!("{value}")
    }
}

/// Python's `round(value, 3)`.
pub fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_is_the_middle_or_the_mean_of_the_two_middles() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&[]), None);
    }

    #[test]
    fn python_float_text() {
        assert_eq!(py_float(300.0), "300.0");
        assert_eq!(py_float(0.589), "0.589");
        assert_eq!(py_float(-2.0), "-2.0");
        assert_eq!(round3(0.12345), 0.123);
    }
}

/// A float the way Python's `format(value, f"{precision}g")` prints it: `precision`
/// significant digits, trailing zeros dropped, scientific notation below 1e-4 or at
/// 10^precision and above (`1.5e+04`).
pub fn py_g(value: f64, precision: usize) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    if !value.is_finite() {
        return format!("{value}");
    }
    let p = precision.max(1);
    let sci = format!("{:.*e}", p - 1, value);
    let Some((mantissa, exp)) = sci.split_once('e') else {
        return sci;
    };
    let exp: i32 = exp.parse().unwrap_or(0);
    if exp < -4 || exp >= p as i32 {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", strip_zeros(mantissa), exp.abs())
    } else {
        let decimals = usize::try_from(p as i32 - 1 - exp).unwrap_or(0);
        strip_zeros(&format!("{value:.decimals$}"))
    }
}

fn strip_zeros(text: &str) -> String {
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text.to_string()
    }
}

#[cfg(test)]
mod py_g_tests {
    use super::py_g;

    #[test]
    fn general_format_matches_python() {
        assert_eq!(py_g(300.5, 4), "300.5");
        assert_eq!(py_g(300.0, 4), "300");
        assert_eq!(py_g(1.55, 4), "1.55");
        assert_eq!(py_g(500.0, 6), "500");
        assert_eq!(py_g(1000.5, 6), "1000.5");
        assert_eq!(py_g(0.0001234, 4), "0.0001234");
        assert_eq!(py_g(0.00001234, 4), "1.234e-05");
        assert_eq!(py_g(123456.0, 4), "1.235e+05");
        assert_eq!(py_g(0.0, 4), "0");
        assert_eq!(py_g(-2.5, 4), "-2.5");
    }
}
