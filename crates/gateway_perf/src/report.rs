//! The `## Numbers` table of a pull request from a run's results.

use std::collections::BTreeMap;

use crate::{cells::Results, median, SIDES};

/// (max - min) / |median| of a sample; None for an empty sample or a zero median.
pub fn spread(values: &[f64]) -> Option<f64> {
    let med = median(values)?;
    if med == 0.0 {
        return None;
    }
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    Some((max - min) / med.abs())
}

/// How a metric's value prints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// `0.362 M`
    Millions,
    /// `1,062`
    Count,
    /// `122.7`
    OneDecimal,
    /// `531 us`
    Micros,
    /// `61.0 MB`
    Megabytes,
    /// `+0.76 MB`
    SignedMegabytes,
}

impl Format {
    fn render(self, value: f64) -> String {
        match self {
            Self::Millions => format!("{value:.3} M"),
            Self::Count => thousands(value, 0, false),
            Self::OneDecimal => format!("{value:.1}"),
            Self::Micros => format!("{value:.0} us"),
            Self::Megabytes => format!("{value:.1} MB"),
            Self::SignedMegabytes => format!("{value:+.2} MB"),
        }
    }
}

/// A metric row: its key in the results, its title, its format, and whether the change
/// prints as a percentage (else as an absolute difference).
pub struct MetricRow {
    pub key: &'static str,
    pub title: &'static str,
    pub format: Format,
    pub relative: bool,
}

pub const METRIC_ROWS: [MetricRow; 12] = [
    MetricRow {
        key: "user_instructions_per_request_M",
        title: "user instructions per request",
        format: Format::Millions,
        relative: true,
    },
    MetricRow {
        key: "allocation_calls_per_request",
        title: "allocation calls per request",
        format: Format::Count,
        relative: false,
    },
    MetricRow {
        key: "kib_requested_per_request",
        title: "KiB requested per request",
        format: Format::OneDecimal,
        relative: false,
    },
    MetricRow {
        key: "cpu_us_per_request",
        title: "gateway CPU per request",
        format: Format::Micros,
        relative: true,
    },
    MetricRow {
        key: "rss_mb_after_window",
        title: "RSS after the window",
        format: Format::Megabytes,
        relative: true,
    },
    MetricRow {
        key: "hwm_mb",
        title: "peak RSS (VmHWM)",
        format: Format::Megabytes,
        relative: true,
    },
    MetricRow {
        key: "instructions_per_request_M",
        title: "instructions per request (user + kernel)",
        format: Format::Millions,
        relative: true,
    },
    MetricRow {
        key: "rss_mb_after_first_window",
        title: "RSS after the first window",
        format: Format::Megabytes,
        relative: true,
    },
    MetricRow {
        key: "rss_mb_after_last_window",
        title: "RSS after the last window",
        format: Format::Megabytes,
        relative: true,
    },
    MetricRow {
        key: "rss_growth_mb_per_10k_requests",
        title: "RSS growth per 10,000 requests",
        format: Format::SignedMegabytes,
        relative: false,
    },
    MetricRow {
        key: "peak_rss_mb_after_burst",
        title: "peak RSS after the burst window",
        format: Format::Megabytes,
        relative: true,
    },
    MetricRow {
        key: "allocator_live_mb_after_last_window",
        title: "allocator live bytes after the last window",
        format: Format::Megabytes,
        relative: true,
    },
];

/// A number with `decimals` decimals and thousands separators in its integer part, the
/// sign always shown when `signed` (Python's `{:,.Nf}` / `{:+,.Nf}`).
pub fn thousands(value: f64, decimals: usize, signed: bool) -> String {
    let text = if signed {
        format!("{value:+.decimals$}")
    } else {
        format!("{value:.decimals$}")
    };
    let (sign, digits) = match text.strip_prefix(['+', '-']) {
        Some(rest) => (&text[..1], rest),
        None => ("", text.as_str()),
    };
    let (int_part, frac_part) = match digits.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (digits, None),
    };
    let mut grouped = String::new();
    for (n, ch) in int_part.chars().enumerate() {
        if n > 0 && (int_part.len() - n) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(ch);
    }
    match frac_part {
        Some(f) => format!("{sign}{grouped}.{f}"),
        None => format!("{sign}{grouped}"),
    }
}

/// The values of every id per side, in run order.
pub fn per_side(results: &Results) -> BTreeMap<&'static str, BTreeMap<String, Vec<f64>>> {
    let mut out: BTreeMap<&'static str, BTreeMap<String, Vec<f64>>> = BTreeMap::new();
    for side in SIDES {
        let table = out.entry(side).or_default();
        for run in &results.side(side).runs {
            for (bench_id, value) in run {
                table.entry(bench_id.clone()).or_default().push(*value);
            }
        }
    }
    out
}

/// The `## Numbers` table of a pull request: metric, before, after, setup; medians
/// over the repeats, the change in the after column, the base's spread in the setup
/// column (the floor under which a change is not a change).
pub fn numbers_table(results: &Results, setup: &str) -> String {
    let sides = per_side(results);
    let empty = BTreeMap::new();
    let base = sides.get("base").unwrap_or(&empty);
    let pr = sides.get("pr").unwrap_or(&empty);
    let mut ids: Vec<&String> = base.keys().chain(pr.keys()).collect();
    ids.sort();
    ids.dedup();
    let mut lines = vec![
        "| metric | before | after | setup |".to_string(),
        "|---|---|---|---|".to_string(),
    ];
    let mut rows: Vec<(String, usize, String, usize)> = Vec::new();
    for bench_id in ids {
        let (prefix, metric) = bench_id.rsplit_once('/').unwrap_or(("", bench_id));
        if let Some(n) = METRIC_ROWS.iter().position(|r| r.key == metric) {
            rows.push((prefix.to_string(), n, bench_id.clone(), n));
        }
    }
    rows.sort();
    for (prefix, _, bench_id, n) in rows {
        let row = &METRIC_ROWS[n];
        let (Some(b), Some(h)) = (base.get(&bench_id), pr.get(&bench_id)) else {
            continue;
        };
        let (Some(bm), Some(hm)) = (median(b), median(h)) else {
            continue;
        };
        let change = if row.relative {
            if bm == 0.0 {
                "-".to_string()
            } else {
                format!("{:+.1} %", (hm - bm) / bm.abs() * 100.0)
            }
        } else if row.key == "allocation_calls_per_request" {
            thousands(hm - bm, 0, true)
        } else {
            thousands(hm - bm, 1, true)
        };
        let floor_text = spread(b).map_or(String::new(), |f| {
            format!("; base spread {:.2} %", f * 100.0)
        });
        lines.push(format!(
            "| {prefix}: {} | {} | {} ({change}) | {setup}; {}+{} interleaved cells{floor_text} |",
            row.title,
            row.format.render(bm),
            row.format.render(hm),
            b.len(),
            h.len()
        ));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::tests::results_of;

    #[test]
    fn spread_is_max_minus_min_over_median() {
        assert!((spread(&[100.0, 102.0, 98.0]).unwrap() - 0.04).abs() < 1e-12);
        assert_eq!(spread(&[]), None);
        assert_eq!(spread(&[0.0, 0.0]), None);
        // An idle-subtracted value can sit below zero: the spread reads against the magnitude.
        assert!((spread(&[-2.2, -2.0, -1.8]).unwrap() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn thousands_match_python() {
        assert_eq!(thousands(1062.0, 0, false), "1,062");
        assert_eq!(thousands(313.0, 0, false), "313");
        assert_eq!(thousands(2.0, 0, true), "+2");
        assert_eq!(thousands(-0.4, 0, true), "-0");
        assert_eq!(thousands(0.0, 0, true), "+0");
        assert_eq!(thousands(1234567.891, 1, true), "+1,234,567.9");
        assert_eq!(thousands(-10.0, 0, true), "-10");
    }

    #[test]
    fn numbers_table_has_the_pull_request_shape_with_medians_change_and_spread() {
        let res = results_of(
            &[
                (
                    "chat/user_instructions_per_request_M",
                    &[0.500, 0.502, 0.498],
                ),
                ("chat/allocation_calls_per_request", &[300.0, 301.0, 300.0]),
                ("chat/rss_mb_after_window", &[100.0, 101.0, 100.0]),
            ],
            &[
                (
                    "chat/user_instructions_per_request_M",
                    &[0.480, 0.481, 0.479],
                ),
                ("chat/allocation_calls_per_request", &[290.0, 290.0, 291.0]),
                ("chat/rss_mb_after_window", &[100.5, 100.0, 100.0]),
            ],
        );
        let table = numbers_table(&res, "10,000 requests per cell");
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(lines[0], "| metric | before | after | setup |");
        assert_eq!(lines[1], "|---|---|---|---|");
        assert_eq!(
            lines[2],
            "| chat: user instructions per request | 0.500 M | 0.480 M (-4.0 %) | 10,000 requests per cell; 3+3 interleaved cells; base spread 0.80 % |"
        );
        assert!(
            lines[3].starts_with("| chat: allocation calls per request | 300 | 290 (-10) |"),
            "{}",
            lines[3]
        );
        assert!(
            lines[4].starts_with("| chat: RSS after the window | 100.0 MB | 100.0 MB (+0.0 %) |"),
            "{}",
            lines[4]
        );
    }

    #[test]
    fn numbers_table_skips_unknown_metrics_and_one_sided_ids() {
        let res = results_of(
            &[("chat/odd_metric", &[1.0]), ("chat/hwm_mb", &[110.0])],
            &[("chat/odd_metric", &[1.0])],
        );
        assert_eq!(numbers_table(&res, "x").lines().count(), 2);
    }
}
