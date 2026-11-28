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
fn the_resident_set_prints_its_anonymous_part_then_its_file_part_then_the_whole() {
    let res = results_of(
        &[
            ("chat/rss_mb_after_window", &[100.0, 101.0, 100.0]),
            ("chat/rss_file_mb_after_window", &[60.0, 61.0, 60.0]),
            ("chat/rss_anon_mb_after_window", &[40.0, 40.0, 40.0]),
        ],
        &[
            ("chat/rss_mb_after_window", &[103.0, 103.0, 104.0]),
            ("chat/rss_file_mb_after_window", &[63.0, 63.0, 64.0]),
            ("chat/rss_anon_mb_after_window", &[40.0, 40.0, 40.0]),
        ],
    );
    let table = numbers_table(&res, "x");
    let lines: Vec<&str> = table.lines().collect();
    for (line, start) in [
        (
            lines[2],
            "| chat: anonymous RSS after the window | 40.0 MB | 40.0 MB (+0.0 %) |",
        ),
        (
            lines[3],
            "| chat: file-backed RSS after the window | 60.0 MB | 63.0 MB (+5.0 %) |",
        ),
        (
            lines[4],
            "| chat: RSS after the window | 100.0 MB | 103.0 MB (+3.0 %) |",
        ),
    ] {
        assert!(line.starts_with(start), "{line}");
    }
}

#[test]
fn numbers_table_skips_unknown_metrics_and_one_sided_ids() {
    let res = results_of(
        &[("chat/odd_metric", &[1.0]), ("chat/hwm_mb", &[110.0])],
        &[("chat/odd_metric", &[1.0])],
    );
    assert_eq!(numbers_table(&res, "x").lines().count(), 2);
}
