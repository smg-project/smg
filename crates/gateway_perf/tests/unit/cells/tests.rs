use super::*;
use crate::{
    bed::Fleet,
    paths::{path_named, DEFAULT_GW_ARGS},
};

/// A results file from per-id samples: run i holds the i-th value of every id that
/// has one.
pub(crate) fn results_of(base: &[(&str, &[f64])], pr: &[(&str, &[f64])]) -> Results {
    fn runs(values: &[(&str, &[f64])]) -> Vec<BTreeMap<String, f64>> {
        let n = values.iter().map(|(_, v)| v.len()).max().unwrap_or(0);
        (0..n)
            .map(|i| {
                values
                    .iter()
                    .filter_map(|(k, v)| v.get(i).map(|x| ((*k).to_string(), *x)))
                    .collect()
            })
            .collect()
    }
    Results {
        base: SideResults {
            runs: runs(base),
            ..SideResults::default()
        },
        pr: SideResults {
            runs: runs(pr),
            ..SideResults::default()
        },
        repeats: 3,
        unit: Some(String::new()),
        ..Results::default()
    }
}

fn idle_cell(side: &str) -> CellRecord {
    CellRecord {
        side: side.to_string(),
        path: "idle".to_string(),
        label: "idle".to_string(),
        window_s: 10.0,
        cpu_s: 0.1,
        perf: BTreeMap::from([
            ("instructions:u".to_string(), 1e8),
            ("instructions".to_string(), 2e8),
            ("cycles".to_string(), 3e8),
        ]),
        jemalloc: json!({"nrequests": 1_000, "bytes_requested": 100_000, "large_nrequests": 1}),
        ..CellRecord::default()
    }
}

fn loaded_cell(side: &str, label: &str) -> CellRecord {
    CellRecord {
        side: side.to_string(),
        path: "chat".to_string(),
        label: label.to_string(),
        sent: 1_000,
        window_s: 20.0,
        cpu_s: 0.7,
        status_1: BTreeMap::from([
            ("VmRSS".to_string(), 102_400),
            ("VmHWM".to_string(), 112_640),
        ]),
        perf: BTreeMap::from([
            ("instructions:u".to_string(), 7e8),
            ("instructions".to_string(), 13e8),
            ("cycles".to_string(), 16e8),
        ]),
        jemalloc: json!({
            "nrequests": 301_000,
            "bytes_requested": 100_000 + 1_000 * 1024 * 120,
            "large_nrequests": 1,
        }),
        ..CellRecord::default()
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9 * b.abs().max(1.0)
}

#[test]
fn window_is_the_ceiling_of_requests_over_the_rate() {
    assert_eq!(window_seconds(10_000, 500.0), 20);
    assert_eq!(window_seconds(10_000, 250.0), 40);
    assert_eq!(window_seconds(10_001, 500.0), 21);
    assert_eq!(window_seconds(10, 0.5), 10);
    assert_eq!(window_seconds(0, 500.0), 1);
}

#[test]
fn per_request_subtracts_the_idle_rate_over_the_window() {
    let r = per_request(&loaded_cell("base", "r1"), Some(&idle_cell("base")));
    // 7e8 user instructions minus 20 s of the idle rate (1e8 / 10 s) = 5e8 over 1,000 requests
    assert!(close(r["user_instructions_per_request_M"], 0.5));
    assert!(close(r["instructions_per_request_M"], 0.9));
    assert!(close(r["cycles_per_request_M"], 1.0));
    assert!(close(r["cpu_us_per_request"], (0.7 - 0.2) / 1_000.0 * 1e6));
    assert!(close(r["allocation_calls_per_request"], 300.0));
    assert!(close(r["kib_requested_per_request"], 120.0));
    assert_eq!(r["large_allocations_per_request"], 0.0);
    assert_eq!(r["rss_mb_after_window"], 100.0);
    assert_eq!(r["hwm_mb"], 110.0);
}

#[test]
fn per_request_without_counters_or_idle_still_gives_cpu_time_and_rss() {
    let mut cell = loaded_cell("base", "r1");
    cell.perf.clear();
    cell.jemalloc = json!({"error": "none"});
    let r = per_request(&cell, None);
    let keys: Vec<&String> = r.keys().collect();
    assert_eq!(
        keys,
        ["cpu_us_per_request", "hwm_mb", "rss_mb_after_window"]
    );
    assert!(close(r["cpu_us_per_request"], 700.0));
    cell.sent = 0;
    assert!(per_request(&cell, None).is_empty());
}

#[test]
fn the_resident_set_is_reported_whole_and_in_its_anonymous_and_file_parts() {
    let mut cell = loaded_cell("base", "r1");
    cell.status_1.insert("RssAnon".to_string(), 40_960);
    cell.status_1.insert("RssFile".to_string(), 61_440);
    let r = per_request(&cell, None);
    assert_eq!(r["rss_mb_after_window"], 100.0);
    assert_eq!(r["rss_anon_mb_after_window"], 40.0);
    assert_eq!(r["rss_file_mb_after_window"], 60.0);
    // One part alone is reported alone: a missing part is no zero.
    cell.status_1.remove("RssFile");
    let r = per_request(&cell, None);
    assert_eq!(r["rss_anon_mb_after_window"], 40.0);
    assert!(!r.contains_key("rss_file_mb_after_window"));
}

#[test]
fn results_file_has_one_run_per_repeat_in_bench_gate_shape() {
    let mut cells = vec![idle_cell("base"), idle_cell("pr")];
    for (i, side) in interleaved_order(3) {
        let mut c = loaded_cell(side, &format!("r{i}"));
        let allocs = if side == "base" { 301_000 } else { 291_000 };
        c.jemalloc = json!({"nrequests": allocs, "bytes_requested": 100_000, "large_nrequests": 1});
        cells.push(c);
    }
    let binaries = BTreeMap::from([
        ("base".to_string(), PathBuf::from("/b")),
        ("pr".to_string(), PathBuf::from("/h")),
    ]);
    let results = results_from_cells(&cells, 3, &binaries, "n");
    assert_eq!(results.unit.as_deref(), Some(""));
    assert_eq!(results.repeats, 3);
    assert_eq!(results.note, "n");
    assert_eq!(
        results.order,
        ["base-1", "pr-1", "pr-2", "base-2", "base-3", "pr-3"]
    );
    assert_eq!(results.base.runs.len(), 3);
    assert_eq!(results.pr.runs.len(), 3);
    assert!(close(
        results.base.runs[0]["chat/allocation_calls_per_request"],
        300.0
    ));
    assert!(close(
        results.pr.runs[0]["chat/allocation_calls_per_request"],
        290.0
    ));
    assert_eq!(results.base.binary, "/b");
    let text = serde_json::to_string(&results).unwrap();
    assert!(text.contains(r#""unit":"""#), "{text}");
    assert!(text.contains(r#""filter":null"#), "{text}");
}

#[test]
fn outputs_are_written_and_a_warned_cell_fails_the_run() {
    let dir = tempfile::tempdir().unwrap();
    std::env::remove_var("GITHUB_STEP_SUMMARY");
    let cells = vec![
        idle_cell("base"),
        idle_cell("pr"),
        loaded_cell("base", "r1"),
        loaded_cell("pr", "r1"),
    ];
    let results = results_from_cells(&cells, 1, &BTreeMap::new(), "");
    let mut values: Vec<Value> = cells
        .iter()
        .map(|c| serde_json::to_value(c).unwrap())
        .collect();
    assert_eq!(
        write_outputs(dir.path(), &values, &results, None, "setup").unwrap(),
        0
    );
    assert!(dir.path().join("numbers.md").exists());
    assert!(dir.path().join("results.json").exists());
    assert!(dir.path().join("gate.md").exists());
    assert!(dir.path().join("cells.json").exists());
    values[3]["warnings"] = json!(["3 failed requests"]);
    assert_eq!(
        write_outputs(dir.path(), &values, &results, None, "setup").unwrap(),
        2
    );
    let numbers = fs::read_to_string(dir.path().join("numbers.md")).unwrap();
    assert!(
        numbers.contains("chat: allocation calls per request"),
        "{numbers}"
    );
    assert!(numbers.contains("1-minute load average"), "{numbers}");
    let reread: Results =
        serde_json::from_str(&fs::read_to_string(dir.path().join("results.json")).unwrap())
            .unwrap();
    assert_eq!(reread.base.runs.len(), 1);
}

#[test]
fn a_cell_whose_requests_the_drain_cut_off_warns_like_one_with_failed_requests() {
    let measured = |summary: Value| Measured {
        summary,
        s0: Snapshot::default(),
        s1: Snapshot::default(),
        counters: BTreeMap::new(),
        script: None,
    };
    let drain_warning = "2 requests cut off at the client's drain".to_string();
    let mut cut = CellRecord::default();
    fill_record(
        &mut cut,
        true,
        &measured(json!({"sent": 10, "ok": 8, "failed": 0, "unfinished": 2})),
        "",
    );
    assert_eq!(
        (cut.sent, cut.ok, cut.failed, cut.unfinished),
        (10, 8, 0, 2)
    );
    assert!(cut.warnings.contains(&drain_warning), "{:?}", cut.warnings);
    let mut whole = CellRecord::default();
    fill_record(
        &mut whole,
        true,
        &measured(json!({"sent": 10, "ok": 10, "failed": 0, "unfinished": 0})),
        "",
    );
    assert_eq!(whole.unfinished, 0);
    assert!(
        !whole.warnings.contains(&drain_warning),
        "{:?}",
        whole.warnings
    );
    let mut idle = CellRecord::default();
    fill_record(&mut idle, false, &measured(json!({"unfinished": 2})), "");
    assert!(
        !idle.warnings.contains(&drain_warning),
        "an idle cell sends nothing: {:?}",
        idle.warnings
    );
}

#[test]
fn mem_soak_record_warns_on_failed_requests_and_script_misses() {
    let snap = |rss: u64| Snapshot {
        status: BTreeMap::from([("VmRSS".to_string(), rss), ("VmHWM".to_string(), 2048)]),
        ..Snapshot::default()
    };
    let rows = vec![
        SoakRow {
            after: "start".to_string(),
            snap: snap(1024),
        },
        SoakRow {
            after: "window 1".to_string(),
            snap: snap(1024),
        },
        SoakRow {
            after: "burst".to_string(),
            snap: snap(1024),
        },
        SoakRow {
            after: "10 s idle".to_string(),
            snap: snap(1024),
        },
    ];
    let counts = ScriptCounts {
        hits: 18,
        misses: 2,
        unreadable: 1,
        queued_left: None,
    };
    let summaries = [
        json!({"sent": 10, "ok": 9, "failed": 1}),
        json!({"sent": 10, "ok": 7, "failed": 2, "unfinished": 1}),
    ];
    let spec = path_named("stream").unwrap();
    let measured = SoakMeasured {
        rows,
        summaries: summaries.to_vec(),
        script: counts.clone(),
    };
    let record = soak_record("pr", spec, "r1", Path::new("/d"), measured, 1, 10).unwrap();
    assert_eq!(record.sent, 20);
    assert_eq!(record.failed, 3);
    assert_eq!(record.unfinished, 1);
    assert_eq!(record.script, counts);
    assert_eq!(
        record.warnings,
        [
            "3 failed requests",
            "1 requests cut off at the client's drain",
            "2 script misses (answers served by the engine simulator)",
            "1 script counters unreadable (misses unknown)",
        ]
    );
    assert_eq!(record.values["mem-stream/rss_mb_after_first_window"], 1.0);
    assert_eq!(record.values["mem-stream/peak_rss_mb_after_burst"], 2.0);
    assert_eq!(
        record.values["mem-stream/rss_growth_mb_per_10k_requests"],
        0.0
    );
    let value = serde_json::to_value(&record).unwrap();
    assert_eq!(value["rows"][0]["after"], "start");
    assert_eq!(value["rows"][0]["status"]["VmRSS"], 1024);
    assert_eq!(value["unfinished"], 1);
}

#[test]
fn setup_text_names_the_cell_parameters() {
    let cfg = BedConfig::new(Path::new("/bed.sh"), Path::new("/m"), Path::new("/l"), "/t");
    let mut settings = Settings::new(Path::new("/out"), cfg, BTreeMap::new(), Vec::new());
    assert_eq!(
        setup_text(&settings, "cpu"),
        "cpu: 10,000 requests per cell at the path's rate x 1, 64-word prompts, 4 scripted mock engines, 8 runtime threads"
    );
    settings.bed.gw_cores = "0-7".to_string();
    settings.distinct = 128;
    settings.rate_scale = 0.5;
    assert_eq!(
        setup_text(&settings, "alloc"),
        "alloc: 10,000 requests per cell at the path's rate x 0.5, 64-word prompts, 128 distinct, 4 scripted mock engines, 8 runtime threads; gateway pinned to cores 0-7"
    );
    settings.bed.fleet = Fleet::PrefillDecode;
    assert_eq!(
        setup_text(&settings, "cpu"),
        "cpu: 10,000 requests per cell at the path's rate x 0.5, 64-word prompts, 128 distinct, 1 prefill and 4 decode scripted mock engines on the prefill-decode path, 8 runtime threads; gateway pinned to cores 0-7"
    );
    settings.bed.fleet = Fleet::Http;
    assert_eq!(
        setup_text(&settings, "cpu"),
        "cpu: 10,000 requests per cell at the path's rate x 0.5, 64-word prompts, 128 distinct, 4 scripted mock engines on the HTTP wire, 8 runtime threads; gateway pinned to cores 0-7"
    );
}

#[test]
fn the_base_runs_under_gateway_flags_of_its_own_where_it_has_any() {
    let cfg = BedConfig::new(Path::new("/bed.sh"), Path::new("/m"), Path::new("/l"), "/t");
    let binaries = BTreeMap::from([
        ("base".to_string(), PathBuf::from("/smg")),
        ("pr".to_string(), PathBuf::from("/smg")),
    ]);
    let mut settings = Settings::new(Path::new("/out"), cfg, binaries, Vec::new());
    let ran_under = |settings: &Settings, side: &str| {
        let bed = settings.side_bed(side, Path::new("/run"), false).unwrap();
        let env: BTreeMap<String, String> = bed.env().into_iter().collect();
        env["GW_ARGS"].clone()
    };
    assert_eq!(ran_under(&settings, "base"), DEFAULT_GW_ARGS);
    assert_eq!(ran_under(&settings, "pr"), DEFAULT_GW_ARGS);
    assert_eq!(settings.flags_text(), "");

    settings.base_gw_args = Some("--parser symphony".to_string());
    settings.bed.gw_args = "--parser symphony --trust-tenant-header".to_string();
    assert_eq!(ran_under(&settings, "base"), "--parser symphony");
    assert_eq!(
        ran_under(&settings, "pr"),
        "--parser symphony --trust-tenant-header"
    );
    assert_eq!(
        setup_text(&settings, "alloc"),
        "alloc: 10,000 requests per cell at the path's rate x 1, 64-word prompts, 4 scripted mock engines, 8 runtime threads; \
             gateway flags before `--parser symphony`, after `--parser symphony --trust-tenant-header`"
    );

    // Flags of its own that equal the head's are no difference to state.
    settings.base_gw_args = Some(settings.bed.gw_args.clone());
    assert_eq!(settings.flags_text(), "");
}
