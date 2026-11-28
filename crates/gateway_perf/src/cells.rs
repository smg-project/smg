//! The cells: a fresh gateway per cell, the client's window, the gateway's counters
//! over it, the per-request arithmetic, the results file and the outputs of a run; the
//! `mem` soak and the bed smoke beside them.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    bed::{enqueue, script_counts, Bed, BedConfig, ScriptCounts},
    counters::{parse_jemalloc, settled_snapshot, snapshot, PerfStat, Snapshot},
    gate, hms, loadavg, now_seconds,
    paths::{answers, interleaved_order, PathSpec},
    py_g,
    report::{numbers_table, thousands},
    SIDES,
};

/// What every mode shares: the run directory, the bed, the binaries per side, the
/// paths and the cell parameters.
#[derive(Clone, Debug)]
pub struct Settings {
    pub out: PathBuf,
    pub bed: BedConfig,
    pub binaries: BTreeMap<String, PathBuf>,
    /// The base's own gateway flags, where it runs under other flags than the
    /// head (`--base-gw-args`): one binary measured under two settings.
    pub base_gw_args: Option<String>,
    pub paths: Vec<PathSpec>,
    pub requests: u64,
    pub repeats: usize,
    pub rate_scale: f64,
    pub idle_seconds: u64,
    pub counters: bool,
    pub counters_note: String,
    pub drain: u64,
    pub prompt_tokens: u64,
    pub distinct: u64,
}

impl Settings {
    pub fn new(
        out: &Path,
        bed: BedConfig,
        binaries: BTreeMap<String, PathBuf>,
        paths: Vec<PathSpec>,
    ) -> Self {
        Self {
            out: out.to_path_buf(),
            bed,
            binaries,
            base_gw_args: None,
            paths,
            requests: 10_000,
            repeats: 3,
            rate_scale: 1.0,
            idle_seconds: 25,
            counters: true,
            counters_note: String::new(),
            drain: 120,
            prompt_tokens: 64,
            distinct: 64,
        }
    }

    pub fn binary(&self, side: &str) -> Result<&Path> {
        self.binaries
            .get(side)
            .map(PathBuf::as_path)
            .ok_or_else(|| anyhow!("no gateway binary for the {side} side"))
    }

    /// The gateway flags `side` runs under: the base's own where it has any,
    /// else the bed's.
    pub fn gw_args(&self, side: &str) -> &str {
        match (side, &self.base_gw_args) {
            ("base", Some(flags)) => flags,
            _ => &self.bed.gw_args,
        }
    }

    /// A bed for `side`: its binary under its gateway flags.
    pub fn side_bed(&self, side: &str, run_dir: &Path, jemalloc_stats: bool) -> Result<Bed> {
        let mut cfg = self.bed.clone();
        cfg.gw_args = self.gw_args(side).to_string();
        Ok(Bed::new(&cfg, self.binary(side)?, run_dir, jemalloc_stats))
    }

    /// What a setup line says where the two sides ran under different gateway
    /// flags: nothing where they ran under the same.
    pub fn flags_text(&self) -> String {
        let (base, head) = (self.gw_args("base"), self.gw_args("pr"));
        if base == head {
            String::new()
        } else {
            format!("; gateway flags before `{base}`, after `{head}`")
        }
    }
}

/// One side of `results.json`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SideResults {
    pub binary: String,
    pub runs: Vec<BTreeMap<String, f64>>,
    pub seconds: Vec<f64>,
}

/// `results.json`: the shape `scripts/bench_gate.py compare` reads. Per side one run per
/// repeat, each run a map of `<path>/<metric>` to its per-request value; `unit` is ""
/// (the ids carry their units) where a criterion file has none (nanoseconds).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Results {
    pub base: SideResults,
    pub bench_args: Vec<String>,
    pub filter: Option<String>,
    pub note: String,
    pub order: Vec<String>,
    pub pr: SideResults,
    pub repeats: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

impl Results {
    pub fn side(&self, side: &str) -> &SideResults {
        if side == "base" {
            &self.base
        } else {
            &self.pr
        }
    }
}

/// The record of one cell (`cell.json`): the client's summary, the counters over the
/// window and the allocator's exit statistics.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CellRecord {
    pub side: String,
    pub path: String,
    pub label: String,
    pub binary: String,
    /// The extra gateway flags the cell's gateway ran under.
    pub gw_args: String,
    pub dir: String,
    pub sent: u64,
    pub ok: u64,
    pub failed: u64,
    pub skipped: u64,
    /// Requests still open when the client's drain ran out, cut off before their
    /// last frame: a warning, as failed requests are.
    pub unfinished: u64,
    pub statuses: Value,
    pub errors: Value,
    pub achieved_rps: Option<f64>,
    pub ttft_p50_ms: Option<f64>,
    pub ttft_p99_ms: Option<f64>,
    pub e2e_p99_ms: Option<f64>,
    pub frames_per_request: Option<f64>,
    pub client_rc: i64,
    pub window_s: f64,
    pub cpu_s: f64,
    pub status_0: BTreeMap<String, u64>,
    pub status_1: BTreeMap<String, u64>,
    pub gauges_0: BTreeMap<String, f64>,
    pub gauges_1: BTreeMap<String, f64>,
    pub loadavg_0: f64,
    pub loadavg_1: f64,
    pub perf: BTreeMap<String, f64>,
    pub script: Value,
    pub jemalloc: Value,
    pub seconds_total: f64,
    pub warnings: Vec<String>,
}

fn count(v: &Value, key: &str) -> u64 {
    v.get(key).and_then(Value::as_u64).unwrap_or(0)
}

fn number(v: &Value, path: &[&str]) -> Option<f64> {
    let mut cur = v;
    for key in path {
        cur = cur.get(key)?;
    }
    cur.as_f64()
}

/// One run of the load client.
pub struct ClientRun<'a> {
    pub spec: &'a PathSpec,
    pub rate: f64,
    pub seconds: u64,
    pub out: &'a Path,
    pub extra: &'a [&'a str],
    pub warmup: u64,
    pub threads: u32,
}

/// Runs `stress_load` for the window and reads its summary (with `client_rc` added).
pub fn run_client(settings: &Settings, bed: &Bed, run: &ClientRun<'_>) -> Result<Value> {
    let mut cmd = if settings.bed.gen_cores.is_empty() {
        Command::new(&settings.bed.load_bin)
    } else {
        let mut c = Command::new("taskset");
        c.arg("-c")
            .arg(&settings.bed.gen_cores)
            .arg(&settings.bed.load_bin);
        c
    };
    cmd.arg("--gateway")
        .arg(bed.gateway_url()?)
        .arg("--distinct")
        .arg(settings.distinct.to_string())
        .arg("--prompt-tokens")
        .arg(settings.prompt_tokens.to_string())
        .arg("--rate")
        .arg(py_g(run.rate, 6))
        .arg("--duration")
        .arg(run.seconds.to_string())
        .arg("--warmup")
        .arg(run.warmup.to_string())
        .arg("--drain")
        .arg(settings.drain.to_string())
        .arg("--threads")
        .arg(run.threads.to_string())
        .arg("--interim")
        .arg("0")
        .args(run.spec.client)
        .args(run.extra)
        .arg("--out")
        .arg(run.out.join("summary.json"));
    let stdout = fs::File::create(run.out.join("client.out"))?;
    let stderr = fs::File::create(run.out.join("client.err"))?;
    let status = cmd
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr)
        .status()?;
    let rc = i64::from(status.code().unwrap_or(-1));
    let summary_path = run.out.join("summary.json");
    if !summary_path.exists() {
        bail!(
            "the client wrote no summary (rc {rc}); see {}",
            run.out.join("client.err").display()
        );
    }
    let mut summary: Value = serde_json::from_str(&fs::read_to_string(&summary_path)?)?;
    if let Some(obj) = summary.as_object_mut() {
        obj.insert("client_rc".to_string(), json!(rc));
    }
    Ok(summary)
}

/// What a cell measured, before the record is built.
struct Measured {
    summary: Value,
    s0: Snapshot,
    s1: Snapshot,
    counters: BTreeMap<String, f64>,
    script: Option<ScriptCounts>,
}

/// The window of a cell: `requests / rate` seconds, at least one (ceil, as the client
/// sends `rate x duration`).
pub fn window_seconds(requests: u64, rate: f64) -> u64 {
    let r = (rate as u64).max(1);
    requests.div_ceil(r).max(1)
}

fn measure(
    settings: &Settings,
    bed: &Bed,
    spec: Option<&PathSpec>,
    cdir: &Path,
) -> Result<Measured> {
    let rate = spec.map_or(0.0, |s| s.rate * settings.rate_scale);
    let seconds = spec.map_or(settings.idle_seconds, |_| {
        window_seconds(settings.requests, rate)
    });
    if let Some(spec) = spec {
        enqueue(
            bed,
            &answers(
                spec,
                (rate * seconds as f64) as usize + 256,
                settings.bed.fleet.wire(),
            ),
        )?;
    }
    thread::sleep(Duration::from_secs(2));
    let s0 = snapshot(bed)?;
    let perf = if settings.counters {
        Some(PerfStat::start(bed.pid()?, &cdir.join("perf.csv"))?)
    } else {
        None
    };
    thread::sleep(Duration::from_millis(500));
    let summary = match spec {
        Some(spec) => run_client(
            settings,
            bed,
            &ClientRun {
                spec,
                rate,
                seconds,
                out: cdir,
                extra: &[],
                warmup: 0,
                threads: 4,
            },
        )?,
        None => {
            thread::sleep(Duration::from_secs(seconds));
            json!({"sent": 0, "ok": 0, "failed": 0, "skipped": 0, "statuses": {}, "errors": {}})
        }
    };
    thread::sleep(Duration::from_secs(1));
    let counters = perf.map(PerfStat::stop).unwrap_or_default();
    // The memory after the window is the gated figure: it is read once the purge this
    // snapshot's scrape starts has ended.
    let s1 = settled_snapshot(bed)?;
    let script = spec.map(|_| script_counts(bed));
    Ok(Measured {
        summary,
        s0,
        s1,
        counters,
        script,
    })
}

/// One fresh gateway of `side`'s binary: a window of requests of `spec` (or an idle
/// window), the gateway's counters over it. Returns the cell record (also written as
/// cell.json under the cell's directory).
pub fn cell(
    settings: &Settings,
    side: &str,
    spec: Option<&PathSpec>,
    label: &str,
) -> Result<CellRecord> {
    let name = spec.map_or("idle", |s| s.name);
    let cdir = settings.out.join("cells").join(side).join(name).join(label);
    fs::create_dir_all(&cdir)?;
    let binary = settings.binary(side)?;
    let mut bed = settings.side_bed(side, &cdir.join("bed"), true)?;
    let t_start = Instant::now();
    println!(
        "[{}] cell {side}/{name}/{label} load={:.2}",
        hms(),
        loadavg()
    );
    bed.up()?;
    let measured = measure(settings, &bed, spec, &cdir);
    bed.down();
    let m = measured?;
    thread::sleep(Duration::from_millis(500));
    let mut record = CellRecord {
        side: side.to_string(),
        path: name.to_string(),
        label: label.to_string(),
        binary: binary.display().to_string(),
        gw_args: settings.gw_args(side).to_string(),
        dir: cdir.display().to_string(),
        ..CellRecord::default()
    };
    fill_record(&mut record, spec.is_some(), &m, &bed.gateway_log());
    record.seconds_total = t_start.elapsed().as_secs_f64();
    let warnings = record.warnings.join("; ");
    println!(
        "    {} sent, {} ok, window {:.1} s, cpu {:.2} s{}",
        record.sent,
        record.ok,
        record.window_s,
        record.cpu_s,
        if warnings.is_empty() {
            String::new()
        } else {
            format!("; WARNING: {warnings}")
        }
    );
    fs::write(
        cdir.join("cell.json"),
        serde_json::to_string_pretty(&record)? + "\n",
    )?;
    Ok(record)
}

/// The record's fields from what was measured, and its warnings: failed requests,
/// requests the client's drain cut off, script misses (answers served by the engine
/// simulator), unreadable script counters and a lost allocator print mean the cell did
/// not measure what it should.
fn fill_record(record: &mut CellRecord, loaded: bool, m: &Measured, gateway_log: &str) {
    let s = &m.summary;
    record.sent = count(s, "sent");
    record.ok = count(s, "ok");
    record.failed = count(s, "failed");
    record.skipped = count(s, "skipped");
    record.unfinished = count(s, "unfinished");
    record.statuses = s.get("statuses").cloned().unwrap_or_else(|| json!({}));
    record.errors = s.get("errors").cloned().unwrap_or_else(|| json!({}));
    record.achieved_rps = number(s, &["achieved_rps"]);
    record.ttft_p50_ms = number(s, &["ttft_ms", "p50"]);
    record.ttft_p99_ms = number(s, &["ttft_ms", "p99"]);
    record.e2e_p99_ms = number(s, &["e2e_ms", "p99"]);
    record.frames_per_request = number(s, &["frames_per_request"]);
    record.client_rc = s.get("client_rc").and_then(Value::as_i64).unwrap_or(0);
    record.window_s = m.s1.t - m.s0.t;
    record.cpu_s = m.s1.cpu_s - m.s0.cpu_s;
    record.status_0.clone_from(&m.s0.status);
    record.status_1.clone_from(&m.s1.status);
    record.gauges_0.clone_from(&m.s0.gauges);
    record.gauges_1.clone_from(&m.s1.gauges);
    record.loadavg_0 = m.s0.loadavg;
    record.loadavg_1 = m.s1.loadavg;
    record.perf.clone_from(&m.counters);
    record.script = m.script.as_ref().map_or_else(|| json!({}), |s| json!(s));
    record.jemalloc = parse_jemalloc(gateway_log);
    let mut bad = Vec::new();
    if loaded && record.failed > 0 {
        bad.push(format!("{} failed requests", record.failed));
    }
    if loaded && record.unfinished > 0 {
        bad.push(format!(
            "{} requests cut off at the client's drain",
            record.unfinished
        ));
    }
    if let Some(script) = &m.script {
        if loaded && script.misses > 0 {
            bad.push(format!(
                "{} script misses (answers served by the engine simulator)",
                script.misses
            ));
        }
        if loaded && script.unreadable > 0 {
            bad.push(format!(
                "{} script counters unreadable (misses unknown)",
                script.unreadable
            ));
        }
    }
    if let Some(err) = record.jemalloc.get("error").and_then(Value::as_str) {
        bad.push(err.to_string());
    }
    record.warnings = bad;
}

/// The per-request numbers of one loaded cell, divided by the requests sent. The
/// counters read over the window (perf's, CPU time) lose the idle cell's per-second
/// rate times the window. The allocator's counters are the process's exit totals, so
/// the idle cell's totals are subtracted whole: that removes the start-up share and
/// the idle allocations over the idle cell's life; what stays beside the requests'
/// own allocations is the idle rate times the difference of the two cells' lives
/// (seconds of a rate of a few allocations per second, spread over the requests), the
/// same on both sides of an A/B, so the comparison is unbiased and the absolute
/// values carry it. Instructions and allocation counts are load-independent; cycles
/// and CPU time are not.
pub fn per_request(c: &CellRecord, idle: Option<&CellRecord>) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    if c.sent == 0 {
        return out;
    }
    let n = c.sent as f64;
    let w = c.window_s;
    let iw = idle.map_or(1.0, |i| if i.window_s == 0.0 { 1.0 } else { i.window_s });
    let idle_rate = |key: &str| idle.map_or(0.0, |i| i.perf.get(key).copied().unwrap_or(0.0) / iw);
    let idle_cpu = idle.map_or(0.0, |i| i.cpu_s);
    out.insert(
        "cpu_us_per_request".to_string(),
        (c.cpu_s - idle_cpu / iw * w) / n * 1e6,
    );
    let kb = |key: &str| c.status_1.get(key).copied().unwrap_or(0) as f64 / 1024.0;
    out.insert("rss_mb_after_window".to_string(), kb("VmRSS"));
    out.insert("hwm_mb".to_string(), kb("VmHWM"));
    // The anonymous and the file part of the resident set, where the kernel reports them
    // (the whole set is these two and shared memory). The anonymous part is what the
    // gateway allocated. The file part is the file pages the kernel has mapped, mostly
    // the gateway's own binary (and its shared libraries): a fault maps the cached
    // neighbours inside the 64 KB window around it, so the part follows how much of the
    // measured file is cached and where the linker put the code that runs, and two
    // copies of one binary read apart. The gate judges the anonymous part.
    for (field, metric) in [
        ("RssAnon", "rss_anon_mb_after_window"),
        ("RssFile", "rss_file_mb_after_window"),
    ] {
        if let Some(kib) = c.status_1.get(field) {
            out.insert(metric.to_string(), *kib as f64 / 1024.0);
        }
    }
    let perf_per_request = |key: &str, divisor: f64| {
        c.perf
            .get(key)
            .map(|v| (v - idle_rate(key) * w) / n / divisor)
    };
    for (key, metric, divisor) in [
        ("instructions:u", "user_instructions_per_request_M", 1e6),
        ("instructions", "instructions_per_request_M", 1e6),
        ("cycles:u", "user_cycles_per_request_M", 1e6),
        ("cycles", "cycles_per_request_M", 1e6),
        ("context-switches", "context_switches_per_request", 1.0),
    ] {
        if let Some(value) = perf_per_request(key, divisor) {
            out.insert(metric.to_string(), value);
        }
    }
    let je = |rec: &CellRecord, key: &str| rec.jemalloc.get(key).and_then(Value::as_f64);
    if let (Some(idle), Some(calls), Some(idle_calls)) = (
        idle,
        je(c, "nrequests"),
        idle.and_then(|i| je(i, "nrequests")),
    ) {
        out.insert(
            "allocation_calls_per_request".to_string(),
            (calls - idle_calls) / n,
        );
        let bytes =
            je(c, "bytes_requested").unwrap_or(0.0) - je(idle, "bytes_requested").unwrap_or(0.0);
        out.insert("kib_requested_per_request".to_string(), bytes / n / 1024.0);
        let large =
            je(c, "large_nrequests").unwrap_or(0.0) - je(idle, "large_nrequests").unwrap_or(0.0);
        out.insert("large_allocations_per_request".to_string(), large / n);
    }
    out
}

/// results.json from the cells: per side one run per repeat (the `r<i>` labels), each
/// run a map of `<path>/<metric>` to its per-request value.
pub fn results_from_cells(
    cells: &[CellRecord],
    repeats: usize,
    binaries: &BTreeMap<String, PathBuf>,
    note: &str,
) -> Results {
    let idle: BTreeMap<&str, &CellRecord> = cells
        .iter()
        .filter(|c| c.path == "idle")
        .map(|c| (c.side.as_str(), c))
        .collect();
    let mut runs: BTreeMap<&str, BTreeMap<usize, BTreeMap<String, f64>>> =
        SIDES.iter().map(|s| (*s, BTreeMap::new())).collect();
    for c in cells.iter().filter(|c| c.path != "idle") {
        let Some(repeat) = c
            .label
            .strip_prefix('r')
            .and_then(|n| n.parse::<usize>().ok())
        else {
            continue;
        };
        let values = per_request(c, idle.get(c.side.as_str()).copied());
        let run = runs
            .entry(c.side.as_str())
            .or_default()
            .entry(repeat)
            .or_default();
        for (metric, value) in values {
            run.insert(format!("{}/{metric}", c.path), value);
        }
    }
    let side = |name: &str| SideResults {
        binary: binaries
            .get(name)
            .map(|p| p.display().to_string())
            .unwrap_or_default(),
        runs: runs
            .get(name)
            .map(|r| r.values().cloned().collect())
            .unwrap_or_default(),
        seconds: Vec::new(),
    };
    Results {
        base: side("base"),
        bench_args: Vec::new(),
        filter: None,
        note: note.to_string(),
        order: interleaved_order(repeats)
            .into_iter()
            .map(|(i, s)| format!("{s}-{i}"))
            .collect(),
        pr: side("pr"),
        repeats,
        unit: Some(String::new()),
    }
}

/// The setup column's text.
pub fn setup_text(settings: &Settings, mode: &str) -> String {
    let cores = if settings.bed.gw_cores.is_empty() {
        String::new()
    } else {
        format!("; gateway pinned to cores {}", settings.bed.gw_cores)
    };
    let mut prompt = format!("{}-word prompts", settings.prompt_tokens);
    if settings.distinct != 64 {
        prompt.push_str(&format!(", {} distinct", settings.distinct));
    }
    format!(
        "{mode}: {} requests per cell at the path's rate x {}, {prompt}, {}, {} runtime threads{cores}{}",
        thousands(settings.requests as f64, 0, false),
        py_g(settings.rate_scale, 6),
        settings.bed.engines_text(),
        settings.bed.gw_threads,
        settings.flags_text()
    )
}

/// Writes cells.json, results.json, numbers.md and gate.md, prints the tables (and
/// appends them to `$GITHUB_STEP_SUMMARY`), and returns the exit code: the gate's, or 2
/// when a cell carries a warning (the tables stay on disk for a read, the run fails).
pub fn write_outputs(
    out: &Path,
    cells: &[Value],
    results: &Results,
    gates: Option<&Path>,
    setup: &str,
) -> Result<i32> {
    fs::write(
        out.join("cells.json"),
        serde_json::to_string_pretty(cells)? + "\n",
    )?;
    fs::write(
        out.join("results.json"),
        serde_json::to_string_pretty(results)? + "\n",
    )?;
    let loads: Vec<f64> = cells
        .iter()
        .flat_map(|c| {
            ["loadavg_0", "loadavg_1"]
                .into_iter()
                .filter_map(|k| c.get(k).and_then(Value::as_f64))
        })
        .collect();
    let mut numbers = numbers_table(results, setup);
    if let (Some(min), Some(max)) = (
        loads.iter().copied().reduce(f64::min),
        loads.iter().copied().reduce(f64::max),
    ) {
        numbers.push_str(&format!(
            "\n\n1-minute load average of the machine over the cells: {min:.0}-{max:.0} \
             (cycles, CPU time and latencies carry it; instructions and allocation counts do not)."
        ));
    }
    fs::write(out.join("numbers.md"), format!("{numbers}\n"))?;
    let (gate_md, rc) = gate::gate_report(results, gates)?;
    fs::write(out.join("gate.md"), &gate_md)?;
    println!("\n## Numbers\n\n{numbers}\n\n{gate_md}");
    if let Ok(summary) = std::env::var("GITHUB_STEP_SUMMARY") {
        if !summary.is_empty() {
            use std::io::Write;
            let mut file = fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(summary)?;
            writeln!(file, "## Gateway perf: {setup}\n\n{numbers}\n\n{gate_md}")?;
        }
    }
    let warnings: Vec<String> = cells
        .iter()
        .flat_map(|c| {
            let who = format!(
                "{}/{}/{}",
                c["side"].as_str().unwrap_or_default(),
                c["path"].as_str().unwrap_or_default(),
                c["label"].as_str().unwrap_or_default()
            );
            c.get("warnings")
                .and_then(Value::as_array)
                .map(|ws| {
                    ws.iter()
                        .filter_map(|w| w.as_str().map(|w| format!("{who}: {w}")))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        })
        .collect();
    println!("results: {}", out.join("results.json").display());
    if !warnings.is_empty() {
        println!("WARNINGS:\n  {}", warnings.join("\n  "));
        eprintln!(
            "gateway-perf: {} cell warning(s); the run did not measure what it should",
            warnings.len()
        );
        return Ok(2);
    }
    Ok(rc)
}

/// `cpu` and `alloc`: the interleaved base/head cells over the paths.
pub fn run_cells(settings: &Settings, mode: &str, gates: Option<&Path>) -> Result<i32> {
    let mut cells = Vec::new();
    for side in SIDES {
        cells.push(cell(settings, side, None, "idle")?);
    }
    for (i, side) in interleaved_order(settings.repeats) {
        for spec in &settings.paths {
            cells.push(cell(settings, side, Some(spec), &format!("r{i}"))?);
        }
    }
    let results = results_from_cells(
        &cells,
        settings.repeats,
        &settings.binaries,
        &settings.counters_note,
    );
    let values: Vec<Value> = cells
        .iter()
        .map(serde_json::to_value)
        .collect::<Result<_, _>>()?;
    write_outputs(
        &settings.out,
        &values,
        &results,
        gates,
        &setup_text(settings, mode),
    )
}

/// One row of a soak: the snapshot after a window.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoakRow {
    pub after: String,
    #[serde(flatten)]
    pub snap: Snapshot,
}

impl SoakRow {
    fn rss_mb(&self) -> f64 {
        self.snap.status.get("VmRSS").copied().unwrap_or(0) as f64 / 1024.0
    }

    fn hwm_mb(&self) -> f64 {
        self.snap.status.get("VmHWM").copied().unwrap_or(0) as f64 / 1024.0
    }

    fn live_mb(&self) -> f64 {
        self.snap
            .gauges
            .get("smg_allocator_allocated_bytes")
            .copied()
            .unwrap_or(0.0)
            / 1e6
    }
}

/// What a soak measured: the rows after every window, the client summaries, the script counts.
pub struct SoakMeasured {
    pub rows: Vec<SoakRow>,
    pub summaries: Vec<Value>,
    pub script: ScriptCounts,
}

/// The record of a soak (`soak.json`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoakRecord {
    pub side: String,
    pub path: String,
    pub label: String,
    pub dir: String,
    pub rows: Vec<SoakRow>,
    pub loadavg_0: f64,
    pub loadavg_1: f64,
    pub values: BTreeMap<String, f64>,
    pub sent: u64,
    pub ok: u64,
    pub failed: u64,
    /// Requests still open when the client's drain ran out, summed over the windows.
    #[serde(default)]
    pub unfinished: u64,
    pub client_rc: i64,
    pub script: ScriptCounts,
    pub warnings: Vec<String>,
}

/// The soak's record from its rows and its client summaries: RSS after the first and
/// the last window, the growth per 10,000 requests, the peak after the burst and the
/// allocator's live bytes; the request counts and the warnings as a cell's, so a soak
/// with failed requests, requests cut off at the drain or script misses fails the run.
pub fn soak_record(
    side: &str,
    spec: &PathSpec,
    label: &str,
    dir: &Path,
    measured: SoakMeasured,
    windows: usize,
    requests: u64,
) -> Result<SoakRecord> {
    let SoakMeasured {
        rows,
        summaries,
        script,
    } = measured;
    let (Some(first), Some(last), Some(after_burst)) =
        (rows.get(1), rows.get(windows), rows.get(windows + 1))
    else {
        bail!("the soak has {} rows for {windows} windows", rows.len());
    };
    let per_10k = 10_000.0 / requests as f64;
    let key = |metric: &str| format!("mem-{}/{metric}", spec.name);
    let values = BTreeMap::from([
        (key("rss_mb_after_first_window"), first.rss_mb()),
        (key("rss_mb_after_last_window"), last.rss_mb()),
        (
            key("rss_growth_mb_per_10k_requests"),
            (last.rss_mb() - first.rss_mb()) / (windows.saturating_sub(1).max(1)) as f64 * per_10k,
        ),
        (key("peak_rss_mb_after_burst"), after_burst.hwm_mb()),
        (key("allocator_live_mb_after_last_window"), last.live_mb()),
    ]);
    let failed: u64 = summaries.iter().map(|s| count(s, "failed")).sum();
    let unfinished: u64 = summaries.iter().map(|s| count(s, "unfinished")).sum();
    let mut bad = Vec::new();
    if failed > 0 {
        bad.push(format!("{failed} failed requests"));
    }
    if unfinished > 0 {
        bad.push(format!(
            "{unfinished} requests cut off at the client's drain"
        ));
    }
    if script.misses > 0 {
        bad.push(format!(
            "{} script misses (answers served by the engine simulator)",
            script.misses
        ));
    }
    if script.unreadable > 0 {
        bad.push(format!(
            "{} script counters unreadable (misses unknown)",
            script.unreadable
        ));
    }
    Ok(SoakRecord {
        side: side.to_string(),
        path: spec.name.to_string(),
        label: label.to_string(),
        dir: dir.display().to_string(),
        loadavg_0: rows.first().map_or(0.0, |r| r.snap.loadavg),
        loadavg_1: rows.last().map_or(0.0, |r| r.snap.loadavg),
        rows,
        values,
        sent: summaries.iter().map(|s| count(s, "sent")).sum(),
        ok: summaries.iter().map(|s| count(s, "ok")).sum(),
        failed,
        unfinished,
        client_rc: summaries
            .iter()
            .map(|s| s.get("client_rc").and_then(Value::as_i64).unwrap_or(0))
            .max()
            .unwrap_or(0),
        script,
        warnings: bad,
    })
}

fn soak_windows(
    settings: &Settings,
    bed: &Bed,
    spec: &PathSpec,
    cdir: &Path,
    windows: usize,
    burst: f64,
) -> Result<SoakMeasured> {
    let rate = spec.rate * settings.rate_scale;
    let seconds = window_seconds(settings.requests, rate);
    let mut rows = Vec::new();
    let mut summaries = Vec::new();
    // Every row is a memory read after idle seconds, as a cell's read after its window is.
    let row = |after: String| -> Result<SoakRow> {
        Ok(SoakRow {
            after,
            snap: settled_snapshot(bed)?,
        })
    };
    thread::sleep(Duration::from_secs(2));
    rows.push(row("start".to_string())?);
    for w in 1..=windows {
        let wdir = cdir.join(format!("window-{w}"));
        fs::create_dir_all(&wdir)?;
        enqueue(
            bed,
            &answers(
                spec,
                (rate * seconds as f64) as usize + 256,
                settings.bed.fleet.wire(),
            ),
        )?;
        summaries.push(run_client(
            settings,
            bed,
            &ClientRun {
                spec,
                rate,
                seconds,
                out: &wdir,
                extra: &[],
                warmup: 0,
                threads: 4,
            },
        )?);
        thread::sleep(Duration::from_secs(2));
        rows.push(row(format!(
            "window {w} ({} requests)",
            thousands((w as u64 * settings.requests) as f64, 0, false)
        ))?);
    }
    let bdir = cdir.join("burst");
    fs::create_dir_all(&bdir)?;
    enqueue(
        bed,
        &answers(
            spec,
            (rate * burst * seconds as f64) as usize + 256,
            settings.bed.fleet.wire(),
        ),
    )?;
    summaries.push(run_client(
        settings,
        bed,
        &ClientRun {
            spec,
            rate: rate * burst,
            seconds,
            out: &bdir,
            extra: &[],
            warmup: 0,
            threads: 4,
        },
    )?);
    thread::sleep(Duration::from_secs(2));
    rows.push(row(format!("burst window (x{} rate)", py_g(burst, 6)))?);
    thread::sleep(Duration::from_secs(10));
    rows.push(row("10 s idle".to_string())?);
    Ok(SoakMeasured {
        rows,
        summaries,
        script: script_counts(bed),
    })
}

/// One gateway, `windows` windows of requests, RSS/HWM and the allocator gauges after
/// every window, then one window at `burst` x the rate for the peak.
pub fn mem_soak(
    settings: &Settings,
    side: &str,
    spec: &PathSpec,
    label: &str,
    windows: usize,
    burst: f64,
) -> Result<SoakRecord> {
    let cdir = settings
        .out
        .join("cells")
        .join(side)
        .join(format!("mem-{}", spec.name))
        .join(label);
    fs::create_dir_all(&cdir)?;
    let mut bed = settings.side_bed(side, &cdir.join("bed"), false)?;
    println!(
        "[{}] mem {side}/{}/{label}: {windows} windows of {} + a burst at x{}",
        hms(),
        spec.name,
        thousands(settings.requests as f64, 0, false),
        py_g(burst, 6)
    );
    bed.up()?;
    let measured = soak_windows(settings, &bed, spec, &cdir, windows, burst);
    bed.down();
    let record = soak_record(
        side,
        spec,
        label,
        &cdir,
        measured?,
        windows,
        settings.requests,
    )?;
    if !record.warnings.is_empty() {
        println!("    WARNING: {}", record.warnings.join("; "));
    }
    fs::write(
        cdir.join("soak.json"),
        serde_json::to_string_pretty(&record)? + "\n",
    )?;
    for r in &record.rows {
        println!(
            "    after {}: RSS {:.1} MB, HWM {:.1} MB, live {:.1} MB",
            r.after,
            r.rss_mb(),
            r.hwm_mb(),
            r.live_mb()
        );
    }
    Ok(record)
}

/// `mem`: the soaks per side, interleaved, and their results file.
pub fn run_mem(
    settings: &Settings,
    windows: usize,
    burst: f64,
    gates: Option<&Path>,
) -> Result<i32> {
    let mut records = Vec::new();
    for (i, side) in interleaved_order(settings.repeats) {
        for spec in &settings.paths {
            records.push(mem_soak(
                settings,
                side,
                spec,
                &format!("r{i}"),
                windows,
                burst,
            )?);
        }
    }
    let side = |name: &str| -> Result<SideResults> {
        let mut runs: BTreeMap<usize, BTreeMap<String, f64>> = BTreeMap::new();
        for r in records.iter().filter(|r| r.side == name) {
            let repeat: usize = r.label.trim_start_matches('r').parse()?;
            runs.entry(repeat)
                .or_default()
                .extend(r.values.iter().map(|(k, v)| (k.clone(), *v)));
        }
        Ok(SideResults {
            binary: settings.binary(name)?.display().to_string(),
            runs: runs.into_values().collect(),
            seconds: Vec::new(),
        })
    };
    let results = Results {
        base: side("base")?,
        bench_args: Vec::new(),
        filter: None,
        note: String::new(),
        order: interleaved_order(settings.repeats)
            .into_iter()
            .map(|(i, s)| format!("{s}-{i}"))
            .collect(),
        pr: side("pr")?,
        repeats: settings.repeats,
        unit: Some(String::new()),
    };
    let setup = format!(
        "mem: {windows} windows of {} requests on one gateway, then a burst at x{}; {}{}",
        thousands(settings.requests as f64, 0, false),
        py_g(burst, 6),
        settings.bed.engines_text(),
        settings.flags_text()
    );
    let values: Vec<Value> = records
        .iter()
        .map(serde_json::to_value)
        .collect::<Result<_, _>>()?;
    write_outputs(&settings.out, &values, &results, gates, &setup)
}

/// `bed smoke`: the bed up, one request per path with the first response shown, the
/// script counts, the bed down. Returns 0 when every path answered from the script.
pub fn smoke(settings: &Settings, paths: &[PathSpec]) -> Result<i32> {
    let mut bed = settings.side_bed("pr", &settings.out.join("bed"), true)?;
    let info = bed.up()?.clone();
    println!(
        "bed up: gateway pid {} at {}, {}, bed.json {}",
        info.gateway_pid,
        info.gateway_url,
        settings.bed.engines_text(),
        settings
            .out
            .join("bed")
            .join("bed")
            .join("bed.json")
            .display()
    );
    let result = smoke_paths(settings, &bed, paths);
    bed.down();
    let rc = result?;
    println!("smoke {}", if rc == 0 { "passed" } else { "FAILED" });
    Ok(rc)
}

fn smoke_paths(settings: &Settings, bed: &Bed, paths: &[PathSpec]) -> Result<i32> {
    let mut rc = 0;
    for spec in paths {
        let sdir = settings.out.join("smoke").join(spec.name);
        fs::create_dir_all(&sdir)?;
        enqueue(bed, &answers(spec, 16, settings.bed.fleet.wire()))?;
        let summary = run_client(
            settings,
            bed,
            &ClientRun {
                spec,
                rate: 4.0,
                seconds: 1,
                out: &sdir,
                extra: &["--debug-first"],
                warmup: 0,
                threads: 2,
            },
        )?;
        let counts = script_counts(bed);
        let first = fs::read_to_string(sdir.join("client.err"))
            .unwrap_or_default()
            .lines()
            .rfind(|l| {
                let lower = l.to_lowercase();
                lower.contains("first") && lower.contains("body")
            })
            .map(|l| l.chars().take(300).collect::<String>())
            .unwrap_or_default();
        let ok = count(&summary, "ok") > 0
            && count(&summary, "failed") == 0
            && counts.misses == 0
            && counts.unreadable == 0;
        if !ok {
            rc = 1;
        }
        println!(
            "{} {}: sent {}, ok {}, failed {}, statuses {}, frames/req {}, script hits {} misses {} unreadable {}",
            if ok { "ok  " } else { "FAIL" },
            spec.name,
            count(&summary, "sent"),
            count(&summary, "ok"),
            count(&summary, "failed"),
            summary.get("statuses").map(Value::to_string).unwrap_or_default(),
            number(&summary, &["frames_per_request"]).map(crate::py_float).unwrap_or_else(|| "None".to_string()),
            counts.hits,
            counts.misses,
            counts.unreadable
        );
        if !first.is_empty() {
            println!("     {first}");
        }
    }
    Ok(rc)
}

/// The record that `bed up` prints and `bed down` needs: the same bed object.
pub fn bed_for(settings: &Settings, jemalloc_stats: bool) -> Result<Bed> {
    settings.side_bed("pr", &settings.out.join("bed"), jemalloc_stats)
}

/// The log line of a sleep-free timestamp (for the `live` mode's steps).
pub fn elapsed_since(t: f64) -> f64 {
    now_seconds() - t
}

#[cfg(test)]
#[path = "../tests/unit/cells/tests.rs"]
pub(crate) mod tests;
