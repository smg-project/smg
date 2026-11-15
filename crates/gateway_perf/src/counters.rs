//! The gateway process's counters, read from outside it: CPU time and memory from
//! `/proc`, the allocator gauges from `/metrics`, hardware counters from `perf stat`,
//! and jemalloc's exit statistics from the gateway's log.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::OnceLock,
    thread,
    time::{Duration, Instant},
};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{bed::Bed, http, loadavg, now_seconds};

/// Clock ticks per second of the `/proc` times (USER_HZ): `getconf CLK_TCK` once, 100
/// (the value of every Linux port) when it cannot be asked.
fn clk_tck() -> f64 {
    static TCK: OnceLock<f64> = OnceLock::new();
    *TCK.get_or_init(|| {
        Command::new("getconf")
            .arg("CLK_TCK")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|v| *v > 0.0)
            .unwrap_or(100.0)
    })
}

/// Where the kernel publishes a process's times and memory; a test reads a directory of
/// its own in its place.
const PROC: &str = "/proc";

/// utime + stime of the whole process (every thread), in seconds, from `<proc>/<pid>/stat`.
fn proc_cpu_seconds(proc: &Path, pid: u32) -> Result<f64> {
    let text = fs::read_to_string(proc.join(pid.to_string()).join("stat"))?;
    cpu_seconds_of_stat(&text, clk_tck())
}

fn cpu_seconds_of_stat(text: &str, tck: f64) -> Result<f64> {
    // The command name sits in parentheses and may hold spaces: the fields are read
    // after the last ')'.
    let rest = &text[text
        .rfind(')')
        .ok_or_else(|| anyhow!("no comm in the stat line"))?
        + 2..];
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let utime: f64 = fields
        .get(11)
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| anyhow!("no utime"))?;
    let stime: f64 = fields
        .get(12)
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| anyhow!("no stime"))?;
    Ok((utime + stime) / tck)
}

/// VmRSS, VmHWM, RssAnon, RssFile and Threads of `<proc>/<pid>/status` (kB, a count).
fn proc_status(proc: &Path, pid: u32) -> Result<BTreeMap<String, u64>> {
    let text = fs::read_to_string(proc.join(pid.to_string()).join("status"))?;
    Ok(status_fields(&text))
}

fn status_fields(text: &str) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if matches!(name, "VmRSS" | "VmHWM" | "RssAnon" | "RssFile" | "Threads") {
            if let Some(v) = value.split_whitespace().next().and_then(|v| v.parse().ok()) {
                out.insert(name.to_string(), v);
            }
        }
    }
    out
}

pub const GAUGES: [&str; 3] = [
    "smg_allocator_allocated_bytes",
    "smg_allocator_resident_bytes",
    "smg_allocator_active_bytes",
];

/// The allocator gauges of a Prometheus text exposition.
pub fn metrics_gauges(text: &str) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        for name in GAUGES {
            if line.starts_with(&format!("{name} ")) || line.starts_with(&format!("{name}{{")) {
                if let Some(value) = line.rsplit(' ').next().and_then(|v| v.parse().ok()) {
                    out.insert(name.to_string(), value);
                }
            }
        }
    }
    out
}

/// The gateway's state at one instant: the clock, its CPU time, its memory, the
/// allocator gauges and the machine's load.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Snapshot {
    pub t: f64,
    pub cpu_s: f64,
    pub status: BTreeMap<String, u64>,
    pub gauges: BTreeMap<String, f64>,
    pub loadavg: f64,
}

/// How long a settled snapshot waits between its scrape and its memory read. The scrape
/// is the gateway's first allocating work after an idle second, and the allocator's purge
/// pass it starts steps the anonymous set down for up to half a second. Read four times a
/// second beside the prefill-decode chat cell, the set stepped down within half a second
/// of the scrape in every run and stayed level from there to the process's end.
pub const STATUS_SETTLE: Duration = Duration::from_secs(1);

/// The gateway's state now: the scrape, then the clock, the CPU time and the memory.
pub fn snapshot(bed: &Bed) -> Result<Snapshot> {
    snapshot_with(Path::new(PROC), bed, || ())
}

/// A [`snapshot`] for the read after a window: the memory is read [`STATUS_SETTLE`]
/// after the scrape, once the purge the scrape started has ended. The clock and the CPU
/// time are taken before the wait, where [`snapshot`] takes them, so the wait is in
/// neither.
pub fn settled_snapshot(bed: &Bed) -> Result<Snapshot> {
    snapshot_with(Path::new(PROC), bed, || thread::sleep(STATUS_SETTLE))
}

/// The order every snapshot keeps: the scrape, the clock and the CPU time, `settle`, the
/// memory.
fn snapshot_with(proc: &Path, bed: &Bed, settle: impl FnOnce()) -> Result<Snapshot> {
    let pid = bed.pid()?;
    let gauges = http::get(bed.metrics_url()?, Duration::from_secs(10))
        .map(|text| metrics_gauges(&text))
        .unwrap_or_default();
    let t = now_seconds();
    let cpu_s = proc_cpu_seconds(proc, pid)?;
    settle();
    Ok(Snapshot {
        t,
        cpu_s,
        status: proc_status(proc, pid)?,
        gauges,
        loadavg: loadavg(),
    })
}

pub const PERF_EVENTS: &str =
    "instructions,instructions:u,cycles,cycles:u,task-clock,context-switches";

/// Whether `perf stat` can count instructions here (the tool installed, the kernel's
/// perf_event_paranoid allowing it, the hardware exposing a PMU: a virtual machine of a
/// hosted CI runner usually does not).
pub fn perf_available() -> (bool, String) {
    let output = match Command::new("perf")
        .args(["stat", "-x,", "-e", "instructions:u", "--", "true"])
        .stdin(Stdio::null())
        .output()
    {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return (false, "perf not installed".to_string())
        }
        Err(err) => return (false, format!("perf stat failed: {err}")),
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.status.success() {
        return (
            false,
            format!(
                "perf stat exited with {}: {}",
                output.status.code().unwrap_or(-1),
                text.trim().chars().take(200).collect::<String>()
            ),
        );
    }
    for line in text.lines() {
        if line.contains("instructions") && !line.starts_with('#') {
            let value = line.split(',').next().unwrap_or_default();
            if value == "<not supported>" || value == "<not counted>" {
                return (false, format!("instructions {value}"));
            }
            return (true, "perf stat counts instructions".to_string());
        }
    }
    (false, "perf stat printed no instructions line".to_string())
}

/// `perf stat` on one pid, without a workload: it counts until SIGINT, then prints.
pub struct PerfStat {
    child: Child,
    out: PathBuf,
}

impl PerfStat {
    pub fn start(pid: u32, out: &Path) -> Result<Self> {
        let child = Command::new("perf")
            .args([
                "stat",
                "-e",
                PERF_EVENTS,
                "-x,",
                "-p",
                &pid.to_string(),
                "-o",
            ])
            .arg(out)
            .env("DEBUGINFOD_URLS", "")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("starting perf stat")?;
        Ok(Self {
            child,
            out: out.to_path_buf(),
        })
    }

    /// SIGINT to perf, its exit awaited (a minute at most), its csv parsed.
    pub fn stop(mut self) -> BTreeMap<String, f64> {
        let _ = Command::new("kill")
            .args(["-INT", &self.child.id().to_string()])
            .status();
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(50)),
                _ => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    break;
                }
            }
        }
        parse_perf_csv(&fs::read_to_string(&self.out).unwrap_or_default())
    }
}

/// The counters of a `perf stat -x,` csv: value by event, the unsupported ones and the
/// comments skipped.
pub fn parse_perf_csv(text: &str) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() >= 3 && fields[0] != "<not counted>" && fields[0] != "<not supported>" {
            if let Ok(value) = fields[0].parse::<f64>() {
                out.insert(fields[2].to_string(), value);
            }
        }
    }
    out
}

/// jemalloc's exit statistics (the JSON the gateway prints at a clean shutdown with
/// `stats_print:true,stats_print_opts:Jaxe`): allocation requests over the process's
/// life (tcache hits included) and the bytes requested, summed as size class x
/// requests over the merged arena's bins and large extents. An object with `error`
/// when the log holds none.
pub fn parse_jemalloc(log: &str) -> Value {
    let Some(i) = log.find("\"jemalloc\"") else {
        return json!({"error": "no jemalloc statistics in the gateway log"});
    };
    let Some(j0) = log[..i].rfind('{') else {
        return json!({"error": "no jemalloc statistics in the gateway log"});
    };
    let mut stream = serde_json::Deserializer::from_str(&log[j0..]).into_iter::<Value>();
    let obj = match stream.next() {
        Some(Ok(obj)) => obj,
        Some(Err(err)) => return json!({"error": format!("jemalloc statistics unreadable: {err}")}),
        None => return json!({"error": "jemalloc statistics unreadable: empty"}),
    };
    let st = &obj["jemalloc"];
    let merged = &st["stats.arenas"]["merged"];
    let (small, large) = (&merged["small"], &merged["large"]);
    let count = |v: &Value, key: &str| v.get(key).and_then(Value::as_u64).unwrap_or(0);
    let empty = Vec::new();
    let bins_cfg = st["arenas"]["bin"].as_array().unwrap_or(&empty);
    let lext_cfg = st["arenas"]["lextent"].as_array().unwrap_or(&empty);
    let mut bytes_requested: u64 = 0;
    for (k, b) in merged["bins"]
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .enumerate()
    {
        if let Some(size) = bins_cfg.get(k).map(|c| count(c, "size")).filter(|s| *s > 0) {
            bytes_requested += size * count(b, "nrequests");
        }
    }
    for (k, b) in merged["lextents"]
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .enumerate()
    {
        if let Some(size) = lext_cfg.get(k).map(|c| count(c, "size")).filter(|s| *s > 0) {
            bytes_requested += size * count(b, "nrequests");
        }
    }
    json!({
        "nrequests": count(small, "nrequests") + count(large, "nrequests"),
        "small_nrequests": count(small, "nrequests"),
        "large_nrequests": count(large, "nrequests"),
        "nmalloc": count(small, "nmalloc") + count(large, "nmalloc"),
        "bytes_requested": bytes_requested,
        "allocated": st["stats"].get("allocated").cloned().unwrap_or(Value::Null),
        "resident": st["stats"].get("resident").cloned().unwrap_or(Value::Null),
    })
}

#[cfg(test)]
mod tests {
    use std::{
        io::{self, Read, Write},
        net::TcpListener,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
    };

    use super::*;
    use crate::bed::{BedConfig, BedScript};

    /// A bed script that only writes the record `Bed::up` reads back.
    struct WritesRecord {
        dir: PathBuf,
        record: Value,
    }

    impl BedScript for WritesRecord {
        fn run(&self, action: &str, _env: &[(String, String)], _log: &Path) -> io::Result<i32> {
            if action == "up" {
                fs::create_dir_all(&self.dir)?;
                fs::write(self.dir.join("bed.json"), self.record.to_string())?;
            }
            Ok(0)
        }
    }

    /// A bed that is up on paper: a gateway process `pid`, its metrics at `metrics_url`.
    fn bed_of(run_dir: &Path, pid: u32, metrics_url: &str) -> Bed {
        let cfg = BedConfig::new(Path::new("/bed.sh"), Path::new("/m"), Path::new("/l"), "/t");
        let script = WritesRecord {
            dir: run_dir.join("bed"),
            record: json!({
                "gateway_pid": pid,
                "gateway_url": "http://127.0.0.1:1",
                "metrics_url": metrics_url,
                "admin_url": "http://127.0.0.1:1",
            }),
        };
        let mut bed = Bed::with_script(&cfg, Path::new("/g"), run_dir, false, Box::new(script));
        bed.up().unwrap();
        bed
    }

    /// A metrics endpoint that answers `scrapes` requests and counts them.
    fn metrics_endpoint(scrapes: usize) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/metrics", listener.local_addr().unwrap());
        let seen = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&seen);
        thread::spawn(move || {
            for _ in 0..scrapes {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                let mut head = Vec::new();
                let mut chunk = [0u8; 512];
                while !head.windows(4).any(|end| end == b"\r\n\r\n") {
                    match stream.read(&mut chunk) {
                        Ok(n) if n > 0 => head.extend_from_slice(&chunk[..n]),
                        _ => break,
                    }
                }
                counted.fetch_add(1, Ordering::SeqCst);
                let body = "smg_allocator_allocated_bytes 1000\n";
                let answer = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(answer.as_bytes());
            }
        });
        (url, seen)
    }

    /// The order a snapshot keeps. What the scrape starts ends during the wait: the purge
    /// steps the anonymous set down and the process runs on. The memory is the value
    /// after it; the clock and the CPU time are the values before it.
    #[test]
    fn a_snapshot_reads_the_memory_after_its_wait_and_the_clock_and_the_cpu_time_before() {
        let dir = tempfile::tempdir().unwrap();
        let proc = dir.path().join("proc");
        let files = proc.join("4242");
        fs::create_dir_all(&files).unwrap();
        let stat = |utime: u32| {
            format!("4242 (smg) S 1 2 3 4 5 6 7 8 9 10 {utime} 50 0 0 20 0 9 0 100 1 2 3\n")
        };
        fs::write(files.join("stat"), stat(250)).unwrap();
        fs::write(files.join("status"), "RssAnon:\t   24432 kB\n").unwrap();
        let (url, scrapes) = metrics_endpoint(1);
        let bed = bed_of(dir.path(), 4242, &url);

        let mut waited_at = 0.0;
        let snap = snapshot_with(&proc, &bed, || {
            waited_at = now_seconds();
            assert_eq!(
                scrapes.load(Ordering::SeqCst),
                1,
                "the wait follows the scrape"
            );
            fs::write(files.join("status"), "RssAnon:\t   22648 kB\n").unwrap();
            fs::write(files.join("stat"), stat(900)).unwrap();
        })
        .unwrap();

        assert_eq!(
            snap.status["RssAnon"], 22648,
            "the memory is read after the wait"
        );
        assert_eq!(
            snap.cpu_s,
            300.0 / clk_tck(),
            "the CPU time is read before the wait"
        );
        assert!(snap.t <= waited_at, "the clock is read before the wait");
        assert_eq!(snap.gauges["smg_allocator_allocated_bytes"], 1000.0);
    }

    /// The read after a window waits `STATUS_SETTLE` between its scrape and its memory
    /// read, on the process's own `/proc` entry.
    #[test]
    fn a_settled_snapshot_waits_out_the_purge_its_scrape_starts() {
        let dir = tempfile::tempdir().unwrap();
        let (url, scrapes) = metrics_endpoint(1);
        let bed = bed_of(dir.path(), std::process::id(), &url);

        let start = Instant::now();
        let snap = settled_snapshot(&bed).unwrap();

        assert!(start.elapsed() >= STATUS_SETTLE);
        assert_eq!(scrapes.load(Ordering::SeqCst), 1);
        assert!(snap.status["VmRSS"] > 0);
        assert!(snap.cpu_s >= 0.0);
    }

    #[test]
    fn perf_csv_skips_unsupported_and_comments() {
        let text = "# started\n123,,instructions:u,1000,100.00,,\n<not supported>,,cycles,0,100.00,,\n4.5,msec,task-clock,1,100.00,,\n";
        let parsed = parse_perf_csv(text);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed["instructions:u"], 123.0);
        assert_eq!(parsed["task-clock"], 4.5);
    }

    const JEMALLOC_LOG: &str = concat!(
        "gateway lines\n",
        r#"{"jemalloc": {"stats": {"allocated": 10, "resident": 20},"#,
        r#" "arenas": {"bin": [{"size": 8}, {"size": 16}], "lextent": [{"size": 16384}]},"#,
        r#" "stats.arenas": {"merged": {"small": {"nrequests": 30, "nmalloc": 20}, "large": {"nrequests": 2, "nmalloc": 2},"#,
        r#" "bins": [{"nrequests": 10}, {"nrequests": 20}], "lextents": [{"nrequests": 2}]}}}}"#,
        "\ntrailer\n"
    );

    #[test]
    fn jemalloc_exit_statistics_count_requests_and_bytes_by_size_class() {
        let je = parse_jemalloc(JEMALLOC_LOG);
        assert_eq!(je["nrequests"], 32);
        assert_eq!(je["small_nrequests"], 30);
        assert_eq!(je["large_nrequests"], 2);
        assert_eq!(je["bytes_requested"], 10 * 8 + 20 * 16 + 2 * 16384);
        assert_eq!(je["nmalloc"], 22);
        assert_eq!(je["allocated"], 10);
        assert!(parse_jemalloc("no statistics here").get("error").is_some());
        assert!(parse_jemalloc(r#"{"jemalloc": {broken"#)
            .get("error")
            .is_some());
    }

    #[test]
    fn proc_stat_and_status_lines_are_read() {
        let stat = "123 (smg worker) S 1 2 3 4 5 6 7 8 9 10 250 50 0 0 20 0 9 0 100 1 2 3\n";
        assert_eq!(cpu_seconds_of_stat(stat, 100.0).unwrap(), 3.0);
        let status =
            "Name:\tsmg\nVmRSS:\t   61440 kB\nVmHWM:\t   70000 kB\nThreads:\t9\nOther:\t1\n";
        let fields = status_fields(status);
        assert_eq!(fields["VmRSS"], 61440);
        assert_eq!(fields["VmHWM"], 70000);
        assert_eq!(fields["Threads"], 9);
        assert!(!fields.contains_key("Other"));
        assert!(clk_tck() > 0.0);
    }

    #[test]
    fn gauges_are_read_with_or_without_labels() {
        let text = "# HELP x\nsmg_allocator_allocated_bytes 19100000\nsmg_allocator_resident_bytes{a=\"b\"} 2e7\nother 1\n";
        let gauges = metrics_gauges(text);
        assert_eq!(gauges["smg_allocator_allocated_bytes"], 19_100_000.0);
        assert_eq!(gauges["smg_allocator_resident_bytes"], 2e7);
        assert_eq!(gauges.len(), 2);
    }
}
