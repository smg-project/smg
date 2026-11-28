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
    let status = "Name:\tsmg\nVmRSS:\t   61440 kB\nVmHWM:\t   70000 kB\nThreads:\t9\nOther:\t1\n";
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
