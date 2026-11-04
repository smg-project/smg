//! One bed (the gateway + the scripted mock engines) under a run directory, through
//! `scripts/bench/grpc_stress/bed.sh`, and the mock engines' script API.

use std::{
    fs, io,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Duration,
};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{http, paths::DEFAULT_GW_ARGS};

/// Whether a loopback port can be bound right now.
pub fn port_free(port: u16) -> bool {
    TcpListener::bind(("127.0.0.1", port)).is_ok()
}

/// The bed's ports: the gateway, its metrics, the mock admin API and the first engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ports {
    pub gateway: u16,
    pub metrics: u16,
    pub admin: u16,
    pub mock_base: u16,
}

impl Ports {
    fn block(base: u16, workers: u16) -> (Self, Vec<u16>) {
        let ports = Self {
            gateway: base,
            metrics: base + 1,
            admin: base + 9,
            mock_base: base + 10,
        };
        let mut all = vec![ports.gateway, ports.metrics, ports.admin];
        all.extend((0..workers).map(|i| ports.mock_base + i));
        (ports, all)
    }
}

/// The bed's ports: `base` when given (gateway base, metrics base+1, admin base+9,
/// workers from base+10), else the first block of 10 + workers free ports in
/// `[first, last]` (bed.sh refuses a port another process holds, so the choice is
/// checked again at start).
pub fn pick_ports(base: u16, workers: u16, first: u16, last: u16) -> Result<Ports> {
    let span = 10 + workers;
    let candidates: Vec<u16> = if base != 0 {
        vec![base]
    } else {
        (first..=last.saturating_sub(span).saturating_add(1))
            .step_by(usize::from(span))
            .filter(|b| b + span - 1 <= last)
            .collect()
    };
    for b in candidates {
        let (ports, all) = Ports::block(b, workers);
        let busy: Vec<u16> = all.into_iter().filter(|p| !port_free(*p)).collect();
        if busy.is_empty() {
            return Ok(ports);
        }
        if base != 0 {
            bail!("port(s) {busy:?} of the block at {base} are in use; pass another --port-base");
        }
    }
    bail!("no free port block of {span} ports in {first}-{last}; pass --port-base")
}

/// What `bed.sh` needs to know, from the flags.
#[derive(Clone, Debug)]
pub struct BedConfig {
    pub bed_script: PathBuf,
    pub mock_bin: PathBuf,
    pub load_bin: PathBuf,
    pub tokenizer: String,
    pub workers: u16,
    pub port_base: u16,
    pub gw_cores: String,
    pub mock_cores: String,
    pub gen_cores: String,
    pub gw_threads: u32,
    pub gw_args: String,
    pub gw_log_level: String,
    pub engine_args: String,
    pub ports: Option<Ports>,
}

impl BedConfig {
    pub fn new(bed_script: &Path, mock_bin: &Path, load_bin: &Path, tokenizer: &str) -> Self {
        Self {
            bed_script: bed_script.to_path_buf(),
            mock_bin: mock_bin.to_path_buf(),
            load_bin: load_bin.to_path_buf(),
            tokenizer: tokenizer.to_string(),
            workers: 4,
            port_base: 0,
            gw_cores: String::new(),
            mock_cores: String::new(),
            gen_cores: String::new(),
            gw_threads: 8,
            gw_args: DEFAULT_GW_ARGS.to_string(),
            gw_log_level: "warn".to_string(),
            engine_args: String::new(),
            ports: None,
        }
    }

    pub fn choose_ports(&mut self) -> Result<()> {
        if self.ports.is_none() {
            self.ports = Some(pick_ports(self.port_base, self.workers, 29700, 29999)?);
        }
        Ok(())
    }

    /// The chosen ports (all zero before `choose_ports`, which `bed down` is fine with).
    pub fn ports(&self) -> Ports {
        self.ports.unwrap_or_default()
    }
}

/// The script that brings a bed up and down (`bed.sh up|down`); a test swaps it.
pub trait BedScript {
    /// Runs `action` with the environment, both outputs into `log`; the exit code.
    fn run(&self, action: &str, env: &[(String, String)], log: &Path) -> io::Result<i32>;
}

/// `bash <bed.sh> <action>`.
pub struct BashScript(pub PathBuf);

impl BedScript for BashScript {
    fn run(&self, action: &str, env: &[(String, String)], log: &Path) -> io::Result<i32> {
        let out = fs::File::create(log)?;
        let err = out.try_clone()?;
        let status = Command::new("bash")
            .arg(&self.0)
            .arg(action)
            .envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .stdin(Stdio::null())
            .stdout(out)
            .stderr(err)
            .status()?;
        Ok(status.code().unwrap_or(-1))
    }
}

/// `bed/bed.json`, written by the script once the probe answered.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BedInfo {
    pub gateway_pid: u32,
    #[serde(default)]
    pub mock_pid: u32,
    pub gateway_url: String,
    pub metrics_url: String,
    pub admin_url: String,
    #[serde(default)]
    pub workers: u16,
}

/// One bed under `run_dir`.
pub struct Bed {
    pub cfg: BedConfig,
    pub gateway_bin: PathBuf,
    pub run_dir: PathBuf,
    pub jemalloc_stats: bool,
    info: Option<BedInfo>,
    script: Box<dyn BedScript>,
}

pub const JEMALLOC_ENV: &str = "_RJEM_MALLOC_CONF=stats_print:true,stats_print_opts:Jaxe";

impl Bed {
    pub fn new(cfg: &BedConfig, gateway_bin: &Path, run_dir: &Path, jemalloc_stats: bool) -> Self {
        let script = Box::new(BashScript(cfg.bed_script.clone()));
        Self::with_script(cfg, gateway_bin, run_dir, jemalloc_stats, script)
    }

    pub fn with_script(
        cfg: &BedConfig,
        gateway_bin: &Path,
        run_dir: &Path,
        jemalloc_stats: bool,
        script: Box<dyn BedScript>,
    ) -> Self {
        Self {
            cfg: cfg.clone(),
            gateway_bin: gateway_bin.to_path_buf(),
            run_dir: run_dir.to_path_buf(),
            jemalloc_stats,
            info: None,
            script,
        }
    }

    /// The script's environment: the run directory, the binaries, the ports, the
    /// pinning, the gateway's flags and the allocator's statistics print at exit.
    pub fn env(&self) -> Vec<(String, String)> {
        let cfg = &self.cfg;
        let ports = cfg.ports();
        let mut env = vec![
            ("GS_DIR", self.run_dir.display().to_string()),
            ("SMG_BIN", self.gateway_bin.display().to_string()),
            ("MOCK_BIN", cfg.mock_bin.display().to_string()),
            ("TOKENIZER", cfg.tokenizer.clone()),
            ("GW_PORT", ports.gateway.to_string()),
            ("PROM_PORT", ports.metrics.to_string()),
            ("ADMIN_PORT", ports.admin.to_string()),
            ("MOCK_BASE", ports.mock_base.to_string()),
            ("WORKERS", cfg.workers.to_string()),
            ("GW_CORES", cfg.gw_cores.clone()),
            ("MOCK_CORES", cfg.mock_cores.clone()),
            ("GW_THREADS", cfg.gw_threads.to_string()),
            ("GW_ARGS", cfg.gw_args.clone()),
            ("GW_LOG_LEVEL", cfg.gw_log_level.clone()),
            ("SCOPE", "0".to_string()),
            (
                "GW_ENV",
                if self.jemalloc_stats {
                    JEMALLOC_ENV
                } else {
                    ""
                }
                .to_string(),
            ),
        ];
        if !cfg.engine_args.is_empty() {
            env.push(("ENGINE_ARGS", cfg.engine_args.clone()));
        }
        env.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }

    /// Starts the bed. The script refuses to start over a running bed (its pid file is
    /// there); that bed is not this call's to stop, so the refusal is read here, before
    /// anything starts and before the failure path below could take it down. A start
    /// that fails after the mock fleet or the gateway came up (the gateway's health,
    /// the workers' registration, the probe), or a start whose `bed.json` cannot be read,
    /// is taken down by the bed's own pid files before the error goes out, so a caller
    /// leaves no process and no port behind.
    pub fn up(&mut self) -> Result<&BedInfo> {
        fs::create_dir_all(&self.run_dir)?;
        if self.run_dir.join("bed").join("gateway.pid").exists() {
            bail!(
                "a bed is up already under {} (take it down first: bed down)",
                self.run_dir.display()
            );
        }
        let log = self.run_dir.join("bed-up.log");
        let rc = self
            .script
            .run("up", &self.env(), &log)
            .context("running the bed script")?;
        if rc != 0 {
            let text = fs::read_to_string(&log).unwrap_or_default();
            let lines: Vec<&str> = text.lines().collect();
            let tail = lines[lines.len().saturating_sub(15)..].join("\n");
            self.down();
            bail!("bed up failed (rc {rc}); {}:\n{tail}", log.display());
        }
        let path = self.run_dir.join("bed").join("bed.json");
        let info = fs::read_to_string(&path)
            .map_err(anyhow::Error::from)
            .and_then(|text| serde_json::from_str::<BedInfo>(&text).map_err(anyhow::Error::from))
            .with_context(|| format!("reading {}", path.display()));
        let info = match info {
            Ok(info) => info,
            Err(err) => {
                // The script said up, but the bed cannot be read: a start that is not usable
                // is a failed start, and it is taken down like one.
                self.down();
                return Err(err);
            }
        };
        Ok(self.info.insert(info))
    }

    /// Stops what the script started, by its pid files.
    pub fn down(&self) {
        let log = self.run_dir.join("bed-down.log");
        let _ = self.script.run("down", &self.env(), &log);
    }

    fn info(&self) -> Result<&BedInfo> {
        self.info
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("the bed is not up"))
    }

    pub fn pid(&self) -> Result<u32> {
        Ok(self.info()?.gateway_pid)
    }

    pub fn gateway_url(&self) -> Result<&str> {
        Ok(&self.info()?.gateway_url)
    }

    pub fn metrics_url(&self) -> Result<&str> {
        Ok(&self.info()?.metrics_url)
    }

    pub fn admin_url(&self) -> Result<&str> {
        Ok(&self.info()?.admin_url)
    }

    /// The gateway's log so far (empty when there is none yet).
    pub fn gateway_log(&self) -> String {
        fs::read_to_string(self.run_dir.join("bed").join("gateway.log")).unwrap_or_default()
    }
}

pub const ENQUEUE_CHUNK: usize = 500;

/// Hands the mock engines the cell's answers: the shared queue every worker reads,
/// emptied first (an earlier window's leftovers would answer the next requests), filled
/// in chunks (ten thousand 128-token answers in one body would exceed the admin API's
/// request size).
pub fn enqueue(bed: &Bed, items: &[Value]) -> Result<()> {
    let admin = bed.admin_url()?;
    let url = format!("{admin}/admin/script/all");
    http::delete(&url, Duration::from_secs(30))?;
    for chunk in items.chunks(ENQUEUE_CHUNK) {
        http::post(
            &url,
            &Value::Array(chunk.to_vec()),
            Duration::from_secs(120),
        )?;
    }
    Ok(())
}

/// Hits and misses over the workers, what is left in the shared queue, and how many
/// of those reads failed (`unreadable`): a counter that cannot be read leaves the misses
/// unknown, which the callers warn about as they do about misses, never a pass.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScriptCounts {
    pub hits: u64,
    pub misses: u64,
    pub unreadable: u64,
    pub queued_left: Option<u64>,
}

pub fn script_counts(bed: &Bed) -> ScriptCounts {
    match bed.admin_url() {
        Ok(admin) => script_counts_at(admin, bed.cfg.workers, bed.cfg.ports().mock_base),
        Err(_) => ScriptCounts {
            unreadable: u64::from(bed.cfg.workers) + 1,
            ..ScriptCounts::default()
        },
    }
}

pub fn script_counts_at(admin_url: &str, workers: u16, mock_base: u16) -> ScriptCounts {
    let timeout = Duration::from_secs(30);
    let read = |url: String| -> Option<Value> {
        http::get(&url, timeout)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
    };
    let count = |v: &Value, key: &str| v.get(key).and_then(Value::as_u64).unwrap_or(0);
    let mut counts = ScriptCounts::default();
    for i in 0..workers {
        match read(format!("{admin_url}/admin/script/grpc:{}", mock_base + i)) {
            Some(state) => {
                counts.hits += count(&state, "hits");
                counts.misses += count(&state, "misses");
            }
            None => counts.unreadable += 1,
        }
    }
    match read(format!("{admin_url}/admin/script/all")) {
        Some(shared) => {
            counts.queued_left = shared
                .get("queued")
                .or_else(|| shared.get("queue"))
                .and_then(Value::as_u64);
        }
        None => counts.unreadable += 1,
    }
    counts
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        sync::{Arc, Mutex},
    };

    use super::*;
    use crate::http::tests::serve;

    #[test]
    fn port_block_is_free_and_laid_out_as_the_bed_expects() {
        // A block's ports, the gateway first (the refusal below reads it at the head of
        // the busy list).
        assert_eq!(
            Ports::block(29800, 2),
            (
                Ports {
                    gateway: 29800,
                    metrics: 29801,
                    admin: 29809,
                    mock_base: 29810
                },
                vec![29800, 29801, 29809, 29810, 29811]
            )
        );
        // The test holds a port of its own, the first of the bed's range it can bind: the
        // bind both finds and holds it, so no other user of the range (another instance of
        // this test included) can take what the test asserts on.
        let holder = (29700..=29999)
            .find_map(|port| TcpListener::bind(("127.0.0.1", port)).ok())
            .unwrap();
        let held = holder.local_addr().unwrap().port();
        // The scan passes the held block by and lays the first free one out as the bed
        // expects.
        let ports = pick_ports(0, 4, 29700, 29999).unwrap();
        let base = ports.gateway;
        assert_ne!(base, held);
        assert_eq!(
            ports,
            Ports {
                gateway: base,
                metrics: base + 1,
                admin: base + 9,
                mock_base: base + 10
            }
        );
        let err = pick_ports(0, 4, 29700, 29705).unwrap_err().to_string();
        assert!(err.contains("no free port block"), "{err}");
        // A block whose gateway port is taken is refused by that port's name: the held port
        // heads the error's list whatever else is busy in its block (an ephemeral port
        // would head a block the host's other connections share).
        let err = pick_ports(held, 4, 29700, 29999).unwrap_err().to_string();
        assert!(err.contains(&format!("port(s) [{held}")), "{err}");
    }

    struct FakeScript {
        calls: Arc<Mutex<Vec<String>>>,
        up_rc: i32,
    }

    impl BedScript for FakeScript {
        fn run(&self, action: &str, _env: &[(String, String)], log: &Path) -> io::Result<i32> {
            self.calls.lock().unwrap().push(action.to_string());
            if action == "up" {
                fs::write(log, "bed: the probe failed\n")?;
                return Ok(self.up_rc);
            }
            Ok(0)
        }
    }

    fn config() -> BedConfig {
        let mut cfg = BedConfig::new(Path::new("/bed.sh"), Path::new("/m"), Path::new("/l"), "/t");
        cfg.ports = Some(Ports {
            gateway: 1,
            metrics: 2,
            admin: 3,
            mock_base: 4,
        });
        cfg
    }

    #[test]
    fn a_failed_bed_up_takes_the_partial_bed_down_and_a_running_bed_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let script = Box::new(FakeScript {
            calls: Arc::clone(&calls),
            up_rc: 1,
        });
        let run_dir = dir.path().join("bed");
        let mut bed = Bed::with_script(&config(), Path::new("/h"), &run_dir, true, script);
        let err = bed.up().unwrap_err().to_string();
        assert!(err.contains("bed up failed"), "{err}");
        assert!(err.contains("the probe failed"), "{err}");
        assert_eq!(calls.lock().unwrap().as_slice(), ["up", "down"]);
        // A script that exits 0 without a readable bed.json is a failed start too: the
        // error names the file and the bed is taken down, so no process or port is left.
        calls.lock().unwrap().clear();
        let script = Box::new(FakeScript {
            calls: Arc::clone(&calls),
            up_rc: 0,
        });
        let mut bed = Bed::with_script(&config(), Path::new("/h"), &run_dir, true, script);
        let err = bed.up().unwrap_err().to_string();
        assert!(err.contains("bed.json"), "{err}");
        assert_eq!(calls.lock().unwrap().as_slice(), ["up", "down"]);
        // A bed that is up already (its pid file under the run directory) is not this
        // call's to stop: the refusal comes before the script runs, nothing is taken down.
        calls.lock().unwrap().clear();
        fs::create_dir_all(run_dir.join("bed")).unwrap();
        fs::write(run_dir.join("bed").join("gateway.pid"), "1\n").unwrap();
        let err = bed.up().unwrap_err().to_string();
        assert!(err.contains("up already"), "{err}");
        assert!(calls.lock().unwrap().is_empty());
    }

    #[test]
    fn the_environment_carries_the_ports_the_pinning_and_the_allocator_print() {
        let bed = Bed::new(&config(), Path::new("/h"), Path::new("/run"), true);
        let env: BTreeMap<String, String> = bed.env().into_iter().collect();
        assert_eq!(env["GW_PORT"], "1");
        assert_eq!(env["PROM_PORT"], "2");
        assert_eq!(env["ADMIN_PORT"], "3");
        assert_eq!(env["MOCK_BASE"], "4");
        assert_eq!(env["WORKERS"], "4");
        assert_eq!(env["SCOPE"], "0");
        assert_eq!(env["GW_ENV"], JEMALLOC_ENV);
        assert_eq!(env["GW_ARGS"], DEFAULT_GW_ARGS);
        assert!(!env.contains_key("ENGINE_ARGS"));
        let plain = Bed::new(&config(), Path::new("/h"), Path::new("/run"), false);
        let env: BTreeMap<String, String> = plain.env().into_iter().collect();
        assert_eq!(env["GW_ENV"], "");
    }

    #[test]
    fn script_counts_report_the_reads_that_failed_instead_of_zero_misses() {
        let answers = BTreeMap::from([
            (
                "GET /admin/script/grpc:40".to_string(),
                (200, r#"{"hits": 5, "misses": 1}"#.to_string()),
            ),
            (
                "GET /admin/script/all".to_string(),
                (200, r#"{"queued": 3}"#.to_string()),
            ),
        ]);
        let server = serve(answers, false);
        assert_eq!(
            script_counts_at(&server.url, 2, 40),
            ScriptCounts {
                hits: 5,
                misses: 1,
                unreadable: 1,
                queued_left: Some(3)
            }
        );
        let empty = serve(BTreeMap::new(), false);
        assert_eq!(script_counts_at(&empty.url, 2, 40).unreadable, 3);
    }
}
