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
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{http, paths::DEFAULT_GW_ARGS};

/// Whether a loopback port can be bound right now.
pub fn port_free(port: u16) -> bool {
    TcpListener::bind(("127.0.0.1", port)).is_ok()
}

/// The fleet behind the gateway: one fleet of regular gRPC engines (the bed as it
/// was); a prefill fleet and a decode fleet registered with their roles, so the
/// gateway serves the model on its prefill-decode path (the prefill admission queue,
/// the pair selection, the decode rooms); or one fleet of regular engines on the
/// mock's HTTP wire, so the gateway serves it on its HTTP regular path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Fleet {
    #[default]
    Grpc,
    PrefillDecode,
    Http,
}

impl Fleet {
    /// The value `bed.sh` reads in `FLEET`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Grpc => "grpc",
            Self::PrefillDecode => "prefill-decode",
            Self::Http => "http",
        }
    }

    /// The wire the scripted engines serve, which the mock's script API keys them by
    /// (`grpc:<port>`, `http:<port>`).
    pub fn wire(self) -> &'static str {
        match self {
            Self::Grpc | Self::PrefillDecode => "grpc",
            Self::Http => "http",
        }
    }

    /// Whether the fleet's mock engines answer a path in the shape it measures: the
    /// HTTP wire answers `/generate` in the chat shape, so the generate paths read a
    /// gRPC fleet only.
    pub fn serves(self, path: &str) -> bool {
        match self {
            Self::Grpc | Self::PrefillDecode => true,
            Self::Http => !path.starts_with("generate"),
        }
    }

    /// The paths a cell run drives when none are named: the same five on
    /// every fleet, the HTTP fleet included since the mock's HTTP wire serves
    /// the Messages route.
    pub fn default_paths(self) -> &'static str {
        match self {
            Self::Grpc | Self::PrefillDecode | Self::Http => crate::paths::DEFAULT_PATHS,
        }
    }
}

/// The bed's ports: the gateway, its metrics, the mock admin API and the first engine;
/// on the prefill-decode fleet also the prefill fleet's admin API, its first engine and
/// its first bootstrap port (zero on the gRPC fleet).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ports {
    pub gateway: u16,
    pub metrics: u16,
    pub admin: u16,
    pub mock_base: u16,
    #[serde(default)]
    pub prefill_admin: u16,
    #[serde(default)]
    pub prefill_base: u16,
    #[serde(default)]
    pub bootstrap_base: u16,
}

impl Ports {
    /// The block at `base`: the gateway, its metrics (+1), the prefill fleet's admin API
    /// (+8, when there are prefill engines), the decode or regular fleet's admin API
    /// (+9), its `workers` engines from +10, then the `prefill` engines and their
    /// bootstrap ports; and every port of it, the gateway first.
    fn block(base: u16, workers: u16, prefill: u16) -> (Self, Vec<u16>) {
        let mock_base = base + 10;
        let prefill_base = mock_base + workers;
        let bootstrap_base = prefill_base + prefill;
        let ports = Self {
            gateway: base,
            metrics: base + 1,
            admin: base + 9,
            mock_base,
            prefill_admin: if prefill > 0 { base + 8 } else { 0 },
            prefill_base: if prefill > 0 { prefill_base } else { 0 },
            bootstrap_base: if prefill > 0 { bootstrap_base } else { 0 },
        };
        let mut all = vec![ports.gateway, ports.metrics];
        if prefill > 0 {
            all.push(ports.prefill_admin);
        }
        all.push(ports.admin);
        all.extend((0..workers).map(|i| mock_base + i));
        all.extend((0..prefill).map(|i| prefill_base + i));
        all.extend((0..prefill).map(|i| bootstrap_base + i));
        (ports, all)
    }
}

/// The ports a block spans: ten ahead of the engines, one per engine, two per prefill
/// engine (its port and its bootstrap port).
pub fn block_span(workers: u16, prefill: u16) -> u16 {
    10 + workers + 2 * prefill
}

/// The bed's ports: `base` when given (gateway base, metrics base+1, admin base+9,
/// workers from base+10, the prefill engines and their bootstrap ports after them),
/// else the first free block of [`block_span`] ports in `[first, last]` (bed.sh refuses
/// a port another process holds, so the choice is checked again at start).
pub fn pick_ports(base: u16, workers: u16, prefill: u16, first: u16, last: u16) -> Result<Ports> {
    let span = block_span(workers, prefill);
    let candidates: Vec<u16> = if base != 0 {
        vec![base]
    } else {
        (first..=last.saturating_sub(span).saturating_add(1))
            .step_by(usize::from(span))
            .filter(|b| b + span - 1 <= last)
            .collect()
    };
    for b in candidates {
        let (ports, all) = Ports::block(b, workers, prefill);
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
    /// The fleet behind the gateway; `workers` counts its decode engines on the
    /// prefill-decode fleet, `prefill_workers` its prefill engines.
    pub fleet: Fleet,
    pub prefill_workers: u16,
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
            fleet: Fleet::Grpc,
            prefill_workers: 1,
        }
    }

    /// The prefill engines the port block and the script start: none on the regular
    /// fleets.
    pub fn prefill_engines(&self) -> u16 {
        match self.fleet {
            Fleet::Grpc | Fleet::Http => 0,
            Fleet::PrefillDecode => self.prefill_workers,
        }
    }

    /// The engines as the setup line names them.
    pub fn engines_text(&self) -> String {
        match self.fleet {
            Fleet::Grpc => format!("{} scripted mock engines", self.workers),
            Fleet::PrefillDecode => format!(
                "{} prefill and {} decode scripted mock engines on the prefill-decode path",
                self.prefill_workers, self.workers
            ),
            Fleet::Http => format!("{} scripted mock engines on the HTTP wire", self.workers),
        }
    }

    pub fn choose_ports(&mut self) -> Result<()> {
        if self.ports.is_none() {
            self.ports = Some(pick_ports(
                self.port_base,
                self.workers,
                self.prefill_engines(),
                29700,
                29999,
            )?);
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

/// `bed/bed.json`, written by the script once the probe answered. `admin_url` and
/// `workers` are the fleet that serves the scripted answers (the decode fleet on the
/// prefill-decode path); the prefill fleet's process, engines and admin API are beside
/// them there, and absent on the gRPC fleet.
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
    #[serde(default)]
    pub fleet: String,
    #[serde(default)]
    pub prefill_pid: u32,
    #[serde(default)]
    pub prefill_workers: u16,
    #[serde(default)]
    pub prefill_admin_url: String,
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

    /// The script's environment: the run directory, the binaries, the fleet and its
    /// ports, the pinning, the gateway's flags and the allocator's statistics print at
    /// exit.
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
            ("FLEET", cfg.fleet.as_str().to_string()),
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
        if cfg.fleet == Fleet::PrefillDecode {
            env.push(("PREFILL_WORKERS", cfg.prefill_workers.to_string()));
            env.push(("PREFILL_BASE", ports.prefill_base.to_string()));
            env.push(("BOOTSTRAP_BASE", ports.bootstrap_base.to_string()));
            env.push(("PREFILL_ADMIN_PORT", ports.prefill_admin.to_string()));
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

/// One enqueue body carries at most this many answers and about `ENQUEUE_BYTES` of
/// them: the admin API reads a bounded body (two mebibytes), and five hundred copies
/// of a two-thousand-word answer are several times that.
pub const ENQUEUE_CHUNK: usize = 500;
pub const ENQUEUE_BYTES: usize = 1 << 20;

/// The answers cut into the bodies `enqueue` posts: each under both bounds, one
/// answer at least however long it is.
pub fn enqueue_chunks(items: &[Value]) -> Vec<&[Value]> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut bytes = 0;
    for (i, item) in items.iter().enumerate() {
        let size = item.to_string().len() + 1;
        if i > start && (i - start >= ENQUEUE_CHUNK || bytes + size > ENQUEUE_BYTES) {
            chunks.push(&items[start..i]);
            start = i;
            bytes = 0;
        }
        bytes += size;
    }
    if start < items.len() {
        chunks.push(&items[start..]);
    }
    chunks
}

/// Hands the mock engines the cell's answers: the shared queue every worker reads,
/// emptied first (an earlier window's leftovers would answer the next requests), filled
/// in bounded bodies (see [`enqueue_chunks`]).
pub fn enqueue(bed: &Bed, items: &[Value]) -> Result<()> {
    let admin = bed.admin_url()?;
    let url = format!("{admin}/admin/script/all");
    http::delete(&url, Duration::from_secs(30))?;
    for chunk in enqueue_chunks(items) {
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
        Ok(admin) => script_counts_at(
            admin,
            bed.cfg.workers,
            bed.cfg.ports().mock_base,
            bed.cfg.fleet.wire(),
        ),
        Err(_) => ScriptCounts {
            unreadable: u64::from(bed.cfg.workers) + 1,
            ..ScriptCounts::default()
        },
    }
}

/// The counts over the `workers` engines from `mock_base` on the `wire` (the script
/// API keys an engine `<wire>:<port>`), and the shared queue.
pub fn script_counts_at(admin_url: &str, workers: u16, mock_base: u16, wire: &str) -> ScriptCounts {
    let timeout = Duration::from_secs(30);
    let read = |url: String| -> Option<Value> {
        http::get(&url, timeout)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
    };
    let count = |v: &Value, key: &str| v.get(key).and_then(Value::as_u64).unwrap_or(0);
    let mut counts = ScriptCounts::default();
    for i in 0..workers {
        match read(format!("{admin_url}/admin/script/{wire}:{}", mock_base + i)) {
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
            Ports::block(29800, 2, 0),
            (
                Ports {
                    gateway: 29800,
                    metrics: 29801,
                    admin: 29809,
                    mock_base: 29810,
                    ..Ports::default()
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
        let ports = pick_ports(0, 4, 0, 29700, 29999).unwrap();
        let base = ports.gateway;
        assert_ne!(base, held);
        assert_eq!(
            ports,
            Ports {
                gateway: base,
                metrics: base + 1,
                admin: base + 9,
                mock_base: base + 10,
                ..Ports::default()
            }
        );
        let err = pick_ports(0, 4, 0, 29700, 29705).unwrap_err().to_string();
        assert!(err.contains("no free port block"), "{err}");
        // A block whose gateway port is taken is refused by that port's name: the held port
        // heads the error's list whatever else is busy in its block (an ephemeral port
        // would head a block the host's other connections share).
        let err = pick_ports(held, 4, 0, 29700, 29999)
            .unwrap_err()
            .to_string();
        assert!(err.contains(&format!("port(s) [{held}")), "{err}");
    }

    #[test]
    fn choose_ports_hands_the_scan_the_given_base_and_a_held_base_is_refused() {
        // A base the test holds itself: the refusal is certain, and it is the scan's
        // answer for the given base, not a block of the scan's own choice (a success on
        // a fixed base is the probe that went red whenever another process held a port).
        let holder = (29700..=29999)
            .find_map(|port| TcpListener::bind(("127.0.0.1", port)).ok())
            .unwrap();
        let held = holder.local_addr().unwrap().port();
        let mut cfg = config();
        cfg.ports = None;
        cfg.port_base = held;
        let err = cfg.choose_ports().unwrap_err().to_string();
        assert!(err.contains(&format!("port(s) [{held}")), "{err}");
        assert_eq!(cfg.ports(), Ports::default());
    }

    #[test]
    fn the_prefill_decode_block_adds_the_second_admin_the_prefill_engines_and_their_bootstrap_ports(
    ) {
        // The decode engines keep the regular fleet's places (+9 admin, +10..), so a
        // script count or a cell that reads them is the same on both fleets; the prefill
        // fleet's admin API takes +8 and its engines and bootstrap ports follow the
        // decode engines.
        let (ports, all) = Ports::block(29800, 4, 1);
        assert_eq!(
            ports,
            Ports {
                gateway: 29800,
                metrics: 29801,
                admin: 29809,
                mock_base: 29810,
                prefill_admin: 29808,
                prefill_base: 29814,
                bootstrap_base: 29815,
            }
        );
        assert_eq!(
            all,
            vec![29800, 29801, 29808, 29809, 29810, 29811, 29812, 29813, 29814, 29815]
        );
        assert_eq!(block_span(4, 1), 16);
        assert_eq!(block_span(4, 0), 14);
        // The span reserves the block (+2..+7 stay free, as on the gRPC fleet); the ports
        // bound are the two admin APIs, the gateway, its metrics and the engines' own.
        let (two, all) = Ports::block(29900, 2, 2);
        assert_eq!((two.prefill_base, two.bootstrap_base), (29912, 29914));
        assert_eq!(all.len(), 4 + 2 + 2 * 2);
        assert!(all.iter().all(|p| *p < 29900 + block_span(2, 2)));
        // A scan for a prefill-decode block steps by its own span.
        let ports = pick_ports(0, 4, 1, 29700, 29999).unwrap();
        assert_eq!(ports.prefill_admin, ports.gateway + 8);
        assert_eq!(ports.bootstrap_base, ports.gateway + 15);
        assert!(pick_ports(0, 4, 1, 29700, 29714).is_err());
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
            ..Ports::default()
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
        assert_eq!(env["FLEET"], "grpc");
        assert_eq!(env["SCOPE"], "0");
        assert_eq!(env["GW_ENV"], JEMALLOC_ENV);
        assert_eq!(env["GW_ARGS"], DEFAULT_GW_ARGS);
        assert!(!env.contains_key("ENGINE_ARGS"));
        for key in [
            "PREFILL_WORKERS",
            "PREFILL_BASE",
            "BOOTSTRAP_BASE",
            "PREFILL_ADMIN_PORT",
        ] {
            assert!(
                !env.contains_key(key),
                "{key} belongs to the prefill-decode fleet alone"
            );
        }
        let plain = Bed::new(&config(), Path::new("/h"), Path::new("/run"), false);
        let env: BTreeMap<String, String> = plain.env().into_iter().collect();
        assert_eq!(env["GW_ENV"], "");
    }

    #[test]
    fn the_prefill_decode_fleet_hands_the_script_its_second_fleet_and_its_ports() {
        let mut cfg = config();
        cfg.fleet = Fleet::PrefillDecode;
        cfg.prefill_workers = 2;
        assert_eq!(cfg.prefill_engines(), 2);
        // The block's layout, not a probe of the host's ports: the fixed-base probe went
        // red whenever another process held a port of the block.
        cfg.ports = Some(Ports::block(29950, cfg.workers, cfg.prefill_engines()).0);
        let bed = Bed::new(&cfg, Path::new("/h"), Path::new("/run"), true);
        let env: BTreeMap<String, String> = bed.env().into_iter().collect();
        assert_eq!(env["FLEET"], "prefill-decode");
        assert_eq!(env["WORKERS"], "4");
        assert_eq!(env["MOCK_BASE"], "29960");
        assert_eq!(env["ADMIN_PORT"], "29959");
        assert_eq!(env["PREFILL_WORKERS"], "2");
        assert_eq!(env["PREFILL_BASE"], "29964");
        assert_eq!(env["BOOTSTRAP_BASE"], "29966");
        assert_eq!(env["PREFILL_ADMIN_PORT"], "29958");
        assert_eq!(
            cfg.engines_text(),
            "2 prefill and 4 decode scripted mock engines on the prefill-decode path"
        );
        let grpc = config();
        assert_eq!(grpc.prefill_engines(), 0);
        assert_eq!(grpc.engines_text(), "4 scripted mock engines");
        assert_eq!(Fleet::Grpc.as_str(), "grpc");
    }

    #[test]
    fn choose_ports_scans_for_a_block_that_holds_the_prefill_engines_too() {
        // No ports and no base: the scan, handed the config's own counts. A block laid
        // out for the decode engines alone leaves the prefill bases at zero, and only
        // the script would notice at start.
        let mut cfg = config();
        cfg.fleet = Fleet::PrefillDecode;
        cfg.prefill_workers = 2;
        cfg.ports = None;
        cfg.choose_ports().unwrap();
        let ports = cfg.ports();
        assert!((29700..=29999).contains(&ports.gateway), "{ports:?}");
        assert_eq!(ports.prefill_admin, ports.gateway + 8);
        assert_eq!(ports.mock_base, ports.gateway + 10);
        assert_eq!(ports.prefill_base, ports.mock_base + cfg.workers);
        assert_eq!(
            ports.bootstrap_base,
            ports.prefill_base + cfg.prefill_engines()
        );
        // The gRPC fleet's scan reserves no prefill ports.
        let mut grpc = config();
        grpc.ports = None;
        grpc.choose_ports().unwrap();
        let ports = grpc.ports();
        assert_eq!(ports.mock_base, ports.gateway + 10);
        assert_eq!(
            (
                ports.prefill_admin,
                ports.prefill_base,
                ports.bootstrap_base
            ),
            (0, 0, 0)
        );
    }

    #[test]
    fn the_http_fleet_keeps_the_regular_block_and_keys_its_engines_by_the_http_wire() {
        let mut cfg = config();
        cfg.fleet = Fleet::Http;
        assert_eq!(cfg.prefill_engines(), 0);
        assert_eq!(
            cfg.engines_text(),
            "4 scripted mock engines on the HTTP wire"
        );
        assert_eq!((Fleet::Http.as_str(), Fleet::Http.wire()), ("http", "http"));
        for path in ["generate_stream", "generate_stream_spaced_long"] {
            assert!(!Fleet::Http.serves(path), "{path}");
            assert!(Fleet::PrefillDecode.serves(path), "{path}");
        }
        for path in [
            "chat",
            "stream",
            "tools",
            "json",
            "completion_stream",
            "stream_logprobs",
            "messages",
            "messages_stream",
            "messages_stream_spaced",
        ] {
            assert!(Fleet::Http.serves(path), "{path}");
        }
        assert_eq!(Fleet::PrefillDecode.wire(), "grpc");
        assert!(Fleet::Http.serves("chat") && Fleet::Http.serves("completion_stream"));
        assert!(Fleet::Http.serves("messages") && !Fleet::Http.serves("generate_stream_spaced"));
        assert!(Fleet::Grpc.serves("messages") && Fleet::PrefillDecode.serves("messages_stream"));
        assert_eq!(Fleet::Http.default_paths(), crate::paths::DEFAULT_PATHS);
        assert_eq!(Fleet::Grpc.default_paths(), crate::paths::DEFAULT_PATHS);
        assert_eq!(
            Fleet::PrefillDecode.default_paths(),
            crate::paths::DEFAULT_PATHS
        );
        let bed = Bed::new(&cfg, Path::new("/h"), Path::new("/run"), true);
        let env: BTreeMap<String, String> = bed.env().into_iter().collect();
        assert_eq!(env["FLEET"], "http");
        assert_eq!(env["MOCK_BASE"], "4");
        assert!(!env.contains_key("PREFILL_WORKERS"));
    }

    #[test]
    fn bed_json_reads_with_and_without_the_prefill_fleet() {
        let grpc: BedInfo = serde_json::from_str(
            r#"{"gateway_pid": 7, "mock_pid": 8, "gateway_url": "http://127.0.0.1:1", "metrics_url": "m", "admin_url": "a", "workers": 4}"#,
        )
        .unwrap();
        assert_eq!((grpc.prefill_pid, grpc.prefill_workers), (0, 0));
        assert!(grpc.fleet.is_empty() && grpc.prefill_admin_url.is_empty());
        let pd: BedInfo = serde_json::from_str(
            r#"{"gateway_pid": 7, "mock_pid": 8, "gateway_url": "http://127.0.0.1:1", "metrics_url": "m", "admin_url": "a", "workers": 4,
                "fleet": "prefill-decode", "prefill_pid": 9, "prefill_workers": 1, "prefill_admin_url": "p"}"#,
        )
        .unwrap();
        assert_eq!(pd.fleet, "prefill-decode");
        assert_eq!((pd.prefill_pid, pd.prefill_workers), (9, 1));
        assert_eq!(pd.prefill_admin_url, "p");
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
            script_counts_at(&server.url, 2, 40, "grpc"),
            ScriptCounts {
                hits: 5,
                misses: 1,
                unreadable: 1,
                queued_left: Some(3)
            }
        );
        // The HTTP fleet's engines are keyed by their wire: the gRPC key reads nothing.
        assert_eq!(script_counts_at(&server.url, 2, 40, "http").unreadable, 2);
        let http_answers = BTreeMap::from([
            (
                "GET /admin/script/http:40".to_string(),
                (200, r#"{"hits": 7, "misses": 0}"#.to_string()),
            ),
            (
                "GET /admin/script/all".to_string(),
                (200, r#"{"queued": 0}"#.to_string()),
            ),
        ]);
        let http_server = serve(http_answers, false);
        let counts = script_counts_at(&http_server.url, 1, 40, "http");
        assert_eq!((counts.hits, counts.misses, counts.unreadable), (7, 0, 0));
        let empty = serve(BTreeMap::new(), false);
        assert_eq!(script_counts_at(&empty.url, 2, 40, "grpc").unreadable, 3);
    }

    #[test]
    fn enqueue_bodies_are_bounded_by_count_and_by_bytes() {
        let small: Vec<Value> = (0..1200)
            .map(|_| serde_json::json!({"output_text": "a"}))
            .collect();
        let sizes: Vec<usize> = enqueue_chunks(&small).iter().map(|c| c.len()).collect();
        assert_eq!(sizes, [500, 500, 200]);

        let long: Vec<Value> = (0..300)
            .map(|_| serde_json::json!({"output_text": "w".repeat(12_000)}))
            .collect();
        let chunks = enqueue_chunks(&long);
        assert!(chunks.len() > 1, "{} chunk(s)", chunks.len());
        assert_eq!(chunks.iter().map(|c| c.len()).sum::<usize>(), 300);
        for chunk in &chunks {
            let bytes: usize = chunk.iter().map(|v| v.to_string().len() + 1).sum();
            assert!(bytes <= ENQUEUE_BYTES, "{bytes} bytes in one body");
        }

        let one: Vec<Value> = vec![serde_json::json!({"output_text": "w".repeat(3_000_000)})];
        assert_eq!(enqueue_chunks(&one).len(), 1);
        assert!(enqueue_chunks(&[]).is_empty());
    }
}
