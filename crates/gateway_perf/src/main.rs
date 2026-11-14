//! `gateway-perf`: the gateway's cost per request, base against head, on the CPU mock
//! bed (see the crate's README and `lib.rs`).

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "a command-line tool reports on its standard streams"
)]

use std::{
    collections::BTreeMap,
    env, fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::ExitCode,
};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use gateway_perf::{
    base::build_base,
    bed::{Bed, BedConfig, Fleet, Ports},
    cells::{self, Results, Settings},
    counters::perf_available,
    gate,
    live::{self, Ramp, Slo},
    paths::{fleet_paths, DEFAULT_GW_ARGS},
    report::numbers_table,
    stamp, tokenizer,
};

#[derive(Parser, Debug)]
#[command(
    name = "gateway-perf",
    version,
    about = "The gateway's cost per request, base against head, on the CPU mock bed"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Instructions, cycles and CPU time per request, base vs head
    Cpu(CellCmd),
    /// Allocation calls and KiB per request, base vs head (jemalloc exit statistics; no perf)
    Alloc(CellCmd),
    /// RSS over windows on one gateway, growth per 10k requests, the peak after a burst, base vs head
    Mem(MemCmd),
    /// An open-loop ramp with the capacity rule (head; base too when given)
    Live(LiveCmd),
    /// bed up | down | smoke
    Bed(BedCmd),
    /// Print the Numbers table and the gate verdicts of a results.json
    Report(ReportCmd),
    /// Write the synthetic tokenizer directory (what the modes write under --out when no tokenizer is given)
    Tokenizer(TokenizerCmd),
    /// The base revision's gateway binary, built by the head's cargo line into its own target directory
    Base(BaseCmd),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Counters {
    Auto,
    On,
    Off,
}

/// The bed's flags (every mode that starts a bed).
#[derive(Args, Clone, Debug)]
struct BedArgs {
    /// The repository checkout: bed.sh and the default binaries live under it
    #[arg(long, default_value = ".")]
    repo: PathBuf,
    /// The gateway under test (the head) [target/release/smg]
    #[arg(long)]
    head: Option<PathBuf>,
    /// mock-worker binary [target/release/mock-worker]
    #[arg(long)]
    mock: Option<PathBuf>,
    /// stress_load binary [target/release/stress_load]
    #[arg(long)]
    load: Option<PathBuf>,
    /// Tokenizer directory or hub id [$TOKENIZER, else a synthetic one written under --out]
    #[arg(long, default_value = "")]
    tokenizer: String,
    /// Output directory [target/gateway-perf/<stamp>]
    #[arg(long)]
    out: Option<PathBuf>,
    /// Mock engines behind the gateway (the decode engines of the prefill-decode fleet), one or more
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u16).range(1..))]
    workers: u16,
    /// The fleet behind the gateway: regular gRPC engines, a prefill fleet and a decode fleet on the prefill-decode path, or regular HTTP engines
    #[arg(long, value_enum, default_value_t = Fleet::Grpc)]
    fleet: Fleet,
    /// Prefill engines of the prefill-decode fleet (one or more)
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u16).range(1..))]
    prefill_workers: u16,
    /// First port of the bed's block (gateway; +1 metrics, +9 admin, +10.. workers, then the prefill engines and their bootstrap ports) [a free block from 29700]
    #[arg(long, default_value_t = 0)]
    port_base: u16,
    /// cpu list the gateway is pinned to [$GW_CORES]
    #[arg(long, default_value = "")]
    gw_cores: String,
    /// cpu list the engines are pinned to [$MOCK_CORES]
    #[arg(long, default_value = "")]
    mock_cores: String,
    /// cpu list the client is pinned to [$GEN_CORES]
    #[arg(long, default_value = "")]
    gen_cores: String,
    /// Gateway runtime worker threads
    #[arg(long, default_value_t = 8)]
    gw_threads: u32,
    /// Extra gateway flags
    #[arg(long, default_value = DEFAULT_GW_ARGS)]
    gw_args: String,
    /// perf stat hardware counters [auto: when perf can count instructions]
    #[arg(long, value_enum, default_value_t = Counters::Auto)]
    counters: Counters,
}

/// The cell flags (`cpu`, `alloc`).
#[derive(Args, Clone, Debug)]
struct CellArgs {
    /// Comma-separated paths [every fleet's default: chat,stream,tools,json,messages]
    #[arg(long)]
    paths: Option<String>,
    /// Requests per cell
    #[arg(long, default_value_t = 10_000)]
    requests: u64,
    /// Interleaved repeats per side
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    /// Multiply every path's offered rate
    #[arg(long, default_value_t = 1.0)]
    rate_scale: f64,
    /// The idle cell's window, seconds
    #[arg(long, default_value_t = 25)]
    idle_seconds: u64,
    /// Prompt length in words, about one token each: a request-length class
    #[arg(long, default_value_t = 64)]
    prompt_tokens: u64,
    /// Distinct prompts the client cycles through
    #[arg(long, default_value_t = 64)]
    distinct: u64,
    /// A bench_gate TOML (.github/bench-gates/gateway-perf.toml) for the verdict table [report only]
    #[arg(long)]
    gates: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct CellCmd {
    #[command(flatten)]
    bed: BedArgs,
    /// The base branch's gateway binary (`gateway-perf base` builds one)
    #[arg(long)]
    base: Option<PathBuf>,
    #[command(flatten)]
    cell: CellArgs,
}

/// `mem`: the cell flags with the soak's defaults (the stream path, one repeat, 5,000
/// requests per window), plus the windows and the burst.
#[derive(Args, Debug)]
struct MemCmd {
    #[command(flatten)]
    bed: BedArgs,
    /// The base branch's gateway binary (`gateway-perf base` builds one)
    #[arg(long)]
    base: Option<PathBuf>,
    /// Comma-separated paths
    #[arg(long, default_value = "stream")]
    paths: String,
    /// Requests per window
    #[arg(long, default_value_t = 5_000)]
    requests: u64,
    /// Interleaved repeats per side
    #[arg(long, default_value_t = 1)]
    repeats: usize,
    /// Multiply every path's offered rate
    #[arg(long, default_value_t = 1.0)]
    rate_scale: f64,
    /// Prompt length in words, about one token each
    #[arg(long, default_value_t = 64)]
    prompt_tokens: u64,
    /// Distinct prompts the client cycles through
    #[arg(long, default_value_t = 64)]
    distinct: u64,
    /// A bench_gate TOML for the verdict table [report only]
    #[arg(long)]
    gates: Option<PathBuf>,
    /// Windows of --requests on one gateway
    #[arg(long, default_value_t = 5)]
    windows: usize,
    /// The burst window's rate multiplier
    #[arg(long, default_value_t = 2.0)]
    burst: f64,
}

impl MemCmd {
    fn cell_args(&self) -> CellArgs {
        CellArgs {
            paths: Some(self.paths.clone()),
            requests: self.requests,
            repeats: self.repeats,
            rate_scale: self.rate_scale,
            idle_seconds: 25,
            prompt_tokens: self.prompt_tokens,
            distinct: self.distinct,
            gates: self.gates.clone(),
        }
    }
}

#[derive(Args, Debug)]
struct LiveCmd {
    #[command(flatten)]
    bed: BedArgs,
    /// The base branch's gateway binary: the ramp runs it too
    #[arg(long)]
    base: Option<PathBuf>,
    /// The path to ramp
    #[arg(long, default_value = "chat")]
    path: String,
    /// Offered rates, req/s
    #[arg(long, default_value = "200,400,800,1200,1600,2400,3200")]
    rates: String,
    /// Sampled seconds per step
    #[arg(long, default_value_t = 20)]
    step_seconds: u64,
    /// Driven, unsampled seconds before each step
    #[arg(long, default_value_t = 3)]
    step_warmup: u64,
    #[arg(long, default_value_t = 100.0)]
    slo_ttft_p99_ms: f64,
    #[arg(long, default_value_t = 1000.0)]
    slo_e2e_p99_ms: f64,
    #[arg(long, default_value_t = 0.0)]
    slo_error_rate: f64,
    /// A step whose achieved rate falls under this share of the offered rate fails
    #[arg(long, default_value_t = 0.95)]
    slo_achieved_fraction: f64,
    #[arg(long, hide = true, default_value_t = 0)]
    requests: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum BedAction {
    Up,
    Down,
    Smoke,
}

#[derive(Args, Debug)]
struct BedCmd {
    #[arg(value_enum)]
    action: BedAction,
    #[command(flatten)]
    bed: BedArgs,
    /// smoke: the paths to exercise [the fleet's default]
    #[arg(long)]
    paths: Option<String>,
}

#[derive(Args, Debug)]
struct ReportCmd {
    /// The run directory holding results.json
    #[arg(long)]
    out: PathBuf,
    /// A bench_gate TOML for the verdicts [report only]
    #[arg(long)]
    gates: Option<PathBuf>,
    /// The setup column's text
    #[arg(long, default_value = "")]
    setup: String,
}

#[derive(Args, Debug)]
struct TokenizerCmd {
    /// The tokenizer directory to write
    out: PathBuf,
    /// A text file whose whitespace-separated words join the vocabulary (repeatable)
    #[arg(long, value_name = "FILE")]
    words: Vec<PathBuf>,
}

#[derive(Args, Debug)]
struct BaseCmd {
    /// The revision to build (a branch, a tag, a commit)
    rev: String,
    /// Where the gateway binary is copied (`<out>.commit` beside it)
    out: PathBuf,
    /// The build's own target directory [target/gateway-perf/base-target]
    #[arg(long)]
    target: Option<PathBuf>,
    /// The repository checkout
    #[arg(long, default_value = ".")]
    repo: PathBuf,
}

fn absolutize(path: &Path) -> PathBuf {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    fs::canonicalize(&joined).unwrap_or(joined)
}

fn executable(path: &Path) -> bool {
    fs::metadata(path)
        .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

fn or_env(flag: &str, name: &str) -> String {
    if flag.is_empty() {
        env::var(name).unwrap_or_default()
    } else {
        flag.to_string()
    }
}

/// The checkout root and its bed script.
fn repo_root(repo: &Path) -> Result<(PathBuf, PathBuf)> {
    let repo = absolutize(repo);
    let bed_script = repo
        .join("scripts")
        .join("bench")
        .join("grpc_stress")
        .join("bed.sh");
    if !bed_script.exists() {
        bail!(
            "no bed script at {}: pass --repo <checkout of the repository>",
            bed_script.display()
        );
    }
    Ok((repo, bed_script))
}

/// The settings every bed mode shares: the run directory (and its tokenizer), the bed,
/// the binaries checked, the counters probed.
fn settings(
    bed: &BedArgs,
    base: Option<&Path>,
    need_base: bool,
    cell: Option<&CellArgs>,
) -> Result<Settings> {
    let (repo, bed_script) = repo_root(&bed.repo)?;
    let out = bed
        .out
        .as_deref()
        .map(absolutize)
        .unwrap_or_else(|| repo.join("target").join("gateway-perf").join(stamp()));
    fs::create_dir_all(&out)?;
    let out = absolutize(&out);
    let tokenizer = {
        let flag = or_env(&bed.tokenizer, "TOKENIZER");
        if flag.is_empty() {
            let dir = out.join("tokenizer");
            tokenizer::write(&dir, &[])?;
            dir.display().to_string()
        } else {
            flag
        }
    };
    let default = |given: Option<&PathBuf>, name: &str| -> PathBuf {
        given.map_or_else(
            || repo.join("target").join("release").join(name),
            |p| absolutize(p),
        )
    };
    let mut cfg = BedConfig::new(
        &bed_script,
        &default(bed.mock.as_ref(), "mock-worker"),
        &default(bed.load.as_ref(), "stress_load"),
        &tokenizer,
    );
    cfg.workers = bed.workers;
    cfg.fleet = bed.fleet;
    cfg.prefill_workers = bed.prefill_workers;
    cfg.port_base = bed.port_base;
    cfg.gw_cores = or_env(&bed.gw_cores, "GW_CORES");
    cfg.mock_cores = or_env(&bed.mock_cores, "MOCK_CORES");
    cfg.gen_cores = or_env(&bed.gen_cores, "GEN_CORES");
    cfg.gw_threads = bed.gw_threads;
    cfg.gw_args.clone_from(&bed.gw_args);
    cfg.choose_ports()?;
    let mut binaries = BTreeMap::from([("pr".to_string(), default(bed.head.as_ref(), "smg"))]);
    if need_base {
        let Some(base) = base else {
            bail!("--base is required: the base branch's gateway binary (see README.md: make perf-base)");
        };
        binaries.insert("base".to_string(), absolutize(base));
    }
    for (side, path) in &binaries {
        if !executable(path) {
            bail!("no executable gateway for {side} at {}", path.display());
        }
    }
    for (name, path) in [("mock_bin", &cfg.mock_bin), ("load_bin", &cfg.load_bin)] {
        if !executable(path) {
            bail!(
                "no executable at {} ({name}); build with: cargo build --release -p mock-worker --bin mock-worker --bin stress_load",
                path.display()
            );
        }
    }
    let (counters, note) = match bed.counters {
        Counters::Off => (false, "counters off by request".to_string()),
        _ => perf_available(),
    };
    if bed.counters == Counters::On && !counters {
        bail!("--counters on, but {note}");
    }
    let ports = cfg.ports();
    let paths = match cell {
        Some(cell) => fleet_paths(cfg.fleet, cell.paths.as_deref())?,
        None => Vec::new(),
    };
    let mut settings = Settings::new(&out, cfg, binaries, paths);
    settings.counters = counters;
    settings.counters_note.clone_from(&note);
    if let Some(cell) = cell {
        settings.requests = cell.requests;
        settings.repeats = cell.repeats;
        settings.rate_scale = cell.rate_scale;
        settings.idle_seconds = cell.idle_seconds;
        settings.prompt_tokens = cell.prompt_tokens;
        settings.distinct = cell.distinct;
    }
    let prefill_ports = if settings.bed.fleet == Fleet::PrefillDecode {
        format!(
            ", 'prefill_admin': {}, 'prefill_base': {}, 'bootstrap_base': {}",
            ports.prefill_admin, ports.prefill_base, ports.bootstrap_base
        )
    } else {
        String::new()
    };
    println!(
        "out {}\nports {{'gateway': {}, 'metrics': {}, 'admin': {}, 'mock_base': {}{prefill_ports}}}\nfleet {}\ntokenizer {tokenizer}\ncounters: {note}",
        out.display(),
        ports.gateway,
        ports.metrics,
        ports.admin,
        ports.mock_base,
        settings.bed.engines_text()
    );
    Ok(settings)
}

fn run_bed(cmd: &BedCmd) -> Result<i32> {
    match cmd.action {
        BedAction::Down => {
            let (repo, bed_script) = repo_root(&cmd.bed.repo)?;
            let out = cmd
                .bed
                .out
                .as_deref()
                .map(absolutize)
                .context("bed down needs --out: the run directory of the bed")?;
            let mut cfg = BedConfig::new(
                &bed_script,
                &cmd.bed
                    .mock
                    .clone()
                    .unwrap_or_else(|| repo.join("target/release/mock-worker")),
                &cmd.bed
                    .load
                    .clone()
                    .unwrap_or_else(|| repo.join("target/release/stress_load")),
                &cmd.bed.tokenizer,
            );
            cfg.ports = Some(Ports::default());
            let head = cmd
                .bed
                .head
                .clone()
                .unwrap_or_else(|| repo.join("target/release/smg"));
            Bed::new(&cfg, &head, &out.join("bed"), true).down();
            println!("bed down");
            Ok(0)
        }
        BedAction::Up => {
            let settings = settings(&cmd.bed, None, false, None)?;
            let mut bed = cells::bed_for(&settings, true)?;
            let info = bed.up()?;
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
            println!(
                "stop it with: gateway-perf bed down --out {}",
                settings.out.display()
            );
            Ok(0)
        }
        BedAction::Smoke => {
            let settings = settings(&cmd.bed, None, false, None)?;
            let paths = fleet_paths(settings.bed.fleet, cmd.paths.as_deref())?;
            cells::smoke(&settings, &paths)
        }
    }
}

fn run_report(cmd: &ReportCmd) -> Result<i32> {
    let path = cmd.out.join("results.json");
    let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let results: Results =
        serde_json::from_str(&text).with_context(|| format!("reading {}", path.display()))?;
    let setup = if !cmd.setup.is_empty() {
        cmd.setup.clone()
    } else if !results.note.is_empty() {
        results.note.clone()
    } else {
        "see results.json".to_string()
    };
    let numbers = numbers_table(&results, &setup);
    let (gate_md, rc) = gate::gate_report(&results, cmd.gates.as_deref())?;
    println!("## Numbers\n\n{numbers}\n\n{gate_md}");
    Ok(rc)
}

fn run(cli: Cli) -> Result<i32> {
    match cli.command {
        Cmd::Cpu(c) => {
            let s = settings(&c.bed, c.base.as_deref(), true, Some(&c.cell))?;
            cells::run_cells(&s, "cpu", c.cell.gates.as_deref())
        }
        Cmd::Alloc(c) => {
            let mut s = settings(&c.bed, c.base.as_deref(), true, Some(&c.cell))?;
            s.counters = false;
            s.counters_note = "alloc mode: jemalloc statistics only".to_string();
            cells::run_cells(&s, "alloc", c.cell.gates.as_deref())
        }
        Cmd::Mem(m) => {
            let cell = m.cell_args();
            let s = settings(&m.bed, m.base.as_deref(), true, Some(&cell))?;
            cells::run_mem(&s, m.windows, m.burst, m.gates.as_deref())
        }
        Cmd::Live(l) => {
            let s = settings(&l.bed, l.base.as_deref(), l.base.is_some(), None)?;
            let spec = fleet_paths(s.bed.fleet, Some(&l.path))?.remove(0);
            let rates = l
                .rates
                .split(',')
                .map(str::trim)
                .filter(|r| !r.is_empty())
                .map(|r| {
                    r.parse::<f64>()
                        .with_context(|| format!("--rates: {r:?} is not a number"))
                })
                .collect::<Result<Vec<f64>>>()?;
            let ramp = Ramp {
                rates,
                step_seconds: l.step_seconds,
                step_warmup: l.step_warmup,
                slo: Slo {
                    ttft_p99_ms: l.slo_ttft_p99_ms,
                    e2e_p99_ms: l.slo_e2e_p99_ms,
                    error_rate: l.slo_error_rate,
                    achieved_fraction: l.slo_achieved_fraction,
                },
            };
            live::run_live(&s, &spec, &ramp)
        }
        Cmd::Bed(b) => run_bed(&b),
        Cmd::Report(r) => run_report(&r),
        Cmd::Tokenizer(t) => {
            let mut extra = Vec::new();
            for file in &t.words {
                let text = fs::read_to_string(file)
                    .with_context(|| format!("reading {}", file.display()))?;
                extra.extend(text.split_whitespace().map(str::to_string));
            }
            let tokens = tokenizer::write(&t.out, &extra)?;
            println!(
                "{}: {} tokens (byte-level BPE), eos <|end|>, chat template with tools",
                t.out.display(),
                tokens.len()
            );
            Ok(0)
        }
        Cmd::Base(b) => {
            let (repo, _) = repo_root(&b.repo)?;
            build_base(&repo, &b.rev, &b.out, b.target.as_deref())?;
            Ok(0)
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => ExitCode::from(u8::try_from(code).unwrap_or(2)),
        Err(err) => {
            eprintln!("gateway-perf: {err:#}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_subcommand_parses_with_defaults() {
        let cli =
            Cli::try_parse_from(["gateway-perf", "cpu", "--base", "/b", "--head", "/h"]).unwrap();
        let Cmd::Cpu(cpu) = cli.command else {
            panic!("not cpu")
        };
        assert_eq!(cpu.cell.repeats, 3);
        assert_eq!(cpu.cell.requests, 10_000);
        assert_eq!(cpu.cell.paths, None, "the fleet's default list");
        assert_eq!(cpu.bed.counters, Counters::Auto);
        assert_eq!(cpu.base.as_deref(), Some(Path::new("/b")));
        let cli = Cli::try_parse_from([
            "gateway-perf",
            "alloc",
            "--base",
            "/b",
            "--rate-scale",
            "0.5",
            "--repeats",
            "5",
        ])
        .unwrap();
        let Cmd::Alloc(alloc) = cli.command else {
            panic!("not alloc")
        };
        assert_eq!(alloc.cell.rate_scale, 0.5);
        assert_eq!(alloc.cell.repeats, 5);
        let cli = Cli::try_parse_from(["gateway-perf", "mem", "--base", "/b"]).unwrap();
        let Cmd::Mem(mem) = cli.command else {
            panic!("not mem")
        };
        assert_eq!(mem.paths, "stream");
        assert_eq!(mem.windows, 5);
        assert_eq!(mem.burst, 2.0);
        assert_eq!(mem.repeats, 1);
        assert_eq!(mem.cell_args().requests, 5_000);
        let cli = Cli::try_parse_from(["gateway-perf", "live", "--rates", "100,200"]).unwrap();
        let Cmd::Live(live) = cli.command else {
            panic!("not live")
        };
        assert_eq!(live.rates, "100,200");
        assert_eq!(live.slo_ttft_p99_ms, 100.0);
        assert!(live.base.is_none());
        let cli = Cli::try_parse_from(["gateway-perf", "bed", "smoke"]).unwrap();
        let Cmd::Bed(bed) = cli.command else {
            panic!("not bed")
        };
        assert_eq!(bed.action, BedAction::Smoke);
        assert_eq!(bed.bed.fleet, Fleet::Grpc);
        assert_eq!(bed.bed.prefill_workers, 1);
        let cli = Cli::try_parse_from([
            "gateway-perf",
            "cpu",
            "--base",
            "/b",
            "--fleet",
            "prefill-decode",
            "--prefill-workers",
            "2",
        ])
        .unwrap();
        let Cmd::Cpu(pd) = cli.command else {
            panic!("not cpu")
        };
        assert_eq!(pd.bed.fleet, Fleet::PrefillDecode);
        assert_eq!(pd.bed.prefill_workers, 2);
        assert!(Cli::try_parse_from([
            "gateway-perf",
            "cpu",
            "--base",
            "/b",
            "--fleet",
            "prefill-decode",
            "--prefill-workers",
            "0",
        ])
        .is_err());
        assert!(
            Cli::try_parse_from(["gateway-perf", "cpu", "--base", "/b", "--workers", "0"]).is_err()
        );
        let cli = Cli::try_parse_from([
            "gateway-perf",
            "bed",
            "smoke",
            "--fleet",
            "http",
            "--paths",
            "chat",
        ])
        .unwrap();
        let Cmd::Bed(http) = cli.command else {
            panic!("not bed")
        };
        assert_eq!(http.bed.fleet, Fleet::Http);
        assert_eq!(http.paths.as_deref(), Some("chat"));
        assert!(
            Cli::try_parse_from(["gateway-perf", "cpu", "--base", "/b", "--fleet", "pd"]).is_err()
        );
        let cli = Cli::try_parse_from(["gateway-perf", "report", "--out", "/tmp/x"]).unwrap();
        let Cmd::Report(report) = cli.command else {
            panic!("not report")
        };
        assert_eq!(report.out, Path::new("/tmp/x"));
        assert!(Cli::try_parse_from(["gateway-perf", "report"]).is_err());
        let cli =
            Cli::try_parse_from(["gateway-perf", "tokenizer", "/tmp/tok", "--words", "a.txt"])
                .unwrap();
        let Cmd::Tokenizer(tok) = cli.command else {
            panic!("not tokenizer")
        };
        assert_eq!(tok.words, [PathBuf::from("a.txt")]);
        let cli = Cli::try_parse_from([
            "gateway-perf",
            "base",
            "origin/main",
            "target/gateway-perf/base/smg",
        ])
        .unwrap();
        let Cmd::Base(base) = cli.command else {
            panic!("not base")
        };
        assert_eq!(base.rev, "origin/main");
    }

    #[test]
    fn a_bed_mode_needs_the_checkout_and_the_base() {
        let dir = tempfile::tempdir().unwrap();
        let bed = BedArgs {
            repo: dir.path().to_path_buf(),
            head: None,
            mock: None,
            load: None,
            tokenizer: String::new(),
            out: None,
            workers: 4,
            fleet: Fleet::Grpc,
            prefill_workers: 1,
            port_base: 0,
            gw_cores: String::new(),
            mock_cores: String::new(),
            gen_cores: String::new(),
            gw_threads: 8,
            gw_args: DEFAULT_GW_ARGS.to_string(),
            counters: Counters::Off,
        };
        let err = settings(&bed, None, true, None).unwrap_err().to_string();
        assert!(err.contains("no bed script"), "{err}");
        fs::create_dir_all(dir.path().join("scripts/bench/grpc_stress")).unwrap();
        fs::write(dir.path().join("scripts/bench/grpc_stress/bed.sh"), "").unwrap();
        let out = dir.path().join("out");
        let bed = BedArgs {
            out: Some(out.clone()),
            ..bed
        };
        let err = settings(&bed, None, true, None).unwrap_err().to_string();
        assert!(err.contains("--base is required"), "{err}");
        assert!(out.join("tokenizer").join("tokenizer.json").exists());
        let err = settings(&bed, None, false, None).unwrap_err().to_string();
        assert!(err.contains("no executable gateway for pr"), "{err}");
    }

    #[test]
    fn report_reads_a_results_file() {
        let dir = tempfile::tempdir().unwrap();
        let results = r#"{"unit": "", "note": "n", "repeats": 1, "base": {"runs": [{"chat/allocation_calls_per_request": 300.0}]}, "pr": {"runs": [{"chat/allocation_calls_per_request": 290.0}]}}"#;
        fs::write(dir.path().join("results.json"), results).unwrap();
        let rc = run_report(&ReportCmd {
            out: dir.path().to_path_buf(),
            gates: None,
            setup: String::new(),
        })
        .unwrap();
        assert_eq!(rc, 0);
        assert!(run_report(&ReportCmd {
            out: dir.path().join("nope"),
            gates: None,
            setup: String::new()
        })
        .is_err());
    }
}
