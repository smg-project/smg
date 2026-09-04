//! sim-loadgen: open-loop `/generate` load generator simulating production
//! ingress in front of N SMG replicas — Poisson session arrivals, paired
//! turn-1/turn-2 requests sharing prefixes, routing keys, and multimodal
//! payloads, with client-side TTFT/E2E measurement. See
//! `.claude/generate-scale-sim/01-design.md` for the workload profile.

mod args;
mod dist;
mod report;
mod session;

use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
    process::ExitCode,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use tokio::{
    sync::{mpsc, Semaphore},
    task::JoinSet,
};

use crate::{args::Args, dist::PiecewiseCdf, report::RequestRecord, session::Ctx};

/// Low anchors of the length CDFs: the token count at cumulative 0.0.
const PROMPT_LOW_TOKENS: u32 = 256;
const OUTPUT_LOW_TOKENS: u32 = 16;

const PROGRESS_INTERVAL_SECS: u64 = 10;

#[tokio::main]
async fn main() -> ExitCode {
    let cli = match Args::from_args() {
        Ok(cli) => Arc::new(cli),
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };

    if let Err(e) = fs::create_dir_all(&cli.out) {
        eprintln!("failed to create output directory {}: {e}", cli.out);
        return ExitCode::from(2);
    }
    let jsonl_path = Path::new(&cli.out).join("requests.jsonl");
    let jsonl = match fs::File::create(&jsonl_path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("failed to create {}: {e}", jsonl_path.display());
            return ExitCode::from(2);
        }
    };
    let clients = match build_clients(&cli) {
        Ok(clients) => clients,
        Err(e) => {
            eprintln!("failed to build HTTP client: {e}");
            return ExitCode::from(2);
        }
    };

    // The collector solely owns the JSONL file and the in-memory records; it
    // drains until every sender (one per Ctx handle) is gone.
    let (records_tx, mut records_rx) = mpsc::unbounded_channel::<RequestRecord>();
    let mut collector: JoinSet<Vec<RequestRecord>> = JoinSet::new();
    collector.spawn(async move {
        let mut out = BufWriter::new(jsonl);
        let mut records = Vec::new();
        let mut write_failed = false;
        while let Some(record) = records_rx.recv().await {
            if !write_failed && writeln!(out, "{}", record.to_json()).is_err() {
                eprintln!("requests.jsonl write failed; keeping records in memory only");
                write_failed = true;
            }
            records.push(record);
        }
        let _ = out.flush();
        records
    });

    let ctx = Arc::new(Ctx {
        args: cli.clone(),
        clients,
        next_client: AtomicU64::new(0),
        limiter: Arc::new(Semaphore::new(cli.max_inflight.min(Semaphore::MAX_PERMITS))),
        records: records_tx,
        prompt_cdf: PiecewiseCdf::new(PROMPT_LOW_TOKENS, &cli.prompt_cdf, cli.prompt_max),
        output_cdf: PiecewiseCdf::new(OUTPUT_LOW_TOKENS, &cli.output_cdf, cli.output_max),
        sent: AtomicU64::new(0),
        done: AtomicU64::new(0),
        errors: AtomicU64::new(0),
    });

    let run_start = Instant::now();
    let run_start_ms = session::epoch_ms();
    let mut aux: JoinSet<()> = JoinSet::new();
    {
        let ctx = ctx.clone();
        aux.spawn(progress(ctx, run_start));
    }

    // Open-loop Poisson arrivals on an ABSOLUTE schedule: each arrival time
    // is the running sum of exponential gaps, and the spawner sleeps until
    // that instant. Sleeping per-gap instead would add the timer's minimum
    // resolution to every gap and undershoot high rates by 20-30%.
    let deadline = run_start + Duration::from_secs(cli.duration_secs);
    let mut arrivals = dist::Rng::new(dist::sub_seed(cli.seed, dist::SALT_ARRIVAL));
    let mut sessions: JoinSet<()> = JoinSet::new();
    let mut spawned: u64 = 0;
    let mut next_arrival = run_start;
    loop {
        let gap = arrivals.next_exp(1.0 / cli.session_rps);
        next_arrival += Duration::from_secs_f64(gap);
        if next_arrival >= deadline {
            break;
        }
        tokio::time::sleep_until(next_arrival.into()).await;
        sessions.spawn(session::run(ctx.clone(), spawned));
        spawned += 1;
    }

    // The arrival window is closed; in-flight sessions (think times included)
    // still run to completion.
    while let Some(joined) = sessions.join_next().await {
        if let Err(e) = joined {
            eprintln!("session task panicked: {e}");
        }
    }
    aux.abort_all();
    while aux.join_next().await.is_some() {}
    // The last live sender drops here, letting the collector drain and exit.
    drop(ctx);
    let records = match collector.join_next().await {
        Some(Ok(records)) => records,
        _ => {
            eprintln!("record collector failed; summary will be empty");
            Vec::new()
        }
    };

    let elapsed_secs = run_start.elapsed().as_secs_f64();
    let summary = report::summarize(&cli, &records, run_start_ms, elapsed_secs, spawned);
    let summary_path = Path::new(&cli.out).join("summary.json");
    match serde_json::to_string_pretty(&summary) {
        Ok(text) => {
            if let Err(e) = fs::write(&summary_path, text) {
                eprintln!("failed to write {}: {e}", summary_path.display());
                return ExitCode::FAILURE;
            }
        }
        Err(e) => {
            eprintln!("failed to serialize summary: {e}");
            return ExitCode::FAILURE;
        }
    }

    let total = records.len();
    let errors = records.iter().filter(|r| !r.is_ok()).count();
    eprintln!(
        "[sim-loadgen] finished: {spawned} sessions, {total} requests, {errors} errors \
         in {elapsed_secs:.1}s; wrote {} and {}",
        jsonl_path.display(),
        summary_path.display(),
    );
    if total > 0 && errors * 2 > total {
        eprintln!("[sim-loadgen] error rate {errors}/{total} exceeds 50%");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// `--conns-per-origin` independent clients, round-robined per request. Each
/// keeps a large idle pool so the h1 mode reuses per-stream connections;
/// `--http2` multiplexes each client's streams over ONE connection per SMG,
/// so the client count bounds concurrent streams per origin — without it a
/// small gateway count throttles the generator instead of the gateway.
fn build_clients(cli: &Args) -> Result<Vec<reqwest::Client>, reqwest::Error> {
    (0..cli.conns_per_origin.max(1))
        .map(|_| {
            let mut builder = reqwest::Client::builder()
                .pool_idle_timeout(Some(Duration::from_secs(90)))
                .pool_max_idle_per_host(4096)
                // A wedged stream must fail, not hang the end-of-run drain.
                .timeout(Duration::from_secs(cli.request_timeout_secs))
                .tcp_nodelay(true);
            if cli.http2 {
                // Hundreds of concurrent multi-hundred-KB uploads share each
                // connection; the h2 defaults (64 KiB stream / 1 MiB conn
                // windows) throttle body upload long before the gateway is
                // the bottleneck and show up as server-side 408s.
                builder = builder
                    .http2_prior_knowledge()
                    .http2_adaptive_window(true)
                    .http2_initial_stream_window_size(4 * 1024 * 1024)
                    .http2_initial_connection_window_size(32 * 1024 * 1024);
            }
            builder.build()
        })
        .collect()
}

/// Progress line to stderr every 10 s with the instantaneous completion rate.
async fn progress(ctx: Arc<Ctx>, run_start: Instant) {
    let mut ticks = tokio::time::interval(Duration::from_secs(PROGRESS_INTERVAL_SECS));
    // The first tick fires immediately; skip it.
    ticks.tick().await;
    let mut last_done: u64 = 0;
    loop {
        ticks.tick().await;
        let sent = ctx.sent.load(Ordering::Relaxed);
        let done = ctx.done.load(Ordering::Relaxed);
        let errors = ctx.errors.load(Ordering::Relaxed);
        let rps = (done - last_done) as f64 / PROGRESS_INTERVAL_SECS as f64;
        eprintln!(
            "[sim-loadgen] t={:.0}s sent={sent} done={done} errors={errors} rps={rps:.1}",
            run_start.elapsed().as_secs_f64(),
        );
        last_done = done;
    }
}
