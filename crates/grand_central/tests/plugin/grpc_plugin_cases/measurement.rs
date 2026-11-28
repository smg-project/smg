//! The measurement for the Numbers table.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use smg_grand_central::{
    engine::{Constraints, Scratch},
    plugins::grpc::Plugin,
};

use crate::{
    facts, pool, pool_facts, profile_of, runtime, single_engine, spec,
    support::grpc_plugin::{Behaviour, TestPlugin, ALL, PICK},
};

/// Reads `VmRSS` of this process, in KiB.
fn rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

/// The user and system CPU time of one `/proc` stat line, in clock ticks
/// (100 per second).
fn ticks(stat: &str) -> Option<u64> {
    let after = stat.rsplit(") ").next()?;
    let fields: Vec<&str> = after.split_whitespace().collect();
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    Some(utime + stime)
}

/// This thread's CPU ticks.
fn own_cpu_ticks() -> u64 {
    std::fs::read_to_string("/proc/thread-self/stat")
        .ok()
        .and_then(|s| ticks(&s))
        .unwrap_or(0)
}

/// The CPU ticks of this process's threads named `comm`, from `/proc`.
fn cpu_ticks_of(comm: &str) -> u64 {
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return 0;
    };
    tasks
        .flatten()
        .filter_map(|task| {
            let name = std::fs::read_to_string(task.path().join("comm")).ok()?;
            (name.trim() == comm).then_some(task.path().join("stat"))
        })
        .filter_map(|stat| ticks(&std::fs::read_to_string(stat).ok()?))
        .sum()
}

/// The time on the CPU of this process's threads named `comm`, in
/// nanoseconds, by the scheduler's own clock.
fn cpu_nanos_of(comm: &str) -> u64 {
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return 0;
    };
    tasks
        .flatten()
        .filter_map(|task| {
            let name = std::fs::read_to_string(task.path().join("comm")).ok()?;
            (name.trim() == comm).then_some(task.path().join("schedstat"))
        })
        .filter_map(|stat| {
            let line = std::fs::read_to_string(stat).ok()?;
            line.split_whitespace().next()?.parse::<u64>().ok()
        })
        .sum()
}

/// A measurement for the Numbers table, not a check: run in release with
/// `--ignored measure --nocapture`. The RSS of 64 open streams beside 64
/// idle threads; the round trip of one decision, each timed alone, as the
/// three stages (one frame at the open) and as the pick alone (one frame at
/// the pick), at 8, 64 and 512 candidates; and the CPU of the deciding
/// thread and of the runtime's I/O thread over 100,000 decisions at 64
/// candidates, by their clock ticks. The plugin is served in this process
/// over a Unix socket, reads each frame and answers at once: a round trip
/// is the host's two ends, the transport and that plugin.
#[test]
#[ignore = "a measurement for the Numbers table; run in release with --nocapture"]
#[expect(
    clippy::print_stdout,
    reason = "the measurement reports its rows on stdout under --nocapture"
)]
fn measure_the_round_trip_the_rss_per_stream_and_the_cpu_per_decision() {
    let runtime = runtime();
    // A plugin that keeps none of the frames it is sent, behind a host
    // that takes 512 candidates.
    let serve = |name: &str| {
        let served = TestPlugin::serve();
        served.script(|s| s.forget = true);
        let mut spec = spec(&served, name);
        spec.max_candidates = 512;
        let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
        (served, plugin)
    };
    let cx = Constraints::default();
    let mut scratch = Scratch::new();

    // 64 threads each open their own stream, after 64 that only wait: the
    // difference is the streams', both ends of each in this process.
    let (_served, plugin) = serve("streams");
    let threads = 64u64;
    let rss_of = |warm: bool| {
        let barrier = Arc::new(std::sync::Barrier::new(threads as usize + 1));
        let before = rss_kib();
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let plugin = Arc::clone(&plugin);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    if warm {
                        plugin.warm().expect("warms");
                    }
                    barrier.wait();
                    barrier.wait();
                })
            })
            .collect();
        barrier.wait();
        let grown = rss_kib().saturating_sub(before);
        barrier.wait();
        for h in handles {
            h.join().expect("joins");
        }
        grown
    };
    let (idle, open) = (rss_of(false), rss_of(true));
    println!(
        "RSS per open stream {} KiB, both ends ({threads} idle threads grew {idle} KiB, {threads} threads with a stream each {open} KiB)",
        open.saturating_sub(idle) / threads
    );

    for (binding, stages) in [("three stages", ALL), ("the pick alone", PICK)] {
        let (_served, plugin) = serve("timed");
        let engine = single_engine(profile_of(&plugin, stages, false));
        for n in [8u32, 64, 512] {
            let candidates = pool(n);
            let pool_facts = pool_facts(&candidates);
            for seed in 0..2_000 {
                engine.decide(&facts(seed), &pool_facts, &candidates, &cx, &mut scratch);
            }
            let rounds = 20_000u64;
            let mut nanos: Vec<u128> = (0..rounds)
                .map(|seed| {
                    let started = Instant::now();
                    engine.decide(&facts(seed), &pool_facts, &candidates, &cx, &mut scratch);
                    started.elapsed().as_nanos()
                })
                .collect();
            nanos.sort_unstable();
            let at = |permille: usize| {
                let rank = (nanos.len() * permille / 1_000).min(nanos.len() - 1);
                nanos[rank] as f64 / 1e3
            };
            println!(
                "{binding} at {n} candidates: round trip p50 {:.1} us, p90 {:.1} us, p99 {:.1} us, p99.9 {:.1} us ({rounds} decisions)",
                at(500),
                at(900),
                at(990),
                at(999)
            );
        }
        let stats = plugin.stats();
        assert_eq!(
            (
                stats.missed,
                stats.invalid,
                stats.declined,
                stats.cold,
                stats.refused
            ),
            (0, 0, 0, 0, 0),
            "{binding}: every frame was answered: {stats:?}"
        );
    }

    let (_served, plugin) = serve("cpu");
    let engine = single_engine(profile_of(&plugin, ALL, false));
    let candidates = pool(64);
    let pool_facts = pool_facts(&candidates);
    let decisions = 100_000u64;
    let io = "smg-policy-grpc";
    let (own0, io0) = (own_cpu_ticks(), cpu_ticks_of(io));
    let started = Instant::now();
    for seed in 0..decisions {
        engine.decide(&facts(seed), &pool_facts, &candidates, &cx, &mut scratch);
    }
    let wall = started.elapsed();
    let own_ms = (own_cpu_ticks() - own0) * 10;
    let io_ms = (cpu_ticks_of(io) - io0) * 10;
    // CPU per decision times 10,000 per second, as a share of one core.
    let share = |ms: u64| ms as f64 / decisions as f64 * 10_000.0 / 1_000.0 * 100.0;
    println!(
        "{decisions} decisions at 64 candidates: wall {:.1} us each; the deciding thread's CPU {own_ms} ms = {:.2}% of a core at 10,000 decisions/s; the I/O thread's CPU {io_ms} ms = {:.2}% of a core at 10,000 decisions/s",
        wall.as_secs_f64() * 1e6 / decisions as f64,
        share(own_ms),
        share(io_ms)
    );
    assert_eq!(
        plugin.stats().answered,
        decisions,
        "every frame was answered"
    );
    println!(
        "the witness over those {decisions} decisions: {} answered across a stall of the I/O thread",
        plugin.stats().stalls
    );

    // The witness's price: the I/O thread's time on the CPU over two seconds
    // with no frame in flight, then over one frame a plugin never answers
    // under a two-second budget: the ticker's wakes alone.
    let window = Duration::from_secs(2);
    let idle0 = cpu_nanos_of(io);
    std::thread::sleep(window);
    let idle = cpu_nanos_of(io) - idle0;
    let (silent, plugin) = serve("ticks");
    silent.behave(Behaviour::Silent);
    let engine = single_engine(profile_of(&plugin, ALL, true));
    let mut held = facts(1);
    held.budget_micros = 2_000_000;
    let flight0 = cpu_nanos_of(io);
    engine.decide(&held, &pool_facts, &candidates, &cx, &mut scratch);
    let flight = cpu_nanos_of(io) - flight0;
    println!(
        "the witness's price: the I/O thread's CPU over 2 s idle {:.1} us, over 2 s with one frame in flight {:.1} us ({:.3}% of a core)",
        idle as f64 / 1e3,
        flight as f64 / 1e3,
        flight as f64 / window.as_nanos() as f64 * 100.0
    );
}
