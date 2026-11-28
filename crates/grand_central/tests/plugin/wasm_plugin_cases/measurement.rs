//! The measurement for the Numbers table.

use std::sync::Arc;

use smg_grand_central::engine::{Constraints, PoolFacts, Scratch};

use crate::{
    facts, load, plugin_engine, pool, runtime,
    support::wasm_guest::{self as guest, mode},
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

/// This thread's user and system CPU time, in clock ticks.
fn cpu_ticks() -> u64 {
    std::fs::read_to_string("/proc/thread-self/stat")
        .ok()
        .and_then(|s| {
            let after = s.rsplit(") ").next()?;
            let fields: Vec<&str> = after.split_whitespace().collect();
            let utime: u64 = fields.get(11)?.parse().ok()?;
            let stime: u64 = fields.get(12)?.parse().ok()?;
            Some(utime + stime)
        })
        .unwrap_or(0)
}

/// A measurement for the Numbers table, not a check: run in release with
/// `--ignored measure --nocapture`. RSS per kept instance over 64 warmed
/// threads (their stacks included), and the CPU of one deciding thread over
/// 200,000 decisions at 64 candidates, read from the thread's own clock
/// ticks.
#[test]
#[ignore = "a measurement for the Numbers table; run in release with --nocapture"]
#[expect(
    clippy::print_stdout,
    reason = "the measurement reports its rows on stdout under --nocapture"
)]
fn measure_rss_per_kept_instance_and_cpu_per_decision() {
    let runtime = runtime();
    for (name, component) in [
        ("record", guest::record_guest()),
        ("packed", guest::packed_guest()),
    ] {
        let plugin = load(&runtime, name, component, mode::LEAST_IN_FLIGHT);
        let threads = 64u64;
        let barrier = Arc::new(std::sync::Barrier::new(threads as usize + 1));
        let before = rss_kib();
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let plugin = Arc::clone(&plugin);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let warmed = plugin.warm();
                    barrier.wait();
                    barrier.wait();
                    warmed
                })
            })
            .collect();
        barrier.wait();
        let after = rss_kib();
        println!(
            "{name}: RSS per kept instance {} KiB ({threads} instances: {before} -> {after} KiB)",
            after.saturating_sub(before) / threads
        );
        barrier.wait();
        for h in handles {
            h.join().expect("joins").expect("warms");
        }
        let engine = plugin_engine(&plugin);
        let candidates = pool(64);
        let mut pool_facts = PoolFacts::default();
        pool_facts.pool_size = 64;
        pool_facts.generation = 1;
        let cx = Constraints::default();
        let mut scratch = Scratch::new();
        let decisions = 200_000u64;
        let ticks0 = cpu_ticks();
        let started = std::time::Instant::now();
        for seed in 0..decisions {
            engine.decide(&facts(seed), &pool_facts, &candidates, &cx, &mut scratch);
        }
        let wall = started.elapsed();
        let cpu_ms = (cpu_ticks() - ticks0) * 10;
        let per_decision_us = wall.as_secs_f64() * 1e6 / decisions as f64;
        // CPU per decision times 10,000 per second, as a share of one core.
        let core_share = cpu_ms as f64 / decisions as f64 * 10_000.0 / 1_000.0 * 100.0;
        println!(
            "{name}: {decisions} decisions at 64 candidates: wall {per_decision_us:.2} us each, CPU {cpu_ms} ms in all = {core_share:.2}% of a core at 10,000 decisions/s"
        );
        let stats = plugin.stats();
        println!(
            "{name}: fuel per decision {} (four calls), fuel per microsecond of wall {:.0}",
            stats.fuel_used / decisions,
            stats.fuel_used as f64 / (wall.as_secs_f64() * 1e6)
        );
    }
    // The epoch ticker with no call in flight: its CPU over two idle seconds.
    let idle = std::time::Duration::from_secs(2);
    let ticks0 = ticker_cpu_ticks();
    std::thread::sleep(idle);
    let ticker_ms = (ticker_cpu_ticks() - ticks0) * 10;
    println!(
        "epoch ticker idle: {ticker_ms} ms CPU over {} ms = {:.3}% of a core",
        idle.as_millis(),
        ticker_ms as f64 / idle.as_millis() as f64 * 100.0
    );
}

/// The CPU ticks (user and system, at 100 per second) of the runtime's epoch
/// ticker thread, from `/proc`.
fn ticker_cpu_ticks() -> u64 {
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return 0;
    };
    tasks
        .flatten()
        .filter_map(|task| {
            let comm = std::fs::read_to_string(task.path().join("comm")).ok()?;
            (comm.trim() == "wasm-epoch").then_some(task.path().join("stat"))
        })
        .filter_map(|stat| {
            let s = std::fs::read_to_string(stat).ok()?;
            let after = s.rsplit(") ").next()?;
            let fields: Vec<&str> = after.split_whitespace().collect();
            let utime: u64 = fields.get(11)?.parse().ok()?;
            let stime: u64 = fields.get(12)?.parse().ok()?;
            Some(utime + stime)
        })
        .sum()
}
