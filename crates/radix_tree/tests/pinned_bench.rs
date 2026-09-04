//! The pinned performance workload, oracle baseline.
//!
//! Normative constants live HERE; gates pass on this configuration
//! only. R0 runs it against the oracle plus the engine's replicated
//! glue (per-holder reverse map, last-chain Vec, id maps) to record
//! the baseline; R1 adds the RadixTree side under the same driver.
//!
//! Run (numbers are meaningless in debug):
//!   cargo test -p radix-tree --release --test pinned_bench \
//!     -- --ignored --nocapture
//!
//! Protocol (§11): RSS sampled after fill, before query-phase
//! allocations; no asserts or allocation inside timed regions;
//! quote the median of >=3 runs.

mod common;

use std::time::Instant;

use common::{oracle::Oracle, Op, Rng};
use radix_tree::{Config, FlatTree, HolderId, OverlapScratch, RadixTree};

// ---- §11 normative constants (default scale) ----
// RADIX_BENCH_SCALE=large runs 8x blocks / 8x holders (~20 GB peak,
// ~60% of the 1.7e8 production target) to expose growth
// nonlinearities the normative scale can't: hash-table growth stalls,
// TLB pressure on a ~17 GB resident structure, query latency vs
// table size. Gates are quoted at the DEFAULT scale; large-scale
// runs are diagnostics.
fn target_blocks() -> u64 {
    match std::env::var("RADIX_BENCH_SCALE").as_deref() {
        Ok("large") => 100_000_000,
        _ => 10_000_000,
    }
}
fn holders() -> usize {
    match std::env::var("RADIX_BENCH_SCALE").as_deref() {
        Ok("large") => 2048,
        _ => profile().holders_default,
    }
}
const BATCH: usize = 8;
const MISS_QUERY_PCT: u64 = 20;
/// The gate cell: depth 78 with 64 candidate holders.
const GATE_DEPTH: u32 = 78;

/// One workload stream shape. `pinned` is the §11 NORMATIVE config —
/// gates are quoted on it and its parameters must never drift. The
/// other profiles answer "does the win hold on different input
/// distributions?" (audit follow-up: one stream shape proves one
/// point on the workload space): RADIX_BENCH_PROFILE=agentic|churn|
/// fleet selects them; they are diagnostics, not gates.
struct Profile {
    name: &'static str,
    holders_default: usize,
    /// (sharing factor H, share of total holder-blocks in percent).
    sharing_mix: [(usize, u64); 3],
    /// Log-uniform shared-prefix length range.
    shared_depth: (u32, u32),
    /// Uniform divergent-tail length range.
    tail_len: (u32, u32),
    duplicate_pct: u64,
    gap_pct: u64,
}

/// The §11 constants, verbatim.
const PINNED: Profile = Profile {
    name: "pinned",
    holders_default: 256,
    sharing_mix: [(1, 50), (8, 35), (64, 15)],
    shared_depth: (8, 512),
    tail_len: (4, 64),
    duplicate_pct: 5,
    gap_pct: 2,
};
/// Long-context / agentic traffic: deep shared prefixes (system
/// prompts, long conversations), heavy cross-worker sharing.
const AGENTIC: Profile = Profile {
    name: "agentic",
    holders_default: 256,
    sharing_mix: [(1, 20), (8, 40), (64, 40)],
    shared_depth: (64, 2048),
    tail_len: (16, 256),
    duplicate_pct: 5,
    gap_pct: 2,
};
/// Short-prompt chat with aggressive eviction churn: shallow chains,
/// little sharing, lots of duplicate re-publish and mid-chain gaps.
const CHURN: Profile = Profile {
    name: "churn",
    holders_default: 256,
    sharing_mix: [(1, 70), (8, 25), (64, 5)],
    shared_depth: (4, 64),
    tail_len: (2, 16),
    duplicate_pct: 20,
    gap_pct: 10,
};
/// Wide fleet at the default block budget: 4x the workers, each
/// holding proportionally less — stresses holder-set interning and
/// span fan-out rather than chain depth.
const FLEET: Profile = Profile {
    name: "fleet",
    holders_default: 1024,
    sharing_mix: [(1, 40), (8, 30), (64, 30)],
    shared_depth: (8, 512),
    tail_len: (4, 64),
    duplicate_pct: 5,
    gap_pct: 2,
};

fn profile() -> &'static Profile {
    match std::env::var("RADIX_BENCH_PROFILE").as_deref() {
        Ok("agentic") => &AGENTIC,
        Ok("churn") => &CHURN,
        Ok("fleet") => &FLEET,
        Ok(other) if other != "pinned" => panic!("unknown RADIX_BENCH_PROFILE {other:?}"),
        _ => &PINNED,
    }
}

fn log_uniform(rng: &mut Rng, lo: u32, hi: u32) -> u32 {
    let (llo, lhi) = ((lo as f64).ln(), (hi as f64).ln());
    let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
    (llo + u * (lhi - llo)).exp().round() as u32
}

struct Family {
    blocks: Vec<(u64, u64)>,
    holders: Vec<usize>,
}

fn build_families(rng: &mut Rng) -> Vec<Family> {
    let mut families = Vec::new();
    let next_content = |rng: &mut Rng| rng.next() | 1;
    // One forced gate family: H=64, shared length >= GATE_DEPTH.
    let mut budgets: Vec<(usize, u64)> = profile()
        .sharing_mix
        .iter()
        .map(|&(h, pct)| (h, target_blocks() * pct / 100))
        .collect();
    let mut force_gate = true;
    for (h, budget) in budgets.iter_mut() {
        let mut used = 0u64;
        while used < *budget {
            let shared_len = if force_gate && *h == 64 {
                force_gate = false;
                96
            } else {
                log_uniform(rng, profile().shared_depth.0, profile().shared_depth.1)
            };
            let mut blocks = Vec::with_capacity(shared_len as usize);
            let mut prev_key = 0u64;
            for i in 0..shared_len {
                let content = next_content(rng);
                let key = if i == 0 {
                    content
                } else {
                    (prev_key ^ content.rotate_left(17)).wrapping_mul(0x2545F4914F6CDD1D) | 1
                };
                blocks.push((key, content));
                prev_key = key;
            }
            let holder_count = holders();
            let mut members = Vec::with_capacity(*h);
            let base = rng.below(holder_count);
            for k in 0..*h {
                members.push((base + k * 7) % holder_count);
            }
            members.sort_unstable();
            members.dedup();
            used += shared_len as u64 * members.len() as u64;
            families.push(Family {
                blocks,
                holders: members,
            });
        }
    }
    families
}

/// Expand families into the mixed write stream (stores + duplicates +
/// gap removes), per-holder order preserved, §7-scoped interleave.
fn build_ops(rng: &mut Rng, families: &[Family]) -> (Vec<Op>, u64, u64) {
    let mut per_holder: Vec<Vec<Op>> = vec![Vec::new(); holders()];
    let mut holder_blocks = 0u64;
    // Exact count of blocks the gap removes take back out: the
    // resident cross-check must not blame the structure for blocks
    // the WORKLOAD removed (the churn profile's 10% gaps tripped the
    // pinned-calibrated 97% floor and killed the run).
    let mut removed_blocks = 0u64;
    for family in families {
        for &holder in &family.holders {
            let mut parent = None;
            let mut batches = Vec::new();
            for chunk in family.blocks.chunks(BATCH) {
                batches.push(Op::Store {
                    holder,
                    parent,
                    blocks: chunk.to_vec(),
                });
                parent = Some(chunk.last().expect("non-empty").0);
            }
            // Divergent tail.
            let (tail_lo, tail_hi) = profile().tail_len;
            let tail_len = tail_lo + (rng.next() % (tail_hi - tail_lo + 1) as u64) as u32;
            let mut prev_key = parent.expect("family non-empty");
            let mut tail = Vec::with_capacity(tail_len as usize);
            for _ in 0..tail_len {
                let content = rng.next() | 1;
                let key = (prev_key ^ content.rotate_left(29))
                    .wrapping_mul(0x9E3779B97F4A7C15)
                    .wrapping_add(holder as u64)
                    | 1;
                tail.push((key, content));
                prev_key = key;
            }
            for chunk in tail.chunks(BATCH) {
                batches.push(Op::Store {
                    holder,
                    parent,
                    blocks: chunk.to_vec(),
                });
                parent = Some(chunk.last().expect("non-empty").0);
            }
            holder_blocks += (family.blocks.len() + tail.len()) as u64;
            let mut dups = Vec::new();
            for b in &batches {
                if rng.chance(profile().duplicate_pct) {
                    dups.push(b.clone());
                }
            }
            let mut gaps = Vec::new();
            if rng.chance(profile().gap_pct * 10) && family.blocks.len() > 2 {
                // ~GAP_PCT of blocks overall: one block per ~10% of
                // instances at these lengths.
                let victim = family.blocks[1 + rng.below(family.blocks.len() - 2)].0;
                gaps.push(Op::Remove {
                    holder,
                    keys: vec![victim],
                });
                // Dups precede gaps in the script, so the victim is
                // present exactly once when the remove lands.
                removed_blocks += 1;
            }
            let script = &mut per_holder[holder];
            script.extend(batches);
            script.extend(dups);
            script.extend(gaps);
        }
    }
    let mut cursors = vec![0usize; holders()];
    let mut ops = Vec::new();
    let mut live: Vec<usize> = (0..holders())
        .filter(|&h| !per_holder[h].is_empty())
        .collect();
    while !live.is_empty() {
        let li = rng.below(live.len());
        let h = live[li];
        ops.push(per_holder[h][cursors[h]].clone());
        cursors[h] += 1;
        if cursors[h] == per_holder[h].len() {
            live.swap_remove(li);
        }
    }
    (ops, holder_blocks, removed_blocks)
}

fn rss_kib() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .unwrap_or(0)
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

/// Which implementation this process measures. One side per process
/// so RSS deltas are clean (§11 protocol):
///   RADIX_BENCH_SIDE=oracle (default) | r1
fn side() -> String {
    std::env::var("RADIX_BENCH_SIDE").unwrap_or_else(|_| "oracle".into())
}

#[allow(clippy::large_enum_variant)] // bench-local, two instances ever
enum Sider {
    Oracle(Oracle, Vec<Vec<u64>>),
    R1(
        FlatTree,
        Vec<HolderId>,
        Vec<radix_tree::Overlap>,
        OverlapScratch,
    ),
    R3(
        RadixTree,
        Vec<HolderId>,
        Vec<radix_tree::Overlap>,
        OverlapScratch,
    ),
}

impl Sider {
    fn apply(&mut self, op: &Op) {
        match self {
            Sider::Oracle(oracle, chains) => {
                oracle.apply(op);
                if let Op::Store {
                    holder,
                    parent,
                    blocks,
                } = op
                {
                    // Engine glue the §11 oracle side must carry:
                    // last-chain Vec (reset on parent-None anchors).
                    let chain = &mut chains[*holder];
                    if parent.is_none() {
                        chain.clear();
                    }
                    chain.extend(blocks.iter().map(|&(k, _)| k));
                }
            }
            Sider::R1(tree, ids, _, _) => match op {
                Op::Store {
                    holder,
                    parent,
                    blocks,
                } => {
                    tree.store(ids[*holder], *parent, blocks)
                        .expect("bench stores are in-contract");
                }
                Op::Remove { holder, keys } => {
                    tree.remove(ids[*holder], keys);
                }
                Op::Clear { holder } => tree.clear(ids[*holder]),
            },
            Sider::R3(tree, ids, _, _) => match op {
                Op::Store {
                    holder,
                    parent,
                    blocks,
                } => {
                    tree.store(ids[*holder], *parent, blocks)
                        .expect("bench stores are in-contract");
                }
                Op::Remove { holder, keys } => {
                    tree.remove(ids[*holder], keys);
                }
                Op::Clear { holder } => tree.clear(ids[*holder]),
            },
        }
    }

    fn query(&mut self, q: &[u64]) -> usize {
        match self {
            Sider::Oracle(oracle, _) => oracle.overlap(q).len(),
            Sider::R1(tree, _, out, qscratch) => {
                tree.overlap(q, qscratch, out);
                out.len()
            }
            Sider::R3(tree, _, out, qscratch) => {
                tree.overlap(q, qscratch, out);
                out.len()
            }
        }
    }
}

#[test]
#[ignore = "pinned benchmark; run --release --ignored --nocapture"]
fn pinned_workload() {
    let mut rng = Rng::new(20260901);
    let families = build_families(&mut rng);
    let (ops, holder_blocks, removed_blocks) = build_ops(&mut rng, &families);
    let total_blocks: u64 = ops
        .iter()
        .map(|op| match op {
            Op::Store { blocks, .. } => blocks.len() as u64,
            Op::Remove { keys, .. } => keys.len() as u64,
            Op::Clear { .. } => 0,
        })
        .sum();
    println!(
        "workload: {} families, {} ops, {} stream blocks, {} resident holder-blocks",
        families.len(),
        ops.len(),
        total_blocks,
        holder_blocks
    );

    let side_name = side();
    println!("side: {side_name}");
    println!("profile: {} ({} holders)", profile().name, holders());
    // P4 mixed-phase probes: built BEFORE the fill so latency can be
    // sampled while the write stream runs.
    let mut mixed_probes: Vec<Vec<u64>> = Vec::new();
    for fam in families.iter().filter(|f| f.holders.len() == 64) {
        if fam.blocks.len() >= GATE_DEPTH as usize && mixed_probes.len() < 32 {
            mixed_probes.push(
                fam.blocks[..GATE_DEPTH as usize]
                    .iter()
                    .map(|&(_, c)| c)
                    .collect(),
            );
        }
    }
    for fam in families.iter().filter(|f| f.holders.len() == 1) {
        if mixed_probes.len() < 64 {
            let d = fam.blocks.len().min(64);
            mixed_probes.push(fam.blocks[..d].iter().map(|&(_, c)| c).collect());
        }
    }
    let rss_before = rss_kib();
    let mut sider = match side_name.as_str() {
        "r1" => {
            let mut tree = FlatTree::new(Config::default());
            let ids = (0..holders())
                .map(|h| tree.create_holder(&format!("holder-{h}")))
                .collect();
            Sider::R1(tree, ids, Vec::new(), OverlapScratch::default())
        }
        "r3" => {
            let mut tree = RadixTree::new(Config::default());
            let ids = (0..holders())
                .map(|h| tree.create_holder(&format!("holder-{h}")))
                .collect();
            Sider::R3(tree, ids, Vec::new(), OverlapScratch::default())
        }
        _ => Sider::Oracle(Oracle::new(holders()), vec![Vec::new(); holders()]),
    };
    let fill_start = Instant::now();
    let mut mixed_lat: Vec<u64> = Vec::new();
    let mut since_probe = 0u64;
    let mut probe_idx = 0usize;
    for op in &ops {
        sider.apply(op);
        if let Op::Store { blocks, .. } = op {
            since_probe += blocks.len() as u64;
        }
        // P4: one probe query every ~250k stream blocks, timed inside
        // the live write phase.
        if since_probe >= 250_000 && !mixed_probes.is_empty() {
            since_probe = 0;
            let q = &mixed_probes[probe_idx % mixed_probes.len()];
            probe_idx += 1;
            let t = Instant::now();
            let n = sider.query(q);
            mixed_lat.push(t.elapsed().as_nanos() as u64);
            std::hint::black_box(n);
        }
    }
    let fill = fill_start.elapsed();
    // Cross-check the memory-gate denominator against the structure
    // itself: a silent drop would otherwise flatter every number at
    // once (audit finding). Gap removes make resident slightly less
    // than generated.
    let resident = match &sider {
        Sider::Oracle(_, _) => holder_blocks, // oracle side has no cheap recount
        Sider::R1(tree, _, _, _) => tree.stats().holder_blocks,
        Sider::R3(tree, _, _, _) => tree.stats().holder_blocks,
    };
    let expected = holder_blocks - removed_blocks;
    assert!(
        resident * 100 >= expected * 97,
        "structure holds {resident} of {expected} expected holder-blocks \
         ({holder_blocks} generated - {removed_blocks} removed) — silent loss"
    );
    let rss_after = rss_kib();
    println!(
        "fill: {:.2}s -> {:.2}M stream blocks/s",
        fill.as_secs_f64(),
        total_blocks as f64 / fill.as_secs_f64() / 1e6
    );
    println!(
        "memory: {} KiB delta -> {:.1} B/holder-block ({} holder-blocks)",
        rss_after - rss_before,
        (rss_after - rss_before) as f64 * 1024.0 / holder_blocks as f64,
        holder_blocks
    );

    // Query phase: prefix probes per sharing cell + misses. Build all
    // query vectors BEFORE timing (no allocation inside the region).
    let mut cells: Vec<(String, Vec<Vec<u64>>)> = Vec::new();
    let mut gate_queries = Vec::new();
    let warm = |fam: &Family, depth: u32| -> Vec<u64> {
        fam.blocks[..depth as usize]
            .iter()
            .map(|&(_, c)| c)
            .collect()
    };
    for &(h, _) in &profile().sharing_mix {
        let mut queries = Vec::new();
        for fam in families.iter().filter(|f| f.holders.len() == h) {
            if queries.len() >= 2000 {
                break;
            }
            let d = 1 + rng.below(fam.blocks.len());
            queries.push(warm(fam, d as u32));
        }
        cells.push((format!("H={h}"), queries));
    }
    for fam in families.iter().filter(|f| f.holders.len() == 64) {
        if fam.blocks.len() >= GATE_DEPTH as usize && gate_queries.len() < 2000 {
            gate_queries.push(warm(fam, GATE_DEPTH));
        }
    }
    cells.push((format!("gate d={GATE_DEPTH} W=64"), gate_queries));
    let mut misses = Vec::new();
    for _ in 0..1000 {
        let len = 1 + rng.below(96);
        misses.push((0..len).map(|_| rng.next() | 1).collect::<Vec<u64>>());
    }
    cells.push(("miss".to_string(), misses));

    if !mixed_lat.is_empty() {
        mixed_lat.sort_unstable();
        println!(
            "cell mixed-phase: n={} p50={}ns p90={}ns p95={}ns p99={}ns (queries during live writes)",
            mixed_lat.len(),
            percentile(&mixed_lat, 0.50),
            percentile(&mixed_lat, 0.90),
            percentile(&mixed_lat, 0.95),
            percentile(&mixed_lat, 0.99),
        );
    }
    for (label, queries) in &cells {
        if queries.is_empty() {
            println!("cell {label}: EMPTY (workload bug)");
            continue;
        }
        // MISS_QUERY_PCT is embodied by the dedicated miss cell; warm
        // cells stay pure so percentiles are per-shape.
        let _ = MISS_QUERY_PCT;
        let mut lat: Vec<u64> = Vec::with_capacity(queries.len());
        for q in queries {
            let t = Instant::now();
            let n = sider.query(q);
            let ns = t.elapsed().as_nanos() as u64;
            std::hint::black_box(n);
            lat.push(ns);
        }
        lat.sort_unstable();
        println!(
            "cell {label}: n={} p50={}ns p90={}ns p95={}ns p99={}ns",
            lat.len(),
            percentile(&lat, 0.50),
            percentile(&lat, 0.90),
            percentile(&lat, 0.95),
            percentile(&lat, 0.99),
        );
    }
    // P4 soak (RADIX_BENCH_SOAK_SECS): steady-state churn — cyclic
    // duplicate re-publish (idempotent placement churn), retire/create
    // holder cycles, and a query stream. RSS must stay flat: growth
    // here is a leak by definition (no new state is being added).
    let soak_secs: u64 = std::env::var("RADIX_BENCH_SOAK_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if soak_secs > 0 {
        println!("soak: {soak_secs}s of duplicate/holder churn + queries");
        let soak_start = Instant::now();
        let mut last_report = Instant::now();
        let rss_soak_start = rss_kib();
        let mut op_i = 0usize;
        let mut probe_i = 0usize;
        let mut churn_cycle = 0u64;
        let mut qps = 0u64;
        let mut ops_applied = 0u64;
        while soak_start.elapsed().as_secs() < soak_secs {
            for _ in 0..2000 {
                sider.apply(&ops[op_i % ops.len()]);
                op_i += 1;
            }
            ops_applied += 2000;
            for _ in 0..64 {
                if !mixed_probes.is_empty() {
                    let q = &mixed_probes[probe_i % mixed_probes.len()];
                    probe_i += 1;
                    std::hint::black_box(sider.query(q));
                    qps += 1;
                }
            }
            if let Sider::R1(tree, _, _, _) = &mut sider {
                churn_cycle += 1;
                let h = tree.create_holder(&format!("soak-churn-{churn_cycle}"));
                let _ = tree.store(h, None, &[(churn_cycle | 1, churn_cycle | 1)]);
                tree.retire_holder(h);
            } else if let Sider::R3(tree, _, _, _) = &mut sider {
                churn_cycle += 1;
                let h = tree.create_holder(&format!("soak-churn-{churn_cycle}"));
                let _ = tree.store(h, None, &[(churn_cycle | 1, churn_cycle | 1)]);
                tree.retire_holder(h);
            }
            if last_report.elapsed().as_secs() >= 60 {
                last_report = Instant::now();
                println!(
                    "soak t={}s rss={} KiB (drift {:+} KiB) ops={} queries={}",
                    soak_start.elapsed().as_secs(),
                    rss_kib(),
                    rss_kib() as i64 - rss_soak_start as i64,
                    ops_applied,
                    qps
                );
                if let Sider::R3(tree, _, _, _) = &sider {
                    println!("  footprint: {}", tree.debug_footprint());
                }
            }
        }
        let drift = rss_kib() as i64 - rss_soak_start as i64;
        println!(
            "soak done: rss drift {:+} KiB over {soak_secs}s ({} ops, {} queries)",
            drift, ops_applied, qps
        );
    }
    std::hint::black_box(&sider as *const _);
}
