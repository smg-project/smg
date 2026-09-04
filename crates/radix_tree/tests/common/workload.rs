//! Seeded workload generator producing §7-scoped operation streams.
//!
//! Shape (mirrors the pinned-workload structure at test
//! scale): prefix FAMILIES — shared base chains that several holders
//! store identically (same keys, same contents: exactly what the
//! placement feed produces) — plus per-holder divergent tails,
//! duplicate re-stores, gap-punching removes, rare clears, and
//! optional cross-family content coincidence to arm the oracle's
//! Single-entry lineage skip.
//!
//! Ordering scope: the emitted stream preserves each holder's own
//! order (ops interleave only ACROSS holders), matching §7. The
//! convergence tests additionally reorder within that scope.

use super::{Op, Rng};

#[derive(Debug, Clone)]
pub struct Config {
    pub holders: usize,
    pub families: usize,
    /// Blocks per family base chain: uniform in [min, max].
    pub family_len: (usize, usize),
    /// Holders per family: uniform in [min, max].
    pub holders_per_family: (usize, usize),
    /// Divergent tail blocks per holder per family: uniform [min, max].
    pub tail_len: (usize, usize),
    pub store_batch: usize,
    /// Percent of batches re-sent verbatim later (duplicates).
    pub duplicate_pct: u64,
    /// Percent of (holder, family) pairs that lose a mid-chain block
    /// (gap injection via remove).
    pub gap_pct: u64,
    /// Percent of holders cleared once mid-stream.
    pub clear_pct: u64,
    /// Reuse content values across families at this percent per
    /// block, arming content coincidence (oracle quirk class 3).
    pub coincidence_pct: u64,
}

impl Config {
    pub fn small() -> Self {
        Self {
            holders: 24,
            families: 8,
            family_len: (8, 96),
            holders_per_family: (1, 8),
            tail_len: (2, 16),
            store_batch: 8,
            duplicate_pct: 10,
            gap_pct: 15,
            clear_pct: 8,
            coincidence_pct: 0,
        }
    }

    pub fn with_coincidence() -> Self {
        Self {
            coincidence_pct: 25,
            ..Self::small()
        }
    }
}

/// The generated workload: an op stream (per-holder order already
/// legal per §7) plus the query set derived from stored prefixes and
/// misses.
#[derive(Debug, Clone)]
pub struct Workload {
    pub ops: Vec<Op>,
    pub queries: Vec<Vec<u64>>,
    pub holders: usize,
}

pub fn generate(seed: u64, cfg: &Config) -> Workload {
    let mut rng = Rng::new(seed);
    let mut content_pool: Vec<u64> = Vec::new();
    let fresh_content = |rng: &mut Rng, pool: &mut Vec<u64>, coincidence_pct: u64| -> u64 {
        if !pool.is_empty() && rng.chance(coincidence_pct) {
            pool[rng.below(pool.len())]
        } else {
            let c = rng.next() | 1; // avoid 0 (the model's lineage filler)
            pool.push(c);
            c
        }
    };

    // Families: shared (key, content) base chains. Keys are the
    // deterministic placement chain of the contents — identical
    // across holders for identical prefixes, as on the wire.
    let mut families: Vec<Vec<(u64, u64)>> = Vec::new();
    for _ in 0..cfg.families {
        let len = cfg.family_len.0 + rng.below(cfg.family_len.1 - cfg.family_len.0 + 1);
        let mut chain = Vec::with_capacity(len);
        let mut prev_key = 0u64;
        for i in 0..len {
            // Position 0 contents stay unique: pos-0 keys ARE the
            // content (wire position-0 rule), so coincidence there
            // would fuse two families into one chain — out of the §7
            // chain-consistent scope this generator promises.
            let coincidence = if i == 0 { 0 } else { cfg.coincidence_pct };
            let content = fresh_content(&mut rng, &mut content_pool, coincidence);
            // Chain-hash-shaped keys without depending on the wire
            // scheme: mix prev key and content deterministically.
            let key = if i == 0 {
                content
            } else {
                let mut k = prev_key ^ content.rotate_left(17);
                k = k.wrapping_mul(0x2545F4914F6CDD1D) | 1;
                k
            };
            chain.push((key, content));
            prev_key = key;
        }
        families.push(chain);
    }

    // Assignment + per-holder scripts (sequences that MUST keep their
    // relative order for that holder). A repeat assignment of the
    // same family to the same holder re-stores the shared chain
    // (pure duplicates) but must NOT grow a second divergent tail:
    // two different keys at one chain position is outside the §7
    // chain-consistent scope this generator promises (caught by the
    // enumerate parity check).
    let mut per_holder: Vec<Vec<Op>> = vec![Vec::new(); cfg.holders];
    let mut tailed: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    // (holder, fork-position, tail blocks) pending mid-chain diverging
    // queries.
    type PendingTail = (usize, u32, Vec<(u64, u64)>);
    let mut tails: Vec<PendingTail> = Vec::new();
    for (fi, family) in families.iter().enumerate() {
        let count = cfg.holders_per_family.0
            + rng.below(cfg.holders_per_family.1 - cfg.holders_per_family.0 + 1);
        for _ in 0..count {
            let holder = rng.below(cfg.holders);
            let first_assignment = tailed.insert((holder, fi));
            // A third of members store only a PREFIX of the family,
            // so membership decays with depth (the run-boundary
            // structure real fleets produce; audit finding: constant
            // membership along every spine left split/merge
            // maintenance under-exercised).
            let span = if rng.chance(33) && family.len() > 2 {
                1 + rng.below(family.len() - 1)
            } else {
                family.len()
            };
            let stored_spine = &family[..span];
            let mut parent = None;
            let mut batches = Vec::new();
            for batch in stored_spine.chunks(cfg.store_batch) {
                batches.push(Op::Store {
                    holder,
                    parent,
                    blocks: batch.to_vec(),
                });
                parent = Some(batch.last().expect("non-empty").0);
            }
            // Divergent tail (first assignment only): unique
            // contents, keys mixed with holder and family so tails
            // never collide. A third of tails fork MID-CHAIN (audit
            // finding: tip-only forks left interior branch handling
            // untested in-contract).
            if first_assignment {
                let fork_at = if rng.chance(33) && span > 2 {
                    1 + rng.below(span - 1)
                } else {
                    span
                };
                let fork_parent = stored_spine[fork_at - 1].0;
                let tail_len = cfg.tail_len.0 + rng.below(cfg.tail_len.1 - cfg.tail_len.0 + 1);
                let mut tail = Vec::with_capacity(tail_len);
                let mut prev_key = fork_parent;
                for _ in 0..tail_len {
                    let content = fresh_content(&mut rng, &mut content_pool, 0);
                    let key = (prev_key ^ content.rotate_left(29))
                        .wrapping_mul(0x9E3779B97F4A7C15)
                        .wrapping_add(holder as u64 ^ (fi as u64) << 32)
                        | 1;
                    tail.push((key, content));
                    prev_key = key;
                }
                let mut tail_parent = Some(fork_parent);
                for batch in tail.chunks(cfg.store_batch) {
                    batches.push(Op::Store {
                        holder,
                        parent: tail_parent,
                        blocks: batch.to_vec(),
                    });
                    tail_parent = Some(batch.last().expect("non-empty").0);
                }
                // Record the tail for query generation.
                tails.push((fi, fork_at as u32, tail));
            }
            // Duplicates: re-send some batches verbatim (later in the
            // holder's script — legal §7 duplication).
            let mut dups = Vec::new();
            for b in &batches {
                if rng.chance(cfg.duplicate_pct) {
                    dups.push(b.clone());
                }
            }
            // Gap: remove one mid-chain family block.
            let mut gaps = Vec::new();
            if rng.chance(cfg.gap_pct) && family.len() > 2 {
                let victim = family[1 + rng.below(family.len() - 2)].0;
                gaps.push(Op::Remove {
                    holder,
                    keys: vec![victim],
                });
            }
            let script = &mut per_holder[holder];
            script.extend(batches);
            script.extend(dups);
            script.extend(gaps);
        }
    }
    for (holder, script) in per_holder.iter_mut().enumerate() {
        if rng.chance(cfg.clear_pct) && !script.is_empty() {
            // Clear mid-script: everything before it is discarded
            // state; everything after rebuilds. Insert at a random
            // point rather than the end so post-clear stores exist.
            let at = rng.below(script.len());
            script.insert(at, Op::Clear { holder });
        }
    }

    // Interleave across holders preserving each holder's order.
    let mut cursors = vec![0usize; cfg.holders];
    let mut ops = Vec::new();
    loop {
        let live: Vec<usize> = (0..cfg.holders)
            .filter(|&h| cursors[h] < per_holder[h].len())
            .collect();
        if live.is_empty() {
            break;
        }
        let h = live[rng.below(live.len())];
        ops.push(per_holder[h][cursors[h]].clone());
        cursors[h] += 1;
    }

    // Post-clear stores may reference parents wiped by the clear;
    // both sides must reject those identically (ParentNotFound), so
    // they stay in the stream on purpose.

    // Queries: spine prefixes, TAIL-EXTENDING (into one holder's
    // divergent tail — distinguishes that holder's deeper answer),
    // MID-DIVERGING (match d blocks then differ in content), and
    // pure misses. The first shape set alone let off-query-path
    // corruption hide (audit finding).
    let mut queries = Vec::new();
    for family in &families {
        for _ in 0..3 {
            let d = 1 + rng.below(family.len());
            queries.push(family[..d].iter().map(|&(_, c)| c).collect::<Vec<u64>>());
        }
        // Mid-diverging: real prefix then foreign content.
        let d = 1 + rng.below(family.len());
        let mut q: Vec<u64> = family[..d].iter().map(|&(_, c)| c).collect();
        q.extend((0..1 + rng.below(8)).map(|_| rng.next() | 1));
        queries.push(q);
    }
    for (fi, fork_at, tail) in &tails {
        if queries.len() > families.len() * 8 {
            break;
        }
        let spine = &families[*fi];
        let mut q: Vec<u64> = spine[..*fork_at as usize].iter().map(|&(_, c)| c).collect();
        let take = 1 + rng.below(tail.len());
        q.extend(tail[..take].iter().map(|&(_, c)| c));
        queries.push(q);
    }
    for _ in 0..families.len() {
        let miss: Vec<u64> = (0..1 + rng.below(24)).map(|_| rng.next() | 1).collect();
        queries.push(miss);
    }
    Workload {
        ops,
        queries,
        holders: cfg.holders,
    }
}

/// Reorder a stream within the FULL §7 scope: arbitrary cross-holder
/// interleaving always; and for holders whose scripts carry no
/// remove/clear, their OWN store order is also shuffled arbitrarily
/// (the stronger half of the guarantee — audit finding: it was never
/// exercised). Order-bearing holders keep their sequence.
pub fn reinterleave(seed: u64, ops: &[Op], holders: usize) -> Vec<Op> {
    let mut per_holder: Vec<Vec<Op>> = vec![Vec::new(); holders];
    for op in ops {
        per_holder[op.holder()].push(op.clone());
    }
    let mut rng0 = Rng::new(seed ^ 0x5EED);
    for script in per_holder.iter_mut() {
        if !script.iter().any(|op| op.orders_holder()) && script.len() > 1 {
            // Fisher-Yates over the store-only script. Parent links
            // may now arrive before their anchor: §7 excludes
            // ParentNotFound compensation from the guarantee, so the
            // comparison target must apply the SAME order — callers
            // compare two subjects on one order, or model-vs-subject
            // on the same order, never across orders with rejects.
            // We keep it in-contract instead: shuffle only WHOLE
            // parent-linked chains (contiguous runs where each op's
            // parent is the previous op's last key).
            let mut runs: Vec<Vec<Op>> = Vec::new();
            let mut current: Vec<Op> = Vec::new();
            let mut last_key: Option<u64> = None;
            for op in script.drain(..) {
                let anchors_prev = matches!(
                    (&op, last_key),
                    (Op::Store { parent: Some(p), .. }, Some(k)) if *p == k
                );
                if !anchors_prev && !current.is_empty() {
                    runs.push(std::mem::take(&mut current));
                }
                last_key = match &op {
                    Op::Store { blocks, .. } => blocks.last().map(|&(k, _)| k),
                    _ => None,
                };
                current.push(op);
            }
            if !current.is_empty() {
                runs.push(current);
            }
            // Random TOPOLOGICAL order: a run whose anchor parent
            // key is produced by another run must stay after it —
            // literal arbitrary order would change which stores get
            // ACCEPTED (ParentNotFound), which is outside §7's
            // chain-consistent scope (rejected stores leave the
            // multiset). Within the dependency partial order, the
            // shuffle is free.
            let produced: Vec<std::collections::HashSet<u64>> = runs
                .iter()
                .map(|run| {
                    run.iter()
                        .flat_map(|op| match op {
                            Op::Store { blocks, .. } => {
                                blocks.iter().map(|&(k, _)| k).collect::<Vec<_>>()
                            }
                            _ => Vec::new(),
                        })
                        .collect()
                })
                .collect();
            let needs: Vec<Option<u64>> = runs
                .iter()
                .map(|run| match run.first() {
                    Some(Op::Store { parent, .. }) => *parent,
                    _ => None,
                })
                .collect();
            let n = runs.len();
            let mut placed = vec![false; n];
            let mut ordered: Vec<Vec<Op>> = Vec::with_capacity(n);
            let mut produced_so_far: std::collections::HashSet<u64> =
                std::collections::HashSet::new();
            while ordered.len() < n {
                let ready: Vec<usize> = (0..n)
                    .filter(|&i| {
                        !placed[i] && needs[i].is_none_or(|k| produced_so_far.contains(&k))
                    })
                    .collect();
                // Runs whose parent was never produced in this script
                // (anchored on another assignment's spine already
                // present) are always ready.
                let ready = if ready.is_empty() {
                    (0..n).filter(|&i| !placed[i]).collect()
                } else {
                    ready
                };
                let pick = ready[rng0.below(ready.len())];
                placed[pick] = true;
                produced_so_far.extend(produced[pick].iter().copied());
                ordered.push(std::mem::take(&mut runs[pick]));
            }
            *script = ordered.into_iter().flatten().collect();
        }
    }
    let mut rng = Rng::new(seed);
    let mut cursors = vec![0usize; holders];
    let mut out = Vec::with_capacity(ops.len());
    loop {
        let live: Vec<usize> = (0..holders)
            .filter(|&h| cursors[h] < per_holder[h].len())
            .collect();
        if live.is_empty() {
            break;
        }
        let h = live[rng.below(live.len())];
        out.push(per_holder[h][cursors[h]].clone());
        cursors[h] += 1;
    }
    out
}
