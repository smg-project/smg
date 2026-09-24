//! The model-referee harness.
//!
//! Three pieces:
//! - [`model`]: a trivially-correct implementation of the §6/§7
//!   contract — per-holder chain forests with literal lineage
//!   vectors. Slow, obvious, and the referee for everything else.
//! - [`oracle`]: `kv_index::PositionalIndexer` behind the engine's
//!   replicated glue (per-holder reverse maps, interning), applied
//!   through the same operation stream.
//! - [`workload`]: a seeded deterministic generator producing
//!   §7-scoped operation streams with shared prefixes, divergent
//!   tails, duplicates, gaps, and content coincidence.

// Each integration-test binary compiles this module independently and
// uses a different slice of it; unused-in-this-binary is expected.
#![allow(dead_code)]

pub mod model;
pub mod oracle;
pub mod workload;

/// One operation in a workload stream. Holder identity is a dense
/// index (the harness maps it to names/ids per implementation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    Store {
        holder: usize,
        /// `None` anchors a new chain at position 0.
        parent: Option<u64>,
        /// (BlockKey, ContentHash) pairs, consecutive positions.
        blocks: Vec<(u64, u64)>,
    },
    Remove {
        holder: usize,
        keys: Vec<u64>,
    },
    Clear {
        holder: usize,
    },
}

impl Op {
    pub fn holder(&self) -> usize {
        match self {
            Op::Store { holder, .. } | Op::Remove { holder, .. } | Op::Clear { holder } => *holder,
        }
    }

    /// Whether reordering this op against its holder's stores is
    /// outside the §7 convergence scope. (Used by the R1 convergence
    /// suite; the R0 reinterleaver conservatively keeps all order.)
    #[allow(dead_code)]
    pub fn orders_holder(&self) -> bool {
        matches!(self, Op::Remove { .. } | Op::Clear { .. })
    }
}

/// SplitMix64: tiny deterministic RNG so the harness has zero deps
/// and identical streams on every platform.
#[derive(Debug, Clone)]
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    /// Next raw 64-bit output. (Not named `next`: an inherent `next`
    /// on a public type trips `clippy::should_implement_trait`.)
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, n).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// Bernoulli with probability pct/100.
    pub fn chance(&mut self, pct: u64) -> bool {
        self.next_u64() % 100 < pct
    }
}

/// A present-but-malformed campaign/bench variable is an operator error,
/// never a silent fall-through to the default (a `10_000` typo would
/// otherwise run the default and report success for a run nobody asked
/// for). Absent => default. One parse contract for every test binary.
pub fn env_u64(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|e| panic!("{name}={v:?} is not a u64: {e}")),
        Err(_) => default,
    }
}

/// Resident set size of this process in KiB, from `ps`. Shared by the
/// two RSS-measuring benchmarks so the parse contract cannot drift: an
/// unparsable `ps` line is a failed measurement, never a silent zero
/// (which would wrap the `after - before` delta in release builds).
pub fn rss_kib() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    let text = String::from_utf8_lossy(&out.stdout);
    text.trim()
        .parse()
        .unwrap_or_else(|e| panic!("cannot parse ps rss output {text:?}: {e}"))
}
