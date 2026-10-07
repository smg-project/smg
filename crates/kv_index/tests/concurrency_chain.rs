//! Concurrency harness for `ChainIndex`: many event lanes writing shared chains at once while
//! readers look them up, then the end state against the `ReferenceIndexer`.
//!
//! Lanes own one worker each and replay random stores, extensions, tail and middle evictions,
//! clears and worker replacements over a shared pool of conversations, so runs are joined, split,
//! truncated and unlinked by different threads at the same time. Every lane logs what it applied;
//! after the lanes stop, the logs are replayed in application order into the reference, which
//! must hold exactly the same blocks and score every pool chain the same way. Readers run
//! throughout and check the invariants that hold under any interleaving: no score exceeds the
//! request, early-exit scores are 1, and nothing panics.
#![expect(clippy::expect_used)]

use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use kv_index::{
    ChainBlockMap, ContentHash, ReferenceIndexer, SequenceHash, ShardedChainIndex, StoredBlock,
};

mod common;
use common::{blocks_of, content, Rng};

/// A shared pool of conversations: a few prompts, each continued by several turns, with sibling
/// turns branching off at various depths.
fn pool(rng: &mut Rng) -> Vec<Vec<ContentHash>> {
    let mut chains: Vec<Vec<ContentHash>> = Vec::new();
    let mut stream = 1u64;
    for _ in 0..6 {
        stream += 1;
        let prompt: Vec<ContentHash> = (0..rng.range(8, 40)).map(|p| content(stream, p)).collect();
        for _ in 0..12 {
            let base = if chains.is_empty() || rng.below(3) == 0 {
                prompt.clone()
            } else {
                let parent = &chains[chains.len() - 1 - rng.below(chains.len().min(12))];
                parent[..rng.range(1, parent.len())].to_vec()
            };
            stream += 1;
            let mut chain = base;
            chain.extend((0..rng.range(1, 24)).map(|p| content(stream, p)));
            chains.push(chain);
        }
    }
    chains
}

enum Event {
    Stored {
        blocks: Vec<StoredBlock>,
        parent: Option<SequenceHash>,
    },
    Removed(Vec<SequenceHash>),
    Cleared,
    WorkerRemoved,
}

struct Logged {
    seq: u64,
    worker: u32,
    event: Event,
}

struct Lane<'a> {
    index: &'a ShardedChainIndex,
    pool: &'a [Vec<ContentHash>],
    clock: &'a AtomicU64,
    worker: u32,
    name_counter: u64,
    lane: usize,
    map: ChainBlockMap,
    rng: Rng,
    log: Vec<Logged>,
    steps: u32,
}

impl Lane<'_> {
    /// The index credits this worker with exactly the blocks its lane map names. A holding the
    /// index drops behind the map's back (or keeps after the map forgot it) shows here, at the
    /// step that caused it, not at the end state or at a later store under the block.
    fn check_count(&self, when: &str) {
        let credited = self.index.worker_block_count(self.worker);
        assert!(
            credited == self.map.len(),
            "lane {} worker {} step {}: the index credits {credited} blocks, the lane map names {} ({when})",
            self.lane,
            self.worker,
            self.steps,
            self.map.len()
        );
    }

    fn record(&mut self, event: Event) {
        let seq = self.clock.fetch_add(1, Ordering::Relaxed);
        self.record_at(seq, event);
    }

    fn record_at(&mut self, seq: u64, event: Event) {
        self.log.push(Logged {
            seq,
            worker: self.worker,
            event,
        });
    }

    fn store(&mut self, contents: &[ContentHash]) {
        let blocks = blocks_of(contents);
        let mut known = 0;
        while known < blocks.len()
            && self
                .index
                .is_held(self.worker, &self.map, blocks[known].seq_hash)
        {
            known += 1;
        }
        let start = if known == blocks.len() {
            blocks.len() - 1
        } else {
            known
        };
        let parent = (start > 0).then(|| blocks[start - 1].seq_hash);
        let outcome = self
            .index
            .apply_stored(self.worker, &blocks[start..], parent, &mut self.map);
        assert!(
            outcome.is_ok(),
            "lane {}: store after a held parent failed: {outcome:?}",
            self.lane
        );
        self.record(Event::Stored {
            blocks: blocks[start..].to_vec(),
            parent,
        });
        self.check_count("after a store");
    }

    /// Remove a contiguous range of the blocks this lane holds on a pool chain.
    fn remove_some(&mut self, contents: &[ContentHash]) {
        let blocks = blocks_of(contents);
        let held: Vec<usize> = (0..blocks.len())
            .filter(|&i| {
                self.index
                    .is_held(self.worker, &self.map, blocks[i].seq_hash)
            })
            .collect();
        if held.is_empty() {
            return;
        }
        let from = self.rng.below(held.len());
        let to = match self.rng.below(4) {
            0 => held.len(),
            _ => (from + self.rng.range(1, 3)).min(held.len()),
        };
        let hashes: Vec<SequenceHash> =
            held[from..to].iter().map(|&i| blocks[i].seq_hash).collect();
        self.index
            .apply_removed(self.worker, &hashes, &mut self.map);
        self.record(Event::Removed(hashes));
        self.check_count("after a removal");
    }

    fn step(&mut self) {
        let chain = self.pool[self.rng.below(self.pool.len())].clone();
        // Worker replacement is part of every run, not a rare roll: each lane swaps its worker
        // every 97 steps (so lanes do it at different times) besides the random 1%.
        self.steps += 1;
        if self.steps.is_multiple_of(97) {
            self.replace_worker();
            return;
        }
        match self.rng.below(1000) {
            0..=549 => {
                let len = if self.rng.below(2) == 0 {
                    chain.len()
                } else {
                    self.rng.range(1, chain.len())
                };
                self.store(&chain[..len]);
            }
            550..=899 => self.remove_some(&chain),
            900..=984 => {
                // Walk a chain in two turns: a prefix now, the rest right after (decode extension).
                let cut = self.rng.range(1, chain.len());
                self.store(&chain[..cut]);
                self.store(&chain);
            }
            985..=989 => {
                self.index.apply_cleared(self.worker, &mut self.map);
                self.record(Event::Cleared);
                self.check_count("after a clear");
            }
            _ => self.replace_worker(),
        }
    }

    /// Remove this lane's worker and intern a fresh one (its slot may come back reused).
    fn replace_worker(&mut self) {
        let map = std::mem::take(&mut self.map);
        // Sequenced before the removal: another lane may intern the freed slot and store under
        // the same id before this lane gets to record, and the replay must see the removal first.
        let seq = self.clock.fetch_add(1, Ordering::Relaxed);
        self.index.remove_worker(self.worker, map);
        self.record_at(seq, Event::WorkerRemoved);
        self.name_counter += 1;
        let name = format!("lane-{}-{}", self.lane, self.name_counter);
        self.worker = self.index.intern_worker(&name).expect("worker slot");
        self.check_count("after a worker replacement");
    }
}

fn scores(
    index: &ShardedChainIndex,
    query: &[ContentHash],
    early_exit: bool,
) -> BTreeMap<u32, u32> {
    index
        .find_matches(query, early_exit)
        .scores
        .into_iter()
        .collect()
}

const READERS: usize = 4;
/// The run normally ends within seconds; past this the watchdog reports every thread's progress
/// to the process's stderr and aborts, so a wedged run fails instead of holding a gate.
const DEADLINE: Duration = Duration::from_secs(300);
const PHASE_LANES: u8 = 0;
const PHASE_READERS: u8 = 1;
const PHASE_DONE: u8 = 2;

/// Write straight to the process's stderr: the test harness holds the test thread's printed
/// output back until the test ends (its capture sits under the print macros, not under a raw
/// write), and a report that precedes an abort must come out before it.
fn shout(message: &str) {
    let line = format!("{message}\n");
    let _ = std::io::stderr().write_all(line.as_bytes());
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_else(|| "a non-string panic payload".to_string())
}

#[test]
fn concurrent_lanes_and_readers_end_in_the_reference_state() {
    let lanes = 16usize;
    let steps = std::env::var("KV_INDEX_CONCURRENCY_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4000usize);
    let mut rng = Rng::new(20261005);
    let pool = pool(&mut rng);
    let shards = std::env::var("KV_INDEX_CONCURRENCY_SHARDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1usize);
    // Lanes intern their workers round robin over the shards, so with two shards every
    // other lane writes the other index and every lookup merges both.
    let index = ShardedChainIndex::new(shards, 64);
    let clock = AtomicU64::new(0);
    let stop = Arc::new(AtomicBool::new(false));
    let logs: Mutex<Vec<Logged>> = Mutex::new(Vec::new());
    let reader_lookups = Arc::new(AtomicU64::new(0));
    // What the watchdog reports if the run overstays its deadline: steps done per lane, rounds
    // done per reader and the phase the run is in.
    let lane_steps: Arc<Vec<AtomicU32>> = Arc::new((0..lanes).map(|_| AtomicU32::new(0)).collect());
    let reader_rounds: Arc<Vec<AtomicU64>> =
        Arc::new((0..READERS).map(|_| AtomicU64::new(0)).collect());
    let phase = Arc::new(AtomicU8::new(PHASE_LANES));
    let watchdog = {
        let (lane_steps, reader_rounds, phase) = (
            Arc::clone(&lane_steps),
            Arc::clone(&reader_rounds),
            Arc::clone(&phase),
        );
        thread::spawn(move || {
            let deadline = Instant::now() + DEADLINE;
            while phase.load(Ordering::Relaxed) != PHASE_DONE {
                if Instant::now() >= deadline {
                    let lane_steps: Vec<u32> = lane_steps
                        .iter()
                        .map(|s| s.load(Ordering::Relaxed))
                        .collect();
                    let reader_rounds: Vec<u64> = reader_rounds
                        .iter()
                        .map(|r| r.load(Ordering::Relaxed))
                        .collect();
                    shout(&format!(
                        "concurrency harness: {} s deadline exceeded in phase {} (0 lanes running, 1 lanes joined and readers stopping); steps per lane {lane_steps:?} of {steps}; rounds per reader {reader_rounds:?}; aborting",
                        DEADLINE.as_secs(),
                        phase.load(Ordering::Relaxed)
                    ));
                    std::process::abort();
                }
                thread::sleep(Duration::from_millis(200));
            }
        })
    };

    thread::scope(|scope| {
        let mut lane_threads = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let (index, pool, clock, logs, lane_steps) =
                (&index, &pool, &clock, &logs, &lane_steps);
            let seed = rng.next();
            lane_threads.push(scope.spawn(move || {
                let worker = index
                    .intern_worker(&format!("lane-{lane}-0"))
                    .expect("worker slot");
                let mut state = Lane {
                    index,
                    pool,
                    clock,
                    worker,
                    name_counter: 0,
                    lane,
                    map: ChainBlockMap::default(),
                    rng: Rng::new(seed),
                    log: Vec::new(),
                    steps: 0,
                };
                for _ in 0..steps {
                    state.step();
                    lane_steps[lane].fetch_add(1, Ordering::Relaxed);
                }
                logs.lock().unwrap().extend(state.log);
            }));
        }
        let mut reader_threads = Vec::with_capacity(READERS);
        for reader in 0..READERS {
            let (index, pool, stop, reader_lookups, reader_rounds) =
                (&index, &pool, &stop, &reader_lookups, &reader_rounds);
            let seed = rng.next() ^ reader as u64;
            reader_threads.push(scope.spawn(move || {
                let mut rng = Rng::new(seed);
                while !stop.load(Ordering::Relaxed) {
                    let chain = &pool[rng.below(pool.len())];
                    let mut query = chain.clone();
                    if rng.below(3) == 0 {
                        let at = rng.below(query.len());
                        query[at] = content(u64::MAX - reader as u64, at);
                    }
                    let full = scores(index, &query, false);
                    for (&worker, &score) in &full {
                        assert!(
                            score as usize <= query.len(),
                            "worker {worker} scored {score} on a {}-block request",
                            query.len()
                        );
                        assert!(score > 0, "worker {worker} reported with score 0");
                    }
                    let early = scores(index, &query, true);
                    assert!(early.values().all(|&score| score == 1));
                    reader_lookups.fetch_add(1, Ordering::Relaxed);
                    reader_rounds[reader].fetch_add(1, Ordering::Relaxed);
                }
            }));
        }
        // Termination is unconditional. The lanes are finite and joined first; a panic among
        // them is kept rather than raised here, because raising it would leave the readers
        // running under a scope that waits for them (a lane's failed assertion then showed as
        // a hang, its message held back by the test harness's output capture). The readers are
        // told to stop and joined, and only then is the first panic re-raised. The log length
        // is not a completion signal: a removal that finds nothing held logs nothing and the
        // two-turn store logs twice.
        let mut first_panic = None;
        for (lane, handle) in lane_threads.into_iter().enumerate() {
            if let Err(payload) = handle.join() {
                shout(&format!(
                    "concurrency harness: lane {lane} panicked: {}",
                    panic_text(payload.as_ref())
                ));
                first_panic.get_or_insert(payload);
            }
        }
        phase.store(PHASE_READERS, Ordering::Relaxed);
        stop.store(true, Ordering::Relaxed);
        for (reader, handle) in reader_threads.into_iter().enumerate() {
            if let Err(payload) = handle.join() {
                shout(&format!(
                    "concurrency harness: reader {reader} panicked: {}",
                    panic_text(payload.as_ref())
                ));
                first_panic.get_or_insert(payload);
            }
        }
        phase.store(PHASE_DONE, Ordering::Relaxed);
        if let Some(payload) = first_panic {
            std::panic::resume_unwind(payload);
        }
    });
    watchdog.join().expect("the watchdog thread returned");

    let mut logs = logs.into_inner().unwrap();
    logs.sort_by_key(|entry| entry.seq);
    let mut reference = ReferenceIndexer::new();
    for entry in &logs {
        match &entry.event {
            Event::Stored { blocks, parent } => {
                reference
                    .apply_stored(entry.worker, blocks, *parent)
                    .expect("the lane stored after a held parent");
            }
            Event::Removed(hashes) => reference.apply_removed(entry.worker, hashes),
            Event::Cleared => reference.apply_cleared(entry.worker),
            Event::WorkerRemoved => reference.remove_worker(entry.worker),
        }
    }
    assert!(
        reader_lookups.load(Ordering::Relaxed) > 1000,
        "readers barely ran: {} lookups",
        reader_lookups.load(Ordering::Relaxed)
    );

    // Every engine hash here is a prefix hash of its chain, so a block has one place and no
    // hash ever moves: a moved-hash count is a re-store taken for a move.
    assert_eq!(
        index.stats().moved_hashes,
        0,
        "the index released holdings for hashes it took as moved"
    );
    let produced = index.debug_blocks();
    let expected = reference.blocks();
    let missing: Vec<_> = expected.difference(&produced).take(5).collect();
    let phantom: Vec<_> = produced.difference(&expected).take(5).collect();
    assert!(
        missing.is_empty() && phantom.is_empty(),
        "end state differs after {} events: {} reference blocks, {} index blocks; missing e.g. \
         {missing:?}; phantom e.g. {phantom:?}",
        logs.len(),
        expected.len(),
        produced.len()
    );
    let mut checked = 0;
    for chain in &pool {
        for query in [chain.clone(), chain[..chain.len().div_ceil(2)].to_vec()] {
            assert_eq!(
                scores(&index, &query, false),
                reference.find_matches(&query),
                "lookup differs for a {}-block query",
                query.len()
            );
            checked += 1;
        }
    }
    assert!(checked > 100);
    // Distinct blocks are counted per shard: content held on two shards is stored twice.
    assert_eq!(
        index.entry_count(),
        expected
            .iter()
            .map(|(worker, position, content, prefix)| {
                (
                    ShardedChainIndex::shard_of(*worker),
                    *position,
                    *content,
                    *prefix,
                )
            })
            .collect::<BTreeSet<_>>()
            .len(),
        "distinct block counter"
    );
}
