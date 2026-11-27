//! Run headers and the slab that holds them: the seqlock window a reader snapshots, the
//! per-run metadata under its lock, the coverage words of whole holders, and the free list of
//! dead ids with their generations.

use super::*;

/// Writer-side bookkeeping of a run, under its lock.
#[derive(Default)]
pub(super) struct RunMeta {
    /// Unlinked from the tree; its id may be reused (with the next generation).
    pub(super) dead: bool,
}

/// A reader's consistent view of a run's window.
#[derive(Clone, Copy)]
pub(super) struct Window {
    /// Data start of the hash array, or `NONE`.
    pub(super) block: u32,
    /// Offset of the run's first hash within the array.
    pub(super) base: u32,
    /// Data start of the engine-hash array, or `NONE`.
    pub(super) engine: u32,
    pub(super) len: u32,
    /// Child table, or `NONE`.
    pub(super) children: u32,
    /// Partial-holder table, or `NONE`.
    pub(super) partials: u32,
    /// Forwarding records of the splits this run has undergone, or `NONE`: a block that sat at
    /// `offset >= at` before a split lives in that split's suffix at `offset - at` (and may have
    /// been forwarded again from there); a `GONE` suffix means nobody holds it any more. Split
    /// points decrease along the records: a run never grows after a split.
    pub(super) forwards: u32,
}

pub(super) struct Run {
    /// Absolute position of the run's first block.
    pub(super) start: AtomicU32,
    /// Run id of the parent (the root's parent is itself).
    pub(super) parent: AtomicU32,
    /// Generation in the high half, seqlock in the low half: odd while an update is in flight.
    pub(super) version: AtomicU64,
    pub(super) block: AtomicU32,
    pub(super) base: AtomicU32,
    /// Data start of the engine-hash array, parallel to `block` (same base, same length, shared
    /// and released with it), or `NONE`. One engine hash per distinct block, the first holder's.
    pub(super) engine: AtomicU32,
    pub(super) len: AtomicU32,
    pub(super) children: AtomicU32,
    /// Workers holding only a prefix of the run, with how much: `(worker, cutoff)` entries in the
    /// arena. A worker is either in the coverage bitset (whole run) or here, never both.
    pub(super) partials: AtomicU32,
    pub(super) forwards: AtomicU32,
    /// Lock-free child inserts in progress on this run; a writer that replaces the child table
    /// or hands it to a suffix waits for this to drain inside its version step.
    pub(super) inflight: AtomicU32,
    pub(super) meta: Mutex<RunMeta>,
}

impl Run {
    pub(super) fn blank() -> Self {
        Self {
            start: AtomicU32::new(0),
            parent: AtomicU32::new(ROOT),
            version: AtomicU64::new(0),
            block: AtomicU32::new(NONE),
            base: AtomicU32::new(0),
            engine: AtomicU32::new(NONE),
            len: AtomicU32::new(0),
            children: AtomicU32::new(NONE),
            partials: AtomicU32::new(NONE),
            forwards: AtomicU32::new(NONE),
            inflight: AtomicU32::new(0),
            meta: Mutex::new(RunMeta::default()),
        }
    }

    #[inline]
    pub(super) fn start(&self) -> usize {
        self.start.load(Ordering::Relaxed) as usize
    }

    #[inline]
    pub(super) fn len(&self) -> usize {
        self.len.load(Ordering::Acquire) as usize
    }

    #[inline]
    pub(super) fn generation(&self) -> u32 {
        (self.version.load(Ordering::Relaxed) >> 32) as u32
    }

    /// The window as a reader sees it, with the version to confirm afterwards. An in-place
    /// append only grows `len`, published with a release store after its hashes, and needs no
    /// version; everything else that changes the window goes through `begin_update`.
    #[inline]
    pub(super) fn snapshot(&self) -> (Window, u64) {
        loop {
            let before = self.version.load(Ordering::Acquire);
            if before & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let window = Window {
                block: self.block.load(Ordering::Relaxed),
                base: self.base.load(Ordering::Relaxed),
                engine: self.engine.load(Ordering::Relaxed),
                len: self.len.load(Ordering::Acquire),
                children: self.children.load(Ordering::Relaxed),
                partials: self.partials.load(Ordering::Relaxed),
                forwards: self.forwards.load(Ordering::Relaxed),
            };
            fence(Ordering::Acquire);
            if self.version.load(Ordering::Relaxed) == before {
                return (window, before);
            }
        }
    }

    /// Whether everything read since the snapshot belongs to it.
    #[inline]
    pub(super) fn confirm(&self, version: u64) -> bool {
        fence(Ordering::Acquire);
        self.version.load(Ordering::Relaxed) == version
    }

    pub(super) fn begin_update(&self) {
        // SeqCst against the inserters' `inflight` increment: either they see the odd version and
        // back off, or the writer sees their count and waits (see `wait_inflight`).
        self.version.fetch_add(1, Ordering::SeqCst);
        // A release fence ahead of every store of the step: a reader that reads one of them
        // and then passes its acquire fence reads this step in its check of the version, and
        // tries again. The increment alone orders nothing for that reader.
        fence(Ordering::Release);
    }

    /// Wait for lock-free child inserts to finish; called inside a version step, so no new one
    /// starts meanwhile.
    pub(super) fn wait_inflight(&self) {
        while self.inflight.load(Ordering::SeqCst) != 0 {
            std::hint::spin_loop();
        }
    }

    pub(super) fn end_update(&self) {
        self.version.fetch_add(1, Ordering::Release);
    }

    /// Start a new life of this header: next generation, fields reset.
    pub(super) fn reincarnate(&self, start: usize, parent: u32, window: Window) {
        self.version.fetch_add((1 << 32) | 1, Ordering::Acquire);
        // As where a step begins: ahead of the stores of the new life.
        fence(Ordering::Release);
        self.start.store(start as u32, Ordering::Relaxed);
        self.parent.store(parent, Ordering::Relaxed);
        self.block.store(window.block, Ordering::Relaxed);
        self.base.store(window.base, Ordering::Relaxed);
        self.engine.store(window.engine, Ordering::Relaxed);
        self.len.store(window.len, Ordering::Relaxed);
        self.children.store(window.children, Ordering::Relaxed);
        self.partials.store(window.partials, Ordering::Relaxed);
        self.forwards.store(window.forwards, Ordering::Relaxed);
        self.end_update();
    }
}

pub(super) struct RunChunk {
    pub(super) runs: Box<[Run]>,
    /// Word 0 of every run's coverage (workers 0 to 63), made with the chunk.
    first: Box<[AtomicU64]>,
    /// Every further word of the runs' coverage, one array a word, made when a worker of
    /// that word first holds a whole run of this chunk. A word whose array is missing reads
    /// as zero, so a fleet that fits one word holds one word a run, not the index's ceiling.
    more: Box<[OnceLock<Box<[AtomicU64]>>]>,
}

/// The coverage words of one run: the workers that hold the whole run.
#[derive(Clone, Copy)]
pub(super) struct Coverage<'a> {
    chunk: &'a RunChunk,
    slot: usize,
}

impl<'a> Coverage<'a> {
    /// Words a run can hold: the index's ceiling of workers, in words.
    #[inline]
    pub(super) fn words(self) -> usize {
        1 + self.chunk.more.len()
    }

    /// Word `index`; zero where no worker of that word ever held a run of the chunk.
    #[inline]
    pub(super) fn word(self, index: usize) -> u64 {
        self.cell(index)
            .map_or(0, |word| word.load(Ordering::Relaxed))
    }

    #[inline]
    fn cell(self, index: usize) -> Option<&'a AtomicU64> {
        if index == 0 {
            return Some(&self.chunk.first[self.slot]);
        }
        self.chunk.more[index - 1]
            .get()
            .map(|words| &words[self.slot])
    }

    /// The cell of word `index`, its array made first where no worker of that word held a
    /// run of the chunk yet.
    #[inline]
    fn cell_or_make(self, index: usize) -> &'a AtomicU64 {
        if index == 0 {
            return &self.chunk.first[self.slot];
        }
        &self.chunk.more[index - 1].get_or_init(zeroed_words)[self.slot]
    }

    /// Every word, in order.
    pub(super) fn to_words(self) -> Vec<u64> {
        (0..self.words()).map(|index| self.word(index)).collect()
    }

    /// Store the first `words` words of this run into `to`, a run not yet in the tree: past
    /// the words a worker can hold a bit in, both sides read zero already.
    pub(super) fn copy_into(self, to: Coverage<'_>, words: usize) {
        for index in 0..words {
            let value = self.word(index);
            if value != 0 {
                to.cell_or_make(index).store(value, Ordering::Relaxed);
            } else if let Some(word) = to.cell(index) {
                word.store(0, Ordering::Relaxed);
            }
        }
    }
}

fn zeroed_words() -> Box<[AtomicU64]> {
    (0..RUN_CHUNK).map(|_| AtomicU64::new(0)).collect()
}

/// Run storage: a directory of fixed-size chunks created on first use, with a free list of dead
/// ids.
pub(super) struct RunSlab {
    pub(super) dir: Box<[OnceLock<RunChunk>]>,
    pub(super) next: AtomicU32,
    pub(super) free: SegQueue<u32>,
    pub(super) words: usize,
}

impl RunSlab {
    pub(super) fn new(words: usize) -> Self {
        Self {
            dir: (0..RUN_DIR).map(|_| OnceLock::new()).collect(),
            next: AtomicU32::new(0),
            free: SegQueue::new(),
            words,
        }
    }

    #[inline]
    pub(super) fn chunk(&self, index: usize) -> &RunChunk {
        self.dir[index].get_or_init(|| RunChunk {
            runs: (0..RUN_CHUNK).map(|_| Run::blank()).collect(),
            first: zeroed_words(),
            more: (1..self.words).map(|_| OnceLock::new()).collect(),
        })
    }

    #[inline]
    pub(super) fn run(&self, id: u32) -> &Run {
        let id = id as usize;
        &self.chunk(id >> RUN_CHUNK_BITS).runs[id & (RUN_CHUNK - 1)]
    }

    #[inline]
    pub(super) fn coverage(&self, id: u32) -> Coverage<'_> {
        let id = id as usize;
        Coverage {
            chunk: self.chunk(id >> RUN_CHUNK_BITS),
            slot: id & (RUN_CHUNK - 1),
        }
    }

    /// A run that is not yet reachable from the tree: a dead header given its next life, or a
    /// fresh one.
    pub(super) fn alloc(&self, start: usize, parent: u32, window: Window) -> u32 {
        if let Some(id) = self.free.pop() {
            let run = self.run(id);
            let mut meta = run.meta.lock();
            debug_assert!(meta.dead && coverage_is_empty(self.coverage(id), self.words));
            meta.dead = false;
            run.reincarnate(start, parent, window);
            return id;
        }
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        assert!(
            (id as usize) < RUN_DIR * RUN_CHUNK,
            "chain index slab exhausted: more than 2^26 runs"
        );
        let run = self.run(id);
        run.start.store(start as u32, Ordering::Relaxed);
        run.parent.store(parent, Ordering::Relaxed);
        run.block.store(window.block, Ordering::Relaxed);
        run.base.store(window.base, Ordering::Relaxed);
        run.engine.store(window.engine, Ordering::Relaxed);
        run.len.store(window.len, Ordering::Relaxed);
        run.children.store(window.children, Ordering::Relaxed);
        run.partials.store(window.partials, Ordering::Relaxed);
        run.forwards.store(window.forwards, Ordering::Relaxed);
        id
    }

    pub(super) fn allocated(&self) -> usize {
        self.next.load(Ordering::Relaxed) as usize
    }

    /// Bytes of chunks taken from the process allocator: the headers and the first coverage
    /// word of every slot, used or not, and every further word array a chunk has made.
    pub(super) fn chunk_bytes(&self) -> usize {
        let word_array = RUN_CHUNK * size_of::<AtomicU64>();
        self.dir
            .iter()
            .filter_map(OnceLock::get)
            .map(|chunk| {
                let made = chunk
                    .more
                    .iter()
                    .filter(|words| words.get().is_some())
                    .count();
                RUN_CHUNK * size_of::<Run>()
                    + (1 + made) * word_array
                    + chunk.more.len() * size_of::<OnceLock<Box<[AtomicU64]>>>()
            })
            .sum()
    }
}

#[inline]
pub(super) fn has(coverage: Coverage<'_>, worker: u32) -> bool {
    coverage.word((worker / 64) as usize) & (1u64 << (worker % 64)) != 0
}

/// By release, as [`clear`]: a later holder writes a word of a run it did not make, and a
/// reader may still hold the run from the life before. One that reads the word and then
/// passes its acquire fence reads, in its check of the version, the step that ended that
/// life.
pub(super) fn set(coverage: Coverage<'_>, worker: u32) {
    coverage
        .cell_or_make((worker / 64) as usize)
        .fetch_or(1u64 << (worker % 64), Ordering::Release);
}

/// A word whose array was never made holds no bit to clear, and none is made for it.
pub(super) fn clear(coverage: Coverage<'_>, worker: u32) {
    if let Some(word) = coverage.cell((worker / 64) as usize) {
        word.fetch_and(!(1u64 << (worker % 64)), Ordering::Release);
    }
}

/// Nobody, among the first `words` words: the words a worker can hold a bit in, or all of
/// `coverage.words()`.
pub(super) fn coverage_is_empty(coverage: Coverage<'_>, words: usize) -> bool {
    (0..words).all(|index| coverage.word(index) == 0)
}

/// Exactly `worker` and nobody else, among the first `words` words.
pub(super) fn covered_only_by(coverage: Coverage<'_>, worker: u32, words: usize) -> bool {
    let word = (worker / 64) as usize;
    let bit = 1u64 << (worker % 64);
    word < words
        && (0..words).all(|index| {
            let value = coverage.word(index);
            if index == word {
                value == bit
            } else {
                value == 0
            }
        })
}

pub(super) fn workers(coverage: Coverage<'_>) -> Vec<u32> {
    (0..coverage.words())
        .flat_map(|index| {
            let value = coverage.word(index);
            (0..64)
                .filter(move |bit| value & (1u64 << bit) != 0)
                .map(move |bit| (index * 64 + bit) as u32)
        })
        .collect()
}
