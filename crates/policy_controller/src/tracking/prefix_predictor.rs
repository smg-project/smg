//! The sent-prefix predictor (design 4.4): which workers each prefix of a
//! model's requests was sent to. A worker the store hears no cache events
//! from is remembered by what it was sent, and a decision reads that as it reads
//! the holders of the event-fed index ([`crate::store`]): the
//! deepest block of its request that a worker was sent, and the workers
//! sent it.
//!
//! An entry is a prediction made at the decision: it says where a prefix
//! was decided to go, whether or not the request then left. A
//! decision turned back behind its issue keeps its entry, and the next
//! request of the prefix follows it: a false entry costs one miss on its
//! worker, where no entry costs one miss on some worker, and it is true
//! once followed. An entry stops speaking for a worker whose cache events
//! are heard, and leaves with its worker or by the bound. The keys hold no
//! hash kind: a host that makes its own hashes keeps a text request's
//! apart from a token request's.
//!
//! This is the `tree` kind of the document's index settings
//! ([`IndexKind`](crate::document::spec::IndexKind)). The `hash` kind is not built.
//!
//! Nothing here is locked or ordered against another thread. A read takes
//! the predictor by a shared reference and writes nothing, so an owner may
//! lend it to many readers at once. A write takes it alone. A caller folds
//! a request's keys once, in a buffer of its own
//! ([`PrefixPredictor::fold_into`]), and hands them to the read and to the
//! write behind it.
//!
//! What is kept. A request carries its prefix as one hash a block of its
//! content ([`RequestFacts::block_hashes`](crate::engine::RequestFacts::block_hashes)).
//! The predictor names a block by a key folded over the hashes up to it,
//! so one key names one prefix, and keeps one entry a block: the block
//! before it, how many blocks follow it, and the workers sent it. The
//! blocks a worker was sent along one request are always a leading part of
//! the request. So the deepest block is found by halving: one probe for a
//! request sent before, and at most one more than the logarithm of its
//! blocks for any other. The fold itself is one multiplication a block.
//!
//! A block keeps the tick of its last use: the tick it was made at, and
//! the tick of each write that ended at it. The tick is the caller's clock
//! in steps of the caller's choice, and the predictor reads no clock. A
//! read says whether its deepest block is fresh at a tick
//! ([`DeepestPrefix::fresh`]), so an owner can pass no write for a request sent
//! a moment ago.
//!
//! What is forgotten. The bound is a count of placements, a block for a
//! worker. Blocks leave by last use, a block that none follows first. The
//! predictor lists every block a request ended at, oldest first. It takes
//! the oldest that no block follows, and the block before it then stands
//! where the taken one stood, unless a later request ended there. So a
//! path leaves from its deep end, a head that many requests share leaves
//! behind the last of them, and a worker's blocks along a request stay a
//! leading part of it. A block leaves with every worker sent it.
//!
//! A worker that left is skipped by the next read. Its placements are
//! taken out by the sweep, and a path no worker is left of leaves with
//! them from its deep end, a block a step: a call takes out as many
//! blocks as its budget and no more. A block that none follows and no
//! worker is left of waits in a list of its own, and the next step
//! takes it ahead of the oldest end. A worker's place is given to
//! another worker once a whole pass has run since it left.
//!
//! Memory: 32 bytes a block and 13 a bucket of its map, about 55 bytes
//! together: a map holds a key in halves, so no bucket is padded. A
//! block that more than one worker was sent keeps their
//! places in a list of its own: two bytes a worker, room for eight
//! places or for twice its workers at most, whichever is more, and the
//! list's own 24 bytes. A block left with one worker holds it in itself
//! again, and a list no block uses keeps room for eight places at most.
//! The blocks stand in pieces of 4,096 that never move, but for the
//! first: it starts at 64 and moves as it doubles to a whole piece, so a
//! model that sent a few blocks holds no room for 4,096. The keys stand
//! in 256 maps by their top byte, so a growth moves one map's entries.

use std::{
    hash::{Hash, Hasher},
    mem, slice,
};

use rustc_hash::FxHashMap;

use crate::engine::view::{fold_block, WorkerId};

/// No block: the end of a list, and the block before a request's first.
const NONE: u32 = u32::MAX;
/// In a block's links: the block is on neither list, the ends or the
/// blocks that wait.
const UNLISTED: u32 = u32::MAX - 1;
/// In a block's count of followers: the entry is free.
const FREE: u32 = u32::MAX;
/// No worker: the lone holder of a block whose holders are a group, or of
/// one that no worker is left of.
const NOBODY: u16 = u16::MAX;
/// The blocks of a whole piece of the slab.
const CHUNK: usize = 1 << 12;
/// The blocks the first piece starts with.
const FIRST: usize = 1 << 6;
/// The maps a key is found through, one for each value of its top byte.
const MAPS: usize = 1 << 8;
/// The places of room a list of workers keeps while no block uses it.
const POOLED: usize = 8;

/// The map a key is found through.
const fn map_of(key: u64) -> usize {
    (key >> 56) as usize
}

/// A block's key as a map holds it: in halves, so that the key and the
/// block's place fill an entry of twelve bytes. A whole word is aligned
/// to eight, and the entry is then padded to sixteen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Halves([u32; 2]);

impl Halves {
    const fn of(key: u64) -> Self {
        Self([key as u32, (key >> 32) as u32])
    }
}

impl Hash for Halves {
    /// One write of the whole key, as the word itself hashes.
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(u64::from(self.0[1]) << 32 | u64::from(self.0[0]));
    }
}

/// One block of a sent prefix.
#[derive(Clone, Copy, Debug)]
struct Block {
    /// The key of the prefix that ends at the block.
    key: u64,
    /// The block before it on its request, `NONE` at a first block. In a
    /// free entry, the next free one.
    before: u32,
    /// How many blocks have it before them; `FREE` in a free entry.
    after: u32,
    /// The next older block of the list of ends. Among the blocks that
    /// wait the links are places, not ages: this is the block linked
    /// after it. `UNLISTED` off both lists.
    older: u32,
    /// The next newer block of the list of ends, or the block linked
    /// before it among those that wait; `UNLISTED` off both lists.
    newer: u32,
    /// The group of the workers sent the block; `NONE` where one worker
    /// was, or none is left.
    group: u32,
    /// The one worker sent the block; `NOBODY` under a group, and where
    /// none is left.
    lone: u16,
    /// The caller's tick at the block's last use: when it was made, and at
    /// each write that ended at it.
    touched: u16,
}

impl Block {
    /// A free entry, the last of its list.
    const BLANK: Self = Self {
        key: 0,
        before: NONE,
        after: FREE,
        older: UNLISTED,
        newer: UNLISTED,
        group: NONE,
        lone: NOBODY,
        touched: 0,
    };

    /// Whether no worker is left of the block.
    const fn unheld(&self) -> bool {
        self.lone == NOBODY && self.group == NONE
    }

    /// Whether the block waits to leave: none follows it and no worker is
    /// left of it.
    const fn waits(&self) -> bool {
        self.after == 0 && self.unheld()
    }
}

/// What a probe asks of a block it finds.
#[derive(Clone, Copy)]
enum Ask<'a> {
    /// That it is there.
    Any,
    /// That a worker of the pool was sent it.
    Live,
    /// That a worker of the pool that the reader may use was sent it.
    Among(&'a dyn Fn(WorkerId) -> bool),
    /// That the worker of this place was sent it.
    Holds(u16),
}

/// The prefixes sent to a model's workers: what a decision reads for the
/// workers the store hears no cache events from, and what its issue writes.
#[derive(Debug)]
pub struct PrefixPredictor {
    /// The blocks, in pieces. A whole piece never moves; the first
    /// doubles to one.
    chunks: Vec<Box<[Block]>>,
    /// How many entries the pieces have handed out.
    span: u32,
    /// The first free entry; `NONE` where every entry handed out is held.
    free: u32,
    /// Where a block stands, by its key.
    maps: Box<[FxHashMap<Halves, u32>]>,
    /// The workers of the blocks that more than one was sent.
    groups: Vec<Vec<u16>>,
    /// The groups no block holds.
    free_groups: Vec<u32>,
    /// The oldest block of the list of ends.
    oldest: u32,
    /// The newest.
    newest: u32,
    /// The first of the blocks that wait: none follows such a block and
    /// no worker is left of it. They stand in a list of their own, and
    /// the next step takes one out.
    waiting: u32,
    /// The worker of each place.
    workers: Vec<WorkerId>,
    /// The place of each worker of the pool.
    places: FxHashMap<WorkerId, u16>,
    /// A bit a place: its worker is in the pool.
    live: Vec<u64>,
    /// The places no worker holds.
    free_places: Vec<u16>,
    /// The places of workers that left before the running pass began: the
    /// pass takes their placements out.
    clearing: Vec<u16>,
    /// The places of workers that left since: the next pass's.
    gone: Vec<u16>,
    /// The next entry the pass visits.
    cursor: u32,
    /// How many entries the pass has still to visit; `0` between passes.
    to_visit: u32,
    /// The most placements held.
    bound: usize,
    /// The placements held: a block for each worker sent it.
    placements: usize,
    /// The blocks held.
    blocks: usize,
    /// The placements a write did not make, for the bound or for places.
    dropped: u64,
}

/// What a read found: the deepest block of the request that a worker of
/// the pool was sent, and the workers sent it. A worker sent a shallower
/// block alone is not among them.
#[derive(Clone, Copy, Debug)]
pub struct DeepestPrefix<'a> {
    predictor: &'a PrefixPredictor,
    /// The block; `NONE` where no worker was sent the request's first.
    at: u32,
    blocks: u32,
}

impl<'a> DeepestPrefix<'a> {
    /// How many leading blocks of the request the workers were sent; `0`
    /// where none was sent its first block.
    pub fn blocks(&self) -> u32 {
        self.blocks
    }

    /// Whether the deepest block's last use was at `tick`: it was made at
    /// that tick, or a write at that tick ended at it. `false` where no
    /// worker was sent the request's first block.
    ///
    /// An owner passes no write where a read finds the whole request
    /// ([`Self::blocks`] is the count of its keys), its end fresh at the
    /// owner's tick and the picked worker among [`Self::holders`]: that
    /// write would move nothing a reader can see. The list of ends is then
    /// true to one tick. A tick is sixteen bits: a block last used 65,536
    /// ticks ago reads fresh once, and one move of the list is passed.
    pub fn fresh(&self, tick: u16) -> bool {
        self.at != NONE && self.predictor.block(self.at).touched == tick
    }

    /// The workers of the pool that were sent that many, in no order.
    pub fn holders(&self) -> impl Iterator<Item = WorkerId> + 'a {
        let predictor = self.predictor;
        let places: &'a [u16] = if self.at == NONE {
            &[]
        } else {
            predictor.holders(predictor.block(self.at))
        };
        places
            .iter()
            .copied()
            .filter(move |&place| place != NOBODY && predictor.is_live(place))
            .map(move |place| predictor.workers[place as usize])
    }
}

impl PrefixPredictor {
    /// A predictor that holds `bound` placements at most: a block sent to
    /// a worker is one, so a document's size bound divided by its block's
    /// size.
    pub fn new(bound: usize) -> Self {
        Self {
            chunks: Vec::new(),
            span: 0,
            free: NONE,
            maps: (0..MAPS).map(|_| FxHashMap::default()).collect(),
            groups: Vec::new(),
            free_groups: Vec::new(),
            oldest: NONE,
            newest: NONE,
            waiting: NONE,
            workers: Vec::new(),
            places: FxHashMap::default(),
            live: Vec::new(),
            free_places: Vec::new(),
            clearing: Vec::new(),
            gone: Vec::new(),
            cursor: 0,
            to_visit: 0,
            bound,
            placements: 0,
            blocks: 0,
            dropped: 0,
        }
    }

    /// The most placements held.
    pub fn bound(&self) -> usize {
        self.bound
    }

    /// Changes the bound in place. A lower bound forgets nothing here: the
    /// next writes and sweeps take the oldest blocks out until it holds.
    pub fn set_bound(&mut self, bound: usize) {
        self.bound = bound;
    }

    /// The placements held: a block for each worker sent it, the workers
    /// that left and are not yet swept among them.
    pub fn len(&self) -> usize {
        self.placements
    }

    /// Whether nothing is held.
    pub fn is_empty(&self) -> bool {
        self.placements == 0
    }

    /// The blocks held, each once however many workers were sent it.
    pub fn blocks(&self) -> usize {
        self.blocks
    }

    /// The placements the writes did not make: those past the bound that
    /// no old block made room for, and those of a worker past the 65,535
    /// the predictor tells apart.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// A request's keys into `keys`, one a block: each is the fold of the
    /// hashes up to its block under `seed`, the request's cache namespace
    /// ([`RequestFacts::namespace_seed`](crate::engine::RequestFacts::namespace_seed)).
    /// The same blocks under another seed are another prefix. `keys` is
    /// cleared and filled, and allocates only while it grows.
    pub fn fold_into(keys: &mut Vec<u64>, seed: u64, hashes: &[u64]) {
        keys.clear();
        let mut key = seed;
        keys.extend(hashes.iter().map(|&hash| {
            key = fold_block(key, hash);
            key
        }));
    }

    /// Reads a request by its keys ([`Self::fold_into`]): the deepest of
    /// its blocks that a worker of the pool was sent, and the workers sent
    /// it. Nothing is written.
    pub fn deepest(&self, keys: &[u64]) -> DeepestPrefix<'_> {
        let (blocks, at) = self.depth(keys, keys.len(), Ask::Live);
        DeepestPrefix {
            predictor: self,
            at,
            blocks: blocks as u32,
        }
    }

    /// [`Self::deepest`] for a reader that may use only some of the pool's
    /// workers: the deepest of the request's blocks that a worker `usable`
    /// passes was sent. A worker the reader cannot use, sent more of the
    /// request, hides no worker it can. [`DeepestPrefix::holders`] lists every
    /// worker sent that block, so the reader passes over the others once
    /// more. A block's workers are asked only where the read probes it.
    pub fn deepest_among(
        &self,
        keys: &[u64],
        usable: &dyn Fn(WorkerId) -> bool,
    ) -> DeepestPrefix<'_> {
        let (blocks, at) = self.depth(keys, keys.len(), Ask::Among(usable));
        DeepestPrefix {
            predictor: self,
            at,
            blocks: blocks as u32,
        }
    }

    /// Records a request, by its keys ([`Self::fold_into`]), as sent to
    /// `worker` at the caller's `tick`: the blocks past what the worker
    /// was already sent of it, and the request as the newest use of its
    /// path. A request the worker was sent before adds nothing. A block
    /// the write makes takes the tick, and so does the block it ends at.
    ///
    /// Below the bound, a write of a longer leading part of a request at
    /// one tick, behind a write of a shorter one at that tick, leaves what
    /// one write of the longer leaves, so an owner may write a long request
    /// in parts. At the bound each part makes its own room, so the parts
    /// may forget other blocks than one write would.
    ///
    /// At the bound the blocks that wait leave first, then the oldest that
    /// none follows, never a block of the request itself. The write takes
    /// at most two steps for each block it needs and eight more: a step
    /// takes one block out or passes over one that others follow. Where
    /// that makes no room for all of it, the request's leading blocks are
    /// written as far as the room goes and the rest is counted
    /// ([`PrefixPredictor::dropped`]).
    ///
    /// Call it only for a worker of the pool. A write gives its worker a
    /// place, and only [`PrefixPredictor::forget`] takes the place back: a
    /// write for a worker told gone gives it a place again, and nothing takes
    /// that one back until the worker is told gone once more.
    pub fn record(&mut self, keys: &[u64], worker: WorkerId, tick: u16) {
        let blocks = keys.len();
        if blocks == 0 {
            return;
        }
        let Some(place) = self.place_of(worker) else {
            self.dropped += blocks as u64;
            return;
        };
        // What anyone was sent of the request, and what the worker was.
        let (have, end) = self.depth(keys, blocks, Ask::Any);
        let (held, mut before) = self.depth(keys, have, Ask::Holds(place));
        let need = blocks - held;
        if need == 0 {
            self.end_at(before, tick);
            return;
        }
        // The request's own path is the newest before any block leaves for
        // it, and its deepest block is kept: the write stands on it. A
        // block that waited is an end again while the write runs.
        if end != NONE {
            if self.block(end).waits() {
                self.unlink_waiting(end);
                self.link_newest(end);
            } else {
                self.touch(end);
            }
        }
        self.make_room(need, end);
        let room = need.min(self.bound.saturating_sub(self.placements));
        self.dropped += (need - room) as u64;
        for (at, &key) in keys.iter().enumerate().take(held + room).skip(held) {
            before = match self.find(key) {
                Some(block) => {
                    if self.add(block, place) {
                        self.placements += 1;
                    }
                    block
                }
                None => {
                    let Some(block) = self.take(key, before, place, tick) else {
                        self.dropped += (held + room - at) as u64;
                        break;
                    };
                    self.placements += 1;
                    block
                }
            };
        }
        if before != NONE {
            self.end_at(before, tick);
        }
        // A block no worker is left of that the write stood on and gave
        // nothing waits again: the next step takes it.
        if end != NONE && self.block(end).waits() {
            self.unlink(end);
            self.link_waiting(end);
        }
    }

    /// Whether the record holds a place for the worker: it was sent a
    /// block and was not forgotten since.
    pub fn holds(&self, worker: WorkerId) -> bool {
        self.places.contains_key(&worker)
    }

    /// The workers the record holds a place for.
    pub fn held(&self) -> impl Iterator<Item = WorkerId> + '_ {
        self.places.keys().copied()
    }

    /// A worker left the pool, or was replaced: the next read skips it.
    /// Its placements are taken out by [`PrefixPredictor::sweep`]. A worker
    /// told gone that comes back under the same id starts with nothing.
    pub fn forget(&mut self, worker: WorkerId) {
        let Some(place) = self.places.remove(&worker) else {
            return;
        };
        self.live[place as usize / 64] &= !(1 << (place % 64));
        self.gone.push(place);
        if self.to_visit == 0 {
            self.begin_pass();
        }
    }

    /// The work that waits, `budget` steps of it at most, on the decider's
    /// thread as it leaves a turn. A block that none follows and no worker
    /// is left of leaves first. The oldest blocks leave while more
    /// placements are held than the bound. The placements of the workers
    /// that left are taken out, and a path no worker is left of leaves
    /// from its deep end. A step visits one entry, takes one block out or
    /// passes over one that others follow, so a call takes out `budget`
    /// blocks at most. Returns the steps taken: fewer than `budget` where
    /// nothing waits.
    pub fn sweep(&mut self, budget: usize) -> usize {
        let mut steps = 0;
        while steps < budget {
            if (self.waiting != NONE || self.placements > self.bound) && self.forget_one(NONE) {
                steps += 1;
                continue;
            }
            if self.to_visit == 0 {
                break;
            }
            self.clear_next();
            steps += 1;
        }
        steps
    }

    fn block(&self, at: u32) -> &Block {
        &self.chunks[at as usize / CHUNK][at as usize % CHUNK]
    }

    fn block_mut(&mut self, at: u32) -> &mut Block {
        &mut self.chunks[at as usize / CHUNK][at as usize % CHUNK]
    }

    fn find(&self, key: u64) -> Option<u32> {
        self.maps[map_of(key)].get(&Halves::of(key)).copied()
    }

    fn is_live(&self, place: u16) -> bool {
        self.live[place as usize / 64] >> (place % 64) & 1 == 1
    }

    /// The places of the workers sent a block, those that left and are not
    /// yet swept among them; a lone `NOBODY` where none is left.
    fn holders<'a>(&'a self, block: &'a Block) -> &'a [u16] {
        match block.group {
            NONE => slice::from_ref(&block.lone),
            group => &self.groups[group as usize],
        }
    }

    fn passes(&self, block: &Block, ask: Ask<'_>) -> bool {
        let holders = self.holders(block);
        match ask {
            Ask::Any => true,
            Ask::Holds(place) => holders.contains(&place),
            Ask::Among(usable) => holders.iter().any(|&place| {
                place != NOBODY && self.is_live(place) && usable(self.workers[place as usize])
            }),
            // Between two sweeps of leavers every holder is of the pool.
            Ask::Live if self.clearing.is_empty() && self.gone.is_empty() => {
                holders.first().is_some_and(|&place| place != NOBODY)
            }
            Ask::Live => holders
                .iter()
                .any(|&place| place != NOBODY && self.is_live(place)),
        }
    }

    /// The block of the request's key at place `at`, where it passes.
    fn probe(&self, keys: &[u64], at: usize, ask: Ask<'_>) -> Option<u32> {
        let block = self.find(keys[at])?;
        self.passes(self.block(block), ask).then_some(block)
    }

    /// How many of the request's first `within` blocks pass, and the last
    /// of them. The blocks that pass are a leading part of the request, so
    /// the last one first, then by halves.
    fn depth(&self, keys: &[u64], within: usize, ask: Ask<'_>) -> (usize, u32) {
        let Some(last) = within.checked_sub(1) else {
            return (0, NONE);
        };
        if let Some(block) = self.probe(keys, last, ask) {
            return (within, block);
        }
        let (mut low, mut high, mut found) = (0, last, NONE);
        while low < high {
            let middle = low + (high - low) / 2;
            match self.probe(keys, middle, ask) {
                Some(block) => {
                    low = middle + 1;
                    found = block;
                }
                None => high = middle,
            }
        }
        (low, found)
    }

    /// The worker's place, a new one where it has none; `None` past the
    /// places a block can name.
    fn place_of(&mut self, worker: WorkerId) -> Option<u16> {
        if let Some(&place) = self.places.get(&worker) {
            return Some(place);
        }
        let place = match self.free_places.pop() {
            Some(place) => {
                self.workers[place as usize] = worker;
                place
            }
            None => {
                let place = u16::try_from(self.workers.len()).ok()?;
                if place == NOBODY {
                    return None;
                }
                self.workers.push(worker);
                if place as usize / 64 == self.live.len() {
                    self.live.push(0);
                }
                place
            }
        };
        self.live[place as usize / 64] |= 1 << (place % 64);
        self.places.insert(worker, place);
        Some(place)
    }

    /// Adds a worker to a block's holders; whether it was not one.
    fn add(&mut self, at: u32, place: u16) -> bool {
        let block = *self.block(at);
        if block.group != NONE {
            let group = &mut self.groups[block.group as usize];
            if group.contains(&place) {
                return false;
            }
            group.push(place);
            return true;
        }
        if block.lone == place {
            return false;
        }
        if block.lone == NOBODY {
            self.block_mut(at).lone = place;
            return true;
        }
        let group = match self.free_groups.pop() {
            Some(group) => group,
            None => {
                self.groups.push(Vec::new());
                (self.groups.len() - 1) as u32
            }
        };
        self.groups[group as usize].extend([block.lone, place]);
        let block = self.block_mut(at);
        block.group = group;
        block.lone = NOBODY;
        true
    }

    /// A new block of one worker behind `before`, made at `tick`; `None`
    /// past the entries a link can name.
    fn take(&mut self, key: u64, before: u32, place: u16, tick: u16) -> Option<u32> {
        let at = if self.free == NONE {
            if self.span == UNLISTED {
                return None;
            }
            let span = self.span as usize;
            if span.is_multiple_of(CHUNK) {
                let room = if span == 0 { FIRST } else { CHUNK };
                self.chunks
                    .push(vec![Block::BLANK; room].into_boxed_slice());
            } else if span < CHUNK {
                let first = self.chunks.first_mut();
                if let Some(first) = first.filter(|first| first.len() == span) {
                    let mut grown = mem::take(first).into_vec();
                    grown.resize(span * 2, Block::BLANK);
                    *first = grown.into_boxed_slice();
                }
            }
            self.span += 1;
            self.span - 1
        } else {
            let at = self.free;
            self.free = self.block(at).before;
            at
        };
        *self.block_mut(at) = Block {
            key,
            before,
            after: 0,
            lone: place,
            touched: tick,
            ..Block::BLANK
        };
        self.maps[map_of(key)].insert(Halves::of(key), at);
        self.blocks += 1;
        if before != NONE {
            let block = self.block_mut(before);
            block.after += 1;
            // A block that gains its first follower is an end no longer.
            if block.after == 1 && block.older != UNLISTED {
                self.unlink(before);
            }
        }
        Some(at)
    }

    fn link_newest(&mut self, at: u32) {
        let newest = mem::replace(&mut self.newest, at);
        let block = self.block_mut(at);
        block.older = newest;
        block.newer = NONE;
        if newest == NONE {
            self.oldest = at;
        } else {
            self.block_mut(newest).newer = at;
        }
    }

    fn link_oldest(&mut self, at: u32) {
        let oldest = mem::replace(&mut self.oldest, at);
        let block = self.block_mut(at);
        block.newer = oldest;
        block.older = NONE;
        if oldest == NONE {
            self.newest = at;
        } else {
            self.block_mut(oldest).older = at;
        }
    }

    fn unlink(&mut self, at: u32) {
        let block = self.block_mut(at);
        let older = mem::replace(&mut block.older, UNLISTED);
        let newer = mem::replace(&mut block.newer, UNLISTED);
        if older == NONE {
            self.oldest = newer;
        } else {
            self.block_mut(older).newer = newer;
        }
        if newer == NONE {
            self.newest = older;
        } else {
            self.block_mut(newer).older = older;
        }
    }

    /// Puts a block that waits first on its list.
    fn link_waiting(&mut self, at: u32) {
        let first = mem::replace(&mut self.waiting, at);
        let block = self.block_mut(at);
        block.older = NONE;
        block.newer = first;
        if first != NONE {
            self.block_mut(first).older = at;
        }
    }

    /// Takes a block off the list of those that wait.
    fn unlink_waiting(&mut self, at: u32) {
        let block = self.block_mut(at);
        let older = mem::replace(&mut block.older, UNLISTED);
        let newer = mem::replace(&mut block.newer, UNLISTED);
        if older == NONE {
            self.waiting = newer;
        } else {
            self.block_mut(older).newer = newer;
        }
        if newer != NONE {
            self.block_mut(newer).older = older;
        }
    }

    /// A request ended at the block: it is the newest of the list.
    fn touch(&mut self, at: u32) {
        if self.newest == at {
            return;
        }
        if self.block(at).older != UNLISTED {
            self.unlink(at);
        }
        self.link_newest(at);
    }

    /// A write at `tick` ended at the block: it is the newest of the list,
    /// and its last use is that tick.
    fn end_at(&mut self, at: u32, tick: u16) {
        self.touch(at);
        self.block_mut(at).touched = tick;
    }

    /// One step of forgetting, never past `keep`. A block that waits
    /// leaves first. With none, the step is at the old end of the list: a
    /// block that others follow leaves the list, one that none follows
    /// leaves the predictor. Whether a step was taken.
    fn forget_one(&mut self, keep: u32) -> bool {
        let at = if self.waiting == NONE {
            self.oldest
        } else {
            debug_assert_ne!(self.waiting, keep, "a write's block is an end as it runs");
            self.take_waiting()
        };
        if at == NONE || at == keep {
            return false;
        }
        if self.block(at).after == 0 {
            self.drop_block(at, keep);
        } else {
            self.unlink(at);
        }
        true
    }

    /// Takes the first block that waits off its list, for the step to
    /// take out. A block waits only behind a worker that left, so this
    /// stands out of a write's own line.
    #[cold]
    #[inline(never)]
    fn take_waiting(&mut self) -> u32 {
        let at = self.waiting;
        self.unlink_waiting(at);
        at
    }

    /// Room for `need` placements of a write that stands on `keep`: at
    /// most two steps of forgetting for each, and eight more.
    fn make_room(&mut self, need: usize, keep: u32) {
        let mut steps = 2 * need + 8;
        while steps != 0 && self.placements + need > self.bound && self.forget_one(keep) {
            steps -= 1;
        }
    }

    /// Takes out a block that none follows, with every worker sent it. A
    /// block that waited is off its list already: the caller took it.
    /// The block before it, where this leaves it with no follower, takes
    /// its place at the old end unless a later request ended there. Where
    /// no worker is left of that block it waits instead, whatever ended
    /// there, and the next step takes it: a path no worker is left of
    /// leaves a block a step. A block a write stands on (`keep`) stays
    /// where it is.
    fn drop_block(&mut self, at: u32, keep: u32) {
        let block = *self.block(at);
        debug_assert_eq!(block.after, 0, "a block that others follow stays");
        if block.older != UNLISTED {
            self.unlink(at);
        }
        self.maps[map_of(block.key)].remove(&Halves::of(block.key));
        if block.group == NONE {
            self.placements -= usize::from(block.lone != NOBODY);
        } else {
            let group = &mut self.groups[block.group as usize];
            self.placements -= group.len();
            if group.capacity() > POOLED {
                *group = Vec::new();
            } else {
                group.clear();
            }
            self.free_groups.push(block.group);
        }
        *self.block_mut(at) = Block {
            before: self.free,
            ..Block::BLANK
        };
        self.free = at;
        self.blocks -= 1;
        if block.before == NONE {
            return;
        }
        let before = self.block_mut(block.before);
        before.after -= 1;
        if before.after != 0 || block.before == keep {
            return;
        }
        let (unheld, listed) = (before.unheld(), before.older != UNLISTED);
        if unheld {
            if listed {
                self.unlink(block.before);
            }
            self.link_waiting(block.before);
        } else if !listed {
            self.link_oldest(block.before);
        }
    }

    /// A pass over every entry held now: it takes out the placements of
    /// the workers that left before it began. A pass over no entry ends
    /// at once.
    fn begin_pass(&mut self) {
        mem::swap(&mut self.clearing, &mut self.gone);
        self.cursor = 0;
        self.to_visit = self.span;
        if self.to_visit == 0 {
            self.end_pass();
        }
    }

    /// The pass visited every entry: no block names the places it
    /// cleared, so each is free for another worker. The workers that left
    /// meanwhile are the next pass's.
    fn end_pass(&mut self) {
        self.free_places.append(&mut self.clearing);
        if !self.gone.is_empty() {
            self.begin_pass();
        }
    }

    /// The pass's next entry: the workers that left are taken out of its
    /// holders, and a block left with no worker and no follower goes.
    fn clear_next(&mut self) {
        debug_assert_eq!(self.waiting, NONE, "the sweep takes what waits first");
        let at = self.cursor;
        self.cursor += 1;
        self.to_visit -= 1;
        let block = *self.block(at);
        if block.after != FREE {
            if block.group == NONE {
                if block.lone != NOBODY && !self.is_live(block.lone) {
                    self.block_mut(at).lone = NOBODY;
                    self.placements -= 1;
                }
            } else {
                let mut group = mem::take(&mut self.groups[block.group as usize]);
                let held = group.len();
                group.retain(|&place| self.is_live(place));
                self.placements -= held - group.len();
                if group.len() < 2 {
                    // A block left with one worker holds it in itself
                    // again, and the list's room goes back.
                    self.free_groups.push(block.group);
                    let lone = group.first().copied().unwrap_or(NOBODY);
                    let block = self.block_mut(at);
                    (block.lone, block.group) = (lone, NONE);
                } else {
                    if group.capacity() > 2 * group.len() {
                        group.shrink_to_fit();
                    }
                    self.groups[block.group as usize] = group;
                }
            }
            // No block waits while the pass visits, so one the visit
            // leaves with no worker and no follower is an end of the
            // list, and goes at once.
            if self.block(at).waits() {
                self.drop_block(at, NONE);
            }
        }
        if self.to_visit == 0 {
            self.end_pass();
        }
    }
}

#[cfg(test)]
mod tests;
