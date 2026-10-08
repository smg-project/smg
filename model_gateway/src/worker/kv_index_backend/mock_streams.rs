//! Engine streams for the index tests and the decision bench, generated
//! in-process by the mock engine (`mock_worker::engine`, the simulator the
//! mock fleet runs). A seeded mix of shared chat-template prefixes,
//! multi-turn sessions and prompts nobody shares drives one or two engines
//! whose prefix cache fills, evicts and announces every change as KV events:
//! stores chained to their parents, prefixes shared across requests and
//! ranks, a cache that turns over many times in a few hundred requests. Each
//! batch leaves its engine on a publisher wire, vLLM's or SGLang's event
//! layout by the mock worker's own encoder, and comes back through the
//! relay's decoder and normalizer, so what reaches the index is what reaches
//! it in production, with nothing recorded and checked in.
//!
//! Beside the payloads the generator keeps the engine's own truth: at
//! checkpoints, taken when the engine has published everything it produced,
//! the keys of the blocks any rank holds on any tier. The index's prefix
//! match for a prompt must equal the engine's at every one of them: the hit
//! a routing decision predicts against the hit the engine serves.

use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    time::Duration,
};

use engine_servicer::kv_wire::{Normalizer, WireBatch};
use engine_zmq_client::codec::TrailingTolerant;
use futures::{
    future::LocalBoxFuture,
    stream::{select_all, FuturesUnordered},
    FutureExt, StreamExt,
};
use mock_worker::{
    engine::{Engine, EngineParams, GenEvent, NewRequest},
    kv_zmq::{encode_batch, Wire},
};
use serde_json::{json, Value};
use smg_grpc_client::common_proto::{
    kv_cache_event, KvBlock, KvBlocksRemoved, KvBlocksStored, KvCacheEvent, KvEventBatch,
};
use tokio::{runtime::Builder, sync::mpsc, time::timeout};

/// The engines' page size in tokens.
pub const BLOCK: usize = 16;
/// KV capacity in blocks: a few hundred requests turn it over many times.
const CAPACITY_BLOCKS: u64 = 256;
/// Blocks the host tier holds before it evicts, oldest write-back first.
const HOST_BLOCKS: usize = 96;
/// Requests in flight at once.
const IN_FLIGHT: usize = 6;
/// Checkpoints asked for per stream, spread over the completed requests.
const CHECKPOINTS: usize = 32;
/// Sessions kept at once; a new one beyond this replaces one at random.
const SESSIONS: usize = 48;
/// A session this long, in blocks, ends; the next turn starts a new one.
const SESSION_BLOCKS: usize = 96;
/// Simulated time the engines get to finish every request.
const SIMULATED_LIMIT: Duration = Duration::from_secs(3600);

/// The engines and the wire a stream comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// One vLLM engine on vLLM's event layout (unsigned hashes; medium,
    /// cache group and spec kind on every event; the rank in the batch),
    /// with vLLM's second physical copies of blocks two requests computed
    /// at once (see [`SecondCopies`]), restarted once part-way
    /// (`AllBlocksCleared`).
    Vllm,
    /// One SGLang engine on SGLang's layout (signed hashes, one removal per
    /// node, no medium, no rank), scheduling prefill first.
    Sglang,
    /// Two data-parallel ranks of one vLLM worker, their streams merged by
    /// receive order. Sessions stick to a rank but a quarter of their turns
    /// cross over, so both ranks hold the same blocks and a block's last
    /// copy may leave from either.
    TwoRank,
    /// One SGLang engine with a host tier: a block leaving the device is
    /// written back to the host first and evicted from there later, so the
    /// index must hold a block until its last copy on any tier goes.
    HostTier,
}

impl Shape {
    pub fn name(self) -> &'static str {
        match self {
            Shape::Vllm => "vllm",
            Shape::Sglang => "sglang",
            Shape::TwoRank => "vllm-dp2",
            Shape::HostTier => "sglang-hicache",
        }
    }

    /// The publisher layout; the mock's SGLang layout carries no rank, so
    /// the two-rank run rides vLLM's `data_parallel_rank`.
    fn wire(self) -> Wire {
        match self {
            Shape::Vllm | Shape::TwoRank => Wire::Vllm,
            Shape::Sglang | Shape::HostTier => Wire::Sglang,
        }
    }

    fn ranks(self) -> usize {
        match self {
            Shape::TwoRank => 2,
            Shape::Vllm | Shape::Sglang | Shape::HostTier => 1,
        }
    }
}

/// One publisher payload as the wire carries it, and the rank it came from.
pub struct Payload {
    pub rank: usize,
    pub bytes: Vec<u8>,
}

/// The engine's truth once `after` payloads are applied: the keys of the
/// blocks any rank holds on any tier, as [`Engine::block_keys`] names them.
pub struct Checkpoint {
    pub after: usize,
    pub held: HashSet<u64>,
}

/// A generated stream: its payloads in receive order, every prompt the
/// requests carried, and the checkpoints.
pub struct Stream {
    pub payloads: Vec<Payload>,
    pub prompts: Vec<Vec<u32>>,
    pub checkpoints: Vec<Checkpoint>,
}

impl Stream {
    /// `requests` seeded requests through the shape's engines, on a paused
    /// clock: the engines' pass times are simulated, not slept.
    pub fn generate(shape: Shape, seed: u64, requests: usize) -> Self {
        let runtime = Builder::new_current_thread()
            .enable_all()
            .start_paused(true)
            .build()
            .expect("a current-thread runtime");
        runtime.block_on(async {
            timeout(SIMULATED_LIMIT, Generator::new(shape, seed, requests).run())
                .await
                .expect("the engines finish every request within the simulated limit")
        })
    }

    /// The stream as the relay forwards it to one worker: every payload
    /// decoded and normalized in order, one normalizer for every rank.
    pub fn normalized(&self) -> Vec<KvEventBatch> {
        let mut normalizer = Normalizer::new();
        let mut event_id = 0;
        self.payloads
            .iter()
            .enumerate()
            .map(|(seq, payload)| {
                let batch = rmp_serde::from_slice::<TrailingTolerant<WireBatch>>(&payload.bytes)
                    .expect("a msgpack event batch")
                    .0;
                normalizer.normalize_batch(batch, seq as u64, &mut event_id)
            })
            .collect()
    }

    /// The engine's prefix match for a prompt against a checkpoint's blocks:
    /// its leading full blocks held, as the engine counts a cache hit.
    pub fn prefix_match(held: &HashSet<u64>, prompt: &[u32]) -> usize {
        Engine::block_keys(prompt, BLOCK)
            .iter()
            .take_while(|key| held.contains(key))
            .count()
    }
}

/// SplitMix64: one seed, one request mix.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// Uniform in `[lo, hi]`.
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }

    fn chance(&mut self, numerator: u64, denominator: u64) -> bool {
        self.next() % denominator < numerator
    }
}

/// Token ids of block `position` of token stream `stream`: distinct per pair.
fn tokens(stream: u64, position: usize) -> Vec<u32> {
    (0..BLOCK as u32)
        .map(|i| {
            let word = stream
                .wrapping_mul(0x2545_f491_4f6c_dd1d)
                .wrapping_add(position as u64 * 0x9e37_79b9)
                .wrapping_add(u64::from(i));
            (word >> 7) as u32 & 0x3_ffff
        })
        .collect()
}

/// A conversation: the tokens its next turn continues from, its rank, and a
/// generation so a turn finishing after the slot was reused does not land.
struct Session {
    tokens: Vec<u32>,
    rank: usize,
    generation: u64,
}

/// A request the engine is running: the session (slot, generation) it
/// extends, and its prompt, which with the output becomes the session's
/// next starting point.
struct InFlight {
    session: Option<(usize, u64)>,
    prompt: Vec<u32>,
}

/// A drawn request: its rank, its prompt, and the session (slot, generation)
/// it extends.
struct Draw {
    rank: usize,
    prompt: Vec<u32>,
    session: Option<(usize, u64)>,
}

/// A request's index and output tokens, once it is done.
type Completion = LocalBoxFuture<'static, (usize, Vec<u32>)>;

struct Generator {
    shape: Shape,
    rng: Rng,
    requests: usize,
    prefixes: Vec<Vec<u32>>,
    sessions: Vec<Session>,
    generations: u64,
    /// The next token stream nobody has used (see [`tokens`]).
    next_stream: u64,
    engines: Vec<Engine>,
    in_flight: HashMap<usize, InFlight>,
    submitted: usize,
    completed: usize,
    /// The last sequence number received from each rank.
    received: Vec<u64>,
    checkpoint_due: bool,
    host: Option<HostTier>,
    copies: Option<SecondCopies>,
    stream: Stream,
}

impl Generator {
    fn new(shape: Shape, seed: u64, requests: usize) -> Self {
        let mut rng = Rng(seed);
        let prefixes = (0..6)
            .map(|p| {
                let len = rng.range(2, 6);
                (0..len)
                    .flat_map(|position| tokens(1_000 + p, position))
                    .collect()
            })
            .collect();
        let params = EngineParams {
            kv_capacity_tokens: CAPACITY_BLOCKS * BLOCK as u64,
            block_size: BLOCK as u32,
            max_running: 8,
            max_batched_tokens: 2048,
            prefill_first: matches!(shape, Shape::Sglang | Shape::HostTier),
            kv_broadcast_capacity: 4096,
            ..Default::default()
        };
        let engines = (0..shape.ranks())
            .map(|rank| {
                Engine::spawn_named(
                    params.clone(),
                    format!("{}-rank{rank}", shape.name()),
                    false,
                )
            })
            .collect();
        let host = (shape == Shape::HostTier).then(|| HostTier::new(seed));
        let copies = (shape == Shape::Vllm).then(|| SecondCopies::new(seed));
        Self {
            shape,
            rng,
            requests,
            prefixes,
            sessions: Vec::new(),
            generations: 0,
            next_stream: 1_000_000,
            engines,
            in_flight: HashMap::new(),
            submitted: 0,
            completed: 0,
            received: vec![0; shape.ranks()],
            checkpoint_due: false,
            host,
            copies,
            stream: Stream {
                payloads: Vec::new(),
                prompts: Vec::new(),
                checkpoints: Vec::new(),
            },
        }
    }

    async fn run(mut self) -> Stream {
        let mut kv = select_all(
            self.engines
                .iter()
                .enumerate()
                .map(|(rank, engine)| engine.subscribe_kv(0).map(move |item| (rank, item))),
        );
        let mut completions: FuturesUnordered<Completion> = FuturesUnordered::new();
        self.fill(&mut completions);
        while self.completed < self.requests {
            tokio::select! {
                biased;
                next = kv.next() => {
                    let Some((rank, Ok(batch))) = next else { break };
                    self.receive(rank, batch);
                }
                Some((index, output)) = completions.next(), if !completions.is_empty() => {
                    self.finish(index, output);
                    self.fill(&mut completions);
                }
            }
        }
        // The final truth is read once everything produced has been received.
        while !self.caught_up() {
            let Some((rank, Ok(batch))) = kv.next().await else {
                break;
            };
            self.receive(rank, batch);
        }
        self.checkpoint();
        // Dropping the handles ends the engines; their streams end once the
        // publishers have released what they still hold.
        self.engines.clear();
        while let Some((rank, Ok(batch))) = kv.next().await {
            self.receive(rank, batch);
        }
        self.stream
    }

    /// A batch from `rank`: the second copies folded in, the host tier's
    /// write-back ahead of it, then the batch on the shape's wire; a due
    /// checkpoint once the engines are caught up.
    fn receive(&mut self, rank: usize, mut batch: KvEventBatch) {
        self.received[rank] = batch.sequence_number;
        if let Some(copies) = &mut self.copies {
            copies.transform(&mut batch);
        }
        if let Some(bytes) = self.host.as_mut().and_then(|host| host.write_back(&batch)) {
            self.stream.payloads.push(Payload { rank, bytes });
        }
        let bytes = encode_batch(&batch, rank as i32, self.shape.wire());
        self.stream.payloads.push(Payload { rank, bytes });
        if self.checkpoint_due && self.caught_up() {
            self.checkpoint();
            self.checkpoint_due = false;
        }
    }

    /// Whether every batch the engines produced has been received. The
    /// actor publishes a pass's batch, its cache mirror and its load
    /// snapshot together after the pass's simulated time, so an equal count
    /// means the engine's cache is exactly what the received batches
    /// describe.
    fn caught_up(&self) -> bool {
        self.engines
            .iter()
            .zip(&self.received)
            .all(|(engine, &received)| {
                u64::try_from(engine.load().num_kv_batches).unwrap_or(0) == received
            })
    }

    fn checkpoint(&mut self) {
        let after = self.stream.payloads.len();
        if self
            .stream
            .checkpoints
            .last()
            .is_some_and(|last| last.after == after)
        {
            return;
        }
        let mut held: HashSet<u64> = self.engines.iter().flat_map(Engine::cache_keys).collect();
        if let Some(host) = &self.host {
            held.extend(host.keys());
        }
        if let Some(copies) = &self.copies {
            held.extend(copies.keys());
        }
        self.stream.checkpoints.push(Checkpoint { after, held });
    }

    /// A request finished: its session continues from prompt and output; a
    /// checkpoint falls due every so many completions; the vLLM run restarts
    /// its engine once part-way.
    fn finish(&mut self, index: usize, output: Vec<u32>) {
        self.completed += 1;
        let flight = self.in_flight.remove(&index).expect("a request in flight");
        if let Some((slot, generation)) = flight.session {
            if let Some(session) = self
                .sessions
                .get_mut(slot)
                .filter(|session| session.generation == generation)
            {
                session.tokens = flight.prompt;
                session.tokens.extend(output);
            }
        }
        if self
            .completed
            .is_multiple_of((self.requests / CHECKPOINTS).max(1))
        {
            self.checkpoint_due = true;
        }
        if self.shape == Shape::Vllm && self.completed == self.requests * 3 / 5 {
            self.engines[0].reset();
        }
    }

    /// Keep [`IN_FLIGHT`] requests running until every request is submitted.
    fn fill(&mut self, completions: &mut FuturesUnordered<Completion>) {
        while self.submitted < self.requests && self.in_flight.len() < IN_FLIGHT {
            let index = self.submitted;
            self.submitted += 1;
            let Draw {
                rank,
                prompt,
                session,
            } = self.draw();
            let max_new = self.rng.range(8, 64) as u32;
            let (events, mut receiver) = mpsc::unbounded_channel();
            self.engines[rank].submit(NewRequest {
                request_id: format!("r{index}"),
                prompt_token_ids: prompt.clone(),
                max_new,
                events,
            });
            self.stream.prompts.push(prompt.clone());
            self.in_flight.insert(index, InFlight { session, prompt });
            completions.push(
                async move {
                    let mut output = Vec::new();
                    while let Some(event) = receiver.recv().await {
                        match event {
                            GenEvent::Token { token_id, .. } => output.push(token_id),
                            GenEvent::Done { .. } => break,
                        }
                    }
                    (index, output)
                }
                .boxed_local(),
            );
        }
    }

    /// The next request: a turn on a session (55 in 100), a new session (30),
    /// or a prompt nobody shares (15); with its rank and the session it
    /// extends.
    fn draw(&mut self) -> Draw {
        let roll = self.rng.below(100);
        if roll < 55 {
            if let Some(turn) = self.next_turn() {
                return turn;
            }
        }
        if roll < 85 {
            return self.new_session();
        }
        self.novel()
    }

    /// A new turn on a session: its tokens so far (the previous turns and
    /// their outputs), one to six blocks of new user tokens and a partial
    /// block. A session that has grown long is replaced by a new one in its
    /// slot. On two ranks a quarter of the turns go to the other rank.
    fn next_turn(&mut self) -> Option<Draw> {
        if self.sessions.is_empty() {
            return None;
        }
        let slot = self.rng.below(self.sessions.len());
        if self.sessions[slot].tokens.len() >= SESSION_BLOCKS * BLOCK {
            return Some(self.new_session_at(slot));
        }
        let mut rank = self.sessions[slot].rank;
        if self.shape.ranks() > 1 && self.rng.chance(1, 4) {
            rank = (rank + 1) % self.shape.ranks();
        }
        let mut prompt = self.sessions[slot].tokens.clone();
        self.append_blocks(&mut prompt, 1, 6);
        Some(Draw {
            rank,
            prompt,
            session: Some((slot, self.sessions[slot].generation)),
        })
    }

    /// A new session in a free slot, or in place of one at random.
    fn new_session(&mut self) -> Draw {
        let slot = if self.sessions.len() < SESSIONS {
            self.sessions.len()
        } else {
            self.rng.below(SESSIONS)
        };
        self.new_session_at(slot)
    }

    /// A new session on a shared prefix (three in four) or on its own: one
    /// to twelve blocks of body and a partial block, on a random rank.
    fn new_session_at(&mut self, slot: usize) -> Draw {
        let mut prompt = if self.rng.chance(3, 4) {
            self.prefixes[self.rng.below(self.prefixes.len())].clone()
        } else {
            Vec::new()
        };
        self.append_blocks(&mut prompt, 1, 12);
        let rank = self.rng.below(self.shape.ranks());
        self.generations += 1;
        let session = Session {
            tokens: prompt.clone(),
            rank,
            generation: self.generations,
        };
        if slot < self.sessions.len() {
            self.sessions[slot] = session;
        } else {
            self.sessions.push(session);
        }
        Draw {
            rank,
            prompt,
            session: Some((slot, self.generations)),
        }
    }

    /// A prompt nobody shares: one to sixteen blocks and a partial block.
    fn novel(&mut self) -> Draw {
        let mut prompt = Vec::new();
        self.append_blocks(&mut prompt, 1, 16);
        Draw {
            rank: self.rng.below(self.shape.ranks()),
            prompt,
            session: None,
        }
    }

    /// `lo` to `hi` fresh blocks and a partial block (none to all but one of
    /// its tokens) onto `prompt`.
    fn append_blocks(&mut self, prompt: &mut Vec<u32>, lo: usize, hi: usize) {
        let stream = self.next_stream;
        self.next_stream += 1;
        let blocks = self.rng.range(lo, hi);
        for position in 0..blocks {
            prompt.extend(tokens(stream, position));
        }
        let partial = self.rng.below(BLOCK);
        prompt.extend(&tokens(stream, blocks)[..partial]);
    }
}

/// A host tier behind an engine, as a hierarchical cache keeps one: a block
/// the device evicts is written back to the host first (two in three, when
/// its parent is held somewhere or it is a root), on a batch of its own
/// ahead of the device's; the host evicts its oldest write-backs beyond its
/// capacity. Its events ride SGLang's layout with `medium: "CPU"`.
struct HostTier {
    rng: Rng,
    /// Tokens and parent of every block the device stored, by engine hash.
    blocks: HashMap<i64, (Vec<u32>, Option<i64>)>,
    /// What the device holds, by the stream so far.
    device: HashSet<i64>,
    /// What the host holds, and the order it was written back in.
    held: HashSet<i64>,
    order: VecDeque<i64>,
}

impl HostTier {
    fn new(seed: u64) -> Self {
        Self {
            rng: Rng(seed ^ 0x5eed_0000_0000_0000),
            blocks: HashMap::new(),
            device: HashSet::new(),
            held: HashSet::new(),
            order: VecDeque::new(),
        }
    }

    /// The host batch a device batch calls for, to go ahead of it.
    fn write_back(&mut self, batch: &KvEventBatch) -> Option<Vec<u8>> {
        let mut events = Vec::new();
        for event in &batch.events {
            match &event.data {
                Some(kv_cache_event::Data::Stored(stored)) => {
                    let mut parent = stored.parent_block_hash;
                    for block in &stored.blocks {
                        self.blocks
                            .insert(block.block_hash, (block.token_ids.clone(), parent));
                        self.device.insert(block.block_hash);
                        parent = Some(block.block_hash);
                    }
                }
                Some(kv_cache_event::Data::Removed(removed)) => {
                    for &hash in &removed.block_hashes {
                        self.device.remove(&hash);
                        events.extend(self.write_back_one(hash));
                    }
                }
                Some(kv_cache_event::Data::Cleared(_)) => {
                    self.device.clear();
                    self.held.clear();
                    self.order.clear();
                }
                None => {}
            }
        }
        while self.held.len() > HOST_BLOCKS {
            let Some(hash) = self.order.pop_front() else {
                break;
            };
            self.held.remove(&hash);
            events.push(json!({"type": "BlockRemoved", "block_hashes": [hash], "medium": "CPU"}));
        }
        if events.is_empty() {
            return None;
        }
        let batch = json!([batch.timestamp, events, null]);
        Some(rmp_serde::to_vec_named(&batch).expect("a msgpack event batch"))
    }

    /// The host store for a block the device is evicting, when it is
    /// written back.
    fn write_back_one(&mut self, hash: i64) -> Option<Value> {
        if self.held.contains(&hash) || !self.rng.chance(2, 3) {
            return None;
        }
        let (tokens, parent) = self.blocks.get(&hash)?;
        let chained = match parent {
            None => true,
            Some(parent) => self.held.contains(parent) || self.device.contains(parent),
        };
        if !chained {
            return None;
        }
        let store = json!({
            "type": "BlockStored",
            "block_hashes": [hash],
            "parent_block_hash": parent,
            "token_ids": tokens,
            "block_size": BLOCK,
            "lora_id": null,
            "medium": "CPU",
        });
        self.held.insert(hash);
        self.order.push_back(hash);
        Some(store)
    }

    fn keys(&self) -> impl Iterator<Item = u64> + '_ {
        self.held.iter().map(|&hash| hash as u64)
    }
}

/// vLLM's second physical copies: when two requests in flight compute the
/// same prefix the engine keeps two blocks under one hash, publishes the
/// second inside the later request's longer store (its first blocks second
/// copies, the rest first copies) and removes each copy on its own. The mock
/// engine keeps one block per hash, so this layer gives a quarter of the
/// chained stores one to four second copies up the parent chain and removes
/// each some batches later. A block with a copy here is held whatever the
/// device did with the first.
struct SecondCopies {
    rng: Rng,
    /// Tokens and parent of every block the device stored, by engine hash.
    blocks: HashMap<i64, (Vec<u32>, Option<i64>)>,
    /// What the device holds, by the stream so far.
    device: HashSet<i64>,
    /// Blocks whose second copy is held, with the batch count it goes at.
    second: BTreeMap<i64, usize>,
    batches: usize,
}

impl SecondCopies {
    fn new(seed: u64) -> Self {
        Self {
            rng: Rng(seed ^ 0x2c0b_0000_0000_0000),
            blocks: HashMap::new(),
            device: HashSet::new(),
            second: BTreeMap::new(),
            batches: 0,
        }
    }

    /// Rewrite a device batch: second copies at the head of some stores,
    /// the due removals appended.
    fn transform(&mut self, batch: &mut KvEventBatch) {
        self.batches += 1;
        for event in &mut batch.events {
            match &mut event.data {
                Some(kv_cache_event::Data::Stored(stored)) => {
                    let mut parent = stored.parent_block_hash;
                    for block in &stored.blocks {
                        self.blocks
                            .insert(block.block_hash, (block.token_ids.clone(), parent));
                        self.device.insert(block.block_hash);
                        parent = Some(block.block_hash);
                    }
                    if self.rng.chance(1, 4) {
                        self.prepend_second_copies(stored);
                    }
                }
                Some(kv_cache_event::Data::Removed(removed)) => {
                    for hash in &removed.block_hashes {
                        self.device.remove(hash);
                    }
                }
                Some(kv_cache_event::Data::Cleared(_)) => {
                    self.device.clear();
                    self.second.clear();
                }
                None => {}
            }
        }
        let due: Vec<i64> = self
            .second
            .iter()
            .filter(|(_, &at)| at <= self.batches)
            .map(|(&hash, _)| hash)
            .collect();
        if due.is_empty() {
            return;
        }
        for hash in &due {
            self.second.remove(hash);
        }
        batch.events.push(KvCacheEvent {
            event_id: 0,
            data: Some(kv_cache_event::Data::Removed(KvBlocksRemoved {
                block_hashes: due,
                ..Default::default()
            })),
        });
    }

    /// One to four blocks up from the store's parent, all on the device and
    /// without a second copy yet, become the store's first blocks; the store
    /// then hangs off the block above them.
    fn prepend_second_copies(&mut self, stored: &mut KvBlocksStored) {
        let Some(parent) = stored.parent_block_hash else {
            return;
        };
        let wanted = self.rng.range(1, 4);
        let mut copies = Vec::new();
        let mut cursor = Some(parent);
        while copies.len() < wanted {
            let Some(hash) = cursor else { break };
            if !self.device.contains(&hash) || self.second.contains_key(&hash) {
                break;
            }
            let Some((_, above)) = self.blocks.get(&hash) else {
                break;
            };
            copies.push(hash);
            cursor = *above;
        }
        if copies.is_empty() {
            return;
        }
        // Collected child to parent; a store lists them root side first.
        copies.reverse();
        let mut blocks: Vec<KvBlock> = copies
            .iter()
            .map(|hash| {
                let (tokens, _) = self.blocks.get(hash).expect("a stored block");
                KvBlock {
                    block_hash: *hash,
                    token_ids: tokens.clone(),
                    block_size: BLOCK as i32,
                    ..Default::default()
                }
            })
            .collect();
        stored.parent_block_hash = self.blocks.get(&copies[0]).and_then(|(_, above)| *above);
        blocks.append(&mut stored.blocks);
        stored.blocks = blocks;
        for hash in copies {
            let at = self.batches + self.rng.range(8, 64);
            self.second.insert(hash, at);
        }
    }

    fn keys(&self) -> impl Iterator<Item = u64> + '_ {
        self.second.keys().map(|&hash| hash as u64)
    }
}
