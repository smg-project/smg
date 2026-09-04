//! Multi-writer load bench: the goal-doc G1–G3 baseline instrument.
//!
//! N concurrent PUBLISHER streams (one per simulated gateway) push a
//! duplicate-dominated placement stream at ONE live gRPC service
//! instance over loopback, while querier streams measure routing-time
//! latency — first idle, then under the full write load. Everything is
//! measured through the real wire: tonic, HTTP/2 flow control, the
//! ingest channel, the engine lock — exactly the path production
//! gateways hit.
//!
//!   radix-index-loadbench [--publishers 16] [--queriers 2]
//!     [--workers 64] [--chain-len 256] [--hot-per-worker 8]
//!     [--dup-pct 90] [--secs 15]
//!
//! Output is a flat key/value report; nothing is written to disk.
#![allow(
    clippy::disallowed_methods,
    clippy::too_many_arguments,
    clippy::type_complexity,
    reason = "standalone load-generator binary: fire-and-forget tasks and wide per-run knobs"
)]

use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use radix_index::{
    proto, proto::radix_index_client::RadixIndexClient, server, wire_hash, ContentHash, Engine,
    EngineConfig,
};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;

const MODEL: &str = "loadbench";
const BLOCK_SIZE: u32 = 128;

fn parse_flag<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Deterministic hot-chain contents for (worker, slot).
fn hot_contents(worker: usize, slot: usize, len: usize) -> Vec<ContentHash> {
    let seed = ((worker as u64) << 20) | slot as u64;
    (0..len as u64)
        .map(|pos| ContentHash(splitmix(seed.wrapping_mul(0x1000) ^ pos) | 1))
        .collect()
}

fn placement_update(holder: &str, contents: &[ContentHash]) -> proto::Update {
    let blocks = wire_hash::placement_chain(contents)
        .into_iter()
        .map(|(seq_hash, content_hash)| proto::Block {
            seq_hash: seq_hash.0,
            content_hash: content_hash.0,
        })
        .collect();
    proto::Update {
        keyspace: Some(proto::Keyspace {
            model: MODEL.into(),
            symbol_kind: proto::SymbolKind::Tokens as i32,
            block_size: BLOCK_SIZE,
            hash_scheme: wire_hash::HASH_SCHEME_V1,
        }),
        holder: holder.into(),
        epoch: 1,
        seq: 0,
        events: vec![proto::Event {
            kind: Some(proto::event::Kind::Stored(proto::Stored {
                parent_seq_hash: None,
                blocks,
            })),
        }],
        added: None,
        dropped: false,
    }
}

fn worker_url(w: usize) -> String {
    format!("grpc://10.0.0.{}:{}", w % 250, 9000 + w / 250)
}

/// A hot chain's rolling tip hash — its digest identity.
fn tip_of(worker: usize, slot: usize, len: usize) -> u64 {
    wire_hash::placement_chain(&hot_contents(worker, slot, len))
        .last()
        .expect("non-empty")
        .0
         .0
}

fn digest_update(holder: &str, tip: u64, len: usize) -> proto::Update {
    proto::Update {
        keyspace: Some(proto::Keyspace {
            model: MODEL.into(),
            symbol_kind: proto::SymbolKind::Tokens as i32,
            block_size: BLOCK_SIZE,
            hash_scheme: wire_hash::HASH_SCHEME_V1,
        }),
        holder: holder.into(),
        epoch: 1,
        seq: 0,
        events: vec![proto::Event {
            kind: Some(proto::event::Kind::StoredDigest(proto::StoredDigest {
                parent_seq_hash: None,
                tip_seq_hash: tip,
                len: len as u32,
            })),
        }],
        added: None,
        dropped: false,
    }
}

struct Percentiles {
    n: usize,
    p50: u64,
    p90: u64,
    p99: u64,
}

fn percentiles(mut ns: Vec<u64>) -> Percentiles {
    // An empty sample reports zeros instead of panicking on the
    // `len - 1` underflow (a leg that measured nothing should still
    // print a legible report).
    if ns.is_empty() {
        return Percentiles {
            n: 0,
            p50: 0,
            p90: 0,
            p99: 0,
        };
    }
    ns.sort_unstable();
    let pick = |p: f64| ns[((ns.len() as f64 * p) as usize).min(ns.len() - 1)];
    Percentiles {
        n: ns.len(),
        p50: pick(0.50),
        p90: pick(0.90),
        p99: pick(0.99),
    }
}

/// One querier: serial queries against hot chains through a real
/// Subscribe stream; returns latencies (ns) recorded while `running`.
async fn run_querier(
    url: String,
    workers: usize,
    hot_per_worker: usize,
    chain_len: usize,
    seed: u64,
    running: Arc<AtomicBool>,
    assert_hit: bool,
) -> Vec<u64> {
    let mut client = RadixIndexClient::connect(url)
        .await
        .expect("querier connect");
    let (tx, rx) = mpsc::channel::<proto::Query>(16);
    let mut answers = client
        .subscribe(tonic::Request::new(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
        .await
        .expect("subscribe")
        .into_inner();
    let mut rng = seed;
    let mut lat = Vec::new();
    let mut query_id = 1u64;
    while running.load(Ordering::Relaxed) {
        rng = splitmix(rng);
        let w = (rng % workers as u64) as usize;
        let slot = ((rng >> 32) % hot_per_worker as u64) as usize;
        let contents: Vec<u64> = hot_contents(w, slot, chain_len)
            .iter()
            .map(|c| c.0)
            .collect();
        let started = Instant::now();
        tx.send(proto::Query {
            query_id,
            keyspace: Some(proto::Keyspace {
                model: MODEL.into(),
                symbol_kind: proto::SymbolKind::Tokens as i32,
                block_size: BLOCK_SIZE,
                hash_scheme: wire_hash::HASH_SCHEME_V1,
            }),
            content_hashes: contents,
        })
        .await
        .expect("query send");
        let answer = answers.next().await.expect("answer").expect("answer ok");
        assert_eq!(answer.query_id, query_id, "serial stream must correlate");
        if assert_hit {
            assert!(
                !answer.scores.is_empty(),
                "hot query must match (warm fill covered it)"
            );
        }
        lat.push(started.elapsed().as_nanos() as u64);
        query_id += 1;
    }
    lat
}

/// A sequenced event-feed Stored batch (real write-lock work at the
/// DB: seq advances so it applies, and the chain walk runs under the
/// exclusive lock even when every block is a duplicate).
fn event_stored_update(holder: &str, seq: u64, contents: &[ContentHash]) -> proto::Update {
    let blocks = wire_hash::placement_chain(contents)
        .into_iter()
        .map(|(seq_hash, content_hash)| proto::Block {
            seq_hash: seq_hash.0,
            content_hash: content_hash.0,
        })
        .collect();
    proto::Update {
        keyspace: Some(proto::Keyspace {
            model: MODEL.into(),
            symbol_kind: proto::SymbolKind::Tokens as i32,
            block_size: BLOCK_SIZE,
            hash_scheme: wire_hash::HASH_SCHEME_V1,
        }),
        holder: holder.into(),
        epoch: 1,
        seq,
        events: vec![proto::Event {
            kind: Some(proto::event::Kind::Stored(proto::Stored {
                parent_seq_hash: None,
                blocks,
            })),
        }],
        added: None,
        dropped: false,
    }
}

/// One event-feed publisher: streams sequenced Stored batches for its
/// share of workers — the KV-event write load the DB must carry while
/// serving routing queries.
async fn run_event_publisher(
    url: String,
    workers: usize,
    hot_per_worker: usize,
    chain_len: usize,
    publisher_id: usize,
    publishers: usize,
    running: Arc<AtomicBool>,
    count_from: Instant,
    sent_blocks: Arc<AtomicU64>,
    target_ups: u64,
) {
    let client = RadixIndexClient::connect(url)
        .await
        .expect("event publisher connect");
    let mut client = client
        .max_decoding_message_size(64 * 1024 * 1024)
        .max_encoding_message_size(64 * 1024 * 1024);
    let (tx, rx) = mpsc::channel::<proto::Update>(256);
    let outbound = tokio_stream::wrappers::ReceiverStream::new(rx);
    let mut acks = client
        .publish(tonic::Request::new(outbound))
        .await
        .expect("publish stream")
        .into_inner();
    tokio::spawn(async move { while let Some(_ack) = acks.next().await {} });

    // Each publisher owns a DISJOINT worker shard (w % publishers ==
    // id), so a worker's sequenced stream has a single source — the
    // sharded-forwarder model. Multiple publisher streams still contend
    // on the shared keyspace lock; that contention is what the batched
    // applier must absorb.
    let shard: Vec<usize> = (0..workers)
        .filter(|w| w % publishers == publisher_id)
        .collect();
    if shard.is_empty() {
        return;
    }
    let mut rng = publisher_id as u64 ^ 0xE7E7;
    // seq base above any warm-fill seq so every load-phase event advances.
    let mut seqs: std::collections::HashMap<usize, u64> = std::collections::HashMap::new();
    let period = (target_ups > 0).then(|| Duration::from_secs_f64(1.0 / target_ups as f64));
    let mut next_at = Instant::now();
    while running.load(Ordering::Relaxed) {
        if let Some(period) = period {
            let now = Instant::now();
            if now < next_at {
                tokio::time::sleep(next_at - now).await;
            }
            next_at += period;
        }
        rng = splitmix(rng);
        let w = shard[(rng as usize) % shard.len()];
        let slot = ((rng >> 32) % hot_per_worker as u64) as usize;
        let s = seqs.entry(w).or_insert(1_000_000);
        *s += 1;
        let contents = hot_contents(w, slot, chain_len);
        if tx
            .send(event_stored_update(&worker_url(w), *s, &contents))
            .await
            .is_err()
        {
            break;
        }
        if Instant::now() >= count_from {
            sent_blocks.fetch_add(chain_len as u64, Ordering::Relaxed);
        }
    }
}

/// One publisher: a gateway identity pushing the duplicate-dominated
/// placement stream. Returns blocks sent after `count_from`.
async fn run_publisher(
    url: String,
    workers: usize,
    hot_per_worker: usize,
    chain_len: usize,
    dup_pct: u64,
    seed: u64,
    running: Arc<AtomicBool>,
    count_from: Instant,
    sent_blocks: Arc<AtomicU64>,
    // Per-publisher target updates/sec; 0 = unthrottled (max hammer).
    target_ups: u64,
    // Reference-gateway digest protocol: re-publish established hot
    // chains as {tip,len} digests instead of full blocks. Off = every
    // publish carries full blocks (the pre-digest baseline).
    use_digest: bool,
    resent_blocks: Arc<AtomicU64>,
) {
    let client = RadixIndexClient::connect(url)
        .await
        .expect("publisher connect");
    let mut client = client
        .max_decoding_message_size(64 * 1024 * 1024)
        .max_encoding_message_size(64 * 1024 * 1024);
    let (tx, rx) = mpsc::channel::<proto::Update>(64);
    let outbound = tokio_stream::wrappers::ReceiverStream::new(rx);
    let mut acks = client
        .publish(tonic::Request::new(outbound))
        .await
        .expect("publish stream")
        .into_inner();

    // Chains this publisher has established (sent full, not since
    // missed): tip -> (worker, slot), so a miss ack can rebuild and
    // resend the exact chain in full. Bounded by workers*hot_per_worker.
    let established: Arc<std::sync::Mutex<std::collections::HashMap<u64, (usize, usize)>>> =
        Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));

    // Ack reader: on a digest miss, resend the chain in FULL — never a
    // silent under-match. This is the publisher-side replay ring.
    let ack_tx = tx.clone();
    let ack_established = Arc::clone(&established);
    let ack_resent = Arc::clone(&resent_blocks);
    let ack_from = count_from;
    tokio::spawn(async move {
        while let Some(Ok(ack)) = acks.next().await {
            if let Some(tip) = ack.digest_miss_tip {
                let ws = ack_established.lock().expect("established").remove(&tip);
                if let Some((w, slot)) = ws {
                    let contents = hot_contents(w, slot, chain_len);
                    let full = placement_update(&worker_url(w), &contents);
                    if Instant::now() >= ack_from {
                        ack_resent.fetch_add(contents.len() as u64, Ordering::Relaxed);
                    }
                    // Re-establish once the full resend is in flight.
                    if ack_tx.send(full).await.is_ok() {
                        ack_established
                            .lock()
                            .expect("established")
                            .insert(tip, (w, slot));
                    }
                }
            }
        }
    });

    let mut rng = seed ^ 0xF00D;
    let mut nonce = 0u64;
    let period = (target_ups > 0).then(|| Duration::from_secs_f64(1.0 / target_ups as f64));
    let mut next_at = Instant::now();
    while running.load(Ordering::Relaxed) {
        if let Some(period) = period {
            let now = Instant::now();
            if now < next_at {
                tokio::time::sleep(next_at - now).await;
            }
            next_at += period;
        }
        rng = splitmix(rng);
        let w = (rng % workers as u64) as usize;
        let slot = ((rng >> 32) % hot_per_worker as u64) as usize;
        let mut wire_blocks = 0u64;
        let update = if rng % 100 < dup_pct {
            // The multi-gateway steady state: another gateway routed the
            // same hot prefix and re-publishes an identical chain.
            let tip = tip_of(w, slot, chain_len);
            let known = use_digest && established.lock().expect("established").contains_key(&tip);
            if known {
                digest_update(&worker_url(w), tip, chain_len)
            } else {
                if use_digest {
                    established
                        .lock()
                        .expect("established")
                        .insert(tip, (w, slot));
                }
                wire_blocks = chain_len as u64;
                placement_update(&worker_url(w), &hot_contents(w, slot, chain_len))
            }
        } else {
            // Fresh traffic: the hot prefix extended by a new tail (a
            // follow-up turn) — always full (a new, unestablished tip).
            nonce += 1;
            let mut contents = hot_contents(w, slot, chain_len);
            let tail_seed = seed.wrapping_mul(0x51D) ^ nonce;
            contents.extend((0..32u64).map(|p| ContentHash(splitmix(tail_seed ^ p) | 1)));
            wire_blocks = contents.len() as u64;
            placement_update(&worker_url(w), &contents)
        };
        if tx.send(update).await.is_err() {
            break;
        }
        if Instant::now() >= count_from {
            sent_blocks.fetch_add(wire_blocks, Ordering::Relaxed);
        }
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let publishers: usize = parse_flag(&args, "--publishers").unwrap_or(16);
    let queriers: usize = parse_flag(&args, "--queriers").unwrap_or(2);
    let workers: usize = parse_flag(&args, "--workers").unwrap_or(64);
    let chain_len: usize = parse_flag(&args, "--chain-len").unwrap_or(256);
    let hot_per_worker: usize = parse_flag(&args, "--hot-per-worker").unwrap_or(8);
    let dup_pct: u64 = parse_flag(&args, "--dup-pct").unwrap_or(90);
    let secs: u64 = parse_flag(&args, "--secs").unwrap_or(15);
    // Per-publisher target updates/sec (0 = max hammer). Model a
    // realistic gateway: 200 req/s each => --target-ups 200.
    let target_ups: u64 = parse_flag(&args, "--target-ups").unwrap_or(0);
    // --connect <URL> drives an EXTERNAL service (separate process, so
    // the OS schedules service vs loadgen independently — the honest
    // measurement). Omitted: an in-process service shares this
    // runtime, which conflates service cost with loadgen cost.
    let connect: Option<String> = parse_flag(&args, "--connect");
    // Reference-gateway digest protocol on the duplicate stream.
    let use_digest: bool = args.iter().any(|a| a == "--digest");
    // --events drives the SEQUENCED event feed (write-lock work) instead
    // of the idempotent placement feed, to measure the DB's event-path
    // apply throughput + query isolation.
    let events_mode: bool = args.iter().any(|a| a == "--events");

    let engine = Arc::new(Engine::new(EngineConfig::default()));
    let stats = Arc::new(server::ServiceStats::default());
    let external = connect.is_some();
    let url = match &connect {
        Some(u) => u.clone(),
        None => {
            let port = {
                let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("probe port");
                probe.local_addr().expect("probe addr").port()
            };
            let url = format!("http://127.0.0.1:{port}");
            let engine = Arc::clone(&engine);
            let stats = Arc::clone(&stats);
            tokio::spawn(server::serve_until(
                engine,
                format!("127.0.0.1:{port}").parse().unwrap(),
                Vec::new(),
                Duration::from_secs(60),
                Duration::ZERO,
                Duration::ZERO,
                stats,
                std::future::pending::<()>(),
            ));
            url
        }
    };
    // Wait for the service to accept.
    let mut attempt = 0;
    loop {
        match RadixIndexClient::connect(url.clone()).await {
            Ok(_) => break,
            Err(_) if attempt < 50 => {
                attempt += 1;
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            Err(error) => panic!("service never came up: {error}"),
        }
    }

    // Warm fill: every hot chain once, through one stream.
    let fill_start = Instant::now();
    {
        let mut client = RadixIndexClient::connect(url.clone()).await.expect("fill");
        let (tx, rx) = mpsc::channel::<proto::Update>(64);
        let fill = tokio::spawn(async move {
            for w in 0..workers {
                for slot in 0..hot_per_worker {
                    let contents = hot_contents(w, slot, chain_len);
                    let u = if events_mode {
                        // establish as event-fed holders (seq 1..hot_per_worker per worker)
                        event_stored_update(&worker_url(w), (slot + 1) as u64, &contents)
                    } else {
                        placement_update(&worker_url(w), &contents)
                    };
                    tx.send(u).await.expect("fill send");
                }
            }
        });
        let mut acks = client
            .publish(tonic::Request::new(
                tokio_stream::wrappers::ReceiverStream::new(rx),
            ))
            .await
            .expect("fill stream")
            .into_inner();
        fill.await.expect("fill task");
        // Acks are advisory (droppable), so drain until the server closes
        // the ack stream — which happens only after the whole fill has
        // been applied — rather than counting a fixed number.
        while let Some(ack) = acks.next().await {
            ack.expect("fill ack ok");
        }
    }
    let hot_blocks = workers * hot_per_worker * chain_len;
    println!(
        "warm_fill_blocks {hot_blocks} in {:.2}s",
        fill_start.elapsed().as_secs_f64()
    );

    // Phase 1: idle queries.
    let idle = {
        let running = Arc::new(AtomicBool::new(true));
        let mut tasks = Vec::new();
        for q in 0..queriers {
            tasks.push(tokio::spawn(run_querier(
                url.clone(),
                workers,
                hot_per_worker,
                chain_len,
                0xA11CE ^ q as u64,
                Arc::clone(&running),
                true,
            )));
        }
        tokio::time::sleep(Duration::from_secs(3)).await;
        running.store(false, Ordering::Relaxed);
        let mut lat = Vec::new();
        for t in tasks {
            lat.extend(t.await.expect("querier"));
        }
        percentiles(lat)
    };
    println!(
        "idle_query_ns n={} p50={} p90={} p99={}",
        idle.n, idle.p50, idle.p90, idle.p99
    );

    // Phase 2: full write load + queries.
    let running = Arc::new(AtomicBool::new(true));
    let sent_blocks = Arc::new(AtomicU64::new(0));
    let resent_blocks = Arc::new(AtomicU64::new(0));
    let warmup = Duration::from_secs(2);
    let count_from = Instant::now() + warmup;
    let mut pubs = Vec::new();
    for p in 0..publishers {
        if events_mode {
            pubs.push(tokio::spawn(run_event_publisher(
                url.clone(),
                workers,
                hot_per_worker,
                chain_len,
                p,
                publishers,
                Arc::clone(&running),
                count_from,
                Arc::clone(&sent_blocks),
                target_ups,
            )));
        } else {
            pubs.push(tokio::spawn(run_publisher(
                url.clone(),
                workers,
                hot_per_worker,
                chain_len,
                dup_pct,
                p as u64,
                Arc::clone(&running),
                count_from,
                Arc::clone(&sent_blocks),
                target_ups,
                use_digest,
                Arc::clone(&resent_blocks),
            )));
        }
    }
    tokio::time::sleep(warmup).await;
    let applies_before = stats.applies.load(Ordering::Relaxed);
    let window_start = Instant::now();
    let mut qtasks = Vec::new();
    for q in 0..queriers {
        qtasks.push(tokio::spawn(run_querier(
            url.clone(),
            workers,
            hot_per_worker,
            chain_len,
            0xB0B ^ q as u64,
            Arc::clone(&running),
            true,
        )));
    }
    tokio::time::sleep(Duration::from_secs(secs)).await;
    running.store(false, Ordering::Relaxed);
    let window = window_start.elapsed().as_secs_f64();
    let applies = stats.applies.load(Ordering::Relaxed) - applies_before;
    let mut lat = Vec::new();
    for t in qtasks {
        lat.extend(t.await.expect("loaded querier"));
    }
    for p in pubs {
        p.await.expect("publisher");
    }
    let loaded = percentiles(lat);
    let blocks = sent_blocks.load(Ordering::Relaxed);

    println!(
        "mode {} feed={} digest={} miss_resent_blocks={}",
        if external { "external" } else { "in-process" },
        if events_mode { "events" } else { "placements" },
        use_digest,
        resent_blocks.load(Ordering::Relaxed),
    );
    if external {
        // Publisher rate is client-observed; applies/gauges live in the
        // other process (scrape its /metrics for those).
        println!(
            "loaded_publish blocks_per_sec {:.0} (window {window:.1}s, {publishers} publishers, dup {dup_pct}%)",
            blocks as f64 / window,
        );
    } else {
        let gauges = engine.stats();
        println!(
            "loaded_publish blocks_per_sec {:.0} updates_per_sec {:.0} (window {window:.1}s, {publishers} publishers, dup {dup_pct}%)",
            blocks as f64 / window,
            applies as f64 / window,
        );
        println!(
            "engine keyspaces={} holders={} blocks={}",
            gauges.keyspaces, gauges.holders, gauges.blocks
        );
    }
    println!(
        "loaded_query_ns n={} p50={} p90={} p99={}",
        loaded.n, loaded.p50, loaded.p90, loaded.p99
    );
    println!(
        "isolation p99_loaded/p99_idle {:.2} (goal G2: <= 2.0)",
        loaded.p99 as f64 / idle.p99.max(1) as f64
    );
}
