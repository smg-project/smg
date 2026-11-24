//! The measurement cells of the comparative tokenizer harness (see `../README.md`).
//!
//! Each adapter binary implements `Tok` over one implementation and calls `run`; every cell
//! below is therefore the same code for every implementation. Output is one JSON object per line.
#![allow(clippy::too_many_lines, clippy::cast_precision_loss, dead_code)]
#![expect(
    unsafe_code,
    reason = "a counting global allocator needs the unsafe GlobalAlloc trait"
)]

use std::{
    collections::HashMap,
    fs,
    io::{BufRead, BufReader, Write},
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Barrier,
    },
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// One incremental decode stream (a fresh per-request decoder).
pub trait Streamer {
    /// Feed one id; append whatever text became final to `out`; true when some did.
    fn step(&mut self, id: u32, out: &mut String) -> Result<bool, String>;
    /// End of stream: append whatever is still held.
    fn finish(&mut self, out: &mut String) -> Result<(), String>;
}

/// The adapter each binary implements over its crate.
pub trait Tok: Send + Sync {
    fn backend(&self) -> String;
    fn encode(&self, text: &str, special: bool) -> Result<Vec<u32>, String>;
    /// The ids appended to `out`. The default moves the
    /// encode's own Vec in, so an implementation without a buffer-taking encode pays exactly its
    /// encode's allocation; an adapter whose crate has one overrides it.
    fn encode_into(&self, text: &str, special: bool, out: &mut Vec<u32>) -> Result<(), String> {
        *out = self.encode(text, special)?;
        Ok(())
    }
    fn decode(&self, ids: &[u32], skip: bool) -> Result<String, String>;
    fn stream<'a>(&'a self, skip: bool) -> Box<dyn Streamer + 'a>;
    fn vocab_size(&self) -> usize;
    fn id_to_token(&self, id: u32) -> Option<String>;
    fn token_to_id(&self, token: &str) -> Option<u32>;
    /// None when the crate has no such API.
    fn is_special(&self, id: u32) -> Option<bool>;
    /// None when the crate has no batch API.
    fn encode_batch(&self, texts: &[&str], special: bool) -> Option<Result<Vec<Vec<u32>>, String>>;
    fn eos_ids(&self) -> Vec<u32>;
    fn specials(&self) -> Value;
    fn added_tokens(&self) -> Vec<String>;
    /// The failure cells: the id of the single-byte token for `byte`, None when the vocabulary has none.
    fn byte_token_id(&self, byte: u8) -> Option<u32> {
        byte_token_id_default(self, byte)
    }
}

pub type Factory = fn(&Path, &HashMap<String, String>) -> Result<Box<dyn Tok>, String>;

#[derive(Serialize, Deserialize)]
pub struct CorpusItem {
    pub i: u64,
    pub cls: String,
    pub text: String,
}

// ---------------------------------------------------------------- helpers

/// The process's allocator calls, counted. `alloc` and `realloc` count as
/// allocations, `dealloc` as frees; one relaxed atomic add per call, the same cost per call in every implementation.
pub struct Counting;
pub static ALLOCS: AtomicU64 = AtomicU64::new(0);
pub static FREES: AtomicU64 = AtomicU64::new(0);
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        std::alloc::System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        FREES.fetch_add(1, Ordering::Relaxed);
        std::alloc::System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        std::alloc::System.realloc(ptr, layout, new_size)
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

pub fn rss_kb() -> (u64, u64) {
    let mut rss = 0;
    let mut hwm = 0;
    if let Ok(s) = fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            if let Some(v) = line.strip_prefix("VmRSS:") {
                rss = v.trim().trim_end_matches("kB").trim().parse().unwrap_or(0);
            } else if let Some(v) = line.strip_prefix("VmHWM:") {
                hwm = v.trim().trim_end_matches("kB").trim().parse().unwrap_or(0);
            }
        }
    }
    (rss, hwm)
}

/// Minor page faults of this process so far (field 10 of /proc/self/stat; 0 where it cannot be read).
pub fn minor_faults() -> u64 {
    fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|s| {
            s.rsplit(") ").next().map(|rest| {
                rest.split_whitespace()
                    .nth(7)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0)
            })
        })
        .unwrap_or(0)
}

pub fn pct(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

pub fn stats(samples: &[f64]) -> Value {
    let mut v: Vec<f64> = samples.iter().copied().filter(|x| x.is_finite()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    };
    json!({
        "n": v.len(),
        "min": v.first().copied().unwrap_or(f64::NAN),
        "p50": pct(&v, 0.5),
        "p90": pct(&v, 0.9),
        "p99": pct(&v, 0.99),
        "max": v.last().copied().unwrap_or(f64::NAN),
        "mean": mean,
    })
}

pub fn read_corpus(path: &Path) -> Vec<CorpusItem> {
    let f = fs::File::open(path).unwrap_or_else(|e| panic!("open corpus {}: {e}", path.display()));
    BufReader::new(f)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<CorpusItem>(&l).expect("corpus line"))
        .collect()
}

pub fn parse_args(args: &[String]) -> (Vec<String>, HashMap<String, String>) {
    let mut pos = Vec::new();
    let mut flags = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        if let Some(k) = args[i].strip_prefix("--") {
            if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                flags.insert(k.to_string(), args[i + 1].clone());
                i += 2;
            } else {
                flags.insert(k.to_string(), "true".to_string());
                i += 1;
            }
        } else {
            pos.push(args[i].clone());
            i += 1;
        }
    }
    (pos, flags)
}

fn flag<T: std::str::FromStr>(flags: &HashMap<String, String>, k: &str, d: T) -> T {
    flags.get(k).and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn emit(v: &Value) {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{v}").ok();
    out.flush().ok();
}

/// Run `f`, reporting a panic as Err("panic: ...") instead of unwinding out.
pub fn guarded<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "non-string panic payload".into());
            Err(format!("panic: {msg}"))
        }
    }
}

pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
    pub fn chance(&mut self, p: f64) -> bool {
        (self.next() % 10_000) as f64 / 10_000.0 < p
    }
}

fn fnv(ids: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for id in ids {
        h ^= *id as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn fnv_str(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub struct Loaded {
    pub tok: Box<dyn Tok>,
    pub load_ms: f64,
    pub rss_before: u64,
    pub rss_after: u64,
}

pub fn load(
    factory: Factory,
    dir: &Path,
    flags: &HashMap<String, String>,
) -> Result<Loaded, String> {
    let (rss_before, _) = rss_kb();
    let t0 = Instant::now();
    let tok = guarded(|| factory(dir, flags))?;
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let (rss_after, _) = rss_kb();
    Ok(Loaded {
        tok,
        load_ms,
        rss_before,
        rss_after,
    })
}

fn stream_all(tok: &dyn Tok, ids: &[u32], skip: bool) -> Result<String, String> {
    let mut s = tok.stream(skip);
    let mut out = String::new();
    for &id in ids {
        s.step(id, &mut out)?;
    }
    s.finish(&mut out)?;
    Ok(out)
}

fn sorted_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    v.sort();
    v
}

// ---------------------------------------------------------------- cells

fn cell_info(l: &Loaded, dir: &Path) {
    let tok = &*l.tok;
    let hello = guarded(|| tok.encode("Hello, world! 你好 🌍", false));
    let hello_sp = guarded(|| tok.encode("Hello, world! 你好 🌍", true));
    let dec = hello.as_ref().ok().map(|ids| tok.decode(ids, false));
    emit(&json!({
        "cell": "info",
        "dir": dir.display().to_string(),
        "backend": tok.backend(),
        "load_ms": l.load_ms,
        "rss_kb_before": l.rss_before,
        "rss_kb_after": l.rss_after,
        "rss_kb_delta": l.rss_after as i64 - l.rss_before as i64,
        "vocab_size": tok.vocab_size(),
        "specials": tok.specials(),
        "eos_ids": tok.eos_ids(),
        "added_tokens": tok.added_tokens().len(),
        "hello_ids": hello.as_ref().map(|v| json!(v)).unwrap_or_else(|e| json!(e)),
        "hello_ids_special": hello_sp.as_ref().map(|v| json!(v)).unwrap_or_else(|e| json!(e)),
        "hello_decoded": dec.map(|d| d.unwrap_or_else(|e| e)),
    }));
}

fn cell_parity(l: &Loaded, corpus: &Path, out: &Path, flags: &HashMap<String, String>) {
    let tok = &*l.tok;
    let items = read_corpus(corpus);
    let mut w = std::io::BufWriter::new(fs::File::create(out).expect("create out"));
    let (mut n_ok, mut n_err, mut n_panic) = (0u64, 0u64, 0u64);
    let want_inc = !flags.contains_key("no-inc");
    for it in &items {
        let mut rec = serde_json::Map::new();
        rec.insert("i".into(), json!(it.i));
        rec.insert("cls".into(), json!(it.cls));
        let t0 = Instant::now();
        let enc = guarded(|| tok.encode(&it.text, false));
        let ns_enc = t0.elapsed().as_nanos() as u64;
        rec.insert("ns_enc".into(), json!(ns_enc));
        match &enc {
            Ok(ids) => {
                rec.insert("ids".into(), json!(ids));
                let t1 = Instant::now();
                let d = guarded(|| tok.decode(ids, false));
                rec.insert("ns_dec".into(), json!(t1.elapsed().as_nanos() as u64));
                match d {
                    Ok(s) => {
                        rec.insert("dec".into(), json!(s));
                    }
                    Err(e) => {
                        rec.insert("dec_err".into(), json!(e));
                    }
                }
                match guarded(|| tok.decode(ids, true)) {
                    Ok(s) => {
                        rec.insert("dec_skip".into(), json!(s));
                    }
                    Err(e) => {
                        rec.insert("dec_skip_err".into(), json!(e));
                    }
                }
                if want_inc {
                    match guarded(|| stream_all(tok, ids, false)) {
                        Ok(s) => {
                            rec.insert("inc".into(), json!(s));
                        }
                        Err(e) => {
                            rec.insert("inc_err".into(), json!(e));
                        }
                    }
                    match guarded(|| stream_all(tok, ids, true)) {
                        Ok(s) => {
                            rec.insert("inc_skip".into(), json!(s));
                        }
                        Err(e) => {
                            rec.insert("inc_skip_err".into(), json!(e));
                        }
                    }
                }
                n_ok += 1;
            }
            Err(e) => {
                if e.starts_with("panic") {
                    n_panic += 1;
                } else {
                    n_err += 1;
                }
                rec.insert("enc_err".into(), json!(e));
            }
        }
        match guarded(|| tok.encode(&it.text, true)) {
            Ok(ids) => {
                rec.insert("ids_sp".into(), json!(ids));
            }
            Err(e) => {
                rec.insert("ids_sp_err".into(), json!(e));
            }
        }
        writeln!(w, "{}", Value::Object(rec)).ok();
    }
    w.flush().ok();
    emit(
        &json!({"cell":"parity","items":items.len(),"encode_ok":n_ok,"encode_err":n_err,"encode_panic":n_panic,"out":out.display().to_string(),"backend":tok.backend()}),
    );
}

/// Encode throughput on one length class: cold (a fresh text per repetition) or warm (one text repeated).
fn cell_enc(l: &Loaded, texts_dir: &Path, flags: &HashMap<String, String>) {
    let tok = &*l.tok;
    let class = flags.get("class").cloned().unwrap_or_else(|| "2k".into());
    let mode = flags.get("mode").cloned().unwrap_or_else(|| "warm".into());
    let threads: usize = flag(flags, "threads", 1);
    let reps: usize = flag(flags, "reps", 30);
    let warm: usize = flag(flags, "warm", 3);
    let special: bool = flags.contains_key("special");
    let files = sorted_files(&texts_dir.join(&class));
    assert!(!files.is_empty(), "no texts for class {class}");
    let texts: Vec<String> = files
        .iter()
        .map(|p| fs::read_to_string(p).expect("read text"))
        .collect();
    let pick = |k: usize| -> &str {
        if mode == "warm" {
            texts[0].as_str()
        } else {
            texts[k % texts.len()].as_str()
        }
    };
    let bytes: u64 = texts.iter().map(|t| t.len() as u64).sum::<u64>() / texts.len() as u64;
    // `--loops N`: the warm-ups and repetitions run N times over
    // in one process and the last pass is reported, so a profile of the process
    // is the encode phase and not its load.
    let loops: usize = flag(flags, "loops", 1);
    // `--into` or SMG_TOK_ENC_INTO: one buffer, cleared and reused across the reps.
    let into = flags.contains_key("into") || std::env::var_os("SMG_TOK_ENC_INTO").is_some();
    if threads <= 1 {
        let mut ns = Vec::with_capacity(reps);
        let mut toks = Vec::with_capacity(reps);
        let mut buf: Vec<u32> = Vec::new();
        let (mut allocs_per_op, mut frees_per_op) = (0.0, 0.0);
        // The minor faults the encode phase takes (first touches of the thread's tables), from /proc/self/stat.
        let minflt_before = minor_faults();
        for _ in 0..loops.max(1) {
            ns.clear();
            toks.clear();
            for k in 0..warm {
                if into {
                    buf.clear();
                    let _ = tok.encode_into(pick(k), special, &mut buf);
                } else {
                    let _ = tok.encode(pick(k), special);
                }
            }
            let (a0, f0) = (
                ALLOCS.load(Ordering::Relaxed),
                FREES.load(Ordering::Relaxed),
            );
            for r in 0..reps {
                let t = pick(warm + r);
                if into {
                    buf.clear();
                    let t0 = Instant::now();
                    tok.encode_into(t, special, &mut buf).expect("encode");
                    let dt = t0.elapsed().as_nanos() as f64;
                    std::hint::black_box(&buf);
                    ns.push(dt);
                    toks.push(buf.len() as f64);
                } else {
                    let t0 = Instant::now();
                    let ids = tok.encode(t, special).expect("encode");
                    let dt = t0.elapsed().as_nanos() as f64;
                    std::hint::black_box(&ids);
                    ns.push(dt);
                    toks.push(ids.len() as f64);
                }
            }
            allocs_per_op = (ALLOCS.load(Ordering::Relaxed) - a0) as f64 / reps.max(1) as f64;
            frees_per_op = (FREES.load(Ordering::Relaxed) - f0) as f64 / reps.max(1) as f64;
        }
        let minflt = minor_faults().saturating_sub(minflt_before);
        let per_tok: Vec<f64> = ns.iter().zip(&toks).map(|(a, b)| a / b.max(1.0)).collect();
        let tps: Vec<f64> = ns.iter().zip(&toks).map(|(a, b)| b / (a / 1e9)).collect();
        emit(&json!({
            "cell":"enc","class":class,"mode":mode,"threads":1,"reps":reps,"special":special,"loops":loops,
            "into": into, "allocs_per_op": allocs_per_op, "frees_per_op": frees_per_op,
            "minor_faults": minflt, "load_ms": l.load_ms, "total_tokens": toks.iter().sum::<f64>(),
            "tokens_per_op": stats(&toks), "bytes_per_op": bytes,
            "ns_per_op": stats(&ns), "ns_per_tok": stats(&per_tok), "tok_per_s": stats(&tps),
            "backend": tok.backend(),
        }));
        return;
    }
    // Multi-thread: every thread encodes its own distinct texts (cold) or the same text (warm),
    // `reps` operations each, released together by a barrier; aggregate throughput = all tokens / wall.
    for k in 0..warm {
        let _ = tok.encode(pick(k), special);
    }
    let barrier = Barrier::new(threads + 1);
    let total_tokens = AtomicU64::new(0);
    let lat = std::sync::Mutex::new(Vec::<f64>::new());
    let wall = std::thread::scope(|s| {
        for t in 0..threads {
            let barrier = &barrier;
            let total_tokens = &total_tokens;
            let lat = &lat;
            let pick = &pick;
            s.spawn(move || {
                barrier.wait();
                let mut mine = Vec::with_capacity(reps);
                let mut toks = 0u64;
                for r in 0..reps {
                    let text = pick(warm + t * reps + r);
                    let t0 = Instant::now();
                    let ids = tok.encode(text, special).expect("encode");
                    mine.push(t0.elapsed().as_nanos() as f64 / ids.len().max(1) as f64);
                    toks += ids.len() as u64;
                    std::hint::black_box(&ids);
                }
                total_tokens.fetch_add(toks, Ordering::Relaxed);
                lat.lock().unwrap().extend(mine);
            });
        }
        let t0 = Instant::now();
        barrier.wait();
        // scope joins the workers before returning
        t0
    });
    let wall_ns = wall.elapsed().as_nanos() as f64;
    let tt = total_tokens.load(Ordering::Relaxed) as f64;
    let lat = lat.into_inner().unwrap();
    emit(&json!({
        "cell":"enc","class":class,"mode":mode,"threads":threads,"reps":reps,"special":special,
        "wall_ns": wall_ns, "total_tokens": tt, "aggregate_tok_per_s": tt / (wall_ns / 1e9),
        "ns_per_tok_per_thread": stats(&lat), "backend": tok.backend(),
    }));
}

/// Full-sequence decode, incremental per-step decode and the first visible delta.
fn cell_dec(l: &Loaded, text_file: &Path, flags: &HashMap<String, String>) {
    let tok = &*l.tok;
    let len: usize = flag(flags, "len", 2048);
    let reps: usize = flag(flags, "reps", 30);
    let warm: usize = flag(flags, "warm", 3);
    let skip = flags.contains_key("skip");
    let text = fs::read_to_string(text_file).expect("read text");
    let mut ids = tok.encode(&text, false).expect("encode");
    if ids.len() > len {
        ids.truncate(len);
    }
    let n = ids.len().max(1) as f64;
    for _ in 0..warm {
        let _ = tok.decode(&ids, skip);
    }
    let mut full = Vec::new();
    let mut full_text = String::new();
    for _ in 0..reps {
        let t0 = Instant::now();
        full_text = tok.decode(&ids, skip).expect("decode");
        full.push(t0.elapsed().as_nanos() as f64 / n);
        std::hint::black_box(&full_text);
    }
    for _ in 0..warm {
        let _ = stream_all(tok, &ids, skip);
    }
    let mut inc_total = Vec::new();
    // The sample buffer is written once before the timed streams: a non-zero fill, then cleared (a zeroed allocation
    // does not touch its pages), so no stream times the first writes into its own samples. `--rawbuf 1` keeps
    // an untouched allocation for a comparison; the cell says which form ran
    // (sample_buf) and how many minor page faults the timed streams took (inc_faults).
    let raw_buf = flags.contains_key("rawbuf");
    let buf_form = if raw_buf { "raw" } else { "written" };
    let mut step_ns = if raw_buf {
        Vec::with_capacity(reps * ids.len())
    } else {
        vec![1f64; reps * ids.len()]
    };
    step_ns.clear();
    let mut streamed = String::new();
    let faults0 = minor_faults();
    for _ in 0..reps {
        let mut s = tok.stream(skip);
        streamed.clear();
        let t0 = Instant::now();
        for &id in &ids {
            let t1 = Instant::now();
            s.step(id, &mut streamed).expect("step");
            step_ns.push(t1.elapsed().as_nanos() as f64);
        }
        s.finish(&mut streamed).expect("finish");
        inc_total.push(t0.elapsed().as_nanos() as f64 / n);
    }
    let inc_faults = minor_faults().saturating_sub(faults0);
    // first delta: a fresh stream fed ids until the first non-empty text
    let fd_reps = reps * 10;
    let mut fd = Vec::with_capacity(fd_reps);
    let mut fed = 0usize;
    for _ in 0..fd_reps {
        let mut s = tok.stream(skip);
        let mut out = String::new();
        let t0 = Instant::now();
        for (k, &id) in ids.iter().enumerate() {
            s.step(id, &mut out).expect("step");
            if !out.is_empty() {
                fed = k + 1;
                break;
            }
        }
        fd.push(t0.elapsed().as_nanos() as f64);
    }
    emit(&json!({
        "cell":"dec","sample_buf":buf_form,"inc_faults":inc_faults,"len":ids.len(),"reps":reps,"skip":skip,
        "full_ns_per_tok": stats(&full),
        "inc_ns_per_tok": stats(&inc_total),
        "step_ns": stats(&step_ns),
        "first_delta_ns": stats(&fd), "first_delta_ids": fed,
        "inc_eq_full": streamed == full_text,
        "backend": tok.backend(),
    }));
}

/// S interleaved streams on one thread (the gateway's shape): per-step latency and per-stream first delta.
fn cell_stream(l: &Loaded, text_file: &Path, flags: &HashMap<String, String>) {
    let tok = &*l.tok;
    let streams: usize = flag(flags, "streams", 64);
    let steps: usize = flag(flags, "steps", 256);
    let reps: usize = flag(flags, "reps", 5);
    let skip = flags.contains_key("skip");
    let text = fs::read_to_string(text_file).expect("read text");
    let ids = tok.encode(&text, false).expect("encode");
    assert!(
        ids.len() >= steps + streams,
        "text too short for stream cell"
    );
    let mut step_ns = Vec::with_capacity(reps * streams * steps);
    let mut fd_wall = Vec::with_capacity(reps * streams);
    let mut fd_own = Vec::with_capacity(reps * streams);
    let mut mism = 0u64;
    for _ in 0..reps {
        let mut ss: Vec<Box<dyn Streamer>> = (0..streams).map(|_| tok.stream(skip)).collect();
        let mut outs: Vec<String> = vec![String::new(); streams];
        let mut first_start: Vec<Option<Instant>> = vec![None; streams];
        let mut own: Vec<f64> = vec![0.0; streams];
        let mut done: Vec<bool> = vec![false; streams];
        for k in 0..steps {
            for s in 0..streams {
                let id = ids[k + s];
                let t0 = Instant::now();
                if first_start[s].is_none() {
                    first_start[s] = Some(t0);
                }
                ss[s].step(id, &mut outs[s]).expect("step");
                let dt = t0.elapsed().as_nanos() as f64;
                step_ns.push(dt);
                if !done[s] {
                    own[s] += dt;
                    if !outs[s].is_empty() {
                        done[s] = true;
                        fd_wall.push(first_start[s].unwrap().elapsed().as_nanos() as f64);
                        fd_own.push(own[s]);
                    }
                }
            }
        }
        for s in 0..streams {
            ss[s].finish(&mut outs[s]).expect("finish");
            let full = tok.decode(&ids[s..s + steps], skip).expect("decode");
            if full != outs[s] {
                mism += 1;
            }
        }
    }
    emit(&json!({
        "cell":"stream","streams":streams,"steps":steps,"reps":reps,"skip":skip,
        "step_ns": stats(&step_ns),
        "first_delta_wall_ns": stats(&fd_wall),
        "first_delta_own_ns": stats(&fd_own),
        "streams_with_first_delta": fd_wall.len(), "inc_ne_full": mism,
        "backend": tok.backend(),
    }));
}

/// Batch encode of N strings: the crate's batch API when it has one, and a plain loop.
fn cell_batch(l: &Loaded, corpus: &Path, flags: &HashMap<String, String>) {
    let tok = &*l.tok;
    let n: usize = flag(flags, "n", 1000);
    let reps: usize = flag(flags, "reps", 10);
    let warm: usize = flag(flags, "warm", 2);
    let items = read_corpus(corpus);
    let texts: Vec<&str> = items.iter().take(n).map(|i| i.text.as_str()).collect();
    let mut loop_ns = Vec::new();
    let mut api_ns = Vec::new();
    let mut tokens = 0usize;
    for r in 0..warm + reps {
        let t0 = Instant::now();
        let mut tt = 0usize;
        for t in &texts {
            tt += tok.encode(t, false).expect("encode").len();
        }
        if r >= warm {
            loop_ns.push(t0.elapsed().as_nanos() as f64);
        }
        tokens = tt;
    }
    let mut has_api = false;
    for r in 0..warm + reps {
        let t0 = Instant::now();
        match tok.encode_batch(&texts, false) {
            Some(res) => {
                has_api = true;
                let v = res.expect("batch");
                std::hint::black_box(&v);
                if r >= warm {
                    api_ns.push(t0.elapsed().as_nanos() as f64);
                }
            }
            None => break,
        }
    }
    emit(&json!({
        "cell":"batch","n":texts.len(),"reps":reps,"tokens":tokens,
        "loop_ns": stats(&loop_ns), "loop_tok_per_s": stats(&loop_ns.iter().map(|x| tokens as f64 / (x / 1e9)).collect::<Vec<_>>()),
        "api": has_api, "api_ns": stats(&api_ns), "api_tok_per_s": stats(&api_ns.iter().map(|x| tokens as f64 / (x / 1e9)).collect::<Vec<_>>()),
        "backend": tok.backend(),
    }));
}

fn gen_input(case: &str) -> String {
    match case {
        "digits100k" => "1234567890".repeat(10_000),
        "spaces100k" => " ".repeat(100_000),
        "punct100k" => "!".repeat(100_000),
        "cjk1m" => "你".repeat(1 << 20),
        "zwj100k" => "\u{200D}a".repeat(50_000),
        "zwjonly100k" => "\u{200D}".repeat(100_000),
        "newlines100k" => "\n".repeat(100_000),
        "mixed100k" => " 1a!".repeat(25_000),
        "big64m" => {
            let unit = "The quick brown fox jumps over the lazy dog. 敏捷的棕色狐狸跳过懒狗。 Быстрая лиса. {\"k\": [1, 2.5, null]} 🚀\n";
            let mut s = String::with_capacity(64 << 20);
            while s.len() < (64 << 20) {
                s.push_str(unit);
            }
            // cut at the last character boundary at or below 64 MiB (truncate itself asserts the boundary)
            let mut n = 64 << 20;
            while !s.is_char_boundary(n) {
                n -= 1;
            }
            s.truncate(n);
            s
        }
        "big16m" => {
            let unit = "The quick brown fox jumps over the lazy dog. 敏捷的棕色狐狸跳过懒狗。 Быстрая лиса. {\"k\": [1, 2.5, null]} 🚀\n";
            let mut s = String::with_capacity(16 << 20);
            while s.len() < (16 << 20) {
                s.push_str(unit);
            }
            let mut n = 16 << 20;
            while !s.is_char_boundary(n) {
                n -= 1;
            }
            s.truncate(n);
            s
        }
        other => panic!("unknown input case {other}"),
    }
}

fn timed_guarded<T>(f: impl FnOnce() -> Result<T, String>) -> (Result<T, String>, f64, i64) {
    let (_, hwm0) = rss_kb();
    let t0 = Instant::now();
    let r = guarded(f);
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let (_, hwm1) = rss_kb();
    (r, ms, hwm1 as i64 - hwm0 as i64)
}

fn cell_robust(l: &Loaded, case: &str) {
    let tok = &*l.tok;
    let vs = tok.vocab_size() as u32;
    let mut rows: Vec<Value> = Vec::new();
    let mut push = |name: &str, r: Result<Value, String>, ms: f64, hwm: i64| {
        let (status, detail) = match r {
            Ok(v) => ("ok", v),
            Err(e) if e.starts_with("panic") => ("panic", json!(e)),
            Err(e) => ("err", json!(e)),
        };
        rows.push(json!({"case": name, "status": status, "wall_ms": ms, "hwm_delta_kb": hwm, "detail": detail}));
    };
    match case {
        "empty" => {
            let (r, ms, h) = timed_guarded(|| tok.encode("", false).map(|v| json!(v)));
            push("encode_empty", r, ms, h);
            let (r, ms, h) = timed_guarded(|| tok.encode("", true).map(|v| json!(v)));
            push("encode_empty_special", r, ms, h);
            let (r, ms, h) = timed_guarded(|| tok.decode(&[], false).map(|v| json!(v)));
            push("decode_empty", r, ms, h);
            let (r, ms, h) = timed_guarded(|| stream_all(tok, &[], false).map(|v| json!(v)));
            push("stream_empty", r, ms, h);
        }
        "oob" => {
            for (name, ids) in [
                ("decode_vocab_size", vec![vs]),
                ("decode_vocab_size_plus_1000", vec![vs + 1000]),
                ("decode_u32_max", vec![u32::MAX]),
                ("decode_i32_minus1_as_u32", vec![(-1i32) as u32]),
                ("decode_mixed_known_unknown", vec![0, vs, 1, u32::MAX, 2]),
            ] {
                let (r, ms, h) = timed_guarded(|| tok.decode(&ids, false).map(|v| json!(v)));
                push(name, r, ms, h);
                let (r, ms, h) = timed_guarded(|| tok.decode(&ids, true).map(|v| json!(v)));
                push(&format!("{name}_skip"), r, ms, h);
                let (r, ms, h) = timed_guarded(|| stream_all(tok, &ids, false).map(|v| json!(v)));
                push(&format!("{name}_stream"), r, ms, h);
            }
            for id in [vs, u32::MAX] {
                let (r, ms, h) = timed_guarded(|| {
                    Ok(
                        json!({"id_to_token": tok.id_to_token(id), "is_special": tok.is_special(id)}),
                    )
                });
                push(&format!("lookup_{id}"), r, ms, h);
            }
        }
        "splitmb" => {
            let text = "你好世界🌍é🇫🇷👨‍👩‍👧‍👦";
            let ids = tok.encode(text, false).expect("encode");
            let mut singles = Vec::new();
            for &id in &ids {
                let (r, _, _) = timed_guarded(|| tok.decode(&[id], false));
                singles.push(r.map(|s| json!(s)).unwrap_or_else(|e| json!(e)));
            }
            let mut prefixes = Vec::new();
            for k in 1..=ids.len() {
                let (r, _, _) = timed_guarded(|| tok.decode(&ids[..k], false));
                prefixes.push(r.map(|s| json!(s)).unwrap_or_else(|e| json!(e)));
            }
            let (inc, ms, h) = timed_guarded(|| stream_all(tok, &ids, false).map(|v| json!(v)));
            let full = tok.decode(&ids, false).ok();
            push(
                "splitmb",
                Ok(
                    json!({"ids": ids, "singles": singles, "prefixes": prefixes, "full": full, "inc": inc.unwrap_or_else(|e| json!(e))}),
                ),
                ms,
                h,
            );
            // every suffix of a dropped-first-id sequence through the stream (a split character at the stream head)
            let (r, ms, h) = timed_guarded(|| stream_all(tok, &ids[1..], false).map(|v| json!(v)));
            push("splitmb_stream_drop_first", r, ms, h);
        }
        "big16m" | "digits100k" | "spaces100k" | "punct100k" | "cjk1m" | "zwj100k"
        | "zwjonly100k" | "newlines100k" | "mixed100k" => {
            let text = gen_input(case);
            let bytes = text.len();
            let (r, ms, h) = timed_guarded(|| tok.encode(&text, false));
            match r {
                Ok(ids) => {
                    let n = ids.len();
                    let (d, dms, dh) = timed_guarded(|| tok.decode(&ids, false));
                    let roundtrip = d.as_ref().map(|s| s == &text).unwrap_or(false);
                    push(
                        &format!("{case}_encode"),
                        Ok(
                            json!({"bytes": bytes, "tokens": n, "ns_per_tok": ms * 1e6 / n.max(1) as f64, "mb_per_s": bytes as f64 / 1e6 / (ms / 1e3)}),
                        ),
                        ms,
                        h,
                    );
                    push(&format!("{case}_decode"), d.map(|_| json!({"roundtrip_exact": roundtrip, "ns_per_tok": dms * 1e6 / n.max(1) as f64})), dms, dh);
                    if n <= 300_000 {
                        let (s, sms, sh) = timed_guarded(|| stream_all(tok, &ids, false));
                        push(&format!("{case}_stream"), s.map(|st| json!({"ns_per_tok": sms * 1e6 / n.max(1) as f64, "eq_full": st == text})), sms, sh);
                    }
                }
                Err(e) => push(&format!("{case}_encode"), Err(e), ms, h),
            }
        }
        "load" => {
            let (r, ms, h) = timed_guarded(|| {
                tok.encode("Hello world, 你好 ☃ 1234 <s> </s>", false)
                    .map(|v| json!(v))
            });
            push("sample_encode", r, ms, h);
            push(
                "vocab",
                Ok(json!({"vocab_size": vs, "backend": tok.backend(), "specials": tok.specials()})),
                0.0,
                0,
            );
        }
        other => panic!("unknown robust case {other}"),
    }
    emit(
        &json!({"cell":"robust","case":case,"rows":rows,"backend":tok.backend(),"load_ms":l.load_ms}),
    );
}

/// T threads share one instance for `seconds`; every result is checked against a single-thread baseline.
fn cell_share(l: &Loaded, corpus: &Path, flags: &HashMap<String, String>) {
    let tok = &*l.tok;
    let threads: usize = flag(flags, "threads", 64);
    let seconds: u64 = flag(flags, "seconds", 60);
    let items = read_corpus(corpus);
    let texts: Vec<&str> = items.iter().map(|i| i.text.as_str()).collect();
    let baseline: Vec<(u64, u64)> = texts
        .iter()
        .map(|t| {
            let ids = tok.encode(t, false).unwrap_or_default();
            let dec = tok.decode(&ids, false).unwrap_or_default();
            (fnv(&ids), fnv_str(&dec))
        })
        .collect();
    let ops = AtomicU64::new(0);
    let mism = AtomicU64::new(0);
    let mism_enc = AtomicU64::new(0);
    let mism_dec = AtomicU64::new(0);
    let mism_inc = AtomicU64::new(0);
    let first_bad = std::sync::Mutex::new(Vec::<(usize, bool, bool, bool)>::new());
    let errs = AtomicU64::new(0);
    let deadline = Instant::now() + Duration::from_secs(seconds);
    std::thread::scope(|s| {
        for t in 0..threads {
            let (ops, mism, errs, texts, baseline) = (&ops, &mism, &errs, &texts, &baseline);
            let (mism_enc, mism_dec, mism_inc, first_bad) =
                (&mism_enc, &mism_dec, &mism_inc, &first_bad);
            s.spawn(move || {
                let mut rng = Rng::new(t as u64 + 7);
                let mut n = 0u64;
                while Instant::now() < deadline {
                    let i = rng.below(texts.len());
                    match guarded(|| {
                        let ids = tok.encode(texts[i], false)?;
                        let dec = tok.decode(&ids, false)?;
                        let mut st = String::new();
                        let mut sm = tok.stream(false);
                        for &id in &ids {
                            sm.step(id, &mut st)?;
                        }
                        sm.finish(&mut st)?;
                        Ok((fnv(&ids), fnv_str(&dec), st == dec))
                    }) {
                        Ok((a, b, inc_ok)) => {
                            let (e_bad, d_bad, i_bad) =
                                (a != baseline[i].0, b != baseline[i].1, !inc_ok);
                            if e_bad || d_bad || i_bad {
                                mism.fetch_add(1, Ordering::Relaxed);
                                if e_bad {
                                    mism_enc.fetch_add(1, Ordering::Relaxed);
                                }
                                if d_bad {
                                    mism_dec.fetch_add(1, Ordering::Relaxed);
                                }
                                if i_bad {
                                    mism_inc.fetch_add(1, Ordering::Relaxed);
                                }
                                let mut fb = first_bad.lock().unwrap();
                                if fb.len() < 20 {
                                    fb.push((i, e_bad, d_bad, i_bad));
                                }
                            }
                        }
                        Err(_) => {
                            errs.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    n += 1;
                }
                ops.fetch_add(n, Ordering::Relaxed);
            });
        }
    });
    let fb: Vec<Value> = first_bad
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|(i, e, d, inc)| json!({"i": i, "enc": e, "dec": d, "inc": inc}))
        .collect();
    emit(
        &json!({"cell":"share","threads":threads,"seconds":seconds,"texts":texts.len(),"ops":ops.load(Ordering::Relaxed),"mismatches":mism.load(Ordering::Relaxed),"mismatch_encode":mism_enc.load(Ordering::Relaxed),"mismatch_decode":mism_dec.load(Ordering::Relaxed),"mismatch_incremental":mism_inc.load(Ordering::Relaxed),"first_bad":fb,"errors_or_panics":errs.load(Ordering::Relaxed),"backend":tok.backend()}),
    );
}

/// Repeated encode+decode over the corpus for `minutes`; VmRSS sampled every `every` seconds.
fn cell_rss(l: &Loaded, corpus: &Path, flags: &HashMap<String, String>) {
    let tok = &*l.tok;
    let minutes: f64 = flag(flags, "minutes", 10.0);
    let every: u64 = flag(flags, "every", 10);
    let items = read_corpus(corpus);
    let deadline = Instant::now() + Duration::from_secs_f64(minutes * 60.0);
    let start = Instant::now();
    let mut next = start;
    let mut samples: Vec<(f64, u64)> = Vec::new();
    let mut ops = 0u64;
    let mut i = 0usize;
    while Instant::now() < deadline {
        if Instant::now() >= next {
            samples.push((start.elapsed().as_secs_f64(), rss_kb().0));
            next += Duration::from_secs(every);
        }
        let it = &items[i % items.len()];
        if let Ok(ids) = tok.encode(&it.text, false) {
            let _ = tok.decode(&ids, false);
            let _ = stream_all(tok, &ids, false);
        }
        ops += 1;
        i += 1;
    }
    samples.push((start.elapsed().as_secs_f64(), rss_kb().0));
    // least squares slope in KB per minute
    let n = samples.len() as f64;
    let (sx, sy) = samples
        .iter()
        .fold((0.0, 0.0), |(a, b), (x, y)| (a + x, b + *y as f64));
    let (mx, my) = (sx / n, sy / n);
    let num: f64 = samples
        .iter()
        .map(|(x, y)| (x - mx) * (*y as f64 - my))
        .sum();
    let den: f64 = samples.iter().map(|(x, _)| (x - mx) * (x - mx)).sum();
    let slope_kb_per_min = if den > 0.0 { num / den * 60.0 } else { 0.0 };
    emit(
        &json!({"cell":"rss","minutes":minutes,"ops":ops,"samples":samples,"first_kb":samples.first().map(|s| s.1),"last_kb":samples.last().map(|s| s.1),"slope_kb_per_min":slope_kb_per_min,"backend":tok.backend()}),
    );
}

const PALETTE: &[&str] = &[
    " ",
    "  ",
    "\n",
    "\r\n",
    "\t",
    "\u{0}",
    "\u{1}",
    "\u{7f}",
    "\u{a0}",
    "\u{200d}",
    "\u{200b}",
    "\u{feff}",
    "\u{301}",
    "\u{308}",
    "\u{fffd}",
    "a",
    "Z",
    "é",
    "ß",
    "İ",
    "ﬁ",
    "你",
    "日本語",
    "한국어",
    "العربية",
    "हिन्दी",
    "ไทย",
    "Русский",
    "🚀",
    "👨‍👩‍👧‍👦",
    "🇫🇷",
    "🏳️‍🌈",
    "'s",
    "'",
    "\"",
    "{",
    "}",
    "[",
    "]",
    ":",
    ",",
    "<",
    ">",
    "|",
    "_",
    "▁",
    "Ġ",
    "Ċ",
    "0",
    "12345",
    "3.14",
    "-",
    "..",
    "...",
    "<|",
    "|>",
    "</",
    "/>",
    "<s>",
    "</s>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
    "<|begin▁of▁sentence|>",
];

fn mutate_text(rng: &mut Rng, base: &str, added: &[String]) -> String {
    let chars: Vec<char> = base.chars().collect();
    let mut out: Vec<char> = chars.clone();
    let n_ops = 1 + rng.below(4);
    for _ in 0..n_ops {
        let op = rng.below(6);
        let pos = rng.below(out.len() + 1);
        match op {
            0 => {
                let p = PALETTE[rng.below(PALETTE.len())];
                out.splice(pos..pos, p.chars());
            }
            1 if !out.is_empty() => {
                let p = pos.min(out.len() - 1);
                out.remove(p);
            }
            2 if out.len() > 1 => {
                let a = rng.below(out.len());
                let b = (a + 1 + rng.below(16)).min(out.len());
                let span: Vec<char> = out[a..b].to_vec();
                out.splice(pos..pos, span);
            }
            3 if !added.is_empty() => {
                let t = &added[rng.below(added.len())];
                out.splice(pos..pos, t.chars());
            }
            4 => {
                let c = char::from_u32(rng.below(0x110000) as u32).unwrap_or('x');
                out.insert(pos, c);
            }
            _ => {
                // random byte-level garbage decoded lossily
                let n = 1 + rng.below(8);
                let bytes: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
                out.splice(pos..pos, String::from_utf8_lossy(&bytes).chars());
            }
        }
    }
    out.into_iter().collect()
}

fn mutate_ids(rng: &mut Rng, base: &[u32], vocab: u32) -> Vec<u32> {
    let mut out = base.to_vec();
    let n_ops = 1 + rng.below(4);
    for _ in 0..n_ops {
        let pos = rng.below(out.len() + 1);
        match rng.below(5) {
            0 => out.insert(pos, rng.below(vocab as usize) as u32),
            1 if !out.is_empty() => {
                let p = pos.min(out.len() - 1);
                out.remove(p);
            }
            2 => out.insert(
                pos,
                if rng.chance(0.5) {
                    vocab + rng.below(1000) as u32
                } else {
                    u32::MAX - rng.below(3) as u32
                },
            ),
            3 if out.len() > 2 => {
                let a = rng.below(out.len());
                let b = (a + 1 + rng.below(8)).min(out.len());
                out.drain(a..b);
            }
            _ => {
                if !out.is_empty() {
                    let p = pos.min(out.len() - 1);
                    out[p] = rng.below(vocab as usize) as u32;
                }
            }
        }
    }
    out
}

/// Structured differential fuzz: text and id mutations; anomalies and a sample are dumped to a file for a reference implementation to check.
fn cell_fuzz(l: &Loaded, corpus: &Path, flags: &HashMap<String, String>) {
    let tok = &*l.tok;
    let seconds: u64 = flag(flags, "seconds", 1200);
    let seed: u64 = flag(flags, "seed", 1);
    let sample_every: u64 = flag(flags, "sample-every", 25);
    let dump = flags.get("dump").map(PathBuf::from).expect("--dump <file>");
    let items = read_corpus(corpus);
    let added = tok.added_tokens();
    let vocab = tok.vocab_size() as u32;
    let mut w = std::io::BufWriter::new(fs::File::create(&dump).expect("create dump"));
    let mut rng = Rng::new(seed);
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut counts: HashMap<&'static str, u64> = HashMap::new();
    let mut n = 0u64;
    let mut dumped = 0u64;
    while Instant::now() < deadline {
        n += 1;
        let base = &items[rng.below(items.len())].text;
        let text_case = rng.chance(0.6);
        let mut anomalies: Vec<&'static str> = Vec::new();
        let mut rec = serde_json::Map::new();
        if text_case {
            let text = mutate_text(&mut rng, base, &added);
            rec.insert("kind".into(), json!("text"));
            rec.insert("text".into(), json!(text));
            match guarded(|| tok.encode(&text, false)) {
                Ok(ids) => {
                    rec.insert("ids".into(), json!(ids));
                    match guarded(|| tok.decode(&ids, false)) {
                        Ok(dec) => {
                            if dec != text {
                                anomalies.push("roundtrip_ne");
                            }
                            rec.insert("dec".into(), json!(dec));
                            match guarded(|| stream_all(tok, &ids, false)) {
                                Ok(inc) => {
                                    if inc != dec {
                                        anomalies.push("inc_ne_full");
                                        rec.insert("inc".into(), json!(inc));
                                    }
                                }
                                Err(e) => {
                                    anomalies.push(if e.starts_with("panic") {
                                        "panic_stream"
                                    } else {
                                        "err_stream"
                                    });
                                    rec.insert("inc_err".into(), json!(e));
                                }
                            }
                        }
                        Err(e) => {
                            anomalies.push(if e.starts_with("panic") {
                                "panic_decode"
                            } else {
                                "err_decode"
                            });
                            rec.insert("dec_err".into(), json!(e));
                        }
                    }
                    if let Err(e) = guarded(|| tok.encode(&text, true)) {
                        anomalies.push(if e.starts_with("panic") {
                            "panic_encode_special"
                        } else {
                            "err_encode_special"
                        });
                        rec.insert("enc_sp_err".into(), json!(e));
                    }
                }
                Err(e) => {
                    anomalies.push(if e.starts_with("panic") {
                        "panic_encode"
                    } else {
                        "err_encode"
                    });
                    rec.insert("enc_err".into(), json!(e));
                }
            }
        } else {
            let base_ids = tok.encode(base, false).unwrap_or_default();
            let ids = mutate_ids(&mut rng, &base_ids, vocab);
            rec.insert("kind".into(), json!("ids"));
            rec.insert("ids".into(), json!(ids));
            for skip in [false, true] {
                match guarded(|| tok.decode(&ids, skip)) {
                    Ok(dec) => {
                        rec.insert(
                            if skip {
                                "dec_skip".into()
                            } else {
                                "dec".into()
                            },
                            json!(dec),
                        );
                        match guarded(|| stream_all(tok, &ids, skip)) {
                            Ok(inc) => {
                                if inc != dec {
                                    anomalies.push(if skip {
                                        "inc_ne_full_skip"
                                    } else {
                                        "inc_ne_full"
                                    });
                                    rec.insert(
                                        if skip {
                                            "inc_skip".into()
                                        } else {
                                            "inc".into()
                                        },
                                        json!(inc),
                                    );
                                }
                            }
                            Err(e) => {
                                anomalies.push(if e.starts_with("panic") {
                                    "panic_stream"
                                } else {
                                    "err_stream"
                                });
                                rec.insert("inc_err".into(), json!(e));
                            }
                        }
                    }
                    Err(e) => {
                        anomalies.push(if e.starts_with("panic") {
                            "panic_decode"
                        } else {
                            "err_decode"
                        });
                        rec.insert("dec_err".into(), json!(e));
                    }
                }
            }
        }
        for a in &anomalies {
            *counts.entry(a).or_insert(0) += 1;
        }
        if !anomalies.is_empty() || n % sample_every == 0 {
            rec.insert("anomalies".into(), json!(anomalies));
            rec.insert("n".into(), json!(n));
            writeln!(w, "{}", Value::Object(rec)).ok();
            dumped += 1;
        }
    }
    w.flush().ok();
    emit(
        &json!({"cell":"fuzz","seconds":seconds,"seed":seed,"cases":n,"dumped":dumped,"anomaly_counts":counts,"dump":dump.display().to_string(),"backend":tok.backend()}),
    );
}

// ---------------------------------------------------------------- entry

pub fn run(factory: Factory, crate_name: &str) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (pos, flags) = parse_args(&args);
    if pos.len() < 2 {
        eprintln!("usage: {crate_name} <info|parity|enc|threads|mem|fail|dec|stream|batch|robust|share|rss|fuzz> <tokenizer-dir> [args] [--flags]");
        std::process::exit(2);
    }
    if !flags.contains_key("show-panics") {
        std::panic::set_hook(Box::new(|_| {}));
    }
    let cmd = pos[0].as_str();
    let dir = PathBuf::from(&pos[1]);
    let l = match load(factory, &dir, &flags) {
        Ok(l) => l,
        Err(e) => {
            emit(
                &json!({"cell": cmd, "load_error": e, "dir": dir.display().to_string(), "crate": crate_name}),
            );
            std::process::exit(if e.starts_with("panic") { 3 } else { 1 });
        }
    };
    match cmd {
        "info" => cell_info(&l, &dir),
        "parity" => cell_parity(&l, Path::new(&pos[2]), Path::new(&pos[3]), &flags),
        "enc" => cell_enc(&l, Path::new(&pos[2]), &flags),
        "threads" => cell_threads(&l, factory, &dir, Path::new(&pos[2]), &flags),
        "mem" => cell_mem(&l, factory, &dir, Path::new(&pos[2]), &flags),
        "fail" => cell_fail(&l, Path::new(&pos[2]), &flags),
        "dec" => cell_dec(&l, Path::new(&pos[2]), &flags),
        "stream" => cell_stream(&l, Path::new(&pos[2]), &flags),
        "batch" => cell_batch(&l, Path::new(&pos[2]), &flags),
        "robust" => cell_robust(&l, &pos[2]),
        "share" => cell_share(&l, Path::new(&pos[2]), &flags),
        "rss" => cell_rss(&l, Path::new(&pos[2]), &flags),
        "fuzz" => cell_fuzz(&l, Path::new(&pos[2]), &flags),
        other => {
            eprintln!("unknown cell {other}");
            std::process::exit(2);
        }
    }
}

// ---------------------------------------------------------------- concurrency, memory, failure handling

/// GPT-2's byte-to-unicode map: the character a byte-level BPE vocabulary spells one raw byte with.
pub fn gpt2_byte_char(b: u8) -> char {
    let direct =
        (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    if direct {
        return b as char;
    }
    // the other bytes take 256.. in byte order
    let mut n = 0u32;
    for x in 0u32..=255 {
        let d =
            (0x21..=0x7E).contains(&x) || (0xA1..=0xAC).contains(&x) || (0xAE..=0xFF).contains(&x);
        if !d {
            if x == b as u32 {
                return char::from_u32(256 + n).unwrap_or('\u{fffd}');
            }
            n += 1;
        }
    }
    '\u{fffd}'
}

/// The default `Tok::byte_token_id`: the sentencepiece spelling `<0xNN>` first, then the byte-level character.
pub fn byte_token_id_default<T: Tok + ?Sized>(tok: &T, byte: u8) -> Option<u32> {
    tok.token_to_id(&format!("<0x{byte:02X}>"))
        .or_else(|| tok.token_to_id(&gpt2_byte_char(byte).to_string()))
}

/// Concurrency: T threads, each `reps` encodes of the class's texts (cold: a different text per call; warm: one text),
/// released together by a barrier; `--own` gives every thread its own tokenizer object built before the barrier, otherwise
/// all threads share the one loaded object. Reports the aggregate throughput, the per-call latency (p50/p99 of ns per op
/// and ns per token over every call of every thread) and the resident set while the threads are alive.
fn cell_threads(
    l: &Loaded,
    factory: Factory,
    dir: &Path,
    texts_dir: &Path,
    flags: &HashMap<String, String>,
) {
    let class = flags.get("class").cloned().unwrap_or_else(|| "2k".into());
    let mode = flags.get("mode").cloned().unwrap_or_else(|| "cold".into());
    let threads: usize = flag(flags, "threads", 4);
    let reps: usize = flag(flags, "reps", 30);
    let warm: usize = flag(flags, "warm", 3);
    let own = flags.contains_key("own");
    let special: bool = flags.contains_key("special");
    let files = sorted_files(&texts_dir.join(&class));
    assert!(!files.is_empty(), "no texts for class {class}");
    let texts: Vec<String> = files
        .iter()
        .map(|p| fs::read_to_string(p).expect("read text"))
        .collect();
    let pick = |k: usize| -> &str {
        if mode == "warm" {
            texts[0].as_str()
        } else {
            texts[k % texts.len()].as_str()
        }
    };
    // own objects: built by each thread before the barrier (the load is not in the timed window)
    let (rss_before, _) = rss_kb();
    let start = Barrier::new(threads + 1);
    let hold = Barrier::new(threads + 1);
    let total_tokens = AtomicU64::new(0);
    let total_ns = AtomicU64::new(0);
    let lat_op = std::sync::Mutex::new(Vec::<f64>::new());
    let lat_tok = std::sync::Mutex::new(Vec::<f64>::new());
    let load_ms = std::sync::Mutex::new(Vec::<f64>::new());
    let shared: &dyn Tok = &*l.tok;
    let (wall_ns, rss_during) = std::thread::scope(|s| {
        for t in 0..threads {
            let (start, hold, total_tokens, total_ns, lat_op, lat_tok, load_ms, pick) = (
                &start,
                &hold,
                &total_tokens,
                &total_ns,
                &lat_op,
                &lat_tok,
                &load_ms,
                &pick,
            );
            s.spawn(move || {
                let mine_box: Option<Box<dyn Tok>> = if own {
                    let t0 = Instant::now();
                    let b = factory(dir, flags).expect("own tokenizer");
                    load_ms
                        .lock()
                        .unwrap()
                        .push(t0.elapsed().as_secs_f64() * 1e3);
                    Some(b)
                } else {
                    None
                };
                let tok: &dyn Tok = match &mine_box {
                    Some(b) => &**b,
                    None => shared,
                };
                for k in 0..warm {
                    let _ = tok.encode(pick(k + t * (warm + reps)), special);
                }
                start.wait();
                let mut ops = Vec::with_capacity(reps);
                let mut per = Vec::with_capacity(reps);
                let mut toks = 0u64;
                let mut ns = 0u64;
                for r in 0..reps {
                    let text = pick(warm + t * (warm + reps) + r);
                    let t0 = Instant::now();
                    let ids = tok.encode(text, special).expect("encode");
                    let dt = t0.elapsed().as_nanos() as u64;
                    ops.push(dt as f64);
                    per.push(dt as f64 / ids.len().max(1) as f64);
                    toks += ids.len() as u64;
                    ns += dt;
                    std::hint::black_box(&ids);
                }
                total_tokens.fetch_add(toks, Ordering::Relaxed);
                total_ns.fetch_add(ns, Ordering::Relaxed);
                lat_op.lock().unwrap().extend(ops);
                lat_tok.lock().unwrap().extend(per);
                // stay alive (tables and objects resident) until the main thread has read the resident set:
                // the first wait says this thread is done, the second returns after the read
                hold.wait();
                hold.wait();
            });
        }
        start.wait();
        // the clock starts when the barrier releases the threads (their own objects are loaded before it)
        let t0 = Instant::now();
        // the threads are running; the resident set is read once they are all done but still alive
        hold.wait();
        let wall = t0.elapsed().as_nanos() as f64;
        let (rss, _) = rss_kb();
        // the threads leave only now, after the resident set is read
        hold.wait();
        (wall, rss)
    });
    let tt = total_tokens.load(Ordering::Relaxed) as f64;
    let lat_op = lat_op.into_inner().unwrap();
    let lat_tok = lat_tok.into_inner().unwrap();
    emit(&json!({
        "cell":"threads","class":class,"mode":mode,"threads":threads,"reps":reps,"own":own,"special":special,
        "wall_ns": wall_ns, "total_tokens": tt, "aggregate_tok_per_s": tt / (wall_ns / 1e9),
        "busy_ns_sum": total_ns.load(Ordering::Relaxed),
        "ns_per_op": stats(&lat_op), "ns_per_tok": stats(&lat_tok),
        "rss_kb_before": rss_before, "rss_kb_during": rss_during,
        "rss_kb_growth_per_thread": (rss_during as f64 - rss_before as f64) / threads as f64,
        "own_load_ms": stats(&load_ms.into_inner().unwrap()),
        "backend": shared.backend(),
    }));
}

/// Memory: the resident set idle after the load, the high-water mark after the largest encodes (the 500k class file and
/// the 1M file), the allocator calls of one 2k encode, and the per-thread growth with 64 threads alive (shared object).
fn cell_mem(
    l: &Loaded,
    factory: Factory,
    dir: &Path,
    texts_dir: &Path,
    flags: &HashMap<String, String>,
) {
    let tok = &*l.tok;
    std::thread::sleep(Duration::from_millis(200));
    let (rss_idle, hwm_idle) = rss_kb();
    let read = |cls: &str| -> String {
        let p = sorted_files(&texts_dir.join(cls));
        fs::read_to_string(&p[0]).expect("read text")
    };
    let t2k = read("2k");
    let t500 = read("500k");
    let one_m_dir = flags.get("c1m").map(PathBuf::from);
    // allocator calls of one 2k encode (after a warm-up), the harness's counting allocator
    let _ = tok.encode(&t2k, false);
    let (a0, f0) = (
        ALLOCS.load(Ordering::Relaxed),
        FREES.load(Ordering::Relaxed),
    );
    let n2k = tok.encode(&t2k, false).map(|v| v.len()).unwrap_or(0);
    let (a1, f1) = (
        ALLOCS.load(Ordering::Relaxed),
        FREES.load(Ordering::Relaxed),
    );
    let (rss_2k, _) = rss_kb();
    // the 500k encode: the high-water mark after it
    let t0 = Instant::now();
    let n500 = guarded(|| tok.encode(&t500, false))
        .map(|v| v.len())
        .unwrap_or(0);
    let ms500 = t0.elapsed().as_secs_f64() * 1e3;
    let (rss_500, hwm_500) = rss_kb();
    // the 1M encode (two 500k texts), when the corpus is given
    let (n1m, ms1m, rss_1m, hwm_1m) = match one_m_dir {
        Some(d) => {
            let p = sorted_files(&d);
            let t = fs::read_to_string(&p[0]).expect("read 1m text");
            let t0 = Instant::now();
            let n = guarded(|| tok.encode(&t, false))
                .map(|v| v.len())
                .unwrap_or(0);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let (r, h) = rss_kb();
            (n, ms, r, h)
        }
        None => (0, 0.0, 0, 0),
    };
    // 64 threads alive on the shared object, each 20 encodes of distinct 2k texts: the growth per thread
    let threads: usize = flag(flags, "threads", 64);
    let files = sorted_files(&texts_dir.join("2k"));
    let texts: Vec<String> = files
        .iter()
        .map(|p| fs::read_to_string(p).expect("read text"))
        .collect();
    let (rss_t0, _) = rss_kb();
    let hold = Barrier::new(threads + 1);
    let rss_t1 = std::thread::scope(|s| {
        for t in 0..threads {
            let (hold, texts) = (&hold, &texts);
            s.spawn(move || {
                for r in 0..20 {
                    let _ = std::hint::black_box(tok.encode(&texts[(t + r) % texts.len()], false));
                }
                // alive until the main thread has read the resident set: the second wait returns after it
                hold.wait();
                hold.wait();
            });
        }
        hold.wait();
        let rss = rss_kb().0;
        hold.wait();
        rss
    });
    let (rss_t2, _) = rss_kb();
    // the same with one object per thread
    let hold2 = Barrier::new(threads + 1);
    let (rss_o0, _) = rss_kb();
    let rss_o1 = std::thread::scope(|s| {
        for t in 0..threads {
            let (hold2, texts) = (&hold2, &texts);
            s.spawn(move || {
                let mine = factory(dir, flags).expect("own tokenizer");
                for r in 0..20 {
                    let _ = std::hint::black_box(mine.encode(&texts[(t + r) % texts.len()], false));
                }
                // the tokenizer of this thread lives until the main thread has read the resident set
                hold2.wait();
                hold2.wait();
            });
        }
        hold2.wait();
        let rss = rss_kb().0;
        hold2.wait();
        rss
    });
    emit(&json!({
        "cell":"mem","load_ms": l.load_ms,
        "rss_kb_idle": rss_idle, "hwm_kb_idle": hwm_idle, "rss_kb_before_load": l.rss_before,
        "allocs_2k_encode": a1 - a0, "frees_2k_encode": f1 - f0, "tokens_2k": n2k, "rss_kb_after_2k": rss_2k,
        "tokens_500k": n500, "ms_500k": ms500, "rss_kb_after_500k": rss_500, "hwm_kb_after_500k": hwm_500,
        "tokens_1m": n1m, "ms_1m": ms1m, "rss_kb_after_1m": rss_1m, "hwm_kb_after_1m": hwm_1m,
        "threads": threads,
        "rss_kb_threads_before": rss_t0, "rss_kb_threads_alive": rss_t1, "rss_kb_threads_after": rss_t2,
        "growth_kb_per_thread_shared": (rss_t1 as f64 - rss_t0 as f64) / threads as f64,
        "rss_kb_own_before": rss_o0, "rss_kb_own_alive": rss_o1,
        // The level a thread ends at with an object of its own, counted from before the FIRST thread phase. The own
        // phase's threads take over the allocator arenas that the shared phase's threads left resident, so the count
        // from rss_o0 (growth_kb_per_thread_own) is only the NEW memory of the phase and depends on what the phase
        // before left behind: level = growth + left.
        "level_kb_per_thread_own": (rss_o1 as f64 - rss_t0 as f64) / threads as f64,
        "left_kb_per_thread_shared": (rss_t2 as f64 - rss_t0 as f64) / threads as f64,
        "growth_kb_per_thread_own": (rss_o1 as f64 - rss_o0 as f64) / threads as f64,
        "backend": tok.backend(),
    }));
}

fn digest_ids(ids: &[u32]) -> String {
    format!("{:016x}", fnv(ids))
}

/// Failure handling: invalid UTF-8 produced by the ids (a lone continuation byte, a truncated multi-byte sequence; in the
/// middle and at the end), ids outside the vocabulary, a truncated special token in the text (and the happy-path text of
/// the same length beside it), and a 64 MiB input. Every case: status (ok / err / panic), the output or the error, the time.
fn cell_fail(l: &Loaded, texts_dir: &Path, flags: &HashMap<String, String>) {
    let tok = &*l.tok;
    let vs = tok.vocab_size() as u32;
    let reps: usize = flag(flags, "reps", 200);
    let mut rows: Vec<Value> = Vec::new();
    let timed =
        |f: &dyn Fn() -> Result<Value, String>, reps: usize| -> (Result<Value, String>, f64, f64) {
            // the result of one call; the time = the p50 of `reps` calls (ns); the hwm delta over them
            let (_, h0) = rss_kb();
            let mut ns = Vec::with_capacity(reps);
            let mut last = Err("no call".to_string());
            for _ in 0..reps.max(1) {
                let t0 = Instant::now();
                last = guarded(f);
                ns.push(t0.elapsed().as_nanos() as f64);
            }
            let (_, h1) = rss_kb();
            ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (last, pct(&ns, 0.5), h1 as f64 - h0 as f64)
        };
    fn push_row(rows: &mut Vec<Value>, name: &str, r: (Result<Value, String>, f64, f64)) {
        let (status, detail) = match r.0 {
            Ok(v) => ("ok", v),
            Err(e) if e.starts_with("panic") => ("panic", json!(e)),
            Err(e) => ("err", json!(e)),
        };
        rows.push(json!({"case": name, "status": status, "ns_p50": r.1, "hwm_delta_kb": r.2, "detail": detail}));
    }
    let base_text = "Hello world, this is a sentence. 你好世界 ";
    let base_ids = tok.encode(base_text, false).unwrap_or_default();
    let byte = |b: u8| tok.byte_token_id(b);
    // invalid UTF-8 in the middle and at the end: a lone continuation byte 0x80, a truncated 3-byte sequence E2 82
    let cont = byte(0x80);
    let trunc: Option<Vec<u32>> = match (byte(0xE2), byte(0x82)) {
        (Some(a), Some(b)) => Some(vec![a, b]),
        _ => None,
    };
    let mid = base_ids.len() / 2;
    for (name, bad) in [
        ("cont_byte", cont.map(|c| vec![c])),
        ("truncated_seq", trunc),
    ] {
        match bad {
            None => rows.push(json!({"case": format!("utf8_{name}_mid"), "status": "no_byte_token", "ns_p50": 0.0, "hwm_delta_kb": 0.0, "detail": "the vocabulary exposes no single-byte token for this byte"})),
            Some(bad) => {
                let mut ids_mid = base_ids.clone();
                ids_mid.splice(mid..mid, bad.iter().copied());
                let mut ids_end = base_ids.clone();
                ids_end.extend(bad.iter().copied());
                for (pos, ids) in [("mid", ids_mid), ("end", ids_end)] {
                    let ids_c = ids.clone();
                    push_row(&mut rows, &format!("utf8_{name}_{pos}_decode"), timed(&|| tok.decode(&ids_c, false).map(|s| json!(s)), reps));
                    let ids_c = ids.clone();
                    push_row(&mut rows, &format!("utf8_{name}_{pos}_stream"), timed(&|| stream_all(tok, &ids_c, false).map(|s| json!(s)), reps));
                }
            }
        }
    }
    // ids outside the vocabulary
    // the crates disagree on what vocab_size() reports (with or without the added tokens), so the out-of-vocabulary ids
    // are vocab_size + 65,536 (past every added block) and u32::MAX; vocab_size itself is kept as a named case
    for (name, ids) in [
        ("oob_vocab_size_as_reported", vec![vs]),
        ("oob_vocab_size_plus_65536", vec![vs + 65_536]),
        ("oob_u32_max", vec![u32::MAX]),
        ("oob_mixed", {
            let mut v = base_ids.clone();
            v.insert(mid, vs + 65_536);
            v
        }),
    ] {
        let ids_c = ids.clone();
        push_row(
            &mut rows,
            &format!("{name}_decode"),
            timed(&|| tok.decode(&ids_c, false).map(|s| json!(s)), reps),
        );
        let ids_c = ids.clone();
        push_row(
            &mut rows,
            &format!("{name}_decode_skip"),
            timed(&|| tok.decode(&ids_c, true).map(|s| json!(s)), reps),
        );
        let ids_c = ids.clone();
        push_row(
            &mut rows,
            &format!("{name}_stream"),
            timed(&|| stream_all(tok, &ids_c, false).map(|s| json!(s)), reps),
        );
    }
    // a truncated special token in the text, the whole special token, and the happy-path text of the same length
    // the same token in every arm: the longest added token, the lexicographically smallest among the longest
    let mut added = tok.added_tokens();
    added.retain(|s| s.len() >= 4 && (s.starts_with('<') || s.starts_with('[')));
    added.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    if let Some(sp) = added.first().cloned() {
        let cut: String = sp
            .chars()
            .take(sp.chars().count().saturating_sub(2))
            .collect();
        let plain: String = "x".repeat(cut.chars().count());
        let with_trunc = format!("{base_text}{cut} and the rest of the line.");
        let with_full = format!("{base_text}{sp} and the rest of the line.");
        let happy = format!("{base_text}{plain} and the rest of the line.");
        for (name, text) in [
            ("special_truncated", &with_trunc),
            ("special_whole", &with_full),
            ("special_happy", &happy),
        ] {
            for special in [false, true] {
                let t = text.clone();
                push_row(
                    &mut rows,
                    &format!("{name}_special{}", special as u8),
                    timed(
                        &|| {
                            tok.encode(&t, special).map(|ids| json!({"n": ids.len(), "digest": digest_ids(&ids), "ids_head": ids.iter().take(24).collect::<Vec<_>>()}))
                        },
                        reps,
                    ),
                );
            }
        }
        rows.push(json!({"case": "special_token_used", "status": "ok", "ns_p50": 0.0, "hwm_delta_kb": 0.0, "detail": {"token": sp, "truncated": cut}}));
    } else {
        rows.push(json!({"case": "special_token_used", "status": "none", "ns_p50": 0.0, "hwm_delta_kb": 0.0, "detail": "no added token of 4+ characters starting with < or ["}));
    }
    // the 64 MiB input: one encode, one decode (the time, the tokens, the high-water mark), unless --no-big
    if !flags.contains_key("no-big") {
        let text = gen_input("big64m");
        let bytes = text.len();
        let (_, h0) = rss_kb();
        let t0 = Instant::now();
        let r = guarded(|| tok.encode(&text, false));
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let (_, h1) = rss_kb();
        match r {
            Ok(ids) => {
                let n = ids.len();
                let t1 = Instant::now();
                let d = guarded(|| tok.decode(&ids, false));
                let dms = t1.elapsed().as_secs_f64() * 1e3;
                let (_, h2) = rss_kb();
                let roundtrip = d.as_ref().map(|s| s == &text).unwrap_or(false);
                rows.push(json!({"case": "big64m_encode", "status": "ok", "ns_p50": ms * 1e6, "hwm_delta_kb": h1 as f64 - h0 as f64,
                    "detail": {"bytes": bytes, "tokens": n, "ns_per_tok": ms * 1e6 / n.max(1) as f64, "mb_per_s": bytes as f64 / 1e6 / (ms / 1e3)}}));
                rows.push(json!({"case": "big64m_decode", "status": if d.is_ok() {"ok"} else {"err"}, "ns_p50": dms * 1e6, "hwm_delta_kb": h2 as f64 - h1 as f64,
                    "detail": d.map(|_| json!({"roundtrip_exact": roundtrip, "ns_per_tok": dms * 1e6 / n.max(1) as f64})).unwrap_or_else(|e| json!(e))}));
            }
            Err(e) => rows.push(json!({"case": "big64m_encode", "status": if e.starts_with("panic") {"panic"} else {"err"}, "ns_p50": ms * 1e6, "hwm_delta_kb": h1 as f64 - h0 as f64, "detail": e})),
        }
    }
    // the happy path of the same shape as the utf8 cases: decode and stream of the base ids
    let ids_c = base_ids.clone();
    push_row(
        &mut rows,
        "happy_decode",
        timed(&|| tok.decode(&ids_c, false).map(|s| json!(s)), reps),
    );
    let ids_c = base_ids.clone();
    push_row(
        &mut rows,
        "happy_stream",
        timed(&|| stream_all(tok, &ids_c, false).map(|s| json!(s)), reps),
    );
    let _ = texts_dir;
    emit(
        &json!({"cell":"fail","rows":rows,"backend":tok.backend(),"load_ms":l.load_ms,"vocab_size":vs}),
    );
}
