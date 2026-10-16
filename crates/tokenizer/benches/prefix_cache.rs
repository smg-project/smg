//! The tokenizer caches on the request path: what one request costs with the
//! caches off, on an L0 hit, on an L1 prefix hit (the next turn of a
//! conversation whose earlier turns were seen) and on an L1 miss, next to the
//! steps a lookup is made of, per vocabulary and prompt size.
//!
//! `PREFIX_BENCH_TOKENIZERS=name=dir[,name=dir...]` (directories holding a
//! `tokenizer.json`), `PREFIX_BENCH_TOKENS=2048,8192,32768,81920` (the prompt
//! sizes, default as shown), `PREFIX_BENCH_VARIANTS=40` (distinct requests
//! prepared per scenario), `PREFIX_BENCH_SCENARIOS=encode,l0,l1,both,components`
//! (the scenario groups to run, default all; a group left out constructs no
//! cached tokenizer, so the inner tokenizer sees only that group's encodes).
//! One JSON line per measurement and a table at the end.
#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "benchmark setup and output"
)]

use std::{
    hash::{BuildHasher, DefaultHasher, Hasher},
    hint::black_box,
    sync::Arc,
    time::{Duration, Instant},
};

use aho_corasick::{AhoCorasick, MatchKind};
use llm_tokenizer::{
    create_tokenizer,
    traits::{Encoder, Encoding, Tokenizer},
    CacheConfig, CachedTokenizer, HuggingFaceTokenizer,
};
use serde_json::json;

const PROSE: &[&str] = &[
    "The harbour wakes before the town does. Fishing boats slip out past the breakwater while the \
     streetlights are still on, and by the time the bakery opens its shutters the first catch is \
     already being sorted on the quay. Visitors who arrive at noon see only the quiet version.",
    "A good map is a story told in one glance: where the rivers bend, which roads were built for \
     carts and which for cars, why the old centre sits on the hill. Reading it slowly rewards you \
     with the reasons behind the shapes, and those reasons rarely change.",
    "Bread dough does not care about schedules. It rises when the kitchen is warm and sulks when \
     it is cold, and the baker who learns to read it by touch stops looking at the clock. The \
     crust tells the rest of the story once the loaf comes out of the oven.",
    "The library kept its card catalogue long after the computers arrived, partly from habit and \
     partly because the cards held notes in the handwriting of librarians who had retired. Each \
     drawer was a small archive of how people had once looked for things.",
    "Mountain weather turns in minutes. A clear morning can close into cloud by the time the \
     path reaches the ridge, and the walkers who packed a jacket despite the forecast are the \
     ones still smiling at the hut. Experience is mostly remembered discomfort.",
    "Every workshop has a drawer of parts that fit nothing currently owned. They are kept because \
     throwing them away feels like admitting the project they came from will never return, and \
     once in a while one of them is exactly what a repair needs.",
];

const CJK: &[&str] = &[
    "清晨的市场里，摊贩们把蔬菜一筐一筐地摆好，水珠还挂在菜叶上。买菜的人不多，大家慢慢地挑选，偶尔聊几句天气和价格。",
    "这座桥建于两百多年前，石块之间没有用一点水泥，却经受住了无数次洪水。当地人说，桥的秘密在于每一块石头都知道自己的位置。",
    "图书馆的旧阅览室保留着木质的长桌和绿色的台灯。虽然现在大家都用电脑查资料，但下雨的午后，那里仍然坐满了人。",
    "山里的茶农在清明前后最忙，一天要采好几次。嫩芽要在太阳出来之前摘下，摊开晾在竹匾上，等到傍晚才开始炒制。",
];

const CODE: &[&str] = &[
    "fn parse_header(line: &str) -> Option<(&str, &str)> {\n    let (name, value) = line.split_once(':')?;\n    \
     Some((name.trim(), value.trim()))\n}\n",
    "def rolling_mean(values, window):\n    total = sum(values[:window])\n    out = [total / window]\n    \
     for i in range(window, len(values)):\n        total += values[i] - values[i - window]\n        \
     out.append(total / window)\n    return out\n",
    "SELECT region, COUNT(*) AS orders, SUM(amount) AS revenue\nFROM orders\nWHERE placed_at >= DATE '2024-01-01'\n\
     GROUP BY region\nORDER BY revenue DESC;\n",
];

/// How a vocabulary's chat markers wrap a turn: enough structure for the
/// boundaries a real conversation has (two special tokens per turn), without
/// depending on a template file.
enum Family {
    ChatMl,
    Llama3,
    MiniMax,
    Glm,
    Sentence,
}

impl Family {
    fn detect(specials: &[String]) -> Self {
        let has = |s: &str| specials.iter().any(|t| t == s);
        if has("<|im_start|>") && has("<|im_end|>") {
            Self::ChatMl
        } else if has("<|start_header_id|>") && has("<|eot_id|>") {
            Self::Llama3
        } else if has("]~b]") && has("[e~[") {
            Self::MiniMax
        } else if has("<|user|>") && has("<|assistant|>") {
            Self::Glm
        } else {
            Self::Sentence
        }
    }

    fn prelude(&self) -> &'static str {
        match self {
            Self::ChatMl | Self::MiniMax => "",
            Self::Llama3 => "<|begin_of_text|>",
            Self::Glm => "[gMASK]<sop>",
            Self::Sentence => "<s>",
        }
    }

    fn turn(&self, role: &str, body: &str) -> String {
        match self {
            Self::ChatMl => format!("<|im_start|>{role}\n{body}<|im_end|>\n"),
            Self::Llama3 => {
                format!("<|start_header_id|>{role}<|end_header_id|>\n\n{body}<|eot_id|>")
            }
            Self::MiniMax => format!("]~b]{role}\n{body}[e~[\n"),
            Self::Glm => format!("<|{role}|>\n{body}"),
            Self::Sentence => format!("<|{role}|>\n{body}</s>\n"),
        }
    }

    fn generation(&self) -> &'static str {
        match self {
            Self::ChatMl => "<|im_start|>assistant\n",
            Self::Llama3 => "<|start_header_id|>assistant<|end_header_id|>\n\n",
            Self::MiniMax => "]~b]assistant\n",
            Self::Glm | Self::Sentence => "<|assistant|>\n",
        }
    }
}

/// The special-token strings the cache scans for, gathered as the cache
/// gathers them (bos, eos, unk, sep, pad, cls, mask, then the additional
/// ones; duplicates included).
fn cache_specials(tokenizer: &dyn Tokenizer) -> Vec<String> {
    let s = tokenizer.get_special_tokens();
    let mut out = Vec::new();
    for token in [
        &s.bos_token,
        &s.eos_token,
        &s.unk_token,
        &s.sep_token,
        &s.pad_token,
        &s.cls_token,
        &s.mask_token,
    ]
    .into_iter()
    .flatten()
    {
        out.push(token.clone());
    }
    out.extend(s.additional_special_tokens.iter().cloned());
    out
}

const SYLLABLES: &[&str] = &[
    "ka", "ri", "to", "ne", "mu", "sa", "lo", "vi", "dek", "pon", "tra", "sel", "qui", "zam",
    "for", "bli", "ost", "ure", "chi", "wen",
];

/// A body of about `bytes` bytes of pseudo-words (two to four syllables),
/// digits and short CJK runs: a vocabulary wide enough that the encoders'
/// word and piece caches rarely hit, unlike [`body`]'s repeated paragraphs.
fn body_varied(seed: u64, bytes: usize) -> String {
    let mut text = String::with_capacity(bytes + 64);
    let mut k = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        k = k
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        k >> 33
    };
    let mut in_sentence = 0;
    while text.len() < bytes {
        let roll = next();
        match roll % 11 {
            0 => {
                let n = 3 + (next() % 6) as usize;
                for _ in 0..n {
                    let c = char::from_u32(0x4E00 + (next() % 2000) as u32).unwrap_or('\u{4E2D}');
                    text.push(c);
                }
            }
            1 => text.push_str(&format!("{}", next() % 100_000)),
            _ => {
                let n = 2 + (next() % 3) as usize;
                for _ in 0..n {
                    text.push_str(SYLLABLES[(next() % SYLLABLES.len() as u64) as usize]);
                }
            }
        }
        in_sentence += 1;
        if in_sentence >= 8 + (roll % 7) as usize {
            text.push_str(".\n");
            in_sentence = 0;
        } else {
            text.push(' ');
        }
    }
    text
}

/// A body of about `bytes` bytes of mixed prose, CJK text and code, with a
/// nonce woven in so two seeds never produce the same text; or, with
/// `PREFIX_BENCH_CORPUS=varied`, pseudo-words that defeat the word caches.
fn body(seed: u64, bytes: usize) -> String {
    if std::env::var("PREFIX_BENCH_CORPUS").is_ok_and(|v| v == "varied") {
        return body_varied(seed, bytes);
    }
    let mut text = String::with_capacity(bytes + 256);
    let mut k = seed;
    while text.len() < bytes {
        let pick = k % 13;
        let piece = if pick < 6 {
            PROSE[(pick as usize) % PROSE.len()]
        } else if pick < 10 {
            CJK[(pick as usize - 6) % CJK.len()]
        } else {
            CODE[(pick as usize - 10) % CODE.len()]
        };
        text.push_str(piece);
        text.push_str(&format!(" (ref {seed}-{k})\n"));
        k = k.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1) >> 3;
    }
    text
}

fn ids(tokenizer: &dyn Tokenizer, text: &str) -> Vec<u32> {
    tokenizer
        .encode(text, false)
        .expect("encode")
        .token_ids()
        .to_vec()
}

struct Stat {
    median_ms: f64,
    min_ms: f64,
    n: usize,
}

/// Runs `f(i)` for `i` in `0..count` (each at most once) until the budget is
/// spent, at least five times; the median and the minimum of the samples.
fn timed(count: usize, budget: Duration, mut f: impl FnMut(usize)) -> Stat {
    let mut samples = Vec::new();
    let started = Instant::now();
    for i in 0..count {
        let t0 = Instant::now();
        f(i);
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
        if samples.len() >= 5 && started.elapsed() > budget {
            break;
        }
    }
    samples.sort_by(f64::total_cmp);
    Stat {
        median_ms: samples[samples.len() / 2],
        min_ms: samples[0],
        n: samples.len(),
    }
}

fn cached(inner: &Arc<dyn Tokenizer>, l0: bool, l1: bool) -> CachedTokenizer {
    CachedTokenizer::new(
        Arc::clone(inner),
        CacheConfig {
            enable_l0: l0,
            l0_max_entries: 10_000,
            l0_max_memory: 1 << 30,
            enable_l1: l1,
            l1_max_memory: 1 << 30,
        },
    )
}

fn counters(t: &CachedTokenizer) -> [u64; 4] {
    let l0 = t.cache_stats();
    let l1 = t.l1_cache_stats();
    [
        l0.as_ref().map_or(0, |s| s.hits),
        l0.as_ref().map_or(0, |s| s.misses),
        l1.as_ref().map_or(0, |s| s.hits),
        l1.as_ref().map_or(0, |s| s.misses),
    ]
}

/// A conversation of `turns` user/assistant exchanges after a system prompt,
/// plus distinct last user messages: `head + turn(user, tail[v]) + generation`.
struct Shape {
    head: String,
    tails: Vec<String>,
    generation: &'static str,
    family: Family,
}

impl Shape {
    fn request(&self, v: usize) -> String {
        let mut text = self.head.clone();
        text.push_str(&self.family.turn("user", &self.tails[v]));
        text.push_str(self.generation);
        text
    }

    /// The bytes past the boundary before the last user turn: what an L1 hit
    /// at that boundary still encodes.
    fn suffix(&self, v: usize) -> String {
        let mut text = self.family.turn("user", &self.tails[v]);
        text.push_str(self.generation);
        text
    }
}

fn shape(
    family: Family,
    seed: u64,
    bytes_per_token: f64,
    tokens: usize,
    tail_fraction: f64,
    variants: usize,
) -> Shape {
    let total = tokens as f64 * bytes_per_token;
    let tail_bytes = (total * tail_fraction) as usize;
    let head_bytes = total as usize - tail_bytes;
    // Fixed per-turn overhead of the markers is small against the bodies.
    let turns = 4usize;
    let per_turn = head_bytes / (turns * 2 + 1);
    let mut head = String::from(family.prelude());
    head.push_str(&family.turn("system", &body(seed * 1000, per_turn)));
    for t in 0..turns {
        head.push_str(&family.turn("user", &body(seed * 1000 + 1 + t as u64 * 2, per_turn)));
        head.push_str(&family.turn("assistant", &body(seed * 1000 + 2 + t as u64 * 2, per_turn)));
    }
    let tails = (0..variants)
        .map(|v| body(seed * 1000 + 500 + v as u64, tail_bytes.max(24)))
        .collect();
    let generation = family.generation();
    Shape {
        head,
        tails,
        generation,
        family,
    }
}

fn main() {
    let specs = std::env::var("PREFIX_BENCH_TOKENIZERS").expect("PREFIX_BENCH_TOKENIZERS");
    let sizes: Vec<usize> = std::env::var("PREFIX_BENCH_TOKENS")
        .unwrap_or_else(|_| "2048,8192,32768,81920".into())
        .split(',')
        .map(|s| s.trim().parse().expect("token count"))
        .collect();
    let variants: usize = std::env::var("PREFIX_BENCH_VARIANTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    assert!(variants >= 3, "PREFIX_BENCH_VARIANTS must be at least 3");
    let groups: Vec<String> = std::env::var("PREFIX_BENCH_SCENARIOS")
        .map(|v| v.split(',').map(|g| g.trim().to_owned()).collect())
        .unwrap_or_default();
    let wanted = |group: &str| groups.is_empty() || groups.iter().any(|g| g == group);
    let budget = Duration::from_millis(300);
    let mut table: Vec<(String, String, usize, String, f64)> = Vec::new();

    for spec in specs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, dir) = spec.split_once('=').expect("name=dir");
        let inner: Arc<dyn Tokenizer> = create_tokenizer(dir).expect("tokenizer loads");
        let specials = cache_specials(inner.as_ref());
        let probe = body(7, 40_000);
        let sample = inner.encode(&probe, false).expect("probe encode");
        let path = match (
            inner
                .as_any()
                .downcast_ref::<HuggingFaceTokenizer>()
                .is_some_and(HuggingFaceTokenizer::uses_v1_encoder),
            &sample,
        ) {
            (true, _) => "v1",
            (false, Encoding::Plain(_)) => "native",
            (false, _) => "hf-0.23",
        };
        let bytes_per_token = probe.len() as f64 / sample.token_ids().len() as f64;
        let mut distinct = specials.clone();
        distinct.sort();
        distinct.dedup();
        let automaton = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&distinct)
            .expect("automaton");
        println!(
            "{}",
            json!({"kind":"model","model":name,"path":path,"special_strings":specials.len(),
                "distinct_special_strings":distinct.len(),"bytes_per_token":bytes_per_token})
        );

        for &tokens in &sizes {
            let mut rows: Vec<(String, Stat, [u64; 4])> = Vec::new();
            let tail10 = shape(
                Family::detect(&specials),
                1,
                bytes_per_token,
                tokens,
                0.10,
                variants,
            );
            let tail1 = shape(
                Family::detect(&specials),
                2,
                bytes_per_token,
                tokens,
                0.01,
                variants,
            );
            let unique: Vec<String> = (0..variants)
                .map(|v| {
                    shape(
                        Family::detect(&specials),
                        100 + v as u64,
                        bytes_per_token,
                        tokens,
                        0.10,
                        1,
                    )
                    .request(0)
                })
                .collect();
            let text0 = tail10.request(0);
            let reference = ids(inner.as_ref(), &text0);
            let measured_tokens = reference.len();

            // The encode itself, the baseline every cache row is read against.
            if wanted("encode") {
                let stat = timed(usize::MAX, budget, |_| {
                    black_box(inner.encode(&text0, false).expect("encode"));
                });
                rows.push(("encode".into(), stat, [0; 4]));
            }

            // L0 alone: a repeat (hit) and distinct inputs (miss + insert).
            if wanted("l0") {
                let t = cached(&inner, true, false);
                assert_eq!(ids(&t, &text0), reference);
                let before = counters(&t);
                let stat = timed(usize::MAX, budget, |_| {
                    black_box(t.encode(&text0, false).expect("encode"));
                });
                rows.push(("l0_hit".into(), stat, delta(before, counters(&t))));
                let t = cached(&inner, true, false);
                let before = counters(&t);
                let stat = timed(variants, budget, |v| {
                    black_box(t.encode(&unique[v], false).expect("encode"));
                });
                rows.push(("l0_miss".into(), stat, delta(before, counters(&t))));
            }

            // L1 alone: the same text again (a hit at the deepest boundary), the
            // next turn of a seen conversation (10 % and 1 % new text), and
            // distinct conversations (a miss that seeds every boundary).
            if wanted("l1") {
                let t = cached(&inner, false, true);
                assert_eq!(ids(&t, &text0), reference);
                let before = counters(&t);
                let stat = timed(usize::MAX, budget, |_| {
                    black_box(t.encode(&text0, false).expect("encode"));
                });
                rows.push(("l1_repeat".into(), stat, delta(before, counters(&t))));
                for (label, sh) in [("l1_next_turn_10", &tail10), ("l1_next_turn_1", &tail1)] {
                    let t = cached(&inner, false, true);
                    let first = sh.request(0);
                    t.encode(&first, false).expect("warm");
                    let second = sh.request(1);
                    assert_eq!(ids(&t, &second), ids(inner.as_ref(), &second), "{label}");
                    let before = counters(&t);
                    let stat = timed(variants - 2, budget, |v| {
                        black_box(t.encode(&sh.request(v + 2), false).expect("encode"));
                    });
                    rows.push((label.into(), stat, delta(before, counters(&t))));
                }
                let t = cached(&inner, false, true);
                let before = counters(&t);
                let stat = timed(variants, budget, |v| {
                    black_box(t.encode(&unique[v], false).expect("encode"));
                });
                rows.push(("l1_unique".into(), stat, delta(before, counters(&t))));
            }

            // Both caches, the gateway's shape when L1 is switched on.
            if wanted("both") {
                let t = cached(&inner, true, true);
                t.encode(&tail10.request(0), false).expect("warm");
                let before = counters(&t);
                let stat = timed(variants - 1, budget, |v| {
                    black_box(t.encode(&tail10.request(v + 1), false).expect("encode"));
                });
                rows.push((
                    "both_next_turn_10".into(),
                    stat,
                    delta(before, counters(&t)),
                ));
                let t = cached(&inner, true, true);
                let before = counters(&t);
                let stat = timed(variants, budget, |v| {
                    black_box(t.encode(&unique[v], false).expect("encode"));
                });
                rows.push(("both_unique".into(), stat, delta(before, counters(&t))));
            }

            // The steps a lookup is made of, on the same text.
            if wanted("components") {
                let text_bytes = text0.as_bytes();
                let stat = timed(usize::MAX, budget, |_| {
                    black_box(boundaries_per_string(&text0, &specials));
                });
                rows.push(("scan_per_special_string".into(), stat, [0; 4]));
                let stat = timed(usize::MAX, budget, |_| {
                    black_box(boundaries_one_pass(&text0, &automaton));
                });
                rows.push(("scan_one_pass".into(), stat, [0; 4]));
                let boundaries = boundaries_one_pass(&text0, &automaton);
                let stat = timed(usize::MAX, budget, |_| {
                    let mut hasher = blake3::Hasher::new();
                    hasher.update(&[0u8]);
                    let mut last = 0;
                    let mut digests = Vec::with_capacity(boundaries.len());
                    for &b in &boundaries {
                        hasher.update(&text_bytes[last..b]);
                        digests.push(*hasher.finalize().as_bytes());
                        last = b;
                    }
                    black_box(digests);
                });
                rows.push(("hash_blake3_prefixes".into(), stat, [0; 4]));
                let stat = timed(usize::MAX, budget, |_| {
                    let mut h = DefaultHasher::new();
                    h.write(text_bytes);
                    black_box(h.finish());
                });
                rows.push(("hash_siphash_text".into(), stat, [0; 4]));
                let keys = foldhash::fast::RandomState::default();
                let stat = timed(usize::MAX, budget, |_| {
                    let mut h = keys.build_hasher();
                    h.write(text_bytes);
                    h.write_u8(0);
                    black_box(h.finish());
                });
                rows.push(("hash_foldhash_text".into(), stat, [0; 4]));
                let copy = text0.clone();
                let stat = timed(usize::MAX, budget, |_| {
                    black_box(text_bytes == copy.as_bytes());
                });
                rows.push(("memcmp_text".into(), stat, [0; 4]));
                let stat = timed(usize::MAX, budget, |_| {
                    black_box(reference.clone());
                });
                rows.push(("memcpy_ids".into(), stat, [0; 4]));
                let stat = timed(usize::MAX, budget, |_| {
                    black_box(text0.to_string());
                });
                rows.push(("memcpy_text".into(), stat, [0; 4]));
                for (label, sh) in [("encode_suffix_10", &tail10), ("encode_suffix_1", &tail1)] {
                    let suffix = sh.suffix(0);
                    let stat = timed(usize::MAX, budget, |_| {
                        black_box(inner.encode(&suffix, false).expect("encode"));
                    });
                    rows.push((label.into(), stat, [0; 4]));
                }
            }

            for (scenario, stat, counts) in rows {
                println!(
                    "{}",
                    json!({"kind":"row","model":name,"path":path,"tokens":measured_tokens,
                        "bytes":text0.len(),"scenario":scenario,"median_ms":stat.median_ms,
                        "min_ms":stat.min_ms,"n":stat.n,"ns_per_token":stat.median_ms*1e6/measured_tokens as f64,
                        "l0_hits":counts[0],"l0_misses":counts[1],"l1_hits":counts[2],"l1_misses":counts[3]})
                );
                table.push((
                    name.to_string(),
                    path.to_string(),
                    measured_tokens,
                    scenario,
                    stat.median_ms,
                ));
            }
        }
    }

    println!(
        "\n{:<24} {:<8} {:>8} {:<26} {:>10}",
        "model", "path", "tokens", "scenario", "median ms"
    );
    for (model, path, tokens, scenario, ms) in &table {
        println!("{model:<24} {path:<8} {tokens:>8} {scenario:<26} {ms:>10.3}");
    }
}

fn delta(before: [u64; 4], after: [u64; 4]) -> [u64; 4] {
    [
        after[0] - before[0],
        after[1] - before[1],
        after[2] - before[2],
        after[3] - before[3],
    ]
}

/// The boundary scan the cache did before the one-pass matcher: every special
/// string searched over the whole text on its own.
fn boundaries_per_string(text: &str, specials: &[String]) -> Vec<usize> {
    let mut boundaries = Vec::new();
    for token in specials {
        let mut start = 0;
        while let Some(pos) = text[start..].find(token.as_str()) {
            let boundary = start + pos + token.len();
            if boundary < text.len() {
                boundaries.push(boundary);
            }
            start = boundary;
        }
    }
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries
}

/// One pass over the text with every special string in one automaton,
/// leftmost-longest as the tokenizer's own added-token extraction matches.
fn boundaries_one_pass(text: &str, automaton: &AhoCorasick) -> Vec<usize> {
    automaton
        .find_iter(text)
        .map(|m| m.end())
        .filter(|&end| end < text.len())
        .collect()
}
