//! Matched default/opt-in decode measurements; see decode/README.md.
#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "benchmark validation and JSONL output"
)]

use std::{
    env, fs,
    hint::black_box,
    path::Path,
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};

use llm_tokenizer::{traits::Tokenizer as SmgTokenizer, HuggingFaceTokenizer, Sequence};
use serde_json::{json, Value};
use tokenizers::{decoders::DecoderWrapper, utils::parallelism, Tokenizer};

fn stream(tokenizer: &Arc<dyn SmgTokenizer>, ids: &[u32], skip: bool) -> usize {
    let mut stream = Sequence::new_with_options(Arc::clone(tokenizer), skip);
    let mut bytes = 0;
    for &id in ids {
        bytes += black_box(stream.append_token(id).expect("stream step")).len();
    }
    bytes + black_box(stream.finish().expect("stream finish")).len()
}

fn stream_digest(tokenizer: &Arc<dyn SmgTokenizer>, ids: &[u32], skip: bool) -> String {
    let mut stream = Sequence::new_with_options(Arc::clone(tokenizer), skip);
    let mut hash = blake3::Hasher::new();
    for &id in ids {
        let text = stream.append_token(id).expect("stream step");
        hash.update(&(text.len() as u64).to_le_bytes());
        hash.update(text.as_bytes());
    }
    let tail = stream.finish().expect("stream finish");
    hash.update(&(tail.len() as u64).to_le_bytes());
    hash.update(tail.as_bytes());
    hash.finalize().to_hex().to_string()
}

fn measure(
    tokenizer: &Arc<dyn SmgTokenizer>,
    ids: &[u32],
    skip: bool,
    streaming: bool,
    callers: usize,
    duration: Duration,
) -> (f64, f64, usize) {
    let barrier = Barrier::new(callers);
    let observations = thread::scope(|scope| {
        let handles: Vec<_> = (0..callers)
            .map(|_| {
                let barrier = &barrier;
                scope.spawn(move || {
                    let call = || {
                        if streaming {
                            stream(tokenizer, ids, skip)
                        } else {
                            black_box(tokenizer.decode(ids, skip).expect("decode")).len()
                        }
                    };
                    for _ in 0..8 {
                        black_box(call());
                    }
                    barrier.wait();
                    let start = Instant::now();
                    let mut calls = 0;
                    // Amortize clock overhead for empty and one-token inputs.
                    loop {
                        for _ in 0..32 {
                            black_box(call());
                        }
                        calls += 32;
                        if start.elapsed() >= duration {
                            break;
                        }
                    }
                    (start.elapsed().as_secs_f64(), calls)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("worker"))
            .collect::<Vec<_>>()
    });
    let calls = observations.iter().map(|(_, count)| count).sum::<usize>();
    let total_seconds = observations.iter().map(|(seconds, _)| seconds).sum::<f64>();
    let wall_seconds = observations
        .iter()
        .map(|(seconds, _)| *seconds)
        .fold(0.0, f64::max);
    (
        total_seconds * 1e6 / calls as f64,
        calls as f64 / wall_seconds,
        calls,
    )
}

struct Case<'a> {
    model: &'a str,
    scenario: &'a str,
    ids: &'a [u32],
    skip: bool,
    streaming: bool,
    callers: usize,
}

fn run(hf: &Tokenizer, smg: &Arc<dyn SmgTokenizer>, case: Case<'_>, duration: Duration) {
    let expected = hf.decode(case.ids, case.skip).expect("reference decode");
    assert_eq!(smg.decode(case.ids, case.skip).expect("decode"), expected);
    let digest = if case.streaming {
        stream_digest(smg, case.ids, case.skip)
    } else {
        blake3::hash(expected.as_bytes()).to_hex().to_string()
    };
    if duration.is_zero() {
        println!(
            "{}",
            json!({"kind":"control-parity", "scenario":case.scenario, "digest":digest})
        );
        return;
    }
    let (mean_us, requests_per_second, samples) = measure(
        smg,
        case.ids,
        case.skip,
        case.streaming,
        case.callers,
        duration,
    );
    println!(
        "{}",
        json!({
            "kind": "timing", "model": case.model, "scenario": case.scenario,
            "tokens": case.ids.len(), "skip_special_tokens": case.skip,
            "streaming": case.streaming, "callers": case.callers,
            "mean_us": mean_us, "requests_per_second": requests_per_second,
            "samples": samples, "output_bytes": expected.len(), "digest": digest,
            "feature": cfg!(feature = "tokenizers-v1")
        })
    );
}

fn parity(hf: &Tokenizer, smg: &Arc<dyn SmgTokenizer>, vocab: &[u32], ascii: u32) -> usize {
    let mut count = 0;
    let mut check = |ids: &[u32]| {
        for skip in [false, true] {
            assert_eq!(
                smg.decode(ids, skip).expect("decode"),
                hf.decode(ids, skip).expect("reference")
            );
            count += 1;
        }
    };
    for &id in vocab {
        check(&[id]);
        check(&[id, ascii]);
        check(&[ascii, id, u32::MAX]);
    }
    check(&[]);
    let mut state = 7u64;
    for round in 0..10_000 {
        let ids: Vec<_> = (0..round % 64)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                if state.is_multiple_of(31) {
                    u32::MAX
                } else {
                    vocab[state as usize % vocab.len()]
                }
            })
            .collect();
        check(&ids);
    }
    count
}

fn real_model(name: &str, root: &Path, duration: Duration, check: bool) {
    let bytes = fs::read(root.join(name).join("tokenizer.json")).expect("pinned tokenizer JSON");
    let start = Instant::now();
    let hf = Tokenizer::from_bytes(&bytes).expect("reference tokenizer");
    let reference_ms = start.elapsed().as_secs_f64() * 1000.0;
    let cloned = hf.clone();
    let start = Instant::now();
    let smg: Arc<dyn SmgTokenizer> = Arc::new(HuggingFaceTokenizer::from_tokenizer(cloned));
    println!(
        "{}",
        json!({"kind":"model", "model":name,
            "hash":blake3::hash(&bytes).to_hex().to_string(), "reference_load_ms":reference_ms,
            "wrapper_load_ms":start.elapsed().as_secs_f64()*1000.0,
            "feature":cfg!(feature="tokenizers-v1"),
            "byte_table_decode":cfg!(feature="tokenizers-v1") && matches!(hf.get_decoder(), Some(DecoderWrapper::ByteLevel(_)))
        })
    );
    let ids = hf
        .encode(
            "hello 世界 café 🙂 function foo(x) { return x + 42; }\n".repeat(10_000),
            false,
        )
        .expect("input encode")
        .get_ids()
        .to_vec();
    let mut vocab: Vec<_> = hf.get_vocab(true).values().copied().collect();
    vocab.sort_unstable();
    vocab.dedup();
    if check {
        let count = parity(&hf, &smg, &vocab, ids[0]);
        thread::scope(|scope| {
            for _ in 0..8 {
                let hf = &hf;
                let smg = &smg;
                let ids = &ids[..2048];
                scope.spawn(move || {
                    for _ in 0..100 {
                        assert_eq!(
                            smg.decode(ids, true).expect("decode"),
                            hf.decode(ids, true).expect("reference")
                        );
                    }
                });
            }
        });
        println!(
            "{}",
            json!({"kind":"parity", "model":name, "cases":count, "concurrent_cases":800})
        );
        return;
    }
    let mut specials: Vec<_> = hf
        .get_added_tokens_decoder()
        .into_iter()
        .filter(|(_, token)| token.special)
        .map(|(id, _)| id)
        .collect();
    specials.sort_unstable();
    assert!(!specials.is_empty());
    let mut special_ids = ids.clone();
    for (i, id) in special_ids.iter_mut().enumerate().step_by(17) {
        *id = specials[i % specials.len()];
    }
    let mut state = 7u64;
    let random_ids: Vec<_> = (0..80_000)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            if state.is_multiple_of(31) {
                u32::MAX
            } else {
                vocab[state as usize % vocab.len()]
            }
        })
        .collect();
    let ascii_ids = hf
        .encode(
            "the quick brown fox jumps over the lazy dog 12345\n".repeat(10_000),
            false,
        )
        .expect("ASCII encode")
        .get_ids()
        .to_vec();
    let unicode_ids = hf
        .encode(
            "中文 日本語 한국어 ไทย العربية 🧬👨‍👩‍👧‍👦 e\u{301}\n".repeat(10_000),
            false,
        )
        .expect("Unicode encode")
        .get_ids()
        .to_vec();
    // C3 A9 is a split 'é'; FF is invalid and the following 'x' settles it.
    let byte_ids: Vec<_> = ["Ã", "©", "ÿ", "x"]
        .into_iter()
        .map(|token| hf.token_to_id(token).expect("byte alphabet token"))
        .collect();
    let fragment_ids: Vec<_> = byte_ids.into_iter().cycle().take(80_000).collect();
    for size in [0, 1, 128, 2048, 50_000, 80_000] {
        for (scenario, input) in [
            ("mixed", &ids),
            ("special", &special_ids),
            ("random", &random_ids),
            ("ascii", &ascii_ids),
            ("unicode", &unicode_ids),
            ("byte-fragments", &fragment_ids),
        ] {
            for skip in [false, true] {
                run(
                    &hf,
                    &smg,
                    Case {
                        model: name,
                        scenario,
                        ids: &input[..size],
                        skip,
                        streaming: false,
                        callers: 1,
                    },
                    duration,
                );
            }
        }
    }
    for callers in [4, 16] {
        for size in [2048, 80_000] {
            run(
                &hf,
                &smg,
                Case {
                    model: name,
                    scenario: "mixed",
                    ids: &ids[..size],
                    skip: true,
                    streaming: false,
                    callers,
                },
                duration,
            );
        }
    }
    for (scenario, input) in [
        ("mixed", &ids),
        ("special", &special_ids),
        ("random", &random_ids),
        ("byte-fragments", &fragment_ids),
    ] {
        for skip in [false, true] {
            run(
                &hf,
                &smg,
                Case {
                    model: name,
                    scenario,
                    ids: &input[..2048],
                    skip,
                    streaming: true,
                    callers: 1,
                },
                duration,
            );
        }
    }
}

fn controls(duration: Duration) {
    for decoder in [
        json!({"type":"ByteLevel", "add_prefix_space":true, "trim_offsets":true, "use_regex":true}),
        json!({"type":"Sequence","decoders":[{"type":"ByteLevel", "add_prefix_space":true, "trim_offsets":true, "use_regex":true}]}),
        json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"}]}),
        Value::Null,
    ] {
        let mut vocab = serde_json::Map::new();
        for byte in 0..256 {
            vocab.insert(format!("<0x{byte:02X}>"), json!(byte));
        }
        vocab.insert("Ã".into(), json!(256));
        vocab.insert("©".into(), json!(257));
        vocab.insert("x".into(), json!(258));
        let definition = json!({"version":"1.0", "model":{"type":"BPE","vocab":vocab,"merges":[]},"decoder":decoder});
        let hf = Tokenizer::from_bytes(serde_json::to_vec(&definition).expect("fixture JSON"))
            .expect("fixture");
        let smg: Arc<dyn SmgTokenizer> = Arc::new(HuggingFaceTokenizer::from_tokenizer(hf.clone()));
        let scenario = decoder.to_string();
        let ids: Vec<_> = (0..2048)
            .map(|i| [0xe6, 0x97, 0xa5, 0xff, 256, 258, 257, u32::MAX][i % 8])
            .collect();
        run(
            &hf,
            &smg,
            Case {
                model: "synthetic",
                scenario: &scenario,
                ids: &ids,
                skip: false,
                streaming: false,
                callers: 1,
            },
            duration,
        );
    }
}

fn main() {
    parallelism::set_parallelism(false);
    #[cfg(feature = "tokenizers-v1")]
    tokenizers_v1::parallelism::set_parallelism(false);
    let args: Vec<_> = env::args().collect();
    let root = Path::new(
        args.get(1)
            .expect("usage: decode_compare MODEL_ROOT [duration_ms] [check]"),
    );
    let duration = Duration::from_millis(
        args.get(2)
            .map_or(Ok(200), |s| s.parse())
            .expect("duration_ms"),
    );
    let check = args.iter().any(|s| s == "check");
    for name in ["minimax-m3", "glm-5.3", "qwen3-8b", "deepseek-v3.2"] {
        real_model(name, root, duration, check);
    }
    controls(if check { Duration::ZERO } else { duration });
}
