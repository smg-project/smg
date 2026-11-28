//! The paths a cell drives: the client's arguments (a deterministic prompt set per seed)
//! and the scripted answer every request of the cell gets. Rates are per second at
//! scale 1.

use std::sync::LazyLock;

use anyhow::{bail, Result};
use serde_json::{json, Value};

use crate::{bed::Fleet, SIDES};

const WORDS: [&str; 9] = [
    "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
];

/// `n` words of the pangram, cycled.
pub fn words(n: usize) -> String {
    (0..n)
        .map(|i| WORDS[i % WORDS.len()])
        .collect::<Vec<_>>()
        .join(" ")
}

pub const TOOL_CALL_ANSWER: &str =
    r#"{"name": "lookup_1", "parameters": {"key": "k1-0", "fields": ["a", "b"], "limit": 10}}"#;
pub const JSON_ANSWER: &str =
    r#"{"answer": "the record is open", "confidence": 42, "status": "ok"}"#;

/// One path: its name, its offered rate at scale 1, the load client's flags, the
/// scripted answer, the finish reason, and the label the tables print.
#[derive(Clone, Debug, PartialEq)]
pub struct PathSpec {
    pub name: &'static str,
    pub rate: f64,
    pub client: &'static [&'static str],
    pub answer: String,
    pub finish: &'static str,
    /// The pause between the frames of a streamed answer, in milliseconds: 0
    /// hands a whole answer over at once (an engine's burst), a pause spaces
    /// the frames as an engine decoding one token per step does.
    pub frame_pause_ms: u64,
    pub label: &'static str,
}

pub static PATHS: LazyLock<Vec<PathSpec>> = LazyLock::new(|| {
    vec![
        PathSpec {
            name: "chat",
            rate: 500.0,
            client: &["--mode", "chat", "--max-tokens", "8", "--seed", "101"],
            answer: words(8),
            finish: "stop",
            frame_pause_ms: 0,
            label: "chat, non-stream, 64-word prompt, 8 output tokens",
        },
        PathSpec {
            name: "stream",
            rate: 250.0,
            client: &[
                "--mode",
                "chat",
                "--stream",
                "--max-tokens",
                "128",
                "--seed",
                "102",
            ],
            answer: words(128),
            finish: "length",
            frame_pause_ms: 0,
            label: "chat stream, 128 deltas of one token",
        },
        PathSpec {
            name: "tools",
            rate: 500.0,
            client: &[
                "--mode",
                "tools",
                "--turns",
                "2",
                "--tools",
                "4",
                "--max-tokens",
                "64",
                "--seed",
                "103",
            ],
            answer: TOOL_CALL_ANSWER.to_string(),
            finish: "stop",
            frame_pause_ms: 0,
            label: "tool call: 4 schemas, 2 earlier rounds, the answer parsed into tool_calls",
        },
        PathSpec {
            name: "json",
            rate: 500.0,
            client: &[
                "--mode",
                "chat",
                "--json-schema",
                "--max-tokens",
                "64",
                "--seed",
                "104",
            ],
            answer: JSON_ANSWER.to_string(),
            finish: "stop",
            frame_pause_ms: 0,
            label: "json_schema answer",
        },
        PathSpec {
            name: "messages",
            rate: 500.0,
            client: &["--mode", "messages", "--max-tokens", "8", "--seed", "105"],
            answer: words(8),
            finish: "stop",
            frame_pause_ms: 0,
            label: "Messages API, non-stream, 8 output tokens",
        },
        // The streamed answers below are not in `DEFAULT_PATHS`: each prices
        // one stream shape (`PERF_PATHS` names them), at rates that keep the
        // frames per second near the `stream` path's.
        PathSpec {
            name: "messages_stream",
            rate: 100.0,
            client: &[
                "--mode",
                "messages",
                "--stream",
                "--max-tokens",
                "300",
                "--seed",
                "106",
            ],
            answer: words(300),
            finish: "length",
            frame_pause_ms: 0,
            label: "Messages stream, 300 deltas of one token",
        },
        PathSpec {
            name: "generate_stream",
            rate: 100.0,
            client: &[
                "--mode",
                "generate",
                "--stream",
                "--max-tokens",
                "300",
                "--seed",
                "107",
            ],
            answer: words(300),
            finish: "length",
            frame_pause_ms: 0,
            label: "generate stream, 300 frames of one token, the whole text in every frame",
        },
        PathSpec {
            name: "completion_stream",
            rate: 100.0,
            client: &[
                "--mode",
                "completion",
                "--stream",
                "--max-tokens",
                "300",
                "--seed",
                "109",
            ],
            answer: words(300),
            finish: "length",
            frame_pause_ms: 0,
            label: "completion stream, 300 deltas of one token",
        },
        PathSpec {
            name: "generate_stream_long",
            rate: 16.0,
            client: &[
                "--mode",
                "generate",
                "--stream",
                "--max-tokens",
                "2000",
                "--seed",
                "108",
            ],
            answer: words(2000),
            finish: "length",
            frame_pause_ms: 0,
            label: "generate stream, 2,000 frames of one token, the whole text in every frame",
        },
        // The generate wire at an engine's pace: the same 300 frames with a
        // 20 ms pause between them (50 tokens a second per stream), at the rate
        // that keeps the frames per second at `generate_stream`'s, so about
        // 600 streams are open at once. The paths above hand a whole answer
        // over at once (an engine's burst); this one reads the stream loops
        // as a decoding engine drives them, one frame at a time.
        PathSpec {
            name: "generate_stream_spaced",
            rate: 100.0,
            client: &[
                "--mode",
                "generate",
                "--stream",
                "--max-tokens",
                "300",
                "--seed",
                "110",
            ],
            answer: words(300),
            finish: "length",
            frame_pause_ms: 20,
            label: "generate stream, 300 frames of one token 20 ms apart, the whole text in every frame",
        },
        // The paced twins of the other streamed shapes: the same answers, 20 ms
        // between frames, the rates of the burst paths (so the frames per second
        // hold and the open streams multiply the same way).
        PathSpec {
            name: "stream_spaced",
            rate: 250.0,
            client: &[
                "--mode",
                "chat",
                "--stream",
                "--max-tokens",
                "128",
                "--seed",
                "111",
            ],
            answer: words(128),
            finish: "length",
            frame_pause_ms: 20,
            label: "chat stream, 128 deltas of one token 20 ms apart",
        },
        PathSpec {
            name: "completion_stream_spaced",
            rate: 100.0,
            client: &[
                "--mode",
                "completion",
                "--stream",
                "--max-tokens",
                "300",
                "--seed",
                "112",
            ],
            answer: words(300),
            finish: "length",
            frame_pause_ms: 20,
            label: "completion stream, 300 deltas of one token 20 ms apart",
        },
        PathSpec {
            name: "messages_stream_spaced",
            rate: 100.0,
            client: &[
                "--mode",
                "messages",
                "--stream",
                "--max-tokens",
                "300",
                "--seed",
                "113",
            ],
            answer: words(300),
            finish: "length",
            frame_pause_ms: 20,
            label: "Messages stream, 300 deltas of one token 20 ms apart",
        },
        // The generate wire at an engine's pace on a long answer: 5,000 frames
        // of one token 20 ms apart, the whole text in every frame, so the text
        // passes the gateway's 16 KiB frame bound about two thirds of the way
        // through and each of the last 1,700 or so events fills a frame of its
        // own, 20 ms after the one before (the shape of a long reasoning
        // answer). `generate_stream_long` fills its frames from a burst (its
        // events stay under 10 KiB; two or three make a frame) and the
        // 300-frame spaced path never reaches the bound; this one reads what a
        // stream loop's frame buffer does between full frames at an engine's
        // pace. The rate keeps the frames per second at
        // `generate_stream`'s: an answer takes about 100 s, so about 600
        // streams are open at once, and a cell of 10,000 requests runs about
        // half an hour (a recipe names a smaller `--requests`). The answer's
        // pauses (about 105 s with the timer's granularity) fit the load
        // client's 120 s drain with about 15 s to spare, and a `live` ramp on
        // it names its own `--rates` (the default steps start at 200/s, the
        // burst paths' pace) and `--slo-e2e-p99-ms` above about 110,000
        // (above the answer, under the drain); a longer or slower paced path
        // needs the drain made a knob first.
        PathSpec {
            name: "generate_stream_spaced_long",
            rate: 6.0,
            client: &[
                "--mode",
                "generate",
                "--stream",
                "--max-tokens",
                "5000",
                "--seed",
                "114",
            ],
            answer: words(5000),
            finish: "length",
            frame_pause_ms: 20,
            label: "generate stream, 5,000 frames of one token 20 ms apart, the whole text in every frame",
        },
        // The stream with the sampled logprob and five candidates per token,
        // the gateway reading, converting and decoding six token ids per
        // delta. Its rate is the gateway's load, not the frame rate: such an
        // answer costs about 3.5 times the instructions and 13 times the
        // allocation calls of the plain stream's, so at 100/s the cell runs
        // about 1.4 times the `stream` cell's gateway instructions per second
        // and about five times its allocation calls (3.5 and 13 times them at
        // the `stream` rate).
        PathSpec {
            name: "stream_logprobs",
            rate: 100.0,
            client: &[
                "--mode",
                "chat",
                "--stream",
                "--max-tokens",
                "128",
                "--logprobs",
                "5",
                "--seed",
                "109",
            ],
            answer: words(128),
            finish: "length",
            frame_pause_ms: 0,
            label:
                "chat stream, 128 deltas of one token, the sampled logprob and 5 candidates each",
        },
    ]
});

pub const DEFAULT_PATHS: &str = "chat,stream,tools,json,messages";
pub const DEFAULT_GW_ARGS: &str = "--parser symphony --tool-call-parser json";

/// The path by name.
pub fn path_named(name: &str) -> Option<&'static PathSpec> {
    PATHS.iter().find(|p| p.name == name)
}

/// The comma-separated path list, checked and in the given order.
pub fn parse_paths(text: &str) -> Result<Vec<PathSpec>> {
    let names: Vec<&str> = text
        .split(',')
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .collect();
    let unknown: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| path_named(n).is_none())
        .collect();
    if !unknown.is_empty() {
        let known: Vec<&str> = PATHS.iter().map(|p| p.name).collect();
        bail!("unknown path(s) {unknown:?}; known: {}", known.join(", "));
    }
    if names.is_empty() {
        bail!("no path given");
    }
    Ok(names
        .iter()
        .filter_map(|n| path_named(n).cloned())
        .collect())
}

/// The paths of a run on `fleet`: the named ones, or the fleet's default list, each
/// one the fleet serves (the HTTP fleet refuses the generate paths by name and reason).
pub fn fleet_paths(fleet: Fleet, text: Option<&str>) -> Result<Vec<PathSpec>> {
    let paths = parse_paths(text.unwrap_or_else(|| fleet.default_paths()))?;
    if let Some(unserved) = paths.iter().find(|p| !fleet.serves(p.name)) {
        bail!(
            "the {} path is not served on the {} fleet (the mock engine's HTTP wire answers /generate in the chat shape); name --paths without it",
            unserved.name,
            fleet.as_str()
        );
    }
    Ok(paths)
}

/// The scripted answers of one cell: `count` copies, served in order to the next
/// requests of every worker (the mock worker's shared queue); one token per streamed
/// frame, the path's pause between frames (none on the burst paths). On a token wire
/// (`grpc`) the text goes as one string the worker encodes with the bed's tokenizer; on
/// the text wire (`http`) it goes as one piece per word, the units that wire streams
/// and cuts by, so a streamed answer frames one token at a time there too. A path whose
/// requests ask for logprobs gets the worker's made-up sampled logprobs on a token wire,
/// or one scripted value per piece on the text wire (which makes up none); the worker
/// draws the candidates from the answer's own tokens either way.
pub fn answers(spec: &PathSpec, count: usize, wire: &str) -> Vec<Value> {
    let output_text = if wire == "http" {
        json!(pieces(&spec.answer))
    } else {
        json!(spec.answer)
    };
    let mut answer = json!({
        "output_text": output_text,
        "finish_reason": spec.finish,
        "chunk_tokens": 1,
        "delay_ms": spec.frame_pause_ms,
    });
    if spec.client.contains(&"--logprobs") {
        answer["logprobs"] = if wire == "http" {
            // The text wire makes up no sampled logprobs (it has no ids): one scripted
            // value per piece, and the worker draws the candidates from the pieces.
            let pieces = answer["output_text"].as_array().map_or(0, Vec::len);
            json!((0..pieces)
                .map(|i| -0.125 * ((i % 8) as f64 + 1.0))
                .collect::<Vec<_>>())
        } else {
            json!("sampled")
        };
    }
    (0..count).map(|_| answer.clone()).collect()
}

/// The text as pieces of one word each, every piece after the first carrying its
/// leading space, so the pieces concatenate back to the text.
fn pieces(text: &str) -> Vec<String> {
    text.split(' ')
        .enumerate()
        .map(|(i, word)| {
            if i == 0 {
                word.to_string()
            } else {
                format!(" {word}")
            }
        })
        .collect()
}

/// Repeat i alternates which side goes first (as scripts/bench_gate.py does).
pub fn interleaved_order(repeats: usize) -> Vec<(usize, &'static str)> {
    let mut order = Vec::with_capacity(repeats * 2);
    for i in 1..=repeats {
        if i % 2 == 1 {
            order.extend(SIDES.iter().map(|side| (i, *side)));
        } else {
            order.extend(SIDES.iter().rev().map(|side| (i, *side)));
        }
    }
    order
}

#[cfg(test)]
#[path = "../tests/unit/paths/tests.rs"]
mod tests;
