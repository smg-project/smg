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
mod tests {
    use super::*;

    #[test]
    fn paths_are_checked_and_ordered() {
        let names: Vec<&str> = parse_paths("stream, chat")
            .unwrap()
            .iter()
            .map(|p| p.name)
            .collect();
        assert_eq!(names, ["stream", "chat"]);
        let err = parse_paths("chat,nosuch").unwrap_err().to_string();
        assert!(err.contains("unknown path"), "{err}");
        let err = parse_paths(" , ").unwrap_err().to_string();
        assert!(err.contains("no path"), "{err}");
    }

    #[test]
    fn every_path_has_a_rate_a_client_mode_and_an_answer() {
        for spec in &*PATHS {
            assert!(spec.rate > 0.0);
            assert!(spec.client.contains(&"--mode"));
            assert!(spec.client.contains(&"--seed"));
            assert!(!spec.answer.is_empty());
            assert!(spec.finish == "stop" || spec.finish == "length");
        }
        let stream = path_named("stream").unwrap();
        assert_eq!(stream.finish, "length");
        assert!(stream.client.contains(&"--stream"));
        for (name, tokens) in [
            ("messages_stream", 300),
            ("generate_stream", 300),
            ("completion_stream", 300),
            ("generate_stream_long", 2000),
            ("generate_stream_spaced", 300),
            ("stream_spaced", 128),
            ("completion_stream_spaced", 300),
            ("messages_stream_spaced", 300),
            ("generate_stream_spaced_long", 5000),
        ] {
            let spec = path_named(name).unwrap();
            assert_eq!(spec.finish, "length", "{name}");
            assert!(spec.client.contains(&"--stream"), "{name}");
            assert_eq!(spec.answer.split_whitespace().count(), tokens, "{name}");
            assert!(
                !DEFAULT_PATHS.split(',').any(|p| p == name),
                "{name} is opt-in"
            );
        }
        let tools: Value = serde_json::from_str(&path_named("tools").unwrap().answer).unwrap();
        assert!(tools["name"].as_str().unwrap().starts_with("lookup_"));
        let json: Value = serde_json::from_str(&path_named("json").unwrap().answer).unwrap();
        let keys: Vec<&String> = json.as_object().unwrap().keys().collect();
        assert_eq!(keys, ["answer", "confidence", "status"]);
    }

    #[test]
    fn answers_are_one_per_request_served_one_token_a_frame() {
        let stream = path_named("stream").unwrap();
        let items = answers(stream, 12, "grpc");
        assert_eq!(items.len(), 12);
        assert_eq!(
            items[0],
            json!({
                "output_text": stream.answer,
                "finish_reason": "length",
                "chunk_tokens": 1,
                "delay_ms": 0,
            })
        );
        assert_eq!(stream.answer.split_whitespace().count(), 128);
        assert_eq!(
            path_named("chat")
                .unwrap()
                .answer
                .split_whitespace()
                .count(),
            8
        );
    }

    #[test]
    fn the_text_wire_gets_the_answer_as_one_piece_per_word() {
        // The HTTP wire streams and cuts by text pieces, so a 128-token stream frames
        // 128 times there only when the text arrives as 128 pieces; the pieces
        // concatenate back to the text the token wire encodes.
        let stream = path_named("stream").unwrap();
        let http = answers(stream, 2, "http");
        assert_eq!(http.len(), 2);
        let pieces = http[0]["output_text"].as_array().unwrap();
        assert_eq!(pieces.len(), 128);
        assert_eq!(pieces[0], json!("The"));
        assert_eq!(pieces[1], json!(" quick"));
        let joined: String = pieces.iter().map(|p| p.as_str().unwrap()).collect();
        assert_eq!(joined, stream.answer);
        assert_eq!(http[0]["chunk_tokens"], json!(1));
        // A path that asks for logprobs gets one scripted value per piece on the text
        // wire (that wire makes up none) and the worker's sampled ones on a token wire.
        let logprobs = path_named("stream_logprobs").unwrap();
        let http = answers(logprobs, 1, "http");
        assert_eq!(
            http[0]["logprobs"].as_array().unwrap().len(),
            http[0]["output_text"].as_array().unwrap().len()
        );
        assert_eq!(
            answers(logprobs, 1, "grpc")[0]["logprobs"],
            json!("sampled")
        );
        let tools = answers(path_named("tools").unwrap(), 1, "http");
        let joined: String = tools[0]["output_text"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap())
            .collect();
        assert_eq!(joined, TOOL_CALL_ANSWER);
        assert_eq!(
            answers(stream, 1, "grpc")[0]["output_text"],
            json!(stream.answer)
        );
    }

    #[test]
    fn a_fleet_names_its_default_paths_and_refuses_the_ones_it_does_not_serve() {
        let names = |paths: Vec<PathSpec>| paths.iter().map(|p| p.name).collect::<Vec<_>>();
        assert_eq!(
            names(fleet_paths(Fleet::Grpc, None).unwrap()),
            ["chat", "stream", "tools", "json", "messages"]
        );
        assert_eq!(
            names(fleet_paths(Fleet::Http, None).unwrap()),
            ["chat", "stream", "tools", "json", "messages"]
        );
        assert_eq!(
            names(fleet_paths(Fleet::PrefillDecode, Some("messages_stream")).unwrap()),
            ["messages_stream"]
        );
        assert_eq!(
            names(fleet_paths(Fleet::Http, Some("chat,messages")).unwrap()),
            ["chat", "messages"]
        );
        let err = fleet_paths(Fleet::Http, Some("chat,generate_stream_spaced_long"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("generate_stream_spaced_long path is not served on the http fleet"),
            "{err}"
        );
        assert!(fleet_paths(Fleet::Http, Some("nosuch")).is_err());
    }

    #[test]
    fn the_spaced_streams_pace_their_frames_and_the_burst_paths_do_not() {
        for name in [
            "generate_stream_spaced",
            "stream_spaced",
            "completion_stream_spaced",
            "messages_stream_spaced",
            "generate_stream_spaced_long",
        ] {
            let spaced = path_named(name).unwrap();
            assert_eq!(spaced.frame_pause_ms, 20, "{name}");
            assert_eq!(
                answers(spaced, 1, "grpc")[0]["delay_ms"],
                json!(20),
                "{name}"
            );
        }
        for spec in PATHS.iter().filter(|p| !p.name.contains("_spaced")) {
            assert_eq!(spec.frame_pause_ms, 0, "{}", spec.name);
            assert_eq!(
                answers(spec, 1, "grpc")[0]["delay_ms"],
                json!(0),
                "{}",
                spec.name
            );
        }
    }

    #[test]
    fn the_long_spaced_generate_stream_passes_the_frame_bound_on_its_last_thousand_frames() {
        // The gateway flushes a streamed frame at 16 KiB, and the generate wire
        // carries the whole text so far in every event: once the text passes
        // the bound, every event is a full frame of its own.
        const FRAME_BOUND: usize = 16 * 1024;
        let spec = path_named("generate_stream_spaced_long").unwrap();
        let tokens = spec.answer.split_whitespace().count();
        assert_eq!(tokens, 5000);
        assert!(
            words(tokens - 1000).len() > FRAME_BOUND,
            "the last thousand frames each carry more than the bound"
        );
        assert!(
            words(tokens / 2).len() < FRAME_BOUND,
            "the first half stays under it, as the 300-frame paths do"
        );
        let generate = path_named("generate_stream").unwrap();
        assert_eq!(
            spec.rate * tokens as f64,
            generate.rate * 300.0,
            "the frames per second are generate_stream's"
        );
    }

    #[test]
    fn the_logprobs_stream_is_opt_in_and_its_answers_carry_sampled_logprobs() {
        let spec = path_named("stream_logprobs").unwrap();
        assert_eq!(spec.finish, "length");
        assert!(spec.client.contains(&"--stream"));
        assert!(spec.client.contains(&"--logprobs"));
        assert_eq!(spec.answer, path_named("stream").unwrap().answer);
        assert!(
            !DEFAULT_PATHS.split(',').any(|p| p == "stream_logprobs"),
            "opt-in"
        );
        let scripted = answers(spec, 2, "grpc");
        assert_eq!(scripted.len(), 2);
        assert_eq!(scripted[0]["logprobs"], json!("sampled"));
        assert_eq!(
            answers(path_named("stream").unwrap(), 1, "grpc")[0].get("logprobs"),
            None,
            "the plain stream scripts none"
        );
    }

    #[test]
    fn interleaving_alternates_which_side_goes_first() {
        assert_eq!(
            interleaved_order(3),
            [
                (1, "base"),
                (1, "pr"),
                (2, "pr"),
                (2, "base"),
                (3, "base"),
                (3, "pr")
            ]
        );
    }
}
