//! A Responses request the Harmony tool loop serves with two model calls
//! reports the usage of both, in the plain and the streamed form, in the
//! answer and in the stored object.

use super::harmony_loop_harness::{serve, token_ids, End, Served, Turn, ANSWER};

/// The first generation calls the mock tool server's search tool, the second
/// answers.
const SERVER_CALL: &str = "<|channel|>commentary to=functions.brave_web_search <|constrain|>json<|message|>{\"query\": \"usage\"}<|call|>";

/// Distinct per generation, so one generation's usage alone cannot read as
/// the sum.
const PROMPT_TOKENS: [u32; 2] = [11, 23];

#[tokio::test]
async fn harmony_mcp_usage_includes_every_model_call_in_both_modes() {
    let turns: Vec<Turn> = token_ids(&[SERVER_CALL, ANSWER])
        .into_iter()
        .zip(PROMPT_TOKENS)
        .map(|(output_ids, prompt_tokens)| Turn {
            output_ids,
            prompt_tokens,
            end: End::Finish("stop"),
        })
        .collect();
    let output_tokens: u64 = turns.iter().map(|turn| turn.output_ids.len() as u64).sum();
    let input_tokens: u64 = PROMPT_TOKENS.iter().map(|&tokens| u64::from(tokens)).sum();

    // Each form's usage, in the answer and then in the stored object.
    let mut reported = Vec::new();
    for stream in [false, true] {
        let Served {
            response,
            stored,
            generations,
            tool_calls,
            ..
        } = serve(&turns, stream).await;

        // Two model calls around one executed tool call, answered with the
        // second's text.
        assert_eq!(generations, 2, "stream={stream}: {response}");
        assert_eq!(tool_calls, 1, "stream={stream}: {response}");
        assert_eq!(
            response["status"], "completed",
            "stream={stream}: {response}"
        );
        assert!(
            response["output"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["type"] == "message"
                    && item["content"][0]["text"] == "The answer after the search"),
            "stream={stream}: {response}"
        );
        for usage in [&response["usage"], &stored["usage"]] {
            reported.push((
                usage["input_tokens"].as_u64(),
                usage["output_tokens"].as_u64(),
                usage["total_tokens"].as_u64(),
            ));
        }
    }
    let both_calls = (
        Some(input_tokens),
        Some(output_tokens),
        Some(input_tokens + output_tokens),
    );
    assert_eq!(reported, vec![both_calls; 4]);
}
