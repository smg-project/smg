//! Narrow synthetic request profile; both gRPC adapters use one engine stream.
use super::{super::SYNTHETIC_VOCAB_SIZE, ts, v, Status};

pub(super) fn normalize(
    req: v::GenerateRequest,
    context: u32,
) -> Result<ts::GenerateRequest, Status> {
    let v::GenerateRequest {
        request_id,
        input,
        sampling_params,
        stream,
        kv_transfer_params,
        mm_inputs,
        extra_mm_inputs,
        kv_transfer_params_json,
        data_parallel_rank,
        media_refs,
    } = req;
    if request_id.is_empty() {
        return Err(Status::invalid_argument("request_id must be nonempty"));
    }
    if kv_transfer_params.is_some()
        || kv_transfer_params_json.is_some()
        || mm_inputs.is_some()
        || !extra_mm_inputs.is_empty()
        || media_refs.is_some()
    {
        return Err(Status::unimplemented(
            "mock vLLM generation supports only local text tokens",
        ));
    }
    if data_parallel_rank.is_some_and(|rank| rank != 0) {
        return Err(Status::invalid_argument("mock-worker has only DP rank 0"));
    }
    let Some(v::generate_request::Input::Tokenized(input)) = input else {
        return Err(Status::invalid_argument("tokenized input is required"));
    };
    if input.input_ids.is_empty() || input.input_ids.iter().any(|id| *id >= SYNTHETIC_VOCAB_SIZE) {
        return Err(Status::invalid_argument(format!(
            "prompt must contain tokens in the synthetic {SYNTHETIC_VOCAB_SIZE}-token vocabulary"
        )));
    }
    let p =
        sampling_params.ok_or_else(|| Status::invalid_argument("sampling_params are required"))?;
    let v::SamplingParams {
        temperature,
        top_p,
        top_k,
        min_p,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
        max_tokens,
        min_tokens,
        stop,
        stop_token_ids,
        skip_special_tokens,
        spaces_between_special_tokens,
        ignore_eos,
        n,
        logprobs,
        prompt_logprobs,
        seed,
        include_stop_str_in_output,
        logit_bias,
        truncate_prompt_tokens,
        grammar_from_first_token,
        constraint,
    } = p;
    if temperature != Some(0.0)
        || top_p != 1.0
        || top_k != 0
        || min_p != 0.0
        || frequency_penalty != 0.0
        || presence_penalty != 0.0
        || repetition_penalty != 1.0
        || min_tokens != 0
        || n != 1
        || !stop.is_empty()
        || !stop_token_ids.is_empty()
        || logprobs.is_some()
        || prompt_logprobs.is_some()
        || seed.is_some()
        || include_stop_str_in_output
        || !logit_bias.is_empty()
        || truncate_prompt_tokens.is_some()
        || grammar_from_first_token
        || constraint.is_some()
    {
        return Err(Status::unimplemented("mock vLLM generation requires temperature=0, neutral sampling, n=1 and no stop, logprob, seed or constraint controls"));
    }
    let max = max_tokens
        .filter(|max| *max > 0)
        .ok_or_else(|| Status::invalid_argument("positive max_tokens is required"))?;
    if input.input_ids.len() as u64 + u64::from(max) > u64::from(context) {
        return Err(Status::invalid_argument(
            "prompt plus output exceeds the advertised context",
        ));
    }
    Ok(ts::GenerateRequest {
        request_id,
        tokenized: Some(ts::TokenizedInput {
            input_ids: input.input_ids,
            original_text: input.original_text,
        }),
        sampling_params: Some(ts::SamplingParams {
            temperature,
            top_p: Some(top_p),
            top_k: Some(0),
            min_p: Some(min_p),
            frequency_penalty: Some(frequency_penalty),
            presence_penalty: Some(presence_penalty),
            repetition_penalty: Some(repetition_penalty),
            max_new_tokens: Some(max),
            n,
            skip_special_tokens,
            spaces_between_special_tokens,
            ignore_eos,
            ..Default::default()
        }),
        stream,
        data_parallel_rank,
        ..Default::default()
    })
}

pub(super) fn response(frame: ts::GenerateResponse) -> Result<v::GenerateResponse, Status> {
    use ts::generate_response::Response as Source;
    use v::generate_response::Response as Target;
    let response = match frame.response {
        Some(Source::Chunk(chunk)) => {
            if chunk.output_logprobs.is_some() {
                return Err(Status::internal(
                    "unexpected logprobs from the synthetic profile",
                ));
            }
            Target::Chunk(v::GenerateStreamChunk {
                token_ids: chunk.token_ids,
                prompt_tokens: chunk.prompt_tokens,
                completion_tokens: chunk.completion_tokens,
                cached_tokens: chunk.cached_tokens,
                index: chunk.index,
                ..Default::default()
            })
        }
        Some(Source::Complete(done)) => {
            if done.output_logprobs.is_some() {
                return Err(Status::internal(
                    "unexpected logprobs from the synthetic profile",
                ));
            }
            let matched_stop = done.matched_stop.map(|stop| match stop {
                ts::generate_complete::MatchedStop::MatchedTokenId(id) => {
                    v::generate_complete::MatchedStop::MatchedTokenId(id)
                }
                ts::generate_complete::MatchedStop::MatchedStopStr(text) => {
                    v::generate_complete::MatchedStop::MatchedStopStr(text)
                }
            });
            Target::Complete(v::GenerateComplete {
                output_ids: done.output_ids,
                finish_reason: done.finish_reason,
                prompt_tokens: done.prompt_tokens,
                completion_tokens: done.completion_tokens,
                cached_tokens: done.cached_tokens,
                index: done.index,
                matched_stop,
                spec_accepted_tokens: done.spec_accepted_tokens,
                spec_draft_tokens: done.spec_draft_tokens,
                ..Default::default()
            })
        }
        None => return Err(Status::internal("mock engine returned an empty response")),
    };
    Ok(v::GenerateResponse {
        response: Some(response),
    })
}
