//! SGLang dialect: proto request → scheduler request translation and
//! scheduler output → vLLM-proto response mapping.
//!
//! The scheduler side is SGLang's own, unchanged; it joins SMG's transport
//! through the SGLang plugin in `smg_grpc_servicer.sglang` (which performs
//! the handshake inside the scheduler process, normalizes and verifies the
//! sampling params SMG sends, and slices each step's output to the slim
//! batch decoded here).

use engine_zmq_client::{
    connector::SglangStream,
    protocol::sglang::{
        output::{MatchedStop, SglangOutput},
        request::TokenizedGenerateReqInput,
        sampling::SamplingParams as SglangSamplingParams,
    },
};
use futures::Stream;
use smg_grpc_client::{sglang_proto, vllm_proto as vllm};
use tracing::warn;

use crate::{
    fanout::fan_out_n,
    stream::{poll_mapped, MappedGenerateStream, StreamState},
};

/// Streaming generate output for one SGLang sub-request, mapping each
/// [`SglangOutput`] to a vLLM-proto `GenerateResponse` tagged with this sub's
/// choice `index`.
pub struct SglangGenerateStream {
    inner: SglangStream,
    state: StreamState,
    /// Choice index stamped on every chunk/complete (0 for n=1; the fan-out
    /// position for n>1).
    index: u32,
    /// Terminal `Complete` held back when the finish tick also carried new
    /// tokens: the tick's delta goes out as a `Chunk` first.
    pending: Option<vllm::GenerateResponse>,
}

impl SglangGenerateStream {
    pub(crate) fn new(inner: SglangStream, index: u32) -> Self {
        Self {
            inner,
            state: StreamState::default(),
            index,
            pending: None,
        }
    }
}

impl MappedGenerateStream for SglangGenerateStream {
    type Output = SglangOutput;
    type Inner = SglangStream;

    fn inner(&mut self) -> &mut Self::Inner {
        &mut self.inner
    }

    fn pending(&mut self) -> &mut Option<vllm::GenerateResponse> {
        &mut self.pending
    }

    fn map_output(
        &mut self,
        output: SglangOutput,
    ) -> Result<vllm::GenerateResponse, tonic::Status> {
        let state = &mut self.state;
        // The scheduler reports per-request counts directly (cumulative for
        // completions).
        if output.prompt_tokens > 0 {
            state.prompt_tokens = output.prompt_tokens;
        }
        if output.cached_tokens > 0 {
            state.cached_tokens = output.cached_tokens;
        }
        state.completion_tokens = output.completion_tokens;
        state.output_ids.extend(output.output_ids.iter().copied());

        // Ranked candidates per position, as the scheduler reports them (the
        // gRPC servicer forwards the same lists unchanged).
        let tick_top_logprobs: Vec<vllm::TopLogProbs> = output
            .output_top_logprobs_val
            .iter()
            .zip(&output.output_top_logprobs_idx)
            .map(|(values, token_ids)| vllm::TopLogProbs {
                values: values.iter().map(|&lp| lp as f32).collect(),
                token_ids: token_ids.clone(),
            })
            .collect();
        let chunk_logprobs =
            (!output.output_logprobs_val.is_empty()).then(|| vllm::OutputLogProbs {
                token_logprobs: output
                    .output_logprobs_val
                    .iter()
                    .map(|&lp| lp as f32)
                    .collect(),
                token_ids: output.output_logprobs_idx.clone(),
                top_logprobs: tick_top_logprobs.clone(),
            });
        state
            .output_logprobs_val
            .extend(output.output_logprobs_val.iter().map(|&lp| lp as f32));
        state
            .output_logprobs_idx
            .extend(output.output_logprobs_idx.iter().copied());
        state.output_top_logprobs.extend(tick_top_logprobs);

        // SGLang has no separate error finish: a request it could not serve
        // (rejected by the plugin's validation, refused at admission, dropped
        // from a full queue) ends as `abort` with the reason and, when the
        // request is at fault, HTTP 400. The frontend never asked for it (its
        // own aborts drop the stream first), so it is an error to the caller,
        // not an empty success, with the status mapped as SGLang's gRPC
        // servicer maps it.
        if output.finish_reason.as_deref() == Some("abort") {
            let message = format!(
                "the scheduler aborted the request: {}",
                output
                    .finish_message
                    .as_deref()
                    .unwrap_or("no reason reported")
            );
            return Err(match output.finish_status {
                Some(400) => tonic::Status::invalid_argument(message),
                Some(503) => tonic::Status::unavailable(message),
                _ => tonic::Status::internal(message),
            });
        }
        let finish = output.finish_reason.map(|reason| {
            (
                normalize_finish_reason(&reason).to_string(),
                output.matched_stop.map(|matched| match matched {
                    MatchedStop::TokenId(id) => {
                        vllm::generate_complete::MatchedStop::MatchedTokenId(id)
                    }
                    MatchedStop::Text(text) => {
                        vllm::generate_complete::MatchedStop::MatchedStopStr(text)
                    }
                }),
            )
        });
        Ok(state.emit_tick(
            self.index,
            output.output_ids,
            chunk_logprobs,
            finish,
            &mut self.pending,
        ))
    }
}

impl Stream for SglangGenerateStream {
    type Item = Result<vllm::GenerateResponse, tonic::Status>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        poll_mapped(self.get_mut(), cx)
    }
}

/// Split an `n > 1` SGLang proto request into `n` single-sample sub-requests
/// (the plugin refuses `n != 1`: the frontend owns the fan-out). The SGLang
/// proto carries no seed, so the samples are independent by construction.
/// The `{rid}-{i}` suffix is the shared fan-out convention; note the scheduler
/// aborts by rid *prefix*, so aborting one sub also aborts the subs whose
/// index extends it (`r-1` covers `r-10..`). The gateway only ever drops a
/// whole fan-out, so that over-match is harmless today.
pub(crate) fn fan_out_sglang_requests(
    req: sglang_proto::GenerateRequest,
) -> Vec<sglang_proto::GenerateRequest> {
    let n = req.sampling_params.as_ref().map_or(1, |sp| sp.n.max(1));
    fan_out_n(req, n, |sub, i| {
        sub.request_id = format!("{}-{i}", sub.request_id);
        if let Some(sp) = sub.sampling_params.as_mut() {
            sp.n = 1;
        }
    })
}

/// Translate an SGLang proto `GenerateRequest` into the scheduler's
/// `TokenizedGenerateReqInput`. ZMQ mode requires pre-tokenized input (SMG
/// tokenizes upstream), and the slim output carries output logprobs only
/// (sampled token and ranked candidates): prompt logprobs, hidden states,
/// multimodal payloads, LoRA and custom logit processing are refused rather
/// than silently dropped.
pub(crate) fn translate_request_sglang(
    req: sglang_proto::GenerateRequest,
) -> Result<TokenizedGenerateReqInput, String> {
    let input_ids = req
        .tokenized
        .map(|tokenized| tokenized.input_ids)
        .ok_or_else(|| "ZMQ mode requires pre-tokenized input; no input provided".to_string())?;
    if req.mm_inputs.is_some() {
        return Err(
            "multimodal inputs are not supported over the SGLang ZMQ backend yet".to_string(),
        );
    }
    if req.return_logprob && req.logprob_start_len >= 0 {
        return Err(
            "prompt logprobs (logprob_start_len >= 0) are not supported over the SGLang ZMQ \
             backend"
                .to_string(),
        );
    }
    if !req.token_ids_logprob.is_empty() {
        return Err("token_ids_logprob is not supported over the SGLang ZMQ backend".to_string());
    }
    if req.return_hidden_states {
        return Err("hidden states are not supported over the SGLang ZMQ backend".to_string());
    }
    // Slots past the emitted prefix stay at the scheduler's defaults; refuse
    // rather than silently run on the base model or without the processor.
    if !req.lora_id.is_empty() {
        return Err("LoRA adapters are not supported over the SGLang ZMQ backend yet".to_string());
    }
    if !req.custom_logit_processor.is_empty() {
        return Err(
            "custom logit processors are not supported over the SGLang ZMQ backend".to_string(),
        );
    }
    if req
        .sampling_params
        .as_ref()
        .is_some_and(|sp| sp.custom_params.is_some())
    {
        return Err(
            "custom sampling params are not supported over the SGLang ZMQ backend".to_string(),
        );
    }
    Ok(TokenizedGenerateReqInput {
        rid: req.request_id,
        input_ids,
        sampling_params: req
            .sampling_params
            .map(translate_sampling_sglang)
            .unwrap_or_default(),
        return_logprob: req.return_logprob,
        logprob_start_len: req.logprob_start_len,
        top_logprobs_num: u32::try_from(req.top_logprobs_num).unwrap_or(0),
        stream: req.stream,
        require_reasoning: req.require_reasoning,
        ..TokenizedGenerateReqInput::default()
    })
}

/// Map the proto sampling params onto SGLang's own struct, in its API-input
/// form: the plugin runs the scheduler's `normalize()`/`verify()` on receipt.
/// String `stop` sequences are not forwarded (the scheduler runs without a
/// tokenizer; the gateway resolved them upstream).
pub(crate) fn translate_sampling_sglang(sp: sglang_proto::SamplingParams) -> SglangSamplingParams {
    use sglang_proto::sampling_params::Constraint;

    let mut params = SglangSamplingParams {
        max_new_tokens: sp.max_new_tokens,
        stop_token_ids: (!sp.stop_token_ids.is_empty()).then_some(sp.stop_token_ids),
        temperature: f64::from(sp.temperature),
        top_p: f64::from(sp.top_p),
        top_k: sp.top_k,
        min_p: f64::from(sp.min_p),
        frequency_penalty: f64::from(sp.frequency_penalty),
        presence_penalty: f64::from(sp.presence_penalty),
        // proto3 cannot tell an unset float from 0.0; a zero penalty factor
        // is never meant and would zero every logit.
        repetition_penalty: if sp.repetition_penalty == 0.0 {
            1.0
        } else {
            f64::from(sp.repetition_penalty)
        },
        min_new_tokens: sp.min_new_tokens,
        n: sp.n.max(1),
        ignore_eos: sp.ignore_eos,
        skip_special_tokens: sp.skip_special_tokens,
        spaces_between_special_tokens: sp.spaces_between_special_tokens,
        no_stop_trim: sp.no_stop_trim,
        stream_interval: sp.stream_interval.and_then(|v| u32::try_from(v).ok()),
        logit_bias: (!sp.logit_bias.is_empty()).then(|| {
            sp.logit_bias
                .into_iter()
                .map(|(token, bias)| (token, f64::from(bias)))
                .collect()
        }),
        ..SglangSamplingParams::default()
    };
    match sp.constraint {
        Some(Constraint::JsonSchema(schema)) => params.json_schema = Some(schema),
        Some(Constraint::Regex(regex)) => params.regex = Some(regex),
        Some(Constraint::EbnfGrammar(grammar)) => params.ebnf = Some(grammar),
        Some(Constraint::StructuralTag(tag)) => params.structural_tag = Some(tag),
        None => {}
    }
    params
}

/// Normalize an SGLang finish reason into the canonical set the gateway's
/// response layer exact-matches (`stop`, `length`, `abort`).
pub(crate) fn normalize_finish_reason(reason: &str) -> &'static str {
    match reason {
        "stop" => "stop",
        "length" => "length",
        "abort" => "abort",
        other => {
            warn!(
                finish_reason = other,
                "unknown SGLang finish_reason; defaulting to \"stop\""
            );
            "stop"
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use engine_zmq_client::{
        codec::{decode_msgpack, encode_msgpack},
        mock_engine::{connect_to_frontend, default_ready_response},
        protocol::sglang::{
            output::BatchTokenIDSlimOutput,
            request::{SglangRequestType, TokenizedGenerateReqInput as WireRequest},
        },
        EngineId,
    };
    use openai_protocol::worker::RuntimeType;

    use super::*;
    use crate::{client::ZmqEngineClient, eos::EosTokenIds};

    #[test]
    fn sglang_sampling_translation_keeps_the_api_form_and_maps_the_constraint() {
        let params = translate_sampling_sglang(sglang_proto::SamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            top_k: -1,
            repetition_penalty: 0.0, // proto3 default: never a real factor
            max_new_tokens: Some(16),
            stop: vec!["###".into()],
            stop_token_ids: vec![2],
            n: 0,
            constraint: Some(sglang_proto::sampling_params::Constraint::JsonSchema(
                "{}".into(),
            )),
            logit_bias: [("5".to_string(), 1.5f32)].into_iter().collect(),
            ..Default::default()
        });
        assert_eq!(params.max_new_tokens, Some(16));
        assert_eq!(params.stop, None, "string stops are the gateway's job");
        assert_eq!(params.stop_token_ids, Some(vec![2]));
        assert_eq!(params.top_k, -1, "the scheduler resolves the API sentinel");
        assert_eq!(params.repetition_penalty, 1.0);
        assert_eq!(params.n, 1);
        assert_eq!(params.json_schema.as_deref(), Some("{}"));
        assert_eq!(params.logit_bias.unwrap()["5"], 1.5);
        assert!(!params.is_normalized);
    }

    #[test]
    fn sglang_request_translation_refuses_what_the_slim_wire_cannot_carry() {
        let base = sglang_proto::GenerateRequest {
            request_id: "r".into(),
            tokenized: Some(sglang_proto::TokenizedInput {
                input_ids: vec![1, 2],
                original_text: String::new(),
            }),
            ..Default::default()
        };
        assert!(translate_request_sglang(sglang_proto::GenerateRequest {
            tokenized: None,
            ..base.clone()
        })
        .is_err());
        assert_eq!(
            translate_request_sglang(sglang_proto::GenerateRequest {
                top_logprobs_num: 3,
                ..base.clone()
            })
            .unwrap()
            .top_logprobs_num,
            3
        );
        assert!(translate_request_sglang(sglang_proto::GenerateRequest {
            return_hidden_states: true,
            ..base.clone()
        })
        .is_err());
        assert!(translate_request_sglang(sglang_proto::GenerateRequest {
            return_logprob: true,
            logprob_start_len: 0,
            ..base.clone()
        })
        .is_err());
        assert!(translate_request_sglang(sglang_proto::GenerateRequest {
            token_ids_logprob: vec![5],
            ..base.clone()
        })
        .is_err());
        assert!(translate_request_sglang(sglang_proto::GenerateRequest {
            lora_id: "adapter".into(),
            ..base.clone()
        })
        .is_err());
        assert!(translate_request_sglang(sglang_proto::GenerateRequest {
            custom_logit_processor: "proc".into(),
            ..base.clone()
        })
        .is_err());
        assert!(translate_request_sglang(sglang_proto::GenerateRequest {
            sampling_params: Some(sglang_proto::SamplingParams {
                custom_params: Some(prost_types::Struct::default()),
                ..Default::default()
            }),
            ..base.clone()
        })
        .is_err());
        // Output logprobs only: `-1` is the builders' "no prompt logprobs"
        // (the proto default of 0 would ask for them from the first token).
        let ok = translate_request_sglang(sglang_proto::GenerateRequest {
            return_logprob: true,
            logprob_start_len: -1,
            stream: true,
            require_reasoning: true,
            ..base
        })
        .unwrap();
        assert_eq!(ok.rid, "r");
        assert!(ok.return_logprob && ok.stream && ok.require_reasoning);
    }

    #[test]
    fn sglang_fan_out_suffixes_rids_and_pins_n() {
        let subs = fan_out_sglang_requests(sglang_proto::GenerateRequest {
            request_id: "r".into(),
            sampling_params: Some(sglang_proto::SamplingParams {
                n: 3,
                ..Default::default()
            }),
            ..Default::default()
        });
        assert_eq!(
            subs.iter()
                .map(|s| s.request_id.as_str())
                .collect::<Vec<_>>(),
            ["r-0", "r-1", "r-2"]
        );
        assert!(subs
            .iter()
            .all(|s| s.sampling_params.as_ref().unwrap().n == 1));
    }

    /// End-to-end over ipc:// for an SGLang backend: the adapter frames the
    /// scheduler's own `TokenizedGenerateReqInput`, and maps slim batches
    /// back to vLLM-proto responses, matched stop included.
    #[tokio::test]
    async fn generate_e2e_translates_and_streams_sglang() {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::Sglang,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = client.expect("adapter connect");
        let engine = engine.expect("mock engine");

        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let frames = input.recv_frames().await.unwrap();
            assert_eq!(
                SglangRequestType::from_frame(frames[0].as_ref()),
                Some(SglangRequestType::Add)
            );
            let request: WireRequest = decode_msgpack(frames[1].as_ref()).unwrap();
            assert_eq!(request.rid, "r1");
            assert_eq!(request.input_ids, vec![1, 2, 3]);
            assert_eq!(request.sampling_params.max_new_tokens, Some(2));
            assert!(request.return_logprob);

            let chunk = BatchTokenIDSlimOutput {
                rids: vec!["r1".into()],
                output_ids: vec![vec![10]],
                finished_reasons: vec![String::new()],
                finished_messages: vec![None],
                finished_matched: vec![None],
                prompt_tokens: vec![3],
                completion_tokens: vec![1],
                cached_tokens: vec![0],
                output_token_logprobs_val: vec![vec![-0.5]],
                output_token_logprobs_idx: vec![vec![10]],
                ..Default::default()
            };
            let done = BatchTokenIDSlimOutput {
                rids: vec!["r1".into()],
                output_ids: vec![vec![11]],
                finished_reasons: vec!["stop".into()],
                finished_messages: vec![None],
                finished_matched: vec![Some(MatchedStop::TokenId(11))],
                prompt_tokens: vec![3],
                completion_tokens: vec![2],
                cached_tokens: vec![0],
                output_token_logprobs_val: vec![vec![-1.25]],
                output_token_logprobs_idx: vec![vec![11]],
                engine_index: 0,
                num_running: 1,
                num_waiting: 0,
                kv_used_tokens: 5,
                kv_total_tokens: 100,
                finished_status: vec![None],
                output_top_logprobs_val: vec![vec![vec![-1.25, -2.0]]],
                output_top_logprobs_idx: vec![vec![vec![11, 12]]],
            };
            for batch in [chunk, done] {
                output
                    .send_frames(vec![bytes::Bytes::from(encode_msgpack(&batch).unwrap())])
                    .await
                    .unwrap();
            }
        });

        let req = sglang_proto::GenerateRequest {
            request_id: "r1".to_string(),
            tokenized: Some(sglang_proto::TokenizedInput {
                input_ids: vec![1, 2, 3],
                original_text: String::new(),
            }),
            sampling_params: Some(sglang_proto::SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            return_logprob: true,
            logprob_start_len: -1,
            stream: true,
            ..Default::default()
        };
        let mut stream = client.generate_sglang(req).await.expect("generate");

        let first = stream.next().await.expect("chunk item").expect("chunk ok");
        match first.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                assert_eq!(chunk.token_ids, vec![10]);
                assert_eq!(chunk.prompt_tokens, 3);
                assert_eq!(chunk.output_logprobs.unwrap().token_logprobs, vec![-0.5]);
            }
            other => panic!("expected a chunk, got {other:?}"),
        }
        let mut complete = None;
        while let Some(item) = stream.next().await {
            let response = item.expect("stream ok");
            if let Some(vllm::generate_response::Response::Complete(done)) = response.response {
                complete = Some(done);
            }
        }
        let done = complete.expect("a terminal Complete");
        assert_eq!(done.finish_reason, "stop");
        assert_eq!(done.output_ids, vec![10, 11]);
        assert_eq!(done.completion_tokens, 2);
        assert_eq!(
            done.matched_stop,
            Some(vllm::generate_complete::MatchedStop::MatchedTokenId(11))
        );
        let done_logprobs = done.output_logprobs.unwrap();
        assert_eq!(done_logprobs.token_logprobs, vec![-0.5, -1.25]);
        // Ranked candidates: only the second token asked for them, and the
        // terminal Complete carries the cumulative list.
        assert_eq!(done_logprobs.top_logprobs.len(), 1);
        assert_eq!(done_logprobs.top_logprobs[0].token_ids, vec![11, 12]);
        // The piggybacked load reached the connector: the KV term as reported,
        // the queue counts zeroed once the rank's last request retired (the
        // batch that carried them was sampled before that finish committed).
        let loads = client.get_loads();
        assert_eq!(loads.loads.len(), 1);
        assert_eq!(loads.loads[0].num_running_reqs, 0);
        assert_eq!(loads.loads[0].token_usage, 0.05);
        engine_task.await.unwrap();
    }

    #[test]
    fn sglang_finish_reason_normalizes() {
        assert_eq!(normalize_finish_reason("stop"), "stop");
        assert_eq!(normalize_finish_reason("length"), "length");
        assert_eq!(normalize_finish_reason("abort"), "abort");
        assert_eq!(normalize_finish_reason("mystery"), "stop");
    }

    /// `n = 2` over the SGLang wire: two single-sample sub-requests with
    /// suffixed rids, both finished in one slim batch, demuxed back into two
    /// indexed `Complete`s that each report the shared prompt in full.
    #[tokio::test]
    async fn generate_e2e_fans_out_n2_sglang() {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::Sglang,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let (client, engine) = (client.unwrap(), engine.unwrap());
        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let mut rids = Vec::new();
            for _ in 0..2 {
                let frames = input.recv_frames().await.unwrap();
                let request: WireRequest = decode_msgpack(frames[1].as_ref()).unwrap();
                assert_eq!(request.sampling_params.n, 1);
                rids.push(request.rid);
            }
            rids.sort();
            assert_eq!(rids, vec!["r1-0".to_string(), "r1-1".to_string()]);
            let done = BatchTokenIDSlimOutput {
                rids: vec!["r1-0".into(), "r1-1".into()],
                output_ids: vec![vec![10], vec![11]],
                finished_reasons: vec!["stop".into(), "stop".into()],
                finished_messages: vec![None, None],
                finished_matched: vec![None, None],
                prompt_tokens: vec![3, 3],
                completion_tokens: vec![1, 1],
                cached_tokens: vec![0, 0],
                output_token_logprobs_val: vec![vec![], vec![]],
                output_token_logprobs_idx: vec![vec![], vec![]],
                ..Default::default()
            };
            output
                .send_frames(vec![bytes::Bytes::from(encode_msgpack(&done).unwrap())])
                .await
                .unwrap();
        });
        let req = sglang_proto::GenerateRequest {
            request_id: "r1".to_string(),
            tokenized: Some(sglang_proto::TokenizedInput {
                input_ids: vec![1, 2, 3],
                original_text: String::new(),
            }),
            sampling_params: Some(sglang_proto::SamplingParams {
                n: 2,
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut stream = client.generate_sglang(req).await.expect("generate");
        let mut completes = Vec::new();
        while let Some(item) = stream.next().await {
            if let Some(vllm::generate_response::Response::Complete(complete)) =
                item.expect("stream item").response
            {
                completes.push(complete);
            }
        }
        completes.sort_by_key(|complete| complete.index);
        assert_eq!(completes.len(), 2, "one Complete per fanned-out sub");
        assert_eq!(
            (completes[0].index, &completes[0].output_ids),
            (0, &vec![10])
        );
        assert_eq!(
            (completes[1].index, &completes[1].output_ids),
            (1, &vec![11])
        );
        assert!(completes.iter().all(|complete| complete.prompt_tokens == 3));
        engine_task.await.unwrap();
    }

    /// An `abort` finish is the scheduler telling the frontend it could not
    /// serve the request; it surfaces as an error with the reason, and the
    /// scheduler's 400 as the caller's fault.
    #[tokio::test]
    async fn sglang_abort_finish_is_an_error_with_the_scheduler_status() {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::Sglang,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let (client, engine) = (client.unwrap(), engine.unwrap());
        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            for (rid, status) in [("r2", Some(400)), ("r3", None)] {
                let _ = input.recv_frames().await.unwrap();
                let rejection = BatchTokenIDSlimOutput {
                    rids: vec![rid.into()],
                    output_ids: vec![vec![]],
                    finished_reasons: vec!["abort".into()],
                    finished_messages: vec![Some("input_ids are required on this wire".into())],
                    finished_matched: vec![None],
                    prompt_tokens: vec![0],
                    completion_tokens: vec![0],
                    cached_tokens: vec![0],
                    output_token_logprobs_val: vec![vec![]],
                    output_token_logprobs_idx: vec![vec![]],
                    finished_status: status.into_iter().map(Some).collect(),
                    ..Default::default()
                };
                output
                    .send_frames(vec![bytes::Bytes::from(
                        encode_msgpack(&rejection).unwrap(),
                    )])
                    .await
                    .unwrap();
            }
        });
        for (rid, code) in [
            ("r2", tonic::Code::InvalidArgument),
            ("r3", tonic::Code::Internal),
        ] {
            let req = sglang_proto::GenerateRequest {
                request_id: rid.to_string(),
                tokenized: Some(sglang_proto::TokenizedInput {
                    input_ids: vec![1],
                    original_text: String::new(),
                }),
                ..Default::default()
            };
            let mut stream = client.generate_sglang(req).await.expect("generate");
            let error = stream.next().await.expect("an item").expect_err("an error");
            assert_eq!(error.code(), code, "{rid}");
            assert!(
                error.message().contains("input_ids are required"),
                "{error}"
            );
        }
        engine_task.await.unwrap();
    }
}
