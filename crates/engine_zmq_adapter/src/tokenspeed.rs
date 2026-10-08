//! TokenSpeed dialect: proto request → scheduler request translation and
//! scheduler output → vLLM-proto response mapping.

use std::collections::BTreeMap;

use engine_zmq_client::{
    codec::tensor::{checked_numel, WireTensor},
    connector::TokenSpeedStream,
    protocol::tokenspeed::{
        multimodal::{
            mm_pad_value, TokenSpeedWireMmInputs, TokenSpeedWireMmItem, TokenSpeedWireModality,
        },
        output::TokenSpeedOutput,
        request::TokenizedGenerateReqInput,
        sampling::SamplingParams as TokenSpeedSamplingParams,
    },
};
use futures::Stream;
use smg_grpc_client::{common_proto as common, tokenspeed_proto, vllm_proto as vllm};
use tracing::warn;

use crate::{
    fanout::fan_out_n,
    stream::{poll_mapped, MappedGenerateStream, StreamState},
};

/// Streaming generate output for one TokenSpeed sub-request, mapping each
/// `TokenSpeedOutput` to a vLLM-proto `GenerateResponse`, tagged with this
/// sub's choice `index`.
pub struct TokenSpeedGenerateStream {
    inner: TokenSpeedStream,
    state: StreamState,
    /// Choice index stamped on every chunk/complete (0 for n=1; the fan-out
    /// position for n>1) — the proto field the pipeline demuxes choices by.
    index: u32,
    /// Terminal `Complete` held back when the finish tick also carried new
    /// tokens: streaming frontends decode text/logprobs from chunks only, so
    /// the tick's delta goes out as a `Chunk` first.
    pending: Option<vllm::GenerateResponse>,
}

impl TokenSpeedGenerateStream {
    pub(crate) fn new(inner: TokenSpeedStream, index: u32) -> Self {
        Self {
            inner,
            state: StreamState::default(),
            index,
            pending: None,
        }
    }

    /// The choice index this stream stamps on its responses.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Whether the engine already finished this choice and its terminal
    /// `Complete` is parked behind the chunk just yielded (the finish tick
    /// carried new tokens). A frontend that matched a stop on that chunk must
    /// not synthesize its own `Complete`: the accumulated state has already
    /// been drained into the parked one.
    pub fn has_parked_complete(&self) -> bool {
        self.pending.is_some()
    }

    /// End this choice now on a frontend-matched string stop: the terminal
    /// `Complete` built from the accumulated state, exactly as if the engine
    /// had reported `stop` with that string. The engine-side request is
    /// aborted when the stream is dropped, so the caller drops it next.
    pub fn complete_with_matched_stop(&mut self, matched: String) -> vllm::GenerateResponse {
        self.complete_with_finish(
            "stop".to_string(),
            Some(vllm::generate_complete::MatchedStop::MatchedStopStr(
                matched,
            )),
        )
    }

    /// End this choice as aborted: the engine's own parked `Complete` when it
    /// already finished, else the terminal `Complete` the Python servicer
    /// yields after an `Abort` RPC (`finish_reason = "abort"`).
    pub fn complete_aborted(&mut self) -> vllm::GenerateResponse {
        if let Some(parked) = self.pending.take() {
            return parked;
        }
        self.complete_with_finish("abort".to_string(), None)
    }

    fn complete_with_finish(
        &mut self,
        finish_reason: String,
        matched: Option<vllm::generate_complete::MatchedStop>,
    ) -> vllm::GenerateResponse {
        let mut pending = None;
        // No new tokens on a frontend finish, so `emit_tick` yields the
        // `Complete` directly and parks nothing.
        self.state.emit_tick(
            self.index,
            Vec::new(),
            None,
            Some((finish_reason, matched)),
            &mut pending,
        )
    }
}

/// A vLLM-proto response of a TokenSpeed stream as the TokenSpeed proto the
/// `TokenSpeedScheduler` service answers with, under `request_id` (the
/// vLLM proto carries none): the two carry the same chunk and completion
/// fields (TokenSpeed's `OutputLogProbs` has no prompt side, and the ZMQ wire
/// reports no speculative counters).
pub fn to_tokenspeed_response(
    request_id: &str,
    response: vllm::GenerateResponse,
) -> tokenspeed_proto::GenerateResponse {
    use tokenspeed_proto::generate_response::Response as TsResponse;
    use vllm::generate_response::Response;
    let logprobs = |logprobs: Option<vllm::OutputLogProbs>| {
        logprobs.map(|lp| tokenspeed_proto::OutputLogProbs {
            token_logprobs: lp.token_logprobs,
            token_ids: lp.token_ids,
            top_logprobs: lp
                .top_logprobs
                .into_iter()
                .map(|top| tokenspeed_proto::TopLogProbs {
                    values: top.values,
                    token_ids: top.token_ids,
                })
                .collect(),
        })
    };
    let mapped = response.response.map(|inner| match inner {
        Response::Chunk(chunk) => TsResponse::Chunk(tokenspeed_proto::GenerateStreamChunk {
            token_ids: chunk.token_ids,
            prompt_tokens: chunk.prompt_tokens,
            completion_tokens: chunk.completion_tokens,
            cached_tokens: chunk.cached_tokens,
            output_logprobs: logprobs(chunk.output_logprobs),
            weight_version: None,
            index: chunk.index,
        }),
        Response::Complete(complete) => {
            use tokenspeed_proto::generate_complete::MatchedStop as TsMatchedStop;
            use vllm::generate_complete::MatchedStop;
            TsResponse::Complete(tokenspeed_proto::GenerateComplete {
                output_ids: complete.output_ids,
                finish_reason: complete.finish_reason,
                prompt_tokens: complete.prompt_tokens,
                completion_tokens: complete.completion_tokens,
                cached_tokens: complete.cached_tokens,
                output_logprobs: logprobs(complete.output_logprobs),
                matched_stop: complete.matched_stop.map(|matched| match matched {
                    MatchedStop::MatchedTokenId(id) => TsMatchedStop::MatchedTokenId(id),
                    MatchedStop::MatchedStopStr(text) => TsMatchedStop::MatchedStopStr(text),
                }),
                index: complete.index,
                ..Default::default()
            })
        }
    });
    tokenspeed_proto::GenerateResponse {
        request_id: request_id.to_string(),
        response: mapped,
    }
}

impl MappedGenerateStream for TokenSpeedGenerateStream {
    type Output = TokenSpeedOutput;
    type Inner = TokenSpeedStream;

    fn inner(&mut self) -> &mut Self::Inner {
        &mut self.inner
    }

    fn pending(&mut self) -> &mut Option<vllm::GenerateResponse> {
        &mut self.pending
    }

    fn map_output(
        &mut self,
        output: TokenSpeedOutput,
    ) -> Result<vllm::GenerateResponse, tonic::Status> {
        let state = &mut self.state;
        // TokenSpeed reports per-request token counts directly (cumulative for
        // completions), rather than vLLM's per-output prefill-stats deltas.
        if output.prompt_tokens > 0 {
            state.prompt_tokens = output.prompt_tokens;
        }
        if output.cached_tokens > 0 {
            state.cached_tokens = output.cached_tokens;
        }
        state.completion_tokens = output.completion_tokens;
        state.output_ids.extend(output.output_ids.iter().copied());

        // Sampled-token logprobs, if requested. The proto column is `float`, so
        // downcast the wire's `f64` values. Chunks carry this tick's increment;
        // the terminal `Complete` carries the cumulative set, so accumulate
        // into `state` and drain it on finish.
        let chunk_logprobs =
            (!output.output_logprobs_val.is_empty()).then(|| vllm::OutputLogProbs {
                token_logprobs: output
                    .output_logprobs_val
                    .iter()
                    .map(|&lp| lp as f32)
                    .collect(),
                token_ids: output.output_logprobs_idx.clone(),
                ..Default::default()
            });
        state
            .output_logprobs_val
            .extend(output.output_logprobs_val.iter().map(|&lp| lp as f32));
        state
            .output_logprobs_idx
            .extend(output.output_logprobs_idx.iter().copied());

        // An engine-side failure must surface as an error, not a normal
        // completion with empty output — mirroring the vLLM stream's guard.
        if output.finish_reason.as_deref() == Some("error") {
            return Err(tonic::Status::internal(
                "engine finished the request with an error (see engine logs)",
            ));
        }
        // No matched_stop on this wire: TokenSpeed reports the finish reason
        // only, and the router-side stop machinery owns string matching.
        let finish = output
            .finish_reason
            .map(|reason| (normalize_finish_reason(&reason).to_string(), None));
        Ok(state.emit_tick(
            self.index,
            output.output_ids,
            chunk_logprobs,
            finish,
            &mut self.pending,
        ))
    }
}

impl Stream for TokenSpeedGenerateStream {
    type Item = Result<vllm::GenerateResponse, tonic::Status>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        poll_mapped(self.get_mut(), cx)
    }
}

/// Split an `n > 1` TokenSpeed proto request into `n` single-sample
/// sub-requests, the TokenSpeed analogue of [`fan_out_requests`] (the wire has
/// no per-sample demux, so `generate` fans out here). An `n <= 1` request passes
/// through untouched. Seed handling matches [`fan_out_requests`]: omission
/// delegates independent seed assignment to the engine, while explicit seed
/// `s` derives deterministic per-sample seeds `s + i`.
pub(crate) fn fan_out_tokenspeed_requests(
    req: tokenspeed_proto::GenerateRequest,
) -> Vec<tokenspeed_proto::GenerateRequest> {
    let n = req.sampling_params.as_ref().map_or(1, |sp| sp.n.max(1));
    fan_out_n(req, n, |sub, i| {
        sub.request_id = format!("{}-{i}", sub.request_id);
        if let Some(sp) = sub.sampling_params.as_mut() {
            sp.n = 1;
            sp.sampling_seed = sp.sampling_seed.map(|seed| seed.wrapping_add(u64::from(i)));
        }
    })
}

/// Translate a TokenSpeed proto `GenerateRequest` into the wire
/// `TokenizedGenerateReqInput`. ZMQ mode requires pre-tokenized input (SMG
/// tokenizes upstream).
pub(crate) fn translate_request_tokenspeed(
    req: tokenspeed_proto::GenerateRequest,
) -> Result<TokenizedGenerateReqInput, String> {
    let mut input_ids = match req.tokenized {
        Some(tokenized) => tokenized.input_ids,
        None => {
            return Err("ZMQ mode requires pre-tokenized input; no input provided".to_string());
        }
    };
    let (input_ids_unpadded, multimodal_inputs) = match req.mm_inputs {
        Some(mm) => {
            let (unpadded, inputs) = translate_tokenspeed_multimodal(mm, &mut input_ids)?;
            (Some(unpadded), Some(inputs))
        }
        None => (None, None),
    };
    // Over the ZMQ wire TokenSpeed returns only the single sampled-token logprob
    // per token: no top-k candidates (`top_logprobs_num > 1`) and no prompt
    // logprobs (`token_ids_logprob`). Reject both rather than silently return
    // fewer than asked. A bare `logprobs: true` (count 0/1) is the plain
    // sampled-token logprob and is wired end-to-end via `return_logprob`.
    if req.top_logprobs_num > 1 {
        return Err("top_logprobs are not supported over the TokenSpeed ZMQ backend".to_string());
    }
    if !req.token_ids_logprob.is_empty() {
        return Err(
            "prompt logprobs are not supported over the TokenSpeed ZMQ backend".to_string(),
        );
    }
    Ok(TokenizedGenerateReqInput {
        rid: req.request_id,
        input_ids,
        sampling_params: req
            .sampling_params
            .map(translate_sampling_tokenspeed)
            .unwrap_or_else(|| {
                let mut params = TokenSpeedSamplingParams::default();
                params.normalize();
                params
            }),
        return_logprob: req.return_logprob,
        stream: req.stream,
        input_ids_unpadded,
        multimodal_inputs,
        // Every other field keeps its neutral default (text requests do not
        // even emit the fields after `stream`; the engine fills them from
        // defaults).
        ..TokenizedGenerateReqInput::default()
    })
}

/// Map the proto multimodal payload onto the TokenSpeed wire structs, doing
/// the work the engine's `InputProcessor` does on other transports but the
/// msgpack path bypasses: derive each item's pad value from its content hash
/// and substitute it into the placeholder ranges of `input_ids`. Returns the
/// original ids (for `input_ids_unpadded`, which detokenization reads) and
/// the wire payload.
pub(crate) fn translate_tokenspeed_multimodal(
    mm: tokenspeed_proto::MultimodalInputs,
    input_ids: &mut [u32],
) -> Result<(Vec<u32>, TokenSpeedWireMmInputs), String> {
    if mm.items.is_empty() {
        return Err("multimodal payload carried no items".to_string());
    }
    let unpadded = input_ids.to_vec();
    let mut im_token_id = None;
    let mut video_token_id = None;
    let mut mm_items = Vec::with_capacity(mm.items.len());
    for item in mm.items {
        // The proto and engine modality enums disagree (proto: AUDIO=2,
        // VIDEO=3; engine: VIDEO=2, AUDIO=3) — translate, never pass through.
        // Unknown values are rejected, mirroring the gRPC servicer's
        // `_modality_from_proto`.
        let modality = match item.modality() {
            common::Modality::Image => TokenSpeedWireModality::Image,
            common::Modality::Video => TokenSpeedWireModality::Video,
            common::Modality::Audio => TokenSpeedWireModality::Audio,
            common::Modality::Unspecified => {
                return Err("multimodal item carried an unspecified modality".to_string());
            }
        };
        if item.content_hash.is_empty() {
            return Err(
                "multimodal item carried no content hash; the engine pad value derives from it"
                    .to_string(),
            );
        }
        // u64 little-endian fold of the leading hash bytes — the same fold the
        // gRPC servicer applies (`int.from_bytes(content_hash[:8], "little")`),
        // so an item hashes identically on both transports.
        let mut hash_bytes = [0u8; 8];
        for (dst, src) in hash_bytes.iter_mut().zip(item.content_hash.iter()) {
            *dst = *src;
        }
        let hash = u64::from_le_bytes(hash_bytes);
        let pad_value = mm_pad_value(modality, hash);

        if item.placeholders.is_empty() {
            return Err("multimodal item carried no placeholders".to_string());
        }
        let mut offsets = Vec::with_capacity(item.placeholders.len());
        for placeholder in &item.placeholders {
            if placeholder.length == 0 {
                return Err("multimodal placeholder length must be > 0".to_string());
            }
            let start = placeholder.offset as usize;
            let end = start + placeholder.length as usize - 1;
            if end >= input_ids.len() {
                return Err(format!(
                    "multimodal placeholder [{start}, {end}] exceeds the {} prompt tokens",
                    input_ids.len()
                ));
            }
            for id in &mut input_ids[start..=end] {
                *id = pad_value;
            }
            offsets.push((start as u64, end as u64));
        }
        match modality {
            TokenSpeedWireModality::Image => {
                im_token_id = im_token_id.or(item.placeholder_token_id);
            }
            TokenSpeedWireModality::Video => {
                video_token_id = video_token_id.or(item.placeholder_token_id);
            }
            TokenSpeedWireModality::Audio => {}
        }

        let feature = wire_tensor_tokenspeed(
            item.encoder_input
                .ok_or("multimodal item carried no encoder_input")?,
        )?;
        let mut model_specific_data = BTreeMap::new();
        for (name, tensor) in item.model_specific_tensors {
            model_specific_data.insert(name, wire_tensor_tokenspeed(tensor)?);
        }
        // Nothing on the msgpack path computes `mrope_positions` (the engine's
        // InputProcessor is bypassed), and shipping nil silently degrades
        // image grounding to 1-D positions. For the known MRoPE families the
        // grid tensors are the tell: fail loudly rather than succeed wrong.
        for key in ["image_grid_thw", "video_grid_thw"] {
            if model_specific_data.contains_key(key) {
                return Err(format!(
                    "multimodal item carries {key:?}: MRoPE position tensors are not \
                     derivable over the TokenSpeed ZMQ wire yet; use the gRPC transport \
                     for this model"
                ));
            }
        }
        mm_items.push(TokenSpeedWireMmItem {
            modality,
            hash,
            pad_value,
            offsets,
            feature,
            model_specific_data,
        });
    }
    // Surface the position-tensor gap in the gateway's own logs (once per
    // process): models outside the gated MRoPE families still receive no
    // mrope_positions on this wire.
    static MROPE_POSITIONS_NOTE: std::sync::Once = std::sync::Once::new();
    MROPE_POSITIONS_NOTE.call_once(|| {
        warn!(
            "TokenSpeed ZMQ multimodal requests carry no mrope_positions; models that \
             require them fall back to 1-D positions engine-side"
        );
    });
    Ok((
        unpadded,
        TokenSpeedWireMmInputs {
            mm_items,
            im_token_id,
            video_token_id,
        },
    ))
}

/// Convert an inline proto tensor to the wire `(dtype, shape, ext)` tuple.
/// SHM/remote payloads never reach this translate: ZMQ assembly runs with SHM
/// disabled and RDMA staging off.
pub(crate) fn wire_tensor_tokenspeed(
    tensor: tokenspeed_proto::TensorData,
) -> Result<WireTensor, String> {
    let bytes = match tensor.payload {
        Some(tokenspeed_proto::tensor_data::Payload::Inline(bytes)) => bytes,
        Some(_) => {
            return Err(
                "ZMQ multimodal requires inline tensor payloads; got a SHM/remote handle"
                    .to_string(),
            );
        }
        None => return Err("multimodal tensor carried no payload".to_string()),
    };
    let shape: Vec<usize> = tensor.shape.iter().map(|&d| d as usize).collect();
    // The engine views raw bytes as the named dtype, and a malformed frame is
    // silently dropped engine-side (no terminal frame back) — so a byte-count
    // mismatch must fail here, where the client still gets an error.
    if let Some(width) = tensor_dtype_width(&tensor.dtype) {
        let numel =
            checked_numel(&shape).ok_or_else(|| "multimodal tensor shape overflows".to_string())?;
        if numel * width != bytes.len() {
            return Err(format!(
                "multimodal tensor byte length {} does not match dtype {:?} shape {:?}",
                bytes.len(),
                tensor.dtype,
                tensor.shape
            ));
        }
    }
    Ok(WireTensor::from_raw_bytes(
        tensor.dtype,
        shape,
        bytes.into(),
    ))
}

/// Byte width of the wire dtypes the gateway emits; `None` for dtypes we do
/// not recognize (passed through unvalidated rather than rejected).
pub(crate) fn tensor_dtype_width(dtype: &str) -> Option<usize> {
    match dtype {
        "float64" | "int64" | "uint64" => Some(8),
        "float32" | "int32" | "uint32" => Some(4),
        "bfloat16" | "float16" | "int16" | "uint16" => Some(2),
        "uint8" | "int8" | "bool" => Some(1),
        _ => None,
    }
}

/// Map TokenSpeed proto sampling params onto the wire `SamplingParams`, in the
/// normalized form: the engine skips its decode-time re-derivation once
/// `is_normalized` is set, so [`TokenSpeedSamplingParams::normalize`] resolves
/// the derived fields (top_k sentinel, greedy collapse) before encoding.
///
/// String `stop` sequences are not forwarded — the token-only engine cannot
/// match them; the router-side stop decoder trims them from the text instead.
pub(crate) fn translate_sampling_tokenspeed(
    sp: tokenspeed_proto::SamplingParams,
) -> TokenSpeedSamplingParams {
    let mut params = TokenSpeedSamplingParams {
        max_new_tokens: sp.max_new_tokens,
        stop_token_ids: (!sp.stop_token_ids.is_empty()).then_some(sp.stop_token_ids),
        temperature: f64::from(sp.temperature.unwrap_or(1.0)),
        top_p: f64::from(sp.top_p.unwrap_or(1.0)),
        // The proto keeps the API convention `-1` = "all tokens" (and unset);
        // `normalize` resolves it to the engine's disabled sentinel.
        top_k: sp.top_k.unwrap_or(-1),
        min_p: f64::from(sp.min_p.unwrap_or(0.0)),
        frequency_penalty: f64::from(sp.frequency_penalty.unwrap_or(0.0)),
        presence_penalty: f64::from(sp.presence_penalty.unwrap_or(0.0)),
        repetition_penalty: f64::from(sp.repetition_penalty.unwrap_or(1.0)),
        min_new_tokens: sp.min_new_tokens,
        ignore_eos: sp.ignore_eos,
        skip_special_tokens: sp.skip_special_tokens,
        spaces_between_special_tokens: sp.spaces_between_special_tokens,
        no_stop_trim: sp.no_stop_trim,
        seed: sp.sampling_seed,
        // Proto `0` means unspecified; TokenSpeed expects at least one sample.
        // n>1 is fanned out before translation, so this is always 1 on the wire.
        n: sp.n.max(1),
        ..TokenSpeedSamplingParams::default()
    };
    apply_tokenspeed_constraint(&mut params, sp.constraint);
    params.normalize();
    params
}

/// Map the proto structured-output `constraint` oneof onto the wire's dedicated
/// fields. The oneof is single-valued, so at most one field is set; the rest
/// stay `None`.
pub(crate) fn apply_tokenspeed_constraint(
    params: &mut TokenSpeedSamplingParams,
    constraint: Option<tokenspeed_proto::sampling_params::Constraint>,
) {
    use tokenspeed_proto::sampling_params::Constraint;
    match constraint {
        Some(Constraint::JsonSchema(schema)) => params.json_schema = Some(schema),
        Some(Constraint::Regex(regex)) => params.regex = Some(regex),
        Some(Constraint::EbnfGrammar(grammar)) => params.ebnf = Some(grammar),
        Some(Constraint::StructuralTag(tag)) => params.structural_tag = Some(tag),
        None => {}
    }
}

/// Normalize a TokenSpeed wire finish-reason string into the canonical set the
/// gateway's response layer exact-matches (`stop`, `length`, `abort`, `error`) —
/// the same set the vLLM path emits via [`finish_reason_str`]. TokenSpeed emits
/// `stop`/`length`/`abort`; an unknown value falls back to `stop` with a warning
/// so a non-canonical string never mis-renders downstream.
pub(crate) fn normalize_finish_reason(reason: &str) -> &'static str {
    match reason {
        "stop" => "stop",
        "length" => "length",
        "abort" => "abort",
        "error" => "error",
        other => {
            tracing::warn!(
                finish_reason = other,
                "unknown TokenSpeed finish_reason; defaulting to \"stop\""
            );
            "stop"
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use engine_zmq_client::{
        mock_engine::{connect_to_frontend, default_ready_response},
        EngineId,
    };
    use openai_protocol::worker::RuntimeType;

    use super::*;
    use crate::{client::ZmqEngineClient, eos::EosTokenIds};

    /// End-to-end over ipc:// for a TokenSpeed backend: the adapter frames a
    /// tagged `TokenizedGenerateReqInput`, and maps `BatchTokenIDOutSlim`
    /// batches back to vLLM-proto responses. The mock engine speaks the shared
    /// transport with raw frames (it decodes/encodes the TokenSpeed structs
    /// directly).
    #[tokio::test]
    async fn generate_e2e_translates_and_streams_tokenspeed() {
        use engine_zmq_client::{
            codec::{decode_msgpack, encode_msgpack},
            protocol::tokenspeed::{
                output::BatchTokenIDOutSlim,
                request::{TokenSpeedRequestType, TokenizedGenerateReqInput},
            },
        };

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
                RuntimeType::TokenSpeed,
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
                TokenSpeedRequestType::from_frame(frames[0].as_ref()),
                Some(TokenSpeedRequestType::Add)
            );
            let request: TokenizedGenerateReqInput = decode_msgpack(frames[1].as_ref()).unwrap();
            assert_eq!(request.rid, "r1");
            assert_eq!(request.input_ids, vec![1, 2, 3]);
            assert_eq!(request.sampling_params.max_new_tokens, Some(2));
            // The adapter always emits the normalized sampling form.
            assert!(request.sampling_params.is_normalized);
            // A plain sampled-token logprob request (logprobs=1) sets the flag.
            assert!(request.return_logprob);

            let chunk = BatchTokenIDOutSlim {
                rids: vec!["r1".into()],
                output_ids: vec![vec![10]],
                finished_reasons: vec![String::new()],
                prompt_tokens: vec![3],
                completion_tokens: vec![1],
                cached_tokens: vec![0],
                output_token_logprobs_val: vec![vec![-0.5]],
                output_token_logprobs_idx: vec![vec![10]],
                ..Default::default()
            };
            let done = BatchTokenIDOutSlim {
                rids: vec!["r1".into()],
                output_ids: vec![vec![11]],
                finished_reasons: vec!["length".into()],
                prompt_tokens: vec![3],
                completion_tokens: vec![2],
                cached_tokens: vec![0],
                output_token_logprobs_val: vec![vec![-1.25]],
                output_token_logprobs_idx: vec![vec![11]],
                ..Default::default()
            };
            output
                .send_frames(vec![bytes::Bytes::from(encode_msgpack(&chunk).unwrap())])
                .await
                .unwrap();
            output
                .send_frames(vec![bytes::Bytes::from(encode_msgpack(&done).unwrap())])
                .await
                .unwrap();
        });

        let req = tokenspeed_proto::GenerateRequest {
            request_id: "r1".to_string(),
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                input_ids: vec![1, 2, 3],
                original_text: String::new(),
            }),
            sampling_params: Some(tokenspeed_proto::SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            // Plain sampled-token logprob; must be wired through.
            return_logprob: true,
            stream: true,
            ..Default::default()
        };
        let mut stream = client.generate_tokenspeed(req).await.expect("generate");

        let first = stream.next().await.expect("chunk item").expect("chunk ok");
        match first.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                assert_eq!(chunk.token_ids, vec![10]);
                assert_eq!(chunk.prompt_tokens, 3);
                let logprobs = chunk.output_logprobs.expect("chunk logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-0.5]);
                assert_eq!(logprobs.token_ids, vec![10]);
            }
            other => panic!("expected chunk, got {other:?}"),
        }
        // The finish tick carried a new token, so its delta is emitted as a
        // chunk before the (cumulative) terminal complete.
        let second = stream.next().await.expect("chunk item").expect("chunk ok");
        match second.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                assert_eq!(chunk.token_ids, vec![11]);
                let logprobs = chunk.output_logprobs.expect("chunk logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-1.25]);
                assert_eq!(logprobs.token_ids, vec![11]);
            }
            other => panic!("expected chunk, got {other:?}"),
        }
        let third = stream
            .next()
            .await
            .expect("complete item")
            .expect("complete ok");
        match third.response {
            Some(vllm::generate_response::Response::Complete(complete)) => {
                assert_eq!(complete.output_ids, vec![10, 11]);
                assert_eq!(complete.finish_reason, "length");
                assert_eq!(complete.completion_tokens, 2);
                // Chunks carry the tick's incremental logprobs; the terminal
                // `Complete` carries the cumulative set, parallel to `output_ids`
                // (the non-streaming renderer reads only the `Complete`).
                let logprobs = complete.output_logprobs.expect("complete logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-0.5, -1.25]);
                assert_eq!(logprobs.token_ids, vec![10, 11]);
            }
            other => panic!("expected complete, got {other:?}"),
        }
        assert!(stream.next().await.is_none());

        engine_task.await.unwrap();
    }

    #[test]
    fn tokenspeed_sampling_maps_top_k_sentinel_and_floors_n() {
        use engine_zmq_client::protocol::tokenspeed::sampling::TOP_K_DISABLED;

        // Unset top_k rides the API convention `-1` ("all tokens") and normalizes
        // to the engine's disabled sentinel; n=0 floors to 1; max_new_tokens
        // forwards.
        let mapped = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            top_k: None,
            sampling_seed: Some(1_234),
            n: 0,
            max_new_tokens: Some(8),
            ..Default::default()
        });
        assert_eq!(mapped.top_k, TOP_K_DISABLED);
        assert_eq!(mapped.seed, Some(1_234));
        assert_eq!(mapped.n, 1);
        assert_eq!(mapped.max_new_tokens, Some(8));
        // The wire form is always normalized (the engine skips re-derivation).
        assert!(mapped.is_normalized);

        // An explicit top_k passes through unchanged.
        let mapped = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            top_k: Some(40),
            ..Default::default()
        });
        assert_eq!(mapped.top_k, 40);

        // A near-zero temperature collapses to greedy on the wire.
        let mapped = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            temperature: Some(0.0),
            ..Default::default()
        });
        assert_eq!(mapped.temperature, 1.0);
        assert_eq!(mapped.top_k, 1);

        // Empty stop_token_ids ride as None (the normalized encoding).
        let mapped = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams::default());
        assert_eq!(mapped.stop_token_ids, None);
    }

    fn ts_tokenized_req(
        sampling: tokenspeed_proto::SamplingParams,
    ) -> tokenspeed_proto::GenerateRequest {
        tokenspeed_proto::GenerateRequest {
            request_id: "r1".to_string(),
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                input_ids: vec![1, 2, 3],
                original_text: String::new(),
            }),
            sampling_params: Some(sampling),
            stream: true,
            ..Default::default()
        }
    }

    #[test]
    fn tokenspeed_return_logprob_flag_passes_through() {
        // The request-level `return_logprob` drives the plain sampled-token
        // logprob (count 0/1 in `top_logprobs_num` is the same case).
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.return_logprob = true;
        let wire = translate_request_tokenspeed(req).expect("return_logprob accepted");
        assert!(wire.return_logprob);

        // Unset -> the flag stays false.
        let wire = translate_request_tokenspeed(ts_tokenized_req(
            tokenspeed_proto::SamplingParams::default(),
        ))
        .expect("no logprobs accepted");
        assert!(!wire.return_logprob);
    }

    #[test]
    fn tokenspeed_rejects_top_logprobs_and_prompt_logprobs() {
        // Top-k logprobs (count > 1) cannot be honored over the wire.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.top_logprobs_num = 5;
        assert!(translate_request_tokenspeed(req).is_err());

        // Prompt (input) logprobs cannot be produced.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.token_ids_logprob = vec![1, 2];
        assert!(translate_request_tokenspeed(req).is_err());

        // A bare count of 0/1 is the plain sampled-token case: accepted.
        for count in [0, 1] {
            let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
            req.top_logprobs_num = count;
            assert!(translate_request_tokenspeed(req).is_ok());
        }
    }

    #[test]
    fn tokenspeed_maps_structured_output_constraints() {
        // The `constraint` oneof maps 1:1 onto the wire's dedicated fields; the
        // oneof is single-valued, so the other three stay unset.
        use tokenspeed_proto::sampling_params::Constraint;

        let json = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            constraint: Some(Constraint::JsonSchema("{\"type\":\"object\"}".into())),
            ..Default::default()
        });
        assert_eq!(json.json_schema.as_deref(), Some("{\"type\":\"object\"}"));
        assert_eq!(json.regex, None);
        assert_eq!(json.ebnf, None);
        assert_eq!(json.structural_tag, None);

        let regex = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            constraint: Some(Constraint::Regex("[0-9]+".into())),
            ..Default::default()
        });
        assert_eq!(regex.regex.as_deref(), Some("[0-9]+"));
        assert_eq!(regex.json_schema, None);

        let ebnf = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            constraint: Some(Constraint::EbnfGrammar("root ::= \"a\"".into())),
            ..Default::default()
        });
        assert_eq!(ebnf.ebnf.as_deref(), Some("root ::= \"a\""));

        let tag = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            constraint: Some(Constraint::StructuralTag("<tag>".into())),
            ..Default::default()
        });
        assert_eq!(tag.structural_tag.as_deref(), Some("<tag>"));

        // No constraint leaves all four structured-output fields unset.
        let none = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams::default());
        assert_eq!(none.json_schema, None);
        assert_eq!(none.regex, None);
        assert_eq!(none.ebnf, None);
        assert_eq!(none.structural_tag, None);
    }

    #[test]
    fn tokenspeed_forwards_stop_token_ids_and_drops_stop_strings() {
        // String stops are resolved upstream; any that reach here are dropped
        // (the token-only engine cannot match them) while stop token ids ride
        // through and the router-side decoder trims residual text.
        let req =
            translate_request_tokenspeed(ts_tokenized_req(tokenspeed_proto::SamplingParams {
                stop: vec!["</s>".to_string()],
                stop_token_ids: vec![13],
                ..Default::default()
            }))
            .expect("residual stop strings must not be rejected");
        assert_eq!(req.sampling_params.stop_token_ids, Some(vec![13]));
        assert_eq!(req.sampling_params.stop, None);
    }

    fn ts_mm_item(hash: &[u8], offset: u32, length: u32) -> tokenspeed_proto::MultimodalItem {
        tokenspeed_proto::MultimodalItem {
            modality: common::Modality::Image as i32,
            content_hash: hash.to_vec(),
            encoder_input: Some(tokenspeed_proto::TensorData {
                shape: vec![2, 4],
                dtype: "bfloat16".to_string(),
                payload: Some(tokenspeed_proto::tensor_data::Payload::Inline(vec![
                    0u8;
                    16
                ])),
            }),
            model_specific_tensors: [(
                "vit_grid".to_string(),
                tokenspeed_proto::TensorData {
                    shape: vec![1, 3],
                    dtype: "uint32".to_string(),
                    payload: Some(tokenspeed_proto::tensor_data::Payload::Inline(vec![
                        0u8;
                        12
                    ])),
                },
            )]
            .into(),
            placeholders: vec![tokenspeed_proto::PlaceholderRange { offset, length }],
            placeholder_token_id: Some(9),
        }
    }

    #[test]
    fn tokenspeed_translates_multimodal_inputs() {
        // The translate does the engine InputProcessor's job (bypassed on the
        // msgpack path): pad-value substitution into input_ids, unpadded ids
        // preserved, offsets converted to inclusive [start, end] pairs.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.tokenized.as_mut().unwrap().input_ids = vec![10, 20, 30, 40, 50];
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs {
            items: vec![ts_mm_item(&0xDEAD_BEEFu64.to_le_bytes(), 1, 3)],
        });
        let wire = translate_request_tokenspeed(req).expect("translated");

        let expected_pad = mm_pad_value(TokenSpeedWireModality::Image, 0xDEAD_BEEF);
        assert_eq!(
            wire.input_ids,
            vec![10, expected_pad, expected_pad, expected_pad, 50]
        );
        assert_eq!(wire.input_ids_unpadded, Some(vec![10, 20, 30, 40, 50]));

        let mm = wire.multimodal_inputs.expect("mm payload");
        assert_eq!(mm.im_token_id, Some(9));
        assert_eq!(mm.video_token_id, None);
        assert_eq!(mm.mm_items.len(), 1);
        let item = &mm.mm_items[0];
        assert_eq!(item.modality, TokenSpeedWireModality::Image);
        assert_eq!(item.hash, 0xDEAD_BEEF);
        assert_eq!(item.pad_value, expected_pad);
        assert_eq!(item.offsets, vec![(1, 3)]);
        assert_eq!(item.model_specific_data["vit_grid"].dtype, "uint32");
    }

    #[test]
    fn tokenspeed_multimodal_rejects_bad_items() {
        // Out-of-bounds placeholder: prompt has 3 tokens, range needs 4.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs {
            items: vec![ts_mm_item(b"12345678", 1, 3)],
        });
        assert!(translate_request_tokenspeed(req)
            .unwrap_err()
            .contains("exceeds"));

        // Missing content hash: pad value cannot be derived.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs {
            items: vec![ts_mm_item(b"", 0, 1)],
        });
        assert!(translate_request_tokenspeed(req)
            .unwrap_err()
            .contains("content hash"));

        // Byte-length mismatch would be silently dropped engine-side; the
        // translate must catch it while the client can still see an error.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        let mut item = ts_mm_item(b"12345678", 0, 1);
        item.encoder_input.as_mut().unwrap().shape = vec![3, 4];
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs { items: vec![item] });
        assert!(translate_request_tokenspeed(req)
            .unwrap_err()
            .contains("byte length"));

        // Unspecified modality is rejected, mirroring the gRPC servicer.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        let mut item = ts_mm_item(b"12345678", 0, 1);
        item.modality = common::Modality::Unspecified as i32;
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs { items: vec![item] });
        assert!(translate_request_tokenspeed(req)
            .unwrap_err()
            .contains("modality"));

        // Known MRoPE families fail loudly: nothing derives mrope_positions
        // on this wire, and 1-D fallback would silently degrade grounding.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        let mut item = ts_mm_item(b"12345678", 0, 1);
        item.model_specific_tensors.insert(
            "image_grid_thw".to_string(),
            tokenspeed_proto::TensorData {
                shape: vec![1, 3],
                dtype: "uint32".to_string(),
                payload: Some(tokenspeed_proto::tensor_data::Payload::Inline(vec![
                    0u8;
                    12
                ])),
            },
        );
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs { items: vec![item] });
        assert!(translate_request_tokenspeed(req)
            .unwrap_err()
            .contains("MRoPE"));
    }

    #[test]
    fn tokenspeed_fan_out_derives_explicit_sampling_seeds() {
        let subs =
            fan_out_tokenspeed_requests(ts_tokenized_req(tokenspeed_proto::SamplingParams {
                n: 3,
                sampling_seed: Some(7),
                ..Default::default()
            }));

        assert_eq!(subs.len(), 3);
        for (i, sub) in (0_u64..).zip(&subs) {
            let sampling = sub.sampling_params.as_ref().expect("sampling params");
            assert_eq!(sub.request_id, format!("r1-{i}"));
            assert_eq!(sampling.n, 1);
            assert_eq!(sampling.sampling_seed, Some(7 + i));
        }

        let subs =
            fan_out_tokenspeed_requests(ts_tokenized_req(tokenspeed_proto::SamplingParams {
                n: 2,
                ..Default::default()
            }));
        assert!(subs.iter().all(|sub| sub
            .sampling_params
            .as_ref()
            .expect("sampling params")
            .sampling_seed
            .is_none()));
    }

    /// n=2 over TokenSpeed: two engine-side requests with distinct sub-rids
    /// (delivered in one wire batch), two indexed `Complete`s, and the shared
    /// prompt reported in full on each (the pipeline de-duplicates via max, so
    /// prompt tokens are not counted n times).
    #[tokio::test]
    async fn generate_e2e_fans_out_n2_tokenspeed() {
        use engine_zmq_client::{
            codec::{decode_msgpack, encode_msgpack},
            protocol::tokenspeed::{
                output::BatchTokenIDOutSlim,
                request::{TokenSpeedRequestType, TokenizedGenerateReqInput},
            },
        };

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
                RuntimeType::TokenSpeed,
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
            let mut rids = Vec::new();
            for _ in 0..2 {
                let frames = input.recv_frames().await.unwrap();
                assert_eq!(
                    TokenSpeedRequestType::from_frame(frames[0].as_ref()),
                    Some(TokenSpeedRequestType::Add)
                );
                let request: TokenizedGenerateReqInput =
                    decode_msgpack(frames[1].as_ref()).unwrap();
                assert_eq!(request.sampling_params.n, 1);
                // TokenSpeed has no seed on the wire; the engine derives one from
                // the (unique) rid so all TP/DP ranks agree.
                assert_eq!(request.sampling_params.seed, None);
                rids.push(request.rid.clone());
            }
            assert_eq!(
                rids,
                vec!["r1-0".to_string(), "r1-1".to_string()],
                "sub-rids must be unique per sub"
            );
            // Both subs finish in one wire batch (the batch demux fans them
            // back out to their sub-streams).
            let done = BatchTokenIDOutSlim {
                rids: vec!["r1-0".into(), "r1-1".into()],
                output_ids: vec![vec![10], vec![11]],
                finished_reasons: vec!["stop".into(), "stop".into()],
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

        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams {
            n: 2,
            ..Default::default()
        });
        req.request_id = "r1".to_string();
        let mut stream = client.generate_tokenspeed(req).await.expect("generate");

        let mut completes = Vec::new();
        while let Some(item) = stream.next().await {
            match item.expect("stream item").response {
                Some(vllm::generate_response::Response::Complete(complete)) => {
                    completes.push(complete);
                }
                Some(vllm::generate_response::Response::Chunk(_)) | None => {}
            }
        }
        completes.sort_by_key(|complete| complete.index);
        assert_eq!(completes.len(), 2, "one Complete per fanned-out sub");
        assert_eq!(completes[0].index, 0);
        assert_eq!(completes[0].output_ids, vec![10]);
        assert_eq!(completes[1].index, 1);
        assert_eq!(completes[1].output_ids, vec![11]);
        // Each sub reports the full shared prompt; the pipeline maxes, so
        // usage is not double-counted.
        assert!(completes.iter().all(|complete| complete.prompt_tokens == 3));

        engine_task.await.unwrap();
    }

    #[test]
    fn tokenspeed_finish_reason_normalizes() {
        assert_eq!(normalize_finish_reason("stop"), "stop");
        assert_eq!(normalize_finish_reason("length"), "length");
        assert_eq!(normalize_finish_reason("abort"), "abort");
        assert_eq!(normalize_finish_reason("error"), "error");
        // An unknown wire value falls back to "stop".
        assert_eq!(normalize_finish_reason("garbage"), "stop");
    }
}
