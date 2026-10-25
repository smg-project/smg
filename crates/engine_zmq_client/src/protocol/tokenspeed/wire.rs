//! The engine's published schema (`tokenspeed-wire`, the `ts_wire` crate) as
//! this wire's framing for a scheduler of the engine's Rust control plane
//! ([`WireRevision::RustControlPlane`](super::WireRevision::RustControlPlane)).
//!
//! The request this crate's callers build ([`TokenizedGenerateReqInput`]) is
//! converted to the crate's struct and framed by the crate's own codec: the
//! full request array (the engine fills nothing from defaults), tensors of
//! 256 bytes and more as auxiliary frames, as the engine's Python frontend
//! writes them. The engine's outputs are decoded by the crate (the 22-column
//! slim batch, a tagged control reply, the dead sentinel) into the
//! per-request shape the connector routes ([`TokenSpeedOutput`]). Nothing
//! here reads a frame to decide the framing: the revision is the
//! connection's, read by the launcher before the first frame.
//!
//! What this revision carries that the other does not (the finish details
//! map, the ranked and prompt logprob columns, the tagged control
//! operations) is decoded by the crate and left unread here: each is its own
//! change on this module, behind the same revision.

use bytes::Bytes;
use ts_wire::{codec::StrOrList, msgpack::Integer, Dict, Value};

use crate::{
    codec::{
        tensor::{WireArrayData, WireTensor},
        OpaqueValue,
    },
    error::{Error, Result},
    protocol::{
        tokenspeed::{
            multimodal::{TokenSpeedWireMmInputs, TokenSpeedWireMmItem, TokenSpeedWireModality},
            output::TokenSpeedOutput,
            request::TokenizedGenerateReqInput,
            sampling::SamplingParams,
        },
        EngineBatch, EngineLoad,
    },
};

/// Encode an add-request as the engine's crate frames it: the payload (the
/// msgpack buffer, frame 0 of the message) and the auxiliary frames its
/// tensors ride in, in the order the crate lifted them.
pub(super) fn encode_add(request: &TokenizedGenerateReqInput) -> Result<(Vec<u8>, Vec<Bytes>)> {
    let request = to_wire_request(request)?;
    let mut frames = ts_wire::encode_frames(&request).into_iter();
    // The crate always writes the buffer first; a message has at least that.
    let payload = frames.next().unwrap_or_default();
    Ok((Vec::from(payload), frames.collect()))
}

/// Decode one message of the engine's output socket (frame 0 and the
/// auxiliary frames) as the engine's crate reads it: a slim batch becomes
/// the connector's outputs and load; a control reply is dropped (no control
/// operation is sent on this revision yet, so none is waited for); the dead
/// sentinel is the engine's end.
pub(super) fn decode_batch(frames: &[Bytes]) -> Result<EngineBatch<TokenSpeedOutput>> {
    match ts_wire::EngineOutput::from_frames(frames)
        .map_err(|error| decode_error("EngineOutput", error))?
    {
        ts_wire::EngineOutput::Batch(batch) => from_wire_batch(&batch),
        ts_wire::EngineOutput::Control(reply) => {
            tracing::warn!(
                reply = reply.tag(),
                "ignoring a control reply on the scheduler's output socket: no control \
                 operation is sent on this wire revision"
            );
            Ok(EngineBatch::default())
        }
        ts_wire::EngineOutput::Dead => Err(Error::EngineCoreDead),
    }
}

/// This crate's request as the engine's struct. The connector's routing
/// instruction (`data_parallel_rank`) stays with the connector: the request
/// reaches the rank the connector chose on that rank's own socket, as on the
/// other revision, and the engine's own pin field is left unset.
pub(super) fn to_wire_request(
    request: &TokenizedGenerateReqInput,
) -> Result<ts_wire::TokenizedGenerateReqInput> {
    Ok(ts_wire::TokenizedGenerateReqInput {
        rid: Some(StrOrList::One(request.rid.clone())),
        http_worker_ipc: request.http_worker_ipc.clone(),
        input_text: request.input_text.clone(),
        input_ids: Some(wire_token_ids(&request.input_ids, "input_ids")?),
        sampling_params: to_wire_sampling(&request.sampling_params)?,
        return_logprob: request.return_logprob,
        logprob_start_len: i64::from(request.logprob_start_len),
        top_logprobs_num: i64::from(request.top_logprobs_num),
        token_ids_logprob: request
            .token_ids_logprob
            .as_deref()
            .map(|ids| wire_token_ids(ids, "token_ids_logprob"))
            .transpose()?,
        stream: request.stream,
        input_embeds: None,
        session_params: None,
        custom_logit_processor: None,
        return_hidden_states: false,
        created_time: 0.0,
        bootstrap_host: request.bootstrap_host.clone(),
        bootstrap_port: request.bootstrap_port.map(i64::from),
        // The room is the gateway's own number for the disaggregated path
        // (drawn from 0..i64::MAX), never the request's: a negative one is
        // the gateway's bug and stays an encode failure, not an invalid
        // argument.
        bootstrap_room: request
            .bootstrap_room
            .map(|room| {
                u64::try_from(room).map_err(|_| {
                    encode_error(
                        "bootstrap_room",
                        format!("the room {room} is negative; the engine's room is unsigned"),
                    )
                })
            })
            .transpose()?,
        input_multi_ids: None,
        input_extra_infos: None,
        input_ids_unpadded: request
            .input_ids_unpadded
            .as_deref()
            .map(|ids| wire_token_ids(ids, "input_ids_unpadded"))
            .transpose()?,
        multimodal_inputs: request
            .multimodal_inputs
            .as_ref()
            .map(to_wire_multimodal)
            .transpose()?,
        validation_error: None,
        data_parallel_rank: None,
    })
}

fn to_wire_sampling(params: &SamplingParams) -> Result<ts_wire::SamplingParams> {
    Ok(ts_wire::SamplingParams {
        max_new_tokens: params.max_new_tokens.map(i64::from),
        stop: params.stop.clone().map(StrOrList::One),
        stop_token_ids: params
            .stop_token_ids
            .as_ref()
            .map(|ids| ids.iter().map(|&id| i64::from(id)).collect()),
        temperature: params.temperature,
        top_p: params.top_p,
        top_k: i64::from(params.top_k),
        min_p: params.min_p,
        frequency_penalty: params.frequency_penalty,
        presence_penalty: params.presence_penalty,
        repetition_penalty: params.repetition_penalty,
        min_new_tokens: i64::from(params.min_new_tokens),
        json_schema: params.json_schema.clone(),
        regex: params.regex.clone(),
        ebnf: params.ebnf.clone(),
        structural_tag: params.structural_tag.clone(),
        ignore_eos: params.ignore_eos,
        skip_special_tokens: params.skip_special_tokens,
        spaces_between_special_tokens: params.spaces_between_special_tokens,
        no_stop_trim: params.no_stop_trim,
        thinking_budget: params
            .thinking_budget
            .map(|budget| wire_i64(budget, "thinking_budget"))
            .transpose()?,
        custom_params: params
            .custom_params
            .as_ref()
            .map(to_wire_custom_params)
            .transpose()?,
        stream_interval: params.stream_interval.map(i64::from),
        logit_bias: params.logit_bias.as_ref().map(|bias| {
            // A map in key order: the order the other revision's map has.
            let mut entries: Vec<(String, f64)> = bias
                .iter()
                .map(|(token, bias)| (token.clone(), *bias))
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            Dict(entries)
        }),
        seed: params.seed.map(|seed| wire_i64(seed, "seed")).transpose()?,
        logprobs: params.logprobs.map(i64::from),
        n: i64::from(params.n),
        stop_strs: Some(StrOrList::Many(params.stop_strs.clone())),
        stop_str_max_len: i64::from(params.stop_str_max_len),
        is_normalized: params.is_normalized,
    })
}

/// The free-form parameters as the engine types them: a map under string
/// keys.
fn to_wire_custom_params(params: &OpaqueValue) -> Result<Dict<String, Value>> {
    let OpaqueValue::Map(entries) = params else {
        return Err(refused_value(
            "custom_params",
            format!("expected a map, got {params}"),
        ));
    };
    let mut converted = Vec::with_capacity(entries.len());
    for (key, value) in entries {
        let Some(key) = key.as_str() else {
            return Err(refused_value(
                "custom_params",
                format!("expected string keys, got the key {key}"),
            ));
        };
        converted.push((key.to_owned(), to_wire_value(value)?));
    }
    Ok(Dict(converted))
}

/// A dynamic msgpack value in the crate's own representation, as it came
/// (map order and integer signedness kept, so the bytes come out the same).
fn to_wire_value(value: &OpaqueValue) -> Result<Value> {
    Ok(match value {
        OpaqueValue::Nil => Value::Nil,
        OpaqueValue::Boolean(value) => Value::Bool(*value),
        OpaqueValue::Integer(value) => {
            if let Some(unsigned) = value.as_u64() {
                Value::Int(Integer::Unsigned(unsigned))
            } else if let Some(signed) = value.as_i64() {
                Value::Int(Integer::Signed(signed))
            } else {
                return Err(refused_value(
                    "custom_params",
                    format!("the integer {value} does not fit the engine's integers"),
                ));
            }
        }
        OpaqueValue::F32(value) => Value::Float(f64::from(*value)),
        OpaqueValue::F64(value) => Value::Float(*value),
        OpaqueValue::String(value) => match value.as_str() {
            Some(text) => Value::Str(text.to_owned()),
            None => Value::Bin(Bytes::copy_from_slice(value.as_bytes())),
        },
        OpaqueValue::Binary(value) => Value::Bin(Bytes::copy_from_slice(value)),
        OpaqueValue::Array(items) => Value::Array(
            items
                .iter()
                .map(to_wire_value)
                .collect::<Result<Vec<Value>>>()?,
        ),
        OpaqueValue::Map(entries) => Value::Map(
            entries
                .iter()
                .map(|(key, value)| Ok((to_wire_value(key)?, to_wire_value(value)?)))
                .collect::<Result<Vec<(Value, Value)>>>()?,
        ),
        OpaqueValue::Ext(code, data) => Value::Ext(*code, Bytes::copy_from_slice(data)),
    })
}

fn to_wire_multimodal(inputs: &TokenSpeedWireMmInputs) -> Result<ts_wire::MultimodalInputs> {
    Ok(ts_wire::MultimodalInputs {
        mm_items: inputs
            .mm_items
            .iter()
            .map(to_wire_item)
            .collect::<Result<Vec<_>>>()?,
        im_token_id: inputs.im_token_id.map(i64::from),
        video_token_id: inputs.video_token_id.map(i64::from),
        mrope_positions: inputs
            .mrope_positions
            .as_ref()
            .map(|tensor| to_wire_tensor(tensor, "mrope_positions"))
            .transpose()?,
        mrope_position_delta: inputs
            .mrope_position_delta
            .as_ref()
            .map(|tensor| to_wire_tensor(tensor, "mrope_position_delta"))
            .transpose()?,
        mrope_position_delta_scalar: inputs.mrope_position_delta_scalar,
        mrope_position_delta_repeated_cache: None,
    })
}

fn to_wire_item(item: &TokenSpeedWireMmItem) -> Result<ts_wire::MultimodalDataItem> {
    // The side tensors in key order: the order the other revision's map has.
    let mut model_specific_data = Vec::with_capacity(item.model_specific_data.len());
    for (name, tensor) in &item.model_specific_data {
        model_specific_data.push((name.clone(), to_wire_tensor(tensor, "model_specific_data")?));
    }
    let mut offsets = Vec::with_capacity(item.offsets.len());
    for &(start, end) in &item.offsets {
        offsets.push((wire_i64(start, "offsets")?, wire_i64(end, "offsets")?));
    }
    Ok(ts_wire::MultimodalDataItem {
        modality: match item.modality {
            TokenSpeedWireModality::Image => ts_wire::Modality::Image,
            TokenSpeedWireModality::Video => ts_wire::Modality::Video,
            TokenSpeedWireModality::Audio => ts_wire::Modality::Audio,
        },
        hash: Some(item.hash),
        pad_value: Some(i64::from(item.pad_value)),
        offsets: Some(offsets),
        feature: item
            .feature
            .as_ref()
            .map(|tensor| to_wire_tensor(tensor, "feature"))
            .transpose()?,
        feature_shm: None,
        model_specific_data: Dict(model_specific_data),
        encoded: None,
        encoded_deepstack: None,
        // The engine's dict, under the keys and in the order the other
        // revision's map has them.
        encode_handshake: item.encode_handshake.as_ref().map(|handshake| {
            Dict(vec![
                (
                    Value::from("bootstrap_host"),
                    Value::Str(handshake.bootstrap_host.clone()),
                ),
                (
                    Value::from("bootstrap_port"),
                    Value::from(i64::from(handshake.bootstrap_port)),
                ),
                (
                    Value::from("bootstrap_room"),
                    Value::from(handshake.bootstrap_room),
                ),
            ])
        }),
    })
}

/// A tensor's bytes shared with the crate's struct (no copy); the crate
/// lifts the large ones into frames itself, so a payload that already
/// addresses a frame has nothing to hand it.
fn to_wire_tensor(tensor: &WireTensor, field: &'static str) -> Result<ts_wire::Tensor> {
    let data = match &tensor.data {
        WireArrayData::RawView(bytes) => bytes.clone(),
        WireArrayData::AuxIndex(index) => {
            return Err(encode_error(
                field,
                format!(
                    "the tensor addresses auxiliary frame {index}; this framing lifts the \
                     tensors itself"
                ),
            ))
        }
    };
    let mut shape = Vec::with_capacity(tensor.shape.len());
    for &dimension in &tensor.shape {
        shape.push(wire_i64(dimension, field)?);
    }
    Ok(ts_wire::Tensor {
        dtype: tensor.dtype.clone(),
        shape,
        data,
    })
}

/// The engine's slim batch as the connector's outputs: one per request,
/// with the batch's load snapshot and the finished ids beside them. The
/// finish details, the speculative counters and the ranked and prompt
/// logprob columns the batch may carry are left unread (their own changes).
fn from_wire_batch(batch: &ts_wire::BatchTokenIDOutSlim) -> Result<EngineBatch<TokenSpeedOutput>> {
    let rows = batch
        .requests()
        .map_err(|error| decode_error("BatchTokenIDOutSlim", error))?;
    let mut outputs = Vec::with_capacity(batch.rids.as_ref().map_or(0, Vec::len));
    for (index, row) in rows.enumerate() {
        outputs.push(TokenSpeedOutput {
            request_id: row.rid.to_owned(),
            output_ids: token_ids(row.output_ids, "output_ids")?,
            finish_reason: row.finish_reason.map(str::to_owned),
            prompt_tokens: count(row.prompt_tokens, "prompt_tokens")?,
            completion_tokens: count(row.completion_tokens, "completion_tokens")?,
            cached_tokens: count(row.cached_tokens, "cached_tokens")?,
            output_logprobs_val: row.output_token_logprobs.to_vec(),
            // Parallel to the values, checked by `requests()`.
            output_logprobs_idx: token_ids(
                &batch.output_token_logprobs_idx[index],
                "output_token_logprobs_idx",
            )?,
            finish_status: None,
            finish_message: None,
        });
    }
    // `kv_total_pages == 0` marks a snapshot the engine could not take: no
    // load, not a fabricated empty scheduler (the other revision's rule). A
    // negative count or page figure is no snapshot either: the columns are
    // signed on this revision, and a bad value read as zero would look like
    // an idle rank (the other revision's decoder refuses such a frame).
    let load = (batch.kv_total_pages > 0
        && batch.num_running >= 0
        && batch.num_waiting >= 0
        && batch.kv_active_pages >= 0)
        .then(|| EngineLoad {
            num_running: u64::try_from(batch.num_running).unwrap_or(0),
            num_waiting: u64::try_from(batch.num_waiting).unwrap_or(0),
            kv_cache_usage: batch.kv_active_pages as f64 / batch.kv_total_pages as f64,
        });
    let finished_request_ids = outputs
        .iter()
        .filter(|output| output.finish_reason.is_some())
        .map(|output| output.request_id.clone())
        .collect();
    Ok(EngineBatch {
        engine_index: u32::try_from(batch.engine_index).map_err(|_| {
            decode_error(
                "BatchTokenIDOutSlim",
                ts_wire::Error::Invalid(format!(
                    "engine_index {} is not a rank index",
                    batch.engine_index
                )),
            )
        })?,
        outputs,
        finished_request_ids,
        load,
        wave: None,
        utility: None,
    })
}

fn wire_token_ids(ids: &[u32], field: &'static str) -> Result<Vec<i32>> {
    ids.iter()
        .map(|&id| {
            i32::try_from(id).map_err(|_| {
                refused_value(
                    field,
                    format!("the token id {id} does not fit the engine's token ids"),
                )
            })
        })
        .collect()
}

/// A request integer as the engine's signed one. A value past `i64::MAX`
/// (a `sampling_seed`, a `thinking_budget`) is the request's own fault, not
/// the wire's: the refusal names the field and reads as an invalid argument.
fn wire_i64<T: TryInto<i64> + Copy + std::fmt::Display>(
    value: T,
    field: &'static str,
) -> Result<i64> {
    value.try_into().map_err(|_| Error::RequestValueOutOfRange {
        field,
        message: format!("the value {value} does not fit the engine's integers"),
    })
}

fn token_ids(ids: &[i32], field: &'static str) -> Result<Vec<u32>> {
    ids.iter()
        .map(|&id| {
            u32::try_from(id).map_err(|_| {
                decode_error(
                    field,
                    ts_wire::Error::Invalid(format!("the token id {id} is negative")),
                )
            })
        })
        .collect()
}

fn count(value: i64, field: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        decode_error(
            field,
            ts_wire::Error::Invalid(format!("the count {value} is not a token count")),
        )
    })
}

fn encode_error(field: &'static str, message: String) -> Error {
    Error::Encode {
        target_type: "ts_wire::TokenizedGenerateReqInput",
        message: format!("field `{field}`: {message}"),
    }
}

/// A value of the request the wire cannot carry: the caller's error, named
/// by field and read as an invalid argument; `encode_error` is for failures
/// of the wire's own (a number the gateway drew, an auxiliary index).
fn refused_value(field: &'static str, message: String) -> Error {
    Error::RequestValueOutOfRange { field, message }
}

fn decode_error(target_type: &'static str, error: ts_wire::Error) -> Error {
    Error::Decode {
        target_type,
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{
        codec::{encode_msgpack, tensor::TENSOR_AUX_MIN_BYTES},
        protocol::{
            tokenspeed::{
                multimodal::{mm_pad_value, EncodeHandshake},
                TokenSpeedLayout, TokenSpeedProtocol, WireRevision,
            },
            EngineProtocol,
        },
    };

    fn rust_wire() -> TokenSpeedLayout {
        TokenSpeedLayout::default().with_wire(WireRevision::RustControlPlane)
    }

    /// A seed past the engine's integer range is the request's value, not a
    /// failure of the wire: the conversion refuses it by name.
    #[test]
    fn a_seed_past_i64_max_is_refused_as_the_requests_value() {
        let mut request = text_request();
        request.sampling_params.seed = Some(u64::MAX);
        let error = to_wire_request(&request).expect_err("the seed does not fit");
        assert!(
            matches!(error, Error::RequestValueOutOfRange { field: "seed", .. }),
            "{error:?}"
        );
    }

    /// A token id or a free-form parameter the wire cannot carry is the
    /// request's value too, not a failure of the wire: refused as an invalid
    /// argument, by field.
    #[test]
    fn token_ids_and_custom_params_the_wire_cannot_carry_are_the_requests_values() {
        let mut request = text_request();
        request.input_ids = vec![u32::MAX];
        let error = to_wire_request(&request).expect_err("the token id does not fit");
        assert!(
            matches!(
                error,
                Error::RequestValueOutOfRange {
                    field: "input_ids",
                    ..
                }
            ),
            "{error:?}"
        );

        let mut request = text_request();
        request.sampling_params.custom_params = Some(OpaqueValue::from(3));
        let error = to_wire_request(&request).expect_err("a scalar is not the engine's map");
        assert!(
            matches!(
                error,
                Error::RequestValueOutOfRange {
                    field: "custom_params",
                    ..
                }
            ),
            "{error:?}"
        );
    }

    /// The text request of the other revision's pinned vector, as the
    /// adapter builds it.
    fn text_request() -> TokenizedGenerateReqInput {
        let mut sampling_params = SamplingParams {
            max_new_tokens: Some(8),
            stop_token_ids: Some(vec![2]),
            temperature: 0.5,
            top_p: 0.9,
            seed: Some(42),
            ..SamplingParams::default()
        };
        sampling_params.normalize();
        TokenizedGenerateReqInput {
            rid: "vec-1".to_string(),
            input_ids: vec![1, 2, 3],
            sampling_params,
            return_logprob: true,
            stream: true,
            ..TokenizedGenerateReqInput::default()
        }
    }

    fn image_item(hash: u64, feature_bytes: usize) -> TokenSpeedWireMmItem {
        TokenSpeedWireMmItem {
            modality: TokenSpeedWireModality::Image,
            hash,
            pad_value: mm_pad_value(TokenSpeedWireModality::Image, hash),
            offsets: vec![(1, 3)],
            feature: Some(WireTensor::from_raw(
                "bfloat16",
                vec![feature_bytes / 2, 1],
                vec![7; feature_bytes],
            )),
            model_specific_data: BTreeMap::from([(
                "vit_grid".to_string(),
                WireTensor::from_raw(
                    "uint32",
                    vec![1, 3],
                    vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0],
                ),
            )]),
            encode_handshake: None,
        }
    }

    fn media_request(feature_bytes: usize) -> TokenizedGenerateReqInput {
        let pad = mm_pad_value(TokenSpeedWireModality::Image, 0xDEAD_BEEF);
        let mut request = text_request();
        request.rid = "mm-1".to_string();
        request.input_ids = vec![10, pad, pad, pad, 50];
        request.input_ids_unpadded = Some(vec![10, 20, 30, 40, 50]);
        request.multimodal_inputs = Some(TokenSpeedWireMmInputs {
            mm_items: vec![image_item(0xDEAD_BEEF, feature_bytes)],
            im_token_id: Some(9),
            video_token_id: None,
            mrope_positions: Some(
                WireTensor::from_i64(vec![3, 2], vec![0, 1, 0, 1, 0, 1]).unwrap(),
            ),
            mrope_position_delta: Some(WireTensor::from_i64(vec![1, 1], vec![-3]).unwrap()),
            mrope_position_delta_scalar: Some(-3),
        });
        request
    }

    /// The engine's own decoder (the crate's) over a message this module
    /// framed: what the scheduler of the Rust control plane reads.
    fn engine_reads(payload: &[u8], aux: &[Bytes]) -> ts_wire::TokenizedGenerateReqInput {
        let mut frames = vec![Bytes::from_static(b"\x00"), Bytes::copy_from_slice(payload)];
        frames.extend(aux.iter().cloned());
        match ts_wire::EngineRequest::from_frames(&frames).unwrap() {
            ts_wire::EngineRequest::Add(request) => *request,
            other => panic!("expected an add message, got {other:?}"),
        }
    }

    /// A text request is framed as the engine's crate frames its own
    /// struct (the full array, field for field what the adapter set, the
    /// rest at the engine's defaults), which is not the other revision's
    /// prefix; the engine's decoder reads back every field.
    #[test]
    fn a_text_request_is_framed_as_the_engines_own_struct() {
        let request = text_request();
        let (payload, aux) = TokenSpeedProtocol::encode_add(&request, &rust_wire()).unwrap();
        assert!(aux.is_empty());
        let expected = to_wire_request(&request).unwrap();
        assert_eq!(payload, ts_wire::encode_frames(&expected)[0]);
        assert_ne!(
            payload,
            encode_msgpack(&request).unwrap(),
            "the other revision's prefix"
        );

        let read = engine_reads(&payload, &aux);
        assert_eq!(read, expected);
        assert_eq!(read.rid, Some(StrOrList::One("vec-1".to_string())));
        assert_eq!(read.input_ids, Some(vec![1, 2, 3]));
        assert!(read.return_logprob && read.stream);
        assert_eq!(read.logprob_start_len, -1);
        assert_eq!(read.top_logprobs_num, 0);
        assert_eq!(read.data_parallel_rank, None);
        let sampling = &read.sampling_params;
        assert_eq!(sampling.max_new_tokens, Some(8));
        assert_eq!(sampling.stop_token_ids, Some(vec![2]));
        assert_eq!(sampling.temperature, 0.5);
        assert_eq!(sampling.top_k, 1 << 30);
        assert_eq!(sampling.seed, Some(42));
        assert_eq!(sampling.n, 1);
        assert_eq!(sampling.stop_strs, Some(StrOrList::Many(Vec::new())));
        assert!(sampling.is_normalized);
    }

    /// The connector's routing instruction never reaches the engine's pin
    /// field: a request pinned to a rank by the sender is framed without it,
    /// as on the other revision (the rank is the socket it goes out on).
    #[test]
    fn the_connectors_rank_is_not_the_engines_pin() {
        let mut request = text_request();
        request.data_parallel_rank = Some(1);
        let (payload, aux) = TokenSpeedProtocol::encode_add(&request, &rust_wire()).unwrap();
        assert_eq!(engine_reads(&payload, &aux).data_parallel_rank, None);
    }

    /// The crate lifts a tensor of 256 bytes or more into its own frame (the
    /// engine's Python frontend's threshold), where the other revision
    /// inlines everything below 32 KiB; the frames are addressed in slot
    /// order and the engine's decoder resolves them, the bytes shared with
    /// the request's buffers.
    #[test]
    fn a_media_request_lifts_its_tensors_at_the_crates_threshold() {
        for feature_bytes in [254, 256, 4096, TENSOR_AUX_MIN_BYTES] {
            let request = media_request(feature_bytes);
            let (payload, aux) = TokenSpeedProtocol::encode_add(&request, &rust_wire()).unwrap();
            // The feature from 256 bytes on; the 12-byte grid, the 48-byte
            // table and the 8-byte delta stay inline.
            let lifted = usize::from(feature_bytes >= 256);
            assert_eq!(aux.len(), lifted, "feature of {feature_bytes} bytes");
            if lifted == 1 {
                let feature = request.multimodal_inputs.as_ref().unwrap().mm_items[0]
                    .feature
                    .as_ref()
                    .unwrap();
                assert_eq!(
                    aux[0].as_ptr(),
                    feature.data.as_raw_view().unwrap().as_ptr()
                );
            }
            let read = engine_reads(&payload, &aux);
            assert_eq!(read, to_wire_request(&request).unwrap());
            let mm = read.multimodal_inputs.unwrap();
            assert_eq!(mm.im_token_id, Some(9));
            assert_eq!(mm.mrope_position_delta_scalar, Some(-3));
            assert_eq!(mm.mrope_positions.as_ref().unwrap().shape, vec![3, 2]);
            let item = &mm.mm_items[0];
            assert_eq!(item.modality, ts_wire::Modality::Image);
            assert_eq!(item.hash, Some(0xDEAD_BEEF));
            assert_eq!(
                item.pad_value,
                Some(i64::from(mm_pad_value(
                    TokenSpeedWireModality::Image,
                    0xDEAD_BEEF
                )))
            );
            assert_eq!(item.offsets, Some(vec![(1, 3)]));
            assert_eq!(item.feature.as_ref().unwrap().data.len(), feature_bytes);
            assert_eq!(item.model_specific_data.0[0].0, "vit_grid");
            assert_eq!(item.encode_handshake, None);
            assert_eq!(read.input_ids_unpadded, Some(vec![10, 20, 30, 40, 50]));
        }
    }

    /// A disaggregated request keeps its rendezvous in the engine's own
    /// fields and an item's encode handshake rides as the engine's dict,
    /// under the keys of the other revision's map; a request whose tensors
    /// already address frames has nothing to hand the crate.
    #[test]
    fn a_disaggregated_request_keeps_its_rendezvous() {
        let mut request = media_request(16);
        request.rid = "pd-1".to_string();
        request.bootstrap_host = Some("10.0.0.7".to_string());
        request.bootstrap_port = Some(8998);
        request.bootstrap_room = Some(6_148_914_691_236_517_205);
        let items = &mut request.multimodal_inputs.as_mut().unwrap().mm_items;
        items[0].feature = None;
        items[0].encode_handshake = Some(EncodeHandshake {
            bootstrap_host: "10.0.0.9".to_string(),
            bootstrap_port: 8999,
            bootstrap_room: 41,
        });
        let (payload, aux) = TokenSpeedProtocol::encode_add(&request, &rust_wire()).unwrap();
        let read = engine_reads(&payload, &aux);
        assert_eq!(read.bootstrap_host.as_deref(), Some("10.0.0.7"));
        assert_eq!(read.bootstrap_port, Some(8998));
        assert_eq!(read.bootstrap_room, Some(6_148_914_691_236_517_205));
        let item = &read.multimodal_inputs.unwrap().mm_items[0];
        assert_eq!(item.feature, None);
        assert_eq!(
            item.encode_handshake,
            Some(Dict(vec![
                (Value::from("bootstrap_host"), Value::from("10.0.0.9")),
                (Value::from("bootstrap_port"), Value::from(8999i64)),
                (Value::from("bootstrap_room"), Value::from(41i64)),
            ]))
        );

        let mut request = media_request(16);
        request.bootstrap_room = Some(-1);
        let error = TokenSpeedProtocol::encode_add(&request, &rust_wire())
            .unwrap_err()
            .to_string();
        assert!(error.contains("bootstrap_room"), "{error}");

        let mut request = media_request(16);
        request.multimodal_inputs.as_mut().unwrap().mm_items[0]
            .feature
            .as_mut()
            .unwrap()
            .data = WireArrayData::AuxIndex(1);
        let error = TokenSpeedProtocol::encode_add(&request, &rust_wire())
            .unwrap_err()
            .to_string();
        assert!(error.contains("auxiliary frame 1"), "{error}");
    }

    /// A value the engine's schema cannot carry is refused by the field's
    /// name, never sent truncated.
    #[test]
    fn values_the_schema_cannot_carry_are_refused_by_name() {
        let mut request = text_request();
        request.sampling_params.seed = Some(u64::MAX);
        let error = TokenSpeedProtocol::encode_add(&request, &rust_wire())
            .unwrap_err()
            .to_string();
        assert!(error.contains("`seed`"), "{error}");

        let mut request = text_request();
        request.input_ids = vec![u32::MAX];
        let error = TokenSpeedProtocol::encode_add(&request, &rust_wire())
            .unwrap_err()
            .to_string();
        assert!(error.contains("`input_ids`"), "{error}");
    }

    /// The free-form parameters go out as the engine's map, the values as
    /// they came.
    #[test]
    fn custom_params_ride_as_the_engines_map() {
        let mut request = text_request();
        request.sampling_params.custom_params = Some(OpaqueValue::Map(vec![
            (
                OpaqueValue::from("k"),
                OpaqueValue::Array(vec![
                    OpaqueValue::from(1),
                    OpaqueValue::from(-2),
                    OpaqueValue::from("v"),
                    OpaqueValue::Nil,
                    OpaqueValue::F64(2.5),
                ]),
            ),
            (OpaqueValue::from("flag"), OpaqueValue::Boolean(true)),
        ]));
        let (payload, aux) = TokenSpeedProtocol::encode_add(&request, &rust_wire()).unwrap();
        assert_eq!(
            engine_reads(&payload, &aux).sampling_params.custom_params,
            Some(Dict(vec![
                (
                    "k".to_string(),
                    Value::Array(vec![
                        Value::from(1i64),
                        Value::from(-2i64),
                        Value::from("v"),
                        Value::Nil,
                        Value::from(2.5),
                    ]),
                ),
                ("flag".to_string(), Value::from(true)),
            ]))
        );
        request.sampling_params.custom_params = Some(OpaqueValue::from(3));
        let error = TokenSpeedProtocol::encode_add(&request, &rust_wire())
            .unwrap_err()
            .to_string();
        assert!(error.contains("`custom_params`"), "{error}");
    }

    fn wire_batch() -> ts_wire::BatchTokenIDOutSlim {
        ts_wire::BatchTokenIDOutSlim {
            rids: Some(vec!["a".to_string(), "b".to_string()]),
            output_ids: vec![vec![10, 11], vec![]],
            finished_reasons: vec![String::new(), "abort".to_string()],
            prompt_tokens: vec![3, 8191],
            completion_tokens: vec![2, 0],
            cached_tokens: vec![1, 0],
            output_token_logprobs_val: vec![vec![-0.5, -0.25], vec![]],
            output_token_logprobs_idx: vec![vec![10, 11], vec![]],
            engine_index: 1,
            num_running: 2,
            num_waiting: 5,
            kv_active_pages: 100,
            kv_total_pages: 400,
            finish_reason_details: vec![
                None,
                Some(Dict(vec![
                    ("type".to_string(), Value::from("abort")),
                    (
                        "message".to_string(),
                        Value::from("Invalid request: too long"),
                    ),
                    ("err_type".to_string(), Value::from(522i64)),
                    ("status_code".to_string(), Value::from(400i64)),
                ])),
            ],
            spec_verify_ct: vec![2, 0],
            accept_draft_tokens: vec![Some(1.5), None],
            input_token_logprobs_val: Vec::new(),
            input_token_logprobs_idx: Vec::new(),
            input_top_logprobs_val: Vec::new(),
            input_top_logprobs_idx: Vec::new(),
            output_top_logprobs_val: vec![vec![vec![-0.5, -3.0], vec![-0.25, -2.0]], vec![]],
            output_top_logprobs_idx: vec![vec![vec![10, 1], vec![11, 2]], vec![]],
        }
    }

    /// The engine's 22-column batch decodes to the connector's outputs: the
    /// columns the other revision reads, the load snapshot, the finished
    /// ids; the details the batch carries beyond them are left unread.
    #[test]
    fn the_engines_slim_batch_decodes_to_the_connectors_outputs() {
        let frames = ts_wire::EngineOutput::Batch(Box::new(wire_batch())).to_frames();
        let decoded = TokenSpeedProtocol::decode_batch(&frames, &rust_wire()).unwrap();
        assert_eq!(decoded.engine_index, 1);
        assert_eq!(
            decoded.load,
            Some(EngineLoad {
                num_running: 2,
                num_waiting: 5,
                kv_cache_usage: 0.25,
            })
        );
        assert_eq!(decoded.finished_request_ids, vec!["b".to_string()]);
        assert_eq!(
            decoded.outputs,
            vec![
                TokenSpeedOutput {
                    request_id: "a".to_string(),
                    output_ids: vec![10, 11],
                    finish_reason: None,
                    prompt_tokens: 3,
                    completion_tokens: 2,
                    cached_tokens: 1,
                    output_logprobs_val: vec![-0.5, -0.25],
                    output_logprobs_idx: vec![10, 11],
                    finish_status: None,
                    finish_message: None,
                },
                TokenSpeedOutput {
                    request_id: "b".to_string(),
                    output_ids: Vec::new(),
                    finish_reason: Some("abort".to_string()),
                    prompt_tokens: 8191,
                    completion_tokens: 0,
                    cached_tokens: 0,
                    output_logprobs_val: Vec::new(),
                    output_logprobs_idx: Vec::new(),
                    finish_status: None,
                    finish_message: None,
                },
            ]
        );
        assert!(decoded.utility.is_none() && decoded.wave.is_none());

        let no_snapshot = ts_wire::BatchTokenIDOutSlim {
            kv_total_pages: 0,
            ..wire_batch()
        };
        let frames = ts_wire::EngineOutput::Batch(Box::new(no_snapshot)).to_frames();
        assert!(TokenSpeedProtocol::decode_batch(&frames, &rust_wire())
            .unwrap()
            .load
            .is_none());
    }

    /// A negative count or page figure is no snapshot: the batch's outputs
    /// still decode, the load reads `None` (never an idle rank or a negative
    /// usage), whichever of the three signed columns carries it.
    #[test]
    fn a_negative_load_figure_is_no_snapshot() {
        for bad in [
            ts_wire::BatchTokenIDOutSlim {
                num_running: -1,
                ..wire_batch()
            },
            ts_wire::BatchTokenIDOutSlim {
                num_waiting: -4,
                ..wire_batch()
            },
            ts_wire::BatchTokenIDOutSlim {
                kv_active_pages: -3,
                ..wire_batch()
            },
        ] {
            let frames = ts_wire::EngineOutput::Batch(Box::new(bad)).to_frames();
            let decoded = TokenSpeedProtocol::decode_batch(&frames, &rust_wire()).unwrap();
            assert!(decoded.load.is_none());
            assert_eq!(
                decoded.outputs.len(),
                wire_batch().rids.map_or(0, |rids| rids.len())
            );
        }
    }

    /// A batch without the tail columns reads the same under both revisions:
    /// the other revision's decoder skips the columns it does not know, this
    /// one leaves them unread.
    #[test]
    fn both_revisions_read_a_batch_the_same_way() {
        let batch = ts_wire::BatchTokenIDOutSlim {
            finish_reason_details: Vec::new(),
            spec_verify_ct: Vec::new(),
            accept_draft_tokens: Vec::new(),
            output_top_logprobs_val: Vec::new(),
            output_top_logprobs_idx: Vec::new(),
            ..wire_batch()
        };
        let frames = ts_wire::EngineOutput::Batch(Box::new(batch)).to_frames();
        let rust = TokenSpeedProtocol::decode_batch(&frames, &rust_wire()).unwrap();
        let python =
            TokenSpeedProtocol::decode_batch(&frames, &TokenSpeedLayout::default()).unwrap();
        assert_eq!(rust.outputs, python.outputs);
        assert_eq!(rust.load, python.load);
        assert_eq!(rust.finished_request_ids, python.finished_request_ids);
        assert_eq!(rust.engine_index, python.engine_index);
    }

    /// A control reply on the output socket is read and dropped (nothing on
    /// this revision sends a control operation yet); the dead sentinel is
    /// the engine's end; a ragged batch and a negative token id are decode
    /// failures that name the column.
    #[test]
    fn a_control_reply_is_dropped_and_bad_batches_are_named() {
        let reply = ts_wire::ControlOutput::FlushCache(ts_wire::control::FlushCacheReqOutput {
            rid: None,
            http_worker_ipc: None,
            success: true,
        });
        let frames = ts_wire::EngineOutput::Control(Box::new(reply)).to_frames();
        let decoded = TokenSpeedProtocol::decode_batch(&frames, &rust_wire()).unwrap();
        assert!(decoded.outputs.is_empty() && decoded.load.is_none() && decoded.utility.is_none());

        let frames = ts_wire::EngineOutput::Dead.to_frames();
        assert!(matches!(
            TokenSpeedProtocol::decode_batch(&frames, &rust_wire()),
            Err(Error::EngineCoreDead)
        ));

        let mut ragged = wire_batch();
        ragged.cached_tokens.pop();
        let frames = ts_wire::EngineOutput::Batch(Box::new(ragged)).to_frames();
        let error = TokenSpeedProtocol::decode_batch(&frames, &rust_wire())
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("cached_tokens"), "{error}");

        let mut negative = wire_batch();
        negative.output_ids[0][0] = -1;
        let frames = ts_wire::EngineOutput::Batch(Box::new(negative)).to_frames();
        let error = TokenSpeedProtocol::decode_batch(&frames, &rust_wire())
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("output_ids"), "{error}");
    }

    /// The other revision's framing is untouched by the switch: the default
    /// layout frames the request as before, byte for byte, and reads the
    /// batch as before.
    #[test]
    fn the_default_layout_keeps_the_other_revisions_bytes() {
        let request = media_request(4096);
        let (payload, aux) =
            TokenSpeedProtocol::encode_add(&request, &TokenSpeedLayout::default()).unwrap();
        assert!(aux.is_empty(), "4 KiB stays inline on the other revision");
        assert_eq!(payload, encode_msgpack(&request).unwrap());
        assert_eq!(
            TokenSpeedProtocol::read_layout(&crate::mock_engine::default_ready_response()).wire,
            WireRevision::PythonLoop
        );
        assert_eq!(
            WireRevision::parse("python_loop"),
            Some(WireRevision::PythonLoop)
        );
        assert_eq!(
            WireRevision::parse("rust_control_plane"),
            Some(WireRevision::RustControlPlane)
        );
        assert_eq!(WireRevision::parse("rust"), None);
        assert_eq!(
            WireRevision::RustControlPlane.as_str(),
            "rust_control_plane"
        );
    }
}
