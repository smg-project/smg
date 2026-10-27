//! Early, pixel-less SGLang PD preprocessing identity. A failed check leaves
//! the original reference request untouched so its worker can process it.
use prost::Message;
use sha2::{Digest, Sha256};

use crate::sglang_proto::{GenerateRequest, MediaIdentity, MultimodalInputs};

pub const LABEL: &str = "sglang_media_identity";
pub const VERSION: u32 = 1;
const MAX_IDENTITY_BYTES: usize = 4 * 1024 * 1024;

/// Recognize reference-only inputs without raw media or processor tensors.
pub fn has_media_refs(req: &GenerateRequest) -> bool {
    req.mm_inputs.as_ref().is_some_and(|mm| {
        mm.pixel_values.is_none()
            && (!mm.image_urls.is_empty() || !mm.video_urls.is_empty())
            && mm.audio_urls.is_empty()
            && mm.image_data.is_empty()
            && mm.video_data.is_empty()
            && mm.audio_data.is_empty()
            && mm.model_specific_tensors.is_empty()
    })
}

/// Bind to the original prompt, ordered media references and KV rendezvous.
/// Ignore side-specific routing/sampling fields. Length prefixes avoid ambiguity.
pub fn request_digest(req: &GenerateRequest) -> Vec<u8> {
    let mut hash = Sha256::new();
    /// Hash a length-prefixed field so adjacent values cannot produce ambiguous bindings.
    fn field(hash: &mut Sha256, bytes: &[u8]) {
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    field(&mut hash, req.request_id.as_bytes());
    if let Some(tokens) = &req.tokenized {
        field(&mut hash, &tokens.encode_to_vec());
    }
    if let Some(mm) = &req.mm_inputs {
        // Reference requests have no tensor map, so serialization is canonical.
        field(&mut hash, &mm.encode_to_vec());
    }
    if let Some(pd) = &req.disaggregated_params {
        field(&mut hash, &pd.encode_to_vec());
    }
    hash.finalize().to_vec()
}

/// Limit reuse to tokenized, single-choice PD requests with immutable media references.
pub fn reusable_request(req: &GenerateRequest) -> bool {
    has_media_refs(req)
        && req
            .tokenized
            .as_ref()
            .is_some_and(|t| !t.input_ids.is_empty())
        && req.sampling_params.as_ref().is_none_or(|p| p.n <= 1)
        && req
            .disaggregated_params
            .as_ref()
            .is_some_and(|p| !p.bootstrap_host.is_empty() && p.bootstrap_port > 0)
        && req.input_embeds.is_empty()
}

/// Conservative structural validation, shared by router and receiving worker.
pub fn validate(identity: &MediaIdentity, req: &GenerateRequest, key: &str) -> bool {
    if key.is_empty()
        || identity.version != VERSION
        || identity.processor_key != key
        || identity.request_digest != request_digest(req)
        || !reusable_request(req)
        || identity.input_ids.is_empty()
        || identity.encoded_len() > MAX_IDENTITY_BYTES
    {
        return false;
    }
    let Some(mm) = &identity.mm_inputs else {
        return false;
    };
    let Some(token) = mm.im_token_id else {
        return false;
    };
    if mm.pixel_values.is_some()
        || mm.feature_hash.is_none()
        || !mm.image_urls.is_empty()
        || !mm.video_urls.is_empty()
        || !mm.audio_urls.is_empty()
        || !mm.image_data.is_empty()
        || !mm.video_data.is_empty()
        || !mm.audio_data.is_empty()
        || mm.mm_placeholders.is_empty()
        || mm.modalities.is_empty()
        || !(mm.modalities.iter().all(|m| m == "image")
            || mm.modalities.iter().all(|m| m == "video"))
    {
        return false;
    }
    let mut previous_end = 0;
    for range in &mm.mm_placeholders {
        let start = range.offset as usize;
        let Some(end) = start.checked_add(range.length as usize) else {
            return false;
        };
        if range.length == 0
            || start < previous_end
            || end > identity.input_ids.len()
            || identity.input_ids[start..end].iter().any(|&id| id != token)
        {
            return false;
        }
        previous_end = end;
    }
    match crate::sglang_deepseek_media::layout(mm) {
        Ok(Some(items)) => {
            let Some(hashes) = mm
                .model_specific_tensors
                .get(crate::sglang_deepseek_media::HASHES)
            else {
                return false;
            };
            if crate::sglang_deepseek_media::integers(hashes, &[items.len() as u32]).is_err() {
                return false;
            }
        }
        Err(_) => return false,
        Ok(None) => {
            let grid_key = if mm.modalities[0] == "video" {
                "video_grid_thw"
            } else {
                "image_grid_thw"
            };
            let Some(grid) = mm.model_specific_tensors.get(grid_key) else {
                return false;
            };
            if grid.dtype != "int64"
                || grid.shape.len() != 2
                || grid.shape[0] == 0
                || grid.shape[1] != 3
            {
                return false;
            }
        }
    }
    // Validate all small position/grid tensors before the engine sees them.
    mm.model_specific_tensors.values().all(|t| {
        let width = match t.dtype.as_str() {
            "int64" => 8,
            "float32" | "uint32" => 4,
            "float16" | "bfloat16" => 2,
            "uint8" => 1,
            _ => return false,
        };
        t.shape
            .iter()
            .try_fold(width, |n: usize, &d| n.checked_mul(d as usize))
            == Some(t.data.len())
    })
}

/// Clone metadata while dropping reference-counted pixel storage from the control response.
pub fn pixel_less(mm: &MultimodalInputs) -> MultimodalInputs {
    let mut identity = mm.clone(); // Tensor bytes are reference counted.
    identity.pixel_values = None;
    identity
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sglang_proto::{
        DisaggregatedParams, PlaceholderRange, SamplingParams, TensorData, TokenizedInput,
    };

    /// Construct a minimal bound image identity with valid placeholders and grid bytes.
    fn fixture() -> (GenerateRequest, MediaIdentity) {
        let req = GenerateRequest {
            request_id: "a".into(),
            tokenized: Some(TokenizedInput {
                input_ids: vec![1, 7, 2],
                original_text: "image".into(),
            }),
            mm_inputs: Some(MultimodalInputs {
                image_urls: vec!["https://example.test/a.png".into()],
                ..Default::default()
            }),
            disaggregated_params: Some(DisaggregatedParams {
                bootstrap_host: "p".into(),
                bootstrap_port: 8998,
                bootstrap_room: 1,
            }),
            ..Default::default()
        };
        let mm = MultimodalInputs {
            feature_hash: Some(u64::MAX),
            im_token_id: Some(7),
            modalities: vec!["image".into()],
            mm_placeholders: vec![PlaceholderRange {
                offset: 1,
                length: 2,
            }],
            model_specific_tensors: [(
                "image_grid_thw".into(),
                TensorData {
                    dtype: "int64".into(),
                    shape: vec![1, 3],
                    data: [1i64, 2, 2]
                        .into_iter()
                        .flat_map(i64::to_le_bytes)
                        .collect::<Vec<_>>()
                        .into(),
                },
            )]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        let identity = MediaIdentity {
            version: VERSION,
            processor_key: "key".into(),
            request_digest: request_digest(&req),
            input_ids: vec![1, 7, 7, 2],
            mm_inputs: Some(mm),
        };
        (req, identity)
    }

    /// DeepSeek reuse accepts its own grid contract, and rejects missing hashes
    /// or changed image spans so the caller can retain the original references.
    #[test]
    fn deepseek_identity_requires_complete_native_image_metadata() {
        let (req, mut identity) = fixture();
        identity.input_ids = vec![1, 7, 7, 7, 7, 2];
        let mm = identity.mm_inputs.as_mut().unwrap();
        mm.mm_placeholders[0].length = 4;
        mm.model_specific_tensors.clear();
        for (name, shape, values) in [
            ("vit_grid", vec![1, 2], vec![3i64, 3]),
            ("llm_grid", vec![1, 2], vec![1, 1]),
            ("patches_per_image", vec![1], vec![9]),
            ("types_per_image", vec![1], vec![4]),
            (crate::sglang_deepseek_media::HASHES, vec![1], vec![-1]),
        ] {
            mm.model_specific_tensors.insert(
                name.into(),
                TensorData {
                    dtype: "int64".into(),
                    shape,
                    data: values
                        .into_iter()
                        .flat_map(i64::to_le_bytes)
                        .collect::<Vec<_>>()
                        .into(),
                },
            );
        }
        assert!(validate(&identity, &req, "key"));
        for field in [
            "vit_grid",
            "llm_grid",
            "patches_per_image",
            "types_per_image",
            crate::sglang_deepseek_media::HASHES,
        ] {
            let mut bad = identity.clone();
            bad.mm_inputs
                .as_mut()
                .unwrap()
                .model_specific_tensors
                .remove(field);
            assert!(!validate(&bad, &req, "key"), "{field}");
        }
        let mut bad = identity.clone();
        bad.mm_inputs.as_mut().unwrap().mm_placeholders[0].length = 3;
        assert!(!validate(&bad, &req, "key"));
        assert!(!validate(&identity, &req, "different-processor"));
    }

    #[test]
    /// Reject altered bindings and incomplete metadata without consuming fallback references.
    fn identity_is_bound_to_request_room_processor_and_complete_metadata() {
        let (req, identity) = fixture();
        assert!(validate(&identity, &req, "key"));
        assert!(!validate(&identity, &req, "other"));
        for case in 0..8 {
            let mut bad = identity.clone();
            match case {
                0 => bad.version += 1,
                1 => bad.request_digest.clear(),
                2 => bad.mm_inputs.as_mut().unwrap().feature_hash = None,
                3 => bad
                    .mm_inputs
                    .as_mut()
                    .unwrap()
                    .model_specific_tensors
                    .clear(),
                4 => bad.mm_inputs.as_mut().unwrap().mm_placeholders[0].offset = u32::MAX,
                5 => bad.input_ids[1] = 8,
                6 => bad.mm_inputs.as_mut().unwrap().pixel_values = Some(TensorData::default()),
                _ => {
                    bad.mm_inputs
                        .as_mut()
                        .unwrap()
                        .model_specific_tensors
                        .get_mut("image_grid_thw")
                        .unwrap()
                        .data = vec![0].into();
                }
            }
            assert!(!validate(&bad, &req, "key"), "case {case}");
        }
        let mut changed = req.clone();
        changed
            .disaggregated_params
            .as_mut()
            .unwrap()
            .bootstrap_room += 1;
        assert!(!validate(&identity, &changed, "key"));
        changed = req.clone();
        changed.request_id.push('b');
        assert!(!validate(&identity, &changed, "key"));
        changed = req.clone();
        changed.mm_inputs.as_mut().unwrap().image_urls[0].push('b');
        assert!(!validate(&identity, &changed, "key"));
        changed = req.clone();
        changed.sampling_params = Some(SamplingParams {
            n: 2,
            ..Default::default()
        });
        assert!(!validate(&identity, &changed, "key"));
        changed = req.clone();
        changed.disaggregated_params = None;
        assert!(!validate(&identity, &changed, "key"));
        assert_eq!(
            req,
            fixture().0,
            "validation never consumes fallback inputs"
        );
    }
}
