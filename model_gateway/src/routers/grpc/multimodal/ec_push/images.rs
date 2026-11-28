//! Image-only fan-out, retaining the original prompt outside image anchors.

use std::sync::Arc;

use smg_grpc_client::{common_proto::Modality, vllm_proto as vllm};
use smg_worker::identity::basic::Worker;

use crate::routers::grpc::{context::EncodeWorkerAssignment, proto_wrapper::ProtoGenerateRequest};

pub(super) struct ImageGroup {
    pub worker: Arc<dyn Worker>,
    indices: Vec<usize>,
    pub request: ProtoGenerateRequest,
}

fn tokens(request: &vllm::GenerateRequest) -> Option<&[u32]> {
    match request.input.as_ref()? {
        vllm::generate_request::Input::Tokenized(input) => Some(&input.input_ids),
        vllm::generate_request::Input::Text(_) => None,
    }
}

/// Batch images assigned to the same E; leave single-E and mixed media alone.
pub(super) fn split(
    request: &ProtoGenerateRequest,
    assignments: &[EncodeWorkerAssignment],
    anchor: u32,
) -> Option<Vec<ImageGroup>> {
    let ProtoGenerateRequest::Vllm(original) = request else {
        return None;
    };
    let refs = original.media_refs.as_ref()?;
    let input = tokens(original)?;
    if refs.items.len() != assignments.len()
        || refs
            .items
            .iter()
            .any(|item| item.modality != Modality::Image as i32)
        || input.iter().filter(|&&token| token == anchor).count() != refs.items.len()
    {
        return None;
    }
    let mut groups: Vec<ImageGroup> = Vec::new();
    for (index, assignment) in assignments.iter().enumerate() {
        if assignment.item_index != index {
            return None;
        }
        let group = groups.iter().position(|group| {
            group.worker.url() == assignment.worker.url()
                && group.worker.dp_rank() == assignment.worker.dp_rank()
        });
        if let Some(group) = group {
            groups[group].indices.push(index);
        } else {
            groups.push(ImageGroup {
                worker: assignment.worker.clone(),
                indices: vec![index],
                request: request.clone(),
            });
        }
    }
    if groups.len() < 2 {
        return None;
    }
    for group in &mut groups {
        let ProtoGenerateRequest::Vllm(child) = &mut group.request else {
            return None;
        };
        let vllm::generate_request::Input::Tokenized(input) = child.input.as_mut()? else {
            return None;
        };
        input.original_text.clear();
        let mut index = 0;
        input.input_ids.retain(|&token| {
            if token != anchor {
                return true;
            }
            let keep = group.indices.binary_search(&index).is_ok();
            index += 1;
            keep
        });
        let mut index = 0;
        child.media_refs.as_mut()?.items.retain(|_| {
            let keep = group.indices.binary_search(&index).is_ok();
            index += 1;
            keep
        });
    }
    Some(groups)
}

/// Merge only updates confined to each image anchor. A processor with other
/// prompt edits falls back to the existing full-prompt encode path.
pub(super) fn merge(
    request: &ProtoGenerateRequest,
    groups: &[ImageGroup],
    anchor: Option<u32>,
) -> Option<ProtoGenerateRequest> {
    let anchor = anchor?;
    let ProtoGenerateRequest::Vllm(original) = request else {
        return None;
    };
    let original_tokens = tokens(original)?;
    let count = original.media_refs.as_ref()?.items.len();
    let mut images = vec![None; count];
    let mut handles = serde_json::Map::new();
    let mut transfer_ids = vec![None; count];
    for group in groups {
        let ProtoGenerateRequest::Vllm(child) = &group.request else {
            return None;
        };
        let output = tokens(child)?;
        let mm = child.mm_inputs.as_ref()?;
        if !child.extra_mm_inputs.is_empty()
            || mm.pixel_values.is_some()
            || !mm.flat_keys.is_empty()
            || mm.mm_hashes.len() != group.indices.len()
            || mm.mm_placeholders.len() != group.indices.len()
        {
            return None;
        }
        let mut cursor = 0;
        let mut index = 0;
        for &token in original_tokens {
            if token != anchor {
                if output.get(cursor) != Some(&token) {
                    return None;
                }
                cursor += 1;
                continue;
            }
            if let Ok(local) = group.indices.binary_search(&index) {
                let range = &mm.mm_placeholders[local];
                let end = cursor.checked_add(range.length as usize)?;
                if range.offset as usize != cursor || range.length == 0 || images[index].is_some() {
                    return None;
                }
                images[index] = Some((mm, local, output.get(cursor..end)?));
                cursor = end;
            }
            index += 1;
        }
        if cursor != output.len() {
            return None;
        }
        let mut params: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(child.ec_transfer_params_json.as_ref()?).ok()?;
        if let Some(items) = params.remove("ec_items") {
            let items = items.as_array()?;
            if items.len() != group.indices.len() {
                return None;
            }
            for (&index, item) in group.indices.iter().zip(items) {
                transfer_ids[index] = Some(item.clone());
            }
        }
        handles.extend(params);
    }
    let images = images.into_iter().collect::<Option<Vec<_>>>()?;
    let first = images.first()?.0;
    if images.iter().any(|&(batch, _, _)| {
        batch.im_token_id != first.im_token_id
            || batch.modality != first.modality
            || batch.batched_keys != first.batched_keys
            || batch.keep_on_cpu_keys != first.keep_on_cpu_keys
            || batch.model_specific_tensors.len() != first.model_specific_tensors.len()
    }) {
        return None;
    }
    let mut mm = first.clone();
    mm.mm_hashes.clear();
    mm.mm_placeholders.clear();
    mm.model_specific_tensors.clear();
    for (key, template) in &first.model_specific_tensors {
        if !first.batched_keys.contains(key) {
            return None;
        }
        let mut data = Vec::new();
        for &(batch, local, _) in &images {
            let tensor = batch.model_specific_tensors.get(key)?;
            let Some(vllm::tensor_data::Payload::Inline(bytes)) = &tensor.payload else {
                return None;
            };
            let rows = batch.mm_hashes.len();
            if tensor.dtype != template.dtype
                || tensor.shape.first().copied()? as usize != rows
                || tensor.shape.get(1..)? != template.shape.get(1..)?
                || bytes.len() % rows != 0
            {
                return None;
            }
            let stride = bytes.len() / rows;
            data.extend_from_slice(bytes.get(local * stride..(local + 1) * stride)?);
        }
        let mut tensor = template.clone();
        tensor.shape[0] = count.try_into().ok()?;
        tensor.payload = Some(vllm::tensor_data::Payload::Inline(data.into()));
        mm.model_specific_tensors.insert(key.clone(), tensor);
    }
    let mut expanded = Vec::new();
    let mut index = 0;
    for &token in original_tokens {
        if token != anchor {
            expanded.push(token);
            continue;
        }
        let (batch, local, image_tokens) = images[index];
        mm.mm_hashes.push(batch.mm_hashes[local].clone());
        mm.mm_placeholders.push(vllm::PlaceholderRange {
            offset: expanded.len().try_into().ok()?,
            length: image_tokens.len().try_into().ok()?,
        });
        expanded.extend_from_slice(image_tokens);
        index += 1;
    }
    if transfer_ids.iter().any(Option::is_some) {
        handles.insert(
            "ec_items".into(),
            transfer_ids.into_iter().collect::<Option<Vec<_>>>()?.into(),
        );
    }
    let mut merged = request.clone();
    if !merged.apply_media_identity(&vllm::MediaIdentity {
        prompt_token_ids: expanded,
        mm_inputs: Some(mm),
        extra_mm_inputs: Vec::new(),
    }) {
        return None;
    }
    let ProtoGenerateRequest::Vllm(result) = &mut merged else {
        return None;
    };
    result.ec_transfer_params_json = Some(serde_json::Value::Object(handles).to_string());
    Some(merged)
}

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/multimodal/ec_push/images/tests.rs"]
mod tests;
