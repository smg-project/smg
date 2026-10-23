//! SGLang IMAGE media and contiguous encoder placement, separate from TS rooms.

use std::sync::Arc;

use anyhow::{anyhow, ensure, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use llm_multimodal::{ImageDetail, MediaContentPart};
use smg_grpc_client::sglang_proto;

use crate::{
    routers::grpc::{
        context::EncodeWorkerAssignment,
        multimodal::{encode_routing_hashes, MediaPlan, MultimodalIntermediate},
    },
    worker::{ConnectionMode, RuntimeType, WorkerType},
};

pub(crate) struct SglangEncodePlan {
    pub images: Arc<[String]>,
    pub hashes: Arc<[Vec<u8>]>,
}

impl SglangEncodePlan {
    pub fn prepare(intermediate: &MultimodalIntermediate) -> Result<Self> {
        let media = intermediate
            .sglang_encoder_media
            .as_ref()
            .ok_or_else(|| anyhow!("original SGLang encoder media is missing"))?;
        let images = image_urls(media)?;
        let hashes = encode_routing_hashes(intermediate)?;
        ensure!(
            images.len() == hashes.len(),
            "SGLang image/hash count differs"
        );
        Ok(Self {
            images: images.into(),
            hashes: hashes.into(),
        })
    }
}

fn image_urls(media: &MediaPlan) -> Result<Vec<String>> {
    let images = media
        .parts()
        .iter()
        .map(|part| {
            let (url, detail) = match part {
                MediaContentPart::ImageUrl {
                    url,
                    detail,
                    max_long_side_pixel: None,
                    ..
                } => (url.clone(), detail),
                MediaContentPart::ImageData {
                    data,
                    mime_type,
                    detail,
                    ..
                } => {
                    let mime = match mime_type {
                        Some(mime) => mime.as_str(),
                        None => image::guess_format(data)?.to_mime_type(),
                    };
                    ensure!(
                        !data.is_empty() && mime.starts_with("image/"),
                        "invalid inline image"
                    );
                    (
                        format!("data:{mime};base64,{}", STANDARD.encode(data)),
                        detail,
                    )
                }
                _ => anyhow::bail!("SGLang encoder requires images without processing overrides"),
            };
            ensure!(!url.is_empty(), "empty encoder image URL");
            ensure!(
                detail.is_none_or(|d| d == ImageDetail::Auto),
                "encoder cannot honor image detail override"
            );
            Ok(url)
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(!images.is_empty(), "encoder image list is empty");
    Ok(images)
}

/// Keep the selected encoder set and image order; redistribute contiguous runs.
/// This intentionally changes per-item cache affinity for A,B,A assignments.
pub(crate) fn contiguous(
    assignments: &[EncodeWorkerAssignment],
) -> Result<Vec<EncodeWorkerAssignment>> {
    ensure!(
        !assignments.is_empty(),
        "SGLang encoder assignments are empty"
    );
    let mut workers = Vec::new();
    for (index, assignment) in assignments.iter().enumerate() {
        ensure!(assignment.item_index == index, "encoder item order differs");
        let worker = &assignment.worker;
        ensure!(
            worker.metadata().spec.runtime_type == RuntimeType::Sglang
                && *worker.worker_type() == WorkerType::Encode
                && *worker.connection_mode() == ConnectionMode::Grpc,
            "SGLang encoder requires actual gRPC encode workers"
        );
        let endpoint = worker.url();
        let url = url::Url::parse(endpoint)?;
        ensure!(
            endpoint.starts_with("grpc://")
                && url.host_str().is_some()
                && url.port().is_some_and(|p| p != 0)
                && url.username().is_empty()
                && url.password().is_none()
                && url.path().is_empty()
                && url.query().is_none()
                && url.fragment().is_none()
                && !endpoint.chars().any(char::is_whitespace),
            "noncanonical encoder endpoint"
        );
        if !workers
            .iter()
            .any(|w: &Arc<dyn crate::worker::Worker>| w.url() == endpoint)
        {
            workers.push(Arc::clone(worker));
        }
    }
    let mut mapped = Vec::with_capacity(assignments.len());
    for (index, worker) in workers.iter().enumerate() {
        let count = assignments.len() / workers.len()
            + usize::from(index < assignments.len() % workers.len());
        for _ in 0..count {
            mapped.push(EncodeWorkerAssignment {
                item_index: mapped.len(),
                worker: Arc::clone(worker),
            });
        }
    }
    tracing::debug!(original = ?assignments.iter().map(|a| a.worker.url()).collect::<Vec<_>>(),
        contiguous = ?mapped.iter().map(|a| a.worker.url()).collect::<Vec<_>>(),
        "SGLang EPD uses contiguous IMAGE assignment; per-item cache affinity differs");
    Ok(mapped)
}

pub(crate) fn handoff(
    request_id: &str,
    assignments: &[EncodeWorkerAssignment],
) -> Result<sglang_proto::EncodeHandoff> {
    ensure!(
        !request_id.is_empty(),
        "SGLang encoder needs the final request ID"
    );
    let mapped = contiguous(assignments)?;
    let mut urls = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    for assignment in &mapped {
        if urls.last().is_none_or(|url| url != assignment.worker.url()) {
            urls.push(assignment.worker.url().to_string());
            counts.push(0);
        }
        let count = counts
            .last_mut()
            .ok_or_else(|| anyhow!("encoder count missing"))?;
        *count = count
            .checked_add(1)
            .ok_or_else(|| anyhow!("encoder count overflow"))?;
    }
    Ok(sglang_proto::EncodeHandoff {
        request_id: request_id.to_string(),
        encoder_urls: urls,
        num_items_assigned: counts,
    })
}
