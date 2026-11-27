//! The engine's encoder budget: the most feature tokens it can place for one
//! media item, advertised by its workers as the `mm_encoder_budget` label and
//! held to by the media pipeline wherever it runs.
//!
//! An engine sizes its encoder cache for one item of the largest feature size
//! its own processor can produce, and never schedules an item above that
//! budget: the request sits in its running set with no output until the
//! client gives up, and the gateway's liveness check reads the silence as a
//! wedged worker. The engine's front end refuses such an item, but never sees
//! the inputs this pipeline precomputes, so the pipeline refuses them itself,
//! on the per-item feature token count the placeholder expansion uses
//! (nothing is counted twice), with the item's size and the budget in the
//! same units. An image is held to the budget twice: on the count its
//! header's size alone gives, before any of its batch is decoded or any
//! tensor built (a family whose crops grow with the plan writes hundreds of
//! megabytes for an item the engine then never places), and on the count its
//! preprocessing produced; a video or audio item, whose count comes from the
//! sampled frames or the decoded length, on the produced count alone. The
//! worker-side pipeline takes the budget from its launcher;
//! the gateway's own placement takes the smallest a model's workers
//! advertise, as it does the per-request item limits ([`super::item_limits`]).

use std::{fmt, sync::Arc};

use llm_multimodal::{
    EncoderDtype, ImageFrame, MediaConnectorError, MediaItemInfo, Modality, TransformError,
    VisionPreProcessor,
};
use smg_worker::identity::basic::Worker;

use super::capability::media_facts;
use crate::routers::common::snapshot::WorkerRegistry;

/// Worker label carrying the engine's encoder budget in feature tokens: the
/// vLLM servicer's `GetServerInfo.mm_encoder_budget` (the "budget of N
/// tokens" of the engine's encoder-cache start-up line).
const MM_ENCODER_BUDGET_LABEL: &str = "mm_encoder_budget";

/// The error code of the refusal on the gateway's own placement.
pub(crate) const ITEM_OVER_ENCODER_BUDGET_CODE: &str = "multimodal_item_too_large";

/// The budget `worker` advertises; `None` without the label, or with one that
/// is not a positive count.
fn worker_encoder_budget(worker: &dyn Worker) -> Option<usize> {
    worker
        .metadata()
        .spec
        .labels
        .get(MM_ENCODER_BUDGET_LABEL)
        .and_then(|label| label.trim().parse::<usize>().ok())
        .filter(|&budget| budget > 0)
}

/// The smallest encoder budget any registered worker of `model_id`
/// advertises (healthy or not: a health flap must not move a model's bound
/// request to request); `None` when none advertises one, which leaves an
/// item's size to the engine, as before. The model's membership snapshot
/// keeps the answer ([`media_facts`]).
pub(crate) fn engine_encoder_budget(registry: &WorkerRegistry, model_id: &str) -> Option<usize> {
    media_facts(registry, model_id, |facts| facts.encoder_budget)
}

/// The rule of [`engine_encoder_budget`] over the members of one snapshot.
pub(super) fn encoder_budget_over(workers: &[Arc<dyn Worker>]) -> Option<usize> {
    workers
        .iter()
        .filter_map(|worker| worker_encoder_budget(worker.as_ref()))
        .min()
}

/// A media item the engine can never place: more feature tokens than its
/// encoder budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ItemOverEncoderBudget {
    pub(crate) modality: Modality,
    /// The item's place among the request's items of its modality, from 1.
    pub(crate) position: usize,
    pub(crate) count: usize,
    pub(crate) feature_tokens: usize,
    /// The frames the clip was sampled to, when the decoder reported them.
    pub(crate) sampled_frames: Option<usize>,
    pub(crate) budget: usize,
}

impl ItemOverEncoderBudget {
    /// Whether `error` is this refusal, at any depth of its chain.
    pub(crate) fn is(error: &anyhow::Error) -> bool {
        error
            .chain()
            .any(|cause| cause.downcast_ref::<Self>().is_some())
    }
}

impl fmt::Display for ItemOverEncoderBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} item {} of {} is {} feature tokens",
            self.modality, self.position, self.count, self.feature_tokens
        )?;
        if let Some(frames) = self.sampled_frames {
            write!(f, " ({frames} sampled frames)")?;
        }
        write!(
            f,
            ", above the {}-token encoder budget the engine can place for one item; {}",
            self.budget,
            match self.modality {
                Modality::Video => "sample fewer frames (a lower fps) or send a shorter clip",
                Modality::Audio => "send a shorter clip",
                _ => "send a smaller image",
            }
        )
    }
}

impl std::error::Error for ItemOverEncoderBudget {}

/// Hold every item of a preprocessed modality batch to the engine's budget,
/// on the feature token counts the processor produced (`items` are the
/// decoder-side facts of the same items, in the same order); the first item
/// above the budget is the refusal. `None` is no budget.
pub(crate) fn check_item_feature_budget(
    modality: Modality,
    feature_token_counts: &[usize],
    items: &[MediaItemInfo],
    budget: Option<usize>,
) -> Result<(), ItemOverEncoderBudget> {
    let Some(budget) = budget else {
        return Ok(());
    };
    match feature_token_counts
        .iter()
        .position(|&tokens| tokens > budget)
    {
        None => Ok(()),
        Some(index) => Err(ItemOverEncoderBudget {
            modality,
            position: index + 1,
            count: feature_token_counts.len(),
            feature_tokens: feature_token_counts[index],
            sampled_frames: items
                .get(index)
                .and_then(|item| item.video_sampling.as_ref())
                .map(|sampling| sampling.frame_indices.len()),
            budget,
        }),
    }
}

/// Hold every image of a batch to the engine's budget before any of the
/// batch is decoded or any tensor built, on the count its size alone gives:
/// the size the header promised (`ImageFrame::size`, the dimensions the
/// decode produces) through the processor's own size-to-count rule
/// (`calculate_num_tokens`, the rule its preprocessing applies to the same
/// size); the first image above the budget is the refusal, with the numbers
/// and words of [`check_item_feature_budget`]. `None` is no budget. A size
/// the processor's rule refuses is not counted: the first one, in request
/// order, is handed back, so the caller refuses it in the preprocessing's
/// own words and the rule is asked once for an image. A header with a side
/// of zero is not handed back: no image has one, and it is left to the
/// decode and the preprocessing.
pub(crate) fn check_image_size_feature_budget(
    processor: &dyn VisionPreProcessor,
    images: &[Arc<ImageFrame>],
    budget: Option<usize>,
) -> Result<Option<TransformError>, ItemOverEncoderBudget> {
    let mut refused = None;
    for (index, image) in images.iter().enumerate() {
        let size = image.size();
        match processor.calculate_num_tokens(size.width, size.height) {
            Ok(feature_tokens) => {
                if let Some(budget) = budget.filter(|budget| feature_tokens > *budget) {
                    return Err(ItemOverEncoderBudget {
                        modality: Modality::Image,
                        position: index + 1,
                        count: images.len(),
                        feature_tokens,
                        sampled_frames: None,
                        budget,
                    });
                }
            }
            Err(error) if refused.is_none() && size.width != 0 && size.height != 0 => {
                refused = Some(error);
            }
            Err(_) => {}
        }
    }
    Ok(refused)
}

/// The longest message the gRPC framing carries: its length prefix is four
/// bytes, and tonic refuses a longer message on whichever side encodes it
/// (`Cannot return body with more than 4GB`), the request's own size.
pub(crate) const WIRE_MESSAGE_CAP: usize = u32::MAX as usize;

/// Hold a batch of images to the wire's own cap before any of the batch is
/// decoded or any tensor built, where the caller resolved one (`cap`:
/// [`WIRE_MESSAGE_CAP`] where the pixels ride inline in the gateway's own
/// gRPC message, `None` where they travel beside it or the caller cannot
/// tell, and then nothing is refused here: the framing itself answers a
/// message it cannot carry): the pixel tensors they would put in one
/// Generate request, planned from the size each header promised through the
/// processor's own size-to-elements rule (`image_input_elements`) at the
/// width the wire writes an element in (`wire_dtype`; when the plan does not
/// know it yet, the processor's own bytes for one that emits pixel bytes and
/// the narrowest float width the engines take, two bytes, otherwise, so the
/// bound refuses only what no serialization could carry), must stay within
/// the cap. The message's other fields are not counted: a
/// request the pixels alone leave within the cap by a hair is refused by the
/// framing itself, as the request's own size. An image the processor plans
/// no element count for is not counted either.
pub(crate) fn check_image_wire_cap(
    processor: &dyn VisionPreProcessor,
    images: &[Arc<ImageFrame>],
    wire_dtype: Option<EncoderDtype>,
    cap: Option<usize>,
) -> Result<(), MediaConnectorError> {
    let Some(cap) = cap else {
        return Ok(());
    };
    let element_size = wire_dtype.map_or_else(
        || if processor.emits_pixel_bytes() { 1 } else { 2 },
        EncoderDtype::element_size,
    );
    let bytes = images
        .iter()
        .filter_map(|image| {
            let size = image.size();
            processor.image_input_elements(size.width, size.height)
        })
        .fold(0usize, |total, elements| {
            total.saturating_add(elements.saturating_mul(element_size))
        });
    if bytes > cap {
        return Err(MediaConnectorError::PixelsOverWireCap { bytes, limit: cap });
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/multimodal/encoder_budget/tests.rs"]
mod tests;
