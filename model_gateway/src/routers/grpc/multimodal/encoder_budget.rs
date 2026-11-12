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

use llm_multimodal::{ImageFrame, MediaItemInfo, Modality, VisionPreProcessor};

use crate::worker::{Worker, WorkerRegistry};

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
/// item's size to the engine, as before.
pub(crate) fn engine_encoder_budget(registry: &WorkerRegistry, model_id: &str) -> Option<usize> {
    registry
        .get_by_model(model_id)
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
/// the processor's rule refuses is left to the preprocessing to report, so
/// its own message stands.
pub(crate) fn check_image_size_feature_budget(
    processor: &dyn VisionPreProcessor,
    images: &[Arc<ImageFrame>],
    budget: Option<usize>,
) -> Result<(), ItemOverEncoderBudget> {
    let Some(budget) = budget else {
        return Ok(());
    };
    for (index, image) in images.iter().enumerate() {
        let size = image.size();
        let Ok(feature_tokens) = processor.calculate_num_tokens(size.width, size.height) else {
            continue;
        };
        if feature_tokens > budget {
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm_multimodal::{
        types::ImageSize, ImageSource, PreprocessedEncoderInputs, TransformError, VideoSamplingInfo,
    };

    use super::*;
    use crate::worker::{
        test_support::{grpc_worker, registry_with},
        RuntimeType,
    };

    const MODEL: &str = "vision-model";

    fn worker(url: &str, labels: &[(&str, &str)]) -> Arc<dyn Worker> {
        grpc_worker(url, MODEL, RuntimeType::Vllm, labels)
    }

    fn sampled(frames: usize) -> MediaItemInfo {
        MediaItemInfo {
            video_sampling: Some(VideoSamplingInfo {
                source_fps: 30.0,
                frame_indices: (0..frames).collect(),
            }),
        }
    }

    /// One token over the budget is refused, with the item's size, its
    /// sampled frames and the budget in the message; at the budget, and
    /// under it, the item is admitted; no budget admits everything.
    #[test]
    fn an_item_one_token_over_the_budget_is_refused_naming_both_numbers() {
        let items = [sampled(600), sampled(900)];
        let error =
            check_item_feature_budget(Modality::Video, &[156_600, 192_001], &items, Some(192_000))
                .unwrap_err();
        assert_eq!(
            error,
            ItemOverEncoderBudget {
                modality: Modality::Video,
                position: 2,
                count: 2,
                feature_tokens: 192_001,
                sampled_frames: Some(900),
                budget: 192_000,
            }
        );
        let message = error.to_string();
        assert!(
            message.contains("video item 2 of 2 is 192001 feature tokens (900 sampled frames)"),
            "{message}"
        );
        assert!(message.contains("192000-token encoder budget"), "{message}");
        assert!(message.contains("fewer frames"), "{message}");
        assert!(check_item_feature_budget(
            Modality::Video,
            &[156_600, 192_000],
            &items,
            Some(192_000)
        )
        .is_ok());
        assert!(
            check_item_feature_budget(Modality::Video, &[156_600, 192_001], &items, None).is_ok()
        );
        // An image names no frames and asks for a smaller image.
        let error = check_item_feature_budget(
            Modality::Image,
            &[577],
            &[MediaItemInfo::default()],
            Some(576),
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "image item 1 of 1 is 577 feature tokens, above the 576-token encoder budget the \
             engine can place for one item; send a smaller image"
        );
        // The refusal is found at any depth of an error chain.
        let wrapped = anyhow::Error::new(error).context("Failed to process the plan");
        assert!(ItemOverEncoderBudget::is(&wrapped));
        assert!(!ItemOverEncoderBudget::is(&anyhow::anyhow!(
            "something else"
        )));
    }

    /// A model's budget is the smallest its workers advertise; a worker
    /// without the label (or with a value that is no count) does not count,
    /// and a model without any has no budget here.
    #[test]
    fn a_models_budget_is_the_smallest_its_workers_advertise() {
        let registry = registry_with(vec![
            worker("http://w1:50051", &[(MM_ENCODER_BUDGET_LABEL, "200000")]),
            worker("http://w2:50051", &[(MM_ENCODER_BUDGET_LABEL, " 192000 ")]),
            worker("http://w3:50051", &[(MM_ENCODER_BUDGET_LABEL, "none")]),
            worker("http://w4:50051", &[]),
        ]);
        assert_eq!(engine_encoder_budget(&registry, MODEL), Some(192_000));
        assert_eq!(engine_encoder_budget(&registry, "other-model"), None);
        let unlabelled = registry_with(vec![
            worker("http://w5:50051", &[]),
            worker("http://w6:50051", &[(MM_ENCODER_BUDGET_LABEL, "0")]),
        ]);
        assert_eq!(engine_encoder_budget(&unlabelled, MODEL), None);
    }

    /// A processor whose count is the image's pixel count, which refuses a
    /// zero side as a resize rule would, and which must never be asked to
    /// preprocess.
    struct PixelCounted;

    impl VisionPreProcessor for PixelCounted {
        fn default_mean(&self) -> [f64; 3] {
            [0.0; 3]
        }
        fn default_std(&self) -> [f64; 3] {
            [1.0; 3]
        }
        fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
            if width == 0 || height == 0 {
                return Err(TransformError::ShapeError("no side to resize".to_string()));
            }
            Ok(width as usize * height as usize)
        }
        fn model_name(&self) -> &'static str {
            "pixel-counted"
        }
        fn preprocess(
            &self,
            _: &[&image::DynamicImage],
        ) -> Result<PreprocessedEncoderInputs, TransformError> {
            panic!("the size check must refuse before any preprocessing")
        }
    }

    /// A `width` x `height` frame whose bytes are empty: its header size is
    /// all there is, and a decode of it would fail, not count.
    fn header_only(width: u32, height: u32) -> Arc<ImageFrame> {
        Arc::new(ImageFrame::encoded(
            Default::default(),
            Default::default(),
            ImageSource::InlineBytes,
            format!("{width}x{height}"),
            ImageSize::new(width, height),
            None,
        ))
    }

    /// One token over the budget on the header's size alone is refused with
    /// the numbers and words of the check on the produced counts, before
    /// anything is decoded; at the budget, and under it, the image is
    /// admitted; no budget admits everything; a size the processor's own
    /// rule refuses is left to the preprocessing to report.
    #[test]
    fn an_image_over_the_budget_on_its_header_size_is_refused_before_preprocessing() {
        let images = [header_only(4, 4), header_only(64, 2)];
        let error = check_image_size_feature_budget(&PixelCounted, &images, Some(127)).unwrap_err();
        assert_eq!(
            error,
            ItemOverEncoderBudget {
                modality: Modality::Image,
                position: 2,
                count: 2,
                feature_tokens: 128,
                sampled_frames: None,
                budget: 127,
            }
        );
        assert_eq!(
            error.to_string(),
            "image item 2 of 2 is 128 feature tokens, above the 127-token encoder budget the \
             engine can place for one item; send a smaller image"
        );
        assert!(check_image_size_feature_budget(&PixelCounted, &images, Some(128)).is_ok());
        assert!(check_image_size_feature_budget(&PixelCounted, &images, None).is_ok());
        let refused_size = [header_only(0, 64)];
        assert!(check_image_size_feature_budget(&PixelCounted, &refused_size, Some(1)).is_ok());
    }
}
