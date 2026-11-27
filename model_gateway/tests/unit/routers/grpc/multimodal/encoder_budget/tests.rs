use std::sync::Arc;

use llm_multimodal::{
    types::ImageSize, ImageSource, MediaErrorCode, MediaFault, PreprocessedEncoderInputs,
    TransformError, VideoSamplingInfo,
};
use smg_worker::identity::basic::RuntimeType;

use super::*;
use crate::routers::common::snapshot::test_support::{grpc_worker, registry_with};

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
    assert!(
        check_item_feature_budget(Modality::Video, &[156_600, 192_000], &items, Some(192_000))
            .is_ok()
    );
    assert!(check_item_feature_budget(Modality::Video, &[156_600, 192_001], &items, None).is_ok());
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
    fn image_input_elements(&self, width: u32, height: u32) -> Option<usize> {
        Some(width as usize * height as usize)
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
/// rule refuses is not counted here.
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

/// A batch whose planned pixel bytes would not ride in one message is
/// refused on the header sizes alone, as the request's own size under
/// `media_too_large`, one byte over the cap; at the cap it is admitted.
/// The width is the wire's: four bytes under float32, one under uint8,
/// two when the plan does not know it; the batch is summed.
#[test]
fn a_batch_whose_pixels_would_not_fit_one_message_is_refused_before_preprocessing() {
    let one_over = |width: u32, height: u32, dtype: Option<EncoderDtype>| {
        check_image_wire_cap(
            &PixelCounted,
            &[header_only(width, height)],
            dtype,
            Some(WIRE_MESSAGE_CAP),
        )
        .unwrap_err()
    };
    let error = one_over(32_768, 32_768, Some(EncoderDtype::Float32));
    assert_eq!(
        error.to_string(),
        "the request's images would be 4294967296 bytes of pixel tensors on the wire, over \
             the 4294967295-byte message cap; send fewer or smaller images"
    );
    assert_eq!(error.code(), MediaErrorCode::TooLarge);
    assert_eq!(error.fault(), MediaFault::Client);
    assert!(matches!(
        one_over(65_536, 65_536, Some(EncoderDtype::Uint8)),
        MediaConnectorError::PixelsOverWireCap {
            bytes: 4_294_967_296,
            ..
        }
    ));
    assert!(matches!(
        one_over(46_341, 46_341, None),
        MediaConnectorError::PixelsOverWireCap {
            bytes: 4_294_976_562,
            ..
        }
    ));
    for (width, height, dtype) in [
        (32_767, 32_768, Some(EncoderDtype::Float32)),
        (65_535, 65_536, Some(EncoderDtype::Uint8)),
        (46_340, 46_340, None),
    ] {
        assert!(
            check_image_wire_cap(
                &PixelCounted,
                &[header_only(width, height)],
                dtype,
                Some(WIRE_MESSAGE_CAP),
            )
            .is_ok(),
            "{width}x{height} {dtype:?}"
        );
    }
    // The batch is summed: two images within the cap each, over it together.
    let pair = [header_only(32_767, 32_768), header_only(1, 32_768)];
    let cap = Some(WIRE_MESSAGE_CAP);
    assert!(check_image_wire_cap(&PixelCounted, &pair, Some(EncoderDtype::Float32), cap).is_err());
    assert!(check_image_wire_cap(&PixelCounted, &pair, Some(EncoderDtype::BFloat16), cap).is_ok());
    assert!(check_image_wire_cap(&PixelCounted, &[], Some(EncoderDtype::Float32), cap).is_ok());
    // No bound where the pixels travel beside the message: nothing is
    // refused here, the framing of the hop that carries them answers.
    assert!(check_image_wire_cap(&PixelCounted, &pair, Some(EncoderDtype::Float32), None).is_ok());
}
