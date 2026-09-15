//! Canonical multimodal request planning shared by protocol adapters, rendering,
//! and preprocessing.

use std::collections::HashMap;

use anyhow::{Context, Result};
use llm_multimodal::{MediaContentPart, MediaPartOrder, Modality, ModelMetadata, ToolResultOrder};
use llm_tokenizer::TokenizerTrait;
use openai_protocol::{
    chat::{ChatMessage, MessageContent},
    common::ContentPart,
    messages::{
        InputContent, InputContentBlock, InputMessage, Role, ToolResultContent,
        ToolResultContentBlock,
    },
};

use super::{config::MultimodalComponents, RegistryTokenizer};

/// Ordered media extracted from an API request.
///
/// This is the single hand-off between protocol-specific parsing and the shared
/// multimodal pipeline. Text remains in the message representation used by the
/// chat template; media is kept in rendered-placeholder order for fetching and
/// count validation.
#[derive(Debug, Clone, Default)]
pub(crate) struct MediaPlan {
    parts: Vec<MediaContentPart>,
    modalities: Vec<Modality>,
    counts: HashMap<Modality, usize>,
}

impl MediaPlan {
    pub(crate) fn new(parts: impl IntoIterator<Item = MediaContentPart>) -> Self {
        let mut plan = Self::default();
        for part in parts {
            let modality = match &part {
                MediaContentPart::ImageUrl { .. } | MediaContentPart::ImageData { .. } => {
                    Some(Modality::Image)
                }
                MediaContentPart::ImageEmbeds { .. } => Some(Modality::ImageEmbeds),
                MediaContentPart::AudioUrl { .. } | MediaContentPart::AudioData { .. } => {
                    Some(Modality::Audio)
                }
                MediaContentPart::VideoUrl { .. } | MediaContentPart::VideoData { .. } => {
                    Some(Modality::Video)
                }
                MediaContentPart::Text { .. } => None,
            };

            let Some(modality) = modality else {
                continue;
            };
            if !plan.modalities.contains(&modality) {
                plan.modalities.push(modality);
            }
            *plan.counts.entry(modality).or_default() += 1;
            plan.parts.push(part);
        }
        plan
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    pub(crate) fn modalities(&self) -> &[Modality] {
        &self.modalities
    }

    pub(crate) fn count(&self, modality: Modality) -> usize {
        self.counts.get(&modality).copied().unwrap_or_default()
    }

    pub(crate) fn into_parts(self) -> Vec<MediaContentPart> {
        self.parts
    }
}

/// Model-specific structural anchor strings keyed by modality.
///
/// String-format templates need the actual anchor string, while OpenAI-format
/// templates receive canonical `image` / `audio` / `video` parts. Keeping the
/// mapping typed prevents the former `image_placeholder` argument from being
/// accidentally reused for every modality.
#[derive(Debug, Clone, Default)]
pub(crate) struct PlaceholderTokens {
    tokens: HashMap<Modality, String>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MediaRenderingContract {
    pub media_part_order: MediaPartOrder,
    pub requires_structured_chat_content: bool,
    pub tool_result_order: ToolResultOrder,
}

impl Default for MediaRenderingContract {
    fn default() -> Self {
        Self {
            media_part_order: MediaPartOrder::MediaFirst,
            requires_structured_chat_content: false,
            tool_result_order: ToolResultOrder::Authored,
        }
    }
}

impl PlaceholderTokens {
    pub(crate) fn insert(&mut self, modality: Modality, token: String) {
        self.tokens.insert(modality, token);
    }

    pub(crate) fn get(&self, modality: Modality) -> Option<&str> {
        self.tokens.get(&modality).map(String::as_str)
    }
}

/// Validate a multimodal request against the model spec and resolve the
/// structural anchors for its active modalities in one config/spec lookup.
pub(crate) async fn prepare_placeholder_tokens(
    plan: &MediaPlan,
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    components: &MultimodalComponents,
    tokenizer_id: &str,
    tokenizer_source: &str,
) -> Result<PlaceholderTokens> {
    anyhow::ensure!(!plan.is_empty(), "multimodal media plan is empty");
    let model_config = components
        .config_registry
        .get_or_load(tokenizer_id, tokenizer_source)
        .await?;
    let registry_tokenizer = RegistryTokenizer(tokenizer);
    let metadata = ModelMetadata {
        model_id,
        tokenizer: &registry_tokenizer,
        config: &model_config.config,
    };
    let spec = components
        .lookup_model(&metadata)
        .with_context(|| format!("multimodal not supported for model: {model_id}"))?;
    let requested = plan
        .modalities()
        .iter()
        .map(|&modality| (modality, plan.count(modality)))
        .collect::<Vec<_>>();
    spec.validate_media_request_with_limits(
        &metadata,
        &requested,
        &components.modality_limit_overrides,
    )
    .map_err(|error| anyhow::anyhow!("invalid media request for model {}: {error}", spec.name()))?;
    let mut placeholders = PlaceholderTokens::default();
    for &modality in plan.modalities() {
        let token = spec
            .placeholder_token_for(&metadata, modality)
            .map_err(|error| {
                anyhow::anyhow!(
                    "model {} supports {modality} but its placeholder token could not be resolved: {error}",
                    spec.name()
                )
            })?;
        anyhow::ensure!(
            tokenizer.token_to_id(&token).is_some(),
            "{modality} placeholder token '{token}' is missing from the tokenizer vocabulary"
        );
        placeholders.insert(modality, token);
    }

    Ok(placeholders)
}

/// Resolve rendering decisions in the same config/spec lookup.
pub(crate) async fn resolve_media_rendering(
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    components: &MultimodalComponents,
    tokenizer_id: &str,
    tokenizer_source: &str,
) -> MediaRenderingContract {
    let model_config = match components
        .config_registry
        .get_or_load(tokenizer_id, tokenizer_source)
        .await
    {
        Ok(config) => config,
        Err(_) => return MediaRenderingContract::default(),
    };
    let registry_tokenizer = RegistryTokenizer(tokenizer);
    let metadata = ModelMetadata {
        model_id,
        tokenizer: &registry_tokenizer,
        config: &model_config.config,
    };
    components
        .lookup_model(&metadata)
        .map(|spec| MediaRenderingContract {
            media_part_order: spec.media_part_order(),
            requires_structured_chat_content: spec.requires_structured_chat_content(),
            tool_result_order: spec.tool_result_order(),
        })
        .unwrap_or_default()
}

fn count_chat_content(content: Option<&MessageContent>) -> usize {
    match content {
        Some(MessageContent::Parts(parts)) => parts
            .iter()
            .filter(|part| matches!(part, ContentPart::ImageUrl { .. }))
            .count(),
        Some(MessageContent::Text(_)) | None => 0,
    }
}

/// Count image markers the OpenAI-format Chat renderer will emit, including
/// roles that media detection intentionally does not fetch.
pub(crate) fn renderable_image_marker_count_chat(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .map(|message| match message {
            ChatMessage::System { content, .. }
            | ChatMessage::User { content, .. }
            | ChatMessage::Tool { content, .. }
            | ChatMessage::Root { content, .. }
            | ChatMessage::Developer { content, .. } => count_chat_content(Some(content)),
            ChatMessage::Assistant { content, .. } => count_chat_content(content.as_ref()),
            ChatMessage::Function { .. } => 0,
        })
        .sum()
}

fn count_message_content(content: &InputContent) -> usize {
    let InputContent::Blocks(blocks) = content else {
        return 0;
    };
    blocks
        .iter()
        .map(|block| match block {
            InputContentBlock::Image(_) => 1,
            InputContentBlock::ToolResult(result) => match &result.content {
                Some(ToolResultContent::Blocks(blocks)) => blocks
                    .iter()
                    .filter(|block| matches!(block, ToolResultContentBlock::Image(_)))
                    .count(),
                Some(ToolResultContent::String(_)) | None => 0,
            },
            _ => 0,
        })
        .sum()
}

/// Count image markers the Messages API conversion emits. Only user messages
/// are rendered with media parts by that protocol adapter.
pub(crate) fn renderable_image_marker_count_messages(messages: &[InputMessage]) -> usize {
    messages
        .iter()
        .filter(|message| message.role == Role::User)
        .map(|message| count_message_content(&message.content))
        .sum()
}

pub(crate) fn validate_marker_backing(marker_count: usize, plan: &MediaPlan) -> Result<()> {
    let planned = plan.count(Modality::Image);
    anyhow::ensure!(
        marker_count == planned,
        "renderable image marker count mismatch: renderer emits {marker_count}, media plan contains {planned} images"
    );
    Ok(())
}

/// Verify a rendered/tokenized multimodal prompt contains exactly one
/// structural anchor for every planned media item before any media is fetched
/// or preprocessed.
///
/// This catches stale/custom templates, adapter omissions, and literal internal
/// anchors in user text at the cheapest point in the pipeline.
pub(crate) fn validate_rendered_media_anchors(
    plan: &MediaPlan,
    placeholders: &PlaceholderTokens,
    tokenizer: &dyn TokenizerTrait,
    token_ids: &[u32],
) -> Result<()> {
    for &modality in plan.modalities() {
        let token = placeholders
            .get(modality)
            .ok_or_else(|| anyhow::anyhow!("missing resolved {modality} placeholder token"))?;
        let token_id = tokenizer.token_to_id(token).ok_or_else(|| {
            anyhow::anyhow!(
                "{modality} placeholder token '{token}' is missing from the tokenizer vocabulary"
            )
        })?;
        let expected = plan.count(modality);
        let actual = token_ids
            .iter()
            .filter(|&&candidate| candidate == token_id)
            .count();
        anyhow::ensure!(
            actual == expected,
            "rendered {modality} anchor count mismatch: expected {expected}, found {actual}; verify the chat template contract and escape literal media anchors in user text"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm_multimodal::PreProcessorConfig;
    use llm_tokenizer::mock::MockTokenizer;
    use openai_protocol::common::ImageUrl;
    use serde_json::json;

    use super::*;

    fn image_content(url: &str) -> MessageContent {
        MessageContent::Parts(vec![ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: url.to_string(),
                detail: None,
                max_long_side_pixel: None,
            },
        }])
    }

    #[tokio::test]
    async fn deepseek_v4_content_format_resolution_uses_structured_openai_blocks() {
        let config_registry = Arc::new(super::super::config::MultimodalConfigRegistry::new());
        config_registry.insert(
            "deepseek-tokenizer".to_string(),
            Arc::new(super::super::config::MultimodalModelConfig {
                config: json!({"model_type": "deepseek_v4", "vision_n_layers": 32}),
                preprocessor_config: PreProcessorConfig::default(),
                video_preprocessor_config: None,
            }),
        );
        let components = MultimodalComponents::new(config_registry, None).unwrap();
        let contract = resolve_media_rendering(
            "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
            &MockTokenizer::new(),
            &components,
            "deepseek-tokenizer",
            "unused-cached-source",
        )
        .await;
        assert_eq!(contract.media_part_order, MediaPartOrder::Authored);
        assert!(contract.requires_structured_chat_content);
        assert_eq!(contract.tool_result_order, ToolResultOrder::AssistantCall);
    }

    #[test]
    fn marker_backing_counts_every_renderable_chat_role() {
        let messages = vec![
            ChatMessage::System {
                content: image_content("system"),
                name: None,
                ext: Default::default(),
            },
            ChatMessage::User {
                content: image_content("user"),
                name: None,
                ext: Default::default(),
            },
            ChatMessage::Developer {
                content: image_content("developer"),
                name: None,
                ext: Default::default(),
            },
            ChatMessage::Tool {
                content: image_content("tool"),
                tool_call_id: "call-1".to_string(),
            },
            ChatMessage::Assistant {
                content: Some(image_content("assistant")),
                name: None,
                tool_calls: None,
                reasoning_content: None,
                ext: Default::default(),
            },
        ];
        assert_eq!(renderable_image_marker_count_chat(&messages), 5);
        let plan = super::super::detect::media_plan_chat(&messages, ToolResultOrder::Authored);
        assert_eq!(plan.count(Modality::Image), 4);
        let error = validate_marker_backing(5, &plan).unwrap_err().to_string();
        assert!(error.contains("renderer emits 5"), "{error}");
        assert!(error.contains("contains 4 images"), "{error}");
    }

    #[test]
    fn marker_backing_rejects_empty_plan_and_both_mismatch_directions() {
        let assistant_only = vec![ChatMessage::Assistant {
            content: Some(image_content("assistant")),
            name: None,
            tool_calls: None,
            reasoning_content: None,
            ext: Default::default(),
        }];
        let empty_plan =
            super::super::detect::media_plan_chat(&assistant_only, ToolResultOrder::Authored);
        assert!(empty_plan.is_empty());
        assert!(validate_marker_backing(
            renderable_image_marker_count_chat(&assistant_only),
            &empty_plan
        )
        .unwrap_err()
        .to_string()
        .contains("renderer emits 1, media plan contains 0 images"));

        let one_image = MediaPlan::new([MediaContentPart::ImageUrl {
            url: "planned".to_string(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        }]);
        validate_marker_backing(1, &one_image).unwrap();
        for marker_count in [0, 2] {
            let error = validate_marker_backing(marker_count, &one_image)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(&format!("renderer emits {marker_count}")),
                "{error}"
            );
            assert!(error.contains("contains 1 images"), "{error}");
        }
    }

    #[test]
    fn validation_rejects_source_less_image_markers() {
        let chat = serde_json::from_value::<Vec<ChatMessage>>(json!([{
            "role": "user",
            "content": [{"type": "image"}]
        }]));
        // Unknown Chat parts are preserved by the protocol and rejected at
        // preparation, before media planning or rendering.
        assert!(crate::routers::grpc::utils::validate_chat_content_parts(&chat.unwrap()).is_err());

        let messages = serde_json::from_value::<Vec<InputMessage>>(json!([{
            "role": "user",
            "content": [{"type": "image"}]
        }]));
        assert!(messages.is_err());
    }

    #[test]
    fn media_plan_preserves_media_order_and_counts() {
        let plan = MediaPlan::new([
            MediaContentPart::Text {
                text: "ignored".to_string(),
            },
            MediaContentPart::AudioUrl {
                url: "audio".to_string(),
                uuid: None,
            },
            MediaContentPart::ImageUrl {
                url: "image".to_string(),
                detail: None,
                uuid: None,
                max_long_side_pixel: None,
            },
            MediaContentPart::AudioUrl {
                url: "audio-2".to_string(),
                uuid: None,
            },
        ]);

        assert_eq!(plan.modalities(), &[Modality::Audio, Modality::Image]);
        assert_eq!(plan.count(Modality::Audio), 2);
        assert_eq!(plan.count(Modality::Image), 1);

        let parts = plan.into_parts();
        assert_eq!(parts.len(), 3);
        assert!(matches!(
            &parts[0],
            MediaContentPart::AudioUrl { url, .. } if url == "audio"
        ));
        assert!(matches!(
            &parts[1],
            MediaContentPart::ImageUrl { url, .. } if url == "image"
        ));
        assert!(matches!(
            &parts[2],
            MediaContentPart::AudioUrl { url, .. } if url == "audio-2"
        ));
    }

    #[test]
    fn rendered_anchor_validation_is_exact_per_modality() {
        let plan = MediaPlan::new([
            MediaContentPart::ImageUrl {
                url: "image".to_string(),
                detail: None,
                uuid: None,
                max_long_side_pixel: None,
            },
            MediaContentPart::AudioUrl {
                url: "audio".to_string(),
                uuid: None,
            },
        ]);
        let mut placeholders = PlaceholderTokens::default();
        placeholders.insert(Modality::Image, "<|im_start|>".to_string());
        placeholders.insert(Modality::Audio, "<|im_end|>".to_string());
        let tokenizer = MockTokenizer::new();

        validate_rendered_media_anchors(&plan, &placeholders, &tokenizer, &[1001, 7, 1002])
            .unwrap();

        let error =
            validate_rendered_media_anchors(&plan, &placeholders, &tokenizer, &[1001, 1001, 1002])
                .unwrap_err();
        assert!(error.to_string().contains("image anchor count mismatch"));
    }
}
