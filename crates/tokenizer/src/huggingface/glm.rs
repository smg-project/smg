//! Checkpoint-specific GLM behavior. A GLM MoE DSA checkpoint (`model_type`
//! `glm_moe_dsa` with the `GlmMoeDsaForCausalLM` architecture, which the
//! text-only GLM-5.2 and GLM-5.3 releases carry) opts out of media processing
//! while its active template renders every media part as the official
//! `<reminder>` text; serving aliases, vision labels and the vision
//! checkpoints (`glm5_next`, `glm4v_moe`) do not.

use std::path::Path;

use crate::{
    chat_template::{ChatTemplateContentFormat, ChatTemplateParams, ChatTemplateState},
    traits::MediaHandling,
};

/// Presence identifies the GLM MoE DSA checkpoint family. Media reminders
/// additionally require the active template to render them: the family's own
/// templates do, a replacement template may not.
pub(super) struct Checkpoint;

impl Checkpoint {
    /// Recognize the checkpoint family from both model type and architecture.
    pub(super) fn load(dir: &Path) -> Option<Self> {
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
        if config.get("model_type")?.as_str()? != "glm_moe_dsa"
            || !config
                .get("architectures")?
                .as_array()?
                .iter()
                .any(|arch| arch.as_str() == Some("GlmMoeDsaForCausalLM"))
        {
            return None;
        }
        Some(Self)
    }

    /// Opt out of preprocessing only when the active template preserves reminders.
    pub(super) fn media_handling(template: &ChatTemplateState) -> MediaHandling {
        if template.content_format() == ChatTemplateContentFormat::OpenAI
            && renders_media_reminders(template)
        {
            MediaHandling::TemplateText
        } else {
            MediaHandling::Preprocess
        }
    }
}

/// Compare actual renders with the official text substitution. No template
/// source matching and no probes on the request path. Reject media anchors,
/// dropped media, or lost surrounding text, even within the same model family.
///
/// The probes carry the part kinds a template is handed: the gateway rewrites
/// a request's media parts (`image_url`, `video_url`, `audio_url`,
/// `input_audio`) to the bare `image`, `video` and `audio` kinds before it
/// renders, so how a template treats the request spellings never decides.
fn renders_media_reminders(template: &ChatTemplateState) -> bool {
    for modality in ["image", "video", "audio"] {
        let visible_text = format!(
            "before<reminder>You are unable to process this {modality} because you don't have \
             multi-modal input ability. Try different methods.</reminder>after"
        );
        for role in ["user", "tool"] {
            let media = [serde_json::json!({"role": role, "content": [
                {"type": "text", "text": "before"}, {"type": modality},
                {"type": "text", "text": "after"}
            ]})];
            let text = [serde_json::json!({"role": role, "content": [
                {"type": "text", "text": visible_text}
            ]})];
            let actual = template.apply(&media, ChatTemplateParams::default());
            let expected = template.apply(&text, ChatTemplateParams::default());
            if !matches!((actual, expected), (Ok(actual), Ok(expected))
                if actual == expected && actual.contains(&visible_text))
            {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{traits::Tokenizer, CacheConfig, CachedTokenizer, HuggingFaceTokenizer};

    const TEMPLATE: &str = include_str!("../../tests/fixtures/glm53/chat_template.jinja");

    /// Load a minimal tokenizer with checkpoint metadata and the reference template.
    fn tokenizer(config: serde_json::Value) -> HuggingFaceTokenizer {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, r#"{"version":"1.0","model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}"#).unwrap();
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        std::fs::write(dir.path().join("chat_template.jinja"), TEMPLATE).unwrap();
        HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap()
    }

    /// Metadata, template replacement, and cache wrappers preserve the resolved policy.
    #[test]
    fn media_policy_requires_checkpoint_and_active_template() {
        use MediaHandling::{Preprocess, TemplateText};

        // Preserve the reminder in source, but render a media anchor instead.
        let anchors = TEMPLATE.replace(
            "{{- \"<reminder>You are unable to process this \" ~ media_type ~ \" because you don't have multi-modal input ability. Try different methods.</reminder>\" }}",
            "{{- '<|image|>' }}{# <reminder>You are unable to process this image because you don't have multi-modal input ability. Try different methods.</reminder> #}",
        );
        assert_ne!(anchors, TEMPLATE);
        for (config, expected) in [
            (
                serde_json::json!({"model_type": "glm_moe_dsa", "architectures": ["GlmMoeDsaForCausalLM"]}),
                TemplateText,
            ),
            // The vision checkpoints as the hub publishes them: GLM-5.3-Flash and GLM-4.5V.
            (
                serde_json::json!({"model_type": "glm5_next", "architectures": ["Glm5NextForConditionalGeneration"], "text_config": {"model_type": "glm5_next_text"}}),
                Preprocess,
            ),
            (
                serde_json::json!({"model_type": "glm4v_moe", "architectures": ["Glm4vMoeForConditionalGeneration"], "text_config": {"model_type": "glm4v_moe_text"}}),
                Preprocess,
            ),
            (
                serde_json::json!({"model_type": "glm5_next", "architectures": ["GlmMoeDsaForCausalLM"]}),
                Preprocess,
            ),
            (
                serde_json::json!({"model_type": "llama", "architectures": ["LlamaForCausalLM"], "_name_or_path": "glm-5.3"}),
                Preprocess,
            ),
            (serde_json::json!({"model_type": "glm_moe_dsa"}), Preprocess),
            (
                serde_json::json!({"model_type": "glm_moe_dsa", "architectures": ["OtherModel"]}),
                Preprocess,
            ),
            (serde_json::json!({}), Preprocess),
        ] {
            let mut tokenizer = tokenizer(config.clone());
            assert!(
                tokenizer
                    .renderer_capabilities()
                    .native_assistant_continuation
            );
            assert_eq!(
                tokenizer.renderer_capabilities().media_handling,
                expected,
                "{config}"
            );
            // Failed replacement keeps the current template and resolved policy.
            assert!(tokenizer.set_chat_template("{{".into()).is_err());
            assert_eq!(tokenizer.renderer_capabilities().media_handling, expected);
            // Neither a media-anchor template nor a plain template may bypass preprocessing.
            for replacement in [
                anchors.as_str(),
                "{% for m in messages %}{% for p in m.content %}{% if p.type == 'text' %}{{ p.text }}{% endif %}{% endfor %}{% endfor %}",
                "{% for m in messages %}{{ m.content }}{% endfor %}",
            ] {
                tokenizer.set_chat_template(replacement.into()).unwrap();
                assert_eq!(tokenizer.renderer_capabilities().media_handling, Preprocess);
            }
            tokenizer.set_chat_template(TEMPLATE.into()).unwrap();
            let cached =
                CachedTokenizer::new(std::sync::Arc::new(tokenizer), CacheConfig::default());
            assert_eq!(cached.renderer_capabilities().media_handling, expected);
            let nested = CachedTokenizer::new(std::sync::Arc::new(cached), CacheConfig::default());
            assert_eq!(nested.renderer_capabilities().media_handling, expected);
        }
    }

    /// The probes ask about the bare `image` / `video` / `audio` parts a template
    /// is handed, not about the request spellings the gateway rewrites.
    #[test]
    fn media_policy_follows_the_part_kinds_a_template_is_handed() {
        use MediaHandling::{Preprocess, TemplateText};

        const KINDS: &str =
            "['image', 'image_url', 'video', 'video_url', 'audio', 'audio_url', 'input_audio']";
        assert!(TEMPLATE.contains(KINDS));
        let mut tokenizer = tokenizer(
            serde_json::json!({"model_type": "glm_moe_dsa", "architectures": ["GlmMoeDsaForCausalLM"]}),
        );
        // A template that renders only the bare kinds renders every request.
        tokenizer
            .set_chat_template(TEMPLATE.replace(KINDS, "['image', 'video', 'audio']"))
            .unwrap();
        assert_eq!(
            tokenizer.renderer_capabilities().media_handling,
            TemplateText
        );
        // One that renders only the request spellings is never handed them.
        tokenizer
            .set_chat_template(TEMPLATE.replace(
                KINDS,
                "['image_url', 'video_url', 'audio_url', 'input_audio']",
            ))
            .unwrap();
        assert_eq!(tokenizer.renderer_capabilities().media_handling, Preprocess);
    }
}
