//! DeepSeek V4 rules calibrated against the official Chat API.
use openai_protocol::{
    chat::ChatCompletionRequest, profile::ProviderProfile, validated::Normalizable,
};
use serde_json::{json, Value};
use validator::Validate;

#[expect(clippy::expect_used, reason = "test helper")]
fn request(extra: Value) -> ChatCompletionRequest {
    let mut body =
        json!({"model": "deepseek-flash", "messages": [{"role": "user", "content": "hello"}]});
    body.as_object_mut()
        .expect("object")
        .extend(extra.as_object().expect("object").clone());
    let mut req: ChatCompletionRequest = serde_json::from_value(body).expect("request");
    req.normalize();
    req
}

#[test]
fn deepseek_profile_is_limited_to_the_verified_v4_family() {
    for model in [
        "deepseek-flash",
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "/models/DeepSeek-V4.1-Flash",
        "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    ] {
        assert_ne!(
            ProviderProfile::for_model(model),
            ProviderProfile::Generic,
            "{model}"
        );
    }
    for model in [
        "deepseek-chat",
        "deepseek-reasoner",
        "deepseek-ai/DeepSeek-V3.2",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-v40-pro",
        "deepseek-v4.10-flash",
        "deepseek-v4.2-flash",
        "deepseek-flashlight",
        "my-deepseek-v4-pro",
        "opaque-alias",
    ] {
        assert_eq!(
            ProviderProfile::for_model(model),
            ProviderProfile::Generic,
            "{model}"
        );
    }
}

#[test]
fn deepseek_effort_aliases_preserve_thinking_and_normalize_levels() {
    for (input, canonical) in [
        ("minimal", "low"),
        ("medium", "high"),
        ("xhigh", "high"),
        ("none", "none"),
        ("low", "low"),
        ("high", "high"),
        ("max", "max"),
    ] {
        let req = request(json!({"reasoning_effort": input}));
        assert!(req.validate().is_ok());
        assert_eq!(req.effective_reasoning_effort(), Some(canonical));
        let req = request(json!({"thinking": {"effort": input}}));
        assert!(req.validate().is_ok());
        assert_eq!(req.effective_reasoning_effort(), Some(canonical));
    }
    let req = request(json!({"model": "gpt-4o", "reasoning_effort": "minimal"}));
    assert_eq!(req.effective_reasoning_effort(), Some("minimal"));
}

#[test]
fn deepseek_rejects_unsupported_thinking_and_effort() {
    for extra in [
        json!({"thinking": {"type": "adaptive"}}),
        json!({"reasoning_effort": "turbo"}),
        json!({"thinking": {"effort": "turbo"}, "reasoning_effort": "high"}),
    ] {
        assert!(request(extra.clone()).validate().is_err(), "{extra}");
    }
}

/// A forced tool choice (a named function, `required`) validates in either
/// thinking mode: a self-hosted engine serves the call after the model's
/// thought, and the vendor's own API refuses the combination by itself where
/// it does. `auto` and `none` validate as before.
#[test]
fn deepseek_forced_tools_validate_in_either_thinking_mode() {
    for choice in [
        json!("required"),
        json!({"type": "function", "function": {"name": "weather"}}),
    ] {
        for mode in [
            json!({}),
            json!({"thinking": {"type": "enabled"}}),
            json!({"thinking": {"type": "disabled"}}),
            json!({"reasoning_effort": "none"}),
            json!({"reasoning_effort": "minimal"}),
            json!({"chat_template_kwargs": {"thinking": false}}),
            json!({"chat_template_kwargs": {"enable_thinking": false}}),
            json!({"chat_template_kwargs": {"reasoning_effort": "none"}}),
        ] {
            let mut extra = json!({"tools": [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object", "properties": {}}}}], "tool_choice": choice});
            extra
                .as_object_mut()
                .unwrap()
                .extend(mode.as_object().unwrap().clone());
            assert!(request(extra.clone()).validate().is_ok(), "{extra}");
        }
    }
    for choice in ["auto", "none"] {
        assert!(request(json!({"tool_choice": choice})).validate().is_ok());
    }
}

#[test]
fn deepseek_projects_one_thinking_decision_for_validation_and_rendering() {
    for model in ["deepseek-v4-pro", "deepseek-ai/DeepSeek-V4.1-Flash"] {
        for (extra, enabled) in [
            (json!({}), true),
            (json!({"reasoning_effort":"minimal"}), true),
            (json!({"reasoning_effort":"low"}), true),
            (json!({"reasoning_effort":"none"}), false),
            (
                json!({"thinking":{"type":"disabled"},"reasoning_effort":"high"}),
                false,
            ),
            (
                json!({"thinking":{"type":"enabled"},"reasoning_effort":"none"}),
                true,
            ),
            (
                json!({"chat_template_kwargs":{"enable_thinking":false},"reasoning_effort":"high"}),
                false,
            ),
            (
                json!({"chat_template_kwargs":{"thinking":true},"thinking":{"type":"disabled"}}),
                true,
            ),
            (
                json!({"chat_template_kwargs":{"reasoning_effort":"none"},"thinking":{"type":"enabled"}}),
                false,
            ),
        ] {
            let mut extra = extra;
            extra["model"] = json!(model);
            let mut req = request(extra.clone());
            assert_eq!(
                req.chat_template_kwargs
                    .as_ref()
                    .and_then(|k| k.get("thinking")),
                Some(&json!(enabled)),
                "{extra}"
            );
            let once = serde_json::to_value(&req).unwrap();
            req.normalize();
            assert_eq!(
                serde_json::to_value(&req).unwrap(),
                once,
                "normalization must be idempotent"
            );
        }
    }
}

/// `allowed_tools` in `required` mode is a forced choice too, and validates in
/// either thinking mode like the others.
#[test]
fn deepseek_required_allowed_tools_validate_in_either_thinking_mode() {
    for enabled in [true, false] {
        let req = request(json!({
            "thinking":{"type": if enabled {"enabled"} else {"disabled"}},
            "tools":[{"type":"function","function":{"name":"weather"}}],
            "tool_choice":{"type":"allowed_tools","mode":"required","tools":[{"type":"function","name":"weather"}]}
        }));
        assert!(req.validate().is_ok(), "thinking enabled: {enabled}");
    }
}

#[test]
fn deepseek_rejects_ambiguous_template_thinking_overrides() {
    for kwargs in [
        json!({"thinking":"bad","enable_thinking":false}),
        json!({"enable_thinking":0}),
        json!({"thinking":true,"enable_thinking":false}),
    ] {
        let req = request(
            json!({"model":"deepseek-v4-pro","reasoning_effort":"high","chat_template_kwargs":kwargs}),
        );
        assert!(req.validate().is_err());
    }
    let req = request(
        json!({"thinking":{"type":"disabled"},"chat_template_kwargs":{"thinking":null,"enable_thinking":null,"reasoning_effort":42}}),
    );
    assert!(req.validate().is_ok());
    assert_eq!(
        req.chat_template_kwargs.as_ref().unwrap()["thinking"],
        false
    );
}

#[test]
fn deepseek_v41_preserves_native_medium_and_xhigh_for_both_effort_fields() {
    for model in [
        "deepseek-v4.1-flash",
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "/models/DEEPSEEK-V4.1-FLASH",
        "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    ] {
        for effort in ["medium", "xhigh"] {
            for extra in [
                json!({"reasoning_effort": effort}),
                json!({"thinking": {"effort": effort}}),
            ] {
                let mut extra = extra;
                extra["model"] = json!(model);
                let req = request(extra);
                assert!(req.validate().is_ok(), "{model} {effort}");
                assert_eq!(req.effective_reasoning_effort(), Some(effort), "{model}");
            }
        }
    }
}

#[test]
fn deepseek_v41_reads_a_template_kwarg_medium_as_a_level_of_its_own() {
    // A `medium` template kwarg switches thinking on under V4.1 like its
    // other levels; under V4 it stays an alias with no opinion on the mode,
    // so the typed toggle decides.
    for (model, enabled) in [
        ("deepseek-ai/DeepSeek-V4.1-Flash", true),
        ("deepseek-v4-pro", false),
    ] {
        let req = request(json!({
            "model": model,
            "thinking": {"type": "disabled"},
            "chat_template_kwargs": {"reasoning_effort": "medium"},
        }));
        assert!(req.validate().is_ok(), "{model}");
        assert_eq!(
            req.chat_template_kwargs
                .as_ref()
                .and_then(|kwargs| kwargs.get("thinking")),
            Some(&json!(enabled)),
            "{model}"
        );
    }
}

#[test]
fn deepseek_v41_forwards_the_vendors_reading_of_medium_to_an_upstream_renderer() {
    // The gateway's own renderer takes V4.1's `medium` as the template's
    // level; an upstream that renders itself gets the vendor's alias, `high`,
    // in both top-level spellings.
    let mut req = request(json!({
        "model": "deepseek-v4.1-flash",
        "reasoning_effort": "medium",
        "thinking": {"effort": "medium"},
    }));
    assert_eq!(req.effective_reasoning_effort(), Some("medium"));
    ProviderProfile::DeepSeek.normalize_chat_for_upstream(&mut req);
    assert_eq!(req.reasoning_effort.as_deref(), Some("high"));
    assert_eq!(
        req.thinking
            .as_ref()
            .and_then(|thinking| thinking.effort.as_deref()),
        Some("high")
    );

    // `xhigh` and a template kwarg go out as sent.
    let mut req = request(json!({
        "model": "deepseek-v4.1-flash",
        "reasoning_effort": "xhigh",
        "chat_template_kwargs": {"reasoning_effort": "medium"},
    }));
    ProviderProfile::DeepSeek.normalize_chat_for_upstream(&mut req);
    assert_eq!(req.reasoning_effort.as_deref(), Some("xhigh"));
    assert_eq!(
        req.chat_template_kwargs
            .as_ref()
            .and_then(|kwargs| kwargs.get("reasoning_effort")),
        Some(&json!("medium"))
    );

    // V4 aliased `medium` at normalization already; nothing more happens.
    let mut req = request(json!({"model": "deepseek-v4-pro", "reasoning_effort": "medium"}));
    assert_eq!(req.reasoning_effort.as_deref(), Some("high"));
    ProviderProfile::DeepSeek.normalize_chat_for_upstream(&mut req);
    assert_eq!(req.reasoning_effort.as_deref(), Some("high"));

    // Another vendor's profile forwards the field as sent.
    let mut req = request(json!({"model": "gpt-4o", "reasoning_effort": "medium"}));
    ProviderProfile::for_model(&req.model).normalize_chat_for_upstream(&mut req);
    assert_eq!(req.reasoning_effort.as_deref(), Some("medium"));
}

#[test]
fn deepseek_v41_accepts_native_integer_budgets() {
    for budget in [1, 42, 75, 100] {
        for mut body in [
            json!({"reasoning_effort": budget}),
            json!({"reasoning_effort": budget.to_string()}),
            // ThinkingConfig.effort is a string; only the top-level field
            // has the JSON integer-to-string compatibility deserializer.
            json!({"thinking": {"effort": budget.to_string()}}),
        ] {
            body["model"] = json!("deepseek-ai/DeepSeek-V4.1-Flash");
            let req = request(body.clone());
            assert!(req.validate().is_ok(), "{body}");
            assert_eq!(
                req.effective_reasoning_effort(),
                Some(budget.to_string().as_str())
            );
        }
    }
}

#[test]
fn deepseek_rejects_invalid_or_non_v41_budgets() {
    for model in ["deepseek-v4.1-flash", "deepseek-flash", "deepseek-v4-pro"] {
        for budget in [
            "0",
            "101",
            "-1",
            "+42",
            " 42 ",
            "42.0",
            "٤٢",
            "999999999999999999999999999",
        ] {
            for extra in [
                json!({"reasoning_effort":budget}),
                json!({"reasoning_effort":"high","thinking":{"effort":budget}}),
            ] {
                let mut extra = extra;
                extra["model"] = json!(model);
                assert!(request(extra.clone()).validate().is_err(), "{extra}");
            }
        }
    }
    for model in ["deepseek-flash", "deepseek-v4-pro"] {
        assert!(request(json!({"model":model,"reasoning_effort":42}))
            .validate()
            .is_err());
    }
}

#[test]
fn deepseek_v41_keeps_its_reading_under_a_registered_alias() {
    // An opaque alias of a V4.1 served model is judged as that model: the
    // profile reads V4.1 from the name the alias resolves to, not from the
    // name as sent, so a request normalizes and validates the same way under
    // either name. Names unique to this test: the alias table is process-wide.
    const ALIAS: &str = "chat-large-under-an-opaque-alias";
    ProviderProfile::register_model_aliases([(ALIAS, "deepseek-ai/DeepSeek-V4.1-Flash")]);
    assert_eq!(ProviderProfile::for_model(ALIAS), ProviderProfile::DeepSeek);

    // `medium` and `xhigh` stay what the client asked, in both spellings;
    // the vendor's `minimal` alias still resolves to `low`.
    for effort in ["medium", "xhigh"] {
        for extra in [
            json!({"model": ALIAS, "reasoning_effort": effort}),
            json!({"model": ALIAS, "thinking": {"effort": effort}}),
        ] {
            let req = request(extra);
            assert!(req.validate().is_ok(), "{effort}");
            assert_eq!(req.effective_reasoning_effort(), Some(effort));
        }
    }
    let req = request(json!({"model": ALIAS, "reasoning_effort": "minimal"}));
    assert_eq!(req.effective_reasoning_effort(), Some("low"));

    // A template kwarg `medium` is a level of V4.1's own table: thinking on,
    // ahead of the typed toggle.
    let req = request(json!({
        "model": ALIAS,
        "thinking": {"type": "disabled"},
        "chat_template_kwargs": {"reasoning_effort": "medium"},
    }));
    assert!(req.validate().is_ok());
    assert_eq!(
        req.chat_template_kwargs
            .as_ref()
            .and_then(|kwargs| kwargs.get("thinking")),
        Some(&json!(true))
    );

    // An integer budget is V4.1's extension and validates under the alias.
    let req = request(json!({"model": ALIAS, "reasoning_effort": 62}));
    assert!(req.validate().is_ok());
    assert_eq!(req.effective_reasoning_effort(), Some("62"));

    // An upstream that renders the prompt itself still gets the vendor's
    // reading of `medium`.
    let mut req = request(json!({"model": ALIAS, "reasoning_effort": "medium"}));
    ProviderProfile::for_model(&req.model).normalize_chat_for_upstream(&mut req);
    assert_eq!(req.reasoning_effort.as_deref(), Some("high"));
}

#[test]
fn deepseek_v41_is_read_from_the_names_that_carry_the_contract() {
    // A served model with an opaque name takes V4.1 from the V4.1-named
    // alias it is served under, whichever name the request carries.
    const SERVED: &str = "served-model-under-a-v41-name";
    const V41_NAME: &str = "alias-test/deepseek-v4.1-flash";
    ProviderProfile::register_model_aliases([(V41_NAME, SERVED)]);
    for model in [SERVED, V41_NAME] {
        let req = request(json!({"model": model, "reasoning_effort": "medium"}));
        assert_eq!(req.effective_reasoning_effort(), Some("medium"), "{model}");
    }

    // A vendor-named alias is read as its own name, as a request naming it
    // always was: under a V4 name the vendor alias of `medium` applies, under
    // a V4.1 name the level is kept, whatever the served model's own name.
    const V4_NAME_OF_A_V41_MODEL: &str = "alias-test-v4/deepseek-v4-flash";
    const V41_NAME_OF_A_V4_MODEL: &str = "alias-test-v41/deepseek-v4.1-flash";
    ProviderProfile::register_model_aliases([
        (V4_NAME_OF_A_V41_MODEL, "deepseek-v4.1-flash"),
        (V41_NAME_OF_A_V4_MODEL, "deepseek-v4-pro"),
    ]);
    let req = request(json!({"model": V4_NAME_OF_A_V41_MODEL, "reasoning_effort": "medium"}));
    assert_eq!(req.effective_reasoning_effort(), Some("high"));
    let req = request(json!({"model": V41_NAME_OF_A_V4_MODEL, "reasoning_effort": "medium"}));
    assert_eq!(req.effective_reasoning_effort(), Some("medium"));

    // One vendor's name for another vendor's model is read as that model: a
    // Kimi-named alias of the V4.1 model takes its profile and keeps `medium`.
    const KIMI_NAME_OF_A_V41_MODEL: &str = "cross-vendor/kimi-k2";
    ProviderProfile::register_model_aliases([(KIMI_NAME_OF_A_V41_MODEL, "deepseek-v4.1-flash")]);
    assert_eq!(
        ProviderProfile::for_model(KIMI_NAME_OF_A_V41_MODEL),
        ProviderProfile::DeepSeek
    );
    let req = request(json!({"model": KIMI_NAME_OF_A_V41_MODEL, "reasoning_effort": "medium"}));
    assert_eq!(req.effective_reasoning_effort(), Some("medium"));
}

#[test]
fn an_opaque_model_under_the_names_of_two_versions_keeps_the_v4_reading() {
    // One checkpoint served under a V4 name and a V4.1 name is an ambiguous
    // configuration: each public name is read as itself, and the opaque name
    // keeps the reading it had before any alias (V4: the vendor alias of
    // `medium`, no integer budgets) instead of taking V4.1 from one of them.
    const SERVED: &str = "served-model-under-two-versions";
    const V4_NAME: &str = "two-versions/deepseek-v4-flash";
    const V41_NAME: &str = "two-versions/deepseek-v4.1-flash";
    ProviderProfile::register_model_aliases([(V4_NAME, SERVED), (V41_NAME, SERVED)]);
    for (model, effort) in [(SERVED, "high"), (V4_NAME, "high"), (V41_NAME, "medium")] {
        let req = request(json!({"model": model, "reasoning_effort": "medium"}));
        assert_eq!(ProviderProfile::for_model(model), ProviderProfile::DeepSeek);
        assert_eq!(req.effective_reasoning_effort(), Some(effort), "{model}");
    }
    assert!(request(json!({"model": SERVED, "reasoning_effort": 62}))
        .validate()
        .is_err());
    assert!(request(json!({"model": V41_NAME, "reasoning_effort": 62}))
        .validate()
        .is_ok());
}
