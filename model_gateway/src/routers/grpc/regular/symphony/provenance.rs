//! The source of the assigned table, retained before its name loses model-specific choices.

use symphony::formats::Table;

/// Which construction produced a table; this carries no request strings or allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "diagnostic-capture",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    feature = "diagnostic-capture",
    serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)
)]
pub(super) enum Source {
    Named {
        #[cfg_attr(feature = "diagnostic-capture", serde(with = "table_name"))]
        table: Table,
    },
    // Empty struct variants make Serde enforce unknown-field rejection too.
    Registry {},
    Template {},
}

#[cfg(feature = "diagnostic-capture")]
mod table_name {
    use serde::{Deserialize, Deserializer, Serializer};
    use symphony::formats::Table;

    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "Serde with-module callbacks receive a reference to the serialized field"
    )]
    pub(super) fn serialize<S: Serializer>(
        table: &Table,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(table.name())
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Table, D::Error> {
        let name = String::deserialize(deserializer)?;
        Table::named(&name)
            .filter(|table| table.name() == name)
            .ok_or_else(|| serde::de::Error::custom("unknown or noncanonical Symphony table"))
    }
}

/// The resolved selector and request variants needed to rebuild an assigned Symphony format.
///
/// This is only a format recipe. Its caller must bind it to the exact recorded SMG revision
/// and separately retain prompt/inputs, tools, call IDs and reader policies. The selector model
/// is retained at assignment time; a canonical routing ID must never replace it. This does
/// not authenticate the model string or bind it to a deployment or request.
/// It neither attests capture completeness nor reproduces the decoder or API adapter.
#[cfg(feature = "diagnostic-capture")]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FormatRecipe {
    pub(super) selector_model: std::sync::Arc<str>,
    pub(super) source: Source,
    pub(super) no_calls: bool,
    pub(super) no_reasoning: bool,
}

#[cfg(feature = "diagnostic-capture")]
impl FormatRecipe {
    /// The original model string used to select the format, before canonical routing.
    pub fn selector_model(&self) -> &str {
        &self.selector_model
    }

    /// Rebuild using the bound selector model at the recorded source revision, offline.
    ///
    /// Registry selection preserves model-specific openers and text after reasoning. Named
    /// selection deliberately ignores the model's registry entry. Compiled templates need a
    /// fuller recipe and return `Unsupported`; an unknown registry model returns `InvalidData`.
    pub fn resolve(&self) -> std::io::Result<std::sync::Arc<symphony::Format>> {
        let format = match self.source {
            Source::Named { table } => table.shared(),
            Source::Registry {} => symphony::formats::for_model_shared(self.selector_model())
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "recorded model has no Symphony table",
                    )
                })?,
            Source::Template {} => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "compiled template replay needs its full construction recipe",
                ))
            }
        };
        Ok(super::reader::asked(
            format,
            self.no_calls,
            self.no_reasoning,
        ))
    }
}

#[cfg(all(test, feature = "diagnostic-capture"))]
mod tests {
    use symphony::formats::{for_model_shared, groups};

    use super::*;
    use crate::{
        config::ParserChoice,
        routers::grpc::regular::symphony::{reader, testing::*, Asks},
    };

    fn round_trip(
        model: &str,
        names: &ModelNames<'_>,
        no_calls: bool,
        no_reasoning: bool,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let context = || {
            format!("model={model} names={names:?} no_calls={no_calls} no_reasoning={no_reasoning}")
        };
        let request = request(json!({"model": model, "tools": tools(),
            "tool_choice": if no_calls { "none" } else { "auto" }, "separate_reasoning": !no_reasoning}));
        let assigned = reader::assign(
            ParserChoice::Symphony,
            names,
            Asks::Chat(&request),
            None,
            PROMPT,
            &templates(),
        )
        .map_err(|error| format!("{}: assign: {error}", context()))?
        .ok_or_else(|| format!("{}: missing Symphony assignment", context()))?;
        let value = serde_json::to_value(&assigned)
            .map_err(|error| format!("{}: serialize: {error}", context()))?;
        assert_eq!(value["prompt"], assigned.prompt);
        let recipe: FormatRecipe = serde_json::from_value(value["recipe"].clone())
            .map_err(|error| format!("{}: deserialize: {error}", context()))?;
        assert_eq!(recipe.selector_model(), model);
        assert_eq!(
            (recipe.no_calls, recipe.no_reasoning),
            (no_calls, no_reasoning)
        );
        let restored = recipe
            .resolve()
            .map_err(|error| format!("{}: resolve {recipe:?}: {error}", context()))?;
        assert_eq!(*restored, *assigned.format, "{}", context());
        Ok(value["recipe"]["source"].clone())
    }

    #[test]
    fn registry_and_named_selection_round_trip_without_changing_the_assigned_format(
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (model, _, table) in groups() {
            // Matching labels keep the model's table, even when flags select something else.
            for labels in [Names::default(), pair(Some(table.name()), None)] {
                let flags = pair(Some("mistral_v13"), None);
                let names = ModelNames {
                    labels,
                    flags: &flags,
                };
                assert_eq!(
                    round_trip(model, &names, false, false)?,
                    json!({"kind": "registry"})
                );
            }
        }
        for table in Table::ALL {
            let flags = pair(Some(table.name()), None);
            let expected = json!({"kind": "named", "table": table.name()});
            assert_eq!(
                round_trip("unknown-model", &ModelNames::flagged(&flags), false, false)?,
                expected
            );
            // An explicit different label overrides a known model's registry entry.
            let model = if table == Table::Qwen3 {
                "skt/A.X-3.1-Light"
            } else {
                "Qwen/Qwen3-8B"
            };
            assert_eq!(
                round_trip(model, &names(Some(table.name()), None), false, false)?,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn a_table_name_alone_loses_the_model_opener_and_request_variants(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let model = "skt/A.X-3.1-Light";
        let named = Table::Qwen2_5.shared();
        let registered = for_model_shared(model).unwrap();
        assert_eq!(registered.name(), named.name());
        assert_ne!(registered, named);
        let prompt = "prefix<|im_start|><|assistant|>";
        assert_ne!(registered.replay(prompt), named.replay(prompt));
        for no_calls in [false, true] {
            for no_reasoning in [false, true] {
                round_trip(model, &names(Some("qwen2.5"), None), no_calls, no_reasoning)?;
                round_trip(
                    "unknown-model",
                    &names(Some("qwen3"), None),
                    no_calls,
                    no_reasoning,
                )?;
            }
        }
        Ok(())
    }

    #[test]
    fn recipe_keeps_selector_after_request_drop_and_spec_clone(
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::routers::grpc::spec::ChatResponseSpec;

        let original_model = "skt/A.X-3.1-Light";
        let canonical_model = "Qwen/Qwen2.5-VL-72B-Instruct";
        let request = request(json!({"model": original_model, "tools": tools()}));
        let mut spec = ChatResponseSpec::from(&request);
        spec.symphony = reader::assign(
            ParserChoice::Symphony,
            &names(None, None),
            Asks::Chat(&request),
            None,
            "prefix<|im_start|><|assistant|>",
            &templates(),
        )?;
        drop(request);
        let cloned = spec.clone();
        let assigned = cloned.symphony.ok_or("missing assignment")?;
        let recipe = assigned.recipe.as_ref().ok_or("missing recipe")?;
        assert_eq!(recipe.selector_model(), original_model);
        assert!(Arc::ptr_eq(
            &recipe.selector_model,
            &spec
                .symphony
                .as_ref()
                .ok_or("missing original assignment")?
                .recipe
                .as_ref()
                .ok_or("missing original recipe")?
                .selector_model,
        ));
        let stored = serde_json::to_vec(recipe)?;
        let restored: FormatRecipe = serde_json::from_slice(&stored)?;
        assert_eq!(*restored.resolve()?, *assigned.format);
        let wrong = for_model_shared(canonical_model).ok_or("missing canonical fixture")?;
        assert_eq!(wrong.name(), assigned.format.name());
        assert_ne!(*wrong, *assigned.format);
        Ok(())
    }

    #[test]
    fn unsupported_and_malformed_recipes_never_fall_back_to_a_named_table() {
        let registry = FormatRecipe {
            selector_model: "unknown-model".into(),
            source: Source::Registry {},
            no_calls: false,
            no_reasoning: false,
        };
        assert_eq!(
            registry.resolve().unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        let template = FormatRecipe {
            source: Source::Template {},
            ..registry
        };
        let restored: FormatRecipe =
            serde_json::from_str(&serde_json::to_string(&template).unwrap()).unwrap();
        assert_eq!(
            restored.resolve().unwrap_err().kind(),
            std::io::ErrorKind::Unsupported
        );
        let manual = assignment(Arc::new(Format::new("custom")), "model", "prompt");
        let value = serde_json::to_value(manual).unwrap();
        assert!(value["recipe"].is_null());
        assert!(serde_json::from_value::<FormatRecipe>(value["recipe"].clone()).is_err());
        for value in [
            json!({"source": {"kind": "registry"}, "no_calls": false, "no_reasoning": false}),
            json!({"selector_model": null, "source": {"kind": "registry"}, "no_calls": false, "no_reasoning": false}),
            json!({"selector_model": "Qwen/Qwen3-8B", "source": {"kind": "named", "table": "QWEN3"}, "no_calls": false, "no_reasoning": false}),
            json!({"selector_model": "Qwen/Qwen3-8B", "source": {"kind": "named", "table": "missing"}, "no_calls": false, "no_reasoning": false}),
            json!({"selector_model": "Qwen/Qwen3-8B", "source": {"kind": "registry", "extra": 1}, "no_calls": false, "no_reasoning": false}),
            json!({"selector_model": "Qwen/Qwen3-8B", "source": {"kind": "template", "extra": 1}, "no_calls": false, "no_reasoning": false}),
            json!({"selector_model": "Qwen/Qwen3-8B", "source": {"kind": "registry"}, "no_calls": false}),
            json!({"selector_model": "Qwen/Qwen3-8B", "source": {"kind": "registry"}, "no_calls": false, "no_reasoning": false, "extra": 1}),
        ] {
            assert!(
                serde_json::from_value::<FormatRecipe>(value.clone()).is_err(),
                "accepted malformed recipe: {value}"
            );
        }
    }
}
