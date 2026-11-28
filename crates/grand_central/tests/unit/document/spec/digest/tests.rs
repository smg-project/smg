use std::collections::BTreeSet;

use super::*;
use crate::document::spec::{tests::doc, AcrossRanks, Pairing, PdPairPlanSpec, PRESETS};

/// The digest copies none of the document: what it hashes is, byte for
/// byte, the JSON the document itself writes under a zeroed line. Held
/// for every preset, under a line and under none, for a document that
/// writes every field the schema names and for one that writes the
/// least: a field left out where it is the default is left out of both.
#[test]
fn a_digest_hashes_what_the_document_itself_writes_under_a_zeroed_line() {
    let base = PolicySpec::preset("least_load").expect("a preset");
    let every_field = PolicySpec {
        explain: ExplainSpec { sample: 2 },
        view: ViewSpec {
            headers: vec!["x-team".to_owned()],
            ..base.view.clone()
        },
        index: BTreeMap::from([("another*".to_owned(), IndexSpec::default())]),
        signals: BTreeMap::from([(
            "held".to_owned(),
            SignalSpec {
                unit: "tokens".to_owned(),
                max_age_ms: Some(4_000),
                across_ranks: AcrossRanks::Max,
            },
        )]),
        plans: PlansSpec {
            pd_pair: PdPairPlanSpec {
                pairing: Pairing::Off,
                ..PdPairPlanSpec::default()
            },
        },
        bindings: vec![BindingSpec::default()],
        ..base
    };
    let written = serde_json::to_value(&every_field).expect("the document is JSON");
    let schema = serde_json::to_value(PolicySpec::schema()).expect("the schema is JSON");
    let fields = |of: &serde_json::Value| -> BTreeSet<String> {
        let fields = of.as_object().expect("a map of fields");
        fields.keys().cloned().collect()
    };
    assert_eq!(
        fields(&written),
        fields(&schema["properties"]),
        "the document writes every field the schema names"
    );
    let least = doc(r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"best"}}}"#);
    let mut documents = vec![every_field, least];
    for name in PRESETS {
        let preset = PolicySpec::preset(name).expect(name);
        documents.push(PolicySpec {
            generation: 7,
            ..preset.clone()
        });
        documents.push(preset);
    }
    for document in documents {
        let borrowed = serde_json::to_string(&document.lineless()).expect("serializes");
        let owned = serde_json::to_string(&PolicySpec {
            generation: 0,
            ..document
        })
        .expect("serializes");
        assert_eq!(borrowed, owned);
    }
}
