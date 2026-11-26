use super::*;
use crate::{
    engine::{Candidate, Constraints, Filter, PoolFacts, RequestFacts, Targets, TieBand},
    stages::Lifecycle,
};

#[test]
fn a_parameter_key_written_twice_is_refused_inside_maps_and_lists() {
    let score_yaml = |score: &str| {
        format!("apiVersion: {API_VERSION}\nprofiles:\n  default:\n    score: {score}\n    pick: best\n")
    };
    for score in [
        "[{ queue-price: { costs: { unit: 17, unit: 31 } } }]",
        "[{ queue-price: { costs: [[{ unit: 17, unit: 31 }]] } }]",
    ] {
        let error =
            serde_yaml::from_str::<PolicySpec>(&score_yaml(score)).expect_err("duplicate key");
        assert!(
            error.to_string().contains("duplicate field `unit`"),
            "{error}"
        );
    }
    let json = format!(
        r#"{{"apiVersion":"{API_VERSION}","profiles":{{"default":{{"score":[{{"queue-price":{{"costs":[{{"unit":17,"unit":31}}]}}}}],"pick":"best"}}}}}}"#
    );
    let error = serde_json::from_str::<PolicySpec>(&json).expect_err("duplicate key");
    assert!(
        error.to_string().contains("duplicate field `unit`"),
        "{error}"
    );

    let read: PolicySpec = serde_yaml::from_str(&score_yaml(
        "[{ queue-price: { costs: [[{ unit: 17 }]], weight: 3.5 } }]",
    ))
    .expect("a well-formed document");
    let value = serde_json::to_value(&read).expect("serializes");
    assert_eq!(
        value["profiles"]["default"]["score"],
        serde_json::json!([{ "queue-price": { "costs": [[{ "unit": 17 }]], "weight": 3.5 } }])
    );
    let again: PolicySpec = serde_json::from_value(value).expect("parses again");
    assert_eq!(
        read.digest().expect("digests"),
        again.digest().expect("digests")
    );
}

/// The order a file writes a registered scorer's parameter keys in is
/// in no digest, at any depth: two files that write one map in two
/// orders are one document. A value that differs is another.
#[test]
fn the_order_of_a_scorers_parameter_keys_is_in_no_digest() {
    let document = |score: &str| -> PolicySpec {
        serde_yaml::from_str(&format!(
                "apiVersion: {API_VERSION}\nprofiles:\n  default:\n    score: {score}\n    pick: best\n"
            ))
            .expect("a well-formed document")
    };
    let one = document(
        "[{ queue-price: { weight: 2, per_request: 500, costs: [{ unit: 17, flat: 3 }] } }]",
    );
    let other = document(
        "[{ queue-price: { costs: [{ flat: 3, unit: 17 }], per_request: 500, weight: 2 } }]",
    );
    assert_eq!(
        one.digest().expect("digests"),
        other.digest().expect("digests"),
        "two orders of one map"
    );
    assert_eq!(
        serde_json::to_string(&one).expect("serializes"),
        serde_json::to_string(&other).expect("serializes")
    );
    let moved = document(
        "[{ queue-price: { weight: 2, per_request: 501, costs: [{ unit: 17, flat: 3 }] } }]",
    );
    assert_ne!(
        one.digest().expect("digests"),
        moved.digest().expect("digests")
    );
}

#[test]
fn an_applied_generation_reads_block_hashes_as_deep_as_its_stages_do() {
    use HashKind::{Text, Token};

    let preset = |name: &str| {
        let mut document = PolicySpec::preset(name).expect("a preset of this build");
        document.generation = 1;
        document
    };
    let applied =
        |document: &PolicySpec| AppliedSpec::validate(document).expect("the document validates");
    // The source each prefix stage a document states with parameters
    // reads, as the presets state theirs.
    let read_from = |document: &mut PolicySpec, source: AffinitySource| {
        let stages = document.profiles.values_mut();
        for stage in stages.flat_map(|profile| &mut profile.score) {
            if let ScoreSpec::Params(ScoreStage::PrefixAffinity(params)) = stage {
                params.source = Some(source);
            }
        }
    };
    // No stage of the random preset reads the hashes, and the unbound
    // start has no stage.
    let random = applied(&preset("random"));
    assert_eq!(random.block_hashes_read(Token, 0), 0);
    assert_eq!(random.block_hashes_read(Text, 0), 0);
    assert_eq!(AppliedSpec::unbound().block_hashes_read(Token, 0), 0);
    assert_eq!(AppliedSpec::unbound().max_block_hashes(), 256);
    // A prefix key 256 tokens deep, in the request's own blocks: 16 of
    // the view's 16 tokens, 32 of a request counted by 8 and 8 of one
    // counted by 32, and of a text its 1,024 characters in blocks of
    // the size it was counted by. The cap stays the document's.
    let mut keyed = preset("prefix_hash");
    let ring = applied(&keyed);
    assert_eq!(ring.block_hashes_read(Token, 0), 16);
    assert_eq!(ring.block_hashes_read(Token, 8), 32);
    assert_eq!(ring.block_hashes_read(Token, 32), 8);
    assert_eq!(ring.block_hashes_read(Text, 0), 64);
    assert_eq!(ring.block_hashes_read(Text, 64), 16);
    assert_eq!(ring.max_block_hashes(), 256);
    // A request that names no block size is counted by the view's, and
    // no depth is past the view's cap.
    keyed.view.block_size = Some(32);
    keyed.view.max_block_hashes = 24;
    let ring = applied(&keyed);
    assert_eq!(ring.max_block_hashes(), 24);
    assert_eq!(ring.block_hashes_read(Token, 0), 8);
    assert_eq!(ring.block_hashes_read(Text, 0), 24);
    // The controller's door builds a generation under a digest it took
    // already: that builder keeps the depth too.
    #[cfg(feature = "controller")]
    {
        let digest = keyed.digest().expect("the document digests");
        let applied =
            AppliedSpec::validate_digested(&keyed, digest, &StageRegistry::builtin(), &[])
                .expect("the preset validates");
        assert_eq!(applied.block_hashes_read(Token, 0), 8);
    }
    // A stage with no depth of its own reads to the view's cap.
    let mut held = preset("cache_aware");
    assert_eq!(applied(&held).block_hashes_read(Token, 0), 256);
    held.view.max_block_hashes = 128;
    // A kind no stage declares is read by none. The prefix stage
    // declares text hashes where its source reads the record of sent
    // prefixes, and none where it reads cache events alone.
    read_from(&mut held, AffinitySource::Index);
    assert_eq!(applied(&held).block_hashes_read(Token, 0), 128);
    assert_eq!(applied(&held).block_hashes_read(Text, 4), 0);
    read_from(&mut held, AffinitySource::Merged);
    assert_eq!(applied(&held).block_hashes_read(Text, 4), 128);
    // A generation reads as deep as the deepest of its profiles.
    let two = doc(
        r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":[{"consistent-hash":{"prefix_tokens":256}}],"pick":"best"},"q":{"score":[{"consistent-hash":{"prefix_tokens":512}}],"pick":"best"}},"bindings":[{"match":{"model":"a*"},"profiles":{"single":"p"}},{"match":{"model":"*"},"profiles":{"single":"q"}}]}"#,
    );
    assert_eq!(applied(&two).block_hashes_read(Token, 0), 32);
    // Under a view's boundaries a prefix key is read at the deepest one
    // a request reaches, so the generation reads as deep as that one,
    // and no shallower than the key's own depth.
    let mut levelled = preset("prefix_hash");
    levelled.view.boundaries = vec![64, 1024];
    assert_eq!(applied(&levelled).block_hashes_read(Token, 0), 64);
    levelled.view.boundaries = vec![64];
    assert_eq!(applied(&levelled).block_hashes_read(Token, 0), 16);
}

/// A guest stage of each kind, under the name it is registered by, that
/// declares the request's token hashes and names no depth.
#[derive(Debug)]
struct ReadsHashes(&'static str);

impl ReadsHashes {
    fn declared() -> Needs {
        Needs {
            token_hashes: true,
            ..Needs::default()
        }
    }
}

impl Lifecycle for ReadsHashes {}

impl Filter for ReadsHashes {
    fn name(&self) -> &'static str {
        self.0
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn needs(&self) -> Needs {
        Self::declared()
    }
}

impl Scorer for ReadsHashes {
    fn name(&self) -> &'static str {
        self.0
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(0.0);
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        Self::declared()
    }
}

impl Picker for ReadsHashes {
    fn name(&self) -> &'static str {
        self.0
    }

    fn pick(
        &self,
        _: &RequestFacts<'_>,
        _: &[Candidate],
        _: &[u32],
        _: &[f32],
        _: TieBand,
        _: &mut Targets,
    ) {
    }

    fn needs(&self) -> Needs {
        Self::declared()
    }
}

#[test]
fn a_stage_that_names_no_depth_reads_the_block_hashes_to_the_cap() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter("hashing-filter", || Box::new(ReadsHashes("hashing-filter")))
        .and_then(|catalog| {
            catalog.scorer("hashing-scorer", || Box::new(ReadsHashes("hashing-scorer")))
        })
        .and_then(|catalog| {
            catalog.picker("hashing-picker", || Box::new(ReadsHashes("hashing-picker")))
        })
        .expect("three free names");
    let depth = |profile: &str| {
        let document = doc(&format!(
            r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{profile}}}}}"#
        ));
        let (_, depth, _) = document
            .built(&catalog, &[])
            .expect("the document validates");
        depth.read
    };
    assert_eq!(depth(r#"{"pick":"random"}"#), HashDepth::Unread);
    // A guest of each kind declares the hashes and names no depth.
    for profile in [
        r#"{"filter":["hashing-filter"],"pick":"random"}"#,
        r#"{"score":["hashing-scorer"],"pick":"best"}"#,
        r#"{"pick":"hashing-picker"}"#,
    ] {
        assert_eq!(depth(profile), HashDepth::Whole, "{profile}");
    }
    // A prefix key reads to its own depth, and beside a stage that
    // reads the list whole the generation reads it whole.
    let key = r#"{"consistent-hash":{"prefix_tokens":256}}"#;
    assert_eq!(
        depth(&format!(r#"{{"score":[{key}],"pick":"best"}}"#)),
        HashDepth::Tokens(256)
    );
    assert_eq!(
        depth(&format!(
            r#"{{"score":[{key},"hashing-scorer"],"pick":"best"}}"#
        )),
        HashDepth::Whole
    );
}

/// A registered scorer that reads the affinity column and says nothing
/// of a gate of its own.
#[derive(Debug)]
struct ReadsColumn;

impl Lifecycle for ReadsColumn {}

impl Scorer for ReadsColumn {
    fn name(&self) -> &'static str {
        "reads-column"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(0.0);
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        Needs {
            affinity: true,
            ..Needs::default()
        }
    }
}

/// A profile weighs load against affinity where one of its scorers says
/// so of itself, and by nothing a stage is fed. Each of the two
/// affinity stages says so, behind the view's report age limit too. A
/// registered scorer that reads the column and says nothing leaves its
/// profile's need standing and its profile unmarked, and beside one of
/// the two it changes nothing.
#[test]
fn a_profile_weighs_by_a_scorers_own_word_and_by_no_need() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer("reads-column", || Box::new(ReadsColumn))
        .expect("a free name");
    // The profile's need for the column, then its word.
    let said = |score: &str, view: &str| {
        let document = doc(&format!(
            r#"{{"apiVersion":"smg.policy/v1","view":{view},"profiles":{{"p":{{"score":[{score}],"pick":"best"}}}}}}"#
        ));
        let (engine, ..) = document
            .built(&catalog, &[])
            .expect("the document validates");
        (engine.profiles()[0].needs().affinity, engine.weighs(0))
    };
    for view in ["{}", r#"{"max_report_age_ms":500}"#] {
        assert_eq!(said(r#""prefix-affinity""#, view), (true, true), "{view}");
        assert_eq!(said(r#""warmup-slice""#, view), (true, true), "{view}");
        assert_eq!(said(r#""reads-column""#, view), (true, false), "{view}");
        let both = r#""reads-column","warmup-slice""#;
        assert_eq!(said(both, view), (true, true), "{view}");
        assert_eq!(said(r#""least-load""#, view), (false, false), "{view}");
    }
}

fn doc(text: &str) -> PolicySpec {
    serde_json::from_str(text).expect("a well-formed document")
}

fn parse_error(text: &str) -> String {
    serde_json::from_str::<PolicySpec>(text)
        .expect_err("a refused document")
        .to_string()
}

fn spec_with_score(score: &str) -> PolicySpec {
    doc(&format!(
        r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{"filter":["healthy"],"score":[{score}],"pick":"best"}}}}}}"#
    ))
}

fn spec_with_pick(pick: &str) -> PolicySpec {
    doc(&format!(
        r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{"score":["least-load"],"pick":{pick}}}}}}}"#
    ))
}

/// Two profiles and an allow-listed header, for the binding tests.
fn spec_with_binding(binding: &str) -> PolicySpec {
    doc(&format!(
        r#"{{"apiVersion":"smg.policy/v1","view":{{"headers":["x-session-id"]}},"profiles":{{"p":{{"pick":"best"}},"q":{{"pick":"random"}}}},"bindings":[{binding}]}}"#
    ))
}

#[test]
fn shadow_absent_keeps_every_preset_digest() {
    let expected = [
        (
            "random",
            "sha256:12f28a338634ac96fb8d5883f9e354341329ffb8cf23417c4b6ace700d03d3ec",
        ),
        (
            "round_robin",
            "sha256:62e5652e4940b185ed50b247c68b27c4a3f7ac43683f66b97404d449669720fc",
        ),
        (
            "passthrough",
            "sha256:3defa376775031aa71f5d87b1dd2c7dbdaf7eaf42f7f0f0616ac345cdddd6f1d",
        ),
        (
            "power_of_two",
            "sha256:4c4b153064585f5e23a4b62e1fd4cf2d6b07c0dea6340c24834260d2e4281f66",
        ),
        (
            "least_load",
            "sha256:80d7fcef3d2df7ee25aa79c1ccfa784cad0dbc91785194d48e108e660e8cda88",
        ),
        (
            "consistent_hashing",
            "sha256:c9bb5fdc96f68381ac99700d79a793499cdf915ab2ce590ffb82942b398e6745",
        ),
        (
            "prefix_hash",
            "sha256:f5bc21e35bb782d5ee2a559ece6b18c0905cebcd20c354b294ecf1c1e1c5d871",
        ),
        (
            "manual",
            "sha256:d12dd3db8aa8ece3fbb62bd354c3db14d97d92e57079dc191ac3546ccd04593e",
        ),
        (
            "cache_aware",
            "sha256:864958791685f8894b437e698f0bcee6ad48a81b56d248b230c93dd75b9b0513",
        ),
        (
            "bucket",
            "sha256:168a1e9a62410e6bba35226e96c40715392d7fe5e73b88cf242e2c48a0686f3c",
        ),
    ];
    assert_eq!(expected.map(|(name, _)| name), PRESETS);
    for (name, digest) in expected {
        let spec = PolicySpec::preset(name).expect("a preset");
        assert_eq!(
            spec.digest().expect("a digest").to_string(),
            digest,
            "{name}"
        );
    }
}

#[test]
fn shadow_round_trips_as_a_profile_name() {
    let spec = spec_with_binding(r#"{"profile":"p","shadow":"q"}"#);
    let json = serde_json::to_value(&spec).expect("writes JSON");
    assert_eq!(json["bindings"][0]["shadow"], "q");
    assert_eq!(
        serde_json::from_value::<PolicySpec>(json).expect("reads JSON"),
        spec
    );
    let yaml = serde_yaml::to_string(&spec).expect("writes YAML");
    assert_eq!(
        serde_yaml::from_str::<PolicySpec>(&yaml).expect("reads YAML"),
        spec
    );
    let engine = spec.validate().expect("a separate shadow profile is valid");
    let binding = engine.binding("m", Phase::Single).expect("the binding");
    assert_eq!(binding.shadow, Some(1));
    assert_eq!(binding.decode_shadow, None);
    assert_eq!(engine.profiles()[1].name, "q");
}

#[test]
fn shadow_refuses_an_unknown_profile_at_the_field() {
    let spec = spec_with_binding(r#"{"profile":"p","shadow":"absent"}"#);
    assert_eq!(
        spec.validate().expect_err("the profile is absent"),
        SpecError::UnknownProfileAt {
            path: "bindings[0].shadow".into(),
            name: "absent".into(),
        }
    );
}

#[test]
fn shadow_refuses_a_profile_named_live_anywhere() {
    for (bindings, name) in [
        (r#"{"profile":"p","shadow":"p"}"#, "p"),
        (r#"{"profiles":{"single":"p"},"shadow":"p"}"#, "p"),
        (r#"{"profile":"p","shadow":"q"},{"profile":"q"}"#, "q"),
        (
            r#"{"profile":"p","shadow":"q"},{"plan":"pd-pair","profiles":{"prefill":"p","decode":"q"}}"#,
            "q",
        ),
        (
            r#"{"profile":"p","shadow":"q"},{"profile":"p","canary":{"profile":"q","percent":19}}"#,
            "q",
        ),
    ] {
        let error = spec_with_binding(bindings)
            .validate()
            .expect_err("the shadow is also named live");
        assert!(
            matches!(
                error,
                SpecError::BadField { ref path, ref value, .. }
                    if path == "bindings[0].shadow" && value == name
            ),
            "{error} for {bindings}"
        );
    }
}

#[test]
fn shadow_on_a_pair_refuses_a_single_name_at_the_field() {
    let spec = spec_with_binding(
        r#"{"plan":"pd-pair","profiles":{"prefill":"p","decode":"p"},"shadow":"q"}"#,
    );
    let error = spec.validate().expect_err("a pair names its sides");
    assert!(
        matches!(
            error,
            SpecError::BadField { ref path, .. } if path == "bindings[0].shadow"
        ),
        "{error}"
    );
}

/// A pair names its shadows by side: the map's keys are `prefill` and
/// `decode`, and a side it leaves out is not shadowed. Each value is a
/// profile no binding decides by live. Any other key, and a map on a
/// binding of one profile, are refused by path.
#[test]
fn a_pairs_shadow_map_names_a_profile_a_side() {
    let pair = |shadow: &str| {
        spec_with_binding(&format!(
            r#"{{"plan":"pd-pair","profiles":{{"prefill":"p","decode":"p"}},"shadow":{shadow}}}"#,
        ))
    };
    let sides = |shadow: &str| {
        let engine = pair(shadow).validate().expect("a pair's shadow map");
        let binding = engine.binding("m", Phase::Prefill);
        let binding = binding.expect("the pair's binding");
        (binding.shadow, binding.decode_shadow)
    };
    // `q` is the document's second profile.
    assert_eq!(sides(r#"{"decode":"q"}"#), (None, Some(1)));
    assert_eq!(sides(r#"{"prefill":"q"}"#), (Some(1), None));
    assert_eq!(sides(r#"{"prefill":"q","decode":"q"}"#), (Some(1), Some(1)));
    assert_eq!(
        bad_field(&pair(r#"{"encode":"q"}"#)),
        "bindings[0].shadow.encode"
    );
    // A binding of one profile takes no map, and no plan an empty
    // one: either would load as a document that shadows nothing.
    let single = spec_with_binding(r#"{"profile":"p","shadow":{"decode":"q"}}"#);
    assert_eq!(bad_field(&single), "bindings[0].shadow");
    let empty = spec_with_binding(r#"{"profile":"p","shadow":{}}"#);
    assert_eq!(bad_field(&empty), "bindings[0].shadow");
    assert_eq!(bad_field(&pair("{}")), "bindings[0].shadow");
    assert_eq!(
        pair(r#"{"decode":"absent"}"#)
            .validate()
            .expect_err("unknown profile"),
        SpecError::UnknownProfileAt {
            path: "bindings[0].shadow.decode".into(),
            name: "absent".into(),
        }
    );
    assert_eq!(
        bad_field(&pair(r#"{"decode":"p"}"#)),
        "bindings[0].shadow.decode"
    );
    let doubled = r#"{"profile":"p","shadow":{"single":"q","single":"q"}}"#;
    assert!(serde_json::from_str::<BindingSpec>(doubled).is_err());
}

/// A bad `shadow` value is refused by what is wrong with it: a side
/// written twice by its key, and a value of neither form by the forms
/// the field takes.
#[test]
fn shadow_refuses_a_bad_value_by_its_key_or_its_form() {
    let doubled = r#"{"profile":"p","shadow":{"single":"q","single":"q"}}"#;
    let refusal = serde_json::from_str::<BindingSpec>(doubled)
        .expect_err("a side written twice")
        .to_string();
    assert!(
        refusal.contains("the key `single` is written twice"),
        "{refusal}"
    );
    let number = r#"{"profile":"p","shadow":5}"#;
    let refusal = serde_json::from_str::<BindingSpec>(number)
        .expect_err("a number in the field")
        .to_string();
    assert!(
        refusal.contains("expected a profile name, or a map of sides to profile names"),
        "{refusal}"
    );
}

#[test]
fn shadow_refuses_each_stage_that_opens_a_plugin_scope_by_name() {
    #[derive(Debug)]
    struct Scoped;

    impl crate::engine::DecisionScope for Scoped {
        fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}
        fn close(&self) {}
    }

    impl Lifecycle for Scoped {}

    impl Filter for Scoped {
        fn name(&self) -> &'static str {
            "scoped-filter"
        }
        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }
        fn scope(&self) -> Option<&dyn crate::engine::DecisionScope> {
            Some(self)
        }
    }

    impl Scorer for Scoped {
        fn name(&self) -> &'static str {
            "scoped-scorer"
        }
        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            _: &mut [f32],
        ) -> TieBand {
            TieBand::NORMALIZED
        }
        fn scope(&self) -> Option<&dyn crate::engine::DecisionScope> {
            Some(self)
        }
    }

    impl Picker for Scoped {
        fn name(&self) -> &'static str {
            "scoped-picker"
        }
        fn pick(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            _: &[f32],
            _: TieBand,
            _: &mut Targets,
        ) {
        }
        fn scope(&self) -> Option<&dyn crate::engine::DecisionScope> {
            Some(self)
        }
    }

    let mut catalog = StageRegistry::builtin();
    catalog
        .filter("scoped-filter", || Box::new(Scoped))
        .and_then(|catalog| catalog.scorer("scoped-scorer", || Box::new(Scoped)))
        .and_then(|catalog| catalog.picker("scoped-picker", || Box::new(Scoped)))
        .expect("the stage names are available");
    for (profile, name) in [
        (
            r#"{"filter":["scoped-filter"],"pick":"best"}"#,
            "scoped-filter",
        ),
        (
            r#"{"score":["scoped-scorer"],"pick":"best"}"#,
            "scoped-scorer",
        ),
        (r#"{"pick":"scoped-picker"}"#, "scoped-picker"),
    ] {
        let mut spec = spec_with_binding(r#"{"profile":"p","shadow":"q"}"#);
        spec.profiles.insert(
            "q".into(),
            serde_json::from_str(profile).expect("a profile"),
        );
        assert_eq!(
            spec.validate_in(&catalog, &[])
                .expect_err("the stage needs a scope"),
            SpecError::NotInBuild {
                path: "bindings[0].shadow".into(),
                what: name,
            }
        );
    }
}

fn spec_with_view(view: &str) -> PolicySpec {
    doc(&format!(
        r#"{{"apiVersion":"smg.policy/v1","view":{view},"profiles":{{"p":{{"pick":"best"}}}}}}"#
    ))
}

fn not_in_build(spec: &PolicySpec) -> (String, &'static str) {
    match spec.validate() {
        Err(SpecError::NotInBuild { path, what }) => (path, what),
        other => panic!("validated as {other:?}"),
    }
}

fn bad_field(spec: &PolicySpec) -> String {
    match spec.validate() {
        Err(SpecError::BadField { path, .. }) => path,
        other => panic!("validated as {other:?}"),
    }
}

#[test]
fn explain_sample_is_an_unsigned_count_that_defaults_to_off() {
    let base = PolicySpec::preset("random").expect("a preset");
    let serde_json::Value::Object(mut json) = serde_json::to_value(&base).expect("a document")
    else {
        panic!("a document is an object");
    };
    assert!(!json.contains_key("explain"));
    for sample in [0, 1, 3, u32::MAX] {
        json.insert("explain".into(), serde_json::json!({ "sample": sample }));
        let read =
            serde_json::from_value::<PolicySpec>(json.clone().into()).expect("an unsigned count");
        assert_eq!(read.explain.sample, sample);
        assert_eq!(
            read.validate().expect("a valid document").explain_sample(),
            sample
        );
        assert_eq!(read.digest().ok() == base.digest().ok(), sample == 0);
    }
    for invalid in [
        serde_json::json!({ "sample": -1 }),
        serde_json::json!({ "sample": 1.5 }),
        serde_json::json!({ "sample": "1" }),
        serde_json::json!({ "sample": u64::from(u32::MAX) + 1 }),
        serde_json::json!({ "slots": 1_024 }),
    ] {
        json.insert("explain".into(), invalid);
        assert!(serde_json::from_value::<PolicySpec>(json.clone().into()).is_err());
    }
    json.insert("explain".into(), serde_json::json!({}));
    let empty = serde_json::from_value::<PolicySpec>(json.into()).expect("an empty section");
    assert_eq!(empty, base);
}

#[test]
fn every_preset_validates_and_declares_its_needs() {
    for name in PRESETS {
        let spec = PolicySpec::preset(name).expect(name);
        let engine = spec.validate().expect(name);
        let load_aware = ["least_load", "power_of_two", "cache_aware"].contains(&name);
        assert_eq!(engine.needs().loads, load_aware, "{name}");
        let ring = ["consistent_hashing", "prefix_hash"].contains(&name);
        assert_eq!(engine.needs().ring, ring, "{name}");
        // A request's block hashes are fed where a stage reads them: the
        // affinity stage and the ring's prefix form.
        let hashes = ["cache_aware", "prefix_hash"].contains(&name);
        assert_eq!(engine.needs().token_hashes, hashes, "{name}");
        assert_eq!(engine.needs().text_hashes, hashes, "{name}");
        assert!(!engine.needs().text, "{name}");
        // The request's size is counted where a stage places by it or
        // takes a share of it: a host that sends a request on unread
        // reads it for that stage.
        let sized = ["bucket", "cache_aware"].contains(&name);
        assert_eq!(engine.needs().prompt_size, sized, "{name}");
        // The affinity stage's source is left out: it reads each worker
        // by what it has, its cache events or the prefixes sent to it.
        assert_eq!(engine.needs().events, name == "cache_aware", "{name}");
        assert_eq!(
            engine.needs().sent_prefixes,
            name == "cache_aware",
            "{name}"
        );
        // The affinity column is fed where an affinity stage reads it
        // and nowhere else: a preset without one declares no column.
        assert_eq!(engine.needs().affinity, name == "cache_aware", "{name}");
        // The same preset alone weighs load against affinity: its two
        // affinity stages say so of themselves, and no other stage does.
        assert_eq!(engine.weighs(0), name == "cache_aware", "{name}");
        assert_eq!(engine.needs().headers, name == "manual", "{name}");
        assert!(!engine.needs().active_sessions, "{name}");
        assert!(!engine.needs().session_in_flight, "{name}");
        assert!(engine.profile_for("any", Phase::Single).is_some(), "{name}");
        // One profile bound to every model and phase: the per-phase and
        // per-binding reads are the summary.
        for phase in [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode] {
            assert_eq!(engine.needs_for_phase(phase), engine.needs(), "{name}");
            assert_eq!(engine.needs_for("any", phase), engine.needs(), "{name}");
        }
    }
    assert_eq!(
        PolicySpec::preset("no_such_policy").err(),
        Some(SpecError::UnknownPreset("no_such_policy".into()))
    );
}

/// The needs are read per binding and per phase: a pd-pair binding's
/// prefill side reads what the prefill profile's stages read, its decode
/// side the decode profile's, and a phase's fold spans every binding
/// that covers it.
#[test]
fn the_needs_are_read_per_binding_and_per_phase() {
    let text = r#"{
          "apiVersion": "smg.policy/v1",
          "profiles": {
            "default": { "filter": ["healthy"], "pick": "round-robin" },
            "prefill": { "filter": ["healthy", "has-capacity"], "score": ["least-load"], "pick": "best" },
            "decode": { "filter": ["healthy", "has-room"], "score": ["session-affinity"], "pick": "best" }
          },
          "bindings": [
            { "match": { "model": "pd/*", "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "prefill", "decode": "decode" } },
            { "match": { "model": "*" }, "profile": "default" }
          ]
        }"#;
    let engine = doc(text).validate().expect("validates");
    let loads = Needs {
        loads: true,
        ..Needs::default()
    };
    let headers = Needs {
        headers: true,
        ..Needs::default()
    };
    assert_eq!(engine.needs(), loads.or(headers));
    assert_eq!(engine.needs_for("pd/x", Phase::Prefill), loads);
    assert_eq!(engine.needs_for("pd/x", Phase::Decode), headers);
    // The pair binding covers neither phase; the catch-all's profile
    // reads nothing.
    assert_eq!(engine.needs_for("pd/x", Phase::Single), Needs::default());
    assert_eq!(engine.needs_for("other", Phase::Prefill), Needs::default());
    // Per phase: the pair binding and the catch-all, whose empty phase
    // list covers every phase.
    assert_eq!(engine.needs_for_phase(Phase::Prefill), loads);
    assert_eq!(engine.needs_for_phase(Phase::Decode), headers);
    assert_eq!(engine.needs_for_phase(Phase::Single), Needs::default());
    assert_eq!(engine.needs_for_phase(Phase::Encode), Needs::default());

    // The shape the gateway's configuration produces: every model's pair
    // phases bound first, the catch-all behind it. The catch-all decides
    // no pair request, so its stages add nothing to the pair phases.
    let text = r#"{
          "apiVersion": "smg.policy/v1",
          "profiles": {
            "default": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" },
            "prefill": { "filter": ["healthy", "has-capacity"], "score": ["session-affinity"], "pick": "best" },
            "decode": { "filter": ["healthy", "has-room"], "pick": "round-robin" }
          },
          "bindings": [
            { "match": { "model": "*", "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "prefill", "decode": "decode" } },
            { "match": { "model": "*" }, "profile": "default" }
          ]
        }"#;
    let engine = doc(text).validate().expect("validates");
    assert_eq!(engine.needs_for_phase(Phase::Prefill), headers);
    assert_eq!(engine.needs_for_phase(Phase::Decode), Needs::default());
    assert_eq!(engine.needs_for_phase(Phase::Single), loads);
    assert_eq!(engine.needs_for_phase(Phase::Encode), loads);

    // A decode-phase binding before the pair binding decides the decode
    // requests that come alone; a pair's decode side is decided by the
    // binding its prefill lookup resolves to, with that pair's decode
    // profile. The decode needs carry both. The pair's decode profile
    // is not the first profile in name order, the order the engine
    // holds them in: a decode slot that read the first profile would
    // find the decode-only one and lose the pair's needs.
    let text = r#"{
          "apiVersion": "smg.policy/v1",
          "profiles": {
            "decode-only": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" },
            "prefill": { "filter": ["healthy"], "pick": "round-robin" },
            "pair-decode": { "filter": ["healthy"], "score": ["session-affinity"], "pick": "best" },
            "default": { "filter": ["healthy"], "pick": "round-robin" }
          },
          "bindings": [
            { "match": { "model": "*", "phase": ["decode"] }, "profile": "decode-only" },
            { "match": { "model": "pd/*", "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "prefill", "decode": "pair-decode" } },
            { "match": { "model": "*" }, "profile": "default" }
          ]
        }"#;
    let engine = doc(text).validate().expect("validates");
    assert_eq!(engine.needs_for("pd/x", Phase::Decode), loads.or(headers));
    assert_eq!(engine.needs_for("other", Phase::Decode), loads);
    assert_eq!(engine.needs_for("pd/x", Phase::Prefill), Needs::default());
    assert_eq!(engine.needs_for_phase(Phase::Decode), loads.or(headers));
    assert_eq!(engine.needs_for_phase(Phase::Prefill), Needs::default());

    // An encode-phase binding before an `epd` binding decides the
    // encode items that come alone. The encode side of an encode
    // request decided in one call is decided by the binding its
    // prefill lookup resolves to, with that pair's encode profile. The
    // encode needs carry both, for the model and for the phase.
    let text = r#"{
          "apiVersion": "smg.policy/v1",
          "profiles": {
            "encode-only": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" },
            "side": { "filter": ["healthy"], "pick": "round-robin" },
            "pair-encode": { "filter": ["healthy"], "score": ["session-affinity"], "pick": "best" },
            "default": { "filter": ["healthy"], "pick": "round-robin" }
          },
          "bindings": [
            { "match": { "model": "*", "phase": ["encode"] }, "profile": "encode-only" },
            { "match": { "model": "epd/*" }, "plan": "epd", "profiles": { "encode": "pair-encode", "prefill": "side", "decode": "side" } },
            { "match": { "model": "*" }, "profile": "default" }
          ]
        }"#;
    let engine = doc(text).validate().expect("validates");
    assert_eq!(engine.needs_for("epd/x", Phase::Encode), loads.or(headers));
    assert_eq!(engine.needs_for("other", Phase::Encode), loads);
    assert_eq!(engine.needs_for("epd/x", Phase::Prefill), Needs::default());
    assert_eq!(engine.needs_for_phase(Phase::Encode), loads.or(headers));
    assert_eq!(engine.needs_for_phase(Phase::Prefill), Needs::default());
}

/// The needs rows are what a host must feed: a host that feeds, for a
/// model and a phase, every row whose glob matches the model feeds what
/// the decision of that model and phase reads, with no copy of the
/// bindings' order or of the pair's rule. A pair's decode profile is a
/// decode row of the pair binding's glob, two bindings of one glob write
/// one row a phase, and nothing to feed is no row.
#[test]
fn the_needs_rows_feed_what_every_decision_reads() {
    /// What a host feeds for `model` and `phase`: the union of the rows
    /// whose glob matches.
    fn fed(engine: &Engine, model: &str, phase: Phase) -> Needs {
        engine
            .needs_rows()
            .into_iter()
            .filter(|(glob, at, _)| {
                let matches = match (glob.strip_prefix('*'), glob.strip_suffix('*')) {
                    _ if *glob == "*" => true,
                    (Some(suffix), _) => model.ends_with(suffix),
                    (_, Some(prefix)) => model.starts_with(prefix),
                    _ => *glob == model,
                };
                matches && *at == phase
            })
            .fold(Needs::default(), |fed, (_, _, needs)| fed.or(needs))
    }
    let loads = Needs {
        loads: true,
        ..Needs::default()
    };
    let headers = Needs {
        headers: true,
        ..Needs::default()
    };
    let phases = [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode];

    // A decode-phase binding before a pair binding, and a catch-all that
    // reads nothing: the pair's decode profile is a row of the pair's
    // glob, and the catch-all writes none.
    let text = r#"{
          "apiVersion": "smg.policy/v1",
          "profiles": {
            "decode-only": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" },
            "prefill": { "filter": ["healthy"], "pick": "round-robin" },
            "pair-decode": { "filter": ["healthy"], "score": ["session-affinity"], "pick": "best" },
            "default": { "filter": ["healthy"], "pick": "round-robin" }
          },
          "bindings": [
            { "match": { "model": "*", "phase": ["decode"] }, "profile": "decode-only" },
            { "match": { "model": "pd/*", "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "prefill", "decode": "pair-decode" } },
            { "match": { "model": "*" }, "profile": "default" }
          ]
        }"#;
    let engine = doc(text).validate().expect("validates");
    assert_eq!(
        engine.needs_rows(),
        [
            ("*", Phase::Decode, loads),
            ("pd/*", Phase::Decode, headers)
        ]
    );
    for model in ["pd/x", "other"] {
        for phase in phases {
            assert_eq!(
                fed(&engine, model, phase),
                engine.needs_for(model, phase),
                "{model} {phase:?}"
            );
        }
    }

    // Two bindings of one glob write one row a phase. The first decides
    // the single phase and reads the headers there: the union feeds the
    // loads over its word, and never under it.
    let text = r#"{
          "apiVersion": "smg.policy/v1",
          "profiles": {
            "keyed": { "filter": ["healthy"], "score": ["session-affinity"], "pick": "best" },
            "loaded": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" }
          },
          "bindings": [
            { "match": { "model": "m", "phase": ["single"] }, "profile": "keyed" },
            { "match": { "model": "m" }, "profile": "loaded" }
          ]
        }"#;
    let engine = doc(text).validate().expect("validates");
    assert_eq!(engine.needs_for("m", Phase::Single), headers);
    assert_eq!(
        engine.needs_rows(),
        phases.map(|phase| match phase {
            Phase::Single => ("m", phase, headers.or(loads)),
            _ => ("m", phase, loads),
        })
    );
}

#[test]
fn a_document_round_trips_and_names_its_mistakes() {
    let text = r#"{
          "apiVersion": "smg.policy/v1",
          "generation": 3,
          "profiles": {
            "default": { "filter": ["healthy", "not-overloaded"], "score": [{"least-load": {"weight": 0.3}}, "session-affinity"], "pick": "best" },
            "prefill": { "filter": ["healthy", "has-capacity"], "pick": "round-robin" },
            "decode": { "filter": ["healthy", "has-room"], "score": ["least-requests"], "pick": "best" }
          },
          "bindings": [
            { "match": { "model": "pd/*", "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "prefill", "decode": "decode" } },
            { "match": { "model": "*" }, "profile": "default" }
          ]
        }"#;
    let spec = doc(text);
    let engine = spec.validate().expect("validates");
    assert_eq!(engine.generation(), 3);
    assert!(engine.needs().loads && engine.needs().headers);
    let (_, prefill) = engine
        .profile_for("pd/x", Phase::Prefill)
        .expect("pd binding");
    assert_eq!(prefill.name, "prefill");
    let (_, decode) = engine
        .profile_for("pd/x", Phase::Decode)
        .expect("pd binding");
    assert_eq!(decode.name, "decode");
    let (_, single) = engine.profile_for("chat", Phase::Single).expect("default");
    assert_eq!(single.name, "default");
    let again: PolicySpec =
        serde_json::from_str(&serde_json::to_string(&spec).expect("serializes"))
            .expect("round trip");
    assert_eq!(again, spec);

    let mut bad = spec.clone();
    bad.profiles.get_mut("default").expect("default").pick = "best-of-all".into();
    assert_eq!(
        bad.validate().err(),
        Some(SpecError::UnknownPicker {
            profile: "default".into(),
            name: "best-of-all".into()
        })
    );
    let mut unknown_filter = spec.clone();
    unknown_filter
        .profiles
        .get_mut("default")
        .expect("default")
        .filter
        .push("wire-compatible".into());
    // The catalog names no filter of a gateway's wires: the deciders
    // read them before any profile runs, and a profile that lists one
    // is refused by its name.
    assert_eq!(
        unknown_filter.validate().err(),
        Some(SpecError::UnknownFilter {
            profile: "default".into(),
            name: "wire-compatible".into()
        })
    );
    let mut unbound = spec.clone();
    unbound.bindings[1].profile = Some("missing".into());
    assert_eq!(
        unbound.validate().err(),
        Some(SpecError::UnknownProfile {
            index: 1,
            name: "missing".into()
        })
    );
    let mut wrong = spec;
    wrong.api_version = "smg.policy/v0".into();
    assert!(matches!(wrong.validate(), Err(SpecError::ApiVersion(_))));
}

#[test]
fn a_kind_line_is_refused_naming_the_decision_and_an_unknown_field_by_name() {
    let with_kind = doc(r#"{ "apiVersion": "smg.policy/v1", "kind": "PolicySpec",
          "profiles": { "p": { "pick": "random" } } }"#);
    let err = with_kind.validate().expect_err("a kind line is refused");
    assert_eq!(err, SpecError::Kind("PolicySpec".into()));
    assert!(err.to_string().contains("no kind field"), "{err}");
    let serialized = serde_json::to_string(&with_kind).expect("serializes");
    assert!(!serialized.contains("kind"), "{serialized}");

    let err = parse_error(
        r#"{ "apiVersion": "smg.policy/v1", "plugins": [],
          "profiles": { "p": { "pick": "random" } } }"#,
    );
    assert!(err.contains("unknown field `plugins`"), "{err}");
    let err = parse_error(
        r#"{ "apiVersion": "smg.policy/v1",
          "profiles": { "p": { "pick": "random" } },
          "bindings": [ { "profile": "p", "fallback": "p" } ] }"#,
    );
    assert!(err.contains("unknown field `fallback`"), "{err}");
    let err = parse_error(
        r#"{ "apiVersion": "smg.policy/v1",
          "profiles": { "p": { "pick": "random" } },
          "bindings": [ { "profile": "p", "sessions": { "ttl": 600 } } ] }"#,
    );
    assert!(err.contains("unknown field `ttl`"), "{err}");
}

/// A key written twice in one of the document's maps is refused by name
/// where it is read: the profiles, the index and a binding's profiles.
/// A JSON document ran the last of the two before.
#[test]
fn a_map_key_written_twice_is_refused_by_name() {
    let documents = [
        (
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"random"},"p":{"pick":"round-robin"}}}"#,
            "p",
        ),
        (
            r#"{"apiVersion":"smg.policy/v1","index":{"a*":{},"a*":{"ttl_s":1}},"profiles":{"p":{"pick":"random"}}}"#,
            "a*",
        ),
        (
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"random"}},"bindings":[{"match":{"model":"*"},"profiles":{"single":"p","single":"p"}}]}"#,
            "single",
        ),
    ];
    for (document, key) in documents {
        let refused = parse_error(document);
        assert!(
            refused.contains(&format!("the key `{key}` is written twice")),
            "{refused}"
        );
    }
    // Each key once reads as before.
    let read = doc(
        r#"{"apiVersion":"smg.policy/v1","index":{"a*":{}},"profiles":{"p":{"pick":"random"},"q":{"pick":"round-robin"}},"bindings":[{"match":{"model":"*"},"profiles":{"single":"p"}}]}"#,
    );
    assert_eq!(read.profiles.len(), 2);
    assert_eq!(read.index.len(), 1);
    assert_eq!(read.bindings[0].profiles["single"], "p");
}

/// The YAML form of the three maps is refused in the same words: its
/// reader hands a key written twice to the map's own read, which kept
/// the last of the two before.
#[test]
fn a_map_key_written_twice_in_yaml_is_refused_by_name() {
    let documents = [
            (
                "apiVersion: smg.policy/v1\nprofiles:\n  p: {pick: random}\n  p: {pick: round-robin}\n",
                "p",
            ),
            (
                "apiVersion: smg.policy/v1\nindex:\n  \"a*\": {}\n  \"a*\": {ttl_s: 1}\nprofiles:\n  p: {pick: random}\n",
                "a*",
            ),
            (
                "apiVersion: smg.policy/v1\nprofiles:\n  p: {pick: random}\nbindings:\n  - match: {model: \"*\"}\n    profiles: {single: p, single: p}\n",
                "single",
            ),
        ];
    for (document, key) in documents {
        let refused = serde_yaml::from_str::<PolicySpec>(document)
            .expect_err("a key written twice")
            .to_string();
        assert!(
            refused.contains(&format!("the key `{key}` is written twice")),
            "{refused}"
        );
    }
}

#[test]
fn the_pair_presets_put_the_admission_filters_in_front_of_each_side() {
    let spec = PolicySpec::pair_presets("least_load", "power_of_two").expect("pair");
    assert!(spec.profiles["prefill"]
        .filter
        .contains(&"has-capacity".into()));
    assert!(spec.profiles["decode"].filter.contains(&"has-room".into()));
    let engine = spec.validate().expect("validates");
    let binding = engine.binding("m", Phase::Prefill).expect("bound");
    assert_eq!(binding.plan, Plan::PdPair);
    assert!(binding.decode_profile.is_some());
}

#[test]
fn a_plan_names_every_side_and_refuses_a_side_outside_it() {
    let mut spec = PolicySpec::pair_presets("least_load", "least_load").expect("pair");
    spec.bindings[0].profiles.remove("decode");
    assert_eq!(
        spec.validate().err(),
        Some(SpecError::SideUnnamed {
            index: 0,
            plan: "pd-pair",
            side: "decode"
        })
    );
    let mut one_profile = PolicySpec::pair_presets("least_load", "least_load").expect("pair");
    one_profile.bindings[0].profile = Some("prefill".into());
    assert_eq!(
        one_profile.validate().err(),
        Some(SpecError::ProfileUnderPlan {
            index: 0,
            plan: "pd-pair"
        })
    );
    let mut encode_under_pair = PolicySpec::pair_presets("least_load", "least_load").expect("pair");
    encode_under_pair.bindings[0]
        .profiles
        .insert("encode".into(), "prefill".into());
    assert_eq!(
        encode_under_pair.validate().err(),
        Some(SpecError::SideNotInPlan {
            index: 0,
            plan: "pd-pair",
            side: "encode".into()
        })
    );
    let mut odd_side = PolicySpec::pair_presets("least_load", "least_load").expect("pair");
    odd_side.bindings[0]
        .profiles
        .insert("tail".into(), "prefill".into());
    assert_eq!(
        odd_side.validate().err(),
        Some(SpecError::UnknownSide {
            index: 0,
            side: "tail".into()
        })
    );
    // A side under the single plan was read and ignored before; it is
    // refused now.
    let under_single = spec_with_binding(r#"{"profile":"p","profiles":{"decode":"q"}}"#);
    assert_eq!(
        under_single.validate().err(),
        Some(SpecError::SideNotInPlan {
            index: 0,
            plan: "single",
            side: "decode".into()
        })
    );
    let twice = spec_with_binding(r#"{"profile":"p","profiles":{"single":"q"}}"#);
    assert_eq!(
        twice.validate().err(),
        Some(SpecError::ProfileTwice { index: 0 })
    );
    let via_single_side = spec_with_binding(r#"{"profiles":{"single":"q"}}"#);
    let mut two_profiles_no_bindings = spec_with_binding(r#"{"profile":"p"}"#);
    two_profiles_no_bindings.bindings.clear();
    assert_eq!(
        two_profiles_no_bindings.validate().err(),
        Some(SpecError::NoBindings { profiles: 2 })
    );
    let mut with_deferred_stage = two_profiles_no_bindings.clone();
    with_deferred_stage
        .profiles
        .get_mut("q")
        .expect("profile q")
        .score
        .push(ScoreSpec::from("kv-pressure"));
    assert_eq!(
        with_deferred_stage.validate().err(),
        Some(SpecError::NoBindings { profiles: 2 }),
        "a shape refusal outranks what the build lacks"
    );
    two_profiles_no_bindings.profiles.remove("q");
    assert!(two_profiles_no_bindings.validate().is_ok());
    let engine = via_single_side
        .validate()
        .expect("the single side names the profile");
    assert_eq!(
        engine
            .profile_for("m", Phase::Single)
            .expect("bound")
            .1
            .name,
        "q"
    );
    let err = parse_error(
        r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"best"}},"bindings":[{"plan":"triple","profile":"p"}]}"#,
    );
    assert!(err.contains("unknown variant `triple`"), "{err}");
}

/// The document's `bounded_load` reaches the scorer: a decision through
/// the validated engine passes a loaded owner on, where the plain ring
/// would keep it.
#[test]
fn the_document_s_bounded_load_reaches_the_consistent_hash_scorer() {
    use crate::engine::{
        framework::Scratch,
        view::{Candidate, Constraints, Health, Identity, PoolFacts, RequestFacts, WorkerId},
    };
    let engine = doc(
        r#"{"apiVersion": "smg.policy/v1", "profiles": {"p": {"filter": ["healthy"],
                "score": [{"consistent-hash": {"bounded_load": {"factor": 1.0, "abs_margin": 0}}}],
                "pick": "best"}}}"#,
    )
    .validate()
    .expect("a bounded consistent-hash profile validates");
    let pool: Vec<Candidate> = (0..4u32)
        .map(|i| Candidate {
            position: i,
            id: WorkerId(100 + u64::from(i)),
            health: Health {
                healthy: true,
                ..Health::default()
            },
            ..Candidate::default()
        })
        .collect();
    let facts = RequestFacts {
        model: "m",
        session_key: Identity::hashed(5),
        seed: 1,
        ..RequestFacts::default()
    };
    let pool_facts = PoolFacts {
        pool_size: 4,
        generation: 1,
        ..PoolFacts::default()
    };
    let cx = Constraints::default();
    let mut scratch = Scratch::new();
    let first = |decision: &crate::engine::Decision| {
        decision
            .pick()
            .expect("a pick")
            .targets
            .as_slice()
            .first()
            .expect("a target")
            .position
    };
    let plain = engine.decide(&facts, &pool_facts, &pool, &cx, &mut scratch);
    let owner = first(&plain);
    let mut loaded = pool.clone();
    loaded[owner as usize].in_flight = 8;
    let bounded = engine.decide(&facts, &pool_facts, &loaded, &cx, &mut scratch);
    assert_ne!(
        first(&bounded),
        owner,
        "the document's rule passes the loaded owner on"
    );
}

/// The `epd` plan binds its three sides: the encode phase resolves to
/// the encode profile, the pair phases to theirs, and a single request
/// to none; a side left unnamed or a phase list leaving one out is
/// refused by name.
#[test]
fn the_epd_plan_binds_its_three_sides() {
    let epd =
        spec_with_binding(r#"{"plan":"epd","profiles":{"encode":"q","prefill":"p","decode":"p"}}"#);
    let engine = epd.validate().expect("the epd plan is in this build");
    let binding = engine.binding("m", Phase::Encode).expect("bound");
    assert_eq!(binding.plan, Plan::Epd { encode: 1 });
    assert_eq!((binding.profile, binding.decode_profile), (0, Some(0)));
    let side = |phase: Phase| engine.profile_for("m", phase).map(|(_, p)| p.name.clone());
    assert_eq!(side(Phase::Encode), Some("q".to_string()));
    assert_eq!(side(Phase::Prefill), Some("p".to_string()));
    assert_eq!(side(Phase::Decode), Some("p".to_string()));
    assert_eq!(side(Phase::Single), None, "no side of the plan");
    let presets = PolicySpec::epd_presets("consistent_hashing", "least_load", "round_robin")
        .expect("the epd presets");
    assert_eq!(presets.bindings[0].plan, Some(PlanName::Epd));
    assert_eq!(presets.profiles["encode"].score.len(), 1);
    assert!(presets.profiles["prefill"]
        .filter
        .contains(&"has-capacity".into()));
    let engine = presets.validate().expect("validates");
    assert!(engine.needs().ring && engine.needs().loads);
    assert!(matches!(
        engine.binding("m", Phase::Prefill).map(|b| b.plan),
        Some(Plan::Epd { .. })
    ));
    let no_encode = spec_with_binding(r#"{"plan":"epd","profiles":{"prefill":"p","decode":"p"}}"#);
    assert_eq!(
        no_encode.validate().err(),
        Some(SpecError::SideUnnamed {
            index: 0,
            plan: "epd",
            side: "encode"
        })
    );
    let two_phases = spec_with_binding(
        r#"{"match":{"phase":["prefill","decode"]},"plan":"epd","profiles":{"encode":"q","prefill":"p","decode":"p"}}"#,
    );
    assert_eq!(
        two_phases.validate().err(),
        Some(SpecError::PlanPhases {
            index: 0,
            plan: "epd",
            side: "encode"
        })
    );
}

#[test]
fn a_sampling_picker_without_a_scorer_is_refused() {
    let mut spec = PolicySpec::preset("power_of_two").expect("preset");
    spec.profiles
        .get_mut("power_of_two")
        .expect("profile")
        .score
        .clear();
    assert_eq!(
        spec.validate().err(),
        Some(SpecError::PickerNeedsScorer {
            profile: "power_of_two".into(),
            picker: "power-of-two".into()
        })
    );
    let count_first_form = doc(
        r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"power-of-two-count-first"}}}"#,
    );
    assert_eq!(
        count_first_form.validate().err(),
        Some(SpecError::PickerNeedsScorer {
            profile: "p".into(),
            picker: "power-of-two-count-first".into()
        })
    );
    let requests_form = doc(r#"{ "apiVersion": "smg.policy/v1",
          "profiles": { "p2": { "score": ["least-requests"], "pick": "power-of-two" } } }"#);
    let engine = requests_form
        .validate()
        .expect("the requests form validates");
    assert!(!engine.needs().loads);
}

#[test]
fn a_spec_without_bindings_binds_its_first_profile_to_everything() {
    let spec = doc(r#"{ "apiVersion": "smg.policy/v1",
          "profiles": { "only": { "filter": ["healthy"], "pick": "passthrough" } } }"#);
    let engine = spec.validate().expect("validates");
    let (_, profile) = engine
        .profile_for("anything", Phase::Decode)
        .expect("bound");
    assert_eq!(profile.name, "only");
    assert_eq!(
        PolicySpec {
            profiles: BTreeMap::new(),
            ..spec
        }
        .validate()
        .err(),
        Some(SpecError::Empty)
    );
}

#[test]
fn a_weight_or_parameter_out_of_range_is_refused_by_its_path() {
    let cases = [
        (r#"{"least-load":{"weight":-1}}"#, "weight"),
        (r#"{"least-load":{"weight":0}}"#, "weight"),
        (
            r#"{"least-load":{"kv_pressure_weight":-0.5}}"#,
            "kv_pressure_weight",
        ),
        (
            r#"{"least-load":{"mean_prefill_tokens":0}}"#,
            "mean_prefill_tokens",
        ),
        (
            r#"{"least-load":{"default_throughput":0}}"#,
            "default_throughput",
        ),
    ];
    for (score, name) in cases {
        assert_eq!(
            bad_field(&spec_with_score(score)),
            format!("profiles.p.score[0].least-load.{name}"),
            "{score}"
        );
    }
}

/// The scores combine by each weight's share of the weights' sum. A
/// profile whose sum is not finite, or has no finite reciprocal, is
/// refused at its score list: its shares would lose the order of every
/// row. A weight under the normal range beside one that is not passes.
#[test]
fn weights_whose_sum_gives_no_finite_share_are_refused_at_the_score_list() {
    for score in [
        r#"{"least-load":{"weight":1e-40}}"#,
        r#"{"least-load":{"weight":3e38}},{"least-requests":{"weight":3e38}}"#,
    ] {
        assert_eq!(
            bad_field(&spec_with_score(score)),
            "profiles.p.score",
            "{score}"
        );
    }
    spec_with_score(r#"{"least-load":{"weight":1e-40}},"least-requests""#)
        .validate()
        .expect("a sum in the normal range");
}

/// A profile with a stage this build does not carry is judged by the
/// weights the build does carry only where a part decides for the
/// whole. A part with no finite reciprocal defers with the profile:
/// the stage left out may bring the sum into range. An infinite part
/// is refused: no weight left out can bring it back.
#[test]
fn a_deferred_profile_is_refused_for_an_infinite_part_of_its_weights_alone() {
    let deferred = r#"{"prefix-affinity":{"decay":0.9}}"#;
    let small = format!(r#"{{"least-load":{{"weight":1e-40}}}},{deferred}"#);
    let (path, what) = not_in_build(&spec_with_score(&small));
    assert_eq!(path, "profiles.p.score[1].prefix-affinity");
    assert!(what.contains("prefix-affinity"), "{what}");
    let large = format!(
        r#"{{"least-load":{{"weight":3e38}}}},{{"least-requests":{{"weight":3e38}}}},{deferred}"#
    );
    assert_eq!(bad_field(&spec_with_score(&large)), "profiles.p.score");
}

#[test]
fn a_misspelt_or_unknown_scorer_is_refused_by_name() {
    // A key that is no built-in's parses, since a catalog may add the
    // name; the validator refuses it by name, whatever is written under
    // it.
    match spec_with_score(r#"{"least_load":{"kv_pressure_weight":5}}"#).validate() {
        Err(SpecError::UnknownScorer { name, .. }) => assert_eq!(name, "least_load"),
        other => panic!("validated as {other:?}"),
    }
    // With parameters, a built-in's one-key map goes to its parameter
    // struct whole, so the message names the key instead of "no variant
    // matched".
    let err = parse_error(
        r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":[{"least-requests":{"kv_pressure_weight":5}}],"pick":"best"}}}"#,
    );
    assert!(err.contains("unknown field `kv_pressure_weight`"), "{err}");
    let err = parse_error(
        r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":[{"least-load":{"weight":1},"least-requests":{}}],"pick":"best"}}}"#,
    );
    assert!(err.contains("single key"), "{err}");
    match spec_with_score(r#""least_load""#).validate() {
        Err(SpecError::UnknownScorer { name, .. }) => assert_eq!(name, "least_load"),
        other => panic!("validated as {other:?}"),
    }
}

#[test]
fn a_model_glob_is_a_single_leading_or_trailing_star() {
    for model in ["llama*-chat", "*-7b*", "", "**"] {
        let spec = spec_with_binding(&format!(
            r#"{{"match":{{"model":{model:?}}},"profile":"p"}}"#
        ));
        assert!(
            matches!(
                spec.validate(),
                Err(SpecError::BadModelGlob { index: 0, .. })
            ),
            "{model:?}"
        );
    }
    for model in ["*", "llama*", "*-chat", "llama-3"] {
        let spec = spec_with_binding(&format!(
            r#"{{"match":{{"model":{model:?}}},"profile":"p"}}"#
        ));
        assert!(spec.validate().is_ok(), "{model:?}");
    }
}

#[test]
fn a_pair_binding_names_both_phases_or_none() {
    let spec = spec_with_binding(
        r#"{"match":{"phase":["prefill"]},"plan":"pd-pair","profiles":{"prefill":"p","decode":"p"}}"#,
    );
    assert_eq!(
        spec.validate().err(),
        Some(SpecError::PlanPhases {
            index: 0,
            plan: "pd-pair",
            side: "decode"
        })
    );
    let both = spec_with_binding(
        r#"{"match":{"phase":["prefill","decode"]},"plan":"pd-pair","profiles":{"prefill":"p","decode":"p"}}"#,
    );
    assert!(both.validate().is_ok());
}

#[test]
fn the_view_refuses_denied_duplicate_and_malformed_headers() {
    assert_eq!(
        bad_field(&spec_with_view(r#"{"headers":["Authorization"]}"#)),
        "view.headers[0]"
    );
    assert_eq!(
        spec_with_view(r#"{"headers":["x-a","cookie"]}"#)
            .validate()
            .err(),
        Some(SpecError::DeniedHeader {
            path: "view.headers[1]".into(),
            name: "cookie".into()
        })
    );
    assert_eq!(
        spec_with_view(r#"{"headers":["x-a","x-a"]}"#)
            .validate()
            .err(),
        Some(SpecError::Duplicate {
            path: "view.headers".into(),
            name: "x-a".into()
        })
    );
    let tenant = spec_with_view(r#"{"headers":["x-smg-tenant-id"]}"#);
    assert!(tenant.validate().is_ok());
    assert_eq!(
        tenant.validate_with(&["X-SMG-Tenant-Id"]).err(),
        Some(SpecError::DeniedHeader {
            path: "view.headers[0]".into(),
            name: "x-smg-tenant-id".into()
        })
    );
    let engine = spec_with_view(r#"{"headers":["x-session-id"]}"#)
        .validate()
        .expect("an allow-listed header");
    assert!(engine.needs().headers);
}

#[test]
fn a_generation_is_refused_for_a_header_the_host_denies() {
    let tenant = spec_with_view(r#"{"headers":["x-smg-tenant-id"]}"#);
    assert!(AppliedSpec::validate(&tenant).is_ok());
    assert_eq!(
        AppliedSpec::validate_with(&tenant, &["X-SMG-Tenant-Id"]).err(),
        Some(SpecError::DeniedHeader {
            path: "view.headers[0]".into(),
            name: "x-smg-tenant-id".into()
        })
    );
}

#[test]
fn a_generation_declares_the_needs_of_its_engine() {
    for name in PRESETS {
        let document = PolicySpec::preset(name).expect(name);
        let applied = AppliedSpec::validate(&document).expect(name);
        assert_eq!(applied.needs(), applied.engine().needs(), "{name}");
    }
    let cache_aware = PolicySpec::preset("cache_aware").expect("a preset of this build");
    let needs = AppliedSpec::validate(&cache_aware)
        .expect("a preset validates")
        .needs();
    assert!(needs.loads && needs.events && needs.affinity);
}

#[test]
fn the_view_lists_at_most_256_headers() {
    let with_headers = |count: usize| {
        let names: Vec<String> = (0..count).map(|i| format!("\"x-h{i}\"")).collect();
        spec_with_view(&format!(r#"{{"headers":[{}]}}"#, names.join(",")))
    };
    let engine = with_headers(MAX_HEADERS)
        .validate()
        .expect("256 headers fill the byte of positions");
    assert!(engine.needs().headers);
    assert_eq!(
        with_headers(MAX_HEADERS + 1).validate().err(),
        Some(SpecError::TooManyHeaders {
            count: 257,
            max: 256
        })
    );
}

#[test]
fn the_view_checks_block_size_boundaries_and_the_hash_cap() {
    assert_eq!(
        bad_field(&spec_with_view(r#"{"block_size":0}"#)),
        "view.block_size"
    );
    assert_eq!(
        bad_field(&spec_with_view(r#"{"boundaries":[0,16]}"#)),
        "view.boundaries[0]"
    );
    for boundaries in ["[256,128]", "[256,256]"] {
        assert_eq!(
            bad_field(&spec_with_view(&format!(
                r#"{{"boundaries":{boundaries}}}"#
            ))),
            "view.boundaries[1]",
            "{boundaries}"
        );
    }
    assert_eq!(
        bad_field(&spec_with_view(r#"{"max_block_hashes":0}"#)),
        "view.max_block_hashes"
    );
    // One frame of the Decide call carries 8,190 hashes: that cap is
    // taken, and one above it is refused by the field's name.
    let cap = |hashes: u32| spec_with_view(&format!(r#"{{"max_block_hashes":{hashes}}}"#));
    assert!(cap(MAX_BLOCK_HASHES).validate().is_ok());
    let refused = cap(MAX_BLOCK_HASHES + 1)
        .validate()
        .expect_err("a cap one frame cannot carry");
    assert_eq!(
        refused.to_string(),
        "view.max_block_hashes = 8191 is not at most 8190, the hashes one Decide frame carries"
    );
    let spec = spec_with_view(r#"{"block_size":128,"boundaries":[256,1024,4096]}"#);
    assert!(spec.validate().is_ok());
    assert_eq!(spec.view.max_block_hashes, 256);
}

#[test]
fn the_view_lists_are_capped_named_and_listed_once() {
    let list = |count: usize| {
        let names: Vec<String> = (0..count).map(|n| format!("\"n{n}\"")).collect();
        format!("[{}]", names.join(","))
    };
    let caps = [
        ("labels", MAX_LABELS),
        ("zones", MAX_ZONES),
        ("engines", MAX_ENGINES),
    ];
    for (field, max) in caps {
        let at_cap = spec_with_view(&format!(r#"{{"{field}":{}}}"#, list(max)));
        assert!(at_cap.validate().is_ok(), "{field}");
        let over = spec_with_view(&format!(r#"{{"{field}":{}}}"#, list(max + 1)));
        assert!(
            matches!(
                over.validate(),
                Err(SpecError::TooManyListed { path, count, max: cap })
                    if path == format!("view.{field}") && count == max + 1 && cap == max
            ),
            "{field}"
        );
        let unnamed = spec_with_view(&format!(r#"{{"{field}":["a",""]}}"#));
        assert_eq!(bad_field(&unnamed), format!("view.{field}[1]"));
        let twice = spec_with_view(&format!(r#"{{"{field}":["a","b","a"]}}"#));
        assert!(
            matches!(
                twice.validate(),
                Err(SpecError::Duplicate { path, name })
                    if path == format!("view.{field}") && name == "a"
            ),
            "{field}"
        );
    }
    // A label is a key, or a key and a value split at the first `=`:
    // neither side is empty, and the value may hold a `=` of its own.
    for label in ["=gold", "tier="] {
        let empty = spec_with_view(&format!(r#"{{"labels":["tier","{label}"]}}"#));
        assert_eq!(bad_field(&empty), "view.labels[1]", "{label}");
    }
    let spec = spec_with_view(r#"{"labels":["tier","tier=gold","note=a=b"]}"#);
    assert!(spec.validate().is_ok());
}

#[test]
fn a_view_that_declares_no_list_is_written_as_before() {
    let written = serde_json::to_string(&ViewSpec::default()).expect("the view serializes");
    for field in ["labels", "zones", "engines"] {
        assert!(!written.contains(field), "{written}");
    }
    let spec = spec_with_view(r#"{"labels":["tier=gold"],"zones":["east"],"engines":["alpha"]}"#);
    let written = serde_json::to_string(&spec.view).expect("the view serializes");
    let read: ViewSpec = serde_json::from_str(&written).expect("the view reads back");
    assert_eq!(read, spec.view);
    assert_eq!(
        (read.labels.len(), read.zones.len(), read.engines.len()),
        (1, 1, 1)
    );
}

#[test]
fn the_lists_read_a_worker_by_position_and_by_bit() {
    let labels = |pairs: &[(&str, &str)]| -> BTreeMap<String, String> {
        let owned = |(key, value): &(&str, &str)| ((*key).to_string(), (*value).to_string());
        pairs.iter().map(owned).collect()
    };
    let spec = spec_with_view(
        r#"{"labels":["tier","tier=gold","note=a=b"],"zones":["east","west"],"engines":["alpha","beta"]}"#,
    );
    let applied = AppliedSpec::validate(&spec).expect("the document validates");
    let lists = applied.lists();
    assert!(!lists.is_empty());
    let held = labels(&[("tier", "gold"), ("note", "a=b")]);
    let gold = lists.positions("beta", Some("west"), &held);
    assert_eq!((gold.engine_kind, gold.zone, gold.labels), (2, 2, 0b111));
    assert_eq!(lists.outside("beta", Some("west")), (false, false));
    let held = labels(&[("tier", "silver"), ("note", "a")]);
    let silver = lists.positions("alpha", Some("east"), &held);
    assert_eq!(
        (silver.engine_kind, silver.zone, silver.labels),
        (1, 1, 0b001)
    );
    // Each position is read from its own list: this worker's family
    // stands second among the families, and its zone first among the
    // zones.
    let apart = lists.positions("beta", Some("east"), &labels(&[]));
    assert_eq!((apart.engine_kind, apart.zone), (2, 1));
    // A bit is the selector's place in the list, whatever the keys'
    // order: two selectors of one key stand apart here.
    let mixed = spec_with_view(r#"{"labels":["z","b=1","m","b","a=2","q=x"]}"#);
    let mixed = AppliedSpec::validate(&mixed).expect("the document validates");
    let held = labels(&[("a", "2"), ("b", "1"), ("c", "9"), ("m", ""), ("q", "y")]);
    assert_eq!(mixed.lists().positions("", None, &held).labels, 0b01_1110);
    // A name the list does not hold reads 0, and the narrow reader says
    // it stands outside. No name reads 0 and is not outside.
    let stray = lists.positions("gamma", Some("north"), &labels(&[]));
    assert_eq!((stray.engine_kind, stray.zone, stray.labels), (0, 0, 0));
    assert_eq!(lists.outside("gamma", Some("north")), (true, true));
    for zone in [None, Some("")] {
        assert_eq!(
            lists.positions("", zone, &labels(&[])),
            ViewListPositions::default()
        );
        assert_eq!(lists.outside("", zone), (false, false));
    }
    // The narrow reader answers its two bits apart, the family's first.
    assert_eq!(lists.outside("gamma", Some("west")), (true, false));
    assert_eq!(lists.outside("beta", Some("north")), (false, true));
    // No list: every worker reads 0 and none is outside.
    let none = AppliedSpec::validate(&spec_with_view("{}")).expect("the document validates");
    assert!(none.lists().is_empty() && AppliedSpec::unbound().lists().is_empty());
    let read = none.lists().positions("beta", Some("west"), &held);
    assert_eq!(read, ViewListPositions::default());
    assert_eq!(none.lists().outside("beta", Some("west")), (false, false));
    // Lists name a family or a zone, or neither: label selectors alone
    // name neither, and so does no list.
    assert!(lists.names_a_family_or_zone());
    assert!(!mixed.lists().names_a_family_or_zone());
    assert!(!none.lists().names_a_family_or_zone());
    // A family's position and a zone's are read from their own lists.
    let crossed = lists.positions("alpha", Some("west"), &labels(&[]));
    assert_eq!((crossed.engine_kind, crossed.zone), (1, 2));
}

#[test]
fn the_index_map_is_per_model_glob_and_a_hash_index_needs_boundaries() {
    let with_index = |index: &str, view: &str| {
        doc(&format!(
            r#"{{"apiVersion":"smg.policy/v1","view":{view},"index":{index},"profiles":{{"p":{{"pick":"best"}}}}}}"#
        ))
    };
    assert_eq!(bad_field(&with_index(r#"{"a*b":{}}"#, "{}")), "index.a*b");
    // Each count is above zero, refused by its own name, the first
    // written in the type's order where two are zero.
    for count in ["ttl_s", "max_size", "sweep_s"] {
        let index = format!(r#"{{"*":{{"{count}":0}}}}"#);
        assert_eq!(
            bad_field(&with_index(&index, "{}")),
            format!("index.*.{count}")
        );
    }
    assert_eq!(
        bad_field(&with_index(r#"{"*":{"sweep_s":0,"ttl_s":0}}"#, "{}")),
        "index.*.ttl_s"
    );
    assert_eq!(
        with_index(r#"{"*":{"kind":"hash"}}"#, "{}")
            .validate()
            .err(),
        Some(SpecError::HashIndexNeedsBoundaries { model: "*".into() })
    );
    let hashed = with_index(
        r#"{"*":{"kind":"hash","ttl_s":300},"chat-*":{"max_size":1024}}"#,
        r#"{"boundaries":[512,2048]}"#,
    );
    assert!(hashed.validate().is_ok());
    assert_eq!(hashed.index["chat-*"].sweep_s, 120);
}

/// A zero count of an index key is refused whole: by its path, its
/// value and its reason.
#[test]
fn a_zero_count_of_an_index_key_is_refused_by_path_value_and_reason() {
    for count in ["ttl_s", "max_size", "sweep_s"] {
        let spec = doc(&format!(
            r#"{{"apiVersion":"smg.policy/v1","index":{{"*":{{"{count}":0}}}},"profiles":{{"p":{{"pick":"best"}}}}}}"#
        ));
        assert_eq!(
            spec.validate().err(),
            Some(SpecError::BadField {
                path: format!("index.*.{count}"),
                value: "0".to_owned(),
                reason: "above zero",
            })
        );
    }
}

/// A document that names no source reads each worker by what it has:
/// its stage asks a host for cache events and for the prefixes sent.
#[test]
fn a_source_left_out_reads_cache_events_and_sent_prefixes() {
    let needs = |score: &str| spec_with_score(score).validate().expect(score).needs();
    let left_out = needs(r#"{"prefix-affinity":{}}"#);
    assert_eq!(
        left_out,
        needs(r#"{"prefix-affinity":{"source":"merged"}}"#)
    );
    assert!(left_out.events && left_out.sent_prefixes);
    let index = needs(r#"{"prefix-affinity":{"source":"index"}}"#);
    assert!(index.events && !index.sent_prefixes);
}

#[test]
fn least_cost_reads_the_compared_rows_whole_behind_a_rank() {
    let spec = spec_with_score(
        r#"{"least-cost":{"terms":{"load.running":7},"unreported":"missing-as-zero"}}"#,
    );
    let catalog = StageRegistry::builtin();
    let entry = &spec.profiles["p"].score[0];
    let from = StageInputs {
        catalog: &catalog,
        reports: LoadReportReader::new(None),
        signals: &[],
    };
    let Scored::Typed(stage) = scored(
        entry,
        "p",
        &At::new("profiles.p.score[0]"),
        from,
        SentSettings::default(),
        &mut SentSlot::default(),
    )
    .expect("a built-in scorer") else {
        panic!("a typed scorer");
    };
    let Resolved::Built(scorer) = scorer_of(
        &stage,
        Gates::of(None, None, None),
        &spec.view,
        &[],
        || panic!("no term reads sends"),
        &At::new("profiles.p.score[0].least-cost"),
    )
    .expect("valid parameters") else {
        panic!("least-cost is built");
    };
    assert_eq!(scorer.name(), "least-cost");
    assert!(scorer.row_wise());
    assert!(!scorer.front_wise());
}

/// Only a `sent.` term makes the profile's count of sends, and a time
/// limit on the sends needs one where `least-cost` is the only scorer.
#[test]
fn least_cost_keeps_a_count_of_sends_only_for_a_sent_term() {
    for (term, limit, want) in [
        ("candidate.in_flight", None, Ok(0)),
        ("sent.tokens", None, Ok(1)),
        ("sent.requests", Some(43), Ok(1)),
        (
            "candidate.in_flight",
            Some(43),
            Err("profiles.p.sent_ttl_ms"),
        ),
    ] {
        let mut spec = spec_with_score(&format!(r#"{{"least-cost":{{"terms":{{"{term}":7}}}}}}"#));
        spec.profiles.get_mut("p").expect("the profile").sent_ttl_ms = limit;
        let counts = match spec.validate() {
            Ok(engine) => Ok(engine.sent_credits.len()),
            Err(SpecError::BadField { path, .. }) => Err(path),
            Err(other) => panic!("{term} under {limit:?}: {other:?}"),
        };
        assert_eq!(
            counts,
            want.map_err(str::to_owned),
            "{term} under {limit:?}"
        );
    }
}

/// Profiles under one time limit and one report delay read one count
/// of sends. A profile under another limit or another delay keeps a
/// count of its own.
#[test]
fn profiles_under_one_sent_setting_read_one_count_of_sends() {
    for (sent, want) in [
        ((None, None), 1),
        ((Some(43), None), 2),
        ((None, Some(5)), 2),
    ] {
        let mut spec = doc(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":["least-load"],"pick":"best"},"q":{"score":["least-load"],"pick":"best"}},"bindings":[{"match":{"model":"m"},"profile":"p"},{"match":{"model":"*"},"profile":"q"}]}"#,
        );
        let q = spec.profiles.get_mut("q").expect("the profile");
        (q.sent_ttl_ms, q.sent_report_delay_ms) = sent;
        let engine = spec.validate().expect("a valid document");
        assert_eq!(engine.sent_credits.len(), want, "q under {sent:?}");
        // Both scorers hold a count of the list, the ones the engine
        // feeds: the list holds each count once beside its readers.
        let readers = engine
            .sent_credits
            .iter()
            .map(|count| Arc::strong_count(count) - 1);
        assert_eq!(readers.sum::<usize>(), 2, "q under {sent:?}");
    }
}

/// A later document's count takes over the standing count under its
/// two settings, whichever profile made that one. Where none stands
/// under them it takes the count its maker's name made, unless a
/// count of the later document took that one, and then the first
/// standing count that none took.
#[test]
fn a_later_count_takes_the_standing_one_by_its_settings_then_by_its_name() {
    let both = r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":["least-load"],"pick":"best"},"q":{"score":["least-load"],"pick":"best"}},"bindings":[{"match":{"model":"m"},"profile":"p"},{"match":{"model":"*"},"profile":"q"}]}"#;
    let q_alone = r#"{"apiVersion":"smg.policy/v1","profiles":{"q":{"score":["least-load"],"pick":"best"}},"bindings":[{"match":{"model":"*"},"profile":"q"}]}"#;
    let worker = crate::engine::WorkerId(7);
    let target = crate::engine::Target {
        id: worker,
        ..Default::default()
    };
    // A document with the limits its `p` and its `q` set.
    let built = |document: &str, limits: [Option<u32>; 2]| {
        let mut spec = doc(document);
        for (name, limit) in ["p", "q"].into_iter().zip(limits) {
            if let Some(profile) = spec.profiles.get_mut(name) {
                profile.sent_ttl_ms = limit;
            }
        }
        spec.validate().expect("a valid document")
    };
    // The limits of the standing `p` and `q`, the later document and
    // its limits, and the sends the later counts hold, in the order
    // of their makers' names. The send is in every standing count.
    for (was, later, limits, want) in [
        // The same document again.
        ([None, None], both, [None, None], vec![1]),
        // `p` made the shared count and leaves: `q` keeps the send.
        ([None, None], q_alone, [None, None], vec![1]),
        // Both under a new limit: the name `p` carries the count.
        ([None, None], both, [Some(43), Some(43)], vec![1]),
        // `p` leaves the shared count for a limit of its own: `q`
        // keeps the send and `p` starts empty.
        ([None, None], both, [Some(43), None], vec![0, 1]),
        // `p` takes the limit `q` had and `q` a new one: `p` takes the
        // count under its limit, and `q` does not take that count too
        // by its name: it takes the count `p` made, which is left.
        ([Some(1), Some(9)], both, [Some(9), Some(43)], vec![1, 1]),
    ] {
        let standing = built(both, was);
        standing.on_dispatched(
            &crate::tracking::Lease::default(),
            &target,
            crate::tracking::Side::Single,
            0,
        );
        let next = built(later, limits);
        next.succeed(&standing);
        let held: Vec<_> = next
            .sent_credits
            .iter()
            .map(|count| count.credits().held(worker))
            .collect();
        assert_eq!(held, want, "{was:?}, then {later} under {limits:?}");
    }
}

/// A later spec keeps the stage objects of each profile whose name and
/// text did not change, and builds every other profile's anew. A count
/// of sends stands where a profile that reads it is kept, whichever of
/// its other readers changed, left or came. A spec that changes
/// `signals` builds every profile anew, and so does one that changes
/// the view's age limit of load reports.
#[cfg(feature = "in-process")]
#[test]
fn a_later_spec_keeps_the_stages_of_a_profile_whose_text_did_not_change() {
    let standing = AppliedSpec::validate(&doc(KEPT)).expect("a valid document");
    type Edit = fn(&mut PolicySpec);
    let filter_of_q: Edit = |spec| {
        let q = spec.profiles.get_mut("q").expect("the profile");
        q.filter.push("healthy".into());
    };
    let pick_of_r: Edit = |spec| {
        let r = spec.profiles.get_mut("r").expect("the profile");
        r.pick = "round-robin".into();
    };
    // The same stages by name under another weight: the text alone moved.
    let weight_of_q: Edit = |spec| {
        let q = spec.profiles.get_mut("q").expect("the profile");
        let score = serde_json::from_str(r#"[{"least-load":{"weight":2}}]"#);
        q.score = score.expect("a score");
    };
    // The first reader of the count goes: `q` keeps the count.
    let without_p: Edit = |spec| {
        spec.profiles.remove("p");
        spec.bindings.remove(0);
    };
    // A later reader of the count goes: `p` keeps the count.
    let without_q: Edit = |spec| {
        spec.profiles.remove("q");
        spec.bindings.remove(1);
    };
    // A reader of the count comes: the others and the count stand.
    let with_s: Edit = |spec| {
        let s = serde_json::from_str(r#"{"score":["least-load"],"pick":"best"}"#);
        spec.profiles.insert("s".into(), s.expect("a profile"));
        let bound = serde_json::from_str(r#"{"match":{"model":"s"},"profile":"s"}"#);
        spec.bindings.insert(0, bound.expect("a binding"));
    };
    let view: Edit = |spec| spec.view.max_report_age_ms = Some(900);
    let signals: Edit = |spec| {
        let signal = serde_json::from_str(r#"{"unit":"tokens"}"#);
        spec.signals
            .insert("free".into(), signal.expect("a signal"));
    };
    // The edit, the profiles kept and whether the count is.
    let cases: [(Edit, [bool; 3], bool); 9] = [
        (|_| {}, [true; 3], true),
        (filter_of_q, [true, false, true], true),
        (weight_of_q, [true, false, true], true),
        (pick_of_r, [true, true, false], true),
        (without_p, [false, true, true], true),
        (without_q, [true, false, true], true),
        (with_s, [true; 3], true),
        (view, [false; 3], false),
        (signals, [false; 3], false),
    ];
    for (at, (edit, profiles, count)) in cases.into_iter().enumerate() {
        let mut later = doc(KEPT);
        edit(&mut later);
        let mut next = AppliedSpec::validate(&later).expect("a valid document");
        next.keep_from(&standing);
        assert_eq!(kept(&next, &standing), (profiles, count), "case {at}");
    }
}

/// A later spec whose view differs in its header list alone keeps its
/// profiles, however the list changed: it grew, it shrank, or it
/// changed inside. Each other field of the view, changed alone, builds
/// every profile anew, beside the same header list and beside another.
#[cfg(feature = "in-process")]
#[test]
fn a_view_that_differs_in_its_header_list_alone_keeps_every_profile() {
    let listing = |headers: &[&str]| {
        let mut spec = doc(KEPT);
        spec.view.headers = headers.iter().map(|name| (*name).to_owned()).collect();
        spec
    };
    let applied = |spec: &PolicySpec| AppliedSpec::validate(spec).expect("a valid document");
    // The standing list and the later one.
    let lists: [(&[&str], &[&str]); 9] = [
        (&["x-a"], &["x-a"]),
        (&["x-a"], &["x-a", "x-b"]),
        (&["x-a", "x-b"], &["x-a"]),
        (&[], &["x-a"]),
        (&["x-a"], &[]),
        (&["x-a", "x-b"], &["x-b", "x-a"]),
        (&["x-a", "x-b"], &["x-b"]),
        (&["x-b"], &["x-a", "x-b"]),
        (&["x-a", "x-b"], &["x-a", "x-c"]),
    ];
    for (at, (stood, later)) in lists.into_iter().enumerate() {
        let standing = applied(&listing(stood));
        let mut next = applied(&listing(later));
        next.keep_from(&standing);
        assert_eq!(kept(&next, &standing), ([true; 3], true), "lists {at}");
    }
    // Each other field of the view, changed alone.
    type Edit = fn(&mut ViewSpec);
    let fields: [Edit; 7] = [
        |view| view.block_size = Some(32),
        |view| view.boundaries = vec![64],
        |view| view.max_block_hashes = 512,
        |view| view.max_report_age_ms = Some(900),
        |view| view.labels = vec!["tier".to_owned()],
        |view| view.zones = vec!["east".to_owned()],
        |view| view.engines = vec!["alpha".to_owned()],
    ];
    let standing = applied(&listing(&["x-a"]));
    for (at, edit) in fields.into_iter().enumerate() {
        for later in [&["x-a"][..], &["x-b", "x-a"]] {
            let mut other = listing(later);
            edit(&mut other.view);
            let mut next = applied(&other);
            next.keep_from(&standing);
            let none = ([false; 3], false);
            assert_eq!(kept(&next, &standing), none, "field {at} beside {later:?}");
        }
    }
    // A list that keeps its count and changes a name builds anew too: each
    // of the four other lists is compared by what it holds.
    let full = |edit: Edit| {
        let mut spec = listing(&["x-a"]);
        spec.view.boundaries = vec![64];
        spec.view.labels = vec!["tier".to_owned()];
        spec.view.zones = vec!["east".to_owned()];
        spec.view.engines = vec!["alpha".to_owned()];
        edit(&mut spec.view);
        spec
    };
    let swaps: [Edit; 4] = [
        |view| view.boundaries = vec![128],
        |view| view.labels = vec!["rank".to_owned()],
        |view| view.zones = vec!["west".to_owned()],
        |view| view.engines = vec!["beta".to_owned()],
    ];
    let standing = applied(&full(|_| {}));
    for (at, swap) in swaps.into_iter().enumerate() {
        let mut next = applied(&full(swap));
        next.keep_from(&standing);
        assert_eq!(kept(&next, &standing), ([false; 3], false), "swap {at}");
    }
}

/// Three profiles for a case of the keep: `p` and `q` read one count of
/// sends, and `r` reads none.
#[cfg(feature = "in-process")]
const KEPT: &str = r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":["least-load"],"pick":"best"},"q":{"score":["least-load"],"pick":"best"},"r":{"pick":"random"}},"bindings":[{"match":{"model":"m"},"profile":"p"},{"match":{"model":"n"},"profile":"q"},{"match":{"model":"*"},"profile":"r"}]}"#;

/// Which of [`KEPT`]'s three profiles `next` holds as `standing`'s own
/// objects, each found by its name, and whether its first count of
/// sends is `standing`'s.
#[cfg(feature = "in-process")]
fn kept(next: &AppliedSpec, standing: &AppliedSpec) -> ([bool; 3], bool) {
    let (new, old) = (next.engine(), standing.engine());
    let named = |engine: &Engine, name: &str| {
        let mut profiles = engine.profiles().iter();
        profiles.find(|profile| profile.name == name).cloned()
    };
    let same = |name| match (named(new, name), named(old, name)) {
        (Some(new), Some(old)) => Arc::ptr_eq(&new, &old),
        _ => false,
    };
    let count = Arc::ptr_eq(&new.sent_credits[0], &old.sent_credits[0]);
    (["p", "q", "r"].map(same), count)
}

/// The keep stands between two specs validated over the stock table,
/// or over one table of a host's own as it stood at both validations.
/// Two tables of a host that hold the same names keep nothing, and
/// neither does one table that took a stage between the two.
#[cfg(feature = "in-process")]
#[test]
fn stages_are_kept_over_the_stock_table_or_one_table_as_it_stood() {
    let own = || {
        let mut own = StageRegistry::builtin();
        let added = own.filter("guest", || Box::new(ReadsHashes("guest")));
        added.expect("a free name");
        own
    };
    let over = |stages: &StageRegistry| {
        let spec = AppliedSpec::validate_in(&doc(KEPT), stages, &[]);
        spec.expect("a valid document")
    };
    let all = ([true; 3], true);
    let none = ([false; 3], false);
    let keeps = |standing: &AppliedSpec, stages: &StageRegistry| {
        let mut next = over(stages);
        next.keep_from(standing);
        kept(&next, standing)
    };
    // Two stock tables, each built for its validation.
    let stock = over(&StageRegistry::builtin());
    assert_eq!(keeps(&stock, &StageRegistry::builtin()), all);
    // One table of a host's own, at two validations.
    let mut table = own();
    let hosted = over(&table);
    assert_eq!(keeps(&hosted, &table), all);
    // The stock table behind a host's, and another table of the host's
    // under the same names.
    assert_eq!(keeps(&hosted, &StageRegistry::builtin()), none);
    assert_eq!(keeps(&hosted, &own()), none);
    // The one table again, behind a stage it took since.
    let added = table.filter("late", || Box::new(ReadsHashes("late")));
    added.expect("a free name");
    assert_eq!(keeps(&hosted, &table), none);
}

/// Each of a table's six adders puts the table under a number of its
/// own. So a table that took a stage by any of them keeps nothing
/// behind the stock table, behind another table under the same names,
/// or behind itself as it stood before a later stage.
#[cfg(feature = "in-process")]
#[test]
fn a_table_goes_under_a_number_of_its_own_by_each_of_its_six_adders() {
    type Add = fn(&mut StageRegistry, &'static str);
    let adders: [(&str, Add); 6] = [
        ("filter", |table, name| {
            let added = table.filter(name, move || Box::new(ReadsHashes(name)));
            added.expect("a free name");
        }),
        ("filter_with", |table, name| {
            let added = table.filter_with(name, move |_| Ok(Box::new(ReadsHashes(name))));
            added.expect("a free name");
        }),
        ("scorer", |table, name| {
            let added = table.scorer(name, move || Box::new(ReadsHashes(name)));
            added.expect("a free name");
        }),
        ("scorer_with", |table, name| {
            let added = table.scorer_with(name, move |_| Ok(Box::new(ReadsHashes(name))));
            added.expect("a free name");
        }),
        ("picker", |table, name| {
            let added = table.picker(name, move || Box::new(ReadsHashes(name)));
            added.expect("a free name");
        }),
        ("picker_with", |table, name| {
            let added = table.picker_with(name, move |_| Ok(Box::new(ReadsHashes(name))));
            added.expect("a free name");
        }),
    ];
    let over = |stages: &StageRegistry| {
        let spec = AppliedSpec::validate_in(&doc(KEPT), stages, &[]);
        spec.expect("a valid document")
    };
    let (all, none) = (([true; 3], true), ([false; 3], false));
    let keeps = |standing: &AppliedSpec, stages: &StageRegistry| {
        let mut next = over(stages);
        next.keep_from(standing);
        kept(&next, standing)
    };
    for (adder, add) in adders {
        let own = || {
            let mut own = StageRegistry::builtin();
            add(&mut own, "guest");
            own
        };
        let mut table = own();
        let hosted = over(&table);
        assert_eq!(keeps(&hosted, &table), all, "{adder}: the one table");
        let stock = StageRegistry::builtin();
        assert_eq!(keeps(&hosted, &stock), none, "{adder}: the stock table");
        assert_eq!(keeps(&hosted, &own()), none, "{adder}: another table");
        add(&mut table, "late");
        assert_eq!(keeps(&hosted, &table), none, "{adder}: a stage since");
    }
}

/// A profile is kept only where its stages bear the standing profile's
/// names in their order: a factory that builds a stage of another name
/// at a later call has its profile built new, and kept once the name
/// stands again.
#[cfg(feature = "in-process")]
#[test]
fn a_profile_whose_factory_builds_a_stage_of_another_name_is_built_new() {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

    let calls = AtomicUsize::new(0);
    let mut own = StageRegistry::builtin();
    // The registration asks the name once, then each validation builds.
    let added = own.picker("turns", move || {
        let call = calls.fetch_add(1, Relaxed);
        Box::new(ReadsHashes(
            ["turns", "turns", "another", "another"][call.min(3)],
        ))
    });
    added.expect("a free name");
    let document = doc(r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"turns"}}}"#);
    let validated = || AppliedSpec::validate_in(&document, &own, &[]);
    let same = |next: &AppliedSpec, standing: &AppliedSpec| {
        let (new, old) = (next.engine(), standing.engine());
        Arc::ptr_eq(&new.profiles()[0], &old.profiles()[0])
    };
    let standing = validated().expect("a valid document");
    let mut next = validated().expect("a valid document");
    next.keep_from(&standing);
    assert!(!same(&next, &standing), "another name: built new");
    let mut third = validated().expect("a valid document");
    third.keep_from(&next);
    assert!(same(&third, &next), "the same name again: kept");
}

/// A profile is kept only where its stages say of themselves what the
/// standing ones said. A factory that builds, at a later call, a stage
/// that answers otherwise one thing a validation or the engine reads
/// of a stage has its profile built new: the words it takes, the load
/// reports, a need, a mark, its ranks, the rule of its heads, its
/// warm-up ruling, a filter of another type under one name, the
/// lifecycle's calls, or no ask for the count of sends. The other
/// reader of its count of sends is kept, and the count with it. The
/// same stage again is kept.
#[cfg(feature = "in-process")]
#[test]
fn a_profile_whose_factory_builds_a_stage_that_says_otherwise_is_built_new() {
    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};

    use crate::{
        engine::{ScoreSpread, WarmUpRule},
        policies::prefix_hash::HeadRule,
        stages::filters::Healthy,
        tracking::{EventKind, EventSet},
    };

    /// A scorer that says of itself what its factory read at its build.
    #[derive(Debug)]
    struct Says(u8);
    impl Lifecycle for Says {
        fn keeps_loads(&self) -> bool {
            self.0 == 2
        }

        fn needs_lifecycle(&self) -> bool {
            self.0 == 9
        }

        fn hears(&self) -> EventSet {
            let heard = [EventSet::NONE, EventSet::NONE.with(EventKind::Completed)];
            heard[usize::from(self.0 == 1)]
        }
    }
    impl Scorer for Says {
        fn name(&self) -> &'static str {
            "says"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            out.fill(0.0);
            TieBand::NORMALIZED
        }

        fn needs(&self) -> Needs {
            Needs {
                text: self.0 == 3,
                ..Needs::default()
            }
        }

        fn places_by_key_text(&self) -> bool {
            self.0 == 4
        }

        fn spread(&self) -> ScoreSpread {
            ScoreSpread::Levels(if self.0 == 5 { 0.25 } else { 0.5 })
        }

        fn places_by_heads(&self) -> Option<HeadRule<'_>> {
            let rule = HeadRule {
                depth: 1,
                boundaries: &[],
            };
            (self.0 == 6).then_some(rule)
        }

        fn warm_up_rule(&self) -> Option<WarmUpRule> {
            let rule = WarmUpRule {
                window_micros: 1,
                blocks: 1,
                thin_ratio: 0.5,
            };
            (self.0 == 7).then_some(rule)
        }
    }
    /// A guest filter that says what `healthy` says, and is no type of
    /// the engine's own.
    #[derive(Debug)]
    struct Wraps(&'static str);
    impl Filter for Wraps {
        fn name(&self) -> &'static str {
            self.0
        }

        fn keep(&self, facts: &RequestFacts<'_>, row: &Candidate, cx: &Constraints<'_>) -> bool {
            Healthy.keep(facts, row, cx)
        }

        fn refuses_by_reports(&self) -> bool {
            Healthy.refuses_by_reports()
        }

        fn row_wise(&self) -> bool {
            Healthy.row_wise()
        }
    }

    let says = Arc::new(AtomicU8::new(u8::MAX));
    let (scorer, filter) = (Arc::clone(&says), Arc::clone(&says));
    let mut own = StageRegistry::builtin();
    // The guest asks its build for the profile's count of sends, but at
    // the last turn.
    let added = own.scorer_with("says", move |build| {
        let said = scorer.load(Relaxed);
        if said != 10 {
            let _count = build.dispatch_credit();
        }
        Ok(Box::new(Says(said)))
    });
    added.expect("a free name");
    // The registration asks the name once: `guest`. A validation then
    // builds a filter that names itself `healthy`, the engine's own
    // type at the last turn.
    let added = own.filter("guest", move || match filter.load(Relaxed) {
        u8::MAX => Box::new(Wraps("guest")),
        8 => Box::new(Healthy),
        _ => Box::new(Wraps("healthy")),
    });
    added.expect("a free name");
    // `p` and `q` read one count of sends, and `p` holds the guests.
    let document = doc(
        r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"filter":["guest"],"score":["says"],"pick":"best"},"q":{"score":["least-load"],"pick":"best"}},"bindings":[{"match":{"model":"m"},"profile":"p"},{"match":{"model":"*"},"profile":"q"}]}"#,
    );
    let validated = || AppliedSpec::validate_in(&document, &own, &[]);
    let worker = crate::engine::WorkerId(7);
    let target = crate::engine::Target {
        id: worker,
        ..Default::default()
    };
    let all = ([true, true, false], true);
    let beside = ([false, true, false], true);
    let turns = [
        (1, "other words"),
        (2, "load reports"),
        (3, "another need"),
        (4, "another mark"),
        (5, "other ranks"),
        (6, "a rule of heads"),
        (7, "a warm-up ruling"),
        (8, "a filter of another type"),
        (9, "the lifecycle's calls"),
        (10, "no ask for the count"),
    ];
    for (other, what) in turns {
        says.store(0, Relaxed);
        let standing = validated().expect("a valid document");
        let lease = crate::tracking::Lease::default();
        let engine = standing.engine();
        engine.on_dispatched(&lease, &target, crate::tracking::Side::Single, 0);
        says.store(other, Relaxed);
        let mut next = validated().expect("a valid document");
        next.keep_from(&standing);
        assert_eq!(kept(&next, &standing), beside, "{what}: `p` alone anew");
        next.engine().succeed(standing.engine());
        let held = next.engine().sent_credits[0].credits().held(worker);
        assert_eq!(held, 1, "{what}: the count stands with its send");
        let mut third = validated().expect("a valid document");
        third.keep_from(&next);
        assert_eq!(kept(&third, &next), all, "{what}: the same stage again");
    }
}

/// A spec that keeps one count of sends and builds a second, under
/// settings no standing count has, counts a send behind it once in
/// each. The count kept is the standing object: it takes nothing over,
/// and the count built beside it takes nothing of it.
#[cfg(feature = "in-process")]
#[test]
fn a_count_kept_is_taken_over_by_no_count_built_beside_it() {
    let one =
        r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":["least-load"],"pick":"best"}}}"#;
    let two = r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":["least-load"],"pick":"best"},"q":{"score":["least-load"],"pick":"best","sent_ttl_ms":43}},"bindings":[{"match":{"model":"m"},"profile":"p"},{"match":{"model":"*"},"profile":"q"}]}"#;
    let worker = crate::engine::WorkerId(7);
    let target = crate::engine::Target {
        id: worker,
        ..Default::default()
    };
    let send = |spec: &AppliedSpec| {
        let lease = crate::tracking::Lease::default();
        spec.engine()
            .on_dispatched(&lease, &target, crate::tracking::Side::Single, 0);
    };
    let standing = AppliedSpec::validate(&doc(one)).expect("a valid document");
    send(&standing);
    let mut next = AppliedSpec::validate(&doc(two)).expect("a valid document");
    next.keep_from(&standing);
    next.engine().succeed(standing.engine());
    send(&next);
    let counts = &next.engine().sent_credits;
    let kept = Arc::ptr_eq(&counts[0], &standing.engine().sent_credits[0]);
    assert!(kept, "the count of `p` is the standing one");
    let held: Vec<_> = counts
        .iter()
        .map(|count| count.credits().held(worker))
        .collect();
    assert_eq!(held, [2, 1], "each send once in each count that stood");
}

/// A count built beside a reader kept follows the standing one. The
/// engine goes on with the standing count, a send behind the keep is
/// counted once, and a worker first heard there stands once, for the
/// reader kept and for the reader built new.
#[cfg(feature = "in-process")]
#[test]
fn a_count_built_beside_a_reader_kept_follows_the_standing_one() {
    let send = |spec: &AppliedSpec, worker: u64| {
        let target = crate::engine::Target {
            id: crate::engine::WorkerId(worker),
            ..Default::default()
        };
        let lease = crate::tracking::Lease::default();
        spec.engine()
            .on_dispatched(&lease, &target, crate::tracking::Side::Single, 0);
    };
    let standing = AppliedSpec::validate(&doc(KEPT)).expect("a valid document");
    send(&standing, 7);
    // The text of `q` moves: it is built new beside `p`, which is kept.
    let mut later = doc(KEPT);
    let q = later.profiles.get_mut("q").expect("the profile");
    q.filter.push("healthy".into());
    let mut next = AppliedSpec::validate(&later).expect("a valid document");
    let built = Arc::clone(&next.engine().sent_credits[0]);
    next.keep_from(&standing);
    next.engine().succeed(standing.engine());
    let own = &next.engine().sent_credits[0];
    let stands = Arc::ptr_eq(own, &standing.engine().sent_credits[0]);
    assert!(stands, "the engine goes on with the standing count");
    assert!(!Arc::ptr_eq(own, &built), "the count built is in no list");
    send(&next, 9);
    let held = |count: &DispatchCredit| {
        [7, 9].map(|worker| count.credits().held(crate::engine::WorkerId(worker)))
    };
    assert_eq!(held(&built), [1, 1], "read through the count built");
    assert_eq!(held(own), [1, 1], "each send once");
}

/// Each later spec adds a reader of the one count. The count each
/// build made follows the first spec's count, and no chain of them
/// grows: every reader reads the first count's own entries, and a
/// worker first heard behind the third spec stands once.
#[cfg(feature = "in-process")]
#[test]
fn a_count_followed_across_three_specs_is_one_set_of_entries() {
    let with = |names: &str| {
        let profile = r#"{"score":["least-load"],"pick":"best"}"#;
        let named = |name: char| format!(r#""{name}":{profile}"#);
        let bound = |name: char| format!(r#"{{"match":{{"model":"{name}"}},"profile":"{name}"}}"#);
        let profiles: Vec<String> = names.chars().map(named).collect();
        let bindings: Vec<String> = names.chars().map(bound).collect();
        let (profiles, bindings) = (profiles.join(","), bindings.join(","));
        let text = format!(
            r#"{{"apiVersion":"smg.policy/v1","profiles":{{{profiles}}},"bindings":[{bindings}]}}"#
        );
        AppliedSpec::validate(&doc(&text)).expect("a valid document")
    };
    let first = with("p");
    let mut second = with("pq");
    let built_second = Arc::clone(&second.engine().sent_credits[0]);
    second.keep_from(&first);
    second.engine().succeed(first.engine());
    let mut third = with("pqs");
    let built_third = Arc::clone(&third.engine().sent_credits[0]);
    third.keep_from(&second);
    third.engine().succeed(second.engine());
    let worker = crate::engine::WorkerId(9);
    let target = crate::engine::Target {
        id: worker,
        ..Default::default()
    };
    let lease = crate::tracking::Lease::default();
    let engine = third.engine();
    engine.on_dispatched(&lease, &target, crate::tracking::Side::Single, 0);
    let leader = &first.engine().sent_credits[0];
    assert!(
        Arc::ptr_eq(leader, &engine.sent_credits[0]),
        "the engine's own"
    );
    for count in [leader, &built_second, &built_third] {
        let entries = std::ptr::eq(count.credits(), leader.credits());
        assert!(entries, "the first count's own entries, by no chain");
        assert_eq!(count.credits().held(worker), 1);
    }
}

/// A count kept bears the name of the profile that made it under an
/// earlier spec, so an engine's counts stand in no order of those
/// names. A count built new under settings no standing count has still
/// finds the count its profile's name made, past a count left free
/// ahead of it.
#[cfg(feature = "in-process")]
#[test]
fn a_count_built_new_finds_its_makers_count_among_counts_kept() {
    let spec = |profiles: &[(&str, u32)]| {
        let block = |&(name, ttl): &(&str, u32)| {
            let stages = r#""score":["least-load"],"pick":"best""#;
            format!(r#""{name}":{{{stages},"sent_ttl_ms":{ttl}}}"#)
        };
        let bound = |&(name, _): &(&str, u32)| {
            format!(r#"{{"match":{{"model":"{name}"}},"profile":"{name}"}}"#)
        };
        let blocks: Vec<String> = profiles.iter().map(block).collect();
        let bindings: Vec<String> = profiles.iter().map(bound).collect();
        let (blocks, bindings) = (blocks.join(","), bindings.join(","));
        let text = format!(
            r#"{{"apiVersion":"smg.policy/v1","profiles":{{{blocks}}},"bindings":[{bindings}]}}"#
        );
        AppliedSpec::validate(&doc(&text)).expect("a valid document")
    };
    let makers = |spec: &AppliedSpec| {
        let counts = spec.engine().sent_credits.iter();
        counts
            .map(|count| count.profile.clone())
            .collect::<Vec<_>>()
    };
    let first = spec(&[("a", 40), ("b", 41), ("x", 42)]);
    // `c` reads the count `x` made: the count kept stands at the place
    // of `c`'s, under `x`'s name.
    let mut second = spec(&[("a", 40), ("b", 41), ("c", 42), ("m", 43), ("x", 42)]);
    second.keep_from(&first);
    second.engine().succeed(first.engine());
    assert_eq!(makers(&second), ["a", "b", "x", "m"], "in no order");
    // `b` goes, and `m` writes another limit: no standing count has
    // its new count's settings, and `b`'s count is left free.
    let mut third = spec(&[("a", 40), ("c", 42), ("m", 44), ("x", 42)]);
    third.keep_from(&second);
    third.engine().succeed(second.engine());
    let entries = |count: &DispatchCredit| count.credits().tally.load_full();
    let new = &third.engine().sent_credits[2];
    let made = &second.engine().sent_credits[3];
    assert_eq!((new.profile.as_str(), made.profile.as_str()), ("m", "m"));
    let taken = Arc::ptr_eq(&entries(new), &entries(made));
    assert!(taken, "the entries of the count its maker's name made");
}

#[test]
fn every_catalog_scorer_parses_and_the_ones_this_build_lacks_are_refused_by_name() {
    let in_build = [
        "least-load",
        "least-requests",
        "fewest-sessions",
        "session-affinity",
        "prefix-affinity",
        "consistent-hash",
        "kv-pressure",
        "warmup-slice",
        "bucket",
    ];
    for &name in SCORERS {
        let bare = spec_with_score(&format!("{name:?}"));
        if in_build.contains(&name) {
            assert!(bare.validate().is_ok(), "{name}");
        } else if matches!(name, "resident-work" | "least-cost") {
            assert_eq!(bad_field(&bare), "profiles.p.score[0]");
        } else {
            assert_eq!(
                not_in_build(&bare).0,
                format!("profiles.p.score[0].{name}"),
                "{name}"
            );
        }
    }
    // The affinity stages and the bucket stage build with their
    // parameters; the forms a later step carries are refused by name.
    for score in [
        r#"{"kv-pressure":{"spread":0.3,"ceiling":0.9}}"#,
        r#"{"warmup-slice":{"secs":0,"miss_share":0.5}}"#,
        r#"{"prefix-affinity":{"threshold":0.5,"source":"merged","spill":{"abs":8,"rel":2.0}}}"#,
        r#"{"bucket":{"abs_threshold":32,"rel_threshold":1.0001,"adjust_interval_s":5}}"#,
    ] {
        assert!(spec_with_score(score).validate().is_ok(), "{score}");
    }
    let with_params = [
        (r#"{"prefix-affinity":{"decay":0.9}}"#, "prefix-affinity"),
        (
            r#"{"prefix-affinity":{"accounting_ttl_ms":200}}"#,
            "prefix-affinity",
        ),
        (
            r#"{"prefix-affinity":{"spill":{"uncached_credit":true}}}"#,
            "prefix-affinity",
        ),
    ];
    for (score, name) in with_params {
        let (path, what) = not_in_build(&spec_with_score(score));
        assert_eq!(path, format!("profiles.p.score[0].{name}"), "{score}");
        assert!(what.contains(name), "{what}");
    }
    // The resident-work scorer is in the build with its parameter, and
    // reads the load reports.
    let resident = spec_with_score(r#"{"resident-work":{"request_tokens":512}}"#);
    assert!(resident.validate().expect("validates").needs().loads);
    // The consistent-hash scorer is in the build with its bounded-load
    // rule and with its prefix form, which reads the request's block
    // hashes; without the form it asks for none.
    let bounded = spec_with_score(r#"{"consistent-hash":{"bounded_load":{"factor":1.5}}}"#);
    let needs = bounded.validate().expect("validates").needs();
    assert!(needs.ring && !needs.token_hashes && !needs.text_hashes);
    let prefix = spec_with_score(
        r#"{"consistent-hash":{"prefix_tokens":512,"bounded_load":{"factor":1.5,"abs_margin":4}}}"#,
    );
    let needs = prefix.validate().expect("validates").needs();
    assert!(needs.ring && needs.token_hashes && needs.text_hashes);
    // Under a view that names boundaries the prefix ring keys at the
    // deepest one a request reaches, and asks for the same.
    let levelled = doc(
        r#"{"apiVersion":"smg.policy/v1","view":{"boundaries":[256,1024]},"profiles":{"p":{"score":[{"consistent-hash":{"prefix_tokens":512}}],"pick":"best"}}}"#,
    );
    let needs = levelled.validate().expect("validates").needs();
    assert!(needs.ring && needs.token_hashes && needs.text_hashes);
    // A ring with no prefix form reads no boundary and asks for no
    // hashes there.
    let plain = doc(
        r#"{"apiVersion":"smg.policy/v1","view":{"boundaries":[256,1024]},"profiles":{"p":{"score":[{"consistent-hash":{"bounded_load":{"factor":1.5}}}],"pick":"best"}}}"#,
    );
    let needs = plain.validate().expect("validates").needs();
    assert!(needs.ring && !needs.token_hashes && !needs.text_hashes);
}

/// The prefix depth is held to the view that carries the hashes: at
/// least one block of the view's block size, at most the blocks a
/// request carries.
#[test]
fn a_prefix_depth_the_view_cannot_carry_is_refused() {
    let spec = |view: &str, tokens: u32| {
        doc(&format!(
            r#"{{"apiVersion":"smg.policy/v1","view":{view},"profiles":{{"p":{{"score":[{{"consistent-hash":{{"prefix_tokens":{tokens}}}}}],"pick":"best"}}}}}}"#
        ))
    };
    let refused = |view: &str, tokens: u32| match spec(view, tokens).validate() {
        Err(SpecError::BadField { path, reason, .. }) => {
            assert_eq!(
                path, "profiles.p.score[0].consistent-hash.prefix_tokens",
                "{view} {tokens}"
            );
            reason
        }
        other => panic!("{view} {tokens}: validated as {other:?}"),
    };
    let under = "at least one block of view.block_size";
    let over = "at most view.max_block_hashes blocks of view.block_size";
    // The default view: blocks of 16 tokens, 256 of them a request.
    for tokens in [16, 256, 4096, 4111] {
        assert!(spec("{}", tokens).validate().is_ok(), "{tokens}");
    }
    assert_eq!(refused("{}", 15), under);
    assert_eq!(refused("{}", 4112), over);
    // The view's own block size and its cap move both edges.
    let view = r#"{"block_size":64,"max_block_hashes":4}"#;
    for tokens in [64, 256, 319] {
        assert!(spec(view, tokens).validate().is_ok(), "{tokens}");
    }
    assert_eq!(refused(view, 63), under);
    assert_eq!(refused(view, 320), over);
    // Under a prefix key each of the view's boundaries is held to the
    // same two edges.
    let boundary = |boundaries: &str| {
        let view = format!(r#"{{"boundaries":{boundaries}}}"#);
        match spec(&view, 256).validate() {
            Err(SpecError::BadField { path, reason, .. }) => (path, reason),
            other => panic!("{boundaries}: validated as {other:?}"),
        }
    };
    assert!(spec(r#"{"boundaries":[16,4096,4111]}"#, 256)
        .validate()
        .is_ok());
    assert_eq!(
        boundary("[8,64]"),
        (
            "view.boundaries[0]".to_string(),
            "at least one block of view.block_size under a consistent-hash prefix key"
        )
    );
    assert_eq!(
            boundary("[64,4112]"),
            (
                "view.boundaries[1]".to_string(),
                "at most view.max_block_hashes blocks of view.block_size under a consistent-hash prefix key"
            )
        );
    // A view that no prefix key reads keeps such boundaries.
    assert!(spec_with_view(r#"{"boundaries":[8,4112]}"#)
        .validate()
        .is_ok());
}

#[test]
fn stage_parameters_are_range_checked_before_the_build_check() {
    let cases = [
        (r#"{"kv-pressure":{"spread":0}}"#, "kv-pressure.spread"),
        (r#"{"warmup-slice":{"share":2}}"#, "warmup-slice.share"),
        (
            r#"{"bucket":{"rel_threshold":0.5}}"#,
            "bucket.rel_threshold",
        ),
        (
            r#"{"consistent-hash":{"prefix_tokens":0}}"#,
            "consistent-hash.prefix_tokens",
        ),
        (
            r#"{"consistent-hash":{"bounded_load":{"factor":0.9}}}"#,
            "consistent-hash.bounded_load.factor",
        ),
        (
            r#"{"prefix-affinity":{"threshold":1.5}}"#,
            "prefix-affinity.threshold",
        ),
        (
            r#"{"prefix-affinity":{"spill":{"rel":0.5}}}"#,
            "prefix-affinity.spill.rel",
        ),
    ];
    for (score, path) in cases {
        assert_eq!(
            bad_field(&spec_with_score(score)),
            format!("profiles.p.score[0].{path}"),
            "{score}"
        );
    }
}

/// One profile for every assignment: the `manual` preset's picker
/// leaves the placement to the binding's rule, the rule names the
/// classic default, and each other assignment is that one field and
/// validates. The recorded rows hold where each one places. The built
/// engine answers each rule with the assignment the document names:
/// under this picker it places a `delegate` by the fewest in flight,
/// and a reader of the binding still reads `delegate`.
#[test]
fn the_manual_preset_is_one_profile_for_every_assignment() {
    let mut spec = PolicySpec::preset("manual").expect("the preset");
    let pick = spec.profiles.get("manual").map(|p| p.pick.name());
    assert_eq!(pick, Some("manual"));
    let default = spec
        .bindings
        .first()
        .and_then(|b| b.sessions.as_ref())
        .map(|rule| rule.assignment);
    assert_eq!(default, Some(Assignment::Random), "the classic default");
    for assignment in [
        Assignment::Random,
        Assignment::MinLoad,
        Assignment::MinGroup,
        Assignment::Delegate,
    ] {
        let rule = spec
            .bindings
            .first_mut()
            .and_then(|b| b.sessions.as_mut())
            .expect("the preset's sessions rule");
        rule.assignment = assignment;
        let engine = spec.validate().expect("every assignment validates");
        assert_eq!(
            engine.needs().active_sessions,
            assignment == Assignment::MinGroup,
            "{assignment:?}"
        );
        let read_back = engine
            .binding("m", Phase::Single)
            .and_then(|binding| binding.sessions)
            .map(|rule| rule.assignment);
        assert_eq!(read_back, Some(assignment), "read back as declared");
    }
}

/// The `manual` preset's whole rule, as the gateway's keyed policy runs
/// today: the routing-key header as the session key and the view's one
/// header, no header list of the rule's own, four hours of idle time
/// swept each minute, a random assignment, no in-flight cap, no session
/// id issued, the first worker back in the lead.
#[test]
fn the_manual_preset_states_its_whole_rule() {
    let spec = PolicySpec::preset("manual").expect("the preset");
    let rule = spec
        .bindings
        .first()
        .and_then(|b| b.sessions.as_ref())
        .expect("the preset's sessions rule");
    assert_eq!(
        *rule,
        SessionsSpec {
            sticky_header: "x-smg-routing-key".into(),
            headers: vec![],
            ttl_s: 14_400,
            sweep_s: 60,
            assignment: Assignment::Random,
            inflight_cap: u16::MAX,
            issue_session_id: false,
            recovery: Recovery::Original,
            rule_placed: RulePlaced::Told,
        }
    );
    assert_eq!(spec.view.headers, ["x-smg-routing-key"]);
}

/// The manual picker places by a sessions rule, and a rule runs under
/// the single plan and on the two sides of a pair: on a side of `epd`
/// the profile is refused, with the side that names it. The built
/// picker answers, so any picker that leaves the placement to the rule
/// is refused the same way.
#[test]
fn the_manual_picker_under_the_epd_plan_is_not_in_this_build() {
    let epd = doc(
        r#"{"apiVersion":"smg.policy/v1","profiles":{"q":{"pick":"random"},"sticky":{"pick":"manual"}},"bindings":[{"plan":"epd","profiles":{"encode":"sticky","prefill":"q","decode":"q"}}]}"#,
    );
    assert_eq!(
        not_in_build(&epd),
        (
            "bindings[0].profiles.encode".to_string(),
            "a picker that places by the sessions rule under the epd plan"
        )
    );
    // On a side of a pair the same profile validates, and so it does
    // under the single plan, here with no rule; the preset is the
    // profile with one.
    let pair = doc(
        r#"{"apiVersion":"smg.policy/v1","profiles":{"q":{"pick":"random"},"sticky":{"pick":"manual"}},"bindings":[{"plan":"pd-pair","profiles":{"prefill":"q","decode":"sticky"}}]}"#,
    );
    assert!(pair.validate().is_ok());
    let single = doc(
        r#"{"apiVersion":"smg.policy/v1","profiles":{"sticky":{"pick":"manual"}},"bindings":[{"profile":"sticky"}]}"#,
    );
    assert!(single.validate().is_ok());
}

/// A sessions rule under the pair plan is in the build of a host that
/// makes some lifecycle calls only, as of one that makes them all: a
/// file of leases keeps each side's own session verdict.
#[test]
fn a_pairs_sessions_rule_is_in_the_build_of_a_host_that_makes_some_lifecycle_calls() {
    let rule = r#"{"sticky_header":"x-session-id","inflight_cap":65535}"#;
    let pair = format!(
        r#"{{"plan":"pd-pair","profiles":{{"prefill":"q","decode":"q"}},"sessions":{rule}}}"#
    );
    let pair = spec_with_binding(&pair);
    let some = StageRegistry::builtin().without_lifecycle();
    for catalog in [StageRegistry::builtin(), some] {
        assert!(pair.validate_in(&catalog, &[]).is_ok());
    }
}

/// A sessions rule runs under the single plan, and on the two sides of
/// a pair, under a cap or under none. A cap under the pair makes the
/// document's engine read a worker's count of a key's requests, the
/// schema's default cap as one that is written. A rule that keeps what
/// it places from the profile runs under a pair in the build of a host
/// that makes the lifecycle's calls, where each side of a pick states
/// its own word. In the build of a host that makes some only it is
/// refused by that field's name, and the refusal says why: that host
/// tells a lease's words off a file, which keeps no word of a pair's
/// side.
///
/// Under `epd` the rule runs on the pair's two sides in the build of a
/// host that makes the lifecycle's calls, under either word, and a cap
/// makes the document's engine read a key's count as under a pair. The
/// build of a host that makes some only refuses the rule by the plan's
/// name, whatever its word.
///
/// A gateway's sticky map places each role of a pair under that role's
/// own key, and hands its policy no request it placed
/// (`model_gateway/src/policies/registry.rs:762-769`, `:794-797`).
#[test]
fn a_sessions_rule_runs_under_a_pair_and_under_epd_by_its_hosts_calls() {
    let built = |plan: &str, profiles: &str, sessions: &str| {
        spec_with_binding(&format!(
            r#"{{"plan":"{plan}","profiles":{profiles},"sessions":{sessions}}}"#
        ))
        .validate()
    };
    let under = |plan: &str, profiles: &str, sessions: &str| built(plan, profiles, sessions).err();
    let pair = r#"{"prefill":"q","decode":"q"}"#;
    let uncapped = r#"{"sticky_header":"x-session-id","inflight_cap":65535}"#;
    // Whether the pair's engine reads the count of a key's requests.
    let counted = |sessions: &str| {
        built("pd-pair", pair, sessions).map(|built| built.needs().session_in_flight)
    };
    assert_eq!(counted(uncapped), Ok(false));
    for capped in [
        r#"{"sticky_header":"x-session-id"}"#,
        r#"{"sticky_header":"x-session-id","inflight_cap":4}"#,
    ] {
        assert_eq!(counted(capped), Ok(true), "{capped}");
    }
    let untold = r#"{"sticky_header":"x-session-id","inflight_cap":65535,"rule_placed":"untold"}"#;
    assert_eq!(under("pd-pair", pair, untold), None);
    let some = StageRegistry::builtin().without_lifecycle();
    let on_file = |sessions: &str| {
        spec_with_binding(&format!(
            r#"{{"plan":"pd-pair","profiles":{pair},"sessions":{sessions}}}"#
        ))
        .validate_in(&some, &[])
        .err()
    };
    assert_eq!(on_file(uncapped), None, "a rule that tells its stages");
    let refused = on_file(untold);
    assert_eq!(
        refused,
        Some(SpecError::BadSessions {
            index: 0,
            name: "rule_placed",
            reason: UNTOLD_ON_FILE,
        })
    );
    assert_eq!(
        refused.map(|err| err.to_string()),
        Some(format!(
            "binding 0: sessions rule_placed is not {UNTOLD_ON_FILE}"
        ))
    );
    assert!(UNTOLD_ON_FILE.ends_with("a file of leases keeps no side's word"));
    // The single plan takes each of the three, under either host.
    for rule in [uncapped, untold, r#"{"sticky_header":"x-session-id"}"#] {
        let single = spec_with_binding(&format!(r#"{{"profile":"q","sessions":{rule}}}"#));
        assert_eq!(single.validate().err(), None, "{rule}");
        assert_eq!(single.validate_in(&some, &[]).err(), None, "{rule}");
    }
    let epd = r#"{"encode":"q","prefill":"q","decode":"q"}"#;
    let capped = r#"{"sticky_header":"x-session-id"}"#;
    for rule in [uncapped, untold, capped] {
        assert_eq!(under("epd", epd, rule), None, "{rule}");
        let refused = spec_with_binding(&format!(
            r#"{{"plan":"epd","profiles":{epd},"sessions":{rule}}}"#
        ))
        .validate_in(&some, &[])
        .err();
        assert_eq!(
            refused,
            Some(SpecError::BadSessions {
                index: 0,
                name: "plan",
                reason: "`single` or `pd-pair`",
            }),
            "{rule}"
        );
        assert_eq!(
            refused.map(|err| err.to_string()),
            Some("binding 0: sessions plan is not `single` or `pd-pair`".to_string()),
            "{rule}"
        );
    }
    let counts =
        |sessions: &str| built("epd", epd, sessions).map(|built| built.needs().session_in_flight);
    assert_eq!((counts(uncapped), counts(capped)), (Ok(false), Ok(true)));
    // The rule's needs are the pair's two sides': the encode phase
    // takes none of them, by any of the engine's three answers.
    let ruled = built("epd", epd, capped).expect("a rule under epd");
    let bare = spec_with_binding(&format!(r#"{{"plan":"epd","profiles":{epd}}}"#))
        .validate()
        .expect("the plan alone");
    for side in [Phase::Prefill, Phase::Decode] {
        assert!(ruled.needs_for_phase(side).session_in_flight, "{side:?}");
        assert!(!bare.needs_for_phase(side).session_in_flight, "{side:?}");
    }
    let of_encode = |engine: &Engine| {
        let rows: Vec<Needs> = engine
            .needs_rows()
            .into_iter()
            .filter(|row| row.1 == Phase::Encode)
            .map(|row| row.2)
            .collect();
        (
            engine.needs_for_phase(Phase::Encode),
            engine.reads_for("m", Phase::Encode),
            rows,
        )
    };
    assert_eq!(of_encode(&ruled), of_encode(&bare));
}

#[test]
fn the_pickers_of_v1_resolve_or_are_refused_by_name() {
    assert_eq!(
        bad_field(&spec_with_pick(r#""best-with-temperature""#)),
        "profiles.p.pick"
    );
    assert_eq!(
        not_in_build(&spec_with_pick(
            r#"{"best-with-temperature":{"temperature":0.05}}"#
        )),
        (
            "profiles.p.pick".to_string(),
            "the best-with-temperature picker"
        )
    );
    assert_eq!(
        bad_field(&spec_with_pick(
            r#"{"best-with-temperature":{"temperature":-1}}"#
        )),
        "profiles.p.pick.best-with-temperature.temperature"
    );
    let err = parse_error(
        r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":{"best-with-temperature":{"heat":1}}}}}"#,
    );
    assert!(err.contains("unknown field `heat`"), "{err}");
    for name in [
        "best",
        "round-robin",
        "random",
        "power-of-two",
        "count-first",
        "power-of-two-count-first",
        "passthrough",
    ] {
        assert!(
            spec_with_pick(&format!("{name:?}")).validate().is_ok(),
            "{name}"
        );
    }
    // The manual picker leaves the placement to the binding's rule: no
    // scorer of its profile would ever run, so one is refused by name.
    match spec_with_pick(r#""manual""#).validate() {
        Err(SpecError::PickerRunsNoScorer { profile, picker }) => {
            assert_eq!((profile.as_str(), picker.as_str()), ("p", "manual"));
        }
        other => panic!("a scorer beside the manual picker validated as {other:?}"),
    }
    let alone = doc(r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"manual"}}}"#);
    assert!(alone.validate().is_ok());
}

/// The start a binding is built from by hand ([`Binding::of`]) is the
/// binding a document gets that states its model and its profile and
/// nothing else: the two are held equal here.
#[test]
fn a_binding_of_a_model_and_a_profile_alone_is_the_start_built_by_hand() {
    let built = spec_with_binding(r#"{"match":{"model":"m"},"profile":"p"}"#)
        .validate()
        .expect("the binding validates");
    let binding = built.binding("m", Phase::Single).expect("the binding");
    assert_eq!(
        format!("{binding:?}"),
        format!("{:?}", Binding::of("m".into(), 0))
    );
}

#[test]
fn the_binding_rules_are_checked_in_shape_then_by_the_build() {
    // A block names no number, and a null is no number. Switched on or
    // off, with a shed word or none, the binding validates, and so
    // does a binding with no block.
    for block in [
        r#"{"profile":"p"}"#,
        r#"{"profile":"p","overload":{}}"#,
        r#"{"profile":"p","overload":{"enabled":true,"shed_when_all_vetoed":true}}"#,
        r#"{"profile":"p","overload":{"waiting_requests":null,"token_usage":null}}"#,
        r#"{"profile":"p","overload":{"enabled":false,"shed_when_all_vetoed":true}}"#,
    ] {
        assert!(spec_with_binding(block).validate().is_ok(), "{block}");
    }

    let sessions = spec_with_binding(
        r#"{"profile":"p","sessions":{"sticky_header":"x-session-id","ttl_s":600,"assignment":"min_load"}}"#,
    );
    let built = sessions
        .validate()
        .expect("the session table is in this build");
    assert!(built.needs().headers && !built.needs().active_sessions);
    assert!(built.needs().session_in_flight, "the default cap of two");
    assert_eq!(
        built.needs_for_phase(Phase::Single),
        built.needs(),
        "the rule's needs are the binding's phase needs"
    );
    assert_eq!(
        spec_with_binding(r#"{"profile":"p","sessions":{"sticky_header":"x-other"}}"#)
            .validate()
            .err(),
        Some(SpecError::HeaderNotAllowed {
            path: "bindings[0].sessions.sticky_header".into(),
            name: "x-other".into()
        })
    );
    assert_eq!(
        bad_field(&spec_with_binding(
            r#"{"profile":"p","sessions":{"sticky_header":"x-session-id","headers":["x-session-id"],"ttl_s":0}}"#
        )),
        "bindings[0].sessions.ttl_s"
    );
    assert_eq!(
        bad_field(&spec_with_binding(
            r#"{"profile":"p","sessions":{"sticky_header":"x-session-id","headers":["x-request-id"]}}"#
        )),
        "bindings[0].sessions.sticky_header"
    );
    assert_eq!(
        bad_field(&spec_with_binding(
            r#"{"profile":"p","sessions":{"sticky_header":"x-session-id","inflight_cap":0}}"#
        )),
        "bindings[0].sessions.inflight_cap"
    );
    // A sweep for ever under a finite idle time would never forget a key.
    assert_eq!(
            spec_with_binding(
                r#"{"profile":"p","sessions":{"sticky_header":"x-session-id","ttl_s":600,"sweep_s":4294967295}}"#
            )
            .validate()
            .err(),
            Some(SpecError::BadSessions {
                index: 0,
                name: "sweep_s",
                reason: "finite while ttl_s is",
            })
        );
    // An idle time for ever keeps every key and sweeps nothing, whatever
    // the sweep says; a finite pair maps to the rule in microseconds.
    let rule = |sessions: &str| {
        spec_with_binding(&format!(
            r#"{{"profile":"p","sessions":{{"sticky_header":"x-session-id",{sessions}}}}}"#
        ))
        .validate()
        .expect("a sessions rule")
        .binding("m", Phase::Single)
        .and_then(|binding| binding.sessions)
        .map(|rule| (rule.ttl_micros, rule.sweep_micros))
    };
    assert_eq!(
        rule(r#""ttl_s":4294967295,"sweep_s":4294967295"#),
        Some((None, 0))
    );
    assert_eq!(rule(r#""ttl_s":4294967295,"sweep_s":60"#), Some((None, 0)));
    assert_eq!(
        rule(r#""ttl_s":600,"sweep_s":60"#),
        Some((Some(600_000_000), 60_000_000))
    );
    assert_eq!(
        not_in_build(&spec_with_binding(
            r#"{"profile":"p","sessions":{"sticky_header":"x-session-id","issue_session_id":true}}"#
        )),
        (
            "bindings[0].sessions.issue_session_id".to_string(),
            "issuing a session id"
        )
    );
    // A rule names its key's header, or the binding names `routing_key`.
    assert_eq!(
        spec_with_binding(r#"{"profile":"p","sessions":{"ttl_s":1}}"#)
            .validate()
            .err(),
        Some(SpecError::BadSessions {
            index: 0,
            name: "sticky_header",
            reason: "named while the binding names no routing_key",
        })
    );

    assert_eq!(
        not_in_build(&spec_with_binding(
            r#"{"profile":"p","canary":{"profile":"q","percent":5}}"#
        )),
        ("bindings[0].canary".to_string(), "the canary binding")
    );
    assert_eq!(
        spec_with_binding(
            r#"{"profile":"p","canary":{"profile":"q","percent":5,"tenants":["acme"]}}"#
        )
        .validate()
        .err(),
        Some(SpecError::CanaryForm {
            path: "bindings[0].canary".into(),
            count: 2
        })
    );
    assert_eq!(
        spec_with_binding(r#"{"profile":"p","canary":{"profile":"q"}}"#)
            .validate()
            .err(),
        Some(SpecError::CanaryForm {
            path: "bindings[0].canary".into(),
            count: 0
        })
    );
    assert_eq!(
        bad_field(&spec_with_binding(
            r#"{"profile":"p","canary":{"profile":"q","percent":101}}"#
        )),
        "bindings[0].canary.percent"
    );
    assert_eq!(
        bad_field(&spec_with_binding(
            r#"{"profile":"p","canary":{"profile":"q","percent":0}}"#
        )),
        "bindings[0].canary.percent"
    );
    assert_eq!(
            not_in_build(&spec_with_binding(
                r#"{"profile":"p","canary":{"profile":"q","header":{"name":"x-session-id","values":["beta"]}}}"#
            ))
            .1,
            "the canary binding"
        );

    assert_eq!(
        not_in_build(&spec_with_binding(
            r#"{"profile":"p","override":"allowed"}"#
        )),
        (
            "bindings[0].override".to_string(),
            "the profile override header"
        )
    );
    assert!(spec_with_binding(r#"{"profile":"p","override":"locked"}"#)
        .validate()
        .is_ok());
    assert_eq!(
        spec_with_binding(r#"{"profile":"p","admission":{"enabled":true}}"#)
            .validate()
            .err(),
        Some(SpecError::Reserved {
            path: "bindings[0].admission.enabled".into(),
            reason: "admission is reserved in v1 and stays off"
        })
    );
    assert!(
        spec_with_binding(r#"{"profile":"p","admission":{"enabled":false}}"#)
            .validate()
            .is_ok()
    );
}

/// A binding's `routing_key` block names one to eight sources, each a
/// header outside the credential names or the request's own id, each
/// once, with a byte cap inside the schema's. Beside it a sessions rule
/// names no header of its own.
#[test]
fn a_routing_key_block_is_checked_in_shape() {
    let block = |block: &str| {
        spec_with_binding(&format!(r#"{{"profile":"p","routing_key":{block}}}"#)).validate()
    };
    // Every form of a source, `cookie` among the headers: no value of
    // these enters a view, so none is held to `view.headers`.
    assert!(block(
            r#"{"sources":[{"header":"x-smg-routing-key"},{"rid":true},{"header":"x-forwarded-for","max_bytes":8192},{"header":"cookie"}],"strip_lineage":true}"#
        )
        .is_ok());
    // A header named `rid` and the request's own id are two sources.
    assert!(block(r#"{"sources":[{"header":"rid"},{"rid":true}]}"#).is_ok());
    let refused = |text: &str| block(text).expect_err(text).to_string();
    let nine: Vec<String> = (0..9).map(|i| format!(r#"{{"header":"x-{i}"}}"#)).collect();
    let nine = format!(r#"{{"sources":[{}]}}"#, nine.join(","));
    // Client sources are counted apart: four stand beside eight.
    let beside_eight = |clients: usize| {
        let listed: Vec<String> = (0..8)
            .map(|i| format!(r#"{{"header":"x-{i}"}}"#))
            .chain((0..clients).map(|i| format!(r#"{{"header":"x-c{i}","client":true}}"#)))
            .collect();
        format!(r#"{{"sources":[{}]}}"#, listed.join(","))
    };
    assert!(block(&beside_eight(4)).is_ok());
    let five = beside_eight(5);
    for (text, message) in [
            (
                r#"{"sources":[]}"#,
                "bindings[0].routing_key.sources = 0 entries is not a list of one to eight sources",
            ),
            (
                nine.as_str(),
                "bindings[0].routing_key.sources = 9 entries is not a list of one to eight sources",
            ),
            (
                five.as_str(),
                "bindings[0].routing_key.sources = 5 client entries is not a list of four client sources at most",
            ),
            (
                r#"{"sources":[{}]}"#,
                "bindings[0].routing_key.sources[0] = neither is not one of header and rid",
            ),
            (
                r#"{"sources":[{"header":"x-a","rid":true}]}"#,
                "bindings[0].routing_key.sources[0] = both is not one of header and rid",
            ),
            (
                r#"{"sources":[{"header":"X-A"}]}"#,
                "bindings[0].routing_key.sources[0].header = X-A is not a lowercase header name",
            ),
            (
                r#"{"sources":[{"header":"authorization"}]}"#,
                "bindings[0].routing_key.sources[0].header: header \"authorization\" is on the deny-list",
            ),
            (
                r#"{"sources":[{"header":"x-a"},{"header":"x-a"}]}"#,
                "bindings[0].routing_key.sources[1]: \"x-a\" appears twice",
            ),
            (
                r#"{"sources":[{"rid":true},{"rid":true}]}"#,
                "bindings[0].routing_key.sources[1]: \"rid\" appears twice",
            ),
            (
                r#"{"sources":[{"rid":true,"max_bytes":0}]}"#,
                "bindings[0].routing_key.sources[0].max_bytes = 0 is not between 1 and 8192",
            ),
            (
                r#"{"sources":[{"rid":true,"max_bytes":8193}]}"#,
                "bindings[0].routing_key.sources[0].max_bytes = 8193 is not between 1 and 8192",
            ),
        ] {
            assert_eq!(refused(text), message);
        }
    // A host's own denied name, its tenant header, is refused as a
    // credential is.
    let tenant = spec_with_binding(
        r#"{"profile":"p","routing_key":{"sources":[{"header":"x-tenant-id"}]}}"#,
    );
    assert!(tenant.validate().is_ok());
    assert_eq!(
        tenant.validate_with(&["x-tenant-id"]).err(),
        Some(SpecError::DeniedHeader {
            path: "bindings[0].routing_key.sources[0].header".to_string(),
            name: "x-tenant-id".to_string(),
        })
    );
    // A client source is a header read whole.
    let client = |source: &str| {
        spec_with_binding(&format!(
            r#"{{"profile":"p","routing_key":{{"sources":[{source}]}}}}"#
        ))
        .validate()
        .err()
    };
    assert_eq!(
        client(r#"{"header":"x-a"},{"header":"cookie","client":true}"#),
        None
    );
    // Client sources alone are no block: a sessions rule beside the
    // block would have no source of a session's key.
    assert_eq!(
            client(r#"{"header":"cookie","client":true}"#).map(|refused| refused.to_string()),
            Some(
                "bindings[0].routing_key.sources = client entries alone is not a list of one to eight sources"
                    .to_string()
            )
        );
    // A header is named once each way.
    assert_eq!(
        client(r#"{"header":"x-forwarded-for"},{"header":"x-forwarded-for","client":true}"#),
        None
    );
    assert!(matches!(
        client(r#"{"header":"cookie","client":true},{"header":"cookie","client":true}"#),
        Some(SpecError::Duplicate { .. })
    ));
    for refused in [
        r#"{"rid":true,"client":true}"#,
        r#"{"header":"cookie","max_bytes":64,"client":true}"#,
    ] {
        assert!(
            matches!(
                client(refused),
                Some(SpecError::BadField { path, .. }) if path.ends_with("sources[0].client")
            ),
            "{refused}"
        );
    }
    // A target source stands beside the eight and the four. It hands
    // no key, so it is no block alone, nor with client sources only: a
    // sessions rule beside the block would have no source of a
    // session's key.
    let with_target =
        |block: String| block.replace("]}", r#",{"header":"x-target","target":true}]}"#);
    assert!(block(&with_target(beside_eight(4))).is_ok());
    for alone in [
        r#"{"header":"x-target","target":true}"#,
        r#"{"header":"cookie","client":true},{"header":"x-target","target":true}"#,
    ] {
        assert_eq!(
                client(alone).map(|refused| refused.to_string()),
                Some(
                    "bindings[0].routing_key.sources = a target entry and no source of a key is not a list of one to eight sources"
                        .to_string()
                ),
                "{alone}"
            );
    }
    // A header is named once as the target's too, and a block names
    // one target.
    assert_eq!(
        client(r#"{"header":"x-a"},{"header":"x-a","target":true}"#),
        None
    );
    assert_eq!(
            client(r#"{"header":"x-a","target":true},{"header":"x-b","target":true}"#)
                .map(|refused| refused.to_string()),
            Some(
                "bindings[0].routing_key.sources = 2 target entries is not a list of one target source at most"
                    .to_string()
            )
        );
    // A target source is a header alone.
    for refused in [
        r#"{"rid":true,"target":true}"#,
        r#"{"header":"x-target","max_bytes":64,"target":true}"#,
        r#"{"header":"x-target","client":true,"target":true}"#,
    ] {
        assert!(
            matches!(
                client(refused),
                Some(SpecError::BadField { path, .. }) if path.ends_with("sources[0].target")
            ),
            "{refused}"
        );
    }
    // One spelling or the other beside a sessions rule.
    let beside = |sessions: &str| {
        spec_with_binding(&format!(
                r#"{{"profile":"p","routing_key":{{"sources":[{{"rid":true}}]}},"sessions":{sessions}}}"#
            ))
            .validate()
            .err()
    };
    assert_eq!(beside(r#"{"ttl_s":600}"#), None);
    for (sessions, name) in [
        (r#"{"sticky_header":"x-session-id"}"#, "sticky_header"),
        (r#"{"headers":["x-session-id"]}"#, "headers"),
    ] {
        assert_eq!(
            beside(sessions),
            Some(SpecError::BadSessions {
                index: 0,
                name,
                reason: "left out while the binding names routing_key",
            })
        );
    }
}

/// The applied document answers a request's key sources from whichever
/// spelling its binding writes: the `routing_key` block, or a sessions
/// rule's own headers. A binding with neither names none.
#[test]
fn the_applied_document_answers_a_bindings_key_sources() {
    let applied = AppliedSpec::validate(&doc(
            r#"{"apiVersion":"smg.policy/v1","view":{"headers":["x-session-id","x-tenant-session"]},"profiles":{"p":{"pick":"best"}},"bindings":[
                {"match":{"model":"chat/*"},"profile":"p","routing_key":{"sources":[{"header":"x-smg-routing-key"},{"rid":true},{"header":"cookie"}]}},
                {"match":{"model":"code/*"},"profile":"p","sessions":{"sticky_header":"x-session-id","headers":["x-tenant-session","x-session-id"]}},
                {"profile":"p"}]}"#,
        ))
        .expect("the document validates");
    let read = |model: &str| {
        applied
            .key_sources_for(model, Phase::Single)
            .map(|sources| (sources.headers().collect::<Vec<_>>(), sources.reads_rid()))
    };
    assert_eq!(
        read("chat/a"),
        Some((vec!["x-smg-routing-key", "cookie"], true))
    );
    assert_eq!(
        read("code/a"),
        Some((vec!["x-tenant-session", "x-session-id"], false))
    );
    assert_eq!(read("other"), None, "a binding with no source");
    // A rule's own header is on the view's allow-list: its key leaves in
    // the clear.
    let rule = applied
        .key_sources_for("code/a", Phase::Single)
        .expect("the rule's headers");
    let derived = rule
        .derive(|name| (name == "x-session-id").then_some(&b"s-1"[..]), None)
        .expect("a key");
    assert_eq!(derived.key, crate::engine::Identity::plain("s-1"));
    assert!(AppliedSpec::unbound()
        .key_sources_for("chat/a", Phase::Single)
        .is_none());
    // The first binding of the three keys by the request's own id, so
    // the document does. So does one where the second binding does,
    // behind a sessions rule's. That document less its one source does
    // not, and neither does one that binds nothing.
    assert!(applied.reads_request_id());
    let second = r#"{"apiVersion":"smg.policy/v1","view":{"headers":["x-session-id","x-tenant-session"]},"profiles":{"p":{"pick":"best"}},"bindings":[
            {"match":{"model":"code/*"},"profile":"p","sessions":{"sticky_header":"x-session-id","headers":["x-tenant-session","x-session-id"]}},
            {"match":{"model":"chat/*"},"profile":"p","routing_key":{"sources":[{"header":"x-smg-routing-key"},{"rid":true}]}},
            {"profile":"p"}]}"#;
    let reads = |text: &str| {
        let applied = AppliedSpec::validate(&doc(text));
        applied.expect("the document validates").reads_request_id()
    };
    assert!(reads(second));
    assert!(!reads(&second.replace(r#",{"rid":true}"#, "")));
    assert!(!AppliedSpec::unbound().reads_request_id());
}

/// Whether a binding names a target source is decided when the document
/// is applied. One binding of several is enough, and its sources alone
/// say so. A document that names none says so, as the generation no
/// document was applied to does.
#[test]
fn the_applied_document_says_whether_a_binding_names_a_target() {
    let applied = |second: &str| {
        AppliedSpec::validate(&doc(&format!(
                r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{"pick":"best"}}}},"bindings":[
                {{"match":{{"model":"chat/*"}},"profile":"p","routing_key":{{"sources":[{{"header":"x-smg-routing-key"}}]}}}},
                {{"profile":"p"{second}}}]}}"#
            )))
            .expect("the document validates")
    };
    assert!(!applied("").names_target());
    let named = applied(
        r#","routing_key":{"sources":[{"header":"x-b"},{"header":"x-smg-target-worker","target":true}]}"#,
    );
    assert!(named.names_target());
    let sources = |model| named.key_sources_for(model, Phase::Single);
    let first = sources("chat/a").expect("the first binding's sources");
    assert!(!first.names_target());
    let second = sources("other").expect("the second binding's sources");
    assert!(second.names_target());
    assert_eq!(
        second.headers().collect::<Vec<_>>(),
        ["x-b"],
        "no key's source"
    );
    assert!(!AppliedSpec::unbound().names_target());
}

/// What the bindings say of the overload switch, taken together, is
/// decided when the document is applied. A binding with no block, or a
/// block that names no switch, says on. Every binding off says off, and
/// one word against another says that they differ, whichever comes
/// first. A document with no binding says on, as the generation no
/// document was applied to does. The shed word is not the switch.
#[test]
fn the_applied_document_says_its_bindings_overload_switch_as_one_word() {
    let said = |first: &str, second: &str| {
        AppliedSpec::validate(&doc(&format!(
            r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{"pick":"best"}}}},"bindings":[
                {{"match":{{"model":"chat/*"}},"profile":"p"{first}}},
                {{"profile":"p"{second}}}]}}"#
        )))
        .expect("the document validates")
        .overload_switch()
    };
    let off = r#","overload":{"enabled":false}"#;
    let on = r#","overload":{"enabled":true}"#;
    let sheds = r#","overload":{"shed_when_all_vetoed":true}"#;
    let empty = r#","overload":{}"#;
    for (first, second) in [("", ""), (on, ""), ("", empty), (sheds, on)] {
        assert_eq!(said(first, second), OverloadSwitch::On, "{first} {second}");
    }
    assert_eq!(said(off, off), OverloadSwitch::Off);
    let off_and_sheds = r#","overload":{"enabled":false,"shed_when_all_vetoed":true}"#;
    assert_eq!(said(off, off_and_sheds), OverloadSwitch::Off);
    for (first, second) in [(off, ""), ("", off), (off, on), (sheds, off)] {
        assert_eq!(
            said(first, second),
            OverloadSwitch::Differ,
            "{first} {second}"
        );
    }
    let unbound = doc(r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"best"}}}"#);
    let no_binding = AppliedSpec::validate(&unbound).expect("the document validates");
    assert_eq!(no_binding.overload_switch(), OverloadSwitch::On);
    assert_eq!(AppliedSpec::unbound().overload_switch(), OverloadSwitch::On);
}

/// The first binding that says another word than the first one does is
/// handed by its place and its word, and none where the bindings say
/// one word. It is the one reading of the bindings' words: a document
/// has a place to name exactly where its word is that they differ.
#[test]
fn the_first_binding_that_differs_on_the_overload_switch_is_handed_by_its_place() {
    let heads = [
        r#"{"match":{"model":"chat/*"},"profile":"p""#,
        r#"{"match":{"model":"code/*"},"profile":"p""#,
        r#"{"profile":"p""#,
    ];
    let document = |words: [Option<bool>; 3]| {
        let bindings = heads.iter().zip(words).map(|(head, word)| {
            let block = word.map_or_else(String::new, |enabled| {
                format!(r#","overload":{{"enabled":{enabled}}}"#)
            });
            format!("{head}{block}}}")
        });
        let bindings: Vec<_> = bindings.collect();
        doc(&format!(
            r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{"pick":"best"}}}},"bindings":[{}]}}"#,
            bindings.join(",")
        ))
    };
    let (on, off) = (Some(true), Some(false));
    let cases = [
        ([None, None, None], None),
        ([off, off, off], None),
        ([on, None, on], None),
        ([None, off, off], Some((1, false))),
        ([off, off, None], Some((2, true))),
        ([off, on, None], Some((1, true))),
        ([on, None, off], Some((2, false))),
    ];
    for (words, place) in cases {
        let document = document(words);
        let differs = OverloadSwitch::differs(&document.bindings);
        assert_eq!(differs, place, "{words:?}");
        let applied = AppliedSpec::validate(&document).expect("the document validates");
        let differ = applied.overload_switch() == OverloadSwitch::Differ;
        assert_eq!(differ, place.is_some(), "{words:?}");
    }
    assert_eq!(OverloadSwitch::differs(&[]), None, "no binding");
}

#[test]
fn the_matchers_are_checked_in_shape_and_refused_as_not_in_this_build() {
    let matching = |m: &str| spec_with_binding(&format!(r#"{{"match":{m},"profile":"p"}}"#));
    assert_eq!(
        bad_field(&matching(r#"{"tenant":""}"#)),
        "bindings[0].match.tenant"
    );
    assert_eq!(
        not_in_build(&matching(r#"{"tenant":"acme"}"#)),
        ("bindings[0].match.tenant".to_string(), "the tenant matcher")
    );
    assert_eq!(
        bad_field(&matching(r#"{"priority":{}}"#)),
        "bindings[0].match.priority"
    );
    assert_eq!(
        bad_field(&matching(r#"{"priority":{"min":5,"max":1}}"#)),
        "bindings[0].match.priority.min"
    );
    assert_eq!(
        not_in_build(&matching(r#"{"priority":{"max":-1}}"#)).1,
        "the priority matcher"
    );
    assert_eq!(
        not_in_build(&matching(r#"{"header":{"name":"x-session-id"}}"#)).1,
        "the header matcher"
    );
    assert_eq!(
        matching(r#"{"header":{"name":"x-other","values":["a"]}}"#)
            .validate()
            .err(),
        Some(SpecError::HeaderNotAllowed {
            path: "bindings[0].match.header.name".into(),
            name: "x-other".into()
        })
    );
    assert_eq!(
        not_in_build(&matching(r#"{"prompt_tokens":{"min":1024}}"#)).1,
        "the prompt-tokens matcher"
    );
    assert_eq!(
        matching(r#"{"labels":["large","large"]}"#).validate().err(),
        Some(SpecError::Duplicate {
            path: "bindings[0].match.labels".into(),
            name: "large".into()
        })
    );
    assert_eq!(
        not_in_build(&matching(r#"{"labels":["large"]}"#)).1,
        "the label matcher"
    );
}

/// A binding's overload block names no number. Either one is refused by
/// its path, with the block switched on, switched off or left as it
/// stands, at the crate's own number too, and the refusal says where a
/// threshold is stated and states the number as it was written. A block
/// that names both is refused by the count, switched off too.
#[test]
fn a_number_in_a_bindings_overload_block_is_refused_by_its_path() {
    const STATED: &str =
        "left out: a worker's registration states this threshold, or its program's default does";
    let refused = |block: &str| {
        let binding = format!(r#"{{"profile":"p","overload":{block}}}"#);
        match spec_with_binding(&binding).validate().err() {
            Some(SpecError::BadField {
                path,
                value,
                reason,
            }) => (path, value, reason),
            other => panic!("{block} validated as {other:?}"),
        }
    };
    for (block, field, value) in [
        (r#"{"waiting_requests":8}"#, "waiting_requests", "8"),
        (r#"{"waiting_requests":0}"#, "waiting_requests", "0"),
        (r#"{"waiting_requests":300}"#, "waiting_requests", "300"),
        (
            r#"{"enabled":true,"waiting_requests":32}"#,
            "waiting_requests",
            "32",
        ),
        (
            r#"{"enabled":false,"waiting_requests":3}"#,
            "waiting_requests",
            "3",
        ),
        (r#"{"token_usage":0.8}"#, "token_usage", "0.8"),
        (r#"{"token_usage":1.5}"#, "token_usage", "1.5"),
        (r#"{"token_usage":0}"#, "token_usage", "0"),
        (r#"{"token_usage":-0.5}"#, "token_usage", "-0.5"),
        (
            r#"{"enabled":true,"token_usage":0.9}"#,
            "token_usage",
            "0.9",
        ),
        (
            r#"{"enabled":false,"token_usage":0.5}"#,
            "token_usage",
            "0.5",
        ),
        (
            r#"{"waiting_requests":32,"token_usage":0.9,"shed_when_all_vetoed":true}"#,
            "waiting_requests",
            "32",
        ),
        (
            r#"{"enabled":false,"waiting_requests":32,"token_usage":0.9}"#,
            "waiting_requests",
            "32",
        ),
    ] {
        assert_eq!(
            refused(block),
            (
                format!("bindings[0].overload.{field}"),
                value.to_owned(),
                STATED
            ),
            "{block}"
        );
    }
    // The refusal as a document's writer reads it.
    let named = spec_with_binding(r#"{"profile":"p","overload":{"token_usage":0.9}}"#);
    assert_eq!(
        named.validate().err().map(|refusal| refusal.to_string()),
        Some(format!(
            "bindings[0].overload.token_usage = 0.9 is not {STATED}"
        ))
    );
}

#[test]
fn a_shape_error_outranks_what_the_build_lacks() {
    let spec = doc(
        r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":["kv-pressure"],"pick":"best"}},"bindings":[{"profile":"p","overload":{"token_usage":2}}]}"#,
    );
    assert_eq!(bad_field(&spec), "bindings[0].overload.token_usage");
}

#[test]
fn the_plans_section_says_when_a_pairs_decode_is_picked() {
    let document = |plans: &str| {
        doc(&format!(
            r#"{{"apiVersion":"smg.policy/v1"{plans},"profiles":{{"p":{{"pick":"best"}}}}}}"#
        ))
    };
    let applied =
        |spec: &PolicySpec| AppliedSpec::validate(spec).map(|applied| applied.decode_pick());
    // A document that names no rule, and one that names the other rule
    // of the plan alone, pick the decode with the prefill.
    for plans in ["", r#","plans":{"pd-pair":{"pairing":"strict"}}"#] {
        let spec = document(plans);
        assert_eq!(
            spec.plans.pd_pair.decode,
            DecodePick::WithPrefill,
            "{plans}"
        );
        assert_eq!(
            applied(&spec).ok(),
            Some(DecodePick::WithPrefill),
            "{plans}"
        );
    }
    let spec = document(r#","plans":{"pd-pair":{"decode":"with-prefill"}}"#);
    assert_eq!(applied(&spec).ok(), Some(DecodePick::WithPrefill));
    let spec = document(r#","plans":{"pd-pair":{"decode":"after-prefill"}}"#);
    assert_eq!(spec.plans.pd_pair.decode, DecodePick::AfterPrefill);
    assert_eq!(spec.plans.pd_pair.pairing, Pairing::default());
    assert_eq!(applied(&spec).ok(), Some(DecodePick::AfterPrefill));
    // The document written back names the rule, and reads back the same.
    let written = serde_json::to_string(&spec).expect("the document writes");
    assert!(written.contains(r#""decode":"after-prefill""#), "{written}");
    assert_eq!(doc(&written), spec);
    // A start with no document picks with the prefill.
    assert_eq!(
        AppliedSpec::unbound().decode_pick(),
        DecodePick::WithPrefill
    );
    let err = parse_error(
        r#"{"apiVersion":"smg.policy/v1","plans":{"pd-pair":{"decode":"later"}},"profiles":{"p":{"pick":"best"}}}"#,
    );
    assert!(err.contains("unknown variant `later`"), "{err}");
}

#[test]
fn the_plans_section_carries_the_pairing_mode() {
    let spec = doc(
        r#"{"apiVersion":"smg.policy/v1","plans":{"pd-pair":{"pairing":"strict"}},"profiles":{"p":{"pick":"best"}}}"#,
    );
    assert_eq!(spec.plans.pd_pair.pairing, Pairing::Strict);
    assert!(spec.validate().is_ok());
    let err = parse_error(
        r#"{"apiVersion":"smg.policy/v1","plans":{"epd":{}},"profiles":{"p":{"pick":"best"}}}"#,
    );
    assert!(err.contains("unknown field `epd`"), "{err}");
}

#[test]
fn a_document_with_every_section_round_trips_and_writes_the_stage_forms() {
    let text = r#"{
          "apiVersion": "smg.policy/v1",
          "generation": 7,
          "view": { "headers": ["x-session-id", "x-tenant-tier"], "block_size": 16, "boundaries": [256, 1024], "max_block_hashes": 128 },
          "index": { "*": { "kind": "hash", "ttl_s": 180, "max_size": 67108864, "sweep_s": 120 } },
          "profiles": {
            "cache_aware": {
              "filter": ["healthy", "not-overloaded", "not-tried"],
              "score": [
                { "prefix-affinity": { "weight": 0.7, "threshold": 0.3, "decay": 0.9, "source": "merged", "accounting_ttl_ms": 200, "spill": { "abs": 64, "rel": 1.5, "prefix_margin_tokens": 0, "uncached_credit": false } } },
                { "kv-pressure": { "spread": 0.3, "ceiling": 0.9 } },
                { "least-load": { "weight": 0.3, "kv_pressure_weight": 0.15 } },
                { "warmup-slice": { "secs": 60, "share": 0.25, "blocks": 1024, "thin_ratio": 0.5, "divert_every": 8 } },
                { "bucket": { "abs_threshold": 64, "rel_threshold": 1.5, "adjust_interval_s": 5 } },
                { "consistent-hash": { "prefix_tokens": 256, "bounded_load": { "factor": 1.25, "abs_margin": 10 } } },
                { "resident-work": { "request_tokens": 512 } },
                "least-requests",
                "session-affinity"
              ],
              "pick": { "best-with-temperature": { "temperature": 0.05 } }
            },
            "least_load": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" },
            "spare": { "pick": "random" }
          },
          "plans": { "pd-pair": { "pairing": "lenient" } },
          "bindings": [
            { "match": { "model": "pd/*", "phase": ["prefill", "decode"], "tenant": "acme", "priority": { "max": -1 }, "header": { "name": "x-tenant-tier", "values": ["gold"] }, "prompt_tokens": { "min": 1, "max": 4096 }, "labels": ["large"] },
              "plan": "pd-pair", "profiles": { "prefill": "least_load", "decode": "least_load" } },
            { "match": { "model": "*" }, "profile": "cache_aware", "shadow": "spare",
              "canary": { "profile": "least_load", "percent": 5 }, "override": "allowed",
              "overload": { "enabled": true, "shed_when_all_vetoed": false },
              "sessions": { "sticky_header": "x-session-id", "headers": ["x-session-id"], "ttl_s": 600, "sweep_s": 120, "assignment": "delegate", "inflight_cap": 2, "issue_session_id": false },
              "admission": { "enabled": false } }
          ]
        }"#;
    let spec = doc(text);
    let written = serde_json::to_string(&spec).expect("serializes");
    assert_eq!(doc(&written), spec);
    assert!(
        written.contains(r#"{"prefix-affinity":{"weight":0.7"#),
        "{written}"
    );
    assert!(
        written.contains(r#""least-requests","session-affinity"]"#),
        "{written}"
    );
    assert!(written.contains(r#""pick":{"best-with-temperature":{"temperature":0.05}}"#));
    assert!(written.contains(r#""override":"allowed""#));
    // The shape is whole; this build lacks the stages, in document order.
    assert_eq!(
        not_in_build(&spec),
        (
            "profiles.cache_aware.score[0].prefix-affinity".to_string(),
            "the prefix-affinity decay"
        )
    );
    let yaml = serde_yaml::to_string(&spec).expect("yaml");
    let from_yaml: PolicySpec = serde_yaml::from_str(&yaml).expect("yaml round trip");
    assert_eq!(from_yaml, spec);
}

/// A filter entry is written as it was read: a bare name as the string
/// it always was, a name with parameters as its one-key map, the keys
/// of every map in order. The order a file wrote the keys in is in no
/// digest.
#[test]
fn a_filter_entry_round_trips_a_bare_name_as_a_string_and_a_map_as_a_map() {
    let document = |entry: &str| {
        doc(&format!(
            r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{"filter":["healthy",{entry},"not-tried"],"pick":"random"}}}}}}"#
        ))
    };
    let spec = document(r#"{"min-free":{"slots":2,"by":{"zone":3,"class":"a"}}}"#);
    assert_eq!(
        spec.profiles["p"].filter,
        [
            FilterSpec::Name("healthy".into()),
            FilterSpec::Added {
                name: "min-free".into(),
                written: serde_json::json!({ "slots": 2, "by": { "zone": 3, "class": "a" } }),
            },
            "not-tried".into(),
        ]
    );
    let written = serde_json::to_string(&spec).expect("serializes");
    assert!(
            written.contains(
                r#""filter":["healthy",{"min-free":{"by":{"class":"a","zone":3},"slots":2}},"not-tried"]"#
            ),
            "{written}"
        );
    assert_eq!(doc(&written), spec);
    let yaml = serde_yaml::to_string(&spec).expect("yaml");
    let from_yaml: PolicySpec = serde_yaml::from_str(&yaml).expect("yaml round trip");
    assert_eq!(from_yaml, spec);
    // Another order of the same keys is the same document, and another
    // value is another.
    let reordered = document(r#"{"min-free":{"by":{"class":"a","zone":3},"slots":2}}"#);
    assert_eq!(reordered.digest(), spec.digest());
    assert_eq!(
        serde_json::to_string(&reordered).expect("serializes"),
        written
    );
    let moved = document(r#"{"min-free":{"slots":3,"by":{"zone":3,"class":"a"}}}"#);
    assert_ne!(moved.digest(), spec.digest());
    // An entry names one filter.
    for (entry, message) in [
        (r#"{"min-free":{"slots":2},"healthy":{}}"#, "single key"),
        ("{}", "names one filter"),
        (
            "7",
            "a filter name, or one filter name mapped to its parameters",
        ),
    ] {
        let err = parse_error(&format!(
            r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{"filter":[{entry}],"pick":"random"}}}}}}"#
        ));
        assert!(err.contains(message), "{entry}: {err}");
    }
}

/// YAML reads a bare `true`, a number or `~` as no string. A filter of
/// such a name is written in quotes, as a scorer of such a name is.
#[test]
fn a_filter_name_yaml_reads_as_no_string_is_written_in_quotes() {
    let document = |entry: &str| {
        format!("apiVersion: smg.policy/v1\nprofiles: {{p: {{filter: [{entry}], pick: random}}}}\n")
    };
    for bare in ["true", "7", "~"] {
        let refused = serde_yaml::from_str::<PolicySpec>(&document(bare))
            .expect_err("no string")
            .to_string();
        assert!(
            refused.contains("a filter name, or one filter name mapped to its parameters"),
            "{bare}: {refused}"
        );
    }
    let quoted: PolicySpec = serde_yaml::from_str(&document(r#""true""#)).expect("a name");
    assert_eq!(
        quoted.profiles["p"].filter,
        [FilterSpec::Name("true".into())]
    );
}

#[test]
fn the_schema_is_generated_from_the_types() {
    let text = PolicySpec::schema_json().expect("the schema serializes");
    assert!(text.ends_with("}\n"));
    let value: serde_json::Value = serde_json::from_str(&text).expect("JSON");
    assert_eq!(value["title"], "PolicySpec");
    assert!(value["properties"]["bindings"].is_object());
    assert!(value["$defs"]["ScoreStage"]["oneOf"].is_array());
}

#[test]
fn the_index_map_resolves_by_specificity_and_refuses_an_ambiguous_pair() {
    let spec = doc(
        r#"{"apiVersion":"smg.policy/v1","index":{"*":{},"chat-*":{},"chat-7b":{},"*-7b":{}},
                "profiles":{"p":{"pick":"round-robin"}}}"#,
    );
    spec.validate().expect("distinct keys");
    let key = |model: &str| spec.index_for(model).map(|(key, _)| key);
    assert_eq!(key("chat-7b"), Some("chat-7b"));
    assert_eq!(key("chat-13b"), Some("chat-*"));
    assert_eq!(key("llama-7b"), Some("*-7b"));
    assert_eq!(key("other"), Some("*"));
    let mut without_default = spec.clone();
    without_default.index.remove("*");
    assert_eq!(without_default.index_for("other"), None);
    let ambiguous = doc(
        r#"{"apiVersion":"smg.policy/v1","index":{"ab*":{},"*cd":{}},
                "profiles":{"p":{"pick":"round-robin"}}}"#,
    );
    assert_eq!(
        ambiguous.validate().err(),
        Some(SpecError::IndexOverlap {
            first: "*cd".into(),
            second: "ab*".into()
        })
    );
}

/// An index key, with a count of the times it is read.
struct Read<'a>(&'a str, &'a std::cell::Cell<usize>);

impl AsRef<str> for Read<'_> {
    fn as_ref(&self) -> &str {
        self.1.set(self.1.get() + 1);
        self.0
    }
}

/// The overlap pass reads each index key once, whatever their number: a
/// thousand keys are a thousand reads and eight thousand are eight
/// thousand. A pass over every pair reads them by the square.
#[test]
fn the_overlap_pass_reads_each_index_key_once() {
    for entries in [1_000, 8_000] {
        // A prefix and a suffix by turns, of two literal lengths: no two
        // overlap, so no pass ends early.
        let keys: Vec<String> = (0..entries)
            .map(|n| {
                if n % 2 == 0 {
                    format!("p{n:05}*")
                } else {
                    format!("*{n:05}ss")
                }
            })
            .collect();
        let reads = std::cell::Cell::new(0);
        let found = overlap(keys.iter().map(|key| Read(key, &reads)));
        assert!(found.is_none(), "{entries} keys of no overlap");
        assert_eq!(reads.get(), entries, "{entries} keys");
    }
}

/// The pass over every pair, as the check ran before: the measure of
/// the pair the overlap pass names.
fn pairwise<'a>(keys: &[&'a str]) -> Option<(&'a str, &'a str)> {
    let ambiguous = |a: &str, b: &str| matches!((side(a), side(b)), (Some((sa, la)), Some((sb, lb))) if sa != sb && la == lb);
    keys.iter().enumerate().find_map(|(place, first)| {
        let second = keys[place + 1..]
            .iter()
            .find(|second| ambiguous(first, second))?;
        Some((*first, *second))
    })
}

/// The next of a draw's numbers, under `bound`.
fn below(state: &mut u64, bound: u64) -> usize {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    usize::try_from((*state >> 33) % bound).expect("under the bound")
}

/// The pair refused is the pair a pass over every pair names: the first
/// key with a later partner, and its first partner. Four thousand key
/// sets by draw, of ids, prefixes, suffixes and `*`, in the map's order.
#[test]
fn the_overlap_named_is_the_pair_a_pass_over_every_pair_names() {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let (mut overlapping, mut distinct) = (0, 0);
    for _ in 0..4_000 {
        let mut keys = BTreeSet::new();
        for _ in 0..below(&mut state, 12) {
            let literal: String = (0..below(&mut state, 4))
                .map(|_| ["a", "b"][below(&mut state, 2)])
                .collect();
            keys.insert(match below(&mut state, 4) {
                0 => format!("{literal}*"),
                1 => format!("*{literal}"),
                2 => "*".to_owned(),
                _ => format!("{literal}x"),
            });
        }
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        let named = overlap(keys.iter().copied());
        assert_eq!(named, pairwise(&keys), "{keys:?}");
        if named.is_some() {
            overlapping += 1;
        } else {
            distinct += 1;
        }
    }
    assert!(
        overlapping > 500 && distinct > 500,
        "{overlapping} sets overlap, {distinct} do not"
    );
}

#[test]
fn more_profiles_than_a_binding_can_address_are_refused() {
    let mut spec = spec_with_binding(r#"{"profile":"p"}"#);
    let template = spec.profiles["p"].clone();
    for i in 0..=u16::MAX {
        spec.profiles.insert(format!("p{i}"), template.clone());
    }
    assert_eq!(
        spec.validate().err(),
        Some(SpecError::TooManyProfiles {
            count: spec.profiles.len(),
            max: u16::MAX
        })
    );
}

/// Every refusal names the field it is about, in the document's own
/// names; a path the validator built is handed back as it stands.
#[test]
fn every_refusal_names_its_field() {
    let word = || "x".to_owned();
    let profile = || "default".to_owned();
    let built = || "a.b[0]".to_owned();
    let rows = [
        (SpecError::Kind(word()), "kind"),
        (SpecError::ApiVersion(word()), "apiVersion"),
        (
            SpecError::UnknownFilter {
                profile: profile(),
                name: word(),
            },
            "profiles.default.filter",
        ),
        (
            SpecError::UnknownScorer {
                profile: profile(),
                name: word(),
            },
            "profiles.default.score",
        ),
        (
            SpecError::PickerRunsNoScorer {
                profile: profile(),
                picker: word(),
            },
            "profiles.default.score",
        ),
        (
            SpecError::UnknownPicker {
                profile: profile(),
                name: word(),
            },
            "profiles.default.pick",
        ),
        (
            SpecError::PickerNeedsScorer {
                profile: profile(),
                picker: word(),
            },
            "profiles.default.pick",
        ),
        (
            SpecError::UnknownProfile {
                index: 2,
                name: word(),
            },
            "bindings[2]",
        ),
        (SpecError::NoProfile { index: 2 }, "bindings[2].profile"),
        (
            SpecError::ProfileUnderPlan {
                index: 2,
                plan: "pd-pair",
            },
            "bindings[2].profile",
        ),
        (SpecError::ProfileTwice { index: 2 }, "bindings[2].profile"),
        (
            SpecError::SideUnnamed {
                index: 2,
                plan: "pd-pair",
                side: "decode",
            },
            "bindings[2].profiles.decode",
        ),
        (
            SpecError::UnknownSide {
                index: 2,
                side: "left".to_owned(),
            },
            "bindings[2].profiles.left",
        ),
        (
            SpecError::SideNotInPlan {
                index: 2,
                plan: "single",
                side: "decode".to_owned(),
            },
            "bindings[2].profiles.decode",
        ),
        (
            SpecError::PlanPhases {
                index: 2,
                plan: "pd-pair",
                side: "decode",
            },
            "bindings[2].match.phase",
        ),
        (
            SpecError::BadModelGlob {
                index: 2,
                model: word(),
            },
            "bindings[2].match.model",
        ),
        (
            SpecError::BadSessions {
                index: 2,
                name: "plan",
                reason: "so",
            },
            "bindings[2].plan",
        ),
        (
            SpecError::BadSessions {
                index: 2,
                name: "sweep_s",
                reason: "so",
            },
            "bindings[2].sessions.sweep_s",
        ),
        (SpecError::NoBindings { profiles: 2 }, "bindings"),
        (
            SpecError::IndexOverlap {
                first: "a*".to_owned(),
                second: "*b".to_owned(),
            },
            "index.a*",
        ),
        (
            SpecError::HashIndexNeedsBoundaries {
                model: "a*".to_owned(),
            },
            "index.a*.kind",
        ),
        (
            SpecError::TooManyProfiles {
                count: 70_000,
                max: u16::MAX,
            },
            "profiles",
        ),
        (SpecError::Empty, "profiles"),
        (
            SpecError::TooManyHeaders {
                count: 300,
                max: 256,
            },
            "view.headers",
        ),
        (
            SpecError::TooManyListed {
                path: "view.zones",
                count: 300,
                max: 256,
            },
            "view.zones",
        ),
        (
            SpecError::BadField {
                path: built(),
                value: word(),
                reason: "so",
            },
            "a.b[0]",
        ),
        (
            SpecError::NotInBuild {
                path: built(),
                what: "it",
            },
            "a.b[0]",
        ),
        (
            SpecError::DeniedHeader {
                path: built(),
                name: word(),
            },
            "a.b[0]",
        ),
        (
            SpecError::HeaderNotAllowed {
                path: built(),
                name: word(),
            },
            "a.b[0]",
        ),
        (
            SpecError::Duplicate {
                path: built(),
                name: word(),
            },
            "a.b[0]",
        ),
        (
            SpecError::UnknownProfileAt {
                path: built(),
                name: word(),
            },
            "a.b[0]",
        ),
        (
            SpecError::CanaryForm {
                path: built(),
                count: 2,
            },
            "a.b[0]",
        ),
        (
            SpecError::Reserved {
                path: built(),
                reason: "so",
            },
            "a.b[0]",
        ),
        (SpecError::UnknownPreset(word()), ""),
        (SpecError::NoCanonicalForm(word()), ""),
    ];
    for (error, path) in rows {
        assert_eq!(error.field_path(), path, "{error}");
    }
}

/// The path is the validator's own finding's: a document refused for a
/// stage, a weight, a binding or its version reads the field to edit.
#[test]
fn a_refused_document_reads_the_field_to_edit() {
    let refused = |document: &str| {
        let document: PolicySpec = serde_yaml::from_str(document).expect("a document");
        document.validate().expect_err("refused").field_path()
    };
    let profile =
        |stages: &str| format!("apiVersion: smg.policy/v1\nprofiles:\n  default:\n{stages}");
    assert_eq!(
        refused(&profile("    filter: [nowhere]\n    pick: random\n")),
        "profiles.default.filter"
    );
    assert_eq!(
        refused(&profile("    pick: nothing\n")),
        "profiles.default.pick"
    );
    let weight = refused(&profile(
        "    score: [{ least-load: { weight: -1.0 } }]\n    pick: best\n",
    ));
    assert!(weight.starts_with("profiles.default.score[0]"), "{weight}");
    assert_eq!(
        refused(&profile(
            "    score: [{ kv-pressure: { spread: 0.3 } }, kv-pressure]\n    pick: best\n",
        )),
        "profiles.default.score[1].kv-pressure.spread"
    );
    assert_eq!(
        refused(&profile(
            "    score: [{ kv-pressure: { ceiling: 0.9 } }, kv-pressure]\n    pick: best\n",
        )),
        "profiles.default.score[1].kv-pressure.ceiling"
    );
    // Entries of one gate load: the weights apart, a bare name as the
    // entry at its defaults, a limit at 1 and above as off.
    for one_gate in [
        "[{ kv-pressure: { spread: 0.3, weight: 2.0 } }, { kv-pressure: { spread: 0.3 } }]",
        "[{ kv-pressure: { ceiling: 1.0 } }, kv-pressure]",
        "[kv-pressure, { kv-pressure: { spread: 2.0 } }]",
    ] {
        let document = profile(&format!("    score: {one_gate}\n    pick: best\n"));
        let document: PolicySpec = serde_yaml::from_str(&document).expect("a document");
        document.validate().expect(one_gate);
    }
    let bound = |binding: &str| {
        refused(&format!(
            "{}bindings:\n  - {binding}\n",
            profile("    pick: random\n")
        ))
    };
    assert_eq!(
        bound("{ match: { model: \"*\" }, profile: nowhere }"),
        "bindings[0]"
    );
    assert_eq!(
        bound("{ match: { model: \"a*b*\" }, profile: default }"),
        "bindings[0].match.model"
    );
    assert_eq!(
        refused("apiVersion: smg.policy/v9\nprofiles:\n  default:\n    pick: random\n"),
        "apiVersion"
    );
    let unnamed: PolicySpec =
        serde_yaml::from_str("apiVersion: smg.policy/v1\nprofiles:\n  \"\":\n    pick: random\n")
            .expect("a document");
    assert!(matches!(
        unnamed.validate(),
        Err(SpecError::BadField { path, .. }) if path == "profiles"
    ));
}

#[test]
fn a_document_built_in_code_is_this_version_and_validates() {
    let mut profiles = spec_with_binding(r#"{"profile":"p"}"#).profiles;
    profiles.remove("q");
    let spec = PolicySpec::new(profiles);
    assert_eq!(spec.api_version, API_VERSION);
    spec.validate().expect("one profile binds to everything");
    let text = serde_json::to_string(&spec).expect("serializes");
    assert!(!text.contains("kind"), "{text}");
}

/// A warm-up slice follows the profile's one spill gate. Beside a slice
/// a `prefix-affinity` entry that writes another `rel` or `abs` than the
/// profile's first is refused by that limit, wherever the slice stands,
/// and a bare name writes the defaults. Entries of one gate load, and
/// so do two gates with no slice: each scorer spills by its own.
#[test]
fn a_warmup_slice_beside_two_spill_gates_is_refused() {
    let refused = |score: &str| match spec_with_score(score).validate() {
        Err(SpecError::BadField { path, .. }) => path,
        Err(other) => panic!("{score}: {other:?}"),
        Ok(_) => panic!("{score}: the document loads"),
    };
    let (wide, tight) = (
        r#"{"prefix-affinity":{"spill":{"rel":4.0}}}"#,
        r#"{"prefix-affinity":{"spill":{"rel":1.0}}}"#,
    );
    for (score, path) in [
        (
            format!(r#"{wide},{tight},"warmup-slice""#),
            "profiles.p.score[1].prefix-affinity.spill.rel",
        ),
        (
            format!(r#"{tight},"warmup-slice",{wide}"#),
            "profiles.p.score[2].prefix-affinity.spill.rel",
        ),
        (
            format!(r#""warmup-slice",{wide},"prefix-affinity""#),
            "profiles.p.score[2].prefix-affinity.spill.rel",
        ),
        (
            r#"{"prefix-affinity":{"spill":{"abs":8}}},"prefix-affinity","warmup-slice""#
                .to_owned(),
            "profiles.p.score[1].prefix-affinity.spill.abs",
        ),
    ] {
        assert_eq!(refused(&score), path, "{score}");
    }
    for loads in [
            format!("{wide},{tight}"),
            format!(r#"{wide},{wide},"warmup-slice""#),
            r#"{"prefix-affinity":{"spill":{"rel":1.5,"abs":64}}},"prefix-affinity","warmup-slice""#
                .to_owned(),
            r#"{"prefix-affinity":{"spill":{"prefix_margin_tokens":512}}},"prefix-affinity","warmup-slice""#
                .to_owned(),
        ] {
            assert!(spec_with_score(&loads).validate().is_ok(), "{loads}");
        }
}

/// The spill parameters of a prefix-affinity entry reach the gates the
/// profile's stages share: under a relative threshold of four a holder
/// three times as loaded as the mean keeps its request; under the
/// default (1.5, plus the margin) it spills. Of two entries each spills
/// by its own gate.
#[test]
fn the_spill_parameters_of_the_spec_reach_the_profiles_gates() {
    use crate::engine::{
        AffinityFacts, Candidate, Constraints, Health, PoolFacts, RequestFacts, Scratch, WorkerId,
    };
    let hashes = [0u64; 16];
    // The first worker holds the whole request; the column says so.
    let held = [
        AffinityFacts {
            overlap_blocks: 16,
            ..AffinityFacts::default()
        },
        AffinityFacts::default(),
        AffinityFacts::default(),
    ];
    let facts = RequestFacts {
        model: "m",
        block_hashes: &hashes,
        prompt_tokens: 256,
        seed: 7,
        affinity: &held,
        ..RequestFacts::default()
    };
    let row = |position: u32, in_flight: u32| Candidate {
        position,
        id: WorkerId(100 + u64::from(position)),
        health: Health {
            healthy: true,
            ..Health::default()
        },
        in_flight,
        ..Candidate::default()
    };
    let pool = [row(0, 20), row(1, 0), row(2, 0)];
    let leader = |score: &str| {
        let engine = spec_with_score(score).validate().expect("a valid profile");
        let decision = engine.decide(
            &facts,
            &PoolFacts::default(),
            &pool,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        decision.pick().expect("a pick").targets.as_slice()[0].position
    };
    assert_eq!(
        leader(r#"{"prefix-affinity":{"spill":{"rel":4.0}}}"#),
        0,
        "a holder under the configured gate keeps its request"
    );
    assert_ne!(
        leader(r#"{"prefix-affinity":{}}"#),
        0,
        "under the default gate the loaded holder spills"
    );
    // Two entries, each under its own gate: the heavier one keeps the
    // holder whichever is written first.
    let heavy = r#"{"prefix-affinity":{"weight":4.0,"spill":{"rel":4.0}}}"#;
    let light = r#"{"prefix-affinity":{"weight":1.0}}"#;
    for score in [format!("{light},{heavy}"), format!("{heavy},{light}")] {
        assert_eq!(leader(&score), 0, "{score}");
    }
}

/// The digest names the document, not the file: the form (YAML or JSON,
/// the order of the keys, a comment) and the `generation` line leave it
/// as it was, and any other field moves it. Its input is the document's
/// JSON with the line zeroed, under SHA-256.
#[test]
fn a_digest_holds_across_the_form_and_the_line_and_moves_with_any_other_field() {
    let written = doc(
        r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"filter":["healthy"],"score":["least-load"],"pick":"best"}}}"#,
    );
    let digest = written.digest().expect("a digest");

    let numbered = PolicySpec {
        generation: 7,
        ..written.clone()
    };
    assert_eq!(numbered.digest(), Ok(digest), "the line is not in it");
    let yaml: PolicySpec = serde_yaml::from_str(
            "profiles:\n  p:\n    pick: best   # a comment\n    score: [least-load]\n    filter: [healthy]\ngeneration: 3\napiVersion: smg.policy/v1\n",
        )
        .expect("the same document in YAML, its keys in another order");
    assert_eq!(yaml.digest(), Ok(digest), "the form is not in it");

    let mut moved = written.clone();
    moved
        .profiles
        .get_mut("p")
        .expect("the profile")
        .filter
        .push("not-tried".into());
    assert_ne!(moved.digest(), Ok(digest), "a filter more is in it");

    // The one line a document can write that no canonical form carries:
    // the validator refuses it, and the digest does not read it.
    let kinded: PolicySpec = serde_yaml::from_str(
            "kind: PolicySpec\napiVersion: smg.policy/v1\nprofiles:\n  p:\n    filter: [healthy]\n    score: [least-load]\n    pick: best\n",
        )
        .expect("a kind line parses");
    assert_eq!(kinded.digest(), Ok(digest), "a kind line is not in it");
    assert_eq!(
        kinded.validate().err(),
        Some(SpecError::Kind("PolicySpec".into()))
    );

    let canonical = serde_json::to_string(&PolicySpec {
        generation: 0,
        ..numbered
    })
    .expect("serializes");
    assert!(
        canonical.starts_with(r#"{"apiVersion":"smg.policy/v1","generation":0,"#),
        "{canonical}"
    );
    let hex: String = Sha256::digest(canonical.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(digest.to_string(), format!("sha256:{hex}"));
}

/// The digest's "any", over the type's fields: a change to each field
/// the schema names moves the digest, the `generation` line alone
/// excepted. The names are the schema's, so a field added to the type
/// joins the case, and fails it until a change to it is shown here.
#[test]
fn a_digest_moves_with_every_field_the_schema_names_but_the_line() {
    let base = PolicySpec::preset("least_load").expect("a preset");
    let digest = base.digest().expect("a digest");

    let mut bindings = base.bindings.clone();
    bindings.push(BindingSpec::default());
    let mut index = base.index.clone();
    index.insert("another*".to_owned(), IndexSpec::default());
    let held = SignalSpec {
        unit: "tokens".to_owned(),
        max_age_ms: None,
        across_ranks: AcrossRanks::default(),
    };
    let signals = BTreeMap::from([("held".to_owned(), held)]);
    let mut profiles = base.profiles.clone();
    let profile = profiles.values().next().expect("a profile").clone();
    profiles.insert("another".to_owned(), profile);
    let plans = PlansSpec {
        pd_pair: PdPairPlanSpec {
            pairing: Pairing::Off,
            ..PdPairPlanSpec::default()
        },
    };
    let view = ViewSpec {
        headers: vec!["x-team".to_owned()],
        ..base.view.clone()
    };
    let changed = [
        (
            "apiVersion",
            PolicySpec {
                api_version: "smg.policy/v2".to_owned(),
                ..base.clone()
            },
        ),
        (
            "bindings",
            PolicySpec {
                bindings,
                ..base.clone()
            },
        ),
        (
            "explain",
            PolicySpec {
                explain: ExplainSpec { sample: 2 },
                ..base.clone()
            },
        ),
        (
            "generation",
            PolicySpec {
                generation: base.generation + 9,
                ..base.clone()
            },
        ),
        (
            "index",
            PolicySpec {
                index,
                ..base.clone()
            },
        ),
        (
            "plans",
            PolicySpec {
                plans,
                ..base.clone()
            },
        ),
        (
            "profiles",
            PolicySpec {
                profiles,
                ..base.clone()
            },
        ),
        (
            "signals",
            PolicySpec {
                signals,
                ..base.clone()
            },
        ),
        (
            "view",
            PolicySpec {
                view,
                ..base.clone()
            },
        ),
    ];

    let schema = serde_json::to_value(PolicySpec::schema()).expect("the schema is JSON");
    let named: BTreeSet<&str> = schema["properties"]
        .as_object()
        .expect("the type's fields")
        .keys()
        .map(String::as_str)
        .collect();
    let shown: BTreeSet<&str> = changed.iter().map(|(field, _)| *field).collect();
    assert_eq!(named, shown, "one change a field the schema names");
    for (field, document) in &changed {
        assert_ne!(document, &base, "{field} is changed");
        let moved = document.digest().expect("a digest") != digest;
        assert_eq!(moved, *field != "generation", "{field}");
    }
}

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

/// A validated generation carries its document's digest; the unbound
/// start has no document and no digest.
#[test]
fn an_applied_generation_carries_its_documents_digest_and_the_unbound_start_none() {
    let document = PolicySpec::preset("least_load").expect("a preset");
    let applied = AppliedSpec::validate(&document).expect("a preset validates");
    assert_eq!(applied.digest(), Some(document.digest().expect("a digest")));
    assert_eq!(AppliedSpec::unbound().digest(), None);
}

#[test]
fn validate_in_builds_the_registered_scorer_for_a_store_decision() {
    use crate::{
        engine::{Health, PoolFacts, Scratch, WorkerId},
        store::MemoryStore,
    };

    const SCORER: &str = "prefer-idle";

    #[derive(Debug)]
    struct PreferIdle;

    impl Lifecycle for PreferIdle {}

    impl Scorer for PreferIdle {
        fn name(&self) -> &'static str {
            SCORER
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            for (&index, score) in kept.iter().zip(out) {
                *score = if candidates[index as usize].in_flight == 0 {
                    1.0
                } else {
                    0.0
                };
            }
            TieBand::NORMALIZED
        }
    }

    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer(SCORER, || Box::new(PreferIdle))
        .expect("a free scorer name");
    let mut document = spec_with_score(&format!(r#""{SCORER}""#));
    document.generation = 43;
    let applied = AppliedSpec::validate_in(&document, &catalog, &[])
        .expect("the registered scorer validates");
    let store = MemoryStore::new();
    store.apply(applied).expect("a newer generation");

    store.deciding(|world| {
        let spec = world.spec();
        let pool = PoolFacts {
            pool_size: 2,
            generation: spec.generation(),
            ..PoolFacts::default()
        };
        let mut scratch = Scratch::new();
        for idle in [0, 1] {
            let row = |position| Candidate {
                position,
                id: WorkerId(137 + u64::from(position)),
                health: Health {
                    healthy: true,
                    ..Health::default()
                },
                in_flight: u32::from(position != idle),
                ..Candidate::default()
            };
            let decision = spec.engine().decide(
                &RequestFacts::default(),
                &pool,
                &[row(0), row(1)],
                &Constraints::default(),
                &mut scratch,
            );
            let targets = &decision.pick().expect("a pick").targets;
            assert_eq!(targets.first().expect("a target").position, idle);
        }
    });
}

#[test]
fn validate_in_refuses_headers_the_host_denies() {
    let document = spec_with_view(r#"{"headers":["x-private-context"]}"#);
    let catalog = StageRegistry::builtin();
    let denied = ["X-Private-Context"];
    let expected = SpecError::DeniedHeader {
        path: "view.headers[0]".into(),
        name: "x-private-context".into(),
    };
    assert!(AppliedSpec::validate_in(&document, &catalog, &[]).is_ok());
    assert_eq!(
        AppliedSpec::validate_in(&document, &catalog, &denied).expect_err("a denied header"),
        expected
    );
    assert_eq!(
        AppliedSpec::validate_with(&document, &denied).expect_err("a denied header"),
        expected
    );
}

#[test]
fn validate_in_preserves_the_stock_generation_digest_and_depth() {
    let mut document = spec_with_score(r#"{"consistent-hash":{"prefix_tokens":143}}"#);
    document.generation = 47;
    let catalog = StageRegistry::builtin();
    let applied = AppliedSpec::validate_in(&document, &catalog, &[])
        .expect("a stock scorer validates in the stock catalog");
    let stock = AppliedSpec::validate_with(&document, &[]).expect("a stock scorer validates");
    let depth = |spec: &AppliedSpec| spec.block_hashes_read(HashKind::Token, 11);
    assert_eq!(applied.generation(), 47);
    assert_eq!(applied.generation(), stock.generation());
    assert_eq!(applied.digest(), Some(document.digest().expect("a digest")));
    assert_eq!(applied.digest(), stock.digest());
    assert_eq!(depth(&applied), 13);
    assert_eq!(depth(&applied), depth(&stock));
}

#[test]
fn validate_in_leaves_an_unnumbered_document_at_generation_zero() {
    use crate::store::{MemoryStore, StaleGeneration};

    let document = spec_with_view("{}");
    let catalog = StageRegistry::builtin();
    let validate = |document: &PolicySpec| {
        AppliedSpec::validate_in(document, &catalog, &[]).expect("the document validates")
    };
    let store = MemoryStore::new();
    let standing = 71;
    store
        .apply(validate(&PolicySpec {
            generation: standing,
            ..document.clone()
        }))
        .expect("a newer generation");

    let unnumbered = validate(&document);
    assert_eq!(unnumbered.generation(), 0);
    assert_eq!(
        store.apply(unnumbered),
        Err(StaleGeneration {
            standing,
            offered: 0,
        })
    );
    assert_eq!(store.snapshot().spec().generation(), standing);
}

/// A digest handed on is the generation's, as given: the door's apply
/// digests its document once.
#[cfg(feature = "controller")]
#[test]
fn a_digest_handed_on_is_carried_as_given() {
    let document = PolicySpec::preset("least_load").expect("a preset");
    let handed = PolicySpec::preset("random")
        .expect("a preset")
        .digest()
        .expect("a digest");
    assert_ne!(document.digest(), Ok(handed));
    let applied = AppliedSpec::validate_digested(&document, handed, &StageRegistry::builtin(), &[])
        .expect("a preset validates");
    assert_eq!(applied.digest(), Some(handed));
}

/// A digest handed on moves nothing else of the generation: its pairing
/// mode is the document's, as where the generation digests the document
/// itself.
#[cfg(feature = "controller")]
#[test]
fn a_generation_under_a_handed_digest_takes_its_documents_pairing_mode() {
    let mut document = PolicySpec::preset("least_load").expect("a preset");
    document.plans.pd_pair.pairing = Pairing::Strict;
    let digest = document.digest().expect("a digest");
    let handed = AppliedSpec::validate_digested(&document, digest, &StageRegistry::builtin(), &[])
        .expect("a preset validates");
    assert_eq!(handed.pairing(), Pairing::Strict);
    let own = AppliedSpec::validate(&document).expect("a preset validates");
    assert_eq!(own.pairing(), Pairing::Strict);
}

/// `best` takes a sample in the parameter form and stays the unit by its
/// name: the sample is a whole number from 1 to 64 and the profile's
/// need, and the cache-aware preset states sixteen.
#[test]
fn the_best_picker_takes_a_sample_in_the_parameter_form() {
    let sampled = spec_with_pick(r#"{"best":{"sample":16}}"#);
    let engine = sampled.validate().expect("validates");
    assert_eq!(engine.needs().sample, 16);
    let written = serde_json::to_string(&sampled).expect("serializes");
    assert!(
        written.contains(r#""pick":{"best":{"sample":16}}"#),
        "{written}"
    );
    assert_eq!(doc(&written), sampled);
    // By its name it is the unit: the whole pool, and no sample needed.
    let unit = spec_with_pick(r#""best""#);
    assert_eq!(unit.validate().expect("validates").needs().sample, 0);
    for sample in [0, 65] {
        let pick = format!(r#"{{"best":{{"sample":{sample}}}}}"#);
        assert_eq!(
            bad_field(&spec_with_pick(&pick)),
            "profiles.p.pick.best.sample"
        );
    }
    let err = parse_error(
        r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":{"best":{"draws":4}}}}}"#,
    );
    assert!(err.contains("unknown field `draws`"), "{err}");
    let preset = PolicySpec::preset("cache_aware").expect("a preset");
    assert_eq!(preset.validate().expect("validates").needs().sample, 16);
}

/// The guard of each stock stage's word on what the gateways report of
/// leases
/// ([`Filter::refuses_by_reports`](crate::engine::Filter::refuses_by_reports)),
/// over the catalog's own lists: a profile of each stock stage says no
/// line. The same lists hold the fewest requests to the rows' counts:
/// its profiles decide otherwise over rows that carry none.
#[test]
fn every_catalog_stage_says_no_line_on_what_gateways_report_of_leases() {
    use crate::engine::{view::Identity, Health, PoolFacts, Scratch, WorkerId};
    let sessions = r#""score":["fewest-sessions"]"#;
    let scored = |pick: &str| format!(r#"{{{sessions},"pick":{pick}}}"#);
    let alone = |pick: &str| format!(r#"{{"filter":["healthy"],"pick":{pick}}}"#);
    // The forms that take parameters, then each other name of the
    // lists in the forms it validates in. A name this build lacks is
    // refused by name in each of its forms and is no reader.
    let bounded = r#"{"consistent-hash":{"bounded_load":{"factor":1.0,"abs_margin":0}}}"#;
    let tempered = r#"{"best-with-temperature":{"temperature":1.0}}"#;
    let mut profiles: Vec<Vec<String>> = vec![
        vec![r#"{"score":[{"resident-work":{"request_tokens":512}}],"pick":"best"}"#.into()],
        vec![
            r#"{"score":[{"least-cost":{"terms":{"candidate.in_flight":3}}}],"pick":"best"}"#
                .into(),
        ],
        vec![format!(r#"{{"score":[{bounded}],"pick":"best"}}"#)],
        vec![scored(tempered), alone(tempered)],
    ];
    let apart = ["resident-work", "least-cost", "best-with-temperature"];

    for name in FILTERS {
        profiles.push(vec![format!(
            r#"{{"filter":["{name}"],{sessions},"pick":"best"}}"#
        )]);
    }
    for name in SCORERS.iter().filter(|name| !apart.contains(name)) {
        profiles.push(vec![format!(r#"{{"score":["{name}"],"pick":"best"}}"#)]);
    }
    for name in PICKERS.iter().filter(|name| !apart.contains(name)) {
        let pick = format!("{name:?}");
        profiles.push(vec![
            scored(&pick),
            alone(&pick),
            format!(r#"{{"pick":{pick}}}"#),
        ]);
    }
    let document = |profile: &String| {
        doc(&format!(
            r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{profile}}}}}"#
        ))
    };
    let lacked =
        |form: &String| matches!(document(form).validate(), Err(SpecError::NotInBuild { .. }));
    profiles.retain(|forms| !forms.iter().all(lacked));
    let build = |forms: &[String]| -> Engine {
        let built = forms.iter().find_map(|form| document(form).validate().ok());
        built.unwrap_or_else(|| panic!("no form validates: {forms:?}"))
    };
    // Counts that part the rows, and rows that read none.
    let counts = [5, 1, 3, 0, 4, 2, 6, 1];
    let row = |(position, in_flight): (u32, u32)| Candidate {
        position,
        id: WorkerId(100 + u64::from(position)),
        in_flight,
        health: Health {
            healthy: true,
            ..Health::default()
        },
        ..Candidate::default()
    };
    let rows = |held: [u32; 8]| -> Vec<Candidate> { (0..).zip(held).map(row).collect() };
    let (own, bare) = (rows(counts), rows([0; 8]));
    let pool = PoolFacts {
        pool_size: 8,
        generation: 1,
        ..PoolFacts::default()
    };
    // One engine a feed: a stage that keeps a rotation starts each
    // feed's decisions at the same place.
    let run = |forms: &[String], rows: &[Candidate]| {
        let engine = build(forms);
        let picks = |seed: u64| {
            let facts = RequestFacts {
                model: "m",
                session_key: Identity::hashed(seed),
                seed,
                ..RequestFacts::default()
            };
            let cx = Constraints::default();
            let decision = engine.decide(&facts, &pool, rows, &cx, &mut Scratch::new());
            let picked = decision.pick();
            picked.map(|pick| {
                pick.targets
                    .as_slice()
                    .iter()
                    .map(|t| t.id)
                    .collect::<Vec<_>>()
            })
        };
        (0..32).map(picks).collect::<Vec<_>>()
    };
    for forms in &profiles {
        // The counts part the rows: the fewest requests answer by them.
        let moved = run(forms, &own) != run(forms, &bare);
        assert!(moved || !forms[0].contains("least-requests"), "{forms:?}");
        // No stock stage takes a silent stage's side: each says it
        // refuses by nothing a gateway reports of leases, so a decision
        // under it applies nothing first.
        let (_, _, applies_first) = build(forms).reads_for("m", Phase::Single);
        assert!(!applies_first, "a stock stage says no line: {forms:?}");
    }
}
