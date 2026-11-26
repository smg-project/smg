use super::*;

/// A host names rows for the live profile, by its cap, its warm-up
/// ruling and the record of sent prefixes. A shadow reads them where
/// its own profile would be named the same, and reads the view
/// without them where that profile reads no column. It is named rows
/// of its own where it differs by its cap, or the live profile is
/// named none, and the pool's ruling and record are its own. Behind
/// a live profile that is named no row the record is the shadow's,
/// whatever it reads a request's holders from. It is not asked
/// anywhere else.
#[test]
fn a_shadow_reads_its_live_rows_only_where_it_would_be_handed_them() {
    use crate::document::spec::{
        AffinitySource, BestParams, PickStage, PrefixAffinityParams, ProfileSpec, ScoreStage,
        WarmupSliceParams,
    };
    type Turn<'a> = &'a dyn Fn(&mut ProfileSpec);
    let rows = |live: &str, shadow: &str, turn: Turn<'_>| {
        let mut document = PolicySpec::preset(live).expect("a preset");
        let next = PolicySpec::preset(shadow).expect("a preset");
        let profile = next.profiles.into_values().next();
        let mut profile = profile.expect("the preset's profile");
        turn(&mut profile);
        document.profiles.insert("second".into(), profile);
        document.bindings[0].shadow = Some(spec::ShadowSpec::Profile("second".into()));
        let engine = document.validate().expect("a binding with a shadow");
        let due = engine.shadow_for("m", Phase::Single);
        due.map(|(at, rows)| (rows, engine.shadow_cap_at(at)))
    };
    let as_it_is = |_: &mut ProfileSpec| {};
    let sample = |profile: &mut ProfileSpec| {
        profile.pick = PickStage::Best(BestParams { sample: 8 }).into();
    };
    let ruling = |profile: &mut ProfileSpec| {
        let slice = WarmupSliceParams {
            weight: 8.0,
            secs: Some(120),
            ..WarmupSliceParams::default()
        };
        profile.score[0] = ScoreStage::WarmupSlice(slice).into();
    };
    // The prefix stage reads the record of sent prefixes alone.
    let record = |profile: &mut ProfileSpec| {
        let prefix = PrefixAffinityParams {
            weight: 4.0,
            source: Some(AffinitySource::Predictor),
            ..PrefixAffinityParams::default()
        };
        profile.score[1] = ScoreStage::PrefixAffinity(prefix).into();
    };
    // The prefix stage reads the index of cache events alone.
    let index = |profile: &mut ProfileSpec| {
        let prefix = PrefixAffinityParams {
            weight: 4.0,
            source: Some(AffinitySource::Index),
            ..PrefixAffinityParams::default()
        };
        profile.score[1] = ScoreStage::PrefixAffinity(prefix).into();
    };
    // Each pairing's word, and the shadow profile's own cap beside it.
    let (aware, least) = ("cache_aware", "least_load");
    let cases: [(&str, &str, Turn<'_>, ShadowRows, u32); 10] = [
        (aware, aware, &as_it_is, ShadowRows::Same, 16),
        (aware, aware, &record, ShadowRows::Apart, 16),
        (aware, aware, &index, ShadowRows::Apart, 16),
        (least, "random", &as_it_is, ShadowRows::Same, 0),
        (aware, aware, &sample, ShadowRows::Own, 8),
        (aware, aware, &ruling, ShadowRows::Apart, 16),
        (least, aware, &as_it_is, ShadowRows::Own, 16),
        (least, aware, &ruling, ShadowRows::Own, 16),
        (least, aware, &record, ShadowRows::Own, 16),
        (aware, least, &as_it_is, ShadowRows::Unnamed, 0),
    ];
    for (live, shadow, turn, wanted, cap) in cases {
        assert_eq!(
            rows(live, shadow, turn),
            Some((wanted, cap)),
            "{shadow} behind {live}"
        );
    }
}

/// A naming that fills its cap is held to its profile's own test of a
/// row: a shadow of other filters than its live profile's would be
/// named other rows where a full cap is refused whole. Under a sample
/// such a shadow is named rows of its own. So is one whose filters
/// are asked by a call each: no two such tests are known to be one.
/// Under no sample no cap fills, and the shadow reads its live
/// decision's rows whatever its filters. Two profiles whose filters
/// are asked of no row share one naming whatever their filters: no
/// test reads past either's cap. One such profile beside one whose
/// filters are asked does not.
#[test]
fn a_shadow_of_other_filters_is_named_rows_of_its_own_under_a_sample() {
    use crate::{
        document::spec::{BestParams, PickStage, ProfileSpec, ShadowSpec},
        stages::registry::StageRegistry,
    };
    /// A filter that is shown the whole view: the default word. The
    /// engine asks its profile's filters of no row ahead of a naming.
    #[derive(Debug)]
    struct Whole;
    impl Filter for Whole {
        fn name(&self) -> &'static str {
            "whole"
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }
    }
    type Turn<'a> = &'a dyn Fn(&mut ProfileSpec);
    let mut catalog = StageRegistry::builtin();
    let added = catalog.filter("whole", || Box::new(Whole));
    added.expect("a free name");
    let rows = |sample: u32, both: Turn<'_>, shadow: Turn<'_>| {
        let mut document = PolicySpec::preset("cache_aware").expect("a preset");
        let live = document.profiles.values_mut().next();
        let live = live.expect("the preset's profile");
        live.pick = match sample {
            0 => "best".into(),
            sample => PickStage::Best(BestParams { sample }).into(),
        };
        both(live);
        let mut second = live.clone();
        shadow(&mut second);
        document.profiles.insert("second".into(), second);
        document.bindings[0].shadow = Some(ShadowSpec::Profile("second".into()));
        let engine = document.validate_in(&catalog, &[]);
        let engine = engine.expect("a binding with a shadow");
        engine.shadow_for("m", Phase::Single).map(|(_, rows)| rows)
    };
    let as_it_is = |_: &mut ProfileSpec| {};
    let whole = |profile: &mut ProfileSpec| profile.filter.push("whole".into());
    // The preset's first filter refuses the rows that are not ready.
    let first_gone = |profile: &mut ProfileSpec| {
        profile.filter.remove(0);
    };
    // The preset's last filter refuses the rows a request tried.
    let fewer = |profile: &mut ProfileSpec| {
        profile.filter.pop();
    };
    // A rule of the document is a stage of its own: the engine folds
    // its profile into no one test.
    let ruled = |profile: &mut ProfileSpec| {
        let rule = r#"{"has-room":{"used":{"candidate.in_flight":1},"capacity":8}}"#;
        profile
            .filter
            .push(serde_json::from_str(rule).expect("a rule"));
    };
    let cases: [(u32, Turn<'_>, Turn<'_>, ShadowRows); 7] = [
        (16, &as_it_is, &as_it_is, ShadowRows::Same),
        (16, &as_it_is, &fewer, ShadowRows::Own),
        (16, &as_it_is, &ruled, ShadowRows::Own),
        (16, &ruled, &as_it_is, ShadowRows::Own),
        (0, &as_it_is, &fewer, ShadowRows::Same),
        // Neither profile's filters are asked of a row: both namings
        // stand as made, whatever the filters.
        (16, &whole, &first_gone, ShadowRows::Same),
        // The live profile's are not, and its shadow's are: the
        // shadow drops the filter that is shown the whole view.
        (16, &whole, &fewer, ShadowRows::Own),
    ];
    for (at, (sample, both, shadow, wanted)) in cases.into_iter().enumerate() {
        assert_eq!(rows(sample, both, shadow), Some(wanted), "pairing {at}");
    }
}

/// A pool has one ruling of its warming rows: the live profile's, and
/// where that profile rules by none and its shadow is named rows of
/// its own, the shadow profile's.
#[test]
fn a_pool_is_ruled_for_its_shadow_where_the_live_profile_rules_by_none() {
    let aware = PolicySpec::preset("cache_aware").expect("a preset");
    let second = aware.profiles.values().next();
    let second = second.expect("the preset's profile").clone();
    let aware = aware.validate().expect("the preset validates");
    let rule = aware.warm_up_rule_for("m", Phase::Single);
    assert!(rule.is_some());
    assert_eq!(aware.pool_rule_for("m", Phase::Single), rule);

    let mut document = PolicySpec::preset("least_load").expect("a preset");
    let alone = document.validate().expect("the preset validates");
    assert_eq!(alone.pool_rule_for("m", Phase::Single), None);
    document.profiles.insert("second".into(), second);
    document.bindings[0].shadow = Some(spec::ShadowSpec::Profile("second".into()));
    let engine = document.validate().expect("a binding with a shadow");
    assert_eq!(engine.warm_up_rule_for("m", Phase::Single), None);
    assert_eq!(engine.pool_rule_for("m", Phase::Single), rule);
}

/// A pool keeps one record of sent prefixes: the live profile's, and
/// where that profile is named no row and its shadow is named rows of
/// its own, the shadow profile's. A document that names no such
/// shadow keeps none there.
#[test]
fn a_pool_keeps_the_record_for_its_shadow_where_the_live_profile_is_named_no_row() {
    use crate::document::spec::{
        AffinitySource, PrefixAffinityParams, ScoreStage, ShadowSpec, WarmupSliceParams,
    };
    let aware = PolicySpec::preset("cache_aware").expect("a preset");
    let second = aware.profiles.values().next();
    let second = second.expect("the preset's profile").clone();
    let aware = aware.validate().expect("the preset validates");
    // The preset reads each worker by what it has: the index's
    // holders keep their place ahead of the record's.
    assert_eq!(aware.pool_record_for("m", Phase::Single), Some(false));

    let mut document = PolicySpec::preset("least_load").expect("a preset");
    let alone = document.validate().expect("the preset validates");
    assert_eq!(alone.pool_record_for("m", Phase::Single), None);
    document.profiles.insert("second".into(), second.clone());
    document.bindings[0].shadow = Some(ShadowSpec::Profile("second".into()));
    let engine = document.validate().expect("a binding with a shadow");
    assert!(!engine.reads_sent_for("m", Phase::Single));
    assert_eq!(engine.pool_record_for("m", Phase::Single), Some(false));

    // A shadow that reads the record alone is named its holders
    // ahead of the index's.
    let mut by_record = second;
    let prefix = PrefixAffinityParams {
        weight: 4.0,
        source: Some(AffinitySource::Predictor),
        ..PrefixAffinityParams::default()
    };
    by_record.score[1] = ScoreStage::PrefixAffinity(prefix).into();
    document.profiles.insert("second".into(), by_record);
    let engine = document.validate().expect("a binding with a shadow");
    assert_eq!(engine.pool_record_for("m", Phase::Single), Some(true));

    // A live profile that is named no row and reads the record keeps
    // it by its own word: here two warm-up stages of different
    // windows leave it the column whole. A shadow that reads the
    // record another way is not named rows of its own beside it.
    let aware = PolicySpec::preset("cache_aware").expect("a preset");
    let live = aware.profiles.into_values().next();
    let mut live = live.expect("the preset's profile");
    let slice = WarmupSliceParams {
        weight: 1.0,
        secs: Some(7),
        ..WarmupSliceParams::default()
    };
    live.score.push(ScoreStage::WarmupSlice(slice).into());
    document.profiles.insert("live".into(), live.clone());
    let unread = document.bindings[0].profile.replace("live".into());
    let engine = document.validate().expect("a live profile named no row");
    assert_eq!(engine.affinity_rows_cap("m", Phase::Single), 0);
    assert_eq!(engine.pool_record_for("m", Phase::Single), Some(false));
    let rows = engine.shadow_for("m", Phase::Single);
    assert_eq!(rows.map(|(_, rows)| rows), Some(ShadowRows::Apart));

    // That profile as the shadow of one that reads no record: its
    // pool keeps none, so the shadow is not asked over a column that
    // lacks the record it would read live.
    document.profiles.insert("second".into(), live);
    document.bindings[0].profile = unread;
    let engine = document.validate().expect("a shadow named no row");
    assert_eq!(engine.pool_record_for("m", Phase::Single), None);
    let rows = engine.shadow_for("m", Phase::Single);
    assert_eq!(rows.map(|(_, rows)| rows), Some(ShadowRows::Apart));
}

/// A model keeps one record of sent prefixes under one bound. A
/// shadow named rows of its own behind a live profile that is named
/// none would have its pool keep the record for it, and write under
/// the bound of every live reader of that model. So it is not asked
/// where another binding's live profile reads the record for a model
/// its binding decides, and its pool keeps none. It is asked where
/// that profile reads the index alone, where the two bindings share
/// no model, and where one of them takes every pool of the models
/// they share.
#[test]
fn a_shadow_is_not_asked_where_a_live_profile_reads_its_models_record() {
    use crate::document::spec::{AffinitySource, PrefixAffinityParams, ScoreStage, ShadowSpec};
    type Side<'a> = (&'a str, &'a [Phase]);
    let word = |own: Side<'_>, other: Side<'_>, first: bool, index: bool, model: &str| {
        let aware = PolicySpec::preset("cache_aware").expect("a preset");
        let aware = aware.profiles.into_values().next();
        let aware = aware.expect("the preset's profile");
        let mut live = aware.clone();
        if index {
            // The live profile's prefix stage reads the index alone.
            let prefix = PrefixAffinityParams {
                weight: 4.0,
                source: Some(AffinitySource::Index),
                ..PrefixAffinityParams::default()
            };
            live.score[1] = ScoreStage::PrefixAffinity(prefix).into();
        }
        let mut document = PolicySpec::preset("least_load").expect("a preset");
        document.profiles.insert("second".into(), aware);
        document.profiles.insert("live".into(), live);
        let mut reader = document.bindings[0].clone();
        reader.matcher.model = other.0.into();
        reader.matcher.phase = other.1.to_vec();
        reader.profile = Some("live".into());
        let shadowed = &mut document.bindings[0];
        shadowed.matcher.model = own.0.into();
        shadowed.matcher.phase = own.1.to_vec();
        shadowed.shadow = Some(ShadowSpec::Profile("second".into()));
        document.bindings.insert(usize::from(first), reader);
        let engine = document.validate().expect("two bindings");
        let rows = engine.shadow_for(model, Phase::Single);
        let record = engine.pool_record_for(model, Phase::Single);
        (rows.map(|(_, rows)| rows), record)
    };
    let (single, prefill) = (&[Phase::Single][..], &[Phase::Prefill][..]);
    let each: &[Phase] = &[];
    let (own, apart) = (ShadowRows::Own, ShadowRows::Apart);
    // The shadowed binding, the other one, whether the shadowed one
    // is listed first, whether the other reads the index alone, the
    // model asked of, and the word.
    let cases: [(Side<'_>, Side<'_>, bool, bool, &str, ShadowRows); 8] = [
        // Another pool of the model is a live reader's.
        (("*", single), ("*", prefill), true, false, "m", apart),
        (("*", single), ("*", prefill), false, false, "m", apart),
        (("m*", single), ("*x", prefill), true, false, "m", apart),
        // One word a binding: another of its models is skipped too.
        (("*", each), ("m", prefill), false, false, "n", apart),
        // No record is read live beside it.
        (("*", single), ("*", prefill), true, true, "m", own),
        (("m", single), ("n", prefill), true, false, "m", own),
        // One binding takes every pool of the models they share.
        (("m", each), ("*", each), true, false, "m", own),
        (("*", each), ("m", each), false, false, "n", own),
    ];
    for (shadowed, other, first, index, model, wanted) in cases {
        let record = (wanted == own).then_some(false);
        assert_eq!(
            word(shadowed, other, first, index, model),
            (Some(wanted), record),
            "{shadowed:?} beside {other:?}"
        );
    }
}

/// A decode pool is read by its model's pair, whose decode side runs
/// under the pair's binding. So the pool keeps the record that side's
/// profile reads where a binding of the decode phase listed ahead
/// reads none, and a shadow of that binding is not asked beside the
/// side. Its word there is never the side's: the first binding's
/// decisions read no record, alone or not. The first binding's word
/// stands where its own profile reads
/// the record, a pair whose decode side reads none adds nothing, and
/// a decode binding listed behind the pair's changes nothing.
#[test]
fn a_decode_pool_keeps_the_record_its_models_pair_reads() {
    use crate::document::spec::{AffinitySource, PrefixAffinityParams, ScoreStage, ShadowSpec};
    // A profile that reads the record alone: its holders are named
    // ahead of the index's, which the preset's are not.
    let aware = PolicySpec::preset("cache_aware").expect("a preset");
    let aware = aware.profiles.into_values().next();
    let aware = aware.expect("the preset's profile");
    let mut by_record = aware.clone();
    let prefix = PrefixAffinityParams {
        weight: 4.0,
        source: Some(AffinitySource::Predictor),
        ..PrefixAffinityParams::default()
    };
    by_record.score[1] = ScoreStage::PrefixAffinity(prefix).into();
    // The pair of `round_robin` prefills and `decode` decodes, behind
    // a `round_robin` binding of the decode phase alone.
    let behind = |decode: &str| {
        let pair = PolicySpec::pair_presets("round_robin", decode);
        let mut document = pair.expect("the pair presets");
        let stock = PolicySpec::preset("round_robin").expect("a preset");
        let mut first = stock.bindings[0].clone();
        first.matcher.phase = vec![Phase::Decode];
        first.profile = Some("first".into());
        document.bindings.insert(0, first);
        let turns = stock.profiles.into_values().next();
        let turns = turns.expect("the preset's profile");
        document.profiles.insert("first".into(), turns);
        document
    };
    // The first binding's shadow word and the pool's record.
    let word = |document: &PolicySpec| {
        let engine = document.validate().expect("a valid document");
        let rows = engine.shadow_for("m", Phase::Decode);
        let record = engine.pool_record_for("m", Phase::Decode);
        (rows.map(|(_, rows)| rows), record)
    };

    // A decode request asked alone resolves to the first binding,
    // whose profile reads no record: the pool keeps the pair's.
    let pairs = behind("cache_aware");
    let engine = pairs.validate().expect("two bindings");
    assert!(!engine.reads_sent_for("m", Phase::Decode));
    assert_eq!(engine.pool_record_for("m", Phase::Prefill), None);
    assert_eq!(word(&pairs), (None, Some(false)));
    assert_eq!(word(&behind("round_robin")), (None, None));
    // The word is never the side's: a side that reads the record
    // alone states its own, and the first binding's decisions are no
    // reader of the record the pool keeps for it.
    let mut sides = behind("cache_aware");
    let decode = sides.profiles.get_mut("decode");
    decode.expect("the pair's decode profile").score[1] = by_record.score[1].clone();
    #[cfg(feature = "in-process")]
    {
        let engine = sides.validate().expect("two bindings");
        assert!(engine.side_reads_sent_alone("m", Phase::Decode));
    }
    assert_eq!(word(&sides), (None, Some(false)));

    // The first binding's own word stands where its profile reads
    // the record.
    let mut reader = behind("cache_aware");
    reader.profiles.insert("first".into(), by_record.clone());
    assert_eq!(word(&reader), (None, Some(true)));

    // The same binding listed behind the pair's takes no decode
    // request: the pool's word is the pair's, as under the pair alone.
    let mut after = behind("cache_aware");
    after.profiles.insert("first".into(), by_record.clone());
    after.bindings.swap(0, 1);
    let alone = PolicySpec::pair_presets("round_robin", "cache_aware");
    let alone = alone.expect("the pair presets");
    assert_eq!(word(&after), (None, Some(false)));
    assert_eq!(word(&alone), (None, Some(false)));

    // A shadow of the first binding has the pool keep the record for
    // it alone where the pair's decode side reads none. Beside a side
    // that reads the record it is not asked, and the pool's is the
    // side's.
    let shadowed = |decode: &str, shadow| {
        let mut document = behind(decode);
        document.profiles.insert("second".into(), shadow);
        document.bindings[0].shadow = Some(ShadowSpec::Profile("second".into()));
        document
    };
    let (own, apart) = (ShadowRows::Own, ShadowRows::Apart);
    let quiet = shadowed("round_robin", by_record.clone());
    assert_eq!(word(&quiet), (Some(own), Some(true)));
    let beside = shadowed("cache_aware", by_record);
    assert_eq!(word(&beside), (Some(apart), Some(false)));
    // A shadow that reads the record as the side does is not asked
    // either: one rule for every reading.
    let same = shadowed("cache_aware", aware);
    assert_eq!(word(&same), (Some(apart), Some(false)));
}

/// A pair is found by its prefill, whatever its own phase list says
/// of the decode phase: a binding built in code may name the prefill
/// alone. Its decode profile counts as a reader beside a binding of
/// the decode phase all the same, in either order. A pair whose
/// decode profile reads no record does not count.
#[test]
fn a_pairs_decode_side_is_a_reader_by_its_prefill_alone() {
    let named = |sent: bool| NamedBy {
        cap: u32::from(sent) * 16,
        rule: None,
        sent,
        sent_alone: false,
        held: Held::Unasked,
    };
    // Profile 0 reads no record and profile 1 reads it.
    let named_by = [named(false), named(true)];
    let alone = || Binding {
        phases: vec![Phase::Decode],
        ..single_binding()
    };
    let pair = |phases: &[Phase], decode: u16| Binding {
        phases: phases.to_vec(),
        decode_profile: Some(decode),
        ..pair_binding()
    };
    let beside = |bindings: [Binding; 2], at| live_reader_beside(&bindings, at, &named_by);
    let both: &[Phase] = &[Phase::Prefill, Phase::Decode];
    let (each, prefill): (&[Phase], &[Phase]) = (&[], &[Phase::Prefill]);
    // The decode binding first: the pair is left the prefill lookup.
    for phases in [both, each, prefill] {
        assert!(beside([alone(), pair(phases, 1)], 0), "{phases:?}");
        assert!(!beside([alone(), pair(phases, 0)], 0), "{phases:?}");
    }
    // The pair first: it takes the decode pool where its list covers
    // the phase, and the decode binding decides none of the model.
    assert!(!beside([pair(both, 1), alone()], 1));
    assert!(!beside([pair(each, 1), alone()], 1));
    assert!(beside([pair(prefill, 1), alone()], 1));
    assert!(!beside([pair(prefill, 0), alone()], 1));
}

/// Two bindings share a model where one id matches both patterns, as
/// [`Binding::matches`] reads each: by every id of one to four
/// letters.
#[test]
fn two_bindings_share_a_model_where_one_id_matches_both() {
    let binding = |model: &str| Binding {
        model: model.into(),
        phases: vec![],
        ..single_binding()
    };
    let ids: Vec<String> = (1..=4u32)
        .flat_map(|len| (0..1u32 << len).map(move |bits| (len, bits)))
        .map(|(len, bits)| {
            let letter = |at| if bits >> at & 1 == 0 { 'a' } else { 'b' };
            (0..len).map(letter).collect()
        })
        .collect();
    let patterns = [
        "*", "a*", "ab*", "b*", "*a", "*ba", "*b", "a", "ab", "ba", "b",
    ];
    for one in patterns {
        for other in patterns {
            let (one, other) = (binding(one), binding(other));
            let both =
                |id: &String| one.matches(id, Phase::Single) && other.matches(id, Phase::Single);
            assert_eq!(
                one.shares_a_model(&other),
                ids.iter().any(both),
                "{} and {}",
                one.model,
                other.model
            );
        }
    }
}

#[test]
fn shadow_needs_join_the_matching_bindings_live_needs() {
    let mut document = PolicySpec::preset("least_load").expect("a preset");
    let mut next = PolicySpec::preset("cache_aware").expect("a preset");
    let mut profile = next
        .profiles
        .remove("cache_aware")
        .expect("the affinity profile");
    profile.pick = spec::PickStage::Best(spec::BestParams { sample: 13 }).into();
    document.profiles.insert("second".into(), profile);
    document.bindings[0].matcher.phase = vec![Phase::Single];
    document.bindings[0].shadow = Some(spec::ShadowSpec::Profile("second".into()));
    let engine = document
        .validate()
        .expect("a binding with affinity in its shadow");
    assert!(!engine.profiles()[0].needs().affinity);
    assert!(engine.needs_for("m", Phase::Single).affinity);
    assert!(engine.needs_for_phase(Phase::Single).affinity);
    assert!(!engine.needs_for("m", Phase::Decode).affinity);
    assert!(!engine.needs_for_phase(Phase::Decode).affinity);
    assert!(
        matches!(engine.needs_rows().as_slice(), [("*", Phase::Single, needs)] if needs.affinity)
    );
    assert_eq!(engine.named_by[1].cap, 13);
    assert!(engine
        .shadow(
            &RequestFacts {
                phase: Phase::Decode,
                ..RequestFacts::default()
            },
            &PoolFacts::default(),
            &[],
            &Constraints::default(),
            &mut Scratch::new(),
        )
        .is_none());
}

/// Hands the last survivor alone.
#[derive(Debug)]
struct LastKept;

impl Lifecycle for LastKept {}

impl Picker for LastKept {
    fn name(&self) -> &'static str {
        "last-kept"
    }

    fn pick(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        if let Some(&row) = kept.last() {
            out.push(target_of(&candidates[row as usize]));
        }
    }
}

/// A pair's side is decided once more by its shadow profile, over the
/// rows its live side decided over. The prefill's shadow reads the
/// prefills that have a partner. The decode's shadow reads the
/// partners of the live pair's prefill: a decode that pairs with
/// another prefill alone is never its pick. A side the binding names
/// no shadow for is not asked, and a pair that names none has no
/// word for the door.
#[test]
fn a_pairs_side_is_decided_once_more_over_the_rows_its_live_side_saw() {
    let profile = |name: &str, picker: Box<dyn Picker>| Profile {
        name: name.into(),
        filters: Vec::new(),
        scorers: Vec::new(),
        picker,
    };
    let engine = |shadow: Option<u16>, decode_shadow: Option<u16>| {
        let binding = Binding {
            shadow,
            decode_shadow,
            ..pair_binding()
        };
        let profiles = vec![
            profile("live", Box::new(FirstKept)),
            profile("prefill-next", Box::new(LastKept)),
            profile("decode-next", Box::new(LastKept)),
        ];
        Engine::new(profiles, vec![binding], 1, Needs::default())
    };
    // The first prefill pairs with the first two decodes, the second
    // with the third alone, and the last with none.
    let of_class = |position, pair_class| Candidate {
        pair_class,
        ..candidate(position, 0)
    };
    let prefills = [of_class(0, 1), of_class(1, 2), of_class(2, 3)];
    let decodes = [of_class(0, 1), of_class(1, 1), of_class(2, 2)];
    let facts = RequestFacts {
        model: "m",
        ..RequestFacts::default()
    };
    let pools = PairPools {
        prefills: &prefills,
        decodes: &decodes,
        ..PairPools::default()
    };
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let both = engine(Some(1), Some(2));
    let live = both.decide_pair(&facts, &pool, pools, &cx, &cx, &mut scratch);
    let pick = live.pick().expect("a pair");
    let first = pick.targets.first().expect("the pair's first target");
    let decode = first.pair.map(|pair| pair.decode_position);
    assert_eq!((first.position, decode), (0, Some(0)));
    let word = both.shadow_for("m", Phase::Prefill);
    assert_eq!(word, Some((0, ShadowRows::Sides)));
    let mut side = |engine: &Engine, side| {
        let shadow =
            engine.shadow_side(0, side, &facts, &pool, pools, &cx, &cx, pick, &mut scratch);
        shadow.map(|shadow| lead(&shadow))
    };
    // The last prefill with a partner: never the one with none.
    assert_eq!(side(&both, PairSide::Prefill), Some(Some(1)));
    // The last partner of the live prefill: never the third decode.
    assert_eq!(side(&both, PairSide::Decode), Some(Some(1)));
    let decode_alone = engine(None, Some(2));
    assert_eq!(side(&decode_alone, PairSide::Prefill), None);
    assert_eq!(side(&decode_alone, PairSide::Decode), Some(Some(1)));
    assert!(decode_alone.shadows_side(0, PairSide::Decode));
    assert!(!decode_alone.shadows_side(0, PairSide::Prefill));
    // A decode decided after its prefill: its shadow reads the
    // partners of the prefill that is held, whichever the first
    // decision's shadow took.
    let call = RequestFacts {
        phase: Phase::Decode,
        ..facts
    };
    let mut later = |engine: &Engine, held: WorkerId| {
        let shadow =
            engine.shadow_decode_later(0, &call, &pool, pools, held, None, &cx, &mut scratch);
        shadow.map(|shadow| lead(&shadow))
    };
    assert_eq!(later(&both, prefills[0].id), Some(Some(1)));
    assert_eq!(later(&both, prefills[1].id), Some(Some(2)));
    assert_eq!(later(&both, WorkerId::default()), None, "the prefill left");
    let prefill_alone = engine(Some(1), None);
    assert_eq!(later(&prefill_alone, prefills[0].id), None, "not named");
    assert_eq!(engine(None, None).shadow_for("m", Phase::Prefill), None);
}

/// The prefill's shadow asks no stage of a live profile again. The
/// live decision's scope closes with it, and a stage asked behind it
/// answers by its fallback. The live decode profile's filter here
/// keeps a row only while its decision stands open: asked again, it
/// would refuse every partner, and the shadow would be shown no
/// prefill. The classes the live prefill was shown stand in the
/// scratch, and the shadow reads them.
#[test]
fn a_pair_sides_shadow_asks_no_stage_of_a_live_profile_again() {
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Keeps every row while its decision stands open, and none
    /// behind it.
    #[derive(Debug)]
    struct WhileOpen(Arc<AtomicBool>);

    impl Filter for WhileOpen {
        fn name(&self) -> &'static str {
            "while-open"
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            self.0.load(Ordering::Relaxed)
        }
    }

    let open = Arc::new(AtomicBool::new(true));
    let live = Profile {
        name: "live".into(),
        filters: vec![Box::new(WhileOpen(Arc::clone(&open)))],
        scorers: Vec::new(),
        picker: Box::new(FirstKept),
    };
    let next = |name: &str| Profile {
        name: name.into(),
        filters: Vec::new(),
        scorers: Vec::new(),
        picker: Box::new(LastKept),
    };
    let binding = Binding {
        shadow: Some(1),
        decode_shadow: Some(2),
        ..pair_binding()
    };
    let profiles = vec![live, next("prefill-next"), next("decode-next")];
    let engine = Engine::new(profiles, vec![binding], 1, Needs::default());
    // The first prefill pairs with the first two decodes, and the
    // second with the third alone.
    let of_class = |position, pair_class| Candidate {
        pair_class,
        ..candidate(position, 0)
    };
    let prefills = [of_class(0, 1), of_class(1, 2)];
    let decodes = [of_class(0, 1), of_class(1, 1), of_class(2, 2)];
    let facts = RequestFacts {
        model: "m",
        ..RequestFacts::default()
    };
    let pools = PairPools {
        prefills: &prefills,
        decodes: &decodes,
        ..PairPools::default()
    };
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let decided = engine.decide_pair(&facts, &pool, pools, &cx, &cx, &mut scratch);
    let pick = decided.pick().expect("a pair");
    // The live decision is over: its stages answer as behind a
    // closed scope.
    open.store(false, Ordering::Relaxed);
    let mut side = |side| {
        let shadow =
            engine.shadow_side(0, side, &facts, &pool, pools, &cx, &cx, pick, &mut scratch);
        shadow.map(|shadow| lead(&shadow))
    };
    assert_eq!(side(PairSide::Prefill), Some(Some(1)));
    assert_eq!(side(PairSide::Decode), Some(Some(1)));
}

/// The `cache_aware` preset's profile with its prefix stage alone
/// ahead of the load, reading cached blocks from the index: a profile
/// that reads the column by no warm-up ruling and no record.
fn by_index() -> spec::ProfileSpec {
    use crate::document::spec::{AffinitySource, PrefixAffinityParams, ScoreStage};
    let aware = PolicySpec::preset("cache_aware").expect("a preset");
    let profile = aware.profiles.into_values().next();
    let mut profile = profile.expect("the preset's profile");
    let prefix = PrefixAffinityParams {
        weight: 4.0,
        source: Some(AffinitySource::Index),
        ..PrefixAffinityParams::default()
    };
    profile.score = vec![
        ScoreStage::PrefixAffinity(prefix).into(),
        "least-load".into(),
    ];
    profile
}

/// The stock pair's document under `decode` as its decode preset,
/// with `shadow` as the decode side's shadow profile.
fn decode_shadowed_document(decode: &str, shadow: spec::ProfileSpec) -> PolicySpec {
    use crate::document::spec::ShadowSpec;
    let document = PolicySpec::pair_presets("least_load", decode);
    let mut document = document.expect("the stock presets");
    document.profiles.insert("decode-next".into(), shadow);
    let sides = [("decode".to_owned(), "decode-next".to_owned())];
    document.bindings[0].shadow = Some(ShadowSpec::Sides(sides.into()));
    document
}

/// The engine of that document.
fn decode_shadowed(decode: &str, shadow: spec::ProfileSpec) -> Engine {
    let document = decode_shadowed_document(decode, shadow);
    document.validate().expect("a pair with a decode shadow")
}

/// A side's shadow reads what its profile needs: its needs join the
/// side's, so the column it reads is filled for the pair's pool. The
/// other side's needs stay its own. Under a binding of the decode
/// phase listed ahead of the pair's, the decode pool still carries
/// what the pair's decode shadow reads.
#[test]
fn a_pair_sides_shadow_joins_its_needs_to_the_sides() {
    use crate::document::spec::{BindingSpec, Match};
    let alone = PolicySpec::pair_presets("least_load", "least_load");
    let alone = alone.expect("the stock presets").validate();
    let alone = alone.expect("the pair document");
    assert!(!alone.reads_for("m", Phase::Decode).0.affinity);
    let engine = decode_shadowed("least_load", by_index());
    assert!(
        !engine.side_apart(0, PairSide::Decode),
        "the shadow is asked"
    );
    assert!(engine.reads_for("m", Phase::Decode).0.affinity);
    assert!(engine.needs_for_phase(Phase::Decode).affinity);
    assert!(!engine.reads_for("m", Phase::Prefill).0.affinity);
    // A decode asked alone resolves to the binding ahead, which reads
    // no column: the pair's decode side brings its shadow's needs.
    let mut ahead = decode_shadowed_document("least_load", by_index());
    let lone = ahead.profiles.get("decode").cloned();
    let lone = lone.expect("the decode side's profile");
    ahead.profiles.insert("lone".into(), lone);
    let first = BindingSpec {
        matcher: Match {
            phase: vec![Phase::Decode],
            ..Match::default()
        },
        profile: Some("lone".into()),
        ..BindingSpec::default()
    };
    ahead.bindings.insert(0, first);
    let ahead = ahead.validate().expect("a decode binding ahead");
    assert!(ahead.reads_for("m", Phase::Decode).0.affinity);
}

/// A pair names no rows: a side reads the column its pool carries,
/// by the live profile's ruling and record. A side's shadow of
/// another ruling or another record is named and not asked. One that
/// reads the column as the pool carries it is asked, and so is one
/// that reads no column.
#[test]
fn a_pair_sides_shadow_of_another_column_is_not_asked() {
    let preset = |name: &str| {
        let preset = PolicySpec::preset(name).expect("a preset");
        let profile = preset.profiles.into_values().next();
        profile.expect("the preset's profile")
    };
    let apart = |decode: &str, shadow: spec::ProfileSpec| {
        let engine = decode_shadowed(decode, shadow);
        assert!(engine.shadows_side(0, PairSide::Decode), "named");
        assert!(!engine.side_apart(0, PairSide::Prefill), "not named");
        engine.side_apart(0, PairSide::Decode)
    };
    // Behind a side that reads no cached blocks: its pool has no
    // ruling and keeps no record.
    assert!(apart("least_load", preset("cache_aware")));
    assert!(!apart("least_load", by_index()));
    assert!(!apart("least_load", preset("least_load")));
    // Behind the preset that rules its warming rows and reads the
    // record.
    assert!(!apart("cache_aware", preset("cache_aware")));
    assert!(apart("cache_aware", by_index()));
    assert!(!apart("cache_aware", preset("least_load")));
}
