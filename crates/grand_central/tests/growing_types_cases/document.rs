//! The records a policy document states, and what a snapshot caller asks
//! of the worker rows, as a crate outside this one comes to them. Ten have
//! no start: each of their cases reads one from a document's text, a road
//! to it, and reads its fields back by a pattern that ends open. Eight
//! start from their default and take a field by assignment. Two are made
//! by a call that takes the one field a document must write, and each call
//! is held to a document that writes that field and nothing else.

#![expect(clippy::expect_used, reason = "cases that stop on a broken setup")]

use std::collections::BTreeMap;

#[cfg(feature = "operator")]
use smg_grand_central::document::operator::snapshot::WorkerRowsQuery;
use smg_grand_central::document::{
    BestParams, BestWithTemperatureParams, BindingSpec, Bounds, BucketParams, CanarySpec, Capacity,
    CostMargin, FilterSpec, FilterStage, HasRoomParams, HeaderMatch, IndexSpec, KeySourceSpec,
    LeastCostParams, Match, OverloadSpec, PickSpec, PickStage, PolicySpec, ProfileSpec,
    ResidentWorkParams, RoutingKeySpec, ScoreSpec, ScoreStage, SessionsSpec, SignalSpec,
    API_VERSION,
};

/// The document that `rest` states under this build's version, read from
/// its YAML text.
fn read(rest: &str) -> PolicySpec {
    serde_yaml::from_str(&format!("apiVersion: {API_VERSION}\n{rest}"))
        .expect("a document's text reads")
}

/// The profile `p` of a document that states it as `profile`.
fn profiled(profile: &str) -> ProfileSpec {
    let mut spec = read(&format!("profiles: {{p: {profile}}}"));
    spec.profiles.remove("p").expect("the profile `p`")
}

/// The one binding of a document that states it as `binding`, beside a
/// profile `p`.
fn bound(binding: &str) -> BindingSpec {
    let mut spec = read(&format!(
        "profiles: {{p: {{pick: random}}}}\nbindings: [{binding}]"
    ));
    spec.bindings.pop().expect("the one binding")
}

/// A resident-work scorer's parameters are read from a document's text,
/// and by a pattern that ends open.
#[test]
fn resident_work_parameters_are_read_from_a_documents_text() {
    let profile = profiled("{score: [{resident-work: {request_tokens: 512}}], pick: best}");
    let [ScoreSpec::Params(ScoreStage::ResidentWork(ResidentWorkParams {
        weight,
        request_tokens,
        ..
    }))] = profile.score.as_slice()
    else {
        panic!("one scorer with its parameters");
    };
    assert_eq!(
        (weight.to_bits(), *request_tokens),
        (1.0_f32.to_bits(), 512)
    );
}

/// A least-cost scorer's parameters are read from a document's text, and
/// by a pattern that ends open.
#[test]
fn least_cost_parameters_are_read_from_a_documents_text() {
    let profile = profiled("{score: [{least-cost: {terms: {load.running: 7}}}], pick: best}");
    let [ScoreSpec::Params(ScoreStage::LeastCost(LeastCostParams {
        weight,
        terms,
        margin,
        unreported,
        ..
    }))] = profile.score.as_slice()
    else {
        panic!("one scorer with its parameters");
    };
    assert_eq!(
        (terms, margin, unreported),
        (
            &BTreeMap::from([("load.running".to_owned(), 7.0)]),
            &CostMargin::default(),
            &None
        )
    );
    assert_eq!(weight.to_bits(), 1.0_f32.to_bits());
}

/// The `best` picker's parameters are read from a document's text, and by
/// a pattern that ends open.
#[test]
fn the_best_pickers_parameters_are_read_from_a_documents_text() {
    let ProfileSpec {
        pick: PickSpec::Params(PickStage::Best(BestParams { sample, .. })),
        ..
    } = profiled("{pick: {best: {sample: 2}}}")
    else {
        panic!("a picker with its parameters");
    };
    assert_eq!(sample, 2);
}

/// The parameters of the picker that draws by temperature are read from a
/// document's text, and by a pattern that ends open.
#[test]
fn the_temperature_pickers_parameters_are_read_from_a_documents_text() {
    let ProfileSpec {
        pick:
            PickSpec::Params(PickStage::BestWithTemperature(BestWithTemperatureParams {
                temperature,
                ..
            })),
        ..
    } = profiled("{pick: {best-with-temperature: {temperature: 0.05}}}")
    else {
        panic!("a picker with its parameters");
    };
    assert_eq!(temperature.to_bits(), 0.05_f32.to_bits());
}

/// A `has-room` rule is read from a document's text, and by a pattern that
/// ends open.
#[test]
fn a_has_room_rule_is_read_from_a_documents_text() {
    let profile =
        profiled("{filter: [{has-room: {used: {load.running: 1}, capacity: 10}}], pick: random}");
    let [FilterSpec::Params(FilterStage::HasRoom(HasRoomParams {
        used,
        capacity,
        headroom,
        ..
    }))] = profile.filter.as_slice()
    else {
        panic!("one filter with its rule");
    };
    assert_eq!(
        (used, capacity, headroom.to_bits()),
        (
            &BTreeMap::from([("load.running".to_owned(), 1.0)]),
            &Capacity::Fixed(10.0),
            0.0_f64.to_bits()
        )
    );
}

/// A matcher's bounds are read from a document's text, and by a pattern
/// that ends open.
#[test]
fn a_matchers_bounds_are_read_from_a_documents_text() {
    let binding = bound("{match: {prompt_tokens: {min: 1, max: 4096}}, profile: p}");
    let Some(Bounds { min, max, .. }) = binding.matcher.prompt_tokens else {
        panic!("the bounds of a prompt's length");
    };
    assert_eq!((min, max), (Some(1), Some(4_096)));
}

/// A matcher's header is read from a document's text, and by a pattern
/// that ends open.
#[test]
fn a_matchers_header_is_read_from_a_documents_text() {
    let binding = bound("{match: {header: {name: x-tier, values: [gold]}}, profile: p}");
    let Some(HeaderMatch { name, values, .. }) = binding.matcher.header else {
        panic!("the header a request is matched by");
    };
    assert_eq!((name.as_str(), values), ("x-tier", vec!["gold".to_owned()]));
}

/// A binding's sessions block is read from a document's text, and by a
/// pattern that ends open.
#[test]
fn a_sessions_block_is_read_from_a_documents_text() {
    let binding = bound("{profile: p, sessions: {sticky_header: x-session, ttl_s: 600}}");
    let Some(SessionsSpec {
        sticky_header,
        headers,
        ttl_s,
        issue_session_id,
        ..
    }) = binding.sessions
    else {
        panic!("the binding's sessions");
    };
    assert_eq!(
        (sticky_header.as_str(), ttl_s, issue_session_id),
        ("x-session", 600, false)
    );
    assert!(headers.is_empty());
}

/// A binding's canary is read from a document's text, and by a pattern
/// that ends open.
#[test]
fn a_canary_is_read_from_a_documents_text() {
    let binding = bound("{profile: p, canary: {profile: q, percent: 5}}");
    let Some(CanarySpec {
        profile,
        percent,
        tenants,
        header,
        ..
    }) = binding.canary
    else {
        panic!("the binding's canary");
    };
    assert_eq!((profile.as_str(), percent, header), ("q", Some(5), None));
    assert!(tenants.is_empty());
}

/// A signal's declaration is read from a document's text, and by a pattern
/// that ends open.
#[test]
fn a_signals_declaration_is_read_from_a_documents_text() {
    let mut spec =
        read("signals: {pending: {unit: tokens, max_age_ms: 4000}}\nprofiles: {p: {pick: random}}");
    let Some(SignalSpec {
        unit, max_age_ms, ..
    }) = spec.signals.remove("pending")
    else {
        panic!("the signal the document declares");
    };
    assert_eq!((unit.as_str(), max_age_ms), ("tokens", Some(4_000)));
}

/// An index entry starts from its default and takes a field by assignment.
/// It is read by a pattern that ends open.
#[test]
fn an_index_entry_starts_from_its_default_and_takes_a_field_by_assignment() {
    let stated = IndexSpec::default();
    let mut index = IndexSpec::default();
    index.max_size = 4_096;
    let IndexSpec {
        kind,
        ttl_s,
        max_size,
        sweep_s,
        ..
    } = index;
    assert_eq!(
        (kind, ttl_s, max_size, sweep_s),
        (stated.kind, stated.ttl_s, 4_096, stated.sweep_s)
    );
}

/// A bucket scorer's parameters start from their default and take a field
/// by assignment. They are read by a pattern that ends open.
#[test]
fn bucket_parameters_start_from_their_default_and_take_a_field_by_assignment() {
    let mut sizes = BucketParams::default();
    sizes.abs_threshold = Some(64);
    let BucketParams {
        weight,
        abs_threshold,
        rel_threshold,
        adjust_interval_s,
        ..
    } = sizes;
    assert_eq!(
        (weight.to_bits(), abs_threshold, adjust_interval_s),
        (1.0_f32.to_bits(), Some(64), None)
    );
    assert!(rel_threshold.is_none());
}

/// A cost margin starts from its default, no margin, and takes a field by
/// assignment. It is read by a pattern that ends open.
#[test]
fn a_cost_margin_starts_from_its_default_and_takes_a_field_by_assignment() {
    let mut margin = CostMargin::default();
    margin.absolute = 12.0;
    let CostMargin {
        absolute, relative, ..
    } = margin;
    assert_eq!(
        (absolute.to_bits(), relative.to_bits()),
        (12.0_f64.to_bits(), 0.0_f64.to_bits())
    );
}

/// A matcher starts from its default, which matches every model, and takes
/// a field by assignment. It is read by a pattern that ends open.
#[test]
fn a_matcher_starts_from_its_default_and_takes_a_field_by_assignment() {
    let mut matcher = Match::default();
    assert_eq!(matcher.model, "*");
    matcher.model = "a model".to_owned();
    let Match {
        model,
        phase,
        tenant,
        labels,
        ..
    } = matcher;
    assert_eq!((model.as_str(), tenant), ("a model", None));
    assert!(phase.is_empty() && labels.is_empty());
}

/// An overload block starts from its default, which is on and sheds
/// nothing, and takes a field by assignment. It is read by a pattern that
/// ends open.
#[test]
fn an_overload_block_starts_from_its_default_and_takes_a_field_by_assignment() {
    let mut overload = OverloadSpec::default();
    overload.shed_when_all_vetoed = true;
    let OverloadSpec {
        enabled,
        waiting_requests,
        token_usage,
        shed_when_all_vetoed,
        ..
    } = overload;
    assert_eq!(
        (enabled, waiting_requests, shed_when_all_vetoed),
        (true, None, true)
    );
    assert!(token_usage.is_none());
}

/// A routing key's source starts from its default, which names nothing,
/// and takes a field by assignment. It is read by a pattern that ends open.
#[test]
fn a_key_source_starts_from_its_default_and_takes_a_field_by_assignment() {
    let mut source = KeySourceSpec::default();
    source.rid = true;
    let KeySourceSpec {
        header,
        rid,
        max_bytes,
        client,
        target,
        ..
    } = source;
    assert_eq!(
        (header, rid, max_bytes, client, target),
        (None, true, None, false, false)
    );
}

/// A binding starts from its default, which matches every model and names
/// no profile, and takes a field by assignment. It is read by a pattern
/// that ends open.
#[test]
fn a_binding_starts_from_its_default_and_takes_a_field_by_assignment() {
    let mut binding = BindingSpec::default();
    binding.profile = Some("p".to_owned());
    binding.matcher.model = "a model".to_owned();
    let BindingSpec {
        matcher,
        profile,
        plan,
        routing_key,
        sessions,
        ..
    } = binding;
    assert_eq!(
        (
            matcher.model.as_str(),
            profile.as_deref(),
            routing_key,
            sessions
        ),
        ("a model", Some("p"), None, None)
    );
    assert!(plan.is_none());
}

/// What a snapshot caller asks of the worker rows starts from its default:
/// every model's rows, under the byte bound alone. It takes a field by
/// assignment and is read by a pattern that ends open.
#[cfg(feature = "operator")]
#[test]
fn a_rows_query_starts_from_its_default_and_takes_a_field_by_assignment() {
    let WorkerRowsQuery { models, limit, .. } = WorkerRowsQuery::default();
    assert!(models.is_empty());
    assert_eq!(limit, usize::MAX);
    let mut asked = WorkerRowsQuery::default();
    asked.models = vec!["a model".to_owned()];
    asked.limit = 2;
    let WorkerRowsQuery { models, limit, .. } = asked;
    assert_eq!((models, limit), (vec!["a model".to_owned()], 2));
}

/// A profile is made by its call from the picker alone, the one field a
/// document must write. It equals the profile a document states by its
/// picker and nothing else, and it is read by a pattern that ends open.
/// The call builds in a constant where its argument does.
#[test]
fn a_profile_spec_is_made_by_its_call_as_a_document_of_its_picker_alone_reads() {
    const NAMED: ProfileSpec = ProfileSpec::of(PickSpec::Name(String::new()));

    assert_eq!(NAMED.pick, PickSpec::Name(String::new()));
    let made = ProfileSpec::of(PickSpec::from("best"));
    assert_eq!(made, profiled("{pick: best}"));
    let ProfileSpec {
        filter,
        score,
        pick,
        sent_ttl_ms,
        sent_report_delay_ms,
        ..
    } = made;
    assert!(filter.is_empty() && score.is_empty());
    assert_eq!(
        (pick, sent_ttl_ms, sent_report_delay_ms),
        (PickSpec::from("best"), None, None)
    );
}

/// A routing key is made by its call from its sources alone, the one field
/// a document must write. It equals the block a document states by its
/// sources and nothing else, and it is read by a pattern that ends open.
/// The call builds in a constant where its argument does.
#[test]
fn a_routing_key_is_made_by_its_call_as_a_document_of_its_sources_alone_reads() {
    const NO_SOURCE: RoutingKeySpec = RoutingKeySpec::of(Vec::new());

    assert!(NO_SOURCE.sources.is_empty());
    let mut source = KeySourceSpec::default();
    source.rid = true;
    let made = RoutingKeySpec::of(vec![source.clone()]);
    let read = bound("{profile: p, routing_key: {sources: [{rid: true}]}}").routing_key;
    assert_eq!(read.as_ref(), Some(&made));
    let RoutingKeySpec {
        sources,
        strip_lineage,
        ..
    } = made;
    assert_eq!((sources, strip_lineage), (vec![source], false));
}
