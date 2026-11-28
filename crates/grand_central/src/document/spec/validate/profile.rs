//! Validation of one profile's stages: the checks of a scorer stage's
//! parameters ([`check_stage`]), the stages a scorer, a filter and a picker
//! entry resolve to, and the profile [`check_profile`] builds of them. Of
//! `least-cost`, [`check_stage`] checks the weight above zero alone, and
//! [`least_cost_of`] makes the stage's other checks where it resolves, as
//! [`has_room_of`] does for `has-room` and [`picker_of`] for a picker.

use std::sync::Arc;

use super::{
    above_zero, at_least_one, at_or_above_zero, named_term, unit_interval, At, Deferred, Named,
    StageInputs,
};
use crate::{
    document::spec::{
        Capacity, ConsistentHashParams, FilterSpec, FilterStage, HasRoomParams, HashDepth,
        HashDepths, LeastCostParams, Needs, PickSpec, PickStage, ProfileSpec, ScoreSpec,
        ScoreStage, SpecError, Unreported, ViewSpec, COST_PARAMETER_LIMIT, MAX_SAMPLE,
        RELATIVE_MARGIN_LIMIT, SAMPLING_PICKERS, TYPED_PICKERS,
    },
    engine::framework::{Filter, Picker, Profile, Scorer, WeightedScorer},
    load::signals::{column_named, SignalColumn},
    policies::{
        bucket::Bucket,
        cache_aware::{Gates, KvPressure, PrefixAffinity, WarmupSlice},
        consistent_hashing::ConsistentHash,
        least_load::LeastLoad,
    },
    stages::{
        cost_terms::signal_named,
        filters::{has_room_rule, HasRoomRule, HasRoomSignals},
        pickers::SampledBest,
        registry::{
            profile_sent, FilterFactory, Found, PickerFactory, SentSlot, StageBuild, StageKind,
        },
        scorers::{
            with_load_report_age_limit, FewestSessions, LeastCost, LeastCostSignals,
            LeastCostWithDispatches, LeastRequests, ResidentWork, SessionAffinity,
            DEFAULT_KV_PRESSURE_WEIGHT, DEFAULT_MEAN_PREFILL_TOKENS, DEFAULT_THROUGHPUT,
        },
    },
    tracking::dispatch_credit::{DispatchCredit, SentSettings},
};

/// A stage's parameters, refused by name: the weight is finite and above
/// zero; each parameter is in its range whether or not this build carries
/// the stage, so a document is well-formed before it is current.
fn check_stage(stage: &ScoreStage, at: &At) -> Result<(), SpecError> {
    if !above_zero(stage.weight()) {
        return Err(at.bad("weight", stage.weight(), "a finite number above zero"));
    }
    match stage {
        ScoreStage::LeastLoad(p) => {
            at.number(
                "kv_pressure_weight",
                p.kv_pressure_weight,
                at_or_above_zero,
                "a finite number at or above zero",
            )?;
            at.nonzero("mean_prefill_tokens", p.mean_prefill_tokens)?;
            at.number(
                "default_throughput",
                p.default_throughput,
                above_zero,
                "a finite number above zero",
            )
        }
        // The resident-work price has no bound: zero compares the tokens
        // alone.
        ScoreStage::LeastRequests(_)
        | ScoreStage::FewestSessions(_)
        | ScoreStage::SessionAffinity(_)
        | ScoreStage::ResidentWork(_) => Ok(()),
        ScoreStage::LeastCost(_) => Ok(()),
        ScoreStage::PrefixAffinity(p) => {
            at.number("threshold", p.threshold, unit_interval, "between 0 and 1")?;
            at.number(
                "decay",
                p.decay,
                at_or_above_zero,
                "a finite number at or above zero",
            )?;
            match &p.spill {
                Some(spill) => at.sub("spill").number(
                    "rel",
                    spill.rel,
                    at_least_one,
                    "a finite number at least 1",
                ),
                None => Ok(()),
            }
        }
        ScoreStage::KvPressure(p) => {
            at.number("spread", p.spread, above_zero, "a finite number above zero")?;
            at.number(
                "ceiling",
                p.ceiling,
                above_zero,
                "a finite number above zero",
            )
        }
        ScoreStage::WarmupSlice(p) => {
            at.number("share", p.share, unit_interval, "between 0 and 1")?;
            at.number("thin_ratio", p.thin_ratio, unit_interval, "between 0 and 1")?;
            at.number("miss_share", p.miss_share, unit_interval, "between 0 and 1")
        }
        ScoreStage::Bucket(p) => {
            at.number(
                "rel_threshold",
                p.rel_threshold,
                at_least_one,
                "a finite number at least 1",
            )?;
            at.nonzero("adjust_interval_s", p.adjust_interval_s)
        }
        ScoreStage::ConsistentHash(p) => {
            at.nonzero("prefix_tokens", p.prefix_tokens)?;
            match &p.bounded_load {
                Some(bounded) => at.sub("bounded_load").number(
                    "factor",
                    bounded.factor,
                    at_least_one,
                    "a finite number at least 1",
                ),
                None => Ok(()),
            }
        }
    }
}

/// Check the cost parameters while resolving the field names into readers
/// and the signals' names into their columns.
fn least_cost_of(
    p: &LeastCostParams,
    signals: &[SignalColumn],
    sent: impl FnOnce() -> Arc<DispatchCredit>,
    max_age_ms: Option<u64>,
    at: &At,
) -> Result<Box<dyn Scorer>, SpecError> {
    const COST_RANGE: &str = "a finite number from 0 to 1e15";
    if f64::from(p.weight) > COST_PARAMETER_LIMIT {
        return Err(at.bad(
            "weight",
            p.weight,
            "a finite number above zero and at most 1e15",
        ));
    }
    if p.terms.is_empty() {
        return Err(at.bad("terms", "{}", "at least one field with its coefficient"));
    }
    let mut terms = Vec::new();
    let mut signal_terms = Vec::new();
    for (name, &coefficient) in &p.terms {
        let term = named_term(signals, at, "terms", name)?;
        if coefficient == 0.0
            || !(-COST_PARAMETER_LIMIT..=COST_PARAMETER_LIMIT).contains(&coefficient)
        {
            return Err(at.sub("terms").bad(
                name,
                coefficient,
                "a finite number other than zero, of magnitude at most 1e15",
            ));
        }
        match term {
            Named::Field(term) => terms.push((term, coefficient)),
            Named::Signal(signal) => signal_terms.push((signal, coefficient)),
        }
    }
    if !(0.0..=COST_PARAMETER_LIMIT).contains(&p.margin.absolute) {
        return Err(at
            .sub("margin")
            .bad("absolute", p.margin.absolute, COST_RANGE));
    }
    if !(0.0..=RELATIVE_MARGIN_LIMIT).contains(&p.margin.relative) {
        return Err(at.sub("margin").bad(
            "relative",
            p.margin.relative,
            "a finite number from 0 to 1e6",
        ));
    }
    if let Some(Unreported::OverLeast { request_cost }) = p.unreported {
        if !(0.0..=COST_PARAMETER_LIMIT).contains(&request_cost) {
            return Err(at.sub("unreported.over-least").bad(
                "request_cost",
                request_cost,
                COST_RANGE,
            ));
        }
    }
    let reads_sent = terms.iter().any(|&(term, _)| term.reads_sent());
    let scorer = LeastCost::new(terms.into_boxed_slice(), p.margin, p.unreported);
    // A row with no fresh value for a named signal is unreported for the
    // scorer, as a row with no report is: the document says what such a
    // row costs.
    if (scorer.needs().loads || !signal_terms.is_empty()) && p.unreported.is_none() {
        return Err(at.bad(
            "unreported", "null",
            "what a row with no report, or with no fresh value of a signal, costs, which a term that reads a report or a signal needs: over-least or missing-as-zero",
        ));
    }
    // Only a term that reads sends makes the profile's count, and only a
    // term that reads a signal the scorer that adds one.
    Ok(if !signal_terms.is_empty() {
        let scorer = LeastCostSignals::new(
            scorer,
            reads_sent.then(sent),
            signal_terms.into_boxed_slice(),
        );
        with_load_report_age_limit(scorer, max_age_ms)
    } else if reads_sent {
        with_load_report_age_limit(LeastCostWithDispatches::new(scorer, sent()), max_age_ms)
    } else {
        with_load_report_age_limit(scorer, max_age_ms)
    })
}

/// A stage of this build, or the name of what the build lacks.
enum Resolved<T> {
    Built(T),
    NotInBuild(&'static str),
}

/// What a prefix depth breaks against the view that carries a request's
/// block hashes: `under` when it holds no whole block of the view's block
/// size, `over` when a request carries no block that deep.
fn depth_edge(
    depth: u32,
    gates: Gates,
    view: &ViewSpec,
    under: &'static str,
    over: &'static str,
) -> Option<&'static str> {
    match depth / gates.block_size {
        0 => Some(under),
        blocks if blocks > view.max_block_hashes => Some(over),
        _ => None,
    }
}

/// A consistent-hash stage's prefix depths against the view. A key ends
/// with the last whole block at or under its depth, so `prefix_tokens` and
/// each of the view's boundaries hold at least one block and no more blocks
/// than a request carries.
fn check_prefix_depth(
    stage: &ScoreStage,
    gates: Gates,
    view: &ViewSpec,
    at: &At,
) -> Result<(), SpecError> {
    let ScoreStage::ConsistentHash(ConsistentHashParams {
        prefix_tokens: Some(tokens),
        ..
    }) = stage
    else {
        return Ok(());
    };
    if let Some(reason) = depth_edge(
        *tokens,
        gates,
        view,
        "at least one block of view.block_size",
        "at most view.max_block_hashes blocks of view.block_size",
    ) {
        return Err(at.bad("prefix_tokens", tokens, reason));
    }
    for (i, &boundary) in view.boundaries.iter().enumerate() {
        if let Some(reason) = depth_edge(
            boundary,
            gates,
            view,
            "at least one block of view.block_size under a consistent-hash prefix key",
            "at most view.max_block_hashes blocks of view.block_size under a consistent-hash prefix key",
        ) {
            return Err(SpecError::BadField {
                path: format!("view.boundaries[{i}]"),
                value: boundary.to_string(),
                reason,
            });
        }
    }
    Ok(())
}

fn built_scorer(scorer: impl Scorer + 'static) -> Resolved<Box<dyn Scorer>> {
    Resolved::Built(Box::new(scorer))
}

/// The gates a profile's affinity stages share: the spill gate of its
/// `prefix-affinity` entry, the fleet gate of its `kv-pressure` entry and
/// the view's block size. Of two entries of a kind that write parameters
/// the last one's stand here, and the defaults where none does; a
/// `prefix-affinity` scorer spills by its own entry's gate. A
/// `warmup-slice` follows this spill gate by its `rel` and `abs`, so beside
/// one `check_profile` refuses `prefix-affinity` entries that write two.
fn gates_of(p: &ProfileSpec, view: &ViewSpec) -> Gates {
    let mut affinity = None;
    let mut kv = None;
    for entry in &p.score {
        match entry {
            ScoreSpec::Params(ScoreStage::PrefixAffinity(params)) => affinity = Some(params),
            ScoreSpec::Params(ScoreStage::KvPressure(params)) => kv = Some(params),
            _ => {}
        }
    }
    Gates::of(affinity, kv, view.block_size)
}

/// The stage's scorer, with its profile's gates and the view's settings.
/// `sent` hands out the profile's count of sends.
fn scorer_of(
    stage: &ScoreStage,
    gates: Gates,
    view: &ViewSpec,
    signals: &[SignalColumn],
    sent: impl FnOnce() -> Arc<DispatchCredit>,
    at: &At,
) -> Result<Resolved<Box<dyn Scorer>>, SpecError> {
    let max_age = view.max_report_age_ms;
    Ok(match stage {
        ScoreStage::LeastLoad(p) => Resolved::Built(with_load_report_age_limit(
            LeastLoad::of_profile(
                p.kv_pressure_weight.unwrap_or(DEFAULT_KV_PRESSURE_WEIGHT),
                p.mean_prefill_tokens.unwrap_or(DEFAULT_MEAN_PREFILL_TOKENS),
                p.default_throughput.unwrap_or(DEFAULT_THROUGHPUT),
                sent(),
            ),
            max_age,
        )),
        ScoreStage::LeastRequests(_) => built_scorer(LeastRequests),
        ScoreStage::FewestSessions(_) => built_scorer(FewestSessions),
        ScoreStage::SessionAffinity(_) => built_scorer(SessionAffinity),
        ScoreStage::PrefixAffinity(p) if p.decay.is_some_and(|decay| decay > 0.0) => {
            Resolved::NotInBuild("the prefix-affinity decay")
        }
        ScoreStage::PrefixAffinity(p) if p.accounting_ttl_ms.is_some_and(|ttl| ttl > 0) => {
            Resolved::NotInBuild("the prefix-affinity accounting")
        }
        ScoreStage::PrefixAffinity(p)
            if p.spill
                .as_ref()
                .is_some_and(|spill| spill.uncached_credit == Some(true)) =>
        {
            Resolved::NotInBuild("the prefix-affinity uncached credit")
        }
        ScoreStage::PrefixAffinity(p) => {
            // The entry spills by the gate it writes itself. The KV gate
            // and the block size stay the profile's.
            let own = Gates::of(Some(p), None, None);
            let gates = Gates {
                rel: own.rel,
                abs: own.abs,
                prefix_margin_tokens: own.prefix_margin_tokens,
                ..gates
            };
            Resolved::Built(with_load_report_age_limit(
                PrefixAffinity::new(p, gates),
                max_age,
            ))
        }
        ScoreStage::KvPressure(_) => built_scorer(KvPressure),
        ScoreStage::WarmupSlice(p) => Resolved::Built(with_load_report_age_limit(
            WarmupSlice::new(p, gates),
            max_age,
        )),
        ScoreStage::Bucket(p) => built_scorer(Bucket::new(p)),
        ScoreStage::ConsistentHash(p) => {
            let ring = ConsistentHash::new(p.bounded_load.as_ref());
            built_scorer(match p.prefix_tokens {
                Some(tokens) => ring.keyed_by_prefix(tokens, gates, &view.boundaries),
                None => ring,
            })
        }
        ScoreStage::ResidentWork(p) => Resolved::Built(with_load_report_age_limit(
            ResidentWork::new(p.request_tokens),
            max_age,
        )),
        ScoreStage::LeastCost(p) => Resolved::Built(least_cost_of(p, signals, sent, max_age, at)?),
    })
}

/// The profile's picker. The catalog is asked for a name first, as it is
/// for a filter: a picker it builds from the document's map is handed what
/// stands under the name, a picker registered with no parameter is written
/// by its name alone, and a name it does not hold is refused as unknown
/// whatever is written under it. `sent` hands a factory that asks for it
/// the profile's count of sends.
fn picker_of(
    pick: &PickSpec,
    profile: &str,
    at: &At,
    from: StageInputs<'_>,
    settings: SentSettings,
    sent: &mut SentSlot,
) -> Result<Resolved<Box<dyn Picker>>, SpecError> {
    let named = pick.name();
    let path = || format!("{}.pick.{named}", at.0);
    // The catalog's factory builds its picker from the document's map. A
    // picker has no weight: the map is the factory's whole.
    let made = |build: &PickerFactory,
                mut params: serde_json::Map<String, serde_json::Value>,
                sent: &mut SentSlot| {
        in_key_order(&mut params);
        let picker = StageBuild::built(
            &params,
            settings,
            sent,
            from.reports,
            from.signals,
            build,
            Picker::needs,
        )
        .map_err(|error| error.at(&path()))?;
        let answers = picker.name();
        if answers != named {
            return Err(misnamed(path(), StageKind::Picker, named, answers));
        }
        Ok(Resolved::Built(picker))
    };
    let unknown = || SpecError::UnknownPicker {
        profile: profile.to_string(),
        name: named.to_string(),
    };
    match pick {
        PickSpec::Name(name) => match from.catalog.find_picker(name) {
            Some(Found::Kept) => Err(kept_name(path(), StageKind::Picker, name)),
            Some(Found::Misnamed(answers)) => {
                Err(misnamed(path(), StageKind::Picker, name, answers))
            }
            Some(Found::Built(picker)) => {
                // A picker the catalog built with no context.
                no_cells_without_a_context(
                    picker.needs(),
                    "register it by picker_with to resolve one, or leave the need unset",
                    path,
                )?;
                Ok(Resolved::Built(picker))
            }
            Some(Found::With(build)) => made(build, serde_json::Map::new(), sent),
            // The one picker with parameters, named without them.
            Some(Found::Typed) => Err(at.bad(
                "pick",
                name,
                "a picker with its temperature: { best-with-temperature: { temperature } }",
            )),
            None => Err(unknown()),
        },
        PickSpec::Params(PickStage::Best(p)) => {
            if !(1..=MAX_SAMPLE).contains(&p.sample) {
                return Err(at.sub("pick").sub("best").bad(
                    "sample",
                    p.sample,
                    "a whole number from 1 to 64",
                ));
            }
            Ok(Resolved::Built(Box::new(SampledBest::new(p.sample))))
        }
        PickSpec::Params(PickStage::BestWithTemperature(p)) => {
            if !at_or_above_zero(p.temperature) {
                return Err(at.sub("pick").sub("best-with-temperature").bad(
                    "temperature",
                    p.temperature,
                    "a finite number at or above zero",
                ));
            }
            Ok(Resolved::NotInBuild("the best-with-temperature picker"))
        }
        PickSpec::Added { name, written } => match from.catalog.find_picker(name) {
            // Only a document built in code: the reader reads a typed
            // built-in's key into its parameter struct.
            _ if TYPED_PICKERS.contains(&name.as_str()) => Err(SpecError::BadField {
                path: path(),
                value: name.clone(),
                reason: "a picker added in code: a built-in's parameters are typed",
            }),
            Some(Found::Kept) => Err(kept_name(path(), StageKind::Picker, name)),
            Some(Found::Misnamed(answers)) => {
                Err(misnamed(path(), StageKind::Picker, name, answers))
            }
            Some(Found::With(build)) => {
                let params = written.as_object().ok_or_else(|| SpecError::BadField {
                    path: path(),
                    value: written.to_string(),
                    reason: "a map of the picker's parameters",
                })?;
                made(build, params.clone(), sent)
            }
            // A picker registered with no parameter is written by its name
            // alone: whatever stands under it is refused.
            Some(Found::Built(_)) => Err(SpecError::BadField {
                path: path(),
                value: written.to_string(),
                reason: "left out: this picker takes no parameter, write its bare name",
            }),
            // The catalog holds no picker by its type alone under a name the
            // reader kept as read.
            None | Some(Found::Typed) => Err(unknown()),
        },
    }
}

/// The refusal of a document that names a host's stage under one of the
/// names the engine refuses under
/// ([`KEPT`](crate::engine::decision::KEPT)): a shed under such a name
/// reads as the engine's word, so no document runs a stage registered
/// under one. The registry takes such a stage, and the catalog's lookup
/// says so where a document names it ([`Found::Kept`]), ahead of any
/// build. `path` is the stage's own, and names the profile.
fn kept_name(path: String, kind: StageKind, name: &str) -> SpecError {
    SpecError::BadStage {
        path,
        reason: format!(
            "{kind} {name:?} bears a name the engine refuses requests under: register the stage under another name"
        ),
    }
}

/// The refusal of a document that names a stage whose build answers
/// another name than the one it is registered under: a factory that
/// reads the document's map, and a plain factory, whose stage the
/// catalog's lookup dropped ([`Found::Misnamed`]). So the name a document
/// writes is the name its stage answers at that build. `path` is the
/// stage's own, and names the profile.
fn misnamed(path: String, kind: StageKind, named: &str, answers: &str) -> SpecError {
    SpecError::BadStage {
        path,
        reason: format!("{kind} registered as {named:?} answers the name {answers:?}"),
    }
}

/// Holds a stage the catalog built with no context to the signal cells,
/// as [`StageBuild::built`] holds one a factory built under a context. It
/// resolved no signal, so its needs declare no cells: a host would hand
/// cells nothing reads. `remedy` is what the stage's author can do, and
/// `path` is the stage's own, made where a refusal names it.
fn no_cells_without_a_context(
    needs: Needs,
    remedy: &'static str,
    path: impl FnOnce() -> String,
) -> Result<(), SpecError> {
    StageBuild::held_to_cells(false, needs, remedy).map_err(|error| error.at(&path()))
}

/// What a filter entry stands for.
enum Filtered<'a> {
    /// A built-in, by the parameters the document wrote.
    Typed(&'a FilterStage),
    /// A filter the catalog built.
    Built(Box<dyn Filter>),
}

/// What the entry at `index` of a profile's list stands for. The catalog
/// is asked for the name first: a name it does not hold is refused as
/// unknown whatever is written under it. `at` is the profile's path: the
/// entry's own is written where a refusal names it, so an entry that
/// validates builds none.
fn filtered<'a>(
    entry: &'a FilterSpec,
    profile: &str,
    at: &At,
    index: usize,
    from: StageInputs<'_>,
    settings: SentSettings,
    sent: &mut SentSlot,
) -> Result<Filtered<'a>, SpecError> {
    let name = entry.name();
    let path = || format!("{}.filter[{index}].{name}", at.0);
    // The catalog's factory builds its filter from the document's map.
    // A filter has no weight: the map is the factory's whole.
    let made = |build: &FilterFactory,
                mut params: serde_json::Map<String, serde_json::Value>,
                sent: &mut SentSlot| {
        in_key_order(&mut params);
        let filter = StageBuild::built(
            &params,
            settings,
            sent,
            from.reports,
            from.signals,
            build,
            Filter::needs,
        )
        .map_err(|error| error.at(&path()))?;
        let answers = filter.name();
        if answers != name {
            return Err(misnamed(path(), StageKind::Filter, name, answers));
        }
        Ok(Filtered::Built(filter))
    };
    match (from.catalog.find_filter(name), entry) {
        // A host's filter under a name the engine keeps, whatever is
        // written under it.
        (Some(Found::Kept), _) => Err(kept_name(path(), StageKind::Filter, name)),
        // A plain filter whose build answers another name, whatever is
        // written under it.
        (Some(Found::Misnamed(answers)), _) => {
            Err(misnamed(path(), StageKind::Filter, name, answers))
        }
        // Only a document built in code: the parser reads the built-in's
        // key into its parameter struct. The catalog holds the name for
        // its bare form, and that arm's refusal would name the bare form.
        (_, FilterSpec::Added { .. }) if name == HasRoomRule::NAME => Err(SpecError::BadField {
            path: format!("{}.filter[{index}]", at.0),
            value: name.to_string(),
            reason: "a filter added in code: a built-in's parameters are typed",
        }),
        // A built-in's parameters are the validator's to check, under a
        // name the catalog holds.
        (Some(Found::Built(_) | Found::With(_)), FilterSpec::Params(stage)) => {
            Ok(Filtered::Typed(stage))
        }
        (Some(Found::Built(filter)), FilterSpec::Name(_)) => {
            no_cells_without_a_context(
                filter.needs(),
                "register it by filter_with to resolve one, or leave the need unset",
                path,
            )?;
            Ok(Filtered::Built(filter))
        }
        // A filter registered with no parameter is written by its name
        // alone: whatever stands under it is refused.
        (Some(Found::Built(_)), FilterSpec::Added { written, .. }) => Err(SpecError::BadField {
            path: path(),
            value: written.to_string(),
            reason: "left out: this filter takes no parameter, write its bare name",
        }),
        (Some(Found::With(build)), FilterSpec::Name(_)) => {
            made(build, serde_json::Map::new(), sent)
        }
        (Some(Found::With(build)), FilterSpec::Added { written, .. }) => {
            let params = written.as_object().ok_or_else(|| SpecError::BadField {
                path: path(),
                value: written.to_string(),
                reason: "a map of the filter's parameters",
            })?;
            made(build, params.clone(), sent)
        }
        // The catalog holds no filter by its type alone.
        (None | Some(Found::Typed), _) => Err(SpecError::UnknownFilter {
            profile: profile.to_string(),
            name: name.to_string(),
        }),
    }
}

/// The stage's filter, with the view's settings. `sent` hands out the
/// profile's count of sends.
fn filter_of(
    stage: &FilterStage,
    view: &ViewSpec,
    signals: &[SignalColumn],
    sent: impl FnOnce() -> Arc<DispatchCredit>,
    at: &At,
) -> Result<Box<dyn Filter>, SpecError> {
    match stage {
        FilterStage::HasRoom(p) => has_room_of(p, signals, sent, view.max_report_age_ms, at),
    }
}

/// Check the rule's parameters while resolving the field names into
/// readers and the signals' names into their columns. Only a term that
/// reads sends makes the profile's count.
fn has_room_of(
    p: &HasRoomParams,
    signals: &[SignalColumn],
    sent: impl FnOnce() -> Arc<DispatchCredit>,
    max_age_ms: Option<u64>,
    at: &At,
) -> Result<Box<dyn Filter>, SpecError> {
    // The first name, in name order, which is the map's own: a rule that
    // comes to read nothing of a worker is refused by it.
    let Some(first) = p.used.keys().next().map(String::as_str) else {
        return Err(at.bad("used", "{}", "at least one field with its coefficient"));
    };
    let mut terms = Vec::new();
    let mut signal_terms = Vec::new();
    let mut request_tokens = 0.0;
    let mut output_tokens = 0.0;
    for (name, &coefficient) in &p.used {
        // The request's own values are read off the request: no term of a
        // worker stands for one.
        let term = if name == HasRoomRule::REQUEST_TOKENS || name == HasRoomRule::OUTPUT_TOKENS {
            None
        } else {
            Some(named_term(signals, at, "used", name)?)
        };
        if coefficient == 0.0
            || !(-COST_PARAMETER_LIMIT..=COST_PARAMETER_LIMIT).contains(&coefficient)
        {
            return Err(at.sub("used").bad(
                name,
                coefficient,
                "a finite number other than zero, of magnitude at most 1e15",
            ));
        }
        match term {
            Some(Named::Field(term)) => terms.push((term, coefficient)),
            Some(Named::Signal(signal)) => signal_terms.push((signal, coefficient)),
            None if name == HasRoomRule::OUTPUT_TOKENS => output_tokens = coefficient,
            None => request_tokens = coefficient,
        }
    }
    // Under a signal's capacity the rule's own bound is that of a capacity
    // of one: a row's value scales it.
    let (capacity, capacity_signal) = match &p.capacity {
        Capacity::Fixed(capacity) if capacity.is_finite() && *capacity > 0.0 => (*capacity, None),
        Capacity::Fixed(capacity) => {
            return Err(at.bad("capacity", capacity, "a finite number above zero"));
        }
        // A name is a declared signal's: no field of this build holds a
        // capacity.
        Capacity::Field(name) => match signal_named(name) {
            Some(signal) => {
                let column = column_named(signals, signal)
                    .ok_or_else(|| at.bad("capacity", name, "a signal the document declares"))?;
                (1.0, Some(column.clone()))
            }
            None => {
                return Err(at.bad("capacity", name, "a declared signal, written signal.<name>"))
            }
        },
    };
    // Under a capacity that is a number, the request's own values alone are
    // one answer for every row: such a rule reads nothing of a worker, and
    // it keeps every row or steps aside. A signal in the sum reads a worker,
    // and so does a capacity read of each worker, so both are settled
    // first. Every name of such a rule is the request's, since any other
    // became a term or was refused: the refusal names the first of them.
    if terms.is_empty() && signal_terms.is_empty() && capacity_signal.is_none() {
        return Err(at.bad(
            "used",
            first,
            "at least one field or signal of the worker beside the request's own values",
        ));
    }
    if !(0.0..1.0).contains(&p.headroom) {
        return Err(at.bad("headroom", p.headroom, "a number from 0 to under 1"));
    }
    let reads_sent = terms.iter().any(|&(term, _)| term.reads_sent());
    let rule = HasRoomRule::new(
        &terms,
        request_tokens,
        output_tokens,
        capacity,
        p.headroom,
        reads_sent.then(sent),
    );
    // The signal form is a type of its own: a rule that names no signal
    // is built as it is with no signal declared.
    Ok(if signal_terms.is_empty() && capacity_signal.is_none() {
        has_room_rule(rule, max_age_ms)
    } else {
        Box::new(HasRoomSignals::new(
            rule,
            signal_terms.into_boxed_slice(),
            capacity_signal,
            max_age_ms,
        ))
    })
}

/// What a score entry stands for.
enum Scored {
    /// A built-in, by its parameters: the written ones, or a bare name's
    /// defaults.
    Typed(ScoreStage),
    /// A scorer the catalog built, with its weight.
    Built(WeightedScorer),
}

/// The stage a score entry stands for. The catalog is asked for the name
/// first: a name it does not hold is refused as unknown whatever is written
/// under it, so a misspelt built-in reads its name and not a key it carries.
fn scored(
    entry: &ScoreSpec,
    profile: &str,
    at: &At,
    from: StageInputs<'_>,
    settings: SentSettings,
    sent: &mut SentSlot,
) -> Result<Scored, SpecError> {
    let name = entry.name();
    let Some(found) = from.catalog.find_scorer(name) else {
        return Err(SpecError::UnknownScorer {
            profile: profile.to_string(),
            name: name.to_string(),
        });
    };
    // A scorer the catalog built with no context, bare or under a weight.
    if let Found::Built(scorer) = &found {
        no_cells_without_a_context(
            scorer.needs(),
            "register it by scorer_with to resolve one, or leave the need unset",
            || at.field(name),
        )?;
    }
    match (entry, found) {
        // A host's scorer under a name the engine keeps, whatever is
        // written under it.
        (_, Found::Kept) => Err(kept_name(at.sub(name).0, StageKind::Scorer, name)),
        // A plain scorer whose build answers another name, whatever is
        // written under it.
        (_, Found::Misnamed(answers)) => {
            Err(misnamed(at.sub(name).0, StageKind::Scorer, name, answers))
        }
        (ScoreSpec::Params(stage), _) => Ok(Scored::Typed(stage.clone())),
        (ScoreSpec::Name(_), Found::Typed) => match ScoreStage::by_name(name) {
            Some(stage) => Ok(Scored::Typed(stage)),
            None => Err(SpecError::BadField {
                path: at.0.clone(),
                value: name.to_string(),
                reason: match name {
                    "resident-work" => {
                        "a scorer with its request_tokens: { resident-work: { request_tokens } }"
                    }
                    LeastCost::NAME => "a scorer with its terms: { least-cost: { terms } }",
                    _ => "a scorer with its required parameters",
                },
            }),
        },
        (ScoreSpec::Name(_), Found::Built(scorer)) => Ok(Scored::Built(WeightedScorer {
            scorer,
            weight: 1.0,
        })),
        (ScoreSpec::Added { written, .. }, Found::Built(scorer)) => {
            Ok(Scored::Built(WeightedScorer {
                scorer,
                weight: added_weight(written, &at.sub(name))?,
            }))
        }
        (entry, Found::With(build)) => {
            let at = at.sub(name);
            let mut params = if let ScoreSpec::Added { written, .. } = entry {
                added_parameters(written, &at)?.clone()
            } else {
                serde_json::Map::new()
            };
            let weight = match params.shift_remove("weight") {
                Some(value) => check_added_weight(&value, &at)?,
                None => 1.0,
            };
            in_key_order(&mut params);
            let scorer = StageBuild::built(
                &params,
                settings,
                sent,
                from.reports,
                from.signals,
                build,
                Scorer::needs,
            )
            .map_err(|error| error.at(&at.0))?;
            let answers = scorer.name();
            if answers != name {
                return Err(misnamed(at.0, StageKind::Scorer, name, answers));
            }
            Ok(Scored::Built(WeightedScorer { scorer, weight }))
        }
        // Only a document built in code: the parser reads a built-in's key
        // into its parameter struct.
        (ScoreSpec::Added { .. }, Found::Typed) => Err(SpecError::BadField {
            path: at.0.clone(),
            value: name.to_string(),
            reason: "a scorer added in code: a built-in's parameters are typed",
        }),
    }
}

/// One order of a stage's parameters for its factory, the canonical bytes
/// and the digest: the keys of every map in order, whatever order the
/// document wrote them in.
fn in_key_order(params: &mut serde_json::Map<String, serde_json::Value>) {
    params.sort_keys();
    params
        .values_mut()
        .for_each(serde_json::Value::sort_all_objects);
}

fn added_parameters<'a>(
    written: &'a serde_json::Value,
    at: &At,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, SpecError> {
    written.as_object().ok_or_else(|| SpecError::BadField {
        path: at.0.clone(),
        value: written.to_string(),
        reason: "a map of the scorer's weight: { weight }",
    })
}

/// The weight of a scorer whose factory takes no parameters. Any other key
/// is refused by name. An empty map weighs one.
fn added_weight(written: &serde_json::Value, at: &At) -> Result<f32, SpecError> {
    let keys = added_parameters(written, at)?;
    let mut weight = 1.0;
    for (key, value) in keys {
        if key != "weight" {
            return Err(at.bad(
                key,
                value,
                "a parameter of a scorer added in code, which takes its weight alone",
            ));
        }
        weight = check_added_weight(value, at)?;
    }
    Ok(weight)
}

fn check_added_weight(value: &serde_json::Value, at: &At) -> Result<f32, SpecError> {
    // A number past an `f32` narrows to infinity and is refused below.
    let weight = value.as_f64().map_or(f32::NAN, |number| number as f32);
    if !above_zero(weight) {
        return Err(at.bad("weight", value, "a finite number above zero"));
    }
    Ok(weight)
}

/// A profile as [`check_profile`] built it: the profile, whether it keeps a
/// session key on its worker, and the count of sends its stages read, where
/// one of them asked for it.
type BuiltProfile = (Profile, bool, Option<Arc<DispatchCredit>>);

/// One profile's stages, built when this build carries every one of them,
/// and whether the profile keeps a session key on its worker over the
/// overload veto: the plain ring alone decides its pick. `depth` is
/// deepened to what the built stages read of a request's block hashes.
///
/// The build reads four values of `view` and no header: the block size, the
/// boundaries, the bound of block hashes and the age limit of load reports.
/// The keep of a profile across a new header list rests on that
/// (`AppliedSpec::keep_from`). `shared` answers the count of sends that an
/// earlier profile's stages asked for under the same two send settings.
pub(super) fn check_profile(
    name: &str,
    p: &ProfileSpec,
    view: &ViewSpec,
    from: StageInputs<'_>,
    deferred: &mut Deferred,
    depth: &mut HashDepths,
    shared: impl FnOnce(SentSettings) -> Option<Arc<DispatchCredit>>,
) -> Result<Option<BuiltProfile>, SpecError> {
    // The empty label is the engine's word for no profile.
    if name.is_empty() {
        return Err(SpecError::BadField {
            path: "profiles".to_owned(),
            value: format!("{name:?}"),
            reason: "a profile name of at least one character",
        });
    }
    let catalog = from.catalog;
    let gates = gates_of(p, view);
    let at = At::new(format!("profiles.{name}"));
    at.nonzero("sent_ttl_ms", p.sent_ttl_ms)?;
    at.nonzero("sent_report_delay_ms", p.sent_report_delay_ms)?;
    let settings = SentSettings {
        ttl_ms: p.sent_ttl_ms,
        report_delay_ms: p.sent_report_delay_ms,
    };
    // The profile's count of sends, made where a stage's factory first
    // asks for it: a filter's may, ahead of any scorer's. An earlier
    // profile's count under the same two settings is the one handed.
    let mut sent = SentSlot {
        asked: None,
        shared: shared(settings),
    };
    let mut filters = Vec::with_capacity(p.filter.len());
    for (index, entry) in p.filter.iter().enumerate() {
        let filter = match filtered(entry, name, &at, index, from, settings, &mut sent)? {
            Filtered::Built(filter) => filter,
            Filtered::Typed(stage) => {
                let stage_at = At::new(format!("{}.filter[{index}].{}", at.0, stage.name()));
                let count = || profile_sent(&mut sent, settings);
                filter_of(stage, view, from.signals, count, &stage_at)?
            }
        };
        filters.push(filter);
    }
    let mut scorers = Vec::with_capacity(p.score.len());
    let mut complete = true;
    let mut plain_ring = false;
    let mut kv_gate = None;
    let mut spill_gate = None;
    let mut other_spill = None;
    let mut sliced = false;
    for (i, entry) in p.score.iter().enumerate() {
        let entry_at = At::new(format!("{}.score[{i}]", at.0));
        let stage = match scored(entry, name, &entry_at, from, settings, &mut sent)? {
            Scored::Typed(stage) => stage,
            Scored::Built(weighted) => {
                depth.deepen(HashDepth::of(weighted.scorer.needs()), true);
                scorers.push(weighted);
                continue;
            }
        };
        let stage_at = entry_at.sub(stage.name());
        check_stage(&stage, &stage_at)?;
        // A profile has one KV gate, and `gates_of` builds it from one
        // entry: an entry that writes other limits would load and do
        // nothing. At 1 and above a limit is off.
        if let ScoreStage::KvPressure(params) = &stage {
            let own = Gates::of(None, Some(params), None);
            let first = *kv_gate.get_or_insert(own);
            for (limit, own, first) in [
                ("spread", own.kv_spread, first.kv_spread),
                ("ceiling", own.kv_ceiling, first.kv_ceiling),
            ] {
                if own.min(1.0) != first.min(1.0) {
                    return Err(stage_at.bad(
                        limit,
                        own,
                        "the limit the profile's first kv-pressure entry writes: a profile has one KV gate, so its entries write the same limits",
                    ));
                }
            }
        }
        // A `prefix-affinity` scorer spills by its own gate; a warm-up slice
        // asks the shared one with no held tokens: its `rel` and `abs` alone.
        match &stage {
            ScoreStage::PrefixAffinity(params) if other_spill.is_none() => {
                let own = Gates::of(Some(params), None, None);
                let first = *spill_gate.get_or_insert(own);
                if own.rel != first.rel || own.abs != first.abs {
                    other_spill = Some((i, own, first));
                }
            }
            ScoreStage::WarmupSlice(_) => sliced = true,
            _ => {}
        }
        if let Some((entry, own, first)) = other_spill.take_if(|_| sliced) {
            let spill_at = At::new(format!("{}.score[{entry}].prefix-affinity.spill", at.0));
            let reason = "the limit the profile's first prefix-affinity entry writes: a warmup-slice follows one spill gate, so beside one the entries write the same rel and abs";
            return Err(if own.rel == first.rel {
                spill_at.bad("abs", own.abs, reason)
            } else {
                spill_at.bad("rel", own.rel, reason)
            });
        }
        check_prefix_depth(&stage, gates, view, &stage_at)?;
        let count = || profile_sent(&mut sent, settings);
        match scorer_of(&stage, gates, view, from.signals, count, &stage_at)? {
            Resolved::Built(scorer) => {
                let read = HashDepth::of_scorer(&stage, &view.boundaries, scorer.needs());
                depth.deepen(read, !matches!(stage, ScoreStage::PrefixAffinity(_)));
                plain_ring |= matches!(
                    &stage,
                    ScoreStage::ConsistentHash(ring)
                        if ring.prefix_tokens.is_none() && ring.bounded_load.is_none()
                );
                scorers.push(WeightedScorer {
                    scorer,
                    weight: stage.weight(),
                });
            }
            Resolved::NotInBuild(what) => {
                complete = false;
                deferred.note(stage_at.0, what);
            }
        }
    }
    let picker = match picker_of(&p.pick, name, &at, from, settings, &mut sent)? {
        Resolved::Built(picker) => Some(picker),
        Resolved::NotInBuild(what) => {
            deferred.note(at.field("pick"), what);
            None
        }
    };
    if let Some(ms) = p.sent_ttl_ms.filter(|_| sent.asked.is_none()) {
        return Err(at.bad("sent_ttl_ms", ms, "a limit any stage of this profile reads"));
    }
    if let Some(ms) = p.sent_report_delay_ms.filter(|_| sent.asked.is_none()) {
        return Err(at.bad(
            "sent_report_delay_ms",
            ms,
            "a delay any stage of this profile reads",
        ));
    }
    // The scores combine by each weight's share of the weights' sum: a sum
    // that is not finite, or whose reciprocal is not, gives shares that
    // lose the order of every row. A stage this build does not carry adds
    // no weight, so a deferred profile's sum is over part of its list. An
    // infinite part is an infinite whole, every weight being above zero,
    // and a part's reciprocal says nothing of the whole's.
    let total: f32 = scorers.iter().map(|weighted| weighted.weight).sum();
    let no_shares = !total.is_finite() || (complete && !total.recip().is_finite());
    if !scorers.is_empty() && no_shares {
        return Err(at.bad(
            "score",
            total,
            "weights whose sum, and one over that sum, are finite numbers",
        ));
    }
    // The key's mark: the plain ring alone decides the profile's pick. It
    // is the only scorer, with no bound, under `best` named bare, so the
    // key's owner holds the best score and the picker takes it. The engine
    // lifts the overload veto for the whole run of a marked profile, and
    // only in this shape does that run keep the key on its owner. A bound
    // passes an owner over and ranks no row, another scorer can outweigh
    // the owner, and a picker that draws, turns or samples does not follow
    // the ring: each keeps the veto for a keyed request. The prefix form
    // takes no mark either, though a request's key leads its prefix there:
    // the gateway's prefix hash moves a key off a worker over its
    // thresholds.
    let by_key = plain_ring
        && p.score.len() == 1
        && matches!(&p.pick, PickSpec::Name(picker) if picker == "best");
    // A host that makes some lifecycle calls only holds no stage that needs
    // them: it would decide on state nothing keeps. Every entry of a
    // complete profile is built, so a scorer stands at its entry's place.
    if complete && !catalog.makes_lifecycle_calls() {
        let mut entries = p.score.iter().zip(&scorers).enumerate();
        let scorer = entries.find_map(|(i, (entry, built))| {
            let needs = built.scorer.needs_lifecycle();
            needs.then(|| format!("{}.score[{i}].{}", at.0, entry.name()))
        });
        let needs = picker
            .as_ref()
            .is_some_and(|picker| picker.needs_lifecycle());
        if let Some(path) = scorer.or_else(|| needs.then(|| at.field("pick"))) {
            complete = false;
            deferred.note(path, "the lifecycle this stage needs");
        }
    }
    if p.score.is_empty() && SAMPLING_PICKERS.contains(&p.pick.name()) {
        return Err(SpecError::PickerNeedsScorer {
            profile: name.to_string(),
            picker: p.pick.name().to_string(),
        });
    }
    // A picker that leaves the placement to the binding's rule: the engine
    // never runs such a profile's scorers, so a document that names one
    // names a stage that never runs.
    if !p.score.is_empty() && picker.as_ref().is_some_and(|picker| picker.by_assignment()) {
        return Err(SpecError::PickerRunsNoScorer {
            profile: name.to_string(),
            picker: p.pick.name().to_string(),
        });
    }
    Ok(match picker {
        Some(picker) if complete => {
            // A filter and a picker name no depth of their own.
            let unkeyed = filters.iter().map(|filter| filter.needs());
            depth.deepen(HashDepth::of(unkeyed.fold(picker.needs(), Needs::or)), true);
            Some((
                Profile {
                    name: name.to_string(),
                    filters,
                    scorers,
                    picker,
                },
                by_key,
                sent.asked,
            ))
        }
        _ => None,
    })
}

#[cfg(test)]
#[path = "../../../../tests/unit/document/spec/validate/profile/tests.rs"]
mod tests;
