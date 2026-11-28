//! Validation of a [`PolicySpec`]: the checks of the view, the index map,
//! the signals and each binding with its rules, and the document's four
//! functions that run them and build the engine: [`PolicySpec::validate`],
//! [`PolicySpec::validate_with`], [`PolicySpec::validate_in`] and
//! [`PolicySpec::built`]. The checks of one profile's stages are in
//! [`profile`].

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::Arc,
};

use self::profile::check_profile;
use super::{
    selector, Assignment, BindingSpec, Bounds, CanarySpec, HashDepths, HeaderMatch, IndexKind,
    IndexSpec, Match, Needs, OverloadSpec, Override, PlanName, PolicySpec, RoutingKeySpec,
    RulePlaced, SessionsSpec, ShadowSpec, SignalSpec, SpecError, ViewSpec, API_VERSION,
    DENIED_HEADERS, DENIED_KEY_HEADERS, MAX_BLOCK_HASHES, MAX_CLIENT_SOURCES, MAX_ENGINES,
    MAX_HEADERS, MAX_KEY_BYTES, MAX_KEY_SOURCES, MAX_LABELS, MAX_SIGNALS, MAX_SIGNAL_NAME,
    MAX_SIGNAL_UNIT, MAX_ZONES, SIDES, STOCK,
};
use crate::{
    engine::{
        framework::{Binding, Engine, Plan, Profile},
        sticky_sessions::{SessionRule, SessionSides},
        view::MAX_LOAD_REPORT_AGE_MS,
    },
    load::{
        load_report_reader::LoadReportReader,
        signals::{column_named, SignalColumn},
    },
    stages::{
        cost_terms::{signal_named, CostTerm},
        registry::StageRegistry,
    },
};

mod profile;

/// A field path under construction, for the messages.
struct At(String);

impl At {
    fn new(path: impl Into<String>) -> Self {
        Self(path.into())
    }

    fn field(&self, name: &str) -> String {
        format!("{}.{name}", self.0)
    }

    fn sub(&self, name: &str) -> Self {
        Self(self.field(name))
    }

    fn bad(&self, name: &str, value: impl fmt::Display, reason: &'static str) -> SpecError {
        SpecError::BadField {
            path: self.field(name),
            value: value.to_string(),
            reason,
        }
    }

    /// A written number passes `ok`, or is refused for `reason`.
    fn number<T: Copy + fmt::Display>(
        &self,
        name: &str,
        value: Option<T>,
        ok: fn(T) -> bool,
        reason: &'static str,
    ) -> Result<(), SpecError> {
        match value {
            Some(v) if !ok(v) => Err(self.bad(name, v, reason)),
            _ => Ok(()),
        }
    }

    /// A written count is above zero.
    fn nonzero<T>(&self, name: &str, value: Option<T>) -> Result<(), SpecError>
    where
        T: Copy + Default + PartialEq + fmt::Display,
    {
        match value {
            Some(v) if v == T::default() => Err(self.bad(name, v, "above zero")),
            _ => Ok(()),
        }
    }
}

fn above_zero(v: f32) -> bool {
    v.is_finite() && v > 0.0
}

fn at_or_above_zero(v: f32) -> bool {
    v.is_finite() && v >= 0.0
}

fn unit_interval(v: f32) -> bool {
    v.is_finite() && (0.0..=1.0).contains(&v)
}

fn at_least_one<T: Into<f64>>(v: T) -> bool {
    let v: f64 = v.into();
    v.is_finite() && v >= 1.0
}

/// The matcher understands an id, `prefix*`, `*suffix` and `*`; any other
/// star would match only its literal self.
fn glob_ok(model: &str) -> bool {
    let stars = model.matches('*').count();
    !model.is_empty()
        && (stars == 0 || (stars == 1 && (model.starts_with('*') || model.ends_with('*'))))
}

/// A lowercase header name outside the deny-lists.
fn check_header_name(name: &str, path: String, denied: &[&str]) -> Result<(), SpecError> {
    check_header_outside(name, path, &DENIED_HEADERS, denied)
}

/// A lowercase header name outside `stock` and the host's own `denied`.
fn check_header_outside(
    name: &str,
    path: String,
    stock: &[&str],
    denied: &[&str],
) -> Result<(), SpecError> {
    let token =
        |b: u8| b.is_ascii_lowercase() || b.is_ascii_digit() || b"!#$%&'*+-.^_`|~".contains(&b);
    if name.is_empty() || !name.bytes().all(token) {
        return Err(SpecError::BadField {
            path,
            value: name.to_string(),
            reason: "a lowercase header name",
        });
    }
    if stock.contains(&name) || denied.iter().any(|d| d.eq_ignore_ascii_case(name)) {
        return Err(SpecError::DeniedHeader {
            path,
            name: name.to_string(),
        });
    }
    Ok(())
}

/// Each name once.
fn check_unique<'a>(
    names: impl IntoIterator<Item = &'a String>,
    path: &str,
) -> Result<(), SpecError> {
    let mut seen = BTreeSet::new();
    for name in names {
        if !seen.insert(name.as_str()) {
            return Err(SpecError::Duplicate {
                path: path.to_string(),
                name: name.clone(),
            });
        }
    }
    Ok(())
}

fn check_view(view: &ViewSpec, denied: &[&str]) -> Result<(), SpecError> {
    let at = At::new("view");
    if view.headers.len() > MAX_HEADERS {
        return Err(SpecError::TooManyHeaders {
            count: view.headers.len(),
            max: MAX_HEADERS,
        });
    }
    for (i, header) in view.headers.iter().enumerate() {
        check_header_name(header, format!("view.headers[{i}]"), denied)?;
    }
    check_unique(&view.headers, "view.headers")?;
    at.nonzero("block_size", view.block_size)?;
    at.number(
        "max_report_age_ms",
        view.max_report_age_ms,
        |ms| (1..=u64::from(MAX_LOAD_REPORT_AGE_MS)).contains(&ms),
        "a whole number from 1 to 3599999",
    )?;
    if view.boundaries.first() == Some(&0) {
        return Err(SpecError::BadField {
            path: "view.boundaries[0]".to_string(),
            value: "0".to_string(),
            reason: "above zero",
        });
    }
    for (i, pair) in view.boundaries.windows(2).enumerate() {
        if let [low, high] = pair {
            if low >= high {
                return Err(SpecError::BadField {
                    path: format!("view.boundaries[{}]", i + 1),
                    value: high.to_string(),
                    reason: "above the boundary before it",
                });
            }
        }
    }
    at.nonzero("max_block_hashes", Some(view.max_block_hashes))?;
    if view.max_block_hashes > MAX_BLOCK_HASHES {
        return Err(at.bad(
            "max_block_hashes",
            view.max_block_hashes,
            "at most 8190, the hashes one Decide frame carries",
        ));
    }
    check_list(&view.labels, "view.labels", MAX_LABELS)?;
    for (i, label) in view.labels.iter().enumerate() {
        let (key, value) = selector(label);
        if key.is_empty() || value == Some("") {
            return Err(SpecError::BadField {
                path: format!("view.labels[{i}]"),
                value: format!("{label:?}"),
                reason: "a label key or `key=value` with no empty side",
            });
        }
    }
    check_list(&view.zones, "view.zones", MAX_ZONES)?;
    check_list(&view.engines, "view.engines", MAX_ENGINES)
}

/// A list the view declares: at most `max` entries, none empty, each once.
fn check_list(list: &[String], path: &'static str, max: usize) -> Result<(), SpecError> {
    if list.len() > max {
        return Err(SpecError::TooManyListed {
            path,
            count: list.len(),
            max,
        });
    }
    if let Some(i) = list.iter().position(String::is_empty) {
        return Err(SpecError::BadField {
            path: format!("{path}[{i}]"),
            value: "\"\"".to_string(),
            reason: "a name",
        });
    }
    check_unique(list, path)
}

fn check_index(index: &BTreeMap<String, IndexSpec>, view: &ViewSpec) -> Result<(), SpecError> {
    for (model, spec) in index {
        // The key's path is built at a refusal alone: a key that passes
        // formats nothing.
        let at = || At::new(format!("index.{model}"));
        if !glob_ok(model) {
            return Err(SpecError::BadField {
                path: at().0,
                value: model.clone(),
                reason: "an id, `prefix*`, `*suffix` or `*`",
            });
        }
        let counts = [
            ("ttl_s", u64::from(spec.ttl_s)),
            ("max_size", spec.max_size),
            ("sweep_s", u64::from(spec.sweep_s)),
        ];
        if let Some((name, zero)) = counts.into_iter().find(|(_, count)| *count == 0) {
            return Err(at().bad(name, zero, "above zero"));
        }
        if spec.kind == IndexKind::Hash && view.boundaries.is_empty() {
            return Err(SpecError::HashIndexNeedsBoundaries {
                model: model.clone(),
            });
        }
    }
    match overlap(index.keys()) {
        Some((first, second)) => Err(SpecError::IndexOverlap {
            first: first.clone(),
            second: second.clone(),
        }),
        None => Ok(()),
    }
}

/// The `signals` section's checks, each refused by its path: at most
/// [`MAX_SIGNALS`] names, each a lower-case identifier of at most 63
/// bytes, a unit of 1 to 31 ASCII characters with no space, and a maximum
/// age of 1 to 3,599,999 where one is written. A signal no stage reads is allowed: a host's own
/// stage may read it.
fn check_signals(signals: &BTreeMap<String, SignalSpec>) -> Result<(), SpecError> {
    if signals.len() > MAX_SIGNALS {
        return Err(SpecError::TooManyListed {
            path: "signals",
            count: signals.len(),
            max: MAX_SIGNALS,
        });
    }
    for (name, spec) in signals {
        let mut bytes = name.bytes();
        let named = name.len() <= MAX_SIGNAL_NAME
            && bytes.next().is_some_and(|first| first.is_ascii_lowercase())
            && bytes.all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
        if !named {
            return Err(SpecError::BadField {
                path: "signals".to_owned(),
                value: name.clone(),
                reason: "a lower-case name of letters, digits and underscores, at most 63 bytes",
            });
        }
        // The key's path is built at a refusal alone.
        let at = || At::new(format!("signals.{name}"));
        let visible = spec.unit.bytes().all(|b| b.is_ascii_graphic());
        if spec.unit.is_empty() || spec.unit.len() > MAX_SIGNAL_UNIT || !visible {
            return Err(at().bad(
                "unit",
                format!("{:?}", spec.unit),
                "a unit of 1 to 31 ASCII characters with no space",
            ));
        }
        match spec.max_age_ms {
            Some(0) => return Err(at().bad("max_age_ms", 0, "above zero")),
            Some(ms) if ms > MAX_LOAD_REPORT_AGE_MS => {
                return Err(at().bad("max_age_ms", ms, "a whole number from 1 to 3599999"));
            }
            _ => {}
        }
    }
    Ok(())
}

/// The document's signals, resolved once: one column a name, in name
/// order, so a name's place in the list is its column's number.
fn resolve_signals(signals: &BTreeMap<String, SignalSpec>) -> Arc<[SignalColumn]> {
    signals
        .iter()
        .zip(0u16..)
        .map(|((name, spec), column)| {
            SignalColumn::new(name, &spec.unit, column, spec.max_age_ms).across(spec.across_ranks)
        })
        .collect()
}

/// What a term's written name reads: a field of this build, or a signal
/// the document declares.
enum Named {
    Field(CostTerm),
    Signal(SignalColumn),
}

/// The term `name` in the map `terms` of the stage at `at`. A name of the
/// signal family (`signal.<name>`) reads a signal the document declares,
/// and any other name a field of this build. A name that reads neither is
/// refused at the map, by the name as written.
fn named_term(
    signals: &[SignalColumn],
    at: &At,
    terms: &str,
    name: &str,
) -> Result<Named, SpecError> {
    match signal_named(name) {
        Some(signal) => column_named(signals, signal)
            .map(|column| Named::Signal(column.clone()))
            .ok_or_else(|| at.bad(terms, name, "a signal the document declares")),
        None => CostTerm::named(name)
            .map(Named::Field)
            .ok_or_else(|| at.bad(terms, name, "a field this build reads")),
    }
}

/// The first two keys that overlap, in the keys' order: the first key with
/// a partner after it, and the first such partner. A `prefix*` and a
/// `*suffix` of one literal length overlap: they match a common model and
/// neither is the more specific, so the pair is refused.
///
/// Each key is read once, whatever their number. Only the first `prefix*`
/// and the first `*suffix` of a literal length can be that pair, so those
/// are what the pass keeps.
fn overlap<K: AsRef<str>>(keys: impl IntoIterator<Item = K>) -> Option<(K, K)> {
    // By literal length: the first key of each side, with its place.
    let mut firsts: BTreeMap<usize, [Option<(usize, K)>; 2]> = BTreeMap::new();
    for (place, key) in keys.into_iter().enumerate() {
        let Some((which, literal)) = side(key.as_ref()) else {
            continue;
        };
        let first = &mut firsts.entry(literal).or_insert([None, None])[which];
        if first.is_none() {
            *first = Some((place, key));
        }
    }
    firsts
        .into_values()
        .filter_map(|sides| match sides {
            [Some(prefix), Some(suffix)] if prefix.0 < suffix.0 => Some((prefix, suffix)),
            [Some(prefix), Some(suffix)] => Some((suffix, prefix)),
            _ => None,
        })
        .min_by_key(|(first, _)| first.0)
        .map(|((_, first), (_, second))| (first, second))
}

/// The side of a model a glob holds, 0 for `prefix*` and 1 for `*suffix`,
/// with its literal length. An id and `*` hold none.
fn side(glob: &str) -> Option<(usize, usize)> {
    match (glob.strip_suffix('*'), glob.strip_prefix('*')) {
        _ if glob == "*" => None,
        (Some(prefix), _) => Some((0, prefix.len())),
        (_, Some(suffix)) => Some((1, suffix.len())),
        _ => None,
    }
}

/// The first thing this build lacks on the validator's walk (profiles in
/// name order, then bindings in their order), kept until the whole
/// document's shape has been read: a shape error outranks it.
#[derive(Default)]
struct Deferred(Option<SpecError>);

impl Deferred {
    fn note(&mut self, path: String, what: &'static str) {
        if self.0.is_none() {
            self.0 = Some(SpecError::NotInBuild { path, what });
        }
    }
}

/// What a document's stages are built from: the catalog that makes them,
/// the view's reading of a report, which a factory's build context hands
/// out, and the document's signals, resolved, which it answers from.
#[derive(Clone, Copy)]
struct StageInputs<'a> {
    catalog: &'a StageRegistry,
    reports: LoadReportReader,
    signals: &'a [SignalColumn],
}

/// What a binding's checks read beside the binding.
struct BindingContext<'a> {
    index: usize,
    view: &'a ViewSpec,
    denied: &'a [&'a str],
    /// Profile name to its position in the document's map.
    profiles: &'a BTreeMap<&'a str, u16>,
    live_profiles: &'a BTreeSet<&'a str>,
    scoped_stages: &'a BTreeMap<&'a str, &'static str>,
    /// The profiles whose built picker leaves the placement to a sessions
    /// rule ([`Picker::by_assignment`](crate::engine::Picker::by_assignment)), by name.
    by_assignment: &'a [&'a str],
    /// Whether the host makes the lifecycle's calls
    /// ([`StageRegistry::makes_lifecycle_calls`]). One that makes some
    /// only tells a lease's words off a file of leases.
    lifecycle: bool,
}

fn check_bounds<T>(bounds: &Bounds<T>, at: &At) -> Result<(), SpecError>
where
    T: Copy + PartialOrd + fmt::Display,
{
    match (bounds.min, bounds.max) {
        (None, None) => Err(SpecError::BadField {
            path: at.0.clone(),
            value: "{}".to_string(),
            reason: "a range with min, max or both",
        }),
        (Some(min), Some(max)) if min > max => Err(at.bad("min", min, "at or below max")),
        _ => Ok(()),
    }
}

/// A header match names an allow-listed header.
fn check_header_match(
    header: &HeaderMatch,
    cx: &BindingContext<'_>,
    at: &At,
) -> Result<(), SpecError> {
    check_header_name(&header.name, at.field("name"), cx.denied)?;
    if !cx.view.headers.contains(&header.name) {
        return Err(SpecError::HeaderNotAllowed {
            path: at.field("name"),
            name: header.name.clone(),
        });
    }
    Ok(())
}

fn check_matcher(
    m: &Match,
    cx: &BindingContext<'_>,
    at: &At,
    deferred: &mut Deferred,
) -> Result<(), SpecError> {
    if !glob_ok(&m.model) {
        return Err(SpecError::BadModelGlob {
            index: cx.index,
            model: m.model.clone(),
        });
    }
    let at = at.sub("match");
    if let Some(tenant) = &m.tenant {
        if tenant.is_empty() {
            return Err(at.bad("tenant", "\"\"", "a tenant id"));
        }
        deferred.note(at.field("tenant"), "the tenant matcher");
    }
    if let Some(priority) = &m.priority {
        check_bounds(priority, &at.sub("priority"))?;
        deferred.note(at.field("priority"), "the priority matcher");
    }
    if let Some(header) = &m.header {
        check_header_match(header, cx, &at.sub("header"))?;
        deferred.note(at.field("header"), "the header matcher");
    }
    if let Some(prompt_tokens) = &m.prompt_tokens {
        check_bounds(prompt_tokens, &at.sub("prompt_tokens"))?;
        deferred.note(at.field("prompt_tokens"), "the prompt-tokens matcher");
    }
    if !m.labels.is_empty() {
        check_unique(&m.labels, &at.field("labels"))?;
        deferred.note(at.field("labels"), "the label matcher");
    }
    Ok(())
}

/// A binding's overload block names no number, with its switch on or off:
/// a block that names one is refused by the number's path. One set of
/// thresholds stands for each worker, its registration's own over its
/// program's default, and a worker is judged by that set alone.
fn check_overload(overload: &OverloadSpec, at: &At) -> Result<(), SpecError> {
    const STATED: &str =
        "left out: a worker's registration states this threshold, or its program's default does";
    if let Some(waiting) = overload.waiting_requests {
        return Err(at.bad("waiting_requests", waiting, STATED));
    }
    match overload.token_usage {
        Some(usage) => Err(at.bad("token_usage", usage, STATED)),
        None => Ok(()),
    }
}

fn check_sessions(
    sessions: &SessionsSpec,
    by_routing_key: bool,
    cx: &BindingContext<'_>,
    at: &At,
) -> Result<(), SpecError> {
    if by_routing_key {
        // A binding names its key's sources once: under `routing_key` the
        // rule's own spelling is refused by name.
        for (name, written) in [
            ("sticky_header", !sessions.sticky_header.is_empty()),
            ("headers", !sessions.headers.is_empty()),
        ] {
            if written {
                return Err(SpecError::BadSessions {
                    index: cx.index,
                    name,
                    reason: "left out while the binding names routing_key",
                });
            }
        }
    } else if sessions.sticky_header.is_empty() {
        return Err(SpecError::BadSessions {
            index: cx.index,
            name: "sticky_header",
            reason: "named while the binding names no routing_key",
        });
    } else {
        check_session_headers(sessions, cx, at)?;
    }
    at.nonzero("ttl_s", Some(sessions.ttl_s))?;
    at.nonzero("sweep_s", Some(sessions.sweep_s))?;
    at.nonzero("inflight_cap", Some(sessions.inflight_cap))
}

/// The headers a sessions rule names itself: each a header name on the
/// view's allow-list, the sticky header among the listed ones.
fn check_session_headers(
    sessions: &SessionsSpec,
    cx: &BindingContext<'_>,
    at: &At,
) -> Result<(), SpecError> {
    check_header_name(
        &sessions.sticky_header,
        at.field("sticky_header"),
        cx.denied,
    )?;
    for (i, header) in sessions.headers.iter().enumerate() {
        check_header_name(header, format!("{}.headers[{i}]", at.0), cx.denied)?;
    }
    check_unique(&sessions.headers, &at.field("headers"))?;
    if !sessions.headers.is_empty() && !sessions.headers.contains(&sessions.sticky_header) {
        return Err(at.bad(
            "sticky_header",
            &sessions.sticky_header,
            "one of sessions.headers",
        ));
    }
    if !cx.view.headers.contains(&sessions.sticky_header) {
        return Err(SpecError::HeaderNotAllowed {
            path: at.field("sticky_header"),
            name: sessions.sticky_header.clone(),
        });
    }
    for header in &sessions.headers {
        if !cx.view.headers.contains(header) {
            return Err(SpecError::HeaderNotAllowed {
                path: at.field("headers"),
                name: header.clone(),
            });
        }
    }
    Ok(())
}

/// A binding's routing key sources: one to eight entries, with up to four
/// client entries and one target entry beside them, never those with none
/// of the eight, each a header outside the credential names or the
/// request's own id, each named once as a session's source, once as a
/// client's and once as the target's, with a byte cap inside the schema's.
fn check_routing_key(
    block: &RoutingKeySpec,
    cx: &BindingContext<'_>,
    at: &At,
) -> Result<(), SpecError> {
    let targets = block.sources.iter().filter(|source| source.target).count();
    let clients = block
        .sources
        .iter()
        .filter(|source| source.client && !source.target);
    let clients = clients.count();
    let listed = block.sources.len() - clients - targets;
    if block.sources.is_empty() || listed > MAX_KEY_SOURCES {
        return Err(at.bad(
            "sources",
            format_args!("{listed} entries"),
            "a list of one to eight sources",
        ));
    }
    if clients > MAX_CLIENT_SOURCES {
        return Err(at.bad(
            "sources",
            format_args!("{clients} client entries"),
            "a list of four client sources at most",
        ));
    }
    if targets > 1 {
        return Err(at.bad(
            "sources",
            format_args!("{targets} target entries"),
            "a list of one target source at most",
        ));
    }
    let mut seen = BTreeSet::new();
    for (i, source) in block.sources.iter().enumerate() {
        let at = At::new(format!("{}.sources[{i}]", at.0));
        let name = match (&source.header, source.rid) {
            (Some(name), false) => {
                check_header_outside(name, at.field("header"), &DENIED_KEY_HEADERS, cx.denied)?;
                name.as_str()
            }
            (None, true) => "rid",
            (written, _) => {
                return Err(SpecError::BadField {
                    path: at.0,
                    value: written.as_ref().map_or("neither", |_| "both").to_string(),
                    reason: "one of header and rid",
                });
            }
        };
        if source.client && (source.rid || source.max_bytes.is_some()) {
            return Err(SpecError::BadField {
                path: at.field("client"),
                value: "true".to_string(),
                reason: "a header read whole, with no rid and no max_bytes beside it",
            });
        }
        if source.target && (source.rid || source.max_bytes.is_some() || source.client) {
            return Err(SpecError::BadField {
                path: at.field("target"),
                value: "true".to_string(),
                reason: "a header alone, with no rid, no max_bytes and no client beside it",
            });
        }
        if !seen.insert((source.rid, source.client, source.target, name)) {
            return Err(SpecError::Duplicate {
                path: at.0,
                name: name.to_string(),
            });
        }
        at.number(
            "max_bytes",
            source.max_bytes,
            |bytes| (1..=MAX_KEY_BYTES).contains(&bytes),
            "between 1 and 8192",
        )?;
    }
    if listed == 0 {
        // Neither hands a session's key: a sessions rule beside the block
        // would have no source of one.
        let alone = match targets {
            0 => "client entries alone",
            _ => "a target entry and no source of a key",
        };
        return Err(at.bad("sources", alone, "a list of one to eight sources"));
    }
    Ok(())
}

fn check_canary(canary: &CanarySpec, cx: &BindingContext<'_>, at: &At) -> Result<(), SpecError> {
    if !cx.profiles.contains_key(canary.profile.as_str()) {
        return Err(SpecError::UnknownProfileAt {
            path: at.field("profile"),
            name: canary.profile.clone(),
        });
    }
    let forms = usize::from(canary.percent.is_some())
        + usize::from(!canary.tenants.is_empty())
        + usize::from(canary.header.is_some());
    if forms != 1 {
        return Err(SpecError::CanaryForm {
            path: at.0.clone(),
            count: forms,
        });
    }
    if let Some(percent) = canary.percent.filter(|p| *p == 0 || *p > 100) {
        return Err(at.bad("percent", percent, "between 1 and 100"));
    }
    check_unique(&canary.tenants, &at.field("tenants"))?;
    match &canary.header {
        Some(header) => check_header_match(header, cx, &at.sub("header")),
        None => Ok(()),
    }
}

/// The first stage that needs to open a decision scope, in stage order.
fn scoped_stage(profile: &Profile) -> Option<&'static str> {
    profile
        .filters
        .iter()
        .find_map(|stage| stage.scope().map(|_| stage.name()))
        .or_else(|| {
            profile
                .scorers
                .iter()
                .find_map(|stage| stage.scorer.scope().map(|_| stage.scorer.name()))
        })
        .or_else(|| profile.picker.scope().map(|_| profile.picker.name()))
}

fn resolve_shadow_profile(
    name: &str,
    at: At,
    cx: &BindingContext<'_>,
    deferred: &mut Deferred,
) -> Result<u16, SpecError> {
    let Some(&position) = cx.profiles.get(name) else {
        return Err(SpecError::UnknownProfileAt {
            path: at.0,
            name: name.to_owned(),
        });
    };
    // Stages may change their state when asked to decide. A live profile
    // must have its own stages.
    if cx.live_profiles.contains(name) {
        return Err(SpecError::BadField {
            path: at.0,
            value: name.to_owned(),
            reason: "a profile unused by live decisions",
        });
    }
    if let Some(&stage) = cx.scoped_stages.get(name) {
        deferred.note(at.0, stage);
    }
    Ok(position)
}

/// The binding's shadow profiles by position: a single's, or the prefill
/// side's of a pair, then the decode side's of a pair. A pair names its
/// shadows by side, and a side the map leaves out is not shadowed. The
/// encode side takes none.
fn check_shadow(
    shadow: &ShadowSpec,
    plan: PlanName,
    cx: &BindingContext<'_>,
    at: &At,
    deferred: &mut Deferred,
) -> Result<[Option<u16>; 2], SpecError> {
    let shadow_at = at.sub("shadow");
    match shadow {
        ShadowSpec::Profile(name) if plan == PlanName::Single => {
            let single = resolve_shadow_profile(name, shadow_at, cx, deferred)?;
            Ok([Some(single), None])
        }
        ShadowSpec::Profile(name) => {
            Err(at.bad("shadow", name, "a map of shadow profiles by side"))
        }
        // A binding of one profile names its shadow by a profile's
        // name, and a map that names no side names no shadow: either
        // loads as a document that shadows nothing, where its writer
        // meant one.
        ShadowSpec::Sides(_) if plan == PlanName::Single => {
            Err(at.bad("shadow", "a map", "one shadow profile's name"))
        }
        ShadowSpec::Sides(sides) if sides.is_empty() => {
            let sides = "a profile on the prefill or the decode side of a pair";
            Err(at.bad("shadow", "{}", sides))
        }
        ShadowSpec::Sides(sides) => {
            let mut by_side = [None; 2];
            for (side, name) in sides {
                let place = match side.as_str() {
                    "prefill" => 0,
                    "decode" => 1,
                    _ => {
                        let sides = "a profile on the prefill or the decode side of a pair";
                        return Err(shadow_at.bad(side, name, sides));
                    }
                };
                let profile = resolve_shadow_profile(name, shadow_at.sub(side), cx, deferred)?;
                by_side[place] = Some(profile);
            }
            Ok(by_side)
        }
    }
}

/// The binding's rules around the decision, in shape; the ones this build
/// does not evaluate are noted as deferred. Returns the shadows' positions
/// ([`check_shadow`]).
fn check_binding_rules(
    b: &BindingSpec,
    cx: &BindingContext<'_>,
    at: &At,
    deferred: &mut Deferred,
) -> Result<[Option<u16>; 2], SpecError> {
    if let Some(overload) = &b.overload {
        check_overload(overload, &at.sub("overload"))?;
    }
    if let Some(block) = &b.routing_key {
        check_routing_key(block, cx, &at.sub("routing_key"))?;
    }
    if let Some(sessions) = &b.sessions {
        let at = at.sub("sessions");
        check_sessions(sessions, b.routing_key.is_some(), cx, &at)?;
        if sessions.issue_session_id {
            deferred.note(at.field("issue_session_id"), "issuing a session id");
        }
    }
    let shadow = match &b.shadow {
        Some(shadow) => check_shadow(shadow, b.plan.unwrap_or_default(), cx, at, deferred)?,
        None => [None; 2],
    };
    if let Some(canary) = &b.canary {
        check_canary(canary, cx, &at.sub("canary"))?;
        deferred.note(at.field("canary"), "the canary binding");
    }
    if b.profile_override == Override::Allowed {
        deferred.note(at.field("override"), "the profile override header");
    }
    if b.admission.as_ref().is_some_and(|a| a.enabled) {
        return Err(SpecError::Reserved {
            path: at.sub("admission").field("enabled"),
            reason: "admission is reserved in v1 and stays off",
        });
    }
    Ok(shadow)
}

/// The binding's plan and sides, resolved to profile positions.
fn resolve_plan(
    b: &BindingSpec,
    cx: &BindingContext<'_>,
    [shadow, decode_shadow]: [Option<u16>; 2],
) -> Result<Option<Binding>, SpecError> {
    let index = cx.index;
    let plan = b.plan.unwrap_or_default();
    for key in b.profiles.keys() {
        if !SIDES.contains(&key.as_str()) {
            return Err(SpecError::UnknownSide {
                index,
                side: key.clone(),
            });
        }
        if !plan.sides().contains(&key.as_str()) {
            return Err(SpecError::SideNotInPlan {
                index,
                plan: plan.name(),
                side: key.clone(),
            });
        }
    }
    let resolve = |name: &str| {
        cx.profiles
            .get(name)
            .copied()
            .ok_or_else(|| SpecError::UnknownProfile {
                index,
                name: name.to_string(),
            })
    };
    let side = |name: &str| b.profiles.get(name).map(String::as_str);
    let binding = |plan: Plan, profile: u16, decode_profile: Option<u16>| Binding {
        model: b.matcher.model.clone(),
        phases: b.matcher.phase.clone(),
        plan,
        profile,
        decode_profile,
        shadow,
        decode_shadow,
        sessions: None,
    };
    if plan == PlanName::Single {
        if b.profile.is_some() && !b.profiles.is_empty() {
            return Err(SpecError::ProfileTwice { index });
        }
        let name = b
            .profile
            .as_deref()
            .or_else(|| side("single"))
            .ok_or(SpecError::NoProfile { index })?;
        return Ok(Some(binding(Plan::Single, resolve(name)?, None)));
    }
    if b.profile.is_some() {
        return Err(SpecError::ProfileUnderPlan {
            index,
            plan: plan.name(),
        });
    }
    let mut resolved = Vec::with_capacity(plan.sides().len());
    for name in plan.sides() {
        let profile = side(name).ok_or_else(|| SpecError::SideUnnamed {
            index,
            plan: plan.name(),
            side: name,
        })?;
        resolved.push(resolve(profile)?);
    }
    // A plan resolves from one side's lookup and serves every side, so its
    // phase list names them all or none.
    if !b.matcher.phase.is_empty() {
        for (phase, name) in plan.phases() {
            if !b.matcher.phase.contains(phase) {
                return Err(SpecError::PlanPhases {
                    index,
                    plan: plan.name(),
                    side: name,
                });
            }
        }
    }
    Ok(match (plan, resolved.as_slice()) {
        (PlanName::PdPair, &[prefill, decode]) => {
            Some(binding(Plan::PdPair, prefill, Some(decode)))
        }
        (PlanName::Epd, &[encode, prefill, decode]) => {
            Some(binding(Plan::Epd { encode }, prefill, Some(decode)))
        }
        _ => None,
    })
}

fn check_binding(
    b: &BindingSpec,
    cx: &BindingContext<'_>,
    needs: &mut Needs,
    deferred: &mut Deferred,
) -> Result<Option<Binding>, SpecError> {
    let at = At::new(format!("bindings[{}]", cx.index));
    check_matcher(&b.matcher, cx, &at, deferred)?;
    let shadow = check_binding_rules(b, cx, &at, deferred)?;
    let mut built = resolve_plan(b, cx, shadow)?;
    // A picker that leaves the placement to a sessions rule places a side
    // of a pair, under `pd-pair` and on the pair's two sides of `epd`: by
    // the rule, or by its assignment where the binding names none. No rule
    // places an item, so on the encode side nothing would place its
    // requests. The built picker answers, as it answers the engine.
    if b.plan.unwrap_or_default() == PlanName::Epd {
        let by_rule = |name: &String| cx.by_assignment.contains(&name.as_str());
        if b.profiles.get("encode").is_some_and(by_rule) {
            deferred.note(
                at.sub("profiles").field("encode"),
                "a picker that places by the sessions rule on the encode side",
            );
        }
    }
    if let (Some(binding), Some(sessions)) = (built.as_mut(), &b.sessions) {
        let rule = session_rule(sessions, cx.index, binding.plan, cx.lifecycle)?;
        // A picker that leaves the placement to the rule places nothing
        // itself: on a side the rule's `sides` leaves out no rule would
        // place its requests. The built picker answers, as under the plan
        // with an encode side above.
        for (decode, side) in [(false, "prefill"), (true, "decode")] {
            let by_rule = (b.profiles.get(side))
                .is_some_and(|name| cx.by_assignment.contains(&name.as_str()));
            if by_rule && !rule.sides.covers(decode) {
                deferred.note(at.sub("profiles").field(side), BY_RULE_ON_A_SIDE_LEFT_OUT);
            }
        }
        needs.active_sessions |= rule.assignment == Assignment::MinGroup;
        needs.session_in_flight |= rule.inflight_cap > 0;
        binding.sessions = Some(rule);
    }
    Ok(built)
}

/// [`UNTOLD_ON_FILE`] under the `epd` plan, whose pair's two sides the same
/// file keeps no word of: the refusal names the plan its rule stands under.
const UNTOLD_ON_FILE_EPD: &str =
    "`told` under the epd plan in this build: a file of leases keeps no side's word";

/// Why a sessions rule that keeps what it places from the profile is refused
/// under the `pd-pair` plan in the build of a host that makes some of the
/// lifecycle's calls only ([`session_rule`]), as the refusal prints it
/// behind "is not".
const UNTOLD_ON_FILE: &str =
    "`told` under the pd-pair plan in this build: a file of leases keeps no side's word";

/// What the build lacks where a rule's `sides` leaves out a side whose
/// picker leaves the placement to the rule. The binding holds one rule, and
/// no second one places that side.
const BY_RULE_ON_A_SIDE_LEFT_OUT: &str =
    "a picker that places by the sessions rule on a side the rule's `sessions.sides` leaves out";

/// The sides of a pair a sessions rule covers (`sides`): both where the
/// document names none. The field is the `pd-pair` plan's: a single request
/// has one side, and under `epd` a rule covers the pair's two sides in this
/// build. A list names `prefill`, `decode` or both, each once. The rule
/// covers no encode side: a gateway's sticky map keeps no pin of one.
fn session_sides(
    sessions: &SessionsSpec,
    index: usize,
    plan: Plan,
) -> Result<SessionSides, SpecError> {
    let Some(named) = &sessions.sides else {
        return Ok(SessionSides::Any);
    };
    let refused = |reason| SpecError::BadSessions {
        index,
        name: "sides",
        reason,
    };
    if matches!(plan, Plan::Single) {
        return Err(refused(
            "left out under the single plan: a single request has one side",
        ));
    }
    if matches!(plan, Plan::Epd { .. }) {
        return Err(refused(
            "left out under the epd plan in this build: a rule covers the pair's two sides there",
        ));
    }
    let mut covered = [false; 2];
    for side in named {
        let at = match side.as_str() {
            "prefill" => 0,
            "decode" => 1,
            _ => {
                return Err(refused(
                    "`prefill`, `decode` or both: a rule covers no other side",
                ))
            }
        };
        if std::mem::replace(&mut covered[at], true) {
            return Err(refused("a list that names each side once"));
        }
    }
    match covered {
        [true, true] => Ok(SessionSides::Any),
        [true, false] => Ok(SessionSides::Prefill),
        [false, true] => Ok(SessionSides::Decode),
        [false, false] => Err(refused(
            "a list of one side at least: leave it out for both",
        )),
    }
}

/// The sessions rule as the engine runs it: the schema's largest cap is no
/// cap, its largest idle time or sweep is for ever (a sweep for ever under a
/// finite idle time is refused, as nothing would forget the keys; an idle
/// time for ever sweeps nothing). The rule runs under the `single` plan, and
/// on the sides of a `pd-pair` plan it covers: each side of a pair's pick states
/// its own word for whether its stages hear of it. A rule that keeps what
/// it places from the profile is in the build of a host that makes the
/// lifecycle's calls (`lifecycle`), which tells each side by the pick in
/// its hand. It is refused by the field's name in the build of a host that
/// makes some only: that host tells a lease's words off a file, and its
/// file keeps no word of a pair's side.
/// Under a cap each side's records carry the count of its key's requests
/// in flight, as a single's do.
///
/// Under the `epd` plan the rule runs on the pair's two sides, each by its
/// own session, and places no item. An item goes by its side's policy.
/// The rule is in both builds where it tells its stages, and under a cap
/// each side's records carry the count as a pair's do. A rule that keeps
/// what it places from the profile is refused as under `pd-pair`: by the
/// field's name, in the build of a host that makes some calls only. Its
/// reason names the plan it stands under.
fn session_rule(
    sessions: &SessionsSpec,
    index: usize,
    plan: Plan,
    lifecycle: bool,
) -> Result<SessionRule, SpecError> {
    // The word a file of leases keeps for no side of a pair, in the build
    // of the host that tells a lease's words off that file.
    let untold = sessions.rule_placed != RulePlaced::Told && !lifecycle;
    let refused = match plan {
        Plan::PdPair if untold => Some(UNTOLD_ON_FILE),
        Plan::Epd { .. } if untold => Some(UNTOLD_ON_FILE_EPD),
        Plan::Single | Plan::PdPair | Plan::Epd { .. } => None,
    };
    if let Some(reason) = refused {
        return Err(SpecError::BadSessions {
            index,
            name: "rule_placed",
            reason,
        });
    }
    let forever = |secs: u32| secs == u32::MAX;
    if forever(sessions.sweep_s) && !forever(sessions.ttl_s) {
        return Err(SpecError::BadSessions {
            index,
            name: "sweep_s",
            reason: "finite while ttl_s is",
        });
    }
    Ok(SessionRule {
        assignment: sessions.assignment,
        inflight_cap: if sessions.inflight_cap == u16::MAX {
            0
        } else {
            sessions.inflight_cap
        },
        ttl_micros: (!forever(sessions.ttl_s)).then(|| u64::from(sessions.ttl_s) * 1_000_000),
        sweep_micros: if forever(sessions.ttl_s) {
            0
        } else {
            u64::from(sessions.sweep_s) * 1_000_000
        },
        recovery: sessions.recovery,
        rule_placed: sessions.rule_placed,
        sides: session_sides(sessions, index, plan)?,
    })
}

impl PolicySpec {
    /// Validate the document and build the engine; the needs summary comes
    /// with it. The built-in header deny-list applies; a host with a tenant
    /// header of its own uses [`PolicySpec::validate_with`].
    pub fn validate(&self) -> Result<Engine, SpecError> {
        self.validate_with(&[])
    }

    /// [`PolicySpec::validate`] with the host's own denied header names
    /// (the configured tenant header) added to the built-in deny-list.
    pub fn validate_with(&self, denied_headers: &[&str]) -> Result<Engine, SpecError> {
        self.validate_in(&STOCK, denied_headers)
    }

    /// [`PolicySpec::validate_with`] over the stages `catalog` holds: the
    /// stock build's and the ones a host added in code, each named in the
    /// document as a built-in is. Every stage name is resolved here, once;
    /// a factory builds its stage for this engine alone.
    ///
    /// Every field is checked in shape first; a field this build cannot
    /// honour (a stage, a plan, a matcher or a binding rule of a later step)
    /// is then refused by name as [`SpecError::NotInBuild`], and so is a
    /// stage that needs the lifecycle's calls
    /// ([`crate::stages::Lifecycle::needs_lifecycle`]) under the standalone
    /// controller, which makes the membership call alone.
    pub fn validate_in(
        &self,
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<Engine, SpecError> {
        let (engine, _, _) = self.built(catalog, denied_headers)?;
        Ok(engine)
    }

    /// [`PolicySpec::validate_in`], and beside the engine how deep its
    /// stages read a request's block hashes and the document's signals,
    /// resolved.
    pub(in crate::document) fn built(
        &self,
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<(Engine, HashDepths, Arc<[SignalColumn]>), SpecError> {
        if let Some(kind) = &self.kind {
            return Err(SpecError::Kind(kind.clone()));
        }
        if self.api_version != API_VERSION {
            return Err(SpecError::ApiVersion(self.api_version.clone()));
        }
        if self.profiles.is_empty() {
            return Err(SpecError::Empty);
        }
        if self.profiles.len() > usize::from(u16::MAX) {
            return Err(SpecError::TooManyProfiles {
                count: self.profiles.len(),
                max: u16::MAX,
            });
        }
        if self.bindings.is_empty() && self.profiles.len() > 1 {
            return Err(SpecError::NoBindings {
                profiles: self.profiles.len(),
            });
        }
        check_view(&self.view, denied_headers)?;
        check_index(&self.index, &self.view)?;
        check_signals(&self.signals)?;
        let signal_columns = resolve_signals(&self.signals);
        let mut deferred = Deferred::default();
        let mut depth = HashDepths::default();
        // What no stage declares: the sessions rules' counts and the view's
        // headers. The engine folds every profile's stages in.
        let mut needs = Needs::default();
        // The build's age limit of load reports and its stage table's
        // number, stated once: the engine takes both where it is made.
        let mut build = Engine::builder(self.view.max_report_age_ms, catalog.own);
        let has_shadows = self.bindings.iter().any(|binding| binding.shadow.is_some());
        let mut scoped_stages = BTreeMap::new();
        // Noted by name: the built vector lines up with the document's
        // positions only once nothing is deferred.
        let mut by_assignment = Vec::new();
        // The built profiles the plain ring alone places by a request's
        // session key, by their place among the built ones.
        let mut keyed = Vec::new();
        // A registered stage reads a report through the view's reading:
        // the limit the built-in scorers are wrapped with, and the one
        // the build's engine checks a report's age by. It resolves a
        // signal's column from the document's, resolved once.
        let from = StageInputs {
            catalog,
            reports: LoadReportReader::new(build.max_report_age_ms()),
            signals: &signal_columns,
        };
        for (name, profile) in &self.profiles {
            let built = check_profile(
                name,
                profile,
                &self.view,
                from,
                &mut deferred,
                &mut depth,
                |settings| build.sent_under(settings),
            )?;
            if let Some((built, by_key, asked)) = built {
                if has_shadows {
                    if let Some(stage) = scoped_stage(&built) {
                        scoped_stages.insert(name.as_str(), stage);
                    }
                }
                if built.picker.by_assignment() {
                    by_assignment.push(name.as_str());
                }
                if by_key {
                    keyed.push(build.len());
                }
                build.push(built, asked);
            }
        }
        // A binding addresses its profile by position in the built vector,
        // which holds every document profile once the deferred check below
        // has passed; the bound above keeps the position in a u16.
        let mut positions: BTreeMap<&str, u16> = BTreeMap::new();
        for (i, name) in self.profiles.keys().enumerate() {
            let position = u16::try_from(i).map_err(|_| SpecError::TooManyProfiles {
                count: self.profiles.len(),
                max: u16::MAX,
            })?;
            positions.insert(name.as_str(), position);
        }
        let mut bindings = Vec::with_capacity(self.bindings.len());
        let mut live_profiles = BTreeSet::new();
        if has_shadows {
            live_profiles.extend(self.bindings.iter().flat_map(|binding| {
                binding
                    .profile
                    .iter()
                    .chain(binding.profiles.values())
                    .chain(binding.canary.iter().map(|canary| &canary.profile))
                    .map(String::as_str)
            }));
        }
        // The built bindings whose document sheds a pool the overload veto
        // empties, by their place in `bindings`.
        let mut sheds = Vec::new();
        for (index, binding) in self.bindings.iter().enumerate() {
            let cx = BindingContext {
                index,
                view: &self.view,
                denied: denied_headers,
                profiles: &positions,
                live_profiles: &live_profiles,
                scoped_stages: &scoped_stages,
                by_assignment: &by_assignment,
                lifecycle: catalog.makes_lifecycle_calls(),
            };
            if let Some(built) = check_binding(binding, &cx, &mut needs, &mut deferred)? {
                if binding
                    .overload
                    .as_ref()
                    .is_some_and(|overload| overload.shed_when_all_vetoed)
                {
                    sheds.push(bindings.len());
                }
                bindings.push(built);
            }
        }
        if let Some(missing) = deferred.0 {
            return Err(missing);
        }
        debug_assert_eq!(
            build.len(),
            self.profiles.len(),
            "every profile is built once the deferred check has passed"
        );
        // A binding's routing key sources are kept by its place in the
        // document and found by its place in the engine
        // ([`AppliedSpec::key_sources_for`]): the two are one place.
        debug_assert_eq!(
            bindings.len(),
            self.bindings.len(),
            "every binding is built once the deferred check has passed"
        );
        if self.bindings.is_empty() {
            // One profile, refused above when there are more.
            bindings.push(Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Single,
                profile: 0,
                decode_profile: None,
                shadow: None,
                decode_shadow: None,
                sessions: None,
            });
        }
        needs.headers |= !self.view.headers.is_empty();
        let mut engine = Engine::built(build, bindings, self.generation, needs);
        engine.set_explain_sample(self.explain.sample);
        for at in sheds {
            engine.shed_when_all_vetoed(at);
        }
        for at in keyed {
            engine.places_by_key(at);
        }
        Ok((engine, depth, signal_columns))
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/document/spec/validate/tests.rs"]
mod tests;
