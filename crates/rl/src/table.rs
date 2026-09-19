//! The RL side table: per-engine weight version and control state, keyed by
//! engine base URL, plus the per-model fleet maximum the pre-filter compares
//! against. Labels are never written; this table is the mutable state.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex, MutexGuard, PoisonError,
};

use dashmap::DashMap;
use openai_protocol::rl::{RlControlState, RlVersionSource};

use crate::{metrics, policy::VersionPolicy, version::Version};

/// Whether an engine can serve requests, as last observed through SMG.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ControlState {
    #[default]
    Active,
    Paused,
    Asleep,
}

impl ControlState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Paused => "paused",
            Self::Asleep => "asleep",
        }
    }

    fn gauge_value(self) -> f64 {
        match self {
            Self::Active => 0.0,
            Self::Paused => 1.0,
            Self::Asleep => 2.0,
        }
    }
}

/// How SMG learned an engine's version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VersionSource {
    Registration,
    Passthrough,
    Api,
}

impl VersionSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Registration => "registration",
            Self::Passthrough => "passthrough",
            Self::Api => "api",
        }
    }
}

impl From<ControlState> for RlControlState {
    fn from(c: ControlState) -> Self {
        match c {
            ControlState::Active => Self::Active,
            ControlState::Paused => Self::Paused,
            ControlState::Asleep => Self::Asleep,
        }
    }
}

impl From<RlControlState> for ControlState {
    fn from(c: RlControlState) -> Self {
        match c {
            RlControlState::Active => Self::Active,
            RlControlState::Paused => Self::Paused,
            RlControlState::Asleep => Self::Asleep,
        }
    }
}

impl From<VersionSource> for RlVersionSource {
    fn from(s: VersionSource) -> Self {
        match s {
            VersionSource::Registration => Self::Registration,
            VersionSource::Passthrough => Self::Passthrough,
            VersionSource::Api => Self::Api,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EngineState {
    /// `Worker::model_id()` at seed time; the fleet maximum is per model.
    pub model: Arc<str>,
    pub version: Option<Version>,
    pub version_source: Option<VersionSource>,
    pub control: ControlState,
}

/// Told when an engine's version changed so the gateway can forget what the
/// engine's cache held (coupling surface (a)).
pub trait VersionEvictionSink: Send + Sync {
    fn on_version_changed(&self, model_id: &str, base_url: &str);
}

/// A sink that does nothing; for crate-only construction and tests.
pub struct NoopEvictionSink;

impl VersionEvictionSink for NoopEvictionSink {
    fn on_version_changed(&self, _model_id: &str, _base_url: &str) {}
}

/// Strip the `@<rank>` suffix a DP-aware worker URL carries, giving the
/// engine base URL the table is keyed by.
pub fn base_url_of(worker_url: &str) -> &str {
    match worker_url.rsplit_once('@') {
        Some((base, rank)) if !rank.is_empty() && rank.bytes().all(|b| b.is_ascii_digit()) => base,
        _ => worker_url,
    }
}

/// A freshly observed engine: unversioned and active until a write says
/// otherwise.
fn fresh(model: &str) -> EngineState {
    EngineState {
        model: Arc::from(model),
        version: None,
        version_source: None,
        control: ControlState::Active,
    }
}

/// Why a candidate was dropped; the `reason` label of
/// `smg_rl_candidates_filtered_total`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterReason {
    Paused,
    Asleep,
    Stale,
    Unversioned,
}

impl FilterReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Paused => "paused",
            Self::Asleep => "asleep",
            Self::Stale => "stale",
            Self::Unversioned => "unversioned",
        }
    }
}

pub struct RlTable {
    entries: DashMap<Arc<str>, EngineState>,
    fleet_max: DashMap<Arc<str>, Version>,
    /// Entries whose control state is not `Active`; lets `any` skip the map.
    inactive: AtomicUsize,
    sink: Arc<dyn VersionEvictionSink>,
    /// Serializes every write. `recompute_fleet_max` is a scan-then-store
    /// across two `DashMap`s with no synchronization of its own, so two
    /// concurrent writers for the same model could otherwise leave
    /// `fleet_max` durably stale. Writes are control-path and rare (a
    /// refit, a pause, a registration), so one uncontended lock is the
    /// whole cost; reads never take it.
    writes: Mutex<()>,
}

impl RlTable {
    pub fn new(sink: Arc<dyn VersionEvictionSink>) -> Self {
        Self {
            entries: DashMap::new(),
            fleet_max: DashMap::new(),
            inactive: AtomicUsize::new(0),
            sink,
            writes: Mutex::new(()),
        }
    }

    fn lock_writes(&self) -> MutexGuard<'_, ()> {
        self.writes.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Insert an entry from a registration label unless one exists. Returns
    /// whether an entry was inserted.
    pub fn seed(&self, base_url: &str, model: &str, label: Option<&str>) -> bool {
        let _guard = self.lock_writes();
        if self.entries.contains_key(base_url) {
            return false;
        }
        let version = Version::from_label(label);
        self.entries.insert(
            Arc::from(base_url),
            EngineState {
                version_source: version.as_ref().map(|_| VersionSource::Registration),
                version,
                ..fresh(model)
            },
        );
        self.recompute_fleet_max(model);
        true
    }

    /// Overwrite the version from a (changed) registration label, keeping
    /// the control state.
    pub fn reseed(&self, base_url: &str, model: &str, label: Option<&str>) {
        let _guard = self.lock_writes();
        let version = Version::from_label(label);
        let (changed, model) = {
            let mut entry = self
                .entries
                .entry(Arc::from(base_url))
                .or_insert_with(|| fresh(model));
            let changed = entry.version != version;
            entry.version_source = version.as_ref().map(|_| VersionSource::Registration);
            entry.version.clone_from(&version);
            (changed, Arc::clone(&entry.model))
        };
        self.recompute_fleet_max(&model);
        if changed {
            self.after_version_change(&model, base_url, version.as_ref());
        }
    }

    pub fn remove(&self, base_url: &str) {
        let _guard = self.lock_writes();
        self.remove_locked(base_url);
    }

    /// `remove`'s body, for callers that already hold `writes` (namely
    /// `retain`, which must not re-lock the mutex it is already holding).
    fn remove_locked(&self, base_url: &str) {
        let Some((_, state)) = self.entries.remove(base_url) else {
            return;
        };
        self.recompute_fleet_max(&state.model);
        self.recompute_inactive();
    }

    /// Drop every entry whose base URL fails `keep` (resync after a lagged
    /// event stream).
    pub fn retain(&self, keep: impl Fn(&str) -> bool) {
        let _guard = self.lock_writes();
        let dropped: Vec<Arc<str>> = self
            .entries
            .iter()
            .filter(|e| !keep(e.key()))
            .map(|e| Arc::clone(e.key()))
            .collect();
        for base_url in dropped {
            self.remove_locked(&base_url);
        }
    }

    /// Record a version. Returns whether it differs from what was held.
    pub fn set_version(
        &self,
        base_url: &str,
        model: &str,
        version: Version,
        source: VersionSource,
    ) -> bool {
        let _guard = self.lock_writes();
        let (changed, model) = {
            let mut entry = self
                .entries
                .entry(Arc::from(base_url))
                .or_insert_with(|| fresh(model));
            let changed = entry.version.as_ref() != Some(&version);
            entry.version = Some(version.clone());
            entry.version_source = Some(source);
            (changed, Arc::clone(&entry.model))
        };
        self.recompute_fleet_max(&model);
        if changed {
            self.after_version_change(&model, base_url, Some(&version));
        }
        changed
    }

    /// Record a control state. Returns whether it changed.
    pub fn set_control(&self, base_url: &str, model: &str, control: ControlState) -> bool {
        let _guard = self.lock_writes();
        let changed = {
            let mut entry = self
                .entries
                .entry(Arc::from(base_url))
                .or_insert_with(|| fresh(model));
            let changed = entry.control != control;
            entry.control = control;
            changed
        };
        if changed {
            self.recompute_inactive();
            metrics::set_worker_control_state(base_url, control.gauge_value());
        }
        changed
    }

    pub fn get(&self, base_url: &str) -> Option<EngineState> {
        self.entries.get(base_url).map(|e| e.value().clone())
    }

    pub fn version_of(&self, base_url: &str) -> Option<Version> {
        self.entries.get(base_url).and_then(|e| e.version.clone())
    }

    pub fn control_of(&self, base_url: &str) -> ControlState {
        self.entries
            .get(base_url)
            .map_or(ControlState::Active, |e| e.control)
    }

    pub fn fleet_max(&self, model: &str) -> Option<Version> {
        self.fleet_max.get(model).map(|v| v.value().clone())
    }

    pub fn inactive_count(&self) -> usize {
        self.inactive.load(Ordering::Relaxed)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn after_version_change(&self, model: &str, base_url: &str, version: Option<&Version>) {
        if let Some(numeric) = version.and_then(Version::numeric) {
            metrics::set_worker_weight_version(base_url, numeric);
        }
        self.sink.on_version_changed(model, base_url);
    }

    /// Scan the model's entries; fleets are small and this runs on writes
    /// only. Always called with `writes` held: the scan-then-store here is
    /// not atomic on its own.
    ///
    /// Prefers the numeric maximum when the fleet has any numeric version:
    /// `Version`'s `Ord` falls back to a lexical compare across a
    /// numeric/text pair, so a stray checkpoint label like `ckpt-7` would
    /// otherwise outrank every numbered release just because `'c' > '1'`
    /// byte-wise, corrupting the staleness baseline for the whole model.
    /// Only when every version for the model is text does that lexical
    /// order stand, matching `Version`'s own documented text-vs-text
    /// ordering.
    fn recompute_fleet_max(&self, model: &str) {
        let versions_of_model = || {
            self.entries
                .iter()
                .filter(|e| &*e.model == model)
                .filter_map(|e| e.version.clone())
        };
        let max = versions_of_model()
            .filter(|v| v.numeric().is_some())
            .max()
            .or_else(|| versions_of_model().max());
        match max {
            Some(v) => {
                self.fleet_max.insert(Arc::from(model), v);
            }
            None => {
                self.fleet_max.remove(model);
            }
        }
    }

    fn recompute_inactive(&self) {
        let n = self
            .entries
            .iter()
            .filter(|e| e.control != ControlState::Active)
            .count();
        self.inactive.store(n, Ordering::Relaxed);
    }

    /// Indices of `candidates` (as `(model_id, base_url)` pairs) that stay
    /// eligible under `policy`; `None` when every candidate is eligible, so
    /// the common case allocates nothing. Paused and asleep engines are
    /// dropped under every policy.
    pub fn eligible<'a, I>(&self, candidates: I, policy: &VersionPolicy) -> Option<Vec<usize>>
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
        I::IntoIter: ExactSizeIterator,
    {
        if *policy == VersionPolicy::Any && self.inactive.load(Ordering::Relaxed) == 0 {
            return None;
        }
        let iter = candidates.into_iter();
        let total = iter.len();
        let mut keep = Vec::with_capacity(total);
        let mut cached_max: Option<(&'a str, Option<Version>)> = None;
        for (idx, (model, base_url)) in iter.enumerate() {
            let (control, version) = match self.entries.get(base_url) {
                Some(e) => (e.control, e.version.clone()),
                None => (ControlState::Active, None),
            };
            let verdict = match control {
                ControlState::Paused => Err(FilterReason::Paused),
                ControlState::Asleep => Err(FilterReason::Asleep),
                ControlState::Active if *policy == VersionPolicy::Any => Ok(()),
                ControlState::Active => {
                    let fleet_max = match &cached_max {
                        Some((m, max)) if *m == model => max.clone(),
                        _ => {
                            let max = self.fleet_max(model);
                            cached_max = Some((model, max.clone()));
                            max
                        }
                    };
                    Self::judge(policy, version.as_ref(), fleet_max.as_ref())
                }
            };
            match verdict {
                Ok(()) => keep.push(idx),
                Err(reason) => metrics::record_candidate_filtered(reason.as_str()),
            }
        }
        // An empty candidate list is only "unroutable" when there was
        // something to drop: a caller with no candidates at all already had
        // nothing to route, and counting it here would blame the filter.
        if keep.is_empty() && total > 0 {
            metrics::record_request_unroutable();
        }
        if keep.len() == total {
            None
        } else {
            Some(keep)
        }
    }

    fn judge(
        policy: &VersionPolicy,
        version: Option<&Version>,
        fleet_max: Option<&Version>,
    ) -> Result<(), FilterReason> {
        match policy {
            VersionPolicy::Any => Ok(()),
            VersionPolicy::LatestOnly => match (version, fleet_max) {
                (_, None) => Ok(()),
                (None, Some(_)) => Err(FilterReason::Unversioned),
                (Some(v), Some(max)) if v == max => Ok(()),
                (Some(_), Some(_)) => Err(FilterReason::Stale),
            },
            // Staleness is a numeric distance, so a version either side of
            // the comparison that is not a `u64` has no distance to measure
            // and counts as stale. That is the whole rule: the candidate is
            // dropped under `FilterReason::Stale` like any other, with no
            // separate log or metric for the non-numeric case.
            VersionPolicy::MaxStaleness(k) => match (version, fleet_max) {
                (_, None) => Ok(()),
                (None, Some(_)) => Err(FilterReason::Unversioned),
                (Some(v), Some(max)) => match (v.numeric(), max.numeric()) {
                    (Some(v), Some(max)) if max.saturating_sub(v) <= *k => Ok(()),
                    _ => Err(FilterReason::Stale),
                },
            },
            VersionPolicy::MinVersion(floor) => match version {
                None => Err(FilterReason::Unversioned),
                Some(v) if v >= floor => Ok(()),
                Some(_) => Err(FilterReason::Stale),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    struct RecordingSink(Mutex<Vec<(String, String)>>);

    impl VersionEvictionSink for RecordingSink {
        fn on_version_changed(&self, model_id: &str, base_url: &str) {
            self.0
                .lock()
                .unwrap()
                .push((model_id.to_string(), base_url.to_string()));
        }
    }

    fn table() -> (RlTable, Arc<RecordingSink>) {
        let sink = Arc::new(RecordingSink(Mutex::new(Vec::new())));
        (RlTable::new(sink.clone()), sink)
    }

    #[test]
    fn base_url_strips_only_a_numeric_rank_suffix() {
        assert_eq!(base_url_of("http://a:1@3"), "http://a:1");
        assert_eq!(base_url_of("http://a:1"), "http://a:1");
        assert_eq!(base_url_of("http://u:p@h:1"), "http://u:p@h:1");
        assert_eq!(base_url_of("http://a:1@"), "http://a:1@");
    }

    #[test]
    fn seed_inserts_once_and_treats_default_as_unversioned() {
        let (t, sink) = table();
        assert!(t.seed("http://a:1", "m", Some("default")));
        assert!(
            !t.seed("http://a:1", "m", Some("9")),
            "second seed is a no-op"
        );
        let s = t.get("http://a:1").unwrap();
        assert_eq!(s.version, None);
        assert_eq!(s.version_source, None);
        assert_eq!(s.control, ControlState::Active);
        assert_eq!(t.fleet_max("m"), None);
        assert!(t.seed("http://b:1", "m", Some("3")));
        assert_eq!(
            t.get("http://b:1").unwrap().version_source,
            Some(VersionSource::Registration)
        );
        assert_eq!(t.fleet_max("m").unwrap().as_str(), "3");
        assert!(sink.0.lock().unwrap().is_empty(), "seeding never evicts");
    }

    #[test]
    fn set_version_maintains_fleet_max_and_calls_the_sink_on_change() {
        let (t, sink) = table();
        t.seed("http://a:1", "m", None);
        t.seed("http://b:1", "m", None);
        t.seed("http://c:1", "other", None);
        assert!(t.set_version("http://a:1", "m", Version::parse("1"), VersionSource::Api));
        assert!(t.set_version(
            "http://b:1",
            "m",
            Version::parse("2"),
            VersionSource::Passthrough
        ));
        assert_eq!(t.fleet_max("m").unwrap().as_str(), "2");
        assert_eq!(t.fleet_max("other"), None);
        assert!(
            !t.set_version("http://b:1", "m", Version::parse("2"), VersionSource::Api),
            "same version: unchanged"
        );
        assert_eq!(
            t.get("http://b:1").unwrap().version_source,
            Some(VersionSource::Api)
        );
        assert_eq!(
            *sink.0.lock().unwrap(),
            vec![
                ("m".to_string(), "http://a:1".to_string()),
                ("m".to_string(), "http://b:1".to_string())
            ]
        );
        t.remove("http://b:1");
        assert_eq!(t.fleet_max("m").unwrap().as_str(), "1");
        t.remove("http://a:1");
        assert_eq!(t.fleet_max("m"), None);
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn set_version_on_an_unknown_engine_creates_the_entry() {
        let (t, _) = table();
        assert!(t.set_version("http://x:1", "m", Version::parse("5"), VersionSource::Api));
        assert_eq!(t.get("http://x:1").unwrap().model.as_ref(), "m");
        assert_eq!(t.version_of("http://x:1").unwrap().as_str(), "5");
    }

    #[test]
    fn reseed_overwrites_from_the_label() {
        let (t, sink) = table();
        t.set_version("http://a:1", "m", Version::parse("7"), VersionSource::Api);
        t.set_control("http://a:1", "m", ControlState::Paused);
        t.reseed("http://a:1", "m", Some("8"));
        let s = t.get("http://a:1").unwrap();
        assert_eq!(s.version.unwrap().as_str(), "8");
        assert_eq!(s.version_source, Some(VersionSource::Registration));
        assert_eq!(
            s.control,
            ControlState::Paused,
            "reseed keeps control state"
        );
        assert_eq!(sink.0.lock().unwrap().len(), 2);
    }

    #[test]
    fn control_state_tracks_the_inactive_count() {
        let (t, _) = table();
        t.seed("http://a:1", "m", None);
        t.seed("http://b:1", "m", None);
        assert_eq!(t.inactive_count(), 0);
        assert!(t.set_control("http://a:1", "m", ControlState::Paused));
        assert!(!t.set_control("http://a:1", "m", ControlState::Paused));
        assert!(t.set_control("http://b:1", "m", ControlState::Asleep));
        assert_eq!(t.inactive_count(), 2);
        assert_eq!(t.control_of("http://a:1"), ControlState::Paused);
        assert_eq!(t.control_of("http://zz:1"), ControlState::Active);
        t.set_control("http://a:1", "m", ControlState::Active);
        assert_eq!(t.inactive_count(), 1);
        t.remove("http://b:1");
        assert_eq!(t.inactive_count(), 0);
    }

    #[test]
    fn retain_drops_entries_outside_the_registry() {
        let (t, _) = table();
        t.seed("http://a:1", "m", Some("1"));
        t.seed("http://b:1", "m", Some("2"));
        t.set_control("http://b:1", "m", ControlState::Asleep);
        t.retain(|url| url == "http://a:1");
        assert_eq!(t.len(), 1);
        assert_eq!(t.fleet_max("m").unwrap().as_str(), "1");
        assert_eq!(t.inactive_count(), 0);
    }

    #[test]
    fn reseed_recomputes_using_the_entrys_stored_model_not_the_callers_argument() {
        let (t, sink) = table();
        t.seed("http://a:1", "real-model", Some("5"));
        assert_eq!(t.fleet_max("real-model").unwrap().as_str(), "5");

        // A caller passes a model that doesn't match what the entry was
        // seeded under; the stored model must win for both the fleet-max
        // recompute and the eviction-sink call.
        t.reseed("http://a:1", "wrong-model", Some("9"));

        assert_eq!(t.get("http://a:1").unwrap().model.as_ref(), "real-model");
        assert_eq!(t.fleet_max("real-model").unwrap().as_str(), "9");
        assert_eq!(t.fleet_max("wrong-model"), None);
        assert_eq!(
            sink.0.lock().unwrap().last(),
            Some(&("real-model".to_string(), "http://a:1".to_string()))
        );
    }

    fn candidates<'a>(urls: &'a [&'a str]) -> Vec<(&'a str, &'a str)> {
        urls.iter().map(|u| ("m", *u)).collect()
    }

    #[test]
    fn any_with_every_engine_active_is_the_allocation_free_path() {
        let (t, _) = table();
        t.seed("http://a:1", "m", Some("1"));
        t.seed("http://b:1", "m", Some("2"));
        assert_eq!(
            t.eligible(
                candidates(&["http://a:1", "http://b:1"]),
                &VersionPolicy::Any
            ),
            None
        );
        // Unknown engines are eligible too.
        assert_eq!(
            t.eligible(candidates(&["http://zz:1"]), &VersionPolicy::Any),
            None
        );
    }

    #[test]
    fn paused_and_asleep_engines_are_dropped_under_every_policy() {
        let (t, _) = table();
        t.seed("http://a:1", "m", Some("2"));
        t.seed("http://b:1", "m", Some("2"));
        t.seed("http://c:1", "m", Some("2"));
        t.set_control("http://a:1", "m", ControlState::Paused);
        t.set_control("http://c:1", "m", ControlState::Asleep);
        let c = candidates(&["http://a:1", "http://b:1", "http://c:1"]);
        assert_eq!(t.eligible(c.clone(), &VersionPolicy::Any), Some(vec![1]));
        assert_eq!(
            t.eligible(c.clone(), &VersionPolicy::LatestOnly),
            Some(vec![1])
        );
        t.set_control("http://b:1", "m", ControlState::Paused);
        assert_eq!(t.eligible(c, &VersionPolicy::Any), Some(vec![]));
    }

    #[test]
    fn latest_only_keeps_the_fleet_max_and_everyone_when_nobody_is_versioned() {
        let (t, _) = table();
        t.seed("http://a:1", "m", None);
        t.seed("http://b:1", "m", None);
        let c = candidates(&["http://a:1", "http://b:1"]);
        assert_eq!(
            t.eligible(c.clone(), &VersionPolicy::LatestOnly),
            None,
            "nothing is stale yet"
        );
        t.set_version("http://a:1", "m", Version::parse("1"), VersionSource::Api);
        assert_eq!(
            t.eligible(c.clone(), &VersionPolicy::LatestOnly),
            Some(vec![0]),
            "unversioned b is behind"
        );
        t.set_version("http://b:1", "m", Version::parse("2"), VersionSource::Api);
        assert_eq!(
            t.eligible(c.clone(), &VersionPolicy::LatestOnly),
            Some(vec![1]),
            "a is stale"
        );
        t.set_version("http://a:1", "m", Version::parse("2"), VersionSource::Api);
        assert_eq!(t.eligible(c, &VersionPolicy::LatestOnly), None);
    }

    #[test]
    fn max_staleness_and_min_version_evaluate_numerically() {
        let (t, _) = table();
        for (url, v) in [
            ("http://a:1", "10"),
            ("http://b:1", "9"),
            ("http://c:1", "7"),
        ] {
            t.set_version(url, "m", Version::parse(v), VersionSource::Api);
        }
        t.seed("http://d:1", "m", None);
        let c = candidates(&["http://a:1", "http://b:1", "http://c:1", "http://d:1"]);
        assert_eq!(
            t.eligible(c.clone(), &VersionPolicy::MaxStaleness(1)),
            Some(vec![0, 1])
        );
        assert_eq!(
            t.eligible(c.clone(), &VersionPolicy::MaxStaleness(3)),
            Some(vec![0, 1, 2])
        );
        assert_eq!(
            t.eligible(c.clone(), &VersionPolicy::MinVersion(Version::parse("9"))),
            Some(vec![0, 1])
        );
        assert_eq!(
            t.eligible(c.clone(), &VersionPolicy::MinVersion(Version::parse("7"))),
            Some(vec![0, 1, 2])
        );
        // A text version cannot satisfy a numeric staleness bound.
        t.set_version(
            "http://c:1",
            "m",
            Version::parse("ckpt-7"),
            VersionSource::Api,
        );
        assert_eq!(
            t.eligible(c, &VersionPolicy::MaxStaleness(100)),
            Some(vec![0, 1])
        );
    }

    #[test]
    fn fleet_max_is_per_model() {
        let (t, _) = table();
        t.set_version("http://a:1", "m", Version::parse("5"), VersionSource::Api);
        t.set_version("http://b:1", "n", Version::parse("1"), VersionSource::Api);
        let c = vec![("m", "http://a:1"), ("n", "http://b:1")];
        assert_eq!(t.eligible(c, &VersionPolicy::LatestOnly), None);
    }

    #[test]
    fn concurrent_writers_leave_fleet_max_consistent() {
        let (t, _) = table();
        let t = Arc::new(t);
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let t = Arc::clone(&t);
                std::thread::spawn(move || {
                    let base_url = format!("http://w{i}:1");
                    for v in 1..=200 {
                        t.set_version(
                            &base_url,
                            "m",
                            Version::parse(&v.to_string()),
                            VersionSource::Api,
                        );
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(t.fleet_max("m").unwrap().as_str(), "200");
        assert_eq!(t.len(), 8);
    }
}
