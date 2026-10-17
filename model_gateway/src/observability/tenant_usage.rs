//! Per-tenant usage accounting: requests, prompt and completion tokens and
//! the time they took, by tenant and model, behind
//! `tenant_usage_accounting_enabled` (off by default).
//!
//! The gateway's request, duration and token families carry no tenant label:
//! a tenant label on them multiplies every series by the number of tenants,
//! and one unbounded dimension on the hot families is what a cardinality
//! blow-up is made of. The per-tenant counts live in a table of their own
//! instead, bounded by `tenant_usage_accounting_max_tenants`: the first N
//! distinct tenant keys seen get a row each, every later tenant is booked
//! under `tenant="other"`, so a stream of one-off callers (an IP-keyed
//! anonymous deployment) can grow neither the table nor the scrape. Rows
//! under `other` are the sign that the cap is too small for the fleet.
//!
//! The table is process-wide, installed once at start-up like the Prometheus
//! recorder, and read two ways: rendered into the `/metrics` scrape as its
//! own families (`smg_tenant_usage_*`), and served as a JSON snapshot by
//! `GET /v1/usage/tenants` under control-plane auth.
//!
//! The request path does one lookup by tenant key, one by model and four
//! atomic adds; no label is interned and no exporter series is touched per
//! request. The scrape walks the table, which is why the families are
//! rendered here rather than recorded through the exporter.
//!
//! A buffered response is booked when it is built. A streamed response is
//! booked by its [`StreamUsage`] guard: at its normal end with the usage
//! block's numbers, and when it ends any other way (the client went away,
//! the engine failed mid-stream) with what was counted up to that point.
//!
//! A request is one engine round trip: a Responses call whose tool loop runs
//! several iterations books each of them, buffered or streamed, so the token
//! sums match what the engine served and the `requests` count says how many
//! times it was asked.

use std::{
    collections::HashMap,
    fmt::Write as _,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, OnceLock,
    },
    time::{Duration, Instant},
};

use dashmap::{mapref::entry::Entry, DashMap};
use metrics_exporter_prometheus::formatting::{
    sanitize_label_value, write_help_line, write_type_line,
};
use parking_lot::RwLock;
use serde::Serialize;

/// The row every tenant past the tracked maximum is booked under.
pub const OTHER_TENANT: &str = "other";
/// The row of a request that reached the pipeline without route meta. The
/// serving paths always resolve one (the anonymous key at the least), so this
/// is the resolver's own catch-all, reused rather than a second one.
pub const UNRESOLVED_TENANT: &str = "anonymous";

const REQUESTS: &str = "smg_tenant_usage_requests_total";
const TOKENS: &str = "smg_tenant_usage_tokens_total";
const SECONDS: &str = "smg_tenant_usage_request_seconds_total";
const TRACKED: &str = "smg_tenant_usage_tracked_tenants";
const MAX: &str = "smg_tenant_usage_max_tenants";
const TOKEN_INPUT: &str = "input";
const TOKEN_OUTPUT: &str = "output";

static INSTALLED: OnceLock<Arc<TenantUsage>> = OnceLock::new();

/// Install the process-wide table. The first call wins, as the recorder
/// installation does; a gateway started with accounting off installs none.
pub fn install(table: Arc<TenantUsage>) {
    let _ = INSTALLED.set(table);
}

/// The installed table, when accounting is on.
pub fn installed() -> Option<&'static Arc<TenantUsage>> {
    INSTALLED.get()
}

/// Book one finished request to `tenant` and `model` when accounting is on:
/// one atomic load and a branch when it is off.
#[inline]
pub fn record(tenant: &str, model: &str, usage: RequestUsage) {
    if let Some(table) = INSTALLED.get() {
        table.record(tenant, model, usage);
    }
}

/// Append the installed table's families to a scrape.
pub fn render_into(out: &mut String) {
    if let Some(table) = INSTALLED.get() {
        table.render_into(out);
    }
}

/// What one finished request adds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RequestUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// From the request's dispatch to its last token.
    pub elapsed: Duration,
}

/// The usage of one streamed response, booked at its end whichever way it
/// ends.
///
/// A stream's completion tokens are counted as they pass
/// ([`add_completion_tokens`](Self::add_completion_tokens)); its normal end
/// books the authoritative numbers through [`finish`](Self::finish). A stream
/// that ends any other way never reaches that point: the client went away and
/// the next send into the response body failed, the engine's stream ended
/// with an error, a prefill leg failed before the first token. The guard's
/// `Drop` books those with what it has: the prompt as the gateway tokenized
/// it at dispatch, the completion tokens streamed so far, the time since the
/// stream began; once the stream has the engine's own counts (its `Complete`
/// messages) it hands them to the guard through [`settle`](Self::settle), so
/// a client that stops reading at the finish or usage chunk is booked with
/// them. One booking per stream either way: `finish` takes the guard, and a
/// guard that was finished has nothing left to book when it drops.
///
/// Every dispatched stream is booked, a stream whose first engine message is
/// an error included: it books one request of the prompt alone. The gateway
/// cannot tell a rejection from an engine that failed after the prefill, and
/// the row counts the work a tenant sent the engine; a buffered request that
/// fails has no stream and books nothing, which is the one place the two
/// modes differ.
///
/// With accounting off (no table installed) the guard is inert: `begin`
/// clones nothing and every method is one branch.
#[must_use = "a guard dropped at once books an empty stream; keep it to the stream's end"]
pub struct StreamUsage {
    /// `None` with accounting off, and once the stream is booked.
    booking: Option<Booking>,
}

/// Where and what a stream books.
struct Booking {
    table: &'static TenantUsage,
    tenant: Box<str>,
    model: Box<str>,
    /// The prompt's length as tokenized at dispatch, then the engine's count
    /// once the stream settled it.
    prompt_tokens: AtomicU64,
    /// Completion tokens streamed so far (the engine's count once settled).
    /// Atomic so the fan-out of a batched request can count into one guard by
    /// shared reference; one relaxed add per chunk.
    completion_tokens: AtomicU64,
    start: Instant,
}

impl StreamUsage {
    /// The guard of a stream for `tenant` and `model` whose prompt, as
    /// tokenized at dispatch, is `prompt_tokens` long. Books to the installed
    /// table; inert when none is.
    #[inline]
    pub fn begin(tenant: &str, model: &str, prompt_tokens: u64) -> Self {
        Self::begin_on(
            INSTALLED.get().map(|table| &**table),
            tenant,
            model,
            prompt_tokens,
        )
    }

    /// [`begin`](Self::begin) on `table`: the installed one, or a test's or a
    /// benchmark's own. `None` is accounting off.
    pub fn begin_on(
        table: Option<&'static TenantUsage>,
        tenant: &str,
        model: &str,
        prompt_tokens: u64,
    ) -> Self {
        Self {
            booking: table.map(|table| Booking {
                table,
                tenant: tenant.into(),
                model: model.into(),
                prompt_tokens: AtomicU64::new(prompt_tokens),
                completion_tokens: AtomicU64::new(0),
                start: Instant::now(),
            }),
        }
    }

    /// Count `tokens` completion tokens the stream delivered.
    #[inline]
    pub fn add_completion_tokens(&self, tokens: u64) {
        if let Some(booking) = &self.booking {
            booking
                .completion_tokens
                .fetch_add(tokens, Ordering::Relaxed);
        }
    }

    /// Take the engine's own counts once the stream has them (its `Complete`
    /// messages, before the finish or usage chunk the client may no longer
    /// read): what a drop books from here on. `None` keeps the prompt count
    /// the dispatch made.
    #[inline]
    pub fn settle(&self, prompt_tokens: Option<u64>, completion_tokens: u64) {
        if let Some(booking) = &self.booking {
            if let Some(prompt_tokens) = prompt_tokens {
                booking
                    .prompt_tokens
                    .store(prompt_tokens, Ordering::Relaxed);
            }
            booking
                .completion_tokens
                .store(completion_tokens, Ordering::Relaxed);
        }
    }

    /// Book the stream's normal end: `completion_tokens`, the prompt count
    /// the end knows (else the settled one, else the one counted at
    /// dispatch), and `elapsed`, the stream's duration as the metrics
    /// measure it.
    pub fn finish(mut self, prompt_tokens: Option<u64>, completion_tokens: u64, elapsed: Duration) {
        if let Some(booking) = self.booking.take() {
            booking.book(RequestUsage {
                prompt_tokens: prompt_tokens
                    .unwrap_or_else(|| booking.prompt_tokens.load(Ordering::Relaxed)),
                completion_tokens,
                elapsed,
            });
        }
    }
}

impl Booking {
    fn book(&self, usage: RequestUsage) {
        self.table.record(&self.tenant, &self.model, usage);
    }
}

impl Drop for StreamUsage {
    /// A stream that ended without its normal end: what was counted so far,
    /// or the engine's counts once settled.
    fn drop(&mut self) {
        if let Some(booking) = self.booking.take() {
            booking.book(RequestUsage {
                prompt_tokens: booking.prompt_tokens.load(Ordering::Relaxed),
                completion_tokens: booking.completion_tokens.load(Ordering::Relaxed),
                elapsed: booking.start.elapsed(),
            });
        }
    }
}

/// The sums of one row: a tenant and model pair, a tenant, or the table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Totals {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Dispatch-to-last-token time, summed; over `requests` it is the mean.
    pub request_seconds: f64,
}

impl Totals {
    fn add(&mut self, other: Totals) {
        self.requests += other.requests;
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.request_seconds += other.request_seconds;
    }
}

#[derive(Default)]
struct Counters {
    requests: AtomicU64,
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    elapsed_micros: AtomicU64,
}

impl Counters {
    fn add(&self, usage: RequestUsage) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens
            .fetch_add(usage.prompt_tokens, Ordering::Relaxed);
        self.completion_tokens
            .fetch_add(usage.completion_tokens, Ordering::Relaxed);
        self.elapsed_micros.fetch_add(
            u64::try_from(usage.elapsed.as_micros()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    fn totals(&self) -> Totals {
        Totals {
            requests: self.requests.load(Ordering::Relaxed),
            prompt_tokens: self.prompt_tokens.load(Ordering::Relaxed),
            completion_tokens: self.completion_tokens.load(Ordering::Relaxed),
            request_seconds: Duration::from_micros(self.elapsed_micros.load(Ordering::Relaxed))
                .as_secs_f64(),
        }
    }
}

/// One tenant's rows, by model. Models are bounded by what the fleet serves
/// (a request for an unserved model never reaches a dispatch).
#[derive(Default)]
struct TenantEntry {
    models: RwLock<HashMap<Arc<str>, Counters>>,
}

impl TenantEntry {
    fn record(&self, model: &str, usage: RequestUsage) {
        if let Some(counters) = self.models.read().get(model) {
            counters.add(usage);
            return;
        }
        self.models
            .write()
            .entry(Arc::from(model))
            .or_default()
            .add(usage);
    }

    fn snapshot(&self, tenant: &str) -> TenantUsageSnapshot {
        let mut models: Vec<ModelUsageSnapshot> = self
            .models
            .read()
            .iter()
            .map(|(model, counters)| ModelUsageSnapshot {
                model: model.to_string(),
                totals: counters.totals(),
            })
            .collect();
        models.sort_by(|a, b| a.model.cmp(&b.model));
        let mut totals = Totals::default();
        for model in &models {
            totals.add(model.totals);
        }
        TenantUsageSnapshot {
            tenant: tenant.to_string(),
            totals,
            models,
        }
    }
}

/// One model's sums for a tenant, in the snapshot.
#[derive(Clone, Debug, Serialize)]
pub struct ModelUsageSnapshot {
    pub model: String,
    #[serde(flatten)]
    pub totals: Totals,
}

/// One tenant's sums and its per-model rows, in the snapshot.
#[derive(Clone, Debug, Serialize)]
pub struct TenantUsageSnapshot {
    pub tenant: String,
    #[serde(flatten)]
    pub totals: Totals,
    pub models: Vec<ModelUsageSnapshot>,
}

/// The table at one instant: `GET /v1/usage/tenants`'s body.
#[derive(Clone, Debug, Serialize)]
pub struct UsageSnapshot {
    pub max_tenants: usize,
    pub tracked_tenants: usize,
    /// The tracked tenants, by key.
    pub tenants: Vec<TenantUsageSnapshot>,
    /// Every tenant seen once the tracked maximum was reached.
    pub other: TenantUsageSnapshot,
}

/// The per-tenant usage table of the process.
pub struct TenantUsage {
    max_tenants: usize,
    tenants: DashMap<Arc<str>, TenantEntry>,
    /// Rows claimed so far: claimed before a row is inserted, so the maximum
    /// holds exactly when several new tenants arrive at once.
    tracked: AtomicUsize,
    other: TenantEntry,
}

impl TenantUsage {
    /// A table with rows for the first `max_tenants` distinct tenant keys.
    pub fn new(max_tenants: usize) -> Self {
        Self {
            max_tenants,
            tenants: DashMap::new(),
            tracked: AtomicUsize::new(0),
            other: TenantEntry::default(),
        }
    }

    pub fn max_tenants(&self) -> usize {
        self.max_tenants
    }

    /// Distinct tenants with a row of their own so far.
    pub fn tracked_tenants(&self) -> usize {
        self.tracked.load(Ordering::Acquire)
    }

    /// Book one finished request. A tenant seen for the first time takes a
    /// row while rows remain, and is booked under [`OTHER_TENANT`] after.
    ///
    /// The count of rows is exact under concurrency (the slot claim below);
    /// the attribution of a request is exact but for one window: the first
    /// requests of a tenant arriving together while the last rows go, where
    /// a request that lost the claim is booked under `other` unless its
    /// tenant's row is already in place when it looks again. The totals
    /// across the rows and `other` are exact either way.
    pub fn record(&self, tenant: &str, model: &str, usage: RequestUsage) {
        if let Some(entry) = self.tenants.get(tenant) {
            entry.record(model, usage);
            return;
        }
        let claimed = self
            .tracked
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |rows| {
                (rows < self.max_tenants).then_some(rows + 1)
            })
            .is_ok();
        if !claimed {
            // The table filled between the lookup and the claim: a tenant
            // whose other request took the last row meanwhile is booked to
            // that row, not under `other`.
            match self.tenants.get(tenant) {
                Some(entry) => entry.record(model, usage),
                None => self.other.record(model, usage),
            }
            return;
        }
        match self.tenants.entry(Arc::from(tenant)) {
            // Another request of the same new tenant inserted the row between
            // the lookup and the claim: the claimed row goes back.
            Entry::Occupied(entry) => {
                self.tracked.fetch_sub(1, Ordering::AcqRel);
                entry.get().record(model, usage);
            }
            Entry::Vacant(slot) => slot.insert(TenantEntry::default()).record(model, usage),
        }
    }

    /// The table at this instant, tenants by key.
    pub fn snapshot(&self) -> UsageSnapshot {
        let mut tenants: Vec<TenantUsageSnapshot> = self
            .tenants
            .iter()
            .map(|entry| entry.value().snapshot(entry.key()))
            .collect();
        tenants.sort_by(|a, b| a.tenant.cmp(&b.tenant));
        UsageSnapshot {
            max_tenants: self.max_tenants,
            tracked_tenants: self.tracked_tenants(),
            tenants,
            other: self.other.snapshot(OTHER_TENANT),
        }
    }

    /// Append the table's families to a scrape, in the exposition format.
    /// The `other` row renders once it has a request; the counters render for
    /// every tenant and model pair seen; the two gauges always.
    pub fn render_into(&self, out: &mut String) {
        let snapshot = self.snapshot();
        let rows: Vec<&TenantUsageSnapshot> = snapshot
            .tenants
            .iter()
            .chain((snapshot.other.totals.requests > 0).then_some(&snapshot.other))
            .collect();

        write_help_line(
            out,
            REQUESTS,
            None,
            None,
            "Requests finished by tenant and model; every tenant past the tracked maximum is booked under the tenant other",
        );
        write_type_line(out, REQUESTS, None, None, "counter");
        for row in &rows {
            for model in &row.models {
                write_sample(
                    out,
                    REQUESTS,
                    &row.tenant,
                    &model.model,
                    None,
                    model.totals.requests,
                );
            }
        }

        write_help_line(
            out,
            TOKENS,
            None,
            None,
            "Tokens by tenant, model and token_type (input is the prompt, output the completion)",
        );
        write_type_line(out, TOKENS, None, None, "counter");
        for row in &rows {
            for model in &row.models {
                write_sample(
                    out,
                    TOKENS,
                    &row.tenant,
                    &model.model,
                    Some(TOKEN_INPUT),
                    model.totals.prompt_tokens,
                );
                write_sample(
                    out,
                    TOKENS,
                    &row.tenant,
                    &model.model,
                    Some(TOKEN_OUTPUT),
                    model.totals.completion_tokens,
                );
            }
        }

        write_help_line(
            out,
            SECONDS,
            None,
            None,
            "Seconds from dispatch to the last token, summed by tenant and model; over the requests series it is the mean",
        );
        write_type_line(out, SECONDS, None, None, "counter");
        for row in &rows {
            for model in &row.models {
                write_sample(
                    out,
                    SECONDS,
                    &row.tenant,
                    &model.model,
                    None,
                    model.totals.request_seconds,
                );
            }
        }

        write_help_line(
            out,
            TRACKED,
            None,
            None,
            "Distinct tenants with usage rows of their own",
        );
        write_type_line(out, TRACKED, None, None, "gauge");
        let _ = writeln!(out, "{TRACKED} {}", snapshot.tracked_tenants);
        write_help_line(
            out,
            MAX,
            None,
            None,
            "Configured maximum of tenants with usage rows of their own",
        );
        write_type_line(out, MAX, None, None, "gauge");
        let _ = writeln!(out, "{MAX} {}", snapshot.max_tenants);
    }
}

fn write_sample(
    out: &mut String,
    family: &str,
    tenant: &str,
    model: &str,
    token_type: Option<&str>,
    value: impl std::fmt::Display,
) {
    let _ = write!(
        out,
        "{family}{{tenant=\"{}\",model=\"{}\"",
        sanitize_label_value(tenant),
        sanitize_label_value(model)
    );
    if let Some(token_type) = token_type {
        let _ = write!(out, ",token_type=\"{token_type}\"");
    }
    let _ = writeln!(out, "}} {value}");
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    fn usage(prompt: u64, completion: u64, millis: u64) -> RequestUsage {
        RequestUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            elapsed: Duration::from_millis(millis),
        }
    }

    fn tenant<'a>(snapshot: &'a UsageSnapshot, key: &str) -> &'a TenantUsageSnapshot {
        snapshot
            .tenants
            .iter()
            .find(|t| t.tenant == key)
            .unwrap_or_else(|| panic!("tenant {key} has a row"))
    }

    #[test]
    fn sums_requests_tokens_and_time_by_tenant_and_model() {
        let table = TenantUsage::new(8);
        table.record("header:acme", "m1", usage(100, 10, 250));
        table.record("header:acme", "m1", usage(50, 5, 250));
        table.record("header:acme", "m2", usage(7, 3, 100));
        table.record("header:bravo", "m1", usage(1, 1, 10));

        let snapshot = table.snapshot();
        assert_eq!(snapshot.tracked_tenants, 2);
        assert_eq!(snapshot.max_tenants, 8);
        let acme = tenant(&snapshot, "header:acme");
        assert_eq!(acme.totals.requests, 3);
        assert_eq!(acme.totals.prompt_tokens, 157);
        assert_eq!(acme.totals.completion_tokens, 18);
        assert!((acme.totals.request_seconds - 0.6).abs() < 1e-9);
        assert_eq!(acme.models.len(), 2);
        assert_eq!(acme.models[0].model, "m1");
        assert_eq!(acme.models[0].totals.requests, 2);
        assert_eq!(acme.models[0].totals.prompt_tokens, 150);
        assert_eq!(acme.models[1].model, "m2");
        assert_eq!(acme.models[1].totals.completion_tokens, 3);
        let bravo = tenant(&snapshot, "header:bravo");
        assert_eq!(bravo.totals.requests, 1);
        assert_eq!(snapshot.other.totals.requests, 0);
        assert!(snapshot.other.models.is_empty());
    }

    #[test]
    fn tenants_past_the_maximum_fold_into_other_and_tracked_rows_keep_counting() {
        let table = TenantUsage::new(2);
        table.record("auth:one", "m", usage(1, 1, 1));
        table.record("auth:two", "m", usage(1, 1, 1));
        table.record("auth:three", "m", usage(10, 10, 10));
        table.record("auth:four", "n", usage(20, 20, 20));
        // A tenant that took a row before the cap keeps it after.
        table.record("auth:one", "m", usage(1, 1, 1));

        let snapshot = table.snapshot();
        assert_eq!(snapshot.tracked_tenants, 2);
        assert_eq!(snapshot.tenants.len(), 2);
        assert_eq!(tenant(&snapshot, "auth:one").totals.requests, 2);
        assert!(snapshot.tenants.iter().all(|t| t.tenant != "auth:three"));
        assert_eq!(snapshot.other.tenant, OTHER_TENANT);
        assert_eq!(snapshot.other.totals.requests, 2);
        assert_eq!(snapshot.other.totals.prompt_tokens, 30);
        assert_eq!(snapshot.other.models.len(), 2);
    }

    #[test]
    fn a_maximum_of_zero_books_everything_under_other() {
        let table = TenantUsage::new(0);
        table.record("auth:one", "m", usage(1, 1, 1));
        let snapshot = table.snapshot();
        assert_eq!(snapshot.tracked_tenants, 0);
        assert!(snapshot.tenants.is_empty());
        assert_eq!(snapshot.other.totals.requests, 1);
    }

    #[test]
    fn the_maximum_holds_exactly_under_concurrent_first_sightings() {
        const THREADS: u64 = 8;
        const TENANTS_PER_THREAD: u64 = 200;
        let table = Arc::new(TenantUsage::new(16));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let table = Arc::clone(&table);
                thread::spawn(move || {
                    // Every thread sees every tenant, so the same new tenant
                    // races itself across threads as well.
                    for i in 0..TENANTS_PER_THREAD {
                        table.record(&format!("ip:10.0.0.{i}"), "m", usage(1, 0, 1));
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("thread");
        }

        let snapshot = table.snapshot();
        assert_eq!(snapshot.tracked_tenants, 16);
        assert_eq!(snapshot.tenants.len(), 16);
        assert_eq!(table.tenants.len(), 16);
        let tracked: u64 = snapshot.tenants.iter().map(|t| t.totals.requests).sum();
        assert_eq!(
            tracked + snapshot.other.totals.requests,
            THREADS * TENANTS_PER_THREAD
        );
    }

    #[test]
    fn render_writes_the_families_with_help_type_and_escaped_labels() {
        let table = TenantUsage::new(4);
        table.record("header:acme \"east\"\\1", "org/model", usage(120, 30, 1500));
        table.record("header:acme \"east\"\\1", "org/model", usage(80, 20, 500));

        let mut out = String::new();
        table.render_into(&mut out);
        let line = |needle: &str| {
            out.lines()
                .find(|l| l.starts_with(needle))
                .unwrap_or_else(|| panic!("line starting with {needle:?} in:\n{out}"))
                .to_string()
        };
        assert!(out.contains("# HELP smg_tenant_usage_requests_total "));
        assert!(out.contains("# TYPE smg_tenant_usage_requests_total counter\n"));
        assert!(out.contains("# TYPE smg_tenant_usage_tokens_total counter\n"));
        assert!(out.contains("# TYPE smg_tenant_usage_request_seconds_total counter\n"));
        assert!(out.contains("# TYPE smg_tenant_usage_tracked_tenants gauge\n"));
        assert!(out.contains("# TYPE smg_tenant_usage_max_tenants gauge\n"));
        assert_eq!(
            line("smg_tenant_usage_requests_total{"),
            "smg_tenant_usage_requests_total{tenant=\"header:acme \\\"east\\\"\\\\1\",model=\"org/model\"} 2"
        );
        assert!(out.contains(
            "smg_tenant_usage_tokens_total{tenant=\"header:acme \\\"east\\\"\\\\1\",model=\"org/model\",token_type=\"input\"} 200\n"
        ));
        assert!(out.contains(
            "smg_tenant_usage_tokens_total{tenant=\"header:acme \\\"east\\\"\\\\1\",model=\"org/model\",token_type=\"output\"} 50\n"
        ));
        assert_eq!(
            line("smg_tenant_usage_request_seconds_total{"),
            "smg_tenant_usage_request_seconds_total{tenant=\"header:acme \\\"east\\\"\\\\1\",model=\"org/model\"} 2"
        );
        assert_eq!(
            line("smg_tenant_usage_tracked_tenants "),
            "smg_tenant_usage_tracked_tenants 1"
        );
        assert_eq!(
            line("smg_tenant_usage_max_tenants "),
            "smg_tenant_usage_max_tenants 4"
        );
        // No request past the cap: no `other` rows.
        assert!(!out.contains("tenant=\"other\""));
    }

    #[test]
    fn render_shows_the_other_row_once_a_tenant_is_booked_there() {
        let table = TenantUsage::new(1);
        table.record("auth:one", "m", usage(1, 1, 1));
        let mut before = String::new();
        table.render_into(&mut before);
        assert!(!before.contains("tenant=\"other\""));

        table.record("auth:two", "m", usage(5, 6, 1));
        let mut after = String::new();
        table.render_into(&mut after);
        assert!(after.contains("smg_tenant_usage_requests_total{tenant=\"other\",model=\"m\"} 1\n"));
        assert!(after.contains(
            "smg_tenant_usage_tokens_total{tenant=\"other\",model=\"m\",token_type=\"output\"} 6\n"
        ));
    }

    #[test]
    fn the_snapshot_serializes_flat_totals_per_row() {
        let table = TenantUsage::new(2);
        table.record("auth:one", "m", usage(3, 4, 2000));
        let json = serde_json::to_value(table.snapshot()).expect("serializes");
        assert_eq!(json["max_tenants"], 2);
        assert_eq!(json["tracked_tenants"], 1);
        assert_eq!(json["tenants"][0]["tenant"], "auth:one");
        assert_eq!(json["tenants"][0]["requests"], 1);
        assert_eq!(json["tenants"][0]["prompt_tokens"], 3);
        assert_eq!(json["tenants"][0]["completion_tokens"], 4);
        assert_eq!(json["tenants"][0]["request_seconds"], 2.0);
        assert_eq!(json["tenants"][0]["models"][0]["model"], "m");
        assert_eq!(json["tenants"][0]["models"][0]["requests"], 1);
        assert_eq!(json["other"]["tenant"], "other");
        assert_eq!(json["other"]["requests"], 0);
    }

    #[test]
    fn nothing_renders_or_records_before_a_table_is_installed() {
        // The unit tests of this module never install one, so the global
        // path is the disabled path here.
        record("auth:one", "m", usage(1, 1, 1));
        let mut out = String::new();
        render_into(&mut out);
        assert!(out.is_empty() || installed().is_some());
    }

    /// A table of a test's own with the `'static` lifetime a guard books to.
    fn leaked_table(max_tenants: usize) -> &'static TenantUsage {
        Box::leak(Box::new(TenantUsage::new(max_tenants)))
    }

    #[test]
    fn a_finished_stream_books_the_end_numbers_once() {
        let table = leaked_table(4);
        let stream = StreamUsage::begin_on(Some(table), "auth:a", "m", 100);
        stream.add_completion_tokens(3);
        stream.add_completion_tokens(4);
        stream.finish(Some(120), 9, Duration::from_secs(2));

        let snapshot = table.snapshot();
        let row = tenant(&snapshot, "auth:a");
        assert_eq!(row.totals.requests, 1);
        assert_eq!(row.totals.prompt_tokens, 120, "the end's prompt count");
        assert_eq!(
            row.totals.completion_tokens, 9,
            "the end's completion count"
        );
        assert!((row.totals.request_seconds - 2.0).abs() < 1e-9);
        assert_eq!(row.models[0].model, "m");
    }

    #[test]
    fn a_finished_stream_without_an_end_prompt_count_books_the_dispatch_count() {
        let table = leaked_table(4);
        let stream = StreamUsage::begin_on(Some(table), "auth:a", "m", 100);
        stream.finish(None, 5, Duration::from_millis(10));

        let snapshot = table.snapshot();
        assert_eq!(tenant(&snapshot, "auth:a").totals.prompt_tokens, 100);
        assert_eq!(tenant(&snapshot, "auth:a").totals.completion_tokens, 5);
    }

    #[test]
    fn a_dropped_stream_books_the_dispatch_prompt_and_the_tokens_streamed_once() {
        let table = leaked_table(4);
        let stream = StreamUsage::begin_on(Some(table), "auth:a", "m", 100);
        stream.add_completion_tokens(3);
        stream.add_completion_tokens(4);
        thread::sleep(Duration::from_millis(2));
        drop(stream);

        let snapshot = table.snapshot();
        let row = tenant(&snapshot, "auth:a");
        assert_eq!(row.totals.requests, 1);
        assert_eq!(
            row.totals.prompt_tokens, 100,
            "the prompt as tokenized at dispatch"
        );
        assert_eq!(
            row.totals.completion_tokens, 7,
            "the tokens streamed before the end"
        );
        assert!(
            row.totals.request_seconds >= 0.002,
            "{}",
            row.totals.request_seconds
        );
    }

    #[test]
    fn a_dropped_stream_books_the_counts_it_settled() {
        let table = leaked_table(4);
        let stream = StreamUsage::begin_on(Some(table), "auth:a", "m", 100);
        stream.add_completion_tokens(3);
        // The engine's Complete arrived: its counts replace the estimates ...
        stream.settle(Some(120), 9);
        drop(stream);
        let snapshot = table.snapshot();
        let row = tenant(&snapshot, "auth:a");
        assert_eq!(row.totals.requests, 1);
        assert_eq!(row.totals.prompt_tokens, 120, "the engine's prompt count");
        assert_eq!(
            row.totals.completion_tokens, 9,
            "the engine's completion count"
        );

        // ... and a settle without a prompt count keeps the dispatch count.
        let stream = StreamUsage::begin_on(Some(table), "auth:b", "m", 100);
        stream.settle(None, 2);
        drop(stream);
        let snapshot = table.snapshot();
        assert_eq!(tenant(&snapshot, "auth:b").totals.prompt_tokens, 100);
        assert_eq!(tenant(&snapshot, "auth:b").totals.completion_tokens, 2);
    }

    #[test]
    fn a_stream_dropped_without_a_token_books_one_request_of_the_prompt_alone() {
        let table = leaked_table(4);
        drop(StreamUsage::begin_on(Some(table), "auth:a", "m", 42));

        let snapshot = table.snapshot();
        assert_eq!(tenant(&snapshot, "auth:a").totals.requests, 1);
        assert_eq!(tenant(&snapshot, "auth:a").totals.prompt_tokens, 42);
        assert_eq!(tenant(&snapshot, "auth:a").totals.completion_tokens, 0);
    }

    #[test]
    fn a_stream_with_accounting_off_holds_nothing() {
        let stream = StreamUsage::begin_on(None, "auth:a", "m", 100);
        assert!(stream.booking.is_none());
        stream.add_completion_tokens(1);
        stream.finish(Some(1), 1, Duration::ZERO);
        drop(StreamUsage::begin_on(None, "auth:a", "m", 100));
    }
}
