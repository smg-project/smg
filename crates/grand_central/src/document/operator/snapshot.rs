//! The worker rows an operator reads, in registration key order.
//!
//! A caller can ask for the workers serving particular models and cap the
//! row count. A page also has a byte bound. Each row's registration and
//! engine addresses, and its provider's key, are told without their
//! userinfo; the store keeps the originals.
//!
//! A page writes itself as JSON ([`WorkerRowsPage`]): a host in process serves
//! the rows an operator reads with no mapping of its own.

#![expect(
    clippy::exhaustive_structs,
    reason = "a caller still builds or matches these types whole: each mark comes with its call"
)]

use std::{borrow::Cow, collections::BTreeMap, sync::Arc, time::Instant};

use serde::{Serialize, Serializer};

use crate::{
    address::{split, Address, UNPARSABLE},
    engine::{
        view::{Role, WireModes},
        DemandFacts,
    },
    store::{
        membership::blank, pairing::LAYOUT_FACTS, EncodeHandoff, Reachability, Registrant,
        StoreCheckpoint, WorkerFacts, WorkerRow, WorkerStatus,
    },
};

/// The byte bound for one page of rows.
pub(crate) const PAGE_BYTES: usize = 2 * 1024 * 1024;

/// The most model ids a snapshot or a watch names.
pub(crate) const MODELS: usize = 128;

/// What a snapshot caller asks of the worker rows.
#[derive(Debug)]
pub struct WorkerRowsQuery {
    /// The model ids to include. An empty list includes every model.
    pub models: Vec<String>,
    /// The maximum number of rows to return. `usize::MAX` uses only the
    /// byte bound.
    pub limit: usize,
}

impl WorkerRowsQuery {
    /// Why a snapshot may not name `models`, and `None` where it may. The
    /// list is read at each row, so it keeps a row's own rules: at most 128
    /// ids, each of at most 256 bytes, none empty, none with a blank or a
    /// control character. Every host of a snapshot refuses by this, under
    /// the key `models`.
    #[must_use]
    pub fn refused(models: &[String]) -> Option<&'static str> {
        if models.len() > MODELS {
            return Some("a snapshot names at most 128 model ids");
        }
        unnamed(models)
    }

    pub(crate) fn serves(&self, row: &WorkerRow) -> bool {
        serves(&self.models, row)
    }
}

/// Why a list read at each row may not hold one of `models`, by a row's own
/// rules of a model id, and `None` where it may hold each.
pub(crate) fn unnamed(models: &[String]) -> Option<&'static str> {
    if models
        .iter()
        .any(|model| model.len() > WorkerFacts::NAME_BYTES)
    {
        return Some("a model id is at most 256 bytes");
    }
    if models.iter().any(|model| model.is_empty() || blank(model)) {
        return Some("a model id is not empty and holds no blank and no control character");
    }
    None
}

/// Whether a row serves any named model. An empty list includes every row.
/// A controller's lone read of a few rows matches by the same rule,
/// searched in a sorted list (the `rows_serving` of its worker objects'
/// holder): the two change together.
pub(crate) fn serves(models: &[String], row: &WorkerRow) -> bool {
    serves_models(models, &row.facts.models)
}

/// The rule of [`serves`] over the models a row serves, `served`: the one
/// fact of a row the rule reads. A status entry of the membership log holds
/// that list in its row's place, and a watch reads it by this rule.
pub(crate) fn serves_models(models: &[String], served: &[String]) -> bool {
    models.is_empty() || served.iter().any(|model| models.contains(model))
}

/// The store rows read for a snapshot.
#[non_exhaustive]
pub enum Rows {
    /// A shared cut, ordered by registration key, before model filtering.
    Shared(Arc<StoreCheckpoint>),
    /// The first serving rows in key order and the total that serve, as
    /// a host's lone read of a few rows returns them.
    Told(Vec<WorkerRow>, usize),
}

/// A page of worker rows as an operator reads it.
#[derive(Debug)]
#[non_exhaustive]
pub struct WorkerRowsPage {
    /// The matching rows in registration key order, with addresses redacted.
    pub rows: Vec<WorkerRow>,
    /// Whether the row count or byte bound left a matching row out.
    pub truncated: bool,
    /// The instant of the take: a row's age is told against it.
    pub taken: Instant,
}

/// A page as JSON: `rows` and `truncated`. A row reads by the field names
/// of the contract's worker row, with the values a person reads. The id is
/// sixteen hex digits, as the controller's messages print a worker. A role,
/// a status and a handoff are names, and the wires a list of names. The age
/// is in milliseconds at the take. A fact the row does not state is `null`,
/// and `pairing` is `null` for a row that states nothing of a pair or of an
/// encode handoff. `demand_facts` is `null` for a row whose worker states
/// nothing for a request's hard demands. Inside that block a fact the
/// worker does not state reads `0` or `false`, as the contract's block
/// tells it.
impl Serialize for WorkerRowsPage {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let rows = self.rows.iter().map(|row| RowJson::of(row, self.taken));
        PageJson {
            rows: rows.collect(),
            truncated: self.truncated,
        }
        .serialize(serializer)
    }
}

/// A page's JSON ([`WorkerRowsPage`]): its rows, and whether a matching
/// row was left out.
#[derive(Serialize)]
struct PageJson<'a> {
    rows: Vec<RowJson<'a>>,
    truncated: bool,
}

/// One row of a page's JSON ([`WorkerRowsPage`]).
#[derive(Serialize)]
struct RowJson<'a> {
    id: String,
    registration_key: &'a str,
    incarnation: u64,
    engine: &'a str,
    wire: Vec<&'static str>,
    role: &'static str,
    runtime: &'a str,
    models: &'a [String],
    dp_rank: u16,
    dp_size: u16,
    labels: &'a BTreeMap<String, String>,
    zone: Option<&'a str>,
    status: &'static str,
    revision: u64,
    reachable: ReachableJson,
    pairing: Option<PairingJson<'a>>,
    max_running: Option<u32>,
    kv_block_size: Option<u32>,
    priority: i32,
    overload: OverloadJson,
    signals: SignalsJson,
    health_policy: HealthPolicyJson,
    admitted_age_ms: u64,
    maker: Option<RegistrantJson<'a>>,
    demand_facts: Option<DemandFactsJson>,
}

/// From where a row is reachable: every gateway, or the one that holds its
/// host-local wire, by the sixteen hex digits the contract tells.
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum ReachableJson {
    Any(bool),
    GatewayId(String),
}

/// What a row states of a pair and of an encode handoff, in one object. A
/// pair's fields stand under the names of the contract's pairing
/// descriptor. `vision` and the encode offer's five fields keep the names
/// the descriptor reserves: the contract states them on the registration
/// and on the row themselves.
#[derive(Serialize)]
struct PairingJson<'a> {
    protocol: Option<&'a str>,
    transport: Option<&'a str>,
    version: Option<&'a str>,
    bootstrap_port: Option<u16>,
    /// The layout facts the row reports, each by the name a mismatch
    /// reports.
    kv_layout: BTreeMap<&'static str, &'a str>,
    ec_producer: bool,
    ec_consumer: bool,
    vision: bool,
    encode_handoff: &'static str,
    ec_connector: Option<&'a str>,
    encode_references: bool,
}

#[derive(Serialize)]
struct OverloadJson {
    queue_limit: Option<u32>,
    kv_usage_permille: Option<u16>,
}

#[derive(Serialize)]
struct SignalsJson {
    events: bool,
    loads: bool,
    scrape_interval_ms: u32,
}

#[derive(Serialize)]
struct HealthPolicyJson {
    failure_threshold: u32,
    success_threshold: u32,
    check_interval_ms: u32,
    timeout_ms: u32,
    disabled: bool,
}

#[derive(Serialize)]
struct RegistrantJson<'a> {
    provider: &'a str,
    key: &'a str,
}

/// What a row's worker states for a request's hard demands, under the
/// names of the contract's block (`WorkerRow.demand_facts`).
#[derive(Serialize)]
struct DemandFactsJson {
    runtime: u32,
    references: bool,
    host_pipeline: bool,
    window: u32,
    decoding: u8,
}

impl<'a> RowJson<'a> {
    fn of(row: &'a WorkerRow, taken: Instant) -> Self {
        let facts = &row.facts;
        let wires = [
            (WireModes::HTTP, "http"),
            (WireModes::GRPC, "grpc"),
            (WireModes::ZMQ, "zmq"),
        ];
        let age = taken.saturating_duration_since(row.admitted_at).as_millis();
        Self {
            id: format!("{:016x}", row.id.0),
            registration_key: &facts.key,
            incarnation: facts.incarnation,
            engine: &facts.engine,
            wire: wires
                .into_iter()
                .filter(|(wire, _)| facts.wire_modes.bits() & wire.bits() != 0)
                .map(|(_, name)| name)
                .collect(),
            role: match facts.role {
                Role::Single => "single",
                Role::Prefill => "prefill",
                Role::Decode => "decode",
                Role::Encode => "encode",
            },
            runtime: &facts.engine_kind,
            models: &facts.models,
            dp_rank: facts.dp_rank,
            dp_size: facts.dp_size,
            labels: &facts.labels,
            zone: facts.zone.as_deref(),
            status: match row.status {
                WorkerStatus::Pending => "pending",
                WorkerStatus::Ready => "ready",
                WorkerStatus::NotReady => "not_ready",
                WorkerStatus::Failed => "failed",
                WorkerStatus::Draining => "draining",
            },
            revision: row.revision,
            reachable: match facts.reachable_from {
                Reachability::Any => ReachableJson::Any(true),
                Reachability::Gateway(gateway) => {
                    ReachableJson::GatewayId(format!("{:016x}", gateway.0))
                }
            },
            pairing: PairingJson::of(row),
            max_running: facts.max_running,
            kv_block_size: facts.kv_block_size,
            priority: facts.priority,
            overload: OverloadJson {
                queue_limit: facts.overload.waiting_requests,
                kv_usage_permille: facts.overload.kv_usage_permille,
            },
            signals: SignalsJson {
                events: facts.signals.events,
                loads: facts.signals.loads,
                scrape_interval_ms: facts.signals.scrape_interval_ms,
            },
            health_policy: HealthPolicyJson {
                failure_threshold: facts.health.failure_threshold,
                success_threshold: facts.health.success_threshold,
                check_interval_ms: facts.health.check_interval_ms,
                timeout_ms: facts.health.timeout_ms,
                disabled: !facts.health.probes,
            },
            admitted_age_ms: u64::try_from(age).unwrap_or(u64::MAX),
            maker: match &row.maker {
                Registrant::Control => None,
                Registrant::Provider(maker) => Some(RegistrantJson {
                    provider: maker.provider(),
                    key: maker.key(),
                }),
            },
            demand_facts: DemandFactsJson::of(facts.demand_facts),
        }
    }
}

impl DemandFactsJson {
    /// What a row's worker states for a request's hard demands: none for a
    /// worker that states no fact, as the contract's row tells no block.
    fn of(stated: DemandFacts) -> Option<Self> {
        // Every fact by its name: one the engine's type gains stops the
        // build here, so no fact stays off the page.
        let DemandFacts {
            runtime,
            references,
            host_pipeline,
            window,
            decoding,
        } = stated;
        (stated != DemandFacts::default()).then_some(Self {
            runtime,
            references,
            host_pipeline,
            window,
            decoding,
        })
    }
}

impl<'a> PairingJson<'a> {
    /// What `row` states of a pair and of an encode handoff: none for a
    /// row that states nothing of either.
    fn of(row: &'a WorkerRow) -> Option<Self> {
        let facts = &row.facts;
        let pairing = facts.pairing.as_deref();
        let encode = &facts.encode;
        let states_none = pairing.is_none()
            && facts.bootstrap_port.is_none()
            && !facts.vision
            && *encode == Default::default();
        if states_none {
            return None;
        }
        let layout = pairing.map(|pairing| &pairing.kv_layout);
        Some(Self {
            protocol: pairing.and_then(|pairing| pairing.protocol.as_deref()),
            transport: pairing.and_then(|pairing| pairing.transport.as_deref()),
            version: pairing.and_then(|pairing| pairing.version.as_deref()),
            bootstrap_port: facts.bootstrap_port,
            kv_layout: LAYOUT_FACTS
                .iter()
                .zip(layout.into_iter().flatten())
                .filter_map(|((_, name), value)| Some((*name, value.as_deref()?)))
                .collect(),
            ec_producer: encode.producer,
            ec_consumer: encode.consumer,
            vision: facts.vision,
            encode_handoff: match encode.handoff {
                EncodeHandoff::None => "none",
                EncodeHandoff::Push => "push",
                EncodeHandoff::Rendezvous => "rendezvous",
            },
            ec_connector: encode.connector.as_deref(),
            encode_references: encode.references,
        })
    }
}

/// Takes a redacted page at `now`, limited by row count and by the bytes
/// its JSON takes ([`WorkerRowsPage`]), written compact. A host that writes
/// the page another way, as with indentation, can write more. A row
/// admitted after `now` reads as new at that instant. The first row must
/// fit the byte bound too.
pub fn worker_rows_page(asked: &WorkerRowsQuery, read: Rows, now: Instant) -> WorkerRowsPage {
    counted_page(asked, read, now).0
}

/// [`worker_rows_page`], with the bytes its paging counted for the page's
/// JSON. The count is the JSON's own length with `truncated` at its longer
/// word: a page that left a row out writes one byte fewer.
fn counted_page(asked: &WorkerRowsQuery, read: Rows, now: Instant) -> (WorkerRowsPage, usize) {
    let framing = Framing {
        around: written(&PageJson {
            rows: Vec::new(),
            truncated: false,
        }),
        between: 1,
    };
    let (rows, truncated, bytes) = paged(asked, read, now, false, framing, |row| {
        // The row as the page writes it: redacted, at the take's instant.
        let bytes = written(&RowJson::of(&row, now));
        (row.into_owned(), bytes)
    });
    let page = WorkerRowsPage {
        rows,
        truncated,
        taken: now,
    };
    (page, bytes)
}

/// Pages the redacted rows using the host's measure of the answer. A host
/// that sends an oversized first row alone can keep that rule here.
#[cfg(feature = "controller")]
pub(crate) fn worker_rows_page_with<T>(
    asked: &WorkerRowsQuery,
    read: Rows,
    now: Instant,
    allow_oversized_first: bool,
    map: impl Fn(Cow<'_, WorkerRow>) -> (T, usize),
) -> (Vec<T>, bool) {
    let framing = Framing::default();
    let (rows, truncated, _) = paged(asked, read, now, allow_oversized_first, framing, map);
    (rows, truncated)
}

/// What a page's bytes hold beside its rows' own: the bytes around the
/// rows, and the bytes between two of them. A host whose measure frames
/// each row states neither.
#[derive(Clone, Copy, Default)]
struct Framing {
    around: usize,
    between: usize,
}

/// The redacted rows `asked` serves out of `read`, paged by `map`'s measure
/// of each under `framing`, with the bytes counted for the page.
fn paged<T>(
    asked: &WorkerRowsQuery,
    read: Rows,
    now: Instant,
    allow_oversized_first: bool,
    framing: Framing,
    map: impl Fn(Cow<'_, WorkerRow>) -> (T, usize),
) -> (Vec<T>, bool, usize) {
    let told = |row| map(redacted(row, now));
    match read {
        Rows::Shared(cut) => page(
            asked.limit,
            cut.rows
                .iter()
                .filter(|row| asked.serves(row))
                .map(Cow::Borrowed),
            None,
            allow_oversized_first,
            framing,
            told,
        ),
        Rows::Told(rows, serving) => page(
            asked.limit,
            rows.into_iter().map(Cow::Owned),
            Some(serving),
            allow_oversized_first,
            framing,
            told,
        ),
    }
}

/// A single pass over the serving rows, stopping at the first bound.
/// The store supplies the serving count when it has already selected a
/// prefix. A shared cut needs just one more matching row to say it was cut.
/// The bytes counted are the page's: a row that did not fit adds none.
fn page<'a, T>(
    limit: usize,
    mut rows: impl Iterator<Item = Cow<'a, WorkerRow>>,
    serving: Option<usize>,
    allow_oversized_first: bool,
    framing: Framing,
    told: impl Fn(Cow<'a, WorkerRow>) -> (T, usize),
) -> (Vec<T>, bool, usize) {
    let mut page = Vec::new();
    let mut bytes = framing.around;
    for row in rows.by_ref().take(limit) {
        let (row, size) = told(row);
        let between = if page.is_empty() { 0 } else { framing.between };
        let with = bytes.saturating_add(between).saturating_add(size);
        if with > PAGE_BYTES && !(allow_oversized_first && page.is_empty()) {
            return (page, true, bytes);
        }
        bytes = with;
        page.push(row);
    }
    let truncated = serving.map_or_else(|| rows.next().is_some(), |count| page.len() < count);
    (page, truncated, bytes)
}

/// Only copy a shared row here if an address or timestamp needs changing.
/// A provider's row tells its maker's key too, the provider's own text,
/// which may hold an address with a credential ([`key_without_userinfo`]).
fn redacted(mut row: Cow<'_, WorkerRow>, now: Instant) -> Cow<'_, WorkerRow> {
    if let Some(key) = without_userinfo(&row.facts.key) {
        row.to_mut().facts.key = key;
    }
    if let Some(engine) = without_userinfo(&row.facts.engine) {
        row.to_mut().facts.engine = engine;
    }
    let maker = match &row.maker {
        Registrant::Provider(maker) => {
            key_without_userinfo(maker.key()).map(|key| told_maker(maker.provider(), key))
        }
        Registrant::Control => None,
    };
    if let Some(maker) = maker {
        row.to_mut().maker = maker;
    }
    if row.admitted_at > now {
        row.to_mut().admitted_at = now;
    }
    row
}

/// The maker a row tells in place of one whose key is not told as it
/// stands: the provider's name with the key a snapshot tells
/// ([`key_without_userinfo`]). That key is inside a key's rule
/// ([`Registrant::provider`]), never empty and never past the bound, so the
/// last arm is never told.
fn told_maker(provider: &str, key: String) -> Registrant {
    Registrant::provider(provider, key).unwrap_or_default()
}

/// A provider's key as a snapshot tells it, or `None` where it is told as
/// it stands. The key is the provider's own text and need be no address,
/// so no part is cut off a key that is none: that would tell a key the
/// provider never wrote. A key with a scheme is read as a URL and loses
/// its userinfo. Any other key that holds an `@` may hide a credential
/// before it, and reads as [`UNPARSABLE`], which says so. A key with no
/// `@` holds no userinfo and is told whole.
fn key_without_userinfo(key: &str) -> Option<String> {
    match split(key) {
        Some(Address {
            userinfo: false, ..
        }) => None,
        Some(Address {
            scheme,
            userinfo: true,
            host,
            tail,
        }) if !scheme.is_empty() => Some(format!("{scheme}{host}{tail}")),
        _ => Some(UNPARSABLE.to_owned()),
    }
}

/// `address` without its userinfo, or `None` where it carries none: the
/// userinfo goes, and every other byte stays as registered. The crate's one
/// split tells the userinfo from the address ([`split`]): an address it
/// cannot split is told as [`UNPARSABLE`], the marker a log line writes for
/// it. The scheme is told beside the host, and the split reads one by its
/// form alone: any other text before a `://` is the address's own, read
/// for its userinfo with the rest.
///
/// The userinfo is the one credential a URL carries by its form. A secret a
/// registrant wrote into a path or a query is not known as one here.
pub(crate) fn without_userinfo(address: &str) -> Option<String> {
    let Some(Address {
        scheme,
        userinfo,
        host,
        tail,
    }) = split(address)
    else {
        return Some(UNPARSABLE.to_owned());
    };
    userinfo.then(|| format!("{scheme}{host}{tail}"))
}

/// The bytes the JSON writer writes for `value`, written compact. The
/// count goes through the writer itself, so it holds no second copy of the
/// JSON's layout. No value a page counts fails to write: its maps are
/// keyed by texts, and the writer that counts takes every byte. A count
/// that failed all the same reads as the largest number, past the page's
/// bound.
fn written<T: Serialize>(value: &T) -> usize {
    let mut counted = Counted(0);
    match serde_json::to_writer(&mut counted, value) {
        Ok(()) => counted.0,
        Err(_) => usize::MAX,
    }
}

/// A writer that counts the bytes it is given and keeps none.
struct Counted(usize);

impl std::io::Write for Counted {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self.0.saturating_add(bytes.len());
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/document/operator/snapshot/tests.rs"]
mod tests;
