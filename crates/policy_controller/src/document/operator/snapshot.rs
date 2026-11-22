//! The worker rows an operator reads, in registration key order.
//!
//! A caller can ask for the workers serving particular models and cap the
//! row count. A page also has a byte bound. Each row's registration and
//! engine addresses, and its provider's key, are told without their
//! userinfo; the store keeps the originals.
//!
//! A page writes itself as JSON ([`WorkerRowsPage`]): a host in process serves
//! the rows an operator reads with no mapping of its own.

use std::{borrow::Cow, collections::BTreeMap, sync::Arc, time::Instant};

use serde::{Serialize, Serializer};

use crate::{
    address::{split, Address, UNPARSABLE},
    engine::view::{Role, WireModes},
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
/// The store's lone read matches by the same rule, searched in a sorted
/// list ([`crate::store::MemoryStore::rows_serving`]): the two change
/// together.
pub(crate) fn serves(models: &[String], row: &WorkerRow) -> bool {
    models.is_empty() || row.facts.models.iter().any(|model| models.contains(model))
}

/// The store rows read for a snapshot.
pub enum Rows {
    /// A shared cut, ordered by registration key, before model filtering.
    Shared(Arc<StoreCheckpoint>),
    /// The first serving rows in key order and the total that serve,
    /// as returned by [`crate::store::MemoryStore::rows_serving`].
    Told(Vec<WorkerRow>, usize),
}

/// A page of worker rows as an operator reads it.
#[derive(Debug)]
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
/// encode handoff.
impl Serialize for WorkerRowsPage {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Page<'a> {
            rows: Vec<RowJson<'a>>,
            truncated: bool,
        }
        let rows = self.rows.iter().map(|row| RowJson::of(row, self.taken));
        Page {
            rows: rows.collect(),
            truncated: self.truncated,
        }
        .serialize(serializer)
    }
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
}

/// From where a row is reachable: every gateway, or the one that holds its
/// host-local wire, by the sixteen hex digits the contract tells.
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum ReachableJson {
    Any(bool),
    GatewayId(String),
}

/// What a row states of a pair and of an encode handoff, under the names
/// of the contract's pairing descriptor.
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
        }
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

/// Takes a redacted page at `now`, limited by row count and a conservative
/// measure of the rows' facts. A row admitted after `now` reads as new at
/// that instant. The first row must fit the byte bound too.
pub fn worker_rows_page(asked: &WorkerRowsQuery, read: Rows, now: Instant) -> WorkerRowsPage {
    let (rows, truncated) = worker_rows_page_with(asked, read, now, false, |row| {
        let bytes = measure(&row);
        (row.into_owned(), bytes)
    });
    WorkerRowsPage {
        rows,
        truncated,
        taken: now,
    }
}

/// Pages the redacted rows using the host's measure of the answer. A host
/// that sends an oversized first row alone can keep that rule here.
pub(crate) fn worker_rows_page_with<T>(
    asked: &WorkerRowsQuery,
    read: Rows,
    now: Instant,
    allow_oversized_first: bool,
    map: impl Fn(Cow<'_, WorkerRow>) -> (T, usize),
) -> (Vec<T>, bool) {
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
            told,
        ),
        Rows::Told(rows, serving) => page(
            asked.limit,
            rows.into_iter().map(Cow::Owned),
            Some(serving),
            allow_oversized_first,
            told,
        ),
    }
}

/// A single pass over the serving rows, stopping at the first bound.
/// The store supplies the serving count when it has already selected a
/// prefix. A shared cut needs just one more matching row to say it was cut.
fn page<'a, T>(
    limit: usize,
    mut rows: impl Iterator<Item = Cow<'a, WorkerRow>>,
    serving: Option<usize>,
    allow_oversized_first: bool,
    told: impl Fn(Cow<'a, WorkerRow>) -> (T, usize),
) -> (Vec<T>, bool) {
    let mut page = Vec::new();
    let mut bytes: usize = 0;
    for row in rows.by_ref().take(limit) {
        let (row, size) = told(row);
        bytes = bytes.saturating_add(size);
        if bytes > PAGE_BYTES && !(allow_oversized_first && page.is_empty()) {
            return (page, true);
        }
        page.push(row);
    }
    let truncated = serving.map_or_else(|| rows.next().is_some(), |count| page.len() < count);
    (page, truncated)
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

/// Count each text with room for its framing and allow 512 bytes for
/// fixed fields and the row envelope. Count the addresses and facts that
/// can be told twice and include the pairing and encode descriptors.
fn measure(row: &WorkerRow) -> usize {
    let facts = &row.facts;
    let text = |text: &str| text.len().saturating_add(16);
    let mut bytes = 512usize
        .saturating_add(text(&facts.key).saturating_mul(2))
        .saturating_add(text(&facts.engine))
        .saturating_add(text(&facts.engine_kind).saturating_mul(2));
    for field in [
        facts.zone.as_deref(),
        facts.labels.get("policy").map(String::as_str),
        facts.encode.connector.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        bytes = bytes.saturating_add(text(field));
    }
    if let Registrant::Provider(maker) = &row.maker {
        bytes = bytes
            .saturating_add(text(maker.provider()))
            .saturating_add(text(maker.key()));
    }
    if let Some(pairing) = &facts.pairing {
        for field in [&pairing.protocol, &pairing.transport, &pairing.version]
            .into_iter()
            .chain(&pairing.kv_layout)
            .flatten()
        {
            bytes = bytes.saturating_add(text(field));
        }
    }
    for model in &facts.models {
        bytes = bytes.saturating_add(text(model));
    }
    for (name, value) in &facts.labels {
        bytes = bytes
            .saturating_add(text(name))
            .saturating_add(text(value))
            .saturating_add(16);
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An address is told without its userinfo and with every other byte as
    /// registered. One with no userinfo is told as it is. One whose `@`
    /// stands past the authority's end is told by the marker: no part of
    /// what may be a userinfo is told.
    #[test]
    fn an_address_is_told_without_its_userinfo() {
        for (address, told) in [
            (
                "grpc://user:secret@10.0.0.5:8000",
                Some("grpc://10.0.0.5:8000"),
            ),
            (
                "https://user@host.example:8443/v1?x=1#top",
                Some("https://host.example:8443/v1?x=1#top"),
            ),
            ("user:secret@10.0.0.5:8000", Some("10.0.0.5:8000")),
            // An `@` inside the userinfo: the authority's last one ends it.
            (
                "grpc://us@er:se@cret@10.0.0.5:8000",
                Some("grpc://10.0.0.5:8000"),
            ),
            (
                "grpc+tls://user@10.0.0.5:8000",
                Some("grpc+tls://10.0.0.5:8000"),
            ),
            // A `://` behind the authority is no scheme's: the authority
            // starts at the address's first byte, and its userinfo goes.
            (
                "user:secret@10.0.0.5:8000/next/http://other",
                Some("10.0.0.5:8000/next/http://other"),
            ),
            (
                "user:secret@10.0.0.5:8000?next=http://other",
                Some("10.0.0.5:8000?next=http://other"),
            ),
            (
                "user:secret@10.0.0.5:8000#http://other",
                Some("10.0.0.5:8000#http://other"),
            ),
            (
                "grpc://user:secret@10.0.0.5:8000?next=http://other",
                Some("grpc://10.0.0.5:8000?next=http://other"),
            ),
            // No userinfo: every byte stays.
            ("grpc://10.0.0.5:8000", None),
            ("10.0.0.5:8000?next=http://other", None),
            ("http://[::1]:8000/v1?x=1", None),
            ("10.0.0.5:8000", None),
            ("1http://10.0.0.5:8000", None),
            ("", None),
            // A userinfo that holds a raw `/`, `?` or `#` reads like these,
            // and so does a path that holds an `@`.
            ("http://user:se/cret@10.0.0.5:8000", Some(UNPARSABLE)),
            ("http://user:se?cret@10.0.0.5:8000", Some(UNPARSABLE)),
            ("http://user:se#cret@10.0.0.5:8000", Some(UNPARSABLE)),
            ("http://10.0.0.5:8000/a@b", Some(UNPARSABLE)),
            // No scheme by its form stands before `://`: its `/` ends the
            // authority, and the `@` behind it stands past that end.
            ("user:se://cret@10.0.0.5:8000", Some(UNPARSABLE)),
            ("1http://user:secret@10.0.0.5:8000", Some(UNPARSABLE)),
        ] {
            assert_eq!(without_userinfo(address).as_deref(), told, "{address}");
        }
    }

    #[test]
    fn a_plain_page_stops_at_its_bytes_even_for_its_first_row() {
        use crate::store::{MemoryStore, WorkerFacts, WorkersStore};

        let store = MemoryStore::new();
        let asked = WorkerRowsQuery {
            models: Vec::new(),
            limit: usize::MAX,
        };
        let now = Instant::now();
        for n in 0..2 {
            let key = format!("grpc://{n}.{}", "x".repeat((2 * 1024 * 1024) / 4));
            store.upsert(WorkerFacts {
                key: key.clone(),
                engine: key,
                incarnation: 1,
                dp_size: 1,
                models: vec!["m".to_owned()],
                ..WorkerFacts::default()
            });
        }
        let read = || store.rows_serving(&asked.models, asked.limit);
        let (rows, serving) = read();
        let page = worker_rows_page(&asked, Rows::Told(rows, serving), now);
        assert_eq!(page.rows.len(), 1);
        assert!(page.truncated);

        let (mut rows, _) = read();
        rows.truncate(1);
        rows[0].facts.key = "x".repeat(2 * 1024 * 1024);
        let too_large = worker_rows_page(&asked, Rows::Told(rows, 1), now);
        assert!(too_large.rows.is_empty() && too_large.truncated);
    }

    /// A page's JSON, literally: a row that states every fact, a
    /// provider's among them with a credential in its key and in its own,
    /// and a row that states the least, admitted after the take.
    #[test]
    fn a_page_writes_its_rows_as_an_operator_reads_them() {
        use std::time::Duration;

        use crate::{
            engine::view::WorkerId,
            store::{
                pairing::PairingDescriptor, EncodeOffer, GatewayId, Overload, Signals, WorkerFacts,
                WorkerStatus,
            },
        };

        let admitted = Instant::now();
        let taken = admitted + Duration::from_millis(1500);
        let full = WorkerRow {
            id: WorkerId(0xab),
            facts: WorkerFacts {
                key: "grpc://user:secret@10.0.0.5:8000".to_owned(),
                engine: "grpc://user:secret@10.0.0.5:8000".to_owned(),
                incarnation: 7,
                role: Role::Prefill,
                wire_modes: WireModes::from_bits(3),
                engine_kind: "runtime-a".to_owned(),
                models: vec!["m1".to_owned(), "m2".to_owned()],
                dp_rank: 1,
                dp_size: 2,
                labels: BTreeMap::from([("policy".to_owned(), "default".to_owned())]),
                zone: Some("zone-a".to_owned()),
                pairing: Some(Arc::new(PairingDescriptor {
                    protocol: Some("p1".to_owned()),
                    transport: Some("t1".to_owned()),
                    version: Some("v1".to_owned()),
                    kv_layout: [
                        Some("float16".to_owned()),
                        None,
                        Some("a1".to_owned()),
                        None,
                    ],
                })),
                encode: EncodeOffer {
                    handoff: EncodeHandoff::Push,
                    consumer: true,
                    connector: Some("c1".to_owned()),
                    ..EncodeOffer::default()
                },
                vision: true,
                max_running: Some(8),
                kv_block_size: Some(16),
                bootstrap_port: Some(9000),
                priority: 5,
                reachable_from: Reachability::Gateway(GatewayId(0x1f)),
                signals: Signals {
                    events: true,
                    loads: true,
                    scrape_interval_ms: 0,
                },
                overload: Overload {
                    kv_usage_permille: Some(900),
                    waiting_requests: Some(64),
                },
                ..WorkerFacts::default()
            },
            status: WorkerStatus::Ready,
            revision: 3,
            admitted_at: admitted,
            maker: Registrant::provider("finder", "http://u:p@host/x").expect("a maker"),
        };
        let least = WorkerRow {
            id: WorkerId(1),
            facts: WorkerFacts {
                key: "http://b:1".to_owned(),
                engine: "http://b:1".to_owned(),
                incarnation: 1,
                dp_size: 1,
                ..WorkerFacts::default()
            },
            status: WorkerStatus::Pending,
            revision: 1,
            admitted_at: taken + Duration::from_secs(2),
            maker: Registrant::Control,
        };
        let asked = WorkerRowsQuery {
            models: Vec::new(),
            limit: usize::MAX,
        };
        let page = worker_rows_page(&asked, Rows::Told(vec![full, least], 2), taken);
        let probes = serde_json::json!({
            "failure_threshold": 3,
            "success_threshold": 2,
            "check_interval_ms": 60_000,
            "timeout_ms": 5_000,
            "disabled": false
        });
        let expected = serde_json::json!({
            "rows": [
                {
                    "id": "00000000000000ab",
                    "registration_key": "grpc://10.0.0.5:8000",
                    "incarnation": 7,
                    "engine": "grpc://10.0.0.5:8000",
                    "wire": ["http", "grpc"],
                    "role": "prefill",
                    "runtime": "runtime-a",
                    "models": ["m1", "m2"],
                    "dp_rank": 1,
                    "dp_size": 2,
                    "labels": {"policy": "default"},
                    "zone": "zone-a",
                    "status": "ready",
                    "revision": 3,
                    "reachable": {"gateway_id": "000000000000001f"},
                    "pairing": {
                        "protocol": "p1",
                        "transport": "t1",
                        "version": "v1",
                        "bootstrap_port": 9000,
                        "kv_layout": {"kv_cache_dtype": "float16", "attention_backend": "a1"},
                        "ec_producer": false,
                        "ec_consumer": true,
                        "vision": true,
                        "encode_handoff": "push",
                        "ec_connector": "c1",
                        "encode_references": false
                    },
                    "max_running": 8,
                    "kv_block_size": 16,
                    "priority": 5,
                    "overload": {"queue_limit": 64, "kv_usage_permille": 900},
                    "signals": {"events": true, "loads": true, "scrape_interval_ms": 0},
                    "health_policy": probes,
                    "admitted_age_ms": 1500,
                    "maker": {"provider": "finder", "key": "http://host/x"}
                },
                {
                    "id": "0000000000000001",
                    "registration_key": "http://b:1",
                    "incarnation": 1,
                    "engine": "http://b:1",
                    "wire": [],
                    "role": "single",
                    "runtime": "",
                    "models": [],
                    "dp_rank": 0,
                    "dp_size": 1,
                    "labels": {},
                    "zone": null,
                    "status": "pending",
                    "revision": 1,
                    "reachable": {"any": true},
                    "pairing": null,
                    "max_running": null,
                    "kv_block_size": null,
                    "priority": 0,
                    "overload": {"queue_limit": null, "kv_usage_permille": null},
                    "signals": {"events": false, "loads": false, "scrape_interval_ms": 0},
                    "health_policy": probes,
                    "admitted_age_ms": 0,
                    "maker": null
                }
            ],
            "truncated": false
        });
        assert_eq!(
            serde_json::to_value(&page).expect("a page's JSON"),
            expected
        );
    }
}
