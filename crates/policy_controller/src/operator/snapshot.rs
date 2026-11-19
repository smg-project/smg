//! The worker rows an operator reads, in registration key order.
//!
//! A caller can ask for the workers serving particular models and cap the
//! row count. A page also has a byte bound. Each row's registration and
//! engine addresses, and its provider's key, are told without their
//! userinfo; the store keeps the originals.

use std::{borrow::Cow, sync::Arc, time::Instant};

use crate::{
    address::{split, Address, UNPARSABLE},
    store::{Cut, Maker, WorkerRow},
};

/// The byte bound for one page of rows.
pub(crate) const PAGE_BYTES: usize = 2 * 1024 * 1024;

/// What a snapshot caller asks of the worker rows.
#[derive(Debug)]
pub struct Asked {
    /// The model ids to include. An empty list includes every model.
    pub models: Vec<String>,
    /// The maximum number of rows to return. `usize::MAX` uses only the
    /// byte bound.
    pub limit: usize,
}

impl Asked {
    pub(crate) fn serves(&self, row: &WorkerRow) -> bool {
        serves(&self.models, row)
    }
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
    Shared(Arc<Cut>),
    /// The first serving rows in key order and the total that serve,
    /// as returned by [`crate::store::MemoryStore::rows_serving`].
    Told(Vec<WorkerRow>, usize),
}

/// A page of worker rows as an operator reads it.
#[derive(Debug)]
pub struct Told {
    /// The matching rows in registration key order, with addresses redacted.
    pub rows: Vec<WorkerRow>,
    /// Whether the row count or byte bound left a matching row out.
    pub truncated: bool,
}

/// Takes a redacted page at `now`, limited by row count and a conservative
/// measure of the rows' facts. A row admitted after `now` reads as new at
/// that instant. The first row must fit the byte bound too.
pub fn told(asked: &Asked, read: Rows, now: Instant) -> Told {
    let (rows, truncated) = told_with(asked, read, now, false, |row| {
        let bytes = measure(&row);
        (row.into_owned(), bytes)
    });
    Told { rows, truncated }
}

/// Pages the redacted rows using the host's measure of the answer. A host
/// that sends an oversized first row alone can keep that rule here.
pub(crate) fn told_with<T>(
    asked: &Asked,
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
/// A provider's row tells its maker's key too: the provider's own text,
/// which may hold an address with a credential, so it is read as the row's
/// addresses are.
fn redacted(mut row: Cow<'_, WorkerRow>, now: Instant) -> Cow<'_, WorkerRow> {
    if let Some(key) = without_userinfo(&row.facts.key) {
        row.to_mut().facts.key = key;
    }
    if let Some(engine) = without_userinfo(&row.facts.engine) {
        row.to_mut().facts.engine = engine;
    }
    let maker = match &row.maker {
        Maker::Provider(maker) => {
            without_userinfo(maker.key()).map(|key| told_maker(maker.provider(), key))
        }
        Maker::Control => None,
    };
    if let Some(maker) = maker {
        row.to_mut().maker = maker;
    }
    if row.admitted_at > now {
        row.to_mut().admitted_at = now;
    }
    row
}

/// The maker a row tells in place of one whose key held a userinfo: the
/// provider's name with the clean key. A key that was all userinfo cleans
/// to no text, which no maker holds ([`Maker::provider`]), so it reads as
/// [`UNPARSABLE`]. The marker is inside a key's rule: the last arm is
/// never told.
fn told_maker(provider: &str, key: String) -> Maker {
    Maker::provider(provider, key)
        .or_else(|_| Maker::provider(provider, UNPARSABLE))
        .unwrap_or_default()
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
    if let Maker::Provider(maker) = &row.maker {
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
        let asked = Asked {
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
        let page = told(&asked, Rows::Told(rows, serving), now);
        assert_eq!(page.rows.len(), 1);
        assert!(page.truncated);

        let (mut rows, _) = read();
        rows.truncate(1);
        rows[0].facts.key = "x".repeat(2 * 1024 * 1024);
        let too_large = told(&asked, Rows::Told(rows, 1), now);
        assert!(too_large.rows.is_empty() && too_large.truncated);
    }
}
