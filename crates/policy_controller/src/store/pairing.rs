//! The pair classes (design 4.1 and 5.1): which decodes a prefill may hand
//! its KV blocks to, decided once per membership write and read by every
//! pair decision as a class on its rows and a table of classes.
//!
//! One invariant: **a worker has one class, and which classes pair is a
//! table of classes.** Two prefills that may pair with the same decodes
//! share one class, and so do two decodes that may pair with the same
//! prefills, whatever else tells them apart. A worker stands once in its
//! pool: a decode that reports no transport pairs, leniently, with prefills
//! of two transports, and is still one row of one class, which both
//! prefill classes list ([`PairTable`]). The table names classes, never
//! rows, so it grows with the classes that differ and a pool never does.
//!
//! What a pair is compared on is the rule of [`mismatch`] under one of
//! three modes ([`Pairing`]): a runtime, an operator's protocol, a KV
//! transport, four KV layout facts and an engine version, each a fact its
//! worker reports or leaves unknown. The mode judges the KV handoff alone.
//! What a wire cannot carry at all is that wire's fact and no mode switches
//! it off ([`runtime_specific`]): where a pair's requests are built for one
//! runtime, two known runtimes never pair.
//!
//! The derivation ([`PairClasses::derive`]) is a function of the membership
//! and the mode and of nothing else. It numbers each side's classes in the
//! order that side's workers are given, which the store makes the workers'
//! id order, so two stores that hold one membership number its classes
//! alike.
//!
//! Every value here is opaque text: a runtime, a transport and a layout
//! fact are compared for equality and never interpreted. Deriving a
//! descriptor from what an engine reports is the registration handshake's.

use std::fmt;

use rustc_hash::FxHashMap;

use crate::{
    framework::PairTable,
    spec::Pairing,
    view::{WireModes, WorkerId},
};

#[cfg(test)]
mod tests;

/// The KV layout facts a descriptor may report, in the order they are
/// compared: the short name a pairing key spells, then the name a mismatch
/// reports.
pub const LAYOUT_FACTS: [(&str, &str); 4] = [
    ("dtype", "kv_cache_dtype"),
    ("page", "page_size"),
    ("attn", "attention_backend"),
    ("model", "model_dtype"),
];

/// What one prefill or decode worker offers a rendezvous partner, beside
/// its runtime (the row's engine kind). A field left `None` is one the
/// worker does not report.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PairingDescriptor {
    /// The protocol an operator set. Two workers that both set one are
    /// compared on it whole.
    pub protocol: Option<String>,
    /// The KV transport.
    pub transport: Option<String>,
    /// The engine's version, compared under [`Pairing::Strict`] alone.
    pub version: Option<String>,
    /// The KV layout facts, in the order of [`LAYOUT_FACTS`].
    pub kv_layout: [Option<String>; 4],
}

/// The descriptor of a worker that has none: it reports nothing.
static NO_DESCRIPTOR: PairingDescriptor = PairingDescriptor {
    protocol: None,
    transport: None,
    version: None,
    kv_layout: [None, None, None, None],
};

/// One prefill or decode worker as the pairing rule reads it.
#[derive(Clone, Copy, Debug)]
pub struct Member<'a> {
    pub id: WorkerId,
    /// The wires the worker serves.
    pub wires: WireModes,
    /// The worker's runtime; empty when it is unknown.
    pub runtime: &'a str,
    pub descriptor: &'a PairingDescriptor,
}

/// What a worker offers a partner, whole: its side of the wire's fact
/// ([`runtime_specific`]), its runtime when it is known and its descriptor.
/// Two workers of one offer pair alike with every worker, so the rule runs
/// once a pair of offers and never once a pair of workers.
///
/// An unknown runtime is `None` here ([`known`]): the offers are numbered
/// with a compare a member.
type Offer<'a> = (bool, Option<&'a str>, &'a PairingDescriptor);

/// A runtime as an offer reads it: `None` when it is unknown, which a row
/// writes as the empty string. Two unknown runtimes are compared so and
/// never as strings: the C library's compare of two empty strings that
/// hold no buffer takes a slow path (about 200 ns by a probe, where two
/// strings of four bytes take 3).
pub(crate) fn known(runtime: &str) -> Option<&str> {
    (!runtime.is_empty()).then_some(runtime)
}

impl<'a> Member<'a> {
    /// A member from the facts of its row: a row with no descriptor reports
    /// nothing.
    pub fn new(
        id: WorkerId,
        wires: WireModes,
        runtime: &'a str,
        descriptor: Option<&'a PairingDescriptor>,
    ) -> Self {
        Self {
            id,
            wires,
            runtime,
            descriptor: descriptor.unwrap_or(&NO_DESCRIPTOR),
        }
    }

    /// The runtime, when it is known.
    fn known_runtime(&self) -> Option<&'a str> {
        known(self.runtime)
    }

    fn offer(&self) -> Offer<'a> {
        (
            runtime_specific(self.wires),
            self.known_runtime(),
            self.descriptor,
        )
    }
}

/// The component two workers cannot pair on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Mismatch {
    /// Two known runtimes differ.
    Runtime,
    /// Both sides set a protocol and the two differ.
    Explicit,
    /// Two known KV transports differ.
    Transport,
    /// Two reported engine versions differ, under [`Pairing::Strict`] alone.
    Version,
    /// Both sides report the named KV layout fact and disagree.
    KvLayout(&'static str),
    /// A component one side does not report, refused under
    /// [`Pairing::Strict`] alone.
    Unknown(&'static str),
}

/// The component in the words a refusal reports it.
impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime => f.write_str("runtime"),
            Self::Explicit => f.write_str("pairing_protocol"),
            Self::Transport => f.write_str("transport"),
            Self::Version => f.write_str("version"),
            Self::KvLayout(fact) => f.write_str(fact),
            Self::Unknown(component) => write!(f, "unknown {component}"),
        }
    }
}

/// How the two sides of a would-be pair report one component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Agreement {
    /// Both report it, the same.
    Agree,
    /// Both report it, differently.
    Differ,
    /// One side reports it, the other does not.
    OneSided,
    /// Neither reports it.
    Unreported,
}

/// What it means when neither side of a pair reports a component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unreported {
    /// The component always exists (a runtime, a transport): two workers
    /// that name neither leave it unknown, which only strict refuses.
    Unknown,
    /// The workers may not carry the fact (a KV layout fact): nothing to
    /// compare.
    NotAFact,
}

impl Agreement {
    fn of(mine: Option<&str>, theirs: Option<&str>) -> Self {
        match (mine, theirs) {
            (Some(a), Some(b)) if a == b => Self::Agree,
            (Some(_), Some(_)) => Self::Differ,
            (None, None) => Self::Unreported,
            _ => Self::OneSided,
        }
    }

    /// The verdict on a hard fact: a known difference is `differ`; a side
    /// that does not report the fact leaves it unknown, which only strict
    /// refuses; a fact nobody reports is what `unreported` says of the
    /// component.
    fn hard_fact(
        self,
        differ: Mismatch,
        name: &'static str,
        strict: bool,
        unreported: Unreported,
    ) -> Option<Mismatch> {
        let unknown = strict.then_some(Mismatch::Unknown(name));
        match self {
            Self::Differ => Some(differ),
            Self::OneSided => unknown,
            Self::Unreported if unreported == Unreported::Unknown => unknown,
            Self::Agree | Self::Unreported => None,
        }
    }
}

/// How the two workers' runtimes compare. An unknown runtime is not a
/// different one.
fn runtimes(prefill: &Member<'_>, decode: &Member<'_>) -> Agreement {
    Agreement::of(prefill.known_runtime(), decode.known_runtime())
}

/// The first component the KV handoff of `prefill` to `decode` fails on
/// under `mode`, if any.
///
/// Runtime, transport and each KV layout fact are hard facts: a known
/// difference refuses the pair in every mode but off, a fact one side
/// reports and the other does not only under strict, and a layout fact
/// neither side reports is not compared. Two protocols set by operators are
/// compared whole and settle the pair when equal; one set on one side alone
/// cannot be checked against reported facts, so strict refuses it and
/// lenient falls through to the facts. The version is a soft fact: workers
/// report it unevenly and a rolling upgrade mixes versions, so it counts
/// under strict alone, and only when both sides report one.
pub fn mismatch(prefill: &Member<'_>, decode: &Member<'_>, mode: Pairing) -> Option<Mismatch> {
    if mode == Pairing::Off {
        return None;
    }
    let strict = mode == Pairing::Strict;
    if let Some(mismatch) = runtimes(prefill, decode).hard_fact(
        Mismatch::Runtime,
        "runtime",
        strict,
        Unreported::Unknown,
    ) {
        return Some(mismatch);
    }
    let (mine, theirs) = (prefill.descriptor, decode.descriptor);
    match Agreement::of(mine.protocol.as_deref(), theirs.protocol.as_deref()) {
        Agreement::Agree => return None,
        Agreement::Differ => return Some(Mismatch::Explicit),
        Agreement::OneSided if strict => return Some(Mismatch::Unknown("pairing_protocol")),
        Agreement::OneSided | Agreement::Unreported => {}
    }
    if let Some(mismatch) = Agreement::of(mine.transport.as_deref(), theirs.transport.as_deref())
        .hard_fact(
            Mismatch::Transport,
            "transport",
            strict,
            Unreported::Unknown,
        )
    {
        return Some(mismatch);
    }
    let facts = mine.kv_layout.iter().zip(&theirs.kv_layout);
    for (&(_, name), (mine, theirs)) in LAYOUT_FACTS.iter().zip(facts) {
        if let Some(mismatch) = Agreement::of(mine.as_deref(), theirs.as_deref()).hard_fact(
            Mismatch::KvLayout(name),
            name,
            strict,
            Unreported::NotAFact,
        ) {
            return Some(mismatch);
        }
    }
    let versions = Agreement::of(mine.version.as_deref(), theirs.version.as_deref());
    (strict && versions == Agreement::Differ).then_some(Mismatch::Version)
}

/// Whether a pair drawn on `wires` must share a runtime: the wire's fact,
/// in its one home.
///
/// The HTTP wire forwards one body to both sides of a pair and has no such
/// fact. Every other wire builds a pair's requests in the message type of
/// one runtime, so a pair of two known runtimes cannot be dispatched on it
/// at all. A registration names one wire and a row's type holds a set, so
/// the function is total: a set that holds HTTP and one wire more is on the
/// runtime-specific side, as the empty set is, and a worker on that side
/// never pairs with an HTTP-only one.
pub const fn runtime_specific(wires: WireModes) -> bool {
    wires.bits() != WireModes::HTTP.bits()
}

/// Why `prefill` and `decode`, two workers on one side of
/// [`runtime_specific`], cannot pair under `mode`: the wire's own fact
/// first, then the mode's rule over the handoff ([`mismatch`]).
///
/// The mode says how much of the KV handoff is compared, and off compares
/// none of it. A known runtime difference on a runtime-specific wire is not
/// the handoff's fact but the wire's, so no mode switches it off. An
/// unknown runtime is not a different one and pairs as the mode says.
pub fn wire_mismatch(prefill: &Member<'_>, decode: &Member<'_>, mode: Pairing) -> Option<Mismatch> {
    if runtime_specific(prefill.wires) && runtimes(prefill, decode) == Agreement::Differ {
        return Some(Mismatch::Runtime);
    }
    mismatch(prefill, decode, mode)
}

/// The group a worker pairs in, as a refusal names it: the operator's
/// protocol, else `runtime/transport/layout`, with `?` for a transport or a
/// layout the worker does not report and `unspecified` for an unknown
/// runtime. The version is left out: it counts under strict alone, and a
/// rolling upgrade would show two keys for workers that pair.
pub fn key(member: &Member<'_>) -> String {
    let descriptor = member.descriptor;
    if let Some(protocol) = &descriptor.protocol {
        return protocol.clone();
    }
    let layout: Vec<String> = LAYOUT_FACTS
        .iter()
        .zip(&descriptor.kv_layout)
        .filter_map(|(&(short, _), value)| value.as_ref().map(|value| format!("{short}={value}")))
        .collect();
    format!(
        "{}/{}/{}",
        member.known_runtime().unwrap_or("unspecified"),
        descriptor.transport.as_deref().unwrap_or("?"),
        if layout.is_empty() {
            "?".to_string()
        } else {
            layout.join(",")
        },
    )
}

/// Why no prefill on one side of [`runtime_specific`] can pair with any
/// decode on it: each pool's distinct pairing keys and the components the
/// two pools disagree on, each list sorted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub prefill: Vec<String>,
    pub decode: Vec<String>,
    pub mismatches: Vec<String>,
}

/// The pair classes of one model's prefill and decode pools, and which of
/// them pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairClasses {
    /// Each prefill with its class, in the order given.
    prefills: Box<[(WorkerId, u32)]>,
    /// Each decode with its class, in the order given.
    decodes: Box<[(WorkerId, u32)]>,
    /// Where prefill class `k`'s list ends in `listed`, at `k - 1`.
    ends: Box<[u32]>,
    /// The lists one after another: the decode classes each prefill class
    /// pairs with, ascending.
    listed: Box<[u32]>,
    /// The refusal on each side of [`runtime_specific`]: the plain side
    /// first.
    refusals: [Option<Refusal>; 2],
}

impl PairClasses {
    /// Derive the pair classes of one model's `prefills` and `decodes`
    /// under `mode`.
    ///
    /// A prefill pairs with every decode on its side of
    /// [`runtime_specific`] that [`wire_mismatch`] does not refuse. The
    /// prefills that pair with the same decodes are one prefill class, and
    /// the decodes that pair with the same prefills are one decode class:
    /// workers of two descriptors that pair alike share a class. Each
    /// side's classes are numbered from `1` in the order their first worker
    /// is given, and a worker that pairs with nobody is class `0` and takes
    /// no number. The table lists, for each prefill class, the decode
    /// classes it pairs with: a list names every decode class, so the
    /// table numbers densely, as the engine asks of a host
    /// ([`PairTable::new`]). Where both pools hold workers on one side of
    /// the wire's fact and no pair is made there, that side's [`Refusal`]
    /// names the two pools' keys and what they disagree on.
    pub fn derive(prefills: &[Member<'_>], decodes: &[Member<'_>], mode: Pairing) -> Self {
        let prefill_offers = Offers::of(prefills);
        let decode_offers = Offers::of(decodes);
        let verdicts: Vec<Vec<Verdict>> = prefill_offers
            .first
            .iter()
            .map(|prefill| {
                let beside = |decode| Verdict::of(prefill, decode, mode);
                decode_offers.first.iter().map(beside).collect()
            })
            .collect();
        // One row a prefill offer: the decode offers it pairs with. One
        // prefill class a distinct row: the offers are numbered as their
        // first prefill comes, so the rows are read in that order.
        let rows: Vec<Vec<u32>> = verdicts
            .iter()
            .map(|verdicts| {
                let pairs = |&(_, verdict): &(u32, &Verdict)| *verdict == Verdict::Pairs;
                let offers = (0u32..).zip(verdicts).filter(pairs);
                offers.map(|(offer, _)| offer).collect()
            })
            .collect();
        let (prefill_class, firsts) = numbered(&rows);
        // One column a decode offer: the prefill classes it pairs with,
        // ascending as the classes are walked. One decode class a distinct
        // column.
        let mut columns: Vec<Vec<u32>> = vec![Vec::new(); decode_offers.first.len()];
        for (class, &first) in (1u32..).zip(&firsts) {
            for &offer in &rows[first] {
                columns[offer as usize].push(class);
            }
        }
        let (decode_class, _) = numbered(&columns);
        // The table: a prefill class lists the classes of the decode
        // offers its row names, each once.
        let mut ends = Vec::with_capacity(firsts.len());
        let mut listed = Vec::new();
        for &first in &firsts {
            let mut list: Vec<u32> = rows[first]
                .iter()
                .map(|&offer| decode_class[offer as usize])
                .collect();
            list.sort_unstable();
            list.dedup();
            listed.extend(list);
            ends.push(listed.len() as u32);
        }
        // The table is valid by construction, every decode class named by
        // some list: the question a host asks where it builds one.
        debug_assert_eq!(PairTable::new(&ends, &listed).validate(), Ok(()));
        let classed = |pool: &[Member<'_>], offers: &Offers<'_>, class: &[u32]| {
            let of = |(member, &offer): (&Member<'_>, &u32)| (member.id, class[offer as usize]);
            pool.iter().zip(&offers.of).map(of).collect()
        };
        let refusal = |side| refusal(side, &prefill_offers.first, &decode_offers.first, &verdicts);
        Self {
            prefills: classed(prefills, &prefill_offers, &prefill_class),
            decodes: classed(decodes, &decode_offers, &decode_class),
            ends: ends.into_boxed_slice(),
            listed: listed.into_boxed_slice(),
            refusals: [refusal(false), refusal(true)],
        }
    }

    /// Each prefill with its class, in the order given: `0` for a prefill
    /// no decode pairs with, else a class numbered from `1`.
    pub fn prefills(&self) -> &[(WorkerId, u32)] {
        &self.prefills
    }

    /// Each decode with its class, in the order given: `0` for a decode no
    /// prefill pairs with, else a class numbered from `1`.
    pub fn decodes(&self) -> &[(WorkerId, u32)] {
        &self.decodes
    }

    /// How many prefill classes there are: the distinct partner lists.
    pub fn classes(&self) -> u32 {
        self.ends.len() as u32
    }

    /// How many entries the lists hold together: with the classes'
    /// count, the table's size.
    pub fn entries(&self) -> usize {
        self.listed.len()
    }

    /// Which classes pair, as a pair decision reads it beside its rows.
    pub fn table(&self) -> PairTable<'_> {
        PairTable::new(&self.ends, &self.listed)
    }

    /// Write the table into `ends` and `listed`, each cleared first: a
    /// pair build hands a decision the table with its rows.
    pub(super) fn table_into(&self, ends: &mut Vec<u32>, listed: &mut Vec<u32>) {
        ends.clear();
        ends.extend_from_slice(&self.ends);
        listed.clear();
        listed.extend_from_slice(&self.listed);
    }

    /// The partner list of prefill class `class`: the decodes of the
    /// classes it pairs with, as indexes into the decodes given, in that
    /// order; empty for class `0` and for a class past the last. Built on
    /// each call: what is held is the table, a list never.
    pub fn list(&self, class: u32) -> Vec<u32> {
        let listed = self.table().of(&class);
        let partner = |&(_, &(_, decode)): &(u32, &(WorkerId, u32))| {
            decode != 0 && listed.binary_search(&decode).is_ok()
        };
        let partners = (0u32..).zip(self.decodes.iter()).filter(partner);
        partners.map(|(at, _)| at).collect()
    }

    /// How many pairs the classes hold: for each prefill, the decodes of
    /// its list. A table that held one list a prefill would be four bytes
    /// a pair; what is held is a class a worker and the table of classes.
    pub fn pairs(&self) -> usize {
        self.prefills
            .iter()
            .map(|&(_, class)| self.list(class).len())
            .sum()
    }

    /// Why no pair was made among the workers on `wires`' side of
    /// [`runtime_specific`]; `None` when one was, and when a pool holds no
    /// worker on that side.
    pub fn refusal(&self, wires: WireModes) -> Option<&Refusal> {
        self.refusals[usize::from(runtime_specific(wires))].as_ref()
    }
}

/// Numbers the distinct lists from `1` in the order each first comes; an
/// empty list is `0` and takes no number. Each list's number, and where the
/// first list of each number stands.
fn numbered(lists: &[Vec<u32>]) -> (Vec<u32>, Vec<usize>) {
    let mut numbers: FxHashMap<&[u32], u32> = FxHashMap::default();
    let mut firsts = Vec::new();
    let of = lists
        .iter()
        .enumerate()
        .map(|(at, list)| {
            if list.is_empty() {
                return 0;
            }
            *numbers.entry(list).or_insert_with(|| {
                firsts.push(at);
                firsts.len() as u32
            })
        })
        .collect();
    (of, firsts)
}

/// The distinct offers of one pool.
struct Offers<'a> {
    /// Each worker's offer, by number in the order the offers first come.
    of: Vec<u32>,
    /// The first worker of each offer.
    first: Vec<Member<'a>>,
}

impl<'a> Offers<'a> {
    fn of(pool: &[Member<'a>]) -> Self {
        let mut numbered: FxHashMap<Offer<'a>, u32> = FxHashMap::default();
        let mut first = Vec::new();
        // Neighbours in a pool mostly share an offer: a member whose offer
        // is the one before it takes that number by one compare, and the
        // map is asked where the two differ.
        let mut before: Option<(Offer<'a>, u32)> = None;
        let of = pool
            .iter()
            .map(|member| {
                let offer = member.offer();
                match before {
                    Some((same, number)) if same == offer => number,
                    _ => {
                        let number = *numbered.entry(offer).or_insert_with(|| {
                            first.push(*member);
                            first.len() as u32 - 1
                        });
                        before = Some((offer, number));
                        number
                    }
                }
            })
            .collect();
        Self { of, first }
    }
}

/// The rule's verdict on one prefill offer beside one decode offer.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Pairs,
    /// The two stand on two sides of the wire's fact: no pair is drawn
    /// across it, and nothing is compared.
    Apart,
    Refused(Mismatch),
}

impl Verdict {
    fn of(prefill: &Member<'_>, decode: &Member<'_>, mode: Pairing) -> Self {
        if runtime_specific(prefill.wires) != runtime_specific(decode.wires) {
            return Self::Apart;
        }
        wire_mismatch(prefill, decode, mode).map_or(Self::Pairs, Self::Refused)
    }
}

/// Why the `prefills` and the `decodes` on one `side` of the wire's fact
/// make no pair, each pool given as its distinct offers with the rule's
/// `verdicts` on them; `None` when a pair is made there, and when a pool
/// holds nobody on that side.
fn refusal(
    side: bool,
    prefills: &[Member<'_>],
    decodes: &[Member<'_>],
    verdicts: &[Vec<Verdict>],
) -> Option<Refusal> {
    let on_side = |member: &Member<'_>| runtime_specific(member.wires) == side;
    if !prefills.iter().any(on_side) || !decodes.iter().any(on_side) {
        return None;
    }
    let mut refused: Vec<Mismatch> = Vec::new();
    for (prefill, verdicts) in prefills.iter().zip(verdicts) {
        if !on_side(prefill) {
            continue;
        }
        for verdict in verdicts {
            match *verdict {
                Verdict::Pairs => return None,
                Verdict::Refused(mismatch) => refused.push(mismatch),
                Verdict::Apart => {}
            }
        }
    }
    refused.sort_unstable();
    refused.dedup();
    let mut mismatches: Vec<String> = refused.iter().map(Mismatch::to_string).collect();
    mismatches.sort_unstable();
    Some(Refusal {
        prefill: keys(prefills.iter().filter(|member| on_side(member))),
        decode: keys(decodes.iter().filter(|member| on_side(member))),
        mismatches,
    })
}

/// The distinct pairing keys of some workers, sorted.
fn keys<'a>(members: impl Iterator<Item = &'a Member<'a>>) -> Vec<String> {
    let mut keys: Vec<String> = members.map(key).collect();
    keys.sort_unstable();
    keys.dedup();
    keys
}
