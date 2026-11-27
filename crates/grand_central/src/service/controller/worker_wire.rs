//! The one conversion between the store's worker rows and the contract's
//! (design 5.2): a registration call's answer, a watch's page and the
//! replication stream carry one row. One half takes the store's row to the
//! wire's, and a registration or a patch off the wire into the store's
//! words; the standby's half takes the leader's rows, events and snapshot
//! back into the store's.
//!
//! A request field the store cannot hold is refused by its path
//! ([`Refused`]), never dropped. So is a field past its bound, and last a
//! row past [`ROW_BYTES`] whole: an answer's two rows and a snapshot's page
//! then fit a frame.

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use prost::Message as _;

use crate::{
    document::operator::snapshot,
    engine::{
        view::{Role, WireModes, WorkerId},
        DemandFacts,
    },
    service::{
        proto::{self, worker_event::Event},
        replication::{Entry, Event as Written},
    },
    store::{
        membership::blank, EncodeHandoff, EncodeOffer, GatewayId, HealthPolicy, MembershipChange,
        Overload, Position, ProviderKeyError, Reachability, Registrant, Signals, WorkerFacts,
        WorkerPatch, WorkerRow, WorkerStatus,
    },
};

/// A request field the door refuses: its path in the request, and why.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Refused {
    pub(crate) path: &'static str,
    pub(crate) why: &'static str,
}

const fn refused(path: &'static str, why: &'static str) -> Refused {
    Refused { path, why }
}

/// What a new row names itself: its wire, and its role.
const NEW_WIRE: Refused = refused("spec.wire", "a new row names its one wire");
const NEW_ROLE: Refused = refused("spec.role", "a new row names its role");

/// The models a watch names, held to a row's own bounds: the list a peer
/// sends is kept for its stream's life and read at each row.
pub(crate) fn watched(models: &[String]) -> Result<(), Refused> {
    if models.len() > MODELS {
        return Err(refused("models", "a watch names at most 128 model ids"));
    }
    match snapshot::unnamed(models) {
        Some(why) => Err(refused("models", why)),
        None => Ok(()),
    }
}

/// Whether `refusal` is a rule of a new row alone, and so no rule of a call
/// that makes no row.
pub(crate) fn of_a_new_row(refusal: &Refused) -> bool {
    *refusal == NEW_WIRE || *refusal == NEW_ROLE
}

/// The bounds of what a row states (contract: `WorkerSpec`). A replacement's
/// answer carries two rows and a snapshot pages at 128 rows. Both fit a
/// frame of 4 MiB.
const URL_BYTES: usize = WorkerFacts::URL_BYTES;
const NAME_BYTES: usize = WorkerFacts::NAME_BYTES;
const KEY_BYTES: usize = WorkerFacts::LABEL_KEY_BYTES;
const VALUE_BYTES: usize = WorkerFacts::LABEL_VALUE_BYTES;
const LABELS: usize = WorkerFacts::LABELS;
/// The most model ids a watch or snapshot request can select.
pub(super) const MODELS: usize = snapshot::MODELS;
/// The row whole on the wire, as [`weight`] measures it.
const ROW_BYTES: usize = WorkerFacts::ROW_BYTES;

/// The contract's status of a row.
pub(crate) const fn status(status: WorkerStatus) -> proto::WorkerStatus {
    match status {
        WorkerStatus::Pending => proto::WorkerStatus::Pending,
        WorkerStatus::Ready => proto::WorkerStatus::Ready,
        WorkerStatus::NotReady => proto::WorkerStatus::NotReady,
        WorkerStatus::Failed => proto::WorkerStatus::Failed,
        WorkerStatus::Draining => proto::WorkerStatus::Draining,
    }
}

/// The health bits a status sets, as a candidate row reads them: only a
/// ready worker is healthy, and a draining one says so.
pub(crate) const fn health_bits(status: WorkerStatus) -> u32 {
    match status {
        WorkerStatus::Ready => proto::HealthFlag::Healthy as u32,
        WorkerStatus::Draining => proto::HealthFlag::Draining as u32,
        WorkerStatus::Pending | WorkerStatus::NotReady | WorkerStatus::Failed => 0,
    }
}

const fn role(role: Role) -> proto::Role {
    match role {
        Role::Single => proto::Role::Single,
        Role::Prefill => proto::Role::Prefill,
        Role::Decode => proto::Role::Decode,
        Role::Encode => proto::Role::Encode,
    }
}

const fn health_policy(policy: HealthPolicy) -> proto::HealthPolicy {
    proto::HealthPolicy {
        failure_threshold: policy.failure_threshold,
        success_threshold: policy.success_threshold,
        check_interval_ms: policy.check_interval_ms,
        timeout_ms: policy.timeout_ms,
        disabled: !policy.probes,
    }
}

fn reachable(from: Reachability) -> proto::Reachability {
    use proto::reachability::Scope;
    proto::Reachability {
        scope: Some(match from {
            Reachability::Any => Scope::Any(true),
            // The store keeps the verified subject's fingerprint, never its
            // text.
            Reachability::Gateway(gateway) => Scope::GatewayId(format!("{:016x}", gateway.0)),
        }),
    }
}

/// The wire's row of a store row. Of the pairing descriptor it tells the
/// bootstrap port, the one field of it the store holds: the rest arrives
/// with the store's typed descriptor.
pub(crate) fn row(row: &WorkerRow) -> proto::WorkerRow {
    proto::WorkerRow {
        maker: maker(&row.maker),
        ..told(row.id, &row.facts, row.status, row.revision)
    }
}

/// The wire's maker of a row's: none for a control call's row, so a row
/// that crosses with no field reads as one.
fn maker(by: &Registrant) -> Option<proto::Maker> {
    match by {
        Registrant::Control => None,
        Registrant::Provider(found) => Some(proto::Maker {
            provider: found.provider().to_owned(),
            key: found.key().to_owned(),
        }),
    }
}

/// The wire's row of a row's parts: its numbers ([`numbers`]) with its text
/// laid over them.
fn told(id: WorkerId, facts: &WorkerFacts, state: WorkerStatus, revision: u64) -> proto::WorkerRow {
    let mut row = proto::WorkerRow {
        registration_key: facts.key.clone(),
        url: facts.key.clone(),
        runtime: facts.engine_kind.clone(),
        models: facts.models.clone(),
        labels: facts
            .labels
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect(),
        zone: facts.zone.clone().unwrap_or_default(),
        reachable: Some(reachable(facts.reachable_from)),
        policy: facts.labels.get("policy").cloned().unwrap_or_default(),
        engine: facts.engine.clone(),
        encode: stated_offer(&facts.encode),
        ..numbers(id, facts, state, revision)
    };
    // A scrape reads the row's engine kind as its family.
    if let Some(signals) = row.signals.as_mut().filter(|signals| signals.scrape) {
        signals.scrape_family.clone_from(&facts.engine_kind);
    }
    row
}

/// The fields of the wire's row that are no text. Nothing is allocated to
/// tell them, so [`weight`] reads a row's numbers here with no row built.
fn numbers(
    id: WorkerId,
    facts: &WorkerFacts,
    state: WorkerStatus,
    revision: u64,
) -> proto::WorkerRow {
    proto::WorkerRow {
        id: id.0,
        incarnation: facts.incarnation,
        // The wire enum's numbers are the store's bits: one wire reads as
        // its own value, several as their sum.
        wire: i32::from(facts.wire_modes.bits()),
        role: role(facts.role) as i32,
        dp_size: u32::from(facts.dp_size),
        status: status(state) as i32,
        revision,
        health: health_bits(state),
        pairing: facts.bootstrap_port.map(|port| proto::PairingDescriptor {
            bootstrap_port: u32::from(port),
            ..Default::default()
        }),
        // An instant's age is told where the send's clock is read: the
        // snapshot sets it.
        admitted_age_ms: 0,
        overload: Some(proto::OverloadThresholds {
            queue_limit: facts.overload.waiting_requests,
            kv_usage_permille: facts.overload.kv_usage_permille.map(u32::from),
        }),
        signals: Some(proto::Signals {
            events: facts.signals.events,
            loads: facts.signals.loads,
            scrape: facts.signals.scrape_interval_ms > 0,
            scrape_interval_ms: facts.signals.scrape_interval_ms,
            scrape_family: String::new(),
        }),
        kv_block_size: facts.kv_block_size.unwrap_or(0),
        max_running: facts.max_running.unwrap_or(0),
        dp_rank: u32::from(facts.dp_rank),
        health_policy: Some(health_policy(facts.health)),
        demand_facts: stated_demand_facts(facts.demand_facts),
        vision: facts.vision,
        priority: facts.priority,
        drain_settle_secs: facts.drain_settle_secs,
        ..Default::default()
    }
}

/// The contract's block of what a worker states for a request's hard
/// demands. A worker that states none tells no block, so its row is the
/// row a build before the block told, byte for byte.
fn stated_demand_facts(facts: DemandFacts) -> Option<proto::DemandFacts> {
    // Every field by its name: one the store's type gains stops the build
    // here, so none stays behind unseen.
    let DemandFacts {
        runtime,
        references,
        host_pipeline,
        window,
        decoding,
    } = facts;
    (facts != DemandFacts::default()).then_some(proto::DemandFacts {
        runtime,
        references,
        host_pipeline,
        window,
        decoding: u32::from(decoding),
    })
}

/// The store's facts of the contract's block. The way to decide a
/// constraint is a byte in the store: a wider number is refused by `path`,
/// in a registration and in a told row alike.
fn demand_facts(block: &proto::DemandFacts, path: &'static str) -> Result<DemandFacts, Refused> {
    // Every field by its name: one the contract gains stops the build here,
    // so none passes unread.
    let proto::DemandFacts {
        runtime,
        references,
        host_pipeline,
        window,
        decoding,
    } = block;
    let decoding = u8::try_from(*decoding)
        .map_err(|_| refused(path, "a way to decide a constraint is at most 255"))?;
    Ok(DemandFacts {
        runtime: *runtime,
        references: *references,
        host_pipeline: *host_pipeline,
        window: *window,
        decoding,
    })
}

/// The contract's block of a worker's end of the encode handoff: its
/// numbers ([`offer_numbers`]) with its connector laid over them. An offer
/// that states nothing tells no block, so its row is the row a build before
/// the block told, byte for byte. An offer that states a role, a connector
/// or references tells its block, also where it names no form.
fn stated_offer(offer: &EncodeOffer) -> Option<proto::EncodeOffer> {
    (*offer != EncodeOffer::default()).then(|| proto::EncodeOffer {
        connector: offer.connector.clone().unwrap_or_default(),
        ..offer_numbers(offer)
    })
}

/// The fields of the block that are no text, in a block whose connector is
/// empty. Nothing is allocated to tell them, so [`weight`] measures an
/// offer's numbers here and copies no connector.
const fn offer_numbers(offer: &EncodeOffer) -> proto::EncodeOffer {
    // Every field by its name: one the store's type gains stops the build
    // here, so none stays behind unseen.
    let EncodeOffer {
        handoff,
        producer,
        consumer,
        connector: _,
        references,
    } = offer;
    let handoff = match handoff {
        EncodeHandoff::None => proto::EncodeHandoff::Unspecified,
        EncodeHandoff::Push => proto::EncodeHandoff::Push,
        EncodeHandoff::Rendezvous => proto::EncodeHandoff::Rendezvous,
    };
    proto::EncodeOffer {
        handoff: handoff as i32,
        producer: *producer,
        consumer: *consumer,
        connector: String::new(),
        references: *references,
    }
}

/// The store's offer of the contract's block, in a registration and in a
/// told row alike. A form this build does not know is refused by the path
/// `handoff`: no reader can hold it. The connector is read as told, with no
/// rule of its words: it is the store's text, compared whole and never
/// parsed, and its one bound is the row's. So a row a leader's store took
/// is a row its standby reads. An empty one states none.
fn offer(block: &proto::EncodeOffer, handoff: &'static str) -> Result<EncodeOffer, Refused> {
    // Every field by its name: one the contract gains stops the build here,
    // so none passes unread.
    let proto::EncodeOffer {
        handoff: form,
        producer,
        consumer,
        connector: name,
        references,
    } = block;
    let form = match proto::EncodeHandoff::try_from(*form) {
        Ok(proto::EncodeHandoff::Unspecified) => EncodeHandoff::None,
        Ok(proto::EncodeHandoff::Push) => EncodeHandoff::Push,
        Ok(proto::EncodeHandoff::Rendezvous) => EncodeHandoff::Rendezvous,
        Err(_) => {
            return Err(refused(
                handoff,
                "an encode handoff this build does not know",
            ))
        }
    };
    Ok(EncodeOffer {
        handoff: form,
        producer: *producer,
        consumer: *consumer,
        connector: (!name.is_empty()).then(|| name.clone()),
        references: *references,
    })
}

/// A new key's row.
pub(crate) fn registered(new: &WorkerRow) -> Event {
    Event::Registered(row(new))
}

/// A row whose facts changed in place.
pub(crate) fn relabelled(now: &WorkerRow) -> Event {
    Event::Relabelled(row(now))
}

/// A key's new incarnation beside the one it replaced; `state_carried`
/// where the old one's status came over at the write.
pub(crate) fn replaced(old: &WorkerRow, new: &WorkerRow, state_carried: bool) -> Event {
    Event::Replaced(proto::Replaced {
        old: Some(row(old)),
        new: Some(row(new)),
        state_carried,
    })
}

/// A removal, told from the row as it last stood.
pub(crate) fn removed(last: &WorkerRow) -> Event {
    Event::Removed(proto::Removed {
        worker_id: last.id.0,
        registration_key: last.facts.key.clone(),
        revision: last.revision,
        draining: last.status == WorkerStatus::Draining,
    })
}

/// A status change.
pub(crate) fn status_changed(id: WorkerId, from: WorkerStatus, to: WorkerStatus) -> Event {
    Event::Status(proto::StatusChanged {
        worker_id: id.0,
        old: status(from) as i32,
        new: status(to) as i32,
        health: health_bits(to),
    })
}

/// One entry of the answer of a registration that states several ranks: the
/// event a watcher is told of that row, without a place of its own. `None`
/// for a case no registration makes of a row.
pub(crate) fn rank_row(event: Event) -> Option<proto::RankRow> {
    use proto::rank_row::Made;
    let made = match event {
        Event::Registered(row) => Made::Registered(row),
        Event::Replaced(replaced) => Made::Replaced(replaced),
        Event::Removed(removed) => Made::Removed(removed),
        Event::Relabelled(row) => Made::Relabelled(row),
        Event::Snapshot(_)
        | Event::Status(_)
        | Event::Needs(_)
        | Event::SnapshotRequired(_)
        | Event::Ranks(_) => return None,
    };
    Some(proto::RankRow { made: Some(made) })
}

/// The event in its envelope: a position of the log, the membership epoch
/// and the term the position was written under.
pub(crate) fn event(
    event: Event,
    seq: u64,
    membership_epoch: u64,
    term: u64,
) -> proto::WorkerEvent {
    proto::WorkerEvent {
        seq,
        membership_epoch,
        event: Some(event),
        term,
    }
}

/// The bytes the wire's row of these facts takes, with the numbers the
/// controller gives a row (its id, incarnation, revision and status) and
/// the overload block a patch may state at their widest: a status change,
/// a later revision or a patched block adds no byte. Summed with no row
/// built and nothing allocated: the numbers as [`numbers`] tells them, then
/// each text as the wire frames it, under the contract's field numbers.
/// `labels` are the row's own, or the ones a patch would leave: nothing is
/// copied to weigh the row it would make.
///
/// This sum and [`told`] tell one contract twice, and the pin over the
/// cases' full row holds them together: a text the row gains later is
/// stated in that row first.
fn weight(facts: &WorkerFacts, labels: &BTreeMap<String, String>) -> usize {
    use prost::encoding::{btree_map, encoded_len_varint, key_len, string};
    // A length's own bytes, and a text under its tag: nothing when empty.
    let length = |bytes: usize| encoded_len_varint(u64::try_from(bytes).unwrap_or(u64::MAX));
    let text = |tag: u32, text: &String| {
        if text.is_empty() {
            0
        } else {
            string::encoded_len(tag, text)
        }
    };
    let mut row = numbers(WorkerId(u64::MAX), facts, WorkerStatus::Ready, u64::MAX);
    row.incarnation = u64::MAX;
    row.overload = Some(proto::OverloadThresholds {
        queue_limit: Some(u32::MAX),
        kv_usage_permille: Some(1_000),
    });
    // A scrape's family stands in the signals block: its bytes, and one
    // more where they take the block's length past a byte.
    let signals = row.signals.as_ref().map_or(0, prost::Message::encoded_len);
    let family = if facts.signals.scrape_interval_ms > 0 {
        text(5, &facts.engine_kind)
    } else {
        0
    };
    // From any gateway is a flag in its block, from one its sixteen digits.
    let reachable = match facts.reachable_from {
        Reachability::Any => 4,
        Reachability::Gateway(_) => 20,
    };
    // An encode offer's block: its numbers, its connector as a text under
    // them, and the block's own key and length. An offer that states
    // nothing is no block.
    let offer = if facts.encode == EncodeOffer::default() {
        0
    } else {
        let numbers = offer_numbers(&facts.encode).encoded_len();
        let connector = facts.encode.connector.as_ref();
        let block = numbers + connector.map_or(0, |connector| text(4, connector));
        key_len(30) + length(block) + block
    };
    row.encoded_len()
        + (family + length(signals + family) - length(signals))
        + reachable
        + offer
        + text(2, &facts.key)
        + text(4, &facts.key)
        + text(7, &facts.engine_kind)
        + text(24, &facts.engine)
        + facts.zone.as_ref().map_or(0, |zone| text(11, zone))
        + labels.get("policy").map_or(0, |policy| text(20, policy))
        + string::encoded_len_repeated(8, &facts.models)
        + btree_map::encoded_len(string::encoded_len, string::encoded_len, 10, labels)
}

/// Refuses, by `path`, facts whose row under `labels` passes [`ROW_BYTES`].
fn fits(
    facts: &WorkerFacts,
    labels: &BTreeMap<String, String>,
    path: &'static str,
) -> Result<(), Refused> {
    if weight(facts, labels) > ROW_BYTES {
        return Err(refused(path, "a row is at most 16384 bytes on the wire"));
    }
    Ok(())
}

/// The bytes a row's maker takes on the wire, under the contract's field
/// numbers: none for a control call's row.
fn maker_weight(maker: &Registrant) -> usize {
    use prost::encoding::{encoded_len_varint, key_len};
    let Registrant::Provider(found) = maker else {
        return 0;
    };
    // A length's own bytes, and a text under its tag: neither of a maker's
    // two is empty.
    let length = |bytes: usize| encoded_len_varint(u64::try_from(bytes).unwrap_or(u64::MAX));
    let text = |tag: u32, text: &str| key_len(tag) + length(text.len()) + text.len();
    let inner = text(1, found.provider()) + text(2, found.key());
    key_len(27) + length(inner) + inner
}

/// Refuses, by its fact's path, a row the log would not give back whole. A
/// standby builds its rows from the log, so it would hold the row without
/// the fact, and a promoted one would decide by other facts than its leader
/// did. The row on the wire carries a pairing's port alone. It reads an
/// empty zone, an empty connector of an encode offer and a zero admission
/// window, block size or port as none. The facts are written as the log
/// writes them and read back as a standby reads them, so a fact the wire
/// gains is taken here with no second list. A text the log tells comes back
/// as it was told, so the door holds an encode offer's connector to no rule
/// of its words.
#[cfg_attr(
    not(any(test, feature = "discovery")),
    expect(dead_code, reason = "the host's door asks it of each registration")
)]
pub(crate) fn carried(facts: &WorkerFacts) -> Result<(), Refused> {
    const WHY: &str = "the log's row does not carry this fact: not served by this build";
    let sent = told(WorkerId(0), facts, WorkerStatus::Ready, 1);
    let back = told_row(&sent, Instant::now())?.facts;
    let lost = [
        ("row.pairing", back.pairing != facts.pairing),
        ("row.encode", back.encode != facts.encode),
        ("row.zone", back.zone != facts.zone),
        ("row.max_running", back.max_running != facts.max_running),
        (
            "row.kv_block_size",
            back.kv_block_size != facts.kv_block_size,
        ),
        (
            "row.bootstrap_port",
            back.bootstrap_port != facts.bootstrap_port,
        ),
        ("row", back != *facts),
    ];
    match lost.iter().find(|(_, lost)| *lost) {
        Some((path, _)) => Err(refused(path, WHY)),
        None => Ok(()),
    }
}

/// The controller's rule of a row's wire and its reach, refused by `path`.
/// A host-local wire is reachable from the one gateway that holds its
/// socket, so a row of that wire names its gateway: the store reads a row as
/// host-local by its reach alone, and a row every gateway reaches stands in
/// every gateway's pool. A control call's registration and a host's found
/// worker are held to it alike.
pub(crate) fn wire_reach(
    wire: WireModes,
    from: Reachability,
    path: &'static str,
) -> Result<(), Refused> {
    if wire.contains(WireModes::ZMQ) && from == Reachability::Any {
        return Err(refused(
            path,
            "a host-local wire is one gateway's and a registration names none: not served by this build",
        ));
    }
    Ok(())
}

/// Refuses, by `path`, a row of these facts under `labels` that passes
/// [`ROW_BYTES`] with its maker: the facts' own bound ([`fits`]) is told
/// where they are read, and a provider's row holds its maker inside the
/// same bound. A control call's row has no maker on the wire: the facts'
/// bound spoke for it where they were read, and it is not weighed again.
pub(crate) fn room(
    facts: &WorkerFacts,
    labels: &BTreeMap<String, String>,
    maker: &Registrant,
    path: &'static str,
) -> Result<(), Refused> {
    let bytes = maker_weight(maker);
    if bytes > 0 && weight(facts, labels) + bytes > ROW_BYTES {
        return Err(refused(path, "a row is at most 16384 bytes on the wire"));
    }
    Ok(())
}

/// The bounds of one label wherever it enters a row: a registration's
/// (`spec.labels`) and a patch's (`labels`).
fn label(name: &str, value: &str, path: &'static str) -> Result<(), Refused> {
    if name.len() > KEY_BYTES {
        return Err(refused(path, "a label's key is at most 128 bytes"));
    }
    if value.len() > VALUE_BYTES {
        return Err(refused(path, "a label's value is at most 4096 bytes"));
    }
    Ok(())
}

/// One page of a snapshot: `workers` as the write at `at` left them, and
/// `last` on the page that ends it. Every page of one snapshot carries that
/// position.
pub(crate) fn page(
    workers: Vec<proto::WorkerRow>,
    last: bool,
    at: Position,
    membership_epoch: u64,
) -> proto::WorkerEvent {
    let snapshot = proto::WorkerSnapshot { workers, last };
    event(Event::Snapshot(snapshot), at.seq, membership_epoch, at.term)
}

/// The word to a watcher whose place the log does not hold, under no
/// position of its own: `floor` is the place before the oldest event the
/// log retains.
pub(crate) fn start_again(floor: Position) -> proto::WorkerEvent {
    let word = proto::SnapshotRequired {
        oldest_seq: floor.seq + 1,
    };
    event(Event::SnapshotRequired(word), 0, 0, 0)
}

/// A log entry of a membership or status write in its envelope, at the
/// entry's position and term. `None` for any other entry, and for a shape
/// the store never tells. An entry's kind says which it is: a status
/// entry holds no row, and its event tells none.
pub(crate) fn member(entry: &Entry) -> Option<proto::WorkerEvent> {
    let (epoch, told) = match &entry.event {
        Written::Member {
            epoch,
            change,
            row,
            left,
        } => {
            let told = match (*change, row.as_deref(), left.as_deref()) {
                (MembershipChange::Registered { .. }, Some(new), _) => registered(new),
                (
                    MembershipChange::Relabelled { .. } | MembershipChange::Resized { .. },
                    Some(now),
                    _,
                ) => relabelled(now),
                // A fresh row is pending: any other status the write left
                // came over from the old incarnation.
                (MembershipChange::Replaced { .. }, Some(new), Some(old)) => {
                    replaced(old, new, new.status != WorkerStatus::Pending)
                }
                (MembershipChange::Removed { .. }, _, Some(last)) => removed(last),
                _ => return None,
            };
            (*epoch, told)
        }
        Written::Status {
            epoch,
            id,
            from,
            to,
            ..
        } => (*epoch, status_changed(*id, *from, *to)),
        Written::Term | Written::Generation { .. } => return None,
    };
    Some(event(told, entry.at.seq, epoch, entry.at.term))
}

/// A log entry as a standby is sent it. An entry this build
/// cannot tell is sent with no case set: the standby reads it as one it
/// does not know, never as nothing.
pub(crate) fn replication(entry: &Entry) -> proto::ReplicationEvent {
    use proto::replication_event::Event as Sent;
    let event = match &entry.event {
        Written::Member { .. } | Written::Status { .. } => member(entry).map(Sent::Membership),
        Written::Term => Some(Sent::TermStart(proto::TermStart {})),
        Written::Generation { generation } => Some(Sent::Generation(generation_told(*generation))),
    };
    proto::ReplicationEvent { event }
}

/// A generation's event as a log entry says it: the number alone. The
/// stream that sends it joins the document where its door still holds that
/// generation; sent as it is here, it tells a standby that the generation
/// stood no longer, and the standby journals the place alone.
pub(crate) fn generation_told(generation: u64) -> proto::GenerationEvent {
    proto::GenerationEvent {
        status: Some(proto::SpecStatus {
            generation,
            ..proto::SpecStatus::default()
        }),
        document: None,
    }
}

const fn unknown(path: &'static str) -> Refused {
    refused(path, "a value this build does not know")
}

/// A count the wire carries wide and the store holds narrow.
fn narrow(value: u32, path: &'static str) -> Result<u16, Refused> {
    u16::try_from(value).map_err(|_| refused(path, "past the store's width"))
}

/// The store's status of the wire's; `None` for a word this build does not
/// know.
fn told_status(status: i32) -> Option<WorkerStatus> {
    Some(match proto::WorkerStatus::try_from(status).ok()? {
        proto::WorkerStatus::Pending => WorkerStatus::Pending,
        proto::WorkerStatus::Ready => WorkerStatus::Ready,
        proto::WorkerStatus::NotReady => WorkerStatus::NotReady,
        proto::WorkerStatus::Failed => WorkerStatus::Failed,
        proto::WorkerStatus::Draining => WorkerStatus::Draining,
        proto::WorkerStatus::Unspecified => return None,
    })
}

/// The store's row of the wire's, the standby's half of [`row`]: every fact
/// the wire's row tells, as told, the admission instant `now` less the age
/// the row crossed with. Of a pairing descriptor the wire's row tells the
/// bootstrap port alone. The one fact it does not tell, the store's typed
/// descriptor, stands at the store's default: no call states it in this
/// build.
///
/// A value this build cannot hold (a status, a role, a scope or an encode
/// handoff it does not know, a count past the store's width) is refused by
/// its path: a standby that guessed would hold another row than its
/// leader's. A fact's text is read as told, an encode offer's connector as
/// a URL, a model id, a zone and a label are: the leader's store took it,
/// and the reader asks no second rule of it.
pub(crate) fn told_row(told: &proto::WorkerRow, now: Instant) -> Result<WorkerRow, Refused> {
    use proto::reachability::Scope;
    let status = told_status(told.status).ok_or_else(|| unknown("row.status"))?;
    let role = match proto::Role::try_from(told.role) {
        Ok(proto::Role::Single) => Role::Single,
        Ok(proto::Role::Prefill) => Role::Prefill,
        Ok(proto::Role::Decode) => Role::Decode,
        Ok(proto::Role::Encode) => Role::Encode,
        Ok(proto::Role::Unspecified) | Err(_) => return Err(unknown("row.role")),
    };
    // A wire this build does not know is a bit `from_bits` drops.
    let wire_modes = u8::try_from(told.wire)
        .map(WireModes::from_bits)
        .ok()
        .filter(|modes| i32::from(modes.bits()) == told.wire)
        .ok_or_else(|| unknown("row.wire"))?;
    let reachable_from = match told.reachable.as_ref().and_then(|from| from.scope.as_ref()) {
        Some(Scope::Any(true)) => Reachability::Any,
        Some(Scope::GatewayId(gateway)) => u64::from_str_radix(gateway, 16)
            .map(|gateway| Reachability::Gateway(GatewayId(gateway)))
            .map_err(|_| unknown("row.reachable"))?,
        Some(Scope::Any(false)) | None => return Err(unknown("row.reachable")),
    };
    let overload = told
        .overload
        .as_ref()
        .ok_or_else(|| unknown("row.overload"))?;
    let signals = told
        .signals
        .as_ref()
        .ok_or_else(|| unknown("row.signals"))?;
    let policy = told
        .health_policy
        .as_ref()
        .ok_or_else(|| unknown("row.health_policy"))?;
    // Of a descriptor the leader tells the bootstrap port alone ([`row`]):
    // one that tells more is of a build this one does not follow. A number
    // the contract reserves is no field: the generated reader passes over
    // it, so a row that states one reads as the row of its port alone.
    let port = told
        .pairing
        .as_ref()
        .map_or(0, |pairing| pairing.bootstrap_port);
    let port_alone = proto::PairingDescriptor {
        bootstrap_port: port,
        ..Default::default()
    };
    if told
        .pairing
        .as_ref()
        .is_some_and(|told| *told != port_alone)
    {
        return Err(unknown("row.pairing"));
    }
    let bootstrap_port = narrow(port, "row.pairing.bootstrap_port")?;
    let facts = WorkerFacts {
        key: told.registration_key.clone(),
        engine: told.engine.clone(),
        incarnation: told.incarnation,
        role,
        wire_modes,
        engine_kind: told.runtime.clone(),
        models: told.models.clone(),
        dp_rank: narrow(told.dp_rank, "row.dp_rank")?,
        dp_size: narrow(told.dp_size, "row.dp_size")?,
        labels: told
            .labels
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect(),
        zone: (!told.zone.is_empty()).then(|| told.zone.clone()),
        max_running: (told.max_running != 0).then_some(told.max_running),
        kv_block_size: (told.kv_block_size != 0).then_some(told.kv_block_size),
        bootstrap_port: (bootstrap_port != 0).then_some(bootstrap_port),
        reachable_from,
        signals: Signals {
            events: signals.events,
            loads: signals.loads,
            scrape_interval_ms: signals.scrape_interval_ms,
        },
        health: HealthPolicy {
            failure_threshold: policy.failure_threshold,
            success_threshold: policy.success_threshold,
            check_interval_ms: policy.check_interval_ms,
            timeout_ms: policy.timeout_ms,
            probes: !policy.disabled,
        },
        overload: Overload {
            kv_usage_permille: overload
                .kv_usage_permille
                .map(|permille| narrow(permille, "row.overload.kv_usage_permille"))
                .transpose()?,
            waiting_requests: overload.queue_limit,
        },
        demand_facts: told
            .demand_facts
            .as_ref()
            .map(|block| demand_facts(block, "row.demand_facts.decoding"))
            .transpose()?
            .unwrap_or_default(),
        vision: told.vision,
        encode: told
            .encode
            .as_ref()
            .map(|block| offer(block, "row.encode.handoff"))
            .transpose()?
            .unwrap_or_default(),
        priority: told.priority,
        drain_settle_secs: told.drain_settle_secs,
        ..WorkerFacts::default()
    };
    let maker = match &told.maker {
        None => Registrant::Control,
        Some(by) => match Registrant::provider(by.provider.clone(), by.key.clone()) {
            Ok(maker) => maker,
            Err(ProviderKeyError::Provider(why)) => return Err(refused("row.maker.provider", why)),
            Err(ProviderKeyError::Key(why)) => return Err(refused("row.maker.key", why)),
        },
    };
    let age = Duration::from_millis(told.admitted_age_ms);
    Ok(WorkerRow {
        id: WorkerId(told.id),
        facts,
        status,
        revision: told.revision,
        admitted_at: now.checked_sub(age).unwrap_or(now),
        maker,
    })
}

/// A membership event of the leader's stream as a standby's store takes it
/// ([`crate::store::MemoryStore::follow`]): the epoch, the change, and the
/// row the write left where the event tells one. A resize crosses as a
/// relabel; the store reads which it was from the widths.
pub(crate) fn told_change(
    told: &proto::WorkerEvent,
    now: Instant,
) -> Result<(u64, MembershipChange, Option<WorkerRow>), Refused> {
    let (change, row) = match told.event.as_ref() {
        Some(Event::Registered(row)) => {
            let row = told_row(row, now)?;
            (MembershipChange::Registered { id: row.id }, Some(row))
        }
        Some(Event::Relabelled(row)) => {
            let row = told_row(row, now)?;
            (MembershipChange::Relabelled { id: row.id }, Some(row))
        }
        Some(Event::Replaced(replaced)) => {
            let (Some(old), Some(new)) = (&replaced.old, &replaced.new) else {
                return Err(unknown("membership.replaced"));
            };
            let row = told_row(new, now)?;
            let change = MembershipChange::Replaced {
                old: WorkerId(old.id),
                new: row.id,
            };
            (change, Some(row))
        }
        Some(Event::Removed(removed)) => {
            let id = WorkerId(removed.worker_id);
            (MembershipChange::Removed { id }, None)
        }
        Some(Event::Status(status)) => {
            let id = WorkerId(status.worker_id);
            let from = told_status(status.old).ok_or_else(|| unknown("membership.status.old"))?;
            let to = told_status(status.new).ok_or_else(|| unknown("membership.status.new"))?;
            (MembershipChange::StatusChanged { id, from, to }, None)
        }
        // The answer of a registration that states several ranks: no watch
        // tells it, and a reader lets the stream go by its name.
        Some(Event::Ranks(_)) => {
            return Err(refused(
                "membership.ranks",
                "a registration's answer rides no watch",
            ));
        }
        // A watch's own cases, and one this build does not know.
        Some(Event::Snapshot(_) | Event::Needs(_) | Event::SnapshotRequired(_)) | None => {
            return Err(unknown("membership.event"));
        }
    };
    Ok((told.membership_epoch, change, row))
}

/// One part of a leader's snapshot, read at `now`, into the rows a standby
/// stages for its restore
/// ([`crate::store::MemoryStore::restore_from`]). Every instant is `now`
/// less the age its row crossed with: the leader told each age at the
/// part's send, so a part is read at its arrival. A part's load rows are
/// read past: a leader of this build sends none, and a standby keeps none
/// of an older leader's.
pub(crate) fn told_part(
    told: &proto::ReplicationSnapshot,
    now: Instant,
    rows: &mut Vec<WorkerRow>,
) -> Result<(), Refused> {
    rows.reserve(told.workers.len());
    for row in &told.workers {
        rows.push(told_row(row, now)?);
    }
    Ok(())
}

/// `value`, or `default` where the wire's zero says unset.
const fn or_default(value: u32, default: u32) -> u32 {
    if value == 0 {
        default
    } else {
        value
    }
}

/// The store's thresholds of the contract's block: a field the registration
/// sets is its own, an unset one takes the store's default field.
fn overload(block: &proto::OverloadThresholds, path: &'static str) -> Result<Overload, Refused> {
    let default = Overload::default();
    let kv_usage_permille = match block.kv_usage_permille {
        None => default.kv_usage_permille,
        Some(permille) => Some(
            u16::try_from(permille)
                .ok()
                .filter(|permille| *permille <= 1_000)
                .ok_or_else(|| refused(path, "a permille is at most 1000"))?,
        ),
    };
    Ok(Overload {
        kv_usage_permille,
        waiting_requests: block.queue_limit.or(default.waiting_requests),
    })
}

/// The URL a registration states, under its own rules: stated, within its
/// bound, with no blank and no control character.
fn key(spec: &proto::WorkerSpec) -> Result<&str, Refused> {
    if spec.url.is_empty() {
        return Err(refused("spec.url", "a registration names its worker's URL"));
    }
    if spec.url.len() > URL_BYTES {
        return Err(refused("spec.url", "a URL is at most 2048 bytes"));
    }
    if blank(&spec.url) {
        return Err(refused(
            "spec.url",
            "a URL holds no blank and no control character",
        ));
    }
    Ok(&spec.url)
}

/// The facts of the row a registration states ([`over`]), with the row that
/// stands under its URL. `standing` reads that row and is called once the
/// URL holds its own rules ([`key`]): a URL past its bound is refused
/// before anything is looked up under it.
pub(crate) fn registration(
    spec: &proto::WorkerSpec,
    standing: impl FnOnce(&str) -> Option<WorkerRow>,
) -> Result<(WorkerFacts, Option<WorkerRow>), Refused> {
    let standing = standing(key(spec)?);
    let facts = over(spec, standing.as_ref().map(|row| &row.facts), None)?;
    Ok((facts, standing))
}

/// The most ranks one registration states: a replacement answers two rows a
/// rank, and the answer stays inside half of a 4 MiB frame.
pub(crate) const RANKS: u32 = 64;

/// The ranks a registration states: one where it states none, and at most
/// [`RANKS`].
pub(crate) fn width(spec: &proto::WorkerSpec) -> Result<u16, Refused> {
    if spec.dp_size > RANKS {
        return Err(refused(
            "spec.dp_size",
            "a registration states at most 64 ranks",
        ));
    }
    narrow(spec.dp_size.max(1), "spec.dp_size")
}

/// The facts of one rank of a registration that states `width` ranks, with
/// the row that stands under the rank's URL. The URL's own rules hold for
/// the engine's address ([`key`]) and for the rank's.
pub(crate) fn rank_registration(
    spec: &proto::WorkerSpec,
    (url, rank, width): (&str, u16, u16),
    standing: impl FnOnce(&str) -> Option<WorkerRow>,
) -> Result<(WorkerFacts, Option<WorkerRow>), Refused> {
    key(spec)?;
    if url.len() > URL_BYTES {
        return Err(refused("spec.url", "a rank's URL is at most 2048 bytes"));
    }
    let standing = standing(url);
    let stated = standing.as_ref().map(|row| &row.facts);
    let facts = over(spec, stated, Some((url, rank, width)))?;
    Ok((facts, standing))
}

/// The facts of the row a registration states, laid over `standing` (the
/// facts of the key's standing row, which keep what the registration leaves
/// unsaid) or over the store's defaults, where a new row names its wire and
/// its role itself. Two facts are the registration's,
/// whole, whatever stood: the labels, and the health policy, so nobody's
/// "do not check" outlives the registration that said it. The incarnation
/// is the caller's to number.
///
/// A name is taken as written, so one that holds a blank or a control
/// character is refused, never mended: the URL, the runtime and each model
/// id key the store's rows, its pools and its classes.
///
/// A registration of one rank makes the row under its URL. One that states
/// more is laid a rank at a time ([`rank_registration`]): `rank` is that
/// rank's URL, its place and the width. A width above one with no rank
/// named is refused: the build lays no ranks.
///
/// The URL's own rules are [`key`]'s, held before this is called.
fn over(
    spec: &proto::WorkerSpec,
    standing: Option<&WorkerFacts>,
    rank: Option<(&str, u16, u16)>,
) -> Result<WorkerFacts, Refused> {
    if spec.runtime.len() > NAME_BYTES {
        return Err(refused("spec.runtime", "a runtime is at most 256 bytes"));
    }
    if blank(&spec.runtime) {
        return Err(refused(
            "spec.runtime",
            "a runtime holds no blank and no control character",
        ));
    }
    if spec.models.iter().any(|model| model.len() > NAME_BYTES) {
        return Err(refused("spec.models", "a model id is at most 256 bytes"));
    }
    if spec
        .models
        .iter()
        .any(|model| model.is_empty() || blank(model))
    {
        return Err(refused(
            "spec.models",
            "a model id is not empty and holds no blank and no control character",
        ));
    }
    // A wire or a role left unsaid is the standing row's. A new row names
    // both, and is told so after every field's own rule: a field past its
    // bound is refused as itself, whether a row stands or not.
    let wire_modes = match proto::WireMode::try_from(spec.wire) {
        Ok(proto::WireMode::Http) => Some(WireModes::HTTP),
        Ok(proto::WireMode::Grpc) => Some(WireModes::GRPC),
        Ok(proto::WireMode::Zmq) => Some(WireModes::ZMQ),
        Ok(proto::WireMode::Unspecified) => standing.map(|row| row.wire_modes),
        Err(_) => return Err(refused("spec.wire", "a wire this build does not know")),
    };
    // No field of a registration names a gateway: the row's reach is the
    // standing row's, and a new row's is every gateway's.
    let from = standing.map(|row| row.reachable_from).unwrap_or_default();
    if let Some(wire) = wire_modes {
        wire_reach(wire, from, "spec.wire")?;
    }
    let role = match proto::Role::try_from(spec.role) {
        Ok(proto::Role::Single) => Some(Role::Single),
        Ok(proto::Role::Prefill) => Some(Role::Prefill),
        Ok(proto::Role::Decode) => Some(Role::Decode),
        Ok(proto::Role::Encode) => Some(Role::Encode),
        Ok(proto::Role::Unspecified) => standing.map(|row| row.role),
        Err(_) => return Err(refused("spec.role", "a role this build does not know")),
    };
    if spec.dp_size > 1 && rank.is_none() {
        return Err(refused(
            "spec.dp_size",
            "a registration makes one rank; a width above one is not served by this build",
        ));
    }
    if !spec.topology.is_empty() {
        return Err(refused(
            "spec.topology",
            "no row holds a topology in this build",
        ));
    }
    if spec.zone.len() > NAME_BYTES {
        return Err(refused("spec.zone", "a zone is at most 256 bytes"));
    }
    if spec.labels.len() > LABELS {
        return Err(refused("spec.labels", "a row holds at most 128 labels"));
    }
    for (name, value) in &spec.labels {
        label(name, value, "spec.labels")?;
    }
    let bootstrap = match &spec.pairing {
        Some(pairing) => bootstrap_port(pairing)?,
        None => None,
    };
    let mut facts = standing.cloned().unwrap_or_default();
    match rank {
        None => {
            facts.key.clone_from(&spec.url);
            if standing.is_none() {
                facts.engine.clone_from(&spec.url);
                facts.dp_size = 1;
            }
        }
        // A rank states its engine, its place and the width at every
        // registration: a worker that comes back wider or narrower says so.
        Some((url, rank, width)) => {
            url.clone_into(&mut facts.key);
            facts.engine.clone_from(&spec.url);
            facts.dp_rank = rank;
            facts.dp_size = width;
        }
    }
    if !spec.runtime.is_empty() {
        facts.engine_kind.clone_from(&spec.runtime);
    }
    if !spec.models.is_empty() {
        facts.models.clone_from(&spec.models);
    }
    // The labels of record are the registration's, whole.
    facts.labels = spec
        .labels
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    if !spec.zone.is_empty() {
        facts.zone = Some(spec.zone.clone());
    }
    if spec.kv_block_size != 0 {
        facts.kv_block_size = Some(spec.kv_block_size);
    }
    if bootstrap.is_some() {
        facts.bootstrap_port = bootstrap;
    }
    if let Some(block) = &spec.overload {
        facts.overload = overload(block, "spec.overload.kv_usage_permille")?;
    }
    // What the worker states for a request's hard demands is the block's,
    // whole: an empty one states none, and one left out keeps the row's.
    if let Some(block) = &spec.demand_facts {
        facts.demand_facts = demand_facts(block, "spec.demand_facts.decoding")?;
    }
    // Whether the worker takes images is the registration's where it states
    // it: one that states none keeps the row's.
    if let Some(vision) = spec.vision {
        facts.vision = vision;
    }
    // The worker's end of the encode handoff is the block's, whole: an
    // empty one states nothing, and one left out keeps the row's. Its
    // connector is read as the store holds it, past the blanks around it,
    // so a registration that states the row's offer again finds it unmoved.
    if let Some(block) = &spec.encode {
        facts.encode = offer(block, "spec.encode.handoff")?.normalized();
    }
    // A priority and a drain window are the registration's where it states
    // them: one that states none keeps the row's.
    if let Some(priority) = spec.priority {
        facts.priority = priority;
    }
    if let Some(window) = spec.drain_settle_secs {
        facts.drain_settle_secs = Some(window);
    }
    // The health policy of record is the registration's, whole: one that
    // states none is checked under the defaults.
    let default = HealthPolicy::default();
    facts.health = spec
        .health_policy
        .as_ref()
        .map_or(default, |policy| HealthPolicy {
            failure_threshold: or_default(policy.failure_threshold, default.failure_threshold),
            success_threshold: or_default(policy.success_threshold, default.success_threshold),
            check_interval_ms: or_default(policy.check_interval_ms, default.check_interval_ms),
            timeout_ms: or_default(policy.timeout_ms, default.timeout_ms),
            probes: !policy.disabled,
        });
    if let Some(signals) = &spec.signals {
        if signals.scrape && signals.scrape_interval_ms == 0 {
            return Err(refused(
                "spec.signals.scrape_interval_ms",
                "a scrape names its interval",
            ));
        }
        if !signals.scrape && signals.scrape_interval_ms != 0 {
            return Err(refused(
                "spec.signals.scrape",
                "an interval comes with the scrape flag",
            ));
        }
        if !signals.scrape_family.is_empty() && signals.scrape_family != facts.engine_kind {
            return Err(refused(
                "spec.signals.scrape_family",
                "a scrape reads the registration's runtime as its family",
            ));
        }
        facts.signals = Signals {
            events: signals.events,
            loads: signals.loads,
            scrape_interval_ms: signals.scrape_interval_ms,
        };
    }
    // Each field is within its own bound here. What a new row leaves
    // unnamed is told next, and the row whole last: only a row that can be
    // made has a size.
    facts.wire_modes = wire_modes.ok_or(NEW_WIRE)?;
    facts.role = role.ok_or(NEW_ROLE)?;
    fits(&facts, &facts.labels, "spec")?;
    Ok(facts)
}

/// The bootstrap port a registration's pairing descriptor states; `None`
/// where it states none. The store holds no other field of a descriptor
/// yet: one a caller states is refused by its path, never dropped. A number
/// the contract reserves is no field: the generated reader passes over it,
/// so a peer that states one is read as one that states none.
fn bootstrap_port(pairing: &proto::PairingDescriptor) -> Result<Option<u16>, Refused> {
    const NOT_HELD: &str = "the store holds a descriptor's bootstrap port alone in this build";
    // Every field by its name: one the contract gains stops the build
    // here, so none passes unread.
    let proto::PairingDescriptor {
        protocol,
        connector,
        transport,
        version,
        bootstrap_port,
        bootstrap_host,
        kv_layout,
    } = pairing;
    let others = [
        (!protocol.is_empty(), "spec.pairing.protocol"),
        (!connector.is_empty(), "spec.pairing.connector"),
        (!transport.is_empty(), "spec.pairing.transport"),
        (!version.is_empty(), "spec.pairing.version"),
        (!kv_layout.is_empty(), "spec.pairing.kv_layout"),
    ];
    if let Some(&(_, path)) = others.iter().find(|(stated, _)| *stated) {
        return Err(refused(path, NOT_HELD));
    }
    if !bootstrap_host.is_empty() {
        return Err(refused(
            "spec.pairing.bootstrap_host",
            "the bootstrap host is the URL's: its reader derives it",
        ));
    }
    match u16::try_from(*bootstrap_port) {
        Ok(0) => Ok(None),
        Ok(port) => Ok(Some(port)),
        Err(_) => Err(refused(
            "spec.pairing.bootstrap_port",
            "a port is at most 65535",
        )),
    }
}

/// The store's patch of the contract's over the row it names: the labels
/// merged (set, then removed), the overload block replaced when present.
/// `None` when the patch changes nothing of `standing`. The labels it would
/// leave are held to a registration's bounds, the row whole last.
pub(crate) fn patch(
    request: &proto::PatchWorkerRequest,
    standing: &WorkerFacts,
) -> Result<Option<WorkerPatch>, Refused> {
    // The count first: a list past it is refused unread.
    if request.remove_labels.len() > LABELS {
        return Err(refused(
            "remove_labels",
            "a patch removes at most 128 labels",
        ));
    }
    if request
        .remove_labels
        .iter()
        .any(|name| request.labels.contains_key(name))
    {
        return Err(refused(
            "remove_labels",
            "a label is set or removed, not both",
        ));
    }
    if request
        .remove_labels
        .iter()
        .any(|name| name.len() > KEY_BYTES)
    {
        return Err(refused(
            "remove_labels",
            "a label's key is at most 128 bytes",
        ));
    }
    // A label is set or removed, not both: the labels a patch sets all stand
    // in the row it would make.
    if request.labels.len() > LABELS {
        return Err(refused("labels", "a row holds at most 128 labels"));
    }
    for (name, value) in &request.labels {
        label(name, value, "labels")?;
    }
    let mut labels: BTreeMap<String, String> = standing.labels.clone();
    for (name, value) in &request.labels {
        labels.insert(name.clone(), value.clone());
    }
    for name in &request.remove_labels {
        labels.remove(name);
    }
    let labels = if labels == standing.labels {
        None
    } else {
        // The bounds are of the row the patch would make.
        if labels.len() > LABELS {
            return Err(refused("labels", "a row holds at most 128 labels"));
        }
        fits(standing, &labels, "labels")?;
        Some(labels)
    };
    let overload = match &request.overload {
        Some(block) => {
            Some(overload(block, "overload.kv_usage_permille")?).filter(|o| *o != standing.overload)
        }
        None => None,
    };
    Ok(
        (labels.is_some() || overload.is_some()).then_some(WorkerPatch {
            labels,
            priority: None,
            overload,
        }),
    )
}

#[cfg(test)]
#[path = "../../../tests/unit/service/controller/worker_wire/tests/mod.rs"]
mod tests;
