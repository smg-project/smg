//! The one conversion between the store's worker rows and the contract's
//! (design 5.2): a registration call's answer, a watch's page and the
//! replication stream carry one row. This half takes the store's row to the
//! wire's, and a registration or a patch off the wire into the store's
//! words; the wire's row back into the store's is the standby's half.
//!
//! A request field the store cannot hold is refused by its path
//! ([`Refused`]), never dropped. So is a field past its bound, and last a
//! row past [`ROW_BYTES`] whole: an answer's two rows and a snapshot's page
//! then fit a frame.

use std::{collections::BTreeMap, time::Instant};

use prost::Message as _;

use crate::{
    proto::{self, worker_event::Event},
    replication::{Entry, Event as Written},
    store::{
        HealthPolicy, LoadSample, MembershipChange, Overload, Position, Reachability, Signals,
        WorkerFacts, WorkerPatch, WorkerRow, WorkerStatus,
    },
    view::{Role, SignalSource, WireModes, WorkerId},
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
    if models.iter().any(|model| model.len() > NAME_BYTES) {
        return Err(refused("models", "a model id is at most 256 bytes"));
    }
    if models.iter().any(|model| model.is_empty() || blank(model)) {
        return Err(refused(
            "models",
            "a model id is not empty and holds no blank and no control character",
        ));
    }
    Ok(())
}

/// Whether `refusal` is a rule of a new row alone, and so no rule of a call
/// that makes no row.
pub(crate) fn of_a_new_row(refusal: &Refused) -> bool {
    *refusal == NEW_WIRE || *refusal == NEW_ROLE
}

/// The bounds of what a row states (contract: `WorkerSpec`). A row tells
/// its URL three times, a replacement's answer carries two rows and a
/// snapshot pages at 128 rows: under these each fits a frame of 4 MiB.
///
/// A URL's bytes.
const URL_BYTES: usize = 2_048;
/// The bytes of a name: a runtime, a zone, a model id.
const NAME_BYTES: usize = 256;
/// A label's key, in bytes.
const KEY_BYTES: usize = 128;
/// A label's value, in bytes.
const VALUE_BYTES: usize = 4_096;
/// The labels of one row.
const LABELS: usize = 128;
/// The model ids of one row: a limit of this build and no number of the
/// contract, said again once the store's cost of a row's ids is measured.
const MODELS: usize = 128;
/// The row whole on the wire, as [`weight`] measures it.
const ROW_BYTES: usize = 16 * 1024;

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
    told(row.id, &row.facts, row.status, row.revision)
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
        ..Default::default()
    }
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
    use prost::encoding::{btree_map, encoded_len_varint, string};
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
    row.encoded_len()
        + (family + length(signals + family) - length(signals))
        + reachable
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

/// Whether a stated name holds a blank or a control character.
fn blank(name: &str) -> bool {
    name.chars().any(|c| c.is_whitespace() || c.is_control())
}

/// A log entry of a membership or status write in its envelope, at the
/// entry's position and term. `None` for any other entry, and for a shape
/// the store never tells.
pub(crate) fn member(entry: &Entry) -> Option<proto::WorkerEvent> {
    let Written::Member {
        epoch,
        change,
        row,
        left,
    } = &entry.event
    else {
        return None;
    };
    let told = match (*change, row.as_deref(), left.as_deref()) {
        (MembershipChange::Registered { .. }, Some(new), _) => registered(new),
        (MembershipChange::Relabelled { .. } | MembershipChange::Resized { .. }, Some(now), _) => {
            relabelled(now)
        }
        // A fresh row is pending: any other status the write left came
        // over from the old incarnation.
        (MembershipChange::Replaced { .. }, Some(new), Some(old)) => {
            replaced(old, new, new.status != WorkerStatus::Pending)
        }
        (MembershipChange::Removed { .. }, _, Some(last)) => removed(last),
        (MembershipChange::StatusChanged { id, from, to }, Some(_), _) => {
            status_changed(id, from, to)
        }
        _ => return None,
    };
    Some(event(told, entry.at.seq, *epoch, entry.at.term))
}

const fn signal_source(source: SignalSource) -> proto::SignalSource {
    match source {
        SignalSource::None => proto::SignalSource::None,
        SignalSource::Events => proto::SignalSource::Events,
        SignalSource::Scrape => proto::SignalSource::Scrape,
        SignalSource::Relay => proto::SignalSource::Relay,
        SignalSource::Poll => proto::SignalSource::Poll,
    }
}

/// One rank's load row as a standby is sent it at `now`. The age is the
/// sample's at the send: never under the age the sample arrived with.
/// Whether the report counts resident tokens crosses beside the row: the
/// row's zero does not say it.
pub(crate) fn load(
    id: WorkerId,
    rank: u16,
    sample: &LoadSample,
    now: Instant,
) -> proto::LoadSample {
    let row = &sample.row;
    let age = now.saturating_duration_since(sample.sampled_at).as_millis();
    proto::LoadSample {
        worker_id: id.0,
        dp_rank: u32::from(rank),
        load: Some(proto::LoadRow {
            running: row.running,
            queued: row.queued,
            waiting_uncached_tokens: row.waiting_uncached_tokens,
            kv_usage_permille: u32::from(row.kv_usage_permille),
            active_kv_permille: u32::from(row.active_kv_permille),
            live_rate: row.live_rate,
            resident_tokens: row.resident_tokens,
            report_age_ms: u32::try_from(age).unwrap_or(u32::MAX),
            source: signal_source(row.source) as i32,
            max_running: row.max_running,
        }),
        resident_counted: row.resident_counted,
    }
}

/// A log entry as a standby is sent it at `now`. An entry this build
/// cannot tell is sent with no case set: the standby reads it as one it
/// does not know, never as nothing.
pub(crate) fn replication(entry: &Entry, now: Instant) -> proto::ReplicationEvent {
    use proto::replication_event::Event as Sent;
    let event = match &entry.event {
        Written::Member { .. } => member(entry).map(Sent::Membership),
        Written::Load { id, rank, sample } => Some(Sent::Load(load(*id, *rank, sample, now))),
        Written::Term => Some(Sent::TermStart(proto::TermStart {})),
    };
    proto::ReplicationEvent { event }
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
    let facts = over(spec, standing.as_ref().map(|row| &row.facts))?;
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
/// One rank a registration: the contract's call answers one row, so a stated
/// width above one is refused until a call can answer a row a rank.
///
/// The URL's own rules are [`key`]'s, held before this is called.
fn over(spec: &proto::WorkerSpec, standing: Option<&WorkerFacts>) -> Result<WorkerFacts, Refused> {
    // The count first, before one id is read or copied: every membership
    // write walks the ids of every standing row.
    if spec.models.len() > MODELS {
        return Err(refused("spec.models", "a row names at most 128 model ids"));
    }
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
        // A host-local wire is reachable from the one gateway that holds its
        // socket, and no field of a registration names a gateway.
        Ok(proto::WireMode::Zmq) => {
            return Err(refused(
                "spec.wire",
                "a host-local wire is one gateway's and a registration names none: not served by this build",
            ));
        }
        Ok(proto::WireMode::Unspecified) => standing.map(|row| row.wire_modes),
        Err(_) => return Err(refused("spec.wire", "a wire this build does not know")),
    };
    let role = match proto::Role::try_from(spec.role) {
        Ok(proto::Role::Single) => Some(Role::Single),
        Ok(proto::Role::Prefill) => Some(Role::Prefill),
        Ok(proto::Role::Decode) => Some(Role::Decode),
        Ok(proto::Role::Encode) => Some(Role::Encode),
        Ok(proto::Role::Unspecified) => standing.map(|row| row.role),
        Err(_) => return Err(refused("spec.role", "a role this build does not know")),
    };
    if spec.dp_size > 1 {
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
    facts.key.clone_from(&spec.url);
    if standing.is_none() {
        facts.engine.clone_from(&spec.url);
        facts.dp_size = 1;
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
/// yet: one a caller states is refused by its path, never dropped.
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
        pair_class,
        ec_producer,
        ec_consumer,
        vision,
        encode_handoff,
        ec_connector,
        encode_references,
    } = pairing;
    let others = [
        (!protocol.is_empty(), "spec.pairing.protocol"),
        (!connector.is_empty(), "spec.pairing.connector"),
        (!transport.is_empty(), "spec.pairing.transport"),
        (!version.is_empty(), "spec.pairing.version"),
        (!kv_layout.is_empty(), "spec.pairing.kv_layout"),
        (*ec_producer, "spec.pairing.ec_producer"),
        (*ec_consumer, "spec.pairing.ec_consumer"),
        (*vision, "spec.pairing.vision"),
        (*encode_handoff != 0, "spec.pairing.encode_handoff"),
        (!ec_connector.is_empty(), "spec.pairing.ec_connector"),
        (*encode_references, "spec.pairing.encode_references"),
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
    if *pair_class != 0 {
        return Err(refused(
            "spec.pairing.pair_class",
            "the store derives a pairing class: a registration states none",
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
mod tests {
    use std::{
        collections::BTreeMap,
        sync::Arc,
        time::{Duration, Instant},
    };

    use prost::Message as _;

    use super::{
        event, load, member, page, patch, refused, registration, removed, replaced, replication,
        row, start_again, status, status_changed, Refused, KEY_BYTES, LABELS, MODELS, NAME_BYTES,
        ROW_BYTES, URL_BYTES, VALUE_BYTES,
    };
    use crate::{
        proto::{self, replication_event::Event as Sent, worker_event::Event},
        replication::{Entry, Event as Written},
        store::{
            identity, GatewayId, HealthPolicy, LoadSample, LoadSource, MembershipChange, Overload,
            Position, Reachability, Signals, WorkerFacts, WorkerRow, WorkerStatus,
        },
        view::{LoadRow, Role, SignalSource, WireModes, WorkerId},
    };

    const URL: &str = "grpc://10.0.0.5:8000";

    fn spec() -> proto::WorkerSpec {
        proto::WorkerSpec {
            url: URL.to_owned(),
            wire: proto::WireMode::Grpc as i32,
            role: proto::Role::Single as i32,
            ..Default::default()
        }
    }

    /// What the row of these facts weighs under its own labels.
    fn weight(facts: &WorkerFacts) -> usize {
        super::weight(facts, &facts.labels)
    }

    /// The facts of a registration over a row of `standing`, through the
    /// door's own entry.
    fn facts(
        spec: &proto::WorkerSpec,
        standing: Option<&WorkerFacts>,
    ) -> Result<WorkerFacts, Refused> {
        let row = standing.map(|facts| WorkerRow {
            facts: facts.clone(),
            ..full()
        });
        registration(spec, |_| row).map(|(facts, _)| facts)
    }

    /// A name of `bytes` bytes.
    fn text(bytes: usize) -> String {
        "a".repeat(bytes)
    }

    /// A registration whose row weighs `bytes`: three labels at a value's
    /// bound and a fourth that fills the rest.
    fn weighing(bytes: usize) -> proto::WorkerSpec {
        let mut stated = proto::WorkerSpec {
            labels: ["a", "b", "c"]
                .map(|key| (key.to_owned(), text(VALUE_BYTES)))
                .into(),
            ..spec()
        };
        stated.labels.insert("fill".to_owned(), text(200));
        let base = weight(&facts(&stated, None).expect("a row under every bound"));
        stated
            .labels
            .insert("fill".to_owned(), text(200 + bytes - base));
        stated
    }

    /// A store row with every fact the wire's row tells set to a value of
    /// its own.
    fn full() -> WorkerRow {
        let facts = WorkerFacts {
            key: URL.to_owned(),
            engine: "grpc://10.0.0.5".to_owned(),
            incarnation: 3,
            role: Role::Decode,
            wire_modes: WireModes::GRPC,
            engine_kind: "runtime-a".to_owned(),
            models: vec!["model-a".to_owned(), "model-b".to_owned()],
            dp_rank: 2,
            dp_size: 4,
            labels: BTreeMap::from([
                ("policy".to_owned(), "least_load".to_owned()),
                ("tier".to_owned(), "gold".to_owned()),
            ]),
            zone: Some("zone-1".to_owned()),
            max_running: Some(96),
            kv_block_size: Some(16),
            bootstrap_port: Some(8_998),
            reachable_from: Reachability::Gateway(GatewayId(0xabc)),
            signals: Signals {
                events: true,
                loads: true,
                scrape_interval_ms: 2_500,
            },
            health: HealthPolicy {
                failure_threshold: 5,
                success_threshold: 4,
                check_interval_ms: 7_000,
                timeout_ms: 900,
                probes: false,
            },
            overload: Overload {
                kv_usage_permille: Some(875),
                waiting_requests: None,
            },
            ..WorkerFacts::default()
        };
        WorkerRow {
            id: identity(URL, 3),
            facts,
            status: WorkerStatus::Draining,
            revision: 11,
            admitted_at: Instant::now(),
        }
    }

    #[test]
    fn the_wires_row_tells_every_fact_of_the_stores_row() {
        let stored = full();
        let told = row(&stored);
        let expected = proto::WorkerRow {
            id: stored.id.0,
            registration_key: URL.to_owned(),
            incarnation: 3,
            url: URL.to_owned(),
            wire: proto::WireMode::Grpc as i32,
            role: proto::Role::Decode as i32,
            runtime: "runtime-a".to_owned(),
            models: vec!["model-a".to_owned(), "model-b".to_owned()],
            dp_size: 4,
            labels: [("policy", "least_load"), ("tier", "gold")]
                .into_iter()
                .map(|(name, value)| (name.to_owned(), value.to_owned()))
                .collect(),
            zone: "zone-1".to_owned(),
            status: proto::WorkerStatus::Draining as i32,
            revision: 11,
            health: proto::HealthFlag::Draining as u32,
            reachable: Some(proto::Reachability {
                scope: Some(proto::reachability::Scope::GatewayId(
                    "0000000000000abc".to_owned(),
                )),
            }),
            pairing: Some(proto::PairingDescriptor {
                bootstrap_port: 8_998,
                ..Default::default()
            }),
            admitted_age_ms: 0,
            overload: Some(proto::OverloadThresholds {
                queue_limit: None,
                kv_usage_permille: Some(875),
            }),
            signals: Some(proto::Signals {
                events: true,
                loads: true,
                scrape: true,
                scrape_interval_ms: 2_500,
                scrape_family: "runtime-a".to_owned(),
            }),
            kv_block_size: 16,
            policy: "least_load".to_owned(),
            topology: Vec::new(),
            max_running: 96,
            dp_rank: 2,
            engine: "grpc://10.0.0.5".to_owned(),
            health_policy: Some(proto::HealthPolicy {
                failure_threshold: 5,
                success_threshold: 4,
                check_interval_ms: 7_000,
                timeout_ms: 900,
                disabled: true,
            }),
        };
        assert_eq!(told, expected);
    }

    #[test]
    fn a_row_of_defaults_reads_any_gateway_and_no_scrape() {
        let stored = WorkerRow {
            id: identity(URL, 1),
            facts: WorkerFacts {
                key: URL.to_owned(),
                wire_modes: WireModes::HTTP.union(WireModes::GRPC),
                ..WorkerFacts::default()
            },
            status: WorkerStatus::Ready,
            revision: 1,
            admitted_at: Instant::now(),
        };
        let told = row(&stored);
        // Two wires read as the sum of their numbers.
        assert_eq!(told.wire, 3);
        assert_eq!(told.health, proto::HealthFlag::Healthy as u32);
        assert_eq!(
            told.reachable,
            Some(proto::Reachability {
                scope: Some(proto::reachability::Scope::Any(true)),
            })
        );
        let signals = told.signals.expect("the row's signals");
        assert!(!signals.scrape && signals.scrape_family.is_empty());
        let default = Overload::default();
        assert_eq!(
            told.overload,
            Some(proto::OverloadThresholds {
                queue_limit: default.waiting_requests,
                kv_usage_permille: default.kv_usage_permille.map(u32::from),
            })
        );
        assert_eq!(
            told.health_policy.map(|policy| policy.disabled),
            Some(false)
        );
    }

    #[test]
    fn every_status_has_its_own_word_on_the_wire() {
        let told = [
            WorkerStatus::Pending,
            WorkerStatus::Ready,
            WorkerStatus::NotReady,
            WorkerStatus::Failed,
            WorkerStatus::Draining,
        ]
        .map(status);
        assert_eq!(
            told,
            [
                proto::WorkerStatus::Pending,
                proto::WorkerStatus::Ready,
                proto::WorkerStatus::NotReady,
                proto::WorkerStatus::Failed,
                proto::WorkerStatus::Draining,
            ]
        );
        let change = status_changed(identity(URL, 1), WorkerStatus::Pending, WorkerStatus::Ready);
        assert_eq!(
            event(change, 12, 9, 4),
            proto::WorkerEvent {
                seq: 12,
                membership_epoch: 9,
                event: Some(Event::Status(proto::StatusChanged {
                    worker_id: identity(URL, 1).0,
                    old: proto::WorkerStatus::Pending as i32,
                    new: proto::WorkerStatus::Ready as i32,
                    health: proto::HealthFlag::Healthy as u32,
                })),
                term: 4,
            }
        );
    }

    #[test]
    fn a_log_entry_of_each_membership_write_has_its_case_at_the_entrys_place() {
        let old = full();
        let new = WorkerRow {
            id: identity(URL, 4),
            status: WorkerStatus::Ready,
            revision: old.revision + 3,
            ..full()
        };
        let at = Position { seq: 9, term: 4 };
        let entry = |change, row: Option<&WorkerRow>, left: Option<&WorkerRow>| Entry {
            at,
            event: Written::Member {
                epoch: 6,
                change,
                row: row.cloned().map(Arc::new),
                left: left.cloned().map(Arc::new),
            },
        };
        let told = |entry: &Entry| {
            let sent = member(entry).expect("a membership event");
            assert_eq!((sent.seq, sent.membership_epoch, sent.term), (9, 6, 4));
            sent.event.expect("a case")
        };

        let registered = MembershipChange::Registered { id: new.id };
        assert_eq!(
            told(&entry(registered, Some(&new), None)),
            Event::Registered(row(&new))
        );
        for change in [
            MembershipChange::Relabelled { id: new.id },
            MembershipChange::Resized { id: new.id },
        ] {
            assert_eq!(
                told(&entry(change, Some(&new), None)),
                Event::Relabelled(row(&new))
            );
        }
        // The new incarnation's status came over from the old one's, or it
        // is a fresh row's.
        let replacement = MembershipChange::Replaced {
            old: old.id,
            new: new.id,
        };
        assert_eq!(
            told(&entry(replacement, Some(&new), Some(&old))),
            replaced(&old, &new, true)
        );
        let fresh = WorkerRow {
            status: WorkerStatus::Pending,
            ..new.clone()
        };
        assert_eq!(
            told(&entry(replacement, Some(&fresh), Some(&old))),
            replaced(&old, &fresh, false)
        );
        assert_eq!(
            told(&entry(
                MembershipChange::Removed { id: old.id },
                None,
                Some(&old)
            )),
            removed(&old)
        );
        // A status change is told with the revision its write left the row
        // at, the row's own.
        let (from, to) = (WorkerStatus::Pending, WorkerStatus::Ready);
        let moved = MembershipChange::StatusChanged {
            id: new.id,
            from,
            to,
        };
        assert_eq!(
            told(&entry(moved, Some(&new), None)),
            status_changed(new.id, from, to)
        );
        // A shape the store never tells, and an entry that is no worker's
        // event.
        assert_eq!(member(&entry(registered, None, None)), None);
        assert_eq!(member(&entry(moved, None, Some(&old))), None);
        assert_eq!(
            member(&Entry {
                at,
                event: Written::Term
            }),
            None
        );
    }

    #[test]
    fn a_page_carries_its_rows_at_the_cuts_place_and_the_word_to_start_again_carries_none() {
        let first = full();
        let second = WorkerRow {
            id: identity(URL, 2),
            ..full()
        };
        let at = Position { seq: 9, term: 4 };
        assert_eq!(
            page(vec![row(&first), row(&second)], false, at, 6),
            proto::WorkerEvent {
                seq: 9,
                membership_epoch: 6,
                event: Some(Event::Snapshot(proto::WorkerSnapshot {
                    workers: vec![row(&first), row(&second)],
                    last: false,
                })),
                term: 4,
            }
        );
        assert_eq!(
            page(Vec::new(), true, at, 6).event,
            Some(Event::Snapshot(proto::WorkerSnapshot {
                workers: Vec::new(),
                last: true,
            }))
        );
        // The oldest event the log retains is the one after its floor.
        assert_eq!(
            start_again(Position { seq: 40, term: 4 }),
            proto::WorkerEvent {
                seq: 0,
                membership_epoch: 0,
                event: Some(Event::SnapshotRequired(proto::SnapshotRequired {
                    oldest_seq: 41,
                })),
                term: 0,
            }
        );
    }

    #[test]
    fn a_bare_registration_is_one_checked_rank_under_the_stores_defaults() {
        let made = facts(&spec(), None).expect("a bare registration");
        assert_eq!(
            made,
            WorkerFacts {
                key: URL.to_owned(),
                engine: URL.to_owned(),
                role: Role::Single,
                wire_modes: WireModes::GRPC,
                dp_size: 1,
                ..WorkerFacts::default()
            }
        );
        assert!(made.health.probes);
        assert_eq!(made.reachable_from, Reachability::Any);
    }

    #[test]
    fn a_registration_states_its_row_field_by_field() {
        let stated = proto::WorkerSpec {
            runtime: "runtime-a".to_owned(),
            models: vec!["model-a".to_owned()],
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            zone: "zone-1".to_owned(),
            overload: Some(proto::OverloadThresholds {
                queue_limit: Some(40),
                kv_usage_permille: None,
            }),
            kv_block_size: 16,
            pairing: Some(proto::PairingDescriptor {
                bootstrap_port: 8_998,
                ..Default::default()
            }),
            dp_size: 1,
            signals: Some(proto::Signals {
                events: true,
                loads: false,
                scrape: true,
                scrape_interval_ms: 3_000,
                scrape_family: "runtime-a".to_owned(),
            }),
            health_policy: Some(proto::HealthPolicy {
                failure_threshold: 0,
                success_threshold: 6,
                check_interval_ms: 0,
                timeout_ms: 250,
                disabled: true,
            }),
            ..spec()
        };
        let made = facts(&stated, None).expect("a full registration");
        let default = HealthPolicy::default();
        assert_eq!(
            made,
            WorkerFacts {
                key: URL.to_owned(),
                engine: URL.to_owned(),
                role: Role::Single,
                wire_modes: WireModes::GRPC,
                engine_kind: "runtime-a".to_owned(),
                models: vec!["model-a".to_owned()],
                dp_size: 1,
                labels: BTreeMap::from([("tier".to_owned(), "gold".to_owned())]),
                zone: Some("zone-1".to_owned()),
                kv_block_size: Some(16),
                bootstrap_port: Some(8_998),
                signals: Signals {
                    events: true,
                    loads: false,
                    scrape_interval_ms: 3_000,
                },
                health: HealthPolicy {
                    // A zero takes the default's value.
                    failure_threshold: default.failure_threshold,
                    success_threshold: 6,
                    check_interval_ms: default.check_interval_ms,
                    timeout_ms: 250,
                    probes: false,
                },
                overload: Overload {
                    // An unset threshold takes the store's default field.
                    kv_usage_permille: Overload::default().kv_usage_permille,
                    waiting_requests: Some(40),
                },
                ..WorkerFacts::default()
            }
        );
    }

    #[test]
    fn a_descriptor_states_the_bootstrap_port_or_leaves_it_as_it_stands() {
        let with = |bootstrap_port| proto::WorkerSpec {
            pairing: Some(proto::PairingDescriptor {
                bootstrap_port,
                ..Default::default()
            }),
            ..spec()
        };
        let port = |stated: &proto::WorkerSpec, standing: Option<&WorkerFacts>| {
            facts(stated, standing)
                .expect("a registration")
                .bootstrap_port
        };
        assert_eq!(port(&spec(), None), None);
        assert_eq!(port(&with(0), None), None);
        assert_eq!(port(&with(8_998), None), Some(8_998));
        assert_eq!(port(&with(65_535), None), Some(65_535));
        // Over a standing row: a port stated replaces it, none stated keeps
        // it, with a descriptor or without.
        let standing = facts(&with(8_998), None).expect("a registration");
        assert_eq!(port(&with(9_001), Some(&standing)), Some(9_001));
        assert_eq!(port(&with(0), Some(&standing)), Some(8_998));
        assert_eq!(port(&spec(), Some(&standing)), Some(8_998));
    }

    #[test]
    fn a_threshold_the_block_leaves_unset_takes_the_stores_default() {
        // Each line reads the store's default field, set or not: the case
        // stands whatever that default is.
        let default = Overload::default();
        let made = |queue_limit, kv_usage_permille| {
            let stated = proto::WorkerSpec {
                overload: Some(proto::OverloadThresholds {
                    queue_limit,
                    kv_usage_permille,
                }),
                ..spec()
            };
            facts(&stated, None).expect("a registration").overload
        };
        assert_eq!(made(None, None), default);
        assert_eq!(
            made(None, Some(900)),
            Overload {
                kv_usage_permille: Some(900),
                ..default
            }
        );
        assert_eq!(
            made(Some(40), None),
            Overload {
                waiting_requests: Some(40),
                ..default
            }
        );
    }

    #[test]
    fn a_registration_over_a_standing_row_keeps_what_it_leaves_unsaid_save_its_labels_and_its_policy(
    ) {
        let standing = full().facts;
        assert!(!standing.health.probes, "the standing row is unchecked");
        let made = facts(&spec(), Some(&standing)).expect("a bare registration");
        assert_eq!(
            made,
            WorkerFacts {
                // What every registration states.
                role: Role::Single,
                wire_modes: WireModes::GRPC,
                // The two facts that are the registration's, whole: it
                // states no label, and no policy, so it is checked under
                // the defaults.
                labels: BTreeMap::new(),
                health: HealthPolicy::default(),
                ..standing
            }
        );
        assert!(made.health.probes);
        // The wire and the role are kept too where a registration names
        // neither, and a number this build does not know is not "unsaid".
        let standing = WorkerFacts {
            wire_modes: WireModes::HTTP,
            ..full().facts
        };
        let unsaid = proto::WorkerSpec {
            wire: 0,
            role: 0,
            ..spec()
        };
        let kept = facts(&unsaid, Some(&standing)).expect("a registration that names neither");
        assert_eq!(
            (kept.role, kept.wire_modes),
            (Role::Decode, WireModes::HTTP)
        );
        for (stated, path) in [
            (proto::WorkerSpec { wire: 3, ..spec() }, "spec.wire"),
            (proto::WorkerSpec { role: 5, ..spec() }, "spec.role"),
        ] {
            assert_eq!(
                facts(&stated, Some(&standing)).map_err(|refused| refused.path),
                Err(path)
            );
        }
    }

    #[test]
    fn a_field_the_store_cannot_hold_is_refused_by_its_path() {
        let signals = |scrape, scrape_interval_ms, family: &str| {
            Some(proto::Signals {
                scrape,
                scrape_interval_ms,
                scrape_family: family.to_owned(),
                ..Default::default()
            })
        };
        let cases: [(proto::WorkerSpec, &str); 18] = [
            (
                proto::WorkerSpec {
                    url: String::new(),
                    ..spec()
                },
                "spec.url",
            ),
            (
                proto::WorkerSpec {
                    url: "grpc://10.0.0.5:8000 ".to_owned(),
                    ..spec()
                },
                "spec.url",
            ),
            (
                proto::WorkerSpec {
                    url: "grpc://10.0.0.5:8000\u{7}".to_owned(),
                    ..spec()
                },
                "spec.url",
            ),
            (
                proto::WorkerSpec {
                    runtime: "runtime a".to_owned(),
                    ..spec()
                },
                "spec.runtime",
            ),
            (
                proto::WorkerSpec {
                    runtime: "runtime-a\n".to_owned(),
                    ..spec()
                },
                "spec.runtime",
            ),
            (
                proto::WorkerSpec {
                    models: vec!["model-a".to_owned(), " model-b".to_owned()],
                    ..spec()
                },
                "spec.models",
            ),
            (
                proto::WorkerSpec {
                    models: vec![String::new()],
                    ..spec()
                },
                "spec.models",
            ),
            (proto::WorkerSpec { wire: 0, ..spec() }, "spec.wire"),
            (proto::WorkerSpec { wire: 3, ..spec() }, "spec.wire"),
            // A host-local wire: no registration names the gateway that
            // holds its socket.
            (
                proto::WorkerSpec {
                    wire: proto::WireMode::Zmq as i32,
                    ..spec()
                },
                "spec.wire",
            ),
            (proto::WorkerSpec { role: 0, ..spec() }, "spec.role"),
            (proto::WorkerSpec { role: 5, ..spec() }, "spec.role"),
            (
                proto::WorkerSpec {
                    dp_size: 2,
                    ..spec()
                },
                "spec.dp_size",
            ),
            (
                proto::WorkerSpec {
                    topology: vec!["rack-1".to_owned()],
                    ..spec()
                },
                "spec.topology",
            ),
            (
                proto::WorkerSpec {
                    overload: Some(proto::OverloadThresholds {
                        queue_limit: None,
                        kv_usage_permille: Some(1_001),
                    }),
                    ..spec()
                },
                "spec.overload.kv_usage_permille",
            ),
            (
                proto::WorkerSpec {
                    signals: signals(true, 0, ""),
                    ..spec()
                },
                "spec.signals.scrape_interval_ms",
            ),
            (
                proto::WorkerSpec {
                    signals: signals(false, 500, ""),
                    ..spec()
                },
                "spec.signals.scrape",
            ),
            (
                proto::WorkerSpec {
                    runtime: "runtime-a".to_owned(),
                    signals: signals(true, 500, "runtime-b"),
                    ..spec()
                },
                "spec.signals.scrape_family",
            ),
        ];
        for (stated, path) in cases {
            let Refused { path: found, why } =
                facts(&stated, None).expect_err("the registration is refused");
            assert_eq!(found, path, "{stated:?}");
            assert!(!why.is_empty());
        }
        // Of a pairing descriptor the store holds the bootstrap port alone:
        // every other field a caller states is refused by its own path.
        let named = |name: &str| name.to_owned();
        let descriptors: [(proto::PairingDescriptor, &str); 14] = [
            (
                proto::PairingDescriptor {
                    protocol: named("protocol-a"),
                    ..Default::default()
                },
                "spec.pairing.protocol",
            ),
            (
                proto::PairingDescriptor {
                    connector: named("connector-a"),
                    ..Default::default()
                },
                "spec.pairing.connector",
            ),
            (
                proto::PairingDescriptor {
                    transport: named("transport-a"),
                    ..Default::default()
                },
                "spec.pairing.transport",
            ),
            (
                proto::PairingDescriptor {
                    version: named("1"),
                    ..Default::default()
                },
                "spec.pairing.version",
            ),
            (
                proto::PairingDescriptor {
                    kv_layout: [(named("layers"), named("32"))].into(),
                    ..Default::default()
                },
                "spec.pairing.kv_layout",
            ),
            (
                proto::PairingDescriptor {
                    ec_producer: true,
                    ..Default::default()
                },
                "spec.pairing.ec_producer",
            ),
            (
                proto::PairingDescriptor {
                    ec_consumer: true,
                    ..Default::default()
                },
                "spec.pairing.ec_consumer",
            ),
            (
                proto::PairingDescriptor {
                    vision: true,
                    ..Default::default()
                },
                "spec.pairing.vision",
            ),
            (
                proto::PairingDescriptor {
                    encode_handoff: proto::EncodeHandoff::Push as i32,
                    ..Default::default()
                },
                "spec.pairing.encode_handoff",
            ),
            (
                proto::PairingDescriptor {
                    ec_connector: named("connector-b"),
                    ..Default::default()
                },
                "spec.pairing.ec_connector",
            ),
            (
                proto::PairingDescriptor {
                    encode_references: true,
                    ..Default::default()
                },
                "spec.pairing.encode_references",
            ),
            (
                proto::PairingDescriptor {
                    bootstrap_host: named("10.0.0.5"),
                    ..Default::default()
                },
                "spec.pairing.bootstrap_host",
            ),
            (
                proto::PairingDescriptor {
                    pair_class: 7,
                    ..Default::default()
                },
                "spec.pairing.pair_class",
            ),
            (
                proto::PairingDescriptor {
                    bootstrap_port: 65_536,
                    ..Default::default()
                },
                "spec.pairing.bootstrap_port",
            ),
        ];
        for (pairing, path) in descriptors {
            let stated = proto::WorkerSpec {
                pairing: Some(pairing),
                ..spec()
            };
            let Refused { path: found, why } =
                facts(&stated, None).expect_err("the descriptor is refused");
            assert_eq!(found, path, "{stated:?}");
            assert!(!why.is_empty());
        }
        // The two zeroes of a width both read one rank.
        for dp_size in [0, 1] {
            let made = facts(&proto::WorkerSpec { dp_size, ..spec() }, None);
            assert_eq!(made.map(|facts| facts.dp_size), Ok(1));
        }
        assert_eq!(
            refused("a", "b"),
            Refused {
                path: "a",
                why: "b"
            }
        );
    }

    #[test]
    fn a_patch_merges_the_labels_and_replaces_the_overload_block() {
        // A label the patch leaves unsaid stays: the patch is merged.
        let mut standing = full().facts;
        standing.labels.insert("zone".to_owned(), "west".to_owned());
        let request = proto::PatchWorkerRequest {
            labels: [
                ("tier".to_owned(), "silver".to_owned()),
                ("lane".to_owned(), "b".to_owned()),
            ]
            .into(),
            remove_labels: vec!["policy".to_owned(), "absent".to_owned()],
            overload: Some(proto::OverloadThresholds {
                queue_limit: Some(12),
                kv_usage_permille: None,
            }),
            ..Default::default()
        };
        let made = patch(&request, &standing)
            .expect("a patch in order")
            .expect("a patch that changes the row");
        assert_eq!(
            made.labels,
            Some(BTreeMap::from([
                ("lane".to_owned(), "b".to_owned()),
                ("tier".to_owned(), "silver".to_owned()),
                ("zone".to_owned(), "west".to_owned()),
            ]))
        );
        assert_eq!(made.priority, None);
        assert_eq!(
            made.overload,
            Some(Overload {
                kv_usage_permille: Overload::default().kv_usage_permille,
                waiting_requests: Some(12),
            })
        );
    }

    #[test]
    fn a_patch_that_changes_nothing_is_none_and_a_label_both_set_and_removed_is_refused() {
        // The row states both thresholds and so does each block below: a
        // block is compared with the row field by field, whatever the
        // store's defaults are.
        let mut standing = full().facts;
        standing.overload.waiting_requests = Some(12);
        let stating = |queue_limit| proto::PatchWorkerRequest {
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            remove_labels: vec!["absent".to_owned()],
            overload: Some(proto::OverloadThresholds {
                queue_limit: Some(queue_limit),
                kv_usage_permille: Some(875),
            }),
            ..Default::default()
        };
        assert_eq!(patch(&stating(12), &standing), Ok(None));
        let empty = proto::PatchWorkerRequest::default();
        assert_eq!(patch(&empty, &standing), Ok(None));
        // A block that differs in one threshold is the patch's one change.
        let made = patch(&stating(13), &standing).expect("a patch in order");
        assert_eq!(
            made.map(|patch| (patch.labels, patch.overload)),
            Some((
                None,
                Some(Overload {
                    kv_usage_permille: Some(875),
                    waiting_requests: Some(13),
                })
            ))
        );

        let both = proto::PatchWorkerRequest {
            labels: [("tier".to_owned(), "silver".to_owned())].into(),
            remove_labels: vec!["tier".to_owned()],
            ..Default::default()
        };
        assert_eq!(
            patch(&both, &standing).map_err(|refused| refused.path),
            Err("remove_labels")
        );
        let over = proto::PatchWorkerRequest {
            overload: Some(proto::OverloadThresholds {
                queue_limit: None,
                kv_usage_permille: Some(4_000),
            }),
            ..Default::default()
        };
        assert_eq!(
            patch(&over, &standing).map_err(|refused| refused.path),
            Err("overload.kv_usage_permille")
        );
    }

    #[test]
    fn a_field_at_its_bound_is_taken_and_one_over_is_refused_by_its_path() {
        type Stating = fn(usize) -> proto::WorkerSpec;
        let cases: [(&str, usize, &str, Stating); 8] = [
            (
                "spec.url",
                URL_BYTES,
                "a URL is at most 2048 bytes",
                |bytes| proto::WorkerSpec {
                    url: text(bytes),
                    ..spec()
                },
            ),
            (
                "spec.runtime",
                NAME_BYTES,
                "a runtime is at most 256 bytes",
                |bytes| proto::WorkerSpec {
                    runtime: text(bytes),
                    ..spec()
                },
            ),
            (
                "spec.zone",
                NAME_BYTES,
                "a zone is at most 256 bytes",
                |bytes| proto::WorkerSpec {
                    zone: text(bytes),
                    ..spec()
                },
            ),
            (
                "spec.models",
                NAME_BYTES,
                "a model id is at most 256 bytes",
                |bytes| proto::WorkerSpec {
                    models: vec!["m".to_owned(), text(bytes)],
                    ..spec()
                },
            ),
            (
                "spec.labels",
                KEY_BYTES,
                "a label's key is at most 128 bytes",
                |bytes| proto::WorkerSpec {
                    labels: [(text(bytes), String::new())].into(),
                    ..spec()
                },
            ),
            (
                "spec.labels",
                VALUE_BYTES,
                "a label's value is at most 4096 bytes",
                |bytes| proto::WorkerSpec {
                    labels: [("k".to_owned(), text(bytes))].into(),
                    ..spec()
                },
            ),
            (
                "spec.labels",
                LABELS,
                "a row holds at most 128 labels",
                |count| {
                    let labels = (0..count).map(|n| (n.to_string(), String::new()));
                    proto::WorkerSpec {
                        labels: labels.collect(),
                        ..spec()
                    }
                },
            ),
            (
                "spec.models",
                MODELS,
                "a row names at most 128 model ids",
                |count| proto::WorkerSpec {
                    models: (0..count).map(|n| n.to_string()).collect(),
                    ..spec()
                },
            ),
        ];
        for (path, bound, why, stating) in cases {
            assert!(facts(&stating(bound), None).is_ok(), "{path} at {bound}");
            assert_eq!(
                facts(&stating(bound + 1), None).map(|_| ()),
                Err(refused(path, why)),
                "{path} past {bound}"
            );
            assert!(why.contains(&bound.to_string()), "{why}");
        }
        // The models' count is read before the list is: past it a
        // registration is refused by the count, whatever its ids hold.
        let unread = proto::WorkerSpec {
            models: vec![text(NAME_BYTES + 1); MODELS + 1],
            ..spec()
        };
        assert_eq!(
            facts(&unread, None).map(|_| ()),
            Err(refused("spec.models", "a row names at most 128 model ids"))
        );
    }

    #[test]
    fn a_url_past_its_rules_is_refused_before_its_row_is_looked_up() {
        let looked = std::cell::Cell::new(0);
        let lookup = |_: &str| {
            looked.set(looked.get() + 1);
            None
        };
        let named = |url: String| proto::WorkerSpec { url, ..spec() };
        let refusals = [
            (String::new(), "a registration names its worker's URL"),
            (text(URL_BYTES + 1), "a URL is at most 2048 bytes"),
            (
                "grpc://10.0.0.5 :8000".to_owned(),
                "a URL holds no blank and no control character",
            ),
        ];
        for (url, why) in refusals {
            assert_eq!(
                registration(&named(url), lookup).map(|_| ()),
                Err(refused("spec.url", why))
            );
        }
        assert_eq!(looked.get(), 0, "a refused URL is never looked up");
        // A URL at its bound is looked up, once.
        assert!(registration(&named(text(URL_BYTES)), lookup).is_ok());
        assert_eq!(looked.get(), 1);
    }

    #[test]
    fn a_row_at_its_bound_is_taken_and_one_byte_over_is_refused_whole() {
        assert_eq!(ROW_BYTES, 16_384);
        let made = facts(&weighing(ROW_BYTES), None).expect("a row at its bound");
        assert_eq!(weight(&made), ROW_BYTES);
        assert_eq!(
            facts(&weighing(ROW_BYTES + 1), None).map(|_| ()),
            Err(refused("spec", "a row is at most 16384 bytes on the wire"))
        );
    }

    #[test]
    fn a_patch_is_held_to_the_bounds_of_the_row_it_would_make() {
        type Asking = fn(usize) -> proto::PatchWorkerRequest;
        let keyed = |labels: Vec<(String, String)>| proto::PatchWorkerRequest {
            labels: labels.into_iter().collect(),
            ..Default::default()
        };
        let cases: [(&str, usize, &str, Asking); 5] = [
            (
                "labels",
                KEY_BYTES,
                "a label's key is at most 128 bytes",
                |bytes| proto::PatchWorkerRequest {
                    labels: [(text(bytes), String::new())].into(),
                    ..Default::default()
                },
            ),
            (
                "labels",
                VALUE_BYTES,
                "a label's value is at most 4096 bytes",
                |bytes| proto::PatchWorkerRequest {
                    labels: [("k".to_owned(), text(bytes))].into(),
                    ..Default::default()
                },
            ),
            (
                "labels",
                LABELS,
                "a row holds at most 128 labels",
                |count| {
                    let labels = (0..count).map(|n| (n.to_string(), String::new()));
                    proto::PatchWorkerRequest {
                        labels: labels.collect(),
                        ..Default::default()
                    }
                },
            ),
            (
                "remove_labels",
                LABELS,
                "a patch removes at most 128 labels",
                |count| {
                    let remove_labels = (0..count).map(|n| n.to_string()).collect();
                    proto::PatchWorkerRequest {
                        remove_labels,
                        ..Default::default()
                    }
                },
            ),
            (
                "remove_labels",
                KEY_BYTES,
                "a label's key is at most 128 bytes",
                |bytes| proto::PatchWorkerRequest {
                    remove_labels: vec![text(bytes)],
                    ..Default::default()
                },
            ),
        ];
        let bare = facts(&spec(), None).expect("a bare row");
        for (path, bound, why, asking) in cases {
            assert!(patch(&asking(bound), &bare).is_ok(), "{path} at {bound}");
            assert_eq!(
                patch(&asking(bound + 1), &bare).map(|_| ()),
                Err(refused(path, why)),
                "{path} past {bound}"
            );
            assert!(why.contains(&bound.to_string()), "{why}");
        }
        // The removals' count is read before the list is: past it a patch
        // is refused by the count, whatever its list holds.
        let both = proto::PatchWorkerRequest {
            labels: [("0".to_owned(), String::new())].into(),
            remove_labels: (0..=LABELS).map(|n| n.to_string()).collect(),
            ..Default::default()
        };
        assert_eq!(
            patch(&both, &bare).map(|_| ()),
            Err(refused(
                "remove_labels",
                "a patch removes at most 128 labels"
            ))
        );
        // The count is of the row the patch would make: a label it removes
        // makes room for the one it sets.
        let mut crowded = bare.clone();
        crowded.labels = (0..LABELS)
            .map(|n| (n.to_string(), String::new()))
            .collect();
        let more = keyed(vec![("more".to_owned(), String::new())]);
        assert_eq!(
            patch(&more, &crowded).map(|_| ()),
            Err(refused("labels", "a row holds at most 128 labels"))
        );
        let swap = proto::PatchWorkerRequest {
            remove_labels: vec!["0".to_owned()],
            ..more
        };
        assert!(patch(&swap, &crowded).is_ok_and(|made| made.is_some()));
        // The row whole: a label of 500 bytes under a one-byte key adds 509.
        let near = facts(&weighing(ROW_BYTES - 509), None).expect("a row under its bound");
        let filling = |bytes| keyed(vec![("p".to_owned(), text(bytes))]);
        let made = patch(&filling(500), &near).expect("a row at its bound");
        let mut after = near.clone();
        after.labels = made
            .and_then(|patch| patch.labels)
            .expect("the labels moved");
        assert_eq!(weight(&after), ROW_BYTES);
        assert_eq!(
            patch(&filling(501), &near).map(|_| ()),
            Err(refused(
                "labels",
                "a row is at most 16384 bytes on the wire"
            ))
        );
        // The policy label is told twice, as a label and under its own
        // field: one of 246 bytes on a row with none adds 510.
        let under = facts(&weighing(ROW_BYTES - 510), None).expect("a row under its bound");
        let policy = |bytes| keyed(vec![("policy".to_owned(), text(bytes))]);
        let made = patch(&policy(246), &under).expect("a row at its bound");
        let mut after = under.clone();
        after.labels = made
            .and_then(|patch| patch.labels)
            .expect("the labels moved");
        assert_eq!(weight(&after), ROW_BYTES);
        assert_eq!(
            patch(&policy(247), &under).map(|_| ()),
            Err(refused(
                "labels",
                "a row is at most 16384 bytes on the wire"
            ))
        );
        // A row at its bound takes a patch of the overload block alone and
        // stays within it at its widest: the block was weighed so.
        let brim = facts(&weighing(ROW_BYTES), None).expect("a row at its bound");
        let block = proto::PatchWorkerRequest {
            overload: Some(proto::OverloadThresholds {
                queue_limit: Some(u32::MAX),
                kv_usage_permille: Some(1_000),
            }),
            ..Default::default()
        };
        let made = patch(&block, &brim).expect("a patch in order");
        let stated = WorkerFacts {
            overload: made.and_then(|patch| patch.overload).expect("the block"),
            ..brim
        };
        let mut widest = row(&WorkerRow {
            id: WorkerId(u64::MAX),
            facts: stated.clone(),
            status: WorkerStatus::Ready,
            revision: u64::MAX,
            ..full()
        });
        widest.incarnation = u64::MAX;
        assert_eq!(widest.encoded_len(), ROW_BYTES);
        // And the row registers again as it stands, its block stated.
        let again = proto::WorkerSpec {
            overload: block.overload,
            ..weighing(ROW_BYTES)
        };
        assert!(facts(&again, Some(&stated)).is_ok_and(|made| made == stated));
        // And a label of the same bytes in other letters still fits it.
        let other = keyed(vec![("a".to_owned(), "b".repeat(VALUE_BYTES))]);
        assert!(patch(&other, &stated).is_ok_and(|made| made.is_some()));
    }

    #[test]
    fn a_rows_weight_is_its_bytes_on_the_wire_with_its_numbers_at_their_widest() {
        // A row whose every text holds `bytes`, a model id and a label's key
        // left empty, under `labels` more labels of no value.
        let edge = |bytes: usize, labels: usize| WorkerRow {
            facts: WorkerFacts {
                key: text(bytes),
                engine: text(bytes),
                engine_kind: text(bytes),
                models: vec![String::new(), text(bytes)],
                zone: Some(text(bytes)),
                labels: (0..labels)
                    .map(|n| (n.to_string(), String::new()))
                    .chain([
                        ("policy".to_owned(), text(bytes)),
                        (String::new(), text(bytes)),
                    ])
                    .collect(),
                ..full().facts
            },
            ..full()
        };
        let bare = WorkerRow {
            facts: facts(&spec(), None).expect("a bare row"),
            ..full()
        };
        // A row that names a runtime and scrapes nothing tells no family.
        let idle = WorkerRow {
            facts: WorkerFacts {
                signals: Signals {
                    scrape_interval_ms: 0,
                    ..full().facts.signals
                },
                ..full().facts
            },
            ..full()
        };
        // The widest a row of these facts is told: the controller's numbers
        // at their most, under a status that sets a health bit, its overload
        // block stated in full. The sum is the wire's own length at each
        // side of a length's second byte, and with a field left empty.
        for stored in [
            full(),
            bare,
            idle,
            edge(127, 127),
            edge(128, 128),
            edge(300, 0),
        ] {
            let widest = WorkerRow {
                id: WorkerId(u64::MAX),
                status: WorkerStatus::Ready,
                revision: u64::MAX,
                ..stored
            };
            let mut told = row(&widest);
            told.incarnation = u64::MAX;
            told.overload = Some(proto::OverloadThresholds {
                queue_limit: Some(u32::MAX),
                kv_usage_permille: Some(1_000),
            });
            let key = widest.facts.key.len();
            assert_eq!(weight(&widest.facts), told.encoded_len(), "a key of {key}");
        }
        // No status a row moves to tells more, and no revision.
        for status in [
            WorkerStatus::Pending,
            WorkerStatus::Ready,
            WorkerStatus::NotReady,
            WorkerStatus::Failed,
            WorkerStatus::Draining,
        ] {
            let at = WorkerRow { status, ..full() };
            assert!(row(&at).encoded_len() <= weight(&at.facts), "{status:?}");
        }
    }

    #[test]
    fn a_page_of_rows_at_their_bound_is_half_a_frame_and_an_answer_far_less() {
        // The contract's page (`WorkerSnapshot`) and a client's default frame.
        const PAGE: usize = 128;
        const FRAME: usize = 4 * 1024 * 1024;
        let fat = WorkerRow {
            facts: facts(&weighing(ROW_BYTES), None).expect("a row at its bound"),
            ..full()
        };
        let page = proto::WorkerSnapshot {
            workers: vec![row(&fat); PAGE],
            last: true,
        };
        // A page frames each row in four bytes.
        assert!(page.encoded_len() <= FRAME / 2 + 4 * PAGE + 2);
        // A replacement's answer carries two rows.
        let answer = event(replaced(&fat, &fat, true), u64::MAX, u64::MAX, u64::MAX);
        assert!(answer.encoded_len() <= 2 * ROW_BYTES + 64);
    }

    #[test]
    fn a_load_row_is_sent_at_its_age_at_the_send() {
        let sampled = Instant::now();
        let sample = LoadSample {
            row: LoadRow {
                running: 3,
                queued: 2,
                waiting_uncached_tokens: 700,
                kv_usage_permille: 815,
                active_kv_permille: 410,
                live_rate: 1_900,
                resident_tokens: 64_000,
                resident_counted: true,
                report_age_ms: 40,
                source: SignalSource::Events,
                max_running: 96,
            },
            received_at: sampled + Duration::from_millis(40),
            sampled_at: sampled,
            source: LoadSource::Stream,
        };
        // Sent 25 ms after it arrived 40 ms old: 65 ms, never under 40.
        let now = sampled + Duration::from_millis(65);
        let sent = load(WorkerId(7), 2, &sample, now);
        assert_eq!((sent.worker_id, sent.dp_rank), (7, 2));
        assert!(sent.resident_counted, "the flag crosses beside the row");
        assert_eq!(
            sent.load,
            Some(proto::LoadRow {
                running: 3,
                queued: 2,
                waiting_uncached_tokens: 700,
                kv_usage_permille: 815,
                active_kv_permille: 410,
                live_rate: 1_900,
                resident_tokens: 64_000,
                report_age_ms: 65,
                source: proto::SignalSource::Events as i32,
                max_running: 96,
            })
        );
        // A send before the sample's own instant reads an age of zero.
        assert_eq!(
            load(WorkerId(7), 2, &sample, sampled)
                .load
                .map(|row| row.report_age_ms),
            Some(0)
        );
    }

    #[test]
    fn every_kind_of_log_entry_is_sent_under_a_case_of_its_own() {
        let at = Position { seq: 3, term: 2 };
        let now = Instant::now();
        let sample = LoadSample {
            row: LoadRow::default(),
            received_at: now,
            sampled_at: now,
            source: LoadSource::Poll,
        };
        let stored = full();
        let entries = [
            Written::Term,
            Written::Load {
                id: WorkerId(7),
                rank: 1,
                sample,
            },
            Written::Member {
                epoch: 1,
                change: MembershipChange::Registered { id: stored.id },
                row: Some(Arc::new(stored.clone())),
                left: None,
            },
        ]
        .map(|event| replication(&Entry { at, event }, now).event);
        assert_eq!(entries[0], Some(Sent::TermStart(proto::TermStart {})));
        assert_eq!(
            entries[1],
            Some(Sent::Load(load(WorkerId(7), 1, &sample, now)))
        );
        assert_eq!(
            entries[2],
            Some(Sent::Membership(event(
                Event::Registered(row(&stored)),
                3,
                1,
                2
            )))
        );
        // A membership entry in a shape the store never tells is sent with
        // no case set: the standby reads an event it does not know.
        let untold = Written::Member {
            epoch: 1,
            change: MembershipChange::Registered { id: stored.id },
            row: None,
            left: None,
        };
        assert_eq!(replication(&Entry { at, event: untold }, now).event, None);
    }
}
