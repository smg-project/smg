//! Every entry of the store, its ingress traits' and the store type's own,
//! and how a standby comes to hold what the entry writes: the one list
//! behind the module's rule that a write of the store is journaled or
//! stands here with what rebuilds it. A case holds the list against the
//! files that declare the entries, so a new entry with no row is a red
//! test, not a standby that silently lacks it.

/// How a standby comes to hold what an entry writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fate {
    /// The store tells its journal: a standby applies the leader's event.
    Journaled,
    /// Never journaled: a standby rebuilds it itself, from the source named.
    Rebuilt(&'static str),
    /// A write no journal carries yet: a promoted standby starts without
    /// it until what is named journals it.
    NotYet(&'static str),
    /// Writes nothing a standby must hold.
    Read,
}

const HELLO: &str = "a gateway's own fact: it returns in the gateway's hello";
const SUBSCRIPTION: &str = "each replica hears the engines' cache events itself";
const DISPATCHED: &str = "a new leader reads dispatched leases from the engines' in-flight \
                          snapshots and rebuilds credits from its own tally";
const PINS: &str = "pins, reservations and lease ends are journaled once pins have one table";
const GENERATION: &str =
    "the applied generation joins the watched kinds with the step that journals an apply";

/// The list: `(trait or type, entry, fate)`, each in its file's own order.
pub const ENTRIES: &[(&str, &str, Fate)] = &[
    ("WorkersStore", "upsert", Fate::Journaled),
    ("WorkersStore", "patch", Fate::Journaled),
    ("WorkersStore", "remove", Fate::Journaled),
    ("WorkersStore", "set_status", Fate::Journaled),
    ("WorkersStore", "get", Fate::Read),
    ("WorkersStore", "by_key", Fate::Read),
    ("WorkersStore", "epoch", Fate::Read),
    ("WorkersStore", "snapshot", Fate::Read),
    ("WorkersStore", "changes_since", Fate::Read),
    // The status a verdict moves is journaled as the status write it makes.
    // The streak between two moves is the prober's own count: a promoted
    // standby's prober starts it again.
    ("HealthSink", "probe_outcome", Fate::Journaled),
    ("HealthSink", "connected", Fate::Journaled),
    ("HealthSink", "gateway_veto", Fate::Rebuilt(HELLO)),
    ("HealthSink", "gateway_veto_noted", Fate::Rebuilt(HELLO)),
    ("HealthSink", "vetoed", Fate::Read),
    (
        "HealthSink",
        "completed",
        Fate::Rebuilt("the gateways' reports of the last probe interval"),
    ),
    ("HealthSink", "completed_ok_since", Fate::Read),
    ("LoadSink", "record", Fate::Journaled),
    // A report's rows are journaled. Its word that no report came, which
    // clears an engine's overload verdict, is in no entry and no cut: a
    // standby holds the verdict of the engine's last rows until its own
    // first poll.
    ("LoadSink", "report", Fate::Journaled),
    ("LoadSink", "poll_mode", Fate::Read),
    // The report period is the load driver's own setting, told where the
    // driver is built: every replica builds its own.
    (
        "LoadSink",
        "heard_every",
        Fate::Rebuilt("each replica's load driver tells its own report period"),
    ),
    ("LoadSink", "load", Fate::Read),
    ("LoadSink", "ranks", Fate::Read),
    (
        "LoadSink",
        "evict",
        Fate::NotYet("no caller yet: its first caller brings its event"),
    ),
    ("LoadSink", "publish_due", Fate::Read),
    (
        "LoadSink",
        "publish",
        Fate::Rebuilt("every replica publishes its own world"),
    ),
    ("KvEventSink", "admit", Fate::Rebuilt(SUBSCRIPTION)),
    ("KvEventSink", "reconnected", Fate::Rebuilt(SUBSCRIPTION)),
    ("KvEventSink", "resync", Fate::Rebuilt(SUBSCRIPTION)),
    ("KvEventSink", "cursor", Fate::Read),
    ("KvEventSink", "removed", Fate::Rebuilt(SUBSCRIPTION)),
    ("KvEventSink", "block_size", Fate::Read),
    ("KvEventSink", "set_block_size", Fate::Rebuilt(SUBSCRIPTION)),
    ("ReservationStore", "window", Fate::Read),
    ("ReservationStore", "try_reserve", Fate::NotYet(PINS)),
    ("ReservationStore", "release", Fate::NotYet(PINS)),
    ("ReservationStore", "held", Fate::Read),
    ("SessionsStore", "pin", Fate::NotYet(PINS)),
    ("SessionsStore", "lookup", Fate::Read),
    ("SessionsStore", "unpin", Fate::NotYet(PINS)),
    ("SessionsStore", "drop_worker", Fate::NotYet(PINS)),
    ("SessionsStore", "expire", Fate::NotYet(PINS)),
    ("SessionsStore", "merge", Fate::Rebuilt(HELLO)),
    ("SessionsStore", "len", Fate::Read),
    ("SessionsStore", "is_empty", Fate::Read),
    ("LeasesStore", "term", Fate::Read),
    // The term is told to the journal: every later position is under it.
    ("LeasesStore", "set_term", Fate::Journaled),
    ("LeasesStore", "issue", Fate::NotYet(PINS)),
    ("LeasesStore", "dispatched", Fate::Rebuilt(DISPATCHED)),
    ("LeasesStore", "completed", Fate::NotYet(PINS)),
    ("LeasesStore", "released", Fate::NotYet(PINS)),
    ("LeasesStore", "get", Fate::Read),
    ("LeasesStore", "expire", Fate::NotYet(PINS)),
    ("LeasesStore", "restore", Fate::Rebuilt(HELLO)),
    ("LeasesStore", "drop_gateways", Fate::NotYet(PINS)),
    ("LeasesStore", "live", Fate::Read),
    // The store type's own entries. A generation's apply is its one write:
    // a standby holds the generation applied to it, not its leader's.
    ("MemoryStore", "new", Fate::Read),
    ("MemoryStore", "with_publish_window", Fate::Read),
    ("MemoryStore", "journaled", Fate::Read),
    ("MemoryStore", "followed", Fate::Read),
    ("MemoryStore", "cut", Fate::Read),
    (
        "MemoryStore",
        "restore_from",
        Fate::Rebuilt("the journal starts again: a reader re-syncs from this store's own snapshot"),
    ),
    ("MemoryStore", "follow", Fate::Journaled),
    ("MemoryStore", "apply", Fate::NotYet(GENERATION)),
    ("MemoryStore", "world", Fate::Read),
    ("MemoryStore", "build_view", Fate::Read),
    ("MemoryStore", "with_world", Fate::Read),
    ("MemoryStore", "build_pair", Fate::Read),
    ("MemoryStore", "deciding", Fate::Read),
    ("MemoryStore", "micros", Fate::Read),
    ("MemoryStore", "overload_flips", Fate::Read),
    ("MemoryStore", "outside_lists", Fate::Read),
    ("MemoryStore", "publication_allocations", Fate::Read),
    ("MemoryStore", "publications", Fate::Read),
    ("MemoryStore", "resyncs", Fate::Read),
    ("MemoryStore", "indexed_blocks", Fate::Read),
    ("MemoryStore", "lane_counts", Fate::Read),
    ("MemoryStore", "unindexed_rows", Fate::Read),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The store's ingress traits, each with its file's text.
    const TRAITS: [(&str, &str); 7] = [
        ("WorkersStore", include_str!("../store/workers.rs")),
        ("HealthSink", include_str!("../store/health.rs")),
        ("LoadSink", include_str!("../store/loads.rs")),
        ("KvEventSink", include_str!("../store/kv_events.rs")),
        ("ReservationStore", include_str!("../store/reservations.rs")),
        ("SessionsStore", include_str!("../store/sessions.rs")),
        ("LeasesStore", include_str!("../store/leases.rs")),
    ];

    /// The entries of `name` as its file declares them, in order.
    fn declared(name: &str, text: &str) -> Vec<String> {
        let head = format!("pub trait {name}");
        text.lines()
            .skip_while(|line| !line.starts_with(&head))
            .skip(1)
            .take_while(|line| !line.starts_with('}'))
            .filter_map(|line| line.strip_prefix("    fn "))
            .filter_map(|rest| rest.split(['(', '<']).next())
            .map(str::to_owned)
            .collect()
    }

    /// The store type, with its own file's text.
    const OWN: (&str, &str) = ("MemoryStore", include_str!("../store/memory.rs"));

    /// The public entries of the type's own block in `text`, in order.
    fn own(name: &str, text: &str) -> Vec<String> {
        let head = format!("impl {name} {{");
        text.lines()
            .skip_while(|line| *line != head)
            .skip(1)
            .take_while(|line| !line.starts_with('}'))
            .filter_map(|line| line.strip_prefix("    pub fn "))
            .filter_map(|rest| rest.split(['(', '<']).next())
            .map(str::to_owned)
            .collect()
    }

    /// The entries the list holds for `name`, in the list's order.
    fn listed(name: &str) -> Vec<&'static str> {
        ENTRIES
            .iter()
            .filter(|(of, _, _)| *of == name)
            .map(|(_, entry, _)| *entry)
            .collect()
    }

    #[test]
    fn every_entry_of_the_store_stands_on_the_list_once_in_its_order() {
        for (name, text) in TRAITS {
            let declared = declared(name, text);
            assert!(!declared.is_empty(), "{name}: no entry read from its file");
            assert_eq!(
                listed(name),
                declared,
                "{name}: the list and the trait differ"
            );
        }
        let (store, text) = OWN;
        let declared = own(store, text);
        assert!(declared.len() > 1, "{store}: no entry read from its file");
        assert_eq!(
            listed(store),
            declared,
            "{store}: the list and the type differ"
        );
        assert!(
            ENTRIES
                .iter()
                .all(|(of, _, _)| *of == store || TRAITS.iter().any(|(name, _)| name == of)),
            "the list names a trait or a type the store does not have"
        );
    }

    #[test]
    fn a_fate_that_names_a_source_names_one() {
        for (of, entry, fate) in ENTRIES {
            if let Fate::Rebuilt(source) | Fate::NotYet(source) = fate {
                assert!(!source.is_empty(), "{of}::{entry}");
            }
        }
    }
}
