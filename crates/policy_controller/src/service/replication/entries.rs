//! Every entry of the store, its ingress traits' and the store type's own,
//! and how a standby comes to hold what the entry writes: the one list
//! behind the module's rule that a write of the store is journaled or
//! stands here with what rebuilds it. A case holds the list against the
//! files that declare the entries, so a new entry with no row is a red
//! test, not a standby that silently lacks it.

/// How a standby comes to hold what an entry writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryReplication {
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
const LEASES: &str = "no journal entry carries a reservation, an issue or a lease's end yet";
const PUBLISHED: &str = "every replica publishes its own world";
const REPORTED: &str = "a host's own writer states the values again at each engine's next \
                        load report, and a controller's serve writes none";

/// The list: `(trait or type, entry, fate)`, each in its file's own order.
pub const ENTRIES: &[(&str, &str, EntryReplication)] = &[
    ("WorkersStore", "upsert", EntryReplication::Journaled),
    ("WorkersStore", "upsert_by", EntryReplication::Journaled),
    ("WorkersStore", "patch", EntryReplication::Journaled),
    ("WorkersStore", "remove", EntryReplication::Journaled),
    ("WorkersStore", "set_status", EntryReplication::Journaled),
    ("WorkersStore", "get", EntryReplication::Read),
    ("WorkersStore", "by_key", EntryReplication::Read),
    ("WorkersStore", "epoch", EntryReplication::Read),
    ("WorkersStore", "all_rows", EntryReplication::Read),
    ("WorkersStore", "changes_since", EntryReplication::Read),
    // Each replica counts the snapshots it takes.
    ("WorkersStore", "history", EntryReplication::Read),
    // A count is of the replica's own worker object: no journal names it.
    (
        "WorkersStore",
        "host_count",
        EntryReplication::Rebuilt("each replica's holder hands its own worker objects"),
    ),
    // The status a verdict moves is journaled as the status write it makes.
    // The streak between two moves is the prober's own count: a promoted
    // standby's prober starts it again.
    ("HealthSink", "probe_outcome", EntryReplication::Journaled),
    ("HealthSink", "connected", EntryReplication::Journaled),
    (
        "HealthSink",
        "gateway_veto",
        EntryReplication::Rebuilt(HELLO),
    ),
    (
        "HealthSink",
        "gateway_veto_noted",
        EntryReplication::Rebuilt(HELLO),
    ),
    ("HealthSink", "vetoed", EntryReplication::Read),
    (
        "HealthSink",
        "completed",
        EntryReplication::Rebuilt("the gateways' reports of the last probe interval"),
    ),
    ("HealthSink", "completed_ok_since", EntryReplication::Read),
    // A log carries no load row: a standby holds none of its leader's and
    // no overload verdict. Its first poll goes out at its take.
    ("LoadSink", "record", EntryReplication::Rebuilt(REPORTED)),
    ("LoadSink", "report", EntryReplication::Rebuilt(REPORTED)),
    (
        "LoadSink",
        "report_sampled",
        EntryReplication::Rebuilt(REPORTED),
    ),
    ("LoadSink", "poll_mode", EntryReplication::Read),
    // The report period is the setting of a host's own writer of load
    // rows, told where that writer is built.
    (
        "LoadSink",
        "heard_every",
        EntryReplication::Rebuilt("a host's own writer of load rows tells its report period"),
    ),
    ("LoadSink", "load", EntryReplication::Read),
    ("LoadSink", "ranks", EntryReplication::Read),
    // A host's own writer drops the rows it wrote: each replica's own
    // rows, and a standby holds none.
    (
        "LoadSink",
        "evict",
        EntryReplication::Rebuilt("each replica drops its own load rows"),
    ),
    ("LoadSink", "publish_due", EntryReplication::Read),
    ("LoadSink", "publish", EntryReplication::Rebuilt(PUBLISHED)),
    ("LoadSink", "wake_with", EntryReplication::Read),
    ("SignalSink", "signal_columns", EntryReplication::Read),
    (
        "SignalSink",
        "signal_value",
        EntryReplication::Rebuilt(REPORTED),
    ),
    (
        "KvEventSink",
        "admit",
        EntryReplication::Rebuilt(SUBSCRIPTION),
    ),
    (
        "KvEventSink",
        "admit_through",
        EntryReplication::Rebuilt(SUBSCRIPTION),
    ),
    (
        "KvEventSink",
        "reconnected",
        EntryReplication::Rebuilt(SUBSCRIPTION),
    ),
    (
        "KvEventSink",
        "resync",
        EntryReplication::Rebuilt(SUBSCRIPTION),
    ),
    ("KvEventSink", "cursor", EntryReplication::Read),
    (
        "KvEventSink",
        "removed",
        EntryReplication::Rebuilt(SUBSCRIPTION),
    ),
    ("KvEventSink", "block_size", EntryReplication::Read),
    (
        "KvEventSink",
        "set_block_size",
        EntryReplication::Rebuilt(SUBSCRIPTION),
    ),
    (
        "KvEventSink",
        "forget_block_size",
        EntryReplication::Rebuilt(SUBSCRIPTION),
    ),
    ("ReservationStore", "window", EntryReplication::Read),
    (
        "ReservationStore",
        "try_reserve",
        EntryReplication::NotYet(LEASES),
    ),
    (
        "ReservationStore",
        "release",
        EntryReplication::NotYet(LEASES),
    ),
    ("ReservationStore", "held", EntryReplication::Read),
    ("LeasesStore", "term", EntryReplication::Read),
    // The term is told to the journal: every later position is under it.
    ("LeasesStore", "set_term", EntryReplication::Journaled),
    ("LeasesStore", "issue", EntryReplication::NotYet(LEASES)),
    (
        "LeasesStore",
        "issue_pair",
        EntryReplication::NotYet(LEASES),
    ),
    (
        "LeasesStore",
        "issue_decode",
        EntryReplication::NotYet(LEASES),
    ),
    (
        "LeasesStore",
        "dispatched",
        EntryReplication::Rebuilt(DISPATCHED),
    ),
    ("LeasesStore", "completed", EntryReplication::NotYet(LEASES)),
    ("LeasesStore", "released", EntryReplication::NotYet(LEASES)),
    ("LeasesStore", "get", EntryReplication::Read),
    ("LeasesStore", "sides", EntryReplication::Read),
    ("LeasesStore", "expire", EntryReplication::NotYet(LEASES)),
    ("LeasesStore", "restore", EntryReplication::Rebuilt(HELLO)),
    (
        "LeasesStore",
        "drop_gateways",
        EntryReplication::NotYet(LEASES),
    ),
    ("LeasesStore", "live", EntryReplication::Read),
    // The store type's own entries. A generation's apply is its one write.
    ("MemoryStore", "new", EntryReplication::Read),
    ("MemoryStore", "with_publish_window", EntryReplication::Read),
    ("MemoryStore", "journaled", EntryReplication::Read),
    ("MemoryStore", "followed", EntryReplication::Read),
    ("MemoryStore", "checkpoint", EntryReplication::Read),
    ("MemoryStore", "rows_serving", EntryReplication::Read),
    (
        "MemoryStore",
        "restore_from",
        EntryReplication::Rebuilt(
            "the journal starts again: a reader re-syncs from this store's own snapshot",
        ),
    ),
    (
        "MemoryStore",
        "watch_from",
        EntryReplication::Rebuilt("a follower takes the worker rows from a watch snapshot"),
    ),
    // The two doors for a followed write, one a kind of the leader's
    // events: each is the standby's side of a journaled write.
    ("MemoryStore", "follow", EntryReplication::Journaled),
    ("MemoryStore", "follow_term", EntryReplication::Journaled),
    // The generation's number is journaled, one of the watched kinds. The
    // document it runs from is its door's to hold: whoever sends the event
    // joins the two, and a standby applies the document through its own
    // door under the leader's number.
    ("MemoryStore", "apply", EntryReplication::Journaled),
    (
        "MemoryStore",
        "follow_generation",
        EntryReplication::Journaled,
    ),
    (
        "MemoryStore",
        "follow_generation_place",
        EntryReplication::Journaled,
    ),
    ("MemoryStore", "snapshot", EntryReplication::Read),
    ("MemoryStore", "with_snapshot", EntryReplication::Read),
    ("MemoryStore", "deciding", EntryReplication::Read),
    ("MemoryStore", "prompt_hashes", EntryReplication::Read),
    ("MemoryStore", "micros", EntryReplication::Read),
    (
        "MemoryStore",
        "publish_now",
        EntryReplication::Rebuilt(PUBLISHED),
    ),
    ("MemoryStore", "stats", EntryReplication::Read),
    ("MemoryStore", "overload_flips", EntryReplication::Read),
    ("MemoryStore", "outside_lists", EntryReplication::Read),
    (
        "MemoryStore",
        "publication_allocations",
        EntryReplication::Read,
    ),
    ("MemoryStore", "publications", EntryReplication::Read),
    ("MemoryStore", "membership_builds", EntryReplication::Read),
    ("MemoryStore", "resyncs", EntryReplication::Read),
    ("MemoryStore", "indexed_blocks", EntryReplication::Read),
    ("MemoryStore", "lane_counts", EntryReplication::Read),
    ("MemoryStore", "unindexed_rows", EntryReplication::Read),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The store's ingress traits, each with its file's text.
    const TRAITS: [(&str, &str); 7] = [
        ("WorkersStore", include_str!("../../store/workers.rs")),
        ("HealthSink", include_str!("../../store/health.rs")),
        ("LoadSink", include_str!("../../store/loads.rs")),
        ("SignalSink", include_str!("../../store/signals.rs")),
        ("KvEventSink", include_str!("../../store/kv_events.rs")),
        (
            "ReservationStore",
            include_str!("../../store/reservations.rs"),
        ),
        ("LeasesStore", include_str!("../../store/leases.rs")),
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
    const OWN: (&str, &str) = ("MemoryStore", include_str!("../../store/memory.rs"));

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
            if let EntryReplication::Rebuilt(source) | EntryReplication::NotYet(source) = fate {
                assert!(!source.is_empty(), "{of}::{entry}");
            }
        }
    }
}
