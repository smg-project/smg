//! Every entry of the store, its ingress traits' and the store type's own,
//! and how a standby comes to hold what the entry writes: the one list
//! behind the module's rule that a write of the store is journaled or
//! stands here with what rebuilds it. A case holds the list against the
//! files that declare the entries, so a new entry with no row is a red
//! test, not a standby that silently lacks it.

/// How a standby comes to hold what an entry writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
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
const DISPATCHED: &str = "a new leader reads dispatched leases from the engines' in-flight \
                          snapshots and rebuilds credits from its own tally";
const LEASES: &str = "no journal entry carries an issue or a lease's end yet";
const PUBLISHED: &str = "every replica publishes its own world";

/// The list: `(trait or type, entry, fate)`, each in its file's own order.
pub const ENTRIES: &[(&str, &str, EntryReplication)] = &[
    ("WorkersStore", "epoch", EntryReplication::Read),
    // A count is of the replica's own worker object: no journal names it.
    (
        "WorkersStore",
        "host_count",
        EntryReplication::Rebuilt("each replica's holder hands its own worker objects"),
    ),
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
    ("MemoryStore", "micros", EntryReplication::Read),
    (
        "MemoryStore",
        "publish_now",
        EntryReplication::Rebuilt(PUBLISHED),
    ),
    ("MemoryStore", "publish_due", EntryReplication::Read),
    (
        "MemoryStore",
        "publish",
        EntryReplication::Rebuilt(PUBLISHED),
    ),
    ("MemoryStore", "wake_with", EntryReplication::Read),
    ("MemoryStore", "stats", EntryReplication::Read),
    ("MemoryStore", "outside_lists", EntryReplication::Read),
    ("MemoryStore", "publications", EntryReplication::Read),
    ("MemoryStore", "membership_builds", EntryReplication::Read),
];

#[cfg(test)]
#[path = "../../../tests/unit/service/replication/entries/tests.rs"]
mod tests;
