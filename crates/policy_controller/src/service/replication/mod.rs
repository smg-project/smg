//! Replication to standbys (design 5.4): the log of the store's replicated
//! writes, which a standby follows to become the leader's twin.
//!
//! One invariant: **the log is the store's own history.** Every replicated
//! write the store applied is in it, in the order the store applied it, each
//! position under the term the store wrote it under. Who may serve that
//! history to a standby is the fence's judgment
//! ([`Fence::leader_term`](crate::service::elector::Fence::leader_term)), never the
//! log's: a replica that lost its lease may write its own history a moment
//! longer, and the positions it wrote under its old term are exactly what
//! the rejoin rule finds nobody else holds. Once it follows another leader
//! (an applied event, a snapshot), a write of its own is under no term at
//! all: the term it led already named other writes at those positions.
//!
//! What the store tells its [`Journal`](crate::store::Journal), each inside
//! the write's own critical section: a membership or status write with its
//! row, an applied generation by its number ([`Event::Generation`]), and
//! the store's take of a term, which is a position of its own
//! ([`Event::Term`]). A standby journals that event too at the place of a
//! load row an older leader tells it. Every other entry of the store stands on one
//! list with what rebuilds it on a standby ([`ENTRIES`]): a gateway's own
//! facts return in its hello, each replica hears the engines' cache events
//! itself, and a decision's writes (pins, reservations, lease ends) are not
//! journaled yet. A generation's document is not the log's to hold: the
//! door that applied it keeps the bytes, and the stream that sends the
//! event joins them.
//!
//! Two readers follow the log: the controller's standby and a watcher of
//! the rows both read the watched kinds (membership, status, a term's
//! start). The log keeps them in one ring, in the one order of positions,
//! and it tells one listener when they grew, once the store that wrote has
//! let its locks go ([`Log::on_watched`]). The log carries no load row: a
//! load write tells nothing there.
//!
//! What is here: the bounded log ([`Log`]) and the reads made of it
//! ([`Log::holds`] for the place a reader brings from another connection or
//! replica, [`Log::read`] for its batches, each from the reader's place
//! with its term), and the standby's side of the same type ([`Log::adopt`]:
//! an applied event takes the leader's position and term; [`Log::restart`]
//! after a snapshot), so a position and its term name one history on every
//! replica that holds them. A write never waits on a reader: a reader whose
//! place the log let go is told so ([`LogPositionLost`]) and starts again from
//! a cut ([`MemoryStore::checkpoint`](crate::store::MemoryStore::checkpoint):
//! the rows and the log's place, as one).
//!
//! The stream that serves the log to a standby, the standby that follows
//! it and the watch are later steps of the controller.

mod entries;
mod log;
#[cfg(test)]
mod tests;

pub use entries::{EntryReplication, ENTRIES};
pub use log::{AdoptionGuard, Entry, Event, Log, LogPositionLost, NotLeading, ReadStamp};
