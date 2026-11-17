//! The gRPC user plugin (design section 8, "the user plugin over gRPC"): a
//! routing policy that answers the committed `smg.policy.v1` protocol as a
//! server, bound to a profile as the framework's filter, scorer and picker
//! through the stage adapters every plugin kind shares ([`crate::plugins`]).
//!
//! What a plugin is: a `PolicyPlugin` server behind an address the spec
//! names ([`Transport`]: a Unix socket whose peer credentials are checked at
//! connect, or TCP under mutual TLS with a server-name pin). The engine is
//! the client. The first frame on a stream is the hello: the protocol range,
//! the plugin's name and its opaque config, the exposure, the declared
//! counters, the stage bits the engine will ask for and how many frames it
//! keeps in flight; the plugin's ack fixes the protocol version and the
//! stages it implements, and an ack that says `init` failed fails the apply.
//! A stage a profile binds and the ack leaves out takes its fallback on
//! every decision, and one warning a bound set says so.
//!
//! How a decision runs: one `Decide` stream per deciding thread, opened at
//! the apply ([`Plugin::warm`] on every shard thread) and kept. No decision
//! waits for a connect: a thread whose stream is not open takes its
//! fallbacks at once and the stream opens behind it. One round trip per
//! decision per side, pipelined on the stream and correlated by decision id
//! (a pair's two sides are in flight together). The frame carries the view
//! (the request under the plugin's exposure, every candidate, the pool) and
//! the stage bits the profile binds; the answer carries the survivors, one
//! score per candidate and the pick, each stage's field its own. A plugin
//! bound as a filter or a scorer is asked at the decision's open, so the
//! native stages run while it computes; a plugin bound as the picker alone
//! is asked at the pick, with the combined native scores. The deciding
//! thread waits for the answer only at the first stage that needs it, under
//! the frame's deadline: the plugin's own budget clamped to what the
//! decision has left (one remainder per thread per decision, shared with
//! every other plugin and side the decision opens), and the round trip is
//! drawn from that remainder.
//!
//! What a failure does: a miss of the deadline takes the fallbacks, and
//! whose miss it is has one table, the one every kind reads a run-out by. It
//! is the plugin's failure, counted on the breaker, where the bound was the
//! plugin's own (the frame had its whole budget, or the plugin's own spend in
//! the decision and this grant reach the decision's budget) and the I/O
//! thread was seen running inside a flight longer than its promise. Under
//! the remainder's clamp it is the decision's budget spent. On a stalled I/O
//! thread it is the host's. Under its own bound on a flight no longer than
//! the promise nothing says whose it is: the decision's budget spent, counted
//! unread. Every miss but the host's is in the plugin's latency sample for
//! the decision at what the frame was given, so a plugin that misses every
//! deadline of a budget past half the decision's opens the breaker on its
//! p50 whatever the table says. A `decline`
//! takes the stage's fallback and keeps the stream (a budget decline is the
//! plugin's Busy: the fallback at once, counted, no failure); an answer that
//! breaks a validity rule (a repeated or out-of-range survivor, a non-finite
//! score or the wrong count, a pick outside the plugin's own survivors or
//! outside the rows a pick-only frame named, a declined stage with its field
//! filled, more counter increments than the row declares counters) is the
//! plugin's fault: the frame falls back whole and the breaker counts it. A
//! message longer than the row lets an answer be ends its stream undecoded.
//! The decision is the breaker's unit, as for every kind: a plugin
//! fails a decision once, its later opens of that decision send no frame, and
//! the answer of a frame already in flight for the other side of a pair is
//! not read. A pick over the whole view that the profile's own filters
//! dropped meanwhile is no fault: the picker falls back. A stream that ends
//! or errs counts once, and the breaker hears a plugin's streams as one: at
//! most one failure a pause, so a plugin that goes away is one outage, not
//! one a deciding thread. A stream is reopened behind a decision of its thread,
//! no sooner than a pause (100 ms) after the failure, whatever the breaker
//! is set to; a wait its stream broke draws the time its thread blocked; a
//! frame past the stream's window is not sent; a pool larger than the
//! plugin's `max_candidates` sends no frame (whole view or none: a truncated
//! view would make the answer depend on the host's candidate order), takes
//! the fallbacks and counts `too_many`; a budget gone on arrival sends no
//! frame and counts a budget hit; the open breaker bypasses every stage for
//! the cool-down, and its opening writes one warning with the plugin, the
//! rule and the cool-down. Without a fallback stage, `failure: open` steps
//! aside and `failure: closed` sheds, as for every kind.
//!
//! The lifecycle: one `Observe` stream per plugin, opened once a `Decide`
//! stream has settled, carries the events the engine forwards (issued,
//! dispatched, completed, released, membership, signals) in order, each
//! numbered by the I/O thread where it enters the stream, so the numbers
//! never go down whichever threads raise the events; an event the queue
//! cannot take at once is dropped and counted,
//! never a wait on a deciding thread. A number counts the events sent and
//! the events refused, for the plugin's life and across its streams, so a
//! gap tells the plugin what it lost: events refused at a full queue, or in
//! flight on a stream that ended. A gap sits where the loss fell: an event
//! carries the refusals its thread had seen as it entered the queue, so
//! every event raised before a loss arrives ahead of its gap, and a loss
//! after the last event raised shows at the next. The stream's task is
//! started again a pause after one that panicked, the queue and the numbers
//! kept. Each profile
//! that binds the plugin holds one [`Bound`] set, taken from its engine
//! build's [`Binder`]: one adapter of a build forwards, so an event reaches
//! the plugin once per engine.
//!
//! The runtime: one I/O thread per process ([`Runtime`]) owns every stream;
//! a deciding thread hands it a frame and waits on that frame's place in
//! its stream's table, the only block the plugin adds to a decision,
//! bounded by the budget. That thread is the host's part of every round
//! trip, so it keeps a witness: while a frame is in flight it stamps the
//! instant it ran once a millisecond, and at every answer. A frame
//! answered or missed while the thread had not run for longer than its
//! promise (four milliseconds) is the host's: counted as a stall and left
//! out of the plugin's latency sample; a missed one falls back all the
//! same. With no frame in flight nothing ticks. A frame whose own budget
//! is no longer than the promise is excused only where another frame's
//! flight already shows the thread stalled, and its miss is never the
//! plugin's failure: alone it ends before the thread can be seen late. An
//! answer in its place when the thread comes to read it is used whenever
//! it arrived and sampled at its whole flight: a miss is a frame the thread
//! stopped waiting for, and a late answer's lateness is the latency rule's.
//!
//! What is not here yet: the spec's `plugins` row that names a gRPC plugin
//! by address (it lands with the configuration schema; until then a host
//! builds a [`GrpcPluginSpec`] itself), the view messages rendered from the
//! engine's types as the WIT is, and the controller's shards, where one
//! stream carries many decisions at once (its window holds one decision's
//! frames today).

pub mod config;
mod plugin;
mod runtime;
mod unit;
mod view;

pub use config::{
    BreakerSpec, Exposure, Failure, GrpcPluginSpec, HeaderSet, Reveal, Tls, Transport,
};
pub use plugin::{Binder, Bound, LoadError, Plugin, Stats};
pub use runtime::Runtime;

pub use crate::plugins::Counters;

/// The plugin's stages over its bound decision: the stage adapters every
/// plugin kind shares ([`crate::plugins`]), over [`Bound`].
pub type GrpcFilter = crate::plugins::PluginFilter<Bound>;
/// See [`GrpcFilter`].
pub type GrpcScorer = crate::plugins::PluginScorer<Bound>;
/// See [`GrpcFilter`].
pub type GrpcPicker = crate::plugins::PluginPicker<Bound>;

/// The protocol version this host speaks: the one committed major.
pub const PROTOCOL: u32 = 1;
