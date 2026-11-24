//! The WebAssembly plugin host (design section 9): user policies as
//! components of the committed `smg:policy` package, run as stages of the
//! framework.
//!
//! What a module is: one component exporting the record world
//! (`smg:policy/policy`, the view as records) or the packed world
//! (`smg:policy/policy-packed`, the view as the bytes of the WIT's layout
//! table); the host reads which from its exports. The host accepts
//! `smg:policy@1.x` from [`Abi::FLOOR_MINOR`] up to
//! [`Abi::SUPPORTED_MINOR`] and a spec may pin the
//! version. `examples/wasm/wasm-guest-policy` is a record-world module to
//! copy, with its build line; `tests/wasm_example.rs` loads what that line
//! builds.
//!
//! The sandbox: no WASI. A component importing any `wasi:` interface, or
//! anything but `smg:policy/host`, is refused at load. The host lends four
//! functions: a rate-limited log with a level cap, the decision's clock, the
//! counters the spec declared, and a draw from the decision's seed.
//!
//! How a module runs: one instance per module per decider, created at apply
//! ([`Plugin::warm`] as every decider; one that was not warmed instantiates
//! at its first decision and the plugin counts it) and kept. A decider is a
//! thread that decides, or the name a host gives one of its deciders
//! ([`DecisionSlot`](crate::engine::decision_slot::DecisionSlot)): a named
//! decider's instance rests with the name and is brought to the thread that
//! acts as it, for a decision or a told event, so it is one instance whichever
//! thread holds the decider's turn. The apply's own instance goes to the first
//! decider that needs one. A thread that names no decider and never decided
//! with the plugin only tells: a lease's event or a signal it tells reaches no
//! instance and makes none ([`Stats::events_dropped`]). The membership is fed
//! apart, to a plugin whose row lists it: the host tells a change once, on
//! whatever thread, with no guest call there, and every instance hears it
//! before its decider's next decision or told event, one call a change. An
//! instance new to the feed, or behind more than the 64 changes the feed keeps,
//! hears the membership as it stands, as one snapshot of ids: the models a
//! worker left, among the changes it missed, are not told. Each profile that
//! binds the plugin holds one [`PluginBinding`] set, taken from its engine
//! build's [`Binder`] (one adapter of a build forwards the lifecycle, so an
//! event reaches the plugin once per engine); the engine opens it once per
//! decision over the profile's view (a pair decision opens one per side) and
//! the stages read one `decision` resource, dropped at the end. The view
//! carries the request's block hashes only where the plugin's row declares them
//! (`needs`, either form), the first
//! [`MAX_BLOCK_HASHES`](crate::document::spec::MAX_BLOCK_HASHES) of a longer
//! list, so a prompt's length sizes no view of a module that reads none. Every
//! call runs under fuel: the plugin's frame budget, or the default frame where
//! it states none
//! ([`FrameBudget::DEFAULT_MICROS`](crate::plugins::shared::FrameBudget::DEFAULT_MICROS)),
//! clamped to the decision's remaining budget (one remainder per thread per
//! decision, shared by every plugin and side it opens, of any runtime built
//! over the same
//! [`OpenDecisions`](crate::plugins::shared::OpenDecisions)), times the host's
//! calibration of fuel to microseconds, and under a bound on the clock of what
//! the decision has left (at most the frame). Fuel prices every operator, a
//! bulk memory operation by its size once it has run; the clock catches what
//! the calibration underprices and the CPU a host call spends. A plugin's bound
//! is measured on the clock the plugin can spend. A guest of these worlds
//! cannot wait by itself (it has no WASI; of its four imports the counter and
//! the draw return at once, and the clock and a log line return when the host's
//! code does: the clock is a number stored for a decision and the host's own
//! clock outside one, a log line runs the host's log subscriber, and a wait
//! there is one the host serves), so the time its call is off the CPU is the
//! host's, and the bound counts the time its thread had the CPU since a reading
//! of the thread's clock: a call ends at the tick that finds it has had its
//! bound, and runs on otherwise, however late the wall clock says it is. A call
//! takes no reading as it starts (the hot path reads no clock): the first tick
//! that finds it in flight takes one, and the bound counts from there. That
//! tick knows nothing of the time before it, so it can end a call and never
//! says one was held. What a late call was (the wall clock put it at or past
//! its bound in whole ticks, the earliest a deadline there could have ended it)
//! is judged once, at its end, on a reading of its start. With none, the call
//! is what it was before the bound counted the CPU: charged the time it took,
//! and counted by name (`unread`: the host held it, or it overran what its fuel
//! did not price). It arms its thread: while its decision's sample is in the
//! plugin's latency ring there (128 decisions) every call reads the clock at
//! its start, and at its end unless it is not late and its fuel covers the time
//! it took (no reading moves that time). It meets no tick before the one its
//! deadline on the wall clock falls on (a reading ends no call the wall clock
//! would let run), and its time is its fuel or the CPU its thread had, late or
//! not. Late and short of its bound on the CPU, a call that answered or
//! declined is the host's (`stalls`: a starved or throttled thread, never a
//! failure of the call): the hold itself drops no instance and moves no
//! breaker, and the call's own time is its fuel or that CPU, never the hold. A
//! hold excuses lateness, never a fault: a held call that failed by itself is
//! that failure alone, and an answer the stage then refuses is counted under
//! its own name beside its stall. At its bound a call is the plugin's. A hold
//! short of late is no late call, and on a thread nothing armed its decision's
//! sample takes the time the call took. So a sample that is slow by time
//! nothing read alone (past half of its frames, or past them, where the fuel
//! of its unread calls leaves it short of that mark) arms the thread too,
//! counted with the unread calls. Where the bound ended a call of the
//! decision under the plugin's own bound it arms the thread and is not
//! counted: at its bound a call is the plugin's. So a ring never holds two
//! slow samples no reading covered, and the latency rule, which wants two
//! slow samples, never opens on time nothing read alone; `unread` counts the
//! first event of each
//! window no reading covered, the host's hold or the guest's own unpriced time,
//! and `stalls` the late holds read inside one, and on a busy thread, where a
//! window is short, the two are read together. This is the rule of this plugin
//! form alone: a plugin that can wait (a remote one; a world that gained an
//! import that blocks) is bounded on the wall clock, and so are every call on a
//! host with no thread clock and an `init`. A cold slot's `init` is timed like
//! a call all the same: its low end is its fuel, and an armed thread reads its
//! clock around it, so a held one is no slow sample there. The thread's clock
//! is the kernel's account of the thread: where a virtual machine's host takes
//! the processor and the kernel counts no stolen time, that time reads as the
//! thread's own, and a call so held is the plugin's, as before. The decision's
//! own deadline stays on the wall clock: a call draws the decision's remainder
//! by the larger of its fuel at the calibration and its elapsed time, so a call
//! the calibration underprices and a held call draw it down too, and a later
//! stage that falls back on a remainder a held call spent is nobody's failure.
//! Nor is a later call that runs out under one: time nothing read draws the
//! decision's remainder, never the plugin's own spend. With no reading a call's
//! time is an interval, from its fuel to its charge (on a host with no thread
//! clock no reading can come, and it is the charge at both ends). The latency
//! rule wants more than one slow sample and takes the high end. A run-out is
//! one event and a verdict, so it convicts on the low end: it is the plugin's
//! when the fuel of its unread calls and the grant reach the budget, the
//! decision's budget spent otherwise (`budget_spent`, the breaker untouched).
//! Where the time those calls took would have made it the plugin's, nothing
//! says whether a hold or their own time past their fuel is the difference: the
//! run-out is counted with the unread calls and arms the thread. There a
//! reading gives every call one time, so a hold short of late leaves the sample
//! too, and the next run-out has one answer. That answer is the plugin's only
//! when every earlier call read whole: a reading is exact, and a thread's clock
//! can run short of its call (a kernel that bills a thread no interrupt time),
//! so behind a call of a millisecond a run-out stays the budget's. A guest slow
//! for that long in what its fuel does not price is held by the latency rule:
//! its decisions sample at their whole budget, and the ring's first read opens
//! the breaker. A frame budget gone on arrival is never forwarded: the stages
//! fall back and the breaker counts a budget hit, the instance standing.
//!
//! What a module hears: `observe` is called with the kinds of lifecycle
//! event the plugin's row lists ([`PluginSpec::observes`]), none by default.
//! A lease's event or a signal is told on the thread that tells it, on the
//! instance of the decider that thread acts as; a membership change reaches
//! an instance on the thread that acts as its decider (the feed, above).
//! Either call is a told event's: it runs under a frame of its own (the
//! plugin's frame budget, a millisecond where the row sets none), outside
//! every decision's budget. So a decision that opens behind the feed waits
//! for the changes it missed, at most the 64 the feed keeps, or for one
//! snapshot, before its first stage call. A row that lists none is never
//! told: no call is made, and a thread that only tells makes no instance. A
//! row that lists any has its plugin's scorer and picker say they need the
//! lifecycle's calls ([`crate::stages::Lifecycle::needs_lifecycle`]), so a host
//! that makes none refuses them.
//!
//! What a failure does: a `decline` falls back to the stage's configured
//! fallback and keeps the instance; a trap, an exhausted budget or an answer
//! that breaks a validity rule (a duplicate or out-of-range filter position,
//! a non-finite score, a pick outside the survivors) falls back too, drops
//! the instance (rebuilt once at the decision's close, for a slot the
//! decision emptied; a faulted instance is never reused) and counts on the
//! breaker. The decision is the breaker's unit: a plugin fails a decision
//! once, however many of its opens fail in it (each side of a pair, each
//! binding), and its later opens of that decision fall back without a call.
//! A call the decision's remainder ends short of the plugin's own bounds (a
//! grant below its frame, by a plugin whose own spend did not consume the
//! budget alone) is the decision's budget spent, its instance dropped and
//! the breaker untouched, whichever scope ran first: under the default
//! frame a plugin is charged only when it alone consumed the budget. A cold
//! slot's `init` is judged under its own frame alone: one the remainder
//! clamps and ends (its charge reached what was left) is the decision's
//! budget spent and the close rebuilds the slot; one that ends by itself
//! short of the clamp is the plugin's own failure, as under the full frame.
//! A resource's destructor runs under a small fuel frame and a bound on the
//! clock of its own, so a spent budget drops no healthy instance and a
//! destructor long on the clock cannot stretch a decision.
//!
//! The breaker opens for a cool-down on a run of failures within its window,
//! or when the moving p50 of the plugin's decisions passes half of the frames
//! the plugin was given in a decision, at most the decision's budget, or the
//! p99 (the second-slowest of the ring) passes them: one sample per
//! decision, the plugin's own time in it (its constructor and stage calls,
//! whatever ended them, each its charge, or for a late call a reading
//! covers its fuel or the CPU its thread had and never the time the host
//! held it; and a lazy `init` the decision paid) over those frames (each
//! call's own frame whole, whatever the remainder left of it, and the init
//! frame of a lazy `init`), taken at its last close, the ring the thread's
//! across rebuilds. An opening writes one warning with the plugin, the rule
//! and the cool-down; a report stamped while the breaker was open, by a call
//! admitted before it opened, is dropped whenever it arrives, so one
//! incident is one opening.
//! Open, the stages take their fallbacks without a call. Without a fallback
//! stage, `failure: open` steps aside (keep all, flat scores, rank by score)
//! and `failure: closed` sheds (an emptied pool or a declined pick naming
//! the plugin).
//!
//! What is not here yet: the spec's `plugins` section that names a module,
//! its memory budget over every row and `max_candidates` (the row lands with
//! the configuration schema; until then a host builds a [`PluginSpec`]
//! itself and no document names a plugin), sourcing (a file or an image
//! reference pulled at validation), and the gRPC plugin service.

mod bindings;
pub(crate) mod config;
mod plugin;
mod view;

pub use config::{Abi, LogLimit, PluginSpec};
pub use plugin::{Binder, LoadError, Plugin, PluginBinding, Runtime, Stats, World};
/// The packed form's layout, as the WIT's table names it.
pub use view::layout;

use crate::plugins::shared;

/// The plugin's `filter` stage: the adapter every plugin kind shares, over
/// this kind's [`PluginBinding`].
pub type WasmFilter = shared::PluginFilter<PluginBinding>;
/// The plugin's `score` stage, over [`PluginBinding`].
pub type WasmScorer = shared::PluginScorer<PluginBinding>;
/// The plugin's `pick` stage, over [`PluginBinding`].
pub type WasmPicker = shared::PluginPicker<PluginBinding>;
