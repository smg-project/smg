//! A decision's round trip over the wire, as a gateway on the controller's
//! host pays it: a served controller (`Controller::bind` and `serve`, its
//! gateway listener on loopback), one `Session`, and warm `Decide` streams,
//! each on a client thread of its own with that thread's connection (the
//! design's one stream a gateway runtime thread). Not a criterion target: a
//! round trip under a fixed count in flight is a distribution read by its
//! tail, and a rate is one number a row.
//!
//! - `policy_decide_stream/round_trip/<in flight>`: one stream holding 1, 8
//!   and 64 decisions in flight over 64 members: the median, the 99th and
//!   the 99.9th percentile of a decision's round trip, from the request's
//!   hand-over to the client's stream to its answer's receipt.
//! - `policy_decide_stream/offered/<rate>/<streams>`: 45,000 and 100,000
//!   decisions a second, one deciding thread's load, offered by an open
//!   loop over one stream and spread over four, whatever the answers do.
//!   A request is due at the start plus its number over its stream's
//!   rate. Due with credit, it is handed over, and its round trip is read
//!   from the due instant to its answer's receipt, so the client's own lag
//!   counts. Due with no credit, it is the gateway's own decision at once,
//!   as the design's gateway decides: counted `local`, never sent. A row
//!   with a fixed count in flight and no pause reads the stream's ceiling;
//!   this one reads a latency at a rate, and the share that went local.
//! - `policy_decide_stream/rate/<members>/<deciders>`: decisions a second
//!   over 8, 64 and 512 members with 128 decisions in flight. On one
//!   decider one model is asked over four streams and two runtime threads.
//!   On eight, 64 models are asked over eight streams and eight runtime
//!   threads, so as many threads feed the deciders as decide: the row
//!   reads its deciders where its rate stands above the one-decider row's
//!   (`rate_over_one`), and its feed where it does not. An eight-decider
//!   row runs over a shared fleet, where every member serves every model
//!   and every decider books its leases on the same members (over 8 and
//!   64 of them on the one member of the least band), and over 64 members
//!   as `/own` too: fleets of their own, 64 members a model, where no two
//!   deciders book on one member. The rate is the streams' measured
//!   answers together over the one interval they share, from the first of
//!   them to the last: a stream that ends early idles in it, so the row
//!   reads no higher than the streams did together.
//! - `policy_decide_stream/gateway/64/<deciders>`: the rate rows over 64
//!   members, both fleets on eight deciders, with each answered request
//!   told on the gateway's `Session` as a gateway tells one served to its
//!   end: a `Dispatched` and a `Completed`, sent by the thread that holds
//!   the session. The frames name the lease the answer carried; where it
//!   carried none they name no lease, which a session takes. A stream
//!   hands the session its reports sixteen at a time: a hand-over wakes
//!   the session's thread, and one a request would time the wake.
//! - `policy_decide_stream/count/<mode>/<members>`, one cell a run, named by
//!   `DECIDE_STREAM_COUNT=<mode>:<members>:<decisions>`: the whole process's
//!   allocator calls a decision, `stream` (the decision asked over one
//!   stream, one at a time so every decision wakes the same threads
//!   whatever its fleet, the client's own calls included) or
//!   `pair` (the `stream` cell under a pair plan, over that many prefills and
//!   as many decodes: each request enters at the pair's prefill, and its one
//!   answer names both workers under two leases of one id). A run's
//!   instructions are read from outside it (`perf stat -e instructions:u`): two
//!   runs of different length give a decision's by their difference. A counted
//!   run's window holds whole exchanges: it opens before the first measured
//!   request is handed over and closes at the last measured answer, which no
//!   request follows. A counted run keeps no round trip and sorts none: a sort
//!   grows faster than its length, and the difference of two runs would keep
//!   part of it.
//!
//! A rate or a gateway row is a bar's cell. It runs `DECIDE_STREAM_PASSES`
//! passes (one by default), each over a controller of its own, and prints
//! the pass of the middle rate among the clean ones (no `Busy` answer and
//! every answer timed; `passes` counts them) with their least and most
//! rate. Beside the process's CPU a decision it prints its user and system
//! parts and, as `<user>+<system>` microseconds a decision, the deciders',
//! the controller's runtime threads' and the gateway's threads'
//! (`clients_us`: the streams' clients and the session), then the switches
//! a decision each of the three asked for (`switches`): all by the
//! kernel's books of each thread, the time split as its ticks fell. A
//! one-decider cell runs one pass more with the deciding thread's cycles
//! and instructions counted in user space (`decider_cycles` and
//! `decider_instructions`, a decision; `n/a` where the kernel gives no
//! counter): a pass apart, whose rate is not read, since a counted thread
//! pays for its counters at every switch. An eight-decider cell prints its
//! rate and its CPU a decision over those of the one-decider cell of its
//! members (`rate_over_one`, `cpu_over_one`).
//!
//! `DECIDE_STREAM_EXPLAIN_SAMPLE` sets the document's sampling rate for
//! each row, with zero as the default. `DECIDE_STREAM_SHADOW` gives a
//! one-side row's binding a twin of its profile as its shadow. Set to
//! `other`, a binding of another model names that shadow. No cell asks a
//! shadow: each then counts a decision under a document that names one.
//!
//! A measured run pins a cell's busy threads to a core each, over the
//! cores the process was given, in their order: the deciders, the
//! controller's runtime threads, the streams' clients, the session, the
//! serving thread, and one core for the allocator's background thread and
//! the election's (`cores`: the six in that order, a slash between them).
//! An eight-decider cell takes 27 cores and a one-decider cell 10. A
//! process given fewer, a test and a target without the call print
//! `cores=none`: such a row is not a pinned cell's, whatever it reads.
//!
//! Every wire row prints the `Busy` answers it met (none is expected: the
//! rows stay inside a stream's window and a decider's queue) and the
//! process's resident size. A closed-loop row also prints its CPU seconds
//! per million decisions, client and controller together; an open-loop
//! client spins between instants, so its row prints none. A reading the
//! kernel's file did not give prints `n/a`. A closed-loop row prints the
//! answers it could not time too (`overtaken`, none expected): an answer
//! whose slot of the ring a later request took while it waited. Every
//! decision of a row or a count cell is a pick over its fleet, or the
//! bench stops: a setup that drifted to another outcome would print
//! another path's numbers. Outside the gateway rows a stream ends the
//! lease an answer carries through its hand at the leases' file as it
//! reads the answer, as a session hands a gateway's frame, so a wire row
//! holds as many leases as it has decisions in flight (a gateway row,
//! those of its frames on their way too) and its numbers hold a lease's
//! issue and its end. A build that files no lease answers with none: its
//! rows run the same, and every wire row prints the leases its measured
//! picks carried (`leases`). A row whose picks carry a lease for some and
//! none for others stops the bench.
//! `DECIDE_STREAM_ROWS=round_trip`, `offered`, `rate` or `gateway` runs one
//! kind alone, and `--test` runs every row for a moment. A run with no
//! `--bench` is a test too, as criterion reads one: `cargo bench` passes
//! `--bench`, and a test of every target passes neither.
//!
//! A row's members report their loads in bands, by one seeded rule
//! (`bands`): dealt in a fixed shuffled order, one member stands alone at
//! the least load, the next two report one request more and score alike,
//! the next one a request more again, and so on by ones and twos; over 512
//! members by eights and sixteens, so the least band holds more rows than
//! a pick has places. A pick's places are then drawn as leases in flight
//! make the least loaded rows tie: over 8 and 64 members one row alone, a
//! band of two and the next band's row; over 512 members from one band.
//! Every member reports 2,097,152 requests beneath its band, more than a
//! row has decisions, and a request names no prompt size, so its lease
//! books no token-work: a lease on file moves no row's cost, and a
//! build that files leases and one that files none pick over the same
//! costs. A dispatch told on the session is credited to its member: the
//! load stage prices one whose request names no size at its mean
//! prefill, one request's work, until the member's next report, and a
//! row's dispatches all fall on its least member. A gateway row's bands
//! therefore stand 4,194,304 requests apart where the other rows' stand
//! one: twice what a wire row has decisions, so its credits stay beneath
//! the next band as its leases stay beneath a report, and no row's place
//! moves. A kind that tells its dispatches needs one row alone in its
//! least band: from 128 members that band's rows tie, and the dispatches
//! told would take them in turn. The bands stand in for the ties a
//! fleet's leases make: here the reports fix them, and a row whose
//! decisions fall under more than one count of the tally stops the bench,
//! its pool having moved. Every wire row prints how the three best rows
//! of its measured decisions scored (`ties`, four counts): each alone,
//! the best alone and the next two alike, the best two alike, all three
//! alike. Two builds are compared where they print one tally.
//!
//! A served decision reads no report of the bench's yet (`READS_LOADS`): a
//! controller's decision is given its load monitor's snapshot, and the
//! bench's members answer no poll, so the bands above are written nowhere
//! yet. Until its members answer the monitor, the bands hold no row's
//! pool: it moves with the leases on file, and a row whose pool moves stops
//! the bench. So no wire row is run, and each kind's line says so. The count
//! cells run: they read a decision's calls of the allocator, not its pool.
//!
//! Every row runs on the allocator the binary registers, under a copy of
//! its settings (`MALLOC_CONF`). A served decision's frame holds lists
//! made on a deciding thread and freed on a stream's, and what a list
//! freed on another thread costs is the allocator's own: over the
//! system's a row reads a rate the binary does not serve. A run that is
//! not on that allocator and those settings is refused as it starts: the
//! allocator is asked for each setting, and for the arena an oversize
//! request of the global allocator must have stood up. Two runs of one
//! count cell are up to a hundred thousand instructions or so apart under
//! it, whatever their length: a decision's instructions are read by the
//! difference of two lengths, as above, never from one run.
//!
//! The process installs a tracing subscriber that takes nothing, as the
//! binary installs one of its own: with none, each span of the transport
//! takes the tracing crate's log path, a cost no served controller pays.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a controller that fails to serve is a bug the tests catch"
)]
#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a decision allocated, and the allocator reads its settings from an exported symbol and answers for them through its control call; a thread's core and its counters are asked of the kernel through calls the standard library wraps none of"
)]

#[path = "decide_stream/allocator.rs"]
mod allocator;
#[path = "decide_stream/clients.rs"]
mod clients;
#[path = "common/filled.rs"]
mod filled;
#[path = "decide_stream/served.rs"]
mod served;
#[path = "decide_stream/threads.rs"]
mod threads;

use std::{
    env,
    io::{self, Write},
    sync::Arc,
    thread,
    time::Duration,
};

use clients::{ask, current_thread, offer, session, Ends, Tally, REPORTS};
use served::{serve, HELD};
use threads::{controller_spent, cpu_ticks, rss_mib, Cores, Counters, Spent, GIVEN};

/// The binary's settings of the shipped allocator and the guard that holds
/// a run to them, where the binary runs on them. An Apple target keeps the
/// allocator's own settings: it compiles no background thread there, and
/// no row read there is the binary's.
#[cfg(all(
    not(target_env = "msvc"),
    not(target_env = "musl"),
    not(target_vendor = "apple")
))]
#[path = "decide_stream/shipped.rs"]
mod shipped;

/// The kernel's calls a row asks for itself: a thread's core, and the
/// deciding thread's counters.
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
#[path = "decide_stream/kernel.rs"]
mod kernel;

/// A target with none of those calls: no thread is pinned and no counter
/// counts.
#[cfg(not(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
)))]
mod kernel {
    pub(super) fn given() -> Vec<usize> {
        Vec::new()
    }

    pub(super) fn pin(_: i32, _: usize) -> bool {
        false
    }

    /// No counter: the type has no value.
    pub(super) enum Counter {}

    impl Counter {
        pub(super) fn open(_: i32, _: u64) -> Option<Self> {
            None
        }

        pub(super) fn count(&self) -> Option<u64> {
            match *self {}
        }
    }
}

/// Whether a served decision reads the load reports a row's members stand
/// in their bands by. A controller's single decision is given its load
/// monitor's snapshot, and the bench's members answer no poll, so no wire
/// row is run: a row's pool moves with the leases on file, and a row's
/// guard stops a pool that moves. It turns when the bench's members answer
/// the load monitor's polls.
const READS_LOADS: bool = false;

/// What a row's members serve.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fleet {
    /// Every member serves every model of the row: every decider books its
    /// leases on the same members, and over 8 and 64 of them on the one
    /// member of the least band.
    Shared,
    /// Each model has the row's count of members to itself, as fleets that
    /// serve one model each: no two deciders book on one member.
    Own,
}

impl Fleet {
    fn name(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::Own => "own",
        }
    }
}

/// A count a decision as a row prints it, to the hundredth.
fn each(count: u64, decisions: u64) -> String {
    let hundredths = (count * 100).checked_div(decisions).unwrap_or(0);
    format!("{}.{:02}", hundredths / 100, hundredths % 100)
}

/// One row's shape.
#[derive(Clone, Copy)]
struct Shape {
    members: u32,
    deciders: usize,
    models: u32,
    /// What the members serve.
    fleet: Fleet,
    streams: u32,
    in_flight: u32,
    /// The controller's runtime threads.
    threads: usize,
    /// The decisions a second an open loop offers each stream; zero keeps
    /// `in_flight` requests in flight instead.
    offered: u64,
    /// The decisions a stream discards, then the ones it measures.
    warm: u64,
    measured: u64,
    /// The allocator counts over the measured decisions.
    counted: bool,
    /// The streams tell each answered request on the gateway's session, as
    /// a gateway does; otherwise they end its lease at the store's door.
    frames: bool,
    /// The row's busy threads sit on a core each, where the process was
    /// given the cores ([`Cores`]).
    pinned: bool,
    /// The deciding thread's cycles and instructions are counted
    /// ([`Counters`]).
    cycles: bool,
    /// A pair plan's row: `members` prefills and as many decodes, each
    /// request asked at the pair's prefill and answered for both sides.
    pair: bool,
}

fn model(index: u32) -> String {
    format!("model-{index}")
}

/// What a wire row measured, over all its streams.
struct Row {
    /// Every measured round trip, sorted.
    nanos: Vec<u64>,
    busy: u64,
    per_second: u64,
    /// CPU milliseconds of the whole process per million decisions, and
    /// its resident size: none where the kernel's file did not read.
    cpu_millis_per_million: Option<u64>,
    /// The process's user and system ticks over the row.
    ticks: Option<(u64, u64)>,
    /// What the deciders, the controller's runtime threads and the
    /// gateway's threads (its streams' clients and its session) spent over
    /// the row; none where the kernel's files did not read.
    deciders: Option<Spent>,
    runtime: Option<Spent>,
    gateway: Option<Spent>,
    /// The cores the row's busy threads sat on, as it prints them; none
    /// for a row that is not pinned.
    cores: Option<String>,
    /// The deciding thread's cycles and instructions in user space, where
    /// the row counted them.
    counted: Option<(u64, u64)>,
    /// The row's decisions, warm and measured, over all its streams.
    decisions: u64,
    rss_mib: Option<u64>,
    calls: u64,
    late: u64,
    local: u64,
    overtaken: u64,
    /// The streams' tallies of ties together ([`Tally::ties`]).
    ties: [u64; 4],
    /// The streams' picks of a short trace together ([`Tally::short`]).
    short: u64,
    /// The leases the streams' measured picks carried.
    leases: u64,
    /// The frames the gateway's session sent for the streams' reports.
    frames: u64,
}

/// Runs one wire row: a served controller, the gateway's session, and the
/// shape's streams, each on its own thread.
fn wire(shape: Shape) -> Row {
    // The floor under the bands stands above the row's whole count of
    // decisions: whatever leases its streams and its session's frames hold
    // at once, a member's leases stay beneath what it reports. A row that
    // tells its dispatches stands its bands twice that floor apart
    // ([`APART`]): the credits of every dispatch it tells, on one member,
    // stay beneath the next band.
    let decisions = u64::from(shape.streams) * (shape.warm + shape.measured);
    assert!(
        decisions < u64::from(HELD),
        "a row of {decisions} decisions reaches the {HELD} requests beneath a member's band"
    );
    let served = serve(shape);
    let addr = served.addr;
    let cores = if shape.pinned { Cores::of(shape) } else { None };
    let (reports, session) = session(addr, cores.as_ref().map(|cores| cores.session));
    let ends = if shape.frames {
        Ends::Session {
            reports: reports.clone(),
            held: Vec::with_capacity(REPORTS),
        }
    } else {
        Ends::Door(Arc::clone(&served.store), None)
    };
    if let Some(cores) = &cores {
        cores.controller();
    }
    let counters = if shape.cycles { Counters::open() } else { None };
    let before = (cpu_ticks(), controller_spent());
    let streams: Vec<_> = (0..shape.streams)
        .zip(0..)
        .map(|(index, place)| {
            let mut ends = ends.clone();
            let core = cores
                .as_ref()
                .and_then(|cores| cores.clients.get(place).copied());
            thread::Builder::new()
                .name(format!("client-{index}"))
                .spawn(move || {
                    if let Some(core) = core {
                        assert!(kernel::pin(0, core), "a client sits on core {core}");
                    }
                    let tally = current_thread().block_on(async move {
                        if shape.offered == 0 {
                            ask(addr, shape, &mut ends).await
                        } else {
                            offer(addr, shape, &mut ends).await
                        }
                    });
                    (tally, Spent::of("thread-self"))
                })
                .expect("a client's thread")
        })
        .collect();
    let (tallies, mut gateway): (Vec<Tally>, Vec<Option<Spent>>) = streams
        .into_iter()
        .map(|stream| stream.join().expect("a client's thread ends cleanly"))
        .unzip();
    // The row's own reporting end kept the session standing for the whole
    // row, whoever reported on it: a gateway whose session ended is let go
    // after its grace period, its streams and its leases with it. With the
    // last end dropped the session tells what is left, ends its stream and
    // is read to its end, so the row's CPU holds every frame.
    drop(ends);
    drop(reports);
    let (frames, told) = session.join().expect("the session's thread ends cleanly");
    gateway.push(told);
    let ticks = cpu_ticks().zip(before.0).map(|(after, before)| {
        (
            after.0.saturating_sub(before.0),
            after.1.saturating_sub(before.1),
        )
    });
    let controller = controller_spent()
        .zip(before.1)
        .map(|(after, before)| (after.0.since(before.0), after.1.since(before.1)));
    let counted = counters.and_then(|counters| counters.read());
    let rss_mib = rss_mib();
    served.stop();

    let mut row = Row {
        nanos: Vec::new(),
        busy: 0,
        per_second: 0,
        cpu_millis_per_million: ticks
            .and_then(|(user, system)| ((user + system) * 10 * 1_000_000).checked_div(decisions)),
        ticks,
        deciders: controller.map(|(deciders, _)| deciders),
        runtime: controller.map(|(_, runtime)| runtime),
        gateway: gateway
            .into_iter()
            .try_fold(Spent::default(), |sum, spent| Some(sum.with(spent?))),
        cores: cores.as_ref().map(Cores::print),
        counted,
        decisions,
        rss_mib,
        calls: 0,
        late: 0,
        local: 0,
        overtaken: 0,
        ties: [0; 4],
        short: 0,
        leases: 0,
        frames,
    };
    // The streams' answers together over the one interval they share, from
    // the first measured answer of any to the last of all: a sum of each
    // stream's own rate would read high by the stagger of their ends.
    let opened = tallies.iter().map(|tally| tally.opened).min();
    let closed = tallies.iter().map(|tally| tally.closed).max();
    let shared = opened
        .zip(closed)
        .map_or(Duration::ZERO, |(opened, closed)| {
            closed.saturating_duration_since(opened)
        });
    let shared = u64::try_from(shared.as_nanos()).expect("an interval under centuries");
    let answered: u64 = tallies.iter().map(|tally| tally.answered).sum();
    row.per_second = (answered * 1_000_000_000).checked_div(shared).unwrap_or(0);
    for tally in tallies {
        row.busy += tally.busy;
        row.calls += tally.calls;
        row.late += tally.late;
        row.local += tally.local;
        row.overtaken += tally.overtaken;
        for (sum, count) in row.ties.iter_mut().zip(tally.ties) {
            *sum += count;
        }
        row.short += tally.short;
        row.leases += tally.leases;
        row.nanos.extend(tally.nanos);
    }
    // A build files a lease for every pick or for none: a row that met both
    // would print numbers of neither.
    let picks: u64 = row.ties.iter().sum();
    assert!(
        row.leases == 0 || row.leases == picks,
        "{} of the row's {picks} measured picks carried a lease",
        row.leases
    );
    // The reports fix the pool, so every decision of a row scores its
    // three best rows one way. A tally under two counts means a row left
    // its band, by a lease on file or by a dispatch's credit: the row
    // would read one pool on a build that files leases and another on one
    // that files none.
    assert!(
        row.ties.iter().filter(|count| **count > 0).count() <= 1,
        "the row's pool moved: its decisions' three best rows scored {}",
        ties(row.ties)
    );
    // A place reads a pick's three best rows. A fleet that has three and
    // traced fewer lost its trace: its picks would all read as no tie.
    assert!(
        row.short == 0 || shape.members < 3,
        "{} of the row's {picks} measured picks traced under three rows",
        row.short
    );
    row.nanos.sort_unstable();
    row
}

impl Row {
    /// Whether the row met no `Busy` answer and timed every answer.
    fn clean(&self) -> bool {
        self.busy == 0 && self.overtaken == 0
    }
}

/// The rate rows' cells, each by its members, its deciders and its fleet:
/// a one-decider cell stands ahead of the eight-decider cells over its
/// members, which are read against it.
const RATE: [(u32, usize, Fleet); 7] = [
    (8, 1, Fleet::Shared),
    (64, 1, Fleet::Shared),
    (512, 1, Fleet::Shared),
    (8, 8, Fleet::Shared),
    (64, 8, Fleet::Shared),
    (64, 8, Fleet::Own),
    (512, 8, Fleet::Shared),
];
/// The gateway rows' cells, in the same form.
const TOLD: [(u32, usize, Fleet); 3] = [
    (64, 1, Fleet::Shared),
    (64, 8, Fleet::Shared),
    (64, 8, Fleet::Own),
];

impl Shape {
    /// A bar's cell over `members`, with 128 decisions in flight: on one
    /// decider, one model asked over four streams and two runtime threads;
    /// on more, 64 models with a stream and a runtime thread a decider,
    /// so the deciders are fed by as many threads as decide.
    fn bar(
        members: u32,
        deciders: usize,
        fleet: Fleet,
        frames: bool,
        warm: u64,
        measured: u64,
    ) -> Self {
        let (models, streams, threads) = if deciders == 1 { (1, 4, 2) } else { (64, 8, 8) };
        Self {
            members,
            deciders,
            models,
            fleet,
            streams,
            in_flight: 128 / streams,
            threads,
            offered: 0,
            warm,
            measured,
            counted: false,
            frames,
            pinned: true,
            cycles: false,
            pair: false,
        }
    }
}

/// A bar's cell: the shape's passes, each a row over a controller of its
/// own.
struct Cell {
    /// The pass of the middle rate among the clean ones ([`Row::clean`]),
    /// or among all of them where none is clean.
    middle: Row,
    /// The clean passes, and the least and the most rate among the passes
    /// the middle is of.
    clean: usize,
    least: u64,
    most: u64,
    /// The deciding thread's cycles and instructions a decision, where a
    /// one-decider cell counted them in a pass of their own.
    counted: Option<(u64, u64)>,
}

/// Runs a bar's cell: `passes` passes of the shape and, on one decider, one
/// more with the deciding thread's counters open, whose rate is not read:
/// a counted thread pays for its counters at every switch.
fn cell(shape: Shape, passes: usize) -> Cell {
    let mut rows: Vec<Row> = (0..passes.max(1)).map(|_| wire(shape)).collect();
    let clean = rows.iter().filter(|row| row.clean()).count();
    if clean > 0 {
        rows.retain(Row::clean);
    }
    rows.sort_unstable_by_key(|row| row.per_second);
    let least = rows.first().map_or(0, |row| row.per_second);
    let most = rows.last().map_or(0, |row| row.per_second);
    let middle = rows.len().saturating_sub(1) / 2;
    let middle = rows.into_iter().nth(middle).expect("a cell runs a pass");
    let counted = (shape.deciders == 1)
        .then(|| {
            wire(Shape {
                cycles: true,
                ..shape
            })
        })
        .and_then(|row| {
            let (cycles, instructions) = row.counted?;
            cycles
                .checked_div(row.decisions)
                .zip(instructions.checked_div(row.decisions))
        });
    Cell {
        middle,
        clean,
        least,
        most,
        counted,
    }
}

/// The value `numerator` in `denominator` of the sorted values are at or
/// under, as microseconds and tenths.
fn percentile(sorted: &[u64], numerator: usize, denominator: usize) -> String {
    let rank = (sorted.len() * numerator).div_ceil(denominator);
    micros(sorted.get(rank.saturating_sub(1)).copied().unwrap_or(0))
}

fn micros(nanos: u64) -> String {
    format!("{}.{}", nanos / 1_000, nanos % 1_000 / 100)
}

fn seconds(millis: u64) -> String {
    format!("{}.{:02}", millis / 1_000, millis % 1_000 / 10)
}

/// A reading of the process as a row prints it: `n/a` where the kernel's
/// file gave none.
fn read<T>(reading: Option<T>, print: impl FnOnce(T) -> String) -> String {
    reading.map_or_else(|| "n/a".to_owned(), print)
}

/// A row's tally of ties as it prints: the four counts in their order.
fn ties(counts: [u64; 4]) -> String {
    let [apart, one_then_two, two_then_one, alike] = counts;
    format!("{apart}/{one_then_two}/{two_then_one}/{alike}")
}

/// One count row, `<mode>:<members>:<decisions>`.
fn count(cell: &str, out: &mut impl Write) -> io::Result<()> {
    let mut parts = cell.split(':');
    let mode = parts.next().unwrap_or_default();
    let mut number = || parts.next().and_then(|part| part.parse::<u64>().ok());
    let (Some(members), Some(decisions)) = (number(), number()) else {
        return writeln!(out, "DECIDE_STREAM_COUNT is <mode>:<members>:<decisions>");
    };
    let shape = Shape {
        members: u32::try_from(members).expect("a fleet that fits"),
        deciders: 1,
        models: 1,
        fleet: Fleet::Shared,
        streams: 1,
        in_flight: 1,
        threads: 2,
        offered: 0,
        warm: 2_000,
        measured: decisions,
        counted: true,
        frames: false,
        pinned: false,
        cycles: false,
        pair: mode == "pair",
    };
    let (calls, busy) = match mode {
        "stream" | "pair" => {
            let row = wire(shape);
            (row.calls, row.busy)
        }
        _ => return writeln!(out, "a count row's mode is stream or pair"),
    };
    writeln!(
        out,
        "policy_decide_stream/count/{mode}/{members} decisions={decisions} allocator_calls={calls} calls_per_decision={}.{:02} busy={busy}",
        calls / decisions.max(1),
        calls % decisions.max(1) * 100 / decisions.max(1),
    )
}

/// Prints a bar's cell: the pass of its middle rate with the passes' least
/// and most, the CPU a decision by thread class, and on one decider the
/// deciding thread's counts, on more the cell over the one-decider cell of
/// its members (`ones`: each one's members, rate and CPU a decision).
fn bar(
    out: &mut impl Write,
    kind: &str,
    shape: Shape,
    passes: usize,
    cell: &Cell,
    ones: &[(u32, u64, Option<u64>)],
) -> io::Result<()> {
    let row = &cell.middle;
    let own = if shape.fleet == Fleet::Own {
        "/own"
    } else {
        ""
    };
    let part = |ticks: Option<u64>| read(ticks, |ticks| each(ticks * 10_000, row.decisions));
    let class = |spent: Option<Spent>| read(spent, |spent| spent.apart(row.decisions));
    let asked = |spent: Option<Spent>| read(spent, |spent| each(spent.asked, row.decisions));
    write!(
        out,
        "policy_decide_stream/{kind}/{}/{}{own} models={} fleet={} streams={} in_flight={} threads={} cores={} passes={}/{passes} decisions={} per_s={} least_per_s={} most_per_s={} p50_us={} p99_us={} busy={} cpu_s_per_million={} user_us={} system_us={} deciders_us={} runtime_us={} clients_us={} switches={}/{}/{} rss_mib={} overtaken={} ties={} leases={}",
        shape.members,
        shape.deciders,
        shape.models,
        shape.fleet.name(),
        shape.streams,
        shape.in_flight,
        shape.threads,
        row.cores.as_deref().unwrap_or("none"),
        cell.clean,
        row.nanos.len(),
        row.per_second,
        cell.least,
        cell.most,
        percentile(&row.nanos, 1, 2),
        percentile(&row.nanos, 99, 100),
        row.busy,
        read(row.cpu_millis_per_million, seconds),
        part(row.ticks.map(|(user, _)| user)),
        part(row.ticks.map(|(_, system)| system)),
        class(row.deciders),
        class(row.runtime),
        class(row.gateway),
        asked(row.deciders),
        asked(row.runtime),
        asked(row.gateway),
        read(row.rss_mib, |mib| mib.to_string()),
        row.overtaken,
        ties(row.ties),
        row.leases,
    )?;
    if shape.frames {
        write!(out, " frames={}", row.frames)?;
    }
    if shape.deciders == 1 {
        writeln!(
            out,
            " decider_cycles={} decider_instructions={}",
            read(cell.counted, |(cycles, _)| cycles.to_string()),
            read(cell.counted, |(_, instructions)| instructions.to_string()),
        )?;
    } else {
        let one = ones
            .iter()
            .find(|(members, _, _)| *members == shape.members);
        writeln!(
            out,
            " rate_over_one={} cpu_over_one={}",
            read(one, |(_, rate, _)| each(row.per_second, *rate)),
            read(
                one.and_then(|(_, _, cpu)| cpu.zip(row.cpu_millis_per_million)),
                |(one, eight)| each(eight, one)
            ),
        )?;
    }
    out.flush()
}

fn main() -> io::Result<()> {
    // A run on another allocator, or under other settings, is refused
    // before a row starts: its rows would pass for the binary's.
    #[cfg(all(
        not(target_env = "msvc"),
        not(target_env = "musl"),
        not(target_vendor = "apple")
    ))]
    shipped::stands()?;
    // As the binary's own subscriber does, this one keeps the transport's
    // spans off the tracing crate's log path.
    tracing::subscriber::set_global_default(tracing::subscriber::NoSubscriber::default())
        .expect("no subscriber stands before the bench's");
    // As criterion reads a run: `cargo bench` passes `--bench`, and a test
    // of every target passes neither flag.
    let quick = env::args().any(|arg| arg == "--test") || !env::args().any(|arg| arg == "--bench");
    if !quick {
        // A measured run pins its bars' threads to these.
        GIVEN.get_or_init(kernel::given);
    }
    let mut out = io::stdout().lock();
    if let Ok(cell) = env::var("DECIDE_STREAM_COUNT") {
        return count(&cell, &mut out);
    }
    let rows = env::var("DECIDE_STREAM_ROWS").unwrap_or_default();
    let (warm, measured) = if quick {
        (200, 2_000)
    } else {
        (20_000, 200_000)
    };
    let all = rows.is_empty();
    if quick && all {
        // A count cell takes any fleet, and one under three members has
        // no third row to trace.
        for cell in ["stream:1:200", "stream:2:200"] {
            count(cell, &mut out)?;
        }
    }
    if !READS_LOADS {
        let kinds = ["round_trip", "offered", "rate", "gateway"];
        for kind in kinds.into_iter().filter(|kind| all || rows == *kind) {
            writeln!(
                out,
                "policy_decide_stream/{kind} not run: a served decision reads no report of the bench's yet"
            )?;
        }
        return Ok(());
    }
    if all || rows == "round_trip" {
        for in_flight in [1, 8, 64] {
            let row = wire(Shape {
                members: 64,
                deciders: 2,
                models: 1,
                fleet: Fleet::Shared,
                streams: 1,
                in_flight,
                threads: 2,
                offered: 0,
                warm,
                measured,
                counted: false,
                frames: false,
                pinned: false,
                cycles: false,
                pair: false,
            });
            writeln!(
                out,
                "policy_decide_stream/round_trip/{in_flight} members=64 deciders=2 decisions={} p50_us={} p99_us={} p999_us={} max_us={} per_s={} busy={} cpu_s_per_million={} rss_mib={} overtaken={} ties={} leases={}",
                row.nanos.len(),
                percentile(&row.nanos, 1, 2),
                percentile(&row.nanos, 99, 100),
                percentile(&row.nanos, 999, 1_000),
                percentile(&row.nanos, 1, 1),
                row.per_second,
                row.busy,
                read(row.cpu_millis_per_million, seconds),
                read(row.rss_mib, |mib| mib.to_string()),
                row.overtaken,
                ties(row.ties),
                row.leases,
            )?;
            out.flush()?;
        }
    }
    if all || rows == "offered" {
        // One model, so one deciding thread takes the whole rate; one
        // stream reads a stream's ceiling, and four spread the rate as the
        // rate rows' gateway does, a stream a runtime thread.
        for (offered, streams) in [(45_000, 1), (100_000, 1), (45_000, 4), (100_000, 4)] {
            let row = wire(Shape {
                members: 64,
                deciders: 2,
                models: 1,
                fleet: Fleet::Shared,
                streams,
                in_flight: 0,
                threads: 2,
                offered: offered / u64::from(streams),
                warm: warm / u64::from(streams),
                measured: measured / u64::from(streams),
                counted: false,
                frames: false,
                pinned: false,
                cycles: false,
                pair: false,
            });
            writeln!(
                out,
                "policy_decide_stream/offered/{offered}/{streams} members=64 deciders=2 decisions={} p50_us={} p99_us={} p999_us={} max_us={} per_s={} busy={} local={} late={} rss_mib={} ties={} leases={}",
                row.nanos.len(),
                percentile(&row.nanos, 1, 2),
                percentile(&row.nanos, 99, 100),
                percentile(&row.nanos, 999, 1_000),
                percentile(&row.nanos, 1, 1),
                row.per_second,
                row.busy,
                row.local,
                row.late,
                read(row.rss_mib, |mib| mib.to_string()),
                ties(row.ties),
                row.leases,
            )?;
            out.flush()?;
        }
    }
    let passes = env::var("DECIDE_STREAM_PASSES")
        .ok()
        .and_then(|passes| passes.parse().ok())
        .unwrap_or(1);
    for (kind, frames, cells) in [("rate", false, &RATE[..]), ("gateway", true, &TOLD[..])] {
        if !all && rows != kind {
            continue;
        }
        let mut ones = Vec::new();
        for &(members, deciders, fleet) in cells {
            let shape = Shape::bar(members, deciders, fleet, frames, warm, measured);
            let cell = cell(shape, passes);
            bar(&mut out, kind, shape, passes, &cell, &ones)?;
            if deciders == 1 {
                let row = &cell.middle;
                ones.push((members, row.per_second, row.cpu_millis_per_million));
            }
        }
    }
    Ok(())
}
