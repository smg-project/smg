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
//! controller's decision is given its load monitor's snapshot, and the bench
//! writes its members' reports into the store's table, which no decision
//! reads. Until its members answer the monitor, the bands hold no row's
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

#[path = "common/filled.rs"]
mod filled;

use std::{
    alloc::{GlobalAlloc, Layout},
    env, fs,
    io::{self, Write},
    mem,
    net::{Ipv4Addr, SocketAddr},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc as std_mpsc, Arc, OnceLock,
    },
    thread,
    time::{Duration, Instant},
};

use smg_policy_controller::{
    document::{AppliedSpec, PolicySpec, ShadowSpec},
    engine::{LoadRow, Role},
    service::{
        controller::{Config, Controller, ElectionConfig, Listener, Scope, State},
        proto::{
            completed::Class, controller_frame, decide_answer::Answer, decision::Outcome,
            gateway_frame::Frame, policy_controller_client::PolicyControllerClient, Completed,
            ControllerFrame, DecideRequest, Decision, Dispatched, GatewayFrame, Hello,
            Phase as WirePhase, RequestFacts as WireFacts, Side,
        },
    },
    store::{
        GatewayId, Hand, LoadSink, LoadSource, MemoryStore, WorkerFacts, WorkerStatus, WorkersStore,
    },
    tracking::{LeaseId, ReleaseReason, Said},
};
use tokio::{
    runtime,
    sync::{mpsc, oneshot},
};
use tonic::{
    codegen::tokio_stream::wrappers::ReceiverStream, metadata::AsciiMetadataValue,
    transport::Channel, Request, Streaming,
};

/// Whether the allocator counts: on for a count row's measured decisions.
static COUNTING: AtomicBool = AtomicBool::new(false);
/// The allocation and reallocation calls of every thread while it counts.
static CALLS: AtomicU64 = AtomicU64::new(0);

/// The allocator every row runs on: the one the binary registers
/// (`src/bin/smg-policy-controller.rs`), on the targets it registers it. A
/// served decision's lists are made on one thread and freed on another,
/// and what that costs is the allocator's: a row read over another one is
/// not the binary's row.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
static SHIPPED: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// The system allocator, on a target where the binary registers none.
#[cfg(any(target_env = "msvc", target_env = "musl"))]
static SHIPPED: std::alloc::System = std::alloc::System;

/// The binary's settings of the shipped allocator and the guard that holds
/// a run to them, where the binary runs on them. An Apple target keeps the
/// allocator's own settings: it compiles no background thread there, and
/// no row read there is the binary's.
#[cfg(all(
    not(target_env = "msvc"),
    not(target_env = "musl"),
    not(target_vendor = "apple")
))]
mod shipped {
    use std::{
        ffi::{c_char, c_int, c_uint, c_void, CStr, CString},
        hint::black_box,
        io, ptr,
    };

    /// The settings, in the symbol the allocator reads them from as it
    /// starts: a copy of the binary's line, which a bench cannot import. A
    /// case reads both sources by path (`tests/binary.rs`): the copy moves
    /// with the binary's, or that case fails.
    #[export_name = "_rjem_malloc_conf"]
    pub static MALLOC_CONF: &[u8; 100] =
        b"background_thread:true,dirty_decay_ms:10000,muzzy_decay_ms:0,narenas:16,oversize_threshold:33554432\0";

    extern "C" {
        /// The allocator's control call, under the prefix its build gives
        /// every name of its own: the one [`MALLOC_CONF`] is exported
        /// under.
        #[link_name = "_rjem_mallctl"]
        fn mallctl(
            name: *const c_char,
            old: *mut c_void,
            old_len: *mut usize,
            new: *mut c_void,
            new_len: usize,
        ) -> c_int;
    }

    /// What the allocator answers for `name`, where the answer is a `T`:
    /// `None` for a name it does not know or an answer of another size.
    fn asked<T: Copy + Default>(name: &CStr) -> Option<T> {
        let mut answer = T::default();
        let mut len = size_of::<T>();
        // SAFETY: `name` ends in NUL, and the allocator writes `len` bytes
        // at most into `answer`, a plain value of that size. Nothing is
        // written to the allocator.
        let refused = unsafe {
            mallctl(
                name.as_ptr(),
                (&raw mut answer).cast(),
                &raw mut len,
                ptr::null_mut(),
                0,
            )
        };
        (refused == 0 && len == size_of::<T>()).then_some(answer)
    }

    /// Refuses a run that is not on the binary's allocator under the
    /// binary's settings: its rows would pass for the binary's. The
    /// allocator answers for itself. Each entry of [`MALLOC_CONF`] must
    /// read back as the option it sets, whatever the environment adds to
    /// the line. And the global allocator must be this one: asked for as
    /// much as its oversize threshold, it stands up the arena it keeps for
    /// such requests, the one past the `narenas` it spreads threads over,
    /// which a request to another allocator does not. That request is one
    /// mapping, made and returned before a row starts.
    pub(super) fn stands() -> io::Result<()> {
        let refused = |why: String| {
            io::Error::other(format!(
                "this run is not on the binary's allocator and settings: {why}"
            ))
        };
        let line = MALLOC_CONF
            .strip_suffix(b"\0")
            .and_then(|line| str::from_utf8(line).ok())
            .ok_or_else(|| refused("the settings line is no text that ends in NUL".into()))?;
        for entry in line.split(',') {
            let (key, value) = entry.split_once(':').unwrap_or((entry, ""));
            let read = match key {
                "background_thread" => {
                    asked::<bool>(c"opt.background_thread").map(|on| on.to_string())
                }
                "dirty_decay_ms" => asked::<isize>(c"opt.dirty_decay_ms").map(|ms| ms.to_string()),
                "muzzy_decay_ms" => asked::<isize>(c"opt.muzzy_decay_ms").map(|ms| ms.to_string()),
                "narenas" => asked::<c_uint>(c"opt.narenas").map(|arenas| arenas.to_string()),
                "oversize_threshold" => {
                    asked::<usize>(c"opt.oversize_threshold").map(|bytes| bytes.to_string())
                }
                // A setting the guard cannot ask for is one it cannot hold.
                _ => None,
            };
            if read.as_deref() != Some(value) {
                let read = read.as_deref().unwrap_or("nothing");
                return Err(refused(format!(
                    "`{key}` reads back as {read}, the line says {value}"
                )));
            }
        }
        let asked_by =
            asked::<usize>(c"opt.oversize_threshold").zip(asked::<c_uint>(c"opt.narenas"));
        let Some((threshold, arenas)) = asked_by else {
            return Err(refused("it names no oversize threshold".into()));
        };
        drop(black_box(Vec::<u8>::with_capacity(threshold)));
        // The allocator's answers on its arenas are of its last epoch.
        let mut epoch = 1_u64;
        // SAFETY: the name ends in NUL, nothing is read back, and the
        // allocator reads the eight bytes of `epoch`.
        let stale = unsafe {
            mallctl(
                c"epoch".as_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
                (&raw mut epoch).cast(),
                size_of::<u64>(),
            )
        };
        let arena = CString::new(format!("arena.{arenas}.initialized"))
            .map_err(|_| refused("an arena's name holds a NUL".into()))?;
        if stale != 0 || asked::<bool>(&arena) != Some(true) {
            return Err(refused(format!(
                "a request of {threshold} bytes did not reach it"
            )));
        }
        Ok(())
    }
}

/// The shipped allocator, counting its calls while [`COUNTING`] is on. The
/// rows that read time pay one read of a flag nobody writes.
struct Counted;

fn count_call() {
    if COUNTING.load(Ordering::Relaxed) {
        CALLS.fetch_add(1, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for Counted {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_call();
        // SAFETY: the caller's layout contract, forwarded to the shipped allocator.
        unsafe { SHIPPED.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_call();
        // SAFETY: the caller's layout contract, forwarded to the shipped allocator.
        unsafe { SHIPPED.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from this allocator with this layout.
        unsafe { SHIPPED.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_call();
        // SAFETY: `ptr` came from this allocator with this layout; the caller upholds `new_size`.
        unsafe { SHIPPED.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counted = Counted;

/// The kernel's calls a row asks for itself: a thread's core, and the
/// deciding thread's counters.
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
mod kernel {
    use std::ffi::{c_int, c_long, c_ulong, c_void};

    extern "C" {
        fn sched_getaffinity(thread: c_int, size: usize, mask: *mut u64) -> c_int;
        fn sched_setaffinity(thread: c_int, size: usize, mask: *const u64) -> c_int;
        fn syscall(number: c_long, ...) -> c_long;
        fn read(file: c_int, buffer: *mut c_void, count: usize) -> isize;
        fn close(file: c_int) -> c_int;
    }

    /// A core mask as the kernel reads one: 1,024 cores.
    type Mask = [u64; 16];

    /// The cores the calling thread may run on, in their order.
    pub(super) fn given() -> Vec<usize> {
        let mut mask: Mask = [0; 16];
        // SAFETY: the mask is `size_of_val(&mask)` writable bytes, and thread 0 is the caller.
        let refused = unsafe { sched_getaffinity(0, size_of_val(&mask), mask.as_mut_ptr()) };
        if refused != 0 {
            return Vec::new();
        }
        let cores = mask.iter().enumerate().flat_map(|(word, bits)| {
            (0..64).filter_map(move |bit| ((bits >> bit) & 1 == 1).then_some(word * 64 + bit))
        });
        cores.collect()
    }

    /// Pins `thread` to `core` alone, the caller for thread 0; whether the
    /// kernel took it.
    pub(super) fn pin(thread: c_int, core: usize) -> bool {
        let mut mask: Mask = [0; 16];
        let Some(word) = mask.get_mut(core / 64) else {
            return false;
        };
        *word = 1 << (core % 64);
        // SAFETY: the mask is `size_of_val(&mask)` readable bytes; the kernel refuses a thread that ended.
        unsafe { sched_setaffinity(thread, size_of_val(&mask), mask.as_ptr()) == 0 }
    }

    /// The number of `perf_event_open`, which the C library wraps in no call.
    #[cfg(target_arch = "x86_64")]
    const PERF_EVENT_OPEN: c_long = 298;
    #[cfg(target_arch = "aarch64")]
    const PERF_EVENT_OPEN: c_long = 241;

    /// The first version of the kernel's `perf_event_attr`: 64 bytes.
    #[repr(C)]
    #[derive(Default)]
    struct Attr {
        kind: u32,
        size: u32,
        config: u64,
        sample: u64,
        sample_type: u64,
        read_format: u64,
        flags: u64,
        wakeup: u32,
        breakpoint: u32,
        config1: u64,
    }

    /// One hardware counter over one thread's own work: user space, no
    /// kernel and no hypervisor.
    pub(super) struct Counter(c_int);

    impl Counter {
        /// A counter of `thread`'s cycles (`event` 0) or instructions (1),
        /// counting from now; none where the kernel or the host has none
        /// to give.
        pub(super) fn open(thread: c_int, event: u64) -> Option<Self> {
            let attr = Attr {
                size: 64,
                config: event,
                // exclude_kernel and exclude_hv.
                flags: (1 << 5) | (1 << 6),
                ..Attr::default()
            };
            let (every_core, no_group): (c_int, c_int) = (-1, -1);
            let no_flags: c_ulong = 0;
            // SAFETY: the kernel reads `size` bytes of the attribute, which outlives the call.
            let file = unsafe {
                syscall(
                    PERF_EVENT_OPEN,
                    &raw const attr,
                    thread,
                    every_core,
                    no_group,
                    no_flags,
                )
            };
            c_int::try_from(file)
                .ok()
                .filter(|file| *file >= 0)
                .map(Self)
        }

        /// The count so far.
        pub(super) fn count(&self) -> Option<u64> {
            let mut count = 0_u64;
            // SAFETY: the buffer is the eight bytes of `count`.
            let read = unsafe { read(self.0, (&raw mut count).cast(), size_of::<u64>()) };
            (read == 8).then_some(count)
        }
    }

    impl Drop for Counter {
        fn drop(&mut self) {
            // SAFETY: the file is this counter's own, closed once.
            unsafe { close(self.0) };
        }
    }
}

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

const GATEWAY: &str = "bench-gateway";
const PRESET: &str = "least_load";
/// Whether a served decision reads the load reports a row's members stand
/// in their bands by. A controller's single decision is given its load
/// monitor's snapshot, and the bench writes its reports into the store's
/// table, so no wire row is run: a row's pool moves with the leases on
/// file, and a row's guard stops a pool that moves. It turns when the
/// bench's members answer the load monitor's polls.
const READS_LOADS: bool = false;
/// The pair plan of the pair count cell: the plan of the engine's own
/// pair cells (`pair_classes`), least load on each side.
const PAIR_PLAN: &str = r#"{
  "apiVersion": "smg.policy/v1",
  "profiles": {
    "side": { "filter": ["healthy", "not-overloaded", "not-tried"], "score": ["least-load"], "pick": "best" }
  },
  "bindings": [
    { "match": { "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "side", "decode": "side" } }
  ]
}"#;
/// The names a row finds its threads by: the controller's runtime threads,
/// the thread that serves it, and the gateway's session.
const RUNTIME: &str = "runtime";
const SERVING: &str = "controller";
const SESSION: &str = "session";
/// The cores the process was given, read once as a measured run starts.
/// Unset in a test, which pins no thread.
static GIVEN: OnceLock<Vec<usize>> = OnceLock::new();
/// The slots a stream keeps its requests' send instants in, each beside
/// its request's id: above every row's count in flight. An answer whose
/// slot a later request took is counted, not timed.
const RING: u64 = 1_024;
/// The budget every request brings.
const BUDGET_MICROS: u32 = 5_000_000;
/// A stream's window of credit, as the `Welcome` states it.
const WINDOW: u64 = 128;
/// An offered request handed over this long after it was due is counted
/// late: the client's own lag.
const LATE: Duration = Duration::from_micros(100);
/// The requests every member reports beneath its band: more than a wire row
/// has decisions ([`wire`] checks each row's count), so no count of leases
/// on file reaches it and a lease on file moves no row's cost.
const HELD: u32 = 1 << 21;
/// The requests between two bands of a row that tells its dispatches,
/// where a row that tells none has one: twice the floor beneath the bands,
/// so more than twice a wire row's decisions. A dispatch told is credited
/// to its member at one request's work (the load stage's mean prefill, a
/// request naming no size) until that member's next report, and every
/// dispatch of a row falls on its least member: the credits stay beneath
/// the next band, and a dispatch told moves no row's place.
const APART: u32 = 2 * HELD;
/// The reports a stream hands the session together: a hand-over wakes the
/// session's thread, and one a request would time the wake.
const REPORTS: usize = 16;
/// The seed of the order the members are dealt to their bands in.
const DEAL: u64 = 0x0B_A4D5;

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

/// A thread's CPU and switches so far, by the kernel's books: its time on
/// a core in nanoseconds, its user and system ticks, and the switches it
/// asked for.
#[derive(Clone, Copy, Default)]
struct Spent {
    nanos: u64,
    user: u64,
    system: u64,
    asked: u64,
}

impl Spent {
    /// Thread `task` of this process, the caller for `thread-self`; none
    /// where the kernel's files do not read.
    fn of(task: &str) -> Option<Self> {
        let read = |file: &str| fs::read_to_string(format!("/proc/{task}/{file}")).ok();
        let nanos = read("schedstat")?.split_whitespace().next()?.parse().ok()?;
        let stat = read("stat")?;
        // Past the command's closing parenthesis, as the process's own file
        // reads: the user time the twelfth field, the system time the next.
        let (_, fields) = stat.rsplit_once(')')?;
        let mut times = fields.split_whitespace().skip(11).map(str::parse::<u64>);
        let (Some(Ok(user)), Some(Ok(system))) = (times.next(), times.next()) else {
            return None;
        };
        let asked = read("status")?
            .lines()
            .find_map(|line| line.strip_prefix("voluntary_ctxt_switches:"))
            .and_then(|count| count.trim().parse().ok())?;
        Some(Self {
            nanos,
            user,
            system,
            asked,
        })
    }

    /// This sum with `other` in it.
    fn with(self, other: Self) -> Self {
        Self {
            nanos: self.nanos + other.nanos,
            user: self.user + other.user,
            system: self.system + other.system,
            asked: self.asked + other.asked,
        }
    }

    /// What was spent since `earlier`.
    fn since(self, earlier: Self) -> Self {
        Self {
            nanos: self.nanos.saturating_sub(earlier.nanos),
            user: self.user.saturating_sub(earlier.user),
            system: self.system.saturating_sub(earlier.system),
            asked: self.asked.saturating_sub(earlier.asked),
        }
    }

    /// The user and the system microseconds a decision, as a row prints
    /// them: the time on a core, split as the kernel's ticks fell. `n/a`
    /// for a time under one tick.
    fn apart(self, decisions: u64) -> String {
        let micros = self.nanos / 1_000;
        match (micros * self.user).checked_div(self.user + self.system) {
            Some(user) => format!(
                "{}+{}",
                each(user, decisions),
                each(micros.saturating_sub(user), decisions)
            ),
            None => "n/a".to_owned(),
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

/// The process's live threads: each one's id and name, in the order of
/// their ids. Empty where the kernel's directory does not read.
fn threads() -> Vec<(i32, String)> {
    let Ok(tasks) = fs::read_dir("/proc/self/task") else {
        return Vec::new();
    };
    let mut threads: Vec<(i32, String)> = tasks
        .filter_map(|task| {
            let id = task.ok()?.file_name().to_str()?.parse().ok()?;
            let name = fs::read_to_string(format!("/proc/self/task/{id}/comm")).ok()?;
            Some((id, name.trim().to_owned()))
        })
        .collect();
    threads.sort_unstable();
    threads
}

/// What the controller's deciders and its runtime threads have spent so
/// far, over the process's live threads by their names; none where the
/// kernel's files do not read.
fn controller_spent() -> Option<(Spent, Spent)> {
    let (mut deciders, mut runtime) = (Spent::default(), Spent::default());
    let threads = threads();
    if threads.is_empty() {
        return None;
    }
    for (id, name) in threads {
        let class = if name.starts_with("decide-") {
            &mut deciders
        } else if name == RUNTIME {
            &mut runtime
        } else {
            continue;
        };
        *class = class.with(Spent::of(&format!("self/task/{id}"))?);
    }
    Some((deciders, runtime))
}

/// The cores a row's busy threads sit on, each alone.
struct Cores {
    deciders: Vec<usize>,
    runtime: Vec<usize>,
    clients: Vec<usize>,
    session: usize,
    serving: usize,
    /// The allocator's background thread and the election's share one.
    rest: usize,
}

impl Cores {
    /// The row's cores, taken from those the process was given in their
    /// order: the deciders', the runtime threads', the streams' clients',
    /// the session's, the serving thread's and the rest's. None where the
    /// process was given fewer cores than the row has busy threads, or
    /// reads none ([`GIVEN`]).
    fn of(shape: Shape) -> Option<Self> {
        let mut cores = GIVEN.get()?.iter().copied();
        let mut take = |count: usize| {
            let taken: Vec<usize> = cores.by_ref().take(count).collect();
            (taken.len() == count).then_some(taken)
        };
        let deciders = take(shape.deciders)?;
        let runtime = take(shape.threads)?;
        let clients = take(usize::try_from(shape.streams).ok()?)?;
        let [session, serving, rest] = take(3)?.try_into().ok()?;
        Some(Self {
            deciders,
            runtime,
            clients,
            session,
            serving,
            rest,
        })
    }

    /// Pins the controller's threads, found by their names: a decider on
    /// the core of its number, the runtime threads in the order of their
    /// ids, the serving thread, and the allocator's and the election's
    /// together. The streams' clients and the session pin themselves as
    /// they start. A row that found fewer deciders or runtime threads than
    /// it has cores for them stops the bench.
    fn controller(&self) {
        let mut runtime = self.runtime.iter();
        let (mut deciders, mut workers) = (0, 0);
        for (id, name) in threads() {
            let core = if let Some(number) = name.strip_prefix("decide-") {
                deciders += 1;
                number
                    .parse::<usize>()
                    .ok()
                    .and_then(|number| self.deciders.get(number))
            } else if name == RUNTIME {
                runtime.next().inspect(|_| workers += 1)
            } else if name == SERVING {
                Some(&self.serving)
            } else if name == "election" || name.starts_with("jemalloc") {
                Some(&self.rest)
            } else {
                None
            };
            if let Some(core) = core {
                assert!(kernel::pin(id, *core), "{name} sits on core {core}");
            }
        }
        assert_eq!(
            (deciders, workers),
            (self.deciders.len(), self.runtime.len()),
            "every decider and every runtime thread of the row is pinned"
        );
    }

    /// The cores as a row prints them: the six classes in their order, a
    /// slash between them.
    fn print(&self) -> String {
        [
            runs(&self.deciders),
            runs(&self.runtime),
            runs(&self.clients),
            self.session.to_string(),
            self.serving.to_string(),
            self.rest.to_string(),
        ]
        .join("/")
    }
}

/// A list of cores as a row prints it: a run of neighbours as its first and
/// its last, a comma between runs.
fn runs(cores: &[usize]) -> String {
    let mut runs = Vec::new();
    let mut cores = cores.iter().copied().peekable();
    while let Some(first) = cores.next() {
        let mut last = first;
        while let Some(next) = cores.next_if_eq(&(last + 1)) {
            last = next;
        }
        runs.push(if last == first {
            first.to_string()
        } else {
            format!("{first}-{last}")
        });
    }
    runs.join(",")
}

/// The cycles and the instructions of one thread of a row, in user space.
struct Counters {
    cycles: kernel::Counter,
    instructions: kernel::Counter,
}

impl Counters {
    /// Counters over the row's first deciding thread, from now; none where
    /// the kernel gives none: a host without counters, or one that keeps
    /// them from its users.
    fn open() -> Option<Self> {
        let (decider, _) = threads().into_iter().find(|(_, name)| name == "decide-0")?;
        Self::over(decider)
    }

    /// Counters over `thread`, the calling thread as 0, from now; none
    /// where the kernel gives none.
    fn over(thread: i32) -> Option<Self> {
        Some(Self {
            cycles: kernel::Counter::open(thread, 0)?,
            instructions: kernel::Counter::open(thread, 1)?,
        })
    }

    /// The cycles and the instructions counted so far.
    fn read(&self) -> Option<(u64, u64)> {
        self.cycles.count().zip(self.instructions.count())
    }
}

/// A controller served on its own thread and runtime.
struct Served {
    addr: SocketAddr,
    store: Arc<MemoryStore>,
    stop: oneshot::Sender<()>,
    thread: thread::JoinHandle<()>,
}

impl Served {
    fn stop(self) {
        // The serve hears the stop unless it already ended, which its
        // thread's end tells.
        drop(self.stop);
        self.thread
            .join()
            .expect("the controller's thread ends cleanly");
    }
}

fn loopback() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, 0))
}

fn listener(name: &str, scopes: &[Scope]) -> Listener {
    Listener {
        name: name.to_owned(),
        addr: loopback(),
        scopes: scopes.to_vec(),
        tls: None,
    }
}

/// The binary's default shape on ephemeral loopback ports.
fn config(deciders: usize) -> Config {
    Config {
        controller_id: "bench-controller".to_owned(),
        listeners: vec![
            listener("gateway", &[Scope::Gateway, Scope::Ingestion]),
            listener("control", &[Scope::Control]),
            listener("replication", &[Scope::Replication]),
        ],
        admin: loopback(),
        shutdown_settle: Duration::from_millis(10),
        shutdown_grace: Duration::from_secs(5),
        load_poll_interval: Config::DEFAULT_LOAD_POLL_INTERVAL,
        decide_shards: deciders,
        election: ElectionConfig::default(),
        discovery: None,
    }
}

fn model(index: u32) -> String {
    format!("model-{index}")
}

/// The band of each of `members` rows, a band a request above the one
/// before it. The rows are dealt in an order shuffled by a fixed seed to
/// bands of one row and of two by turns; over 512 members to bands of
/// eight and of sixteen, a sixty-fourth of the rows a unit.
fn bands(members: u32) -> Vec<u32> {
    let rows = usize::try_from(members).expect("a count that fits");
    let mut order: Vec<usize> = (0..rows).collect();
    let mut state = DEAL;
    for last in (1..rows).rev() {
        // One value of SplitMix64 a swap: the same order on every build.
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut drawn = state;
        drawn = (drawn ^ (drawn >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        drawn = (drawn ^ (drawn >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        drawn ^= drawn >> 31;
        let below = u64::try_from(last).expect("an index that fits") + 1;
        order.swap(last, usize::try_from(drawn % below).expect("an index"));
    }
    let unit = (rows / 64).max(1);
    let mut band_of = vec![0; rows];
    let (mut band, mut left) = (0, unit);
    for row in order {
        if left == 0 {
            band += 1;
            left = if band % 2 == 0 { unit } else { 2 * unit };
        }
        if let Some(its) = band_of.get_mut(row) {
            *its = band;
        }
        left -= 1;
    }
    band_of
}

/// The shape's fleet: ready members, each with a load report in its band
/// ([`bands`]), under the row's preset as generation 1, their rows built.
/// A row that tells its dispatches stands its bands apart ([`APART`]), and
/// each member's ceiling at twice the highest band's report.
/// A shared fleet is one deal of the bands over members that serve every
/// model; fleets of their own are the same deal a model, over members
/// that serve it alone. A pair plan's row holds that many members a side,
/// under the pair cell's plan.
fn fill(store: &MemoryStore, shape: Shape) {
    let models: Vec<String> = (0..shape.models).map(model).collect();
    let fleets: Vec<(String, Vec<String>)> = match shape.fleet {
        Fleet::Shared => vec![(String::new(), models)],
        Fleet::Own => models
            .into_iter()
            .enumerate()
            .map(|(fleet, model)| (format!("{fleet}-"), vec![model]))
            .collect(),
    };
    let sides: &[(Role, &str)] = if shape.pair {
        &[(Role::Prefill, "prefill"), (Role::Decode, "decode")]
    } else {
        &[(Role::Single, "worker")]
    };
    let bands = bands(shape.members);
    let (step, ceiling) = if shape.frames {
        let highest = bands.iter().copied().max().unwrap_or(0);
        (APART, 2 * (HELD + APART * highest))
    } else {
        (1, 2 * HELD)
    };
    let now = Instant::now();
    for (fleet, models) in &fleets {
        let rows = sides.iter().flat_map(|&(role, name)| {
            let banded = (0..shape.members).zip(&bands);
            banded.map(move |(index, band)| (role, name, index, band))
        });
        for (role, name, index, band) in rows {
            let key = format!("http://{name}-{fleet}{index}.fleet:8000");
            let id = store
                .upsert(WorkerFacts {
                    key: key.clone(),
                    engine: key,
                    incarnation: 1,
                    role,
                    models: models.clone(),
                    dp_size: 1,
                    max_running: Some(ceiling),
                    ..WorkerFacts::default()
                })
                .id();
            let revision = WorkersStore::get(store, id).expect("the row").revision;
            store.set_status(id, WorkerStatus::Ready, revision);
            store.record(
                id,
                0,
                LoadRow {
                    running: HELD + step * band,
                    max_running: ceiling,
                    ..LoadRow::default()
                },
                now,
                LoadSource::Stream,
            );
        }
    }
    let mut document = if shape.pair {
        serde_json::from_str::<PolicySpec>(PAIR_PLAN).expect("a well-formed document")
    } else {
        PolicySpec::preset(PRESET).expect("a preset of this build")
    };
    document.generation = 1;
    if let Ok(sample) = env::var("DECIDE_STREAM_EXPLAIN_SAMPLE") {
        document.explain.sample = sample.parse().expect("an unsigned sampling rate");
    }
    if let (false, Ok(shadow)) = (shape.pair, env::var("DECIDE_STREAM_SHADOW")) {
        let twin = document.profiles[PRESET].clone();
        document.profiles.insert("twin".to_owned(), twin);
        let named = Some(ShadowSpec::Profile("twin".to_owned()));
        if shadow == "other" {
            let mut other = document.bindings[0].clone();
            "another-model".clone_into(&mut other.matcher.model);
            other.shadow = named;
            document.bindings.insert(0, other);
        } else {
            document.bindings[0].shadow = named;
        }
    }
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    store.apply(spec).expect("above the unbound start");
    store.publish_now();
    for &(role, _) in sides {
        filled::filled(store, &model(0), role, shape.members as usize);
    }
}

/// Serves a controller over the shape's fleet; returns once it leads.
fn serve(shape: Shape) -> Served {
    let (ready, bound) = std_mpsc::channel();
    let thread = thread::Builder::new()
        .name("controller".to_owned())
        .spawn(move || {
            let runtime = runtime::Builder::new_multi_thread()
                .worker_threads(shape.threads)
                .thread_name(RUNTIME)
                .enable_all()
                .build()
                .expect("the controller's runtime");
            runtime.block_on(async move {
                let mut controller = Controller::bind(config(shape.deciders))
                    .await
                    .expect("the listeners bind");
                controller
                    .identity_as_told("gateway")
                    .expect("a listener of the configuration");
                let store = controller.store();
                fill(&store, shape);
                let addr = controller
                    .addresses()
                    .of(Scope::Gateway)
                    .expect("the gateway scope is mounted");
                let mut readiness = controller.readiness();
                let (stop, stopped) = oneshot::channel::<()>();
                let serving = controller.serve(async {
                    // The bench's drop of its end is the stop.
                    let _ = stopped.await;
                });
                let told = async {
                    assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
                    ready
                        .send((addr, store, stop))
                        .expect("the bench waits for the address");
                };
                let (served, ()) = tokio::join!(serving, told);
                served.expect("a clean stop");
            });
        })
        .expect("the controller's thread");
    let (addr, store, stop) = bound.recv().expect("the controller serves");
    Served {
        addr,
        store,
        stop,
        thread,
    }
}

/// A call that states the bench's gateway on the listener that takes
/// identities as told.
fn stated<T>(message: T) -> Request<T> {
    let mut request = Request::new(message);
    request
        .metadata_mut()
        .insert("x-smg-principal", AsciiMetadataValue::from_static(GATEWAY));
    request
}

async fn connect(addr: SocketAddr) -> PolicyControllerClient<Channel> {
    PolicyControllerClient::connect(format!("http://{addr}"))
        .await
        .expect("the listener accepts")
}

/// The gateway's session: its `Hello` is welcomed, and the stream stands
/// for as long as the returned ends are held.
async fn hello(addr: SocketAddr) -> (mpsc::Sender<GatewayFrame>, Streaming<ControllerFrame>) {
    let mut client = connect(addr).await;
    // Room for the frames of one hand-over of reports: they leave together.
    let (frames, sent) = mpsc::channel(2 * REPORTS);
    let mut answers = client
        .session(stated(ReceiverStream::new(sent)))
        .await
        .expect("the stream opens")
        .into_inner();
    let hello = Hello {
        gateway_id: GATEWAY.to_owned(),
        protocol_min: 1,
        protocol_max: 1,
        ..Hello::default()
    };
    frames
        .send(GatewayFrame {
            seq: 0,
            term_floor: 0,
            frame: Some(Frame::Hello(hello)),
        })
        .await
        .expect("the stream takes the Hello");
    let answer = answers
        .message()
        .await
        .expect("a frame, not a status")
        .expect("a frame, not the stream's end");
    let window = match &answer.frame {
        Some(controller_frame::Frame::Welcome(welcome)) => u64::from(welcome.decide_window),
        _ => 0,
    };
    assert_eq!(
        window, WINDOW,
        "a Welcome answers a Hello, with the window the open loop counts by: {answer:?}"
    );
    (frames, answers)
}

/// What one stream measured.
struct Tally {
    /// Each measured decision's round trip, in nanoseconds; none of a
    /// counted run's.
    nanos: Vec<u64>,
    /// The measured answers, whether their round trips were kept.
    answered: u64,
    busy: u64,
    /// The first measured answer's receipt, and the last one's.
    opened: Instant,
    closed: Instant,
    /// The allocator's calls over the measured decisions, where it counted.
    calls: u64,
    /// The measured requests an open loop handed over late.
    late: u64,
    /// The measured requests an open loop decided itself: due with no
    /// credit.
    local: u64,
    /// The measured answers a closed loop could not time: a later request
    /// took their slot of the ring.
    overtaken: u64,
    /// The measured decisions by how their three best rows scored: each
    /// alone, the best alone and the next two alike, the best two alike,
    /// all three alike.
    ties: [u64; 4],
    /// The measured picks whose trace held fewer than three rows.
    short: u64,
    /// The measured picks whose answer carried a lease.
    leases: u64,
}

impl Tally {
    /// Counts a measured pick: under its place of [`Self::ties`], and its
    /// lease where its answer carried one.
    fn count(&mut self, pick: Option<&Answered>) {
        let Some(pick) = pick else {
            return;
        };
        if let Some(count) = self.ties.get_mut(pick.place) {
            *count += 1;
        }
        self.short += u64::from(pick.short);
        self.leases += u64::from(pick.lease.is_some());
    }
}

/// The bench's gateway as the store keys its leases: by the fingerprint of
/// the subject its calls state.
fn gateway() -> GatewayId {
    GatewayId(smg_policy_controller::engine::fingerprint(
        GATEWAY.as_bytes(),
    ))
}

/// What a stream reads of a pick's answer.
struct Answered {
    /// Its place in the stream's tally of ties ([`Tally::ties`]).
    place: usize,
    /// Its trace held fewer than the three rows a place reads.
    short: bool,
    /// The lease the answer carried.
    lease: Option<u64>,
    /// The pick's first target: the member a gateway dispatches to.
    served: u64,
}

/// A request that is over, as its stream tells the gateway's session: the
/// lease its answer carried, zero where it carried none (a lease the store
/// does not hold is no error to a session), and the member that served it.
#[derive(Clone, Copy)]
struct Over {
    lease: u64,
    served: u64,
}

/// How a stream reports an answered request as over.
enum Ends {
    /// Through the stream's own hand at the leases' file, in process: the
    /// hand is made at the stream's first report.
    Door(Arc<MemoryStore>, Option<Hand>),
    /// As a gateway does: a `Dispatched` and a `Completed` on its session,
    /// whose thread takes the stream's reports [`REPORTS`] at a time.
    Session {
        reports: mpsc::Sender<Vec<Over>>,
        held: Vec<Over>,
    },
}

impl Clone for Ends {
    /// A stream's own end: a door's hand is not shared.
    fn clone(&self) -> Self {
        match self {
            Self::Door(store, _) => Self::Door(Arc::clone(store), None),
            Self::Session { reports, held } => Self::Session {
                reports: reports.clone(),
                held: held.clone(),
            },
        }
    }
}

impl Ends {
    /// Reports a picked request as over: its lease, where its answer
    /// carried one, ends at the store's door, or the gateway's session is
    /// told. A row holds as many leases as it has decisions in flight and
    /// frames on their way.
    async fn over(&mut self, gateway: GatewayId, pick: &Answered) {
        match self {
            Self::Door(store, hand) => {
                if let Some(lease) = pick.lease {
                    let hand = hand.get_or_insert_with(|| Hand::new(gateway));
                    let ended = Said::Released(None, ReleaseReason::ClientGone);
                    store.hand(hand, LeaseId(lease), ended);
                }
            }
            Self::Session { reports, held } => {
                held.push(Over {
                    lease: pick.lease.unwrap_or(0),
                    served: pick.served,
                });
                if held.len() >= REPORTS {
                    let together = mem::replace(held, Vec::with_capacity(REPORTS));
                    reports
                        .send(together)
                        .await
                        .expect("the session takes a stream's reports");
                }
            }
        }
    }

    /// Hands the session the reports the stream still holds, as it ends.
    async fn rest(&mut self) {
        if let Self::Session { reports, held } = self {
            if !held.is_empty() {
                reports
                    .send(mem::take(held))
                    .await
                    .expect("the session takes a stream's last reports");
            }
        }
    }
}

/// Every decision of a row is a pick over its fleet. Another outcome means
/// the setup drifted, and the row would print another path's numbers.
/// Reads the decision's place in a stream's tally of ties, by its trace's
/// three best rows (two score alike where their scores are one value, bit
/// for bit), the lease its answer carried and its first target.
fn answered(decision: &Decision) -> Answered {
    let first = match &decision.outcome {
        Some(Outcome::Pick(pick)) => pick.targets.first(),
        _ => None,
    };
    assert!(first.is_some(), "a pick over the fleet: {decision:?}");
    let top = decision.trace.as_ref().map_or(&[][..], |trace| &trace.top);
    let place = match top {
        [first, second, third, ..] => {
            let alike = |one: f32, other: f32| usize::from(one.to_bits() == other.to_bits());
            2 * alike(first.score, second.score) + alike(second.score, third.score)
        }
        _ => 0,
    };
    Answered {
        place,
        short: top.len() < 3,
        lease: decision.leases.first().map(|lease| lease.id),
        served: first.map_or(0, |target| target.id),
    }
}

/// Opens the allocator's count.
fn counting() {
    CALLS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
}

fn slot(id: u64) -> usize {
    usize::try_from(id % RING).expect("a slot of the ring")
}

/// Request `id` of a row, for one of its models.
fn request(id: u64, models: &[String], pair: bool) -> DecideRequest {
    let name = models
        .get(slot(id) % models.len().max(1))
        .expect("a model of the row");
    let phase = if pair {
        WirePhase::Prefill
    } else {
        WirePhase::default()
    };
    DecideRequest {
        request_id: id.to_be_bytes().to_vec(),
        facts: Some(WireFacts {
            model: name.clone(),
            seed: id,
            budget_micros: BUDGET_MICROS,
            phase: phase.into(),
            ..WireFacts::default()
        }),
        constraints: None,
        again: false,
        ..Default::default()
    }
}

/// Hands the stream its next request and keeps the instant.
async fn hand(
    requests: &mpsc::Sender<DecideRequest>,
    sent_at: &mut [(u64, Instant)],
    next: &mut u64,
    models: &[String],
    pair: bool,
) {
    let id = *next;
    *next += 1;
    let frame = request(id, models, pair);
    if let Some(at) = sent_at.get_mut(slot(id)) {
        *at = (id, Instant::now());
    }
    requests
        .send(frame)
        .await
        .expect("the stream takes the request");
}

/// One stream of the shape: `in_flight` requests kept in flight until the
/// warm and the measured decisions are answered.
async fn ask(addr: SocketAddr, shape: Shape, ends: &mut Ends) -> Tally {
    let gateway = gateway();
    let models: Vec<String> = (0..shape.models).map(model).collect();
    let mut client = connect(addr).await;
    let (requests, sent) = mpsc::channel(256);
    let mut answers = client
        .decide(stated(ReceiverStream::new(sent)))
        .await
        .expect("a present gateway's stream opens")
        .into_inner();
    let mut sent_at = vec![(u64::MAX, Instant::now()); slot(RING - 1) + 1];
    let mut next = 0;
    let mut tally = Tally {
        nanos: Vec::with_capacity(usize::try_from(shape.measured).expect("a count that fits")),
        answered: 0,
        busy: 0,
        opened: Instant::now(),
        closed: Instant::now(),
        calls: 0,
        late: 0,
        local: 0,
        overtaken: 0,
        ties: [0; 4],
        short: 0,
        leases: 0,
    };
    if shape.counted && shape.warm == 0 {
        counting();
    }
    for _ in 0..shape.in_flight {
        hand(&requests, &mut sent_at, &mut next, &models, shape.pair).await;
    }
    let mut done = 0;
    loop {
        let answer = answers
            .message()
            .await
            .expect("an answer, not a status")
            .expect("an answer, not the stream's end");
        let received = Instant::now();
        let pick = match &answer.answer {
            Some(Answer::Decision(decision)) => Some(answered(decision)),
            Some(Answer::Busy(_)) => {
                tally.busy += 1;
                None
            }
            Some(Answer::Credit(_)) | None => continue,
        };
        if let Some(pick) = &pick {
            ends.over(gateway, pick).await;
        }
        if done == shape.warm {
            tally.opened = received;
        }
        if done >= shape.warm {
            tally.answered += 1;
            tally.count(pick.as_ref());
        }
        // A counted run keeps no round trip: a sort of them would stay in
        // the difference of two runs.
        if done >= shape.warm && !shape.counted {
            let id: [u8; 8] = answer
                .request_id
                .as_slice()
                .try_into()
                .expect("the request's eight bytes");
            let id = u64::from_be_bytes(id);
            match sent_at.get(slot(id)) {
                Some((held, at)) if *held == id => {
                    let took = received.saturating_duration_since(*at);
                    tally.nanos.push(
                        u64::try_from(took.as_nanos()).expect("a round trip under centuries"),
                    );
                }
                // A later request took the slot while this one waited:
                // its instant is not this request's.
                _ => tally.overtaken += 1,
            }
        }
        done += 1;
        // No request follows the last measured answer, and a counted run
        // opens its window before the first measured request is handed
        // over: the window holds whole exchanges.
        if done == shape.warm + shape.measured {
            break;
        }
        if shape.counted && done == shape.warm {
            counting();
        }
        hand(&requests, &mut sent_at, &mut next, &models, shape.pair).await;
    }
    if shape.counted {
        COUNTING.store(false, Ordering::Relaxed);
        tally.calls = CALLS.load(Ordering::Relaxed);
    }
    tally.closed = Instant::now();
    ends.rest().await;
    tally
}

/// One stream of the shape under an open loop, as the design's gateway
/// runs one: `shape.offered` requests a second, whatever the answers do.
/// Request `i` is due at the start plus `i` over the rate. Due while the
/// stream has credit, it is handed over, and its round trip is read from
/// the due instant; due with no credit, it is the gateway's own decision
/// at once, counted and never sent. The client spins between instants,
/// since a timer's millisecond is coarser than the gaps, and yields to its
/// runtime each turn so the connection is read and written.
async fn offer(addr: SocketAddr, shape: Shape, ends: &mut Ends) -> Tally {
    let gateway = gateway();
    let models: Vec<String> = (0..shape.models).map(model).collect();
    let mut client = connect(addr).await;
    let (requests, sent) = mpsc::channel(256);
    let mut answers = client
        .decide(stated(ReceiverStream::new(sent)))
        .await
        .expect("a present gateway's stream opens")
        .into_inner();
    let total = shape.warm + shape.measured;
    let gap = Duration::from_nanos(1_000_000_000 / shape.offered.max(1));
    let mut tally = Tally {
        nanos: Vec::with_capacity(usize::try_from(shape.measured).expect("a count that fits")),
        answered: 0,
        busy: 0,
        opened: Instant::now(),
        closed: Instant::now(),
        calls: 0,
        late: 0,
        local: 0,
        overtaken: 0,
        ties: [0; 4],
        short: 0,
        leases: 0,
    };
    let start = Instant::now();
    let due = |id: u64| start + gap.saturating_mul(u32::try_from(id).unwrap_or(u32::MAX));
    let (mut next, mut done, mut credit) = (0, 0, WINDOW);
    while done < total {
        let now = Instant::now();
        while next < total && due(next) <= now {
            if credit == 0 {
                if next >= shape.warm {
                    tally.local += 1;
                }
                done += 1;
            } else {
                if next >= shape.warm && now.saturating_duration_since(due(next)) > LATE {
                    tally.late += 1;
                }
                requests
                    .try_send(request(next, &models, shape.pair))
                    .expect("a window of requests fits the stream's channel");
                credit -= 1;
            }
            next += 1;
        }
        if done >= total {
            break;
        }
        let answer = tokio::select! {
            biased;
            answer = answers.message() => answer,
            () = tokio::task::yield_now() => continue,
        };
        let answer = answer
            .expect("an answer, not a status")
            .expect("an answer, not the stream's end");
        let received = Instant::now();
        let pick = match &answer.answer {
            Some(Answer::Decision(decision)) => Some(answered(decision)),
            Some(Answer::Busy(_)) => {
                tally.busy += 1;
                None
            }
            Some(Answer::Credit(grant)) => {
                credit += u64::from(grant.window);
                continue;
            }
            None => continue,
        };
        if let Some(pick) = &pick {
            ends.over(gateway, pick).await;
        }
        let id: [u8; 8] = answer
            .request_id
            .as_slice()
            .try_into()
            .expect("the request's eight bytes");
        let id = u64::from_be_bytes(id);
        if id >= shape.warm {
            if tally.answered == 0 {
                tally.opened = received;
            }
            tally.answered += 1;
            tally.count(pick.as_ref());
            let took = received.saturating_duration_since(due(id));
            tally
                .nanos
                .push(u64::try_from(took.as_nanos()).expect("a round trip under centuries"));
        }
        done += 1;
    }
    tally.closed = Instant::now();
    ends.rest().await;
    tally
}

/// What the session's thread returns: the frames it sent and what it
/// spent.
type Told = (u64, Option<Spent>);

/// The gateway's session on a thread of its own: its `Hello` is welcomed,
/// then each request a stream reports as over is told as a gateway tells
/// one served to its end, a `Dispatched` and a `Completed`. Once every
/// reporting end is dropped the thread ends its side of the stream and
/// reads the controller's to its end: every frame was applied by then.
/// Returns the frames it sent and what its thread spent.
fn session(
    addr: SocketAddr,
    core: Option<usize>,
) -> (mpsc::Sender<Vec<Over>>, thread::JoinHandle<Told>) {
    let (ready, opened) = std_mpsc::channel();
    let thread = thread::Builder::new()
        .name(SESSION.to_owned())
        .spawn(move || {
            if let Some(core) = core {
                assert!(kernel::pin(0, core), "the session sits on core {core}");
            }
            let sent = current_thread().block_on(async move {
                let (frames, mut answers) = hello(addr).await;
                let (reports, mut over) = mpsc::channel::<Vec<Over>>(64);
                ready.send(reports).expect("the row waits for the session");
                let mut seq = 0;
                while let Some(together) = over.recv().await {
                    for Over { lease, served } in together {
                        let dispatched = Dispatched {
                            lease_id: lease,
                            target_index: 0,
                            side: Side::Single.into(),
                            since_decision_micros: 0,
                        };
                        let completed = Completed {
                            lease_id: lease,
                            side: Side::Single.into(),
                            served,
                            status: 200,
                            class: Class::Ok.into(),
                            ..Completed::default()
                        };
                        for frame in [Frame::Dispatched(dispatched), Frame::Completed(completed)] {
                            seq += 1;
                            let taken = frames
                                .send(GatewayFrame {
                                    seq,
                                    term_floor: 0,
                                    frame: Some(frame),
                                })
                                .await;
                            // A session the controller ended takes no
                            // frame, and its status says why.
                            let ended = if taken.is_err() {
                                answers.message().await.err()
                            } else {
                                None
                            };
                            assert!(taken.is_ok(), "the controller ended the session: {ended:?}");
                        }
                    }
                }
                drop(frames);
                // The session ends as its frames do; a status in place
                // of that end says the last frames were not applied.
                while answers
                    .message()
                    .await
                    .expect("the controller ended the session")
                    .is_some()
                {}
                seq
            });
            (sent, Spent::of("thread-self"))
        })
        .expect("the session's thread");
    (opened.recv().expect("the session opens"), thread)
}

fn current_thread() -> runtime::Runtime {
    runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a client's runtime")
}

/// The process's user and system CPU time so far, in the kernel's ticks of
/// ten milliseconds; none where the kernel's file does not read.
fn cpu_ticks() -> Option<(u64, u64)> {
    let stat = fs::read_to_string("/proc/self/stat").ok()?;
    // Past the command's closing parenthesis the state is the first field,
    // the user time the twelfth and the system time the thirteenth.
    let (_, fields) = stat.rsplit_once(')')?;
    let mut times = fields.split_whitespace().skip(11).map(str::parse::<u64>);
    let (Some(Ok(user)), Some(Ok(system))) = (times.next(), times.next()) else {
        return None;
    };
    Some((user, system))
}

/// The process's resident size in MiB; none where the kernel's file does
/// not read.
fn rss_mib() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|kib| kib.parse::<u64>().ok())
        .map(|kib| kib / 1_024)
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
