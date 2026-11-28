//! A row's threads by the kernel's books: what each spent, the cores they
//! sit on, and the deciding thread's counters, with the process's own CPU
//! time and size. A child of the bench's root: it reads the root's private
//! `Shape` and `each`.

use std::{fs, sync::OnceLock};

use super::{each, kernel, Shape};

/// The names a row finds its threads by: the controller's runtime threads,
/// the thread that serves it, and the gateway's session.
pub(super) const RUNTIME: &str = "runtime";
const SERVING: &str = "controller";
pub(super) const SESSION: &str = "session";
/// The cores the process was given, read once as a measured run starts.
/// Unset in a test, which pins no thread.
pub(super) static GIVEN: OnceLock<Vec<usize>> = OnceLock::new();

/// A thread's CPU and switches so far, by the kernel's books: its time on
/// a core in nanoseconds, its user and system ticks, and the switches it
/// asked for.
#[derive(Clone, Copy, Default)]
pub(super) struct Spent {
    nanos: u64,
    user: u64,
    system: u64,
    pub(super) asked: u64,
}

impl Spent {
    /// Thread `task` of this process, the caller for `thread-self`; none
    /// where the kernel's files do not read.
    pub(super) fn of(task: &str) -> Option<Self> {
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
    pub(super) fn with(self, other: Self) -> Self {
        Self {
            nanos: self.nanos + other.nanos,
            user: self.user + other.user,
            system: self.system + other.system,
            asked: self.asked + other.asked,
        }
    }

    /// What was spent since `earlier`.
    pub(super) fn since(self, earlier: Self) -> Self {
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
    pub(super) fn apart(self, decisions: u64) -> String {
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
pub(super) fn controller_spent() -> Option<(Spent, Spent)> {
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
pub(super) struct Cores {
    deciders: Vec<usize>,
    runtime: Vec<usize>,
    pub(super) clients: Vec<usize>,
    pub(super) session: usize,
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
    pub(super) fn of(shape: Shape) -> Option<Self> {
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
    pub(super) fn controller(&self) {
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
    pub(super) fn print(&self) -> String {
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
pub(super) struct Counters {
    cycles: kernel::Counter,
    instructions: kernel::Counter,
}

impl Counters {
    /// Counters over the row's first deciding thread, from now; none where
    /// the kernel gives none: a host without counters, or one that keeps
    /// them from its users.
    pub(super) fn open() -> Option<Self> {
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
    pub(super) fn read(&self) -> Option<(u64, u64)> {
        self.cycles.count().zip(self.instructions.count())
    }
}

/// The process's user and system CPU time so far, in the kernel's ticks of
/// ten milliseconds; none where the kernel's file does not read.
pub(super) fn cpu_ticks() -> Option<(u64, u64)> {
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
pub(super) fn rss_mib() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|kib| kib.parse::<u64>().ok())
        .map(|kib| kib / 1_024)
}
