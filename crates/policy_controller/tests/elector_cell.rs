//! The two-process lease-file cell (design 10; the step's gate for the lease
//! file): two controllers on one host, each in its own process, over one
//! lease file. The holder takes the lease where no record stood and renews
//! it four times: its fence opens at the third renewal, one renew deadline
//! after such a take. Then it stalls until the lease changed hands and
//! tries once more; the taker follows, takes the lease when its sight of
//! the record is a duration old, and renews. The parent reads both logs and
//! checks what the fence promises: a fenced instant under a term begins
//! only when each fence under a lower term has closed, and a renewal is
//! refused once the record moved. A loaded host may hold either child back
//! for a record's duration, and the lease then changes hands once more: the
//! parent reads each log as a walk the election allows and holds the
//! promises over every pair of lines, never over a count of steps. The
//! children are this test binary run again with a role, so the lock and the
//! rename cross real process boundaries.
#![expect(
    clippy::expect_used,
    reason = "the cell's helpers run inside the test and its child processes; a failure there is the test's failure"
)]

use std::{
    env, fs,
    path::Path,
    process::Command,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use smg_policy_controller::service::elector::{
    ElectionSettings, Elector, FileLease, LeaseBackend, TickOutcome,
};

const ROLE: &str = "SMG_ELECTOR_CELL_ROLE";
const LEASE: &str = "SMG_ELECTOR_CELL_LEASE";
const TEST: &str = "two_processes_over_one_lease_file_hand_over_without_an_overlap";

/// A tenth of the design's settings, the same order between them.
fn settings() -> ElectionSettings {
    ElectionSettings {
        duration: Duration::from_millis(400),
        renew_period: Duration::from_millis(100),
        renew_deadline: Duration::from_millis(250),
    }
}

fn wall_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| u64::try_from(since.as_micros()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

fn word(tick: &TickOutcome) -> String {
    match tick {
        TickOutcome::Acquired { term } => format!("acquired:{term}"),
        TickOutcome::Renewed { term } => format!("renewed:{term}"),
        TickOutcome::Lost { term, .. } => format!("lost:{term}"),
        TickOutcome::Following { term, .. } => format!("following:{term}"),
        TickOutcome::Gone { term } => format!("gone:{term}"),
        TickOutcome::Waiting { term } => format!("waiting:{term}"),
        TickOutcome::Raced => "raced".to_string(),
    }
}

/// One step of a child: the wall time before the tick, what the tick did,
/// and the wall time by which the fence closes if it is open now. The fence
/// counts from `now`, taken just before the wall time, so `before` plus the
/// deadline is at or past the close: a strict bound, however long the tick
/// itself takes.
fn step(elector: &mut Elector, lines: &mut Vec<String>) -> TickOutcome {
    let now = Instant::now();
    let before = wall_micros();
    let tick = elector.tick(now).expect("a tick over the file");
    let closes = elector
        .fence()
        .leader_term(now)
        .map(|_| before + deadline_micros());
    lines.push(format!(
        "{before} {} {}",
        word(&tick),
        closes.map_or_else(|| "-".to_string(), |at| at.to_string())
    ));
    tick
}

fn deadline_micros() -> u64 {
    u64::try_from(settings().renew_deadline.as_micros()).unwrap_or(u64::MAX)
}

fn run_child(role: &str, path: &Path) -> Vec<String> {
    let mut elector = Elector::new(
        Box::new(FileLease::new(path)),
        format!("{role}@host:1"),
        settings(),
    )
    .expect("valid settings");
    let mut lines = Vec::new();
    let period = settings().renew_period;
    match role {
        "holder" => {
            for _ in 0..5 {
                step(&mut elector, &mut lines);
                thread::sleep(period);
            }
            // The stall: no renewal until the lease changed hands (bounded, so
            // a takeover that never comes fails the test instead of hanging).
            let mut watch = FileLease::new(path);
            let stalled = Instant::now();
            while stalled.elapsed() < Duration::from_secs(10) {
                let moved = watch
                    .read()
                    .expect("read")
                    .is_some_and(|record| record.holder != elector.identity());
                if moved {
                    break;
                }
                thread::sleep(Duration::from_millis(20));
            }
            for _ in 0..6 {
                step(&mut elector, &mut lines);
                thread::sleep(period);
            }
        }
        _ => {
            let mut renewals = 0;
            for _ in 0..100 {
                if let TickOutcome::Renewed { .. } = step(&mut elector, &mut lines) {
                    renewals += 1;
                    if renewals == 5 {
                        break;
                    }
                }
                thread::sleep(period);
            }
        }
    }
    lines
}

struct Line {
    before: u64,
    word: String,
    closes: Option<u64>,
}

impl Line {
    /// The word without its term: `renewed:2` reads `renewed`.
    fn kind(&self) -> &str {
        self.word
            .split_once(':')
            .map_or(self.word.as_str(), |(kind, _)| kind)
    }

    /// The term in the word: `renewed:2` reads 2.
    fn term(&self) -> u64 {
        self.word
            .split_once(':')
            .and_then(|(_, term)| term.parse().ok())
            .expect("a word with its term")
    }

    /// Whether the tick holds the lease: a take or a renewal.
    fn holds(&self) -> bool {
        matches!(self.kind(), "acquired" | "renewed")
    }
}

fn parse(log: &str) -> Vec<Line> {
    log.lines()
        .map(|line| {
            let mut fields = line.split(' ');
            let before = fields
                .next()
                .and_then(|f| f.parse().ok())
                .expect("the wall time");
            let word = fields.next().expect("the word").to_string();
            let closes = fields.next().and_then(|f| f.parse().ok());
            Line {
                before,
                word,
                closes,
            }
        })
        .collect()
}

/// Holds one child's lines to a walk the election allows, whatever the host
/// did to its schedule. A take opens a run under a new term: first of all,
/// or after following or a loss, above the term read there. A renewal stays
/// in its run and a loss ends it, both under the run's term. Following comes
/// first of all, after a loss under a higher term, or after following under
/// no lower one. A line that holds no lease holds no fence, and a fence that
/// is open stays open at each later renewal of its run.
fn walk(lines: &[Line], told: &str) {
    for (at, line) in lines.iter().enumerate() {
        let prior = at.checked_sub(1).and_then(|before| lines.get(before));
        let allowed = match (line.kind(), prior) {
            ("acquired" | "following", None) => true,
            ("acquired", Some(prior)) => !prior.holds() && line.term() > prior.term(),
            ("renewed" | "lost", Some(prior)) => prior.holds() && line.term() == prior.term(),
            ("following", Some(prior)) if prior.kind() == "lost" => line.term() > prior.term(),
            ("following", Some(prior)) => {
                prior.kind() == "following" && line.term() >= prior.term()
            }
            _ => false,
        };
        assert!(allowed, "line {at} is no step of an election: {told}");
        if line.holds() {
            let open = prior.is_some_and(|prior| prior.holds() && prior.closes.is_some());
            assert!(
                line.closes.is_some() || !open,
                "line {at} lets an open fence go at a renewal: {told}"
            );
        } else {
            assert!(
                line.closes.is_none(),
                "line {at} holds a fence without the lease: {told}"
            );
        }
    }
}

#[test]
fn two_processes_over_one_lease_file_hand_over_without_an_overlap() {
    if let Ok(role) = env::var(ROLE) {
        let path = env::var(LEASE).expect("the lease path");
        let lines = run_child(&role, Path::new(&path));
        fs::write(format!("{path}.{role}.log"), lines.join("\n")).expect("the child's log");
        return;
    }
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("lease");
    let spawn = |role: &str| {
        Command::new(env::current_exe().expect("this test binary"))
            .args(["--exact", TEST, "--test-threads=1"])
            .env(ROLE, role)
            .env(LEASE, &path)
            .spawn()
            .expect("a child process")
    };
    let mut holder = spawn("holder");
    // The taker starts once the holder's first write landed, so its first
    // look finds the record and the two never race for the first take.
    let started = Instant::now();
    while !path.exists() {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "the holder never wrote the lease"
        );
        thread::sleep(Duration::from_millis(5));
    }
    let mut taker = spawn("taker");
    assert!(
        holder.wait().expect("the holder").success(),
        "the holder's run"
    );
    assert!(
        taker.wait().expect("the taker").success(),
        "the taker's run"
    );

    // A tick whose take lost the compare-and-swap to a renewal landing at the
    // same instant logs `raced`: it carries no term and no fence, and a loaded
    // host produces one whenever a tick slips past the record's duration. The
    // cell's promises are read over the other ticks, and the cell checks that
    // no raced tick opened a fence.
    let settled = |log: &str| {
        let (raced, kept): (Vec<Line>, Vec<Line>) =
            parse(log).into_iter().partition(|l| l.word == "raced");
        assert!(
            raced.iter().all(|l| l.closes.is_none()),
            "a raced tick carries no fence: {:?}",
            raced
                .iter()
                .map(|l| (l.before, l.closes))
                .collect::<Vec<_>>()
        );
        kept
    };
    let holder = settled(&fs::read_to_string(dir.path().join("lease.holder.log")).expect("log"));
    let taker = settled(&fs::read_to_string(dir.path().join("lease.taker.log")).expect("log"));
    let words = |lines: &[Line]| lines.iter().map(|l| l.word.clone()).collect::<Vec<_>>();
    let (holder_words, taker_words) = (words(&holder), words(&taker));
    let told = format!("the holder {holder_words:?}, the taker {taker_words:?}");

    // A child held back for a record's duration loses the lease to the
    // other, at any step and more than once on a loaded host. So each log
    // is read as a walk the election allows, the promises are held over
    // every pair of lines, and a position is pinned only where the scripts
    // make it certain.
    walk(&holder, &told);
    walk(&taker, &told);

    // The holder takes, renews four times and reads its loss at the late
    // step. A holder descheduled for the record's duration between two of
    // its first five steps reads the loss at an earlier one, after fewer
    // renewals.
    let lost = holder_words
        .iter()
        .position(|w| w == "lost:1")
        .expect("the holder read its loss");
    assert!(
        (1..=5).contains(&lost)
            && holder_words[0] == "acquired:1"
            && holder_words[1..lost].iter().all(|w| w == "renewed:1"),
        "the holder's run: {told}"
    );
    // The taker follows first, takes the term above and renews, and its
    // script ends at its fifth renewal. A taker descheduled for its own
    // record's duration between two of them reads its loss at the next
    // step: the old holder took the term above once the taker's fence was
    // closed. The taker follows again, takes once that record stood still,
    // and its renewals count on.
    let took = taker_words
        .iter()
        .position(|w| w == "acquired:2")
        .expect("the taker took the lease");
    assert!(
        took >= 1 && taker_words[..took].iter().all(|w| w == "following:1"),
        "the taker followed first: {told}"
    );
    assert!(
        taker.last().is_some_and(|l| l.kind() == "renewed")
            && taker.iter().filter(|l| l.kind() == "renewed").count() == 5,
        "the taker's script ends at its fifth renewal: {told}"
    );
    // A loss is read only of a lease the other process took, and no term is
    // held by both.
    for (own, other) in [(&holder, &taker), (&taker, &holder)] {
        for loss in own.iter().filter(|l| l.kind() == "lost") {
            assert!(
                other
                    .iter()
                    .any(|l| l.kind() == "acquired" && l.term() > loss.term()),
                "a loss of a lease the other process never took: {told}"
            );
        }
    }
    assert!(
        holder.iter().filter(|l| l.holds()).all(|held| taker
            .iter()
            .filter(|l| l.holds())
            .all(|other| other.term() != held.term())),
        "one term held by both: {told}"
    );
    // A followed term is a term the other process took, and the takes of
    // the two logs are one a term: 1, 2, 3 and on, each once.
    for (own, other) in [(&holder, &taker), (&taker, &holder)] {
        for followed in own.iter().filter(|l| l.kind() == "following") {
            assert!(
                other
                    .iter()
                    .any(|l| l.kind() == "acquired" && l.term() == followed.term()),
                "a followed term the other process never took: {told}"
            );
        }
    }
    let mut takes: Vec<u64> = holder
        .iter()
        .chain(&taker)
        .filter(|l| l.kind() == "acquired")
        .map(Line::term)
        .collect();
    takes.sort_unstable();
    assert!(
        (1..).zip(&takes).all(|(term, took)| term == *took),
        "one take a term: {told}"
    );

    // The lease file was not there: the take opens no fence, and the third
    // renewal, three periods after it, is past the renew deadline. A holder
    // that read its loss before that renewal never led. Every later take
    // found a record that stood still its duration and leads from the take.
    assert!(
        holder[0].closes.is_none(),
        "the take of an absent record opens no fence"
    );
    assert!(
        lost <= 3 || holder[3].closes.is_some(),
        "the fence opens at the first renewal a renew deadline after the take"
    );
    assert!(
        holder
            .iter()
            .chain(&taker)
            .filter(|l| l.holds() && l.term() > 1)
            .all(|l| l.closes.is_some()),
        "a take of a record that stood still leads from the take: {told}"
    );
    // The fence's promise across processes: a fenced instant under a term
    // begins only when each fence under a lower term has closed.
    for (lower, higher) in [(&holder, &taker), (&taker, &holder)] {
        for early in lower {
            let Some(closed) = early.closes else {
                continue;
            };
            for late in higher
                .iter()
                .filter(|l| l.closes.is_some() && l.term() > early.term())
            {
                assert!(
                    late.before >= closed,
                    "term {} was fenced at {} while term {} was open until {closed}: {told}",
                    late.term(),
                    late.before,
                    early.term()
                );
            }
        }
    }
}
