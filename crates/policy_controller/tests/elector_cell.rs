//! The two-process lease-file cell (design 10; the step's gate for the lease
//! file): two controllers on one host, each in its own process, over one
//! lease file. The holder takes the lease and renews it twice, then stalls
//! until the lease changed hands and tries once more; the taker follows,
//! takes the lease when its sight of the record is a duration old, and
//! renews. The parent reads both logs and checks what the fence promises:
//! the holder's fence closed before the taker's first fenced instant, and
//! the holder's late renewal was refused by the moved record. The children
//! are this test binary run again with a role, so the lock and the rename
//! cross real process boundaries.
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

use smg_policy_controller::elector::{ElectionSettings, Elector, FileLease, LeaseBackend, Tick};

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

fn word(tick: &Tick) -> String {
    match tick {
        Tick::Acquired { term } => format!("acquired:{term}"),
        Tick::Renewed { term } => format!("renewed:{term}"),
        Tick::Lost { term, .. } => format!("lost:{term}"),
        Tick::Following { term, .. } => format!("following:{term}"),
        Tick::Raced => "raced".to_string(),
    }
}

/// One step of a child: the wall time before the tick, what the tick did,
/// and the wall time by which the fence closes if it is open now. The fence
/// counts from `now`, taken just before the wall time, so `before` plus the
/// deadline is at or past the close: a strict bound, however long the tick
/// itself takes.
fn step(elector: &mut Elector, lines: &mut Vec<String>) -> Tick {
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
            for _ in 0..3 {
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
                if let Tick::Renewed { .. } = step(&mut elector, &mut lines) {
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

    let holder = parse(&fs::read_to_string(dir.path().join("lease.holder.log")).expect("log"));
    let taker = parse(&fs::read_to_string(dir.path().join("lease.taker.log")).expect("log"));
    let words = |lines: &[Line]| lines.iter().map(|l| l.word.clone()).collect::<Vec<_>>();

    assert_eq!(
        words(&holder),
        [
            "acquired:1",
            "renewed:1",
            "renewed:1",
            "lost:1",
            "following:2",
            "following:2",
            "following:2",
            "following:2",
            "following:2"
        ],
        "the holder's run"
    );
    let taker_words = words(&taker);
    let took = taker_words
        .iter()
        .position(|w| w == "acquired:2")
        .expect("the taker took the lease");
    assert!(took >= 1, "the taker followed first: {taker_words:?}");
    assert!(
        taker_words[..took].iter().all(|w| w == "following:1"),
        "{taker_words:?}"
    );
    assert_eq!(
        &taker_words[took + 1..],
        ["renewed:2"; 5],
        "the taker renews under its term"
    );

    // The fence's promise across processes: the holder's fence had closed
    // before the taker's take began.
    let holder_closed = holder
        .iter()
        .filter_map(|l| l.closes)
        .max()
        .expect("the holder led");
    let take_began = taker[took].before;
    assert!(
        take_began >= holder_closed,
        "the taker took at {take_began} while the holder's fence was open until {holder_closed}"
    );
    assert!(
        holder[3..].iter().all(|l| l.closes.is_none()),
        "the holder's fence stays closed after the loss"
    );
    assert!(
        taker[took..].iter().all(|l| l.closes.is_some()),
        "the taker's fence is open from its take"
    );
}
