//! The dry run: one pass at a time, what waits its turn and what is
//! refused at once, and what it answers.

use std::{
    io::Write as _,
    sync::Arc,
    time::{Duration, Instant},
};

use tokio::task::JoinSet;

use super::{
    super::{Generations, Unanswered, DIGEST_TEXT_LEN, DRY_RUNS_WAITING},
    gate, gated, generations, yaml, Gate, Gated,
};
use crate::{
    document::operator::{
        test_support::{document_of, standing, with_line, LEAST_LOAD, RANDOM},
        ApplyCounts, ApplyOutcome, DryRun, EnforceRefused, Format, SpecRejected,
        MAX_DOCUMENT_BYTES,
    },
    service::{controller::wire, proto},
};

/// One dry run at a time: while a pass stands in the validator a second
/// call's pass does not enter it, and both answer once the first ends.
#[tokio::test]
async fn a_second_dry_run_waits_its_turn() {
    gate!(Here);
    let (gate, generations) = (Here::gate(), gated::<Here>());
    let run =
        |document: &str| generations.dry_run_in_turn(String::new(), document.into(), Format::Yaml);
    gate.arm();
    // The first pass stands in the validator; a pace later the passes
    // that entered are counted. The validator is released before the
    // count is judged, so a count that fails ends the case.
    let beside = async {
        let entered = gate.entered_a_pace_on().await;
        gate.release();
        entered
    };
    let (first, second, entered) = tokio::join!(run(LEAST_LOAD), run(RANDOM), beside);
    assert_eq!(entered, 1, "a second dry run ran beside the first");
    for answer in [first, second] {
        let answer = answer.expect("a verdict");
        assert_eq!(answer.would_run.map(|(number, _)| number), Ok(1));
    }
    assert_eq!(gate.entered(), 2);
}

/// The pass holds the turn, not its caller: a caller that leaves while
/// its pass stands in the validator frees no turn, and the next call's
/// pass enters once that one ends.
#[tokio::test]
async fn a_caller_that_leaves_frees_no_turn_before_its_pass_ends() {
    gate!(Here);
    let (gate, generations) = (Here::gate(), gated::<Here>());
    let run =
        |document: &str| generations.dry_run_in_turn(String::new(), document.into(), Format::Yaml);
    gate.arm();
    // The first caller leaves a pace after its pass entered.
    tokio::select! {
        answer = run(LEAST_LOAD) => panic!("the first pass ended: {answer:?}"),
        _ = gate.entered_a_pace_on() => {}
    }
    let beside = async {
        tokio::time::sleep(Duration::from_millis(200)).await;
        let entered = gate.entered();
        gate.release();
        entered
    };
    let (second, entered) = tokio::join!(run(RANDOM), beside);
    assert_eq!(entered, 1, "the turn was freed under a pass that still ran");
    let second = second.expect("a verdict");
    assert_eq!(second.would_run.map(|(number, _)| number), Ok(1));
    assert_eq!(gate.entered(), 2);
}

/// The guard is read behind the wait: a guarded call that waits its turn
/// through an apply is refused by the generation that stands after it,
/// as the same call asked then is.
#[tokio::test]
async fn a_guard_is_read_once_the_turn_comes() {
    gate!(Here);
    let (gate, generations) = (Here::gate(), gated::<Here>());
    assert_eq!(
        yaml(&generations, LEAST_LOAD),
        Ok(ApplyOutcome::Generation(1))
    );
    let (_, first) = generations.standing();
    let first = first.expect("a digest stands").to_string();
    let run = |expected: &str| {
        generations.dry_run_in_turn(expected.to_owned(), RANDOM.into(), Format::Yaml)
    };
    gate.arm();
    // While the first pass stands in the validator the source applies
    // the next generation, and the guarded call waits through it.
    let beside = async {
        gate.entered_a_pace_on().await;
        let applied = yaml(&generations, RANDOM);
        gate.release();
        applied
    };
    let (unguarded, guarded, applied) = tokio::join!(run(""), run(&first), beside);
    assert_eq!(applied, Ok(ApplyOutcome::Generation(2)));
    // The first call read generation 1 before its pass stood.
    let unguarded = unguarded.expect("a verdict");
    assert_eq!(unguarded.would_run.map(|(number, _)| number), Ok(2));
    assert!(
        matches!(
            guarded,
            Err(Unanswered::Manager(EnforceRefused::Guarded(2, Some(_))))
        ),
        "{guarded:?}"
    );
}

/// A document over the bound takes no turn: it is refused by its size
/// while another's pass stands in the validator.
#[tokio::test]
async fn a_document_over_the_bound_waits_on_no_pass() {
    gate!(Here);
    let (gate, generations) = (Here::gate(), gated::<Here>());
    let over = MAX_DOCUMENT_BYTES + 1;
    let document = vec![b'#'; usize::try_from(over).expect("the bound fits")];
    gate.arm();
    let beside = async {
        gate.entered_a_pace_on().await;
        let call = generations.dry_run_in_turn(String::new(), document, Format::Yaml);
        let refused = tokio::time::timeout(Duration::from_secs(5), call).await;
        gate.release();
        refused
    };
    let first = generations.dry_run_in_turn(String::new(), LEAST_LOAD.into(), Format::Yaml);
    let (first, refused) = tokio::join!(first, beside);
    first.expect("a verdict");
    let refused = refused
        .expect("the refusal waits on no pass")
        .expect("a verdict");
    assert_eq!(refused.would_run, Err(SpecRejected::TooLarge(over)));
    // The guard is read ahead of the size: where nothing stands a guard
    // names no standing digest, under the bound and over it.
    let guard = format!("sha256:{}", "0".repeat(64));
    let small = generations.dry_run(&guard, LEAST_LOAD.as_bytes(), Format::Yaml);
    assert!(
        matches!(small, Err(EnforceRefused::Guarded(0, None))),
        "{small:?}"
    );
    let document = vec![b'#'; usize::try_from(over).expect("the bound fits")];
    let large = generations
        .dry_run_in_turn(guard, document, Format::Yaml)
        .await;
    assert!(
        matches!(
            large,
            Err(Unanswered::Manager(EnforceRefused::Guarded(0, None)))
        ),
        "{large:?}"
    );
}

/// A guard longer than a digest's text takes no turn: it names no
/// digest, and the guard refuses it while another's pass stands in the
/// validator.
#[tokio::test]
async fn a_guard_longer_than_a_digest_waits_on_no_pass() {
    gate!(Here);
    let (gate, generations) = (Here::gate(), gated::<Here>());
    assert_eq!(
        yaml(&generations, LEAST_LOAD),
        Ok(ApplyOutcome::Generation(1))
    );
    let (_, standing) = generations.standing();
    let standing = standing.expect("a digest stands").to_string();
    assert_eq!(standing.len(), DIGEST_TEXT_LEN);
    let longer = format!("{standing}0");
    gate.arm();
    let beside = async {
        gate.entered_a_pace_on().await;
        let call = generations.dry_run_in_turn(longer, RANDOM.into(), Format::Yaml);
        let refused = tokio::time::timeout(Duration::from_secs(5), call).await;
        gate.release();
        refused
    };
    let first = generations.dry_run_in_turn(String::new(), RANDOM.into(), Format::Yaml);
    let (first, refused) = tokio::join!(first, beside);
    first.expect("a verdict");
    let refused = refused.expect("the refusal waits on no pass");
    assert!(
        matches!(
            refused,
            Err(Unanswered::Manager(EnforceRefused::Guarded(1, Some(_))))
        ),
        "{refused:?}"
    );
}

/// No more dry runs wait than [`DRY_RUNS_WAITING`]: the call past them
/// is refused at once, a waiting call that ends frees its place, and
/// the ones let in all answer.
#[tokio::test]
async fn a_dry_run_past_the_waiting_ones_is_refused() {
    gate!(Here);
    let (gate, generations) = (Here::gate(), gated::<Here>());
    let run = |door: Arc<Generations>| async move {
        door.dry_run_in_turn(String::new(), LEAST_LOAD.into(), Format::Yaml)
            .await
    };
    // The places come to `free` once the calls let in were polled.
    let places = |door: Arc<Generations>, free: usize| async move {
        let started = Instant::now();
        while door.dry_turn.available() != free {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "the places did not come to {free} free"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    };
    gate.arm();
    let mut let_in = JoinSet::new();
    for _ in 0..DRY_RUNS_WAITING {
        let_in.spawn(run(Arc::clone(&generations)));
    }
    let last = let_in.spawn(run(Arc::clone(&generations)));
    places(Arc::clone(&generations), 0).await;
    let past = run(Arc::clone(&generations)).await;
    assert!(matches!(past, Err(Unanswered::Full)), "{past:?}");
    // A waiting call that ends leaves the wait: its place is one again.
    last.abort();
    places(Arc::clone(&generations), 1).await;
    let_in.spawn(run(Arc::clone(&generations)));
    places(Arc::clone(&generations), 0).await;
    gate.release();
    let (mut answered, mut left) = (0, 0);
    while let Some(answer) = let_in.join_next().await {
        match answer {
            Ok(answer) => {
                answer.expect("a verdict");
                answered += 1;
            }
            Err(gone) => {
                assert!(gone.is_cancelled(), "{gone}");
                left += 1;
            }
        }
    }
    assert_eq!((answered, left), (DRY_RUNS_WAITING + 1, 1));
    // A call that ran gave its place back, as one that left did.
    assert_eq!(generations.dry_turn.available(), DRY_RUNS_WAITING + 1);
}

/// A dry run answers what an apply would do at that instant and applies,
/// holds and counts nothing. The number is the one an apply would mint
/// now and it is not reserved: two dry runs answer one number, and an
/// apply that lands moves the next answer.
#[test]
fn a_dry_run_answers_what_an_apply_would_and_applies_nothing() {
    let (store, generations) = generations();
    let dry = |document: &[u8]| {
        generations
            .dry_run("", document, Format::Yaml)
            .expect("no guard")
    };
    let number = |run: DryRun| run.would_run.map(|(number, _)| number);
    let counts = |applied| ApplyCounts {
        applied,
        rejected: 0,
    };

    let first = dry(LEAST_LOAD.as_bytes());
    assert_eq!(first.standing, (0, None));
    // The feed before and after: nothing stands, and the document's one
    // profile reads the loads in every role.
    let feed = first.needs.map(wire::needs_diff).expect("the needs diff");
    assert_eq!(feed.before, Some(proto::NeedsSummary::default()));
    let after = feed.after.expect("the feed after");
    assert_eq!(after.models.len(), 4, "{after:?}");
    assert!(
        after.models.iter().all(|row| row.model == "*" && row.loads),
        "{after:?}"
    );
    let (would, digest) = first.would_run.expect("it would run");
    assert_eq!(would, 1);
    assert_eq!(number(dry(LEAST_LOAD.as_bytes())), Ok(1), "not reserved");
    assert_eq!(standing(&store), (0, None));
    assert!(!generations.bound());
    assert_eq!(generations.apply_counts(), counts(0));

    assert_eq!(
        yaml(&generations, LEAST_LOAD),
        Ok(ApplyOutcome::Generation(1))
    );
    let again = dry(LEAST_LOAD.as_bytes());
    assert_eq!(again.standing, (1, Some(digest)));
    assert_eq!(again.would_run, Ok((1, digest)), "the standing document");
    let feed = again.needs.map(wire::needs_diff).expect("the needs diff");
    assert_eq!(feed.before, feed.after, "the feed stands");
    assert_eq!(number(dry(RANDOM.as_bytes())), Ok(2));
    assert_eq!(number(dry(&with_line(RANDOM, 9))), Ok(9));
    assert_eq!(number(dry(&with_line(LEAST_LOAD, 9))), Ok(9));

    assert_eq!(yaml(&generations, RANDOM), Ok(ApplyOutcome::Generation(2)));
    assert_eq!(number(dry(LEAST_LOAD.as_bytes())), Ok(3), "an apply landed");
    assert_eq!(standing(&store).0, 2);
    assert_eq!(generations.apply_counts(), counts(2));
}

/// The Numbers table's dry-run row: one dry run that would run (parse,
/// digest, validate, the feed before and after) by the document's size,
/// with the bytes of the needs diff it answers.
#[test]
#[ignore = "a measurement for the Numbers table, run by name"]
fn a_dry_runs_time_by_the_documents_size() {
    use prost::Message as _;

    const ROUNDS: usize = 15;
    let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
    let mut out = std::io::stdout().lock();
    for size in [2 << 10, 64 << 10, bound - 128] {
        let (document, bindings) = document_of(size);
        let (_store, generations) = generations();
        assert_eq!(
            yaml(&generations, LEAST_LOAD),
            Ok(ApplyOutcome::Generation(1))
        );
        let mut micros = Vec::with_capacity(ROUNDS);
        let mut diff = 0;
        for _ in 0..ROUNDS {
            let began = Instant::now();
            let dry = generations
                .dry_run("", &document, Format::Yaml)
                .expect("no guard");
            micros.push(began.elapsed().as_micros());
            assert_eq!(dry.would_run.map(|(number, _)| number), Ok(2));
            diff = dry
                .needs
                .map_or(0, |needs| wire::needs_diff(needs).encoded_len());
        }
        micros.sort_unstable();
        writeln!(
                out,
                "DRY bytes={} bindings={bindings} rounds={ROUNDS} median_us={} worst_us={} diff_bytes={diff}",
                document.len(),
                micros[ROUNDS / 2],
                micros[ROUNDS - 1]
            )
            .expect("stdout");
    }
}
