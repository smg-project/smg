//! The scheduler's phases over a three-step definition: `a`, `b` after
//! `a`, and `c` after either of them.

use serde::{Deserialize, Serialize};

use super::*;
use crate::executor::StepExecutor;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Data;

impl WorkflowData for Data {
    fn workflow_type() -> &'static str {
        "scheduler-test"
    }
}

struct Noop;

#[async_trait::async_trait]
impl StepExecutor<Data> for Noop {
    async fn execute(&self, _context: &mut WorkflowContext<Data>) -> WorkflowResult<StepResult> {
        Ok(StepResult::Success)
    }
}

fn step(id: &str) -> StepDefinition<Data> {
    StepDefinition::new(id, id, Arc::new(Noop))
}

/// `a`; `b` after `a`; `c` after `a` or `b`; `c` delayed when `delay_c`.
fn three_steps(delay_c: Option<Duration>) -> WorkflowDefinition<Data> {
    let mut c = step("c").depends_on_any(&["a", "b"]);
    if let Some(delay) = delay_c {
        c = c.with_delay(delay);
    }
    let mut definition = WorkflowDefinition::new("wf", "wf")
        .add_step(step("a"))
        .add_step(step("b").depends_on(&["a"]))
        .add_step(c);
    definition.validate().expect("a valid definition");
    definition
}

#[test]
fn the_drain_puts_the_dependents_of_a_finished_step_up_for_a_check_and_says_whether_any_came() {
    let definition = three_steps(None);
    let (tx, mut rx) = mpsc::channel::<(StepId, StepResult)>(8);
    let mut pending = VecDeque::new();

    assert!(!drain_completions(&mut rx, &definition, &mut pending, "t"));
    assert!(pending.is_empty());

    tx.try_send((StepId::new("a"), StepResult::Success))
        .unwrap();
    assert!(drain_completions(&mut rx, &definition, &mut pending, "t"));
    assert_eq!(
        pending.iter().copied().collect::<Vec<_>>(),
        [1, 2],
        "b and c follow a"
    );

    pending.clear();
    tx.try_send((StepId::new("b"), StepResult::Failure))
        .unwrap();
    tx.try_send((StepId::new("a"), StepResult::Skip)).unwrap();
    assert!(drain_completions(&mut rx, &definition, &mut pending, "t"));
    assert_eq!(
        pending.iter().copied().collect::<Vec<_>>(),
        [1, 2],
        "a failed step's dependents are not put up; a skipped step's are"
    );
}

#[test]
fn a_completion_puts_dependents_up_for_a_check_only_when_its_step_succeeded_or_was_skipped() {
    let definition = three_steps(None);
    let a = StepId::new("a");
    let dependents_of_a: Vec<usize> = definition.get_dependent_indices(&a).to_vec();
    assert!(
        !dependents_of_a.is_empty(),
        "the fixture gives `a` a dependent"
    );

    let mut pending = VecDeque::new();
    enqueue_dependents(&definition, &a, StepResult::Failure, &mut pending);
    assert!(pending.is_empty(), "a failed step unblocks nothing");

    enqueue_dependents(&definition, &a, StepResult::Skip, &mut pending);
    assert_eq!(Vec::from(pending.clone()), dependents_of_a);

    pending.clear();
    enqueue_dependents(&definition, &a, StepResult::Success, &mut pending);
    assert_eq!(Vec::from(pending), dependents_of_a);
}

#[test]
fn the_scan_finds_the_steps_whose_dependencies_are_satisfied() {
    let definition = three_steps(None);
    let mut tracker = StepTracker::default();
    tracker.completed.insert(StepId::new("a"));
    let mut pending: VecDeque<usize> = [1, 2].into_iter().collect();

    let scan = scan_ready(&tracker, &definition, &mut pending);
    assert_eq!(scan.deps_ready, [1, 2], "b after a, c after any of a or b");
    assert!(scan.wait_ready.is_empty());
    assert_eq!(
        (scan.total_processed, scan.running, scan.waiting),
        (1, 0, 0)
    );
    assert!(pending.is_empty(), "the checks are consumed");
}

#[test]
fn scheduling_launches_an_any_step_once_and_defers_a_delayed_step_to_its_wait() {
    let scan = |deps_ready: Vec<usize>| ReadyScan {
        wait_ready: Vec::new(),
        deps_ready,
        total_processed: 1,
        running: 0,
        waiting: 0,
    };
    type Engine = WorkflowEngine<Data, InMemoryStore<Data>>;

    // c satisfied twice (by a and by b) launches once.
    let definition = three_steps(None);
    let mut tracker = StepTracker::default();
    let (ready, waiting) = Engine::schedule_ready(&mut tracker, &definition, &scan(vec![1, 2, 2]));
    assert_eq!((ready, waiting), (vec![1, 2], 0));

    // A delayed c goes to wait instead, and a wait that is over launches.
    let definition = three_steps(Some(Duration::from_secs(60)));
    let mut tracker = StepTracker::default();
    let (ready, waiting) = Engine::schedule_ready(&mut tracker, &definition, &scan(vec![1, 2]));
    assert_eq!((ready, waiting), (vec![1], 1));
    assert!(tracker.waiting_until.contains_key(&2));
    let over = ReadyScan {
        wait_ready: vec![2],
        deps_ready: Vec::new(),
        total_processed: 2,
        running: 0,
        waiting: 1,
    };
    let (ready, waiting) = Engine::schedule_ready(&mut tracker, &definition, &over);
    assert_eq!((ready, waiting), (vec![2], 0));
    assert!(!tracker.waiting_until.contains_key(&2));
}

#[test]
fn the_wait_is_until_the_soonest_delayed_step_and_a_short_tick_once_every_wait_is_over() {
    let now = std::time::Instant::now();
    let mut tracker = StepTracker::default();
    tracker.set_waiting(1, now + Duration::from_secs(50));
    tracker.set_waiting(2, now + Duration::from_secs(5));
    let wait = time_to_next_wait(&tracker);
    assert!(
        wait > Duration::from_secs(4) && wait <= Duration::from_secs(5),
        "{wait:?}"
    );

    let mut tracker = StepTracker::default();
    tracker.set_waiting(1, now - Duration::from_secs(1));
    assert_eq!(time_to_next_wait(&tracker), Duration::from_millis(10));
}

#[tokio::test]
async fn a_deadlock_with_no_failed_step_keeps_the_last_attempts_step() {
    let engine: WorkflowEngine<Data> = WorkflowEngine::new();
    let id = WorkflowInstanceId::new();
    let last = StepId::new("last_attempt");
    let mut state = WorkflowState::new(id, WorkflowId::new("deadlock"), Data);
    state.status = WorkflowStatus::Running;
    state.current_step = Some(last.clone());
    engine
        .state_store()
        .save(state)
        .await
        .expect("the state is saved");

    // No step failed, so the blocked verdict is a deadlock.
    let tracker = Arc::new(RwLock::new(StepTracker::default()));
    engine
        .fail_blocked(id, &tracker)
        .await
        .expect("the verdict is written");

    let state = engine.get_status(id).await.expect("the run's state");
    assert_eq!(state.status, WorkflowStatus::Failed);
    assert_eq!(state.current_step, Some(last));
}
