# wfaas

A high-performance, async Rust workflow engine for orchestrating multi-step operations with full DAG (Directed Acyclic Graph) support.

## Features

- **Parallel Execution** - Independent steps run concurrently via DAG-based scheduling
- **Flexible Dependencies** - Support for `depends_on` (all) and `depends_on_any` (at least one) semantics
- **Configurable Retry Logic** - Fixed, exponential, or linear backoff strategies per step
- **Conditional Branching** - Runtime conditions to skip or execute steps
- **Scheduled Execution** - Delay steps or schedule them for specific times
- **State Persistence** - Pluggable state stores for durability and recovery
- **Event-Driven Observability** - Subscribe to workflow and step lifecycle events
- **Graceful Shutdown** - Clean shutdown with timeout and force-cancel options

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
wfaas = "1.0"
```

## Quick Start

```rust
use wfaas::{
    WorkflowEngine, WorkflowDefinition, StepDefinition, StepExecutor, StepResult,
    WorkflowContext, WorkflowData, WorkflowId, WorkflowResult,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

// 1. Define your workflow data
#[derive(Clone, Serialize, Deserialize, Default)]
struct MyWorkflowData {
    input: String,
    result: Option<String>,
}

impl WorkflowData for MyWorkflowData {
    fn workflow_type() -> &'static str {
        "my_workflow"
    }
}

// 2. Implement step executors
struct ProcessStep;

#[async_trait]
impl StepExecutor<MyWorkflowData> for ProcessStep {
    async fn execute(&self, ctx: &mut WorkflowContext<MyWorkflowData>) -> WorkflowResult<StepResult> {
        ctx.data.result = Some(format!("Processed: {}", ctx.data.input));
        Ok(StepResult::Success)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 3. Create the engine and register a workflow
    let engine = WorkflowEngine::new();

    let workflow = WorkflowDefinition::new("process_workflow", "Process Workflow")
        .add_step(StepDefinition::new("process", "Process Input", Arc::new(ProcessStep)));

    engine.register_workflow(workflow)?;

    // 4. Start workflow instances
    let instance_id = engine
        .start_workflow(WorkflowId::new("process_workflow"), MyWorkflowData {
            input: "Hello, World!".into(),
            result: None,
        })
        .await?;

    // 5. Wait for completion (returns Result<String, String>)
    match engine.wait_for_completion(instance_id, "example", Duration::from_secs(30)).await {
        Ok(msg) => println!("{}", msg),
        Err(e) => return Err(e.into()),
    }

    Ok(())
}
```

## Core Concepts

### WorkflowData

Your workflow context must implement `WorkflowData`:

```rust
pub trait WorkflowData: Serialize + DeserializeOwned + Send + Sync + Clone + 'static {
    fn workflow_type() -> &'static str;
}
```

This enables state serialization for persistence and recovery.

**Handling Non-Serializable Fields**

For fields that cannot be serialized (like database connections or application context), use `#[serde(skip)]`:

```rust
#[derive(Clone, Serialize, Deserialize)]
struct MyWorkflowData {
    pub config: MyConfig,
    pub result: Option<String>,

    // Non-serializable fields must be skipped and re-initialized after deserialization
    #[serde(skip, default)]
    pub app_context: Option<Arc<AppContext>>,
}
```

### StepExecutor

Each step requires an executor that implements `StepExecutor<D>`:

```rust
#[async_trait]
pub trait StepExecutor<D: WorkflowData>: Send + Sync {
    async fn execute(&self, context: &mut WorkflowContext<D>) -> WorkflowResult<StepResult>;

    // Optional: control retry behavior
    fn is_retryable(&self, _error: &WorkflowError) -> bool { true }

    // Optional: lifecycle hooks
    async fn on_success(&self, _context: &WorkflowContext<D>) -> WorkflowResult<()> { Ok(()) }
    async fn on_failure(&self, _context: &WorkflowContext<D>, _error: &WorkflowError) -> WorkflowResult<()> { Ok(()) }
}
```

### FunctionStep

For simple steps, use `FunctionStep` instead of implementing a struct:

```rust
use wfaas::FunctionStep;

let step = FunctionStep::new(|ctx| Box::pin(async move {
    ctx.data.result = Some("done".into());
    Ok(StepResult::Success)
}));
```

### Step Results

```rust
pub enum StepResult {
    Success,  // Step completed successfully
    Failure,  // Step failed (may trigger retry)
    Skip,     // Step was skipped
}
```

## Building DAG Workflows

### Dependencies

Steps can declare dependencies on other steps:

```rust
let workflow = WorkflowDefinition::new("dag_example", "DAG Example")
    // Independent steps run in parallel
    .add_step(StepDefinition::new("fetch_a", "Fetch A", Arc::new(FetchA)))
    .add_step(StepDefinition::new("fetch_b", "Fetch B", Arc::new(FetchB)))
    // This step waits for BOTH fetch_a AND fetch_b
    .add_step(
        StepDefinition::new("combine", "Combine Results", Arc::new(Combine))
            .depends_on(&["fetch_a", "fetch_b"])
    );
```

Execution flow:
```
fetch_a ──┐
          ├──> combine
fetch_b ──┘
```

### depends_on vs depends_on_any

- `depends_on(&["a", "b"])` - Waits for **ALL** listed steps to complete
- `depends_on_any(&["a", "b"])` - Waits for **AT LEAST ONE** to complete

```rust
// This step runs when step_a succeeds AND (step_b OR step_c completes)
.add_step(
    StepDefinition::new("final", "Final Step", Arc::new(FinalStep))
        .depends_on(&["step_a"])
        .depends_on_any(&["step_b", "step_c"])
)
```

## Retry Configuration

Configure retry behavior per step or set workflow defaults:

```rust
use wfaas::{RetryPolicy, BackoffStrategy};

// Per-step retry
.add_step(
    StepDefinition::new("unreliable", "Unreliable Step", Arc::new(UnreliableStep))
        .with_retry(RetryPolicy {
            max_attempts: 5,
            backoff: BackoffStrategy::Exponential {
                base: Duration::from_millis(100),
                max: Duration::from_secs(30),
            },
        })
        .with_timeout(Duration::from_secs(60))
)

// Workflow default retry
let workflow = WorkflowDefinition::new("retry_example", "Retry Example")
    .with_default_retry(RetryPolicy {
        max_attempts: 3,
        backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
    })
    .with_default_timeout(Duration::from_secs(30));
```

### Backoff Strategies

```rust
pub enum BackoffStrategy {
    Fixed(Duration),                           // Same delay each retry
    Exponential { base: Duration, max: Duration }, // Doubles each retry up to max
    Linear { increment: Duration, max: Duration }, // Adds increment each retry up to max
}
```

## Failure Handling

Control what happens when a step fails:

```rust
use wfaas::FailureAction;

.add_step(
    StepDefinition::new("optional", "Optional Step", Arc::new(OptionalStep))
        .with_failure_action(FailureAction::ContinueNextStep) // Don't fail workflow
)
```

Options:
- `FailureAction::FailWorkflow` - Entire workflow fails (default)
- `FailureAction::ContinueNextStep` - Skip failed step, continue workflow
- `FailureAction::RetryIndefinitely` - Keep retrying until manual intervention

A panic in a step's `execute` method or `run_if` condition is logged and treated as a failure with the
error text `Step failed`; its failure action applies, and no further attempt is made whatever its retry
policy says. The step reads `Failed` when the action ends the workflow and `Skipped` when the action
continues it. A panic in an `on_success` or `on_failure` hook is also logged and leaves the outcome the
step has written, as a hook that returns an error does.

## Conditional Execution

Skip steps based on runtime conditions:

```rust
.add_step(
    StepDefinition::new("conditional", "Conditional Step", Arc::new(ConditionalStep))
        .run_if(|ctx| ctx.data.should_execute)
)
```

When a condition returns `false`, the step is skipped but dependents can still proceed.

## Scheduled and Delayed Execution

```rust
use chrono::{Utc, Duration as ChronoDuration};

// Delay after dependencies are met
.add_step(
    StepDefinition::new("delayed", "Delayed Step", Arc::new(DelayedStep))
        .with_delay(Duration::from_secs(60))
)

// Run at a specific time
.add_step(
    StepDefinition::new("scheduled", "Scheduled Step", Arc::new(ScheduledStep))
        .scheduled_at(Utc::now() + ChronoDuration::hours(1))
)
```

## Event Observability

Subscribe to workflow events for monitoring and logging:

```rust
use wfaas::{EventSubscriber, WorkflowEvent};

struct MySubscriber;

#[async_trait]
impl EventSubscriber for MySubscriber {
    async fn on_event(&self, event: &WorkflowEvent) {
        match event {
            WorkflowEvent::WorkflowStarted { instance_id, .. } => {
                println!("Workflow {} started", instance_id);
            }
            WorkflowEvent::StepSucceeded { step_id, duration, .. } => {
                println!("Step {} completed in {:?}", step_id, duration);
            }
            WorkflowEvent::WorkflowFailed { instance_id, error, .. } => {
                eprintln!("Workflow {} failed: {}", instance_id, error);
            }
            _ => {}
        }
    }
}

// Register subscriber
engine.event_bus().subscribe(Arc::new(MySubscriber)).await;
```

Available events:
- `WorkflowStarted`, `WorkflowCompleted`, `WorkflowFailed`, `WorkflowCancelled`
- `StepStarted`, `StepSucceeded`, `StepFailed`, `StepRetrying`

A built-in `LoggingSubscriber` integrates with the `tracing` crate.

## State Persistence

Implement custom state stores for durability:

```rust
use wfaas::{StateStore, WorkflowState};

#[async_trait]
impl<D: WorkflowData> StateStore<D> for MyDatabaseStore {
    async fn save(&self, state: WorkflowState<D>) -> WorkflowResult<()> { /* ... */ }
    async fn load(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<WorkflowState<D>> { /* ... */ }
    async fn update<F>(&self, instance_id: WorkflowInstanceId, f: F) -> WorkflowResult<()>
    where F: FnOnce(&mut WorkflowState<D>) + Send { /* ... */ }
    async fn delete(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<()> { /* ... */ }
    async fn list_active(&self) -> WorkflowResult<Vec<WorkflowState<D>>> { /* ... */ }
    async fn list_all(&self) -> WorkflowResult<Vec<WorkflowState<D>>> { /* ... */ }
    async fn is_cancelled(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<bool> { /* ... */ }
    async fn cleanup_old_workflows(&self, ttl: Duration) -> usize { /* ... */ }
    async fn get_context(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<WorkflowContext<D>> { /* ... */ }
    async fn cleanup_if_terminal(&self, instance_id: WorkflowInstanceId) -> bool { /* ... */ }
}

// Use custom store
let engine = WorkflowEngine::with_store(MyDatabaseStore::new());
```

The default `InMemoryStore` is thread-safe and suitable for development or ephemeral workflows.

## Workflow Management

### Cancellation

```rust
engine.cancel_workflow(instance_id).await?;
```

### Status Queries

```rust
let state = engine.get_status(instance_id).await?;
println!("Status: {:?}", state.status);
println!("Step states: {:?}", state.step_states);
```

### Graceful Shutdown

```rust
// Signal shutdown (stops accepting new workflows)
engine.shutdown();

// Wait for active workflows to complete
if !engine.wait_for_shutdown(Duration::from_secs(30)).await {
    // Force cancel remaining workflows
    let cancelled = engine.force_cancel_all().await;
    println!("Force cancelled {} workflows", cancelled);
}
```

### Automatic Cleanup

```rust
// Start background cleanup task (removes completed workflows after TTL)
let cleanup_handle = engine.start_cleanup_task(
    Some(Duration::from_secs(24 * 60 * 60)),  // TTL: 24 hours
    Some(Duration::from_secs(5 * 60)),        // Check interval: 5 minutes
).await;
```

## Error Handling

```rust
use wfaas::WorkflowError;

match engine.start_workflow(workflow_id, data).await {
    Ok(instance_id) => println!("Started: {}", instance_id),
    Err(WorkflowError::DefinitionNotFound(id)) => eprintln!("Unknown workflow: {}", id),
    Err(WorkflowError::ShuttingDown) => eprintln!("Engine is shutting down"),
    Err(e) => eprintln!("Error: {}", e),
}
```

Error types:
- `NotFound` - Workflow instance not found
- `DefinitionNotFound` - Workflow definition not registered
- `StepFailed` - Step execution failed
- `StepTimeout` - Step exceeded timeout
- `Cancelled` - Workflow was cancelled
- `InvalidStateTransition` - Invalid status change
- `ContextValueNotFound` - Missing context value
- `TypeMismatch` - Type conversion error
- `ShuttingDown` - Engine is shutting down

## License

Apache-2.0
