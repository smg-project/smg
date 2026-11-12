//! The engine admin operations the gateway fans out to every worker
//! (`/flush_cache`, `/start_profile`, `/stop_profile`), answered as the
//! engines' servers answer them: a flush clears the worker's cache, as
//! `POST /admin/reset/{worker}` does; a profiling run is recorded at its
//! start and ends with one trace file written into its output directory.
//! One [`AdminOps`] per worker, registered in the process under the worker's
//! name, so `GET /admin/ops/{worker}` reports what the fan-out reached.

use std::{
    fs, io,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock, PoisonError,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::engine::Engine;

/// The options of a profiling run, as the gateway sends them: the JSON body
/// of `POST /start_profile` on the HTTP wire, the `StartProfileRequest` on
/// the gRPC wires. Every field is optional but the stage flag.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ProfileOptions {
    pub output_dir: Option<String>,
    pub start_step: Option<i32>,
    pub num_steps: Option<i32>,
    #[serde(default)]
    pub activities: Vec<String>,
    pub with_stack: Option<bool>,
    pub record_shapes: Option<bool>,
    #[serde(default)]
    pub profile_by_stage: bool,
}

/// A profiling run between its start and its stop.
struct ProfileRun {
    options: ProfileOptions,
    started_unix_ms: u128,
}

/// What the admin operations did on one worker, and what an encode worker
/// served (its encode requests reach no engine and no script, so the admin
/// API is where a bed reads that the encode side ran).
pub struct AdminOps {
    name: String,
    flushes: AtomicU64,
    profiles_started: AtomicU64,
    profiles_stopped: AtomicU64,
    encodes: AtomicU64,
    running: Mutex<Option<ProfileRun>>,
    last_trace: Mutex<Option<PathBuf>>,
}

static OPS: OnceLock<Mutex<Vec<Arc<AdminOps>>>> = OnceLock::new();

fn registry() -> &'static Mutex<Vec<Arc<AdminOps>>> {
    OPS.get_or_init(|| Mutex::new(Vec::new()))
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

impl AdminOps {
    /// The operations of the worker called `name`, registered in the process
    /// (a worker restarted under the same name keeps its counts).
    pub fn register(name: &str) -> Arc<Self> {
        let mut all = registry().lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(ops) = all.iter().find(|ops| ops.name == name) {
            return Arc::clone(ops);
        }
        let ops = Arc::new(Self {
            name: name.to_string(),
            flushes: AtomicU64::new(0),
            profiles_started: AtomicU64::new(0),
            profiles_stopped: AtomicU64::new(0),
            encodes: AtomicU64::new(0),
            running: Mutex::new(None),
            last_trace: Mutex::new(None),
        });
        all.push(Arc::clone(&ops));
        ops
    }

    /// The worker called `name`, if the process has one.
    pub fn find(name: &str) -> Option<Arc<Self>> {
        registry()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .find(|ops| ops.name == name)
            .cloned()
    }

    /// Every worker of the process.
    pub fn all() -> Vec<Arc<Self>> {
        registry()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// One encode request served by an encode worker (`encodes` in the
    /// status): the vLLM wire's encoder completion or the TokenSpeed wire's
    /// `Encode`. The gateway's own per-worker request count does not see the
    /// encode side, so a bed reads it here.
    pub fn record_encode(&self) {
        self.encodes.fetch_add(1, Ordering::Relaxed);
    }

    /// The flush: every engine of the worker clears its cache and publishes
    /// `AllBlocksCleared` (none on a canned worker, which has no cache), at
    /// once, whatever is in flight; the message names the count.
    pub fn flush<'a>(&self, engines: impl IntoIterator<Item = &'a Engine>) -> String {
        let cleared = engines.into_iter().map(Engine::reset).count();
        self.flushes.fetch_add(1, Ordering::Relaxed);
        format!("{}: cache cleared on {cleared} engine(s)", self.name)
    }

    /// The start of a profiling run; a second start while one runs is
    /// refused, as an engine's profiler refuses it.
    pub fn start_profile(&self, options: ProfileOptions) -> Result<String, String> {
        let mut running = self.running.lock().unwrap_or_else(PoisonError::into_inner);
        if running.is_some() {
            return Err(format!(
                "{}: a profiling run is already in progress",
                self.name
            ));
        }
        let message = format!(
            "{}: profiling started, traces go to {}",
            self.name,
            Self::output_dir(&options).display()
        );
        *running = Some(ProfileRun {
            options,
            started_unix_ms: unix_ms(),
        });
        self.profiles_started.fetch_add(1, Ordering::Relaxed);
        Ok(message)
    }

    /// The end of the profiling run: one trace file, the run's options and
    /// its span in it, written into the run's output directory (created
    /// when missing); no run in progress is refused.
    pub fn stop_profile(&self) -> Result<String, String> {
        let run = self
            .running
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .ok_or_else(|| format!("{}: no profiling run in progress", self.name))?;
        let stopped_unix_ms = unix_ms();
        let dir = Self::output_dir(&run.options);
        let path = dir.join(format!(
            "mock-worker-{}-{stopped_unix_ms}.trace.json",
            self.name.replace(':', "-")
        ));
        let trace = json!({
            "worker": self.name,
            "started_unix_ms": run.started_unix_ms,
            "stopped_unix_ms": stopped_unix_ms,
            "options": run.options,
            "traceEvents": [],
        });
        let write = || -> io::Result<()> {
            fs::create_dir_all(&dir)?;
            fs::write(&path, serde_json::to_vec_pretty(&trace).unwrap_or_default())
        };
        write().map_err(|e| format!("{}: trace file {}: {e}", self.name, path.display()))?;
        self.profiles_stopped.fetch_add(1, Ordering::Relaxed);
        *self
            .last_trace
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(path.clone());
        Ok(format!(
            "{}: trace written to {}",
            self.name,
            path.display()
        ))
    }

    /// The run's output directory: the requested one, else the process's
    /// temporary directory, as an engine falls back to its own default.
    fn output_dir(options: &ProfileOptions) -> PathBuf {
        options
            .output_dir
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("mock-worker-profiles"))
    }

    /// The counts and the run in progress, for the admin API.
    pub fn status(&self) -> Value {
        let running = self.running.lock().unwrap_or_else(PoisonError::into_inner);
        let last_trace = self
            .last_trace
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        json!({
            "worker": self.name,
            "flushes": self.flushes.load(Ordering::Relaxed),
            "profiles_started": self.profiles_started.load(Ordering::Relaxed),
            "profiles_stopped": self.profiles_stopped.load(Ordering::Relaxed),
            "encodes": self.encodes.load(Ordering::Relaxed),
            "profiling": running.is_some(),
            "last_trace": last_trace.as_ref().map(|p| p.display().to_string()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_profiling_run_writes_one_trace_file_into_its_output_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ops = AdminOps::register("test:1");
        let options = ProfileOptions {
            output_dir: Some(dir.path().join("traces").display().to_string()),
            num_steps: Some(3),
            ..Default::default()
        };
        assert!(ops.start_profile(options.clone()).is_ok());
        assert_eq!(ops.status()["profiling"], true);
        assert!(
            ops.start_profile(options).is_err(),
            "a second start while one runs is refused"
        );
        let message = ops.stop_profile().expect("stopped");
        let files: Vec<PathBuf> = fs::read_dir(dir.path().join("traces"))
            .expect("the output directory was created")
            .map(|entry| entry.expect("entry").path())
            .collect();
        assert_eq!(files.len(), 1, "{message}");
        let trace: Value =
            serde_json::from_slice(&fs::read(&files[0]).expect("trace")).expect("json");
        assert_eq!(trace["worker"], "test:1");
        assert_eq!(trace["options"]["num_steps"], 3);
        assert!(trace["traceEvents"].is_array());
        let status = ops.status();
        assert_eq!(status["profiles_started"], 1);
        assert_eq!(status["profiles_stopped"], 1);
        assert_eq!(status["profiling"], false);
        assert_eq!(status["last_trace"], files[0].display().to_string());
        assert!(
            ops.stop_profile().is_err(),
            "a stop with no run in progress is refused"
        );
    }

    #[test]
    fn a_flush_counts_and_names_the_engines_it_cleared() {
        let ops = AdminOps::register("test:2");
        assert_eq!(ops.flush([]), "test:2: cache cleared on 0 engine(s)");
        assert_eq!(ops.status()["flushes"], 1);
        assert!(AdminOps::find("test:2").is_some());
        assert!(AdminOps::find("test:none").is_none());
        assert!(
            Arc::ptr_eq(&ops, &AdminOps::register("test:2")),
            "a worker registered again keeps its counts"
        );
    }
}
