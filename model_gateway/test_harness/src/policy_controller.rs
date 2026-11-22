//! The switch that turns the policy controller on in every context this
//! harness builds.
//!
//! A test builds its context from a `RouterConfig` and does not name
//! `--policy-controller`. With `SMG_TEST_POLICY_CONTROLLER=on` in a test
//! run's environment, each context is built as a gateway started with that
//! flag and no controller's address builds its own: the configuration
//! carries the flag, and the policy registry, the policy engine and the
//! engine object are the ones the gateway's start builds under it. The
//! run's requests are then placed as that gateway places them. Unset, or
//! set to anything else, every context is built as before.
//!
//! With the flag on the policy engine is to place every request. Until it
//! does, the policy registry still hands out a classic policy for some
//! places (a side of a prefill/decode pair, a policy the engine does not
//! pick by itself yet), and the placement asks it. A run that names a file
//! in `SMG_TEST_CLASSIC_PICKS` gets one line there for each such pick: the
//! test binary, the thread and the policy.
//!
//! The pull-request tests run the gateway's integration tests a second
//! time with the switch on, and hold the picks to the list of the cases
//! known to be placed so (`scripts/ci_rust_tests.sh policy-controller`,
//! `model_gateway/tests/policy_controller_known.tsv`).

use std::{
    env,
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    sync::{Arc, OnceLock},
    thread,
};

use smg::{
    app_context::{AppContext, AppContextBuilder},
    config::RouterConfig,
    policies::PolicyRegistry,
    routers::common::snapshot::WorkerRegistry,
};

/// The environment variable of the switch.
pub const SWITCH: &str = "SMG_TEST_POLICY_CONTROLLER";

/// The environment variable that names the file a run's classic picks are
/// written to.
pub const CLASSIC_PICKS: &str = "SMG_TEST_CLASSIC_PICKS";

/// Whether this test run turns the policy controller on.
pub fn on() -> bool {
    env::var_os(SWITCH).is_some_and(|value| value == "on")
}

/// Where this process writes its classic picks, and the name it writes them
/// under.
struct Picks {
    file: PathBuf,
    /// The running executable's file name: a test binary's is its target's
    /// name and a hash.
    executable: String,
}

/// The picks file of this process: read from the environment once, `None`
/// where the variable is unset.
fn picks() -> Option<&'static Picks> {
    static PICKS: OnceLock<Option<Picks>> = OnceLock::new();
    PICKS
        .get_or_init(|| {
            let file = PathBuf::from(env::var_os(CLASSIC_PICKS)?);
            let executable = env::current_exe()
                .ok()
                .and_then(|path| Some(path.file_name()?.to_string_lossy().into_owned()))
                .unwrap_or_default();
            Some(Picks { file, executable })
        })
        .as_ref()
}

/// Appends the line of one pick by the classic policy named `policy`: the
/// executable, the thread's name and the policy's name, tab separated. A
/// test's thread carries the test's name, so the line names the case whose
/// request the policy placed. The gateway calls this at each such pick
/// ([`PolicyRegistry::on_classic_pick`]).
///
/// A line that cannot be written fails the case that picked: the run that
/// reads the file would otherwise pass on a pick it never saw.
#[expect(
    clippy::panic,
    reason = "test helper - a pick that cannot be written down fails its case"
)]
fn write_classic_pick(policy: &'static str) {
    let Some(picks) = picks() else {
        return;
    };
    let line = format!(
        "{}\t{}\t{policy}\n",
        picks.executable,
        thread::current().name().unwrap_or("?")
    );
    // One write a line, to a file opened for append: the lines of threads
    // that pick at once stay whole.
    let written = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&picks.file)
        .and_then(|mut file| file.write_all(line.as_bytes()));
    if let Err(error) = written {
        panic!(
            "{CLASSIC_PICKS} names {}, which cannot be written: {error}",
            picks.file.display()
        );
    }
}

/// The policy parts of a context: the builder it is built on, its policy
/// registry and the configuration it keeps.
///
/// With the switch off they are a new builder, the registry `classic`
/// builds from `config`, and `config` as given. With it on they are the
/// gateway's own: `config` with `policy_controller` set and no controller's
/// address, and over `workers` what the start's two policy steps build
/// under it ([`AppContextBuilder::with_policies_started`]). A configuration
/// the start refuses under the flag fails the case here, in the start's
/// words.
///
/// A run that names a picks file ([`CLASSIC_PICKS`]) has its classic picks
/// written from here on: the gateway is told the function that writes them.
#[expect(
    clippy::panic,
    reason = "test helper - panicking on failure is intentional"
)]
pub(crate) fn parts(
    config: RouterConfig,
    workers: &Arc<WorkerRegistry>,
    classic: impl FnOnce(&RouterConfig) -> PolicyRegistry,
) -> (AppContextBuilder, Arc<PolicyRegistry>, RouterConfig) {
    if picks().is_some() {
        PolicyRegistry::on_classic_pick(write_classic_pick);
    }
    if !on() {
        let policies = Arc::new(classic(&config));
        return (AppContext::builder(), policies, config);
    }
    let config = RouterConfig {
        policy_controller: true,
        policy_controller_address: None,
        ..config
    };
    match AppContextBuilder::with_policies_started(&config, Arc::clone(workers)) {
        Ok((builder, policies)) => (builder, policies, config),
        Err(refusal) => {
            panic!("with {SWITCH}=on the gateway's start refuses this configuration: {refusal}")
        }
    }
}
