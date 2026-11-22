//! The picks a classic policy makes in a gateway that holds the policy
//! engine, written to a file for the test run with the policy controller on.
//!
//! With `--policy-controller` and no controller's address the policy engine
//! is to place every request. Until it does, the policy registry of such a
//! gateway still hands out a classic policy for some places (a side of a
//! prefill/decode pair, a policy the engine does not pick by itself yet),
//! and the placement asks it. The test run that turns the policy controller
//! on has to see each such pick: it names a file in `SMG_TEST_CLASSIC_PICKS`,
//! every such pick appends one line to it, and
//! `scripts/ci_rust_tests.sh policy-controller` holds the lines to the list
//! of the cases known to be placed so
//! (`model_gateway/tests/policy_controller_known.tsv`).
//!
//! Built under the `test-util` feature alone, which only a test build turns
//! on: a product build holds none of this.

use std::{env, fs::OpenOptions, io::Write, path::PathBuf, sync::OnceLock, thread};

/// The environment variable that names the file the picks are written to.
const FILE: &str = "SMG_TEST_CLASSIC_PICKS";

/// Where this process writes its picks, and the name it writes them under.
struct Record {
    file: PathBuf,
    /// The running executable's file name: a test binary's is its target's
    /// name and a hash.
    executable: String,
}

/// The record of this process: read once, `None` where the variable is unset.
fn record() -> Option<&'static Record> {
    static RECORD: OnceLock<Option<Record>> = OnceLock::new();
    RECORD
        .get_or_init(|| {
            let file = PathBuf::from(env::var_os(FILE)?);
            let executable = env::current_exe()
                .ok()
                .and_then(|path| Some(path.file_name()?.to_string_lossy().into_owned()))
                .unwrap_or_default();
            Some(Record { file, executable })
        })
        .as_ref()
}

/// Appends the line of one pick by the classic policy named `policy`: the
/// executable, the thread's name and the policy's name, tab separated. A
/// test's thread carries the test's name, so the line names the case whose
/// request the policy placed. Nothing where the variable is unset.
///
/// A line that cannot be written fails the case that picked: the run that
/// reads the file would otherwise pass on a pick it never saw.
#[expect(
    clippy::panic,
    reason = "a test build's record: a pick it cannot write down fails its case"
)]
pub(super) fn note(policy: &str) {
    let Some(record) = record() else {
        return;
    };
    let line = format!(
        "{}\t{}\t{policy}\n",
        record.executable,
        thread::current().name().unwrap_or("?")
    );
    // One write a line, to a file opened for append: the lines of threads
    // that pick at once stay whole.
    let written = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&record.file)
        .and_then(|mut file| file.write_all(line.as_bytes()));
    if let Err(error) = written {
        panic!(
            "{FILE} names {}, which cannot be written: {error}",
            record.file.display()
        );
    }
}
