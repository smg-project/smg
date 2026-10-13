//! SMG's media pipeline against bellwether's multimodal suite, image by image.
//!
//! bellwether (smg-project/bellwether) keeps a corpus of public-benchmark images in tiers
//! and, per processor family, what the reference implementation's own image path gives for
//! each image: the decoded RGB bytes, the u8 patches, the bf16 and the f32 pixel tensors,
//! each as a SHA-256. The suite's `verify` command runs SMG's dump example
//! (`examples/bellwether_dump.rs`) over a tier for every family, compares each image with
//! the family's reference, counts the differences the suite lists as known apart, and exits
//! 0 only when every image matches or is known. This test builds the example with the cargo
//! that runs it, runs `verify` on the `corpus-50` tier through `uv`, as the bellwether
//! workflow runs the suite's commands, prints the suite's output, and fails on a non-zero
//! exit or on a family whose images are not all matched or known.
//!
//! The run is opt-in: `BELLWETHER_MULTIMODAL` points at a bellwether checkout, or at its
//! `multimodal/` directory, with the tier fetched (`bellwether multimodal fetch --tier
//! corpus-50`: the images and the reference sets live in Git LFS and a clone holds
//! pointers); without it the test prints a skip notice and passes. A tier that is not
//! fetched fails the run and names the fetch command, so it cannot pass on pointers.
//! `BELLWETHER_MULTIMODAL_SOURCE`, when set, is passed to `verify --source`: a source name
//! from the suite's README (unset, the suite compares with its default, the reference
//! implementation's own image path). `uv` must be on the path. The example is built in this
//! test's profile, so `cargo test --release` runs the pipeline at its release speed; the
//! suite's report, the dumps and its logs stay under `target/tmp/bellwether_multimodal/`.

use std::{
    env,
    ffi::OsStr,
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
    process::{Command, Output},
    time::Instant,
};

use serde_json::Value;

const SUITE_ENV: &str = "BELLWETHER_MULTIMODAL";
const SOURCE_ENV: &str = "BELLWETHER_MULTIMODAL_SOURCE";
const TIER: &str = "corpus-50";
const EXAMPLE: &str = "bellwether_dump";
const LFS_POINTER: &[u8] = b"version https://git-lfs.github.com/spec/v1";

/// The suite's two directories: the checkout `uv` runs the project from, and the suite's
/// root `verify --root` reads.
struct Suite {
    checkout: PathBuf,
    root: PathBuf,
}

impl Suite {
    fn locate(given: &Path) -> Result<Self, String> {
        let given =
            fs::canonicalize(given).map_err(|e| format!("{SUITE_ENV}={}: {e}", given.display()))?;
        let nested = given.join("multimodal");
        if given.join("pyproject.toml").is_file() && nested.join("manifest.tsv").is_file() {
            return Ok(Self {
                checkout: given,
                root: nested,
            });
        }
        if given.join("manifest.tsv").is_file() {
            if let Some(parent) = given
                .parent()
                .filter(|p| p.join("pyproject.toml").is_file())
            {
                return Ok(Self {
                    checkout: parent.to_path_buf(),
                    root: given.clone(),
                });
            }
        }
        Err(format!(
            "{SUITE_ENV}={}: neither a bellwether checkout (pyproject.toml and multimodal/manifest.tsv) nor its \
             multimodal directory",
            given.display()
        ))
    }

    /// The tier's images that are not there or are still Git LFS pointers.
    fn unfetched(&self, tier: &str) -> Result<Vec<String>, String> {
        let list = self.root.join("tiers").join(format!("{tier}.txt"));
        let text = fs::read_to_string(&list).map_err(|e| format!("{}: {e}", list.display()))?;
        let mut unfetched = Vec::new();
        for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
            if !is_content(&self.root.join(line)) {
                unfetched.push(line.to_string());
            }
        }
        Ok(unfetched)
    }
}

/// Whether the file is there and is not a Git LFS pointer.
fn is_content(path: &Path) -> bool {
    let Ok(mut file) = File::open(path) else {
        return false;
    };
    let mut head = [0u8; 64];
    let Ok(read) = file.read(&mut head) else {
        return false;
    };
    read > 0 && !head[..read].starts_with(LFS_POINTER)
}

/// The variables cargo sets for a test process (`CARGO_PKG_*`, the manifest and crate names,
/// `OUT_DIR`) are not passed on to a cargo this test runs: build scripts list them under
/// `rerun-if-env-changed`, so a cargo that inherits them takes every artifact of a build from
/// a shell as stale, and the other way round, and the two rebuild the dependency tree in turns.
fn without_test_environment(command: &mut Command) {
    for (name, _) in env::vars_os() {
        let text = name.to_string_lossy();
        let per_test = text.starts_with("CARGO_PKG_")
            || matches!(
                text.as_ref(),
                "CARGO_MANIFEST_DIR"
                    | "CARGO_MANIFEST_PATH"
                    | "CARGO_CRATE_NAME"
                    | "CARGO_BIN_NAME"
                    | "CARGO_PRIMARY_PACKAGE"
                    | "OUT_DIR"
            );
        if per_test {
            command.env_remove(&name);
        }
    }
}

/// Build the dump example with the cargo that runs this test, in this test's profile, and
/// take its path from cargo's own artifact message.
fn build_example() -> Result<PathBuf, String> {
    let cargo = env!("CARGO");
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let mut command = Command::new(cargo);
    without_test_environment(&mut command);
    command
        .args(["build", "--example", EXAMPLE, "--manifest-path"])
        .arg(&manifest)
        .arg("--message-format=json-render-diagnostics");
    if !cfg!(debug_assertions) {
        command.arg("--release");
    }
    let output = command.output().map_err(|e| format!("{cargo}: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "`cargo build --example {EXAMPLE}` ended with {}:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Ok(message) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let is_example = message["target"]["kind"]
            .as_array()
            .is_some_and(|kinds| kinds.iter().any(|kind| kind == "example"));
        if message["reason"] == "compiler-artifact"
            && is_example
            && message["target"]["name"] == EXAMPLE
        {
            if let Some(executable) = message["executable"].as_str() {
                return Ok(PathBuf::from(executable));
            }
        }
    }
    Err(format!(
        "cargo built the example but its messages name no executable for {EXAMPLE}"
    ))
}

/// `bellwether multimodal verify` over the tier with the example, through `uv` from the
/// checkout, the way the bellwether workflow runs the suite's commands.
fn verify(
    suite: &Suite,
    dump: &Path,
    source: Option<&OsStr>,
    work: &Path,
    report: &Path,
) -> Result<Output, String> {
    let mut command = Command::new("uv");
    command
        .args(["run", "--locked", "--project"])
        .arg(&suite.checkout)
        .args(["bellwether", "multimodal", "verify", "--root"])
        .arg(&suite.root)
        .args(["--tier", TIER, "--smg-dump"])
        .arg(dump)
        .args(["--strict-missing", "--show", "60", "--work"])
        .arg(work)
        .arg("--report")
        .arg(report)
        .current_dir(&suite.checkout);
    if let Some(source) = source {
        command.arg("--source").arg(source);
    }
    command.output().map_err(|e| {
        format!("uv: {e}; uv runs the suite's commands, as the bellwether workflow does")
    })
}

/// A family's totals from the suite's report.
struct Totals {
    images: u64,
    matched: u64,
    known: u64,
    difference: u64,
    error: u64,
    missing: u64,
}

impl Totals {
    fn of(family: &Value) -> Self {
        let count = |key: &str| family["total"][key].as_u64().unwrap_or(0);
        Self {
            images: count("images"),
            matched: count("match"),
            known: count("known"),
            difference: count("difference"),
            error: count("error"),
            missing: count("missing") + count("no_reference"),
        }
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and the suite's output are test diagnostic output"
)]
fn corpus_50_matches_the_reference() {
    let Some(given) = env::var_os(SUITE_ENV).map(PathBuf::from) else {
        eprintln!(
            "skipping: {SUITE_ENV} is not set; point it at a bellwether checkout with `bellwether multimodal fetch \
             --tier {TIER}` done"
        );
        return;
    };
    let suite = Suite::locate(&given).unwrap_or_else(|e| panic!("{e}"));
    let unfetched = suite.unfetched(TIER).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        unfetched.is_empty(),
        "{} of the {TIER} images under {} are missing or Git LFS pointers (the first: {}); run `bellwether \
         multimodal fetch --tier {TIER}` in the checkout",
        unfetched.len(),
        suite.root.display(),
        unfetched[0]
    );
    let dump = build_example().unwrap_or_else(|e| panic!("{e}"));
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("bellwether_multimodal");
    fs::create_dir_all(&out).unwrap_or_else(|e| panic!("{}: {e}", out.display()));
    let report_path = out.join("report.json");
    // A report of an earlier run must not stand in for one this run did not write.
    if let Err(e) = fs::remove_file(&report_path) {
        assert!(
            e.kind() == io::ErrorKind::NotFound,
            "{}: {e}",
            report_path.display()
        );
    }
    let source = env::var_os(SOURCE_ENV);
    let started = Instant::now();
    let output = verify(
        &suite,
        &dump,
        source.as_deref(),
        &out.join("work"),
        &report_path,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let elapsed = started.elapsed();
    eprintln!(
        "bellwether multimodal verify --tier {TIER}: {} in {:.1} s\n{}{}",
        output.status,
        elapsed.as_secs_f64(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = fs::read_to_string(&report_path)
        .map_err(|e| e.to_string())
        .and_then(|text| serde_json::from_str(&text).map_err(|e| e.to_string()))
        .unwrap_or_else(|e| {
            panic!(
                "{}: {e}; the suite's output is above",
                report_path.display()
            )
        });
    let families = report["families"]
        .as_object()
        .filter(|families| !families.is_empty())
        .unwrap_or_else(|| panic!("{}: no family in the report", report_path.display()));
    let mut short = Vec::new();
    for (name, family) in families {
        let totals = Totals::of(family);
        let verdict = if totals.images == totals.matched + totals.known {
            "ok"
        } else {
            short.push(name.clone());
            "SHORT"
        };
        eprintln!(
            "{name}: {} images: match {}, known {}, difference {}, error {}, missing {}: {verdict}",
            totals.images,
            totals.matched,
            totals.known,
            totals.difference,
            totals.error,
            totals.missing
        );
    }
    assert!(
        output.status.success() && short.is_empty(),
        "verify ended with {} and {} of {} families fall short of every image matched or known ({}); the report is \
         {}",
        output.status,
        short.len(),
        families.len(),
        short.join(", "),
        report_path.display()
    );
}
