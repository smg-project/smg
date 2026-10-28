use std::{path::PathBuf, process::Command};

const DEFAULT_VERSION: &str = "0.0.0";
const DEFAULT_PROJECT_NAME: &str = "smg";
/// Set (non-empty) by a pipeline that builds consecutive commits in one warm
/// target directory (a local or self-hosted one; the repository's CI checks out
/// fresh): the baked version then follows the checked-out commit.
const VERSION_FOLLOWS_HEAD: &str = "SMG_VERSION_FOLLOWS_HEAD";

/// Set a compile-time environment variable with the SMG_ prefix
macro_rules! set_env {
    ($name:expr, $value:expr) => {
        println!("cargo:rustc-env=SMG_{}={}", $name, $value);
    };
}

fn main() {
    // Rebuild triggers: the manifest, and, when a pipeline asks for it, the git
    // files whose change means the checkout may be at another commit. Cargo
    // re-runs a build script only when a declared trigger changes; with the
    // manifest alone the git values below are cached for the life of a target
    // directory, and every later build in it carries the first commit's label.
    // A re-run recompiles this crate, which bakes the values with `env!`, so the
    // git triggers are opt-in: a local or self-hosted pipeline that keeps its
    // target between commits sets `SMG_VERSION_FOLLOWS_HEAD` (the repository's
    // CI jobs check out fresh and restore no workspace-crate artifacts, so the
    // manifest trigger already covers them); a developer's checkout keeps the
    // one-bake behaviour, where a `git stash`, a bare `git reset --hard` or a
    // checkout of the current branch (each moves the HEAD reflog) would
    // otherwise recompile the gateway's lib, bin and test targets.
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-env-changed={VERSION_FOLLOWS_HEAD}");
    if std::env::var_os(VERSION_FOLLOWS_HEAD).is_some_and(|value| !value.is_empty()) {
        for file in git_state_files() {
            println!("cargo:rerun-if-changed={}", file.display());
        }
    }

    // Set version info environment variables
    let version = read_cargo_version().unwrap_or_else(|_| DEFAULT_VERSION.to_string());
    let target = std::env::var("TARGET").unwrap_or_else(|_| get_rustc_host().unwrap_or_default());
    let profile = std::env::var("PROFILE").unwrap_or_default();

    set_env!("PROJECT_NAME", DEFAULT_PROJECT_NAME);
    set_env!("VERSION", version);
    set_env!(
        "BUILD_TIME",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    );
    set_env!(
        "BUILD_MODE",
        if profile == "release" {
            "release"
        } else {
            "debug"
        }
    );
    set_env!("TARGET_TRIPLE", target);
    set_env!(
        "GIT_BRANCH",
        git_branch().unwrap_or_else(|| "unknown".into())
    );
    set_env!(
        "GIT_COMMIT",
        git_commit().unwrap_or_else(|| "unknown".into())
    );
    set_env!(
        "GIT_STATUS",
        git_status().unwrap_or_else(|| "unknown".into())
    );
    set_env!(
        "RUSTC_VERSION",
        rustc_version().unwrap_or_else(|| "unknown".into())
    );
    set_env!(
        "CARGO_VERSION",
        cargo_version().unwrap_or_else(|| "unknown".into())
    );
}

/// The files that change when HEAD comes to name another commit (declared
/// under `SMG_VERSION_FOLLOWS_HEAD` only: the HEAD reflog also moves on a
/// `git stash`, a bare `git reset --hard` and a checkout of the current
/// branch, each a gateway recompile for no commit change), resolved
/// with `git rev-parse --git-path` so a worktree's `.git` file and its common
/// directory are handled: `HEAD` (a detached checkout, a branch switch), the
/// branch's loose ref file (a commit on it) or `packed-refs` while the ref is
/// packed, the HEAD reflog (every move of HEAD, which also covers a packed
/// ref that becomes loose again on the next commit), and the reftable table
/// list on that backend. Only files that exist are returned: cargo treats a
/// missing trigger as always stale and would re-run the script on every
/// build. The index and the working tree are not watched (`git status`
/// rewrites the index, which would re-run the script on every build too), so
/// the dirty flag is as fresh as the last re-run. Outside a git checkout (a
/// crate tarball, a source archive) the list is empty and the values fall
/// back as before.
fn git_state_files() -> Vec<PathBuf> {
    let mut files = Vec::new();
    let Some(head) = git_path("HEAD") else {
        return files;
    };
    files.push(head);
    if let Some(reference) = run_cmd("git", &["symbolic-ref", "-q", "HEAD"]) {
        match git_path(&reference) {
            Some(loose) => files.push(loose),
            None => files.extend(git_path("packed-refs")),
        }
    }
    files.extend(git_path("logs/HEAD"));
    files.extend(git_path("reftable/tables.list"));
    files
}

/// `git rev-parse --git-path <name>` as an absolute path, when the file exists.
fn git_path(name: &str) -> Option<PathBuf> {
    let path = run_cmd("git", &["rev-parse", "--git-path", name])?;
    std::fs::canonicalize(path).ok()
}

fn read_cargo_version() -> Result<String, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string("Cargo.toml")?;
    let toml: toml::Value = toml::from_str(&content)?;
    toml.get("package")
        .and_then(|p| p.get("version"))
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or_else(|| "Missing version in Cargo.toml".into())
}

fn run_cmd(cmd: &str, args: &[&str]) -> Option<String> {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}

fn git_branch() -> Option<String> {
    run_cmd("git", &["rev-parse", "--abbrev-ref", "HEAD"])
}

fn git_commit() -> Option<String> {
    run_cmd("git", &["rev-parse", "--short", "HEAD"])
}

fn git_status() -> Option<String> {
    run_cmd("git", &["status", "--porcelain"])
        .map(|s| if s.is_empty() { "clean" } else { "dirty" }.into())
}

fn rustc_version() -> Option<String> {
    run_cmd("rustc", &["--version"])
}

fn cargo_version() -> Option<String> {
    run_cmd("cargo", &["--version"])
}

fn get_rustc_host() -> Option<String> {
    run_cmd("rustc", &["-vV"])?
        .lines()
        .find(|l| l.starts_with("host: "))
        .and_then(|l| l.strip_prefix("host: "))
        .map(|s| s.trim().to_string())
}
