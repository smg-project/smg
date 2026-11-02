//! `base`: the base revision's gateway binary for an A/B. A detached worktree of the
//! revision is built with the SAME cargo line as the head (`cargo build --release -p smg
//! --bin smg`: the gateway alone, so the shared dependencies get the same feature set on
//! both sides; building it together with other packages changes the binary through
//! feature unification) into a target directory of its own (default
//! `target/gateway-perf/base-target`: a full release build, slower than sharing the
//! checkout's target directory but never served another tree's artifacts), and the
//! binary is copied to the output path with the commit beside it.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{bail, Context, Result};

/// A detached worktree that is removed when dropped, however the build ended.
struct Worktree {
    repo: PathBuf,
    dir: PathBuf,
}

impl Drop for Worktree {
    fn drop(&mut self) {
        let _ = Command::new("git")
            .args(["-C"])
            .arg(&self.repo)
            .args(["worktree", "remove", "--force"])
            .arg(&self.dir)
            .output();
    }
}

fn git(repo: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .context("running git")?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn absolute(repo: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo.join(path)
    }
}

/// Builds `rev`'s gateway into `out` (`<out>.commit` beside it).
pub fn build_base(repo: &Path, rev: &str, out: &Path, target: Option<&Path>) -> Result<()> {
    let out = absolute(repo, out);
    let target = absolute(
        repo,
        target.unwrap_or_else(|| Path::new("target/gateway-perf/base-target")),
    );
    let src = repo.join("target").join("gateway-perf").join("base-src");
    let commit = git(
        repo,
        &["rev-parse", "--verify", &format!("{rev}^{{commit}}")],
    )?;
    // `cargo clean` deletes the worktree's directory and leaves its record: without the
    // prune, `worktree add` refuses the registered path.
    git(repo, &["worktree", "prune"])?;
    if src.exists() {
        git(
            repo,
            &["worktree", "remove", "--force", &src.display().to_string()],
        )?;
    }
    git(
        repo,
        &[
            "worktree",
            "add",
            "--detach",
            &src.display().to_string(),
            &commit,
        ],
    )?;
    let worktree = Worktree {
        repo: repo.to_path_buf(),
        dir: src.clone(),
    };
    println!(
        "base: {commit} ({rev}) -> {} (target {})",
        out.display(),
        target.display()
    );
    let status = Command::new("cargo")
        .args(["build", "--release", "-p", "smg", "--bin", "smg"])
        .current_dir(&worktree.dir)
        .env("CARGO_TARGET_DIR", &target)
        .status()
        .context("running cargo build")?;
    if !status.success() {
        bail!("cargo build of the base failed ({status})");
    }
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(target.join("release").join("smg"), &out)?;
    fs::write(
        PathBuf::from(format!("{}.commit", out.display())),
        format!("{commit}\n"),
    )?;
    let size = fs::metadata(&out)?.len();
    println!("base: done ({size} bytes)");
    drop(worktree);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_resolve_against_the_repository() {
        assert_eq!(
            absolute(Path::new("/r"), Path::new("x/y")),
            PathBuf::from("/r/x/y")
        );
        assert_eq!(
            absolute(Path::new("/r"), Path::new("/x")),
            PathBuf::from("/x")
        );
    }

    #[test]
    fn a_missing_revision_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = build_base(dir.path(), "nosuch", Path::new("out/smg"), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("git rev-parse"), "{err}");
    }
}
