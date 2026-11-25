//! Documents and handles shared by the spec source's cases.

use std::{fmt::Write as _, io, sync::Arc};
#[cfg(unix)]
use std::{path::Path, process::Command};

use parking_lot::Mutex;

use crate::{
    document::spec::{AppliedSpec, Digest, PolicySpec, SpecError},
    stages::StageRegistry,
    store::MemoryStore,
};

/// A document in the form of the design's example: no `generation` line.
pub(crate) const LEAST_LOAD: &str = "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    score: [least-load]\n    pick: best\n";
/// Another one.
pub(crate) const RANDOM: &str =
    "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    pick: random\n";

/// `document` under a written line.
pub(crate) fn with_line(document: &str, line: u64) -> Vec<u8> {
    format!("generation: {line}\n{document}").into_bytes()
}

/// `document` grown to exactly `len` bytes by one comment line at its
/// end.
pub(crate) fn padded(document: &str, len: usize) -> Vec<u8> {
    let mut bytes = document.as_bytes().to_vec();
    assert!(len > bytes.len() + 1, "no room for a comment line");
    bytes.push(b'#');
    bytes.resize(len - 1, b'x');
    bytes.push(b'\n');
    bytes
}

/// A document whose view's first header is one text of `text` bytes
/// under an anchor, with `aliases` aliases of it after it: four bytes
/// an alias in the file, the text's bytes an alias in what it reads as.
pub(crate) fn long_text_under_aliases(text: usize, aliases: usize) -> String {
    let mut document = String::with_capacity(text + 4 * aliases + 256);
    document.push_str(LEAST_LOAD);
    document.push_str("view:\n  headers: [&h ");
    document.extend(std::iter::repeat_n('x', text));
    for _ in 0..aliases {
        document.push_str(",*h ");
    }
    document.push_str("]\n");
    document
}

/// The stock build's stages.
pub(crate) fn stock() -> Arc<StageRegistry> {
    Arc::new(StageRegistry::builtin())
}

/// A validator that panics on a document that picks at random, and
/// validates any other.
pub(crate) fn faulting(
    document: &PolicySpec,
    digest: Digest,
    catalog: &StageRegistry,
    denied_headers: &[&str],
) -> Result<AppliedSpec, SpecError> {
    let random = document
        .profiles
        .values()
        .any(|profile| profile.pick == "random".into());
    assert!(!random, "a fault of the validator");
    AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
}

/// The number and the digest the store runs.
pub(crate) fn standing(store: &MemoryStore) -> (u64, Option<Digest>) {
    let world = store.snapshot();
    (world.spec().generation(), world.spec().digest())
}

/// `LEAST_LOAD` with one binding of its own, by `n`: a document apart
/// from every other `n`'s.
pub(crate) fn bound_to(n: u64) -> String {
    format!("{LEAST_LOAD}bindings:\n  - match: {{ model: \"model-{n}\" }}\n    profile: default\n")
}

/// A pipe at `path`, with no writer. The program makes it: the linked
/// system-call crate has no call for one on every system the crate
/// builds on.
#[cfg(unix)]
pub(crate) fn pipe_at(path: &Path) {
    let made = Command::new("mkfifo")
        .arg(path)
        .status()
        .expect("mkfifo runs");
    assert!(made.success(), "mkfifo: {made}");
}

/// The lines a case's own subscriber wrote.
#[derive(Clone, Default)]
pub(crate) struct Logged(pub(crate) Arc<Mutex<Vec<u8>>>);

impl io::Write for Logged {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A document of about `bytes` bytes: one profile, and bindings of one
/// exact model id each until the size is reached.
pub(crate) fn document_of(bytes: usize) -> (Vec<u8>, u32) {
    let mut text = format!("{LEAST_LOAD}bindings:\n");
    let mut bindings = 0;
    while text.len() < bytes {
        let _ = write!(
            text,
            "  - match: {{ model: \"tenant-{bindings:06}/model\" }}\n    profile: default\n"
        );
        bindings += 1;
    }
    (text.into_bytes(), bindings)
}
