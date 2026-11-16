//! The policy spec's file source (design 4.6): the document is a file, read
//! at the start.
//!
//! The start reads the file as it stands, through every link on its path (a
//! mounted ConfigMap's file is a link), and applies its document as the
//! source's first ([`Generations`]). A file is read whole or not at all: its
//! identity (where the path leads, its size and times, on Unix its inode)
//! is looked at before and after the read, and a file that moved under the
//! read is read again. A file that cannot be read, a file over the bound
//! (refused by its size, unread) and a document that does not run each
//! refuse the start.
//!
//! The file is not read again after the start.

use std::{
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    time::SystemTime,
};

use super::generations::{Applied, Format, Generations, SpecRejected, MAX_DOCUMENT_BYTES};

/// How many times a start reads a file that moves under its read before it
/// refuses.
const START_READS: usize = 3;

/// What one look at the path found.
#[derive(Clone, Debug, Eq, PartialEq)]
enum Seen {
    /// A file, by its identity.
    File(Identity),
    /// No file a look could read the identity of, by the error's text.
    Unreadable(String),
}

/// A file's identity: two looks that read the same one saw the same bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Identity {
    /// Where the path leads once every link is followed.
    resolved: PathBuf,
    len: u64,
    modified: Option<SystemTime>,
    #[cfg(unix)]
    inode: Inode,
}

/// The file's inode, with the time of its last change.
#[cfg(unix)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Inode {
    device: u64,
    number: u64,
    changed: (i64, i64),
}

fn look(path: &Path) -> Seen {
    match identity(path) {
        Ok(identity) => Seen::File(identity),
        Err(error) => Seen::Unreadable(error.to_string()),
    }
}

fn identity(path: &Path) -> io::Result<Identity> {
    let resolved = fs::canonicalize(path)?;
    let metadata = fs::metadata(&resolved)?;
    Ok(Identity {
        len: metadata.len(),
        modified: metadata.modified().ok(),
        #[cfg(unix)]
        inode: {
            use std::os::unix::fs::MetadataExt;
            Inode {
                device: metadata.dev(),
                number: metadata.ino(),
                changed: (metadata.ctime(), metadata.ctime_nsec()),
            }
        },
        resolved,
    })
}

/// The file's bytes, whole. A file over the bound is refused by its size,
/// unread.
fn read(identity: &Identity) -> Result<Vec<u8>, SpecRejected> {
    if identity.len > MAX_DOCUMENT_BYTES {
        return Err(SpecRejected::TooLarge(identity.len));
    }
    let unreadable = |error: io::Error| SpecRejected::Unreadable(error.to_string());
    let file = fs::File::open(&identity.resolved).map_err(unreadable)?;
    let mut bytes = Vec::with_capacity(usize::try_from(identity.len).unwrap_or(0));
    // One byte past the bound tells a file that grew since the look: the
    // apply refuses it by its size.
    file.take(MAX_DOCUMENT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(unreadable)?;
    Ok(bytes)
}

/// The start: reads the file at `path` as it stands and applies its
/// document as the source's first. A file that cannot be read, or a
/// document that does not run, refuses the start.
pub(crate) fn start(path: &Path, generations: &Generations) -> Result<Applied, SpecRejected> {
    start_looking(path, generations, look)
}

/// [`start`] with the look as its argument: the identity is the file
/// system's word, and a case hands in a file that moves under its read.
fn start_looking(
    path: &Path,
    generations: &Generations,
    mut look: impl FnMut(&Path) -> Seen,
) -> Result<Applied, SpecRejected> {
    let format = Format::of(path);
    // A file that moves under its read is read again; one that never rests
    // is no document.
    for _ in 0..START_READS {
        let found = look(path);
        let document = match &found {
            Seen::File(identity) => read(identity)?,
            Seen::Unreadable(error) => return Err(SpecRejected::Unreadable(error.clone())),
        };
        if look(path) == found {
            return generations.apply(document, format);
        }
    }
    Err(SpecRejected::Unreadable(format!(
        "the file moved under each of {START_READS} reads"
    )))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempfile::TempDir;

    use super::*;
    use crate::{
        controller::generations::tests::{with_line, LEAST_LOAD, RANDOM},
        spec::SpecError,
        store::MemoryStore,
    };

    /// A directory with `document` as its `policy.yaml`, and the generations
    /// of a store of its own.
    fn bed(document: &[u8]) -> (TempDir, PathBuf, Generations) {
        let dir = tempfile::tempdir().expect("a directory");
        let path = dir.path().join("policy.yaml");
        fs::write(&path, document).expect("the file");
        let generations = Generations::new(Arc::new(MemoryStore::new()));
        (dir, path, generations)
    }

    /// The bytes the read call answers, with the number and the format.
    fn answered(generations: &Generations) -> (u64, String, Vec<u8>) {
        let answer = generations.document(0).expect("the standing document");
        (answer.generation, answer.format, answer.document)
    }

    /// The start reads the file as it stands: a document with no line is
    /// generation 1 and a written line is the number, in the form the
    /// file's name says, through every link on the path.
    #[test]
    fn a_start_applies_the_file_as_it_stands() {
        let (_dir, path, generations) = bed(LEAST_LOAD.as_bytes());
        assert_eq!(start(&path, &generations), Ok(Applied::Generation(1)));
        assert_eq!(
            answered(&generations),
            (1, "yaml".to_owned(), LEAST_LOAD.into())
        );

        let json = br#"{"apiVersion":"smg.policy/v1","generation":4,"profiles":{"default":{"pick":"random"}}}"#;
        let (dir, _, generations) = bed(b"");
        let path = dir.path().join("policy.json");
        fs::write(&path, json).expect("the file");
        assert_eq!(start(&path, &generations), Ok(Applied::Generation(4)));
        assert_eq!(
            answered(&generations),
            (4, "json".to_owned(), json.to_vec())
        );
    }

    /// A mounted ConfigMap's file is a link through `..data` into a
    /// version's directory: the start follows it.
    #[cfg(unix)]
    #[test]
    fn a_start_follows_the_links_of_a_mounted_file() {
        use std::os::unix::fs::symlink;

        let (dir, _, generations) = bed(b"");
        let mount = dir.path().join("mount");
        fs::create_dir_all(mount.join("..version_1")).expect("the version's directory");
        fs::write(mount.join("..version_1/policy.yaml"), with_line(RANDOM, 6)).expect("the file");
        symlink("..version_1", mount.join("..data")).expect("the data link");
        symlink("..data/policy.yaml", mount.join("policy.yaml")).expect("the file's link");

        let started = start(&mount.join("policy.yaml"), &generations);
        assert_eq!(started, Ok(Applied::Generation(6)));
        assert_eq!(answered(&generations).2, with_line(RANDOM, 6));
    }

    /// A file that is not there, a directory, a file over the bound (by its
    /// size, unread) and a document the validator refuses each refuse the
    /// start by name, and nothing is applied.
    #[test]
    fn a_start_refuses_a_file_it_cannot_run() {
        let (dir, path, generations) = bed(b"");
        fs::remove_file(&path).expect("the file goes");
        match start(&path, &generations) {
            Err(SpecRejected::Unreadable(error)) => {
                assert!(error.contains("No such file"), "{error}");
            }
            other => panic!("started as {other:?}"),
        }
        assert!(matches!(
            start(dir.path(), &generations),
            Err(SpecRejected::Unreadable(_))
        ));

        // Five bytes over: the refusal names the file's size, which no read
        // gave it (a read stops one byte past the bound).
        let over = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits") + 5;
        fs::write(&path, vec![b'#'; over]).expect("the file");
        assert_eq!(
            start(&path, &generations),
            Err(SpecRejected::TooLarge(MAX_DOCUMENT_BYTES + 5))
        );

        fs::write(&path, LEAST_LOAD.replace("[healthy]", "[nowhere]")).expect("the file");
        let unknown_filter = SpecError::UnknownFilter {
            profile: "default".to_owned(),
            name: "nowhere".to_owned(),
        };
        assert_eq!(
            start(&path, &generations),
            Err(SpecRejected::Invalid(unknown_filter))
        );
        assert!(!generations.bound());
        assert_eq!(generations.generation(), 0);
    }

    /// A file is read whole or not at all: one that moved under its read is
    /// read again, and the bytes of the read it stood still under are the
    /// document; one that moves under every read refuses the start.
    #[test]
    fn a_file_that_moves_under_its_read_is_read_again() {
        let (_dir, path, generations) = bed(LEAST_LOAD.as_bytes());
        let mut looks = 0;
        let started = start_looking(&path, &generations, |path| {
            looks += 1;
            if looks == 2 {
                // After the first read: a writer replaces the document.
                fs::write(path, RANDOM).expect("the file");
            }
            look(path)
        });
        assert_eq!(started, Ok(Applied::Generation(1)));
        assert_eq!(looks, 4, "two reads, each between two looks");
        assert_eq!(answered(&generations).2, RANDOM.as_bytes());

        let (_dir, path, generations) = bed(LEAST_LOAD.as_bytes());
        let mut looks = 0;
        let restless = start_looking(&path, &generations, |path| {
            looks += 1;
            if looks % 2 == 0 {
                let document = if looks % 4 == 0 { LEAST_LOAD } else { RANDOM };
                fs::write(path, document).expect("the file");
            }
            look(path)
        });
        let refused = SpecRejected::Unreadable("the file moved under each of 3 reads".to_owned());
        assert_eq!(restless, Err(refused));
        assert!(!generations.bound());
    }
}
