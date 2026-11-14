//! A typed document the control plane holds in force: the live revision
//! behind one swap, read without a lock, replaced by one writer at a time
//! as the next revision. The ETag is the SHA-256 of the document as JSON,
//! quoted and strong, so a writer that read revision 3 writes against
//! revision 3 and no other.

use std::sync::Arc;

use arc_swap::ArcSwap;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, MutexGuard};
use serde::Serialize;
use sha2::{Digest, Sha256};

/// The principal recorded on the document the start read from its file.
pub const STARTUP_PRINCIPAL: &str = "startup";

/// A revision of a document: the document and the record of its
/// application.
#[derive(Debug, Clone, Serialize)]
pub struct Applied<T> {
    /// The document in force at this revision.
    pub document: T,
    /// 1 for the document the start read, one more per application.
    pub revision: u64,
    /// The quoted SHA-256 of `document` as JSON: the `ETag` of every answer
    /// and the `If-Match` a write carries.
    pub etag: String,
    /// When the revision was put in force.
    pub applied_at: DateTime<Utc>,
    /// Who put the revision in force: a control-plane principal, or
    /// [`STARTUP_PRINCIPAL`].
    pub principal: String,
}

/// The holder of one document: the revision in force and the writer's turn.
pub struct AdminDocument<T> {
    live: ArcSwap<Applied<T>>,
    /// One writer at a time: the check of the ETag read, the application to
    /// the component the document drives and the swap happen in one turn.
    writer: Mutex<()>,
}

impl<T: Serialize> AdminDocument<T> {
    /// `document` as revision 1 under [`STARTUP_PRINCIPAL`].
    pub fn from_startup(document: T) -> Result<Self, serde_json::Error> {
        let etag = etag(&document)?;
        Ok(Self {
            live: ArcSwap::from_pointee(Applied {
                document,
                revision: 1,
                etag,
                applied_at: Utc::now(),
                principal: STARTUP_PRINCIPAL.to_string(),
            }),
            writer: Mutex::new(()),
        })
    }

    /// The revision in force.
    pub fn load(&self) -> Arc<Applied<T>> {
        self.live.load_full()
    }

    /// The writer's turn, held until the guard drops: no other writer gets
    /// between the read of the live revision and the swap.
    pub fn write(&self) -> Writer<'_, T> {
        Writer {
            holder: self,
            _turn: self.writer.lock(),
        }
    }
}

/// One writer's turn on an [`AdminDocument`].
pub struct Writer<'a, T> {
    holder: &'a AdminDocument<T>,
    _turn: MutexGuard<'a, ()>,
}

impl<T: Serialize> Writer<'_, T> {
    /// The revision in force at this point of the turn.
    pub fn live(&self) -> Arc<Applied<T>> {
        self.holder.load()
    }

    /// `document`, whose ETag the caller computed with [`etag`], in force as
    /// the next revision under `principal`.
    pub fn apply(&self, document: T, etag: String, principal: &str) -> Arc<Applied<T>> {
        let applied = Arc::new(Applied {
            document,
            revision: self.live().revision + 1,
            etag,
            applied_at: Utc::now(),
            principal: principal.to_string(),
        });
        self.holder.live.store(Arc::clone(&applied));
        applied
    }
}

/// The strong ETag of `document`: the SHA-256 of its JSON, quoted.
pub fn etag<T: Serialize>(document: &T) -> Result<String, serde_json::Error> {
    let digest = Sha256::digest(serde_json::to_vec(document)?);
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    Ok(format!("\"{hex}\""))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Serialize)]
    struct Limit {
        tokens: u32,
    }

    #[test]
    fn the_start_is_revision_one_under_the_startup_principal() {
        let holder = AdminDocument::from_startup(Limit { tokens: 10 }).unwrap();
        let live = holder.load();
        assert_eq!(live.revision, 1);
        assert_eq!(live.principal, STARTUP_PRINCIPAL);
        assert_eq!(live.etag, etag(&Limit { tokens: 10 }).unwrap());
    }

    #[test]
    fn the_etag_is_the_quoted_hash_of_the_document_alone() {
        let one = etag(&Limit { tokens: 10 }).unwrap();
        assert_eq!(one.len(), 66, "{one}");
        assert!(one.starts_with('"') && one.ends_with('"'), "{one}");
        assert_eq!(one, etag(&Limit { tokens: 10 }).unwrap());
        assert_ne!(one, etag(&Limit { tokens: 11 }).unwrap());
    }

    #[test]
    fn a_write_is_the_next_revision_under_its_principal() {
        let holder = AdminDocument::from_startup(Limit { tokens: 10 }).unwrap();
        let next = Limit { tokens: 20 };
        let applied = {
            let writer = holder.write();
            assert_eq!(writer.live().revision, 1);
            writer.apply(next.clone(), etag(&next).unwrap(), "ops")
        };
        assert_eq!(applied.revision, 2);
        assert_eq!(applied.principal, "ops");
        assert_eq!(applied.document, next);
        assert_eq!(holder.load().revision, 2);
        assert_eq!(holder.load().etag, etag(&next).unwrap());
    }
}
