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
