//! A request of text decided inside its host's string tree: a worker the
//! index holds a block for, a text of several bytes a char, a partition's
//! marker, and the target a text is stored for.

use super::*;

/// A worker the model's index holds a block for is named by the string
/// tree's match as any other: a decision inside a tree reads no word of
/// who the index speaks for. The landed case beside an index stores the
/// index's block for another worker than the one the text goes to.
///
/// The classic policy sends every request of text to its string tree,
/// whatever its index holds (`select_worker` in
/// `model_gateway/src/policies/cache_aware.rs`).
#[test]
fn a_request_of_text_is_named_by_the_string_tree_for_a_worker_the_index_holds() {
    let tree = Arc::new(StringTree::new());
    let index = Arc::new(ChainIndex::new());
    // The index holds a block for the worker the text goes to.
    store(&index, 2, &[7]);
    let host = Arc::new(Host {
        index: Some(index),
        text: Some(Arc::clone(&tree)),
        ..Host::default()
    });
    let affinity = Affinity::default();
    affinity.set_indexes(host as Arc<dyn CacheIndexes>);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = text(1_000, 4);
    assert!(bed.worded(&asked, 0, Some(2)).is_empty());
    assert_eq!(bed.worded(&asked, 0, None), [(2, 0, 4)]);
    // The tree holds the text for that worker and no other.
    let found = tree.match_prefix_with_counts(&asked);
    assert_eq!(found.matched_char_count, asked.chars().count());
    assert_eq!(found.matched_tenants, [TenantId::from(address(2))]);
}

/// A string tree counts chars and a request of text is sized in bytes: a
/// match is turned into the bytes of the text it matched before it is
/// counted in the request's windows. So a text of several bytes a char is
/// named by every window it was sent, where a char read as a byte would
/// name fewer.
///
/// The classic policy divides the chars its tree matched by the text's
/// chars (`select_worker_with_text` in
/// `model_gateway/src/policies/cache_aware.rs`).
#[test]
fn a_text_of_several_bytes_a_char_is_counted_in_its_bytes() {
    let (affinity, _tree) = lending_text(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    for (first, fill) in [(2_000, 'é'), (3_000, '語'), (4_000, '🦀')] {
        let asked = text_of(first, 4, fill);
        assert_eq!(asked.len(), 4 * WINDOW);
        assert!(asked.chars().count() < asked.len(), "{fill}");
        assert!(bed.worded(&asked, 0, Some(1)).is_empty(), "{fill}");
        assert_eq!(bed.worded(&asked, 0, None), [(1, 0, 4)], "{fill}");
    }
    // The bytes of a match behind a marker, by its chars.
    assert_eq!(bytes_behind("é語🦀a", 0, 0), 0);
    assert_eq!(bytes_behind("é語🦀a", 0, 2), 5);
    assert_eq!(bytes_behind("é語🦀a", 1, 3), 7);
    assert_eq!(bytes_behind("é語🦀a", 1, 9), 8);
    // A match that ends inside the marker holds nothing of the prompt.
    assert_eq!(bytes_behind("é語🦀a", 3, 2), 0);
}

/// The naming of a cache partition is the host's, for text as for tokens.
/// An entry named behind a partition's marker shares no prefix with another
/// partition's nor with a plain one, and the marker's chars are no part of
/// the windows a worker is named by, whatever their bytes.
#[test]
fn a_hosts_marker_keeps_two_partitions_of_text_apart() {
    // Eighteen chars no prompt begins with: one of them of two bytes, and
    // then each of one byte. A key of one byte a char is counted by its
    // chars, with no pass over it.
    type Under = fn(u32, &str) -> String;
    let wide: Under = |partition, prompt| format!("\u{1}§{partition:016x}{prompt}");
    let narrow: Under = |partition, prompt| format!("\u{1}{partition:016x}\u{1}{prompt}");
    let marker = 18;
    let asked = text(1_000, 4);
    for (under, bytes) in [(wide, marker + 1), (narrow, marker)] {
        assert_eq!(under(1, "").len(), bytes);
        let (affinity, _) = lending_text(false);
        let bed = Bed::new(affinity, preset("cache_aware"), 4);
        assert!(bed.worded(&under(1, &asked), marker, Some(2)).is_empty());
        assert_eq!(bed.worded(&under(1, &asked), marker, None), [(2, 0, 4)]);
        assert!(bed.worded(&under(2, &asked), marker, None).is_empty());
        assert!(bed.worded(&asked, 0, None).is_empty());
        // Another prompt of the partition shares the marker alone, and a
        // marker is no window of a prompt.
        let other = under(1, &text(5_000, 4));
        assert!(bed.worded(&other, marker, None).is_empty());
        // A prompt that ends fourteen bytes short of its fourth window is
        // named by three: the marker's bytes fill no window of it.
        let cut = &asked[..4 * WINDOW - 14];
        assert_eq!(bed.worded(&under(1, cut), marker, None), [(2, 0, 3)]);
    }
}

/// The string tree stores a text for the first target of the pick its
/// caller sends to, asked best first, and for no worker where the caller
/// sends to none, as the token tree does
/// ([`a_tree_stores_a_prompt_for_the_first_target_its_caller_sends_to`]).
/// The host and the engine read and write one tree: what the engine decided
/// stands in the host's tree under the worker's address, every char of it.
#[test]
fn a_string_tree_stores_a_text_for_the_first_target_its_caller_sends_to() {
    let (affinity, tree) = lending_text(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    // The workers at 1 and 2, best first.
    let two = || {
        let mut targets = Targets::new();
        for position in 1..3 {
            targets.push(Target {
                position,
                id: bed.ids[position as usize],
                ..Target::default()
            });
        }
        Decision::Pick(Pick {
            targets,
            trace: Trace::default(),
            session_verdict: SessionVerdict::None,
        })
    };
    // One decision of `key` that picks the two, under a caller that sends
    // to the places `sent` names. Says the places the caller was asked of.
    let place = |key: &str, sent: &[u32]| {
        let hashes = worded(key);
        let asked = RefCell::new(Vec::new());
        let entry = TextKey {
            text: Cow::Borrowed(key),
            marker: 0,
        };
        decide(
            &bed.affinity,
            &bed.pool,
            &bed.spec,
            &bed.workers,
            &bed.ids,
            &mut request(&hashes, 7, NOW),
            None,
            |_| None,
            || Some(entry),
            |target| {
                asked.borrow_mut().push(target.position);
                sent.contains(&target.position)
            },
            |_| two(),
        );
        asked.into_inner()
    };
    let (first, second, none) = (text(1_000, 4), text(2_000, 4), text(3_000, 4));
    assert_eq!(place(&first, &[1, 2]), [1]);
    assert_eq!(bed.worded(&first, 0, None), [(1, 0, 4)]);
    // A caller that passes the first over is stored for the second alone.
    assert_eq!(place(&second, &[2]), [1, 2]);
    assert_eq!(bed.worded(&second, 0, None), [(2, 0, 4)]);
    // One that sends to none is asked of each, and nothing is stored.
    assert_eq!(place(&none, &[]), [1, 2]);
    assert!(bed.worded(&none, 0, None).is_empty());
    for (text, at) in [(&first, 1), (&second, 2)] {
        let found = tree.match_prefix_with_counts(text);
        assert_eq!(found.matched_char_count, text.chars().count());
        assert_eq!(found.matched_tenants, [TenantId::from(address(at))]);
    }
    assert_eq!(tree.get_tenant_char_count().len(), 2);
}
