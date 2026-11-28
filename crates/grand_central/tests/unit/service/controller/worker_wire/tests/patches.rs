//! A patch of a standing row: what it merges, what it replaces, and the
//! bounds of the row it would make.

use super::*;

#[test]
fn a_patch_merges_the_labels_and_replaces_the_overload_block() {
    // A label the patch leaves unsaid stays: the patch is merged.
    let mut standing = full().facts;
    standing.labels.insert("zone".to_owned(), "west".to_owned());
    let request = proto::PatchWorkerRequest {
        labels: [
            ("tier".to_owned(), "silver".to_owned()),
            ("lane".to_owned(), "b".to_owned()),
        ]
        .into(),
        remove_labels: vec!["policy".to_owned(), "absent".to_owned()],
        overload: Some(proto::OverloadThresholds {
            queue_limit: Some(12),
            kv_usage_permille: None,
        }),
        ..Default::default()
    };
    let made = patch(&request, &standing)
        .expect("a patch in order")
        .expect("a patch that changes the row");
    assert_eq!(
        made.labels,
        Some(BTreeMap::from([
            ("lane".to_owned(), "b".to_owned()),
            ("tier".to_owned(), "silver".to_owned()),
            ("zone".to_owned(), "west".to_owned()),
        ]))
    );
    assert_eq!(made.priority, None);
    assert_eq!(
        made.overload,
        Some(Overload {
            kv_usage_permille: Overload::default().kv_usage_permille,
            waiting_requests: Some(12),
        })
    );
}

#[test]
fn a_patch_that_changes_nothing_is_none_and_a_label_both_set_and_removed_is_refused() {
    // The row states both thresholds and so does each block below: a
    // block is compared with the row field by field, whatever the
    // store's defaults are.
    let mut standing = full().facts;
    standing.overload.waiting_requests = Some(12);
    let stating = |queue_limit| proto::PatchWorkerRequest {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        remove_labels: vec!["absent".to_owned()],
        overload: Some(proto::OverloadThresholds {
            queue_limit: Some(queue_limit),
            kv_usage_permille: Some(875),
        }),
        ..Default::default()
    };
    assert_eq!(patch(&stating(12), &standing), Ok(None));
    let empty = proto::PatchWorkerRequest::default();
    assert_eq!(patch(&empty, &standing), Ok(None));
    // A block that differs in one threshold is the patch's one change.
    let made = patch(&stating(13), &standing).expect("a patch in order");
    assert_eq!(
        made.map(|patch| (patch.labels, patch.overload)),
        Some((
            None,
            Some(Overload {
                kv_usage_permille: Some(875),
                waiting_requests: Some(13),
            })
        ))
    );

    let both = proto::PatchWorkerRequest {
        labels: [("tier".to_owned(), "silver".to_owned())].into(),
        remove_labels: vec!["tier".to_owned()],
        ..Default::default()
    };
    assert_eq!(
        patch(&both, &standing).map_err(|refused| refused.path),
        Err("remove_labels")
    );
    let over = proto::PatchWorkerRequest {
        overload: Some(proto::OverloadThresholds {
            queue_limit: None,
            kv_usage_permille: Some(4_000),
        }),
        ..Default::default()
    };
    assert_eq!(
        patch(&over, &standing).map_err(|refused| refused.path),
        Err("overload.kv_usage_permille")
    );
}

#[test]
fn a_patch_is_held_to_the_bounds_of_the_row_it_would_make() {
    type Asking = fn(usize) -> proto::PatchWorkerRequest;
    let keyed = |labels: Vec<(String, String)>| proto::PatchWorkerRequest {
        labels: labels.into_iter().collect(),
        ..Default::default()
    };
    let cases: [(&str, usize, &str, Asking); 5] = [
        (
            "labels",
            KEY_BYTES,
            "a label's key is at most 128 bytes",
            |bytes| proto::PatchWorkerRequest {
                labels: [(text(bytes), String::new())].into(),
                ..Default::default()
            },
        ),
        (
            "labels",
            VALUE_BYTES,
            "a label's value is at most 4096 bytes",
            |bytes| proto::PatchWorkerRequest {
                labels: [("k".to_owned(), text(bytes))].into(),
                ..Default::default()
            },
        ),
        (
            "labels",
            LABELS,
            "a row holds at most 128 labels",
            |count| {
                let labels = (0..count).map(|n| (n.to_string(), String::new()));
                proto::PatchWorkerRequest {
                    labels: labels.collect(),
                    ..Default::default()
                }
            },
        ),
        (
            "remove_labels",
            LABELS,
            "a patch removes at most 128 labels",
            |count| {
                let remove_labels = (0..count).map(|n| n.to_string()).collect();
                proto::PatchWorkerRequest {
                    remove_labels,
                    ..Default::default()
                }
            },
        ),
        (
            "remove_labels",
            KEY_BYTES,
            "a label's key is at most 128 bytes",
            |bytes| proto::PatchWorkerRequest {
                remove_labels: vec![text(bytes)],
                ..Default::default()
            },
        ),
    ];
    let bare = facts(&spec(), None).expect("a bare row");
    for (path, bound, why, asking) in cases {
        assert!(patch(&asking(bound), &bare).is_ok(), "{path} at {bound}");
        assert_eq!(
            patch(&asking(bound + 1), &bare).map(|_| ()),
            Err(refused(path, why)),
            "{path} past {bound}"
        );
        assert!(why.contains(&bound.to_string()), "{why}");
    }
    // The removals' count is read before the list is: past it a patch
    // is refused by the count, whatever its list holds.
    let both = proto::PatchWorkerRequest {
        labels: [("0".to_owned(), String::new())].into(),
        remove_labels: (0..=LABELS).map(|n| n.to_string()).collect(),
        ..Default::default()
    };
    assert_eq!(
        patch(&both, &bare).map(|_| ()),
        Err(refused(
            "remove_labels",
            "a patch removes at most 128 labels"
        ))
    );
    // The count is of the row the patch would make: a label it removes
    // makes room for the one it sets.
    let mut crowded = bare.clone();
    crowded.labels = (0..LABELS)
        .map(|n| (n.to_string(), String::new()))
        .collect();
    let more = keyed(vec![("more".to_owned(), String::new())]);
    assert_eq!(
        patch(&more, &crowded).map(|_| ()),
        Err(refused("labels", "a row holds at most 128 labels"))
    );
    let swap = proto::PatchWorkerRequest {
        remove_labels: vec!["0".to_owned()],
        ..more
    };
    assert!(patch(&swap, &crowded).is_ok_and(|made| made.is_some()));
    // The row whole: a label of 500 bytes under a one-byte key adds 509.
    let near = facts(&weighing(ROW_BYTES - 509), None).expect("a row under its bound");
    let filling = |bytes| keyed(vec![("p".to_owned(), text(bytes))]);
    let made = patch(&filling(500), &near).expect("a row at its bound");
    let mut after = near.clone();
    after.labels = made
        .and_then(|patch| patch.labels)
        .expect("the labels moved");
    assert_eq!(weight(&after), ROW_BYTES);
    assert_eq!(
        patch(&filling(501), &near).map(|_| ()),
        Err(refused(
            "labels",
            "a row is at most 16384 bytes on the wire"
        ))
    );
    // The policy label is told twice, as a label and under its own
    // field: one of 246 bytes on a row with none adds 510.
    let under = facts(&weighing(ROW_BYTES - 510), None).expect("a row under its bound");
    let policy = |bytes| keyed(vec![("policy".to_owned(), text(bytes))]);
    let made = patch(&policy(246), &under).expect("a row at its bound");
    let mut after = under.clone();
    after.labels = made
        .and_then(|patch| patch.labels)
        .expect("the labels moved");
    assert_eq!(weight(&after), ROW_BYTES);
    assert_eq!(
        patch(&policy(247), &under).map(|_| ()),
        Err(refused(
            "labels",
            "a row is at most 16384 bytes on the wire"
        ))
    );
    // A row at its bound takes a patch of the overload block alone and
    // stays within it at its widest: the block was weighed so.
    let brim = facts(&weighing(ROW_BYTES), None).expect("a row at its bound");
    let block = proto::PatchWorkerRequest {
        overload: Some(proto::OverloadThresholds {
            queue_limit: Some(u32::MAX),
            kv_usage_permille: Some(1_000),
        }),
        ..Default::default()
    };
    let made = patch(&block, &brim).expect("a patch in order");
    let stated = WorkerFacts {
        overload: made.and_then(|patch| patch.overload).expect("the block"),
        ..brim
    };
    let mut widest = row(&WorkerRow {
        id: WorkerId(u64::MAX),
        facts: stated.clone(),
        status: WorkerStatus::Ready,
        revision: u64::MAX,
        ..full()
    });
    widest.incarnation = u64::MAX;
    assert_eq!(widest.encoded_len(), ROW_BYTES);
    // And the row registers again as it stands, its block stated.
    let again = proto::WorkerSpec {
        overload: block.overload,
        ..weighing(ROW_BYTES)
    };
    assert!(facts(&again, Some(&stated)).is_ok_and(|made| made == stated));
    // And a label of the same bytes in other letters still fits it.
    let other = keyed(vec![("a".to_owned(), "b".repeat(VALUE_BYTES))]);
    assert!(patch(&other, &stated).is_ok_and(|made| made.is_some()));
}
