//! The policy spec calls: the read, a write and its dry run, and a
//! rollback, under a file's document and under a spec store.

use super::*;

/// A controller with a spec source serves the read call on the control
/// scope: the source's bytes under the standing number, to a caller with
/// a principal. A number that does not stand is not found, by name, and
/// the scope's rules stand as for every call.
#[tokio::test]
async fn the_read_call_answers_the_standing_document_once_a_source_applied_one() {
    let (_elector, _standing, control) = service(&[Scope::Control]);
    let (_elector, _standing, gateway) = service(&[Scope::Gateway]);
    for service in [&control, &gateway] {
        service
            .shared
            .generations
            .apply(LEAST_LOAD.into(), Format::Yaml)
            .expect("a valid document");
    }
    let read = |generation| from("operator-a", proto::GetPolicySpecRequest { generation });

    let answer = control
        .get_policy_spec(read(0))
        .await
        .expect("the standing document")
        .into_inner();
    assert_eq!(answer.generation, 1);
    assert_eq!(answer.document, LEAST_LOAD.as_bytes());
    assert_eq!(answer.format, "yaml");
    assert!(answer.digest.starts_with("sha256:"), "{}", answer.digest);

    let older = control.get_policy_spec(read(4)).await.expect_err("refused");
    assert_eq!(older.code(), Code::NotFound);
    assert_eq!(
            older.message(),
            "no document is held for generation 4: the spec source's document is generation 1, generation 1 stands, and the standing generation's document alone is retained"
        );

    let unplaced = control
        .get_policy_spec(Request::new(proto::GetPolicySpecRequest::default()))
        .await
        .expect_err("refused");
    assert_eq!(unplaced.code(), Code::Unauthenticated);
    let denied = gateway.get_policy_spec(read(0)).await.expect_err("refused");
    assert_eq!(denied.code(), Code::PermissionDenied);
}

/// The same under a filter no catalog holds: the validator refuses it.
const NOWHERE: &str =
    "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [nowhere]\n    pick: random\n";

/// A dry run on the leader answers its document's verdict and applies
/// nothing: accepted, with the number the document would run under now
/// and its digest, or refused, with the standing number and digest and
/// one issue by its field. Either answer reads `read_only`: the source
/// is a file.
#[tokio::test]
async fn a_dry_run_answers_its_documents_verdict_and_applies_nothing() {
    let (_elector, _standing, control) = sourced(true);
    let standing = control.shared.generations.standing();
    let standing_digest = standing.1.expect("a digest").to_string();

    let would = control
        .apply_policy_spec(apply(proto::ApplyMode::DryRun, RANDOM, "yaml", ""))
        .await
        .expect("a verdict")
        .into_inner();
    assert!(would.accepted && would.read_only, "{would:?}");
    assert_eq!(would.generation, 2);
    assert!(would.digest.starts_with("sha256:"), "{would:?}");
    assert_ne!(would.digest, standing_digest);
    assert!(would.issues.is_empty(), "{would:?}");
    // The feed before and after: the standing document reads the loads
    // in every role, and the offered one reads nothing.
    let feed = would.needs.expect("the needs diff");
    let before = feed.before.expect("the feed before");
    assert_eq!(before.models.len(), 4, "{before:?}");
    assert!(
        before
            .models
            .iter()
            .all(|row| row.model == "*" && row.loads),
        "{before:?}"
    );
    assert_eq!(feed.after, Some(proto::NeedsSummary::default()));

    // A guard that names the standing digest passes, in JSON as in YAML.
    let json = br#"{"apiVersion":"smg.policy/v1","profiles":{"default":{"pick":"random"}}}"#;
    let guarded = control
        .apply_policy_spec(apply(
            proto::ApplyMode::DryRun,
            json.as_slice(),
            "json",
            &standing_digest,
        ))
        .await
        .expect("a verdict")
        .into_inner();
    assert!(guarded.accepted, "{guarded:?}");
    assert_eq!(guarded.generation, 2, "the first dry run reserved nothing");

    // A document sent as JSON is read as JSON: its refusal says so.
    let unread = control
        .apply_policy_spec(apply(proto::ApplyMode::DryRun, "{", "json", ""))
        .await
        .expect("a verdict")
        .into_inner();
    let [issue] = unread.issues.as_slice() else {
        panic!("one issue: {unread:?}")
    };
    assert!(
        issue.message.starts_with("not a policy spec in json: "),
        "{issue:?}"
    );

    let refused = control
        .apply_policy_spec(apply(proto::ApplyMode::DryRun, NOWHERE, "yaml", ""))
        .await
        .expect("a verdict")
        .into_inner();
    assert!(!refused.accepted && refused.read_only, "{refused:?}");
    assert!(refused.needs.is_none(), "{refused:?}");
    assert_eq!(
        (refused.generation, refused.digest.as_str()),
        (1, standing_digest.as_str())
    );
    let [issue] = refused.issues.as_slice() else {
        panic!("one issue: {refused:?}")
    };
    assert_eq!(issue.field_path, "profiles.default.filter");
    assert_eq!(
        issue.message,
        "profile \"default\": unknown filter \"nowhere\""
    );

    assert_eq!(control.shared.generations.standing(), standing);
}

/// A dry run a registered filter's factory refuses names the parameter's
/// own field: the path an operator edits.
#[tokio::test]
async fn a_dry_run_names_the_field_a_registered_filters_factory_refuses() {
    /// Keeps the workers with `slots` of their eight request slots free.
    #[derive(Debug, serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct MinFree {
        slots: u32,
    }

    impl crate::engine::Filter for MinFree {
        fn name(&self) -> &'static str {
            "min-free"
        }

        fn keep(
            &self,
            _: &crate::engine::RequestFacts<'_>,
            candidate: &crate::engine::Candidate,
            _: &crate::engine::Constraints<'_>,
        ) -> bool {
            candidate.in_flight.saturating_add(self.slots) <= 8
        }
    }

    const FIELD: &str = "profiles.default.filter[1].min-free.slots";
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter_with("min-free", |build| {
            let rule: MinFree = build.params()?;
            if rule.slots == 0 {
                return Err(crate::stages::StageError::bad(
                    "slots",
                    rule.slots,
                    "above zero",
                ));
            }
            Ok(Box::new(rule))
        })
        .expect("a free name");
    let (_elector, _standing, control) = electing_in(true, catalog);
    control
        .shared
        .generations
        .apply(LEAST_LOAD.into(), Format::Yaml)
        .expect("a valid document");
    let standing = control.shared.generations.standing();
    let dry_run = |slots: u32| {
        let document = format!(
                "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy, {{ min-free: {{ slots: {slots} }} }}]\n    pick: random\n"
            );
        apply(proto::ApplyMode::DryRun, document, "yaml", "")
    };

    let would = control
        .apply_policy_spec(dry_run(2))
        .await
        .expect("a verdict")
        .into_inner();
    assert!(would.accepted, "{would:?}");

    let refused = control
        .apply_policy_spec(dry_run(0))
        .await
        .expect("a verdict")
        .into_inner();
    assert!(!refused.accepted, "{refused:?}");
    let [issue] = refused.issues.as_slice() else {
        panic!("one issue: {refused:?}")
    };
    assert_eq!(issue.field_path, FIELD);
    assert_eq!(issue.message, format!("{FIELD} = 0 is not above zero"));
    assert_eq!(control.shared.generations.standing(), standing);
}

/// The write call refuses a document whose bindings differ on the
/// overload switch, at a dry run and at an enforce alike: one issue by
/// the field of the first binding that differs, in the refusal's words,
/// and the standing generation stays. Bindings that agree are taken.
#[tokio::test]
async fn the_write_call_names_the_binding_that_differs_on_the_overload_switch() {
    const FIELD: &str = "bindings[1].overload.enabled";
    const OFF: &str = "    overload:\n      enabled: false\n";
    let (_dir, _elector, _standing, control) = stored(true);
    let write = |mode: proto::ApplyMode, second: &str| {
        let head = "apiVersion: smg.policy/v1\nprofiles:\n  p:\n    pick: random\nbindings:\n";
        let first = format!("  - match:\n      model: chat/*\n    profile: p\n{OFF}");
        let document = format!("{head}{first}  - profile: p\n{second}");
        control.apply_policy_spec(apply(mode, document, "yaml", ""))
    };
    let agreed = write(proto::ApplyMode::Enforce, OFF).await;
    let agreed = agreed.expect("an answer").into_inner();
    assert!(agreed.accepted, "{agreed:?}");
    assert_eq!(agreed.generation, 1);
    for mode in [proto::ApplyMode::DryRun, proto::ApplyMode::Enforce] {
        let refused = write(mode, "").await.expect("an answer").into_inner();
        assert!(!refused.accepted && !refused.read_only, "{refused:?}");
        assert_eq!(refused.generation, 1, "the standing generation stays");
        let [issue] = refused.issues.as_slice() else {
            panic!("one issue: {refused:?}")
        };
        assert_eq!(issue.field_path, FIELD);
        assert_eq!(
                issue.message,
                format!("{FIELD} = true is not the false bindings[0] says, by its block or the block's default: a controller builds its workers with one default set of overload thresholds, so the bindings of its document say one word")
            );
    }
    assert_eq!(control.shared.generations.generation(), 1);
}

/// The spec's source is a file, which no call writes: an enforce and a
/// rollback answer `read_only` with the standing number and digest,
/// whatever the document, and validate nothing.
#[tokio::test]
async fn a_write_under_a_file_source_answers_read_only() {
    let (_elector, _standing, control) = sourced(true);
    let (generation, digest) = control.shared.generations.standing();
    let read_only = proto::ApplyResult {
        accepted: false,
        generation,
        digest: digest.expect("a digest").to_string(),
        read_only: true,
        ..proto::ApplyResult::default()
    };
    for document in [RANDOM, "no document at all: ["] {
        let answer = control
            .apply_policy_spec(apply(proto::ApplyMode::Enforce, document, "yaml", ""))
            .await
            .expect("an answer")
            .into_inner();
        assert_eq!(answer, read_only);
    }
    let rollback = control
        .rollback_policy_spec(from("operator-a", proto::RollbackRequest { generation: 1 }))
        .await
        .expect("an answer")
        .into_inner();
    assert_eq!(rollback, read_only);
    assert_eq!(control.shared.generations.standing().0, 1);
}

/// The lease is taken before the store is told its term. In between a
/// spec write, its dry run and a rollback are refused as a registration
/// is, `UNAVAILABLE`, and the standing generation does not move; each
/// answers once the store stands in the term.
#[tokio::test]
async fn a_spec_call_is_refused_until_the_store_stands_in_its_term() {
    // The dry run's kit before the store enters the term: the take of a
    // candidate alone, its fence open.
    let (mut elector, standing, control) = sourced(false);
    elector.lead_at(Instant::now());
    standing.send_replace(saw(IDENTITY, 1));
    let stands = control.shared.generations.standing();
    let calls = async || {
        [
            control
                .apply_policy_spec(apply(proto::ApplyMode::Enforce, RANDOM, "yaml", ""))
                .await,
            control
                .apply_policy_spec(apply(proto::ApplyMode::DryRun, RANDOM, "yaml", ""))
                .await,
            control
                .rollback_policy_spec(from("operator-a", proto::RollbackRequest { generation: 1 }))
                .await,
        ]
    };
    for early in calls().await {
        let early = early.expect_err("refused");
        assert_eq!(
            (early.code(), early.message()),
            (Code::Unavailable, super::super::NOT_IN_TERM)
        );
    }
    assert_eq!(control.shared.generations.standing(), stands);

    control.shared.store.set_term(1);
    for answer in calls().await {
        let answer = answer.expect("an answer").into_inner();
        assert!(answer.read_only, "{answer:?}");
    }
    assert_eq!(control.shared.generations.standing(), stands);
}

/// The checks of a write call run in one order, each neighbour pair by
/// a call that fails both: the scope before the lead, the lead before
/// the mode, the mode before the source, the source before the guard,
/// the guard before the document.
#[tokio::test]
async fn the_checks_of_a_write_call_run_in_one_order() {
    let (_elector, _standing, follower) = sourced(false);
    let (_elector, _standing, leader) = sourced(true);
    let unstated = proto::ApplyMode::Unspecified;

    // The scope before the lead: a follower refuses a caller with no
    // principal as it refuses one everywhere.
    let unplaced = follower
        .apply_policy_spec(Request::new(proto::ApplyRequest::default()))
        .await
        .expect_err("refused");
    assert_eq!(unplaced.code(), Code::Unauthenticated);

    // The lead before the mode: a follower refuses a write that states
    // no mode by the leader's name, and a rollback with it.
    let not_leading = [
        follower
            .apply_policy_spec(apply(unstated, RANDOM, "yaml", ""))
            .await
            .expect_err("refused"),
        follower
            .rollback_policy_spec(from("operator-a", proto::RollbackRequest::default()))
            .await
            .expect_err("refused"),
    ];
    for refused in not_leading {
        assert_eq!(refused.code(), Code::FailedPrecondition);
        assert_eq!(refused.message(), "this replica does not lead");
    }

    // The mode before the source: a write states its mode, and audit is
    // not served; neither reads the read-only answer of the source.
    for mode in [unstated.into(), 99] {
        let refused = leader
            .apply_policy_spec(apply(mode, RANDOM, "yaml", ""))
            .await
            .expect_err("refused");
        assert_eq!(refused.code(), Code::InvalidArgument);
        assert_eq!(
            refused.message(),
            "mode: a write states its mode, enforce, audit or dry-run"
        );
    }
    let audit = leader
        .apply_policy_spec(apply(proto::ApplyMode::Audit, RANDOM, "yaml", ""))
        .await
        .expect_err("refused");
    assert_eq!(audit.code(), Code::Unimplemented);
    assert_eq!(
        audit.message(),
        "ApplyPolicySpec in audit mode is not served by this build"
    );

    // The source before the guard: an enforce whose guard names another
    // digest is answered read-only; no write could follow it.
    let enforced = leader
        .apply_policy_spec(apply(
            proto::ApplyMode::Enforce,
            RANDOM,
            "yaml",
            "sha256:another",
        ))
        .await
        .expect("an answer")
        .into_inner();
    assert!(enforced.read_only && !enforced.accepted, "{enforced:?}");

    // The guard before the document: a dry run whose guard names
    // another digest is refused whatever its document and its form.
    for (document, format) in [(RANDOM, "yaml"), (NOWHERE, "yaml"), (RANDOM, "toml")] {
        let guarded = leader
            .apply_policy_spec(apply(
                proto::ApplyMode::DryRun,
                document,
                format,
                "sha256:another",
            ))
            .await
            .expect_err("refused");
        assert_eq!(guarded.code(), Code::FailedPrecondition);
        assert!(
            guarded
                .message()
                .starts_with("expected_digest: generation 1 stands with digest sha256:"),
            "{}",
            guarded.message()
        );
    }

    // The document last: its form, then its verdict.
    let form = leader
        .apply_policy_spec(apply(proto::ApplyMode::DryRun, RANDOM, "toml", ""))
        .await
        .expect_err("refused");
    assert_eq!(form.code(), Code::InvalidArgument);
    assert_eq!(form.message(), "format: name yaml or json");
}

/// The call asks through the door's turn: while a dry run's pass stands
/// in the validator a second call's pass does not enter it, and both
/// are answered once the first ends.
#[tokio::test]
async fn a_second_call_of_a_dry_run_waits_its_turn() {
    use crate::service::controller::generations::tests::{gate, gated, Gate, Gated};

    gate!(Here);
    let gate = Here::gate();
    let (_elector, _standing, mut control) = sourced(true);
    let door = gated::<Here>();
    door.apply(LEAST_LOAD.into(), Format::Yaml)
        .expect("a valid document");
    Arc::get_mut(&mut control.shared)
        .expect("the service alone holds it")
        .generations = door;
    let run = |document: &'static str| {
        control.apply_policy_spec(apply(proto::ApplyMode::DryRun, document, "yaml", ""))
    };
    gate.arm();
    let beside = async {
        let entered = gate.entered_a_pace_on().await;
        gate.release();
        entered
    };
    let (first, second, entered) = tokio::join!(run(RANDOM), run(NOWHERE), beside);
    assert_eq!(entered, 1, "a second call's pass ran beside the first");
    assert!(first.expect("an answer").into_inner().accepted);
    assert!(!second.expect("an answer").into_inner().accepted);
}

/// A dry run that no place is left for is told to ask again.
#[test]
fn a_dry_run_let_in_nowhere_is_told_to_ask_again() {
    use crate::service::controller::generations::Unanswered;

    let full = unanswered(Unanswered::Full);
    assert_eq!(full.code(), Code::ResourceExhausted);
    assert_eq!(
        full.message(),
        "32 dry runs wait their turn: ask again later"
    );
}

/// An enforce under a spec store, on the leader: the document is
/// persisted and runs as the next generation, and the answer says so:
/// accepted, the number, the digest, the feed before and after, and not
/// `read_only`. The read call answers the bytes. A document that does
/// not run answers the standing number and one issue, a guard that
/// names another digest is refused, and a dry run answers what an
/// enforce would with nothing applied.
#[tokio::test]
async fn an_enforce_under_a_spec_store_runs_as_the_next_generation() {
    let (dir, _elector, _standing, control) = stored(true);
    let on_file = || names(dir.path()).len();
    let read = || {
        let standing = proto::GetPolicySpecRequest::default();
        control.get_policy_spec(from("operator-a", standing))
    };
    let write = |mode: proto::ApplyMode, document: &'static str, expected: &str| {
        control.apply_policy_spec(apply(mode, document, "yaml", expected))
    };
    // No generation yet: the spec calls are served, and the read has
    // no document to answer.
    let none = read().await.expect_err("no document yet");
    assert_eq!(none.code(), Code::NotFound);

    let first = write(proto::ApplyMode::Enforce, RANDOM, "")
        .await
        .expect("an answer")
        .into_inner();
    assert!(first.accepted && !first.read_only, "{first:?}");
    assert_eq!(first.generation, 1);
    assert!(first.digest.starts_with("sha256:"), "{first:?}");
    assert!(first.issues.is_empty(), "{first:?}");
    let nothing = Some(proto::NeedsSummary::default());
    let feed = first.needs.clone().expect("the needs diff");
    assert_eq!((&feed.before, &feed.after), (&nothing, &nothing));
    assert_eq!((control.shared.generations.generation(), on_file()), (1, 1));
    let standing = read().await.expect("the standing document").into_inner();
    assert_eq!(
        (standing.generation, standing.document.as_slice()),
        (1, RANDOM.as_bytes())
    );
    assert_eq!(standing.digest, first.digest);

    // A document that does not run: the standing number and digest,
    // and one issue by its field.
    let refused = write(proto::ApplyMode::Enforce, NOWHERE, &first.digest)
        .await
        .expect("an answer")
        .into_inner();
    assert!(!refused.accepted && !refused.read_only, "{refused:?}");
    assert_eq!((refused.generation, &refused.digest), (1, &first.digest));
    let [issue] = refused.issues.as_slice() else {
        panic!("one issue: {refused:?}")
    };
    assert!(!issue.field_path.is_empty(), "{issue:?}");
    assert!(issue.message.contains("nowhere"), "{issue:?}");
    assert!(refused.needs.is_none(), "{refused:?}");

    // A guard that names another digest than the standing one: another
    // apply landed first.
    let other = format!("sha256:{}", "0".repeat(64));
    let late = write(proto::ApplyMode::Enforce, LEAST_LOAD, &other)
        .await
        .expect_err("guarded");
    assert_eq!(late.code(), Code::FailedPrecondition);
    assert_eq!(
        late.message(),
        format!(
            "expected_digest: generation 1 stands with digest {}",
            first.digest
        )
    );

    // A dry run answers what an enforce would and applies nothing.
    let dry = write(proto::ApplyMode::DryRun, LEAST_LOAD, &first.digest)
        .await
        .expect("a verdict")
        .into_inner();
    assert!(dry.accepted && !dry.read_only, "{dry:?}");
    assert_eq!(dry.generation, 2);
    assert_eq!((control.shared.generations.generation(), on_file()), (1, 1));

    let second = write(proto::ApplyMode::Enforce, LEAST_LOAD, &first.digest)
        .await
        .expect("an answer")
        .into_inner();
    assert!(second.accepted, "{second:?}");
    assert_eq!((second.generation, on_file()), (2, 2));
    let feed = second.needs.expect("the needs diff");
    assert_eq!(feed.before, nothing);
    assert!(
        feed.after.is_some_and(|after| !after.models.is_empty()),
        "the standing document reads the loads"
    );
}

/// A follower takes no write under a spec store either: it refuses an
/// enforce and a rollback by the leader's name, and nothing is on file.
#[tokio::test]
async fn a_follower_refuses_a_write_under_a_spec_store() {
    let (dir, _elector, _standing, control) = stored(false);
    let refused = control
        .apply_policy_spec(apply(proto::ApplyMode::Enforce, RANDOM, "yaml", ""))
        .await
        .expect_err("a follower");
    assert_eq!(refused.code(), Code::FailedPrecondition);
    assert_eq!(refused.message(), "this replica does not lead");
    let rollback = from("operator-a", proto::RollbackRequest { generation: 1 });
    let refused = control
        .rollback_policy_spec(rollback)
        .await
        .expect_err("a follower");
    assert_eq!(refused.code(), Code::FailedPrecondition);
    assert_eq!(refused.message(), "this replica does not lead");
    assert_eq!(names(dir.path()).len(), 0);
    assert_eq!(control.shared.generations.generation(), 0);
}

/// A rollback under a spec store, on the leader: the document of a
/// generation on file runs again as the next generation, and the answer
/// is an enforce's. The read call answers each generation on file by
/// its number. A rollback that names no generation is refused by the
/// field, and a generation that is not on file is not found, by a
/// rollback and by the read call alike.
#[tokio::test]
async fn a_rollback_under_a_spec_store_runs_a_generation_on_file_again() {
    let (dir, _elector, _standing, control) = stored(true);
    let write = |document: &'static str| {
        control.apply_policy_spec(apply(proto::ApplyMode::Enforce, document, "yaml", ""))
    };
    let rollback = |generation: u64| {
        let request = proto::RollbackRequest { generation };
        control.rollback_policy_spec(from("operator-a", request))
    };
    let read = |generation: u64| {
        let request = proto::GetPolicySpecRequest { generation };
        control.get_policy_spec(from("operator-a", request))
    };
    let first = write(RANDOM).await.expect("an answer").into_inner();
    let second = write(LEAST_LOAD).await.expect("an answer").into_inner();
    assert_eq!((first.generation, second.generation), (1, 2));

    let back = rollback(1).await.expect("an answer").into_inner();
    assert!(back.accepted && !back.read_only, "{back:?}");
    assert_eq!((back.generation, &back.digest), (3, &first.digest));
    assert!(back.issues.is_empty(), "{back:?}");
    let feed = back.needs.expect("the needs diff");
    assert!(
        feed.before.is_some_and(|before| !before.models.is_empty()),
        "the document rolled away from reads the loads"
    );
    assert_eq!(feed.after, Some(proto::NeedsSummary::default()));
    assert_eq!(control.shared.generations.generation(), 3);
    assert_eq!(names(dir.path()).len(), 3);

    for (asked, answered, document, digest) in [
        (1, 1, RANDOM, &first.digest),
        (2, 2, LEAST_LOAD, &second.digest),
        (3, 3, RANDOM, &first.digest),
        (0, 3, RANDOM, &first.digest),
    ] {
        let held = read(asked).await.expect("a document").into_inner();
        assert_eq!(held.generation, answered, "asked for {asked}");
        assert_eq!(
            (held.document.as_slice(), &held.digest, held.format.as_str()),
            (document.as_bytes(), digest, "yaml")
        );
    }

    let unnamed = rollback(0).await.expect_err("no generation is named");
    assert_eq!(unnamed.code(), Code::InvalidArgument);
    assert_eq!(
        unnamed.message(),
        "generation: a rollback names a generation on file"
    );
    let gone = [
        rollback(9).await.expect_err("not on file"),
        read(9).await.expect_err("not on file"),
    ];
    for refused in gone {
        assert_eq!(refused.code(), Code::NotFound);
        assert_eq!(
                refused.message(),
                "no document is held for generation 9: the spec store has 3 on file, from generation 1 to generation 3"
            );
    }
    assert_eq!(control.shared.generations.generation(), 3);
}

#[test]
fn a_write_call_that_found_no_place_is_told_to_ask_again() {
    let full = unanswered(Unanswered::Waiting);
    assert_eq!(full.code(), Code::ResourceExhausted);
    assert_eq!(
        full.message(),
        "32 applies wait their turn: ask again later"
    );
}
