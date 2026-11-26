use super::{
    busy, credit, decided, Checked, Items, ItemsPlaced, Lists, Refusal, MAX_FRAME, MAX_ITEMS,
};
use crate::{
    document::spec::{AppliedSpec, MAX_BLOCK_HASHES},
    engine::{
        decision::{
            self, Decision, LiftedVeto, Pair, Pick, Queue, Shed, ShedReason, Survivors, Target,
            Targets, Trace,
        },
        decision_engine::{Request, RequestMaker, WorkerLists},
        view::{Constraints, Identity, Phase, RequestFacts, WorkerId},
    },
    service::{
        controller::frame_request::FrameRequest,
        proto::{self, busy::Reason, decide_answer::Answer, decision::Outcome, identity::Form},
    },
};

#[test]
fn plain_spec_answers_keep_each_wire_field() {
    use std::time::{Duration, UNIX_EPOCH};

    use prost::Message as _;

    use crate::document::{
        operator::{Document, Format, NeedsRow, Status},
        spec::PolicySpec,
        Needs,
    };

    let digest = PolicySpec::preset("random")
        .expect("a policy")
        .digest()
        .expect("a digest");
    let applied = UNIX_EPOCH + Duration::from_micros(123_456);
    let status = super::spec_status(Status {
        generation: 7,
        digest: Some(digest),
        applied: Some(applied),
        retained: vec![2, 4],
        writable: true,
    });
    assert_eq!(
        status.encode_to_vec(),
        proto::SpecStatus {
            generation: 7,
            digest: digest.to_string(),
            applied_unix_micros: 123_456,
            mode: proto::ApplyMode::Enforce.into(),
            retained: vec![2, 4],
            source: "api".to_owned(),
        }
        .encode_to_vec()
    );
    let empty = super::spec_status(Status {
        generation: 0,
        digest: None,
        applied: None,
        retained: Vec::new(),
        writable: false,
    });
    assert_eq!(
        empty,
        proto::SpecStatus {
            source: "file".to_owned(),
            ..proto::SpecStatus::default()
        }
    );
    let document = super::spec_document(Document {
        generation: 7,
        digest,
        document: b"the source's bytes\n".to_vec(),
        format: Format::Yaml,
        applied,
    });
    assert_eq!(
        document.encode_to_vec(),
        proto::PolicySpecDocument {
            generation: 7,
            digest: digest.to_string(),
            document: b"the source's bytes\n".to_vec(),
            format: "yaml".to_owned(),
            applied_unix_micros: 123_456,
            mode: proto::ApplyMode::Enforce.into(),
            acks: Vec::new(),
        }
        .encode_to_vec()
    );
    let row = |model: &str, phase, needs| NeedsRow {
        model: model.to_owned(),
        phase,
        needs,
    };
    let diff = super::needs_diff((
        vec![
            row(
                "local",
                Phase::Single,
                Needs {
                    affinity: true,
                    ..Needs::default()
                },
            ),
            row(
                "paired",
                Phase::Prefill,
                Needs {
                    headers: true,
                    media: true,
                    ..Needs::default()
                },
            ),
        ],
        vec![row(
            "paired",
            Phase::Decode,
            Needs {
                text: true,
                token_hashes: true,
                text_hashes: true,
                loads: true,
                events: true,
                ring: true,
                ..Needs::default()
            },
        )],
    ));
    assert_eq!(
        diff.encode_to_vec(),
        proto::NeedsDiff {
            before: Some(proto::NeedsSummary {
                models: vec![proto::ModelNeeds {
                    model: "paired".to_owned(),
                    role: proto::Role::Prefill.into(),
                    headers: true,
                    media: true,
                    ..proto::ModelNeeds::default()
                }],
            }),
            after: Some(proto::NeedsSummary {
                models: vec![proto::ModelNeeds {
                    model: "paired".to_owned(),
                    role: proto::Role::Decode.into(),
                    text: true,
                    token_hashes: true,
                    text_hashes: true,
                    loads: true,
                    events: true,
                    ring: true,
                    ..proto::ModelNeeds::default()
                }],
            }),
        }
        .encode_to_vec(),
    );
}

use crate::{
    document::{
        operator::{needs_rows, test_support::LEAST_LOAD},
        spec::PolicySpec,
    },
    engine::{Candidate, Filter},
    stages::StageRegistry,
};

/// A need the contract's row has no flag for writes no row: the engine
/// reads the affinity column of a host's own stage in every phase, and
/// the column is the controller's to hold.
#[test]
fn a_need_with_no_flag_in_the_contract_writes_no_row() {
    #[derive(Debug)]
    struct Held;

    impl Filter for Held {
        fn name(&self) -> &'static str {
            "held"
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }

        fn needs(&self) -> crate::document::Needs {
            crate::document::Needs {
                affinity: true,
                ..crate::document::Needs::default()
            }
        }
    }

    let mut stages = StageRegistry::builtin();
    stages
        .filter("held", || Box::new(Held))
        .expect("a free name");
    let document = LEAST_LOAD.replace("[healthy]", "[held]");
    let spec: PolicySpec = serde_yaml::from_str(&document).expect("a document");
    let engine = spec.validate_in(&stages, &[]).expect("validates");
    let rows = super::needs_summary(needs_rows(&engine)).models;
    // The profile's scorer reads the loads: its rows stand, and say
    // nothing of the column.
    assert_eq!(rows.len(), 4, "{rows:?}");
    let quiet = document
        .replace("    score: [least-load]\n", "")
        .replace("best", "random");
    let spec: PolicySpec = serde_yaml::from_str(&quiet).expect("a document");
    let engine = spec.validate_in(&stages, &[]).expect("validates");
    assert_eq!(engine.needs_rows().len(), 4, "the column in every phase");
    assert_eq!(
        super::needs_summary(needs_rows(&engine)),
        proto::NeedsSummary::default()
    );
}

/// The summary tells each row under its own role and each need under
/// its own flag: a pair's prefill row and its decode row part, a row of
/// media alone is told, and a sessions rule's key is a row of its
/// binding in every phase it covers.
#[test]
fn the_summary_tells_each_role_and_each_flag_apart() {
    /// Keeps every worker and reads what it is built to read.
    #[derive(Debug)]
    struct Reads(&'static str, crate::document::Needs);

    impl Filter for Reads {
        fn name(&self) -> &'static str {
            self.0
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }

        fn needs(&self) -> crate::document::Needs {
            self.1
        }
    }

    let tokens = crate::document::Needs {
        token_hashes: true,
        ..crate::document::Needs::default()
    };
    let media = crate::document::Needs {
        media: true,
        ..crate::document::Needs::default()
    };
    let mut stages = StageRegistry::builtin();
    stages
        .filter("tokens", move || Box::new(Reads("tokens", tokens)))
        .expect("a free name")
        .filter("media", move || Box::new(Reads("media", media)))
        .expect("a free name");
    let document = r#"{
          "apiVersion": "smg.policy/v1",
          "view": { "headers": ["x-session-id"] },
          "profiles": {
            "prefill": { "filter": ["tokens"], "pick": "random" },
            "decode": { "filter": ["media"], "pick": "random" },
            "quiet": { "pick": "random" }
          },
          "bindings": [
            { "match": { "model": "pd/*", "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "prefill", "decode": "decode" } },
            { "match": { "model": "chat/*" }, "profile": "quiet", "sessions": { "sticky_header": "x-session-id", "ttl_s": 600 } },
            { "match": { "model": "*" }, "profile": "quiet" }
          ]
        }"#;
    let spec: PolicySpec = serde_json::from_str(document).expect("a document");
    let engine = spec.validate_in(&stages, &[]).expect("validates");
    let rows = super::needs_summary(needs_rows(&engine)).models;
    let row = |model: &str, role: proto::Role| {
        rows.iter()
            .find(|row| row.model == model && row.role == i32::from(role))
            .cloned()
    };
    let told = |model: &str, role: proto::Role| proto::ModelNeeds {
        model: model.to_owned(),
        role: role.into(),
        ..proto::ModelNeeds::default()
    };
    assert_eq!(
        row("pd/*", proto::Role::Prefill),
        Some(proto::ModelNeeds {
            token_hashes: true,
            ..told("pd/*", proto::Role::Prefill)
        })
    );
    assert_eq!(
        row("pd/*", proto::Role::Decode),
        Some(proto::ModelNeeds {
            media: true,
            ..told("pd/*", proto::Role::Decode)
        })
    );
    // The sessions rule reads its key under the view's headers, in
    // every phase its binding covers.
    for role in [
        proto::Role::Single,
        proto::Role::Prefill,
        proto::Role::Decode,
        proto::Role::Encode,
    ] {
        assert_eq!(
            row("chat/*", role),
            Some(proto::ModelNeeds {
                headers: true,
                ..told("chat/*", role)
            })
        );
    }
    assert_eq!(rows.len(), 6, "{rows:?}");
}

fn key_form(form: Form) -> proto::Identity {
    proto::Identity { form: Some(form) }
}

/// A frame whose every field holds a value of its own.
pub(crate) fn full() -> proto::DecideRequest {
    proto::DecideRequest {
        request_id: vec![7, 8, 9],
        facts: Some(proto::RequestFacts {
            model: "model-a".to_owned(),
            namespace_seed: 11,
            tenant: Some(key_form(Form::Plain("tenant-a".to_owned()))),
            priority: -100,
            streaming: true,
            max_output_tokens: 12,
            prompt_tokens: 13,
            block_hashes: vec![14, 15, 16],
            hash_kind: proto::HashKind::Text.into(),
            session_key: Some(key_form(Form::Hash(17))),
            phase: proto::Phase::Decode.into(),
            attempt: 2,
            tried: vec![18, 19],
            budget_micros: 2_000,
            headers: vec![
                proto::Header {
                    name: 0,
                    value: "first".to_owned(),
                },
                proto::Header {
                    name: 255,
                    value: "last".to_owned(),
                },
            ],
            media: Some(proto::MediaCounts {
                images: 20,
                audio: 21,
                video: 22,
                bytes: 23,
            }),
            seed: 24,
            prompt_text: Vec::new(),
            block_size: 25,
            client_key: 29,
            session_key_from_rid: true,
        }),
        constraints: Some(proto::Constraints {
            excluded: vec![26, 27, 28],
            wire_modes: 0b110,
            overload_lifted: true,
            subset_hint: Some(0),
        }),
        again: false,
        ..Default::default()
    }
}

/// The room of `lists`' three buffers, and how many headers stand in
/// theirs: what a case reads of a decider's buffers between two frames.
pub(crate) fn kept(lists: &Lists) -> ([usize; 3], usize) {
    let room = [
        lists.tried.capacity(),
        lists.excluded.capacity(),
        lists.headers.capacity(),
    ];
    (room, lists.headers.len())
}

/// Makes `frame`'s request as a controller's maker makes it
/// ([`FrameRequest`]), on `lists` and with `left` of the frame's
/// budget, and hands `check` its facts and its constraints. The
/// request bears the frame's id whole: that is held here, for every
/// frame a case makes a request of. The document it is made under has
/// a ring keyed by the prompt: it reads a prompt's hashes of either
/// kind 256 tokens deep, past the three a case's frame carries.
pub(crate) fn made<R>(
    frame: &Checked,
    lists: &mut Lists,
    left: std::time::Duration,
    check: impl FnOnce(&RequestFacts<'_>, &Constraints<'_>) -> R,
) -> R {
    let document = PolicySpec::preset("prefix_hash").expect("a built-in preset");
    let spec = AppliedSpec::validate(&document).expect("the preset validates");
    let lists = lists.of(frame);
    let maker = FrameRequest::new(frame, &lists, left, WorkerLists::None);
    let Request {
        id,
        facts,
        constraints,
    } = maker.make(&spec);
    assert_eq!(id, frame.id(), "a request bears its frame's id whole");
    check(&facts, &constraints)
}

/// The full frame with its facts edited.
fn with(edit: impl FnOnce(&mut proto::RequestFacts)) -> proto::DecideRequest {
    let mut frame = full();
    edit(frame.facts.as_mut().expect("the full frame has facts"));
    frame
}

/// The text is said and dropped at the check: the frame a queue holds
/// keeps none of it, its buffer neither.
#[test]
fn a_frame_that_carries_its_prompt_as_text_is_checked_and_says_so() {
    let plain = Checked::new(full()).expect("a frame the record holds");
    assert!(!plain.has_text());
    let text = Checked::new(with(|facts| facts.prompt_text = b"hello".to_vec()))
        .expect("text is no field out of its range");
    assert!(text.has_text());
    assert_eq!(text.facts.prompt_text.capacity(), 0);
}

/// The two words few frames say are a bit each of one byte: a frame
/// that says neither reads as none, and each word reads apart from
/// the other.
#[test]
fn a_frame_that_is_asked_again_says_so_beside_its_text() {
    let said = |frame: proto::DecideRequest| {
        let frame = Checked::new(frame).expect("no word is a field out of its range");
        (frame.says_more(), frame.has_text(), frame.again())
    };
    let texted = |text: &[u8], again| {
        let mut frame = with(|facts| facts.prompt_text = text.to_vec());
        frame.again = again;
        frame
    };
    assert_eq!(said(full()), (false, false, false));
    assert_eq!(said(texted(b"", true)), (true, false, true));
    assert_eq!(said(texted(b"hello", false)), (true, true, false));
    assert_eq!(said(texted(b"hello", true)), (true, true, true));
}

/// A pair's two words of its decode ride the same byte. A frame keeps
/// its `decode_later` bit, and its `decode_of` id as it came. A frame
/// that asks a lease's decode side is read at the decode phase, whatever
/// its facts say, and is no pair's first request: its bit is not read.
#[test]
fn a_frame_keeps_its_decode_later_bit_and_its_decode_of_id() {
    use crate::engine::view::Phase;
    let said = |later: bool, of: Option<u64>| {
        let mut frame = with(|facts| facts.phase = proto::Phase::Prefill.into());
        (frame.decode_later, frame.decode_of) = (later, of);
        let frame = Checked::new(frame).expect("no word is a field out of its range");
        let of = frame.decode_of().map(|lease| lease.0);
        (frame.says_more(), frame.decode_later(), of, frame.phase())
    };
    let lease = 1 << 32 | 7;
    assert_eq!(said(false, None), (false, false, None, Phase::Prefill));
    assert_eq!(said(true, None), (true, true, None, Phase::Prefill));
    let call = (true, false, Some(lease), Phase::Decode);
    assert_eq!(said(false, Some(lease)), call);
    assert_eq!(said(true, Some(lease)), call);
    let last = Some(u64::MAX);
    assert_eq!(said(false, last), (true, false, last, Phase::Decode));
}

/// The number a frame's request names its worker by is kept as it came:
/// none for a frame that states none, the first worker's apart from
/// none, and the widest number. It is kept whatever the frame's phase:
/// its decision reads it or does not. It is no word of the byte few
/// frames say one of: a decider's one test reads a frame that names its
/// worker as it reads any other.
#[test]
fn a_frame_keeps_the_number_its_request_names_its_worker_by() {
    let said = |named: Option<u32>, phase: proto::Phase| {
        let mut frame = with(|facts| facts.phase = phase.into());
        frame.named_worker = named;
        let frame = Checked::new(frame).expect("no word is a field out of its range");
        (frame.says_more(), frame.named_worker())
    };
    let single = proto::Phase::Single;
    assert_eq!(said(None, single), (false, None));
    assert_eq!(said(Some(0), single), (false, Some(0)));
    assert_eq!(said(Some(7), proto::Phase::Prefill), (false, Some(7)));
    let widest = Some(u32::MAX);
    assert_eq!(said(widest, single), (false, widest));
}

/// No lease has the id 0: a frame that names it is refused by its
/// field, as any field out of its range is.
#[test]
fn a_decode_of_zero_is_refused_by_its_field() {
    let mut frame = full();
    frame.decode_of = Some(0);
    let refused = Checked::new(frame).err();
    assert_eq!(refused, Some(Refusal::OutOfRange("decode_of")));
}

/// The full frame at `phase` with the items of an encode request: `keys`
/// keys, each a hash of its own.
fn with_items(phase: proto::Phase, keys: u64) -> proto::DecideRequest {
    let mut frame = with(|facts| facts.phase = phase.into());
    frame.encode = Some(proto::EncodeItems {
        keys: (1..=keys).map(|key| key_form(Form::Hash(key))).collect(),
        ..proto::EncodeItems::default()
    });
    frame
}

/// An encode request is a frame of the encode phase that carries its
/// items. It is read at the prefill phase, as its pair's one request,
/// and says so in the byte of the words few frames say. Its items read
/// as the engine takes them: each key in the frame's order, the one key
/// where the frame states one, with a form or with none, and what the
/// request asks of its decode.
/// The field's presence marks the request, with a key or with none. A
/// frame of the encode phase that does not state the field is read at
/// its own phase and is no encode request.
#[test]
fn an_encode_phase_frame_that_carries_items_is_read_at_the_prefill_phase() {
    let bare = with(|facts| facts.phase = proto::Phase::Encode.into());
    let bare = Checked::new(bare).expect("a frame the record holds");
    assert_eq!((bare.phase(), bare.carries_items()), (Phase::Encode, false));
    assert!(!bare.says_more() && bare.item_keys(Vec::new()).is_empty());
    let none = bare.items(&[]);
    assert!(none.keys.is_empty() && none.one.is_none());
    assert!(!none.references && !none.decode_images);

    let mut frame = with_items(proto::Phase::Encode, 2);
    let stated = frame.encode.as_mut().expect("the case's items");
    stated.one = Some(key_form(Form::Plain("digest".to_owned())));
    (stated.references, stated.decode_images) = (true, true);
    let frame = Checked::new(frame).expect("items are no field out of its range");
    assert_eq!(
        (frame.phase(), frame.carries_items()),
        (Phase::Prefill, true)
    );
    assert!(frame.says_more());
    // Room for an encoder a key, made at the check.
    let room = frame.items.as_deref().map(|items| &items.placed);
    let room = room.expect("the frame's items");
    assert!(room.encoders.is_empty() && room.encoders.capacity() >= 2);
    let keys = frame.item_keys(Vec::new());
    let items = frame.items(&keys);
    assert_eq!(items.keys, [Identity::hashed(1), Identity::hashed(2)]);
    assert_eq!(items.one, Some(Identity::plain("digest")));
    assert!(items.references && items.decode_images);

    // A one key stated with no form is stated all the same: the items
    // are taken as one. A frame that states none has none.
    let mut formless = with_items(proto::Phase::Encode, 2);
    let stated = formless.encode.as_mut().expect("the case's items");
    stated.one = Some(proto::Identity { form: None });
    let formless = Checked::new(formless).expect("a key with no form is in range");
    let keys = formless.item_keys(Vec::new());
    assert_eq!(formless.items(&keys).one, Some(Identity::NONE));
    let each = with_items(proto::Phase::Encode, 2);
    let each = Checked::new(each).expect("items are no field out of its range");
    let keys = each.item_keys(Vec::new());
    assert!(each.items(&keys).one.is_none());

    let keyless = with_items(proto::Phase::Encode, 0);
    let keyless = Checked::new(keyless).expect("no key is in the field's range");
    assert_eq!(
        (keyless.phase(), keyless.carries_items()),
        (Phase::Prefill, true)
    );
    let keys = keyless.item_keys(Vec::new());
    let items = keyless.items(&keys);
    assert!(items.keys.is_empty() && items.one.is_none());
}

/// The hash form of a session key is the session's value under the
/// phase the facts state. An encode request's frame states the encode
/// phase and is decided at the prefill phase: its key is read back
/// under the phase it states, so the request made of it holds the key
/// its host holds, and the pair's session is that key's at the prefill
/// phase. A value made under the prefill phase reads back as another
/// key.
#[test]
fn a_hash_form_session_key_on_an_encode_frame_is_read_back_under_the_encode_phase() {
    let key = Identity::hashed(0x5e55_10f7);
    let session = |phase| {
        let facts = RequestFacts {
            model: "model-a",
            session_key: key,
            phase,
            ..RequestFacts::default()
        };
        facts.session().get()
    };
    let read = |value| {
        let mut frame = with_items(proto::Phase::Encode, 1);
        let facts = frame.facts.as_mut().expect("the full frame has facts");
        facts.session_key = Some(key_form(Form::Hash(value)));
        let frame = Checked::new(frame).expect("a frame the record holds");
        assert_eq!(frame.phase(), Phase::Prefill);
        let left = std::time::Duration::from_millis(1);
        made(&frame, &mut Lists::default(), left, |facts, _| {
            (facts.session_key.hash, facts.session().get())
        })
    };
    let pair = session(Phase::Prefill);
    assert_eq!(read(session(Phase::Encode)), (key.hash, pair));
    assert_ne!(read(pair).0, key.hash);
}

/// A key with no form reads as no key, and so do an empty text and a
/// hash of zero. Each such item keeps its place in the list, so the
/// answer's encoders stand in the keys' order.
#[test]
fn an_item_key_with_no_form_reads_as_no_key_at_its_place() {
    let mut frame = with_items(proto::Phase::Encode, 5);
    let stated = frame.encode.as_mut().expect("the case's items");
    stated.keys[1] = proto::Identity { form: None };
    stated.keys[2] = key_form(Form::Plain(String::new()));
    stated.keys[3] = key_form(Form::Hash(0));
    let frame = Checked::new(frame).expect("no key is in the field's range");
    let keys = frame.item_keys(Vec::new());
    let none = Identity::NONE;
    let read = [Identity::hashed(1), none, none, none, Identity::hashed(5)];
    assert_eq!(keys, read);
}

/// The field is read on a frame of the encode phase and on no other. A
/// frame of another phase that carries it is refused by the field, with
/// a key or with none: no frame with items is decided without them.
#[test]
fn a_frame_of_another_phase_that_carries_items_is_refused_by_the_field() {
    for phase in [
        proto::Phase::Single,
        proto::Phase::Prefill,
        proto::Phase::Decode,
    ] {
        for keys in [0, 1] {
            let refused = Checked::new(with_items(phase, keys)).err();
            let field = Some(Refusal::OutOfRange("encode"));
            assert_eq!(refused, field, "{phase:?}, {keys} keys");
        }
    }
}

/// The contract bounds an encode request's keys: a frame of as many as
/// the bound is read, and a frame of one more is refused whole by the
/// field. Its list is never cut.
#[test]
fn a_frame_of_more_keys_than_the_contract_bounds_is_refused_whole() {
    assert_eq!(MAX_ITEMS, 256, "the bound the contract states");
    let most = Checked::new(with_items(proto::Phase::Encode, 256));
    let most = most.expect("the bound is in the field's range");
    assert_eq!(most.item_keys(Vec::new()).len(), 256);
    let refused = Checked::new(with_items(proto::Phase::Encode, 257)).err();
    assert_eq!(refused, Some(Refusal::OutOfRange("encode.keys")));
}

/// A decode asked again names its lease and states no items. A frame
/// that names a lease is read at the decode phase, whatever phase its
/// facts state, so one that carries items is a frame of another phase:
/// refused by the field, its facts of the encode phase or of any other,
/// with a key or with none.
#[test]
fn a_frame_that_names_a_lease_and_carries_items_is_refused_by_the_field() {
    for (phase, keys) in [
        (proto::Phase::Encode, 1),
        (proto::Phase::Encode, 0),
        (proto::Phase::Prefill, 1),
        (proto::Phase::Decode, 1),
    ] {
        let mut frame = with_items(phase, keys);
        frame.decode_of = Some(1 << 32 | 7);
        let refused = Checked::new(frame).err();
        let field = Some(Refusal::OutOfRange("encode"));
        assert_eq!(refused, field, "{phase:?}, {keys} keys");
    }
}

/// The spec's range for `view.max_block_hashes` ends where the frame's
/// bound does. A frame of that many hashes and an id of one byte is
/// read, five bytes under the bound, and one hash more carries it over.
#[test]
fn the_specs_highest_cap_of_hashes_is_what_one_frame_carries() {
    let frame = |hashes: u32| proto::DecideRequest {
        request_id: vec![1],
        facts: Some(proto::RequestFacts {
            block_hashes: (0..u64::from(hashes)).collect(),
            ..proto::RequestFacts::default()
        }),
        constraints: None,
        again: false,
        ..Default::default()
    };
    let at = frame(MAX_BLOCK_HASHES);
    assert_eq!(prost::Message::encoded_len(&at), MAX_FRAME - 5);
    let read = Checked::new(at).expect("a frame under the bound");
    assert_eq!(u32::try_from(read.hashes()), Ok(MAX_BLOCK_HASHES));
    let over = frame(MAX_BLOCK_HASHES + 1);
    assert_eq!(prost::Message::encoded_len(&over), MAX_FRAME + 3);
}

/// The wire's trace carries the veto the all-vetoed rule lifted, and
/// none for a pick no rule made.
#[test]
fn the_wire_carries_the_lifted_veto() {
    let steered = Trace {
        lifted: LiftedVeto::Overload,
        ..Trace::default()
    };
    assert_eq!(steered.wire().lifted, proto::LiftedVeto::Overload as i32);
    assert_eq!(
        Trace::default().wire().lifted,
        proto::LiftedVeto::None as i32
    );
}

fn traced() -> Trace {
    Trace {
        profile: 3,
        survivors: Survivors::counted(5),
        // The third entry is past the trace's length: not a part of it.
        top: [
            (WorkerId(31), 0.5),
            (WorkerId(32), 0.25),
            (WorkerId(33), 9.0),
        ],
        top_len: 2,
        seed: 41,
        generation: 6,
        ..Trace::default()
    }
}

fn wire_trace() -> proto::Trace {
    proto::Trace {
        profile: 3,
        survivors: Some(5),
        top: vec![
            proto::Scored {
                worker_id: 31,
                score: 0.5,
                ..proto::Scored::default()
            },
            proto::Scored {
                worker_id: 32,
                score: 0.25,
                ..proto::Scored::default()
            },
        ],
        seed: 41,
        generation: 6,
        ..proto::Trace::default()
    }
}

/// The decision an answer carries: its outcome, its trace and how many
/// leases ride it.
fn carried(answer: proto::DecideAnswer) -> (Outcome, Option<proto::Trace>, usize) {
    match answer.answer {
        Some(Answer::Decision(decision)) => (
            decision.outcome.expect("a decision has an outcome"),
            decision.trace,
            decision.leases.len(),
        ),
        other => panic!("a decision's answer: {other:?}"),
    }
}

#[test]
fn a_first_profile_pick_keeps_its_trace_bytes_and_an_early_shed_adds_one_field() {
    use prost::Message as _;

    let pick = Decision::Pick(Pick {
        trace: Trace {
            profile: 0,
            survivors: Survivors::counted(1),
            generation: 6,
            ..Trace::default()
        },
        ..Pick::default()
    });
    let shed = decision::shed(ShedReason::NoEndpoints, "wires", 0, 6);
    let encoded = |decision| {
        super::decision(decision)
            .trace
            .expect("a trace")
            .encode_to_vec()
    };
    assert_eq!(encoded(&pick), [0x10, 1, 0x28, 6]);
    assert_eq!(encoded(&shed), [0x10, 0, 0x28, 6, 0x70, 1]);
}

#[test]
fn a_pick_is_its_targets_in_order_with_the_trace_and_no_lease() {
    let mut targets = Targets::new();
    targets.push(Target {
        position: 4,
        id: WorkerId(51),
        dp_rank: 2,
        pair: Some(Pair {
            decode_position: 1,
            decode_id: WorkerId(52),
            decode_dp_rank: 3,
            ..Pair::default()
        }),
    });
    targets.push(Target {
        position: 0,
        id: WorkerId(53),
        dp_rank: 0,
        pair: None,
    });
    let pick = Decision::Pick(Pick {
        targets,
        trace: traced(),
        ..Pick::default()
    });
    let answer = decided(vec![1, 2], 9, &pick, None, None);
    assert_eq!(
        (answer.request_id.as_slice(), answer.term),
        (&[1, 2][..], 9)
    );
    let (outcome, trace, leases) = carried(answer);
    assert_eq!(
        outcome,
        Outcome::Pick(proto::Pick {
            targets: vec![
                proto::Target {
                    id: 51,
                    dp_rank: 2,
                    pair: Some(proto::Pair {
                        decode_id: 52,
                        decode_dp_rank: 3
                    }),
                },
                proto::Target {
                    id: 53,
                    dp_rank: 0,
                    pair: None
                },
            ],
        })
    );
    assert_eq!(trace, Some(wire_trace()));
    assert_eq!(leases, 0);
}

/// An encode request's pick carries where its items go beside the
/// decision, as its decider handed it over: the list leaves in the
/// frame as it was filled, with the word of the one key. No lease
/// stands for an encoder. A pick that hands none carries no such field.
#[test]
fn an_encode_requests_pick_carries_each_items_encoder_beside_the_decision() {
    let mut targets = Targets::new();
    targets.push(Target {
        position: 4,
        id: WorkerId(51),
        dp_rank: 0,
        pair: Some(Pair {
            decode_position: 1,
            decode_id: WorkerId(52),
            ..Pair::default()
        }),
    });
    let pick = Decision::Pick(Pick {
        targets,
        trace: traced(),
        ..Pick::default()
    });
    let named = |id, dp_rank| proto::Target {
        id,
        dp_rank,
        pair: None,
    };
    let placed = proto::EncodePlaced {
        encoders: vec![named(61, 1), named(62, 0), named(61, 1)],
        one: true,
    };
    let items = Items {
        stated: proto::EncodeItems::default(),
        placed: placed.clone(),
    };
    let handed = Some(ItemsPlaced(Box::new(items)));
    let answer = decided(vec![1], 9, &pick, None, handed);
    assert_eq!(answer.encode, Some(placed));
    let (_, _, leases) = carried(answer);
    assert_eq!(leases, 0, "no lease for an encoder");
    let plain = decided(vec![1], 9, &pick, None, None);
    assert_eq!(plain.encode, None);
}

/// A decision that drew counted no survivor, and the wire says so in
/// the contract's two words: `survivors` stays unset, which is no
/// count and no zero, and `drew_first` is set. The trace's other facts
/// leave as a counted trace's do.
#[test]
fn a_drawn_decisions_trace_says_drew_first_and_carries_no_count() {
    let mut targets = Targets::new();
    targets.push(Target {
        position: 0,
        id: WorkerId(53),
        dp_rank: 0,
        pair: None,
    });
    let pick = Decision::Pick(Pick {
        targets,
        trace: Trace {
            survivors: Survivors::NOT_COUNTED,
            ..traced()
        },
        ..Pick::default()
    });
    let (_, trace, _) = carried(decided(Vec::new(), 1, &pick, None, None));
    let drawn = proto::Trace {
        survivors: None,
        drew_first: true,
        ..wire_trace()
    };
    assert_eq!(trace, Some(drawn));
}

#[test]
fn a_shed_names_its_reason_and_its_stage_and_a_ticket_its_wait() {
    for (record, wire) in [
        (ShedReason::Saturated, proto::ShedReason::Saturated),
        (ShedReason::FlowLimit, proto::ShedReason::FlowLimit),
        (ShedReason::NoEndpoints, proto::ShedReason::NoEndpoints),
        (
            ShedReason::PolicyDeclined,
            proto::ShedReason::PolicyDeclined,
        ),
        (ShedReason::Deadline, proto::ShedReason::Deadline),
    ] {
        let shed = Decision::Shed(Shed {
            reason: record,
            emptied_by: "healthy",
            retry_after_secs: 7,
            wait_allowed: true,
            side: None,
            trace: traced(),
        });
        let (outcome, trace, leases) = carried(decided(Vec::new(), 1, &shed, None, None));
        assert_eq!(
            outcome,
            Outcome::Shed(proto::Shed {
                reason: wire.into(),
                emptied_by: "healthy".to_owned(),
                retry_after_secs: 7,
                wait_allowed: true,
            }),
            "{record:?}"
        );
        assert_eq!((trace, leases), (Some(wire_trace()), 0));
    }
    let queue = Decision::Queue(Queue {
        ticket: 61,
        max_wait_micros: 62,
        trace: traced(),
    });
    let (outcome, trace, leases) = carried(decided(Vec::new(), 1, &queue, None, None));
    assert_eq!(
        outcome,
        Outcome::Queue(proto::Queue {
            ticket: 61,
            max_wait_micros: 62
        })
    );
    assert_eq!((trace, leases), (Some(wire_trace()), 0));
}

#[test]
fn a_credit_frame_carries_its_term_and_its_grant_and_no_requests_id() {
    assert_eq!(
        credit(3, 64),
        proto::DecideAnswer {
            request_id: Vec::new(),
            term: 3,
            answer: Some(Answer::Credit(proto::Credit { window: 64 })),
            encode: None,
        }
    );
}

#[test]
fn a_busy_answer_carries_its_requests_id_its_term_and_its_reason_alone() {
    for reason in [
        Reason::Unspecified,
        Reason::QueueDepth,
        Reason::NoCredit,
        Reason::BudgetGoneOnArrival,
        Reason::NotLeader,
    ] {
        assert_eq!(
            busy(vec![5], 3, reason),
            proto::DecideAnswer {
                request_id: vec![5],
                term: 3,
                answer: Some(Answer::Busy(proto::Busy {
                    reason: reason.into(),
                    retry_after_micros: 0,
                })),
                encode: None,
            }
        );
    }
}
