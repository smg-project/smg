//! The overload switch of a document's bindings: bindings that differ on
//! it are refused, and a generation that enters states its word on the
//! holder.

use std::sync::Arc;

use smg_worker::{
    health::overload::OverloadThresholds, identity::basic::Worker, registry::WorkerRegistry,
    worker::WorkerStatus as ObjectStatus,
};

use super::{
    super::{states_on, Generations},
    bind, election, enforce, followed, generations, roll_back, sent, switched, switched_by, yaml,
    DIFFERS,
};
use crate::{
    document::{
        operator::{
            test_support::{standing, stock},
            DocumentManager, Format, SpecRejected,
        },
        spec::AppliedSpec,
    },
    engine::view::WorkerId,
    service::elector::Fence,
    store::{MemoryStore, Registrant, WorkerFacts, WorkerStatus},
    worker_host::{Members, Objects},
};

/// A controller's door refuses a document whose bindings differ on the
/// overload switch, by the path of the first binding that differs and
/// in the words a start ends on. An apply leaves the standing
/// generation, and a dry run answers what the apply would. Bindings
/// that say one word pass, by their blocks or by none.
#[test]
fn a_document_whose_bindings_differ_on_the_overload_switch_is_refused_in_its_words() {
    let (store, door) = generations();
    yaml(&door, switched(Some(false), Some(false))).expect("every binding off");
    yaml(&door, switched(None, None)).expect("no block");
    yaml(&door, switched(Some(true), None)).expect("one word, by a block and by none");
    let own = standing(&store);
    assert_eq!(own.0, 3);

    let refused = yaml(&door, switched(Some(false), None)).expect_err("they differ");
    assert_eq!(refused.to_string(), DIFFERS);
    let SpecRejected::Invalid(invalid) = &refused else {
        panic!("the pass refuses it: {refused:?}");
    };
    assert_eq!(invalid.field_path(), "bindings[1].overload.enabled");
    // The other way round, the second binding says off.
    let refused = yaml(&door, switched(None, Some(false))).expect_err("they differ");
    let other = DIFFERS.replace("true is not the false", "false is not the true");
    assert_eq!(refused.to_string(), other);
    assert_eq!(standing(&store), own, "the standing generation stays");

    let differs = switched(Some(false), None);
    let dry = door.dry_run("", differs.as_bytes(), Format::Yaml);
    let would = dry.expect("no guard").would_run;
    assert_eq!(
        would.map_err(|refused| refused.to_string()),
        Err(DIFFERS.to_owned())
    );
    assert_eq!(standing(&store), own, "a dry run applies nothing");
}

/// The thresholds a start states, as the holder of these cases is
/// handed them.
const START: OverloadThresholds = OverloadThresholds {
    waiting_requests: Some(8),
    token_usage: Some(0.8),
};

/// A door that states its generations' overload switch on a holder
/// handed [`START`]: the store, the holder and the door, of a replica
/// alone over its lease.
fn stating() -> (Arc<MemoryStore>, Arc<Objects<Members>>, Generations) {
    stating_under(None)
}

/// [`stating`] with the door built under `fence`: a replica's under a
/// shared lease where one is handed.
fn stating_under(
    fence: Option<Arc<Fence>>,
) -> (Arc<MemoryStore>, Arc<Objects<Members>>, Generations) {
    let store = Arc::new(MemoryStore::new());
    let holder = Objects::new(Arc::new(WorkerRegistry::new())).with_overload_thresholds(START);
    let holder = Arc::new(holder);
    let door = Generations::stating(Arc::clone(&store), stock(), Arc::clone(&holder), fence);
    (store, holder, door)
}

/// An engine's worker that names no threshold, registered through the
/// holder as a control call registers one: its row.
fn register(store: &MemoryStore, holder: &Objects<Members>, url: &str) -> WorkerId {
    let facts = WorkerFacts {
        key: url.to_owned(),
        engine: url.to_owned(),
        models: vec!["m".to_owned()],
        dp_size: 1,
        ..WorkerFacts::default()
    };
    let written = holder.upsert(store, facts, Registrant::Control);
    written.expect("the store takes the registration").id()
}

/// The status the table holds for `row`.
fn told(store: &MemoryStore, row: WorkerId) -> WorkerStatus {
    store.get(row).expect("the row").status
}

/// The object that stands under `url`.
fn object(holder: &Objects<Members>, url: &str) -> Arc<dyn Worker> {
    holder.registry().get_by_url(url).expect("an object")
}

/// The thresholds the object under `url` states.
fn stated(holder: &Objects<Members>, url: &str) -> OverloadThresholds {
    object(holder, url).metadata().overload
}

/// A generation that enters by an apply states its overload switch on
/// the holder before the apply returns. Every binding off leaves a
/// worker that names no threshold none, and on again the start's. The
/// same word under another document lays no object again. Bindings
/// that differ are refused at the pass, so no word is stated. Two
/// entries in a row leave the rows of the last word, and a row that
/// comes behind them is built by it. A status that moved on the
/// standing object with no word to the table stands in the table when
/// the apply returns: this replica leads, and the lay settles the two.
#[test]
fn an_applied_document_states_its_overload_switch_on_the_holder() {
    let (a, b) = ("http://a:8000", "http://b:8000");
    let none = OverloadThresholds::default();
    let (store, holder, door) = stating();
    let row = register(&store, &holder, a);
    assert_eq!(stated(&holder, a), START, "the unbound start says on");
    let unbound = object(&holder, a);
    // On, by a block and by none: the word stands and nothing is laid.
    yaml(&door, switched(Some(true), None)).expect("one word");
    assert!(Arc::ptr_eq(&object(&holder, a), &unbound));

    // A health driver's move whose telling finds the object replaced.
    unbound.set_status(ObjectStatus::Ready);
    assert_eq!(told(&store, row), WorkerStatus::Pending);
    yaml(&door, switched(Some(false), Some(false))).expect("every binding off");
    assert_eq!(stated(&holder, a), none, "stated when the apply returns");
    assert_eq!(told(&store, row), WorkerStatus::Ready, "and settled");
    let off = object(&holder, a);
    assert_eq!(off.status(), ObjectStatus::Ready);
    // The same word under another document lays nothing.
    let other = switched_by("round-robin", Some(false), Some(false));
    yaml(&door, other).expect("off again");
    assert!(Arc::ptr_eq(&object(&holder, a), &off));

    // Bindings that differ are refused at the pass: no generation
    // enters, and the word and its objects stand.
    let other = switched_by("round-robin", Some(false), None);
    yaml(&door, other).expect_err("they differ");
    assert!(Arc::ptr_eq(&object(&holder, a), &off));

    // Two entries in a row: the rows of the last word.
    yaml(&door, switched(Some(false), Some(false))).expect("off");
    yaml(&door, switched(None, None)).expect("on");
    assert_eq!(stated(&holder, a), START);
    register(&store, &holder, b);
    assert_eq!(stated(&holder, b), START, "built by the word that stands");
    assert_eq!(standing(&store).0, 5);
}

/// A document whose bindings differ on the overload switch does not
/// pass a controller's door. Where one stands past the pass, under a
/// door with the stock pass, the hook keeps the start's default on,
/// either binding first: it reads such a document as the unbound start.
#[test]
fn a_document_past_the_pass_whose_bindings_differ_keeps_the_starts_default() {
    let a = "http://a:8000";
    let none = OverloadThresholds::default();
    let store = Arc::new(MemoryStore::new());
    let holder = Objects::new(Arc::new(WorkerRegistry::new())).with_overload_thresholds(START);
    let holder = Arc::new(holder);
    let stock_pass = AppliedSpec::validate_digested;
    let door = DocumentManager::validating(Arc::clone(&store), stock(), Vec::new(), stock_pass);
    let entered = states_on(Arc::clone(&holder), None);
    let door = Generations::over(door.telling(entered), None);
    register(&store, &holder, a);
    yaml(&door, switched(Some(false), Some(false))).expect("every binding off");
    assert_eq!(stated(&holder, a), none);
    yaml(&door, switched(Some(false), None)).expect("the stock pass takes it");
    assert_eq!(stated(&holder, a), START, "the default is back on");
    let differ = object(&holder, a);
    yaml(&door, switched(None, Some(false))).expect("the other one first");
    assert!(Arc::ptr_eq(&object(&holder, a), &differ), "on stands");
}

/// A spec store's bind, an enforce and a rollback each enter a
/// generation, and each states its overload switch on the holder. A
/// start over a store whose newest generation switches the default off
/// states the word at the bind: a worker that stood is laid again, and
/// one registered behind the bind is built once, with none.
#[test]
fn a_spec_stores_bind_and_its_writes_state_the_overload_switch_on_the_holder() {
    let (a, b) = ("http://a:8000", "http://b:8000");
    let none = OverloadThresholds::default();
    let dir = tempfile::tempdir().expect("a temp dir");
    // An earlier run left generation 1 on file: every binding off.
    {
        let (_, earlier) = generations();
        bind(&earlier, dir.path());
        enforce(&earlier, switched(Some(false), Some(false))).expect("on file");
    }
    let (store, holder, door) = stating();
    register(&store, &holder, a);
    assert_eq!(stated(&holder, a), START);
    bind(&door, dir.path());
    assert_eq!(standing(&store).0, 1, "the newest generation on file");
    assert_eq!(stated(&holder, a), none, "stated at the bind");
    register(&store, &holder, b);
    assert_eq!(stated(&holder, b), none, "built once, by the word");
    // An enforce's write: every binding on.
    enforce(&door, switched(None, None)).expect("a changed document");
    assert_eq!((stated(&holder, a), stated(&holder, b)), (START, START));
    // A rollback to the generation on file that switches off.
    roll_back(&door, 1).expect("on file");
    assert_eq!((stated(&holder, a), stated(&holder, b)), (none, none));
}

/// A standby's followed generation enters through the same door: its
/// leader's word is stated on the standby's holder when it lands. A
/// worker laid again keeps its id and its count in flight. The standby
/// is under a shared lease and has not led, so it writes no row: a
/// status that moved on its object with no word to the table gives way
/// to the one its row holds.
#[test]
fn a_followed_generation_states_its_overload_switch_on_the_holder() {
    let a = "http://a:8000";
    let none = OverloadThresholds::default();
    let (_, leader) = generations();
    yaml(&leader, switched(Some(false), Some(false))).expect("every binding off");
    let elector = election();
    let (store, holder, standby) = stating_under(Some(elector.fence()));
    let row = register(&store, &holder, a);
    let revision = store.get(row).expect("the row").revision;
    holder.connect_row(&store, row, revision);
    assert_eq!(told(&store, row), WorkerStatus::Ready);
    let before = object(&holder, a);
    before.increment_load();
    before.set_status(ObjectStatus::NotReady);
    let id = holder.registry().get_id_by_url(a);
    followed(&standby, &sent(&leader)).expect("above the unbound one");
    let after = object(&holder, a);
    assert!(!Arc::ptr_eq(&before, &after), "laid again");
    assert_eq!(stated(&holder, a), none);
    assert_eq!(holder.registry().get_id_by_url(a), id, "the same id");
    assert_eq!(told(&store, row), WorkerStatus::Ready, "no row is written");
    assert_eq!((after.status(), after.load()), (ObjectStatus::Ready, 1));
    // Its leader's next generation says on again.
    yaml(&leader, switched(None, None)).expect("a changed document");
    followed(&standby, &sent(&leader)).expect("above the first");
    assert_eq!(stated(&holder, a), START);
}

/// A door that was handed no hook states nothing: its generation
/// enters, and a holder beside its store keeps the word it had.
#[test]
fn a_door_with_no_hook_states_no_word_on_a_holder() {
    let a = "http://a:8000";
    let (store, door) = generations();
    let holder: Objects<Members> =
        Objects::new(Arc::new(WorkerRegistry::new())).with_overload_thresholds(START);
    register(&store, &holder, a);
    yaml(&door, switched(Some(false), Some(false))).expect("every binding off");
    assert_eq!(standing(&store).0, 1);
    assert_eq!(stated(&holder, a), START, "nobody stated the word");
}
