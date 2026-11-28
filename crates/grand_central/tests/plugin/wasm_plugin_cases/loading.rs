//! What loads and what is refused: the guests' canonical layout, both worlds,
//! the imports and exports a load turns away, the ABI's pin and range, and
//! the sandbox's caps on memory and tables.

use std::sync::Arc;

use smg_grand_central::{
    engine::Decision,
    plugins::wasm::{Abi, LoadError, Plugin, PluginSpec, Runtime, World},
};
use wit_component::{ComponentEncoder, StringEncoding};
use wit_parser::Resolve;

use crate::{
    bound_clock, decide, load, loaded, made_again, off_cpu_since, plugin_engine, pool, runtime,
    spec,
    support::wasm_guest::{self as guest, canonical, mode},
    the_close_rebuilt, tried, INIT_FRAME, TRIES,
};

/// Loads `spec` and returns what the load said. Its `init` ends at a
/// deadline on the wall clock, so a runner that holds the loading thread for
/// the init frame has the host end an `init` that did nothing. Such a load
/// is made again, [`TRIES`] loads at most: its thread was off the CPU half
/// the frame. A guest that spends the frame itself has the CPU, and its load
/// ends here as it did.
fn load_unheld(runtime: &Arc<Runtime>, spec: PluginSpec) -> Result<Arc<Plugin>, LoadError> {
    let mut loads = 0;
    loop {
        let (wall, cpu) = (std::time::Instant::now(), bound_clock());
        let load = Plugin::load(runtime, spec.clone());
        loads += 1;
        let held = matches!(
            &load,
            Err(LoadError::Trap { reason, .. }) if reason.ends_with("wasm trap: interrupt")
        ) && off_cpu_since(wall, cpu) >= INIT_FRAME / 2;
        if !held || loads == TRIES {
            return load;
        }
        made_again("a load", "its thread was off the CPU half the init frame");
    }
}

#[test]
fn the_guests_offsets_match_the_canonical_layout() {
    guest::check_canonical_layout();
    assert_eq!(canonical::CANDIDATE_SIZE % 8, 0);
}

#[test]
fn both_worlds_load_and_the_load_instantiates_on_the_applying_thread() {
    let runtime = runtime();
    let record = load(
        &runtime,
        "record",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let packed = load(
        &runtime,
        "packed",
        guest::packed_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    assert_eq!(record.world(), World::Record);
    assert_eq!(packed.world(), World::Packed);
    assert_eq!(record.abi(), Abi::V1_2);
    assert_eq!(packed.abi(), Abi::V1_2);
    assert!(record.is_warm() && packed.is_warm());
    assert_eq!(record.stats().instantiations, 1);
    assert_eq!(record.name(), "record");
}

#[test]
fn a_wasi_import_is_refused_at_load() {
    let runtime = runtime();
    let err = Plugin::load(
        &runtime,
        spec("wasi", guest::wasi_importing_guest(), mode::LEAST_IN_FLIGHT),
    )
    .expect_err("refused");
    assert!(
        matches!(&err, LoadError::Wasi { import, .. } if import.starts_with("wasi:clocks/monotonic-clock")),
        "{err}"
    );
}

#[test]
fn a_component_of_another_world_has_no_policy_world() {
    let mut resolve = Resolve::new();
    let package = resolve
        .push_str("empty.wit", "package test:empty;\nworld nothing {}")
        .expect("the world parses");
    let world = resolve
        .select_world(&[package], Some("nothing"))
        .expect("the world exists");
    let mut module = wat::parse_str("(module)").expect("assembles");
    wit_component::embed_component_metadata(&mut module, &resolve, world, StringEncoding::UTF8)
        .expect("embeds");
    let component = ComponentEncoder::default()
        .module(&module)
        .expect("matches")
        .validate(true)
        .encode()
        .expect("encodes");
    let err = Plugin::load(&runtime(), spec("empty", component, 0)).expect_err("refused");
    assert_eq!(
        err,
        LoadError::NoWorld {
            name: "empty".into()
        }
    );
}

#[test]
fn a_component_with_an_async_export_is_refused_at_its_compile() {
    // The runtime's own engine keeps no state for concurrent calls, so a
    // component that needs the async component support does not compile. An
    // engine that kept it would compile this one, and the loader would
    // refuse it later, for having no policy world.
    let needs_it = wat::parse_str(
        r#"(component
            (core module $m
                (func (export "run") (result i32) i32.const 0)
                (func (export "cb") (param i32 i32 i32) (result i32) i32.const 0))
            (core instance $i (instantiate $m))
            (type $run (func async))
            (func (export "run") (type $run)
                (canon lift (core func $i "run") async (callback (core func $i "cb")))))"#,
    )
    .expect("the component assembles");
    let err = Plugin::load(&runtime(), spec("async-export", needs_it, 0)).expect_err("refused");
    assert!(matches!(err, LoadError::Compile { .. }), "{err}");
}

#[test]
fn the_abi_pin_the_memory_floor_the_init_failure_and_bad_bytes_are_refused() {
    let runtime = runtime();
    let mut pinned = spec("pinned", guest::record_guest(), mode::LEAST_IN_FLIGHT);
    pinned.abi = Some(Abi::of(1, 3));
    let err = Plugin::load(&runtime, pinned).expect_err("refused");
    assert!(
        matches!(
            err,
            LoadError::AbiPin {
                found: Abi::V1_2,
                ..
            }
        ),
        "{err}"
    );

    let mut no_memory = spec("tiny", guest::record_guest(), mode::LEAST_IN_FLIGHT);
    no_memory.memory_mib = 0;
    assert!(matches!(
        Plugin::load(&runtime, no_memory).expect_err("refused"),
        LoadError::Memory { .. }
    ));

    let mut unfuelled = spec("unfuelled", guest::record_guest(), mode::LEAST_IN_FLIGHT);
    unfuelled.fuel_per_micro = 0;
    assert!(matches!(
        Plugin::load(&runtime, unfuelled).expect_err("refused"),
        LoadError::Fuel { .. }
    ));

    let err = load_unheld(
        &runtime,
        spec("refusing", guest::record_guest(), mode::REFUSE_INIT),
    )
    .expect_err("init failed");
    assert_eq!(
        err,
        LoadError::Init {
            name: "refusing".into(),
            reason: "refused".into()
        }
    );

    let err = Plugin::load(&runtime, spec("bytes", b"not wasm".to_vec(), 0)).expect_err("refused");
    assert!(matches!(err, LoadError::Compile { .. }), "{err}");

    let specs = [
        spec("a", Vec::new(), 0),
        spec("b", Vec::new(), 0),
        spec("c", Vec::new(), 0),
    ];
    assert_eq!(
        Plugin::check_memory_budget(&specs, 47),
        Err(LoadError::MemoryBudget {
            asked: 48,
            budget: 47
        })
    );
    assert_eq!(Plugin::check_memory_budget(&specs, 48), Ok(()));
}

#[test]
fn the_memory_cap_is_the_specs_and_a_growth_past_it_traps() {
    let runtime = runtime();
    let capped = load(&runtime, "capped", guest::record_guest(), mode::GROW_32_MIB);
    decide(&plugin_engine(&capped), &pool(4), 1);
    assert_eq!(
        capped.stats().traps,
        1,
        "16 MiB by default: 32 MiB more traps"
    );

    let mut roomy = spec("roomy", guest::record_guest(), mode::GROW_32_MIB);
    roomy.memory_mib = 64;
    let roomy = loaded(&runtime, roomy);
    decide(&plugin_engine(&roomy), &pool(4), 1);
    let stats = roomy.stats();
    assert_eq!(
        (stats.traps, stats.answered),
        (0, 4),
        "under 64 MiB the growth fits"
    );
}

/// The sandbox is one linear memory under the cap and tables of a bounded
/// size: a second memory is refused at instantiation, a table grown past the
/// cap traps.
#[test]
fn a_second_memory_and_a_table_past_the_cap_are_refused() {
    let runtime = runtime();
    let err = Plugin::load(
        &runtime,
        spec(
            "two-memories",
            guest::two_memory_guest(),
            mode::LEAST_IN_FLIGHT,
        ),
    )
    .expect_err("a second memory is refused");
    assert!(matches!(err, LoadError::Trap { .. }), "{err}");

    // The close's rebuild is an `init`, on the wall clock.
    tried(|made| {
        let plugin = load(
            &runtime,
            "table-grower",
            guest::record_guest(),
            mode::GROW_TABLE,
        );
        let engine = plugin_engine(&plugin);
        let (wall, cpu) = (std::time::Instant::now(), bound_clock());
        let decision = decide(&engine, &pool(4), 1);
        let held = off_cpu_since(wall, cpu);
        assert!(matches!(decision, Decision::Pick(_)));
        let stats = plugin.stats();
        assert_eq!(
            (stats.traps, stats.answered),
            (1, 1),
            "the constructor answered, the filter trapped"
        );
        the_close_rebuilt("a table past the cap", &plugin, held, made)
    });

    let err = Plugin::load(
        &runtime,
        spec(
            "four-tables",
            guest::four_table_guest(),
            mode::LEAST_IN_FLIGHT,
        ),
    )
    .expect_err("a fifth table is refused");
    assert!(matches!(err, LoadError::Trap { .. }), "{err}");
}

/// A module built against an older minor of the world is refused by the
/// host's range, both versions named, before anything of it is linked, in
/// either world. Its shapes are today's here, so nothing but its version
/// stops it.
#[test]
fn a_module_of_an_older_minor_is_refused_by_the_range() {
    let runtime = runtime();
    for world in ["policy", "policy-packed"] {
        let older = spec(
            "older",
            guest::older_minor_guest(world),
            mode::LEAST_IN_FLIGHT,
        );
        let err = Plugin::load(&runtime, older).expect_err("refused");
        assert!(matches!(err, LoadError::Abi { .. }), "{world}: {err}");
        assert_eq!(
            err.to_string(),
            "plugin older: targets smg:policy@1.1.0; this host accepts 1.2",
            "{world}"
        );
    }
}
