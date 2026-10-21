//! What a request declares about its calls costs nothing to hold: [`Declared::shared`] over the
//! request's tools allocates nothing, however many tools the request declares and however large
//! their schemas, and a parameter's schema is read in place. And a steady stream costs nothing per
//! delta: a delta read as text whole allocates nothing, its event borrows it and the buffer is the
//! stream's, and the engine's hold-back of a possible marker costs the delta after a held byte one
//! copy, the held bytes joined to it. A separate executable, so it can declare the counting
//! allocator the pins need.

#![expect(unsafe_code, reason = "a global allocator that counts allocations")]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    hint::black_box,
    sync::Arc,
};

use openai_protocol::common::{Function, Tool};
use serde_json::json;
use symphony::{formats::Table, Declared, Engine, EngineFinish, Events, Input, Parser};

struct Counting;

thread_local! {
    /// The allocations the thread made: each test counts its own, whichever run beside it. A
    /// constant without a destructor, so the count is reached from the allocator without one.
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn count_one() {
    ALLOCATIONS.with(|count| count.set(count.get() + 1));
}

fn allocations() -> usize {
    ALLOCATIONS.with(Cell::get)
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_one();
        // SAFETY: the layout is the caller's, handed to the system allocator unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_one();
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocations_during(work: impl FnOnce()) -> usize {
    let before = allocations();
    work();
    allocations() - before
}

/// Ten function tools whose schemas run to about two kilobytes each.
fn tools() -> Arc<[Tool]> {
    let filler =
        "a description long enough to make the schema weigh what a real one does ".repeat(24);
    (0..10)
        .map(|k| Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: format!("lookup_{k}"),
                description: Some(filler.clone()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "the record key"},
                        "fields": {"type": "array", "items": {"type": "string"}},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                        "filter": {
                            "type": "object",
                            "properties": {
                                "since": {"type": "string", "format": "date-time"},
                                "status": {"type": "string", "enum": ["open", "closed", "any"]}
                            }
                        }
                    },
                    "required": ["key"]
                }),
                strict: None,
                extra: Default::default(),
            },
        })
        .collect()
}

#[test]
fn holding_what_a_request_declares_allocates_nothing() {
    let tools = tools();
    assert!(
        tools
            .iter()
            .all(|tool| serde_json::to_string(tool).map_or(0, |json| json.len()) > 2_000),
        "the schemas are meant to be large"
    );

    let before = allocations();
    let declared = black_box(Declared::shared(Arc::clone(&tools)));
    let building = allocations() - before;
    assert_eq!(building, 0, "building Declared allocated {building} times");

    let names: Vec<String> = (0..10).map(|k| format!("lookup_{k}")).collect();
    let mut found = 0;
    let reading = allocations_during(|| {
        for name in &names {
            if let Some(schemas) = declared.parameter(name, "limit") {
                black_box(schemas);
                found += 1;
            }
        }
    });
    assert_eq!(found, 10, "every function declares the parameter");
    assert_eq!(
        reading, 0,
        "reading ten parameters' schemas allocated {reading} times"
    );
    assert_eq!(
        Arc::strong_count(&tools),
        2,
        "the tools are shared, not copied"
    );
}

/// Feeds `text` as one delta into the stream's buffer, reads the events in place and recycles
/// the buffer, as a stream's reader does; the allocations it took.
fn allocations_for_delta(engine: &mut Engine, spare: &mut Events<'static>, text: &str) -> usize {
    let mut fed = Ok(());
    let mut out = std::mem::take(spare);
    let allocations = allocations_during(|| {
        fed = engine.feed(
            Input::Delta {
                token_ids: &[],
                text,
                spans: &[],
            },
            &mut out,
        );
        black_box(out.as_slice());
    });
    *spare = out.recycle();
    assert!(fed.is_ok(), "the delta is taken: {fed:?}");
    allocations
}

#[test]
fn a_delta_read_as_text_allocates_nothing_and_one_after_a_held_byte_one_copy() {
    let mut engine = Engine::new(Table::Qwen3.format(), Declared::of(&[]));
    let mut out = Events::new();
    engine
        .feed(
            Input::Prompt {
                token_ids: &[],
                text: "<|im_start|>assistant\n",
            },
            &mut out,
        )
        .expect("prompt");
    black_box(out.drain());
    // The stream's first deltas: the content begins, the event list and the scanner's buffers
    // take their capacity. Each `<` at the end of a delta may begin a marker, so it is held and
    // the next delta is scanned after it.
    for text in ["The answer: ", "a<", "b. ", "c<", "d. "] {
        allocations_for_delta(&mut engine, &mut out, text);
    }
    let whole = allocations_for_delta(&mut engine, &mut out, "Steady text. ");
    let holding = allocations_for_delta(&mut engine, &mut out, "e<");
    let after_a_hold = allocations_for_delta(&mut engine, &mut out, "f. ");
    assert_eq!(
        whole, 0,
        "a delta read as text whole allocated {whole} times: its event borrows it and the buffer \
         is the stream's"
    );
    assert_eq!(
        (holding, after_a_hold),
        (whole, whole + 1),
        "a delta that ends in a held byte took {holding} allocations and the one after it \
         {after_a_hold}, where a delta read as text whole takes {whole}; the one after a hold \
         pays one copy, the held bytes joined to it, which no event can borrow"
    );
    engine
        .feed(
            Input::End {
                finish: EngineFinish::Stop,
            },
            &mut out,
        )
        .expect("end");
    black_box(out.drain());
}
