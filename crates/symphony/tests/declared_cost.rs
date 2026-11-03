//! What a request declares about its calls costs nothing to hold: [`Declared::shared`] over the
//! request's tools allocates nothing, however many tools the request declares and however large
//! their schemas, and a parameter's schema is read in place. And a steady stream costs nothing per
//! delta: a delta read as text whole allocates nothing, its event borrows it and the buffer is the
//! stream's, and the engine's hold-back of a possible marker costs the delta after a held byte one
//! copy, the held bytes joined to it; inside a streamed call, a delta of the call's arguments the
//! JSON syntax passes on whole allocates nothing of its own either, its fragment borrows the delta
//! (the assembler's own copy of the call's bytes still grows by doubling as the value lengthens,
//! the one cost that remains, outside the pin). What a tagged syntax still pays is pinned too: a
//! delta of a string argument costs two strings today, the escaped text and its source, a count
//! held so that the change which removes them flips the pin in the same commit. A separate
//! executable, so it can declare the counting allocator the pins need.

#![expect(unsafe_code, reason = "a global allocator that counts allocations")]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    borrow::Cow,
    cell::Cell,
    hint::black_box,
    sync::Arc,
};

use openai_protocol::common::{Function, Tool};
use serde_json::json;
use symphony::{formats::Table, Declared, Engine, EngineFinish, Event, Events, Input, Parser};

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
                parameters: Some(json!({
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
                })),
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

#[test]
fn a_delta_of_a_calls_arguments_read_whole_allocates_nothing() {
    let mut engine = Engine::new(Table::Qwen3.format(), Declared::shared(tools()));
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
    // The call's opener and name: the start's events own the id, the name and the bytes before
    // the arguments; the event list, the scanner's buffers, the object's outline and the judge's
    // own buffers take their capacity as the first members, the first container and the first
    // number arrive.
    for text in [
        "Let me look that up.\n",
        "<tool_call>\n",
        "{\"name\": \"lookup_0\", ",
        "\"arguments\": {\"key\": ",
        "\"k7-1\", \"fields\": [\"a\", \"b\", \"c\"], ",
        "\"limit\": 10, \"note\": \"the ",
    ] {
        allocations_for_delta(&mut engine, &mut out, text);
    }
    // The assembler keeps its own copy of the call's bytes in a buffer that grows by doubling, so
    // a long run of the string value sets that buffer's length and the short delta after it is
    // where the doubling lands: the measured deltas below stay a few hundred bytes within the room
    // that leaves, and the pin reads the fragments' cost, not the buffer's growth (which is what
    // remains of an arguments delta's cost, outside the pin).
    let long_run = "x".repeat(512);
    allocations_for_delta(&mut engine, &mut out, &long_run);
    allocations_for_delta(&mut engine, &mut out, "record ");
    // Deltas inside a string value, each a fragment the JSON syntax passes on as the model wrote
    // it: the fragment is a slice of the delta, as a content delta's event is, json and source
    // alike. Each delta is fed twice: the first feed is where the fragment's borrow is read, the
    // second, of the same bytes, where the count is read.
    for text in ["record is ", "open and ", "current, as ", "it always was"] {
        let mut events = std::mem::take(&mut out);
        engine
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text,
                    spans: &[],
                },
                &mut events,
            )
            .expect("the delta is taken");
        assert!(
            matches!(
                events.as_slice(),
                [Event::ToolCallArguments { json: Cow::Borrowed(json), source, .. }]
                    if *json == text && matches!(source.text, Cow::Borrowed(bytes) if bytes == text)
            ),
            "the arguments delta {text:?} gave {:?}: one fragment, borrowed whole",
            events.as_slice()
        );
        out = events.recycle();
        let again = allocations_for_delta(&mut engine, &mut out, text);
        assert_eq!(
            again, 0,
            "the arguments delta {text:?} allocated {again} times: its fragment borrows the delta \
             and the buffer is the stream's"
        );
    }
    for text in ["\", \"filter\": {\"status\": \"open\"}}\n", "</tool_call>"] {
        allocations_for_delta(&mut engine, &mut out, text);
    }
    engine
        .feed(
            Input::End {
                finish: EngineFinish::Stop,
            },
            &mut out,
        )
        .expect("end");
    assert!(
        matches!(
            out.as_slice().last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ),
        "the output ends with its one call counted: {:?}",
        out.as_slice()
    );
}

/// A tagged syntax writes the call's JSON itself, so every streamed piece of a string argument
/// is written as its escaped text and carried as its own source: two strings per delta today,
/// where the JSON syntax's fragments borrow the delta. The pin holds that count, so the change
/// that lets the tagged assembler's pieces borrow the delta flips it to zero in the same commit,
/// and a third copy cannot arrive unnoticed. The attributed, keyed and XML assemblers write their
/// strings through the same object.
#[test]
fn a_delta_of_a_tagged_calls_string_argument_costs_two_strings_today() {
    let mut engine = Engine::new(Table::Qwen3Tagged.format(), Declared::shared(tools()));
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
    // The call's opener, the function and the parameter tags, and the value's first bytes: the
    // start's events, the member's opening fragment and the buffers take their capacity here.
    for text in [
        "Let me look that up.\n",
        "<tool_call>\n",
        "<function=lookup_0>\n",
        "<parameter=key>\n",
        "k7-1",
    ] {
        allocations_for_delta(&mut engine, &mut out, text);
    }
    // The string argument streams in deltas of four bytes, 2,000 bytes in all; the first few are
    // warm-up, the rest are the pin. No delta ends in a byte that could begin a tag or in the
    // newline the template puts before the closing tag, so each streams whole.
    let piece = "abcd";
    let mut costs = Vec::with_capacity(500);
    for _ in 0..500 {
        costs.push(allocations_for_delta(&mut engine, &mut out, piece));
    }
    let steady = &costs[4..];
    assert!(
        steady.iter().all(|&cost| cost == 2),
        "the deltas of a tagged call's string argument allocated {:?} times (the first four \
         {:?}): today each piece is written as one escaped JSON string and carried as one source \
         string, two allocations per delta where the JSON syntax's fragment borrows the delta; \
         pieces that borrow the delta take this to 0",
        steady
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        &costs[..4]
    );
    // One more piece, read for its shape: one fragment, the escaped text and its source both
    // owned, which is where the two strings go.
    let mut events = std::mem::take(&mut out);
    engine
        .feed(
            Input::Delta {
                token_ids: &[],
                text: piece,
                spans: &[],
            },
            &mut events,
        )
        .expect("the delta is taken");
    assert!(
        matches!(
            events.as_slice(),
            [Event::ToolCallArguments { json: Cow::Owned(json), source, .. }]
                if json == piece && matches!(&source.text, Cow::Owned(bytes) if bytes == piece)
        ),
        "the arguments delta {piece:?} gave {:?}: one fragment, its text and its source each a \
         string of its own",
        events.as_slice()
    );
    out = events.recycle();
    for text in ["\n</parameter>\n", "</function>\n", "</tool_call>"] {
        allocations_for_delta(&mut engine, &mut out, text);
    }
    engine
        .feed(
            Input::End {
                finish: EngineFinish::Stop,
            },
            &mut out,
        )
        .expect("end");
    assert!(
        matches!(
            out.as_slice().last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ),
        "the output ends with its one call counted: {:?}",
        out.as_slice()
    );
}
