//! The test guests of the WebAssembly plugin host: two small policies
//! written as WebAssembly text, assembled and wrapped into components of the
//! committed `smg:policy` worlds at test time (no binary fixture, no cross
//! target). Both pick the candidate with the fewest requests in flight; a
//! mode byte in `init`'s config makes one misbehave on purpose, so the
//! host's failure rules can be proven.
//!
//! The record guest reads the canonical ABI layout of the constructor's
//! arguments; [`check_canonical_layout`] holds its offsets against
//! wit-parser's own size computation, so a change to the WIT fails here by
//! name before it fails inside the guest.

use std::path::PathBuf;

use wit_component::{ComponentEncoder, StringEncoding};
use wit_parser::{PackageId, Resolve, SizeAlign, TypeDefKind};

/// `init` modes, by the first config byte.
pub mod mode {
    /// The honest policy.
    pub const LEAST_IN_FLIGHT: u8 = 0;
    /// `filter` traps.
    pub const TRAP_IN_FILTER: u8 = 1;
    /// `score` spins until the fuel runs out.
    pub const SPIN_IN_SCORE: u8 = 2;
    /// `filter` declines as not applicable.
    pub const DECLINE_FILTER: u8 = 3;
    /// `filter` names a position past the pool.
    pub const FILTER_OUT_OF_RANGE: u8 = 4;
    /// `score` answers NaN.
    pub const NAN_SCORE: u8 = 5;
    /// `filter` keeps the even positions only.
    pub const KEEP_EVEN: u8 = 6;
    /// `pick` names a filtered-out row.
    pub const PICK_FILTERED_OUT: u8 = 7;
    /// `pick` declines.
    pub const DECLINE_PICK: u8 = 8;
    /// `score` declines as not applicable.
    pub const DECLINE_SCORE: u8 = 25;
    /// `filter` names a position twice.
    pub const FILTER_DUPLICATE: u8 = 9;
    /// `filter` grows the linear memory by 32 MiB first.
    pub const GROW_32_MIB: u8 = 10;
    /// `filter`, `score` and `pick` each burn a fixed loop first.
    pub const BURN_IN_STAGES: u8 = 11;
    /// `pick` draws `random-u64` into a counter first.
    pub const DRAW_IN_PICK: u8 = 12;
    /// `filter` grows the table past the host's cap first.
    pub const GROW_TABLE: u8 = 13;
    /// `score` fills the whole memory (grown to the cap) in a loop: cheap in
    /// fuel, long on the clock.
    pub const FILL_IN_SCORE: u8 = 14;
    /// `filter` traps when the decision has four candidates: the decode side
    /// of a pair of three prefills and four decodes.
    pub const TRAP_ON_FOUR_CANDIDATES: u8 = 15;
    /// `filter` fills the whole memory eight hundred times and returns, then `score`
    /// fills it in a loop: a long call that answers, then an endless one.
    pub const FILL_IN_FILTER_THEN_FILL: u8 = 16;
    /// The constructor burns a fixed amount of fuel: a decision's opens spend
    /// before any stage runs.
    pub const BURN_IN_CONSTRUCTOR: u8 = 17;
    /// The destructor makes fifty host calls: long on the clock,
    /// cheap in fuel.
    pub const SLOW_DESTRUCTOR: u8 = 18;
    /// The lifecycle export traps.
    pub const TRAP_IN_OBSERVE: u8 = 19;
    /// The destructor fills its memory without end: wall, little fuel.
    pub const FILL_IN_DESTRUCTOR: u8 = 20;
    /// `init` burns eight fixed loops: heavier than a small decision budget.
    pub const BURN_IN_INIT: u8 = 21;
    /// The decision's constructor spins without end.
    pub const SPIN_IN_CONSTRUCTOR: u8 = 22;
    /// The pick spins without end: the third stage, behind two cheap ones.
    pub const SPIN_IN_PICK: u8 = 23;
    /// The decision's constructor traps.
    pub const TRAP_IN_CONSTRUCTOR: u8 = 24;
    /// A second config byte: `init` asks the host's live clock once. An
    /// answer of zero lets it go on, of one makes it refuse, of two spin
    /// without end, of three trap at once.
    pub const INIT_ASKS_THE_CLOCK: u8 = 1;
    /// `init` fails.
    pub const REFUSE_INIT: u8 = 255;
}

/// The counters the guests add to (declared by the test's spec in this
/// order).
pub mod counter {
    /// One per `observe` call.
    pub const OBSERVED: u32 = 0;
    /// `now-micros` as read in the constructor.
    pub const NOW: u32 = 1;
    /// `random-u64` as drawn in the constructor.
    pub const RANDOM: u32 = 2;
    /// `now-micros` as read in `observe`.
    pub const OBSERVE_NOW: u32 = 3;
    /// The lease's session key as `observe` saw it.
    pub const OBSERVE_SESSION: u32 = 4;
    /// `random-u64` as drawn in `pick` under `DRAW_IN_PICK`.
    pub const PICK_RANDOM: u32 = 5;
}

/// The canonical ABI layout the record guest reads: the constructor's
/// arguments as one record of (request, list<candidate>, pool).
pub mod canonical {
    /// Offset of the candidate list's pointer in the arguments.
    pub const ARGS_CANDIDATES_PTR: usize = 136;
    /// Offset of the candidate list's length in the arguments.
    pub const ARGS_CANDIDATES_LEN: usize = 140;
    /// Size of one `candidate` record in guest memory.
    pub const CANDIDATE_SIZE: usize = 120;
    /// Offset of `in-flight` within a `candidate`.
    pub const CANDIDATE_IN_FLIGHT: usize = 20;
}

fn contracts_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("contracts")
}

/// The committed package, resolved.
pub fn resolve() -> (Resolve, PackageId) {
    let mut resolve = Resolve::new();
    let (package, _sources) = resolve
        .push_path(contracts_dir().join("smg-policy.wit"))
        .expect("the committed WIT resolves");
    (resolve, package)
}

/// Holds the record guest's offsets against wit-parser's size computation
/// of the committed WIT.
pub fn check_canonical_layout() {
    let (resolve, package) = resolve();
    let mut sizes = SizeAlign::default();
    sizes.fill(&resolve);
    let types = resolve.packages[package].interfaces["types"];
    let type_named = |name: &str| resolve.interfaces[types].types[name];
    let record_offsets = |name: &str| -> Vec<(String, usize)> {
        let TypeDefKind::Record(record) = &resolve.types[type_named(name)].kind else {
            // A non-record has no fields: the asserts below fail on the gap.
            return Vec::new();
        };
        sizes
            .field_offsets(record.fields.iter().map(|f| &f.ty))
            .into_iter()
            .zip(&record.fields)
            .map(|((offset, _), f)| (f.name.clone(), offset.size_wasm32()))
            .collect()
    };
    let candidate = record_offsets("candidate");
    let in_flight = candidate
        .iter()
        .find(|(n, _)| n == "in-flight")
        .map(|(_, o)| *o)
        .expect("candidate has in-flight");
    assert_eq!(in_flight, canonical::CANDIDATE_IN_FLIGHT, "{candidate:?}");
    let candidate_size = sizes.size(&wit_parser::Type::Id(type_named("candidate")));
    assert_eq!(
        candidate_size.size_wasm32(),
        canonical::CANDIDATE_SIZE,
        "{candidate:?}"
    );
    let request_size = sizes.size(&wit_parser::Type::Id(type_named("request")));
    // The arguments record: request, then the list (align 4), then the pool.
    let list_at = request_size.size_wasm32().next_multiple_of(4);
    assert_eq!(list_at, canonical::ARGS_CANDIDATES_PTR);
    assert_eq!(list_at + 4, canonical::ARGS_CANDIDATES_LEN);
}

/// Assembles `wat_text` and wraps it as a component of `world`.
pub fn component(world: &str, wat_text: &str) -> Vec<u8> {
    let mut module = wat::parse_str(wat_text).expect("the guest assembles");
    let (resolve, package) = resolve();
    let world = resolve
        .select_world(&[package], Some(world))
        .expect("the world exists");
    wit_component::embed_component_metadata(&mut module, &resolve, world, StringEncoding::UTF8)
        .expect("the world embeds");
    ComponentEncoder::default()
        .module(&module)
        .expect("the module matches the world")
        .validate(true)
        .encode()
        .expect("the component encodes")
}

/// A guest of the record world.
pub fn record_guest() -> Vec<u8> {
    component("policy", &guest_wat(Guest::Record))
}

/// A guest of the packed world.
pub fn packed_guest() -> Vec<u8> {
    component("policy-packed", &guest_wat(Guest::Packed))
}

/// A guest that imports a WASI interface: refused at load.
pub fn wasi_importing_guest() -> Vec<u8> {
    let text = guest_wat(Guest::Record).replacen(
        "(memory (export \"memory\") 4)",
        "(import \"wasi:clocks/monotonic-clock@0.2.0\" \"now\" (func $wasi_now (result i64)))\n  (memory (export \"memory\") 4)",
        1,
    );
    // The embedded world must name the import for the encoder to accept it:
    // a world of our own with the WASI import added.
    let mut module = wat::parse_str(&text).expect("the guest assembles");
    let (mut resolve, package) = resolve();
    let wasi_text = "package wasi:clocks@0.2.0;\ninterface monotonic-clock { now: func() -> u64; }";
    let wasi = resolve
        .push_str("wasi-clocks.wit", wasi_text)
        .expect("the wasi package parses");
    let extra_text = "package test:extra;\nworld with-wasi { include smg:policy/policy@1.0.0; import wasi:clocks/monotonic-clock@0.2.0; }";
    let _ = (package, wasi);
    let extra = resolve
        .push_str("with-wasi.wit", extra_text)
        .expect("the test world parses");
    let world = resolve
        .select_world(&[extra], Some("with-wasi"))
        .expect("the test world exists");
    wit_component::embed_component_metadata(&mut module, &resolve, world, StringEncoding::UTF8)
        .expect("the world embeds");
    ComponentEncoder::default()
        .module(&module)
        .expect("the module matches the world")
        .validate(true)
        .encode()
        .expect("the component encodes")
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Guest {
    Record,
    Packed,
}

/// The guest's text for one world. The two differ in the interface they
/// export, the constructor's shape and where a candidate's in-flight count
/// sits.
fn guest_wat(guest: Guest) -> String {
    let (iface, row, in_flight, constructor) = match guest {
        Guest::Record => (
            "smg:policy/plugin@1.0.0",
            canonical::CANDIDATE_SIZE,
            canonical::CANDIDATE_IN_FLIGHT,
            // One pointer to the arguments record: the list sits at +136.
            format!(
                r#"(func (export "smg:policy/plugin@1.0.0#[constructor]decision") (param $args i32) (result i32)
    (local $rep i32)
    (local.set $rep (call $alloc (i32.const 8) (i32.const 4)))
    (i32.store (local.get $rep) (i32.load offset={ptr} (local.get $args)))
    (i32.store offset=4 (local.get $rep) (i32.load offset={len} (local.get $args)))
    (call $on_construct)
    (call $new (local.get $rep)))"#,
                ptr = canonical::ARGS_CANDIDATES_PTR,
                len = canonical::ARGS_CANDIDATES_LEN,
            ),
        ),
        Guest::Packed => (
            "smg:policy/plugin-packed@1.0.0",
            104,
            8,
            // Seven flat params: request (ptr, len), candidates (ptr, len),
            // count, pool (ptr, len).
            r#"(func (export "smg:policy/plugin-packed@1.0.0#[constructor]decision") (param $rptr i32) (param $rlen i32) (param $cptr i32) (param $cand_len i32) (param $count i32) (param $pptr i32) (param $plen i32) (result i32)
    (local $rep i32)
    (local.set $rep (call $alloc (i32.const 8) (i32.const 4)))
    (i32.store (local.get $rep) (local.get $cptr))
    (i32.store offset=4 (local.get $rep) (local.get $count))
    (call $on_construct)
    (call $new (local.get $rep)))"#
                .to_string(),
        ),
    };
    format!(
        r#"(module
  (import "smg:policy/host@1.0.0" "counter" (func $counter (param i32 i64)))
  (import "smg:policy/host@1.0.0" "now-micros" (func $now (result i64)))
  (import "smg:policy/host@1.0.0" "random-u64" (func $random (result i64)))
  (import "smg:policy/host@1.0.0" "log" (func $log (param i32 i32 i32)))
  (import "[export]{iface}" "[resource-new]decision" (func $new (param i32) (result i32)))
  (memory (export "memory") 4)
  (table $t 1 funcref)
  (global $heap (mut i32) (i32.const 1024))
  (global $mode (mut i32) (i32.const 0))
  (data (i32.const 16) "refused")
  (data (i32.const 32) "observed")

  ;; A bump allocator over the arena past the static data; the decision's
  ;; destructor resets it, so a kept instance never grows across decisions.
  (func $alloc (param $size i32) (param $align i32) (result i32)
    (local $p i32)
    (local.set $p
      (i32.and
        (i32.add (global.get $heap) (i32.sub (local.get $align) (i32.const 1)))
        (i32.xor (i32.sub (local.get $align) (i32.const 1)) (i32.const -1))))
    (global.set $heap (i32.add (local.get $p) (local.get $size)))
    (if (i32.gt_u (global.get $heap) (i32.mul (memory.size) (i32.const 65536)))
      (then
        (if (i32.eq
              (memory.grow
                (i32.add
                  (i32.shr_u
                    (i32.sub (global.get $heap) (i32.mul (memory.size) (i32.const 65536)))
                    (i32.const 16))
                  (i32.const 1)))
              (i32.const -1))
          (then unreachable))))
    (local.get $p))

  (func (export "cabi_realloc") (param $old i32) (param $old_size i32) (param $align i32) (param $new_size i32) (result i32)
    (local $p i32)
    (local.set $p (call $alloc (local.get $new_size) (local.get $align)))
    (if (i32.ne (local.get $old) (i32.const 0))
      (then (memory.copy (local.get $p) (local.get $old) (local.get $old_size))))
    (local.get $p))

  ;; init(config) -> result<_, string>, by pointer: byte 0 the discriminant,
  ;; the string at +4.
  (func (export "{iface}#init") (param $ptr i32) (param $len i32) (result i32)
    (local $r i32) (local $k i32) (local $asked i64)
    (local.set $r (call $alloc (i32.const 12) (i32.const 4)))
    (if (i32.gt_u (local.get $len) (i32.const 0))
      (then (global.set $mode (i32.load8_u (local.get $ptr)))))
    ;; A second config byte, when set: `init` asks the host's live clock, one
    ;; host call that costs a few units of fuel however long the host takes
    ;; to answer. An answer of one makes it refuse, of two spin without end,
    ;; of three trap at once.
    (if (i32.gt_u (local.get $len) (i32.const 1))
      (then
        (if (i32.load8_u offset=1 (local.get $ptr))
          (then
            (local.set $asked (call $now))
            (if (i64.eq (local.get $asked) (i64.const 1))
              (then (global.set $mode (i32.const 255))))
            (if (i64.eq (local.get $asked) (i64.const 2))
              (then (loop $spin (br $spin))))
            (if (i64.eq (local.get $asked) (i64.const 3))
              (then unreachable))))))
    (if (i32.eq (global.get $mode) (i32.const {burn_init}))
      (then
        (local.set $k (i32.const 8))
        (loop $again
          (call $burn)
          (local.set $k (i32.sub (local.get $k) (i32.const 1)))
          (br_if $again (local.get $k)))))
    (if (i32.eq (global.get $mode) (i32.const 255))
      (then
        (i32.store8 (local.get $r) (i32.const 1))
        (i32.store offset=4 (local.get $r) (i32.const 16))
        (i32.store offset=8 (local.get $r) (i32.const 7)))
      (else (i32.store8 (local.get $r) (i32.const 0))))
    (local.get $r))

  ;; What the constructor tells the host through its counters.
  (func $on_construct
    (if (i32.eq (global.get $mode) (i32.const {trap_ctor})) (then unreachable))
    (call $counter (i32.const 1) (call $now))
    (call $counter (i32.const 2) (call $random))
    (if (i32.eq (global.get $mode) (i32.const {burn_ctor})) (then (call $burn)))
    (if (i32.eq (global.get $mode) (i32.const {spin_ctor})) (then (loop $spin (br $spin)))))

  ;; A fixed amount of work, so a call's fuel is the same every time.
  (func $burn (local $i i32)
    (local.set $i (i32.const 20000))
    (loop $more
      (local.set $i (i32.sub (local.get $i) (i32.const 1)))
      (br_if $more (local.get $i))))

  {constructor}

  ;; filter() -> result<list<u32>, decline>, by pointer: byte 0 the
  ;; discriminant; ok: the list at +4 (ptr) and +8 (len); err: the decline's
  ;; discriminant at +4. A method's `self` is the rep itself: a borrow of a
  ;; resource the callee implements lowers as its rep, not as a handle.
  (func (export "{iface}#[method]decision.filter") (param $self i32) (result i32)
    (local $rep i32) (local $len i32) (local $r i32) (local $out i32) (local $i i32) (local $n i32)
    (local.set $rep (local.get $self))
    (local.set $len (i32.load offset=4 (local.get $rep)))
    (local.set $r (call $alloc (i32.const 16) (i32.const 4)))
    (if (i32.eq (global.get $mode) (i32.const {trap_in_filter})) (then unreachable))
    (if (i32.and (i32.eq (global.get $mode) (i32.const {trap_on_four})) (i32.eq (local.get $len) (i32.const 4)))
      (then unreachable))
    (if (i32.eq (global.get $mode) (i32.const {fill_then_fill}))
      (then
        (drop (memory.grow (i32.const 252)))
        (local.set $i (i32.const 800))
        (loop $some
          (memory.fill (i32.const 0) (i32.const 0) (i32.mul (memory.size) (i32.const 65536)))
          (local.set $i (i32.sub (local.get $i) (i32.const 1)))
          (br_if $some (local.get $i)))))
    (if (i32.eq (global.get $mode) (i32.const {burn})) (then (call $burn)))
    (if (i32.eq (global.get $mode) (i32.const {grow_table}))
      (then (drop (table.grow $t (ref.null func) (i32.const 70000)))))
    (if (i32.eq (global.get $mode) (i32.const {grow}))
      (then (drop (call $alloc (i32.const 33554432) (i32.const 8)))))
    (if (i32.eq (global.get $mode) (i32.const {decline_filter}))
      (then
        (i32.store8 (local.get $r) (i32.const 1))
        (i32.store8 offset=4 (local.get $r) (i32.const 0))
        (return (local.get $r))))
    (local.set $out (call $alloc (i32.mul (local.get $len) (i32.const 4)) (i32.const 4)))
    (local.set $i (i32.const 0))
    (local.set $n (i32.const 0))
    (block $done
      (loop $next
        (br_if $done (i32.ge_u (local.get $i) (local.get $len)))
        (if (i32.or
              (i32.ne (global.get $mode) (i32.const {keep_even}))
              (i32.eqz (i32.and (local.get $i) (i32.const 1))))
          (then
            (i32.store (i32.add (local.get $out) (i32.mul (local.get $n) (i32.const 4))) (local.get $i))
            (local.set $n (i32.add (local.get $n) (i32.const 1)))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $next)))
    (if (i32.and (i32.eq (global.get $mode) (i32.const {out_of_range})) (i32.gt_u (local.get $n) (i32.const 0)))
      (then (i32.store (i32.add (local.get $out) (i32.mul (i32.sub (local.get $n) (i32.const 1)) (i32.const 4))) (local.get $len))))
    (if (i32.and (i32.eq (global.get $mode) (i32.const {duplicate})) (i32.gt_u (local.get $n) (i32.const 1)))
      (then (i32.store (i32.add (local.get $out) (i32.const 4)) (i32.const 0))))
    (i32.store8 (local.get $r) (i32.const 0))
    (i32.store offset=4 (local.get $r) (local.get $out))
    (i32.store offset=8 (local.get $r) (local.get $n))
    (local.get $r))

  ;; score() -> result<list<f32>, decline>: 1 - min(in-flight, 100) / 100.
  (func (export "{iface}#[method]decision.score") (param $self i32) (result i32)
    (local $rep i32) (local $cands i32) (local $len i32) (local $r i32) (local $out i32) (local $i i32) (local $f i32)
    (local.set $rep (local.get $self))
    (local.set $cands (i32.load (local.get $rep)))
    (local.set $len (i32.load offset=4 (local.get $rep)))
    (if (i32.eq (global.get $mode) (i32.const {spin})) (then (loop $spin (br $spin))))
    (if (i32.eq (global.get $mode) (i32.const {burn})) (then (call $burn)))
    (if (i32.or (i32.eq (global.get $mode) (i32.const {fill})) (i32.eq (global.get $mode) (i32.const {fill_then_fill})))
      (then
        ;; The filter of the second mode grew the memory already: a second
        ;; grow past the cap would trap.
        (if (i32.eq (global.get $mode) (i32.const {fill})) (then (drop (memory.grow (i32.const 252)))))
        (loop $fill
          (memory.fill (i32.const 0) (i32.const 0) (i32.mul (memory.size) (i32.const 65536)))
          (br $fill))))
    (local.set $r (call $alloc (i32.const 16) (i32.const 4)))
    (if (i32.eq (global.get $mode) (i32.const {decline_score}))
      (then
        (i32.store8 (local.get $r) (i32.const 1))
        (i32.store8 offset=4 (local.get $r) (i32.const 0))
        (return (local.get $r))))
    (local.set $out (call $alloc (i32.mul (local.get $len) (i32.const 4)) (i32.const 4)))
    (local.set $i (i32.const 0))
    (block $done
      (loop $next
        (br_if $done (i32.ge_u (local.get $i) (local.get $len)))
        (local.set $f (i32.load offset={in_flight} (i32.add (local.get $cands) (i32.mul (local.get $i) (i32.const {row})))))
        (local.set $f (select (local.get $f) (i32.const 100) (i32.lt_u (local.get $f) (i32.const 100))))
        (f32.store (i32.add (local.get $out) (i32.mul (local.get $i) (i32.const 4)))
          (select
            (f32.const nan)
            (f32.sub (f32.const 1) (f32.div (f32.convert_i32_u (local.get $f)) (f32.const 100)))
            (i32.eq (global.get $mode) (i32.const {nan_score}))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $next)))
    (i32.store8 (local.get $r) (i32.const 0))
    (i32.store offset=4 (local.get $r) (local.get $out))
    (i32.store offset=8 (local.get $r) (local.get $len))
    (local.get $r))

  ;; pick(scores) -> result<option<u32>, decline>: the best score at or
  ;; above zero; a row below zero was filtered out.
  (func (export "{iface}#[method]decision.pick") (param $self i32) (param $sptr i32) (param $slen i32) (result i32)
    (local $r i32) (local $i i32) (local $best i32) (local $top f32) (local $s f32) (local $found i32)
    (if (i32.eq (global.get $mode) (i32.const {spin_pick})) (then (loop $spin (br $spin))))
    (local.set $r (call $alloc (i32.const 16) (i32.const 4)))
    (i32.store8 (local.get $r) (i32.const 0))
    (if (i32.eq (global.get $mode) (i32.const {burn})) (then (call $burn)))
    (if (i32.eq (global.get $mode) (i32.const {draw_in_pick}))
      (then (call $counter (i32.const 5) (call $random))))
    (if (i32.eq (global.get $mode) (i32.const {decline_pick}))
      (then
        (i32.store8 offset=4 (local.get $r) (i32.const 0))
        (return (local.get $r))))
    (local.set $i (i32.const 0))
    (local.set $found (i32.const 0))
    (local.set $top (f32.const -2))
    (block $done
      (loop $next
        (br_if $done (i32.ge_u (local.get $i) (local.get $slen)))
        (local.set $s (f32.load (i32.add (local.get $sptr) (i32.mul (local.get $i) (i32.const 4)))))
        (if (i32.eq (global.get $mode) (i32.const {pick_filtered_out}))
          (then
            (if (f32.lt (local.get $s) (f32.const 0))
              (then
                (local.set $best (local.get $i))
                (local.set $found (i32.const 1))
                (br $done))))
          (else
            (if (i32.and (f32.ge (local.get $s) (f32.const 0)) (f32.gt (local.get $s) (local.get $top)))
              (then
                (local.set $top (local.get $s))
                (local.set $best (local.get $i))
                (local.set $found (i32.const 1))))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $next)))
    (if (i32.and (i32.eq (global.get $mode) (i32.const {pick_filtered_out})) (i32.eqz (local.get $found)))
      (then
        (local.set $best (local.get $slen))
        (local.set $found (i32.const 1))))
    (i32.store8 offset=4 (local.get $r) (local.get $found))
    (i32.store offset=8 (local.get $r) (local.get $best))
    (local.get $r))

  ;; observe(event): the flattened variant, sixteen values; counted. The
  ;; sixth value is the lease's session key on every lease-bearing case.
  (func (export "{iface}#observe") (param i32 i64 i64 i64 i32 i64 i64 i32 i64 i64 i32 i64 i32 i32 i32 i32)
    (if (i32.eq (global.get $mode) (i32.const {trap_in_observe})) (then unreachable))
    (call $counter (i32.const 0) (i64.const 1))
    (call $counter (i32.const 3) (call $now))
    (call $counter (i32.const 4) (local.get 6))
    (call $log (i32.const 2) (i32.const 32) (i32.const 8)))

  ;; The destructor resets the arena.
  (func (export "{iface}#[dtor]decision") (param $rep i32) (local $i i32)
    (if (i32.eq (global.get $mode) (i32.const {fill_dtor}))
      (then
        (loop $fill
          (memory.fill (i32.const 0) (i32.const 0) (i32.mul (memory.size) (i32.const 65536)))
          (br $fill))))
    (if (i32.eq (global.get $mode) (i32.const {slow_dtor}))
      (then
        (local.set $i (i32.const 50))
        (loop $calls
          (drop (call $now))
          (local.set $i (i32.sub (local.get $i) (i32.const 1)))
          (br_if $calls (local.get $i)))))
    (global.set $heap (i32.const 1024)))
)"#,
        iface = iface,
        row = row,
        in_flight = in_flight,
        constructor = constructor,
        trap_in_filter = mode::TRAP_IN_FILTER,
        decline_filter = mode::DECLINE_FILTER,
        keep_even = mode::KEEP_EVEN,
        out_of_range = mode::FILTER_OUT_OF_RANGE,
        duplicate = mode::FILTER_DUPLICATE,
        grow = mode::GROW_32_MIB,
        spin = mode::SPIN_IN_SCORE,
        nan_score = mode::NAN_SCORE,
        decline_score = mode::DECLINE_SCORE,
        decline_pick = mode::DECLINE_PICK,
        pick_filtered_out = mode::PICK_FILTERED_OUT,
        burn = mode::BURN_IN_STAGES,
        draw_in_pick = mode::DRAW_IN_PICK,
        grow_table = mode::GROW_TABLE,
        fill = mode::FILL_IN_SCORE,
        trap_on_four = mode::TRAP_ON_FOUR_CANDIDATES,
        fill_then_fill = mode::FILL_IN_FILTER_THEN_FILL,
        burn_ctor = mode::BURN_IN_CONSTRUCTOR,
        spin_ctor = mode::SPIN_IN_CONSTRUCTOR,
        spin_pick = mode::SPIN_IN_PICK,
        slow_dtor = mode::SLOW_DESTRUCTOR,
        trap_in_observe = mode::TRAP_IN_OBSERVE,
        fill_dtor = mode::FILL_IN_DESTRUCTOR,
        burn_init = mode::BURN_IN_INIT,
        trap_ctor = mode::TRAP_IN_CONSTRUCTOR,
    )
}

/// The record guest with four tables: more than the host allows one
/// instance.
pub fn four_table_guest() -> Vec<u8> {
    let wat = guest_wat(Guest::Record).replacen(
        "(table $t 1 funcref)",
        "(table $t 1 funcref)\n  (table $t2 1 funcref)\n  (table $t3 1 funcref)\n  (table $t4 1 funcref)",
        1,
    );
    component("policy", &wat)
}

/// The record guest with a second linear memory: more than the host allows
/// one instance.
pub fn two_memory_guest() -> Vec<u8> {
    let wat = guest_wat(Guest::Record).replacen(
        "(memory (export \"memory\") 4)",
        "(memory (export \"memory\") 4)\n  (memory 1)",
        1,
    );
    component("policy", &wat)
}
