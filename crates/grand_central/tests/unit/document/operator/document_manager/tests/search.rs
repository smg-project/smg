//! What the largest document can build and how much of it the pass holds.

use std::{env, fs, io, process::Command};

use super::*;
use crate::document::operator::reader::read_yaml;

/// What an apply of a YAML file may hold beside the file, for each
/// byte of the size bound: the most the search finds, with the room
/// its spread wants. One file's resident peak moves by a twentieth
/// from a run and a build to the next (367 to 406 by run). Every
/// file of the search is held to it.
const HELD_A_BYTE: usize = 448;

/// The budget of one apply, which sets the size bound: what a
/// document of the size bound may hold beside its bytes.
const BUDGET: usize = 64 << 20;

/// The measurement by its name, and the variable that hands one
/// file of it to a process of its own.
const NAME: &str =
    "document::operator::document_manager::tests::search::what_a_file_at_the_bound_holds";
const ONE: &str = "SMG_POLICY_SPEC_HELD_FILE";

/// `path`'s file applied in a process of its own, through the
/// measurement's one-file entry: whether that process held it to
/// [`HELD_A_BYTE`], and what it wrote.
fn alone(path: &Path) -> (bool, String) {
    let this = env::current_exe().expect("this executable");
    let child = Command::new(this)
        .args([
            "--exact",
            NAME,
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ONE, path)
        .output()
        .expect("the file's own process");
    let mut said = String::from_utf8_lossy(&child.stdout).into_owned();
    said.push_str(&String::from_utf8_lossy(&child.stderr));
    (child.status.success(), said)
}

/// A profile of a thousand scorers of a letter, under an anchor.
fn scorers() -> String {
    let letters = "a,".repeat(1000);
    format!("&p {{filter: [healthy], score: [{letters}], pick: best}}")
}

/// A document's head: one profile, and `count` names more that each
/// write `alias`.
fn named(anchor: &str, alias: &str, count: usize) -> String {
    let mut head = format!("apiVersion: smg.policy/v1\nprofiles:\n  default: {anchor}\n");
    for name in 0..count {
        let _ = writeln!(head, "  t{name:04}: {alias}");
    }
    head
}

/// `head`, then bindings of nothing to `len` bytes.
fn beside_bindings(head: String, len: usize) -> Vec<u8> {
    filled(&format!("{head}bindings: ["), "{},", "]\n", len)
}

/// One file of the search.
struct Shape {
    name: String,
    /// Whether an alias is in it: a file that writes its values out
    /// is refused by no bound of the read.
    aliases: bool,
    file: Vec<u8>,
}

/// `text` with blanks to `len` bytes, a line's end the last.
fn blanked(text: String, len: usize) -> Vec<u8> {
    let mut file = text.into_bytes();
    assert!(file.len() < len, "the text is past the size");
    file.resize(len - 1, b' ');
    file.push(b'\n');
    file
}

/// `open`, then `item` until the next would pass `len` bytes, then
/// `close`, then blanks to `len` bytes.
fn filled(open: &str, item: &str, close: &str, len: usize) -> Vec<u8> {
    let mut text = String::with_capacity(len);
    text.push_str(open);
    while text.len() + item.len() + close.len() < len {
        text.push_str(item);
    }
    text.push_str(close);
    blanked(text, len)
}

/// A flow map of `count` keys with no values, each key once.
fn keys(count: usize) -> String {
    let mut text = String::from("{");
    for key in 0..count {
        let _ = write!(text, "{key:x},");
    }
    text.push('}');
    text
}

/// Numbers from a seed, the same from the same seed.
struct Draw(u64);

impl Draw {
    fn below(&mut self, bound: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % bound
    }
}

/// A value of the draw's choosing in flow style, at most `depth`
/// lists and maps deep.
fn drawn(draw: &mut Draw, depth: u32, out: &mut String) {
    match draw.below(if depth == 0 { 5 } else { 8 }) {
        0 => out.push('~'),
        1 => out.push_str("true"),
        2 => {
            let _ = write!(out, "{}", draw.below(100_000));
        }
        3 => out.push_str("[]"),
        4 => {
            out.push('"');
            for _ in 0..draw.below(32) {
                out.push('x');
            }
            out.push('"');
        }
        5 | 6 => {
            out.push('[');
            for _ in 0..draw.below(6) {
                drawn(draw, depth - 1, out);
                out.push(',');
            }
            out.push(']');
        }
        _ => {
            out.push('{');
            for key in 0..draw.below(6) {
                let _ = write!(out, "k{key}: ");
                drawn(draw, depth - 1, out);
                out.push(',');
            }
            out.push('}');
        }
    }
}

/// Values of a draw's choosing: eight under anchors, `aliases`
/// aliases of them, then values written out to `len` bytes.
fn mixed(seed: u64, aliases: usize, len: usize) -> Vec<u8> {
    let mut draw = Draw(seed);
    let mut text = String::with_capacity(len);
    text.push('[');
    for anchor in 0..8 {
        let _ = write!(text, "&a{anchor} ");
        drawn(&mut draw, 3, &mut text);
        text.push(',');
    }
    for _ in 0..aliases {
        let _ = write!(text, "*a{},", draw.below(8));
    }
    let mut next = String::new();
    loop {
        next.clear();
        drawn(&mut draw, 3, &mut next);
        if text.len() + next.len() + 2 >= len {
            break;
        }
        text.push_str(&next);
        text.push(',');
    }
    text.push(']');
    blanked(text, len)
}

/// The files that write their values out.
fn written_out(len: usize) -> Vec<Shape> {
    let nest = |depth: usize| format!("{}{},", "[".repeat(depth), "]".repeat(depth));
    let line = |opens: &str| format!("{}a\n", opens.repeat(100));
    let maps = format!("{}1{},", "{a: ".repeat(100), "}".repeat(100));
    let runs = document_of(len - 256).0;
    let runs = std::str::from_utf8(&runs).expect("text");
    let files = [
        ("texts of a byte", filled("[", "a,", "]", len)),
        ("numbers", filled("[", "1,", "]", len)),
        ("nulls", filled("[", "~,", "]", len)),
        ("nulls, a line each", filled("", "-\n", "", len)),
        ("empty lists", filled("[", "[],", "]", len)),
        ("empty maps", filled("[", "{},", "]", len)),
        ("maps of one key", filled("[", "{a},", "]", len)),
        (
            "nests of a hundred lists",
            filled("[", &nest(100), "]", len),
        ),
        (
            "nests past the reader's depth",
            filled("[", &nest(200), "]", len),
        ),
        ("a hundred lists a line", filled("", &line("- "), "", len)),
        (
            "a hundred keys of a key a line",
            filled("", &line("? "), "", len),
        ),
        ("keys of nothing", filled("[", "? ,", "]", len)),
        (
            "a hundred keys of nothing a line",
            filled("", &line(": "), "", len),
        ),
        (
            "lists and keys of a key by turns",
            filled("", &line("- ? "), "", len),
        ),
        (
            "a key written twice, to the size",
            filled("{", "a,", "}", len),
        ),
        ("maps a hundred deep", filled("[", &maps, "]", len)),
        ("keys, each once", blanked(keys(len / 7), len)),
        ("one text", filled("\"", "x", "\"", len)),
        (
            "texts of a thousand bytes",
            filled("[", &format!("{},", "x".repeat(1000)), "]", len),
        ),
        (
            "escapes of two bytes for three",
            filled("\"", "\\L", "\"", len),
        ),
        ("values by draw 1", mixed(1, 0, len)),
        ("values by draw 2", mixed(2, 0, len)),
        ("a document that runs", padded(runs, len)),
        (
            "bindings of nothing",
            filled(&format!("{LEAST_LOAD}bindings: ["), "{},", "]\n", len),
        ),
    ];
    files
        .into_iter()
        .map(|(name, file)| Shape {
            name: name.to_owned(),
            aliases: false,
            file,
        })
        .collect()
}

/// A count of `0..=most` whose file the read's bounds admit and
/// whose next they refuse, found by halving; `most` where they admit
/// that one.
fn heaviest(most: usize, file: &dyn Fn(usize) -> Vec<u8>) -> usize {
    let admitted = |count: usize| read_yaml(&file(count)).is_ok();
    assert!(admitted(0), "the bounds admit a family's least file");
    if admitted(most) {
        return most;
    }
    let (mut holds, mut over) = (0, most);
    while over - holds > 1 {
        let middle = holds + (over - holds) / 2;
        if admitted(middle) {
            holds = middle;
        } else {
            over = middle;
        }
    }
    holds
}

/// An anchor's value of a count of a thing.
type Anchor = fn(usize) -> String;

/// What an alias is tried on, each by its name.
const ANCHORS: [(&str, Anchor); 7] = [
    ("a text of N bytes", |count| {
        format!("\"{}\"", "x".repeat(count))
    }),
    ("N numbers", |count| format!("[{}]", "1,".repeat(count))),
    ("N texts of a byte", |count| {
        format!("[{}]", "a,".repeat(count))
    }),
    ("N empty lists", |count| {
        format!("[{}]", "[],".repeat(count))
    }),
    ("N maps of one key", |count| {
        format!("[{}]", "{a},".repeat(count))
    }),
    ("a nest of N lists and one", |count| {
        format!("{}{}", "[".repeat(count + 1), "]".repeat(count + 1))
    }),
    ("a map of N keys", keys),
];

/// The files an alias multiplies: each family at the most the
/// bounds admit, and one past.
fn aliased(len: usize) -> Vec<Shape> {
    let mut shapes = Vec::new();
    let mut both = |name: &str, most: usize, file: &dyn Fn(usize) -> Vec<u8>| {
        let count = heaviest(most, file);
        for (count, said) in [
            (count, "the most the bounds admit"),
            (count + 1, "one more"),
        ] {
            shapes.push(Shape {
                name: format!("{name}, N = {count}: {said}"),
                aliases: true,
                file: file(count),
            });
        }
    };
    for (name, anchor) in ANCHORS {
        both(&format!("aliases of {name}"), 4096, &|count| {
            filled(&format!("[&a {},", anchor(count)), "*a,", "]", len)
        });
    }
    let numbers = "1,".repeat(1000);
    both(
        "N aliases of a thousand numbers, the rest a comment",
        (len - 4096) / 3,
        &|count| padded(&format!("[&a [{numbers}],{}]\n", "*a,".repeat(count)), len),
    );
    both(
        "N aliases of a thousand numbers in a file of no more bytes",
        100_000,
        &|count| format!("[&a [{numbers}],{}]\n", "*a,".repeat(count)).into_bytes(),
    );
    let nests = format!("{}{},", "[".repeat(100), "]".repeat(100));
    for (name, item) in [("texts of a byte", "a,"), ("maps of one key", "{a},")] {
        let head = format!("[&a [{}],", item.repeat(100));
        for (rest, tail) in [("nests", nests.as_str()), ("maps of one key", "{a},")] {
            both(
                &format!("N aliases of a hundred {name}, the rest {rest}"),
                len / 4,
                &|count| filled(&format!("{head}{}", "*a,".repeat(count)), tail, "]", len),
            );
        }
    }
    both("a header of N bytes under aliases", 4096, &|count| {
        let head = format!(
            "{LEAST_LOAD}view:\n  headers: [&h \"{}\"",
            "x".repeat(count)
        );
        filled(&head, ",*h", "]\n", len)
    });
    // The heaviest typed list written out, bindings of nothing,
    // beside what an alias multiplies: the bindings take the file's
    // bytes and a third of what the read may build, and the aliases
    // take the rest in a few thousand bytes.
    let letters = "a,".repeat(1000);
    let beside_bindings = |head: String| beside_bindings(head, len);
    for (name, anchor, alias) in [
        (
            "names on one profile of a thousand scorers of a letter",
            scorers(),
            "*p",
        ),
        (
            "names on one profile of a thousand filters of a letter",
            format!("&p {{filter: [{letters}], pick: best}}"),
            "*p",
        ),
        (
            "profiles on one list of a thousand scorers of a letter",
            format!("{{score: &p [{letters}], pick: best}}"),
            "{score: *p, pick: best}",
        ),
    ] {
        let named = |count: usize| named(&anchor, alias, count);
        both(
            &format!("N {name}, the rest bindings of nothing"),
            (len - 8192) / 40,
            &|count| beside_bindings(named(count)),
        );
        // What a small file's aliases build alone.
        both(
            &format!("N {name}, in a file of no more bytes"),
            (len - 8192) / 40,
            &|count| named(count).into_bytes(),
        );
    }
    both(
        "N aliases of a header of a thousand bytes, the rest bindings of nothing",
        4096,
        &|count| {
            beside_bindings(format!(
                "{LEAST_LOAD}view:\n  headers: [&h \"{}\"{}]\n",
                "x".repeat(1000),
                ",*h".repeat(count)
            ))
        },
    );
    // A document that runs: one binding that names a profile, under
    // aliases to the size, each a binding the generation keeps.
    both(
        "N aliases of a binding that names a profile",
        (len - 4096) / 3,
        &|count| {
            let bindings = ",*b".repeat(count);
            padded(
                &format!("{LEAST_LOAD}bindings: [&b {{profile: default}}{bindings}]\n"),
                len,
            )
        },
    );
    for seed in 3..11 {
        both(
            &format!("values by draw {seed} with N aliases"),
            (len - 8192) / 4,
            &|count| mixed(seed, count, len),
        );
    }
    // The values an alias builds, then nodes the read never reaches:
    // the parser lists a whole file before the first value is read.
    let texts = format!("- &a [{}]\n", "a,".repeat(100));
    let count = heaviest((len - 4096) / 5, &|count| {
        padded(&format!("{texts}{}", "- *a\n".repeat(count)), len)
    });
    let keys = format!("- {}a\n", "? ".repeat(100));
    for (name, between) in [
        ("keys of a key", ""),
        ("a second document of keys of a key", "---\n"),
    ] {
        let head = format!("{texts}{}{between}", "- *a\n".repeat(count));
        shapes.push(Shape {
            name: format!("{count} aliases of a hundred texts of a byte, then {name}"),
            aliases: true,
            file: filled(&head, &keys, "", len),
        });
    }
    let mut doubling = String::from("[&a0 [x,x],");
    for level in 1..40 {
        let _ = write!(doubling, "&a{level} [*a{0},*a{0}],", level - 1);
    }
    doubling.push_str("]\n");
    shapes.push(Shape {
        name: "aliases of aliases, forty deep".to_owned(),
        aliases: true,
        file: padded(&doubling, len),
    });
    shapes
}

/// The process's resident set, or its peak: the line of that name
/// in the process's status, in bytes.
fn resident(line: &str) -> usize {
    let status = fs::read_to_string("/proc/self/status").expect("the process's status");
    let kib = status
        .lines()
        .find_map(|found| found.strip_prefix(line))
        .and_then(|rest| rest.trim().strip_suffix("kB"))
        .and_then(|number| number.trim().parse::<usize>().ok())
        .expect("the line's number");
    kib * 1024
}

/// One file applied in this process: what the apply held at its
/// peak, beside what the process held before it.
fn hold(path: &Path) {
    let file = fs::read(path).expect("the file");
    let bytes = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
    let (_store, generations) = generations();
    // The peak counts from here where the kernel takes the word;
    // where it does not, the peak read is no lower.
    let _ = fs::write("/proc/self/clear_refs", "5");
    let before = resident("VmRSS:");
    let began = Instant::now();
    let applied = yaml(&generations, file);
    let micros = began.elapsed().as_micros();
    let held = resident("VmHWM:").saturating_sub(before);
    let outcome = match applied {
        Ok(applied) => format!("{applied:?}"),
        Err(SpecRejected::Unparsed { message, .. }) => message,
        Err(other) => format!("{other:?}"),
    };
    let outcome: String = outcome
        .chars()
        .take(96)
        .map(|letter| if letter == '\n' { ' ' } else { letter })
        .collect();
    writeln!(
        io::stdout(),
        "HOLDS a_byte={} held={held} micros={micros} outcome={outcome}",
        held / bytes
    )
    .expect("stdout");
    assert!(
        held <= HELD_A_BYTE * bytes,
        "{held} bytes held where the size bound is {bytes}"
    );
}

/// The heaviest file the search finds, with its count of names:
/// bindings of nothing to the size bound, beside one profile of a
/// thousand scorers under the most names the bounds admit.
pub(super) fn heaviest_file() -> (Vec<u8>, usize) {
    let len = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
    let profile = scorers();
    let file = |count: usize| beside_bindings(named(&profile, "*p", count), len);
    let count = heaviest((len - 8192) / 40, &file);
    (file(count), count)
}

/// The budget holds in every run: the heaviest file the search finds
/// (bindings of nothing to the size bound, beside one profile of a
/// thousand scorers under the most names the bounds admit), applied
/// in a process of its own, holds no more than one apply's budget. A
/// larger size bound, or a type that grows past the spread between
/// runs, is told here. A copy of the document that comes back reads
/// at this case's edge, and the search tells it.
#[test]
fn the_heaviest_file_found_holds_no_more_than_an_applys_budget() {
    let (file, count) = heaviest_file();
    let dir = tempfile::tempdir().expect("a directory");
    let path = dir.path().join("spec.yaml");
    fs::write(&path, file).expect("the file");
    let (_, said) = alone(&path);
    // The validator's refusal: the document was built and digested.
    assert!(said.contains("UnknownScorer"), "{count} names: {said}");
    let held = said
        .split_once(" held=")
        .and_then(|(_, rest)| rest.split(' ').next())
        .and_then(|bytes| bytes.parse::<usize>().ok())
        .expect("the bytes the apply held");
    assert!(
        held <= BUDGET,
        "{count} names: {held} bytes held, past the budget of {BUDGET}"
    );
}

/// The size bound is the budget's: at the most the search finds for
/// each byte, a document of the size bound holds no more than one
/// apply's budget.
#[test]
fn a_document_of_the_size_bound_holds_no_more_than_an_applys_budget() {
    let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
    assert!(
        HELD_A_BYTE * bound <= BUDGET,
        "{HELD_A_BYTE} bytes for each of {bound} is past {BUDGET}"
    );
}

/// The search: files in the shapes that build the most (values
/// written out, nests, values of no text, long texts, a typed
/// document of nothing, aliases at the most the bounds admit and one
/// past, that typed document beside what an alias shares, a document
/// that runs under aliases, values by draw), of the size bound's
/// bytes but for the small files their aliases take to the bounds,
/// each applied in a process of its own, which reads its resident
/// peak over the apply.
/// Every file is held to [`HELD_A_BYTE`], and a file that writes its
/// values out is refused by no bound of the read.
#[test]
#[ignore = "a measurement for the Numbers table, run by name"]
fn what_a_file_at_the_bound_holds() {
    if let Some(path) = env::var_os(ONE) {
        hold(Path::new(&path));
        return;
    }
    let len = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
    let dir = tempfile::tempdir().expect("a directory");
    let path = dir.path().join("spec.yaml");
    let mut out = io::stdout().lock();
    writeln!(out).expect("stdout");
    let mut most = 0;
    for shape in written_out(len).into_iter().chain(aliased(len)) {
        fs::write(&path, &shape.file).expect("the file");
        let (held, said) = alone(&path);
        // The harness writes a test's name before the test's first line.
        let row = said
            .lines()
            .find_map(|line| line.split_once("HOLDS ").map(|(_, row)| row))
            .unwrap_or("nothing");
        writeln!(
            out,
            "HOLDS {row} bytes={} shape={}",
            shape.file.len(),
            shape.name
        )
        .expect("stdout");
        assert!(held, "{}: {said}", shape.name);
        assert!(
            shape.aliases || !row.contains("of the size bound"),
            "{}: no alias is in it and a bound of the read refuses it",
            shape.name
        );
        let a_byte = row
            .split(['=', ' '])
            .nth(1)
            .and_then(|number| number.parse::<usize>().ok())
            .expect("the row's number");
        most = most.max(a_byte);
    }
    writeln!(out, "HOLDS_MOST a_byte={most} of {HELD_A_BYTE}").expect("stdout");
}
