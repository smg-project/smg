//! What a fuzz run starts from: a dictionary per target (every terminal of its table, the declared
//! tools' names and the atoms of the call syntaxes, in libFuzzer's dictionary syntax) and a seed
//! per recorded output slice of the crate's fixtures.
//!
//! `cargo +nightly run --release --example prepare -- <fixtures dir> <out dir>` writes
//! `<out dir>/dict/<target>.dict` and `<out dir>/seeds/<table>/<family>`; `run.sh` calls it.

use std::{env, fs, io::Write, path::Path, process::ExitCode};

use symphony::formats::Table;
use symphony_fuzz::{declared_names, slices, TARGETS};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let [_, fixtures, out] = args.as_slice() else {
        eprintln!("usage: prepare <fixtures dir> <out dir>");
        return ExitCode::FAILURE;
    };
    match prepare(Path::new(fixtures), Path::new(out)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(why) => {
            eprintln!("prepare: {why}");
            ExitCode::FAILURE
        }
    }
}

fn prepare(fixtures: &Path, out: &Path) -> Result<(), String> {
    let dict = out.join("dict");
    fs::create_dir_all(&dict).map_err(|e| format!("{}: {e}", dict.display()))?;
    for (target, table) in TARGETS {
        let tables: Vec<Table> = match table {
            Some(table) => vec![*table],
            None => Table::ALL.to_vec(),
        };
        let path = dict.join(format!("{target}.dict"));
        let mut file = fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut entries: Vec<String> = Vec::new();
        for table in tables {
            let format = table.format();
            entries.extend(format.terminal_texts().map(str::to_owned));
        }
        entries.extend(declared_names().into_iter().map(str::to_owned));
        entries.extend(ATOMS.iter().map(|atom| (*atom).to_owned()));
        entries.sort();
        entries.dedup();
        for entry in entries {
            writeln!(file, "{}", quoted(&entry)).map_err(|e| format!("{}: {e}", path.display()))?;
        }
    }
    let mut written = 0;
    for (table, family, slice) in slices(fixtures)? {
        let dir = out.join("seeds").join(table.name());
        fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = dir.join(family);
        fs::write(&path, slice.output.as_bytes())
            .map_err(|e| format!("{}: {e}", path.display()))?;
        written += 1;
    }
    println!(
        "prepare: {} dictionaries, {written} seeds under {}",
        TARGETS.len(),
        out.display()
    );
    Ok(())
}

/// Pieces of the call syntaxes a mutation composes markers with.
const ATOMS: &[&str] = &[
    "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}",
    "\"arguments\"",
    "\"name\"",
    "city",
    "days",
    "Paris",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<function=",
    "<parameter=",
    "</parameter>",
    "</function>",
    " name=\"",
    " string=\"true\"",
    " key=\"",
    " type=\"string\"",
    " index=\"1\"",
    "call:",
    "<|\"|>",
    "[get_weather(city='Paris')]",
    "\\u00e9",
    "\\\"",
    "\u{fffd}",
];

/// `entry` in libFuzzer's dictionary syntax: quoted, with every byte outside printable ASCII, the
/// quote and the backslash escaped as `\xNN`.
fn quoted(entry: &str) -> String {
    let mut out = String::from("\"");
    for byte in entry.bytes() {
        match byte {
            b'"' | b'\\' => out.push_str(&format!("\\x{byte:02x}")),
            0x20..=0x7e => out.push(byte as char),
            _ => out.push_str(&format!("\\x{byte:02x}")),
        }
    }
    out.push('"');
    out
}
