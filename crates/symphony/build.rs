//! Writes the registry's checkpoint groups as Rust.
//!
//! `src/formats/groups.json` is the one place a group is spelled: the primary's id, its other
//! checkpoints, the table that reads the group by name, the group's own turn opener and what its
//! template writes after the thought's close where that is not the table's, one group per line in
//! bellwether's slug order. This script reads the file and writes the constant
//! `formats::registry` includes, so the compiled tables are the literals they were when they were
//! written by hand, and a tool that records a group writes a line of data, not Rust.

use std::{env, error::Error, fmt::Write as _, fs, path::Path};

use serde::Deserialize;

/// The data file, relative to the crate.
const GROUPS: &str = "src/formats/groups.json";

/// One group as the file spells it; a field the registry does not know fails the build.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Group {
    model: String,
    also: Vec<String>,
    table: String,
    opener: Option<String>,
    after: Option<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed={GROUPS}");
    let text = fs::read_to_string(GROUPS).map_err(|e| format!("{GROUPS}: {e}"))?;
    let groups: Vec<Group> = serde_json::from_str(&text).map_err(|e| format!("{GROUPS}: {e}"))?;
    let out_dir = env::var_os("OUT_DIR").ok_or("OUT_DIR is not set")?;
    fs::write(Path::new(&out_dir).join("groups.rs"), render(&groups)?)?;
    Ok(())
}

/// The constant as `registry.rs` wrote it by hand. A table is named, and looked up where the
/// constant is evaluated, so a name the registry does not have fails the build at the group.
fn render(groups: &[Group]) -> Result<String, std::fmt::Error> {
    let mut out = String::new();
    writeln!(
        out,
        "// Written by build.rs from {GROUPS}: edit the data file, not this."
    )?;
    writeln!(out, "const GROUPS: &[Group] = &[")?;
    for group in groups {
        writeln!(out, "    Group {{")?;
        writeln!(out, "        model: {:?},", group.model)?;
        writeln!(out, "        also: &{:?},", group.also)?;
        writeln!(out, "        table: table_named({:?}),", group.table)?;
        match &group.opener {
            Some(opener) => writeln!(out, "        opener: Some({opener:?}),")?,
            None => writeln!(out, "        opener: None,")?,
        }
        match &group.after {
            Some(after) => writeln!(out, "        after: Some({after:?}),")?,
            None => writeln!(out, "        after: None,")?,
        }
        writeln!(out, "    }},")?;
    }
    writeln!(out, "];")?;
    Ok(out)
}
