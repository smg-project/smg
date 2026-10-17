//! What a fuzz run starts from: a dictionary per target (every terminal of its table, the declared
//! tools' names and the atoms of the call syntaxes, in libFuzzer's dictionary syntax) and a seed
//! per recorded output slice of bellwether's hand-written sets, each the case that replays the
//! slice.
//!
//! `cargo +nightly run --release --example prepare -- <fixture tree> <out dir>` writes
//! `<out dir>/dict/<target>.dict` and `<out dir>/seeds/<table>/<family>` from the tree
//! `bellwether unpack` writes; `run.sh` calls it with the tree `BELLWETHER_FIXTURES` names. A
//! group the tree has not recorded is named and left out; a tree with no group recorded fails
//! the run, so a campaign never starts without its seeds. The work is
//! [`symphony_fuzz::prepare`], which the harness tests over that tree.

use std::{env, path::Path, process::ExitCode};

use symphony_fuzz::prepare;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let [_, fixtures, out] = args.as_slice() else {
        eprintln!("usage: prepare <fixture tree> <out dir>");
        return ExitCode::FAILURE;
    };
    match prepare(Path::new(fixtures), Path::new(out)) {
        Ok(prepared) => {
            for notice in &prepared.missing {
                eprintln!("prepare: skipping {notice}");
            }
            if prepared.seeds == 0 {
                eprintln!("prepare: no seeds: none of the groups is recorded under {fixtures}");
                return ExitCode::FAILURE;
            }
            println!(
                "prepare: {} dictionaries, {} seeds under {out}",
                prepared.dictionaries, prepared.seeds
            );
            ExitCode::SUCCESS
        }
        Err(why) => {
            eprintln!("prepare: {why}");
            ExitCode::FAILURE
        }
    }
}
