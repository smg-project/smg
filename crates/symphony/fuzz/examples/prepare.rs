//! What a fuzz run starts from: a dictionary per target (every terminal of its table, the declared
//! tools' names and the atoms of the call syntaxes, in libFuzzer's dictionary syntax) and a seed
//! per recorded output slice of the crate's fixtures, each the case that replays the slice.
//!
//! `cargo +nightly run --release --example prepare -- <fixtures dir> <out dir>` writes
//! `<out dir>/dict/<target>.dict` and `<out dir>/seeds/<table>/<family>`; `run.sh` calls it. The
//! work is [`symphony_fuzz::prepare`], which the harness tests over the crate's fixtures.

use std::{env, path::Path, process::ExitCode};

use symphony_fuzz::prepare;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let [_, fixtures, out] = args.as_slice() else {
        eprintln!("usage: prepare <fixtures dir> <out dir>");
        return ExitCode::FAILURE;
    };
    match prepare(Path::new(fixtures), Path::new(out)) {
        Ok(prepared) => {
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
