//! Reads a fuzzer input (a crash artifact, a corpus unit) the way a target does and prints the
//! case: the table, the prompt's tail, the output as a Rust string literal, the delta plan and the
//! finish, then runs the checks and prints what they say. The decoded case is what a report
//! carries; `cargo +nightly run --release --example explain -- <target> <file>`. With
//! `--output`, the file is an output alone, read as a seed of the target's table would be (in
//! pieces of about a token, with a stop): the way to try a shape by hand.

use std::{env, fs, process::ExitCode};

use symphony_fuzz::{check, decode, encode, table_of};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let (as_output, target, file) = match args.as_slice() {
        [_, target, file] => (false, target, file),
        [_, flag, target, file] if flag == "--output" => (true, target, file),
        _ => {
            eprintln!("usage: explain [--output] <target> <file>");
            return ExitCode::FAILURE;
        }
    };
    let Some(table) = table_of(target) else {
        eprintln!("explain: no target named {target}");
        return ExitCode::FAILURE;
    };
    let mut data = match fs::read(file) {
        Ok(data) => data,
        Err(why) => {
            eprintln!("explain: {file}: {why}");
            return ExitCode::FAILURE;
        }
    };
    if as_output {
        let text = String::from_utf8_lossy(&data).into_owned();
        let seed_table = table.unwrap_or(symphony::formats::Table::ALL[0]);
        data = match encode(seed_table, "", &text) {
            Ok(data) => data,
            Err(why) => {
                eprintln!("explain: {why}");
                return ExitCode::FAILURE;
            }
        };
    }
    let case = decode(&data, table);
    println!("table: {}", case.table.name());
    println!("prompt: {:?}", case.prompt);
    println!("output: {:?}", case.output);
    println!("cuts: {:?}", case.cuts);
    println!("empty deltas between the pieces: {}", case.empty_between);
    println!("finish: {:?}", case.finish);
    match std::panic::catch_unwind(|| check(&case)) {
        Ok(()) => {
            println!("checks: every invariant holds");
            ExitCode::SUCCESS
        }
        Err(_) => {
            println!("checks: a breach, reported above");
            ExitCode::FAILURE
        }
    }
}
