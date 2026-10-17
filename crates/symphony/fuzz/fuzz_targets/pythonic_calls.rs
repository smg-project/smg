#![no_main]

use symphony::formats::Table;
use symphony_fuzz::fuzz_target;

fuzz_target!(|data: &[u8]| symphony_fuzz::run(data, Some(Table::Lfm2_5)));
