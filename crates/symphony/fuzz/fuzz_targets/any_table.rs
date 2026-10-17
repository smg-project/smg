#![no_main]

use symphony_fuzz::fuzz_target;

fuzz_target!(|data: &[u8]| symphony_fuzz::run(data, None));
