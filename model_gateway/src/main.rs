use std::process::ExitCode;

use smg::observability::metrics::register_jemalloc_as_global_allocator;

smg::stock_allocator!();

fn main() -> Result<ExitCode, Box<dyn std::error::Error>> {
    register_jemalloc_as_global_allocator();
    smg::main_from_args()
}
