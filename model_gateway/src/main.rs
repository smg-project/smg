use std::process::ExitCode;

smg::stock_allocator!();

fn main() -> Result<ExitCode, Box<dyn std::error::Error>> {
    smg::main_from_args()
}
