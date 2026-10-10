//! Version information module
//!
//! Provides version information including version number, build time, and Git metadata.

macro_rules! build_env {
    ($name:ident) => {
        env!(concat!("SMG_", stringify!($name)))
    };
}

pub const PROJECT_NAME: &str = build_env!(PROJECT_NAME);
pub const VERSION: &str = build_env!(VERSION);
pub const BUILD_TIME: &str = build_env!(BUILD_TIME);
pub const GIT_BRANCH: &str = build_env!(GIT_BRANCH);
pub const GIT_COMMIT: &str = build_env!(GIT_COMMIT);
pub const GIT_STATUS: &str = build_env!(GIT_STATUS);
pub const RUSTC_VERSION: &str = build_env!(RUSTC_VERSION);
pub const CARGO_VERSION: &str = build_env!(CARGO_VERSION);
pub const TARGET_TRIPLE: &str = build_env!(TARGET_TRIPLE);
pub const BUILD_MODE: &str = build_env!(BUILD_MODE);

/// Get simple version string (default for --version)
pub fn get_version_string() -> String {
    format!("{PROJECT_NAME} {VERSION}")
}

/// Get verbose version information string with full build details (for --version-verbose)
pub fn get_verbose_version_string() -> String {
    format!(
        "{}\n\n\
Build Information:\n\
  Build Time: {}\n\
  Build Mode: {}\n\
  Platform: {}\n\n\
Version Control:\n\
  Git Branch: {}\n\
  Git Commit: {}\n\
  Git Status: {}\n\n\
Compiler:\n\
  {}\n\
  {}",
        get_version_string(),
        BUILD_TIME,
        BUILD_MODE,
        TARGET_TRIPLE,
        GIT_BRANCH,
        GIT_COMMIT,
        GIT_STATUS,
        RUSTC_VERSION,
        CARGO_VERSION
    )
}

/// Get version number only
pub fn get_version() -> &'static str {
    VERSION
}

/// Print the startup banner with braille art and key configuration info.
///
/// Layout inspired by a serving engine's startup banner — art on the left,
/// useful context on the right. Shepherd with sheep motif.
#[expect(
    clippy::print_stdout,
    reason = "startup banner is printed before tracing is initialized; stdout is intentional"
)]
pub fn print_banner(host: &str, port: u16, mode: &str) {
    let info: [(&str, String); 4] = [
        ("", PROJECT_NAME.to_string()),
        ("version", VERSION.to_string()),
        (
            "listening",
            crate::config::bind_socket_addr(host, port)
                .map(|addr| addr.to_string())
                .unwrap_or_else(|_| format!("{host}:{port}")),
        ),
        ("mode", mode.to_string()),
    ];

    let art: [&str; 15] = [
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣤⣴⠟⠛⢶⣤⣤⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀",
        "⠀⠀⠀⠀⠀⠀⠀⠀⣰⡏⠀⠀⠁⠀⠀⠈⠁⠀⢹⣇⠀⠀⠀⠀⠀⠀⠀⠀",
        "⠀⠀⠀⠀⠀⠀⠀⡾⠋⠉⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⢳⡄⠀⠀⠀⠀⠀⠀",
        "⢀⡾⠛⠛⠀⢶⣤⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣾⣧⡶⠞⠛⠛⢳⡄",
        "⠈⣷⡀⠀⠀⠀⠈⣟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⡇⠀⠀⠀⠀⣼⠃",
        "⠀⠈⠻⣦⣤⣤⣤⡿⢷⡦⠀⠀⠀⠀⠀⠀⠀⠀⢰⡶⢿⣤⣤⣤⣴⠾⠃⠀",
        "⠀⢠⡾⠋⠁⢸⠏⠀⠈⢷⣤⣤⣦⠀⠀⣰⣤⣠⡼⠃⠀⠘⣧⠈⠉⢻⡆⠀",
        "⠀⢈⣷⡄⠀⣿⠀⠀⠀⣴⡄⠀⠙⠛⠛⠋⠀⢠⣦⡄⠀⠀⣿⠀⢠⣾⡃⠀",
        "⠀⣾⠁⠀⠀⣿⠀⠀⠀⠉⠁⠀⠀⠀⠀⠀⠀⠀⠉⠀⠀⠀⣼⠀⠀⠈⣿⠀",
        "⠀⢘⣿⠆⠀⢻⡄⠀⠀⠀⢀⠀⠈⣳⣾⠃⠀⡀⠀⠀⠀⢀⣿⠀⠰⢾⡋⠀",
        "⠀⠘⣧⣀⡀⠈⣷⡀⠀⠀⠈⠛⠛⠋⠉⠛⠛⠉⠀⠀⢀⣼⠃⢀⣀⣸⠇⠀",
        "⠀⠀⠈⣿⠃⠀⠈⠻⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠾⠁⠀⠈⣿⠁⠀⠀",
        "⠀⠀⠀⠘⠷⢶⡖⠀⠈⠛⠶⣤⣤⣤⣤⣤⣤⠶⠛⠁⠀⢰⡦⠾⠋⠀⠀⠀",
        "⠀⠀⠀⠀⠀⠘⢷⣤⣴⡆⠀⠀⡀⠀⠀⢀⠀⠀⢠⣦⣤⡾⠃⠀⠀⠀⠀⠀",
        "⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⠶⠾⢻⣤⣤⡾⠷⠶⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀",
    ];

    // Info appears to the right of lines 11-14 (lower body area)
    let info_start: usize = 11;
    let art_width: usize = art.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    let pad = 3;

    for (i, line) in art.iter().enumerate() {
        let idx = i.wrapping_sub(info_start);
        if idx < info.len() {
            let chars = line.chars().count();
            let padding = " ".repeat(art_width - chars + pad);
            let (label, value) = &info[idx];
            if label.is_empty() {
                println!("{line}{padding}{value}");
            } else {
                println!("{line}{padding}{label}  {value}");
            }
        } else {
            println!("{line}");
        }
    }
    println!();
}
