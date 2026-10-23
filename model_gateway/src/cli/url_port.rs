//! The `--prefill <url> [bootstrap_port|none]` and `--encode <url>
//! [bootstrap_port|none]` occurrences of the command line: clap cannot
//! express the optional trailing port, so they are read and stripped here
//! before clap parses the rest.

/// The command line as the gateway reads it, before clap sees it: the
/// `--prefill <url> [bootstrap_port|none]` and `--encode <url>
/// [bootstrap_port|none]` occurrences are hand-parsed (clap cannot express
/// the optional trailing port) and stripped from the arguments clap parses.
pub(crate) struct CommandLine {
    /// argv without the `--prefill` / `--encode` occurrences and their values.
    pub(crate) clap_args: Vec<String>,
    pub(crate) prefill_urls: Vec<(String, Option<u16>)>,
    pub(crate) encode_urls: Vec<(String, Option<u16>)>,
}

impl CommandLine {
    /// Split `args` (argv, the program name first). Each url list is what
    /// [`parse_url_port_args`] reads for its flag over the whole of `args`,
    /// independently of the other flag; `clap_args` is `args` with every
    /// `<flag> <url> [port|none]` run removed by the same token rule, so a
    /// flag given without a url (last on the line) reaches clap, which
    /// refuses it as unexpected.
    pub(crate) fn split(args: &[String]) -> Self {
        Self {
            clap_args: strip_url_port_args(args),
            prefill_urls: parse_url_port_args(args, "--prefill"),
            encode_urls: parse_url_port_args(args, "--encode"),
        }
    }
}

/// Whether the token after a `--prefill` / `--encode` url is its optional
/// bootstrap port: a port number or `none` (any case). Another flag, or a
/// url, is not consumed.
fn is_bootstrap_port_token(token: &str) -> bool {
    !token.starts_with("--") && (token.parse::<u16>().is_ok() || token.to_lowercase() == "none")
}

/// Parse the repeated `<flag> <url> [bootstrap_port|none]` occurrences of
/// `args` into (url, optional bootstrap port) pairs; `none` and an absent
/// port both read as `None`.
fn parse_url_port_args(args: &[String], flag: &str) -> Vec<(String, Option<u16>)> {
    let mut entries = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag && i + 1 < args.len() {
            let url = args[i + 1].clone();
            let mut bootstrap_port = None;
            if args
                .get(i + 2)
                .is_some_and(|token| is_bootstrap_port_token(token))
            {
                bootstrap_port = args[i + 2].parse::<u16>().ok();
                i += 1;
            }
            entries.push((url, bootstrap_port));
            i += 2;
        } else {
            i += 1;
        }
    }
    entries
}

/// `args` without the `--prefill` / `--encode` occurrences and their url and
/// optional port, by the token rule of [`parse_url_port_args`].
fn strip_url_port_args(args: &[String]) -> Vec<String> {
    let mut kept = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if (args[i] == "--prefill" || args[i] == "--encode") && i + 1 < args.len() {
            i += 2;
            if args
                .get(i)
                .is_some_and(|token| is_bootstrap_port_token(token))
            {
                i += 1;
            }
        } else {
            kept.push(args[i].clone());
            i += 1;
        }
    }
    kept
}
