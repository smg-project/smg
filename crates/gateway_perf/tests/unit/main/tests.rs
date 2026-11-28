use super::*;

#[test]
fn the_base_takes_gateway_flags_of_its_own_beside_a_base_binary() {
    for mode in ["cpu", "alloc", "mem", "live"] {
        let cli = Cli::try_parse_from([
            "gateway-perf",
            mode,
            "--base",
            "/b",
            "--base-gw-args",
            "--parser symphony",
            "--gw-args",
            "--parser symphony --trust-tenant-header",
        ])
        .unwrap();
        let (base, head) = match cli.command {
            Cmd::Cpu(c) | Cmd::Alloc(c) => (c.base_gw_args, c.bed.gw_args),
            Cmd::Mem(m) => (m.base_gw_args, m.bed.gw_args),
            Cmd::Live(l) => (l.base_gw_args, l.bed.gw_args),
            _ => panic!("{mode} has no base"),
        };
        assert_eq!(base.as_deref(), Some("--parser symphony"), "{mode}");
        assert_eq!(head, "--parser symphony --trust-tenant-header", "{mode}");
    }
    let alone = Cli::try_parse_from(["gateway-perf", "live", "--base-gw-args", "--x"]);
    assert_eq!(
        alone.unwrap_err().kind(),
        clap::error::ErrorKind::MissingRequiredArgument,
        "flags of a base that does not run"
    );
}

#[test]
fn every_subcommand_parses_with_defaults() {
    let cli = Cli::try_parse_from(["gateway-perf", "cpu", "--base", "/b", "--head", "/h"]).unwrap();
    let Cmd::Cpu(cpu) = cli.command else {
        panic!("not cpu")
    };
    assert_eq!(cpu.cell.repeats, 3);
    assert_eq!(cpu.cell.requests, 10_000);
    assert_eq!(cpu.cell.paths, None, "the fleet's default list");
    assert_eq!(cpu.bed.counters, Counters::Auto);
    assert_eq!(cpu.base.as_deref(), Some(Path::new("/b")));
    let cli = Cli::try_parse_from([
        "gateway-perf",
        "alloc",
        "--base",
        "/b",
        "--rate-scale",
        "0.5",
        "--repeats",
        "5",
    ])
    .unwrap();
    let Cmd::Alloc(alloc) = cli.command else {
        panic!("not alloc")
    };
    assert_eq!(alloc.cell.rate_scale, 0.5);
    assert_eq!(alloc.cell.repeats, 5);
    let cli = Cli::try_parse_from(["gateway-perf", "mem", "--base", "/b"]).unwrap();
    let Cmd::Mem(mem) = cli.command else {
        panic!("not mem")
    };
    assert_eq!(mem.paths, "stream");
    assert_eq!(mem.windows, 5);
    assert_eq!(mem.burst, 2.0);
    assert_eq!(mem.repeats, 1);
    assert_eq!(mem.cell_args().requests, 5_000);
    let cli = Cli::try_parse_from(["gateway-perf", "live", "--rates", "100,200"]).unwrap();
    let Cmd::Live(live) = cli.command else {
        panic!("not live")
    };
    assert_eq!(live.rates, "100,200");
    assert_eq!(live.slo_ttft_p99_ms, 100.0);
    assert!(live.base.is_none());
    let cli = Cli::try_parse_from(["gateway-perf", "bed", "smoke"]).unwrap();
    let Cmd::Bed(bed) = cli.command else {
        panic!("not bed")
    };
    assert_eq!(bed.action, BedAction::Smoke);
    assert_eq!(bed.bed.fleet, Fleet::Grpc);
    assert_eq!(bed.bed.prefill_workers, 1);
    let cli = Cli::try_parse_from([
        "gateway-perf",
        "cpu",
        "--base",
        "/b",
        "--fleet",
        "prefill-decode",
        "--prefill-workers",
        "2",
    ])
    .unwrap();
    let Cmd::Cpu(pd) = cli.command else {
        panic!("not cpu")
    };
    assert_eq!(pd.bed.fleet, Fleet::PrefillDecode);
    assert_eq!(pd.bed.prefill_workers, 2);
    assert!(Cli::try_parse_from([
        "gateway-perf",
        "cpu",
        "--base",
        "/b",
        "--fleet",
        "prefill-decode",
        "--prefill-workers",
        "0",
    ])
    .is_err());
    assert!(
        Cli::try_parse_from(["gateway-perf", "cpu", "--base", "/b", "--workers", "0"]).is_err()
    );
    let cli = Cli::try_parse_from([
        "gateway-perf",
        "bed",
        "smoke",
        "--fleet",
        "http",
        "--paths",
        "chat",
    ])
    .unwrap();
    let Cmd::Bed(http) = cli.command else {
        panic!("not bed")
    };
    assert_eq!(http.bed.fleet, Fleet::Http);
    assert_eq!(http.paths.as_deref(), Some("chat"));
    assert!(Cli::try_parse_from(["gateway-perf", "cpu", "--base", "/b", "--fleet", "pd"]).is_err());
    let cli = Cli::try_parse_from(["gateway-perf", "report", "--out", "/tmp/x"]).unwrap();
    let Cmd::Report(report) = cli.command else {
        panic!("not report")
    };
    assert_eq!(report.out, Path::new("/tmp/x"));
    assert!(Cli::try_parse_from(["gateway-perf", "report"]).is_err());
    let cli =
        Cli::try_parse_from(["gateway-perf", "tokenizer", "/tmp/tok", "--words", "a.txt"]).unwrap();
    let Cmd::Tokenizer(tok) = cli.command else {
        panic!("not tokenizer")
    };
    assert_eq!(tok.words, [PathBuf::from("a.txt")]);
    let cli = Cli::try_parse_from([
        "gateway-perf",
        "base",
        "origin/main",
        "target/gateway-perf/base/smg",
    ])
    .unwrap();
    let Cmd::Base(base) = cli.command else {
        panic!("not base")
    };
    assert_eq!(base.rev, "origin/main");
}

#[test]
fn a_bed_mode_needs_the_checkout_and_the_base() {
    let dir = tempfile::tempdir().unwrap();
    let bed = BedArgs {
        repo: dir.path().to_path_buf(),
        head: None,
        mock: None,
        load: None,
        tokenizer: String::new(),
        out: None,
        workers: 4,
        fleet: Fleet::Grpc,
        prefill_workers: 1,
        port_base: 0,
        gw_cores: String::new(),
        mock_cores: String::new(),
        gen_cores: String::new(),
        gw_threads: 8,
        gw_args: DEFAULT_GW_ARGS.to_string(),
        counters: Counters::Off,
    };
    let err = settings(&bed, None, true, None).unwrap_err().to_string();
    assert!(err.contains("no bed script"), "{err}");
    fs::create_dir_all(dir.path().join("scripts/bench/grpc_stress")).unwrap();
    fs::write(dir.path().join("scripts/bench/grpc_stress/bed.sh"), "").unwrap();
    let out = dir.path().join("out");
    let bed = BedArgs {
        out: Some(out.clone()),
        ..bed
    };
    let err = settings(&bed, None, true, None).unwrap_err().to_string();
    assert!(err.contains("--base is required"), "{err}");
    assert!(out.join("tokenizer").join("tokenizer.json").exists());
    let err = settings(&bed, None, false, None).unwrap_err().to_string();
    assert!(err.contains("no executable gateway for pr"), "{err}");
}

#[test]
fn report_reads_a_results_file() {
    let dir = tempfile::tempdir().unwrap();
    let results = r#"{"unit": "", "note": "n", "repeats": 1, "base": {"runs": [{"chat/allocation_calls_per_request": 300.0}]}, "pr": {"runs": [{"chat/allocation_calls_per_request": 290.0}]}}"#;
    fs::write(dir.path().join("results.json"), results).unwrap();
    let rc = run_report(&ReportCmd {
        out: dir.path().to_path_buf(),
        gates: None,
        setup: String::new(),
    })
    .unwrap();
    assert_eq!(rc, 0);
    assert!(run_report(&ReportCmd {
        out: dir.path().join("nope"),
        gates: None,
        setup: String::new()
    })
    .is_err());
}
