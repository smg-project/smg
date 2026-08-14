"""
Unit tests for the serve command's two-pass argument parsing.

Tests cover parse_serve_args, add_serve_args, _import_backend_args,
and the backend registry (BACKEND_ARG_ADDERS, BACKEND_CHOICES, DEFAULT_BACKEND).
"""

import argparse
import signal
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from smg.serve import (
    BACKEND_ARG_ADDERS,
    BACKEND_CHOICES,
    BACKEND_LAUNCHERS,
    DEFAULT_BACKEND,
    ServeOrchestrator,
    SglangWorkerLauncher,
    TokenspeedWorkerLauncher,
    TrtllmWorkerLauncher,
    VllmWorkerLauncher,
    _add_trtllm_stub_args,
    _find_available_ports,
    _grpc_health_check,
    _http_health_check,
    _import_backend_args,
    _is_port_available,
    _reject_handshake_port_collisions,
    _zmq_handshake_port,
    _zmq_ipc_url,
    add_serve_args,
    parse_serve_args,
)


class TestBackendRegistry:
    """Test the backend registry constants."""

    def test_default_backend_is_sglang(self):
        assert DEFAULT_BACKEND == "sglang"

    def test_backend_choices_match_registry(self):
        assert BACKEND_CHOICES == list(BACKEND_ARG_ADDERS.keys())

    def test_all_backends_registered(self):
        assert "sglang" in BACKEND_ARG_ADDERS
        assert "vllm" in BACKEND_ARG_ADDERS
        assert "trtllm" in BACKEND_ARG_ADDERS
        assert "tokenspeed" in BACKEND_ARG_ADDERS

    def test_registry_values_are_callable(self):
        for name, adder in BACKEND_ARG_ADDERS.items():
            assert callable(adder), f"Backend {name} adder is not callable"


class TestBackendLauncherRegistry:
    """Test the BACKEND_LAUNCHERS registry."""

    def test_all_backends_have_launchers(self):
        assert "sglang" in BACKEND_LAUNCHERS
        assert "vllm" in BACKEND_LAUNCHERS
        assert "trtllm" in BACKEND_LAUNCHERS
        assert "tokenspeed" in BACKEND_LAUNCHERS

    def test_launcher_classes(self):
        assert BACKEND_LAUNCHERS["sglang"] is SglangWorkerLauncher
        assert BACKEND_LAUNCHERS["vllm"] is VllmWorkerLauncher
        assert BACKEND_LAUNCHERS["trtllm"] is TrtllmWorkerLauncher


class TestAddServeArgs:
    """Test add_serve_args populates parser correctly."""

    def test_adds_backend_arg(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args(["--backend", "vllm"])
        assert args.backend == "vllm"

    def test_backend_default_is_sglang(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args([])
        assert args.backend == "sglang"

    def test_backend_rejects_invalid_choice(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--backend", "nonexistent"])

    def test_adds_data_parallel_size(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args(["--dp-size", "4"])
        assert args.data_parallel_size == 4

    def test_data_parallel_size_default(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args([])
        assert args.data_parallel_size == 1

    def test_adds_connection_mode(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args(["--connection-mode", "http"])
        assert args.connection_mode == "http"

    def test_connection_mode_default_is_grpc(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args([])
        assert args.connection_mode == "grpc"

    def test_adds_worker_host(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args(["--worker-host", "0.0.0.0"])
        assert args.worker_host == "0.0.0.0"

    def test_worker_host_default(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args([])
        assert args.worker_host == "127.0.0.1"

    def test_adds_worker_base_port(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args(["--worker-base-port", "40000"])
        assert args.worker_base_port == 40000

    def test_worker_base_port_default(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args([])
        assert args.worker_base_port == 31000

    def test_adds_worker_startup_timeout(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args(["--worker-startup-timeout", "600"])
        assert args.worker_startup_timeout == 600

    def test_worker_startup_timeout_default(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args([])
        assert args.worker_startup_timeout == 300

    def test_host_default_is_localhost(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args([])
        assert args.host == "127.0.0.1"

    def test_enable_token_usage_details_default_false(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args([])
        assert args.enable_token_usage_details is False

    def test_enable_token_usage_details_enabled(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args(["--enable-token-usage-details"])
        assert args.enable_token_usage_details is True


class TestImportBackendArgs:
    """Test _import_backend_args for each backend."""

    def test_trtllm_adds_model_arg(self):
        parser = argparse.ArgumentParser()
        _import_backend_args("trtllm", parser)
        args, backend_args = parser.parse_known_args(
            ["--model", "/path/to/model", "--config", "/path/to/config.yml"]
        )
        assert args.model_path == "/path/to/model"
        assert "--config" in backend_args
        assert "/path/to/config.yml" in backend_args

    def test_sglang_import_error(self):
        """sglang is not installed in test env, so parser.error should be called."""
        parser = argparse.ArgumentParser()
        with pytest.raises(SystemExit) as exc_info:
            _import_backend_args("sglang", parser)
        assert exc_info.value.code == 2

    def test_vllm_import_error(self):
        """vllm is not installed in test env, so parser.error should be called."""
        parser = argparse.ArgumentParser()
        with pytest.raises(SystemExit) as exc_info:
            _import_backend_args("vllm", parser)
        assert exc_info.value.code == 2


class TestAddTrtllmStubArgs:
    """Test the TRT-LLM stub argument group."""

    def test_adds_model_arg(self):
        parser = argparse.ArgumentParser()
        _add_trtllm_stub_args(parser)
        args = parser.parse_args(["--model", "/tmp/model"])
        assert args.model_path == "/tmp/model"

    def test_model_default_is_none(self):
        parser = argparse.ArgumentParser()
        _add_trtllm_stub_args(parser)
        args = parser.parse_args([])
        assert args.model_path is None


class TestParseServeArgs:
    """Test the two-pass parse_serve_args function."""

    def test_trtllm_basic(self):
        backend, args, backend_args = parse_serve_args(
            ["--backend", "trtllm", "--model", "/tmp/m", "--config", "/tmp/config.yml"]
        )
        assert backend == "trtllm"
        assert args.backend == "trtllm"
        assert args.model_path == "/tmp/m"
        assert "--config" in backend_args  # config should be in backend_args
        assert "/tmp/config.yml" in backend_args

    def test_trtllm_defaults(self):
        backend, args, _ = parse_serve_args(["--backend", "trtllm"])
        assert backend == "trtllm"
        assert args.data_parallel_size == 1
        assert args.worker_host == "127.0.0.1"
        assert args.worker_base_port == 31000
        assert args.worker_startup_timeout == 300
        assert args.connection_mode == "grpc"

    def test_trtllm_with_serve_args(self):
        backend, args, backend_args = parse_serve_args(
            [
                "--backend",
                "trtllm",
                "--dp-size",
                "8",
                "--worker-host",
                "0.0.0.0",
                "--worker-base-port",
                "35000",
                "--worker-startup-timeout",
                "600",
            ]
        )
        assert backend == "trtllm"
        assert args.data_parallel_size == 8
        assert args.worker_host == "0.0.0.0"
        assert args.worker_base_port == 35000
        assert args.worker_startup_timeout == 600

    def test_trtllm_includes_router_args(self):
        """Router args should be included with --router- prefix."""
        backend, args, backend_args = parse_serve_args(
            [
                "--backend",
                "trtllm",
                "--router-policy",
                "round_robin",
            ]
        )
        assert args.router_policy == "round_robin"

    def test_trtllm_accepts_router_disable_arg_fallback_flag(self):
        """--router-disable-arg-fallback should parse and be available on the namespace."""
        _, args, _ = parse_serve_args(
            [
                "--backend",
                "trtllm",
                "--router-disable-arg-fallback",
            ]
        )
        assert args.router_disable_arg_fallback is True

    def test_trtllm_router_args_defaults(self):
        """Router args should have sensible defaults."""
        _, args, _ = parse_serve_args(["--backend", "trtllm"])
        assert args.router_policy == "cache_aware"
        assert args.router_pd_disaggregation is False
        assert args.router_disable_retries is False

    def test_default_backend_is_sglang_exits(self):
        """Default backend (sglang) is not installed, so should exit with error."""
        with pytest.raises(SystemExit) as exc_info:
            parse_serve_args([])
        assert exc_info.value.code == 2

    def test_sglang_explicit_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            parse_serve_args(["--backend", "sglang"])
        assert exc_info.value.code == 2

    def test_vllm_explicit_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            parse_serve_args(["--backend", "vllm"])
        assert exc_info.value.code == 2

    def test_vllm_http_accepts_frontend_args(self):
        def _mock_vllm_args(backend, parser):
            if backend == "vllm":
                parser.add_argument("--model", type=str)
            else:
                _import_backend_args(backend, parser)

        def _mock_vllm_frontend_args(parser):
            parser.add_argument("--enable-auto-tool-choice", action="store_true")
            parser.add_argument("--tool-call-parser", type=str)

        with (
            patch("smg.serve._import_backend_args", side_effect=_mock_vllm_args),
            patch("smg.serve._add_vllm_frontend_args", side_effect=_mock_vllm_frontend_args),
        ):
            backend, args, backend_args = parse_serve_args(
                [
                    "--backend",
                    "vllm",
                    "--connection-mode",
                    "http",
                    "--model",
                    "/tmp/model",
                    "--enable-auto-tool-choice",
                    "--tool-call-parser",
                    "minimax_m2",
                ]
            )

        assert backend == "vllm"
        assert args.connection_mode == "http"
        assert args.model == "/tmp/model"
        assert args.enable_auto_tool_choice is True
        assert args.tool_call_parser == "minimax_m2"
        assert "--enable-auto-tool-choice" in backend_args
        assert "--tool-call-parser" in backend_args
        assert "minimax_m2" in backend_args

    def test_vllm_grpc_rejects_frontend_args(self):
        def _mock_vllm_args(backend, parser):
            if backend == "vllm":
                parser.add_argument("--model", type=str)
            else:
                _import_backend_args(backend, parser)

        with (
            patch("smg.serve._import_backend_args", side_effect=_mock_vllm_args),
            patch("smg.serve._add_vllm_frontend_args") as mock_frontend_args,
        ):
            with pytest.raises(SystemExit) as exc_info:
                parse_serve_args(
                    [
                        "--backend",
                        "vllm",
                        "--connection-mode",
                        "grpc",
                        "--model",
                        "/tmp/model",
                        "--enable-auto-tool-choice",
                        "--tool-call-parser",
                        "minimax_m2",
                    ]
                )

        assert exc_info.value.code == 2
        mock_frontend_args.assert_not_called()

    def test_invalid_backend_exits(self):
        with pytest.raises(SystemExit):
            parse_serve_args(["--backend", "nonexistent"])

    def test_none_argv_uses_default(self):
        """parse_serve_args(None) should behave like empty list (default backend)."""
        with pytest.raises(SystemExit) as exc_info:
            parse_serve_args(None)
        assert exc_info.value.code == 2

    def test_help_exits_zero(self):
        """--help should display help and exit with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            parse_serve_args(["--backend", "trtllm", "--help"])
        assert exc_info.value.code == 0

    def test_two_pass_extracts_backend_first(self):
        """Backend-specific args should not cause errors during pass 1."""
        # --model is only valid for trtllm; pass 1 should ignore it
        backend, args, backend_args = parse_serve_args(
            [
                "--backend",
                "trtllm",
                "--model",
                "/some/path",
            ]
        )
        assert backend == "trtllm"
        assert args.model_path == "/some/path"

    def test_unknown_arg_rejected_in_pass2(self):
        """Unknown args should be rejected by the full parser in pass 2."""
        with pytest.raises(SystemExit):
            parse_serve_args(["--backend", "sglang", "--totally-unknown-flag"])


# ---------------------------------------------------------------------------
# Worker launcher tests
# ---------------------------------------------------------------------------


class TestWorkerLauncherGpuEnv:
    """Test GPU assignment via _get_tp_size() and gpu_env().

    Integration tests go through parse_serve_args to verify that CLI flags
    actually produce the correct attribute names for each launcher.
    """

    # -- Integration: CLI flags → _get_tp_size → gpu_env ---------------------

    @pytest.mark.parametrize(
        "backend, cli_flag, tp_value, expected_devices",
        [
            ("sglang", "--tp-size", "4", "0,1,2,3"),
            ("sglang", "--tensor-parallel-size", "2", "0,1"),
        ],
    )
    def test_sglang_tp_from_cli(self, backend, cli_flag, tp_value, expected_devices):
        """CLI --tp-size / --tensor-parallel-size flows through to CUDA_VISIBLE_DEVICES."""

        def _mock_sglang_args(b, parser):
            if b == "sglang":
                parser.add_argument("--tensor-parallel-size", "--tp-size", type=int, default=1)
                parser.add_argument("--model-path", type=str)
            else:
                _import_backend_args(b, parser)

        with patch("smg.serve._import_backend_args", side_effect=_mock_sglang_args):
            _, args, _ = parse_serve_args(
                ["--backend", backend, "--model-path", "/tmp/m", cli_flag, tp_value]
            )
        launcher = SglangWorkerLauncher()
        env = launcher.gpu_env(args, dp_rank=0, env={})
        assert env["CUDA_VISIBLE_DEVICES"] == expected_devices

    @pytest.mark.parametrize(
        "cli_flag, tp_value, expected_devices",
        [
            ("--tensor-parallel-size", "2", "0,1"),
        ],
    )
    def test_vllm_tp_from_cli(self, cli_flag, tp_value, expected_devices):
        """CLI --tensor-parallel-size flows through to CUDA_VISIBLE_DEVICES for vllm."""

        def _mock_vllm_args(b, parser):
            if b == "vllm":
                parser.add_argument("--tensor-parallel-size", type=int, default=1)
                parser.add_argument("--model", type=str)
            else:
                _import_backend_args(b, parser)

        with patch("smg.serve._import_backend_args", side_effect=_mock_vllm_args):
            _, args, _ = parse_serve_args(
                ["--backend", "vllm", "--model", "/tmp/m", cli_flag, tp_value]
            )
        launcher = VllmWorkerLauncher()
        env = launcher.gpu_env(args, dp_rank=0, env={})
        assert env["CUDA_VISIBLE_DEVICES"] == expected_devices

    def test_trtllm_tp_from_cli(self):
        """CLI --tp_size flows through to CUDA_VISIBLE_DEVICES for trtllm."""
        _, args, _ = parse_serve_args(
            ["--backend", "trtllm", "--model-path", "/tmp/m", "--tp_size", "4"]
        )
        launcher = TrtllmWorkerLauncher()
        env = launcher.gpu_env(args, dp_rank=0, env={})
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"

    # -- Unit: _get_tp_size with direct Namespace (TRT-LLM config paths) ------

    @pytest.mark.parametrize(
        "launcher_class, args, expected_tp",
        [
            (TrtllmWorkerLauncher, argparse.Namespace(tp_size=8), 8),
            (TrtllmWorkerLauncher, argparse.Namespace(tensor_parallel_size=8), 8),
            (TrtllmWorkerLauncher, argparse.Namespace(), 1),
        ],
    )
    def test_get_tp_size_trtllm(self, launcher_class, args, expected_tp):
        """TRT-LLM _get_tp_size supports both tp_size and tensor_parallel_size attrs."""
        launcher = launcher_class()
        assert launcher._get_tp_size(args) == expected_tp

    # -- Unit: gpu_env math ---------------------------------------------------

    def test_gpu_env_dp_rank_0_tp_2(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(tensor_parallel_size=2)
        env = launcher.gpu_env(args, dp_rank=0, env={})
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1"

    def test_gpu_env_dp_rank_1_tp_2(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(tensor_parallel_size=2)
        env = launcher.gpu_env(args, dp_rank=1, env={})
        assert env["CUDA_VISIBLE_DEVICES"] == "2,3"

    def test_gpu_env_dp_rank_2_tp_4(self):
        launcher = VllmWorkerLauncher()
        args = argparse.Namespace(tensor_parallel_size=4)
        env = launcher.gpu_env(args, dp_rank=2, env={})
        assert env["CUDA_VISIBLE_DEVICES"] == "8,9,10,11"

    def test_gpu_env_tp_1_default(self):
        launcher = TrtllmWorkerLauncher()
        args = argparse.Namespace()  # no tp_size → default 1
        env = launcher.gpu_env(args, dp_rank=3, env={})
        assert env["CUDA_VISIBLE_DEVICES"] == "3"

    def test_gpu_env_preserves_existing_env(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(tensor_parallel_size=1)
        base_env = {"PATH": "/usr/bin", "HOME": "/root"}
        env = launcher.gpu_env(args, dp_rank=0, env=base_env)
        assert env["PATH"] == "/usr/bin"
        assert env["HOME"] == "/root"
        assert env["CUDA_VISIBLE_DEVICES"] == "0"

    def test_gpu_env_does_not_mutate_input(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(tensor_parallel_size=1)
        base_env = {"FOO": "bar"}
        env = launcher.gpu_env(args, dp_rank=0, env=base_env)
        assert "CUDA_VISIBLE_DEVICES" not in base_env
        assert "CUDA_VISIBLE_DEVICES" in env

    def test_gpu_env_none_copies_os_environ(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(tensor_parallel_size=1)
        with patch.dict("os.environ", {"TEST_VAR": "123"}, clear=False):
            env = launcher.gpu_env(args, dp_rank=0)
        assert env["TEST_VAR"] == "123"
        assert env["CUDA_VISIBLE_DEVICES"] == "0"

    # -- Unit: gpu_env respects existing CUDA_VISIBLE_DEVICES ----------------

    @pytest.mark.parametrize(
        "visible, tp, dp_rank, expected",
        [
            ("4", 1, 0, "4"),
            ("4,5,6,7", 2, 0, "4,5"),
            ("4,5,6,7", 2, 1, "6,7"),
            ("4,5,6,7", 1, 3, "7"),
        ],
    )
    def test_gpu_env_indexes_into_visible_pool(self, visible, tp, dp_rank, expected):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(tensor_parallel_size=tp)
        env = launcher.gpu_env(args, dp_rank=dp_rank, env={"CUDA_VISIBLE_DEVICES": visible})
        assert env["CUDA_VISIBLE_DEVICES"] == expected

    def test_gpu_env_visible_pool_vllm(self):
        launcher = VllmWorkerLauncher()
        args = argparse.Namespace(tensor_parallel_size=2)
        env = launcher.gpu_env(args, dp_rank=1, env={"CUDA_VISIBLE_DEVICES": "4,5,6,7"})
        assert env["CUDA_VISIBLE_DEVICES"] == "6,7"

    # -- Unit: gpu_env bounds check on visible pool --------------------------

    @pytest.mark.parametrize(
        "visible, tp, dp_rank",
        [
            ("4,5", 2, 1),  # 2 GPUs, need index 2..3
            ("4", 1, 1),  # 1 GPU, need index 1
        ],
    )
    def test_gpu_env_raises_on_insufficient_visible_gpus(self, visible, tp, dp_rank):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(tensor_parallel_size=tp)
        with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES has"):
            launcher.gpu_env(args, dp_rank=dp_rank, env={"CUDA_VISIBLE_DEVICES": visible})

    def test_gpu_env_raises_on_invalid_tp_size(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(tensor_parallel_size=0)
        with pytest.raises(ValueError, match="tp_size must be positive"):
            launcher.gpu_env(args, dp_rank=0, env={})

    def test_gpu_env_empty_visible_falls_back(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(tensor_parallel_size=2)
        env = launcher.gpu_env(args, dp_rank=0, env={"CUDA_VISIBLE_DEVICES": ""})
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1"


class TestFilterBackendArgs:
    """Test _filter_backend_args handles both --key value and --key=value."""

    def test_filters_key_value_pair(self):
        launcher = SglangWorkerLauncher()
        backend_args = ["--model-path", "/tmp/m", "--trust-remote-code"]
        result = launcher._filter_backend_args(backend_args, ["--model-path"])
        assert result == ["--trust-remote-code"]

    def test_filters_key_equals_value(self):
        launcher = SglangWorkerLauncher()
        backend_args = ["--model-path=/tmp/m", "--trust-remote-code"]
        result = launcher._filter_backend_args(backend_args, ["--model-path"])
        assert result == ["--trust-remote-code"]

    def test_filters_multiple_keys(self):
        launcher = SglangWorkerLauncher()
        backend_args = ["--model", "/tmp/m", "--host", "0.0.0.0", "--extra-flag"]
        result = launcher._filter_backend_args(backend_args, ["--model", "--host"])
        assert result == ["--extra-flag"]

    def test_empty_backend_args(self):
        launcher = SglangWorkerLauncher()
        assert launcher._filter_backend_args([], ["--model"]) == []

    def test_no_filter_args(self):
        launcher = SglangWorkerLauncher()
        backend_args = ["--foo", "bar"]
        assert launcher._filter_backend_args(backend_args, []) == ["--foo", "bar"]

    def test_filtered_boolean_flag_does_not_swallow_next_flag(self):
        # A filtered boolean flag has no value; the following flag must survive.
        launcher = SglangWorkerLauncher()
        backend_args = ["--trust-remote-code", "--tp-size", "2"]
        result = launcher._filter_backend_args(backend_args, ["--trust-remote-code"])
        assert result == ["--tp-size", "2"]

    def test_filtered_boolean_flag_at_end_of_args(self):
        launcher = SglangWorkerLauncher()
        backend_args = ["--tp-size", "2", "--trust-remote-code"]
        result = launcher._filter_backend_args(backend_args, ["--trust-remote-code"])
        assert result == ["--tp-size", "2"]

    def test_filtered_key_value_keeps_following_flag(self):
        launcher = SglangWorkerLauncher()
        backend_args = ["--model-path", "/tmp/m", "--trust-remote-code"]
        result = launcher._filter_backend_args(backend_args, ["--model-path"])
        assert result == ["--trust-remote-code"]


class TestZmqHandshakePort:
    """Handshake-port derivation must mirror derive_handshake_port in worker.rs."""

    def test_matches_pinned_rust_vectors(self):
        # These MUST equal derive_handshake_port() in worker.rs for the same path.
        assert _zmq_handshake_port("ipc:///tmp/smg-zmq/ts0.ipc") == 25152
        assert _zmq_handshake_port("ipc:///tmp/smg-zmq/ts1.ipc") == 22735
        assert _zmq_handshake_port("ipc:///tmp/smg-zmq/engine-31000") == 22714
        # The path after ipc:// is what gets hashed, so the bare path matches.
        assert _zmq_handshake_port("/tmp/smg-zmq/ts0.ipc") == 25152
        assert _zmq_handshake_port("ts0.ipc") == 23912

    def test_always_in_band(self):
        for port in range(30000, 30050):
            hp = _zmq_handshake_port(_zmq_ipc_url(port))
            assert 20000 <= hp <= 29999

    def test_reject_collisions_passes_for_distinct_ports(self):
        # A normal contiguous port range must not raise.
        _reject_handshake_port_collisions([30000, 30001, 30002])

    def test_reject_collisions_raises_and_names_urls(self):
        # Two ports whose ipc paths derive the same handshake port must be
        # rejected before launch, naming both colliding URLs.
        base = 30000
        collider = next(
            p
            for p in range(base + 1, base + 20000)
            if _zmq_handshake_port(_zmq_ipc_url(p)) == _zmq_handshake_port(_zmq_ipc_url(base))
        )
        with pytest.raises(ValueError) as exc:
            _reject_handshake_port_collisions([base, collider])
        assert _zmq_ipc_url(base) in str(exc.value)
        assert _zmq_ipc_url(collider) in str(exc.value)


class TestSglangWorkerLauncher:
    """Test SglangWorkerLauncher.build_command()."""

    def test_build_command_grpc_mode_default(self):
        """Default connection_mode is grpc, so --grpc-mode should be present."""
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(model_path="/tmp/model", connection_mode="grpc")
        backend_args = ["--model-path", "/tmp/model", "--trust-remote-code"]
        cmd = launcher.build_command(args, backend_args, "127.0.0.1", 31000)
        assert "--model-path" in cmd
        assert "/tmp/model" in cmd
        assert "--host" in cmd
        assert "127.0.0.1" in cmd
        assert "--port" in cmd
        assert "31000" in cmd
        assert "--grpc-mode" in cmd
        for arg in backend_args:
            assert arg in cmd

    def test_build_command_http_mode(self):
        """When connection_mode is http, --grpc-mode should not be present."""
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(model_path="/tmp/model", connection_mode="http")
        backend_args = ["--trust-remote-code"]
        cmd = launcher.build_command(args, backend_args, "127.0.0.1", 31000)
        assert "--grpc-mode" not in cmd
        for arg in backend_args:
            assert arg in cmd
        assert "--enable-cache-report" not in cmd

    def test_build_command_http_mode_with_token_usage_details(self):
        """When connection_mode is http and enable_token_usage_details is True,
        --enable-cache-report should be present."""
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(
            model_path="/tmp/model", connection_mode="http", enable_token_usage_details=True
        )
        backend_args = ["--trust-remote-code"]
        cmd = launcher.build_command(args, backend_args, "127.0.0.1", 31000)
        assert "--grpc-mode" not in cmd
        for arg in backend_args:
            assert arg in cmd
        assert "--enable-cache-report" in cmd

    def test_worker_url_grpc_mode(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(connection_mode="grpc")
        assert launcher.worker_url(args, "127.0.0.1", 31000) == "grpc://127.0.0.1:31000"

    def test_worker_url_http_mode(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(connection_mode="http")
        assert launcher.worker_url(args, "127.0.0.1", 31000) == "http://127.0.0.1:31000"

    def test_health_check_grpc_mode(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(connection_mode="grpc")
        with patch("smg.serve._grpc_health_check", return_value=True) as mock:
            result = launcher.health_check(args, "127.0.0.1", 31000, 5.0)
        assert result is True
        mock.assert_called_once_with("127.0.0.1", 31000, 5.0)

    def test_health_check_http_mode(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(connection_mode="http")
        with patch("smg.serve._http_health_check", return_value=True) as mock:
            result = launcher.health_check(args, "127.0.0.1", 31000, 5.0)
        assert result is True
        mock.assert_called_once_with("http://127.0.0.1:31000/health", 5.0)


class TestVllmWorkerLauncher:
    """Test VllmWorkerLauncher.build_command()."""

    def test_build_command(self):
        launcher = VllmWorkerLauncher()
        args = argparse.Namespace(model="/tmp/model", connection_mode="grpc")
        backend_args = ["--trust-remote-code"]
        cmd = launcher.build_command(args, backend_args, "0.0.0.0", 32000)
        assert "vllm.entrypoints.grpc_server" in cmd
        assert "--model" in cmd
        assert "/tmp/model" in cmd
        assert "--host" in cmd
        assert "0.0.0.0" in cmd
        assert "--port" in cmd
        assert "32000" in cmd
        for arg in backend_args:
            assert arg in cmd

    def test_build_zmq_command_defaults_to_single_engine(self):
        launcher = VllmWorkerLauncher()
        args = argparse.Namespace(model="/tmp/model", connection_mode="zmq")
        cmd = launcher.build_command(args, [], "127.0.0.1", 31000)
        assert cmd[cmd.index("--data-parallel-size") + 1] == "1"
        assert cmd[cmd.index("--data-parallel-size-local") + 1] == "1"

    def test_build_zmq_command_grouped_engine_dp(self):
        """Engine-level --data-parallel-size N launches a grouped worker:
        the launcher owns both dp flags (operator's copy is filtered) and
        emits N for size and size-local."""
        launcher = VllmWorkerLauncher()
        args = argparse.Namespace(model="/tmp/model", connection_mode="zmq")
        cmd = launcher.build_command(args, ["--data-parallel-size", "2"], "127.0.0.1", 31000)
        assert cmd.count("--data-parallel-size") == 1
        assert cmd[cmd.index("--data-parallel-size") + 1] == "2"
        assert cmd[cmd.index("--data-parallel-size-local") + 1] == "2"

    def test_build_zmq_command_grouped_engine_dp_equals_form(self):
        launcher = VllmWorkerLauncher()
        args = argparse.Namespace(model="/tmp/model", connection_mode="zmq")
        cmd = launcher.build_command(args, ["--data-parallel-size=4"], "127.0.0.1", 31000)
        assert cmd[cmd.index("--data-parallel-size") + 1] == "4"

    def test_worker_url(self):
        launcher = VllmWorkerLauncher()
        args = argparse.Namespace(connection_mode="grpc")
        assert launcher.worker_url(args, "127.0.0.1", 32000) == "grpc://127.0.0.1:32000"

    def test_health_check_delegates_to_grpc(self):
        launcher = VllmWorkerLauncher()
        args = argparse.Namespace(connection_mode="grpc")
        with patch("smg.serve._grpc_health_check", return_value=True) as mock:
            result = launcher.health_check(args, "127.0.0.1", 32000, 5.0)
        assert result is True
        mock.assert_called_once_with("127.0.0.1", 32000, 5.0)

    def test_build_command_http_mode(self):
        """When connection_mode is http, --grpc-mode should not be present."""
        launcher = VllmWorkerLauncher()
        args = argparse.Namespace(model_path="/tmp/model", connection_mode="http")
        backend_args = ["--trust-remote-code"]
        cmd = launcher.build_command(args, backend_args, "127.0.0.1", 31000)
        assert "vllm.entrypoints.openai.api_server" in cmd
        assert "vllm.entrypoints.grpc_server" not in cmd
        for arg in backend_args:
            assert arg in cmd
        assert "--enable-prompt-tokens-details" not in cmd

    def test_build_command_http_mode_with_token_usage_details(self):
        """When connection_mode is http and enable_token_usage_details is True,
        --enable-prompt-tokens-details should be present."""
        launcher = VllmWorkerLauncher()
        args = argparse.Namespace(
            model="/tmp/model", connection_mode="http", enable_token_usage_details=True
        )
        backend_args = ["--trust-remote-code"]
        cmd = launcher.build_command(args, backend_args, "127.0.0.1", 31000)
        assert "vllm.entrypoints.openai.api_server" in cmd
        assert "vllm.entrypoints.grpc_server" not in cmd
        for arg in backend_args:
            assert arg in cmd
        assert "--enable-prompt-tokens-details" in cmd


class TestTokenspeedWorkerLauncher:
    """Test TokenspeedWorkerLauncher (ZMQ direct-backend only)."""

    def test_build_zmq_command(self):
        launcher = TokenspeedWorkerLauncher()
        args = argparse.Namespace(model="/tmp/model", connection_mode="zmq")
        backend_args = ["--mem-fraction-static", "0.8"]
        cmd = launcher.build_command(args, backend_args, "127.0.0.1", 31000)

        assert "tokenspeed.cli" in cmd
        assert "serve" in cmd
        assert "--headless" in cmd
        assert "--model" in cmd
        assert "/tmp/model" in cmd
        assert "--data-parallel-address" in cmd
        assert "127.0.0.1" in cmd
        # The engine dials the handshake port SMG derives from the worker's
        # ipc:// URL; both sides must agree without a side channel.
        expected_port = _zmq_handshake_port(_zmq_ipc_url(31000))
        assert "--data-parallel-rpc-port" in cmd
        assert str(expected_port) in cmd
        assert "--zmq-engine-index" in cmd
        assert "0" in cmd
        # The launcher owns the engine's control-plane port layout: --port
        # seeds the derived cluster per worker, --dist-init-addr pins the
        # torch.distributed store at TokenSpeed's own dp==1 derivation
        # (port + 233) so dp==1 and dp>1 share one layout.
        assert cmd[cmd.index("--port") + 1] == "31000"
        assert cmd[cmd.index("--dist-init-addr") + 1] == "127.0.0.1:31233"
        for arg in backend_args:
            assert arg in cmd

    def test_build_zmq_command_passes_dp_size_through(self):
        # DP is the engine's flag: the launcher forwards it untouched and the
        # ranks each dial the shared socket set with their own identity.
        launcher = TokenspeedWorkerLauncher()
        args = argparse.Namespace(model="/tmp/model", connection_mode="zmq")
        cmd = launcher.build_command(args, ["--data-parallel-size", "2"], "127.0.0.1", 31000)

        assert cmd[cmd.index("--data-parallel-size") + 1] == "2"
        # dp>1 hard-requires the explicit store address the launcher always passes.
        assert cmd[cmd.index("--dist-init-addr") + 1] == "127.0.0.1:31233"

    def test_build_zmq_command_reflects_dist_port_below_u16_ceiling(self):
        launcher = TokenspeedWorkerLauncher()
        args = argparse.Namespace(model="/tmp/model", connection_mode="zmq")
        cmd = launcher.build_command(args, [], "127.0.0.1", 65500)

        assert cmd[cmd.index("--dist-init-addr") + 1] == f"127.0.0.1:{65500 - 233}"

    def test_dist_port_never_enters_the_handshake_band(self):
        # The handshake port is a hash of the ipc path folded into
        # 20000..=29999, so a dist port inside that band can land on some
        # worker's rpc listener — which port collides depends on the socket
        # dir (uid). The launcher must keep the dist port out of the band
        # entirely; that also implies it never equals this worker's own
        # rpc port. Sweep every worker port whose naive +233 derivation
        # lands in the band, plus the band edges and the u16 reflection.
        launcher = TokenspeedWorkerLauncher()
        args = argparse.Namespace(model="/tmp/model", connection_mode="zmq")
        ports = [*range(19767, 29767), 19766, 29767, 31000, 65500]
        for port in ports:
            cmd = launcher.build_command(args, [], "127.0.0.1", port)
            dist_port = int(cmd[cmd.index("--dist-init-addr") + 1].rsplit(":", 1)[1])
            rpc_port = _zmq_handshake_port(_zmq_ipc_url(port))
            assert not 20000 <= dist_port <= 29999, (port, dist_port)
            assert dist_port != rpc_port, (port, dist_port)
            assert dist_port != port, (port, dist_port)
            assert 1 <= dist_port <= 65535, (port, dist_port)

    def test_build_zmq_command_filters_launcher_owned_flags(self):
        launcher = TokenspeedWorkerLauncher()
        args = argparse.Namespace(model="/tmp/model", connection_mode="zmq")
        backend_args = [
            "--model",
            "/tmp/other-model",
            "--data-parallel-rpc-port",
            "1234",
            "--zmq-engine-index",
            "7",
            "--mem-fraction-static",
            "0.8",
        ]
        cmd = launcher.build_command(args, backend_args, "127.0.0.1", 31000)

        # Launcher-owned flags from backend_args must not duplicate/override.
        assert "/tmp/other-model" not in cmd
        assert "1234" not in cmd
        assert "7" not in cmd
        assert "--mem-fraction-static" in cmd
        assert "0.8" in cmd

    def test_build_zmq_command_enables_constraint_enforcement(self):
        # TokenSpeed defaults its grammar backend to none and drops wire
        # constraints without complaint, so tool_choice=required and
        # response_format=json_schema would answer as freeform text.
        launcher = TokenspeedWorkerLauncher()
        args = argparse.Namespace(model="/tmp/model", connection_mode="zmq")
        cmd = launcher.build_command(args, [], "127.0.0.1", 31000)

        assert cmd[cmd.index("--grammar-backend") + 1] == "xgrammar"
        assert "--enable-output-logprobs" in cmd

    def test_build_zmq_command_defaults_yield_to_the_operator(self):
        launcher = TokenspeedWorkerLauncher()
        args = argparse.Namespace(model="/tmp/model", connection_mode="zmq")
        cmd = launcher.build_command(args, ["--grammar-backend", "llguidance"], "127.0.0.1", 31000)

        assert cmd.count("--grammar-backend") == 1
        assert cmd[cmd.index("--grammar-backend") + 1] == "llguidance"
        # An override of one default leaves the others in place.
        assert "--enable-output-logprobs" in cmd

    def test_build_zmq_command_default_honors_equals_form(self):
        launcher = TokenspeedWorkerLauncher()
        args = argparse.Namespace(model="/tmp/model", connection_mode="zmq")
        cmd = launcher.build_command(args, ["--grammar-backend=none"], "127.0.0.1", 31000)

        assert "--grammar-backend=none" in cmd
        assert "xgrammar" not in cmd

    def test_build_command_rejects_non_zmq_modes(self):
        launcher = TokenspeedWorkerLauncher()
        for mode in ("grpc", "http"):
            args = argparse.Namespace(model="/tmp/model", connection_mode=mode)
            with pytest.raises(ValueError, match="only supports --connection-mode zmq"):
                launcher.build_command(args, [], "127.0.0.1", 31000)

    def test_worker_url_is_ipc(self):
        launcher = TokenspeedWorkerLauncher()
        args = argparse.Namespace(connection_mode="zmq")
        url = launcher.worker_url(args, "127.0.0.1", 31000)
        assert url == _zmq_ipc_url(31000)
        assert url.startswith("ipc://")

    def test_health_check_zmq_always_proceeds(self):
        # SMG binds the sockets and the engine dials in; there is nothing to
        # probe, the router's own health checker gates readiness.
        launcher = TokenspeedWorkerLauncher()
        args = argparse.Namespace(connection_mode="zmq")
        assert launcher.health_check(args, "127.0.0.1", 31000, 5.0) is True

    def test_get_tp_size(self):
        launcher = TokenspeedWorkerLauncher()
        assert launcher._get_tp_size(argparse.Namespace(tensor_parallel_size=4)) == 4
        assert launcher._get_tp_size(argparse.Namespace()) == 1


class TestTokenspeedServeParsing:
    """parse_serve_args for the tokenspeed backend (stub args, zmq guard)."""

    def test_tokenspeed_zmq_parses(self):
        backend, args, backend_args = parse_serve_args(
            [
                "--backend",
                "tokenspeed",
                "--connection-mode",
                "zmq",
                "--model",
                "/tmp/model",
                "--mem-fraction-static",
                "0.8",
            ]
        )
        assert backend == "tokenspeed"
        assert args.connection_mode == "zmq"
        assert args.model == "/tmp/model"
        # Unknown engine flags stay in backend_args for verbatim passthrough.
        assert "--mem-fraction-static" in backend_args
        assert "0.8" in backend_args

    def test_zmq_still_rejected_for_sglang_and_trtllm(self):
        for backend in ("sglang", "trtllm"):
            with pytest.raises(SystemExit) as exc_info:
                parse_serve_args(["--backend", backend, "--connection-mode", "zmq"])
            assert exc_info.value.code == 2


class TestTrtllmWorkerLauncher:
    """Test TrtllmWorkerLauncher.build_command()."""

    def test_build_command(self):
        launcher = TrtllmWorkerLauncher()
        args = argparse.Namespace(model_path="/tmp/model", connection_mode="grpc")
        backend_args = ["--config", "/tmp/config.yml"]
        cmd = launcher.build_command(args, backend_args, "0.0.0.0", 50051)
        assert "tensorrt_llm.commands.serve" in cmd
        assert "/tmp/model" in cmd
        assert "--grpc" in cmd
        assert "--host" in cmd
        assert "0.0.0.0" in cmd
        assert "--port" in cmd
        assert "50051" in cmd
        for arg in backend_args:
            assert arg in cmd

    def test_build_command_rejects_http_mode(self):
        launcher = TrtllmWorkerLauncher()
        args = argparse.Namespace(model_path="/tmp/model", connection_mode="http")
        backend_args = ["--config", "/tmp/config.yml"]
        with pytest.raises(ValueError, match="TensorRT-LLM backend only supports grpc"):
            launcher.build_command(args, backend_args, "0.0.0.0", 50051)

    def test_worker_url(self):
        launcher = TrtllmWorkerLauncher()
        args = argparse.Namespace(connection_mode="grpc")
        assert launcher.worker_url(args, "127.0.0.1", 50051) == "grpc://127.0.0.1:50051"

    def test_health_check_delegates_to_grpc(self):
        launcher = TrtllmWorkerLauncher()
        args = argparse.Namespace(connection_mode="grpc")
        with patch("smg.serve._grpc_health_check", return_value=True) as mock:
            result = launcher.health_check(args, "127.0.0.1", 50051, 5.0)
        assert result is True
        mock.assert_called_once_with("127.0.0.1", 50051, 5.0)

    def test_get_tp_size_from_config_tensor_parallel_size(self, tmp_path):
        """_get_tp_size reads tensor_parallel_size from config YAML when no args set."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("tensor_parallel_size: 4\n")
        launcher = TrtllmWorkerLauncher()
        args = argparse.Namespace(config=str(config_file))
        assert launcher._get_tp_size(args) == 4

    def test_get_tp_size_from_config_tp_size(self, tmp_path):
        """_get_tp_size reads tp_size from config YAML when tensor_parallel_size not present."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("tp_size: 2\n")
        launcher = TrtllmWorkerLauncher()
        args = argparse.Namespace(config=str(config_file))
        assert launcher._get_tp_size(args) == 2

    def test_get_tp_size_from_config_tensor_parallel_size_takes_precedence(self, tmp_path):
        """When both keys exist in config, tensor_parallel_size is used."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("tensor_parallel_size: 8\ntp_size: 2\n")
        launcher = TrtllmWorkerLauncher()
        args = argparse.Namespace(config=str(config_file))
        assert launcher._get_tp_size(args) == 8

    def test_get_tp_size_from_config_read_fails_raises(self):
        """When config file does not exist, FileNotFoundError propagates."""
        launcher = TrtllmWorkerLauncher()
        args = argparse.Namespace(config="/nonexistent/config.yaml")
        with pytest.raises(FileNotFoundError):
            launcher._get_tp_size(args)

    def test_get_tp_size_from_config_empty_or_no_keys_returns_default(self, tmp_path):
        """When config has no tp keys or is empty, warn and return default 1."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("other_key: 42\n")
        launcher = TrtllmWorkerLauncher()
        args = argparse.Namespace(config=str(config_file))
        with patch("smg.serve.logger") as mock_logger:
            result = launcher._get_tp_size(args)
        assert result == 1
        mock_logger.warning.assert_called_once()
        assert (
            "does not contain tensor_parallel_size or tp_size"
            in mock_logger.warning.call_args[0][0]
        )

    def test_get_tp_size_from_config_malformed_yaml_raises(self, tmp_path):
        """When config file contains invalid YAML, error propagates."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{{invalid yaml")
        launcher = TrtllmWorkerLauncher()
        args = argparse.Namespace(config=str(config_file))
        import yaml

        with pytest.raises(yaml.YAMLError):
            launcher._get_tp_size(args)


# ---------------------------------------------------------------------------
# Health check utility tests
# ---------------------------------------------------------------------------


class TestHttpHealthCheck:
    """Test _http_health_check with mocked urllib."""

    def test_returns_true_on_200(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert _http_health_check("http://localhost:31000/health", 5.0) is True

    def test_returns_false_on_error(self):
        with patch("urllib.request.urlopen", side_effect=ConnectionError):
            assert _http_health_check("http://localhost:31000/health", 5.0) is False

    def test_returns_false_on_non_200(self):
        mock_resp = MagicMock()
        mock_resp.status = 503
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert _http_health_check("http://localhost:31000/health", 5.0) is False


class TestGrpcHealthCheck:
    """Test _grpc_health_check with mocked grpc and health stub."""

    def test_returns_true_on_serving(self):
        from grpc_health.v1 import health_pb2, health_pb2_grpc

        mock_channel = MagicMock()
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.status = health_pb2.HealthCheckResponse.SERVING
        mock_stub.Check.return_value = mock_response

        with (
            patch("grpc.insecure_channel", return_value=mock_channel),
            patch.object(health_pb2_grpc, "HealthStub", return_value=mock_stub),
        ):
            assert _grpc_health_check("127.0.0.1", 31000, 5.0) is True
        mock_stub.Check.assert_called_once()

    def test_returns_false_on_error(self):
        with patch("grpc.insecure_channel", side_effect=ConnectionError):
            assert _grpc_health_check("127.0.0.1", 31000, 5.0) is False

    def test_returns_false_on_non_serving(self):
        from grpc_health.v1 import health_pb2, health_pb2_grpc

        mock_channel = MagicMock()
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.status = health_pb2.HealthCheckResponse.NOT_SERVING
        mock_stub.Check.return_value = mock_response

        with (
            patch("grpc.insecure_channel", return_value=mock_channel),
            patch.object(health_pb2_grpc, "HealthStub", return_value=mock_stub),
        ):
            assert _grpc_health_check("127.0.0.1", 31000, 5.0) is False

    def test_returns_true_when_unimplemented_then_channel_ready(self):
        import grpc
        from grpc_health.v1 import health_pb2_grpc

        class UnimplementedRpcError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.UNIMPLEMENTED

        mock_channel = MagicMock()
        mock_stub = MagicMock()
        mock_stub.Check.side_effect = UnimplementedRpcError()
        mock_ready_future = MagicMock()

        with (
            patch("grpc.insecure_channel", return_value=mock_channel),
            patch.object(health_pb2_grpc, "HealthStub", return_value=mock_stub),
            patch("grpc.channel_ready_future", return_value=mock_ready_future),
        ):
            assert _grpc_health_check("127.0.0.1", 31000, 5.0) is True
        mock_ready_future.result.assert_called_once_with(timeout=5.0)

    def test_returns_false_when_unimplemented_and_channel_ready_fails(self):
        import grpc
        from grpc_health.v1 import health_pb2_grpc

        class UnimplementedRpcError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.UNIMPLEMENTED

        mock_channel = MagicMock()
        mock_stub = MagicMock()
        mock_stub.Check.side_effect = UnimplementedRpcError()
        mock_ready_future = MagicMock()
        mock_ready_future.result.side_effect = OSError("connection refused")

        with (
            patch("grpc.insecure_channel", return_value=mock_channel),
            patch.object(health_pb2_grpc, "HealthStub", return_value=mock_stub),
            patch("grpc.channel_ready_future", return_value=mock_ready_future),
        ):
            assert _grpc_health_check("127.0.0.1", 31000, 5.0) is False


# ---------------------------------------------------------------------------
# Port discovery tests
# ---------------------------------------------------------------------------


class TestFindAvailablePorts:
    """Test _find_available_ports returns correct count of ports."""

    def test_returns_requested_count(self):
        with patch("smg.serve._is_port_available", return_value=True):
            ports = _find_available_ports(31000, 3)
        assert len(ports) == 3

    def test_first_port_is_base(self):
        with patch("smg.serve._is_port_available", return_value=True):
            ports = _find_available_ports(31000, 1)
        assert ports[0] == 31000

    def test_skips_unavailable_ports(self):
        # First call: port 31000 unavailable, then 31001 available
        call_count = 0

        def mock_available(port):
            nonlocal call_count
            call_count += 1
            return port != 31000

        with patch("smg.serve._is_port_available", side_effect=mock_available):
            ports = _find_available_ports(31000, 1)
        assert ports[0] == 31001

    def test_ports_are_unique(self):
        with patch("smg.serve._is_port_available", return_value=True):
            ports = _find_available_ports(31000, 5)
        assert len(set(ports)) == 5

    def test_raises_when_no_ports_available(self):
        with patch("smg.serve._is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="Could not find"):
                _find_available_ports(65530, 10)


class TestIsPortAvailable:
    """Test _is_port_available socket check."""

    def test_available_port(self):
        # Port 0 lets the OS pick a free port; we test a high ephemeral port
        # that is almost certainly free
        assert isinstance(_is_port_available(39999), bool)

    def test_occupied_port(self):
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        try:
            assert _is_port_available(port) is False
        finally:
            s.close()


# ---------------------------------------------------------------------------
# ServeOrchestrator tests
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    """Create a minimal argparse.Namespace for orchestrator tests."""
    defaults = {
        "backend": "sglang",
        "data_parallel_size": 2,
        "connection_mode": "grpc",
        "worker_host": "127.0.0.1",
        "worker_base_port": 31000,
        "worker_startup_timeout": 10,
        "model_path": "/tmp/model",
        # router args with router_ prefix
        "router_policy": "cache_aware",
        "router_pd_disaggregation": False,
        "router_disable_retries": False,
        "router_disable_circuit_breaker": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestServeOrchestrator:
    """Test ServeOrchestrator lifecycle methods."""

    def test_build_router_args_injects_worker_urls_grpc(self):
        args = _make_args(data_parallel_size=2, connection_mode="grpc")
        backend_args = ["--model-path", "/tmp/model"]
        orch = ServeOrchestrator("sglang", args, backend_args)
        # Simulate workers already launched
        mock_proc1 = MagicMock()
        mock_proc2 = MagicMock()
        orch.workers = [(mock_proc1, 31000), (mock_proc2, 31003)]

        with patch("smg.serve.RouterArgs.from_cli_args") as mock_from_cli:
            mock_router_args = MagicMock()
            mock_from_cli.return_value = mock_router_args
            result = orch._build_router_args()

        mock_from_cli.assert_called_once_with(args, use_router_prefix=True)
        assert mock_router_args.worker_urls == [
            "grpc://127.0.0.1:31000",
            "grpc://127.0.0.1:31003",
        ]
        assert result is mock_router_args

    def test_build_router_args_http_mode(self):
        args = _make_args(data_parallel_size=2, connection_mode="http")
        backend_args = ["--model-path", "/tmp/model"]
        orch = ServeOrchestrator("sglang", args, backend_args)
        mock_proc = MagicMock()
        orch.workers = [(mock_proc, 31000), (mock_proc, 31003)]

        with patch("smg.serve.RouterArgs.from_cli_args") as mock_from_cli:
            mock_router_args = MagicMock()
            mock_from_cli.return_value = mock_router_args
            orch._build_router_args()

        assert mock_router_args.worker_urls == [
            "http://127.0.0.1:31000",
            "http://127.0.0.1:31003",
        ]

    def test_build_router_args_vllm_grpc_urls(self):
        args = _make_args(
            backend="vllm", data_parallel_size=2, model="/tmp/m", connection_mode="grpc"
        )
        backend_args = ["--model", "/tmp/m"]
        orch = ServeOrchestrator("vllm", args, backend_args)
        mock_proc = MagicMock()
        orch.workers = [(mock_proc, 32000), (mock_proc, 32003)]

        with patch("smg.serve.RouterArgs.from_cli_args") as mock_from_cli:
            mock_router_args = MagicMock()
            mock_from_cli.return_value = mock_router_args
            orch._build_router_args()

        assert mock_router_args.worker_urls == [
            "grpc://127.0.0.1:32000",
            "grpc://127.0.0.1:32003",
        ]

    def test_build_router_args_zmq_forwards_backend(self):
        """ZMQ workers cannot be runtime-probed: serve's --backend must reach
        RouterArgs so the Rust side stamps the startup workers' runtime."""
        from types import SimpleNamespace

        for backend in ("vllm", "tokenspeed"):
            args = _make_args(backend=backend, data_parallel_size=1, connection_mode="zmq")
            orch = ServeOrchestrator(backend, args, [])
            orch.workers = [(MagicMock(), 31000)]

            router_args = SimpleNamespace(
                backend="sglang",
                disable_retries=True,
                disable_circuit_breaker=True,
                policy="passthrough",
            )
            with patch("smg.serve.RouterArgs.from_cli_args", return_value=router_args):
                result = orch._build_router_args()

            assert result.backend == backend
            assert result.worker_urls == [_zmq_ipc_url(31000)]

    def test_build_router_args_grpc_keeps_router_backend(self):
        """Off the ZMQ path the router backend stays whatever --router-backend
        chose (gRPC/HTTP workers keep runtime auto-detection)."""
        from types import SimpleNamespace

        args = _make_args(backend="vllm", data_parallel_size=1, connection_mode="grpc")
        orch = ServeOrchestrator("vllm", args, [])
        orch.workers = [(MagicMock(), 31000)]

        router_args = SimpleNamespace(
            backend="sglang",
            disable_retries=True,
            disable_circuit_breaker=True,
            policy="passthrough",
        )
        with patch("smg.serve.RouterArgs.from_cli_args", return_value=router_args):
            result = orch._build_router_args()

        assert result.backend == "sglang"

    def test_build_router_args_dp1_disables_retries_and_cb(self):
        """With dp<=1, router-level retries and circuit breaker are auto-disabled."""
        from types import SimpleNamespace

        args = _make_args(data_parallel_size=1)
        orch = ServeOrchestrator("sglang", args, ["--model-path", "/tmp/model"])
        orch.workers = [(MagicMock(), 31000)]

        router_args = SimpleNamespace(
            disable_retries=False, disable_circuit_breaker=False, policy="cache_aware"
        )
        with patch("smg.serve.RouterArgs.from_cli_args", return_value=router_args):
            result = orch._build_router_args()

        assert result.disable_retries is True
        assert result.disable_circuit_breaker is True
        assert result.worker_urls == ["grpc://127.0.0.1:31000"]

    def test_build_router_args_dp2_preserves_retries_and_cb(self):
        """With dp>1, router-level retries and circuit breaker are left untouched."""
        from types import SimpleNamespace

        args = _make_args(data_parallel_size=2)
        orch = ServeOrchestrator("sglang", args, ["--model-path", "/tmp/model"])
        orch.workers = [(MagicMock(), 31000), (MagicMock(), 31003)]

        router_args = SimpleNamespace(
            disable_retries=False, disable_circuit_breaker=False, policy="cache_aware"
        )
        with patch("smg.serve.RouterArgs.from_cli_args", return_value=router_args):
            result = orch._build_router_args()

        assert result.disable_retries is False
        assert result.disable_circuit_breaker is False

    def test_build_router_args_dp1_respects_explicit_disable(self):
        """dp=1 auto-disable is a no-op when user already disabled retries/cb."""
        from types import SimpleNamespace

        args = _make_args(data_parallel_size=1)
        orch = ServeOrchestrator("sglang", args, ["--model-path", "/tmp/model"])
        orch.workers = [(MagicMock(), 31000)]

        router_args = SimpleNamespace(
            disable_retries=True, disable_circuit_breaker=True, policy="cache_aware"
        )
        with patch("smg.serve.RouterArgs.from_cli_args", return_value=router_args):
            result = orch._build_router_args()

        assert result.disable_retries is True
        assert result.disable_circuit_breaker is True

    def test_build_router_args_dp1_forces_passthrough_policy(self):
        """dp<=1 overrides an explicit --policy with passthrough (single backend)."""
        from types import SimpleNamespace

        args = _make_args(data_parallel_size=1)
        orch = ServeOrchestrator("sglang", args, ["--model-path", "/tmp/model"])
        orch.workers = [(MagicMock(), 31000)]

        # User explicitly asked for cache_aware; dp<=1 must override it.
        router_args = SimpleNamespace(
            disable_retries=False, disable_circuit_breaker=False, policy="cache_aware"
        )
        with patch("smg.serve.RouterArgs.from_cli_args", return_value=router_args):
            result = orch._build_router_args()

        assert result.policy == "passthrough"

    def test_build_router_args_dp2_preserves_policy(self):
        """dp>1 leaves the chosen policy untouched."""
        from types import SimpleNamespace

        args = _make_args(data_parallel_size=2)
        orch = ServeOrchestrator("sglang", args, ["--model-path", "/tmp/model"])
        orch.workers = [(MagicMock(), 31000), (MagicMock(), 31003)]

        router_args = SimpleNamespace(
            disable_retries=False, disable_circuit_breaker=False, policy="cache_aware"
        )
        with patch("smg.serve.RouterArgs.from_cli_args", return_value=router_args):
            result = orch._build_router_args()

        assert result.policy == "cache_aware"

    def test_cleanup_workers_handles_already_dead_process(self):
        args = _make_args()
        backend_args = ["--model-path", "/tmp/model"]
        orch = ServeOrchestrator("sglang", args, backend_args)
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.wait.return_value = 0
        orch.workers = [(mock_proc, 31000)]

        # killpg raises ProcessLookupError (process already dead)
        with patch("os.killpg", side_effect=ProcessLookupError):
            orch._cleanup_workers()  # should not raise

        mock_proc.wait.assert_called_once()

    def test_cleanup_workers_sigkill_on_timeout(self):
        args = _make_args()
        backend_args = ["--model-path", "/tmp/model"]
        orch = ServeOrchestrator("sglang", args, backend_args)
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=30)
        orch.workers = [(mock_proc, 31000)]

        with patch("os.killpg") as mock_killpg:
            orch._cleanup_workers()

        # First call is SIGTERM, second is SIGKILL after timeout
        assert mock_killpg.call_count == 2
        mock_killpg.assert_any_call(12345, signal.SIGTERM)
        mock_killpg.assert_any_call(12345, signal.SIGKILL)

    def test_cleanup_workers_empty_list(self):
        args = _make_args()
        backend_args = ["--model-path", "/tmp/model"]
        orch = ServeOrchestrator("sglang", args, backend_args)
        orch.workers = []
        # Should be a no-op
        orch._cleanup_workers()

    def test_signal_handler_sets_guard_flag(self):
        args = _make_args()
        backend_args = ["--model-path", "/tmp/model"]
        orch = ServeOrchestrator("sglang", args, backend_args)
        orch.workers = []

        assert orch._shutting_down is False
        with pytest.raises(SystemExit) as exc_info:
            orch._signal_handler(signal.SIGINT, None)
        assert exc_info.value.code == 128 + signal.SIGINT
        assert orch._shutting_down is True

    def test_signal_handler_guard_prevents_reentry(self):
        args = _make_args()
        backend_args = ["--model-path", "/tmp/model"]
        orch = ServeOrchestrator("sglang", args, backend_args)
        orch._shutting_down = True

        # Should return immediately without calling sys.exit
        orch._signal_handler(signal.SIGINT, None)

    def test_trtllm_orchestrator_launches_grpc_workers(self):
        args = _make_args(
            backend="trtllm", data_parallel_size=1, model="/tmp/m", connection_mode="grpc"
        )
        backend_args = ["--config", "/tmp/config.yml"]
        orch = ServeOrchestrator("trtllm", args, backend_args)

        with patch("smg.serve._find_available_ports", return_value=[50051]):
            with patch.object(orch.launcher, "launch") as mock_launch:
                mock_launch.return_value = MagicMock(pid=1234)
                orch._launch_workers()

        assert len(orch.workers) == 1
        assert orch.workers[0][1] == 50051

    def test_launch_workers_passes_gpu_env(self):
        """_launch_workers passes CUDA_VISIBLE_DEVICES via gpu_env for each dp_rank."""
        args = _make_args(data_parallel_size=2, tensor_parallel_size=2, connection_mode="grpc")
        backend_args = ["--model-path", "/tmp/model"]
        orch = ServeOrchestrator("sglang", args, backend_args)

        launched_envs = []

        def capture_launch(a, b, host, port, env):
            launched_envs.append(env)
            mock_proc = MagicMock()
            mock_proc.pid = 1000 + port
            return mock_proc

        with patch("smg.serve._find_available_ports", return_value=[31000, 31003]):
            with patch.object(orch.launcher, "launch", side_effect=capture_launch):
                orch._launch_workers()

        assert len(launched_envs) == 2
        assert launched_envs[0]["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert launched_envs[1]["CUDA_VISIBLE_DEVICES"] == "2,3"
        assert launched_envs[0]["PYTHONUNBUFFERED"] == "1"
        assert launched_envs[1]["PYTHONUNBUFFERED"] == "1"
