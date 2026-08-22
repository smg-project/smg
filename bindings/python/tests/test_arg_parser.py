"""
Unit tests for argument parsing functionality in smg.

These tests focus on testing the argument parsing logic in isolation,
without starting actual router instances.
"""

import argparse
from types import SimpleNamespace

import pytest
from smg.launch_router import RouterArgs, parse_router_args
from smg.router import Router, policy_from_str


class TestRouterArgs:
    """Test RouterArgs dataclass and its methods."""

    def test_default_values(self):
        """Test that RouterArgs has correct default values."""
        args = RouterArgs()

        # Test basic defaults
        assert args.host == "0.0.0.0"
        assert args.port == 30000
        assert args.policy == "cache_aware"
        assert args.worker_urls == []
        assert args.pd_disaggregation is False
        assert args.prefill_urls == []
        assert args.decode_urls == []

        # Test PD-specific defaults
        assert args.prefill_policy is None
        assert args.decode_policy is None
        assert args.chars_per_token == 4
        assert args.long_prefill_threshold == 100_000
        assert args.long_pool_max_load == 4
        assert args.short_pool_max_load == 32
        assert args.long_prefill_indices == []

        # Test service discovery defaults
        assert args.service_discovery is False
        assert args.selector == {}
        assert args.service_discovery_port == 80
        assert args.service_discovery_namespace is None

        # Test retry and circuit breaker defaults
        assert args.retry_max_retries == 5
        assert args.cb_failure_threshold == 10
        assert args.disable_retries is False
        assert args.disable_circuit_breaker is False
        assert args.mesh_advertise_host is None

    def test_parse_selector_valid(self):
        """Test parsing valid selector arguments."""
        # Test single key-value pair
        result = RouterArgs._parse_selector(["app=worker"])
        assert result == {"app": "worker"}

        # Test multiple key-value pairs
        result = RouterArgs._parse_selector(["app=worker", "env=prod", "version=v1"])
        assert result == {"app": "worker", "env": "prod", "version": "v1"}

        # Test empty list
        result = RouterArgs._parse_selector([])
        assert result == {}

        # Test None
        result = RouterArgs._parse_selector(None)
        assert result == {}

    def test_parse_selector_invalid(self):
        """Test parsing invalid selector arguments."""
        # Test malformed selector (no equals sign)
        result = RouterArgs._parse_selector(["app"])
        assert result == {}

        # Test multiple equals signs (should use first one)
        result = RouterArgs._parse_selector(["app=worker=extra"])
        assert result == {"app": "worker=extra"}

    def test_parse_selector_spaced_items(self):
        """Every item may hold several space-separated pairs, in any mix."""
        result = RouterArgs._parse_selector(["app=worker env=prod"])
        assert result == {"app": "worker", "env": "prod"}

        result = RouterArgs._parse_selector(["app=worker env=prod", "tier=engine"])
        assert result == {"app": "worker", "env": "prod", "tier": "engine"}

    def test_parse_model_aliases(self):
        result = RouterArgs._parse_model_aliases(["GLM-5.2-Coding=GLM-5.2", "glm-5.2=GLM-5.2"])

        assert result == {
            "GLM-5.2-Coding": "GLM-5.2",
            "glm-5.2": "GLM-5.2",
        }

    @pytest.mark.parametrize(
        "entries",
        [
            ["missing-separator"],
            ["=GLM-5.2"],
            ["GLM-5.2-Coding="],
            ["GLM-5.2=GLM-5.2"],
            ["shared=model-a", "shared=model-b"],
        ],
    )
    def test_parse_model_aliases_rejects_invalid_entries(self, entries):
        with pytest.raises(ValueError):
            RouterArgs._parse_model_aliases(entries)

    def test_parse_prefill_urls_valid(self):
        """Test parsing valid prefill URL arguments."""
        # Test with bootstrap port
        result = RouterArgs._parse_prefill_urls([["http://prefill1:8000", "9000"]])
        assert result == [("http://prefill1:8000", 9000)]

        # Test with 'none' bootstrap port
        result = RouterArgs._parse_prefill_urls([["http://prefill1:8000", "none"]])
        assert result == [("http://prefill1:8000", None)]

        # Test without bootstrap port
        result = RouterArgs._parse_prefill_urls([["http://prefill1:8000"]])
        assert result == [("http://prefill1:8000", None)]

        # Test multiple prefill URLs
        result = RouterArgs._parse_prefill_urls(
            [
                ["http://prefill1:8000", "9000"],
                ["http://prefill2:8000", "none"],
                ["http://prefill3:8000"],
            ]
        )
        expected = [
            ("http://prefill1:8000", 9000),
            ("http://prefill2:8000", None),
            ("http://prefill3:8000", None),
        ]
        assert result == expected

        # Test empty list
        result = RouterArgs._parse_prefill_urls([])
        assert result == []

        # Test None
        result = RouterArgs._parse_prefill_urls(None)
        assert result == []

    def test_parse_prefill_urls_invalid(self):
        """Test parsing invalid prefill URL arguments."""
        # Test invalid bootstrap port
        with pytest.raises(ValueError, match="Invalid bootstrap port"):
            RouterArgs._parse_prefill_urls([["http://prefill1:8000", "invalid"]])

    def test_parse_encode_urls_valid(self):
        """Test parsing valid encode URL arguments."""
        result = RouterArgs._parse_encode_urls([["http://encode1:8000", "9000"]])
        assert result == [("http://encode1:8000", 9000)]

        result = RouterArgs._parse_encode_urls([["http://encode1:8000", "none"]])
        assert result == [("http://encode1:8000", None)]

        result = RouterArgs._parse_encode_urls([["http://encode1:8000"]])
        assert result == [("http://encode1:8000", None)]

    def test_parse_decode_urls_valid(self):
        """Test parsing valid decode URL arguments."""
        # Test single decode URL
        result = RouterArgs._parse_decode_urls([["http://decode1:8001"]])
        assert result == ["http://decode1:8001"]

        # Test multiple decode URLs
        result = RouterArgs._parse_decode_urls([["http://decode1:8001"], ["http://decode2:8001"]])
        assert result == ["http://decode1:8001", "http://decode2:8001"]

        # Test empty list
        result = RouterArgs._parse_decode_urls([])
        assert result == []

        # Test None
        result = RouterArgs._parse_decode_urls(None)
        assert result == []

    def test_parse_cache_aware_length_prefill_pool_command(self):
        """The Python entrypoint accepts a static long/short prefill pool."""
        router_args = parse_router_args(
            [
                "--pd-disaggregation",
                "--prefill-policy",
                "cache_aware_length",
                "--long-prefill-threshold",
                "100000",
                "--balance-abs-threshold",
                "16",
                "--balance-rel-threshold",
                "2.0",
                "--eviction-interval",
                "120",
                "--long-pool-max-load",
                "4",
                "--short-pool-max-load",
                "32",
                "--cache-threshold",
                "0.25",
                "--prefill",
                "http://p1:8000",
                "--prefill",
                "http://p2:8000",
                "--prefill",
                "http://p3:8000",
                "--prefill",
                "http://p4:8000",
                "--prefill",
                "http://p5:8000",
                "--decode",
                "http://d1:8000",
                "--long-prefill-indices",
                "3,4",
            ]
        )
        router_args._validate_router_args()

        assert router_args.pd_disaggregation is True
        assert router_args.prefill_policy == "cache_aware_length"
        assert router_args.long_prefill_threshold == 100_000
        assert router_args.balance_abs_threshold == 16
        assert router_args.balance_rel_threshold == 2.0
        assert router_args.eviction_interval_secs == 120
        assert router_args.long_pool_max_load == 4
        assert router_args.short_pool_max_load == 32
        assert router_args.cache_threshold == 0.25
        assert router_args.long_prefill_indices == [3, 4]

    def test_cache_aware_length_pool_arguments_reach_rust_router(self, monkeypatch):
        router_args = parse_router_args(
            [
                "--pd-disaggregation",
                "--prefill-policy",
                "cache_aware_length",
                "--prefill",
                "http://p1:8000",
                "--prefill",
                "http://p2:8000",
                "--long-prefill-indices",
                "1",
                "--long-prefill-threshold",
                "100000",
                "--long-pool-max-load",
                "4",
                "--short-pool-max-load",
                "32",
                "--decode",
                "http://d1:8000",
            ]
        )
        captured = {}

        def fake_rust_router(**kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr("smg.router._Router", fake_rust_router)
        Router.from_args(router_args)

        assert captured["prefill_policy"] == policy_from_str("cache_aware_length")
        assert captured["long_prefill_indices"] == [1]
        assert captured["long_prefill_threshold"] == 100_000
        assert captured["long_pool_max_load"] == 4
        assert captured["short_pool_max_load"] == 32

    @pytest.mark.parametrize(
        ("indices", "message"),
        [
            ([-1], "non-negative"),
            ([3, 3], "duplicate"),
            ([5], "out of range"),
        ],
    )
    def test_long_prefill_indices_reject_invalid_values(self, indices, message):
        router_args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[(f"http://p{i}:8000", None) for i in range(1, 6)],
            decode_urls=["http://d1:8000"],
            prefill_policy="cache_aware_length",
            long_prefill_indices=indices,
        )

        with pytest.raises(ValueError, match=message):
            router_args._validate_router_args()

    def test_prefill_decode_urls_require_disaggregation_mode(self):
        router_args = parse_router_args(
            [
                "--prefill",
                "http://p1:8000",
                "--decode",
                "http://d1:8000",
            ]
        )

        with pytest.raises(ValueError, match="--pd-disaggregation"):
            router_args._validate_router_args()

    def test_prefixed_eviction_interval_alias_maps_to_router_args(self):
        parser = argparse.ArgumentParser()
        RouterArgs.add_cli_args(parser, use_router_prefix=True)

        namespace = parser.parse_args(["--router-eviction-interval", "120"])
        router_args = RouterArgs.from_cli_args(namespace, use_router_prefix=True)

        assert router_args.eviction_interval_secs == 120

    def test_from_cli_args_basic(self):
        """Test creating RouterArgs from basic CLI arguments."""
        args = SimpleNamespace(
            host="0.0.0.0",
            port=30001,
            worker_urls=["http://worker1:8000", "http://worker2:8000"],
            policy="round_robin",
            prefill=None,
            decode=None,
            router_policy="round_robin",
            router_pd_disaggregation=False,
            router_prefill_policy=None,
            router_decode_policy=None,
            router_worker_startup_timeout_secs=300,
            router_worker_startup_check_interval=15,
            router_cache_threshold=0.7,
            router_balance_abs_threshold=128,
            router_balance_rel_threshold=2.0,
            router_eviction_interval=180,
            router_max_tree_size=2**28,
            router_max_payload_size=1024 * 1024 * 1024,  # 1GB
            router_dp_aware=True,
            router_api_key="test-key",
            router_log_dir="/tmp/logs",
            router_log_level="debug",
            router_service_discovery=True,
            router_selector=["app=worker", "env=test"],
            router_service_discovery_port=8080,
            router_service_discovery_namespace="default",
            router_prefill_selector=["app=prefill"],
            router_decode_selector=["app=decode"],
            router_prometheus_port=29000,
            router_prometheus_host="0.0.0.0",
            router_request_id_headers=["x-request-id", "x-trace-id"],
            router_storage_context_headers=["x-tenant-id=tenant_id", "x-user-id=user_id"],
            router_request_timeout_secs=1200,
            router_max_concurrent_requests=512,
            router_queue_size=200,
            router_queue_timeout_secs=120,
            router_rate_limit_tokens_per_second=100,
            router_cors_allowed_origins=["http://localhost:3000"],
            router_retry_max_retries=3,
            router_retry_initial_backoff_ms=100,
            router_retry_max_backoff_ms=10000,
            router_retry_backoff_multiplier=2.0,
            router_retry_jitter_factor=0.1,
            router_cb_failure_threshold=5,
            router_cb_success_threshold=2,
            router_cb_timeout_duration_secs=30,
            router_cb_window_duration_secs=60,
            router_disable_retries=False,
            router_disable_circuit_breaker=False,
            router_health_failure_threshold=2,
            router_health_success_threshold=1,
            router_health_check_timeout_secs=3,
            router_health_check_interval_secs=30,
            router_health_check_endpoint="/healthz",
        )

        router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)

        # Test basic configuration
        assert router_args.host == "0.0.0.0"
        assert router_args.port == 30001
        assert router_args.worker_urls == ["http://worker1:8000", "http://worker2:8000"]
        assert router_args.policy == "round_robin"

        # Test PD configuration
        assert router_args.pd_disaggregation is False
        assert router_args.prefill_urls == []
        assert router_args.decode_urls == []

        # Test service discovery
        assert router_args.service_discovery is True
        assert router_args.selector == {"app": "worker", "env": "test"}
        assert router_args.service_discovery_port == 8080
        assert router_args.service_discovery_namespace == "default"
        assert router_args.prefill_selector == {"app": "prefill"}
        assert router_args.decode_selector == {"app": "decode"}

        # Test other configurations
        assert router_args.dp_aware is True
        assert router_args.api_key == "test-key"
        assert router_args.log_dir == "/tmp/logs"
        assert router_args.log_level == "debug"
        assert router_args.prometheus_port == 29000
        assert router_args.prometheus_host == "0.0.0.0"
        assert router_args.request_id_headers == ["x-request-id", "x-trace-id"]
        assert router_args.storage_context_headers == {
            "x-tenant-id": "tenant_id",
            "x-user-id": "user_id",
        }
        assert router_args.request_timeout_secs == 1200
        assert router_args.max_concurrent_requests == 512
        assert router_args.queue_size == 200
        assert router_args.queue_timeout_secs == 120
        assert router_args.rate_limit_tokens_per_second == 100
        assert router_args.cors_allowed_origins == ["http://localhost:3000"]

        # Test retry configuration
        assert router_args.retry_max_retries == 3
        assert router_args.retry_initial_backoff_ms == 100
        assert router_args.retry_max_backoff_ms == 10000
        assert router_args.retry_backoff_multiplier == 2.0
        assert router_args.retry_jitter_factor == 0.1

        # Test circuit breaker configuration
        assert router_args.cb_failure_threshold == 5
        assert router_args.cb_success_threshold == 2
        assert router_args.cb_timeout_duration_secs == 30
        assert router_args.cb_window_duration_secs == 60
        assert router_args.disable_retries is False
        assert router_args.disable_circuit_breaker is False

        # Test health check configuration
        assert router_args.health_failure_threshold == 2
        assert router_args.health_success_threshold == 1
        assert router_args.health_check_timeout_secs == 3
        assert router_args.health_check_interval_secs == 30
        assert router_args.health_check_endpoint == "/healthz"

        # Note: model_path and tokenizer_path are not available in current RouterArgs

    def test_from_cli_args_pd_mode(self):
        """Test creating RouterArgs from CLI arguments in PD mode."""
        args = SimpleNamespace(
            host="127.0.0.1",
            port=30000,
            worker_urls=[],
            policy="cache_aware",
            prefill=[
                ["http://prefill1:8000", "9000"],
                ["http://prefill2:8000", "none"],
            ],
            decode=[["http://decode1:8001"], ["http://decode2:8001"]],
            router_prefill=[
                ["http://prefill1:8000", "9000"],
                ["http://prefill2:8000", "none"],
            ],
            router_decode=[["http://decode1:8001"], ["http://decode2:8001"]],
            router_policy="cache_aware",
            router_pd_disaggregation=True,
            router_prefill_policy="power_of_two",
            router_decode_policy="round_robin",
            # Include all required fields with defaults
            router_worker_startup_timeout_secs=600,
            router_worker_startup_check_interval=30,
            router_cache_threshold=0.3,
            router_balance_abs_threshold=64,
            router_balance_rel_threshold=1.5,
            router_eviction_interval=120,
            router_max_tree_size=2**26,
            router_max_payload_size=512 * 1024 * 1024,
            router_dp_aware=False,
            router_api_key=None,
            router_log_dir=None,
            router_log_level=None,
            router_service_discovery=False,
            router_selector=None,
            router_service_discovery_port=80,
            router_service_discovery_namespace=None,
            router_prefill_selector=None,
            router_decode_selector=None,
            router_prometheus_port=None,
            router_prometheus_host=None,
            router_request_id_headers=None,
            router_storage_context_headers=None,
            router_request_timeout_secs=1800,
            router_max_concurrent_requests=256,
            router_queue_size=100,
            router_queue_timeout_secs=60,
            router_rate_limit_tokens_per_second=None,
            router_cors_allowed_origins=[],
            router_retry_max_retries=5,
            router_retry_initial_backoff_ms=50,
            router_retry_max_backoff_ms=30000,
            router_retry_backoff_multiplier=1.5,
            router_retry_jitter_factor=0.2,
            router_cb_failure_threshold=10,
            router_cb_success_threshold=3,
            router_cb_timeout_duration_secs=60,
            router_cb_window_duration_secs=120,
            router_disable_retries=False,
            router_disable_circuit_breaker=False,
            router_health_failure_threshold=3,
            router_health_success_threshold=2,
            router_health_check_timeout_secs=5,
            router_health_check_interval_secs=60,
            router_health_check_endpoint="/health",
        )

        router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)

        # Test PD configuration
        assert router_args.pd_disaggregation is True
        assert router_args.prefill_urls == [
            ("http://prefill1:8000", 9000),
            ("http://prefill2:8000", None),
        ]
        assert router_args.decode_urls == ["http://decode1:8001", "http://decode2:8001"]
        assert router_args.prefill_policy == "power_of_two"
        assert router_args.decode_policy == "round_robin"
        assert router_args.policy == "cache_aware"  # Main policy still set

    def test_from_cli_args_without_prefix(self):
        """Test creating RouterArgs from CLI arguments without router prefix."""
        args = SimpleNamespace(
            host="127.0.0.1",
            port=30000,
            worker_urls=["http://worker1:8000"],
            policy="random",
            prefill=None,
            decode=None,
            pd_disaggregation=False,
            prefill_policy=None,
            decode_policy=None,
            worker_startup_timeout_secs=600,
            worker_startup_check_interval=30,
            cache_threshold=0.3,
            balance_abs_threshold=64,
            balance_rel_threshold=1.5,
            eviction_interval=120,
            max_tree_size=2**26,
            max_payload_size=512 * 1024 * 1024,
            dp_aware=False,
            api_key=None,
            log_dir=None,
            log_level=None,
            service_discovery=False,
            selector=None,
            service_discovery_port=80,
            service_discovery_namespace=None,
            prefill_selector=None,
            decode_selector=None,
            prometheus_port=None,
            prometheus_host=None,
            request_id_headers=None,
            storage_context_headers=None,
            request_timeout_secs=1800,
            max_concurrent_requests=256,
            queue_size=100,
            queue_timeout_secs=60,
            rate_limit_tokens_per_second=None,
            cors_allowed_origins=[],
            retry_max_retries=5,
            retry_initial_backoff_ms=50,
            retry_max_backoff_ms=30000,
            retry_backoff_multiplier=1.5,
            retry_jitter_factor=0.2,
            cb_failure_threshold=10,
            cb_success_threshold=3,
            cb_timeout_duration_secs=60,
            cb_window_duration_secs=120,
            disable_retries=False,
            disable_circuit_breaker=False,
            health_failure_threshold=3,
            health_success_threshold=2,
            health_check_timeout_secs=5,
            health_check_interval_secs=60,
            health_check_endpoint="/health",
            model_path=None,
            tokenizer_path=None,
        )

        router_args = RouterArgs.from_cli_args(args, use_router_prefix=False)

        assert router_args.host == "127.0.0.1"
        assert router_args.port == 30000
        assert router_args.worker_urls == ["http://worker1:8000"]
        assert router_args.policy == "random"
        assert router_args.pd_disaggregation is False

    def test_prefixed_args_fall_back_to_backend_args_by_default(self):
        """Prefixed router args should still fall back to backend args unless disabled."""
        args = SimpleNamespace(
            router_model_path=None,
            router_disable_arg_fallback=False,
            model_path="backend/model",
            router_tokenizer_path=None,
            tokenizer_path="backend/tokenizer",
            router_worker_urls=[],
            worker_urls=[],
            router_prefill=None,
            router_decode=None,
            router_selector=None,
            router_prefill_selector=None,
            router_decode_selector=None,
            router_router_selector=None,
        )

        router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)

        assert router_args.model_path == "backend/model"
        assert router_args.tokenizer_path == "backend/tokenizer"

    def test_prefixed_args_can_disable_backend_fallback(self):
        """When router fallback is disabled, backend args should not fill router args."""
        args = SimpleNamespace(
            router_model_path=None,
            router_disable_arg_fallback=True,
            model_path="backend/model",
            router_tokenizer_path=None,
            tokenizer_path="backend/tokenizer",
            router_worker_urls=[],
            worker_urls=[],
            router_prefill=None,
            router_decode=None,
            router_selector=None,
            router_prefill_selector=None,
            router_decode_selector=None,
            router_router_selector=None,
        )

        router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)

        assert router_args.model_path is None
        assert router_args.tokenizer_path is None


class TestPolicyFromStr:
    """Test policy string to enum conversion."""

    def test_valid_policies(self):
        """Test conversion of valid policy strings."""
        from smg.smg_rs import PolicyType

        assert policy_from_str("random") == PolicyType.Random
        assert policy_from_str("round_robin") == PolicyType.RoundRobin
        assert policy_from_str("cache_aware") == PolicyType.CacheAware
        assert policy_from_str("power_of_two") == PolicyType.PowerOfTwo
        assert policy_from_str("consistent_hashing") == PolicyType.ConsistentHashing
        assert policy_from_str("prefix_hash") == PolicyType.PrefixHash

    def test_invalid_policy(self):
        """Test conversion of invalid policy string."""
        with pytest.raises(KeyError):
            policy_from_str("invalid_policy")


class TestParseRouterArgs:
    """Test the parse_router_args function."""

    def test_parse_basic_args(self):
        """Test parsing basic router arguments."""
        args = [
            "--host",
            "0.0.0.0",
            "--port",
            "30001",
            "--worker-urls",
            "http://worker1:8000",
            "http://worker2:8000",
            "--policy",
            "round_robin",
        ]

        router_args = parse_router_args(args)

        assert router_args.host == "0.0.0.0"
        assert router_args.port == 30001
        assert router_args.worker_urls == ["http://worker1:8000", "http://worker2:8000"]
        assert router_args.policy == "round_robin"

    def test_parse_routing_key_headers(self):
        """Ordered list flag; unset keeps the x-smg-routing-key default."""
        router_args = parse_router_args(
            ["--routing-key-headers", "x-routing-key", "x-smg-routing-key"]
        )
        assert router_args.routing_key_headers == ["x-routing-key", "x-smg-routing-key"]

        defaults = parse_router_args([])
        assert defaults.routing_key_headers == ["x-smg-routing-key"]

    def test_parse_cache_index_args(self):
        """Comma-separated boundaries plus the index/TTL knobs."""
        router_args = parse_router_args(
            [
                "--cache-boundaries",
                "2048,8192,32768",
                "--cache-index",
                "hash",
                "--cache-ttl-secs",
                "120",
            ]
        )
        assert router_args.cache_boundaries == [2048, 8192, 32768]
        assert router_args.cache_index == "hash"
        assert router_args.cache_ttl_secs == 120

        defaults = parse_router_args([])
        assert defaults.cache_boundaries == []
        assert defaults.cache_index == "tree"
        assert defaults.cache_ttl_secs == 180

    def test_parse_worker_overload_args(self):
        """Both overload flags round-trip, and both default to unset.

        The argparse names are built from an f-string prefix, so a typo or a
        dest/field mismatch would leave the field at its default and silently
        disable the feature from Python -- `from_cli_args` skips keys it cannot
        find.
        """
        router_args = parse_router_args(
            [
                "--worker-overload-waiting-requests",
                "64",
                "--worker-overload-token-usage",
                "0.9",
            ]
        )
        assert router_args.worker_overload_waiting_requests == 64
        assert router_args.worker_overload_token_usage == pytest.approx(0.9)

        defaults = parse_router_args([])
        assert defaults.worker_overload_waiting_requests is None
        assert defaults.worker_overload_token_usage is None

    def test_prefixed_worker_overload_args(self):
        """The --router-prefixed aliases reach the same fields."""
        parser = argparse.ArgumentParser()
        RouterArgs.add_cli_args(parser, use_router_prefix=True)
        namespace = parser.parse_args(
            [
                "--router-worker-overload-waiting-requests",
                "8",
                "--router-worker-overload-token-usage",
                "0.75",
            ]
        )

        router_args = RouterArgs.from_cli_args(namespace, use_router_prefix=True)

        assert router_args.worker_overload_waiting_requests == 8
        assert router_args.worker_overload_token_usage == pytest.approx(0.75)

    def test_parse_overload_protection_and_monitoring_flags(self):
        """The enable/opt-out flags round-trip, and both default to False.

        Same failure mode as the threshold flags: a dest/field mismatch would
        silently disable the feature from Python.
        """
        router_args = parse_router_args(
            ["--worker-overload-protection", "--disable-load-monitoring"]
        )
        assert router_args.worker_overload_protection is True
        assert router_args.disable_load_monitoring is True

        defaults = parse_router_args([])
        assert defaults.worker_overload_protection is False
        assert defaults.disable_load_monitoring is False

    def test_prefixed_overload_protection_and_monitoring_flags(self):
        """The --router-prefixed aliases reach the same fields."""
        parser = argparse.ArgumentParser()
        RouterArgs.add_cli_args(parser, use_router_prefix=True)
        namespace = parser.parse_args(
            ["--router-worker-overload-protection", "--router-disable-load-monitoring"]
        )

        router_args = RouterArgs.from_cli_args(namespace, use_router_prefix=True)

        assert router_args.worker_overload_protection is True
        assert router_args.disable_load_monitoring is True

    def test_parse_pd_args(self):
        """Test parsing PD disaggregated mode arguments."""
        args = [
            "--pd-disaggregation",
            "--prefill",
            "http://prefill1:8000",
            "9000",
            "--prefill",
            "http://prefill2:8000",
            "none",
            "--decode",
            "http://decode1:8001",
            "--decode",
            "http://decode2:8001",
            "--prefill-policy",
            "power_of_two",
            "--decode-policy",
            "round_robin",
        ]

        router_args = parse_router_args(args)

        assert router_args.pd_disaggregation is True
        assert router_args.prefill_urls == [
            ("http://prefill1:8000", 9000),
            ("http://prefill2:8000", None),
        ]
        assert router_args.decode_urls == ["http://decode1:8001", "http://decode2:8001"]
        assert router_args.prefill_policy == "power_of_two"
        assert router_args.decode_policy == "round_robin"

    def test_parse_epd_args(self):
        """Test parsing EPD disaggregated mode arguments."""
        args = [
            "--epd-disaggregation",
            "--encode",
            "http://encode1:8000",
            "9000",
            "--encode",
            "http://encode2:8000",
            "none",
            "--prefill",
            "http://prefill1:8000",
            "9001",
            "--decode",
            "http://decode1:8001",
            "--encode-policy",
            "consistent_hashing",
            "--prefill-policy",
            "cache_aware",
            "--decode-policy",
            "round_robin",
        ]

        router_args = parse_router_args(args)

        assert router_args.epd_disaggregation is True
        assert router_args.encode_urls == [
            ("http://encode1:8000", 9000),
            ("http://encode2:8000", None),
        ]
        assert router_args.prefill_urls == [("http://prefill1:8000", 9001)]
        assert router_args.decode_urls == ["http://decode1:8001"]
        assert router_args.encode_policy == "consistent_hashing"
        assert router_args.prefill_policy == "cache_aware"
        assert router_args.decode_policy == "round_robin"

    def test_parse_pd_args_with_new_policies(self):
        """Test parsing PD disaggregated mode arguments with new policy options."""
        # Test consistent_hashing for both prefill and decode
        args = [
            "--pd-disaggregation",
            "--prefill",
            "http://prefill1:8000",
            "--decode",
            "http://decode1:8001",
            "--prefill-policy",
            "consistent_hashing",
            "--decode-policy",
            "consistent_hashing",
        ]

        router_args = parse_router_args(args)

        assert router_args.pd_disaggregation is True
        assert router_args.prefill_policy == "consistent_hashing"
        assert router_args.decode_policy == "consistent_hashing"

        # Test prefix_hash for both prefill and decode
        args = [
            "--pd-disaggregation",
            "--prefill",
            "http://prefill1:8000",
            "--decode",
            "http://decode1:8001",
            "--prefill-policy",
            "prefix_hash",
            "--decode-policy",
            "prefix_hash",
        ]

        router_args = parse_router_args(args)

        assert router_args.pd_disaggregation is True
        assert router_args.prefill_policy == "prefix_hash"
        assert router_args.decode_policy == "prefix_hash"

        # Test mixed policies
        args = [
            "--pd-disaggregation",
            "--prefill",
            "http://prefill1:8000",
            "--decode",
            "http://decode1:8001",
            "--prefill-policy",
            "consistent_hashing",
            "--decode-policy",
            "prefix_hash",
        ]

        router_args = parse_router_args(args)

        assert router_args.pd_disaggregation is True
        assert router_args.prefill_policy == "consistent_hashing"
        assert router_args.decode_policy == "prefix_hash"

    def test_parse_service_discovery_args(self):
        """Test parsing service discovery arguments."""
        args_a = [
            "--service-discovery",
            "--selector",
            "app=worker",
            "env=prod",
            "--service-discovery-port",
            "8080",
            "--service-discovery-namespace",
            "default",
        ]
        args_b = [
            "--service-discovery",
            "--selector",
            # OME has this style
            "app=worker env=prod",
            "--service-discovery-port",
            "8080",
            "--service-discovery-namespace",
            "default",
        ]

        for args in [args_a, args_b]:
            router_args = parse_router_args(args)

            assert router_args.service_discovery is True
            assert router_args.selector == {"app": "worker", "env": "prod"}
            assert router_args.service_discovery_port == 8080
            assert router_args.service_discovery_namespace == "default"

    def test_repeated_list_flags_accumulate(self):
        """Repeated occurrences of list flags append, matching the Rust CLI."""
        router_args = parse_router_args(
            [
                "--selector",
                "component=engine",
                "--selector",
                "ome.io/inferenceservice=svc",
                "--encode-selector",
                "app=e",
                "--encode-selector",
                "role=encode",
                "--prefill-selector",
                "app=p",
                "--prefill-selector",
                "role=prefill",
                "--decode-selector",
                "app=d",
                "--decode-selector",
                "role=decode",
                "--router-selector",
                "app=r",
                "--router-selector",
                "role=router",
                "--storage-context-headers",
                "x-tenant-id=tenant_id",
                "--storage-context-headers",
                "x-user-id=user_id",
                "--worker-urls",
                "http://w1:8000",
                "--worker-urls",
                "http://w2:8000",
                "--request-id-headers",
                "x-request-id",
                "--request-id-headers",
                "x-trace-id",
                "--cors-allowed-origins",
                "http://a",
                "--cors-allowed-origins",
                "http://b",
                "--ca-cert-paths",
                "/ca/one.pem",
                "--ca-cert-paths",
                "/ca/two.pem",
                "--mesh-peer-urls",
                "peer1:39527",
                "--mesh-peer-urls",
                "peer2:39527",
                "--prometheus-duration-buckets",
                "0.1",
                "--prometheus-duration-buckets",
                "0.5",
                "--control-plane-api-keys",
                "k1:Svc:admin:s1",
                "--control-plane-api-keys",
                "k2:Ro:user:s2",
                "--jwt-role-mapping",
                "Gateway.Admin=admin",
                "--jwt-role-mapping",
                "Gateway.User=user",
            ]
        )

        assert router_args.selector == {
            "component": "engine",
            "ome.io/inferenceservice": "svc",
        }
        assert router_args.encode_selector == {"app": "e", "role": "encode"}
        assert router_args.prefill_selector == {"app": "p", "role": "prefill"}
        assert router_args.decode_selector == {"app": "d", "role": "decode"}
        assert router_args.router_selector == {"app": "r", "role": "router"}
        assert router_args.storage_context_headers == {
            "x-tenant-id": "tenant_id",
            "x-user-id": "user_id",
        }
        assert router_args.worker_urls == ["http://w1:8000", "http://w2:8000"]
        assert router_args.request_id_headers == ["x-request-id", "x-trace-id"]
        assert router_args.cors_allowed_origins == ["http://a", "http://b"]
        assert router_args.ca_cert_paths == ["/ca/one.pem", "/ca/two.pem"]
        assert router_args.mesh_peer_urls == ["peer1:39527", "peer2:39527"]
        assert router_args.prometheus_duration_buckets == [0.1, 0.5]
        assert router_args.control_plane_api_keys == [
            ("k1", "Svc", "s1", "admin"),
            ("k2", "Ro", "s2", "user"),
        ]
        assert router_args.jwt_role_mapping == {
            "Gateway.Admin": "admin",
            "Gateway.User": "user",
        }

    def test_selector_spaced_values_mix_with_repeats(self):
        """Spaced pairs in one occurrence compose with additional occurrences."""
        router_args = parse_router_args(
            [
                "--selector",
                "app=worker env=prod",
                "--selector",
                "tier=engine",
            ]
        )

        assert router_args.selector == {"app": "worker", "env": "prod", "tier": "engine"}

    def test_list_flags_default_empty(self):
        """Absent list flags parse to the documented empty defaults."""
        router_args = parse_router_args([])

        assert router_args.selector == {}
        assert router_args.encode_selector == {}
        assert router_args.prefill_selector == {}
        assert router_args.decode_selector == {}
        assert router_args.router_selector == {}
        assert router_args.storage_context_headers == {}
        assert router_args.worker_urls == []
        assert router_args.cors_allowed_origins == []
        assert router_args.ca_cert_paths == []
        assert router_args.mesh_peer_urls == []
        assert router_args.control_plane_api_keys == []
        assert router_args.jwt_role_mapping == {}

    def test_prefixed_repeated_selector_accumulates(self):
        """Repeated --router-selector occurrences accumulate in prefixed mode."""
        parser = argparse.ArgumentParser()
        RouterArgs.add_cli_args(parser, use_router_prefix=True)
        namespace = parser.parse_args(
            [
                "--router-selector",
                "component=engine",
                "--router-selector",
                "env=prod",
            ]
        )

        router_args = RouterArgs.from_cli_args(namespace, use_router_prefix=True)

        assert router_args.selector == {"component": "engine", "env": "prod"}

    def test_parse_repeated_model_alias_args(self):
        router_args = parse_router_args(
            [
                "--model-alias",
                "GLM-5.2-Coding=GLM-5.2",
                "--model-alias",
                "glm-5.2=GLM-5.2",
            ]
        )

        assert router_args.model_aliases == {
            "GLM-5.2-Coding": "GLM-5.2",
            "glm-5.2": "GLM-5.2",
        }

    def test_parse_prefixed_model_alias_args(self):
        parser = argparse.ArgumentParser()
        RouterArgs.add_cli_args(parser, use_router_prefix=True)
        namespace = parser.parse_args(["--router-model-alias", "GLM-5.2-Coding=GLM-5.2"])

        router_args = RouterArgs.from_cli_args(namespace, use_router_prefix=True)

        assert router_args.model_aliases == {"GLM-5.2-Coding": "GLM-5.2"}

    def test_parse_retry_and_circuit_breaker_args(self):
        """Test parsing retry and circuit breaker arguments."""
        args = [
            "--retry-max-retries",
            "3",
            "--retry-initial-backoff-ms",
            "100",
            "--retry-max-backoff-ms",
            "10000",
            "--retry-backoff-multiplier",
            "2.0",
            "--retry-jitter-factor",
            "0.1",
            "--disable-retries",
            "--cb-failure-threshold",
            "5",
            "--cb-success-threshold",
            "2",
            "--cb-timeout-duration-secs",
            "30",
            "--cb-window-duration-secs",
            "60",
            "--disable-circuit-breaker",
        ]

        router_args = parse_router_args(args)

        # Test retry configuration
        assert router_args.retry_max_retries == 3
        assert router_args.retry_initial_backoff_ms == 100
        assert router_args.retry_max_backoff_ms == 10000
        assert router_args.retry_backoff_multiplier == 2.0
        assert router_args.retry_jitter_factor == 0.1
        assert router_args.disable_retries is True

        # Test circuit breaker configuration
        assert router_args.cb_failure_threshold == 5
        assert router_args.cb_success_threshold == 2
        assert router_args.cb_timeout_duration_secs == 30
        assert router_args.cb_window_duration_secs == 60
        assert router_args.disable_circuit_breaker is True

    def test_parse_rate_limiting_args(self):
        """Test parsing rate limiting arguments."""
        args = [
            "--max-concurrent-requests",
            "512",
            "--queue-size",
            "200",
            "--queue-timeout-secs",
            "120",
            "--rate-limit-tokens-per-second",
            "100",
        ]

        router_args = parse_router_args(args)

        assert router_args.max_concurrent_requests == 512
        assert router_args.queue_size == 200
        assert router_args.queue_timeout_secs == 120
        assert router_args.rate_limit_tokens_per_second == 100

    def test_parse_health_check_args(self):
        """Test parsing health check arguments."""
        args = [
            "--health-failure-threshold",
            "2",
            "--health-success-threshold",
            "1",
            "--health-check-timeout-secs",
            "3",
            "--health-check-interval-secs",
            "30",
            "--health-check-endpoint",
            "/healthz",
        ]

        router_args = parse_router_args(args)

        assert router_args.health_failure_threshold == 2
        assert router_args.health_success_threshold == 1
        assert router_args.health_check_timeout_secs == 3
        assert router_args.health_check_interval_secs == 30
        assert router_args.health_check_endpoint == "/healthz"

    def test_parse_mesh_advertise_host_args(self):
        """Test parsing mesh advertise host arguments."""
        args = [
            "--enable-mesh",
            "--mesh-host",
            "0.0.0.0",
            "--mesh-advertise-host",
            "10.0.0.42",
            "--mesh-port",
            "39527",
            "--mesh-peer-urls",
            "10.0.0.43:39527",
        ]

        router_args = parse_router_args(args)

        assert router_args.enable_mesh is True
        assert router_args.mesh_host == "0.0.0.0"
        assert router_args.mesh_advertise_host == "10.0.0.42"
        assert router_args.mesh_port == 39527
        assert router_args.mesh_peer_urls == ["10.0.0.43:39527"]

    def test_parse_cors_args(self):
        """Test parsing CORS arguments."""
        args = [
            "--cors-allowed-origins",
            "http://localhost:3000",
            "https://example.com",
        ]

        router_args = parse_router_args(args)

        assert router_args.cors_allowed_origins == [
            "http://localhost:3000",
            "https://example.com",
        ]

    def test_parse_tokenizer_args(self):
        """Test parsing tokenizer arguments."""
        # Note: model-path and tokenizer-path arguments are not available in current implementation
        # This test is skipped until those arguments are added
        pytest.skip("Tokenizer arguments not available in current implementation")

    def test_parse_valid_policies(self):
        """Test parsing all valid policy arguments."""
        # Test consistent_hashing policy
        router_args = parse_router_args(["--policy", "consistent_hashing"])
        assert router_args.policy == "consistent_hashing"

        # Test prefix_hash policy
        router_args = parse_router_args(["--policy", "prefix_hash"])
        assert router_args.policy == "prefix_hash"

        # Test all policies in the choices list
        valid_policies = [
            "random",
            "round_robin",
            "cache_aware",
            "power_of_two",
            "manual",
            "consistent_hashing",
            "prefix_hash",
        ]
        for policy in valid_policies:
            router_args = parse_router_args(["--policy", policy])
            assert router_args.policy == policy

    def test_parse_invalid_args(self):
        """Test parsing invalid arguments."""
        # Test invalid policy
        with pytest.raises(SystemExit):
            parse_router_args(["--policy", "invalid_policy"])

        # Test invalid bootstrap port
        with pytest.raises(ValueError, match="Invalid bootstrap port"):
            parse_router_args(
                [
                    "--pd-disaggregation",
                    "--prefill",
                    "http://prefill1:8000",
                    "invalid_port",
                ]
            )

    def test_help_output(self):
        """Test that help output is generated correctly."""
        with pytest.raises(SystemExit) as exc_info:
            parse_router_args(["--help"])

        # SystemExit with code 0 indicates help was displayed
        assert exc_info.value.code == 0


class TestPrefixHashArgs:
    """The prefix_hash knobs parse, default, and honor the router prefix."""

    def test_defaults(self):
        parser = argparse.ArgumentParser()
        RouterArgs.add_cli_args(parser)
        namespace = parser.parse_args([])
        router_args = RouterArgs.from_cli_args(namespace)
        assert router_args.prefix_token_count == 256
        assert router_args.prefix_hash_load_factor == 1.25

    def test_flags(self):
        parser = argparse.ArgumentParser()
        RouterArgs.add_cli_args(parser)
        namespace = parser.parse_args(
            ["--prefix-token-count", "4096", "--prefix-hash-load-factor", "1.5"]
        )
        router_args = RouterArgs.from_cli_args(namespace)
        assert router_args.prefix_token_count == 4096
        assert router_args.prefix_hash_load_factor == 1.5

    def test_router_prefix_flags(self):
        parser = argparse.ArgumentParser()
        RouterArgs.add_cli_args(parser, use_router_prefix=True)
        namespace = parser.parse_args(
            [
                "--router-prefix-token-count",
                "8192",
                "--router-prefix-hash-load-factor",
                "2.0",
            ]
        )
        router_args = RouterArgs.from_cli_args(namespace, use_router_prefix=True)
        assert router_args.prefix_token_count == 8192
        assert router_args.prefix_hash_load_factor == 2.0


class TestCacheBoundariesArgs:
    """Boundary-list parsing beyond the basic parse/default coverage."""

    def test_router_prefix_flag(self):
        parser = argparse.ArgumentParser()
        RouterArgs.add_cli_args(parser, use_router_prefix=True)
        namespace = parser.parse_args(["--router-cache-boundaries", "4096"])
        router_args = RouterArgs.from_cli_args(namespace, use_router_prefix=True)
        assert router_args.cache_boundaries == [4096]

    def test_non_numeric_rejected(self):
        parser = argparse.ArgumentParser()
        RouterArgs.add_cli_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--cache-boundaries", "2048,big"])


class TestFlagAliases:
    """Intent-revealing alias flags land on the same dests as the canonical names."""

    def test_alias_flags_land_on_canonical_dests(self):
        parser = argparse.ArgumentParser()
        RouterArgs.add_cli_args(parser)
        namespace = parser.parse_args(
            [
                "--sticky-sessions",
                "--worker-auto-recovery",
                "--cache-match-threshold",
                "0.6",
                "--spill-abs-threshold",
                "8",
                "--spill-rel-threshold",
                "1.2",
                "--sticky-key-idle-secs",
                "300",
            ]
        )
        router_args = RouterArgs.from_cli_args(namespace)
        assert router_args.routing_key_override is True
        assert router_args.remove_unhealthy_workers is True
        assert router_args.cache_threshold == 0.6
        assert router_args.balance_abs_threshold == 8
        assert router_args.balance_rel_threshold == 1.2
        assert router_args.max_idle_secs == 300

    def test_alias_flags_match_canonical_parse(self):
        parser = argparse.ArgumentParser()
        RouterArgs.add_cli_args(parser)
        canonical = RouterArgs.from_cli_args(
            parser.parse_args(["--routing-key-override", "--cache-threshold", "0.6"])
        )
        aliased = RouterArgs.from_cli_args(
            parser.parse_args(["--sticky-sessions", "--cache-match-threshold", "0.6"])
        )
        assert canonical == aliased

    def test_alias_flags_honor_router_prefix(self):
        parser = argparse.ArgumentParser()
        RouterArgs.add_cli_args(parser, use_router_prefix=True)
        namespace = parser.parse_args(["--router-sticky-sessions", "--router-worker-auto-recovery"])
        router_args = RouterArgs.from_cli_args(namespace, use_router_prefix=True)
        assert router_args.routing_key_override is True
        assert router_args.remove_unhealthy_workers is True


class TestRouterArgsFieldOrder:
    """RouterArgs generates a positional __init__, so field order is a public
    contract: a mid-list insertion silently rebinds every later positional
    argument. New fields must be APPENDED (after the reserve marker at
    worker_startup_delay). The full-sequence snapshot below fails on any
    reorder or insertion — when adding a field, append it both to the
    dataclass tail and to the end of this list."""

    EXPECTED_FIELD_SEQUENCE = [
        "worker_urls",
        "host",
        "port",
        "health_check_port",
        "pd_disaggregation",
        "epd_disaggregation",
        "encode_urls",
        "prefill_urls",
        "decode_urls",
        "policy",
        "encode_policy",
        "prefill_policy",
        "decode_policy",
        "worker_startup_timeout_secs",
        "worker_startup_check_interval",
        "load_monitor_interval",
        "cache_threshold",
        "balance_abs_threshold",
        "balance_rel_threshold",
        "balance_token_usage_threshold",
        "overload_token_usage_threshold",
        "eviction_interval_secs",
        "max_tree_size",
        "block_size",
        "least_load_kv_pressure_weight",
        "least_load_default_throughput",
        "least_load_mean_prefill_tokens",
        "max_idle_secs",
        "assignment_mode",
        "max_payload_size",
        "bucket_adjust_interval_secs",
        "dp_aware",
        "multimodal_tensor_transport",
        "multimodal_shm_min_bytes",
        "routing_key_override",
        "dp_minimum_tokens_scheduler",
        "enable_igw",
        "api_key",
        "log_dir",
        "log_level",
        "log_json",
        "service_discovery",
        "selector",
        "service_discovery_port",
        "service_discovery_namespace",
        "encode_selector",
        "prefill_selector",
        "decode_selector",
        "router_selector",
        "bootstrap_port_annotation",
        "worker_ports_annotation",
        "model_id_from",
        "prometheus_port",
        "prometheus_host",
        "prometheus_duration_buckets",
        "request_id_headers",
        "storage_context_headers",
        "request_timeout_secs",
        "shutdown_grace_period_secs",
        "max_concurrent_requests",
        "queue_size",
        "queue_timeout_secs",
        "rate_limit_tokens_per_second",
        "cors_allowed_origins",
        "retry_max_retries",
        "retry_initial_backoff_ms",
        "retry_max_backoff_ms",
        "retry_backoff_multiplier",
        "retry_jitter_factor",
        "disable_retries",
        "health_failure_threshold",
        "health_success_threshold",
        "health_check_timeout_secs",
        "health_check_interval_secs",
        "health_check_endpoint",
        "disable_health_check",
        "remove_unhealthy_workers",
        "cb_failure_threshold",
        "cb_success_threshold",
        "cb_timeout_duration_secs",
        "cb_window_duration_secs",
        "disable_circuit_breaker",
        "model_path",
        "tokenizer_path",
        "chat_template",
        "disable_tokenizer_autoload",
        "tokenizer_cache_enable_l0",
        "tokenizer_cache_l0_max_entries",
        "tokenizer_cache_enable_l1",
        "tokenizer_cache_l1_max_memory",
        "reasoning_parser",
        "tool_call_parser",
        "mcp_config_path",
        "backend",
        "enable_wasm",
        "storage_hook_wasm_path",
        "history_backend",
        "oracle_wallet_path",
        "oracle_tns_alias",
        "oracle_connect_descriptor",
        "oracle_username",
        "oracle_password",
        "oracle_external_auth",
        "oracle_pool_min",
        "oracle_pool_max",
        "oracle_pool_timeout_secs",
        "postgres_db_url",
        "postgres_pool_max",
        "redis_url",
        "redis_pool_max",
        "redis_retention_days",
        "schema_config",
        "client_cert_path",
        "client_key_path",
        "ca_cert_paths",
        "server_cert_path",
        "server_key_path",
        "enable_trace",
        "otlp_traces_endpoint",
        "control_plane_api_keys",
        "control_plane_audit_enabled",
        "jwt_issuer",
        "jwt_audience",
        "jwt_jwks_uri",
        "jwt_role_mapping",
        "enable_mesh",
        "mesh_server_name",
        "mesh_host",
        "mesh_advertise_host",
        "mesh_port",
        "mesh_peer_urls",
        "model_aliases",
        "worker_startup_delay",
        "zmq_engine_count",
        "prefix_token_count",
        "prefix_hash_load_factor",
        "prefix_hash_balance_abs_threshold",
        "upstream_http2",
        "overlap_decay",
        "selection_temperature",
        "upstream_pool_idle_timeout_secs",
        "least_load_max_waiting_requests",
        "stream_request_bodies_over",
        "stream_body_stall_timeout_secs",
        "routing_key_headers",
        "cache_boundaries",
        "cache_index",
        "cache_ttl_secs",
        "job_queue_capacity",
        "job_queue_concurrency",
        "worker_overload_waiting_requests",
        "worker_overload_token_usage",
        "worker_overload_protection",
        "disable_load_monitoring",
        "chars_per_token",
        "long_prefill_threshold",
        "long_pool_max_load",
        "short_pool_max_load",
        "long_prefill_indices",
    ]

    def test_complete_field_sequence_is_frozen(self):
        import dataclasses

        names = [f.name for f in dataclasses.fields(RouterArgs)]
        assert names == self.EXPECTED_FIELD_SEQUENCE, (
            "RouterArgs field order changed. Positional callers bind by "
            "position: fields may only be APPENDED at the tail (and to "
            "EXPECTED_FIELD_SEQUENCE), never inserted or reordered."
        )

    def test_new_fields_appended_after_positional_reserve(self):
        import dataclasses

        names = [f.name for f in dataclasses.fields(RouterArgs)]
        marker = names.index("worker_startup_delay")
        for appended in (
            "overlap_decay",
            "selection_temperature",
            "upstream_pool_idle_timeout_secs",
            "least_load_max_waiting_requests",
            "stream_request_bodies_over",
            "stream_body_stall_timeout_secs",
            "routing_key_headers",
            "cache_boundaries",
            "cache_index",
            "cache_ttl_secs",
            "worker_overload_waiting_requests",
            "worker_overload_token_usage",
            "worker_overload_protection",
            "disable_load_monitoring",
            "chars_per_token",
            "long_prefill_threshold",
            "long_pool_max_load",
            "short_pool_max_load",
            "long_prefill_indices",
        ):
            assert names.index(appended) > marker, (
                f"{appended} must be appended after worker_startup_delay to "
                "preserve positional callers"
            )

    def test_cache_aware_length_cli_args(self):
        """Test --chars-per-token, --long-prefill-* CLI args are parsed."""
        parser = argparse.ArgumentParser()
        RouterArgs.add_cli_args(parser)
        args = parser.parse_args(
            [
                "--chars-per-token",
                "8",
                "--long-prefill-threshold",
                "200000",
                "--long-pool-max-load",
                "10",
                "--short-pool-max-load",
                "64",
                "--long-prefill-indices",
                "0,2",
            ]
        )
        assert args.chars_per_token == 8
        assert args.long_prefill_threshold == 200000
        assert args.long_pool_max_load == 10
        assert args.short_pool_max_load == 64
        assert args.long_prefill_indices == [0, 2]
