"""Unit tests for KV-transfer param passthrough (engine-free, no vLLM required).

Run with: pytest grpc_servicer/tests/test_kv_transfer_params.py
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto import vllm_engine_pb2  # noqa: E402

# Import the module directly to avoid pulling vllm via the package __init__
_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "kv_transfer.py"
_spec = importlib.util.spec_from_file_location("kv_transfer", _MODULE_PATH)
kv_transfer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kv_transfer)

NIXL_HANDOFF = {
    "do_remote_prefill": True,
    "do_remote_decode": False,
    "remote_engine_id": "engine-1",
    "remote_request_id": "req-1",
    "remote_block_ids": [[1, 2, 3]],
    "remote_host": "10.0.0.1",
    "remote_port": 5600,
    "tp_size": 2,
}


def _kv_config(connector, engine_id="parent-engine", extra=None):
    return SimpleNamespace(
        kv_connector=connector,
        engine_id=engine_id,
        kv_connector_extra_config=extra,
    )


class TestResolvePdConnector:
    def test_ordinary_connector_is_unchanged(self):
        config = _kv_config("ExampleConnector", engine_id="engine-1")

        assert kv_transfer.resolve_pd_connector(config) == (
            "ExampleConnector",
            "engine-1",
        )

    def test_projects_mooncake_child_with_parent_engine_id(self):
        config = _kv_config(
            "MultiConnector",
            extra={
                "connectors": [
                    {"kv_connector": "MooncakeConnector"},
                    {"kv_connector": "MooncakeStoreConnector"},
                ]
            },
        )

        assert kv_transfer.resolve_pd_connector(config) == (
            "MooncakeConnector",
            "parent-engine",
        )

    def test_projects_nixl_child_with_child_engine_id(self):
        config = _kv_config(
            "MultiConnector",
            extra={
                "connectors": [
                    {"kv_connector": "NixlConnector", "engine_id": "child-engine"},
                    {"kv_connector": "MooncakeStoreConnector"},
                ]
            },
        )

        assert kv_transfer.resolve_pd_connector(config) == (
            "NixlConnector",
            "child-engine",
        )

    def test_empty_child_engine_id_is_not_projected(self):
        config = _kv_config(
            "MultiConnector",
            extra={
                "connectors": [
                    {"kv_connector": "NixlConnector", "engine_id": ""},
                    {"kv_connector": "MooncakeStoreConnector"},
                ]
            },
        )

        assert kv_transfer.resolve_pd_connector(config) == (
            "MultiConnector",
            "parent-engine",
        )

    def test_empty_parent_engine_id_is_not_projected(self):
        config = _kv_config(
            "MultiConnector",
            engine_id="",
            extra={"connectors": [{"kv_connector": "MooncakeConnector"}]},
        )

        assert kv_transfer.resolve_pd_connector(config) == (
            "MultiConnector",
            "",
        )

    @pytest.mark.parametrize(
        "extra",
        [
            None,
            {},
            {"connectors": "not-a-list"},
            {"connectors": [None, {"kv_connector": "NixlConnector"}]},
            {"connectors": [{}, {"kv_connector": "NixlConnector"}]},
            {"connectors": [{"kv_connector": 42}]},
            {"connectors": [{"kv_connector": "NixlConnector", "engine_id": None}]},
        ],
    )
    def test_malformed_multi_connector_is_not_projected(self, extra):
        config = _kv_config("MultiConnector", extra=extra)

        assert kv_transfer.resolve_pd_connector(config) == (
            "MultiConnector",
            "parent-engine",
        )

    # A store child is the mover when no P2P child is configured: the sides
    # meet in the store, and the router's store profile (no bootstrap port, no
    # tagging) applies only when it sees the store's name, not the wrapper's.
    @pytest.mark.parametrize(
        "connectors",
        [
            [{"kv_connector": "MooncakeStoreConnector"}],
            [{"kv_connector": "MooncakeStoreConnector"}, {"kv_connector": "OffloadingConnector"}],
            [{"kv_connector": "OffloadingConnector"}, {"kv_connector": "MooncakeStoreConnector"}],
            [
                {"kv_connector": "OffloadingConnector"},
                {"kv_connector": "MooncakeStoreConnector"},
                {"kv_connector": "LMCacheConnectorV1"},
            ],
        ],
        ids=["alone", "first", "last", "middle"],
    )
    def test_projects_a_lone_store_child_in_any_position(self, connectors):
        config = _kv_config("MultiConnector", extra={"connectors": connectors})

        assert kv_transfer.resolve_pd_connector(config) == (
            "MooncakeStoreConnector",
            "parent-engine",
        )

    def test_projects_a_store_child_with_its_own_engine_id(self):
        config = _kv_config(
            "MultiConnector",
            extra={
                "connectors": [
                    {"kv_connector": "OffloadingConnector"},
                    {"kv_connector": "SharedStorageConnector", "engine_id": "child-engine"},
                ]
            },
        )

        assert kv_transfer.resolve_pd_connector(config) == (
            "SharedStorageConnector",
            "child-engine",
        )

    @pytest.mark.parametrize("connector", ["FlexKVConnectorV1", "HF3FSConnector"])
    def test_store_families_are_recognised_by_prefix(self, connector):
        config = _kv_config(
            "MultiConnector",
            extra={"connectors": [{"kv_connector": connector}, {"kv_connector": "Other"}]},
        )

        assert kv_transfer.resolve_pd_connector(config) == (connector, "parent-engine")

    def test_empty_store_child_engine_id_is_not_projected(self):
        config = _kv_config(
            "MultiConnector",
            extra={"connectors": [{"kv_connector": "MooncakeStoreConnector", "engine_id": ""}]},
        )

        assert kv_transfer.resolve_pd_connector(config) == (
            "MultiConnector",
            "parent-engine",
        )

    # The P2P child the router drives carries the pair's handoff wherever it
    # sits in the list, whatever rides along (a store, an offloading tier, a
    # connector this projection does not know); the store child is not the
    # mover beside it.
    @pytest.mark.parametrize("mover", ["NixlConnector", "MooncakeConnector"])
    @pytest.mark.parametrize("position", [0, 1, 2], ids=["first", "middle", "last"])
    def test_projects_the_p2p_child_in_any_position(self, mover, position):
        riders = [
            {"kv_connector": "MooncakeStoreConnector"},
            {"kv_connector": "UnknownThirdConnector"},
        ]
        connectors = list(riders)
        connectors.insert(position, {"kv_connector": mover, "engine_id": "child-engine"})
        config = _kv_config("MultiConnector", extra={"connectors": connectors})

        assert kv_transfer.resolve_pd_connector(config) == (mover, "child-engine")

    # A MoRI-IO child is projected when nothing else moves the KV (the gRPC
    # router then refuses the pair explicitly instead of a silent recompute),
    # in any position among connectors this projection does not know.
    @pytest.mark.parametrize("position", [0, 1, 2], ids=["first", "middle", "last"])
    def test_projects_a_moriio_child_without_a_store_in_any_position(self, position):
        connectors = [
            {"kv_connector": "OffloadingConnector"},
            {"kv_connector": "UnknownThirdConnector"},
        ]
        connectors.insert(
            position, {"kv_connector": "MoRIIOConnector", "engine_id": "child-engine"}
        )
        config = _kv_config("MultiConnector", extra={"connectors": connectors})

        assert kv_transfer.resolve_pd_connector(config) == ("MoRIIOConnector", "child-engine")

    # Beside exactly one store child a MoRI-IO child steps aside for the
    # store: the gRPC router refuses a disaggregated request on a MoRI-IO
    # worker, while the store served the pair when the wrapper passed through.
    @pytest.mark.parametrize(
        "connectors",
        [
            [{"kv_connector": "MooncakeStoreConnector"}, {"kv_connector": "MoRIIOConnector"}],
            [{"kv_connector": "MoRIIOConnector"}, {"kv_connector": "MooncakeStoreConnector"}],
            [
                {"kv_connector": "MoRIIOConnector", "engine_id": "moriio-engine"},
                {"kv_connector": "OffloadingConnector"},
                {"kv_connector": "MooncakeStoreConnector"},
            ],
        ],
        ids=["store-first", "moriio-first", "with-a-rider"],
    )
    def test_a_moriio_child_steps_aside_for_the_one_store_child(self, connectors):
        config = _kv_config("MultiConnector", extra={"connectors": connectors})

        assert kv_transfer.resolve_pd_connector(config) == (
            "MooncakeStoreConnector",
            "parent-engine",
        )

    def test_a_moriio_child_beside_two_store_children_keeps_the_wrapper(self):
        config = _kv_config(
            "MultiConnector",
            extra={
                "connectors": [
                    {"kv_connector": "MoRIIOConnector"},
                    {"kv_connector": "MooncakeStoreConnector"},
                    {"kv_connector": "SharedStorageConnector"},
                ]
            },
        )

        assert kv_transfer.resolve_pd_connector(config) == (
            "MultiConnector",
            "parent-engine",
        )

    @pytest.mark.parametrize(
        "connectors",
        [
            # Two P2P children: which one hands off is not the servicer's guess.
            [{"kv_connector": "MooncakeConnector"}, {"kv_connector": "NixlConnector"}],
            # Two store children: the same.
            [
                {"kv_connector": "MooncakeStoreConnector"},
                {"kv_connector": "SharedStorageConnector"},
            ],
            # No child this projection knows as a mover.
            [{"kv_connector": "OffloadingConnector"}, {"kv_connector": "ExampleConnector"}],
            [],
        ],
        ids=["two-p2p", "two-store", "no-mover", "no-children"],
    )
    def test_ambiguous_or_unrecognised_children_keep_the_wrapper(self, connectors):
        config = _kv_config("MultiConnector", extra={"connectors": connectors})

        assert kv_transfer.resolve_pd_connector(config) == (
            "MultiConnector",
            "parent-engine",
        )


class TestParamsFromRequest:
    def test_no_params_returns_none(self):
        request = vllm_engine_pb2.GenerateRequest()
        assert kv_transfer.params_from_request(request) is None

    def test_json_field_parsed_verbatim(self):
        request = vllm_engine_pb2.GenerateRequest(kv_transfer_params_json=json.dumps(NIXL_HANDOFF))
        assert kv_transfer.params_from_request(request) == NIXL_HANDOFF

    def test_json_field_preferred_over_typed_field(self):
        request = vllm_engine_pb2.GenerateRequest(
            kv_transfer_params=vllm_engine_pb2.KvTransferParams(
                remote_host="typed-host", remote_port=8998
            ),
            kv_transfer_params_json='{"do_remote_decode": true}',
        )
        assert kv_transfer.params_from_request(request) == {"do_remote_decode": True}

    def test_invalid_json_raises_value_error(self):
        request = vllm_engine_pb2.GenerateRequest(kv_transfer_params_json="{not json")
        with pytest.raises(ValueError, match="Invalid kv_transfer_params_json"):
            kv_transfer.params_from_request(request)

    def test_non_object_json_raises_value_error(self):
        request = vllm_engine_pb2.GenerateRequest(kv_transfer_params_json="[1, 2]")
        with pytest.raises(ValueError, match="must be a JSON object"):
            kv_transfer.params_from_request(request)

    def test_typed_fallback_builds_mooncake_dict(self):
        request = vllm_engine_pb2.GenerateRequest(
            kv_transfer_params=vllm_engine_pb2.KvTransferParams(
                remote_host="prefill-host", remote_port=8998
            )
        )
        assert kv_transfer.params_from_request(request) == {
            "remote_host": "prefill-host",
            "remote_port": 8998,
        }

    @pytest.mark.parametrize("host,port", [("", 8998), ("host", 0), ("host", 65536)])
    def test_typed_fallback_validates_host_and_port(self, host, port):
        request = vllm_engine_pb2.GenerateRequest(
            kv_transfer_params=vllm_engine_pb2.KvTransferParams(remote_host=host, remote_port=port)
        )
        with pytest.raises(ValueError, match="Invalid kv_transfer_params"):
            kv_transfer.params_from_request(request)


class TestParamsToResponseFields:
    def test_none_and_empty_yield_unset_fields(self):
        assert kv_transfer.params_to_response_fields(None) == (None, None)
        assert kv_transfer.params_to_response_fields({}) == (None, None)

    def test_nixl_handoff_roundtrips_through_json(self):
        legacy, params_json = kv_transfer.params_to_response_fields(NIXL_HANDOFF)
        assert json.loads(params_json) == NIXL_HANDOFF
        roundtripped = json.loads(params_json)
        assert roundtripped["remote_port"] == 5600
        assert roundtripped["remote_block_ids"] == [[1, 2, 3]]
        # Legacy mirror carries host/port for old routers
        assert legacy.remote_host == "10.0.0.1"
        assert legacy.remote_port == 5600

    def test_tuple_block_ids_serialize_as_lists(self):
        params = dict(NIXL_HANDOFF, remote_block_ids=((1, 2), (3,)))
        _, params_json = kv_transfer.params_to_response_fields(params)
        assert json.loads(params_json)["remote_block_ids"] == [[1, 2], [3]]

    def test_legacy_mirror_skipped_for_non_scalar_host_port(self):
        legacy, params_json = kv_transfer.params_to_response_fields(
            {"remote_host": ["not", "a", "string"], "remote_port": 5600}
        )
        assert legacy is None
        assert params_json is not None

    @pytest.mark.parametrize(
        "params",
        [
            {"do_remote_prefill": True},
            {"remote_host": "", "remote_port": 5600},
            {"remote_host": "h", "remote_port": 0},
            {"remote_host": "h", "remote_port": 65536},
        ],
    )
    def test_legacy_mirror_skipped_for_missing_or_invalid_host_port(self, params):
        legacy, params_json = kv_transfer.params_to_response_fields(params)
        assert legacy is None
        assert params_json is not None

    def test_non_serializable_params_drop_json_but_keep_mirror(self):
        legacy, params_json = kv_transfer.params_to_response_fields(
            {"remote_host": "h", "remote_port": 1, "bad": object()}
        )
        assert params_json is None
        assert legacy.remote_host == "h"

    def test_response_proto_accepts_fields(self):
        legacy, params_json = kv_transfer.params_to_response_fields(NIXL_HANDOFF)
        complete = vllm_engine_pb2.GenerateComplete(
            kv_transfer_params=legacy,
            kv_transfer_params_json=params_json,
        )
        assert complete.HasField("kv_transfer_params_json")
        assert json.loads(complete.kv_transfer_params_json) == NIXL_HANDOFF

    def test_unset_json_field_when_none(self):
        complete = vllm_engine_pb2.GenerateComplete(
            kv_transfer_params=None,
            kv_transfer_params_json=None,
        )
        assert not complete.HasField("kv_transfer_params_json")
