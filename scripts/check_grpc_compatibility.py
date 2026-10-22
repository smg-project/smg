"""Check two generated gRPC packages over real local peers; no engine is loaded."""

import asyncio
import importlib
import json
import os
import select
import signal
import subprocess
import sys
from pathlib import Path


def load(generated):
    sys.path.insert(0, generated)
    from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc

    return vllm_engine_pb2, vllm_engine_pb2_grpc


async def server(repo, generated):
    import grpc
    from grpc_health.v1 import health_pb2, health_pb2_grpc
    from grpc_reflection.v1alpha import reflection

    sys.path.insert(0, str(Path(repo) / "grpc_servicer"))
    from smg_grpc_servicer.health_frame import HealthFrame

    pb, stubs = load(generated)
    service = pb.DESCRIPTOR.services_by_name["VllmEngine"].full_name

    class Engine(stubs.VllmEngineServicer):
        async def HealthCheck(self, request, context):
            await context.send_initial_metadata((("qa-server", grpc.__version__),))
            context.set_trailing_metadata((("qa-finished", "yes"),))
            return pb.HealthCheckResponse(healthy=True, message="cpu-peer")

        async def Generate(self, request, context):
            for index in range(3):
                yield pb.GenerateResponse(chunk=pb.GenerateStreamChunk(token_ids=[10 + index]))
                if request.request_id == "cancel":
                    await asyncio.sleep(30)

        async def Abort(self, request, context):
            if "invalid" in request.request_ids:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "cpu-negative")
            if "slow" in request.request_ids:
                await asyncio.sleep(30)
            return pb.AbortResponse()

    class Readiness(HealthFrame):
        ENGINE_SERVICE = service

        async def engine_status(self, service_name):
            return health_pb2.HealthCheckResponse.SERVING

    endpoint = grpc.aio.server()
    stubs.add_VllmEngineServicer_to_server(Engine(), endpoint)
    health_pb2_grpc.add_HealthServicer_to_server(Readiness(), endpoint)
    reflection.enable_server_reflection(
        (service, "grpc.health.v1.Health", reflection.SERVICE_NAME), endpoint
    )
    port = endpoint.add_insecure_port("127.0.0.1:0")
    assert port
    await endpoint.start()
    print(json.dumps({"port": port, "grpc": grpc.__version__}), flush=True)
    stop = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
    await stop.wait()
    await endpoint.stop(0)


async def client(generated, address):
    import grpc
    from google.protobuf import descriptor_pb2
    from grpc_health.v1 import health_pb2, health_pb2_grpc
    from grpc_reflection.v1alpha import reflection_pb2, reflection_pb2_grpc

    pb, stubs = load(generated)
    checks = []
    async with grpc.aio.insecure_channel(address) as channel:
        stub = stubs.VllmEngineStub(channel)
        call = stub.HealthCheck(pb.HealthCheckRequest(), timeout=3)
        assert (await call).healthy
        assert [value for key, value in await call.initial_metadata() if key == "qa-server"]
        assert [value for key, value in await call.trailing_metadata() if key == "qa-finished"] == [
            "yes"
        ]
        checks.append("unary-and-metadata")
        rows = [
            response
            async for response in stub.Generate(pb.GenerateRequest(request_id="finite"), timeout=3)
        ]
        assert [list(row.chunk.token_ids) for row in rows] == [[10], [11], [12]]
        checks.append("server-streaming")
        for request_id, expected in [
            ("invalid", grpc.StatusCode.INVALID_ARGUMENT),
            ("slow", grpc.StatusCode.DEADLINE_EXCEEDED),
        ]:
            try:
                await stub.Abort(
                    pb.AbortRequest(request_ids=[request_id]),
                    timeout=0.05 if request_id == "slow" else 3,
                )
                raise AssertionError("Expected RPC refusal")
            except grpc.aio.AioRpcError as error:
                assert error.code() == expected
            checks.append(request_id)
        stream = stub.Generate(pb.GenerateRequest(request_id="cancel"), timeout=3)
        assert list((await stream.read()).chunk.token_ids) == [10]
        assert stream.cancel() and stream.cancelled()
        checks.append("stream-cancel")
        health = health_pb2_grpc.HealthStub(channel)
        assert (
            await health.Check(health_pb2.HealthCheckRequest(), timeout=3)
        ).status == health_pb2.HealthCheckResponse.SERVING
        checks.append("health-unary")
        try:
            await health.Check(health_pb2.HealthCheckRequest(service="unknown"), timeout=3)
            raise AssertionError("Expected unknown-service refusal")
        except grpc.aio.AioRpcError as error:
            assert error.code() == grpc.StatusCode.NOT_FOUND
        checks.append("health-unknown-check")
        watch = [
            row
            async for row in health.Watch(
                health_pb2.HealthCheckRequest(service="unknown"), timeout=3
            )
        ]
        assert [row.status for row in watch] == [health_pb2.HealthCheckResponse.SERVICE_UNKNOWN]
        checks.append("health-unknown-watch")
        reflect = reflection_pb2_grpc.ServerReflectionStub(channel)
        service = pb.DESCRIPTOR.services_by_name["VllmEngine"].full_name
        for request, kind in [
            (reflection_pb2.ServerReflectionRequest(list_services=""), "list"),
            (reflection_pb2.ServerReflectionRequest(file_containing_symbol=service), "descriptor"),
            (
                reflection_pb2.ServerReflectionRequest(file_containing_symbol="unknown.Service"),
                "unknown",
            ),
        ]:
            (response,) = [
                row async for row in reflect.ServerReflectionInfo(iter([request]), timeout=3)
            ]
            if kind == "list":
                assert service in [entry.name for entry in response.list_services_response.service]
            elif kind == "descriptor":
                files = [
                    descriptor_pb2.FileDescriptorProto.FromString(data)
                    for data in response.file_descriptor_response.file_descriptor_proto
                ]
                assert any(file.name == "vllm_engine.proto" for file in files)
            else:
                assert response.error_response.error_code == grpc.StatusCode.NOT_FOUND.value[0]
            checks.append("reflection-" + kind)
    print(json.dumps({"grpc": grpc.__version__, "checks": checks, "check_count": len(checks)}))


def fingerprint(generated):
    sys.path.insert(0, generated)
    files = {}

    def include(descriptor):
        files[descriptor.name] = descriptor.serialized_pb.hex()
        for dependency in descriptor.dependencies:
            include(dependency)

    for path in sorted((Path(generated) / "smg_grpc_proto/generated").glob("*_pb2.py")):
        include(importlib.import_module("smg_grpc_proto.generated." + path.stem).DESCRIPTOR)
    assert len(files) >= 8
    print(json.dumps(files, sort_keys=True))


def check(peers):
    script = str(Path(__file__).resolve())
    repo = str(Path(script).parents[1])
    fingerprints = [
        subprocess.check_output([python, script, "fingerprint", generated], text=True)
        for python, generated in peers
    ]
    assert json.loads(fingerprints[0]) == json.loads(fingerprints[1])
    results = []
    for python, generated in peers:
        process = subprocess.Popen(
            [python, script, "server", repo, generated], text=True, stdout=subprocess.PIPE
        )
        try:
            assert select.select([process.stdout], [], [], 10)[0], "Server startup timeout"
            ready = json.loads(process.stdout.readline())
            for client_python, client_generated in peers:
                output = subprocess.check_output(
                    [
                        client_python,
                        script,
                        "client",
                        client_generated,
                        f"127.0.0.1:{ready['port']}",
                    ],
                    text=True,
                    timeout=15,
                )
                results.append({"server": ready, "client": json.loads(output)})
        finally:
            process.terminate()
            process.wait(timeout=5)
    refused = subprocess.run(
        [peers[0][0], "-c", "from smg_grpc_proto.generated import common_pb2_grpc"],
        env=dict(os.environ, PYTHONPATH=peers[1][1]),
        text=True,
        capture_output=True,
    )
    assert refused.returncode != 0 and "grpcio>=1.81.1" in refused.stderr
    print(
        json.dumps(
            {
                "descriptor_files": len(json.loads(fingerprints[0])),
                "pairs": results,
                "old_gencode_rejected": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    if sys.argv[1] == "server":
        asyncio.run(server(*sys.argv[2:]))
    elif sys.argv[1] == "client":
        asyncio.run(client(*sys.argv[2:]))
    elif sys.argv[1] == "fingerprint":
        fingerprint(sys.argv[2])
    else:
        assert len(sys.argv) == 5, "Expected python178 gen178 python181 gen181"
        check([(sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])])
