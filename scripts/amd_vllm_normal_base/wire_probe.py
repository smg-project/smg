"""Same-source generated gRPC transport probe; explicit CPU fixture, not an engine."""
import asyncio
import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys


async def main():
    # Run in a fresh, normally installed probe environment, never via PYTHONPATH.
    assert not any(k == 'torch' or k.startswith(('vllm', 'sglang')) for k in sys.modules)
    runtime_pins = json.loads(Path(__file__).with_name('postflight-inputs.json').read_text())['runtime_versions']
    assert set(runtime_pins) == {'grpcio', 'smg-grpc-proto', 'smg-grpc-servicer'}
    assert all(isinstance(v, str) and v for v in runtime_pins.values()), 'unbound normal RPC runtime'
    assert all(metadata.version(k) == v for k, v in runtime_pins.items()), 'normal RPC runtime identity'
    origins = {}
    names = ['common', 'mlx_engine', 'sglang_encoder', 'sglang_scheduler',
             'tokenspeed_encoder', 'tokenspeed_scheduler', 'trtllm_service', 'vllm_engine']
    for name in names:
        for suffix in ['_pb2', '_pb2_grpc']:
            mod = importlib.import_module('smg_grpc_proto.generated.' + name + suffix)
            p = Path(mod.__file__).resolve()
            origins[mod.__name__] = {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    import grpc
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc
    from grpc_reflection.v1alpha import reflection, reflection_pb2, reflection_pb2_grpc
    from smg_grpc_proto.generated import common_pb2 as common, vllm_engine_pb2 as pb
    from smg_grpc_proto.generated import vllm_engine_pb2_grpc as rpc
    for name in ['sglang_scheduler', 'tokenspeed_scheduler', 'vllm_engine']:
        desc = importlib.import_module('smg_grpc_proto.generated.' + name + '_pb2').DESCRIPTOR
        assert any({'StartProfile', 'StopProfile'} <= set(s.methods_by_name) for s in desc.services_by_name.values())
    assert common.EngineLoad.DESCRIPTOR.fields_by_name['engine_silence_ms'].number == 19
    absent = common.EngineLoad()
    present = common.EngineLoad(engine_silence_ms=0)
    assert not absent.HasField('engine_silence_ms') and present.HasField('engine_silence_ms')

    timed_out = asyncio.Event()
    cancelled = asyncio.Event()

    class Fixture(rpc.VllmEngineServicer):
        async def StartProfile(self, request, context):
            if request.output_dir == 'fixture-deadline':
                try:
                    await asyncio.Event().wait()
                finally:
                    timed_out.set()
            assert request.HasField('output_dir') and request.output_dir == 'cpu-fixture-only'
            return common.ProfileResponse(success=True, message='fixture start')

        async def StopProfile(self, request, context):
            return common.ProfileResponse(success=True, message='fixture stop')

        async def SubscribeKvEvents(self, request, context):
            yield common.KvEventBatch(sequence_number=1, load=present)

        async def Generate(self, request, context):
            assert request.request_id in ['cpu-fixture-rid', 'cpu-cancel-rid']
            try:
                yield pb.GenerateResponse(complete=pb.GenerateComplete(output_ids=[1], finish_reason='stop'))
                if request.request_id == 'cpu-cancel-rid':
                    await asyncio.Event().wait()
            finally:
                if request.request_id == 'cpu-cancel-rid':
                    cancelled.set()

        async def Abort(self, request, context):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'fixture refusal')

    server = grpc.aio.server()
    rpc.add_VllmEngineServicer_to_server(Fixture(), server)
    h = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(h, server)
    service = pb.DESCRIPTOR.services_by_name['VllmEngine'].full_name
    reflection.enable_server_reflection((service, health.SERVICE_NAME, reflection.SERVICE_NAME), server)
    port = server.add_insecure_port('127.0.0.1:0')
    assert port > 0
    await server.start()
    try:
        await h.set('', health_pb2.HealthCheckResponse.SERVING)
        async with grpc.aio.insecure_channel(f'127.0.0.1:{port}') as channel:
            stub = rpc.VllmEngineStub(channel)
            assert (await stub.StartProfile(common.StartProfileRequest(output_dir='cpu-fixture-only'), timeout=2)).success
            assert (await stub.StopProfile(common.StopProfileRequest(), timeout=2)).success
            try:
                await stub.StartProfile(common.StartProfileRequest(output_dir='fixture-deadline'), timeout=0.05)
            except grpc.aio.AioRpcError as error:
                assert error.code() is grpc.StatusCode.DEADLINE_EXCEEDED
            else:
                raise AssertionError('deadline was lost')
            await asyncio.wait_for(timed_out.wait(), timeout=2)
            events = [x async for x in stub.SubscribeKvEvents(common.SubscribeKvEventsRequest(), timeout=2)]
            assert len(events) == 1 and events[0].load.HasField('engine_silence_ms') and events[0].load.engine_silence_ms == 0
            outputs = [x async for x in stub.Generate(pb.GenerateRequest(request_id='cpu-fixture-rid'), timeout=2)]
            assert len(outputs) == 1 and list(outputs[0].complete.output_ids) == [1]
            stream = stub.Generate(pb.GenerateRequest(request_id='cpu-cancel-rid'), timeout=2)
            assert list((await stream.read()).complete.output_ids) == [1]
            assert stream.cancel()
            assert await stream.code() is grpc.StatusCode.CANCELLED
            await asyncio.wait_for(cancelled.wait(), timeout=2)
            try:
                await stub.Abort(pb.AbortRequest(), timeout=2)
            except grpc.aio.AioRpcError as error:
                assert error.code() is grpc.StatusCode.INVALID_ARGUMENT
            else:
                raise AssertionError('fixture refusal was lost')
            health_reply = await health_pb2_grpc.HealthStub(channel).Check(health_pb2.HealthCheckRequest(), timeout=2)
            assert health_reply.status == health_pb2.HealthCheckResponse.SERVING
            request = reflection_pb2.ServerReflectionRequest(list_services='')
            replies = [x async for x in reflection_pb2_grpc.ServerReflectionStub(channel).ServerReflectionInfo(iter([request]), timeout=2)]
            assert any(service == s.name for reply in replies for s in reply.list_services_response.service)
    finally:
        await server.stop(0)
        await server.wait_for_termination(timeout=2)
    assert not any(k == 'torch' or k.startswith(('vllm', 'sglang')) for k in sys.modules)
    print(json.dumps({'result': 'PASS_CPU_FIXTURE_GENERATED_GRPC_WIRE_ONLY',
                      'origins16': origins, 'field19_absent_vs_zero': True,
                      'new_vllm_StartStopProfile': True, 'health_reflection': True,
                      'deadline_and_cancel': True,
                      'production_servicer': False, 'engine': False, 'GPU': False}))


if __name__ == '__main__':
    asyncio.run(asyncio.wait_for(main(), timeout=15))
