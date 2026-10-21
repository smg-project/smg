"""Engine-neutral NIXL pixel-payload puller for gateway-exported multimodal inputs."""

from __future__ import annotations

import logging
import os
import queue
import socket
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_DESCRIPTOR_MAGIC = b"SMGRDMA1"
_GEN_BYTES = 8
_FRAME_BYTES = 2 * _GEN_BYTES
# The gateway's fixed NIXL agent name (see RDMA_GATEWAY_AGENT_NAME gateway-side).
DEFAULT_GATEWAY_AGENT_NAME = "smg-gateway-encode"
_LOCAL_IP_CACHE: dict[str, bool] = {}

# The spellings that turn an SMG env switch on, the same set the gateway
# reads (``1``, ``true``, ``yes``, ``on``, in any case, surrounding blanks
# ignored); anything else leaves the switch off.
_SWITCH_ON = frozenset({"1", "true", "yes", "on"})


def rdma_pixel_lane_enabled(environ) -> bool:
    """Whether ``SMG_MM_PIXEL_RDMA`` asks for the RDMA pixel lane, read with
    the gateway's spelling rule, so gateway and worker agree on one value."""
    return (environ.get("SMG_MM_PIXEL_RDMA") or "").strip().lower() in _SWITCH_ON


@dataclass(frozen=True)
class _RemotePixelDescriptor:
    remote_addr: int
    expected_gen: int
    room: int
    port: int
    ip: str


def _ip_is_local(ip: str) -> bool:
    """True iff ``ip`` is assigned to a local interface of this host."""
    cached = _LOCAL_IP_CACHE.get(ip)
    if cached is not None:
        return cached
    result = False
    if ip in ("localhost", "::1") or ip.startswith("127."):
        result = True
    else:
        for fam in (socket.AF_INET, socket.AF_INET6):
            try:
                sock = socket.socket(fam, socket.SOCK_DGRAM)
            except OSError:
                continue
            try:
                sock.bind((ip, 0))
                result = True
            except OSError:
                pass
            finally:
                sock.close()
            if result:
                break
    _LOCAL_IP_CACHE[ip] = result
    return result


def _parse_descriptor(td, explicit_room: int | None) -> _RemotePixelDescriptor:
    """Parse both the current room-carrying descriptor and the legacy EPD form."""
    desc = bytes(td.remote.descriptor)
    if desc.startswith(_DESCRIPTOR_MAGIC):
        min_len = len(_DESCRIPTOR_MAGIC) + 8 + _GEN_BYTES + 8 + 2
        if len(desc) < min_len:
            raise ValueError("remote descriptor too short")
        off = len(_DESCRIPTOR_MAGIC)
        remote_addr = int.from_bytes(desc[off : off + 8], "little")
        off += 8
        expected_gen = int.from_bytes(desc[off : off + _GEN_BYTES], "little")
        off += _GEN_BYTES
        room = int.from_bytes(desc[off : off + 8], "little", signed=True)
        off += 8
        port = int.from_bytes(desc[off : off + 2], "little")
        off += 2
        ip = desc[off:].decode()
        if explicit_room is not None and int(explicit_room) != room:
            raise ValueError(
                f"remote descriptor room mismatch: descriptor={room} request={int(explicit_room)}"
            )
        return _RemotePixelDescriptor(
            remote_addr=remote_addr,
            expected_gen=expected_gen,
            room=room,
            port=port,
            ip=ip,
        )

    if explicit_room is None:
        raise ValueError(
            "legacy remote descriptor lacks room; gateway must emit SMGRDMA1 descriptor"
        )
    if len(desc) < 8 + _GEN_BYTES + 2:
        raise ValueError("remote descriptor too short")
    return _RemotePixelDescriptor(
        remote_addr=int.from_bytes(desc[:8], "little"),
        expected_gen=int.from_bytes(desc[8 : 8 + _GEN_BYTES], "little"),
        room=int(explicit_room),
        port=int.from_bytes(desc[8 + _GEN_BYTES : 10 + _GEN_BYTES], "little"),
        ip=desc[10 + _GEN_BYTES :].decode(),
    )


class RdmaPixelPuller:
    """Persistent NIXL READ agent for gateway-exported multimodal pixel buffers."""

    def __init__(
        self,
        *,
        agent_name: str,
        log_prefix: str,
        gateway_agent_name: str = DEFAULT_GATEWAY_AGENT_NAME,
    ):
        self._log_prefix = log_prefix
        self._gateway_agent_name = gateway_agent_name
        self._nixl_agent = None
        self._rdma_md_ready = set()
        self._rdma_md_lock = threading.Lock()
        self._landing = None
        self._landing_np = None
        self._landing_base = 0
        self._landing_slot_bytes = 0
        self._landing_free = None

        if not rdma_pixel_lane_enabled(os.environ):
            return

        try:
            try:
                from nixl_cu13._api import nixl_agent, nixl_agent_config
            except ImportError:
                from nixl._api import nixl_agent, nixl_agent_config
            import torch

            self._nixl_agent = nixl_agent(
                agent_name,
                nixl_agent_config(
                    enable_listen_thread=True,
                    listen_port=0,
                    backends=["UCX"],
                ),
            )

            slot_bytes = int(os.environ.get("SMG_RDMA_SLOT_BYTES", 32 * 1024 * 1024))
            n_slots = int(os.environ.get("SMG_RDMA_LANDING_SLOTS", 64))
            self._landing = torch.empty(
                n_slots * slot_bytes,
                dtype=torch.uint8,
                pin_memory=True,
            )
            self._landing_np = self._landing.numpy()
            self._landing_base = self._landing.data_ptr()
            self._landing_slot_bytes = slot_bytes
            reg = self._nixl_agent.get_reg_descs(
                [(self._landing_base, n_slots * slot_bytes, 0, "")],
                "DRAM",
            )
            self._nixl_agent.register_memory(reg)
            self._landing_free = queue.Queue()
            for i in range(n_slots):
                self._landing_free.put(i)
            logger.info(
                "%s: puller up (landing %d slots x %d B)",
                self._log_prefix,
                n_slots,
                slot_bytes,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("%s: NIXL init failed (%s); inline only", self._log_prefix, exc)
            self._nixl_agent = None
            self._landing_free = None

    def _ensure_remote_ready(self, ip: str, port: int, remote, room: int) -> None:
        """One-time metadata handshake per gateway listener."""
        key = (ip, port)
        if key in self._rdma_md_ready:
            return
        with self._rdma_md_lock:
            if key in self._rdma_md_ready:
                return

            agent = self._nixl_agent
            agent.fetch_remote_metadata(self._gateway_agent_name, ip, port)

            send_md_env = os.environ.get("SMG_RDMA_SEND_MD")
            if send_md_env in ("1", "true"):
                do_send_md = True
            elif send_md_env in ("0", "false"):
                do_send_md = False
            else:
                do_send_md = _ip_is_local(ip)
            logger.info(
                "%s: gateway %s:%s local=%s send_md=%s (env=%s) -- one-time md handshake",
                self._log_prefix,
                ip,
                port,
                _ip_is_local(ip),
                do_send_md,
                send_md_env,
            )
            if do_send_md:
                agent.send_local_metadata(ip, port)

            ready = False
            for _ in range(5000):
                if agent.check_remote_metadata(self._gateway_agent_name, remote):
                    ready = True
                    break
                time.sleep(0.001)
            if not ready:
                raise RuntimeError(f"NIXL remote metadata not ready room={room}")
            self._rdma_md_ready.add(key)

    def feature_from_remote(self, td, *, explicit_room: int | None, cast_to):
        """Read a remote TensorData payload and return the feature tensor."""
        import numpy as np
        import torch

        transport = getattr(td.remote, "transport", "")
        if transport and transport != "nixl":
            raise ValueError(f"unsupported remote tensor transport: {transport!r}")

        agent = self._nixl_agent
        if agent is None or self._landing_free is None:
            raise RuntimeError(f"{self._log_prefix}: remote payload but landing pool unavailable")

        descriptor = _parse_descriptor(td, explicit_room)
        nbytes = int(td.remote.nbytes)
        if nbytes + _FRAME_BYTES > self._landing_slot_bytes:
            raise ValueError(
                f"remote pixel {nbytes}B (+{_FRAME_BYTES}B gen frame) exceeds "
                f"landing slot {self._landing_slot_bytes}B"
            )

        remote = agent.get_xfer_descs(
            [(descriptor.remote_addr, nbytes + _FRAME_BYTES, 0)],
            "DRAM",
        )
        self._ensure_remote_ready(
            descriptor.ip,
            descriptor.port,
            remote,
            descriptor.room,
        )

        wait_budget = float(os.environ.get("SMG_RDMA_LANDING_WAIT_S", 120))
        t_acq = time.monotonic()
        slot = None
        while slot is None:
            try:
                slot = self._landing_free.get(timeout=5)
            except queue.Empty:
                waited = time.monotonic() - t_acq
                if waited >= wait_budget:
                    raise RuntimeError(
                        f"{self._log_prefix}: landing-ring starvation for {waited:.0f}s "
                        f"(room={descriptor.room}); raise SMG_RDMA_LANDING_SLOTS / "
                        f"SMG_RDMA_LANDING_WAIT_S"
                    ) from None
                logger.warning(
                    "%s: landing ring exhausted for %.0fs (room=%s, qsize=%d); backpressuring",
                    self._log_prefix,
                    waited,
                    descriptor.room,
                    self._landing_free.qsize(),
                )

        try:
            off = slot * self._landing_slot_bytes
            laddr = self._landing_base + off
            local = agent.get_xfer_descs([(laddr, nbytes + _FRAME_BYTES, 0)], "DRAM")
            handle = agent.initialize_xfer(
                "READ",
                local,
                remote,
                self._gateway_agent_name,
                str(descriptor.room).encode(),
            )
            read_deadline = time.monotonic() + float(os.environ.get("SMG_RDMA_READ_TIMEOUT_S", 60))
            spins = 0
            state = agent.transfer(handle)
            while state in ("PROC", "IN_PROG"):
                spins += 1
                if spins > 64:
                    time.sleep(0.0005)
                if time.monotonic() > read_deadline:
                    agent.release_xfer_handle(handle)
                    raise RuntimeError(f"NIXL READ timed out room={descriptor.room}")
                state = agent.check_xfer_state(handle)
            agent.release_xfer_handle(handle)
            if state != "DONE":
                raise RuntimeError(f"NIXL READ state={state} room={descriptor.room}")

            shape = list(td.shape)
            itemsize = 2 if td.dtype == "bfloat16" else np.dtype(td.dtype).itemsize
            expected = itemsize
            for dim in shape:
                expected *= dim
            if nbytes != expected:
                raise ValueError(
                    f"remote pixel size mismatch: nbytes={nbytes} expected={expected} "
                    f"(shape={shape} dtype={td.dtype})"
                )

            hdr = int.from_bytes(
                self._landing_np[off : off + _GEN_BYTES].tobytes(),
                "little",
            )
            trl = int.from_bytes(
                self._landing_np[off + _GEN_BYTES + nbytes : off + _FRAME_BYTES + nbytes].tobytes(),
                "little",
            )
            if hdr != descriptor.expected_gen or trl != descriptor.expected_gen:
                raise ValueError(
                    f"{self._log_prefix}: slot generation mismatch room={descriptor.room} "
                    f"expected={descriptor.expected_gen} header={hdr} trailer={trl} "
                    f"(slot recycled/torn under a live READ -> wrong-image guard)"
                )

            slot_np = self._landing_np[off + _GEN_BYTES : off + _GEN_BYTES + nbytes]
            if td.dtype == "bfloat16":
                tensor = torch.from_numpy(slot_np.view(np.uint16).reshape(shape)).view(
                    torch.bfloat16
                )
            else:
                tensor = torch.from_numpy(slot_np.view(np.dtype(td.dtype)).reshape(shape))
            copied = False
            if cast_to is not None and tensor.dtype != cast_to and tensor.is_floating_point():
                tensor = tensor.to(cast_to)
                copied = True

            # Copy out of the landing slot before it returns to the free ring.
            return tensor if copied else tensor.clone()
        finally:
            self._landing_free.put(slot)
