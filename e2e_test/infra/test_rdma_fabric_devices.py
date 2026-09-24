"""Unit tests for RDMA fabric device detection (no GPU, no RDMA hardware)."""

from __future__ import annotations

from infra.process_utils import detect_rdma_fabric_devices


def _make_sysfs(tmp_path, devices: dict[str, str | None]):
    """Lay out a fake /sys/class/infiniband: device name -> bound netdev."""
    root = tmp_path / "infiniband"
    for name, netdev in devices.items():
        net_dir = root / name / "device" / "net"
        if netdev is None:
            # Device present but bound to no netdev at all.
            (root / name / "device").mkdir(parents=True)
            continue
        net_dir.mkdir(parents=True)
        (net_dir / netdev).mkdir()
    return root


def test_selects_fabric_devices_and_skips_vm_nics(tmp_path, monkeypatch):
    # The layout of an OCI H100 host: 16 fabric NICs on rdma0-15, plus two
    # NICs carrying ordinary VM traffic on eth0/eth1.
    devices = {f"mlx5_{i}": f"rdma{i}" for i in range(16)}
    devices["mlx5_16"] = "eth0"
    devices["mlx5_17"] = "eth1"
    root = _make_sysfs(tmp_path, devices)
    monkeypatch.setattr("infra.process_utils.os.listdir", _listdir_under(root))

    found = detect_rdma_fabric_devices()

    assert "mlx5_16:1" not in found
    assert "mlx5_17:1" not in found
    assert len(found) == 16
    assert set(found) == {f"mlx5_{i}:1" for i in range(16)}


def test_device_without_a_netdev_is_skipped(tmp_path, monkeypatch):
    root = _make_sysfs(tmp_path, {"mlx5_0": "rdma0", "mlx5_1": None})
    monkeypatch.setattr("infra.process_utils.os.listdir", _listdir_under(root))

    assert detect_rdma_fabric_devices() == ["mlx5_0:1"]


def test_no_fabric_devices_returns_empty(tmp_path, monkeypatch):
    """Off-cluster, callers must leave UCX to its own defaults."""
    root = _make_sysfs(tmp_path, {"mlx5_0": "eth0"})
    monkeypatch.setattr("infra.process_utils.os.listdir", _listdir_under(root))

    assert detect_rdma_fabric_devices() == []


def test_missing_sysfs_returns_empty(monkeypatch):
    def explode(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr("infra.process_utils.os.listdir", explode)

    assert detect_rdma_fabric_devices() == []


def _listdir_under(root):
    """Redirect listdir of /sys/class/infiniband/... into the fake tree."""
    import os as _os

    real = _os.listdir
    prefix = "/sys/class/infiniband"

    def fake(path):
        text = str(path)
        if text.startswith(prefix):
            return real(str(root) + text[len(prefix) :])
        return real(path)

    return fake
