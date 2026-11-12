"""The fixed SDK patch rejects unknown inputs and validates both outputs before writes."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "ci_sglang_mxfp4_fallback.py"


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("fixed_sglang_patch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "sdk"
    root.mkdir()

    def digest(body):
        return hashlib.sha256(body).hexdigest()

    module.FILES = {name: (digest(b"before\n"), digest(b"after\n")) for name in ("a.py", "b.py")}
    for name in module.FILES:
        (root / name).write_bytes(b"before\n")
    patch = tmp_path / "fixed.patch"
    patch.write_text(
        "".join(
            f"--- a/python/sglang/{name}\n+++ b/python/sglang/{name}\n@@ -1 +1 @@\n-before\n+after\n"
            for name in module.FILES
        )
    )
    module.PATCH = patch
    module.PATCH_SHA = digest(patch.read_bytes())
    monkeypatch.setenv(module.ROW_ENV, "1")
    return module, root


def test_original_pair_installs_once_and_verifies_final_pair(sdk):
    module, root = sdk
    assert module.install(root)["state"] == "patched"
    assert module.install(root) == module.verify_installed(root)
    assert all((root / name).read_bytes() == b"after\n" for name in module.FILES)


@pytest.mark.parametrize("body", [b"unknown\n", b"after\n"])
def test_unknown_or_mixed_pair_never_changes_other_source(sdk, body):
    module, root = sdk
    (root / "a.py").write_bytes(body)
    with pytest.raises(RuntimeError, match="unknown or mixed"):
        module.install(root)
    assert (root / "a.py").read_bytes() == body and (root / "b.py").read_bytes() == b"before\n"


def test_second_output_mismatch_leaves_both_installed_sources_original(sdk):
    module, root = sdk
    module.FILES["b.py"] = (module.FILES["b.py"][0], "0" * 64)
    with pytest.raises(RuntimeError, match="output differs"):
        module.install(root)
    assert all((root / name).read_bytes() == b"before\n" for name in module.FILES)


def test_changed_patch_rejected_before_source_writes(sdk):
    module, root = sdk
    module.PATCH.write_bytes(b"changed patch")
    with pytest.raises(RuntimeError, match="patch bytes changed"):
        module.install(root)
    assert all((root / name).read_bytes() == b"before\n" for name in module.FILES)


def test_disabled_mode_requires_original_sources_and_invalid_mode_fails(sdk, monkeypatch):
    module, root = sdk
    monkeypatch.setenv(module.ROW_ENV, "0")
    assert module.install(root)["state"] == "original"
    (root / "a.py").write_bytes(b"after\n")
    with pytest.raises(RuntimeError, match="mixed"):
        module.verify_installed(root)
    monkeypatch.setenv(module.ROW_ENV, "yes")
    with pytest.raises(RuntimeError, match="must be 0 or 1"):
        module.install(root)


# Exact public metadata at the original 6cee revision, not model weights.
_ORIGINAL_METADATA_ZLIB = {
    "config.json": (
        "eJztVUtu2zAQ3fsUhpZFK8iOnTjdFmg3LXqAIiBocSQRoUiWQ/qTIHfvkNTHsdETtFrY0LzhcD7vjV4Xy2XBXd1J"
        "D7UPDrD4vPxFRjJ/s/4n4lfjvvCAXH3/UZD96WM64T1oL41me8njEe8CXCHCGWuCJ7Aqq4SBQebNM2gmBZnXFT3r"
        "jJwsOI+MfrMHwZuEdMAFE7Inw/1gkUJQCF7H0AVKFYpLO8oXiMF3u3yp1NJLrlhttIeTZwp067sYv3q8v/Sgc445"
        "rlvIKa8H0IPrQUju4Sa24ueY8Nletg2VFFK3bOpESo+AJih1a/3v/i+7T4rq+YlZgzJJB/o9iHg+0mp1t6oeMhl7"
        "I0AlvkXqt9Yzg5jZr0M/h2VRNThLJoJXEpsEFrFBOonO8dh6hp7hzA5cBZhi7iZMmZqENQQm5C5nSaK3wTNH/3SX"
        "Mq1MaMMV5h1hSdIXe2D1GJ+E/A6cCnjhqQhSbCNbcnjNraPqg4K4Qpg2PsIHungS3tifMpdRfigRVBObMjb/1qNX"
        "tsx5XrukEeQsccZUn9pQpPenYaQpadaD70wsp+hPjd1El7dUk+uR8nU0AZvGCZ+qbQaMpZVCLaRBz1XuwXPWcPSp"
        "oXlzjmZU5hhDTNaGtqBx7z2Nk63UNJi/U2pcfWMSI6PO3E1EpY2ua1p64+TmetKJjvKJqWzjEh/sad48nGjmiDQf"
        "aNImzaMdFXGUWuQq1plJeJStCkzJXsaaH4aPhZfAjsaJ95nPJPK0qrGhvtIkGREBo5yohk253ZZVKeBQZWUEBFbz"
        "uoPLz9SBqLuf9nm1qna7xdviD5VELaw="
    ),
    "model.safetensors.index.json": (
        "eJy13N1u00oUhuFbqXIMEf6JneybiQxxITSJQ+zCBtR7py0HIDQer1nfOxxREV68LPRoNIrWz9W5n7pDN3Wr/+5+"
        "rqZh6k778fijf/6xqNqmKJu6bbZPb+5W3/rjx0/T/txdXz96Hg79aX3qvve3cf1ufbxcH6f964+X4XZe//708yd/"
        "f/Dtu5dfb4f719+U67G776f+Mg63cfXm7t/Y2J/u9900XdbD/nobPq/fH7uRakFP9gV8sgew9RVsfUHf2ANa+4rW"
        "xuPlwfu+zqfr+jY8Tv1Neet/VaSZXjr9/9f+No3rj93U7x+vr29qLz5buHkaPkjvLVgdP3Snnqgehm8XdPi/gtTk"
        "f5LS2NdhnF7+K/eX6ThcGIgLUvUCVD3Qgp5MVb0AVS9A1QMzQm9MVz0wJ1Tzq14gqheQ6kUG1WNNv22RqsBbQas+"
        "G6QmR1Qv8qiOHtYL8rQeilHPJstOHtgL8sQeGpN6a4Du7KG9YE7tBXNsL6hze5Hj4B6NCtLlOboX+Nl9vogNzzif"
        "6fjOnt/RAzx8gkeP8OgZHj3Ew6d4+BgPn+Ohgzx0kseO8lnO8pkO85lO8/xxPsd5PseBPtOJvkShL0noAzHq2WTo"
        "SxL6koQ+MCb11gDoA6NSOQH6koG+pKAvc0AfiwrWRbIKdyUO/WwRG56BvswDfYVCX5HQB2LUs8nQVyT0FQl9YEzq"
        "rQHQB0alcgL0FQN9RUFf5YA+FhWsi2QV7ioc+tkiNjwDfZUH+hqFviahD8SoZ5Ohr0noaxL6wJjUWwOgD4xK5QTo"
        "awb6moK+zgF9LCpYF8kq3NU49LNFbHgG+joP9BsU+g0JfSBGPZsM/YaEfkNCHxiTemsA9IFRqZwA/YaBfkNBv8kB"
        "fSwqWBfJKtxtcOhni9jwDPSbPNA3KPQNCX0gRj2bDH1DQt+Q0AfGpN4aAH1gVConQN8w0DcU9E0O6GNRwbpIVuGu"
        "waGfLWLDM9A3eaBvUehbEvpAjHo2GfqWhL4loQ+MSb01APrAqFROgL5loG8p6Nsc0MeignWRrMJdi0M/W8SGZ6Bv"
        "80C/RaHfktAHYtSzydBvSei3JPSBMUNvrfAPSuUi0HtyAejtmSj0row2VgL0RDRgHZANcefLxqCXi9jwcejtzSTo"
        "rdmdCXp7zQC9EKOebRZ6T2wWek9sFnphTOqtAdAHRqVyAvQ7BvodBf0uB/SxqGBdJKtwt8Ohny1iwzPQ77JAb/sa"
        "vTnGMR9oQU+mIl+Cxpcg8YEZoTemAx+YE6r5eV/48rynIs3E2574zXm9Kvhm/968GqQmR1hP/NK8tWrbd2Cvga4b"
        "9x14crLspn0Hnphsu3HfgTAolQN4X9h3YM8gwC/uO0gLwcQn7jsAsop29n0HchEbnnE+cd+BNWvbd2CvgdAb9x14"
        "cjL0pn0HnpgMvXHfgTAolQOgX9h3YM8g0C/uO0gLwdAn7jsAsgp39n0HchEbnoE+cd+BNcte1KA3NfBVDXpXg17W"
        "oLc18HUNfF8DX9hANzbQlQ12Z5Pl0ibTrU2maxv+3ibHxU2Om5tMVze2fQf2Ggi9cd+BJydDb9p34InJ0Bv3HQiD"
        "UjkA+oV9B/YMAv3ivoO0EAx94r4DIKtwZ993IBex4RnoE/cdGLOo8yTzrPIk8qTxJPGs8CzwrO8M74zuFO45bM9D"
        "ex7ZcdgzuJ6B9Tyq25bYmGOc6sYVNo6aqrppgY2jpapuXF/jnxKq6aov7K4xVwjVFzfXJHVY1RP31uhVgTf71ho1"
        "SE2OqJ64ssZYtW2sMcc41Y37ahw1VXXTthpHS1XduKvGPyVU01VfWFRjrhCqL66pSeqwqicuqdGrAm/2FTVqkJoc"
        "UT1xP42xaltPY45xqhuX0zhqquqm1TSOlqq6cTGNf0qopqu+sJXGXCFUX9xJk9RhVU/cSFPK1RBvrmpMdTVITR5X"
        "3ZxMUt1Yte2iMccMqvtb0JPNqu5ozaruaM2q7p8RemMR1f1zQrWA6uZKVHVPRZopQXWg6bctcfuMqwqonrJ7RkxK"
        "Y+dQ3bZ4xhzjVDeunXHUVNVNS2ccLVV148oZ/5RQTVd9Yd+MuUKovrhtJqnDqp64a0avCrzZN82oQWpyRPXENTPG"
        "qm3LjDnGqW7cMeOoqaqbNsw4Wqrqxv0y/imhmq76wnIZc4VQfXG1TFKHVT1xsYxeFXizr5VRg9TkiOqJO2WWqv35"
        "fX/YT8PD8x+4Ao5/+XTef+q7Q8rfenr6Be/mNog="
    ),
}


@pytest.fixture
def local_checkpoint(tmp_path, monkeypatch):
    import base64
    import json
    import zlib

    root = tmp_path / "models"
    snapshot = (
        root / "hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
    )
    snapshot.mkdir(parents=True)
    for name, packed in _ORIGINAL_METADATA_ZLIB.items():
        (snapshot / name).write_bytes(zlib.decompress(base64.b64decode(packed)))
    index = json.loads((snapshot / "model.safetensors.index.json").read_bytes())
    # Empty files establish layout only; these fixtures never claim weight validity.
    for name in set(index["weight_map"].values()):
        (snapshot / name).touch()
    alias = root / "openai/gpt-oss-20b"
    alias.parent.mkdir()
    alias.symlink_to(snapshot, target_is_directory=True)
    monkeypatch.setenv("ROUTER_LOCAL_MODEL_PATH", str(root))
    patch = SCRIPT.with_name("patches") / "sglang-0.5.21-gptoss-gfx942.patch"
    added = "\n".join(
        line[1:]
        for line in patch.read_text().splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    helpers = added[
        added.index("_GPTOSS_REVISION =") : added.index("def _validate_gfx942_gptoss_bf16_runtime")
    ]
    import json as json_module

    namespace = dict(
        Path=Path,
        os=__import__("os"),
        hashlib=hashlib,
        json=json_module,
        _gfx942_gptoss_bf16_requested=lambda: True,
        _gfx942_gptoss_bf16_available=lambda: True,
    )
    exec(compile(helpers, str(patch) + ":actual-added-functions", "exec"), namespace)
    return root, snapshot, alias, namespace


def test_original_local_snapshot_alias_and_copy(local_checkpoint):
    import shutil
    from types import SimpleNamespace

    root, snapshot, alias, ns = local_checkpoint
    validate = ns["_validate_gfx942_gptoss_bf16_model"]
    config = SimpleNamespace(model_type="gpt_oss")
    for path in (snapshot, alias, "openai/gpt-oss-20b"):
        validate(path, config, "mxfp4")
    alias.unlink()
    shutil.copytree(snapshot, alias)
    validate(alias, config, "mxfp4")
    with pytest.raises(ValueError, match="MXFP4"):
        validate(alias, SimpleNamespace(model_type="foreign"), "mxfp4")
    with pytest.raises(ValueError, match="MXFP4"):
        validate(alias, config, "foreign")


@pytest.mark.parametrize(
    "bad",
    [
        "foreign_repository",
        "wrong_revision",
        "escape",
        "wrong_config",
        "wrong_version",
        "wrong_index",
        "shard_escape",
        "missing_shard",
    ],
)
def test_local_identity_rejects_foreign_or_escaping_checkpoint(local_checkpoint, bad):
    import json
    import shutil

    root, snapshot, alias, ns = local_checkpoint
    if bad in ("foreign_repository", "wrong_revision", "escape"):
        alias.unlink()
        target = {
            "foreign_repository": root
            / "hub/models--foreign--gpt-oss-20b/snapshots"
            / snapshot.name,
            "wrong_revision": snapshot.parent / ("0" * 40),
            "escape": root.parent / "outside",
        }[bad]
        shutil.copytree(snapshot, target)
        alias.symlink_to(target, target_is_directory=True)
    elif bad in ("wrong_config", "wrong_version"):
        path = snapshot / "config.json"
        config = json.loads(path.read_bytes())
        config["model_type" if bad == "wrong_config" else "transformers_version"] = "foreign"
        path.write_text(json.dumps(config))
    elif bad == "wrong_index":
        (snapshot / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    else:
        shard = next(snapshot.glob("*.safetensors"))
        shard.unlink()
        if bad == "shard_escape":
            target = root.parent / "foreign.safetensors"
            target.touch()
            shard.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        ns["_validate_gptoss_local_checkpoint"](alias)


def test_snapshot_metadata_blob_links_are_supported(local_checkpoint):
    root, snapshot, alias, ns = local_checkpoint
    blobs = snapshot.parents[1] / "blobs"
    blobs.mkdir()
    for source in list(snapshot.iterdir()):
        target = blobs / hashlib.sha256(source.name.encode()).hexdigest()
        source.rename(target)
        source.symlink_to(target)
    ns["_validate_gptoss_local_checkpoint"](alias)
    with pytest.raises(ValueError):
        ns["_validate_gptoss_local_checkpoint"](root / "unrelated/gpt-oss-20b")
