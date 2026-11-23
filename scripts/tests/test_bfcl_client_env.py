"""Exercise client/engine interpreter separation without downloads or GPU imports."""

import json
import os
import subprocess
import sys
import textwrap
import venv
from pathlib import Path

import pytest

SETUP = Path(__file__).resolve().parents[1] / "bfcl/setup_client_env.sh"


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def environment(tmp_path, *, broken_engine=False, fail_install=False):
    engine = tmp_path / "engine"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(engine)
    python = engine / "bin/python"
    site = Path(
        subprocess.check_output(
            [str(python), "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))'],
            text=True,
        ).strip()
    )
    version = "1.26.4" if broken_engine else "2.3.5"
    write(site / "numpy.py", f'__version__ = "{version}"\n')
    write(
        site / f"numpy-{version}.dist-info/METADATA",
        f"Name: numpy\nVersion: {version}\n",
    )
    write(site / "vllm/__init__.py", "")
    write(
        site / "vllm/entrypoints/cli/main.py",
        "def main():\n    import vllm.entrypoints.cli.launch\n",
    )
    write(
        site / "vllm/entrypoints/cli/launch.py",
        'import numpy\nassert numpy.__version__ == "2.3.5", "engine NumPy ABI changed"\n',
    )
    fake_uv = tmp_path / "tools/uv"
    # The fixture installer supplies tiny incompatible packages. The real
    # interpreter isolation and canaries run; no real BFCL or Torch is installed.
    write(
        fake_uv,
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            r"""
            import json, os, subprocess, sys
            from pathlib import Path

            args = sys.argv[1:]
            with Path(os.environ['TEST_UV_TRACE']).open('a') as stream:
                stream.write(json.dumps(args) + '\n')
            if args[0] == 'venv':
                python = args[args.index('--python') + 1]
                subprocess.run([python, '-m', 'venv', '--without-pip', args[-1]], check=True)
            elif args[:2] == ['pip', 'install']:
                if os.environ.get('TEST_FAIL_INSTALL') == '1':
                    raise SystemExit(67)
                python = Path(args[args.index('--python') + 1])
                site = Path(subprocess.check_output(
                    [str(python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'],
                    text=True,
                ).strip())
                (site / 'numpy.py').write_text('__version__ = "1.26.4"\n')
                for name, version in [('numpy', '1.26.4'), ('bfcl-eval', '2026.3.23')]:
                    dist = site / (name.replace('-', '_') + '-' + version + '.dist-info')
                    dist.mkdir()
                    (dist / 'METADATA').write_text('Name: ' + name + '\nVersion: ' + version + '\n')
                (site / 'torch.py').write_text(
                    'from types import SimpleNamespace\nversion = SimpleNamespace(cuda=None, hip=None)\n'
                )
                cli = python.parent / 'bfcl'
                cli.write_text('#!' + str(python) + '\nimport numpy\nprint(numpy.__version__)\n')
                cli.chmod(0o700)
            elif args[:2] != ['pip', 'check']:
                raise SystemExit('unexpected fixture installer command')
            """
        ),
    )
    fake_uv.chmod(0o700)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(
        PATH=f"{engine / 'bin'}:{fake_uv.parent}:/usr/bin:/bin",
        GITHUB_ENV=str(tmp_path / "github-env"),
        TEST_UV_TRACE=str(tmp_path / "uv-trace"),
        TEST_FAIL_INSTALL="1" if fail_install else "0",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return engine, env


def invoke(tmp_path, env):
    return subprocess.run(
        ["bash", str(SETUP), str(tmp_path / "client")],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_client_can_use_numpy_126_without_changing_numpy_2_engine(tmp_path):
    engine, env = environment(tmp_path)
    result = invoke(tmp_path, env)
    assert result.returncode == 0, result.stderr
    published = dict(
        line.split("=", 1) for line in Path(env["GITHUB_ENV"]).read_text().splitlines()
    )
    assert set(published) == {"BFCL_EXECUTABLE", "BFCL_CLIENT_PYTHON"}
    assert subprocess.check_output([published["BFCL_EXECUTABLE"]], text=True).strip() == "1.26.4"
    assert (
        subprocess.check_output(
            [str(engine / "bin/python"), "-c", "import numpy; print(numpy.__version__)"], text=True
        ).strip()
        == "2.3.5"
    )
    config = (tmp_path / "client/pyvenv.cfg").read_text()
    assert "include-system-site-packages = false" in config
    commands = [json.loads(line) for line in Path(env["TEST_UV_TRACE"]).read_text().splitlines()]
    install = next(args for args in commands if args[:2] == ["pip", "install"])
    assert "--torch-backend=cpu" in install
    assert "bfcl-eval==2026.3.23" in install
    assert "--no-deps" not in install
    assert any(args[:2] == ["pip", "check"] for args in commands)
    assert str(tmp_path / "client/bin") not in env["PATH"]


@pytest.mark.parametrize("failure", ["install", "engine-import"])
def test_failed_setup_does_not_publish_a_fallback_client(tmp_path, failure):
    engine, env = environment(
        tmp_path, broken_engine=failure == "engine-import", fail_install=failure == "install"
    )
    # The real failure is reached through main.main()'s later launch import.
    # Merely importing main must remain insufficient to detect this regression.
    subprocess.run(
        [str(engine / "bin/python"), "-c", "import vllm.entrypoints.cli.main"],
        check=True,
    )
    result = invoke(tmp_path, env)
    assert result.returncode != 0
    assert not Path(env["GITHUB_ENV"]).exists()
    expected = "1.26.4" if failure == "engine-import" else "2.3.5"
    assert (
        subprocess.check_output(
            [str(engine / "bin/python"), "-c", "import numpy; print(numpy.__version__)"], text=True
        ).strip()
        == expected
    )


def test_existing_environment_is_preserved_and_rejected(tmp_path):
    _, env = environment(tmp_path)
    marker = tmp_path / "client/keep"
    write(marker, "belongs to an earlier operation\n")
    result = invoke(tmp_path, env)
    assert result.returncode == 2
    assert marker.read_text() == "belongs to an earlier operation\n"
    assert not Path(env["TEST_UV_TRACE"]).exists()
    assert not Path(env["GITHUB_ENV"]).exists()
