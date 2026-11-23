"""Exercise client/engine interpreter separation without downloads or GPU imports."""

import json
import os
import subprocess
import sys
import textwrap
import venv
from pathlib import Path

import pytest
import yaml

SETUP = Path(__file__).resolve().parents[1] / "bfcl/setup_client_env.sh"


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def environment(
    tmp_path,
    *,
    broken_engine=False,
    fail_install=False,
    transient_failures=0,
    change_engine=False,
    missing_bfcl=False,
):
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
                attempts = sum(
                    json.loads(line)[:2] == ['pip', 'install']
                    for line in Path(os.environ['TEST_UV_TRACE']).read_text().splitlines()
                )
                if (os.environ.get('TEST_FAIL_INSTALL') == '1' or
                        attempts <= int(os.environ['TEST_TRANSIENT_INSTALL_FAILURES'])):
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
                if os.environ.get('TEST_MISSING_BFCL') != '1':
                    cli = python.parent / 'bfcl'
                    cli.write_text('#!' + str(python) + '\nimport numpy\nprint(numpy.__version__)\n')
                    cli.chmod(0o700)
                if os.environ.get('TEST_CHANGE_ENGINE') == '1':
                    engine_site = Path(os.environ['TEST_ENGINE_SITE'])
                    (engine_site / 'numpy.py').write_text('__version__ = "1.26.4"\n')
                    metadata = next(engine_site.glob('numpy-*.dist-info/METADATA'))
                    metadata.write_text('Name: numpy\nVersion: 1.26.4\n')
            elif args[:2] != ['pip', 'check']:
                raise SystemExit('unexpected fixture installer command')
            """
        ),
    )
    fake_uv.chmod(0o700)
    # Exercise the real ci_retry wrapper without waiting its production10/20s.
    # This fixture replaces only the external sleep, and records every delay.
    fake_sleep = fake_uv.with_name("sleep")
    write(fake_sleep, '#!/bin/sh\nprintf \'%s\\n\' "$1" >> "$TEST_SLEEP_TRACE"\n')
    fake_sleep.chmod(0o700)
    write(tmp_path / "github-path", "existing runner path\n")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(
        PATH=f"{engine / 'bin'}:{fake_uv.parent}:/usr/bin:/bin",
        GITHUB_ENV=str(tmp_path / "github-env"),
        GITHUB_PATH=str(tmp_path / "github-path"),
        TEST_SLEEP_TRACE=str(tmp_path / "sleep-trace"),
        TEST_ENGINE_SITE=str(site),
        TEST_TRANSIENT_INSTALL_FAILURES=str(transient_failures),
        TEST_CHANGE_ENGINE="1" if change_engine else "0",
        TEST_MISSING_BFCL="1" if missing_bfcl else "0",
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
    assert Path(env["GITHUB_PATH"]).read_text() == "existing runner path\n"


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
    assert Path(env["GITHUB_PATH"]).read_text() == "existing runner path\n"
    if failure == "install":
        commands = [
            json.loads(line) for line in Path(env["TEST_UV_TRACE"]).read_text().splitlines()
        ]
        assert sum(args[:2] == ["pip", "install"] for args in commands) == 3
        assert Path(env["TEST_SLEEP_TRACE"]).read_text().splitlines() == ["10", "20"]
        assert result.returncode == 67
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


def test_client_install_recovers_only_within_the_existing_bounded_retry(tmp_path):
    _, env = environment(tmp_path, transient_failures=2)
    result = invoke(tmp_path, env)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in Path(env["TEST_UV_TRACE"]).read_text().splitlines()]
    assert sum(args[0] == "venv" for args in commands) == 1
    assert sum(args[:2] == ["pip", "install"] for args in commands) == 3
    assert Path(env["TEST_SLEEP_TRACE"]).read_text().splitlines() == ["10", "20"]
    assert Path(env["GITHUB_ENV"]).exists()
    assert Path(env["GITHUB_PATH"]).read_text() == "existing runner path\n"


def test_changed_engine_numpy_fails_before_publishing_client_paths(tmp_path):
    _, env = environment(tmp_path, change_engine=True)
    result = invoke(tmp_path, env)
    assert result.returncode == 1
    assert "changed engine NumPy from 2.3.5 to 1.26.4" in result.stderr
    assert str(tmp_path / "engine/bin/python") in result.stderr
    assert "BFCL engine launch import canary" not in result.stdout
    assert not Path(env["GITHUB_ENV"]).exists()
    assert Path(env["GITHUB_PATH"]).read_text() == "existing runner path\n"


def test_missing_client_bfcl_is_an_explicit_unpublished_setup_failure(tmp_path):
    _, env = environment(tmp_path, missing_bfcl=True)
    result = invoke(tmp_path, env)
    assert result.returncode == 1
    assert "BFCL client executable is missing or not executable" in result.stderr
    assert str(tmp_path / "client/bin/bfcl") in result.stderr
    assert not Path(env["GITHUB_ENV"]).exists()
    assert Path(env["GITHUB_PATH"]).read_text() == "existing runner path\n"


def registration_script():
    workflow = SETUP.parents[2] / ".github/workflows/nightly-bfcl.yml"
    steps = [
        step["run"]
        for job in yaml.safe_load(workflow.read_text())["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == "Register model in bfcl (no-op if bfcl already ships a handler)"
    ]
    assert len(steps) == 1
    return steps[0]


@pytest.mark.parametrize("vendor", ["amd", "nvidia"])
@pytest.mark.parametrize("already_registered", [False, True])
def test_real_registration_step_preserves_noop_and_refuses_real_failure(
    tmp_path, vendor, already_registered
):
    engine, env = environment(tmp_path)
    if vendor == "nvidia":
        setup = invoke(tmp_path, env)
        assert setup.returncode == 0, setup.stderr
        env.update(line.split("=", 1) for line in Path(env["GITHUB_ENV"]).read_text().splitlines())
        python = env["BFCL_CLIENT_PYTHON"]
    else:
        env.pop("BFCL_CLIENT_PYTHON", None)
        python = str(engine / "bin/python")
    site = Path(
        subprocess.check_output(
            [python, "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True
        ).strip()
    )
    write(site / "bfcl_eval/__init__.py", "")
    write(site / "bfcl_eval/constants/__init__.py", "")
    config = site / "bfcl_eval/constants/model_config.py"
    contents = (
        'MODEL_CONFIG_MAPPING = {"unit/existing-FC": None}\n'
        if already_registered
        else "MODEL_CONFIG_MAPPING = {}\n"
    )
    write(config, contents)
    env.update(GPU_VENDOR=vendor, MODEL="unit/existing")
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", registration_script()],
        cwd=SETUP.parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == (0 if already_registered else 1)
    assert ("already registered" in result.stdout) == already_registered
    if not already_registered:
        assert "anchor not found" in result.stderr
    assert config.read_text() == contents
    assert Path(env["GITHUB_PATH"]).read_text() == "existing runner path\n"


def test_nvidia_registration_requires_the_prepared_client_interpreter(tmp_path):
    _, env = environment(tmp_path)
    env.pop("BFCL_CLIENT_PYTHON", None)
    env.update(GPU_VENDOR="nvidia", MODEL="unit/existing")
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", registration_script()],
        cwd=SETUP.parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "isolated NVIDIA BFCL client Python was not prepared" in result.stderr
