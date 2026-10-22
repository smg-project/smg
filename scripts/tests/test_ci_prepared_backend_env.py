"""Exercise setup/install decisions without GPUs, downloads or system changes."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class PreparedBackendEnvTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scripts = self.root / "scripts"
        self.scripts.mkdir()
        for script in REPO.joinpath("scripts").glob("ci_*.sh"):
            shutil.copy(script, self.scripts)
        self.job = self.root / "job"
        self.job.mkdir()
        self.tools = self.root / "bin"
        self.tools.mkdir()
        self.packages = self.root / "packages"
        self.packages.mkdir()
        self.headers = self.root / "headers"
        self.headers.mkdir()
        self.headers.joinpath("Python.h").touch()
        self.env = os.environ | {
            "PATH": f"{self.tools}:{os.environ['PATH']}",
            "PYTHONPATH": str(self.packages),
            "SMG_PREINSTALLED_ROOT": str(self.root / "prepared"),
            "CI_PYTHON_VERSION": f"{sys.version_info.major}.{sys.version_info.minor}",
            "CI_APT_ROOT": str(self.root / "apt"),
            "GITHUB_PATH": str(self.root / "github.path"),
            "GITHUB_ENV": str(self.root / "github.env"),
            "TEST_PACKAGES": str(self.packages),
            "TEST_HEADERS": str(self.headers),
            "TEST_LOG": str(self.root / "commands.log"),
            "TEST_UID": "1000",
            "CUDA_HOME": str(self.root / "cuda"),
        }
        for name in ("uv", "sudo", "apt-get", "curl"):
            self.write_tool(
                name,
                """#!/bin/bash
echo "$(basename "$0") $*" >> "$TEST_LOG"
if [ "$(basename "$0")" = uv ] && [ -n "${UV_CUDA_DRIVER_VERSION:-}" ]; then
    echo "UV_CUDA_DRIVER_VERSION=$UV_CUDA_DRIVER_VERSION" >> "$TEST_LOG"
fi
if [ "$(basename "$0")" = sudo ] && [ "$TEST_UID" = 0 ]; then echo 'root is not in the sudoers file' >&2; exit 1; fi
if [ "$(basename "$0")" = apt-get ]; then touch "$TEST_HEADERS/Python.h"; fi
if [ "$1" = --version ]; then echo 'uv 0.12.5'; fi
if [ "$*" = 'pip show flashinfer-python' ]; then echo 'Version: 0.7.0'; fi
""",
            )
        self.write_tool("id", '#!/bin/sh\necho "$TEST_UID"\n')
        cuda_bin = Path(self.env["CUDA_HOME"]) / "bin"
        cuda_bin.mkdir(parents=True)
        cuda_bin.joinpath("nvcc").write_text(
            "#!/bin/sh\necho 'Cuda compilation tools, release 13.0'\n"
        )
        cuda_bin.joinpath("nvcc").chmod(0o755)
        self.write_tool("python3", self.python_wrapper())
        for name, version in {
            "vllm": "0.31.0",
            "sglang": "0.5.21",
            "torch": "2.13.0+cu132",
            "flashinfer-python": "0.7.0",
            "flashinfer-jit-cache": "0.7.0",
            "nixl": "1.3.0",
            "nixl-cu13": "1.3.0",
            "mooncake-transfer-engine-cuda13": "0.3.11.post1",
        }.items():
            self.metadata(name, version)
        for name in ("vllm", "sglang", "torchcodec"):
            self.packages.joinpath(f"{name}.py").touch()
        self.write_torch()
        self.packages.joinpath("flashinfer.py").write_text("__version__ = '0.7.0'\n")
        for name in ("nixl", "mooncake"):
            self.packages.joinpath(name).mkdir()
            self.packages.joinpath(name, "__init__.py").touch()
        self.packages.joinpath("nixl/_api.py").write_text("nixl_agent = object\n")
        self.packages.joinpath("nixl/_bindings.py").write_text(
            "nixlRemoteDisconnectError = Exception\n"
        )
        self.packages.joinpath("mooncake/engine.py").write_text(
            "import os\n"
            "if os.environ.get('TEST_DRIVER_ABSENT') == '1':\n"
            "    raise ImportError('libcuda.so.1: cannot open shared object file')\n"
            "TransferEngine = object\n"
        )

    def write_tool(self, name, source):
        path = self.tools / name
        path.write_text(source)
        path.chmod(0o755)

    def write_torch(self, cuda_version="13.2"):
        self.packages.joinpath("torch.py").write_text(
            "import os\n"
            f"class version: cuda = {cuda_version!r}\n"
            "class cuda:\n"
            "    @staticmethod\n"
            "    def is_available(): return os.environ.get('TEST_CUDA_AVAILABLE', '1') == '1'\n"
            "    @staticmethod\n"
            "    def synchronize(): pass\n"
            "def empty(*args, **kwargs):\n"
            "    if os.environ.get('TEST_CUDA_ALLOCATION_FAIL') == '1':\n"
            "        raise RuntimeError('CUDA allocation failed')\n"
            "    return object()\n"
        )

    def python_wrapper(self):
        return f"""#!{sys.executable}
import os, pathlib, shutil, sys
if sys.argv[1:3] == ['-m', 'venv']:
    failed = pathlib.Path(os.environ['TEST_LOG'] + '.venv-failed')
    if os.environ.get('TEST_FAIL_VENV') == '1' and not failed.exists():
        failed.touch(); sys.exit(1)
    target = pathlib.Path(sys.argv[3]).resolve()
    target.joinpath('bin').mkdir(parents=True, exist_ok=True)
    for name in ('python', 'python3'):
        shutil.copy(sys.argv[0], target / 'bin' / name)
    target.joinpath('bin/activate').write_text(
        'export VIRTUAL_ENV=' + str(target) + '\\nexport PATH="' + str(target / 'bin') + ':$PATH"\\n')
    sys.exit(0)
if sys.argv[1:3] == ['-m', 'pip']:
    print('pip (test fixture)')
    sys.exit(0)
if len(sys.argv) > 2 and sys.argv[1] == '-c' and 'sysconfig' in sys.argv[2]:
    print(os.environ['TEST_PACKAGES'] if 'platlib' in sys.argv[2] else os.environ['TEST_HEADERS'])
    sys.exit(0)
os.execv({sys.executable!r}, [{sys.executable!r}] + sys.argv[1:])
"""

    def metadata(self, name, version):
        folder = self.packages / (name.replace("-", "_") + ".dist-info")
        folder.mkdir(exist_ok=True)
        folder.joinpath("METADATA").write_text(f"Name: {name}\nVersion: {version}\n")

    def run_script(self, name, **env):
        return subprocess.run(
            ["bash", str(self.scripts / name)],
            cwd=self.job,
            env=self.env | env,
            text=True,
            capture_output=True,
            check=True,
        )

    def make_prepared(self, backend):
        venv = Path(self.env["SMG_PREINSTALLED_ROOT"]) / backend / ".venv"
        subprocess.run(
            [str(self.tools / "python3"), "-m", "venv", str(venv)], env=self.env, check=True
        )
        subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; smg_mark_prepared_env "$2"',
                "test",
                str(self.scripts / "ci_prepared_backend_env.sh"),
                backend,
            ],
            env=self.env | {"VIRTUAL_ENV": str(venv)},
            text=True,
            capture_output=True,
            check=True,
        )
        return venv.resolve()

    def commands(self):
        path = Path(self.env["TEST_LOG"])
        return path.read_text() if path.exists() else ""

    def test_valid_backend_is_adopted_and_only_pr_packages_are_installed(self):
        for backend in ("vllm", "sglang"):
            with self.subTest(backend=backend):
                if backend == "sglang":
                    self.metadata("torch", "2.13.0")
                    self.write_torch("13.0")
                venv = self.make_prepared(backend)
                self.run_script("ci_setup_python_venv.sh", SMG_CI_BACKEND=backend)
                self.assertEqual(self.job.joinpath(".venv").resolve(), venv)
                result = self.run_script(
                    f"ci_install_{backend}.sh",
                    E2E_KV_BACKEND="nixl",
                    E2E_VLLM_EXTRA_KV_BACKENDS="mooncake",
                )
                self.assertIn("Using prepared", result.stdout)
                installs = [line for line in self.commands().splitlines() if "pip install" in line]
                self.assertTrue(installs)
                self.assertTrue(all("pip install -e " in line for line in installs), installs)
                self.assertNotIn("apt-get", self.commands())

    def test_changed_recipe_falls_back_to_fresh_venv_and_backend_install(self):
        venv = self.make_prepared("vllm")
        with self.scripts.joinpath("ci_install_vllm.sh").open("a") as f:
            f.write("\n# changed dependency recipe\n")
        result = self.run_script("ci_setup_python_venv.sh", SMG_CI_BACKEND="vllm")
        self.assertNotEqual(self.job.joinpath(".venv").resolve(), venv)
        self.assertIn("using a fresh environment", result.stdout)
        self.run_script("ci_install_vllm.sh")
        self.assertIn("vllm==0.31.0 --torch-backend=auto", self.commands())
        self.assertIn("sudo apt-get", self.commands())

    def test_missing_transport_invalidates_reuse(self):
        venv = self.make_prepared("sglang")
        shutil.rmtree(self.packages / "nixl.dist-info")
        self.run_script("ci_setup_python_venv.sh", SMG_CI_BACKEND="sglang")
        self.assertNotEqual(self.job.joinpath(".venv").resolve(), venv)

    def test_other_backend_is_not_adopted(self):
        venv = self.make_prepared("vllm")
        self.run_script("ci_setup_python_venv.sh", SMG_CI_BACKEND="sglang")
        self.assertNotEqual(self.job.joinpath(".venv").resolve(), venv)
        self.assertFalse(self.job.joinpath(".venv").is_symlink())

    def test_wrong_python_interpreter_is_not_adopted(self):
        venv = self.make_prepared("vllm")
        wrapper = self.python_wrapper().replace(
            "import os, pathlib, shutil, sys\n",
            "import os, pathlib, shutil, sys\n"
            "if len(sys.argv) > 2 and 'sys.version_info.major' in sys.argv[2]:\n"
            "    print('0.0'); sys.exit(0)\n",
        )
        venv.joinpath("bin/python").write_text(wrapper)
        self.run_script("ci_setup_python_venv.sh", SMG_CI_BACKEND="vllm")
        self.assertNotEqual(self.job.joinpath(".venv").resolve(), venv)

    def test_changed_native_package_invalidates_reuse(self):
        venv = self.make_prepared("vllm")
        self.metadata("torch", "2.14.0+cu132")
        self.run_script("ci_setup_python_venv.sh", SMG_CI_BACKEND="vllm")
        self.assertNotEqual(self.job.joinpath(".venv").resolve(), venv)

    def test_unusable_cuda_rejects_prepared_environment(self):
        for env in ({"TEST_CUDA_AVAILABLE": "0"}, {"TEST_CUDA_ALLOCATION_FAIL": "1"}):
            with self.subTest(env=env):
                venv = self.make_prepared("vllm")
                result = self.run_script("ci_setup_python_venv.sh", SMG_CI_BACKEND="vllm", **env)
                self.assertNotEqual(self.job.joinpath(".venv").resolve(), venv)
                self.assertIn("using a fresh environment", result.stdout)

    def test_cpu_build_can_reuse_cuda_wheels_without_a_gpu(self):
        venv = self.make_prepared("vllm")
        self.run_script(
            "ci_setup_python_venv.sh",
            SMG_CI_BACKEND="vllm",
            SMG_BUILD_PREPARED_ENV="1",
            TEST_CUDA_AVAILABLE="0",
            TEST_CUDA_ALLOCATION_FAIL="1",
        )
        self.assertEqual(self.job.joinpath(".venv").resolve(), venv)

    def test_fresh_environment_keeps_legacy_install_with_unavailable_cuda(self):
        self.run_script("ci_setup_python_venv.sh", SMG_CI_BACKEND="vllm", TEST_CUDA_AVAILABLE="0")
        self.run_script("ci_install_vllm.sh", TEST_CUDA_AVAILABLE="0")
        self.assertIn("vllm==0.31.0 --torch-backend=auto", self.commands())

    def test_cpu_build_requires_driver_profile_and_never_bakes_pr_packages(self):
        result = subprocess.run(
            ["bash", str(self.scripts / "ci_install_vllm.sh")],
            cwd=self.job,
            env=self.env | {"SMG_BUILD_PREPARED_ENV": "1"},
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("set UV_CUDA_DRIVER_VERSION", result.stderr)
        self.run_script("ci_setup_python_venv.sh", SMG_BUILD_PREPARED_ENV="1")
        self.run_script(
            "ci_install_vllm.sh", SMG_BUILD_PREPARED_ENV="1", UV_CUDA_DRIVER_VERSION="580"
        )
        self.assertIn("https://astral.sh/uv/0.12.5/install.sh", self.commands())
        self.assertIn("--torch-backend=auto", self.commands())
        self.assertIn("UV_CUDA_DRIVER_VERSION=580", self.commands())
        self.assertIn("mooncake-transfer-engine-cuda13==0.3.11.post1", self.commands())
        self.assertNotIn("pip install -e ", self.commands())
        self.assertTrue(self.job.joinpath(".venv/.smg-ci-recipe").is_file())

    def test_sglang_cpu_build_preserves_pypi_recipe_and_bakes_both_transports(self):
        self.metadata("torch", "2.13.0")
        self.write_torch("13.0")
        self.run_script("ci_setup_python_venv.sh")
        self.run_script("ci_install_sglang.sh", SMG_BUILD_PREPARED_ENV="1")
        self.assertIn("sglang[all]==0.5.21", self.commands())
        self.assertNotIn("--torch-backend", self.commands())
        self.assertIn("nixl==1.3.0 nixl-cu13==1.3.0", self.commands())
        self.assertIn("mooncake-transfer-engine-cuda13==0.3.12.post1", self.commands())
        self.assertNotIn("pip install -e ", self.commands())

    def test_cpu_torch_cannot_be_marked_as_prepared(self):
        self.write_torch(None)
        with self.assertRaises(subprocess.CalledProcessError) as error:
            self.make_prepared("vllm")
        self.assertIn("require CUDA-enabled PyTorch", error.exception.stderr)
        venv = Path(self.env["SMG_PREINSTALLED_ROOT"]) / "vllm/.venv"
        self.assertFalse(venv.joinpath(".smg-ci-recipe").exists())

    def test_cpu_build_defers_mooncake_driver_probe_but_gpu_job_requires_it(self):
        for backend in ("vllm", "sglang"):
            with self.subTest(backend=backend):
                self.run_script("ci_setup_python_venv.sh")
                result = self.run_script(
                    f"ci_install_{backend}.sh",
                    SMG_BUILD_PREPARED_ENV="1",
                    UV_CUDA_DRIVER_VERSION="580",
                    TEST_DRIVER_ABSENT="1",
                )
                self.assertIn("Deferring Mooncake driver import", result.stdout)
                self.assertTrue(self.job.joinpath(".venv/.smg-ci-recipe").is_file())
                with self.assertRaises(subprocess.CalledProcessError) as error:
                    self.run_script(
                        f"ci_install_{backend}.sh",
                        E2E_KV_BACKEND="mooncake",
                        TEST_DRIVER_ABSENT="1",
                    )
                self.assertIn("libcuda.so.1", error.exception.stderr)

    def test_sglang_mooncake_probe_requires_explicit_transport(self):
        for transport in (
            {"E2E_SGLANG_TRANSFER_BACKEND": "mooncake"},
            {"E2E_KV_BACKEND": "mooncake"},
            {"E2E_KV_BACKEND": "mooncake", "E2E_SGLANG_TRANSFER_BACKEND": "nixl"},
        ):
            with self.subTest(transport=transport):
                self.run_script("ci_setup_python_venv.sh")
                with self.assertRaises(subprocess.CalledProcessError) as error:
                    self.run_script("ci_install_sglang.sh", TEST_DRIVER_ABSENT="1", **transport)
                self.assertIn("libcuda.so.1", error.exception.stderr)
        for transport in (
            {},
            {"E2E_SGLANG_TRANSFER_BACKEND": "nixl"},
            {"E2E_KV_BACKEND": "nixl", "E2E_SGLANG_TRANSFER_BACKEND": "mooncake"},
        ):
            with self.subTest(transport=transport):
                self.run_script("ci_setup_python_venv.sh")
                self.run_script("ci_install_sglang.sh", TEST_DRIVER_ABSENT="1", **transport)

    def test_root_with_sudo_present_installs_backends_without_sudo(self):
        self.run_script("ci_setup_python_venv.sh", TEST_UID="0")
        for backend in ("vllm", "sglang"):
            with self.subTest(backend=backend):
                self.run_script(f"ci_install_{backend}.sh", TEST_UID="0", E2E_KV_BACKEND="mooncake")
        self.assertIn("apt-get install", self.commands())
        self.assertNotIn("sudo ", self.commands())

    def test_root_with_sudo_present_repairs_venv_without_sudo(self):
        self.run_script("ci_setup_python_venv.sh", TEST_UID="0", TEST_FAIL_VENV="1")
        self.assertIn("apt-get install -y python3-pip python3-venv", self.commands())
        self.assertNotIn("sudo ", self.commands())

    def test_root_with_sudo_present_installs_headers_without_sudo(self):
        self.headers.joinpath("Python.h").unlink()
        self.run_script("ci_ensure_python_headers.sh", TEST_UID="0")
        self.assertIn("apt-get install", self.commands())
        self.assertNotIn("sudo ", self.commands())

    def rocm_cairo_tools(self):
        self.write_tool(
            "python3",
            f"#!{sys.executable}\n"
            + """
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
with open(os.environ['TEST_LOG'],'a') as output:
    output.write('PYTHON_ARGV='+json.dumps(args)+'\\n')
if args[:2]==['-m','venv']:
    folder=Path('.venv/bin');folder.mkdir(parents=True,exist_ok=True)
    (folder/'activate').write_text('export VIRTUAL_ENV="$PWD/.venv"\\n')
if args[:1]==['-c']:
    if 'version("vllm")' in args[-1]:print('0.31.0+rocm723 1.78.0')
    elif 'version("pycairo")' in args[-1]:sys.exit(0 if os.environ.get('TEST_CAIRO_PRESENT')=='1' else 1)
    elif 'import cairo' in args[-1] and os.environ.get('TEST_CAIRO_NATIVE_FAIL')=='1':sys.exit(7)
""",
        )
        self.write_tool("pip", '#!/bin/sh\necho "pip $*" >> "$TEST_LOG"\n')
        self.write_tool("pkg-config", '#!/bin/sh\nexit "${TEST_CAIRO_PKG_FAIL:-0}"\n')

    def test_rocm_cairo_skips_usable_packages_and_requires_native_import_after_build(self):
        self.rocm_cairo_tools()
        for present, native_fail in ((True, False), (False, False), (False, True)):
            with self.subTest(present=present, native_fail=native_fail):
                Path(self.env["TEST_LOG"]).unlink(missing_ok=True)
                kwargs = dict(
                    TEST_CAIRO_PRESENT=str(int(present)),
                    TEST_CAIRO_NATIVE_FAIL=str(int(native_fail)),
                )
                if native_fail:
                    with self.assertRaises(subprocess.CalledProcessError) as error:
                        self.run_script("ci_setup_vllm_rocm_image.sh", **kwargs)
                    self.assertEqual(error.exception.returncode, 7)
                else:
                    self.run_script("ci_setup_vllm_rocm_image.sh", **kwargs)
                commands = [
                    json.loads(line.removeprefix("PYTHON_ARGV="))
                    for line in self.commands().splitlines()
                    if line.startswith("PYTHON_ARGV=")
                ]
                installs = [args for args in commands if args[:3] == ["-m", "pip", "install"]]
                self.assertEqual(len(installs), int(not present))
                if installs:
                    self.assertIn("--no-deps", installs[0])
                    self.assertIn("--no-cache-dir", installs[0])
                    self.assertIn("--no-binary=pycairo", installs[0])
                    self.assertEqual(
                        installs[0][-1].split("#sha256=")[1],
                        "3e69fff74fe64f5ba2dfa31f67c6bdf26413342574047437d2ac520d35e9a489",
                    )
                self.assertNotIn("apt-get", self.commands())
                if native_fail:
                    self.assertNotIn("pip install vllm", self.commands())
                else:
                    self.assertIn("pip install vllm==0.31.0+rocm723", self.commands())
                    self.assertNotIn("pip install torch", self.commands())

    def test_rocm_cairo_cannot_install_system_packages_outside_root_actions(self):
        self.rocm_cairo_tools()
        for actions, uid in (("false", "0"), ("true", "1000")):
            with self.subTest(actions=actions, uid=uid):
                Path(self.env["TEST_LOG"]).unlink(missing_ok=True)
                with self.assertRaises(subprocess.CalledProcessError) as error:
                    self.run_script(
                        "ci_setup_vllm_rocm_image.sh",
                        TEST_CAIRO_PKG_FAIL="1",
                        GITHUB_ACTIONS=actions,
                        TEST_UID=uid,
                    )
                self.assertIn("root Actions container", error.exception.stderr)
                self.assertNotIn("apt-get", self.commands())
                self.assertNotIn("pycairo-1.29.2.tar.gz", self.commands())


if __name__ == "__main__":
    unittest.main()
