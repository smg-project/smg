"""Restore this job's checkout ownership; never inspect or clean another runner."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import stat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute or os.environ.get('SMG_PRIVATE_SGLANG_EXECUTE') != 'true':
        return 78
    assert os.getuid() == os.geteuid() == 0
    assert re.fullmatch('[a-f0-9]{64}', os.environ['ACTUAL_JOB_CONTAINER_ID'])
    run = os.environ['GITHUB_RUN_ID']
    assert run.isdigit() and os.environ['GITHUB_RUN_ATTEMPT'] == '1'
    workspace = Path(os.environ['GITHUB_WORKSPACE'])
    assert workspace.is_absolute() and workspace.is_relative_to('/__w') and workspace != Path('/__w')
    assert workspace.is_dir() and not workspace.is_symlink()
    seen = 0
    for parent, dirs, files in os.walk(workspace, followlinks=False):
        for name in list(dirs):
            if (Path(parent) / name).is_symlink():
                dirs.remove(name)
        for path in [Path(parent), *[Path(parent) / name for name in files]]:
            seen += 1
            assert seen <= 16384 and (path == workspace or path.parent.resolve().is_relative_to(workspace))
            assert stat.S_ISREG(path.lstat().st_mode) or path.is_dir() or path.is_symlink()
            os.chown(path, 995, 985, follow_symlinks=False)
    temporary = Path(os.environ['RUNNER_TEMP'])
    transport = temporary / ('sglang-full231-' + run + '-1')
    assert temporary.is_absolute() and temporary.is_relative_to('/__w')
    assert not transport.is_symlink()
    if transport.exists():
        assert transport.is_dir()
        shutil.rmtree(transport)
    assert not transport.exists()
    receipt = {'scope': 'this actual job checkout and exact run transport only',
               'CID': os.environ['ACTUAL_JOB_CONTAINER_ID'], 'run_id': run,
               'checkout_entries_restored': seen, 'transport_absent': True,
               'runner_post_job_container_removal_verified': False,
               'global_completed_hook_clean_temp_is_strict': False}
    target = temporary / ('sglang-full231-cleanup-' + run + '.json')
    with target.open('x') as out:
        json.dump(receipt, out)
    os.chown(target, 995, 985)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
