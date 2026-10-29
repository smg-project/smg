"""Copy only Root-prestaged retained bytes; no build, model, Docker or GPU call."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat


def need(value, message):
    if not value:
        raise RuntimeError(message)


def identity(info):
    return [info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def pin(path):
    with Path(path).open('rb') as source:
        return {'bytes': Path(path).stat().st_size,
                'sha256': hashlib.file_digest(source, 'sha256').hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--cleanup', action='store_true')
    parser.add_argument('--nonce', required=True)
    parser.add_argument('--profile', choices=('SG0', 'Symphony'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.execute or os.environ.get('SMG_PRIVATE_SGLANG_EXECUTE') != 'true':
        return 78
    need(re.fullmatch('[a-f0-9]{32}', args.nonce), 'fresh Root transport nonce')
    need((os.getuid(), os.geteuid(), os.getgid(), os.getegid()) == (995, 995, 985, 985),
         'actual host relay UID995:GID985, not image default user')
    root = Path('/tmp') / ('smg-ci-byte-relay-' + args.nonce)
    run = os.environ['GITHUB_RUN_ID']
    need(run.isdigit() and os.environ['GITHUB_RUN_ATTEMPT'] == '1', 'one new run attempt')
    expected_out = Path(os.environ['RUNNER_TEMP']) / ('smg-byte-relay-' + run + '-1')
    need(args.out == expected_out and not args.out.is_symlink(), 'one own relay output')
    config = json.loads((Path(__file__).with_name('route-inputs.json')).read_bytes())
    names = set(config['staging_files'])
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        held = os.fstat(fd)
        need(held.st_uid == 995 and held.st_gid == 985 and stat.S_IMODE(held.st_mode) == 0o700,
             'actual private nonce stage owner and mode')
        need(set(os.listdir(fd)) == names, 'exact explicit Root stage files')
        if args.cleanup:
            receipt = json.loads((args.out / 'relay-receipt.json').read_bytes())
            need(receipt['nonce'] == args.nonce and receipt['stage_identity'] == identity(held),
                 'same own stage identity')
            for name in sorted(names):
                item = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
                try:
                    need(identity(os.fstat(item)) == receipt['files'][name]['identity'], 'same explicit cleanup leaf')
                finally:
                    os.close(item)
                os.unlink(name, dir_fd=fd)
            need(not os.listdir(fd) and identity(os.stat(root, follow_symlinks=False))[:6] == identity(held)[:6],
                 'same now-empty nonce directory')
        else:
            need(not args.out.exists(), 'new relay output; no overwrite/replay')
            args.out.mkdir(mode=0o700)
            files = {}
            for name in sorted(names):
                need('/' not in name and name not in ('.', '..'), 'explicit single-leaf input')
                item = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
                try:
                    before = os.fstat(item)
                    need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid == 995 and
                         stat.S_IMODE(before.st_mode) == 0o444, 'held immutable regular owned input')
                    cap = 80 * 1024**2 if name.endswith('.tar') else 2 * 1024**2
                    need(before.st_size <= cap, 'finite input bound')
                    h, length = hashlib.sha256(), 0
                    with os.fdopen(os.dup(item), 'rb') as source, (args.out / name).open('xb') as target:
                        while chunk := source.read(1024**2):
                            target.write(chunk)
                            h.update(chunk)
                            length += len(chunk)
                    need(length == before.st_size and identity(os.fstat(item)) == identity(before) and
                         identity(os.stat(name, dir_fd=fd, follow_symlinks=False)) == identity(before),
                         'complete held copy with stable pre/post identity')
                    expected = config['staging_files'][name]
                    observed = {'bytes': length, 'sha256': h.hexdigest()}
                    if expected is not None:
                        need(observed == expected, 'reviewed retained byte identity')
                    files[name] = {**observed, 'identity': identity(before)}
                finally:
                    os.close(item)
            grant = json.loads((args.out / 'Root-start-authority.json').read_bytes())
            need(grant['authorized'] is True and grant['nonce'] == args.nonce and
                 grant['profile'] == args.profile and grant['source_commit'] == os.environ['GITHUB_SHA'] and
                 grant['scope'] == 'Root_controlled_first_SG0_Symphony_full231_GHA' and
                 socket.gethostname() in grant['eligible_hostnames'], 'actual Root staging/selected host authority')
            need(files['component-proof.json']['sha256'] == grant['credential_component_proof_sha256'] and
                 grant['credential_component_verified'] is True, 'Root verified real credential component proof')
            need(files['root-chown-component-proof.json']['sha256'] == grant['root_chown_component_proof_sha256'] and
                 grant['root_chown_component_verified'] is True, 'Root verified actual root CHOWN-only component proof')
            receipt = {'kind': 'new-retained-byte-transport-not-a-new-producer', 'nonce': args.nonce,
                       'run_id': run, 'attempt': 1, 'profile': args.profile,
                       'host': socket.gethostname(), 'stage_identity': identity(held), 'files': files,
                       'no_build_or_GPU': True, 'old_four_artifact_policy_passed': False}
            (args.out / 'relay-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    finally:
        os.close(fd)
    if args.cleanup:
        os.rmdir(root)
        need(not root.exists() and not root.is_symlink(), 'own stage absence')
        (args.out / 'relay-cleanup.json').write_text(json.dumps({'nonce': args.nonce, 'stage_absent': True}) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
