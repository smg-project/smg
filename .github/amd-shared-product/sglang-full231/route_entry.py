"""Finite in-container adapter for the two original full231 profiles."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import tarfile
import traceback
from datetime import datetime, timezone
from urllib.request import Request, urlopen


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(path, cap=96 * 1024**2):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= cap, 'bounded regular file')
        h, size = hashlib.sha256(), 0
        while True:
            chunk = os.read(fd, 1024**2)
            if not chunk:
                break
            size += len(chunk)
            need(size <= cap, 'complete file cap')
            h.update(chunk)
        after = os.fstat(fd)
        stable_fields = ('st_dev', 'st_ino', 'st_mode', 'st_uid', 'st_gid', 'st_nlink',
                         'st_size', 'st_mtime_ns', 'st_ctime_ns')
        need(all(getattr(before, key) == getattr(after, key) for key in stable_fields)
             and size == before.st_size, 'stable full file')
        return {'bytes': size, 'sha256': h.hexdigest()}
    finally:
        os.close(fd)


def save(path, value, cap=64 * 1024):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()
    need(len(raw) <= cap, 'complete JSON cap')
    temporary = path.with_name(path.name + '.new')
    with temporary.open('wb') as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)


def reserve_start(epoch):
    current = datetime.fromtimestamp(epoch, timezone.utc)
    start = current.replace(hour=5, minute=0, second=0, microsecond=0).timestamp()
    need(not start <= epoch < start + 5400, 'Router 05:00-06:30 UTC window')
    return start if epoch < start else start + 86400


def runtime_package(transport, config):
    archive = transport / 'runtime.tar'
    need(pin(archive) == config['staging_files']['runtime.tar'], 'reviewed private QA runtime full bytes')
    package = transport / 'runtime'
    need(not package.exists() and not package.is_symlink(), 'fresh private QA runtime')
    package.mkdir(mode=0o700)
    with tarfile.open(archive, 'r:') as bundle:
        members = bundle.getmembers()
        need(len(members) == len(config['runtime_files']) and
             {member.name for member in members} == set(config['runtime_files']), 'exact private runtime files')
        for member in members:
            need(member.isfile() and '/' not in member.name and member.name not in ('.', '..') and
                 member.size <= 2 * 1024**2, 'bounded regular runtime leaf')
            target = package / member.name
            with bundle.extractfile(member) as source, target.open('xb') as output:
                output.write(source.read(2 * 1024**2 + 1))
            target.chmod(0o444)
            need(pin(target) == config['runtime_files'][member.name], 'actual full runtime leaf SHA')
    return package


def actual_job():
    """One bounded read of this run's complete jobs, never a dispatch."""
    run_id, attempt = os.environ['GITHUB_RUN_ID'], os.environ['GITHUB_RUN_ATTEMPT']
    need(run_id.isdigit() and attempt == '1', 'new run; no allowance replay')
    repository = os.environ['GITHUB_REPOSITORY']
    need(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository), 'repository')
    token = os.environ.pop('GITHUB_TOKEN')
    jobs, total = [], None
    for page in range(1, 5):
        url = ('https://api.github.com/repos/' + repository + '/actions/runs/' + run_id
               + '/attempts/1/jobs?per_page=100&page=' + str(page))
        request = Request(url, headers={'Authorization': 'Bearer ' + token,
                          'Accept': 'application/vnd.github+json',
                          'X-GitHub-Api-Version': '2022-11-28'})
        with urlopen(request, timeout=10) as response:
            raw = response.read(2 * 1024**2 + 1)
            need(response.status == 200 and len(raw) <= 2 * 1024**2, 'actual bounded job GET')
            data = json.loads(raw)
            total = data['total_count'] if total is None else total
            need(data['total_count'] == total, 'stable total job inventory')
            jobs.extend(data['jobs'])
            if not response.headers.get('Link') or 'rel="next"' not in response.headers['Link']:
                break
    need(total == len(jobs) == len({job['id'] for job in jobs}), 'complete actual job pages')
    matches = [job for job in jobs if job['name'] == 'private-amd-sglang-run / run']
    need(len(matches) == 1, 'one actual private GPU job')
    job = matches[0]
    need(job['status'] == 'in_progress' and job['runner_name'] == os.environ['RUNNER_NAME'],
         'actual assigned runner')
    return {key: job[key] for key in ('id', 'name', 'status', 'runner_name', 'started_at', 'head_sha')}


class Adapter:
    def __init__(self, profile, grant, job, package):
        self.profile, self.grant, self.job, self.package = profile, grant, job, package
        self.origin_epoch = datetime.fromisoformat(job['started_at'].replace('Z', '+00:00')).timestamp()
        self.origin = time.monotonic() - (time.time() - self.origin_epoch)
        self.work_stop = self.origin + 1980
        self.cleanup_stop = self.origin + 2100
        self.whole_stop = self.origin + 2160
        self.next_window = reserve_start(self.origin_epoch)
        need(reserve_start(time.time()) == self.next_window, 'queued start and current UTC window')
        need(self.origin_epoch + 2460 < min(self.next_window, grant['occupancy_cutoff_epoch']),
             'whole2160 + buffer300 fits actual hard/Root occupancy cutoff')
        need(grant['submitted_epoch'] <= self.origin_epoch <= grant['queue_expiry_epoch'],
             'actual job assignment inside finite Root queue TTL')
        need(0 <= time.time() - self.origin_epoch < 180, 'normal180 includes container preparation')
        self.nonce = ('smg-sg0plain-' if profile == 'SG0' else 'smg-symphony-') + grant['nonce']
        self.window = 'GHA-full231-' + grant['nonce']
        self.stage = Path('/tmp') / self.nonce
        need(not self.stage.exists() and not self.stage.is_symlink(), 'fresh private container stage')
        self.stage.mkdir(mode=0o700)
        self.evidence = Path('/evidence')
        need(not self.evidence.exists() and not self.evidence.is_symlink(), 'fresh evidence tree')
        self.evidence.mkdir(mode=0o755)
        (self.evidence / 'facts').mkdir(mode=0o755)
        (self.evidence / 'logs').mkdir(mode=0o755)
        self.share_evidence_directories()
        os.environ.update({'MAX_JOBS': '8', 'SMG_SGLANG_ROCM_IMAGE': grant['image'],
            'HOME': '/evidence/home', 'XDG_CACHE_HOME': '/evidence/cache',
            'AITER_JIT_DIR': '/evidence/cache/aiter-jit', 'HF_HOME': '/models',
            'HF_XET_CACHE': '/evidence/cache/hf-xet', 'HF_TOKEN_PATH': '/run/smg-hf/token',
            'PYTHONPATH': '/source/e2e_test', 'PYTHONDONTWRITEBYTECODE': '1',
            'SPT_NOENV': '1', 'E2E_JOB_MARKER': self.nonce,
            'CUDA_VISIBLE_DEVICES': '0', 'HIP_VISIBLE_DEVICES': '0', 'ROCR_VISIBLE_DEVICES': '0',
            'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'})
        self.logs, self.children = [], []
        self.state = {'CID': os.environ['ACTUAL_JOB_CONTAINER_ID'],
                      'hostname': socket.gethostname(), 'nonce': self.nonce,
                      'window_id': self.window, 'actual_GHA_job': job,
                      'image_binding': 'workflow-fixed digest; runtime logs/exact CID postcheck still required'}
        self.result = {'passed': False, 'work_passed': False, 'commands': [],
                       'scope': 'GHA original full231 ' + profile, 'coverage_increment': 0,
                       'cleanup': {}, 'origin_epoch': self.origin_epoch,
                       'origin_monotonic': self.origin, 'whole_seconds': 2160}

    def share_evidence_directories(self):
        # Only these three fresh private directories serve both existing roles.
        # UID995 remains owner; root retains group0 write without DAC_OVERRIDE.
        need(os.getresuid() == (0, 0, 0) and os.getresgid() == (0, 0, 0),
             'root parent identity for private evidence ownership handoff')
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        fds = [os.open('/', flags)]
        bindings = []
        try:
            for name, parent_index in (('evidence', 0), ('facts', 1), ('logs', 1)):
                parent_fd = fds[parent_index]
                fd = os.open(name, flags, dir_fd=parent_fd)
                fds.append(fd)
                held = os.fstat(fd)
                current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                need(stat.S_ISDIR(held.st_mode) and
                     (held.st_dev, held.st_ino) == (current.st_dev, current.st_ino) and
                     (held.st_uid, held.st_gid) in ((0, 0), (995, 985), (995, 0)) and
                     stat.S_IMODE(held.st_mode) in (0o755, 0o775),
                     'fixed nonsymlink evidence directory preimage')
                bindings.append((name, parent_fd, fd, held.st_dev, held.st_ino))
            for name, parent_fd, fd, device, inode in bindings:
                current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                need((current.st_dev, current.st_ino) == (device, inode),
                     'held evidence directory still names the same private object')
                # CHOWN can reclaim the owner; chmod runs only while root owns it.
                os.fchown(fd, 0, 0)
                os.fchmod(fd, 0o775)
                os.fchown(fd, 995, 0)
                after = os.fstat(fd)
                current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                need((after.st_dev, after.st_ino) == (current.st_dev, current.st_ino) ==
                     (device, inode) and stat.S_ISDIR(after.st_mode) and
                     (after.st_uid, after.st_gid, stat.S_IMODE(after.st_mode)) == (995, 0, 0o775),
                     'both original roles can write only their private evidence directories')
        finally:
            for fd in reversed(fds):
                os.close(fd)

    def scope(self, deadline, check_transport=True):
        need(time.monotonic() < deadline and time.time() < self.origin_epoch + deadline - self.origin,
             'one absolute phase deadline')
        need(reserve_start(time.time()) == self.next_window and time.time() < self.next_window - 300,
             'hard UTC release margin')
        need(time.time() < self.grant['occupancy_cutoff_epoch'] - 300, 'Root occupancy release margin')
        need(all(path.stat().st_size <= 32 * 1024**2 for path in self.logs), 'complete stream32MiB cap')

    def command(self, name, argv, seconds=30, deadline=None):
        deadline = self.origin + 180 if deadline is None else deadline
        self.scope(deadline)
        need(argv[0] == 'exec' and argv[1] in ('--user=0:0', '--user=995:985'), 'fixed profile exec')
        user = argv[1]
        rest, extra_env = argv[2:], {}
        if rest[0] == '--env=SGLANG_USE_AITER=1':
            extra_env['SGLANG_USE_AITER'] = '1'
            rest = rest[1:]
        need(rest[0] == self.state['CID'], 'actual job.container.id')
        target = rest[1:]
        need(target[0] == '/opt/venv/bin/python' and all(type(v) is str for v in target),
             'original Python phase argv')
        index = len(self.result['commands'])
        base = self.evidence / 'logs' / ('command-' + str(index))
        stdout, stderr = Path(str(base) + '.stdout'), Path(str(base) + '.stderr')
        raw_stdout = Path(str(base) + '.raw-stdout')
        command = target
        conversion = None
        conversion_kind = None
        if user == '--user=995:985':
            conversion_kind = 'UID995'
            template = (self.package / 'root_to_uid995.py').read_text()
            if target[-1] not in ('/input/full-sg0plain/uid995_plainchat_entry.py',
                                  '/input/full-symphony/uid995_symphony_entry.py'):
                wrapper = self.stage / ('phase-' + str(index) + '.py')
                wrapper.write_text('import os\nos.execv(' + repr(target[0]) + ',' + repr(target) + ')\n')
                wrapper.chmod(0o444)
                tree = ast.parse(template)
                node = next(node for node in tree.body if isinstance(node, ast.Assign) and
                            any(isinstance(value, ast.Name) and value.id == 'ENTRIES' for value in node.targets))
                lines = template.splitlines(keepends=True)
                old = ''.join(lines[node.lineno - 1:node.end_lineno])
                new = 'ENTRIES = ' + repr({self.profile: (str(wrapper),
                       pin(wrapper)['bytes'], pin(wrapper)['sha256'])}) + '\n'
                need(template.count(old) == 1, 'one phase ENTRY descriptor replacement')
                template = template.replace(old, new, 1)
                conversion = {'fixed_phase_argv': target, 'wrapper': pin(wrapper),
                              'ENTRY_metadata_replacement_only': {'old': old, 'new': new}}
            need(template.count('ROOT_SOURCE_PREVIEW_ONLY = True') == 1, 'one converter opt-in')
            active = template.replace('ROOT_SOURCE_PREVIEW_ONLY = True',
                                      'ROOT_SOURCE_PREVIEW_ONLY = False', 1)
            converter = self.stage / ('converter-' + str(index) + '.py')
            converter.write_text(active)
            converter.chmod(0o444)
            command = ['/opt/venv/bin/python', '-I', '-S', '-B', str(converter),
                       '--profile', self.profile, '--execute']
            conversion = {**(conversion or {}), 'active_converter': pin(converter),
                          'algorithm_source': pin(self.package / 'root_to_uid995.py')}
        else:
            conversion_kind = 'root_CHOWN'
            wrapper = self.stage / ('root-phase-' + str(index) + '.py')
            wrapper.write_text('import os\nos.execv(' + repr(target[0]) + ',' + repr(target) + ')\n')
            wrapper.chmod(0o444)
            template = (self.package / 'root_chown_entry.py').read_text()
            need(template.count('ROOT_SOURCE_PREVIEW_ONLY = True') == 1, 'one root-CHOWN opt-in')
            active = template.replace('ROOT_SOURCE_PREVIEW_ONLY = True',
                                      'ROOT_SOURCE_PREVIEW_ONLY = False', 1)
            converter = self.stage / ('root-converter-' + str(index) + '.py')
            converter.write_text(active)
            converter.chmod(0o444)
            command = ['/opt/venv/bin/python', '-I', '-S', '-B', str(converter), '--execute',
                       '--entry', str(wrapper), '--entry-bytes', str(pin(wrapper)['bytes']),
                       '--entry-sha256', pin(wrapper)['sha256']]
            conversion = {'fixed_phase_argv': target, 'wrapper': pin(wrapper),
                          'active_converter': pin(converter),
                          'algorithm_source': pin(self.package / 'root_chown_entry.py')}
        record = {'name': name, 'argv': target, 'UID': 0 if user.endswith('0:0') else 995,
                  'started_monotonic': time.monotonic(), 'cap_seconds': min(seconds, deadline - time.monotonic()),
                  'exit_code': None}
        proc = None
        try:
            with raw_stdout.open('xb') as out, stderr.open('xb') as err:
                proc = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True,
                                        env=dict(os.environ, **extra_env))
                self.children.append(proc)
                limit = min(deadline, time.monotonic() + seconds)
                while proc.poll() is None:
                    self.scope(limit)
                    need(out.tell() <= 32 * 1024**2 and err.tell() <= 32 * 1024**2,
                         'bounded original phase streams')
                    time.sleep(.05)
                record['exit_code'] = proc.returncode
        finally:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=1)
                record['timed_out_or_interrupted'] = True
                record['exit_code'] = proc.returncode
            if name == 'original-normal11':
                # Its original finally recursively returns /evidence to995:985.
                # Restore the shared three-directory protocol before any new fact/log.
                self.share_evidence_directories()
            raw = raw_stdout.read_bytes()
            need(len(raw) <= 32 * 1024**2 and stderr.stat().st_size <= 32 * 1024**2,
                 'complete bounded phase streams')
            if conversion is not None and raw:
                first, separator, output = raw.partition(b'\n')
                need(separator and len(first) <= 65536, 'complete converter identity frame')
                observed = json.loads(first)
                need(observed['before']['pid'] == observed['after']['pid'] == proc.pid and
                     observed['before']['start_ticks'] == observed['after']['start_ticks'], 'same-PID conversion')
                if conversion_kind == 'UID995':
                    need(observed['profile'] == self.profile and observed['after']['uids'] == [995] * 4 and
                         observed['after']['gids'] == [985] * 4 and observed['after']['groups'] == [992] and
                         observed['after']['no_new_privs'] == 1 and observed['after']['caps'] ==
                         {'CapEff': 0, 'CapPrm': 0, 'CapInh': 0, 'CapAmb': 0, 'CapBnd': 1},
                         'original UID995 phase identity')
                else:
                    need(observed['after']['uids'] == [0] * 4 and observed['after']['gids'] == [0] * 4 and
                         observed['after']['groups'] == observed['before']['groups'] and
                         observed['after']['NoNewPrivs'] == 1 and observed['after']['caps'] ==
                         {'CapEff': 1, 'CapPrm': 1, 'CapInh': 0, 'CapAmb': 0, 'CapBnd': 1},
                         'original root CHOWN-only phase identity')
                save(self.evidence / 'facts' / ('conversion-' + str(index) + '.json'),
                     {'source': conversion, 'observed': observed})
                stdout.write_bytes(output)
            else:
                stdout.write_bytes(raw)
            self.logs += [stdout, stderr]
            record.update({'stdout': pin(stdout), 'stderr': pin(stderr), 'raw_stdout': pin(raw_stdout),
                           'finished_monotonic': time.monotonic()})
            self.result['commands'].append(record)
            save(self.evidence / 'root-result.json', self.result, 256 * 1024)
        need(record['exit_code'] == 0, 'original phase nonzero: ' + name)
        return stdout.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--profile', choices=('SG0', 'Symphony'), required=True)
    parser.add_argument('--transport', type=Path, required=True)
    args = parser.parse_args()
    if not args.execute or os.environ.get('SMG_PRIVATE_SGLANG_EXECUTE') != 'true':
        return 78
    need(os.getuid() == os.geteuid() == os.getgid() == os.getegid() == 0, 'job container root prepare')
    need(re.fullmatch('[a-f0-9]{64}', os.environ['ACTUAL_JOB_CONTAINER_ID']), 'actual runner CID context')
    checkout_package = Path(__file__).resolve().parent
    config = json.loads((checkout_package / 'route-inputs.json').read_bytes())
    need(all(name not in os.environ for name in config['denied_initial_env']) and
         all(name not in os.environ for name in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN')) and
         os.environ.get('HF_HUB_OFFLINE') not in ('1', 'true', 'True') and
         os.environ.get('TRANSFORMERS_OFFLINE') not in ('1', 'true', 'True'),
         'original no inherited AOT/PREBUILD/private/offline/priority credential gate')
    need(pin(args.transport / 'Root-start-authority.json')['bytes'] <= 65536, 'complete finite authority')
    grant = json.loads((args.transport / 'Root-start-authority.json').read_bytes())
    need(all(type(grant[key]) is int for key in ('whole_seconds', 'GPU_jobs', 'CPU', 'memory_GiB')) and
         all(type(grant[key]) in (int, float) and math.isfinite(grant[key]) and grant[key] > 0 for key in
             ('submitted_epoch', 'queue_expiry_epoch', 'occupancy_cutoff_epoch')) and
         0 <= grant['queue_expiry_epoch'] - grant['submitted_epoch'] <= 900,
         'strict numeric resource/finite queue900 authority')
    need(grant['scope'] == 'Root_controlled_first_SG0_Symphony_full231_GHA' and
         grant['authorized'] is True and grant['profile'] == args.profile and
         grant['source_commit'] == os.environ['GITHUB_SHA'] and
         re.fullmatch('[a-f0-9]{32}', grant['nonce']) and
         grant['whole_seconds'] == 2160 and grant['GPU_jobs'] == 1 and
         grant['CPU'] == 8 and grant['memory_GiB'] == 96, 'actual Root authority/profile')
    need(grant['image'] == config['image'] and
         grant['manifest_digest'] == 'sha256:' + config['image'].split('@sha256:')[1] and
         re.fullmatch('sha256:[a-f0-9]{64}', grant['config_digest']), 'fixed4420 manifest/config declaration')
    need(grant['external_root_watch_verified'] is True and
         grant['credential_component_verified'] is True and
         grant['credential_component_source_manifest_sha256'] == config['converter_source_manifest_sha256'] and
         pin(args.transport / 'component-proof.json')['sha256'] == grant['credential_component_proof_sha256'],
         'actual Root-controlled watchdog and credential component dependencies')
    need(grant['root_chown_component_verified'] is True and
         grant['root_chown_component_source_manifest_sha256'] == config['root_chown_source_manifest_sha256'] and
         pin(args.transport / 'root-chown-component-proof.json', 2 * 1024**2)['sha256'] ==
         grant['root_chown_component_proof_sha256'], 'actual root CHOWN-only component proof')
    job = actual_job()
    need(job['head_sha'] == grant['source_commit'], 'actual job source commit')
    need(job['runner_name'] in grant['eligible_runner_names'], 'fresh Root verified eligible runner name')
    package = runtime_package(args.transport, config)
    sys.modules['route_entry'] = sys.modules[__name__]
    adapter = Adapter(args.profile, grant, job, package)
    def interrupted(signum, frame):
        raise RuntimeError('owned phase interrupted by signal ' + str(signum))
    for kind in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM):
        signal.signal(kind, interrupted)
    signal.alarm(max(1, int(adapter.whole_stop - time.monotonic())))
    result = adapter.result
    try:
        metadata = json.loads((package / 'profiles.json').read_bytes())[args.profile]
        need(pin(args.transport / (args.profile + '.tar')) == metadata['payload'], 'original payload full SHA')
        need(pin(package / 'root_to_uid995.py') == metadata['converter'], 'reviewed converter source')
        spec = importlib.util.spec_from_file_location('reviewed_profile', package / ('profile_' + args.profile + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.run_profile(adapter, args.transport / (args.profile + '.tar'))
        need(result['work_passed'] is True, 'complete original profile work')
        result['passed'] = True
    except BaseException as error:
        result['error'] = {'type': type(error).__name__, 'message': str(error)[:2048],
                           'traceback': traceback.format_exc(limit=8)[-4096:]}
    finally:
        result['elapsed_seconds'] = time.monotonic() - adapter.origin
        result['within_original2160'] = result['elapsed_seconds'] < 2160
        result['cleanup']['tracked_phase_children_reaped'] = all(proc.poll() is not None for proc in adapter.children)
        result['cleanup']['job_container_removal'] = 'Runner post-job; actual logs/exact CID readback required'
        result['passed'] = (result['passed'] and result['within_original2160'] and
                            result['cleanup']['tracked_phase_children_reaped'])
        result['passed_scope'] = 'profile work only; strict GHA finish and Root physical closure separate'
        save(adapter.evidence / 'root-result.json', result, 256 * 1024)
        with tarfile.open('/evidence.tar', 'w:') as archive:
            def select_member(info):
                adapter.scope(adapter.whole_stop)
                if Path(info.name).parts[:2] in (('evidence', 'home'), ('evidence', 'cache')):
                    return None
                need(info.isfile() or info.isdir(), 'regular/directory complete evidence only')
                need(info.size <= 64 * 1024**2 and Path('/evidence.tar').stat().st_size <= 768 * 1024**2,
                     'original complete evidence member/whole caps; no truncation')
                return info
            archive.add(adapter.evidence, arcname='evidence', filter=select_member)
            for child in sorted(adapter.stage.iterdir()):
                if child.name != 'delivered':
                    archive.add(child, arcname='phase-source/' + child.name, filter=select_member)
        adapter.scope(adapter.whole_stop)
        save(Path('/evidence-tar-pin.json'), pin('/evidence.tar', 768 * 1024**2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
