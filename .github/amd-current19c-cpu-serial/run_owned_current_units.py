"""Root-only future NEW600, local cached-image CPU replay. Never dispatch a workflow."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import socket
import subprocess
import sys
import tarfile
import time
import uuid

HERE=Path(__file__).resolve().parent
if hashlib.sha256((HERE/"common.py").read_bytes()).hexdigest()!="848ad435d97423ed3ae443bc40597800de2f8821284c013383e1439c611f8a11":raise ValueError("common source changed")
sys.path.insert(0,str(HERE))
from common import checked, identity, inputs, require
if hashlib.sha256((HERE/"worker_identity.py").read_bytes()).hexdigest()!="909b87b28d5a98efbd546df590c144d078e79ad7744e65cc8f76622fa89245c2":raise ValueError("worker identity source changed")
from worker_identity import local_worker, WORKFLOW


def ticks(pid):
    return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]


def evidence_inventory(root, deadline):
    """Lazy no-follow bounded DATA walk; exact CID is stopped for final capture."""
    require(root.is_dir() and not root.is_symlink(), 'durable evidence directory missing/aliased')
    result=[]; todo=[root]; total=0
    while todo:
        require(time.monotonic()<deadline,'absolute evidence deadline')
        parent=todo.pop()
        with os.scandir(parent) as entries:
            while True:
                # Check before requesting the next directory entry. At the cap,
                # only one bounded end-of-iterator/overflow observation is allowed.
                require(time.monotonic()<deadline and len(result)<=512,'evidence iteration bound')
                try:entry=next(entries)
                except StopIteration:break
                path=Path(entry.path);rel=path.relative_to(root).as_posix()
                require(len(result)<512 and len(PurePosixPath(rel).parts)<=12,'evidence entry/depth cap')
                value=entry.stat(follow_symlinks=False)
                require(stat.S_ISDIR(value.st_mode) or stat.S_ISREG(value.st_mode),'evidence link/special file rejected')
                if stat.S_ISDIR(value.st_mode):todo.append(path);kind='directory'
                else:
                    require(value.st_nlink==1,'evidence hardlink rejected')
                    total+=value.st_size;kind='regular'
                    require(total<128*1024**2,'durable evidence128MiB cap')
                result.append((rel,kind,value.st_size,value.st_dev,value.st_ino))
    return result,total


def main():
    parser=argparse.ArgumentParser()
    for name in ('packet','approval','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--approval-sha256',required=True)
    args=parser.parse_args()
    started,epoch=time.monotonic(),time.time()
    require(__debug__,'product assertions must remain enabled')
    require(identity(args.approval)['sha256']==args.approval_sha256,'approval changed')
    approval=json.loads(args.approval.read_text()); age=epoch-approval['issued_epoch']
    require(approval['root_admitted'] is True and approval['CPU_seconds_new']==600 and approval['GPU']==0,'NEW600 CPU-only approval required')
    require(0<=age<30 and 0<=epoch-approval['capacity_epoch']<180,'approval/capacity stale')
    require(socket.gethostname()==approval['hostname'] and approval['hostname'] in {'amd-mi300x-1','amd-mi300x-2'},'wrong target node')
    require(approval['root_CPU_slot_registered_only_this_window'] is True,'root CPU slot required')
    require(approval['controlled_network_for_normal_APT_and_PyPI'] is True,'normal dependency fetch not approved')
    packet=args.packet.resolve(strict=True); config=inputs(packet)
    require(identity(Path(__file__))['sha256']==approval['driver_sha256'],'driver mismatch')
    require(identity(packet/'inputs.json')['sha256']==approval['inputs_sha256'],'inputs mismatch')
    peer_path=Path(approval['source_peer_path']); checked(peer_path,approval['source_peer'])
    peer=json.loads(peer_path.read_text())
    for name in ('driver_sha256','payload_sha256','inputs_sha256'):
        actual=identity(packet/({'driver_sha256':'run_owned_current_units.py','payload_sha256':'payload.py','inputs_sha256':'inputs.json'}[name]))['sha256']
        require(peer[name]==actual,'source peer byte mismatch: '+name)
    require(peer['result'].startswith('PASS') and peer['reviewed_head']==config['source_head'],'source peer not accepted')
    require(identity(packet/'frozen-files.json')['sha256']==peer['frozen_manifest_sha256'],'source peer frozen manifest mismatch')
    for name,digest in json.loads((packet/'frozen-files.json').read_text()).items():
        require(name and not Path(name).is_absolute() and '..'not in Path(name).parts,'frozen source path')
        require(identity(packet/name)['sha256']==digest,'frozen source byte mismatch')
    manifest=json.loads((packet/'packet-files.json').read_text())
    require(identity(packet/'packet-files.json')['sha256']==approval['packet_manifest_sha256'],'packet manifest mismatch')
    files=list(packet.rglob('*'))
    require(all(not f.is_symlink() for f in files),'packet symlink')
    regular=[f for f in files if f.is_file()]
    require({f.relative_to(packet).as_posix() for f in regular}==set(manifest)|{'packet-files.json'},'packet exact set mismatch')
    require(len(regular)<256 and sum(f.stat().st_size for f in regular)<192*1024**2,'packet cap')
    for name,digest in manifest.items():
        require(name and not Path(name).is_absolute() and '..'not in Path(name).parts,'packet path')
        require(identity(packet/name)['sha256']==digest,'packet bytes mismatch')
    out=args.out.resolve()
    require(str(out)==approval['output_directory'] and not out.exists() and not out.is_relative_to(packet) and not packet.is_relative_to(out),'exclusive output required')
    out.mkdir(mode=0o700)
    persistent=out/'persistent'; persistent.mkdir(mode=0o700)
    persistent_identity=(persistent.stat().st_dev,persistent.stat().st_ino)
    deadline,work_end=started+600-age,started+565-age
    nonce='smg-ts173-current-units-'+uuid.uuid4().hex; label='smg.ts173.current.cpu.nonce'
    result={'status':'FAIL','source_head':config['source_head'],'CPU_seconds_new':600,'GPU':0,'commands':[],
        'durable_evidence_root':str(persistent),'host_evidence_owner':{'UID':os.getuid(),'GID':os.getgid()},
        'work_tmpfs_bytes':4*1024**3,'durable_evidence_acceptance_cap_bytes':128*1024**2,
        'nonce':nonce,'PID':os.getpid(),'start_ticks':ticks(os.getpid()),'cleanup':{},'approval_sha256':args.approval_sha256}
    cid=image_id=process=process_start=None
    def save():
        (out/'driver-result.json').write_text(json.dumps(result,indent=2)+'\n')
    def call(argv,limit=5,cleanup=False,binary=False):
        remaining=(deadline if cleanup else work_end)-time.monotonic()
        require(remaining>0,'absolute NEW600 expired')
        i=len(result['commands']); row={'argv':argv,'started_epoch':time.time()}; result['commands'].append(row); save()
        stdout,stderr=out/(str(i)+'.stdout'),out/(str(i)+'.stderr')
        with stdout.open('xb') as a,stderr.open('xb') as b:
            done=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=a,stderr=b,timeout=min(limit,remaining))
        row.update(exit_code=done.returncode,finished_epoch=time.time(),stdout=str(stdout),stderr=str(stderr)); save()
        require(done.returncode==0,'command failed: '+str(argv[:3]))
        require(stdout.stat().st_size<128*1024**2 and stderr.stat().st_size<4*1024**2,'output cap')
        return stdout if binary else stdout.read_text()
    def owned():
        actual=json.loads(call(['docker','inspect',cid],cleanup=True))[0]
        require(actual['Id']==cid and actual['Image']==image_id and actual['Config']['Labels'].get(label)==nonce,'refuse foreign CID operation')
        return actual
    def persistent_unchanged():
        require(not persistent.is_symlink() and persistent.is_dir(),'durable output root changed')
        require((persistent.stat().st_dev,persistent.stat().st_ino)==persistent_identity,'durable output incarnation changed')
        seen=set()
        with os.scandir(persistent) as entries:
            for _ in range(3):  # Two legal entries, plus one bounded overflow/end observation.
                require(time.monotonic()<deadline,'absolute durable root deadline')
                try:entry=next(entries)
                except StopIteration:return
                require(len(seen)<2 and entry.name in {'owner.json','evidence'} and entry.name not in seen,'unexpected durable output root member')
                seen.add(entry.name)
        raise RuntimeError('durable output root entry overflow')
    def capture():
        # Host DATA persists after CID stop/removal; do not use docker cp on tmpfs/bind.
        persistent_unchanged()
        evidence=persistent/'evidence'
        members,total=evidence_inventory(evidence,deadline) if evidence.exists() else ([],0)
        archive=out/'CPU-evidence.tar'; partial=out/'CPU-evidence.tar.partial'
        with partial.open('xb') as stream,tarfile.open(fileobj=stream,mode='w:') as tar:
            for rel,kind,size,dev,ino in members:
                require(time.monotonic()<deadline,'absolute capture deadline')
                path=evidence/rel; current=path.lstat()
                require((current.st_dev,current.st_ino,current.st_size)==(dev,ino,size),'evidence changed during capture')
                item=tarfile.TarInfo('evidence/'+rel);item.mtime=int(current.st_mtime)
                if kind=='directory':item.type=tarfile.DIRTYPE;item.mode=0o755;tar.addfile(item)
                else:
                    item.mode=0o644;item.size=size
                    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
                    with os.fdopen(fd,'rb') as source:
                        actual=os.fstat(source.fileno())
                        require(stat.S_ISREG(actual.st_mode) and (actual.st_dev,actual.st_ino,actual.st_size)==(dev,ino,size),'evidence opened identity changed')
                        tar.addfile(item,source)
            stream.flush();os.fsync(stream.fileno())
        partial.rename(archive)
        payload=evidence/'payload-result.json'
        if payload.is_file() and not payload.is_symlink():
            require(payload.stat().st_size<512*1024,'payload result cap')
            result['payload_result']=json.loads(payload.read_text())
        else:result['payload_result_missing']=True
        result['evidence_archive']=identity(archive)
        result['durable_evidence']={'retained':True,'root':str(persistent),'entries':len(members),'regular_bytes':total,'host_data_capture':True}
    (persistent/'owner.json').write_text(json.dumps({'nonce':nonce,'driver_PID':os.getpid(),'driver_start_ticks':result['start_ticks'],'UID':os.getuid(),'GID':os.getgid(),'source_head':config['source_head']},indent=2)+'\n')
    cleaning=False
    def interrupted(signum, frame):
        result['cancellation_signal']=signum; result['status']='FAIL'
        # One cooperative cancellation enters the original bounded finally.
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,signal.SIG_IGN)
        if not cleaning:raise TimeoutError('actual GHA job cancelled')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    try:
        require(not call(['docker','ps','-q']).splitlines(),'foreign/ordinary container present')
        require(approval['GHA_workflow_path']==WORKFLOW and os.environ['GITHUB_EVENT_NAME']=='workflow_dispatch','wrong scheduled workflow')
        require(os.environ['GITHUB_RUN_ATTEMPT']=='1' and approval['GHA_run_id']==int(os.environ['GITHUB_RUN_ID']) and approval['GHA_head_sha']==os.environ['GITHUB_SHA'],'wrong scheduled run/head')
        require(local_worker(approval['hostname'])==approval['GHA_local_worker'],'assigned sole Worker incarnation/argv/ancestry changed')
        observed=Path(approval['GHA_identity_evidence_path'])
        require(identity(observed)['sha256']==approval['GHA_identity_evidence_sha256'],'actual GHA evidence changed')
        result['actual_GHA_identity']=json.loads(observed.read_text())
        require(result['actual_GHA_identity']['job']['id']==approval['GHA_job_id'],'scheduled job changed')
        result['actual_GHA_window_nonce']=approval['GHA_window_nonce']
        image=json.loads(call(['docker','image','inspect',config['image']]))[0]
        image_id=image['Id']
        require(image_id==approval['image_id'] and any(x.endswith('@'+config['image'].split('@')[1])for x in image['RepoDigests']),'cached image identity mismatch')
        argv=['docker','create','--pull=never','--runtime=runc','--name',nonce,'--label',label+'='+nonce,
            '--network=bridge','--security-opt=no-new-privileges','--cpus=2','--memory=8g','--memory-swap=8g',
            '--pids-limit=256','--ulimit','fsize=268435456:268435456','--user=0:0','--workdir=/out',
            '--tmpfs','/tmp:rw,exec,nosuid,nodev,size=1g','--tmpfs','/out:rw,exec,nosuid,nodev,size=4g',
            '--mount','type=bind,src='+str(packet)+',dst=/input,readonly',
            '--mount','type=bind,src='+str(persistent)+',dst=/out/persistent']
        for name,value in {'HOME':'/tmp','CUDA_VISIBLE_DEVICES':'','HIP_VISIBLE_DEVICES':'','ROCR_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void','GPU_DEVICE_ORDINAL':'',
            'FROZEN19C_EVIDENCE_UID':str(os.getuid()),'FROZEN19C_EVIDENCE_GID':str(os.getgid()),
            'FROZEN19C_WORK_DEADLINE_EPOCH':str(time.time()+started+540-age-time.monotonic())}.items():argv.extend(['--env',name+'='+value])
        argv.extend(['--entrypoint=/usr/bin/python3',config['image'],'-I','-B','-S','/input/payload.py'])
        cid=call(argv,10).strip()
        require(len(cid)==64 and set(cid)<=set('0123456789abcdef'),'invalid CID')
        actual=owned(); hc=actual['HostConfig']
        require(hc['Runtime']=='runc' and hc['NetworkMode']=='bridge' and not hc['Devices'] and not hc.get('DeviceRequests') and not hc['Privileged'] and not hc['ReadonlyRootfs'],'container isolation mismatch')
        require(hc['NanoCpus']==2000000000 and hc['Memory']==8*1024**3 and hc['PidsLimit']==256,'resource caps mismatch')
        require({(m['Destination'],m['RW'])for m in actual['Mounts']if m['Type']=='bind'}=={('/input',False),('/out/persistent',True)},'bind mount mismatch')
        require({m['Destination']:m['Source'] for m in actual['Mounts']if m['Type']=='bind'}=={'/input':str(packet),'/out/persistent':str(persistent)},'bind source mismatch')
        require(set(hc['Tmpfs'])=={'/tmp','/out'} and all('exec'in v.split(',') and 'noexec'not in v.split(',')for v in hc['Tmpfs'].values()),'private executable tmpfs mismatch')
        require(actual['Config']['Entrypoint']==['/usr/bin/python3'] and actual['Config']['Cmd']==['-I','-B','-S','/input/payload.py'],'payload entrypoint mismatch')
        result['container_inspect_before']=actual
        with (out/'container.stdout').open('xb') as a,(out/'container.stderr').open('xb') as b:
            process=subprocess.Popen(['docker','start','--attach',cid],stdin=subprocess.DEVNULL,stdout=a,stderr=b,start_new_session=True)
            process_start=ticks(process.pid); result['attached']={'PID':process.pid,'start_ticks':process_start,'PGID':process.pid}; save()
            while process.poll() is None:
                require(time.monotonic()<work_end,'absolute NEW600 work deadline')
                persistent_unchanged()
                if (persistent/'evidence').exists():evidence_inventory(persistent/'evidence',work_end)
                try:process.wait(timeout=min(.25,max(.01,work_end-time.monotonic())))
                except subprocess.TimeoutExpired:pass
        state=owned()['State']; result['container_terminal']=state
        require(not state['Running'],'attached client exited while owned CID is running')
        capture()
        require(not state['OOMKilled'] and state['ExitCode']==0 and process.returncode==0,'target payload failed')
        require(result['payload_result']['status']=='PASS_CPU_SHARED_NORMAL_EDITABLE_CURRENT_UNITS_COLLECTION_AND_CANONICAL14_ONLY','CPU gates failed')
        result['status']=result['payload_result']['status']
    except BaseException as error:
        result['first_failure']={'type':type(error).__name__,'message':str(error)[:4000]}
    finally:
        cleaning=True
        try:
            if process and process.poll() is None:
                require(ticks(process.pid)==process_start and os.getpgid(process.pid)==process.pid,'attach incarnation changed')
                os.killpg(process.pid,signal.SIGKILL); process.wait(timeout=2)
        except BaseException as error:result['cleanup']['attach_error']=str(error)[:1000]
        try:
            ids=call(['docker','ps','-a','--no-trunc','--filter',('id='+cid)if cid else('name=^/'+nonce+'$'),'--format','{{.ID}}'],cleanup=True).splitlines()
            if ids:
                require(len(ids)==1 and (cid is None or ids==[cid]),'ambiguous CID cleanup')
                cid=ids[0]; actual=owned()
                if actual['State']['Running']:
                    call(['docker','kill',cid],cleanup=True)
                    require(not owned()['State']['Running'],'owned CID did not stop')
                call(['docker','rm','--force',cid],cleanup=True)
            require(not cid or not call(['docker','ps','-a','--no-trunc','--filter','id='+cid,'--format','{{.ID}}'],cleanup=True).splitlines(),'owned CID still present')
            result['cleanup']['owned_CID_absent_machine_readable']=True
            if process:require(str(process.pid)not in call(['ps','-eo','pgid='],cleanup=True).split(),'owned attach PG remains')
            result['cleanup']['owned_attach_PG_absent']=True
            result['cleanup']['owned_tmpfs_work_removed_with_CID']=True
            # Durable host DATA survives external CID removal. Capture only after
            # exact CID absence and attach-PG absence, independent of prior ids.
            if 'evidence_archive'not in result:
                try:capture()
                except BaseException as error:result['capture_error']=str(error)[:1000]
            persistent_unchanged()
            result['cleanup']['durable_evidence_retained_not_deleted']=True
        except BaseException as error:result['cleanup']['container_error']=str(error)[:2000]
        result['elapsed_seconds']=time.monotonic()-started+age
        result['within_NEW600']=result['elapsed_seconds']<=600
        if not result['within_NEW600'] or any(k.endswith('error')for k in result['cleanup']) or not all(result['cleanup'].get(k)for k in ('owned_CID_absent_machine_readable','owned_attach_PG_absent','owned_tmpfs_work_removed_with_CID')):result['status']='FAIL'
        result['coverage_increment']=0; save()
    return 0 if result['status'].startswith('PASS_')else 1


if __name__=='__main__':
    raise SystemExit(main())
