"""Sequential normalized 43/44 training with source snapshot and progress log."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime
from types import SimpleNamespace
from scripts.train_normalized_ablation_suite import (
    expected_run_configuration, completed_run_is_compatible, training_command,
)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/training_suites/seeds43_44_queue'
ORDER = [('posvel_force11_k1','posvel_force11',1),
         ('posvel_force11_k4','posvel_force11',4),
         ('posvel9_k1','posvel9',1),('posvel9_k4','posvel9',4),('pos6_k1','pos6',1)]
def now(): return datetime.now().astimezone().isoformat(timespec='seconds')
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    cli=parser.parse_args()
    jobs=[]
    for seed in (43,44):
        args=SimpleNamespace(seed=seed,run_prefix='formal_norm',timesteps=500000,
            friction_min=0.3,friction_max=0.9,checkpoint_freq=50000,ent_coef=0.01,device='cpu')
        for label,mode,k in ORDER:
            job=expected_run_configuration(args,label,mode,k)
            job['command']=training_command(args,job)
            jobs.append(job)
    if cli.dry_run:
        for i,j in enumerate(jobs,1): print(i,j['run_name'],' '.join(j['command']))
        return
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Validate all destinations before starting any long training.
        for j in jobs:
            j['status']='skipped_completed' if completed_run_is_compatible(ROOT/'outputs/runs'/j['run_name'],j) else 'pending'
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        source_list=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT).split(b'\0')
        hashes={}
        archive=OUT/f'source_{stamp}.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for raw in sorted(set(source_list)):
                if not raw: continue
                name=os.fsdecode(raw); p=ROOT/name
                if p.is_file():
                    hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
                    tar.add(p,arcname=name,recursive=False)
        state={'started_at':now(),'status':'running','pid':os.getpid(),
            'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
            'source_archive':str(archive),'source_sha256':hashes,'jobs':jobs}
        def save():
            tmp=OUT/'status.json.tmp'; tmp.write_text(json.dumps(state,indent=2)+'\n'); tmp.replace(OUT/'status.json')
        save()
        try:
            for i,j in enumerate(jobs,1):
                print(f"[{i}/10] {j['run_name']} {j['status']}",flush=True)
                if j['status']=='skipped_completed': continue
                # Reject edits to the code actually used by subsequent jobs.
                for name,h in hashes.items():
                    if name.endswith('.py') and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:
                        raise RuntimeError(f'Source changed during queue: {name}')
                log=OUT/f"{j['run_name']}_{stamp}.log"
                j.update(status='running',started_at=now(),log=str(log));save()
                with log.open('w') as stream:
                    subprocess.run(j['command'],cwd=ROOT,env={**os.environ,'PYTHONUNBUFFERED':'1'},stdout=stream,stderr=subprocess.STDOUT,check=True)
                if not completed_run_is_compatible(ROOT/'outputs/runs'/j['run_name'],j):
                    raise RuntimeError('Training did not produce a compatible completed model')
                j.update(status='completed',finished_at=now());save()
            state.update(status='completed',finished_at=now());save()
            print('All 10 jobs completed.',flush=True)
        except BaseException as exc:
            state.update(status='failed',error=str(exc),finished_at=now());save();raise
if __name__=='__main__': main()
