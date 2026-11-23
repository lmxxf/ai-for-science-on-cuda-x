#!/usr/bin/env python3
"""Reproduce measured AI4S modules; cached outputs require explicit --reuse-results."""
import argparse, datetime, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODULES={
    'rdkit':('exp1_rdkit_data_processing.py','results/module1_results.json'),
    'unimol':('exp2_unimol_finetune.py','results/module2_results.json'),
    'cuspa':('exp3_cudax_cuspa_benchmark.py','results/module3_results.json'),
    'bbbp':('exp4_bbbp_unimol_and_amd.py','results/module6_results.json'),
    'forcefield':('exp5_open_cupotential_hip.py','results/module7_results.json'),
    'protein':('exp6_protein_esm2_finetune.py','results/module8_results.json'),
}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--modules',nargs='+',choices=list(MODULES),default=None)
    ap.add_argument('--phase',choices=['1','2','all'],default='1',help='Phase 1: RDKit/ESOL/cuSpa; phase 2: BBBP/forcefield/ESM-2')
    ap.add_argument('--reuse-results',action='store_true',help='Validate existing outputs without rerunning expensive stages')
    a=ap.parse_args(); os.chdir(ROOT)
    if a.modules is None:
        a.modules=list(MODULES) if a.phase=='all' else list(MODULES)[:3] if a.phase=='1' else list(MODULES)[3:]
    records=[]
    for name in a.modules:
        script,result=MODULES[name]; started=time.perf_counter()
        if a.reuse_results:
            if not Path(result).is_file(): raise SystemExit(f'Missing cached result: {result}')
            json.loads(Path(result).read_text())
            mode='validated_cached_result'
        else:
            if not Path(script).is_file(): raise SystemExit(f'Missing module: {script}')
            cmd=[sys.executable,'guard_process.py','--audit',f'logs/{name}_pipeline_memory.json','--',sys.executable,script]
            subprocess.run(cmd,check=True,env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MPLBACKEND':'Agg'})
            if not Path(result).is_file(): raise SystemExit(f'Module did not produce {result}')
            mode='executed'
        records.append(dict(module=name,mode=mode,elapsed_s=time.perf_counter()-started,result=result,sha256=hashlib.sha256(Path(result).read_bytes()).hexdigest()))
    if set(list(MODULES)[:3]).issubset(a.modules):
        subprocess.run([sys.executable,'verify_artifacts.py'],check=True)
    if set(list(MODULES)[3:]).issubset(a.modules):
        subprocess.run([sys.executable,'verify_phase2.py'],check=True)
    subprocess.run([sys.executable,'build_report.py'],check=True)
    Path('results/pipeline_manifest.json').write_text(json.dumps(dict(timestamp=datetime.datetime.now().astimezone().isoformat(),python=sys.executable,modules=records),indent=2)+'\n')
    print(json.dumps(records,indent=2))
if __name__=='__main__':main()
