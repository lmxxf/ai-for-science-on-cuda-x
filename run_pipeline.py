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
    'pme_md':('exp7_pme_and_fused_md_hip.py','results/module9_results.json'),
    'davidson':('exp8_daviator_batched_davidson_hip.py','results/module10_results.json'),
    'blackhole_fx':('exp9_cuphoton_blackhole_fx_hip.py','results/module11_results.json'),
    'profile_hmm':('exp10_gpuhmmer_viterbi_hip.py','results/module12_results.json'),
    'unimol_full15':('exp11_unimol_full15layer_amd9070.py','results/module13_results.json'),
}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--modules',nargs='+',choices=list(MODULES),default=None)
    ap.add_argument('--phase',choices=['1','2','3','all'],default='1',help='1: RDKit/ESOL/cuSpa; 2: BBBP/forcefield/ESM-2; 3: PME/Davidson/FX/HMM/full Uni-Mol')
    ap.add_argument('--reuse-results',action='store_true',help='Validate existing outputs without rerunning expensive stages')
    a=ap.parse_args(); os.chdir(ROOT)
    phase_groups={'1':list(MODULES)[:3],'2':list(MODULES)[3:6],'3':list(MODULES)[6:]}
    if a.modules is None:
        a.modules=list(MODULES) if a.phase=='all' else phase_groups[a.phase]
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
    if set(phase_groups['1']).issubset(a.modules):
        subprocess.run([sys.executable,'verify_artifacts.py'],check=True)
    if set(phase_groups['2']).issubset(a.modules):
        subprocess.run([sys.executable,'verify_phase2.py'],check=True)
    if set(phase_groups['3']).issubset(a.modules):
        subprocess.run([sys.executable,'verify_phase3.py'],check=True)
    subprocess.run([sys.executable,'build_report.py'],check=True)
    Path('results/pipeline_manifest.json').write_text(json.dumps(dict(timestamp=datetime.datetime.now().astimezone().isoformat(),python=sys.executable,modules=records),indent=2)+'\n')
    print(json.dumps(records,indent=2))
if __name__=='__main__':main()
