#!/usr/bin/env python3
"""Copy Phase 3 source, compact evidence and figures without input/trace binaries."""
from pathlib import Path
import hashlib,json,shutil,subprocess
R=Path(__file__).resolve().parent;D=R.parent/'ai-for-science-on-amd-9070xt'
PORTS=['cupotential_pme_md','daviator_batched_dft','cuphoton_xbeam_fx','gpuhmmer_viterbi','unimol_full15_hip']
def main():
 copied=[]
 def copy(p):
  rel=p.relative_to(R);dest=D/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);copied.append({'path':str(rel),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 for name in ['PHASE3_GRAND_SLAM_AMD_PLAN.md','REPORT.md','DevHistory.md','guard_process.py','verify_phase3.py',*['MODULE'+str(n)+'_NOTES.md' for n in range(9,14)],'exp7_pme_and_fused_md_hip.py','exp8_daviator_batched_davidson_hip.py','exp9_cuphoton_blackhole_fx_hip.py','exp10_gpuhmmer_viterbi_hip.py','exp11_unimol_full15layer_amd9070.py']:
  copy(R/name)
 for name in ['render_module9.py']:
  if (R/name).exists():copy(R/name)
 for port in PORTS:
  for p in (R/'amd_hip_port'/port).rglob('*'):
   if not p.is_file():continue
   rel=p.relative_to(R/'amd_hip_port'/port)
   if '__pycache__' in rel.parts or 'input' in rel.parts or 'data' in rel.parts:continue
   if p.suffix in ['.bin','.exe','.obj','.pyc','.pdb'] or p.stat().st_size>2000000:continue
   copy(p)
 for p in (R/'results').rglob('*'):
  if not p.is_file():continue
  first=p.relative_to(R/'results').parts[0]
  if first.startswith(tuple('module'+str(n) for n in range(9,14))) and p.suffix in ['.json','.jsonl','.csv','.npz','.log'] and p.stat().st_size<2000000:copy(p)
 for n in range(8,13):
  for p in (R/'visuals').glob(f'fig{n}_*'):copy(p)
 # Only remove our own redundant untracked duplicate after proving identical content.
 for port in PORTS:
  old=D/port;canonical=D/'amd_hip_port'/port
  if not old.is_dir():continue
  tracked=subprocess.check_output(['git','-C',str(D),'ls-files',port],text=True).strip()
  if tracked:continue
  same=all((canonical/p.relative_to(old)).is_file() and (canonical/p.relative_to(old)).read_bytes()==p.read_bytes() for p in old.rglob('*') if p.is_file())
  if same:shutil.rmtree(old)
 (D/'results/phase3_sync_manifest.json').write_text(json.dumps({'source_workspace':str(R),'files':copied},indent=2)+'\n');print(f'Synced {len(copied)} compact Phase3 files')
if __name__=='__main__':main()
