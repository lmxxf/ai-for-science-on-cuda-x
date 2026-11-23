#!/usr/bin/env python3
"""Copy Phase 2 source and compact evidence to the sibling AMD workspace."""
from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parent
D=R.parent/'ai-for-science-on-amd-9070xt'

def main():
 D.mkdir(exist_ok=True);copied=[]
 def copy(p):
  rel=p.relative_to(R);dest=D/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
  copied.append({'path':str(rel),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 for name in ['PHASE2_FULL_AI4S_PLAN.md','REPORT.md','DevHistory.md','MODULE6_NOTES.md','MODULE7_NOTES.md','MODULE8_NOTES.md','exp5_open_cupotential_hip.py','render_module7.py','guard_process.py']:
  copy(R/name)
 for folder in ['amd_hip_port/unimol_pairbias','amd_hip_port/cupotential_open']:
  for p in (R/folder).rglob('*'):
   if not p.is_file():continue
   rel=p.relative_to(R/folder)
   if '__pycache__' in rel.parts or 'input' in rel.parts or p.suffix in ['.bin','.exe','.pt','.obj','.pyc','.pdb'] or p.name=='pairbias_cuda':continue
   copy(p)
 for p in (R/'results').rglob('*'):
  if p.is_file() and p.suffix in ['.json','.jsonl','.csv','.log'] and p.name.startswith(('module6','module7','module8','bbbp_')):copy(p)
  elif p.is_file() and p.name=='timings.json' and p.parent.name.startswith('module7_'):copy(p)
 for p in (R/'visuals').glob('fig[567]*'):copy(p)
 # Remove only this run's redundant, untracked second source copy.
 redundant=D/'cupotential_open';canonical=D/'amd_hip_port/cupotential_open'
 if redundant.exists():
  for p in redundant.rglob('*'):
   if p.is_file():
    q=canonical/p.relative_to(redundant)
    if not q.exists() or q.read_bytes()!=p.read_bytes():raise RuntimeError(f'Duplicate content differs: {p}')
  shutil.rmtree(redundant)
 (D/'results/phase2_sync_manifest.json').write_text(json.dumps({'source_workspace':str(R),'files':copied},indent=2)+'\n')
 print(f'Synced {len(copied)} compact source/evidence files to {D}')
if __name__=='__main__':main()
