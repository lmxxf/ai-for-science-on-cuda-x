#!/usr/bin/env python3
"""Mirror Phase4 sources/compact evidence; leave raw coordinate histories and H matrices local."""
from pathlib import Path
import hashlib,json,shutil,subprocess
R=Path(__file__).resolve().parent;D=R.parent/'ai-for-science-on-amd-9070xt'
def main():
 copied=[]
 def copy(p):
  rel=p.relative_to(R);q=D/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);copied.append({'path':str(rel),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 for name in ['PHASE4_NVMOLKIT_AND_AUDIT_PLAN.md','REPORT.md','DevHistory.md','MODULE14_NOTES.md','MODULE15_NOTES.md','MODULE16_NOTES.md','exp12_open_nvmolkit_batched_conformer_hip.py','exp13_daviator_resident_hip.py','exp14_audit381_record.py','verify_phase4.py','render_module14.py']:
  copy(R/name)
 for name in ['nvmolkit_batched_conformer','daviator_batched_dft']:
  for p in (R/'amd_hip_port'/name).rglob('*'):
   if not p.is_file():continue
   rel=p.relative_to(R/'amd_hip_port'/name)
   if '__pycache__' in rel.parts or 'data' in rel.parts or 'input' in rel.parts or p.suffix in ['.bin','.exe','.obj','.pyc','.pdb'] or p.stat().st_size>2000000:continue
   copy(p)
 for p in (R/'results').rglob('*'):
  if p.is_file() and p.relative_to(R/'results').parts[0].startswith(('module14','module15','module16')) and p.suffix in ['.json','.jsonl','.csv','.npz','.log'] and p.stat().st_size<2000000:copy(p)
 for name in ['fig9_daviator_batched_davidson_vasp_band.svg','fig9_daviator_batched_davidson_vasp_band.png','fig9_daviator_hybrid_original.svg','fig13_open_nvmolkit_10000_conformers_amd_vs_nvidia.svg','fig13_open_nvmolkit_10000_conformers_amd_vs_nvidia.png']:copy(R/'visuals'/name)
 for name in ['nvmolkit_batched_conformer','daviator_batched_dft']:
  old=D/name;canonical=D/'amd_hip_port'/name
  if old.is_dir() and not subprocess.check_output(['git','-C',str(D),'ls-files',name],text=True).strip():
   if all((canonical/p.relative_to(old)).is_file() and (canonical/p.relative_to(old)).read_bytes()==p.read_bytes() for p in old.rglob('*') if p.is_file()):shutil.rmtree(old)
 (D/'results/phase4_sync_manifest.json').write_text(json.dumps({'source_workspace':str(R),'files':copied},indent=2)+'\n');print(f'Synced {len(copied)} compact Phase4 source/evidence files')
if __name__=='__main__':main()
