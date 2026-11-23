#!/usr/bin/env python3
"""Bytewise and elementwise cross-vendor agreement of canonical int32 outputs."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent

def compare():
 ref=ROOT/'results/module5_gb10_reference';amd=ROOT/'results/module5_amd9070'
 report={'validation_scope':'all 2000000 point IDs and canonical CSR int32 arrays; exact bytes and values','arrays':{},'total_mismatches':0}
 for name in ['assignment','indptr','indices','data']:
  a=(ref/f'{name}.bin').read_bytes();b=(amd/f'{name}.bin').read_bytes()
  av=np.frombuffer(a,dtype='<i4');bv=np.frombuffer(b,dtype='<i4')
  if av.shape!=bv.shape:raise ValueError(f'{name} shape mismatch: {av.shape} vs {bv.shape}')
  count=int(np.count_nonzero(av!=bv));report['total_mismatches']+=count
  report['arrays'][name]={'elements':int(av.size),'mismatches':count,'gb10_sha256':hashlib.sha256(a).hexdigest(),'amd_sha256':hashlib.sha256(b).hexdigest(),'byte_identical':a==b}
 report['all_byte_identical']=all(v['byte_identical'] for v in report['arrays'].values())
 (ROOT/'results/module5_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2));assert report['total_mismatches']==0 and report['all_byte_identical']
 return report
if __name__=='__main__':compare()
