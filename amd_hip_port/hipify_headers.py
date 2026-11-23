#!/usr/bin/env python3
"""Mechanical, reviewable translation of original NVIDIA Apache-2.0 headers."""
from pathlib import Path
import re,json,hashlib
ROOT=Path(__file__).resolve().parent
UP=ROOT.parent/'cuspa/cpp/src'
files=['common/cuda_utils.cuh','common/geometry.cuh','index/kernels_index.cuh','query/kernels_query.cuh','aggregate/kernels_aggregate.cuh']
manifest=[]
for f in files:
 s=(UP/f).read_text();before=s
 s=s.replace('<cuda_runtime.h>','<hip/hip_runtime.h>')
 s=re.sub(r'#include <cub/device/[^>]+>', '#include <hipcub/hipcub.hpp>',s)
 s=re.sub(r'\bcub::','hipcub::',s)
 s=re.sub(r'\bcuda([A-Z][A-Za-z0-9_]*)',r'hip\1',s)
 # cudaSuccess's small s is the only runtime token not covered above.
 s=s.replace('cudaSuccess','hipSuccess').replace('CUDA error:','HIP error:')
 out=ROOT/f;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
 manifest.append(dict(path=f,upstream_sha256=hashlib.sha256(before.encode()).hexdigest(),hip_sha256=hashlib.sha256(s.encode()).hexdigest(),source_lines=len(before.splitlines())))
(ROOT/'translation_manifest.json').write_text(json.dumps(manifest,indent=2))
