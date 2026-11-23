#!/usr/bin/env python3
"""Export exact input bytes and native GB10 reference output for HIP validation."""
import os,sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'results/module3_builddeps'))
import numpy as np
sys.path.insert(0,str(ROOT/'cuspa/python'))
import torch,cuspa

def main():
 out=ROOT/'results/module5_data';ref=ROOT/'results/module5_gb10_reference'
 out.mkdir(exist_ok=True);ref.mkdir(exist_ok=True)
 P,N,G=50000,2000000,500;rng=np.random.default_rng(20261123)
 cols=int(np.ceil(np.sqrt(P)));rows=int(np.ceil(P/cols));angles=np.arange(9)*np.pi/4
 centers=np.column_stack((np.arange(P)%cols,np.arange(P)//cols)).astype(np.float64)*10+5
 radial=rng.uniform(3.2,4.3,(P,8));radial=np.concatenate([radial,radial[:,:1]],axis=1)
 polys=(centers[:,None,:]+radial[:,:,None]*np.stack((np.cos(angles),np.sin(angles)),axis=1)).astype(np.float64);polys[:,-1]=polys[:,0]
 points=(rng.uniform(size=(N,2))*[cols*10,rows*10]).astype(np.float64);genes=rng.integers(G,size=N,dtype=np.int32)
 part=np.arange(P+1,dtype=np.int32);ring=part*9
 arrays={'polys':polys,'points':points,'genes':genes,'part':part,'ring':ring};meta={'seed':20261123,'cells':P,'points':N,'genes':G,'endianness':'little','arrays':{}}
 for name,a in arrays.items():
  path=out/f'{name}.bin';a.tofile(path);meta['arrays'][name]={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 gp=torch.from_numpy(polys.reshape(-1,2)).cuda();pp=torch.from_numpy(points).cuda();gg=torch.from_numpy(genes).cuda();pa=torch.from_numpy(part).cuda();rr=torch.from_numpy(ring).cuda()
 p=cuspa.Polygons(pa,rr,gp);p.ensure_index();ix=p.index
 params=[P,N,G,ix.nx,ix.ny,ix.origin_x,ix.origin_y,ix.cell_size_x,ix.cell_size_y]
 (out/'params.txt').write_text(' '.join(str(x) if isinstance(x,int) else format(x,'.17g') for x in params)+'\n');meta['grid_params']=params
 (out/'header.json').write_text(json.dumps(meta,indent=2)+'\n')
 assignments=cuspa.tl.assign_points(pp,p)
 ip=torch.empty(P+1,device='cuda',dtype=torch.int32);indices=torch.empty(N,device='cuda',dtype=torch.int32);values=torch.empty_like(indices)
 k=cuspa._core.aggregate_to_cells(assignments,gg,P,ip,indices,values,0);torch.cuda.synchronize()
 stats={'device':torch.cuda.get_device_name(),'nnz':k,'matched_points':int((assignments>=0).sum()),'hashes':{}}
 for name,a in {'assignment':assignments,'indptr':ip,'indices':indices[:k],'data':values[:k]}.items():
  path=ref/f'{name}.bin';a.cpu().numpy().astype('<i4').tofile(path);stats['hashes'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
 old=json.loads((ROOT/'results/module3_results.json').read_text())
 assert k==old['matrix_nnz'] and stats['matched_points']==old['matched_points']
 (ref/'reference.json').write_text(json.dumps(stats,indent=2)+'\n')
 print(json.dumps({'params':params,**stats},indent=2))
if __name__=='__main__':main()
