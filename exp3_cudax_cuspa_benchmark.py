#!/usr/bin/env python3
"""Original cuSpa CUDA kernels vs single-core compiled CPU grid+ray casting.
Synthetic spatial-omics-sized geometry, NOT a biological tissue dataset.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse, time, json, sys, statistics, subprocess, re
sys.path.insert(0,str(Path(__file__).resolve().parent/'results/module3_builddeps'))
import numpy as np
from numba import njit
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'cuspa/python'))

@njit(cache=True)
def cpu_index(polys, nx, ny, ox, oy, sx, sy):
    p=len(polys); aabb=np.empty((p,4),np.float64); counts=np.zeros(nx*ny,np.int32)
    for i in range(p):
        x0=polys[i,0,0];y0=polys[i,0,1];x1=x0;y1=y0
        for j in range(len(polys[i])):
            x0=min(x0,polys[i,j,0]);x1=max(x1,polys[i,j,0]);y0=min(y0,polys[i,j,1]);y1=max(y1,polys[i,j,1])
        aabb[i]=np.array([x0,y0,x1,y1])
        for gy in range(max(0,int((y0-oy)/sy)),min(ny-1,int((y1-oy)/sy))+1):
            for gx in range(max(0,int((x0-ox)/sx)),min(nx-1,int((x1-ox)/sx))+1): counts[gy*nx+gx]+=1
    offsets=np.zeros(nx*ny+1,np.int32)
    for i in range(nx*ny): offsets[i+1]=offsets[i]+counts[i]
    ids=np.empty(offsets[-1],np.int32); cursor=offsets[:-1].copy()
    for i in range(p):
        x0,y0,x1,y1=aabb[i]
        for gy in range(max(0,int((y0-oy)/sy)),min(ny-1,int((y1-oy)/sy))+1):
            for gx in range(max(0,int((x0-ox)/sx)),min(nx-1,int((x1-ox)/sx))+1):
                c=gy*nx+gx;ids[cursor[c]]=i;cursor[c]+=1
    return offsets,ids,aabb

@njit(cache=True)
def cpu_query(points,polys,offsets,ids,aabbs,nx,ny,ox,oy,sx,sy):
    out=np.full(len(points),-1,np.int32)
    for i in range(len(points)):
        x,y=points[i];gx=int(np.floor((x-ox)/sx));gy=int(np.floor((y-oy)/sy))
        if gx<0 or gy<0 or gx>=nx or gy>=ny:continue
        c=gy*nx+gx
        for k in range(offsets[c],offsets[c+1]):
            p=ids[k];x0,y0,x1,y1=aabbs[p]
            if x<x0 or x>x1 or y<y0 or y>y1:continue
            inside=False
            for j in range(len(polys[p])-1):
                ax,ay=polys[p,j];bx,by=polys[p,j+1]
                if (ay>y)!=(by>y) and x<(bx-ax)*(y-ay)/(by-ay)+ax:inside=not inside
            if inside:out[i]=p;break
    return out

def timed(fn, repeats=3):
    vals=[];out=None
    for _ in range(repeats):
        t=time.perf_counter();out=fn();vals.append(time.perf_counter()-t)
    return out,vals

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cells',type=int,default=50000);ap.add_argument('--points',type=int,default=2000000);ap.add_argument('--genes',type=int,default=500);args=ap.parse_args()
    import torch, cuspa
    from scipy.sparse import csr_matrix
    torch.set_num_threads(1)
    rng=np.random.default_rng(20261123); cols=int(np.ceil(np.sqrt(args.cells))); rows=int(np.ceil(args.cells/cols));angles=np.arange(9)*np.pi/4
    centers=np.column_stack((np.arange(args.cells)%cols,np.arange(args.cells)//cols)).astype(np.float64)*10+5
    radial=rng.uniform(3.2,4.3,(args.cells,8)); radial=np.concatenate([radial,radial[:,:1]],axis=1)
    polys=(centers[:,None,:]+radial[:,:,None]*np.stack((np.cos(angles),np.sin(angles)),axis=1)).astype(np.float64)
    polys[:,-1]=polys[:,0]
    points=(rng.uniform(size=(args.points,2))*[cols*10,rows*10]).astype(np.float64);genes=rng.integers(args.genes,size=args.points,dtype=np.int32)
    ab=np.column_stack((polys[:,:,0].min(1),polys[:,:,1].min(1),polys[:,:,0].max(1),polys[:,:,1].max(1)))
    nx=ny=int(np.ceil(np.sqrt(args.cells)));ox=float(ab[:,0].min());oy=float(ab[:,1].min());sx=float((ab[:,2].max()-ox)*(1+1e-6)/nx);sy=float((ab[:,3].max()-oy)*(1+1e-6)/ny)
    # JIT is measured separately and excluded from steady-state timings.
    t=time.perf_counter();ci=cpu_index(polys[:1],nx,ny,ox,oy,sx,sy);cpu_query(points[:1],polys[:1],*ci,nx,ny,ox,oy,sx,sy);jit=time.perf_counter()-t
    ci,cit=timed(lambda:cpu_index(polys,nx,ny,ox,oy,sx,sy));cpu,cqt=timed(lambda:cpu_query(points,polys,*ci,nx,ny,ox,oy,sx,sy))
    def cpu_aggregate():
        valid=cpu>=0;keys=cpu[valid].astype(np.int64)*args.genes+genes[valid];u,c=np.unique(keys,return_counts=True);return csr_matrix((c,(u//args.genes,u%args.genes)),shape=(args.cells,args.genes))
    cpu_csr,cat=timed(cpu_aggregate)
    torch.cuda.synchronize();t=time.perf_counter();gpoly=torch.from_numpy(polys.reshape(-1,2)).cuda();gp=torch.from_numpy(points).cuda();gg=torch.from_numpy(genes).cuda();part=torch.arange(args.cells+1,device='cuda',dtype=torch.int32);ring=part*9;torch.cuda.synchronize();transfer=time.perf_counter()-t
    def makepoly():return cuspa.Polygons(part,ring,gpoly)
    p=makepoly()
    def gpu_index():
        p.index=None;p.aabbs=None;p.ensure_index();torch.cuda.synchronize();return p.index
    _,git=timed(gpu_index)
    def gpu_query():
        o=cuspa.tl.assign_points(gp,p);torch.cuda.synchronize();return o
    gout,gqt=timed(gpu_query,5)
    def gpu_aggregate_only():
        ip=torch.empty(args.cells+1,device='cuda',dtype=torch.int32);ix=torch.empty(args.points,device='cuda',dtype=torch.int32);v=torch.empty_like(ix);k=cuspa._core.aggregate_to_cells(gout,gg,args.cells,ip,ix,v,0);torch.cuda.synchronize();return ip,ix[:k],v[:k]
    gc,gat=timed(gpu_aggregate_only,5)
    def gpu_public_pipeline():
        q=makepoly();ids=cuspa.tl.assign_points(gp,q);ip=torch.empty(args.cells+1,device='cuda',dtype=torch.int32);ix=torch.empty(args.points,device='cuda',dtype=torch.int32);v=torch.empty_like(ix);k=cuspa._core.aggregate_to_cells(ids,gg,args.cells,ip,ix,v,0);torch.cuda.synchronize();return ip,ix[:k],v[:k]
    public,gpt=timed(gpu_public_pipeline,3)
    output=gout.cpu().numpy();mismatches=int(np.count_nonzero(output!=cpu));gpu_csr=csr_matrix((gc[2].cpu().numpy(),gc[1].cpu().numpy(),gc[0].cpu().numpy()),shape=(args.cells,args.genes));matrix_diff=int((gpu_csr!=cpu_csr).nnz)
    assert mismatches==0,(mismatches,'assignment mismatch');assert matrix_diff==0,(matrix_diff,'count mismatch')
    result=dict(workload='seeded synthetic spatial-omics-scale polygons; not experimental biological observations',seed=20261123,cells=args.cells,points=args.points,genes=args.genes,vertices_per_polygon=8,matched_points=int((cpu>=0).sum()),matrix_nnz=int(cpu_csr.nnz),assignment_mismatches=mismatches,count_matrix_mismatches=matrix_diff,cpu='single-thread Numba compiled uniform-grid AABB + ray casting; NumPy sort/count CSR',gpu='original cuspa source compiled nvcc sm_121, native CUB sort/run-length CSR',cpu_jit_s=jit,host_to_device_s=transfer,cpu_index_s=cit,cpu_query_s=cqt,cpu_aggregate_s=cat,gpu_index_s=git,gpu_query_s=gqt,gpu_aggregate_s=gat,gpu_public_pipeline_s=gpt,gpu_entrypoint='cuspa.tl.assign_points + cuspa._core.aggregate_to_cells (high-level aggregation requires CuPy; native CUDA core supports PyTorch)')
    result['cpu_pipeline_median_s']=sum(map(statistics.median,[cit,cqt,cat]));result['gpu_pipeline_median_s']=statistics.median(gpt);result['pipeline_speedup']=result['cpu_pipeline_median_s']/result['gpu_pipeline_median_s'];result['query_speedup']=statistics.median(cqt)/statistics.median(gqt)
    result['host_to_device_plus_pipeline_estimate_s']=transfer+result['gpu_pipeline_median_s'];result['speedup_including_h2d_estimate']=result['cpu_pipeline_median_s']/result['host_to_device_plus_pipeline_estimate_s'];result['timing_scope']='device-resident steady state; H2D separate, D2H excluded; CPU sum of stage medians; JIT and CUDA compilation excluded'
    result['gpu']=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    out=ROOT/'results';out.mkdir(exist_ok=True);(out/'module3_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));np.savez_compressed(out/'module3_preview.npz',polys=polys[:cols*3],points=points[(points[:,0]<40)&(points[:,1]<30)],genes=genes[(points[:,0]<40)&(points[:,1]<30)])
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
    from render_module3 import render
    render()
if __name__=='__main__':main()
