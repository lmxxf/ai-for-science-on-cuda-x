// MIT original GPU-resident restarted Block-Davidson educational prototype.
// Each block owns a k-point and executes all iterations, Jacobi and MGS on device.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define GP(x) do{auto _gpu_status=(x);if(_gpu_status!=hipSuccess){fprintf(stderr,"HIP %s\n",hipGetErrorString(_gpu_status));exit(2);}}while(0)
#define MAL hipMalloc
#define FREE hipFree
#define CP hipMemcpy
#define H2D hipMemcpyHostToDevice
#define D2H hipMemcpyDeviceToHost
#define SYNC hipDeviceSynchronize
#else
#include <cuda_runtime.h>
#define GP(x) do{auto _gpu_status=(x);if(_gpu_status!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(_gpu_status));exit(2);}}while(0)
#define MAL cudaMalloc
#define FREE cudaFree
#define CP cudaMemcpy
#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define SYNC cudaDeviceSynchronize
#endif
using Clock=std::chrono::steady_clock;double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
constexpr int N=384,M=32,K=16,KEEP=24,NT=256;
__device__ double reduce_sum(double x,double*red){int t=threadIdx.x;red[t]=x;__syncthreads();for(int s=NT/2;s;s/=2){if(t<s)red[t]+=red[t+s];__syncthreads();}return red[0];}
__global__ void solve(const double*Hall,double*Vall,double*HVall,double*Nextall,double*Rall,double*eall,double*xall,double*histall,int*iters,int first,int maxit){
 int b=blockIdx.x+first,t=threadIdx.x;const double*H=Hall+(size_t)b*N*N;double*V=Vall+(size_t)b*N*M,*HV=HVall+(size_t)b*N*M,*Next=Nextall+(size_t)b*N*M,*R=Rall+(size_t)b*N*KEEP,*outx=xall+(size_t)b*N*K,*hist=histall+(size_t)b*maxit;double*outE=eall+b*K;
 __shared__ double A[M*M],Q[M*M],C[M*KEEP],eval[KEEP],red[NT],co,si,largest,maxres;
 __shared__ int order[M],selected[M],converged,sweep_done;
 if(t==0){for(int j=0;j<M;j++){int best=-1;double emin=1e300;for(int i=0;i<N;i++){bool used=false;for(int k=0;k<j;k++)if(selected[k]==i)used=true;if(!used&&H[i*N+i]<emin){best=i;emin=H[i*N+i];}}selected[j]=best;}}
 __syncthreads();for(int q=t;q<N*M;q+=NT)V[q]=(q/M==selected[q%M]?1.:0.);__syncthreads();
 for(int iteration=0;iteration<maxit;iteration++){
  // Hamiltonian action; all vectors and H remain in global device memory.
  for(int q=t;q<N*M;q+=NT){int i=q/M,j=q%M;double z=0;for(int k=0;k<N;k++)z+=H[i*N+k]*V[k*M+j];HV[q]=z;}__syncthreads();
  for(int q=t;q<M*M;q+=NT){int i=q/M,j=q%M;double z=0;for(int k=0;k<N;k++)z+=V[k*M+i]*HV[k*M+j];A[q]=z;Q[q]=(i==j?1.:0.);}__syncthreads();
  for(int q=t;q<M*M;q+=NT){int i=q/M,j=q%M;if(i<j){double z=.5*(A[i*M+j]+A[j*M+i]);A[i*M+j]=A[j*M+i]=z;}}__syncthreads();
  // Shared-memory cyclic Jacobi, 32 rows updated in parallel at each rotation.
  for(int sweep=0;sweep<35;sweep++){
   if(t==0)largest=0;__syncthreads();
   for(int p=0;p<M;p++)for(int q=p+1;q<M;q++){
    if(t==0){double a=A[p*M+q];largest=fmax(largest,fabs(a));if(fabs(a)<1e-16){co=1;si=0;}else{double tau=(A[q*M+q]-A[p*M+p])/(2*a),v=(tau>=0?1.:-1.)/(fabs(tau)+sqrt(1+tau*tau));co=1/sqrt(1+v*v);si=v*co;A[p*M+p]-=v*a;A[q*M+q]+=v*a;A[p*M+q]=A[q*M+p]=0;}}
    __syncthreads();if(t<M){double c=co,s=si;if(t!=p&&t!=q){double x=A[t*M+p],y=A[t*M+q];A[t*M+p]=A[p*M+t]=c*x-s*y;A[t*M+q]=A[q*M+t]=s*x+c*y;}double x=Q[t*M+p],y=Q[t*M+q];Q[t*M+p]=c*x-s*y;Q[t*M+q]=s*x+c*y;}__syncthreads();
   }
   if(t==0)sweep_done=(largest<5e-16);__syncthreads();if(sweep_done)break;
  }
  if(t==0){for(int j=0;j<M;j++)order[j]=j;for(int j=0;j<M;j++)for(int k=j+1;k<M;k++)if(A[order[k]*M+order[k]]<A[order[j]*M+order[j]]){int z=order[j];order[j]=order[k];order[k]=z;}for(int j=0;j<KEEP;j++){eval[j]=A[order[j]*M+order[j]];if(j<K)outE[j]=eval[j];}}
  __syncthreads();for(int q=t;q<M*KEEP;q+=NT)C[q]=Q[(q/KEEP)*M+order[q%KEEP]];__syncthreads();
  for(int q=t;q<N*KEEP;q+=NT){int i=q/KEEP,j=q%KEEP;double x=0,hx=0;for(int k=0;k<M;k++){x+=V[i*M+k]*C[k*KEEP+j];hx+=HV[i*M+k]*C[k*KEEP+j];}double r=hx-eval[j]*x;if(j<K)outx[i*K+j]=x;Next[i*M+j]=x;R[q]=r;}
  __syncthreads();if(t<K){double z=0;for(int i=0;i<N;i++)z+=R[i*KEEP+t]*R[i*KEEP+t];red[t]=sqrt(z);}__syncthreads();if(t==0){maxres=0;for(int j=0;j<K;j++)maxres=fmax(maxres,isfinite(red[j])?red[j]:1e300);hist[iteration]=maxres;iters[b]=iteration+1;converged=maxres<1e-10;}__syncthreads();if(converged)break;
  // Thick restart retains 24 Ritz vectors (8 guard roots) plus 8 largest target residual directions.
  if(t==0){for(int j=0;j<K;j++)order[j]=j;for(int j=0;j<K;j++)for(int k=j+1;k<K;k++)if(red[order[k]]>red[order[j]]){int z=order[j];order[j]=order[k];order[k]=z;}}__syncthreads();
  for(int q=t;q<N*(M-KEEP);q+=NT){int i=q/(M-KEEP),j=q%(M-KEEP),root=order[j];double den=H[i*N+i]-eval[root];if(fabs(den)<.08)den=den<0?-.08:.08;Next[i*M+KEEP+j]=-R[i*KEEP+root]/den;}__syncthreads();
  // Two-pass modified Gram-Schmidt of all 32 retained/correction columns.
  for(int j=0;j<M;j++){
   for(int pass=0;pass<2;pass++)for(int k=0;k<j;k++){
    double local=0;for(int i=t;i<N;i+=NT)local+=Next[i*M+j]*Next[i*M+k];double dot=reduce_sum(local,red);for(int i=t;i<N;i+=NT)Next[i*M+j]-=dot*Next[i*M+k];__syncthreads();
   }
   double local=0;for(int i=t;i<N;i+=NT)local+=Next[i*M+j]*Next[i*M+j];double norm=sqrt(reduce_sum(local,red));
   if(norm<1e-13){
    // Deterministic completion vector for numerically dependent residual columns.
    for(int i=t;i<N;i+=NT)Next[i*M+j]=sin((i+1.)*(j+17.)*.12345);__syncthreads();
    for(int pass=0;pass<2;pass++)for(int k=0;k<j;k++){double z=0;for(int i=t;i<N;i+=NT)z+=Next[i*M+j]*Next[i*M+k];double dot=reduce_sum(z,red);for(int i=t;i<N;i+=NT)Next[i*M+j]-=dot*Next[i*M+k];__syncthreads();}
    local=0;for(int i=t;i<N;i+=NT)local+=Next[i*M+j]*Next[i*M+j];norm=sqrt(reduce_sum(local,red));
   }
   for(int i=t;i<N;i+=NT)Next[i*M+j]/=norm;__syncthreads();
  }
  for(int q=t;q<N*M;q+=NT)V[q]=Next[q];__syncthreads();
 }
}
int main(int argc,char**argv){if(argc<5){fprintf(stderr,"usage resident inputdir outprefix seq|batch maxiter\n");return 2;}std::string dir=argv[1],pre=argv[2],mode=argv[3];int maxit=atoi(argv[4]),B,n,k;std::ifstream meta(dir+"/params.txt");meta>>B>>n>>k;if(n!=N||k!=K)return 3;std::vector<double>H((size_t)B*N*N);std::ifstream in(dir+"/H.bin",std::ios::binary);in.read((char*)H.data(),H.size()*8);if(!in)return 4;double *dh,*dv,*dhv,*dn,*dr,*de,*dx,*dhi;int*di;auto begin=Clock::now();GP(MAL((void**)&dh,H.size()*8));GP(MAL((void**)&dv,(size_t)B*N*M*8));GP(MAL((void**)&dhv,(size_t)B*N*M*8));GP(MAL((void**)&dn,(size_t)B*N*M*8));GP(MAL((void**)&dr,(size_t)B*N*KEEP*8));GP(MAL((void**)&de,(size_t)B*K*8));GP(MAL((void**)&dx,(size_t)B*N*K*8));GP(MAL((void**)&dhi,(size_t)B*maxit*8));GP(MAL((void**)&di,B*4));GP(CP(dh,H.data(),H.size()*8,H2D));GP(SYNC());double setup=seconds(begin);auto t=Clock::now();
 if(mode=="seq"){for(int b=0;b<B;b++)solve<<<1,NT>>>(dh,dv,dhv,dn,dr,de,dx,dhi,di,b,maxit);}else solve<<<B,NT>>>(dh,dv,dhv,dn,dr,de,dx,dhi,di,0,maxit);
 GP(SYNC());double device=seconds(t);t=Clock::now();std::vector<double>e(B*K),x((size_t)B*N*K),h((size_t)B*maxit);std::vector<int>it(B);GP(CP(e.data(),de,e.size()*8,D2H));GP(CP(x.data(),dx,x.size()*8,D2H));GP(CP(h.data(),dhi,h.size()*8,D2H));GP(CP(it.data(),di,it.size()*4,D2H));GP(SYNC());double download=seconds(t),total=seconds(begin);double maxres=0;int maxiter=0,miniter=maxit;for(int b=0;b<B;b++){maxres=std::max(maxres,h[b*maxit+it[b]-1]);maxiter=std::max(maxiter,it[b]);miniter=std::min(miniter,it[b]);}std::ofstream ef(pre+"_eigenvalues.bin",std::ios::binary);ef.write((char*)e.data(),e.size()*8);std::ofstream xf(pre+"_eigenvectors.bin",std::ios::binary);xf.write((char*)x.data(),x.size()*8);std::ofstream hf(pre+"_history.csv");hf<<"iteration,max_residual\n";for(int j=0;j<maxiter;j++){double z=0;for(int b=0;b<B;b++)z=std::max(z,h[b*maxit+std::min(j,it[b]-1)]);hf<<j+1<<","<<z<<"\n";}
 std::ofstream jf(pre+"_timings.json");jf.precision(17);jf<<"{\"systems\":"<<B<<",\"mode\":\""<<mode<<"\",\"resident_kernel_seconds\":"<<device<<",\"setup_upload_seconds\":"<<setup<<",\"final_download_seconds\":"<<download<<",\"total_seconds\":"<<total<<",\"min_iterations\":"<<miniter<<",\"max_iterations\":"<<maxiter<<",\"max_residual\":"<<maxres<<",\"iteration_host_transfers\":0,\"iteration_host_eigensolves\":0,\"launch_count\":"<<(mode=="seq"?B:1)<<"}";printf("systems %d mode %s iterations %d..%d maxres %.12g GPU %.8gs total %.8gs\n",B,mode.c_str(),miniter,maxiter,maxres,device,total);GP(FREE(dh));GP(FREE(dv));GP(FREE(dhv));GP(FREE(dn));GP(FREE(dr));GP(FREE(de));GP(FREE(dx));GP(FREE(dhi));GP(FREE(di));return maxres<1e-7?0:5;}
