// MIT original FP64 radix-2 FFT + FX correlator. NOT NVIDIA cuPhoton source.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define GP(x) do{auto e=(x);if(e!=hipSuccess){fprintf(stderr,"HIP %s\n",hipGetErrorString(e));exit(2);}}while(0)
#define MAL hipMalloc
#define FREE hipFree
#define CP hipMemcpy
#define H2D hipMemcpyHostToDevice
#define D2H hipMemcpyDeviceToHost
#define SYNC hipDeviceSynchronize
#else
#include <cuda_runtime.h>
#define GP(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);}}while(0)
#define MAL cudaMalloc
#define FREE cudaFree
#define CP cudaMemcpy
#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define SYNC cudaDeviceSynchronize
#endif
struct C{double r,i;};using Clock=std::chrono::steady_clock;double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
__global__ void fft1024(const C*in,C*out){__shared__ C s[1024];int j=threadIdx.x;unsigned r=0,v=j;for(int k=0;k<10;k++){r=(r<<1)|(v&1);v>>=1;}s[r]=in[blockIdx.x*1024+j];__syncthreads();for(int len=2;len<=1024;len*=2){if(j<512){int group=j/(len/2),k=j%(len/2),a=group*len+k,b=a+len/2;double angle=-6.283185307179586476925286766559*k/len;double wr=cos(angle),wi=sin(angle);C u=s[a],v=s[b];double vr=wr*v.r-wi*v.i,vi=wr*v.i+wi*v.r;s[a]={u.r+vr,u.i+vi};s[b]={u.r-vr,u.i-vi};}__syncthreads();}out[blockIdx.x*1024+j]=s[j];}
__global__ void correlate(const C*s,C*v,const int*ab,int frames){int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=120*1024)return;int base=q/1024,ch=q%1024,a=ab[base*2],b=ab[base*2+1];double re=0,im=0;for(int t=0;t<frames;t++){C x=s[(t*16+a)*1024+ch],y=s[(t*16+b)*1024+ch];re+=x.r*y.r+x.i*y.i;im+=x.i*y.r-x.r*y.i;}v[q]={re/frames,im/frames};}
int main(int argc,char**argv){if(argc<3)return 2;std::string data=argv[1],pre=argv[2];const int B=64,F=64,A=16,N=1024;std::ifstream input(data+"/voltage.bin",std::ios::binary);if(!input)return 3;std::ofstream output(pre+"_vis.bin",std::ios::binary);std::vector<C>h(F*A*N),hv(120*N);C *di,*ds,*dv;int *dab;std::vector<int>ab;for(int a=0;a<A;a++)for(int b=a+1;b<A;b++){ab.push_back(a);ab.push_back(b);}GP(MAL((void**)&di,h.size()*sizeof(C)));GP(MAL((void**)&ds,h.size()*sizeof(C)));GP(MAL((void**)&dv,hv.size()*sizeof(C)));GP(MAL((void**)&dab,ab.size()*sizeof(int)));GP(CP(dab,ab.data(),ab.size()*sizeof(int),H2D));GP(SYNC());double ft=0,xt=0,copy=0,io=0;auto all=Clock::now();
for(int bin=0;bin<B;bin++){auto t=Clock::now();input.read((char*)h.data(),h.size()*sizeof(C));if(!input)return 4;io+=elapsed(t);t=Clock::now();GP(CP(di,h.data(),h.size()*sizeof(C),H2D));GP(SYNC());copy+=elapsed(t);t=Clock::now();fft1024<<<F*A,1024>>>(di,ds);GP(SYNC());ft+=elapsed(t);t=Clock::now();correlate<<<(120*N+127)/128,128>>>(ds,dv,dab,F);GP(SYNC());xt+=elapsed(t);t=Clock::now();GP(CP(hv.data(),dv,hv.size()*sizeof(C),D2H));GP(SYNC());copy+=elapsed(t);t=Clock::now();output.write((char*)hv.data(),hv.size()*sizeof(C));io+=elapsed(t);if(bin%8==0){printf("timebin %d/64\n",bin);fflush(stdout);}}
std::ofstream j(pre+"_timings.json");j.precision(17);j<<"{\"FFT_seconds\":"<<ft<<",\"X_seconds\":"<<xt<<",\"transfer_seconds\":"<<copy<<",\"IO_seconds\":"<<io<<",\"total_seconds\":"<<elapsed(all)<<",\"antennas\":16,\"frames\":4096,\"FFT_length\":1024,\"baselines\":120,\"time_bins\":64,\"precision\":\"complex128\"}";GP(FREE(di));GP(FREE(ds));GP(FREE(dv));GP(FREE(dab));return 0;}
