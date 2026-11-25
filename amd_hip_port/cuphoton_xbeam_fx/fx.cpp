// SPDX-License-Identifier: MIT
// Original FP64 FX correlator, shared CUDA/HIP source. Not NVIDIA cuPhoton code.
// Variants: v0 original trig/per-baseline control; v1 table/register reuse;
// v2 table/512-thread FFT and shared channel panel. Every sum preserves t order.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define RT(x) hip##x
using Prop=hipDeviceProp_t;
#else
#include <cuda_runtime.h>
#define RT(x) cuda##x
using Prop=cudaDeviceProp;
#endif
#define GP(a) do{auto e=(a);if(e!=RT(Success))throw std::runtime_error(std::string(#a)+": "+RT(GetErrorString)(e));}while(0)
struct C{double r,i;};using Clock=std::chrono::steady_clock;
constexpr int B=64,F=64,A=16,N=1024,NB=120;
__global__ void fft_original(C const*in,C*out){__shared__ C s[N];int j=threadIdx.x;unsigned r=0,v=j;for(int k=0;k<10;++k){r=(r<<1)|(v&1);v>>=1;}s[r]=in[size_t(blockIdx.x)*N+j];__syncthreads();for(int len=2;len<=N;len*=2){if(j<512){int group=j/(len/2),k=j%(len/2),a=group*len+k,b=a+len/2;double angle=-6.283185307179586476925286766559*k/len;double wr=cos(angle),wi=sin(angle);C u=s[a],v=s[b];double vr=wr*v.r-wi*v.i,vi=wr*v.i+wi*v.r;s[a]={u.r+vr,u.i+vi};s[b]={u.r-vr,u.i-vi};}__syncthreads();}out[size_t(blockIdx.x)*N+j]=s[j];}
template<int THREADS>__global__ void fft_table(C const*in,C*out,C const*twiddle){__shared__ C s[N];int j=threadIdx.x;for(int jj=j;jj<N;jj+=THREADS){unsigned r=0,v=jj;for(int k=0;k<10;++k){r=(r<<1)|(v&1);v>>=1;}s[r]=in[size_t(blockIdx.x)*N+jj];}__syncthreads();
#pragma unroll
 for(int len=2;len<=N;len*=2){if(j<512){int group=j/(len/2),k=j%(len/2),a=group*len+k,b=a+len/2;C w=twiddle[k*(N/len)],u=s[a],v=s[b];double vr=w.r*v.r-w.i*v.i,vi=w.r*v.i+w.i*v.r;s[a]={u.r+vr,u.i+vi};s[b]={u.r-vr,u.i-vi};}__syncthreads();}for(int jj=j;jj<N;jj+=THREADS)out[size_t(blockIdx.x)*N+jj]=s[jj];}
__global__ void correlate_original(C const*s,C*v,int const*ab){int q=blockIdx.x*blockDim.x+threadIdx.x,bin=blockIdx.z;if(q>=NB*N)return;int base=q/N,ch=q%N,a=ab[base*2],b=ab[base*2+1];double re=0,im=0;size_t offset=size_t(bin)*F*A*N;for(int t=0;t<F;++t){C x=s[offset+(t*A+a)*N+ch],y=s[offset+(t*A+b)*N+ch];re+=x.r*y.r+x.i*y.i;im+=x.i*y.r-x.r*y.i;}v[size_t(bin)*NB*N+q]={re/F,im/F};}
__global__ void correlate_register(C const*s,C*v){int ch=blockIdx.x*blockDim.x+threadIdx.x,a=blockIdx.y,bin=blockIdx.z;if(ch>=N)return;double re[A]={0},im[A]={0};size_t offset=size_t(bin)*F*A*N;for(int t=0;t<F;++t){C x=s[offset+(t*A+a)*N+ch];
#pragma unroll
 for(int b=0;b<A;++b){if(b>a){C y=s[offset+(t*A+b)*N+ch];re[b]+=x.r*y.r+x.i*y.i;im[b]+=x.i*y.r-x.r*y.i;}}}
#pragma unroll
 for(int b=0;b<A;++b){if(b>a){int pair=a*(31-a)/2+(b-a-1);v[size_t(bin)*NB*N+pair*N+ch]={re[b]/F,im[b]/F};}}}
// 2 adjacent channels x all64frames x16antennas =32KiB shared panel.
// Each spectral sample loaded once per panel; full baselines reuse the panel.
__global__ void correlate_panel(C const*s,C*v,int const*ab){constexpr int CH=2;__shared__ C panel[F*A*CH];int tid=threadIdx.x,first=blockIdx.x*CH,bin=blockIdx.z;size_t offset=size_t(bin)*F*A*N;for(int k=tid;k<F*A*CH;k+=256){int ta=k/CH,ch=k%CH;panel[k]=s[offset+size_t(ta)*N+first+ch];}__syncthreads();if(tid<NB*CH){int pair=tid/CH,ch=tid%CH,a=ab[2*pair],b=ab[2*pair+1];double re=0,im=0;
#pragma unroll 4
 for(int t=0;t<F;++t){C x=panel[(t*A+a)*CH+ch],y=panel[(t*A+b)*CH+ch];re+=x.r*y.r+x.i*y.i;im+=x.i*y.r-x.r*y.i;}v[size_t(bin)*NB*N+pair*N+first+ch]={re/F,im/F};}}
template<class T>struct Mem{T*p=nullptr;explicit Mem(size_t n){GP(RT(Malloc)(&p,n*sizeof(T)));}~Mem(){RT(Free)(p);}};
struct Measurement{double wall,event;};
template<class Fn>Measurement measure(Fn fn){RT(Event_t)a,b;GP(RT(EventCreate)(&a));GP(RT(EventCreate)(&b));GP(RT(DeviceSynchronize)());auto start=Clock::now();GP(RT(EventRecord)(a,0));fn();GP(RT(EventRecord)(b,0));GP(RT(EventSynchronize)(b));double wall=std::chrono::duration<double>(Clock::now()-start).count();float ms=0;GP(RT(EventElapsedTime)(&ms,a,b));GP(RT(EventDestroy)(a));GP(RT(EventDestroy)(b));return{wall,ms*.001};}
void array(std::ostream&s,std::vector<double>const&a){s<<'[';for(size_t i=0;i<a.size();++i){if(i)s<<',';s<<a[i];}s<<']';}
double median(std::vector<double>a){std::sort(a.begin(),a.end());return a[a.size()/2];}
int main(int argc,char**argv){try{if(argc<3)throw std::runtime_error("usage fx DATA_DIR OUTPUT_DIR [repeat>=3]");std::string data=argv[1],out=argv[2];int repeats=argc>3?std::stoi(argv[3]):7;int warmups=argc>4?std::stoi(argv[4]):5;if(warmups<1)throw std::runtime_error("need warmup");if(repeats<3)throw std::runtime_error("need >=3repeats");std::filesystem::create_directories(out);auto all=Clock::now(),start=all;std::vector<C>h(size_t(B)*F*A*N),hv(size_t(B)*NB*N),tw(N/2);std::ifstream input(data+"/voltage.bin",std::ios::binary);if(!input.read((char*)h.data(),h.size()*sizeof(C)))throw std::runtime_error("voltage read");double input_io=std::chrono::duration<double>(Clock::now()-start).count();start=Clock::now();for(int k=0;k<N/2;++k){double ang=-6.283185307179586476925286766559*k/N;tw[k]={cos(ang),sin(ang)};}double twiddle_setup=std::chrono::duration<double>(Clock::now()-start).count();std::vector<int>ab;for(int a=0;a<A;++a)for(int b=a+1;b<A;++b){ab.push_back(a);ab.push_back(b);}Mem<C>di(h.size()),ds(h.size()),dv(hv.size()),dt(tw.size());Mem<int>dab(ab.size());Prop prop{};GP(RT(GetDeviceProperties)(&prop,0));GP(RT(DeviceSynchronize)());start=Clock::now();GP(RT(Memcpy)(di.p,h.data(),h.size()*sizeof(C),RT(MemcpyHostToDevice)));GP(RT(Memcpy)(dt.p,tw.data(),tw.size()*sizeof(C),RT(MemcpyHostToDevice)));GP(RT(Memcpy)(dab.p,ab.data(),ab.size()*sizeof(int),RT(MemcpyHostToDevice)));GP(RT(DeviceSynchronize)());double h2d=std::chrono::duration<double>(Clock::now()-start).count();std::ofstream js(out+"/timings.json");js<<std::setprecision(17)<<"{\"device\":\""<<prop.name<<"\",\"complex_samples\":"<<h.size()<<",\"frames\":4096,\"channels\":1024,\"baselines\":120,\"time_bins\":64,\"input_IO_seconds\":"<<input_io<<",\"twiddle_setup_seconds\":"<<twiddle_setup<<",\"h2d_seconds\":"<<h2d<<",\"repeats\":"<<repeats<<",\"warmup_passes\":"<<warmups<<",\"variants\":{\n";
 for(int variant=0;variant<3;++variant){auto fft=[&]{if(variant==0)fft_original<<<B*F*A,1024>>>(di.p,ds.p);else if(variant==1)fft_table<1024><<<B*F*A,1024>>>(di.p,ds.p,dt.p);else fft_table<512><<<B*F*A,512>>>(di.p,ds.p,dt.p);GP(RT(GetLastError)());};auto x=[&]{if(variant==0)correlate_original<<<dim3((NB*N+127)/128,1,B),128>>>(ds.p,dv.p,dab.p);else if(variant==1)correlate_register<<<dim3((N+127)/128,15,B),128>>>(ds.p,dv.p);else correlate_panel<<<dim3(N/2,1,B),256>>>(ds.p,dv.p,dab.p);GP(RT(GetLastError)());};auto both=[&]{fft();x();};auto cold=measure(both);std::vector<double>warm{cold.wall};for(int w=1;w<warmups;++w)warm.push_back(measure(both).wall);std::vector<double>fw,fe,xw,xe,pw,pe;for(int r=0;r<repeats;++r){auto f=measure(fft),xx=measure(x),p=measure(both);fw.push_back(f.wall);fe.push_back(f.event);xw.push_back(xx.wall);xe.push_back(xx.event);pw.push_back(p.wall);pe.push_back(p.event);}
 start=Clock::now();GP(RT(Memcpy)(hv.data(),dv.p,hv.size()*sizeof(C),RT(MemcpyDeviceToHost)));GP(RT(DeviceSynchronize)());double d2h=std::chrono::duration<double>(Clock::now()-start).count();start=Clock::now();std::ofstream file(out+"/v"+std::to_string(variant)+"_vis.bin",std::ios::binary);file.write((char*)hv.data(),hv.size()*sizeof(C));file.close();double output_io=std::chrono::duration<double>(Clock::now()-start).count();if(variant)js<<",\n";js<<"\"v"<<variant<<"\":{\"description\":\""<<(variant==0?"same-harness original trig FFT and per-baseline X":variant==1?"precomputed twiddle FFT1024 and register reuse X":"precomputed FFT512 and 2-channel shared-panel X")<<"\",\"warmup_wall_seconds\":"<<cold.wall<<",\"warmup_pipeline_wall_seconds\":";array(js,warm);js<<",\"FFT_wall_seconds\":";array(js,fw);js<<",\"FFT_event_seconds\":";array(js,fe);js<<",\"X_wall_seconds\":";array(js,xw);js<<",\"X_event_seconds\":";array(js,xe);js<<",\"pipeline_wall_seconds\":";array(js,pw);js<<",\"pipeline_event_seconds\":";array(js,pe);js<<",\"FFT_wall_median_seconds\":"<<median(fw)<<",\"X_wall_median_seconds\":"<<median(xw)<<",\"pipeline_wall_median_seconds\":"<<median(pw)<<",\"d2h_seconds\":"<<d2h<<",\"output_IO_seconds\":"<<output_io<<",\"host_transfer_plus_pipeline_estimate_seconds\":"<<h2d+median(pw)+d2h<<"}";std::cout<<prop.name<<" v"<<variant<<" F="<<median(fw)<<" X="<<median(xw)<<" pipeline="<<median(pw)<<" seconds\n";}
 js<<"},\"all_variants_program_wall_seconds\":"<<std::chrono::duration<double>(Clock::now()-all).count()<<",\"scope\":\"All67108864 complex voltages and all7864320 visibilities. Resident input/output buffers, synchronized wall and separate events, configured complete-workload warmups then >=3 complete repeats. I/O and H2D/D2H separate; transfer+pipeline is estimate, not measured application end-to-end. No FMA/fastmath. CLEAN remains common CPU postprocessing.\"}";return 0;}catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 1;}}
