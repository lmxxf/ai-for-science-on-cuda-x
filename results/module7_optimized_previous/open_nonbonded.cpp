// SPDX-License-Identifier: MIT
// Original educational nonbonded implementation. NOT NVIDIA libcupo / OpenMM API.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
namespace cub=hipcub;
#define GPU 1
#define RT(x) hip##x
#define HD __host__ __device__
#define PROP hipDeviceProp_t
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#define GPU 1
#define RT(x) cuda##x
#define HD __host__ __device__
#define PROP cudaDeviceProp
#else
#define GPU 0
#define HD
#endif
using I=std::int32_t;using Clock=std::chrono::steady_clock;
constexpr double KE=1389.354576; // kJ mol^-1 Angstrom e^-2
struct Config{int n,nc,cells;double box,rc,erf,cell,krf,crf;};
HD int wrap(int c,int n){return c<0?c+n:c>=n?c-n:c;}
HD int cid(double const*a,Config const&c){int x=(int)(a[0]/c.cell),y=(int)(a[1]/c.cell),z=(int)(a[2]/c.cell);x=x<c.nc?x:c.nc-1;y=y<c.nc?y:c.nc-1;z=z<c.nc?z:c.nc-1;return (z*c.nc+y)*c.nc+x;}
HD double mic(double x,double L){return x-L*floor(x/L+0.5);}
HD void pair(double const*a,double const*b,Config const&c,double &e,double &fx,double &fy,double &fz){
 double dx=mic(a[0]-b[0],c.box),dy=mic(a[1]-b[1],c.box),dz=mic(a[2]-b[2],c.box);double r2=dx*dx+dy*dy+dz*dz;if(r2>=c.rc*c.rc)return;
 double ri=1.0/sqrt(r2),r2i=1.0/r2,sigma=0.5*(a[4]+b[4]),eps=sqrt(a[5]*b[5]);double t=sigma*sigma*r2i;double s6=t*t*t,s12=s6*s6,qq=KE*a[3]*b[3];
 e+=4*eps*(s12-s6)+qq*(ri+c.krf*r2-c.crf);
 double f=24*eps*(2*s12-s6)*r2i+qq*(ri*r2i-2*c.krf);fx+=f*dx;fy+=f*dy;fz+=f*dz;
}
HD void one_atom(int i,double const*atoms,I const*offsets,I const*sorted,Config const&c,double &e,double &fx,double &fy,double &fz){
 auto a=atoms+6*i;int cell=cid(a,c),cx=cell%c.nc,cy=(cell/c.nc)%c.nc,cz=cell/(c.nc*c.nc);e=fx=fy=fz=0;
 for(int dz=-1;dz<=1;++dz)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){int cc=(wrap(cz+dz,c.nc)*c.nc+wrap(cy+dy,c.nc))*c.nc+wrap(cx+dx,c.nc);for(int k=offsets[cc];k<offsets[cc+1];++k){int j=sorted[k];if(j!=i)pair(a,atoms+6*j,c,e,fx,fy,fz);}}
 e*=0.5; // each pair was visited in both atom-centered directions
}
void cpu_index(std::vector<double>const&a,Config const&c,std::vector<I>&off,std::vector<I>&ids){std::fill(off.begin(),off.end(),0);for(int i=0;i<c.n;++i)off[cid(a.data()+6*i,c)+1]++;for(int i=0;i<c.cells;++i)off[i+1]+=off[i];auto cur=off;for(int i=0;i<c.n;++i)ids[cur[cid(a.data()+6*i,c)]++]=i;}
#if GPU
#define CHECK(expr) do{auto err=(expr);if(err!=RT(Success))throw std::runtime_error(std::string(#expr)+": "+RT(GetErrorString)(err));}while(0)
template<class T>struct Mem{T*p=nullptr;size_t n;explicit Mem(size_t n):n(n){CHECK(RT(Malloc)(&p,n*sizeof(T)));}~Mem(){RT(Free)(p);}Mem(Mem const&)=delete;};
__global__ void bucket(double const*a,Config c,I*keys,I*ids,I*counts){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.n)return;int k=cid(a+6*i,c);keys[i]=k;ids[i]=i;atomicAdd(counts+k,1);}
__global__ void evaluate(double const*a,I const*off,I const*ids,Config c,double*energy,double*forces){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.n)return;double e,x,y,z;one_atom(i,a,off,ids,c,e,x,y,z);energy[i]=e;forces[3*i]=x;forces[3*i+1]=y;forces[3*i+2]=z;}
#endif
std::string path(std::string a,std::string b){return (std::filesystem::path(a)/b).string();}
template<class T>std::vector<T>load(std::string f,size_t n){std::vector<T>a(n);std::ifstream s(f,std::ios::binary);if(!s.read((char*)a.data(),n*sizeof(T)))throw std::runtime_error("read "+f);return a;}
template<class T>void save(std::string f,std::vector<T>const&a){std::ofstream s(f,std::ios::binary);if(!s.write((char*)a.data(),a.size()*sizeof(T)))throw std::runtime_error("write "+f);}
void arr(std::ostream&o,std::vector<double>const&a){o<<'[';for(size_t i=0;i<a.size();++i){if(i)o<<',';o<<a[i];}o<<']';}
double med(std::vector<double>a){std::sort(a.begin(),a.end());return a[a.size()/2];}
template<class F>double elapsed(F f){auto t=Clock::now();f();
#if GPU
 CHECK(RT(DeviceSynchronize)());
#endif
 return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
int main(int argc,char**argv){try{if(argc<3){std::cerr<<"usage open_nonbonded INPUT_DIR OUTPUT_DIR [repeats=5]\n";return 2;}std::string in=argv[1],out=argv[2];int repeats=argc>3?std::stoi(argv[3]):5;Config c{};std::ifstream p(path(in,"params.txt"));if(!(p>>c.n>>c.box>>c.rc>>c.erf))throw std::runtime_error("params");c.nc=(int)floor(c.box/c.rc);if(c.nc<3)throw std::runtime_error("require >=3 cells per dimension to avoid duplicate 27-cell visits");c.cells=c.nc*c.nc*c.nc;c.cell=c.box/c.nc;c.krf=(c.erf-1)/(2*c.erf+1)/(c.rc*c.rc*c.rc);c.crf=3*c.erf/(2*c.erf+1)/c.rc;
 auto atoms=load<double>(path(in,"atoms.bin"),size_t(c.n)*6);std::filesystem::create_directories(out);std::vector<double>energy(c.n),forces(size_t(c.n)*3),ti,tf,tp;double h2d=0,d2h=0;std::string device="single-thread compiled C++ CPU";
#if GPU
 PROP prop{};CHECK(RT(GetDeviceProperties)(&prop,0));device=prop.name;
 Mem<double>da(atoms.size()),de(c.n),df(forces.size());Mem<I>key(c.n),key_sorted(c.n),ids(c.n),sorted(c.n),off(c.cells+1);CHECK(RT(DeviceSynchronize)());h2d=elapsed([&]{CHECK(RT(Memcpy)(da.p,atoms.data(),atoms.size()*8,RT(MemcpyHostToDevice)));});
 size_t sb=0,scanb=0;CHECK(cub::DeviceRadixSort::SortPairs(nullptr,sb,key.p,key_sorted.p,ids.p,sorted.p,c.n));CHECK(cub::DeviceScan::ExclusiveSum(nullptr,scanb,off.p,off.p,c.cells+1));Mem<char>scratch(std::max(sb,scanb));
 auto index=[&]{CHECK(RT(MemsetAsync)(off.p,0,size_t(c.cells+1)*4,0));bucket<<<(c.n+255)/256,256>>>(da.p,c,key.p,ids.p,off.p);CHECK(RT(GetLastError)());CHECK(cub::DeviceRadixSort::SortPairs(scratch.p,sb,key.p,key_sorted.p,ids.p,sorted.p,c.n));CHECK(cub::DeviceScan::ExclusiveSum(scratch.p,scanb,off.p,off.p,c.cells+1));};
 auto force=[&]{evaluate<<<(c.n+127)/128,128>>>(da.p,off.p,sorted.p,c,de.p,df.p);CHECK(RT(GetLastError)());};
#else
 std::vector<I>off(c.cells+1),sorted(c.n);auto index=[&]{cpu_index(atoms,c,off,sorted);};auto force=[&]{for(int i=0;i<c.n;++i)one_atom(i,atoms.data(),off.data(),sorted.data(),c,energy[i],forces[3*i],forces[3*i+1],forces[3*i+2]);};
#endif
 auto pipeline=[&]{index();force();};double cold=elapsed(pipeline);for(int i=0;i<2;++i)pipeline();
#if GPU
 CHECK(RT(DeviceSynchronize)());
#endif
 for(int r=0;r<repeats;++r){ti.push_back(elapsed(index));tf.push_back(elapsed(force));tp.push_back(elapsed(pipeline));}
#if GPU
 d2h=elapsed([&]{CHECK(RT(Memcpy)(energy.data(),de.p,energy.size()*8,RT(MemcpyDeviceToHost)));CHECK(RT(Memcpy)(forces.data(),df.p,forces.size()*8,RT(MemcpyDeviceToHost)));});
#endif
 save(path(out,"energy.bin"),energy);save(path(out,"forces.bin"),forces);long double total=0;for(double e:energy)total+=e;
 std::ofstream j(path(out,"timings.json"));j<<std::setprecision(17)<<"{\n\"device\":\""<<device<<"\",\n\"atoms\":"<<c.n<<",\n\"box_angstrom\":"<<c.box<<",\n\"cutoff_angstrom\":"<<c.rc<<",\n\"rf_dielectric\":"<<c.erf<<",\n\"grid_side\":"<<c.nc<<",\n\"energy_kj_per_mol\":"<<(double)total<<",\n\"h2d_ms\":"<<h2d<<",\n\"d2h_ms\":"<<d2h<<",\n\"cold_pipeline_ms\":"<<cold<<",\n\"index_ms\":";arr(j,ti);j<<",\n\"force_ms\":";arr(j,tf);j<<",\n\"pipeline_ms\":";arr(j,tp);j<<",\n\"index_median_ms\":"<<med(ti)<<",\n\"force_median_ms\":"<<med(tf)<<",\n\"pipeline_median_ms\":"<<med(tp)<<",\n\"evaluations_per_s\":"<<1000/med(tp)<<",\n\"warmups\":2,\n\"repeats\":"<<repeats<<",\n\"precision\":\"float64; contraction disabled; no fastmath\",\n\"scope\":\"nonbonded evaluation, not complete MD step or PME; inputs resident, transfers separate\"\n}\n";std::cout<<device<<" N="<<c.n<<" E="<<(double)total<<" pipeline_ms="<<med(tp)<<'\n';return 0;}catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 1;}}
