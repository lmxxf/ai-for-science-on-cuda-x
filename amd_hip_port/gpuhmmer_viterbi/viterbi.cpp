// SPDX-License-Identifier: MIT
// Original profile-HMM Viterbi reference: match/insert/delete, glocal full domain.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define GPU 1
#define RT(x) hip##x
#define HD __host__ __device__
using DeviceProp=hipDeviceProp_t;
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#define GPU 1
#define RT(x) cuda##x
#define HD __host__ __device__
using DeviceProp=cudaDeviceProp;
#else
#define GPU 0
#define HD
#endif
using Clock=std::chrono::steady_clock;
constexpr int M=56;
HD double mx(double a,double b){return a>b?a:b;}
HD double score_one(uint8_t const* seq,int len,double const* em,double const* t){
 double pm[M+1],pi[M+1],pd[M+1];constexpr double neg=-1e100;
 for(int j=0;j<=M;++j){pm[j]=pi[j]=pd[j]=neg;}
 pm[0]=0;double best=neg;
 for(int i=0;i<len;++i){
  double dm=0,di=neg,dd=neg; // previous-row j=0: begin is free at each sequence position.
  double leftm=neg,leftd=neg;
  for(int j=1;j<=M;++j){
   double om=pm[j],oi=pi[j],od=pd[j];
   double a=mx(mx(dm+t[0],di+t[3]),dd+t[5])+em[(j-1)*20+seq[i]];
   double b=mx(om+t[1],oi+t[4]); // insert emission equals background: zero log odds.
   double c=mx(leftm+t[2],leftd+t[6]);
   pm[j]=a;pi[j]=b;pd[j]=c;dm=om;di=oi;dd=od;leftm=a;leftd=c;
  }
  best=mx(best,mx(pm[M],pd[M]));
 }
 return best;
}
#if GPU
#define CHECK(x) do{auto checked_status=(x);if(checked_status!=RT(Success))throw std::runtime_error(RT(GetErrorString)(checked_status));}while(0)
__global__ void viterbi(uint8_t const*s,int const*l,int n,int stride,double const*e,double const*t,double*out){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=score_one(s+(size_t)i*stride,l[i],e,t);}
template<class T>struct Mem{T*p=nullptr;Mem(size_t n){CHECK(RT(Malloc)(&p,n*sizeof(T)));}~Mem(){RT(Free)(p);}};
#endif
std::string path(std::string a,std::string b){return (std::filesystem::path(a)/b).string();}
template<class T>std::vector<T>load(std::string f,size_t n){std::vector<T>a(n);std::ifstream s(f,std::ios::binary);if(!s.read((char*)a.data(),n*sizeof(T)))throw std::runtime_error("read "+f);return a;}
template<class F>double wall(F f){auto start=Clock::now();f();
#if GPU
 CHECK(RT(DeviceSynchronize)());
#endif
 return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
double median(std::vector<double>x){std::sort(x.begin(),x.end());return x[x.size()/2];}
int main(int argc,char**argv){try{
 if(argc<3)throw std::runtime_error("usage viterbi INPUT_DIR OUTPUT_DIR [repeats=3]");std::string in=argv[1],out=argv[2];int repeats=argc>3?std::stoi(argv[3]):3;
 int n,stride;std::ifstream p(path(in,"params.txt"));if(!(p>>n>>stride)||n<1||stride<1||stride>256)throw std::runtime_error("bad dimensions");
 auto seq=load<uint8_t>(path(in,"sequences.bin"),(size_t)n*stride);auto lengths=load<int>(path(in,"lengths.bin"),n);auto em=load<double>(path(in,"emissions.bin"),M*20);auto trans=load<double>(path(in,"transitions.bin"),7);std::vector<double>scores(n);std::string device="DGX Spark single-thread C++ CPU";double h2d=0,d2h=0;std::vector<double>times;
#if GPU
 DeviceProp prop{};CHECK(RT(GetDeviceProperties)(&prop,0));device=prop.name;Mem<uint8_t>s(seq.size());Mem<int>l(n);Mem<double>e(em.size()),t(trans.size()),o(n);
 h2d=wall([&]{CHECK(RT(Memcpy)(s.p,seq.data(),seq.size(),RT(MemcpyHostToDevice)));CHECK(RT(Memcpy)(l.p,lengths.data(),n*4,RT(MemcpyHostToDevice)));CHECK(RT(Memcpy)(e.p,em.data(),em.size()*8,RT(MemcpyHostToDevice)));CHECK(RT(Memcpy)(t.p,trans.data(),trans.size()*8,RT(MemcpyHostToDevice)));});
 auto run=[&]{viterbi<<<(n+63)/64,64>>>(s.p,l.p,n,stride,e.p,t.p,o.p);CHECK(RT(GetLastError)());};
#else
 auto run=[&]{for(int i=0;i<n;++i)scores[i]=score_one(seq.data()+(size_t)i*stride,lengths[i],em.data(),trans.data());};
#endif
 double cold=wall(run);for(int i=0;i<repeats;++i)times.push_back(wall(run));
#if GPU
 d2h=wall([&]{CHECK(RT(Memcpy)(scores.data(),o.p,n*8,RT(MemcpyDeviceToHost)));});
#endif
 std::filesystem::create_directories(out);std::ofstream ofs(path(out,"scores.bin"),std::ios::binary);ofs.write((char*)scores.data(),n*8);ofs.close();
 long long residues=0;for(auto x:lengths)residues+=x;
 std::ofstream j(path(out,"timings.json"));j<<std::setprecision(17)<<"{\"device\":\""<<device<<"\",\"sequences\":"<<n<<",\"residues\":"<<residues<<",\"profile_length\":56,\"score_dtype\":\"float64 log2 odds\",\"cold_ms\":"<<cold<<",\"h2d_ms\":"<<h2d<<",\"d2h_ms\":"<<d2h<<",\"score_ms\":[";for(int i=0;i<repeats;++i){if(i)j<<',';j<<times[i];}j<<"],\"median_ms\":"<<median(times)<<",\"residues_per_s\":"<<residues*1000.0/median(times)<<",\"sequences_per_s\":"<<n*1000.0/median(times)<<",\"repeats\":"<<repeats<<",\"scope\":\"resident sequence buffers; full glocal M/I/D Viterbi, not official HMMER calibration/filter pipeline\"}\n";
 std::cout<<device<<" sequences="<<n<<" median_ms="<<median(times)<<'\n';return 0;
 }catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 1;}}
