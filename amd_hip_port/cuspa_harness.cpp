// Standalone test harness; kernels retain NVIDIA Apache-2.0 notices.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <filesystem>
#ifdef __HIPCC__
#include "index/kernels_index.cuh"
#include "query/kernels_query.cuh"
#include "aggregate/kernels_aggregate.cuh"
#else
#include "../cuspa/cpp/src/index/kernels_index.cuh"
#include "../cuspa/cpp/src/query/kernels_query.cuh"
#include "../cuspa/cpp/src/aggregate/kernels_aggregate.cuh"
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemcpy cudaMemcpy
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipGetDeviceProperties cudaGetDeviceProperties
#define hipDeviceProp_t cudaDeviceProp
#define hipEvent_t cudaEvent_t
#define hipEventCreate cudaEventCreate
#define hipEventRecord cudaEventRecord
#define hipEventSynchronize cudaEventSynchronize
#define hipEventElapsedTime cudaEventElapsedTime
#define hipEventDestroy cudaEventDestroy
#endif
using Clock=std::chrono::steady_clock;
using I=std::int32_t;
std::string path(std::string const& a,std::string const& b){return (std::filesystem::path(a)/b).string();}
template<class T>std::vector<T> load(std::string const& f,size_t n){std::vector<T> a(n);std::ifstream s(f,std::ios::binary);if(!s.read(reinterpret_cast<char*>(a.data()),n*sizeof(T)))throw std::runtime_error("failed read "+f);return a;}
template<class T>void save(std::string const& f,T const* a,size_t n){std::ofstream s(f,std::ios::binary);if(!s.write(reinterpret_cast<char const*>(a),n*sizeof(T)))throw std::runtime_error("failed write "+f);}
template<class T>struct Mem{T*p=nullptr;explicit Mem(size_t n){CS_CHECK(hipMalloc(&p,n*sizeof(T)));}~Mem(){hipFree(p);}Mem(Mem const&)=delete;};
template<class F>double wall(F f){CS_CHECK(hipDeviceSynchronize());auto t=Clock::now();f();CS_CHECK(hipDeviceSynchronize());return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct Measure {double wall_ms,event_ms;};
template<class F>Measure measure(F f){hipEvent_t a,b;CS_CHECK(hipEventCreate(&a));CS_CHECK(hipEventCreate(&b));CS_CHECK(hipDeviceSynchronize());auto t=Clock::now();CS_CHECK(hipEventRecord(a,0));f();CS_CHECK(hipEventRecord(b,0));CS_CHECK(hipEventSynchronize(b));auto ms=std::chrono::duration<double,std::milli>(Clock::now()-t).count();float e;CS_CHECK(hipEventElapsedTime(&e,a,b));hipEventDestroy(a);hipEventDestroy(b);return {ms,e};}
void arr(std::ostream&o,std::vector<double>const&a){o<<'[';for(size_t i=0;i<a.size();++i){if(i)o<<',';o<<a[i];}o<<']';}
double median(std::vector<double>a){std::sort(a.begin(),a.end());return a[a.size()/2];}
int main(int argc,char**argv){try{
 if(argc<3){std::cerr<<"usage: cuspa_harness INPUT_DIR OUTPUT_DIR [repeats=7]\n";return 2;}
 std::string in=argv[1],out=argv[2];std::filesystem::create_directories(out);int repeats=argc>3?std::stoi(argv[3]):7;
 I P,N,G,nx,ny;double ox,oy,sx,sy;std::ifstream params(path(in,"params.txt"));if(!(params>>P>>N>>G>>nx>>ny>>ox>>oy>>sx>>sy))throw std::runtime_error("params.txt malformed");
 if(P<=0||N<=0||G<=0||nx<=0||ny<=0||sx<=0||sy<=0)throw std::runtime_error("invalid input dimensions");
 auto polys=load<double>(path(in,"polys.bin"),size_t(P)*9*2);auto points=load<double>(path(in,"points.bin"),size_t(N)*2);auto genes=load<I>(path(in,"genes.bin"),N);auto parts=load<I>(path(in,"part.bin"),P+1);auto rings=load<I>(path(in,"ring.bin"),P+1);
 hipDeviceProp_t prop{};CS_CHECK(hipGetDeviceProperties(&prop,0));
 Mem<double>dp(polys.size()),dt(points.size()),aabb(size_t(P)*4);Mem<I>dg(N),part(P+1),ring(P+1),assigned(N),grid(size_t(nx)*ny+1),poly_ids(size_t(P)*16),ip(P+1),ix(N),counts(N);
 double h2d=wall([&]{CS_CHECK(hipMemcpy(dp.p,polys.data(),polys.size()*8,hipMemcpyHostToDevice));CS_CHECK(hipMemcpy(dt.p,points.data(),points.size()*8,hipMemcpyHostToDevice));CS_CHECK(hipMemcpy(dg.p,genes.data(),size_t(N)*4,hipMemcpyHostToDevice));CS_CHECK(hipMemcpy(part.p,parts.data(),size_t(P+1)*4,hipMemcpyHostToDevice));CS_CHECK(hipMemcpy(ring.p,rings.data(),size_t(P+1)*4,hipMemcpyHostToDevice));});
 std::int64_t K=0,nnz=0;auto index=[&]{cuspa::launch_poly_aabbs(part.p,ring.p,dp.p,P,aabb.p,0);K=cuspa::run_compute_index_size(aabb.p,P,ox,oy,1/sx,1/sy,nx,ny,0);if(K>size_t(P)*16)throw std::runtime_error("index capacity exceeded");cuspa::run_build_index(aabb.p,P,ox,oy,1/sx,1/sy,nx,ny,(I)K,poly_ids.p,grid.p,0);};
 auto query=[&]{cuspa::run_assign_points(dt.p,(std::int64_t)N,ox,oy,1/sx,1/sy,nx,ny,grid.p,poly_ids.p,aabb.p,part.p,ring.p,dp.p,0,assigned.p,0);};
 auto aggregate=[&]{nnz=cuspa::run_aggregate_to_cells(assigned.p,dg.p,(std::int64_t)N,P,ip.p,ix.p,counts.p,0);};
 auto pipeline=[&]{index();query();aggregate();};
 auto cold=measure(pipeline);for(int i=0;i<3;++i)pipeline();CS_CHECK(hipDeviceSynchronize());
 std::vector<double>wi,wq,wa,wp,ei,eq,ea,ep;for(int i=0;i<repeats;++i){auto a=measure(index);wi.push_back(a.wall_ms);ei.push_back(a.event_ms);auto b=measure(query);wq.push_back(b.wall_ms);eq.push_back(b.event_ms);auto c=measure(aggregate);wa.push_back(c.wall_ms);ea.push_back(c.event_ms);auto d=measure(pipeline);wp.push_back(d.wall_ms);ep.push_back(d.event_ms);}
 std::vector<I>ha(N),hi(P+1),hx(nnz),hv(nnz);double d2h=wall([&]{CS_CHECK(hipMemcpy(ha.data(),assigned.p,size_t(N)*4,hipMemcpyDeviceToHost));CS_CHECK(hipMemcpy(hi.data(),ip.p,size_t(P+1)*4,hipMemcpyDeviceToHost));CS_CHECK(hipMemcpy(hx.data(),ix.p,size_t(nnz)*4,hipMemcpyDeviceToHost));CS_CHECK(hipMemcpy(hv.data(),counts.p,size_t(nnz)*4,hipMemcpyDeviceToHost));});
 save(path(out,"assignment.bin"),ha.data(),ha.size());save(path(out,"indptr.bin"),hi.data(),hi.size());save(path(out,"indices.bin"),hx.data(),hx.size());save(path(out,"data.bin"),hv.data(),hv.size());
 std::ofstream j(path(out,"timings.json"));j<<std::setprecision(12)<<"{\n\"device\":\""<<prop.name<<"\",\n\"backend\":\"";
#ifdef __HIPCC__
 j<<"HIP hipCUB";
#else
 j<<"CUDA CUB";
#endif
 j<<"\",\n\"cells\":"<<P<<",\n\"points\":"<<N<<",\n\"genes\":"<<G<<",\n\"nnz\":"<<nnz<<",\n\"index_pairs\":"<<K<<",\n\"dtype\":\"float64\",\n\"warmups\":3,\n\"repeats\":"<<repeats<<",\n\"h2d_ms\":"<<h2d<<",\n\"d2h_ms\":"<<d2h<<",\n\"cold_pipeline_wall_ms\":"<<cold.wall_ms<<",\n\"index_wall_ms\":";arr(j,wi);j<<",\n\"query_wall_ms\":";arr(j,wq);j<<",\n\"aggregate_wall_ms\":";arr(j,wa);j<<",\n\"pipeline_wall_ms\":";arr(j,wp);j<<",\n\"index_event_ms\":";arr(j,ei);j<<",\n\"query_event_ms\":";arr(j,eq);j<<",\n\"aggregate_event_ms\":";arr(j,ea);j<<",\n\"pipeline_event_ms\":";arr(j,ep);j<<",\n\"pipeline_median_wall_ms\":"<<median(wp)<<",\n\"index_median_wall_ms\":"<<median(wi)<<",\n\"query_median_wall_ms\":"<<median(wq)<<",\n\"aggregate_median_wall_ms\":"<<median(wa)<<",\n\"timing_scope\":\"input device resident; synchronized host wall plus GPU events; H2D/D2H separately measured; original allocation and host scalar sync retained\"\n}\n";
 std::cout<<prop.name<<" P="<<P<<" N="<<N<<" nnz="<<nnz<<" pipeline_ms="<<median(wp)<<"\n";return 0;
 }catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 1;}}
