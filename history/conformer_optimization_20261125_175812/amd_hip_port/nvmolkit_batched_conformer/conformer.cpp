// SPDX-License-Identifier: MIT
// Original deterministic united-heavy-atom prototype, NOT MMFF94 or nvMolKit.
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
using Prop=hipDeviceProp_t;
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#define GPU 1
#define RT(x) cuda##x
#define HD __host__ __device__
using Prop=cudaDeviceProp;
#else
#define GPU 0
#define HD
#endif
constexpr int MAXN=64,STEPS=200;
struct Mol{int n,start,count,atomstart;};
struct Term{int type,i,j,k,l,period;double p1,p2,p3,p4;};
static_assert(sizeof(Term)==56);
struct V{double x,y,z;HD V operator+(V a)const{return{x+a.x,y+a.y,z+a.z};}HD V operator-(V a)const{return{x-a.x,y-a.y,z-a.z};}HD V operator-()const{return{-x,-y,-z};}HD V operator*(double a)const{return{x*a,y*a,z*a};}};
HD double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
HD V cross(V a,V b){return{a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
HD V get(double const*x,int i){return{x[3*i],x[3*i+1],x[3*i+2]};}
HD void term_value(Term const&t,double const*x,int atom,double &E,V &G){
 V a=get(x,t.i),b=get(x,t.j);int arity=t.type==0||t.type==3?2:t.type==1?3:4;
 if(t.type==0){V d=a-b;double rr=sqrt(dot(d,d)+1e-20),dr=rr-t.p2;E+=.5*t.p1*dr*dr/arity;V g=d*(t.p1*dr/rr);G=G+(atom==t.i?g:-g);return;}
 if(t.type==3){V d=a-b;double r2=dot(d,d)+.01,rr=sqrt(r2),ri=1/rr,s2=t.p2*t.p2/r2,s6=s2*s2*s2;E+=(4*t.p1*(s6*s6-s6)+t.p3*ri)/arity;V g=d*(-24*t.p1*(2*s6*s6-s6)/r2-t.p3*ri/r2);G=G+(atom==t.i?g:-g);return;}
 V c=get(x,t.k);
 if(t.type==1){V u=a-b,v=c-b;double uu=dot(u,u)+1e-20,vv=dot(v,v)+1e-20,prod=1/sqrt(uu*vv),co=dot(u,v)*prod,dc=co-t.p2;E+=.5*t.p1*dc*dc/arity;V gi=(v*prod-u*(co/uu))*(t.p1*dc),gk=(u*prod-v*(co/vv))*(t.p1*dc);G=G+(atom==t.i?gi:atom==t.k?gk:-gi-gk);return;}
 V d=get(x,t.l),u=a-b,v=c-b,w=d-c,n1=cross(u,v),n2=cross(v,w);double aa=dot(n1,n1)+1e-16,bb=dot(n2,n2)+1e-16,prod=1/sqrt(aa*bb),co=dot(n1,n2)*prod,der;
 if(t.period==3){E+=t.p1*(1+4*co*co*co-3*co)/4;der=t.p1*(12*co*co-3);}else{E+=2*t.p1*(1-co*co)/4;der=-4*t.p1*co;}
 V g1=(n2*prod-n1*(co/aa))*der,g2=(n1*prod-n2*(co/bb))*der;V gu=cross(v,g1),gv=cross(g1,u)+cross(w,g2),gw=cross(g2,v);G=G+(atom==t.i?gu:atom==t.j?-gu-gv:atom==t.k?gv-gw:gw);
}
HD void atom_eval(int i,double const*x,Mol const&m,Term const*terms,int const*ptr,int const*adj,double&e,V&g){e=0;g={0,0,0};int global=m.atomstart+i;for(int j=ptr[global];j<ptr[global+1];++j)term_value(terms[adj[j]],x,i,e,g);}
void cpu_opt(int ci,int stride,double const*initial,double*out,double*history,double*gradient,Mol const*mols,Term const*terms,int const*ptr,int const*adj){Mol m=mols[ci/10];double x[MAXN*3],cand[MAXN*3],grad[MAXN*3],ea[MAXN];for(int j=0;j<m.n*3;++j)x[j]=initial[ci*stride*3+j];double step=.003;
 for(int it=0;it<=STEPS;++it){double E=0,gmax=0;for(int i=0;i<m.n;++i){V g;atom_eval(i,x,m,terms,ptr,adj,ea[i],g);grad[3*i]=g.x;grad[3*i+1]=g.y;grad[3*i+2]=g.z;E+=ea[i];double q=sqrt(dot(g,g));if(q>gmax)gmax=q;}history[ci*(STEPS+1)+it]=E;if(it==STEPS){gradient[ci]=gmax;break;}double scale=gmax>10?10/gmax:1;bool accepted=false;for(int bt=0;bt<16;++bt){for(int j=0;j<m.n*3;++j)cand[j]=x[j]-step*scale*grad[j];double next=0;for(int i=0;i<m.n;++i){V g;double e;atom_eval(i,cand,m,terms,ptr,adj,e,g);next+=e;}if(next<=E){for(int j=0;j<m.n*3;++j)x[j]=cand[j];step=std::min(.02,step*1.15);accepted=true;break;}step*=.5;}if(!accepted)step=.003;}
 for(int j=0;j<m.n*3;++j)out[ci*stride*3+j]=x[j];}
#if GPU
#define CHECK(a) do{auto e=(a);if(e!=RT(Success))throw std::runtime_error(RT(GetErrorString)(e));}while(0)
__global__ void optimize(int nc,int stride,double const*initial,double*out,double*history,double*gradient,Mol const*mols,Term const*terms,int const*ptr,int const*adj){int ci=blockIdx.x,tid=threadIdx.x;if(ci>=nc)return;Mol m=mols[ci/10];__shared__ double x[MAXN*3],cand[MAXN*3],grad[MAXN*3],ea[MAXN],E,step,scale;__shared__ int accept;
 if(tid<m.n){for(int a=0;a<3;++a)x[3*tid+a]=initial[(ci*stride+tid)*3+a];}if(tid==0)step=.003;__syncthreads();
 for(int it=0;it<=STEPS;++it){if(tid<m.n){V g;atom_eval(tid,x,m,terms,ptr,adj,ea[tid],g);grad[3*tid]=g.x;grad[3*tid+1]=g.y;grad[3*tid+2]=g.z;}__syncthreads();if(tid==0){E=0;double gm=0;for(int i=0;i<m.n;++i){E+=ea[i];double q=sqrt(grad[3*i]*grad[3*i]+grad[3*i+1]*grad[3*i+1]+grad[3*i+2]*grad[3*i+2]);if(q>gm)gm=q;}history[ci*(STEPS+1)+it]=E;scale=gm>10?10/gm:1;gradient[ci]=gm;}__syncthreads();if(it==STEPS)break;
 for(int bt=0;bt<16;++bt){if(tid<m.n){for(int a=0;a<3;++a)cand[3*tid+a]=x[3*tid+a]-step*scale*grad[3*tid+a];}__syncthreads();if(tid<m.n){V g;atom_eval(tid,cand,m,terms,ptr,adj,ea[tid],g);}__syncthreads();if(tid==0){double next=0;for(int i=0;i<m.n;++i)next+=ea[i];accept=next<=E;if(accept)step=fmin(.02,step*1.15);else step*=.5;}__syncthreads();if(accept){if(tid<m.n){for(int a=0;a<3;++a)x[3*tid+a]=cand[3*tid+a];}__syncthreads();break;}if(bt==15&&tid==0)step=.003;__syncthreads();}}
 if(tid<m.n){for(int a=0;a<3;++a)out[(ci*stride+tid)*3+a]=x[3*tid+a];}}
template<class T>struct Mem{T*p;Mem(size_t n){CHECK(RT(Malloc)(&p,n*sizeof(T)));}~Mem(){RT(Free)(p);}};
#endif
std::string path(std::string a,std::string b){return (std::filesystem::path(a)/b).string();}
template<class T>std::vector<T>load(std::string f,size_t n){std::vector<T>a(n);std::ifstream s(f,std::ios::binary);if(!s.read((char*)a.data(),n*sizeof(T)))throw std::runtime_error("read "+f);return a;}
template<class T>void save(std::string f,std::vector<T>const&a){std::ofstream s(f,std::ios::binary);s.write((char*)a.data(),a.size()*sizeof(T));}
int main(int argc,char**argv){try{if(argc<3)return 2;std::string in=argv[1],out=argv[2];int nc,nmol,stride,nt,nap,nadj;std::ifstream params(path(in,"params.txt"));params>>nc>>nmol>>stride>>nt>>nap>>nadj;if(stride>MAXN||nc!=10*nmol)throw std::runtime_error("bad dims");auto x=load<double>(path(in,"initial.bin"),size_t(nc)*stride*3);auto mol=load<Mol>(path(in,"mols.bin"),nmol);auto term=load<Term>(path(in,"terms.bin"),nt);auto ptr=load<int>(path(in,"ptr.bin"),nap+1);auto adj=load<int>(path(in,"adj.bin"),nadj);if(argc>3&&std::string(argv[3])=="gradient"){int ci=argc>4?std::stoi(argv[4]):0;Mol m=mol[ci/10];std::vector<double>gg(m.n*3);double E=0;for(int i=0;i<m.n;++i){V grad;double e;atom_eval(i,x.data()+ci*stride*3,m,term.data(),ptr.data(),adj.data(),e,grad);E+=e;gg[3*i]=grad.x;gg[3*i+1]=grad.y;gg[3*i+2]=grad.z;}std::filesystem::create_directories(out);save(path(out,"gradient.bin"),gg);std::ofstream(path(out,"energy.txt"))<<std::setprecision(17)<<E;return 0;}std::vector<double>y(x.size()),h(size_t(nc)*(STEPS+1)),g(nc);std::string dev="CPU single-thread C++";double h2d=0,d2h=0;auto before=std::chrono::steady_clock::now();
#if GPU
 Prop prop{};CHECK(RT(GetDeviceProperties)(&prop,0));dev=prop.name;Mem<double>dx(x.size()),dy(x.size()),dh(h.size()),dg(g.size());Mem<Mol>dm(mol.size());Mem<Term>dt(term.size());Mem<int>dp(ptr.size()),da(adj.size());CHECK(RT(DeviceSynchronize)());before=std::chrono::steady_clock::now();CHECK(RT(Memcpy)(dx.p,x.data(),x.size()*8,RT(MemcpyHostToDevice)));CHECK(RT(Memset)(dy.p,0,y.size()*8));CHECK(RT(Memcpy)(dm.p,mol.data(),mol.size()*sizeof(Mol),RT(MemcpyHostToDevice)));CHECK(RT(Memcpy)(dt.p,term.data(),term.size()*sizeof(Term),RT(MemcpyHostToDevice)));CHECK(RT(Memcpy)(dp.p,ptr.data(),ptr.size()*4,RT(MemcpyHostToDevice)));CHECK(RT(Memcpy)(da.p,adj.data(),adj.size()*4,RT(MemcpyHostToDevice)));CHECK(RT(DeviceSynchronize)());h2d=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();before=std::chrono::steady_clock::now();optimize<<<nc,64>>>(nc,stride,dx.p,dy.p,dh.p,dg.p,dm.p,dt.p,dp.p,da.p);CHECK(RT(GetLastError)());CHECK(RT(DeviceSynchronize)());
#else
 for(int ci=0;ci<nc;++ci)cpu_opt(ci,stride,x.data(),y.data(),h.data(),g.data(),mol.data(),term.data(),ptr.data(),adj.data());
#endif
 double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
#if GPU
 before=std::chrono::steady_clock::now();CHECK(RT(Memcpy)(y.data(),dy.p,y.size()*8,RT(MemcpyDeviceToHost)));CHECK(RT(Memcpy)(h.data(),dh.p,h.size()*8,RT(MemcpyDeviceToHost)));CHECK(RT(Memcpy)(g.data(),dg.p,g.size()*8,RT(MemcpyDeviceToHost)));CHECK(RT(DeviceSynchronize)());d2h=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
#endif
 std::filesystem::create_directories(out);save(path(out,"coordinates.bin"),y);save(path(out,"history.bin"),h);save(path(out,"max_gradient.bin"),g);std::ofstream js(path(out,"timings.json"));js<<std::setprecision(16)<<"{\"device\":\""<<dev<<"\",\"conformers\":"<<nc<<",\"molecules\":"<<nmol<<",\"steps\":200,\"seconds\":"<<secs<<",\"h2d_seconds\":"<<h2d<<",\"d2h_seconds\":"<<d2h<<",\"scope\":\"single complete optimization; shared coordinate/candidate/gradient state on GPU; global topology reads and diagnostic writes retained; no fastmath\"}";std::cout<<dev<<" seconds="<<secs<<'\n';return 0;}catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 1;}}
