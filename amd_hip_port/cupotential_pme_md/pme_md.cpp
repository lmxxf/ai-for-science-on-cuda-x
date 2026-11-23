// SPDX-License-Identifier: MIT
// Original PME/MD prototype, not libcupo/OpenMM-compatible.
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
#include <hipfft/hipfft.h>
#define RT(x) hip##x
#define FFT(x) hipfft##x
#define Z2Z HIPFFT_Z2Z
#define FWD HIPFFT_FORWARD
#define INV HIPFFT_BACKWARD
#define PROP hipDeviceProp_t
#else
#include <cuda_runtime.h>
#include <cufft.h>
#define RT(x) cuda##x
#define FFT(x) cufft##x
#define Z2Z CUFFT_Z2Z
#define FWD CUFFT_FORWARD
#define INV CUFFT_INVERSE
#define PROP cudaDeviceProp
#endif
#define CHECK(e) do{auto er=(e);if(er!=RT(Success))throw std::runtime_error(std::string(#e)+RT(GetErrorString)(er));}while(0)
#define FCHECK(e) do{int er=(e);if(er)throw std::runtime_error(std::string(#e)+" FFT error="+std::to_string(er));}while(0)
using Z=FFT(DoubleComplex);using Clock=std::chrono::steady_clock;
constexpr double PI=3.14159265358979323846,KE=1389.354576,KB=.00831446261815324;
struct Cfg{int n,m;double L,alpha,rc,kbond,kangle,l0,theta0;};
__device__ double mic(double d,double L){return d-L*floor(d/L+.5);}
__device__ void weights(double t,double*w,double*d){w[0]=(1-t)*(1-t)*(1-t)/6;w[1]=(3*t*t*t-6*t*t+4)/6;w[2]=(-3*t*t*t+3*t*t+3*t+1)/6;w[3]=t*t*t/6;d[0]=-.5*(1-t)*(1-t);d[1]=1.5*t*t-2*t;d[2]=-1.5*t*t+t+.5;d[3]=.5*t*t;}
__device__ int wrap(int v,int n){return (v%n+n)%n;}
__global__ void spread(double const*x,double const*q,Z*rho,Cfg c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.n)return;int b[3];double w[3][4],dw[4];for(int a=0;a<3;++a){double u=x[3*i+a]*c.m/c.L;double fl=floor(u);b[a]=(int)fl-1;weights(u-fl,w[a],dw);}for(int a=0;a<4;++a)for(int b1=0;b1<4;++b1)for(int d=0;d<4;++d){int g=(wrap(b[0]+a,c.m)*c.m+wrap(b[1]+b1,c.m))*c.m+wrap(b[2]+d,c.m);atomicAdd(&rho[g].x,q[i]*w[0][a]*w[1][b1]*w[2][d]);}}
__global__ void green(Z*rho,Cfg c){int i=blockIdx.x*blockDim.x+threadIdx.x;int M=c.m*c.m*c.m;if(i>=M)return;int z=i%c.m,y=(i/c.m)%c.m,x=i/(c.m*c.m);int kx=x<=c.m/2?x:x-c.m,ky=y<=c.m/2?y:y-c.m,kz=z<=c.m/2?z:z-c.m;double k2=(2*PI/c.L)*(2*PI/c.L)*(kx*kx+ky*ky+kz*kz);double deconv=1;int ks[3]={kx,ky,kz};for(int a=0;a<3;++a){double u=PI*ks[a]/c.m;double v=ks[a]==0?1:sin(u)/u;double v2=v*v;deconv*=v2*v2*v2*v2;}double G=k2==0?0:KE/(c.L*c.L*c.L)*4*PI*exp(-k2/(4*c.alpha*c.alpha))/k2/deconv;rho[i].x*=G;rho[i].y*=G;}
__device__ void bonded(int i,double const*x,Cfg c,double &e,double *f){int b=(i/3)*3,local=i%3;double a[3],d[3];double ra=0,rd=0,dot=0;for(int j=0;j<3;++j){a[j]=mic(x[3*b+j]-x[3*(b+1)+j],c.L);d[j]=mic(x[3*(b+2)+j]-x[3*(b+1)+j],c.L);ra+=a[j]*a[j];rd+=d[j]*d[j];dot+=a[j]*d[j];}ra=sqrt(ra);rd=sqrt(rd);double co=dot/(ra*rd);co=fmax(-.999999999999,fmin(.999999999999,co));double theta=acos(co),sn=sqrt(1-co*co);e+=(.5*c.kbond*((ra-c.l0)*(ra-c.l0)+(rd-c.l0)*(rd-c.l0))+.5*c.kangle*(theta-c.theta0)*(theta-c.theta0))/3;
for(int j=0;j<3;++j){double f0=-c.kbond*(ra-c.l0)*a[j]/ra-c.kangle*(theta-c.theta0)*(co*a[j]/(ra*ra)-d[j]/(ra*rd))/sn;double f2=-c.kbond*(rd-c.l0)*d[j]/rd-c.kangle*(theta-c.theta0)*(co*d[j]/(rd*rd)-a[j]/(ra*rd))/sn;f[j]+=local==0?f0:local==2?f2:-f0-f2;}}
__global__ void real_force(double const*x,double const*q,double*f,double*energy,Cfg c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.n)return;double ff[3]={0,0,0},e=0;for(int j=0;j<c.n;++j){if(i==j)continue;double d[3],r2=0;for(int a=0;a<3;++a){d[a]=mic(x[3*i+a]-x[3*j+a],c.L);r2+=d[a]*d[a];}if(r2>=c.rc*c.rc)continue;double r=sqrt(r2),ri=1/r,qq=KE*q[i]*q[j],ar=c.alpha*r,er=erfc(ar);double sr2=1.44/r2,sr6=sr2*sr2*sr2; e+=.2*(sr6*sr6-sr6)+qq*er*ri;double cf=1.2*(2*sr6*sr6-sr6)/r2+qq*(er*ri/r2+2*c.alpha/sqrt(PI)*exp(-ar*ar)/r2);for(int a=0;a<3;++a)ff[a]+=cf*d[a];}e*=.5;e-=KE*c.alpha/sqrt(PI)*q[i]*q[i];bonded(i,x,c,e,ff);energy[i]=e;for(int a=0;a<3;++a)f[3*i+a]=ff[a];}
__global__ void gather(double const*x,double const*q,Z const*phi,double*f,double*energy,Cfg c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.n)return;int b[3];double w[3][4],dw[3][4];for(int a=0;a<3;++a){double u=x[3*i+a]*c.m/c.L;double fl=floor(u);b[a]=(int)fl-1;weights(u-fl,w[a],dw[a]);}double v=0,der[3]={0,0,0};for(int a=0;a<4;++a)for(int z=0;z<4;++z)for(int d=0;d<4;++d){int g=(wrap(b[0]+a,c.m)*c.m+wrap(b[1]+z,c.m))*c.m+wrap(b[2]+d,c.m);double p=phi[g].x;v+=w[0][a]*w[1][z]*w[2][d]*p;der[0]+=dw[0][a]*w[1][z]*w[2][d]*p;der[1]+=w[0][a]*dw[1][z]*w[2][d]*p;der[2]+=w[0][a]*w[1][z]*dw[2][d]*p;}energy[i]+=.5*q[i]*v;for(int a=0;a<3;++a)f[3*i+a]-=q[i]*c.m/c.L*der[a];}
__device__ std::uint64_t mix(std::uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
__device__ double normal(int step,int i){auto a=mix(20261124ULL+std::uint64_t(step)*123457+2*i),b=mix(a);double u=((a>>11)+.5)/9007199254740992.,v=((b>>11)+.5)/9007199254740992.;return sqrt(-2*log(u))*cos(2*PI*v);}
__global__ void kick_drift(double*x,double*v,double const*f,double const*mass,Cfg c,double dt,int phase,int step,int nvt){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=3*c.n)return;int i=k/3;double a=100*f[k]/mass[i];if(phase==0){v[k]+=.5*dt*a;if(nvt){x[k]+=.5*dt*v[k];double damp=exp(-10*dt);v[k]=damp*v[k]+sqrt((1-damp*damp)*KB*300/(.01*mass[i]))*normal(step,k);x[k]+=.5*dt*v[k];}else{x[k]+=dt*v[k];}x[k]-=c.L*floor(x[k]/c.L);}else v[k]+=.5*dt*a;}
__global__ void record(double const*x,double const*v,double const*mass,double const*e,double*history,Cfg c,int step,double dt){if(blockIdx.x||threadIdx.x)return;double U=0,K=0;for(int i=0;i<c.n;++i){U+=e[i];for(int a=0;a<3;++a)K+=.005*mass[i]*v[3*i+a]*v[3*i+a];}double*h=history+step*7;h[0]=step*dt;h[1]=U;h[2]=K;h[3]=U+K;h[4]=2*K/(3*c.n*KB);double r2=0;for(int a=0;a<3;++a){double d=mic(x[a]-x[3+a],c.L);r2+=d*d;}h[5]=sqrt(r2);h[6]=x[0];}
template<class T>struct Mem{T*p;Mem(size_t n){CHECK(RT(Malloc)(&p,n*sizeof(T)));}~Mem(){RT(Free)(p);}};
template<class T>std::vector<T>load(std::string f,size_t n){std::vector<T>r(n);std::ifstream s(f,std::ios::binary);if(!s.read((char*)r.data(),n*sizeof(T)))throw std::runtime_error("read "+f);return r;}
template<class T>void save(std::string f,std::vector<T>const&r){std::ofstream s(f,std::ios::binary);s.write((char*)r.data(),r.size()*sizeof(T));}
std::string path(std::string a,std::string b){return (std::filesystem::path(a)/b).string();}
int main(int argc,char**argv){try{if(argc<3)return 2;std::string in=argv[1],out=argv[2];std::filesystem::create_directories(out);Cfg c{};std::ifstream p(path(in,"params.txt"));p>>c.n>>c.m>>c.L>>c.alpha>>c.rc>>c.kbond>>c.kangle>>c.l0>>c.theta0;if(c.n%3||c.n<3)throw std::runtime_error("trimer topology required");auto a=load<double>(path(in,"atoms.bin"),c.n*8);std::vector<double>x(c.n*3),v(c.n*3),q(c.n),mass(c.n);for(int i=0;i<c.n;++i){for(int j=0;j<3;++j){x[3*i+j]=a[8*i+j];v[3*i+j]=a[8*i+3+j];}q[i]=a[8*i+6];mass[i]=a[8*i+7];}Mem<double>dx(x.size()),dv(v.size()),dq(q.size()),dm(mass.size()),df(x.size()),de(c.n);Mem<Z>mesh(size_t(c.m)*c.m*c.m);CHECK(RT(Memcpy)(dx.p,x.data(),x.size()*8,RT(MemcpyHostToDevice)));CHECK(RT(Memcpy)(dv.p,v.data(),v.size()*8,RT(MemcpyHostToDevice)));CHECK(RT(Memcpy)(dq.p,q.data(),q.size()*8,RT(MemcpyHostToDevice)));CHECK(RT(Memcpy)(dm.p,mass.data(),mass.size()*8,RT(MemcpyHostToDevice)));FFT(Handle)plan;FCHECK(FFT(Plan3d)(&plan,c.m,c.m,c.m,Z2Z));
 auto force=[&]{CHECK(RT(MemsetAsync)(mesh.p,0,size_t(c.m)*c.m*c.m*sizeof(Z),0));spread<<<(c.n+63)/64,64>>>(dx.p,dq.p,mesh.p,c);FCHECK(FFT(ExecZ2Z)(plan,mesh.p,mesh.p,FWD));green<<<(c.m*c.m*c.m+255)/256,256>>>(mesh.p,c);FCHECK(FFT(ExecZ2Z)(plan,mesh.p,mesh.p,INV));real_force<<<(c.n+63)/64,64>>>(dx.p,dq.p,df.p,de.p,c);gather<<<(c.n+63)/64,64>>>(dx.p,dq.p,mesh.p,df.p,de.p,c);CHECK(RT(GetLastError)());};
 force();CHECK(RT(DeviceSynchronize)());std::vector<double>fe(x.size()),en(c.n);CHECK(RT(Memcpy)(fe.data(),df.p,fe.size()*8,RT(MemcpyDeviceToHost)));CHECK(RT(Memcpy)(en.data(),de.p,en.size()*8,RT(MemcpyDeviceToHost)));save(path(out,"initial_forces.bin"),fe);save(path(out,"initial_energy.bin"),en);
 PROP prop{};CHECK(RT(GetDeviceProperties)(&prop,0));std::ofstream js(path(out,"timings.json"));js<<std::setprecision(16)<<"{\"device\":\""<<prop.name<<"\",\"grid\":"<<c.m<<",\"atoms\":"<<c.n<<",\"trajectories\":[";
 for(int run=0;run<3;++run){int steps=run==1?2000:1000,nvt=run==2;double dt=run==0?.0005:run==1?.00025:.001;CHECK(RT(Memcpy)(dx.p,x.data(),x.size()*8,RT(MemcpyHostToDevice)));CHECK(RT(Memcpy)(dv.p,v.data(),v.size()*8,RT(MemcpyHostToDevice)));Mem<double>hist(size_t(steps+1)*7);force();record<<<1,1>>>(dx.p,dv.p,dm.p,de.p,hist.p,c,0,dt);CHECK(RT(DeviceSynchronize)());auto start=Clock::now();for(int step=1;step<=steps;++step){kick_drift<<<(c.n*3+127)/128,128>>>(dx.p,dv.p,df.p,dm.p,c,dt,0,step,nvt);force();kick_drift<<<(c.n*3+127)/128,128>>>(dx.p,dv.p,df.p,dm.p,c,dt,1,step,nvt);record<<<1,1>>>(dx.p,dv.p,dm.p,de.p,hist.p,c,step,dt);}CHECK(RT(DeviceSynchronize)());double sec=std::chrono::duration<double>(Clock::now()-start).count();std::vector<double>h(size_t(steps+1)*7);CHECK(RT(Memcpy)(h.data(),hist.p,h.size()*8,RT(MemcpyDeviceToHost)));std::string name=run==0?"nve":run==1?"nve_halfdt":"nvt";save(path(out,name+"_history.bin"),h);std::vector<double>finalx(x.size()),finalv(v.size());CHECK(RT(Memcpy)(finalx.data(),dx.p,x.size()*8,RT(MemcpyDeviceToHost)));CHECK(RT(Memcpy)(finalv.data(),dv.p,v.size()*8,RT(MemcpyDeviceToHost)));save(path(out,name+"_positions.bin"),finalx);save(path(out,name+"_velocities.bin"),finalv);if(run)js<<',';js<<"{\"name\":\""<<name<<"\",\"steps\":"<<steps<<",\"dt_ps\":"<<dt<<",\"seconds\":"<<sec<<"}";std::cout<<prop.name<<' '<<name<<' '<<sec<<" seconds\n";}
 js<<"]}";FCHECK(FFT(Destroy)(plan));return 0;}catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 1;}}
