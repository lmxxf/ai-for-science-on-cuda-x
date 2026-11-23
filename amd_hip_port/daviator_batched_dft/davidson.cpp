// MIT original educational Block-Davidson. Not NVIDIA Daviator/VASP source.
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
#define GP(x) do{auto e=(x);if(e!=hipSuccess){fprintf(stderr,"HIP %s\n",hipGetErrorString(e));exit(2);}}while(0)
#define MAL hipMalloc
#define FREE hipFree
#define CP hipMemcpy
#define H2D hipMemcpyHostToDevice
#define D2H hipMemcpyDeviceToHost
#define SYNC hipDeviceSynchronize
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#define GP(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);}}while(0)
#define MAL cudaMalloc
#define FREE cudaFree
#define CP cudaMemcpy
#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define SYNC cudaDeviceSynchronize
#endif
using V=std::vector<double>;using Clock=std::chrono::steady_clock;
double secs(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
#if defined(__HIPCC__) || defined(__CUDACC__)
__global__ void hmul(const double*H,const double*X,double*Y,int n,int m,int b,int first){int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=b*n*m)return;int sys=q/(n*m),i=(q/m)%n,j=q%m;double z=0;for(int k=0;k<n;k++)z+=H[((sys+first)*n+i)*n+k]*X[((sys+first)*n+k)*m+j];Y[((sys+first)*n+i)*m+j]=z;}

__global__ void project(const double*X,const double*Y,double*S,int n,int m,int B){int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=B*m*m)return;int b=q/(m*m),i=(q/m)%m,j=q%m;double z=0;for(int k=0;k<n;k++)z+=X[(b*n+k)*m+i]*Y[(b*n+k)*m+j];S[q]=z;}
__global__ void ritz_residual(const double*H,const double*V,const double*HV,const double*Q,const double*E,double*X,double*HX,double*R,double*P,int n,int m,int K,int B){int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=B*n*K)return;int b=q/(n*K),i=(q/K)%n,j=q%K;double x=0,hx=0;for(int k=0;k<m;k++){x+=V[(b*n+i)*m+k]*Q[(b*m+k)*K+j];hx+=HV[(b*n+i)*m+k]*Q[(b*m+k)*K+j];}double e=E[b*K+j],r=hx-e*x,den=H[(b*n+i)*n+i]-e;if(fabs(den)<.08)den=den<0?-.08:.08;X[q]=x;HX[q]=hx;R[q]=r;P[q]=-r/den;}
#endif
// Cyclic Jacobi diagonalization, independent of the full dense NumPy reference.
void jacobi(V A,int n,V& vals,V& Q){Q.assign(n*n,0);for(int i=0;i<n;i++)Q[i*n+i]=1;
for(int sweep=0;sweep<80;sweep++){double largest=0;for(int p=0;p<n;p++)for(int q=p+1;q<n;q++){double apq=A[p*n+q];largest=std::max(largest,std::abs(apq));if(std::abs(apq)<1e-14)continue;double tau=(A[q*n+q]-A[p*n+p])/(2*apq),t=(tau>=0?1.:-1.)/(std::abs(tau)+std::sqrt(1+tau*tau)),c=1/std::sqrt(1+t*t),s=t*c;double app=A[p*n+p],aqq=A[q*n+q];A[p*n+p]=app-t*apq;A[q*n+q]=aqq+t*apq;A[p*n+q]=A[q*n+p]=0;
for(int k=0;k<n;k++){if(k!=p&&k!=q){double x=A[k*n+p],y=A[k*n+q];A[k*n+p]=A[p*n+k]=c*x-s*y;A[k*n+q]=A[q*n+k]=s*x+c*y;}double x=Q[k*n+p],y=Q[k*n+q];Q[k*n+p]=c*x-s*y;Q[k*n+q]=s*x+c*y;}}
if(largest<1e-13)break;}
std::vector<int>ord(n);std::iota(ord.begin(),ord.end(),0);std::sort(ord.begin(),ord.end(),[&](int a,int b){return A[a*n+a]<A[b*n+b];});V T=Q;vals.resize(n);for(int j=0;j<n;j++){vals[j]=A[ord[j]*n+ord[j]];for(int i=0;i<n;i++)Q[i*n+j]=T[i*n+ord[j]];}}
struct State{V basis,eval,x;int m=0;double residual=1e99;};
int main(int argc,char**argv){if(argc<5){fprintf(stderr,"inputdir outprefix mode(seq/batch/cpu) maxiter\n");return 2;}std::string dir=argv[1],out=argv[2],mode=argv[3];int maxit=atoi(argv[4]),B,N,K;std::ifstream meta(dir+"/params.txt");meta>>B>>N>>K;V H((size_t)B*N*N);std::ifstream f(dir+"/H.bin",std::ios::binary);f.read((char*)H.data(),H.size()*8);if(!f){fprintf(stderr,"input fail\n");return 2;}
std::vector<State>st(B);for(int b=0;b<B;b++){std::vector<int>ind(N);std::iota(ind.begin(),ind.end(),0);std::sort(ind.begin(),ind.end(),[&](int a,int c){return H[((size_t)b*N+a)*N+a]<H[((size_t)b*N+c)*N+c];});st[b].m=24;st[b].basis.assign(N*24,0);for(int j=0;j<24;j++)st[b].basis[ind[j]*24+j]=1;}
#if defined(__HIPCC__) || defined(__CUDACC__)
double *dh,*dx,*dy,*dsmall,*dQ,*dE,*dRX,*dRH,*dR,*dP;GP(MAL((void**)&dh,H.size()*8));GP(MAL((void**)&dx,(size_t)B*N*128*8));GP(MAL((void**)&dy,(size_t)B*N*128*8));GP(MAL((void**)&dsmall,(size_t)B*128*128*8));GP(MAL((void**)&dQ,(size_t)B*128*K*8));GP(MAL((void**)&dE,(size_t)B*K*8));GP(MAL((void**)&dRX,(size_t)B*N*K*8));GP(MAL((void**)&dRH,(size_t)B*N*K*8));GP(MAL((void**)&dR,(size_t)B*N*K*8));GP(MAL((void**)&dP,(size_t)B*N*K*8));GP(CP(dh,H.data(),H.size()*8,H2D));GP(SYNC());
#endif
std::ofstream hist(out+"_history.csv");hist<<"iteration,max_residual\n";double apply=0,copy=0,host=0;auto whole=Clock::now();int iteration;
for(iteration=1;iteration<=maxit;iteration++){int M=0;for(auto&s:st)M=std::max(M,s.m);V X((size_t)B*N*M,0),Y(X.size());for(int b=0;b<B;b++)for(int i=0;i<N;i++)for(int j=0;j<st[b].m;j++)X[((size_t)b*N+i)*M+j]=st[b].basis[i*st[b].m+j];auto tt=Clock::now();
#if defined(__HIPCC__) || defined(__CUDACC__)
GP(CP(dx,X.data(),X.size()*8,H2D));GP(SYNC());copy+=secs(tt);tt=Clock::now();if(mode=="seq"){for(int b=0;b<B;b++)hmul<<<(N*M+127)/128,128>>>(dh,dx,dy,N,M,1,b);}else hmul<<<(B*N*M+127)/128,128>>>(dh,dx,dy,N,M,B,0);GP(SYNC());apply+=secs(tt);
#else
for(int b=0;b<B;b++)for(int i=0;i<N;i++)for(int j=0;j<M;j++){double z=0;for(int k=0;k<N;k++)z+=H[((size_t)b*N+i)*N+k]*X[((size_t)b*N+k)*M+j];Y[((size_t)b*N+i)*M+j]=z;}apply+=secs(tt);
#endif
tt=Clock::now();double maxres=0;
#if defined(__HIPCC__) || defined(__CUDACC__)
if(mode=="seq"){for(int b=0;b<B;b++)project<<<(M*M+127)/128,128>>>(dx+(size_t)b*N*M,dy+(size_t)b*N*M,dsmall+(size_t)b*M*M,N,M,1);}else project<<<(B*M*M+127)/128,128>>>(dx,dy,dsmall,N,M,B);GP(SYNC());apply+=secs(tt);V smallall((size_t)B*M*M),qall((size_t)B*M*K,0),evalall(B*K),rxall((size_t)B*N*K),rhall(rxall.size()),rrall(rxall.size()),ppall(rxall.size());tt=Clock::now();GP(CP(smallall.data(),dsmall,smallall.size()*8,D2H));GP(SYNC());copy+=secs(tt);tt=Clock::now();
for(int b=0;b<B;b++){int m=st[b].m;V sm(m*m),val,Q;for(int i=0;i<m;i++)for(int j=0;j<m;j++)sm[i*m+j]=.5*(smallall[(b*M+i)*M+j]+smallall[(b*M+j)*M+i]);jacobi(sm,m,val,Q);for(int j=0;j<K;j++){evalall[b*K+j]=val[j];for(int i=0;i<m;i++)qall[(b*M+i)*K+j]=Q[i*m+j];}}host+=secs(tt);tt=Clock::now();GP(CP(dQ,qall.data(),qall.size()*8,H2D));GP(CP(dE,evalall.data(),evalall.size()*8,H2D));GP(SYNC());copy+=secs(tt);tt=Clock::now();
if(mode=="seq"){for(int b=0;b<B;b++)ritz_residual<<<(N*K+127)/128,128>>>(dh+(size_t)b*N*N,dx+(size_t)b*N*M,dy+(size_t)b*N*M,dQ+(size_t)b*M*K,dE+b*K,dRX+(size_t)b*N*K,dRH+(size_t)b*N*K,dR+(size_t)b*N*K,dP+(size_t)b*N*K,N,M,K,1);}else ritz_residual<<<(B*N*K+127)/128,128>>>(dh,dx,dy,dQ,dE,dRX,dRH,dR,dP,N,M,K,B);GP(SYNC());apply+=secs(tt);tt=Clock::now();GP(CP(rxall.data(),dRX,rxall.size()*8,D2H));GP(CP(rhall.data(),dRH,rhall.size()*8,D2H));GP(CP(rrall.data(),dR,rrall.size()*8,D2H));GP(CP(ppall.data(),dP,ppall.size()*8,D2H));GP(SYNC());copy+=secs(tt);tt=Clock::now();
#endif
for(int b=0;b<B;b++){auto&s=st[b];int m=s.m;
#if defined(__HIPCC__) || defined(__CUDACC__)
V val(K),x(N*K),hx(N*K),R(N*K);for(int j=0;j<K;j++)val[j]=evalall[b*K+j];std::copy(rxall.begin()+(size_t)b*N*K,rxall.begin()+(size_t)(b+1)*N*K,x.begin());std::copy(rhall.begin()+(size_t)b*N*K,rhall.begin()+(size_t)(b+1)*N*K,hx.begin());
#else
V small(m*m,0);for(int i=0;i<m;i++)for(int j=0;j<m;j++){double z=0;for(int k=0;k<N;k++)z+=s.basis[k*m+i]*Y[((size_t)b*N+k)*M+j];small[i*m+j]=z;}for(int i=0;i<m;i++)for(int j=0;j<i;j++)small[i*m+j]=small[j*m+i]=(small[i*m+j]+small[j*m+i])*.5;
V val,Q;jacobi(small,m,val,Q);V x(N*K,0),hx(N*K,0),R(N*K);for(int i=0;i<N;i++)for(int j=0;j<K;j++)for(int k=0;k<m;k++){x[i*K+j]+=s.basis[i*m+k]*Q[k*m+j];hx[i*K+j]+=Y[((size_t)b*N+i)*M+k]*Q[k*m+j];}
#endif
s.eval.assign(val.begin(),val.begin()+K);s.x=x;s.residual=0;for(int j=0;j<K;j++){double rr=0;for(int i=0;i<N;i++){double z=hx[i*K+j]-val[j]*x[i*K+j];R[i*K+j]=z;rr+=z*z;}s.residual=std::max(s.residual,std::sqrt(rr));}maxres=std::max(maxres,s.residual);if(s.residual<3e-9)continue;
// Restart with Ritz vectors when adding a block would exceed 96 dimensions.
if(m+K>96){s.basis=x;s.m=K;m=K;}
V cols;for(int j=0;j<K;j++){V p(N);for(int i=0;i<N;i++){double den=H[((size_t)b*N+i)*N+i]-val[j];if(std::abs(den)<.08)den=den<0?-.08:.08;
#if defined(__HIPCC__) || defined(__CUDACC__)
p[i]=ppall[((size_t)b*N+i)*K+j];
#else
p[i]=-R[i*K+j]/den;
#endif
}
for(int pass=0;pass<2;pass++){for(int k=0;k<m;k++){double dot=0;for(int i=0;i<N;i++)dot+=p[i]*s.basis[i*m+k];for(int i=0;i<N;i++)p[i]-=dot*s.basis[i*m+k];}for(int k=0;k<(int)cols.size()/N;k++){double dot=0;for(int i=0;i<N;i++)dot+=p[i]*cols[k*N+i];for(int i=0;i<N;i++)p[i]-=dot*cols[k*N+i];}}
double norm=0;for(double z:p)norm+=z*z;norm=std::sqrt(norm);if(norm>1e-11){for(double z:p)cols.push_back(z/norm);}}
int add=cols.size()/N;if(add){V nb(N*(m+add));for(int i=0;i<N;i++){for(int j=0;j<m;j++)nb[i*(m+add)+j]=s.basis[i*m+j];for(int j=0;j<add;j++)nb[i*(m+add)+m+j]=cols[j*N+i];}s.m=m+add;s.basis.swap(nb);}}
host+=secs(tt);hist<<iteration<<","<<maxres<<"\n";printf("iter %d maxres %.8g\n",iteration,maxres);fflush(stdout);if(maxres<1e-7)break;}
double total=secs(whole);V e(B*K),x(B*N*K);double maxr=0;for(int b=0;b<B;b++){std::copy(st[b].eval.begin(),st[b].eval.end(),e.begin()+b*K);std::copy(st[b].x.begin(),st[b].x.end(),x.begin()+(size_t)b*N*K);maxr=std::max(maxr,st[b].residual);}std::ofstream ef(out+"_eigenvalues.bin",std::ios::binary);ef.write((char*)e.data(),e.size()*8);std::ofstream xf(out+"_eigenvectors.bin",std::ios::binary);xf.write((char*)x.data(),x.size()*8);std::ofstream jf(out+"_timings.json");jf.precision(17);jf<<"{\"iterations\":"<<iteration<<",\"total_seconds\":"<<total<<",\"H_apply_seconds\":"<<apply<<",\"host_RR_orthogonalization_seconds\":"<<host<<",\"transfer_seconds\":"<<copy<<",\"max_residual\":"<<maxr<<",\"mode\":\""<<mode<<"\"}";
#if defined(__HIPCC__) || defined(__CUDACC__)
GP(FREE(dsmall));GP(FREE(dQ));GP(FREE(dE));GP(FREE(dRX));GP(FREE(dRH));GP(FREE(dR));GP(FREE(dP));GP(FREE(dh));GP(FREE(dx));GP(FREE(dy));
#endif
return maxr<1e-7?0:3;}
