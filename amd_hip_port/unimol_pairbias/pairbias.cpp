// MIT; educational forward reference. Same source compiled by nvcc and HIP clang.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
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
#define hipGetErrorString cudaGetErrorString
#define hipSuccess cudaSuccess
#endif
#define CK(x) do{auto e=(x);if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}while(0)
using F=float;
template<class T>std::vector<T>load(std::string p,size_t n){std::vector<T>v(n);std::ifstream f(p,std::ios::binary);if(!f.read((char*)v.data(),n*sizeof(T)))throw std::runtime_error("read "+p);return v;}
template<class T>struct Mem{T*p;size_t n;Mem(size_t sz):n(sz){CK(hipMalloc(&p,n*sizeof(T)));}Mem(std::vector<T>const&v):Mem(v.size()){CK(hipMemcpy(p,v.data(),n*sizeof(T),hipMemcpyHostToDevice));}~Mem(){hipFree(p);}void save(std::string pth){std::vector<T>v(n);CK(hipMemcpy(v.data(),p,n*sizeof(T),hipMemcpyDeviceToHost));std::ofstream f(pth,std::ios::binary);f.write((char*)v.data(),n*sizeof(T));}};
__global__ void gaussian_kernel(F const*d,int const*edge,F const*mu,F const*std,F const*mul,F const*bi,F*g,int pairs){int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=pairs*128)return;int p=t/128,k=t%128;F x=mul[edge[p]]*d[p]+bi[edge[p]],z=(x-mu[k])/(fabsf(std[k])+1e-5f);g[t]=expf(-.5f*z*z)/(sqrtf(2.f*3.14159f)*(fabsf(std[k])+1e-5f));}
__global__ void linear_gelu(F const*x,F const*w,F const*b,F*y,int rows,int ni,int no,int gelu){int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=rows*no)return;int row=t/no,o=t%no;F s=b[o];for(int k=0;k<ni;k++)s+=x[row*ni+k]*w[o*ni+k];y[t]=gelu?.5f*s*(1.f+erff(s*.7071067811865475f)):s;}
__global__ void attention(F const*qkv,F const*bias,int const*mask,F*probs,F*out,int B,int N){int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=B*64*N)return;int i=t%N,h=(t/N)%64,b=t/(N*64);F maximum=-INFINITY;
 for(int j=0;j<N;j++){F score=0;for(int k=0;k<8;k++)score+=qkv[(b*N+i)*1536+h*8+k]*qkv[(b*N+j)*1536+512+h*8+k];score=score*.3535533905932738f+bias[((b*N+i)*N+j)*64+h];if(mask[b*N+j])score=-INFINITY;probs[((b*64+h)*N+i)*N+j]=score;maximum=fmaxf(maximum,score);}
 F sum=0;for(int j=0;j<N;j++){int p=((b*64+h)*N+i)*N+j;F z=expf(probs[p]-maximum);probs[p]=z;sum+=z;}for(int j=0;j<N;j++)probs[((b*64+h)*N+i)*N+j]/=sum;
 for(int k=0;k<8;k++){F s=0;for(int j=0;j<N;j++)s+=probs[((b*64+h)*N+i)*N+j]*qkv[(b*N+j)*1536+1024+h*8+k];out[(b*N+i)*512+h*8+k]=s;}}
int main(int argc,char**argv){try{if(argc<3)throw std::runtime_error("pairbias INPUT OUTPUT [repeats]");std::string in=argv[1],out=argv[2];std::filesystem::create_directories(out);int B,N,E;std::ifstream(in+"/shape.txt")>>B>>N>>E;int repeat=argc>3?std::stoi(argv[3]):7;size_t pairs=size_t(B)*N*N;hipDeviceProp_t prop;CK(hipGetDeviceProperties(&prop,0));
 auto f=[&](std::string n,size_t sz){return load<F>(in+"/"+n+".bin",sz);};
 Mem<F>d(f("distance",pairs)),mu(f("mu",128)),sd(f("std",128)),mul(f("mul",E)),bi(f("edge_bias",E)),w1(f("w1",128*128)),b1(f("b1",128)),w2(f("w2",64*128)),b2(f("b2",64)),x(f("normalized_input",size_t(B)*N*512)),wq(f("qkv_weight",1536*512)),bq(f("qkv_bias",1536));
 Mem<int>edge(load<int>(in+"/edge.bin",pairs)),mask(load<int>(in+"/mask.bin",size_t(B)*N));Mem<F>g(pairs*128),hidden(pairs*128),bias(pairs*64),qkv(size_t(B)*N*1536),probs(pairs*64),context(size_t(B)*N*512);
 auto run=[&]{gaussian_kernel<<<(pairs*128+255)/256,256>>>(d.p,edge.p,mu.p,sd.p,mul.p,bi.p,g.p,pairs);linear_gelu<<<(pairs*128+255)/256,256>>>(g.p,w1.p,b1.p,hidden.p,pairs,128,128,1);linear_gelu<<<(pairs*64+255)/256,256>>>(hidden.p,w2.p,b2.p,bias.p,pairs,128,64,0);linear_gelu<<<(B*N*1536+255)/256,256>>>(x.p,wq.p,bq.p,qkv.p,B*N,512,1536,0);attention<<<(B*64*N+127)/128,128>>>(qkv.p,bias.p,mask.p,probs.p,context.p,B,N);};
 run();CK(hipDeviceSynchronize());std::vector<double>ms;for(int i=0;i<repeat;i++){hipEvent_t a,b;CK(hipEventCreate(&a));CK(hipEventCreate(&b));CK(hipEventRecord(a,0));run();CK(hipEventRecord(b,0));CK(hipEventSynchronize(b));float m;CK(hipEventElapsedTime(&m,a,b));ms.push_back(m);hipEventDestroy(a);hipEventDestroy(b);}std::sort(ms.begin(),ms.end());g.save(out+"/gaussian.bin");bias.save(out+"/bias.bin");qkv.save(out+"/qkv.bin");probs.save(out+"/attention.bin");context.save(out+"/context.bin");std::ofstream log(out+"/timing.json");log<<std::setprecision(10)<<"{\"device\":\""<<prop.name<<"\",\"median_ms\":"<<ms[ms.size()/2]<<",\"repeats\":"<<repeat<<",\"batch\":"<<B<<",\"padded_atoms\":"<<N<<",\"dtype\":\"float32\",\"kernel_scope\":\"Gaussian128 + Linear128/GELU/Linear64 + QKV1536 + masked_softmax + attention_V; pre-normalized input exported from official layernorm\"}";std::cout<<"PASS forward "<<ms[ms.size()/2]<<" ms\n";
}catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 1;}}
