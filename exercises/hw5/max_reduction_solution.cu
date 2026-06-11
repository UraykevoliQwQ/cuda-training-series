#include <stdio.h>

// 错误检查宏
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


const size_t N = 8ULL*1024ULL*1024ULL;  // 数据大小
const int BLOCK_SIZE = 256;  // CUDA 最大值为 1024

__global__ void reduce(float *gdata, float *out, size_t n){
     __shared__ float sdata[BLOCK_SIZE];
     int tid = threadIdx.x;
     sdata[tid] = 0.0f;
     size_t idx = threadIdx.x+blockDim.x*blockIdx.x;

     while (idx < n) {  // 使用网格步长循环加载数据
        sdata[tid] = max(gdata[idx], sdata[tid]);
        idx += gridDim.x*blockDim.x;  
        }

     for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        __syncthreads();
        if (tid < s)  // 并行扫描式归约
            sdata[tid] = max(sdata[tid + s], sdata[tid]);
        }
     if (tid == 0) out[blockIdx.x] = sdata[0];
  }

int main(){

  float *h_A, *h_sum, *d_A, *d_sums;
  const int blocks = 640;
  h_A = new float[N];  // 在主机内存中为数据分配空间
  h_sum = new float;
  float max_val = 5.0f;
  for (size_t i = 0; i < N; i++)  // 在主机内存中初始化矩阵
    h_A[i] = 1.0f;
  h_A[100] = max_val;
  cudaMalloc(&d_A, N*sizeof(float));  // 为 A 分配设备端空间
  cudaMalloc(&d_sums, blocks*sizeof(float));  // 为部分和分配设备端空间
  cudaCheckErrors("cudaMalloc failure"); // 错误检查
  // 将矩阵 A 复制到设备端：
  cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  //CUDA 处理序列第 1 步完成
  reduce<<<blocks, BLOCK_SIZE>>>(d_A, d_sums, N); // 归约第 1 阶段
  cudaCheckErrors("reduction kernel launch failure");
  reduce<<<1, BLOCK_SIZE>>>(d_sums, d_A, blocks); // 归约第 2 阶段
  cudaCheckErrors("reduction kernel launch failure");
  //CUDA 处理序列第 2 步完成
  // 将向量 sums 从设备端复制回主机端：
  cudaMemcpy(h_sum, d_A, sizeof(float), cudaMemcpyDeviceToHost);
  //CUDA 处理序列第 3 步完成
  cudaCheckErrors("reduction w/atomic kernel execution failure or cudaMemcpy D2H failure");
  printf("reduction output: %f, expected sum reduction output: %f, expected max reduction output: %f\n", *h_sum, (float)((N-1)+max_val), max_val);
  return 0;
}
