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
//const size_t N = 256*640; // 数据大小
const int BLOCK_SIZE = 256;  // CUDA 最大值为 1024
// 朴素原子归约核函数
__global__ void atomic_red(const float *gdata, float *out){
  size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx < N) atomicAdd(out, gdata[idx]);
}

__global__ void reduce(float *gdata, float *out){
     __shared__ float sdata[BLOCK_SIZE];
     int tid = threadIdx.x;
     sdata[tid] = 0.0f;
     size_t idx = threadIdx.x+blockDim.x*blockIdx.x;

     while (idx < N) {  // 使用网格步长循环加载数据
        sdata[tid] += gdata[idx];
        idx += gridDim.x*blockDim.x;  
        }

     for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        __syncthreads();
        if (tid < s)  // 并行扫描式归约
            sdata[tid] += sdata[tid + s];
        }
     if (tid == 0) out[blockIdx.x] = sdata[0];
  }

 __global__ void reduce_a(float *gdata, float *out){
     __shared__ float sdata[BLOCK_SIZE];
     int tid = threadIdx.x;
     sdata[tid] = 0.0f;
     size_t idx = threadIdx.x+blockDim.x*blockIdx.x;

     while (idx < N) {  // 使用网格步长循环加载数据
        sdata[tid] += gdata[idx];
        idx += gridDim.x*blockDim.x;  
        }

     for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        __syncthreads();
        if (tid < s)  // 并行扫描式归约
            sdata[tid] += sdata[tid + s];
        }
     if (tid == 0) atomicAdd(out, sdata[0]);
  }


__global__ void reduce_ws(float *gdata, float *out){
     __shared__ float sdata[32];
     int tid = threadIdx.x;
     int idx = threadIdx.x+blockDim.x*blockIdx.x;
     float val = 0.0f;
     unsigned mask = 0xFFFFFFFFU;
     int lane = threadIdx.x % warpSize;
     int warpID = threadIdx.x / warpSize;
     while (idx < N) {  // 使用网格步长循环加载
        val += gdata[idx];
        idx += gridDim.x*blockDim.x;  
        }

 // 第 1 次 warp shuffle 归约
    for (int offset = warpSize/2; offset > 0; offset >>= 1) 
       val += __shfl_down_sync(mask, val, offset);
    if (lane == 0) sdata[warpID] = val;
   __syncthreads(); // 将 warp 结果放入共享内存

// 之后只使用 warp 0
    if (warpID == 0){
 // 如果对应 warp 存在，则从共享内存重新加载 val
       val = (tid < blockDim.x/warpSize)?sdata[lane]:0;

 // 最终 warp shuffle 归约
       for (int offset = warpSize/2; offset > 0; offset >>= 1) 
          val += __shfl_down_sync(mask, val, offset);

       if  (tid == 0) atomicAdd(out, val);
     }
  }




int main(){

  float *h_A, *h_sum, *d_A, *d_sum;
  h_A = new float[N];  // 在主机内存中为数据分配空间
  h_sum = new float;
  for (int i = 0; i < N; i++)  // 在主机内存中初始化矩阵
    h_A[i] = 1.0f;
  cudaMalloc(&d_A, N*sizeof(float));  // 为 A 分配设备端空间
  cudaMalloc(&d_sum, sizeof(float));  // 为 sum 分配设备端空间
  cudaCheckErrors("cudaMalloc failure"); // 错误检查
  // 将矩阵 A 复制到设备端：
  cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  cudaMemset(d_sum, 0, sizeof(float));
  cudaCheckErrors("cudaMemset failure");
  //CUDA 处理序列第 1 步完成
  atomic_red<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_sum);
  cudaCheckErrors("atomic reduction kernel launch failure");
  //CUDA 处理序列第 2 步完成
  // 将向量 sums 从设备端复制回主机端：
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  //CUDA 处理序列第 3 步完成
  cudaCheckErrors("atomic reduction kernel execution failure or cudaMemcpy H2D failure");
  if (*h_sum != (float)N) {printf("atomic sum reduction incorrect!\n"); return -1;}
  printf("atomic sum reduction correct!\n");
  const int blocks = 640;
  cudaMemset(d_sum, 0, sizeof(float));
  cudaCheckErrors("cudaMemset failure");
  //CUDA 处理序列第 1 步完成
  reduce_a<<<blocks, BLOCK_SIZE>>>(d_A, d_sum);
  cudaCheckErrors("reduction w/atomic kernel launch failure");
  //CUDA 处理序列第 2 步完成
  // 将向量 sums 从设备端复制回主机端：
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  //CUDA 处理序列第 3 步完成
  cudaCheckErrors("reduction w/atomic kernel execution failure or cudaMemcpy H2D failure");
  if (*h_sum != (float)N) {printf("reduction w/atomic sum incorrect!\n"); return -1;}
  printf("reduction w/atomic sum correct!\n");
  cudaMemset(d_sum, 0, sizeof(float));
  cudaCheckErrors("cudaMemset failure");
  //CUDA 处理序列第 1 步完成
  reduce_ws<<<blocks, BLOCK_SIZE>>>(d_A, d_sum);
  cudaCheckErrors("reduction warp shuffle kernel launch failure");
  //CUDA 处理序列第 2 步完成
  // 将向量 sums 从设备端复制回主机端：
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  //CUDA 处理序列第 3 步完成
  cudaCheckErrors("reduction warp shuffle kernel execution failure or cudaMemcpy H2D failure");
  if (*h_sum != (float)N) {printf("reduction warp shuffle sum incorrect!\n"); return -1;}
  printf("reduction warp shuffle sum correct!\n");
  return 0;
}
  
