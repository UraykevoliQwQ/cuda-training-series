#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdio>

// 交错调和级数：https://en.wikipedia.org/wiki/Harmonic_series_(mathematics)#Alternating_harmonic_series
// 根据索引 n 计算交错调和级数项
__device__ auto ahs(size_t n){ return ((n&1)?1:-1)/(double)n;}

// blocksize 必须是 2 的幂，且小于或等于 1024
#define BLOCK_SIZE 512

// 估计交错调和级数的和
template <typename T>
__global__ void estimate_sum_ahs(size_t length, T *sum){
  __shared__ T smem[BLOCK_SIZE];
  size_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  smem[threadIdx.x] = (idx < length)?ahs(idx):0;
  if (idx == 0) smem[0] = 0;

  for (int i = blockDim.x>>1; i > 0; i >>= 1){
    __syncthreads();
    if (threadIdx.x < i) smem[threadIdx.x] += smem[threadIdx.x+i];}

  if (threadIdx.x == 0) atomicAdd(sum, smem[0]);
}

typedef double ft;

int main(int argc, char* argv[]){
  size_t my_length = 1048576; // 允许用户通过命令行参数覆盖默认估计长度
  if (argc > 1) my_length = atol(argv[1]);
  ft *sum;
  cudaError_t err = cudaMallocManaged(&sum, sizeof(ft));
  if (err != cudaSuccess) {std::cout << "Error: " << cudaGetErrorString(err) << std::endl; return 0;}
  *sum = 0;
  dim3 block(BLOCK_SIZE);
  dim3 grid((my_length+block.x-1)/block.x);
  estimate_sum_ahs<<<grid, block>>>(my_length, sum);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {std::cout << "Error: " << cudaGetErrorString(err) << std::endl; return 0;}
  std::cout << "Estimated value: " << *sum << " Expected value: "  << log(2)  << std::endl;
  return 0;
}
