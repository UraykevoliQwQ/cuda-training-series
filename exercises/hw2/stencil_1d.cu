#include <stdio.h>
#include <algorithm>

using namespace std;

#define N 4096
#define RADIUS 3
#define BLOCK_SIZE 16

__global__ void stencil_1d(int *in, int *out)
{
  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = threadIdx.x + RADIUS;

  // 将输入元素读入共享内存
  temp[lindex] = in[gindex];
  if (threadIdx.x < RADIUS)
  {
    temp[lindex - RADIUS] = in[gindex - RADIUS];
    temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
  }

  // 同步（确保所有数据都已可用）
  __syncthreads();

  // 应用 stencil 计算
  int result = 0;
  for (int offset = -RADIUS; offset <= RADIUS; offset++)
    result += temp[lindex + offset];

  // 存储结果
  out[gindex] = result;
}

void fill_ints(int *x, int n)
{
  fill_n(x, n, 1);
}

int main(void)
{
  int *in, *out;     // a、b、c 的主机端副本
  int *d_in, *d_out; // a、b、c 的设备端副本

  // 为主机端副本分配空间并设置初值
  int size = (N + 2 * RADIUS) * sizeof(int);
  in = (int *)malloc(size);
  fill_ints(in, N + 2 * RADIUS);
  out = (int *)malloc(size);
  fill_ints(out, N + 2 * RADIUS);

  // 为设备端副本分配空间
  cudaMalloc((void **)&d_in, size);
  cudaMalloc((void **)&d_out, size);

  // 复制到设备端
  cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

  // 在 GPU 上启动 stencil_1d() 核函数
  stencil_1d<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_in + RADIUS, d_out + RADIUS);

  // 将结果复制回主机端
  cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

  // 错误检查
  for (int i = 0; i < N + 2 * RADIUS; i++)
  {
    if (i < RADIUS || i >= N + RADIUS)
    {
      if (out[i] != 1)
        printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1);
    }
    else
    {
      if (out[i] != 1 + 2 * RADIUS)
        printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1 + 2 * RADIUS);
    }
  }

  // 清理
  free(in);
  free(out);
  cudaFree(d_in);
  cudaFree(d_out);
  printf("Success!\n");
  return 0;
}
