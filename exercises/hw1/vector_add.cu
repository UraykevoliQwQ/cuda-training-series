#include <stdio.h>

// 错误检查宏
#define cudaCheckErrors(msg)                             \
  do                                                     \
  {                                                      \
    cudaError_t __err = cudaGetLastError();              \
    if (__err != cudaSuccess)                            \
    {                                                    \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
              msg, cudaGetErrorString(__err),            \
              __FILE__, __LINE__);                       \
      fprintf(stderr, "*** FAILED - ABORTING\n");        \
      exit(1);                                           \
    }                                                    \
  } while (0)

const int DSIZE = 4096;
const int block_size = 256; // CUDA 最大值为 1024
// 向量加法核函数：C = A + B
__global__ void vadd(const float *A, const float *B, float *C, int ds)
{

  int idx = threadIdx.x + blockDim.x * blockIdx.x; // 用内置变量创建典型的一维线程索引
  if (idx < ds)
    C[idx] = A[idx] + B[idx]; // 在这里执行向量逐元素加法
}

int main()
{

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
  h_A = new float[DSIZE]; // 在主机内存中为向量分配空间
  h_B = new float[DSIZE];
  h_C = new float[DSIZE];
  for (int i = 0; i < DSIZE; i++)
  { // 在主机内存中初始化向量
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
    h_C[i] = 0;
  }
  cudaMalloc(&d_A, DSIZE * sizeof(float)); // 为向量 A 分配设备端空间
  cudaMalloc(&d_B, DSIZE * sizeof(float)); // 为向量 B 分配设备端空间
  cudaMalloc(&d_C, DSIZE * sizeof(float)); // 为向量 C 分配设备端空间
  cudaCheckErrors("cudaMalloc failure");   // 错误检查
  // 将向量 A 复制到设备端：
  cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  // 将向量 B 复制到设备端：
  cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

  cudaCheckErrors("cudaMemcpy H2D failure");
  // CUDA 处理序列第 1 步完成
  vadd<<<(DSIZE + block_size - 1) / block_size, block_size>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");
  // CUDA 处理序列第 2 步完成
  //  将向量 C 从设备端复制回主机端：
  cudaMemcpy(h_C, d_C, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
  // CUDA 处理序列第 3 步完成
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  printf("A[0] = %f\n", h_A[0]);
  printf("B[0] = %f\n", h_B[0]);
  printf("C[0] = %f\n", h_C[0]);
  return 0;
}
