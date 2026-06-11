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


const int DSIZE = 32*1048576;
// 向量加法核函数：C = A + B
__global__ void vadd(const float *A, const float *B, float *C, int ds){

  for (int idx = threadIdx.x+blockDim.x*blockIdx.x; idx < ds; idx+=gridDim.x*blockDim.x)         // 网格步长循环
    FIXME         // 在这里执行向量逐元素加法
}

int main(){

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
  h_A = new float[DSIZE];  // 在主机内存中为向量分配空间
  h_B = new float[DSIZE];
  h_C = new float[DSIZE];
  for (int i = 0; i < DSIZE; i++){  // 在主机内存中初始化向量
    h_A[i] = rand()/(float)RAND_MAX;
    h_B[i] = rand()/(float)RAND_MAX;
    h_C[i] = 0;}
  cudaMalloc(&d_A, DSIZE*sizeof(float));  // 为向量 A 分配设备端空间
  FIXME // 为向量 B 分配设备端空间
  FIXME // 为向量 C 分配设备端空间
  cudaCheckErrors("cudaMalloc failure"); // 错误检查
  // 将向量 A 复制到设备端：
  cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  // 将向量 B 复制到设备端：
  FIXME
  cudaCheckErrors("cudaMemcpy H2D failure");
  //CUDA 处理序列第 1 步完成
  int blocks = 1;  // 修改这一行进行实验
  int threads = 1; // 修改这一行进行实验
  vadd<<<blocks, threads>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");
  //CUDA 处理序列第 2 步完成
  // 将向量 C 从设备端复制回主机端：
  FIXME
  //CUDA 处理序列第 3 步完成
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  printf("A[0] = %f\n", h_A[0]);
  printf("B[0] = %f\n", h_B[0]);
  printf("C[0] = %f\n", h_C[0]);
  return 0;
}
