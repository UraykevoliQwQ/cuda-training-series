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


const size_t DSIZE = 16384;      // 矩阵边长
const int block_size = 256;  // CUDA 最大值为 1024
// 矩阵行求和核函数
__global__ void row_sums(const float *A, float *sums, size_t ds){

  int idx = threadIdx.x+blockDim.x*blockIdx.x; // 用内置变量创建典型的一维线程索引
  if (idx < ds){
    float sum = 0.0f;
    for (size_t i = 0; i < ds; i++)
      sum += A[idx*ds+i];         // 编写一个 for 循环，让线程遍历一整行、持续累加，并将结果写入 sums
    sums[idx] = sum;
}}
// 矩阵列求和核函数
__global__ void column_sums(const float *A, float *sums, size_t ds){

  int idx = threadIdx.x+blockDim.x*blockIdx.x; // 用内置变量创建典型的一维线程索引
  if (idx < ds){
    float sum = 0.0f;
    for (size_t i = 0; i < ds; i++)
      sum += A[idx+ds*i];         // 编写一个 for 循环，让线程沿列向下遍历、持续累加，并将结果写入 sums
    sums[idx] = sum;
}}
bool validate(float *data, size_t sz){
  for (size_t i = 0; i < sz; i++)
    if (data[i] != (float)sz) {printf("results mismatch at %lu, was: %f, should be: %f\n", i, data[i], (float)sz); return false;}
    return true;
}
int main(){

  float *h_A, *h_sums, *d_A, *d_sums;
  h_A = new float[DSIZE*DSIZE];  // 在主机内存中为数据分配空间
  h_sums = new float[DSIZE]();
  for (int i = 0; i < DSIZE*DSIZE; i++)  // 在主机内存中初始化矩阵
    h_A[i] = 1.0f;
  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));  // 为 A 分配设备端空间
  cudaMalloc(&d_sums, DSIZE*sizeof(float));  // 为向量 d_sums 分配设备端空间
  cudaCheckErrors("cudaMalloc failure"); // 错误检查
  // 将矩阵 A 复制到设备端：
  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  //CUDA 处理序列第 1 步完成
  row_sums<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");
  //CUDA 处理序列第 2 步完成
  // 将向量 sums 从设备端复制回主机端：
  cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  //CUDA 处理序列第 3 步完成
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  if (!validate(h_sums, DSIZE)) return -1; 
  printf("row sums correct!\n");
  cudaMemset(d_sums, 0, DSIZE*sizeof(float));
  column_sums<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");
  //CUDA 处理序列第 2 步完成
  // 将向量 sums 从设备端复制回主机端：
  cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  //CUDA 处理序列第 3 步完成
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  if (!validate(h_sums, DSIZE)) return -1; 
  printf("column sums correct!\n");
  return 0;
}
  
