#include <stdio.h>

// 这些仅用于计时测量
#include <time.h>

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


const int DSIZE = 4096;
const int block_size = 16;  // CUDA 每个线程块最多共有 1024 个线程
const float A_val = 1.0f;
const float B_val = 2.0f;

// 矩阵乘法（朴素）核函数：C = A * B
__global__ void mmul(const float *A, const float *B, float *C, int ds) {

  int idx = threadIdx.x+blockDim.x*blockIdx.x; // 创建线程 x 索引
  int idy = threadIdx.y+blockDim.y*blockIdx.y; // 创建线程 y 索引

  if ((idx < ds) && (idy < ds)){
    float temp = 0;
    for (int i = 0; i < ds; i++)
      temp += A[FIXME*ds+i] * B[i*ds+FIXME];   // 行与列的点积
    C[idy*ds+idx] = temp;
  }
}

int main(){

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

  // 这些仅用于计时
  clock_t t0, t1, t2;
  double t1sum=0.0;
  double t2sum=0.0;

  // 开始计时
  t0 = clock();

  h_A = new float[DSIZE*DSIZE];
  h_B = new float[DSIZE*DSIZE];
  h_C = new float[DSIZE*DSIZE];
  for (int i = 0; i < DSIZE*DSIZE; i++){
    h_A[i] = A_val;
    h_B[i] = B_val;
    h_C[i] = 0;}

  // 初始化计时
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("Init took %f seconds.  Begin compute\n", t1sum);

  // 分配设备内存并将输入数据复制到 GPU
  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&d_B, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&d_C, DSIZE*DSIZE*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // CUDA 处理序列第 1 步完成

  // 启动核函数
  dim3 block(block_size, block_size);  // dim3 变量保存 3 个维度
  dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);
  mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");

  // CUDA 处理序列第 2 步完成

  // 将结果复制回主机端
  cudaMemcpy(h_C, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

  // GPU 计时
  t2 = clock();
  t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
  printf ("Done. Compute took %f seconds\n", t2sum);

  // CUDA 处理序列第 3 步完成

  // 验证结果
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  for (int i = 0; i < DSIZE*DSIZE; i++) if (h_C[i] != A_val*B_val*DSIZE) {printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_val*B_val*DSIZE); return -1;}
  printf("Success!\n"); 

  return 0;
}
  
