#include <stdio.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

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

#define N 500000

// 简单的短小核函数
__global__
void kernel_a(float* x, float* y){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] += 1;
}

__global__
void kernel_c(float* x, float* y){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] += 1;
}

int main(){

cudaStream_t stream1;

cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);

cublasHandle_t cublas_handle;
cublasCreate(&cublas_handle);
cublasSetStream(cublas_handle, stream1);

// 设置主机端数据并初始化
float* h_x;
float* h_y;

h_x = (float*) malloc(N * sizeof(float));
h_y = (float*) malloc(N * sizeof(float));

for (int i = 0; i < N; ++i){
    h_x[i] = float(i);
    h_y[i] = float(i);
}

// 打印 h_y 的前 25 个值
for (int i = 0; i < 25; ++i){
    printf("%2.0f ", h_y[i]);
}
printf("\n");

// 设置设备端数据
float* d_x;
float* d_y;
float d_a = 5.0;

cudaMalloc((void**) &d_x, N * sizeof(float));
cudaMalloc((void**) &d_y, N * sizeof(float));
cudaCheckErrors("cudaMalloc failed");

cublasSetVector(N, sizeof(h_x[0]), h_x, 1, d_x, 1); // 类似于 cudaMemcpyHtoD
cublasSetVector(N, sizeof(h_y[0]), h_y, 1, d_y, 1); // 类似于 cudaMemcpyHtoD
cudaCheckErrors("cublasSetVector failed");

int threads = 512;
int blocks = (N + (threads - 1) / threads);

for (int i = 0; i < 100; ++i){
    kernel_a<<<blocks, thread, 0, stream1>>>(d_x, d_y)

    // 库调用
    cublasSaxpy(cublas_handle, N, &d_a, d_x, 1, d_y, 1);

    kernel_a<<<blocks, thread, 0, stream1>>>(d_x, d_y)

    cudaStreamSynchronize(stream1);
}
cudaDeviceSynchronize();

// 将内存复制回主机端
cudaMemcpy(h_y, d_y, N, cudaMemcpyDeviceToHost);
cudaCheckErrors("Finishing memcpy failed");

cudaDeviceSynchronize();

// 打印 h_y 的前 25 个值
for (int i = 0; i < 25; ++i){
    printf("%2.0f ", h_y[i]);
}
printf("\n");

return 0;

}
