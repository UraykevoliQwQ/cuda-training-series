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

// 设置图
cudaGraph_t graph; // 主图
cudaGraph_t libraryGraph; // cuBLAS 调用的子图
std::vector<cudaGraphNode_t> nodeDependencies;
cudaGraphNode_t kernelNode1, kernelNode2, libraryNode;

cudaKernelNodeParams kernelNode1Params {0};
cudaKernelNodeParams kernelNode2Params {0};

// FIXME 需要创建图
cudaGraphCreate(FIXME, 0); // 创建图
cudaCheckErrors("cudaGraphCreate failure");

// kernel_a 和 kernel_c 使用相同参数
void *kernelArgs[2] = {(void *)&d_x, (void *)&d_y};

int threads = 512;
int blocks = (N + (threads - 1) / threads);

// 添加第 1 个节点 kernel_a，作为图的头节点
kernelNode1Params.func = (void *)kernel_a;
kernelNode1Params.gridDim = dim3(blocks, 1, 1);
kernelNode1Params.blockDim = dim3(threads, 1, 1);
kernelNode1Params.sharedMemBytes = 0;
kernelNode1Params.kernelParams = (void **)kernelArgs;
kernelNode1Params.extra = NULL;

cudaGraphAddKernelNode(&kernelNode1, graph, NULL,
                         0, &kernelNode1Params);
cudaCheckErrors("Adding kernelNode1 failed");

nodeDependencies.push_back(kernelNode1); // 管理依赖向量

// 添加第 2 个节点 libraryNode，并依赖 kernelNode1
cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
cudaCheckErrors("Stream capture begin failure");

// 库调用
cublasSaxpy(cublas_handle, N, &d_a, d_x, 1, d_y, 1);
cudaCheckErrors("cublasSaxpy failure");

cudaStreamEndCapture(stream1, &libraryGraph);
cudaCheckErrors("Stream capture end failure");

// FIXME 需要修正 cudaGraphAddChildNode 调用
cudaGraphAddChildGraphNode(FIXME, graph, FIXME,
                             nodeDependencies.size(), libraryGraph);
cudaCheckErrors("Adding libraryNode failed");

nodeDependencies.clear();
nodeDependencies.push_back(libraryNode); // 管理依赖向量

// 添加第 3 个节点 kernel_c，并依赖 libraryNode
kernelNode2Params.func = (void *)kernel_c;
kernelNode2Params.gridDim = dim3(blocks, 1, 1);
kernelNode2Params.blockDim = dim3(threads, 1, 1);
kernelNode2Params.sharedMemBytes = 0;
kernelNode2Params.kernelParams = (void **)kernelArgs;
kernelNode2Params.extra = NULL;

cudaGraphAddKernelNode(&kernelNode2, graph, nodeDependencies.data(),
                         nodeDependencies.size(), &kernelNode2Params);
cudaCheckErrors("Adding kernelNode2 failed");

nodeDependencies.clear();
nodeDependencies.push_back(kernelNode2); // 管理依赖向量

cudaGraphNode_t *nodes = NULL;
size_t numNodes = 0;
cudaGraphGetNodes(graph, nodes, &numNodes);
cudaCheckErrors("Graph get nodes failed");
printf("Number of the nodes in the graph = %zu\n", numNodes);

cudaGraphExec_t instance;
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
cudaCheckErrors("Graph instantiation failed");

// 启动图实例 100 次
for (int i = 0; i < 100; ++i){
    // FIXME 需要启动图
    cudaGraphLaunch(FIXME, stream1);
    cudaStreamSynchronize(stream1);
}
cudaCheckErrors("Graph launch failed");
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
