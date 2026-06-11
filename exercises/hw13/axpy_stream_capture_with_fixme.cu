#include <stdio.h>
#include <cuda_runtime_api.h>
#include <ctime>
#include <ratio>
#include <chrono>
#include <iostream>

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
void kernel_a(float * x, float * y){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 2.0*x[idx] + y[idx];

}

__global__
void kernel_b(float * x, float * y){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 2.0*x[idx] + y[idx];

}

__global__
void kernel_c(float * x, float * y){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 2.0*x[idx] + y[idx];

}

__global__
void kernel_d(float * x, float * y){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 2.0*x[idx] + y[idx];

}

int main(){

// 设置并创建事件
cudaEvent_t event1;
cudaEvent_t event2;

cudaEventCreateWithFlags(&event1, cudaEventDisableTiming);
cudaEventCreateWithFlags(&event2, cudaEventDisableTiming);

// 设置并创建流
const int num_streams = 2;

cudaStream_t streams[num_streams];

for (int i = 0; i < num_streams; ++i){
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
}

// 设置并初始化主机端数据
float* h_x;
float* h_y;

h_x = (float*) malloc(N * sizeof(float));
h_y = (float*) malloc(N * sizeof(float));

for (int i = 0; i < N; ++i){
    h_x[i] = (float)i;
    h_y[i] = (float)i;
//    printf("%2.0f ", h_x[i]);
}
printf("\n");

// 设置设备端数据
float* d_x;
float* d_y;

cudaMalloc((void**) &d_x, N * sizeof(float));
cudaMalloc((void**) &d_y, N * sizeof(float));
cudaCheckErrors("cudaMalloc failed");

cudaMemcpy(d_x, h_x, N, cudaMemcpyHostToDevice);
cudaMemcpy(d_y, h_y, N, cudaMemcpyHostToDevice);
cudaCheckErrors("cudaMalloc failed");

// 设置图
bool graphCreated=false;
cudaGraph_t graph;
cudaGraphExec_t instance;

// FIXME
cudaGraphCreate(FIXME, 0);

int threads = 512;
int blocks = (N + (threads - 1) / threads);

// 启动工作负载
for (int i = 0; i < 100; ++i){
    if (graphCreated == false){
    // 如果是第一次执行，则开始流捕获
        cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal);
        cudaCheckErrors("Stream begin capture failed");

        kernel_a<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);
        cudaCheckErrors("Kernel a failed");

        cudaEventRecord(event1, streams[0]);
        cudaCheckErrors("Event record failed");

        kernel_b<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);
        cudaCheckErrors("Kernel b failed");

        cudaStreamWaitEvent(streams[1], event1);
        cudaCheckErrors("Event wait failed");

        kernel_c<<<blocks, threads, 0, streams[1]>>>(d_x, d_y);
        cudaCheckErrors("Kernel c failed");

        cudaEventRecord(event2, streams[1]);
        cudaCheckErrors("Event record failed");

        cudaStreamWaitEvent(streams[0], event2);
        cudaCheckErrors("Event wait failed");

        kernel_d<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);
        cudaCheckErrors("Kernel d failed");

        cudaStreamEndCapture(streams[0], &graph);
        cudaCheckErrors("Stream end capture failed");

        // 创建图实例
        // FIXME
        cudaGraphInstantiate(FIXME, graph, NULL, NULL, 0);
        cudaCheckErrors("instantiating graph failed");

        // FIXME
        graphCreated = FIXME;
    }
// 启动图实例
// FIXME
cudaGraphLaunch(FIXME, streams[0]);
cudaCheckErrors("Launching graph failed");
cudaStreamSynchronize(streams[0]);
}

// 统计图中节点数量
cudaGraphNode_t *nodes = NULL;
size_t numNodes = 0;
cudaGraphGetNodes(graph, nodes, &numNodes);
cudaCheckErrors("Graph get nodes failed");
printf("Number of the nodes in the graph = %zu\n", numNodes);

// 以下用于计时
cudaDeviceSynchronize();

using namespace std::chrono;

high_resolution_clock::time_point t1 = high_resolution_clock::now();

for (int i = 0; i < 1000; ++i){
cudaGraphLaunch(instance, streams[0]);
cudaCheckErrors("Launching graph failed");
}

cudaDeviceSynchronize();
high_resolution_clock::time_point t2 = high_resolution_clock::now();

duration<double> total_time = duration_cast<duration<double>>(t2 - t1);

std::cout << "Time " << total_time.count() << " s" << std::endl;

// 将数据复制回主机端
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
