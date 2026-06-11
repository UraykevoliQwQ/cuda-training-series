#include <cooperative_groups.h>
#include <stdio.h>
using namespace cooperative_groups;
const int nTPB = 256;
__device__ int reduce(thread_group g, int *x, int val) { 
  int lane = g.thread_rank();
  for (int i = g.size()/2; i > 0; i /= 2) {
    x[lane] = val;       g.sync();
    if (lane < i) val += x[lane + i];  g.sync();
  }
  if (g.thread_rank() == 0) printf("group partial sum: %d\n", val);
  return val;
}

__global__ void my_reduce_kernel(int *data){

  __shared__ int sdata[nTPB];
  // 任务 1a：在下方创建正确的线程块组
  auto g1 = this_thread_block();
  size_t gindex = g1.group_index().x * nTPB + g1.thread_index().x;
  // 任务 1b：取消下方注释，并使用上面创建的 g1 创建正确的 32 线程 tile
  auto g2 = tiled_partition(g1, 32);
  // 任务 1c：取消下方注释，并使用上面创建的 g2 创建正确的 16 线程 tile
  auto g3 = tiled_partition(g2, 16);
  // 每个任务都要调整 group，使其指向上方最后创建的组
  auto g = g3;
  // 确保传入合适的一段共享内存
  int sdata_offset = (g1.thread_index().x / g.size()) * g.size();
  reduce(g, sdata + sdata_offset, data[gindex]);
}

int main(){

  int *data;
  cudaMallocManaged(&data, nTPB*sizeof(data[0]));
  for (int i = 0; i < nTPB; i++) data[i] = 1;
  my_reduce_kernel<<<1,nTPB>>>(data);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) printf("cuda error: %s\n", cudaGetErrorString(err));
}

