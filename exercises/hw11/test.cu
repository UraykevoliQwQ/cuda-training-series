#ifndef NO_MPI
#include <mpi.h>
#endif
#include <cstdio>
#include <chrono>
#include <iostream>

__global__ void kernel (double* x, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        x[i] = 2 * x[i];
    }
}

int main(int argc, char** argv) {
#ifndef NO_MPI
    int rank, num_ranks;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    // 总问题规模
    size_t N = 1024 * 1024 * 1024;

    if (argc >= 2) {
        N = atoi(argv[1]);
    }

#ifdef NO_MPI
    // 如果不使用 MPI，则通过命令行指定有多少个“rank”
    int num_ranks = 1;
    if (argc >= 3) {
        num_ranks = atoi(argv[2]);
    }
#endif

    // 每个 rank 的问题规模（假设 N 可整除）
    size_t N_per_rank = N / num_ranks;

    double* x;
    cudaMalloc((void**) &x, N_per_rank * sizeof(double));

    // 重复次数

    const int num_reps = 1000;

    using namespace std::chrono;

    auto start = high_resolution_clock::now();

    int threads_per_block = 256;
    size_t blocks = (N_per_rank + threads_per_block - 1) / threads_per_block;

    for (int i = 0; i < num_reps; ++i) {
        kernel<<<blocks, threads_per_block>>>(x, N_per_rank);
        cudaDeviceSynchronize();
    }

    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);

    std::cout << "Time per kernel = " << duration.count() / (double) num_reps << " ms " << std::endl;

#ifndef NO_MPI
    MPI_Finalize();
#endif
}
