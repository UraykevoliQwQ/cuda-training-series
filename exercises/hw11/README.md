# 多进程服务

在 Cori GPU 上，首先获取一个交互式会话。请确保为 MPI 申请至少几个进程槽位，但这里只需要一块 GPU。

```
module purge
module load cgpu gcc/8.3.0 cuda/11.4.0 openmpi/4.0.3
salloc -A ntrain -q shared --reservation=cuda_mps -C gpu -N 1 -n 4 -t 60 -c 4 --gpus=1
```

课程中使用的测试代码位于 `test.cu`，可使用以下命令编译：

```
nvcc -o test -ccbin=mpicxx test.cu
```

如果运行环境中没有 MPI，可以按如下方式编译不使用 MPI 的应用程序：

```
nvcc -DNO_MPI -o test test.cu
```

在下面所有示例中，请使用所提供的 `run_no_mpi.sh` 脚本代替 `mpirun`。该脚本会启动同一进程的 4 个副本。对于 Summit 等从计算节点之外的其他节点启动作业的系统，此脚本也可能很有用，因为 `jsrun ... nsys` 比 `nsys jsrun ...` 更合适。

## 验证课程中的结论

本练习要求尝试课程中的一些实验，并查看能否复现其结论。首先在不使用 MPS 的情况下运行以下实验（请注意，该应用程序大约需要 20 秒，请耐心等待）：

```
nsys profile --stats=true -t nvtx,cuda -s none -o 1_rank_no_MPS_N_1e9 -f true mpirun -np 1 ./test 1073741824
nsys profile --stats=true -t nvtx,cuda -s none -o 4_ranks_no_MPS_N_1e9 -f true mpirun -np 4 ./test 1073741824
```

根据应用程序的标准输出和性能分析数据，确认在同一块 GPU 上使用 4 个进程时，核函数平均运行时间更长。

现在启动 MPS，并使用 4 个进程重复上述实验，确认核函数平均运行时间与单进程情况大致相同。请同时查看标准输出和性能分析数据。

```
nvidia-cuda-mps-control -d
nsys profile --stats=true -t nvtx,cuda -s none -o 4_ranks_with_MPS_N_1e9 -f true mpirun -np 4 ./test 1073741824
```

接着确认停止 MPS 后，原来的行为会恢复。

```
echo "quit" | nvidia-cuda-mps-control
nsys profile --stats=true -t nvtx,cuda -s none -o 4_ranks_no_MPS_N_1e9 -f true mpirun -np 4 ./test 1073741824
```

## 尝试不同的问题规模

改变问题规模 `N`，直到找到一个最小规模，使你能够明确判断 MPS 相较默认计算模式具有明显优势。
