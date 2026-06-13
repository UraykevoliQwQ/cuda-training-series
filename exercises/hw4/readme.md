## **1. 矩阵行和与列和**

第一个任务是使用 CUDA 创建一个简单的矩阵行求和与列求和应用程序。代码框架已在 *matrix_sums.cu* 中提供。请编辑该文件，重点关注标有 FIXME 的位置，使程序运行时输出类似：

```
row sums correct!
column sums correct!
```

编辑代码后，使用以下命令进行编译：

```
module load cuda
nvcc -o matrix_sums matrix_sums.cu
```

`module load` 命令用于选择 CUDA 编译器。每次会话或登录只需执行一次。*nvcc* 是调用 CUDA 编译器的命令，其语法通常与 gcc/g++ 类似。

使用以下 LSF 命令运行代码：

```
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 ./matrix_sums
```

也可以为 bsub 命令创建别名，以便后续运行：

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun ./matrix_sums
```

在 NERSC 的 Cori 上，可以使用 Slurm：

```
module load esslurm
srun -C gpu -N 1 -n 1 -t 10 -A m3502 --reservation cuda_training --gres=gpu:1 -c 10 ./matrix_sums
```

`m3502` 是专为 Cori 上的本培训系列设置的资源配额，提前注册的参与者应当可以使用。如果无法使用此配额提交作业，但你已经拥有其他可访问 Cori GPU 节点的配额（例如 m1759），也可以改用该配额。

如果愿意，也可以在交互式会话中预留一块 GPU，并在 Slurm 资源分配有效期间多次运行可执行文件（如果有足够的可用节点，推荐采用这种方式）：

```
salloc -C gpu -N 1 -t 60 -A m3502 --reservation cuda_training --gres=gpu:1 -c 10
srun -n 1 ./matrix_sums
```

每次登录会话只需执行一次 `module load esslurm`；该命令使你能够向 Cori GPU 节点提交作业。

如果遇到困难，可以查看 *matrix_sums_solution.cu* 中的完整示例。

## **2. 性能分析**

下面引入一个新工具：性能分析器（此处使用 Nsight Compute）。我们先用它测量核函数执行时间，再收集一些可能有助于解释观察结果的“指标”信息。

必须先完成任务 1。然后加载 Nsight Compute 模块：

```
module load nsight-compute
```

接着按如下方式启动 Nsight：
（可以适当加宽终端窗口，以便阅读输出。）

```
lsfrun nv-nsight-cu-cli ./matrix_sums
```

输出提供了哪些信息？
你能找到标识核函数运行时间的行吗？
两个核函数的运行时间相同还是不同？
你原本预期它们相同还是不同？

接下来，按如下方式启动 *Nsight*：

```
lsfrun nv-nsight-cu-cli --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./matrix_sums
```

我们的目标是测量核函数的全局内存加载效率。这里请求了两个指标：*l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum*（全局内存加载请求数）和 *l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum*（全局加载所请求的扇区数）。前一个指标是目标度量“每次请求的事务数”的分母（请求数），后一个指标是分子（事务数）。两者相除即可得到每次请求的事务数。

*row_sum* 和 *column_sum* 核函数之间有哪些相同点或不同点？
两个核函数（*row_sum*、*column_sum*）的效率相同还是不同？
为什么？
这与第一次性能分析中观察到的核函数执行时间有何对应关系？

我们能否改进它？（请关注下一次 CUDA 培训课程。）

以下博客有助于熟悉 Nsight Compute：https://devblogs.nvidia.com/using-nsight-compute-to-inspect-your-kernels/
