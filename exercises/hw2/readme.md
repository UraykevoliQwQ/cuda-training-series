# 作业 2

这些练习将帮助你巩固对 GPU 共享内存概念的理解。

## **1. 使用共享内存的一维模板计算**

第一个任务是创建一个使用共享内存的一维模板计算应用程序。代码框架位于 *stencil_1d.cu*。请编辑该文件，重点关注标有 FIXME 的位置。代码会验证输出并报告所有错误。

编辑代码后，使用以下命令进行编译：

```
module load cuda
nvcc -o stencil_1d stencil_1d.cu
```

`module load` 命令用于选择要使用的 CUDA 编译器。每次会话或登录只需执行一次该命令。*nvcc* 是调用 CUDA 编译器的命令，其语法通常与 gcc/g++ 类似。

我们将使用以下 LSF 命令运行代码：

```
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 ./stencil_1d
```

你也可以为 *bsub* 命令创建别名，以便后续运行：

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun ./stencil_1d
```

在 NERSC 的 Cori 上，可以使用 Slurm 运行代码：

```
module load esslurm
srun -C gpu -N 1 -n 1 -t 10 -A m3502 --reservation cuda_training --gres=gpu:1 -c 10 ./stencil_1d
```

`m3502` 是专为 Cori 上的本培训系列设置的资源配额，提前注册的参与者应当可以使用。如果无法使用此配额提交作业，但你已经拥有其他可访问 Cori GPU 节点的配额（例如 m1759），也可以改用该配额。

如果愿意，也可以在交互式会话中预留一块 GPU，并在 Slurm 资源分配有效期间多次运行可执行文件（如果有足够的可用节点，推荐采用这种方式）：

```
salloc -C gpu -N 1 -t 60 -A m3502 --reservation cuda_training --gres=gpu:1 -c 10
srun -n 1 ./stencil_1d
```

请注意，每次登录会话只需执行一次 `module load esslurm`；该命令使你能够向 Cori GPU 节点提交作业。

如果遇到困难，可以查看 *stencil_1d_solution* 中的完整示例。

## **2. 使用共享内存的二维矩阵乘法**

接下来，将共享内存应用到作业 1 中编写的二维矩阵乘法。*matrix_mul_shared.cu* 的代码框架中标出了 FIXME 位置。请尝试将所需数据正确加载到共享内存中，并相应地更新点积计算。使用以下命令编译并运行代码：

```
module load cuda
nvcc -o matrix_mul matrix_mul_shared.cu
lsfrun ./matrix_mul
```

请注意，程序中包含计时信息。重新运行作业 1 中的解决方案并观察运行时间。将共享内存应用到该二维矩阵乘法后，你注意到了怎样的运行时间变化？它与此前实现的运行时间有何不同？

如果遇到困难，可以查看 *matrix_mul_shared_solution* 中的完整示例。
