# 作业 1

这些练习将引导你编写一些基础 CUDA 应用程序。你将学习如何分配 GPU 内存、在主机与 GPU 之间传输数据，以及启动核函数。

## **1. Hello World（你好，世界）**

第一个任务是使用 CUDA 创建一个简单的 Hello World 应用程序。`hello.cu` 中已经提供了代码框架。请编辑该文件，重点关注标有 FIXME 的位置，使程序运行时输出类似以下内容：

```
Hello from block: 0, thread: 0
Hello from block: 0, thread: 1
Hello from block: 1, thread: 0
Hello from block: 1, thread: 1
```

（以上各行的顺序可能不同；顺序差异并不代表结果错误。）

请注意核函数启动后使用了 `cudaDeviceSynchronize()`。在 CUDA 中，相对于主机线程而言，核函数启动是*异步*的。主机线程启动核函数后，不会等待其执行完毕，便会继续执行下一行主机代码。因此，为了防止应用程序在核函数来得及输出消息之前就终止，我们必须使用此同步函数。

编辑代码后，使用以下命令进行编译：

```
module load cuda
nvcc -o hello hello.cu
```

`module load` 命令用于选择要使用的 CUDA 编译器。每次会话或登录只需执行一次该命令。`nvcc` 是调用 CUDA 编译器的命令，其语法通常与 gcc/g++ 类似。

如果遇到困难，可以查看 `hello_solution.cu` 中的完整示例。

在 OLCF 的 Summit 上运行代码时，我们将使用以下 LSF 命令：

```
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 ./hello
```

你也可以为 `bsub` 命令创建别名，以便后续运行：

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun ./hello
```

在 NERSC 的 Cori 上，可以使用 Slurm 运行代码：

```
module load esslurm
srun -C gpu -N 1 -n 1 -t 10 -A m3502 --gres=gpu:1 -c 10 ./hello
```

`m3502` 是专为 Cori 上的本培训系列设置的资源配额，提前注册的参与者在 2020 年 1 月 18 日之前应当都可使用。如果无法使用此配额提交作业，但你已经拥有其他可访问 Cori GPU 节点的配额，也可以改用该配额。

如果愿意，也可以在交互式会话中预留一块 GPU，并在 Slurm 资源分配有效期间多次运行可执行文件：

```
salloc -C gpu -N 1 -t 60 -A m3502 --gres=gpu:1 -c 10
srun -n 1 ./hello
```

请注意，每次登录会话只需执行一次 `module load esslurm`；该命令使你能够向 Cori GPU 节点提交作业。

## **2. 向量加法**

如果你想挑战一下自己，可以尝试从头编写一个完整的向量加法程序。你也可以使用 `vector_add.cu` 中提供的代码框架。编辑代码，构建完整的 vector_add 程序，并采用与练习 1 类似的方式编译和运行。完整示例可参考 `vector_add_solution.cu`。

请注意，此代码框架包含了第 1 课中尚未介绍的内容：CUDA 错误检查。每个 CUDA 运行时 API 调用都会返回错误码。严格检查这些错误码是一种良好实践，尤其是在排查问题时。代码中提供了一个宏来简化这项工作。请特别注意核函数调用后的错误检查方式。

完成后的典型输出如下：

```
A[0] = 0.840188
B[0] = 0.394383
C[0] = 1.234571
```

## **3. 矩阵乘法（朴素实现）**

`matrix_mul.cu` 中提供了朴素矩阵乘法的代码框架。请尝试补全代码并得到正确结果。如需帮助，可以参考 `matrix_mul_solution.cu`。

本示例引入了二维线程块和网格索引，这是第 1 课中未涉及的内容。仔细研究代码后，你应该能够看出它在结构上是对一维情况的扩展。

该代码内置了错误检查，程序会指示结果是否正确。
