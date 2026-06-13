# 作业 6

这些练习将引导你使用统一内存（Unified Memory），在非简单数据结构上利用 GPU。

## **1. 将链表移植到 GPU**

第一个任务提供了一段代码，它在 CPU 上构建链表，然后尝试打印链表中的一个元素。你的任务是使用统一内存技术修改代码，使 CPU 代码或 GPU 代码都能正确遍历链表。提示：本练习只需修改文件中的一行。

使用以下命令进行编译：

```
module load cuda
nvcc -o linked_list linked_list.cu
```

`module load` 命令用于选择 CUDA 编译器。每次会话或登录只需执行一次。*nvcc* 是调用 CUDA 编译器的命令，其语法通常与 gcc/g++ 类似。

使用以下 LSF 命令运行代码：

```
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 ./linked_list
```

也可以为 bsub 命令创建别名，以便后续运行：

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun ./linked_list
```

在 NERSC 的 Cori 上，可以使用 Slurm：

```
module load esslurm
srun -C gpu -N 1 -n 1 -t 10 -A m3502 --gres=gpu:1 -c 10 ./linked_list
```

`m3502` 是专为 Cori 上的本培训系列设置的资源配额，提前注册的参与者应当可以使用。如果无法使用此配额提交作业，但你已经拥有其他可访问 Cori GPU 节点的配额（例如 m1759），也可以改用该配额。

如果愿意，也可以在交互式会话中预留一块 GPU，并在 Slurm 资源分配有效期间多次运行可执行文件（如果有足够的可用节点，推荐采用这种方式）：

```
salloc -C gpu -N 1 -t 60 -A m3502 --gres=gpu:1 -c 10
srun -n 1 ./linked_list
```

每次登录会话只需执行一次 `module load esslurm`；该命令使你能够向 Cori GPU 节点提交作业。

正确输出应类似：

```
key = 3
key = 3
```

如需帮助，请参考 *linked_list_solution.cu*。

## **2. 数组递增**

本练习提供了一段在 GPU 上递增大型数组中各元素的代码。

a. 首先，直接编译代码并进行性能分析：

```
module load nsight-systems
nvcc -o array_inc array_inc.cu
lsfrun nsys profile --stats=true ./array_inc
```

记录核函数执行时间。

b. 现在修改代码以使用托管内存。将 malloc 操作替换为 cudaMallocManaged，并删除 cudaMemcpy 操作。是否需要用 *cudaDeviceSynchronize()* 替换从设备到主机的 *cudaMemcpy* 操作？为什么？然后再次编译并分析代码。将核函数执行时间与此前结果比较，并注意性能分析器所显示的 CPU 和 GPU 缺页情况。

c. 现在修改代码，在核函数调用前立即将数组预取到 GPU，并在核函数调用后立即预取回 CPU。再次编译并分析代码。将核函数执行时间与此前结果比较。是否仍存在缺页？为什么？

d. 附加任务：修改代码，连续运行 *inc()* 核函数 10000 次，而不是只运行一次。内存操作对运行时间有何影响？这对实际应用程序有何启示？

如需帮助，请参考 *array_inc_solution.cu*。
