## **1. 研究复制与计算重叠**

第一个任务提供了一段代码，它对向量中的每个元素执行一个简单计算。你可以先编译、运行并分析原始代码。

使用以下命令编译：

```
module load cuda
nvcc -o overlap overlap.cu
```

`module load` 命令用于选择 CUDA 编译器。每次会话或登录只需执行一次。*nvcc* 是调用 CUDA 编译器的命令，其语法通常与 gcc/g++ 类似。

使用以下 LSF 命令运行代码：

```
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 ./overlap
```

也可以为 bsub 命令创建别名，以便后续运行：

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun ./overlap
```

在 NERSC 的 Cori 上，可以使用 Slurm：

```
module load esslurm
srun -C gpu -N 1 -n 1 -t 10 -A m3502 -G 1 -c 10 ./overlap
```

`m3502` 是专为 Cori 上的本培训系列设置的资源配额，提前注册的参与者应当可以使用。如果无法使用此配额提交作业，但你已经拥有其他可访问 Cori GPU 节点的配额（例如 m1759），也可以改用该配额。

如果愿意，也可以在交互式会话中预留一块 GPU，并在 Slurm 资源分配有效期间多次运行可执行文件（如果有足够的可用节点，推荐采用这种方式）：

```
salloc -C gpu -N 1 -t 60 -A m3502 -G 1 -c 10
srun -n 1 ./overlap
```

每次登录会话只需执行一次 `module load esslurm`；该命令使你能够向 Cori GPU 节点提交作业。

在这种情况下，输出会显示代码非重叠版本的耗时。该版本先将整个向量复制到设备，再启动处理核函数，最后将整个向量复制回主机。

也可以使用 Nsight Systems 运行该代码：

```
module load nsight-systems
lsfrun nsys profile -o <destination_dir>/overlap.qdrep ./overlap
```

请注意，需要将此文件复制到本地计算机，并安装 Nsight Systems 才能进行可视化。可在此处下载 Nsight Systems：
https://developer.nvidia.com/nsight-systems

可视化输出应显示操作序列：*cudaMemcpy*（主机到设备）、核函数调用和 *cudaMemcpy*（设备到主机）。请注意，开头会有一次核函数“预热”运行，可忽略。各操作的开始时间和持续时间应能表明它们之间没有重叠。

你的目标是创建代码的完全重叠版本。利用流的知识，将工作划分成若干分块；对于每个分块，在同一个流中依次执行复制到设备、启动核函数和复制回主机，然后为下一个分块切换所用的流。`#ifdef` 语句后的代码已经提供了起始框架。找到其中的 FIXME 标记，并用适当代码替换，以完成任务。

准备好测试后，使用以下额外开关编译：

```
nvcc -o overlap overlap.cu -DUSE_STREAMS
```

运行代码时会执行验证检查，确保整个向量都已按分块正确处理。如果通过验证，程序将显示流版本的耗时。其速度应至少提升 2 倍，即耗时约为非流版本的一半。也可以使用上述命令通过 Nsight Systems 分析代码。生成的可视化输出应能在放大核函数启动相关部分后确认操作确实发生了重叠。你会先看到非重叠版本运行，随后是重叠版本。重叠版本不仅应更快，还应显示计算与数据传输操作交错执行。

如需帮助，请参考 *overlap_solution.cu*。

## **2. 简单的多 GPU**

本练习提供了一段非常简单的代码，它在单块 GPU 上依次执行 4 次核函数调用。你可以先直接编译并运行代码，程序会显示完成 4 次核函数调用的总耗时。你的任务是修改代码，让每个核函数分别在不同 GPU 上运行（Summit 的每个节点实际上有 6 块 GPU）。完成后，确认执行时间显著减少。

使用以下命令编译：

```
nvcc -o multi multi.cu
```

在 Summit 上使用以下命令运行：

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g4'
lsfrun ./multi
```

在 Cori 上，请确保申请包含 4 块 GPU 的资源，例如：

```
srun -C gpu -N 1 -n 1 -t 10 -A m3502 -G 4 -c 40 ./multi
```

**提示**：本练习可能比想象中简单。完全不需要使用流，只需对每个 for 循环做一个简单修改。

如需帮助，请参考 *multi_solution.cu*。
