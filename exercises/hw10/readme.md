## **1. 流回顾**

第一个任务提供了一段对向量中每个元素执行简单计算的代码。我们已经在作业 7 中使用多个 CUDA 流实现了该代码的分块版本。首先回顾 CUDA 流对其性能的影响。

使用以下命令编译：

```
module load cuda/11.4.0
nvcc -o streams streams.cu -DUSE_STREAMS
```

`module load` 命令用于选择 CUDA 编译器。每次会话或登录只需执行一次。*nvcc* 是调用 CUDA 编译器的命令，其语法通常与 gcc/g++ 类似。

使用以下 LSF 命令运行代码：

```
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 ./streams
```

也可以为 bsub 命令创建别名，以便后续运行：

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun ./streams
```

在 NERSC 的 Cori-GPU 上构建代码：

```
module load cgpu cuda/11.4.0
nvcc -o streams streams.cu -DUSE_STREAMS
```

在 7 月 16 日太平洋时间 10:30-12:30 的节点预留时段运行：

```
module load cgpu cuda/11.4.0
srun -C gpu -N 1 -n 1 -t 10 -A ntrain --reservation=cuda_training -q shared -G 1 -c 8 ./streams
```

或者先获取一个 GPU 节点，再进行交互式运行：

```
module load cgpu cuda 
salloc -C gpu -N 1 -t 60 -A ntrain --reservation=cuda_training -q shared -G 1 -c 8
srun -n 1 ./streams
```

在节点预留时段之外运行时，步骤相同，但不要在 srun 或 salloc 命令中包含 `--reservation=cuda_training -q shared`。

输出会比较代码非重叠版本与重叠版本的耗时。非重叠版本先将整个向量复制到设备，再启动处理核函数，最后将整个向量复制回主机。重叠版本将向量拆分成多个分块，并使用 CUDA 流在 GPU 上异步复制和处理每个分块。

如果想观察重叠行为，也可以使用 Nsight Systems 运行代码。

在 Summit 上：

```
module load nsight-systems
lsfrun nsys profile -o <destination_dir>/streams.qdrep ./streams
```

在 Cori 上：

```
module load nsight-systems
srun -n 1 nsys profile -o <destination_dir>/streams.qdrep ./streams
```

请注意，需要将此文件复制到本地计算机，并安装 Nsight Systems 才能进行可视化。可在此处下载 Nsight Systems：
https://developer.nvidia.com/nsight-systems

可视化输出应显示操作序列：*cudaMemcpy*（主机到设备）、核函数调用和 *cudaMemcpy*（设备到主机）。
运行代码时会执行验证检查，确保整个向量都已按分块正确处理。如果通过验证，程序将显示流版本的耗时。重叠版本的速度应约为非流版本的 2 倍，即耗时约为其一半。如果使用 Nsight Systems 分析代码，放大核函数启动相关部分后，应能确认操作确实发生了重叠。你会先看到非重叠版本运行，随后是重叠版本。重叠版本不仅应更快，还应显示计算与数据传输操作交错执行。

## **2. OpenMP + CUDA 流**

对于本应用程序，从单个 CPU 线程异步启动核函数已经足够。不过，对那些使用 OpenMP 在节点内进行共享内存处理的传统 HPC 应用而言，情况未必如此。许多此类应用使用 MPI 在节点之间分配工作，并使用 OpenMP 改善节点内共享内存处理。然而，每个 OpenMP 线程可能仍有相当一部分工作可从 GPU 加速中受益，尽管单个线程的工作量不足以让 GPU 饱和。在这种情况下，可以将 OpenMP 线程与 CUDA 流结合，确保 GPU 得到充分利用。

为了模拟这种行为，你的任务是将代码中各向量分块的处理分配给 OpenMP 线程。如果实现正确，每个线程都会使用代码中已有的 CUDA 流分解方式，异步向 GPU 提交工作。请注意，这不会影响本示例代码的性能。目标是展示如何结合 CPU 线程并行与 CUDA 流，在一块或多块 GPU 上实现并发执行。

插入 OpenMP 语句后，按照以下说明编译并运行。

在 Summit 上：

```
nvcc -Xcompiler -fopenmp -o streams streams.cu -DUSE_STREAMS
export OMP_NUM_THREADS=8
jsrun -n1 -a1 -c8 -bpacked:8 -g1 ./streams
```

在 Cori 上：

```
nvcc -Xcompiler -fopenmp -o streams streams.cu -DUSE_STREAMS
export OMP_NUM_THREADS=8
srun -C gpu -N 1 -n 1 -t 10 -A ntrain --reservation=cuda_training -q shared -G 1 -c 8 ./streams
```

与练习 1 相比，性能如何？结果应非常接近。分析代码时又如何？遗憾的是，性能分析器目前在跨 CPU 线程分析时需要进行一定程度的串行化，因此实际性能应比非重叠版本更慢，生成的 qdrep 文件也应反映这一点。可以注意到，GPU 上不再出现那么多并发执行。我们正在改进这一问题，未来版本的性能分析器将不再受到此限制。

如需帮助，请参考 *streams_solution.cu*。

## **3. 附加任务：多 GPU**

请记住，一个 CUDA 流绑定到特定 GPU。如何将 CPU 多线程与多块 GPU 结合？如果想挑战一下，可以尝试修改本作业代码，使其向 4 块 GPU 而非一块 GPU 提交工作。创建流时需要记录每个 CUDA 流绑定到哪块 GPU。可以增大问题规模，确保有足够工作量体现性能影响。使用以下说明编译并运行代码。

在 Summit 上：

```
nvcc -Xcompiler -fopenmp -o streams streams.cu -DUSE_STREAMS
export OMP_NUM_THREADS=8
jsrun -n1 -a1 -c8 -bpacked:8 -g4 ./streams
```

在 Cori 上：

```
nvcc -Xcompiler -fopenmp -o streams streams.cu -DUSE_STREAMS
export OMP_NUM_THREADS=8
srun -C gpu -N 1 -n 1 -t 10 -A ntrain --reservation=cuda_training -q shared -G 4 -c 8 ./streams
```
