# CUDA 图

本作业将研究两段使用 CUDA 图的不同代码。这些代码由小型核函数组成，考虑到其特定工作流，可能会从 CUDA 图中获益。两段代码分别是 axpy_stream_capture 和 axpy_cublas，每段都有 with_fixme 和 from_scratch 两种版本。建议先完成两段代码的 with_fixme 版本，若想挑战自己，再尝试 from_scratch 版本。with_fixme 版本中有若干位置需要修复才能运行，但整体框架已经搭好。from_scratch 版本则要求手动实现图的设置与逻辑。

遇到困难时，可以参考 Solutions 目录中的解决方案和提示。

### 任务 1

#### 流捕获

本任务演示如何在 CUDA 图中使用流捕获。我们将根据跨两个流的一系列核函数启动创建一个图。

目标是实现下图所示的结构，图形有助于直观理解：

![](graph_stream_capture.png)

这与幻灯片中的示例相同，可随时参考幻灯片获取帮助和提示。

现在先浏览代码，了解新的图 API 调用。第一次阅读时可以先忽略图 API，尝试理解底层代码及其行为。核函数本身并不执行特定数学运算，只是代表一些任意的小型核函数。请思考两个流的作用，并参照上图，确保理解 CUDA 事件所创建的内在依赖关系。

`bool graphCreated=false;` 用于只在第一次执行（for 循环第 0 次迭代）时设置图；之后每次迭代（1 到 N-1）都直接启动该图。

需要特别区分 `cudaGraph_t` 与 `cudaGraphExec_t`。`cudaGraph_t` 用于定义整个图的形状和参数；`cudaGraphExec_t` 是经过实例化步骤后得到的可调用图实例。

首先，在 Summit 上编译代码：

```
module load cuda/11.4.0
nvcc -arch=sm_70 axpy_stream_capture_with_fixme.cu -o axpy_stream_capture_with_fixme
```

这里为正在使用的 GPU 架构进行编译，本例为 Volta SM 7.0。CUDA 10 之后的所有 CUDA 工具包都包含 CUDA 图，但某些功能可能依赖具体版本。

使用以下 LSF 命令运行代码：

```
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 ./axpy_stream_capture_with_fixme
```

也可以为 bsub 命令创建别名，以便后续运行：

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun ./axpy_stream_capture_with_fixme
```

在 NERSC 的 Cori-GPU 上构建代码：

```
module load cgpu cuda/11.4.0
nvcc -arch=sm_70 axpy_stream_capture_with_fixme.cu -o axpy_stream_capture_with_fixme
```

在 10 月 13 日太平洋时间 10:30-12:30 的节点预留时段运行：

```
module load cgpu cuda/11.4.0
srun -C gpu -N 1 -n 1 -t 10 -A ntrain2 --reservation=cuda_graphs -q shared -G 1 -c 1 ./axpy_stream_capture_with_fixme
```

或者先获取一个 GPU 节点，再进行交互式运行：

```
module load cgpu cuda 
salloc -C gpu -N 1 -t 60 -A ntrain2 --reservation=cuda_graphs -q shared -G 1 -c 1
srun -n 1 ./axpy_stream_capture_with_fixme
```

在节点预留时段之外运行时，步骤相同，但不要在 srun 或 salloc 命令中包含 *--reservation=cuda_graphs -q shared*。

FIXME：

1. cudaGraphCreate(FIXME, 0);
2. cudaGraphInstantiate(FIXME, graph, NULL, NULL, 0);
3. graphCreated = FIXME;
4. cudaGraphLaunch(FIXME, streams[0]);

补全 FIXME 后，运行程序会打印一个时间值，即图运行 1000 次的总耗时。可以将其与 axpy_stream_capture_timer.cu 的时间进行比较，后者运行相同的 CUDA 工作，但使用流而不是图。这些示例主要用于介绍主题和 API，因此性能并未经过特别优化。即便如此，通过节省启动开销，使用图仍应带来小幅性能提升。不过，计时不包含实例化阶段，因此两者并非完全对等的比较。该实验只是用于突出幻灯片中介绍的概念。

### 任务 2

#### 显式创建包含库调用的图

本任务将研究一些显式图创建 API，以及如何通过流捕获来捕获库调用。本例的关键在于：虽然同时使用了显式图创建和流捕获，但二者都只是定义 `cudaGraph_t` 的方式，随后再将其实例化为 `cudaGraphExec_t`。

我们将创建 2 个核函数节点，以及一个由 cuBLAS axpy 函数调用生成的子图。可参考下图：

![](graph_with_library_call.png)

https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html

以上链接是当前 CUDA 工具包图管理 API 的文档。不查阅文档，仅依靠幻灯片和代码上下文线索也可以完成示例；如果在 FIXME 处遇到困难，查看 `cudaGraphAddChildGraphNode` 的定义会有所帮助。

与第一个示例不同，如果忽略图 API，本代码将更难以理解。事实上，没有图 API 调用，程序根本无法运行。如此程度的代码改动带来了更强的控制能力，但代价是修改量更大，并且对不熟悉 CUDA 图的用户而言可读性更差。

该 API 初看之下有些棘手，因为它与 CUDA 中的其他内容很不一样，但其模式实际上相当熟悉，只是定义 CUDA 工作的另一种方式。

使用与此前相同的编译说明，但这次增加 `-lcublas` 以链接该库：

```
nvcc -arch=sm_70 -lcublas axpy_cublas_with_fixme.cu -o axpy_cublas_with_fixme
```

使用之前为 Summit 创建的别名运行：

```
lsfrun ./axpy_cublas_with_fixme
```

或者在 Cori GPU 的交互式节点上运行：

```
srun -n 1 ./axpy_stream_capture_with_fixme
```

查看 axpy_cublas_with_fixme.cu，尝试补全 FIXME，使代码能够编译并运行。请参考示意图理解程序流程。

FIXME：

1. cudaGraphCreate(FIXME, 0);
2. cudaGraphAddChildGraphNode(FIXME, graph, FIXME, nodeDependencies.size(), libraryGraph);
3. cudaGraphLaunch(FIXME, stream1);
