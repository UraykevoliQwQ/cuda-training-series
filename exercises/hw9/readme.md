# **1. 探索线程块级组**

## **1a. 创建组**

首先，使用 *task1.cu* 代码补全标有 **FIXME** 的部分，创建正确的线程块组，并将其赋给用于输出的组。在第一步中，只需修改包含 **FIXME** 的两行。

使用以下命令编译：

```bash
module load cuda
nvcc -arch=sm_70 -o task1 task1.cu -std=c++11
```

`module load` 命令用于选择 CUDA 编译器。每次会话或登录只需执行一次。*nvcc* 是调用 CUDA 编译器的命令，其语法通常与 gcc/g++ 类似。请注意，由于使用协作组需要 C++11，因此需要足够现代的编译器（gcc >= 5 即可）。如果使用 Summit，请务必执行 `module load gcc`，因为系统默认 gcc 版本不够新。

使用以下 LSF 命令运行代码：

```bash
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 ./task1
```

也可以为 bsub 命令创建别名，以便后续运行：

```bash
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun ./task1
```

在 NERSC 的 Cori 上，可以使用 Slurm：

```bash
module load esslurm
srun -C gpu -N 1 -n 1 -t 10 -A m3502 --gres=gpu:1 -c 10 ./task1
```

`m3502` 是专为 Cori 上的本培训系列设置的资源配额，提前注册的参与者应当可以使用。如果无法使用此配额提交作业，但你已经拥有其他可访问 Cori GPU 节点的配额（例如 m1759），也可以改用该配额。

如果愿意，也可以在交互式会话中预留一块 GPU，并在 Slurm 资源分配有效期间多次运行可执行文件（如果有足够的可用节点，推荐采用这种方式）：

```bash
salloc -C gpu -N 1 -t 60 -A m3502 --gres=gpu:1 -c 10
srun -n 1 ./task1
```

每次登录会话只需执行一次 `module load esslurm`；该命令使你能够向 Cori GPU 节点提交作业。

正确输出应类似：

```bash
group partial sum: 256
```

如需帮助，请参考 *task1_solution1.cu*，其中包含任务 1a、1b 和 1c 的解决方案。

## **1b. 划分组**

接下来，取消注释下一行以 auto 关键字开头的代码，并补全该行。使用此前创建的线程块组，通过动态（运行时）划分方法将其细分为若干个包含 32 个线程的分区。

按上述方式编译并运行代码。正确输出应类似：

```bash
group partial sum: 32
group partial sum: 32
group partial sum: 32
group partial sum: 32
group partial sum: 32
group partial sum: 32
group partial sum: 32
group partial sum: 32
```

## **1c. 第三次组创建/分解**

现在执行第三次组创建/分解。

按上述方式编译并运行代码。正确输出应类似：

```bash
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
```

# **2. 探索网格级同步**

采用网格级同步的一个动机，是合并那些必须按顺序完成、通常需要通过多次独立 CUDA 核函数调用实现的算法阶段。在这种情况下，核函数启动边界提供了隐式或等效的网格级同步。然而，协作组允许直接在核函数代码中执行网格级同步，而不必依赖核函数启动边界。

流压缩就是此类算法之一。流压缩应用广泛，其基本目标是根据特定的删除规则或谓词测试来缩短数据流。例如，若有以下数据流：

```bash
3 4 3 7 0 5 0 8 0 0 0 4
```

删除其中的零即可完成流压缩，得到：

```bash
3 4 3 7 5 8 4
```

与许多归约类算法一样（此处输出可能比输入小得多），很容易想象串行实现方式，但快速的并行流压缩需要进一步思考。一种常见方法是使用前缀和。前缀和数据集中的每个元素，表示从输入开头到当前位置之前所有输入元素的和。我们可以使用前缀和辅助并行化流压缩。首先创建一个由 1 和 0 组成的数组：要保留的元素对应 1，要丢弃的元素对应 0：

```bash
3 4 3 7 0 5 0 8 0 0 0 4 (input data)
1 1 1 1 0 1 0 1 0 0 0 1 (filtering of input)
```

然后对过滤数组执行排他前缀和。排他意味着求和只包含当前位置“左侧”的元素，不包含当前位置本身。

```bash
3 4 3 7 0 5 0 8 0 0 0 4 (input data)
1 1 1 1 0 1 0 1 0 0 0 1 (filtering of input)
0 1 2 3 4 4 5 5 6 6 6 6 (exclusive prefix sum of filtered data)
```

此前缀和现在包含每个输入位置应复制到输出数组中的索引。只有对应过滤元素非零时，才会将输入位置复制到输出。以上说明了如何用前缀和辅助流压缩，但尚未说明如何高效并行计算前缀和。完整介绍超出了本文档范围，可参考：https://people.eecs.berkeley.edu/~driscoll/cs267/papers/gpugems3_ch39.html。需要注意的是，前缀和也包含扫描操作，与并行归约中连续执行的扫描操作类似，但二者存在关键差异。其中两项是：前缀和中的扫描从“左”到“右”，典型并行归约通常从右到左；各扫描阶段参与线程的分界点也不同。

并行计算前缀和通常需要多个阶段，例如先执行线程块级扫描（前缀和），再根据其他“先前”线程块的数据修正线程块级结果。这些阶段可能需要网格级同步，而 thrust 等库中的典型扫描会使用多次核函数调用。下面尝试通过一次核函数调用完成它。除插入适当的协作组同步点外，不需要编写任何扫描代码。我们既需要线程块级同步点（基于已创建的线程块级组），也需要网格级同步点。

从 *task2.cu* 代码开始，完成两项工作：

- 修改核函数中的 **FIXME** 语句，按要求基于核函数顶部创建的两种组插入适当的同步操作。只需要一个网格级同步点，其余均为线程块级同步点。
- 修改主机代码中的 **FIXME** 语句，执行正确的协作式启动。启动函数已经提供，只需填写剩余 4 个参数。可参考 *task2_solution.cu*，或查看该启动函数的 CUDA 运行时 API 文档：https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g504b94170f83285c71031be6d5d15f73

完成修改后，使用以下命令编译：

```bash
nvcc -arch=sm_70 -o task2 task2.cu -rdc=true -std=c++11
```

并按如下方式运行：

```bash
lsfrun ./task2
```

正确输出应仅包含：

```bash
number of SMs = 80
number of blocks per SM = 8
kernel time: 0.043872ms
thrust time: 0.083200ms
```

（以上数值仅供参考。如果使用的 GPU 不是 Tesla V100，数据可能不同，但仍应只看到以上 4 行。）

上述信息包含由占用率 API 生成的“占用率”数据。可以看到，单个 SM 能够容纳 8 个各含 256 个线程的线程块，达到每个 SM 理论上限 2048 个线程，即 100% 占用率。该核函数相当简单，资源使用量和需求都很低。

代码内置了静默验证，因此除上述信息外不会打印实际结果。如果出现 *mismatch* 消息，则实现存在问题。

可选任务：

task2 代码会将该操作与 thrust 中的等效操作进行比较。对于这个非常小的数据集，单体核函数看起来比 thrust 更快。使用 nsight-compute 分析该小数据集，确认 thrust 实际上通过 2 次核函数调用解决此问题：

```bash
module load nsight-compute
lsfrun nv-nsight-cu-cli ./task2
```

现在增大数据集，合理的上限可以是 32M 个元素。请确保选择能被线程块大小 256 整除的数。例如，将：

```cpp
const int test_dsize = 256;
```

改为：

```cpp
const int test_dsize = 1048576*16;
```

然后重新编译并运行代码。现在 thrust 和朴素代码哪个更快？

结论：如果能找到高质量的库实现，就不要自行编写代码。对于排序、前缀和及矩阵乘法等更复杂的算法，这一点尤其重要。
