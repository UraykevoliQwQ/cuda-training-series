# **任务 1**

本任务将探索 compute-sanitizer 的使用。CUDA 编程指南中提供了一个完整的分块矩阵乘法示例。*task1.cu* 包含经过少量修改的该示例，以及用于驱动操作的 main() 函数。假设你正在为集群用户群体提供支持服务，其中一位用户提交了这段代码并报告：“CUDA 错误检查没有显示任何错误，但我没有得到正确答案。请帮忙！”

首先，使用以下命令编译并运行代码，观察所报告的行为：

```
module load cuda
nvcc -arch=sm_70 task1.cu -o task1 -lineinfo
```

这里为正在使用的 GPU 架构进行编译，本例为 Volta SM 7.0，同时还使用了 `--lineinfo` 开关。作为 CUDA 支持工程师，你知道该开关在使用 compute-sanitizer 时会很有用。

使用以下 LSF 命令运行代码：

```
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 ./task1
```

也可以为 bsub 命令创建别名，以便后续运行：

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun ./task1
```

在 NERSC 的 Cori-GPU 上构建代码：

```
module load cgpu cuda/11.4.0
nvcc -arch=sm_70 task1.cu -o task1 -lineinfo
```

在 9 月 14 日太平洋时间 10:30-12:30 的节点预留时段运行：

```
module load cgpu cuda/11.4.0
srun -C gpu -N 1 -n 1 -t 10 -A ntrain --reservation=cuda_debug -q shared -G 1 -c 1 ./task1
```

或者先获取一个 GPU 节点，再进行交互式运行：

```
module load cgpu cuda
salloc -C gpu -N 1 -t 60 -A ntrain --reservation=cuda_debug -q shared -G 1 -c 1
srun -n 1 ./task1
```

在节点预留时段之外运行时，步骤相同，但不要在 srun 或 salloc 命令中包含 *--reservation=cuda_debug -q shared*。

如果代码生成了正确的矩阵结果，将显示：

```
Success!
```

但遗憾的是，现在看不到该输出。

## A 部分

使用 *compute-sanitizer* 的基本功能（不添加额外开关）找出代码中的问题。根据 *compute-sanitizer* 输出，确定导致问题的代码行并修复它。

提示：

- 请记住，*-lineinfo* 会使 compute-sanitizer 在此用法中报告实际导致问题的代码行
- 即使没有行号信息，能否使用 compute-sanitizer 的其他信息快速推断本例中应重点检查的代码行？可以将内存访问违规的类型作为线索。核函数中哪些代码行正在执行这种类型的内存访问？提示：只有一行核函数代码执行这种访问
- 内存访问问题通常由索引错误引起。尝试找出可能导致该问题的索引错误，提示：经典的计算机科学“差一错误”
- 如果遇到困难，请参考 *task1_solution.cu*

## B 部分

很好，你已经解决问题、修改了索引，代码现在可以打印 `Success!`。是时候让用户继续工作了吗？也许还不行，是否还存在其他错误？使用其他 compute-sanitizer 开关（*--tool racecheck*、*--tool initcheck*、*--tool synccheck*）找出其他“潜在”问题并修复它们。

提示：

- 此时只有 racecheck 工具应报告问题
- 尝试使用错误报告中包含的行号信息，确定核函数代码中的问题区域
- racecheck 工具只报告共享内存使用中的竞争问题，而此类问题通常与缺少同步有关。你能否确定应在核函数代码中的哪个位置插入适当的同步？可以通过实验进行判断。在 CUDA 核函数中插入额外同步通常不会破坏代码正确性
- 如果遇到困难，请参考 *task1_solution.cu*

# **任务 2**

本任务将探索 cuda-gdb 的基本用法。再次假设你正在集群服务台提供用户支持。用户有一段代码产生了意料之外的 *-inf*（负浮点无穷大）结果。代码由一个变换操作（每个线程创建或修改一个数据元素）和后续的归约操作（将每个线程的结果相加）组成，归约输出为 *-inf*。请尝试使用 *cuda-gdb* 找出并修复问题。

为了使用 *cuda-gdb*，需要编译调试版本。因此使用以下命令编译代码：

```
nvcc -arch=sm_70 task2.cu -o task2 -G -g -std=c++14
```

然后即可开始调试。

在 Summit 上：

```
jsrun -n1 -a1 -c1 -g1 cuda-gdb ./task2
```

在 Cori 上：

```
srun -n 1 ./task2
```

请勿忘记：只有在设备代码断点之后暂停时，才能检查设备数据。

确定问题来源后，尝试提出一个简单的代码修改方案来规避问题。如果在提出解决方案时遇到困难，请参考 *task2_solution.cu*。仔细检查代码可能会立刻发现问题，但本任务的真正目的并不是以这种方式修复代码，而是学习使用 *cuda-gdb*。

提示：

- 代码尝试估算交错调和级数（ahs）的和，其结果应等于 2 的自然对数
- 代码分为两部分：由设备函数 ahs 实现的 ahs 项生成器（仅接受要生成项的索引），以及标准的扫描式并行归约，与本培训系列第 5 课的内容类似
- 一般而言，对 *inf* 或 *-inf* 输入进行浮点运算，会产生 *inf* 或 *-inf* 输出
- 判断 *-inf* 更可能由最初的变换操作产生，还是由后续归约操作产生
- 根据此判断选择初始断点位置
- 检查数据，查看能否在中间数据中观察到 *-inf*
- 根据观察结果，重复设置断点和检查数据的过程
- 也可以按线性方式逐步执行代码，先设置初始断点，再单步执行，观察错误数据何时出现
- 可能需要切换当前线程或观察其他线程的数据
- 归约过程也为通过分治或二分查找方式定位问题提供了机会
- 考虑减小问题规模，即减少用于生成估算值的项数，以简化调试工作
