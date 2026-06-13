本练习分为 3 个部分，旨在引导你完成一个由 Nsight Compute 驱动、基于分析逐步优化的过程。整个练习聚焦于优化方阵转置。该操作可简单描述为：

Bij = Aji

其中 A 为输入矩阵，B 为输出矩阵，索引 i 和 j 在方阵边长范围内变化。该算法不涉及计算活动，因此属于内存受限算法。最终目标是尽可能接近当前 GPU 的可用内存带宽。

## **1. 使用朴素全局内存的矩阵转置**

第一个任务是进入 *task1* 目录，编辑 *task1.cu* 文件以完成矩阵转置操作。大部分代码已经提供，请使用正确代码替换 **FIXME**，通过全局内存完成矩阵转置。上面的公式可以指导你完成任务。提示如下：

- 每个线程从 (row, col) 读取，并写入 (col, row)
- 使用索引宏：

```cpp
#define INDX( row, col, ld ) ( ( (row) * (ld) ) + (col) ) 
ld = leading dimension (width)
```

如需帮助，可以参考 *task1_solution.cu*。然后编译并测试代码：

```bash
module load cuda
./build_nvcc
```

`module load` 命令用于选择 CUDA 编译器。每次会话或登录只需执行一次。*nvcc* 是调用 CUDA 编译器的命令，其语法通常与 gcc/g++ 类似。

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

程序应显示 PASS 结果，并给出以实际带宽表示的性能测量值。

得到 PASS 结果后，运行性能分析器开始第一轮分析：

```bash
module load nsight-compute
lsfrun nv-nsight-cu-cli --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./task1
```

以下是所请求性能指标的说明：

- *l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum*：全局加载事务数
- *l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum*：全局加载请求数
- *l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio*：每次全局加载请求的事务数
- *smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct*：全局加载效率
- *l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum*：全局存储事务数
- *l1tex__t_requests_pipe_lsu_mem_global_op_st.sum*：全局存储请求数
- *l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio*：每次全局存储请求的事务数
- *smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct*：全局存储效率

利用这些指标，可以轻松观察核函数的各种特征。许多指标不言自明，但全局加载和存储*效率*的计算方式可能并不直观。也可以用每次请求的理论最少事务数除以上述指标计算出的实际事务数，自行计算全局加载和存储效率。

如何确定每次请求的理论最少事务数？缓存行大小为 128 字节，一个线程束有 32 个线程。如果这 32 个线程访问连续的 4 字节字（即单精度浮点数），该请求应产生 4 个事务，也就是请求 DRAM 中连续的四个 32 字节扇区。本例使用双精度浮点数，因此 32 个线程会访问连续的 8 字节字，总计 256 字节。因此，本例每次请求的理论最少事务数为 8，即 DRAM 中连续的八个 32 字节扇区。

根据性能分析器的输出，全局加载效率和全局存储效率是否都达到了 100%？为什么？此时可以仔细研究加载和存储索引，并回顾作业 4 中学到的全局内存合并访问规则。

## **2. 使用共享内存分块转置修复全局内存合并访问问题**

在任务 1 中，我们了解到，朴素全局内存转置算法必然使加载或存储操作之一无法合并，即产生按列的内存访问。要解决这个问题，必须设计一种过程，使全局内存的加载和存储都能合并。因此，我们先将输入矩阵中的数据块移入共享内存，再将该数据块写入输出矩阵，从而完成块内转置。整个过程包括：从全局内存读取、写入共享内存、从共享内存读取、写入全局内存。

在这两个步骤中，需要：

- 执行“块内”转置，即按行读取并按列写入，或反之
- 执行“块位置”转置，即输入矩阵中块索引为 *i,j* 的数据块必须存储到输出矩阵中块索引为 *j,i* 的位置

从 *task1* 目录切换到 *task2*。编辑 *task2.cu* 文件中所有出现 **FIXME** 的位置，实现上述两项操作。如需帮助，请参考 *task2_solution.cu*。这是本练习 3 个任务中最难的编程任务。

与 task1 一样，编译并运行代码：

```bash
./build_nvcc
lsfrun ./task2
```

程序应输出 PASS。测得的带宽是否有所提升？

我们再次使用性能分析器解释观察结果。由于算法中引入了共享内存操作，分析时还要加入共享内存测量指标：

```bash
lsfrun nv-nsight-cu-cli --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct ./task2
```

- *l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum*：共享内存加载事务数
- *l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum*：共享内存存储事务数
- *l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum*：共享内存加载的存储体冲突数
- *l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum*：共享内存存储的存储体冲突数
- *smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct*：共享内存效率

你应能确认，此前的全局加载/全局存储效率问题已通过正确的合并访问得到解决。不过现在共享内存出现了存储体冲突。请回顾模块 4 中关于存储体冲突的内容，了解这种问题在共享内存访问期间如何产生。

## **3. 修复共享内存存储体冲突**

本例修复共享内存存储体冲突的策略非常简单。保持练习 2 中的共享内存索引不变，但在代码的共享内存定义中增加一列。这样即可在无存储体冲突的情况下同时按行和按列访问共享内存，而这正是块内转置步骤所需要的。

切换到 *task3* 目录。

根据需要修改 *task3.cu*。如需帮助，请参考 *task3_solution.cu*。

采用与前两个任务类似的方式编译并运行代码。

程序应通过验证。实际带宽是否有所提升？

可以分析代码，确认加载和存储操作现在都能高效使用共享内存：

```bash
lsfrun nv-nsight-cu-cli --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct  ./task3
```

最后，如果愿意，可以运行 CUDA 示例 bandwidthTest，并以设备到设备的内存带宽数值作为参照，将代码报告的实际带宽与可达到的峰值带宽近似测量值进行比较。
