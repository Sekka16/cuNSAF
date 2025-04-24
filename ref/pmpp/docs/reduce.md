# reduce

## V0 SimpleSumReductionKernel

求`2 * blockDim.x`个数的和，使用一个block

`stride`从1开始递增到`blockDim.x`，线程处理的是相邻的元素。

在第一轮迭代中，所有的线程都参与两个数的计算。但是在第二轮中，线程0继续计算而线程1不再计算。他们同属于一个warp中，就产生了控制分歧。一个warp对应一个simd运算单元，遇到控制分歧之后会进行两次执行，一次执行if，另一个执行else，只不过只保留分支对应的结果。这样造成了资源的浪费。

第二点是内存的分歧，GPU的内存访问是基于**内存事务**的，如果访问的地址不连续，GPU无法将这个访问请求合并成一个内存事务。此外，内存的访问读取的一定字节对齐的段，只使用其中一些字节仍然会整段载入。

## V1 ConvergentSumReductionKernel

求`2 * blockDim.x`个数的和，使用一个block

`stride`从`blockDim.x`递减到1，线程处理的隔了`stride`的步长的两个元素。

在每迭代中，前`stride`个线程工作，活跃的线程集中在一起，这样有效减少了warp中的控制分歧。

## V2 SharedMemorySumReducetionKernel

求`2 * blockDim.x`个数的和，使用一个block

首先将相隔了`blockDim.x`的两个元素相加，然后存放在共享内存中。

之后处理共享内存中的共`blockDim.x`个元素。

与V1使用的总线程数是一样的，都是用了`blockDim.x`个线程，胜在使用了共享内存，访问更快。

## V3 SegmentedSumReductionKernel

求任意输入长度的层次化规约

每个block还是处理`2 * blockDim.x`个元素，但这是block的数量不再是一个。

需要计算当前块中的线程处理元素在内存中位置，通过`segment = 2 * blockDim.x * blockIdx.x`变量。

最后使用`atomicAdd()`来将所有block的结果加到一起。如果block的数量较多需要多级规约。

## V4 CoarsenedSumReductionKernel

在之前的实现了，每个block都只会处理`2 * blockDim.x`个元素，也就是每个线程处理两个元素。

线程粗化是通过让每个线程做更多工作来**减少调度/内存/指令的开销**，从而提升性能。但需要根据实际硬件资源和算法特点**谨慎调整粒度**，避免过粗导致其他瓶颈。
