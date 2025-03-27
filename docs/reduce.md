# reduce算子优化

## baseline

将数据从global memory读入shared memory，然后相邻元素两两相加。

```cpp
template<const int THREAD_PER_BLOCK = 256>
__global__ void reduce_baseline_kernel(const float *input, float *output, int n) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 将数据加载到共享内存，并检查数组边界
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // 归约操作：每轮步长翻倍
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0 && (tid + s) < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 每个 Block 的归约结果保存在 Block 内第一个线程处
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

