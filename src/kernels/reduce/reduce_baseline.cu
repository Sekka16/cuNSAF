#include "core/utils.h"
#include "kernels/kernels.h"

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

__global__ void reduce_baseline(const float *input, float *output, int n) {
  __shared__ float sdata[THREAD_PER_BLOCK];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (idx < n) ? input[idx] : 0.0f;
  __syncthreads();

  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0 && (tid + s) < blockDim.x) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

__global__ void reduce_without_warp_divergence(const float *input, float *output, int n) {
  __shared__ float sdata[THREAD_PER_BLOCK];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (idx < n) ? input[idx] : 0.0f;
  __syncthreads();

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int targetIdx = 2 * stride * tid; // 本轮中计算得到数据存储到sram中的位置
    if (targetIdx < blockDim.x) {
      sdata[targetIdx] += sdata[targetIdx + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

__global__ void reduce_without_bank_conflict(const float *input, float *output, int n) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// 显著变化是原本每个block中的256个线程，后128个不工作，现在有加载两个数据并且做加法的工作
// 并且由于每个block的空闲线程得到了利用，gridSize也大大减小
__global__ void reduce_with_idle_used(const float *input, float *output, int n) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float x = (idx < n) ? input[idx] : 0.0f;
    float y = (idx + blockDim.x < n) ? input[idx + blockDim.x] : 0.0f;
    sdata[tid] = x + y;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
	        sdata[tid] += sdata[tid + stride];
	    }
	    __syncthreads();
    }

    if (tid == 0) {
       output[blockIdx.x] = sdata[0];
    }
}

__device__ void warpReduce(volatile float* cache, int tid) {
    cache[tid] += cache[tid + 32];
    cache[tid] += cache[tid + 16];
    cache[tid] += cache[tid + 8];
    cache[tid] += cache[tid + 4];
    cache[tid] += cache[tid + 2];
    cache[tid] += cache[tid + 1];
}

__global__ void reduce_with_expand_last_dim(const float *input, float *output, int n) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float x = (idx < n) ? input[idx] : 0.0f;
    float y = (idx + blockDim.x < n) ? input[idx + blockDim.x] : 0.0f;
    sdata[tid] = x + y;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid < 32) {
        warpReduce(sdata, tid);
    }
    if (tid == 0) {
        output[blockIdx.x] = sdata[tid];
    }
}

template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (blockSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
    if (blockSize >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (blockSize >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);
    if (blockSize >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (blockSize >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

template <unsigned int blockSize>
__global__ void reduce_with_shuffle(const float *input, float *output, int n) {
    float sum = 0.0f;

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float x = (idx < n) ? input[idx] : 0.0f;
    float y = (idx + blockDim.x < n) ? input[idx + blockDim.x] : 0.0f;
    sum = x + y;
    __syncthreads();

    static __shared__ float warpLevelSums[WARP_SIZE];
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    sum = warpReduceSum<blockSize>(sum);

    if (laneId == 0) {
        warpLevelSums[warpId] = sum;
    }
    __syncthreads();

    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    
    if (warpId == 0) {
        sum = warpReduceSum<blockSize/WARP_SIZE>(sum);
    }

    if (tid == 0) {
        output[blockIdx.x] = sum;
    }
}

template __global__ void reduce_with_shuffle<256>(float*, float*, unsigned int);

void reduce(ReduceKernel kernel, const float *input, float *output, int n) {
    // Step 1: 分配设备内存并拷贝输入数据
    float *d_input = nullptr, *d_current = nullptr, *d_next = nullptr;

    int blockSize = 256;  // 固定线程块大小
    int gridSize = (n + blockSize - 1) / blockSize;
    if (kernel == reduce_with_idle_used ||
        kernel == reduce_with_expand_last_dim ||
        kernel == reduce_with_shuffle<256>) {
        gridSize = (n + blockSize * 2 - 1) / (blockSize * 2);
    }
    const int output_size = gridSize * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_current, output_size));
    CHECK_CUDA(cudaMalloc(&d_next, output_size));  // 分配临时内存用于多级归约

    CHECK_CUDA(cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice));

    // Step 2: 第一次归约
    kernel<<<gridSize, blockSize>>>(d_input, d_current, n);  // 调用传入的核函数
    CHECK_CUDA(cudaGetLastError());
    cudaFree(d_input);

    // 如果只需要一个块，直接返回结果
    if (gridSize == 1) {
        CHECK_CUDA(cudaMemcpy(output, d_current, sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_current);
        cudaFree(d_next);
        return;
    }

    // Step 3: 多级归约
    int s = gridSize;
    while (s > 1) {
        blockSize = min(blockSize, s);
        gridSize = (s + blockSize - 1) / blockSize;
        if (kernel == reduce_with_idle_used ||
            kernel == reduce_with_expand_last_dim ||
            kernel == reduce_with_shuffle<256>) {
            gridSize = (s + blockSize * 2 - 1) / (blockSize * 2);
        }

        // 归约：将 d_current 的结果写入 d_next
        kernel<<<gridSize, blockSize>>>(d_current, d_next, s);  // 调用传入的核函数
        CHECK_CUDA(cudaGetLastError());

        // 更新状态：将 d_next 的内容作为下一次归约的输入
        std::swap(d_current, d_next);  // 确保逻辑清晰
        s = gridSize;
    }

    // Step 4: 拷贝最终结果到主机
    CHECK_CUDA(cudaMemcpy(output, d_current, sizeof(float), cudaMemcpyDeviceToHost));

    // Step 5: 释放设备内存
    cudaFree(d_current);
    cudaFree(d_next);
}

// ----------- 精度测试相关 -------------
float cpu_reduce(const std::vector<float> &input) {
    float sum = 0.0f, compensation = 0.0f;
    for (float v : input) {
        float y = v - compensation;
        float t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    return sum;
}

bool verify_result(const std::vector<float> &input, float gpu_result,
                   float tolerance = 1e-5f) {
    float cpu_result = cpu_reduce(input);
    float abs_error = std::fabs(cpu_result - gpu_result);
    float rel_error = (cpu_result != 0) ? abs_error / std::fabs(cpu_result) : 0.0f;

    std::cout << std::scientific << std::setprecision(7);
    std::cout << "[Validation] CPU: " << cpu_result << ", GPU: " << gpu_result
              << ", Abs Error: " << std::abs(cpu_result - gpu_result) << ", Rel Error: " << rel_error << std::endl;

    // 如果cpu_result比较小，允许一定绝对误差
    if (std::fabs(cpu_result) < 1e-3f)
        return abs_error < tolerance;
    else
        return rel_error < tolerance;
}

void test_accuracy_only(ReduceKernel kernel, int N) {
    std::vector<float> input(N);
    float gpu_output = 0.0f;
    initialize_array_random(input.data(), N, 0.0f, 1.0f);

    reduce(kernel, input.data(), &gpu_output, N);
    bool passed = verify_result(input, gpu_output);

    if (!passed) {
        std::cerr << "[FAIL] Accuracy test failed for N = " << N << std::endl;
    } else {
        std::cout << "[PASS] Accuracy test passed for N = " << N << std::endl;
    }
}

// ----------- 性能测试相关 -------------
void benchmark_only(ReduceKernel kernel, int N, int repeat = 1) {
    const int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    if (kernel == reduce_with_idle_used || kernel == reduce_with_expand_last_dim) {
        gridSize = (N + blockSize * 2 - 1) / (blockSize * 2);
    }

    std::vector<float> input(N);
    initialize_array_random(input.data(), N, 0.0f, 1.0f);

    float *d_input = nullptr;
    float *d_output = nullptr;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, gridSize * sizeof(float));
    cudaMemcpy(d_input, input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0;
    for (int i = 0; i < repeat; ++i) {
        cudaMemset(d_output, 0, gridSize * sizeof(float));
        cudaEventRecord(start);
        kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        total_time += gpu_timer_ms(start, stop);
    }

    std::cout << "[Performance] N = " << N
              << ", Avg Time = " << (total_time / repeat) << " ms"
              << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
}

// ----------- 测试入口 -------------
void reduce_accuracy_tests() {
    const int delimiter_length = 100;

    TestConfig tests[] = {
        {reduce_baseline, "REDUCE baseline"},
        {reduce_without_warp_divergence, "REDUCE without warp divergence"},
        {reduce_without_bank_conflict, "REDUCE without bank conflict"},
	    {reduce_with_idle_used, "REDUCE with idle used"},
        {reduce_with_expand_last_dim, "REDUCE with expand last dim"},
        {reduce_with_shuffle<256>, "REDUCE with shuffle"},
    };

    for (const auto& test : tests) {
        print_section_header("Accuracy Test: " + std::string(test.name), delimiter_length);
        std::vector<int> sizes = {1 << 23};
        for (int N : sizes) {
            test_accuracy_only(test.kernel, N);
        }
        std::cout << std::string(delimiter_length, '=') << std::endl;
    }
}

void reduce_benchmark_tests() {
    const int delimiter_length = 100;

    TestConfig tests[] = {
        {reduce_baseline, "REDUCE baseline"},
        {reduce_without_warp_divergence, "REDUCE without warp divergence"},
        {reduce_without_bank_conflict, "REDUCE without bank conflict"},
	    {reduce_with_idle_used, "REDUCE with idle used"},
        {reduce_with_expand_last_dim, "REDUCE with expand last dim"},
        {reduce_with_shuffle<256>, "REDUCE with shuffle"},
    };

    for (const auto& test : tests) {
        print_section_header("Benchmark: " + std::string(test.name), delimiter_length);
        std::vector<int> sizes = {1 << 23};
        for (int N : sizes) {
            benchmark_only(test.kernel, N);
        }
        std::cout << std::string(delimiter_length, '=') << std::endl;
    }
}
