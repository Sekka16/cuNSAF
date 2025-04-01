#include "core/utils.h"
#include "kernels/kernels.h"

#define THREAD_PER_BLOCK 256
// 归约 Kernel：将每个 Block 内的数据归约为一个值
__global__ void reduce_baseline(const float *input, float *output, int n) {
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

__global__ void reduce_without_warp_divergence(const float *input, float *output, int n) {
  __shared__ float sdata[THREAD_PER_BLOCK];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (idx < n) ? input[idx] : 0.0f;
  __syncthreads();

  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid < blockDim.x - s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

void reduce(ReduceKernel kernel, const float *input, float *output, int n) {
    // Step 1: 分配设备内存并拷贝输入数据
    float *d_input = nullptr, *d_current = nullptr, *d_next = nullptr;

    const int blockSize = 256;  // 固定线程块大小
    const int gridSize = (n + blockSize - 1) / blockSize;
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
        int threads = min(blockSize, s);
        int blocks = (s + threads - 1) / threads;

        // 归约：将 d_current 的结果写入 d_next
        kernel<<<blocks, threads>>>(d_current, d_next, s);  // 调用传入的核函数
        CHECK_CUDA(cudaGetLastError());

        // 更新状态：将 d_next 的内容作为下一次归约的输入
        std::swap(d_current, d_next);  // 确保逻辑清晰
        s = blocks;
    }

    // Step 4: 拷贝最终结果到主机
    CHECK_CUDA(cudaMemcpy(output, d_current, sizeof(float), cudaMemcpyDeviceToHost));

    // Step 5: 释放设备内存
    cudaFree(d_current);
    cudaFree(d_next);
}

// 精度测试
bool verify_result(const std::vector<float> &input, float gpu_result,
                   float tolerance = 1e-5f) {
  float cpu_result = cpu_reduce(input);
  float abs_error = std::fabs(cpu_result - gpu_result);
  float rel_error =
      (cpu_result != 0) ? abs_error / std::fabs(cpu_result) : 0.0f;

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "[Validation] CPU: " << cpu_result << ", GPU: " << gpu_result
            << ", Abs Error: " << abs_error << ", Rel Error: " << rel_error
            << std::endl;

  return abs_error < tolerance;
}

// Kahan求和算法
float cpu_reduce(const std::vector<float> &input) {
  float sum = 0.0f;
  float compensation = 0.0f; // 补偿值
  for (float v : input) {
    float y = v - compensation;
    float t = sum + y;
    compensation = (t - sum) - y;
    sum = t;
  }
  return sum;
}

void benchmark_reduce(ReduceKernel kernel, int N, int repeat = 5) {
    std::vector<float> input(N);
    float gpu_output = 0.0f;

    initialize_array_random(input.data(), N, 0.0f, 1.0f);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 运行多次，取平均值
    float total_time = 0;
    for (int i = 0; i < repeat; i++) {
        cudaEventRecord(start);
        reduce(kernel, input.data(), &gpu_output, N);  // 调用 reduce 函数
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        total_time += gpu_timer_ms(start, stop);
    }

    // 输出性能结果
    std::cout << "[Performance] N = " << N
              << ", Avg Time = " << (total_time / repeat) << " ms" << std::endl;

    verify_result(input, gpu_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void run_reduce_tests() {
    const int delimiter_length = 100;

    TestConfig tests[] = {
        {reduce_baseline, "REDUCE baseline"},
        {reduce_without_warp_divergence, "REDUCE without warp divergence"}
    };

    for (const auto& test : tests) {
        print_section_header("Test for " + std::string(test.name), delimiter_length);

        std::vector<int> sizes = {1 << 8, 1 << 16, 1 << 24};
        for (int N : sizes) {
            benchmark_reduce(test.kernel, N);
        }

        std::cout << std::string(delimiter_length, '=') << std::endl;
    }
}