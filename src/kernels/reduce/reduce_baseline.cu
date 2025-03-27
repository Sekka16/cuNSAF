#include "core/utils.h"
#include "kernels/kernels.h"

// 归约 Kernel：将每个 Block 内的数据归约为一个值
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

// 多级归约包装函数：当初始 Block 数大于 1 时，进行多次归约直到最终只剩一个结果
void reduce_baseline_wrapper(const float *input, float *output, int n) {
    float *d_input = nullptr, *d_output = nullptr;

    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    const int output_size = gridSize * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, output_size));

    CHECK_CUDA(cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice));

    reduce_baseline_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    CHECK_CUDA(cudaGetLastError());
    cudaFree(d_input);

    // 多级归约：不断归约，直到结果只剩一个
    int s = gridSize;
    while (s > 1) {
        int threads = (s < blockSize) ? 1 << static_cast<int>(ceil(log2(s))) : blockSize;
        int blocks = (s + threads - 1) / threads;
        float *d_temp = nullptr;
        CHECK_CUDA(cudaMalloc(&d_temp, blocks * sizeof(float)));

        reduce_baseline_kernel<<<blocks, threads>>>(d_output, d_temp, s);
        CHECK_CUDA(cudaGetLastError());
        cudaFree(d_output);
        d_output = d_temp;
        s = blocks;
    }

    CHECK_CUDA(cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_output);
}

// 验证函数：计算 CPU 上的和，并与 GPU 结果对比
bool verify_result(const float* input, int n, float gpu_result, float tolerance = 1e-5f) {
    float cpu_sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        cpu_sum += input[i];
    }

    float abs_error = fabs(cpu_sum - gpu_result);
    float rel_error = (fabs(cpu_sum) > 0) ? abs_error / fabs(cpu_sum) : 0.0f;

    std::cout << std::scientific << std::setprecision(4);
    std::cout << "Validation Report:\n"
              << "  CPU Sum:     " << cpu_sum << "\n"
              << "  GPU Sum:     " << gpu_result << "\n"
              << "  Abs Error:   " << abs_error << "\n"
              << "  Rel Error:   " << rel_error << "\n"
              << "  Tolerance:   " << tolerance << "\n"
              << "  Elements:    " << n << std::endl;

    bool is_valid = (abs_error < tolerance);
    std::cout << "Result:        " << (is_valid ? "PASS" : "FAIL") << std::endl;
    
    return is_valid;
}

// 测试函数：随机初始化数组后执行归约，并验证结果
void test_reduce_baseline() {
    // 此处规模设置为 32 * 1024 * 1024，即 32M 个元素（约 128MB 内存，注意测试时内存消耗）
    const int N = 32 * 1024 * 1024;
    float* input = new float[N];
    float gpu_output = 0.0f;

    // 用随机数初始化数组，例如生成 0 到 1 之间均匀分布的浮点数
    initialize_array_random(input, N, 0.0f, 1.0f);

    // 执行 GPU 归约计算
    reduce_baseline_wrapper(input, &gpu_output, N);

    // 验证 GPU 归约结果
    verify_result(input, N, gpu_output);

    // 额外理论检查：对于均匀分布[0, 1]，期望均值为0.5，因此期望和约为 N * 0.5
    float expected_sum = static_cast<float>(N) * 0.5f;
    float error_to_expected = fabs(gpu_output - expected_sum);
    std::cout << "\nTheoretical Check:\n"
              << "  Expected:    " << expected_sum << "\n"
              << "  Actual:      " << gpu_output << "\n"
              << "  Error:       " << error_to_expected 
              << " (" << (error_to_expected / N) * 100 << "% of elements)\n";

    delete[] input;
}