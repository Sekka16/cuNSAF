#include "core/utils.h"
#include "kernels/kernels.h"
#include <iostream>

template<const int THREAD_PER_BLOCK=256>
__global__ void reduce_baseline_kernel(const float *input, float *output, int n) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

void reduce_baseline_wrapper(const float *input, float *output, int n) {
    float *d_input, *d_output;

    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    const int output_size = gridSize * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, output_size));

    CHECK_CUDA(cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice));

    reduce_baseline_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);

    if (gridSize > 1) {
        float *d_final;
        CHECK_CUDA(cudaMalloc(&d_final, sizeof(float)));
        reduce_baseline_kernel<<<1, blockSize>>>(d_output, d_final, gridSize);
        CHECK_CUDA(cudaMemcpy(output, d_final, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(d_final));
    } else {
	    CHECK_CUDA(cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

// 改进后的验证函数
bool verify_result(const float* input, int n, float gpu_result, float tolerance = 1e-5f) {
    // 计算CPU参考值
    float cpu_sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        cpu_sum += input[i];
    }

    // 计算绝对误差和相对误差
    float abs_error = fabs(cpu_sum - gpu_result);
    float rel_error = abs_error / fabs(cpu_sum);

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

// 改进后的测试函数
void test_reduce_baseline() {
    const int N = 32 * 1024 * 1024;  // 256M elements
    float* input = new float[N];
    float gpu_output = 0.0f;

    initialize_array(input, N, 1.0f); 

    // 执行GPU计算
    reduce_baseline_wrapper(input, &gpu_output, N);

    // 详细验证
    verify_result(input, N, gpu_output);

    // 额外检查理论值
    float expected_sum = static_cast<float>(N);
    float error_to_expected = fabs(gpu_output - expected_sum);
    std::cout << "\nTheoretical Check:\n"
              << "  Expected:    " << expected_sum << "\n"
              << "  Actual:      " << gpu_output << "\n"
              << "  Error:       " << error_to_expected 
              << " (" << (error_to_expected/N)*100 << "% of elements)\n";

    delete[] input;
}