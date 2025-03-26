#include "core/utils.h"
#include "kernels/reduce.h"
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

    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_output, output, n * sizeof(float), cudaMemcpyHostToDevice));

    const int blockSize = 256;                              // block中线程数量
    const int gridSize = (n + blockSize - 1) / blockSize;   // grid中block数量

    reduce_baseline_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);

    if (gridSize > 1) {
        float *d_final;
        cudaMalloc(&d_final, sizeof(float));
        reduce_baseline_kernel<<<1, blockSize>>>(d_output, d_final, gridSize);
        cudaMemcpy(output, d_final, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_final);
    } else {
        cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_input);
    cudaFree(d_output);
}

// 验证函数
bool verify_result(const float *input, int n, float result) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += input[i];
    }
    return fabs(sum - result) < 1e-5;
}

void test_reduce_baseline() {
    const int N = 1024;
    float *input = new float[N];
    float output = 0.0f;

    initialize_array(input, N, 1.0f);

    reduce_baseline_wrapper(input, &output, N);
    std::cout << "Verification: " 
              << (verify_result(input, N, output) ? "PASS" : "FAIL") 
              << std::endl;
    delete[] input;
}