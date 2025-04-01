#include "core/utils.h"
#include "kernels/kernels.h"

// 归约 Kernel：将每个 Block 内的数据归约为一个值
template<const int THREAD_PER_BLOCK=256>
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

template<const int THREAD_PER_BLOCK=256>
__global__ void reduce_without_warp_divergence(const float *input, float *output, int n) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx]: 0.0f;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (idx < blockDim.x) {
            sdata[idx] += sdata[idx + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];      
    }
}

template<typename REDUCE_OP, int THREAD_PER_BLOCK=256>
void reduce(const float *input, float* output, int n) {
    float *d_input = nullptr, *d_output = nullptr;

    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    const int output_size = gridSize * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, output_size));

    CHECK_CUDA(cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice));

    REDUCE_OP<<<gridSize, blockSize>>>(d_input, d_output, n);
    CHECK_CUDA(cudaGetLastError());
    cudaFree(d_input);

    // 多级归约：不断归约，直到结果只剩一个
    int s = gridSize;
    while (s > 1) {
        int threads = (s < blockSize) ? 1 << static_cast<int>(ceil(log2(s))) : blockSize;
        int blocks = (s + threads - 1) / threads;
        float *d_temp = nullptr;
        CHECK_CUDA(cudaMalloc(&d_temp, blocks * sizeof(float)));

        REDUCE_OP<<<blocks, threads>>>(d_output, d_temp, s);
        CHECK_CUDA(cudaGetLastError());
        cudaFree(d_output);
        d_output = d_temp;
        s = blocks;
    }

    CHECK_CUDA(cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_output);
}

// // 多级归约包装函数：当初始 Block 数大于 1 时，进行多次归约直到最终只剩一个结果
// void reduce_baseline_wrapper(const float *input, float *output, int n) {
//     float *d_input = nullptr, *d_output = nullptr;

//     const int blockSize = 256;
//     const int gridSize = (n + blockSize - 1) / blockSize;
//     const int output_size = gridSize * sizeof(float);

//     CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(float)));
//     CHECK_CUDA(cudaMalloc(&d_output, output_size));

//     CHECK_CUDA(cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice));

//     reduce_baseline_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);
//     CHECK_CUDA(cudaGetLastError());
//     cudaFree(d_input);

//     // 多级归约：不断归约，直到结果只剩一个
//     int s = gridSize;
//     while (s > 1) {
//         int threads = (s < blockSize) ? 1 << static_cast<int>(ceil(log2(s))) : blockSize;
//         int blocks = (s + threads - 1) / threads;
//         float *d_temp = nullptr;
//         CHECK_CUDA(cudaMalloc(&d_temp, blocks * sizeof(float)));

//         reduce_baseline_kernel<<<blocks, threads>>>(d_output, d_temp, s);
//         CHECK_CUDA(cudaGetLastError());
//         cudaFree(d_output);
//         d_output = d_temp;
//         s = blocks;
//     }

//     CHECK_CUDA(cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
//     cudaFree(d_output);
// }

// 精度测试
bool verify_result(const std::vector<float>& input, float gpu_result, float tolerance = 1e-5f) {
    float cpu_result = cpu_reduce(input);
    float abs_error = std::fabs(cpu_result - gpu_result);
    float rel_error = (cpu_result != 0) ? abs_error / std::fabs(cpu_result) : 0.0f;

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "[Validation] CPU: " << cpu_result 
              << ", GPU: " << gpu_result
              << ", Abs Error: " << abs_error
              << ", Rel Error: " << rel_error << std::endl;

    return abs_error < tolerance;
}

// Kahan求和算法
float cpu_reduce(const std::vector<float>& input) {
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

// GPU 归约性能测试
template <typename REDUCE_OP>
void benchmark_reduce(int N, int repeat = 5) {
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
        recude<REDUCE_OP>(input.data(), &gpu_output, N);
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

// 测试不同数据规模
void test_reduce_baseline() {
    std::cout << "========================================= Test for REDUCE basline =========================================" << std::endl;
    std::vector<int> sizes = {1 << 20, 1 << 24, 1 << 27};  // 1M, 16M, 128M 元素
    for (int N : sizes) {
        benchmark_reduce<reduce_baseline>(N);
    }
    std::cout << "===========================================================================================================" << std::endl;
}