#ifndef KERNELS_REDUCE_H
#define KERNELS_REDUCE_H

typedef void (*ReduceKernel)(const float *, float *, int);

struct TestConfig {
    ReduceKernel kernel;
    const char* name;
};

__global__ void reduce_baseline(const float *input, float *output, int n);
__global__ void reduce_without_warp_divergence(const float *input, float *output, int n);
__global__ void reduce_without_bank_conflict(const float *input, float *output, int n);

// 测试函数
float cpu_reduce(const std::vector<float> &input);

void reduce_accuracy_tests();

void reduce_benchmark_tests();

#endif
