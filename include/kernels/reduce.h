#ifndef KERNELS_REDUCE_H
#define KERNELS_REDUCE_H

struct TestConfig {
    ReduceKernel kernel;
    const char* name;
};

// 测试函数
void run_reduce_tests();

float cpu_reduce(const std::vector<float> &input);

void benchmark_reduce(ReduceKernel kernel, int N, int repeat);

#endif