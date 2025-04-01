#ifndef KERNELS_REDUCE_H
#define KERNELS_REDUCE_H

// 测试函数
void test_reduce_baseline();

float cpu_reduce(const std::vector<float>& input);

void benchmark_reduce(int N, int repeat);

#endif