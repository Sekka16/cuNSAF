#ifndef KERNELS_REDUCE_H
#define KERNELS_REDUCE_H

#include "core/common.h"
#include <cstddef> // for size_t

// 非优化版Reduce求和
float reduce_sum(const float* input, size_t n);

// 测试函数
void test_reduce_baseline();

#endif