#ifndef KERNELS_ADD_H
#define KERNELS_ADD_H

#include "core/common.h"

void add_wrapper(const float *a, const float *b, float *c, int n);

void test_add();
#endif
