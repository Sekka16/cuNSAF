#ifndef KERNELS_ADD_H
#define KERNELS_ADD_H

typedef void (*ReduceKernel)(const float *, float *, int);

__global__ void reduce_baseline(const float *input, float *output, int n);
__global__ void reduce_without_warp_divergence(const float *input, float *output, int n);
#endif
