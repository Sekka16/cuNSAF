#include <bits/stdc++.h>
#include <cuda_runtime.h>

__global__ void SimpleSumReductionKernel(float* input, float* output) {
  unsigned int i = 2 * threadIdx.x;
  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
    if (threadIdx.x % stride == 0) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *output = input[0];
  }
}

__global__ void ConvergentSumReductionKernel(float* input, float* output) {
  unsigned int i = threadIdx.x;
  for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
    if (threadIdx.x < stride) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *output = input[0];
  }
}

#define BLOCK_DIM 256

__global__ void SharedMemorySumReducetionKernel(float* input, float* output) {
  __shared__ float input_s[BLOCK_DIM];
  unsigned int t = threadIdx.x;
  input_s[t] = input[t] + input[t + BLOCK_DIM];
  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      input_s[t] += input_s[t + stride];
    }
    if (threadIdx.x == 0) {
      *output = input_s[0];
    }
  }
}

__global__ void SegmentedSumReductionKernel(float* input, float* output) {
  __shared__ float input_s[BLOCK_DIM];
  unsigned int segment = 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  unsigned int t = threadIdx.x;
  input_s[t] = input[i] + input[i + BLOCK_DIM];
  for(unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      input_s[t] += input_s[t + stride];
    }
  }
  if (t == 0) {
    atomicAdd(output, input_s[0]);
  }
}

#define COARSE_FACTOR 2

__global__ void CoarsenedSumReductionKernel(float* input, float* output) {
  __shared__ float input_s[BLOCK_DIM];
  unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  unsigned int t = threadIdx.x;
  float sum = input[i];
  for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile) {
    sum += input[i + tile * BLOCK_DIM];
  }
  input_s[t] = sum;
  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      input_s[t] += input_s[t + stride];
    }
  }
  if (t == 0) {
    atomicAdd(output, input_s[0]);
  }
}