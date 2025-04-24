#include <bits/stdc++.h>
#include <cuda_runtime.h>

__global__ void histo_kernel(char *data, unsigned int length, unsigned int *histo) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&histo[alphabet_position / 4], 1);
        }
    }
}

#define NUM_BINS 7 // 直方图中项的数量

__global__ void histo_private_kernel_simple(char *data, unsigned int length, unsigned int *histo) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&histo[blockIdx.x * NUM_BINS + alphabet_position / 4], 1);
        }
    }
    if (blockIdx.x > 0) {
        __syncthreads();
        for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
            unsigned int binValue = histo[blockIdx.x * NUM_BINS + bin];
            if (binValue > 0) {
                atomicAdd(&histo[bin], binValue);
            }
        }
    }
}

// 每个block有块共享内存作为私有histogram
__global__ void histo_private_kernel_advanced(char *data, unsigned int length, unsigned int *histo) {
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }
    __syncthreads();
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&histo_s[alphabet_position / 4], 1);
        }
    }
    __syncthreads();
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        if (binValue > 0) {
            atomicAdd(&histo[blockIdx.x * NUM_BINS + bin], binValue);
        }
    }
}

#define CFACTOR 2

// 连续分区进行粗化
__global__ void histo_private_kernel_coarsened(char *data, unsigned int length, unsigned int *histo) {
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }
    __syncthreads();
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid * CFACTOR; i < min((tid + 1) * CFACTOR, length); i++) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&histo_s[alphabet_position / 4], 1);
        }
    }
    __syncthreads();
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        if (binValue > 0) {
            atomicAdd(&histo[bin], binValue);
        }
    }
}

// 交错分区进行粗化
__global__ void histo_private_kernel_coarsened_interleaved(char *data, unsigned int length, unsigned int *histo) {
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }
    __syncthreads();
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid; i < length; i += gridDim.x * blockDim.x) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&histo_s[alphabet_position / 4], 1);
        }
    }
    __syncthreads();
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        if (binValue > 0) {
            atomicAdd(&histo[bin], binValue);
        }
    }
}

// 将每个线程连续的更新聚合成单个更新
__global__ void histo_private_kernel_accumulator(char *data, unsigned int length, unsigned int *histo) {
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }
    __syncthreads();
    unsigned int accumulator = 0;
    int prevBinIdx = -1;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid; i < length; i+= blockDim.x * gridDim.x) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            int bin = alphabet_position / 4;
            if (bin != prevBinIdx) {
                ++accumulator;
            } else {
                if (accumulator > 0) {
                    atomicAdd(&histo_s[prevBinIdx], accumulator);
                }
                accumulator = 1;
                prevBinIdx = bin;
            }
        }
    }
    if (accumulator > 0) {
        atomicAdd(&histo_s[prevBinIdx], accumulator);
    }
    __syncthreads();
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        if (binValue > 0) {
            atomicAdd(&histo[bin], binValue);
        }
    }
}