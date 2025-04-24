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

std::function<float()> getRandomFloat = []() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
};

void initArray(float* arr, int size) {
  for (int i = 0; i < size; ++i) {
    arr[i] = getRandomFloat();
  }
}

int main() {
  // 1 SimpleSumReductionKernel
  int elemnums = 256;
  float *h_input = (float *)malloc(sizeof(float) * elemnums);
  float *h_output = (float *)malloc(sizeof(float));
  initArray(h_input, elemnums);
  float *d_input, *d_output;
  cudaMalloc((void **)&d_input, sizeof(float) * elemnums);
  cudaMalloc((void **)&d_output, sizeof(float) * elemnums);
  cudaMemcpy(d_input, h_input, sizeof(float) * elemnums, cudaMemcpyHostToDevice);
  SimpleSumReductionKernel<<<1, 256>>>(d_input, d_output);
  cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
  printf("SimpleSumReductionKernel result: %f\n", *h_output);
  cudaFree(d_output);
  cudaFree(d_input);
  free(h_input);
  free(h_output);
  
  // 2 ConvergentSumReductionKernel 
  h_input = (float *)malloc(sizeof(float) * elemnums);
  h_output = (float *)malloc(sizeof(float));
  initArray(h_input, elemnums);
  cudaMalloc((void **)&d_input, sizeof(float) * elemnums);
  cudaMalloc((void **)&d_output, sizeof(float) * elemnums);
  cudaMemcpy(d_input, h_input, sizeof(float) * elemnums, cudaMemcpyHostToDevice);
  ConvergentSumReductionKernel<<<1, 256>>>(d_input, d_output);
  cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
  printf("ConvergentSumReductionKernel result: %f\n", *h_output);
  cudaFree(d_output);
  cudaFree(d_input);
  free(h_input);
  free(h_output);

  // 3 SharedMemorySumReducetionKernel
  h_input = (float *)malloc(sizeof(float) * elemnums);
  h_output = (float *)malloc(sizeof(float));
  initArray(h_input, elemnums);
  cudaMalloc((void **)&d_input, sizeof(float) * elemnums);
  cudaMalloc((void **)&d_output, sizeof(float) * elemnums);
  cudaMemcpy(d_input, h_input, sizeof(float) * elemnums, cudaMemcpyHostToDevice);
  SharedMemorySumReducetionKernel<<<1, BLOCK_DIM>>>(d_input, d_output);
  cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
  printf("SharedMemorySumReducetionKernel result: %f\n", *h_output);
  cudaFree(d_output);
  cudaFree(d_input);
  free(h_input); 
  free(h_output);

  // 4 SegmentedSumReductionKernel
  elemnums = 1024;
  int blockSize = BLOCK_DIM;
  int gridSize = (elemnums + blockSize - 1) / blockSize;
  h_input = (float *)malloc(sizeof(float) * elemnums);
  h_output = (float *)malloc(sizeof(float));
  initArray(h_input, elemnums);
  cudaMalloc((void **)&d_input, sizeof(float) * elemnums);
  cudaMalloc((void **)&d_output, sizeof(float) * elemnums);
  cudaMemcpy(d_input, h_input, sizeof(float) * elemnums, cudaMemcpyHostToDevice);
  SegmentedSumReductionKernel<<<gridSize, blockSize>>>(d_input, d_output);
  cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
  printf("SegmentedSumReductionKernel result: %f\n", *h_output);
  cudaFree(d_output);
  cudaFree(d_input);
  free(h_input);
  free(h_output);

  // 5 CoarsenedSumReductionKernel
  elemnums = 1024;
  blockSize = BLOCK_DIM;
  gridSize = (elemnums + blockSize - 1) / blockSize;
  h_input = (float *)malloc(sizeof(float) * elemnums);
  h_output = (float *)malloc(sizeof(float));
  initArray(h_input, elemnums);
  cudaMalloc((void **)&d_input, sizeof(float) * elemnums);
  cudaMalloc((void **)&d_output, sizeof(float) * elemnums);
  cudaMemcpy(d_input, h_input, sizeof(float) * elemnums, cudaMemcpyHostToDevice);
  CoarsenedSumReductionKernel<<<gridSize, blockSize>>>(d_input, d_output);
  cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
  printf("CoarsenedSumReductionKernel result: %f\n", *h_output);
  cudaFree(d_output);
  cudaFree(d_input);
  free(h_input);  
  free(h_output);

  return 0;
}