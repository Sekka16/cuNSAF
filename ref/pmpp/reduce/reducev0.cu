#include <bits/stdc++.h>
#include <cuda_runtime.h>

// 单个block计算出总和
__global__ void SimpleSumReductionKernel(float *input, float *output) {

  int idx = 2 * threadIdx.x;
  
  for (int stride = 1; stride < blockDim.x; stride <<= 1) {
    if (idx % stride == 0)
      input[idx] += input[idx + stride];  
    __syncthreads();
  }

  if (idx == 0) {
    *output = input[0];
  }
}

int main() {
  // 1. 设置blockSize和gridSize
  const int elemNums = 256;
  const int blockSize = 256;
  const int gridSize = (elemNums + blockSize - 1) / blockSize;

  // 2. 分配host端内存
  float *h_input = (float *)malloc(sizeof(float) * elemNums);
  float *h_output = (float *)malloc(sizeof(float) * elemNums);

  srand(time(NULL));
  for (int i = 0; i < elemNums; i++) {
    h_input[i] = (float)rand() / RAND_MAX;
  }

  // 3. 分配device端内存
  float* d_input, *d_output;
  cudaMalloc((void **)&d_input, elemNums * sizeof(float));
  cudaMalloc((void **)&d_output, sizeof(float));

  // 4. 拷贝host端的input到device端
  cudaMemcpy(d_input, h_input, elemNums * sizeof(float), cudaMemcpyHostToDevice);

  // 5. 启动内核
  SimpleSumReductionKernel<<<gridSize, blockSize>>>(d_input, d_output);

  // 6. 拷贝device端output到host端
  cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << *h_output << std::endl;

  // 7. 释放内存
  free(h_input);
  free(h_output);
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}
