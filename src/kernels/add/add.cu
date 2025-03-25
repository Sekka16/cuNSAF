#include "core/utils.h"
#include "kernels/add.h"
#include <iostream>

__global__ void add_kernel(const float *a, const float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

void add_wrapper(const float *a, const float *b, float *c, int n) {
  float *d_a, *d_b, *d_c;

  // 设备内存分配
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_c, n * sizeof(float)));

  // 数据拷贝
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

  // 启动核函数
  const int blockSize = 256;
  const int gridSize = (n + blockSize - 1) / blockSize;
  add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

  // 同步并检查错误
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // 拷贝回结果
  CHECK_CUDA(cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

  // 释放设备内存
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
}

// 新增测试入口函数
void test_add()
{
  const int N = 100;
  float *a = new float[N];
  float *b = new float[N];
  float *c = new float[N];

  // 初始化输入
  initialize_array(a, N, 1.0f); // [1, 2, 3, 4, 5]
  initialize_array(b, N, 2.0f); // [2, 3, 4, 5, 6]

  // 执行加法
  add_wrapper(a, b, c, N);

  // // 打印结果
  // std::cout << "Result: ";
  // print_array(c, N); // 应该输出 [3, 5, 7, 9, 11]

  // 验证结果
  bool success = true;
  for (int i = 0; i < N; ++i)
  {
    if (c[i] != a[i] + b[i])
    {
      std::cerr << "Error at index " << i << ": " << c[i] << " vs "
                << a[i] + b[i] << std::endl;
      success = false;
    }
  }
  std::string message = success ? "[Test Passed]" : "[Test Failed]";
  std::cout << message << std::endl;

  // 清理内存
  delete[] a;
  delete[] b;
  delete[] c;
}