#ifndef CORE_UTILS_H
#define CORE_UTILS_H

#include <iostream>
#include <random>      // 新增随机数库
#include <type_traits> // 用于类型检查

template <typename T>
void initialize_array(T *arr, size_t size, T min = T(0), T max = T(1))
{
  // 随机数生成器
  std::random_device rd;
  std::mt19937 gen(rd());

  // 根据类型选择分布
  if constexpr (std::is_integral_v<T>)
  {
    std::uniform_int_distribution<T> dis(min, max);
    for (size_t i = 0; i < size; ++i)
    {
      arr[i] = dis(gen);
    }
  }
  else
  {
    std::uniform_real_distribution<T> dis(min, max);
    for (size_t i = 0; i < size; ++i)
    {
      arr[i] = dis(gen);
    }
  }
}

template <typename T>
void print_array(const T *arr, size_t size)
{
  for (size_t i = 0; i < size; ++i)
  {
    std::cout << arr[i] << " ";
  }
  std::cout << std::endl;
}

#endif
