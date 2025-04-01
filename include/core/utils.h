#ifndef CORE_UTILS_H
#define CORE_UTILS_H

#include <iostream>
#include <random>      // 新增随机数库
#include <type_traits> // 用于类型检查

template <typename T>
void initialize_array_random(T* array, int n, T lower_bound = 0, T upper_bound = 1) {
    std::random_device rd;  // 真随机数种子（若系统支持）
    std::mt19937 gen(rd()); // Mersenne Twister 伪随机数生成器
    std::uniform_real_distribution<float> dis(lower_bound, upper_bound);
    for (int i = 0; i < n; ++i) {
        array[i] = dis(gen);
    }
}

template <typename T>
void print_array(const T *arr, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    std::cout << arr[i] << " ";
  }
  std::cout << std::endl;
}

// 计时工具
float gpu_timer_ms(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
}

std::string center(const std::string& text, int width) {
    int left = (width - text.length()) / 2;
    return std::string(left, ' ') + text + std::string(width - left - text.length(), ' ');
}

void print_section_header(const std::string& title, int delimiter_length = 80) {
    const std::string delimiter(delimiter_length, '=');
    std::cout << delimiter << std::endl;
    std::cout << title << std::endl;
    std::cout << delimiter << std::endl;
}

#endif
