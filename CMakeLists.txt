cmake_minimum_required(VERSION 3.18)
project(cuNSAF LANGUAGES CXX CUDA)

# 设置CUDA架构（根据实际GPU计算能力修改）
set(CMAKE_CUDA_ARCHITECTURES "native")  # 自动检测或手动指定如 "86"

# 查找CUDA工具包
find_package(CUDAToolkit REQUIRED)

# 包含目录配置
include_directories(
    include
    ${CUDAToolkit_INCLUDE_DIRS}  # 关键：添加CUDA头文件路径
)

# 源文件配置
file(GLOB_RECURSE SOURCES "src/kernels/add/add.cu")
file(GLOB_RECURSE SOURCES "src/kernels/reduce/reduce_baseline.cu")

# 可执行文件配置
add_executable(main 
    src/main.cpp 
    ${SOURCES}
)

# 链接库配置
target_link_libraries(main PRIVATE 
    CUDA::cudart  # CUDA运行时库
)