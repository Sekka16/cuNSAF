# 声明CUDA源文件
set(ADD_SOURCES
    add.cu
)

# 创建独立静态库
add_library(add_kernel STATIC ${ADD_SOURCES})

# 设置目标属性
set_target_properties(add_kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# 私有包含目录
target_include_directories(add_kernel PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)