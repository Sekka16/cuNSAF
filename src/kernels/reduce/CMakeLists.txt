set(ADD_SOURCES
    reduce_baseline.cu
)

add_library(reduce_baseline_kernel STATIC $(ADD_SOURCES))

set_target_properties(reduce_baseline_kernel EXPORT_PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(reduce_baseline_kernel PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)