#!/bin/bash
BUILD_DIR="build"

# 显式指定CUDA路径（如果自动检测失败）
export CUDA_PATH=/usr/local/cuda  # 根据实际安装路径修改

cmake -B ${BUILD_DIR} -S . \
    -DCMAKE_CUDA_ARCHITECTURES="native" \
    -DCUDAToolkit_ROOT=${CUDA_PATH}

cmake --build ${BUILD_DIR} --parallel $(nproc)