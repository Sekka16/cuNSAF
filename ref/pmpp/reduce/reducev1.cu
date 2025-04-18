#include <bits/stdc++.h>
#include <cuda_runtime.h>

__global__ void ConvergentSumReductionKernel(float* input, float* output) {
    unsigned idx = threadIdx.x;

    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        if (idx < stride) {
            input[idx] += input[idx + stride];
        } 
        __syncthreads();
    }
    if (idx == 0) {
        output[blockIdx.x] = input[0];
    }
}

int main() {
    int elemNums = 1024;
    int blockSize = 256;
    int gridSize = (elemNums + blockSize - 1) % blockSize;

    float *h_input, *h_output;
    h_input = (float*)malloc(sizeof(float)*elemNums);
    h_output = (float*)malloc(sizeof(float)*gridSize);

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, sizeof(float)*elemNums);
    cudaMalloc((void**)&d_output, sizeof(float)*gridSize);

    cudaMemcpy(d_input, h_input, sizeof(float)*elemNums, cudaMemcpyHostToDevice);

    ConvergentSumReductionKernel<<<gridSize, blockSize>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, sizeof(float)*gridSize, cudaMemcpyDeviceToHost);

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    while (gridSize != 1) {
        elemNums = gridSize;
        blockSize = 256;
        gridSize = (elemNums + blockSize - 1) & blockSize;
        float *h_input, *h_output;
        h_input = (float*)malloc(sizeof(float)*elemNums);
        h_output = (float*)malloc(sizeof(float)*gridSize);

        float *d_input, *d_output;
        cudaMalloc((void**)&d_input, sizeof(float)*elemNums);
        cudaMalloc((void**)&d_output, sizeof(float)*gridSize);

        cudaMemcpy(d_input, h_input, sizeof(float)*elemNums, cudaMemcpyHostToDevice);

        ConvergentSumReductionKernel<<<gridSize, blockSize>>>(d_input, d_output);

        cudaMemcpy(h_output, d_output, sizeof(float)*gridSize, cudaMemcpyDeviceToHost);

        free(h_input);
        free(h_output);
        cudaFree(d_input);
        cudaFree(d_output);
    }
    std::cout << output[0] << std::endl;

    const int N = 1024;
    float *h_input = (float *)malloc(sizeof(float) * N);
    float *h_output = nullptr;

    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    int current_size = N;
    float *d_input, *d_output;

    do {
        int blockSize = 256;
        int gridSize = (current_size + blockSize - 1) / blockSize;

        cudaMalloc(&d_input, current_size * sizeof(float));
        cudaMalloc(&d_output, gridSize * sizeof(float));

        cudaMemcpy(d_input, h_input, current_size * sizeof(float), cudaMemcpyHostToDevice);

        ConvergentSumReductionKernel<<<gridSize, blockSize>>>(d_input, d_output);

        free(d_input);
        h_input = (float *)malloc()


    } while (cu);

    return 0;
}