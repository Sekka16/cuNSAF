#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "histogram.cu"

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

class HistogramTest : public ::testing::Test {
protected:
    const char *input = "abcxyzabcxyz";
    const unsigned int length = strlen(input);
    const unsigned int num_bins = NUM_BINS;

    char *d_data;
    unsigned int *d_histo;
    unsigned int h_histo[NUM_BINS];

    void SetUp() override {
        CHECK_CUDA(cudaMalloc(&d_data, length * sizeof(char)));
        CHECK_CUDA(cudaMalloc(&d_histo, num_bins * sizeof(unsigned int)));
        CHECK_CUDA(cudaMemcpy(d_data, input, length * sizeof(char), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_histo, 0, num_bins * sizeof(unsigned int)));
        memset(h_histo, 0, num_bins * sizeof(unsigned int));
    }

    void TearDown() override {
        CHECK_CUDA(cudaFree(d_data));
        CHECK_CUDA(cudaFree(d_histo));
    }

    void RunKernel(void (*kernel)(char *, unsigned int, unsigned int *)) {
        dim3 blockDim(256);
        dim3 gridDim((length + blockDim.x - 1) / blockDim.x);
        kernel<<<gridDim, blockDim>>>(d_data, length, d_histo);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_histo, d_histo, num_bins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }
};

TEST_F(HistogramTest, HistoKernel) {
    RunKernel(histo_kernel);
    ASSERT_EQ(h_histo['a' - 'a'], 2);
    ASSERT_EQ(h_histo['b' - 'a'], 2);
    ASSERT_EQ(h_histo['c' - 'a'], 2);
    ASSERT_EQ(h_histo['x' - 'a'], 2);
    ASSERT_EQ(h_histo['y' - 'a'], 2);
    ASSERT_EQ(h_histo['z' - 'a'], 2);
}

TEST_F(HistogramTest, HistoPrivateKernelSimple) {
    RunKernel(histo_private_kernel_simple);
    ASSERT_EQ(h_histo['a' - 'a'], 2);
    ASSERT_EQ(h_histo['b' - 'a'], 2);
    ASSERT_EQ(h_histo['c' - 'a'], 2);
    ASSERT_EQ(h_histo['x' - 'a'], 2);
    ASSERT_EQ(h_histo['y' - 'a'], 2);
    ASSERT_EQ(h_histo['z' - 'a'], 2);
}

TEST_F(HistogramTest, HistoPrivateKernelAdvanced) {
    RunKernel(histo_private_kernel_advanced);
    ASSERT_EQ(h_histo['a' - 'a'], 2);
    ASSERT_EQ(h_histo['b' - 'a'], 2);
    ASSERT_EQ(h_histo['c' - 'a'], 2);
    ASSERT_EQ(h_histo['x' - 'a'], 2);
    ASSERT_EQ(h_histo['y' - 'a'], 2);
    ASSERT_EQ(h_histo['z' - 'a'], 2);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}