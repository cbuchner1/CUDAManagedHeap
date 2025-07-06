#include <gtest/gtest.h>
#define _DEBUG 1 // makes CUDA_CHECKED_CALL throw
#include "examples/cudamanagedheap.cuh"
#include <cstring>
#include <iostream>

constexpr size_t HEAP_SIZE = 8U*1024U*1024U;

namespace {
int g_cuda_device = 0;
bool g_device_exists = true;
bool g_device_supported = true;
}

class ManagedHeapTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (cudaSetDevice(::g_cuda_device) != 0) {
            ::g_device_exists = false;
        } else {
            cudaDeviceProp deviceProp;
            if (cudaGetDeviceProperties(&deviceProp, ::g_cuda_device) != 0) {
                ::g_device_exists = false;
            } else {
                std::cout << "Using device: " << deviceProp.name << std::endl;
                if (deviceProp.major < 2) {
                    std::cerr << "This GPU with Compute Capability " << deviceProp.major
                            << '.' << deviceProp.minor
                            << " does not meet minimum requirements." << std::endl;
                    std::cerr << "A GPU with Compute Capability >= 2.0 is required." << std::endl;
                    ::g_device_supported = false;
                }
            }
        }
        initHeap(HEAP_SIZE);
    }

    void SetUp() override {
        if (!::g_device_exists) {
            GTEST_SKIP() << "CUDA device #" << ::g_cuda_device << " does not exist or does not return device properties";
        }
        if (!::g_device_supported) {
            GTEST_SKIP() << "CUDA device #" << ::g_cuda_device << " has insufficient compute capability.";
        }
    }

    static void TearDownTestSuite() {
        destroyHeap();
        cudaDeviceReset();
    }
};

__global__ void Malloc_kernel(uint** parray)
{
    parray[threadIdx.x + blockIdx.x*blockDim.x] = static_cast<uint*>(malloc(sizeof(uint)));
}

__global__ void Free_kernel(uint** parray)
{
    free(parray[threadIdx.x + blockIdx.x*blockDim.x]);
}

TEST_F(ManagedHeapTest, MallocFreeOnDevice)
{
    size_t block = 128;
    size_t grid = 64;

    uint** data;
    CUDA_CHECKED_CALL(cudaMallocManaged(&data, grid * block * sizeof(uint*)));

    Malloc_kernel<<<grid, block>>>(data);
    CUDA_CHECKED_CALL(cudaDeviceSynchronize());

    Free_kernel<<<grid, block>>>(data);
    CUDA_CHECKED_CALL(cudaDeviceSynchronize());

    cudaFree(data);
}


TEST_F(ManagedHeapTest, getAvailableBytes)
{
    size_t block = 128;
    size_t grid = 64;

    uint** data;
    CUDA_CHECKED_CALL(cudaMallocManaged(&data, grid * block * sizeof(uint*)));

    size_t bytes_initialAmount = theHeap_ph->getAvailableBytes();
    EXPECT_LT(bytes_initialAmount, HEAP_SIZE);

    Malloc_kernel<<<grid, block>>>(data);
    CUDA_CHECKED_CALL(cudaDeviceSynchronize());

    size_t bytes_afterAllocation = theHeap_ph->getAvailableBytes();

    EXPECT_LT(bytes_afterAllocation, bytes_initialAmount);

    Free_kernel<<<grid, block>>>(data);
    CUDA_CHECKED_CALL(cudaDeviceSynchronize());

    size_t bytes_afterFree = theHeap_ph->getAvailableBytes();

    EXPECT_GT(bytes_afterFree, bytes_afterAllocation);

    // NOTE: this expectation isn't yet holding up for some reason
    //EXPECT_EQ(bytes_initialAmount, bytes_afterFree);

    cudaFree(data);
}


__global__ void New_kernel(uint** parray)
{
    parray[threadIdx.x + blockIdx.x*blockDim.x] = new uint;
}

__global__ void Delete_kernel(uint** parray)
{
    delete parray[threadIdx.x + blockIdx.x*blockDim.x];
}

TEST_F(ManagedHeapTest, NewDeleteOnDevice)
{
    size_t block = 128;
    size_t grid = 64;

    uint** data;
    CUDA_CHECKED_CALL(cudaMallocManaged(&data, grid * block * sizeof(uint*)));

    New_kernel<<<grid, block>>>(data);
    CUDA_CHECKED_CALL(cudaDeviceSynchronize());

    Delete_kernel<<<grid, block>>>(data);
    CUDA_CHECKED_CALL(cudaDeviceSynchronize());

    cudaFree(data);
}


__global__ void ArrayNew_kernel(uint** parray)
{
    parray[threadIdx.x + blockIdx.x*blockDim.x] = new uint[threadIdx.x % 4];
}

__global__ void ArrayDelete_kernel(uint** parray)
{
    delete[] parray[threadIdx.x + blockIdx.x*blockDim.x];
}

TEST_F(ManagedHeapTest, ArrayNewDeleteOnDevice)
{
    size_t block = 128;
    size_t grid = 64;

    uint** data;
    CUDA_CHECKED_CALL(cudaMallocManaged(&data, grid * block * sizeof(uint*)));

    ArrayNew_kernel<<<grid, block>>>(data);
    CUDA_CHECKED_CALL(cudaDeviceSynchronize());

    ArrayDelete_kernel<<<grid, block>>>(data);
    CUDA_CHECKED_CALL(cudaDeviceSynchronize());

    cudaFree(data);
}

int main(int argc, char** argv) {
    const char prefix[] = "--device=";
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], prefix, strlen(prefix)) == 0) {
            ::g_cuda_device = std::atoi(argv[i] + strlen(prefix));
            for (int j = i; j + 1 < argc; ++j) {
                argv[j] = argv[j + 1];
            }
            --argc;
            --i;
        }
    }

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
