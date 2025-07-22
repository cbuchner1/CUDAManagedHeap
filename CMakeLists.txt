cmake_minimum_required(VERSION 3.20)

# Use clang-20 for all languages
set(CMAKE_C_COMPILER clang-20)
set(CMAKE_CXX_COMPILER clang++-20)
set(CMAKE_CUDA_COMPILER clang++-20)

# CUDA specific settings
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unknown-cuda-version")

# use RelWithDebInfo build type
set(CMAKE_BUILD_TYPE RelWithDebInfo)

# set language standards to C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)

project(cudamanagedheap LANGUAGES CUDA CXX)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
    DOWNLOAD_EXTRACT_TIMESTAMP true
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

enable_testing()

add_executable(test_managedheap tests/test_managedheap.cu)
target_link_libraries(test_managedheap gtest)
gtest_discover_tests(test_managedheap)

target_include_directories(test_managedheap PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)


add_executable(cudamanagedheap examples/cudamanagedheap.cu)

target_include_directories(cudamanagedheap PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
