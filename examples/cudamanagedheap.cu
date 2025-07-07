/*
  ScatterAlloc: Massively Parallel Dynamic Memory Allocation for the GPU.
  http://www.icg.tugraz.at/project/mvp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
              Michael Kenzel - kenzel ( at ) icg.tugraz.at

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#include <cuda.h>
typedef unsigned int uint;

//set the template arguments using HEAPARGS
// pagesize ... bytes per page
// accessblocks ... number of superblocks
// regionsize ... number of regions for meta data structure
// wastefactor ... how much memory can be wasted per alloc (multiplicative factor)
// use_coalescing ... combine memory requests within each warp
// resetfreedpages ... allow pages to be reused with a different size
#define HEAPARGS 65536, 8, 16, 2, true, true

#define _DEBUG 1 // makes CUDA_CHECKED_CALL throw

//include the scatter alloc heap
#include "cudamanagedheap.cuh"

#include <cassert>
#include <cstdio>
#include <initializer_list>
#include <iostream>
#include <memory>

void runexample(int cuda_device);

int main(int argc, char** argv)
{
   try
  {
    int cuda_device = argc > 1 ? atoi(argv[1]) : 0;

    cudaDeviceProp deviceProp;
	  CUDA_CHECKED_CALL(cudaGetDeviceProperties(&deviceProp, cuda_device));
    std::cout << "Using device: " << deviceProp.name << std::endl;

	  if( deviceProp.major < 2 ) {
      std::cerr << "This GPU with Compute Capability " << deviceProp.major
        << "." << deviceProp.minor <<  " does not meet minimum requirements." << std::endl;
		  std::cerr << "A GPU with Compute Capability >= 2.0 is required." << std::endl;
      return -2;
	  }

    runexample(cuda_device);

    cudaDeviceReset();
  }
  catch (const std::exception& e)
  {
    std::cout << e.what()  << std::endl;
    return -1;
  }
  catch (...)
  {
    std::cout << "unknown exception!" << std::endl;
    return -1;
  }

  return 0;
}

/**
 * Example vector class using the managed heap. Uses custom allocators to force creation in this heap.
 * The _data member is also initialized and free'd using the managed heap's alloc and dealloc.
 * GPU's malloc and free (and hence new and delete) is assumed to be redirected to the managed heap also.
 */
template<typename T>
class TestVector
{
public:

  // new and delete on the host are handled via theHeap_ph
  __host__ __device__
  void* operator new(std::size_t count) {
#ifndef __CUDA_ARCH__
    return theHeap_ph->alloc(count);
#else
    return malloc(count);
#endif
  }
  __host__ __device__
  void operator delete( void* ptr)
  {
#ifndef __CUDA_ARCH__
    theHeap_ph->dealloc(ptr);
#else
    free(ptr);
#endif
  }

  // constructor taking an initializer list
  __host__ __device__
  TestVector(const std::initializer_list<T>& inputs) {
    _size = inputs.size();
#ifndef __CUDA_ARCH__
    _data = static_cast<T*>(theHeap_ph->alloc(_size * sizeof(T)));
#else
    _data = ::new T[_size];
#endif
    T *tmp = _data;
    for (auto i : inputs)
      *tmp ++ = i;
  }

  // copy constructor
  __host__ __device__
  TestVector(const TestVector<T>& rhs)
  {
    _size = rhs._size;
#ifndef __CUDA_ARCH__
    _data = static_cast<T*>(theHeap_ph->alloc(_size * sizeof(T)));
#else
    _data = new T[_size];
#endif
    memcpy(_data, rhs._data, _size * sizeof(T));
  }

  // assignment operator is deleted
  __host__ __device__ TestVector<T>& operator=(const TestVector<T>& rhs) = delete;

  // destructor
  __host__ __device__
  ~TestVector() {
#ifndef __CUDA_ARCH__
    theHeap_ph->dealloc(_data);
#else
    delete[] _data;
#endif
    _data = nullptr;
  }

  // size accessor
  __host__ __device__ size_t size() const {
    return _size;
  }

  // element accessor, via const reference
  __host__ __device__ const T& operator[](size_t i) const {
    return _data[i];
  }

protected:
  size_t _size;
  T* _data;
};


/**
 * Kernel demonstrating sharing an object between CPU and GPU (object passed by reference).
 * NOTE: passing input by value would also work (resulting in a GPU-local copy being made)
 */
__global__ void readVectorOnGPU(const TestVector<int>& input)
{
  // test vector is passed in by reference. This verifies we can access it safely.
  printf("Hello from readVectorOnGPU()!\n");
  printf("input.size() = %d\n", (int)input.size());
  for (int i=0; i < input.size(); ++i)
  {
    printf("input[%d] = %d\n", i, input[i]);
  }
}

void runexample(int cuda_device)
{
  CUDA_CHECKED_CALL(cudaSetDevice(cuda_device));

  //init the heap
  initHeap(8U*1024U*1024U);

  // we can't create this on the stack because this must live on the managed heap
  auto input_p = std::make_unique<TestVector<int>>(std::initializer_list<int>{1,2,3});
  const TestVector<int>& input = *input_p;

  // test basic functionality on CPU
  assert(input.size() == 3);
  assert(input[0] == 1);
  assert(input[1] == 2);
  assert(input[2] == 3);

  printf("Hello from runexample()!\n");
  printf("input.size() = %d\n", (int)input.size());
  for (int i=0; i < input.size(); ++i)
  {
    printf("input[%d] = %d\n", i, input[i]);
  }
  fflush(stdout);

  readVectorOnGPU<<<1,1>>>(input);
  CUDA_CHECKED_CALL(cudaDeviceSynchronize());

  printf("Success!\n");
}
